#include "tensor_file_reader.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <thread>

namespace luce::common {

namespace {

constexpr size_t kPieceBytes = (size_t) 32 << 20;  // one read
constexpr size_t kMaxThreads = 8;                  // reads in flight

struct Piece {
    const TensorFileSpan * span;
    size_t at;    // offset inside the span
    size_t size;
};

}  // namespace

bool TensorFileReader::open(const std::string & path, std::string * err) {
    file_.close();
    // Kill switch: loaders then copy out of their mapping as before.
    if (const char * off = std::getenv("LUCE_NO_PREAD"); off && *off && std::strcmp(off, "0") != 0) {
        if (err) *err = "threaded weight reads disabled (LUCE_NO_PREAD)";
        return false;
    }
    // Direct reads for a file the page cache could hold only by evicting what
    // the host needs, including the weights themselves where the device
    // shares host memory; a smaller file stays cached for the next start.
    uint64_t file_size = 0;
    {
        ReadOnlyFile probe;
        if (!probe.open(path, /*direct=*/false, err)) return false;
        file_size = probe.size();
    }
    const uint64_t available = host_available_bytes();
    if (available > 0 && file_size > available / 2 && file_.open(path, /*direct=*/true)) return true;
    return file_.open(path, /*direct=*/false, err);
}

bool TensorFileReader::load(const std::vector<TensorFileSpan> & spans, std::string * err) const {
    if (!file_.is_open()) {
        if (err) *err = "weight file is not open";
        return false;
    }
    std::vector<Piece> pieces;
    for (const TensorFileSpan & s : spans) {
        if (!s.tensor || s.size == 0) continue;
        if (s.file_offset > file_.size() || s.size > file_.size() - s.file_offset ||
            s.tensor_offset > ggml_nbytes(s.tensor) || s.size > ggml_nbytes(s.tensor) - s.tensor_offset) {
            if (err) *err = std::string("weight span out of range for tensor ") + ggml_get_name(s.tensor);
            return false;
        }
        for (size_t at = 0; at < s.size; at += kPieceBytes) {
            pieces.push_back({&s, at, std::min(kPieceBytes, s.size - at)});
        }
    }
    if (pieces.empty()) return true;

    constexpr size_t A = ReadOnlyFile::kDirectAlign;
    const bool direct = file_.direct();
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex write_mu;  // one tensor write at a time
    std::string first_error;
    const auto worker = [&] {
        std::vector<uint8_t> buf(kPieceBytes + 3 * A);
        uint8_t * aligned = reinterpret_cast<uint8_t *>(
            ((uintptr_t) buf.data() + A - 1) & ~(uintptr_t) (A - 1));
        size_t i;
        while (!failed.load(std::memory_order_relaxed) && (i = next.fetch_add(1)) < pieces.size()) {
            const Piece & p = pieces[i];
            const uint64_t off = p.span->file_offset + p.at;
            const uint8_t * src = aligned;
            bool ok;
            if (direct) {
                // Whole aligned blocks around the piece; the last may pass
                // the end of the file.
                const uint64_t lo = off & ~(uint64_t) (A - 1);
                const uint64_t hi = (off + p.size + A - 1) & ~(uint64_t) (A - 1);
                const int64_t got = file_.read_upto(lo, aligned, (size_t) (hi - lo));
                ok = got >= 0 && (uint64_t) got >= off + p.size - lo;
                src = aligned + (off - lo);
            } else {
                ok = file_.read_at(off, aligned, p.size);
            }
            std::lock_guard<std::mutex> lk(write_mu);
            if (!ok) {
                if (!failed.exchange(true)) {
                    first_error = std::string("failed to read weights of tensor ") +
                                  ggml_get_name(p.span->tensor);
                }
                return;
            }
            ggml_backend_tensor_set(p.span->tensor, src, p.span->tensor_offset + p.at, p.size);
        }
    };
    const size_t n_threads = std::min(kMaxThreads, pieces.size());
    std::vector<std::thread> threads;
    for (size_t t = 1; t < n_threads; ++t) threads.emplace_back(worker);
    worker();
    for (std::thread & t : threads) t.join();
    if (failed.load() && err) *err = first_error;
    return !failed.load();
}

bool load_tensor_spans(const std::string & path, const void * mapping,
                       const std::vector<TensorFileSpan> & spans, std::string * err) {
    size_t total = 0, cached = 0;
    for (const TensorFileSpan & s : spans) {
        total += s.size;
        cached += mapped_resident_bytes(mapping, s.file_offset, s.size);
    }
    TensorFileReader reader;
    if (cached < total - total / 10 && reader.open(path)) return reader.load(spans, err);
    for (const TensorFileSpan & s : spans) {
        if (!s.tensor || s.size == 0) continue;
        ggml_backend_tensor_set(s.tensor, static_cast<const uint8_t *>(mapping) + s.file_offset,
                                s.tensor_offset, s.size);
    }
    return true;
}

}  // namespace luce::common
