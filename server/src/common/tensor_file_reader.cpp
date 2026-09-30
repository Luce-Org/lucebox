#include "tensor_file_reader.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

namespace luce::common {

namespace {

constexpr size_t kPieceBytes = (size_t) 32 << 20;  // one read
constexpr size_t kMaxThreads = 8;                  // reads in flight
constexpr size_t kAlign = ReadOnlyFile::kDirectAlign;

struct Piece {
    const TensorFileSpan * span;
    size_t at;    // offset inside the span
    size_t size;
};

bool span_fits_tensor(const TensorFileSpan & s) {
    return s.tensor_offset <= ggml_nbytes(s.tensor) && s.size <= ggml_nbytes(s.tensor) - s.tensor_offset;
}

// A meta (tensor-parallel) buffer splits tensors along rows, so it takes a
// tensor only whole.
bool whole_writes_only(const ggml_tensor * t) {
    return t->buffer && ggml_backend_buft_is_meta(ggml_backend_buffer_get_type(t->buffer));
}

}  // namespace

bool TensorFileReader::open(const std::string & path, std::string * err, Mode mode) {
    buffered_.close();
    direct_.close();
    // Kill switch: loaders then copy out of their mapping as before.
    if (const char * off = std::getenv("LUCE_NO_PREAD"); off && *off && std::strcmp(off, "0") != 0) {
        if (err) *err = "threaded weight reads disabled (LUCE_NO_PREAD)";
        return false;
    }
    if (!buffered_.open(path, /*direct=*/false, err)) return false;
    // Direct reads for a file the page cache could hold only by evicting what
    // the host needs, including the weights themselves where the device
    // shares host memory; a smaller file stays cached for the next start.
    bool want_direct = mode == Mode::Direct;
    if (mode == Mode::Auto) {
        const uint64_t available = host_available_bytes();
        want_direct = available > 0 && buffered_.size() > available / 2;
    }
    if (want_direct) {
        std::string direct_err;
        if (!direct_.open(path, /*direct=*/true, &direct_err)) {
            if (mode == Mode::Direct) {
                buffered_.close();
                if (err) *err = "direct reads unavailable: " + direct_err;
                return false;
            }
            // Auto: the file still loads, through the page cache.
            std::fprintf(stderr, "[loader] direct reads unavailable for %s (%s); reading through the page cache\n",
                         path.c_str(), direct_err.c_str());
        }
    }
    return true;
}

bool TensorFileReader::load(const std::vector<TensorFileSpan> & spans, std::string * err) const {
    if (!buffered_.is_open()) {
        if (err) *err = "weight file is not open";
        return false;
    }
    // Pieces of plain buffers are read on several threads; whole-tensor
    // spans follow on this thread alone, so only one full copy is staged.
    std::vector<Piece> pieces, whole;
    for (const TensorFileSpan & s : spans) {
        if (!s.tensor || s.size == 0) continue;
        if (s.file_offset > size() || s.size > size() - s.file_offset || !span_fits_tensor(s)) {
            if (err) *err = std::string("weight span out of range for tensor ") + ggml_get_name(s.tensor);
            return false;
        }
        if (whole_writes_only(s.tensor)) {
            whole.push_back({&s, 0, s.size});
            continue;
        }
        for (size_t at = 0; at < s.size; at += kPieceBytes) {
            pieces.push_back({&s, at, std::min(kPieceBytes, s.size - at)});
        }
    }
    const size_t n_split = pieces.size();
    pieces.insert(pieces.end(), whole.begin(), whole.end());
    if (pieces.empty()) return true;

    std::atomic<size_t> next{0};
    size_t end = n_split;
    std::atomic<bool> failed{false};
    std::mutex write_mu;  // one tensor write at a time
    std::string first_error;
    const auto worker = [&] {
        std::vector<uint8_t> buf;
        size_t i;
        while (!failed.load(std::memory_order_relaxed) && (i = next.fetch_add(1)) < end) {
            const Piece & p = pieces[i];
            if (buf.size() < p.size + 3 * kAlign) buf.resize(p.size + 3 * kAlign);
            uint8_t * aligned = reinterpret_cast<uint8_t *>(
                ((uintptr_t) buf.data() + kAlign - 1) & ~(uintptr_t) (kAlign - 1));
            const uint64_t off = p.span->file_offset + p.at;
            const uint8_t * src = aligned;
            bool ok = false;
            if (direct_.is_open()) {
                // Whole aligned blocks around the piece; the last may pass
                // the end of the file.
                const uint64_t lo = off & ~(uint64_t) (kAlign - 1);
                const uint64_t hi = (off + p.size + kAlign - 1) & ~(uint64_t) (kAlign - 1);
                const int64_t got = direct_.read_upto(lo, aligned, (size_t) (hi - lo));
                ok = got >= 0 && (uint64_t) got >= off + p.size - lo;
                src = aligned + (off - lo);
            }
            if (!ok) {
                // Buffered, or a direct read the file system refused.
                if (direct_.is_open()) buffered_fallbacks_.fetch_add(1, std::memory_order_relaxed);
                ok = buffered_.read_at(off, aligned, p.size);
                src = aligned;
            }
            const int read_errno = errno;
            std::lock_guard<std::mutex> lk(write_mu);
            if (!ok) {
                if (!failed.exchange(true)) {
                    first_error = std::string("failed to read weights of tensor ") +
                                  ggml_get_name(p.span->tensor) + ": " + std::strerror(read_errno);
                }
                return;
            }
            ggml_backend_tensor_set(p.span->tensor, src, p.span->tensor_offset + p.at, p.size);
        }
    };
    const size_t n_threads = std::min(kMaxThreads, n_split);
    std::vector<std::thread> threads;
    for (size_t t = 1; t < n_threads; ++t) threads.emplace_back(worker);
    worker();
    for (std::thread & t : threads) t.join();
    next = n_split;
    end = pieces.size();
    worker();
    if (failed.load() && err) *err = first_error;
    return !failed.load();
}

bool load_tensor_spans(const std::string & path, const void * mapping, size_t mapping_size,
                       const std::vector<TensorFileSpan> & spans, std::string * err) {
    std::vector<std::pair<uint64_t, uint64_t>> ranges;  // file bytes the spans cover
    for (const TensorFileSpan & s : spans) {
        if (!s.tensor || s.size == 0) continue;
        if (s.file_offset > mapping_size || s.size > mapping_size - s.file_offset || !span_fits_tensor(s)) {
            if (err) *err = std::string("weight span out of range for tensor ") + ggml_get_name(s.tensor);
            return false;
        }
        ranges.emplace_back(s.file_offset, s.file_offset + s.size);
    }
    if (ranges.empty()) return true;
    // Residency of the bytes the spans read (overlaps and touching spans
    // merged, gaps between them left out): mostly cached means a warm start,
    // which the mapping serves fastest.
    std::sort(ranges.begin(), ranges.end());
    size_t wanted = 0, cached = 0;
    for (size_t i = 0; i < ranges.size();) {
        uint64_t lo = ranges[i].first, hi = ranges[i].second;
        for (++i; i < ranges.size() && ranges[i].first <= hi; ++i) hi = std::max(hi, ranges[i].second);
        wanted += (size_t) (hi - lo);
        cached += mapped_resident_bytes(mapping, lo, (size_t) (hi - lo));
    }
    TensorFileReader reader;
    if (cached < wanted - wanted / 10 && reader.open(path)) return reader.load(spans, err);
    for (const TensorFileSpan & s : spans) {
        if (!s.tensor || s.size == 0) continue;
        ggml_backend_tensor_set(s.tensor, static_cast<const uint8_t *>(mapping) + s.file_offset,
                                s.tensor_offset, s.size);
    }
    return true;
}

}  // namespace luce::common
