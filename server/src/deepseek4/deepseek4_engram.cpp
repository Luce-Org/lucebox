// DeepSeek V4.1 Engram: addressing and the table read. The ggml apply and the
// init from loaded weights live in deepseek4_engram_apply.cpp so this file
// builds without ggml (the standalone test compiles it directly).
#include "deepseek4_engram.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace luce::common {

// ── 1. addressing ─────────────────────────────────────────────────────────

bool DeepSeek4EngramHasher::init_raw(const std::vector<int> & layer_ids, int n_heads, int max_ngram,
                                     int32_t pad_id, std::vector<int32_t> token_map,
                                     std::vector<uint64_t> multipliers, std::vector<uint64_t> primes,
                                     std::vector<uint64_t> offsets, std::vector<uint64_t> rows,
                                     std::string * err) {
    n_layers_ = 0;
    const int n_layers = (int) layer_ids.size();
    if (n_layers == 0) return true;
    if (n_heads <= 0 || max_ngram < 2 || max_ngram > DeepSeek4EngramHistory::kMaxTail + 1) {
        if (err) *err = "engram geometry out of range";
        return false;
    }
    const int cols = (max_ngram - 1) * n_heads;
    if (multipliers.size() != (size_t) n_layers * max_ngram ||
        primes.size() != (size_t) n_layers * cols ||
        offsets.size() != (size_t) n_layers * cols ||
        rows.size() != (size_t) n_layers || token_map.empty() || pad_id < 0) {
        if (err) *err = "engram hash constants do not match the geometry";
        return false;
    }
    for (uint64_t m : multipliers) {
        if ((m & 1) == 0) { if (err) *err = "engram multiplier is even"; return false; }
    }
    for (size_t l = 0; l < (size_t) n_layers; ++l) {
        uint64_t sum = 0;
        for (int c = 0; c < cols; ++c) {
            const uint64_t p = primes[l * cols + c];
            if (p < 2 || offsets[l * cols + c] != sum) {
                if (err) *err = "engram primes/offsets are not a running sum";
                return false;
            }
            sum += p;
        }
        if (sum != rows[l]) { if (err) *err = "engram rows != sum of primes"; return false; }
    }
    n_layers_ = n_layers;
    n_heads_ = n_heads;
    max_ngram_ = max_ngram;
    cols_ = cols;
    pad_id_ = pad_id;
    layer_ids_ = layer_ids;
    rows_ = std::move(rows);
    token_map_ = std::move(token_map);
    multipliers_ = std::move(multipliers);
    primes_ = std::move(primes);
    offsets_ = std::move(offsets);
    return true;
}

int DeepSeek4EngramHasher::layer_index(int il) const {
    for (int i = 0; i < n_layers_; ++i) if (layer_ids_[(size_t) i] == il) return i;
    return -1;
}

void DeepSeek4EngramHasher::hash(DeepSeek4EngramHistory & history,
                                 const int32_t * tokens, const uint8_t * dead, size_t count,
                                 uint32_t * out) const {
    const int n = max_ngram_;
    uint32_t ids[DeepSeek4EngramHistory::kMaxTail + 1];
    for (size_t i = 0; i < count; ++i) {
        const int32_t current = (dead && dead[i]) ? DeepSeek4EngramHistory::kDead
                                                  : compress(tokens[i]);
        // A dead slot blocks every longer n-gram that would cross it.
        bool blocked = false;
        for (int j = 0; j < n; ++j) {
            const int32_t id = j == 0 ? current : history.tail[j - 1];
            blocked = blocked || id == DeepSeek4EngramHistory::kDead;
            ids[j] = blocked ? (uint32_t) pad_id_ : (uint32_t) id;
        }
        for (int l = 0; l < n_layers_; ++l) {
            const uint64_t * mult = &multipliers_[(size_t) l * n];
            const uint64_t * prime = &primes_[(size_t) l * cols_];
            const uint64_t * off = &offsets_[(size_t) l * cols_];
            uint64_t h = (uint64_t) ids[0] * mult[0];
            for (int j = 1; j < n; ++j) {
                h ^= (uint64_t) ids[j] * mult[j];
                for (int head = 0; head < n_heads_; ++head) {
                    const int col = (j - 1) * n_heads_ + head;
                    *out++ = (uint32_t) (h % prime[col] + off[col]);
                }
            }
        }
        for (int j = n - 2; j > 0; --j) history.tail[j] = history.tail[j - 1];
        history.tail[0] = current;
    }
}

// ── 2. the table read ─────────────────────────────────────────────────────

static inline float e4m3_to_float(uint8_t b) {
    const int exponent = (b >> 3) & 15, mantissa = b & 7;
    const float v = exponent ? std::ldexp((float) (8 + mantissa), exponent - 10)
                             : std::ldexp((float) mantissa, -9);
    return (b & 0x80) ? -v : v;
}

float deepseek4_engram_decode_value(uint8_t e4m3, uint8_t e8m0) {
    float v = std::ldexp(e4m3_to_float(e4m3), (int) e8m0 - 127);
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;   // RNE to bf16
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

bool DeepSeek4EngramTable::open(const std::string & path, uint64_t offset, uint64_t rows,
                                std::string * err) {
    close();
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) { if (err) *err = "engram table: open failed: " + path; return false; }
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) ||
        offset + rows * (uint64_t) kRowBytes > (uint64_t) st.st_size) {
        ::close(fd);
        if (err) *err = "engram table: offset/rows exceed the file";
        return false;
    }
    fd_ = fd; offset_ = offset; rows_ = rows;
    return true;
}

void DeepSeek4EngramTable::close() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1; offset_ = 0; rows_ = 0;
}

namespace {
struct RowRequest { uint32_t row, slot; };

bool read_one(int fd, uint64_t off, uint8_t * buf) {
    size_t done = 0;
    while (done < (size_t) DeepSeek4EngramTable::kRowBytes) {
        const ssize_t n = pread(fd, buf + done, DeepSeek4EngramTable::kRowBytes - done, (off_t) (off + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { if (n == 0) errno = EIO; return false; }
        done += (size_t) n;
    }
    return true;
}

bool decode_row(const uint8_t * raw, float * out) {
    for (int j = 0; j < DeepSeek4EngramTable::kDim; ++j) {
        const uint8_t code = raw[j], scale = raw[DeepSeek4EngramTable::kDim + j / 32];
        if ((code & 127) == 127 || scale == 255) { errno = EDOM; return false; }   // NaN codes
        out[j] = deepseek4_engram_decode_value(code, scale);
    }
    return true;
}
}  // namespace

bool DeepSeek4EngramTable::read(const uint32_t * row_ids, size_t count, float * out, int threads) const {
    if (fd_ < 0) { errno = EBADF; return false; }
    for (size_t i = 0; i < count; ++i) {
        if (row_ids[i] >= rows_) { errno = EINVAL; return false; }
    }
    if (count == 0) return true;
    std::vector<RowRequest> req(count);
    for (size_t i = 0; i < count; ++i) req[i] = { row_ids[i], (uint32_t) i };
    std::sort(req.begin(), req.end(), [](const RowRequest & a, const RowRequest & b) { return a.row < b.row; });

    const int n_workers = count >= 256 ? std::max(1, threads) : 1;
    std::vector<int> errs((size_t) n_workers, 0);
    auto part = [&](int p) {
        const size_t begin = count * (size_t) p / (size_t) n_workers;
        const size_t end = count * (size_t) (p + 1) / (size_t) n_workers;
        uint8_t raw[kRowBytes];
        const float * previous = nullptr;
        for (size_t i = begin; i < end; ++i) {
            float * dst = out + (size_t) req[i].slot * kDim;
            if (i > begin && req[i].row == req[i - 1].row) {
                std::memcpy(dst, previous, kDim * sizeof(float));
            } else {
                if (!read_one(fd_, offset_ + (uint64_t) req[i].row * kRowBytes, raw) ||
                    !decode_row(raw, dst)) {
                    errs[(size_t) p] = errno ? errno : EIO;
                    return;
                }
                previous = dst;
            }
        }
    };
    if (n_workers == 1) {
        part(0);
    } else {
        std::vector<std::thread> pool;
        for (int p = 1; p < n_workers; ++p) pool.emplace_back(part, p);
        part(0);
        for (auto & t : pool) t.join();
    }
    for (int e : errs) if (e) { errno = e; return false; }
    return true;
}


void DeepSeek4EngramTable::prefetch(const uint32_t * row_ids, size_t count) const {
    if (fd_ < 0) return;
    for (size_t i = 0; i < count; ++i) {
        if (row_ids[i] >= rows_) continue;
        (void) posix_fadvise(fd_, (off_t) (offset_ + (uint64_t) row_ids[i] * kRowBytes), kRowBytes,
                             POSIX_FADV_WILLNEED);
    }
}

// ── host-side runtime ─────────────────────────────────────────────────────

// Row ids of `count` tokens at first_pos.., [count][n_layers][cols], hashed in
// the context `ctx` holds before first_pos.
bool DeepSeek4EngramRuntime::row_ids(const DeepSeek4EngramTokens & ctx, const int32_t * tokens,
                                     int first_pos, size_t count, std::vector<uint32_t> & ids,
                                     std::string * err) const {
    if (first_pos < 0) { if (err) *err = "engram: negative position"; return false; }
    // The tokens before first_pos, most recent first; a position before the
    // start of the sequence carries no token.
    DeepSeek4EngramHistory history;
    for (int j = 0; j < hasher_.max_ngram() - 1; ++j) {
        const int p = first_pos - 1 - j;
        if (p < 0) break;
        const int32_t t = ctx.at(p);
        if (t < 0) {
            if (err) *err = "engram: the n-gram context before position " + std::to_string(first_pos) +
                            " is not known";
            return false;
        }
        history.tail[j] = hasher_.compress(t);
    }
    ids.resize(count * (size_t) hasher_.n_layers() * (size_t) hasher_.cols());
    hasher_.hash(history, tokens, nullptr, count, ids.data());
    return true;
}

void DeepSeek4EngramRuntime::prefetch(const DeepSeek4EngramTokens & ctx, const int32_t * tokens,
                                      int first_pos, size_t count) const {
    std::vector<uint32_t> ids;
    if (!row_ids(ctx, tokens, first_pos, count, ids, nullptr)) return;
    const size_t cols = (size_t) hasher_.cols();
    for (size_t t = 0; t < count; ++t) {
        for (int l = 0; l < hasher_.n_layers(); ++l) {
            tables_[(size_t) l].prefetch(&ids[(t * (size_t) hasher_.n_layers() + (size_t) l) * cols], cols);
        }
    }
}

bool DeepSeek4EngramRuntime::prepare(DeepSeek4EngramTokens & ctx, const int32_t * tokens, int first_pos,
                                     size_t count, float * keys, std::string * err) const {
    const int n_layers = hasher_.n_layers();
    const size_t cols = (size_t) hasher_.cols();
    std::vector<uint32_t> ids;
    if (!row_ids(ctx, tokens, first_pos, count, ids, err)) return false;
    for (size_t t = 0; t < count; ++t) ctx.put(first_pos + (int) t, tokens[t]);

    // Split the rows per layer and start every read before waiting on any.
    std::vector<std::vector<uint32_t>> layer_ids((size_t) n_layers, std::vector<uint32_t>(count * cols));
    for (int l = 0; l < n_layers; ++l) {
        for (size_t t = 0; t < count; ++t) {
            std::memcpy(&layer_ids[(size_t) l][t * cols], &ids[(t * (size_t) n_layers + (size_t) l) * cols],
                        cols * sizeof(uint32_t));
        }
        tables_[(size_t) l].prefetch(layer_ids[(size_t) l].data(), count * cols);
    }
    for (int l = 0; l < n_layers; ++l) {
        float * dst = keys + (size_t) l * count * key_floats();
        if (!tables_[(size_t) l].read(layer_ids[(size_t) l].data(), count * cols, dst, threads_)) {
            if (err) *err = std::string("engram: table read failed for layer ") +
                            std::to_string(hasher_.layer_id(l)) + ": " + std::strerror(errno);
            return false;
        }
        rows_read_.fetch_add(count * cols, std::memory_order_relaxed);
    }
    return true;
}

}  // namespace luce::common
