// Disk-backed prefix cache implementation.

#include "disk_prefix_cache.h"
#include "common/sha1.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <system_error>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace luce::common {

namespace fs = std::filesystem;

// ─── Utility ────────────────────────────────────────────────────────────

static std::string hex(const uint8_t * data, int len) {
    static const char hex_chars[] = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (int i = 0; i < len; ++i) {
        out.push_back(hex_chars[data[i] >> 4]);
        out.push_back(hex_chars[data[i] & 0x0f]);
    }
    return out;
}

static bool mkdir_p(const std::string & path) {
    std::error_code ec;
    if (fs::is_directory(path, ec)) return true;
    fs::create_directories(path, ec);
    return fs::is_directory(path, ec);
}

static uint64_t now_unix() {
    return (uint64_t)std::time(nullptr);
}

const char * disk_prefix_cache_mode_name(DiskPrefixCacheMode mode) {
    switch (mode) {
        case DiskPrefixCacheMode::Off:   return "off";
        case DiskPrefixCacheMode::Full:  return "full";
        case DiskPrefixCacheMode::Auto:  return "auto";
        case DiskPrefixCacheMode::Fixed: return "fixed";
    }
    return "full";
}

std::string disk_prefix_cache_policy_name(const DiskPrefixCachePolicy & policy) {
    std::string base;
    if (policy.mode == DiskPrefixCacheMode::Fixed) {
        base = "fixed:" + std::to_string(policy.fixed_tokens);
    } else if (policy.mode == DiskPrefixCacheMode::Auto) {
        base = "auto:" + std::to_string(policy.auto_window);
    } else {
        base = disk_prefix_cache_mode_name(policy.mode);
    }
    if (policy.compress) base += "+compress";
    return base;
}

bool parse_disk_prefix_cache_policy(const std::string & value,
                                    DiskPrefixCachePolicy & out) {
    std::string v;
    v.reserve(value.size());
    for (char c : value) v.push_back((char)std::tolower((unsigned char)c));

    if (v == "off" || v == "none" || v == "disabled") {
        out = {};
        out.mode = DiskPrefixCacheMode::Off;
        return true;
    }
    if (v == "full" || v == "full-prefix") {
        out = {};
        out.mode = DiskPrefixCacheMode::Full;
        return true;
    }
    if (v == "auto") {
        out = {};
        out.mode = DiskPrefixCacheMode::Auto;
        return true;
    }

    const std::string auto_prefix = "auto:";
    if (v.rfind(auto_prefix, 0) == 0) {
        char * end = nullptr;
        long n = std::strtol(v.c_str() + auto_prefix.size(), &end, 10);
        if (!end || *end != '\0' || n <= 0 || n > 1000000) return false;
        out = {};
        out.mode = DiskPrefixCacheMode::Auto;
        out.auto_window = (int)n;
        return true;
    }

    char * end = nullptr;
    long n = std::strtol(v.c_str(), &end, 10);
    if (end && *end == '\0' && n > 0 && n <= 1000000) {
        out = {};
        out.mode = DiskPrefixCacheMode::Fixed;
        out.fixed_tokens = (int)n;
        return true;
    }
    return false;
}

bool apply_request_scope_override(DiskPrefixCachePolicy & server_policy,
                                  const std::string & scope_str) {
    DiskPrefixCachePolicy parsed;
    if (!parse_disk_prefix_cache_policy(scope_str, parsed)) {
        return false;
    }
    // Preserve server-level flags (e.g. compress) across the scope override.
    parsed.compress = server_policy.compress;
    server_policy = parsed;
    return true;
}

static bool valid_boundary(int n, int full_len) {
    return n > 0 && n <= full_len;
}

std::vector<int> disk_prefix_cache_full_lookup_lengths(
        int prompt_len,
        const std::vector<int> & boundaries,
        int min_tokens) {
    std::vector<int> out;
    // Nothing shorter than the persistence minimum is ever written, so such
    // lengths are not worth a probe.
    if (prompt_len >= min_tokens) out.push_back(prompt_len);
    for (auto it = boundaries.rbegin(); it != boundaries.rend(); ++it) {
        if (*it >= min_tokens && *it < prompt_len) out.push_back(*it);
    }
    return out;
}

int disk_prefix_cache_fixed_boundary(const DiskPrefixCachePolicy & policy,
                                     int full_len,
                                     int min_tokens) {
    if (policy.mode != DiskPrefixCacheMode::Fixed) return 0;
    if (policy.fixed_tokens < min_tokens) return 0;
    return valid_boundary(policy.fixed_tokens, full_len) ? policy.fixed_tokens : 0;
}

static int lcp_len(const std::vector<int32_t> & a,
                   const std::vector<int32_t> & b) {
    const int n = std::min((int)a.size(), (int)b.size());
    int i = 0;
    while (i < n && a[(size_t)i] == b[(size_t)i]) i++;
    return i;
}

static int floor_to_safe_boundary(int n, const std::vector<int> & safe_boundaries) {
    if (n <= 0) return 0;
    if (safe_boundaries.empty()) return n;

    int best = 0;
    for (int b : safe_boundaries) {
        if (b > 0 && b <= n) best = std::max(best, b);
    }
    return best;
}

int disk_prefix_cache_auto_boundary(
    const std::vector<int32_t> & prompt_ids,
    const std::vector<std::vector<int32_t>> & recent_prompts,
    int window,
    const std::vector<int> & safe_boundaries,
    int min_tokens) {
    if (prompt_ids.empty() || recent_prompts.empty() || window <= 0) return 0;

    const int n_recent = std::min(window, (int)recent_prompts.size());
    int common = 0;
    for (int i = 0; i < n_recent; ++i) {
        common = std::max(common, lcp_len(prompt_ids, recent_prompts[(size_t)i]));
    }
    common = floor_to_safe_boundary(common, safe_boundaries);
    return common >= min_tokens ? common : 0;
}

// Little-endian I/O helpers.
static void write_u32(FILE * f, uint32_t v) { std::fwrite(&v, 4, 1, f); }
static void write_u64(FILE * f, uint64_t v) { std::fwrite(&v, 8, 1, f); }
static bool read_u32(FILE * f, uint32_t & out) { return std::fread(&out, 4, 1, f) == 1; }
static bool read_u64(FILE * f, uint64_t & out) { return std::fread(&out, 8, 1, f) == 1; }
static bool read_i64(FILE * f, int64_t & out)  { return std::fread(&out, 8, 1, f) == 1; }
static bool read_u16(FILE * f, uint16_t & out) { return std::fread(&out, 2, 1, f) == 1; }

// 64-bit safe absolute seek.
static bool seek_to(FILE * f, uint64_t offset) {
#if defined(_WIN32)
    return _fseeki64(f, (__int64)offset, SEEK_SET) == 0;
#else
    return fseeko(f, (off_t)offset, SEEK_SET) == 0;
#endif
}

// ─── Checksum ───────────────────────────────────────────────────────────
// Streaming XXH64 (seed 0): fast enough for multi-GB payloads, unlike SHA-1,
// and detects bit flips and torn writes in any part of the file.
namespace {
class Checksum64 {
public:
    Checksum64() {
        v_[0] = P1 + P2;
        v_[1] = P2;
        v_[2] = 0;
        v_[3] = 0 - P1;
    }
    void update(const void * data, size_t n) {
        const uint8_t * p = (const uint8_t *)data;
        total_ += n;
        if (buf_len_ > 0) {
            const size_t take = std::min(n, sizeof(buf_) - buf_len_);
            std::memcpy(buf_ + buf_len_, p, take);
            buf_len_ += take;
            p += take;
            n -= take;
            if (buf_len_ < sizeof(buf_)) return;
            stripe(buf_);
            buf_len_ = 0;
        }
        while (n >= 32) {
            stripe(p);
            p += 32;
            n -= 32;
        }
        if (n > 0) {
            std::memcpy(buf_, p, n);
            buf_len_ = n;
        }
    }
    uint64_t digest() const {
        uint64_t h;
        if (total_ >= 32) {
            h = rotl(v_[0], 1) + rotl(v_[1], 7) + rotl(v_[2], 12) + rotl(v_[3], 18);
            for (int i = 0; i < 4; ++i) h = merge(h, v_[i]);
        } else {
            h = P5;
        }
        h += total_;
        const uint8_t * p = buf_;
        size_t n = buf_len_;
        while (n >= 8) {
            h ^= round(0, read64(p));
            h = rotl(h, 27) * P1 + P4;
            p += 8;
            n -= 8;
        }
        if (n >= 4) {
            uint32_t w;
            std::memcpy(&w, p, 4);
            h ^= (uint64_t)w * P1;
            h = rotl(h, 23) * P2 + P3;
            p += 4;
            n -= 4;
        }
        while (n > 0) {
            h ^= (*p) * P5;
            h = rotl(h, 11) * P1;
            ++p;
            --n;
        }
        h ^= h >> 33;
        h *= P2;
        h ^= h >> 29;
        h *= P3;
        h ^= h >> 32;
        return h;
    }

private:
    static constexpr uint64_t P1 = 11400714785074694791ULL;
    static constexpr uint64_t P2 = 14029467366897019727ULL;
    static constexpr uint64_t P3 = 1609587929392839161ULL;
    static constexpr uint64_t P4 = 9650029242287828579ULL;
    static constexpr uint64_t P5 = 2870177450012600261ULL;
    static uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
    static uint64_t read64(const uint8_t * p) {
        uint64_t v;
        std::memcpy(&v, p, 8);
        return v;
    }
    static uint64_t round(uint64_t acc, uint64_t input) {
        acc += input * P2;
        acc = rotl(acc, 31);
        return acc * P1;
    }
    static uint64_t merge(uint64_t acc, uint64_t v) {
        acc ^= round(0, v);
        return acc * P1 + P4;
    }
    void stripe(const uint8_t * p) {
        v_[0] = round(v_[0], read64(p));
        v_[1] = round(v_[1], read64(p + 8));
        v_[2] = round(v_[2], read64(p + 16));
        v_[3] = round(v_[3], read64(p + 24));
    }
    uint64_t v_[4];
    uint8_t buf_[32];
    size_t buf_len_ = 0;
    uint64_t total_ = 0;
};

// Sequential file I/O that feeds every byte through the checksum.
struct HashedWriter {
    FILE * f;
    Checksum64 sum;
    bool ok = true;
    void write(const void * p, size_t n) {
        if (!ok || n == 0) return;
        ok = std::fwrite(p, 1, n, f) == n;
        sum.update(p, n);
    }
};

struct HashedReader {
    FILE * f;
    Checksum64 sum;
    bool read(void * p, size_t n) {
        if (n == 0) return true;
        if (std::fread(p, 1, n, f) != n) return false;
        sum.update(p, n);
        return true;
    }
};
}  // namespace

bool disk_cache_digest_enabled() {
    static const bool on = [] {
        const char * v = std::getenv("LUCE_DISK_CACHE_DIGEST");
        return v && *v && std::strcmp(v, "0") != 0;
    }();
    return on;
}

uint64_t snapshot_state_digest(const ModelBackend::SnapshotRef & ref,
                               int * n_tensors, uint64_t * n_bytes) {
    Checksum64 sum;
    int tensors = 0;
    uint64_t bytes = 0;
    const int64_t pos = ref.cur_pos;
    const int64_t tok = ref.last_tok;
    sum.update(&pos, sizeof(pos));
    sum.update(&tok, sizeof(tok));
    std::vector<uint8_t> tmp;
    if (ref.ctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(ref.ctx); t;
             t = ggml_get_next_tensor(ref.ctx, t)) {
            const size_t nb = ggml_nbytes(t);
            const int32_t type = (int32_t)t->type;
            sum.update(t->name, std::strlen(t->name));
            sum.update(&type, sizeof(type));
            sum.update(t->ne, sizeof(t->ne));
            if (t->buffer && nb > 0) {
                if (ggml_backend_buffer_is_host(t->buffer)) {
                    sum.update(t->data, nb);
                } else {
                    const size_t chunk = 64u << 20;
                    tmp.resize(std::min(nb, chunk));
                    for (size_t off = 0; off < nb; off += chunk) {
                        const size_t n = std::min(chunk, nb - off);
                        ggml_backend_tensor_get(t, tmp.data(), off, n);
                        sum.update(tmp.data(), n);
                    }
                }
            }
            ++tensors;
            bytes += nb;
        }
    }
    if (n_tensors) *n_tensors = tensors;
    if (n_bytes) *n_bytes = bytes;
    return sum.digest();
}

// ─── Header (de)serialization ───────────────────────────────────────────

static void serialize_header(const DiskCacheHeader & hdr,
                             uint8_t raw[DISK_CACHE_HEADER_SIZE]) {
    std::memset(raw, 0, DISK_CACHE_HEADER_SIZE);
    std::memcpy(raw + 0,  hdr.magic, 4);
    std::memcpy(raw + 4,  &hdr.version, 4);
    std::memcpy(raw + 8,  hdr.layout_id, 16);
    std::memcpy(raw + 24, &hdr.cur_pos, 4);
    std::memcpy(raw + 28, &hdr.n_tensors, 4);
    std::memcpy(raw + 32, &hdr.token_count, 4);
    std::memcpy(raw + 36, hdr.token_hash, 16);
    std::memcpy(raw + 52, &hdr.payload_bytes, 8);
    std::memcpy(raw + 60, &hdr.created_at, 8);
    std::memcpy(raw + DISK_CACHE_LAST_USED_OFFSET, &hdr.last_used, 8);
    const uint32_t last_tok = (uint32_t)hdr.last_tok;
    std::memcpy(raw + 76, &last_tok, 4);
    std::memcpy(raw + 80, hdr.identity_salt, 16);
}

static void parse_header(const uint8_t raw[DISK_CACHE_HEADER_SIZE],
                         DiskCacheHeader & hdr) {
    std::memcpy(hdr.magic,          raw + 0, 4);
    std::memcpy(&hdr.version,       raw + 4, 4);
    std::memcpy(hdr.layout_id,      raw + 8, 16);
    std::memcpy(&hdr.cur_pos,       raw + 24, 4);
    std::memcpy(&hdr.n_tensors,     raw + 28, 4);
    std::memcpy(&hdr.token_count,   raw + 32, 4);
    std::memcpy(hdr.token_hash,     raw + 36, 16);
    std::memcpy(&hdr.payload_bytes, raw + 52, 8);
    std::memcpy(&hdr.created_at,    raw + 60, 8);
    std::memcpy(&hdr.last_used,     raw + DISK_CACHE_LAST_USED_OFFSET, 8);
    uint32_t last_tok = 0;
    std::memcpy(&last_tok,          raw + 76, 4);
    hdr.last_tok = (int32_t)last_tok;
    std::memcpy(hdr.identity_salt,  raw + 80, 16);
}

// The checksum covers the header with last_used zeroed: a hit rewrites
// that field in place.
static void hash_header(Checksum64 & sum, const uint8_t raw[DISK_CACHE_HEADER_SIZE]) {
    uint8_t copy[DISK_CACHE_HEADER_SIZE];
    std::memcpy(copy, raw, DISK_CACHE_HEADER_SIZE);
    std::memset(copy + DISK_CACHE_LAST_USED_OFFSET, 0, 8);
    sum.update(copy, DISK_CACHE_HEADER_SIZE);
}

// Skip the tensor table and key tokens of a file positioned after its
// header, then read the auxiliary blob.
static bool read_aux_section(FILE * f, uint32_t n_tensors, uint32_t token_count,
                             std::string & aux) {
    for (uint32_t i = 0; i < n_tensors; ++i) {
        uint16_t name_len = 0;
        if (!read_u16(f, name_len) ||
            std::fseek(f, (long)name_len + 4 + 32 + 8, SEEK_CUR) != 0) {
            return false;
        }
    }
    // Key tokens: up to 4 * 2^32 bytes, so step in bounded seeks.
    uint64_t skip = 4ull * token_count;
    while (skip > 0) {
        const long step = (long)std::min<uint64_t>(skip, 1u << 30);
        if (std::fseek(f, step, SEEK_CUR) != 0) return false;
        skip -= (uint64_t)step;
    }
    uint32_t len = 0;
    if (!read_u32(f, len) || len > (64u << 20)) return false;
    aux.resize(len);
    return len == 0 || std::fread(aux.data(), 1, len, f) == len;
}

// Flush a written file to stable storage.
static bool sync_file(FILE * f) {
#if defined(_WIN32)
    return _commit(_fileno(f)) == 0;
#else
    return ::fdatasync(fileno(f)) == 0;
#endif
}

// Make a rename inside `dir` durable (POSIX; a no-op elsewhere).
static void sync_directory(const std::string & dir) {
#if !defined(_WIN32)
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd >= 0) {
        (void)::fsync(fd);
        ::close(fd);
    }
#else
    (void)dir;
#endif
}

static double mib(uint64_t bytes) { return (double)bytes / (1024.0 * 1024.0); }

// Check file metadata without constructing a GGML tensor: removed types have
// zero block sizes, and ggml_new_tensor/ggml_row_size assume valid dimensions.
static bool valid_tensor_metadata(uint32_t type, const int64_t ne[4], uint64_t nbytes) {
    if (type >= GGML_TYPE_COUNT) return false;
    const auto tensor_type = (ggml_type)type;
    const int64_t block = ggml_blck_size(tensor_type);
    const size_t width = ggml_type_size(tensor_type);
    if (block <= 0 || width == 0 || ne[0] < 0 || ne[0] % block != 0) return false;
    // ggml_row_size multiplies by width before dividing by the block size.
    if ((uint64_t)ne[0] > SIZE_MAX / width) return false;
    uint64_t bytes = width * (uint64_t)ne[0] / (uint64_t)block;
    uint64_t elements = (uint64_t)ne[0];
    for (int d = 1; d < 4; ++d) {
        if (ne[d] < 0) return false;
        const uint64_t dim = (uint64_t)ne[d];
        if (dim && (bytes > SIZE_MAX / dim || elements > INT64_MAX / dim)) return false;
        bytes *= dim;
        elements *= dim;
    }
    return bytes == nbytes && bytes <= SIZE_MAX - GGML_MEM_ALIGN;
}

// ─── Construction ───────────────────────────────────────────────────────

DiskPrefixCache::DiskPrefixCache(const DiskCacheConfig & cfg, ModelBackend & backend)
    : config_(cfg), backend_(backend) {}

DiskPrefixCache::~DiskPrefixCache() {
    if (writer_.joinable()) {
        drain();
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_writer_ = true;
        }
        writer_cv_.notify_all();
        writer_.join();
    }
#if defined(_WIN32)
    if (lock_handle_) CloseHandle(lock_handle_);
#else
    if (lock_fd_ >= 0) ::close(lock_fd_);
#endif
}

uint64_t DiskPrefixCache::now() const {
    return clock_ ? clock_() : now_unix();
}

static bool has_suffix(const std::string & s, const char * suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// Validate a snapshot file without reading tensor data: header, version,
// tensor table, payload sum and exact file size. A file cut short by a crash
// or a full disk, or written by an older format, fails here. Bit corruption
// inside the file is caught by the checksum when it is read (read_file).
static bool validate_snapshot_file(const std::string & path,
                                   DiskCacheHeader & hdr,
                                   uint64_t & file_size,
                                   const char *& why) {
    std::error_code ec;
    file_size = (uint64_t) fs::file_size(path, ec);
    if (ec) { why = "unreadable size"; return false; }
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { why = "unreadable"; return false; }
    bool ok = false;
    do {
        uint8_t raw[DISK_CACHE_HEADER_SIZE];
        if (std::fread(raw, 1, sizeof(raw), f) != sizeof(raw)) {
            why = "short header"; break;
        }
        DiskCacheHeader h{};
        parse_header(raw, h);
        if (std::memcmp(h.magic, "DKVC", 4) != 0) { why = "bad magic"; break; }
        if (h.version != DISK_CACHE_VERSION) { why = "old or unknown version"; break; }
        if (h.cur_pos == 0 || h.cur_pos > h.token_count) {
            why = "snapshot past key"; break;
        }
        if (h.n_tensors == 0 || h.n_tensors > 4096) { why = "tensor count"; break; }
        uint64_t table_bytes = 0;
        uint64_t payload = 0;
        bool table_ok = true;
        for (uint32_t i = 0; i < h.n_tensors && table_ok; ++i) {
            uint16_t name_len = 0;
            uint32_t type = 0;
            int64_t ne[4] = {};
            uint64_t nbytes = 0;
            table_ok = read_u16(f, name_len) && name_len < GGML_MAX_NAME &&
                std::fseek(f, name_len, SEEK_CUR) == 0 && read_u32(f, type) &&
                type < GGML_TYPE_COUNT;
            for (int d = 0; d < 4 && table_ok; ++d) table_ok = read_i64(f, ne[d]);
            table_ok = table_ok && read_u64(f, nbytes) &&
                valid_tensor_metadata(type, ne, nbytes) &&
                payload <= file_size && nbytes <= file_size - payload;
            payload += nbytes;
            table_bytes += 2 + name_len + 4 + 32 + 8;
        }
        if (!table_ok) { why = "bad tensor table"; break; }
        if (payload != h.payload_bytes) { why = "payload mismatch"; break; }
        uint32_t aux_len = 0;
        const uint64_t aux_at = DISK_CACHE_HEADER_SIZE + table_bytes +
            4ull * h.token_count;
        if (aux_at + 4 > file_size ||
            !seek_to(f, aux_at) || !read_u32(f, aux_len)) {
            why = "truncated metadata"; break;
        }
        const uint64_t expected =
            aux_at + 4 + aux_len + payload + DISK_CACHE_CHECKSUM_SIZE;
        if (file_size != expected) { why = "truncated or oversized"; break; }
        hdr = h;
        ok = true;
    } while (false);
    std::fclose(f);
    return ok;
}

// ─── Initialization ─────────────────────────────────────────────────────

bool DiskPrefixCache::init() {
    if (disabled()) return true;
    if (!mkdir_p(config_.cache_dir)) {
        std::fprintf(stderr, "[disk-cache] disabled: failed to create dir: %s\n",
                     config_.cache_dir.c_str());
        // No writer was started. Leaving the cache enabled would queue
        // payloads forever and block serving once backpressure is reached.
        config_.cache_dir.clear();
        return false;
    }
    const std::string lock_path = config_.cache_dir + "/.lock";
#if defined(_WIN32)
    // A share-mode-zero handle excludes every other server before scanning
    // temporaries or enforcing the budget, just like flock on POSIX.
    lock_handle_ = CreateFileW(fs::path(lock_path).c_str(), GENERIC_READ | GENERIC_WRITE,
                              0, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    const bool locked = lock_handle_ != INVALID_HANDLE_VALUE;
    if (!locked) lock_handle_ = nullptr;
#else
    lock_fd_ = ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    const bool locked = lock_fd_ >= 0 && ::flock(lock_fd_, LOCK_EX | LOCK_NB) == 0;
#endif
    if (!locked) {
        std::fprintf(stderr,
                     "[disk-cache] disabled: cannot lock %s (in use or not writable) "
                     "(pass --kv-cache-dir <other dir> to give this one its own)\n",
                     config_.cache_dir.c_str());
#if !defined(_WIN32)
        if (lock_fd_ >= 0) ::close(lock_fd_);
        lock_fd_ = -1;
#endif
        config_.cache_dir.clear();
        return true;
    }
    std::lock_guard<std::mutex> lock(mu_);
    scan_all();
    // Budget and TTL hold from the first moment, before any request.
    expire_locked();
    make_room_locked(0);
    // Restore into the layout this identity used most recently; learn_layout
    // confirms it against the live model before anything is written.
    const DiskEntry * best = nullptr;
    for (const auto & e : entries_) {
        if (e.own_salt && (!best || e.last_used > best->last_used)) best = &e;
    }
    if (best) {
        layout_id_ = best->layout_id;
        layout_known_ = true;
        layout_from_disk_ = true;
        layout_dir_ = config_.cache_dir + "/" + hex(layout_id_.data(), 16);
    }
    std::fprintf(stderr,
                 "[disk-cache] initialized dir=%s budget=%.1f GB (configured %.1f GB) "
                 "ttl=%s files=%zu used=%.1f MB layout=%s\n",
                 config_.cache_dir.c_str(),
                 (double)effective_budget_ / (1024.0 * 1024.0 * 1024.0),
                 (double)config_.budget_bytes / (1024.0 * 1024.0 * 1024.0),
                 config_.ttl_seconds
                     ? (std::to_string(config_.ttl_seconds) + "s").c_str()
                     : "never",
                 entries_.size(), mib(total_bytes_),
                 layout_known_ ? hex(layout_id_.data(), 16).c_str() : "pending");
    if (!writer_.joinable()) writer_ = std::thread([this] { writer_loop(); });
    return true;
}

// ─── Layout fingerprint ─────────────────────────────────────────────────

std::array<uint8_t, 16> DiskPrefixCache::layout_of(ggml_context * ctx) const {
    // Collect tensor metadata sorted by name for deterministic fingerprint.
    struct TInfo { std::string name; uint32_t type; int64_t ne[4]; };
    std::vector<TInfo> tensors;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        TInfo ti;
        ti.name = t->name;
        ti.type = (uint32_t)t->type;
        ti.ne[0] = t->ne[0];
        ti.ne[1] = 1;  // normalize sequence-length dimension
        ti.ne[2] = t->ne[2];
        ti.ne[3] = t->ne[3];
        tensors.push_back(std::move(ti));
    }
    std::sort(tensors.begin(), tensors.end(), [](const TInfo & a, const TInfo & b) {
        return a.name < b.name;
    });
    // Build a single buffer and hash it.
    // Prepend identity_salt_ so that config/model differences (model file,
    // KV types, max_ctx, chat_template) rotate the layout_id independently of
    // tensor structure. All-zero salt (the default) is back-compatible.
    std::vector<uint8_t> buf;
    buf.insert(buf.end(), identity_salt_.begin(), identity_salt_.end());
    for (const auto & ti : tensors) {
        buf.insert(buf.end(), ti.name.begin(), ti.name.end());
        buf.insert(buf.end(), (uint8_t *)&ti.type, (uint8_t *)&ti.type + 4);
        buf.insert(buf.end(), (uint8_t *)ti.ne, (uint8_t *)ti.ne + 32);
    }
    uint8_t digest[20];
    sha1_hash(buf.data(), buf.size(), digest);
    std::array<uint8_t, 16> out{};
    std::memcpy(out.data(), digest, 16);
    return out;
}

void DiskPrefixCache::learn_layout_from(ggml_context * ctx) {
    const auto id = layout_of(ctx);
    std::lock_guard<std::mutex> lock(mu_);
    if (layout_known_ && !layout_from_disk_) return;  // verified from the live model
    if (layout_from_disk_ && id != layout_id_) {
        // The files stay indexed (budget/TTL) but are never restored.
        std::fprintf(stderr, "[disk-cache] layout mismatch: disk=%s model=%s — switching\n",
                     hex(layout_id_.data(), 16).c_str(), hex(id.data(), 16).c_str());
    }
    layout_id_ = id;
    layout_known_ = true;
    layout_from_disk_ = false;
    layout_dir_ = config_.cache_dir + "/" + hex(layout_id_.data(), 16);
    mkdir_p(layout_dir_);
    std::fprintf(stderr, "[disk-cache] layout learned: %s\n",
                 hex(layout_id_.data(), 16).c_str());
}

void DiskPrefixCache::learn_layout(int slot) {
    if (disabled()) return;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (layout_known_ && !layout_from_disk_) return;
    }
    auto ref = backend_.snapshot_ref(slot);
    if (!ref.ctx) {
        if (backend_.snapshot_used(slot)) {
            backend_snapshot_unsupported_ = true;
            std::fprintf(stderr,
                         "[disk-cache] disabled: backend snapshots are "
                         "memory-only\n");
        }
        return;
    }
    if (ref.pooled) return;  // no dense KV tensors: would fingerprint a wrong layout
    learn_layout_from(ref.ctx);
}

// ─── Directory scanning ─────────────────────────────────────────────────

void DiskPrefixCache::scan_all() {
    entries_.clear();
    total_bytes_ = 0;
    size_t removed = 0;
    std::error_code ec;
    for (const auto & de : fs::directory_iterator(config_.cache_dir, ec)) {
        const std::string base = de.path().filename().string();
        std::error_code dec;
        if (base.empty() || base[0] == '.' || !de.is_directory(dec)) continue;
        bool kept_any = false;
        std::error_code sec;
        for (const auto & se : fs::directory_iterator(de.path(), sec)) {
            const std::string name = se.path().filename().string();
            const std::string path = se.path().string();
            std::error_code rec;
            if (has_suffix(name, ".dkv.tmp")) {
                // A write the previous process never renamed into place.
                fs::remove(path, rec);
                ++removed;
                continue;
            }
            if (!has_suffix(name, ".dkv")) continue;
            DiskCacheHeader hdr{};
            uint64_t size = 0;
            const char * why = "";
            bool ok = validate_snapshot_file(path, hdr, size, why);
            if (ok && (hex(hdr.layout_id, 16) != base ||
                       hex(hdr.token_hash, 16) + ".dkv" != name)) {
                ok = false;
                why = "name does not match header";
            }
            if (!ok) {
                std::fprintf(stderr, "[disk-cache] removing %s: %s\n",
                             path.c_str(), why);
                fs::remove(path, rec);
                ++removed;
                continue;
            }
            DiskEntry entry;
            entry.id          = next_entry_id_++;
            entry.path        = path;
            std::memcpy(entry.token_hash.data(), hdr.token_hash, 16);
            std::memcpy(entry.layout_id.data(), hdr.layout_id, 16);
            entry.token_count = hdr.token_count;
            entry.cur_pos     = hdr.cur_pos;
            entry.last_used   = hdr.last_used;
            entry.file_size   = size;
            entry.own_salt    = std::memcmp(hdr.identity_salt,
                                            identity_salt_.data(), 16) == 0;
            total_bytes_ += entry.file_size;
            entries_.push_back(std::move(entry));
            kept_any = true;
        }
        std::error_code rec;
        if (!kept_any) fs::remove(de.path(), rec);  // only removes if empty
    }
    std::fprintf(stderr, "[disk-cache] scanned %zu files, %.1f MB (%zu removed)\n",
                 entries_.size(), mib(total_bytes_), removed);
}

// ─── Index maintenance ──────────────────────────────────────────────────

bool DiskPrefixCache::expired(const DiskEntry & e, uint64_t t) const {
    return config_.ttl_seconds > 0 && t > e.last_used &&
           t - e.last_used > config_.ttl_seconds;
}

void DiskPrefixCache::release_pending_locked(DiskEntry & e) {
    if (!e.pending) return;
    pending_bytes_ -= std::min<uint64_t>(pending_bytes_, e.file_size);
    e.pending.reset();
}

void DiskPrefixCache::touch_locked(DiskEntry & e) {
    e.last_used = now();
    if (e.pending) return;  // the writer publishes the latest timestamp
    // Not fsynced: a crash may age an entry, but clean restarts retain uses.
    if (FILE * f = std::fopen(e.path.c_str(), "r+b")) {
        if (std::fseek(f, DISK_CACHE_LAST_USED_OFFSET, SEEK_SET) == 0) write_u64(f, e.last_used);
        std::fclose(f);
    }
}

void DiskPrefixCache::remove_entry(size_t idx, const char * why) {
    DiskEntry & e = entries_[idx];
    const bool queued = e.pending != nullptr;
    std::fprintf(stderr, "[disk-cache] %s: removing %s%s (%u tokens, %.1f MB)\n",
                 why, e.path.c_str(), queued ? " (not yet written)" : "",
                 e.token_count, mib(e.file_size));
    if (queued) {
        // Not on disk yet: drop it from the queue; a write in progress
        // finds its entry gone and deletes its temporary file.
        queue_.erase(std::remove(queue_.begin(), queue_.end(), e.id), queue_.end());
        release_pending_locked(e);
        ++stats_.write_cancelled;
    } else {
        std::error_code ec;
        fs::remove(e.path, ec);
    }
    total_bytes_ -= std::min<size_t>(total_bytes_, e.file_size);
    entries_.erase(entries_.begin() + (long)idx);
    done_cv_.notify_all();
}

void DiskPrefixCache::expire_locked() {
    const uint64_t t = now();
    for (size_t i = entries_.size(); i-- > 0;) {
        if (expired(entries_[i], t)) remove_entry(i, "ttl expired");
    }
}

uint64_t DiskPrefixCache::budget_locked() {
    uint64_t budget = config_.budget_bytes;
    // Never plan past the filesystem: the cache's bytes already on disk
    // plus the free space, minus a reserve for everything else.
    uint64_t available = 0;
    uint64_t reserve = config_.free_space_reserve_bytes;
    bool known = false;
    if (free_space_) {
        available = free_space_();
        known = true;
    } else {
        std::error_code ec;
        const auto space = fs::space(config_.cache_dir, ec);
        if (!ec) {
            available = (uint64_t)space.available;
            reserve = std::max<uint64_t>(
                reserve, std::min<uint64_t>((uint64_t)space.capacity / 20, 16ull << 30));
            known = true;
        }
    }
    if (known) {
        const uint64_t on_disk = total_bytes_ - std::min<uint64_t>(total_bytes_, pending_bytes_);
        const uint64_t room = available > reserve ? available - reserve : 0;
        const uint64_t limit = on_disk + room;
        if (limit < budget) budget = limit;
    }
    const bool clamped = budget < config_.budget_bytes;
    const bool was_clamped = effective_budget_ < config_.budget_bytes;
    const uint64_t moved = budget > effective_budget_
        ? budget - effective_budget_ : effective_budget_ - budget;
    if (effective_budget_ == 0 || clamped != was_clamped ||
        (clamped && moved >= (1ull << 30))) {
        std::fprintf(stderr,
                     "[disk-cache] budget %.1f GB (configured %.1f GB, free space "
                     "%.1f GB)\n",
                     (double)budget / 1073741824.0,
                     (double)config_.budget_bytes / 1073741824.0,
                     (double)available / 1073741824.0);
    }
    effective_budget_ = budget;
    return budget;
}

uint64_t DiskPrefixCache::budget_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return effective_budget_;
}

size_t DiskPrefixCache::total_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return total_bytes_;
}

size_t DiskPrefixCache::entry_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return entries_.size();
}

DiskPrefixCache::Stats DiskPrefixCache::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    Stats out = stats_;
    out.pending_bytes = pending_bytes_;
    out.pending_files = 0;
    for (const auto & e : entries_) out.pending_files += e.pending ? 1 : 0;
    return out;
}

void DiskPrefixCache::make_room_locked(uint64_t incoming) {
    const uint64_t budget = budget_locked();
    while (!entries_.empty() && (uint64_t)total_bytes_ + incoming > budget) {
        size_t lru = 0;
        for (size_t i = 1; i < entries_.size(); ++i) {
            if (entries_[i].last_used < entries_[lru].last_used) lru = i;
        }
        remove_entry(lru, "budget (lru)");
    }
}

// ─── Lookup ─────────────────────────────────────────────────────────────

OwnedSnapshot DiskPrefixCache::load(const std::vector<int32_t> & prompt_ids) {
    if (disabled()) return {};
    const PrefixHash hash = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());
    std::shared_ptr<PendingWrite> pending;
    std::string path;
    uint64_t id = 0;
    uint32_t cur_pos = 0;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!layout_known_) return {};
        const int idx = find_entry(hash);
        if (idx < 0) return {};
        if (expired(entries_[(size_t)idx], now())) {
            remove_entry((size_t)idx, "ttl expired");
            return {};
        }
        const DiskEntry & e = entries_[(size_t)idx];
        id = e.id;
        pending = e.pending;
        path = e.path;
        cur_pos = e.cur_pos;
    }
    OwnedSnapshot out;
    const auto t_read = std::chrono::steady_clock::now();
    if (pending) {
        // Still queued for the writer: copy it from memory. The shared
        // pointer keeps it alive even if the write finishes meanwhile.
        if (pending->key != prompt_ids) return {};
        out = OwnedSnapshot::clone_to_host(pending->snap.ctx(),
                                           pending->snap.cur_pos(),
                                           pending->snap.last_tok());
        if (out.empty()) return {};
    } else {
        out = read_file(path, prompt_ids);
    }
    std::lock_guard<std::mutex> lock(mu_);
    const int idx = find_entry_id(id);
    if (out.empty()) {
        ++stats_.load_failures;
        // A corrupt, partial or mismatched file is never retried.
        if (idx >= 0 && !entries_[(size_t)idx].pending) {
            remove_entry((size_t)idx, "lookup failed");
        }
        return {};
    }
    std::fprintf(stderr, "[disk-cache] read %s (%u pos, %.1f MB, %.2fs)\n",
                 pending ? "pending write" : fs::path(path).filename().string().c_str(),
                 cur_pos, mib(out.bytes()),
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t_read).count());
    ++stats_.loads;
    if (idx >= 0) {
        // Persist last_used in the file so LRU order and the TTL survive a
        // restart. Not fsynced: losing it in a crash only ages the entry.
        touch_locked(entries_[(size_t)idx]);
    }
    return out;
}

bool DiskPrefixCache::adopt(OwnedSnapshot snap, int slot) {
    if (snap.empty()) return false;
    {
        // The first hit after startup confirms the layout read from disk.
        const auto id = layout_of(snap.ctx());
        std::lock_guard<std::mutex> lock(mu_);
        if (layout_from_disk_) {
            if (id != layout_id_) {
                std::fprintf(stderr,
                             "[disk-cache] adopted layout mismatch: disk=%s model=%s\n",
                             hex(layout_id_.data(), 16).c_str(),
                             hex(id.data(), 16).c_str());
                layout_id_ = id;
                layout_from_disk_ = false;
                layout_dir_ = config_.cache_dir + "/" + hex(layout_id_.data(), 16);
                mkdir_p(layout_dir_);
                return false;
            }
        }
    }
    if (!backend_.snapshot_adopt(slot, snap.ctx(), snap.buf(), snap.cur_pos(),
                                 snap.last_tok())) {
        return false;  // `snap` still owns and frees the payload
    }
    snap.release();
    std::lock_guard<std::mutex> lock(mu_);
    layout_from_disk_ = false;
    return true;
}

bool DiskPrefixCache::lookup(const std::vector<int32_t> & prompt_ids, int slot) {
    return adopt(load(prompt_ids), slot);
}

void DiskPrefixCache::for_each_aux(
        const std::function<void(const std::string &)> & visit) {
    if (disabled()) return;
    std::vector<std::pair<uint64_t, std::string>> files;
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto & e : entries_) {
            if (e.own_salt && !e.pending) files.push_back({e.last_used, e.path});
        }
    }
    std::sort(files.begin(), files.end());
    for (const auto & file : files) {
        FILE * f = std::fopen(file.second.c_str(), "rb");
        if (!f) continue;
        DiskCacheHeader hdr{};
        std::string aux;
        const bool ok = read_header(f, hdr) &&
            read_aux_section(f, hdr.n_tensors, hdr.token_count, aux);
        std::fclose(f);
        if (ok && !aux.empty()) visit(aux);
    }
}

DiskPrefixCache::PrefixProbe DiskPrefixCache::longest_prefix(
        const std::vector<int32_t> & prompt_ids,
        const std::vector<int> * allowed_lengths) {
    PrefixProbe out;
    if (disabled()) return out;
    std::lock_guard<std::mutex> lock(mu_);
    if (!layout_known_) return out;
    const uint64_t t = now();
    std::vector<uint32_t> lengths;
    for (const auto & e : entries_) {
        if (e.layout_id != layout_id_ || expired(e, t)) continue;
        if (e.token_count > prompt_ids.size()) continue;
        if ((int)e.token_count < config_.min_tokens) continue;
        if (allowed_lengths &&
            std::find(allowed_lengths->begin(), allowed_lengths->end(),
                      (int)e.token_count) == allowed_lengths->end()) continue;
        lengths.push_back(e.token_count);
    }
    std::sort(lengths.begin(), lengths.end(), std::greater<uint32_t>());
    lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
    for (uint32_t len : lengths) {
        const int idx = find_entry(hash_prefix(prompt_ids.data(), (int)len));
        if (idx >= 0) {
            out.len = (int)len;
            out.cur_pos = (int)entries_[(size_t)idx].cur_pos;
            return out;
        }
    }
    return out;
}

int DiskPrefixCache::prune_ancestors(const std::vector<int32_t> & tokens,
                                     int keep_max) {
    if (disabled()) return 0;
    std::lock_guard<std::mutex> lock(mu_);
    if (!layout_known_) return 0;
    // The session's older states on disk: strict prefixes of `tokens`.
    std::vector<size_t> ancestors;
    bool head_kept = false;  // one at or below keep_max: the shared head
    size_t shallowest = SIZE_MAX;
    for (size_t i = 0; i < entries_.size(); ++i) {
        const DiskEntry & e = entries_[i];
        if (e.layout_id != layout_id_ || e.token_count >= tokens.size()) continue;
        if (hash_prefix(tokens.data(), (int)e.token_count) != e.token_hash) continue;
        if ((int)e.token_count <= keep_max) {
            head_kept = true;
            continue;
        }
        ancestors.push_back(i);
        if (shallowest == SIZE_MAX ||
            e.token_count < entries_[shallowest].token_count) shallowest = i;
    }
    // Without a short head on disk, the shallowest ancestor stands in for
    // it (a long system/tools head other sessions restore), as the RAM
    // cache keeps its shallowest entry.
    int removed = 0;
    std::sort(ancestors.begin(), ancestors.end(), std::greater<size_t>());
    for (const size_t i : ancestors) {
        if (!head_kept && i == shallowest) continue;
        remove_entry(i, "superseded by deeper session state");
        ++removed;
    }
    return removed;
}

// ─── Save ───────────────────────────────────────────────────────────────

bool DiskPrefixCache::save_async(OwnedSnapshot snap,
                                 const std::vector<int32_t> & prompt_ids,
                                 const std::string & aux) {
    if (disabled() || snap.empty()) return false;
    if ((int)prompt_ids.size() < config_.min_tokens) return false;
    // A file is keyed by the hash of `prompt_ids` only, and a hit restores
    // the whole snapshot. A snapshot that extends past its key would be
    // restored for any prompt sharing just the key prefix, with unverified
    // tokens in between, so such files are never written.
    if (snap.cur_pos() > (int)prompt_ids.size() || snap.cur_pos() <= 0) {
        std::fprintf(stderr,
                     "[disk-cache] skip save: snapshot pos=%d does not fit a key of "
                     "%zu tokens\n", snap.cur_pos(), prompt_ids.size());
        return false;
    }
    learn_layout_from(snap.ctx());

    uint64_t payload = 0;
    size_t table_bytes = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(snap.ctx()); t;
         t = ggml_get_next_tensor(snap.ctx(), t)) {
        payload += ggml_nbytes(t);
        table_bytes += 2 + std::strlen(t->name) + 4 + 32 + 8;
    }
    const uint64_t est_size = DISK_CACHE_HEADER_SIZE + table_bytes +
        4ull * prompt_ids.size() + 4 + aux.size() + payload +
        DISK_CACHE_CHECKSUM_SIZE;
    const PrefixHash hash = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());

    std::unique_lock<std::mutex> lock(mu_);
    // Snapshots waiting for the writer hold host memory: bound it.
    if (pending_bytes_ > 0 && pending_bytes_ + est_size > config_.max_pending_bytes) {
        ++stats_.backpressure_waits;
        const auto t0 = std::chrono::steady_clock::now();
        done_cv_.wait(lock, [&] {
            return pending_bytes_ == 0 ||
                pending_bytes_ + est_size <= config_.max_pending_bytes;
        });
        std::fprintf(stderr,
                     "[disk-cache] waited %.2fs for the writer (queued writes over "
                     "%.0f MB)\n",
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0).count(),
                     mib(config_.max_pending_bytes));
    }
    expire_locked();
    // Already on disk (or queued): count it as a use.
    const int existing = find_entry(hash);
    if (existing >= 0) {
        touch_locked(entries_[(size_t)existing]);
        return true;
    }
    // Make room first, so the budget holds while the file is being written.
    if (est_size > budget_locked()) {
        std::fprintf(stderr, "[disk-cache] skip save: %.1f MB exceeds budget\n",
                     mib(est_size));
        return false;
    }
    make_room_locked(est_size);

    auto job = std::make_shared<PendingWrite>();
    job->snap = std::move(snap);
    job->key = prompt_ids;
    job->aux = aux;
    job->path = make_path(hash);
    job->created_at = now();
    job->layout_id = layout_id_;

    DiskEntry entry;
    entry.id          = next_entry_id_++;
    entry.path        = job->path;
    entry.token_hash  = hash;
    entry.layout_id   = layout_id_;
    entry.token_count = (uint32_t)prompt_ids.size();
    entry.cur_pos     = (uint32_t)job->snap.cur_pos();
    entry.last_used   = job->created_at;
    entry.own_salt    = true;
    entry.file_size   = est_size;
    entry.pending     = job;
    total_bytes_ += est_size;
    pending_bytes_ += est_size;
    queue_.push_back(entry.id);
    entries_.push_back(std::move(entry));
    lock.unlock();
    writer_cv_.notify_one();
    return true;
}

bool DiskPrefixCache::save(int slot, const std::vector<int32_t> & prompt_ids,
                           const std::string & aux) {
    if (disabled()) return false;
    learn_layout(slot);
    if (disabled()) return false;
    if ((int)prompt_ids.size() < config_.min_tokens) return false;
    const auto ref = backend_.snapshot_ref(slot);
    if (!ref.ctx) return false;
    // Pooled snapshots keep KV in a pager blob that is not a ggml tensor in
    // ref.ctx; a file written from ref.ctx would restore with no KV.
    if (ref.pooled) {
        std::fprintf(stderr, "[disk-cache] skip save: pooled (KVFlash) snapshot not persistable\n");
        return false;
    }
    if (ref.cur_pos > (int)prompt_ids.size()) {
        std::fprintf(stderr,
                     "[disk-cache] skip save: snapshot pos=%d exceeds key of "
                     "%zu tokens\n", ref.cur_pos, prompt_ids.size());
        return false;
    }
    {
        // Already on disk: no copy needed.
        const PrefixHash hash = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());
        std::lock_guard<std::mutex> lock(mu_);
        const int existing = find_entry(hash);
        if (existing >= 0) {
            touch_locked(entries_[(size_t)existing]);
            return true;
        }
    }
    return save_async(OwnedSnapshot::clone_to_host(ref.ctx, ref.cur_pos, ref.last_tok),
                      prompt_ids, aux);
}

// ─── Writer ─────────────────────────────────────────────────────────────

void DiskPrefixCache::writer_loop() {
    std::unique_lock<std::mutex> lock(mu_);
    while (true) {
        writer_cv_.wait(lock, [&] { return stop_writer_ || !queue_.empty(); });
        if (queue_.empty()) break;  // stopping
        const uint64_t id = queue_.front();
        queue_.pop_front();
        const int idx = find_entry_id(id);
        if (idx < 0 || !entries_[(size_t)idx].pending) continue;
        std::shared_ptr<PendingWrite> job = entries_[(size_t)idx].pending;
        job->last_used = entries_[(size_t)idx].last_used;
        writing_id_ = id;
        abort_write_.store(false);
        lock.unlock();

        const std::string tmp_path = job->path + ".tmp";
        const std::string dir = fs::path(job->path).parent_path().string();
        mkdir_p(dir);
        const auto t0 = std::chrono::steady_clock::now();
        bool ok = write_file(tmp_path, *job);
        const int err = ok ? 0 : errno;

        lock.lock();
        const int at = find_entry_id(id);
        if (ok && at >= 0 && !abort_write_.load()) {
            ok = std::rename(tmp_path.c_str(), job->path.c_str()) == 0;
            if (ok) sync_directory(dir);
        }
        if (!ok || at < 0 || abort_write_.load()) std::remove(tmp_path.c_str());
        if (at >= 0) {
            DiskEntry & e = entries_[(size_t)at];
            if (ok && !abort_write_.load()) {
                // A hit while the file was being written: record it too.
                if (e.last_used != job->last_used) {
                    if (FILE * f = std::fopen(e.path.c_str(), "r+b")) {
                        if (std::fseek(f, DISK_CACHE_LAST_USED_OFFSET, SEEK_SET) == 0) {
                            write_u64(f, e.last_used);
                        }
                        std::fclose(f);
                    }
                }
                release_pending_locked(e);
                ++stats_.writes;
                stats_.write_bytes += e.file_size;
                std::fprintf(stderr,
                             "[disk-cache] wrote %s (%u tokens, %u pos, %.1f MB, %.2fs)\n",
                             fs::path(e.path).filename().string().c_str(),
                             e.token_count, e.cur_pos, mib(e.file_size),
                             std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - t0).count());
            } else {
                ++stats_.write_failures;
                std::fprintf(stderr, "[disk-cache] write failed (%s): %s\n",
                             abort_write_.load() ? "time limit" : std::strerror(err),
                             e.path.c_str());
                remove_entry((size_t)at, "write failed");
            }
        }
        job.reset();  // frees the payload unless a lookup still copies it
        writing_id_ = 0;
        done_cv_.notify_all();
    }
}

DiskPrefixCache::DrainResult DiskPrefixCache::drain(double max_seconds) {
    DrainResult out;
    if (!writer_.joinable()) return out;
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mu_);
    const uint64_t files0 = stats_.writes;
    const uint64_t bytes0 = stats_.write_bytes;
    const auto idle = [&] { return queue_.empty() && writing_id_ == 0; };
    if (max_seconds < 0) {
        done_cv_.wait(lock, idle);
    } else if (!done_cv_.wait_until(
                   lock, t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                       std::chrono::duration<double>(max_seconds)),
                   idle)) {
        out.cancelled = queue_.size() + (writing_id_ != 0 ? 1 : 0);
        while (!queue_.empty()) {
            const int idx = find_entry_id(queue_.front());
            if (idx >= 0) {
                remove_entry((size_t)idx, "flush time limit");
            } else {
                queue_.pop_front();
            }
        }
        abort_write_.store(true);
        done_cv_.wait(lock, idle);
    }
    out.files = stats_.writes - files0;
    out.bytes = stats_.write_bytes - bytes0;
    out.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    return out;
}

// ─── Continued checkpoints ──────────────────────────────────────────────

bool DiskPrefixCache::maybe_store_continued(int slot,
                                            const std::vector<int32_t> & all_tokens,
                                            int cur_pos) {
    if (disabled()) return false;
    if (cur_pos < config_.min_tokens) return false;
    const int interval = config_.continued_interval;
    if (interval <= 0) return false;

    // The interval only paces the checkpoints: fire once each time cur_pos
    // crosses a new multiple of it. The file itself is keyed by the full
    // token prefix the snapshot covers (cur_pos tokens), so a hit can only
    // come from a prompt that actually contains everything in the snapshot.
    int target = (cur_pos / interval) * interval;
    if (target <= continued_last_store_pos_) return false;
    if (target < config_.min_tokens) return false;
    if (cur_pos > (int)all_tokens.size()) return false;

    std::vector<int32_t> prefix(all_tokens.begin(), all_tokens.begin() + cur_pos);
    bool ok = save(slot, prefix);
    if (ok) {
        continued_last_store_pos_ = target;
        std::fprintf(stderr, "[disk-cache] continued checkpoint at %d tokens\n", cur_pos);
    }
    return ok;
}

// ─── Cold prefix boundary ───────────────────────────────────────────────

int DiskPrefixCache::cold_prefix_boundary(const std::vector<int32_t> & prompt_ids,
                                          const std::vector<int> & boundaries) {
    if (disabled()) return 0;
    const int prompt_len = (int)prompt_ids.size();
    if (prompt_len <= config_.cold_max_tokens) return 0;
    if (boundaries.empty()) return 0;

    // Find the last turn boundary <= cold_max_tokens.
    int best = 0;
    for (int b : boundaries) {
        if (b <= config_.cold_max_tokens && b >= config_.min_tokens) {
            best = b;
        }
    }
    if (best == 0) return 0;

    // Check if we already have a disk entry covering this prefix.
    PrefixHash hash = hash_prefix(prompt_ids.data(), best);
    std::lock_guard<std::mutex> lock(mu_);
    if (!layout_known_) return 0;
    if (find_entry(hash) >= 0) return 0;  // already cached
    return best;
}

// ─── Helpers ────────────────────────────────────────────────────────────

std::string DiskPrefixCache::make_path(const PrefixHash & hash) const {
    return layout_dir_ + "/" + hex(hash.data(), 16) + ".dkv";
}

int DiskPrefixCache::find_entry(const PrefixHash & hash) const {
    for (int i = 0; i < (int)entries_.size(); ++i) {
        if (entries_[i].token_hash == hash &&
            entries_[i].layout_id == layout_id_) return i;
    }
    return -1;
}

int DiskPrefixCache::find_entry_id(uint64_t id) const {
    for (int i = 0; i < (int)entries_.size(); ++i) {
        if (entries_[(size_t)i].id == id) return i;
    }
    return -1;
}

// ─── File I/O: Write ────────────────────────────────────────────────────

bool DiskPrefixCache::write_file(const std::string & path, const PendingWrite & job) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    ggml_context * ctx = job.snap.ctx();

    uint32_t n_tensors = 0;
    uint64_t payload_bytes = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        n_tensors++;
        payload_bytes += ggml_nbytes(t);
    }

    DiskCacheHeader hdr{};
    std::memcpy(hdr.magic, "DKVC", 4);
    hdr.version = DISK_CACHE_VERSION;
    std::memcpy(hdr.layout_id, job.layout_id.data(), 16);
    hdr.cur_pos       = (uint32_t)job.snap.cur_pos();
    hdr.n_tensors     = n_tensors;
    hdr.token_count   = (uint32_t)job.key.size();
    const PrefixHash ph = hash_prefix(job.key.data(), (int)job.key.size());
    std::memcpy(hdr.token_hash, ph.data(), 16);
    hdr.payload_bytes = payload_bytes;
    hdr.created_at    = job.created_at;
    hdr.last_used     = job.last_used;
    hdr.last_tok      = job.snap.last_tok();
    std::memcpy(hdr.identity_salt, identity_salt_.data(), 16);

    uint8_t raw[DISK_CACHE_HEADER_SIZE];
    serialize_header(hdr, raw);
    HashedWriter w{f, {}};
    w.ok = std::fwrite(raw, 1, sizeof(raw), f) == sizeof(raw);
    hash_header(w.sum, raw);

    for (ggml_tensor * t = ggml_get_first_tensor(ctx); w.ok && t; t = ggml_get_next_tensor(ctx, t)) {
        const uint16_t name_len = (uint16_t)std::strlen(t->name);
        const uint32_t type = (uint32_t)t->type;
        const uint64_t nbytes = ggml_nbytes(t);
        w.write(&name_len, 2);
        w.write(t->name, name_len);
        w.write(&type, 4);
        for (int d = 0; d < 4; ++d) w.write(&t->ne[d], 8);
        w.write(&nbytes, 8);
    }
    // The exact key: a hit compares it with the prompt before restoring.
    w.write(job.key.data(), job.key.size() * sizeof(int32_t));
    const uint32_t aux_len = (uint32_t)job.aux.size();
    w.write(&aux_len, 4);
    w.write(job.aux.data(), job.aux.size());
    // Tensor data: the snapshot's bytes as stored, no conversion. The payload
    // is host memory (OwnedSnapshot), written in bounded chunks so a drain
    // time limit can stop it.
    constexpr size_t kChunk = 8u << 20;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); w.ok && t; t = ggml_get_next_tensor(ctx, t)) {
        const size_t nbytes = ggml_nbytes(t);
        for (size_t offset = 0; w.ok && offset < nbytes; offset += kChunk) {
            if (abort_write_.load(std::memory_order_relaxed)) w.ok = false;
            w.write((const char *)t->data + offset, std::min(kChunk, nbytes - offset));
        }
    }
    const uint64_t digest = w.sum.digest();
    bool ok = w.ok && std::fwrite(&digest, 1, 8, f) == 8;
    // Durable before the rename publishes it.
    ok = ok && std::fflush(f) == 0 && sync_file(f);
    ok = (std::fclose(f) == 0) && ok;
    return ok;
}

// ─── File I/O: Read ─────────────────────────────────────────────────────

OwnedSnapshot DiskPrefixCache::read_file(const std::string & path,
                                         const std::vector<int32_t> & prompt_ids) {
    std::array<uint8_t, 16> layout_id{};
    {
        std::lock_guard<std::mutex> lock(mu_);
        layout_id = layout_id_;
    }
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    const auto fail = [&](const char * why) {
        if (why) std::fprintf(stderr, "[disk-cache] rejecting %s: %s\n", path.c_str(), why);
        std::fclose(f);
        return OwnedSnapshot{};
    };
    HashedReader r{f, {}};
    std::error_code ec;
    const uint64_t file_size = fs::file_size(path, ec);
    if (ec) return fail("unreadable size");
    uint8_t raw[DISK_CACHE_HEADER_SIZE];
    if (std::fread(raw, 1, sizeof(raw), f) != sizeof(raw)) return fail("short header");
    hash_header(r.sum, raw);
    DiskCacheHeader hdr{};
    parse_header(raw, hdr);
    if (std::memcmp(hdr.magic, "DKVC", 4) != 0 || hdr.version != DISK_CACHE_VERSION) {
        return fail("bad magic or version");
    }
    if (std::memcmp(hdr.layout_id, layout_id.data(), 16) != 0) return fail("layout");
    if (std::memcmp(hdr.identity_salt, identity_salt_.data(), 16) != 0) {
        return fail("written by another model/KV type/config");
    }
    // Never restore a snapshot that extends past the tokens its key covers
    // (see save_async()).
    if (hdr.cur_pos == 0 || hdr.cur_pos > hdr.token_count || hdr.cur_pos > INT_MAX) {
        return fail("snapshot past its key");
    }
    if (hdr.token_count != prompt_ids.size()) return fail("key length");
    if (hdr.n_tensors == 0 || hdr.n_tensors > 4096) return fail("tensor count");
    if (hdr.payload_bytes > file_size) return fail("payload size");

    std::vector<DiskTensorEntry> table;
    table.reserve(hdr.n_tensors);
    uint64_t payload = 0;
    for (uint32_t i = 0; i < hdr.n_tensors; ++i) {
        DiskTensorEntry ent;
        uint16_t name_len = 0;
        char name_buf[GGML_MAX_NAME] = {};
        uint64_t nbytes = 0;
        if (!r.read(&name_len, 2) || name_len >= GGML_MAX_NAME ||
            !r.read(name_buf, name_len) || !r.read(&ent.type, 4) ||
            ent.type >= GGML_TYPE_COUNT) {
            return fail("bad tensor table");
        }
        for (int d = 0; d < 4; ++d) {
            if (!r.read(&ent.ne[d], 8) || ent.ne[d] < 0) return fail("bad tensor table");
        }
        if (!r.read(&nbytes, 8) || !valid_tensor_metadata(ent.type, ent.ne, nbytes) ||
            payload > hdr.payload_bytes || nbytes > hdr.payload_bytes - payload) {
            return fail("bad tensor metadata");
        }
        ent.name = name_buf;
        ent.nbytes = (size_t)nbytes;
        payload += nbytes;
        table.push_back(std::move(ent));
    }
    if (payload != hdr.payload_bytes) return fail("payload size");
    // The stored key must be exactly this prompt (not just its hash): a
    // state keyed on another token stream, e.g. a compressed view, can
    // never be restored for it.
    std::vector<int32_t> stored(prompt_ids.size());
    if (!r.read(stored.data(), stored.size() * sizeof(int32_t)) || stored != prompt_ids) {
        return fail("stored tokens differ from the prompt");
    }
    uint32_t aux_len = 0;
    if (!r.read(&aux_len, 4) || aux_len > (64u << 20)) return fail("aux");
    std::string aux(aux_len, '\0');
    if (!r.read(aux.data(), aux_len)) return fail("aux");

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (size_t)(hdr.n_tensors + 4) + 4096;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return fail("ggml_init");
    for (const auto & ent : table) {
        ggml_tensor * t = ggml_new_tensor(ctx, (ggml_type)ent.type, 4, ent.ne);
        if (!t || ggml_nbytes(t) != ent.nbytes) {
            ggml_free(ctx);
            return fail("tensor size");
        }
        ggml_set_name(t, ent.name.c_str());
    }
    // Host buffer from the CPU buffer type: no backend object to keep alive;
    // the adopting backend frees the buffer in snapshot_free().
    ggml_backend_buffer_t buf =
        ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    if (!buf) {
        ggml_free(ctx);
        return fail("buffer allocation");
    }
    OwnedSnapshot out(ctx, buf, (int)hdr.cur_pos, hdr.last_tok);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (!r.read(t->data, ggml_nbytes(t))) return fail("truncated payload");
    }
    uint64_t stored_sum = 0;
    if (std::fread(&stored_sum, 1, 8, f) != 8) return fail("missing checksum");
    if (std::fgetc(f) != EOF) return fail("trailing bytes");
    if (stored_sum != r.sum.digest()) return fail("checksum mismatch (corrupt file)");
    std::fclose(f);
    return out;
}

// ─── Header I/O ─────────────────────────────────────────────────────────

bool DiskPrefixCache::write_header(FILE * f, const DiskCacheHeader & hdr) {
    uint8_t raw[DISK_CACHE_HEADER_SIZE];
    serialize_header(hdr, raw);
    return std::fwrite(raw, 1, sizeof(raw), f) == sizeof(raw);
}

bool DiskPrefixCache::read_header(FILE * f, DiskCacheHeader & hdr) {
    uint8_t raw[DISK_CACHE_HEADER_SIZE];
    if (std::fread(raw, 1, sizeof(raw), f) != sizeof(raw)) return false;
    parse_header(raw, hdr);
    return true;
}

}  // namespace luce::common
