// The files of a KV cache directory: index, budget, TTL, writer thread
// and the .dkv format (disk_store.h).

#include "server/kv/disk_store.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <map>
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

// Subdirectories of the cache directory that are not prefix layouts: the
// sessions tenant's and the pin catalog's.
static constexpr const char * kTenantDir = "sessions";
static constexpr const char * kCatalogDir = "pins";

// ─── Utility ────────────────────────────────────────────────────────────

static bool mkdir_p(const std::string & path) {
    std::error_code ec;
    if (fs::is_directory(path, ec)) return true;
    fs::create_directories(path, ec);
    return fs::is_directory(path, ec);
}

static uint64_t now_unix() {
    return (uint64_t)std::time(nullptr);
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

DiskStore::DiskStore(const DiskCacheConfig & config) : config_(config) {}

DiskStore::~DiskStore() {
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

uint64_t DiskStore::now() const {
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

std::shared_ptr<DiskStore> DiskStore::open(const DiskCacheConfig & config,
                                           std::function<uint64_t()> clock,
                                           std::function<uint64_t()> free_space,
                                           bool & failed) {
    failed = false;
    if (!mkdir_p(config.cache_dir)) {
        std::fprintf(stderr, "[disk-cache] disabled: failed to create dir: %s\n",
                     config.cache_dir.c_str());
        failed = true;
        return {};
    }
    // One store per directory per process: every model of the process that
    // uses the directory shares its lock, index, writer and budget.
    static std::mutex registry_mu;
    static std::map<std::string, std::weak_ptr<DiskStore>> registry;
    std::error_code ec;
    std::string key = fs::weakly_canonical(fs::absolute(config.cache_dir, ec), ec).string();
    if (key.empty()) key = config.cache_dir;
    std::lock_guard<std::mutex> registry_lock(registry_mu);
    if (auto open = registry[key].lock()) {
        std::fprintf(stderr,
                     "[disk-cache] %s is already open in this process: sharing it "
                     "(one budget and one writer for every model)\n",
                     config.cache_dir.c_str());
        return open;
    }
    std::shared_ptr<DiskStore> store(new DiskStore(config));
    store->clock_ = std::move(clock);
    store->free_space_ = std::move(free_space);
    if (!store->lock_dir()) {
        std::fprintf(stderr,
                     "[disk-cache] disabled: cannot lock %s (in use or not writable) "
                     "(pass --kv-cache-dir <other dir> to give this one its own)\n",
                     config.cache_dir.c_str());
        return {};
    }
    {
        std::lock_guard<std::mutex> lock(store->mu_);
        store->scan_all();
        // Budget and TTL hold from the first moment, before any request.
        store->expire_locked();
        (void)store->make_room_locked(0);  // over it with pins alone: they stay
    }
    store->writer_ = std::thread([raw = store.get()] { raw->writer_loop(); });
    registry[key] = store;
    return store;
}

bool DiskStore::lock_dir() {
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
    if (!locked && lock_fd_ >= 0) {
        ::close(lock_fd_);
        lock_fd_ = -1;
    }
#endif
    return locked;
}

// ─── Directory scanning ─────────────────────────────────────────────────

void DiskStore::scan_all() {
    entries_.clear();
    total_bytes_ = 0;
    size_t removed = 0;
    std::error_code ec;
    for (const auto & de : fs::directory_iterator(config_.cache_dir, ec)) {
        const std::string base = de.path().filename().string();
        std::error_code dec;
        if (base.empty() || base[0] == '.' || !de.is_directory(dec)) continue;
        if (base == kTenantDir || base == kCatalogDir) continue;  // not a layout
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
            if (ok && (hex_string(hdr.layout_id, 16) != base ||
                       hex_string(hdr.token_hash, 16) + ".dkv" != name)) {
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
            entry.last_used   = hdr.last_used & ~DISK_CACHE_PINNED_BIT;
            entry.pinned      = (hdr.last_used & DISK_CACHE_PINNED_BIT) != 0;
            entry.file_size   = size;
            std::memcpy(entry.salt.data(), hdr.identity_salt, 16);
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

bool DiskStore::expired(const DiskEntry & e, uint64_t t) const {
    return config_.ttl_seconds > 0 && t > e.last_used &&
           t - e.last_used > config_.ttl_seconds;
}

void DiskStore::release_pending_locked(DiskEntry & e) {
    if (!e.pending) return;
    pending_bytes_ -= std::min<uint64_t>(pending_bytes_, e.file_size);
    e.pending.reset();
}

// The stored last_used of `e`: the time, with the pinned bit.
static uint64_t stored_last_used(const DiskStore::DiskEntry & e) {
    return e.last_used | (e.pinned ? DISK_CACHE_PINNED_BIT : 0);
}

static void rewrite_last_used(const DiskStore::DiskEntry & e) {
    // Not fsynced: a crash may age an entry, but clean restarts retain uses.
    if (FILE * f = std::fopen(e.path.c_str(), "r+b")) {
        if (std::fseek(f, DISK_CACHE_LAST_USED_OFFSET, SEEK_SET) == 0) write_u64(f, stored_last_used(e));
        std::fclose(f);
    }
}

void DiskStore::touch_locked(DiskEntry & e) {
    e.last_used = now();
    if (e.pending) return;  // the writer publishes the latest timestamp
    rewrite_last_used(e);
}

void DiskStore::set_pinned_locked(DiskEntry & e, bool on) {
    if (e.pinned == on) return;
    e.pinned = on;
    if (!e.pending) rewrite_last_used(e);
}

void DiskStore::remove_entry(size_t idx, const char * why) {
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

void DiskStore::expire_locked() {
    const uint64_t t = now();
    for (size_t i = entries_.size(); i-- > 0;) {
        if (!pinned_locked(entries_[i]) && expired(entries_[i], t)) {
            remove_entry(i, "ttl expired");
        }
    }
    if (config_.ttl_seconds == 0) return;
    for (auto & kv : tenants_) {
        uint64_t last_used = 0;
        while (kv.second.oldest(last_used) && t > last_used &&
               t - last_used > config_.ttl_seconds && kv.second.evict_oldest() > 0) {
            std::fprintf(stderr, "[disk-cache] ttl expired: removed a tenant unit unused for %llus\n",
                         (unsigned long long)(t - last_used));
        }
    }
}



uint64_t DiskStore::budget_locked() {
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
        const uint64_t on_disk = total_bytes_ - std::min<uint64_t>(total_bytes_, pending_bytes_) +
                                 tenant_bytes_locked();
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

uint64_t DiskStore::budget_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return effective_budget_;
}

size_t DiskStore::total_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return total_bytes_;
}

size_t DiskStore::entry_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return entries_.size();
}

DiskCacheStats DiskStore::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    DiskCacheStats out = stats_;
    out.pending_bytes = pending_bytes_;
    out.pending_files = 0;
    for (const auto & e : entries_) out.pending_files += e.pending ? 1 : 0;
    return out;
}

uint64_t DiskStore::tenant_bytes_locked() const {
    uint64_t n = 0;
    for (const auto & kv : tenants_) n += kv.second.usage();
    return n;
}

uint64_t DiskStore::oldest_tenant_locked(uint64_t & last_used) const {
    uint64_t id = 0;
    for (const auto & kv : tenants_) {
        uint64_t lu = 0;
        if (kv.second.oldest(lu) && (id == 0 || lu < last_used)) { id = kv.first; last_used = lu; }
    }
    return id;
}

uint64_t DiskStore::attach_tenant(Tenant tenant) {
    std::lock_guard<std::mutex> lock(mu_);
    const uint64_t id = next_tenant_id_++;
    tenants_.emplace(id, std::move(tenant));
    return id;
}

void DiskStore::detach_tenant(uint64_t id) {
    std::lock_guard<std::mutex> lock(mu_);
    tenants_.erase(id);
}

uint64_t DiskStore::tenant_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return tenant_bytes_locked();
}

bool DiskStore::reserve(uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    expire_locked();
    if (!make_room_locked(bytes)) return false;
    reserved_bytes_ += bytes;
    return true;
}

void DiskStore::release(uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    reserved_bytes_ -= std::min(reserved_bytes_, bytes);
}

bool DiskStore::make_room_locked(uint64_t incoming) {
    const uint64_t budget = budget_locked();
    while (used_locked() + incoming > budget) {
        size_t lru = entries_.size();
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (pinned_locked(entries_[i])) continue;
            if (lru == entries_.size() || entries_[i].last_used < entries_[lru].last_used) {
                lru = i;
            }
        }
        // One LRU across the files and the tenants' units (a tie goes to the file).
        uint64_t unit_used = 0;
        const uint64_t tenant = oldest_tenant_locked(unit_used);
        if (tenant != 0 && (lru == entries_.size() || unit_used < entries_[lru].last_used)) {
            const uint64_t freed = tenants_.at(tenant).evict_oldest();
            std::fprintf(stderr, "[disk-cache] budget (lru): removed a tenant unit (%.1f MB)\n", mib(freed));
            if (freed == 0) return false;  // it said it could, and did not: do not spin
            continue;
        }
        if (lru == entries_.size()) return false;  // the pinned files and the tenants' protected units fill it
        remove_entry(lru, "budget (lru)");
    }
    return true;
}

// ─── Writer ─────────────────────────────────────────────────────────────

void DiskStore::writer_loop() {
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
        job->pinned = entries_[(size_t)idx].pinned;
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
                // A hit or a pin while the file was being written: record it too.
                if (e.last_used != job->last_used || e.pinned != job->pinned) {
                    rewrite_last_used(e);
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

DiskDrainResult DiskStore::drain(double max_seconds,
                                 const Owner * only) {
    DiskDrainResult out;
    if (!writer_.joinable()) return out;
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mu_);
    const uint64_t files0 = stats_.writes;
    const uint64_t bytes0 = stats_.write_bytes;
    const auto mine = [&](uint64_t id) {
        if (!only) return true;
        const int idx = find_entry_id(id);
        return idx >= 0 && entries_[(size_t)idx].layout_id == only->layout &&
            entries_[(size_t)idx].salt == only->salt;
    };
    const auto idle = [&] {
        for (const uint64_t id : queue_) {
            if (mine(id)) return false;
        }
        return writing_id_ == 0 || !mine(writing_id_);
    };
    if (max_seconds < 0) {
        done_cv_.wait(lock, idle);
    } else if (!done_cv_.wait_until(
                   lock, t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                       std::chrono::duration<double>(max_seconds)),
                   idle)) {
        const std::deque<uint64_t> queued = queue_;
        for (const uint64_t id : queued) {
            if (!mine(id)) continue;
            ++out.cancelled;
            const int idx = find_entry_id(id);
            if (idx >= 0) remove_entry((size_t)idx, "flush time limit");
            queue_.erase(std::remove(queue_.begin(), queue_.end(), id), queue_.end());
        }
        if (writing_id_ != 0 && mine(writing_id_)) {
            ++out.cancelled;
            abort_write_.store(true);
        }
        done_cv_.wait(lock, idle);
    }
    out.files = stats_.writes - files0;
    out.bytes = stats_.write_bytes - bytes0;
    out.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    return out;
}

int DiskStore::find_entry_id(uint64_t id) const {
    for (int i = 0; i < (int)entries_.size(); ++i) {
        if (entries_[(size_t)i].id == id) return i;
    }
    return -1;
}

// ─── File I/O: Write ────────────────────────────────────────────────────

bool DiskStore::write_file(const std::string & path, const PendingWrite & job) {
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
    hdr.last_used     = job.last_used | (job.pinned ? DISK_CACHE_PINNED_BIT : 0);
    hdr.last_tok      = job.snap.last_tok();
    std::memcpy(hdr.identity_salt, job.salt.data(), 16);

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

OwnedSnapshot DiskStore::read_file(const std::string & path,
                                   const std::vector<int32_t> & prompt_ids,
                                   const std::array<uint8_t, 16> & layout_id,
                                   const std::array<uint8_t, 16> & salt) {
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
    if (std::memcmp(hdr.identity_salt, salt.data(), 16) != 0) {
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

bool DiskStore::write_header(FILE * f, const DiskCacheHeader & hdr) {
    uint8_t raw[DISK_CACHE_HEADER_SIZE];
    serialize_header(hdr, raw);
    return std::fwrite(raw, 1, sizeof(raw), f) == sizeof(raw);
}

bool DiskStore::read_aux(const std::string & path, std::string & aux) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    DiskCacheHeader hdr{};
    const bool ok = read_header(f, hdr) &&
        read_aux_section(f, hdr.n_tensors, hdr.token_count, aux);
    std::fclose(f);
    return ok;
}

bool DiskStore::read_header(FILE * f, DiskCacheHeader & hdr) {
    uint8_t raw[DISK_CACHE_HEADER_SIZE];
    if (std::fread(raw, 1, sizeof(raw), f) != sizeof(raw)) return false;
    parse_header(raw, hdr);
    return true;
}

}  // namespace luce::common
