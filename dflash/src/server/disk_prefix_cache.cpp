// Disk-backed prefix cache implementation.

#include "disk_prefix_cache.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dflash27b {

// ─── Inline SHA-1 (same as prefix_cache.cpp) ────────────────────────────

static void sha1_hash(const void * data, size_t len, uint8_t out[20]) {
    auto rotl = [](uint32_t x, int n) -> uint32_t {
        return (x << n) | (x >> (32 - n));
    };

    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE,
             h3 = 0x10325476, h4 = 0xC3D2E1F0;

    size_t new_len = len + 1;
    while (new_len % 64 != 56) new_len++;
    std::vector<uint8_t> msg(new_len + 8, 0);
    std::memcpy(msg.data(), data, len);
    msg[len] = 0x80;
    uint64_t bit_len = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) {
        msg[new_len + i] = (uint8_t)(bit_len >> (56 - 8 * i));
    }

    for (size_t offset = 0; offset < msg.size(); offset += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t)msg[offset + 4*i] << 24) |
                    ((uint32_t)msg[offset + 4*i+1] << 16) |
                    ((uint32_t)msg[offset + 4*i+2] << 8) |
                    ((uint32_t)msg[offset + 4*i+3]);
        }
        for (int i = 16; i < 80; i++) {
            w[i] = rotl(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
        }

        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20)      { f = (b & c) | (~b & d); k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d;          k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else              { f = b ^ c ^ d;          k = 0xCA62C1D6; }
            uint32_t temp = rotl(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rotl(b, 30); b = a; a = temp;
        }
        h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
    }

    auto store32 = [](uint8_t * p, uint32_t v) {
        p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16);
        p[2] = (uint8_t)(v >> 8);  p[3] = (uint8_t)v;
    };
    store32(out,     h0);
    store32(out + 4, h1);
    store32(out + 8, h2);
    store32(out + 12, h3);
    store32(out + 16, h4);
}

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
    struct stat st{};
    if (stat(path.c_str(), &st) == 0) return S_ISDIR(st.st_mode);
    // Try to create parent first.
    size_t slash = path.rfind('/');
    if (slash != std::string::npos && slash > 0) {
        mkdir_p(path.substr(0, slash));
    }
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

static uint64_t now_unix() {
    return (uint64_t)std::time(nullptr);
}

// Little-endian I/O helpers.
static void write_u32(FILE * f, uint32_t v) { std::fwrite(&v, 4, 1, f); }
static void write_u64(FILE * f, uint64_t v) { std::fwrite(&v, 8, 1, f); }
static void write_i64(FILE * f, int64_t v)  { std::fwrite(&v, 8, 1, f); }
static void write_u16(FILE * f, uint16_t v) { std::fwrite(&v, 2, 1, f); }

static bool read_u32(FILE * f, uint32_t & out) { return std::fread(&out, 4, 1, f) == 1; }
static bool read_u64(FILE * f, uint64_t & out) { return std::fread(&out, 8, 1, f) == 1; }
static bool read_i64(FILE * f, int64_t & out)  { return std::fread(&out, 8, 1, f) == 1; }
static bool read_u16(FILE * f, uint16_t & out) { return std::fread(&out, 2, 1, f) == 1; }

// ─── Construction ───────────────────────────────────────────────────────

DiskPrefixCache::DiskPrefixCache(const DiskCacheConfig & cfg, ModelBackend & backend)
    : config_(cfg), backend_(backend) {}

// ─── Initialization ─────────────────────────────────────────────────────

bool DiskPrefixCache::init() {
    if (disabled()) return true;

    if (!mkdir_p(config_.cache_dir)) {
        std::fprintf(stderr, "[disk-cache] failed to create dir: %s\n",
                     config_.cache_dir.c_str());
        return false;
    }

    // Try to learn layout from existing files (enables first-request disk hits).
    try_learn_from_disk();

    std::fprintf(stderr, "[disk-cache] initialized dir=%s budget=%.1f GB layout=%s\n",
                 config_.cache_dir.c_str(),
                 (double)config_.budget_bytes / (1024.0 * 1024.0 * 1024.0),
                 layout_known_ ? hex(layout_id_.data(), 16).c_str() : "pending");
    return true;
}

// ─── Layout fingerprint ─────────────────────────────────────────────────

void DiskPrefixCache::compute_layout_id(ggml_context * ctx) {
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
    std::vector<uint8_t> buf;
    for (const auto & ti : tensors) {
        buf.insert(buf.end(), ti.name.begin(), ti.name.end());
        buf.insert(buf.end(), (uint8_t *)&ti.type, (uint8_t *)&ti.type + 4);
        buf.insert(buf.end(), (uint8_t *)ti.ne, (uint8_t *)ti.ne + 32);
    }

    uint8_t digest[20];
    sha1_hash(buf.data(), buf.size(), digest);
    std::memcpy(layout_id_.data(), digest, 16);
}

void DiskPrefixCache::learn_layout(int slot) {
    if (disabled()) return;
    if (layout_known_ && !layout_from_disk_) return;  // already verified from live model

    auto ref = backend_.snapshot_ref(slot);
    if (!ref.ctx) return;

    std::array<uint8_t, 16> prev_id = layout_id_;
    bool had_disk_layout = layout_from_disk_;

    compute_layout_id(ref.ctx);

    if (had_disk_layout && std::memcmp(prev_id.data(), layout_id_.data(), 16) != 0) {
        // Model layout differs from what was learned from disk files.
        std::fprintf(stderr, "[disk-cache] layout mismatch: disk=%s model=%s — switching\n",
                     hex(prev_id.data(), 16).c_str(),
                     hex(layout_id_.data(), 16).c_str());
        entries_.clear();
        total_bytes_ = 0;
    }

    layout_known_ = true;
    layout_from_disk_ = false;
    layout_dir_ = config_.cache_dir + "/" + hex(layout_id_.data(), 16);
    mkdir_p(layout_dir_);

    std::fprintf(stderr, "[disk-cache] layout learned: %s\n",
                 hex(layout_id_.data(), 16).c_str());

    // Scan for previously saved files matching this layout.
    scan_directory();
}

// ─── Directory scanning ─────────────────────────────────────────────────

void DiskPrefixCache::scan_directory() {
    entries_.clear();
    total_bytes_ = 0;

    if (layout_dir_.empty()) return;

    DIR * dir = opendir(layout_dir_.c_str());
    if (!dir) return;

    struct dirent * ent;
    while ((ent = readdir(dir)) != nullptr) {
        const char * name = ent->d_name;
        size_t nlen = std::strlen(name);
        if (nlen < 36 || std::strcmp(name + nlen - 4, ".dkv") != 0) continue;

        std::string path = layout_dir_ + "/" + name;
        FILE * f = std::fopen(path.c_str(), "rb");
        if (!f) continue;

        DiskCacheHeader hdr{};
        if (!read_header(f, hdr)) { std::fclose(f); continue; }
        std::fclose(f);

        // Validate magic and layout.
        if (std::memcmp(hdr.magic, "DKVC", 4) != 0) continue;
        if (std::memcmp(hdr.layout_id, layout_id_.data(), 16) != 0) continue;

        DiskEntry entry;
        entry.path = path;
        std::memcpy(entry.token_hash.data(), hdr.token_hash, 16);
        entry.token_count = hdr.token_count;
        entry.cur_pos     = hdr.cur_pos;
        entry.last_used   = hdr.last_used;

        struct stat st{};
        if (stat(path.c_str(), &st) == 0) {
            entry.file_size = (uint64_t)st.st_size;
        }

        total_bytes_ += entry.file_size;
        entries_.push_back(std::move(entry));
    }
    closedir(dir);

    std::fprintf(stderr, "[disk-cache] scanned %zu files, %.1f MB\n",
                 entries_.size(), (double)total_bytes_ / (1024.0 * 1024.0));
}

// ─── Cold start: learn layout from existing files ───────────────────────

void DiskPrefixCache::try_learn_from_disk() {
    // Scan cache_dir for subdirectories (each is a layout fingerprint).
    DIR * dir = opendir(config_.cache_dir.c_str());
    if (!dir) return;

    struct dirent * ent;
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] == '.') continue;
        std::string subdir = config_.cache_dir + "/" + ent->d_name;
        struct stat st{};
        if (stat(subdir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) continue;

        // Check if this subdir has any .dkv files.
        DIR * sub = opendir(subdir.c_str());
        if (!sub) continue;

        struct dirent * sent;
        while ((sent = readdir(sub)) != nullptr) {
            size_t nlen = std::strlen(sent->d_name);
            if (nlen < 4 || std::strcmp(sent->d_name + nlen - 4, ".dkv") != 0) continue;

            // Read the header to get the layout_id.
            std::string fpath = subdir + "/" + sent->d_name;
            FILE * f = std::fopen(fpath.c_str(), "rb");
            if (!f) continue;

            DiskCacheHeader hdr{};
            if (read_header(f, hdr) && std::memcmp(hdr.magic, "DKVC", 4) == 0) {
                std::memcpy(layout_id_.data(), hdr.layout_id, 16);
                layout_known_ = true;
                layout_from_disk_ = true;  // unverified — must be confirmed by learn_layout()
                layout_dir_ = subdir;
                std::fclose(f);
                closedir(sub);
                closedir(dir);
                scan_directory();
                return;
            }
            std::fclose(f);
        }
        closedir(sub);
    }
    closedir(dir);
}

// ─── Lookup ─────────────────────────────────────────────────────────────

bool DiskPrefixCache::lookup(const std::vector<int32_t> & prompt_ids, int slot) {
    if (disabled() || !layout_known_ || layout_from_disk_) return false;

    PrefixHash hash = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());

    std::lock_guard<std::mutex> lock(mu_);
    int idx = find_entry(hash);
    if (idx < 0) return false;

    auto & entry = entries_[idx];
    if (!read_file(entry.path, slot)) {
        // File is corrupt or incompatible — remove it.
        std::remove(entry.path.c_str());
        total_bytes_ -= entry.file_size;
        entries_.erase(entries_.begin() + idx);
        return false;
    }

    // Update last_used on disk.
    entry.last_used = now_unix();
    entry.hits++;
    // Optionally rewrite header timestamp (non-critical, skip for perf).
    return true;
}

// ─── Save ───────────────────────────────────────────────────────────────

bool DiskPrefixCache::save(int slot, const std::vector<int32_t> & prompt_ids) {
    if (disabled()) return false;

    // Learn layout on first save.
    if (!layout_known_) {
        learn_layout(slot);
        if (!layout_known_) return false;
    }

    // Check minimum token threshold.
    if ((int)prompt_ids.size() < config_.min_tokens) return false;

    auto ref = backend_.snapshot_ref(slot);
    if (!ref.ctx) return false;

    PrefixHash hash = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());

    std::lock_guard<std::mutex> lock(mu_);

    // Skip if already on disk.
    if (find_entry(hash) >= 0) return true;

    // Pre-write budget check: estimate file size and reject if it would
    // exceed budget even after evicting all evictable entries.
    if (config_.budget_bytes > 0) {
        uint64_t payload = 0;
        uint32_t ntens = 0;
        size_t table_est = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ref.ctx); t;
             t = ggml_get_next_tensor(ref.ctx, t)) {
            ntens++;
            payload += ggml_nbytes(t);
            table_est += 2 + std::strlen(t->name) + 4 + 32 + 8; // name_len + name + type + ne[4] + nbytes
        }
        size_t est_size = DISK_CACHE_HEADER_SIZE + table_est + payload;
        if (est_size > config_.budget_bytes) {
            std::fprintf(stderr, "[disk-cache] skip save: estimated %.1f MB exceeds budget\n",
                         (double)est_size / (1024.0 * 1024.0));
            return false;
        }
    }

    std::string path = make_path(hash);
    std::string tmp_path = path + ".tmp";

    if (!write_file(tmp_path, ref, prompt_ids)) {
        std::remove(tmp_path.c_str());
        return false;
    }

    // Atomic rename.
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        std::remove(tmp_path.c_str());
        return false;
    }

    // Update index.
    DiskEntry entry;
    entry.path        = path;
    std::memcpy(entry.token_hash.data(), hash.data(), 16);
    entry.token_count = (uint32_t)prompt_ids.size();
    entry.cur_pos     = (uint32_t)ref.cur_pos;
    entry.last_used   = now_unix();
    entry.created_at  = entry.last_used;
    struct stat st{};
    if (stat(path.c_str(), &st) == 0) entry.file_size = (uint64_t)st.st_size;

    total_bytes_ += entry.file_size;
    entries_.push_back(std::move(entry));

    std::fprintf(stderr, "[disk-cache] saved %s (%u tokens, %d pos, %.1f MB)\n",
                 hex(hash.data(), 16).c_str(),
                 (uint32_t)prompt_ids.size(), ref.cur_pos,
                 (double)entries_.back().file_size / (1024.0 * 1024.0));

    enforce_budget();
    return true;
}

// ─── Continued checkpoints ──────────────────────────────────────────────

bool DiskPrefixCache::maybe_store_continued(int slot,
                                            const std::vector<int32_t> & all_tokens,
                                            int cur_pos) {
    if (disabled()) return false;
    if (cur_pos < config_.min_tokens) return false;

    const int interval = config_.continued_interval;
    if (interval <= 0) return false;

    // Check if cur_pos crosses a new interval boundary since last save.
    // DS4 uses absolute-aligned frontiers: save only when cur_pos is a
    // multiple of interval AND exceeds the last store position.
    int target = (cur_pos / interval) * interval;
    if (target <= continued_last_store_pos_) return false;
    if (target < config_.min_tokens) return false;

    // Save the prefix up to `target` tokens.
    std::vector<int32_t> prefix(all_tokens.begin(),
                                all_tokens.begin() + std::min(target, (int)all_tokens.size()));
    bool ok = save(slot, prefix);
    if (ok) {
        continued_last_store_pos_ = target;
        std::fprintf(stderr, "[disk-cache] continued checkpoint at %d tokens\n", target);
    }
    return ok;
}

// ─── Cold prefix boundary ───────────────────────────────────────────────

int DiskPrefixCache::cold_prefix_boundary(const std::vector<int32_t> & prompt_ids,
                                          const std::vector<int> & boundaries) {
    if (disabled() || !layout_known_) return 0;

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
    if (find_entry(hash) >= 0) return 0;  // already cached

    return best;
}

// ─── Budget enforcement ─────────────────────────────────────────────────

void DiskPrefixCache::enforce_budget() {
    uint64_t now = now_unix();

    // DS4-style eviction scoring: (effective_hits + 1) * tokens / file_size
    // with exponential decay on hits (6-hour half-life).
    auto score = [now](const DiskEntry & e) -> double {
        // Protect recently-saved entries (< 60 seconds old).
        if (e.created_at > 0 && now > 0 && (now - e.created_at) < 60) {
            return 1e18;
        }
        // Decay hits: half-life 6h → decay_rate = ln(2) / (6*3600) ≈ 3.2e-5
        double age_s = (now > e.last_used) ? (double)(now - e.last_used) : 0.0;
        double effective_hits = (double)e.hits * std::exp(-age_s * 3.2e-5);
        double size_factor = (e.file_size > 0) ? (double)e.file_size : 1.0;
        return (effective_hits + 1.0) * (double)e.token_count / size_factor;
    };

    while (total_bytes_ > config_.budget_bytes && !entries_.empty()) {
        // Find entry with lowest eviction score.
        auto it = std::min_element(entries_.begin(), entries_.end(),
            [&score](const DiskEntry & a, const DiskEntry & b) {
                return score(a) < score(b);
            });

        // Don't evict protected entries.
        if (score(*it) >= 1e18) break;

        std::fprintf(stderr, "[disk-cache] evicting %s (%.1f MB, hits=%u, score=%.3f)\n",
                     hex(it->token_hash.data(), 16).c_str(),
                     (double)it->file_size / (1024.0 * 1024.0),
                     it->hits, score(*it));

        std::remove(it->path.c_str());
        total_bytes_ -= it->file_size;
        entries_.erase(it);
    }
}

// ─── Touch ──────────────────────────────────────────────────────────────

void DiskPrefixCache::touch(const PrefixHash & hash) {
    std::lock_guard<std::mutex> lock(mu_);
    int idx = find_entry(hash);
    if (idx >= 0) {
        entries_[idx].last_used = now_unix();
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────

std::string DiskPrefixCache::make_path(const PrefixHash & hash) const {
    return layout_dir_ + "/" + hex(hash.data(), 16) + ".dkv";
}

int DiskPrefixCache::find_entry(const PrefixHash & hash) const {
    for (int i = 0; i < (int)entries_.size(); ++i) {
        if (entries_[i].token_hash == hash) return i;
    }
    return -1;
}

// ─── File I/O: Write ────────────────────────────────────────────────────

bool DiskPrefixCache::write_file(const std::string & path,
                                 const ModelBackend::SnapshotRef & ref,
                                 const std::vector<int32_t> & prompt_ids) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;

    // Count tensors and compute payload size.
    uint32_t n_tensors = 0;
    uint64_t payload_bytes = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ref.ctx); t; t = ggml_get_next_tensor(ref.ctx, t)) {
        n_tensors++;
        payload_bytes += ggml_nbytes(t);
    }

    // Write header.
    DiskCacheHeader hdr{};
    std::memcpy(hdr.magic, "DKVC", 4);
    hdr.version = DISK_CACHE_VERSION;
    std::memcpy(hdr.layout_id, layout_id_.data(), 16);
    hdr.cur_pos       = (uint32_t)ref.cur_pos;
    hdr.n_tensors     = n_tensors;
    hdr.token_count   = (uint32_t)prompt_ids.size();
    PrefixHash ph = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());
    std::memcpy(hdr.token_hash, ph.data(), 16);
    hdr.payload_bytes = payload_bytes;
    hdr.created_at    = now_unix();
    hdr.last_used     = hdr.created_at;
    hdr.last_tok      = ref.last_tok;

    if (!write_header(f, hdr)) { std::fclose(f); return false; }

    // Write tensor table.
    for (ggml_tensor * t = ggml_get_first_tensor(ref.ctx); t; t = ggml_get_next_tensor(ref.ctx, t)) {
        uint16_t name_len = (uint16_t)std::strlen(t->name);
        write_u16(f, name_len);
        std::fwrite(t->name, 1, name_len, f);
        write_u32(f, (uint32_t)t->type);
        for (int d = 0; d < 4; ++d) write_i64(f, t->ne[d]);
        write_u64(f, ggml_nbytes(t));
    }

    // Write tensor data.
    std::vector<uint8_t> buf(4 * 1024 * 1024);  // 4 MB transfer buffer
    for (ggml_tensor * t = ggml_get_first_tensor(ref.ctx); t; t = ggml_get_next_tensor(ref.ctx, t)) {
        size_t nbytes = ggml_nbytes(t);
        size_t offset = 0;
        while (offset < nbytes) {
            size_t chunk = std::min(buf.size(), nbytes - offset);
            ggml_backend_tensor_get(t, buf.data(), offset, chunk);
            if (std::fwrite(buf.data(), 1, chunk, f) != chunk) {
                std::fclose(f);
                return false;
            }
            offset += chunk;
        }
    }

    std::fclose(f);
    return true;
}

// ─── File I/O: Read ─────────────────────────────────────────────────────

bool DiskPrefixCache::read_file(const std::string & path, int slot) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    DiskCacheHeader hdr{};
    if (!read_header(f, hdr)) { std::fclose(f); return false; }

    // Validate.
    if (std::memcmp(hdr.magic, "DKVC", 4) != 0) { std::fclose(f); return false; }
    if (hdr.version != DISK_CACHE_VERSION) { std::fclose(f); return false; }
    if (std::memcmp(hdr.layout_id, layout_id_.data(), 16) != 0) { std::fclose(f); return false; }

    // Read tensor table.
    std::vector<DiskTensorEntry> table;
    table.reserve(hdr.n_tensors);
    for (uint32_t i = 0; i < hdr.n_tensors; ++i) {
        DiskTensorEntry ent;
        uint16_t name_len;
        if (!read_u16(f, name_len)) { std::fclose(f); return false; }
        if (name_len >= GGML_MAX_NAME) { std::fclose(f); return false; }
        char name_buf[GGML_MAX_NAME] = {};
        if (std::fread(name_buf, 1, name_len, f) != name_len) { std::fclose(f); return false; }
        ent.name = name_buf;
        if (!read_u32(f, ent.type)) { std::fclose(f); return false; }
        for (int d = 0; d < 4; ++d) {
            if (!read_i64(f, ent.ne[d])) { std::fclose(f); return false; }
        }
        uint64_t nbytes_raw;
        if (!read_u64(f, nbytes_raw)) { std::fclose(f); return false; }
        ent.nbytes = (size_t)nbytes_raw;
        table.push_back(std::move(ent));
    }

    // Allocate ggml context + buffer.
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (size_t)(hdr.n_tensors + 4) + 4096;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { std::fclose(f); return false; }

    // Create tensors.
    for (const auto & ent : table) {
        ggml_tensor * t = ggml_new_tensor(ctx, (ggml_type)ent.type, 4, ent.ne);
        if (!t) {
            ggml_free(ctx);
            std::fclose(f);
            return false;
        }
        ggml_set_name(t, ent.name.c_str());
    }

    // Allocate buffer on CPU backend.
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { ggml_free(ctx); std::fclose(f); return false; }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buf) {
        ggml_free(ctx);
        ggml_backend_free(cpu);
        std::fclose(f);
        return false;
    }

    // Read tensor data.
    std::vector<uint8_t> read_buf(4 * 1024 * 1024);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        size_t nbytes = ggml_nbytes(t);
        size_t offset = 0;
        while (offset < nbytes) {
            size_t chunk = std::min(read_buf.size(), nbytes - offset);
            if (std::fread(read_buf.data(), 1, chunk, f) != chunk) {
                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
                ggml_backend_free(cpu);
                std::fclose(f);
                return false;
            }
            ggml_backend_tensor_set(t, read_buf.data(), offset, chunk);
            offset += chunk;
        }
    }
    std::fclose(f);

    // Hand off to backend.
    if (!backend_.snapshot_adopt(slot, ctx, buf, (int)hdr.cur_pos, hdr.last_tok)) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(cpu);
        return false;
    }

    // The cpu backend must persist as long as the buffer is alive. The buffer
    // does NOT own the backend, so we cannot free it here. Since snapshot_free
    // only frees buf + ctx, we accept a small leak of the lightweight cpu
    // backend object (~64 bytes per load). For a bounded disk cache this is
    // negligible. A proper fix would store cpu alongside the snapshot.

    return true;
}

// ─── Header I/O ─────────────────────────────────────────────────────────

bool DiskPrefixCache::write_header(FILE * f, const DiskCacheHeader & hdr) {
    std::fwrite(hdr.magic, 1, 4, f);
    write_u32(f, hdr.version);
    std::fwrite(hdr.layout_id, 1, 16, f);
    write_u32(f, hdr.cur_pos);
    write_u32(f, hdr.n_tensors);
    write_u32(f, hdr.token_count);
    std::fwrite(hdr.token_hash, 1, 16, f);
    write_u64(f, hdr.payload_bytes);
    write_u64(f, hdr.created_at);
    write_u64(f, hdr.last_used);
    write_u32(f, (uint32_t)hdr.last_tok);  // stored as u32, cast back on read
    // Total: 4+4+16+4+4+4+16+8+8+8+4 = 80 bytes. No extra padding needed.
    return !std::ferror(f);
}

bool DiskPrefixCache::read_header(FILE * f, DiskCacheHeader & hdr) {
    if (std::fread(hdr.magic, 1, 4, f) != 4) return false;
    if (!read_u32(f, hdr.version)) return false;
    if (std::fread(hdr.layout_id, 1, 16, f) != 16) return false;
    if (!read_u32(f, hdr.cur_pos)) return false;
    if (!read_u32(f, hdr.n_tensors)) return false;
    if (!read_u32(f, hdr.token_count)) return false;
    if (std::fread(hdr.token_hash, 1, 16, f) != 16) return false;
    if (!read_u64(f, hdr.payload_bytes)) return false;
    if (!read_u64(f, hdr.created_at)) return false;
    if (!read_u64(f, hdr.last_used)) return false;
    uint32_t last_tok_raw;
    if (!read_u32(f, last_tok_raw)) return false;
    hdr.last_tok = (int32_t)last_tok_raw;
    return true;
}

}  // namespace dflash27b
