// Disk-backed prefix cache — persists KV snapshots to disk.
//
// Complements the in-memory PrefixCache by serializing snapshot tensors to
// files, enabling cache survival across restarts and overflow to disk.
//
// File format: 80-byte header + tensor table + raw tensor data.
// Files are keyed by SHA-1 of prompt token IDs (same as in-memory cache).
// A layout fingerprint (SHA-1 of tensor names/types/shapes) prevents loading
// snapshots from incompatible models.
//
// Directory structure:
//   <cache_dir>/<layout_fingerprint_hex>/<token_hash_hex>.dkv

#pragma once

#include "prefix_cache.h"
#include "common/model_backend.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace dflash::common {

// ─── Configuration ──────────────────────────────────────────────────────

struct DiskCacheConfig {
    std::string cache_dir;          // base directory (empty = disabled)
    size_t      budget_bytes = (size_t)4 * 1024 * 1024 * 1024;  // 4 GB default
    int         min_tokens   = 512; // only persist snapshots >= this many tokens
    int         continued_interval = 10240; // save every N tokens during long sessions
    int         cold_max_tokens = 10240;    // create cold checkpoint for prompts longer than this
};

// ─── File header (80 bytes, little-endian) ──────────────────────────────

struct DiskCacheHeader {
    char     magic[4];          // "DKVC"
    uint32_t version;           // 1
    uint8_t  layout_id[16];    // SHA-1 truncated: tensor structure fingerprint
    uint32_t cur_pos;
    uint32_t n_tensors;
    uint32_t token_count;      // number of prompt tokens
    uint8_t  token_hash[16];   // SHA-1 of prompt token IDs (same as PrefixHash)
    uint64_t payload_bytes;    // total tensor data bytes
    uint64_t created_at;       // unix seconds
    uint64_t last_used;        // unix seconds (updated on hit)
    int32_t  last_tok;         // last prefill token (needed for decode seeding)
    // Padding to 80 bytes.
};
// NOTE: The header is serialized field-by-field, not as a raw struct,
// to avoid alignment/packing issues. The on-disk size is exactly 80 bytes.

static constexpr size_t DISK_CACHE_HEADER_SIZE = 80;
static constexpr uint32_t DISK_CACHE_VERSION = 1;

// ─── Tensor table entry (on-disk) ──────────────────────────────────────

struct DiskTensorEntry {
    std::string name;
    uint32_t    type;          // ggml_type enum
    int64_t     ne[4];
    size_t      nbytes;
};

// ─── DiskPrefixCache ────────────────────────────────────────────────────

class DiskPrefixCache {
public:
    DiskPrefixCache(const DiskCacheConfig & cfg, ModelBackend & backend);
    ~DiskPrefixCache() = default;

    DiskPrefixCache(const DiskPrefixCache &) = delete;
    DiskPrefixCache & operator=(const DiskPrefixCache &) = delete;

    bool disabled() const { return config_.cache_dir.empty(); }

    // Initialize: create directory, scan existing files, learn layout from
    // first available snapshot. Returns false on fatal error.
    bool init();

    // Look up a prompt on disk. On hit, loads the snapshot into `out_slot`
    // using backend.snapshot_adopt(). Returns true if loaded successfully.
    bool lookup(const std::vector<int32_t> & prompt_ids, int slot);

    // Save the snapshot in `slot` to disk, keyed by prompt_ids.
    // Returns true on success.
    bool save(int slot, const std::vector<int32_t> & prompt_ids);

    // Check if a continued checkpoint should be saved after generation.
    // `all_tokens` = prompt + generated tokens, `cur_pos` = final position.
    // If cur_pos crosses a continued-interval boundary since last save,
    // saves a snapshot using `slot`. Returns true if a save occurred.
    bool maybe_store_continued(int slot, const std::vector<int32_t> & all_tokens,
                               int cur_pos);

    // Reset the continued-store tracking (call at start of each request).
    void reset_continued() { continued_last_store_pos_ = 0; }

    // Find the cold boundary for a long prompt. Returns the token count at
    // which to create a cold checkpoint, or 0 if no cold save is needed.
    // A cold save is needed when: prompt is longer than cold_max_tokens AND
    // there's no existing disk entry covering a prefix of this prompt.
    int cold_prefix_boundary(const std::vector<int32_t> & prompt_ids,
                             const std::vector<int> & boundaries);

    // Evict files until total disk usage is within budget.
    void enforce_budget();

    // Update last_used timestamp for a file (on cache hit).
    void touch(const PrefixHash & hash);

    // Get total bytes used on disk.
    size_t total_bytes() const { return total_bytes_; }

    // Get the continued-interval setting.
    int continued_interval() const { return config_.continued_interval; }

    // Learn the layout fingerprint from a live snapshot (call once after
    // first snapshot_save, before any disk operations).
    void learn_layout(int slot);

private:
    DiskCacheConfig config_;
    ModelBackend &  backend_;

    // Continued checkpoint tracking (per-session).
    int continued_last_store_pos_ = 0;

    // Layout fingerprint (learned from first snapshot).
    std::array<uint8_t, 16> layout_id_{};
    bool layout_known_ = false;
    bool layout_from_disk_ = false;  // true if learned from file, unverified
    std::string layout_dir_;  // <cache_dir>/<fingerprint_hex>/

    // In-memory index of on-disk files.
    struct DiskEntry {
        std::string path;
        PrefixHash  token_hash;
        uint32_t    token_count = 0;
        uint32_t    cur_pos     = 0;
        uint64_t    file_size   = 0;
        uint64_t    last_used   = 0;
        uint32_t    hits        = 0;
        uint64_t    created_at  = 0;  // for eviction protection
    };
    std::vector<DiskEntry> entries_;
    size_t total_bytes_ = 0;
    std::mutex mu_;

    // Helpers.
    void compute_layout_id(ggml_context * ctx);
    void scan_directory();
    void try_learn_from_disk();
    std::string make_path(const PrefixHash & hash) const;
    int find_entry(const PrefixHash & hash) const;

    bool write_file(const std::string & path,
                    const ModelBackend::SnapshotRef & ref,
                    const std::vector<int32_t> & prompt_ids);
    bool read_file(const std::string & path, int slot);

    // Header I/O.
    static bool write_header(FILE * f, const DiskCacheHeader & hdr);
    static bool read_header(FILE * f, DiskCacheHeader & hdr);
};

}  // namespace dflash::common
