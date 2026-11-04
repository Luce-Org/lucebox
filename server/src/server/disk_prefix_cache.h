// Disk-backed prefix cache — persists KV snapshots to disk.
//
// Complements the in-memory PrefixCache by serializing snapshot tensors to
// files, enabling cache survival across restarts and overflow to disk.
//
// File format (v5): 96-byte header + tensor table + the exact key token ids
// + an auxiliary blob (u32 length + bytes; the server stores the session's
// tool-call replay text there) + raw tensor data + an 8-byte checksum of
// everything before it (the header with last_used zeroed, since a hit
// rewrites that field in place). Tensor bytes are copied verbatim in the
// cache's own type (e.g. q4_0 KV stays q4_0): no requantization on save or
// restore. Files are keyed by SHA-1 of prompt token IDs (same as in-memory
// cache); a hit also compares the stored token ids with the prompt before
// adopting anything, and verifies the checksum over the whole file: a
// corrupt or partial file is a logged miss and is deleted. A layout
// fingerprint (SHA-1 of an identity salt plus the tensor names/types/shapes)
// and the identity salt itself prevent loading snapshots from another
// model, KV type or config.
//
// Writes are asynchronous: save_async() takes ownership of a snapshot that
// left RAM, indexes it at once (it counts against the budget and a lookup
// serves it from memory until it is on disk), and a writer thread writes it
// (temporary file, fdatasync, rename, directory sync).
//
// Directory structure:
//   <cache_dir>/<layout_fingerprint_hex>/<token_hash_hex>.dkv
//
// Every layout directory is indexed for the disk budget and the TTL; only
// the live layout's files are ever restored.

#pragma once

#include "prefix_cache.h"
#include "common/model_backend.h"
#include "server/kv/checkpoint_store.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace luce::common {

// ─── Configuration ──────────────────────────────────────────────────────

struct DiskCacheConfig {
    std::string cache_dir;          // base directory (empty = disabled)
    size_t      budget_bytes = (size_t)128 * 1024 * 1024 * 1024;  // 128 GiB default
    int         min_tokens   = 512; // only persist snapshots >= this many tokens
    int         continued_interval = 10240; // save every N tokens during long sessions
    int         cold_max_tokens = 10240;    // create cold checkpoint for prompts longer than this
    // Files unused for longer than this are deleted (at startup, before each
    // save, and when a lookup meets one). 0 = never expire.
    uint64_t    ttl_seconds = 72ull * 3600;
    // Snapshots handed to save_async() but not yet on disk hold host memory.
    // Past this many bytes, save_async() waits for the writer (logged).
    uint64_t    max_pending_bytes = 4ull * 1024 * 1024 * 1024;
    // The budget never exceeds what fits on the filesystem: the cache's own
    // bytes plus the free space, minus this reserve or 5% of the disk (at
    // most 16 GiB), whichever is larger.
    uint64_t    free_space_reserve_bytes = 1ull * 1024 * 1024 * 1024;
};

enum class DiskPrefixCacheMode {
    Off,
    Full,
    Auto,
    Fixed,
};

struct DiskPrefixCachePolicy {
    DiskPrefixCacheMode mode = DiskPrefixCacheMode::Full;
    int fixed_tokens = 0;
    int auto_window = 30;
    // When true, clamp FlowKV disk snapshots to its stable system prefix.
    // False preserves the operator-selected disk-cache scope.
    bool compress = false;
};

const char * disk_prefix_cache_mode_name(DiskPrefixCacheMode mode);
std::string disk_prefix_cache_policy_name(const DiskPrefixCachePolicy & policy);

// Debug aid for the SSD-vs-RAM numerics check: XXH64 over every tensor of a
// snapshot (name, type, shape, bytes) plus cur_pos and last_tok. Enabled at
// restore time by LUCE_DISK_CACHE_DIGEST=1; costs a full device-to-host copy.
bool     disk_cache_digest_enabled();
uint64_t snapshot_state_digest(const ModelBackend::SnapshotRef & ref,
                               int * n_tensors = nullptr,
                               uint64_t * n_bytes = nullptr);
bool parse_disk_prefix_cache_policy(const std::string & value,
                                    DiskPrefixCachePolicy & out);

// Apply a request-level scope string on top of a server-level policy.
// Parses scope_str into a new mode/window/fixed_tokens, then merges it with
// server_policy so that server-level flags (e.g. compress) are preserved.
// Returns false (and leaves server_policy unchanged) if scope_str is invalid.
bool apply_request_scope_override(DiskPrefixCachePolicy & server_policy,
                                  const std::string & scope_str);

// Prefix lengths a "full" policy lookup probes: the whole prompt, then every
// chat boundary deepest first. Those are the only lengths this policy ever
// persists (exact prompts, inline snapshots, cold prefixes).
std::vector<int> disk_prefix_cache_full_lookup_lengths(
    int prompt_len,
    const std::vector<int> & boundaries,
    int min_tokens);

int disk_prefix_cache_fixed_boundary(const DiskPrefixCachePolicy & policy,
                                     int full_len,
                                     int min_tokens = 1);
int disk_prefix_cache_auto_boundary(
    const std::vector<int32_t> & prompt_ids,
    const std::vector<std::vector<int32_t>> & recent_prompts,
    int window,
    const std::vector<int> & safe_boundaries,
    int min_tokens);

// ─── File header (96 bytes, little-endian) ──────────────────────────────

struct DiskCacheHeader {
    char     magic[4];          // "DKVC"
    uint32_t version;           // DISK_CACHE_VERSION
    uint8_t  layout_id[16];    // SHA-1 truncated: tensor structure fingerprint
    uint32_t cur_pos;
    uint32_t n_tensors;
    uint32_t token_count;      // number of prompt tokens
    uint8_t  token_hash[16];   // SHA-1 of prompt token IDs (same as PrefixHash)
    uint64_t payload_bytes;    // total tensor data bytes
    uint64_t created_at;       // unix seconds
    uint64_t last_used;        // unix seconds (updated on hit)
    int32_t  last_tok;         // last prefill token (needed for decode seeding)
    uint8_t  identity_salt[16]; // model/KV-type/config identity (v4)
};
// NOTE: The header is serialized field-by-field, not as a raw struct,
// to avoid alignment/packing issues. The on-disk size is exactly 96 bytes.

static constexpr size_t DISK_CACHE_HEADER_SIZE = 96;
// Byte offset of last_used, rewritten in place when a hit touches a file.
static constexpr long DISK_CACHE_LAST_USED_OFFSET = 68;
// v3: ggml_type id 42 changed meaning (TQ3_0 -> Q2_0).
// v4: identity salt in the header and the key token ids after the tensor
// table.
// v5: an 8-byte checksum trailer over the whole file. Older files are
// deleted at startup.
static constexpr uint32_t DISK_CACHE_VERSION = 5;
static constexpr size_t DISK_CACHE_CHECKSUM_SIZE = 8;

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
    // Snapshots are read from and adopted into `store`'s checkpoints.
    DiskPrefixCache(const DiskCacheConfig & cfg, CheckpointStore & store);
    // Over one backend's snapshot slots, with a store of its own (tests).
    DiskPrefixCache(const DiskCacheConfig & cfg, ModelBackend & backend);
    // Finishes the writes already queued (see drain()).
    ~DiskPrefixCache();

    DiskPrefixCache(const DiskPrefixCache &) = delete;
    DiskPrefixCache & operator=(const DiskPrefixCache &) = delete;

    bool disabled() const {
        return config_.cache_dir.empty() || backend_snapshot_unsupported_;
    }

    // Initialize: create directory, take its lock (another process using it
    // turns this cache off, logged), index every snapshot file, delete
    // corrupt/partial/outdated files and leftover temporaries, clamp the
    // budget to the free space, apply the TTL and the budget, pick the most
    // recently used layout written with this identity salt (enables
    // first-request disk hits), and start the writer thread. Returns false
    // when initialization fails; the disk cache disables itself in that case.
    bool init();

    // Look up a prompt on disk. On hit, adopts the snapshot into
    // `checkpoint`. Returns true if loaded successfully.
    // Equivalent to adopt(load(prompt_ids), checkpoint).
    bool lookup(const std::vector<int32_t> & prompt_ids, Checkpoint checkpoint);

    // Read the snapshot keyed by exactly `prompt_ids` into host memory: from
    // the file (verifying key tokens and checksum; a bad file is deleted),
    // or from memory while its write is still queued. Touches the backend
    // not at all, so it may run on another thread. Empty on miss.
    OwnedSnapshot load(const std::vector<int32_t> & prompt_ids);

    // Hand a loaded snapshot to the store in `checkpoint` (checks the layout
    // against the live model on the first hit after startup).
    bool adopt(OwnedSnapshot snap, Checkpoint checkpoint);

    // Take ownership of a snapshot that left RAM and write it in the
    // background, keyed by `prompt_ids`, with an optional auxiliary blob.
    // It is indexed at once: it counts against the budget (older files are
    // evicted first) and lookups find it. Returns false when it is not
    // written (disabled, below min_tokens, larger than the budget, a
    // snapshot past its key); true when queued or already on disk.
    bool save_async(OwnedSnapshot snap, const std::vector<int32_t> & prompt_ids,
                    const std::string & aux = std::string());

    // Queue a copy of the snapshot in `checkpoint` (save_async); the
    // checkpoint keeps its own. Returns true on success (or when the entry
    // already exists).
    bool save(Checkpoint checkpoint, const std::vector<int32_t> & prompt_ids,
              const std::string & aux = std::string());

    // Wait until every queued write is on disk, at most `max_seconds`
    // (< 0: no limit). Writes still queued then are cancelled and the one
    // in progress is abandoned (its temporary file is deleted).
    struct DrainResult {
        uint64_t files = 0;       // written during the drain
        uint64_t bytes = 0;
        uint64_t cancelled = 0;   // dropped by the time limit
        double   seconds = 0.0;
    };
    DrainResult drain(double max_seconds = -1.0);

    // Visit the auxiliary blobs of the restorable files (this identity),
    // least recently used first. Reads only the small metadata sections.
    void for_each_aux(const std::function<void(const std::string &)> & visit);

    // Index-only probe (no file read): the deepest restorable entry whose
    // key is a prefix of `prompt_ids`. len = key tokens, cur_pos = KV rows
    // the snapshot holds (<= len). {0, 0} on miss.
    struct PrefixProbe { int len = 0; int cur_pos = 0; };
    // With allowed_lengths, only those exact scoped boundaries qualify.
    PrefixProbe longest_prefix(const std::vector<int32_t> & prompt_ids,
                               const std::vector<int> * allowed_lengths = nullptr);

    // Delete live-layout entries whose key is a strict prefix of `tokens`
    // and longer than `keep_max` tokens: a deeper state of the same session
    // supersedes them. Entries up to keep_max (the shared system/tools head)
    // stay for other sessions; when there is none, the shallowest ancestor
    // stays in its place (a longer head). Returns the number of files removed.
    int prune_ancestors(const std::vector<int32_t> & tokens, int keep_max);

    // Check if a continued checkpoint should be saved after generation.
    // `all_tokens` = prompt + generated tokens, `cur_pos` = final position.
    // If cur_pos crosses a continued-interval boundary since last save,
    // saves the snapshot in `checkpoint`. Returns true if a save occurred.
    bool maybe_store_continued(Checkpoint checkpoint, const std::vector<int32_t> & all_tokens,
                               int cur_pos);

    // Reset the continued-store tracking (call at start of each request).
    void reset_continued() { continued_last_store_pos_ = 0; }

    // Find the cold boundary for a long prompt. Returns the token count at
    // which to create a cold checkpoint, or 0 if no cold save is needed.
    // A cold save is needed when: prompt is longer than cold_max_tokens AND
    // there's no existing disk entry covering a prefix of this prompt.
    int cold_prefix_boundary(const std::vector<int32_t> & prompt_ids,
                             const std::vector<int> & boundaries);

    // Get total bytes used on disk (all layouts), queued writes included.
    size_t total_bytes() const;
    size_t entry_count() const;
    int    cold_max_tokens() const { return config_.cold_max_tokens; }
    // The budget in force: the configured one, clamped to the free space.
    uint64_t budget_bytes() const;

    struct Stats {
        uint64_t writes = 0;          // files written
        uint64_t write_bytes = 0;
        uint64_t write_failures = 0;
        uint64_t write_cancelled = 0; // removed (budget, TTL, superseded) before written
        uint64_t pending_files = 0;
        uint64_t pending_bytes = 0;
        uint64_t backpressure_waits = 0;
        uint64_t loads = 0;           // hits served (file or queued write)
        uint64_t load_failures = 0;   // corrupt, partial or mismatched files
    };
    Stats stats() const;

    // Get the continued-interval setting.
    int continued_interval() const { return config_.continued_interval; }

    // Learn the layout fingerprint from a live snapshot (call once after
    // first snapshot_save, before any disk operations).
    void learn_layout(Checkpoint checkpoint);

    // Set a config/model identity salt prepended to the layout hash buffer
    // so that layout_id encodes model identity in addition to tensor structure.
    // Call this BEFORE init(). All-zeroes (the default) → back-compat behavior.
    void set_identity_salt(const std::array<uint8_t, 16> & salt) {
        identity_salt_ = salt;
    }

    // Unix-seconds clock for created_at/last_used/TTL. Tests inject one.
    void set_clock(std::function<uint64_t()> clock) { clock_ = std::move(clock); }

    // Free bytes on the cache's filesystem. Tests inject one.
    void set_free_space_probe(std::function<uint64_t()> probe) {
        free_space_ = std::move(probe);
    }

private:
    DiskCacheConfig config_;
    std::unique_ptr<CheckpointStore> owned_store_;  // the ModelBackend constructor's
    CheckpointStore & store_;
    std::function<uint64_t()> clock_;
    std::function<uint64_t()> free_space_;
    // Some backends support in-memory snapshots but deliberately do not
    // implement SnapshotRef/adoption. Detect that once from a live slot and
    // stop scheduling disk work for the rest of the process.
    bool backend_snapshot_unsupported_ = false;

    // Continued checkpoint tracking (per-session).
    int continued_last_store_pos_ = 0;

    // Config/model identity salt (set via set_identity_salt before init()).
    // All-zeroes by default → backward-compatible behavior.
    std::array<uint8_t, 16> identity_salt_{};

    // Layout fingerprint (learned from first snapshot). Guarded by mu_.
    std::array<uint8_t, 16> layout_id_{};
    bool layout_known_ = false;
    bool layout_from_disk_ = false;  // true if learned from file, unverified
    std::string layout_dir_;  // <cache_dir>/<fingerprint_hex>/

    // A snapshot waiting for the writer. Shared with the index entry so a
    // lookup can copy it while the writer is busy with it.
    struct PendingWrite {
        OwnedSnapshot snap;
        std::vector<int32_t> key;
        std::string aux;
        std::string path;
        uint64_t created_at = 0;
        uint64_t last_used = 0;  // the entry's, when the write started
        std::array<uint8_t, 16> layout_id{};
    };

    // In-memory index of on-disk files (every layout directory).
    struct DiskEntry {
        uint64_t    id = 0;
        std::string path;
        PrefixHash  token_hash;
        std::array<uint8_t, 16> layout_id{};
        uint32_t    token_count = 0;
        uint32_t    cur_pos     = 0;
        uint64_t    file_size   = 0;
        uint64_t    last_used   = 0;
        bool        own_salt    = false;  // written with identity_salt_
        std::shared_ptr<PendingWrite> pending;  // non-null until on disk
    };
    std::vector<DiskEntry> entries_;
    size_t total_bytes_ = 0;
    uint64_t next_entry_id_ = 1;
    uint64_t effective_budget_ = 0;
    mutable std::mutex mu_;

    // Writer thread and its queue (entry ids, oldest first).
    std::deque<uint64_t> queue_;
    uint64_t writing_id_ = 0;       // entry being written, 0 = none
    std::atomic<bool> abort_write_{false};  // drain() time limit hit
    bool     stop_writer_ = false;
    uint64_t pending_bytes_ = 0;
    std::condition_variable writer_cv_;   // work for the writer
    std::condition_variable done_cv_;     // a write finished
    std::thread writer_;
    Stats stats_;
    // Exclusive lock on <cache_dir>/.lock: one process (and one model) per
    // directory, since startup deletes temporaries and the budget evicts
    // files of every layout in it.
#if defined(_WIN32)
    void * lock_handle_ = nullptr;
#else
    int lock_fd_ = -1;
#endif

    // Helpers (the _locked ones need mu_).
    uint64_t now() const;
    std::array<uint8_t, 16> layout_of(ggml_context * ctx) const;
    void learn_layout_from(ggml_context * ctx);
    void scan_all();
    std::string make_path(const PrefixHash & hash) const;
    int find_entry(const PrefixHash & hash) const;
    int find_entry_id(uint64_t id) const;
    bool expired(const DiskEntry & e, uint64_t t) const;
    void remove_entry(size_t idx, const char * why);
    void expire_locked();
    uint64_t budget_locked();
    // Evict least-recently-used files until `incoming` more bytes fit.
    void make_room_locked(uint64_t incoming);
    void writer_loop();
    void release_pending_locked(DiskEntry & e);
    void touch_locked(DiskEntry & e);

    bool write_file(const std::string & path, const PendingWrite & job);
    OwnedSnapshot read_file(const std::string & path,
                            const std::vector<int32_t> & prompt_ids);

    // Header I/O.
    static bool write_header(FILE * f, const DiskCacheHeader & hdr);
    static bool read_header(FILE * f, DiskCacheHeader & hdr);
};

}  // namespace luce::common
