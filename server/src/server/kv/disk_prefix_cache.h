// One model's disk prefix cache over its directory's DiskStore
// (disk_store.h): its identity salt and layout, lookups and saves. Also the
// disk scope policies (full, fixed:N, off) a server or a request selects.

#pragma once

#include "common/model_backend.h"
#include "server/kv/checkpoint_store.h"
#include "server/kv/disk_store.h"
#include "server/kv/prefix_hash.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace luce::common {

// ─── Disk scope policies ────────────────────────────────────────────────

enum class DiskPrefixCacheMode {
    Off,
    Full,
    Fixed,
};

struct DiskPrefixCachePolicy {
    DiskPrefixCacheMode mode = DiskPrefixCacheMode::Full;
    int fixed_tokens = 0;
};

const char * disk_prefix_cache_mode_name(DiskPrefixCacheMode mode);
std::string disk_prefix_cache_policy_name(const DiskPrefixCachePolicy & policy);

bool parse_disk_prefix_cache_policy(const std::string & value,
                                    DiskPrefixCachePolicy & out);

// Apply a request-level scope string on top of a server-level policy.
// Parses scope_str into a new mode/fixed_tokens that replaces server_policy.
// Returns false (and leaves server_policy unchanged) if scope_str is invalid.
bool apply_request_scope_override(DiskPrefixCachePolicy & server_policy,
                                  const std::string & scope_str);

// Prefix lengths a "full" policy lookup probes: the whole prompt, then every
// chat boundary deepest first. Those are the only lengths this policy ever
// persists (exact prompts and inline snapshots).
std::vector<int> disk_prefix_cache_full_lookup_lengths(
    int prompt_len,
    const std::vector<int> & boundaries,
    int min_tokens);

int disk_prefix_cache_fixed_boundary(const DiskPrefixCachePolicy & policy,
                                     int full_len,
                                     int min_tokens = 1);

// ─── DiskPrefixCache ────────────────────────────────────────────────────

// One model's disk prefix cache: its identity salt and layout directory over
// the directory's DiskStore.
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

    // Initialize: open the directory's store (DiskStore::open: shared with
    // the other models of this process; another process using the directory
    // turns this cache off, logged), then pick the most recently used layout
    // written with this identity salt (enables first-request disk hits).
    // Returns false when the directory cannot be created; the disk cache
    // disables itself then, and when the directory is locked.
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

    // Wait for this model's queued writes (DiskDrainResult); the other
    // models' sharing the directory stay queued.
    using DrainResult = DiskDrainResult;
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

    // Pin (or unpin) the file keyed by `tokens` in this layout: a pinned
    // file is exempt from the TTL, the LRU, the budget and ancestor pruning
    // (it may not be written yet; a save that would need its room fails). Returns its path relative to the cache dir; empty
    // when there is no such file.
    std::string pin(const std::vector<int32_t> & tokens, bool on);
    // The path pin() returns, the file left as it is.
    std::string file_path(const std::vector<int32_t> & tokens);
    // Unpin the file at `relative` (a pin's file of any model's layout).
    void unpin_file(const std::string & relative);
    // This layout's pinned files, as pin() names them.
    std::vector<std::string> pinned_files();
    // This model's layout, as its directory name; empty when not known.
    std::string layout_name() const;
    // A file keyed by exactly `tokens` exists (or is queued) in this layout.
    bool contains(const std::vector<int32_t> & tokens);

    // Bytes on disk (every layout of every model using the directory),
    // queued writes included.
    size_t total_bytes() const;
    size_t entry_count() const;
    // The budget in force: the configured one, clamped to the free space.
    uint64_t budget_bytes() const;

    // The directory's, shared by every model of the process using it.
    using Stats = DiskCacheStats;
    Stats stats() const;

    // Learn the layout fingerprint from a live snapshot (call once after
    // first snapshot_save, before any disk operations). A store that packs
    // its payloads itself (CheckpointStore::exports) gives no SnapshotRef:
    // its layout is learned from the first snapshot saved.
    void learn_layout(Checkpoint checkpoint);

    // Set a config/model identity salt prepended to the layout hash buffer
    // so that layout_id encodes model identity in addition to tensor structure.
    // Call this BEFORE init(). All-zeroes (the default) → back-compat behavior.
    void set_identity_salt(const std::array<uint8_t, 16> & salt) {
        identity_salt_ = salt;
    }

    // Unix-seconds clock for created_at/last_used/TTL. Tests inject one.
    void set_clock(std::function<uint64_t()> clock);

    // Free bytes on the cache's filesystem. Tests inject one.
    void set_free_space_probe(std::function<uint64_t()> probe);

private:
    DiskCacheConfig config_;
    std::unique_ptr<CheckpointStore> owned_store_;  // the ModelBackend constructor's
    CheckpointStore & store_;
    std::function<uint64_t()> clock_;
    std::function<uint64_t()> free_space_;
    // The directory's files, shared with the other models of the process
    // (null before init and when the disk cache is off).
    std::shared_ptr<DiskStore> disk_;
    bool holds_identity_ = false;  // DiskStore::hold_identity
    // Some backends support in-memory snapshots but deliberately do not
    // implement SnapshotRef/adoption. Detect that once from a live slot and
    // stop scheduling disk work for the rest of the process.
    bool backend_snapshot_unsupported_ = false;
    bool off() const { return disabled() || !disk_; }

    // Config/model identity salt (set via set_identity_salt before init()).
    // All-zeroes by default → backward-compatible behavior.
    std::array<uint8_t, 16> identity_salt_{};
    // This model's layout fingerprint (learned from its first snapshot) and
    // directory. Guarded by the store's mutex.
    std::array<uint8_t, 16> layout_id_{};
    bool layout_known_ = false;
    bool layout_from_disk_ = false;  // true if learned from file, unverified
    std::string layout_dir_;  // <cache_dir>/<fingerprint_hex>/

    std::array<uint8_t, 16> layout_of(ggml_context * ctx) const;
    void learn_layout_from(ggml_context * ctx);
    std::string make_path(const PrefixHash & hash) const;
    // This layout's entry for `hash`; needs the store's mutex.
    int find_entry(const PrefixHash & hash) const;
};

}  // namespace luce::common
