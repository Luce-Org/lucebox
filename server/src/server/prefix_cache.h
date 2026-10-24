// Prefix cache — LRU snapshot cache for system-prompt and full-prompt reuse.
//
// Ported from prefix_cache.py. The C++ version calls ModelBackend snapshot
// methods directly instead of stdin/stdout pipe commands.
//
// Two caching tiers:
//   1. Inline prefix cache: caches system-prompt KV state at turn boundaries.
//      On cache hit, restore_and_generate() diff-prefills only the new turns.
//   2. Full-compress cache: caches the entire compressed prompt's KV state,
//      keyed on the raw (pre-compression) prompt IDs. Skips both compression
//      and prefill on exact-match hits.

#pragma once

#include "tokenizer.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

namespace luce::common {

// ─── Chat marker detection ──────────────────────────────────────────────

struct ChatMarkers {
    std::string family;  // "deepseek", "qwen", "gemma", or "laguna"
    // Token sequences for boundary detection
    std::vector<int32_t> sys_role_prefix;
    std::vector<std::vector<int32_t>> end_msg_seqs;
    std::vector<std::vector<int32_t>> next_role_starts;
    // The template closes only some turns with an end marker (DeepSeek: the
    // system text and user turns end where the next role starts). Every role
    // marker is then a boundary on its own.
    bool role_starts_delimit = false;
};

// Resolve chat markers from the tokenizer (detects Qwen, Gemma, or Laguna family).
bool resolve_chat_markers(const Tokenizer & tok, ChatMarkers & out);

// Find all turn-boundary cut points in a token stream.
std::vector<int> find_all_boundaries(const std::vector<int32_t> & ids,
                                     const ChatMarkers & markers);

// SHA-1 hash of a prefix (truncated to 16 bytes).
using PrefixHash = std::array<uint8_t, 16>;
PrefixHash hash_prefix(const int32_t * ids, int count);

// Prefix-aware inline eviction: given cached prefixes in LRU order (0 = oldest),
// return the index of the oldest "leaf" — an entry that is not a strict prefix
// of any other — so shared ancestors stay resident. A turn kept between a
// conversation's head and its newest snapshot (see
// PrefixCache::prune_superseded_ancestors) goes before any leaf: it only
// serves an edit of the last message. Pointer overload is the core (no token
// copies); the value overload is for tests.
//
// protected_lru (optional, same size): entries marked true go last. When
// every other leaf is protected, the oldest protected leaf is the victim.
// skip_index (default -1): the in-flight restore source, never a victim; if it
// is the only unprotected leaf, evict the shallowest non-protected ancestor
// instead so the restore point can slide, before any protected leaf. A
// protected ancestor of the restore source is never evicted.
//
// Returns the victim index [0, n-1]; -1 if skip_index is set and only the
// restore source and its ancestors remain; 0 if ids_lru is empty or,
// impossibly, no leaf exists.
int select_inline_evict_victim(const std::vector<const std::vector<int32_t> *> & ids_lru,
                               const std::vector<bool> * protected_lru = nullptr,
                               int skip_index = -1);
int select_inline_evict_victim(const std::vector<std::vector<int32_t>> & ids_lru,
                               const std::vector<bool> * protected_lru = nullptr,
                               int skip_index = -1);

// Pick the inline snapshot boundary for a request: the last marker, the end
// of the prompt's last message and the start of the generation prompt, only
// when it advances past an already-restored prefix. That is the point the
// next turn extends: a person's follow-up, or an agent's next step after its
// tool results. A person who edits that message resumes from the previous
// turn's snapshot instead (PrefixCache::prune_superseded_ancestors keeps it).
// When prefer_head is set (a conversation's first turn, or a tool-heavy agent
// request), prefer the first marker (the system/tools head) until that cut is
// already restored, so new conversations on the same head restore it.
// Returns 0 when there is no useful new boundary.
// Cuts below `reachable_from` (the first position the backend can save at,
// see ModelBackend::snapshot_granularity) are skipped.
int select_inline_snapshot_boundary(const std::vector<int> & boundaries,
                                    int restored_prefix_len = 0,
                                    bool prefer_head = false,
                                    int reachable_from = 0);

// Restore points spaced at least `min_gap` tokens apart: the first boundary
// (the system/tools head), then each boundary at least `min_gap` past the last
// kept one. Prefix-stable: the points of a prompt are the points, up to its
// end, of every prompt that extends it, so a checkpoint saved at one is a
// restore point of every later turn. min_gap <= 0 keeps every boundary.
std::vector<int> spaced_restore_points(const std::vector<int> & boundaries, int min_gap);

// Every position a later request may restore `prompt`'s prefix from: its chat
// boundaries, spaced by `spacing` (spaced_restore_points), plus the extra cuts
// a cache may take (a PPP pin, a fixed disk scope), ascending. See
// GenerateRequest::restore_points. `drop_last_boundary` leaves out the
// generation prompt's own boundary: a request that does not snapshot there
// saves no state past it, so that split would only cost a short extra prefill
// step (a snapshot there comes back through `cuts`).
std::vector<int> prefix_restore_points(const std::vector<int32_t> & prompt,
                                       const ChatMarkers & markers,
                                       std::initializer_list<int> cuts,
                                       bool drop_last_boundary = false,
                                       int spacing = 0);

// Return true when a PPP forced cut should override normal boundary
// selection. Once the tools head is already restored, forcing the pin again
// prevents the cache from deepening into the conversation.
bool should_force_inline_snapshot_boundary(
    const std::vector<int> & boundaries,
    int prompt_len,
    int restored_prefix_len,
    bool prefer_tools_boundary,
    int forced_cut,
    int reachable_from = 0);

// ─── Prefix cache entry ─────────────────────────────────────────────────

struct FullCacheEntry {
    int         slot = -1;
    std::string cur_bin_path;
    int         cur_ids_len = 0;
    int         raw_prompt_len = 0;
    int64_t     last_used_ns = 0;
    int         hits = 0;
};

// ─── PrefixCache ────────────────────────────────────────────────────────

class PrefixCache {
public:
    static constexpr int MAX_SLOTS = 64;
    // The HTTP server owns the final backend slot for disk-cache staging.
    static constexpr int MAX_CACHE_SLOTS = MAX_SLOTS - 1;

    // cap = number of prefix-cache slots (0 disables).
    PrefixCache(int cap, const Tokenizer & tokenizer,
                size_t max_resident_bytes = 0);

    bool disabled() const { return disabled_; }

    // Expose chat markers for cold prefix boundary detection.
    const ChatMarkers & chat_markers() const { return markers_; }

    // ── Inline prefix cache ─────────────────────────────────────────

    // Look up the longest cached prefix. Returns (slot, prefix_len) or (-1, 0).
    std::pair<int, int> lookup(const std::vector<int32_t> & prompt_ids);
    // Side-effect-free candidate for engines that must validate payloads.
    std::pair<int, int> lookup_candidate(
        const std::vector<int32_t> & prompt_ids,
        int max_prefix_tokens);

    // Promote and count only after an engine restored this checkpoint.
    void record_inline_hit(
        int slot, int prefix_len, size_t prompt_len);


    class InlineReservation {
    public:
        InlineReservation() = default;
        ~InlineReservation();

        InlineReservation(const InlineReservation &) = delete;
        InlineReservation & operator=(const InlineReservation &) = delete;
        InlineReservation(InlineReservation && other) noexcept;
        InlineReservation & operator=(InlineReservation && other) noexcept;

        bool active() const;
        int slot() const { return slot_; }
        int target_cut() const { return target_cut_; }
        // Committed entries besides the slot's own victim that this capture
        // evicts when it commits (see reserve_inline_snap), in eviction
        // order. Only those still resident are listed. A caller that owns
        // their payloads (the single-sequence path) writes and frees them
        // just before the commit; the commit drops their metadata.
        std::vector<int> extra_victim_slots() const;

        // Commit after the engine saved the payload, cancel before the target
        // slot was touched, or abort after a failed write invalidated it.
        bool commit(const std::vector<int32_t> & prompt_ids,
                    size_t resident_bytes = 0, bool protect = false);
        bool commit_at(const std::vector<int32_t> & prompt_ids,
                       int committed_cut, size_t resident_bytes = 0,
                       bool protect = false);
        void cancel();
        void abort();

    private:
        friend class PrefixCache;
        InlineReservation(PrefixCache * cache, uint64_t id, int slot,
                          int target_cut, PrefixHash victim, bool has_victim,
                          bool protect);
        void clear();
        void take(InlineReservation && other);

        PrefixCache * cache_ = nullptr;
        uint64_t id_ = 0;
        int slot_ = -1;
        int target_cut_ = 0;
        PrefixHash victim_{};
        bool has_victim_ = false;
        bool protect_ = false;
    };

    using InlineSnapshotSize = std::function<size_t(int target_cut)>;

    // Resize the resident budget (the concurrent scheduler sizes it once the
    // batch engine can estimate a checkpoint).
    void set_max_resident_bytes(size_t bytes) {
        max_resident_bytes_ = bytes;
        max_resident_bytes_published_.store(bytes, std::memory_order_relaxed);
    }

    // Select a boundary, destination, and optional budget victim as one owned
    // operation. Each prefilling request can hold one; a live reservation's
    // slot and bytes are out of reach of the others. Destroying it cancels
    // without changing committed metadata.
    // `restored_prefix_len` prevents reserving a slot for a boundary already
    // covered by the restored snapshot. `prefer_tools_boundary` pins the
    // system/tools head first (see select_inline_snapshot_boundary), as a
    // conversation's first turn saves it too. When
    // `forced_cut` > restored, that cut is used instead (PPP pin_end,
    // including mid-message LCP cuts). `restore_source_slot` (default -1) is
    // the slot this request restores from; it is never chosen as the
    // eviction or budget victim and the free-slot path skips it, so the new
    // snapshot lands in a different slot and the restore point can slide
    // forward past the deepest slot. `reachable_from` is forwarded to
    // select_inline_snapshot_boundary; a forced cut below it is not forced.
    // With a `restore_point_spacing` (ModelBackend::restore_point_spacing), a
    // cut other than the end of the last message moves down to the deepest
    // spaced restore point, and nothing is reserved when that adds no prefix.
    // Resident budget: when no single entry makes room for a capture that
    // fits the budget at all, the reservation plans several, least recently
    // used first (edit backups, then leaves); see the definition for what is
    // never planned. All or nothing: a plan exists only when the capture
    // fits once they are gone, else the capture is skipped and nothing is
    // touched. They are evicted when it commits, never when it fails or is
    // cancelled.
    InlineReservation reserve_inline_snap(
        const std::vector<int32_t> & prompt_ids,
        int restored_prefix_len = 0,
        bool prefer_tools_boundary = false,
        int forced_cut = 0,
        int restore_source_slot = -1,
        InlineSnapshotSize estimate_bytes = {},
        int reachable_from = 0,
        int restore_point_spacing = 0);

    // Commit an already-materialized snapshot without a reservation. Used by
    // cache import/bootstrap paths and tests.
    void confirm_inline_snap(int slot, int target_cut,
                             const std::vector<int32_t> & prompt_ids,
                             bool protect = false,
                             size_t resident_bytes = 0);

    // Remove committed metadata for an engine-invalidated checkpoint.
    void invalidate_inline_snap(int slot);

    // Drop the entries that the entry committed in `slot` supersedes: its
    // strict-prefix ancestors, except the shallowest one, protected pins and,
    // while the resident budget holds it, the deepest one: the previous turn,
    // where an edit of the last message resumes.
    // Returns the dropped slots (and, with `pruned_tokens`, each one's prefix
    // length); the caller frees their snapshot payloads.
    std::vector<int> prune_superseded_ancestors(
        int slot, std::vector<int> * pruned_tokens = nullptr);

    // Evicts committed entries, pins last and never `keep_slot` or a pinned
    // ancestor of it, until the resident bytes fit the budget again. Returns
    // the evicted slots; the caller frees their snapshot payloads.
    // `persist` (optional) gets persist_on_evict() of each evicted slot,
    // decided before it was erased.
    std::vector<int> enforce_resident_budget(
        int keep_slot, std::vector<bool> * persist = nullptr);

    // ── Disk persistence of evicted entries ─────────────────────────
    // Whether the entry in `slot`, about to leave RAM, should be written to
    // disk: every conversation's newest snapshot and the shared heads are;
    // a previous turn kept only as an edit backup (see
    // prune_superseded_ancestors) is not while a deeper snapshot of its
    // conversation is resident. False when `slot` holds no entry.
    bool persist_on_evict(int slot) const;
    // Token ids an entry in `slot` covers (empty when none).
    std::vector<int32_t> entry_tokens(int slot) const;
    struct Evicted {
        int slot = -1;
        std::vector<int32_t> ids;
        bool persist = false;
    };
    // Opt-in log of entries a commit replaced (its reservation's victim) or
    // the resident budget evicted, for callers whose payload outlives the
    // metadata (the concurrent scheduler writes them to disk).
    void set_record_evictions(bool on) { record_evictions_ = on; }
    std::vector<Evicted> take_evicted() {
        std::vector<Evicted> out;
        out.swap(evicted_log_);
        return out;
    }
    // Every committed entry, most recently used first, with its
    // persist_on_evict() flag: graceful shutdown evicts them all.
    std::vector<Evicted> entries_for_shutdown() const;

    // Declares that the caller prunes after every commit, so the budget check
    // counts the entries a capture supersedes as freed when it lands.
    void set_prunes_superseded(bool prunes) { prunes_superseded_ = prunes; }

    // Record synchronous scheduler stalls caused by copied checkpoints.
    void record_capture_attempt(uint64_t elapsed_us, bool success);
    void record_restore_attempt(uint64_t elapsed_us, bool restored);

    // Drop all entries (e.g., after OOM recovery).
    void mark_all_cleared();

    // ── Full-compress cache ─────────────────────────────────────────

    // Initialize the full-cache pool. full_cap slots start at cap.
    void init_full_cache(int full_cap);

    // Exact-match lookup. Returns (slot, cur_ids_len) or (-1, 0).
    std::pair<int, int> lookup_full(const std::vector<int32_t> & prompt_ids);

    // Reserve a slot. Returns slot or -1.
    int prepare_full_snap(const std::vector<int32_t> & prompt_ids);

    // Confirm after successful snapshot save.
    void confirm_full_snap(int slot, const std::vector<int32_t> & prompt_ids,
                           int cur_ids_len);

    // Abort reservation.
    void abort_full_snap(int slot);

    // ── Introspection (for /props) ──────────────────────────────────

    struct InlineStats {
        int capacity;
        int in_use;
        int64_t lifetime_hits;
        uint64_t max_resident_bytes;
        uint64_t resident_bytes;
        uint64_t budget_skips;
        uint64_t capture_attempts;
        uint64_t capture_failures;
        uint64_t capture_stall_us_total;
        uint64_t capture_stall_us_max;
        uint64_t restore_attempts;
        uint64_t restore_invalidations;
        uint64_t restore_stall_us_total;
        uint64_t restore_stall_us_max;
    };
    struct FullStats {
        bool enabled;
        int capacity;
        int in_use;
        int64_t disk_bytes;
        int64_t lifetime_hits;
    };

    // Lockless snapshot for /props. Every published field — hit
    // counters, disk-bytes, AND the two in-use counts — is mirrored to
    // an std::atomic that the daemon thread updates alongside the
    // backing vector. /props reads those atomics with
    // memory_order_relaxed, so the cross-thread read is well-defined
    // under the C++ memory model. Used for an ops dashboard; not safe
    // for control-flow decisions.
    InlineStats stats() const;
    FullStats full_stats() const;

private:
    bool disabled_ = true;
    int cap_ = 0;
    ChatMarkers markers_;

    // LRU for inline prefix cache: ordered map of hash → slot.
    // We use a vector to maintain insertion order (front = oldest).
    struct LruEntry {
        PrefixHash           hash;
        int                  slot;
        std::vector<int32_t> ids;  // prefix tokens [0, target_cut) for prefix-aware eviction
        bool                 protect = false;  // sticky tools-boundary pin
        size_t               resident_bytes = 0;
        // Kept behind a deeper snapshot of its conversation only so an
        // edit of the last message can resume (prune_superseded_ancestors).
        bool                 edit_backup = false;
    };
    std::vector<LruEntry> entries_;
    int next_slot_ = 0;
    // Captures in flight, one per prefilling request. Each holds its slot (a
    // free one, or its budget victim's) and its estimated bytes until it
    // commits, aborts or cancels; no other capture selects that slot.
    struct ActiveReservation {
        uint64_t   id;
        int        slot;
        PrefixHash key;
        PrefixHash victim;
        bool       has_victim;
        size_t     bytes;
        // Further entries this capture evicts when it commits, in order.
        std::vector<PrefixHash> extra_victims;
        std::vector<int> extra_slots;
        // Slots of every entry it counts as freed (its superseded ancestors
        // and its victims): no other capture counts them again.
        std::vector<int> credited_slots;
    };
    std::vector<ActiveReservation> active_reservations_;
    uint64_t next_inline_reservation_ = 1;
    size_t max_resident_bytes_ = 0;
    bool prunes_superseded_ = false;
    bool record_evictions_ = false;
    std::vector<Evicted> evicted_log_;
    // Log entries_[idx] into evicted_log_ (when recording) before an erase.
    void record_eviction(int idx);
    size_t resident_bytes_ = 0;

    // Full-cache state
    bool full_disabled_ = true;
    int  full_cap_ = 0;
    int  full_slot_base_ = 0;
    int  full_next_slot_ = 0;

    struct FullLruEntry {
        PrefixHash     hash;
        FullCacheEntry entry;
    };
    std::vector<FullLruEntry> full_entries_;
    PrefixHash full_pending_evict_key_{};
    bool full_has_pending_evict_ = false;
    // Atomic so /props can read them from a client thread without
    // tearing across the daemon thread's increments. Relaxed ordering
    // is sufficient — no synchronization with other state required.
    std::atomic<int64_t> lifetime_hits_{0};       // inline cache hits
    std::atomic<uint64_t> resident_bytes_count_{0};
    std::atomic<uint64_t> max_resident_bytes_published_{0};   // /props mirror of max_resident_bytes_
    std::atomic<uint64_t> budget_skips_{0};
    std::atomic<uint64_t> capture_attempts_{0};
    std::atomic<uint64_t> capture_failures_{0};
    std::atomic<uint64_t> capture_stall_us_total_{0};
    std::atomic<uint64_t> capture_stall_us_max_{0};
    std::atomic<uint64_t> restore_attempts_{0};
    std::atomic<uint64_t> restore_invalidations_{0};
    std::atomic<uint64_t> restore_stall_us_total_{0};
    std::atomic<uint64_t> restore_stall_us_max_{0};
    std::atomic<int64_t> full_lifetime_hits_{0};  // full-compress cache hits
    std::atomic<int64_t> full_disk_bytes_{0};     // best-effort snapshot of disk usage
    // Atomic mirrors of `entries_.size()` and `full_entries_.size()`.
    // The vectors themselves are mutated only on the daemon thread
    // under the daemon's serialised request loop, but `/props` reads
    // happen from the client thread — calling `.size()` there is a
    // data race per the C++ memory model. Bump these alongside every
    // push_back / erase / clear so the public introspection counters
    // stay well-defined. (Codex r1 P2 follow-up.)
    std::atomic<int64_t> entries_size_count_{0};       // mirrors entries_.size()
    std::atomic<int64_t> full_entries_size_count_{0};  // mirrors full_entries_.size()

    // Helpers
    int find_entry(const PrefixHash & h) const;
    int find_slot_entry(int slot) const;
    // Indices of entries a new entry for ids[0, len) supersedes: its strict
    // prefixes except the shallowest and protected pins.
    std::vector<int> superseded_entries(const int32_t * ids, size_t len) const;
    // select_inline_evict_victim over the entries in LRU order.
    int pick_evict_victim(int skip_index) const;
    void erase_inline_entry(int idx);
    void move_to_end(int idx);
    std::pair<int, int> lookup_impl(
        const std::vector<int32_t> & prompt_ids,
        int max_prefix_tokens,
        bool record_hit);
    bool inline_reservation_active(uint64_t id) const;
    void release_inline_reservation(uint64_t id);
    // A capture in flight writes this slot or evicts its entry.
    bool slot_reserved(int slot) const;
    // A capture in flight counts this slot's entry as freed.
    bool slot_credited(int slot) const;
    const ActiveReservation * find_active_reservation(uint64_t id) const;
    bool commit_inline_reservation(InlineReservation & reservation,
                                   const std::vector<int32_t> & prompt_ids,
                                   int committed_cut, size_t resident_bytes,
                                   bool protect);
    void abort_inline_reservation(InlineReservation & reservation);
    void replace_inline_entry(int slot, int target_cut,
                              const std::vector<int32_t> & prompt_ids,
                              bool protect, size_t resident_bytes);

    int find_full_entry(const PrefixHash & h) const;
    void move_full_to_end(int idx);
};

}  // namespace luce::common
