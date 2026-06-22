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
#include <map>
#include <string>
#include <vector>

namespace dflash::common {

// ─── Chat marker detection ──────────────────────────────────────────────

struct ChatMarkers {
    std::string family;  // "qwen", "gemma", or "laguna"
    // Token sequences for boundary detection
    std::vector<int32_t> sys_role_prefix;
    std::vector<std::vector<int32_t>> end_msg_seqs;
    std::vector<std::vector<int32_t>> next_role_starts;
};

// Resolve chat markers from the tokenizer (detects Qwen, Gemma, or Laguna family).
bool resolve_chat_markers(const Tokenizer & tok, ChatMarkers & out);

// Find all turn-boundary cut points in a token stream.
std::vector<int> find_all_boundaries(const std::vector<int32_t> & ids,
                                     const ChatMarkers & markers);

// SHA-1 hash of a prefix (truncated to 16 bytes).
using PrefixHash = std::array<uint8_t, 16>;
PrefixHash hash_prefix(const int32_t * ids, int count);

// Prefix-aware inline eviction policy. Given the cached prefixes in LRU order
// (index 0 = oldest), return the index of the eviction victim: the oldest entry
// whose ids are NOT a strict prefix of any other entry's ids (a "leaf"). Keeping
// shared ancestor prefixes resident avoids re-prefilling them for later branches.
// Returns 0 (pure-LRU fallback) when ids_lru is empty or, impossibly, no leaf
// is found. Pure and model-free so it can be unit-tested without a PrefixCache.
// The pointer overload is the core (the caller passes pointers into its own
// entries so no token vectors are copied); the value overload is a convenience
// wrapper for tests.
int select_inline_evict_victim(const std::vector<const std::vector<int32_t> *> & ids_lru);
int select_inline_evict_victim(const std::vector<std::vector<int32_t>> & ids_lru);

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

    // cap = number of prefix-cache slots (0 disables).
    PrefixCache(int cap, const Tokenizer & tokenizer);

    bool disabled() const { return disabled_; }

    // Expose chat markers for cold prefix boundary detection.
    const ChatMarkers & chat_markers() const { return markers_; }

    // ── Inline prefix cache ─────────────────────────────────────────

    // Look up the longest cached prefix. Returns (slot, prefix_len) or (-1, 0).
    std::pair<int, int> lookup(const std::vector<int32_t> & prompt_ids);

    // Prepare an inline snapshot. Returns (slot, target_cut) or (-1, 0).
    std::pair<int, int> prepare_inline_snap(const std::vector<int32_t> & prompt_ids);

    // Confirm after daemon successfully saved the snapshot.
    void confirm_inline_snap(int slot, int target_cut,
                             const std::vector<int32_t> & prompt_ids);

    // Abort if the snapshot failed.
    void abort_inline_snap(int slot);

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
    };
    std::vector<LruEntry> entries_;
    int next_slot_ = 0;
    PrefixHash pending_evict_key_{};
    bool has_pending_evict_ = false;

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
    void move_to_end(int idx);
    int find_full_entry(const PrefixHash & h) const;
    void move_full_to_end(int idx);
};

}  // namespace dflash::common
