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

#include "prefix_cache_state.h"
#include "tokenizer.h"

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
PrefixHash hash_prefix(const int32_t * ids, int count);

// Prefix-aware inline eviction policy. Given the cached prefixes in LRU order
// (index 0 = oldest), return the index of the eviction victim: the oldest entry
// whose ids are NOT a strict prefix of any other entry's ids (a "leaf"). Keeping
// shared ancestor prefixes resident avoids re-prefilling them for later branches.
// Returns 0 (pure-LRU fallback) when ids_lru is empty or, impossibly, no leaf
// is found. The implementation lives in prefix_cache_state.h so the exact
// production policy can also be model-checked without server dependencies.

// Pick the inline snapshot boundary for a request. We cache the boundary before
// the current user turn (second-to-last marker) and only when it advances past
// an already-restored prefix. Returns 0 when there is no useful new boundary.
int select_inline_snapshot_boundary(const std::vector<int> & boundaries,
                                    int restored_prefix_len = 0);

// Preserve the existing public entry name while keeping one authoritative
// representation in the state core.
using FullCacheEntry = FullPrefixCacheState::Entry;

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

    // Prepare an inline snapshot. `restored_prefix_len` prevents reserving a
    // slot for a boundary already covered by the restored snapshot. Returns
    // (slot, target_cut) or (-1, 0).
    std::pair<int, int> prepare_inline_snap(
        const std::vector<int32_t> & prompt_ids,
        int restored_prefix_len = 0);

    // Confirm after daemon successfully saved the snapshot.
    void confirm_inline_snap(int slot, int target_cut,
                             const std::vector<int32_t> & prompt_ids);

    // Abort if the snapshot failed.
    void abort_inline_snap(int slot);

    // Cancel before the backend slot is touched (for example when the selected
    // destination is also the snapshot being restored). Unlike abort, this
    // preserves the existing entry and only drops the pending reservation.
    void cancel_inline_snap(int slot);

    // Drop all entries (e.g., after OOM recovery).
    void mark_all_cleared();

    // ── Full-compress cache ─────────────────────────────────────────

    // Initialize the full-cache pool. full_cap slots start at cap.
    void init_full_cache(int full_cap);

    // Exact-match lookup. Returns (slot, cur_ids_len) or (-1, 0).
    std::pair<int, int> lookup_full(const std::vector<int32_t> & prompt_ids);

    // Reserve a slot and bind the effective-prompt boundary beyond which the
    // backend must not report a saved snapshot. Returns slot or -1.
    int prepare_full_snap(const std::vector<int32_t> & prompt_ids,
                          int expected_snapshot_len);

    // Confirm after successful snapshot save. Returns false without committing
    // unless slot, raw-prompt key, and saved position match the reservation.
    bool confirm_full_snap(int slot, const std::vector<int32_t> & prompt_ids,
                           int saved_snapshot_len);

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

    // Boundary detection and hashing live in PrefixCache; all inline-cache
    // transitions live in this dependency-free core shared with ESBMC.
    InlinePrefixCacheState inline_state_;

    // Full-cache transitions and LRU ownership live in the same
    // dependency-light state core used by native and formal tests.
    bool full_disabled_ = true;
    int  full_cap_ = 0;
    FullPrefixCacheState full_state_;
    // Atomic so /props can read them from a client thread without
    // tearing across the daemon thread's increments. Relaxed ordering
    // is sufficient — no synchronization with other state required.
    std::atomic<int64_t> lifetime_hits_{0};       // inline cache hits
    std::atomic<int64_t> full_lifetime_hits_{0};  // full-compress cache hits
    std::atomic<int64_t> full_disk_bytes_{0};     // best-effort snapshot of disk usage
    // Atomic mirrors of `inline_state_.size()` and `full_state_.size()`.
    // The backing states are mutated only on the daemon thread
    // under the daemon's serialised request loop, but `/props` reads
    // happen from the client thread — calling `.size()` there is a
    // data race per the C++ memory model. Store these after mutations so the
    // public counters stay well-defined. (Codex r1 P2 follow-up.)
    std::atomic<int64_t> entries_size_count_{0};       // mirrors inline_state_.size()
    std::atomic<int64_t> full_entries_size_count_{0};  // mirrors full_state_.size()

    // Helpers
    void sync_inline_size();
    void sync_full_size();
};

}  // namespace dflash::common
