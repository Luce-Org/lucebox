// KVFlash C1.4: per-slot mode-4 turn bookkeeping, pure.
//
// A slot's turn is the single-slot mode 4 (qwen35_backend.cpp) over the step
// engine: a first pass over [kv_offset, end) under LRU with the state snapshot
// at R (start of the last user turn) and the last T prompt queries captured,
// then score, rewind to R, apply the picks, replay [R, end). Mode 5 keeps the
// turn one pass and applies the drafter's picks when the prefill reaches R. The
// engine keeps one KvFlashSlotTurn per slot and asks these helpers what the next
// cut is and whether the step that runs it snapshots or captures.
#pragma once

#include "kvflash_prefill_reselect.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace luce::common {

enum class KvFlashTurnPhase : uint8_t { Lru, FirstPass, Replay };

struct KvFlashSlotTurn {
    KvFlashTurnPhase phase = KvFlashTurnPhase::Lru;
    int  kv_offset = 0;             // resume point: tokens [0, kv_offset) are the session
    int  r = -1;                    // rewind point
    int  end = 0;                   // prompt end
    int  reserve_chunks = 0;        // pool room kept for the replay (single-slot rule)
    int  apply_at = -1;             // mode 5: the cut at which the drafter's picks are applied (one pass, no snapshot at R); -1: none
    std::vector<int> cuts;          // first-pass cut ends, ascending
    bool recall() const { return phase != KvFlashTurnPhase::Lru; }
    bool first_pass() const { return phase == KvFlashTurnPhase::FirstPass; }
    bool snapshot_at_r() const { return recall() && r > kv_offset; }
};

// First-pass cut ends over [kv_offset, end): r is a cut end (the state snapshot
// is taken there), cuts are U long after it, and the last one is at least
// min(T, end - r) long (its predecessor shrinks) so one step captures all T queries.
inline std::vector<int> kvflash_slot_turn_cuts(int kv_offset, int r, int end, int cut_u, int n_queries) {
    std::vector<int> cuts;
    if (cut_u <= 0 || end <= kv_offset) return cuts;
    r = std::clamp(r, kv_offset, end);
    for (int p = kv_offset; p < r;) { p = std::min(p + cut_u, r); cuts.push_back(p); }
    const size_t tail0 = cuts.size();
    for (int p = r; p < end;) { p = std::min(p + cut_u, end); cuts.push_back(p); }
    const int need = std::min(std::max(n_queries, 1), end - r);
    if (cuts.size() - tail0 >= 2 && cuts.back() - cuts[cuts.size() - 2] < need) cuts[cuts.size() - 2] = end - need;
    return cuts;
}

// The slot's turn for a prompt resumed at kv_offset. Mode 4 runs when the recall
// profile is on and the prompt with the replay's reserve does not fit the slot's
// pool (kvflash_pool_fits, the single-slot rule); otherwise the slot stays LRU.
// pflash2 (mode 5): the picks come from the drafter's scores of the prompt ids, so the turn stays one
// pass and only records where to apply them (apply_at = R).
inline KvFlashSlotTurn kvflash_slot_turn_plan(const std::vector<int32_t> & prompt, int kv_offset,
                                              int32_t im_start, int32_t user, int chunk_tokens,
                                              int n_pool_blocks, int cut_u, int n_queries, bool profile_on,
                                              bool pflash2 = false) {
    KvFlashSlotTurn t;
    t.kv_offset = std::clamp(kv_offset, 0, (int)prompt.size());
    t.end = (int)prompt.size();
    if (!profile_on || chunk_tokens <= 0 || t.end <= t.kv_offset) return t;
    const std::vector<int32_t> suffix(prompt.begin() + t.kv_offset, prompt.end());
    const KvFlashMode4Plan p4 = kvflash_mode4_plan(suffix, t.kv_offset, im_start, user);
    if (!p4.run) return t;
    t.r = p4.r;
    t.reserve_chunks = (t.end - t.r + chunk_tokens - 1) / chunk_tokens + 1;
    // fits, or no sealed chunk before r to score: LRU turn
    if (kvflash_pool_fits(t.end, chunk_tokens, t.reserve_chunks, n_pool_blocks) || t.r / chunk_tokens == 0) { t.r = -1; t.reserve_chunks = 0; return t; }
    if (pflash2) { t.apply_at = t.r; return t; }
    t.phase = KvFlashTurnPhase::FirstPass;
    t.cuts = kvflash_slot_turn_cuts(t.kv_offset, t.r, t.end, cut_u, n_queries);
    return t;
}


// Tokens of the next cut from cur_pos: the planned cut in the first pass (its
// last cut carries the captured queries), otherwise a plain U cut (the next
// multiple of U or the prompt end, a function of position only) that stops at
// the mode-5 apply point. 0 at or past the end.
inline int kvflash_slot_turn_next_cut(const KvFlashSlotTurn & t, int cur_pos, int cut_u) {
    if (cur_pos >= t.end) return 0;
    if (t.first_pass())
        for (int c : t.cuts) if (c > cur_pos) return c - cur_pos;
    int n = std::min(t.end, (cur_pos / cut_u + 1) * cut_u) - cur_pos;
    if (t.apply_at > cur_pos) n = std::min(n, t.apply_at - cur_pos);
    return n;
}

// Mode 5: the prefill reached the point where the drafter's picks are applied (the engine then clears apply_at).
inline bool kvflash_slot_turn_apply_due(const KvFlashSlotTurn & t, int cur_pos) {
    return t.apply_at >= 0 && cur_pos >= t.apply_at;
}

// What the step that runs the cut [cur_pos, cur_pos + n) must do for this slot.
struct KvFlashTurnStep {
    bool snapshot = false;      // after the graph: capture the slot's recurrent state (it is the R state)
    bool capture = false;       // build the graph with q capture ending at this slot's segment (the first pass's last cut)
    int  q_rows = 0;            // first pass: how many of this cut's last rows are among the last n_queries prompt tokens (to capture)
};
inline KvFlashTurnStep kvflash_slot_turn_step(const KvFlashSlotTurn & t, int cur_pos, int n, int n_queries = 0) {
    KvFlashTurnStep s;
    if (n <= 0 || !t.first_pass()) return s;
    s.snapshot = t.snapshot_at_r() && cur_pos + n == t.r;
    s.capture = cur_pos + n == t.end;
    s.q_rows = n_queries > 0 ? std::clamp(cur_pos + n - (t.end - n_queries), 0, n) : 0;
    return s;
}

// The serial engine's selector-v2 queries: the last `keep` prompt queries over all the first pass's batches,
// oldest first (qwen35_backend.cpp kvflash_q_tail_). Append one cut's rows (n_rows of row_size floats) and trim.
inline void kvflash_q_tail_push(std::vector<float> & tail, const float * rows, int n_rows, size_t row_size, int keep) {
    tail.insert(tail.end(), rows, rows + (size_t)n_rows * row_size);
    const size_t cap = (size_t)std::max(keep, 0) * row_size;
    if (tail.size() > cap) tail.erase(tail.begin(), tail.end() - cap);
}

}  // namespace luce::common
