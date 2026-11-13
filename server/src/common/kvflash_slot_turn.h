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
#include "kvflash_slot_ladder.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace luce::common {

enum class KvFlashTurnPhase : uint8_t { Lru, FirstPass, Replay };

struct KvFlashSlotTurn {
    KvFlashTurnPhase phase = KvFlashTurnPhase::Lru;
    int  kv_offset = 0;             // resume point: tokens [0, kv_offset) are the session
    int  r = -1;                    // rewind point
    int  end = 0;                   // prompt end
    int  reserve_chunks = 0;        // pool room kept for the replay (single-slot rule)
    int  ckpt = -1;                 // the prompt-end checkpoint: the last <|im_start|> (the assistant header)
    bool end_rung = false;          // C1.5 F7: a recall turn whose replay stops at ckpt and checkpoints there too
    int  apply_at = -1;             // mode 5: the cut at which the drafter's picks are applied (one pass, no snapshot at R); -1: none
    int  uq_cut = -1;               // the latest human question's end inside this prefill: a cut end of every pass (kvflash_uq_cut); -1: none
    int  capture_at = -1;           // the cut at which the head is captured (a multiple of k chunks); -1: none
    std::vector<int> cuts;          // first-pass cut ends, ascending
    bool recall() const { return phase != KvFlashTurnPhase::Lru; }
    bool first_pass() const { return phase == KvFlashTurnPhase::FirstPass; }
    bool snapshot_at_r() const { return recall() && r > kv_offset; }
    // The pass that checkpoints at ckpt: an LRU turn's only pass, a recall turn's replay with the end rung.
    bool ckpt_pass() const { return !recall() || (end_rung && phase == KvFlashTurnPhase::Replay); }
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

// Blocks a live slot keeps resident: settle() holds it at the pool minus the step headroom.
inline int kvflash_usable_blocks(int n_pool_blocks, int headroom_blocks) { return std::max(0, n_pool_blocks - headroom_blocks); }

// The slot's turn for a prompt resumed at kv_offset. Mode 4 runs when the recall
// profile is on and the prompt with the replay's reserve does not fit the slot's
// pool (kvflash_pool_fits, the single-slot rule); otherwise the slot stays LRU.
// end_rung_min (LUCE_KVFLASH_END_RUNG_MIN): a recall turn also checkpoints at the
// header when end - r reaches it; < 0: never (the R rung alone, C1.4).
// pflash2 (mode 5): the picks come from the drafter's scores of the prompt ids, so the turn stays one
// pass (the LRU turn's checkpoint rule) and only records where to apply them (apply_at = R).
// capture_at: a cut the turn must stop at so the engine can capture the pinned head there; honoured inside (kv_offset, end)
// and, in a recall turn, only before R (the recall is planned around it).
inline KvFlashSlotTurn kvflash_slot_turn_plan(const std::vector<int32_t> & prompt, int kv_offset,
                                              int32_t im_start, int32_t user, int chunk_tokens,
                                              int n_pool_blocks, int cut_u, int n_queries, bool profile_on,
                                              int end_rung_min = -1, bool pflash2 = false, int max_turn_tokens = 0,
                                              int capture_at = -1) {
    KvFlashSlotTurn t;
    t.kv_offset = std::clamp(kv_offset, 0, (int)prompt.size());
    t.end = (int)prompt.size();
    if (capture_at > t.kv_offset && capture_at < t.end) t.capture_at = capture_at;
    // The next prompt re-renders everything before the generation prompt's assistant header
    // unchanged (the reply after it is re-tokenized from text): the sticky checkpoint is taken there.
    for (int i = t.end - 1; i > t.kv_offset; --i) if (prompt[(size_t)i] == im_start) { t.ckpt = i; break; }
    if (!profile_on || chunk_tokens <= 0 || t.end <= t.kv_offset) return t;
    const std::vector<int32_t> suffix(prompt.begin() + t.kv_offset, prompt.end());
    const KvFlashMode4Plan p4 = kvflash_mode4_plan(suffix, t.kv_offset, im_start, user, max_turn_tokens);
    if (!p4.run) return t;
    t.r = p4.r;
    t.reserve_chunks = (t.end - t.r + chunk_tokens - 1) / chunk_tokens + 1;
    // fits, or no sealed chunk before r to score: LRU turn
    if (kvflash_pool_fits(t.end, chunk_tokens, t.reserve_chunks, n_pool_blocks) || t.r / chunk_tokens == 0) { t.r = -1; t.reserve_chunks = 0; return t; }
    if (t.capture_at > t.r) t.capture_at = -1;   // a recall turn: only a cut before R
    if (pflash2) { t.apply_at = t.r; return t; }
    t.phase = KvFlashTurnPhase::FirstPass;
    t.cuts = kvflash_slot_turn_cuts(t.kv_offset, t.r, t.end, cut_u, n_queries);
    if (t.capture_at > 0 && std::find(t.cuts.begin(), t.cuts.end(), t.capture_at) == t.cuts.end())
        t.cuts.insert(std::upper_bound(t.cuts.begin(), t.cuts.end(), t.capture_at), t.capture_at);
    t.end_rung = end_rung_min >= 0 && t.ckpt > t.r && t.end - t.r >= end_rung_min;
    return t;
}

// Tokens of the next cut from cur_pos: the planned cut in the first pass (its
// last cut carries the captured queries), a plain U cut in the replay, a plain U
// cut stopping at the checkpoint in the pass that takes it. 0 at or past the end.
// tail (G-D): the cut that would end the turn stops `tail` tokens short, so the last prompt tokens are a
// cut of their own and the engine can stash the state before them (the first pass keeps its capture cut
// whole: a replay or an LRU pass always ends the turn). 0: off.
inline int kvflash_slot_turn_next_cut(const KvFlashSlotTurn & t, int cur_pos, int cut_u, int tail = 0) {
    if (cur_pos >= t.end) return 0;
    if (t.first_pass())
        for (int c : t.cuts) if (c > cur_pos) return c - cur_pos;
    int n = kvflash_next_cut(cur_pos, t.end, cut_u) - cur_pos;
    if (t.apply_at > cur_pos) n = std::min(n, t.apply_at - cur_pos);
    if (t.uq_cut > cur_pos) n = std::min(n, t.uq_cut - cur_pos);
    if (t.capture_at > cur_pos) n = std::min(n, t.capture_at - cur_pos);
    if (t.ckpt_pass() && t.ckpt > cur_pos) n = std::min(n, t.ckpt - cur_pos);
    if (tail > 0 && n > tail && cur_pos + n == t.end) n -= tail;
    return n;
}

// LUCE_KVFLASH_UQ_QUERIES: the question's end `cut` (kvflash_uq_cut, -1: none) becomes a cut end of every pass
// of the turn, so the question's last queries are the last rows of the cuts that hold them.
inline void kvflash_slot_turn_force_cut(KvFlashSlotTurn & t, int cut) {
    t.uq_cut = cut;
    const auto at = std::lower_bound(t.cuts.begin(), t.cuts.end(), cut);
    if (cut >= 0 && t.first_pass() && (at == t.cuts.end() || *at != cut)) t.cuts.insert(at, cut);
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
    bool ckpt = false;          // after the graph: the slot's recurrent state is the sticky checkpoint at the header
    bool preamble = false;      // after the graph: the slot holds exactly the head (the capture cut)
};
inline KvFlashTurnStep kvflash_slot_turn_step(const KvFlashSlotTurn & t, int cur_pos, int n, int n_queries = 0) {
    KvFlashTurnStep s;
    if (n <= 0) return s;
    s.ckpt = t.ckpt_pass() && t.ckpt > 0 && cur_pos + n == t.ckpt;
    s.preamble = t.capture_at > 0 && cur_pos + n == t.capture_at;
    if (!t.first_pass()) return s;
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

// Sticky sessions: a parked slot and the tokens it holds (history.size() == its cur_pos).
struct KvFlashParkedSession {
    int slot = -1;
    const std::vector<int32_t> * history = nullptr;
    uint64_t tick = 0;          // park order; newest wins a tie
    int64_t bytes = 0;          // what its eviction frees: the slot's slice plus its host reservation (0: unknown, one unit)
    double stamp = 0;           // the eviction clock when it parked
};
struct KvFlashSessionPick {
    int slot = -1;
    int lcp = 0;
    bool exact() const { return slot >= 0; }
};

// The parked session with the longest common prefix with `prompt` (at least
// min_lcp tokens, ties to the newest); slot -1 when none qualifies.
inline KvFlashSessionPick kvflash_pick_session(const std::vector<KvFlashParkedSession> & parked,
                                               const std::vector<int32_t> & prompt, int min_lcp) {
    KvFlashSessionPick best;
    uint64_t best_tick = 0;
    for (const KvFlashParkedSession & s : parked) {
        if (s.slot < 0 || !s.history) continue;
        const size_t cap = std::min(prompt.size(), s.history->size());
        const size_t l = (size_t)(std::mismatch(prompt.begin(), prompt.begin() + (ptrdiff_t)cap, s.history->begin()).first - prompt.begin());
        if ((int)l < std::max(min_lcp, 1)) continue;
        if ((int)l > best.lcp || ((int)l == best.lcp && s.tick > best_tick)) { best = {s.slot, (int)l}; best_tick = s.tick; }
    }
    return best;
}

// C1.5 F1: a session keeps two rungs, `ckpt` (its prompt-end checkpoint: the state before the
// reply, which the client re-renders and may re-tokenize differently) and `trunk` below it (the
// checkpoint before that: where a branch forked). Either may be absent (<= 0).
struct KvFlashRungs { int ckpt = -1, trunk = -1; };

// Rule B. Where a parked session of `len` tokens resumes for a prompt sharing `lcp` tokens with
// its history: the highest of len, ckpt and trunk the prompt covers; -1 when none (cold). A
// resume always leaves new tokens to prefill: the position must be below prompt_len.
inline int kvflash_resume_point(int lcp, int len, int ckpt, int trunk, int prompt_len, int extra = -1) {
    int best = -1;
    for (int p : {len, ckpt, trunk, extra}) if (p > 0 && lcp >= p && p < prompt_len) best = std::max(best, p);
    return best;
}

// G-D: where a session of `len` fed tokens parks. The state after the last fed token has no
// logits for a next token and a hybrid model cannot step back from it, so a prompt equal to the
// history could only resume at a rung. A slot that kept an earlier state (stash_pos: the token count
// it covers, at most `tail` tokens back) parks there, short of len: the history is stash_pos tokens
// and an identical prompt resumes at stash_pos and prefills the rest (len - stash_pos tokens, the
// reselect's queries). Never below a rung (the rung would be orphaned) and never at 0.
inline int kvflash_park_point(int len, int stash_pos, int ckpt, int trunk, int tail = 1) {
    return stash_pos > 0 && stash_pos < len && stash_pos >= len - tail && stash_pos >= ckpt && stash_pos >= trunk ? stash_pos : len;
}

// The park-short switches (env strings, null when unset). LUCE_KVFLASH_PARK_SHORT=0: everything off. The prefill-only
// tail stash (one copy per turn) is on by default with tail 4 (R's replay window); LUCE_KVFLASH_PARK_DECODE=1 adds the
// per-step decode stash (a copy before every decode step, ~4-5% decode), off by default.
struct KvFlashParkCfg { bool tail_on = false; bool decode_on = false; int tail = 4; bool any() const { return tail_on || decode_on; } };
inline KvFlashParkCfg kvflash_park_cfg(bool sticky, const char * park_short, const char * tail, const char * decode) {
    KvFlashParkCfg c;
    const bool off = park_short && park_short[0] == '0' && !park_short[1];
    c.tail_on = sticky && !off;
    c.decode_on = c.tail_on && decode && decode[0] == '1' && !decode[1];
    if (tail && *tail) { const int n = std::atoi(tail); c.tail = n < 1 ? 1 : n > 64 ? 64 : n; }
    return c;
}

// Rule B's bookkeeping. The rungs a session keeps when it resumes at L: the rung at L (or the
// highest below it) is the fallback `ckpt`, a rung below that the `trunk`; rungs above L are gone
// with their rows. At L == len both stay.
inline KvFlashRungs kvflash_rungs_after_resume(int ckpt, int trunk, int L) {
    KvFlashRungs r;
    if (ckpt > 0 && ckpt <= L) { r.ckpt = ckpt; if (trunk > 0 && trunk < ckpt) r.trunk = trunk; }
    else if (trunk > 0 && trunk <= L) r.ckpt = trunk;
    return r;
}

// The rungs of a parked session rewound to L (slab restored, pager and history cut to L): the ones above L are gone, a trunk at or below
// L becomes the checkpoint when the checkpoint is above it. Applied at the rewind, so a session left parked (a busy refusal) never
// holds a rung past its history, which the bundle head refuses.
inline KvFlashRungs kvflash_rungs_after_rewind(int ckpt, int trunk, int L) {
    return kvflash_rungs_after_resume(ckpt, trunk, L);
}


// A session keeps a third rung beside ckpt and trunk. A rewind to a rung
// below the checkpoint (a branch: the client re-rendered an earlier prompt) pins that rung in the third slot; with no
// branch the slot holds the rung the ladder demoted, so it never costs a rung the two-rung ladder had.
struct KvFlashLadder { int ckpt = -1, trunk = -1, extra = -1; bool pinned = false; };
// Which slab backs each rung of a new ladder: 0 ckpt, 1 trunk, 2 extra of the ladder before, 3 the live slot state, -1 none.
// Two rungs naming one source are two copies (a branch event: the pinned rung is also the checkpoint until the next ckpt).
enum : int { kRungNone = -1, kRungCkpt = 0, kRungTrunk = 1, kRungExtra = 2, kRungLive = 3 };
struct KvFlashLadderMove { KvFlashLadder to; int ckpt_from = -1, trunk_from = -1, extra_from = -1; };

namespace detail {
struct LadderRung { int pos, from; };
// The rungs in descending position, one per position (the first listed wins a tie).
inline std::vector<LadderRung> ladder_sorted(std::vector<LadderRung> v) {
    std::stable_sort(v.begin(), v.end(), [](const LadderRung & a, const LadderRung & b) { return a.pos > b.pos; });
    v.erase(std::unique(v.begin(), v.end(), [](const LadderRung & a, const LadderRung & b) { return a.pos == b.pos; }), v.end());
    return v;
}
}  // namespace detail

// A resume at L of a session holding `len` tokens (L == len: nothing is rewound). Rungs above L go with their rows. A rewind
// below the checkpoint is the branch event: the rung at L is pinned in the third slot (the old third goes). The two highest
// rungs at or below L are the new checkpoint and trunk; the pinned rung is also the checkpoint until the next checkpoint.
inline KvFlashLadderMove kvflash_ladder_resume(const KvFlashLadder & in, int L, int len) {
    int extra = in.extra, extra_from = kRungExtra;
    bool pinned = in.pinned;
    if (L < len && extra > L) { extra = -1; pinned = false; }
    if (in.ckpt > 0 && L < in.ckpt && L < len) {
        const int src = in.extra == L ? kRungExtra : in.trunk == L ? kRungTrunk : kRungNone;
        if (src != kRungNone) { extra = L; extra_from = src; pinned = true; }
    }
    // the two-rung ladder as before; only when the rung at L lives in the third slot alone does it become the checkpoint,
    // with the highest rung below it as the trunk
    const KvFlashRungs r = kvflash_rungs_after_resume(in.ckpt, in.trunk, L);
    KvFlashLadderMove m;
    m.to.ckpt = r.ckpt; m.ckpt_from = r.ckpt > 0 ? (r.ckpt == in.ckpt ? kRungCkpt : kRungTrunk) : kRungNone;
    m.to.trunk = r.trunk; m.trunk_from = r.trunk > 0 ? kRungTrunk : kRungNone;
    if (L < len && in.extra == L && r.ckpt != L) {
        m.to.ckpt = L; m.ckpt_from = kRungExtra; m.to.trunk = -1; m.trunk_from = kRungNone;
        for (const detail::LadderRung c : {detail::LadderRung{in.ckpt, kRungCkpt}, {in.trunk, kRungTrunk}})
            if (c.pos > 0 && c.pos < L && c.pos > m.to.trunk) { m.to.trunk = c.pos; m.trunk_from = c.from; }
    }
    if (extra > 0) { m.to.extra = extra; m.extra_from = extra_from; m.to.pinned = pinned; }
    return m;
}

// How a ladder move is carried out on the slabs, in order: the held sources no rung of the new ladder names are freed first (so a
// branch copy never sits beside slabs it makes dead), then each destination takes its source; a source two rungs name is copied
// for the earlier ones and moved into the last. `to`: 0 ckpt, 1 trunk, 2 extra. Free has no `to`.
struct KvFlashSlabOp { enum Kind { Free, Copy, Move } kind; int from; int to; };
inline std::vector<KvFlashSlabOp> kvflash_ladder_ops(const KvFlashLadderMove & m) {
    const int from[3] = {m.ckpt_from, m.trunk_from, m.extra_from};
    const int pos[3] = {m.to.ckpt, m.to.trunk, m.to.extra};
    bool used[4] = {false, false, false, false};
    for (int i = 0; i < 3; ++i) if (pos[i] > 0 && from[i] >= 0 && from[i] <= 3) used[from[i]] = true;
    std::vector<KvFlashSlabOp> ops;
    for (int s = 0; s < 3; ++s) if (!used[s]) ops.push_back({KvFlashSlabOp::Free, s, -1});
    for (int i = 0; i < 3; ++i) {
        if (pos[i] <= 0 || from[i] < 0 || from[i] > 3) continue;
        bool last = true;
        for (int j = i + 1; j < 3; ++j) last &= !(pos[j] > 0 && from[j] == from[i]);
        ops.push_back({last ? KvFlashSlabOp::Move : KvFlashSlabOp::Copy, from[i], i});
    }
    return ops;
}
// Rule A: a checkpoint at p (the live state). The rungs below p, the pinned position aside, are the trunk and, unless the
// third slot is pinned, the third; a pinned third stays.
inline KvFlashLadderMove kvflash_ladder_ckpt(const KvFlashLadder & in, int p) {
    std::vector<detail::LadderRung> c;
    for (const detail::LadderRung r : {detail::LadderRung{in.ckpt, kRungCkpt}, {in.trunk, kRungTrunk}, {in.extra, kRungExtra}})
        if (r.pos > 0 && r.pos < p && !(in.pinned && r.pos == in.extra)) c.push_back(r);
    c = detail::ladder_sorted(c);
    KvFlashLadderMove m;
    m.to.ckpt = p; m.ckpt_from = kRungLive;
    if (!c.empty()) { m.to.trunk = c[0].pos; m.trunk_from = c[0].from; }
    if (in.pinned) { m.to.extra = in.extra; m.extra_from = kRungExtra; m.to.pinned = true; }
    else if (c.size() > 1) { m.to.extra = c[1].pos; m.extra_from = c[1].from; }
    return m;
}

// What a save writes of the third rung: its slab (-1: none; a third at a rung's position holds the same state, so no second file) and the pin.
struct KvFlashHeadRungs { int extra = -1; int pin = 0; };
inline KvFlashHeadRungs kvflash_ladder_to_head(const KvFlashLadder & in) {
    const bool dup = in.extra > 0 && (in.extra == in.ckpt || in.extra == in.trunk);
    return {in.extra > 0 && !dup ? in.extra : -1, in.pinned && in.extra > 0 ? in.extra : 0};
}
// The ladder a head restores to, as a move over the loaded slabs (0 ckpt, 1 trunk, 2 third): the pin comes back with the third, or as a copy of
// the ckpt or trunk slab it names.
inline KvFlashLadderMove kvflash_ladder_restored(int ckpt, int trunk, int extra, int pin) {
    KvFlashLadderMove m;
    m.to.ckpt = ckpt; m.ckpt_from = ckpt > 0 ? kRungCkpt : kRungNone;
    m.to.trunk = trunk; m.trunk_from = trunk > 0 ? kRungTrunk : kRungNone;
    if (extra > 0) { m.to.extra = extra; m.extra_from = kRungExtra; m.to.pinned = pin == extra; }
    else if (pin > 0 && (pin == ckpt || pin == trunk)) { m.to.extra = pin; m.extra_from = pin == ckpt ? kRungCkpt : kRungTrunk; m.to.pinned = true; }
    return m;
}

// The victim among parked sessions (GreedyDual-Size, as FlashStates eq. 7): priority = the eviction clock when it parked +
// its restore cost per byte freed. The lowest goes and the caller raises its clock to the victim's priority, so a session
// nobody resumes ages out. The restore cost is the re-prefill of its history, linear in tokens under a fixed pool, so tokens
// stand for seconds and no weight is tuned; a slot's slice is the same for every session, so a long history outranks a short one.
// ponytail: the cost is re-prefill tokens only; with the disk tier it is min(re-prefill, bundle bytes / link rate).
inline double kvflash_session_priority(const KvFlashParkedSession & s) {
    return s.stamp + (double)(s.history ? s.history->size() : 0) / (double)std::max<int64_t>(s.bytes, 1);
}
// The parked session to drop when room is needed (ties to the oldest); -1 when none.
inline int kvflash_cheapest_session(const std::vector<KvFlashParkedSession> & parked) {
    const KvFlashParkedSession * v = nullptr;
    for (const KvFlashParkedSession & s : parked) {
        if (s.slot < 0) continue;
        const double h = kvflash_session_priority(s);
        if (!v || h < kvflash_session_priority(*v) || (h == kvflash_session_priority(*v) && s.tick < v->tick)) v = &s;
    }
    return v ? v->slot : -1;
}

// Frees the victim among the parked sessions other than `keep`: `to_disk(slot)` is the session's move to disk (false: its
// park was refused, it stays), `drop(slot)` then releases it (its next turn prefills cold or from its last bundle). A refused
// park is dropped, not skipped: the parked set is a cache and never blocks a request. -1 only when nothing is parked.
struct KvFlashEvicted { int slot = -1; bool dropped = false; double priority = 0; };
template <class ToDisk, class Drop>
inline KvFlashEvicted kvflash_evict_cheapest(const std::vector<KvFlashParkedSession> & parked, ToDisk to_disk, Drop drop, int keep = -1) {
    std::vector<KvFlashParkedSession> list;
    for (const KvFlashParkedSession & s : parked) if (s.slot != keep) list.push_back(s);
    KvFlashEvicted r;
    r.slot = kvflash_cheapest_session(list);
    if (r.slot < 0) return r;
    for (const KvFlashParkedSession & s : list) if (s.slot == r.slot) r.priority = kvflash_session_priority(s);
    if (!to_disk(r.slot)) { drop(r.slot); r.dropped = true; }
    return r;
}

// LUCE_KVFLASH_SLOT_STICKY=0 turns sticky sessions off. Read here only: the host ledger's per-slot charge and the engine must agree.
inline bool kvflash_slot_sticky_env() {
    const char * e = std::getenv("LUCE_KVFLASH_SLOT_STICKY");
    return !(e && e[0] == '0' && e[1] == '\0');
}

// The ONE early refusal of a sticky admit, asked before the restore gate, the archive of the sid's old instance, any victim eviction
// and any rewind, so a refused request leaves every session as it was. A prompt past max_ctx is refused first (admit_at would say the
// same, but only after a rewind); the host probe is asked only for a prompt that fits: the ledger clamps a total to max_ctx.
enum class KvFlashEarlyRefusal { none, prompt_over_max_ctx, host_budget };
template <class ExceedsHostBudget>
inline KvFlashEarlyRefusal kvflash_early_refusal(int64_t prompt_tokens, int64_t max_ctx, ExceedsHostBudget exceeds_host_budget) {
    if (prompt_tokens > max_ctx) return KvFlashEarlyRefusal::prompt_over_max_ctx;
    return exceeds_host_budget() ? KvFlashEarlyRefusal::host_budget : KvFlashEarlyRefusal::none;
}

// A resume the pool refused (admit_at on the parked slot): what the engine does with the parked session.
//  evict_other_retry - busy and another session is parked: it leaves for disk and the resume is asked again
//  wait_parked       - busy and nothing else can leave: the session stays parked as the ledger left it, the request waits (Deferred)
//  drop_cold         - the pool cannot ever take the resume (not busy): the session goes, the request prefills cold
//  refuse            - capacity_exceeded: the request alone exceeds a fixed bound, a cold admit needs the same bytes; the session
//                      is neither evicted (never for room it is itself asking for) nor dropped, and the request is refused
enum class KvFlashResumeRefused { evict_other_retry, wait_parked, drop_cold, refuse };
inline KvFlashResumeRefused kvflash_resume_refused(bool busy, bool other_parked, bool exceeds_capacity = false) {
    if (exceeds_capacity) return KvFlashResumeRefused::refuse;
    if (!busy) return KvFlashResumeRefused::drop_cold;
    return other_parked ? KvFlashResumeRefused::evict_other_retry : KvFlashResumeRefused::wait_parked;
}

// Restoring a session from disk reserves host bytes like any admit. Busy (the ledger lacks room) evicts the cheapest
// parked session and tries again; Refused (the bundle or the pager cannot take it) is final. Returns Busy only
// when `evict()` has nothing left to free.
enum class KvFlashAdopt { Ok, Busy, Refused };
// `release()` runs before every eviction: the returning bundle's memory is freed (the retry re-reads it) so no allocation the
// ledger has not charged is live while a victim is staged.
template <class TryAdopt, class Release, class Evict>
inline KvFlashAdopt kvflash_adopt_with_evict(TryAdopt try_adopt, Release release, Evict evict) {
    for (;;) {
        const KvFlashAdopt r = try_adopt();
        if (r != KvFlashAdopt::Busy) return r;
        release();
        if (!evict()) return r;
    }
}
template <class TryAdopt, class Evict>
inline KvFlashAdopt kvflash_adopt_with_evict(TryAdopt try_adopt, Evict evict) {
    return kvflash_adopt_with_evict(try_adopt, [] {}, evict);
}

// A returning session's restore from disk: `probe()` answers from the ledger alone whether the host reservation fits
// (Busy = it does not; anything else = cannot decide or fits), `has_victim()` whether a parked session could leave,
// `validate()` streams the bundle once without holding it (false: unreadable), `load()` reads the bundle (false: unreadable), `try_adopt`
// reserves and installs it, `release` frees the loaded bundle before a victim is staged, `evict_one` frees the cheapest
// parked session (false: nothing left). `load_failed` says the bundle could not be read (first load or the reload).
struct KvFlashRestoreResult { KvFlashAdopt adopt = KvFlashAdopt::Refused; bool load_failed = false; };
template <class Probe, class HasVictim, class Validate, class Load, class TryAdopt, class Release, class EvictOne>
inline KvFlashRestoreResult kvflash_restore_adopt(Probe probe, HasVictim has_victim, Validate validate, Load load, TryAdopt try_adopt, Release release, EvictOne evict_one, bool evict_for_room) {
    // The ledger first: a reservation that does not fit is Busy without a read when nothing can leave. Otherwise the bundle is
    // validated ONCE before the first eviction (an unreadable one evicts nothing and returns its own error), then one victim
    // leaves at a time until the reservation fits.
    if (probe() == KvFlashAdopt::Busy) {
        if (!evict_for_room || !has_victim()) return {KvFlashAdopt::Busy, false};
        if (!validate()) return {KvFlashAdopt::Refused, true};
        while (probe() == KvFlashAdopt::Busy)
            if (!evict_one()) return {KvFlashAdopt::Busy, false};
    }
    if (!load()) return {KvFlashAdopt::Refused, true};
    bool reload_failed = false;
    const KvFlashAdopt got = kvflash_adopt_with_evict(try_adopt, release, [&] {
        if (!evict_for_room || !evict_one()) return false;
        if (!load()) { reload_failed = true; return false; }
        return true;
    });
    return {got, reload_failed};
}

}  // namespace luce::common
