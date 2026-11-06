// Test-only views of production types: what the unit suite reads and no engine path does.
#pragma once
#include "../../src/common/kvflash_slot_plan.h"
#include "../../src/common/kvflash_slot_turn.h"

namespace luce::common {

// First row of the scratch block (owned by no slot).
inline int64_t kvflash_scratch_row(int n_slots, int quota_chunks, int chunk_tokens = kKvFlashChunkTokens) {
    return (int64_t)n_slots * quota_chunks * chunk_tokens;
}

// The cut ends from `start` to `end`: kvflash_next_cut applied until the turn is covered.
inline std::vector<int> kvflash_cut_sequence(int start, int end, int cut_u) {
    std::vector<int> out;
    for (int pos = start; pos < end;) { pos = kvflash_next_cut(pos, end, cut_u); out.push_back(pos); }
    return out;
}

// Where a sticky session checkpoints this turn: a recall turn keeps its R snapshot (the
// first-pass state at R, what a cold turn has there: the replay's state, shaped by the picks,
// drifted the next turn's queries and cost recall), an LRU turn the state at the assistant
// header from its only pass. -1: nothing to keep from this turn.
inline int kvflash_sticky_ckpt_pos(const KvFlashSlotTurn & t) { return t.recall() ? t.r : t.ckpt; }

// The most slabs held at once while the ops run. `held`: bit i of 0..2 = rung i holds a slab, bit 3 = the live slab is in hand.
// live_after_frees: the live slab is not taken yet; it is snapshotted after the frees (take_ckpt_), so it counts from there.
inline int kvflash_ladder_peak(int held, const std::vector<KvFlashSlabOp> & ops, bool live_after_frees) {
    int n = 0; for (int i = 0; i < 4; ++i) n += (held >> i) & 1;
    int peak = n; bool pending = live_after_frees;
    for (const KvFlashSlabOp & op : ops) {
        if (op.kind != KvFlashSlabOp::Free && pending) { peak = std::max(peak, ++n); pending = false; }
        if (op.kind == KvFlashSlabOp::Free && ((held >> op.from) & 1)) { --n; held &= ~(1 << op.from); }
        if (op.kind == KvFlashSlabOp::Copy) peak = std::max(peak, ++n);
    }
    return peak;
}

// Chunks a slot's next step allocates: what [cur_pos, cur_pos + n_next) needs past the chunks the pager already holds.
inline int kvflash_slot_need_chunks(int cur_pos, int n_next, int chunk_tokens, int n_chunks) {
    if (chunk_tokens <= 0 || n_next <= 0) return 0;
    return std::max(0, (cur_pos + n_next + chunk_tokens - 1) / chunk_tokens - n_chunks);
}

struct KvFlashTestLedger : KvFlashHostLedger {
    using KvFlashHostLedger::KvFlashHostLedger;
    int reservation_count() const { return (int)reserved_.size(); }
};

}  // namespace luce::common
