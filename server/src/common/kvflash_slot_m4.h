// The slot engine's mode 4 on a turn that continues the same human question (the serial engine's
// kvflash_m4_reuse.h path, per slot): pure decisions. The turn runs once, picks applied at its resume point.

#pragma once

#include "kvflash_m4_reuse.h"
#include "kvflash_prefill_reselect.h"
#include "kvflash_slot_turn.h"

#include <utility>
#include <vector>

namespace luce::common {

// Full unless the knob is on, recall has blocks, the turn is a first-pass mode-4 turn and the recall gate at the
// resume point (kvflash_recall_gate with r = kv_offset) passes; then the serial decision.
inline KvFlashM4Path kvflash_slot_m4_path(bool single_on, int recall_blocks, bool first_pass, KvFlashRecallSkip gate_at_offset,
                                          const KvFlashM4Last & last, const KvFlashM4Ask & ask, const std::vector<int32_t> & ids) {
    if (!single_on || recall_blocks <= 0 || !first_pass || gate_at_offset != KvFlashRecallSkip::None) return KvFlashM4Path::Full;
    return kvflash_m4_path(last, ask, ids);
}

// A first-pass turn becomes the one pass of Reuse / Prescore: no snapshot at R, no rewind, the picks applied at the
// resume point, room for the one pass, and the tail queries kept for the next turn.
inline void kvflash_slot_turn_single(KvFlashSlotTurn & t, int chunk_tokens) {
    t.phase = KvFlashTurnPhase::Lru;
    t.r = t.kv_offset;
    t.apply_at = t.kv_offset;
    t.cuts.clear();
    t.header_rung = false;
    t.reserve_chunks = chunk_tokens > 0 ? (t.end - t.kv_offset + chunk_tokens - 1) / chunk_tokens + 1 : 0;
    t.track_tail = true;
}

// What a mode-4 request leaves for the next one of its slot; invalid without a question key.
inline KvFlashM4Last kvflash_slot_m4_record(const std::vector<int32_t> & ids, int uq_hi, uint64_t uq_hash,
                                            std::vector<float> scores, std::vector<float> q_tail) {
    KvFlashM4Last l;
    if (uq_hi <= 0) return l;
    l.uq_hi = uq_hi; l.uq_hash = uq_hash; l.ids = ids; l.scores = std::move(scores); l.q_tail = std::move(q_tail);
    l.valid = true;
    return l;
}

}  // namespace luce::common
