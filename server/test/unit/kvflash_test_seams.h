// Test-only views of production types: what the unit suite reads and no engine path does.
#pragma once
#include "../../src/common/kvflash_slot_plan.h"
#include "../../src/common/kvflash_slot_turn.h"

namespace luce::common {

// First row of the scratch block (owned by no slot).
inline int64_t kvflash_scratch_row(int n_slots, int quota_chunks, int chunk_tokens = kKvFlashChunkTokens) {
    return (int64_t)n_slots * quota_chunks * chunk_tokens;
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
