// Test-only views of production types: what the unit suite reads and no engine path does.
#pragma once
#include "../../src/common/kvflash_slot_plan.h"

namespace luce::common {

// First row of the scratch block (owned by no slot).
inline int64_t kvflash_scratch_row(int n_slots, int quota_chunks, int chunk_tokens = kKvFlashChunkTokens) {
    return (int64_t)n_slots * quota_chunks * chunk_tokens;
}

struct KvFlashTestLedger : KvFlashHostLedger {
    using KvFlashHostLedger::KvFlashHostLedger;
    int reservation_count() const { return (int)reserved_.size(); }
};

}  // namespace luce::common
