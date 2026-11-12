// What the seq engine decides on its own for KVFlash slots,
// kept pure so the unit suite covers it without a device.

#pragma once

#include "concurrency/seq_engine.h"
#include "kvflash_slot_plan.h"

#include <algorithm>

namespace luce::common {

// D1: a KVFlash slot prefills in cuts of U logical positions (8 chunks), never
// shorter because of peers.
constexpr int kKvFlashCutU = 512;

// The scheduler grants a KVFlash slice whole cuts only: each slice gets U (the
// allocation quantum), and the step's total is a whole number of them.
inline StepPlanLimits kvflash_step_limits(StepPlanLimits l, int cut_u = kKvFlashCutU) {
    l.max_prefill_tokens_per_sequence = std::min(l.max_prefill_tokens_per_sequence, cut_u);
    l.prefill_allocation_quantum = cut_u;
    l.max_prefill_tokens_total = std::max(cut_u, l.max_prefill_tokens_total / cut_u * cut_u);
    return l;
}

// Entries of a slot's block-table column that cover kv_len tokens: 4 per
// 64-token chunk at 16-token blocks, clamped to the table height.
inline int kvflash_column_entries(int kv_len, int table_height, int chunk_tokens = 64, int block_tokens = 16) {
    const int chunks = (kv_len + chunk_tokens - 1) / chunk_tokens;
    return std::min(table_height, chunks * kvflash_table_entries(chunk_tokens, block_tokens));
}

}  // namespace luce::common
