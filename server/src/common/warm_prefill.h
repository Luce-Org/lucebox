#pragma once

#include "restore_delta.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

// Result of planning a warm-prefill step.
struct WarmPrefillPlan {
    std::vector<int32_t> delta;           // suffix to actually prefill
    std::vector<int32_t> drafter_history; // full reconstructed history for the scorer
};

// Plan a warm-prefill for a single request.
//
// full_prompt       — the complete token sequence for this request
// cached_prefix_len — how many leading tokens are already in the KV cache
//                     (0 = cold start; full_prompt.size() = full hit)
//
// Throws std::invalid_argument if cached_prefix_len < 0.
// Throws std::out_of_range    if cached_prefix_len > full_prompt.size().
//
// drafter_history is always == full_prompt (prefix + delta), so the scorer
// sees real token IDs for the entire context rather than zero-padding.
inline WarmPrefillPlan plan_warm_prefill(const std::vector<int32_t> & full_prompt,
                                         int cached_prefix_len) {
    WarmPrefillPlan plan;
    plan.delta = restore_prompt_delta(full_prompt, cached_prefix_len); // throws on bad input
    // drafter_history = full_prompt[0:cached_prefix_len] + delta == full_prompt
    plan.drafter_history = full_prompt;
    return plan;
}

}  // namespace dflash::common
