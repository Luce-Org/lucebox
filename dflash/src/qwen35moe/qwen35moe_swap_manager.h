// Request-boundary swap planning for qwen35moe expert placement.

#pragma once

#include "qwen35moe_expert_placement.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct Qwen35MoeSwapAction {
    int layer_idx = -1;
    int evict_expert = -1;
    int promote_expert = -1;
    uint64_t evict_count = 0;
    uint64_t promote_count = 0;
};

struct Qwen35MoeSwapPlan {
    Qwen35MoeExpertPlacement next_placement;
    std::vector<Qwen35MoeSwapAction> actions;
};

struct Qwen35MoeSwapPolicy {
    int max_swaps_total = 0;          // 0 = no swaps
    uint64_t min_promote_gain = 1;    // promoted expert count must exceed evicted by at least this amount
};

bool build_qwen35moe_swap_plan(const Qwen35MoeExpertPlacement & current,
                               const Qwen35MoeRoutingStats & stats,
                               const Qwen35MoeSwapPolicy & policy,
                               Qwen35MoeSwapPlan & out,
                               std::string * err = nullptr);

}  // namespace dflash::common
