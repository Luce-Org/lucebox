// Planner-to-storage bridge for ordered multi-target MoE placements.
//
// The planner/runtime layers decide which target owns each routed expert. This
// materialization step derives the primary local placement plus the ordered
// union of non-primary experts so backend-local storage can preserve the same
// target order before handing work to remote IPC targets or CPU fallback.

#pragma once

#include "expert_split_runtime.h"

#include <string>
#include <vector>

namespace dflash::common {

struct ExpertSplitMaterializationTarget {
    int target_index = -1;
    ExpertSplitTarget target;
    MoeHybridPlacement placement;
};

struct ExpertSplitMaterialization {
    int n_layer = 0;
    int n_expert = 0;
    int n_expert_used = 0;
    MoeHybridPlacement primary_placement;
    std::vector<std::vector<int32_t>> cold_expert_ids_by_layer;
    std::vector<ExpertSplitMaterializationTarget> targets;
    bool ordered_cold_union = false;

    bool matches(int n_layer_, int n_expert_, int n_expert_used_) const;
};

bool build_expert_split_materialization(const ExpertSplitRuntime & runtime,
                                        int n_expert_used,
                                        ExpertSplitMaterialization & out,
                                        std::string * err = nullptr);

}  // namespace dflash::common
