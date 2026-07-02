// Materialized per-target expert layout derived from ExpertSplitPlan.
//
// This is the bridge between model-agnostic placement planning and later
// backend-specific storage/compute runtimes. Target order stays identical to
// the planner output, and each target receives per-layer local expert indices.

#pragma once

#include "expert_split_plan.h"
#include "moe_hybrid_placement.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct ExpertSplitLayerTarget {
    int layer = -1;
    int target_index = -1;
    std::vector<int32_t> global_expert_ids;
    std::vector<int32_t> local_by_global;
    uint64_t total_bytes = 0;
    double total_score = 0.0;

    bool empty() const { return global_expert_ids.empty(); }
};

struct ExpertSplitRuntimeTarget {
    ExpertSplitTarget target;
    std::vector<ExpertSplitLayerTarget> layers;
    int total_experts = 0;
    uint64_t used_bytes = 0;
    double total_score = 0.0;
};

struct ExpertSplitTargetPlacement {
    int target_index = -1;
    MoeHybridPlacement placement;
};

struct ExpertSplitRuntime {
    int n_layer = 0;
    int n_expert = 0;
    std::vector<ExpertSplitRuntimeTarget> targets;
    std::vector<int32_t> target_index_by_global;
    std::vector<int32_t> local_index_by_global;

    bool matches(int n_layer_, int n_expert_) const;
    int index(int layer, int expert) const;
    int target_index(int layer, int expert) const;
    int local_index(int layer, int expert) const;
    const ExpertSplitRuntimeTarget * target_ptr(int target_index) const;
    const ExpertSplitLayerTarget * layer_target_ptr(int target_index, int layer) const;
};

bool build_expert_split_runtime(const ExpertSplitPlan & plan,
                                ExpertSplitRuntime & out,
                                std::string * err = nullptr);

bool build_expert_split_target_placement(const ExpertSplitRuntime & runtime,
                                         int n_expert_used,
                                         int target_index,
                                         MoeHybridPlacement & out,
                                         std::string * err = nullptr);

bool build_all_expert_split_target_placements(const ExpertSplitRuntime & runtime,
                                              int n_expert_used,
                                              std::vector<ExpertSplitTargetPlacement> & out,
                                              std::string * err = nullptr);

}  // namespace dflash::common
