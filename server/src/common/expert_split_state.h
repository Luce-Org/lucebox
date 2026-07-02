// Shared expert-split state assembly helpers.
//
// Sparse adapters and backends all build the same four planning/runtime
// products: plan, runtime, compute runtime, and materialized placement views.
// Keep that assembly in one place so model-specific adapters only need to
// provide units, targets, and any layer mapping they require.

#pragma once

#include "expert_split_compute_runtime.h"
#include "expert_split_materialization.h"
#include "expert_split_plan.h"
#include "expert_split_runtime.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct ExpertSplitStateComponents {
    ExpertSplitPlan plan;
    ExpertSplitRuntime runtime;
    ExpertSplitComputeRuntime compute_runtime;
    ExpertSplitMaterialization materialization;

    bool empty() const;
    bool matches(int n_layer_, int n_expert_, int n_expert_used_) const;
};

struct ExpertSplitLayerMapping {
    int n_total_layer = 0;
    std::vector<int32_t> physical_layer_by_split_layer;
    std::vector<int32_t> split_layer_by_physical_layer;

    int split_layer_count() const {
        return (int) physical_layer_by_split_layer.size();
    }
    int split_layer_for_physical(int physical_layer) const;
    bool is_split_layer(int physical_layer) const {
        return split_layer_for_physical(physical_layer) >= 0;
    }
};

bool build_expert_split_state(const ExpertSplitConfig & cfg,
                              const std::vector<ExpertSplitTarget> & targets,
                              const std::vector<ExpertSplitUnit> & units,
                              int n_expert_used,
                              ExpertSplitStateComponents & out,
                              std::string * err = nullptr);

bool build_expert_split_layer_mapping(
    int n_total_layer,
    const std::vector<int32_t> & physical_layer_by_split_layer,
    ExpertSplitLayerMapping & out,
    std::string * err = nullptr);

bool build_contiguous_expert_split_layer_mapping(
    int n_total_layer,
    int first_split_layer,
    int n_split_layer,
    ExpertSplitLayerMapping & out,
    std::string * err = nullptr);

}  // namespace dflash::common
