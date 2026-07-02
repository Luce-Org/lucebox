// ExpertSplit compute runtime scaffolding.
//
// This layer is intentionally non-executing for now: it derives per-target
// expert placements and leaves target-local compute/storage construction to a
// later phase. The goal is to preserve the ordered multi-target contract
// without disturbing the existing hot/cold execution path.

#pragma once

#include "expert_split_runtime.h"

#include <vector>

namespace dflash::common {

struct ExpertSplitComputeTargetRuntime {
    int target_index = -1;
    ExpertSplitTarget target;
    MoeHybridPlacement placement;
};

struct ExpertSplitComputeRuntime {
    int n_layer = 0;
    int n_expert = 0;
    int n_expert_used = 0;
    std::vector<ExpertSplitComputeTargetRuntime> targets;
    std::vector<int32_t> target_index_by_global;
    std::vector<int32_t> local_index_by_global;

    bool matches(int n_layer_, int n_expert_, int n_expert_used_) const;
    int index(int layer, int expert) const;
    int target_index(int layer, int expert) const;
    int local_index(int layer, int expert) const;
};

bool build_expert_split_compute_runtime(const ExpertSplitRuntime & runtime,
                                        int n_expert_used,
                                        ExpertSplitComputeRuntime & out,
                                        std::string * err = nullptr);

uint64_t expert_split_compute_runtime_fingerprint(
    const ExpertSplitComputeRuntime & runtime);

}  // namespace dflash::common
