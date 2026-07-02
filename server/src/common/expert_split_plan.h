// Generic ordered expert-residency planner for sparse MoE models.
//
// This planner is intentionally model-agnostic. Adapters describe routed expert
// units and an ordered list of expert targets (for example cuda:0, cuda:1,
// hip:0, cpu), then consume the resulting placement plan. Target order is the
// user-facing tier definition: earlier targets receive hotter experts first.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct ExpertSplitTarget {
    std::string name;           // Logical name, e.g. "cuda:0" or "cpu".
    std::string backend;        // "cuda", "hip", "cpu", ...
    int device_id = -1;         // Backend-local device id, or -1 for host/implicit.
    uint64_t capacity_bytes = 0;
    uint64_t reserved_bytes = 0;
    bool unlimited = false;

    uint64_t usable_bytes() const;
};

struct ExpertSplitUnit {
    int layer = 0;
    int expert = 0;
    uint64_t bytes = 0;
    double score = 0.0; // Routing count, calibrated probability, or hotness.
};

struct ExpertSplitConfig {
    int n_layer = 0;
    int n_expert = 0;

    // Optional per-target floor. Entry i means target i should receive at
    // least this many distinct experts from each layer before later targets
    // get considered. Missing entries default to 0.
    std::vector<int> min_per_layer_by_target;

    bool allow_implicit_cpu_fallback = true;
    bool require_full_grid = true;
};

struct ExpertSplitAssignment {
    int target_index = -1; // Index into ExpertSplitPlan::targets.
    uint64_t bytes = 0;
    double score = 0.0;

    bool assigned() const { return target_index >= 0; }
};

struct ExpertSplitPlan {
    int n_layer = 0;
    int n_expert = 0;
    std::vector<ExpertSplitTarget> targets;
    std::vector<uint64_t> target_used_bytes;
    std::vector<ExpertSplitAssignment> assignments;

    bool matches(int n_layer_, int n_expert_) const;
    int index(int layer, int expert) const;
    const ExpertSplitAssignment & at(int layer, int expert) const;
    ExpertSplitAssignment & at(int layer, int expert);
    int count_on_target(int target_index) const;
    int layer_count_on_target(int layer, int target_index) const;
};

bool build_expert_split_plan(const ExpertSplitConfig & cfg,
                             const std::vector<ExpertSplitTarget> & targets,
                             const std::vector<ExpertSplitUnit> & units,
                             ExpertSplitPlan & out,
                             std::string * err = nullptr);

}  // namespace dflash::common
