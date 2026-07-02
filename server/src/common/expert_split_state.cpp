#include "expert_split_state.h"

namespace dflash::common {

bool ExpertSplitStateComponents::empty() const {
    return plan.assignments.empty() &&
           runtime.targets.empty() &&
           compute_runtime.targets.empty() &&
           materialization.targets.empty();
}

bool ExpertSplitStateComponents::matches(int n_layer_,
                                         int n_expert_,
                                         int n_expert_used_) const {
    return plan.matches(n_layer_, n_expert_) &&
           runtime.matches(n_layer_, n_expert_) &&
           compute_runtime.matches(n_layer_, n_expert_, n_expert_used_) &&
           materialization.matches(n_layer_, n_expert_, n_expert_used_);
}

int ExpertSplitLayerMapping::split_layer_for_physical(int physical_layer) const {
    if (physical_layer < 0 ||
        (size_t) physical_layer >= split_layer_by_physical_layer.size()) {
        return -1;
    }
    return split_layer_by_physical_layer[(size_t) physical_layer];
}

bool build_expert_split_state(const ExpertSplitConfig & cfg,
                              const std::vector<ExpertSplitTarget> & targets,
                              const std::vector<ExpertSplitUnit> & units,
                              int n_expert_used,
                              ExpertSplitStateComponents & out,
                              std::string * err) {
    out = ExpertSplitStateComponents{};

    if (!build_expert_split_plan(cfg, targets, units, out.plan, err)) {
        out = ExpertSplitStateComponents{};
        return false;
    }
    if (!build_expert_split_runtime(out.plan, out.runtime, err)) {
        out = ExpertSplitStateComponents{};
        return false;
    }
    if (!build_expert_split_compute_runtime(
            out.runtime, n_expert_used, out.compute_runtime, err)) {
        out = ExpertSplitStateComponents{};
        return false;
    }
    if (!build_expert_split_materialization(
            out.runtime, n_expert_used, out.materialization, err)) {
        out = ExpertSplitStateComponents{};
        return false;
    }
    return true;
}

bool build_expert_split_layer_mapping(
    int n_total_layer,
    const std::vector<int32_t> & physical_layer_by_split_layer,
    ExpertSplitLayerMapping & out,
    std::string * err) {
    out = ExpertSplitLayerMapping{};
    if (n_total_layer <= 0) {
        if (err) *err = "expert split layer mapping requires n_total_layer > 0";
        return false;
    }

    out.n_total_layer = n_total_layer;
    out.physical_layer_by_split_layer = physical_layer_by_split_layer;
    out.split_layer_by_physical_layer.assign((size_t) n_total_layer, -1);

    for (size_t split_layer = 0;
         split_layer < physical_layer_by_split_layer.size();
         ++split_layer) {
        const int32_t physical_layer =
            physical_layer_by_split_layer[split_layer];
        if (physical_layer < 0 || physical_layer >= n_total_layer) {
            out = ExpertSplitLayerMapping{};
            if (err) *err = "expert split physical layer out of range";
            return false;
        }
        const int32_t existing =
            out.split_layer_by_physical_layer[(size_t) physical_layer];
        if (existing >= 0) {
            out = ExpertSplitLayerMapping{};
            if (err) *err = "expert split physical layer mapped more than once";
            return false;
        }
        out.split_layer_by_physical_layer[(size_t) physical_layer] =
            (int32_t) split_layer;
    }
    return true;
}

bool build_contiguous_expert_split_layer_mapping(
    int n_total_layer,
    int first_split_layer,
    int n_split_layer,
    ExpertSplitLayerMapping & out,
    std::string * err) {
    if (n_total_layer <= 0) {
        if (err) *err = "expert split layer mapping requires n_total_layer > 0";
        return false;
    }
    if (first_split_layer < 0 || first_split_layer > n_total_layer) {
        if (err) *err = "expert split first_split_layer out of range";
        return false;
    }
    if (n_split_layer < 0 || n_split_layer > n_total_layer - first_split_layer) {
        if (err) *err = "expert split contiguous layer count out of range";
        return false;
    }

    std::vector<int32_t> physical_layer_by_split_layer((size_t) n_split_layer);
    for (int split_layer = 0; split_layer < n_split_layer; ++split_layer) {
        physical_layer_by_split_layer[(size_t) split_layer] =
            first_split_layer + split_layer;
    }
    return build_expert_split_layer_mapping(
        n_total_layer, physical_layer_by_split_layer, out, err);
}

}  // namespace dflash::common
