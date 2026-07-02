#include "expert_split_materialization.h"

#include <algorithm>
#include <sstream>
#include <utility>
#include <vector>

namespace dflash::common {

namespace {

bool validate_runtime(const ExpertSplitRuntime & runtime,
                      int n_expert_used,
                      std::string * err) {
    if (!runtime.matches(runtime.n_layer, runtime.n_expert) ||
        runtime.n_layer <= 0 || runtime.n_expert <= 0) {
        if (err) *err = "expert split runtime not initialized";
        return false;
    }
    if (runtime.targets.empty()) {
        if (err) *err = "expert split runtime has no targets";
        return false;
    }
    if (n_expert_used <= 0 || n_expert_used > runtime.n_expert) {
        if (err) *err = "invalid n_expert_used for expert split materialization";
        return false;
    }
    return true;
}

}  // namespace

bool ExpertSplitMaterialization::matches(int n_layer_,
                                         int n_expert_,
                                         int n_expert_used_) const {
    if (n_layer != n_layer_ || n_expert != n_expert_ ||
        n_expert_used != n_expert_used_) {
        return false;
    }
    if (!primary_placement.matches(n_layer_, n_expert_, n_expert_used_)) {
        return false;
    }
    if (ordered_cold_union &&
        (int) cold_expert_ids_by_layer.size() != n_layer_) {
        return false;
    }
    if (!cold_expert_ids_by_layer.empty() &&
        (int) cold_expert_ids_by_layer.size() != n_layer_) {
        return false;
    }
    return true;
}

bool build_expert_split_materialization(const ExpertSplitRuntime & runtime,
                                        int n_expert_used,
                                        ExpertSplitMaterialization & out,
                                        std::string * err) {
    if (!validate_runtime(runtime, n_expert_used, err)) {
        return false;
    }

    std::vector<ExpertSplitTargetPlacement> placements;
    if (!build_all_expert_split_target_placements(runtime, n_expert_used,
                                                  placements, err)) {
        return false;
    }
    if (placements.size() != runtime.targets.size()) {
        if (err) *err = "expert split target placement count mismatch";
        return false;
    }

    ExpertSplitMaterialization materialization;
    materialization.n_layer = runtime.n_layer;
    materialization.n_expert = runtime.n_expert;
    materialization.n_expert_used = n_expert_used;
    materialization.primary_placement = placements[0].placement;
    materialization.targets.reserve(runtime.targets.size());

    int explicit_non_cpu_targets = 0;
    for (size_t i = 0; i < runtime.targets.size(); ++i) {
        ExpertSplitMaterializationTarget target;
        target.target_index = (int) i;
        target.target = runtime.targets[i].target;
        target.placement = std::move(placements[i].placement);
        if (target.target.backend != "cpu") {
            ++explicit_non_cpu_targets;
        }
        materialization.targets.push_back(std::move(target));
    }

    materialization.ordered_cold_union =
        explicit_non_cpu_targets > 1 && materialization.targets.size() > 1;
    if (!materialization.ordered_cold_union) {
        out = std::move(materialization);
        return true;
    }

    materialization.cold_expert_ids_by_layer.resize((size_t) runtime.n_layer);
    for (int layer = 0; layer < runtime.n_layer; ++layer) {
        std::vector<uint8_t> seen((size_t) runtime.n_expert, 0);
        const auto & primary_hot =
            materialization.primary_placement.hot_expert_ids[(size_t) layer];
        for (int32_t expert : primary_hot) {
            if (expert < 0 || expert >= runtime.n_expert) {
                if (err) *err = "primary placement expert id out of range";
                return false;
            }
            seen[(size_t) expert] = 1;
        }

        auto & cold = materialization.cold_expert_ids_by_layer[(size_t) layer];
        cold.reserve((size_t) runtime.n_expert - primary_hot.size());
        for (size_t ti = 1; ti < runtime.targets.size(); ++ti) {
            const ExpertSplitLayerTarget * layer_target =
                runtime.layer_target_ptr((int) ti, layer);
            if (!layer_target) {
                if (err) *err = "expert split runtime layer target lookup failed";
                return false;
            }
            for (int32_t expert : layer_target->global_expert_ids) {
                if (expert < 0 || expert >= runtime.n_expert) {
                    if (err) *err = "materialized cold expert id out of range";
                    return false;
                }
                if (seen[(size_t) expert]) {
                    if (err) {
                        std::ostringstream ss;
                        ss << "materialized cold expert duplicated on layer "
                           << layer << " expert " << expert;
                        *err = ss.str();
                    }
                    return false;
                }
                seen[(size_t) expert] = 1;
                cold.push_back(expert);
            }
        }

        const size_t expected =
            (size_t) runtime.n_expert - primary_hot.size();
        if (cold.size() != expected) {
            if (err) {
                std::ostringstream ss;
                ss << "materialized cold union count mismatch on layer "
                   << layer << " expected=" << expected
                   << " actual=" << cold.size();
                *err = ss.str();
            }
            return false;
        }
        if (std::find(seen.begin(), seen.end(), 0) != seen.end()) {
            if (err) {
                std::ostringstream ss;
                ss << "materialized cold union missing experts on layer "
                   << layer;
                *err = ss.str();
            }
            return false;
        }
    }

    out = std::move(materialization);
    return true;
}

}  // namespace dflash::common
