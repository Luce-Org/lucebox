#include "expert_split_runtime.h"

#include <algorithm>
#include <sstream>
#include <utility>
#include <vector>

namespace dflash::common {

namespace {

bool better_local_order(const ExpertSplitAssignment & lhs,
                        int lhs_expert,
                        const ExpertSplitAssignment & rhs,
                        int rhs_expert) {
    if (lhs.score != rhs.score) return lhs.score > rhs.score;
    if (lhs.bytes != rhs.bytes) return lhs.bytes < rhs.bytes;
    return lhs_expert < rhs_expert;
}

bool validate_plan(const ExpertSplitPlan & plan, std::string * err) {
    if (!plan.matches(plan.n_layer, plan.n_expert) ||
        plan.n_layer <= 0 || plan.n_expert <= 0) {
        if (err) *err = "expert split plan not initialized";
        return false;
    }
    if (plan.targets.empty()) {
        if (err) *err = "expert split plan has no targets";
        return false;
    }
    if (plan.target_used_bytes.size() != plan.targets.size()) {
        if (err) *err = "expert split plan target byte accounting mismatch";
        return false;
    }

    std::vector<uint64_t> recomputed(plan.targets.size(), 0);
    for (int layer = 0; layer < plan.n_layer; ++layer) {
        for (int expert = 0; expert < plan.n_expert; ++expert) {
            const ExpertSplitAssignment & assignment = plan.at(layer, expert);
            if (!assignment.assigned()) {
                if (err) {
                    std::ostringstream ss;
                    ss << "expert split plan missing assignment for layer "
                       << layer << " expert " << expert;
                    *err = ss.str();
                }
                return false;
            }
            if (assignment.target_index < 0 ||
                (size_t) assignment.target_index >= plan.targets.size()) {
                if (err) {
                    std::ostringstream ss;
                    ss << "expert split plan target index out of range for layer "
                       << layer << " expert " << expert;
                    *err = ss.str();
                }
                return false;
            }
            recomputed[(size_t) assignment.target_index] += assignment.bytes;
        }
    }

    for (size_t i = 0; i < plan.targets.size(); ++i) {
        if (recomputed[i] != plan.target_used_bytes[i]) {
            if (err) {
                std::ostringstream ss;
                ss << "expert split plan target byte accounting drift on target "
                   << i << " expected=" << plan.target_used_bytes[i]
                   << " actual=" << recomputed[i];
                *err = ss.str();
            }
            return false;
        }
    }

    return true;
}

}  // namespace

bool ExpertSplitRuntime::matches(int n_layer_, int n_expert_) const {
    return n_layer == n_layer_ &&
           n_expert == n_expert_ &&
           (int) target_index_by_global.size() == n_layer * n_expert &&
           (int) local_index_by_global.size() == n_layer * n_expert;
}

int ExpertSplitRuntime::index(int layer, int expert) const {
    if (layer < 0 || layer >= n_layer || expert < 0 || expert >= n_expert) {
        return -1;
    }
    return layer * n_expert + expert;
}

int ExpertSplitRuntime::target_index(int layer, int expert) const {
    const int idx = index(layer, expert);
    return idx >= 0 ? target_index_by_global[(size_t) idx] : -1;
}

int ExpertSplitRuntime::local_index(int layer, int expert) const {
    const int idx = index(layer, expert);
    return idx >= 0 ? local_index_by_global[(size_t) idx] : -1;
}

const ExpertSplitRuntimeTarget * ExpertSplitRuntime::target_ptr(int target_index_) const {
    if (target_index_ < 0 || (size_t) target_index_ >= targets.size()) {
        return nullptr;
    }
    return &targets[(size_t) target_index_];
}

const ExpertSplitLayerTarget * ExpertSplitRuntime::layer_target_ptr(int target_index_,
                                                                    int layer) const {
    const ExpertSplitRuntimeTarget * target = target_ptr(target_index_);
    if (!target || layer < 0 || layer >= n_layer) {
        return nullptr;
    }
    return &target->layers[(size_t) layer];
}

bool build_expert_split_runtime(const ExpertSplitPlan & plan,
                                ExpertSplitRuntime & out,
                                std::string * err) {
    if (!validate_plan(plan, err)) {
        return false;
    }

    ExpertSplitRuntime runtime;
    runtime.n_layer = plan.n_layer;
    runtime.n_expert = plan.n_expert;
    runtime.targets.resize(plan.targets.size());
    runtime.target_index_by_global.assign(
        (size_t) runtime.n_layer * (size_t) runtime.n_expert, -1);
    runtime.local_index_by_global.assign(
        (size_t) runtime.n_layer * (size_t) runtime.n_expert, -1);

    for (size_t target_index = 0; target_index < plan.targets.size(); ++target_index) {
        ExpertSplitRuntimeTarget & dst = runtime.targets[target_index];
        dst.target = plan.targets[target_index];
        dst.used_bytes = plan.target_used_bytes[target_index];
        dst.layers.resize((size_t) runtime.n_layer);
        for (int layer = 0; layer < runtime.n_layer; ++layer) {
            ExpertSplitLayerTarget & layer_dst = dst.layers[(size_t) layer];
            layer_dst.layer = layer;
            layer_dst.target_index = (int) target_index;
            layer_dst.local_by_global.assign((size_t) runtime.n_expert, -1);
        }
    }

    for (int layer = 0; layer < runtime.n_layer; ++layer) {
        for (int expert = 0; expert < runtime.n_expert; ++expert) {
            const ExpertSplitAssignment & assignment = plan.at(layer, expert);
            ExpertSplitRuntimeTarget & target =
                runtime.targets[(size_t) assignment.target_index];
            ExpertSplitLayerTarget & layer_target =
                target.layers[(size_t) layer];
            layer_target.global_expert_ids.push_back((int32_t) expert);
            layer_target.total_bytes += assignment.bytes;
            layer_target.total_score += assignment.score;
            target.total_experts++;
            target.total_score += assignment.score;
        }
    }

    for (size_t target_index = 0; target_index < runtime.targets.size(); ++target_index) {
        ExpertSplitRuntimeTarget & target = runtime.targets[target_index];
        for (int layer = 0; layer < runtime.n_layer; ++layer) {
            ExpertSplitLayerTarget & layer_target = target.layers[(size_t) layer];
            std::stable_sort(layer_target.global_expert_ids.begin(),
                             layer_target.global_expert_ids.end(),
                [&](int32_t lhs, int32_t rhs) {
                    return better_local_order(plan.at(layer, lhs), lhs,
                                              plan.at(layer, rhs), rhs);
                });
            for (size_t local = 0; local < layer_target.global_expert_ids.size(); ++local) {
                const int32_t expert = layer_target.global_expert_ids[local];
                layer_target.local_by_global[(size_t) expert] = (int32_t) local;
                const int flat = runtime.index(layer, expert);
                runtime.target_index_by_global[(size_t) flat] = (int32_t) target_index;
                runtime.local_index_by_global[(size_t) flat] = (int32_t) local;
            }
        }
    }

    out = std::move(runtime);
    return true;
}

bool build_expert_split_target_placement(const ExpertSplitRuntime & runtime,
                                         int n_expert_used,
                                         int target_index_,
                                         MoeHybridPlacement & out,
                                         std::string * err) {
    if (!runtime.matches(runtime.n_layer, runtime.n_expert) ||
        runtime.n_layer <= 0 || runtime.n_expert <= 0) {
        if (err) *err = "expert split runtime not initialized";
        return false;
    }
    if (n_expert_used <= 0 || n_expert_used > runtime.n_expert) {
        if (err) *err = "invalid n_expert_used for expert split target placement";
        return false;
    }
    const ExpertSplitRuntimeTarget * target = runtime.target_ptr(target_index_);
    if (!target) {
        if (err) *err = "expert split runtime target index out of range";
        return false;
    }
    if ((int) target->layers.size() != runtime.n_layer) {
        if (err) *err = "expert split runtime target layer count mismatch";
        return false;
    }

    MoeHybridPlacement placement;
    placement.n_layer = runtime.n_layer;
    placement.n_expert = runtime.n_expert;
    placement.n_expert_used = n_expert_used;
    placement.total_hot = target->total_experts;
    placement.hot_counts.assign((size_t) runtime.n_layer, 0);
    placement.hot_expert_ids.resize((size_t) runtime.n_layer);

    for (int layer = 0; layer < runtime.n_layer; ++layer) {
        const ExpertSplitLayerTarget & layer_target = target->layers[(size_t) layer];
        if (layer_target.layer != layer || layer_target.target_index != target_index_) {
            if (err) {
                std::ostringstream ss;
                ss << "expert split runtime layer target mismatch target="
                   << target_index_ << " layer=" << layer;
                *err = ss.str();
            }
            return false;
        }
        if ((int) layer_target.local_by_global.size() != runtime.n_expert) {
            if (err) *err = "expert split runtime local_by_global size mismatch";
            return false;
        }

        placement.hot_counts[(size_t) layer] = (int) layer_target.global_expert_ids.size();
        auto & hot = placement.hot_expert_ids[(size_t) layer];
        hot.reserve(layer_target.global_expert_ids.size());
        for (int32_t expert : layer_target.global_expert_ids) {
            if (expert < 0 || expert >= runtime.n_expert) {
                if (err) *err = "expert split runtime global expert id out of range";
                return false;
            }
            hot.push_back(expert);
        }
    }

    out = std::move(placement);
    return true;
}

bool build_all_expert_split_target_placements(const ExpertSplitRuntime & runtime,
                                              int n_expert_used,
                                              std::vector<ExpertSplitTargetPlacement> & out,
                                              std::string * err) {
    if (!runtime.matches(runtime.n_layer, runtime.n_expert)) {
        if (err) *err = "expert split runtime not initialized";
        return false;
    }

    std::vector<ExpertSplitTargetPlacement> placements;
    placements.reserve(runtime.targets.size());
    for (size_t target_index = 0; target_index < runtime.targets.size(); ++target_index) {
        ExpertSplitTargetPlacement placement;
        placement.target_index = (int) target_index;
        if (!build_expert_split_target_placement(runtime, n_expert_used,
                                                 placement.target_index,
                                                 placement.placement, err)) {
            return false;
        }
        placements.push_back(std::move(placement));
    }

    out = std::move(placements);
    return true;
}

}  // namespace dflash::common
