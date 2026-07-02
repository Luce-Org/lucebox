#include "expert_split_compute_runtime.h"

#include <utility>

namespace dflash::common {

namespace {

uint64_t hash_u64(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

uint64_t hash_string(uint64_t h, const std::string & s) {
    for (unsigned char c : s) {
        h = hash_u64(h, (uint64_t)c);
    }
    return h;
}

}  // namespace

bool ExpertSplitComputeRuntime::matches(int n_layer_, int n_expert_, int n_expert_used_) const {
    return n_layer == n_layer_ &&
           n_expert == n_expert_ &&
           n_expert_used == n_expert_used_ &&
           (int) target_index_by_global.size() == n_layer * n_expert &&
           (int) local_index_by_global.size() == n_layer * n_expert;
}

int ExpertSplitComputeRuntime::index(int layer, int expert) const {
    if (layer < 0 || layer >= n_layer || expert < 0 || expert >= n_expert) {
        return -1;
    }
    return layer * n_expert + expert;
}

int ExpertSplitComputeRuntime::target_index(int layer, int expert) const {
    const int idx = index(layer, expert);
    return idx >= 0 ? target_index_by_global[(size_t)idx] : -1;
}

int ExpertSplitComputeRuntime::local_index(int layer, int expert) const {
    const int idx = index(layer, expert);
    return idx >= 0 ? local_index_by_global[(size_t)idx] : -1;
}

bool build_expert_split_compute_runtime(const ExpertSplitRuntime & runtime,
                                        int n_expert_used,
                                        ExpertSplitComputeRuntime & out,
                                        std::string * err) {
    if (!runtime.matches(runtime.n_layer, runtime.n_expert) ||
        runtime.n_layer <= 0 || runtime.n_expert <= 0) {
        if (err) *err = "expert split runtime not initialized";
        return false;
    }
    if (n_expert_used <= 0 || n_expert_used > runtime.n_expert) {
        if (err) *err = "invalid n_expert_used for expert split compute runtime";
        return false;
    }

    std::vector<ExpertSplitTargetPlacement> placements;
    if (!build_all_expert_split_target_placements(runtime, n_expert_used, placements, err)) {
        return false;
    }
    if (placements.size() != runtime.targets.size()) {
        if (err) *err = "expert split target placement count mismatch";
        return false;
    }

    ExpertSplitComputeRuntime compute_runtime;
    compute_runtime.n_layer = runtime.n_layer;
    compute_runtime.n_expert = runtime.n_expert;
    compute_runtime.n_expert_used = n_expert_used;
    compute_runtime.target_index_by_global = runtime.target_index_by_global;
    compute_runtime.local_index_by_global = runtime.local_index_by_global;
    compute_runtime.targets.reserve(runtime.targets.size());

    for (size_t i = 0; i < runtime.targets.size(); ++i) {
        ExpertSplitComputeTargetRuntime target_runtime;
        target_runtime.target_index = (int) i;
        target_runtime.target = runtime.targets[i].target;
        target_runtime.placement = std::move(placements[i].placement);
        compute_runtime.targets.push_back(std::move(target_runtime));
    }

    out = std::move(compute_runtime);
    return true;
}

uint64_t expert_split_compute_runtime_fingerprint(
    const ExpertSplitComputeRuntime & runtime) {
    uint64_t h = 1469598103934665603ULL;
    h = hash_u64(h, (uint64_t)runtime.n_layer);
    h = hash_u64(h, (uint64_t)runtime.n_expert);
    h = hash_u64(h, (uint64_t)runtime.n_expert_used);
    h = hash_u64(h, (uint64_t)runtime.targets.size());
    for (size_t ti = 0; ti < runtime.targets.size(); ++ti) {
        const ExpertSplitComputeTargetRuntime & target = runtime.targets[ti];
        h = hash_u64(h, (uint64_t)ti);
        h = hash_string(h, target.target.name);
        h = hash_string(h, target.target.backend);
        h = hash_u64(h, (uint64_t)(int64_t)target.target.device_id);
        h = hash_u64(h, target.placement.n_layer);
        h = hash_u64(h, target.placement.n_expert);
        h = hash_u64(h, target.placement.n_expert_used);
        h = hash_u64(h, target.placement.total_hot);
        h = hash_u64(h, (uint64_t)target.placement.hot_counts.size());
        for (int hot_count : target.placement.hot_counts) {
            h = hash_u64(h, (uint64_t)hot_count);
        }
        for (size_t il = 0; il < target.placement.hot_expert_ids.size(); ++il) {
            h = hash_u64(h, (uint64_t)il);
            for (int32_t expert : target.placement.hot_expert_ids[il]) {
                h = hash_u64(h, (uint64_t)(uint32_t)expert);
            }
        }
    }
    return h;
}

}  // namespace dflash::common
