// Layer-split utilities for multi-GPU model serving.
//
// Provides pure functions for computing how to distribute model layers
// across multiple GPUs and validating device configurations.

#pragma once

#include "placement/placement_config.h"

#include <string>
#include <vector>

#include "ggml-backend.h"

namespace dflash::common {

struct LayerSplitRange {
    int begin = 0;
    int end = 0;
};

struct LayerSplitShardMeta {
    PlacementBackend placement_backend = PlacementBackend::Auto;
    int gpu = 0;
    int layer_begin = 0;
    int layer_end = 0;
    ggml_backend_t backend = nullptr;
};

struct MixedLayerSplitPlan {
    size_t remote_begin = 0;
    PlacementBackend remote_backend = PlacementBackend::Auto;
};

template <typename LoadPlan>
inline LoadPlan make_layer_split_load_plan(
        const LayerSplitShardMeta & shard,
        bool is_last_shard) {
    LoadPlan plan;
    plan.layer_begin = shard.layer_begin;
    plan.layer_end = shard.layer_end;
    plan.load_output = is_last_shard;
    return plan;
}

template <typename Shard>
std::vector<LayerSplitShardMeta *> layer_split_shard_metas(
        std::vector<Shard> & shards) {
    std::vector<LayerSplitShardMeta *> metas;
    metas.reserve(shards.size());
    for (auto & shard : shards) {
        metas.push_back(&shard);
    }
    return metas;
}

template <typename Shard>
Shard * find_layer_split_shard(std::vector<Shard> & shards, int layer_idx) {
    for (auto & shard : shards) {
        if (layer_idx >= shard.layer_begin && layer_idx < shard.layer_end) {
            return &shard;
        }
    }
    return nullptr;
}

// Compute [begin, end) layer ranges for each GPU shard.
// If weights is empty, splits layers equally.
// If weights has entries, distributes proportionally (at least 1 layer per GPU).
// Returns empty vector on error (n_layer <= 0 or n_gpus <= 0).
std::vector<LayerSplitRange> compute_layer_ranges(
    int n_layer,
    int n_gpus,
    const std::vector<double> & weights);

bool compute_mixed_layer_split_plan(
    const DevicePlacement & device,
    PlacementBackend local_backend,
    MixedLayerSplitPlan & out,
    const char * log_prefix);

bool compute_target_shard_layer_split_plan(
    const DevicePlacement & device,
    PlacementBackend local_backend,
    MixedLayerSplitPlan & out,
    const char * log_prefix);

bool init_layer_split_shard_metas(
    std::vector<LayerSplitShardMeta *> shards,
    const std::vector<int> & gpus,
    const std::vector<LayerSplitRange> & ranges,
    const char * log_prefix);

bool enable_layer_split_peer_access(
    const std::vector<int> & gpus,
    bool peer_access);

bool init_layer_split_snapshot_backends(
    const std::vector<LayerSplitShardMeta *> & shards,
    std::vector<ggml_backend_t> & snapshot_backends,
    const char * log_prefix);

void free_layer_split_snapshot_backends(
    const std::vector<LayerSplitShardMeta *> & shards,
    std::vector<ggml_backend_t> & snapshot_backends);

}  // namespace dflash::common
