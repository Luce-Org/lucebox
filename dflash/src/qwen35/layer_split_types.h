// layer_split_types.h — qwen35 layer-split shard types.
//
// Generic split-runtime helpers (LayerSplitRuntimeConfig, ActivationPair)
// live in `common/dflash_layer_split_runtime.h`. This header keeps the
// qwen35-specific shard layout that embeds TargetWeights/TargetCache.

#pragma once

#include "internal.h"
#include "step_graph.h"
#include "dflash_layer_split_runtime.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>

namespace dflash27b {

// ── Per-GPU shard for layer-split target ────────────────────────────

struct TargetLayerSplitShard {
    int gpu = 0;
    int layer_begin = 0;
    int layer_end = 0;
    ggml_backend_t backend = nullptr;
    TargetWeights weights;
    TargetCache cache;
    StepGraph layer_graph;
};

inline TargetLayerSplitShard * find_target_shard(
        std::vector<TargetLayerSplitShard> & shards,
        int layer_idx) {
    for (auto & shard : shards) {
        if (layer_idx >= shard.layer_begin && layer_idx < shard.layer_end)
            return &shard;
    }
    return nullptr;
}

} // namespace dflash27b
