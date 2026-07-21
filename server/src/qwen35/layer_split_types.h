// layer_split_types.h — qwen35 layer-split shard types.
//
// Shared layer-split metadata lives in common/layer_split_utils.h. This header
// keeps only the qwen35-specific shard payload: TargetWeights, TargetCache, and
// the qwen35 step graph.

#pragma once

#include "internal.h"
#include "step_graph.h"
#include "common/layer_split_utils.h"
#include "dflash_layer_split_runtime.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdint>
#include <vector>

namespace dflash::common {

// ── Per-GPU qwen35 shard for layer-split target ─────────────────────

struct Qwen35LayerSplitShard : LayerSplitShardMeta {
    TargetWeights weights;
    TargetCache cache;
    StepGraph layer_graph;
};

struct Qwen35SplitTreeInputs {
    const int32_t * parent_ids = nullptr;
    const uint8_t * visibility = nullptr;
    int n_actual = 0;
    int committed = 0;
};

struct Qwen35SplitCaptureStats {
    uint64_t requested = 0;
    uint64_t enabled = 0;
    uint64_t missing_owner_count = 0;
    std::vector<uint64_t> layers_owned_per_shard;
    std::vector<uint64_t> slots_written_per_shard;
    std::vector<uint64_t> feature_taps_owned_per_shard;
    std::vector<uint64_t> feature_slots_written_per_shard;

    void reset(size_t n_shards) {
        requested = 0;
        enabled = 0;
        missing_owner_count = 0;
        layers_owned_per_shard.assign(n_shards, 0);
        slots_written_per_shard.assign(n_shards, 0);
        feature_taps_owned_per_shard.assign(n_shards, 0);
        feature_slots_written_per_shard.assign(n_shards, 0);
    }
};

} // namespace dflash::common
