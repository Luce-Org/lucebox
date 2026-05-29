// Shared target layer-split runtime helpers.
//
// Model adapters still own their partial loaders, graph builders, caches, and
// snapshot payloads. This file keeps shared adapter runtime flow in one place
// so new adapters do not copy the same shell.

#pragma once

#include "gguf_inspect.h"
#include "layer_split_utils.h"
#include "model_backend.h"
#include "placement/placement_config.h"
#include "sampler.h"

#include <cstdio>
#include <functional>
#include <random>
#include <vector>

namespace dflash::common {

struct LayerSplitRuntimeInit {
    const char * target_path = nullptr;
    const DevicePlacement * device = nullptr;
    const char * log_prefix = "target-split";
};

template <typename Shard>
bool init_layer_split_runtime(const LayerSplitRuntimeInit & cfg,
                              std::vector<Shard> & shards,
                              std::vector<ggml_backend_t> & snapshot_backends) {
    if (!cfg.target_path || !cfg.device ||
        cfg.device->layer_split_gpus.size() < 2) {
        std::fprintf(stderr, "[%s] invalid layer-split config\n", cfg.log_prefix);
        return false;
    }

    const auto info = inspect_gguf_model_info(cfg.target_path);
    const int n_layer = info.n_layer;
    if (n_layer <= 0) {
        std::fprintf(stderr, "[%s] failed to inspect target layer count\n",
                     cfg.log_prefix);
        return false;
    }

    const auto ranges = compute_layer_ranges(
        n_layer,
        (int)cfg.device->layer_split_gpus.size(),
        cfg.device->layer_split_weights);
    if (ranges.size() != cfg.device->layer_split_gpus.size()) {
        std::fprintf(stderr,
            "[%s] bad layer split for %zu GPUs and %d layers\n",
            cfg.log_prefix, cfg.device->layer_split_gpus.size(), n_layer);
        return false;
    }

    shards.resize(cfg.device->layer_split_gpus.size());
    auto shard_metas = layer_split_shard_metas(shards);
    if (!init_layer_split_shard_metas(
            shard_metas, cfg.device->layer_split_gpus, ranges,
            cfg.log_prefix)) {
        return false;
    }

    (void)enable_layer_split_peer_access(
        cfg.device->layer_split_gpus, cfg.device->peer_access);

    return init_layer_split_snapshot_backends(
        shard_metas, snapshot_backends, cfg.log_prefix);
}

using LayerSplitForwardStep = std::function<bool(
    const std::vector<int32_t> & tokens,
    int committed,
    int & next_tok,
    std::vector<float> * logits_out)>;

bool run_layer_split_ar_decode(
    int last_tok,
    int committed,
    int n_gen,
    int vocab,
    const std::vector<float> & prefill_last_logits,
    const SamplerCfg & sampler,
    std::mt19937_64 & rng,
    const LayerSplitForwardStep & forward_one,
    const std::function<bool(int)> & is_eos,
    std::vector<int32_t> & out_tokens,
    const DaemonIO & io);

}  // namespace dflash::common
