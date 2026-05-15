// Layer-split daemon loop — extracted from test_dflash.cpp.

#include "layer_split_daemon_loop.h"
#include "layer_split_types.h"
#include "layer_split_daemon.h"  // run_target_layer_split_request
#include "layer_split_forward.h" // free_target_layer_split_shards
#include "dflash_feature_ring.h"
#include "peer_access.h"
#include "common/io_utils.h"
#include "common/sampler.h"
#include "common/layer_split_utils.h"
#include "common/gguf_inspect.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>

namespace dflash27b {

int run_layer_split_daemon(const LayerSplitDaemonConfig & cfg) {
    const auto info = inspect_gguf_model_info(cfg.target_path);
    const int n_layer = info.n_layer;
    if (n_layer <= 0) {
        std::fprintf(stderr, "target-split could not read qwen35.block_count\n");
        return 1;
    }
    const auto ranges = compute_layer_ranges(
        n_layer, (int)cfg.target_gpus.size(), cfg.split_weights);
    if ((int)ranges.size() != (int)cfg.target_gpus.size()) {
        std::fprintf(stderr, "bad --target-layer-split for %zu target GPUs and %d layers\n",
                     cfg.target_gpus.size(), n_layer);
        return 2;
    }

    // Initialize shards.
    std::vector<TargetLayerSplitShard> shards(cfg.target_gpus.size());
    for (size_t i = 0; i < cfg.target_gpus.size(); i++) {
        shards[i].gpu = cfg.target_gpus[i];
        shards[i].layer_begin = ranges[i].first;
        shards[i].layer_end = ranges[i].second;
        shards[i].backend = ggml_backend_cuda_init(shards[i].gpu);
        if (!shards[i].backend) {
            std::fprintf(stderr, "target-split cuda init failed for gpu %d\n", shards[i].gpu);
            free_target_layer_split_shards(shards);
            return 1;
        }
    }

    // Enable peer access between all GPU pairs.
    for (size_t i = 0; i < cfg.target_gpus.size(); i++) {
        for (size_t j = i + 1; j < cfg.target_gpus.size(); j++) {
            if (cfg.peer_access) {
                (void)enable_peer_access_pair(cfg.target_gpus[i], cfg.target_gpus[j]);
            }
        }
    }

    // Load partial target weights + caches.
    for (auto & shard : shards) {
        TargetLoadPlan plan;
        plan.layer_begin = shard.layer_begin;
        plan.layer_end = shard.layer_end;
        plan.load_output = (&shard == &shards.back());
        if (!load_target_gguf_partial(cfg.target_path, shard.backend, plan, shard.weights) ||
            !create_target_cache_partial(shard.weights, cfg.max_ctx, cfg.max_verify_tokens,
                                         shard.backend, shard.cache,
                                         /*prefill_only=*/!cfg.run_dflash,
                                         shard.layer_begin, shard.layer_end,
                                         /*allocate_target_feat=*/false)) {
            std::fprintf(stderr, "target-split load/cache gpu=%d: %s\n",
                         shard.gpu, dflash27b_last_error());
            free_target_layer_split_shards(shards);
            return 1;
        }
    }

    // Load draft model if requested.
    ggml_backend_t draft_backend = nullptr;
    DraftWeights draft_weights;
    DraftFeatureMirror feature_ring;
    bool draft_backend_owned = false;
    if (cfg.load_draft) {
        for (auto & shard : shards) {
            if (shard.gpu == cfg.draft_gpu) draft_backend = shard.backend;
        }
        if (!draft_backend) {
            draft_backend = ggml_backend_cuda_init(cfg.draft_gpu);
            if (!draft_backend) {
                free_target_layer_split_shards(shards);
                return 1;
            }
            draft_backend_owned = true;
        }
        std::string dp(cfg.draft_path ? cfg.draft_path : "");
        const bool is_gguf = (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf");
        const bool draft_ok = is_gguf
            ? load_draft_gguf(cfg.draft_path, draft_backend, draft_weights)
            : load_draft_safetensors(cfg.draft_path, draft_backend, draft_weights);
        if (!draft_ok) {
            std::fprintf(stderr, "target-split draft load gpu=%d: %s\n",
                         cfg.draft_gpu, dflash27b_last_error());
            if (draft_backend_owned) ggml_backend_free(draft_backend);
            free_target_layer_split_shards(shards);
            return 1;
        }
        const int cap = std::min(cfg.max_ctx, 4096);
        if (!draft_feature_mirror_init(feature_ring, draft_backend,
                                       cfg.draft_gpu, cfg.draft_gpu, cap,
                                       draft_weights.n_target_layers,
                                       draft_weights.n_embd)) {
            std::fprintf(stderr, "target-split feature ring init failed on gpu=%d\n",
                         cfg.draft_gpu);
            free_draft_weights(draft_weights);
            if (draft_backend_owned) ggml_backend_free(draft_backend);
            free_target_layer_split_shards(shards);
            return 1;
        }
    }

    // Per-request sampler state.
    SamplerCfg sampler;
    std::mt19937_64 sampler_rng{std::random_device{}()};

    std::printf("[daemon] ready\n");
    std::fflush(stdout);

    for (std::string line; std::getline(std::cin, line); ) {
        sampler = SamplerCfg{};
        if (parse_sampler_token(line, sampler) && sampler.seed != 0) {
            sampler_rng.seed(sampler.seed);
        }
        if (line == "LIST_SLOTS") {
            std::printf("[snap] slots=\n");
            std::fflush(stdout);
            continue;
        }
        if (line.rfind("FREE_SNAPSHOT ", 0) == 0) {
            int slot = -1;
            std::sscanf(line.c_str() + 14, "%d", &slot);
            std::printf("[snap] freed slot=%d\n", slot);
            std::fflush(stdout);
            continue;
        }
        if (line.rfind("SNAPSHOT ", 0) == 0 ||
            line.rfind("RESTORE ", 0) == 0 ||
            line.rfind("RESTORE_CHAIN ", 0) == 0 ||
            line.rfind("SNAPSHOT_THIN ", 0) == 0) {
            std::fprintf(stderr,
                         "[target-split] SNAPSHOT/RESTORE are unsupported in sharded daemon mode\n");
            stream_emit_fd(cfg.stream_fd, -1);
            continue;
        }

        char ppath[1024] = {0};
        int n_gen = 0;
        if (std::sscanf(line.c_str(), "%1023s %d", ppath, &n_gen) != 2) {
            stream_emit_fd(cfg.stream_fd, -1);
            continue;
        }
        auto prompt = read_int32_file(ppath);
        if (prompt.empty()) {
            std::fprintf(stderr, "target-split empty prompt\n");
            stream_emit_fd(cfg.stream_fd, -1);
            continue;
        }

        for (auto & shard : shards) {
            reset_target_cache(shard.cache);
        }
        const bool ok = run_target_layer_split_request(
            shards,
            cfg.load_draft ? &draft_weights : nullptr,
            draft_backend,
            cfg.draft_gpu,
            cfg.load_draft ? &feature_ring : nullptr,
            prompt, n_gen, cfg.max_ctx, cfg.run_dflash,
            /*out_path=*/nullptr,
            cfg.kq_stride_pad, cfg.fa_window, cfg.draft_ctx_max,
            cfg.stream_fd);
        (void)ok;
    }

    draft_feature_mirror_free(feature_ring);
    free_draft_weights(draft_weights);
    if (draft_backend_owned && draft_backend) ggml_backend_free(draft_backend);
    free_target_layer_split_shards(shards);
    return 0;
}

}  // namespace dflash27b
