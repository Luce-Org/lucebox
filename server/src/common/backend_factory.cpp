// Backend factory implementation.

#include "backend_factory.h"
#include "gguf_inspect.h"

#include "qwen35_backend.h"
#include "qwen35moe_backend.h"
#include "laguna_backend.h"
#include "laguna_layer_split_adapter.h"
#include "qwen3_backend.h"
#include "gemma4_backend.h"
#include "gemma4_layer_split_adapter.h"
#include "deepseek4_backend.h"
#include "deepseek4_layer_split_adapter.h"
#include "layer_split_backend.h"
#include "qwen35_layer_split_adapter.h"

#include <cstdio>
#include <algorithm>

namespace dflash::common {

std::string detect_arch(const char * model_path) {
    auto info = inspect_gguf_model_info(model_path);
    return info.arch;
}

bool arch_supports_remote_draft(const std::string & arch) {
    return arch == "qwen35";
}

bool arch_supports_pflash_compression(const std::string & arch) {
    return arch == "qwen35" || arch == "qwen3";
}

std::unique_ptr<ModelBackend> create_backend(const BackendArgs & args) {
    if (!args.model_path) {
        std::fprintf(stderr, "[backend_factory] model_path is null\n");
        return nullptr;
    }

    const std::string arch = detect_arch(args.model_path);
    if (arch.empty()) {
        std::fprintf(stderr, "[backend_factory] failed to detect architecture from %s\n",
                     args.model_path);
        return nullptr;
    }

    std::fprintf(stderr, "[backend_factory] detected arch=%s\n", arch.c_str());

    if (arch == "qwen35") {
        if (args.device.is_layer_split()) {
            Qwen35LayerSplitAdapterConfig cfg;
            cfg.target_path        = args.model_path;
            cfg.draft_path         = args.draft_path;
            cfg.device             = args.device;
            cfg.draft_gpu          = args.draft_device.gpu;
            cfg.remote_draft       = args.remote_draft;
            cfg.remote_target_shard = args.remote_target_shard;
            cfg.fa_window          = args.fa_window;
            cfg.kq_stride_pad      = args.kq_stride_pad;
            cfg.draft_swa_window   = args.draft_swa_window;
            cfg.draft_ctx_max      = args.draft_ctx_max;
            cfg.chunk              = args.chunk;
            cfg.max_verify_tokens  = args.ddtree_mode
                ? std::max<int>(DFLASH27B_DRAFT_BLOCK_SIZE, args.ddtree_budget + 1)
                : DFLASH27B_DRAFT_BLOCK_SIZE;
            cfg.run_dflash         = args.draft_path != nullptr;

            auto adapter = std::make_unique<Qwen35LayerSplitAdapter>(cfg);
            auto backend = std::make_unique<LayerSplitBackend>(std::move(adapter));
            if (!backend->init()) {
                std::fprintf(stderr, "[backend_factory] LayerSplitBackend(qwen35) init failed\n");
                return nullptr;
            }
            return backend;
        }

        Qwen35Config cfg;
        cfg.target_path        = args.model_path;
        cfg.draft_path         = args.draft_path;
        cfg.device             = args.device;
        cfg.draft_gpu          = args.draft_device.gpu;
        cfg.remote_draft       = args.remote_draft;
        cfg.stream_fd          = args.stream_fd;
        cfg.fa_window          = args.fa_window;
        cfg.kq_stride_pad      = args.kq_stride_pad;
        cfg.draft_swa_window   = args.draft_swa_window;
        cfg.draft_ctx_max      = args.draft_ctx_max;
        cfg.fast_rollback      = args.fast_rollback;
        cfg.seq_verify         = args.seq_verify;
        cfg.ddtree_mode        = args.ddtree_mode;
        cfg.ddtree_budget      = args.ddtree_budget;
        cfg.ddtree_temp        = args.ddtree_temp;
        cfg.ddtree_chain_seed  = args.ddtree_chain_seed;
        cfg.use_feature_mirror = args.use_feature_mirror;

        auto backend = std::make_unique<Qwen35Backend>(cfg);
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] Qwen35Backend init failed\n");
            return nullptr;
        }
        return backend;

    } else if (arch == "qwen35moe") {
        Qwen35Config cfg;
        cfg.target_path        = args.model_path;
        cfg.draft_path         = args.draft_path;
        cfg.device             = args.device;
        cfg.draft_gpu          = args.draft_device.gpu;
        cfg.stream_fd          = args.stream_fd;
        cfg.fa_window          = args.fa_window;
        cfg.kq_stride_pad      = args.kq_stride_pad;
        cfg.draft_swa_window   = args.draft_swa_window;
        cfg.draft_ctx_max      = args.draft_ctx_max;
        cfg.fast_rollback      = args.fast_rollback;
        cfg.seq_verify         = args.seq_verify;
        cfg.ddtree_mode        = args.ddtree_mode;
        cfg.ddtree_budget      = args.ddtree_budget;
        cfg.ddtree_temp        = args.ddtree_temp;
        cfg.ddtree_chain_seed  = args.ddtree_chain_seed;
        cfg.use_feature_mirror = args.use_feature_mirror;

        auto backend = std::make_unique<Qwen35MoeBackend>(cfg);
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] Qwen35MoeBackend init failed\n");
            return nullptr;
        }
        return backend;

    } else if (arch == "laguna") {
        if (args.device.is_layer_split()) {
            LagunaLayerSplitAdapterConfig cfg;
            cfg.target_path = args.model_path;
            cfg.device      = args.device;
            cfg.remote_target_shard = args.remote_target_shard;
            cfg.chunk       = args.chunk;

            auto adapter = std::make_unique<LagunaLayerSplitAdapter>(cfg);
            auto backend = std::make_unique<LayerSplitBackend>(std::move(adapter));
            if (!backend->init()) {
                std::fprintf(stderr, "[backend_factory] LayerSplitBackend(laguna) init failed\n");
                return nullptr;
            }
            return backend;
        }

        LagunaBackendArgs lcfg;
        lcfg.target_path = args.model_path;
        lcfg.draft_path  = args.draft_path ? args.draft_path : "";
        lcfg.draft_gpu   = args.draft_device.gpu;
        lcfg.draft_ctx_max = args.draft_ctx_max;
        lcfg.ddtree_mode = args.ddtree_mode;
        lcfg.ddtree_budget = args.ddtree_budget;
        lcfg.ddtree_temp = args.ddtree_temp;
        lcfg.verify_width = args.verify_width;
        lcfg.device      = args.device;
        lcfg.max_ctx     = args.device.max_ctx;
        lcfg.chunk       = args.chunk;
        // kv_type defaults to Q8_0 in LagunaBackendArgs

        auto backend = std::make_unique<LagunaBackend>(lcfg);
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] LagunaBackend init failed\n");
            return nullptr;
        }
        return backend;

    } else if (arch == "qwen3") {
        Qwen3BackendConfig qcfg;
        qcfg.model_path = args.model_path;
        qcfg.device     = args.device;
        qcfg.stream_fd  = args.stream_fd;
        qcfg.chunk      = args.chunk;

        auto backend = std::make_unique<Qwen3Backend>(qcfg);
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] Qwen3Backend init failed\n");
            return nullptr;
        }
        return backend;

    } else if (arch == "gemma4") {
        if (args.device.is_layer_split()) {
            Gemma4LayerSplitAdapterConfig cfg;
            cfg.target_path = args.model_path;
            cfg.device      = args.device;
            cfg.remote_target_shard = args.remote_target_shard;
            cfg.chunk       = args.chunk;
            cfg.fa_window   = args.fa_window;

            auto adapter = std::make_unique<Gemma4LayerSplitAdapter>(cfg);
            auto backend = std::make_unique<LayerSplitBackend>(std::move(adapter));
            if (!backend->init()) {
                std::fprintf(stderr, "[backend_factory] LayerSplitBackend(gemma4) init failed\n");
                return nullptr;
            }
            return backend;
        }

        Gemma4BackendConfig gcfg;
        gcfg.model_path    = args.model_path;
        gcfg.draft_path    = args.draft_path;
        gcfg.draft_gpu     = args.draft_device.gpu;
        gcfg.draft_ctx_max = args.draft_ctx_max;
        gcfg.device        = args.device;
        gcfg.stream_fd     = args.stream_fd;
        gcfg.chunk         = args.chunk;

        auto backend = std::make_unique<Gemma4Backend>(gcfg);
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] Gemma4Backend init failed\n");
            return nullptr;
        }
        return backend;

    } else if (arch == "deepseek4") {
        const PlacementBackend target_backend =
            args.device.backend == PlacementBackend::Auto
                ? compiled_placement_backend()
                : args.device.backend;

        // HIP single-device launches cannot rely on the CUDA/Halo auto-split
        // path; use the single-backend loader, which can fall back to hybrid
        // expert placement when a full monolithic load does not fit.
        if (target_backend == PlacementBackend::Hip &&
            !args.device.is_layer_split() &&
            !args.remote_target_shard.enabled()) {
            DeepSeek4BackendConfig cfg;
            cfg.model_path = args.model_path;
            cfg.device     = args.device;
            cfg.stream_fd  = args.stream_fd;
            cfg.max_ctx    = args.device.max_ctx;
            cfg.chunk      = args.chunk;
            cfg.expert_top_k = args.ds4_expert_top_k;
            cfg.fused_decode = args.ds4_fused_decode;

            auto backend = std::make_unique<DeepSeek4Backend>(cfg);
            if (!backend->init()) {
                std::fprintf(stderr, "[backend_factory] DeepSeek4Backend init failed\n");
                return nullptr;
            }
            return backend;
        }

        // CUDA builds keep the layer-split backend so they can auto-split
        // across CUDA and remote HIP target shards.
        DeepSeek4LayerSplitAdapterConfig cfg;
        cfg.target_path        = args.model_path;
        cfg.device             = args.device;
        cfg.remote_target_shard = args.remote_target_shard;
        cfg.chunk              = args.chunk;

        auto adapter = std::make_unique<DeepSeek4LayerSplitAdapter>(cfg);
        auto backend = std::make_unique<LayerSplitBackend>(std::move(adapter));
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] LayerSplitBackend(deepseek4) init failed\n");
            return nullptr;
        }
        return backend;

    } else {
        std::fprintf(stderr, "[backend_factory] unsupported architecture: %s\n",
                     arch.c_str());
        return nullptr;
    }
}

}  // namespace dflash::common
