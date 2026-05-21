// Backend factory implementation.

#include "backend_factory.h"
#include "gguf_inspect.h"

#include "qwen35_backend.h"
#include "laguna_backend.h"
#include "qwen3_backend.h"
#include "gemma4_backend.h"

#include "gguf.h"

#include <cstdio>

namespace dflash::common {

std::string detect_arch(const char * model_path) {
    auto info = inspect_gguf_model_info(model_path);
    return info.arch;
}

bool gguf_contains_mtp_tensors(const std::string & path) {
    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx      = nullptr;
    gguf_context * gguf = gguf_init_from_file(path.c_str(), gp);
    if (!gguf) return false;

    // MTP-capable GGUF files carry `qwen35.nextn_predict_layers` > 0.
    // This is the canonical indicator used by qwen35_mtp_loader.cpp.
    bool found = false;
    int64_t kid = gguf_find_key(gguf, "qwen35.nextn_predict_layers");
    if (kid >= 0) {
        uint32_t n = gguf_get_val_u32(gguf, kid);
        found = (n > 0);
    }

    gguf_free(gguf);
    return found;
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

    // Resolve MtpSource::Auto before constructing the backend.
    MtpSource resolved_source = args.mtp_source;
    if (resolved_source == MtpSource::Auto) {
        if (gguf_contains_mtp_tensors(args.model_path)) {
            std::fprintf(stderr, "[backend_factory] mtp=auto: nextn_predict_layers found -> Native\n");
            resolved_source = MtpSource::Native;
        } else {
            std::fprintf(stderr, "[backend_factory] mtp=auto: no nextn_predict_layers -> None\n");
            resolved_source = MtpSource::None;
        }
    }

    if (arch == "qwen35") {
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
        cfg.mtp_gamma        = args.mtp_gamma;
        cfg.mtp_use_topk     = args.mtp_use_topk;
        cfg.mtp_draft_topk   = args.mtp_draft_topk;

        // Map resolved MtpSource to the paths Qwen35Backend expects.
        // Qwen35Backend uses cfg_.mtp_gguf_path != nullptr as the MTP-active sentinel.
        switch (resolved_source) {
            case MtpSource::Native:
                // MTP tensors live inside the target GGUF itself.
                cfg.mtp_gguf_path = args.model_path;
                break;
            case MtpSource::ExternalDrafter:
                cfg.mtp_gguf_path = args.mtp_gguf_path;
                break;
            case MtpSource::None:
            case MtpSource::Auto:  // Auto is fully resolved above; this arm is unreachable.
            default:
                cfg.mtp_gguf_path = nullptr;
                break;
        }

        auto backend = std::make_unique<Qwen35Backend>(cfg);
        if (!backend->init()) {
            std::fprintf(stderr, "[backend_factory] Qwen35Backend init failed\n");
            return nullptr;
        }
        return backend;

    } else if (arch == "laguna") {
        LagunaBackendArgs lcfg;
        lcfg.target_path = args.model_path;
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

    } else {
        std::fprintf(stderr, "[backend_factory] unsupported architecture: %s\n",
                     arch.c_str());
        return nullptr;
    }
}

}  // namespace dflash::common
