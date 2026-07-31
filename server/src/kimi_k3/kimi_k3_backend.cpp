#include "kimi_k3_backend.h"

#include "common/moe_hybrid_placement.h"
#include "common/sampler.h"
#include "dflash27b.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash::common {

KimiK3Backend::KimiK3Backend(const KimiK3BackendConfig & cfg) : cfg_(cfg) {}

KimiK3Backend::~KimiK3Backend() {
    shutdown();
}

void KimiK3Backend::release_expert_backend() {
    if (expert_backend_) {
        ggml_backend_free(expert_backend_);
        expert_backend_ = nullptr;
    }
    expert_gpu_ = -1;
}

bool KimiK3Backend::init_streaming() {
    if (!weights_.routed_experts_streamed ||
        weights_.streamed_layer_regions.empty() ||
        weights_.max_streamed_expert_bytes == 0) {
        std::fprintf(stderr,
                     "[kimi-k3] routed expert streaming metadata is incomplete\n");
        return false;
    }

    MoeExpertOwnerPlacement owner;
    std::string error;
    if (!resolve_moe_expert_owner_placement(
            cfg_.device.primary_gpu(), cfg_.expert_gpu,
            owner, &error)) {
        std::fprintf(stderr,
                     "[kimi-k3] invalid expert-owner placement: %s\n",
                     error.c_str());
        return false;
    }
    expert_gpu_ = owner.expert_gpu;
    if (owner.heterogeneous()) {
        expert_backend_ = ggml_backend_cuda_init(expert_gpu_);
        if (!expert_backend_) {
            std::fprintf(stderr,
                         "[kimi-k3] expert backend init failed for device %d\n",
                         expert_gpu_);
            expert_gpu_ = -1;
            return false;
        }
    }
    auto fail_streaming = [&]() {
        dual_stream_executor_.destroy();
        stream_engine_.destroy();
        secondary_stream_engine_.destroy();
        stream_owner_policy_ = MoeStreamDualOwnerPolicy{};
        stream_placement_ = MoeHybridPlacement{};
        release_expert_backend();
        return false;
    };

    size_t routed_pool_bytes = 0;
    for (const LayerExpertRegions & regions :
         weights_.streamed_layer_regions) {
        size_t bytes_per_expert = 0;
        bool component_overflow = false;
        for (size_t component : {
                 regions.expert_bytes_gate, regions.expert_bytes_up,
                 regions.expert_bytes_down, regions.expert_bytes_gate_up}) {
            if (component >
                std::numeric_limits<size_t>::max() - bytes_per_expert) {
                component_overflow = true;
                break;
            }
            bytes_per_expert += component;
        }
        if (component_overflow) {
            routed_pool_bytes = std::numeric_limits<size_t>::max();
            break;
        }
        if (bytes_per_expert >
            (std::numeric_limits<size_t>::max() - routed_pool_bytes) /
                static_cast<size_t>(weights_.n_expert)) {
            routed_pool_bytes = std::numeric_limits<size_t>::max();
            break;
        }
        routed_pool_bytes +=
            bytes_per_expert * static_cast<size_t>(weights_.n_expert);
    }
    auto stream_config_for = [&](int gpu, const char * owner_name) {
        MoeStreamConfig stream_config = MoeStreamConfig::from_env();
        if (std::getenv("DFLASH_MOE_NVME_DEVICE_CACHE_MB")) {
            stream_config.device_cache_bytes =
                std::min(stream_config.device_cache_bytes, routed_pool_bytes);
            return stream_config;
        }
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_cuda_get_device_memory(gpu, &free_bytes, &total_bytes);
        const size_t gib = 1024ULL * 1024ULL * 1024ULL;
        const size_t reserve = std::max<size_t>(2 * gib, total_bytes / 20);
        stream_config.device_cache_bytes =
            free_bytes > reserve
                ? std::min(free_bytes - reserve, routed_pool_bytes)
                : 0;
        std::fprintf(stderr,
            "[kimi-k3] %s streamed cache: gpu=%d free=%.2f GiB "
            "reserve=%.2f GiB pool=%.2f GiB cache=%.2f GiB\n",
            owner_name, gpu,
            static_cast<double>(free_bytes) / gib,
            static_cast<double>(reserve) / gib,
            static_cast<double>(routed_pool_bytes) / gib,
            static_cast<double>(stream_config.device_cache_bytes) / gib);
        return stream_config;
    };

    const MoeStreamConfig primary_config = stream_config_for(
        cfg_.device.primary_gpu(), "primary");
    if (!stream_engine_.init(
            backend_, weights_.max_streamed_expert_bytes,
            primary_config, &error)) {
        std::fprintf(stderr,
                     "[kimi-k3] primary stream engine initialization failed: %s\n",
                     error.c_str());
        return fail_streaming();
    }
    if (expert_backend_) {
        const MoeStreamConfig secondary_config = stream_config_for(
            expert_gpu_, "secondary");
        if (!secondary_stream_engine_.init(
                expert_backend_, weights_.max_streamed_expert_bytes,
                secondary_config, &error)) {
            std::fprintf(stderr,
                         "[kimi-k3] secondary stream engine initialization failed: %s\n",
                         error.c_str());
            return fail_streaming();
        }
    }

    stream_owner_policy_ = MoeStreamDualOwnerPolicy::from_env();
    stream_placement_ = MoeHybridPlacement{};
    const char * placement_path = std::getenv("DFLASH_MOE_PLACEMENT");
    if (expert_backend_ && placement_path && *placement_path) {
        if (!MoeHybridPlacement::load_json(
                placement_path, stream_placement_, &error) ||
            !stream_placement_.matches(
                static_cast<int>(weights_.streamed_layer_regions.size()),
                weights_.n_expert, weights_.n_expert_used)) {
            std::fprintf(stderr,
                         "[kimi-k3] invalid dual-owner placement %s: %s\n",
                         placement_path,
                         error.empty() ? "model shape mismatch" : error.c_str());
            return fail_streaming();
        }
        stream_owner_policy_.primary_placement = &stream_placement_;
    }

    std::vector<int> descriptors;
    std::vector<MoeNvmeSource> sources;
    descriptors.reserve(weights_.shard_paths.size());
    sources.reserve(weights_.shard_paths.size());
    for (const std::string & shard : weights_.shard_paths) {
#if defined(_WIN32)
        const int fd = ::_open(shard.c_str(), _O_RDONLY | _O_BINARY);
#else
        const int fd = ::open(shard.c_str(), O_RDONLY | O_CLOEXEC);
#endif
        if (fd < 0) {
            std::fprintf(stderr,
                         "[kimi-k3] cannot open expert shard %s: %s\n",
                         shard.c_str(), std::strerror(errno));
            for (int opened : descriptors) {
#if defined(_WIN32)
                ::_close(opened);
#else
                ::close(opened);
#endif
            }
            return fail_streaming();
        }
        uint64_t shard_bytes = 0;
#if defined(_WIN32)
        struct _stat64 stat_buffer {};
        if (::_fstat64(fd, &stat_buffer) == 0 && stat_buffer.st_size > 0) {
            shard_bytes = static_cast<uint64_t>(stat_buffer.st_size);
        }
#else
        struct stat stat_buffer {};
        if (::fstat(fd, &stat_buffer) == 0 && stat_buffer.st_size > 0) {
            shard_bytes = static_cast<uint64_t>(stat_buffer.st_size);
        }
#endif
        if (shard_bytes == 0 ||
            shard_bytes > std::numeric_limits<size_t>::max()) {
            std::fprintf(stderr,
                         "[kimi-k3] cannot determine expert shard size: %s\n",
                         shard.c_str());
#if defined(_WIN32)
            ::_close(fd);
#else
            ::close(fd);
#endif
            for (int opened : descriptors) {
#if defined(_WIN32)
                ::_close(opened);
#else
                ::close(opened);
#endif
            }
            return fail_streaming();
        }
        descriptors.push_back(fd);
        sources.push_back({
            nullptr, static_cast<size_t>(shard_bytes), fd});
    }
    const bool primary_bound = stream_engine_.bind_sources(
        sources, weights_.streamed_layer_regions, &error);
    bool secondary_bound = true;
    std::string secondary_error;
    if (primary_bound && expert_backend_) {
        secondary_bound = secondary_stream_engine_.bind_sources(
            sources, weights_.streamed_layer_regions, &secondary_error);
    }
    for (int fd : descriptors) {
#if defined(_WIN32)
        ::_close(fd);
#else
        ::close(fd);
#endif
    }
    if (!primary_bound || !secondary_bound) {
        std::fprintf(stderr,
                     "[kimi-k3] stream source binding failed: %s\n",
                     primary_bound ? secondary_error.c_str() : error.c_str());
        return fail_streaming();
    }
    if (expert_backend_ && !dual_stream_executor_.init(
            stream_engine_, secondary_stream_engine_, &error)) {
        std::fprintf(stderr,
                     "[kimi-k3] dual-owner executor initialization failed: %s\n",
                     error.c_str());
        return fail_streaming();
    }
    if (expert_backend_) {
        std::fprintf(stderr,
            "[kimi-k3] routed experts dual-owner: shards=%zu layers=%zu "
            "primary=%d/%s/%.2fGiB secondary=%d/%s/%.2fGiB "
            "primary_share=%d/1000 placement=%s\n",
            weights_.shard_paths.size(),
            weights_.streamed_layer_regions.size(),
            cfg_.device.primary_gpu(), stream_engine_.io_backend_name(),
            static_cast<double>(stream_engine_.device_cache_bytes()) /
                (1024.0 * 1024.0 * 1024.0),
            expert_gpu_, secondary_stream_engine_.io_backend_name(),
            static_cast<double>(secondary_stream_engine_.device_cache_bytes()) /
                (1024.0 * 1024.0 * 1024.0),
            stream_owner_policy_.primary_share_per_mille,
            stream_owner_policy_.primary_placement ? "profile" : "hash");
    } else {
        std::fprintf(stderr,
            "[kimi-k3] routed experts file-backed: shards=%zu layers=%zu "
            "io=%s gpu=%d cache=%.2f GiB\n",
            weights_.shard_paths.size(),
            weights_.streamed_layer_regions.size(),
            stream_engine_.io_backend_name(),
            cfg_.device.primary_gpu(),
            static_cast<double>(stream_engine_.device_cache_bytes()) /
                (1024.0 * 1024.0 * 1024.0));
    }
    return true;
}

bool KimiK3Backend::init() {
    if (!cfg_.model_path) {
        std::fprintf(stderr, "[kimi-k3] model path is null\n");
        return false;
    }
    backend_ = ggml_backend_cuda_init(cfg_.device.primary_gpu());
    if (!backend_) {
        std::fprintf(stderr, "[kimi-k3] GPU backend init failed for device %d\n",
                     cfg_.device.primary_gpu());
        return false;
    }
    const bool stream_routed_experts =
        cfg_.moe_storage != MoeStoragePolicy::Resident;
    if (!load_kimi_k3_gguf(
            cfg_.model_path, backend_, weights_,
            stream_routed_experts)) {
        std::fprintf(stderr, "[kimi-k3] model load failed: %s\n",
                     dflash27b_last_error());
        return false;
    }
    const int max_ctx = std::max(1, cfg_.device.max_ctx);
    if (!create_kimi_k3_cache(backend_, weights_, max_ctx, cache_)) {
        std::fprintf(stderr, "[kimi-k3] cache allocation failed (max_ctx=%d)\n",
                     max_ctx);
        return false;
    }
    if (weights_.routed_experts_streamed && !init_streaming()) return false;
    std::fprintf(stderr,
        "[kimi-k3] native backend ready on device %d (max_ctx=%d, "
        "experts=%s, correctness-first sequential prefill)\n",
        cfg_.device.primary_gpu(), max_ctx,
        !weights_.routed_experts_streamed ? "resident" :
            (expert_backend_ ? "nvme-dual-owner" : "nvme-single-owner"));
    std::fflush(stderr);
    return true;
}

void KimiK3Backend::print_ready_banner() const {
    std::printf("[kimi-k3-daemon] ready (layers=%d hidden=%d experts=%d "
                "vocab=%d max_ctx=%d)\n",
                weights_.n_layer, weights_.n_embd, weights_.n_expert,
                weights_.n_vocab, cache_.max_ctx);
    std::fflush(stdout);
}

bool KimiK3Backend::park(ParkTarget target) {
    if (!park_target_includes_target_model(target)) return false;
    if (!parked_) {
        dual_stream_executor_.destroy();
        stream_engine_.destroy();
        secondary_stream_engine_.destroy();
        release_expert_backend();
        free_kimi_k3_weights(weights_);
        parked_ = true;
    }
    return true;
}

bool KimiK3Backend::unpark(ParkTarget target) {
    if (!park_target_includes_target_model(target)) return false;
    if (parked_) {
        if (!load_kimi_k3_gguf(
                cfg_.model_path, backend_, weights_,
                cfg_.moe_storage != MoeStoragePolicy::Resident) ||
            (weights_.routed_experts_streamed && !init_streaming())) {
            return false;
        }
        parked_ = false;
    }
    return true;
}

int32_t KimiK3Backend::choose_token(const std::vector<float> & logits,
                                    const SamplerCfg & sampler,
                                    const std::vector<int32_t> & history) {
    if (sampler.needs_logit_processing()) {
        return sample_logits(logits.data(), weights_.n_vocab,
                             sampler, history, rng_);
    }
    return static_cast<int32_t>(std::distance(logits.begin(),
        std::max_element(logits.begin(), logits.end())));
}

GenerateResult KimiK3Backend::generate_impl(const GenerateRequest & req,
                                            const DaemonIO & io) {
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    if (parked_) {
        result.fail(GenerateErrorCode::ModelParked);
        out_io.emit(-1);
        return result;
    }
    if (req.prompt.empty()) {
        result.fail(GenerateErrorCode::PrefillFailed, "empty prompt");
        out_io.emit(-1);
        return result;
    }
    if (req.prompt.size() + static_cast<size_t>(std::max(0, req.n_gen)) >
        static_cast<size_t>(cache_.max_ctx)) {
        result.fail(GenerateErrorCode::ContextOverflow,
                    "prompt plus generation exceeds Kimi-K3 cache");
        out_io.emit(-1);
        return result;
    }
    if (req.do_sample && req.sampler.seed != 0) rng_.seed(req.sampler.seed);

    reset_kimi_k3_cache(cache_);
    std::vector<float> logits;
    const auto prefill_begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < req.prompt.size(); ++i) {
        if (!kimi_k3_step(
                backend_, weights_, cache_, req.prompt[i],
                static_cast<int>(i), logits, &stream_engine_,
                dual_stream_executor_.is_ready()
                    ? &dual_stream_executor_ : nullptr,
                &stream_owner_policy_)) {
            result.fail(GenerateErrorCode::PrefillFailed,
                        dflash27b_last_error());
            out_io.emit(-1);
            return result;
        }
    }
    const auto prefill_end = std::chrono::steady_clock::now();
    result.prefill_s = std::chrono::duration<double>(prefill_end - prefill_begin).count();

    const auto decode_begin = std::chrono::steady_clock::now();
    bool budget_close_started = false;
    size_t close_inject_pos = 0;
    for (int i = 0; i < req.n_gen; ++i) {
        int32_t next = choose_token(logits, req.sampler, result.tokens);

        // Preserve the shared Level-2 budget contract even before speculative
        // decode support lands for Kimi-K3.
        const auto & close_ids = req.budget_hook.close_token_ids;
        if (!close_ids.empty()) {
            if (budget_close_started && close_inject_pos < close_ids.size()) {
                next = close_ids[close_inject_pos++];
                result.budget_forced_close = true;
            } else if (!budget_close_started &&
                       req.n_gen - i <=
                           req.budget_hook.hard_limit_remaining) {
                budget_close_started = true;
                if (next == close_ids.front()) {
                    close_inject_pos = 1;
                } else {
                    next = close_ids.front();
                    close_inject_pos = 1;
                    result.budget_forced_close = true;
                }
            }
        }

        result.tokens.push_back(next);
        out_io.emit(next);
        if (out_io.cancelled || next == weights_.eos_token_id) break;
        if (i + 1 < req.n_gen) {
            if (!kimi_k3_step(
                    backend_, weights_, cache_, next,
                    cache_.cur_pos, logits, &stream_engine_,
                    dual_stream_executor_.is_ready()
                        ? &dual_stream_executor_ : nullptr,
                    &stream_owner_policy_)) {
                result.fail(GenerateErrorCode::DecodeFailed,
                            dflash27b_last_error());
                out_io.emit(-1);
                return result;
            }
        }
    }
    const auto decode_end = std::chrono::steady_clock::now();
    result.decode_s = std::chrono::duration<double>(decode_end - decode_begin).count();
    out_io.emit(-1);
    result.succeed();
    return result;
}

bool KimiK3Backend::snapshot_save(int slot) {
    (void)slot;
    return false;
}

void KimiK3Backend::snapshot_free(int slot) {
    (void)slot;
}

bool KimiK3Backend::snapshot_used(int slot) const {
    (void)slot;
    return false;
}

int KimiK3Backend::snapshot_cur_pos(int slot) const {
    (void)slot;
    return 0;
}

GenerateResult KimiK3Backend::restore_and_generate_impl(
        int slot, const GenerateRequest & req, const DaemonIO & io) {
    (void)slot;
    (void)req;
    GenerateResult result;
    result.fail(GenerateErrorCode::InvalidSnapshotSlot,
                "Kimi-K3 prefix snapshots are not implemented yet");
    io.emit(-1);
    return result;
}

bool KimiK3Backend::handle_compress(const std::string & line,
                                    const DaemonIO & io) {
    (void)line;
    (void)io;
    return false;
}

void KimiK3Backend::shutdown() {
    dual_stream_executor_.destroy();
    stream_engine_.destroy();
    secondary_stream_engine_.destroy();
    release_expert_backend();
    free_kimi_k3_cache(cache_);
    free_kimi_k3_weights(weights_);
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    parked_ = false;
}

} // namespace dflash::common
