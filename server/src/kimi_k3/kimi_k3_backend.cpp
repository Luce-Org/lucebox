#include "kimi_k3_backend.h"

#include "common/sampler.h"
#include "dflash27b.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace dflash::common {

KimiK3Backend::KimiK3Backend(const KimiK3BackendConfig & cfg) : cfg_(cfg) {}

KimiK3Backend::~KimiK3Backend() {
    shutdown();
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
    if (!load_kimi_k3_gguf(cfg_.model_path, backend_, weights_)) {
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
    std::fprintf(stderr,
        "[kimi-k3] native backend ready on device %d (max_ctx=%d, "
        "correctness-first sequential prefill)\n",
        cfg_.device.primary_gpu(), max_ctx);
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
        free_kimi_k3_weights(weights_);
        parked_ = true;
    }
    return true;
}

bool KimiK3Backend::unpark(ParkTarget target) {
    if (!park_target_includes_target_model(target)) return false;
    if (parked_) {
        if (!load_kimi_k3_gguf(cfg_.model_path, backend_, weights_)) return false;
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
        if (!kimi_k3_step(backend_, weights_, cache_, req.prompt[i],
                          static_cast<int>(i), logits)) {
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
            if (!kimi_k3_step(backend_, weights_, cache_, next,
                              cache_.cur_pos, logits)) {
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
    free_kimi_k3_cache(cache_);
    free_kimi_k3_weights(weights_);
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    parked_ = false;
}

} // namespace dflash::common
