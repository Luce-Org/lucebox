#include "qwen4exp_backend.h"
#include "qwen4exp_graph.h"

#include "common/sampler.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

namespace luce::common {

namespace {
bool enabled_env(const char * name) {
    const char * value = std::getenv(name);
    return value && std::atoi(value) != 0;
}
}

Qwen4ExpBackend::Qwen4ExpBackend(Qwen4ExpBackendConfig cfg)
    : cfg_(std::move(cfg)) {}

Qwen4ExpBackend::~Qwen4ExpBackend() {
    shutdown();
}

bool Qwen4ExpBackend::init() {
    if (cfg_.device.is_layer_split()) {
        std::fprintf(stderr, "[qwen4exp] layer split is not supported yet\n");
        return false;
    }
    backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!backend_) {
        std::fprintf(stderr, "[qwen4exp] backend init failed for GPU %d\n",
                     cfg_.device.gpu);
        return false;
    }
    if (!load_qwen4exp_gguf(cfg_.model_path, backend_, weights_)) {
        std::fprintf(stderr, "[qwen4exp] model load failed: %s\n",
                     luce_last_error());
        return false;
    }
    if (!create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx,
                               GGML_TYPE_F16, cache_)) {
        std::fprintf(stderr, "[qwen4exp] cache creation failed\n");
        return false;
    }
    if (cfg_.max_concurrency > 1 && enabled_env("LUCE_QWEN4EXP_SEQ_ENGINE") &&
        enabled_env("QWEN4EXP_BATCHED_DECODE") &&
        !enabled_env("QWEN4EXP_UPSTREAM")) {
        if (cfg_.max_concurrency > 4 || cfg_.device.max_ctx != 32768) {
            std::fprintf(stderr, "[qwen4exp] sequence engine v1 requires <=4 slots at ctx=32768\n");
            return false;
        }
        seq_caches_.resize((size_t)cfg_.max_concurrency - 1);
        std::vector<Qwen4ExpCache *> caches;
        caches.reserve((size_t)cfg_.max_concurrency);
        caches.push_back(&cache_);
        for (Qwen4ExpCache & cache : seq_caches_) {
            if (!create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx,
                                       GGML_TYPE_F16, cache)) {
                std::fprintf(stderr, "[qwen4exp] full-cache slot allocation failed\n");
                return false;
            }
            caches.push_back(&cache);
        }
        seq_engine_ = std::make_unique<Qwen4ExpSeqEngine>(
            backend_, weights_, std::move(caches), cfg_.device.max_ctx,
            std::min(cfg_.chunk, 512));
        std::fprintf(stderr,
            "[qwen4exp-seq] experimental independent-slot engine enabled: %d full F16 caches, ctx=%d\n",
            cfg_.max_concurrency, cfg_.device.max_ctx);
    }
    return true;
}

void Qwen4ExpBackend::print_ready_banner() const {
    const Qwen4ExpWeights & w = weights_;
    std::printf(
        "[qwen4exp-daemon] ready layers=%d linear=%d full=%d experts=%d/%d "
        "hc=%d/%d ple=%zu ngram=%d ctx=%d\n",
        w.n_layer,
        w.n_layer - w.n_layer / w.full_attention_interval,
        w.n_layer / w.full_attention_interval,
        w.n_expert_used, w.n_expert, w.n_hc, w.hc_lowrank,
        w.ple_layer_ids.size(), w.ple_ngram_size, cfg_.device.max_ctx);
    std::fflush(stdout);
}

bool Qwen4ExpBackend::park(ParkTarget target) {
    if (target != ParkTarget::TargetModel && target != ParkTarget::All) {
        return false;
    }
    if (parked_) return true;
    seq_engine_.reset();
    for (Qwen4ExpCache & cache : seq_caches_) free_qwen4exp_cache(cache);
    seq_caches_.clear();
    free_qwen4exp_cache(cache_);
    free_qwen4exp_weights(weights_);
    parked_ = true;
    std::printf("[qwen4exp] target parked\n");
    std::fflush(stdout);
    return true;
}

bool Qwen4ExpBackend::unpark(ParkTarget target) {
    if (target != ParkTarget::TargetModel && target != ParkTarget::All) {
        return false;
    }
    if (!parked_) return true;
    if (!load_qwen4exp_gguf(cfg_.model_path, backend_, weights_)) {
        std::fprintf(stderr, "[qwen4exp] unpark reload failed: %s\n",
                     luce_last_error());
        return false;
    }
    if (!create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx,
                               GGML_TYPE_F16, cache_)) {
        std::fprintf(stderr, "[qwen4exp] unpark cache creation failed\n");
        free_qwen4exp_weights(weights_);
        return false;
    }
    if (cfg_.max_concurrency > 1 && enabled_env("LUCE_QWEN4EXP_SEQ_ENGINE") &&
        enabled_env("QWEN4EXP_BATCHED_DECODE") &&
        !enabled_env("QWEN4EXP_UPSTREAM")) {
        if (cfg_.max_concurrency > 4 || cfg_.device.max_ctx != 32768) {
            std::fprintf(stderr, "[qwen4exp] sequence engine v1 requires <=4 slots at ctx=32768\n");
            free_qwen4exp_cache(cache_);
            free_qwen4exp_weights(weights_);
            return false;
        }
        seq_caches_.resize((size_t)cfg_.max_concurrency - 1);
        std::vector<Qwen4ExpCache *> caches{&cache_};
        for (Qwen4ExpCache & cache : seq_caches_) {
            if (!create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx,
                                       GGML_TYPE_F16, cache)) return false;
            caches.push_back(&cache);
        }
        seq_engine_ = std::make_unique<Qwen4ExpSeqEngine>(
            backend_, weights_, std::move(caches), cfg_.device.max_ctx,
            std::min(cfg_.chunk, 512));
    }
    parked_ = false;
    std::printf("[qwen4exp] target unparked\n");
    std::fflush(stdout);
    return true;
}

GenerateResult Qwen4ExpBackend::generate_impl(const GenerateRequest & req,
                                              const DaemonIO & io) {
    GenerateResult result;
    if (parked_) {
        result.fail(GenerateErrorCode::ModelParked, "qwen4exp target is parked");
        return result;
    }
    if (req.prompt.empty()) {
        result.fail(GenerateErrorCode::PrefillFailed, "empty prompt");
        return result;
    }

    std::vector<float> logits;
    const int chunk = std::max(1, cfg_.chunk);
    int pos = 0;

    // A generate() call starts a fresh sequence.
    reset_qwen4exp_state(backend_, cache_);

    const auto t_pre0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < req.prompt.size(); i += (size_t) chunk) {
        const int n = (int) std::min((size_t) chunk, req.prompt.size() - i);
        const Qwen4ExpForwardResult r = qwen4exp_forward(
            backend_, weights_, cache_, req.prompt.data() + i, n, pos, logits);
        if (!r.ok) {
            result.fail(GenerateErrorCode::PrefillFailed,
                        "qwen4exp prefill forward failed");
            return result;
        }
        pos += n;
        if (io.is_cancelled()) {
            result.fail(GenerateErrorCode::Cancelled, "cancelled during prefill");
            return result;
        }
    }
    const auto t_pre1 = std::chrono::steady_clock::now();
    result.prefill_s = std::chrono::duration<double>(t_pre1 - t_pre0).count();

    std::mt19937_64 rng(req.sampler.seed != 0 ? req.sampler.seed
                                              : std::random_device{}());
    std::vector<int32_t> history = req.prompt;

    const auto t_dec0 = std::chrono::steady_clock::now();
    int32_t next = sample_logits(logits.data(), weights_.n_vocab, req.sampler, history, rng);
    for (int g = 0; g < req.n_gen; ++g) {
        result.tokens.push_back(next);
        io.emit(next);
        if (io.is_cancelled()) {
            result.fail(GenerateErrorCode::Cancelled, "cancelled during decode");
            return result;
        }
        if (next == weights_.eos_id || next == weights_.eos_chat_id) {
            break;
        }
        if (g + 1 >= req.n_gen) {
            break;
        }
        const Qwen4ExpForwardResult r = qwen4exp_forward(
            backend_, weights_, cache_, &next, 1, pos, logits);
        if (!r.ok) {
            result.fail(GenerateErrorCode::DecodeFailed,
                        "qwen4exp decode forward failed");
            return result;
        }
        pos += 1;
        history.push_back(next);
        next = sample_logits(logits.data(), weights_.n_vocab, req.sampler, history, rng);
    }
    const auto t_dec1 = std::chrono::steady_clock::now();
    result.decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();

    result.succeed();
    return result;
}

bool Qwen4ExpBackend::snapshot_save(int slot) {
    (void) slot;
    return false;
}

void Qwen4ExpBackend::snapshot_free(int slot) {
    (void) slot;
}

bool Qwen4ExpBackend::snapshot_used(int slot) const {
    (void) slot;
    return false;
}

int Qwen4ExpBackend::snapshot_cur_pos(int slot) const {
    (void) slot;
    return 0;
}

GenerateResult Qwen4ExpBackend::restore_and_generate_impl(
        int slot, const GenerateRequest & req, const DaemonIO & io) {
    (void) slot;
    (void) req;
    (void) io;
    GenerateResult result;
    result.fail(GenerateErrorCode::InvalidSnapshotSlot,
                "qwen4exp snapshots are not implemented yet");
    return result;
}

bool Qwen4ExpBackend::handle_compress(const std::string & line,
                                      const DaemonIO & io) {
    (void) line;
    (void) io;
    return false;
}

void Qwen4ExpBackend::free_drafter() {}

void Qwen4ExpBackend::shutdown() {
    seq_engine_.reset();
    for (Qwen4ExpCache & cache : seq_caches_) free_qwen4exp_cache(cache);
    seq_caches_.clear();
    free_qwen4exp_cache(cache_);
    free_qwen4exp_weights(weights_);
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

}  // namespace luce::common
