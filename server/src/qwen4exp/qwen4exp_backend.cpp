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

namespace dflash::common {

Qwen4ExpBackend::Qwen4ExpBackend(Qwen4ExpBackendConfig cfg)
    : cfg_(std::move(cfg)) {}

Qwen4ExpBackend::~Qwen4ExpBackend() {
    shutdown();
}

bool Qwen4ExpBackend::init() {
    // bf16 WMMA dequant GEMM (journey step 05). RDNA3.5/4 only; mmb_enabled()
    // still gates on the device cc. Process-global by design: this daemon
    // builds a single backend, so it is effectively qwen4exp-scoped. Set before
    // the first graph compute so the cached env lookup in mmb.cu sees it,
    // without clobbering an explicit GGML_CUDA_MMB=0 from the operator.
    setenv("GGML_CUDA_MMB", "1", 0);
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
                     dflash27b_last_error());
        return false;
    }
    if (!create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx,
                               GGML_TYPE_F16, cache_)) {
        std::fprintf(stderr, "[qwen4exp] cache creation failed\n");
        return false;
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
    (void) target;
    // Weight residency control lands with the graph; nothing to release yet.
    parked_ = true;
    return true;
}

bool Qwen4ExpBackend::unpark(ParkTarget target) {
    (void) target;
    parked_ = false;
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

    // A generate() call is one fresh sequence: clear the recurrent state and
    // the PLE n-gram window before the prefill writes position 0.
    reset_qwen4exp_state(backend_, cache_);
    cache_.ple_prev.clear();
    cache_.cur_pos = 0;

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
    free_qwen4exp_cache(cache_);
    free_qwen4exp_weights(weights_);
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

}  // namespace dflash::common
