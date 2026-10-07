#include "qwen4exp_backend.h"
#include "qwen4exp_chunk.h"
#include "qwen4exp_graph.h"

#include "common/sampler.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <future>
#include <random>
#include <utility>
#include <vector>

namespace luce::common {

int qwen4exp_select_chunk(ggml_backend_t backend, const Qwen4ExpWeights & w,
        Qwen4ExpCache & cache) {
    size_t free_device = 0, total = 0, free_host = 0;
    ggml_backend_cuda_get_device_memory(ggml_backend_cuda_get_device_id(backend), &free_device, &total);
    ggml_backend_dev_memory(ggml_backend_get_device(backend), &free_host, &total);
    // The generic UMA query substitutes MemAvailable; retain the HIP/GTT limit too.
    const size_t available = std::min(free_device, free_host);
    size_t shadow = 0, shadow_tmp = 0;
    if (w.gfx1151) {
        auto count_shadow = [&](ggml_context * ctx) {
            for (auto * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
                if (t->buffer && t->ne[2] == 1 && t->ne[3] == 1 && t->ne[1] <= 32768 &&
                    (t->type == GGML_TYPE_IQ4_NL || t->type == GGML_TYPE_Q6_K || t->type == GGML_TYPE_Q5_K)) {
                    const size_t bytes = (size_t) ggml_nelements(t) * 2;
                    shadow += bytes;
                    if (t->type == GGML_TYPE_Q5_K) shadow_tmp = std::max(shadow_tmp, bytes);
                }
            }
        };
        count_shadow(w.ctx);
        for (auto * ctx : w.extra_meta_ctxs) count_shadow(ctx);
    }
    const auto decode = qwen4exp_graph_memory(backend, w, cache, 1, cache.max_ctx - 1);
    if (decode.graph == SIZE_MAX) return 0;
    // The decode workspace stays resident while the next prompt prefills.
    const size_t fixed = shadow + shadow_tmp + decode.graph;
    struct Plan { size_t graph = 0, ring = 0, host = 0, scratch = 0; };
    std::vector<std::pair<int, Plan>> plans;   // one per probed chunk size
    const int chunk = qwen4exp_fit_chunk(cache.max_ctx, available, fixed, [&](int n) {
        size_t graph = 0, inputs = 0, mask = 0, host = 0, scratch = 0;
        // End-of-context QSA workspace, and the largest dense span before QSA.
        // Include an unaligned context tail, which can fall back to dense FA.
        const int ratio = w.compress_ratios.empty() ? 1 : std::max(1, *std::max_element(w.compress_ratios.begin(), w.compress_ratios.end()));
        const int dense_end = std::min(cache.max_ctx, w.indexer_top_k + ratio - 1);
        auto measure = [&](int rows, int end) {
            if (rows <= 0 || end <= 0) return true;
            rows = std::min(rows, end);
            const auto m = qwen4exp_graph_memory(backend, w, cache, rows, end - rows);
            if (m.graph == SIZE_MAX) return false;
            graph = std::max(graph, m.graph);
            inputs = std::max(inputs, m.inputs);
            mask = std::max(mask, m.mask);
            host = std::max(host, m.host);
            scratch = std::max(scratch, m.scratch);
            return true;
        };
        for (const int end : {n, dense_end, std::min(cache.max_ctx, dense_end + n),
                             cache.max_ctx - cache.max_ctx % ratio, cache.max_ctx}) {
            if (!measure(n, end)) return SIZE_MAX;
        }
        // Partial final chunks can cross the MMB (512) or packed QSA (128)
        // dispatch floors. Their F32 buffers/masks can outgrow the full chunk.
        for (const int tail : {511, 127, 1}) {
            if (tail < n && !measure(tail, cache.max_ctx)) return SIZE_MAX;
        }
        plans.push_back({n, {graph, inputs + mask, host, scratch}});
        return graph + inputs + mask + host + scratch;
    });
    for (const auto & [rows, p] : plans) {
        if (rows != chunk) continue;
        std::fprintf(stderr, "[qwen4exp] chunk-plan rows=%d graph=%zu ring=%zu host=%zu scratch=%zu required=%zu available=%zu\n",
            rows, p.graph, p.ring, p.host, p.scratch, fixed + p.graph + p.ring + p.host + p.scratch, available);
        break;
    }
    std::fprintf(stderr, "[qwen4exp] chunk-auto ctx=%d chunk=%d fixed=%zu available=%zu headroom=%zu\n",
        cache.max_ctx, chunk, fixed, available, available / 10);
    return chunk;
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
    if (!create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx, cache_)) {
        std::fprintf(stderr, "[qwen4exp] cache creation failed\n");
        return false;
    }
    chunk_ = cfg_.chunk > 0 ? cfg_.chunk : qwen4exp_select_chunk(backend_, weights_, cache_);
    if (chunk_ <= 0) {
        std::fprintf(stderr, "[qwen4exp] insufficient prefill memory at the configured context\n");
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
    if (target != ParkTarget::TargetModel && target != ParkTarget::All) {
        return false;
    }
    if (parked_) return true;
    free_qwen4exp_cache(cache_);
    free_qwen4exp_weights(weights_);
    ggml_backend_cuda_trim_pool(backend_); // also frees the bf16 weight shadows keyed by the freed weights' addresses
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
    if (!create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx, cache_)) {
        std::fprintf(stderr, "[qwen4exp] unpark cache creation failed\n");
        free_qwen4exp_weights(weights_);
        return false;
    }
    // Keep the resolved serving policy across park/unpark (and /props stable).
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
    const int chunk = chunk_;
    int pos = 0;

    // A generate() call starts a fresh sequence.
    reset_qwen4exp_state(backend_, cache_);

    const auto t_pre0 = std::chrono::steady_clock::now();
    // Exactly one host-only gather ahead. A future joins before returning on
    // failure/cancellation, so neither the prompt nor the reader can be freed
    // while a worker is using it. No GPU or cache mutation on that thread.
    auto prepare = [&](size_t offset) {
        const size_t start = offset > (size_t) (weights_.ple_ngram_size - 1)
            ? offset - (weights_.ple_ngram_size - 1) : 0;
        std::vector<int32_t> prev(req.prompt.begin() + start, req.prompt.begin() + offset);
        return qwen4exp_prepare_inputs(weights_, req.prompt.data() + offset,
            (int) std::min((size_t) chunk, req.prompt.size() - offset), prev);
    };
    auto inputs = prepare(0);
    for (size_t i = 0; i < req.prompt.size();) {
        const int n = (int) std::min((size_t) chunk, req.prompt.size() - i);
        const size_t next = i + n;
        std::future<Qwen4ExpInputs> pending;
        if (next < req.prompt.size()) pending = std::async(std::launch::async, prepare, next);
        const Qwen4ExpForwardResult r = qwen4exp_forward(
            backend_, weights_, cache_, req.prompt.data() + i, n, pos, logits, &inputs);
        if (!r.ok) {
            result.fail(GenerateErrorCode::PrefillFailed, "qwen4exp prefill forward failed");
            return result;
        }
        pos += n;
        if (io.is_cancelled()) {
            result.fail(GenerateErrorCode::Cancelled, "cancelled during prefill");
            return result;
        }
        if (pending.valid()) inputs = pending.get();
        i = next;
    }
    const auto t_pre1 = std::chrono::steady_clock::now();
    result.prefill_s = std::chrono::duration<double>(t_pre1 - t_pre0).count();

    std::mt19937_64 rng(req.sampler.seed != 0 ? req.sampler.seed
                                              : std::random_device{}());
    std::vector<int32_t> history = req.prompt;

    const auto t_dec0 = std::chrono::steady_clock::now();
    BudgetHookState budget;   // thinking force-close: keeps the reply reserve of the budget for the answer
    int32_t next = sample_logits(logits.data(), weights_.n_vocab, req.sampler, history, rng);
    for (int g = 0; g < req.n_gen; ++g) {
        if (budget.apply(req.budget_hook, g, req.n_gen, next)) result.budget_forced_close = true;
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

}  // namespace luce::common
