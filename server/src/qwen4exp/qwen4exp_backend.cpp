#include "qwen4exp_backend.h"
#include "qwen4exp_chunk.h"
#include "qwen4exp_graph.h"

#include "common/sampler.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
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
    const bool mtp = qwen4exp_verify_supported(cache);
    const int ratio = w.compress_ratios.empty() ? 1 : std::max(1, *std::max_element(w.compress_ratios.begin(), w.compress_ratios.end()));
    const int dense_end = std::min(cache.max_ctx, w.indexer_top_k + ratio - 1);
    size_t decode = 0, verify = 0, draft = 0, runtime_scratch = 0, runtime_host = 0;
    auto retain = [&](const Qwen4ExpGraphMemory & m, size_t & resident) {
        if (m.graph == SIZE_MAX) return false;
        resident = std::max(resident, m.graph + m.metadata);
        runtime_scratch = std::max(runtime_scratch, m.scratch);
        runtime_host = std::max(runtime_host, m.host);
        return true;
    };
    // Resident caches (including MTP KV + rollback snapshots) and weights are
    // already deducted from free memory. Reserve all retained graph allocators,
    // including the hidden-output trunk and every admitted catch-up/verify width.
    for (const int end : {dense_end, std::min(cache.max_ctx, dense_end + cache.mtp_draft + 1), cache.max_ctx}) {
        if (!retain(qwen4exp_graph_memory(backend, w, cache, 1, end - 1), decode)) return 0;
        if (mtp) for (int n = 1; n <= cache.mtp_draft + 1 && n <= end; ++n) {
            if (!retain(qwen4exp_mtp_graph_memory(backend, w, cache, n, end - n), draft)) return 0;
            if (n > 1 && !retain(qwen4exp_graph_memory(backend, w, cache, n, end - n, true), verify)) return 0;
        }
    }
    // The decode workspaces stay resident while the next prompt prefills.
    const size_t fixed = shadow + shadow_tmp + decode + verify + draft;
    std::fprintf(stderr, "[qwen4exp] chunk-runtime mtp=%d decode=%zu verify=%zu draft=%zu scratch=%zu host=%zu\n",
        (int) mtp, decode, verify, draft, runtime_scratch, runtime_host);
    struct Plan { size_t graph = 0, ring = 0, host = 0, scratch = 0; };
    std::vector<std::pair<int, Plan>> plans;   // one per probed chunk size
    const int chunk = qwen4exp_fit_chunk(cache.max_ctx, available, fixed, [&](int n) {
        size_t graph = 0, inputs = 0, mask = 0, host = runtime_host, scratch = runtime_scratch;
        // End-of-context QSA workspace, and the largest dense span before QSA.
        // Include an unaligned context tail, which can fall back to dense FA.
        auto measure = [&](int rows, int end) {
            if (rows <= 0 || end <= 0) return true;
            rows = std::min(rows, end);
            const auto m = qwen4exp_graph_memory(backend, w, cache, rows, end - rows);
            if (m.graph == SIZE_MAX) return false;
            graph = std::max(graph, m.graph + m.metadata);
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
    if (cfg_.verify_width < 0 || cfg_.verify_width > QWEN4EXP_MTP_MAX_VERIFY) {
        std::fprintf(stderr, "[qwen4exp] --verify-width must be 0..%d\n", QWEN4EXP_MTP_MAX_VERIFY);
        return false;
    }
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
    if (!load_qwen4exp_gguf(cfg_.model_path, backend_, weights_,
                           cfg_.verify_width == 1 ? "0" : cfg_.draft_path.value_or(""))) {
        std::fprintf(stderr, "[qwen4exp] model load failed: %s\n",
                     luce_last_error());
        return false;
    }
    if (!create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx, cache_, /*mtp=*/true,
                               cfg_.verify_width == 0 ? QWEN4EXP_MTP_MAX_DRAFT : std::max(1, cfg_.verify_width - 1))) {
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
    if (!load_qwen4exp_gguf(cfg_.model_path, backend_, weights_,
                           cfg_.verify_width == 1 ? "0" : cfg_.draft_path.value_or(""))) {
        std::fprintf(stderr, "[qwen4exp] unpark reload failed: %s\n",
                     luce_last_error());
        return false;
    }
    if (!create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx, cache_, /*mtp=*/true,
                               cfg_.verify_width == 0 ? QWEN4EXP_MTP_MAX_DRAFT : std::max(1, cfg_.verify_width - 1))) {
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

    // MTP speculation (sidecar loaded, default graph, a budget that leaves room for a draft): the draft head
    // predicts x_{p+2} from the pair (h_p, x_{p+1}), h_p being the trunk's final HC residual at p. Pairs at positions
    // [mtp_pos, ...) not yet run through the draft layer: mtp_h holds their hidden rows, mtp_tok the tokens known so
    // far (a pair's token arrives with the next forward).
    const bool spec = !req.force_ar_decode && req.n_gen > 2 && qwen4exp_verify_supported(cache_);
    const size_t hd = (size_t) weights_.n_embd * weights_.n_hc;
    std::vector<float> hidden, mtp_h, mtp_logits;
    std::vector<int32_t> mtp_tok;
    int mtp_pos = 0;

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
        const bool mtp_prefill = spec && n > 1;
        const Qwen4ExpForwardResult r = qwen4exp_forward(
            backend_, weights_, cache_, req.prompt.data() + i, n, pos, logits, spec ? &hidden : nullptr,
            false, mtp_prefill, &inputs);
        if (!r.ok) {
            result.fail(GenerateErrorCode::PrefillFailed, "qwen4exp prefill forward failed");
            return result;
        }
        if (mtp_prefill) {
            mtp_h.swap(hidden);
            mtp_pos = pos + n - 1;
        } else if (spec) {   // single-row chunk: retain the original catch-up path
            mtp_tok.assign(req.prompt.begin() + pos + (pos == 0 ? 1 : 0), req.prompt.begin() + pos + n);
            mtp_h.insert(mtp_h.end(), hidden.begin(), hidden.end());
            const int n_pairs = (int) mtp_tok.size();
            if (n_pairs > 0 && !qwen4exp_mtp_forward(backend_, weights_, cache_, mtp_tok.data(), mtp_h.data(),
                                                     n_pairs, mtp_pos, mtp_logits, nullptr, /*kv_only=*/true)) {
                result.fail(GenerateErrorCode::PrefillFailed, "qwen4exp MTP catch-up failed");
                return result;
            }
            mtp_h.erase(mtp_h.begin(), mtp_h.begin() + (std::ptrdiff_t) ((size_t) n_pairs * hd));
            mtp_pos += n_pairs;
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
    auto sample = [&](const float * row) {
        return (int32_t) sample_logits(row, weights_.n_vocab, req.sampler, history, rng);
    };
    BudgetHookState budget;   // thinking force-close: keeps the reply reserve of the budget for the answer
    bool cancelled = false;
    // Commits a sampled token, after the budget hook's substitution; false once generation ends.
    auto commit = [&](int32_t & tok) {
        if (budget.apply(req.budget_hook, (int) result.tokens.size(), req.n_gen, tok)) result.budget_forced_close = true;
        result.tokens.push_back(tok);
        io.emit(tok);
        if (io.is_cancelled()) {
            cancelled = true;
            return false;
        }
        return tok != weights_.eos_id && tok != weights_.eos_chat_id && (int) result.tokens.size() < req.n_gen;
    };

    // Sample trunk rows lazily, updating history and applying the budget hook
    // exactly once per emitted token. No RNG draws for unvisited verify rows.
    long long drafts = 0, accepted = 0, steps = 0;
    std::array<long long, QWEN4EXP_MTP_MAX_VERIFY> width_steps{};
    auto width_policy = qwen4exp_mtp_width_policy(cache_.mtp_draft,
        spec && cfg_.verify_width == 0, (int) req.prompt.size());
    double draft_s = 0.0;
    const auto t_dec0 = std::chrono::steady_clock::now();
    int32_t next = sample(logits.data());
    bool more = req.n_gen > 0 && commit(next);
    if (spec) mtp_tok.assign(1, next);
    std::vector<int32_t> draft_tokens;
    while (more) {
        const int k = spec ? std::max(0, std::min({qwen4exp_mtp_next_width(width_policy) - 1,
            req.n_gen - (int) result.tokens.size() - 1, cache_.max_ctx - pos - 1})) : 0;
        const bool verify = k > 0;
        const auto step_start = verify ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (verify) {
            const auto td0 = std::chrono::steady_clock::now();
            if (!qwen4exp_mtp_draft(backend_, weights_, cache_, mtp_tok.data(), mtp_h.data(), (int) mtp_tok.size(),
                                    mtp_pos, k, draft_tokens)) {
                result.fail(GenerateErrorCode::DecodeFailed, "qwen4exp MTP draft failed");
                return result;
            }
            draft_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();
        }
        std::array<int32_t, QWEN4EXP_MTP_MAX_VERIFY> in{}, samples{};
        in[0] = next;
        if (verify) std::copy(draft_tokens.begin(), draft_tokens.end(), in.begin() + 1);
        const Qwen4ExpForwardResult r = qwen4exp_forward(
            backend_, weights_, cache_, in.data(), k + 1, pos, logits, verify ? &hidden : nullptr, verify);
        if (!r.ok) {
            result.fail(GenerateErrorCode::DecodeFailed, "qwen4exp decode forward failed");
            return result;
        }
        ++steps;
        ++width_steps[k];
        drafts += k;
        Qwen4ExpMtpAcceptance decision;
        for (int i = 0; i <= k; ++i) {
            history.push_back(in[i]);
            int32_t tok = sample(logits.data() + (size_t) i * weights_.n_vocab);
            more = commit(tok);
            samples[i] = tok;
            decision = qwen4exp_mtp_accept(draft_tokens.data(), k, samples.data(), i + 1);
            if (!more || decision.n_accepted != i + 1) break;
        }
        const int retained = decision.n_emitted;
        next = decision.emitted[retained - 1];
        accepted += decision.n_accepted;
        if (verify) {
            // Replace every predicted MTP hidden/KV row with the corresponding
            // trunk pair on next catch-up, including after a partial accept.
            mtp_h.assign(hidden.begin(), hidden.begin() + (std::ptrdiff_t) ((size_t) retained * hd));
            mtp_tok.assign(decision.emitted.begin(), decision.emitted.begin() + retained);
            mtp_pos = pos;
            if (!qwen4exp_verify_rollback(backend_, weights_, cache_, pos, retained)) {
                result.fail(GenerateErrorCode::DecodeFailed, "qwen4exp verify rollback failed");
                return result;
            }
        }
        pos += retained;
        if (verify) width_policy.observe(decision.n_accepted + 1, k + 1,
            (float) std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - step_start).count());
    }
    if (cancelled) {
        result.fail(GenerateErrorCode::Cancelled, "cancelled during decode");
        return result;
    }
    const auto t_dec1 = std::chrono::steady_clock::now();
    result.decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    result.spec_decode_ran = drafts > 0;
    result.accept_rate = drafts > 0 ? (float) accepted / (float) drafts : 0.0f;
    if (drafts > 0) {
        const double decoded = (double) result.tokens.size() - 1.0;   // the first token came from the prefill
        std::fprintf(stderr,
            "[qwen4exp-mtp] k=%d drafts=%lld accepted=%lld rate=%.3f tokens_per_step=%.3f draft_ms=%.2f decode=%.2f tok/s "
            "adaptive=%d steps_k1=%lld steps_k2=%lld steps_k3=%lld steps_k4=%lld steps_k5=%lld steps_k6=%lld steps_k7=%lld\n",
            cache_.mtp_draft, drafts, accepted, (double) accepted / (double) drafts, steps > 0 ? decoded / (double) steps : 0.0,
            1e3 * draft_s / (double) drafts, result.decode_s > 0.0 ? decoded / result.decode_s : 0.0,
            (int) width_policy.enabled(), width_steps[1], width_steps[2], width_steps[3], width_steps[4],
            width_steps[5], width_steps[6], width_steps[7]);
    }

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
