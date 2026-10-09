// DeepSeek V4.1 Engram: init from loaded weights and the ggml apply subgraph.
#include "deepseek4_engram.h"
#include "deepseek4_internal.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace luce::common {

bool DeepSeek4EngramHasher::init(const DeepSeek4Weights & w, std::string * err) {
    const auto & e = w.engram;
    n_layers_ = 0;
    if (!e.present()) return true;
    if (e.token_map.size() != (size_t) w.n_vocab) {
        if (err) *err = "engram token_map does not cover the vocabulary";
        return false;
    }
    return init_raw(e.layer_ids, e.n_heads, e.max_ngram, e.pad_id, e.token_map,
                    e.multipliers, e.primes, e.offsets, e.rows, err);
}

bool DeepSeek4EngramRuntime::init(const DeepSeek4Weights & w, const std::string & gguf_path, std::string * err) {
    tables_.clear();
    rows_read_ = 0;
    if (!hasher_.init(w, err)) return false;
    if (!hasher_.present()) return true;
    for (int l = 0; l < hasher_.n_layers(); ++l) {
        const int il = hasher_.layer_id(l);
        const DeepSeek4Weights::Engram::Table * tab = nullptr;
        for (const auto & t : w.engram.tables) if (t.layer_id == il) tab = &t;
        if (!tab || tab->rows == 0 || tab->row_bytes != (uint32_t) DeepSeek4EngramTable::kRowBytes) {
            if (err) *err = "engram: layer " + std::to_string(il) + " has no embedded 264-byte-row table in the GGUF "
                            "(PR #28696 files carry Q8_0 tables, which this reader does not decode)";
            return false;
        }
        if (tab->rows != hasher_.rows(l)) {
            if (err) *err = "engram: layer " + std::to_string(il) + " table rows disagree with the hash";
            return false;
        }
        DeepSeek4EngramTable table;
        if (!table.open(gguf_path, tab->file_offset, tab->rows, err)) return false;
        tables_.push_back(std::move(table));
    }
    return true;
}

// ── 3. the apply ──────────────────────────────────────────────────────────

ggml_tensor * deepseek4_build_engram_apply(ggml_context * ctx,
                                           ggml_tensor * h,
                                           ggml_tensor * keys,
                                           const DeepSeek4Layer & L,
                                           int n_embd, int n_hc,
                                           float rms_eps,
                                           ggml_tensor ** gate_out,
                                           ggml_tensor * kv_pre) {
    if (!ctx || !h || !keys || !L.engram_wkv || !L.engram_q || !L.engram_k) return nullptr;
    const int64_t n_tokens = keys->ne[1];
    GGML_ASSERT(h->ne[0] == n_embd && h->ne[1] == n_hc && h->ne[2] == n_tokens);
    GGML_ASSERT(L.engram_wkv->ne[0] == keys->ne[0]);
    GGML_ASSERT(L.engram_wkv->ne[1] == (int64_t) n_embd * (n_hc + 1));

    // [n_embd * (n_hc + 1), n_tokens]: the n_hc keys first, the value last.
    ggml_tensor * kv = kv_pre;
    if (!kv) {
        kv = ggml_mul_mat(ctx, L.engram_wkv, keys);
        // The F16 projection must accumulate in F32: with F16 accumulation (the
        // BLAS default on gfx1151) the update drifts past 2e-3 of its RMS.
        ggml_mul_mat_set_prec(kv, GGML_PREC_F32);
    }
    GGML_ASSERT(kv->ne[0] == L.engram_wkv->ne[1] && kv->ne[1] == n_tokens);
    ggml_tensor * key = ggml_view_3d(ctx, kv, n_embd, n_hc, n_tokens,
                                     (size_t) n_embd * ggml_element_size(kv), kv->nb[1], 0);
    ggml_tensor * value = ggml_view_3d(ctx, kv, n_embd, 1, n_tokens,
                                       (size_t) n_embd * ggml_element_size(kv), kv->nb[1],
                                       (size_t) n_embd * n_hc * ggml_element_size(kv));

    // weight[c, d] = q[c, d] * k[c, d] (the reference only ever uses the product).
    ggml_tensor * weight = ggml_mul(ctx, L.engram_q, L.engram_k);            // [n_embd, n_hc]
    ggml_tensor * h_n = ggml_rms_norm(ctx, h, rms_eps);                      // per (copy, token)
    ggml_tensor * k_n = ggml_rms_norm(ctx, ggml_cont(ctx, key), rms_eps);
    ggml_tensor * prod = ggml_mul(ctx, ggml_mul(ctx, h_n, weight), k_n);     // [n_embd, n_hc, n_tokens]
    ggml_tensor * dot = ggml_scale(ctx, ggml_sum_rows(ctx, prod),
                                   1.0f / std::sqrt((float) n_embd));       // [1, n_hc, n_tokens]

    // gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)): a zero dot takes
    // the positive branch, as copysign does (sign = 1 - 2 * (dot < 0)).
    ggml_tensor * mag = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_abs(ctx, dot), 1e-6f, INFINITY));
    ggml_tensor * sign = ggml_scale_bias(ctx, ggml_step(ctx, ggml_neg(ctx, dot)), -2.0f, 1.0f);
    ggml_tensor * gate = ggml_sigmoid(ctx, ggml_mul(ctx, sign, mag));
    if (gate_out) *gate_out = gate;

    // h_c += gate_c * value
    ggml_tensor * value_rep = ggml_repeat(ctx, ggml_cont(ctx, value), h);    // [n_embd, n_hc, n_tokens]
    return ggml_add(ctx, h, ggml_mul(ctx, value_rep, gate));
}

void DeepSeek4EngramApplyRunner::release() {
    if (alloc_) ggml_gallocr_free(alloc_);
    alloc_ = nullptr;
    alloc_backend_ = nullptr;
}

bool DeepSeek4EngramApplyRunner::run(ggml_backend_t backend, const DeepSeek4Layer & L, int n_embd, int n_hc,
                                     float rms_eps, float * hc, const float * keys, int n_tokens) {
    return hc && run_chunks(backend, L, n_embd, n_hc, rms_eps, hc, nullptr, keys, n_tokens);
}

bool DeepSeek4EngramApplyRunner::run_device(ggml_backend_t backend, const DeepSeek4Layer & L, int n_embd,
                                            int n_hc, float rms_eps, ggml_tensor * hc_dev,
                                            const float * keys, int n_tokens) {
    if (!hc_dev || hc_dev->type != GGML_TYPE_F32 || !ggml_is_contiguous(hc_dev) ||
        ggml_nelements(hc_dev) != (int64_t) n_embd * n_hc * n_tokens) {
        return false;
    }
    return run_chunks(backend, L, n_embd, n_hc, rms_eps, nullptr, hc_dev, keys, n_tokens);
}

// The residual comes from and returns to host memory (hc_host) or is
// updated in place on the device (hc_dev); only the keys upload then.
bool DeepSeek4EngramApplyRunner::run_chunks(ggml_backend_t backend, const DeepSeek4Layer & L, int n_embd,
                                            int n_hc, float rms_eps, float * hc_host, ggml_tensor * hc_dev,
                                            const float * keys, int n_tokens) {
    if (!backend || !L.engram_wkv || n_tokens <= 0) return false;
    if (alloc_backend_ != backend) {
        release();
        alloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!alloc_) return false;
        alloc_backend_ = backend;
    }
    const int64_t key_width = L.engram_wkv->ne[0];
    const size_t hc_width = (size_t) n_embd * n_hc;
    if (meta_.empty()) meta_.resize(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
    for (int first = 0; first < n_tokens; first += kMaxTokens) {
        const int count = std::min(kMaxTokens, n_tokens - first);
        ggml_init_params params{};
        params.mem_size = meta_.size();
        params.mem_buffer = meta_.data();
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        if (!ctx) return false;
        ggml_tensor * h = hc_dev
            ? ggml_view_3d(ctx, hc_dev, n_embd, n_hc, count, sizeof(float) * (size_t) n_embd,
                           sizeof(float) * hc_width, (size_t) first * sizeof(float) * hc_width)
            : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_hc, count);
        ggml_tensor * k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, key_width, count);
        if (!hc_dev) ggml_set_input(h);
        ggml_set_input(k);
        ggml_tensor * out = deepseek4_build_engram_apply(ctx, h, k, L, n_embd, n_hc, rms_eps);
        if (!out) { ggml_free(ctx); return false; }
        if (hc_dev) out = ggml_cpy(ctx, out, h);
        ggml_set_output(out);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(gf, out);
        bool ok = ggml_gallocr_alloc_graph(alloc_, gf);
        if (ok) {
            if (!hc_dev) {
                ggml_backend_tensor_set(h, hc_host + (size_t) first * hc_width, 0, sizeof(float) * hc_width * count);
            }
            ggml_backend_tensor_set(k, keys + (size_t) first * key_width, 0, sizeof(float) * key_width * count);
            ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        }
        if (ok && !hc_dev) {
            ggml_backend_tensor_get(out, hc_host + (size_t) first * hc_width, 0, sizeof(float) * hc_width * count);
        }
        ggml_free(ctx);
        if (!ok) return false;
    }
    return true;
}

static uint64_t elapsed_us(std::chrono::steady_clock::time_point t0) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();
}

// ─── Engram on the host hyper-connection paths ──────────────────────────
// A step reads the Engram rows of its tokens once, for every Engram layer;
// each Engram layer then updates the residual copies entering it, before its
// attention HC pre (model.py Transformer.forward), on the GPU that holds its
// weights. `ctx` is the sequence's n-gram context (deepseek4_engram.h).
bool ds4_engram_read_keys(const DeepSeek4Weights & w, DeepSeek4EngramTokens & ctx,
                          const int32_t * token_ids, int kv_start, int n_tokens,
                          std::vector<float> & keys, DeepSeek4StepTelemetry * telemetry) {
    const DeepSeek4EngramRuntime * engram = w.engram_runtime.get();
    if (!engram) return true;
    if (!token_ids) {
        std::fprintf(stderr, "[deepseek4] engram: the step carries no token ids\n");
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    keys.resize((size_t) engram->n_layers() * (size_t) n_tokens * engram->key_floats());
    std::string err;
    if (!engram->prepare(ctx, token_ids, kv_start, (size_t) n_tokens, keys.data(), &err)) {
        std::fprintf(stderr, "[deepseek4] %s\n", err.c_str());
        return false;
    }
    if (telemetry) telemetry->engram_read_us += elapsed_us(t0);
    return true;
}

bool ds4_engram_apply_host(ggml_backend_t backend, const DeepSeek4Weights & w, int il,
                           const std::vector<float> & keys, float * hc_state, int n_tokens,
                           DeepSeek4EngramApplyRunner & runner,
                           DeepSeek4StepTelemetry * telemetry) {
    const DeepSeek4EngramRuntime * engram = w.engram_runtime.get();
    const int e = engram ? engram->layer_index(il) : -1;
    if (e < 0) return true;
    const auto t0 = std::chrono::steady_clock::now();
    const float * layer_keys = keys.data() + (size_t) e * (size_t) n_tokens * engram->key_floats();
    if (!runner.run(backend, w.layers[(size_t) il], w.n_embd, w.n_hc, w.rms_eps,
                    hc_state, layer_keys, n_tokens)) {
        std::fprintf(stderr, "[deepseek4] engram apply failed at layer %d\n", il);
        return false;
    }
    if (telemetry) telemetry->engram_apply_us += elapsed_us(t0);
    return true;
}

bool ds4_engram_apply_device(ggml_backend_t backend, const DeepSeek4Weights & w, int il,
                             const std::vector<float> & keys, ggml_tensor * hc_dev, int n_tokens,
                             DeepSeek4EngramApplyRunner & runner,
                             DeepSeek4StepTelemetry * telemetry) {
    const DeepSeek4EngramRuntime * engram = w.engram_runtime.get();
    const int e = engram ? engram->layer_index(il) : -1;
    if (e < 0) return true;
    const auto t0 = std::chrono::steady_clock::now();
    const float * layer_keys = keys.data() + (size_t) e * (size_t) n_tokens * engram->key_floats();
    if (!runner.run_device(backend, w.layers[(size_t) il], w.n_embd, w.n_hc, w.rms_eps,
                           hc_dev, layer_keys, n_tokens)) {
        std::fprintf(stderr, "[deepseek4] engram device apply failed at layer %d\n", il);
        return false;
    }
    if (telemetry) telemetry->engram_apply_us += elapsed_us(t0);
    return true;
}

}  // namespace luce::common
