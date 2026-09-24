// DeepSeek V4.1 Engram: init from loaded weights and the ggml apply subgraph.
#include "deepseek4_engram.h"
#include "deepseek4_internal.h"

#include <cmath>
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
    slots_.clear();
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
                                           ggml_tensor ** gate_out) {
    if (!ctx || !h || !keys || !L.engram_wkv || !L.engram_q || !L.engram_k) return nullptr;
    const int64_t n_tokens = keys->ne[1];
    GGML_ASSERT(h->ne[0] == n_embd && h->ne[1] == n_hc && h->ne[2] == n_tokens);
    GGML_ASSERT(L.engram_wkv->ne[0] == keys->ne[0]);
    GGML_ASSERT(L.engram_wkv->ne[1] == (int64_t) n_embd * (n_hc + 1));

    // [n_embd * (n_hc + 1), n_tokens]: the n_hc keys first, the value last.
    ggml_tensor * kv = ggml_mul_mat(ctx, L.engram_wkv, keys);
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

    // gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))
    ggml_tensor * mag = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_abs(ctx, dot), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx, ggml_mul(ctx, ggml_sgn(ctx, dot), mag));
    if (gate_out) *gate_out = gate;

    // h_c += gate_c * value
    ggml_tensor * value_rep = ggml_repeat(ctx, ggml_cont(ctx, value), h);    // [n_embd, n_hc, n_tokens]
    return ggml_add(ctx, h, ggml_mul(ctx, value_rep, gate));
}

}  // namespace luce::common
