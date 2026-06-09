// DeepSeek V4 Flash ggml compute graph builder.
//
// Implements the full forward pass using ggml ops:
//   1. HC pre (Sinkhorn-normalized residual stream mixing)
//   2. MLA attention (low-rank Q, single KV head, grouped output)
//   3. KV compression (learned gate+kv pooling, RoPE on compressed rows)
//   4. Indexer (top-k selective attention over compressed KV)
//   5. HC post (update residual streams)
//   6. MoE FFN (hash routing + top-k + shared expert + clamped SwiGLU)

#include "deepseek4_internal.h"
#include "internal.h"
#include "../common/moe_hybrid_ffn_eval.h"
#include "../common/moe_hybrid_types.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace dflash::common {

struct DeepSeek4I32InputBinding {
    ggml_tensor * tensor = nullptr;
    int32_t       value  = 0;
};

// ─── Helper: RMSNorm ────────────────────────────────────────────────────

static ggml_tensor * build_rms_norm(ggml_context * ctx, ggml_tensor * x,
                                     ggml_tensor * weight, float eps) {
    ggml_tensor * normed = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, normed, weight);
}

// ─── Helper: Clamped SwiGLU ─────────────────────────────────────────────

static ggml_tensor * build_clamped_swiglu(ggml_context * ctx,
                                           ggml_tensor * gate,
                                           ggml_tensor * up,
                                           float clamp) {
    // clamp gate and up to [-clamp, +clamp]
    gate = ggml_clamp(ctx, gate, -clamp, clamp);
    up   = ggml_clamp(ctx, up,   -clamp, clamp);
    // silu(gate) * up
    gate = ggml_silu(ctx, gate);
    return ggml_mul(ctx, gate, up);
}

// ─── Helper: Partial RoPE ───────────────────────────────────────────────
// DS4 applies RoPE only to the last n_rot dimensions of each head.
// For a single KV head of size head_dim with rotation on last n_rot dims,
// we split, apply rope to the tail, and concat back.

static ggml_tensor * build_partial_rope(ggml_context * ctx,
                                         ggml_tensor * x,
                                         int n_rot,
                                         int head_dim,
                                         int n_heads,
                                         int n_tokens,
                                         int position_offset,
                                         float freq_base,
                                         float scale_factor) {
    // x: [head_dim * n_heads, n_tokens] or [head_dim, n_tokens] for KV
    // RoPE is applied to the LAST n_rot dims of each head.
    // ggml_rope applies to the first n_rot dims, so we need to handle the split.
    //
    // For now, we use ggml_rope with mode flags to handle partial rotation.
    // ggml_rope mode=0 rotates first n_rot dims of each head.
    // DS4 rotates the TAIL, so we'd need mode=GGML_ROPE_TYPE_NEOX style or manual split.
    //
    // TODO: Implement exact DS4 tail-rotation. For initial correctness,
    // use ggml_rope with appropriate mode that handles DS4's convention.
    // The GGUF should encode the rope style appropriately.

    (void)head_dim; (void)n_heads; (void)scale_factor;

    // Placeholder: apply standard rope (will need adjustment for DS4's tail convention)
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    return ggml_rope_ext(ctx, x, positions, nullptr,
                         n_rot, 2 /* NEOX mode */,
                         0 /* context size (unused) */,
                         freq_base, 1.0f /* ext_factor */,
                         0.0f, 0.0f, 0.0f, 0.0f);
}

// ─── KV Compressor Step ────────────────────────────────────────────────

static void build_compressor_step(
        ggml_context * ctx,
        ggml_cgraph * gf,
        ggml_tensor * cur_last,      // [n_embd, 1]
        ggml_tensor * ape,
        ggml_tensor * kv_proj,
        ggml_tensor * gate_proj,
        ggml_tensor * norm_weight,
        DeepSeek4CompressorState & state,
        ggml_tensor * comp_cache,
        int ratio,
        int comp_width,
        int token_pos,
        int n_rot,
        float rms_eps,
        float compress_rope_freq_base,
        std::vector<DeepSeek4I32InputBinding> & i32_inputs) {
    if (!gf || !cur_last || !ape || !kv_proj || !gate_proj || !norm_weight ||
        !state.state_kv || !state.state_score || !comp_cache || ratio <= 0) {
        return;
    }

    const int slot = token_pos % ratio;

    // DS4 compression mirrors ds4.c::compressor_decode_one():
    //   1. Project the current post-attn-norm hidden state into value content
    //      and gating/score spaces.
    //   2. Add the learned absolute-position bias for the slot within the
    //      rolling compression window.
    //   3. Store both vectors into rolling state.
    //   4. On window boundaries, pool the entire window with a per-dimension
    //      softmax, RMSNorm the pooled row, RoPE it, and append to comp_cache.
    ggml_tensor * kv_cur = ggml_mul_mat(ctx, kv_proj, cur_last);
    ggml_tensor * sc_cur = ggml_mul_mat(ctx, gate_proj, cur_last);

    ggml_tensor * ape_col = ggml_view_2d(
        ctx, ape, comp_width, 1, ape->nb[1], (size_t)slot * ape->nb[1]);
    sc_cur = ggml_add(ctx, sc_cur, ape_col);

    ggml_tensor * kv_slot = ggml_view_2d(
        ctx, state.state_kv, comp_width, 1, state.state_kv->nb[1],
        (size_t)slot * state.state_kv->nb[1]);
    ggml_tensor * sc_slot = ggml_view_2d(
        ctx, state.state_score, comp_width, 1, state.state_score->nb[1],
        (size_t)slot * state.state_score->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, kv_cur, kv_slot));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, sc_cur, sc_slot));

    if (((token_pos + 1) % ratio) != 0) {
        return;
    }

    ggml_tensor * score_t = ggml_cont(ctx, ggml_transpose(ctx, state.state_score));
    ggml_tensor * weights_t = ggml_soft_max(ctx, score_t);
    ggml_tensor * weights = ggml_transpose(ctx, weights_t);
    ggml_tensor * weighted = ggml_mul(ctx, state.state_kv, weights);
    ggml_tensor * pooled = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, weighted)));
    pooled = ggml_reshape_2d(ctx, pooled, comp_width, 1);
    pooled = build_rms_norm(ctx, pooled, norm_weight, rms_eps);

    // The compressed row gets its own RoPE frequency base. We materialize the
    // single compressed position as a tiny graph input so the boundary path can
    // stay inside ggml even though the absolute position is decided CPU-side.
    ggml_tensor * comp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    i32_inputs.push_back({comp_pos, token_pos / ratio});
    pooled = ggml_rope_ext(ctx, pooled, comp_pos, nullptr,
                           n_rot, GGML_ROPE_TYPE_NEOX, 0,
                           compress_rope_freq_base, 1.0f,
                           0.0f, 0.0f, 0.0f, 0.0f);

    ggml_tensor * pooled_f16 = ggml_cast(ctx, pooled, GGML_TYPE_F16);
    const int comp_row = token_pos / ratio;
    if (comp_row >= (int) comp_cache->ne[1]) {
        return;
    }

    ggml_tensor * comp_slot = ggml_view_2d(
        ctx, comp_cache, comp_width, 1, comp_cache->nb[1],
        (size_t)comp_row * comp_cache->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, pooled_f16, comp_slot));
}

static void build_indexer_compressor_step(
        ggml_context * ctx,
        ggml_cgraph * gf,
        ggml_tensor * cur_last,
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L,
        DeepSeek4LayerCache & lc,
        int token_pos,
        std::vector<DeepSeek4I32InputBinding> & i32_inputs) {
    const int index_comp_width = w.n_indexer_head * w.n_indexer_head_dim;
    build_compressor_step(ctx, gf, cur_last,
                          L.indexer_compressor_ape,
                          L.indexer_compressor_kv,
                          L.indexer_compressor_gate,
                          L.indexer_compressor_norm,
                          lc.indexer_compressor,
                          lc.index_comp_kv,
                          4,
                          index_comp_width,
                          token_pos,
                          w.n_indexer_head_dim,
                          w.rms_eps,
                          w.compress_rope_freq_base,
                          i32_inputs);
}

static int ds4_comp_rows_used(const ggml_tensor * comp_cache, int n_cached, int ratio, int token_pos) {
    if (!comp_cache || ratio <= 0) {
        return 0;
    }
    const int grew_this_step = ((token_pos + 1) % ratio) == 0 ? 1 : 0;
    return std::min(n_cached + grew_this_step, (int) comp_cache->ne[1]);
}

static ggml_tensor * build_indexer_score(
        ggml_context * ctx,
        ggml_tensor * qr_norm_last,   // [n_lora_q, 1]
        ggml_tensor * cur_last,       // [n_embd, 1]
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L,
        const DeepSeek4LayerCache & lc,
        int token_pos,
        std::vector<DeepSeek4I32InputBinding> & i32_inputs) {
    const int n_comp = ds4_comp_rows_used(lc.index_comp_kv, lc.n_index_comp, 4, token_pos);
    if (!qr_norm_last || !cur_last || !L.indexer_attn_q_b || !L.indexer_proj ||
        !lc.index_comp_kv || n_comp <= 0) {
        return nullptr;
    }

    const int n_indexer_head = w.n_indexer_head;
    const int head_dim = w.n_indexer_head_dim;
    const int index_comp_width = n_indexer_head * head_dim;

    // DS4 indexer decode scoring mirrors ds4.c::indexer_allowed_decode_one():
    //   1. Build an indexer query from qr_norm (after q_a + RMSNorm, before q_b).
    //   2. Apply full-dim RoPE in indexer head space.
    //   3. Project per-head scalar weights from the current hidden state.
    //   4. Score every compressed row with ReLU(dot(key_h, query_h)) * weight_h.
    //   5. Return the top-k compressed-row indices.
    ggml_tensor * index_q = ggml_mul_mat(ctx, L.indexer_attn_q_b, qr_norm_last);
    index_q = ggml_reshape_3d(ctx, index_q, head_dim, n_indexer_head, 1);

    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    i32_inputs.push_back({pos, token_pos});
    index_q = ggml_rope_ext(ctx, index_q, pos, nullptr,
                            head_dim, GGML_ROPE_TYPE_NEOX, 0,
                            w.rope_freq_base, 1.0f,
                            0.0f, 0.0f, 0.0f, 0.0f);

    ggml_tensor * head_weights = ggml_mul_mat(ctx, L.indexer_proj, cur_last);
    head_weights = ggml_scale(ctx, head_weights,
                              1.0f / std::sqrt((float) head_dim * (float) n_indexer_head));

    ggml_tensor * comp_view = ggml_view_2d(ctx, lc.index_comp_kv,
                                           index_comp_width, n_comp,
                                           lc.index_comp_kv->nb[1], 0);
    comp_view = ggml_cast(ctx, comp_view, GGML_TYPE_F32);
    comp_view = ggml_reshape_3d(ctx, comp_view, head_dim, n_indexer_head, n_comp);

    ggml_tensor * q_rep = ggml_repeat(ctx, index_q, comp_view);
    ggml_tensor * dots = ggml_mul(ctx, comp_view, q_rep);
    dots = ggml_sum_rows(ctx, dots);
    dots = ggml_cont(ctx, dots);
    dots = ggml_reshape_2d(ctx, dots, n_indexer_head, n_comp);
    dots = ggml_relu(ctx, dots);

    ggml_tensor * weight_rep = ggml_repeat(ctx, head_weights, dots);
    ggml_tensor * weighted = ggml_mul(ctx, dots, weight_rep);
    ggml_tensor * scores = ggml_sum_rows(ctx, weighted);
    scores = ggml_cont(ctx, scores);
    scores = ggml_reshape_2d(ctx, scores, n_comp, 1);

    return ggml_top_k(ctx, scores, std::min(n_comp, w.n_indexer_top_k));
}

static ggml_tensor * build_selected_comp_context(
        ggml_context * ctx,
        ggml_tensor * selected_rows,  // [head_dim, n_selected]
        ggml_tensor * query_seed,     // [head_dim, 1]
        ggml_tensor * q_template,     // [head_dim, n_head, n_tokens]
        int head_dim) {
    if (!selected_rows || !query_seed || !q_template || selected_rows->ne[1] <= 0) {
        return nullptr;
    }

    ggml_tensor * score = ggml_mul_mat(ctx, selected_rows, query_seed);
    ggml_tensor * probs = ggml_soft_max(ctx, score);
    ggml_tensor * rows_t = ggml_cont(ctx, ggml_transpose(ctx, selected_rows));
    ggml_tensor * context = ggml_mul_mat(ctx, rows_t, probs);
    context = ggml_reshape_3d(ctx, context, head_dim, 1, 1);
    return ggml_repeat(ctx, context, q_template);
}

// ─── MLA Attention Block ────────────────────────────────────────────────

static ggml_tensor * build_mla_attention(
        ggml_context * ctx,
        ggml_cgraph * gf,
        ggml_tensor * cur,           // [n_embd, n_tokens]
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L,
        DeepSeek4LayerCache & lc,
        int layer_idx,
        int kv_start,
        int n_tokens,
        std::vector<DeepSeek4I32InputBinding> & i32_inputs) {

    const int n_embd    = w.n_embd;
    const int head_dim  = w.head_dim;
    const int n_head    = w.n_head;
    const int n_lora_q  = w.n_lora_q;
    const int n_rot     = w.n_rot;
    const int n_out_group = w.n_out_group;
    const int n_lora_o  = w.n_lora_o;
    const int ratio     = w.compress_ratios[layer_idx];

    // ── Q path: cur → q_a → norm → q_b → per-head norm ─────────────
    // q_a: [n_embd, n_tokens] → [n_lora_q, n_tokens]
    ggml_tensor * qr = ggml_mul_mat(ctx, L.attn_q_a, cur);
    // qr_norm is reused by the ratio-4 indexer before the main q_b projection.
    qr = build_rms_norm(ctx, qr, L.attn_q_a_norm, w.rms_eps);
    // q_b: [n_lora_q, n_tokens] → [n_head * head_dim, n_tokens]
    ggml_tensor * q = ggml_mul_mat(ctx, L.attn_q_b, qr);
    // Reshape to [head_dim, n_head, n_tokens] for per-head ops
    q = ggml_reshape_3d(ctx, q, head_dim, n_head, n_tokens);

    // ── KV path: cur → kv → norm ───────────────────────────────────
    // kv: [n_embd, n_tokens] → [head_dim, n_tokens]
    ggml_tensor * kv = ggml_mul_mat(ctx, L.attn_kv, cur);
    kv = build_rms_norm(ctx, kv, L.attn_kv_a_norm, w.rms_eps);

    // ── RoPE on Q and KV (partial rotation on tail dims) ────────────
    // TODO: Apply partial RoPE correctly (tail n_rot dims)
    // For now, this is a placeholder that marks where RoPE goes.
    (void)n_rot;

    // ── Store newest KV row in the raw SWA ring ─────────────────────
    const int token_pos = kv_start + n_tokens - 1;
    ggml_tensor * kv_last = ggml_view_2d(
        ctx, kv, head_dim, 1, kv->nb[1], (size_t)(n_tokens - 1) * kv->nb[1]);
    ggml_tensor * kv_slot = ggml_view_2d(
        ctx, lc.raw_kv, head_dim, 1, lc.raw_kv->nb[1],
        (size_t)(token_pos % w.n_swa) * lc.raw_kv->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_cast(ctx, kv_last, GGML_TYPE_F16), kv_slot));

    // ── Learned compression update ──────────────────────────────────
    ggml_tensor * cur_last = ggml_view_2d(
        ctx, cur, n_embd, 1, cur->nb[1], (size_t)(n_tokens - 1) * cur->nb[1]);
    ggml_tensor * qr_last = ggml_view_2d(
        ctx, qr, n_lora_q, 1, qr->nb[1], (size_t)(n_tokens - 1) * qr->nb[1]);
    build_compressor_step(ctx, gf, cur_last,
                          L.attn_compressor_ape,
                          L.attn_compressor_kv,
                          L.attn_compressor_gate,
                          L.attn_compressor_norm,
                          lc.attn_compressor,
                          lc.comp_kv,
                          ratio,
                          head_dim,
                          token_pos,
                          w.n_rot,
                          w.rms_eps,
                          w.compress_rope_freq_base,
                          i32_inputs);

    ggml_tensor * allowed_comp = nullptr;
    if (ratio == 4) {
        build_indexer_compressor_step(ctx, gf, cur_last, w, L, lc, token_pos, i32_inputs);
        allowed_comp = build_indexer_score(ctx, qr_last, cur_last, w, L, lc, token_pos, i32_inputs);
    }

    // ── Attention: placeholder dense path + DS4 selective compressed context ──
    // The full MLA kernel is still stubbed, but ratio-4 layers now follow the
    // DS4 indexer flow: maintain an indexer-specific compressed cache, score all
    // compressed rows, take top-k, and only build compressed context from the
    // allowed rows.
    ggml_tensor * attn_out = ggml_mul_mat(ctx, kv, q);  // Existing dense placeholder

    if (n_tokens == 1 && ratio > 0 && lc.comp_kv) {
        const int n_comp_used = ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio, token_pos);
        if (n_comp_used > 0) {
            ggml_tensor * comp_rows = ggml_view_2d(ctx, lc.comp_kv,
                                                   head_dim, n_comp_used,
                                                   lc.comp_kv->nb[1], 0);
            if (ratio == 4 && allowed_comp) {
                comp_rows = ggml_get_rows(ctx, comp_rows, allowed_comp);
            }
            ggml_tensor * comp_ctx = build_selected_comp_context(ctx, ggml_cast(ctx, comp_rows, GGML_TYPE_F32),
                                                                 kv_last, q, head_dim);
            if (comp_ctx) {
                attn_out = ggml_add(ctx, attn_out, comp_ctx);
            }
        }
    }

    // ── Grouped output projection ──────────────────────────────────
    // attn_out: [head_dim * n_head, n_tokens]
    // → grouped A: [head_dim * (n_head/n_out_group), n_tokens] per group → [n_lora_o, n_tokens]
    // → B: [n_lora_o, n_tokens] → [n_embd, n_tokens]
    attn_out = ggml_reshape_2d(ctx, attn_out, head_dim * n_head, n_tokens);
    ggml_tensor * attn_low = ggml_mul_mat(ctx, L.attn_output_a, attn_out);
    ggml_tensor * out = ggml_mul_mat(ctx, L.attn_output_b, attn_low);

    (void)n_out_group; (void)n_lora_o; (void)n_embd; (void)n_lora_q;
    return out;
}

// ─── MoE FFN Block ──────────────────────────────────────────────────────

struct Ds4MoeRouting {
    ggml_tensor * selected = nullptr;
    ggml_tensor * weights = nullptr;
};

static MoeHybridConfig make_ds4_moe_hybrid_config(const DeepSeek4Weights & w) {
    MoeHybridConfig cfg;
    cfg.n_embd = w.n_embd;
    cfg.n_expert = w.n_expert;
    cfg.n_expert_used = w.n_expert_used;
    cfg.n_ff_exp = w.n_ff_exp;
    cfg.n_ff_shexp = w.n_ff_exp;
    cfg.n_layer = w.n_layer;
    cfg.first_moe_layer = 0;
    return cfg;
}

static MoeLayerDesc make_ds4_moe_layer_desc(const DeepSeek4Layer & L) {
    MoeLayerDesc desc;
    desc.ffn_gate_exps = L.ffn_gate_exps;
    desc.ffn_up_exps = L.ffn_up_exps;
    desc.ffn_down_exps = L.ffn_down_exps;
    desc.ffn_gate_up_exps = nullptr;
    desc.ffn_gate_shexp = L.ffn_gate_shexp;
    desc.ffn_up_shexp = L.ffn_up_shexp;
    desc.ffn_down_shexp = L.ffn_down_shexp;
    desc.ffn_gate_inp_shexp = nullptr;
    return desc;
}

static ggml_tensor * build_shared_ffn(
        ggml_context * ctx,
        ggml_tensor * cur,
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L) {
    ggml_tensor * gate_sh = ggml_mul_mat(ctx, L.ffn_gate_shexp, cur);
    ggml_tensor * up_sh = ggml_mul_mat(ctx, L.ffn_up_shexp, cur);
    ggml_tensor * mid_sh = build_clamped_swiglu(ctx, gate_sh, up_sh, w.swiglu_clamp_exp);
    return ggml_mul_mat(ctx, L.ffn_down_shexp, mid_sh);
}

static Ds4MoeRouting build_moe_routing(
        ggml_context * ctx,
        ggml_tensor * cur,
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L,
        int n_tokens) {
    Ds4MoeRouting out;
    ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, cur);

    // DS4 routes with sqrt(softplus(logit)). Optional bias affects only the
    // top-k expert selection, while expert weights come from the unbiased
    // router probabilities and are normalized after selection.
    ggml_tensor * probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));
    ggml_tensor * selection = probs;
    if (L.ffn_exp_probs_b) {
        selection = ggml_add(ctx, selection, L.ffn_exp_probs_b);
    }

    out.selected = ggml_top_k(ctx, selection, w.n_expert_used);
    ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, w.n_expert, n_tokens);
    out.weights = ggml_get_rows(ctx, probs_3d, out.selected);
    out.weights = ggml_reshape_2d(ctx, out.weights, w.n_expert_used, n_tokens);

    ggml_tensor * w_sum = ggml_sum_rows(ctx, out.weights);
    w_sum = ggml_clamp(ctx, w_sum, 6.103515625e-5f, INFINITY);
    out.weights = ggml_div(ctx, out.weights, w_sum);
    if (w.expert_weight_scale != 1.0f) {
        out.weights = ggml_scale(ctx, out.weights, w.expert_weight_scale);
    }
    return out;
}

static ggml_tensor * build_moe_ffn(
        ggml_context * ctx,
        ggml_tensor * cur,
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L,
        int layer_idx,
        int n_tokens) {

    const int n_embd = w.n_embd;
    const int n_used = w.n_expert_used;
    const int n_ff_exp = w.n_ff_exp;
    ggml_tensor * shared_out = build_shared_ffn(ctx, cur, w, L);
    ggml_tensor * routed_out = nullptr;

    if (layer_idx < w.n_hash_layer && L.ffn_gate_tid2eid) {
        routed_out = ggml_scale(ctx, cur, 0.0f);
    } else {
        Ds4MoeRouting routing = build_moe_routing(ctx, cur, w, L, n_tokens);
        ggml_tensor * cur_3d = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
        ggml_tensor * gate_e = ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur_3d, routing.selected);
        ggml_tensor * up_e = ggml_mul_mat_id(ctx, L.ffn_up_exps, cur_3d, routing.selected);

        gate_e = ggml_reshape_3d(ctx, gate_e, n_ff_exp, n_used, n_tokens);
        up_e = ggml_reshape_3d(ctx, up_e, n_ff_exp, n_used, n_tokens);
        ggml_tensor * mid_e = build_clamped_swiglu(ctx, gate_e, up_e, w.swiglu_clamp_exp);

        ggml_tensor * down_e = ggml_mul_mat_id(ctx, L.ffn_down_exps, mid_e, routing.selected);
        down_e = ggml_reshape_3d(ctx, down_e, n_embd, n_used, n_tokens);

        ggml_tensor * weights_3d = ggml_reshape_3d(ctx, routing.weights, 1, n_used, n_tokens);
        routed_out = ggml_mul(ctx, down_e, weights_3d);
        routed_out = ggml_sum_rows(ctx, routed_out);
        routed_out = ggml_reshape_2d(ctx, routed_out, n_embd, n_tokens);
    }

    return ggml_add(ctx, shared_out, routed_out);
}

// ─── HC (Hierarchical Controller) Pre ───────────────────────────────────
// Mixes n_hc residual streams into a single working vector via Sinkhorn.

static ggml_tensor * build_hc_pre(
        ggml_context * ctx,
        ggml_tensor * hc_state,      // [n_hc * n_embd] persistent residual
        const DeepSeek4Weights & w,
        ggml_tensor * hc_fn,         // [n_hc * n_embd, hc_mix_dim]
        ggml_tensor * hc_scale,      // [3]
        ggml_tensor * hc_base,       // [n_hc]
        int n_tokens) {

    const int n_embd = w.n_embd;
    const int n_hc   = w.n_hc;
    (void)n_tokens;

    // RMSNorm over each HC stream independently
    ggml_tensor * flat = ggml_rms_norm(ctx, hc_state, w.hc_eps);

    // Mix projection: flat → [hc_mix_dim]
    // hc_mix_dim = 2*n_hc + n_hc*n_hc (pre weights + post gates + combine matrix)
    ggml_tensor * mix = ggml_mul_mat(ctx, hc_fn, flat);

    // Split mix into: pre_logits [n_hc], post_logits [n_hc], comb_logits [n_hc*n_hc]
    // Then:
    //   pre_weights = sigmoid(pre_logits * pre_scale + base) + eps
    //   post_gates  = 2 * sigmoid(post_logits * post_scale)
    //   combine     = sinkhorn(reshape(comb_logits * comb_scale, [n_hc, n_hc]))
    //
    // Output = weighted sum of HC streams: Σ pre[i] * hc_state[i*n_embd : (i+1)*n_embd]

    // Placeholder: return first HC stream as the working vector
    // Full Sinkhorn implementation will be added
    ggml_tensor * out = ggml_view_1d(ctx, hc_state, n_embd, 0);

    (void)mix; (void)hc_scale; (void)hc_base; (void)n_hc;
    return out;
}

static bool deepseek4_step_hybrid(
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        DeepSeek4Cache & cache,
        MoeHybridStorage & moe_hybrid,
        const float * embed,
        int n_tokens,
        int kv_start,
        std::vector<float> & out_logits) {
    const int n_embd = w.n_embd;
    std::vector<float> cur(embed, embed + (size_t) n_embd * (size_t) n_tokens);
    ggml_backend_t cpu_backend = moe_hybrid.cpu_backend;
    ggml_gallocr_t hot_alloc = nullptr;
    ggml_gallocr_t cold_alloc = nullptr;

    for (int il = 0; il < w.n_layer; ++il) {
        const DeepSeek4Layer & L = w.layers[(size_t) il];
        DeepSeek4LayerCache & lc = cache.layers[(size_t) il];
        const size_t ctx_size = 48 * 1024 * 1024;
        ggml_init_params params{};
        params.mem_size = ctx_size;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            if (hot_alloc) ggml_gallocr_free(hot_alloc);
            if (cold_alloc) ggml_gallocr_free(cold_alloc);
            return false;
        }

        ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp);
        ggml_tensor * cur_tensor = inp;
        std::vector<DeepSeek4I32InputBinding> i32_inputs;
        ggml_cgraph * gf = ggml_new_graph(ctx);

        ggml_tensor * attn_in = cur_tensor;
        if (L.hc_attn_fn && cache.hc_state) {
            attn_in = build_hc_pre(ctx, cache.hc_state, w,
                                   L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base,
                                   n_tokens);
        }
        ggml_tensor * normed = build_rms_norm(ctx, attn_in, L.attn_norm, w.rms_eps);
        ggml_tensor * attn_out = build_mla_attention(ctx, gf, normed, w, L, lc, il,
                                                     kv_start, n_tokens, i32_inputs);
        ggml_tensor * residual = ggml_add(ctx, cur_tensor, attn_out);

        ggml_tensor * ffn_in = residual;
        if (L.hc_ffn_fn && cache.hc_state) {
            ffn_in = build_hc_pre(ctx, cache.hc_state, w,
                                  L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base,
                                  n_tokens);
        }
        ggml_tensor * ffn_post = build_rms_norm(ctx, ffn_in, L.ffn_norm, w.rms_eps);

        if (il < w.n_hash_layer && L.ffn_gate_tid2eid) {
            ggml_tensor * ffn_out = build_shared_ffn(ctx, ffn_post, w, L);
            ggml_tensor * next = ggml_add(ctx, residual, ffn_out);
            ggml_build_forward_expand(gf, next);
            ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(alloc, gf)) {
                ggml_gallocr_free(alloc);
                ggml_free(ctx);
                if (hot_alloc) ggml_gallocr_free(hot_alloc);
                if (cold_alloc) ggml_gallocr_free(cold_alloc);
                return false;
            }
            ggml_backend_tensor_set(inp, cur.data(), 0, sizeof(float) * cur.size());
            for (const DeepSeek4I32InputBinding & binding : i32_inputs) {
                ggml_backend_tensor_set(binding.tensor, &binding.value, 0, sizeof(binding.value));
            }
            const bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
            if (ok) {
                ggml_backend_tensor_get(next, cur.data(), 0, sizeof(float) * cur.size());
            }
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            if (!ok) {
                if (hot_alloc) ggml_gallocr_free(hot_alloc);
                if (cold_alloc) ggml_gallocr_free(cold_alloc);
                return false;
            }
            continue;
        }

        Ds4MoeRouting routing = build_moe_routing(ctx, ffn_post, w, L, n_tokens);
        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, ffn_post);
        ggml_build_forward_expand(gf, routing.selected);
        ggml_build_forward_expand(gf, routing.weights);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            if (hot_alloc) ggml_gallocr_free(hot_alloc);
            if (cold_alloc) ggml_gallocr_free(cold_alloc);
            return false;
        }

        ggml_backend_tensor_set(inp, cur.data(), 0, sizeof(float) * cur.size());
        for (const DeepSeek4I32InputBinding & binding : i32_inputs) {
            ggml_backend_tensor_set(binding.tensor, &binding.value, 0, sizeof(binding.value));
        }
        const bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        if (!ok) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            if (hot_alloc) ggml_gallocr_free(hot_alloc);
            if (cold_alloc) ggml_gallocr_free(cold_alloc);
            return false;
        }

        std::vector<float> residual_host((size_t) n_embd * (size_t) n_tokens);
        std::vector<float> ffn_post_host((size_t) n_embd * (size_t) n_tokens);
        std::vector<int32_t> selected_host((size_t) w.n_expert_used * (size_t) n_tokens);
        std::vector<float> weights_host((size_t) w.n_expert_used * (size_t) n_tokens);
        ggml_backend_tensor_get(residual, residual_host.data(), 0, sizeof(float) * residual_host.size());
        ggml_backend_tensor_get(ffn_post, ffn_post_host.data(), 0, sizeof(float) * ffn_post_host.size());
        ggml_backend_tensor_get(routing.selected, selected_host.data(), 0, sizeof(int32_t) * selected_host.size());
        ggml_backend_tensor_get(routing.weights, weights_host.data(), 0, sizeof(float) * weights_host.size());
        ggml_gallocr_free(alloc);
        ggml_free(ctx);

        std::vector<float> ffn_out_host;
        MoeHybridConfig hybrid_cfg = make_ds4_moe_hybrid_config(w);
        MoeLayerDesc desc = make_ds4_moe_layer_desc(L);
        auto & storage = moe_hybrid.layers[(size_t) il];
        bool ffn_ok = eval_moe_hybrid_ffn_batched(
            backend, cpu_backend, hybrid_cfg, desc, storage,
            ffn_post_host.data(), selected_host.data(), weights_host.data(),
            n_tokens, ffn_out_host, nullptr, &hot_alloc, &cold_alloc);
        if (!ffn_ok) {
            ffn_out_host.assign((size_t) n_embd * (size_t) n_tokens, 0.0f);
            std::vector<float> single_out;
            for (int ti = 0; ti < n_tokens; ++ti) {
                if (!eval_moe_hybrid_ffn_single(
                        backend, hybrid_cfg, desc, storage, cpu_backend,
                        ffn_post_host.data() + (size_t) ti * (size_t) n_embd,
                        selected_host.data() + (size_t) ti * (size_t) w.n_expert_used,
                        weights_host.data() + (size_t) ti * (size_t) w.n_expert_used,
                        w.n_expert_used, single_out)) {
                    if (hot_alloc) ggml_gallocr_free(hot_alloc);
                    if (cold_alloc) ggml_gallocr_free(cold_alloc);
                    return false;
                }
                std::memcpy(ffn_out_host.data() + (size_t) ti * (size_t) n_embd,
                            single_out.data(), sizeof(float) * (size_t) n_embd);
            }
        }

        cur.resize(residual_host.size());
        for (size_t i = 0; i < cur.size(); ++i) {
            cur[i] = residual_host[i] + ffn_out_host[i];
        }
    }

    if (hot_alloc) ggml_gallocr_free(hot_alloc);
    if (cold_alloc) ggml_gallocr_free(cold_alloc);

    const size_t final_ctx_size = 16 * 1024 * 1024;
    ggml_init_params params{};
    params.mem_size = final_ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp);
    ggml_tensor * cur_tensor = inp;
    if (w.output_hc_fn && cache.hc_state) {
        cur_tensor = build_hc_pre(ctx, cache.hc_state, w,
                                  w.output_hc_fn, w.output_hc_scale, w.output_hc_base,
                                  n_tokens);
    }
    cur_tensor = build_rms_norm(ctx, cur_tensor, w.out_norm, w.rms_eps);
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, cur_tensor);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, logits);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(inp, cur.data(), 0, sizeof(float) * cur.size());
    const bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    if (ok) {
        out_logits.resize((size_t) w.n_vocab);
        const size_t logits_offset = (size_t) (n_tokens - 1) * (size_t) w.n_vocab * sizeof(float);
        ggml_backend_tensor_get(logits, out_logits.data(), logits_offset,
                                sizeof(float) * (size_t) w.n_vocab);
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    if (!ok) return false;

    cache.cur_pos = kv_start + n_tokens;
    return true;
}

// ─── Full forward step ──────────────────────────────────────────────────

bool deepseek4_step(
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        DeepSeek4Cache & cache,
        const float * embed,
        int n_tokens,
        int kv_start,
        std::vector<float> & out_logits,
        MoeHybridStorage * moe_hybrid) {

    if (w.moe_hybrid && moe_hybrid != nullptr) {
        return deepseek4_step_hybrid(backend, w, cache, *moe_hybrid,
                                     embed, n_tokens, kv_start, out_logits);
    }

    const int n_embd = w.n_embd;
    const int n_layer = w.n_layer;

    // Create compute graph context
    const size_t ctx_size = ggml_tensor_overhead() * 4096 + 1024 * 1024;
    ggml_init_params params{};
    params.mem_size = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    // Input embeddings
    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(inp, "inp_embed");
    ggml_set_input(inp);

    ggml_tensor * cur = inp;
    ggml_cgraph * gf = ggml_new_graph(ctx);
    std::vector<DeepSeek4I32InputBinding> i32_inputs;

    // Layer loop
    for (int il = 0; il < n_layer; il++) {
        const DeepSeek4Layer & L = w.layers[il];
        DeepSeek4LayerCache & lc = cache.layers[il];

        // ── HC pre (attention) ──────────────────────────────────────
        // TODO: Full HC implementation. For now, pass cur through directly.
        ggml_tensor * attn_in = cur;
        if (L.hc_attn_fn && cache.hc_state) {
            attn_in = build_hc_pre(ctx, cache.hc_state, w,
                                    L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base,
                                    n_tokens);
        }

        // ── Attention norm ──────────────────────────────────────────
        ggml_tensor * normed = build_rms_norm(ctx, attn_in, L.attn_norm, w.rms_eps);

        // ── MLA attention ───────────────────────────────────────────
        ggml_tensor * attn_out = build_mla_attention(ctx, gf, normed, w, L, lc,
                                                      il, kv_start, n_tokens,
                                                      i32_inputs);

        // ── Residual ────────────────────────────────────────────────
        cur = ggml_add(ctx, cur, attn_out);

        // ── HC pre (FFN) ────────────────────────────────────────────
        ggml_tensor * ffn_in = cur;
        if (L.hc_ffn_fn && cache.hc_state) {
            ffn_in = build_hc_pre(ctx, cache.hc_state, w,
                                   L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base,
                                   n_tokens);
        }

        // ── FFN norm ────────────────────────────────────────────────
        ggml_tensor * ffn_normed = build_rms_norm(ctx, ffn_in, L.ffn_norm, w.rms_eps);

        // ── MoE FFN ─────────────────────────────────────────────────
        ggml_tensor * ffn_out = build_moe_ffn(ctx, ffn_normed, w, L, il, n_tokens);

        // ── Residual ────────────────────────────────────────────────
        cur = ggml_add(ctx, cur, ffn_out);
    }

    // ── Output head ─────────────────────────────────────────────────────
    // HC output pre (merge residual streams for final projection)
    if (w.output_hc_fn && cache.hc_state) {
        cur = build_hc_pre(ctx, cache.hc_state, w,
                            w.output_hc_fn, w.output_hc_scale, w.output_hc_base,
                            n_tokens);
    }

    // Final RMSNorm
    cur = build_rms_norm(ctx, cur, w.out_norm, w.rms_eps);

    // lm_head projection
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    // ── Build and run graph ─────────────────────────────────────────────
    ggml_build_forward_expand(gf, logits);

    // Allocate
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "[deepseek4] graph allocation failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    // Set input data
    ggml_backend_tensor_set(inp, embed, 0, n_embd * n_tokens * sizeof(float));
    for (const DeepSeek4I32InputBinding & binding : i32_inputs) {
        ggml_backend_tensor_set(binding.tensor, &binding.value, 0, sizeof(binding.value));
    }

    // Compute
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[deepseek4] graph compute failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    // Read logits (only last token for generation)
    out_logits.resize(w.n_vocab);
    const size_t logits_offset = (size_t)(n_tokens - 1) * w.n_vocab * sizeof(float);
    ggml_backend_tensor_get(logits, out_logits.data(), logits_offset,
                            w.n_vocab * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    const int next_pos = kv_start + n_tokens;
    for (int il = 0; il < n_layer; ++il) {
        const uint32_t ratio = w.compress_ratios[il];
        if (ratio <= 0 || (next_pos % (int) ratio) != 0) {
            continue;
        }
        cache.layers[il].n_comp = std::max(cache.layers[il].n_comp, next_pos / (int) ratio);
        if (ratio == 4) {
            cache.layers[il].n_index_comp = std::max(cache.layers[il].n_index_comp,
                                                     next_pos / (int) ratio);
        }
    }

    cache.cur_pos = next_pos;
    return true;
}

// ─── Cache management ───────────────────────────────────────────────────

bool create_deepseek4_cache(ggml_backend_t backend,
                             const DeepSeek4Weights & w,
                             int max_ctx,
                             DeepSeek4Cache & out) {
    out.n_layer = w.n_layer;
    out.max_ctx = max_ctx;
    out.cur_pos = 0;
    out.layers.resize(w.n_layer);

    ggml_init_params ctx_params{};
    ctx_params.mem_size = ggml_tensor_overhead() * (size_t)(w.n_layer * 9 + 8) + 4096;
    ctx_params.no_alloc = true;
    out.ctx = ggml_init(ctx_params);
    if (!out.ctx) {
        return false;
    }

    for (int il = 0; il < w.n_layer; ++il) {
        DeepSeek4LayerCache & lc = out.layers[il];
        const uint32_t ratio = w.compress_ratios[il];

        lc.raw_kv = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F16, w.head_dim, w.n_swa);
        char name[64];
        std::snprintf(name, sizeof(name), "ds4_raw_kv_%d", il);
        ggml_set_name(lc.raw_kv, name);

        lc.n_comp = 0;
        lc.n_index_comp = 0;

        if (ratio <= 0) {
            continue;
        }

        const int comp_cap = max_ctx / (int) ratio + 16;
        lc.comp_kv = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F16, w.head_dim, comp_cap);
        std::snprintf(name, sizeof(name), "ds4_comp_kv_%d", il);
        ggml_set_name(lc.comp_kv, name);

        lc.attn_compressor.state_kv = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, w.head_dim, ratio);
        lc.attn_compressor.state_score = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, w.head_dim, ratio);
        std::snprintf(name, sizeof(name), "ds4_comp_state_kv_%d", il);
        ggml_set_name(lc.attn_compressor.state_kv, name);
        std::snprintf(name, sizeof(name), "ds4_comp_state_score_%d", il);
        ggml_set_name(lc.attn_compressor.state_score, name);

        if (ratio == 4) {
            const int index_comp_width = w.n_indexer_head * w.n_indexer_head_dim;
            lc.index_comp_kv = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F16,
                                                  index_comp_width, comp_cap);
            lc.indexer_compressor.state_kv = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32,
                                                                index_comp_width, ratio);
            lc.indexer_compressor.state_score = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32,
                                                                   index_comp_width, ratio);
            std::snprintf(name, sizeof(name), "ds4_index_comp_kv_%d", il);
            ggml_set_name(lc.index_comp_kv, name);
            std::snprintf(name, sizeof(name), "ds4_index_state_kv_%d", il);
            ggml_set_name(lc.indexer_compressor.state_kv, name);
            std::snprintf(name, sizeof(name), "ds4_index_state_score_%d", il);
            ggml_set_name(lc.indexer_compressor.state_score, name);
        }
    }

    out.hc_state = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, (int64_t)w.n_hc * w.n_embd);
    ggml_set_name(out.hc_state, "ds4_hc_state");

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    ggml_backend_buffer_clear(out.buf, 0);
    const size_t total_bytes = ggml_backend_buffer_get_size(out.buf);
    std::fprintf(stderr, "[deepseek4] KV cache: %.1f MB for ctx=%d\n",
                 (double)total_bytes / (1024.0 * 1024.0), max_ctx);
    return true;
}

void free_deepseek4_cache(DeepSeek4Cache & c) {
    if (c.ctx) { ggml_free(c.ctx); c.ctx = nullptr; }
    if (c.buf) { ggml_backend_buffer_free(c.buf); c.buf = nullptr; }
    c.layers.clear();
    c.hc_state = nullptr;
}

void free_deepseek4_snapshot(DeepSeek4Snapshot & s) {
    if (s.ctx) { ggml_free(s.ctx); s.ctx = nullptr; }
    if (s.buf) { ggml_backend_buffer_free(s.buf); s.buf = nullptr; }
    s.layers.clear();
    s.cur_pos = 0;
    s.hc_state_snap = nullptr;
}

}  // namespace dflash::common
