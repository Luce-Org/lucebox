// GLM-5.3-Flash forward graph (reference: upstream llama.cpp glm5-next.cpp).
// KDA and MLA follow bailingmoe3, mHC follows deepseek4, the SwiGLU clamp is ggml_swiglu_ds4_split.
// DSA: a query keeps the top_k/kpool best complete pools plus its own unfinished pool, as a mask.

#include "glm5_graph.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace luce::common {

namespace {

// ~25.6k nodes for 45 layers.
constexpr int kGraphNodes = 1 << 16;

ggml_tensor * rms_norm_mul(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    x = ggml_rms_norm(ctx, x, eps);
    return w ? ggml_mul(ctx, x, w) : x;
}

// ── mHC ─────────────────────────────────────────────────────────────────

struct HcPre {
    ggml_tensor * cur  = nullptr;  // [n_embd, T]    stream mix fed to the sub-block
    ggml_tensor * post = nullptr;  // [hc, T]        weight of the sub-block output per stream
    ggml_tensor * comb = nullptr;  // [hc(dst), hc(src), T]
    ggml_tensor * xt   = nullptr;  // [hc, n_embd, T] the residual, streams first
    // Fused path.
    ggml_tensor * split = nullptr;  // [(2+hc)*hc, T] pre|post|comb
    ggml_tensor * x2d   = nullptr;  // [hc*n_embd, T]
};

// DS4 HC kernels (GPU only); GLM5_HC_UNFUSED=1 uses the reference ops.
bool use_fused_hc(const Glm5Weights & w) {
    static const bool off = std::getenv("GLM5_HC_UNFUSED") != nullptr;
    return w.hc_fused && !off && w.hc_mult == 4 && std::fabs(w.hc_eps - 1e-6f) < 1e-12f;
}

// comb [dst, src, T]: softmax over dst, then alternating row/column normalization.
ggml_tensor * hc_sinkhorn(ggml_context * ctx, ggml_tensor * comb, int iters, float eps) {
    comb = ggml_soft_max(ctx, comb);
    comb = ggml_scale_bias(ctx, comb, 1.0f, eps);

    auto norm_cols = [&]() {
        ggml_tensor * t = ggml_cont(ctx, ggml_permute(ctx, comb, 1, 0, 2, 3));   // [src, dst, T]
        ggml_tensor * s = ggml_scale_bias(ctx, ggml_sum_rows(ctx, t), 1.0f, eps); // [1, dst, T]
        s = ggml_cont(ctx, ggml_permute(ctx, s, 1, 0, 2, 3));                     // [dst, 1, T]
        comb = ggml_div(ctx, comb, s);
    };
    auto norm_rows = [&]() {
        ggml_tensor * s = ggml_scale_bias(ctx, ggml_sum_rows(ctx, comb), 1.0f, eps); // [1, src, T]
        comb = ggml_div(ctx, comb, s);
    };

    norm_cols();
    for (int i = 1; i < iters; ++i) {
        norm_rows();
        norm_cols();
    }
    return comb;
}

HcPre build_hc_pre(ggml_context * ctx, const Glm5Weights & w, ggml_tensor * x,
                   ggml_tensor * fn, ggml_tensor * base, const float scale[3]) {
    const int hc = w.hc_mult;
    const int64_t E = w.n_embd;
    const int64_t T = x->ne[2];

    ggml_tensor * flat = ggml_reshape_2d(ctx, x, hc * E, T);
    ggml_tensor * mixes = ggml_mul_mat(ctx, fn, ggml_rms_norm(ctx, flat, w.rms_eps));  // [(2+hc)*hc, T]

    if (use_fused_hc(w)) {
        const int64_t mix_dim = (2 + hc) * hc;
        ggml_tensor * out = ggml_ds4_hc_pre(ctx, mixes, base, flat, hc, w.hc_iters,
                                            scale[0], scale[1], scale[2]);
        const size_t row = (size_t) (E + mix_dim) * sizeof(float);
        HcPre r;
        r.cur = ggml_cont(ctx, ggml_view_2d(ctx, out, E, T, row, 0));
        r.split = ggml_view_2d(ctx, out, mix_dim, T, row, (size_t) E * sizeof(float));
        r.x2d = flat;
        return r;
    }

    auto part = [&](int off, int n) {
        return ggml_cont(ctx, ggml_view_2d(ctx, mixes, n, T, mixes->nb[1],
                                           (size_t) off * ggml_element_size(mixes)));
    };
    auto base_part = [&](int off, int n) {
        return ggml_view_1d(ctx, base, n, (size_t) off * ggml_element_size(base));
    };

    HcPre r;
    ggml_tensor * pre = ggml_add(ctx, ggml_scale(ctx, part(0, hc), scale[0]), base_part(0, hc));
    pre = ggml_scale_bias(ctx, ggml_sigmoid(ctx, pre), 1.0f, w.hc_eps);

    ggml_tensor * post = ggml_add(ctx, ggml_scale(ctx, part(hc, hc), scale[1]), base_part(hc, hc));
    r.post = ggml_scale(ctx, ggml_sigmoid(ctx, post), 2.0f);

    ggml_tensor * comb = ggml_add(ctx, ggml_scale(ctx, part(2 * hc, hc * hc), scale[2]),
                                  base_part(2 * hc, hc * hc));
    r.comb = hc_sinkhorn(ctx, ggml_reshape_3d(ctx, comb, hc, hc, T), w.hc_iters, w.hc_eps);

    r.xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));  // [hc, E, T]
    ggml_tensor * cur = ggml_mul_mat(ctx, r.xt, ggml_reshape_3d(ctx, pre, hc, 1, T));  // [E, 1, T]
    r.cur = ggml_reshape_2d(ctx, cur, E, T);
    return r;
}

// out[:, dst] = f * post[dst] + sum_src residual[:, src] * comb[dst, src]
ggml_tensor * build_hc_post(ggml_context * ctx, const Glm5Weights & w, ggml_tensor * f,
                            const HcPre & pre) {
    const int hc = w.hc_mult;
    const int64_t E = w.n_embd;
    const int64_t T = f->ne[1];

    if (pre.split) {
        return ggml_reshape_3d(ctx, ggml_ds4_hc_post(ctx, pre.x2d, f, pre.split, hc), E, hc, T);
    }

    ggml_tensor * comb_t = ggml_cont(ctx, ggml_permute(ctx, pre.comb, 1, 0, 2, 3));  // [src, dst, T]
    ggml_tensor * mixed = ggml_mul_mat(ctx, pre.xt, comb_t);                         // [E, dst, T]
    ggml_tensor * f3 = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, f, E, 1, T), E, hc, T, 1);
    ggml_tensor * fp = ggml_mul(ctx, f3, ggml_reshape_3d(ctx, pre.post, 1, hc, T));
    return ggml_add(ctx, mixed, fp);
}

// ── KDA ─────────────────────────────────────────────────────────────────

// Causal conv over q, k or v; keeps the last d_conv-1 inputs as state.
ggml_tensor * kda_conv(ggml_context * ctx, ggml_cgraph * gf, const Glm5Weights & w,
                       ggml_tensor * all_state, int qkv, ggml_tensor * x,
                       ggml_tensor * proj, ggml_tensor * conv_w) {
    const int d_conv = w.ssm_d_conv;
    const int d_inner = w.d_inner();
    const int64_t T = x->ne[1];
    const size_t elem = ggml_element_size(all_state);

    ggml_tensor * state = ggml_view_3d(ctx, all_state, d_conv - 1, d_inner, 1,
                                       all_state->nb[1], all_state->nb[1] * d_inner,
                                       (size_t) qkv * d_inner * all_state->nb[1]);
    ggml_tensor * projected = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, proj, x), d_inner, T, 1);
    ggml_tensor * input = ggml_concat(ctx, state, ggml_transpose(ctx, projected), 0);  // [d_conv-1+T, d_inner, 1]

    ggml_tensor * last = ggml_view_3d(ctx, input, d_conv - 1, d_inner, 1,
                                      input->nb[1], input->nb[2], (size_t) T * elem);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, last, state));

    ggml_tensor * out = ggml_ssm_conv(ctx, input, ggml_reshape_2d(ctx, conv_w, d_conv, d_inner));
    out = ggml_silu(ctx, ggml_reshape_2d(ctx, out, d_inner, T));
    return ggml_reshape_4d(ctx, out, w.kda_head_dim, w.n_head, T, 1);
}

ggml_tensor * build_kda(ggml_context * ctx, ggml_cgraph * gf, const Glm5Weights & w,
                        const Glm5Layer & L, const Glm5LayerCache & C, ggml_tensor * x) {
    const int hd = w.kda_head_dim;
    const int H = w.n_head;
    const int d_inner = w.d_inner();
    const int64_t T = x->ne[1];

    ggml_tensor * q = kda_conv(ctx, gf, w, C.conv, 0, x, L.wq, L.conv_q);
    ggml_tensor * k = kda_conv(ctx, gf, w, C.conv, 1, x, L.wk, L.conv_k);
    ggml_tensor * v = kda_conv(ctx, gf, w, C.conv, 2, x, L.wv, L.conv_v);

    // log decay = lower_bound * sigmoid(-g*A), A = -exp(A_log)
    ggml_tensor * g = ggml_mul_mat(ctx, L.ssm_f_b, ggml_mul_mat(ctx, L.ssm_f_a, x));
    g = ggml_add(ctx, g, L.ssm_dt_b);
    g = ggml_mul(ctx, ggml_reshape_3d(ctx, g, hd, H, T), ggml_reshape_3d(ctx, L.ssm_a, 1, H, 1));
    g = ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, g, -1.0f)), w.kda_gate_lower_bound);
    g = ggml_reshape_4d(ctx, g, hd, H, T, 1);

    ggml_tensor * beta = ggml_mul_mat(ctx, L.ssm_beta, x);
    beta = ggml_sigmoid(ctx, ggml_reshape_4d(ctx, beta, 1, H, T, 1));

    q = ggml_l2_norm(ctx, q, 1e-6f);
    k = ggml_l2_norm(ctx, k, 1e-6f);

    ggml_tensor * state = ggml_reshape_4d(ctx, C.ssm, hd, hd, H, 1);
    ggml_tensor * packed = ggml_gated_delta_net(ctx, q, k, v, g, beta, state);
    ggml_gated_delta_net_set_skip_intermediate(packed, true);

    const size_t elem = ggml_element_size(packed);
    ggml_tensor * out = ggml_view_4d(ctx, packed, hd, H, T, 1,
                                     (size_t) hd * elem, (size_t) hd * H * elem,
                                     (size_t) hd * H * T * elem, 0);
    ggml_tensor * new_state = ggml_view_4d(ctx, packed, hd, hd, H, 1,
                                           (size_t) hd * elem, (size_t) hd * hd * elem,
                                           (size_t) hd * hd * H * elem,
                                           (size_t) hd * H * T * elem);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, state));

    // RMSNorm(o) * sigmoid(g2)
    ggml_tensor * g2 = ggml_mul_mat(ctx, L.ssm_g_b, ggml_mul_mat(ctx, L.ssm_g_a, x));
    g2 = ggml_reshape_3d(ctx, g2, hd, H, T);
    out = ggml_reshape_3d(ctx, out, hd, H, T);
    out = rms_norm_mul(ctx, out, L.ssm_norm, w.rms_eps);
    out = ggml_mul(ctx, out, ggml_sigmoid(ctx, g2));
    out = ggml_cont_2d(ctx, out, d_inner, T);
    return ggml_mul_mat(ctx, L.wo, out);
}

// ── DSA indexer ─────────────────────────────────────────────────────────

// Cache rows [lo, kv_start) then the step's own rows: never reads rows it writes.
ggml_tensor * rows_with_current(ggml_context * ctx, ggml_tensor * cache, int lo, int kv_start,
                                ggml_tensor * cur, int n_cur) {
    ggml_tensor * old = nullptr;
    if (kv_start > lo) {
        old = ggml_view_2d(ctx, cache, cache->ne[0], kv_start - lo, cache->nb[1],
                           (size_t) lo * cache->nb[1]);
    }
    ggml_tensor * now = nullptr;
    if (n_cur > 0) {
        now = ggml_view_2d(ctx, cur, cur->ne[0], n_cur, cur->nb[1], 0);
    }
    if (old && now) return ggml_concat(ctx, old, now, 1);
    return old ? old : now;
}

// Caches indexer keys, gates and new pooled keys; returns the F16 mask when sparse.
ggml_tensor * build_dsa(ggml_context * ctx, ggml_cgraph * gf, const Glm5Weights & w,
                        const Glm5Layer & L, const Glm5LayerCache & C, const Glm5StepGraph & g,
                        ggml_tensor * x, ggml_tensor * qr) {
    const int g_layer = (int) (&L - w.layers.data());
    const int kpool = w.idx_kpool;
    const int D = w.idx_dim;
    const int T = g.n_tokens;
    const int kv_start = g.kv_start;
    const int kv_end = kv_start + T;

    ggml_tensor * ik = ggml_norm(ctx, ggml_mul_mat(ctx, L.idx_k, x), w.norm_eps);
    ik = ggml_add(ctx, ggml_mul(ctx, ik, L.idx_k_norm), L.idx_k_norm_b);   // [D, T]
    ggml_tensor * ig = ggml_mul_mat(ctx, L.idx_gate, x);                    // [D, T]
    ggml_build_forward_expand(gf, ggml_cpy(ctx, ik,
        ggml_view_2d(ctx, C.idx_k, D, T, C.idx_k->nb[1], (size_t) kv_start * C.idx_k->nb[1])));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, ig,
        ggml_view_2d(ctx, C.idx_g, D, T, C.idx_g->nb[1], (size_t) kv_start * C.idx_g->nb[1])));

    // Pools completed by this step: p in [p_lo, p_lo + n_new).
    const int p_lo = kv_start / kpool;
    const int n_new = kv_end / kpool - p_lo;
    ggml_tensor * pooled_new = nullptr;
    if (n_new > 0) {
        const int row_lo = p_lo * kpool;
        const int n_cur = (p_lo + n_new) * kpool - kv_start;
        ggml_tensor * kr = rows_with_current(ctx, C.idx_k, row_lo, kv_start, ik, n_cur);
        ggml_tensor * gr = rows_with_current(ctx, C.idx_g, row_lo, kv_start, ig, n_cur);
        kr = ggml_cont(ctx, kr);
        gr = ggml_cont(ctx, gr);

        // softmax over the pool members of (gate + ape), per channel
        ggml_tensor * logit = ggml_add(ctx, ggml_reshape_3d(ctx, gr, D, kpool, n_new), L.idx_ape);
        logit = ggml_cont(ctx, ggml_permute(ctx, logit, 1, 0, 2, 3));          // [kpool, D, n_new]
        ggml_tensor * prob = ggml_soft_max(ctx, ggml_reshape_2d(ctx, logit, kpool, (int64_t) D * n_new));
        prob = ggml_reshape_3d(ctx, prob, kpool, D, n_new);
        ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_3d(ctx, kr, D, kpool, n_new), 1, 0, 2, 3));            // [kpool, D, n_new]
        pooled_new = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, prob, kp)), D, n_new);
        ggml_format_name(pooled_new, "glm5_dsa_pooled_new-%d", g_layer);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, pooled_new,
            ggml_view_2d(ctx, C.pooled, D, n_new, C.pooled->nb[1], (size_t) p_lo * C.pooled->nb[1])));
    }

    if (!g.sparse) return nullptr;

    ggml_tensor * keep = g.tail;  // [kv_pad, T], 1 = attend
    const int n_pool = g.n_pool;
    if (n_pool > 0) {
        ggml_tensor * pooled = rows_with_current(ctx, C.pooled, 0, p_lo, pooled_new, n_new);  // [D, n_pool]
        pooled = ggml_cont(ctx, pooled);

        const int n_ih = w.idx_n_head;
        // F32 scores: near ties must not flip.
        ggml_tensor * iq = ggml_mul_mat(ctx, L.idx_q_b, qr);                                    // [D*n_ih, T]
        ggml_mul_mat_set_prec(iq, GGML_PREC_F32);
        ggml_tensor * wl = ggml_mul_mat(ctx, L.idx_proj, x);
        ggml_mul_mat_set_prec(wl, GGML_PREC_F32);
        ggml_tensor * wts = ggml_scale(ctx, wl, 1.0f / std::sqrt((float) D * n_ih));             // [n_ih, T]
        ggml_tensor * kq = ggml_mul_mat(ctx, pooled, ggml_reshape_2d(ctx, iq, D, (int64_t) n_ih * T));
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_relu(ctx, ggml_reshape_3d(ctx, kq, n_pool, n_ih, T));
        kq = ggml_mul(ctx, kq, ggml_reshape_3d(ctx, wts, 1, n_ih, T));
        ggml_tensor * score = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_permute(ctx, kq, 1, 0, 2, 3)));
        score = ggml_add(ctx, ggml_reshape_2d(ctx, score, n_pool, T), g.pool_bias);              // [n_pool, T]
        ggml_format_name(score, "glm5_dsa_score-%d", g_layer);

        const int n_top = std::min(n_pool, w.idx_top_k / kpool);
        ggml_tensor * top = ggml_top_k(ctx, score, n_top);                                       // [n_top, T]

        const int n_slot = g.kv_pad / kpool;
        ggml_tensor * sel = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_slot, T), 0.0f);
        ggml_tensor * ones = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_top, T), 1.0f);
        sel = ggml_set_rows(ctx, sel, ones, top);
        // drop picks of invisible pools
        sel = ggml_mul(ctx, sel, g.pool_vis);
        ggml_tensor * cells = ggml_reshape_2d(ctx, ggml_repeat_4d(ctx, sel, kpool, n_slot, T, 1),
                                              g.kv_pad, T);
        keep = ggml_add(ctx, cells, keep);
    }
    ggml_format_name(keep, "glm5_dsa_keep-%d", g_layer);
    // keep is 0/1, so log gives 0/-inf.
    return ggml_cast(ctx, ggml_log(ctx, keep), GGML_TYPE_F16);
}

// ── MLA ─────────────────────────────────────────────────────────────────

ggml_tensor * build_mla(ggml_context * ctx, ggml_cgraph * gf, const Glm5Weights & w,
                        const Glm5Layer & L, const Glm5LayerCache & C, const Glm5StepGraph & g,
                        ggml_tensor * x) {
    const int H = w.n_head;
    const int64_t T = g.n_tokens;

    ggml_tensor * qr = rms_norm_mul(ctx, ggml_mul_mat(ctx, L.q_a, x), L.q_a_norm, w.rms_eps);
    ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.q_b, qr), w.qk_head, H, T);

    ggml_tensor * kv = rms_norm_mul(ctx, ggml_mul_mat(ctx, L.kv_a, x), L.kv_a_norm, w.rms_eps);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, kv,
        ggml_view_2d(ctx, C.latent, w.kv_lora, T, C.latent->nb[1],
                     (size_t) g.kv_start * C.latent->nb[1])));

    ggml_tensor * mask = build_dsa(ctx, gf, w, L, C, g, x, qr);
    if (!mask) mask = g.causal_mask;

    // Absorb k_b: [qk_head, T, H] -> [kv_lora, T, H].
    ggml_tensor * q_abs = ggml_mul_mat(ctx, L.k_b, ggml_permute(ctx, q, 0, 2, 1, 3));

    ggml_tensor * kv_read = ggml_view_3d(ctx, C.latent, w.kv_lora, g.kv_pad, 1,
                                         C.latent->nb[1], C.latent->nb[1] * C.latent->ne[1], 0);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, q_abs, kv_read, kv_read, mask,
                                             1.0f / std::sqrt((float) w.qk_head), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);

    // [kv_lora, H, T] -> per-head v_b expansion in [kv_lora, T, H].
    attn = ggml_mul_mat(ctx, L.v_b, ggml_permute(ctx, attn, 0, 2, 1, 3));    // [v_head, T, H]
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));              // [v_head, H, T]
    attn = ggml_reshape_2d(ctx, attn, (int64_t) w.v_head * H, T);
    return ggml_mul_mat(ctx, L.wo, attn);
}

// ── FFN ─────────────────────────────────────────────────────────────────

ggml_tensor * build_swiglu(ggml_context * ctx, ggml_tensor * x, ggml_tensor * gate_w,
                           ggml_tensor * up_w, ggml_tensor * down_w, float clamp) {
    ggml_tensor * gate = ggml_mul_mat(ctx, gate_w, x);
    ggml_tensor * up = ggml_mul_mat(ctx, up_w, x);
    ggml_tensor * act = clamp > 1e-6f ? ggml_swiglu_ds4_split(ctx, gate, up, clamp)
                                      : ggml_swiglu_split(ctx, gate, up);
    return ggml_mul_mat(ctx, down_w, act);
}

ggml_tensor * build_moe(ggml_context * ctx, const Glm5Weights & w, const Glm5Layer & L,
                        ggml_tensor * x) {
    const int64_t E = w.n_embd;
    const int64_t T = x->ne[1];
    const int k = w.n_expert_used;

    ggml_tensor * probs = ggml_sigmoid(ctx, ggml_mul_mat(ctx, L.ffn_gate_inp, x));   // [n_expert, T]
    // The bias affects selection only.
    ggml_tensor * selected = ggml_argsort_top_k(ctx, ggml_add(ctx, probs, L.ffn_exp_bias), k);  // [k, T]
    ggml_tensor * weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, w.n_expert, T), selected);
    if (w.expert_norm) {
        ggml_tensor * w2 = ggml_reshape_2d(ctx, weights, k, T);
        ggml_tensor * sum = ggml_clamp(ctx, ggml_sum_rows(ctx, w2), 6.103515625e-5f, INFINITY);
        weights = ggml_reshape_3d(ctx, ggml_div(ctx, w2, sum), 1, k, T);
    }
    if (w.expert_scale != 1.0f) weights = ggml_scale(ctx, weights, w.expert_scale);

    ggml_tensor * xin = ggml_reshape_3d(ctx, x, E, 1, T);
    ggml_tensor * up = ggml_mul_mat_id(ctx, L.ffn_up_exps, xin, selected);       // [n_ff_exp, k, T]
    ggml_tensor * gate = ggml_mul_mat_id(ctx, L.ffn_gate_exps, xin, selected);
    ggml_tensor * act = L.swiglu_clamp_exp > 1e-6f
        ? ggml_swiglu_ds4_split(ctx, gate, up, L.swiglu_clamp_exp)
        : ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * down = ggml_mul_mat_id(ctx, L.ffn_down_exps, act, selected);   // [E, k, T]
    down = ggml_mul(ctx, down, weights);

    ggml_tensor * moe = ggml_view_2d(ctx, down, E, T, down->nb[2], 0);
    for (int i = 1; i < k; ++i) {
        moe = ggml_add(ctx, moe, ggml_view_2d(ctx, down, E, T, down->nb[2], (size_t) i * down->nb[1]));
    }

    ggml_tensor * shared = build_swiglu(ctx, x, L.ffn_gate_shexp, L.ffn_up_shexp, L.ffn_down_shexp,
                                        L.swiglu_clamp_shexp);
    return ggml_add(ctx, moe, shared);
}

}  // namespace

void Glm5StepGraph::free() {
    if (ctx) ggml_free(ctx);
    ctx = nullptr;
    gf = nullptr;
    tokens = causal_mask = tail = pool_vis = pool_bias = logits = nullptr;
}

bool glm5_step_is_sparse(const Glm5Weights & w, int kv_end) {
    return kv_end / w.idx_kpool > w.idx_top_k / w.idx_kpool;
}

bool build_glm5_step_graph(const Glm5Weights & w, const Glm5Cache & cache,
                           int n_tokens, int kv_start, bool all_logits,
                           Glm5StepGraph & g) {
    g.free();
    const int kv_end = kv_start + n_tokens;
    if (n_tokens <= 0 || kv_end > cache.max_ctx) return false;

    g.n_tokens = n_tokens;
    g.kv_start = kv_start;
    g.kv_pad = std::min(((kv_end + 255) / 256) * 256, cache.max_ctx);
    g.n_pool = kv_end / w.idx_kpool;
    g.sparse = glm5_step_is_sparse(w, kv_end);
    g.all_logits = all_logits;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * kGraphNodes + ggml_graph_overhead_custom(kGraphNodes, false);
    ip.no_alloc = true;
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    ggml_context * ctx = g.ctx;
    g.gf = ggml_new_graph_custom(ctx, kGraphNodes, false);
    ggml_cgraph * gf = g.gf;

    const int T = n_tokens;
    g.tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.tokens, "glm5_tokens");
    ggml_set_input(g.tokens);

    const bool has_mla = w.n_mla_layers() > 0;
    // Only inputs a node reads; the scheduler skips unused ones.
    if (has_mla && !g.sparse) {
        g.causal_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, g.kv_pad, T);
        ggml_set_name(g.causal_mask, "glm5_causal_mask");
        ggml_set_input(g.causal_mask);
    }
    if (has_mla && g.sparse) {
        g.tail = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, g.kv_pad, T);
        ggml_set_name(g.tail, "glm5_tail");
        ggml_set_input(g.tail);
        if (g.n_pool > 0) {
            g.pool_vis = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, g.kv_pad / w.idx_kpool, T);
            ggml_set_name(g.pool_vis, "glm5_pool_vis");
            ggml_set_input(g.pool_vis);
            g.pool_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, g.n_pool, T);
            ggml_set_name(g.pool_bias, "glm5_pool_bias");
            ggml_set_input(g.pool_bias);
        }
    }

    const int hc = w.hc_mult;
    const int64_t E = w.n_embd;
    ggml_tensor * emb = ggml_get_rows(ctx, w.tok_embd, g.tokens);   // [E, T]
    ggml_tensor * x = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, emb, E, 1, T), E, hc, T, 1);

    for (int il = 0; il < w.n_layer; ++il) {
        const Glm5Layer & L = w.layers[il];
        const Glm5LayerCache & C = cache.layers[il];

        HcPre a = build_hc_pre(ctx, w, x, L.hc_attn_fn, L.hc_attn_base, L.hc_attn_scale);
        ggml_tensor * cur = rms_norm_mul(ctx, a.cur, L.attn_norm, w.rms_eps);
        cur = L.is_mla ? build_mla(ctx, gf, w, L, C, g, cur) : build_kda(ctx, gf, w, L, C, cur);
        x = build_hc_post(ctx, w, cur, a);

        HcPre f = build_hc_pre(ctx, w, x, L.hc_ffn_fn, L.hc_ffn_base, L.hc_ffn_scale);
        cur = rms_norm_mul(ctx, f.cur, L.ffn_norm, w.rms_eps);
        cur = L.is_moe ? build_moe(ctx, w, L, cur)
                       : build_swiglu(ctx, cur, L.ffn_gate, L.ffn_up, L.ffn_down, L.swiglu_clamp_shexp);
        x = build_hc_post(ctx, w, cur, f);
    }

    // Mean over streams, for the output rows only.
    if (!all_logits && T > 1) {
        x = ggml_view_3d(ctx, x, E, hc, 1, x->nb[1], x->nb[2], (size_t) (T - 1) * x->nb[2]);
    }
    const int64_t n_out = all_logits ? T : 1;
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));   // [hc, E, n_out]
    ggml_tensor * h = ggml_scale(ctx, ggml_reshape_2d(ctx, ggml_sum_rows(ctx, xt), E, n_out), 1.0f / hc);
    h = rms_norm_mul(ctx, h, w.out_norm, w.rms_eps);
    g.logits = ggml_mul_mat(ctx, w.output, h);
    ggml_set_name(g.logits, "glm5_logits");
    ggml_set_output(g.logits);
    ggml_build_forward_expand(gf, g.logits);
    return true;
}

void glm5_dsa_host_inputs(int kv_start, int n_tokens, int kv_pad, int n_pool,
                          int kpool, bool select_tail, bool sparse,
                          Glm5DsaHostInputs & out) {
    const uint16_t zero_h = ggml_fp32_to_fp16(0.0f);
    const uint16_t ninf_h = ggml_fp32_to_fp16(-INFINITY);
    out.causal_mask.assign((size_t) kv_pad * n_tokens, ninf_h);
    for (int t = 0; t < n_tokens; ++t) {
        const int p = kv_start + t;
        std::fill_n(out.causal_mask.begin() + (size_t) t * kv_pad, p + 1, zero_h);
    }
    if (!sparse) {
        out.tail.clear();
        out.pool_vis.clear();
        out.pool_bias.clear();
        return;
    }

    const int n_slot = kv_pad / kpool;
    out.tail.assign((size_t) kv_pad * n_tokens, 0.0f);
    out.pool_vis.assign((size_t) n_slot * n_tokens, 0.0f);
    out.pool_bias.assign((size_t) n_pool * n_tokens, -INFINITY);
    for (int t = 0; t < n_tokens; ++t) {
        const int p = kv_start + t;
        // complete pools ending at or before p
        const int n_vis = std::min((p + 1) / kpool, n_pool);
        std::fill_n(out.pool_vis.begin() + (size_t) t * n_slot, n_vis, 1.0f);
        std::fill_n(out.pool_bias.begin() + (size_t) t * n_pool, n_vis, 0.0f);
        if (select_tail) {
            const int n_tail = (p + 1) % kpool;
            for (int k = 0; k < n_tail; ++k) {
                out.tail[(size_t) t * kv_pad + (p - k)] = 1.0f;
            }
        }
    }
}

void set_glm5_step_inputs(const Glm5Weights & w, const Glm5StepGraph & g,
                          const int32_t * tokens) {
    ggml_backend_tensor_set(g.tokens, tokens, 0, sizeof(int32_t) * (size_t) g.n_tokens);
    if (!g.causal_mask && !g.tail) return;
    Glm5DsaHostInputs in;
    glm5_dsa_host_inputs(g.kv_start, g.n_tokens, g.kv_pad, g.n_pool, w.idx_kpool,
                         w.idx_select_tail, g.sparse, in);
    if (g.causal_mask) {
        ggml_backend_tensor_set(g.causal_mask, in.causal_mask.data(), 0,
                                in.causal_mask.size() * sizeof(uint16_t));
    }
    if (g.tail) ggml_backend_tensor_set(g.tail, in.tail.data(), 0, in.tail.size() * sizeof(float));
    if (g.pool_vis) {
        ggml_backend_tensor_set(g.pool_vis, in.pool_vis.data(), 0, in.pool_vis.size() * sizeof(float));
    }
    if (g.pool_bias) {
        ggml_backend_tensor_set(g.pool_bias, in.pool_bias.data(), 0, in.pool_bias.size() * sizeof(float));
    }
}

}  // namespace luce::common
