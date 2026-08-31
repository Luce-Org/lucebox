// Builds a ggml compute graph for one forward pass of the DFlash draft
// (5-layer Qwen3-flavored block-diffusion model).
//
// Stateless: no KV cache. Each call takes:
//   - noise_embed         [hidden,   q_len, 1]   f32    (target.tok_embd on [last_tok, MASK*15])
//   - target_hidden_cat   [N*hidden, ctx_len, 1] f32    (N target layers concat along features)
//   - positions_q         [q_len]                i32    values [ctx_len..ctx_len+q_len-1]
//   - positions_k         [ctx_len+q_len]        i32    values [0..ctx_len+q_len-1]
//   - causal_mask_swa     [kv_pad, q_len]        f32    (optional; causal mask for SWA layers)
// and returns:
//   - hidden_states       [hidden,   q_len, 1]   f32    (final RMSNorm; NO lm_head here)
//
// The caller projects `hidden_states` through the TARGET's lm_head separately
// (the draft has no lm_head of its own, it shares the target's).
//
// Semantics:
//   - optional per-capture RMSNorm on target_hidden_cat slices
//   - fc @ target_hidden_cat -> rms_norm with hidden_norm -> target_feat
//   - Per layer:
//       h_norm = rms_norm(h) * input_layernorm
//       Q  = wq  @ h_norm   -> per-head q_norm
//       K_ctx/V_ctx = wk/wv @ target_feat
//       K_noi/V_noi = wk/wv @ h_norm
//       K = concat[K_ctx, K_noi]  -> per-head k_norm
//       V = concat[V_ctx, V_noi]
//       RoPE(Q, positions_q); RoPE(K, positions_k)    (NEOX style)
//       attn = flash_attn_ext(Q, K, V, mask, scale)   SWA=causal, full=non-causal
//       optional Laguna XS 2.1 gate: attn *= softplus(attn_gate @ h_norm)
//       h   += wo @ attn
//       h_norm = rms_norm(h) * post_attention_layernorm
//       h   += w_down @ (silu(w_gate @ h_norm) * (w_up @ h_norm))
//   - h = rms_norm(h) * norm

#include "internal.h"
#include "draft_graph.h"

#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <limits>

namespace dflash::common {

// RoPE with the drafter's scaling config. YaRN-trained drafters (e.g. the
// Qwen3.8 DSpark release: factor 32, orig ctx 8192) apply the scaled rotary
// at every position, so plain-RoPE inference silently degrades acceptance.
static ggml_tensor * draft_rope(ggml_context * ctx, ggml_tensor * t,
                                ggml_tensor * positions,
                                const DraftWeights & w) {
    return ggml_rope_ext(ctx, t, positions, /*freq_factors=*/nullptr,
                         w.head_dim, GGML_ROPE_TYPE_NEOX, w.rope_n_ctx_orig,
                         w.rope_theta, w.rope_freq_scale,
                         w.rope_ext_factor, w.rope_attn_factor,
                         w.rope_beta_fast, w.rope_beta_slow);
}

// Feature fusion shared by the legacy one-shot graph and the cached-KV
// builders: optional per-capture RMSNorm slices, fc projection, hidden_norm.
// Row-independent, so it is bit-identical whether run over the full window
// or over appended rows only.
static ggml_tensor * draft_fuse_features(
    ggml_context *       ctx,
    const DraftWeights & w,
    ggml_tensor *        target_hidden_cat,
    int                  n_rows,
    bool                 disable_aux_hidden_norms) {
    const float eps = DFLASH27B_RMS_EPS;
    ggml_tensor * thc = target_hidden_cat;
    if (!disable_aux_hidden_norms && !w.aux_hidden_norms.empty()) {
        ggml_tensor * aux_cat = nullptr;
        const size_t elem_sz = ggml_element_size(target_hidden_cat);
        for (size_t i = 0; i < w.aux_hidden_norms.size(); i++) {
            ggml_tensor * slice = ggml_view_3d(ctx, target_hidden_cat,
                w.n_embd, n_rows, 1,
                target_hidden_cat->nb[1], target_hidden_cat->nb[2],
                i * (size_t)w.n_embd * elem_sz);
            slice = ggml_rms_norm(ctx, slice, eps);
            slice = ggml_mul(ctx, slice, w.aux_hidden_norms[i]);
            aux_cat = aux_cat ? ggml_concat(ctx, aux_cat, slice, 0) : slice;
        }
        thc = aux_cat;
    }
    ggml_tensor * target_feat = ggml_mul_mat(ctx, w.fc, thc);
    target_feat = ggml_rms_norm(ctx, target_feat, eps);
    target_feat = ggml_mul    (ctx, target_feat, w.hidden_norm);
    return target_feat;
}

// ── DFlash 2 grouped dynamic causal conv ────────────────────────────
//
// Two taps over the draft block (positions within the block; the block's
// first slot has no predecessor). For each tap k the coefficient is a
// per-element base kernel plus a per-group dynamic kernel projected from
// the block's normalized hidden state:
//   dyn      = proj @ x_norm                          [2*K*groups, q_len]
//   coef_s_k = base[s][k] (per element) + dyn[s][k] (per group, broadcast)
//   out      = sum_k coef_s_k * shift_k(x)
// s = 0 ("prepare", applied to the sub-block input) or 1 ("finish", applied
// to the sub-block output); both use the dyn computed from the input.
struct DraftDynConv {
    ggml_tensor * dyn = nullptr;   // [2*K*groups, q_len]
};

static DraftDynConv draft_dyn_conv_kernel(ggml_context * ctx,
                                          const DraftConvWeights & cw,
                                          ggml_tensor * x_norm) {
    DraftDynConv dc;
    dc.dyn = ggml_mul_mat(ctx, cw.proj, x_norm);   // [2*K*groups, q_len]
    return dc;
}

static ggml_tensor * draft_dyn_conv_apply(ggml_context *           ctx,
                                          const DraftWeights &     w,
                                          const DraftConvWeights & cw,
                                          const DraftDynConv &     dc,
                                          int                      s,      // 0 = prepare, 1 = finish
                                          ggml_tensor *            x) {    // [hidden, q_len]
    const int64_t hidden = x->ne[0];
    const int64_t q_len  = x->ne[1];
    const int     K      = w.conv_kernel_size;
    const int64_t gs     = w.conv_group_size;
    const int64_t groups = hidden / gs;
    const size_t  e      = ggml_element_size(dc.dyn);

    // Fused single-node path (bit-identical to the expansion below);
    // DFLASH_DYN_CONV_FUSED=0 restores the unfused graph.
    static const bool dyn_conv_fused = []() {
        const char * env = std::getenv("DFLASH_DYN_CONV_FUSED");
        return !(env && env[0] == '0' && env[1] == '\0');
    }();
    if (dyn_conv_fused && x->ne[2] <= 1 && x->ne[3] <= 1 &&
        ggml_is_contiguous(dc.dyn) && ggml_is_contiguous(cw.base)) {
        ggml_tensor * xc = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
        return ggml_dflash_dyn_conv(ctx, xc, cw.base, dc.dyn, s, K, (int)gs);
    }

    ggml_tensor * out = nullptr;
    for (int k = 0; k < K; ++k) {
        // shift_k(x): column l takes x[:, l-k], zero for l < k
        ggml_tensor * xs = x;
        if (k > 0) {
            if (q_len <= k) break;
            ggml_tensor * head = ggml_view_2d(ctx, x, hidden, q_len - k, x->nb[1], 0);
            xs = ggml_pad_ext(ctx, head, 0, 0, k, 0, 0, 0, 0, 0);   // [hidden, q_len]
        }
        // per-group dynamic coefficient for (s, k): rows [(s*K+k)*groups, +groups)
        ggml_tensor * dyn_sk = ggml_view_3d(ctx, dc.dyn, 1, groups, q_len,
                                            e, dc.dyn->nb[1],
                                            (size_t)((s * K + k) * groups) * e);
        ggml_tensor * xs3   = ggml_reshape_3d(ctx, xs, gs, groups, q_len);
        ggml_tensor * dyn3  = ggml_repeat(ctx, dyn_sk, xs3);                // [gs, groups, q_len]
        // per-element base coefficient base[s][k]: [hidden] at offset (s*K+k)*hidden
        ggml_tensor * base_sk = ggml_view_3d(ctx, cw.base, gs, groups, 1,
                                             cw.base->nb[0] * gs, cw.base->nb[0] * hidden,
                                             (size_t)(s * K + k) * cw.base->nb[1]);
        ggml_tensor * coef  = ggml_add(ctx, dyn3, base_sk);                 // broadcast over q_len
        ggml_tensor * term  = ggml_mul(ctx, xs3, coef);
        term = ggml_reshape_2d(ctx, term, hidden, q_len);
        out = out ? ggml_add(ctx, out, term) : term;
    }
    return out;
}

DraftGraphOutputs build_draft_graph(
    ggml_context *            ctx,
    const DraftWeights &      w,
    const DraftGraphInputs &  in) {

    const int q_len    = w.block_size;
    const int ctx_len  = in.ctx_len;
    const int n_head   = w.n_head;
    const int n_kv     = w.n_head_kv;
    const int head_dim = w.head_dim;
    const float eps    = DFLASH27B_RMS_EPS;

    // ── 1. Feature fusion: target_feat = rms_norm(fc @ target_hidden_cat, hidden_norm)
    //    fc:                [5*hidden, hidden]  (ggml: ne[0]=5*hidden, ne[1]=hidden)
    //    target_hidden_cat: [5*hidden, ctx_len, 1]
    //    Result:            [hidden,   ctx_len, 1]
    static const bool disable_aux_hidden_norms =
        std::getenv("DFLASH_DISABLE_DRAFT_AUX_NORMS") != nullptr;
    static const bool disable_attn_gate =
        std::getenv("DFLASH_DISABLE_DRAFT_ATTN_GATE") != nullptr;
    static const bool disable_swa =
        std::getenv("DFLASH_DISABLE_DRAFT_SWA") != nullptr;
    static const bool disable_attn =
        std::getenv("DFLASH_DISABLE_DRAFT_ATTN") != nullptr;
    static const bool disable_ffn =
        std::getenv("DFLASH_DISABLE_DRAFT_FFN") != nullptr;

    ggml_tensor * target_feat = draft_fuse_features(
        ctx, w, in.target_hidden_cat, ctx_len, disable_aux_hidden_norms);
    ggml_set_name(target_feat, "target_feat");

    // ── 2. Decoder layers
    ggml_tensor * h = in.noise_embed;  // [hidden, q_len, 1]

    for (int il = 0; il < w.n_layer; il++) {
        const DraftLayer & L = w.layers[il];
        char probe_name[64];

        // ── SWA: determine effective context for this layer
        const bool layer_is_swa = L.is_swa && !disable_swa;
        const bool use_swa = layer_is_swa && w.swa_window > 0 && ctx_len > w.swa_window;
        const int eff_ctx     = use_swa ? w.swa_window : ctx_len;
        const int eff_total_k = eff_ctx + q_len;
        const int ctx_offset  = use_swa ? (ctx_len - w.swa_window) : 0;

        const bool dyn_conv = w.conv_kernel_size > 0 && L.attn_conv.present() && L.mlp_conv.present();

        if (!disable_attn) {
            // ── 2a. Attention pre-norm
            ggml_tensor * hn = ggml_rms_norm(ctx, h, eps);
            hn = ggml_mul(ctx, hn, L.attn_norm);
            // DFlash 2: dynamic conv "prepare" on the attention input
            DraftDynConv attn_dc;
            if (dyn_conv) {
                attn_dc = draft_dyn_conv_kernel(ctx, L.attn_conv, hn);
                hn = draft_dyn_conv_apply(ctx, w, L.attn_conv, attn_dc, 0, hn);
            }
            std::snprintf(probe_name, sizeof(probe_name), "draft_l%d_hn", il);
            ggml_set_name(hn, probe_name);

            // ── 2b. Q from noise only, then per-head RMSNorm
            //     wq: [hidden, q_dim=4096]
            ggml_tensor * Q = ggml_mul_mat(ctx, L.wq, hn);  // [q_dim, q_len, 1]
            Q = ggml_reshape_3d(ctx, Q, head_dim, n_head, q_len);  // [head_dim, n_head, q_len]
            Q = ggml_rms_norm(ctx, Q, eps);                        // normalize along head_dim
            Q = ggml_mul     (ctx, Q, L.q_norm);                   // broadcast [head_dim]
            std::snprintf(probe_name, sizeof(probe_name), "draft_l%d_Q_norm", il);
            ggml_set_name(Q, probe_name);

            // ── 2c. K and V from target_feat AND noise, then concat along sequence
            //     wk, wv: [hidden, kv_dim=1024]
            //   For SWA layers: window target_feat to last swa_window positions.
            ggml_tensor * tf = target_feat;
            if (use_swa) {
                tf = ggml_view_3d(ctx, target_feat,
                    w.n_embd, eff_ctx, 1,
                    target_feat->nb[1], target_feat->nb[2],
                    target_feat->nb[1] * ctx_offset);
            }
            ggml_tensor * tf_kv = tf;
            if (w.context_kv_layer_norm) {
                tf_kv = ggml_rms_norm(ctx, tf_kv, eps);
                tf_kv = ggml_mul(ctx, tf_kv, L.attn_norm);
            }
            ggml_tensor * Kctx = ggml_mul_mat(ctx, L.wk, tf_kv);  // [kv_dim, eff_ctx, 1]
            ggml_tensor * Kn   = ggml_mul_mat(ctx, L.wk, hn);  // [kv_dim, q_len,   1]
            ggml_tensor * Vctx = ggml_mul_mat(ctx, L.wv, tf_kv);
            ggml_tensor * Vn   = ggml_mul_mat(ctx, L.wv, hn);
            std::snprintf(probe_name, sizeof(probe_name), "draft_l%d_Kctx", il);
            ggml_set_name(Kctx, probe_name);
            std::snprintf(probe_name, sizeof(probe_name), "draft_l%d_Kn", il);
            ggml_set_name(Kn, probe_name);
            std::snprintf(probe_name, sizeof(probe_name), "draft_l%d_Vctx", il);
            ggml_set_name(Vctx, probe_name);
            std::snprintf(probe_name, sizeof(probe_name), "draft_l%d_Vn", il);
            ggml_set_name(Vn, probe_name);

            // concat along ne[1] (sequence) — ggml_concat second arg dim=1
            ggml_tensor * K = ggml_concat(ctx, Kctx, Kn, 1);  // [kv_dim, eff_total_k, 1]
            ggml_tensor * V = ggml_concat(ctx, Vctx, Vn, 1);

            // Per-head k_norm
            K = ggml_reshape_3d(ctx, K, head_dim, n_kv, eff_total_k);
            K = ggml_rms_norm(ctx, K, eps);
            K = ggml_mul     (ctx, K, L.k_norm);
            std::snprintf(probe_name, sizeof(probe_name), "draft_l%d_K_norm", il);
            ggml_set_name(K, probe_name);

            V = ggml_reshape_3d(ctx, V, head_dim, n_kv, eff_total_k);
            std::snprintf(probe_name, sizeof(probe_name), "draft_l%d_V", il);
            ggml_set_name(V, probe_name);

            // ── 2d. RoPE (NEOX, theta=10M)
            //   Q: positions_q  [q_len]           values [ctx_len..ctx_len+q_len-1]
            //   K: positions_k  [eff_total_k]     — for SWA, starts from ctx_offset
            ggml_tensor * pk = in.positions_k;
            if (use_swa) {
                pk = ggml_view_1d(ctx, in.positions_k, eff_total_k,
                                  ctx_offset * ggml_element_size(in.positions_k));
            }
            Q = draft_rope(ctx, Q, in.positions_q, w);
            K = draft_rope(ctx, K, pk, w);

            // ── 2e. Permute into the layout flash_attn_ext wants
            //   q: [n_embd_k=head_dim, n_batch=q_len, n_head,   ne3]
            //   k: [n_embd_k=head_dim, n_kv=eff_total_k, n_head_kv, ne3]
            //   v: [n_embd_v=head_dim, n_kv=eff_total_k, n_head_kv, ne3]  (not transposed)
            Q = ggml_permute(ctx, Q, 0, 2, 1, 3);  // [head_dim, q_len,        n_head, 1]
            Q = ggml_cont   (ctx, Q);
            K = ggml_permute(ctx, K, 0, 2, 1, 3);  // [head_dim, eff_total_k,  n_kv,   1]
            K = ggml_cont   (ctx, K);
            V = ggml_permute(ctx, V, 0, 2, 1, 3);  // [head_dim, eff_total_k,  n_kv,   1]
            V = ggml_cont   (ctx, V);

            // ── 2f. Attention: causal for SWA layers, non-causal for full layers.
            const float scale = 1.0f / std::sqrt((float)head_dim);
            ggml_tensor * mask = layer_is_swa
                ? (in.causal_mask_swa ? in.causal_mask_swa : nullptr)
                : in.pad_mask_full;
            ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q, K, V, mask,
                                                     scale, /*max_bias=*/0.0f,
                                                     /*logit_softcap=*/0.0f);
            std::snprintf(probe_name, sizeof(probe_name), "draft_l%d_attn", il);
            ggml_set_name(attn, probe_name);
            // attn result: [n_embd_v=head_dim, n_head, n_batch=q_len, 1]
            if (!disable_attn_gate && L.attn_gate) {
                ggml_tensor * gate = ggml_mul_mat(ctx, L.attn_gate, hn);  // [n_head|q_dim, q_len]
                gate = ggml_softplus(ctx, gate);
                std::snprintf(probe_name, sizeof(probe_name), "draft_l%d_gate", il);
                ggml_set_name(gate, probe_name);
                if (L.attn_gate_per_head) {
                    gate = ggml_reshape_3d(ctx, gate, 1, n_head, q_len);
                } else {
                    gate = ggml_reshape_3d(ctx, gate, head_dim, n_head, q_len);
                }
                gate = ggml_cast(ctx, gate, attn->type);
                attn = ggml_mul(ctx, attn, gate);
            }
            attn = ggml_reshape_2d(ctx, attn, head_dim * n_head, q_len);
            // attn: [q_dim, q_len]

            // ── 2g. Output projection + residual
            //     wo: [q_dim, hidden]  (ne[0]=q_dim, ne[1]=hidden)
            ggml_tensor * attn_out = ggml_mul_mat(ctx, L.wo, attn);  // [hidden, q_len]
            if (dyn_conv) {
                attn_out = draft_dyn_conv_apply(ctx, w, L.attn_conv, attn_dc, 1, attn_out);
            }
            std::snprintf(probe_name, sizeof(probe_name), "draft_l%d_attn_out", il);
            ggml_set_name(attn_out, probe_name);
            h = ggml_add(ctx, h, attn_out);
            std::snprintf(probe_name, sizeof(probe_name), "draft_l%d_h_after_attn", il);
            ggml_set_name(h, probe_name);
        }

        if (!disable_ffn) {
            // ── 2h. FFN pre-norm
            ggml_tensor * hf = ggml_rms_norm(ctx, h, eps);
            hf = ggml_mul(ctx, hf, L.ffn_norm);
            DraftDynConv mlp_dc;
            if (dyn_conv) {
                mlp_dc = draft_dyn_conv_kernel(ctx, L.mlp_conv, hf);
                hf = draft_dyn_conv_apply(ctx, w, L.mlp_conv, mlp_dc, 0, hf);
            }

            // ── 2i. SwiGLU: down(silu(gate(x)) * up(x))
            //     w_gate, w_up: [hidden, intermediate]
            //     w_down:       [intermediate, hidden]
            ggml_tensor * g  = ggml_mul_mat(ctx, L.w_gate, hf);  // [inter, q_len]
            g = ggml_silu(ctx, g);
            ggml_tensor * u  = ggml_mul_mat(ctx, L.w_up,   hf);  // [inter, q_len]
            ggml_tensor * gu = ggml_mul(ctx, g, u);
            ggml_tensor * ffn_out = ggml_mul_mat(ctx, L.w_down, gu);  // [hidden, q_len]
            if (dyn_conv) {
                ffn_out = draft_dyn_conv_apply(ctx, w, L.mlp_conv, mlp_dc, 1, ffn_out);
            }

            h = ggml_add(ctx, h, ffn_out);
            std::snprintf(probe_name, sizeof(probe_name), "draft_l%d_h_after_ffn", il);
            ggml_set_name(h, probe_name);
        }
    }

    // ── 3. Final norm
    ggml_tensor * out = ggml_rms_norm(ctx, h, eps);
    out = ggml_mul(ctx, out, w.out_norm);
    ggml_set_name(out, "draft_hidden_out");

    DraftGraphOutputs og{};
    og.hidden_states = out;
    og.logits = nullptr;

    // ── 4. Optional: project through target's lm_head to emit vocab logits
    if (in.lm_head) {
        ggml_tensor * logits = ggml_mul_mat(ctx, in.lm_head, out);
        ggml_set_name(logits, "draft_logits");
        og.logits = logits;
    }
    return og;
}

// ── Cached drafter context-KV builders ──────────────────────────────
//
// The per-(head,position) k_norm and RoPE are separable, so computing the ctx
// K/V rows here (append) and the noise K/V rows in the step graph matches the
// legacy concat-then-normalize math exactly; the only numeric difference is
// the F16 cache storage (legacy keeps K/V in F32 for the one shot).

static ggml_tensor * draft_pack_columns(
        ggml_context * ctx,
        const std::vector<ggml_tensor *> & lanes) {
    ggml_tensor * packed = lanes.front();
    for (size_t lane = 1; lane < lanes.size(); ++lane) {
        packed = ggml_concat(ctx, packed, lanes[lane], 1);
    }
    return packed;
}

static ggml_tensor * draft_lane_columns(
        ggml_context * ctx, ggml_tensor * packed,
        int q_len, size_t lane) {
    return ggml_view_2d(
        ctx, packed, packed->ne[0], q_len, packed->nb[1],
        lane * static_cast<size_t>(q_len) * packed->nb[1]);
}

static std::vector<DraftDynConv> draft_dyn_conv_kernels(
        ggml_context * ctx, const DraftConvWeights & weights,
        const std::vector<ggml_tensor *> & lanes, int q_len) {
    if (lanes.size() == 1) {
        return {draft_dyn_conv_kernel(ctx, weights, lanes.front())};
    }
    const DraftDynConv packed = draft_dyn_conv_kernel(
        ctx, weights, draft_pack_columns(ctx, lanes));
    std::vector<DraftDynConv> kernels(lanes.size());
    for (size_t lane = 0; lane < lanes.size(); ++lane) {
        kernels[lane].dyn = draft_lane_columns(ctx, packed.dyn, q_len, lane);
    }
    return kernels;
}

bool build_draft_kv_appends(
        ggml_context *                         ctx,
        ggml_cgraph *                          gf,
        const DraftWeights &                   w,
        const std::vector<DraftKvAppendLane> & lanes) {
    if (!ctx || !gf || lanes.empty()) {
        return false;
    }

    const int64_t append_width = lanes.front().feat
        ? lanes.front().feat->ne[1]
        : 0;
    if (append_width <= 0 ||
        lanes.size() > static_cast<size_t>(
            std::numeric_limits<int>::max() / append_width)) {
        return false;
    }
    const int packed_width =
        static_cast<int>(append_width) * static_cast<int>(lanes.size());
    const int64_t feature_width = lanes.front().feat->ne[0];

    std::vector<ggml_tensor *> lane_features;
    lane_features.reserve(lanes.size());
    for (const DraftKvAppendLane & lane : lanes) {
        if (!lane.cache || !lane.feat || !lane.positions || !lane.rows ||
            lane.feat->ne[1] != append_width ||
            lane.feat->ne[0] != feature_width ||
            lane.positions->ne[0] != append_width ||
            lane.rows->ne[0] != append_width ||
            lane.cache->k.size() != static_cast<size_t>(w.n_layer) ||
            lane.cache->v.size() != static_cast<size_t>(w.n_layer)) {
            return false;
        }
        lane_features.push_back(lane.feat);
    }

    const int width = static_cast<int>(append_width);
    static const bool disable_aux_hidden_norms =
        std::getenv("DFLASH_DISABLE_DRAFT_AUX_NORMS") != nullptr;
    ggml_tensor * packed_features =
        draft_pack_columns(ctx, lane_features);
    ggml_tensor * target_feat = draft_fuse_features(
        ctx, w, packed_features, packed_width,
        disable_aux_hidden_norms);
    ggml_set_name(target_feat, "draft_kv_append_feat");

    const float eps = DFLASH27B_RMS_EPS;
    for (int il = 0; il < w.n_layer; ++il) {
        const DraftLayer & layer = w.layers[il];
        ggml_tensor * tf_kv = target_feat;
        if (w.context_kv_layer_norm) {
            tf_kv = ggml_rms_norm(ctx, tf_kv, eps);
            tf_kv = ggml_mul(ctx, tf_kv, layer.attn_norm);
        }
        ggml_tensor * packed_k = ggml_mul_mat(ctx, layer.wk, tf_kv);
        ggml_tensor * packed_v = ggml_mul_mat(ctx, layer.wv, tf_kv);

        for (size_t lane_index = 0;
             lane_index < lanes.size(); ++lane_index) {
            const DraftKvAppendLane & lane = lanes[lane_index];
            ggml_tensor * K = draft_lane_columns(
                ctx, packed_k, width, lane_index);
            K = ggml_reshape_3d(
                ctx, K, w.head_dim, w.n_head_kv, width);
            K = ggml_rms_norm(ctx, K, eps);
            K = ggml_mul(ctx, K, layer.k_norm);
            K = draft_rope(ctx, K, lane.positions, w);
            ggml_tensor * Krows = ggml_view_2d(
                ctx, K,
                static_cast<int64_t>(w.head_dim) * w.n_head_kv,
                width, K->nb[2], 0);
            ggml_tensor * Vrows = draft_lane_columns(
                ctx, packed_v, width, lane_index);
            ggml_build_forward_expand(
                gf, ggml_set_rows(
                    ctx, lane.cache->k[il], Krows, lane.rows));
            ggml_build_forward_expand(
                gf, ggml_set_rows(
                    ctx, lane.cache->v[il], Vrows, lane.rows));
        }
    }
    return true;
}

std::vector<DraftGraphOutputs> build_draft_kv_steps(
        ggml_context *                         ctx,
        ggml_cgraph *                          gf,
        const DraftWeights &                   w,
        const std::vector<DraftKvLaneInputs> & lanes) {
    const size_t n_lanes = lanes.size();
    if (!ctx || !gf || n_lanes == 0) {
        return {};
    }

    const int q_len = w.block_size;
    const int n_head = w.n_head;
    const int n_kv = w.n_head_kv;
    const int head_dim = w.head_dim;
    const int64_t q_dim = static_cast<int64_t>(head_dim) * n_head;
    const int64_t kv_dim = static_cast<int64_t>(head_dim) * n_kv;
    const float eps = DFLASH27B_RMS_EPS;
    static const bool disable_attn_gate =
        std::getenv("DFLASH_DISABLE_DRAFT_ATTN_GATE") != nullptr;
    static const bool disable_swa =
        std::getenv("DFLASH_DISABLE_DRAFT_SWA") != nullptr;

    for (const DraftKvLaneInputs & lane : lanes) {
        const DraftKvCacheRefs * cache = lane.cache;
        const DraftKvStepInputs & in = lane.inputs;
        if (!cache || !in.noise_embed || !in.positions_q || !in.noise_rows ||
            cache->kv_total <= 0 ||
            cache->k.size() != static_cast<size_t>(w.n_layer) ||
            cache->v.size() != static_cast<size_t>(w.n_layer)) {
            return {};
        }
    }

    std::vector<ggml_tensor *> h(n_lanes);
    for (size_t lane = 0; lane < n_lanes; ++lane) {
        h[lane] = lanes[lane].inputs.noise_embed;
    }

    for (int il = 0; il < w.n_layer; ++il) {
        const DraftLayer & layer = w.layers[il];
        const bool layer_is_swa = layer.is_swa && !disable_swa;
        const bool dyn_conv = w.conv_kernel_size > 0 &&
            layer.attn_conv.present() && layer.mlp_conv.present();

        std::vector<ggml_tensor *> hn(n_lanes);
        std::vector<DraftDynConv> attn_dc(n_lanes);
        for (size_t lane = 0; lane < n_lanes; ++lane) {
            hn[lane] = ggml_rms_norm(ctx, h[lane], eps);
            hn[lane] = ggml_mul(ctx, hn[lane], layer.attn_norm);
        }
        if (dyn_conv) {
            attn_dc = draft_dyn_conv_kernels(
                ctx, layer.attn_conv, hn, q_len);
            for (size_t lane = 0; lane < n_lanes; ++lane) {
                hn[lane] = draft_dyn_conv_apply(
                    ctx, w, layer.attn_conv, attn_dc[lane], 0, hn[lane]);
            }
        }

        ggml_tensor * hn_packed = draft_pack_columns(ctx, hn);
        ggml_tensor * q_packed = ggml_mul_mat(ctx, layer.wq, hn_packed);
        ggml_tensor * k_packed = ggml_mul_mat(ctx, layer.wk, hn_packed);
        ggml_tensor * v_packed = ggml_mul_mat(ctx, layer.wv, hn_packed);
        ggml_tensor * gate_packed = nullptr;
        if (!disable_attn_gate && layer.attn_gate) {
            gate_packed = ggml_mul_mat(ctx, layer.attn_gate, hn_packed);
            gate_packed = ggml_softplus(ctx, gate_packed);
        }

        std::vector<ggml_tensor *> attn_by_lane(n_lanes);
        for (size_t lane = 0; lane < n_lanes; ++lane) {
            const DraftKvCacheRefs & cache = *lanes[lane].cache;
            const DraftKvStepInputs & in = lanes[lane].inputs;

            ggml_tensor * q = draft_lane_columns(ctx, q_packed, q_len, lane);
            q = ggml_reshape_3d(ctx, q, head_dim, n_head, q_len);
            q = ggml_rms_norm(ctx, q, eps);
            q = ggml_mul(ctx, q, layer.q_norm);
            q = draft_rope(ctx, q, in.positions_q, w);

            ggml_tensor * kn = draft_lane_columns(ctx, k_packed, q_len, lane);
            kn = ggml_reshape_3d(ctx, kn, head_dim, n_kv, q_len);
            kn = ggml_rms_norm(ctx, kn, eps);
            kn = ggml_mul(ctx, kn, layer.k_norm);
            kn = draft_rope(ctx, kn, in.positions_q, w);
            ggml_tensor * kn_rows = ggml_view_2d(
                ctx, kn, kv_dim, q_len, kn->nb[2], 0);
            ggml_tensor * vn_rows = draft_lane_columns(
                ctx, v_packed, q_len, lane);
            ggml_build_forward_expand(
                gf, ggml_set_rows(ctx, cache.k[il], kn_rows, in.noise_rows));
            ggml_build_forward_expand(
                gf, ggml_set_rows(ctx, cache.v[il], vn_rows, in.noise_rows));

            ggml_tensor * qfa = ggml_permute(ctx, q, 0, 2, 1, 3);
            qfa = ggml_cont(ctx, qfa);
            ggml_tensor * kview = ggml_view_3d(
                ctx, cache.k[il], head_dim, n_kv, cache.kv_total,
                ggml_row_size(cache.k[il]->type, head_dim),
                cache.k[il]->nb[1], 0);
            ggml_tensor * vview = ggml_view_3d(
                ctx, cache.v[il], head_dim, n_kv, cache.kv_total,
                ggml_row_size(cache.v[il]->type, head_dim),
                cache.v[il]->nb[1], 0);
            ggml_tensor * kfa = ggml_permute(ctx, kview, 0, 2, 1, 3);
            ggml_tensor * vfa = ggml_permute(ctx, vview, 0, 2, 1, 3);
            ggml_tensor * mask = layer_is_swa ? in.mask_swa : in.mask_full;
            ggml_tensor * attn = ggml_flash_attn_ext(
                ctx, qfa, kfa, vfa, mask,
                1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f, 0.0f);

            if (gate_packed) {
                ggml_tensor * gate = draft_lane_columns(
                    ctx, gate_packed, q_len, lane);
                if (layer.attn_gate_per_head) {
                    gate = ggml_reshape_3d(ctx, gate, 1, n_head, q_len);
                } else {
                    gate = ggml_reshape_3d(ctx, gate, head_dim, n_head, q_len);
                }
                gate = ggml_cast(ctx, gate, attn->type);
                attn = ggml_mul(ctx, attn, gate);
            }
            attn_by_lane[lane] = ggml_reshape_2d(
                ctx, attn, q_dim, q_len);
        }

        ggml_tensor * attn_packed = draft_pack_columns(ctx, attn_by_lane);
        ggml_tensor * attn_out_packed =
            ggml_mul_mat(ctx, layer.wo, attn_packed);
        for (size_t lane = 0; lane < n_lanes; ++lane) {
            ggml_tensor * attn_out = draft_lane_columns(
                ctx, attn_out_packed, q_len, lane);
            if (dyn_conv) {
                attn_out = draft_dyn_conv_apply(
                    ctx, w, layer.attn_conv, attn_dc[lane], 1, attn_out);
            }
            h[lane] = ggml_add(ctx, h[lane], attn_out);
        }

        std::vector<ggml_tensor *> hf(n_lanes);
        std::vector<DraftDynConv> mlp_dc(n_lanes);
        for (size_t lane = 0; lane < n_lanes; ++lane) {
            hf[lane] = ggml_rms_norm(ctx, h[lane], eps);
            hf[lane] = ggml_mul(ctx, hf[lane], layer.ffn_norm);
        }
        if (dyn_conv) {
            mlp_dc = draft_dyn_conv_kernels(
                ctx, layer.mlp_conv, hf, q_len);
            for (size_t lane = 0; lane < n_lanes; ++lane) {
                hf[lane] = draft_dyn_conv_apply(
                    ctx, w, layer.mlp_conv, mlp_dc[lane], 0, hf[lane]);
            }
        }

        ggml_tensor * hf_packed = draft_pack_columns(ctx, hf);
        ggml_tensor * gate = ggml_mul_mat(ctx, layer.w_gate, hf_packed);
        gate = ggml_silu(ctx, gate);
        ggml_tensor * up = ggml_mul_mat(ctx, layer.w_up, hf_packed);
        ggml_tensor * ffn = ggml_mul(ctx, gate, up);
        ggml_tensor * ffn_out_packed = ggml_mul_mat(ctx, layer.w_down, ffn);
        for (size_t lane = 0; lane < n_lanes; ++lane) {
            ggml_tensor * ffn_out = draft_lane_columns(
                ctx, ffn_out_packed, q_len, lane);
            if (dyn_conv) {
                ffn_out = draft_dyn_conv_apply(
                    ctx, w, layer.mlp_conv, mlp_dc[lane], 1, ffn_out);
            }
            h[lane] = ggml_add(ctx, h[lane], ffn_out);
        }
    }

    std::vector<DraftGraphOutputs> outputs(n_lanes);
    for (size_t lane = 0; lane < n_lanes; ++lane) {
        ggml_tensor * out = ggml_rms_norm(ctx, h[lane], eps);
        out = ggml_mul(ctx, out, w.out_norm);
        outputs[lane].hidden_states = out;
        outputs[lane].logits = nullptr;
        if (lanes[lane].inputs.lm_head) {
            outputs[lane].logits = ggml_mul_mat(
                ctx, lanes[lane].inputs.lm_head, out);
        }
    }
    return outputs;
}

} // namespace dflash::common
