// qwen36_mtp_graph.cpp — CUDA cgraph builder for Qwen3.6 MTP head's
// per-step forward.  Mirrors the backbone's full-attention TRMBlock shape
// (qwen35_target_graph.cpp:build_full_attn_block) using ggml ops on the
// backbone's CUDA backend, so the head's matmuls use ggml's quant-aware
// MMQ kernels (matching backbone precision) instead of a CPU fp32 forward
// that drifts on Q2_K / Q3_K weights.

#include "qwen36_mtp_graph.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cmath>

namespace dflash27b {
namespace mtp {

static ggml_tensor * rms_norm_mul(ggml_context * ctx, ggml_tensor * x,
                                  ggml_tensor * weight, float eps) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, weight);
}

void qwen36_mtp_step_graph_free(Qwen36MtpStepGraph & sg) {
    if (sg.alloc) {
        ggml_gallocr_free(sg.alloc);
        sg.alloc = nullptr;
    }
    if (sg.ctx) {
        ggml_free(sg.ctx);
        sg.ctx = nullptr;
    }
    sg.gf                = nullptr;
    sg.inp_embed         = nullptr;
    sg.inp_h_prev        = nullptr;
    sg.inp_pos           = nullptr;
    sg.inp_kv_idx_write  = nullptr;
    sg.inp_kv_idxs_read  = nullptr;
    sg.inp_kv_mask       = nullptr;
    sg.out_x_normed      = nullptr;
    sg.out_h_pre_norm    = nullptr;
    sg.out_argmax_token  = nullptr;
    sg.out_logits        = nullptr;
    sg.fa_window         = 0;
    sg.fa_max            = 0;
    sg.topk_k            = 0;
    sg.fused_lm_head     = false;
}

void qwen36_mtp_warm_graph_free(Qwen36MtpWarmGraph & sg) {
    if (sg.alloc) {
        ggml_gallocr_free(sg.alloc);
        sg.alloc = nullptr;
    }
    if (sg.ctx) {
        ggml_free(sg.ctx);
        sg.ctx = nullptr;
    }
    sg.gf            = nullptr;
    sg.inp_embed_seq = nullptr;
    sg.inp_h_seq     = nullptr;
    sg.inp_pos       = nullptr;
}

bool build_qwen36_mtp_warm_graph(
        Qwen36MtpWarmGraph & sg,
        const Qwen36MtpHeadWeights & head,
        ggml_tensor * head_k_cache,
        ggml_tensor * head_v_cache,
        ggml_backend_t backend,
        int n_embd,
        int n_head_kv,
        int key_len,
        int val_len,
        int n_rot,
        int rope_sections[4],
        float rope_freq_base,
        float rms_eps,
        int slot_start,
        int n_tokens) {
    qwen36_mtp_warm_graph_free(sg);
    if (n_tokens <= 0) return false;

    const int head_dim = key_len;  // qwen3.6 has key_length == value_length

    ggml_init_params ip{};
    // n_tokens up to a few hundred prompt tokens; concat + matmul of size
    // (kv_dim, n_embd) needs scratch.  256 MB headroom is plenty.
    ip.mem_size   = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;
    sg.gf = ggml_new_graph_custom(sg.ctx, 4096, false);

    // Inputs.
    sg.inp_embed_seq = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(sg.inp_embed_seq, "mtp_warm_embed_seq");
    ggml_set_input(sg.inp_embed_seq);

    sg.inp_h_seq = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(sg.inp_h_seq, "mtp_warm_h_seq");
    ggml_set_input(sg.inp_h_seq);

    sg.inp_pos = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(sg.inp_pos, "mtp_warm_pos");
    ggml_set_input(sg.inp_pos);

    // Eq 21: eh_proj([e_norm; h_norm]) — concat order matches llama.cpp PR #22673.
    ggml_tensor * e_in = rms_norm_mul(sg.ctx, sg.inp_embed_seq, head.enorm, rms_eps);
    ggml_tensor * h_in = rms_norm_mul(sg.ctx, sg.inp_h_seq,     head.hnorm, rms_eps);
    ggml_tensor * cat  = ggml_concat(sg.ctx, e_in, h_in, /*dim=*/0);
    ggml_tensor * x    = ggml_mul_mat(sg.ctx, head.eh_proj, cat);  // [n_embd, n_tokens]

    // Pre-attn norm.
    ggml_tensor * cur = rms_norm_mul(sg.ctx, x, head.attn_norm, rms_eps);

    // K, V projections only (no Q, no attention, no FFN).
    ggml_tensor * Kcur = ggml_mul_mat(sg.ctx, head.attn_k, cur);  // [kv_dim, n_tokens]
    Kcur = ggml_reshape_3d(sg.ctx, Kcur, head_dim, n_head_kv, n_tokens);
    Kcur = rms_norm_mul(sg.ctx, Kcur, head.attn_k_norm, rms_eps);

    ggml_tensor * Vcur = ggml_mul_mat(sg.ctx, head.attn_v, cur);
    Vcur = ggml_reshape_3d(sg.ctx, Vcur, head_dim, n_head_kv, n_tokens);

    Kcur = ggml_rope_multi(sg.ctx, Kcur, sg.inp_pos, /*freq_factors=*/nullptr,
                            n_rot, rope_sections, GGML_ROPE_TYPE_MROPE,
                            /*n_ctx_orig=*/0, rope_freq_base, 1.0f,
                            0.0f, 1.0f, 0.0f, 0.0f);

    // Permute to [head_dim, n_tokens, n_head_kv] so cpy maps element-wise
    // into the cache view (slot range along dim 1).
    ggml_tensor * K_T = ggml_permute(sg.ctx, Kcur, 0, 2, 1, 3);
    ggml_tensor * V_T = ggml_permute(sg.ctx, Vcur, 0, 2, 1, 3);

    ggml_tensor * k_dst = ggml_view_3d(sg.ctx, head_k_cache,
        head_dim, n_tokens, n_head_kv,
        head_k_cache->nb[1], head_k_cache->nb[2],
        head_k_cache->nb[1] * slot_start);
    ggml_tensor * v_dst = ggml_view_3d(sg.ctx, head_v_cache,
        head_dim, n_tokens, n_head_kv,
        head_v_cache->nb[1], head_v_cache->nb[2],
        head_v_cache->nb[1] * slot_start);

    ggml_build_forward_expand(sg.gf, ggml_cpy(sg.ctx, K_T, k_dst));
    ggml_build_forward_expand(sg.gf, ggml_cpy(sg.ctx, V_T, v_dst));

    sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!sg.alloc) {
        std::fprintf(stderr, "[qwen36_mtp_warm_graph] gallocr_new failed\n");
        return false;
    }
    if (!ggml_gallocr_alloc_graph(sg.alloc, sg.gf)) {
        std::fprintf(stderr, "[qwen36_mtp_warm_graph] alloc_graph failed\n");
        return false;
    }
    return true;
}

bool build_qwen36_mtp_step_graph(
        Qwen36MtpStepGraph & sg,
        const Qwen36MtpHeadWeights & head,
        ggml_tensor * head_k_cache,
        ggml_tensor * head_v_cache,
        ggml_backend_t backend,
        int n_embd,
        int n_head,
        int n_head_kv,
        int key_len,
        int val_len,
        int ffn_len,
        int n_rot,
        int rope_sections[4],
        float rope_freq_base,
        float rms_eps,
        int n_ctx,
        int fa_window,
        ggml_tensor * lm_head_weight,
        int lm_head_topk) {
    qwen36_mtp_step_graph_free(sg);

    const int q_dim = n_head * key_len;
    const int head_dim = key_len;  // qwen3.6 has key_length == value_length
    const int fa_max = (fa_window > 0 && fa_window < n_ctx) ? fa_window : n_ctx;

    ggml_init_params ip{};
    ip.mem_size   = 128 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;
    sg.gf = ggml_new_graph_custom(sg.ctx, 2048, false);

    // ─── Inputs ────────────────────────────────────────────────────
    sg.inp_embed  = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_F32, n_embd);
    ggml_set_name(sg.inp_embed, "mtp_inp_embed");
    ggml_set_input(sg.inp_embed);

    sg.inp_h_prev = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_F32, n_embd);
    ggml_set_name(sg.inp_h_prev, "mtp_inp_h_prev");
    ggml_set_input(sg.inp_h_prev);

    // MROPE expects positions as [n_tokens * 4] i32 (4 axes per token).
    sg.inp_pos = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4);
    ggml_set_name(sg.inp_pos, "mtp_inp_pos");
    ggml_set_input(sg.inp_pos);

    // Runtime KV slot routing (bug #5: avoid baking draft_pos into views).
    sg.inp_kv_idx_write = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I64, 1);
    ggml_set_name(sg.inp_kv_idx_write, "mtp_inp_kv_idx_write");
    ggml_set_input(sg.inp_kv_idx_write);

    sg.inp_kv_idxs_read = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_I32, fa_max, n_head_kv);
    ggml_set_name(sg.inp_kv_idxs_read, "mtp_inp_kv_idxs_read");
    ggml_set_input(sg.inp_kv_idxs_read);

    sg.inp_kv_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, fa_max, 1);
    ggml_set_name(sg.inp_kv_mask, "mtp_inp_kv_mask");
    ggml_set_input(sg.inp_kv_mask);

    // ─── Eq 21: eh_proj([hnorm(h_prev); enorm(embed)]) ────────────
    ggml_tensor * e_in = rms_norm_mul(sg.ctx, sg.inp_embed,  head.enorm, rms_eps);
    ggml_tensor * h_in = rms_norm_mul(sg.ctx, sg.inp_h_prev, head.hnorm, rms_eps);
    // Concat order [e_in; h_in] (embed first, hidden second) matches the
    // reference impl in llama.cpp PR #22673 (graph_mtp in src/models/qwen35.cpp:
    // `ggml_concat(ctx0, e_norm, h_norm, 0)`).  The earlier redesign doc had
    // the order backwards; with the wrong order, eh_proj receives swapped
    // halves of its trained input and the head produces useless logits.
    ggml_tensor * cat = ggml_concat(sg.ctx, e_in, h_in, /*dim=*/0);
    ggml_tensor * x   = ggml_mul_mat(sg.ctx, head.eh_proj, cat);
    // x: [n_embd]

    // ─── TRMBlock: pre-attn norm ─────────────────────────────────
    ggml_tensor * cur = rms_norm_mul(sg.ctx, x, head.attn_norm, rms_eps);
    // cur: [n_embd]

    // ─── Q/gate projection (packed [Q; gate] per head) ───────────
    // attn_q is [n_embd, 2*q_dim]; output is [2*q_dim].
    ggml_tensor * qg = ggml_mul_mat(sg.ctx, head.attn_q, cur);
    // Reshape to [head_dim*2, n_head] to split Q (offset 0) and gate (offset head_dim).
    qg = ggml_reshape_2d(sg.ctx, qg, head_dim * 2, n_head);

    // Q half: shape [head_dim, n_head], stride head_dim*2 between heads.
    ggml_tensor * Q = ggml_view_2d(sg.ctx, qg,
        head_dim, n_head,
        ggml_element_size(qg) * head_dim * 2,
        /*offset=*/0);
    Q = rms_norm_mul(sg.ctx, Q, head.attn_q_norm, rms_eps);

    // gate half: same shape, offset head_dim.
    ggml_tensor * gate = ggml_view_2d(sg.ctx, qg,
        head_dim, n_head,
        ggml_element_size(qg) * head_dim * 2,
        ggml_element_size(qg) * head_dim);
    gate = ggml_cont_2d(sg.ctx, gate, q_dim, 1);  // [q_dim, 1]

    // ─── K / V projections ─────────────────────────────────────
    ggml_tensor * Kcur = ggml_mul_mat(sg.ctx, head.attn_k, cur);
    Kcur = ggml_reshape_2d(sg.ctx, Kcur, head_dim, n_head_kv);
    Kcur = rms_norm_mul(sg.ctx, Kcur, head.attn_k_norm, rms_eps);

    ggml_tensor * Vcur = ggml_mul_mat(sg.ctx, head.attn_v, cur);
    Vcur = ggml_reshape_2d(sg.ctx, Vcur, head_dim, n_head_kv);

    // ─── MROPE on Q and K (4-axis positions, sections [11,11,10,0]) ──
    // rope_multi expects [n_dims, n_head, n_tokens, 1].
    ggml_tensor * Q3 = ggml_reshape_3d(sg.ctx, Q, head_dim, n_head, 1);
    Q3 = ggml_rope_multi(sg.ctx, Q3, sg.inp_pos, /*freq_factors=*/nullptr,
                          n_rot, rope_sections, GGML_ROPE_TYPE_MROPE,
                          /*n_ctx_orig=*/0, rope_freq_base, 1.0f,
                          0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * K3 = ggml_reshape_3d(sg.ctx, Kcur, head_dim, n_head_kv, 1);
    K3 = ggml_rope_multi(sg.ctx, K3, sg.inp_pos, nullptr,
                          n_rot, rope_sections, GGML_ROPE_TYPE_MROPE,
                          0, rope_freq_base, 1.0f,
                          0.0f, 1.0f, 0.0f, 0.0f);

    // ─── Write Kcur/Vcur at runtime slot (inp_kv_idx_write) ──────
    // Bug #5: per-slot views can't be built at graph-build time (offset
    // must be static).  Use ggml_set_rows on a 3D cache view; b is the
    // F32 K/V with shape [head_dim, 1, n_head_kv]; c is i64[1] broadcast
    // across the head_kv axis (b->ne[2] % c->ne[1] == 0 satisfies the
    // broadcast rule, so all heads write the same slot).
    // K3 is post-RoPE [head_dim, n_head_kv, 1]; reshape to [head_dim,1,n_head_kv]
    // so set_rows sees ne[1]=1 (==c.ne[0]) and broadcasts the i64[1] index over
    // the n_head_kv axis.
    ggml_tensor * K_b = ggml_reshape_3d(sg.ctx, K3,   head_dim, 1, n_head_kv);
    ggml_tensor * V_b = ggml_reshape_3d(sg.ctx, Vcur, head_dim, 1, n_head_kv);
    ggml_tensor * k_after = ggml_set_rows(sg.ctx, head_k_cache, K_b, sg.inp_kv_idx_write);
    ggml_tensor * v_after = ggml_set_rows(sg.ctx, head_v_cache, V_b, sg.inp_kv_idx_write);
    ggml_build_forward_expand(sg.gf, k_after);
    ggml_build_forward_expand(sg.gf, v_after);

    // ─── Flash attention over runtime-selected slots ─────────────
    // Read fa_max rows per head via ggml_get_rows.  Indices live in
    // inp_kv_idxs_read [fa_max, n_head_kv]; rows past live kv_len are
    // gathered from slot 0 then masked to -INF via inp_kv_mask.  Read
    // from k_after / v_after so the DAG sees the set_rows write as a
    // dependency (set_rows returns view(a) so direct dep chaining works).
    ggml_tensor * Qfa = ggml_permute(sg.ctx, Q3, 0, 2, 1, 3);
    Qfa = ggml_cont(sg.ctx, Qfa);

    ggml_tensor * Kfa = ggml_get_rows(sg.ctx, k_after, sg.inp_kv_idxs_read);
    ggml_tensor * Vfa = ggml_get_rows(sg.ctx, v_after, sg.inp_kv_idxs_read);

    const float kq_scale = 1.0f / std::sqrt((float)head_dim);
    ggml_tensor * attn = ggml_flash_attn_ext(sg.ctx, Qfa, Kfa, Vfa,
                                              sg.inp_kv_mask, kq_scale, 0.0f, 0.0f);
    // attn: [head_dim, n_head, n_tokens=1] (permuted output of FA)
    attn = ggml_reshape_2d(sg.ctx, attn, q_dim, 1);

    // ─── Sigmoid gate ────────────────────────────────────────────
    ggml_tensor * gate_sig = ggml_sigmoid(sg.ctx, gate);
    attn = ggml_mul(sg.ctx, attn, gate_sig);

    // ─── Output projection + residual ────────────────────────────
    ggml_tensor * attn_out = ggml_mul_mat(sg.ctx, head.attn_output, attn);
    // attn_out: [n_embd]; flatten x and attn_out to same rank.
    attn_out = ggml_reshape_1d(sg.ctx, attn_out, n_embd);
    x = ggml_add(sg.ctx, x, attn_out);

    // ─── Post-attn norm + SwiGLU FFN ─────────────────────────────
    cur = rms_norm_mul(sg.ctx, x, head.post_attention_norm, rms_eps);
    ggml_tensor * ffn_g = ggml_mul_mat(sg.ctx, head.ffn_gate, cur);
    ffn_g = ggml_silu(sg.ctx, ffn_g);
    ggml_tensor * ffn_u = ggml_mul_mat(sg.ctx, head.ffn_up, cur);
    ggml_tensor * ffn_gu = ggml_mul(sg.ctx, ffn_g, ffn_u);
    ggml_tensor * ffn_out = ggml_mul_mat(sg.ctx, head.ffn_down, ffn_gu);
    ffn_out = ggml_reshape_1d(sg.ctx, ffn_out, n_embd);
    x = ggml_add(sg.ctx, x, ffn_out);

    // ─── Pre-shared_head_norm hidden (chain-state output) ────────
    // Per llama.cpp PR #22673 (`t_h_pre_norm` in src/models/qwen35.cpp),
    // the hidden fed back as h_prev for the NEXT autoregressive step
    // must be POST-residual-add but PRE-`shared_head_norm`.  Feeding
    // back `out_x_normed` (post-norm) double-normalises on the next
    // iter's `hnorm` and compounds rejection per depth.  See the
    // CPU-path reference at qwen36_mtp.cpp:1166 (`last_hidden = x`).
    ggml_set_name(x, "mtp_out_h_pre_norm");
    ggml_set_output(x);
    sg.out_h_pre_norm = x;
    ggml_build_forward_expand(sg.gf, sg.out_h_pre_norm);

    // ─── Shared head norm ────────────────────────────────────────
    ggml_tensor * out = head.shared_head_norm
        ? rms_norm_mul(sg.ctx, x, head.shared_head_norm, rms_eps)
        : ggml_rms_norm(sg.ctx, x, rms_eps);
    ggml_set_name(out, "mtp_out_x_normed");
    ggml_set_output(out);
    sg.out_x_normed = out;

    ggml_build_forward_expand(sg.gf, sg.out_x_normed);

    // ─── Fused LM-head projection (optional) ─────────────────────
    // When the caller passes lm_head_weight non-null we append the LM
    // head matmul + argmax directly to this graph so step_batch_gpu_ can
    // avoid the hidden -> CPU -> separate projection-graph round trip
    // that dominates per-step latency (see qwen35_dflash_target.cpp:
    // project_hidden_to_tokens for the unfused path).
    if (lm_head_weight) {
        // out is shape [n_embd]; reshape to [n_embd, 1] so mul_mat against
        // the [n_embd, n_vocab] weight matches the LM-head projection
        // step graph (graph_builders.cpp:251).
        ggml_tensor * x_for_lm = ggml_reshape_2d(sg.ctx, out, n_embd, 1);
        ggml_tensor * logits   = ggml_mul_mat(sg.ctx, lm_head_weight, x_for_lm);
        ggml_set_name(logits, "mtp_fused_logits");
        if (lm_head_topk > 0) {
            // Surface raw logits so the host can run log-softmax for
            // top-K without re-running the matmul.  For K=1 we skip this
            // and download just the argmax to keep transfer minimal.
            ggml_set_output(logits);
            sg.out_logits = logits;
            ggml_build_forward_expand(sg.gf, logits);
        }
        ggml_tensor * argmax = ggml_argmax(sg.ctx, logits);
        ggml_set_name(argmax, "mtp_fused_argmax");
        ggml_set_output(argmax);
        sg.out_argmax_token = argmax;
        ggml_build_forward_expand(sg.gf, argmax);
    }

    // ─── Allocate ────────────────────────────────────────────────
    sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!sg.alloc) {
        std::fprintf(stderr, "[qwen36_mtp_graph] ggml_gallocr_new failed\n");
        return false;
    }
    if (!ggml_gallocr_alloc_graph(sg.alloc, sg.gf)) {
        std::fprintf(stderr, "[qwen36_mtp_graph] ggml_gallocr_alloc_graph failed\n");
        return false;
    }

    // Record build keys for cache invalidation in Qwen36MtpModule.
    sg.fa_window     = fa_window;
    sg.fa_max        = fa_max;
    sg.topk_k        = lm_head_topk;
    sg.fused_lm_head = (lm_head_weight != nullptr);
    return true;
}

}  // namespace mtp
}  // namespace dflash27b
