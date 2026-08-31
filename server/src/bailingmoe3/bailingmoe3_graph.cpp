#include "bailingmoe3_graph.h"

#include "internal.h"
#include "qwen35_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace dflash::common {
namespace {

ggml_tensor * build_causal_conv1d(
    ggml_context * ctx,
    ggml_cgraph * gf,
    ggml_tensor * all_conv_state,
    int qkv_index,
    ggml_tensor * cur,
    ggml_tensor * projection,
    ggml_tensor * conv_weight,
    int d_conv,
    int d_inner,
    int head_dim,
    int n_head,
    int n_tokens) {
    const size_t state_element = ggml_element_size(all_conv_state);
    const size_t channel_stride = all_conv_state->nb[1];
    ggml_tensor * conv_state = ggml_view_3d(
        ctx, all_conv_state, d_conv - 1, d_inner, 1,
        channel_stride, all_conv_state->nb[2],
        static_cast<size_t>(qkv_index) * d_inner * channel_stride);

    ggml_tensor * projected = ggml_mul_mat(ctx, projection, cur);
    projected = ggml_reshape_3d(ctx, projected, d_inner, n_tokens, 1);
    ggml_tensor * conv_input =
        ggml_concat(ctx, conv_state, ggml_transpose(ctx, projected), 0);

    ggml_tensor * last = ggml_view_3d(
        ctx, conv_input, d_conv - 1, d_inner, 1,
        conv_input->nb[1], conv_input->nb[2],
        static_cast<size_t>(n_tokens) * state_element);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, last, conv_state));

    ggml_tensor * conv_2d = ggml_reshape_2d(ctx, conv_weight, d_conv, d_inner);
    ggml_tensor * result = ggml_ssm_conv(ctx, conv_input, conv_2d);
    result = ggml_silu(ctx, ggml_reshape_2d(ctx, result, d_inner, n_tokens));
    return ggml_reshape_4d(ctx, result, head_dim, n_head, n_tokens, 1);
}

}  // namespace

// Ling 3 stores MLA K/V in compressed latent space. K contains the normalized
// 512-wide latent plus the 64 RoPE dimensions; V is the latent alone. Query
// absorption through attn_k_b turns the 128 non-RoPE Q dimensions into the
// same latent space, and attn_v_b expands the attention result back to 128
// value dimensions per head.
ggml_tensor * build_bailingmoe3_mla_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const TargetWeights & w,
    const TargetLayer & L,
    ggml_tensor * cur,
    ggml_tensor * positions,
    ggml_tensor * cache_k,
    ggml_tensor * cache_v,
    ggml_tensor * attn_mask,
    int kv_start,
    int n_tokens) {
    const int n_head = w.n_head;
    const int qk_dim = w.mla_qk_head_dim;
    const int rope_dim = w.rope_dimension_count;
    const int nope_dim = qk_dim - rope_dim;
    const int kv_rank = w.kv_lora_rank;
    const int v_dim = w.mla_v_head_dim;
    GGML_ASSERT(attn_mask != nullptr);

    ggml_tensor * q_all = nullptr;
    if (L.attn_q_a) {
        q_all = ggml_mul_mat(ctx, L.attn_q_a, cur);
        q_all = rms_norm_mul(ctx, q_all, L.attn_q_a_norm, w.rms_eps);
        q_all = ggml_mul_mat(ctx, L.attn_q_b, q_all);
    } else {
        q_all = ggml_mul_mat(ctx, L.wq, cur);
    }
    q_all = ggml_reshape_3d(ctx, q_all, qk_dim, n_head, n_tokens);
    ggml_tensor * q_nope = ggml_view_3d(
        ctx, q_all, nope_dim, n_head, n_tokens,
        q_all->nb[1], q_all->nb[2], 0);
    ggml_tensor * q_pe = ggml_view_3d(
        ctx, q_all, rope_dim, n_head, n_tokens,
        q_all->nb[1], q_all->nb[2],
        static_cast<size_t>(nope_dim) * ggml_element_size(q_all));

    ggml_tensor * kv_all = ggml_mul_mat(ctx, L.attn_kv_a_mqa, cur);
    ggml_tensor * kv = ggml_view_2d(
        ctx, kv_all, kv_rank, n_tokens, kv_all->nb[1], 0);
    ggml_tensor * k_pe = ggml_view_3d(
        ctx, kv_all, rope_dim, 1, n_tokens,
        kv_all->nb[1], kv_all->nb[1],
        static_cast<size_t>(kv_rank) * ggml_element_size(kv_all));

    // The shared hybrid graph input reserves four position lanes for M-RoPE.
    // Ling uses ordinary RoPE, so expose just the first lane to ggml_rope.
    ggml_tensor * rope_positions = positions;
    if (positions->ne[0] != n_tokens) {
        GGML_ASSERT(positions->ne[0] >= n_tokens);
        rope_positions = ggml_view_1d(ctx, positions, n_tokens, 0);
    }

    // Bailing V3 uses interleaved rotary pairs, i.e. GGML's NORMAL layout.
    q_pe = ggml_cont(ctx, q_pe);
    k_pe = ggml_cont(ctx, k_pe);
    q_pe = ggml_rope_ext(ctx, q_pe, rope_positions, nullptr,
                          rope_dim, GGML_ROPE_TYPE_NORMAL, 0,
                          w.rope_theta, 1.0f,
                          0.0f, 1.0f, 0.0f, 0.0f);
    k_pe = ggml_rope_ext(ctx, k_pe, rope_positions, nullptr,
                          rope_dim, GGML_ROPE_TYPE_NORMAL, 0,
                          w.rope_theta, 1.0f,
                          0.0f, 1.0f, 0.0f, 0.0f);
    kv = rms_norm_mul(ctx, kv, L.attn_kv_a_norm, w.rms_eps);

    // Absorb the K projection into Q: [nope,T,H] -> [latent,T,H].
    q_nope = ggml_permute(ctx, q_nope, 0, 2, 1, 3);
    q_nope = ggml_mul_mat(ctx, L.attn_k_b, q_nope);
    q_nope = ggml_permute(ctx, q_nope, 0, 2, 1, 3);
    ggml_tensor * q = ggml_concat(ctx, q_nope, q_pe, 0);

    ggml_tensor * kv_3d = ggml_reshape_3d(ctx, kv, kv_rank, 1, n_tokens);
    ggml_tensor * k_cur = ggml_concat(ctx, kv_3d, k_pe, 0);
    ggml_tensor * v_cur = kv_3d;

    // Persistent latent cache: [D,T,1].
    ggml_tensor * k_write = ggml_permute(ctx, k_cur, 0, 2, 1, 3);
    ggml_tensor * v_write = ggml_permute(ctx, v_cur, 0, 2, 1, 3);
    ggml_tensor * k_slot = ggml_view_3d(
        ctx, cache_k, kv_rank + rope_dim, n_tokens, 1,
        cache_k->nb[1], cache_k->nb[2],
        static_cast<size_t>(kv_start) * cache_k->nb[1]);
    ggml_tensor * v_slot = ggml_view_3d(
        ctx, cache_v, kv_rank, n_tokens, 1,
        cache_v->nb[1], cache_v->nb[2],
        static_cast<size_t>(kv_start) * cache_v->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, k_write, k_slot));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, v_write, v_slot));

    const int kv_len = kv_start + n_tokens;
    // The 576x512 CUDA flash-attention specialization requires the compressed
    // K/V span to be padded to its 256-token launch stride. The caller always
    // supplies a causal mask for Ling, so zero-initialized future cache rows
    // remain invisible.
    const int kv_len_padded = std::min(
        ((kv_len + 255) / 256) * 256, static_cast<int>(cache_k->ne[1]));
    ggml_tensor * k_read = ggml_view_3d(
        ctx, cache_k, kv_rank + rope_dim, kv_len_padded, 1,
        cache_k->nb[1], cache_k->nb[2], 0);
    ggml_tensor * v_read = ggml_view_3d(
        ctx, cache_v, kv_rank, kv_len_padded, 1,
        cache_v->nb[1], cache_v->nb[2], 0);
    ggml_tensor * q_fa = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * attn = ggml_flash_attn_ext(
        ctx, q_fa, k_read, v_read, attn_mask,
        1.0f / std::sqrt(static_cast<float>(qk_dim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);

    // Flash attention returns [latent,H,T]. Apply the per-head V expansion
    // as a batched matmul in [latent,T,H] layout.
    attn = ggml_permute(ctx, attn, 0, 2, 1, 3);
    attn = ggml_mul_mat(ctx, L.attn_v_b, attn);
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));

    ggml_tensor * gate = ggml_mul_mat(ctx, L.wqkv_gate, cur);
    gate = ggml_sigmoid(ctx,
        ggml_reshape_3d(ctx, gate, 1, n_head, n_tokens));
    attn = ggml_mul(ctx, attn, gate);
    attn = ggml_cont_2d(ctx, attn, v_dim * n_head, n_tokens);
    return ggml_mul_mat(ctx, L.wo, attn);
}

// Ling 3 KDA block. The shared ggml CUDA primitive detects KDA from the
// vector gate's first dimension (128 rather than scalar 1).
ggml_tensor * build_bailingmoe3_kda_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const TargetWeights & w,
    const TargetLayer & L,
    ggml_tensor * cur,
    ggml_tensor * conv_state,
    ggml_tensor * ssm_state,
    int n_tokens) {
    const int head_dim = w.kda_head_dim;
    const int n_head = w.n_head;
    const int d_inner = head_dim * n_head;

    ggml_tensor * q = build_causal_conv1d(
        ctx, gf, conv_state, 0, cur, L.wq, L.ssm_conv1d_q,
        w.ssm_d_conv, d_inner, head_dim, n_head, n_tokens);
    ggml_tensor * k = build_causal_conv1d(
        ctx, gf, conv_state, 1, cur, L.wk, L.ssm_conv1d_k,
        w.ssm_d_conv, d_inner, head_dim, n_head, n_tokens);
    ggml_tensor * v = build_causal_conv1d(
        ctx, gf, conv_state, 2, cur, L.wv, L.ssm_conv1d_v,
        w.ssm_d_conv, d_inner, head_dim, n_head, n_tokens);

    ggml_tensor * gate = ggml_mul_mat(ctx, L.ssm_f_a, cur);
    gate = ggml_add(ctx, gate, L.ssm_dt_bias);
    gate = ggml_reshape_4d(ctx, gate, head_dim, n_head, n_tokens, 1);
    ggml_tensor * a = ggml_reshape_4d(ctx, L.ssm_a, 1, n_head, 1, 1);
    gate = ggml_scale(ctx, ggml_sigmoid(ctx, ggml_mul(ctx, gate, a)),
                      w.kda_gate_lower_bound);

    ggml_tensor * beta = ggml_mul_mat(ctx, L.ssm_beta, cur);
    beta = ggml_sigmoid(ctx,
        ggml_reshape_4d(ctx, beta, 1, n_head, n_tokens, 1));
    q = ggml_l2_norm(ctx, q, w.rms_eps);
    k = ggml_l2_norm(ctx, k, w.rms_eps);

    ggml_tensor * state = ggml_reshape_4d(
        ctx, ssm_state, head_dim, head_dim, n_head, 1);
    ggml_tensor * packed =
        ggml_gated_delta_net(ctx, q, k, v, gate, beta, state);
    ggml_gated_delta_net_set_skip_intermediate(packed, true);

    const size_t element = ggml_element_size(packed);
    ggml_tensor * output = ggml_view_4d(
        ctx, packed, head_dim, n_head, n_tokens, 1,
        static_cast<size_t>(head_dim) * element,
        static_cast<size_t>(head_dim) * n_head * element,
        static_cast<size_t>(head_dim) * n_head * n_tokens * element,
        0);
    ggml_tensor * new_state = ggml_view_4d(
        ctx, packed, head_dim, head_dim, n_head, 1,
        static_cast<size_t>(head_dim) * element,
        static_cast<size_t>(head_dim) * head_dim * element,
        static_cast<size_t>(head_dim) * head_dim * n_head * element,
        static_cast<size_t>(head_dim) * n_head * n_tokens * element);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, state));

    ggml_tensor * output_gate = ggml_mul_mat(ctx, L.ssm_g_a, cur);
    output_gate = ggml_reshape_3d(
        ctx, output_gate, head_dim, n_head, n_tokens);
    output = ggml_reshape_3d(ctx, output, head_dim, n_head, n_tokens);
    output = rms_norm_mul(ctx, output, L.ssm_norm, w.rms_eps);
    output = ggml_mul(ctx, output, ggml_sigmoid(ctx, output_gate));
    output = ggml_cont_2d(ctx, output, d_inner, n_tokens);
    return ggml_mul_mat(ctx, L.wo, output);
}

}  // namespace dflash::common
