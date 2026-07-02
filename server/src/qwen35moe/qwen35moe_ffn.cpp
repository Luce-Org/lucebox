#include "qwen35moe_ffn.h"

#include "qwen35_ops.h"

#include <cstdlib>
#include <cmath>

namespace dflash::common {

Qwen35MoeRouterOutputs build_qwen35moe_router(
    ggml_context *        ctx,
    ggml_tensor *         cur,
    const TargetWeights & w,
    const TargetLayer &   L,
    bool                  allow_fused_router) {
    const int n_tokens = (int)cur->ne[1];
    const int n_expert = w.n_expert;
    const int n_used   = w.n_expert_used;

    ggml_tensor * logits = apply_scale2(ctx, ggml_mul_mat(ctx, L.ffn_gate_inp, cur), L.ffn_gate_inp_s);
    ggml_tensor * probs = nullptr;
    switch (w.expert_gating_func) {
        case 2:
            probs = ggml_sigmoid(ctx, logits);
            break;
        case 1:
        default:
            probs = ggml_soft_max(ctx, logits);
            break;
    }

    // ggml_argsort_top_k emits GGML_OP_ARGSORT (+view), which ggml-cuda's
    // topk-moe fusion (ggml_cuda_topk_moe_fusion) recognizes and fuses the whole
    // softmax->topk->get_rows->norm router into ~1 kernel. ggml_top_k emits
    // GGML_OP_TOP_K, which the fusion does NOT match -> 6-7 separate kernels/layer
    // x30 MoE layers (the launch-bound decode gap vs llama, which uses argsort_top_k).
    // Same top-k selection -> bit-identical. DFLASH_NO_MOE_ROUTER_FUSE=1 = old path.
    static const bool router_fuse = (std::getenv("DFLASH_NO_MOE_ROUTER_FUSE") == nullptr);
    ggml_tensor * selected = (router_fuse && allow_fused_router)
        ? ggml_argsort_top_k(ctx, probs, n_used)
        : ggml_top_k(ctx, probs, n_used);

    ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);
    ggml_tensor * weights  = ggml_get_rows(ctx, probs_3d, selected);
    weights = ggml_reshape_2d(ctx, weights, n_used, n_tokens);

    // Always normalize selected expert weights by their sum (matches
    // llama.cpp's norm_w=true for qwen35moe). Without this, top-k softmax
    // weights sum to much less than 1.0, causing systematically underscaled
    // FFN output.
    {
        ggml_tensor * w_sum = ggml_sum_rows(ctx, weights);
        w_sum = ggml_clamp(ctx, w_sum, 6.103515625e-5f, INFINITY);
        weights = ggml_div(ctx, weights, w_sum);
    }
    if (w.expert_weights_scale != 0.0f && w.expert_weights_scale != 1.0f) {
        weights = ggml_scale(ctx, weights, w.expert_weights_scale);
    }

    Qwen35MoeRouterOutputs out;
    out.selected = selected;
    out.weights = weights;
    return out;
}

ggml_tensor * build_qwen35moe_ffn(
    ggml_context *        ctx,
    ggml_tensor *         cur,
    const TargetWeights & w,
    const TargetLayer &   L,
    ggml_tensor **        selected_out) {
    const int n_tokens = (int)cur->ne[1];
    const int n_used   = w.n_expert_used;
    const int n_embd   = w.n_embd;
    const int n_ff_exp = w.n_ff_exp;

    Qwen35MoeRouterOutputs router = build_qwen35moe_router(ctx, cur, w, L);
    ggml_tensor * selected = router.selected;
    ggml_tensor * weights = router.weights;
    if (selected_out) {
        *selected_out = selected;
    }

    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
    ggml_tensor * gu = nullptr;
    // Combined gate_up: split + swiglu in ONE op. ggml_swiglu reads gate from
    // [0,nc) and up from [nc,2nc) of the same buffer, so the two ggml_cont copies
    // that materialised the strided halves are eliminated — 2 extra copy kernels
    // per layer x 30 MoE layers that llama's qwen3moe graph never emits.
    // DFLASH_NO_MOE_SWIGLU_FUSE=1 restores the view+cont+split path (bit-id gate).
    static const bool moe_swiglu_fuse = (std::getenv("DFLASH_NO_MOE_SWIGLU_FUSE") == nullptr);
    if (L.ffn_gate_up_exps) {
        ggml_tensor * gate_up_e = apply_scale2(
            ctx, ggml_mul_mat_id(ctx, L.ffn_gate_up_exps, cur_3d, selected), L.ffn_gate_up_exps_s);
        if (moe_swiglu_fuse) {
            gu = ggml_swiglu(ctx, gate_up_e);   // silu(gate) * up, no views/conts
        } else {
            ggml_tensor * gate_e = ggml_view_3d(ctx, gate_up_e,
                n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
                gate_up_e->nb[1], gate_up_e->nb[2], 0);
            ggml_tensor * up_e = ggml_view_3d(ctx, gate_up_e,
                n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
                gate_up_e->nb[1], gate_up_e->nb[2],
                (size_t)n_ff_exp * ggml_element_size(gate_up_e));
            gate_e = ggml_cont(ctx, gate_e);
            up_e   = ggml_cont(ctx, up_e);
            gu = ggml_swiglu_split(ctx, gate_e, up_e);
        }
    } else {
        ggml_tensor * gate_e = apply_scale2(
            ctx, ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur_3d, selected), L.ffn_gate_exps_s);
        ggml_tensor * up_e   = apply_scale2(
            ctx, ggml_mul_mat_id(ctx, L.ffn_up_exps,   cur_3d, selected), L.ffn_up_exps_s);
        gu = ggml_swiglu_split(ctx, gate_e, up_e);
    }

    ggml_tensor * experts = apply_scale2(
        ctx, ggml_mul_mat_id(ctx, L.ffn_down_exps, gu, selected), L.ffn_down_exps_s);
    ggml_tensor * w_view = ggml_reshape_3d(ctx, weights, 1, n_used, n_tokens);
    experts = ggml_mul(ctx, experts, w_view);

    // Sum over experts (single op: repeat_back sums dim 1)
    ggml_tensor * sum_shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * moe_sum   = ggml_repeat_back(ctx, experts, sum_shape);
    ggml_tensor * routed    = ggml_reshape_2d(ctx, moe_sum, n_embd, n_tokens);

    if (L.ffn_up_shexp && L.ffn_gate_shexp && L.ffn_down_shexp) {
        ggml_tensor * sh_gate = apply_scale2(
            ctx, ggml_mul_mat(ctx, L.ffn_gate_shexp, cur), L.ffn_gate_shexp_s);
        ggml_tensor * sh_up = apply_scale2(
            ctx, ggml_mul_mat(ctx, L.ffn_up_shexp, cur), L.ffn_up_shexp_s);
        ggml_tensor * sh_gu = ggml_swiglu_split(ctx, sh_gate, sh_up);
        ggml_tensor * shared = apply_scale2(
            ctx, ggml_mul_mat(ctx, L.ffn_down_shexp, sh_gu), L.ffn_down_shexp_s);

        if (L.ffn_gate_inp_shexp) {
            ggml_tensor * shared_gate = apply_scale2(
                ctx, ggml_mul_mat(ctx, L.ffn_gate_inp_shexp, cur), L.ffn_gate_inp_shexp_s);
            shared_gate = ggml_sigmoid(ctx, shared_gate);
            shared = ggml_mul(ctx, shared, shared_gate);
        }

        return routed ? ggml_add(ctx, routed, shared) : shared;
    }

    return routed;
}

}  // namespace dflash::common
