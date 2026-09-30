#pragma once
#include "common.cuh"

struct ggml_cuda_hc_combine_norm_args {
    const ggml_tensor * inject;     // [hc, T]  F32 contiguous (pre-scale/sigmoid)
    const ggml_tensor * residual;   // [n_embd, hc, T] F32 contiguous
    const ggml_tensor * block_out;  // [n_embd, 1, T]  F32 contiguous
    const ggml_tensor * gamma;
    ggml_tensor *       out_res;
    ggml_tensor *       out_xn;     // [n_embd, hc, T]
    float               s1, b1, s2, b2;
    float               eps;
    uint16_t *          out_xn_bf16 = nullptr;
    bool                store_xn_f32 = true;     // false: consumers all read the BF16 copy
    const uint16_t *    res_in_bf16  = nullptr;   // `residual` is BF16 in place (marked bf16-only)
    const uint16_t *    blk_in_bf16  = nullptr;
    uint16_t *          res_out_bf16 = nullptr;
    // MoE mode (block_out unused): block_out[t] = sum_e down[t,e]*w[t,e] (route order, weight-0 routes skipped)
    // + shared[t]*sigmoid(shared_logit[t]), the exact arithmetic of DS4_MOE_COMBINE after the gated shared-expert MUL.
    const ggml_tensor * moe_down     = nullptr;   // [n_embd, n_used, T] F32
    const ggml_tensor * moe_w        = nullptr;   // [n_used, T]        F32
    const ggml_tensor * moe_shared   = nullptr;   // [n_embd, T]        F32 (before the sigmoid gate)
    const ggml_tensor * moe_sh_logit = nullptr;   // [1, T]             F32
    bool                moe_down_f16 = false;     // moe_down holds F16 in place (F16-only mark)
    int8_t *            out_q8       = nullptr;   // Q8 activation tiles of xn for the W8A8 HC down (mmb-w8a8.cuh layout)
};

bool ggml_cuda_hc_combine_norm_supported(const ggml_cuda_hc_combine_norm_args & args, int warp_size);
void ggml_cuda_op_hc_combine_norm(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_combine_norm_args & args);

// Packed two-output entry: writes out_res[n_embd,hc,T] and out_xn[n_embd,hc,T].
void ggml_cuda_hc_combine_norm_ptrs(ggml_backend_cuda_context & ctx,
        const ggml_tensor * inject, const ggml_tensor * residual, const ggml_tensor * block_out,
        const ggml_tensor * gamma, float * out_res, float * out_xn,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        float s1, float b1, float s2, float b2, float eps);
