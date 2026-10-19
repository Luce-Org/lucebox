#pragma once
#include "common.cuh"

// same expression as op_sigmoid in unary.cu
static __device__ __forceinline__ float gn_sigmoid(const float x) {
    return 1.0f / (1.0f + expf(-x));
}

// same saturating conversion as mq_f2h_sat in mmb-q8f16.cuh
static __device__ __forceinline__ half gn_f2h_sat(const float v) {
    return __float2half(fabsf(v) > 65504.0f ? copysignf(65504.0f, v) : v);
}

// Fused gated RMS norm for the qwen4exp gated-delta-net tail: rms_norm(x) * gamma * sigmoid(z), written as F16 so the
// following Q8_0 -> F16 GEMM reads it directly. Same arithmetic as rms_norm_f32<256, true> + unary_mul(sigmoid) + the
// MMB F16 activation conversion, so it is bit-exact against that chain.
void ggml_cuda_op_gated_rms_norm_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Parity-test entry point: runs the block-per-row (wave == 0) or wave-per-row (wave != 0, gamma && ncols == 128 only;
// other shapes fall back to the block kernels) gated norm family on raw device pointers with the op's stride contract.
extern "C" { GGML_BACKEND_API int ggml_cuda_test_gated_rms_norm_f16(int wave,
        const float * x, const float * gamma, const float * z, void * dst,
        int ncols, int nh, int T, int64_t x_s1, int64_t x_s2, int64_t z_sh, int64_t z_st, int gamma_rows, float eps,
        void * stream); }
