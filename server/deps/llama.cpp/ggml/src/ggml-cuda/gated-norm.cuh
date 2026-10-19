#pragma once
#include "common.cuh"

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
