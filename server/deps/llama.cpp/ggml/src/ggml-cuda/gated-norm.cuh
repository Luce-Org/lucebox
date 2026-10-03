#pragma once
#include "common.cuh"

// Fused gated RMS norm for the qwen4exp gated-delta-net tail: rms_norm(x) * gamma * sigmoid(z), written as F16 so the
// following Q8_0 -> F16 GEMM reads it directly. Same arithmetic as rms_norm_f32<256, true> + unary_mul(sigmoid) + the
// MMB F16 activation conversion, so it is bit-exact against that chain.
void ggml_cuda_op_gated_rms_norm_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
