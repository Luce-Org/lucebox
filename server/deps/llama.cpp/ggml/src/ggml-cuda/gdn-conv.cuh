#pragma once
#include "common.cuh"
struct ggml_cuda_gdn_conv_match {
    int concat_idx = -1, conv_idx = -1;   // node indices
    const ggml_tensor * input = nullptr;  // CONCAT src[1], transpose view of [C, T]
    const ggml_tensor * state = nullptr;  // [3, C] F32 contiguous
    const ggml_tensor * concat = nullptr; // [T+3, C] F32
    const ggml_tensor * w = nullptr;
    ggml_tensor * conv_out = nullptr;
    int64_t C = 0, T = 0, tail_from = 0;  // first concat column that must be materialized
    // q/k L2 norm folded into the conv epilogue (SSM_CONV op_params[1..4], set by the qwen4exp graph): channels
    // [0, qk_channels) are rms_norm(eps) * post_scale + bias per 128-channel head. 0 = plain conv.
    int qk_channels = 0;
    float norm_eps = 0.0f, norm_scale = 1.0f, norm_bias = 0.0f;
};
// Parity-test entry point: plain conv (norm == 0) or the qk-norm split launch (norm != 0) on raw device pointers;
// saved_state is the [3, C] history (row stride `state_stride` floats), x is [C, T], w is [4, C], y is [C, T].
extern "C" { GGML_BACKEND_API int ggml_cuda_test_gdn_conv_direct(int norm, const float * saved_state, const float * x,
        const float * w, float * y, int C, int T, int state_stride, int qk_channels, float eps, float post_scale,
        float bias, void * stream); }
bool ggml_cuda_gdn_conv_match_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m);
bool ggml_cuda_gdn_conv_match_at_conv(const ggml_cgraph * cgraph, int j, ggml_cuda_gdn_conv_match & m);
void ggml_cuda_gdn_conv_write_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m);
void ggml_cuda_gdn_conv_direct(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m);
