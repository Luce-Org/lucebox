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

#if defined(GGML_USE_HIP)
// A decode or verify step's conv in one launch (see ggml_cuda_try_gdn_conv_step): reads the [3, C] history, the [C, T]
// input (row stride x_stride) and the [4, C] weights; writes the concat (ci, null when nothing reads it), the new
// history, every token's window (win, null when absent), SILU(conv) and, for channels [0, qk_channels), the q/k
// RMS_NORM + SCALE (qk). state may be hist itself.
// With slot_ids (the batched decode): S sequences of one token each, whose histories are rows of stacked conv states.
// Sequence s reads and rewrites the history at hist/state + slot_ids[s] * slab and its x, y and qk at s times their
// slot strides; ci and win stay null.
constexpr int GGML_CUDA_GDN_CONV_STEP_MAX_T = 8;
struct ggml_cuda_gdn_conv_step_args {
    const float * hist = nullptr; const float * x = nullptr; const float * w = nullptr;
    float * ci = nullptr; float * state = nullptr; float * win = nullptr; float * y = nullptr; float * qk = nullptr;
    int64_t x_stride = 0, win_s1 = 0, win_s2 = 0, y_stride = 0, qk_stride = 0;   // in floats
    int C = 0, T = 0, qk_channels = 0;
    float eps = 0.0f, post_scale = 1.0f, bias = 0.0f;
    const int32_t * slot_ids = nullptr;
    int64_t slab = 0, x_slot = 0, y_slot = 0, qk_slot = 0;   // in floats
    int S = 1;
};
void ggml_cuda_gdn_conv_step(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_step_args & a);
#endif // defined(GGML_USE_HIP)
