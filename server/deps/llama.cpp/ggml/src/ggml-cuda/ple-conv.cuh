#pragma once
#include "common.cuh"
struct ggml_cuda_ple_conv_match {
    int concat_idx = -1, first_tap_idx = -1, silu_idx = -1;
    const ggml_tensor * x = nullptr;       // [C, T] F32 contiguous
    const ggml_tensor * state = nullptr;   // [H, C] F32 contiguous (H = (K-1)*dil)
    const ggml_tensor * concat = nullptr;  // [T+H, C]
    const ggml_tensor * w = nullptr;
    ggml_tensor * out = nullptr;           // silu output [C, T]
    int64_t C = 0, T = 0, H = 0, K = 0, dil = 0, tail_from = 0;
    int starts[8] = {0};
};
bool ggml_cuda_ple_conv_match_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_ple_conv_match & m);
bool ggml_cuda_ple_conv_match_at_tap(const ggml_cgraph * cgraph, int i, ggml_cuda_ple_conv_match & m);
void ggml_cuda_ple_conv_write_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_ple_conv_match & m);
void ggml_cuda_ple_conv_direct(ggml_backend_cuda_context & ctx, const ggml_cuda_ple_conv_match & m);
