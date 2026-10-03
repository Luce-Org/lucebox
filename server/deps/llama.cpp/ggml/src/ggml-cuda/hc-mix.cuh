#pragma once
#include "common.cuh"
struct ggml_cuda_hc_mix_args {
    const ggml_tensor * xn;
    const ggml_tensor * gate;
    ggml_tensor * dst;
    int hc;
    float scale;
    float bias;
};
void ggml_cuda_op_hc_mix_reduce(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_mix_args & args);
