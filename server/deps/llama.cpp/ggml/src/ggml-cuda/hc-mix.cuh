#pragma once
#include "common.cuh"
struct ggml_cuda_hc_mix_args {
    const ggml_tensor * xn;
    const ggml_tensor * gate;
    ggml_tensor * dst;
    int hc;
    float scale;
    float bias;
    void * q8 = nullptr;   // also write dst's q8_1 form here (a q8 memo buffer; rows unpadded, n_embd % 512 == 0)
};
void ggml_cuda_op_hc_mix_reduce(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_mix_args & args);
// Whether ggml_cuda_op_hc_mix_reduce takes its hc4 kernel (the one that can write args.q8) without staging.
bool ggml_cuda_hc_mix_reduce_q8_ok(const ggml_cuda_hc_mix_args & args);
