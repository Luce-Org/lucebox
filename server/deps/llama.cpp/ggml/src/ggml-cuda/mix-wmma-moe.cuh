#pragma once

#include "common.cuh"

// F16 WMMA routed-expert GEMM for qtype 105/106 prefill batches on RDNA3.
bool ggml_cuda_mix_wmma_moe_enabled(const ggml_tensor * src0, const ggml_tensor * src1,
                                    const ggml_tensor * ids, int64_t n_tokens, int cc);
void ggml_cuda_mix_wmma_moe(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                            const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
void ggml_cuda_mix_wmma_moe_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * src0_a, const ggml_tensor * src0_b,
                                 const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst_a, ggml_tensor * dst_b);
void ggml_cuda_mix_wmma_moe_pair_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * w_up, const ggml_tensor * w_gate,
                                     const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * gate_dst,
                                     ggml_tensor * glu_dst, float limit);
