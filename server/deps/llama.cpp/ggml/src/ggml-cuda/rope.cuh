#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

// ROPE (multi-section, F32) -> PERMUTE -> CONT in one kernel: the rotation is written straight into cont's layout.
void ggml_cuda_op_rope_permuted(ggml_backend_cuda_context & ctx, ggml_tensor * rope, ggml_tensor * cont);
