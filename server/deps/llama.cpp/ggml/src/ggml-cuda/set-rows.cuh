#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Fused dual set_rows: two independent quantized SET_ROWS in one launch,
// bit-identical to running them separately.
bool ggml_cuda_set_rows_dual_supported(const ggml_tensor * a, const ggml_tensor * b);
void ggml_cuda_op_set_rows_dual(ggml_backend_cuda_context & ctx, ggml_tensor * a, ggml_tensor * b);
