#pragma once

#include "common.cuh"

void ggml_cuda_op_ds4_indexer_qat(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst);

void ggml_cuda_op_ds4_indexer_score(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst);

void ggml_cuda_op_ds4_indexer_mask(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst);
