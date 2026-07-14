#pragma once

#include "ggml-cuda/common.cuh"

void ggml_cuda_op_moe_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
