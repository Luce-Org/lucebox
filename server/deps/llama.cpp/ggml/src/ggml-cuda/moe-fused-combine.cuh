#pragma once

#include "common.cuh"

void ggml_cuda_op_ds4_moe_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
