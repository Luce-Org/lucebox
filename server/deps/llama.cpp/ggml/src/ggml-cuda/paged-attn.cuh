#pragma once

#include "common.cuh"

bool ggml_cuda_paged_attn_supported(const ggml_tensor * dst);

void ggml_cuda_paged_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
