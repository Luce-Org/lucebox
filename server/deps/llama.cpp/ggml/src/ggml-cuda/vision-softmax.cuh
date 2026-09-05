#pragma once

#include "common.cuh"

#if defined(GGML_USE_HIP)
bool ggml_hip_vision_softmax_f32_supported(int device, const ggml_tensor * op);
void ggml_cuda_op_soft_max_vision_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#endif
