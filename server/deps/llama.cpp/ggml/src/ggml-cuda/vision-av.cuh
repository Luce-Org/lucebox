#pragma once
#include "common.cuh"

#if defined(GGML_USE_HIP)
bool ggml_hip_vision_av_f32_supported(int device, const ggml_tensor * op);
void ggml_hip_vision_av_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#endif
