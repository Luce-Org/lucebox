#pragma once
#include "common.cuh"
#if defined(GGML_USE_HIP)
// One workspace/handle/event shared by explicit linear and AV operations.
constexpr size_t GGML_HIP_VISION_WORKSPACE_BYTES = 76ULL*1024*1024;
void ggml_hip_vision_workspace_acquire(ggml_backend_cuda_context & ctx);
void ggml_hip_vision_workspace_record(ggml_backend_cuda_context & ctx);
// DS4V-only explicit op; no ordinary MUL_MAT/ADD fusion or fallback.
bool ggml_hip_vision_bias_supported(const ggml_tensor * dst);
void ggml_hip_vision_bias(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#endif
