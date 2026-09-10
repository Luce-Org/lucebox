#pragma once
#include "common.cuh"

#if defined(GGML_USE_HIP)
// Caller supplies two distinct host buffers of height*width*32 floats each.
// Success fills both buffers synchronously and records one complete table call.
bool ggml_hip_vision_rotary(ggml_backend_cuda_context & ctx, int64_t height, int64_t width,
                          float * cosine, float * sine);
#endif
