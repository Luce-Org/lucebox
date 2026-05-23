// Shared qwen35-family graph helpers used by both dense qwen35 and qwen35moe.

#pragma once

#include "ggml.h"

namespace dflash::common {

inline ggml_tensor * rms_norm_mul(ggml_context * ctx, ggml_tensor * x,
                                  ggml_tensor * weight, float eps) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, weight);
}

// NVFP4 scale2: if weight has a per-tensor scale, multiply the matmul result
// by that scale. No-op when scale==1.0f (non-NVFP4 models).
inline ggml_tensor * apply_scale2(ggml_context * ctx, ggml_tensor * mm_result,
                                  float scale) {
    if (scale == 1.0f) return mm_result;
    return ggml_scale(ctx, mm_result, scale);
}

}  // namespace dflash::common
