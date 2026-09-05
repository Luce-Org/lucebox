#pragma once
#include "ggml.h"

namespace dflash::common::detail {
inline ggml_tensor * build_rms_norm(ggml_context * ctx, ggml_tensor * x,
                                    ggml_tensor * weight, float eps) {
    ggml_tensor * normed = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, normed, weight);
}
}
