#pragma once
#include "ggml.h"

namespace dflash::common::detail {
inline ggml_tensor * build_rms_norm(ggml_context * ctx, ggml_tensor * x,
                                    ggml_tensor * weight, float eps) {
    // HIP binary broadcast does not accept BF16 affine operands. Widen only
    // this small vector; stored weights and matrix products remain unchanged.
    if (weight->type == GGML_TYPE_BF16) weight = ggml_cast(ctx, weight, GGML_TYPE_F32);
    ggml_tensor * normed = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, normed, weight);
}
}
