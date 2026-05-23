// Qwen35MoE FFN builder used by the shared qwen35-family graph path.

#pragma once

#include "internal.h"

namespace dflash::common {

ggml_tensor * build_qwen35moe_ffn(
    ggml_context *        ctx,
    ggml_tensor *         cur,   // [hidden, n_tokens], post-attention normed
    const TargetWeights & w,
    const TargetLayer &   L);

}  // namespace dflash::common
