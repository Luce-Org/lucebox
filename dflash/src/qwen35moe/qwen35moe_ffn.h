// Qwen35MoE FFN builder used by the shared qwen35-family graph path.

#pragma once

#include "internal.h"

namespace dflash::common {

struct Qwen35MoeRouterOutputs {
    ggml_tensor * selected = nullptr; // [n_used, n_tokens] i32
    ggml_tensor * weights  = nullptr; // [n_used, n_tokens] f32, post-normalized/scaled
};

Qwen35MoeRouterOutputs build_qwen35moe_router(
    ggml_context *        ctx,
    ggml_tensor *         cur,   // [hidden, n_tokens], post-attention normed
    const TargetWeights & w,
    const TargetLayer &   L);

ggml_tensor * build_qwen35moe_ffn(
    ggml_context *        ctx,
    ggml_tensor *         cur,   // [hidden, n_tokens], post-attention normed
    const TargetWeights & w,
    const TargetLayer &   L,
    ggml_tensor **        selected_out = nullptr);

}  // namespace dflash::common
