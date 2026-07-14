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
    const TargetLayer &   L,
    // Pass false when selected/weights are read back by the host instead of
    // feeding an in-graph mul_mat_id: ggml_argsort_top_k returns a strided
    // VIEW into the full [n_expert, n_tokens] argsort, and the hybrid raw
    // readback (ggml_backend_tensor_get, packed [n_used x n_tokens]) then
    // yields garbage expert ids for every token after the first (decode
    // reads one row and is unaffected). ggml_top_k is contiguous by
    // construction, and cheaper than a full argsort where the fusion
    // cannot apply anyway.
    bool                  allow_fused_router = true);

ggml_tensor * build_qwen35moe_ffn(
    ggml_context *        ctx,
    ggml_tensor *         cur,   // [hidden, n_tokens], post-attention normed
    const TargetWeights & w,
    const TargetLayer &   L,
    ggml_tensor **        selected_out = nullptr);

}  // namespace dflash::common
