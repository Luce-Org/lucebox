#pragma once

#include "ggml.h"

#include <cmath>

namespace dflash::common {

struct TopKMoeRouterResult {
    ggml_tensor * selected   = nullptr; // [n_used, n_tokens] i32
    ggml_tensor * weights_2d = nullptr; // [n_used, n_tokens] f32
    ggml_tensor * weights_3d = nullptr; // [1, n_used, n_tokens] f32
};

// Build a sigmoid top-k MoE router using the node order recognized by
// ggml-cuda's topk-moe fusion:
//   sigmoid -> reshape -> optional bias add -> argsort_top_k -> get_rows
//   -> optional sum-normalize/clamp/div -> reshape -> optional scale.
//
// The selection bias is used only for expert choice. Weights are always gathered
// from the unbiased probabilities, matching llama.cpp's build_moe_ffn.
static inline TopKMoeRouterResult build_sigmoid_topk_moe_router(
        ggml_context * ctx,
        ggml_cgraph * gf,
        ggml_tensor * logits,          // [n_expert, n_tokens]
        ggml_tensor * selection_bias,  // [n_expert] or nullptr
        int           n_expert,
        int           n_used,
        int           n_tokens,
        bool          normalize_weights,
        float         weight_scale,
        bool          expand_weights) {
    ggml_tensor * probs = ggml_sigmoid(ctx, logits);

    // This reshape must be adjacent to sigmoid for ggml-cuda's topk-moe fusion.
    ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);

    ggml_tensor * selection_probs = selection_bias ? ggml_add(ctx, probs, selection_bias) : probs;
    ggml_tensor * selected = ggml_argsort_top_k(ctx, selection_probs, n_used);

    ggml_tensor * weights = ggml_get_rows(ctx, probs_3d, selected);
    weights = ggml_reshape_2d(ctx, weights, n_used, n_tokens);

    if (normalize_weights) {
        ggml_tensor * weights_sum = ggml_sum_rows(ctx, weights);
        weights_sum = ggml_clamp(ctx, weights_sum, 6.103515625e-5f, INFINITY);
        weights = ggml_div(ctx, weights, weights_sum);
    }

    ggml_tensor * weights_3d = ggml_reshape_3d(ctx, weights, 1, n_used, n_tokens);
    if (weight_scale != 0.0f && weight_scale != 1.0f) {
        weights_3d = ggml_scale(ctx, weights_3d, weight_scale);
    }
    if (expand_weights && gf) {
        ggml_build_forward_expand(gf, weights_3d);
    }

    TopKMoeRouterResult out;
    out.selected = selected;
    out.weights_3d = weights_3d;
    out.weights_2d = ggml_reshape_2d(ctx, weights_3d, n_used, n_tokens);
    return out;
}

} // namespace dflash::common
