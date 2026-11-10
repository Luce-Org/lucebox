#pragma once

#include "common.cuh"

// shared_logit set: the shared output is shared_ungated times sigmoid(shared_logit) (a [1, T] gate), computed in the
// combine kernel with the SIGMOID and MUL ops' arithmetic; dst->src[2] is not read.
void ggml_cuda_op_ds4_moe_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                  const ggml_tensor * shared_ungated = nullptr, const ggml_tensor * shared_logit = nullptr);

// Tokens one launch block of ggml_cuda_moe_route_weights_norm holds: up to this many, its output may share memory
// with its inputs.
#define MOE_ROUTE_NORM_MAX_TOKENS 1024

// dst [k, T] = renormalized router weights: probs3 [1, n_expert, T] gathered by ids [k, T], divided by their row sum
// clamped to [clamp_min, clamp_max] (GET_ROWS, SUM_ROWS, CLAMP, DIV with the same bits). Callers check
// ggml_cuda_moe_route_weights_norm_ok first: more tokens than MOE_ROUTE_NORM_MAX_TOKENS need dst apart from both inputs.
bool ggml_cuda_moe_route_weights_norm_ok(const ggml_tensor * probs3, const ggml_tensor * ids, const ggml_tensor * dst);
void ggml_cuda_moe_route_weights_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * probs3,
                                      const ggml_tensor * ids, float clamp_min, float clamp_max, ggml_tensor * dst);

// The softmax router of a top-k MoE in one launch per token row: probs = SOFT_MAX(logits * scale) over n_expert
// (a power of two up to 1024), ARGSORT descending, the top k (k <= WARP_SIZE) picks, and their renormalized weights
// (GET_ROWS of probs, SUM_ROWS, CLAMP, DIV). Each phase replays the unfused kernel's arithmetic (soft_max_f32,
// k_argsort_f32_i32, moe_route_weights_norm_kernel), so every output has the unfused bits. Outputs, each optional:
// sorted (the argsort rows; only their first k columns are written), picks [k, T] contiguous, lut0/lut1 [n_expert]
// I32 tables gathered by the picks into out0/out1 [k, T], and weights [k, T].
struct ggml_cuda_moe_router_topk_args {
    const float * logits = nullptr;
    int64_t logits_row = 0;   // floats between token rows
    int n_expert = 0, k = 0, n_tokens = 0;
    float scale = 1.0f, clamp_min = 0.0f, clamp_max = 0.0f;
    int32_t * sorted = nullptr;
    int64_t sorted_row = 0;   // int32s between token rows
    int32_t * picks = nullptr;
    const int32_t * lut0 = nullptr;
    int32_t * out0 = nullptr;
    const int32_t * lut1 = nullptr;
    int32_t * out1 = nullptr;
    float * weights = nullptr;
    uint32_t * post = nullptr;   // a wake-hold word: block 0 adds 1 at its end (ggml_moe_fused_wake_hold)
};
// The wake-hold words in pinned host memory, visible to every device: GGML_MOE_FUSED_WAKE_SLOTS posts of the routers,
// one per slot, then as many counts the holds have seen (ggml_moe_fused_wake_hold).
uint32_t * ggml_cuda_wake_flags();
// The post word of a router weights node marked with a wake-hold slot (ggml_backend_cuda_div_wake_post: op_params[0]),
// else nullptr.
uint32_t * ggml_cuda_wake_post_word(const ggml_tensor * weights);
// For the router's unfused and partly fused paths: a one-lane launch that posts after the weights node.
void ggml_cuda_wake_post(ggml_backend_cuda_context & ctx, const ggml_tensor * weights);
bool ggml_cuda_moe_router_topk_supported(int n_expert, int k);
void ggml_cuda_moe_router_topk(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_router_topk_args & a);
