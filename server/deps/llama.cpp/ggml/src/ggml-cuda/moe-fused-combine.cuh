#pragma once

#include "common.cuh"

// shared_logit set: the shared output is shared_ungated times sigmoid(shared_logit) (a [1, T] gate), computed in the
// combine kernel with the SIGMOID and MUL ops' arithmetic; dst->src[2] is not read.
void ggml_cuda_op_ds4_moe_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                  const ggml_tensor * shared_ungated = nullptr, const ggml_tensor * shared_logit = nullptr);

// dst [k, T] = renormalized router weights: probs3 [1, n_expert, T] gathered by ids [k, T], divided by their row sum
// clamped to [clamp_min, clamp_max] (GET_ROWS, SUM_ROWS, CLAMP, DIV with the same bits).
void ggml_cuda_moe_route_weights_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * probs3,
                                      const ggml_tensor * ids, float clamp_min, float clamp_max, ggml_tensor * dst);
