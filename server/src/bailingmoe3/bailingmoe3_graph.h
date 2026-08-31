#pragma once

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

namespace dflash::common {

struct TargetLayer;
struct TargetWeights;

ggml_tensor * build_bailingmoe3_mla_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const TargetWeights & weights,
    const TargetLayer & layer,
    ggml_tensor * cur,
    ggml_tensor * positions,
    ggml_tensor * cache_k,
    ggml_tensor * cache_v,
    ggml_tensor * attn_mask,
    int kv_start,
    int n_tokens);

ggml_tensor * build_bailingmoe3_kda_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const TargetWeights & weights,
    const TargetLayer & layer,
    ggml_tensor * cur,
    ggml_tensor * conv_state,
    ggml_tensor * ssm_state,
    int n_tokens);

}  // namespace dflash::common
