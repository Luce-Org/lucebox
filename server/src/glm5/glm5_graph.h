// GLM-5.3-Flash step graph: n_tokens positions of one sequence, all layers,
// updating Glm5Cache in place.

#pragma once

#include "glm5_model.h"

#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace luce::common {

struct Glm5StepGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;

    // Inputs; null when unused.
    ggml_tensor * tokens      = nullptr;  // I32 [n_tokens]
    ggml_tensor * causal_mask = nullptr;  // F16 [kv_pad, n_tokens]; dense steps
    ggml_tensor * tail        = nullptr;  // F32 [kv_pad, n_tokens]; sparse: 1 on the query's unfinished pool
    ggml_tensor * pool_vis    = nullptr;  // F32 [1, kv_pad/kpool, n_tokens]; sparse: 1 on visible complete pools
    ggml_tensor * pool_bias   = nullptr;  // F32 [n_pool, n_tokens]; sparse: 0 visible, -inf otherwise

    ggml_tensor * logits = nullptr;  // F32 [n_vocab, n_out]

    int  n_tokens   = 0;
    int  kv_start   = 0;
    int  kv_pad     = 0;   // attended span, kv_start + n_tokens rounded up to 256
    int  n_pool     = 0;   // complete pools of kpool tokens after this step
    bool sparse     = false;
    bool all_logits = false;

    void free();
};

// True once a query can see more than top_k/kpool pools; until then DSA == causal.
bool glm5_step_is_sparse(const Glm5Weights & w, int kv_end);

bool build_glm5_step_graph(const Glm5Weights & w, const Glm5Cache & cache,
                           int n_tokens, int kv_start, bool all_logits,
                           Glm5StepGraph & out);

// Host-side mask inputs for positions kv_start.., laid out as in Glm5StepGraph.
struct Glm5DsaHostInputs {
    std::vector<uint16_t> causal_mask;  // fp16
    std::vector<float>    tail;
    std::vector<float>    pool_vis;
    std::vector<float>    pool_bias;
};
void glm5_dsa_host_inputs(int kv_start, int n_tokens, int kv_pad, int n_pool,
                          int kpool, bool select_tail, bool sparse,
                          Glm5DsaHostInputs & out);

// Upload the inputs once the graph is allocated.
void set_glm5_step_inputs(const Glm5Weights & w, const Glm5StepGraph & g,
                          const int32_t * tokens);

}  // namespace luce::common
