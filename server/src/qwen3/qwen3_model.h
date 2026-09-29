// Qwen3-0.6B model weights, loaded in-process (no libllama).
//
// Used by Qwen3Backend for standalone inference; the pflash drafter moved to
// the Qwen3.5-0.8B scorer under src/pflash/.
//
// Public API:
//   bool load_qwen3_model(path, backend, out)  → load GGUF weights
//   void free_qwen3_model(weights)
//
#pragma once

#include "ggml.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

namespace luce::common {

struct Qwen3Layer {
    ggml_tensor * attn_norm   = nullptr;  // [hidden]
    ggml_tensor * wq          = nullptr;  // [hidden, q_dim] = [1024, 2048]
    ggml_tensor * wk          = nullptr;  // [hidden, kv_dim] = [1024, 1024]
    ggml_tensor * wv          = nullptr;  // [hidden, kv_dim]
    ggml_tensor * wo          = nullptr;  // [q_dim, hidden] = [2048, 1024]
    ggml_tensor * q_norm      = nullptr;  // [head_dim] = [128]
    ggml_tensor * k_norm      = nullptr;  // [head_dim]
    ggml_tensor * ffn_norm    = nullptr;  // [hidden]
    ggml_tensor * ffn_gate    = nullptr;  // [hidden, ffn]
    ggml_tensor * ffn_up      = nullptr;  // [hidden, ffn]
    ggml_tensor * ffn_down    = nullptr;  // [ffn, hidden]
};

struct Qwen3Weights {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    ggml_type             weight_type  = GGML_TYPE_BF16;
    ggml_type             compute_type = GGML_TYPE_BF16;

    ggml_tensor * tok_embd    = nullptr;  // [hidden, vocab]
    ggml_tensor * out_norm    = nullptr;  // [hidden]
    ggml_tensor * output      = nullptr;  // [hidden, vocab] (lm_head)

    std::vector<Qwen3Layer> layers;  // size = n_layer = 28

    // Architecture metadata.
    int n_layer    = 28;
    int n_head     = 16;
    int n_head_kv  = 8;
    int n_embd     = 1024;
    int n_ff       = 3072;
    int head_dim   = 128;
    int n_vocab    = 151936;
    int n_ctx_max  = 40960;
    float rope_theta = 1000000.0f;
};

bool load_qwen3_model(const std::string & gguf_path,
                      ggml_backend_t backend,
                      Qwen3Weights & out);

void free_qwen3_model(Qwen3Weights & w);

} // namespace luce::common
