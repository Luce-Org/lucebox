// Custom Qwen3-0.6B drafter forward, in dflash, replacing libllama.
//
// Uses the FlashPrefill dispatch path for the attention compute. Single
// process, single backend context, single ggml allocator — no Python, no
// Triton, no subprocess.
//
// Public API:
//   bool load_qwen3_drafter_model(path, backend, out)  → load GGUF weights
//   bool forward_qwen3_drafter_model(weights, ids, out_q_capture, out_k_capture)
//   void free_qwen3_drafter_model(weights)
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

namespace dflash::common {

struct Qwen3DrafterLayer {
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

struct Qwen3DrafterWeights {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    ggml_type             weight_type  = GGML_TYPE_BF16;
    ggml_type             compute_type = GGML_TYPE_BF16;

    ggml_tensor * tok_embd    = nullptr;  // [hidden, vocab]
    ggml_tensor * out_norm    = nullptr;  // [hidden]
    ggml_tensor * output      = nullptr;  // [hidden, vocab] (lm_head)

    std::vector<Qwen3DrafterLayer> layers;  // size = n_layer = 28

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

bool load_qwen3_drafter_model(const std::string & gguf_path,
                              ggml_backend_t backend,
                              Qwen3DrafterWeights & out);

void free_qwen3_drafter_model(Qwen3DrafterWeights & w);

// Custom Qwen3-0.6B forward, fused with Liu Q-hook tail attention scoring.
//
// Inputs:
//   w           — loaded weights (must be on the selected GPU backend)
//   ids         — input token IDs of length S (drafter vocab)
//   n_lookahead — number of query tokens for scorer attention (=8)
//   score_query_end — exclusive end of query window; negative selects the tail
//
// Outputs:
//   running_max — flat [n_lookahead, S] f32, max-over-heads-and-layers of
//                 softmax(Q_query @ K^T / sqrt(D)) per (lookahead, key) pair.
//                 Caller does AvgPool + chunk-top-K + span merge.
//
// Returns true on success. On failure sets last_error and returns false.
bool forward_qwen3_drafter_model(
    const Qwen3DrafterWeights & w,
    const std::vector<int32_t> & ids,
    int n_lookahead,
    std::vector<float> & running_max,
    int score_query_end = -1);

struct QueryCaptureSlice {
    int chunk_offset = 0;
    int query_offset = 0;
    int tokens = 0;

    bool valid() const { return tokens > 0; }
};

inline QueryCaptureSlice query_capture_slice(
        int query_start,
        int query_end,
        int chunk_start,
        int chunk_tokens) {
    const int chunk_end = chunk_start + chunk_tokens;
    const int overlap_start = query_start > chunk_start ? query_start : chunk_start;
    const int overlap_end = query_end < chunk_end ? query_end : chunk_end;
    if (overlap_start >= overlap_end) return {};
    return {
        overlap_start - chunk_start,
        overlap_start - query_start,
        overlap_end - overlap_start,
    };
}

inline size_t count_nonfinite_scores(const float * values, size_t count) {
    size_t nonfinite = 0;
    for (size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) ++nonfinite;
    }
    return nonfinite;
}

} // namespace dflash::common
