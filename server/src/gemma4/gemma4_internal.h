// Gemma4 (iSWA + MoE) target structs for dflash daemon.
//
// Architecture summary (from Google Gemma-4 config):
//   - Hybrid iSWA: per-layer sliding window pattern (full vs SWA).
//   - MoE on sparse layers (all but lead dense). Routing: softmax-normalized.
//   - Per-layer embeddings: additional per-layer token embedding + projection.
//   - KV sharing: later layers may reuse KV from earlier layers.
//   - Logit softcapping after final lm_head.
//   - Q/K RMSNorm per head, RoPE (with per-layer-type freq base).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#include "internal.h"  // CpuEmbedder

namespace dflash::common {

struct Gemma4Layer {
    // Pre-attn norm
    ggml_tensor * attn_norm       = nullptr;  // [n_embd]

    // Attention projections
    ggml_tensor * wq              = nullptr;  // [n_embd, n_head * head_dim]
    ggml_tensor * wk              = nullptr;  // [n_embd, n_head_kv * head_dim] (nullptr on KV-reuse layers)
    ggml_tensor * wv              = nullptr;  // [n_embd, n_head_kv * head_dim] (nullptr on KV-reuse layers)
    ggml_tensor * wo              = nullptr;  // [n_head * head_dim, n_embd]
    ggml_tensor * q_norm          = nullptr;  // [head_dim]
    ggml_tensor * k_norm          = nullptr;  // [head_dim]

    // Post-attention norm
    ggml_tensor * attn_post_norm  = nullptr;  // [n_embd]

    // Dense FFN (lead dense layers only)
    ggml_tensor * ffn_norm        = nullptr;  // [n_embd]
    ggml_tensor * ffn_gate        = nullptr;  // [n_embd, n_ff]
    ggml_tensor * ffn_up          = nullptr;  // [n_embd, n_ff]
    ggml_tensor * ffn_down        = nullptr;  // [n_ff, n_embd]
    ggml_tensor * ffn_post_norm   = nullptr;  // [n_embd]

    // MoE FFN (sparse layers): shared expert
    ggml_tensor * ffn_norm_moe    = nullptr;  // [n_embd] (pre-norm for shared exp)
    ggml_tensor * ffn_gate_shexp  = nullptr;  // [n_embd, n_ff_shexp]
    ggml_tensor * ffn_up_shexp    = nullptr;  // [n_embd, n_ff_shexp]
    ggml_tensor * ffn_down_shexp  = nullptr;  // [n_ff_shexp, n_embd]
    ggml_tensor * ffn_post_norm_1 = nullptr;  // [n_embd]

    // MoE FFN (sparse layers): routed experts
    ggml_tensor * ffn_pre_norm_2  = nullptr;  // [n_embd]
    ggml_tensor * ffn_gate_inp    = nullptr;  // [n_embd, n_expert] router weights
    ggml_tensor * ffn_gate_inp_s  = nullptr;  // [n_embd] router scale
    ggml_tensor * ffn_gate_up_exps = nullptr; // packed gate+up for all experts
    ggml_tensor * ffn_down_exps   = nullptr;  // packed down for all experts
    ggml_tensor * ffn_down_exps_s = nullptr;  // scale for quantized down experts
    ggml_tensor * ffn_post_norm_2 = nullptr;  // [n_embd]

    // Per-layer embedding gate + projection
    ggml_tensor * per_layer_inp_gate   = nullptr;  // [n_embd, n_embd_per_layer]
    ggml_tensor * per_layer_proj       = nullptr;  // [n_embd_per_layer, n_embd]
    ggml_tensor * per_layer_post_norm  = nullptr;  // [n_embd]

    // Layer output scale
    ggml_tensor * out_scale       = nullptr;  // scalar or [1]

    // RoPE freq factors (full-attention layers only)
    ggml_tensor * rope_freqs      = nullptr;
};

struct Gemma4Weights {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;

    // Global tensors
    ggml_tensor * tok_embd               = nullptr;  // [n_embd, n_vocab]
    ggml_tensor * out_norm               = nullptr;  // [n_embd]
    ggml_tensor * output                 = nullptr;  // [n_embd, n_vocab] (lm_head, may be tied to tok_embd)
    ggml_tensor * rope_freqs_global      = nullptr;  // [head_dim/2] global rope freq factors
    ggml_tensor * per_layer_tok_embd     = nullptr;  // [n_embd_per_layer * n_layer, n_vocab]
    ggml_tensor * per_layer_model_proj   = nullptr;  // [n_embd, n_embd_per_layer * n_layer]
    ggml_tensor * per_layer_proj_norm    = nullptr;  // [n_embd_per_layer * n_layer]

    std::vector<Gemma4Layer> layers;

    CpuEmbedder embedder;

    // Architecture metadata
    int n_layer               = 0;
    int n_head                = 0;
    int n_head_kv             = 0;       // max n_head_kv (for backward compat)
    int head_dim              = 128;     // head_dim for SWA layers (smaller)
    int head_dim_full         = 128;     // head_dim for full-attention layers
    int n_embd                = 0;
    int n_ff                  = 0;       // dense FFN intermediate
    int n_ff_exp              = 0;       // expert FFN intermediate
    int n_ff_shexp            = 0;       // shared expert FFN intermediate
    int n_expert              = 0;
    int n_expert_used         = 0;
    int n_layer_dense_lead    = 1;
    int n_embd_per_layer      = 0;       // per-layer embedding dim
    int n_vocab               = 0;

    // Per-layer head counts (Gemma4 can have variable n_head_kv per layer)
    std::vector<int> n_head_kv_per_layer;

    // iSWA
    int  sliding_window       = 0;
    std::vector<bool> swa_layers;        // true = SWA, false = full attn
    std::vector<bool> has_kv;            // true = layer has own K/V
    int  kv_sharing_start     = 0;       // first layer that reuses KV

    // RoPE
    float rope_freq_base_full = 1000000.0f;
    float rope_freq_base_swa  = 10000.0f;

    // Logit softcapping
    float final_logit_softcap = 0.0f;

    // Tokenizer
    int32_t bos_id      = 2;
    int32_t eos_id      = 1;
    int32_t eos_chat_id = -1;

    float   norm_eps    = 1e-6f;
};

inline bool gemma4_is_swa_layer(const Gemma4Weights & w, int il) {
    return il < (int)w.swa_layers.size() && w.swa_layers[il];
}

inline bool gemma4_has_kv(const Gemma4Weights & w, int il) {
    return il < (int)w.has_kv.size() && w.has_kv[il];
}

inline int gemma4_head_dim(const Gemma4Weights & w, int il) {
    return gemma4_is_swa_layer(w, il) ? w.head_dim : w.head_dim_full;
}

inline int gemma4_n_head_kv(const Gemma4Weights & w, int il) {
    if (il < (int)w.n_head_kv_per_layer.size()) return w.n_head_kv_per_layer[il];
    return w.n_head_kv;
}

// GGUF loader
bool load_gemma4_gguf(const std::string & path,
                       ggml_backend_t backend,
                       Gemma4Weights & out);

void free_gemma4_weights(Gemma4Weights & w);

// KV cache
struct Gemma4Cache {
    int cur_pos  = 0;
    int max_ctx  = 0;
    int n_layer  = 0;
    int swa_size = 0;   // ring-buffer size for SWA layers (= sliding_window)
    int fa_window = 0;  // sparse decode window for full-attn layers (0 = full)
    int32_t last_tok = -1;  // argmax of last prefill token (for spec-decode entry)

    // Only layers where has_kv[il] == true have real K/V tensors.
    // KV-reuse layers reference an earlier layer's cache.
    std::vector<ggml_tensor *> k;   // n_layer entries (nullptr for reuse layers)
    std::vector<ggml_tensor *> v;
    std::vector<int>           kv_source;  // for each layer, which layer's KV to use

    // DFlash feature capture ring buffer (BF16, allocated when draft is active)
    ggml_tensor *         target_feat = nullptr;  // [fc_in, target_feat_cap]
    int                   target_feat_cap = 0;
    int                   n_capture_layers = 0;
    std::vector<int>      capture_layer_ids;

    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    // Separate context/buffer for target_feat (allocated after draft load)
    ggml_context *        feat_ctx = nullptr;
    ggml_backend_buffer_t feat_buf = nullptr;
};

bool  create_gemma4_cache(ggml_backend_t backend, const Gemma4Weights & w,
                           int max_ctx, Gemma4Cache & out);
void  free_gemma4_cache(Gemma4Cache & c);

// Allocate target_feat ring buffer (call after draft load determines n_capture_layers).
bool  create_gemma4_target_feat(ggml_backend_t backend, Gemma4Cache & cache,
                                 int n_capture_layers, int hidden_size, int cap);

// Snapshot
struct Gemma4Snapshot {
    int cur_pos = 0;
    int32_t last_tok = -1;
    std::vector<ggml_tensor *> k_snap;
    std::vector<ggml_tensor *> v_snap;
    ggml_tensor *             feat_snap = nullptr;  // [fc_in, feat_len]
    int                       feat_cap  = 0;
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

void free_gemma4_snapshot(Gemma4Snapshot & s);

// Forward: run a single step (prefill chunk or decode token).
// Returns logits for last token.
// token_ids: raw token IDs needed for per-layer embedding lookup (may be nullptr
//            if the model has no per-layer embeddings).
bool gemma4_step(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    Gemma4Cache &           cache,
    const float *           embed,
    const int32_t *         token_ids,
    int                     n_tokens,
    int                     kv_start,
    std::vector<float> &    out_logits);

// Verify batch: run forward pass returning argmax for ALL positions.
// Used by DFlash speculative decode target.
bool gemma4_verify_batch(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    Gemma4Cache &           cache,
    const float *           embed,
    const int32_t *         token_ids,
    int                     n_tokens,
    int                     kv_start,
    std::vector<int32_t> &  out_argmax);

// Project hidden states through lm_head (out_norm + output + softcap + argmax).
// Used by DFlash draft to convert draft hidden states to token IDs.
bool gemma4_project_hidden(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    const float *           hidden,
    int                     n_tokens,
    std::vector<int32_t> &  out_tokens);

// BSA sparse-FA prefill: process the full prompt at once using block-sparse
// attention for SWA layers (flash_prefill_forward_bf16). Full-attention layers
// use dense FA. Returns logits for the last token.  Populates the KV cache
// for subsequent decode. Returns false on failure.
bool gemma4_prefill_bsa(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    Gemma4Cache &           cache,
    const float *           embed,       // [n_embd, S] scaled
    const int32_t *         token_ids,   // [S] (for per-layer embedding)
    int                     S,           // total prompt length
    std::vector<float> &    out_logits);

}  // namespace dflash::common
