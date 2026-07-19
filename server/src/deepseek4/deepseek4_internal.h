// DeepSeek V4 Flash target structs for dflash daemon.
//
// Architecture summary (from DeepSeek V4 Flash):
//   - MLA: Multi-head Latent Attention with low-rank Q projection and single
//     KV head shared across all attention heads.
//   - KV Compression: learned compressor pools SWA windows into compressed KV
//     rows (ratio-4 for even layers ≥2, ratio-128 for odd layers ≥2).
//   - Indexer: on ratio-4 layers, learned scorer selects top-k compressed rows.
//   - HC: Hierarchical Controller with 4 parallel residual streams, mixed via
//     Sinkhorn-normalized combine matrices at each sublayer.
//   - MoE: 256 routed experts (top-6) + 1 shared expert per layer.
//     First 3 layers use hash-based routing (token_id → expert_ids).
//   - RoPE: partial rotation (64 of 512 dims), YaRN scaling.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "internal.h"
#include "common/layer_split_utils.h"
#include "common/prefill_attention_mode.h"

namespace dflash::common {

// Layer-major prefill may schedule two 2K numerical bands while preserving
// the raw-cache rounding boundary between them.
inline constexpr int DS4_NUMERICAL_PREFILL_BAND = 2048;
inline constexpr int DS4_MAX_LAYER_MAJOR_PREFILL_TOKENS = 4096;

struct MoeHybridPlacement;
struct MoeHybridConfig;
struct MoeHybridRoutingStats;
class MoeHybridStreamEngine;

struct DeepSeek4StepTelemetry {
    uint64_t total_us = 0;
    uint64_t embed_us = 0;
    uint64_t hc_pre_attn_us = 0;
    uint64_t hc_pre_build_us = 0;
    uint64_t hc_pre_input_us = 0;
    uint64_t hc_pre_compute_us = 0;
    uint64_t attn_build_us = 0;
    uint64_t attn_compute_us = 0;
    uint64_t attn_read_us = 0;
    uint64_t hc_post_attn_us = 0;
    uint64_t hc_pre_ffn_us = 0;
    uint64_t ffn_build_us = 0;
    uint64_t ffn_compute_us = 0;
    uint64_t ffn_read_us = 0;
    uint64_t route_build_us = 0;
    uint64_t route_compute_us = 0;
    uint64_t route_read_us = 0;
    uint64_t route_select_us = 0;
    uint64_t ffn_eval_us = 0;
    uint64_t ffn_hot_us = 0;
    uint64_t ffn_cold_us = 0;
    uint64_t ffn_combine_us = 0;
    uint64_t ffn_partition_us = 0;
    uint64_t ffn_hot_graph_builds = 0;
    uint64_t ffn_hot_graph_hits = 0;
    uint64_t ffn_cold_graph_builds = 0;
    uint64_t ffn_cold_graph_hits = 0;
    uint64_t hc_post_ffn_us = 0;
    uint64_t output_us = 0;
    uint64_t sample_us = 0;
    uint64_t emit_us = 0;
    uint64_t full_graph_build_us = 0;
    uint64_t full_graph_alloc_us = 0;
    uint64_t full_graph_set_us = 0;
    uint64_t full_graph_compute_us = 0;
    uint64_t full_graph_read_us = 0;
    int hot_selected = 0;
    int cold_selected = 0;
};

// ─── Per-layer tensor pointers ──────────────────────────────────────────

struct DeepSeek4Layer {
    // ── Attention ────────────────────────────────────────────────────
    ggml_tensor * attn_norm          = nullptr;  // [n_embd]

    // Q low-rank path: x → q_a → norm → q_b → heads
    ggml_tensor * attn_q_a           = nullptr;  // [n_embd, n_lora_q]
    ggml_tensor * attn_q_a_norm      = nullptr;  // [n_lora_q]
    ggml_tensor * attn_q_b           = nullptr;  // [n_lora_q, n_head * head_dim]

    // KV path: single head, x → kv → norm → RoPE
    ggml_tensor * attn_kv            = nullptr;  // [n_embd, head_dim]
    ggml_tensor * attn_kv_a_norm     = nullptr;  // [head_dim]

    // Sink tokens (optional, for layers with learnable sink positions)
    ggml_tensor * attn_sinks         = nullptr;  // optional

    // Grouped low-rank output: heads → A → B → embd
    ggml_tensor * attn_output_a      = nullptr;  // [head_dim * n_head/n_out_group, n_lora_o]
    ggml_tensor * attn_output_b      = nullptr;  // [n_lora_o, n_embd]

    // ── KV Compression ───────────────────────────────────────────────
    // Compressor: pools SWA windows into compressed KV representations.
    ggml_tensor * attn_compressor_ape  = nullptr;  // [comp_width, ratio] positional bias
    ggml_tensor * attn_compressor_kv   = nullptr;  // [n_embd, comp_width] value projection
    ggml_tensor * attn_compressor_gate = nullptr;  // [n_embd, comp_width] score/gating
    ggml_tensor * attn_compressor_norm = nullptr;  // [head_dim] post-pool RMS norm

    // ── Indexer (ratio-4 layers only) ────────────────────────────────
    // Selects which compressed rows to attend via top-k scoring.
    ggml_tensor * indexer_attn_q_b     = nullptr;  // [n_lora_q, n_indexer_head * indexer_head_dim]
    ggml_tensor * indexer_proj         = nullptr;  // [n_embd, n_indexer_head] head weight projection

    // Indexer has its own compressor for the indexer key cache
    ggml_tensor * indexer_compressor_ape  = nullptr;
    ggml_tensor * indexer_compressor_kv   = nullptr;
    ggml_tensor * indexer_compressor_gate = nullptr;
    ggml_tensor * indexer_compressor_norm = nullptr;

    // ── HC Attention ─────────────────────────────────────────────────
    ggml_tensor * hc_attn_fn         = nullptr;  // [n_hc * n_embd, hc_mix_dim] F16
    ggml_tensor * hc_attn_scale      = nullptr;  // [3] F32 (pre_scale, post_scale, comb_scale)
    ggml_tensor * hc_attn_base       = nullptr;  // [n_hc] F32

    // ── FFN / MoE ────────────────────────────────────────────────────
    ggml_tensor * ffn_norm           = nullptr;  // [n_embd]

    // Router
    ggml_tensor * ffn_gate_inp       = nullptr;  // [n_embd, n_expert] router weights F16
    ggml_tensor * ffn_exp_probs_b    = nullptr;  // [n_expert] optional routing bias

    // Hash routing table (first n_hash_layer layers only)
    ggml_tensor * ffn_gate_tid2eid   = nullptr;  // [n_expert_used, n_vocab] I32

    // Routed experts (3D tensors: [in, out, n_expert])
    ggml_tensor * ffn_gate_exps      = nullptr;  // [n_embd, n_ff_exp, n_expert]
    ggml_tensor * ffn_up_exps        = nullptr;  // [n_embd, n_ff_exp, n_expert]
    ggml_tensor * ffn_down_exps      = nullptr;  // [n_ff_exp, n_embd, n_expert]

    // Shared expert
    ggml_tensor * ffn_gate_shexp     = nullptr;  // [n_embd, n_ff_exp]
    ggml_tensor * ffn_up_shexp       = nullptr;  // [n_embd, n_ff_exp]
    ggml_tensor * ffn_down_shexp     = nullptr;  // [n_ff_exp, n_embd]

    // ── HC FFN ───────────────────────────────────────────────────────
    ggml_tensor * hc_ffn_fn          = nullptr;  // [n_hc * n_embd, hc_mix_dim] F16
    ggml_tensor * hc_ffn_scale       = nullptr;  // [3] F32
    ggml_tensor * hc_ffn_base        = nullptr;  // [n_hc] F32
};

// ─── Global weights ─────────────────────────────────────────────────────

struct DeepSeek4Weights {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;

    // Global tensors
    ggml_tensor * tok_embd       = nullptr;  // [n_embd, n_vocab]
    ggml_tensor * out_norm       = nullptr;  // [n_embd]
    ggml_tensor * output         = nullptr;  // [n_embd, n_vocab]

    // Output HC (final residual stream merge before lm_head)
    ggml_tensor * output_hc_fn    = nullptr;  // [n_hc * n_embd, hc_mix_dim]
    ggml_tensor * output_hc_scale = nullptr;  // [3]
    ggml_tensor * output_hc_base  = nullptr;  // [n_hc]

    std::vector<DeepSeek4Layer> layers;

    CpuEmbedder embedder;

    // ── Architecture metadata ────────────────────────────────────────
    int n_layer           = 43;
    int n_embd            = 4096;
    int n_vocab           = 129280;
    int n_head            = 64;
    int n_head_kv         = 1;     // single KV head (MLA)
    int head_dim          = 512;   // = value_dim for DS4
    int n_rot             = 64;    // partial RoPE rotation dims
    int n_out_group       = 8;     // grouped output projection

    // Low-rank attention dimensions
    int n_lora_q          = 1024;  // Q low-rank bottleneck
    int n_lora_o          = 1024;  // output low-rank dim

    // MoE
    int n_expert          = 256;
    int n_expert_used     = 6;
    int n_expert_shared   = 1;
    int n_ff_exp          = 2048;
    int n_hash_layer      = 3;     // first 3 layers use hash routing
    float expert_weight_scale = 1.5f;

    // Compression
    int n_swa             = 128;   // raw SWA window size
    int n_indexer_head    = 64;
    int n_indexer_head_dim = 128;
    int n_indexer_top_k   = 512;

    // HC (Hierarchical Controller)
    int n_hc              = 4;
    int n_hc_sinkhorn_iter = 20;

    // Per-layer compression ratios (0 = no compression, 4 or 128)
    std::vector<uint32_t> compress_ratios;

    // RoPE
    float rope_freq_base        = 10000.0f;
    float rope_scale_factor     = 16.0f;
    float rope_yarn_beta_fast   = 32.0f;
    float rope_yarn_beta_slow   = 1.0f;
    float compress_rope_freq_base = 160000.0f;
    uint64_t rope_orig_ctx      = 65536;

    // Norms
    float rms_eps         = 1.0e-6f;
    float hc_eps          = 1.0e-6f;

    // SwiGLU
    float swiglu_clamp_exp = 10.0f;

    // Tokenizer special tokens from GGUF metadata.
    int32_t eos_id      = -1;
    int32_t eos_chat_id = -1;

    // MoE hybrid placement (deprecated — layer split replaces expert split)
    bool moe_hybrid       = false;

    // Runtime serving policy. These values are set by the backend after the
    // GGUF is loaded; they are not model metadata.
    int  routed_expert_top_k = 0;  // 0 = model default (n_expert_used)
    bool fused_decode        = false;
};

inline bool deepseek4_is_eos_tok(int tok, const DeepSeek4Weights & w) {
    return (w.eos_chat_id >= 0 && tok == w.eos_chat_id)
        || (w.eos_id >= 0 && tok == w.eos_id);
}

// ─── KV Cache ───────────────────────────────────────────────────────────

// Per-layer compressor rolling state
struct DeepSeek4CompressorState {
    ggml_tensor * state_kv    = nullptr;  // [window_size, head_dim] rolling buffer
    ggml_tensor * state_score = nullptr;  // [window_size, head_dim] rolling scores
};

// Per-layer cache
struct DeepSeek4LayerCache {
    // Raw SWA ring buffer
    ggml_tensor * raw_kv      = nullptr;  // [n_swa, head_dim] ring buffer

    // Compressed KV (grows during inference)
    ggml_tensor * comp_kv     = nullptr;  // [comp_cap, head_dim] compressed rows
    int           n_comp      = 0;        // current number of compressed rows

    // Indexer compressed KV (for ratio-4 layers with indexer)
    ggml_tensor * index_comp_kv = nullptr;  // [n_indexer_head * indexer_head_dim, index_comp_cap]
    int           n_index_comp  = 0;

    // Compressor rolling state
    DeepSeek4CompressorState attn_compressor;
    DeepSeek4CompressorState indexer_compressor;
};

struct DeepSeek4Cache {
    int cur_pos  = 0;
    int max_ctx  = 0;
    int n_layer  = 0;

    std::vector<DeepSeek4LayerCache> layers;
    PrefillAttentionMode prefill_mode = PrefillAttentionMode::Exact;

    // HC residual streams: [n_hc * n_embd] persistent state
    ggml_tensor * hc_state    = nullptr;  // [n_hc * n_embd]

    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

struct DeepSeek4Snapshot;

struct DeepSeek4RawRingSpan {
    int row = 0;
    int count = 0;
};

// ─── Configuration ──────────────────────────────────────────────────────

struct DeepSeek4BackendConfig {
    const char * model_path   = nullptr;
    DevicePlacement device;
    int          stream_fd    = -1;
    int          chunk        = 512;   // prefill chunk size
    PrefillAttentionMode prefill_mode = PrefillAttentionMode::Exact;
    int          max_ctx      = 0;     // 0 = auto from SWA + compression capacity
    int          expert_top_k = 0;     // 0 = use all model-routed experts
    bool         fused_decode = false; // single-graph GPU decode
};

// ─── Function declarations ──────────────────────────────────────────────

bool load_deepseek4_gguf(const std::string & path,
                          ggml_backend_t backend,
                          DeepSeek4Weights & out);

bool load_deepseek4_gguf_partial(const std::string & path,
                                  ggml_backend_t backend,
                                  const TargetLoadPlan & plan,
                                  DeepSeek4Weights & out);

void free_deepseek4_weights(DeepSeek4Weights & w);

// Release graph allocators and host mirrors that retain model tensor pointers.
// This must run before the owning ggml context is destroyed.
void deepseek4_release_runtime_graphs(const DeepSeek4Weights & w);

bool create_deepseek4_cache(ggml_backend_t backend,
                             const DeepSeek4Weights & w,
                             int max_ctx,
                             DeepSeek4Cache & out);

void free_deepseek4_cache(DeepSeek4Cache & c);
void reset_deepseek4_cache(DeepSeek4Cache & c);
int deepseek4_previous_raw_ring_spans(
    int kv_start,
    int n_swa,
    DeepSeek4RawRingSpan spans[2]);
bool deepseek4_snapshot_save(const DeepSeek4Cache & cache,
                             ggml_backend_t snapshot_backend,
                             DeepSeek4Snapshot & out);
bool deepseek4_snapshot_restore(const DeepSeek4Snapshot & snap,
                                DeepSeek4Cache & cache);

// Largest prefix of [kv_start, kv_start + n_tokens) that reaches at most the
// next learned-compressor boundary. Multi-token dynamic forwards split on
// this prefix so state writes after a boundary cannot race ahead of pooling.
int deepseek4_safe_compressor_batch_tokens(const DeepSeek4Weights & w,
                                           int kv_start,
                                           int n_tokens);

// Forward: single step (prefill chunk or decode token).
// embed: [n_embd, n_tokens] input embeddings (post-embedding lookup).
// hc_state: [n_hc * n_embd] persistent HC residual (updated in-place).
// Returns logits for last token.
struct Ds4VerifyHooks;

bool deepseek4_step(
    ggml_backend_t              backend,
    const DeepSeek4Weights &    w,
    DeepSeek4Cache &            cache,
    const float *               embed,
    int                         n_tokens,
    int                         kv_start,
    std::vector<float> &        out_logits,
    MoeHybridStorage *          moe_hybrid = nullptr,
    const int32_t *             token_ids = nullptr,
    MoeHybridStreamEngine *     stream_engine = nullptr,
    DeepSeek4StepTelemetry *    telemetry = nullptr,
    MoeHybridRoutingStats *     routing_stats = nullptr,
    Ds4VerifyHooks *            verify_hooks = nullptr);

// Optional hooks for the DSpark spec-decode batched verify (deepseek4_dspark).
// When set on a multi-token deepseek4_step_layer_range call they add: per-layer
// mean-over-HC feature capture and full per-position logits. Null on the normal
// (23 tok/s) decode path so it is completely unaffected.
struct Ds4VerifyHooks {
    const std::vector<int> * capture_layer_ids = nullptr;  // e.g. {40,41,42}
    std::vector<float> *     capture_out = nullptr;         // [n_cap*n_embd * n_tokens]
    std::vector<float> *     all_logits_out = nullptr;      // [n_vocab * n_tokens]
};

bool deepseek4_step_layer_range(
    ggml_backend_t              backend,
    const DeepSeek4Weights &    w,
    DeepSeek4Cache &            cache,
    std::vector<float> &        hc_state,
    const float *               embed,
    int                         n_tokens,
    int                         kv_start,
    int                         layer_begin,
    int                         layer_end,
    std::vector<float> *        out_logits,
    const int32_t *             token_ids = nullptr,
    DeepSeek4StepTelemetry *    telemetry = nullptr,
    bool                        allow_decode_graph_reuse = true,
    Ds4VerifyHooks *            verify_hooks = nullptr);

bool build_deepseek4_moe_hybrid_storage_from_file(
    const std::string &         path,
    ggml_backend_t              backend,
    const DeepSeek4Weights &    w,
    const MoeHybridPlacement &  placement,
    const MoeHybridConfig *     cfg_override,
    MoeHybridStorage &          out,
    std::string *               err = nullptr);

bool build_deepseek4_moe_hybrid_storage_from_file(
    const std::string &         path,
    ggml_backend_t              backend,
    const DeepSeek4Weights &    w,
    const MoeHybridPlacement &  placement,
    MoeHybridStorage &          out,
    std::string *               err = nullptr);

bool build_deepseek4_moe_hybrid_storage_from_file_with_mmap(
    const std::string &         path,
    ggml_backend_t              backend,
    const DeepSeek4Weights &    w,
    const MoeHybridPlacement &  placement,
    const MoeHybridConfig *     cfg_override,
    MoeHybridStorage &          out,
    std::string *               err = nullptr);

// Snapshot
struct DeepSeek4Snapshot {
    int cur_pos = 0;
    ggml_tensor * hc_state_snap = nullptr;
    // Per-layer: raw KV + compressed KV snapshots
    struct LayerSnap {
        ggml_tensor * raw_kv       = nullptr;
        ggml_tensor * comp_kv      = nullptr;
        int           n_comp       = 0;
        ggml_tensor * index_comp_kv = nullptr;
        int           n_index_comp = 0;
        DeepSeek4CompressorState attn_compressor;
        DeepSeek4CompressorState indexer_compressor;
    };
    std::vector<LayerSnap> layers;
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

void free_deepseek4_snapshot(DeepSeek4Snapshot & s);

}  // namespace dflash::common
