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
#include "common/expert_split_compute_runtime.h"
#include "common/layer_split_utils.h"

namespace dflash::common {

struct MoeHybridPlacement;
struct MoeHybridConfig;
struct MoeHybridRoutingStats;
struct MoeExpertLayer;
struct MoeExpertCompute;
class ExpertIpcClient;

struct DeepSeek4StepTelemetry {
    uint64_t total_us = 0;
    uint64_t embed_us = 0;
    uint64_t full_graph_build_us = 0;
    uint64_t full_graph_alloc_us = 0;
    uint64_t full_graph_set_us = 0;
    uint64_t full_graph_compute_us = 0;
    uint64_t full_graph_read_us = 0;
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
    uint64_t ffn_cache_promote_us = 0;
    uint64_t ffn_hot_graph_builds = 0;
    uint64_t ffn_hot_graph_hits = 0;
    uint64_t ffn_cold_graph_builds = 0;
    uint64_t ffn_cold_graph_hits = 0;
    uint64_t worker_us = 0;
    uint64_t worker_parent_write_us = 0;
    uint64_t worker_parent_wait_us = 0;
    uint64_t worker_parent_read_us = 0;
    uint64_t worker_request_read_us = 0;
    uint64_t worker_partition_us = 0;
    uint64_t worker_resident_eval_us = 0;
    uint64_t worker_miss_build_us = 0;
    uint64_t worker_miss_eval_us = 0;
    uint64_t worker_request_bytes = 0;
    uint64_t worker_response_bytes = 0;
    uint64_t worker_hot_graph_builds = 0;
    uint64_t worker_hot_graph_hits = 0;
    uint64_t worker_cold_graph_builds = 0;
    uint64_t worker_cold_graph_hits = 0;
    uint64_t worker_hot_graph_build_us = 0;
    uint64_t worker_hot_input_us = 0;
    uint64_t worker_hot_compute_us = 0;
    uint64_t worker_hot_read_us = 0;
    uint64_t worker_cold_graph_build_us = 0;
    uint64_t worker_cold_input_us = 0;
    uint64_t worker_cold_compute_us = 0;
    uint64_t worker_cold_read_us = 0;
    uint64_t hc_post_ffn_us = 0;
    uint64_t output_us = 0;
    uint64_t sample_us = 0;
    uint64_t emit_us = 0;
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

    // MoE hybrid/expert-split placement state.
    bool moe_hybrid       = false;
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

    // Optional routing bias cached on host for CPU-side top-k selection.
    std::vector<float> route_bias_host;
    bool route_bias_loaded = false;
};

struct DeepSeek4Cache {
    int cur_pos  = 0;
    int max_ctx  = 0;
    int n_layer  = 0;

    std::vector<DeepSeek4LayerCache> layers;

    // HC residual streams: [n_hc * n_embd] persistent state
    ggml_tensor * hc_state    = nullptr;  // [n_hc * n_embd]

    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

struct DeepSeek4Snapshot;

// ─── Configuration ──────────────────────────────────────────────────────

struct DeepSeek4BackendConfig {
    const char * model_path   = nullptr;
    DevicePlacement device;
    int          stream_fd    = -1;
    int          chunk        = 512;   // prefill chunk size
    int          max_ctx      = 0;     // 0 = auto from SWA + compression capacity
};

struct DeepSeek4ExpertSplitGpuMemoryInfo {
    uint64_t free_bytes = 0;
    uint64_t total_bytes = 0;
};

struct DeepSeek4ExpertSplitBudgetMemoryInfo {
    uint64_t total_bytes = 0;
    uint64_t free_bytes = 0;
    uint64_t core_bytes = 0;
    uint64_t kv_bytes = 0;
    uint64_t warm_bytes = 0;
    uint64_t safety_bytes = 0;
    uint64_t total_expert_bytes = 0;
    bool parent_is_igpu = false;
    uint64_t host_available_bytes = 0;
    uint64_t host_budget_cap_bytes = 0;
};

struct DeepSeek4ExpertSplitHotCachePlan {
    uint64_t hot_bytes = 0;
    int cache_slots = 0;
};

int deepseek4_expert_split_prefill_chunk_limit_from_memory(
    int requested_chunk,
    bool expert_split_enabled,
    bool parent_is_gpu,
    const DeepSeek4ExpertSplitGpuMemoryInfo & memory);

bool deepseek4_expert_split_should_disable_cached_decode_from_memory(
    bool expert_split_enabled,
    bool parent_is_gpu,
    bool parent_is_cuda,
    bool parent_is_hip,
    const DeepSeek4ExpertSplitGpuMemoryInfo & memory);

bool deepseek4_expert_split_requires_parent_cuda_graph_disable(
    bool expert_split_enabled,
    bool parent_is_cuda,
    const std::vector<ExpertSplitComputeTargetRuntime> & targets);

bool deepseek4_should_use_gpu_resident_decode_ffn_policy(
    bool use_cached_decode,
    bool env_enabled,
    bool env_disabled,
    bool backend_is_hip,
    bool backend_is_integrated_gpu,
    bool worker_active);

uint64_t deepseek4_expert_split_budget_from_memory(
    const DeepSeek4ExpertSplitBudgetMemoryInfo & memory);

uint64_t deepseek4_expert_split_primary_capacity_for_targets(
    uint64_t expert_budget_bytes,
    uint64_t hot_budget_bytes,
    size_t configured_targets);

uint64_t deepseek4_expert_split_non_cpu_capacity_bytes(
    const std::vector<ExpertSplitTarget> & targets);

uint64_t deepseek4_expert_split_effective_budget_for_targets(
    uint64_t requested_budget_bytes,
    uint64_t total_expert_bytes,
    const std::vector<ExpertSplitTarget> & targets);

uint64_t deepseek4_expert_split_effective_hot_budget(
    uint64_t expert_budget_bytes,
    uint64_t hot_budget_bytes,
    size_t configured_targets);

bool deepseek4_expert_split_hash_ids_to_hotness_counts(
    const int32_t * expert_ids,
    int n_token_ids,
    int n_expert_used,
    int n_expert,
    std::vector<uint64_t> & out_counts);

bool deepseek4_expert_split_route_bias_to_hotness_counts(
    const float * route_bias,
    int n_expert,
    uint64_t layer_total,
    std::vector<uint64_t> & out_counts);

int deepseek4_expert_split_effective_cache_slots(
    int requested_cache_slots,
    size_t configured_targets);

DeepSeek4ExpertSplitHotCachePlan deepseek4_expert_split_hot_cache_plan(
    uint64_t expert_budget_bytes,
    uint64_t total_expert_bytes,
    int n_expert,
    int n_expert_used,
    bool parent_is_igpu,
    uint64_t igpu_host_cap_bytes = 0,
    uint64_t igpu_free_bytes = 0);

// ─── Function declarations ──────────────────────────────────────────────

bool load_deepseek4_gguf(const std::string & path,
                          ggml_backend_t backend,
                          DeepSeek4Weights & out);

bool load_deepseek4_gguf_partial(const std::string & path,
                                  ggml_backend_t backend,
                                  const TargetLoadPlan & plan,
                                  DeepSeek4Weights & out);

void free_deepseek4_weights(DeepSeek4Weights & w);

bool create_deepseek4_cache(ggml_backend_t backend,
                             const DeepSeek4Weights & w,
                             int max_ctx,
                             DeepSeek4Cache & out);

void free_deepseek4_cache(DeepSeek4Cache & c);
bool deepseek4_snapshot_save(const DeepSeek4Cache & cache,
                             ggml_backend_t snapshot_backend,
                             DeepSeek4Snapshot & out);
bool deepseek4_snapshot_restore(const DeepSeek4Snapshot & snap,
                                DeepSeek4Cache & cache);

// Forward: single step (prefill chunk or decode token).
// embed: [n_embd, n_tokens] input embeddings (post-embedding lookup).
// hc_state: [n_hc * n_embd] persistent HC residual (updated in-place).
// When want_logits is false, the step updates KV / routing state but skips the
// final output head and logits readback.
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
    ExpertIpcClient *  expert_worker = nullptr,
    bool                        worker_owns_hot_ids = false,
    bool                        want_logits = true,
    bool                        disable_cached_decode = false,
    DeepSeek4StepTelemetry *    telemetry = nullptr,
    MoeHybridRoutingStats *     routing_stats = nullptr,
    MoeExpertCompute *          expert_compute = nullptr,
    const MoeExpertLayer *      expert_layers = nullptr);

// Compatibility overload for builds that still route call sites through the
// original DeepSeek4 step signature without an explicit cached-decode toggle.
bool deepseek4_step(
    ggml_backend_t              backend,
    const DeepSeek4Weights &    w,
    DeepSeek4Cache &            cache,
    const float *               embed,
    int                         n_tokens,
    int                         kv_start,
    std::vector<float> &        out_logits,
    MoeHybridStorage *          moe_hybrid,
    const int32_t *             token_ids,
    ExpertIpcClient *           expert_worker,
    bool                        worker_owns_hot_ids,
    bool                        want_logits,
    DeepSeek4StepTelemetry *    telemetry,
    MoeHybridRoutingStats *     routing_stats = nullptr,
    MoeExpertCompute *          expert_compute = nullptr,
    const MoeExpertLayer *      expert_layers = nullptr);

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
    DeepSeek4StepTelemetry *    telemetry = nullptr);

bool build_deepseek4_moe_hybrid_storage_from_file(
    const std::string &         path,
    ggml_backend_t              backend,
    const DeepSeek4Weights &    w,
    const MoeHybridPlacement &  placement,
    const MoeHybridConfig *     cfg_override,
    MoeHybridStorage &          out,
    std::string *               err = nullptr,
    int                         cache_slots = 0,
    bool                        load_cold_tensors = true,
    const std::vector<std::vector<int32_t>> * cold_expert_order_by_layer = nullptr);

bool build_deepseek4_moe_hybrid_storage_from_file(
    const std::string &         path,
    ggml_backend_t              backend,
    const DeepSeek4Weights &    w,
    const MoeHybridPlacement &  placement,
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
