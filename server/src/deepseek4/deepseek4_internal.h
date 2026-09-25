// DeepSeek V4 Flash target structs for luce daemon.
//
// Architecture summary (from DeepSeek V4 Flash):
//   - MLA: Multi-head Latent Attention with low-rank Q projection and single
//     KV head shared across all attention heads.
//   - KV Compression: learned compressor pools SWA windows into compressed KV
//     rows (ratio-4 for even layers ≥2, ratio-128 for odd layers ≥2).
//   - Indexer: on ratio-4 layers, learned scorer selects top-k compressed rows.
//   - HC: Hierarchical Controller with 4 parallel residual streams, mixed via
//     Sinkhorn-normalized combine matrices at each sublayer.
//   - MoE: routed experts (top-6, 256 in V4 / 384 in V4.1) + 1 shared expert
//     per layer. V4's first 3 layers use hash-based routing (token_id → expert_ids).
//   - RoPE: partial rotation (64 of 512 dims), YaRN scaling.
//
// DeepSeek V4.1 Flash ("deepseek41") shares every struct here. Its deltas:
// compress ratios 2 (layers 2-19) and 1 (20-39) with the compressed rows
// owned by a few kv source layers and read by the layers after them, index
// keys derived from the attention latent at those sources, no per-head query
// norm, a staggered hyper-connection pre-mix, candidate block pre-selection,
// and the Engram n-gram memory on two layers. See docs/DS41.md for what the
// runtime implements today.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "internal.h"
#include "common/layer_split_utils.h"
#include "common/paged_attention_config.h"
#include "common/prefill_attention_mode.h"
#include "deepseek4_image_spans.h"
#include "common/concurrency/paged_kv_pool.h"
#include "deepseek4_paged_cache.h"

namespace luce::common {

// Layer-major prefill may schedule five 2K numerical bands while preserving
// the raw-cache rounding boundary between them.
inline constexpr int DS4_NUMERICAL_PREFILL_BAND = 2048;
inline constexpr int DS4_MAX_LAYER_MAJOR_PREFILL_TOKENS = 10240;
// Normal verification stays within one ratio-4 compressor window. Q5 is an
// explicit opt-in whose fused graph models a second boundary.
inline constexpr int DS4_CONSERVATIVE_VERIFY_MAX_TOKENS = 4;
inline constexpr int DS4_Q5_VERIFY_TOKENS = 5;

struct MoeHybridPlacement;
struct MoeHybridConfig;
struct MoeHybridRoutingStats;
struct MoeExpertComputeRuntime;
class MoeHybridStreamEngine;
class MoeStreamedExpertCache;
struct MoeExpertCacheOptions;

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

    // ── Indexer (V4: ratio-4 layers; V4.1: index source layers) ──────
    // Selects which compressed rows to attend via top-k scoring.
    ggml_tensor * indexer_attn_q_b     = nullptr;  // [n_lora_q, n_indexer_head * indexer_head_dim]
    ggml_tensor * indexer_proj         = nullptr;  // [n_embd, n_indexer_head] head weight projection

    // V4: the indexer has its own compressor for the indexer key cache.
    ggml_tensor * indexer_compressor_ape  = nullptr;
    ggml_tensor * indexer_compressor_kv   = nullptr;
    ggml_tensor * indexer_compressor_gate = nullptr;
    ggml_tensor * indexer_compressor_norm = nullptr;

    // V4.1: index keys are rope_tail(rms_norm(indexer_k · latent)) of the
    // compressed latent, written at kv source layers that are index sources.
    ggml_tensor * indexer_k            = nullptr;  // [head_dim, indexer_head_dim]
    ggml_tensor * indexer_k_norm       = nullptr;  // [indexer_head_dim]

    // ── Engram (V4.1 engram layers only) ─────────────────────────────
    // The hash table itself (blk.N.engram_embd) is never loaded; its rows are
    // read on demand (deepseek4_engram.h). TODO(deepseek41): apply on the
    // layer input of every HC copy.
    ggml_tensor * engram_q             = nullptr;  // [n_embd, n_hc]
    ggml_tensor * engram_k             = nullptr;  // [n_embd, n_hc]
    ggml_tensor * engram_wkv           = nullptr;  // [n_hash_cols * key_len, n_embd * (n_hc + 1)]

    // ── HC Attention ─────────────────────────────────────────────────
    ggml_tensor * hc_attn_fn         = nullptr;  // [n_hc * n_embd, hc_mix_dim] F16
    ggml_tensor * hc_attn_scale      = nullptr;  // [3] F32 (pre_scale, post_scale, comb_scale)
    ggml_tensor * hc_attn_base       = nullptr;  // [n_hc] F32

    // ── FFN / MoE ────────────────────────────────────────────────────
    ggml_tensor * ffn_norm           = nullptr;  // [n_embd]

    // Router
    ggml_tensor * ffn_gate_inp       = nullptr;  // [n_embd, n_expert] router weights F16
    ggml_tensor * ffn_exp_probs_b    = nullptr;  // [n_expert] optional routing bias
    // With protected experts and a router bias (see protected_experts): the
    // selection bias without the router bias (F32 [n_expert]) and the
    // protected mask (I32 [n_expert]), for graphs that route on the device.
    ggml_tensor * native_selection_bias = nullptr;
    ggml_tensor * protected_mask        = nullptr;
    ggml_tensor * ffn_gate_bias_vl   = nullptr;  // image router bias, loaded only with --mmproj

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
    ggml_mixed_mmq_policy mixed_mmq_policy = GGML_MIXED_MMQ_DEFAULT;
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    // Optional row-split buffer for selected dense projections. The buffer
    // owns per-device allocations while the tensor metadata stays in ctx.
    ggml_backend_buffer_t dense_split_buf = nullptr;
    // Holds each layer's native_selection_bias and protected_mask.
    ggml_context *        routing_ctx = nullptr;
    ggml_backend_buffer_t routing_buf = nullptr;

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
    // Model family, general.architecture: "deepseek4" (V4 Flash) or
    // "deepseek41" (V4.1 Flash). Metadata keys are read as `arch + "."`.
    std::string arch      = "deepseek4";
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

    // Per-layer compression ratios (0 = no compression; V4: 4 or 128;
    // V4.1: 2 or 1).
    std::vector<uint32_t> compress_ratios;

    // Compressed-cache ownership. Every compressing layer keeps its ratio, but
    // only kv source layers run a compressor and write compressed rows; the
    // layers after a source read that source's rows. Index source layers score
    // the shared index keys and hand their top-k to the layers after them.
    // V4 declares no sources: every compressing layer is its own kv source and
    // every ratio-4 layer its own index source, so every kv_src[il] == il.
    std::vector<int>     kv_source_layer_ids;     // declared or inferred, ascending
    std::vector<int>     index_source_layer_ids;
    std::vector<int>     kv_src;                  // per layer: owner of its compressed rows
    std::vector<int>     idx_src;                 // per layer: owner of its top-k
    std::vector<uint8_t> kv_source_flags;         // per layer: is a kv source
    std::vector<uint8_t> index_source_flags;      // per layer: is an index source
    bool shared_comp_cache = false;               // some layer reads another layer's rows

    // Forward-pass rules that differ between the families (set by the loader).
    bool attn_q_head_norm = true;    // unit-RMS per query head after wq_b (V4 only)
    bool hc_staggered_pre = false;   // V4.1 pre-mix: see ds4_hc_collapse in the graph

    // Candidate block pre-selection (V4.1): the index sources after
    // candidate_source_layer restrict their top-k to the candidate blocks
    // that layer selected. -1 = off. TODO(deepseek41): an exact no-op until
    // more than candidate_topk_blocks * candidate_block_size compressed rows.
    int candidate_source_layer = -1;
    int candidate_topk_blocks  = 0;
    int candidate_block_size   = 0;

    // Engram n-gram hash memory (V4.1): hash constants as written by the
    // converter (deepseek41.engram.*) and where each layer's table lives.
    struct Engram {
        std::vector<int>      layer_ids;
        int                   n_heads   = 0;
        int                   key_len   = 0;
        int                   max_ngram = 0;
        std::vector<uint64_t> multipliers;  // [n_engram_layers * max_ngram]
        std::vector<uint64_t> primes;       // [n_engram_layers * (max_ngram - 1) * n_heads]
        std::vector<uint64_t> offsets;      // same shape as primes
        std::vector<int32_t>  token_map;    // [n_vocab] compressed token ids
        int32_t               pad_id = -1;  // already compressed
        std::vector<uint64_t> rows;         // [n_engram_layers] table rows = sum of that layer's primes
        // Where each layer's table lives when the GGUF embeds it
        // (blk.N.engram_embd, I8 [row_bytes, rows]: 256 E4M3 + 8 E8M0 block
        // scales per row). Never mapped; rows are pread on demand.
        struct Table {
            int      layer_id    = -1;
            uint64_t file_offset = 0;   // absolute byte offset in the GGUF
            uint64_t rows        = 0;
            uint32_t row_bytes   = 0;
            int      ggml_type   = -1;  // ggml_type of the tensor as stored
        };
        std::vector<Table>    tables;
        bool present() const { return !layer_ids.empty(); }
    } engram;

    // RoPE
    float rope_freq_base        = 10000.0f;
    float rope_scale_factor     = 16.0f;
    float rope_yarn_beta_fast   = 32.0f;
    float rope_yarn_beta_slow   = 1.0f;
    float compress_rope_freq_base = 160000.0f;
    uint64_t rope_orig_ctx      = 65536;

    // Norms
    float rms_eps         = 1.0e-6f;
    float hc_eps          = 1.0e-6f;   // RMS eps of the HC mixes (V4.1: rms_eps)

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
    // Routing adjustments loaded with the model (--ds4-router-bias,
    // --ds4-protected-experts), [n_layer * n_expert] each, empty when unused.
    // The delta is already added to every ffn_exp_probs_b; host routing
    // subtracts it again to find a token's native top-k.
    std::vector<float>   router_bias_delta;
    std::vector<uint8_t> protected_experts;
    // Host copy of every layer's selection bias (exp_probs_b with the router
    // bias applied), [n_layer * n_expert], taken once after the adjustments.
    // Empty when a layer has no F32 bias; host routing then reads the device.
    std::vector<float>   selection_bias_host;
    bool fused_decode        = false;
    bool fused_verify_f16_kv = false;
};

// True when the image router biases were loaded, i.e. the backend was started
// with a vision projector.
inline bool ds4_image_capable(const DeepSeek4Weights & w) {
    return !w.layers.empty() && w.layers.front().ffn_gate_bias_vl != nullptr;
}

inline bool deepseek4_is_eos_tok(int tok, const DeepSeek4Weights & w) {
    return (w.eos_chat_id >= 0 && tok == w.eos_chat_id)
        || (w.eos_id >= 0 && tok == w.eos_id);
}

// Source-layer indirection. The loader always fills the arrays; weights built
// by hand (tests) fall back to the V4 rule: every compressing layer owns its
// rows and the ratio-4 layers carry the indexer.
inline bool deepseek4_is_kv_source(const DeepSeek4Weights & w, int il) {
    if (il < 0 || (size_t) il >= w.compress_ratios.size()) return false;
    if (w.kv_source_flags.empty()) return w.compress_ratios[(size_t) il] > 0;
    return w.kv_source_flags[(size_t) il] != 0;
}
inline bool deepseek4_is_index_source(const DeepSeek4Weights & w, int il) {
    if (il < 0 || (size_t) il >= w.compress_ratios.size()) return false;
    if (w.index_source_flags.empty()) return w.compress_ratios[(size_t) il] == 4;
    return w.index_source_flags[(size_t) il] != 0;
}
// Layer whose compressed rows (and index keys) `il` attends over.
inline int deepseek4_kv_source_layer(const DeepSeek4Weights & w, int il) {
    return (il >= 0 && (size_t) il < w.kv_src.size()) ? w.kv_src[(size_t) il] : il;
}
// Layer whose top-k `il` reuses. TODO(deepseek41): carry the top-k.
inline int deepseek4_index_source_layer(const DeepSeek4Weights & w, int il) {
    return (il >= 0 && (size_t) il < w.idx_src.size()) ? w.idx_src[(size_t) il] : il;
}

// ─── KV Cache ───────────────────────────────────────────────────────────

// Per-layer compressor rolling state
struct DeepSeek4CompressorState {
    ggml_tensor * state_kv    = nullptr;  // [window_size, head_dim] rolling buffer
    ggml_tensor * state_score = nullptr;  // [window_size, head_dim] rolling scores
};

// Device-resident snapshot of the ratio-4 previous-window rows immediately
// after the first flush in a q5 verification step. A q5 batch that begins at
// position 3 mod 4 can flush twice; retaining this intermediate state lets a
// rejected prefix commit the first window without replaying the target.
struct DeepSeek4SpecBoundaryCheckpointLayer {
    ggml_tensor * attn_kv_src = nullptr;
    ggml_tensor * attn_kv_dst = nullptr;
    ggml_tensor * attn_score_src = nullptr;
    ggml_tensor * attn_score_dst = nullptr;
    ggml_tensor * index_kv_src = nullptr;
    ggml_tensor * index_kv_dst = nullptr;
    ggml_tensor * index_score_src = nullptr;
    ggml_tensor * index_score_dst = nullptr;
};

struct DeepSeek4SpecBoundaryCheckpoint {
    std::vector<DeepSeek4SpecBoundaryCheckpointLayer> layers;
    bool available = false;

    void clear() {
        layers.clear();
        available = false;
    }
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

// Per-shard runtime state for deepseek4_step_layer_range (host-side HC weight
// cache + cached decode graphs). Defined in deepseek4_graph.cpp; owned by the
// DeepSeek4Cache below and released by free_deepseek4_cache().
struct DeepSeek4LayerRangeCache;

struct DeepSeek4Cache {
    int cur_pos  = 0;
    int max_ctx  = 0;
    int n_layer  = 0;

    std::vector<DeepSeek4LayerCache> layers;
    PrefillAttentionMode prefill_mode = PrefillAttentionMode::Exact;

    // HC residual streams: [n_hc * n_embd] persistent state
    ggml_tensor * hc_state    = nullptr;  // [n_hc * n_embd]

    // Lazily created on the first deepseek4_step_layer_range call.
    DeepSeek4LayerRangeCache * layer_range_cache = nullptr;

    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

struct DeepSeek4PagedLayerCache : DeepSeek4LayerCache {
    uint32_t ratio = 0;
    uint64_t physical_rows = 0;
};

struct DeepSeek4PagedCache {
    std::unique_ptr<PagedKvPool> pool;
    DeepSeek4PagedCachePlan plan;
    std::vector<DeepSeek4PagedLayerCache> layers;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    // Dedicated bounded gathered-reference graph cache (opaque here because
    // its implementation shares the fused verifier's private machinery).
    void * gathered_runtime = nullptr;
};

struct DeepSeek4Snapshot;

struct DeepSeek4RawRingSpan {
    int row = 0;
    int count = 0;
};

struct DeepSeek4Head4Tail2Routes {
    ggml_tensor * head_ids = nullptr;
    ggml_tensor * head_weights = nullptr;
    ggml_tensor * tail_ids = nullptr;
    ggml_tensor * tail_weights = nullptr;
};

// ─── Configuration ──────────────────────────────────────────────────────

struct DeepSeek4BackendConfig {
    std::string  model_path;
    std::string  mmproj_path;
    DevicePlacement device;
    int          stream_fd    = -1;
    int          chunk        = 512;   // prefill chunk size
    PrefillAttentionMode prefill_mode = PrefillAttentionMode::Exact;
    int          max_ctx      = 0;     // 0 = auto from SWA + compression capacity
    int          expert_top_k = 0;     // 0 = use all model-routed experts
    bool         fused_decode = false; // single-graph GPU decode
    bool         fused_verify_f16_kv = false; // F16 KV in batched verifier attention
    bool         paged_attention = false;
    int          max_concurrency = 1;
    long long    kv_pool_tokens = 0;
    std::string  expert_placement_path;   // three-tier expert ownership (JSON)
    std::string  router_bias_path;        // f32 [n_layer][n_expert] selection bias delta
    std::string  protected_experts_path;  // {"layer": [expert ids]} (JSON)
};

// ─── Function declarations ──────────────────────────────────────────────

// Select compressed rows plus the saved raw suffix of a batched verifier.
// Indices are relative to the end of the physical raw ring. Causal visibility
// remains in the attention mask; saving a row does not make it visible to all lanes.
ggml_tensor * deepseek4_indexed_attention_rows(
    ggml_context * ctx, ggml_tensor * compressed_topk,
    int compressed_rows, int preserved_rows);

// Snapshot the raw rows a cached verifier is about to overwrite. Expand this
// tensor before the ring writes; row indices are supplied again on each replay.
ggml_tensor * deepseek4_preserve_raw_rows(
    ggml_context * ctx, ggml_tensor * raw_kv, ggml_tensor * rows);

// Keep a per-token indexer visibility mask aligned with the scored suffix.
ggml_tensor * deepseek4_indexer_visibility_suffix(
    ggml_context * ctx, ggml_tensor * mask, int first_scored, int n_scored);

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

// Per-layer cache geometry implied by the weights (struct and rules in
// deepseek4_paged_cache.h). Single source of truth for create_deepseek4_cache(),
// the paged planner, the cache byte estimate and snapshot declaration and
// validation (deepseek4_snapshot.h), so none of them can disagree on shapes.
DeepSeek4LayerGeometry deepseek4_layer_geometry(const DeepSeek4Weights & w, int layer);
std::vector<DeepSeek4LayerGeometry> deepseek4_layer_geometries(const DeepSeek4Weights & w);
// The compressed rows `il` attends over: its own cache at a kv source (every
// V4 layer), the source's cache at a V4.1 reader. Readers never write there.
template <typename Cache>
inline auto & ds4_comp_cache(Cache & cache, const DeepSeek4Weights & w, int il) {
    return cache.layers[(size_t) deepseek4_kv_source_layer(w, il)];
}
inline int64_t deepseek4_hc_state_elements(const DeepSeek4Weights & w) {
    return (int64_t) w.n_hc * (int64_t) w.n_embd;
}

void free_deepseek4_cache(DeepSeek4Cache & c);
bool create_deepseek4_paged_cache(ggml_backend_t backend,
                                  const DeepSeek4Weights & w,
                                  uint32_t slots, uint32_t max_ctx,
                                  uint32_t physical_blocks,
                                  DeepSeek4PagedCache & out);
void reset_deepseek4_paged_slot(DeepSeek4PagedCache & c, uint32_t slot);
void free_deepseek4_paged_cache(DeepSeek4PagedCache & c);
// Exact gathered-reference decode for up to six independent lanes. Inputs are
// lane-major; negative slots are inactive padding lanes. `logit_lanes` marks
// the lanes that need full host logits. `out_logits` is empty when none do;
// otherwise it is [n_vocab, lanes], with unrequested rows zeroed.
bool deepseek4_paged_gathered_step(
    ggml_backend_t backend, int device, const DeepSeek4Weights & w,
    DeepSeek4PagedCache & cache, const float * embeddings,
    const int32_t * token_ids, const int64_t * positions,
    const int32_t * slots, uint32_t lanes, const int32_t * block_tables,
    uint32_t block_table_stride, bool bucket_history,
    const uint8_t * logit_lanes,
    std::vector<float> & out_logits, std::vector<int32_t> & out_argmax,
    MoeHybridStorage * moe_hybrid = nullptr,
    MoeHybridRoutingStats * routing_stats = nullptr,
    DeepSeek4StepTelemetry * telemetry = nullptr);
void log_deepseek4_step_telemetry(
    const char * phase, int tokens, int steps, double wall_s,
    const DeepSeek4StepTelemetry & telemetry);
void deepseek4_release_paged_gathered_runtime(DeepSeek4PagedCache & cache);
void reset_deepseek4_cache(DeepSeek4Cache & c);
// Release only reproducible large-batch graph arenas after prefill. KV/model
// state and the DSpark feature tail remain live for the following decode.
void deepseek4_release_prefill_scratch(DeepSeek4Cache & c,
                                       MoeHybridStorage * moe_hybrid);
// Retire all disposable decoder/owner graphs before the vision tower uses the
// shared scratch allowance. KV and saved snapshots are left intact.
void deepseek4_release_image_scratch(DeepSeek4Cache & c,
                                     MoeHybridStorage * moe_hybrid);
// Invalid/future raw-ring rows after all writes of a batched verifier.
// Each span is bounded by n_swa, including batches that overwrite the full ring.
int deepseek4_verify_raw_mask_spans(
    int kv_start, int n_swa, int q, int lane, DeepSeek4RawRingSpan spans[2]);
int deepseek4_previous_raw_ring_spans(
    int kv_start,
    int n_swa,
    DeepSeek4RawRingSpan spans[2]);
bool build_deepseek4_head4_tail2_routes(
    ggml_context * ctx,
    ggml_tensor * selected,
    ggml_tensor * router_weights,
    int n_tokens,
    DeepSeek4Head4Tail2Routes & out);

// Largest prefix of [kv_start, kv_start + n_tokens) that reaches at most the
// next learned-compressor boundary. Multi-token dynamic forwards split on
// this prefix so state writes after a boundary cannot race ahead of pooling.
int deepseek4_safe_compressor_batch_tokens(const DeepSeek4Weights & w,
                                           int kv_start,
                                           int n_tokens);

// Sets up the device cache for the experts the hybrid storage streams from
// the model file (see common/moe_hybrid_expert_cache.h).
bool init_deepseek4_streamed_expert_cache(
    const DeepSeek4Weights &      w,
    const MoeHybridStorage &      hybrid,
    const MoeExpertCacheOptions & opts,
    MoeStreamedExpertCache &      cache,
    std::string *                 err);

// Forward: single step (prefill chunk or decode token).
// embed: [n_embd, n_tokens] input embeddings (post-embedding lookup).
// hc_state: [n_hc * n_embd] persistent HC residual (updated in-place).
// Returns logits for last token.
struct Ds4VerifyHooks;

bool deepseek4_step(
    ggml_backend_t              backend,
    int                         device,
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
    Ds4VerifyHooks *            verify_hooks = nullptr,
    MoeExpertComputeRuntime *   expert_runtime = nullptr,
    bool                        need_logits = true);

// Optional hooks for the DSpark spec-decode batched verify (deepseek4_dspark).
// When set on a multi-token deepseek4_step_layer_range call they add: per-layer
// mean-over-HC feature capture and full per-position logits. Null on the normal
// (23 tok/s) decode path so it is completely unaffected.
// Rows a verify batch wrote into small pooled compressor windows (V4.1 ratio
// 2: a window's state rows are its tokens' projections, pooled when it
// completes). A later token of the batch overwrites the row of an earlier one
// at the same window slot, so a rejection that ends mid-window puts the
// accepted tokens' rows back from here (see the DSpark rollback).
struct Ds4VerifyWindowRows {
    std::vector<std::vector<uint8_t>> kv;     // [layer] -> [batch token][row bytes]
    std::vector<std::vector<uint8_t>> score;
};

struct Ds4VerifyHooks {
    const std::vector<int> * capture_layer_ids = nullptr;  // e.g. {40,41,42}
    std::vector<float> *     capture_out = nullptr;         // [n_cap*n_embd * n_tokens]
    // Optional relative token range for layer-major feature readback. Generic
    // verifier paths may ignore this and return the complete batch.
    int                      capture_token_begin = 0;
    int                      capture_token_end = -1;        // exclusive; -1 = n_tokens
    std::vector<float> *     all_logits_out = nullptr;      // [n_vocab * n_tokens]
    std::vector<int32_t> *   argmax_out = nullptr;          // [n_tokens], optional GPU result
    bool                     prefer_argmax_only = false;     // skip logits D2H when available
    DeepSeek4SpecBoundaryCheckpoint * boundary_checkpoint_out = nullptr;
    Ds4VerifyWindowRows *    window_rows = nullptr;         // V4.1 tokenwise verify
};

// True for a compressor state that is one pooled window of `ratio` rows
// (V4.1 ratio 2), as opposed to V4's overlapping ratio-4 state or its
// ratio-128 ring.
inline bool deepseek4_is_window_state(const DeepSeek4CompressorState & st, int ratio) {
    return ratio > 1 && ratio < 4 && st.state_kv && st.state_score &&
           st.state_kv->ne[1] == ratio && st.state_score->ne[1] == ratio;
}

bool deepseek4_step_layer_range(
    ggml_backend_t              backend,
    int                         device,
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
    Ds4VerifyHooks *            verify_hooks = nullptr,
    MoeHybridStorage *          moe_hybrid = nullptr,
    MoeExpertComputeRuntime *   expert_runtime = nullptr,
    MoeHybridRoutingStats *     routing_stats = nullptr,
    vision::ImageSpanView       image_spans = {});

bool deepseek4_validate_image_batch(
    const DeepSeek4Weights & w, const DeepSeek4Cache & cache,
    const MoeHybridStorage * hybrid, const int32_t * tokens,
    int count, int position, vision::ImageSpanView spans,
    bool & has_images, std::string & error);

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
    std::string *               err = nullptr,
    ggml_backend_t              cold_gpu_backend = nullptr);

// Attach each compact GPU owner tensor to the learned decode-table rows for
// the global experts stored in that tensor.
bool register_deepseek4_moe_hybrid_mix_tables(
    const std::string &         path,
    const DeepSeek4Weights &    w,
    MoeHybridStorage &          storage,
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
    // Optional serialization sidecars (ondisk prefix cache). Present when the
    // snapshot was saved with DeepSeek4SnapshotAux or adopted from disk.
    //   meta_snap        I32 [kDeepSeek4SnapMetaBase + 2 * n_layer]
    //   last_logits_snap F32 [n_vocab]
    //   spec_feat_snap   F32 [1, max(1, n_spec_feat)]  (logical length in meta)
    ggml_tensor * meta_snap        = nullptr;
    ggml_tensor * last_logits_snap = nullptr;
    ggml_tensor * spec_feat_snap   = nullptr;
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    // false when ctx/buf are shared with (and freed by) another owner, e.g.
    // one merged ondisk context bound into several layer-split shard snapshots.
    bool                  owns_storage = true;
};


}  // namespace luce::common
