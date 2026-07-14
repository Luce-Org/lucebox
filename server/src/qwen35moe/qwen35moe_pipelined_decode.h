// Pipelined hybrid MoE decode: optimized layer-by-layer decode that caches
// DeltaNet pre-FFN graphs and reduces per-layer synchronization overhead.
//
// Key optimizations vs eval_qwen35moe_hybrid_ffn_gpu_resident:
// 1. Cache DeltaNet pre-FFN graphs (30/40 layers) — avoid per-layer rebuild
// 2. Skip cold path entirely for all-hot layers (no ffn_post readback)
// 3. Persistent zero buffer for cold_in (no per-layer allocation)
// 4. Reduced tensor_copy/set calls for all-hot path

#pragma once

#include "internal.h"
#include "../common/moe_hybrid_ffn_eval.h"
#include "../common/moe_hybrid_storage.h"
#include "../common/moe_routing_collector.h"
#include "../common/moe_expert_compute.h"
#include "graph_builders.h"

#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dflash::common {

struct MoeHybridRoutingStats;

// Per-layer cached pre-FFN graph for DeltaNet layers.
// For DeltaNet layers, the graph structure doesn't depend on kv_start (recurrent),
// so we build once and reuse by updating inp_embed data only.
struct CachedPrefnGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_tensor * inp_embed = nullptr;     // [n_embd, 1, 1] F32 input
    ggml_tensor * positions = nullptr;     // [4] I32 input (attn layers only)
    ggml_tensor * kv_write_rows = nullptr; // [1, n_head_kv] I64 input (attn layers only)
    int kv_win = 0;                        // FA span baked into the graph (attn layers; 256-aligned)
    ggml_tensor * ffn_post = nullptr;      // output: post-norm hidden state
    ggml_tensor * ffn_residual = nullptr;  // output: pre-FFN residual
    ggml_tensor * moe_selected = nullptr;  // output: selected expert IDs
    ggml_tensor * moe_weights = nullptr;   // output: routing weights

    CachedPrefnGraph() = default;
    ~CachedPrefnGraph() { free(); }
    CachedPrefnGraph(const CachedPrefnGraph &) = delete;
    CachedPrefnGraph & operator=(const CachedPrefnGraph &) = delete;
    CachedPrefnGraph(CachedPrefnGraph && o) noexcept { *this = std::move(o); }
    CachedPrefnGraph & operator=(CachedPrefnGraph && o) noexcept {
        if (this != &o) {
            free();
            ctx = o.ctx; gf = o.gf; alloc = o.alloc;
            inp_embed = o.inp_embed; ffn_post = o.ffn_post;
            ffn_residual = o.ffn_residual;
            moe_selected = o.moe_selected; moe_weights = o.moe_weights;
            positions = o.positions; kv_write_rows = o.kv_write_rows;
            kv_win = o.kv_win;
            o.ctx = nullptr; o.gf = nullptr; o.alloc = nullptr;
            o.inp_embed = nullptr; o.ffn_post = nullptr;
            o.ffn_residual = nullptr;
            o.moe_selected = nullptr; o.moe_weights = nullptr;
            o.positions = nullptr; o.kv_write_rows = nullptr;
            o.kv_win = 0;
        }
        return *this;
    }
    bool valid() const { return ctx && gf && alloc && ffn_post && ffn_residual; }
    void free();
};

struct PipelinedDecodeTelemetry {
    uint64_t total_us = 0;
    uint64_t prefn_graph_build_us = 0;
    uint64_t prefn_compute_us = 0;
    uint64_t routing_readback_us = 0;
    uint64_t ffn_us = 0;
    uint64_t ffn_allhot_us = 0;
    uint64_t ffn_mixed_us = 0;
    // GPU utilization diagnosis: time the GPU is idle waiting for CPU
    uint64_t gpu_idle_us = 0;       // total GPU idle (tensor_io + combine_overhead + sync_wait)
    uint64_t tensor_io_us = 0;      // hot path setup: D2H readback + GPU copies + kernel launch
    uint64_t combine_overhead_us = 0; // combine graph dispatch + copy
    uint64_t cold_cpu_us = 0;       // non-local expert path total (CPU or remote expert compute)
    uint64_t expert_compute_us = 0; // non-local expert compute call time, excluding combine
    uint64_t hot_graph_build_us = 0; // hot graph rebuild (only when n_hot changes)
    uint64_t ffn_post_get_us = 0;   // D2H readback of ffn_post for cold path
    uint64_t sync_wait_us = 0;      // time in ggml_backend_synchronize (waiting for GPU)
    int allhot_layers = 0;
    int mixed_layers = 0;
    int total_layers = 0;
    int hot_graph_rebuilds = 0;     // count of hot graph rebuilds
    int routed_ffn_layers = 0;      // layers handled by routed FFN (async pipeline)

    // ── Routed path breakdown (StreamMoE fast path) ──
    uint64_t routed_prefn_us = 0;       // prefn graph compute (async dispatch + sync)
    uint64_t routed_sync_us = 0;        // GPU sync stall waiting for prefn
    uint64_t routed_readback_us = 0;    // D2H readback of routing IDs + weights
    uint64_t routed_cpu_remap_us = 0;   // CPU-side local ID mapping + cold masking
    uint64_t routed_ffn_dispatch_us = 0;// FFN graph dispatch + combine (async)
    uint64_t routed_final_sync_us = 0;  // final sync at end of token (if measured)
    int routed_cold_expert_hits = 0;    // experts masked (weight=0) in routed path
    int routed_total_expert_slots = 0;  // total expert slots processed
};

// State for pipelined decode: holds cached DeltaNet pre-FFN graphs +
// the GpuResidentState for FFN + persistent buffers.
struct PipelinedDecodeState {
    GpuResidentState gpu_state;

    // Cached pre-FFN graphs for DeltaNet layers (layer index → graph)
    // Attention layers (every full_attention_interval-th) are nullptr (rebuilt each token)
    std::vector<CachedPrefnGraph> cached_prefn;

    // Cached routed FFN graphs for DeltaNet layers (layer index → graph)
    // StreamMoE-inspired: reads routing from GPU, eliminates CPU sync.
    std::vector<CachedFfnGraph> cached_routed_ffn;

    // Persistent host buffers (avoid per-layer allocation)
    std::vector<int32_t> routing_ids_buf;
    std::vector<float> routing_weights_buf;
    std::vector<int32_t> ffn_local_ids_buf;
    std::vector<float> ffn_masked_weights_buf;
    std::vector<int32_t> ffn_hot_ids_buf;
    std::vector<float> ffn_hot_weights_buf;
    std::vector<int32_t> ffn_cold_ids_buf;
    std::vector<float> ffn_cold_weights_buf;
    std::vector<float> combine_zero_buf;
    std::vector<float> ffn_post_host_buf;

    // Persistent zero buffer for cold_in (set once at init)
    bool cold_in_zeroed = false;

    // When true (default), non-local experts are computed through the expert
    // compute path instead of being dropped via cold-masking. Exact but slower.
    // Set DFLASH_DROP_COLD=1 to disable (fast but lossy).
    bool expert_compute = true;

    // Shared MoE expert compute runtime (CPU or backend IPC).
    MoeExpertComputeRuntime expert_runtime;
    std::vector<float> expert_output_buf;          // [n_embd] scratch for expert compute output
    std::vector<float> capture_host_buf;          // [n_embd] scratch for DFlash capture

    // Tracking
    int n_layer = 0;
    int n_embd = 0;
    int n_expert_used = 0;
    int full_attention_interval = 0;

    // Optional routing data collector (not owned, set from backend)
    MoeRoutingCollector * routing_collector = nullptr;

    PipelinedDecodeState() = default;
    ~PipelinedDecodeState() { destroy(); }
    PipelinedDecodeState(const PipelinedDecodeState &) = delete;
    PipelinedDecodeState & operator=(const PipelinedDecodeState &) = delete;
    PipelinedDecodeState(PipelinedDecodeState && o) noexcept
        : gpu_state(std::move(o.gpu_state)),
          cached_prefn(std::move(o.cached_prefn)),
          cached_routed_ffn(std::move(o.cached_routed_ffn)),
          routing_ids_buf(std::move(o.routing_ids_buf)),
          routing_weights_buf(std::move(o.routing_weights_buf)),
          ffn_local_ids_buf(std::move(o.ffn_local_ids_buf)),
          ffn_masked_weights_buf(std::move(o.ffn_masked_weights_buf)),
          ffn_hot_ids_buf(std::move(o.ffn_hot_ids_buf)),
          ffn_hot_weights_buf(std::move(o.ffn_hot_weights_buf)),
          ffn_cold_ids_buf(std::move(o.ffn_cold_ids_buf)),
          ffn_cold_weights_buf(std::move(o.ffn_cold_weights_buf)),
          combine_zero_buf(std::move(o.combine_zero_buf)),
          ffn_post_host_buf(std::move(o.ffn_post_host_buf)),
          cold_in_zeroed(o.cold_in_zeroed),
          expert_compute(o.expert_compute),
          expert_runtime(std::move(o.expert_runtime)),
          expert_output_buf(std::move(o.expert_output_buf)),
          capture_host_buf(std::move(o.capture_host_buf)),
          n_layer(o.n_layer), n_embd(o.n_embd),
          n_expert_used(o.n_expert_used),
          full_attention_interval(o.full_attention_interval) {
        o.n_layer = 0;
    }
    PipelinedDecodeState & operator=(PipelinedDecodeState && o) noexcept {
        if (this != &o) {
            destroy();
            gpu_state = std::move(o.gpu_state);
            cached_prefn = std::move(o.cached_prefn);
            cached_routed_ffn = std::move(o.cached_routed_ffn);
            routing_ids_buf = std::move(o.routing_ids_buf);
            routing_weights_buf = std::move(o.routing_weights_buf);
            ffn_local_ids_buf = std::move(o.ffn_local_ids_buf);
            ffn_masked_weights_buf = std::move(o.ffn_masked_weights_buf);
            ffn_hot_ids_buf = std::move(o.ffn_hot_ids_buf);
            ffn_hot_weights_buf = std::move(o.ffn_hot_weights_buf);
            ffn_cold_ids_buf = std::move(o.ffn_cold_ids_buf);
            ffn_cold_weights_buf = std::move(o.ffn_cold_weights_buf);
            combine_zero_buf = std::move(o.combine_zero_buf);
            ffn_post_host_buf = std::move(o.ffn_post_host_buf);
            cold_in_zeroed = o.cold_in_zeroed;
            expert_compute = o.expert_compute;
            expert_runtime = std::move(o.expert_runtime);
            expert_output_buf = std::move(o.expert_output_buf);
            capture_host_buf = std::move(o.capture_host_buf);
            n_layer = o.n_layer; n_embd = o.n_embd;
            n_expert_used = o.n_expert_used;
            full_attention_interval = o.full_attention_interval;
            o.n_layer = 0;
        }
        return *this;
    }
    bool valid() const { return gpu_state.valid() && n_layer > 0; }
    void reset_request_graphs();
    void destroy();
};

// Initialize pipelined decode state: build cached DeltaNet pre-FFN graphs,
// allocate persistent buffers, init GPU-resident state.
bool init_pipelined_decode_state(
    PipelinedDecodeState & out,
    ggml_backend_t backend,
    const TargetWeights & w,
    const std::string & target_path,
    TargetCache & cache,
    MoeHybridStorage & hybrid,
    int kv_start,           // initial KV position for graph caching
    int kq_stride_pad);

// Initialize or refresh the MoE expert compute backend without building request
// graphs. Used to prewarm remote mixed-backend expert workers at backend
// startup while keeping per-request graph lifetime separate.
bool ensure_pipelined_moe_expert_compute(
    PipelinedDecodeState & out,
    const TargetWeights & w,
    const std::string & target_path,
    MoeHybridStorage & hybrid);

// Run one full token through the pipelined decode loop (all n_layer layers).
// On success, gpu_state.act_cur holds the final hidden state on GPU.
// selected_ids_out / weights_out: optional per-layer routing capture for telemetry.
// kv_slot: physical KV row to write (kvflash pool slot); -1 = kv_pos (identity,
// no pool). The FA span clamps to the cache tensor's physical capacity, so
// pool-sized tensors bound the cached-graph window automatically.
bool pipelined_decode_one_token(
    PipelinedDecodeState & state,
    ggml_backend_t backend,
    const TargetWeights & w,
    TargetCache & cache,
    MoeHybridStorage & hybrid,
    int kv_pos,              // current KV position (logical; drives RoPE)
    int kq_stride_pad,
    PipelinedDecodeTelemetry * telemetry = nullptr,
    int kv_slot = -1,
    bool capture_layers = false,
    MoeHybridRoutingStats * routing_stats = nullptr);

}  // namespace dflash::common
