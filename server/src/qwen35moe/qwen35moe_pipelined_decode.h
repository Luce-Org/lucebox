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
#include "qwen35moe_hybrid_ffn_eval.h"
#include "qwen35moe_hybrid_storage.h"
#include "graph_builders.h"

#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

// Per-layer cached pre-FFN graph for DeltaNet layers.
// For DeltaNet layers, the graph structure doesn't depend on kv_start (recurrent),
// so we build once and reuse by updating inp_embed data only.
struct CachedPrefnGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_tensor * inp_embed = nullptr;     // [n_embd, 1, 1] F32 input
    ggml_tensor * ffn_post = nullptr;      // output: post-norm hidden state
    ggml_tensor * ffn_residual = nullptr;  // output: pre-FFN residual
    ggml_tensor * moe_selected = nullptr;  // output: selected expert IDs
    ggml_tensor * moe_weights = nullptr;   // output: routing weights

    ~CachedPrefnGraph() { free(); }
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
    int allhot_layers = 0;
    int mixed_layers = 0;
    int total_layers = 0;
};

// State for pipelined decode: holds cached DeltaNet pre-FFN graphs +
// the GpuResidentState for FFN + persistent buffers.
struct PipelinedDecodeState {
    GpuResidentState gpu_state;

    // Cached pre-FFN graphs for DeltaNet layers (layer index → graph)
    // Attention layers (every full_attention_interval-th) are nullptr (rebuilt each token)
    std::vector<CachedPrefnGraph> cached_prefn;

    // Persistent host buffers (avoid per-layer allocation)
    std::vector<int32_t> routing_ids_buf;
    std::vector<float> routing_weights_buf;
    std::vector<float> ffn_post_host_buf;

    // Persistent zero buffer for cold_in (set once at init)
    bool cold_in_zeroed = false;

    // Tracking
    int n_layer = 0;
    int n_embd = 0;
    int n_expert_used = 0;
    int full_attention_interval = 0;

    ~PipelinedDecodeState() { destroy(); }
    bool valid() const { return gpu_state.valid() && n_layer > 0; }
    void destroy();
};

// Initialize pipelined decode state: build cached DeltaNet pre-FFN graphs,
// allocate persistent buffers, init GPU-resident state.
bool init_pipelined_decode_state(
    PipelinedDecodeState & out,
    ggml_backend_t backend,
    const TargetWeights & w,
    TargetCache & cache,
    int kv_start,           // initial KV position for graph caching
    int kq_stride_pad);

// Run one full token through the pipelined decode loop (all n_layer layers).
// On success, gpu_state.act_cur holds the final hidden state on GPU.
// selected_ids_out / weights_out: optional per-layer routing capture for telemetry.
bool pipelined_decode_one_token(
    PipelinedDecodeState & state,
    ggml_backend_t backend,
    const TargetWeights & w,
    TargetCache & cache,
    Qwen35MoeHybridStorage & hybrid,
    int kv_pos,              // current KV position
    int kq_stride_pad,
    PipelinedDecodeTelemetry * telemetry = nullptr);

}  // namespace dflash::common
