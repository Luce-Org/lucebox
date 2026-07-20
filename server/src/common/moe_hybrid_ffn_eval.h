// Common MoE hybrid FFN evaluation — hot experts on GPU, cold on CPU, concurrent.

#pragma once

#include "moe_hybrid_types.h"
#include "moe_hybrid_storage.h"
#include "moe_expert_compute.h"

#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

// GPU-resident residual combine graph: output = residual + hot_out + cold_correction.
struct ResidualCombineGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_tensor * residual_in = nullptr;
    ggml_tensor * hot_in = nullptr;
    ggml_tensor * cold_in = nullptr;
    ggml_tensor * output = nullptr;

    ResidualCombineGraph() = default;
    ~ResidualCombineGraph() { free(); }
    ResidualCombineGraph(const ResidualCombineGraph &) = delete;
    ResidualCombineGraph & operator=(const ResidualCombineGraph &) = delete;
    ResidualCombineGraph(ResidualCombineGraph && o) noexcept
        : ctx(o.ctx), gf(o.gf), alloc(o.alloc),
          residual_in(o.residual_in), hot_in(o.hot_in),
          cold_in(o.cold_in), output(o.output) {
        o.ctx = nullptr; o.gf = nullptr; o.alloc = nullptr;
        o.residual_in = nullptr; o.hot_in = nullptr;
        o.cold_in = nullptr; o.output = nullptr;
    }
    ResidualCombineGraph & operator=(ResidualCombineGraph && o) noexcept {
        if (this != &o) {
            free();
            ctx = o.ctx; gf = o.gf; alloc = o.alloc;
            residual_in = o.residual_in; hot_in = o.hot_in;
            cold_in = o.cold_in; output = o.output;
            o.ctx = nullptr; o.gf = nullptr; o.alloc = nullptr;
            o.residual_in = nullptr; o.hot_in = nullptr;
            o.cold_in = nullptr; o.output = nullptr;
        }
        return *this;
    }
    bool valid() const { return ctx && gf && alloc && output; }
    void free();
    void destroy();
};

bool build_residual_combine_graph(ResidualCombineGraph & out, ggml_backend_t backend, int n_embd);

// GPU-resident state for the decode loop.
struct GpuResidentState {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * act_cur = nullptr;

    ResidualCombineGraph combine;

    GpuResidentState() = default;
    ~GpuResidentState() { destroy(); }
    GpuResidentState(const GpuResidentState &) = delete;
    GpuResidentState & operator=(const GpuResidentState &) = delete;
    GpuResidentState(GpuResidentState && o) noexcept
        : ctx(o.ctx), buf(o.buf), act_cur(o.act_cur),
          combine(std::move(o.combine)) {
        o.ctx = nullptr; o.buf = nullptr; o.act_cur = nullptr;
    }
    GpuResidentState & operator=(GpuResidentState && o) noexcept {
        if (this != &o) {
            destroy();
            ctx = o.ctx; buf = o.buf; act_cur = o.act_cur;
            combine = std::move(o.combine);
            o.ctx = nullptr; o.buf = nullptr; o.act_cur = nullptr;
        }
        return *this;
    }
    bool valid() const { return ctx && buf && act_cur && combine.valid(); }
    void destroy();
};

bool init_gpu_resident_state(GpuResidentState & out, ggml_backend_t backend, int n_embd);

struct MoeHybridFfnTelemetry {
    uint64_t ffn_wall_us = 0;
    uint64_t partition_us = 0;
    uint64_t hot_us = 0;
    uint64_t cold_us = 0;
    uint64_t shared_us = 0;
    uint64_t combine_us = 0;
    uint64_t hot_graph_build_us = 0;
    uint64_t hot_input_us = 0;
    uint64_t hot_compute_us = 0;
    uint64_t hot_read_us = 0;
    uint64_t cold_graph_build_us = 0;
    uint64_t cold_input_us = 0;
    uint64_t cold_compute_us = 0;
    uint64_t cold_read_us = 0;
    uint64_t hot_graph_builds = 0;
    uint64_t hot_graph_hits = 0;
    uint64_t cold_graph_builds = 0;
    uint64_t cold_graph_hits = 0;
    int hot_selected = 0;
    int cold_selected = 0;
};

// Inputs owned by a scheduler-allocated hybrid FFN graph. The lookup tensors
// map global router IDs to each backend's compact expert stack and mask the
// slots owned by the other backend without a host-side routing round trip.
struct MoeHybridGraphInputs {
    ggml_tensor * router_weights = nullptr;
    std::vector<ggml_tensor *> router_nodes;
    // q>1 decomposes the six selected routes into a four-wide head and a
    // padded two-wide tail.  Keep those derived ID/weight tensors on the main
    // owner and schedule them before either expert branch.  Otherwise the
    // scheduler discovers the cold branch first and inserts a second
    // main->peer copy in the middle of cold execution, which synchronizes the
    // peer stream before the hot branch can be submitted.
    std::vector<ggml_tensor *> route_prefork_nodes;
    ggml_tensor * hot_local_lut = nullptr;
    ggml_tensor * hot_valid_lut = nullptr;
    ggml_tensor * cold_local_lut = nullptr;
    ggml_tensor * cold_valid_lut = nullptr;
    ggml_tensor * shard_local_lut = nullptr;
    ggml_tensor * shard_valid_lut = nullptr;
    ggml_tensor * output = nullptr;
    // Exact owner-local partials exposed for consumers that can fold the
    // hot+cold reduction into their own kernel.  peer_output is the stable
    // main-backend activation produced by the existing deferred peer copy;
    // neither tensor changes expert placement or routing semantics.
    ggml_tensor * main_output = nullptr;
    ggml_tensor * peer_output = nullptr;
    // Backend-affinity hints consumed after the multi-backend scheduler is
    // created. Keeping every intermediate of a routed branch on its weight
    // backend avoids gate/up -> activation -> down ping-pong copies.
    std::vector<ggml_tensor *> hot_remap_nodes;
    std::vector<ggml_tensor *> cold_remap_nodes;
    std::vector<ggml_tensor *> shard_remap_nodes;
    std::vector<ggml_tensor *> hot_nodes;
    std::vector<ggml_tensor *> shard_nodes;
    std::vector<ggml_tensor *> cold_nodes;
    // Main-backend nodes that first consume a completed cold branch. Hash
    // layers can append two joins because six routes are lowered as 4 + 2.
    std::vector<ggml_tensor *> join_nodes;
    // Main-backend peer-copy ops whose src[0] stays on the cold owner. The
    // scheduler attaches a dedicated producer event to each node.
    std::vector<ggml_tensor *> deferred_peer_copy_nodes;
};

// Append a device-resident hot+cold+shared MoE FFN to an existing graph.
// `global_ids` and `router_weights` are [n_expert_used, n_tokens]. Weight
// tensors in `storage` determine scheduler placement on the two GPU backends.
// When `schedule_graph` is non-null, the cold branch is expanded immediately
// and a peer-owned fence is inserted before the final main-backend join. This
// forces three scheduler splits (cold, hot/shared, join), preventing the join's
// cold-result copy from blocking hot/shared launch. Consumers may use
// main_output + peer_output to fuse the exact final add into their next op.
bool build_moe_hybrid_ffn_graph(
    ggml_context *                 ctx,
    ggml_cgraph *                  schedule_graph,
    const MoeHybridConfig &        cfg,
    const MoeLayerDesc &           desc,
    const MoeHybridLayerStorage &  storage,
    ggml_tensor *                  inp,
    ggml_tensor *                  global_ids,
    ggml_tensor *                  router_weights,
    int                            n_tokens,
    MoeHybridGraphInputs &         out,
    bool                           include_shared = true,
    bool                           allow_fused_combine = false);

int moe_hybrid_expert_compute_batch_limit();
int moe_hybrid_expert_compute_ipc_batch_limit(int n_tokens);
int moe_hybrid_prefill_hot_sub_batch_limit();

// Single-token hybrid FFN: hot on GPU, cold on CPU, combine on host.
bool eval_moe_hybrid_ffn_single(
    ggml_backend_t                  gpu_backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    MoeHybridLayerStorage &         storage,
    ggml_backend_t                  cpu_backend,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_selected,
    std::vector<float> &            out,
    MoeHybridFfnTelemetry *         telemetry = nullptr,
    std::string *                   err = nullptr);

// Batched prefill FFN: all experts on GPU (no hybrid split).
bool eval_moe_batched_prefill_ffn(
    ggml_backend_t                  gpu_backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err = nullptr);

// Batched hybrid prefill FFN: hot on GPU, cold on CPU concurrently.
bool eval_moe_hybrid_ffn_batched(
    ggml_backend_t                  gpu_backend,
    ggml_backend_t                  cpu_backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    MoeHybridLayerStorage &         storage,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err = nullptr,
    ggml_gallocr_t *                p_hot_alloc = nullptr,
    ggml_gallocr_t *                p_cold_alloc = nullptr,
    MoeExpertCompute *                expert_compute = nullptr,
    const MoeExpertLayer *            expert_layer = nullptr,
    MoeHybridFfnTelemetry *         telemetry = nullptr);

// Hot-only batched prefill: all selected experts are in VRAM.
// Skips cold graph build, CPU compute, and merge — pure GPU path.
bool eval_moe_hot_only_batched(
    ggml_backend_t                  gpu_backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    MoeHybridLayerStorage &         storage,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err = nullptr,
    ggml_gallocr_t *                p_hot_alloc = nullptr);

// GPU-resident single-token hybrid FFN: keeps data on GPU, only reads router
// IDs to CPU for hot/cold partitioning.
bool eval_moe_hybrid_ffn_gpu_resident(
    ggml_backend_t                  gpu_backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    MoeHybridLayerStorage &         storage,
    ggml_backend_t                  cpu_backend,
    ggml_tensor *                   ffn_post_gpu,
    ggml_tensor *                   ffn_residual_gpu,
    GpuResidentState &              gpu_state,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_selected,
    MoeExpertCompute *                expert_compute = nullptr,
    const MoeExpertLayer *            expert_layer = nullptr);

struct CachedHotGraphOptions {
    float swiglu_clamp = 0.0f;
    bool gpu_remap = false;
    int n_expert = 0;
};

// Build/rebuild cached hot FFN graph.
bool build_cached_hot_graph(
    CachedFfnGraph & out,
    ggml_backend_t backend,
    ggml_tensor * gate_tensor,
    ggml_tensor * up_tensor,
    ggml_tensor * down_tensor,
    ggml_tensor * gate_up_tensor,
    float gate_scale,
    float up_scale,
    float down_scale,
    float gate_up_scale,
    const MoeLayerDesc & desc,
    int n_embd,
    int n_ff_exp,
    int n_hot,
    CachedHotGraphOptions options = {});

// Build/rebuild cached MoE expert compute graph.
bool build_cached_cold_graph(
    CachedFfnGraph & out,
    ggml_backend_t cpu_backend,
    ggml_tensor * gate_tensor,
    ggml_tensor * up_tensor,
    ggml_tensor * down_tensor,
    ggml_tensor * gate_up_tensor,
    float gate_scale,
    float up_scale,
    float down_scale,
    float gate_up_scale,
    int n_embd,
    int n_ff_exp,
    int n_cold,
    float swiglu_clamp = 0.0f);

// Build cached hot-only batched graph for prefill (n_tokens=MMQ_SAFE_SUB_BATCH).
bool build_cached_hot_batched_graph(
    CachedHotBatchedGraph & out,
    ggml_backend_t gpu_backend,
    const MoeHybridLayerStorage & storage,
    const MoeLayerDesc & desc,
    const MoeHybridConfig & cfg,
    int n_tokens);

}  // namespace dflash::common
