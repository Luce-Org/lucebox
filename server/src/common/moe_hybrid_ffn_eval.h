// Common MoE hybrid FFN evaluation — hot experts on GPU, cold on CPU, concurrent.

#pragma once

#include "moe_hybrid_types.h"
#include "moe_hybrid_storage.h"

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
    int hot_selected = 0;
    int cold_selected = 0;
};

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
    ggml_gallocr_t *                p_cold_alloc = nullptr);

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
    int                             n_selected);

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
    int n_hot);

// Build/rebuild cached cold FFN graph.
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
    int n_cold);

}  // namespace dflash::common
