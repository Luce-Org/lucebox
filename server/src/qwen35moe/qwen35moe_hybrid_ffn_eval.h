// Single-token hybrid qwen35moe FFN evaluation helpers.

#pragma once

#include "internal.h"
#include "qwen35moe_hybrid_storage.h"

#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

// GPU-resident residual combine graph: output = residual + hot_out + cold_correction.
// Built once at decode start, reused every layer to keep act_cur on GPU.
struct ResidualCombineGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_tensor * residual_in = nullptr;   // [n_embd] F32 input
    ggml_tensor * hot_in = nullptr;        // [n_embd] F32 input
    ggml_tensor * cold_in = nullptr;       // [n_embd] F32 input (zeros when no cold)
    ggml_tensor * output = nullptr;        // [n_embd] F32 output

    bool valid() const { return ctx && gf && alloc && output; }
    void free();
    void destroy();
};

// Build the residual combine graph on the given GPU backend.
bool build_residual_combine_graph(ResidualCombineGraph & out, ggml_backend_t backend, int n_embd);

// GPU-resident state for the decode loop: persistent act_cur + combine graph.
struct GpuResidentState {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * act_cur = nullptr;       // [n_embd] F32 persistent GPU tensor

    ResidualCombineGraph combine;

    bool valid() const { return ctx && buf && act_cur && combine.valid(); }
    void destroy();
};

// Initialize GPU-resident state for decode.
bool init_gpu_resident_state(GpuResidentState & out, ggml_backend_t backend, int n_embd);

struct Qwen35MoeHybridFfnTelemetry {
    uint64_t ffn_wall_us = 0;
    uint64_t partition_us = 0;
    uint64_t hot_us = 0;
    uint64_t cold_us = 0;
    uint64_t shared_us = 0;
    uint64_t combine_us = 0;
    int hot_selected = 0;
    int cold_selected = 0;
};

bool eval_qwen35moe_reference_ffn_single(
    ggml_backend_t         gpu_backend,
    const TargetWeights &  w,
    const TargetLayer &    L,
    const float *          cur_host,
    const int32_t *        selected_ids,
    const float *          selected_weights,
    int                    n_selected,
    std::vector<float> &   out,
    std::string *          err = nullptr);

bool eval_qwen35moe_hybrid_ffn_single(
    ggml_backend_t                      gpu_backend,
    const TargetWeights &               w,
    const TargetLayer &                 L,
    Qwen35MoeHybridLayerStorage &       storage,
    ggml_backend_t                      cpu_backend,
    const float *                       cur_host,
    const int32_t *                     selected_ids,
    const float *                       selected_weights,
    int                                 n_selected,
    std::vector<float> &                out,
    Qwen35MoeHybridFfnTelemetry *       telemetry = nullptr,
    std::string *                       err = nullptr);

// Batched prefill FFN: processes n_tokens at once using the full GPU expert tensors.
// Uses pre-computed routing (selected_ids, selected_weights) from the pre-FFN graph.
// cur_host: [n_embd × n_tokens] post-norm hidden states
// selected_ids: [n_expert_used × n_tokens] expert selections (global IDs)
// selected_weights: [n_expert_used × n_tokens] routing weights
// out: [n_embd × n_tokens] output (resized internally)
bool eval_qwen35moe_batched_prefill_ffn(
    ggml_backend_t         gpu_backend,
    const TargetWeights &  w,
    const TargetLayer &    L,
    const float *          cur_host,
    const int32_t *        selected_ids,
    const float *          selected_weights,
    int                    n_tokens,
    std::vector<float> &   out,
    std::string *          err = nullptr);

// Batched hybrid prefill FFN: processes n_tokens at once with hot experts on GPU
// and cold experts on CPU concurrently.  Uses pre-computed routing from the pre-FFN
// graph.  Falls back to eval_qwen35moe_batched_prefill_ffn when all selected experts
// are hot.
// cur_host: [n_embd × n_tokens] post-norm hidden states (row-major)
// selected_ids: [n_expert_used × n_tokens] expert selections (global IDs)
// selected_weights: [n_expert_used × n_tokens] routing weights
// out: [n_embd × n_tokens] output (resized internally)
bool eval_qwen35moe_hybrid_ffn_batched(
    ggml_backend_t                      gpu_backend,
    ggml_backend_t                      cpu_backend,
    const TargetWeights &               w,
    const TargetLayer &                 L,
    Qwen35MoeHybridLayerStorage &       storage,
    const float *                       cur_host,
    const int32_t *                     selected_ids,
    const float *                       selected_weights,
    int                                 n_tokens,
    std::vector<float> &                out,
    std::string *                       err = nullptr);

// GPU-resident single-token hybrid FFN eval: keeps data on GPU, only reads
// router IDs to CPU for hot/cold partitioning.  Uses tensor_copy for GPU→GPU
// transfers instead of round-tripping through host memory.
// ffn_post_gpu: [n_embd] F32 on GPU — the post-attention-norm hidden state
// ffn_residual_gpu: [n_embd] F32 on GPU — the pre-FFN residual
// gpu_state: persistent GPU state with act_cur and combine graph
// After call: gpu_state.act_cur holds the layer output on GPU.
bool eval_qwen35moe_hybrid_ffn_gpu_resident(
    ggml_backend_t                      gpu_backend,
    const TargetWeights &               w,
    const TargetLayer &                 L,
    Qwen35MoeHybridLayerStorage &       storage,
    ggml_backend_t                      cpu_backend,
    ggml_tensor *                       ffn_post_gpu,
    ggml_tensor *                       ffn_residual_gpu,
    GpuResidentState &                  gpu_state,
    const int32_t *                     selected_ids,
    const float *                       selected_weights,
    int                                 n_selected);

// Build/rebuild cached hot FFN graph for a given number of hot experts.
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
    const TargetLayer & L,
    int n_embd,
    int n_ff_exp,
    int n_hot);

// Build/rebuild cached cold FFN graph for a given number of cold experts.
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
