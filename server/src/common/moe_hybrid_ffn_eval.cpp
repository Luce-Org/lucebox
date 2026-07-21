#include <cstdlib>
#include "moe_hybrid_ffn_eval.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>

extern "C" void ggml_backend_cuda_set_graphs_disabled_override(bool disabled);

namespace dflash::common {

// Private contract with ggml-cuda/mmq.cu. Large owner-local MUL_MAT_ID graphs
// know the exact maximum number of live rows assigned to any expert. Passing
// that bound avoids launching the full n_tokens-wide grid for every expert.
struct MmidOwnerGridHint {
    uint32_t magic;
    int32_t  max_expert_rows;
    int32_t  live_routes;
};
static constexpr uint32_t MMID_OWNER_GRID_HINT_MAGIC = 0x4D4F4752u; // "MOGR"

// NVFP4 scale2: if weight has a per-tensor scale, multiply the matmul result
// by that scale. No-op when scale==1.0f (non-NVFP4 models).
inline ggml_tensor * apply_scale2(ggml_context * ctx, ggml_tensor * mm_result, float scale) {
    if (scale == 1.0f) return mm_result;
    return ggml_scale(ctx, mm_result, scale);
}

inline ggml_tensor * swiglu_maybe_clamped(ggml_context * ctx,
                                          ggml_tensor * gate,
                                          ggml_tensor * up,
                                          float clamp) {
    if (clamp > 1.0e-6f) {
        // DeepSeek V4 clamps only the upper side of the gate, and both sides of up.
        gate = ggml_clamp(ctx, gate, -INFINITY, clamp);
        up   = ggml_clamp(ctx, up,   -clamp, clamp);
    }
    return ggml_swiglu_split(ctx, gate, up);
}

using HybridClock = std::chrono::steady_clock;

class HybridCudaGraphDisableScope {
public:
    explicit HybridCudaGraphDisableScope(bool active) : active_(active) {
        if (active_) ggml_backend_cuda_set_graphs_disabled_override(true);
    }
    ~HybridCudaGraphDisableScope() {
        if (active_) ggml_backend_cuda_set_graphs_disabled_override(false);
    }

    HybridCudaGraphDisableScope(const HybridCudaGraphDisableScope &) = delete;
    HybridCudaGraphDisableScope & operator=(const HybridCudaGraphDisableScope &) = delete;

private:
    bool active_;
};

static bool heterogeneous_prefill_eager_enabled() {
    const char * raw = std::getenv("DFLASH_DS4_HYBRID_PREFILL_EAGER");
    return raw && *raw && std::strcmp(raw, "0") != 0;
}

static bool prefill_masked_cold_routes_enabled() {
    static const bool enabled = []() {
        const char * raw = std::getenv("DFLASH_MOE_PREFILL_MASKED_COLD");
        return !raw || !*raw || std::strcmp(raw, "0") != 0;
    }();
    return enabled;
}

static bool prefill_compact_cold_grid_enabled() {
    static const bool enabled = []() {
        const char * raw = std::getenv("DFLASH_MOE_PREFILL_COMPACT_COLD_GRID");
        return !raw || !*raw || std::strcmp(raw, "0") != 0;
    }();
    return enabled;
}

static uint64_t elapsed_us(HybridClock::time_point start, HybridClock::time_point end) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

static bool compact_materialized_experts_enabled() {
    static const bool enabled = [] {
        const char * raw = std::getenv("DFLASH_MOE_COMPACT_MATERIALIZED");
        return raw && *raw && std::strcmp(raw, "0") != 0;
    }();
    return enabled;
}

// The legacy ROCmFP2 verifier graph expands a q-token routed FFN into q
// independent single-token subgraphs.  That was required before the CUDA/HIP
// backend gained its grouped MUL_MAT_ID MMVQ kernel, but it multiplies graph
// nodes, scheduler copies, and launches by the verify width.  Keep the old
// lowering as the default while the grouped path is qualified on each ROCm
// architecture; opt in with DFLASH_DS4_TP_GROUPED_MMVQ=1.
static bool grouped_mmvq_moe_enabled() {
    static const bool enabled = [] {
        const char * raw = std::getenv("DFLASH_DS4_TP_GROUPED_MMVQ");
        return raw && *raw && std::strcmp(raw, "0") != 0;
    }();
    return enabled;
}

static bool gpu_i32_repeat_enabled() {
    static const bool enabled = [] {
        const char * raw = std::getenv("LUCE_CUDA_I32_REPEAT");
        return raw && *raw && std::strcmp(raw, "0") != 0;
    }();
    return enabled;
}

// The regular DeepSeek graph already uses ggml_laguna_moe_combine for the
// route-weighted expert reduction.  Keep the heterogeneous graph A/B-able
// while replacing its MUL + shape-only REPEAT_BACK sequence with the same
// exact owner-local kernel.
static bool fused_moe_combine_enabled() {
    static const bool enabled = [] {
        const char * raw = std::getenv("DFLASH_MOE_FUSED_COMBINE");
        return raw && *raw && std::strcmp(raw, "0") != 0;
    }();
    return enabled;
}

// DeepSeek V4 stores routed gate and up projections as one tensor with the
// output rows concatenated.  The generic graph computes that full tensor,
// materializes two contiguous views, clamps both, and only then runs SwiGLU.
// Present the two weight halves as views instead: the CUDA/HIP graph optimizer
// can fuse both MUL_MAT_ID operations and DS4 SwiGLU into one MMVQ launch.
// This is exact when the external gate/up scale is one (the ROCmFP checkpoint
// used by the heterogeneous path); other scale values retain the old graph.
static bool fused_gate_up_mmvq_enabled() {
    static const bool enabled = [] {
        const char * raw = std::getenv("DFLASH_DS4_TP_FUSED_GATE_UP");
        return raw && *raw && std::strcmp(raw, "0") != 0;
    }();
    return enabled;
}

static bool coarse_owner_op_enabled() {
    static const bool enabled = [] {
        const char * raw = std::getenv("DFLASH_DS4_TP_COARSE_OWNER");
        return raw && *raw && std::strcmp(raw, "0") != 0;
    }();
    return enabled;
}

static bool coarse_owner_split_op_enabled() {
    static const bool enabled = [] {
        const char * raw = std::getenv("DFLASH_DS4_TP_COARSE_OWNER_SPLIT");
        return raw && *raw && std::strcmp(raw, "0") != 0;
    }();
    return enabled;
}

static bool align_shared_moe_ids_enabled() {
    static const bool enabled = [] {
        const char * raw = std::getenv("DFLASH_CUDA_MMVQ_MOE_ALIGN_SHARED_IDS");
        const bool requested = raw && *raw && std::strcmp(raw, "0") != 0;
        const char * kernel = std::getenv("DFLASH_CUDA_MMVQ_MOE_KERNEL");
        const bool dedicated_kernel = !kernel || !*kernel ||
            std::strcmp(kernel, "0") != 0;
        if (requested && !dedicated_kernel) {
            std::fprintf(stderr,
                "[ds4-tp] shared-ID alignment disabled because the dedicated "
                "MMVQ MoE kernel is disabled\n");
        }
        return requested && dedicated_kernel;
    }();
    return enabled;
}

static bool device_join_enabled() {
    static const bool enabled = [] {
        const char * raw = std::getenv("DFLASH_DS4_TP_DEVICE_JOIN");
        return raw && *raw && std::strcmp(raw, "0") != 0;
    }();
    return enabled;
}

static bool route_prefork_enabled() {
    static const bool enabled = [] {
        const char * raw = std::getenv("DFLASH_DS4_TP_ROUTE_PREFORK");
        return raw && *raw && std::strcmp(raw, "0") != 0;
    }();
    return enabled;
}

static void add_hybrid_telemetry(MoeHybridFfnTelemetry & dst,
                                 const MoeHybridFfnTelemetry & src) {
    dst.ffn_wall_us += src.ffn_wall_us;
    dst.partition_us += src.partition_us;
    dst.hot_us += src.hot_us;
    dst.cold_us += src.cold_us;
    dst.shared_us += src.shared_us;
    dst.combine_us += src.combine_us;
    dst.hot_graph_build_us += src.hot_graph_build_us;
    dst.hot_input_us += src.hot_input_us;
    dst.hot_compute_us += src.hot_compute_us;
    dst.hot_read_us += src.hot_read_us;
    dst.cold_graph_build_us += src.cold_graph_build_us;
    dst.cold_input_us += src.cold_input_us;
    dst.cold_compute_us += src.cold_compute_us;
    dst.cold_read_us += src.cold_read_us;
    dst.hot_graph_builds += src.hot_graph_builds;
    dst.hot_graph_hits += src.hot_graph_hits;
    dst.cold_graph_builds += src.cold_graph_builds;
    dst.cold_graph_hits += src.cold_graph_hits;
    dst.hot_selected += src.hot_selected;
    dst.cold_selected += src.cold_selected;
}

static int env_int_or_default(const char * name, int fallback) {
    const char * raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    char * end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0') return fallback;
    if (value < 1) return 1;
    if (value > 4096) return 4096;
    return (int)value;
}

static int moe_expert_compute_batch_max() {
    const int raw = env_int_or_default("DFLASH_MOE_EXPERT_COMPUTE_BATCH_MAX", 32);
    return raw > 0 ? raw : 32;
}

enum class MoeExpertComputeIpcMode {
    Stream,
    Batched,
};

static MoeExpertComputeIpcMode parse_moe_expert_compute_ipc_mode() {
    const char * raw = std::getenv("DFLASH_MOE_EXPERT_COMPUTE_IPC_MODE");
    if (!raw || !*raw ||
        std::strcmp(raw, "auto") == 0 ||
        std::strcmp(raw, "AUTO") == 0) {
        return MoeExpertComputeIpcMode::Batched;
    }
    if (std::strcmp(raw, "stream") == 0 ||
        std::strcmp(raw, "STREAM") == 0) {
        return MoeExpertComputeIpcMode::Stream;
    }
    if (std::strcmp(raw, "batched") == 0 ||
        std::strcmp(raw, "BATCHED") == 0) {
        return MoeExpertComputeIpcMode::Batched;
    }
    std::fprintf(stderr,
                 "[hybrid-ffn] ignoring unsupported "
                 "DFLASH_MOE_EXPERT_COMPUTE_IPC_MODE=%s; using auto\n",
                 raw);
    return MoeExpertComputeIpcMode::Batched;
}

// Build the shared-expert FFN subgraph onto an existing ggml_context.
// Returns the output tensor (or nullptr if no shared expert is present).
static ggml_tensor * build_shared_expert_subgraph(
        ggml_context * ctx, const MoeLayerDesc & desc, ggml_tensor * inp,
        float swiglu_clamp = 0.0f) {
    if (!desc.ffn_up_shexp || !desc.ffn_gate_shexp || !desc.ffn_down_shexp)
        return nullptr;
    ggml_tensor * sh_gate = apply_scale2(ctx,
        ggml_mul_mat(ctx, desc.ffn_gate_shexp, inp), desc.ffn_gate_shexp_s);
    ggml_tensor * sh_up = apply_scale2(ctx,
        ggml_mul_mat(ctx, desc.ffn_up_shexp, inp), desc.ffn_up_shexp_s);
    ggml_tensor * sh_gu = swiglu_maybe_clamped(ctx, sh_gate, sh_up, swiglu_clamp);
    ggml_tensor * shared = apply_scale2(ctx,
        ggml_mul_mat(ctx, desc.ffn_down_shexp, sh_gu), desc.ffn_down_shexp_s);
    if (desc.ffn_gate_inp_shexp) {
        // The shared-expert gate is a single-row weight (M=1): out[0,n] = sum_k W[k]*inp[k,n].
        // Computing it as ggml_mul_mat routes to cublas, and on the shipped CUDA 12.0
        // cublasLt the M=1 heuristic selects a gemv/split-K reduce algorithm whose kernel
        // is ABSENT from the library once N>1 (spec-decode verify/replay batches) — for
        // BOTH F32 (cublasSgemm SSS) and F16 (cublasGemmEx HHH splitKreduce). That poisons
        // the stream and surfaces as an illegal access in the next op. Compute the gate as
        // broadcast elementwise-mul + sum_rows instead: identical math, ggml kernels only,
        // no cublas. This is what unblocks single-pass full-batch verify.
        ggml_tensor * gate_prod = ggml_mul(ctx, inp, desc.ffn_gate_inp_shexp);
        ggml_tensor * shared_gate = apply_scale2(ctx,
            ggml_sum_rows(ctx, gate_prod), desc.ffn_gate_inp_shexp_s);
        shared_gate = ggml_sigmoid(ctx, shared_gate);
        shared = ggml_mul(ctx, shared, shared_gate);
    }
    return shared;
}

static int fixed_slot_graphs_mode() {
    static const int mode = [] {
        const char * env = std::getenv("DFLASH_MOE_FIXED_SLOT_GRAPHS");
        if (!env || !env[0] || std::strcmp(env, "0") == 0) return 0;
        if (std::strcmp(env, "adaptive") == 0) return 2;
        return 1;
    }();
    return mode;
}

static int fixed_slot_max() {
    static const int max_slots = [] {
        const char * env = std::getenv("DFLASH_MOE_FIXED_SLOT_MAX");
        return env ? std::max(0, std::atoi(env)) : 0;
    }();
    return max_slots;
}

// Run routed expert subset on a given backend (GPU or CPU).
static bool run_routed_subset(ggml_backend_t backend,
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
                              float swiglu_clamp,
                              const float * cur_host,
                              const int32_t * selected_ids,
                              const float * selected_weights,
                              int n_selected,
                              std::vector<float> & out,
                              std::string * err) {
    out.assign((size_t)n_embd, 0.0f);
    if (n_selected <= 0) return true;

    ggml_init_params ip{};
    ip.mem_size = 32 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        if (err) *err = "ggml_init failed";
        return false;
    }

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_input(inp);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_selected, 1);
    ggml_set_input(ids);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_selected, 1);
    ggml_set_input(weights);

    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, inp, n_embd, 1, 1);
    auto validate_mm_id_tensor = [&](const char * name, ggml_tensor * t) -> bool {
        if (!t) return true;
        if (t->ne[3] != 1) {
            std::fprintf(stderr, "[hybrid-ffn] %s ptr=%p ne=[%lld,%lld,%lld,%lld]\n",
                         name, (void *)t,
                         (long long)t->ne[0], (long long)t->ne[1],
                         (long long)t->ne[2], (long long)t->ne[3]);
            if (err) {
                *err = std::string(name) + " has ne[3]=" + std::to_string((long long)t->ne[3]);
            }
            ggml_free(ctx);
            return false;
        }
        return true;
    };
    if (!validate_mm_id_tensor("gate_tensor", gate_tensor) ||
        !validate_mm_id_tensor("up_tensor", up_tensor) ||
        !validate_mm_id_tensor("down_tensor", down_tensor) ||
        !validate_mm_id_tensor("gate_up_tensor", gate_up_tensor)) {
        return false;
    }
    ggml_tensor * gu = nullptr;
    if (gate_up_tensor) {
        ggml_tensor * gate_up_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, gate_up_tensor, cur_3d, ids), gate_up_scale);
        ggml_tensor * gate_e = ggml_view_3d(ctx, gate_up_e,
            n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
            gate_up_e->nb[1], gate_up_e->nb[2], 0);
        ggml_tensor * up_e = ggml_view_3d(ctx, gate_up_e,
            n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
            gate_up_e->nb[1], gate_up_e->nb[2],
            (size_t)n_ff_exp * ggml_element_size(gate_up_e));
        gate_e = ggml_cont(ctx, gate_e);
        up_e = ggml_cont(ctx, up_e);
        gu = swiglu_maybe_clamped(ctx, gate_e, up_e, swiglu_clamp);
    } else {
        ggml_tensor * gate_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, gate_tensor, cur_3d, ids), gate_scale);
        ggml_tensor * up_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, up_tensor, cur_3d, ids), up_scale);
        gu = swiglu_maybe_clamped(ctx, gate_e, up_e, swiglu_clamp);
    }

    ggml_tensor * experts = apply_scale2(ctx,
        ggml_mul_mat_id(ctx, down_tensor, gu, ids), down_scale);
    ggml_tensor * w_view = ggml_reshape_3d(ctx, weights, 1, n_selected, 1);
    experts = ggml_mul(ctx, experts, w_view);

    ggml_tensor * routed = nullptr;
    for (int i = 0; i < n_selected; ++i) {
        ggml_tensor * slice = ggml_view_2d(ctx, experts, n_embd, 1, experts->nb[2],
                                           (size_t)i * experts->nb[1]);
        routed = (i == 0) ? slice : ggml_add(ctx, routed, slice);
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);
    ggml_set_output(routed);
    ggml_build_forward_expand(gf, routed);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        if (err) *err = "ggml_gallocr_alloc_graph failed";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(inp, cur_host, 0, sizeof(float) * (size_t)n_embd);
    ggml_backend_tensor_set(ids, selected_ids, 0, sizeof(int32_t) * (size_t)n_selected);
    ggml_backend_tensor_set(weights, selected_weights, 0, sizeof(float) * (size_t)n_selected);

    auto st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        if (err) *err = "ggml_backend_graph_compute failed";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(routed, out.data(), 0, sizeof(float) * (size_t)n_embd);
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

// Shared expert FFN on GPU.
static bool run_shared_ffn_gpu(ggml_backend_t backend,
                               const MoeLayerDesc & desc,
                               int n_embd,
                               const float * cur_host,
                               std::vector<float> & out,
                               std::string * err) {
    out.assign((size_t)n_embd, 0.0f);
    if (!desc.ffn_up_shexp || !desc.ffn_gate_shexp || !desc.ffn_down_shexp) {
        return true;
    }

    ggml_init_params ip{};
    ip.mem_size = 16 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        if (err) *err = "ggml_init failed";
        return false;
    }

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_input(inp);

    ggml_tensor * shared = build_shared_expert_subgraph(ctx, desc, inp);
    if (!shared) {
        ggml_free(ctx);
        return true;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 512, false);
    ggml_set_output(shared);
    ggml_build_forward_expand(gf, shared);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        if (err) *err = "ggml_gallocr_alloc_graph failed";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(inp, cur_host, 0, sizeof(float) * (size_t)n_embd);
    auto st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        if (err) *err = "ggml_backend_graph_compute failed";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_get(shared, out.data(), 0, sizeof(float) * (size_t)n_embd);
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

// Fused hot routed + shared FFN in a single GPU graph compute.
static bool run_hot_and_shared_ffn_gpu(
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
    float swiglu_clamp,
    const float * cur_host,
    const int32_t * hot_ids,
    const float * hot_weights,
    int n_hot,
    std::vector<float> & out,
    std::string * err) {

    out.assign((size_t)n_embd, 0.0f);

    const bool has_hot = (n_hot > 0);
    const bool has_shared = (desc.ffn_up_shexp && desc.ffn_gate_shexp && desc.ffn_down_shexp);
    if (!has_hot && !has_shared) return true;

    ggml_init_params ip{};
    ip.mem_size = 48 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        if (err) *err = "ggml_init failed";
        return false;
    }

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_input(inp);

    ggml_tensor * routed = nullptr;
    ggml_tensor * ids_tensor = nullptr;
    ggml_tensor * weights_tensor = nullptr;

    if (has_hot) {
        ids_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_hot, 1);
        ggml_set_input(ids_tensor);
        weights_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hot, 1);
        ggml_set_input(weights_tensor);

        ggml_tensor * cur_3d = ggml_reshape_3d(ctx, inp, n_embd, 1, 1);
        ggml_tensor * gu = nullptr;
        if (gate_up_tensor) {
            ggml_tensor * gate_up_e = apply_scale2(ctx,
                ggml_mul_mat_id(ctx, gate_up_tensor, cur_3d, ids_tensor), gate_up_scale);
            ggml_tensor * gate_e = ggml_view_3d(ctx, gate_up_e,
                n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
                gate_up_e->nb[1], gate_up_e->nb[2], 0);
            ggml_tensor * up_e = ggml_view_3d(ctx, gate_up_e,
                n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
                gate_up_e->nb[1], gate_up_e->nb[2],
                (size_t)n_ff_exp * ggml_element_size(gate_up_e));
            gate_e = ggml_cont(ctx, gate_e);
            up_e = ggml_cont(ctx, up_e);
            gu = swiglu_maybe_clamped(ctx, gate_e, up_e, swiglu_clamp);
        } else {
            ggml_tensor * gate_e = apply_scale2(ctx,
                ggml_mul_mat_id(ctx, gate_tensor, cur_3d, ids_tensor), gate_scale);
            ggml_tensor * up_e = apply_scale2(ctx,
                ggml_mul_mat_id(ctx, up_tensor, cur_3d, ids_tensor), up_scale);
            gu = swiglu_maybe_clamped(ctx, gate_e, up_e, swiglu_clamp);
        }

        ggml_tensor * experts = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, down_tensor, gu, ids_tensor), down_scale);
        ggml_tensor * w_view = ggml_reshape_3d(ctx, weights_tensor, 1, n_hot, 1);
        experts = ggml_mul(ctx, experts, w_view);

        for (int i = 0; i < n_hot; ++i) {
            ggml_tensor * slice = ggml_view_2d(ctx, experts, n_embd, 1, experts->nb[2],
                                               (size_t)i * experts->nb[1]);
            routed = (i == 0) ? slice : ggml_add(ctx, routed, slice);
        }
    }

    ggml_tensor * shared = build_shared_expert_subgraph(ctx, desc, inp, swiglu_clamp);

    // Combine hot routed + shared into a single output tensor
    ggml_tensor * combined = nullptr;
    if (routed && shared) {
        combined = ggml_add(ctx, routed, shared);
    } else if (routed) {
        combined = routed;
    } else {
        combined = shared;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 2048, false);
    ggml_set_output(combined);
    ggml_build_forward_expand(gf, combined);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        if (err) *err = "fused hot+shared gallocr failed";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(inp, cur_host, 0, sizeof(float) * (size_t)n_embd);
    if (ids_tensor) {
        ggml_backend_tensor_set(ids_tensor, hot_ids, 0, sizeof(int32_t) * (size_t)n_hot);
    }
    if (weights_tensor) {
        ggml_backend_tensor_set(weights_tensor, hot_weights, 0, sizeof(float) * (size_t)n_hot);
    }

    auto st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        if (err) *err = "fused hot+shared compute failed";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(combined, out.data(), 0, sizeof(float) * (size_t)n_embd);
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

// Build batched routed graph helper for batched prefill.
static bool build_batched_routed_graph(
    ggml_context * ctx,
    ggml_tensor * gate_tensor,
    ggml_tensor * up_tensor,
    ggml_tensor * down_tensor,
    ggml_tensor * gate_up_tensor,
    float gate_scale,
    float up_scale,
    float down_scale,
    float gate_up_scale,
    ggml_tensor * inp,
    ggml_tensor * sel,
    ggml_tensor * wts,
    int n_embd, int n_ff_exp, int n_used, int n_tokens,
    float swiglu_clamp,
    ggml_tensor ** out_routed,
    bool tokenwise = false,
    std::vector<ggml_tensor *> * backend_nodes = nullptr,
    bool allow_fused_combine = false,
    bool force_fused_combine = false)
{
    const auto track = [&](ggml_tensor * t) -> ggml_tensor * {
        if (backend_nodes && t) backend_nodes->push_back(t);
        return t;
    };
    if (tokenwise && n_tokens > 1) {
        ggml_tensor * joined = nullptr;
        for (int t = 0; t < n_tokens; ++t) {
            ggml_tensor * inp_col = ggml_cont(ctx, ggml_view_2d(
                ctx, inp, n_embd, 1, inp->nb[1], (size_t) t * inp->nb[1]));
            ggml_tensor * sel_col = ggml_cont(ctx, ggml_view_2d(
                ctx, sel, n_used, 1, sel->nb[1], (size_t) t * sel->nb[1]));
            ggml_tensor * wts_col = ggml_cont(ctx, ggml_view_2d(
                ctx, wts, n_used, 1, wts->nb[1], (size_t) t * wts->nb[1]));
            ggml_tensor * routed_col = nullptr;
            if (!build_batched_routed_graph(
                    ctx, gate_tensor, up_tensor, down_tensor, gate_up_tensor,
                    gate_scale, up_scale, down_scale, gate_up_scale,
                    inp_col, sel_col, wts_col,
                    n_embd, n_ff_exp, n_used, 1, swiglu_clamp,
                    &routed_col, false, backend_nodes,
                    allow_fused_combine, force_fused_combine)) {
                return false;
            }
            joined = joined ? track(ggml_concat(ctx, joined, routed_col, 1))
                            : routed_col;
        }
        *out_routed = joined;
        return joined != nullptr;
    }

    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, inp, n_embd, 1, n_tokens);
    ggml_tensor * gu = nullptr;
    const bool coarse_split_requested =
        coarse_owner_op_enabled() && coarse_owner_split_op_enabled();
    const bool coarse_split_eligible =
        gate_tensor && up_tensor &&
        gate_tensor->type == GGML_TYPE_Q2_0_ROCMFP2 &&
        up_tensor->type == GGML_TYPE_Q2_0_ROCMFP2 &&
        down_tensor->type == GGML_TYPE_Q3_0_ROCMFPX;
    if (coarse_split_requested) {
        static bool logged_active = false;
        static bool logged_ineligible = false;
        bool & logged = coarse_split_eligible ? logged_active : logged_ineligible;
        if (!logged) {
            std::fprintf(stderr,
                "[ds4-tp] split-owner %s gate=%s up=%s down=%s gate_up=%s tokens=%d routes=%d\n",
                coarse_split_eligible ? "active" : "ineligible",
                gate_tensor ? ggml_type_name(gate_tensor->type) : "none",
                up_tensor ? ggml_type_name(up_tensor->type) : "none",
                down_tensor ? ggml_type_name(down_tensor->type) : "none",
                gate_up_tensor ? ggml_type_name(gate_up_tensor->type) : "none",
                n_tokens, n_used);
            logged = true;
        }
    }
    if (coarse_split_requested && coarse_split_eligible) {
        *out_routed = track(ggml_ds4_moe_owner_split(
            ctx, inp, gate_tensor, up_tensor, down_tensor, sel, wts,
            n_ff_exp, swiglu_clamp,
            gate_scale, up_scale, down_scale));
        return *out_routed != nullptr;
    } else if (coarse_owner_op_enabled() &&
        gate_up_tensor &&
        gate_up_scale == 1.0f &&
        gate_up_tensor->type == GGML_TYPE_Q2_0_ROCMFP2 &&
        down_tensor->type == GGML_TYPE_Q3_0_ROCMFPX) {
        *out_routed = track(ggml_ds4_moe_owner(
            ctx, inp, gate_up_tensor, down_tensor, sel, wts,
            n_ff_exp, swiglu_clamp, down_scale));
        return *out_routed != nullptr;
    } else if (gate_up_tensor &&
        fused_gate_up_mmvq_enabled() &&
        gate_up_scale == 1.0f) {
        GGML_ASSERT(gate_up_tensor->ne[1] == 2 * n_ff_exp);
        ggml_tensor * gate_w = ggml_view_3d(
            ctx, gate_up_tensor,
            gate_up_tensor->ne[0], n_ff_exp, gate_up_tensor->ne[2],
            gate_up_tensor->nb[1], gate_up_tensor->nb[2], 0);
        ggml_tensor * up_w = ggml_view_3d(
            ctx, gate_up_tensor,
            gate_up_tensor->ne[0], n_ff_exp, gate_up_tensor->ne[2],
            gate_up_tensor->nb[1], gate_up_tensor->nb[2],
            (size_t) n_ff_exp * gate_up_tensor->nb[1]);
        ggml_tensor * gate_e = track(
            ggml_mul_mat_id(ctx, gate_w, cur_3d, sel));
        ggml_tensor * up_e = track(
            ggml_mul_mat_id(ctx, up_w, cur_3d, sel));
        gu = track(swiglu_clamp > 1.0e-6f
            ? ggml_swiglu_ds4_split(ctx, gate_e, up_e, swiglu_clamp)
            : ggml_swiglu_split(ctx, gate_e, up_e));
    } else if (gate_up_tensor) {
        ggml_tensor * gate_up_e = track(apply_scale2(ctx,
            ggml_mul_mat_id(ctx, gate_up_tensor, cur_3d, sel), gate_up_scale));
        ggml_tensor * gate_e = ggml_view_3d(ctx, gate_up_e,
            n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
            gate_up_e->nb[1], gate_up_e->nb[2], 0);
        ggml_tensor * up_e = ggml_view_3d(ctx, gate_up_e,
            n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
            gate_up_e->nb[1], gate_up_e->nb[2],
            (size_t)n_ff_exp * ggml_element_size(gate_up_e));
        gate_e = track(ggml_cont(ctx, gate_e));
        up_e = track(ggml_cont(ctx, up_e));
        gu = track(swiglu_maybe_clamped(ctx, gate_e, up_e, swiglu_clamp));
    } else {
        ggml_tensor * gate_e = track(apply_scale2(ctx,
            ggml_mul_mat_id(ctx, gate_tensor, cur_3d, sel), gate_scale));
        ggml_tensor * up_e = track(apply_scale2(ctx,
            ggml_mul_mat_id(ctx, up_tensor, cur_3d, sel), up_scale));
        gu = track(swiglu_maybe_clamped(ctx, gate_e, up_e, swiglu_clamp));
    }

    ggml_tensor * experts = track(apply_scale2(ctx,
        ggml_mul_mat_id(ctx, down_tensor, gu, sel), down_scale));

    // Weight and sum over experts: [n_embd, n_used, n_tokens] * [1, n_used, n_tokens]
    if (allow_fused_combine &&
        (force_fused_combine || fused_moe_combine_enabled())) {
        *out_routed = track(ggml_laguna_moe_combine(ctx, experts, wts));
        return *out_routed != nullptr;
    }

    ggml_tensor * w_view = ggml_reshape_3d(ctx, wts, 1, n_used, n_tokens);
    experts = track(ggml_mul(ctx, experts, w_view));

    // repeat_back uses this tensor for shape only, but the scheduler still
    // treats it as a leaf. Keep it on the branch backend; otherwise every MoE
    // branch acquires a tiny CPU split solely for an uninitialized shape leaf.
    ggml_tensor * sum_shape = track(
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens));
    ggml_tensor * moe_sum = track(ggml_repeat_back(ctx, experts, sum_shape));
    *out_routed = track(ggml_reshape_2d(ctx, moe_sum, n_embd, n_tokens));
    return true;
}

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
    bool                           include_shared,
    bool                           allow_fused_combine) {

    out.output = nullptr;
    out.main_output = nullptr;
    out.peer_output = nullptr;
    if (!ctx || !inp || !global_ids || !router_weights || n_tokens <= 0 ||
        cfg.n_embd <= 0 || cfg.n_ff_exp <= 0 || cfg.n_expert <= 0 ||
        cfg.n_expert_used <= 0) {
        return false;
    }

    const int n_used = cfg.n_expert_used;
    // Both owner remaps consume the same normalized top-k route weights.
    // Expose the canonical tensor so the heterogeneous scheduler can keep it
    // on the main GPU. Otherwise expanding the cold branch first lets backend
    // assignment migrate this shared dependency to Strix, forcing the hot
    // R9700 branch to wait for a reverse peer copy before it can launch.
    out.router_weights = router_weights;
    auto build_remap = [&](const std::vector<int32_t> & local_by_global,
                           ggml_tensor ** local_lut,
                           ggml_tensor ** valid_lut,
                           ggml_tensor ** local_ids,
                           ggml_tensor ** masked_weights,
                           std::vector<ggml_tensor *> * backend_nodes) {
        auto track = [backend_nodes](ggml_tensor * tensor) {
            if (tensor && backend_nodes) backend_nodes->push_back(tensor);
            return tensor;
        };
        if (!*local_lut) {
            *local_lut = ggml_new_tensor_2d(
                ctx, GGML_TYPE_I32, 1, cfg.n_expert);
            ggml_set_input(*local_lut);
            // These inputs are consumed late in a whole-model graph. Preserve
            // their allocation from graph start; otherwise gallocr may reuse
            // the tiny buffer as activation scratch before its layer executes.
            ggml_set_output(*local_lut);
        }
        if (!*valid_lut) {
            *valid_lut = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, 1, cfg.n_expert);
            ggml_set_input(*valid_lut);
            ggml_set_output(*valid_lut);
        }

        // q1 can address the 2-D LUT directly.  Historically q>1 left its tiny
        // I32 repeat unpinned for CPU fallback.  With the exact GPU I32 repeat
        // enabled, track it with the rest of the owner-local remap nodes.
        ggml_tensor * local_lut_batched = *local_lut;
        if (n_tokens > 1) {
            ggml_tensor * repeated = ggml_repeat_4d(
                ctx, *local_lut, 1, cfg.n_expert, n_tokens, 1);
            local_lut_batched =
                gpu_i32_repeat_enabled() ? track(repeated) : repeated;
        }
        ggml_tensor * mapped = track(ggml_get_rows(
            ctx, local_lut_batched, global_ids));
        mapped = track(ggml_reshape_2d(ctx, mapped, n_used, n_tokens));
        *local_ids = track(ggml_cont(ctx, mapped));
        ggml_tensor * valid_lut_batched = *valid_lut;
        if (n_tokens > 1) {
            valid_lut_batched = track(ggml_repeat_4d(
                ctx, *valid_lut, 1, cfg.n_expert, n_tokens, 1));
        }
        ggml_tensor * valid = track(ggml_get_rows(
            ctx, valid_lut_batched, global_ids));
        valid = track(ggml_reshape_2d(ctx, valid, n_used, n_tokens));
        *masked_weights = track(ggml_mul(ctx, router_weights, valid));
        return (int)local_by_global.size() == cfg.n_expert;
    };

    ggml_tensor * hot_ids = nullptr;
    ggml_tensor * hot_weights = nullptr;
    if (!build_remap(storage.hot_local_by_global,
                     &out.hot_local_lut, &out.hot_valid_lut,
                     &hot_ids, &hot_weights, &out.hot_remap_nodes)) {
        return false;
    }

    ggml_tensor * cold_ids = nullptr;
    ggml_tensor * cold_weights = nullptr;
    if (!build_remap(storage.cold_local_by_global,
                     &out.cold_local_lut, &out.cold_valid_lut,
                     &cold_ids, &cold_weights, &out.cold_remap_nodes)) {
        return false;
    }

    ggml_tensor * shard_ids = nullptr;
    ggml_tensor * shard_weights = nullptr;
    const bool has_cold_shard = storage.expert_shard_channels > 0 &&
        storage.gate_shard_hot && storage.up_shard_hot &&
        storage.down_shard_hot;
    if (has_cold_shard &&
        !build_remap(storage.cold_local_by_global,
                     &out.shard_local_lut, &out.shard_valid_lut,
                     &shard_ids, &shard_weights, &out.shard_remap_nodes)) {
        return false;
    }

    // q-token verification often routes adjacent tokens to the same expert
    // at different top-k ranks.  Align those owner-local IDs before MMVQ so
    // equal weights are consumed by warps in one block.  The encoded original
    // route slot is decoded by the dedicated MoE kernel, which scatters every
    // result back before the unchanged weighted reduction.
    if (n_tokens > 1 && align_shared_moe_ids_enabled()) {
        hot_ids = ggml_ds4_moe_align_ids(ctx, hot_ids);
        cold_ids = ggml_ds4_moe_align_ids(ctx, cold_ids);
        out.hot_remap_nodes.push_back(hot_ids);
        out.cold_remap_nodes.push_back(cold_ids);
        if (has_cold_shard) {
            shard_ids = ggml_ds4_moe_align_ids(ctx, shard_ids);
            out.shard_remap_nodes.push_back(shard_ids);
        }
    }


    ggml_tensor * shard = nullptr;
    if (has_cold_shard) {
        const bool tokenwise =
            storage.gate_shard_hot->type == GGML_TYPE_Q2_0_ROCMFP2 &&
            !(n_tokens > 1 && grouped_mmvq_moe_enabled());
        if (!build_batched_routed_graph(
                ctx,
                storage.gate_shard_hot, storage.up_shard_hot,
                storage.down_shard_hot, nullptr,
                desc.ffn_gate_exps_s, desc.ffn_up_exps_s,
                desc.ffn_down_exps_s, 1.0f,
                inp, shard_ids, shard_weights,
                cfg.n_embd, storage.expert_shard_channels,
                n_used, n_tokens, cfg.swiglu_clamp,
                &shard, tokenwise, &out.shard_nodes,
                allow_fused_combine)) {
            return false;
        }
    }

    ggml_tensor * hot = nullptr;
    if ((storage.gate_up_hot || (storage.gate_hot && storage.up_hot)) &&
        storage.down_hot) {
        const ggml_tensor * hot_gate = storage.gate_up_hot
            ? storage.gate_up_hot : storage.gate_hot;
        const bool tokenwise =
            hot_gate->type == GGML_TYPE_Q2_0_ROCMFP2 &&
            !(n_tokens > 1 && grouped_mmvq_moe_enabled());
        if (!build_batched_routed_graph(
                ctx,
                storage.gate_hot, storage.up_hot, storage.down_hot,
                storage.gate_up_hot,
                desc.ffn_gate_exps_s, desc.ffn_up_exps_s,
                desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                inp, hot_ids, hot_weights,
                cfg.n_embd, cfg.n_ff_exp, n_used, n_tokens,
                cfg.swiglu_clamp, &hot, tokenwise,
                &out.hot_nodes, allow_fused_combine)) {
            return false;
        }
    }

    ggml_tensor * cold = nullptr;
    if ((storage.gate_up_cold || (storage.gate_cold && storage.up_cold)) &&
        storage.down_cold) {
        const ggml_tensor * cold_gate = storage.gate_up_cold
            ? storage.gate_up_cold : storage.gate_cold;
        const bool tokenwise =
            cold_gate->type == GGML_TYPE_Q2_0_ROCMFP2 &&
            !(n_tokens > 1 && grouped_mmvq_moe_enabled());
        if (!build_batched_routed_graph(
                ctx,
                storage.gate_cold, storage.up_cold, storage.down_cold,
                storage.gate_up_cold,
                desc.ffn_gate_exps_s, desc.ffn_up_exps_s,
                desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                inp, cold_ids, cold_weights,
                cfg.n_embd,
                storage.expert_shard_channels > 0
                    ? cfg.n_ff_exp - storage.expert_shard_channels
                    : cfg.n_ff_exp,
                n_used, n_tokens,
                cfg.swiglu_clamp, &cold, tokenwise,
                &out.cold_nodes, allow_fused_combine)) {
            return false;
        }
    }

    ggml_tensor * main_branch = hot;
    if (shard) {
        main_branch = main_branch ? ggml_add(ctx, main_branch, shard) : shard;
        out.shard_nodes.push_back(main_branch);
    }
    ggml_tensor * shared = include_shared
        ? build_shared_expert_subgraph(ctx, desc, inp, cfg.swiglu_clamp)
        : nullptr;
    if (shared) main_branch = main_branch ? ggml_add(ctx, main_branch, shared) : shared;

    // The generic scheduler copies every cross-backend input before launching
    // any node in a split. If hot/shared and the final add are one contiguous
    // main-backend split, the add's cold input makes that whole split wait for
    // the peer, serializing the nominally parallel branches.
    //
    // Expand cold now, then visit hot/shared, then a new peer-owned CONT fence,
    // and finally the main-backend add. This yields:
    //   peer cold compute -> main hot/shared compute -> peer fence -> main join
    // The scheduler can enqueue cold first and hot/shared second; the fence
    // separates the final join so its event wait is inserted after hot/shared.
    ggml_tensor * combined = nullptr;
    if (schedule_graph && cold && device_join_enabled()) {
        // Materialize both route IDs and normalized route weights before the
        // cold split is expanded.  Cross-device copies are not guaranteed to
        // remain asynchronous on this heterogeneous ROCm pair.  If the tiny
        // weight copy is discovered after cold expert execution, the fallback
        // copy synchronizes the Strix stream and prevents the host from
        // enqueueing independent R9700 hot work until cold has completed.
        //
        // q4 verification calls this builder twice (4 routes + padded 2), so
        // retain every derived route tensor rather than only the canonical
        // six-wide routing output.
        if (route_prefork_enabled()) {
            out.route_prefork_nodes.push_back(global_ids);
            out.route_prefork_nodes.push_back(router_weights);
            ggml_build_forward_expand(schedule_graph, global_ids);
            ggml_build_forward_expand(schedule_graph, router_weights);
        }
        // Enforce fork order without adding another backend graph:
        //   cold owner -> hot/shared -> in-graph event wait/copy -> add.
        // The deferred copy remains in the same main-backend split as the
        // hot branch, so its wait is reached only after useful main work.
        // Keep the peer result live explicitly. The generic allocator normally
        // derives lifetime from children on the same execution backend; this
        // custom foreign-buffer edge intentionally bypasses that copy path.
        ggml_set_output(cold);
        ggml_build_forward_expand(schedule_graph, cold);
        if (main_branch) {
            ggml_build_forward_expand(schedule_graph, main_branch);
        }
        ggml_tensor * cold_ready =
            ggml_ds4_deferred_peer_copy(ctx, cold);
        // The scheduler may prefill this tensor in the host-copy diagnostic
        // before its containing main split launches. Reserve a stable buffer
        // from graph start so earlier hot-branch scratch cannot alias it.
        ggml_set_input(cold_ready);
        ggml_set_output(cold_ready);
        out.deferred_peer_copy_nodes.push_back(cold_ready);
        out.main_output = main_branch;
        out.peer_output = cold_ready;
        combined = main_branch ? ggml_add(ctx, cold_ready, main_branch)
                               : cold_ready;
    } else if (schedule_graph && cold) {
        ggml_build_forward_expand(schedule_graph, cold);
        ggml_tensor * cold_fence = ggml_cont(ctx, cold);
        out.cold_nodes.push_back(cold_fence);
        combined = main_branch ? ggml_add(ctx, main_branch, cold_fence)
                               : cold_fence;
    } else {
        // Preserve the established default dependency order exactly.
        if (cold && main_branch) {
            combined = ggml_add(ctx, cold, main_branch);
            out.join_nodes.push_back(combined);
        } else {
            combined = cold ? cold : main_branch;
        }
    }
    if (!combined) return false;

    out.output = ggml_cont(ctx, combined);
    return true;
}

// ── Public API ──────────────────────────────────────────────────────────────────

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
    CachedHotGraphOptions options) {

    const float swiglu_clamp = options.swiglu_clamp;
    const bool gpu_remap = options.gpu_remap;
    const int n_expert = options.n_expert;

    out.free();
    out.n_hot = n_hot;

    ggml_init_params ip{};
    ip.mem_size = 48 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    out.inp = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_input(out.inp);

    ggml_tensor * routed = nullptr;
    if (n_hot > 0) {
        out.ids = ggml_new_tensor_2d(out.ctx, GGML_TYPE_I32, n_hot, 1);
        ggml_set_input(out.ids);
        out.weights = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_hot, 1);
        ggml_set_input(out.weights);
        if (gpu_remap && n_expert > 0) {
            out.global_ids = ggml_new_tensor_2d(out.ctx, GGML_TYPE_I32, n_hot, 1);
            ggml_set_input(out.global_ids);
            out.raw_weights = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_hot, 1);
            ggml_set_input(out.raw_weights);
            out.hot_local_lut = ggml_new_tensor_2d(out.ctx, GGML_TYPE_I32, 1, n_expert);
            ggml_set_input(out.hot_local_lut);
            out.valid_lut = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, 1, n_expert);
            ggml_set_input(out.valid_lut);
            out.residual_in = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_embd, 1);
            ggml_set_input(out.residual_in);
            ggml_tensor * lid = ggml_get_rows(out.ctx, out.hot_local_lut, out.global_ids);
            out.ids = ggml_cont(out.ctx, ggml_reshape_2d(out.ctx, lid, n_hot, 1));
            ggml_tensor * vm = ggml_get_rows(out.ctx, out.valid_lut, out.global_ids);
            vm = ggml_reshape_2d(out.ctx, vm, n_hot, 1);
            out.weights = ggml_mul(out.ctx, out.raw_weights, vm);
        }

        ggml_tensor * cur_3d = ggml_reshape_3d(out.ctx, out.inp, n_embd, 1, 1);
        ggml_tensor * gu = nullptr;
        if (gate_up_tensor) {
            ggml_tensor * gate_up_e = apply_scale2(out.ctx,
                ggml_mul_mat_id(out.ctx, gate_up_tensor, cur_3d, out.ids), gate_up_scale);
            ggml_tensor * gate_e = ggml_view_3d(out.ctx, gate_up_e,
                n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
                gate_up_e->nb[1], gate_up_e->nb[2], 0);
            ggml_tensor * up_e = ggml_view_3d(out.ctx, gate_up_e,
                n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
                gate_up_e->nb[1], gate_up_e->nb[2],
                (size_t)n_ff_exp * ggml_element_size(gate_up_e));
            gate_e = ggml_cont(out.ctx, gate_e);
            up_e = ggml_cont(out.ctx, up_e);
            gu = swiglu_maybe_clamped(out.ctx, gate_e, up_e, swiglu_clamp);
        } else {
            ggml_tensor * gate_e = apply_scale2(out.ctx,
                ggml_mul_mat_id(out.ctx, gate_tensor, cur_3d, out.ids), gate_scale);
            ggml_tensor * up_e = apply_scale2(out.ctx,
                ggml_mul_mat_id(out.ctx, up_tensor, cur_3d, out.ids), up_scale);
            gu = swiglu_maybe_clamped(out.ctx, gate_e, up_e, swiglu_clamp);
        }

        ggml_tensor * experts = apply_scale2(out.ctx,
            ggml_mul_mat_id(out.ctx, down_tensor, gu, out.ids), down_scale);
        ggml_tensor * w_view = ggml_reshape_3d(out.ctx, out.weights, 1, n_hot, 1);
        experts = ggml_mul(out.ctx, experts, w_view);

        for (int i = 0; i < n_hot; ++i) {
            ggml_tensor * slice = ggml_view_2d(out.ctx, experts, n_embd, 1, experts->nb[2],
                                               (size_t)i * experts->nb[1]);
            routed = (i == 0) ? slice : ggml_add(out.ctx, routed, slice);
        }
    }

    ggml_tensor * shared = build_shared_expert_subgraph(out.ctx, desc, out.inp, swiglu_clamp);

    if (routed && shared) {
        out.output = ggml_add(out.ctx, routed, shared);
    } else if (routed) {
        out.output = routed;
    } else {
        out.output = shared;
    }
    if (!out.output) { out.free(); return false; }
    if (gpu_remap && out.residual_in) { out.output = ggml_cont(out.ctx, ggml_add(out.ctx, out.output, out.residual_in)); }

    out.gf = ggml_new_graph_custom(out.ctx, 2048, false);
    ggml_set_output(out.output);
    ggml_build_forward_expand(out.gf, out.output);
    out.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(out.alloc, out.gf)) {
        out.free();
        return false;
    }
    return true;
}

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
    float swiglu_clamp) {

    out.free();
    out.n_hot = n_cold;  // reuse field for "n experts in this graph"

    ggml_init_params ip{};
    ip.mem_size = 32 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    out.inp = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_input(out.inp);
    out.ids = ggml_new_tensor_2d(out.ctx, GGML_TYPE_I32, n_cold, 1);
    ggml_set_input(out.ids);
    out.weights = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_cold, 1);
    ggml_set_input(out.weights);

    ggml_tensor * cur_3d = ggml_reshape_3d(out.ctx, out.inp, n_embd, 1, 1);
    ggml_tensor * gu = nullptr;
    if (gate_up_tensor) {
        ggml_tensor * gate_up_e = apply_scale2(out.ctx,
            ggml_mul_mat_id(out.ctx, gate_up_tensor, cur_3d, out.ids), gate_up_scale);
        ggml_tensor * gate_e = ggml_view_3d(out.ctx, gate_up_e,
            n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
            gate_up_e->nb[1], gate_up_e->nb[2], 0);
        ggml_tensor * up_e = ggml_view_3d(out.ctx, gate_up_e,
            n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
            gate_up_e->nb[1], gate_up_e->nb[2],
            (size_t)n_ff_exp * ggml_element_size(gate_up_e));
        gate_e = ggml_cont(out.ctx, gate_e);
        up_e = ggml_cont(out.ctx, up_e);
        gu = swiglu_maybe_clamped(out.ctx, gate_e, up_e, swiglu_clamp);
    } else {
        ggml_tensor * gate_e = apply_scale2(out.ctx,
            ggml_mul_mat_id(out.ctx, gate_tensor, cur_3d, out.ids), gate_scale);
        ggml_tensor * up_e = apply_scale2(out.ctx,
            ggml_mul_mat_id(out.ctx, up_tensor, cur_3d, out.ids), up_scale);
        gu = swiglu_maybe_clamped(out.ctx, gate_e, up_e, swiglu_clamp);
    }

    ggml_tensor * experts = apply_scale2(out.ctx,
        ggml_mul_mat_id(out.ctx, down_tensor, gu, out.ids), down_scale);
    ggml_tensor * w_view = ggml_reshape_3d(out.ctx, out.weights, 1, n_cold, 1);
    experts = ggml_mul(out.ctx, experts, w_view);

    out.output = nullptr;
    for (int i = 0; i < n_cold; ++i) {
        ggml_tensor * slice = ggml_view_2d(out.ctx, experts, n_embd, 1, experts->nb[2],
                                           (size_t)i * experts->nb[1]);
        out.output = (i == 0) ? slice : ggml_add(out.ctx, out.output, slice);
    }
    if (!out.output) { out.free(); return false; }

    out.gf = ggml_new_graph_custom(out.ctx, 1024, false);
    ggml_set_output(out.output);
    ggml_build_forward_expand(out.gf, out.output);
    out.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu_backend));
    if (!ggml_gallocr_alloc_graph(out.alloc, out.gf)) {
        out.free();
        return false;
    }
    return true;
}

bool build_cached_hot_batched_graph(
    CachedHotBatchedGraph & out,
    ggml_backend_t gpu_backend,
    const MoeHybridLayerStorage & storage,
    const MoeLayerDesc & desc,
    const MoeHybridConfig & cfg,
    int n_tokens) {

    out.free();
    out.n_tokens = n_tokens;

    const int n_embd = cfg.n_embd;
    const int n_used = cfg.n_expert_used;
    const int n_ff_exp = cfg.n_ff_exp;

    ggml_init_params ip{};
    ip.mem_size = 128 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    out.inp = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(out.inp);

    ggml_tensor * routed = nullptr;
    const bool has_hot_stack =
        (storage.gate_up_hot || (storage.gate_hot && storage.up_hot)) &&
        storage.down_hot;
    if (has_hot_stack) {
        out.sel = ggml_new_tensor_2d(out.ctx, GGML_TYPE_I32, n_used, n_tokens);
        ggml_set_input(out.sel);
        out.wts = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_used, n_tokens);
        ggml_set_input(out.wts);
        build_batched_routed_graph(out.ctx,
            storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
            desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
            out.inp, out.sel, out.wts, n_embd, n_ff_exp, n_used, n_tokens,
            cfg.swiglu_clamp, &routed, false, nullptr,
            ggml_backend_is_cuda(gpu_backend));
    }

    // Shared expert (always on GPU)
    ggml_tensor * combined = routed;
    ggml_tensor * shared = build_shared_expert_subgraph(out.ctx, desc, out.inp, cfg.swiglu_clamp);
    if (shared) {
        combined = combined ? ggml_add(out.ctx, combined, shared) : shared;
    }

    if (!combined) { out.free(); return false; }
    out.output = combined;

    out.gf = ggml_new_graph_custom(out.ctx, 4096, false);
    ggml_set_output(out.output);
    ggml_build_forward_expand(out.gf, out.output);
    out.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu_backend));
    if (!ggml_gallocr_alloc_graph(out.alloc, out.gf)) {
        out.free();
        return false;
    }
    return true;
}

// Cached batched COLD routed graph (CPU backend, no shared expert). Mirror of
// build_cached_hot_batched_graph for the cold expert stack; used by the mixed
// batched path so spec-decode verify/replay reuse the graph instead of
// rebuilding it every call.
static bool build_cached_cold_batched_graph(
    CachedHotBatchedGraph & out,
    ggml_backend_t cpu_backend,
    const MoeHybridLayerStorage & storage,
    const MoeLayerDesc & desc,
    const MoeHybridConfig & cfg,
    int n_tokens) {

    out.free();
    out.n_tokens = n_tokens;
    const int n_embd = cfg.n_embd;
    const int n_used = cfg.n_expert_used;
    const int n_ff_exp = cfg.n_ff_exp;

    ggml_init_params ip{};
    ip.mem_size = 128 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    out.inp = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(out.inp);
    out.sel = ggml_new_tensor_2d(out.ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_input(out.sel);
    out.wts = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_used, n_tokens);
    ggml_set_input(out.wts);

    ggml_tensor * routed = nullptr;
    build_batched_routed_graph(out.ctx,
        storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
        desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
        out.inp, out.sel, out.wts, n_embd, n_ff_exp, n_used, n_tokens,
        cfg.swiglu_clamp, &routed, false, nullptr,
        ggml_backend_is_cuda(cpu_backend));
    if (!routed) { out.free(); return false; }
    out.output = routed;

    out.gf = ggml_new_graph_custom(out.ctx, 4096, false);
    ggml_set_output(out.output);
    ggml_build_forward_expand(out.gf, out.output);
    out.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu_backend));
    if (!ggml_gallocr_alloc_graph(out.alloc, out.gf)) {
        out.free();
        return false;
    }
    return true;
}

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
    MoeHybridFfnTelemetry *         telemetry,
    std::string *                   err) {

    if (telemetry) *telemetry = {};
    const auto ffn_wall_t0 = HybridClock::now();
    const auto partition_t0 = HybridClock::now();

    std::vector<int32_t> hot_ids;
    std::vector<float> hot_weights;
    std::vector<int32_t> cold_ids;
    std::vector<float> cold_weights;
    for (int i = 0; i < n_selected; ++i) {
        const int32_t gid = selected_ids[i];
        if (gid < 0 || gid >= (int32_t)storage.hot_local_by_global.size()) {
            if (err) *err = "selected id out of range";
            return false;
        }
        const int32_t hot_local = storage.hot_local_by_global[(size_t)gid];
        if (hot_local >= 0) {
            hot_ids.push_back(hot_local);
            hot_weights.push_back(selected_weights[i]);
            continue;
        }
        const int32_t cold_local = storage.cold_local_by_global[(size_t)gid];
        if (cold_local >= 0) {
            cold_ids.push_back(cold_local);
            cold_weights.push_back(selected_weights[i]);
        }
    }
    const auto partition_t1 = HybridClock::now();
    if (telemetry) {
        telemetry->partition_us = elapsed_us(partition_t0, partition_t1);
        telemetry->hot_selected = (int)hot_ids.size();
        telemetry->cold_selected = (int)cold_ids.size();
    }

    std::vector<float> hot_and_shared, cold;

    const int n_hot = (int)hot_ids.size();
    const bool has_hot = (n_hot > 0);
    const bool has_shared = (desc.ffn_up_shexp && desc.ffn_gate_shexp && desc.ffn_down_shexp);
    const bool has_cold = !cold_ids.empty();
    const int n_cold = (int)cold_ids.size();
    const int fixed_slot_mode = fixed_slot_graphs_mode();
    const int fixed_slot_max_val = fixed_slot_max();
    const int fixed_slot_limit = std::max(1, std::min(
        fixed_slot_max_val > 0 ? fixed_slot_max_val : cfg.n_expert_used, cfg.n_expert_used));
    const int min_adaptive_slots = std::min(cfg.n_expert_used, 3);
    const int n_hot_graph =
        (fixed_slot_mode == 1 && has_hot) ? std::max(n_hot, fixed_slot_limit) :
        (fixed_slot_mode == 2 && has_hot) ? std::max(n_hot, min_adaptive_slots) :
        n_hot;
    const int n_cold_graph =
        (fixed_slot_mode == 1 && has_cold) ? std::max(n_cold, fixed_slot_limit) :
        (fixed_slot_mode == 2 && has_cold) ? std::max(n_cold, min_adaptive_slots) :
        n_cold;
    const size_t graph_cache_size = (size_t)cfg.n_expert_used + 1;
    if (storage.hot_graph_by_width.size() < graph_cache_size) {
        storage.hot_graph_by_width.resize(graph_cache_size);
    }
    if (storage.cold_graph_by_width.size() < graph_cache_size) {
        storage.cold_graph_by_width.resize(graph_cache_size);
    }
    CachedFfnGraph & hot_graph =
        storage.hot_graph_by_width[(size_t)n_hot_graph];
    CachedFfnGraph & cold_graph =
        storage.cold_graph_by_width[(size_t)n_cold_graph];
    ggml_backend_t cold_backend = storage.cold_backend ? storage.cold_backend : cpu_backend;
    const bool cold_on_gpu = has_cold &&
                             storage.cold_backend_kind == MoeHybridColdBackend::Gpu &&
                             cold_backend == gpu_backend;

    // ── Hot + Shared path on GPU ──
    bool hot_async_launched = false;
    const auto hot_t0 = HybridClock::now();
    if (has_hot && !storage.down_hot && !storage.gate_up_hot) {
        if (err) *err = "selected hot experts are not materialized";
        return false;
    }
    if (!has_hot && !has_shared) {
        hot_and_shared.assign((size_t)cfg.n_embd, 0.0f);
    } else {
        std::vector<int32_t> hot_ids_padded;
        std::vector<float> hot_weights_padded;
        const int32_t * hot_ids_data = hot_ids.empty() ? nullptr : hot_ids.data();
        const float * hot_weights_data = hot_weights.empty() ? nullptr : hot_weights.data();
        if (n_hot_graph > n_hot) {
            hot_ids_padded = hot_ids;
            hot_weights_padded = hot_weights;
            const int32_t dummy = hot_ids.empty() ? 0 : hot_ids.front();
            hot_ids_padded.resize((size_t)n_hot_graph, dummy);
            hot_weights_padded.resize((size_t)n_hot_graph, 0.0f);
            hot_ids_data = hot_ids_padded.data();
            hot_weights_data = hot_weights_padded.data();
        }
        // Lazily build cached hot graph on first use
        if (!hot_graph.valid() || hot_graph.n_hot != n_hot_graph) {
            if (telemetry) telemetry->hot_graph_builds++;
            const auto graph_build_t0 = HybridClock::now();
            build_cached_hot_graph(hot_graph, gpu_backend,
                                   storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                                   desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                   desc, cfg.n_embd, cfg.n_ff_exp, n_hot_graph,
                                   CachedHotGraphOptions{cfg.swiglu_clamp});
            if (telemetry) telemetry->hot_graph_build_us += elapsed_us(graph_build_t0, HybridClock::now());
        } else if (telemetry) {
            telemetry->hot_graph_hits++;
        }
        if (hot_graph.valid() && hot_graph.n_hot == n_hot_graph) {
            const auto input_t0 = HybridClock::now();
            ggml_backend_tensor_set(hot_graph.inp, cur_host, 0, sizeof(float) * (size_t)cfg.n_embd);
            if (hot_graph.ids && n_hot_graph > 0) {
                ggml_backend_tensor_set(hot_graph.ids, hot_ids_data, 0, sizeof(int32_t) * (size_t)n_hot_graph);
            }
            if (hot_graph.weights && n_hot_graph > 0) {
                ggml_backend_tensor_set(hot_graph.weights, hot_weights_data, 0, sizeof(float) * (size_t)n_hot_graph);
            }
            if (telemetry) telemetry->hot_input_us += elapsed_us(input_t0, HybridClock::now());
            // Launch GPU async — kernel runs while the cold backend runs.
            const auto compute_t0 = HybridClock::now();
            ggml_backend_graph_compute_async(gpu_backend, hot_graph.gf);
            if (telemetry) telemetry->hot_compute_us += elapsed_us(compute_t0, HybridClock::now());
            hot_async_launched = true;
            if (cold_on_gpu) {
                const auto read_t0 = HybridClock::now();
                ggml_backend_synchronize(gpu_backend);
                if (telemetry) telemetry->hot_read_us += elapsed_us(read_t0, HybridClock::now());
            }
        } else {
            // Fallback: sync compute (no overlap)
            if (!run_hot_and_shared_ffn_gpu(gpu_backend,
                                            storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                                            desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                            desc, cfg.n_embd, cfg.n_ff_exp,
                                            cfg.swiglu_clamp,
                                            cur_host,
                                            hot_ids.empty() ? nullptr : hot_ids.data(),
                                            hot_weights.empty() ? nullptr : hot_weights.data(),
                                            n_hot, hot_and_shared, err)) {
                return false;
            }
        }
    }

    // ── Cold path on selected cold backend ──
    const auto cold_t0 = HybridClock::now();
    if (has_cold) {
        if (!storage.down_cold) {
            if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
            if (err) *err = "selected cold experts are not materialized";
            return false;
        }
        std::vector<int32_t> cold_ids_padded;
        std::vector<float> cold_weights_padded;
        const int32_t * cold_ids_data = cold_ids.data();
        const float * cold_weights_data = cold_weights.data();
        if (n_cold_graph > n_cold) {
            cold_ids_padded = cold_ids;
            cold_weights_padded = cold_weights;
            cold_ids_padded.resize((size_t)n_cold_graph, cold_ids.front());
            cold_weights_padded.resize((size_t)n_cold_graph, 0.0f);
            cold_ids_data = cold_ids_padded.data();
            cold_weights_data = cold_weights_padded.data();
        }
        if (!cold_graph.valid() || cold_graph.n_hot != n_cold_graph) {
            if (telemetry) telemetry->cold_graph_builds++;
            const auto graph_build_t0 = HybridClock::now();
            build_cached_cold_graph(cold_graph, cold_backend,
                                    storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
                                    desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                    cfg.n_embd, cfg.n_ff_exp, n_cold_graph, cfg.swiglu_clamp);
            if (telemetry) telemetry->cold_graph_build_us += elapsed_us(graph_build_t0, HybridClock::now());
        } else if (telemetry) {
            telemetry->cold_graph_hits++;
        }
        if (cold_graph.valid() && cold_graph.n_hot == n_cold_graph) {
            const auto input_t0 = HybridClock::now();
            ggml_backend_tensor_set(cold_graph.inp, cur_host, 0, sizeof(float) * (size_t)cfg.n_embd);
            ggml_backend_tensor_set(cold_graph.ids, cold_ids_data, 0, sizeof(int32_t) * (size_t)n_cold_graph);
            ggml_backend_tensor_set(cold_graph.weights, cold_weights_data, 0, sizeof(float) * (size_t)n_cold_graph);
            if (telemetry) telemetry->cold_input_us += elapsed_us(input_t0, HybridClock::now());
            const auto compute_t0 = HybridClock::now();
            auto st = ggml_backend_graph_compute(cold_backend, cold_graph.gf);
            if (telemetry) telemetry->cold_compute_us += elapsed_us(compute_t0, HybridClock::now());
            if (st != GGML_STATUS_SUCCESS) {
                if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
                if (err) *err = "cached cold graph compute failed";
                return false;
            }
            const auto read_t0 = HybridClock::now();
            cold.resize((size_t)cfg.n_embd);
            ggml_backend_tensor_get(cold_graph.output, cold.data(), 0, sizeof(float) * (size_t)cfg.n_embd);
            if (telemetry) telemetry->cold_read_us += elapsed_us(read_t0, HybridClock::now());
        } else {
            if (!run_routed_subset(cold_backend,
                                   storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
                                   desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                   cfg.n_embd, cfg.n_ff_exp, cfg.swiglu_clamp,
                                   cur_host, cold_ids.data(), cold_weights.data(), n_cold, cold, err)) {
                if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
                return false;
            }
        }
    } else {
        cold.assign((size_t)cfg.n_embd, 0.0f);
    }
    const auto cold_t1 = HybridClock::now();

    // ── Sync GPU and read result ──
    if ((has_hot || has_shared) && hot_graph.valid() && hot_graph.n_hot == n_hot_graph) {
        const auto read_t0 = HybridClock::now();
        ggml_backend_synchronize(gpu_backend);
        hot_and_shared.resize((size_t)cfg.n_embd);
        ggml_backend_tensor_get(hot_graph.output, hot_and_shared.data(), 0, sizeof(float) * (size_t)cfg.n_embd);
        if (telemetry) telemetry->hot_read_us += elapsed_us(read_t0, HybridClock::now());
    }
    const auto hot_t1 = HybridClock::now();

    if (telemetry) {
        telemetry->hot_us = elapsed_us(hot_t0, hot_t1);
        telemetry->cold_us = has_cold ? elapsed_us(cold_t0, cold_t1) : 0;
        telemetry->shared_us = 0;
    }

    const auto combine_t0 = HybridClock::now();
    out.assign((size_t)cfg.n_embd, 0.0f);
    for (int i = 0; i < cfg.n_embd; ++i) {
        out[(size_t)i] = hot_and_shared[(size_t)i] + cold[(size_t)i];
    }
    const auto combine_t1 = HybridClock::now();
    if (telemetry) {
        telemetry->combine_us = elapsed_us(combine_t0, combine_t1);
        telemetry->ffn_wall_us = elapsed_us(ffn_wall_t0, combine_t1);
    }
    return true;
}

bool eval_moe_batched_prefill_ffn(
    ggml_backend_t                  gpu_backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err) {

    const int n_embd = cfg.n_embd;
    const int n_used = cfg.n_expert_used;
    const int n_ff_exp = cfg.n_ff_exp;
    out.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
    if (n_tokens <= 0) return true;

    ggml_init_params ip{};
    ip.mem_size = 128 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        if (err) *err = "ggml_init failed";
        return false;
    }

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp);
    ggml_tensor * sel = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_input(sel);
    ggml_tensor * wts = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_used, n_tokens);
    ggml_set_input(wts);

    // Routed expert computation using full GPU expert tensors
    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, inp, n_embd, 1, n_tokens);
    ggml_tensor * gu = nullptr;
    if (desc.ffn_gate_up_exps) {
        ggml_tensor * gate_up_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, desc.ffn_gate_up_exps, cur_3d, sel), desc.ffn_gate_up_exps_s);
        ggml_tensor * gate_e = ggml_view_3d(ctx, gate_up_e,
            n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
            gate_up_e->nb[1], gate_up_e->nb[2], 0);
        ggml_tensor * up_e = ggml_view_3d(ctx, gate_up_e,
            n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
            gate_up_e->nb[1], gate_up_e->nb[2],
            (size_t)n_ff_exp * ggml_element_size(gate_up_e));
        gate_e = ggml_cont(ctx, gate_e);
        up_e = ggml_cont(ctx, up_e);
        gu = swiglu_maybe_clamped(ctx, gate_e, up_e, cfg.swiglu_clamp);
    } else {
        ggml_tensor * gate_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, desc.ffn_gate_exps, cur_3d, sel), desc.ffn_gate_exps_s);
        ggml_tensor * up_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, desc.ffn_up_exps, cur_3d, sel), desc.ffn_up_exps_s);
        gu = swiglu_maybe_clamped(ctx, gate_e, up_e, cfg.swiglu_clamp);
    }

    ggml_tensor * experts = apply_scale2(ctx,
        ggml_mul_mat_id(ctx, desc.ffn_down_exps, gu, sel), desc.ffn_down_exps_s);

    // Weight and sum over experts
    ggml_tensor * w_view = ggml_reshape_3d(ctx, wts, 1, n_used, n_tokens);
    experts = ggml_mul(ctx, experts, w_view);

    ggml_tensor * sum_shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * moe_sum = ggml_repeat_back(ctx, experts, sum_shape);
    ggml_tensor * routed = ggml_reshape_2d(ctx, moe_sum, n_embd, n_tokens);

    // Shared expert
    ggml_tensor * combined = routed;
    ggml_tensor * shared = build_shared_expert_subgraph(ctx, desc, inp, cfg.swiglu_clamp);
    if (shared) {
        combined = combined ? ggml_add(ctx, combined, shared) : shared;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_set_output(combined);
    ggml_build_forward_expand(gf, combined);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu_backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        if (err) *err = "batched prefill gallocr failed";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(inp, cur_host, 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
    ggml_backend_tensor_set(sel, selected_ids, 0, sizeof(int32_t) * (size_t)n_used * (size_t)n_tokens);
    ggml_backend_tensor_set(wts, selected_weights, 0, sizeof(float) * (size_t)n_used * (size_t)n_tokens);

    auto st = ggml_backend_graph_compute(gpu_backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        if (err) *err = "batched prefill compute failed";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(combined, out.data(), 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

// MMQ full-batch mul_mat_id on a reduced hot stack is only stable for large
// batches. Small batches (spec verify/replay, <=~24 tokens) spread n_used*n_tokens
// slots over thousands of hot experts; that extreme imbalance hits an unbounded
// stream-k tile load in the MMQ kernel and faults (observed on sm_86, not just
// sm_75). Prefill chunks (>=64 tokens) are dense enough and run clean, so keep
// the sm_80+ fast path for them and route small batches through the proven
// <=4-token MMVQ sub-batch path.
static bool mmq_full_batch_ok(const MoeHybridConfig & cfg, int n_tokens) {
    static const int min_tokens = [](){
        const char * v = std::getenv("DFLASH_MMQ_FULL_BATCH_MIN");
        return v ? std::atoi(v) : 64;
    }();
    return cfg.mmq_safe_full_batch && n_tokens >= min_tokens;
}

// Sub-batch size for the reduced-hot-stack routed mul_mat_id. The MMQ path
// (n_tokens > 8) illegal-accesses on a REDUCED expert stack for sparse/
// imbalanced sub-64 batches (a genuine ggml-cuda MMQ mul_mat_id bug, observed
// on sm_86 + gfx1151); the MMVQ-mmid path is stable. Q4_K MMVQ-mmid handles up
// to 8 tokens on CUDA sm_80+ (MMVQ_MAX_BATCH_SIZE) and 4 on AMD. Earlier this
// had to be 1 because the F32 shared-expert gate (cublasSgemm, M=1) also faulted
// at N>1 on the shipped CUDA 12.0 cublasLt; that is now computed cublas-free
// (mul + sum_rows), so sub-batch=8 is safe and validated on sm_86. Default to 8
// on sm_80+ (CUDA), 1 elsewhere (proven single-token path on unvalidated archs);
// env override tunes per arch without a rebuild.
static int mmq_safe_sub_batch() {
    static const int v = [](){
        const char * e = std::getenv("DFLASH_MMQ_SUB_BATCH");
        if (e) return std::max(1, std::atoi(e));
        return (query_gpu_compute_sm() >= 80) ? 8 : 1;
    }();
    return v;
}

int moe_hybrid_expert_compute_batch_limit() {
    static const int value = []() {
        const int requested = env_int_or_default("DFLASH_MOE_EXPERT_COMPUTE_BATCH", 32);
        const int max_batch = moe_expert_compute_batch_max();
        const int effective = std::min(requested, max_batch);
        if (effective < requested) {
            std::fprintf(stderr,
                         "[hybrid-ffn] clamped MoE expert compute batch=%d to %d; "
                         "set DFLASH_MOE_EXPERT_COMPUTE_BATCH_MAX to override\n",
                         requested, effective);
        }
        return effective;
    }();
    return value;
}

int moe_hybrid_expert_compute_ipc_batch_limit(int n_tokens) {
    if (n_tokens <= 0) return 1;
    const int requested = parse_moe_expert_compute_ipc_mode() == MoeExpertComputeIpcMode::Batched
        ? env_int_or_default("DFLASH_MOE_EXPERT_COMPUTE_IPC_BATCH_CAPACITY", 1024)
        : moe_hybrid_expert_compute_batch_limit();
    return std::min(std::max(1, std::min(requested, 4096)), n_tokens);
}

int moe_hybrid_prefill_hot_sub_batch_limit() {
    const char * raw = std::getenv("DFLASH_MOE_PREFILL_HOT_SUB_BATCH");
    int requested = 4;
    if (raw && *raw) {
        char * end = nullptr;
        const long value = std::strtol(raw, &end, 10);
        if (end != raw && *end == '\0') {
            requested = value > 4096 ? 4096 : (int)value;
        }
    }
    if (requested <= 0) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr,
                         "[hybrid-ffn] ignoring MoE prefill hot sub-batch=%d; "
                         "using safe default 4\n",
                         requested);
            warned = true;
        }
        return 4;
    }
    if (requested > 4) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr,
                         "[hybrid-ffn] clamped MoE prefill hot sub-batch=%d to 4\n",
                         requested);
            warned = true;
        }
        return 4;
    }
    return requested;
}

static bool eval_moe_hybrid_remote_cold_batched(
    const MoeHybridConfig &         cfg,
    const MoeHybridLayerStorage &   storage,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err,
    MoeExpertCompute *              expert_compute,
    const MoeExpertLayer *          expert_layer);

static bool eval_moe_hybrid_ffn_batched_core(
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
    std::string *                   err,
    ggml_gallocr_t *                p_hot_alloc,
    ggml_gallocr_t *                p_cold_alloc,
    MoeExpertCompute *                expert_compute,
    const MoeExpertLayer *            expert_layer,
    MoeHybridFfnTelemetry *         telemetry,
    bool                            skip_cold = false,
    bool                            skip_hot = false,
    ggml_tensor *                   cur_backend = nullptr,
    ggml_tensor *                   device_output = nullptr,
    ggml_backend_t                  device_output_owner = nullptr) {

    const auto ffn_wall_t0 = HybridClock::now();
    const auto partition_t0 = HybridClock::now();
    const int n_embd = cfg.n_embd;
    const int n_used = cfg.n_expert_used;
    const int n_ff_exp = cfg.n_ff_exp;
    out.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
    if (n_tokens <= 0) return true;
    if (!cur_host && !cur_backend) {
        if (err) *err = "hybrid batched activation is unavailable";
        return false;
    }
    if (device_output && (!skip_hot || !device_output_owner ||
                          device_output->type != GGML_TYPE_F32 ||
                          device_output->ne[0] != n_embd ||
                          device_output->ne[1] != n_tokens)) {
        if (err) *err = "invalid cold-owner device output";
        return false;
    }
    auto set_cur_input = [&](ggml_tensor * dst,
                             ggml_backend_t dst_backend) {
        if (cur_backend) {
            // Use the backend-aware copy path. On heterogeneous HIP this
            // submits on the producer stream and publishes an event to the
            // consumer stream; the generic buffer copy uses a per-thread
            // stream with no cross-device dependency and can leave the Strix
            // destination containing zeros.
            ggml_backend_tensor_copy_async(
                gpu_backend, dst_backend, cur_backend, dst);
        } else {
            ggml_backend_tensor_set(
                dst, cur_host, 0,
                sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
        }
    };

    // ── Fast path: cached hot+cold batched graphs (spec-decode verify/replay) ──
    // Mixed layers used to rebuild+free their hot and cold ggml graphs on every
    // call; that graph churn (not the matmul) dominated the verify FFN time.
    // Reuse per-n_tokens cached graphs so steady-state rebuilds nothing. Large
    // prefill batches (n_tokens >= kMaxBatchedCache) fall through to the inline
    // path below.
    if (n_tokens > 0 && n_tokens < MoeHybridLayerStorage::kMaxBatchedCache) {
        const int total_slots = n_used * n_tokens;
        const int n_cold_stack = std::max(1, (int)(storage.down_cold ? storage.down_cold->ne[2] : 1));
        std::vector<int32_t> hot_sel(total_slots);
        std::vector<float>   hot_wts(total_slots, 0.0f);
        std::vector<int32_t> cold_sel(total_slots);
        std::vector<float>   cold_wts(total_slots, 0.0f);
        // Dummy (unused) slots must point at INITIALIZED pinned experts, not
        // uninitialized cache-ring spare slots (hot_active..n_hot_stack), whose
        // garbage Q4_K scale bits could dequantize to NaN (x weight 0 = NaN).
        const int n_hot_init = std::max(1, storage.hot_active);
        for (int i = 0; i < total_slots; ++i) { hot_sel[i] = i % n_hot_init; cold_sel[i] = i % n_cold_stack; }
        bool fp_has_cold = false;
        for (int i = 0; i < total_slots; ++i) {
            const int32_t gid = selected_ids[i];
            if (gid < 0 || gid >= (int32_t)storage.hot_local_by_global.size()) continue;
            const int32_t hl = storage.hot_local_by_global[(size_t)gid];
            if (hl >= 0) { hot_sel[i] = hl; hot_wts[i] = selected_weights[i]; }
            else {
                const int32_t cl = storage.cold_local_by_global[(size_t)gid];
                if (cl >= 0) { cold_sel[i] = cl; cold_wts[i] = selected_weights[i]; fp_has_cold = true; }
            }
        }

        CachedHotBatchedGraph & hg = storage.hot_batched_mixed[n_tokens];
        const bool hg_ok = (hg.valid() && hg.n_tokens == n_tokens)
            || build_cached_hot_batched_graph(hg, gpu_backend, storage, desc, cfg, n_tokens);
        const bool remote_cold = !skip_cold && fp_has_cold &&
            expert_compute && expert_layer;
        CachedHotBatchedGraph * cg = nullptr;
        bool cg_ok = true;
        if (!skip_cold && fp_has_cold && !remote_cold) {
            cg = &storage.cold_batched_mixed[n_tokens];
            ggml_backend_t cached_cold_backend =
                storage.cold_backend ? storage.cold_backend : cpu_backend;
            cg_ok = (cg->valid() && cg->n_tokens == n_tokens)
                || build_cached_cold_batched_graph(
                    *cg, cached_cold_backend, storage, desc, cfg, n_tokens);
        }

        if (hg_ok && cg_ok) {
            // Hot (GPU, async): shared expert + routed hot (zero-weight dummy slots
            // keep an all-cold batch's shared-expert contribution).
            set_cur_input(hg.inp, gpu_backend);
            if (hg.sel) {
                ggml_backend_tensor_set(hg.sel, hot_sel.data(), 0,
                                        sizeof(int32_t) * (size_t)total_slots);
            }
            if (hg.wts) {
                ggml_backend_tensor_set(hg.wts, hot_wts.data(), 0,
                                        sizeof(float) * (size_t)total_slots);
            }
            ggml_backend_graph_compute_async(gpu_backend, hg.gf);

            std::vector<float> cold_partial;
            if (remote_cold) {
                if (!cur_host) {
                    ggml_backend_synchronize(gpu_backend);
                    if (err) *err = "remote cold prefill requires a host activation";
                    return false;
                }
                if (!eval_moe_hybrid_remote_cold_batched(
                        cfg, storage, cur_host, selected_ids, selected_weights,
                        n_tokens, cold_partial, err,
                        expert_compute, expert_layer)) {
                    ggml_backend_synchronize(gpu_backend);
                    return false;
                }
            } else if (cg) {
                cold_partial.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
                ggml_backend_t cached_cold_backend =
                    storage.cold_backend ? storage.cold_backend : cpu_backend;
                set_cur_input(cg->inp, cached_cold_backend);
                ggml_backend_tensor_set(cg->sel, cold_sel.data(), 0, sizeof(int32_t) * (size_t)total_slots);
                ggml_backend_tensor_set(cg->wts, cold_wts.data(), 0, sizeof(float) * (size_t)total_slots);
                if (ggml_backend_graph_compute(
                        cached_cold_backend, cg->gf) != GGML_STATUS_SUCCESS) {
                    ggml_backend_synchronize(gpu_backend);
                    if (err) *err = "batched cold cached compute failed";
                    return false;
                }
                ggml_backend_tensor_get(cg->output, cold_partial.data(), 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
            }

            ggml_backend_synchronize(gpu_backend);
            ggml_backend_tensor_get(hg.output, out.data(), 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
            if (remote_cold || cg) {
                const size_t ntot = (size_t)n_embd * (size_t)n_tokens;
                for (size_t i = 0; i < ntot; ++i) out[i] += cold_partial[i];
            }
            return true;
        }
        // build failed -> fall through to the inline rebuild path
    }

    // ── Step 1: Partition routing into hot and cold ──
    // Dummy slots use weight 0.0 and are distributed evenly across all experts
    // to avoid pathological routing imbalance that triggers OOB in MMQ stream-k.
    const int total_slots = n_used * n_tokens;
    std::vector<int32_t> hot_sel(total_slots);
    // Dummy slots -> pinned (initialized) experts only; see note above.
    const int n_hot_init = std::max(1, storage.hot_active);
    for (int i = 0; i < total_slots; ++i) hot_sel[i] = i % n_hot_init;
    std::vector<float>   hot_wts(total_slots, 0.0f);
    const int n_cold_stack =
        std::max(1, (int)(storage.down_cold ? storage.down_cold->ne[2] : 1));
    // Owner-only prefill must not turn routes assigned to the other GPU into
    // valid zero-weight expert IDs. MUL_MAT_ID executes gate/up/down before
    // route weighting, so those dummies duplicate all nominally offloaded work.
    // Negative IDs are compacted by the ROCm MMQ sorter; fused combine avoids
    // reading their intentionally unwritten outputs.
    const bool mask_skipped_cold =
        skip_hot && storage.cold_backend_kind == MoeHybridColdBackend::Gpu &&
        prefill_masked_cold_routes_enabled();
    std::vector<int32_t> cold_sel(total_slots);
    for (int i = 0; i < total_slots; ++i) {
        cold_sel[i] = mask_skipped_cold ? -1 : i % n_cold_stack;
    }
    std::vector<float>   cold_wts(total_slots, 0.0f);
    bool has_hot = false, has_cold = false;
    int hot_selected = 0;
    int cold_selected = 0;
    std::vector<int32_t> cold_rows_per_expert(
        mask_skipped_cold ? (size_t)n_cold_stack : 0, 0);

    for (int i = 0; i < total_slots; ++i) {
        const int32_t gid = selected_ids[i];
        if (gid < 0 || gid >= (int32_t)storage.hot_local_by_global.size()) continue;
        const int32_t hot_lid = storage.hot_local_by_global[(size_t)gid];
        if (hot_lid >= 0) {
            hot_sel[i] = hot_lid;
            hot_wts[i] = selected_weights[i];
            has_hot = true;
            hot_selected++;
        } else {
            const int32_t cold_lid = storage.cold_local_by_global[(size_t)gid];
            if (cold_lid >= 0) {
                cold_sel[i] = cold_lid;
                cold_wts[i] = selected_weights[i];
                has_cold = true;
                cold_selected++;
                if (!cold_rows_per_expert.empty()) {
                    cold_rows_per_expert[(size_t)cold_lid]++;
                }
            }
        }
    }
    int32_t max_cold_expert_rows = 0;
    for (int32_t count : cold_rows_per_expert) {
        max_cold_expert_rows = std::max(max_cold_expert_rows, count);
    }
    MmidOwnerGridHint cold_grid_hint{
        MMID_OWNER_GRID_HINT_MAGIC, max_cold_expert_rows, cold_selected};
    const auto partition_t1 = HybridClock::now();
    if (telemetry) {
        telemetry->partition_us += elapsed_us(partition_t0, partition_t1);
        telemetry->hot_selected += hot_selected;
        telemetry->cold_selected += cold_selected;
    }
    if (mask_skipped_cold) {
        static bool logged = false;
        if (!logged) {
            std::fprintf(stderr,
                         "[hybrid-ffn] masked cold prefill routes active; "
                         "Strix skips R9700-owned expert GEMMs "
                         "max_expert_rows=%d\n",
                         (int)max_cold_expert_rows);
            logged = true;
        }
    }

    ggml_backend_t cold_backend = storage.cold_backend ? storage.cold_backend : cpu_backend;
    const bool cold_on_gpu = has_cold &&
                             storage.cold_backend_kind == MoeHybridColdBackend::Gpu &&
                             cold_backend == gpu_backend;

    // ── Step 2: Build and run hot GPU graph (includes shared expert always) ──
    std::vector<float> hot_partial((size_t)n_embd * (size_t)n_tokens, 0.0f);
    bool hot_async_launched = false;
    const auto hot_t0 = HybridClock::now();

    ggml_context * hot_ctx = nullptr;
    ggml_cgraph * hot_gf = nullptr;
    ggml_gallocr_t hot_alloc = nullptr;
    ggml_tensor * hot_output = nullptr;

    const bool has_shared = (desc.ffn_up_shexp && desc.ffn_gate_shexp && desc.ffn_down_shexp);
    if (!skip_hot && (has_hot || has_shared)) {
        ggml_init_params ip{};
        ip.mem_size = 128 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        hot_ctx = ggml_init(ip);
        if (!hot_ctx) { if (err) *err = "hot ggml_init failed"; return false; }

        ggml_tensor * inp = ggml_new_tensor_2d(hot_ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp);

        ggml_tensor * sel = nullptr;
        ggml_tensor * wts = nullptr;
        ggml_tensor * routed = nullptr;
        if (has_hot) {
            sel = ggml_new_tensor_2d(hot_ctx, GGML_TYPE_I32, n_used, n_tokens);
            ggml_set_input(sel);
            wts = ggml_new_tensor_2d(hot_ctx, GGML_TYPE_F32, n_used, n_tokens);
            ggml_set_input(wts);

            build_batched_routed_graph(hot_ctx,
                storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                inp, sel, wts, n_embd, n_ff_exp, n_used, n_tokens,
                cfg.swiglu_clamp, &routed, false, nullptr,
                ggml_backend_is_cuda(gpu_backend));
        }

        // Shared expert (always on GPU)
        ggml_tensor * combined = routed;
        ggml_tensor * shared = build_shared_expert_subgraph(hot_ctx, desc, inp, cfg.swiglu_clamp);
        if (shared) {
            combined = combined ? ggml_add(hot_ctx, combined, shared) : shared;
        }
        hot_output = combined;

        hot_gf = ggml_new_graph_custom(hot_ctx, 4096, false);
        ggml_set_output(hot_output);
        ggml_build_forward_expand(hot_gf, hot_output);
        if (p_hot_alloc) {
            if (!*p_hot_alloc)
                *p_hot_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu_backend));
            hot_alloc = *p_hot_alloc;
        } else {
            hot_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu_backend));
        }
        if (!ggml_gallocr_alloc_graph(hot_alloc, hot_gf)) {
            if (err) *err = "hybrid batched hot gallocr failed";
            if (!p_hot_alloc) ggml_gallocr_free(hot_alloc);
            ggml_free(hot_ctx);
            return false;
        }

        set_cur_input(inp, gpu_backend);
        if (has_hot) {
            ggml_backend_tensor_set(sel, hot_sel.data(), 0, sizeof(int32_t) * (size_t)total_slots);
            ggml_backend_tensor_set(wts, hot_wts.data(), 0, sizeof(float) * (size_t)total_slots);
        }

        // Launch GPU async
        ggml_backend_graph_compute_async(gpu_backend, hot_gf);
        hot_async_launched = true;
        if (cold_on_gpu) {
            ggml_backend_synchronize(gpu_backend);
        }
    }

    // ── Step 3: Build and run cold graph on selected backend ──
    std::vector<float> cold_partial((size_t)n_embd * (size_t)n_tokens, 0.0f);
    const auto cold_t0 = HybridClock::now();

    if (has_cold && skip_cold) {
        // Used by reduced-stack prefill: hot/shared must be sliced for MMVQ
        // stability, but remote cold expert compute can still run as one
        // larger batched IPC request for the full prefill chunk.
    } else if (has_cold && expert_compute && expert_layer) {
        if (!cur_host) {
            if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
            if (!p_hot_alloc && hot_alloc) ggml_gallocr_free(hot_alloc);
            if (hot_ctx) ggml_free(hot_ctx);
            if (err) *err = "remote cold prefill requires a host activation";
            return false;
        }
        if (!eval_moe_hybrid_remote_cold_batched(
                cfg, storage, cur_host, selected_ids, selected_weights,
                n_tokens, cold_partial, err, expert_compute, expert_layer)) {
            if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
            if (!p_hot_alloc && hot_alloc) ggml_gallocr_free(hot_alloc);
            if (hot_ctx) ggml_free(hot_ctx);
            return false;
        }
        if (telemetry) telemetry->cold_us += elapsed_us(cold_t0, HybridClock::now());
    } else if (has_cold) {
        if (!storage.down_cold) {
            if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
            if (!p_hot_alloc && hot_alloc) ggml_gallocr_free(hot_alloc);
            if (hot_ctx) ggml_free(hot_ctx);
            if (err) *err = "selected cold experts are not materialized";
            return false;
        }
        ggml_init_params ip{};
        ip.mem_size = 128 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        ggml_context * cold_ctx = ggml_init(ip);
        if (!cold_ctx) {
            if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
            if (hot_alloc) ggml_gallocr_free(hot_alloc);
            if (hot_ctx) ggml_free(hot_ctx);
            if (err) *err = "cold ggml_init failed";
            return false;
        }

        ggml_tensor * inp = ggml_new_tensor_2d(
            cold_ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp);
        ggml_tensor * sel = ggml_new_tensor_2d(cold_ctx, GGML_TYPE_I32, n_used, n_tokens);
        ggml_set_input(sel);
        if (mask_skipped_cold && max_cold_expert_rows > 0 &&
            prefill_compact_cold_grid_enabled()) {
            sel->extra = &cold_grid_hint;
        }
        ggml_tensor * wts = ggml_new_tensor_2d(cold_ctx, GGML_TYPE_F32, n_used, n_tokens);
        ggml_set_input(wts);

        ggml_tensor * cold_routed = nullptr;
        build_batched_routed_graph(cold_ctx,
            storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
            desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
            inp, sel, wts, n_embd, n_ff_exp, n_used, n_tokens,
            cfg.swiglu_clamp, &cold_routed, false, nullptr,
            ggml_backend_is_cuda(cold_backend),
            /*force_fused_combine=*/mask_skipped_cold);

        ggml_cgraph * cold_gf = ggml_new_graph_custom(cold_ctx, 4096, false);
        ggml_set_output(cold_routed);
        ggml_build_forward_expand(cold_gf, cold_routed);
        ggml_gallocr_t cold_alloc;
        if (p_cold_alloc) {
            if (!*p_cold_alloc)
                *p_cold_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cold_backend));
            cold_alloc = *p_cold_alloc;
        } else {
            cold_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cold_backend));
        }
        if (!ggml_gallocr_alloc_graph(cold_alloc, cold_gf)) {
            if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
            if (!p_hot_alloc && hot_alloc) ggml_gallocr_free(hot_alloc);
            if (hot_ctx) ggml_free(hot_ctx);
            if (!p_cold_alloc) ggml_gallocr_free(cold_alloc);
            ggml_free(cold_ctx);
            if (err) *err = "hybrid batched cold gallocr failed";
            return false;
        }

        set_cur_input(inp, cold_backend);
        ggml_backend_tensor_set(sel, cold_sel.data(), 0, sizeof(int32_t) * (size_t)total_slots);
        ggml_backend_tensor_set(wts, cold_wts.data(), 0, sizeof(float) * (size_t)total_slots);

        auto st = ggml_backend_graph_compute(cold_backend, cold_gf);
        if (st != GGML_STATUS_SUCCESS) {
            if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
            if (!p_hot_alloc && hot_alloc) ggml_gallocr_free(hot_alloc);
            if (hot_ctx) ggml_free(hot_ctx);
            if (!p_cold_alloc) ggml_gallocr_free(cold_alloc);
            ggml_free(cold_ctx);
            if (err) *err = "hybrid batched cold compute failed";
            return false;
        }

        if (device_output) {
            ggml_backend_tensor_copy_async(
                cold_backend, device_output_owner,
                cold_routed, device_output);
            // The cold graph uses a reusable gallocr arena. Keep its output
            // storage live until the peer transfer has completed.
            ggml_backend_synchronize(device_output_owner);
        } else {
            ggml_backend_tensor_get(cold_routed, cold_partial.data(), 0,
                sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
        }
        if (telemetry) telemetry->cold_us += elapsed_us(cold_t0, HybridClock::now());
        if (!p_cold_alloc) ggml_gallocr_free(cold_alloc);
        ggml_free(cold_ctx);
    }

    // ── Step 4: Sync GPU and read hot result ──
    if (hot_async_launched) {
        ggml_backend_synchronize(gpu_backend);
        ggml_backend_tensor_get(hot_output, hot_partial.data(), 0,
            sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
    }
    if (telemetry) telemetry->hot_us += elapsed_us(hot_t0, HybridClock::now());
    if (!p_hot_alloc && hot_alloc) ggml_gallocr_free(hot_alloc);
    if (hot_ctx) ggml_free(hot_ctx);

    if (device_output) {
        if (telemetry) {
            telemetry->ffn_wall_us +=
                elapsed_us(ffn_wall_t0, HybridClock::now());
        }
        return true;
    }

    // ── Step 5: Merge hot + cold ──
    const auto combine_t0 = HybridClock::now();
    const size_t total_floats = (size_t)n_embd * (size_t)n_tokens;
    for (size_t i = 0; i < total_floats; ++i) {
        out[i] = hot_partial[i] + cold_partial[i];
    }
    if (telemetry) {
        const auto end = HybridClock::now();
        telemetry->combine_us += elapsed_us(combine_t0, end);
        telemetry->ffn_wall_us += elapsed_us(ffn_wall_t0, end);
    }

    return true;
}

static bool eval_moe_hybrid_remote_cold_batched(
    const MoeHybridConfig &         cfg,
    const MoeHybridLayerStorage &   storage,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err,
    MoeExpertCompute *              expert_compute,
    const MoeExpertLayer *          expert_layer) {

    const int n_embd = cfg.n_embd;
    const int n_used = cfg.n_expert_used;
    const int n_ff_exp = cfg.n_ff_exp;
    out.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
    if (n_tokens <= 0) return true;
    if (!expert_compute || !expert_layer) return true;

    std::vector<int> cold_counts((size_t)n_tokens, 0);
    std::vector<int32_t> cold_sel((size_t)n_tokens * (size_t)n_used, 0);
    std::vector<float> cold_wts((size_t)n_tokens * (size_t)n_used, 0.0f);
    int max_cold_selected = 0;

    for (int t = 0; t < n_tokens; ++t) {
        int count = 0;
        for (int i = 0; i < n_used; ++i) {
            const size_t src = (size_t)t * (size_t)n_used + (size_t)i;
            const int32_t gid = selected_ids[src];
            if (gid < 0 || gid >= (int32_t)storage.hot_local_by_global.size()) {
                continue;
            }
            if (storage.hot_local_by_global[(size_t)gid] >= 0) {
                continue;
            }
            const int32_t cold_lid = storage.cold_local_by_global[(size_t)gid];
            if (cold_lid >= 0) {
                if (selected_weights[src] == 0.0f) {
                    continue;
                }
                const size_t dst = (size_t)t * (size_t)n_used + (size_t)count;
                cold_sel[dst] = cold_lid;
                cold_wts[dst] = selected_weights[src];
                count++;
            }
        }
        cold_counts[(size_t)t] = count;
        max_cold_selected = std::max(max_cold_selected, count);
    }

    if (max_cold_selected == 0) return true;
    if (expert_layer->cold_global_by_local.empty()) {
        if (err) *err = "hybrid batched remote cold layer has no cold experts";
        return false;
    }

    if (!expert_compute->compute_batch_ragged(
            *expert_layer, cur_host, cold_sel.data(), cold_wts.data(),
            cold_counts.data(), n_tokens, n_used, n_embd, n_ff_exp,
            out.data())) {
        if (err) *err = "hybrid ragged remote cold compute failed";
        return false;
    }
    return true;
}

// Long heterogeneous prefill has enough rows per expert to use ordinary
// dense matmuls efficiently.  Packing routes by owner-local expert avoids the
// reduced-stack MUL_MAT_ID stream-k path (which is both slow for a 24-expert
// stack and unstable for very large batches) and lets each expert's weights be
// reused across all of its prompt rows.
static bool expert_major_prefill_enabled(int n_tokens) {
    static const bool enabled = []() {
        const char * raw = std::getenv("DFLASH_MOE_EXPERT_MAJOR_PREFILL");
        return !raw || !*raw || std::strcmp(raw, "0") != 0;
    }();
    static const int min_tokens =
        env_int_or_default("DFLASH_MOE_EXPERT_MAJOR_MIN_TOKENS", 512);
    return enabled && n_tokens >= min_tokens;
}

// Expert-major prefill groups prompt rows by expert so the quantized GEMMs can
// reuse each expert's weights.  Keep the inverse permutation and route weights
// on the owner GPU as well: gathering the packed rows back to route-major order
// and reducing there avoids reading every per-route hidden vector to the host,
// doing an O(n_tokens * n_used * n_embd) CPU scatter, and uploading the reduced
// result again.  The old host reduction remains as an emergency A/B fallback.
static bool expert_major_gpu_reduce_enabled() {
    static const bool enabled = []() {
        const char * raw = std::getenv("DFLASH_MOE_EXPERT_MAJOR_GPU_REDUCE");
        return !raw || !*raw || std::strcmp(raw, "0") != 0;
    }();
    return enabled;
}

static bool full_cold_expert_major_enabled() {
    static const bool enabled = []() {
        const char * raw =
            std::getenv("DFLASH_MOE_FULL_COLD_EXPERT_MAJOR");
        return raw && *raw && std::strcmp(raw, "0") != 0;
    }();
    return enabled;
}

static bool full_cold_packed_mmid_enabled() {
    static const bool enabled = []() {
        const char * raw =
            std::getenv("DFLASH_MOE_FULL_COLD_PACKED_MMID");
        return raw && *raw && std::strcmp(raw, "0") != 0;
    }();
    return enabled;
}

static bool eval_moe_owner_expert_major_batched(
    ggml_backend_t                  backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    ggml_tensor *                   gate_tensor,
    ggml_tensor *                   up_tensor,
    ggml_tensor *                   down_tensor,
    ggml_tensor *                   gate_up_tensor,
    const std::vector<int32_t> &     local_by_global,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    bool                            include_shared,
    std::vector<float> &            out,
    std::string *                   err,
    ggml_tensor *                   cur_backend = nullptr,
    ggml_backend_t                  cur_backend_owner = nullptr,
    ggml_tensor *                   device_output = nullptr,
    ggml_backend_t                  device_output_owner = nullptr) {
    const int n_embd = cfg.n_embd;
    const int n_used = cfg.n_expert_used;
    const int n_ff = cfg.n_ff_exp;
    out.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
    if (!backend || (!cur_host && !cur_backend) ||
        !selected_ids || !selected_weights ||
        n_tokens <= 0 || n_embd <= 0 || n_used <= 0 || n_ff <= 0 ||
        !down_tensor || (!gate_up_tensor && (!gate_tensor || !up_tensor))) {
        if (err) *err = "invalid expert-major owner inputs";
        return false;
    }

    ggml_tensor * stack_ref = gate_up_tensor ? gate_up_tensor : gate_tensor;
    const int n_stack = (int)stack_ref->ne[2];
    if (n_stack <= 0 || down_tensor->ne[2] != n_stack) {
        if (err) *err = "invalid expert-major owner stack";
        return false;
    }

    std::vector<int> counts((size_t)n_stack, 0);
    size_t n_pairs = 0;
    for (int t = 0; t < n_tokens; ++t) {
        for (int slot = 0; slot < n_used; ++slot) {
            const size_t route = (size_t)t * (size_t)n_used + (size_t)slot;
            const int32_t global = selected_ids[route];
            if (global < 0 || (size_t)global >= local_by_global.size() ||
                selected_weights[route] == 0.0f) {
                continue;
            }
            const int32_t local = local_by_global[(size_t)global];
            if (local >= 0 && local < n_stack) {
                counts[(size_t)local]++;
                n_pairs++;
            }
        }
    }

    const bool has_shared = include_shared && desc.ffn_gate_shexp &&
                            desc.ffn_up_shexp && desc.ffn_down_shexp;
    if (n_pairs == 0 && !has_shared) return true;

    std::vector<size_t> offsets((size_t)n_stack + 1, 0);
    for (int e = 0; e < n_stack; ++e) {
        offsets[(size_t)e + 1] = offsets[(size_t)e] +
                                  (size_t)counts[(size_t)e];
    }
    std::vector<size_t> cursor(offsets.begin(), offsets.end() - 1);
    // Device input keeps the normalized activation on the R9700 and gathers
    // the owner rows on-device. The host fallback remains for decode and A/B.
    std::vector<float> packed_input;
    if (!cur_backend) {
        packed_input.resize(n_pairs * (size_t)n_embd);
    }
    std::vector<int32_t> packed_tokens(n_pairs);
    std::vector<float> packed_weights(n_pairs);
    std::vector<int32_t> packed_routes(n_pairs);
    std::vector<int32_t> route_to_packed(
        (size_t)n_tokens * (size_t)n_used, 0);
    std::vector<float> owner_weights(
        (size_t)n_tokens * (size_t)n_used, 0.0f);
    for (int t = 0; t < n_tokens; ++t) {
        for (int slot = 0; slot < n_used; ++slot) {
            const size_t route = (size_t)t * (size_t)n_used + (size_t)slot;
            const int32_t global = selected_ids[route];
            if (global < 0 || (size_t)global >= local_by_global.size() ||
                selected_weights[route] == 0.0f) {
                continue;
            }
            const int32_t local = local_by_global[(size_t)global];
            if (local < 0 || local >= n_stack) continue;
            const size_t packed = cursor[(size_t)local]++;
            if (!cur_backend) {
                std::memcpy(packed_input.data() + packed * (size_t)n_embd,
                            cur_host + (size_t)t * (size_t)n_embd,
                            sizeof(float) * (size_t)n_embd);
            }
            packed_tokens[packed] = t;
            packed_weights[packed] = selected_weights[route];
            packed_routes[packed] = (int32_t)route;
            route_to_packed[route] = (int32_t)packed;
            owner_weights[route] = selected_weights[route];
        }
    }

    const bool gpu_reduce = expert_major_gpu_reduce_enabled() && n_pairs > 0;

    ggml_init_params ip{};
    ip.mem_size = 128 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        if (err) *err = "expert-major ggml_init failed";
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
    if (!gf) {
        ggml_free(ctx);
        if (err) *err = "expert-major graph allocation failed";
        return false;
    }

    ggml_tensor * packed_in = nullptr;
    ggml_tensor * packed_out = nullptr;
    ggml_tensor * owner_input = nullptr;
    ggml_tensor * packed_token_ids = nullptr;
    ggml_tensor * inverse_routes = nullptr;
    ggml_tensor * route_weights = nullptr;
    if (cur_backend && (n_pairs > 0 || has_shared)) {
        owner_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                         n_embd, n_tokens);
        ggml_set_input(owner_input);
    }
    if (n_pairs > 0) {
        if (owner_input) {
            packed_token_ids = ggml_new_tensor_1d(
                ctx, GGML_TYPE_I32, (int64_t)n_pairs);
            ggml_set_input(packed_token_ids);
            packed_in = ggml_get_rows(ctx, owner_input, packed_token_ids);
        } else {
            packed_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                           n_embd, (int64_t)n_pairs);
            ggml_set_input(packed_in);
        }
        packed_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                        n_embd, (int64_t)n_pairs);
        ggml_set_output(packed_out);
        if (gpu_reduce) {
            inverse_routes = ggml_new_tensor_2d(
                ctx, GGML_TYPE_I32, n_used, n_tokens);
            route_weights = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, n_used, n_tokens);
            ggml_set_input(inverse_routes);
            ggml_set_input(route_weights);
        }
    }

    auto expert_view = [&](ggml_tensor * stack, int local) {
        return ggml_view_2d(ctx, stack, stack->ne[0], stack->ne[1],
                            stack->nb[1], (size_t)local * stack->nb[2]);
    };

    for (int local = 0; local < n_stack; ++local) {
        const int count = counts[(size_t)local];
        if (count == 0) continue;
        const size_t begin = offsets[(size_t)local];
        ggml_tensor * expert_in = ggml_view_2d(
            ctx, packed_in, n_embd, count, packed_in->nb[1],
            begin * packed_in->nb[1]);
        ggml_tensor * mid = nullptr;
        if (gate_up_tensor) {
            ggml_tensor * gate_up = apply_scale2(
                ctx, ggml_mul_mat(ctx, expert_view(gate_up_tensor, local),
                                  expert_in),
                desc.ffn_gate_up_exps_s);
            ggml_tensor * gate = ggml_view_2d(
                ctx, gate_up, n_ff, count, gate_up->nb[1], 0);
            ggml_tensor * up = ggml_view_2d(
                ctx, gate_up, n_ff, count, gate_up->nb[1],
                (size_t)n_ff * ggml_element_size(gate_up));
            gate = ggml_cont(ctx, gate);
            up = ggml_cont(ctx, up);
            mid = swiglu_maybe_clamped(ctx, gate, up, cfg.swiglu_clamp);
        } else {
            ggml_tensor * gate = apply_scale2(
                ctx, ggml_mul_mat(ctx, expert_view(gate_tensor, local),
                                  expert_in),
                desc.ffn_gate_exps_s);
            ggml_tensor * up = apply_scale2(
                ctx, ggml_mul_mat(ctx, expert_view(up_tensor, local),
                                  expert_in),
                desc.ffn_up_exps_s);
            mid = swiglu_maybe_clamped(ctx, gate, up, cfg.swiglu_clamp);
        }
        ggml_tensor * expert_out = apply_scale2(
            ctx, ggml_mul_mat(ctx, expert_view(down_tensor, local), mid),
            desc.ffn_down_exps_s);
        ggml_tensor * dst = ggml_view_2d(
            ctx, packed_out, n_embd, count, packed_out->nb[1],
            begin * packed_out->nb[1]);
        ggml_tensor * copy = ggml_cpy(ctx, expert_out, dst);
        ggml_set_output(copy);
        ggml_build_forward_expand(gf, copy);
    }

    ggml_tensor * shared_in = nullptr;
    ggml_tensor * shared_out = nullptr;
    if (has_shared) {
        if (owner_input) {
            shared_in = owner_input;
        } else {
            shared_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                           n_embd, n_tokens);
            ggml_set_input(shared_in);
        }
        shared_out = build_shared_expert_subgraph(
            ctx, desc, shared_in, cfg.swiglu_clamp);
        ggml_set_output(shared_out);
        ggml_build_forward_expand(gf, shared_out);
    }


    ggml_tensor * combined_out = shared_out;
    if (gpu_reduce) {
        // packed_out is [n_embd, n_pairs].  GET_ROWS applies the inverse
        // expert-major permutation and produces [n_embd, n_used, n_tokens].
        // Invalid/non-owner routes gather row zero but have an exact zero
        // weight, so the standard fused MoE reduction masks them out.
        // GET_ROWS interprets ids dimension 1 as a batched lookup and requires
        // the source to expose the same batch dimension.  Every token indexes
        // the same packed table, so use a zero-stride view instead of physically
        // repeating an O(n_pairs * n_tokens) tensor.
        ggml_tensor * packed_table = ggml_view_tensor(ctx, packed_out);
        packed_table->ne[2] = n_tokens;
        packed_table->nb[2] = 0;
        packed_table->ne[3] = 1;
        packed_table->nb[3] = 0;
        ggml_tensor * route_major =
            ggml_get_rows(ctx, packed_table, inverse_routes);
        ggml_tensor * routed_out =
            ggml_laguna_moe_combine(ctx, route_major, route_weights);
        combined_out = combined_out
            ? ggml_add(ctx, routed_out, combined_out)
            : routed_out;
        ggml_set_output(combined_out);
        ggml_build_forward_expand(gf, combined_out);
    }

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
        if (alloc) ggml_gallocr_free(alloc);
        ggml_free(ctx);
        if (err) *err = "expert-major scratch allocation failed";
        return false;
    }
    if (owner_input) {
        if (cur_backend_owner) {
            ggml_backend_tensor_copy_async(
                cur_backend_owner, backend, cur_backend, owner_input);
        } else {
            ggml_backend_tensor_copy(cur_backend, owner_input);
        }
    }
    if (packed_token_ids) {
        ggml_backend_tensor_set(packed_token_ids, packed_tokens.data(), 0,
                                sizeof(int32_t) * packed_tokens.size());
    } else if (packed_in) {
        ggml_backend_tensor_set(packed_in, packed_input.data(), 0,
                                sizeof(float) * packed_input.size());
    }
    if (shared_in && !owner_input) {
        ggml_backend_tensor_set(shared_in, cur_host, 0,
                                sizeof(float) * (size_t)n_embd *
                                    (size_t)n_tokens);
    }
    if (gpu_reduce) {
        ggml_backend_tensor_set(inverse_routes, route_to_packed.data(), 0,
                                sizeof(int32_t) * route_to_packed.size());
        ggml_backend_tensor_set(route_weights, owner_weights.data(), 0,
                                sizeof(float) * owner_weights.size());
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        if (err) *err = "expert-major graph compute failed";
        return false;
    }

    if (device_output) {
        if (!device_output_owner || !gpu_reduce || !combined_out ||
            device_output->type != GGML_TYPE_F32 ||
            device_output->ne[0] != n_embd ||
            device_output->ne[1] != n_tokens) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            if (err) *err = "invalid expert-major device output";
            return false;
        }
        // Publish the owner result directly to the target GPU. Synchronizing
        // the destination before releasing this graph keeps the source arena
        // alive until the peer/local copy has retired.
        ggml_backend_tensor_copy_async(
            backend, device_output_owner, combined_out, device_output);
        ggml_backend_synchronize(device_output_owner);
    } else if (gpu_reduce) {
        ggml_backend_tensor_get(combined_out, out.data(), 0,
                                sizeof(float) * out.size());
    } else {
        std::vector<float> packed_result(n_pairs * (size_t)n_embd);
        if (packed_out) {
            ggml_backend_tensor_get(packed_out, packed_result.data(), 0,
                                    sizeof(float) * packed_result.size());
        }
        if (shared_out) {
            ggml_backend_tensor_get(shared_out, out.data(), 0,
                                    sizeof(float) * out.size());
        }
        for (size_t packed = 0; packed < n_pairs; ++packed) {
            const size_t route = (size_t)packed_routes[packed];
            const size_t token = route / (size_t)n_used;
            float * dst = out.data() + token * (size_t)n_embd;
            const float * src =
                packed_result.data() + packed * (size_t)n_embd;
            const float weight = packed_weights[packed];
            for (int i = 0; i < n_embd; ++i) {
                dst[i] += src[i] * weight;
            }
        }
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

// Pack only this owner's live token/expert pairs before MUL_MAT_ID. The
// ordinary masked graph retains n_tokens*n_used rows and merely marks the
// other owner's rows negative; its activation quantizer still visits that
// fixed-capacity tensor. This path shrinks the tensor to the actual owner
// routes, executes one full-stack MMID graph, and performs the inverse
// permutation plus route reduction on the owner GPU.
static bool eval_moe_owner_packed_mmid_batched(
    ggml_backend_t                  backend,
    const MoeHybridConfig &         cfg,
    const MoeLayerDesc &            desc,
    ggml_tensor *                   gate_tensor,
    ggml_tensor *                   up_tensor,
    ggml_tensor *                   down_tensor,
    ggml_tensor *                   gate_up_tensor,
    const std::vector<int32_t> &     local_by_global,
    const float *                   cur_host,
    const int32_t *                 selected_ids,
    const float *                   selected_weights,
    int                             n_tokens,
    std::vector<float> &            out,
    std::string *                   err,
    ggml_gallocr_t *                p_alloc,
    ggml_tensor *                   cur_backend = nullptr,
    ggml_backend_t                  cur_backend_owner = nullptr) {
    const int n_embd = cfg.n_embd;
    const int n_used = cfg.n_expert_used;
    const int n_ff = cfg.n_ff_exp;
    const size_t n_routes = (size_t)n_tokens * (size_t)n_used;
    out.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
    if (!backend || (!cur_host && !cur_backend) || !selected_ids ||
        !selected_weights || n_tokens <= 0 || n_embd <= 0 || n_used <= 0 ||
        n_ff <= 0 || !down_tensor ||
        (!gate_up_tensor && (!gate_tensor || !up_tensor))) {
        if (err) *err = "invalid packed-MMID owner inputs";
        return false;
    }

    ggml_tensor * stack_ref = gate_up_tensor ? gate_up_tensor : gate_tensor;
    const int n_stack = (int)stack_ref->ne[2];
    if (n_stack <= 0 || down_tensor->ne[2] != n_stack) {
        if (err) *err = "invalid packed-MMID owner stack";
        return false;
    }

    std::vector<int32_t> packed_tokens;
    std::vector<int32_t> packed_ids;
    std::vector<int32_t> route_to_packed(n_routes, 0);
    std::vector<float> route_weights(n_routes, 0.0f);
    std::vector<int32_t> rows_per_expert((size_t)n_stack, 0);
    packed_tokens.reserve(n_routes);
    packed_ids.reserve(n_routes);
    for (int t = 0; t < n_tokens; ++t) {
        for (int slot = 0; slot < n_used; ++slot) {
            const size_t route =
                (size_t)t * (size_t)n_used + (size_t)slot;
            const int32_t global = selected_ids[route];
            if (global < 0 || (size_t)global >= local_by_global.size() ||
                selected_weights[route] == 0.0f) {
                continue;
            }
            const int32_t local = local_by_global[(size_t)global];
            if (local < 0 || local >= n_stack) continue;
            const int32_t packed = (int32_t)packed_ids.size();
            packed_tokens.push_back(t);
            packed_ids.push_back(local);
            route_to_packed[route] = packed;
            route_weights[route] = selected_weights[route];
            rows_per_expert[(size_t)local]++;
        }
    }
    if (packed_ids.empty()) return true;

    int32_t max_expert_rows = 0;
    for (int32_t count : rows_per_expert) {
        max_expert_rows = std::max(max_expert_rows, count);
    }
    MmidOwnerGridHint grid_hint{
        MMID_OWNER_GRID_HINT_MAGIC, max_expert_rows,
        (int32_t)packed_ids.size()};
    const int64_t n_pairs = (int64_t)packed_ids.size();
    std::vector<float> unit_weights((size_t)n_pairs, 1.0f);

    ggml_init_params ip{};
    ip.mem_size = 64 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        if (err) *err = "packed-MMID ggml_init failed";
        return false;
    }

    ggml_tensor * owner_input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(owner_input);
    ggml_tensor * packed_token_ids = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, n_pairs);
    ggml_set_input(packed_token_ids);
    ggml_tensor * packed_input =
        ggml_get_rows(ctx, owner_input, packed_token_ids);
    ggml_tensor * ids = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, 1, n_pairs);
    ggml_set_input(ids);
    ids->extra = &grid_hint;
    ggml_tensor * weights = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, 1, n_pairs);
    ggml_set_input(weights);

    ggml_tensor * packed_output = nullptr;
    if (!build_batched_routed_graph(
            ctx, gate_tensor, up_tensor, down_tensor, gate_up_tensor,
            desc.ffn_gate_exps_s, desc.ffn_up_exps_s,
            desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
            packed_input, ids, weights, n_embd, n_ff, 1, (int)n_pairs,
            cfg.swiglu_clamp, &packed_output, false, nullptr,
            /*allow_fused_combine=*/true)) {
        ggml_free(ctx);
        if (err) *err = "packed-MMID graph build failed";
        return false;
    }

    ggml_tensor * inverse_routes = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_input(inverse_routes);
    ggml_tensor * final_weights = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_used, n_tokens);
    ggml_set_input(final_weights);

    ggml_tensor * combined =
        ggml_laguna_moe_packed_combine(
            ctx, packed_output, inverse_routes, final_weights);
    ggml_set_output(combined);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    // Register the external leaves explicitly. The dynamically packed graph
    // is assembled around a caller-owned device tensor and its shape changes
    // per layer, so it must not inherit a fixed graph's leaf set.
    ggml_build_forward_expand(gf, owner_input);
    ggml_build_forward_expand(gf, packed_token_ids);
    ggml_build_forward_expand(gf, ids);
    ggml_build_forward_expand(gf, weights);
    ggml_build_forward_expand(gf, inverse_routes);
    ggml_build_forward_expand(gf, final_weights);
    ggml_build_forward_expand(gf, combined);

    // n_pairs changes from layer to layer. The reusable fixed-shape owner
    // allocator can leave new dynamic input leaves without a buffer, so keep
    // this graph's arena local until a shape-keyed cache is introduced.
    (void)p_alloc;
    ggml_gallocr_t alloc =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
        if (alloc) ggml_gallocr_free(alloc);
        ggml_free(ctx);
        if (err) *err = "packed-MMID scratch allocation failed";
        return false;
    }
    if (!owner_input->buffer) {
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        if (err) *err = "packed-MMID owner input was not allocated";
        return false;
    }

    if (cur_backend) {
        if (cur_backend_owner) {
            ggml_backend_tensor_copy_async(
                cur_backend_owner, backend, cur_backend, owner_input);
        } else {
            ggml_backend_tensor_copy(cur_backend, owner_input);
        }
    } else {
        ggml_backend_tensor_set(owner_input, cur_host, 0,
            sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
    }
    ggml_backend_tensor_set(packed_token_ids, packed_tokens.data(), 0,
        sizeof(int32_t) * packed_tokens.size());
    ggml_backend_tensor_set(ids, packed_ids.data(), 0,
        sizeof(int32_t) * packed_ids.size());
    ggml_backend_tensor_set(weights, unit_weights.data(), 0,
        sizeof(float) * unit_weights.size());
    ggml_backend_tensor_set(inverse_routes, route_to_packed.data(), 0,
        sizeof(int32_t) * route_to_packed.size());
    ggml_backend_tensor_set(final_weights, route_weights.data(), 0,
        sizeof(float) * route_weights.size());

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        if (err) *err = "packed-MMID graph compute failed";
        return false;
    }
    ggml_backend_tensor_get(combined, out.data(), 0,
        sizeof(float) * out.size());

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

// ── Hot-Only Batched Prefill ──
// When all selected experts are in VRAM, skip cold entirely: no CPU graph,
// no partition into hot/cold, no merge loop. Pure GPU.

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
    std::string *                   err,
    ggml_gallocr_t *                p_hot_alloc) {

    const int n_embd = cfg.n_embd;
    const int n_used = cfg.n_expert_used;
    const int n_ff_exp = cfg.n_ff_exp;
    out.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
    if (n_tokens <= 0) return true;

    // Workaround for the ggml-cuda MMQ mul_mat_id stream-k fault on a REDUCED
    // hot stack (sm_75/gfx1151 AND sm_86): slice sub-64 batches to a size the
    // MMVQ-mmid path handles. See mmq_safe_sub_batch().
    const int n_hot_stack = storage.gate_up_hot ? (int)storage.gate_up_hot->ne[2]
                          : storage.gate_hot    ? (int)storage.gate_hot->ne[2]
                          : 0;
    const int MMQ_SAFE_SUB_BATCH = mmq_safe_sub_batch();
    if (!mmq_full_batch_ok(cfg, n_tokens)
        && n_hot_stack > 0 && n_tokens > MMQ_SAFE_SUB_BATCH) {
        std::vector<float> sub_out;
        for (int t0 = 0; t0 < n_tokens; t0 += MMQ_SAFE_SUB_BATCH) {
            const int tc = std::min(MMQ_SAFE_SUB_BATCH, n_tokens - t0);
            if (!eval_moe_hot_only_batched(
                    gpu_backend, cfg, desc, storage,
                    cur_host + (size_t)t0 * (size_t)n_embd,
                    selected_ids + (size_t)t0 * (size_t)n_used,
                    selected_weights + (size_t)t0 * (size_t)n_used,
                    tc, sub_out, err, p_hot_alloc)) {
                return false;
            }
            std::memcpy(out.data() + (size_t)t0 * (size_t)n_embd,
                        sub_out.data(),
                        sizeof(float) * (size_t)n_embd * (size_t)tc);
        }
        return true;
    }

    // Remap global expert IDs → hot-local IDs
    const int total_slots = n_used * n_tokens;
    std::vector<int32_t> hot_sel(total_slots);
    for (int i = 0; i < total_slots; ++i) {
        const int32_t gid = selected_ids[i];
        if (gid < 0 || gid >= (int32_t)storage.hot_local_by_global.size()) {
            hot_sel[i] = 0;
        } else {
            hot_sel[i] = storage.hot_local_by_global[(size_t)gid];
        }
    }

    // ── Fast path: use cached graph (avoids rebuild + realloc) ──
    // Per-n_tokens cache: spec-decode alternates verify (verify_width) and
    // replay (commit_n) sizes; a single slot would rebuild all 40 layers' FFN
    // graphs on every flip. Index by n_tokens so each size keeps its own graph.
    auto & cached = (n_tokens > 0 && n_tokens < MoeHybridLayerStorage::kMaxBatchedCache)
                  ? storage.hot_batched_mixed[n_tokens]
                  : storage.hot_batched_graph;
    if (cached.n_tokens == n_tokens && cached.valid()) {
        // Reuse pre-built graph: just upload data and compute
        ggml_backend_tensor_set(cached.inp, cur_host, 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
        ggml_backend_tensor_set(cached.sel, hot_sel.data(), 0, sizeof(int32_t) * (size_t)total_slots);
        ggml_backend_tensor_set(cached.wts, selected_weights, 0, sizeof(float) * (size_t)total_slots);

        auto st = ggml_backend_graph_compute(gpu_backend, cached.gf);
        if (st != GGML_STATUS_SUCCESS) {
            if (err) *err = "hot_only cached compute failed";
            return false;
        }
        ggml_backend_tensor_get(cached.output, out.data(), 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
        return true;
    }

    // ── Slow path: build graph (first call or size mismatch) ──
    // Try to build and cache for this n_tokens size.
    // Cache when: sub-batch size (legacy), full stack (all hot), or full-batch safe (sm_80+).
    if (mmq_full_batch_ok(cfg, n_tokens) || n_tokens <= MMQ_SAFE_SUB_BATCH
        || (n_hot_stack == 0 || n_hot_stack >= cfg.n_expert)) {
        if (build_cached_hot_batched_graph(cached, gpu_backend, storage, desc, cfg, n_tokens)) {
            // Successfully cached — use it immediately
            ggml_backend_tensor_set(cached.inp, cur_host, 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
            ggml_backend_tensor_set(cached.sel, hot_sel.data(), 0, sizeof(int32_t) * (size_t)total_slots);
            ggml_backend_tensor_set(cached.wts, selected_weights, 0, sizeof(float) * (size_t)total_slots);

            auto st = ggml_backend_graph_compute(gpu_backend, cached.gf);
            if (st != GGML_STATUS_SUCCESS) {
                if (err) *err = "hot_only cached compute failed (first)";
                return false;
            }
            ggml_backend_tensor_get(cached.output, out.data(), 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
            return true;
        }
        // Fall through to uncached path if build fails
    }

    // ── Uncached fallback (remainder sub-batches with n_tokens < MMQ_SAFE_SUB_BATCH) ──
    ggml_init_params ip{};
    ip.mem_size = 128 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { if (err) *err = "hot_only ggml_init failed"; return false; }

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp);
    ggml_tensor * sel = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_input(sel);
    ggml_tensor * wts = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_used, n_tokens);
    ggml_set_input(wts);

    ggml_tensor * routed = nullptr;
    build_batched_routed_graph(ctx,
        storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
        desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
        inp, sel, wts, n_embd, n_ff_exp, n_used, n_tokens,
        cfg.swiglu_clamp, &routed, false, nullptr,
        ggml_backend_is_cuda(gpu_backend));

    // Shared expert (always on GPU)
    ggml_tensor * combined = routed;
    ggml_tensor * shared = build_shared_expert_subgraph(ctx, desc, inp, cfg.swiglu_clamp);
    if (shared) {
        combined = combined ? ggml_add(ctx, combined, shared) : shared;
    }

    if (!combined) {
        ggml_free(ctx);
        if (err) *err = "hot_only: no routed or shared output";
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_set_output(combined);
    ggml_build_forward_expand(gf, combined);

    ggml_gallocr_t alloc = nullptr;
    if (p_hot_alloc) {
        if (!*p_hot_alloc)
            *p_hot_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu_backend));
        alloc = *p_hot_alloc;
    } else {
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu_backend));
    }
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        if (err) *err = "hot_only gallocr failed";
        if (!p_hot_alloc) ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(inp, cur_host, 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
    ggml_backend_tensor_set(sel, hot_sel.data(), 0, sizeof(int32_t) * (size_t)total_slots);
    ggml_backend_tensor_set(wts, selected_weights, 0, sizeof(float) * (size_t)total_slots);

    auto st = ggml_backend_graph_compute(gpu_backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        if (err) *err = "hot_only compute failed";
        if (!p_hot_alloc) ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(combined, out.data(), 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
    if (!p_hot_alloc) ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

// ── GPU-Resident Residual State ──

// Public entry. Workaround for a ggml-cuda/HIP defect: the MMQ mul_mat_id
// kernel illegal-accesses on gfx1151 when the per-layer hot expert stack is
// REDUCED (n_hot_stack < n_expert); the full-stack (all-hot) case is fine.
// MMVQ is used instead of MMQ only when the matmul batch dim (= n_tokens) is
// small (Q4_K AMD MMVQ-mmid cap is 4). So for reduced hot stacks we slice the
// prefill batch into <=4-token sub-batches, routing the routed mul_mat_id
// through the stable MMVQ path. Full stacks keep the fast single-shot MMQ.
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
    std::string *                   err,
    ggml_gallocr_t *                p_hot_alloc,
    ggml_gallocr_t *                p_cold_alloc,
    MoeExpertCompute *                expert_compute,
    const MoeExpertLayer *            expert_layer,
    MoeHybridFfnTelemetry *         telemetry,
    ggml_tensor *                   cur_backend,
    const MoeHybridDeviceOutputs *  device_outputs) {
    if (telemetry) *telemetry = {};
    const bool materialized_cold = storage.down_cold || storage.gate_up_cold;
    if (compact_materialized_experts_enabled() && materialized_cold &&
        !expert_compute && n_tokens > 0 && n_tokens <= 4) {
        out.assign((size_t)cfg.n_embd * (size_t)n_tokens, 0.0f);
        std::vector<float> token_out;
        for (int t = 0; t < n_tokens; ++t) {
            MoeHybridFfnTelemetry token_telemetry;
            if (!eval_moe_hybrid_ffn_single(
                    gpu_backend, cfg, desc, storage, cpu_backend,
                    cur_host + (size_t)t * (size_t)cfg.n_embd,
                    selected_ids + (size_t)t * (size_t)cfg.n_expert_used,
                    selected_weights + (size_t)t * (size_t)cfg.n_expert_used,
                    cfg.n_expert_used, token_out,
                    telemetry ? &token_telemetry : nullptr, err)) {
                return false;
            }
            if (telemetry) add_hybrid_telemetry(*telemetry, token_telemetry);
            std::memcpy(out.data() + (size_t)t * (size_t)cfg.n_embd,
                        token_out.data(),
                        sizeof(float) * (size_t)cfg.n_embd);
        }
        return true;
    }
    const int n_hot_stack = storage.gate_up_hot ? (int)storage.gate_up_hot->ne[2]
                          : storage.gate_hot    ? (int)storage.gate_hot->ne[2]
                          : 0;
    const int n_cold_stack = storage.gate_up_cold ? (int)storage.gate_up_cold->ne[2]
                            : storage.gate_cold    ? (int)storage.gate_cold->ne[2]
                            : 0;
    const bool cold_on_gpu = storage.cold_backend_kind == MoeHybridColdBackend::Gpu;
    const bool inprocess_expert_major =
        !expert_compute && expert_major_prefill_enabled(n_tokens) &&
        cold_on_gpu && storage.cold_backend &&
        storage.cold_backend != gpu_backend &&
        n_hot_stack > 0 && n_cold_stack > 0 &&
        n_cold_stack < cfg.n_expert;
    if (inprocess_expert_major) {
        static bool logged = false;
        if (!logged) {
            std::fprintf(stderr,
                         "[hybrid-ffn] heterogeneous expert-major prefill "
                         "active tokens=%d hot_stack=%d cold_stack=%d\n",
                         n_tokens, n_hot_stack, n_cold_stack);
            logged = true;
        }
        const auto wall_t0 = HybridClock::now();
        std::vector<float> hot_partial;
        std::vector<float> cold_partial;
        std::string hot_err;
        std::string cold_err;
        const auto cold_t0 = HybridClock::now();
        auto cold_future = std::async(std::launch::async, [&]() {
            HybridCudaGraphDisableScope graph_scope(
                heterogeneous_prefill_eager_enabled());
            return eval_moe_owner_expert_major_batched(
                storage.cold_backend, cfg, desc,
                storage.gate_cold, storage.up_cold, storage.down_cold,
                storage.gate_up_cold, storage.cold_local_by_global,
                cur_host, selected_ids, selected_weights, n_tokens,
                /*include_shared=*/false, cold_partial, &cold_err,
                cur_backend, gpu_backend);
        });
        const auto hot_t0 = HybridClock::now();
        const bool hot_ok = eval_moe_owner_expert_major_batched(
            gpu_backend, cfg, desc,
            storage.gate_hot, storage.up_hot, storage.down_hot,
            storage.gate_up_hot, storage.hot_local_by_global,
            cur_host, selected_ids, selected_weights, n_tokens,
            /*include_shared=*/true, hot_partial, &hot_err,
            cur_backend, gpu_backend);
        const auto hot_t1 = HybridClock::now();
        const bool cold_ok = cold_future.get();
        const auto done = HybridClock::now();
        if (!hot_ok || !cold_ok) {
            if (err) {
                *err = !hot_ok ? hot_err : cold_err;
            }
            return false;
        }
        const size_t total = (size_t)cfg.n_embd * (size_t)n_tokens;
        out.resize(total);
        for (size_t i = 0; i < total; ++i) {
            out[i] = hot_partial[i] + cold_partial[i];
        }
        if (telemetry) {
            telemetry->hot_us += elapsed_us(hot_t0, hot_t1);
            telemetry->cold_us += elapsed_us(cold_t0, done);
            telemetry->ffn_wall_us += elapsed_us(wall_t0, done);
        }
        return true;
    }

    // A prefill-only full cold stack keeps Strix on its proven full-stack
    // MUL_MAT_ID path.  Do not pair it with hundreds of q<=4 hot sub-batches:
    // pack the R9700 routes by expert and launch one owner graph instead.  The
    // hot-local map takes priority in eval_moe_hybrid_ffn_batched_core(), so
    // skip_hot makes the duplicated Strix stack evaluate cold routes only.
    const bool inprocess_full_cold_hot_expert_major =
        !expert_compute && expert_major_prefill_enabled(n_tokens) &&
        cold_on_gpu && storage.cold_backend &&
        storage.cold_backend != gpu_backend &&
        n_hot_stack > 0 && n_cold_stack == cfg.n_expert;
    if (inprocess_full_cold_hot_expert_major) {
        const bool cold_expert_major =
            full_cold_expert_major_enabled();
        const bool cold_packed_mmid =
            full_cold_packed_mmid_enabled();
        const bool device_join =
            device_outputs && device_outputs->valid() && cur_backend;
        if (device_join && (cold_expert_major || cold_packed_mmid)) {
            if (err) {
                *err = "device-resident join requires the full-cold MMID path";
            }
            return false;
        }
        static bool logged = false;
        if (!logged) {
            const char * cold_mode = cold_packed_mmid
                ? "packed-MMID"
                : (cold_expert_major ? "expert-major" : "MMID batch");
            std::fprintf(stderr,
                         "[hybrid-ffn] full-cold Strix %s + expert-major "
                         "R9700 prefill active tokens=%d hot_stack=%d\n",
                         cold_mode,
                         n_tokens, n_hot_stack);
            logged = true;
        }

        const auto wall_t0 = HybridClock::now();
        std::vector<float> hot_partial;
        std::vector<float> cold_partial;
        std::string hot_err;
        std::string cold_err;
        std::vector<int32_t> cold_only_local;
        if (cold_expert_major || cold_packed_mmid) {
            cold_only_local = storage.cold_local_by_global;
            const size_t n = std::min(cold_only_local.size(),
                                      storage.hot_local_by_global.size());
            for (size_t global = 0; global < n; ++global) {
                if (storage.hot_local_by_global[global] >= 0) {
                    cold_only_local[global] = -1;
                }
            }
        }

        const auto cold_t0 = HybridClock::now();
        auto cold_future = std::async(std::launch::async, [&]() {
            HybridCudaGraphDisableScope graph_scope(
                heterogeneous_prefill_eager_enabled());
            if (cold_packed_mmid) {
                return eval_moe_owner_packed_mmid_batched(
                    storage.cold_backend, cfg, desc,
                    storage.gate_cold, storage.up_cold,
                    storage.down_cold, storage.gate_up_cold,
                    cold_only_local,
                    cur_host, selected_ids, selected_weights, n_tokens,
                    cold_partial, &cold_err, p_cold_alloc,
                    cur_backend, gpu_backend);
            }
            if (cold_expert_major) {
                return eval_moe_owner_expert_major_batched(
                    storage.cold_backend, cfg, desc,
                    storage.gate_cold, storage.up_cold,
                    storage.down_cold, storage.gate_up_cold,
                    cold_only_local,
                    cur_host, selected_ids, selected_weights, n_tokens,
                    /*include_shared=*/false, cold_partial, &cold_err,
                    cur_backend, gpu_backend);
            }
            return eval_moe_hybrid_ffn_batched_core(
                gpu_backend, cpu_backend, cfg, desc, storage,
                cur_host, selected_ids, selected_weights,
                n_tokens, cold_partial, &cold_err,
                nullptr, p_cold_alloc, nullptr, nullptr, nullptr,
                /*skip_cold=*/false, /*skip_hot=*/true, cur_backend,
                device_join ? device_outputs->cold : nullptr,
                device_join ? device_outputs->backend : nullptr);
        });

        const auto hot_t0 = HybridClock::now();
        const bool hot_ok = eval_moe_owner_expert_major_batched(
            gpu_backend, cfg, desc,
            storage.gate_hot, storage.up_hot, storage.down_hot,
            storage.gate_up_hot, storage.hot_local_by_global,
            cur_host, selected_ids, selected_weights, n_tokens,
            /*include_shared=*/true, hot_partial, &hot_err,
            cur_backend, gpu_backend,
            device_join ? device_outputs->hot : nullptr,
            device_join ? device_outputs->backend : nullptr);
        const auto hot_t1 = HybridClock::now();
        const bool cold_ok = cold_future.get();
        const auto done = HybridClock::now();
        if (!hot_ok || !cold_ok) {
            if (err) *err = !hot_ok ? hot_err : cold_err;
            return false;
        }

        if (!device_join) {
            const size_t total = (size_t)cfg.n_embd * (size_t)n_tokens;
            out.resize(total);
            for (size_t i = 0; i < total; ++i) {
                out[i] = hot_partial[i] + cold_partial[i];
            }
        } else {
            out.clear();
        }
        if (telemetry) {
            telemetry->hot_us += elapsed_us(hot_t0, hot_t1);
            telemetry->cold_us += elapsed_us(cold_t0, done);
            telemetry->ffn_wall_us += elapsed_us(wall_t0, done);
        }
        return true;
    }

    const int hot_sub_batch =
        std::min(mmq_safe_sub_batch(), moe_hybrid_prefill_hot_sub_batch_limit());
    if (!mmq_full_batch_ok(cfg, n_tokens)
        && ((n_hot_stack > 0 && n_hot_stack < cfg.n_expert) ||
            (cold_on_gpu && n_cold_stack > 0 && n_cold_stack < cfg.n_expert))
        && n_tokens > hot_sub_batch) {
        const int n_embd = cfg.n_embd;
        const int n_used = cfg.n_expert_used;
        out.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
        const bool remote_cold_full_batch =
            expert_compute && expert_layer &&
            parse_moe_expert_compute_ipc_mode() == MoeExpertComputeIpcMode::Batched;
        // The in-process heterogeneous path owns a nearly-full cold stack on
        // Strix and a small hot stack on the R9700.  A full reduced-stack MMQ
        // is pathological for the 24-expert hot side, but is stable once the
        // cold stack covers most experts and the prompt supplies enough rows.
        // Run that one cold batch on Strix while retaining the proven q<=4
        // MMVQ slices for R9700 hot/shared work.
        const bool inprocess_cold_full_batch =
            !expert_compute && cold_on_gpu &&
            storage.cold_backend && storage.cold_backend != gpu_backend &&
            n_cold_stack * 4 >= cfg.n_expert * 3 && n_tokens >= 64;
        if (remote_cold_full_batch || inprocess_cold_full_batch) {
            std::vector<float> hot_partial((size_t)n_embd * (size_t)n_tokens, 0.0f);
            std::vector<float> cold_partial;
            std::string cold_err;
            const auto cold_t0 = HybridClock::now();
            auto cold_future = std::async(std::launch::async, [&]() {
                HybridCudaGraphDisableScope graph_scope(
                    heterogeneous_prefill_eager_enabled());
                if (remote_cold_full_batch) {
                    return eval_moe_hybrid_remote_cold_batched(
                        cfg, storage, cur_host, selected_ids, selected_weights,
                        n_tokens, cold_partial, &cold_err,
                        expert_compute, expert_layer);
                }
                return eval_moe_hybrid_ffn_batched_core(
                    gpu_backend, cpu_backend, cfg, desc, storage,
                    cur_host, selected_ids, selected_weights,
                    n_tokens, cold_partial, &cold_err,
                    nullptr, p_cold_alloc, nullptr, nullptr, nullptr,
                    /*skip_cold=*/false, /*skip_hot=*/true);
            });

            std::vector<float> sub_out;
            bool hot_ok = true;
            for (int t0 = 0; t0 < n_tokens; t0 += hot_sub_batch) {
                const int tc = std::min(hot_sub_batch, n_tokens - t0);
                if (!eval_moe_hybrid_ffn_batched_core(
                        gpu_backend, cpu_backend, cfg, desc, storage,
                        cur_host + (size_t)t0 * (size_t)n_embd,
                        selected_ids + (size_t)t0 * (size_t)n_used,
                        selected_weights + (size_t)t0 * (size_t)n_used,
                        tc, sub_out, err, p_hot_alloc, p_cold_alloc,
                        expert_compute, expert_layer, nullptr, /*skip_cold=*/true)) {
                    hot_ok = false;
                    break;
                }
                std::memcpy(hot_partial.data() + (size_t)t0 * (size_t)n_embd,
                            sub_out.data(),
                            sizeof(float) * (size_t)n_embd * (size_t)tc);
            }

            const bool cold_ok = cold_future.get();
            if (!hot_ok) return false;
            if (!cold_ok) {
                if (err && !cold_err.empty()) *err = cold_err;
                return false;
            }
            if (telemetry) telemetry->cold_us += elapsed_us(cold_t0, HybridClock::now());
            const size_t total_floats = (size_t)n_embd * (size_t)n_tokens;
            for (size_t i = 0; i < total_floats; ++i) {
                out[i] = hot_partial[i] + cold_partial[i];
            }
            return true;
        }
        std::vector<float> sub_out;
        for (int t0 = 0; t0 < n_tokens; t0 += hot_sub_batch) {
            const int tc = std::min(hot_sub_batch, n_tokens - t0);
            MoeHybridFfnTelemetry sub_tel;
            if (!eval_moe_hybrid_ffn_batched_core(
                    gpu_backend, cpu_backend, cfg, desc, storage,
                    cur_host + (size_t)t0 * (size_t)n_embd,
                    selected_ids + (size_t)t0 * (size_t)n_used,
                    selected_weights + (size_t)t0 * (size_t)n_used,
                    tc, sub_out, err, p_hot_alloc, p_cold_alloc,
                    expert_compute, expert_layer,
                    telemetry ? &sub_tel : nullptr)) {
                return false;
            }
            if (telemetry) {
                telemetry->ffn_wall_us += sub_tel.ffn_wall_us;
                telemetry->partition_us += sub_tel.partition_us;
                telemetry->hot_us += sub_tel.hot_us;
                telemetry->cold_us += sub_tel.cold_us;
                telemetry->combine_us += sub_tel.combine_us;
                telemetry->hot_graph_build_us += sub_tel.hot_graph_build_us;
                telemetry->hot_input_us += sub_tel.hot_input_us;
                telemetry->hot_compute_us += sub_tel.hot_compute_us;
                telemetry->hot_read_us += sub_tel.hot_read_us;
                telemetry->cold_graph_build_us += sub_tel.cold_graph_build_us;
                telemetry->cold_input_us += sub_tel.cold_input_us;
                telemetry->cold_compute_us += sub_tel.cold_compute_us;
                telemetry->cold_read_us += sub_tel.cold_read_us;
                telemetry->hot_graph_builds += sub_tel.hot_graph_builds;
                telemetry->hot_graph_hits += sub_tel.hot_graph_hits;
                telemetry->cold_graph_builds += sub_tel.cold_graph_builds;
                telemetry->cold_graph_hits += sub_tel.cold_graph_hits;
                telemetry->hot_selected += sub_tel.hot_selected;
                telemetry->cold_selected += sub_tel.cold_selected;
            }
            std::memcpy(out.data() + (size_t)t0 * (size_t)n_embd,
                        sub_out.data(),
                        sizeof(float) * (size_t)n_embd * (size_t)tc);
        }
        return true;
    }
    return eval_moe_hybrid_ffn_batched_core(
        gpu_backend, cpu_backend, cfg, desc, storage,
        cur_host, selected_ids, selected_weights, n_tokens, out, err,
        p_hot_alloc, p_cold_alloc, expert_compute, expert_layer, telemetry);
}

void ResidualCombineGraph::free() {
    if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
    if (ctx) { ggml_free(ctx); ctx = nullptr; }
    gf = nullptr;
    residual_in = nullptr;
    hot_in = nullptr;
    cold_in = nullptr;
    output = nullptr;
}

void ResidualCombineGraph::destroy() {
    free();
}

bool build_residual_combine_graph(ResidualCombineGraph & out, ggml_backend_t backend, int n_embd) {
    out.free();

    ggml_init_params ip{};
    ip.mem_size = 4 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    out.residual_in = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
    ggml_set_input(out.residual_in);
    out.hot_in = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
    ggml_set_input(out.hot_in);
    out.cold_in = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
    ggml_set_input(out.cold_in);

    // output = residual + hot + cold
    ggml_tensor * sum = ggml_add(out.ctx, out.residual_in, out.hot_in);
    out.output = ggml_add(out.ctx, sum, out.cold_in);
    ggml_set_output(out.output);

    out.gf = ggml_new_graph_custom(out.ctx, 64, false);
    ggml_build_forward_expand(out.gf, out.output);

    out.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(out.alloc, out.gf)) {
        out.free();
        return false;
    }
    return true;
}

void GpuResidentState::destroy() {
    combine.destroy();
    if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
    if (ctx) { ggml_free(ctx); ctx = nullptr; }
    act_cur = nullptr;
}

bool init_gpu_resident_state(GpuResidentState & out, ggml_backend_t backend, int n_embd) {
    out.destroy();

    ggml_init_params ip{};
    ip.mem_size = 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    out.act_cur = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, n_embd, 1, 1);
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        out.destroy();
        return false;
    }

    if (!build_residual_combine_graph(out.combine, backend, n_embd)) {
        out.destroy();
        return false;
    }

    std::vector<float> zeros((size_t)n_embd, 0.0f);
    ggml_backend_tensor_set(out.combine.cold_in, zeros.data(), 0, sizeof(float) * (size_t)n_embd);

    return true;
}

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
    MoeExpertCompute *                expert_compute,
    const MoeExpertLayer *            expert_layer) {

    const int n_embd = cfg.n_embd;

    // ── Partition into hot/cold ──
    std::vector<int32_t> hot_ids;
    std::vector<float> hot_weights;
    std::vector<int32_t> cold_ids;
    std::vector<float> cold_weights;
    hot_ids.reserve((size_t)n_selected);
    hot_weights.reserve((size_t)n_selected);

    for (int i = 0; i < n_selected; ++i) {
        const int32_t gid = selected_ids[i];
        if (gid < 0 || gid >= (int32_t)storage.hot_local_by_global.size()) return false;
        const int32_t hot_local = storage.hot_local_by_global[(size_t)gid];
        if (hot_local >= 0) {
            hot_ids.push_back(hot_local);
            hot_weights.push_back(selected_weights[i]);
        } else {
            const int32_t cold_local = storage.cold_local_by_global[(size_t)gid];
            if (cold_local >= 0) {
                cold_ids.push_back(cold_local);
                cold_weights.push_back(selected_weights[i]);
            }
        }
    }

    const int n_hot = (int)hot_ids.size();
    const bool has_hot = (n_hot > 0);
    const bool has_shared = (desc.ffn_up_shexp && desc.ffn_gate_shexp && desc.ffn_down_shexp);
    const bool has_cold = !cold_ids.empty();
    const int n_cold = (int)cold_ids.size();

    // ── GPU-remap fast path (laguna): fold residual + hot-routed + shared into a
    // single cached GPU graph that consumes the router's expert ids directly
    // (cold experts masked to 0 via valid_lut). Removes the separate per-layer
    // residual-combine graph_compute and the host hot/cold partition for the GPU
    // path. Cold experts (rare under realistic placement) are added on CPU after.
    // IEEE add is commutative, so this is bit-exact vs the split+combine path.
    static const bool kLagunaGpuRemap = (std::getenv("DFLASH_LAGUNA_GPU_REMAP") != nullptr);
    if (kLagunaGpuRemap) {
        // Reactive bounded expert cache: pull selected cold experts into spare
        // GPU slots (LRU evict) so the unified GPU FFN serves them on-die. After
        // warmup the working set is resident and the CPU cold path is rarely taken.
        static const bool kCache = (std::getenv("DFLASH_LAGUNA_EXPERT_CACHE") != nullptr);
        if (kCache && storage.cache_slots > 0) {
            for (int i = 0; i < n_selected; ++i)
                moe_hybrid_cache_swap_in(storage, selected_ids[i], gpu_backend);
        }
        // Cold residue after caching (equals the original cold set when disabled).
        std::vector<int32_t> cache_cold_ids; std::vector<float> cache_cold_w;
        for (int i = 0; i < n_selected; ++i) {
            const int32_t g = selected_ids[i];
            if (g < 0 || g >= (int)storage.hot_local_by_global.size()) continue;
            if (storage.hot_local_by_global[(size_t)g] < 0) {
                const int32_t cl = storage.cold_local_by_global[(size_t)g];
                if (cl >= 0) { cache_cold_ids.push_back(cl); cache_cold_w.push_back(selected_weights[i]); }
            }
        }
        const bool has_cold2 = !cache_cold_ids.empty();
        const int n_cold2 = (int)cache_cold_ids.size();
        if (!storage.hot_graph.valid() || storage.hot_graph.n_hot != n_selected ||
            !storage.hot_graph.global_ids) {
            build_cached_hot_graph(storage.hot_graph, gpu_backend,
                                   storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                                   desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                   desc, n_embd, cfg.n_ff_exp, n_selected,
                                   CachedHotGraphOptions{cfg.swiglu_clamp, true, cfg.n_expert});
        }
        if (!storage.hot_graph.valid() || !storage.hot_graph.global_ids ||
            !storage.hot_graph.hot_local_lut || !storage.hot_graph.valid_lut ||
            !storage.hot_graph.residual_in) {
            return false;
        }
        {
            std::vector<int32_t> lut((size_t)cfg.n_expert);
            std::vector<float>   vlut((size_t)cfg.n_expert);
            for (int e = 0; e < cfg.n_expert; ++e) {
                const int32_t l = storage.hot_local_by_global[(size_t)e];
                lut[(size_t)e]  = (l >= 0) ? l : 0;
                vlut[(size_t)e] = (l >= 0) ? 1.0f : 0.0f;
            }
            ggml_backend_tensor_set(storage.hot_graph.hot_local_lut, lut.data(), 0,
                                    sizeof(int32_t) * (size_t)cfg.n_expert);
            ggml_backend_tensor_set(storage.hot_graph.valid_lut, vlut.data(), 0,
                                    sizeof(float) * (size_t)cfg.n_expert);
        }
        ggml_backend_tensor_copy(ffn_post_gpu, storage.hot_graph.inp);
        ggml_backend_tensor_copy(ffn_residual_gpu, storage.hot_graph.residual_in);
        ggml_backend_tensor_set(storage.hot_graph.global_ids, selected_ids, 0,
                                sizeof(int32_t) * (size_t)n_selected);
        ggml_backend_tensor_set(storage.hot_graph.raw_weights, selected_weights, 0,
                                sizeof(float) * (size_t)n_selected);
        if (ggml_backend_graph_compute(gpu_backend, storage.hot_graph.gf) != GGML_STATUS_SUCCESS) {
            return false;
        }
        if (!has_cold2) {
            ggml_backend_tensor_copy(storage.hot_graph.output, gpu_state.act_cur);
            return true;
        }
        std::vector<float> post_host((size_t)n_embd);
        ggml_backend_tensor_get(ffn_post_gpu, post_host.data(), 0, sizeof(float) * (size_t)n_embd);
        std::vector<float> cold_res((size_t)n_embd);
        if (expert_compute && expert_layer) {
            if (!expert_compute->compute(*expert_layer, post_host.data(),
                                       cache_cold_ids.data(), cache_cold_w.data(),
                                       n_cold2, n_embd, cfg.n_ff_exp,
                                       cold_res.data())) {
                return false;
            }
        } else {
            if (!storage.cold_graph.valid() || storage.cold_graph.n_hot != n_cold2) {
                build_cached_cold_graph(storage.cold_graph, cpu_backend,
                                        storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
                                        desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                        n_embd, cfg.n_ff_exp, n_cold2, cfg.swiglu_clamp);
            }
            if (!storage.cold_graph.valid() || storage.cold_graph.n_hot != n_cold2) return false;
            ggml_backend_tensor_set(storage.cold_graph.inp, post_host.data(), 0, sizeof(float) * (size_t)n_embd);
            ggml_backend_tensor_set(storage.cold_graph.ids, cache_cold_ids.data(), 0, sizeof(int32_t) * (size_t)n_cold2);
            ggml_backend_tensor_set(storage.cold_graph.weights, cache_cold_w.data(), 0, sizeof(float) * (size_t)n_cold2);
            if (ggml_backend_graph_compute(cpu_backend, storage.cold_graph.gf) != GGML_STATUS_SUCCESS) return false;
            ggml_backend_tensor_get(storage.cold_graph.output, cold_res.data(), 0, sizeof(float) * (size_t)n_embd);
        }
        std::vector<float> hot_res((size_t)n_embd);
        ggml_backend_tensor_get(storage.hot_graph.output, hot_res.data(), 0, sizeof(float) * (size_t)n_embd);
        for (int i = 0; i < n_embd; ++i) hot_res[(size_t)i] += cold_res[(size_t)i];
        ggml_backend_tensor_set(gpu_state.act_cur, hot_res.data(), 0, sizeof(float) * (size_t)n_embd);
        return true;
    }

    // ── GPU→GPU: copy residual to combine input ──
    ggml_backend_tensor_copy(ffn_residual_gpu, gpu_state.combine.residual_in);

    // ── Prepare hot graph input via GPU→GPU copy ──
    bool hot_async_launched = false;
    if (has_hot || has_shared) {
        if (!storage.hot_graph.valid() || storage.hot_graph.n_hot != n_hot) {
            build_cached_hot_graph(storage.hot_graph, gpu_backend,
                                   storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                                   desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                   desc, n_embd, cfg.n_ff_exp, n_hot,
                                   CachedHotGraphOptions{cfg.swiglu_clamp});
        }
        if (storage.hot_graph.valid() && storage.hot_graph.n_hot == n_hot) {
            // GPU→GPU copy: ffn_post → hot_graph.inp (no PCIe!)
            ggml_backend_tensor_copy(ffn_post_gpu, storage.hot_graph.inp);
            if (storage.hot_graph.ids && has_hot) {
                ggml_backend_tensor_set(storage.hot_graph.ids, hot_ids.data(), 0,
                                        sizeof(int32_t) * (size_t)n_hot);
            }
            if (storage.hot_graph.weights && has_hot) {
                ggml_backend_tensor_set(storage.hot_graph.weights, hot_weights.data(), 0,
                                        sizeof(float) * (size_t)n_hot);
            }
        }
    }

    ggml_backend_t cold_backend = storage.cold_backend ? storage.cold_backend : cpu_backend;
    const bool cold_on_gpu = storage.cold_backend_kind == MoeHybridColdBackend::Gpu &&
                             cold_backend == gpu_backend;

    // ── If CPU cold is needed, read ffn_post before launching hot async ──
    std::vector<float> post_host;
    if (has_cold && !cold_on_gpu) {
        post_host.resize((size_t)n_embd);
        ggml_backend_tensor_get(ffn_post_gpu, post_host.data(), 0, sizeof(float) * (size_t)n_embd);
    }

    // ── Launch hot async (GPU kernels in flight) ──
    if ((has_hot || has_shared) && storage.hot_graph.valid() && storage.hot_graph.n_hot == n_hot) {
        ggml_backend_graph_compute_async(gpu_backend, storage.hot_graph.gf);
        hot_async_launched = true;
        if (cold_on_gpu) {
            ggml_backend_synchronize(gpu_backend);
        }
    }

    // ── Cold path on selected cold backend ──
    std::vector<float> cold_result;
    if (has_cold) {
        if (expert_compute && expert_layer && !cold_on_gpu) {
            cold_result.resize((size_t)n_embd);
            if (!expert_compute->compute(*expert_layer, post_host.data(),
                                       cold_ids.data(), cold_weights.data(),
                                       n_cold, n_embd, cfg.n_ff_exp,
                                       cold_result.data())) {
                if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
                return false;
            }
        } else {
            if (!storage.down_cold) {
                if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
                return false;
            }
            if (!storage.cold_graph.valid() || storage.cold_graph.n_hot != n_cold) {
                build_cached_cold_graph(storage.cold_graph, cold_backend,
                                        storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
                                        desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                        n_embd, cfg.n_ff_exp, n_cold, cfg.swiglu_clamp);
            }
            if (!storage.cold_graph.valid() || storage.cold_graph.n_hot != n_cold) {
                if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
                return false;
            }
            if (cold_on_gpu) {
                ggml_backend_tensor_copy(ffn_post_gpu, storage.cold_graph.inp);
            } else {
                ggml_backend_tensor_set(storage.cold_graph.inp, post_host.data(), 0,
                                        sizeof(float) * (size_t)n_embd);
            }
            ggml_backend_tensor_set(storage.cold_graph.ids, cold_ids.data(), 0,
                                    sizeof(int32_t) * (size_t)n_cold);
            ggml_backend_tensor_set(storage.cold_graph.weights, cold_weights.data(), 0,
                                    sizeof(float) * (size_t)n_cold);
            auto st = ggml_backend_graph_compute(cold_backend, storage.cold_graph.gf);
            if (st != GGML_STATUS_SUCCESS) {
                if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
                return false;
            }
            if (!cold_on_gpu) {
                cold_result.resize((size_t)n_embd);
                ggml_backend_tensor_get(storage.cold_graph.output, cold_result.data(), 0,
                                        sizeof(float) * (size_t)n_embd);
            }
        }
    }

    // ── Sync hot graph and copy output to combine.hot_in ──
    if (hot_async_launched) {
        ggml_backend_synchronize(gpu_backend);
        // GPU→GPU: hot output → combine.hot_in
        ggml_backend_tensor_copy(storage.hot_graph.output, gpu_state.combine.hot_in);
    } else {
        std::vector<float> zeros((size_t)n_embd, 0.0f);
        ggml_backend_tensor_set(gpu_state.combine.hot_in, zeros.data(), 0,
                                sizeof(float) * (size_t)n_embd);
    }

    // ── Upload/copy cold result (or zeros) to combine.cold_in ──
    if (has_cold) {
        if (cold_on_gpu) {
            ggml_backend_tensor_copy(storage.cold_graph.output, gpu_state.combine.cold_in);
        } else {
            ggml_backend_tensor_set(gpu_state.combine.cold_in, cold_result.data(), 0,
                                    sizeof(float) * (size_t)n_embd);
        }
    } else {
        std::vector<float> zeros((size_t)n_embd, 0.0f);
        ggml_backend_tensor_set(gpu_state.combine.cold_in, zeros.data(), 0,
                                sizeof(float) * (size_t)n_embd);
    }

    // ── Compute residual combine on GPU: output = residual + hot + cold ──
    auto st = ggml_backend_graph_compute(gpu_backend, gpu_state.combine.gf);
    if (st != GGML_STATUS_SUCCESS) return false;

    // ── Copy combine output to persistent act_cur (GPU→GPU) ──
    ggml_backend_tensor_copy(gpu_state.combine.output, gpu_state.act_cur);

    return true;
}

}  // namespace dflash::common
