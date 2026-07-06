#include <cstdlib>
#include "moe_hybrid_ffn_eval.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>

namespace dflash::common {

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

static uint64_t elapsed_us(HybridClock::time_point start, HybridClock::time_point end) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
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
    ggml_tensor ** out_routed)
{
    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, inp, n_embd, 1, n_tokens);
    ggml_tensor * gu = nullptr;
    if (gate_up_tensor) {
        ggml_tensor * gate_up_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, gate_up_tensor, cur_3d, sel), gate_up_scale);
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
            ggml_mul_mat_id(ctx, gate_tensor, cur_3d, sel), gate_scale);
        ggml_tensor * up_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, up_tensor, cur_3d, sel), up_scale);
        gu = swiglu_maybe_clamped(ctx, gate_e, up_e, swiglu_clamp);
    }

    ggml_tensor * experts = apply_scale2(ctx,
        ggml_mul_mat_id(ctx, down_tensor, gu, sel), down_scale);

    // Weight and sum over experts: [n_embd, n_used, n_tokens] * [1, n_used, n_tokens]
    ggml_tensor * w_view = ggml_reshape_3d(ctx, wts, 1, n_used, n_tokens);
    experts = ggml_mul(ctx, experts, w_view);

    ggml_tensor * sum_shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * moe_sum = ggml_repeat_back(ctx, experts, sum_shape);
    *out_routed = ggml_reshape_2d(ctx, moe_sum, n_embd, n_tokens);
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
    MoeHybridLayerStorage & storage,
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
        storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
        desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
        out.inp, out.sel, out.wts, n_embd, n_ff_exp, n_used, n_tokens, cfg.swiglu_clamp, &routed);

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
    MoeHybridLayerStorage & storage,
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
        out.inp, out.sel, out.wts, n_embd, n_ff_exp, n_used, n_tokens, cfg.swiglu_clamp, &routed);
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
        if (!storage.hot_graph.valid() || storage.hot_graph.n_hot != n_hot_graph) {
            if (telemetry) telemetry->hot_graph_builds++;
            const auto graph_build_t0 = HybridClock::now();
            build_cached_hot_graph(storage.hot_graph, gpu_backend,
                                   storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                                   desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                   desc, cfg.n_embd, cfg.n_ff_exp, n_hot_graph,
                                   CachedHotGraphOptions{cfg.swiglu_clamp});
            if (telemetry) telemetry->hot_graph_build_us += elapsed_us(graph_build_t0, HybridClock::now());
        } else if (telemetry) {
            telemetry->hot_graph_hits++;
        }
        if (storage.hot_graph.valid() && storage.hot_graph.n_hot == n_hot_graph) {
            const auto input_t0 = HybridClock::now();
            ggml_backend_tensor_set(storage.hot_graph.inp, cur_host, 0, sizeof(float) * (size_t)cfg.n_embd);
            if (storage.hot_graph.ids && n_hot_graph > 0) {
                ggml_backend_tensor_set(storage.hot_graph.ids, hot_ids_data, 0, sizeof(int32_t) * (size_t)n_hot_graph);
            }
            if (storage.hot_graph.weights && n_hot_graph > 0) {
                ggml_backend_tensor_set(storage.hot_graph.weights, hot_weights_data, 0, sizeof(float) * (size_t)n_hot_graph);
            }
            if (telemetry) telemetry->hot_input_us += elapsed_us(input_t0, HybridClock::now());
            // Launch GPU async — kernel runs while the cold backend runs.
            const auto compute_t0 = HybridClock::now();
            ggml_backend_graph_compute_async(gpu_backend, storage.hot_graph.gf);
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
        if (!storage.cold_graph.valid() || storage.cold_graph.n_hot != n_cold_graph) {
            if (telemetry) telemetry->cold_graph_builds++;
            const auto graph_build_t0 = HybridClock::now();
            build_cached_cold_graph(storage.cold_graph, cold_backend,
                                    storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
                                    desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                    cfg.n_embd, cfg.n_ff_exp, n_cold_graph, cfg.swiglu_clamp);
            if (telemetry) telemetry->cold_graph_build_us += elapsed_us(graph_build_t0, HybridClock::now());
        } else if (telemetry) {
            telemetry->cold_graph_hits++;
        }
        if (storage.cold_graph.valid() && storage.cold_graph.n_hot == n_cold_graph) {
            const auto input_t0 = HybridClock::now();
            ggml_backend_tensor_set(storage.cold_graph.inp, cur_host, 0, sizeof(float) * (size_t)cfg.n_embd);
            ggml_backend_tensor_set(storage.cold_graph.ids, cold_ids_data, 0, sizeof(int32_t) * (size_t)n_cold_graph);
            ggml_backend_tensor_set(storage.cold_graph.weights, cold_weights_data, 0, sizeof(float) * (size_t)n_cold_graph);
            if (telemetry) telemetry->cold_input_us += elapsed_us(input_t0, HybridClock::now());
            const auto compute_t0 = HybridClock::now();
            auto st = ggml_backend_graph_compute(cold_backend, storage.cold_graph.gf);
            if (telemetry) telemetry->cold_compute_us += elapsed_us(compute_t0, HybridClock::now());
            if (st != GGML_STATUS_SUCCESS) {
                if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
                if (err) *err = "cached cold graph compute failed";
                return false;
            }
            const auto read_t0 = HybridClock::now();
            cold.resize((size_t)cfg.n_embd);
            ggml_backend_tensor_get(storage.cold_graph.output, cold.data(), 0, sizeof(float) * (size_t)cfg.n_embd);
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
    if ((has_hot || has_shared) && storage.hot_graph.valid() && storage.hot_graph.n_hot == n_hot_graph) {
        const auto read_t0 = HybridClock::now();
        ggml_backend_synchronize(gpu_backend);
        hot_and_shared.resize((size_t)cfg.n_embd);
        ggml_backend_tensor_get(storage.hot_graph.output, hot_and_shared.data(), 0, sizeof(float) * (size_t)cfg.n_embd);
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
    bool                            skip_cold = false) {

    const auto ffn_wall_t0 = HybridClock::now();
    const auto partition_t0 = HybridClock::now();
    const int n_embd = cfg.n_embd;
    const int n_used = cfg.n_expert_used;
    const int n_ff_exp = cfg.n_ff_exp;
    out.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
    if (n_tokens <= 0) return true;

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
        CachedHotBatchedGraph * cg = nullptr;
        bool cg_ok = true;
        if (fp_has_cold) {
            cg = &storage.cold_batched_mixed[n_tokens];
            cg_ok = (cg->valid() && cg->n_tokens == n_tokens)
                || build_cached_cold_batched_graph(*cg, cpu_backend, storage, desc, cfg, n_tokens);
        }

        if (hg_ok && cg_ok) {
            // Hot (GPU, async): shared expert + routed hot (zero-weight dummy slots
            // keep an all-cold batch's shared-expert contribution).
            ggml_backend_tensor_set(hg.inp, cur_host, 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
            ggml_backend_tensor_set(hg.sel, hot_sel.data(), 0, sizeof(int32_t) * (size_t)total_slots);
            ggml_backend_tensor_set(hg.wts, hot_wts.data(), 0, sizeof(float) * (size_t)total_slots);
            ggml_backend_graph_compute_async(gpu_backend, hg.gf);

            std::vector<float> cold_partial;
            if (cg) {
                cold_partial.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
                ggml_backend_tensor_set(cg->inp, cur_host, 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
                ggml_backend_tensor_set(cg->sel, cold_sel.data(), 0, sizeof(int32_t) * (size_t)total_slots);
                ggml_backend_tensor_set(cg->wts, cold_wts.data(), 0, sizeof(float) * (size_t)total_slots);
                if (ggml_backend_graph_compute(cpu_backend, cg->gf) != GGML_STATUS_SUCCESS) {
                    ggml_backend_synchronize(gpu_backend);
                    if (err) *err = "batched cold cached compute failed";
                    return false;
                }
                ggml_backend_tensor_get(cg->output, cold_partial.data(), 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
            }

            ggml_backend_synchronize(gpu_backend);
            ggml_backend_tensor_get(hg.output, out.data(), 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
            if (cg) {
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
    std::vector<int32_t> cold_sel(total_slots);
    for (int i = 0; i < total_slots; ++i) cold_sel[i] = i % std::max(1, (int)(storage.down_cold ? storage.down_cold->ne[2] : 1));
    std::vector<float>   cold_wts(total_slots, 0.0f);
    bool has_hot = false, has_cold = false;
    int hot_selected = 0;
    int cold_selected = 0;

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
            }
        }
    }
    const auto partition_t1 = HybridClock::now();
    if (telemetry) {
        telemetry->partition_us += elapsed_us(partition_t0, partition_t1);
        telemetry->hot_selected += hot_selected;
        telemetry->cold_selected += cold_selected;
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
    if (has_hot || has_shared) {
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
                inp, sel, wts, n_embd, n_ff_exp, n_used, n_tokens, cfg.swiglu_clamp, &routed);
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

        ggml_backend_tensor_set(inp, cur_host, 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
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

        ggml_tensor * inp = ggml_new_tensor_2d(cold_ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp);
        ggml_tensor * sel = ggml_new_tensor_2d(cold_ctx, GGML_TYPE_I32, n_used, n_tokens);
        ggml_set_input(sel);
        ggml_tensor * wts = ggml_new_tensor_2d(cold_ctx, GGML_TYPE_F32, n_used, n_tokens);
        ggml_set_input(wts);

        ggml_tensor * cold_routed = nullptr;
        build_batched_routed_graph(cold_ctx,
            storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
            desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
            inp, sel, wts, n_embd, n_ff_exp, n_used, n_tokens, cfg.swiglu_clamp, &cold_routed);

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

        ggml_backend_tensor_set(inp, cur_host, 0, sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
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

        ggml_backend_tensor_get(cold_routed, cold_partial.data(), 0,
            sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
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

    const int cold_batch = moe_hybrid_expert_compute_ipc_batch_limit(n_tokens);
    std::vector<int> token_group;
    std::vector<float> group_input;
    std::vector<int32_t> group_ids;
    std::vector<float> group_wts;
    std::vector<float> group_output;
    for (int n_cold = 1; n_cold <= max_cold_selected; ++n_cold) {
        token_group.clear();
        for (int t = 0; t < n_tokens; ++t) {
            if (cold_counts[(size_t)t] == n_cold) {
                token_group.push_back(t);
            }
        }
        for (size_t base = 0; base < token_group.size(); base += (size_t)cold_batch) {
            const int tc = (int)std::min((size_t)cold_batch, token_group.size() - base);
            bool contiguous_tokens = true;
            for (int gi = 1; gi < tc; ++gi) {
                if (token_group[base + (size_t)gi] !=
                    token_group[base] + gi) {
                    contiguous_tokens = false;
                    break;
                }
            }
            const int first_token = token_group[base];
            const float * compute_input = contiguous_tokens
                ? cur_host + (size_t)first_token * (size_t)n_embd
                : nullptr;
            float * compute_output = contiguous_tokens
                ? out.data() + (size_t)first_token * (size_t)n_embd
                : nullptr;
            if (!contiguous_tokens) {
                group_input.resize((size_t)tc * (size_t)n_embd);
                compute_input = group_input.data();
            }
            group_ids.resize((size_t)tc * (size_t)n_cold);
            group_wts.resize((size_t)tc * (size_t)n_cold);
            if (!contiguous_tokens) {
                group_output.resize((size_t)tc * (size_t)n_embd);
                compute_output = group_output.data();
            }

            for (int gi = 0; gi < tc; ++gi) {
                const int t = token_group[base + (size_t)gi];
                if (!contiguous_tokens) {
                    std::memcpy(group_input.data() + (size_t)gi * (size_t)n_embd,
                                cur_host + (size_t)t * (size_t)n_embd,
                                sizeof(float) * (size_t)n_embd);
                }
                for (int i = 0; i < n_cold; ++i) {
                    const size_t src = (size_t)t * (size_t)n_used + (size_t)i;
                    const size_t dst = (size_t)gi * (size_t)n_cold + (size_t)i;
                    group_ids[dst] = cold_sel[src];
                    group_wts[dst] = cold_wts[src];
                }
            }

            if (!expert_compute->compute_batch(*expert_layer,
                                               compute_input,
                                               group_ids.data(),
                                               group_wts.data(),
                                               tc, n_cold,
                                               n_embd, n_ff_exp,
                                               compute_output)) {
                if (err) *err = "hybrid batched remote cold compute failed";
                return false;
            }

            if (!contiguous_tokens) {
                for (int gi = 0; gi < tc; ++gi) {
                    const int t = token_group[base + (size_t)gi];
                    std::memcpy(out.data() + (size_t)t * (size_t)n_embd,
                                group_output.data() + (size_t)gi * (size_t)n_embd,
                                sizeof(float) * (size_t)n_embd);
                }
            }
        }
    }
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
        inp, sel, wts, n_embd, n_ff_exp, n_used, n_tokens, cfg.swiglu_clamp, &routed);

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
    MoeHybridFfnTelemetry *         telemetry) {
    if (telemetry) *telemetry = {};
    const int n_hot_stack = storage.gate_up_hot ? (int)storage.gate_up_hot->ne[2]
                          : storage.gate_hot    ? (int)storage.gate_hot->ne[2]
                          : 0;
    const int n_cold_stack = storage.gate_up_cold ? (int)storage.gate_up_cold->ne[2]
                            : storage.gate_cold    ? (int)storage.gate_cold->ne[2]
                            : 0;
    const bool cold_on_gpu = storage.cold_backend_kind == MoeHybridColdBackend::Gpu;
    const int hot_sub_batch =
        std::min(mmq_safe_sub_batch(), moe_hybrid_prefill_hot_sub_batch_limit());
    if (!mmq_full_batch_ok(cfg, n_tokens)
        && ((n_hot_stack > 0 && n_hot_stack < cfg.n_expert) ||
            (cold_on_gpu && n_cold_stack > 0 && n_cold_stack < cfg.n_expert))
        && n_tokens > hot_sub_batch) {
        const int n_embd = cfg.n_embd;
        const int n_used = cfg.n_expert_used;
        out.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
        if (expert_compute && expert_layer &&
            parse_moe_expert_compute_ipc_mode() == MoeExpertComputeIpcMode::Batched) {
            std::vector<float> hot_partial((size_t)n_embd * (size_t)n_tokens, 0.0f);
            std::vector<float> cold_partial;
            std::string cold_err;
            const auto cold_t0 = HybridClock::now();
            auto cold_future = std::async(std::launch::async, [&]() {
                return eval_moe_hybrid_remote_cold_batched(
                    cfg, storage, cur_host, selected_ids, selected_weights,
                    n_tokens, cold_partial, &cold_err,
                    expert_compute, expert_layer);
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
