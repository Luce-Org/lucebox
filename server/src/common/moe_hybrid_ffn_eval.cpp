#include "moe_hybrid_ffn_eval.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>

namespace dflash::common {

namespace {

// NVFP4 scale2: if weight has a per-tensor scale, multiply the matmul result
// by that scale. No-op when scale==1.0f (non-NVFP4 models).
inline ggml_tensor * apply_scale2(ggml_context * ctx, ggml_tensor * mm_result, float scale) {
    if (scale == 1.0f) return mm_result;
    return ggml_scale(ctx, mm_result, scale);
}

using HybridClock = std::chrono::steady_clock;

static uint64_t elapsed_us(HybridClock::time_point start, HybridClock::time_point end) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
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
        gu = ggml_swiglu_split(ctx, gate_e, up_e);
    } else {
        ggml_tensor * gate_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, gate_tensor, cur_3d, ids), gate_scale);
        ggml_tensor * up_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, up_tensor, cur_3d, ids), up_scale);
        gu = ggml_swiglu_split(ctx, gate_e, up_e);
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

    ggml_tensor * sh_gate = apply_scale2(ctx, ggml_mul_mat(ctx, desc.ffn_gate_shexp, inp), desc.ffn_gate_shexp_s);
    ggml_tensor * sh_up   = apply_scale2(ctx, ggml_mul_mat(ctx, desc.ffn_up_shexp,   inp), desc.ffn_up_shexp_s);
    ggml_tensor * sh_gu   = ggml_swiglu_split(ctx, sh_gate, sh_up);
    ggml_tensor * shared  = apply_scale2(ctx, ggml_mul_mat(ctx, desc.ffn_down_shexp, sh_gu), desc.ffn_down_shexp_s);
    if (desc.ffn_gate_inp_shexp) {
        ggml_tensor * shared_gate = apply_scale2(ctx,
            ggml_mul_mat(ctx, desc.ffn_gate_inp_shexp, inp), desc.ffn_gate_inp_shexp_s);
        shared_gate = ggml_sigmoid(ctx, shared_gate);
        shared = ggml_mul(ctx, shared, shared_gate);
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
            gu = ggml_swiglu_split(ctx, gate_e, up_e);
        } else {
            ggml_tensor * gate_e = apply_scale2(ctx,
                ggml_mul_mat_id(ctx, gate_tensor, cur_3d, ids_tensor), gate_scale);
            ggml_tensor * up_e = apply_scale2(ctx,
                ggml_mul_mat_id(ctx, up_tensor, cur_3d, ids_tensor), up_scale);
            gu = ggml_swiglu_split(ctx, gate_e, up_e);
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

    ggml_tensor * shared = nullptr;
    if (has_shared) {
        ggml_tensor * sh_gate = apply_scale2(ctx, ggml_mul_mat(ctx, desc.ffn_gate_shexp, inp), desc.ffn_gate_shexp_s);
        ggml_tensor * sh_up   = apply_scale2(ctx, ggml_mul_mat(ctx, desc.ffn_up_shexp,   inp), desc.ffn_up_shexp_s);
        ggml_tensor * sh_gu   = ggml_swiglu_split(ctx, sh_gate, sh_up);
        shared = apply_scale2(ctx, ggml_mul_mat(ctx, desc.ffn_down_shexp, sh_gu), desc.ffn_down_shexp_s);
        if (desc.ffn_gate_inp_shexp) {
            ggml_tensor * shared_gate = apply_scale2(ctx,
                ggml_mul_mat(ctx, desc.ffn_gate_inp_shexp, inp), desc.ffn_gate_inp_shexp_s);
            shared_gate = ggml_sigmoid(ctx, shared_gate);
            shared = ggml_mul(ctx, shared, shared_gate);
        }
    }

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
        gu = ggml_swiglu_split(ctx, gate_e, up_e);
    } else {
        ggml_tensor * gate_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, gate_tensor, cur_3d, sel), gate_scale);
        ggml_tensor * up_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, up_tensor, cur_3d, sel), up_scale);
        gu = ggml_swiglu_split(ctx, gate_e, up_e);
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

} // namespace (anon)

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
    int n_hot) {

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
            gu = ggml_swiglu_split(out.ctx, gate_e, up_e);
        } else {
            ggml_tensor * gate_e = apply_scale2(out.ctx,
                ggml_mul_mat_id(out.ctx, gate_tensor, cur_3d, out.ids), gate_scale);
            ggml_tensor * up_e = apply_scale2(out.ctx,
                ggml_mul_mat_id(out.ctx, up_tensor, cur_3d, out.ids), up_scale);
            gu = ggml_swiglu_split(out.ctx, gate_e, up_e);
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

    ggml_tensor * shared = nullptr;
    const bool has_shared = (desc.ffn_up_shexp && desc.ffn_gate_shexp && desc.ffn_down_shexp);
    if (has_shared) {
        ggml_tensor * sh_gate = apply_scale2(out.ctx, ggml_mul_mat(out.ctx, desc.ffn_gate_shexp, out.inp), desc.ffn_gate_shexp_s);
        ggml_tensor * sh_up   = apply_scale2(out.ctx, ggml_mul_mat(out.ctx, desc.ffn_up_shexp,   out.inp), desc.ffn_up_shexp_s);
        ggml_tensor * sh_gu   = ggml_swiglu_split(out.ctx, sh_gate, sh_up);
        shared = apply_scale2(out.ctx, ggml_mul_mat(out.ctx, desc.ffn_down_shexp, sh_gu), desc.ffn_down_shexp_s);
        if (desc.ffn_gate_inp_shexp) {
            ggml_tensor * shared_gate = apply_scale2(out.ctx,
                ggml_mul_mat(out.ctx, desc.ffn_gate_inp_shexp, out.inp), desc.ffn_gate_inp_shexp_s);
            shared_gate = ggml_sigmoid(out.ctx, shared_gate);
            shared = ggml_mul(out.ctx, shared, shared_gate);
        }
    }

    if (routed && shared) {
        out.output = ggml_add(out.ctx, routed, shared);
    } else if (routed) {
        out.output = routed;
    } else {
        out.output = shared;
    }
    if (!out.output) { out.free(); return false; }

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
    int n_cold) {

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
        gu = ggml_swiglu_split(out.ctx, gate_e, up_e);
    } else {
        ggml_tensor * gate_e = apply_scale2(out.ctx,
            ggml_mul_mat_id(out.ctx, gate_tensor, cur_3d, out.ids), gate_scale);
        ggml_tensor * up_e = apply_scale2(out.ctx,
            ggml_mul_mat_id(out.ctx, up_tensor, cur_3d, out.ids), up_scale);
        gu = ggml_swiglu_split(out.ctx, gate_e, up_e);
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

    // ── Hot + Shared path on GPU ──
    bool hot_async_launched = false;
    const auto hot_t0 = HybridClock::now();
    if (!has_hot && !has_shared) {
        hot_and_shared.assign((size_t)cfg.n_embd, 0.0f);
    } else {
        // Lazily build cached hot graph on first use
        if (!storage.hot_graph.valid() || storage.hot_graph.n_hot != n_hot) {
            build_cached_hot_graph(storage.hot_graph, gpu_backend,
                                   storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                                   desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                   desc, cfg.n_embd, cfg.n_ff_exp, n_hot);
        }
        if (storage.hot_graph.valid() && storage.hot_graph.n_hot == n_hot) {
            ggml_backend_tensor_set(storage.hot_graph.inp, cur_host, 0, sizeof(float) * (size_t)cfg.n_embd);
            if (storage.hot_graph.ids && has_hot) {
                ggml_backend_tensor_set(storage.hot_graph.ids, hot_ids.data(), 0, sizeof(int32_t) * (size_t)n_hot);
            }
            if (storage.hot_graph.weights && has_hot) {
                ggml_backend_tensor_set(storage.hot_graph.weights, hot_weights.data(), 0, sizeof(float) * (size_t)n_hot);
            }
            // Launch GPU async — kernel runs while we do cold on CPU
            ggml_backend_graph_compute_async(gpu_backend, storage.hot_graph.gf);
            hot_async_launched = true;
        } else {
            // Fallback: sync compute (no overlap)
            if (!run_hot_and_shared_ffn_gpu(gpu_backend,
                                            storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                                            desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                            desc, cfg.n_embd, cfg.n_ff_exp,
                                            cur_host,
                                            hot_ids.empty() ? nullptr : hot_ids.data(),
                                            hot_weights.empty() ? nullptr : hot_weights.data(),
                                            n_hot, hot_and_shared, err)) {
                return false;
            }
        }
    }

    // ── Cold path on CPU (overlaps with GPU kernels in flight) ──
    const auto cold_t0 = HybridClock::now();
    if (has_cold) {
        if (!storage.cold_graph.valid() || storage.cold_graph.n_hot != n_cold) {
            build_cached_cold_graph(storage.cold_graph, cpu_backend,
                                    storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
                                    desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                    cfg.n_embd, cfg.n_ff_exp, n_cold);
        }
        if (storage.cold_graph.valid() && storage.cold_graph.n_hot == n_cold) {
            ggml_backend_tensor_set(storage.cold_graph.inp, cur_host, 0, sizeof(float) * (size_t)cfg.n_embd);
            ggml_backend_tensor_set(storage.cold_graph.ids, cold_ids.data(), 0, sizeof(int32_t) * (size_t)n_cold);
            ggml_backend_tensor_set(storage.cold_graph.weights, cold_weights.data(), 0, sizeof(float) * (size_t)n_cold);
            auto st = ggml_backend_graph_compute(cpu_backend, storage.cold_graph.gf);
            if (st != GGML_STATUS_SUCCESS) {
                if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
                if (err) *err = "cached cold graph compute failed";
                return false;
            }
            cold.resize((size_t)cfg.n_embd);
            ggml_backend_tensor_get(storage.cold_graph.output, cold.data(), 0, sizeof(float) * (size_t)cfg.n_embd);
        } else {
            if (!run_routed_subset(cpu_backend,
                                   storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
                                   desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                   cfg.n_embd, cfg.n_ff_exp,
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
    if ((has_hot || has_shared) && storage.hot_graph.valid() && storage.hot_graph.n_hot == n_hot) {
        ggml_backend_synchronize(gpu_backend);
        hot_and_shared.resize((size_t)cfg.n_embd);
        ggml_backend_tensor_get(storage.hot_graph.output, hot_and_shared.data(), 0, sizeof(float) * (size_t)cfg.n_embd);
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
        gu = ggml_swiglu_split(ctx, gate_e, up_e);
    } else {
        ggml_tensor * gate_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, desc.ffn_gate_exps, cur_3d, sel), desc.ffn_gate_exps_s);
        ggml_tensor * up_e = apply_scale2(ctx,
            ggml_mul_mat_id(ctx, desc.ffn_up_exps, cur_3d, sel), desc.ffn_up_exps_s);
        gu = ggml_swiglu_split(ctx, gate_e, up_e);
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
    if (desc.ffn_up_shexp && desc.ffn_gate_shexp && desc.ffn_down_shexp) {
        ggml_tensor * sh_gate = apply_scale2(ctx,
            ggml_mul_mat(ctx, desc.ffn_gate_shexp, inp), desc.ffn_gate_shexp_s);
        ggml_tensor * sh_up = apply_scale2(ctx,
            ggml_mul_mat(ctx, desc.ffn_up_shexp, inp), desc.ffn_up_shexp_s);
        ggml_tensor * sh_gu = ggml_swiglu_split(ctx, sh_gate, sh_up);
        ggml_tensor * shared = apply_scale2(ctx,
            ggml_mul_mat(ctx, desc.ffn_down_shexp, sh_gu), desc.ffn_down_shexp_s);
        if (desc.ffn_gate_inp_shexp) {
            ggml_tensor * shared_gate = apply_scale2(ctx,
                ggml_mul_mat(ctx, desc.ffn_gate_inp_shexp, inp), desc.ffn_gate_inp_shexp_s);
            shared_gate = ggml_sigmoid(ctx, shared_gate);
            shared = ggml_mul(ctx, shared, shared_gate);
        }
        combined = ggml_add(ctx, routed, shared);
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
    ggml_gallocr_t *                p_cold_alloc) {

    const int n_embd = cfg.n_embd;
    const int n_used = cfg.n_expert_used;
    const int n_ff_exp = cfg.n_ff_exp;
    out.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
    if (n_tokens <= 0) return true;

    // ── Step 1: Partition routing into hot and cold ──
    // Dummy slots use weight 0.0 and are distributed evenly across all experts
    // to avoid pathological routing imbalance that triggers OOB in MMQ stream-k.
    const int total_slots = n_used * n_tokens;
    const int n_hot_stack = storage.gate_up_hot ? (int)storage.gate_up_hot->ne[2]
                          : storage.gate_hot    ? (int)storage.gate_hot->ne[2]
                          : 1;
    std::vector<int32_t> hot_sel(total_slots);
    for (int i = 0; i < total_slots; ++i) hot_sel[i] = i % n_hot_stack;
    std::vector<float>   hot_wts(total_slots, 0.0f);
    std::vector<int32_t> cold_sel(total_slots);
    for (int i = 0; i < total_slots; ++i) cold_sel[i] = i % std::max(1, (int)(storage.down_cold ? storage.down_cold->ne[2] : 1));
    std::vector<float>   cold_wts(total_slots, 0.0f);
    bool has_hot = false, has_cold = false;

    for (int i = 0; i < total_slots; ++i) {
        const int32_t gid = selected_ids[i];
        if (gid < 0 || gid >= (int32_t)storage.hot_local_by_global.size()) continue;
        const int32_t hot_lid = storage.hot_local_by_global[(size_t)gid];
        if (hot_lid >= 0) {
            hot_sel[i] = hot_lid;
            hot_wts[i] = selected_weights[i];
            has_hot = true;
        } else {
            const int32_t cold_lid = storage.cold_local_by_global[(size_t)gid];
            if (cold_lid >= 0) {
                cold_sel[i] = cold_lid;
                cold_wts[i] = selected_weights[i];
                has_cold = true;
            }
        }
    }

    // ── Step 2: Build and run hot GPU graph (includes shared expert always) ──
    std::vector<float> hot_partial((size_t)n_embd * (size_t)n_tokens, 0.0f);
    bool hot_async_launched = false;

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
                inp, sel, wts, n_embd, n_ff_exp, n_used, n_tokens, &routed);
        }

        // Shared expert (always on GPU)
        ggml_tensor * combined = routed;
        if (has_shared) {
            ggml_tensor * sh_gate = apply_scale2(hot_ctx,
                ggml_mul_mat(hot_ctx, desc.ffn_gate_shexp, inp), desc.ffn_gate_shexp_s);
            ggml_tensor * sh_up = apply_scale2(hot_ctx,
                ggml_mul_mat(hot_ctx, desc.ffn_up_shexp, inp), desc.ffn_up_shexp_s);
            ggml_tensor * sh_gu = ggml_swiglu_split(hot_ctx, sh_gate, sh_up);
            ggml_tensor * shared = apply_scale2(hot_ctx,
                ggml_mul_mat(hot_ctx, desc.ffn_down_shexp, sh_gu), desc.ffn_down_shexp_s);
            if (desc.ffn_gate_inp_shexp) {
                ggml_tensor * shared_gate = apply_scale2(hot_ctx,
                    ggml_mul_mat(hot_ctx, desc.ffn_gate_inp_shexp, inp), desc.ffn_gate_inp_shexp_s);
                shared_gate = ggml_sigmoid(hot_ctx, shared_gate);
                shared = ggml_mul(hot_ctx, shared, shared_gate);
            }
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
    }

    // ── Step 3: Build and run cold CPU graph (overlaps with GPU) ──
    std::vector<float> cold_partial((size_t)n_embd * (size_t)n_tokens, 0.0f);

    if (has_cold) {
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
            inp, sel, wts, n_embd, n_ff_exp, n_used, n_tokens, &cold_routed);

        ggml_cgraph * cold_gf = ggml_new_graph_custom(cold_ctx, 4096, false);
        ggml_set_output(cold_routed);
        ggml_build_forward_expand(cold_gf, cold_routed);
        ggml_gallocr_t cold_alloc;
        if (p_cold_alloc) {
            if (!*p_cold_alloc)
                *p_cold_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu_backend));
            cold_alloc = *p_cold_alloc;
        } else {
            cold_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu_backend));
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

        // Run CPU synchronously (overlaps with GPU async)
        auto st = ggml_backend_graph_compute(cpu_backend, cold_gf);
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
        if (!p_cold_alloc) ggml_gallocr_free(cold_alloc);
        ggml_free(cold_ctx);
    }

    // ── Step 4: Sync GPU and read hot result ──
    if (hot_async_launched) {
        ggml_backend_synchronize(gpu_backend);
        ggml_backend_tensor_get(hot_output, hot_partial.data(), 0,
            sizeof(float) * (size_t)n_embd * (size_t)n_tokens);
    }
    if (!p_hot_alloc && hot_alloc) ggml_gallocr_free(hot_alloc);
    if (hot_ctx) ggml_free(hot_ctx);

    // ── Step 5: Merge hot + cold ──
    const size_t total_floats = (size_t)n_embd * (size_t)n_tokens;
    for (size_t i = 0; i < total_floats; ++i) {
        out[i] = hot_partial[i] + cold_partial[i];
    }

    return true;
}

// ── GPU-Resident Residual State ──

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
    int                             n_selected) {

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

    // ── GPU→GPU: copy residual to combine input ──
    ggml_backend_tensor_copy(ffn_residual_gpu, gpu_state.combine.residual_in);

    // ── Prepare hot graph input via GPU→GPU copy ──
    bool hot_async_launched = false;
    if (has_hot || has_shared) {
        if (!storage.hot_graph.valid() || storage.hot_graph.n_hot != n_hot) {
            build_cached_hot_graph(storage.hot_graph, gpu_backend,
                                   storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                                   desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                   desc, n_embd, cfg.n_ff_exp, n_hot);
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

    // ── If cold needed, read ffn_post to CPU BEFORE launching hot async ──
    std::vector<float> post_host;
    if (has_cold) {
        post_host.resize((size_t)n_embd);
        ggml_backend_tensor_get(ffn_post_gpu, post_host.data(), 0, sizeof(float) * (size_t)n_embd);
    }

    // ── Launch hot async (GPU kernels in flight) ──
    if ((has_hot || has_shared) && storage.hot_graph.valid() && storage.hot_graph.n_hot == n_hot) {
        ggml_backend_graph_compute_async(gpu_backend, storage.hot_graph.gf);
        hot_async_launched = true;
    }

    // ── Cold path on CPU (overlaps with hot GPU kernels) ──
    std::vector<float> cold_result;
    if (has_cold) {
        if (!storage.cold_graph.valid() || storage.cold_graph.n_hot != n_cold) {
            build_cached_cold_graph(storage.cold_graph, cpu_backend,
                                    storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
                                    desc.ffn_gate_exps_s, desc.ffn_up_exps_s, desc.ffn_down_exps_s, desc.ffn_gate_up_exps_s,
                                    n_embd, cfg.n_ff_exp, n_cold);
        }
        if (storage.cold_graph.valid() && storage.cold_graph.n_hot == n_cold) {
            ggml_backend_tensor_set(storage.cold_graph.inp, post_host.data(), 0,
                                    sizeof(float) * (size_t)n_embd);
            ggml_backend_tensor_set(storage.cold_graph.ids, cold_ids.data(), 0,
                                    sizeof(int32_t) * (size_t)n_cold);
            ggml_backend_tensor_set(storage.cold_graph.weights, cold_weights.data(), 0,
                                    sizeof(float) * (size_t)n_cold);
            auto st = ggml_backend_graph_compute(cpu_backend, storage.cold_graph.gf);
            if (st != GGML_STATUS_SUCCESS) {
                if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
                return false;
            }
            cold_result.resize((size_t)n_embd);
            ggml_backend_tensor_get(storage.cold_graph.output, cold_result.data(), 0,
                                    sizeof(float) * (size_t)n_embd);
        } else {
            if (hot_async_launched) ggml_backend_synchronize(gpu_backend);
            return false;
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

    // ── Upload cold result (or zeros) to combine.cold_in ──
    if (has_cold) {
        ggml_backend_tensor_set(gpu_state.combine.cold_in, cold_result.data(), 0,
                                sizeof(float) * (size_t)n_embd);
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
