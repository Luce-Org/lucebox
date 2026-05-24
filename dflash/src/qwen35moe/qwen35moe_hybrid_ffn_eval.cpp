#include "qwen35moe_hybrid_ffn_eval.h"

#include "qwen35_ops.h"

#include "ggml-alloc.h"

#include <chrono>
#include <cmath>

namespace dflash::common {

namespace {

using HybridClock = std::chrono::steady_clock;

static uint64_t elapsed_us(HybridClock::time_point start, HybridClock::time_point end) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

// Build a cached FFN graph for the hot+shared path with a fixed n_hot.
static bool build_cached_hot_graph(
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
    const bool has_shared = (L.ffn_up_shexp && L.ffn_gate_shexp && L.ffn_down_shexp);
    if (has_shared) {
        ggml_tensor * sh_gate = apply_scale2(out.ctx, ggml_mul_mat(out.ctx, L.ffn_gate_shexp, out.inp), L.ffn_gate_shexp_s);
        ggml_tensor * sh_up   = apply_scale2(out.ctx, ggml_mul_mat(out.ctx, L.ffn_up_shexp,   out.inp), L.ffn_up_shexp_s);
        ggml_tensor * sh_gu   = ggml_swiglu_split(out.ctx, sh_gate, sh_up);
        shared = apply_scale2(out.ctx, ggml_mul_mat(out.ctx, L.ffn_down_shexp, sh_gu), L.ffn_down_shexp_s);
        if (L.ffn_gate_inp_shexp) {
            ggml_tensor * shared_gate = apply_scale2(out.ctx,
                ggml_mul_mat(out.ctx, L.ffn_gate_inp_shexp, out.inp), L.ffn_gate_inp_shexp_s);
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

// Build a cached FFN graph for the cold (CPU) routed subset.
static bool build_cached_cold_graph(
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

static bool run_shared_ffn_gpu(ggml_backend_t backend,
                               const TargetLayer & L,
                               int n_embd,
                               const float * cur_host,
                               std::vector<float> & out,
                               std::string * err) {
    out.assign((size_t)n_embd, 0.0f);
    if (!L.ffn_up_shexp || !L.ffn_gate_shexp || !L.ffn_down_shexp) {
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

    ggml_tensor * sh_gate = apply_scale2(ctx, ggml_mul_mat(ctx, L.ffn_gate_shexp, inp), L.ffn_gate_shexp_s);
    ggml_tensor * sh_up   = apply_scale2(ctx, ggml_mul_mat(ctx, L.ffn_up_shexp,   inp), L.ffn_up_shexp_s);
    ggml_tensor * sh_gu   = ggml_swiglu_split(ctx, sh_gate, sh_up);
    ggml_tensor * shared  = apply_scale2(ctx, ggml_mul_mat(ctx, L.ffn_down_shexp, sh_gu), L.ffn_down_shexp_s);
    if (L.ffn_gate_inp_shexp) {
        ggml_tensor * shared_gate = apply_scale2(ctx,
            ggml_mul_mat(ctx, L.ffn_gate_inp_shexp, inp), L.ffn_gate_inp_shexp_s);
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
// Eliminates one graph compute phase per layer vs separate run_routed_subset + run_shared_ffn_gpu.
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
    const TargetLayer & L,
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
    const bool has_shared = (L.ffn_up_shexp && L.ffn_gate_shexp && L.ffn_down_shexp);
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
        ggml_tensor * sh_gate = apply_scale2(ctx, ggml_mul_mat(ctx, L.ffn_gate_shexp, inp), L.ffn_gate_shexp_s);
        ggml_tensor * sh_up   = apply_scale2(ctx, ggml_mul_mat(ctx, L.ffn_up_shexp,   inp), L.ffn_up_shexp_s);
        ggml_tensor * sh_gu   = ggml_swiglu_split(ctx, sh_gate, sh_up);
        shared = apply_scale2(ctx, ggml_mul_mat(ctx, L.ffn_down_shexp, sh_gu), L.ffn_down_shexp_s);
        if (L.ffn_gate_inp_shexp) {
            ggml_tensor * shared_gate = apply_scale2(ctx,
                ggml_mul_mat(ctx, L.ffn_gate_inp_shexp, inp), L.ffn_gate_inp_shexp_s);
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

} // namespace

bool eval_qwen35moe_reference_ffn_single(
    ggml_backend_t         gpu_backend,
    const TargetWeights &  w,
    const TargetLayer &    L,
    const float *          cur_host,
    const int32_t *        selected_ids,
    const float *          selected_weights,
    int                    n_selected,
    std::vector<float> &   out,
    std::string *          err) {
    // Reference path: fused hot+shared in one graph (same as hybrid but all experts on GPU)
    if (!run_hot_and_shared_ffn_gpu(gpu_backend,
                                    L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps, L.ffn_gate_up_exps,
                                    L.ffn_gate_exps_s, L.ffn_up_exps_s, L.ffn_down_exps_s, L.ffn_gate_up_exps_s,
                                    L, w.n_embd, w.n_ff_exp,
                                    cur_host, selected_ids, selected_weights, n_selected,
                                    out, err)) {
        return false;
    }
    return true;
}

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
    Qwen35MoeHybridFfnTelemetry *       telemetry,
    std::string *                       err) {
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

    // ── Cold path: use cached graph if n_cold matches ──
    const auto cold_t0 = HybridClock::now();
    if (!cold_ids.empty()) {
        const int n_cold = (int)cold_ids.size();
        // Lazily build cached cold graph on first use with this n_cold
        if (!storage.cold_graph.valid() || storage.cold_graph.n_hot != n_cold) {
            build_cached_cold_graph(storage.cold_graph, cpu_backend,
                                    storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
                                    L.ffn_gate_exps_s, L.ffn_up_exps_s, L.ffn_down_exps_s, L.ffn_gate_up_exps_s,
                                    w.n_embd, w.n_ff_exp, n_cold);
        }
        if (storage.cold_graph.valid() && storage.cold_graph.n_hot == n_cold) {
            // Use cached graph
            ggml_backend_tensor_set(storage.cold_graph.inp, cur_host, 0, sizeof(float) * (size_t)w.n_embd);
            ggml_backend_tensor_set(storage.cold_graph.ids, cold_ids.data(), 0, sizeof(int32_t) * (size_t)n_cold);
            ggml_backend_tensor_set(storage.cold_graph.weights, cold_weights.data(), 0, sizeof(float) * (size_t)n_cold);
            auto st = ggml_backend_graph_compute(cpu_backend, storage.cold_graph.gf);
            if (st != GGML_STATUS_SUCCESS) {
                if (err) *err = "cached cold graph compute failed";
                return false;
            }
            cold.resize((size_t)w.n_embd);
            ggml_backend_tensor_get(storage.cold_graph.output, cold.data(), 0, sizeof(float) * (size_t)w.n_embd);
        } else {
            // Fallback to dynamic
            if (!run_routed_subset(cpu_backend,
                                   storage.gate_cold, storage.up_cold, storage.down_cold, storage.gate_up_cold,
                                   L.ffn_gate_exps_s, L.ffn_up_exps_s, L.ffn_down_exps_s, L.ffn_gate_up_exps_s,
                                   w.n_embd, w.n_ff_exp,
                                   cur_host, cold_ids.data(), cold_weights.data(), n_cold, cold, err)) {
                return false;
            }
        }
    } else {
        cold.assign((size_t)w.n_embd, 0.0f);
    }
    const auto cold_t1 = HybridClock::now();
    if (telemetry) {
        telemetry->cold_us = cold_ids.empty() ? 0 : elapsed_us(cold_t0, cold_t1);
    }

    // ── Hot + Shared path: use cached graph if n_hot matches ──
    const auto hot_t0 = HybridClock::now();
    const int n_hot = (int)hot_ids.size();
    const bool has_hot = (n_hot > 0);
    const bool has_shared = (L.ffn_up_shexp && L.ffn_gate_shexp && L.ffn_down_shexp);

    if (!has_hot && !has_shared) {
        hot_and_shared.assign((size_t)w.n_embd, 0.0f);
    } else {
        // Lazily build cached hot graph on first use
        if (!storage.hot_graph.valid() || storage.hot_graph.n_hot != n_hot) {
            build_cached_hot_graph(storage.hot_graph, gpu_backend,
                                   storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                                   L.ffn_gate_exps_s, L.ffn_up_exps_s, L.ffn_down_exps_s, L.ffn_gate_up_exps_s,
                                   L, w.n_embd, w.n_ff_exp, n_hot);
        }
        if (storage.hot_graph.valid() && storage.hot_graph.n_hot == n_hot) {
            // Use cached graph
            ggml_backend_tensor_set(storage.hot_graph.inp, cur_host, 0, sizeof(float) * (size_t)w.n_embd);
            if (storage.hot_graph.ids && has_hot) {
                ggml_backend_tensor_set(storage.hot_graph.ids, hot_ids.data(), 0, sizeof(int32_t) * (size_t)n_hot);
            }
            if (storage.hot_graph.weights && has_hot) {
                ggml_backend_tensor_set(storage.hot_graph.weights, hot_weights.data(), 0, sizeof(float) * (size_t)n_hot);
            }
            auto st = ggml_backend_graph_compute(gpu_backend, storage.hot_graph.gf);
            if (st != GGML_STATUS_SUCCESS) {
                if (err) *err = "cached hot graph compute failed";
                return false;
            }
            hot_and_shared.resize((size_t)w.n_embd);
            ggml_backend_tensor_get(storage.hot_graph.output, hot_and_shared.data(), 0, sizeof(float) * (size_t)w.n_embd);
        } else {
            // Fallback to dynamic
            if (!run_hot_and_shared_ffn_gpu(gpu_backend,
                                            storage.gate_hot, storage.up_hot, storage.down_hot, storage.gate_up_hot,
                                            L.ffn_gate_exps_s, L.ffn_up_exps_s, L.ffn_down_exps_s, L.ffn_gate_up_exps_s,
                                            L, w.n_embd, w.n_ff_exp,
                                            cur_host,
                                            hot_ids.empty() ? nullptr : hot_ids.data(),
                                            hot_weights.empty() ? nullptr : hot_weights.data(),
                                            n_hot, hot_and_shared, err)) {
                return false;
            }
        }
    }
    const auto hot_t1 = HybridClock::now();
    if (telemetry) {
        telemetry->hot_us = elapsed_us(hot_t0, hot_t1);
        telemetry->shared_us = 0;
    }

    const auto combine_t0 = HybridClock::now();
    out.assign((size_t)w.n_embd, 0.0f);
    for (int i = 0; i < w.n_embd; ++i) {
        out[(size_t)i] = hot_and_shared[(size_t)i] + cold[(size_t)i];
    }
    const auto combine_t1 = HybridClock::now();
    if (telemetry) {
        telemetry->combine_us = elapsed_us(combine_t0, combine_t1);
        telemetry->ffn_wall_us = elapsed_us(ffn_wall_t0, combine_t1);
    }
    return true;
}

}  // namespace dflash::common
