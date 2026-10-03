// Bitwise regression test for the fused concurrent-verify Gated DeltaNet
// route (Qwen3.5-family DFlash2 serving).
//
// The op-by-op verify graph derives the gates with sigmoid / add / softplus /
// mul over contiguity copies of the stacked projection, gathers the conv
// history and the recurrent base state for each mapped slot, transposes and
// concatenates the conv window, runs the tree conv, and runs the plain
// recurrence over the gathered copy while capturing the replay log. The fused
// route reads strided gate views, assembles and convolves the window in one
// kernel (ggml_ssm_conv_tree_step), and runs the recurrence against the
// mapped base slab in place (ggml_gated_delta_net_mapped_verify) with the
// gates derived in the kernel. This test builds both on the HIP backend and
// requires bit-identical attention output, replay log, conv output and conv
// window, and that the fused route leaves the persistent states untouched.
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

uint32_t lcg_state = 0x12345678u;
float lcg_uniform() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (float) ((lcg_state >> 8) & 0xFFFFFF) / (float) 0x1000000 * 2.0f - 1.0f;
}

struct Cfg {
    int H_k, H_v, S_v, T, S, n_slots, K, Z;
};

struct Inputs {
    std::vector<float> qkvz, conv_w, conv_state, ssm_state, ba, gate_ba;
    std::vector<int32_t> state_ids, parent_ids;
};

struct Outputs {
    std::vector<float> attn, replay, conv_out, conv_input, conv_state_after, ssm_state_after;
};

void fill(std::vector<float> & v, float scale) {
    for (float & x : v) x = lcg_uniform() * scale;
}

Inputs make_inputs(const Cfg & c) {
    const int C = 2 * c.H_k * c.S_v + c.H_v * c.S_v;
    Inputs in;
    in.qkvz.resize((size_t) (c.Z + C) * c.T * c.S);
    in.conv_w.resize((size_t) c.K * C);
    in.conv_state.resize((size_t) (c.K - 1) * C * c.n_slots);
    in.ssm_state.resize((size_t) c.S_v * c.S_v * c.H_v * c.n_slots);
    in.ba.resize((size_t) 2 * c.H_v * c.T * c.S);
    in.gate_ba.resize((size_t) 2 * c.H_v);
    fill(in.qkvz, 1.5f);
    fill(in.conv_w, 0.7f);
    fill(in.conv_state, 1.2f);
    fill(in.ssm_state, 0.2f);
    fill(in.ba, 3.0f);
    for (int h = 0; h < c.H_v; ++h) {
        in.gate_ba[(size_t) h] = lcg_uniform() * 0.5f;                  // dt_bias
        in.gate_ba[(size_t) c.H_v + h] = -0.5f - 1.5f * std::fabs(lcg_uniform()); // A
    }
    // Distinct slots per sequence, deliberately not in order.
    in.state_ids.resize((size_t) c.S);
    for (int s = 0; s < c.S; ++s) in.state_ids[(size_t) s] = c.n_slots - 1 - s;
    // Trees: token 0 is a root, later tokens pick any earlier token or -1.
    in.parent_ids.resize((size_t) c.T * c.S);
    for (int s = 0; s < c.S; ++s) {
        for (int t = 0; t < c.T; ++t) {
            int parent;
            if (t == 0) parent = -1;
            else if (t % 3 == 0) parent = -1;
            else if (t % 4 == 1) parent = t - 1;
            else parent = (int) ((lcg_uniform() + 1.0f) * 0.5f * t) % t;
            in.parent_ids[(size_t) s * c.T + t] = parent;
        }
    }
    return in;
}

// Builds either route and computes it. Returns false on failure.
bool run(ggml_backend_t backend, const Cfg & c, const Inputs & in, bool fused, Outputs & out) {
    const int C = 2 * c.H_k * c.S_v + c.H_v * c.S_v;
    const int S_k = c.S_v;
    ggml_init_params params{};
    params.mem_size = 32 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * qkvz = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.Z + C, (int64_t) c.T * c.S);
    ggml_tensor * conv_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.K, C);
    ggml_tensor * conv_state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K - 1, C, c.n_slots);
    ggml_tensor * ssm_state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.S_v, c.H_v, c.n_slots);
    ggml_tensor * ba = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * c.H_v, (int64_t) c.T * c.S);
    ggml_tensor * gate_ba = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * c.H_v);
    ggml_tensor * state_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.S);
    ggml_tensor * parent_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.T, c.S);
    for (ggml_tensor * t : {qkvz, conv_w, conv_state, ssm_state, ba, gate_ba, state_ids, parent_ids}) {
        ggml_set_input(t);
    }

    const size_t f32 = sizeof(float);
    // Row slice of the stacked projection: [C, T, S] with the stacked row stride.
    ggml_tensor * x = ggml_view_3d(ctx, qkvz, C, c.T, c.S, qkvz->nb[1], qkvz->nb[1] * c.T, (size_t) c.Z * f32);

    ggml_tensor * beta = nullptr;
    ggml_tensor * g = nullptr;
    ggml_tensor * conv_out = nullptr;
    ggml_tensor * conv_input = nullptr;
    if (!fused) {
        ggml_tensor * beta_2d = ggml_cont(ctx, ggml_view_2d(ctx, ba, c.H_v, (int64_t) c.T * c.S, ba->nb[1], 0));
        ggml_tensor * alpha_2d = ggml_cont(ctx, ggml_view_2d(ctx, ba, c.H_v, (int64_t) c.T * c.S, ba->nb[1], (size_t) c.H_v * f32));
        beta = ggml_reshape_4d(ctx, beta_2d, 1, c.H_v, c.T, c.S);
        ggml_tensor * alpha = ggml_reshape_3d(ctx, alpha_2d, c.H_v, c.T, c.S);
        ggml_tensor * dt_bias = ggml_view_1d(ctx, gate_ba, c.H_v, 0);
        ggml_tensor * A = ggml_view_1d(ctx, gate_ba, c.H_v, (size_t) c.H_v * f32);
        beta = ggml_sigmoid(ctx, beta);
        alpha = ggml_add(ctx, alpha, dt_bias);
        alpha = ggml_softplus(ctx, alpha);
        g = ggml_mul(ctx, alpha, A);
        g = ggml_reshape_4d(ctx, g, 1, c.H_v, c.T, c.S);

        ggml_tensor * all_conv = ggml_reshape_2d(ctx, conv_state, (int64_t) (c.K - 1) * C, c.n_slots);
        ggml_tensor * gathered = ggml_get_rows(ctx, all_conv, state_ids);
        ggml_tensor * conv_states_r = ggml_reshape_3d(ctx, gathered, c.K - 1, C, c.S);
        ggml_tensor * x_T = ggml_transpose(ctx, x);
        conv_input = ggml_concat(ctx, conv_states_r, x_T, 0);
        conv_out = ggml_silu(ctx, ggml_ssm_conv_tree(ctx, conv_input, conv_w, parent_ids));
    } else {
        beta = ggml_view_4d(ctx, ba, 1, c.H_v, c.T, c.S, f32, ba->nb[1], ba->nb[1] * c.T, 0);
        g = ggml_view_4d(ctx, ba, 1, c.H_v, c.T, c.S, f32, ba->nb[1], ba->nb[1] * c.T, (size_t) c.H_v * f32);
        ggml_tensor * packed = ggml_ssm_conv_tree_step(ctx, x, conv_w, conv_state, state_ids, parent_ids);
        const int64_t window = c.K - 1 + c.T;
        conv_out = ggml_view_3d(ctx, packed, C, c.T, c.S, (size_t) C * f32, (size_t) C * c.T * f32, 0);
        conv_input = ggml_view_3d(ctx, packed, window, C, c.S, (size_t) window * f32, (size_t) window * C * f32,
                                  (size_t) C * c.T * c.S * f32);
    }

    // q/k l2 norm and v slice, identical in both routes.
    const size_t row = (size_t) C * f32;
    ggml_tensor * qk_c = ggml_view_4d(ctx, conv_out, S_k, 2 * c.H_k, c.T, c.S, (size_t) S_k * f32, row, row * c.T, 0);
    ggml_tensor * qk_n = ggml_l2_norm(ctx, qk_c, 1e-6f);
    ggml_tensor * q_c = ggml_view_4d(ctx, qk_n, S_k, c.H_k, c.T, c.S, qk_n->nb[1], qk_n->nb[2], qk_n->nb[3], 0);
    ggml_tensor * k_c = ggml_view_4d(ctx, qk_n, S_k, c.H_k, c.T, c.S, qk_n->nb[1], qk_n->nb[2], qk_n->nb[3],
                                     (size_t) c.H_k * S_k * f32);
    ggml_tensor * v_c = ggml_view_4d(ctx, conv_out, c.S_v, c.H_v, c.T, c.S, (size_t) c.S_v * f32, row, row * c.T,
                                     (size_t) 2 * c.H_k * S_k * f32);

    ggml_tensor * result = nullptr;
    if (!fused) {
        ggml_tensor * all_ssm = ggml_reshape_2d(ctx, ssm_state, (int64_t) c.S_v * c.S_v * c.H_v, c.n_slots);
        ggml_tensor * gathered = ggml_get_rows(ctx, all_ssm, state_ids);
        ggml_tensor * s = ggml_reshape_4d(ctx, gathered, c.S_v, c.S_v, c.H_v, c.S);
        result = ggml_gated_delta_net(ctx, q_c, k_c, v_c, g, beta, s);
    } else {
        result = ggml_gated_delta_net_mapped_verify(ctx, q_c, k_c, v_c, g, beta, ssm_state, state_ids);
        ggml_gated_delta_net_set_raw_gates(result, gate_ba);
    }
    ggml_tensor * replay = ggml_gated_delta_net_capture_replay_log(ctx, result);
    ggml_tensor * attn = ggml_view_4d(ctx, result, c.S_v, c.H_v, c.T, c.S, (size_t) c.S_v * f32,
                                      (size_t) c.S_v * c.H_v * f32, (size_t) c.S_v * c.H_v * c.T * f32, 0);
    for (ggml_tensor * t : {attn, replay, conv_out, conv_input}) ggml_set_output(t);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
    for (ggml_tensor * t : {attn, replay, conv_out, conv_input}) ggml_build_forward_expand(graph, t);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    auto set = [](ggml_tensor * t, const void * data, size_t n) { ggml_backend_tensor_set(t, data, 0, n); };
    set(qkvz, in.qkvz.data(), in.qkvz.size() * f32);
    set(conv_w, in.conv_w.data(), in.conv_w.size() * f32);
    set(conv_state, in.conv_state.data(), in.conv_state.size() * f32);
    set(ssm_state, in.ssm_state.data(), in.ssm_state.size() * f32);
    set(ba, in.ba.data(), in.ba.size() * f32);
    set(gate_ba, in.gate_ba.data(), in.gate_ba.size() * f32);
    set(state_ids, in.state_ids.data(), in.state_ids.size() * sizeof(int32_t));
    set(parent_ids, in.parent_ids.data(), in.parent_ids.size() * sizeof(int32_t));

    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(backend);
    if (ok) {
        auto get = [](ggml_tensor * t, std::vector<float> & v) {
            v.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        };
        get(attn, out.attn);
        get(replay, out.replay);
        get(conv_out, out.conv_out);
        get(conv_input, out.conv_input);
        get(conv_state, out.conv_state_after);
        get(ssm_state, out.ssm_state_after);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

size_t bit_mismatches(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return a.size() + b.size();
    size_t mm = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++mm;
    }
    return mm;
}

bool check(ggml_backend_t backend, const Cfg & c) {
    lcg_state = 0x9e37u ^ (uint32_t) (c.H_v * 131 + c.T * 17 + c.S * 3 + c.S_v);
    const Inputs in = make_inputs(c);
    Outputs a, b;
    if (!run(backend, c, in, false, a) || !run(backend, c, in, true, b)) {
        std::printf("FAIL H_k=%d H_v=%d S_v=%d T=%d S=%d: compute failed\n", c.H_k, c.H_v, c.S_v, c.T, c.S);
        return false;
    }
    const size_t mm_attn = bit_mismatches(a.attn, b.attn);
    const size_t mm_replay = bit_mismatches(a.replay, b.replay);
    const size_t mm_conv = bit_mismatches(a.conv_out, b.conv_out);
    const size_t mm_window = bit_mismatches(a.conv_input, b.conv_input);
    const size_t mm_cstate = bit_mismatches(in.conv_state, b.conv_state_after);
    const size_t mm_sstate = bit_mismatches(in.ssm_state, b.ssm_state_after);
    // Sanity: the outputs are not degenerate.
    size_t nonzero = 0;
    for (float v : b.attn) nonzero += v != 0.0f;
    const bool ok = mm_attn == 0 && mm_replay == 0 && mm_conv == 0 && mm_window == 0 && mm_cstate == 0 &&
                    mm_sstate == 0 && nonzero > b.attn.size() / 2;
    std::printf("%s H_k=%d H_v=%d S_v=%d T=%d S=%d slots=%d: attn %zu values %zu mismatches, replay %zu/%zu, "
                "conv %zu/%zu, window %zu/%zu, conv_state changed %zu, ssm_state changed %zu\n",
                ok ? "PASS" : "FAIL", c.H_k, c.H_v, c.S_v, c.T, c.S, c.n_slots, b.attn.size(), mm_attn,
                mm_replay, b.replay.size(), mm_conv, b.conv_out.size(), mm_window, b.conv_input.size(),
                mm_cstate, mm_sstate);
    return ok;
}

} // namespace

int main() {
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) {
        std::fprintf(stderr, "failed to query HIP device 0\n");
        return 1;
    }
    ggml_backend_t hip = ggml_backend_cuda_init(0);
    if (!hip) {
        std::fprintf(stderr, "failed to initialize the HIP backend\n");
        return 1;
    }
    const bool previous_graphs = ggml_backend_cuda_set_graphs_disabled_override(true);
    bool ok = true;
    // Qwen3.8-27B verify shape: 16 k-heads, 48 v-heads of 128, 8-token
    // fixed-width blocks on 4 lanes, and a stacked projection row stride.
    ok = check(hip, {16, 48, 128, 8, 4, 6, 4, 64}) && ok;
    // Non-grouped recurrence kernel (S_v != 128), odd widths, dense rows.
    ok = check(hip, {8, 12, 64, 5, 3, 4, 4, 0}) && ok;
    // Single lane, single token.
    ok = check(hip, {16, 48, 128, 1, 1, 2, 4, 64}) && ok;
    // Wider trees than the fixed block.
    ok = check(hip, {16, 48, 128, 12, 2, 5, 4, 64}) && ok;
    ggml_backend_cuda_set_graphs_disabled_override(previous_graphs);
    ggml_backend_free(hip);
    std::printf("%s qwen35 gdn verify fusion\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
