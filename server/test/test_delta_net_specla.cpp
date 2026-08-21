// GPU parity test for the SpecLA topology-masked delta-net verify builder
// (src/delta_net_specla.cpp) against the fused sequential ggml_gated_delta_net
// kernel, which advances the recurrent state token by token and is the
// numerical ground truth for the recurrence.
//
// Checks, per shape/topology case:
//   1. per-node outputs of build_delta_net_specla match the fused kernel
//      (chain cases use ggml_gated_delta_net, tree cases the _tree variant);
//   2. host-side DeltaConstruct over the captured factors —
//         S_A = exp(g⁺_A) S0 + Σ_{u ∈ path(A)} exp(g⁺_A − g⁺_u) k_u ⊗ ṽ_u
//      — matches the fused kernel's per-token intermediate state at EVERY
//      possible accepted endpoint A (every prefix of a chain, every node of a
//      tree). This is the correctness contract the factor-based accepted-state
//      commit (SPECLA.md §1-§3) relies on;
//   3. g⁺ equals the host-computed ancestor path sum of g.

#include "delta_net_specla.h"
#include "specla_commit_cuda.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"  // ggml_backend_cuda_init; maps to HIP under GGML_USE_HIP

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using dflash::common::build_delta_net_specla;
using dflash::common::fill_specla_masks;
using dflash::common::make_specla_hld_schedule;

static int failures = 0;

#define CHECK_MSG(cond, ...) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n"); \
        failures++; \
    } \
} while (0)

namespace {

struct CaseInputs {
    int S = 0, H = 0, n = 0;
    std::vector<int32_t> parents;             // parents[0] = -1, parents[t] < t
    std::vector<float> q, k, v, g, b, s0;     // layouts as fed to the builders
};

CaseInputs make_inputs(int S, int H, int n, const std::vector<int32_t> & parents,
                       unsigned seed) {
    CaseInputs in;
    in.S = S; in.H = H; in.n = n; in.parents = parents;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);

    auto fill_unit_heads = [&](std::vector<float> & dst) {
        // [S, H, n]: one l2-normalized S-vector per (head, token), like the
        // post-l2_norm q/k the real block feeds the recurrence.
        dst.resize((size_t)S * H * n);
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < H; h++) {
                float norm2 = 0.0f;
                float * vec = dst.data() + (size_t)t * S * H + (size_t)h * S;
                for (int s = 0; s < S; s++) { vec[s] = nd(rng); norm2 += vec[s] * vec[s]; }
                const float inv = 1.0f / std::sqrt(norm2 + 1e-6f);
                for (int s = 0; s < S; s++) vec[s] *= inv;
            }
        }
    };
    fill_unit_heads(in.q);
    fill_unit_heads(in.k);

    in.v.resize((size_t)S * H * n);
    for (auto & x : in.v) x = 0.5f * nd(rng);
    in.g.resize((size_t)H * n);               // [1, H, n] — log-decay, negative
    for (auto & x : in.g) x = -(0.05f + 1.5f * ud(rng));
    in.b.resize((size_t)H * n);               // [1, H, n] — sigmoid-like in (0,1)
    for (auto & x : in.b) x = 0.1f + 0.8f * ud(rng);
    in.s0.resize((size_t)S * S * H);
    for (auto & x : in.s0) x = 0.1f * nd(rng);
    return in;
}

struct RefOutputs {
    std::vector<float> attn;    // [S*H per token][n] token-major
    std::vector<float> inter;   // [S*S*H per token][n] state after node t
};

struct SpecLAOutputs {
    std::vector<float> out;     // [S, H, n] — same layout as RefOutputs::attn
    std::vector<float> v_new;   // [n, S_v, 1, H]
    std::vector<float> g_ps;    // [n, 1, 1, H]
};

struct PendingFactors {
    int count = 0;
    std::vector<float> k;
    std::vector<float> delta;
    std::vector<float> g;
    std::vector<float> state_after;
};

PendingFactors make_pending_factors(const CaseInputs & in) {
    PendingFactors out;
    out.count = in.n;
    out.k = in.k;
    out.g = in.g;
    out.delta.resize(in.v.size());
    out.state_after = in.s0;
    for (int t = 0; t < in.n; ++t) {
        for (int h = 0; h < in.H; ++h) {
            const size_t th = (size_t)t*in.H + h;
            const float decay = std::exp(in.g[th]);
            const float beta = in.b[th];
            const float * k = in.k.data() + th*in.S;
            const float * v = in.v.data() + th*in.S;
            float * delta = out.delta.data() + th*in.S;
            float * state = out.state_after.data() + (size_t)h*in.S*in.S;
            for (int col = 0; col < in.S; ++col) {
                float kv = 0.0f;
                for (int row = 0; row < in.S; ++row) {
                    kv += state[(size_t)col*in.S + row]*k[row];
                }
                delta[col] = (v[col] - decay*kv)*beta;
            }
            for (int col = 0; col < in.S; ++col) {
                for (int row = 0; row < in.S; ++row) {
                    float & cell = state[(size_t)col*in.S + row];
                    cell = std::fma(k[row], delta[col], decay*cell);
                }
            }
        }
    }
    return out;
}

float max_abs_diff(const std::vector<float> & a,
                   const std::vector<float> & b);
std::vector<int32_t> chain_parents(int n);
std::vector<int32_t> random_tree_parents(int n, unsigned seed);

struct GraphEnv {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t galloc = nullptr;

    explicit GraphEnv(size_t n_tensors = 512) {
        ggml_init_params ip{};
        ip.mem_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        gf = ggml_new_graph_custom(ctx, 2048, false);
    }
    ~GraphEnv() {
        if (galloc) ggml_gallocr_free(galloc);
        if (ctx) ggml_free(ctx);
    }
    bool alloc_and_run(ggml_backend_t backend) {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) return false;
        return true;
    }
};

void set_f32(ggml_tensor * t, const std::vector<float> & host) {
    GGML_ASSERT((size_t)ggml_nelements(t) == host.size());
    ggml_backend_tensor_set(t, host.data(), 0, host.size() * sizeof(float));
}

std::vector<float> get_f32(const ggml_tensor * t, size_t off_elems, size_t n_elems) {
    std::vector<float> out(n_elems);
    ggml_backend_tensor_get(t, out.data(), off_elems * sizeof(float),
                            n_elems * sizeof(float));
    return out;
}

// Reference pass: fused sequential kernel, chain (plain op) or tree variant.
bool run_reference(ggml_backend_t backend, const CaseInputs & in, bool tree_op,
                   RefOutputs & ref) {
    const int S = in.S, H = in.H, n = in.n;
    GraphEnv env;
    ggml_context * ctx = env.ctx;

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n, 1);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n, 1);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n, 1);
    ggml_tensor * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, 1);
    for (ggml_tensor * t : {q, k, v, g, b, s}) ggml_set_input(t);

    ggml_tensor * parent_ids = nullptr;
    ggml_tensor * result = nullptr;
    if (tree_op) {
        parent_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_set_input(parent_ids);
        result = ggml_gated_delta_net_tree(ctx, q, k, v, g, b, s, parent_ids);
    } else {
        result = ggml_gated_delta_net(ctx, q, k, v, g, b, s);
    }
    // Intermediates deliberately kept (no set_skip_intermediate): they are the
    // per-token ground-truth states DeltaConstruct is validated against.
    ggml_set_output(result);
    ggml_build_forward_expand(env.gf, result);

    if (!env.alloc_and_run(backend)) return false;
    set_f32(q, in.q); set_f32(k, in.k); set_f32(v, in.v);
    set_f32(g, in.g); set_f32(b, in.b); set_f32(s, in.s0);
    if (parent_ids) {
        ggml_backend_tensor_set(parent_ids, in.parents.data(), 0,
                                sizeof(int32_t) * n);
    }
    if (ggml_backend_graph_compute(backend, env.gf) != GGML_STATUS_SUCCESS) return false;

    // Packed result: [ attn: S*H*n | final_state: S*S*H | inter: S*S*H*n ]
    ref.attn  = get_f32(result, 0, (size_t)S * H * n);
    ref.inter = get_f32(result, (size_t)S * H * n + (size_t)S * S * H,
                        (size_t)S * S * H * n);
    return true;
}

bool run_specla(ggml_backend_t backend, const CaseInputs & in, SpecLAOutputs & out) {
    const int S = in.S, H = in.H, n = in.n;
    GraphEnv env;
    ggml_context * ctx = env.ctx;

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n, 1);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n, 1);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n, 1);
    ggml_tensor * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, 1);
    ggml_tensor * m_strict = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    ggml_tensor * m_incl   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    ggml_tensor * m_eye    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    for (ggml_tensor * t : {q, k, v, g, b, s, m_strict, m_incl, m_eye}) ggml_set_input(t);

    auto r = build_delta_net_specla(ctx, q, k, v, g, b, s, m_strict, m_incl, m_eye);
    for (ggml_tensor * t : {r.output, r.v_new, r.g_ps}) {
        ggml_set_output(t);
        ggml_build_forward_expand(env.gf, t);
    }

    if (!env.alloc_and_run(backend)) return false;
    set_f32(q, in.q); set_f32(k, in.k); set_f32(v, in.v);
    set_f32(g, in.g); set_f32(b, in.b); set_f32(s, in.s0);
    std::vector<float> ms((size_t)n * n), mi((size_t)n * n), me((size_t)n * n);
    fill_specla_masks(in.parents.data(), n, ms.data(), mi.data(), me.data());
    set_f32(m_strict, ms); set_f32(m_incl, mi); set_f32(m_eye, me);
    if (ggml_backend_graph_compute(backend, env.gf) != GGML_STATUS_SUCCESS) return false;

    out.out   = get_f32(r.output, 0, (size_t)S * H * n);
    out.v_new = get_f32(r.v_new,  0, (size_t)n * S * H);
    out.g_ps  = get_f32(r.g_ps,   0, (size_t)n * H);
    return true;
}

bool run_hld_specla(ggml_backend_t backend, const CaseInputs & in,
                    SpecLAOutputs & out, float & durable_diff,
                    const PendingFactors * pending = nullptr) {
    const int S = in.S, H = in.H, n = in.n;
    const auto schedule = make_specla_hld_schedule(
        in.parents.data(), n, pending ? pending->count : 0);
    if (schedule.packed.empty()) return false;

    GraphEnv env;
    ggml_context * ctx = env.ctx;
    ggml_init_params fp_ip{};
    fp_ip.mem_size = 16*ggml_tensor_overhead();
    fp_ip.no_alloc = true;
    ggml_context * fp_ctx = ggml_init(fp_ip);
    ggml_tensor * banks[8] = {
        ggml_new_tensor_4d(fp_ctx, GGML_TYPE_F32, S, H, 1, n),
        ggml_new_tensor_4d(fp_ctx, GGML_TYPE_F32, S, H, 1, n),
        ggml_new_tensor_3d(fp_ctx, GGML_TYPE_F32, H, 1, n),
        ggml_new_tensor_3d(fp_ctx, GGML_TYPE_F32, 1, 1, n),
        ggml_new_tensor_4d(fp_ctx, GGML_TYPE_F32, S, H, 1, n),
        ggml_new_tensor_4d(fp_ctx, GGML_TYPE_F32, S, H, 1, n),
        ggml_new_tensor_3d(fp_ctx, GGML_TYPE_F32, H, 1, n),
        ggml_new_tensor_3d(fp_ctx, GGML_TYPE_F32, 1, 1, n),
    };
    ggml_backend_buffer_t fp_buf = ggml_backend_alloc_ctx_tensors(fp_ctx, backend);
    if (!fp_buf) { ggml_free(fp_ctx); return false; }
    auto free_fp = [&]() {
        ggml_backend_buffer_free(fp_buf);
        ggml_free(fp_ctx);
    };
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n, 1);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n, 1);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n, 1);
    ggml_tensor * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, 1);
    ggml_tensor * meta = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, (int64_t)schedule.packed.size());
    ggml_tensor * factor_ptrs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 8);
    for (ggml_tensor * t : {q, k, v, g, b, s, meta, factor_ptrs}) {
        ggml_set_input(t);
    }
    ggml_tensor * result = ggml_gated_delta_net_specla(
        ctx, q, k, v, g, b, s, meta, factor_ptrs,
        /*n_layers=*/1, /*layer=*/0, /*pending_bank=*/0,
        schedule.n_boundaries, schedule.n_chains, schedule.n_waves,
        schedule.max_parallel_chains);
    ggml_set_output(result);
    ggml_build_forward_expand(env.gf, result);

    if (!env.alloc_and_run(backend)) {
        free_fp();
        return false;
    }
    set_f32(q, in.q); set_f32(k, in.k); set_f32(v, in.v);
    set_f32(g, in.g); set_f32(b, in.b); set_f32(s, in.s0);
    ggml_backend_tensor_set(meta, schedule.packed.data(), 0,
                            schedule.packed.size()*sizeof(int32_t));
    int64_t ptrs[8];
    for (int i = 0; i < 8; ++i) ptrs[i] = (int64_t)(intptr_t)banks[i]->data;
    ggml_backend_tensor_set(factor_ptrs, ptrs, 0, sizeof(ptrs));
    if (pending) {
        GGML_ASSERT(pending->count <= n);
        ggml_backend_tensor_set(banks[0], pending->k.data(), 0,
                                pending->k.size()*sizeof(float));
        ggml_backend_tensor_set(banks[1], pending->delta.data(), 0,
                                pending->delta.size()*sizeof(float));
        ggml_backend_tensor_set(banks[2], pending->g.data(), 0,
                                pending->g.size()*sizeof(float));
    }
    if (ggml_backend_graph_compute(backend, env.gf) != GGML_STATUS_SUCCESS) {
        free_fp();
        return false;
    }

    const size_t factors = (size_t)S*H*n;
    out.out   = get_f32(result, 0, factors);
    out.v_new = get_f32(banks[5], 0, factors);
    out.g_ps  = get_f32(banks[6], 0, (size_t)H*n);
    durable_diff = max_abs_diff(
        pending ? pending->state_after : in.s0,
        get_f32(s, 0, in.s0.size()));

    const std::vector<float> captured_k = get_f32(banks[4], 0, factors);
    CHECK_MSG(max_abs_diff(in.k, captured_k) == 0.0f,
              "HLD raw k capture differs from input");
    CHECK_MSG(max_abs_diff(in.g, out.g_ps) == 0.0f,
              "HLD raw gate capture differs from input");
    free_fp();
    return true;
}

void run_hld_delayed_case(ggml_backend_t backend) {
    constexpr int S = 64;
    constexpr int H = 4;
    const CaseInputs pending_input = make_inputs(
        S, H, 3, chain_parents(3), 201);
    const PendingFactors pending = make_pending_factors(pending_input);
    CaseInputs current = make_inputs(
        S, H, 13, random_tree_parents(13, 202), 202);
    current.s0 = pending.state_after;
    CaseInputs kernel_current = current;
    kernel_current.s0 = pending_input.s0;

    RefOutputs reference;
    SpecLAOutputs hld;
    float durable_diff = INFINITY;
    const bool ok = run_reference(backend, current, true, reference) &&
        run_hld_specla(backend, kernel_current, hld, durable_diff, &pending);
    CHECK_MSG(ok, "HLD delayed-update/reference compute failed");
    if (!ok) return;
    const float out_diff = max_abs_diff(reference.attn, hld.out);
    CHECK_MSG(out_diff <= 5e-4f,
              "HLD delayed-update output diff %.3e", out_diff);
    CHECK_MSG(durable_diff <= 5e-6f,
              "HLD delayed-update durable-state diff %.3e", durable_diff);
    std::printf("%-28s HLD out=%.3e durable=%.3e\n",
                "hld-delayed-tree", out_diff, durable_diff);
}

void run_hld_conv_delayed_case(ggml_backend_t backend) {
    constexpr int C = 96;
    constexpr int K = 4;
    constexpr int N = 13;
    constexpr int P = 3;
    const std::vector<int32_t> parents = random_tree_parents(N, 302);
    const auto schedule = make_specla_hld_schedule(parents.data(), N, P);
    std::mt19937 rng(301);
    std::normal_distribution<float> nd(0.0f, 0.2f);
    std::vector<float> x((size_t)C*N), weight((size_t)K*C);
    std::vector<float> state((size_t)(K - 1)*C);
    std::vector<float> pending((size_t)C*P);
    for (auto * vec : {&x, &weight, &state, &pending}) {
        for (float & value : *vec) value = nd(rng);
    }

    std::vector<float> durable = state;
    for (int t = 0; t < P; ++t) {
        for (int c = 0; c < C; ++c) {
            float * window = durable.data() + (size_t)c*(K - 1);
            for (int j = 0; j < K - 2; ++j) window[j] = window[j + 1];
            window[K - 2] = pending[(size_t)t*C + c];
        }
    }
    std::vector<float> reference((size_t)C*N);
    std::vector<float> node_states((size_t)N*C*(K - 1));
    for (int node = 0; node < N; ++node) {
        const int parent = parents[(size_t)node];
        for (int c = 0; c < C; ++c) {
            float window[K - 1];
            const float * source = parent < 0
                ? durable.data() + (size_t)c*(K - 1)
                : node_states.data() + ((size_t)parent*C + c)*(K - 1);
            for (int j = 0; j < K - 1; ++j) window[j] = source[j];
            float sum = x[(size_t)node*C + c]*weight[(size_t)c*K + K - 1];
            for (int j = 0; j < K - 1; ++j) {
                sum += window[j]*weight[(size_t)c*K + j];
            }
            reference[(size_t)node*C + c] = sum/(1.0f + std::exp(-sum));
            float * endpoint = node_states.data() + ((size_t)node*C + c)*(K - 1);
            for (int j = 0; j < K - 2; ++j) endpoint[j] = window[j + 1];
            endpoint[K - 2] = x[(size_t)node*C + c];
        }
    }

    GraphEnv env;
    ggml_context * ctx = env.ctx;
    ggml_init_params fp_ip{};
    fp_ip.mem_size = 4*ggml_tensor_overhead();
    fp_ip.no_alloc = true;
    ggml_context * fp_ctx = ggml_init(fp_ip);
    ggml_tensor * pending_bank = ggml_new_tensor_3d(
        fp_ctx, GGML_TYPE_F32, C, 1, N);
    ggml_tensor * current_bank = ggml_new_tensor_3d(
        fp_ctx, GGML_TYPE_F32, C, 1, N);
    ggml_backend_buffer_t fp_buf = ggml_backend_alloc_ctx_tensors(fp_ctx, backend);
    CHECK_MSG(fp_buf != nullptr, "conv factor bank allocation failed");
    if (!fp_buf) { ggml_free(fp_ctx); return; }
    auto free_fp = [&]() {
        ggml_backend_buffer_free(fp_buf);
        ggml_free(fp_ctx);
    };

    ggml_tensor * tx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, N, 1);
    ggml_tensor * tw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, C);
    ggml_tensor * ts = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K - 1, C);
    ggml_tensor * meta = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, (int64_t)schedule.packed.size());
    ggml_tensor * ptr_table = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 8);
    for (ggml_tensor * tensor : {tx, tw, ts, meta, ptr_table}) ggml_set_input(tensor);
    ggml_tensor * result = ggml_ssm_conv_specla(
        ctx, tx, tw, ts, meta, ptr_table,
        /*n_layers=*/1, /*layer=*/0, /*pending_bank=*/0,
        schedule.n_boundaries, schedule.n_chains, schedule.n_waves,
        schedule.max_parallel_chains);
    const bool supported = ggml_backend_supports_op(backend, result);
    CHECK_MSG(supported,
              "CUDA rejected partial-block SpecLA conv channels=%d", C);
    if (!supported) {
        free_fp();
        return;
    }
    ggml_set_output(result);
    ggml_build_forward_expand(env.gf, result);
    const bool allocated = env.alloc_and_run(backend);
    CHECK_MSG(allocated, "conv HLD graph allocation failed");
    if (!allocated) {
        free_fp();
        return;
    }
    set_f32(tx, x); set_f32(tw, weight); set_f32(ts, state);
    ggml_backend_tensor_set(meta, schedule.packed.data(), 0,
                            schedule.packed.size()*sizeof(int32_t));
    ggml_backend_tensor_set(pending_bank, pending.data(), 0,
                            pending.size()*sizeof(float));
    int64_t ptrs[8]{};
    ptrs[3] = (int64_t)(intptr_t)pending_bank->data;
    ptrs[7] = (int64_t)(intptr_t)current_bank->data;
    ggml_backend_tensor_set(ptr_table, ptrs, 0, sizeof(ptrs));
    const bool computed = ggml_backend_graph_compute(backend, env.gf) == GGML_STATUS_SUCCESS;
    CHECK_MSG(computed, "conv HLD graph compute failed");
    if (computed) {
        const float out_diff = max_abs_diff(
            reference, get_f32(result, 0, reference.size()));
        const float state_diff = max_abs_diff(
            durable, get_f32(ts, 0, durable.size()));
        const float factor_diff = max_abs_diff(
            x, get_f32(current_bank, 0, x.size()));
        CHECK_MSG(out_diff <= 2e-6f, "conv HLD output diff %.3e", out_diff);
        CHECK_MSG(state_diff == 0.0f, "conv delayed-state diff %.3e", state_diff);
        CHECK_MSG(factor_diff == 0.0f, "conv factor capture diff %.3e", factor_diff);
        std::printf("%-28s HLD out=%.3e durable=%.3e\n",
                    "conv-hld-delayed-tree", out_diff, state_diff);
    }
    free_fp();
}

void run_factorized_conv_commit_case(ggml_backend_t backend) {
    constexpr int C = 7;
    constexpr int L = 2;
    constexpr int T = 4;
    constexpr int K = 4;
    constexpr int W = K - 1;

    ggml_init_params ip{};
    ip.mem_size = 8*ggml_tensor_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * bank = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, L, T);
    ggml_tensor * state0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, W, C);
    ggml_tensor * state1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, W, C);
    ggml_tensor * ptr_table = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, L);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK_MSG(buffer != nullptr, "factorized conv commit allocation failed");
    if (!buffer) {
        ggml_free(ctx);
        return;
    }

    std::vector<float> factors((size_t)C*L*T);
    for (int t = 0; t < T; ++t) {
        for (int l = 0; l < L; ++l) {
            for (int c = 0; c < C; ++c) {
                factors[(size_t)c + (size_t)C*(l + L*t)] =
                    100.0f*t + 10.0f*l + c;
            }
        }
    }
    std::vector<std::vector<float>> initial((size_t)L,
                                             std::vector<float>((size_t)W*C));
    for (int l = 0; l < L; ++l) {
        for (int c = 0; c < C; ++c) {
            for (int j = 0; j < W; ++j) {
                initial[(size_t)l][(size_t)j + (size_t)W*c] =
                    -100.0f*l - 10.0f*c - j;
            }
        }
    }
    set_f32(bank, factors);
    int64_t ptrs[L] = {
        (int64_t)(intptr_t)state0->data,
        (int64_t)(intptr_t)state1->data,
    };
    ggml_backend_tensor_set(ptr_table, ptrs, 0, sizeof(ptrs));
    ggml_tensor * states[L] = {state0, state1};

    for (int accepted : {1, T}) {
        for (int l = 0; l < L; ++l) set_f32(states[l], initial[(size_t)l]);
        std::vector<std::vector<float>> expected = initial;
        for (int t = 0; t < accepted; ++t) {
            for (int l = 0; l < L; ++l) {
                for (int c = 0; c < C; ++c) {
                    float * window = expected[(size_t)l].data() + (size_t)W*c;
                    for (int j = 0; j < W - 1; ++j) window[j] = window[j + 1];
                    window[W - 1] =
                        factors[(size_t)c + (size_t)C*(l + L*t)];
                }
            }
        }
        const bool committed = dflash::common::specla_commit_conv_raw_fused(
            (float * const *)ptr_table->data, (const float *)bank->data,
            accepted, T, L, C, K, /*stream=*/nullptr);
        CHECK_MSG(committed, "factorized conv commit failed A=%d", accepted);
        if (committed) {
            for (int l = 0; l < L; ++l) {
                const float diff = max_abs_diff(
                    expected[(size_t)l], get_f32(states[l], 0, (size_t)W*C));
                CHECK_MSG(diff == 0.0f,
                          "factorized conv commit A=%d layer=%d diff %.3e",
                          accepted, l, diff);
            }
        }
    }
    CHECK_MSG(!dflash::common::specla_commit_conv_raw_fused(
                  (float * const *)ptr_table->data, (const float *)bank->data,
                  T + 1, T, L, C, K, /*stream=*/nullptr),
              "factorized conv commit accepted an out-of-bounds window");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        CHECK_MSG(false, "max_abs_diff size mismatch: %zu != %zu", a.size(), b.size());
        return std::numeric_limits<float>::infinity();
    }
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

void run_case(ggml_backend_t backend, const char * name, const CaseInputs & in,
              bool tree_op, float out_tol, float state_tol) {
    const int S = in.S, H = in.H, n = in.n;

    RefOutputs ref;
    SpecLAOutputs sp;
    if (!run_reference(backend, in, tree_op, ref)) {
        CHECK_MSG(false, "%s: reference compute failed", name);
        return;
    }
    if (!run_specla(backend, in, sp)) {
        CHECK_MSG(false, "%s: specla compute failed", name);
        return;
    }

    // 1. Per-node outputs.
    const float out_diff = max_abs_diff(ref.attn, sp.out);
    CHECK_MSG(out_diff <= out_tol, "%s: output max diff %.3e > %.3e", name, out_diff, out_tol);

    // 3. g⁺ vs host ancestor path sums (g host layout [H, n] per token block).
    float gps_diff = 0.0f;
    std::vector<float> gps_host((size_t)n * H);
    for (int t = 0; t < n; t++) {
        for (int h = 0; h < H; h++) {
            float acc = 0.0f;
            for (int u = t; u >= 0; u = in.parents[u]) acc += in.g[(size_t)u * H + h];
            gps_host[(size_t)t * H + h] = acc;
            // sp.g_ps layout [n, 1, 1, H]: (t, h) at t + h*n
            gps_diff = std::max(gps_diff,
                std::fabs(acc - sp.g_ps[(size_t)h * n + t]));
        }
    }
    CHECK_MSG(gps_diff <= 1e-5f, "%s: g_ps max diff %.3e", name, gps_diff);

    // 2. DeltaConstruct at every accepted endpoint t: reconstruct S_t from
    //    {k, ṽ, g⁺} along root→t and compare with the kernel's state after t.
    float state_diff = 0.0f;
    std::vector<float> s_rec((size_t)S * S * H);
    for (int t = 0; t < n; t++) {
        // path root→t
        std::vector<int> path;
        for (int u = t; u >= 0; u = in.parents[u]) path.push_back(u);
        for (int h = 0; h < H; h++) {
            const float gA = gps_host[(size_t)t * H + h];
            const float decay0 = std::exp(gA);
            for (int c = 0; c < S; c++) {
                for (int sk = 0; sk < S; sk++) {
                    s_rec[(size_t)h * S * S + (size_t)c * S + sk] =
                        decay0 * in.s0[(size_t)h * S * S + (size_t)c * S + sk];
                }
            }
            for (int u : path) {
                const float w = std::exp(gA - gps_host[(size_t)u * H + h]);
                // k host layout [S, H, n]; ṽ layout [n, S_v, 1, H]
                const float * ku = in.k.data() + (size_t)u * S * H + (size_t)h * S;
                for (int c = 0; c < S; c++) {
                    const float wv = w * sp.v_new[(size_t)h * n * S + (size_t)c * n + u];
                    float * dst = s_rec.data() + (size_t)h * S * S + (size_t)c * S;
                    for (int sk = 0; sk < S; sk++) dst[sk] += wv * ku[sk];
                }
            }
        }
        const std::vector<float> s_ref(ref.inter.begin() + (size_t)t * S * S * H,
                                       ref.inter.begin() + (size_t)(t + 1) * S * S * H);
        state_diff = std::max(state_diff, max_abs_diff(s_rec, s_ref));
    }
    CHECK_MSG(state_diff <= state_tol, "%s: DeltaConstruct state max diff %.3e > %.3e",
              name, state_diff, state_tol);

    std::printf("%-28s S=%-3d H=%-2d n=%-3d out=%.3e g+=%.3e state=%.3e\n",
                name, S, H, n, out_diff, gps_diff, state_diff);
}

void run_hld_case(ggml_backend_t backend, const char * name,
                  const CaseInputs & in, bool tree_op, float tolerance) {
    RefOutputs ref;
    SpecLAOutputs hld;
    float durable_diff = INFINITY;
    if (!run_reference(backend, in, tree_op, ref) ||
        !run_hld_specla(backend, in, hld, durable_diff)) {
        CHECK_MSG(false, "%s: HLD/reference compute failed", name);
        return;
    }
    const float out_diff = max_abs_diff(ref.attn, hld.out);
    CHECK_MSG(out_diff <= tolerance, "%s: HLD output diff %.3e > %.3e",
              name, out_diff, tolerance);
    CHECK_MSG(durable_diff == 0.0f,
              "%s: zero-pending verify mutated durable state %.3e",
              name, durable_diff);
    std::printf("%-28s HLD out=%.3e durable=%.3e\n",
                name, out_diff, durable_diff);
}

void test_hld_schedule() {
    const std::vector<int32_t> parents = {-1, 0, 1, 1, 3, 0, 5, 5};
    const auto hld = make_specla_hld_schedule(
        parents.data(), (int)parents.size(), /*pending_count=*/3);
    CHECK_MSG(!hld.packed.empty(), "HLD schedule unexpectedly empty");
    if (hld.packed.empty()) return;
    CHECK_MSG(hld.n_nodes == (int)parents.size(), "HLD node count mismatch");
    CHECK_MSG(hld.n_chains >= 2 && hld.n_waves >= 2,
              "HLD tree was not decomposed into dependent waves");
    CHECK_MSG(hld.packed[0] == 0x534c4148 && hld.packed[5] == 3,
              "HLD ABI header mismatch");
    const int order_off = hld.packed[6];
    std::vector<int32_t> order(
        hld.packed.begin() + order_off,
        hld.packed.begin() + order_off + parents.size());
    std::sort(order.begin(), order.end());
    for (int i = 0; i < (int)order.size(); ++i) {
        CHECK_MSG(order[(size_t)i] == i, "HLD schedule lost/duplicated node %d", i);
    }
}

std::vector<int32_t> chain_parents(int n) {
    std::vector<int32_t> p(n);
    for (int t = 0; t < n; t++) p[t] = t - 1;
    return p;
}

std::vector<int32_t> random_tree_parents(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<int32_t> p(n);
    p[0] = -1;
    for (int t = 1; t < n; t++) {
        // Bias toward recent nodes so trees have realistic depth.
        std::uniform_int_distribution<int> d(std::max(0, t - 4), t - 1);
        p[t] = d(rng);
    }
    return p;
}

void test_production_commit_kernel(ggml_backend_t backend) {
    constexpr int S_k = 5, S_v = 7, H = 3, L = 4, T = 6;
    const std::vector<int32_t> accepted = {0, 2, 5};

    ggml_init_params ip{};
    ip.mem_size = 64 * ggml_tensor_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> states((size_t)L);
    for (int l = 0; l < L; l++) {
        states[(size_t)l] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_k, S_v, H);
    }
    ggml_tensor * fk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_k, H, L, T);
    ggml_tensor * fv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, L, T);
    ggml_tensor * fg = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, L, T);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_tensor * state_ptrs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, L);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK_MSG(buf != nullptr, "production commit: allocation failed");
    if (!buf) { ggml_free(ctx); return; }

    auto pattern = [](size_t i, float scale) {
        return scale * (float)((int)(i % 17) - 8);
    };
    std::vector<float> fk_h((size_t)S_k * H * L * T);
    std::vector<float> fv_h((size_t)S_v * H * L * T);
    std::vector<float> fg_h((size_t)H * L * T);
    for (size_t i = 0; i < fk_h.size(); i++) fk_h[i] = pattern(i, 0.013f);
    for (size_t i = 0; i < fv_h.size(); i++) fv_h[i] = pattern(i + 3, 0.017f);
    for (size_t i = 0; i < fg_h.size(); i++) fg_h[i] = -0.02f * (float)(1 + i % 9);
    set_f32(fk, fk_h); set_f32(fv, fv_h); set_f32(fg, fg_h);
    ggml_backend_tensor_set(idx, accepted.data(), 0, accepted.size() * sizeof(int32_t));

    std::vector<std::vector<float>> state_h((size_t)L);
    std::vector<int64_t> state_ptr_h((size_t)L);
    for (int l = 0; l < L; l++) {
        auto & s = state_h[(size_t)l];
        s.resize((size_t)S_k * S_v * H);
        for (size_t i = 0; i < s.size(); i++) s[i] = pattern(i + (size_t)l, 0.01f);
        set_f32(states[(size_t)l], s);
        state_ptr_h[(size_t)l] = (int64_t)(intptr_t)states[(size_t)l]->data;
    }
    ggml_backend_tensor_set(state_ptrs, state_ptr_h.data(), 0,
                            state_ptr_h.size() * sizeof(int64_t));

    bool launched = false;
    bool ok = dflash::common::specla_commit_fused(
        (float * const *)state_ptrs->data, (const float *)fk->data,
        (const float *)fv->data, (const float *)fg->data,
        (const int32_t *)idx->data, (int)accepted.size(),
        S_k, S_v, H, L, nullptr, &launched);
    CHECK_MSG(ok && launched, "production SSM commit failed (launched=%d)", (int)launched);
    if (ok) {
        float max_diff = 0.0f;
        for (int l = 0; l < L; l++) {
            std::vector<float> expected = state_h[(size_t)l];
            for (int h = 0; h < H; h++) {
                const int tA = accepted.back();
                const size_t ga_off = (size_t)h + (size_t)H * (l + L * tA);
                const float gA = fg_h[ga_off];
                for (int c = 0; c < S_v; c++) {
                    for (int i = 0; i < S_k; i++) {
                        const size_t se = (size_t)h * S_k * S_v + (size_t)c * S_k + i;
                        float value = std::exp(gA) * expected[se];
                        for (int t : accepted) {
                            const size_t fo = (size_t)h + (size_t)H * (l + L * t);
                            value += std::exp(gA - fg_h[fo]) *
                                fk_h[fo * S_k + i] * fv_h[fo * S_v + c];
                        }
                        expected[se] = value;
                    }
                }
            }
            max_diff = std::max(max_diff,
                max_abs_diff(expected, get_f32(states[(size_t)l], 0, expected.size())));
        }
        CHECK_MSG(max_diff <= 2e-6f, "production SSM commit max diff %.3e", max_diff);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

}  // namespace

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "test_delta_net_specla: no GPU backend available\n");
        return 77;  // ctest SKIP
    }

    const float kOutTol   = 5e-4f;
    const float kStateTol = 5e-4f;

    test_hld_schedule();
    test_production_commit_kernel(backend);

    run_hld_case(backend, "hld-chain-model-shape",
                 make_inputs(128, 8, 16, chain_parents(16), 101),
                 false, kOutTol);
    run_hld_case(backend, "hld-tree-model-shape",
                 make_inputs(128, 8, 24, random_tree_parents(24, 102), 102),
                 true, kOutTol);
    run_hld_delayed_case(backend);
    run_hld_conv_delayed_case(backend);
    run_factorized_conv_commit_case(backend);

    // Chain drafts vs the plain fused op.
    run_case(backend, "chain-small",
             make_inputs(64, 4, 8, chain_parents(8), 1), false, kOutTol, kStateTol);
    run_case(backend, "chain-model-shape",
             make_inputs(128, 8, 16, chain_parents(16), 2), false, kOutTol, kStateTol);
    run_case(backend, "chain-n1",
             make_inputs(64, 2, 1, chain_parents(1), 3), false, kOutTol, kStateTol);
    run_case(backend, "chain-odd",
             make_inputs(64, 2, 33, chain_parents(33), 4), false, kOutTol, kStateTol);

    // Tree drafts vs the fused tree op.
    run_case(backend, "tree-chain-shaped",
             make_inputs(64, 4, 8, chain_parents(8), 5), true, kOutTol, kStateTol);
    {
        std::vector<int32_t> star(9, 0);
        star[0] = -1;
        run_case(backend, "tree-star",
                 make_inputs(64, 4, 9, star, 6), true, kOutTol, kStateTol);
    }
    run_case(backend, "tree-random-small",
             make_inputs(64, 4, 15, random_tree_parents(15, 42), 7), true, kOutTol, kStateTol);
    run_case(backend, "tree-model-shape",
             make_inputs(128, 8, 31, random_tree_parents(31, 43), 8), true, kOutTol, kStateTol);

    // Full qwen35-27B delta-net shape: S=128, H_v=48, 16-token verify window.
    run_case(backend, "chain-qwen35-27b",
             make_inputs(128, 48, 16, chain_parents(16), 9), false, kOutTol, kStateTol);
    run_case(backend, "tree-qwen35-27b",
             make_inputs(128, 48, 24, random_tree_parents(24, 44), 10), true, kOutTol, kStateTol);

    ggml_backend_free(backend);
    if (failures) {
        std::fprintf(stderr, "test_delta_net_specla: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_delta_net_specla: all cases passed\n");
    return 0;
}
