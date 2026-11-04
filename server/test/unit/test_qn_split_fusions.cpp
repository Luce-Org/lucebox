// Bit parity of the Qwen3.8-Flash-Next two-GPU split fusions and RDNA4 decode kernels.
//
// Each case builds its pattern the way the qwen4exp graph does and computes it twice on every GPU: as one graph,
// where the fusion runs, and in segments that a fusion cannot cross (node by node, or a q8_1 producer apart from the
// matmul that reads its rows). Every output must match bit for bit, and the fusion counters
// (ggml_backend_cuda_get_qn_fusion_count) must show that the fusion ran in the one-graph run and not in the other,
// so a pattern that stops matching fails here instead of silently costing launches.
//
// The RDNA4 kernels are compared with the products they replace: the short-row Q8_0 matvec with the same rows in
// slices of 1024 (the old shape), the Q8_0 gate/up pair with separate products and GLU, and the one-wave MoE kernel
// at every width with each token alone. The argmax that greedy decode reads is compared with a host scan of rows
// full of ties, on every GPU and on the CPU backend, so both backends break ties the same way.
//
// Exit codes: 0 pass, 1 mismatch, 77 no GPU backend.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"   // struct ggml_cgraph, for graph segments
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    ++g_failures;
}

// The qwen4exp profile (its fusions), batch-invariant matmul columns and the all-widths MoE kernel, as the split
// forwards set them, restored on scope exit.
struct Profile {
    explicit Profile(bool on) : prev(ggml_backend_cuda_set_qwen4exp_profile(on ? GGML_CUDA_QWEN4EXP_DEFAULT : GGML_CUDA_QWEN4EXP_OFF)) {}
    ~Profile() { ggml_backend_cuda_set_qwen4exp_profile(prev); }
    ggml_cuda_qwen4exp_profile prev;
};
struct Invariant {
    Invariant(bool on, bool all_widths)
        : prev(ggml_backend_cuda_set_mmvq_batch_invariant(on)),
          prev_all(ggml_backend_cuda_set_mmvq_moe_all_widths(all_widths)) {}
    ~Invariant() {
        ggml_backend_cuda_set_mmvq_moe_all_widths(prev_all);
        ggml_backend_cuda_set_mmvq_batch_invariant(prev);
    }
    bool prev, prev_all;
};

// One pattern: its tensors (each with its own memory), its graph and the tensors compared.
struct Pattern {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> outs;
    std::mt19937 rng{1234};

    Pattern() {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(512, false);
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        gf = ggml_new_graph_custom(ctx, 512, false);
    }
    ~Pattern() {
        if (buf) ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    Pattern(const Pattern &) = delete;
    Pattern & operator=(const Pattern &) = delete;

    ggml_tensor * input(ggml_type type, int64_t n0, int64_t n1 = 1, int64_t n2 = 1) {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, type, n0, n1, n2);
        ggml_set_input(t);
        inputs.push_back(t);
        return t;
    }
    void output(ggml_tensor * t) {
        ggml_set_output(t);
        outs.push_back(t);
        ggml_build_forward_expand(gf, t);
    }
    bool alloc(ggml_backend_t be) {
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        return buf != nullptr;
    }
    int index_of(const ggml_tensor * t) const {
        for (int i = 0; i < gf->n_nodes; ++i) if (gf->nodes[i] == t) return i;
        return -1;
    }

    // F32 inputs normal(0, scale) unless set explicitly; quantized ones quantize such values.
    void fill_f32(ggml_tensor * t, float scale = 1.0f) {
        std::normal_distribution<float> nd(0.0f, scale);
        std::vector<float> v((size_t) ggml_nelements(t));
        for (float & x : v) x = nd(rng);
        set(t, v.data());
    }
    void fill_quant(ggml_tensor * t, float scale = 0.05f) {
        std::normal_distribution<float> nd(0.0f, scale);
        const int64_t n_per_row = t->ne[0], nrows = ggml_nrows(t);
        std::vector<float> v((size_t) (n_per_row * nrows));
        for (float & x : v) x = nd(rng);
        std::vector<uint8_t> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, v.data(), q.data(), 0, nrows, n_per_row, nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
        filled.push_back(t);
    }
    void set(ggml_tensor * t, const void * data) {
        ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
        filled.push_back(t);
    }
    void fill_rest() {
        for (ggml_tensor * t : inputs) {
            if (std::find(filled.begin(), filled.end(), t) != filled.end()) continue;
            if (ggml_is_quantized(t->type)) fill_quant(t); else fill_f32(t);
        }
    }

    std::vector<ggml_tensor *> inputs, filled;
};

std::vector<uint8_t> read(const ggml_tensor * t) {
    std::vector<uint8_t> v(ggml_nbytes(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size());
    return v;
}

// Nodes [i0, i1) of gf as their own graph, as ggml_graph_view slices one.
ggml_cgraph slice(ggml_cgraph * gf, int i0, int i1) {
    ggml_cgraph v{};
    v.size = 0;
    v.n_nodes = i1 - i0;
    v.n_leafs = 0;
    v.nodes = gf->nodes + i0;
    v.use_counts = gf->use_counts;
    v.visited_hash_set = gf->visited_hash_set;
    v.order = gf->order;
    v.uid = 0;
    return v;
}

// Computes gf in segments starting at `cuts` (no cuts: every node alone).
bool compute_segments(ggml_backend_t be, ggml_cgraph * gf, std::vector<int> cuts) {
    if (cuts.empty()) for (int i = 1; i < gf->n_nodes; ++i) cuts.push_back(i);
    cuts.insert(cuts.begin(), 0);
    cuts.push_back(gf->n_nodes);
    for (size_t k = 0; k + 1 < cuts.size(); ++k) {
        if (cuts[k + 1] <= cuts[k]) continue;
        ggml_cgraph v = slice(gf, cuts[k], cuts[k + 1]);
        if (ggml_backend_graph_compute(be, &v) != GGML_STATUS_SUCCESS) return false;
    }
    return true;
}

size_t count(ggml_cuda_qn_fusion kind) { return ggml_backend_cuda_get_qn_fusion_count(kind); }

// HC_COMBINE_NORM and the HC fusions exist on HIP builds only.
bool hc_supported(ggml_backend_t be) {
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 8;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    const int n_embd = 512, hc = 4;
    ggml_tensor * cn = ggml_hc_combine_norm(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, 1),
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, 1), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, 1),
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd * hc), 1.0f, 0.0f, 1.0f, 0.0f, 1e-6f);
    const bool ok = ggml_backend_supports_op(be, cn);
    ggml_free(ctx);
    return ok;
}

// The pattern in segments (the reference), then as one graph: equal outputs, and `kind` counted `fused` times in
// the one-graph run and never in the segments.
void check(ggml_backend_t be, const std::string & name, Pattern & p, const std::vector<int> & cuts,
           ggml_cuda_qn_fusion kind, size_t fused) {
    for (int c : cuts) {
        if (c < 1 || c >= p.gf->n_nodes) { fail(name + ": bad segment cut"); return; }
    }
    const size_t c0 = count(kind);
    if (!compute_segments(be, p.gf, cuts)) { fail(name + ": segmented compute"); return; }
    const size_t c1 = count(kind);
    std::vector<std::vector<uint8_t>> ref;
    for (const ggml_tensor * t : p.outs) ref.push_back(read(t));
    if (ggml_backend_graph_compute(be, p.gf) != GGML_STATUS_SUCCESS) { fail(name + ": compute"); return; }
    const size_t c2 = count(kind);
    for (size_t k = 0; k < p.outs.size(); ++k) {
        const std::vector<uint8_t> got = read(p.outs[k]);
        if (got != ref[k]) {
            size_t first = 0;
            while (first < got.size() && got[first] == ref[k][first]) ++first;
            fail(name + ": output " + std::to_string(k) + " (" + ggml_op_desc(p.outs[k]) + ") differs at byte " +
                 std::to_string(first));
        }
    }
    if (c1 != c0) fail(name + ": fused in the segmented reference");
    if (c2 - c1 != fused) {
        fail(name + ": fusion ran " + std::to_string(c2 - c1) + " times, expected " + std::to_string(fused));
    }
}

// ── The router's weight renormalization (GET_ROWS, SUM_ROWS, CLAMP, DIV in one launch) ──────────────────────────
void route_norm(ggml_backend_t be, int n_expert, int k, int T, bool profile) {
    const std::string name = "route_norm n_expert=" + std::to_string(n_expert) + " k=" + std::to_string(k) +
                             " T=" + std::to_string(T) + (profile ? "" : " (no profile)");
    Pattern p;
    ggml_tensor * probs = p.input(GGML_TYPE_F32, n_expert, T);
    ggml_tensor * ids = p.input(GGML_TYPE_I32, k, T);
    // As moe_router builds it.
    ggml_tensor * probs3 = ggml_reshape_3d(p.ctx, probs, 1, n_expert, T);
    ggml_tensor * w = ggml_reshape_2d(p.ctx, ggml_get_rows(p.ctx, probs3, ids), k, T);
    w = ggml_div(p.ctx, w, ggml_clamp(p.ctx, ggml_sum_rows(p.ctx, w), 6.103515625e-5f, INFINITY));
    p.output(w);
    if (!p.alloc(be)) { fail(name + ": alloc"); return; }
    // Softmax-like probabilities; the last token's are tiny, so its sum takes the clamp.
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float> hp((size_t) n_expert * T);
    for (int t = 0; t < T; ++t) {
        float s = 0.0f;
        for (int e = 0; e < n_expert; ++e) s += hp[(size_t) t * n_expert + e] = std::exp(4.0f * u(p.rng));
        const float scale = t == T - 1 && T > 1 ? 1e-7f : 1.0f;
        for (int e = 0; e < n_expert; ++e) hp[(size_t) t * n_expert + e] *= scale / s;
    }
    std::vector<int32_t> hi((size_t) k * T);
    std::vector<int32_t> perm(n_expert);
    for (int t = 0; t < T; ++t) {
        for (int e = 0; e < n_expert; ++e) perm[e] = e;
        std::shuffle(perm.begin(), perm.end(), p.rng);
        std::copy(perm.begin(), perm.begin() + k, hi.begin() + (size_t) t * k);
    }
    p.set(probs, hp.data());
    p.set(ids, hi.data());
    const Profile prof(profile);
    check(be, name, p, {}, GGML_CUDA_QN_ROUTE_NORM, profile ? 1 : 0);
}

// ── The shared expert's sigmoid gate folded into the MoE combine ────────────────────────────────────────────────
void shared_gate(ggml_backend_t be, int n_embd, int k, int T) {
    const std::string name = "shared_gate T=" + std::to_string(T);
    Pattern p;
    ggml_tensor * down = p.input(GGML_TYPE_F32, n_embd, k, T);
    ggml_tensor * wts = p.input(GGML_TYPE_F32, k, T);
    ggml_tensor * shared = p.input(GGML_TYPE_F32, n_embd, T);
    ggml_tensor * logit = p.input(GGML_TYPE_F32, 1, T);
    // As build_moe_route and build_moe build it.
    ggml_tensor * gated = ggml_mul(p.ctx, shared, ggml_sigmoid(p.ctx, logit));
    p.output(ggml_ds4_moe_fused_combine_shared(p.ctx, down, wts, gated));
    if (!p.alloc(be)) { fail(name + ": alloc"); return; }
    p.fill_rest();
    const Profile prof(true);
    check(be, name, p, {}, GGML_CUDA_QN_SHARED_GATE, 1);
}

// ── A gated unary (MUL of SIGMOID) writing the q8_1 rows of the matmul that reads it ───────────────────────────
void unary_mul_q8(ggml_backend_t be, int T) {
    const std::string name = "unary_mul_q8 T=" + std::to_string(T);
    const int K = 4096, N = 2560;
    Pattern p;
    ggml_tensor * x = p.input(GGML_TYPE_F32, K, T);
    ggml_tensor * z = p.input(GGML_TYPE_F32, K, T);
    ggml_tensor * W = p.input(GGML_TYPE_Q8_0, K, N);
    ggml_tensor * g = ggml_mul(p.ctx, x, ggml_sigmoid(p.ctx, z));
    p.output(g);
    ggml_tensor * mm = ggml_mul_mat(p.ctx, W, g);
    p.output(mm);
    if (!p.alloc(be)) { fail(name + ": alloc"); return; }
    p.fill_rest();
    const Invariant inv(true, false);
    check(be, name, p, {p.index_of(mm)}, GGML_CUDA_QN_UNARY_MUL_Q8, 1);
}

// ── SCALE then SILU (the hyper-connection low-rank gate) in one launch, writing the up projection's q8_1 rows ──
void scale_silu(ggml_backend_t be, int T) {
    const std::string name = "scale_silu T=" + std::to_string(T);
    const int rank = 320, hc_dim = 10240;
    Pattern p;
    ggml_tensor * lo = p.input(GGML_TYPE_F32, rank, T);
    ggml_tensor * W = p.input(GGML_TYPE_Q8_0, rank, hc_dim);
    ggml_tensor * s = ggml_silu(p.ctx, ggml_scale(p.ctx, lo, 0.25f));   // hc_mix_body
    p.output(s);
    p.output(ggml_mul_mat(p.ctx, W, s));
    if (!p.alloc(be)) { fail(name + ": alloc"); return; }
    p.fill_rest();
    const Invariant inv(true, false);
    check(be, name, p, {}, GGML_CUDA_QN_SCALE_SILU, 1);
}

// ── HC_COMBINE_NORM writing the HC down projection's q8_1 rows ─────────────────────────────────────────────────
void hc_cn_q8(ggml_backend_t be, int T) {
    if (!hc_supported(be)) return;
    const std::string name = "hc_cn_q8 T=" + std::to_string(T);
    const int n_embd = 2560, hc = 4, rank = 320;
    Pattern p;
    ggml_tensor * inject = p.input(GGML_TYPE_F32, hc, T);
    ggml_tensor * residual = p.input(GGML_TYPE_F32, n_embd, hc, T);
    ggml_tensor * block = p.input(GGML_TYPE_F32, n_embd, T);
    ggml_tensor * gamma = p.input(GGML_TYPE_F32, n_embd * hc);
    ggml_tensor * W = p.input(GGML_TYPE_Q8_0, n_embd * hc, rank);
    // hc_combine_norm, hc_norm_xn and hc_mix_from_xn.
    ggml_tensor * cn = ggml_hc_combine_norm(p.ctx, inject, residual, ggml_reshape_3d(p.ctx, block, n_embd, 1, T),
                                            gamma, 1.0f / hc, 0.0f, 2.0f, 0.0f, 1e-6f);
    ggml_tensor * xn = ggml_view_4d(p.ctx, cn, n_embd, hc, T, 1, cn->nb[1], cn->nb[2], cn->nb[3],
                                    (size_t) n_embd * hc * T * sizeof(float));
    ggml_tensor * mm = ggml_mul_mat(p.ctx, W, ggml_reshape_2d(p.ctx, xn, (int64_t) n_embd * hc, T));
    p.output(cn);
    p.output(mm);
    if (!p.alloc(be)) { fail(name + ": alloc"); return; }
    p.fill_rest();
    const Profile prof(true);
    const Invariant inv(true, false);
    check(be, name, p, {p.index_of(mm)}, GGML_CUDA_QN_HC_CN_Q8, 1);
}

// ── The HC mix reduce writing the q8_1 rows of the matmul that reads the mix ───────────────────────────────────
void hc_mix_q8(ggml_backend_t be, int T) {
    if (!hc_supported(be)) return;
    const std::string name = "hc_mix_q8 T=" + std::to_string(T);
    const int n_embd = 2560, hc = 4, N = 512;
    Pattern p;
    ggml_tensor * xn = p.input(GGML_TYPE_F32, n_embd * hc, T);
    ggml_tensor * gate_in = p.input(GGML_TYPE_F32, n_embd * hc, T);
    ggml_tensor * W = p.input(GGML_TYPE_Q8_0, n_embd, N);
    // hc_mix_body after the gate projection.
    ggml_tensor * gated = ggml_reshape_3d(p.ctx, ggml_mul(p.ctx, xn, ggml_sigmoid(p.ctx, gate_in)), n_embd, hc, T);
    const size_t row = ggml_row_size(GGML_TYPE_F32, n_embd);
    ggml_tensor * mixed = ggml_cont(p.ctx, ggml_view_2d(p.ctx, gated, n_embd, T, row * hc, 0));
    for (int s = 1; s < hc; ++s) mixed = ggml_add(p.ctx, mixed, ggml_view_2d(p.ctx, gated, n_embd, T, row * hc, row * s));
    mixed = ggml_scale(p.ctx, mixed, 1.0f / hc);
    p.output(mixed);
    ggml_tensor * mm = ggml_mul_mat(p.ctx, W, mixed);
    p.output(mm);
    if (!p.alloc(be)) { fail(name + ": alloc"); return; }
    p.fill_rest();
    const Profile prof(true);
    const Invariant inv(true, false);
    check(be, name, p, {p.index_of(mm)}, GGML_CUDA_QN_HC_MIX_Q8, 1);
}

// ── RDNA4: Q8_0 short rows over many rows (the HC up projection) vs the same rows in slices of 1024 ────────────
void q8_small_k(ggml_backend_t be, int T) {
    const std::string name = "q8_small_k T=" + std::to_string(T);
    const int K = 320, N = 10240, slice_rows = 1024;
    Pattern p;
    ggml_tensor * W = p.input(GGML_TYPE_Q8_0, K, N);
    ggml_tensor * x = p.input(GGML_TYPE_F32, K, T);
    ggml_tensor * full = ggml_mul_mat(p.ctx, W, x);
    p.output(full);
    std::vector<ggml_tensor *> parts;
    for (int r = 0; r < N; r += slice_rows) {
        ggml_tensor * Wr = ggml_view_2d(p.ctx, W, K, slice_rows, W->nb[1], (size_t) r * W->nb[1]);
        parts.push_back(ggml_mul_mat(p.ctx, Wr, x));
        p.output(parts.back());
    }
    if (!p.alloc(be)) { fail(name + ": alloc"); return; }
    p.fill_rest();
    const Invariant inv(true, false);
    const size_t c0 = count(GGML_CUDA_QN_Q8_SMALL_K);
    if (ggml_backend_graph_compute(be, p.gf) != GGML_STATUS_SUCCESS) { fail(name + ": compute"); return; }
    if (count(GGML_CUDA_QN_Q8_SMALL_K) - c0 != 1) fail(name + ": short-row shape not taken exactly once");
    const std::vector<uint8_t> f = read(full);
    for (size_t k = 0; k < parts.size(); ++k) {
        const std::vector<uint8_t> s = read(parts[k]);
        for (int t = 0; t < T; ++t) {
            const size_t row_bytes = (size_t) slice_rows * sizeof(float);
            if (std::memcmp(f.data() + ((size_t) t * N + k * slice_rows) * sizeof(float),
                            s.data() + (size_t) t * row_bytes, row_bytes) != 0) {
                fail(name + ": rows " + std::to_string(k * slice_rows) + " token " + std::to_string(t));
            }
        }
    }
}

// ── RDNA4: the Q8_0 gate/up pair serving plain SwiGLU (the shared expert) ───────────────────────────────────────
void swiglu_pair(ggml_backend_t be, int T) {
    const std::string name = "swiglu_pair T=" + std::to_string(T);
    const int K = 2560, N = 640;
    Pattern p;
    ggml_tensor * Wg = p.input(GGML_TYPE_Q8_0, K, N);
    ggml_tensor * Wu = p.input(GGML_TYPE_Q8_0, K, N);
    ggml_tensor * x = p.input(GGML_TYPE_F32, K, T);
    p.output(ggml_swiglu_split(p.ctx, ggml_mul_mat(p.ctx, Wg, x), ggml_mul_mat(p.ctx, Wu, x)));
    if (!p.alloc(be)) { fail(name + ": alloc"); return; }
    p.fill_rest();
    const Invariant inv(true, false);
    check(be, name, p, {}, GGML_CUDA_QN_SWIGLU_PAIR, 1);
}

// ── RDNA4: MUL_MAT_ID on the one-wave MoE kernel at every width vs each token alone ────────────────────────────
// up: the gate/up form (activations [K, 1, T] shared by the routes), else the down form ([K, k, T], one per route).
void moe_all_widths(ggml_backend_t be, ggml_type type, bool up, int T) {
    const std::string name = std::string("moe_all_widths ") + ggml_type_name(type) + (up ? " up" : " down") +
                             " T=" + std::to_string(T);
    const int n_expert = 16, k = 4;
    const int K = up ? 2560 : 512, N = up ? 512 : 2560;
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> hx((size_t) K * (up ? 1 : k) * T);
    for (float & v : hx) v = nd(rng);
    std::vector<int32_t> hids((size_t) k * T);
    std::vector<int32_t> perm(n_expert);
    for (int t = 0; t < T; ++t) {
        for (int e = 0; e < n_expert; ++e) perm[e] = e;
        std::shuffle(perm.begin(), perm.end(), rng);
        std::copy(perm.begin(), perm.begin() + k, hids.begin() + (size_t) t * k);
    }
    // Same weights in every graph: quantize once.
    std::vector<float> hw((size_t) K * N * n_expert);
    for (float & v : hw) v = 0.05f * nd(rng);
    std::vector<uint8_t> qw(ggml_row_size(type, K) * N * n_expert);
    ggml_quantize_chunk(type, hw.data(), qw.data(), 0, (int64_t) N * n_expert, K, nullptr);

    // Width w starting at token t0: the [N, k, w] product.
    auto run = [&](int t0, int w, std::vector<uint8_t> & out, size_t & taken) {
        Pattern p;
        ggml_tensor * W = p.input(type, K, N, n_expert);
        ggml_tensor * x = p.input(GGML_TYPE_F32, K, up ? 1 : k, w);
        ggml_tensor * ids = p.input(GGML_TYPE_I32, k, w);
        ggml_tensor * y = ggml_mul_mat_id(p.ctx, W, x, ids);
        p.output(y);
        if (!p.alloc(be)) return false;
        ggml_backend_tensor_set(W, qw.data(), 0, qw.size());
        p.set(x, hx.data() + (size_t) t0 * K * (up ? 1 : k));
        p.set(ids, hids.data() + (size_t) t0 * k);
        const Invariant inv(true, true);
        const size_t c0 = count(GGML_CUDA_QN_MOE_ALL_WIDTHS);
        if (ggml_backend_graph_compute(be, p.gf) != GGML_STATUS_SUCCESS) return false;
        taken = count(GGML_CUDA_QN_MOE_ALL_WIDTHS) - c0;
        out = read(y);
        return true;
    };
    std::vector<uint8_t> all;
    size_t taken = 0;
    if (!run(0, T, all, taken)) { fail(name + ": compute"); return; }
    if (taken != 1) fail(name + ": all-widths kernel not taken");
    const size_t token_bytes = (size_t) N * k * sizeof(float);
    for (int t = 0; t < T; ++t) {
        std::vector<uint8_t> one;
        if (!run(t, 1, one, taken)) { fail(name + ": compute token"); return; }
        if (taken != 1) fail(name + ": all-widths kernel not taken at one token");
        if (std::memcmp(all.data() + (size_t) t * token_bytes, one.data(), token_bytes) != 0) {
            fail(name + ": token " + std::to_string(t) + " differs from its one-token product");
        }
    }
}

// ── The argmax greedy decode reads: ties go to the lowest index, as a host scan's ──────────────────────────────
void argmax_ties(ggml_backend_t be) {
    const int V = 151936, rows = 6;
    Pattern p;
    ggml_tensor * logits = p.input(GGML_TYPE_F32, V, rows);
    ggml_tensor * am = ggml_argmax(p.ctx, logits);
    p.output(am);
    if (!p.alloc(be)) { fail("argmax: alloc"); return; }
    std::vector<float> h((size_t) V * rows);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float & v : h) v = nd(p.rng);
    // Equal maxima in different threads, warps and strides of the kernel's block.
    const int ties[rows][3] = {{1, 1024, 70000}, {37, 5, -1}, {3, 700, 151935}, {100000, 31, 32}, {0, 1, 2}, {-1, -1, -1}};
    for (int r = 0; r < rows; ++r) {
        float * row = h.data() + (size_t) r * V;
        if (r == rows - 1) {
            std::fill(row, row + V, 0.25f);   // every entry ties
            continue;
        }
        for (int j = 0; j < 3; ++j) if (ties[r][j] >= 0) row[ties[r][j]] = 50.0f;
    }
    p.set(logits, h.data());
    if (ggml_backend_graph_compute(be, p.gf) != GGML_STATUS_SUCCESS) { fail("argmax: compute"); return; }
    std::vector<int32_t> got(rows);
    ggml_backend_tensor_get(am, got.data(), 0, sizeof(int32_t) * rows);
    for (int r = 0; r < rows; ++r) {
        const float * row = h.data() + (size_t) r * V;
        const int want = (int) (std::max_element(row, row + V) - row);
        if (got[r] != want) fail("argmax row " + std::to_string(r) + ": device " + std::to_string(got[r]) +
                                 ", host " + std::to_string(want));
    }
}

}  // namespace

int main() {
    if (ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr)) {
        argmax_ties(cpu);
        std::printf("CPU argmax: %s\n", g_failures == 0 ? "PASS" : "FAIL");
        ggml_backend_free(cpu);
    }
    const int n_dev = ggml_backend_cuda_get_device_count();
    if (n_dev <= 0) {
        std::printf("SKIP: no GPU\n");
        return g_failures == 0 ? 77 : 1;
    }
    const bool prev_graphs = ggml_backend_cuda_set_graphs_disabled_override(true);
    for (int dev = 0; dev < n_dev; ++dev) {
        ggml_backend_t be = ggml_backend_cuda_init(dev);
        if (!be) {
            std::printf("SKIP: no backend for GPU %d\n", dev);
            return 77;
        }
        char desc[256] = {};
        ggml_backend_cuda_get_device_description(dev, desc, sizeof(desc));
        // gfx1151 (qwen4exp_supported) and RDNA4 run the QSA kernels; RDNA4 is the split target.
        const bool rdna4 = ggml_backend_cuda_qsa_supported(be) && !ggml_backend_cuda_qwen4exp_supported(be);
        std::printf("GPU %d: %s%s\n", dev, desc, rdna4 ? " (RDNA4)" : "");
        const int before = g_failures;

        for (int T : {1, 3, 8}) route_norm(be, 512, 10, T, true);
        route_norm(be, 256, 6, 2, false);   // DeepSeek4-sized router outside the profile: untouched
        for (int T : {1, 2, 8}) shared_gate(be, 2560, 10, T);
        for (int T : {1, 2, 4}) unary_mul_q8(be, T);
        for (int T : {1, 2, 8}) scale_silu(be, T);
        for (int T : {1, 2, 8}) hc_cn_q8(be, T);
        for (int T : {1, 2, 8}) hc_mix_q8(be, T);
        argmax_ties(be);
        if (rdna4) {
            for (int T : {1, 2, 4, 8}) q8_small_k(be, T);
            for (int T : {2, 4, 8}) swiglu_pair(be, T);
            for (ggml_type type : {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) {
                for (bool up : {true, false}) moe_all_widths(be, type, up, 8);
            }
        }
        std::printf("GPU %d: %s\n", dev, g_failures == before ? "PASS" : "FAIL");
        ggml_backend_free(be);
    }
    ggml_backend_cuda_set_graphs_disabled_override(prev_graphs);
    return g_failures == 0 ? 0 : 1;
}
