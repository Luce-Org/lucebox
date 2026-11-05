// Bit parity of the Qwen3.8-Flash-Next two-GPU split fusions.
//
// Each case builds its pattern the way the qwen4exp graph does and computes it twice on every GPU: as one graph,
// where the fusion runs, and in segments that a fusion cannot cross (node by node, or a q8_1 producer apart from the
// matmul that reads its rows). Every output must match bit for bit, and the fusion counters
// (ggml_backend_cuda_get_qn_fusion_count) must show that the fusion ran in the one-graph run and not in the other,
// so a pattern that stops matching fails here instead of silently costing launches.
//
// The argmax that greedy decode reads is compared with a host scan of rows full of ties, on every GPU and on the
// CPU backend, so both backends break ties the same way.
//
// One CppUnit case per fusion, each run on every GPU; a case with no GPU (or, for the HC fusions, no HIP GPU) skips.
#include "CppUnitTestFramework.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
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

// The qwen4exp profile (its fusions) and batch-invariant matmul columns, as the split forwards set them, restored on
// scope exit.
struct Profile {
    explicit Profile(bool on) : prev(ggml_backend_cuda_set_qwen4exp_profile(on ? GGML_CUDA_QWEN4EXP_DEFAULT : GGML_CUDA_QWEN4EXP_OFF)) {}
    ~Profile() { ggml_backend_cuda_set_qwen4exp_profile(prev); }
    ggml_cuda_qwen4exp_profile prev;
};
struct Invariant {
    explicit Invariant(bool on) : prev(ggml_backend_cuda_set_mmvq_batch_invariant(on)) {}
    ~Invariant() { ggml_backend_cuda_set_mmvq_batch_invariant(prev); }
    bool prev;
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

// CUDA graphs off for the fixture's cases, so every compute launches its kernels and the fusion counters see them.
struct GraphsOff {
    GraphsOff() : prev(ggml_backend_cuda_set_graphs_disabled_override(true)) {}
    ~GraphsOff() { ggml_backend_cuda_set_graphs_disabled_override(prev); }
    bool prev;
};

struct QnSplitFusionsFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;

    // A failed comparison: logged with its description, and the case carries on to the next one.
    void fail(const std::string & what) {
        HandleAssert(CppUnitTestFramework::AssertType::Continue, _CPPUTF_ASSERT_LOCATION,
                     CppUnitTestFramework::AssertException(what));
    }

    // f(backend) on every GPU; the case skips when there is none.
    template <typename F> void each_gpu(F && f) {
        const int n_dev = ggml_backend_cuda_get_device_count();
        if (n_dev <= 0) SKIP("no GPU");
        const GraphsOff graphs;
        for (int dev = 0; dev < n_dev; ++dev) {
            ggml_backend_ptr be(ggml_backend_cuda_init(dev));
            REQUIRE(be != nullptr);
            char desc[256] = {};
            ggml_backend_cuda_get_device_description(dev, desc, sizeof(desc));
            std::printf("GPU %d: %s\n", dev, desc);
            f(be.get());
        }
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
        const Invariant inv(true);
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
        const Invariant inv(true);
        check(be, name, p, {}, GGML_CUDA_QN_SCALE_SILU, 1);
    }

    // ── HC_COMBINE_NORM writing the HC down projection's q8_1 rows ─────────────────────────────────────────────────
    void hc_cn_q8(ggml_backend_t be, int T) {
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
        const Invariant inv(true);
        check(be, name, p, {p.index_of(mm)}, GGML_CUDA_QN_HC_CN_Q8, 1);
    }

    // ── The HC mix reduce writing the q8_1 rows of the matmul that reads the mix ───────────────────────────────────
    void hc_mix_q8(ggml_backend_t be, int T) {
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
        const Invariant inv(true);
        check(be, name, p, {p.index_of(mm)}, GGML_CUDA_QN_HC_MIX_Q8, 1);
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
};

}  // namespace

TEST_CASE(QnSplitFusionsFixture, route_norm) {
    each_gpu([&](ggml_backend_t be) {
        for (int T : {1, 3, 8}) route_norm(be, 512, 10, T, true);
        route_norm(be, 256, 6, 2, false);   // DeepSeek4-sized router outside the profile: untouched
    });
}

TEST_CASE(QnSplitFusionsFixture, shared_gate) {
    each_gpu([&](ggml_backend_t be) { for (int T : {1, 2, 8}) shared_gate(be, 2560, 10, T); });
}

TEST_CASE(QnSplitFusionsFixture, unary_mul_q8) {
    each_gpu([&](ggml_backend_t be) { for (int T : {1, 2, 4}) unary_mul_q8(be, T); });
}

TEST_CASE(QnSplitFusionsFixture, scale_silu) {
    each_gpu([&](ggml_backend_t be) { for (int T : {1, 2, 8}) scale_silu(be, T); });
}

TEST_CASE(QnSplitFusionsFixture, hc_cn_q8) {
    bool ran = false;
    each_gpu([&](ggml_backend_t be) {
        if (!hc_supported(be)) return;
        ran = true;
        for (int T : {1, 2, 8}) hc_cn_q8(be, T);
    });
    if (!ran) SKIP("HC_COMBINE_NORM is HIP-only");
}

TEST_CASE(QnSplitFusionsFixture, hc_mix_q8) {
    bool ran = false;
    each_gpu([&](ggml_backend_t be) {
        if (!hc_supported(be)) return;
        ran = true;
        for (int T : {1, 2, 8}) hc_mix_q8(be, T);
    });
    if (!ran) SKIP("the HC mix fusion is HIP-only");
}

TEST_CASE(QnSplitFusionsFixture, argmax_ties_gpu) {
    each_gpu([&](ggml_backend_t be) { argmax_ties(be); });
}

TEST_CASE(QnSplitFusionsFixture, argmax_ties_cpu) {
    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    REQUIRE(cpu != nullptr);
    argmax_ties(cpu.get());
}
