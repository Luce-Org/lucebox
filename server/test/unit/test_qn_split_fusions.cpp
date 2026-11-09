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
// One CppUnit case per fusion, each run on every GPU; a case with no GPU (or, for the HIP-only fusions, no HIP GPU)
// skips.
#include "CppUnitTestFramework.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"   // struct ggml_cgraph, for graph segments
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
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
        if (galloc) ggml_gallocr_free(galloc);
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
    // The graph allocator instead, which gives a tensor the memory of tensors the graph has finished reading, as the
    // scheduler does.
    bool alloc_graph(ggml_backend_t be) {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        return galloc && ggml_gallocr_alloc_graph(galloc, gf);
    }
    ggml_gallocr_t galloc = nullptr;
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

    // The pattern in segments (the reference), then as one graph: equal outputs, and each kind counted its expected
    // times in the one-graph run and never in the segments. `reset`, when given, runs before each of the two (it
    // restores state the pattern updates in place).
    using Expect = std::vector<std::pair<ggml_cuda_qn_fusion, size_t>>;
    void check(ggml_backend_t be, const std::string & name, Pattern & p, const std::vector<int> & cuts,
               ggml_cuda_qn_fusion kind, size_t fused) {
        check(be, name, p, cuts, Expect{{kind, fused}});
    }
    void check(ggml_backend_t be, const std::string & name, Pattern & p, const std::vector<int> & cuts,
               const Expect & expect, const std::function<void()> & reset = {}) {
        for (int c : cuts) {
            if (c < 1 || c >= p.gf->n_nodes) { fail(name + ": bad segment cut"); return; }
        }
        auto counts = [&] { std::vector<size_t> v; for (const auto & e : expect) v.push_back(count(e.first)); return v; };
        const std::vector<size_t> c0 = counts();
        if (reset) reset();
        if (!compute_segments(be, p.gf, cuts)) { fail(name + ": segmented compute"); return; }
        const std::vector<size_t> c1 = counts();
        std::vector<std::vector<uint8_t>> ref;
        for (const ggml_tensor * t : p.outs) ref.push_back(read(t));
        if (reset) reset();
        if (ggml_backend_graph_compute(be, p.gf) != GGML_STATUS_SUCCESS) { fail(name + ": compute"); return; }
        const std::vector<size_t> c2 = counts();
        for (size_t k = 0; k < p.outs.size(); ++k) {
            const std::vector<uint8_t> got = read(p.outs[k]);
            if (got != ref[k]) {
                size_t first = 0;
                while (first < got.size() && got[first] == ref[k][first]) ++first;
                fail(name + ": output " + std::to_string(k) + " (" + ggml_op_desc(p.outs[k]) + ") differs at byte " +
                     std::to_string(first));
            }
        }
        for (size_t e = 0; e < expect.size(); ++e) {
            const std::string kind = "fusion " + std::to_string((int) expect[e].first);
            if (c1[e] != c0[e]) fail(name + ": " + kind + " ran in the segmented reference");
            if (c2[e] - c1[e] != expect[e].second) {
                fail(name + ": " + kind + " ran " + std::to_string(c2[e] - c1[e]) + " times, expected " +
                     std::to_string(expect[e].second));
            }
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

    // ── The whole router after its logits matmul (SOFT_MAX, ARGSORT top-k, table gathers, weights in one launch) ──
    // Built as moe_router and build_moe_route build it, in the order their gf expansion gives. `n_lut` tables: the
    // split layers' cold and hot slots (2); 0 is the plain layers' router, whose picks the expert matmuls read (here a
    // CONT of them, so the fused launch must write them too). `ties`: logits on a coarse grid, so the sort breaks
    // many ties.
    void router_topk(ggml_backend_t be, int n_expert, int k, int T, int n_lut, bool ties, bool profile) {
        const std::string name = "router_topk n_expert=" + std::to_string(n_expert) + " k=" + std::to_string(k) +
                                 " T=" + std::to_string(T) + " luts=" + std::to_string(n_lut) +
                                 (ties ? " ties" : "") + (profile ? "" : " (no profile)");
        Pattern p;
        ggml_tensor * logits = p.input(GGML_TYPE_F32, n_expert, T);
        ggml_tensor * tables[2] = { nullptr, nullptr };
        for (int l = 0; l < n_lut; ++l) tables[l] = p.input(GGML_TYPE_I32, 1, n_expert);
        ggml_tensor * probs = ggml_soft_max(p.ctx, logits);
        ggml_tensor * sel = ggml_argsort_top_k(p.ctx, probs, k);
        ggml_tensor * probs3 = ggml_reshape_3d(p.ctx, probs, 1, n_expert, T);
        ggml_tensor * wsel = ggml_reshape_2d(p.ctx, ggml_get_rows(p.ctx, probs3, sel), k, T);
        wsel = ggml_div(p.ctx, wsel, ggml_clamp(p.ctx, ggml_sum_rows(p.ctx, wsel), 6.103515625e-5f, INFINITY));
        if (n_lut > 0) {
            ggml_tensor * flat =
                ggml_reshape_1d(p.ctx, ggml_is_contiguous(sel) ? sel : ggml_cont(p.ctx, sel), (int64_t) k * T);
            for (int l = 0; l < n_lut; ++l) p.output(ggml_reshape_2d(p.ctx, ggml_get_rows(p.ctx, tables[l], flat), k, T));
            p.output(wsel);
        } else {
            p.output(wsel);
            p.output(ggml_cont(p.ctx, sel));
        }
        if (!p.alloc(be)) { fail(name + ": alloc"); return; }
        std::vector<float> hl((size_t) n_expert * T);
        std::normal_distribution<float> nd(0.0f, 2.0f);
        for (float & x : hl) x = ties ? std::round(nd(p.rng)) * 0.5f : nd(p.rng);
        p.set(logits, hl.data());
        // A quarter of the experts hot: the cold table maps the others to themselves, the hot one numbers the hot.
        std::vector<int32_t> perm(n_expert), cold(n_expert, -1), hot(n_expert, -1);
        for (int e = 0; e < n_expert; ++e) perm[e] = e;
        std::shuffle(perm.begin(), perm.end(), p.rng);
        for (int r = 0; r < n_expert; ++r) {
            if (r < n_expert / 4) hot[perm[r]] = r; else cold[perm[r]] = perm[r];
        }
        if (n_lut > 0) p.set(tables[0], cold.data());
        if (n_lut > 1) p.set(tables[1], hot.data());
        const Profile prof(profile);
        check(be, name, p, {}, Expect{{GGML_CUDA_QN_ROUTER_TOPK, profile ? 1u : 0u}, {GGML_CUDA_QN_ROUTE_NORM, 0}});
    }

    // ── The same renormalization after the router's softmax and top-k, from the graph allocator ───────────────────────
    // GET_ROWS is the last reader of the probabilities and the picks, so the allocator gives the DIV output their
    // memory: the launch must read every token's picks before it writes, and past one block of tokens it declines.
    void route_norm_reuse(ggml_backend_t be, int n_expert, int k, int T) {
        const std::string name = "route_norm_reuse T=" + std::to_string(T);
        Pattern p;
        ggml_tensor * logits = p.input(GGML_TYPE_F32, n_expert, T);
        // As moe_router builds it.
        ggml_tensor * probs = ggml_soft_max(p.ctx, logits);
        ggml_tensor * ids = ggml_argsort_top_k(p.ctx, probs, k);
        ggml_tensor * probs3 = ggml_reshape_3d(p.ctx, probs, 1, n_expert, T);
        ggml_tensor * w = ggml_reshape_2d(p.ctx, ggml_get_rows(p.ctx, probs3, ids), k, T);
        w = ggml_div(p.ctx, w, ggml_clamp(p.ctx, ggml_sum_rows(p.ctx, w), 6.103515625e-5f, INFINITY));
        p.output(w);
        if (!p.alloc_graph(be)) { fail(name + ": alloc"); return; }
        auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
            const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
            return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
        };
        const bool shared = overlap(w, probs) || overlap(w, ids->view_src ? ids->view_src : ids);
        if (!shared) { fail(name + ": the allocator kept the output apart, so this case no longer covers the reuse"); return; }
        // SOFT_MAX runs in place over the logits and the output lands there too, so each run starts from the logits.
        std::vector<float> lv((size_t) n_expert * T);
        std::normal_distribution<float> nd(0.0f, 1.5f);
        for (float & x : lv) x = nd(p.rng);
        auto reset = [&] { ggml_backend_tensor_set(logits, lv.data(), 0, ggml_nbytes(logits)); };
        const Profile prof(true);
        // One token may write over its own logits, so the router launch takes the whole chain; more tokens over
        // the logits leave it to the route-norm launch (which declines past one block).
        check(be, name, p, {}, Expect{{GGML_CUDA_QN_ROUTER_TOPK, T == 1 ? 1 : 0},
                                     {GGML_CUDA_QN_ROUTE_NORM, T > 1 && T <= 1024 ? 1 : 0}}, reset);
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

    // ── The same gated unary where MMVQ would not read its q8_1 rows: no entry, no writes ──────────────────────────
    // [D, H, T] rows read as [D*H, T] (the GDN gated output into ssm_out) miss MMVQ's key, and at prefill widths the
    // matmul takes another kernel that quantizes for itself.
    void unary_mul_unread_q8(ggml_backend_t be, int D, int H, int T) {
        const std::string name = "unary_mul_unread_q8 D=" + std::to_string(D) + " H=" + std::to_string(H) +
                                 " T=" + std::to_string(T);
        const int N = 2560;
        Pattern p;
        ggml_tensor * x = p.input(GGML_TYPE_F32, D, H, T);
        ggml_tensor * z = p.input(GGML_TYPE_F32, D, H, T);
        ggml_tensor * W = p.input(GGML_TYPE_Q8_0, (int64_t) D * H, N);
        ggml_tensor * g = ggml_mul(p.ctx, x, ggml_sigmoid(p.ctx, z));
        p.output(g);
        ggml_tensor * mm = ggml_mul_mat(p.ctx, W, ggml_reshape_2d(p.ctx, g, (int64_t) D * H, T));
        p.output(mm);
        if (!p.alloc(be)) { fail(name + ": alloc"); return; }
        p.fill_rest();
        const Invariant inv(true);
        check(be, name, p, {p.index_of(mm)}, GGML_CUDA_QN_UNARY_MUL_Q8, 0);
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

    // ── The layer output's ADD (split mode: cold plus hot expert sums) folded into the next HC_COMBINE_NORM ─────────
    // Node by node is the reference. q8: an MMVQ product reads xn, so the fused launch writes its q8_1 rows too.
    // reuse: the ADD's sources are computed and the graph allocator places every tensor, as the scheduler does; it may
    // give the norm's output their memory, and then the fold must decline.
    void hc_cn_add(ggml_backend_t be, int T, bool q8, bool reuse) {
        const std::string name = "hc_cn_add T=" + std::to_string(T) + (q8 ? " q8" : "") + (reuse ? " reuse" : "");
        const int n_embd = 2560, hc = 4, rank = 320;
        Pattern p;
        ggml_tensor * inject = p.input(GGML_TYPE_F32, hc, T);
        ggml_tensor * residual = p.input(GGML_TYPE_F32, n_embd, hc, T);
        ggml_tensor * routed = p.input(GGML_TYPE_F32, n_embd, T);
        ggml_tensor * hot = p.input(GGML_TYPE_F32, n_embd, T);
        ggml_tensor * gamma = p.input(GGML_TYPE_F32, n_embd * hc);
        ggml_tensor * W = q8 ? p.input(GGML_TYPE_Q8_0, n_embd * hc, rank) : nullptr;
        ggml_tensor * x = reuse ? ggml_scale(p.ctx, routed, 0.5f) : routed;
        ggml_tensor * y = reuse ? ggml_scale(p.ctx, hot, 2.0f) : hot;
        ggml_tensor * add = ggml_add(p.ctx, x, y);   // moe_join
        ggml_tensor * cn = ggml_hc_combine_norm(p.ctx, inject, residual, ggml_reshape_3d(p.ctx, add, n_embd, 1, T),
                                                gamma, 1.0f / hc, 0.0f, 2.0f, 0.0f, 1e-6f);
        p.output(cn);
        if (W) {
            ggml_tensor * xn = ggml_view_4d(p.ctx, cn, n_embd, hc, T, 1, cn->nb[1], cn->nb[2], cn->nb[3],
                                            (size_t) n_embd * hc * T * sizeof(float));
            p.output(ggml_mul_mat(p.ctx, W, ggml_reshape_2d(p.ctx, xn, (int64_t) n_embd * hc, T)));
        }
        if (!(reuse ? p.alloc_graph(be) : p.alloc(be))) { fail(name + ": alloc"); return; }
        p.fill_rest();
        const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
            const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
            return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
        };
        const bool fold = !overlap(cn, x) && !overlap(cn, y);
        if (reuse) std::printf("  %s: %s\n", name.c_str(), fold ? "folds" : "declines (the norm sits on a source)");
        // The allocator may run the SCALEs in place over their inputs, so each run starts from the uploaded inputs.
        std::vector<std::vector<uint8_t>> saved;
        for (const ggml_tensor * t : p.inputs) saved.push_back(read(t));
        auto reset = [&] {
            for (size_t k = 0; k < p.inputs.size(); ++k) ggml_backend_tensor_set(p.inputs[k], saved[k].data(), 0, saved[k].size());
        };
        const Profile prof(true);
        const Invariant inv(true);
        Expect expect{{GGML_CUDA_QN_HC_CN_ADD, fold ? 1u : 0u}};
        if (W) expect.push_back({GGML_CUDA_QN_HC_CN_Q8, 1u});
        check(be, name, p, {}, expect, reset);
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

    // ── The whole HC low-rank gate and mix on RDNA4 (hc_mix_from_xn): the down product with SCALE and SILU, then the
    // up product with the mix reduce and the q8_1 rows of the projection that reads the mix, in two launches. Node by
    // node is the reference; the RDNA4 GPUs must fuse at 1..4 columns (several only under batch-invariant products).
    // The fused launches never write the SILU rows, so a graph that reads them back (lo_out) must not fuse. Without
    // cn read back the allocator may put the mix in place on xn (whole rows apart, as in the decode graphs). With
    // from_norm xn is a plain RMS_NORM and MUL (hc_mix, the MTP draft's mixes): the fusion quantizes xn itself, and
    // the inject product after the mix reads those rows.
    void hc_mix_rdna4(ggml_backend_t be, int T, bool invariant, bool lo_out = false, bool cn_out = true,
                      bool from_norm = false) {
        const std::string name = "hc_mix_rdna4 T=" + std::to_string(T) + (invariant ? "" : " (not batch-invariant)") +
                                 (lo_out ? " (SILU rows read back)" : "") + (cn_out ? "" : " (xn reusable)") +
                                 (from_norm ? " (xn from RMS_NORM)" : "");
        const int n_embd = 2560, hc = 4, rank = 320, N = 512;
        Pattern p;
        ggml_tensor * residual = p.input(GGML_TYPE_F32, n_embd, hc, T);
        ggml_tensor * gamma = p.input(GGML_TYPE_F32, n_embd * hc);
        ggml_tensor * w_down = p.input(GGML_TYPE_Q8_0, n_embd * hc, rank);
        ggml_tensor * w_up = p.input(GGML_TYPE_Q8_0, rank, n_embd * hc);
        ggml_tensor * W = p.input(GGML_TYPE_Q8_0, n_embd, N);
        ggml_tensor * cn = nullptr;
        ggml_tensor * xn = nullptr;
        ggml_tensor * w_inject = nullptr;
        if (from_norm) {
            w_inject = p.input(GGML_TYPE_Q8_0, n_embd * hc, hc);
            ggml_tensor * xn3 = ggml_mul(p.ctx, ggml_rms_norm(p.ctx, residual, 1e-6f), ggml_reshape_2d(p.ctx, gamma, n_embd, hc));
            xn = ggml_reshape_2d(p.ctx, xn3, (int64_t) n_embd * hc, T);
        } else {
            ggml_tensor * inject = p.input(GGML_TYPE_F32, hc, T);
            ggml_tensor * block = p.input(GGML_TYPE_F32, n_embd, T);
            cn = ggml_hc_combine_norm(p.ctx, inject, residual, ggml_reshape_3d(p.ctx, block, n_embd, 1, T),
                                      gamma, 1.0f / hc, 0.0f, 2.0f, 0.0f, 1e-6f);
            ggml_tensor * xn3 = ggml_view_4d(p.ctx, cn, n_embd, hc, T, 1, cn->nb[1], cn->nb[2], cn->nb[3],
                                             (size_t) n_embd * hc * T * sizeof(float));
            xn = ggml_reshape_2d(p.ctx, xn3, (int64_t) n_embd * hc, T);
        }
        ggml_tensor * lo = ggml_silu(p.ctx, ggml_scale(p.ctx, ggml_mul_mat(p.ctx, w_down, xn), 1.0f / hc));
        ggml_tensor * gate = ggml_sigmoid(p.ctx, ggml_mul_mat(p.ctx, w_up, lo));
        ggml_tensor * gated = ggml_reshape_3d(p.ctx, ggml_mul(p.ctx, xn, gate), n_embd, hc, T);
        const size_t row = ggml_row_size(GGML_TYPE_F32, n_embd);
        ggml_tensor * mixed = ggml_cont(p.ctx, ggml_view_2d(p.ctx, gated, n_embd, T, row * hc, 0));
        for (int s = 1; s < hc; ++s) mixed = ggml_add(p.ctx, mixed, ggml_view_2d(p.ctx, gated, n_embd, T, row * hc, row * s));
        mixed = ggml_scale(p.ctx, mixed, 1.0f / hc);
        if (cn_out && cn) p.output(cn);
        if (lo_out) p.output(lo);
        p.output(mixed);
        p.output(ggml_mul_mat(p.ctx, W, mixed));
        if (w_inject) p.output(ggml_mul_mat(p.ctx, w_inject, xn));
        if (!(cn_out ? p.alloc(be) : p.alloc_graph(be))) { fail(name + ": alloc"); return; }
        if (!cn_out && cn) {
            const uintptr_t a = (uintptr_t) mixed->data, b = (uintptr_t) cn->data;
            const bool alias = a <= b ? b - a < ggml_nbytes(mixed) : a - b < ggml_nbytes(cn);
            std::printf("  %s: the mix %s cn\n", name.c_str(), alias ? "aliases" : "does not alias");
        }
        p.fill_rest();
        // The graph allocator may hand an input's memory to a later node, so each run starts from the inputs.
        std::vector<std::vector<uint8_t>> saved;
        for (ggml_tensor * t : p.inputs) saved.push_back(read(t));
        auto reset = [&] {
            for (size_t k = 0; k < p.inputs.size(); ++k) {
                ggml_backend_tensor_set(p.inputs[k], saved[k].data(), 0, saved[k].size());
            }
        };
        const Profile prof(true);
        const Invariant inv(invariant);
        const bool fuses = ggml_backend_cuda_mmvq_twin_supported(be) && T <= 4 && (T == 1 || invariant) && !lo_out;
        check(be, name, p, {}, Expect{{GGML_CUDA_QN_HC_MIX_RDNA4, fuses ? 1 : 0}}, reset);
    }

    // ── The same gated unary in a qwen4exp graph: [D, H, T] rows read as [D*H, T] get their q8_1 rows when D*H needs
    // no row padding (MMVQ's rows are then the producer's in order); otherwise, or at prefill widths, none ─────────
    void unary_mul_long_rows_q8(ggml_backend_t be, int D, int H, int T, size_t fused) {
        const std::string name = "unary_mul_long_rows_q8 D=" + std::to_string(D) + " H=" + std::to_string(H) +
                                 " T=" + std::to_string(T);
        const int N = 2560;
        Pattern p;
        ggml_tensor * x = p.input(GGML_TYPE_F32, D, H, T);
        ggml_tensor * z = p.input(GGML_TYPE_F32, D, H, T);
        ggml_tensor * W = p.input(GGML_TYPE_Q8_0, (int64_t) D * H, N);
        ggml_tensor * g = ggml_mul(p.ctx, x, ggml_sigmoid(p.ctx, z));
        p.output(g);
        ggml_tensor * mm = ggml_mul_mat(p.ctx, W, ggml_reshape_2d(p.ctx, g, (int64_t) D * H, T));
        p.output(mm);
        if (!p.alloc(be)) { fail(name + ": alloc"); return; }
        p.fill_rest();
        const Profile prof(true);
        const Invariant inv(true);
        check(be, name, p, {p.index_of(mm)}, GGML_CUDA_QN_UNARY_MUL_Q8, fused);
    }

    // ── A GDN step's conv as build_linear_attn's step path builds it: concat, history copy, window copy (verify),
    // SSM_CONV, SILU and the q/k RMS_NORM + SCALE, in one launch ───────────────────────────────────────────────────
    void gdn_conv_step(ggml_backend_t be, int T, bool spec, bool norm, bool concat_read, size_t fused) {
        const std::string name = "gdn_conv_step T=" + std::to_string(T) + (spec ? " verify" : " decode") +
                                 (norm ? "" : " no-norm") + (concat_read ? " concat-read" : "");
        const int D = 128, Hk = 16, d_in = 6144, C = 2 * Hk * D + d_in, K = 4, max_verify = 8;
        const float eps = 1e-6f;
        Pattern p;
        ggml_tensor * qkv = p.input(GGML_TYPE_F32, C, T);
        ggml_tensor * conv_state = p.input(GGML_TYPE_F32, K - 1, C);
        ggml_tensor * spec_conv = spec ? p.input(GGML_TYPE_F32, K - 1, C, max_verify) : nullptr;
        ggml_tensor * conv1d = p.input(GGML_TYPE_F32, K, C);
        ggml_context * c = p.ctx;
        ggml_tensor * hist = ggml_reshape_3d(c, conv_state, K - 1, C, 1);
        ggml_tensor * qkv_t = ggml_transpose(c, ggml_reshape_2d(c, qkv, C, T));
        ggml_tensor * conv_input = ggml_concat(c, hist, qkv_t, 0);
        ggml_tensor * new_hist = ggml_view_3d(c, conv_input, K - 1, C, 1, conv_input->nb[1], conv_input->nb[2],
                                              (size_t) T * conv_input->nb[0]);
        ggml_build_forward_expand(p.gf, ggml_cpy(c, new_hist, ggml_reshape_3d(c, conv_state, K - 1, C, 1)));
        if (spec) {
            ggml_tensor * windows = ggml_view_3d(c, conv_input, K - 1, C, 1, conv_input->nb[1], conv_input->nb[2],
                                                 conv_input->nb[0]);
            windows->ne[2] = T;
            windows->nb[2] = conv_input->nb[0];
            windows->nb[3] = windows->nb[2] * T;
            ggml_build_forward_expand(p.gf, ggml_cpy(c, windows, ggml_view_3d(c, spec_conv, K - 1, C, T,
                spec_conv->nb[1], spec_conv->nb[2], 0)));
        }
        ggml_tensor * conv_op = ggml_ssm_conv(c, conv_input, conv1d);
        conv_op->src[3] = qkv_t;
        ggml_tensor * conv = ggml_silu(c, conv_op);
        if (norm) {
            ggml_tensor * qk = ggml_scale(c, ggml_rms_norm(c, ggml_view_3d(c, conv, D, 2 * Hk, T, D * sizeof(float),
                (size_t) C * sizeof(float), 0), eps / (float) D), 1.0f / sqrtf((float) D));
            qk->src[1] = qkv_t;
            p.output(qk);
        }
        p.output(conv);
        if (concat_read) p.output(conv_input);
        p.outs.push_back(conv_state);
        if (spec) p.outs.push_back(spec_conv);
        if (!p.alloc(be)) { fail(name + ": alloc"); return; }
        p.fill_rest();
        // The step rewrites the history (and the windows) in place: both runs start from the same values.
        const std::vector<uint8_t> state0 = read(conv_state);
        const std::vector<uint8_t> spec0 = spec ? read(spec_conv) : std::vector<uint8_t>();
        auto reset = [&] {
            ggml_backend_tensor_set(conv_state, state0.data(), 0, state0.size());
            if (spec) ggml_backend_tensor_set(spec_conv, spec0.data(), 0, spec0.size());
        };
        const Profile prof(true);
        check(be, name, p, {}, Expect{{GGML_CUDA_QN_GDN_CONV, fused}}, reset);
    }

    // ── The GDN output gate (RMS_NORM, MUL, SIGMOID, MUL) in one launch, writing ssm_out's q8_1 rows ───────────────
    void gated_norm(ggml_backend_t be, int T, bool profile) {
        const std::string name = "gated_norm T=" + std::to_string(T) + (profile ? "" : " (no profile)");
        const int D = 128, Hv = 48, d_in = D * Hv, n_embd = 2560;
        Pattern p;
        ggml_tensor * attn = p.input(GGML_TYPE_F32, D, Hv, T);
        ggml_tensor * gamma = p.input(GGML_TYPE_F32, D);
        ggml_tensor * z = p.input(GGML_TYPE_F32, d_in, T);
        ggml_tensor * W = p.input(GGML_TYPE_Q8_0, d_in, n_embd);
        ggml_context * c = p.ctx;
        // build_linear_attn's tail.
        ggml_tensor * normed = ggml_mul(c, ggml_rms_norm(c, attn, 1e-6f), gamma);
        ggml_tensor * out = ggml_mul(c, normed, ggml_sigmoid(c, ggml_reshape_4d(c, z, D, Hv, T, 1)));
        p.output(out);
        p.output(ggml_mul_mat(c, W, ggml_reshape_3d(c, out, d_in, T, 1)));
        if (!p.alloc(be)) { fail(name + ": alloc"); return; }
        p.fill_rest();
        const Profile prof(profile);
        const Invariant inv(true);
        const size_t fused = profile ? 1 : 0;
        check(be, name, p, {}, Expect{{GGML_CUDA_QN_GATED_NORM, fused}, {GGML_CUDA_QN_GATED_NORM_Q8, fused}});
    }

    // ── RMS_NORM then SCALE (the GDN q/k norm outside the conv step) in one launch ────────────────────────────────
    void rms_norm_scale(ggml_backend_t be, int T) {
        const std::string name = "rms_norm_scale T=" + std::to_string(T);
        const int D = 128, H = 32, C = 10240;
        Pattern p;
        ggml_tensor * x = p.input(GGML_TYPE_F32, C, T);
        ggml_tensor * v = ggml_view_3d(p.ctx, x, D, H, T, D * sizeof(float), x->nb[1], 0);
        p.output(ggml_scale(p.ctx, ggml_rms_norm(p.ctx, v, 1e-6f / D), 1.0f / sqrtf((float) D)));
        if (!p.alloc(be)) { fail(name + ": alloc"); return; }
        p.fill_rest();
        const Profile prof(true);
        check(be, name, p, {}, GGML_CUDA_QN_RMS_NORM_SCALE, 1);
    }

    // ── The argmax greedy decode reads: ties go to the lowest index, as a host scan's ──────────────────────────────
    // Rows of V >= 8192 take the sliced two-launch kernel (an LM head's 248320 / 64000 logits per token), shorter ones
    // one block per row; both must pick what `if (x[i] > x[best]) best = i` from best = 0 picks, NaNs and -inf too
    // (nans: the CPU backend's argmax starts from -inf instead, so a NaN at x[0] is the one case it differs on).
    void argmax_ties(ggml_backend_t be, int V, bool nans = true) {
        const int rows = 10;
        Pattern p;
        ggml_tensor * logits = p.input(GGML_TYPE_F32, V, rows);
        ggml_tensor * am = ggml_argmax(p.ctx, logits);
        p.output(am);
        const std::string name = "argmax V=" + std::to_string(V);
        if (!p.alloc(be)) { fail(name + ": alloc"); return; }
        std::vector<float> h((size_t) V * rows);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float & v : h) v = nd(p.rng);
        const float nan = std::nanf("");
        // Equal maxima in different threads, warps, strides and slices of the kernels' blocks.
        const int ties[6][3] = {{1, 1024, V * 7 / 10}, {37, 5, -1}, {3, 700, V - 1}, {V * 2 / 3, 31, 32}, {0, 1, 2},
                                {-1, -1, -1}};
        for (int r = 0; r < rows; ++r) {
            float * row = h.data() + (size_t) r * V;
            if (r < 5) {
                for (int j = 0; j < 3; ++j) if (ties[r][j] >= 0) row[ties[r][j]] = 50.0f;
            } else if (r == 5) {
                std::fill(row, row + V, 0.25f);                        // every entry ties
            } else if (r == 8) {
                std::fill(row, row + V, -INFINITY);                    // all -inf: the first
            } else if (!nans) {
                continue;
            } else if (r == 6) {
                row[V / 2] = 50.0f;                                    // NaNs around and above the maximum are never picked
                for (int j = 0; j < V; j += 997) row[j == 0 ? 1 : j] = nan;
                row[V - 1] = nan;
            } else if (r == 7) {
                row[0] = nan;                                          // x[0] a NaN: the host scan stays at 0
                row[V / 3] = 50.0f;
            } else {
                std::fill(row, row + V, nan);                          // all NaN: 0
                row[V - 2] = -INFINITY;
            }
        }
        p.set(logits, h.data());
        if (ggml_backend_graph_compute(be, p.gf) != GGML_STATUS_SUCCESS) { fail(name + ": compute"); return; }
        std::vector<int32_t> got(rows);
        ggml_backend_tensor_get(am, got.data(), 0, sizeof(int32_t) * rows);
        for (int r = 0; r < rows; ++r) {
            const float * row = h.data() + (size_t) r * V;
            int want = 0;
            for (int i = 1; i < V; ++i) if (row[i] > row[want]) want = i;
            if (got[r] != want) fail(name + " row " + std::to_string(r) + ": device " + std::to_string(got[r]) +
                                     ", host " + std::to_string(want));
        }
    }

    // ── Two Q8_0 products of the same activations (the GDN step's qkv and z) in one RDNA4 launch ────────────────────
    // Rows counts that take different block shapes (more and fewer than 1024 rows of 80 blocks) stay two launches.
    void mmvq_twin(ggml_backend_t be, int T, int rows_a, int rows_b, bool invariant) {
        const std::string name = "mmvq_twin T=" + std::to_string(T) + " rows=" + std::to_string(rows_a) + "+" +
                                 std::to_string(rows_b) + (invariant ? "" : " (not batch-invariant)");
        const int K = 2560;
        Pattern p;
        ggml_tensor * x = p.input(GGML_TYPE_F32, K, T);
        ggml_tensor * wa = p.input(GGML_TYPE_Q8_0, K, rows_a);
        ggml_tensor * wb = p.input(GGML_TYPE_Q8_0, K, rows_b);
        p.output(ggml_mul_mat(p.ctx, wa, x));
        p.output(ggml_mul_mat(p.ctx, wb, x));
        if (!p.alloc(be)) { fail(name + ": alloc"); return; }
        p.fill_rest();
        const Profile prof(true);
        const Invariant inv(invariant);
        const bool same_shape = (rows_a > 1024) == (rows_b > 1024);
        const bool fuses = ggml_backend_cuda_mmvq_twin_supported(be) && same_shape && (T == 1 || invariant);
        check(be, name, p, {}, GGML_CUDA_QN_MMVQ_TWIN, fuses ? 1 : 0);
    }

    // ── Two F32 products in one launch (the GDN step's HC inject, then its stacked alpha/beta gates) ────────────────
    // Each row must have the bits of its own MMVF launch. dependent: b reads a's output, so they must stay apart.
    void mmvf_twin(ggml_backend_t be, int T, int K_a, int rows_a, int K_b, int rows_b, bool dependent) {
        const std::string name = "mmvf_twin T=" + std::to_string(T) + " " + std::to_string(K_a) + "x" +
                                 std::to_string(rows_a) + " + " + std::to_string(K_b) + "x" + std::to_string(rows_b) +
                                 (dependent ? " (dependent)" : "");
        Pattern p;
        ggml_tensor * wa = p.input(GGML_TYPE_F32, K_a, rows_a);
        ggml_tensor * xa = p.input(GGML_TYPE_F32, K_a, T);
        ggml_tensor * wb = p.input(GGML_TYPE_F32, dependent ? rows_a : K_b, rows_b);
        ggml_tensor * ya = ggml_mul_mat(p.ctx, wa, xa);
        ggml_tensor * yb = ggml_mul_mat(p.ctx, wb, dependent ? ya : p.input(GGML_TYPE_F32, K_b, T));
        p.output(ya);
        p.output(yb);
        if (!p.alloc(be)) { fail(name + ": alloc"); return; }
        p.fill_rest();
        const Profile prof(true);
        const Invariant inv(true);
        check(be, name, p, {}, GGML_CUDA_QN_MMVF_TWIN, hc_supported(be) && !dependent ? 1 : 0);
    }

    // ── The next layer's weights read into the last-level cache in the wait for the expert device ─────────────────
    // qwen4exp's split steps put a GGML_MOE_FUSED_PREFETCH node after the target's hot experts. It must leave every
    // other node's bytes and the weights it reads as they are, under any byte budget (cut inside a tensor, past all).
    void moe_prefetch(ggml_backend_t be, size_t budget) {
        const std::string name = "moe_prefetch budget=" + std::to_string(budget);
        std::vector<uint8_t> want;
        for (const bool with : {false, true}) {
            Pattern p;   // the same seed: both graphs get the same inputs
            ggml_tensor * wa = p.input(GGML_TYPE_Q8_0, 2560, 512);
            ggml_tensor * x  = p.input(GGML_TYPE_F32, 2560, 3);
            ggml_tensor * wb = p.input(GGML_TYPE_Q8_0, 2560, 2048);
            ggml_tensor * wc = p.input(GGML_TYPE_F32, 2560, 48);
            ggml_tensor * ya = ggml_mul_mat(p.ctx, wa, x);
            p.output(ya);
            ggml_tensor * pf = nullptr;
            if (with) {
                ggml_tensor * srcs[] = {wb, wc};
                pf = ggml_moe_fused_prefetch(p.ctx, ya, srcs, 2, budget);
                ggml_build_forward_expand(p.gf, pf);
            }
            ggml_tensor * yb = ggml_mul_mat(p.ctx, wb, x);
            ggml_tensor * yc = ggml_mul_mat(p.ctx, wc, x);
            p.output(yb);
            p.output(yc);
            if (!p.alloc(be)) { fail(name + ": alloc"); return; }
            p.fill_rest();
            if (with && !ggml_backend_supports_op(be, pf)) { fail(name + ": not supported"); return; }
            const std::vector<uint8_t> wb0 = read(wb), wc0 = read(wc);
            const Profile prof(true);
            if (ggml_backend_graph_compute(be, p.gf) != GGML_STATUS_SUCCESS) { fail(name + ": compute"); return; }
            if (read(wb) != wb0 || read(wc) != wc0) fail(name + ": a prefetched weight changed");
            std::vector<uint8_t> got = read(ya);
            for (ggml_tensor * t : {yb, yc}) {
                const std::vector<uint8_t> v = read(t);
                got.insert(got.end(), v.begin(), v.end());
            }
            if (!with) want = got;
            else if (got != want) fail(name + ": outputs differ with the prefetch");
        }
    }

    // ── The expert device's read-ahead of a split layer's picks ──────────────────────────────────────────────────
    // qwen4exp's split steps put a GGML_MOE_FUSED_PREFETCH_ROWS node before each split layer's experts: each token's
    // top picks of a router copy over the cold experts, mapped to expert ids. It must leave the experts' outputs and
    // the stacks it reads as they are, at any width and cap (below the picks, above them, above the kernel's 64), and
    // with picks the read skips: repeats, -1, out of range, unmapped (-1) and out-of-range map entries.
    void moe_prefetch_rows(ggml_backend_t be, int T, int max_experts, bool odd) {
        const std::string name = "moe_prefetch_rows T=" + std::to_string(T) + " max=" + std::to_string(max_experts) +
                                 (odd ? " odd picks" : "");
        const int n_embd = 512, n_ff = 256, n_exp = 24, n_used = 4, n_cold = 16, k = 4;
        std::vector<int32_t> ids(n_used * T), map(n_cold), raw(3 * T);
        for (int i = 0; i < n_used * T; ++i) ids[i] = (i * 5 + 1) % n_exp;
        for (int j = 0; j < n_cold; ++j) map[j] = (j * 7 + 3) % n_exp;
        if (odd) { map[2] = -1; map[5] = n_exp + 2; }
        for (int t = 0; t < T; ++t) { raw[3 * t] = t % 2 ? -1 : 7; raw[3 * t + 1] = 7; raw[3 * t + 2] = n_exp + t; }
        std::vector<uint8_t> want;
        for (const bool with : {false, true}) {
            Pattern p;   // the same seed: both graphs get the same inputs
            ggml_tensor * gate = p.input(GGML_TYPE_Q4_K, n_embd, n_ff, n_exp);
            ggml_tensor * up = p.input(GGML_TYPE_Q4_K, n_embd, n_ff, n_exp);
            ggml_tensor * down = p.input(GGML_TYPE_Q8_0, n_ff, n_embd, n_exp);
            ggml_tensor * router = p.input(GGML_TYPE_Q8_0, n_embd, n_cold);
            ggml_tensor * x = p.input(GGML_TYPE_F32, n_embd, 1, T);
            ggml_tensor * sel = p.input(GGML_TYPE_I32, n_used, T);
            ggml_tensor * cold = p.input(GGML_TYPE_I32, n_cold);
            ggml_tensor * picks = p.input(GGML_TYPE_I32, 3, T);
            ggml_tensor * logits = ggml_mul_mat(p.ctx, router, x);   // [n_cold, 1, T]
            p.output(logits);
            std::vector<ggml_tensor *> pfs;
            if (with) {
                ggml_tensor * stacks[3] = {gate, up, down};
                ggml_tensor * top = ggml_argsort_top_k(p.ctx, logits, k);
                pfs.push_back(ggml_moe_fused_prefetch_rows(p.ctx, top, cold, stacks, 3, max_experts));
                if (odd) pfs.push_back(ggml_moe_fused_prefetch_rows(p.ctx, picks, nullptr, stacks, 3, max_experts));
                for (ggml_tensor * pf : pfs) ggml_build_forward_expand(p.gf, pf);
            }
            ggml_tensor * yg = ggml_mul_mat_id(p.ctx, gate, x, sel);   // [n_ff, n_used, T]
            ggml_tensor * yu = ggml_mul_mat_id(p.ctx, up, x, sel);
            ggml_tensor * yd = ggml_mul_mat_id(p.ctx, down, yg, sel);  // [n_embd, n_used, T]
            for (ggml_tensor * t : {yg, yu, yd}) p.output(t);
            if (!p.alloc(be)) { fail(name + ": alloc"); return; }
            p.set(sel, ids.data());
            p.set(cold, map.data());
            p.set(picks, raw.data());
            p.fill_rest();
            for (ggml_tensor * pf : pfs) {
                if (!ggml_backend_supports_op(be, pf)) { fail(name + ": not supported"); return; }
            }
            const std::vector<uint8_t> g0 = read(gate), u0 = read(up), d0 = read(down);
            const Profile prof(true);
            if (ggml_backend_graph_compute(be, p.gf) != GGML_STATUS_SUCCESS) { fail(name + ": compute"); return; }
            if (read(gate) != g0 || read(up) != u0 || read(down) != d0) fail(name + ": a read stack changed");
            std::vector<uint8_t> got;
            for (ggml_tensor * t : {logits, yg, yu, yd}) {
                const std::vector<uint8_t> v = read(t);
                got.insert(got.end(), v.begin(), v.end());
            }
            if (!with) want = got;
            else if (got != want) fail(name + ": outputs differ with the read-ahead");
        }
    }

    // ── The gfx1151 Q4_K expert loop at any rows per wave ────────────────────────────────────────────────────────
    // The Strix runs the cold gate/up experts through mul_mat_vec_q_moe_q4k_prefetch, two weight rows per wave. One,
    // two and four rows must give the same bytes, fused with the SwiGLU or not, with and without masked (-1) picks.
    // Elsewhere the override is ignored and the case compares the generic loop with itself.
    void q4k_rows(ggml_backend_t be, int T, int masked, bool fused) {
        const std::string name = "q4k_rows T=" + std::to_string(T) + " masked=" + std::to_string(masked) +
                                 (fused ? " fused" : " unfused");
        const int n_embd = 2560, n_ff = 640, n_exp = 16, n_used = 10;
        std::vector<int32_t> ids(n_used * T);
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < n_used; ++k) ids[t * n_used + k] = k < masked ? -1 : (t * 3 + k * 5) % n_exp;
        }
        std::vector<uint8_t> want;
        for (const int rows : {1, 2, 4}) {
            Pattern p;   // the same seed: every run gets the same inputs
            ggml_tensor * gate = p.input(GGML_TYPE_Q4_K, n_embd, n_ff, n_exp);
            ggml_tensor * up = p.input(GGML_TYPE_Q4_K, n_embd, n_ff, n_exp);
            ggml_tensor * x = p.input(GGML_TYPE_F32, n_embd, 1, T);
            ggml_tensor * sel = p.input(GGML_TYPE_I32, n_used, T);
            ggml_tensor * yg = ggml_mul_mat_id(p.ctx, gate, x, sel);   // [n_ff, n_used, T]
            ggml_tensor * yu = ggml_mul_mat_id(p.ctx, up, x, sel);
            ggml_tensor * gu = ggml_swiglu_split(p.ctx, yg, yu);
            if (!fused) { p.output(yg); p.output(yu); }
            p.output(gu);
            if (!p.alloc(be)) { fail(name + ": alloc"); return; }
            p.set(sel, ids.data());
            p.fill_rest();
            const Profile prof(true);
            const bool fusion_was_off = ggml_backend_cuda_set_fusion_disabled_override(!fused);
            const int rows_before = ggml_backend_cuda_set_mmvq_moe_prefetch_rows_override(rows);
            const bool ok = ggml_backend_graph_compute(be, p.gf) == GGML_STATUS_SUCCESS;
            ggml_backend_cuda_set_mmvq_moe_prefetch_rows_override(rows_before);
            ggml_backend_cuda_set_fusion_disabled_override(fusion_was_off);
            if (!ok) { fail(name + ": compute"); return; }
            std::vector<uint8_t> got;
            for (ggml_tensor * t : p.outs) {
                const std::vector<uint8_t> v = read(t);
                got.insert(got.end(), v.begin(), v.end());
            }
            if (rows == 1) want = got;
            else if (got != want) fail(name + ": " + std::to_string(rows) + " rows per wave differ from 1");
        }
    }

    // ── Stacked projections: one F32 matmul over two weights that sit back to back ───────────────────────────────
    // GDN steps project alpha and beta with one [K, 2R] view (qwen4exp_loader's ssm_alpha_beta). Its rows must be the
    // bytes the two [K, R] products give, at every step width.
    void stacked_rows(ggml_backend_t be, int T, int K, int rows_a, int rows_b) {
        const std::string name = "stacked_rows T=" + std::to_string(T) + " K=" + std::to_string(K) + " rows=" +
                                 std::to_string(rows_a) + "+" + std::to_string(rows_b);
        Pattern p;
        ggml_tensor * x = p.input(GGML_TYPE_F32, K, T);
        ggml_tensor * wab = p.input(GGML_TYPE_F32, K, rows_a + rows_b);
        ggml_tensor * wa = p.input(GGML_TYPE_F32, K, rows_a);
        ggml_tensor * wb = p.input(GGML_TYPE_F32, K, rows_b);
        ggml_tensor * yab = ggml_mul_mat(p.ctx, wab, x);
        ggml_tensor * ya = ggml_mul_mat(p.ctx, wa, x);
        ggml_tensor * yb = ggml_mul_mat(p.ctx, wb, x);
        for (ggml_tensor * t : {yab, ya, yb}) p.output(t);
        if (!p.alloc(be)) { fail(name + ": alloc"); return; }
        p.fill_f32(x);
        p.fill_f32(wab, 0.05f);
        const std::vector<uint8_t> w = read(wab);
        const size_t split = (size_t) rows_a * K * sizeof(float);
        ggml_backend_tensor_set(wa, w.data(), 0, split);
        ggml_backend_tensor_set(wb, w.data() + split, 0, w.size() - split);
        const Profile prof(true);
        const Invariant inv(true);
        if (ggml_backend_graph_compute(be, p.gf) != GGML_STATUS_SUCCESS) { fail(name + ": compute"); return; }
        const std::vector<uint8_t> all = read(yab), a = read(ya), b = read(yb);
        const size_t ra = (size_t) rows_a * sizeof(float), rb = (size_t) rows_b * sizeof(float);
        for (int t = 0; t < T; ++t) {
            const uint8_t * row = all.data() + (size_t) t * (ra + rb);
            if (memcmp(row, a.data() + (size_t) t * ra, ra) != 0) fail(name + ": first rows differ at token " + std::to_string(t));
            if (memcmp(row + ra, b.data() + (size_t) t * rb, rb) != 0) fail(name + ": last rows differ at token " + std::to_string(t));
        }
    }

    // ── The recurrence reading its raw gates as the two strided halves of one [2*Hv, T] projection ────────────────
    // ggml_gated_delta_net_shared_gates over views, against the in-place step over contiguous copies of the same
    // gates: the attention output and the updated state must match bit for bit.
    void gdn_shared_gates(ggml_backend_t be, int T) {
        const std::string name = "gdn_shared_gates T=" + std::to_string(T);
        const int D = 128, Hk = 16, Hv = 48;
        Pattern p;
        ggml_tensor * q = p.input(GGML_TYPE_F32, D, Hk, T);
        ggml_tensor * k = p.input(GGML_TYPE_F32, D, Hk, T);
        ggml_tensor * v = p.input(GGML_TYPE_F32, D, Hv, T);
        ggml_tensor * ab = p.input(GGML_TYPE_F32, 2 * Hv, T);
        ggml_tensor * gba = p.input(GGML_TYPE_F32, 2 * Hv);
        ggml_tensor * st1 = p.input(GGML_TYPE_F32, D * D, Hv);
        ggml_tensor * st2 = p.input(GGML_TYPE_F32, D * D, Hv);
        ggml_tensor * g = ggml_view_4d(p.ctx, ab, 1, Hv, T, 1, ab->nb[0], ab->nb[1], ab->nb[1] * T, 0);
        ggml_tensor * beta = ggml_view_4d(p.ctx, ab, 1, Hv, T, 1, ab->nb[0], ab->nb[1], ab->nb[1] * T, Hv * ab->nb[0]);
        ggml_tensor * s1 = ggml_reshape_4d(p.ctx, st1, D, D, Hv, 1);
        ggml_tensor * s2 = ggml_reshape_4d(p.ctx, st2, D, D, Hv, 1);
        ggml_tensor * shared = ggml_gated_delta_net_shared_gates(p.ctx, q, k, v, g, beta, s1, true);
        ggml_tensor * plain = ggml_gated_delta_net_inplace(p.ctx, q, k, v, ggml_cont(p.ctx, g), ggml_cont(p.ctx, beta), s2);
        for (ggml_tensor * gdn : {shared, plain}) {
            ggml_gated_delta_net_set_raw_gates(gdn, gba);
            ggml_gated_delta_net_set_skip_intermediate(gdn, true);
        }
        ggml_tensor * out1 = ggml_view_1d(p.ctx, shared, (int64_t) D * Hv * T, 0);
        ggml_tensor * out2 = ggml_view_1d(p.ctx, plain, (int64_t) D * Hv * T, 0);
        p.output(ggml_cont(p.ctx, out1));
        p.output(ggml_cont(p.ctx, out2));
        if (!p.alloc(be)) { fail(name + ": alloc"); return; }
        for (ggml_tensor * t : {q, k}) p.fill_f32(t, 0.09f);
        p.fill_f32(v, 0.5f);
        p.fill_f32(ab, 1.0f);
        std::vector<float> hba(2 * Hv);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        for (int h = 0; h < Hv; ++h) { hba[h] = u(p.rng) - 0.5f; hba[Hv + h] = -0.1f - u(p.rng); }   // dt_bias | A
        p.set(gba, hba.data());
        p.fill_f32(st1, 0.02f);
        const std::vector<uint8_t> st0 = read(st1);
        ggml_backend_tensor_set(st2, st0.data(), 0, st0.size());
        const Profile prof(true);
        if (ggml_backend_graph_compute(be, p.gf) != GGML_STATUS_SUCCESS) { fail(name + ": compute"); return; }
        if (read(p.outs[0]) != read(p.outs[1])) fail(name + ": attention output differs");
        if (read(st1) != read(st2)) fail(name + ": state differs");
        if (read(st1) == st0) fail(name + ": the state did not change");
    }

    // The router softmax ([512, T], one 512-thread block per row) while other work runs on the same GPU: a second
    // backend on the device (its own stream) keeps it busy with matmuls. Every result must match the softmax computed
    // on an otherwise idle GPU bit for bit. Its max and sum block reductions share one shared-memory buffer; without a
    // barrier between them a warp that is ahead overwrites a maximum another warp has not read yet, which the R9700
    // did in up to 80% of launches beside other work (none when idle).
    void softmax_beside_load(ggml_backend_t be, int64_t T) {
        const std::string what = "softmax [512, " + std::to_string(T) + "] beside other work";
        Pattern p;
        ggml_tensor * logits = p.input(GGML_TYPE_F32, 512, T);
        p.output(ggml_soft_max(p.ctx, logits));
        if (!p.alloc(be)) { fail(what + ": alloc"); return; }
        p.fill_f32(logits, 1.5f);
        if (ggml_backend_graph_compute(be, p.gf) != GGML_STATUS_SUCCESS) { fail(what + ": compute"); return; }
        const std::vector<uint8_t> want = read(p.outs[0]);

        ggml_backend_ptr side(ggml_backend_dev_init(ggml_backend_get_device(be), nullptr));
        if (!side) { fail(what + ": second backend"); return; }
        Pattern load;
        ggml_tensor * w = load.input(GGML_TYPE_F32, 2048, 2048);
        ggml_tensor * y = load.input(GGML_TYPE_F32, 2048, 64);
        for (int k = 0; k < 8; ++k) y = ggml_mul_mat(load.ctx, w, y);
        load.output(y);
        if (!load.alloc(side.get())) { fail(what + ": load alloc"); return; }
        load.fill_f32(w, 0.02f);
        load.fill_rest();

        const int reps = 300;
        int differ = 0;
        for (int r = 0; r < reps; ++r) {
            if (ggml_backend_graph_compute_async(side.get(), load.gf) != GGML_STATUS_SUCCESS ||
                ggml_backend_graph_compute_async(be, p.gf) != GGML_STATUS_SUCCESS) {
                fail(what + ": compute");
                return;
            }
            ggml_backend_synchronize(be);
            if (read(p.outs[0]) != want) ++differ;
            ggml_backend_synchronize(side.get());
        }
        if (differ) fail(what + ": " + std::to_string(differ) + " of " + std::to_string(reps) + " results differ");
    }
};

}  // namespace

TEST_CASE(QnSplitFusionsFixture, route_norm) {
    each_gpu([&](ggml_backend_t be) {
        for (int T : {1, 3, 8}) route_norm(be, 512, 10, T, true);
        route_norm(be, 256, 6, 2, false);   // DeepSeek4-sized router outside the profile: untouched
    });
}

TEST_CASE(QnSplitFusionsFixture, route_norm_reuse) {
    each_gpu([&](ggml_backend_t be) {
        for (int T : {1, 8, 33, 86, 600, 1024, 1500}) route_norm_reuse(be, 512, 10, T);
    });
}

TEST_CASE(QnSplitFusionsFixture, router_topk) {
    each_gpu([&](ggml_backend_t be) {
        for (int T : {1, 2, 3, 8}) router_topk(be, 512, 10, T, 2, false, true);   // split layers, decode and verify
        for (int T : {1, 3}) router_topk(be, 512, 10, T, 0, false, true);         // plain layers: picks read outside
        router_topk(be, 512, 10, 3, 2, true, true);
        router_topk(be, 512, 10, 1, 0, true, true);
        router_topk(be, 512, 10, 3, 1, false, true);
        router_topk(be, 256, 8, 3, 2, false, true);
        router_topk(be, 512, 10, 3, 2, false, false);   // outside the profile: untouched
    });
}

TEST_CASE(QnSplitFusionsFixture, softmax_beside_load) {
    each_gpu([&](ggml_backend_t be) { for (int T : {1, 4, 8}) softmax_beside_load(be, T); });
}

TEST_CASE(QnSplitFusionsFixture, shared_gate) {
    each_gpu([&](ggml_backend_t be) { for (int T : {1, 2, 8}) shared_gate(be, 2560, 10, T); });
}

TEST_CASE(QnSplitFusionsFixture, unary_mul_q8) {
    each_gpu([&](ggml_backend_t be) { for (int T : {1, 2, 4}) unary_mul_q8(be, T); });
}

TEST_CASE(QnSplitFusionsFixture, unary_mul_unread_q8) {
    each_gpu([&](ggml_backend_t be) {
        for (int T : {1, 4}) unary_mul_unread_q8(be, 128, 32, T);
        unary_mul_unread_q8(be, 4096, 1, 64);
    });
}

TEST_CASE(QnSplitFusionsFixture, unary_mul_long_rows_q8) {
    each_gpu([&](ggml_backend_t be) {
        for (int T : {1, 4}) unary_mul_long_rows_q8(be, 128, 32, T, 1);
        unary_mul_long_rows_q8(be, 128, 3, 4, 0);    // 384-wide rows: MMVQ pads them
        unary_mul_long_rows_q8(be, 128, 32, 64, 0);  // prefill width: not MMVQ
    });
}

TEST_CASE(QnSplitFusionsFixture, gdn_conv_step) {
    bool ran = false;
    each_gpu([&](ggml_backend_t be) {
        if (!hc_supported(be)) return;
        ran = true;
        for (int T = 1; T <= 8; ++T) gdn_conv_step(be, T, true, true, false, 1);   // verify
        for (int T : {1, 2}) gdn_conv_step(be, T, false, true, false, 1);         // decode
        gdn_conv_step(be, 3, true, true, true, 1);     // the concat is read elsewhere: written too
        gdn_conv_step(be, 4, true, false, false, 1);   // no q/k norm after the conv
        gdn_conv_step(be, 9, false, true, false, 0);   // more tokens than a step
    });
    if (!ran) SKIP("the conv step fusion is HIP-only");
}

TEST_CASE(QnSplitFusionsFixture, gated_norm) {
    each_gpu([&](ggml_backend_t be) {
        for (int T = 1; T <= 8; ++T) gated_norm(be, T, true);
        gated_norm(be, 2, false);
    });
}

TEST_CASE(QnSplitFusionsFixture, rms_norm_scale) {
    bool ran = false;
    each_gpu([&](ggml_backend_t be) {
        if (!hc_supported(be)) return;
        ran = true;
        for (int T : {1, 8, 32}) rms_norm_scale(be, T);
    });
    if (!ran) SKIP("the split RMS_NORM + SCALE fusion is HIP-only");
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

TEST_CASE(QnSplitFusionsFixture, hc_cn_add) {
    bool ran = false;
    each_gpu([&](ggml_backend_t be) {
        if (!hc_supported(be)) return;
        ran = true;
        for (int T : {1, 2, 3, 8}) hc_cn_add(be, T, false, false);
        for (int T : {1, 3}) {
            hc_cn_add(be, T, true, false);
            hc_cn_add(be, T, false, true);
            hc_cn_add(be, T, true, true);
        }
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

TEST_CASE(QnSplitFusionsFixture, hc_mix_rdna4) {
    bool ran = false;
    each_gpu([&](ggml_backend_t be) {
        if (!hc_supported(be)) return;
        ran = true;
        for (int T : {1, 2, 3, 4, 8}) hc_mix_rdna4(be, T, true);
        hc_mix_rdna4(be, 3, false);
        hc_mix_rdna4(be, 3, true, /*lo_out=*/true);
        for (int T : {1, 2, 3, 4}) hc_mix_rdna4(be, T, true, /*lo_out=*/false, /*cn_out=*/false);
        for (int T : {1, 2, 3}) hc_mix_rdna4(be, T, true, /*lo_out=*/false, /*cn_out=*/true, /*from_norm=*/true);
    });
    if (!ran) SKIP("the HC mix fusion is HIP-only");
}

TEST_CASE(QnSplitFusionsFixture, mmvq_twin) {
    each_gpu([&](ggml_backend_t be) {
        for (int T : {1, 2, 3, 8}) mmvq_twin(be, T, 10240, 6144, true);
        mmvq_twin(be, 1, 6144, 10240, false);
        mmvq_twin(be, 2, 10240, 6144, false);
        mmvq_twin(be, 4, 10240, 512, true);
        mmvq_twin(be, 4, 1023, 513, true);
    });
}

TEST_CASE(QnSplitFusionsFixture, mmvf_twin) {
    each_gpu([&](ggml_backend_t be) {
        for (int T : {1, 2, 3, 4}) mmvf_twin(be, T, 10240, 4, 2560, 96, false);
        mmvf_twin(be, 3, 2560, 96, 10240, 4, false);
        mmvf_twin(be, 3, 2560, 2560, 0, 96, true);
    });
}

TEST_CASE(QnSplitFusionsFixture, moe_prefetch) {
    each_gpu([&](ggml_backend_t be) {
        for (size_t budget : {(size_t) 0, (size_t) 1 << 20, (size_t) 6 << 20, (size_t) 64 << 20}) moe_prefetch(be, budget);
    });
}

TEST_CASE(QnSplitFusionsFixture, moe_prefetch_rows) {
    each_gpu([&](ggml_backend_t be) {
        for (int T : {1, 3, 4}) {
            for (int max_experts : {2, 12, 100}) {
                for (bool odd : {false, true}) moe_prefetch_rows(be, T, max_experts, odd);
            }
        }
    });
}

TEST_CASE(QnSplitFusionsFixture, q4k_rows) {
    each_gpu([&](ggml_backend_t be) {
        for (int T : {1, 2, 3, 4}) {
            for (int masked : {0, 6}) {
                for (bool fused : {true, false}) q4k_rows(be, T, masked, fused);
            }
        }
    });
}

TEST_CASE(QnSplitFusionsFixture, stacked_rows) {
    each_gpu([&](ggml_backend_t be) {
        for (int T : {1, 2, 3, 4, 8}) stacked_rows(be, T, 2560, 48, 48);   // ssm_alpha | ssm_beta
        for (int T : {1, 3}) stacked_rows(be, T, 2560, 512, 1);           // router | shared-expert gate
    });
}

TEST_CASE(QnSplitFusionsFixture, gdn_shared_gates) {
    each_gpu([&](ggml_backend_t be) { for (int T : {1, 2, 3, 8}) gdn_shared_gates(be, T); });
}

// The LM head row split (qwen4exp_graph.cpp, build_lm_head). On every GPU that runs the RDNA4 Q8_0 matvec for a
// flagged product (ggml_backend_cuda_rdna4_rows_device), the flagged product of a Q8_0 matrix, and its first rows and
// its last rows (a matrix of their own, as on the expert device) joined by a concat, must give the first such GPU's
// product bit for bit, at one column and at batch-invariant widths. The unflagged product's differing outputs are
// printed: on RDNA3.5 its own matvec sums in another order, so the flag is what makes the rows match.
TEST_CASE(QnSplitFusionsFixture, lm_head_row_split) {
    std::vector<int> devs;
    for (int d = 0; d < ggml_backend_cuda_get_device_count(); ++d) {
        if (ggml_backend_cuda_rdna4_rows_device(d)) devs.push_back(d);
    }
    if (devs.size() < 2) SKIP("fewer than two GPUs that run the RDNA4 Q8_0 matvec");
    const GraphsOff graphs;
    const Profile profile(true);
    const Invariant invariant(true);
    const int64_t K = 2560, V = 20000, tail = 5003;   // an odd split: a row block across the cut
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    std::vector<float> wf((size_t) (K * V));
    for (float & v : wf) v = nd(rng);
    std::vector<uint8_t> wq(ggml_row_size(GGML_TYPE_Q8_0, K) * (size_t) V);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, wf.data(), wq.data(), 0, V, K, nullptr);
    const size_t head_bytes = ggml_row_size(GGML_TYPE_Q8_0, K) * (size_t) (V - tail);
    for (int T : {1, 3, 4, 8}) {
        std::vector<float> xv((size_t) (K * T));
        for (float & v : xv) v = 20.0f * nd(rng);
        std::vector<uint8_t> ref;
        for (int dev : devs) {
            ggml_backend_ptr be(ggml_backend_cuda_init(dev));
            REQUIRE(be != nullptr);
            Pattern p;
            ggml_tensor * w = p.input(GGML_TYPE_Q8_0, K, V);
            ggml_tensor * w_tail = p.input(GGML_TYPE_Q8_0, K, tail);
            ggml_tensor * x = p.input(GGML_TYPE_F32, K, T);
            ggml_tensor * whole = ggml_mul_mat(p.ctx, w, x);
            ggml_tensor * head = ggml_mul_mat(p.ctx, ggml_view_2d(p.ctx, w, K, V - tail, w->nb[1], 0), x);
            ggml_tensor * rest = ggml_mul_mat(p.ctx, w_tail, x);
            for (ggml_tensor * t : {whole, head, rest}) ggml_backend_cuda_mul_mat_rdna4_rows(t);
            ggml_tensor * joined = ggml_concat(p.ctx, head, rest, 0);
            ggml_tensor * plain = ggml_mul_mat(p.ctx, w, x);
            for (ggml_tensor * t : {whole, joined, plain}) p.output(t);
            REQUIRE(p.alloc(be.get()));
            p.set(w, wq.data());
            ggml_backend_tensor_set(w_tail, wq.data() + head_bytes, 0, ggml_nbytes(w_tail));
            p.set(x, xv.data());
            REQUIRE(ggml_backend_graph_compute(be.get(), p.gf) == GGML_STATUS_SUCCESS);
            const std::vector<uint8_t> got_whole = read(whole), got_joined = read(joined), got_plain = read(plain);
            if (ref.empty()) ref = got_whole;
            auto differ = [&](const std::vector<uint8_t> & v) {
                size_t n = 0;
                for (size_t i = 0; i < v.size(); i += sizeof(float)) n += std::memcmp(&v[i], &ref[i], sizeof(float)) != 0;
                return n;
            };
            const size_t d_whole = differ(got_whole), d_joined = differ(got_joined), d_plain = differ(got_plain);
            std::printf("[lm-head-split] GPU %d T=%d: flagged %zu, split %zu, unflagged %zu of %lld outputs differ\n",
                        dev, T, d_whole, d_joined, d_plain, (long long) (V * T));
            if (d_whole || d_joined) {
                fail("lm head row split: GPU " + std::to_string(dev) + " T=" + std::to_string(T) + " differs");
            }
        }
    }
}

TEST_CASE(QnSplitFusionsFixture, argmax_ties_gpu) {
    each_gpu([&](ggml_backend_t be) { for (int V : {248320, 151936, 100003, 64000, 8192, 8191, 5000}) argmax_ties(be, V); });
}

TEST_CASE(QnSplitFusionsFixture, argmax_ties_cpu) {
    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    REQUIRE(cpu != nullptr);
    for (int V : {151936, 5000}) argmax_ties(cpu.get(), V, false);
}

// The verify rollback's copy batch on the side stream: the next use of the compute stream reads the copies (or, with
// side overlap on, synchronize waits for them). An idle hold neither delays synchronize (it waits for the work queued
// before the hold) nor the next use, which releases it.
TEST_CASE(QnSplitFusionsFixture, side_copies_and_hold) {
    each_gpu([&](ggml_backend_t be) {
        ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true}));
        ggml_cgraph * gf = ggml_new_graph(ctx.get());
        std::vector<ggml_tensor *> src, dst, out;
        for (int i = 0; i < 4; ++i) {
            const int64_t n = (2 << 20) + 7 * i;   // ~8 MiB each, one not a multiple of 16 bytes
            src.push_back(ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n));
            dst.push_back(ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n));
            out.push_back(ggml_scale(ctx.get(), dst.back(), 1.0f));
            ggml_build_forward_expand(gf, out.back());
        }
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), be));
        REQUIRE(buf != nullptr);
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::vector<ggml_cuda_copy_desc> descs;
        for (int i = 0; i < 4; ++i) {
            std::vector<float> v((size_t) ggml_nelements(src[i]));
            for (float & x : v) x = u(rng);
            ggml_backend_tensor_set(src[i], v.data(), 0, ggml_nbytes(src[i]));
            descs.push_back({src[i]->data, dst[i]->data, ggml_nbytes(dst[i])});
        }
        auto zero_dst = [&] {
            for (ggml_tensor * t : dst) {
                const std::vector<uint8_t> z(ggml_nbytes(t), 0);
                ggml_backend_tensor_set(t, z.data(), 0, z.size());
            }
        };
        auto expect_copied = [&](const std::string & what, const std::vector<ggml_tensor *> & got) {
            for (int i = 0; i < 4; ++i) {
                if (read(got[i]) != read(src[i])) fail(what + ": tensor " + std::to_string(i) + " differs");
            }
        };

        zero_dst();
        ggml_backend_cuda_copy_batch_side(be, descs.data(), (int) descs.size());
        REQUIRE(ggml_backend_graph_compute_async(be, gf) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(be);
        expect_copied("joined at the next use", out);

        zero_dst();
        ggml_backend_cuda_set_side_overlap(be, true);
        ggml_backend_cuda_copy_batch_side(be, descs.data(), (int) descs.size());
        ggml_backend_synchronize(be);
        ggml_backend_cuda_set_side_overlap(be, false);
        expect_copied("joined by synchronize", dst);

        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();
        ggml_backend_cuda_hold(be, 1000000);   // 1 s unless released
        ggml_backend_synchronize(be);
        const auto t1 = clock::now();
        REQUIRE(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS);
        const auto t2 = clock::now();
        const double sync_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double next_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::printf("hold: synchronize %.2f ms, next use %.2f ms\n", sync_ms, next_ms);
        if (sync_ms > 300.0) fail("synchronize waited for the hold");
        if (next_ms > 300.0) fail("the next use did not release the hold");
        expect_copied("after a hold", out);
    });
}

// The split decode hop, both ways between two GPUs: each step a graph on the producer updates its tensors (so their
// lines sit dirty in its L2), a peer copy batch sends them to the consumer, and a graph there adds them into its sums,
// with no host wait anywhere, once with a 2-byte sized f16 tensor in the batch. A read of stale data on either side
// changes the sums (a discrete consumer whose event skipped the system-scope acquire summed stale lines of its L2).
TEST_CASE(QnSplitFusionsFixture, peer_copy_hops) {
    if (ggml_backend_cuda_get_device_count() < 2) SKIP("needs two GPUs");
    const Profile prof(true);   // batched kernel copies, as the split forwards queue them
    const GraphsOff graphs;
    constexpr int steps = 200;
    const int64_t sizes[3] = {7680, 40, 3};   // a hidden row of 3 tokens, the expert ids, a 12-byte tail
    for (int from = 0; from < 2; ++from) {
        for (int with_f16 = 0; with_f16 < 2; ++with_f16) {
            ggml_backend_ptr bp(ggml_backend_cuda_init(from)), bc(ggml_backend_cuda_init(1 - from));
            REQUIRE(bp != nullptr && bc != nullptr);
            const std::string name = "GPU " + std::to_string(from) + " -> " + std::to_string(1 - from) +
                                     (with_f16 ? ", with an f16 tensor" : "");
            ggml_context_ptr cp(ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true}));
            ggml_context_ptr cc(ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true}));
            ggml_cgraph * gp = ggml_new_graph(cp.get());
            ggml_cgraph * gc = ggml_new_graph(cc.get());
            std::vector<ggml_tensor *> base, src, dst, acc;
            const int n = with_f16 ? 4 : 3;
            for (int i = 0; i < n; ++i) {
                const ggml_type type = i < 3 ? GGML_TYPE_F32 : GGML_TYPE_F16;
                const int64_t ne = i < 3 ? sizes[i] : 1001;
                base.push_back(ggml_new_tensor_1d(cp.get(), type, ne));
                src.push_back(ggml_new_tensor_1d(cp.get(), type, ne));
                ggml_build_forward_expand(gp, ggml_add_inplace(cp.get(), src.back(), base.back()));
                dst.push_back(ggml_new_tensor_1d(cc.get(), type, ne));
                acc.push_back(ggml_new_tensor_1d(cc.get(), GGML_TYPE_F32, ne));
                ggml_tensor * in = type == GGML_TYPE_F32 ? dst.back() : ggml_cast(cc.get(), dst.back(), GGML_TYPE_F32);
                ggml_build_forward_expand(gc, ggml_add_inplace(cc.get(), acc.back(), in));
            }
            ggml_backend_buffer_ptr bufp(ggml_backend_alloc_ctx_tensors(cp.get(), bp.get()));
            ggml_backend_buffer_ptr bufc(ggml_backend_alloc_ctx_tensors(cc.get(), bc.get()));
            REQUIRE(bufp != nullptr && bufc != nullptr);
            ggml_backend_buffer_clear(bufp.get(), 0);
            ggml_backend_buffer_clear(bufc.get(), 0);
            // Small integers: every sum below is exact in f32 (and every value in f16).
            std::vector<std::vector<float>> b(n);
            for (int i = 0; i < n; ++i) {
                b[i].resize((size_t) ggml_nelements(base[i]));
                for (size_t j = 0; j < b[i].size(); ++j) b[i][j] = (float) (1 + (j * 7 + i) % 3);
                if (base[i]->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_set(base[i], b[i].data(), 0, ggml_nbytes(base[i]));
                } else {
                    std::vector<ggml_fp16_t> h(b[i].size());
                    for (size_t j = 0; j < h.size(); ++j) h[j] = ggml_fp32_to_fp16(b[i][j]);
                    ggml_backend_tensor_set(base[i], h.data(), 0, ggml_nbytes(base[i]));
                }
            }
            ggml_backend_event_ptr consumed(ggml_backend_event_new(ggml_backend_get_device(bc.get())));
            REQUIRE(consumed != nullptr);
            for (int step = 0; step < steps; ++step) {
                REQUIRE(ggml_backend_graph_compute_async(bp.get(), gp) == GGML_STATUS_SUCCESS);
                // The copy overwrites what the consumer's last step read.
                if (step > 0) ggml_backend_event_wait(bp.get(), consumed.get());
                for (int i = 0; i < n; ++i) ggml_backend_tensor_copy_async(bp.get(), bc.get(), src[i], dst[i]);
                REQUIRE(ggml_backend_graph_compute_async(bc.get(), gc) == GGML_STATUS_SUCCESS);
                ggml_backend_event_record(consumed.get(), bc.get());
            }
            ggml_backend_synchronize(bp.get());
            ggml_backend_synchronize(bc.get());
            // After step s the producer holds (s + 1) * base; the consumer sums them.
            const float tri = (float) steps * (steps + 1) / 2;
            for (int i = 0; i < n; ++i) {
                std::vector<float> got((size_t) ggml_nelements(acc[i]));
                ggml_backend_tensor_get(acc[i], got.data(), 0, ggml_nbytes(acc[i]));
                size_t bad = 0;
                for (size_t j = 0; j < got.size(); ++j) bad += got[j] != tri * b[i][j];
                if (bad) {
                    std::vector<float> d((size_t) ggml_nelements(acc[i]));
                    if (dst[i]->type == GGML_TYPE_F32) ggml_backend_tensor_get(dst[i], d.data(), 0, ggml_nbytes(dst[i]));
                    std::printf("  tensor %d: sum[0] %.1f (want %.1f, base %.0f), dst[0] %.1f\n", i, got[0], tri * b[i][0],
                                b[i][0], d[0]);
                    fail(name + ": tensor " + std::to_string(i) + ": " + std::to_string(bad) + " stale sums");
                }
            }
            std::printf("%s: %d hops summed\n", name.c_str(), steps);
        }
    }
}
