#include "dflash2_head.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace dflash::common {

namespace {

std::atomic<uint64_t> s_selector_generation{1};

// Selector projection graph, built once per (drafter, backend, n_cand, K,
// weight-load generation).
struct SelectorGraph {
    const DraftWeights * dw = nullptr;
    ggml_backend_t backend = nullptr;
    int n_cand = 0;
    int K = 0;
    uint64_t gen = 0;
    std::vector<uint8_t> arena;
    ggml_context *  ctx = nullptr;
    ggml_cgraph *   gf = nullptr;
    ggml_gallocr_t  galloc = nullptr;
    ggml_tensor * inp_hidden = nullptr;
    ggml_tensor * inp_succ = nullptr;
    ggml_tensor * inp_pred = nullptr;
    ggml_tensor * hproj = nullptr;
    ggml_tensor * succ = nullptr;
    ggml_tensor * pred = nullptr;

    // The graph is thread_local, so a worker thread that drafts and then
    // exits must return its ggml context and the gallocr's device buffer.
    // Without this each thread that ever drafted leaks one GPU allocation for
    // the lifetime of the process.
    ~SelectorGraph() {
        if (galloc) { ggml_gallocr_free(galloc); galloc = nullptr; }
        if (ctx)    { ggml_free(ctx); ctx = nullptr; }
    }
};

SelectorGraph & selector_graph() {
    static thread_local SelectorGraph g;
    return g;
}

void selector_graph_free(SelectorGraph & g) {
    if (g.galloc) { ggml_gallocr_free(g.galloc); g.galloc = nullptr; }
    if (g.ctx)    { ggml_free(g.ctx); g.ctx = nullptr; }
    g.gf = nullptr;
    g.dw = nullptr;
    g.n_cand = 0;
    g.K = 0;
}

}  // namespace

void dflash2_selector_graph_invalidate() {
    s_selector_generation.fetch_add(1, std::memory_order_relaxed);
}

bool dflash2_score_candidates(const DraftWeights & dw,
                              ggml_backend_t backend,
                              DFlashTarget & target,
                              const float * local_hidden,
                              int q_len,
                              int32_t last_tok,
                              float temperature,
                              Dflash2TreeScores & out) {
    const DraftSelectorWeights & sel = dw.selector;
    if (!sel.enabled || !sel.hproj || !sel.pred_cb || !sel.succ_cb) return false;
    if (q_len <= 1 || !local_hidden || !backend) return false;
    const int hdim   = dw.n_embd;
    const int rank   = sel.rank;
    const int K      = sel.top_k;
    const int n_cand = q_len - 1;
    if (hdim <= 0 || rank <= 0 || K <= 0) return false;

    // 1. Top-k candidates (log-probs) per block position through the target
    //    lm_head. Position 0 of local_hidden is the seed slot; candidates are
    //    rows 1 .. q_len-1.
    if (!target.project_hidden_to_topk(local_hidden + (size_t)hdim, n_cand, K, temperature,
                                       out.lp, out.ids)) {
        return false;
    }
    if (out.lp.size() != (size_t)n_cand * K || out.ids.size() != (size_t)n_cand * K) return false;

    // 2. One graph on the draft backend: hproj(h) for every candidate position,
    //    successor rows for every candidate, predecessor rows for the seed and
    //    every candidate. Built once per (n_cand, K) and reused across steps.
    const int n_rows_pred = 1 + n_cand * K;
    SelectorGraph & g = selector_graph();
    const uint64_t cur_gen = s_selector_generation.load(std::memory_order_relaxed);
    if (!g.ctx || g.dw != &dw || g.backend != backend || g.n_cand != n_cand ||
        g.K != K || g.gen != cur_gen) {
        selector_graph_free(g);
        const size_t arena_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead() + 4096;
        g.arena.assign(arena_size, 0);
        ggml_init_params ip{};
        ip.mem_size   = g.arena.size();
        ip.mem_buffer = g.arena.data();
        ip.no_alloc   = true;
        g.ctx = ggml_init(ip);
        if (!g.ctx) return false;
        g.gf = ggml_new_graph(g.ctx);
        g.inp_hidden = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, hdim, n_cand);
        g.inp_succ   = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_cand * K);
        g.inp_pred   = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_rows_pred);
        ggml_set_input(g.inp_hidden);
        ggml_set_input(g.inp_succ);
        ggml_set_input(g.inp_pred);
        g.hproj = ggml_mul_mat(g.ctx, sel.hproj, g.inp_hidden);      // [rank, n_cand]
        g.succ  = ggml_get_rows(g.ctx, sel.succ_cb, g.inp_succ);      // [rank, n_cand*K] f32
        g.pred  = ggml_get_rows(g.ctx, sel.pred_cb, g.inp_pred);      // [rank, 1+n_cand*K] f32
        ggml_set_output(g.hproj);
        ggml_set_output(g.succ);
        ggml_set_output(g.pred);
        ggml_build_forward_expand(g.gf, g.hproj);
        ggml_build_forward_expand(g.gf, g.succ);
        ggml_build_forward_expand(g.gf, g.pred);
        g.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!g.galloc || !ggml_gallocr_alloc_graph(g.galloc, g.gf)) {
            std::fprintf(stderr, "dflash2_score_candidates: gallocr_alloc_graph failed\n");
            selector_graph_free(g);
            return false;
        }
        g.dw = &dw; g.backend = backend; g.n_cand = n_cand; g.K = K; g.gen = cur_gen;
    }

    std::vector<int32_t> pred_ids((size_t)n_rows_pred);
    pred_ids[0] = last_tok;
    std::memcpy(pred_ids.data() + 1, out.ids.data(), sizeof(int32_t) * (size_t)n_cand * K);
    ggml_backend_tensor_set(g.inp_hidden, local_hidden + (size_t)hdim, 0, sizeof(float) * (size_t)hdim * n_cand);
    ggml_backend_tensor_set(g.inp_succ, out.ids.data(), 0, sizeof(int32_t) * (size_t)n_cand * K);
    ggml_backend_tensor_set(g.inp_pred, pred_ids.data(), 0, sizeof(int32_t) * (size_t)n_rows_pred);
    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "dflash2_score_candidates: graph_compute failed\n");
        return false;
    }
    out.hproj.resize((size_t)rank * n_cand);
    out.succ.resize((size_t)rank * n_cand * K);
    out.pred.resize((size_t)rank * n_rows_pred);
    ggml_backend_tensor_get_async(backend, g.hproj, out.hproj.data(), 0, sizeof(float) * out.hproj.size());
    ggml_backend_tensor_get_async(backend, g.succ,  out.succ.data(),  0, sizeof(float) * out.succ.size());
    ggml_backend_tensor_get_async(backend, g.pred,  out.pred.data(),  0, sizeof(float) * out.pred.size());
    ggml_backend_synchronize(backend);
    out.n_cand = n_cand; out.K = K; out.rank = rank; out.seed = last_tok;
    return true;
}

bool Dflash2TreeScores::topk(const std::vector<int32_t> & prefix, int next_depth,
                             std::vector<float> & out_lp, std::vector<int32_t> & out_ids) const {
    const int i = next_depth - 1;   // candidate position
    if (i < 0 || i >= n_cand || (int)prefix.size() != i) return false;
    // predecessor row: 0 = seed, 1 + (i-1)*K + j = candidate j of position i-1
    int prev_row = 0;
    if (i > 0) {
        const int32_t parent_tok = prefix.back();
        prev_row = -1;
        for (int j = 0; j < K; ++j) {
            if (ids[(size_t)(i - 1) * K + j] == parent_tok) { prev_row = 1 + (i - 1) * K + j; break; }
        }
        if (prev_row < 0) prev_row = 0;   // unknown parent: fall back to raw log-probs via seed row? no — zero compat
    }
    const float * pr = pred.data() + (size_t)prev_row * rank;
    const float * hp = hproj.data() + (size_t)i * rank;
    std::vector<std::pair<float, int>> scored((size_t)K);
    for (int k = 0; k < K; ++k) {
        const float * sc = succ.data() + ((size_t)i * K + k) * rank;
        float dot = 0.0f;
        for (int r = 0; r < rank; ++r) dot += pr[r] * hp[r] * sc[r];
        scored[(size_t)k] = { lp[(size_t)i * K + k] + dot, k };
    }
    std::sort(scored.begin(), scored.end(),
              [](const std::pair<float,int> & a, const std::pair<float,int> & b) { return a.first > b.first; });
    // The compatibility dot is not on a log-prob scale; renormalize the
    // adjusted scores per position (log-softmax) so the tree builder's
    // cumulative best-first comparison across depths stays meaningful.
    float lse = 0.0f;
    const float mx = scored[0].first;
    for (int k = 0; k < K; ++k) lse += std::exp(scored[(size_t)k].first - mx);
    lse = mx + std::log(lse);
    out_lp.resize((size_t)K);
    out_ids.resize((size_t)K);
    for (int k = 0; k < K; ++k) {
        out_lp[(size_t)k]  = scored[(size_t)k].first - lse;
        out_ids[(size_t)k] = ids[(size_t)i * K + scored[(size_t)k].second];
    }
    return true;
}

bool dflash2_select_chain(const DraftWeights & dw,
                          ggml_backend_t backend,
                          DFlashTarget & target,
                          const float * local_hidden,
                          int q_len,
                          int32_t last_tok,
                          std::vector<int32_t> & draft_tok) {
    Dflash2TreeScores sc;
    if (!dflash2_score_candidates(dw, backend, target, local_hidden, q_len, last_tok,
                                  /*temperature=*/1.0f, sc)) {
        return false;
    }
    // Greedy path over the candidates, conditioned on the previous pick.
    draft_tok.assign((size_t)q_len, last_tok);
    std::vector<int32_t> prefix;
    std::vector<float>   top_lp;
    std::vector<int32_t> top_ids;
    for (int i = 0; i < sc.n_cand; ++i) {
        if (!sc.topk(prefix, i + 1, top_lp, top_ids)) return false;
        draft_tok[(size_t)i + 1] = top_ids[0];
        prefix.push_back(top_ids[0]);
    }
    return true;
}

}  // namespace dflash::common
