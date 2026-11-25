// LUCE_Q4_INDEXER_RECALL on a synthetic full-attention layer (no model file): under a KVFlash-style pool (K/V rows
// are slots, not positions) the raw indexer keys are written by position, the last forward's indexer queries are
// captured, and qwen4exp_recall_picks pools the context's blocks and returns each query's top-k block ids.
// Compiles qwen4exp_graph.cpp into the test for its builders.
#include "../../src/qwen4exp/qwen4exp_graph.cpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <set>

using namespace luce::common;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

namespace {

constexpr int kEmbd = 256, kCtx = 4096, kPool = 1024, kT = 64, kPos0 = 3000, kTopK = 512;

std::vector<float> random_floats(size_t n, std::mt19937 & rng, float scale, float bias = 0.0f) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (float & x : v) x = bias + scale * u(rng);
    return v;
}

void fill(ggml_tensor * t, std::mt19937 & rng, float scale, float bias = 0.0f) {
    const std::vector<float> v = random_floats((size_t) ggml_nelements(t), rng, scale, bias);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        return;
    }
    std::vector<uint8_t> q(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F16) ggml_fp32_to_fp16_row(v.data(), (ggml_fp16_t *) q.data(), (int64_t) v.size());
    else ggml_quantize_chunk(t->type, v.data(), q.data(), 0, ggml_nelements(t) / t->ne[0], t->ne[0], nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

std::vector<float> get(ggml_tensor * t, size_t off_floats = 0, size_t n = 0) {
    if (!n) n = (size_t) ggml_nelements(t) - off_floats;
    std::vector<float> v(n);
    ggml_backend_tensor_get(t, v.data(), off_floats * sizeof(float), n * sizeof(float));
    return v;
}

struct Graph {
    ggml_backend_t be;
    ggml_context * ctx;
    ggml_cgraph * gf;
    ggml_gallocr_t alloc = nullptr;
    explicit Graph(ggml_backend_t be) : be(be) {
        ctx = ggml_init({8192 * ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false), nullptr, true});
        gf = ggml_new_graph_custom(ctx, 4096, false);
    }
    ~Graph() {
        ggml_backend_cuda_graph_invalidate_range(be, ggml_get_mem_buffer(ctx), ggml_get_mem_size(ctx));
        if (alloc) ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }
    ggml_tensor * input(ggml_type type, int64_t n0, int64_t n1 = 1) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, type, n0, n1);
        ggml_set_input(t);
        return t;
    }
    void allocate(ggml_tensor * out) {
        ggml_set_output(out);
        ggml_build_forward_expand(gf, out);
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        CHECK(alloc && ggml_gallocr_alloc_graph(alloc, gf));
    }
};

}  // namespace

int main() {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) return 77;
    std::mt19937 rng(20261117);
    Qwen4ExpWeights w;
    w.n_layer = 1;
    w.n_embd = kEmbd;
    w.indexer_top_k = kTopK;
    w.compress_ratios = {4};
    w.layers.resize(1);
    Qwen4ExpLayer & L = w.layers[0];
    L.is_full_attention = true;
    const int64_t D = w.n_embd_head_k, idim = w.indexer_head_size, nih = w.indexer_n_head;
    ggml_context * wctx = ggml_init({16 * ggml_tensor_overhead(), nullptr, true});
    auto proj = [&](int64_t out) { return ggml_new_tensor_2d(wctx, GGML_TYPE_Q8_0, kEmbd, out); };
    auto norm = [&](int64_t n) { return ggml_new_tensor_1d(wctx, GGML_TYPE_F32, n); };
    L.wq = proj(2 * D * w.n_head); L.wk = proj(D * w.n_head_kv); L.wv = proj(D * w.n_head_kv);
    L.wo = ggml_new_tensor_2d(wctx, GGML_TYPE_Q8_0, D * w.n_head, kEmbd);
    L.indexer_q_proj = proj(idim * nih); L.indexer_k_proj = proj(idim);
    L.q_norm = norm(D); L.k_norm = norm(D); L.indexer_q_norm = norm(idim); L.indexer_k_norm = norm(idim);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
    CHECK(wbuf);
    for (ggml_tensor * t : {L.wq, L.wk, L.wv, L.wo, L.indexer_q_proj, L.indexer_k_proj}) fill(t, rng, 0.1f);
    for (ggml_tensor * t : {L.q_norm, L.k_norm, L.indexer_q_norm, L.indexer_k_norm}) fill(t, rng, 0.2f, 1.0f);

    // A pooled cache: kPool K/V slots over a kCtx-token context; the indexer rows are position-indexed.
    Qwen4ExpCache cache;
    cache.ctx = ggml_init({16 * ggml_tensor_overhead(), nullptr, true});
    cache.max_ctx = kCtx;
    cache.full_layer_ids = {0};
    cache.attn_k = {ggml_new_tensor_3d(cache.ctx, GGML_TYPE_F16, D, kPool, w.n_head_kv)};
    cache.attn_v = {ggml_new_tensor_3d(cache.ctx, GGML_TYPE_F16, D, kPool, w.n_head_kv)};
    cache.indexer_raw = {ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, idim, kCtx)};
    cache.indexer_k = {ggml_new_tensor_2d(cache.ctx, GGML_TYPE_F32, idim, kCtx / 4 + 1)};
    cache.recall_q = {ggml_new_tensor_3d(cache.ctx, GGML_TYPE_F32, idim, nih, kT)};
    cache.recall_queries = kT;
    cache.buf = ggml_backend_alloc_ctx_tensors(cache.ctx, be);
    CHECK(cache.buf);
    ggml_backend_buffer_clear(cache.buf, 0);
    fill(cache.indexer_raw[0], rng, 1.0f);   // the context before the last forward
    const std::vector<float> raw_before = get(cache.indexer_raw[0]);

    // The last prompt forward: kT tokens at kPos0.., K/V written to slots (position mod pool), raw keys by position.
    const std::vector<float> x = random_floats((size_t) kEmbd * kT, rng, 1.0f);
    std::vector<float> kraw_want, q_want;
    {
        Graph g(be);
        ggml_tensor * in = g.input(GGML_TYPE_F32, kEmbd, kT), * positions = g.input(GGML_TYPE_I32, 4 * kT);
        ggml_tensor * kv_row = g.input(GGML_TYPE_I32, kT), * mask = g.input(GGML_TYPE_F16, kPool, kT);
        ggml_tensor * out = build_full_attn(g.ctx, g.gf, in, L, w, cache.attn_k[0], cache.attn_v[0], cache.indexer_k[0],
            cache.indexer_raw[0], positions, mask, kv_row, kPool, kPos0, 4, 0, QSA_DENSE, nullptr, nullptr, false, false, 0,
            /*raw_by_pos=*/true, cache.recall_q[0]);
        g.allocate(out);
        std::vector<int32_t> pos(4 * kT, 0), rows(kT);
        for (int i = 0; i < kT; ++i) { pos[i] = pos[kT + i] = pos[2 * kT + i] = kPos0 + i; rows[i] = (kPos0 + i) % kPool; }
        std::vector<ggml_fp16_t> m((size_t) kPool * kT, ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(in, x.data(), 0, ggml_nbytes(in));
        ggml_backend_tensor_set(positions, pos.data(), 0, ggml_nbytes(positions));
        ggml_backend_tensor_set(kv_row, rows.data(), 0, ggml_nbytes(kv_row));
        ggml_backend_tensor_set(mask, m.data(), 0, ggml_nbytes(mask));
        CHECK(ggml_backend_graph_compute(be, g.gf) == GGML_STATUS_SUCCESS);

        // References, built apart: the raw keys of the same input and its post-RoPE indexer queries.
        Graph r(be);
        ggml_tensor * rin = r.input(GGML_TYPE_F32, kEmbd, kT), * rpos = r.input(GGML_TYPE_I32, 4 * kT);
        ggml_tensor * rk = mm(r.ctx, L.indexer_k_proj, rin);
        ggml_tensor * rq = qsa_indexer_queries(r.ctx, rin, L, w, rpos);
        ggml_build_forward_expand(r.gf, rk);
        r.allocate(rq);
        ggml_backend_tensor_set(rin, x.data(), 0, ggml_nbytes(rin));
        ggml_backend_tensor_set(rpos, pos.data(), 0, ggml_nbytes(rpos));
        CHECK(ggml_backend_graph_compute(be, r.gf) == GGML_STATUS_SUCCESS);
        kraw_want = get(rk);
        q_want = get(rq);
    }
    // Raw keys landed on the positions, and nowhere else (the slots they would have gone to hold the old keys).
    const std::vector<float> raw_after = get(cache.indexer_raw[0]);
    for (int p = 0; p < kCtx; ++p) {
        const bool written = p >= kPos0 && p < kPos0 + kT;
        for (int j = 0; j < idim; ++j) {
            const float got = raw_after[(size_t) p * idim + j];
            if (written) CHECK(std::fabs(got - kraw_want[(size_t) (p - kPos0) * idim + j]) < 1e-4f);
            else CHECK(got == raw_before[(size_t) p * idim + j]);
        }
    }
    // The captured queries equal the indexer queries of the same input.
    const std::vector<float> q_got = get(cache.recall_q[0]);
    CHECK(q_got.size() == q_want.size());
    float q_err = 0;
    for (size_t i = 0; i < q_got.size(); ++i) q_err = std::max(q_err, std::fabs(q_got[i] - q_want[i]));
    CHECK(q_err < 1e-4f);

    // Picks.
    cache.cur_pos = kPos0 + kT;
    cache.recall_q_n = kT;
    cache.recall_q_pos0 = kPos0;
    Qwen4ExpRecallPicks picks;
    CHECK(qwen4exp_recall_picks(be, w, cache, picks));
    const int nb = (kPos0 + kT) / 4, budget = kTopK / 4;
    CHECK(picks.layers == 1 && picks.queries == kT && picks.budget == budget && picks.blocks == nb && picks.ratio == 4);
    CHECK(cache.indexer_blocks == nb);
    CHECK(picks.ids.size() == (size_t) kT * budget);
    for (int t = 0; t < kT; ++t) {
        std::set<int32_t> seen;
        for (int i = 0; i < budget; ++i) {
            const int32_t id = picks.ids[(size_t) t * budget + i];
            CHECK(id >= 0 && id < (kPos0 + t + 1) / 4);   // only blocks complete at the query's own position
            seen.insert(id);
        }
        CHECK((int) seen.size() == budget);
    }
    // Block 0 was pooled and stored: mean of raw rows 0..3, RMS-normed, scaled, M-RoPE at position 0 (identity).
    {
        std::vector<float> mean(idim, 0.f), want(idim);
        for (int j = 0; j < idim; ++j) for (int k = 0; k < 4; ++k) mean[j] += raw_after[(size_t) k * idim + j] / 4.f;
        float ss = 0; for (float v : mean) ss += v * v;
        const float inv = 1.0f / std::sqrt(ss / idim + w.rms_eps);
        const std::vector<float> nw = get(L.indexer_k_norm);
        const std::vector<float> k0 = get(cache.indexer_k[0], 0, idim);
        float err = 0;
        for (int j = 0; j < idim; ++j) err = std::max(err, std::fabs(k0[j] - mean[j] * inv * nw[j]));
        CHECK(err < 1e-3f);
    }
    // The picks are the top-k of an independent scoring over the stored pooled keys.
    {
        Graph g(be);
        ggml_tensor * keys = ggml_view_2d(g.ctx, cache.indexer_k[0], idim, nb, cache.indexer_k[0]->nb[1], 0);
        ggml_tensor * qi = ggml_view_3d(g.ctx, cache.recall_q[0], idim, nih, kT, cache.recall_q[0]->nb[1], cache.recall_q[0]->nb[2], 0);
        ggml_tensor * score = ggml_ds4_indexer_score(g.ctx, qi, qsa_head_weights(g.ctx, w, kT), ggml_cast(g.ctx, keys, GGML_TYPE_F16), kPos0, 4);
        ggml_tensor * top = ggml_top_k(g.ctx, score, budget);
        g.allocate(top);
        CHECK(ggml_backend_graph_compute(be, g.gf) == GGML_STATUS_SUCCESS);
        std::vector<int32_t> ids((size_t) kT * budget);
        ggml_backend_tensor_get(top, ids.data(), 0, sizeof(int32_t) * ids.size());
        for (int t = 0; t < kT; ++t) {
            std::set<int32_t> a(ids.begin() + (size_t) t * budget, ids.begin() + (size_t) (t + 1) * budget),
                              b(picks.ids.begin() + (size_t) t * budget, picks.ids.begin() + (size_t) (t + 1) * budget);
            CHECK(a == b);
        }
    }
    std::printf("indexer-recall: raw keys by position, %d queries captured (max err %.1e), %d blocks pooled, picks OK\n", kT, q_err, nb);
    ggml_backend_buffer_free(cache.buf);
    ggml_free(cache.ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(be);
    return 0;
}
