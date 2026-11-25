// The batched verify's full attention (build_full_attn_verify) against plain T=1 decode, bit for bit, with slots
// whose rows are dense, QSA, or cross the dense/QSA boundary inside the step. One synthetic full-attention layer at
// Qwen3.8-Flash-Next's attention shape (head 256, 12 query heads per KV head, indexer 4 x 128, block ratio 4, 2048-key
// budget) with random Q8_0 projections: no model file. Every slot's rows run once in the batched graph and once as
// consecutive T=1 decode steps (build_full_attn as the stable decode graph builds it) on a twin cache; the attention
// outputs and the caches (K/V, raw indexer keys, pooled blocks) must be identical. One-row slots run the multi-slot
// decode step's attention (build_full_attn_slots), which reads its QSA slots through the same inputs.
// The graph builders are internal to qwen4exp_graph.cpp, so this file compiles that source into the test.
// Exit 77 when no CUDA/HIP device is available or the device has no QSA kernels.
#include "../../src/qwen4exp/qwen4exp_graph.cpp"

#include <random>

using namespace luce::common;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

namespace {

constexpr int kEmbd = 256, kCtx = 4096;

struct Case {
    int rows;                     // per slot
    std::vector<int32_t> pos0;    // each slot's first position
    int qsa_rows;                 // rows past the QSA boundary, all slots
};

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
    if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(v.data(), (ggml_fp16_t *) q.data(), (int64_t) v.size());
    } else {
        ggml_quantize_chunk(t->type, v.data(), q.data(), 0, ggml_nelements(t) / t->ne[0], t->ne[0], nullptr);
    }
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

// The first n bytes of both tensors are equal (n = 0: every byte).
bool same_bytes(ggml_tensor * a, ggml_tensor * b, size_t n = 0) {
    if (n == 0) n = ggml_nbytes(a);
    std::vector<uint8_t> x(n), y(n);
    ggml_backend_tensor_get(a, x.data(), 0, n);
    ggml_backend_tensor_get(b, y.data(), 0, n);
    return x == y;
}

// The layer's weights: Q8_0 projections (the matvec whose columns are batch-invariant) and F32 norms.
struct Model {
    Qwen4ExpWeights w;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    Model(ggml_backend_t be, std::mt19937 & rng) {
        w.gfx1151 = ggml_backend_cuda_qwen4exp_supported(be);
        w.qsa = ggml_backend_cuda_qsa_supported(be);
        w.n_layer = 1;
        w.n_embd = kEmbd;
        w.compress_ratios = {4};
        w.layers.resize(1);
        Qwen4ExpLayer & L = w.layers[0];
        L.is_full_attention = true;
        const int64_t D = w.n_embd_head_k, idim = w.indexer_head_size;
        ctx = ggml_init({16 * ggml_tensor_overhead(), nullptr, true});
        auto proj = [&](int64_t out) { return ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, kEmbd, out); };
        auto norm = [&](int64_t n) { return ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n); };
        L.wq = proj(2 * D * w.n_head);
        L.wk = proj(D * w.n_head_kv);
        L.wv = proj(D * w.n_head_kv);
        L.wo = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, D * w.n_head, kEmbd);
        L.indexer_q_proj = proj(idim * w.indexer_n_head);
        L.indexer_k_proj = proj(idim);
        L.q_norm = norm(D);
        L.k_norm = norm(D);
        L.indexer_q_norm = norm(idim);
        L.indexer_k_norm = norm(idim);
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        CHECK(buf);
        for (ggml_tensor * t : {L.wq, L.wk, L.wv, L.wo, L.indexer_q_proj, L.indexer_k_proj}) fill(t, rng, 0.1f);
        for (ggml_tensor * t : {L.q_norm, L.k_norm, L.indexer_q_norm, L.indexer_k_norm}) fill(t, rng, 0.2f, 1.0f);
    }
    ~Model() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

// One sequence's full-attention rows, with every row random: the rows at and past a slot's position are the stale
// suffix a rejected verify leaves, which masks and block visibility must hide.
struct Cache {
    Qwen4ExpCache c;

    Cache(ggml_backend_t be, const Qwen4ExpWeights & w, int rows, uint32_t seed) {
        c.ctx = ggml_init({8 * ggml_tensor_overhead(), nullptr, true});
        c.max_ctx = kCtx;
        c.full_layer_ids = {0};
        c.attn_k = {ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, w.n_embd_head_k, kCtx, w.n_head_kv)};
        c.attn_v = {ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, w.n_embd_head_k, kCtx, w.n_head_kv)};
        c.indexer_raw = {ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, w.indexer_head_size, kCtx)};
        c.indexer_k = {ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, w.indexer_head_size, (kCtx + 3) / 4 + 1)};
        c.buf = ggml_backend_alloc_ctx_tensors(c.ctx, be);
        CHECK(c.buf);
        std::mt19937 rng(seed);
        for (ggml_tensor * t : {c.attn_k[0], c.attn_v[0], c.indexer_raw[0], c.indexer_k[0]}) fill(t, rng, 1.0f);
        // What qwen4exp_verify_supported asks of a cache; the attention layer reads neither.
        c.mtp_k = c.spec_slot0 = c.attn_k[0];
        c.mtp_draft = std::max(1, rows - 1);
    }
    ~Cache() {
        ggml_backend_buffer_free(c.buf);
        ggml_free(c.ctx);
    }
    void start(int pos) { c.cur_pos = pos; c.indexer_blocks = pos / 4; }
};

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

// One T=1 decode step's attention at `pos`, as forward_impl's stable decode graph builds and feeds it: dense over the
// step's span behind its causal mask, or QSA through the graph's runtime inputs.
std::vector<float> decode_row(ggml_backend_t be, const Qwen4ExpWeights & w, Qwen4ExpCache & cache, const float * cur,
                              int pos) {
    const bool qsa = qsa_mode(w, cache, 1, pos, w.qsa) == QSA_DECODE;
    int64_t base = 0;
    const int64_t bucket = qsa ? qwen4exp_kv_bucket(cache.max_ctx, pos + 1, true)
                               : qwen4exp_stable_kv_span(base, qwen4exp_attn_cap(cache), pos + 1);
    Graph g(be);
    ggml_tensor * in = g.input(GGML_TYPE_F32, w.n_embd);
    ggml_tensor * positions = g.input(GGML_TYPE_I32, 4);
    ggml_tensor * kv_row = g.input(GGML_TYPE_I32, 1);
    ggml_tensor * mask = qsa ? nullptr : g.input(GGML_TYPE_F16, bucket);
    Qwen4ExpDecodeWorkspace ws;
    if (qsa) {
        ws.kv_bucket = bucket;
        ws.qsa_blocks = bucket / 4;
        ws.qsa_visibility = g.input(GGML_TYPE_F32, ws.qsa_blocks);
        ws.qsa_params = g.input(GGML_TYPE_I32, 10);
    }
    ggml_tensor * out = build_full_attn(g.ctx, g.gf, in, w.layers[0], w, cache.attn_k[0], cache.attn_v[0],
        cache.indexer_k[0], cache.indexer_raw[0], positions, mask, kv_row, bucket, pos, 4, cache.indexer_blocks,
        qsa ? QSA_DECODE : QSA_DENSE, nullptr, qsa ? &ws : nullptr);
    g.allocate(out);

    const int32_t p[4] = {pos, pos, pos, 0}, row = qwen4exp_kv_row(cache, pos);
    ggml_backend_tensor_set(in, cur, 0, ggml_nbytes(in));
    ggml_backend_tensor_set(positions, p, 0, sizeof p);
    ggml_backend_tensor_set(kv_row, &row, 0, sizeof row);
    if (mask) {
        std::vector<ggml_fp16_t> m((size_t) bucket);
        qwen4exp_causal_mask(cache, m.data(), bucket, pos);
        ggml_backend_tensor_set(mask, m.data(), 0, ggml_nbytes(mask));
    }
    Qwen4ExpStableInputs stable;
    stable.backend = be;
    stable.qsa = qsa;
    stable.pos0 = pos;
    stable.T = 1;
    stable.kv_len = pos + 1;
    stable.max_ctx = cache.max_ctx;
    upload_stable_qsa(stable, ws);
    CHECK(ggml_backend_graph_compute(be, g.gf) == GGML_STATUS_SUCCESS);
    if (qsa) cache.indexer_blocks = (pos + 1) / 4;
    std::vector<float> y((size_t) w.n_embd);
    ggml_backend_tensor_get(out, y.data(), 0, ggml_nbytes(out));
    return y;
}

// Every slot's rows in one batched verify group, built and fed as qwen4exp_verify_batched does (R = 1: as one
// multi-slot decode step, qwen4exp_forward_batched). `copied`: the QSA rows read copies of the QSA inputs, as a
// batched verify's layers on a second device do.
std::vector<float> verify_rows(ggml_backend_t be, const Qwen4ExpWeights & w, Qwen4ExpCache * const * caches,
                               const int32_t * pos0, int n_slots, int R, const std::vector<float> & cur, bool copied,
                               int & qsa_rows) {
    const int T = n_slots * R;
    const std::vector<int64_t> spans = verify_row_spans(w, caches, pos0, n_slots, R);
    int64_t span = 0;
    qsa_rows = 0;
    for (const int64_t s : spans) {
        span = std::max(span, s);
        qsa_rows += s < 0;
    }
    Graph g(be);
    Qwen4ExpBatchedDecodeWorkspace ws;
    build_slot_qsa(g.ctx, spans, R, ws);
    std::vector<Qwen4ExpDecodeWorkspace> copies;
    if (copied && ws.qsa_params) {
        copies = slot_qsa_views(g.ctx, spans, R, ggml_cont(g.ctx, ws.qsa_params), ggml_cont(g.ctx, ws.qsa_visibility),
                                ggml_cont(g.ctx, ws.qsa_positions));
    }
    ggml_tensor * in = g.input(GGML_TYPE_F32, w.n_embd, T);
    ggml_tensor * positions = g.input(GGML_TYPE_I32, 4 * T);
    ggml_tensor * mask = span > 0 ? g.input(GGML_TYPE_F16, span, T) : nullptr;
    std::vector<ggml_tensor *> kv_rows;
    for (int s = 0; s < n_slots; ++s) kv_rows.push_back(g.input(GGML_TYPE_I32, R));
    const Qwen4ExpLayer & L = w.layers[0];
    const Qwen4ExpDecodeWorkspace * qsa = copies.empty() ? ws.slot_qsa.data() : copies.data();
    ggml_tensor * qfull = mm(g.ctx, L.wq, in), * kraw = mm(g.ctx, L.wk, in), * vraw = mm(g.ctx, L.wv, in);
    ggml_tensor * attn;
    if (R == 1) {
        std::vector<ggml_tensor *> masks;   // each dense slot's column of `mask`
        for (int s = 0; s < n_slots; ++s) {
            masks.push_back(spans[(size_t) s] < 0 ? nullptr
                : ggml_view_2d(g.ctx, mask, spans[(size_t) s], 1, mask->nb[1], (size_t) s * mask->nb[1]));
        }
        attn = build_full_attn_slots(g.ctx, g.gf, qfull, kraw, vraw, in, positions, masks.data(), kv_rows.data(),
                                     spans.data(), L, w, caches, 0, w.qsa, qsa);
    } else {
        attn = build_full_attn_verify(g.ctx, g.gf, qfull, kraw, vraw, in, positions, mask, kv_rows.data(), spans.data(),
                                      L, w, caches, n_slots, R, 0, w.qsa, qsa);
    }
    ggml_tensor * out = mm(g.ctx, L.wo, attn);
    g.allocate(out);

    ggml_backend_tensor_set(in, cur.data(), 0, ggml_nbytes(in));
    std::vector<int32_t> p((size_t) 4 * T, 0);
    for (int r = 0; r < T; ++r)
        for (int section = 0; section < 3; ++section) p[(size_t) section * T + r] = pos0[r / R] + r % R;
    ggml_backend_tensor_set(positions, p.data(), 0, ggml_nbytes(positions));
    for (int s = 0; s < n_slots; ++s) {
        std::vector<int32_t> rows((size_t) R);
        for (int j = 0; j < R; ++j) rows[(size_t) j] = qwen4exp_kv_row(*caches[s], pos0[s] + j);
        ggml_backend_tensor_set(kv_rows[(size_t) s], rows.data(), 0, rows.size() * sizeof(int32_t));
    }
    if (mask) {
        std::vector<ggml_fp16_t> m((size_t) (span * T), ggml_fp32_to_fp16(-INFINITY));
        for (int r = 0; r < T; ++r) {
            if (spans[(size_t) r] < 0) continue;
            qwen4exp_causal_mask(*caches[r / R], m.data() + (size_t) r * span, spans[(size_t) r], pos0[r / R] + r % R);
        }
        ggml_backend_tensor_set(mask, m.data(), 0, ggml_nbytes(mask));
    }
    Qwen4ExpStableInputs qsa_inputs;
    upload_slot_qsa(be, false, caches, pos0, spans, R, ws, qsa_inputs);
    {
        const ScopedCudaGraphOverrides invariant(false, 0, false, 0, /*mmvq_batch_invariant=*/true);
        CHECK(ggml_backend_graph_compute(be, g.gf) == GGML_STATUS_SUCCESS);
    }
    std::vector<float> y((size_t) w.n_embd * T);
    ggml_backend_tensor_get(out, y.data(), 0, ggml_nbytes(out));
    return y;
}

bool run(ggml_backend_t be, const Qwen4ExpWeights & w, const Case & cs, bool copied, std::mt19937 & rng) {
    const int n = (int) cs.pos0.size(), R = cs.rows;
    std::vector<std::unique_ptr<Cache>> batched, solo;
    std::vector<Qwen4ExpCache *> caches;
    for (int s = 0; s < n; ++s) {
        const uint32_t seed = rng();
        for (auto * side : {&batched, &solo}) {
            side->push_back(std::make_unique<Cache>(be, w, R, seed));
            side->back()->start(cs.pos0[(size_t) s]);
        }
        caches.push_back(&batched.back()->c);
    }
    CHECK(R == 1 || qwen4exp_verify_batched_supported(be, w, caches.data(), cs.pos0.data(), n, R));
    const std::vector<float> cur = random_floats((size_t) w.n_embd * n * R, rng, 1.0f);
    int qsa_rows = 0;
    const std::vector<float> got = verify_rows(be, w, caches.data(), cs.pos0.data(), n, R, cur, copied, qsa_rows);
    CHECK(qsa_rows == cs.qsa_rows);

    bool ok = true;
    for (int s = 0; s < n; ++s) {
        Qwen4ExpCache & ref = solo[(size_t) s]->c;
        const int pos0 = cs.pos0[(size_t) s];
        for (int j = 0; j < R; ++j) {
            const size_t r = (size_t) s * R + j;
            const std::vector<float> want = decode_row(be, w, ref, cur.data() + r * w.n_embd, pos0 + j);
            if (std::memcmp(want.data(), got.data() + r * w.n_embd, want.size() * sizeof(float)) != 0) {
                std::printf("batched-verify-qsa: slot %d row %d (pos %d) differs from T=1 decode\n", s, j, pos0 + j);
                ok = false;
            }
        }
        // The cache rows a decode leaves: every K/V row and raw key, and the blocks pooled through the slot's last row
        // (the pooled scratch row past them is free to differ).
        const Qwen4ExpCache & c = *caches[(size_t) s];
        const size_t pooled = (size_t) ((pos0 + R) / 4) * c.indexer_k[0]->nb[1];
        if (!same_bytes(c.attn_k[0], ref.attn_k[0]) || !same_bytes(c.attn_v[0], ref.attn_v[0]) ||
            !same_bytes(c.indexer_raw[0], ref.indexer_raw[0]) ||
            (pooled > 0 && !same_bytes(c.indexer_k[0], ref.indexer_k[0], pooled))) {
            std::printf("batched-verify-qsa: slot %d (pos %d) leaves other cache rows than T=1 decode\n", s, pos0);
            ok = false;
        }
    }
    std::printf("batched-verify-qsa rows=%d slots=%d qsa_rows=%-2d copied=%d %s\n", R, n, qsa_rows, (int) copied,
                ok ? "OK" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) return 77;
    bool ok = true;
    int cases = 0;
    {
        std::mt19937 rng(20261110);
        const Model model(be, rng);
        const Qwen4ExpWeights & w = model.w;
        const Cache probe(be, w, 2, 0);
        if (!w.qsa || !stable_qsa_supported(be, w, probe.c, kCtx)) {
            std::printf("batched-verify-qsa: device has no stable QSA rows, skipped\n");
            ggml_backend_free(be);
            return 77;
        }
        const Qwen4ExpCudaScope profile(w.gfx1151);
        // The first QSA position is 2051 (513 complete blocks). A QSA slot beside a dense one, with its rows across
        // a 256-key bucket edge; a slot crossing the boundary beside a QSA slot that completes a block mid-step; one
        // dense row then QSA rows beside the first all-QSA step; eight rows completing two blocks; four two-row slots;
        // a decode step of four slots.
        for (const Case & cs : std::vector<Case>{
                {4, {2558, 40}, 4}, {4, {2049, 3001}, 6}, {4, {2050, 2051}, 7}, {8, {2503}, 8},
                {2, {2302, 2303, 100, 2050}, 5}, {1, {2559, 40, 2051, 3003}, 3}}) {
            for (const bool copied : {false, true}) {
                ok &= run(be, w, cs, copied, rng);
                ++cases;
            }
        }
    }
    ggml_backend_free(be);
    std::printf("batched-verify-qsa: %d cases, %s\n", cases, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
