// The batched verify's full attention (build_full_attn_verify) against plain T=1 decode, bit for bit, with slots
// whose rows are dense, QSA, or cross the dense/QSA boundary inside the step. One synthetic full-attention layer at
// Qwen3.8-Flash-Next's attention shape (head 256, 12 query heads per KV head, indexer 4 x 128, block ratio 4, 2048-key
// budget) with random Q8_0 projections: no model file. Every slot's rows run once in the batched graph and once as
// consecutive T=1 decode steps (build_full_attn as the stable decode graph builds it) on a twin cache; the attention
// outputs and the caches (K/V, raw indexer keys, pooled blocks) must be identical. The slots take different numbers
// of rows, down to the one row of a slot that drafted nothing, in any order. A step of one-row slots runs both the
// verify's attention and the multi-slot decode step's (build_full_attn_slots), which reads its QSA slots through the
// same inputs.
// The same for the batched verify's linear attention (build_linear_attn_verify) and PLE (build_ple_verify), with one
// synthetic gated delta net layer at the model's shape and one synthetic PLE layer: every row against T=1 decode steps
// (build_linear_attn, build_ple), and the recurrent and conv states each slot keeps after its rollback (a one-row slot
// keeps its row) against the states after that many decode steps. And the groups the slots form (verify_groups).
// The graph builders are internal to qwen4exp_graph.cpp, so this file compiles that source into the test.
// Exit 77 when no CUDA/HIP device is available or the device has no QSA kernels.
#include "../../src/qwen4exp/qwen4exp_graph.cpp"

#include <functional>
#include <numeric>
#include <random>

using namespace luce::common;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

namespace {

constexpr int kEmbd = 256, kCtx = 4096;

struct Case {
    std::vector<int> rows;        // each slot's rows
    std::vector<int32_t> pos0;    // each slot's first position
    int qsa_rows;                 // rows past the QSA boundary, all slots
    bool decode_step = false;     // one-row slots as one multi-slot decode step instead
};

std::string join(const std::vector<int> & v) {
    std::string s;
    for (const int x : v) s += (s.empty() ? "" : ",") + std::to_string(x);
    return s;
}

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

std::vector<uint8_t> bytes(ggml_tensor * t, size_t n = 0) {
    std::vector<uint8_t> x(n ? n : ggml_nbytes(t));
    ggml_backend_tensor_get(t, x.data(), 0, x.size());
    return x;
}

// The first n bytes of both tensors are equal (n = 0: every byte).
bool same_bytes(ggml_tensor * a, ggml_tensor * b, size_t n = 0) {
    return bytes(a, n) == bytes(b, n ? n : ggml_nbytes(a));
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

// Every slot's rows in one batched verify group, built and fed as qwen4exp_verify_batched does (`decode_step`: as one
// multi-slot decode step, qwen4exp_forward_batched). `copied`: the QSA rows read copies of the QSA inputs, as a
// batched verify's layers on a second device do. Adds the group's QSA rows to `qsa_rows`.
std::vector<float> verify_rows(ggml_backend_t be, const Qwen4ExpWeights & w, Qwen4ExpCache * const * caches,
                               const int32_t * pos0, const std::vector<int> & rows, const std::vector<float> & cur,
                               bool decode_step, bool copied, int & qsa_rows) {
    const int n_slots = (int) rows.size();
    const std::vector<int> row0 = verify_row_offsets(rows.data(), n_slots);
    const int T = row0[(size_t) n_slots];
    std::vector<int> row_slot((size_t) T);
    std::vector<int32_t> row_pos((size_t) T);
    for (int s = 0; s < n_slots; ++s) {
        for (int j = 0; j < rows[(size_t) s]; ++j) {
            row_slot[(size_t) (row0[(size_t) s] + j)] = s;
            row_pos[(size_t) (row0[(size_t) s] + j)] = pos0[s] + j;
        }
    }
    const std::vector<int64_t> spans = verify_row_spans(w, caches, pos0, rows.data(), n_slots);
    int64_t span = 0;
    for (const int64_t s : spans) {
        span = std::max(span, s);
        qsa_rows += s < 0;
    }
    Graph g(be);
    Qwen4ExpBatchedDecodeWorkspace ws;
    build_slot_qsa(g.ctx, spans, rows, ws);
    std::vector<Qwen4ExpDecodeWorkspace> copies;
    if (copied && ws.qsa_params) {
        copies = slot_qsa_views(g.ctx, spans, rows, ggml_cont(g.ctx, ws.qsa_params),
                                ggml_cont(g.ctx, ws.qsa_visibility), ggml_cont(g.ctx, ws.qsa_positions));
    }
    ggml_tensor * in = g.input(GGML_TYPE_F32, w.n_embd, T);
    ggml_tensor * positions = g.input(GGML_TYPE_I32, 4 * T);
    ggml_tensor * mask = span > 0 ? g.input(GGML_TYPE_F16, span, T) : nullptr;
    std::vector<ggml_tensor *> kv_rows;
    for (int s = 0; s < n_slots; ++s) kv_rows.push_back(g.input(GGML_TYPE_I32, rows[(size_t) s]));
    const Qwen4ExpLayer & L = w.layers[0];
    const Qwen4ExpDecodeWorkspace * qsa = copies.empty() ? ws.slot_qsa.data() : copies.data();
    ggml_tensor * qfull = mm(g.ctx, L.wq, in), * kraw = mm(g.ctx, L.wk, in), * vraw = mm(g.ctx, L.wv, in);
    ggml_tensor * attn;
    if (decode_step) {
        std::vector<ggml_tensor *> masks;   // each dense slot's column of `mask`
        for (int s = 0; s < n_slots; ++s) {
            masks.push_back(spans[(size_t) s] < 0 ? nullptr
                : ggml_view_2d(g.ctx, mask, spans[(size_t) s], 1, mask->nb[1], (size_t) s * mask->nb[1]));
        }
        attn = build_full_attn_slots(g.ctx, g.gf, qfull, kraw, vraw, in, positions, masks.data(), kv_rows.data(),
                                     spans.data(), L, w, caches, 0, w.qsa, qsa);
    } else {
        attn = build_full_attn_verify(g.ctx, g.gf, qfull, kraw, vraw, in, positions, mask, kv_rows.data(), spans.data(),
                                      L, w, caches, n_slots, rows.data(), 0, w.qsa, qsa);
    }
    ggml_tensor * out = mm(g.ctx, L.wo, attn);
    g.allocate(out);

    ggml_backend_tensor_set(in, cur.data(), 0, ggml_nbytes(in));
    std::vector<int32_t> p((size_t) 4 * T, 0);
    for (int r = 0; r < T; ++r)
        for (int section = 0; section < 3; ++section) p[(size_t) section * T + r] = row_pos[(size_t) r];
    ggml_backend_tensor_set(positions, p.data(), 0, ggml_nbytes(positions));
    for (int s = 0; s < n_slots; ++s) {
        std::vector<int32_t> kv((size_t) rows[(size_t) s]);
        for (size_t j = 0; j < kv.size(); ++j) kv[j] = qwen4exp_kv_row(*caches[s], pos0[s] + (int) j);
        ggml_backend_tensor_set(kv_rows[(size_t) s], kv.data(), 0, kv.size() * sizeof(int32_t));
    }
    if (mask) {
        std::vector<ggml_fp16_t> m((size_t) (span * T), ggml_fp32_to_fp16(-INFINITY));
        for (int r = 0; r < T; ++r) {
            if (spans[(size_t) r] < 0) continue;
            qwen4exp_causal_mask(*caches[row_slot[(size_t) r]], m.data() + (size_t) r * span, spans[(size_t) r],
                                 row_pos[(size_t) r]);
        }
        ggml_backend_tensor_set(mask, m.data(), 0, ggml_nbytes(mask));
    }
    Qwen4ExpStableInputs qsa_inputs;
    upload_slot_qsa(be, false, caches, pos0, spans, rows, ws, qsa_inputs);
    {
        const ScopedCudaGraphOverrides invariant(false, 0, false, 0, /*mmvq_batch_invariant=*/true,
                                                 /*mmvq_invariant_max_ncols=*/verify_group_rows(false));
        CHECK(ggml_backend_graph_compute(be, g.gf) == GGML_STATUS_SUCCESS);
    }
    std::vector<float> y((size_t) w.n_embd * T);
    ggml_backend_tensor_get(out, y.data(), 0, ggml_nbytes(out));
    return y;
}

bool run(ggml_backend_t be, const Qwen4ExpWeights & w, const Case & cs, bool copied, std::mt19937 & rng) {
    const int n = (int) cs.pos0.size();
    const std::vector<int> row0 = verify_row_offsets(cs.rows.data(), n);
    std::vector<std::unique_ptr<Cache>> batched, solo;
    std::vector<Qwen4ExpCache *> caches;
    for (int s = 0; s < n; ++s) {
        const uint32_t seed = rng();
        for (auto * side : {&batched, &solo}) {
            side->push_back(std::make_unique<Cache>(be, w, *std::max_element(cs.rows.begin(), cs.rows.end()), seed));
            side->back()->start(cs.pos0[(size_t) s]);
        }
        caches.push_back(&batched.back()->c);
    }
    CHECK(cs.decode_step || qwen4exp_verify_batched_supported(be, w, caches.data(), cs.pos0.data(), cs.rows.data(), n));
    const std::vector<float> cur = random_floats((size_t) w.n_embd * row0[(size_t) n], rng, 1.0f);
    // Group by group, as the graph forms them.
    const std::vector<int> first = verify_groups(cs.rows.data(), n, false);
    std::vector<float> got;
    int qsa_rows = 0;
    for (size_t g = 0; g + 1 < first.size(); ++g) {
        const int s0 = first[g], s1 = first[g + 1];
        const std::vector<float> y = verify_rows(be, w, caches.data() + s0, cs.pos0.data() + s0,
            std::vector<int>(cs.rows.begin() + s0, cs.rows.begin() + s1),
            std::vector<float>(cur.begin() + (std::ptrdiff_t) row0[(size_t) s0] * w.n_embd,
                               cur.begin() + (std::ptrdiff_t) row0[(size_t) s1] * w.n_embd),
            cs.decode_step, copied, qsa_rows);
        got.insert(got.end(), y.begin(), y.end());
    }
    CHECK(qsa_rows == cs.qsa_rows);

    bool ok = true;
    for (int s = 0; s < n; ++s) {
        Qwen4ExpCache & ref = solo[(size_t) s]->c;
        const int pos0 = cs.pos0[(size_t) s], R = cs.rows[(size_t) s];
        for (int j = 0; j < R; ++j) {
            const size_t r = (size_t) row0[(size_t) s] + j;
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
    std::printf("batched-verify-qsa rows=%s%s qsa_rows=%-2d copied=%d %s\n", join(cs.rows).c_str(),
                cs.decode_step ? " (decode step)" : "", qsa_rows, (int) copied, ok ? "OK" : "FAIL");
    return ok;
}

// One gated delta net layer at the model's shape (128-wide state, 16 key and 48 value heads, kernel 4) with random
// weights: Q8_0 projections, and F32 alpha and beta projections back to back, as the loader finds them. `stacked`
// adds the loader's one view over both, so verify rows take one product for their gates and decode rows two.
struct LinearModel {
    Qwen4ExpWeights w;
    ggml_context * ctx = nullptr, * views = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    LinearModel(ggml_backend_t be, bool stacked, std::mt19937 & rng) {
        w.gfx1151 = ggml_backend_cuda_qwen4exp_supported(be);
        w.n_layer = 1;
        w.n_embd = kEmbd;
        w.layers.resize(1);
        Qwen4ExpLayer & L = w.layers[0];
        const int64_t Hv = w.linear_value_heads, d_in = w.ssm_d_inner;
        const int64_t channels = 2 * (int64_t) w.ssm_n_group * w.ssm_d_state + d_in;
        ctx = ggml_init({16 * ggml_tensor_overhead(), nullptr, true});
        L.attn_qkv = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, kEmbd, channels);
        L.attn_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, kEmbd, d_in);
        L.ssm_out = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, d_in, kEmbd);
        L.ssm_alpha = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kEmbd, Hv);
        L.ssm_beta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kEmbd, Hv);
        L.ssm_conv1d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.ssm_d_conv, channels);
        L.ssm_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, w.ssm_d_state);
        L.ssm_dt_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, Hv);
        L.ssm_a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, Hv);
        L.ssm_gate_ba = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * Hv);
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        CHECK(buf);
        for (ggml_tensor * t : {L.attn_qkv, L.attn_gate, L.ssm_out, L.ssm_alpha, L.ssm_beta}) fill(t, rng, 0.1f);
        fill(L.ssm_conv1d, rng, 0.5f);
        fill(L.ssm_norm, rng, 0.2f, 1.0f);
        // [dt_bias | A] with A < 0, as the model's -exp(A_log).
        std::vector<float> ba = random_floats((size_t) Hv, rng, 1.0f);
        for (const float x : random_floats((size_t) Hv, rng, 0.5f)) ba.push_back(-std::exp(x));
        ggml_backend_tensor_set(L.ssm_dt_bias, ba.data(), 0, ggml_nbytes(L.ssm_dt_bias));
        ggml_backend_tensor_set(L.ssm_a, ba.data() + Hv, 0, ggml_nbytes(L.ssm_a));
        ggml_backend_tensor_set(L.ssm_gate_ba, ba.data(), 0, ggml_nbytes(L.ssm_gate_ba));
        if (!stacked) return;
        CHECK((const char *) L.ssm_beta->data == (const char *) L.ssm_alpha->data + ggml_nbytes(L.ssm_alpha));
        views = ggml_init({ggml_tensor_overhead(), nullptr, true});
        L.ssm_alpha_beta = ggml_new_tensor_2d(views, GGML_TYPE_F32, kEmbd, 2 * Hv);
        CHECK(ggml_backend_tensor_alloc(buf, L.ssm_alpha_beta, L.ssm_alpha->data) == GGML_STATUS_SUCCESS);
    }
    ~LinearModel() {
        ggml_backend_buffer_free(buf);
        if (views) ggml_free(views);
        ggml_free(ctx);
    }
};

// A step's slots: their stacked recurrent states (random, the same for every `seed`) and each slot's cache over its
// slab, with the rollback rows of a verify of up to `rows` rows.
struct LinearSlots {
    Qwen4ExpSlotStates states;
    std::vector<Qwen4ExpCache> caches;

    LinearSlots(ggml_backend_t be, const Qwen4ExpWeights & w, int n_slots, int rows, uint32_t seed)
        : caches((size_t) n_slots) {
        CHECK(create_qwen4exp_slot_states(be, w, n_slots, states));
        ggml_tensor * ssm = states.ssm[0], * conv = states.conv[0];
        std::mt19937 rng(seed);
        fill(ssm, rng, 0.1f);
        fill(conv, rng, 1.0f);
        for (int s = 0; s < n_slots; ++s) {
            Qwen4ExpCache & c = caches[(size_t) s];
            c.ctx = ggml_init({(8 + (size_t) rows) * ggml_tensor_overhead(), nullptr, true});
            c.linear_layer_ids = {0};
            c.ssm_state = {ggml_view_3d(c.ctx, ssm, ssm->ne[0], ssm->ne[1], ssm->ne[2], ssm->nb[1], ssm->nb[2],
                                        (size_t) s * ssm->nb[3])};
            c.conv_state = {ggml_view_2d(c.ctx, conv, conv->ne[0], conv->ne[1], conv->nb[1], (size_t) s * conv->nb[2])};
            c.slot_states = &states;
            c.state_slot = s;
            c.mtp_draft = std::max(1, rows - 1);
            c.spec_slot0 = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I32, 1);
            ggml_tensor * windows = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, conv->ne[0], conv->ne[1], rows);
            c.spec_conv = {windows};
            Qwen4ExpCache::SpecRows each{};
            for (int t = 0; t < rows; ++t) {
                each[(size_t) t] = ggml_view_2d(c.ctx, windows, conv->ne[0], conv->ne[1], windows->nb[1],
                                                (size_t) t * windows->nb[2]);
            }
            c.spec_conv_rows = {each};
            c.buf = ggml_backend_alloc_ctx_tensors(c.ctx, be);
            CHECK(c.buf);
            const int32_t slab = 0;
            ggml_backend_tensor_set(c.spec_slot0, &slab, 0, sizeof slab);
        }
    }
    ~LinearSlots() {
        for (Qwen4ExpCache & c : caches) {
            ggml_backend_buffer_free(c.buf);
            ggml_free(c.ctx);
        }
        free_qwen4exp_slot_states(states);
    }
};

struct StateCase {
    std::vector<int> rows;   // each slot's rows
    std::vector<int> kept;   // the rows each slot keeps (a one-row slot: its one)
};

int max_rows(const StateCase & cs) { return *std::max_element(cs.rows.begin(), cs.rows.end()); }

// One T=1 decode step's linear attention, as a solo decode graph builds it.
std::vector<float> linear_decode_row(ggml_backend_t be, const Qwen4ExpWeights & w, Qwen4ExpCache & cache,
                                     const float * cur) {
    Graph g(be);
    ggml_tensor * in = g.input(GGML_TYPE_F32, w.n_embd);
    ggml_tensor * out = build_linear_attn(g.ctx, g.gf, in, w.layers[0], w, cache.ssm_state[0], cache.conv_state[0]);
    g.allocate(out);
    ggml_backend_tensor_set(in, cur, 0, ggml_nbytes(in));
    CHECK(ggml_backend_graph_compute(be, g.gf) == GGML_STATUS_SUCCESS);
    std::vector<float> y((size_t) w.n_embd);
    ggml_backend_tensor_get(out, y.data(), 0, ggml_nbytes(out));
    return y;
}

// The rows of one batched verify group (slots [s0, s0 + n_slots) of the step) through its linear attention, then each
// slot's rollback to the rows it keeps.
std::vector<float> linear_rows(ggml_backend_t be, const Qwen4ExpWeights & w, LinearSlots & slots, const StateCase & cs,
                               int s0, int n_slots, const std::vector<float> & cur) {
    const int * rows = cs.rows.data() + s0;
    std::vector<Qwen4ExpCache *> caches;
    for (int s = 0; s < n_slots; ++s) caches.push_back(&slots.caches[(size_t) (s0 + s)]);
    Graph g(be);
    ggml_tensor * in = g.input(GGML_TYPE_F32, w.n_embd, (int64_t) (cur.size() / (size_t) w.n_embd));
    std::vector<ggml_tensor *> logs((size_t) n_slots, nullptr);
    ggml_tensor * out = build_linear_attn_verify(g.ctx, g.gf, in, w.layers[0], w, caches.data(), n_slots, rows, 0,
        logs.data(), 1);
    g.allocate(out);
    ggml_backend_tensor_set(in, cur.data(), 0, ggml_nbytes(in));
    {
        const ScopedCudaGraphOverrides invariant(false, 0, false, 0, /*mmvq_batch_invariant=*/true,
                                                 /*mmvq_invariant_max_ncols=*/verify_group_rows(false));
        CHECK(ggml_backend_graph_compute(be, g.gf) == GGML_STATUS_SUCCESS);
    }
    std::vector<float> y(cur.size());
    ggml_backend_tensor_get(out, y.data(), 0, ggml_nbytes(out));
    for (int s = 0; s < n_slots; ++s) {
        Qwen4ExpCache & c = *caches[(size_t) s];
        c.spec_pos = 0;
        c.spec_tokens = rows[s];
        c.spec_log = {logs[(size_t) s]};
        CHECK(qwen4exp_verify_rollback(be, w, c, 0, cs.kept[(size_t) (s0 + s)]));
    }
    ggml_backend_synchronize(be);
    return y;
}

bool run_linear(ggml_backend_t be, const StateCase & cs, bool stacked, std::mt19937 & rng) {
    const LinearModel model(be, stacked, rng);
    const Qwen4ExpWeights & w = model.w;
    const int n = (int) cs.rows.size();
    const std::vector<int> row0 = verify_row_offsets(cs.rows.data(), n);
    const uint32_t seed = rng();
    LinearSlots batched(be, w, n, max_rows(cs), seed), solo(be, w, n, max_rows(cs), seed);
    const std::vector<uint8_t> ssm0 = bytes(batched.states.ssm[0]);
    const std::vector<float> cur = random_floats((size_t) w.n_embd * row0[(size_t) n], rng, 1.0f);
    // Group by group, as the graph forms them.
    const std::vector<int> first = verify_groups(cs.rows.data(), n, false);
    std::vector<float> got;
    for (size_t g = 0; g + 1 < first.size(); ++g) {
        const int s0 = first[g], s1 = first[g + 1];
        const std::vector<float> y = linear_rows(be, w, batched, cs, s0, s1 - s0,
            std::vector<float>(cur.begin() + (std::ptrdiff_t) row0[(size_t) s0] * w.n_embd,
                               cur.begin() + (std::ptrdiff_t) row0[(size_t) s1] * w.n_embd));
        got.insert(got.end(), y.begin(), y.end());
    }
    // The comparisons below mean something: the rows are numbers and the step moved the states.
    CHECK(std::all_of(got.begin(), got.end(), [](float x) { return std::isfinite(x); }));
    CHECK(bytes(batched.states.ssm[0]) != ssm0);

    bool ok = true;
    for (int s = 0; s < n; ++s) {
        Qwen4ExpCache & ref = solo.caches[(size_t) s];
        std::vector<uint8_t> ssm, conv;   // the decode's states after the rows the slot keeps
        for (int j = 0; j < cs.rows[(size_t) s]; ++j) {
            const size_t r = (size_t) row0[(size_t) s] + j;
            const std::vector<float> want = linear_decode_row(be, w, ref, cur.data() + r * w.n_embd);
            if (std::memcmp(want.data(), got.data() + r * w.n_embd, want.size() * sizeof(float)) != 0) {
                std::printf("batched-verify-linear: slot %d row %d differs from T=1 decode\n", s, j);
                ok = false;
            }
            if (j + 1 == cs.kept[(size_t) s]) {
                ssm = bytes(ref.ssm_state[0]);
                conv = bytes(ref.conv_state[0]);
            }
        }
        const Qwen4ExpCache & c = batched.caches[(size_t) s];
        if (bytes(c.ssm_state[0]) != ssm || bytes(c.conv_state[0]) != conv) {
            std::printf("batched-verify-linear: slot %d keeps other states than %d T=1 decode steps\n", s,
                        cs.kept[(size_t) s]);
            ok = false;
        }
    }
    std::printf("batched-verify-linear rows=%s kept=%s stacked=%d %s\n", join(cs.rows).c_str(),
                join(cs.kept).c_str(), (int) stacked, ok ? "OK" : "FAIL");
    return ok;
}

// One PLE layer with random weights: 256-wide table rows, four hyper-connection streams, kernel 4 over a dilation of
// 3 (the model's), Q8_0 projections.
struct PleModel {
    static constexpr int kTable = 256;
    Qwen4ExpWeights w;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    PleModel(ggml_backend_t be, std::mt19937 & rng) {
        w.gfx1151 = ggml_backend_cuda_qwen4exp_supported(be);
        w.n_layer = 1;
        w.n_embd = kEmbd;
        w.layers.resize(1);
        Qwen4ExpLayer & L = w.layers[0];
        L.is_ple = true;
        const int64_t hc_dim = (int64_t) w.n_hc * kEmbd;
        ctx = ggml_init({8 * ggml_tensor_overhead(), nullptr, true});
        L.ple_key = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, kTable, hc_dim);
        L.ple_value = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, kTable, kEmbd);
        L.ple_conv1d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.ple_conv_kernel, hc_dim);
        L.ple_norm_key = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hc_dim);
        L.ple_norm_query = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hc_dim);
        L.ple_norm_conv = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hc_dim);
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        CHECK(buf);
        for (ggml_tensor * t : {L.ple_key, L.ple_value}) fill(t, rng, 0.1f);
        fill(L.ple_conv1d, rng, 0.5f);
        for (ggml_tensor * t : {L.ple_norm_key, L.ple_norm_query, L.ple_norm_conv}) fill(t, rng, 0.2f, 1.0f);
    }
    ~PleModel() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

// A step's slots: each one's PLE convolution history (random, the same for every `seed`) and the rollback rows of a
// verify of up to `rows` rows.
struct PleSlots {
    std::vector<Qwen4ExpCache> caches;

    PleSlots(ggml_backend_t be, const Qwen4ExpWeights & w, int n_slots, int rows, uint32_t seed)
        : caches((size_t) n_slots) {
        const int64_t hist = (int64_t) (w.ple_conv_kernel - 1) * w.ple_ngram_size, hc_dim = (int64_t) w.n_hc * w.n_embd;
        std::mt19937 rng(seed);
        for (Qwen4ExpCache & c : caches) {
            c.ctx = ggml_init({(8 + (size_t) rows) * ggml_tensor_overhead(), nullptr, true});
            c.ple_layer_ids = {0};
            c.ple_conv_state = {ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, hist, hc_dim)};
            c.mtp_draft = std::max(1, rows - 1);
            c.spec_ple = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, hist, hc_dim, rows);
            for (int t = 0; t < rows; ++t) {
                c.spec_ple_rows[(size_t) t] = ggml_view_2d(c.ctx, c.spec_ple, hist, hc_dim, c.spec_ple->nb[1],
                                                           (size_t) t * c.spec_ple->nb[2]);
            }
            c.buf = ggml_backend_alloc_ctx_tensors(c.ctx, be);
            CHECK(c.buf);
            fill(c.ple_conv_state[0], rng, 1.0f);
        }
    }
    ~PleSlots() {
        for (Qwen4ExpCache & c : caches) {
            ggml_backend_buffer_free(c.buf);
            ggml_free(c.ctx);
        }
    }
};

// One T=1 decode step's PLE, as a solo decode graph builds it.
std::vector<float> ple_decode_row(ggml_backend_t be, const Qwen4ExpWeights & w, Qwen4ExpCache & cache,
                                  const float * hidden, const float * table) {
    Graph g(be);
    ggml_tensor * res = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, w.n_embd, w.n_hc, 1);
    ggml_set_input(res);
    ggml_tensor * in = g.input(GGML_TYPE_F32, PleModel::kTable);
    ggml_tensor * out = build_ple(g.ctx, g.gf, res, in, w.layers[0], w, cache.ple_conv_state[0]);
    g.allocate(out);
    ggml_backend_tensor_set(res, hidden, 0, ggml_nbytes(res));
    ggml_backend_tensor_set(in, table, 0, ggml_nbytes(in));
    CHECK(ggml_backend_graph_compute(be, g.gf) == GGML_STATUS_SUCCESS);
    std::vector<float> y((size_t) ggml_nelements(out));
    ggml_backend_tensor_get(out, y.data(), 0, ggml_nbytes(out));
    return y;
}

// The rows of one batched verify group (slots [s0, s0 + n_slots) of the step) through its PLE, then each slot's
// rollback to the rows it keeps.
std::vector<float> ple_rows(ggml_backend_t be, const Qwen4ExpWeights & w, PleSlots & slots, const StateCase & cs,
                            int s0, int n_slots, const std::vector<float> & hidden, const std::vector<float> & table) {
    const int * rows = cs.rows.data() + s0;
    const int64_t T = (int64_t) (table.size() / PleModel::kTable);
    std::vector<Qwen4ExpCache *> caches;
    for (int s = 0; s < n_slots; ++s) caches.push_back(&slots.caches[(size_t) (s0 + s)]);
    Graph g(be);
    ggml_tensor * res = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, w.n_embd, w.n_hc, T);
    ggml_set_input(res);
    ggml_tensor * in = g.input(GGML_TYPE_F32, PleModel::kTable, T);
    ggml_tensor * out = build_ple_verify(g.ctx, g.gf, res, in, w.layers[0], w, caches.data(), n_slots, rows, 0);
    g.allocate(out);
    ggml_backend_tensor_set(res, hidden.data(), 0, ggml_nbytes(res));
    ggml_backend_tensor_set(in, table.data(), 0, ggml_nbytes(in));
    {
        const ScopedCudaGraphOverrides invariant(false, 0, false, 0, /*mmvq_batch_invariant=*/true,
                                                 /*mmvq_invariant_max_ncols=*/verify_group_rows(false));
        CHECK(ggml_backend_graph_compute(be, g.gf) == GGML_STATUS_SUCCESS);
    }
    std::vector<float> y(hidden.size());
    ggml_backend_tensor_get(out, y.data(), 0, ggml_nbytes(out));
    for (int s = 0; s < n_slots; ++s) {
        Qwen4ExpCache & c = *caches[(size_t) s];
        c.spec_pos = 0;
        c.spec_tokens = rows[s];
        CHECK(qwen4exp_verify_rollback(be, w, c, 0, cs.kept[(size_t) (s0 + s)]));
    }
    ggml_backend_synchronize(be);
    return y;
}

bool run_ple(ggml_backend_t be, const StateCase & cs, std::mt19937 & rng) {
    const PleModel model(be, rng);
    const Qwen4ExpWeights & w = model.w;
    const int n = (int) cs.rows.size();
    const size_t hd = (size_t) w.n_embd * w.n_hc;
    const std::vector<int> row0 = verify_row_offsets(cs.rows.data(), n);
    const uint32_t seed = rng();
    PleSlots batched(be, w, n, max_rows(cs), seed), solo(be, w, n, max_rows(cs), seed);
    const std::vector<float> hidden = random_floats(hd * row0[(size_t) n], rng, 1.0f);
    const std::vector<float> table = random_floats((size_t) PleModel::kTable * row0[(size_t) n], rng, 1.0f);
    // Group by group, as the graph forms them.
    const std::vector<int> first = verify_groups(cs.rows.data(), n, false);
    std::vector<float> got;
    for (size_t g = 0; g + 1 < first.size(); ++g) {
        const std::ptrdiff_t r0 = row0[(size_t) first[g]], r1 = row0[(size_t) first[g + 1]];
        const std::vector<float> y = ple_rows(be, w, batched, cs, first[g], first[g + 1] - first[g],
            std::vector<float>(hidden.begin() + r0 * (std::ptrdiff_t) hd, hidden.begin() + r1 * (std::ptrdiff_t) hd),
            std::vector<float>(table.begin() + r0 * PleModel::kTable, table.begin() + r1 * PleModel::kTable));
        got.insert(got.end(), y.begin(), y.end());
    }
    CHECK(std::all_of(got.begin(), got.end(), [](float x) { return std::isfinite(x); }));

    bool ok = true;
    for (int s = 0; s < n; ++s) {
        Qwen4ExpCache & ref = solo.caches[(size_t) s];
        std::vector<uint8_t> state;   // the decode's history after the rows the slot keeps
        for (int j = 0; j < cs.rows[(size_t) s]; ++j) {
            const size_t r = (size_t) row0[(size_t) s] + j;
            const std::vector<float> want = ple_decode_row(be, w, ref, hidden.data() + r * hd,
                                                           table.data() + r * PleModel::kTable);
            if (std::memcmp(want.data(), got.data() + r * hd, want.size() * sizeof(float)) != 0) {
                std::printf("batched-verify-ple: slot %d row %d differs from T=1 decode\n", s, j);
                ok = false;
            }
            if (j + 1 == cs.kept[(size_t) s]) state = bytes(ref.ple_conv_state[0]);
        }
        if (bytes(batched.caches[(size_t) s].ple_conv_state[0]) != state) {
            std::printf("batched-verify-ple: slot %d keeps another history than %d T=1 decode steps\n", s,
                        cs.kept[(size_t) s]);
            ok = false;
        }
    }
    std::printf("batched-verify-ple rows=%s kept=%s %s\n", join(cs.rows).c_str(), join(cs.kept).c_str(),
                ok ? "OK" : "FAIL");
    return ok;
}

// The groups of every step of up to four slots of one to four rows, in any order: whole slots in order, at most
// verify_group_rows rows a group, no more groups than the rows need (two for a split step of several slots), and
// equal slots spread evenly.
bool groups_ok() {
    bool ok = true;
    auto check = [&](bool pass, const std::vector<int> & rows, bool split, const char * what) {
        if (pass) return;
        std::printf("batched-verify-groups: rows=%s split=%d %s\n", join(rows).c_str(), (int) split, what);
        ok = false;
    };
    std::vector<int> rows;
    std::function<void()> visit = [&] {
        for (const bool split : {false, true}) {
            if (rows.empty()) continue;
            const int n = (int) rows.size();
            const std::vector<int> first = verify_groups(rows.data(), n, split);
            const int groups = (int) first.size() - 1;
            check(first.front() == 0 && first.back() == n, rows, split, "does not cover the slots");
            int most = 0, fewest = n;
            for (int g = 0; g < groups; ++g) {
                const int s0 = first[(size_t) g], slots = first[(size_t) g + 1] - s0;
                const int sum = std::accumulate(rows.begin() + s0, rows.begin() + s0 + slots, 0);
                check(slots > 0 && sum <= verify_group_rows(split), rows, split, "has an empty or oversized group");
                most = std::max(most, slots);
                fewest = std::min(fewest, slots);
            }
            int needed = 1;   // filling each group in turn
            for (int s = 0, sum = 0; s < n; sum += rows[(size_t) s++]) {
                if (sum + rows[(size_t) s] <= verify_group_rows(split)) continue;
                ++needed;
                sum = 0;
            }
            check(groups == std::max(needed, split && n >= 2 ? 2 : 1), rows, split, "does not form the fewest groups");
            if (std::count(rows.begin(), rows.end(), rows[0]) == n)
                check(most - fewest <= 1, rows, split, "spreads equal slots unevenly");
        }
        if (rows.size() == (size_t) kQwen4ExpMaxBatchedSlots) return;
        for (int r = kQwen4ExpBatchedVerifyRows; r >= 1; --r) {
            rows.push_back(r);
            visit();
            rows.pop_back();
        }
    };
    visit();
    // A drafting slot alone in its group beside three one-row slots in a split step; slots of four, three, one and
    // one rows as four and five rows there, and as one group on one device; one-row slots ahead of drafting ones.
    const std::vector<int> lone{4, 1, 1, 1}, mixed{4, 3, 1, 1}, ahead{1, 4, 1, 4};
    check(verify_groups(lone.data(), 4, true) == std::vector<int>({0, 1, 4}), lone, true, "is not 4 | 1,1,1");
    check(verify_groups(lone.data(), 4, false) == std::vector<int>({0, 4}), lone, false, "is not one group");
    check(verify_groups(mixed.data(), 4, true) == std::vector<int>({0, 1, 4}), mixed, true, "is not 4 | 3,1,1");
    check(verify_groups(mixed.data(), 4, false) == std::vector<int>({0, 4}), mixed, false, "is not one group");
    check(verify_groups(ahead.data(), 4, true) == std::vector<int>({0, 2, 4}), ahead, true, "is not 1,4 | 1,4");
    check(verify_groups(ahead.data(), 4, false) == std::vector<int>({0, 4}), ahead, false, "is not one group");
    std::printf("batched-verify-groups %s\n", ok ? "OK" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) return 77;
    bool ok = groups_ok();
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
        // a decode step of four slots, and the same four one-row slots through the verify.
        // Slots of different sizes: four QSA rows and three dense ones beside two one-row QSA slots; a one-row QSA
        // slot, then a one-row dense one, beside four rows (dense, crossing the boundary); a single drafting slot
        // beside the last dense position, the first QSA one and a one-row slot that completes a block; three, two and
        // one rows, all meeting the boundary or a block's end; thirteen rows in one group; one-row slots ahead of
        // drafting ones (the first QSA position, then a dense one), across the boundary.
        for (const Case & cs : std::vector<Case>{
                {{4, 4}, {2558, 40}, 4}, {{4, 4}, {2049, 3001}, 6}, {{4, 4}, {2050, 2051}, 7}, {{8}, {2503}, 8},
                {{2, 2, 2, 2}, {2302, 2303, 100, 2050}, 5}, {{1, 1, 1, 1}, {2559, 40, 2051, 3003}, 3, true},
                {{1, 1, 1, 1}, {2559, 40, 2051, 3003}, 3},
                {{4, 3, 1, 1}, {2558, 40, 2051, 3003}, 6}, {{4, 1}, {40, 2559}, 1}, {{4, 1}, {2049, 100}, 2},
                {{4, 1, 1, 1}, {300, 2050, 2051, 2303}, 2}, {{3, 2, 1}, {2050, 2302, 2559}, 5},
                {{4, 4, 4, 1}, {40, 300, 2049, 2559}, 3}, {{1, 4, 1, 4}, {2051, 2048, 40, 2300}, 6},
                {{1, 4}, {2559, 2049}, 3}}) {
            for (const bool copied : {false, true}) {
                ok &= run(be, w, cs, copied, rng);
                ++cases;
            }
        }
    }
    if (ggml_backend_cuda_gdn_chain_uses_grouped_cols(ggml_backend_cuda_get_device_id(be), 128)) {
        std::mt19937 rng(20261111);
        const Qwen4ExpCudaScope profile(ggml_backend_cuda_qwen4exp_supported(be));
        // Slots of four, three, one and one rows; a one-row slot beside four rows; a single drafting slot beside
        // three one-row slots; three, two and one rows; equal slots, which keep one to all of their rows; thirteen
        // rows in one group; one-row slots ahead of drafting ones; four one-row slots.
        for (const StateCase & cs : std::vector<StateCase>{
                {{4, 3, 1, 1}, {2, 3, 1, 1}}, {{4, 1}, {1, 1}}, {{4, 1, 1, 1}, {4, 1, 1, 1}}, {{3, 2, 1}, {3, 1, 1}},
                {{4, 4}, {3, 1}}, {{2, 2, 2, 2}, {1, 2, 2, 1}}, {{4, 4, 4, 1}, {2, 4, 1, 1}},
                {{1, 4, 1, 4}, {1, 2, 1, 4}}, {{1, 4}, {1, 3}}, {{1, 1, 1, 1}, {1, 1, 1, 1}}}) {
            for (const bool stacked : {false, true}) {
                ok &= run_linear(be, cs, stacked, rng);
                ++cases;
            }
            ok &= run_ple(be, cs, rng);
            ++cases;
        }
    } else {
        std::printf("batched-verify-linear: device has no verify rollback, skipped\n");
    }
    ggml_backend_free(be);
    std::printf("batched-verify-qsa: %d cases, %s\n", cases, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
