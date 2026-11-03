// Exact attention-mass scorers (pure CPU): the AVX2 scorer against the reference oracle (kvflash_mass_synth.h).

#include "CppUnitTestFramework.hpp"

#define KVFLASH_QK_PURE_ONLY
#include "kvflash_exact_mass.h"
#include "kvflash_exact_fast.h"
#include "kvflash_mass_synth.h"
#include "kvflash_query_capture.h"
#include "scoped_env.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <random>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashExactMassFixture {};

// Naive double reference over float keys K[c][l][hk][t][j].
std::vector<double> naive(const std::vector<float> & K, int C, int ct, const KvFlashQkDims & d,
                          const std::vector<float> & q) {
    const int group = d.n_q_heads / d.n_kv_heads;
    std::vector<double> out((size_t)C, 0.0);
    for (int l = 0; l < d.n_layers; l++)
        for (int hq = 0; hq < d.n_q_heads; hq++) {
            std::vector<double> s;
            for (int c = 0; c < C; c++)
                for (int t = 0; t < ct; t++) {
                    const float * k = &K[((((size_t)c * d.n_layers + l) * d.n_kv_heads + hq / group) * ct + t) * d.head_dim];
                    const float * qq = &q[((size_t)l * d.n_q_heads + hq) * d.head_dim];
                    double dot = 0;
                    for (int j = 0; j < d.head_dim; j++) dot += (double)qq[j] * k[j];
                    s.push_back(dot / std::sqrt((double)d.head_dim));
                }
            double mx = *std::max_element(s.begin(), s.end()), Z = 0;
            for (double v : s) Z += std::exp(v - mx);
            for (int c = 0; c < C; c++) {
                double m = 0;
                for (int t = 0; t < ct; t++) m += std::exp(s[(size_t)c * ct + t] - mx);
                out[(size_t)c] += m / Z / (d.n_layers * d.n_q_heads);
            }
        }
    return out;
}

// Returns false on a setup failure (REQUIRE cannot be used in a helper).
bool run_case(ggml_type type, double tol, std::vector<float> & out, std::vector<double> & ref) {
    KvFlashQkDims d; d.n_layers = 2; d.n_q_heads = 4; d.n_kv_heads = 2; d.head_dim = 32;
    const int C = 5, ct = 8;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> K((size_t)C * d.n_layers * d.n_kv_heads * ct * d.head_dim);
    for (auto & v : K) v = nd(rng);
    std::vector<float> q((size_t)d.n_layers * d.n_q_heads * d.head_dim);
    for (auto & v : q) v = nd(rng);
    std::vector<uint8_t> buf;
    const size_t nel = K.size();
    if (type == GGML_TYPE_F32) {
        buf.resize(nel * 4);
        std::memcpy(buf.data(), K.data(), nel * 4);
    } else {
        buf.resize(nel * 2);
        ggml_fp32_to_fp16_row(K.data(), (ggml_fp16_t *)buf.data(), (int64_t)nel);
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)buf.data(), K.data(), (int64_t)nel);   // reference sees the rounded keys
    }
    const size_t esz = type == GGML_TYPE_F32 ? 4 : 2;
    KvFlashExactKeys keys;
    keys.n_chunks = C; keys.chunk_tokens = ct; keys.type = type;
    keys.seg = [&](int c, int l, int h) -> const uint8_t * {
        return buf.data() + ((((size_t)c * d.n_layers + l) * d.n_kv_heads + h) * ct * d.head_dim) * esz;
    };
    (void)tol;
    if (!kvflash_exact_mass_scores(keys, d, q.data(), out, 3)) return false;
    ref = naive(K, C, ct, d, q);
    return true;
}

}

TEST_CASE(KvflashExactMassFixture, matches_naive_reference_f32_gqa) {
    std::vector<float> out; std::vector<double> ref;
    REQUIRE(run_case(GGML_TYPE_F32, 1e-5, out, ref));
    double sum = 0;
    for (size_t c = 0; c < ref.size(); c++) { CHECK(std::fabs(out[c] - ref[c]) < 1e-5); sum += out[c]; }
    CHECK(std::fabs(sum - 1.0) < 1e-4);   // each head's masses sum to 1
}
TEST_CASE(KvflashExactMassFixture, matches_naive_reference_f16_gqa) {
    std::vector<float> out; std::vector<double> ref;
    REQUIRE(run_case(GGML_TYPE_F16, 1e-5, out, ref));
    double sum = 0;
    for (size_t c = 0; c < ref.size(); c++) { CHECK(std::fabs(out[c] - ref[c]) < 1e-5); sum += out[c]; }
    CHECK(std::fabs(sum - 1.0) < 1e-4);
}

TEST_CASE(KvflashExactMassFixture, absent_chunk_scores_zero_and_rest_renormalize) {
    KvFlashQkDims d; d.n_layers = 1; d.n_q_heads = 2; d.n_kv_heads = 1; d.head_dim = 4;
    std::vector<float> k(2 * 4, 0.5f), q(2 * 4, 1.0f);
    KvFlashExactKeys keys;
    keys.n_chunks = 3; keys.chunk_tokens = 2; keys.type = GGML_TYPE_F32;
    keys.seg = [&](int c, int, int) -> const uint8_t * { return c == 1 ? nullptr : (const uint8_t *)k.data(); };
    std::vector<float> out;
    REQUIRE(kvflash_exact_mass_scores(keys, d, q.data(), out, 1));
    CHECK(out[1] == 0.0f);
    CHECK(std::fabs(out[0] - 0.5f) < 1e-6);
    CHECK(std::fabs(out[2] - 0.5f) < 1e-6);
}

TEST_CASE(KvflashExactMassFixture, multi_query_equals_single) {
    KvFlashQkDims d; d.n_layers = 1; d.n_q_heads = 2; d.n_kv_heads = 1; d.head_dim = 8;
    std::mt19937 rng(3); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> k(4 * 3 * 8), q1(16), q2(16);
    for (auto & v : k) v = nd(rng);
    for (auto & v : q1) v = nd(rng);
    for (auto & v : q2) v = nd(rng);
    KvFlashExactKeys keys;
    keys.n_chunks = 4; keys.chunk_tokens = 3; keys.type = GGML_TYPE_F32;
    keys.seg = [&](int c, int, int) -> const uint8_t * { return (const uint8_t *)(k.data() + (size_t)c * 24); };
    std::vector<std::vector<float>> o;
    REQUIRE(kvflash_exact_mass_scores_multi(keys, d, {q1.data(), q2.data()}, o, 2));
    std::vector<float> a, b;
    REQUIRE(kvflash_exact_mass_scores(keys, d, q1.data(), a, 1));
    REQUIRE(kvflash_exact_mass_scores(keys, d, q2.data(), b, 1));
    for (int c = 0; c < 4; c++) {
        CHECK(std::fabs(o[0][(size_t)c] - a[(size_t)c]) < 1e-6);
        CHECK(std::fabs(o[1][(size_t)c] - b[(size_t)c]) < 1e-6);
    }
}

// Microbenchmark, off by default: LUCE_EXACT_BENCH=1. 36K tokens, 16 layers,
// 24 q heads, 4 kv heads, 256 dim, f16 (one chunk's data reused for all).
TEST_CASE(KvflashExactMassFixture, bench_36k_tokens_f16) {
    if (!std::getenv("LUCE_EXACT_BENCH")) return;
    KvFlashQkDims d; d.n_layers = 16; d.n_q_heads = 24; d.n_kv_heads = 4; d.head_dim = 256;
    const int ct = 64, C = 576;
    std::mt19937 rng(1); std::normal_distribution<float> nd(0.f, 1.f);
    const size_t nel = (size_t)d.n_layers * d.n_kv_heads * ct * d.head_dim;
    std::vector<float> f(nel), q((size_t)d.n_layers * d.n_q_heads * d.head_dim);
    for (auto & v : f) v = nd(rng);
    for (auto & v : q) v = nd(rng);
    std::vector<ggml_fp16_t> buf(nel);
    ggml_fp32_to_fp16_row(f.data(), buf.data(), (int64_t)nel);
    KvFlashExactKeys keys;
    keys.n_chunks = C; keys.chunk_tokens = ct; keys.type = GGML_TYPE_F16;
    const size_t seg_elems = (size_t)ct * d.head_dim;
    keys.seg = [&](int, int l, int h) -> const uint8_t * {
        return (const uint8_t *)(buf.data() + ((size_t)l * d.n_kv_heads + h) * seg_elems);
    };
    std::vector<float> out;
    const auto t0 = std::chrono::steady_clock::now();
    REQUIRE(kvflash_exact_mass_scores(keys, d, q.data(), out));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[bench] exact mass 36864 tokens x16L x24H x256D f16: %.1f ms (%u hw threads)\n",
                ms, std::thread::hardware_concurrency());
}

TEST_CASE(KvflashExactMassFixture, select_scores_take_the_max_over_queries_of_top_planes) {
    std::vector<KvFlashMassPlanes> pl(2);
    pl[0].top = {0.1f, 0.5f, 0.0f}; pl[0].max = {9.0f, 9.0f, 9.0f};
    pl[1].top = {0.3f, 0.2f, 0.0f}; pl[1].max = {9.0f, 9.0f, 9.0f};
    CHECK((kvflash_select_scores(pl) == std::vector<float>{0.3f, 0.5f, 0.0f}));
    CHECK(kvflash_select_scores({}).empty());
}
TEST_CASE(KvflashExactMassFixture, stage_rows_exist_only_in_a_kvflash_process) {
    luce_test::ScopedEnvVar off("LUCE_KVFLASH", nullptr);
    CHECK(kvflash_q_stage_rows(16) == 0);
    {
        luce_test::ScopedEnvVar on("LUCE_KVFLASH", "auto");
        CHECK(kvflash_q_stage_rows(16) == 16);
        CHECK(kvflash_q_stage_rows(-3) == 0);
    }
    {
        luce_test::ScopedEnvVar pool("LUCE_KVFLASH", "4096");
        CHECK(kvflash_q_stage_rows(16) == 16);
    }
    luce_test::ScopedEnvVar zero("LUCE_KVFLASH", "0");
    CHECK(kvflash_q_stage_rows(16) == 0);
}

// ---- fast scorer vs the double-precision reference -------------------------
namespace {
struct FastCase {
    KvFlashQkDims d; int C = 0, ct = 0; ggml_type type = GGML_TYPE_Q4_0;
    std::vector<uint8_t> buf; std::vector<float> q; size_t seg_bytes = 0;
    KvFlashExactKeys keys() const {
        KvFlashExactKeys k; k.n_chunks = C; k.chunk_tokens = ct; k.type = type;
        k.seg = [this](int c, int l, int h) -> const uint8_t * {
            return buf.data() + (((size_t)c * d.n_layers + l) * d.n_kv_heads + h) * seg_bytes;
        };
        return k;
    }
};
FastCase make_case(ggml_type type, int n_layers, int hq, int hkv, int D, int C, int ct, unsigned seed) {
    FastCase f; f.type = type; f.C = C; f.ct = ct;
    f.d.n_layers = n_layers; f.d.n_q_heads = hq; f.d.n_kv_heads = hkv; f.d.head_dim = D;
    std::mt19937 rng(seed); std::normal_distribution<float> nd(0.f, 1.f);
    const size_t nseg = (size_t)C * n_layers * hkv;
    f.seg_bytes = ggml_row_size(type, D) * (size_t)ct;
    std::vector<float> src((size_t)ct * D);
    f.buf.resize(nseg * f.seg_bytes);
    const auto * tt = ggml_get_type_traits(type);
    for (size_t s = 0; s < nseg; s++) {
        for (auto & v : src) v = nd(rng);
        uint8_t * dst = f.buf.data() + s * f.seg_bytes;
        if (type == GGML_TYPE_F32) std::memcpy(dst, src.data(), f.seg_bytes);
        else for (int t = 0; t < ct; t++) tt->from_float_ref(&src[(size_t)t * D], dst + (size_t)t * ggml_row_size(type, D), D);
    }
    f.q.resize((size_t)n_layers * hq * D);
    for (auto & v : f.q) v = nd(rng);
    return f;
}
// The fast scorer on one query.
bool fast_one(const KvFlashExactKeys & keys, const KvFlashQkDims & d, const float * q, std::vector<float> & out,
              int n_threads = 0, bool simd = true) {
    std::vector<std::vector<float>> o;
    const bool ok = kvflash_exact_mass_scores_fast_multi(keys, d, {q}, o, n_threads, nullptr, nullptr, simd);
    out = std::move(o[0]);
    return ok;
}
// Max relative per-chunk and per-block difference of fast vs reference; false on failure.
bool compare_fast(const FastCase & f, int cpb, double & chunk_rel, double & block_rel, bool simd = true) {
    std::vector<float> a, b;
    const auto keys = f.keys();
    if (!kvflash_exact_mass_scores(keys, f.d, f.q.data(), a, 3)) return false;
    if (!fast_one(keys, f.d, f.q.data(), b, 3, simd)) return false;
    chunk_rel = 0;
    for (size_t c = 0; c < a.size(); c++) chunk_rel = std::max(chunk_rel, std::fabs((double)a[c] - b[c]) / std::fabs((double)a[c]));
    const auto ba = kvflash_block_scores(a, cpb), bb = kvflash_block_scores(b, cpb);
    block_rel = 0;
    for (size_t i = 0; i < ba.size(); i++) block_rel = std::max(block_rel, std::fabs(ba[i] - bb[i]) / std::fabs(ba[i]));
    return true;
}
}

TEST_CASE(KvflashExactMassFixture, fast_matches_reference_q4_0_gqa_6) {
    for (bool scalar : {false, true}) {
        // qwen35-27B shape scaled down: group 6, D multiple of 32, 5 blocks of 4 chunks
        const auto f = make_case(GGML_TYPE_Q4_0, 3, 12, 2, 64, 20, 16, 11);
        double cr, br;
        REQUIRE(compare_fast(f, 4, cr, br, !scalar));
        CHECK(cr < 1e-4);
        CHECK(br < 1e-4);
    }
}
TEST_CASE(KvflashExactMassFixture, fast_matches_reference_other_types_and_groups) {
    struct Cfg { ggml_type t; int hq, hkv; };
    for (const Cfg c : {Cfg{GGML_TYPE_F16, 8, 2}, Cfg{GGML_TYPE_F32, 3, 1}, Cfg{GGML_TYPE_Q8_0, 14, 2}, Cfg{GGML_TYPE_Q4_0, 7, 1}, Cfg{GGML_TYPE_Q4_0, 2, 2}}) {
        const auto f = make_case(c.t, 2, c.hq, c.hkv, 64, 12, 8, 5);   // group 4, 3, 7, 7, 1 (7 > the 6-wide tile)
        double cr, br;
        REQUIRE(compare_fast(f, 3, cr, br));
        CHECK(cr < 1e-4);
        CHECK(br < 1e-4);
    }
}
TEST_CASE(KvflashExactMassFixture, fast_absent_chunk_scores_zero) {
    auto f = make_case(GGML_TYPE_Q4_0, 1, 6, 1, 32, 6, 8, 2);
    auto keys = f.keys();
    const auto seg = keys.seg;
    keys.seg = [&](int c, int l, int h) -> const uint8_t * { return c == 2 ? nullptr : seg(c, l, h); };
    std::vector<float> a, b;
    REQUIRE(kvflash_exact_mass_scores(keys, f.d, f.q.data(), a, 2));
    REQUIRE(fast_one(keys, f.d, f.q.data(), b, 2));
    CHECK(b[2] == 0.0f);
    for (int c = 0; c < 6; c++) CHECK(std::fabs(a[(size_t)c] - b[(size_t)c]) <= 1e-4 * std::max(a[(size_t)c], 1e-6f));
}
// Documented near-tie behaviour. (1) Bit-identical blocks tie exactly in BOTH
// scorers (same ops on the same bytes), so the newer-first tie-break is kept.
// (2) Blocks whose scores differ by less than the float noise (~1e-6 relative)
// may legitimately rank either way between the two scorers.
TEST_CASE(KvflashExactMassFixture, fast_exact_ties_keep_newer_first_and_near_ties_are_bounded) {
    auto f = make_case(GGML_TYPE_Q4_0, 2, 12, 2, 64, 40, 16, 3);
    const size_t per_chunk = (size_t)f.d.n_layers * f.d.n_kv_heads * f.seg_bytes;
    // blocks 2 and 6 (chunks 8-11 and 24-27) get identical key bytes
    for (int i = 0; i < 4; i++) std::memcpy(&f.buf[(size_t)(24 + i) * per_chunk], &f.buf[(size_t)(8 + i) * per_chunk], per_chunk);
    std::vector<float> a, b;
    REQUIRE(kvflash_exact_mass_scores(f.keys(), f.d, f.q.data(), a, 3));
    REQUIRE(fast_one(f.keys(), f.d, f.q.data(), b, 3));
    const auto ba = kvflash_block_scores(a, 4), bb = kvflash_block_scores(b, 4);
    // identical K gives identical logsumexp per chunk, but the softmax is over all chunks, so block 2 == block 6 in both
    CHECK(ba[2] == ba[6]);
    CHECK(bb[2] == bb[6]);
    // No ranking helper ships in this PR (the recall pick is a later one), so assert the tie-break contract it
    // uses on the block scores: newest first, then a stable sort by score descending. The newer of the tied blocks (6) must
    // come out ahead of the older (2) in both scorers.
    for (const auto & bs : {ba, bb}) {
        std::vector<int> order;
        for (int b = (int)bs.size() - 1; b >= 0; b--) order.push_back(b);
        std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return bs[(size_t)x] > bs[(size_t)y]; });
        const auto at = [&](int b) { return std::find(order.begin(), order.end(), b) - order.begin(); };
        CHECK(at(6) < at(2));
    }
    // gap between the two scorers on any block stays far below any realistic rank gap
    for (size_t i = 0; i < ba.size(); i++) CHECK(std::fabs(ba[i] - bb[i]) / ba[i] < 1e-4);
}
// ---- run_workers / bad inputs (review fixes) -----------------------------------
TEST_CASE(KvflashExactMassFixture, run_workers_zero_or_negative_workers_run_nothing) {
    std::atomic<int> calls{0};
    run_workers(0, [&](int) { calls++; });
    run_workers(-3, [&](int) { calls++; });
    CHECK(calls == 0);
    run_workers(1, [&](int) { calls++; });
    CHECK(calls == 1);
}
TEST_CASE(KvflashExactMassFixture, run_workers_rethrows_a_spawned_worker_exception_after_joining_all) {
    std::atomic<int> done{0};
    bool caught = false;
    try {
        run_workers(4, [&](int w) { if (w == 2) throw std::runtime_error("worker 2"); done++; });
    } catch (const std::runtime_error & e) { caught = std::string(e.what()) == "worker 2"; }
    CHECK(caught);
    CHECK(done == 3);   // every other worker ran to the end before the rethrow
}
TEST_CASE(KvflashExactMassFixture, run_workers_rethrows_a_caller_worker_exception_after_joining_all) {
    std::atomic<int> done{0};
    bool caught = false;
    try {
        run_workers(3, [&](int w) { if (w == 0) throw std::runtime_error("worker 0"); done++; });
    } catch (const std::runtime_error & e) { caught = std::string(e.what()) == "worker 0"; }
    CHECK(caught);
    CHECK(done == 2);
}
TEST_CASE(KvflashExactMassFixture, fast_multi_refuses_dims_that_would_drop_query_heads) {
    const auto f = make_case(GGML_TYPE_Q4_0, 2, 12, 2, 64, 6, 16, 9);
    std::vector<float> q = f.q;
    q.resize(q.size() * 2, 0.5f);   // room for the oversized dims below: a refusal must not read it
    std::vector<std::vector<float>> o;
    CHECK(kvflash_exact_mass_scores_fast_multi(f.keys(), f.d, {q.data()}, o, 2));
    auto bad = [&](KvFlashQkDims d) { return !kvflash_exact_mass_scores_fast_multi(f.keys(), d, {q.data()}, o, 2); };
    KvFlashQkDims d = f.d;
    d.n_q_heads = 13; CHECK(bad(d));                      // 13 % 2: the last q head would be dropped
    d = f.d; d.n_q_heads = 1; CHECK(bad(d));              // fewer q heads than kv heads
    d = f.d; d.n_layers = 0; CHECK(bad(d));
    d = f.d; d.n_layers = -1; CHECK(bad(d));
    d = f.d; d.n_kv_heads = 0; CHECK(bad(d));
    d = f.d; d.n_q_heads = 0; CHECK(bad(d));
    d = f.d; d.head_dim = 0; CHECK(bad(d));
}
TEST_CASE(KvflashExactMassFixture, reference_and_fast_scorers_both_refuse_a_null_query) {
    const auto f = make_case(GGML_TYPE_Q4_0, 2, 12, 2, 64, 6, 16, 9);
    std::vector<std::vector<float>> a, b;
    CHECK(!kvflash_exact_mass_scores_multi(f.keys(), f.d, {f.q.data(), nullptr}, a, 2));
    CHECK(!kvflash_exact_mass_scores_fast_multi(f.keys(), f.d, {f.q.data(), nullptr}, b, 2));
}
TEST_CASE(KvflashExactMassFixture, fast_bench_36k_q4_0) {
    if (!std::getenv("LUCE_EXACT_BENCH")) return;
    const auto f = make_case(GGML_TYPE_Q4_0, 16, 24, 4, 256, 64, 64, 1);   // 64 chunks, replicated
    KvFlashExactKeys keys = f.keys();
    const auto seg = keys.seg;
    keys.n_chunks = 576;
    keys.seg = [&](int c, int l, int h) { return seg(c % 64, l, h); };
    std::vector<float> a, b;
    auto t0 = std::chrono::steady_clock::now();
    REQUIRE(kvflash_exact_mass_scores(keys, f.d, f.q.data(), a));
    auto t1 = std::chrono::steady_clock::now();
    REQUIRE(fast_one(keys, f.d, f.q.data(), b));
    auto t2 = std::chrono::steady_clock::now();
    std::printf("[bench] q4_0 36864 tokens: reference %.1f ms, fast %.1f ms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(), std::chrono::duration<double, std::milli>(t2 - t1).count());
}

// ---- selector v2 scoring: several queries per pass, head aggregates ------------
namespace {
// Naive double chunk mass of plane x = l * n_q_heads + hq for query q.
std::vector<double> naive_plane(const FastCase & f, const float * q, int x) {
    const int l = x / f.d.n_q_heads, hq = x % f.d.n_q_heads, group = f.d.n_q_heads / f.d.n_kv_heads, D = f.d.head_dim;
    const auto * tt = ggml_get_type_traits(f.type);
    const auto keys = f.keys();
    std::vector<float> rows((size_t)f.ct * D);
    std::vector<double> lse((size_t)f.C);
    for (int c = 0; c < f.C; c++) {
        const uint8_t * p = keys.seg(c, l, hq / group);
        if (f.type == GGML_TYPE_F32) std::memcpy(rows.data(), p, rows.size() * 4);
        else tt->to_float(p, rows.data(), (int64_t)rows.size());
        std::vector<double> s((size_t)f.ct);
        for (int t = 0; t < f.ct; t++) {
            double dot = 0;
            for (int j = 0; j < D; j++) dot += (double)q[(size_t)x * D + j] * rows[(size_t)t * D + j];
            s[(size_t)t] = dot / std::sqrt((double)D);
        }
        const double mx = *std::max_element(s.begin(), s.end());
        double z = 0;
        for (double v : s) z += std::exp(v - mx);
        lse[(size_t)c] = mx + std::log(z);
    }
    const double mx = *std::max_element(lse.begin(), lse.end());
    double Z = 0;
    for (double v : lse) Z += std::exp(v - mx);
    for (auto & v : lse) v = std::exp(v - mx) / Z;
    return lse;
}
std::vector<float> random_query(const FastCase & f, unsigned seed) {
    std::mt19937 rng(seed); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q(f.q.size());
    for (auto & v : q) v = nd(rng);
    return q;
}
}

TEST_CASE(KvflashExactMassFixture, fast_multi_equals_single_per_query) {
    for (bool scalar : {false, true}) {
        for (int hq : {12, 14}) {   // group 6 (tile-aligned) and 7 (tiles straddle queries)
            const auto f = make_case(GGML_TYPE_Q4_0, 2, hq, 2, 64, 20, 16, 9);
            const auto q1 = random_query(f, 21), q2 = random_query(f, 22);
            std::vector<std::vector<float>> o;
            REQUIRE(kvflash_exact_mass_scores_fast_multi(f.keys(), f.d, {f.q.data(), q1.data(), q2.data()}, o, 3, nullptr, nullptr, !scalar));
            REQUIRE(o.size() == 3);
            std::vector<float> s;
            for (int i = 0; i < 3; i++) {
                REQUIRE(fast_one(f.keys(), f.d, i == 0 ? f.q.data() : i == 1 ? q1.data() : q2.data(), s, 3, !scalar));
                CHECK((o[(size_t)i] == s));
            }
        }
    }
}

TEST_CASE(KvflashExactMassFixture, fast_multi_plane_aggregates_match_naive) {
    const auto f = make_case(GGML_TYPE_Q4_0, 2, 12, 2, 64, 10, 16, 4);   // 24 planes, top 8 of them
    const auto q1 = random_query(f, 31);
    std::vector<std::vector<float>> o;
    std::vector<KvFlashMassPlanes> pl;
    REQUIRE(kvflash_exact_mass_scores_fast_multi(f.keys(), f.d, {f.q.data(), q1.data()}, o, 2, nullptr, &pl));
    REQUIRE(pl.size() == 2);
    const int LH = f.d.n_layers * f.d.n_q_heads;
    for (int i = 0; i < 2; i++) {
        std::vector<std::vector<double>> m;
        for (int x = 0; x < LH; x++) m.push_back(naive_plane(f, i == 0 ? f.q.data() : q1.data(), x));
        REQUIRE(pl[(size_t)i].max.size() == (size_t)f.C);
        REQUIRE(pl[(size_t)i].top.size() == (size_t)f.C);
        for (int c = 0; c < f.C; c++) {
            std::vector<double> v;
            double mean = 0;
            for (int x = 0; x < LH; x++) { v.push_back(m[(size_t)x][(size_t)c]); mean += m[(size_t)x][(size_t)c] / LH; }
            std::sort(v.rbegin(), v.rend());
            double top = 0;
            for (int k = 0; k < kExactTopPlanes; k++) top += v[(size_t)k] / kExactTopPlanes;
            CHECK(std::fabs(o[(size_t)i][(size_t)c] - mean) <= 1e-4 * mean);
            CHECK(std::fabs(pl[(size_t)i].max[(size_t)c] - v[0]) <= 1e-4 * v[0]);
            CHECK(std::fabs(pl[(size_t)i].top[(size_t)c] - top) <= 1e-4 * top);
        }
    }
}

TEST_CASE(KvflashExactMassFixture, fast_multi_absent_chunk_aggregates_zero) {
    auto f = make_case(GGML_TYPE_Q4_0, 1, 6, 1, 32, 6, 8, 2);
    auto keys = f.keys();
    const auto seg = keys.seg;
    keys.seg = [&](int c, int l, int h) -> const uint8_t * { return c == 2 ? nullptr : seg(c, l, h); };
    std::vector<std::vector<float>> o;
    std::vector<KvFlashMassPlanes> pl;
    REQUIRE(kvflash_exact_mass_scores_fast_multi(keys, f.d, {f.q.data()}, o, 2, nullptr, &pl));
    CHECK(o[0][2] == 0.0f);
    CHECK(pl[0].max[2] == 0.0f);
    CHECK(pl[0].top[2] == 0.0f);
    CHECK(pl[0].max[0] > 0.0f);
}

