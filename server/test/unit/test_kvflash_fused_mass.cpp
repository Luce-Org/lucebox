// Amendment 10 step 0: the fused chunk-mass math (per plane and chunk a local max and local sum-exp, then a log-sum-exp
// over chunks) against the existing exact scorers. Pure CPU; the GPU kernel is tested against this reference.

#include "CppUnitTestFramework.hpp"

#define KVFLASH_QK_PURE_ONLY
#include "kvflash_exact_fast.h"
#include "kvflash_exact_mass.h"
#include "kvflash_fused_mass.h"
#include "kvflash_mass_synth.h"   // the reference scorer
#ifdef LUCE_HAVE_KVFLASH_FUSED_MASS
#include "kvflash_fused_mass_cuda.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#endif

#include <random>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashFusedMassFixture {};

struct FSynth {
    KvFlashQkDims d;
    int C = 0, ct = 0;
    ggml_type type = GGML_TYPE_F32;
    std::vector<uint8_t> buf;
    std::vector<std::vector<float>> q;
    std::vector<bool> absent;
    size_t seg_bytes = 0;
    KvFlashExactKeys keys;
    void build(ggml_type t, int L, int Hq, int HK, int D, int chunks, int ctok, int nq, unsigned seed) {
        d.n_layers = L; d.n_q_heads = Hq; d.n_kv_heads = HK; d.head_dim = D;
        C = chunks; ct = ctok; type = t;
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.f, 1.5f);
        seg_bytes = (size_t)ct * ggml_row_size(t, D);
        buf.assign(seg_bytes * (size_t)C * L * HK, 0);
        std::vector<float> row((size_t)D);
        for (size_t r = 0; r < (size_t)C * L * HK * ct; r++) {
            for (auto & v : row) v = nd(rng);
            if (t == GGML_TYPE_F32) std::memcpy(&buf[r * D * 4], row.data(), (size_t)D * 4);
            else ggml_get_type_traits(t)->from_float_ref(row.data(), &buf[r * ggml_row_size(t, D)], D);
        }
        q.assign((size_t)nq, std::vector<float>((size_t)L * Hq * D));
        for (auto & x : q) for (auto & v : x) v = nd(rng);
        absent.assign((size_t)C, false);
        keys.n_chunks = C; keys.chunk_tokens = ct; keys.type = t;
        keys.seg = [this](int c, int l, int h) -> const uint8_t * {
            return absent[(size_t)c] ? nullptr : buf.data() + ((((size_t)c * d.n_layers + l) * d.n_kv_heads + h)) * seg_bytes;
        };
    }
    std::vector<const float *> qptrs() const { std::vector<const float *> p; for (auto & x : q) p.push_back(x.data()); return p; }
};

double rel(double a, double b) { return std::fabs(a - b) / std::max({std::fabs(a), std::fabs(b), 1e-30}); }

struct Worst { double mean = 0, planes_vs_double = 0, planes_vs_fast = 0; bool ok = false; };

// mean: fused vs kvflash_exact_mass_scores_multi (same float dot order, double lse). planes_vs_double: top/max from the float
// reduce vs a double recomputation from the same partials (isolates the reduce). planes_vs_fast: top/max vs the fast scorer,
// whose AVX dot order moves a single plane's logits by rounding, so its floor is a few 1e-6.
Worst worst_vs_reference(const FSynth & s, double floor_abs) {
    Worst w;
    std::vector<std::vector<float>> ref, got, fast;
    std::vector<KvFlashMassPlanes> fp, gp;
    if (!kvflash_exact_mass_scores_multi(s.keys, s.d, s.qptrs(), ref, 2)) return w;
    if (!kvflash_exact_mass_scores_fast_multi(s.keys, s.d, s.qptrs(), fast, 2, nullptr, &fp)) return w;
    KvFlashFusedPartials parts;
    if (!kvflash_fused_partials_ref(s.keys, s.d, s.qptrs(), parts)) return w;
    kvflash_fused_reduce_ref(parts, s.d, got, gp);
    w.ok = true;
    const int LH = parts.planes, C = parts.chunks;
    for (int i = 0; i < parts.nq; i++) {
        std::vector<double> mx((size_t)C, 0.0);
        std::vector<std::vector<double>> pm((size_t)C);
        for (int x = 0; x < LH; x++) {
            double M = -INFINITY, Z = 0;
            for (int c = 0; c < C; c++) if (parts.s(i, x, c) > 0) M = std::max(M, (double)parts.m(i, x, c) + std::log((double)parts.s(i, x, c)));
            if (M == -INFINITY) continue;
            for (int c = 0; c < C; c++) if (parts.s(i, x, c) > 0) Z += std::exp((double)parts.m(i, x, c) + std::log((double)parts.s(i, x, c)) - M);
            for (int c = 0; c < C; c++) {
                const double m = parts.s(i, x, c) > 0 ? std::exp((double)parts.m(i, x, c) + std::log((double)parts.s(i, x, c)) - M) / Z : 0.0;
                mx[(size_t)c] = std::max(mx[(size_t)c], m);
                pm[(size_t)c].push_back(m);
            }
        }
        for (int c = 0; c < C; c++) {
            auto & v = pm[(size_t)c];
            std::sort(v.begin(), v.end(), std::greater<double>());
            double top = 0;
            for (int k = 0; k < std::min<int>(kExactTopPlanes, (int)v.size()); k++) top += v[(size_t)k];
            top /= std::min(kExactTopPlanes, LH);
            const size_t k = (size_t)c;
            if (mx[k] > floor_abs) w.planes_vs_double = std::max(w.planes_vs_double, rel(mx[k], gp[(size_t)i].max[k]));
            if (top > floor_abs) w.planes_vs_double = std::max(w.planes_vs_double, rel(top, gp[(size_t)i].top[k]));
            if (ref[(size_t)i][k] > floor_abs) w.mean = std::max(w.mean, rel(ref[(size_t)i][k], got[(size_t)i][k]));
            else if (std::fabs(ref[(size_t)i][k] - got[(size_t)i][k]) > 1e-9) w.mean = 1.0;
            if (fp[(size_t)i].top[k] > floor_abs) w.planes_vs_fast = std::max(w.planes_vs_fast, rel(fp[(size_t)i].top[k], gp[(size_t)i].top[k]));
            if (fp[(size_t)i].max[k] > floor_abs) w.planes_vs_fast = std::max(w.planes_vs_fast, rel(fp[(size_t)i].max[k], gp[(size_t)i].max[k]));
        }
    }
    return w;
}

bool check_worst(const Worst & w, double fast_tol) {
    std::fprintf(stderr, "mean %.3g planes(double) %.3g planes(fast) %.3g\n", w.mean, w.planes_vs_double, w.planes_vs_fast);
    return w.ok && w.mean <= 1e-6 && w.planes_vs_double <= 1e-6 && w.planes_vs_fast <= fast_tol;
}
}  // namespace

TEST_CASE(KvflashFusedMassFixture, fused_math_matches_the_exact_scorer_f32_and_q8_0) {
    for (ggml_type t : {GGML_TYPE_F32, GGML_TYPE_Q8_0}) {
        FSynth s;
        s.build(t, 3, 6, 2, 64, 40, 8, 5, 11);   // 18 planes, G = 3
        CHECK(check_worst(worst_vs_reference(s, 1e-8), 1e-5));
    }
}

TEST_CASE(KvflashFusedMassFixture, qwen35_shaped_planes_g6_and_absent_chunks) {
    FSynth s;
    s.build(GGML_TYPE_Q8_0, 2, 24, 4, 64, 33, 16, 3, 5);   // G = 6, 48 planes > top-8
    s.absent[7] = s.absent[20] = true;
    CHECK(check_worst(worst_vs_reference(s, 1e-8), 1e-5));
}

TEST_CASE(KvflashFusedMassFixture, sharp_logits_do_not_overflow) {
    FSynth s;
    s.build(GGML_TYPE_F32, 1, 4, 2, 64, 12, 8, 2, 3);
    for (auto & x : s.q) for (auto & v : x) v *= 12.0f;   // logits in the hundreds
    CHECK(check_worst(worst_vs_reference(s, 1e-6), 1e-4));
}

// The MMQ-numerics reference stays near the float exact scorer (q is int8-quantized, as in today's graph).
TEST_CASE(KvflashFusedMassFixture, q8_reference_is_close_to_the_float_scorer) {
    FSynth s;
    s.build(GGML_TYPE_Q8_0, 2, 12, 2, 64, 20, 8, 3, 21);
    std::vector<int8_t> q8; std::vector<float> qd;
    const int LH = s.d.n_layers * s.d.n_q_heads, D = s.d.head_dim;
    q8.resize(s.q.size() * LH * D); qd.resize(s.q.size() * LH * D / 32);
    for (size_t i = 0; i < s.q.size(); i++) kvflash_quantize_q8_1(s.q[i].data(), LH * D, &q8[i * LH * D], &qd[i * LH * D / 32]);
    KvFlashFusedPartials a, b;
    REQUIRE(kvflash_fused_partials_ref(s.keys, s.d, s.qptrs(), a));
    REQUIRE(kvflash_fused_partials_q8_ref(s.keys, s.d, (int)s.q.size(), q8.data(), qd.data(), b));
    std::vector<std::vector<float>> oa, ob;
    std::vector<KvFlashMassPlanes> pa, pb;
    kvflash_fused_reduce_ref(a, s.d, oa, pa);
    kvflash_fused_reduce_ref(b, s.d, ob, pb);
    double w = 0;
    for (size_t i = 0; i < oa.size(); i++) for (size_t c = 0; c < oa[i].size(); c++) w = std::max(w, rel(oa[i][c], ob[i][c]));
    std::fprintf(stderr, "q8 vs float mean worst %.3g\n", w);
    CHECK(w < 0.1);
}

#ifdef LUCE_HAVE_KVFLASH_FUSED_MASS
namespace {
// Device run of the fused kernel on `s` (Q8_0): chunks alternate between the pool tensors (shuffled blocks) and the device copy.
// Returns the worst relative difference against the CPU reference with the same numerics; -1 on a setup failure.
double fused_gpu_worst(const FSynth & s, size_t max_partial_bytes, double floor_abs, int * tiles, int impl = 0) {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) SKIP("no CUDA/HIP device");
    const int L = s.d.n_layers, HK = s.d.n_kv_heads, D = s.d.head_dim, ct = s.ct, C = s.C, nq = (int)s.q.size();
    const int LH = L * s.d.n_q_heads, nblk = D / 32;
    const int NB = C / 2 + 3, cap = C + 5;
    ggml_init_params ip{ggml_tensor_overhead() * (size_t)(2 * L + 4), nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> pool, copy;
    for (int l = 0; l < L; l++) {
        pool.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, D, (int64_t)NB * ct, HK));
        copy.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, D, (int64_t)cap * ct, HK));
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) return -1;
    KvFlashFusedK k;
    k.src.resize((size_t)C);
    int nextblk = NB - 1;
    for (int c = 0; c < C; c++) k.src[(size_t)c] = (c % 2 == 0 && nextblk >= 0) ? nextblk-- : -1 - c;   // pool blocks in reverse, rest in the copy
    for (int l = 0; l < L; l++)
        for (int h = 0; h < HK; h++)
            for (int c = 0; c < C; c++) {
                const uint8_t * seg = s.keys.seg(c, l, h);
                const int sc = k.src[(size_t)c];
                ggml_tensor * t = sc >= 0 ? pool[(size_t)l] : copy[(size_t)l];
                const size_t at = (size_t)h * t->nb[2] + (size_t)(sc >= 0 ? sc : -1 - sc) * ct * t->nb[1];
                ggml_backend_tensor_set(t, seg, at, s.seg_bytes);
            }
    for (int l = 0; l < L; l++) { k.pool.push_back((const uint8_t *)pool[(size_t)l]->data); k.copy.push_back((const uint8_t *)copy[(size_t)l]->data); }
    k.pool_nb2 = pool[0]->nb[2]; k.copy_nb2 = copy[0]->nb[2];
    std::vector<int8_t> q8(s.q.size() * (size_t)LH * D); std::vector<float> qd(s.q.size() * (size_t)LH * nblk);
    for (size_t i = 0; i < s.q.size(); i++) kvflash_quantize_q8_1(s.q[i].data(), LH * D, &q8[i * LH * D], &qd[i * LH * nblk]);
    KvFlashFusedGeom g; g.L = L; g.HK = HK; g.Hq = s.d.n_q_heads; g.D = D; g.ct = ct; g.C = C; g.nq = nq; g.impl = impl;
    KvFlashFusedWs * ws = kvflash_fused_ws_new();
    std::vector<float> om((size_t)nq * C), ot((size_t)nq * C), ox((size_t)nq * C);
    KvFlashFusedTiming tm;
    std::string err;
    const bool ok = kvflash_fused_run(ws, g, k, q8.data(), qd.data(), max_partial_bytes, om.data(), ot.data(), ox.data(), &tm, &err);
    kvflash_fused_ws_free(ws);
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(be);
    if (!ok) { std::fprintf(stderr, "fused run failed: %s\n", err.c_str()); return -1; }
    if (tiles) *tiles = tm.tiles;
    KvFlashFusedPartials parts;
    if (!kvflash_fused_partials_q8_ref(s.keys, s.d, nq, q8.data(), qd.data(), parts)) return -1;
    std::vector<std::vector<float>> ro; std::vector<KvFlashMassPlanes> rp;
    kvflash_fused_reduce_ref(parts, s.d, ro, rp);
    double w = 0;
    for (int i = 0; i < nq; i++)
        for (int c = 0; c < C; c++) {
            const size_t o = (size_t)i * C + c;
            if (ro[(size_t)i][(size_t)c] > floor_abs) w = std::max(w, rel(ro[(size_t)i][(size_t)c], om[o]));
            if (rp[(size_t)i].top[(size_t)c] > floor_abs) w = std::max(w, rel(rp[(size_t)i].top[(size_t)c], ot[o]));
            if (rp[(size_t)i].max[(size_t)c] > floor_abs) w = std::max(w, rel(rp[(size_t)i].max[(size_t)c], ox[o]));
        }
    return w;
}
}  // namespace

TEST_CASE(KvflashFusedMassFixture, device_kernel_matches_the_cpu_reference) {
    FSynth s;
    s.build(GGML_TYPE_Q8_0, 3, 6, 2, 64, 41, 16, 5, 9);   // G = 3, 18 planes
    int tiles = 0;
    const double w = fused_gpu_worst(s, (size_t)1 << 30, 1e-8, &tiles);
    std::fprintf(stderr, "device vs reference worst %.3g (tiles %d)\n", w, tiles);
    CHECK(w >= 0);
    CHECK(w <= 1e-5);
}

TEST_CASE(KvflashFusedMassFixture, device_kernel_tiles_the_queries_without_changing_scores) {
    FSynth s;
    s.build(GGML_TYPE_Q8_0, 2, 24, 4, 128, 67, 64, 19, 4);   // G = 6, 48 planes, ct 64, D 128
    int t1 = 0, t2 = 0;
    const double a = fused_gpu_worst(s, (size_t)1 << 30, 1e-8, &t1);
    const double b = fused_gpu_worst(s, (size_t)48 * 67 * 2 * 4 * 4, 1e-8, &t2);   // 4 queries per tile
    std::fprintf(stderr, "device vs reference worst %.3g / %.3g (tiles %d / %d)\n", a, b, t1, t2);
    CHECK(a >= 0); CHECK(b >= 0);
    CHECK(a <= 1e-5); CHECK(b <= 1e-5);
    CHECK(t1 == 1);
    CHECK(t2 > 1);
}

TEST_CASE(KvflashFusedMassFixture, dp4a_impl_matches_and_wmma_impl_is_refused_off_gfx12) {
    FSynth s;
    s.build(GGML_TYPE_Q8_0, 3, 6, 2, 64, 41, 16, 5, 9);
    const double w = fused_gpu_worst(s, (size_t)1 << 30, 1e-8, nullptr, 1);
    CHECK(w >= 0); CHECK(w <= 1e-5);
#if !defined(GGML_USE_HIP)
    CHECK(fused_gpu_worst(s, (size_t)1 << 30, 1e-8, nullptr, 2) < 0);   // CUDA has no gfx12 WMMA kernel: explicit request fails, auto falls back
#endif
}

// Timing probe, off unless LUCE_FUSED_BENCH=<chunks> is set: random q8_0 K on the device, 128 queries, qwen35 shape.
TEST_CASE(KvflashFusedMassFixture, bench_device_kernel_when_asked) {
    const char * env = std::getenv("LUCE_FUSED_BENCH");
    if (!env) SKIP("LUCE_FUSED_BENCH not set");
    const int C = std::atoi(env), L = 16, HK = 4, Hq = 24, D = 256, ct = 64, nq = 128, nblk = D / 32;
    ggml_backend_t be = ggml_backend_cuda_init(0);
    REQUIRE(be != nullptr);
    ggml_init_params ip{ggml_tensor_overhead() * (size_t)(L + 4), nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> pool;
    for (int l = 0; l < L; l++) pool.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, D, (int64_t)C * ct, HK));
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    REQUIRE(buf != nullptr);
    std::mt19937 rng(1);
    std::vector<uint8_t> blob((size_t)64 << 20);
    for (size_t i = 0; i < blob.size(); i += 34) {
        const uint16_t d = ggml_fp32_to_fp16(0.02f);
        std::memcpy(&blob[i], &d, 2);
        for (int j = 0; j < 32 && i + 2 + j < blob.size(); j++) blob[i + 2 + j] = (uint8_t)(int8_t)((int)(rng() % 255) - 127);
    }
    for (auto * t : pool)
        for (size_t off = 0; off < ggml_nbytes(t); off += blob.size())
            ggml_backend_tensor_set(t, blob.data(), off, std::min(blob.size(), ggml_nbytes(t) - off));
    KvFlashFusedK k;
    k.src.resize((size_t)C);
    for (int c = 0; c < C; c++) k.src[(size_t)c] = c;
    for (auto * t : pool) k.pool.push_back((const uint8_t *)t->data);
    k.pool_nb2 = pool[0]->nb[2];
    const size_t LH = (size_t)L * Hq;
    std::vector<int8_t> q8((size_t)nq * LH * D);
    std::vector<float> qd((size_t)nq * LH * nblk);
    for (auto & v : q8) v = (int8_t)((int)(rng() % 255) - 127);
    for (auto & v : qd) v = 0.01f;
    KvFlashFusedGeom g; g.L = L; g.HK = HK; g.Hq = Hq; g.D = D; g.ct = ct; g.C = C; g.nq = nq;
    KvFlashFusedWs * ws = kvflash_fused_ws_new();
    std::vector<float> a((size_t)nq * C), b(a.size()), c2(a.size());
    KvFlashFusedTiming tm;
    std::string err;
    for (int rep = 0; rep < 3; rep++) {
        REQUIRE(kvflash_fused_run(ws, g, k, q8.data(), qd.data(), (size_t)256 << 20, a.data(), b.data(), c2.data(), &tm, &err));
        std::fprintf(stderr, "bench C=%d: up %.1f kernel %.1f reduce %.1f down %.1f ms, tiles %d, partials %.0f MiB\n", C, tm.up_ms, tm.kernel_ms,
                     tm.reduce_ms, tm.down_ms, tm.tiles, tm.partial_bytes / 1048576.0);
    }
    kvflash_fused_ws_free(ws);
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(be);
}
#endif
