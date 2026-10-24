// Amendment 10 step 0: the fused chunk-mass math (per plane and chunk a local max and local sum-exp, then a log-sum-exp
// over chunks) against the existing exact scorers. Pure CPU; the GPU kernel is tested against this reference.

#include "CppUnitTestFramework.hpp"

#define KVFLASH_QK_PURE_ONLY
#include "kvflash_exact_fast.h"
#include "kvflash_exact_mass.h"
#include "kvflash_fused_mass.h"
#include "kvflash_mass_synth.h"   // the reference scorer

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
