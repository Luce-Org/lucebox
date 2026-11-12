// Fast exact attention-mass chunk scorer: same quantity as
// kvflash_exact_mass_scores (kvflash_exact_mass.h), computed with a fused
// q4_0 dequant + AVX2/FMA dot (one dequantized key row feeds every q head of
// its kv group), a vectorized float exp for the in-chunk logsumexp, and the
// same double-precision cross-chunk softmax. Falls back to scalar code when
// AVX2 is unavailable. Keys of other types go through ggml's to_float.
//
// Differences from the reference are float rounding only (dot order, expf):
// per-chunk scores agree to ~1e-6 relative; see test_kvflash_exact_mass.cpp.

#pragma once

#include "kvflash_block_scores.h"
#include "kvflash_exact_mass.h"
#include "parallel_for.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define KVFLASH_FAST_X86 1
#endif

namespace luce::common {
namespace fastmass {

constexpr int kQ4Block = 32;
constexpr int kMaxGroup = 6;

// Row dequant, bit-identical to ggml's dequantize_row_q4_0: (nib - 8) * d.
inline void dequant_row_q4_0_scalar(const uint8_t * p, float * out, int D) {
    for (int b = 0; b < D / kQ4Block; b++, p += 18) {
        uint16_t h; std::memcpy(&h, p, 2);
        const float d = ggml_fp16_to_fp32((ggml_fp16_t)h);
        for (int j = 0; j < 16; j++) {
            out[b * kQ4Block + j]      = (float)((int)(p[2 + j] & 0xF) - 8) * d;
            out[b * kQ4Block + j + 16] = (float)((int)(p[2 + j] >> 4) - 8) * d;
        }
    }
}

// max_g = number of q heads sharing the row; logits[g][t] = dot(q_g, row_t).
inline void dots_scalar(const float * row, const float * const * qg, int G, int D, float * out /*[G]*/) {
    for (int g = 0; g < G; g++) {
        float s = 0.0f;
        for (int j = 0; j < D; j++) s += qg[g][j] * row[j];
        out[g] = s;
    }
}

// lse = mx + log(sum exp(x - mx)); returns the float sum, x already minus nothing.
inline double lse_scalar(const float * x, int n) {
    float mx = -INFINITY;
    for (int t = 0; t < n; t++) mx = std::max(mx, x[t]);
    double s = 0.0;
    for (int t = 0; t < n; t++) s += std::exp((double)x[t] - mx);
    return mx + std::log(s);
}

#ifdef KVFLASH_FAST_X86
#define KVFLASH_AVX2 __attribute__((target("avx2,fma,f16c")))

KVFLASH_AVX2 inline __m256 nib8_to_f32(__m128i v, __m256 eight, __m256 d) {
    return _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v)), eight), d);
}

KVFLASH_AVX2 inline void dequant_row_q4_0_avx2(const uint8_t * p, float * out, int D) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m256 eight = _mm256_set1_ps(8.0f);
    for (int b = 0; b < D / kQ4Block; b++, p += 18, out += kQ4Block) {
        uint16_t h; std::memcpy(&h, p, 2);
        const __m256 d = _mm256_set1_ps(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))));
        const __m128i x  = _mm_loadu_si128((const __m128i *)(p + 2));
        const __m128i lo = _mm_and_si128(x, m4);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(x, 4), m4);
        _mm256_storeu_ps(out,      nib8_to_f32(lo, eight, d));
        _mm256_storeu_ps(out + 8,  nib8_to_f32(_mm_srli_si128(lo, 8), eight, d));
        _mm256_storeu_ps(out + 16, nib8_to_f32(hi, eight, d));
        _mm256_storeu_ps(out + 24, nib8_to_f32(_mm_srli_si128(hi, 8), eight, d));
    }
}

KVFLASH_AVX2 inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

// G q heads against one dequantized row (D % 8 == 0).
template <int G>
KVFLASH_AVX2 inline void dots_avx2(const float * row, const float * const * qg, int D, float * out) {
    __m256 acc[G];
    for (int g = 0; g < G; g++) acc[g] = _mm256_setzero_ps();
    for (int j = 0; j < D; j += 8) {
        const __m256 r = _mm256_loadu_ps(row + j);
        for (int g = 0; g < G; g++) acc[g] = _mm256_fmadd_ps(r, _mm256_loadu_ps(qg[g] + j), acc[g]);
    }
    for (int g = 0; g < G; g++) out[g] = hsum256(acc[g]);
}

// Two rows x G heads: each q load feeds two FMAs (register blocking, 2G accumulators).
template <int G>
KVFLASH_AVX2 inline void dots2_avx2(const float * row0, const float * row1, const float * const * qg, int D,
                                    float * out0, float * out1) {
    __m256 a0[G], a1[G];
    for (int g = 0; g < G; g++) a0[g] = a1[g] = _mm256_setzero_ps();
    for (int j = 0; j < D; j += 8) {
        const __m256 r0 = _mm256_loadu_ps(row0 + j), r1 = _mm256_loadu_ps(row1 + j);
        for (int g = 0; g < G; g++) {
            const __m256 qv = _mm256_loadu_ps(qg[g] + j);
            a0[g] = _mm256_fmadd_ps(r0, qv, a0[g]);
            a1[g] = _mm256_fmadd_ps(r1, qv, a1[g]);
        }
    }
    for (int g = 0; g < G; g++) { out0[g] = hsum256(a0[g]); out1[g] = hsum256(a1[g]); }
}

// exp(x) for x <= 0 (clamped at -80), relative error ~2e-7.
KVFLASH_AVX2 inline __m256 exp256_nonpos(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-80.0f));
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504089f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

KVFLASH_AVX2 inline double lse_avx2(const float * x, int n) {
    float mx = -INFINITY;
    for (int t = 0; t < n; t++) mx = std::max(mx, x[t]);
    const __m256 vm = _mm256_set1_ps(mx);
    __m256 acc = _mm256_setzero_ps();
    int t = 0;
    for (; t + 8 <= n; t += 8) acc = _mm256_add_ps(acc, exp256_nonpos(_mm256_sub_ps(_mm256_loadu_ps(x + t), vm)));
    double s = hsum256(acc);
    for (; t < n; t++) s += std::exp((double)x[t] - mx);
    return mx + std::log(s);
}

inline bool have_avx2() {
    static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    return ok;
}

// n (1..kMaxGroup) heads against row0, and against row1 too when non-null.
inline void dot_tile_avx2(const float * row0, const float * row1, const float * const * qg, int n, int D,
                          float * out0, float * out1) {
#define KVFLASH_CASE(N) case N: if (row1) dots2_avx2<N>(row0, row1, qg, D, out0, out1); else dots_avx2<N>(row0, qg, D, out0); break;
    switch (n) { KVFLASH_CASE(1) KVFLASH_CASE(2) KVFLASH_CASE(3) KVFLASH_CASE(4) KVFLASH_CASE(5) default: KVFLASH_CASE(6) }
#undef KVFLASH_CASE
}
#endif  // KVFLASH_FAST_X86

}  // namespace fastmass

// Per-chunk scores for several queries sharing each dequantized key row; same
// contract as kvflash_exact_mass_scores_multi. outs[i] is bit-identical to a
// single-query call (each head keeps its own accumulator however heads are
// tiled). planes, when non-null, gets one entry per query.
inline bool kvflash_exact_mass_scores_fast_multi(
    const KvFlashExactKeys & keys, const KvFlashQkDims & d, const std::vector<const float *> & qs,
    std::vector<std::vector<float>> & outs, int n_threads = 0, KvFlashExactTiming * tm = nullptr,
    std::vector<KvFlashMassPlanes> * planes = nullptr, bool simd = true) {
    const int nq = (int)qs.size();
    const int C = keys.n_chunks, ct = keys.chunk_tokens, D = d.head_dim;
    const int group = d.n_kv_heads > 0 ? d.n_q_heads / d.n_kv_heads : 0;
    const std::vector<float> zeros((size_t)std::max(C, 0), 0.0f);
    outs.assign((size_t)nq, zeros);
    if (planes) planes->assign((size_t)nq, KvFlashMassPlanes{zeros, zeros});
    if (nq == 0) return true;
    if (d.n_layers <= 0 || d.n_kv_heads <= 0 || d.n_q_heads <= 0 || d.n_q_heads % d.n_kv_heads) return false;   // no dropped query heads
    if (C <= 0 || ct <= 0 || D <= 0 || group <= 0 || !keys.seg) return false;
    if (std::find(qs.begin(), qs.end(), nullptr) != qs.end()) return false;
    const auto * tt = ggml_get_type_traits(keys.type);
    if (keys.type != GGML_TYPE_F32 && (!tt || !tt->to_float)) return false;
    const bool q4 = keys.type == GGML_TYPE_Q4_0 && D % fastmass::kQ4Block == 0;
#ifdef KVFLASH_FAST_X86
    const bool avx = simd && fastmass::have_avx2() && D % 8 == 0;
#else
    const bool avx = false;
#endif
    const int LH = d.n_layers * d.n_q_heads;
    const int N = nq * group;   // q heads dotted against each key row
    const float scale = 1.0f / std::sqrt((float)D);
    const double NEG = -INFINITY;
    std::vector<double> lse((size_t)C * nq * LH, NEG);   // [(c*nq + i)*LH + l*Hq + hq]; absent chunks stay -inf

    if (n_threads <= 0) n_threads = (int)std::max(1u, std::thread::hardware_concurrency());
    n_threads = std::min(n_threads, C);
    std::atomic<int> next{0};
    auto worker = [&]() {
        std::vector<float> row((size_t)D), row2((size_t)D), logit((size_t)N * ct);
        std::vector<const float *> qg((size_t)N);
        int64_t t_deq = 0, t_dot = 0, t_exp = 0;
        using clk = std::chrono::steady_clock;
        auto ns = [](clk::time_point a, clk::time_point b) { return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count(); };
        for (int c; (c = next.fetch_add(1)) < C;) {
            bool absent = false;
            for (int l = 0; l < d.n_layers && !absent; l++) {
                for (int hk = 0; hk < d.n_kv_heads && !absent; hk++) {
                    const uint8_t * p = keys.seg(c, l, hk);
                    if (!p) { absent = true; break; }
                    for (int h = 0; h < N; h++)
                        qg[(size_t)h] = qs[(size_t)(h / group)] + ((size_t)l * d.n_q_heads + hk * group + h % group) * D;
                    const size_t row_bytes = q4 ? (size_t)(D / 32) * 18 : (keys.type == GGML_TYPE_F32 ? (size_t)D * 4 : (size_t)ggml_row_size(keys.type, D));
                    auto load_row = [&](int t, float * dst) -> const float * {
                        const uint8_t * src = p + (size_t)t * row_bytes;
                        if (keys.type == GGML_TYPE_F32) return (const float *)src;
                        if (q4) {
#ifdef KVFLASH_FAST_X86
                            if (avx) fastmass::dequant_row_q4_0_avx2(src, dst, D); else
#endif
                            fastmass::dequant_row_q4_0_scalar(src, dst, D);
                        } else tt->to_float(src, dst, D);
                        return dst;
                    };
                    for (int t = 0; t < ct;) {
                        const bool pair = avx && t + 1 < ct;
                        const auto a = tm ? clk::now() : clk::time_point();
                        const float * kr = load_row(t, row.data());
                        const float * kr1 = pair ? load_row(t + 1, row2.data()) : nullptr;
                        const auto b = tm ? clk::now() : clk::time_point();
                        for (int g0 = 0; g0 < N; g0 += fastmass::kMaxGroup) {
                            const int n = std::min(fastmass::kMaxGroup, N - g0);
                            float r[fastmass::kMaxGroup], r1[fastmass::kMaxGroup];
#ifdef KVFLASH_FAST_X86
                            if (avx) fastmass::dot_tile_avx2(kr, kr1, qg.data() + g0, n, D, r, r1); else
#endif
                            fastmass::dots_scalar(kr, qg.data() + g0, n, D, r);
                            for (int g = 0; g < n; g++) {
                                logit[(size_t)(g0 + g) * ct + t] = r[g] * scale;
                                if (pair) logit[(size_t)(g0 + g) * ct + t + 1] = r1[g] * scale;
                            }
                        }
                        if (tm) { const auto e = clk::now(); t_deq += ns(a, b); t_dot += ns(b, e); }
                        t += pair ? 2 : 1;
                    }
                    const auto e0 = tm ? clk::now() : clk::time_point();
                    for (int h = 0; h < N; h++) {
                        const float * x = &logit[(size_t)h * ct];
                        double & dst = lse[((size_t)c * nq + h / group) * LH + (size_t)l * d.n_q_heads + hk * group + h % group];
#ifdef KVFLASH_FAST_X86
                        dst = avx ? fastmass::lse_avx2(x, ct) : fastmass::lse_scalar(x, ct);
#else
                        dst = fastmass::lse_scalar(x, ct);
#endif
                    }
                    if (tm) t_exp += ns(e0, clk::now());
                }
            }
            if (absent) std::fill_n(lse.begin() + (size_t)c * nq * LH, (size_t)nq * LH, NEG);
        }
        if (tm) { tm->dequant_ns += t_deq; tm->dot_ns += t_dot; tm->exp_ns += t_exp; }
    };
    run_workers(n_threads, [&](int) { worker(); });

    // Cross-chunk softmax per (query, layer, head), double precision, as the reference.
    constexpr int K = kExactTopPlanes;
    std::vector<float> top;   // [c][K] largest plane masses of the current query
    for (int i = 0; i < nq; i++) {
        if (planes) top.assign((size_t)C * K, 0.0f);
        auto at = [&](int c, int x) { return lse[((size_t)c * nq + i) * LH + x]; };
        for (int x = 0; x < LH; x++) {
            double mx = NEG;
            for (int c = 0; c < C; c++) {
                if (!(at(c, x) < std::numeric_limits<double>::infinity())) return false;   // NaN or +Inf (-inf marks an absent chunk): the scores would be NaN for every chunk
                mx = std::max(mx, at(c, x));
            }
            if (mx == NEG) continue;
            double Z = 0.0;
            for (int c = 0; c < C; c++) Z += std::exp(at(c, x) - mx);
            const double logZ = mx + std::log(Z);
            for (int c = 0; c < C; c++) {
                const double m = std::exp(at(c, x) - logZ);
                outs[(size_t)i][(size_t)c] += (float)(m / LH);
                if (!planes) continue;
                float & mxp = (*planes)[(size_t)i].max[(size_t)c];
                mxp = std::max(mxp, (float)m);
                float * lo = std::min_element(&top[(size_t)c * K], &top[(size_t)c * K] + K);
                *lo = std::max(*lo, (float)m);
            }
        }
        if (planes)
            for (int c = 0; c < C; c++) {
                double s = 0.0;
                for (int k = 0; k < K; k++) s += top[(size_t)c * K + k];
                (*planes)[(size_t)i].top[(size_t)c] = (float)(s / std::min(K, LH));
            }
    }
    return true;
}

// Selector v2 chunk score: the largest, over the queries, of the chunk's mean top-plane mass.
inline std::vector<float> kvflash_select_scores(const std::vector<KvFlashMassPlanes> & planes) {
    std::vector<float> s = planes.empty() ? std::vector<float>() : planes[0].top;
    for (const auto & p : planes)
        for (size_t c = 0; c < s.size(); c++) s[c] = std::max(s[c], p.top[c]);
    return s;
}

}  // namespace luce::common
