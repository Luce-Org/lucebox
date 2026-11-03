// Synthetic selector-v2 scorer fixture, used by test_kvflash_gpu_mass: a pool of
// quantized keys in the pager's layout (per layer [kv head][pool token][row]), the evicted chunks on a host store
// (pager layout: chunk -> [layer][kv head][chunk_tokens rows]), and N queries. ~n_hot planted chunks of graded
// strength (hot chunk j = 10 + 7j in kv plane (j % L, j % HK), rows 0-7 along v_j; the queries n % 8 == j % 8 carry
// v_j on that group's q heads, a logit boost of 3 + 5 j / n_hot after the 1/sqrt(D) scale) give the ranking a real
// order. Pool blocks that hold no sealed chunk (free, or chunks >= sealed as in the server's first pass) are filled
// with loud junk keys of chunks sealed + b, so a scorer that fails to mask them is caught.
#pragma once

#include "ggml.h"
#include "kvflash_exact_mass.h"
#include "kvflash_qk.h"
#include "parallel_for.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

namespace luce::common {

struct SynthRng {   // xorshift, uniform with unit variance
    uint64_t s;
    float next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (float)((s >> 40) * (1.0 / (1 << 24))) * 3.4641016f - 1.7320508f; }
};

struct SynthCfg {
    int L = 16, HK = 4, G = 6, D = 256, CT = 64, NB = 256;   // layers, kv heads, q per kv head, head dim, chunk tokens, pool blocks
    ggml_type KT = GGML_TYPE_Q4_0;
    int sealed = 514, N = 64;
    int keep_newest = -1;   // resident = sinks 0-1 + this many newest chunks; -1 = NB - 4 (the engine's log), 0 = sinks only
    int tail = 191;         // chunks of the recent window; planted chunks live before sealed - tail
    int max_hot = 40;
    int junk_chunks = 0;    // pool blocks left free are filled with junk keys of chunks sealed + b (loud, must be masked)
    uint64_t seed = 42;
};

struct Synth {
    SynthCfg c;
    int HQ = 0, PL = 0, n_hot = 0, first_window_chunk = 0;
    size_t RB = 0, SEG = 0, CHUNK = 0, QS = 0;
    std::vector<int> blk;                                 // chunk -> pool block, -1 evicted
    std::vector<int> ev, ev_idx;                          // evicted chunks in staging order; chunk -> index or -1
    std::vector<std::vector<uint8_t>> pool_img;           // [layer] -> [HK][NB * CT][row] in the K type
    std::vector<uint8_t> host;                            // evicted chunks' K in the pager layout
    std::vector<float> qs;                                // [N][L][HQ][D]
    std::vector<std::vector<float>> v;

    int hot_of(int ch) const { return (ch >= 10 && (ch - 10) % 7 == 0 && (ch - 10) / 7 < n_hot) ? (ch - 10) / 7 : -1; }
    float boost(int j) const { return 3.0f + 5.0f * (float)j / (float)std::max(1, n_hot); }
    const uint8_t * host_seg(int ch, int l, int h) const {
        return host.data() + (size_t)ev_idx[(size_t)ch] * CHUNK + (size_t)l * c.HK * SEG + (size_t)h * SEG;
    }
    uint8_t * pool_seg(int b, int l, int h) { return pool_img[(size_t)l].data() + (size_t)h * c.NB * c.CT * RB + (size_t)b * SEG; }
    KvFlashQkDims dims() const {
        KvFlashQkDims d;
        d.n_layers = c.L; d.n_q_heads = HQ; d.n_kv_heads = c.HK; d.head_dim = c.D;
        return d;
    }
    std::vector<const float *> qptrs(int n = -1) const {
        std::vector<const float *> p;
        for (int i = 0; i < (n < 0 ? c.N : n); i++) p.push_back(qs.data() + (size_t)i * QS);
        return p;
    }

    void build(const SynthCfg & cfg) {
        c = cfg;
        HQ = c.HK * c.G; PL = c.L * HQ;
        RB = ggml_row_size(c.KT, c.D); SEG = (size_t)c.CT * RB; CHUNK = (size_t)c.L * c.HK * SEG; QS = (size_t)c.L * HQ * c.D;
        const int keep = c.keep_newest >= 0 ? c.keep_newest : c.NB - 4;
        std::vector<int> perm((size_t)c.NB);
        for (int b = 0; b < c.NB; b++) perm[(size_t)b] = b;
        std::shuffle(perm.begin(), perm.end(), std::mt19937_64(c.seed));
        blk.assign((size_t)c.sealed, -1);
        size_t nres = 0;
        for (int ch = 0; ch < c.sealed; ch++)
            if (nres < perm.size() && (ch < 2 || ch >= c.sealed - keep)) blk[(size_t)ch] = perm[nres++];
        ev.clear(); ev_idx.assign((size_t)c.sealed, -1);
        for (int ch = 0; ch < c.sealed; ch++) if (blk[(size_t)ch] < 0) { ev_idx[(size_t)ch] = (int)ev.size(); ev.push_back(ch); }
        first_window_chunk = c.sealed - c.tail;
        n_hot = std::max(0, std::min(c.max_hot, (first_window_chunk - 10) / 7));
        v.assign((size_t)n_hot, std::vector<float>((size_t)c.D));
        for (int j = 0; j < n_hot; j++) {
            SynthRng r{1234567ull + (uint64_t)j};
            double nn = 0;
            for (float & x : v[(size_t)j]) { x = r.next(); nn += x * x; }
            for (float & x : v[(size_t)j]) x /= (float)std::sqrt(nn);
        }
        pool_img.assign((size_t)c.L, std::vector<uint8_t>((size_t)c.HK * c.NB * c.CT * RB, 0));
        host.assign(std::max<size_t>(1, ev.size() * CHUNK), 0);
        std::vector<int> chunk_in_block((size_t)c.NB, -1);
        for (int ch = 0; ch < c.sealed; ch++) if (blk[(size_t)ch] >= 0) chunk_in_block[(size_t)blk[(size_t)ch]] = ch;
        // jobs: every sealed chunk, then (junk_chunks) the free blocks as chunk ids sealed + b
        const int n_jobs = c.sealed + (c.junk_chunks ? c.NB : 0);
        parallel_for(n_jobs, [&](int job) {
            const bool junk = job >= c.sealed;
            if (junk && chunk_in_block[(size_t)(job - c.sealed)] >= 0) return;
            const int ch = job;   // junk ids sealed + b never plant (hot_of needs ch < first_window_chunk)
            std::vector<float> k((size_t)c.CT * c.D);
            for (int l = 0; l < c.L; l++)
                for (int h = 0; h < c.HK; h++) {
                    SynthRng r{((uint64_t)ch * c.L + l) * c.HK + h + 77};
                    for (float & x : k) x = r.next() * (junk ? 3.0f : 1.0f);
                    const int j = junk ? -1 : hot_of(ch);
                    if (j >= 0 && l == j % c.L && h == j % c.HK)
                        for (int t = 0; t < std::min(8, c.CT); t++)
                            for (int d = 0; d < c.D; d++) k[(size_t)t * c.D + d] += 2.0f * boost(j) * v[(size_t)j][(size_t)d];
                    uint8_t * dst = junk ? pool_seg(job - c.sealed, l, h) : blk[(size_t)ch] >= 0 ? pool_seg(blk[(size_t)ch], l, h)
                                 : host.data() + (size_t)ev_idx[(size_t)ch] * CHUNK + (size_t)l * c.HK * SEG + (size_t)h * SEG;
                    ggml_quantize_chunk(c.KT, k.data(), dst, 0, c.CT, c.D, nullptr);
                }
        });
        qs.assign((size_t)c.N * QS, 0.0f);
        parallel_for(c.N, [&](int n) {
            SynthRng r{999ull + (uint64_t)n};
            float * q = qs.data() + (size_t)n * QS;
            for (size_t i = 0; i < QS; i++) q[i] = r.next();
            for (int j = 0; j < n_hot; j++) {
                if (n % 8 != j % 8) continue;
                const int l = j % c.L, h = j % c.HK;
                for (int g = 0; g < c.G; g++)
                    for (int d = 0; d < c.D; d++) q[((size_t)l * HQ + h * c.G + g) * c.D + d] += 8.0f * v[(size_t)j][(size_t)d];
            }
        });
    }
};


// ---- Reference scorer: the test oracle -------------------------------------------------------------------------
// Plain per-chunk logsumexp in double, one thread pool, no fusing. Production runs kvflash_exact_fast.h (AVX2) or
// kvflash_gpu_mass.h; this body exists to check them.
// Several queries share one dequantization pass. qs[i]: n_layers*n_q_heads*
// head_dim floats. outs[i][c] = score of chunk c for qs[i].
inline bool kvflash_exact_mass_scores_multi(
    const KvFlashExactKeys & keys, const KvFlashQkDims & d,
    const std::vector<const float *> & qs, std::vector<std::vector<float>> & outs,
    int n_threads = 0, KvFlashExactTiming * tm = nullptr) {
    const int nq = (int)qs.size();
    const int C = keys.n_chunks, ct = keys.chunk_tokens, D = d.head_dim;
    const int group = d.n_kv_heads > 0 ? d.n_q_heads / d.n_kv_heads : 0;
    outs.assign((size_t)nq, std::vector<float>((size_t)std::max(C, 0), 0.0f));
    if (nq == 0 || C <= 0 || ct <= 0 || D <= 0 || group <= 0 || !keys.seg) return nq == 0;
    const auto * tt = ggml_get_type_traits(keys.type);
    if (keys.type != GGML_TYPE_F32 && (!tt || !tt->to_float)) return false;

    const int LH = d.n_layers * d.n_q_heads;
    const float scale = 1.0f / std::sqrt((float)D);
    const double NEG = -INFINITY;
    // lse[(i*C + c)*LH + l*Hq + hq]; absent chunks stay -inf.
    std::vector<double> lse((size_t)nq * C * LH, NEG);

    if (n_threads <= 0) n_threads = (int)std::max(1u, std::thread::hardware_concurrency());
    n_threads = std::min(n_threads, C);
    std::atomic<int> next{0};
    auto worker = [&]() {
        std::vector<float> rows((size_t)ct * D), logit((size_t)ct);
        for (int c; (c = next.fetch_add(1)) < C;) {
            bool absent = false;
            for (int l = 0; l < d.n_layers && !absent; l++) {
                for (int hk = 0; hk < d.n_kv_heads && !absent; hk++) {
                    const uint8_t * p = keys.seg(c, l, hk);
                    if (!p) { absent = true; break; }
                    const float * kr;
                    const auto td0 = std::chrono::steady_clock::now();
                    if (keys.type == GGML_TYPE_F32) kr = (const float *)p;
                    else { tt->to_float(p, rows.data(), (int64_t)ct * D); kr = rows.data(); }
                    if (tm) tm->dequant_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - td0).count();
                    for (int i = 0; i < nq; i++) {
                        for (int g = 0; g < group; g++) {
                            const int hq = hk * group + g;
                            const float * q = qs[(size_t)i] + ((size_t)l * d.n_q_heads + hq) * D;
                            const auto tq0 = std::chrono::steady_clock::now();
                            float mx = -INFINITY;
                            for (int t = 0; t < ct; t++) {
                                const float * k = kr + (size_t)t * D;
                                float dot = 0.0f;
                                for (int j = 0; j < D; j++) dot += q[j] * k[j];
                                logit[(size_t)t] = dot * scale;
                                mx = std::max(mx, logit[(size_t)t]);
                            }
                            const auto tq1 = std::chrono::steady_clock::now();
                            double s = 0.0;
                            for (int t = 0; t < ct; t++) s += std::exp((double)logit[(size_t)t] - mx);
                            if (tm) {
                                const auto tq2 = std::chrono::steady_clock::now();
                                tm->dot_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tq1 - tq0).count();
                                tm->exp_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tq2 - tq1).count();
                            }
                            lse[((size_t)i * C + c) * LH + (size_t)l * d.n_q_heads + hq] = mx + std::log(s);
                        }
                    }
                }
            }
            if (absent)
                for (int i = 0; i < nq; i++)
                    for (int x = 0; x < LH; x++) lse[((size_t)i * C + c) * LH + x] = NEG;
        }
    };
    run_workers(n_threads, [&](int) { worker(); });

    for (int i = 0; i < nq; i++) {
        for (int x = 0; x < LH; x++) {
            double mx = NEG;
            for (int c = 0; c < C; c++) mx = std::max(mx, lse[((size_t)i * C + c) * LH + x]);
            if (mx == NEG) continue;
            double Z = 0.0;
            for (int c = 0; c < C; c++) Z += std::exp(lse[((size_t)i * C + c) * LH + x] - mx);
            const double logZ = mx + std::log(Z);
            for (int c = 0; c < C; c++)
                outs[(size_t)i][(size_t)c] += (float)(std::exp(lse[((size_t)i * C + c) * LH + x] - logZ) / LH);
        }
    }
    return true;
}

inline bool kvflash_exact_mass_scores(const KvFlashExactKeys & keys, const KvFlashQkDims & d,
                                      const float * q, std::vector<float> & out,
                                      int n_threads = 0) {
    std::vector<std::vector<float>> o;
    const bool ok = kvflash_exact_mass_scores_multi(keys, d, {q}, o, n_threads);
    out = o.empty() ? std::vector<float>() : std::move(o[0]);
    return ok;
}

}  // namespace luce::common
