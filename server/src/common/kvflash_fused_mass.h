// CPU reference of the fused exact chunk-mass math (Amendment 10). One pass over K, no stored logits:
//   partials[plane][chunk] = { m = max over the chunk's tokens of the scaled logit, s = sum over the tokens of exp(logit - m) }
//   mass_c = s_c * exp(m_c - M) / sum_c' s_c' * exp(m_c' - M), M = max m (this is exp(lse_c - lse) without forming the logs)
// then per chunk: mean over planes, mean of the kExactTopPlanes largest plane masses, max over planes.
// Same quantity as kvflash_exact_mass_scores_multi (rounding differs); the device kernel is checked against this.
// A plane is (layer, query head); an absent chunk has m = -inf, s = 0 and takes no mass.
//
// Roofline at 217K (n_q = 128, sealed 3382 chunks of 64, L = 16, Hq = 24, HK = 4, G = Hq/HK = 6, D = 256, q8_0 K):
//   planes per query 384; tokens T = 216,448; logits per call = T * 384 * 128 = 1.064e10.
//   flops (q.k, 2*D per logit)            5.45e12, identical for graph and kernel
//   exps                                  1.06e10 (kernel, once per logit)
//   K bytes, one read                     T * 272 B * HK * L = 3.77e9
//   current graph: K re-read per query batch (19 batches) 7.2e10; f32 logits 4.26e10 B per pass x about 8 passes
//     (mul_mat write, concat r+w, bias add r+w, softmax r+w, sum_rows r) 3.4e11; total about 4.1e11 B
//   fused kernel: K 3.77e9 + partials 128*384*3382*2*4 = 1.33e9 written and read by the reduce = 6.4e9 B
// so the kernel is compute bound (5.45e12 flops) and needs >= 22 TFLOP/s sustained to meet 250 ms.

#pragma once

#include "kvflash_exact_fast.h"
#include "kvflash_exact_mass.h"

namespace luce::common {

struct KvFlashFusedPartials {
    int nq = 0, planes = 0, chunks = 0;
    std::vector<float> v;   // [((query * planes + plane) * chunks + chunk) * 2 + {0: max, 1: sum exp}]
    float & m(int i, int x, int c) { return v[(((size_t)i * planes + x) * chunks + c) * 2]; }
    float & s(int i, int x, int c) { return v[(((size_t)i * planes + x) * chunks + c) * 2 + 1]; }
    float m(int i, int x, int c) const { return v[(((size_t)i * planes + x) * chunks + c) * 2]; }
    float s(int i, int x, int c) const { return v[(((size_t)i * planes + x) * chunks + c) * 2 + 1]; }
};

inline bool kvflash_fused_partials_ref(const KvFlashExactKeys & keys, const KvFlashQkDims & d,
                                       const std::vector<const float *> & qs, KvFlashFusedPartials & out) {
    const int nq = (int)qs.size(), C = keys.n_chunks, ct = keys.chunk_tokens, D = d.head_dim;
    const int group = d.n_kv_heads > 0 ? d.n_q_heads / d.n_kv_heads : 0;
    if (nq == 0 || C <= 0 || ct <= 0 || D <= 0 || group <= 0 || !keys.seg) return false;
    const auto * tt = ggml_get_type_traits(keys.type);
    if (keys.type != GGML_TYPE_F32 && (!tt || !tt->to_float)) return false;
    out.nq = nq; out.planes = d.n_layers * d.n_q_heads; out.chunks = C;
    out.v.assign((size_t)nq * out.planes * C * 2, 0.0f);
    const float scale = 1.0f / std::sqrt((float)D);
    std::vector<float> rows((size_t)ct * D);
    for (int c = 0; c < C; c++)
        for (int l = 0; l < d.n_layers; l++)
            for (int hk = 0; hk < d.n_kv_heads; hk++) {
                const uint8_t * p = keys.seg(c, l, hk);
                const float * kr = nullptr;
                if (p) {
                    if (keys.type == GGML_TYPE_F32) kr = (const float *)p;
                    else { tt->to_float(p, rows.data(), (int64_t)ct * D); kr = rows.data(); }
                }
                for (int i = 0; i < nq; i++)
                    for (int g = 0; g < group; g++) {
                        const int hq = hk * group + g, x = l * d.n_q_heads + hq;
                        if (!kr) { out.m(i, x, c) = -INFINITY; out.s(i, x, c) = 0.0f; continue; }
                        const float * q = qs[(size_t)i] + (size_t)x * D;
                        std::vector<float> lg((size_t)ct);
                        float mx = -INFINITY;
                        for (int t = 0; t < ct; t++) {
                            float dot = 0.0f;
                            for (int j = 0; j < D; j++) dot += q[j] * kr[(size_t)t * D + j];
                            lg[(size_t)t] = dot * scale;
                            mx = std::max(mx, lg[(size_t)t]);
                        }
                        float sum = 0.0f;
                        for (int t = 0; t < ct; t++) sum += std::exp(lg[(size_t)t] - mx);
                        out.m(i, x, c) = mx; out.s(i, x, c) = sum;
                    }
            }
    return true;
}

inline void kvflash_fused_reduce_ref(const KvFlashFusedPartials & p, const KvFlashQkDims &,
                                     std::vector<std::vector<float>> & outs, std::vector<KvFlashMassPlanes> & planes) {
    constexpr int K = kExactTopPlanes;
    const int C = p.chunks, LH = p.planes;
    const std::vector<float> zeros((size_t)C, 0.0f);
    outs.assign((size_t)p.nq, zeros);
    planes.assign((size_t)p.nq, KvFlashMassPlanes{zeros, zeros});
    std::vector<float> lse((size_t)C), top((size_t)C * K);
    for (int i = 0; i < p.nq; i++) {
        std::fill(top.begin(), top.end(), 0.0f);
        for (int x = 0; x < LH; x++) {
            // Rescale by the largest local max instead of forming lse = m + log(s): the differences m - mx stay small, so no
            // precision is lost to a large lse (the same trick as the online softmax).
            float mx = -INFINITY;
            for (int c = 0; c < C; c++) if (p.s(i, x, c) > 0.0f) mx = std::max(mx, p.m(i, x, c));
            if (mx == -INFINITY) continue;
            float Z = 0.0f;
            for (int c = 0; c < C; c++) {
                lse[(size_t)c] = p.s(i, x, c) > 0.0f ? p.s(i, x, c) * std::exp(p.m(i, x, c) - mx) : 0.0f;
                Z += lse[(size_t)c];
            }
            for (int c = 0; c < C; c++) {
                const float m = lse[(size_t)c] / Z;
                outs[(size_t)i][(size_t)c] += m / (float)LH;
                float & mxp = planes[(size_t)i].max[(size_t)c];
                mxp = std::max(mxp, m);
                float * lo = std::min_element(&top[(size_t)c * K], &top[(size_t)c * K] + K);
                *lo = std::max(*lo, m);
            }
        }
        for (int c = 0; c < C; c++) {
            double s = 0.0;
            for (int k = 0; k < K; k++) s += top[(size_t)c * K + k];
            planes[(size_t)i].top[(size_t)c] = (float)(s / std::min(K, LH));
        }
    }
}

}  // namespace luce::common
