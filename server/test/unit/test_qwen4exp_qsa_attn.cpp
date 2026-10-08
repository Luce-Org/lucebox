// QSA selected attention at Qwen3.8-Flash-Next's shape (head 256, 12 query heads per KV head, block ratio 4)
// against a double-precision reference over exactly the selected keys: the packed prefill kernel (T >= 128) and
// the per-query decode kernel (T < 128), fed by GGML_OP_QSA_DECODE_IDS as build_qsa_attn feeds them. Without a
// mask a sparse FLASH_ATTN_EXT only runs on the QSA kernels (the generic path asserts); with one, the generic
// path would attend to every key and miss the reference. Each case prints an output hash for comparing builds.
// Exit 77 when no CUDA/HIP device is available or the device has no QSA kernels.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

static constexpr int D = 256, HK = 2, HQ = 12 * HK, R = 4;

struct Case { int T, pos0, budget, ctx; bool mask; };

// build_qsa_attn's packed layouts: keys [16 dims, 4 keys, 16 dim chunks, block], values [4 keys, 256 dims, block].
static ggml_tensor * pack_keys(ggml_context * c, ggml_tensor * keys) {
    ggml_tensor * blocks = ggml_reshape_4d(c, keys, 16, 16, 4, keys->ne[1] / 4 * keys->ne[2]);
    return ggml_cont(c, ggml_permute(c, blocks, 0, 2, 1, 3));
}

static ggml_tensor * pack_values(ggml_context * c, ggml_tensor * values) {
    ggml_tensor * blocks = ggml_reshape_3d(c, values, 256, 4, values->ne[1] / 4 * values->ne[2]);
    return ggml_cont(c, ggml_permute(c, blocks, 1, 0, 2, 3));
}

static bool run(ggml_backend_t be, const Case & cs, std::mt19937 & rng) {
    const int T = cs.T, kv_len = cs.pos0 + T;
    const bool packed = T >= 128;
    const int nk = packed ? (kv_len + 3) / 4 * 4 : kv_len;   // the graph pads K/V to whole 4-key blocks
    const int n_after = kv_len / R;
    const bool dense = n_after <= cs.budget;                 // all complete blocks fit: enumerate them
    const int budget = dense ? std::max(1, n_after) : cs.budget;
    const float scale = 1.0f / 16.0f;

    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> q((size_t) D * T * HQ);
    for (float & x : q) x = 4.0f * u(rng);
    std::vector<ggml_fp16_t> k((size_t) D * cs.ctx * HK), v(k.size());
    for (ggml_fp16_t & x : k) x = ggml_fp32_to_fp16(u(rng));
    for (ggml_fp16_t & x : v) x = ggml_fp32_to_fp16(u(rng));
    // Selected rows come from top-k in arbitrary order and may name blocks the query cannot see yet.
    std::vector<int32_t> blocks((size_t) budget * T), all(std::max(1, n_after));
    std::iota(all.begin(), all.end(), 0);
    for (int t = 0; t < T; ++t) {
        if (!dense) std::shuffle(all.begin(), all.end(), rng);
        std::copy_n(all.begin(), budget, blocks.begin() + (size_t) t * budget);
    }
    std::vector<int32_t> pos(T);
    std::iota(pos.begin(), pos.end(), cs.pos0);
    std::vector<ggml_fp16_t> mask(cs.mask ? (size_t) nk * T : 0);
    for (ggml_fp16_t & x : mask) x = ggml_fp32_to_fp16(u(rng) < -0.8f ? -INFINITY : 2.0f * u(rng));

    ggml_context * c = ggml_init({4 * 1024 * 1024, nullptr, true});
    ggml_tensor * tq = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, T, HQ);
    ggml_tensor * kc = ggml_new_tensor_3d(c, GGML_TYPE_F16, D, cs.ctx, HK);
    ggml_tensor * vc = ggml_new_tensor_3d(c, GGML_TYPE_F16, D, cs.ctx, HK);
    ggml_tensor * tb = ggml_new_tensor_2d(c, GGML_TYPE_I32, budget, T);
    ggml_tensor * tp = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_tensor * tm = cs.mask ? ggml_new_tensor_2d(c, GGML_TYPE_F16, nk, T) : nullptr;
    // K/V are views of a longer cache, as in the graph: decode reads them through strides, prefill packs a copy.
    ggml_tensor * kf = ggml_view_3d(c, kc, D, nk, HK, kc->nb[1], kc->nb[2], 0);
    ggml_tensor * vf = ggml_view_3d(c, vc, D, nk, HK, vc->nb[1], vc->nb[2], 0);
    if (packed) { kf = ggml_cont(c, kf); vf = ggml_cont(c, vf); }
    ggml_tensor * ids = ggml_qsa_decode_ids(c, tb, tp, R);
    ggml_tensor * attn = ggml_flash_attn_ext(c, tq, kf, vf, tm, scale, 0.0f, 0.0f);
    attn->src[5] = ids;
    if (packed) {
        attn->src[6] = pack_keys(c, kf);
        attn->src[7] = pack_values(c, vf);
    }
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, attn);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, be);
    ggml_backend_tensor_set(tq, q.data(), 0, ggml_nbytes(tq));
    ggml_backend_tensor_set(kc, k.data(), 0, ggml_nbytes(kc));
    ggml_backend_tensor_set(vc, v.data(), 0, ggml_nbytes(vc));
    ggml_backend_tensor_set(tb, blocks.data(), 0, ggml_nbytes(tb));
    ggml_backend_tensor_set(tp, pos.data(), 0, ggml_nbytes(tp));
    if (tm) ggml_backend_tensor_set(tm, mask.data(), 0, ggml_nbytes(tm));
    const bool computed = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    const int n_ids = (int) ids->ne[0];
    std::vector<float> got((size_t) D * HQ * T);
    std::vector<int32_t> cells((size_t) n_ids * T);
    if (computed) {
        ggml_backend_tensor_get(attn, got.data(), 0, ggml_nbytes(attn));
        ggml_backend_tensor_get(ids, cells.data(), 0, ggml_nbytes(ids));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(c);
    if (!computed) return false;

    double max_err = 0.0;
    std::vector<double> s(n_ids), o(D);
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < HQ; ++h) {
            const int kvh = h / 12;
            double m = -INFINITY;
            for (int j = 0; j < n_ids; ++j) {
                const int key = cells[(size_t) t * n_ids + j];
                const double bias = key >= 0 && key < nk && cs.mask ? ggml_fp16_to_fp32(mask[(size_t) t * nk + key]) : 0.0;
                s[j] = -INFINITY;
                if (key < 0 || key >= nk || bias == -INFINITY) continue;
                double dot = 0.0;
                for (int d = 0; d < D; ++d) {
                    dot += (double) q[((size_t) h * T + t) * D + d] * ggml_fp16_to_fp32(k[((size_t) kvh * cs.ctx + key) * D + d]);
                }
                s[j] = scale * dot + bias;
                m = std::max(m, s[j]);
            }
            double l = 0.0;
            std::fill(o.begin(), o.end(), 0.0);
            for (int j = 0; j < n_ids && m > -INFINITY; ++j) {
                if (s[j] == -INFINITY) continue;
                const double p = std::exp(s[j] - m);
                const int key = cells[(size_t) t * n_ids + j];
                l += p;
                for (int d = 0; d < D; ++d) o[d] += p * ggml_fp16_to_fp32(v[((size_t) kvh * cs.ctx + key) * D + d]);
            }
            for (int d = 0; d < D; ++d) {
                const double want = l > 0.0 ? o[d] / l : 0.0;
                max_err = std::max(max_err, std::fabs(got[((size_t) t * HQ + h) * D + d] - want));
            }
        }
    }
    uint64_t hash = 1469598103934665603ull;
    for (float x : got) {
        uint32_t bits;
        std::memcpy(&bits, &x, 4);
        for (int b = 0; b < 4; ++b) hash = (hash ^ ((bits >> (8 * b)) & 0xFFu)) * 1099511628211ull;
    }
    const bool pass = std::isfinite(max_err) && max_err < 1e-2;
    std::printf("qsa-attn %-7s T=%-3d pos=%-4d budget=%-3d mask=%d max_abs_err=%.2e hash=%016llx %s\n",
                packed ? "prefill" : "decode", T, cs.pos0, budget, cs.mask ? 1 : 0, max_err,
                (unsigned long long) hash, pass ? "OK" : "FAIL");
    return pass;
}

int main() {
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) return 77;
    if (!ggml_backend_cuda_qsa_supported(gpu)) {
        std::printf("qsa-attn: device has no QSA kernels, skipped\n");
        ggml_backend_free(gpu);
        return 77;
    }
    std::mt19937 rng(20261007);
    bool ok = true;
    int cases = 0;
    // Prefill: dense-regime enumeration, group and block tails, the budget crossing, past the budget, masked.
    // Decode: short dense-regime rows, past the budget, one row, the largest decode batch, masked.
    for (const Case & cs : std::vector<Case>{
            {128, 0, 512, 4096, false}, {130, 1001, 512, 4096, false}, {300, 1900, 512, 4096, false},
            {257, 3000, 512, 8192, false}, {256, 6001, 512, 8192, true},
            {5, 10, 512, 4096, false}, {1, 3000, 512, 4096, false}, {2, 4095, 512, 8192, false},
            {127, 2100, 512, 4096, false}, {8, 6000, 512, 8192, true}}) {
        ok &= run(gpu, cs, rng);
        ++cases;
    }
    ggml_backend_free(gpu);
    std::printf("qsa-attn: %d cases, %s\n", cases, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
