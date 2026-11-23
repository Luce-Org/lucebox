// Adjacent-segment mean-key cosine host implementation + GPU dispatch.
// See pflash_seg_kcos.h.

#include "pflash_seg_kcos.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace luce::common {

void pflash_adjacent_segment_key_cosine_cpu(
        const float * keys,
        int n_tokens,
        int k_dim,
        const std::vector<PFlashTokenSpan> & segments,
        std::vector<float> & out_kcos,
        std::vector<float> * out_seg_keys,
        const float * token_mass,
        std::vector<float> * out_seg_katt) {
    const size_t n = segments.size();
    out_kcos.assign(n, 0.0f);
    if (out_seg_keys) out_seg_keys->assign(n * (size_t) std::max(0, k_dim), 0.0f);
    if (out_seg_katt) out_seg_katt->assign(n * (size_t) std::max(0, k_dim), 0.0f);
    if (!keys || n_tokens <= 0 || k_dim <= 0 || n == 0) return;

    const bool do_katt = (token_mass != nullptr && out_seg_katt != nullptr);
    std::vector<float> prev((size_t) k_dim, 0.0f);
    std::vector<float> curr((size_t) k_dim, 0.0f);
    std::vector<float> curr_att(do_katt ? (size_t) k_dim : 0u, 0.0f);
    bool have_prev = false;

    for (size_t i = 0; i < n; ++i) {
        const int begin = std::max(0, segments[i].begin);
        const int end   = std::min(n_tokens, segments[i].end);
        const int len   = end - begin;
        if (len <= 0) {
            have_prev = false;
            continue;
        }
        std::fill(curr.begin(), curr.end(), 0.0f);
        double sum_m = 0.0;
        if (do_katt) {
            std::fill(curr_att.begin(), curr_att.end(), 0.0f);
            for (int t = begin; t < end; ++t) {
                const float mt = token_mass[t];
                if (mt > 0.0f) sum_m += (double) mt;
            }
        }
        const bool use_att = do_katt && sum_m > 1e-18 && std::isfinite(sum_m);
        const float inv_sum_m = use_att ? (float) (1.0 / sum_m) : 0.0f;
        for (int t = begin; t < end; ++t) {
            const float * row = keys + (size_t) t * (size_t) k_dim;
            const float wt = (use_att && token_mass[t] > 0.0f) ? (token_mass[t] * inv_sum_m) : 0.0f;
            for (int d = 0; d < k_dim; ++d) {
                curr[(size_t) d] += row[d];
                if (wt > 0.0f) {
                    curr_att[(size_t) d] += wt * row[d];
                }
            }
        }
        const float inv_len = 1.0f / (float) len;
        double norm_sq = 0.0;
        for (int d = 0; d < k_dim; ++d) {
            curr[(size_t) d] *= inv_len;
            norm_sq += (double) curr[(size_t) d] * (double) curr[(size_t) d];
        }
        bool have_kbar = false;
        if (norm_sq > 1e-16 && std::isfinite(norm_sq)) {
            const float inv_norm = (float) (1.0 / std::sqrt(norm_sq));
            for (int d = 0; d < k_dim; ++d) curr[(size_t) d] *= inv_norm;
            have_kbar = true;
            if (out_seg_keys) {
                std::copy(curr.begin(), curr.end(),
                          out_seg_keys->begin() + i * (size_t) k_dim);
            }
            if (have_prev) {
                double dot = 0.0;
                for (int d = 0; d < k_dim; ++d) {
                    dot += (double) prev[(size_t) d] * (double) curr[(size_t) d];
                }
                if (std::isfinite(dot) && dot > 0.0) {
                    out_kcos[i] = (float) std::min(1.0, dot);
                }
            }
        }
        if (do_katt) {
            double norm_sq_att = 0.0;
            if (use_att) {
                for (int d = 0; d < k_dim; ++d) {
                    norm_sq_att += (double) curr_att[(size_t) d] * (double) curr_att[(size_t) d];
                }
            }
            if (use_att && norm_sq_att > 1e-16 && std::isfinite(norm_sq_att)) {
                const float inv_norm_att = (float) (1.0 / std::sqrt(norm_sq_att));
                for (int d = 0; d < k_dim; ++d) curr_att[(size_t) d] *= inv_norm_att;
                std::copy(curr_att.begin(), curr_att.end(),
                          out_seg_katt->begin() + i * (size_t) k_dim);
            } else if (have_kbar) {
                std::copy(curr.begin(), curr.end(),
                          out_seg_katt->begin() + i * (size_t) k_dim);
            }
        }
        if (have_kbar) {
            prev.swap(curr);
            have_prev = true;
        } else {
            have_prev = false;
        }
    }
}

bool pflash_adjacent_segment_key_cosine(
        ggml_backend_t backend,
        ggml_tensor * keys,
        const std::vector<PFlashTokenSpan> & segments,
        std::vector<float> & out_kcos,
        std::vector<float> * out_seg_keys,
        const float * token_mass,
        std::vector<float> * out_seg_katt) {
    const int n = (int) segments.size();
    out_kcos.assign((size_t) std::max(0, n), 0.0f);
    if (out_seg_keys) out_seg_keys->clear();
    if (out_seg_katt) out_seg_katt->clear();
    if (n == 0) return true;
    if (!backend || !keys || keys->type != GGML_TYPE_F32) return false;
    const int D  = (int) keys->ne[0];
    const int Hk = (int) keys->ne[1];
    const int cap = (int) keys->ne[2];
    if (D <= 0 || Hk <= 0 || cap <= 0) return false;
    const int k_dim = D * Hk;
    const bool want_katt = (token_mass != nullptr && out_seg_katt != nullptr);
    if (out_seg_keys) out_seg_keys->assign((size_t) n * (size_t) k_dim, 0.0f);
    if (out_seg_katt) out_seg_katt->assign((size_t) n * (size_t) k_dim, 0.0f);
    if (n < 2 && !out_seg_keys && !want_katt) return true;
    const bool contiguous =
        keys->nb[0] == sizeof(float) &&
        keys->nb[1] == (size_t) D * sizeof(float) &&
        keys->nb[2] == (size_t) k_dim * sizeof(float);
    if (!contiguous) return false;

    int max_end = 0;
    for (const auto & seg : segments) {
        if (seg.begin < 0 || seg.end < seg.begin || seg.end > cap) return false;
        max_end = std::max(max_end, seg.end);
    }
    if (max_end == 0) return true;

#if defined(LUCE_HAVE_PFLASH_SEG_KCOS)
    if (ggml_backend_is_cuda(backend) && keys->data) {
        ggml_init_params ip{};
        ip.mem_size = (size_t) 10 * ggml_tensor_overhead() + 4096;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) return false;
        ggml_tensor * bounds_t   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) n * 2);
        ggml_tensor * seg_keys_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_dim, n);
        ggml_tensor * kcos_t     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_tensor * mass_t     = want_katt ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, max_end) : nullptr;
        ggml_tensor * seg_katt_t = want_katt ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_dim, n) : nullptr;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            ggml_free(ctx);
            return false;
        }
        std::vector<int32_t> bounds_h((size_t) n * 2);
        for (int i = 0; i < n; ++i) {
            bounds_h[(size_t) 2 * i]     = segments[(size_t) i].begin;
            bounds_h[(size_t) 2 * i + 1] = segments[(size_t) i].end;
        }
        ggml_backend_tensor_set(bounds_t, bounds_h.data(), 0,
                                bounds_h.size() * sizeof(int32_t));
        if (want_katt) {
            ggml_backend_tensor_set(mass_t, token_mass, 0,
                                    (size_t) max_end * sizeof(float));
        }
        ggml_backend_synchronize(backend);
        void * stream = ggml_backend_cuda_enter_stream(backend);
        const bool ok = pflash_adjacent_segment_key_cosine_device(
            static_cast<const float *>(keys->data),
            static_cast<const int32_t *>(bounds_t->data),
            static_cast<float *>(seg_keys_t->data),
            static_cast<float *>(kcos_t->data),
            n, k_dim, stream,
            want_katt ? static_cast<const float *>(mass_t->data) : nullptr,
            want_katt ? static_cast<float *>(seg_katt_t->data) : nullptr);
        if (ok) {
            ggml_backend_tensor_get(kcos_t, out_kcos.data(), 0,
                                    (size_t) n * sizeof(float));
            if (out_seg_keys) {
                ggml_backend_tensor_get(seg_keys_t, out_seg_keys->data(), 0,
                                        (size_t) n * (size_t) k_dim * sizeof(float));
            }
            if (want_katt) {
                ggml_backend_tensor_get(seg_katt_t, out_seg_katt->data(), 0,
                                        (size_t) n * (size_t) k_dim * sizeof(float));
            }
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        if (ok) return true;
        out_kcos.assign((size_t) n, 0.0f);
        if (out_seg_keys) out_seg_keys->assign((size_t) n * (size_t) k_dim, 0.0f);
        if (out_seg_katt) out_seg_katt->assign((size_t) n * (size_t) k_dim, 0.0f);
    }
#endif

    // Host path: a CPU backend, or the device kernel failed.
    std::vector<float> host_keys((size_t) max_end * (size_t) k_dim);
    ggml_backend_tensor_get(keys, host_keys.data(), 0,
                            host_keys.size() * sizeof(float));
    pflash_adjacent_segment_key_cosine_cpu(
        host_keys.data(), max_end, k_dim, segments, out_kcos, out_seg_keys,
        want_katt ? token_mass : nullptr, want_katt ? out_seg_katt : nullptr);
    return true;
}

} // namespace luce::common
