// Unit test for all custom GGML_OP_INKLING2_OP GPU kernels against exact CPU FP32 reference.
// Run on any CUDA/HIP GPU (pass GPU index via LUCE_TEST_GPU or argv[1], default 0).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "inkling2_internal.h"
#include "inkling2_2dsd_executor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <unordered_map>
#include <vector>

namespace {

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float mx = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        mx = std::max(mx, std::fabs(a[i] - b[i]));
    }
    return mx;
}

static bool test_situ(ggml_backend_t backend) {
    const int n_ff = 1152, T = 4, K = 8;
    const float beta = 4.0f, lbeta = 25.0f;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-6.0f, 6.0f);

    std::vector<float> in((size_t) 2 * n_ff * K * T);
    for (float & v : in) v = dist(rng);

    std::vector<float> ref((size_t) n_ff * K * T);
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < K; ++k) {
            const float * gp = in.data() + ((size_t) t * K + k) * (2 * n_ff);
            const float * up = gp + n_ff;
            float * op = ref.data() + ((size_t) t * K + k) * n_ff;
            for (int i = 0; i < n_ff; ++i) {
                float g = gp[i], u = up[i];
                float sg = beta * std::tanh(g / beta) / (1.0f + std::exp(-g));
                float su = lbeta * std::tanh(u / lbeta);
                op[i] = sg * su;
            }
        }
    }

    struct ggml_init_params gp = { 64 * ggml_tensor_overhead() + ggml_graph_overhead() + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * t_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2 * n_ff, K, T);
    ggml_set_input(t_in);
    ggml_tensor * t_out = ggml_inkling2_situ(ctx, t_in, beta, lbeta);
    ggml_set_output(t_out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_out);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(t_in, in.data(), 0, in.size() * sizeof(float));
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> got(ref.size());
    ggml_backend_tensor_get(t_out, got.data(), 0, got.size() * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(ctx);

    const float err = max_abs_diff(got, ref);
    std::printf("[test_inkling2_ops] situ max_abs_err=%.3e\n", err);
    return err < 1e-4f;
}

static bool test_sconv(ggml_backend_t backend) {
    const int Hk = 8, D = 128, C = Hk * D, T = 5;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> x((size_t) C * T), w_st((size_t) 4 * C), dyn((size_t) 4 * Hk * T);
    std::vector<float> st0((size_t) 3 * C);
    for (float & v : x) v = dist(rng);
    for (float & v : w_st) v = dist(rng) * 0.2f;
    for (float & v : dyn) v = dist(rng) * 0.2f;
    for (float & v : st0) v = dist(rng);

    std::vector<float> ref((size_t) C * T);
    std::vector<float> ref_ring((size_t) 3 * C * T);
    for (int c = 0; c < C; ++c) {
        float h0 = st0[c * 3 + 0], h1 = st0[c * 3 + 1], h2 = st0[c * 3 + 2];
        const float * wc = &w_st[c * 4];
        const int head = c / D;
        for (int t = 0; t < T; ++t) {
            float cur = x[t * C + c];
            const float * dp = &dyn[t * (4 * Hk) + head * 4];
            float acc = wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * cur
                      + dp[0] * cur + dp[1] * h2 + dp[2] * h1 + dp[3] * h0;
            ref[t * C + c] = (cur + acc) / 3.0f;
            h0 = h1; h1 = h2; h2 = cur;
            ref_ring[(t * C + c) * 3 + 0] = h0;
            ref_ring[(t * C + c) * 3 + 1] = h1;
            ref_ring[(t * C + c) * 3 + 2] = h2;
        }
    }

    struct ggml_init_params gp = { 64 * ggml_tensor_overhead() + ggml_graph_overhead() + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * t_x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, T);
    ggml_tensor * t_w   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, C);
    ggml_tensor * t_st  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, C);
    ggml_tensor * t_rng = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, C, T);
    ggml_tensor * t_dyn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4 * Hk, T);
    ggml_set_input(t_x); ggml_set_input(t_w); ggml_set_input(t_st); ggml_set_input(t_rng); ggml_set_input(t_dyn);
    ggml_tensor * t_out = ggml_inkling2_sconv(ctx, t_x, t_w, t_st, t_rng, t_dyn, 0, D, 2);
    ggml_set_output(t_out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_out);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(t_x, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(t_w, w_st.data(), 0, w_st.size() * sizeof(float));
    ggml_backend_tensor_set(t_st, st0.data(), 0, st0.size() * sizeof(float));
    ggml_backend_tensor_set(t_dyn, dyn.data(), 0, dyn.size() * sizeof(float));
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> got(ref.size()), got_ring(ref_ring.size());
    ggml_backend_tensor_get(t_out, got.data(), 0, got.size() * sizeof(float));
    ggml_backend_tensor_get(t_rng, got_ring.data(), 0, got_ring.size() * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(ctx);

    const float err = max_abs_diff(got, ref);
    const float err_r = max_abs_diff(got_ring, ref_ring);
    std::printf("[test_inkling2_ops] sconv(dyn+ring) max_abs_err=%.3e ring_err=%.3e\n", err, err_r);
    return err < 1e-5f && err_r < 1e-6f;
}

static bool test_attn_rel(ggml_backend_t backend, bool use_q8, int T, int pos0) {
    const int Hq = 24, Hk = 8, D = 128, d_rel = 16, rel_ext = 64, kv_cap = 256;
    const int kv_len = pos0 + T;
    std::mt19937 rng(777 + T + (use_q8 ? 100 : 0));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Q((size_t) D * Hq * T);
    std::vector<float> K_all((size_t) D * kv_len * Hk);
    std::vector<float> V_all((size_t) D * kv_len * Hk);
    std::vector<float> rel_bias((size_t) d_rel * rel_ext);
    std::vector<float> R((size_t) d_rel * Hq * T);
    std::vector<float> Sink((size_t) Hq * T);
    std::vector<float> Gate((size_t) D * Hq * T);
    for (float & v : Q) v = dist(rng);
    for (float & v : K_all) v = dist(rng);
    for (float & v : V_all) v = dist(rng);
    for (float & v : rel_bias) v = dist(rng) * 0.2f;
    for (float & v : R) v = dist(rng);
    for (float & v : Sink) v = dist(rng) * 0.5f;
    for (float & v : Gate) v = dist(rng);

    // Exact CPU reference (with zero-sink virtual key and gated output)
    std::vector<float> ref((size_t) D * Hq * T, 0.0f);
    const float scale = 1.0f / std::sqrt((float) D);
    for (int t = 0; t < T; ++t) {
        const int q_pos = pos0 + t;
        for (int h = 0; h < Hq; ++h) {
            const int kh = h / (Hq / Hk);
            const float * qp = &Q[((size_t) t * Hq + h) * D];
            const float * rp = &R[((size_t) t * Hq + h) * d_rel];
            const float sv = Sink[(size_t) t * Hq + h];

            std::vector<float> scores(q_pos + 2, 0.0f); // last entry is zero-sink (score = 0)
            float mx = 0.0f;
            for (int kp = 0; kp <= q_pos; ++kp) {
                const float * kptr = &K_all[((size_t) kh * kv_len + kp) * D];
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += qp[d] * kptr[d];
                dot *= scale;
                const int dist_k = q_pos - kp;
                float rb = 0.0f;
                if (dist_k < rel_ext) {
                    const float * rbp = &rel_bias[(size_t) dist_k * d_rel];
                    for (int r = 0; r < d_rel; ++r) rb += rbp[r] * rp[r];
                }
                scores[kp] = dot + rb - sv;
                mx = std::max(mx, scores[kp]);
            }
            scores[q_pos + 1] = 0.0f; // zero-sink
            mx = std::max(mx, 0.0f);

            float sum = std::exp(0.0f - mx);
            std::vector<float> probs(q_pos + 1);
            for (int kp = 0; kp <= q_pos; ++kp) {
                probs[kp] = std::exp(scores[kp] - mx);
                sum += probs[kp];
            }
            float * op = &ref[((size_t) t * Hq + h) * D];
            const float * gp = &Gate[((size_t) t * Hq + h) * D];
            for (int d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (int kp = 0; kp <= q_pos; ++kp) {
                    acc += (probs[kp] / sum) * V_all[((size_t) kh * kv_len + kp) * D + d];
                }
                op[d] = acc / (1.0f + std::exp(-gp[d]));
            }
        }
    }

    // Build GPU graph: pack K/V into cache -> rel_table -> attn_rel
    const ggml_type kv_t = use_q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    struct ggml_init_params gp = { 128 * ggml_tensor_overhead() + ggml_graph_overhead() + 8192, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * t_Q    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hq, T);
    ggml_tensor * t_Kc   = ggml_new_tensor_3d(ctx, kv_t, D, kv_cap, Hk);
    ggml_tensor * t_Vc   = ggml_new_tensor_3d(ctx, kv_t, D, kv_cap, Hk);
    ggml_tensor * t_rb   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_rel, rel_ext);
    ggml_tensor * t_R    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_rel, Hq, T);
    ggml_tensor * t_Sink = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Hq, T);
    ggml_tensor * t_Gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D * Hq, T);
    ggml_tensor * t_meta = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    ggml_set_input(t_Q); ggml_set_input(t_Kc); ggml_set_input(t_Vc);
    ggml_set_input(t_rb); ggml_set_input(t_R); ggml_set_input(t_Sink);
    ggml_set_input(t_Gate); ggml_set_input(t_meta);

    ggml_tensor * rt  = ggml_inkling2_rel_table(ctx, t_rb, t_R, t_Sink);
    ggml_tensor * out = ggml_inkling2_attn_rel(ctx, t_Q, t_Kc, t_Vc, rt, t_meta, nullptr, nullptr, t_Gate, 0, rel_ext, 0, 1);
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);

    // Upload K/V cache
    const size_t row_b = ggml_row_size(kv_t, D);
    std::vector<uint8_t> kbuf((size_t) Hk * kv_cap * row_b, 0), vbuf((size_t) Hk * kv_cap * row_b, 0);
    for (int h = 0; h < Hk; ++h) {
        for (int kp = 0; kp < kv_len; ++kp) {
            const float * ks = &K_all[((size_t) h * kv_len + kp) * D];
            const float * vs = &V_all[((size_t) h * kv_len + kp) * D];
            uint8_t * kd = kbuf.data() + ((size_t) h * kv_cap + kp) * row_b;
            uint8_t * vd = vbuf.data() + ((size_t) h * kv_cap + kp) * row_b;
            if (use_q8) {
                ggml_quantize_chunk(GGML_TYPE_Q8_0, ks, kd, 0, 1, D, nullptr);
                ggml_quantize_chunk(GGML_TYPE_Q8_0, vs, vd, 0, 1, D, nullptr);
            } else {
                auto * kh = reinterpret_cast<ggml_fp16_t *>(kd);
                auto * vh = reinterpret_cast<ggml_fp16_t *>(vd);
                for (int d = 0; d < D; ++d) {
                    kh[d] = ggml_fp32_to_fp16(ks[d]);
                    vh[d] = ggml_fp32_to_fp16(vs[d]);
                }
            }
        }
    }

    int32_t meta[4] = { pos0, kv_len, 0, kv_len };
    ggml_backend_tensor_set(t_Q, Q.data(), 0, Q.size() * sizeof(float));
    ggml_backend_tensor_set(t_Kc, kbuf.data(), 0, kbuf.size());
    ggml_backend_tensor_set(t_Vc, vbuf.data(), 0, vbuf.size());
    ggml_backend_tensor_set(t_rb, rel_bias.data(), 0, rel_bias.size() * sizeof(float));
    ggml_backend_tensor_set(t_R, R.data(), 0, R.size() * sizeof(float));
    ggml_backend_tensor_set(t_Sink, Sink.data(), 0, Sink.size() * sizeof(float));
    ggml_backend_tensor_set(t_Gate, Gate.data(), 0, Gate.size() * sizeof(float));
    ggml_backend_tensor_set(t_meta, meta, 0, sizeof(meta));
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> got(ref.size());
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(ctx);

    const float err = max_abs_diff(got, ref);
    const float tol = use_q8 ? 2e-2f : 2e-3f;
    std::printf("[test_inkling2_ops] attn_rel(%s, T=%d, pos0=%d) max_abs_err=%.3e (tol=%.1e)\n",
                use_q8 ? "Q8_0" : "F16", T, pos0, err, tol);
    return err < tol;
}

static bool test_sigmoid_topk(ggml_backend_t backend) {
    const int E = 128, K = 8, T = 4;
    const float scale = 8.0f;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> logits((size_t) E * T), bias(E);
    for (float & v : logits) v = dist(rng);
    for (float & v : bias) v = dist(rng) * 0.2f;

    std::vector<int32_t> ref_ids((size_t) K * T);
    std::vector<float> ref_wts((size_t) K * T);
    for (int t = 0; t < T; ++t) {
        std::vector<std::pair<float, int>> sc(E);
        std::vector<float> probs(E);
        for (int e = 0; e < E; ++e) {
            probs[e] = 1.0f / (1.0f + std::exp(-logits[t * E + e]));
            sc[e] = {probs[e] + bias[e], e};
        }
        std::partial_sort(sc.begin(), sc.begin() + K, sc.end(),
            [](auto a, auto b) { return a.first > b.first; });
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            ref_ids[t * K + k] = sc[k].second;
            sum += probs[sc[k].second];
        }
        float inv = scale / std::max(sum, 1e-20f);
        for (int k = 0; k < K; ++k) ref_wts[t * K + k] = probs[sc[k].second] * inv;
    }

    struct ggml_init_params gp = { 64 * ggml_tensor_overhead() + ggml_graph_overhead() + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * t_l = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, T);
    ggml_tensor * t_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, E);
    ggml_set_input(t_l); ggml_set_input(t_b);
    ggml_tensor * t_ids = ggml_inkling2_sigmoid_topk_ids(ctx, t_l, t_b, K);
    ggml_tensor * t_wts = ggml_inkling2_sigmoid_topk_wts(ctx, t_l, t_ids, scale, 1e-20f);
    ggml_set_output(t_ids); ggml_set_output(t_wts);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_wts);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(t_l, logits.data(), 0, logits.size() * sizeof(float));
    ggml_backend_tensor_set(t_b, bias.data(), 0, bias.size() * sizeof(float));
    ggml_backend_graph_compute(backend, gf);

    std::vector<int32_t> got_ids(ref_ids.size());
    std::vector<float> got_wts(ref_wts.size());
    ggml_backend_tensor_get(t_ids, got_ids.data(), 0, got_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_get(t_wts, got_wts.data(), 0, got_wts.size() * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(ctx);

    bool ids_ok = (got_ids == ref_ids);
    float err = max_abs_diff(got_wts, ref_wts);
    std::printf("[test_inkling2_ops] sigmoid_topk ids=%s wts_err=%.3e\n", ids_ok ? "EXACT" : "MISMATCH", err);
    return ids_ok && err < 1e-5f;
}

static bool test_dynamic_split_lut(ggml_backend_t backend) {
    const int E = 128, K = 8, T = 8, pct = 73;
    std::mt19937 rng(2026);
    std::uniform_int_distribution<int> dist(0, E - 1);
    std::vector<int32_t> sel((size_t) K * T);
    std::vector<int> freq(E, 0);
    for (int i = 0; i < K * T; ++i) {
        sel[i] = dist(rng);
        freq[sel[i]]++;
    }
    int n_unique = 0;
    for (int e = 0; e < E; ++e) if (freq[e] > 0) ++n_unique;
    const int expected_hot = (n_unique * pct + 99) / 100;

    struct ggml_init_params gp = { 64 * ggml_tensor_overhead() + ggml_graph_overhead() + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * t_sel = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K, T);
    ggml_set_input(t_sel);
    ggml_tensor * t_lut = ggml_inkling2_dynamic_split_lut(ctx, t_sel, E, pct);
    ggml_set_output(t_lut);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_lut);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(t_sel, sel.data(), 0, sel.size() * sizeof(int32_t));
    ggml_backend_graph_compute(backend, gf);

    std::vector<int32_t> got(E);
    ggml_backend_tensor_get(t_lut, got.data(), 0, got.size() * sizeof(int32_t));
    ggml_gallocr_free(ga);
    ggml_free(ctx);

    int got_hot = 0, min_hot_freq = 999, max_cold_freq = 0;
    bool ok = true;
    for (int e = 0; e < E; ++e) {
        if (got[e] >= 0) {
            if (got[e] != e || freq[e] == 0) ok = false;
            ++got_hot;
            min_hot_freq = std::min(min_hot_freq, freq[e]);
        } else if (freq[e] > 0) {
            max_cold_freq = std::max(max_cold_freq, freq[e]);
        }
    }
    ok &= (got_hot == expected_hot) && (min_hot_freq >= max_cold_freq);
    std::printf("[test_inkling2_ops] dynamic_split_lut unique=%d hot=%d/%d min_hot_f=%d max_cold_f=%d (%s)\n",
                n_unique, got_hot, expected_hot, min_hot_freq, max_cold_freq, ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_dspark_topk(ggml_backend_t backend) {
    const int V = 200058, R = 256;
    std::mt19937 rng(31415);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> base(V), latent(R), fold_b(V);
    std::vector<ggml_fp16_t> fold_w((size_t) R * V);
    for (float & v : base) v = dist(rng);
    for (float & v : latent) v = dist(rng) * 0.2f;
    for (float & v : fold_b) v = dist(rng) * 0.1f;
    for (size_t i = 0; i < fold_w.size(); ++i) fold_w[i] = ggml_fp32_to_fp16(dist(rng) * 0.05f);
    // Plant a clear winner inside base_logits
    const int target_id = 123456;
    base[target_id] = 12.0f;

    struct ggml_init_params gp = { 64 * ggml_tensor_overhead() + ggml_graph_overhead() + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * t_base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, V, 1);
    ggml_tensor * t_lat  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, R, 1);
    ggml_tensor * t_fw   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, R, V);
    ggml_tensor * t_fb   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, V);
    ggml_set_input(t_base); ggml_set_input(t_lat); ggml_set_input(t_fw); ggml_set_input(t_fb);

    ggml_tensor * cands = ggml_inkling2_dspark_topk_cands(ctx, t_base, V);
    ggml_tensor * best  = ggml_inkling2_dspark_topk_score(ctx, ggml_view_1d(ctx, cands, 1024, 0), t_base, t_lat, t_fw, t_fb, 30.0f);
    ggml_set_output(best);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, best);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(t_base, base.data(), 0, base.size() * sizeof(float));
    ggml_backend_tensor_set(t_lat, latent.data(), 0, latent.size() * sizeof(float));
    ggml_backend_tensor_set(t_fw, fold_w.data(), 0, fold_w.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(t_fb, fold_b.data(), 0, fold_b.size() * sizeof(float));
    ggml_backend_graph_compute(backend, gf);

    int32_t got_id = -1;
    ggml_backend_tensor_get(best, &got_id, 0, sizeof(int32_t));
    ggml_gallocr_free(ga);
    ggml_free(ctx);

    const bool ok = (got_id == target_id);
    std::printf("[test_inkling2_ops] dspark_topk got=%d expected=%d (%s)\n", got_id, target_id, ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_dspark_markov_argmax(ggml_backend_t backend) {
    const int V = 200058, R = 256;
    std::mt19937 rng(27182);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> base(V), latent(R), fold_b(V), fold_w_f32((size_t) R * V);
    for (float & v : base) v = dist(rng);
    for (float & v : latent) v = dist(rng) * 0.25f;
    for (float & v : fold_b) v = dist(rng) * 0.1f;
    for (float & v : fold_w_f32) v = dist(rng) * 0.05f;
    const int target_id = 177777;
    for (int r = 0; r < R; ++r) fold_w_f32[(size_t) target_id * R + r] = latent[r] * 0.5f;
    base[target_id] = 6.0f;

    const size_t q8_row = ggml_row_size(GGML_TYPE_Q8_0, R);
    std::vector<uint8_t> fold_w_q8((size_t) V * q8_row);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, fold_w_f32.data(), fold_w_q8.data(), 0, V, R, nullptr);

    struct ggml_init_params gp = { 64 * ggml_tensor_overhead() + ggml_graph_overhead() + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * t_base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, V, 1);
    ggml_tensor * t_lat  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, R, 1);
    ggml_tensor * t_fw   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, R, V);
    ggml_tensor * t_fb   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, V);
    ggml_set_input(t_base); ggml_set_input(t_lat); ggml_set_input(t_fw); ggml_set_input(t_fb);

    ggml_tensor * best = ggml_inkling2_dspark_markov_argmax(ctx, t_base, t_lat, t_fw, t_fb, V);
    ggml_set_output(best);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, best);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(t_base, base.data(), 0, base.size() * sizeof(float));
    ggml_backend_tensor_set(t_lat, latent.data(), 0, latent.size() * sizeof(float));
    ggml_backend_tensor_set(t_fw, fold_w_q8.data(), 0, fold_w_q8.size());
    ggml_backend_tensor_set(t_fb, fold_b.data(), 0, fold_b.size() * sizeof(float));
    ggml_backend_graph_compute(backend, gf);

    int32_t got_id = -1;
    ggml_backend_tensor_get(best, &got_id, 0, sizeof(int32_t));
    ggml_gallocr_free(ga);
    ggml_free(ctx);

    const bool ok = (got_id == target_id);
    std::printf("[test_inkling2_ops] dspark_markov_argmax(Q8_0) got=%d expected=%d (%s)\n", got_id, target_id, ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_moe_masked_sum(ggml_backend_t backend) {
    const int H = 2304, K = 8, T = 6;
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    std::vector<float> down((size_t) H * K * T), wsel((size_t) K * T), peer((size_t) H * T);
    std::vector<int32_t> msel((size_t) K * T);
    for (float & v : down) v = dist(rng);
    for (float & v : wsel) v = std::fabs(dist(rng));
    for (float & v : peer) v = dist(rng);
    for (int i = 0; i < K * T; ++i) msel[i] = (i % 3 == 0) ? -1 : (i % 128);

    std::vector<float> ref((size_t) H * T, 0.0f);
    for (int t = 0; t < T; ++t) {
        for (int e = 0; e < H; ++e) {
            float acc = peer[(size_t) t * H + e];
            for (int k = 0; k < K; ++k) {
                if (msel[(size_t) t * K + k] >= 0) {
                    acc += wsel[(size_t) t * K + k] * down[((size_t) t * K + k) * H + e];
                }
            }
            ref[(size_t) t * H + e] = acc;
        }
    }

    struct ggml_init_params gp = { 64 * ggml_tensor_overhead() + ggml_graph_overhead() + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * t_dn = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, K, T);
    ggml_tensor * t_ws = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    ggml_tensor * t_ms = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K, T);
    ggml_tensor * t_pr = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_set_input(t_dn); ggml_set_input(t_ws); ggml_set_input(t_ms); ggml_set_input(t_pr);
    ggml_tensor * out = ggml_inkling2_moe_masked_sum(ctx, t_dn, t_ws, t_ms, t_pr, 0);
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(t_dn, down.data(), 0, down.size() * sizeof(float));
    ggml_backend_tensor_set(t_ws, wsel.data(), 0, wsel.size() * sizeof(float));
    ggml_backend_tensor_set(t_ms, msel.data(), 0, msel.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_pr, peer.data(), 0, peer.size() * sizeof(float));
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> got(ref.size());
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(ctx);

    const float err = max_abs_diff(got, ref);
    std::printf("[test_inkling2_ops] moe_masked_sum max_abs_err=%.3e\n", err);
    return err < 1e-5f;
}

static bool test_sconv_kv_write_and_clip_r(ggml_backend_t backend) {
    // Verifies fused sconv_kv_write (4-warp LDS-tiled) matches separate sconv + kv_write,
    // and fused clip_r_rel_table matches separate clip_norm + rel_table!
    const int Hq = 24, Hk = 8, D = 128, C = Hk * D, T = 6, kv_cap = 64, d_rel = 16, rel_ext = 64;
    std::mt19937 rng(9876);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    std::vector<float> x((size_t) C * T), w_st((size_t) 4 * C), st0((size_t) 3 * C);
    std::vector<float> r_raw((size_t) d_rel * Hq * T), r_cw((size_t) d_rel * Hq), rb((size_t) d_rel * rel_ext), sink((size_t) Hq * T);
    for (float & v : x) v = dist(rng);
    for (float & v : w_st) v = dist(rng) * 0.2f;
    for (float & v : st0) v = dist(rng);
    for (float & v : r_raw) v = dist(rng);
    for (float & v : r_cw) v = std::fabs(dist(rng)) + 0.5f;
    for (float & v : rb) v = dist(rng) * 0.2f;
    for (float & v : sink) v = dist(rng) * 0.5f;
    std::vector<int32_t> pos(T);
    for (int t = 0; t < T; ++t) pos[t] = 10 + t;

    struct ggml_init_params gp = { 128 * ggml_tensor_overhead() + ggml_graph_overhead() + 8192, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * t_x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, T);
    ggml_tensor * t_w   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, C);
    ggml_tensor * t_st1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, C);
    ggml_tensor * t_st2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, C);
    ggml_tensor * t_kv1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, kv_cap, Hk);
    ggml_tensor * t_kv2 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, kv_cap, Hk);
    ggml_tensor * t_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_tensor * t_rr  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_rel * Hq, T);
    ggml_tensor * t_rcw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_rel, Hq);
    ggml_tensor * t_rb  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_rel, rel_ext);
    ggml_tensor * t_sk  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Hq, T);
    ggml_set_input(t_x); ggml_set_input(t_w); ggml_set_input(t_st1); ggml_set_input(t_st2);
    ggml_set_input(t_kv1); ggml_set_input(t_kv2); ggml_set_input(t_pos);
    ggml_set_input(t_rr); ggml_set_input(t_rcw); ggml_set_input(t_rb); ggml_set_input(t_sk);

    ggml_tensor * kc = ggml_inkling2_sconv(ctx, t_x, t_w, t_st1, t_st1, nullptr, 0, D, 1);
    ggml_tensor * d1 = ggml_inkling2_kv_write(ctx, kc, t_kv1, t_pos, D, Hk, 0, 0, 22.6f, 1e-6f);
    ggml_tensor * d2 = ggml_inkling2_sconv_kv_write(ctx, t_x, t_w, t_st2, t_st2, nullptr, t_kv2, t_pos, 0, D, Hk, 0, 0, 22.6f, 1e-6f, 1);

    ggml_tensor * r_cl = ggml_inkling2_clip_norm(ctx, t_rr, t_rcw, nullptr, d_rel, Hq, 0, 8.0f, 1e-6f, 0, 0.0f);
    ggml_tensor * rt1  = ggml_inkling2_rel_table(ctx, t_rb, r_cl, t_sk);
    ggml_tensor * rt2  = ggml_inkling2_clip_r_rel_table(ctx, t_rr, t_rcw, nullptr, t_rb, t_sk, d_rel, Hq, 8.0f, 1e-6f, 0, 0.0f);
    ggml_set_output(d1); ggml_set_output(d2); ggml_set_output(rt1); ggml_set_output(rt2);
    ggml_set_output(t_kv1); ggml_set_output(t_kv2);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, d1);
    ggml_build_forward_expand(gf, d2);
    ggml_build_forward_expand(gf, rt1);
    ggml_build_forward_expand(gf, rt2);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(t_x, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(t_w, w_st.data(), 0, w_st.size() * sizeof(float));
    ggml_backend_tensor_set(t_st1, st0.data(), 0, st0.size() * sizeof(float));
    ggml_backend_tensor_set(t_st2, st0.data(), 0, st0.size() * sizeof(float));
    ggml_backend_tensor_memset(t_kv1, 0, 0, ggml_nbytes(t_kv1));
    ggml_backend_tensor_memset(t_kv2, 0, 0, ggml_nbytes(t_kv2));
    ggml_backend_tensor_set(t_pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_rr, r_raw.data(), 0, r_raw.size() * sizeof(float));
    ggml_backend_tensor_set(t_rcw, r_cw.data(), 0, r_cw.size() * sizeof(float));
    ggml_backend_tensor_set(t_rb, rb.data(), 0, rb.size() * sizeof(float));
    ggml_backend_tensor_set(t_sk, sink.data(), 0, sink.size() * sizeof(float));
    ggml_backend_graph_compute(backend, gf);

    std::vector<ggml_fp16_t> kv1((size_t) D * kv_cap * Hk), kv2((size_t) D * kv_cap * Hk);
    ggml_backend_tensor_get(t_kv1, kv1.data(), 0, kv1.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(t_kv2, kv2.data(), 0, kv2.size() * sizeof(ggml_fp16_t));
    std::vector<float> v_rt1((size_t)(rel_ext + 1) * Hq * T), v_rt2((size_t)(rel_ext + 1) * Hq * T);
    ggml_backend_tensor_get(rt1, v_rt1.data(), 0, v_rt1.size() * sizeof(float));
    ggml_backend_tensor_get(rt2, v_rt2.data(), 0, v_rt2.size() * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(ctx);

    float kv_err = 0.0f;
    for (size_t i = 0; i < kv1.size(); ++i) {
        kv_err = std::max(kv_err, std::fabs(ggml_fp16_to_fp32(kv1[i]) - ggml_fp16_to_fp32(kv2[i])));
    }
    const float rt_err = max_abs_diff(v_rt1, v_rt2);
    std::printf("[test_inkling2_ops] sconv_kv_write err=%.3e clip_r_rel_table err=%.3e\n", kv_err, rt_err);
    return kv_err < 1e-4f && rt_err < 1e-5f;
}

static inline void test_set_op_i32(ggml_tensor * t, int idx, int32_t v) {
    std::memcpy(&t->op_params[idx], &v, sizeof(v));
}
static inline void test_set_op_f32(ggml_tensor * t, int idx, float v) {
    std::memcpy(&t->op_params[idx], &v, sizeof(v));
}

// ─── 2DSD Unit Test 1: Tree SConv + Tree KV Write + Tree Attention + Tree Commit ──
static bool test_attn_rel_paged(ggml_backend_t backend, bool use_q8, int T, int pos0, int block_size, bool with_holes) {
    const int Hq = 24, Hk = 8, D = 128, d_rel = 16, rel_ext = 64;
    const int kv_len = pos0 + T;
    const int n_blocks = (kv_len + block_size - 1) / block_size;
    const int n_phys_blocks = n_blocks + 3; // extra spare blocks to verify non-trivial physical mapping
    const int kv_cap = n_phys_blocks * block_size;

    std::mt19937 rng(9001 + T * 13 + block_size + (use_q8 ? 100 : 0) + (with_holes ? 500 : 0));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Q((size_t) D * Hq * T);
    std::vector<float> K_all((size_t) D * kv_len * Hk);
    std::vector<float> V_all((size_t) D * kv_len * Hk);
    std::vector<float> rel_bias((size_t) d_rel * rel_ext);
    std::vector<float> R((size_t) d_rel * Hq * T);
    std::vector<float> Sink((size_t) Hq * T);
    std::vector<float> Gate((size_t) D * Hq * T);
    for (float & v : Q) v = dist(rng);
    for (float & v : K_all) v = dist(rng);
    for (float & v : V_all) v = dist(rng);
    for (float & v : rel_bias) v = dist(rng) * 0.2f;
    for (float & v : R) v = dist(rng);
    for (float & v : Sink) v = dist(rng) * 0.5f;
    for (float & v : Gate) v = dist(rng);

    // Build a shuffled physical block table (optionally punching evicted -1 holes, incl. full 32-tok WMMA tiles).
    std::vector<int32_t> perm(n_phys_blocks);
    for (int i = 0; i < n_phys_blocks; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int32_t> bt(n_blocks, -1);
    for (int b = 0; b < n_blocks; ++b) {
        if (with_holes && ((b % 3 == 1) || (block_size == 16 && (b == 2 || b == 3)))) {
            bt[b] = -1; // mix of single-block and full-32-token-tile holes
        } else {
            bt[b] = perm[b];
        }
    }

    // Exact CPU reference honoring block_table holes and zero-sink
    std::vector<float> ref((size_t) D * Hq * T, 0.0f);
    const float scale = 1.0f / std::sqrt((float) D);
    for (int t = 0; t < T; ++t) {
        const int q_pos = pos0 + t;
        for (int h = 0; h < Hq; ++h) {
            const int kh = h / (Hq / Hk);
            const float * qp = &Q[((size_t) t * Hq + h) * D];
            const float * rp = &R[((size_t) t * Hq + h) * d_rel];
            const float sv = Sink[(size_t) t * Hq + h];

            std::vector<float> scores(q_pos + 1, -INFINITY);
            float mx = 0.0f; // zero-sink
            for (int kp = 0; kp <= q_pos; ++kp) {
                if (bt[kp / block_size] < 0) continue;
                const float * kptr = &K_all[((size_t) kh * kv_len + kp) * D];
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += qp[d] * kptr[d];
                dot *= scale;
                const int dist_k = q_pos - kp;
                float rb = 0.0f;
                if (dist_k < rel_ext) {
                    const float * rbp = &rel_bias[(size_t) dist_k * d_rel];
                    for (int r = 0; r < d_rel; ++r) rb += rbp[r] * rp[r];
                }
                scores[kp] = dot + rb - sv;
                mx = std::max(mx, scores[kp]);
            }
            float sum = std::exp(0.0f - mx);
            std::vector<float> probs(q_pos + 1, 0.0f);
            for (int kp = 0; kp <= q_pos; ++kp) {
                if (std::isfinite(scores[kp])) {
                    probs[kp] = std::exp(scores[kp] - mx);
                    sum += probs[kp];
                }
            }
            float * op = &ref[((size_t) t * Hq + h) * D];
            const float * gp = &Gate[((size_t) t * Hq + h) * D];
            for (int d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (int kp = 0; kp <= q_pos; ++kp) {
                    if (probs[kp] > 0.0f) acc += (probs[kp] / sum) * V_all[((size_t) kh * kv_len + kp) * D + d];
                }
                op[d] = acc / (1.0f + std::exp(-gp[d]));
            }
        }
    }

    const ggml_type kv_t = use_q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    struct ggml_init_params gp = { 128 * ggml_tensor_overhead() + ggml_graph_overhead() + 8192, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * t_Q    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hq, T);
    ggml_tensor * t_Kc   = ggml_new_tensor_3d(ctx, kv_t, D, kv_cap, Hk);
    ggml_tensor * t_Vc   = ggml_new_tensor_3d(ctx, kv_t, D, kv_cap, Hk);
    ggml_tensor * t_rb   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_rel, rel_ext);
    ggml_tensor * t_R    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_rel, Hq, T);
    ggml_tensor * t_Sink = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Hq, T);
    ggml_tensor * t_Gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D * Hq, T);
    ggml_tensor * t_meta = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    ggml_tensor * t_bt   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_blocks);
    ggml_set_input(t_Q); ggml_set_input(t_Kc); ggml_set_input(t_Vc);
    ggml_set_input(t_rb); ggml_set_input(t_R); ggml_set_input(t_Sink);
    ggml_set_input(t_Gate); ggml_set_input(t_meta); ggml_set_input(t_bt);

    ggml_tensor * rt  = ggml_inkling2_rel_table(ctx, t_rb, t_R, t_Sink);
    ggml_tensor * out = ggml_inkling2_attn_rel_paged(
        ctx, t_Q, t_Kc, t_Vc, rt, t_meta, nullptr, nullptr, t_Gate, t_bt, block_size, rel_ext, 0, 1);
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);

    // Fill unused physical slots with poison values to catch any accidental read through a -1 hole!
    const size_t row_b = ggml_row_size(kv_t, D);
    std::vector<uint8_t> kbuf((size_t) Hk * kv_cap * row_b, 0x7F), vbuf((size_t) Hk * kv_cap * row_b, 0x7F);
    for (int h = 0; h < Hk; ++h) {
        for (int kp = 0; kp < kv_len; ++kp) {
            const int pb = bt[kp / block_size];
            if (pb < 0) continue;
            const int slot = pb * block_size + (kp % block_size);
            const float * ks = &K_all[((size_t) h * kv_len + kp) * D];
            const float * vs = &V_all[((size_t) h * kv_len + kp) * D];
            uint8_t * kd = kbuf.data() + ((size_t) h * kv_cap + slot) * row_b;
            uint8_t * vd = vbuf.data() + ((size_t) h * kv_cap + slot) * row_b;
            if (use_q8) {
                ggml_quantize_chunk(GGML_TYPE_Q8_0, ks, kd, 0, 1, D, nullptr);
                ggml_quantize_chunk(GGML_TYPE_Q8_0, vs, vd, 0, 1, D, nullptr);
            } else {
                auto * kh = reinterpret_cast<ggml_fp16_t *>(kd);
                auto * vh = reinterpret_cast<ggml_fp16_t *>(vd);
                for (int d = 0; d < D; ++d) {
                    kh[d] = ggml_fp32_to_fp16(ks[d]);
                    vh[d] = ggml_fp32_to_fp16(vs[d]);
                }
            }
        }
    }

    int32_t meta[4] = { pos0, kv_len, 0, kv_len };
    ggml_backend_tensor_set(t_Q, Q.data(), 0, Q.size() * sizeof(float));
    ggml_backend_tensor_set(t_Kc, kbuf.data(), 0, kbuf.size());
    ggml_backend_tensor_set(t_Vc, vbuf.data(), 0, vbuf.size());
    ggml_backend_tensor_set(t_rb, rel_bias.data(), 0, rel_bias.size() * sizeof(float));
    ggml_backend_tensor_set(t_R, R.data(), 0, R.size() * sizeof(float));
    ggml_backend_tensor_set(t_Sink, Sink.data(), 0, Sink.size() * sizeof(float));
    ggml_backend_tensor_set(t_Gate, Gate.data(), 0, Gate.size() * sizeof(float));
    ggml_backend_tensor_set(t_meta, meta, 0, sizeof(meta));
    ggml_backend_tensor_set(t_bt, bt.data(), 0, bt.size() * sizeof(int32_t));
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> got(ref.size());
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(ctx);

    const float err = max_abs_diff(got, ref);
    const float tol = use_q8 ? 2e-2f : 2e-3f;
    std::printf("[test_inkling2_ops] attn_rel_paged(%s, T=%d, pos0=%d, B=%d, holes=%d) max_abs_err=%.3e (tol=%.1e)\n",
                use_q8 ? "Q8_0" : "F16", T, pos0, block_size, with_holes ? 1 : 0, err, tol);
    return err < tol;
}

static bool test_attn_rel_paged_dual_split_combine(ggml_backend_t backend, bool use_q8, int T, int pos0, bool all_remote) {
    // Disjoint Dual-GPU Split-KV test: partition logical blocks across Tier-0 (local) & Tier-1.5 (remote)
    // plus evicted -1 holes on both, run partial=1 paged attention on each, merge via split_combine,
    // and assert bit-close equivalence to single-table paged attention!
    const int Hq = 24, Hk = 8, D = 128, d_rel = 16, rel_ext = 64, block_size = 16;
    const int kv_len = pos0 + T;
    const int n_blocks = (kv_len + block_size - 1) / block_size;
    const int kv_cap = (n_blocks + 2) * block_size;

    std::mt19937 rng(4242 + T * 7 + (use_q8 ? 10 : 0) + (all_remote ? 1000 : 0));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Q((size_t) D * Hq * T), K_all((size_t) D * kv_len * Hk), V_all((size_t) D * kv_len * Hk);
    std::vector<float> rel_bias((size_t) d_rel * rel_ext), R((size_t) d_rel * Hq * T), Sink((size_t) Hq * T), Gate((size_t) D * Hq * T);
    for (float & v : Q) v = dist(rng);
    for (float & v : K_all) v = dist(rng);
    for (float & v : V_all) v = dist(rng);
    for (float & v : rel_bias) v = dist(rng) * 0.2f;
    for (float & v : R) v = dist(rng);
    for (float & v : Sink) v = dist(rng) * 0.5f;
    for (float & v : Gate) v = dist(rng);

    std::vector<int32_t> bt_full(n_blocks, -1), bt_local(n_blocks, -1), bt_remote(n_blocks, -1);
    for (int b = 0; b < n_blocks; ++b) {
        if (!all_remote && (b % 5 == 2)) continue; // evicted on both tiers
        const int phys = (n_blocks - 1 - b) + 1;
        bt_full[b] = phys;
        if (all_remote || (b % 2 == 0)) {
            bt_remote[b] = phys;
        } else {
            bt_local[b] = phys;
        }
    }

    const ggml_type kv_t = use_q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    struct ggml_init_params gp = { 256 * ggml_tensor_overhead() + ggml_graph_overhead() + 8192, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * t_Q    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hq, T);
    ggml_tensor * t_Kc   = ggml_new_tensor_3d(ctx, kv_t, D, kv_cap, Hk);
    ggml_tensor * t_Vc   = ggml_new_tensor_3d(ctx, kv_t, D, kv_cap, Hk);
    ggml_tensor * t_rb   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_rel, rel_ext);
    ggml_tensor * t_R    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_rel, Hq, T);
    ggml_tensor * t_Sink = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Hq, T);
    ggml_tensor * t_Gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D * Hq, T);
    ggml_tensor * t_meta = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    ggml_tensor * t_btf  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_blocks);
    ggml_tensor * t_btl  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_blocks);
    ggml_tensor * t_btr  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_blocks);
    ggml_set_input(t_Q); ggml_set_input(t_Kc); ggml_set_input(t_Vc);
    ggml_set_input(t_rb); ggml_set_input(t_R); ggml_set_input(t_Sink);
    ggml_set_input(t_Gate); ggml_set_input(t_meta);
    ggml_set_input(t_btf); ggml_set_input(t_btl); ggml_set_input(t_btr);

    ggml_tensor * rt     = ggml_inkling2_rel_table(ctx, t_rb, t_R, t_Sink);
    ggml_tensor * single = ggml_inkling2_attn_rel_paged(
        ctx, t_Q, t_Kc, t_Vc, rt, t_meta, nullptr, nullptr, t_Gate, t_btf, block_size, rel_ext, /*partial=*/0, /*include_sink=*/1);
    ggml_tensor * part_r = ggml_inkling2_attn_rel_paged(
        ctx, t_Q, t_Kc, t_Vc, rt, t_meta, nullptr, nullptr, nullptr, t_btr, block_size, rel_ext, /*partial=*/1, /*include_sink=*/1);
    ggml_tensor * part_l = ggml_inkling2_attn_rel_paged(
        ctx, t_Q, t_Kc, t_Vc, rt, t_meta, nullptr, nullptr, nullptr, t_btl, block_size, rel_ext, /*partial=*/1, /*include_sink=*/0);
    ggml_tensor * merged = ggml_inkling2_attn_split_combine(ctx, part_r, part_l, t_Gate, D, Hq);
    ggml_set_output(single);
    ggml_set_output(merged);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, single);
    ggml_build_forward_expand(gf, merged);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);

    const size_t row_b = ggml_row_size(kv_t, D);
    std::vector<uint8_t> kbuf((size_t) Hk * kv_cap * row_b, 0), vbuf((size_t) Hk * kv_cap * row_b, 0);
    for (int h = 0; h < Hk; ++h) {
        for (int kp = 0; kp < kv_len; ++kp) {
            const int pb = bt_full[kp / block_size];
            if (pb < 0) continue;
            const int slot = pb * block_size + (kp % block_size);
            const float * ks = &K_all[((size_t) h * kv_len + kp) * D];
            const float * vs = &V_all[((size_t) h * kv_len + kp) * D];
            uint8_t * kd = kbuf.data() + ((size_t) h * kv_cap + slot) * row_b;
            uint8_t * vd = vbuf.data() + ((size_t) h * kv_cap + slot) * row_b;
            if (use_q8) {
                ggml_quantize_chunk(GGML_TYPE_Q8_0, ks, kd, 0, 1, D, nullptr);
                ggml_quantize_chunk(GGML_TYPE_Q8_0, vs, vd, 0, 1, D, nullptr);
            } else {
                auto * kh = reinterpret_cast<ggml_fp16_t *>(kd);
                auto * vh = reinterpret_cast<ggml_fp16_t *>(vd);
                for (int d = 0; d < D; ++d) {
                    kh[d] = ggml_fp32_to_fp16(ks[d]);
                    vh[d] = ggml_fp32_to_fp16(vs[d]);
                }
            }
        }
    }

    int32_t meta[4] = { pos0, kv_len, 0, kv_len };
    ggml_backend_tensor_set(t_Q, Q.data(), 0, Q.size() * sizeof(float));
    ggml_backend_tensor_set(t_Kc, kbuf.data(), 0, kbuf.size());
    ggml_backend_tensor_set(t_Vc, vbuf.data(), 0, vbuf.size());
    ggml_backend_tensor_set(t_rb, rel_bias.data(), 0, rel_bias.size() * sizeof(float));
    ggml_backend_tensor_set(t_R, R.data(), 0, R.size() * sizeof(float));
    ggml_backend_tensor_set(t_Sink, Sink.data(), 0, Sink.size() * sizeof(float));
    ggml_backend_tensor_set(t_Gate, Gate.data(), 0, Gate.size() * sizeof(float));
    ggml_backend_tensor_set(t_meta, meta, 0, sizeof(meta));
    ggml_backend_tensor_set(t_btf, bt_full.data(), 0, bt_full.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_btl, bt_local.data(), 0, bt_local.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_btr, bt_remote.data(), 0, bt_remote.size() * sizeof(int32_t));
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> got_single((size_t) D * Hq * T), got_merged((size_t) D * Hq * T);
    ggml_backend_tensor_get(single, got_single.data(), 0, got_single.size() * sizeof(float));
    ggml_backend_tensor_get(merged, got_merged.data(), 0, got_merged.size() * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(ctx);

    const float err = max_abs_diff(got_single, got_merged);
    bool finite = true;
    for (float v : got_merged) if (!std::isfinite(v)) finite = false;
    std::printf("[test_inkling2_ops] dual_split_combine(%s, T=%d, all_remote=%d) max_abs_err=%.3e finite=%d\n",
                use_q8 ? "Q8_0" : "F16", T, all_remote ? 1 : 0, err, finite ? 1 : 0);
    return finite && err < 2e-3f;
}

static bool test_paged_kv_writes(ggml_backend_t backend, bool use_q8, int T) {
    const int Hk = 8, D = 128, C = Hk * D, block_size = 16;
    const int pos0 = 12; // spans across block boundaries [0, 1, 2, ...]
    const int max_pos = pos0 + T;
    const int n_blocks = (max_pos + block_size - 1) / block_size;
    const int kv_cap = (n_blocks + 2) * block_size;

    std::mt19937 rng(5150 + T * 3 + (use_q8 ? 17 : 0));
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> x((size_t) C * T), w_st((size_t) 4 * C), st0((size_t) 3 * C);
    for (float & v : x) v = dist(rng);
    for (float & v : w_st) v = dist(rng) * 0.2f;
    for (float & v : st0) v = dist(rng);

    std::vector<int32_t> pos(T);
    for (int t = 0; t < T; ++t) pos[t] = pos0 + t;

    // Permuted block table with block 1 marked -1 (unowned peer block: write skipped, conv state still updates!)
    std::vector<int32_t> bt(n_blocks, -1);
    for (int b = 0; b < n_blocks; ++b) {
        bt[b] = (b == 1) ? -1 : (n_blocks - b);
    }

    const ggml_type kv_t = use_q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    struct ggml_init_params gp = { 128 * ggml_tensor_overhead() + ggml_graph_overhead() + 8192, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * t_x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, T);
    ggml_tensor * t_w   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, C);
    ggml_tensor * t_st0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, C);
    ggml_tensor * t_st1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, C);
    ggml_tensor * t_st2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, C);
    ggml_tensor * t_kvd = ggml_new_tensor_3d(ctx, kv_t, D, kv_cap, Hk); // dense reference
    ggml_tensor * t_kvp = ggml_new_tensor_3d(ctx, kv_t, D, kv_cap, Hk); // paged kv_write
    ggml_tensor * t_kvs = ggml_new_tensor_3d(ctx, kv_t, D, kv_cap, Hk); // paged sconv_kv_write
    ggml_tensor * t_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_tensor * t_bt  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_blocks);
    ggml_set_input(t_x); ggml_set_input(t_w); ggml_set_input(t_st0); ggml_set_input(t_st1); ggml_set_input(t_st2);
    ggml_set_input(t_kvd); ggml_set_input(t_kvp); ggml_set_input(t_kvs); ggml_set_input(t_pos); ggml_set_input(t_bt);

    ggml_tensor * kc  = ggml_inkling2_sconv(ctx, t_x, t_w, t_st0, t_st0, nullptr, 0, D, 1);
    ggml_tensor * d_d = ggml_inkling2_kv_write(ctx, kc, t_kvd, t_pos, D, Hk, 0, 0, 22.6f, 1e-6f);
    ggml_tensor * d_p = ggml_inkling2_kv_write_paged(ctx, kc, t_kvp, t_pos, t_bt, block_size, D, Hk, 0, 22.6f, 1e-6f);
    ggml_tensor * d_s = ggml_inkling2_sconv_kv_write_paged(
        ctx, t_x, t_w, t_st1, t_st2, nullptr, t_kvs, t_pos, t_bt, block_size, 0, D, Hk, 0, 22.6f, 1e-6f, 1);
    ggml_set_output(d_d); ggml_set_output(d_p); ggml_set_output(d_s);
    ggml_set_output(t_kvd); ggml_set_output(t_kvp); ggml_set_output(t_kvs);
    ggml_set_output(t_st0); ggml_set_output(t_st2);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, d_d);
    ggml_build_forward_expand(gf, d_p);
    ggml_build_forward_expand(gf, d_s);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);

    ggml_backend_tensor_set(t_x, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(t_w, w_st.data(), 0, w_st.size() * sizeof(float));
    ggml_backend_tensor_set(t_st0, st0.data(), 0, st0.size() * sizeof(float));
    ggml_backend_tensor_set(t_st1, st0.data(), 0, st0.size() * sizeof(float));
    ggml_backend_tensor_memset(t_st2, 0, 0, ggml_nbytes(t_st2));
    ggml_backend_tensor_memset(t_kvd, 0, 0, ggml_nbytes(t_kvd));
    ggml_backend_tensor_memset(t_kvp, 0, 0, ggml_nbytes(t_kvp));
    ggml_backend_tensor_memset(t_kvs, 0, 0, ggml_nbytes(t_kvs));
    ggml_backend_tensor_set(t_pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_bt, bt.data(), 0, bt.size() * sizeof(int32_t));
    ggml_backend_graph_compute(backend, gf);

    const size_t row_b = ggml_row_size(kv_t, D);
    const size_t total_b = (size_t) Hk * kv_cap * row_b;
    std::vector<uint8_t> kvd(total_b), kvp(total_b), kvs(total_b);
    std::vector<float> got_st0(st0.size()), got_st2(st0.size());
    ggml_backend_tensor_get(t_kvd, kvd.data(), 0, total_b);
    ggml_backend_tensor_get(t_kvp, kvp.data(), 0, total_b);
    ggml_backend_tensor_get(t_kvs, kvs.data(), 0, total_b);
    ggml_backend_tensor_get(t_st0, got_st0.data(), 0, got_st0.size() * sizeof(float));
    ggml_backend_tensor_get(t_st2, got_st2.data(), 0, got_st2.size() * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(ctx);

    bool exact_bytes = true;
    for (int h = 0; h < Hk; ++h) {
        for (int t = 0; t < T; ++t) {
            const int p = pos[t];
            const int pb = bt[p / block_size];
            if (pb < 0) continue;
            const int slot = pb * block_size + (p % block_size);
            const uint8_t * ref_row = kvd.data() + ((size_t) h * kv_cap + p) * row_b;
            const uint8_t * p_row   = kvp.data() + ((size_t) h * kv_cap + slot) * row_b;
            const uint8_t * s_row   = kvs.data() + ((size_t) h * kv_cap + slot) * row_b;
            if (std::memcmp(ref_row, p_row, row_b) != 0) exact_bytes = false;
            if (std::memcmp(ref_row, s_row, row_b) != 0) exact_bytes = false;
        }
    }
    const float st_err = max_abs_diff(got_st0, got_st2);
    std::printf("[test_inkling2_ops] paged_kv_writes(%s, T=%d) exact_bytes=%s conv_state_err=%.3e\n",
                use_q8 ? "Q8_0" : "F16", T, exact_bytes ? "EXACT" : "DIFF", st_err);
    return exact_bytes && st_err < 1e-6f;
}

// ─── 2DSD Unit Test 1: Tree SConv + Tree KV Write + Tree Attention + Tree Commit ──
static bool test_2dsd_tree_ops(ggml_backend_t backend, bool q8_kv) {
    const int Hq = 24, Hk = 8, D = 128, P = 2, S_max = 16, prefix_cap = 64;
    const int kv_sz = Hk * D;
    const int s_ch = 2 * kv_sz + 128; // K + V + tiny post-conv slice
    const int pref_len = 5;
    const int path_w = 8;
    const int m_stride = 11 + path_w;
    const int rel_ext = 32;
    const float qk_clip = 22.627417f;
    const float eps = 1e-6f;

    // 4-node tree on plane 1: prefix [0..4], then n0(slot 2, pos 5) -> n1(slot 5, pos 6) -> n2(slot 7, pos 7),
    // plus fork n1_alt(slot 9, pos 6, parent=n0). All 4 advance in ONE call!
    const int T = 4;
    const int slots[T] = { 2, 5, 7, 9 };
    const int poss[T]  = { 5, 6, 7, 6 };

    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> x_k((size_t) kv_sz * T), x_v((size_t) kv_sz * T);
    std::vector<float> wk((size_t) 4 * kv_sz), wv((size_t) 4 * kv_sz);
    std::vector<float> pc((size_t) 3 * s_ch * P, 0.0f);
    std::vector<float> q_in((size_t) D * Hq * T), rt_in((size_t) (rel_ext + 1) * Hq * T), gate_in((size_t) D * Hq * T);
    for (float & v : x_k) v = dist(rng);
    for (float & v : x_v) v = dist(rng);
    for (float & v : wk) v = dist(rng) * 0.2f;
    for (float & v : wv) v = dist(rng) * 0.2f;
    for (float & v : pc) v = dist(rng) * 0.5f;
    for (float & v : q_in) v = dist(rng) * 0.5f;
    for (float & v : rt_in) v = dist(rng) * 0.2f;
    for (float & v : gate_in) v = dist(rng);

    // Pack tree meta for all 4 rows on plane 1
    std::vector<int32_t> meta((size_t) m_stride * T, -1);
    auto set_row = [&](int r, int slot, int pos, int hr0, int hr1, int hr2, int hs0, int hs1, int hs2, std::vector<int> path) {
        int32_t * mt = meta.data() + (size_t) r * m_stride;
        mt[0] = slot; mt[1] = pos; mt[2] = 1; mt[3] = 0; mt[4] = 0;
        mt[5] = hr0; mt[6] = hr1; mt[7] = hr2;
        mt[8] = hs0; mt[9] = hs1; mt[10] = hs2;
        for (size_t e = 0; e < path.size(); ++e) mt[11 + e] = path[e];
    };
    set_row(0, 2, 5, -1, -1, -1, -1, -2, -3, {2});
    set_row(1, 5, 6, -1, -1,  0, -2, -3,  2, {2, 5});
    set_row(2, 7, 7, -1,  0,  1, -3,  2,  5, {2, 5, 7});
    set_row(3, 9, 6, -1, -1,  0, -2, -3,  2, {2, 9});

    // Build GPU graph: TREE_SCONV_KV_WRITE (K & V) -> TREE_ATTN_REL -> TREE_COMMIT [slot 2, 5, 7]
    const ggml_type kv_t = q8_kv ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;

    // Persistent state buffers (allocated once on backend, shared across advance + commit graphs)
    struct ggml_init_params sgp = { 16 * ggml_tensor_overhead() + 4096, nullptr, true };
    ggml_context * sctx = ggml_init(sgp);
    ggml_tensor * t_pc = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, 3, s_ch, P);
    ggml_tensor * t_tc = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, s_ch, S_max, P);
    ggml_tensor * t_pk = ggml_new_tensor_4d(sctx, kv_t, D, prefix_cap, Hk, P);
    ggml_tensor * t_pv = ggml_new_tensor_4d(sctx, kv_t, D, prefix_cap, Hk, P);
    ggml_tensor * t_tk = ggml_new_tensor_4d(sctx, kv_t, D, S_max, Hk, P);
    ggml_tensor * t_tv = ggml_new_tensor_4d(sctx, kv_t, D, S_max, Hk, P);
    ggml_backend_buffer_t sbuf = ggml_backend_alloc_ctx_tensors(sctx, backend);

    struct ggml_init_params gp = { 64 * ggml_tensor_overhead() + 2 * ggml_graph_overhead() + (1 << 20), nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * t_xk   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_sz, T);
    ggml_tensor * t_xv   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_sz, T);
    ggml_tensor * t_wk   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, kv_sz);
    ggml_tensor * t_wv   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, kv_sz);
    ggml_tensor * t_meta = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, m_stride, T);
    ggml_tensor * t_plen = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * t_q    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hq, T);
    ggml_tensor * t_rt   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rel_ext + 1, Hq, T);
    ggml_tensor * t_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D * Hq, T);
    ggml_tensor * t_csl  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    ggml_tensor * t_cmt  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);

    ggml_set_input(t_xk); ggml_set_input(t_xv); ggml_set_input(t_wk); ggml_set_input(t_wv);
    ggml_set_input(t_meta); ggml_set_input(t_plen);
    ggml_set_input(t_q); ggml_set_input(t_rt); ggml_set_input(t_gate);
    ggml_set_input(t_csl); ggml_set_input(t_cmt);

    ggml_tensor * dep_k = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    dep_k->op = GGML_OP_INKLING2_OP;
    dep_k->src[0] = t_xk; dep_k->src[1] = t_wk; dep_k->src[2] = t_pc; dep_k->src[3] = t_tc;
    dep_k->src[4] = t_meta; dep_k->src[5] = nullptr; dep_k->src[6] = t_tk;
    test_set_op_i32(dep_k, 0, 34); test_set_op_i32(dep_k, 1, 0); test_set_op_i32(dep_k, 2, D);
    test_set_op_i32(dep_k, 3, Hk); test_set_op_i32(dep_k, 4, 0); test_set_op_i32(dep_k, 5, m_stride);
    test_set_op_f32(dep_k, 6, qk_clip); test_set_op_f32(dep_k, 7, eps);

    ggml_tensor * dep_v = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    dep_v->op = GGML_OP_INKLING2_OP;
    dep_v->src[0] = t_xv; dep_v->src[1] = t_wv; dep_v->src[2] = t_pc; dep_v->src[3] = t_tc;
    dep_v->src[4] = t_meta; dep_v->src[5] = nullptr; dep_v->src[6] = t_tv; dep_v->src[7] = dep_k;
    test_set_op_i32(dep_v, 0, 34); test_set_op_i32(dep_v, 1, kv_sz); test_set_op_i32(dep_v, 2, D);
    test_set_op_i32(dep_v, 3, Hk); test_set_op_i32(dep_v, 4, 1); test_set_op_i32(dep_v, 5, m_stride);
    test_set_op_f32(dep_v, 6, qk_clip); test_set_op_f32(dep_v, 7, eps);

    ggml_tensor * attn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D * Hq, T);
    attn->op = GGML_OP_INKLING2_OP;
    attn->src[0] = t_q; attn->src[1] = t_pk; attn->src[2] = t_pv; attn->src[3] = t_tk; attn->src[4] = t_tv;
    attn->src[5] = t_rt; attn->src[6] = t_meta; attn->src[7] = t_plen; attn->src[8] = t_gate; attn->src[9] = dep_v;
    test_set_op_i32(attn, 0, 35); test_set_op_i32(attn, 1, 0); test_set_op_i32(attn, 2, rel_ext);
    test_set_op_i32(attn, 3, m_stride); test_set_op_i32(attn, 4, path_w);
    ggml_set_output(attn);

    // Also verify TREE_SCONV standalone on row 0..3 with addend
    ggml_tensor * sc_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_sz, T);
    sc_out->op = GGML_OP_INKLING2_OP;
    sc_out->src[0] = t_xk; sc_out->src[1] = t_wk; sc_out->src[2] = t_pc; sc_out->src[3] = t_tc;
    sc_out->src[4] = t_meta; sc_out->src[5] = nullptr; sc_out->src[6] = t_xv;
    test_set_op_i32(sc_out, 0, 33); test_set_op_i32(sc_out, 1, 0); test_set_op_i32(sc_out, 2, 0);
    test_set_op_i32(sc_out, 3, m_stride);
    ggml_set_output(sc_out);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, attn);
    ggml_build_forward_expand(gf, sc_out);

    // Seed non-zero prefix K/V for positions [0..pref_len-1] on plane 1!
    std::vector<float> pref_k_f32((size_t) D * pref_len * Hk);
    std::vector<float> pref_v_f32((size_t) D * pref_len * Hk);
    for (float & v : pref_k_f32) v = dist(rng) * 0.5f;
    for (float & v : pref_v_f32) v = dist(rng) * 0.5f;

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(t_xk, x_k.data(), 0, x_k.size() * sizeof(float));
    ggml_backend_tensor_set(t_xv, x_v.data(), 0, x_v.size() * sizeof(float));
    ggml_backend_tensor_set(t_wk, wk.data(), 0, wk.size() * sizeof(float));
    ggml_backend_tensor_set(t_wv, wv.data(), 0, wv.size() * sizeof(float));
    ggml_backend_tensor_set(t_pc, pc.data(), 0, pc.size() * sizeof(float));
    ggml_backend_tensor_memset(t_tc, 0, 0, ggml_nbytes(t_tc));
    ggml_backend_tensor_memset(t_pk, 0, 0, ggml_nbytes(t_pk));
    ggml_backend_tensor_memset(t_pv, 0, 0, ggml_nbytes(t_pv));
    ggml_backend_tensor_memset(t_tk, 0, 0, ggml_nbytes(t_tk));
    ggml_backend_tensor_memset(t_tv, 0, 0, ggml_nbytes(t_tv));

    const size_t row_bytes = ggml_row_size(kv_t, D);
    for (int h = 0; h < Hk; ++h) {
        std::vector<uint8_t> kb((size_t) pref_len * row_bytes), vb((size_t) pref_len * row_bytes);
        if (q8_kv) {
            ggml_quantize_chunk(GGML_TYPE_Q8_0, pref_k_f32.data() + (size_t) h * pref_len * D, kb.data(), 0, pref_len, D, nullptr);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, pref_v_f32.data() + (size_t) h * pref_len * D, vb.data(), 0, pref_len, D, nullptr);
        } else {
            ggml_fp32_to_fp16_row(pref_k_f32.data() + (size_t) h * pref_len * D, reinterpret_cast<ggml_fp16_t *>(kb.data()), pref_len * D);
            ggml_fp32_to_fp16_row(pref_v_f32.data() + (size_t) h * pref_len * D, reinterpret_cast<ggml_fp16_t *>(vb.data()), pref_len * D);
        }
        const size_t off = (size_t) 1 * t_pk->nb[3] + (size_t) h * t_pk->nb[2];
        ggml_backend_tensor_set(t_pk, kb.data(), off, kb.size());
        ggml_backend_tensor_set(t_pv, vb.data(), off, vb.size());
    }

    ggml_backend_tensor_set(t_meta, meta.data(), 0, meta.size() * sizeof(int32_t));
    int32_t pl = pref_len;
    ggml_backend_tensor_set(t_plen, &pl, 0, sizeof(pl));
    ggml_backend_tensor_set(t_q, q_in.data(), 0, q_in.size() * sizeof(float));
    ggml_backend_tensor_set(t_rt, rt_in.data(), 0, rt_in.size() * sizeof(float));
    ggml_backend_tensor_set(t_gate, gate_in.data(), 0, gate_in.size() * sizeof(float));
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> got_sc((size_t) kv_sz * T);
    std::vector<float> got_attn((size_t) D * Hq * T);
    ggml_backend_tensor_get(sc_out, got_sc.data(), 0, got_sc.size() * sizeof(float));
    ggml_backend_tensor_get(attn, got_attn.data(), 0, got_attn.size() * sizeof(float));

    // Reference CPU check for TREE_SCONV + addend AND exact CPU oracle for TREE_ATTN_REL!
    const float inv_sqrt5 = 1.0f / std::sqrt(5.0f);
    std::vector<float> ref_sc((size_t) kv_sz * T);
    std::vector<float> tree_k_ref((size_t) kv_sz * T), tree_v_ref((size_t) kv_sz * T);
    auto conv_row = [&](int r, const std::vector<int> & chain_rows, const std::vector<float> & x_src,
                        const std::vector<float> & w_src, int chan_off, std::vector<float> & dst_row,
                        const float * add_row) {
        dst_row.resize(kv_sz);
        for (int c = 0; c < kv_sz; ++c) {
            const float * pbase = pc.data() + ((size_t) 1 * s_ch + chan_off + c) * 3;
            float h[3] = { pbase[0], pbase[1], pbase[2] };
            for (int cr : chain_rows) {
                h[0] = h[1]; h[1] = h[2]; h[2] = x_src[(size_t) cr * kv_sz + c];
            }
            const float cur = x_src[(size_t) r * kv_sz + c];
            const float * wc = w_src.data() + (size_t) c * 4;
            float y = (cur + wc[0] * h[0] + wc[1] * h[1] + wc[2] * h[2] + wc[3] * cur) * inv_sqrt5;
            if (add_row) y += add_row[c];
            dst_row[c] = y;
        }
    };
    const std::vector<std::vector<int>> chains = { {}, {0}, {0, 1}, {0} };
    for (int r = 0; r < T; ++r) {
        std::vector<float> tmp, tk_r, tv_r;
        conv_row(r, chains[r], x_k, wk, 0, tmp, x_v.data() + (size_t) r * kv_sz);
        std::memcpy(ref_sc.data() + (size_t) r * kv_sz, tmp.data(), (size_t) kv_sz * sizeof(float));
        conv_row(r, chains[r], x_k, wk, 0, tk_r, nullptr);
        conv_row(r, chains[r], x_v, wv, kv_sz, tv_r, nullptr);
        for (int h = 0; h < Hk; ++h) {
            float kss = 0.0f, vss = 0.0f;
            for (int d = 0; d < D; ++d) {
                kss += tk_r[h * D + d] * tk_r[h * D + d];
                vss += tv_r[h * D + d] * tv_r[h * D + d];
            }
            const float km = std::min(1.0f, qk_clip / (std::sqrt(kss) + eps));
            const float vm = 1.0f / std::sqrt(vss / (float) D + eps);
            for (int d = 0; d < D; ++d) {
                tree_k_ref[(size_t) r * kv_sz + h * D + d] = tk_r[h * D + d] * km;
                tree_v_ref[(size_t) r * kv_sz + h * D + d] = tv_r[h * D + d] * vm;
            }
        }
    }
    const float sc_err = max_abs_diff(got_sc, ref_sc);

    // Exact FP32 CPU attention reference (prefix[0..4] + tree path per row, zero-sink, rel_table, sigmoid gate)
    const std::vector<std::vector<int>> paths = { {0}, {0, 1}, {0, 1, 2}, {0, 3} };
    const float inv_sqrt128 = 1.0f / std::sqrt(128.0f);
    const int gqa = Hq / Hk;
    std::vector<float> ref_attn((size_t) D * Hq * T, 0.0f);
    for (int r = 0; r < T; ++r) {
        const int q_pos = poss[r];
        for (int h = 0; h < Hq; ++h) {
            const int kh = h / gqa;
            const float * qp = q_in.data() + ((size_t) r * Hq + h) * D;
            const float * rtp = rt_in.data() + ((size_t) r * Hq + h) * (rel_ext + 1);
            const float * gp = gate_in.data() + ((size_t) r * Hq + h) * D;

            std::vector<float> scores;
            std::vector<const float *> vptrs;
            scores.push_back(0.0f); // zero-sink virtual key
            vptrs.push_back(nullptr);
            for (int kp = 0; kp <= q_pos; ++kp) {
                const float * kptr = (kp < pref_len)
                    ? (pref_k_f32.data() + ((size_t) kh * pref_len + kp) * D)
                    : (tree_k_ref.data() + (size_t) paths[r][kp - pref_len] * kv_sz + kh * D);
                const float * vptr = (kp < pref_len)
                    ? (pref_v_f32.data() + ((size_t) kh * pref_len + kp) * D)
                    : (tree_v_ref.data() + (size_t) paths[r][kp - pref_len] * kv_sz + kh * D);
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += qp[d] * inv_sqrt128 * kptr[d];
                scores.push_back(dot + rtp[std::min(q_pos - kp, rel_ext)]);
                vptrs.push_back(vptr);
            }
            float mx = scores[0];
            for (float s : scores) mx = std::max(mx, s);
            float denom = 0.0f;
            for (float & s : scores) { s = std::exp(s - mx); denom += s; }
            float * op = ref_attn.data() + ((size_t) r * Hq + h) * D;
            for (int d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (size_t i = 0; i < scores.size(); ++i) {
                    if (vptrs[i]) acc += (scores[i] / denom) * vptrs[i][d];
                }
                op[d] = acc * (1.0f / (1.0f + std::exp(-gp[d])));
            }
        }
    }
    const float attn_err = max_abs_diff(got_attn, ref_attn);
    const float attn_tol = q8_kv ? 2.0e-2f : 2.0e-3f;

    // Verify fork isolation between row 1 (slot 5) and row 3 (slot 9)
    float fork_diff = 0.0f;
    for (int i = 0; i < D * Hq; ++i) {
        fork_diff = std::max(fork_diff, std::fabs(got_attn[(size_t) 1 * D * Hq + i] - got_attn[(size_t) 3 * D * Hq + i]));
    }

    // Now test TREE_COMMIT of slots {2, 5, 7} into prefix positions {5, 6, 7}
    ggml_tensor * cmt_op = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    cmt_op->op = GGML_OP_INKLING2_OP;
    cmt_op->src[0] = t_pk; cmt_op->src[1] = t_pv; cmt_op->src[2] = t_tk; cmt_op->src[3] = t_tv;
    cmt_op->src[4] = t_pc; cmt_op->src[5] = t_tc; cmt_op->src[6] = t_csl; cmt_op->src[7] = t_cmt;
    test_set_op_i32(cmt_op, 0, 36); test_set_op_i32(cmt_op, 1, 0);
    ggml_cgraph * gf_c = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf_c, cmt_op);
    ggml_gallocr_t ga_c = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga_c, gf_c);
    int32_t csl[4] = { 2, 5, 7, 0 };
    int32_t cmt[2] = { pref_len, 3 };
    ggml_backend_tensor_set(t_csl, csl, 0, sizeof(csl));
    ggml_backend_tensor_set(t_cmt, cmt, 0, sizeof(cmt));
    ggml_backend_graph_compute(backend, gf_c);

    // Verify prefix_conv shifted to {row 0, row 1, row 2} AND prefix_k/v match tree_k/v bit-for-bit!
    std::vector<float> got_pc(pc.size());
    ggml_backend_tensor_get(t_pc, got_pc.data(), 0, got_pc.size() * sizeof(float));
    float cmt_conv_err = 0.0f;
    for (int c = 0; c < kv_sz; ++c) {
        const float * p1 = got_pc.data() + ((size_t) 1 * s_ch + c) * 3;
        cmt_conv_err = std::max(cmt_conv_err, std::fabs(p1[0] - x_k[(size_t) 0 * kv_sz + c]));
        cmt_conv_err = std::max(cmt_conv_err, std::fabs(p1[1] - x_k[(size_t) 1 * kv_sz + c]));
        cmt_conv_err = std::max(cmt_conv_err, std::fabs(p1[2] - x_k[(size_t) 2 * kv_sz + c]));
    }
    bool cmt_kv_exact = true;
    std::vector<uint8_t> pk_row(row_bytes), tk_row(row_bytes);
    for (int i = 0; i < 3; ++i) {
        for (int h = 0; h < Hk; ++h) {
            const size_t p_off = (size_t) 1 * t_pk->nb[3] + (size_t) h * t_pk->nb[2] + (size_t) (pref_len + i) * t_pk->nb[1];
            const size_t t_off = (size_t) 1 * t_tk->nb[3] + (size_t) h * t_tk->nb[2] + (size_t) csl[i] * t_tk->nb[1];
            ggml_backend_tensor_get(t_pk, pk_row.data(), p_off, row_bytes);
            ggml_backend_tensor_get(t_tk, tk_row.data(), t_off, row_bytes);
            if (pk_row != tk_row) cmt_kv_exact = false;
        }
    }

    ggml_gallocr_free(ga_c);
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    ggml_backend_buffer_free(sbuf);
    ggml_free(sctx);

    std::printf("[test_inkling2_ops] 2dsd_tree(%s) sconv_err=%.3e attn_err=%.3e (tol=%.1e) cmt_conv_err=%.3e cmt_kv=%s fork_diff=%.3f\n",
                q8_kv ? "Q8_0" : "F16", sc_err, attn_err, attn_tol, cmt_conv_err, cmt_kv_exact ? "EXACT" : "DIFF", fork_diff);
    return sc_err < 1e-5f && attn_err < attn_tol && cmt_conv_err < 1e-6f && cmt_kv_exact && fork_diff > 1e-4f;
}

// ─── 2DSD Unit Test 2: Flagged Loop Controller + Padding Router + Vanilla Markov + Conf + Mask ──
static bool test_2dsd_drafter_and_flags(ggml_backend_t backend) {
    const int H = 256, R = 256, V = 4096, L = 3, B = 8, S_b = 8, pref_len = 13;
    const int kv_span = 64;
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 1. Batched Vanilla Markov (Q8_0) + Confidence Head + Draft Mask
    std::vector<float> base((size_t) V * L), w1_f32((size_t) R * V), w2_f32((size_t) R * V);
    std::vector<float> h_spec((size_t) H * L), conf_w((size_t) (H + R)), conf_b = { -0.25f };
    for (float & v : base) v = dist(rng);
    for (float & v : w1_f32) v = dist(rng) * 0.5f;
    for (float & v : w2_f32) v = dist(rng) * 0.5f;
    for (float & v : h_spec) v = dist(rng);
    for (float & v : conf_w) v = dist(rng) * 0.1f;

    std::vector<ggml_fp16_t> w1_f16(w1_f32.size());
    for (size_t i = 0; i < w1_f32.size(); ++i) w1_f16[i] = ggml_fp32_to_fp16(w1_f32[i]);
    const size_t q8_bytes = (size_t) V * ggml_row_size(GGML_TYPE_Q8_0, R);
    std::vector<uint8_t> w2_q8(q8_bytes);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, w2_f32.data(), w2_q8.data(), 0, V, R, nullptr);

    // Dequantize w2_q8 for exact CPU oracle
    std::vector<float> w2_deq((size_t) R * V);
    const auto * traits_q8 = ggml_get_type_traits(GGML_TYPE_Q8_0);
    const size_t q8_row = ggml_row_size(GGML_TYPE_Q8_0, R);
    for (int v = 0; v < V; ++v) traits_q8->to_float(w2_q8.data() + (size_t) v * q8_row, w2_deq.data() + (size_t) v * R, R);

    int32_t prev_tok[L] = { 17, 1024, 4090 };
    int32_t ref_tok[L] = {};
    float   ref_conf[L] = {};
    for (int l = 0; l < L; ++l) {
        float best_v = -1e30f;
        int32_t best_i = 0;
        for (int v = 0; v < V; ++v) {
            float dot = 0.0f;
            for (int r = 0; r < R; ++r) {
                dot += ggml_fp16_to_fp32(w1_f16[(size_t) prev_tok[l] * R + r]) * w2_deq[(size_t) v * R + r];
            }
            const float sc = base[(size_t) l * V + v] + dot;
            if (sc > best_v || (sc == best_v && v < best_i)) { best_v = sc; best_i = v; }
        }
        ref_tok[l] = best_i;

        float acc = conf_b[0];
        for (int i = 0; i < H; ++i) acc += h_spec[(size_t) l * H + i] * conf_w[i];
        for (int r = 0; r < R; ++r) acc += ggml_fp16_to_fp32(w1_f16[(size_t) prev_tok[l] * R + r]) * conf_w[H + r];
        ref_conf[l] = 1.0f / (1.0f + std::exp(-acc));
    }

    // 2. Flagged LOOP_CTRL_POST (rows 0,2 pass-0 => FLAG_LATER=0; rows 1,3 pass-1 => FLAG_LATER=1)
    const int T_f = 4;
    std::vector<float> cur_f((size_t) H * T_f), msg_f((size_t) H * T_f), gf_f((size_t) 2 * H * T_f), nw_f(H);
    for (float & v : cur_f) v = dist(rng);
    for (float & v : msg_f) v = dist(rng);
    for (float & v : gf_f)  v = dist(rng) * 0.5f;
    for (float & v : nw_f)  v = std::fabs(dist(rng)) + 0.5f;
    int32_t lc_flags[T_f] = { 0, 1, 0, 1 };

    struct ggml_init_params gp = { 64 * ggml_tensor_overhead() + ggml_graph_overhead() + (1 << 20), nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * t_base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, V, L);
    ggml_tensor * t_prev = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
    ggml_tensor * t_w1   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, R, V);
    ggml_tensor * t_w2   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, R, V);
    ggml_tensor * t_hs   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, L);
    ggml_tensor * t_cw   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H + R);
    ggml_tensor * t_cb   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * t_dm   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3 + L);
    ggml_tensor * t_cur  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T_f);
    ggml_tensor * t_msg  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T_f);
    ggml_tensor * t_gf   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * H, T_f);
    ggml_tensor * t_nw   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_tensor * t_flg  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_f);
    // 3. Padding-aware FUSED_ROUTER_Q8_TOPK (rows 2,3 flagged as padding -> copy row 0!)
    const int n_exp = 128, top_k = 8;
    std::vector<float> r_logits((size_t) n_exp * T_f), r_bias(n_exp, 0.0f);
    for (float & v : r_logits) v = dist(rng);
    int32_t r_flags[T_f] = { 0, 0, 2, 2 };
    ggml_tensor * t_rl   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_exp, T_f);
    ggml_tensor * t_rb   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_exp);
    ggml_tensor * t_rflg = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_f);
    ggml_set_input(t_base); ggml_set_input(t_prev); ggml_set_input(t_w1); ggml_set_input(t_w2);
    ggml_set_input(t_hs); ggml_set_input(t_cw); ggml_set_input(t_cb); ggml_set_input(t_dm);
    ggml_set_input(t_cur); ggml_set_input(t_msg); ggml_set_input(t_gf); ggml_set_input(t_nw); ggml_set_input(t_flg);
    ggml_set_input(t_rl); ggml_set_input(t_rb); ggml_set_input(t_rflg);

    ggml_tensor * out_tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
    out_tok->op = GGML_OP_INKLING2_OP;
    out_tok->src[0] = t_base; out_tok->src[1] = t_prev; out_tok->src[2] = t_w1; out_tok->src[3] = t_w2;
    test_set_op_i32(out_tok, 0, 37); test_set_op_i32(out_tok, 1, V);

    ggml_tensor * out_conf = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, L);
    out_conf->op = GGML_OP_INKLING2_OP;
    out_conf->src[0] = t_hs; out_conf->src[1] = t_prev; out_conf->src[2] = t_w1; out_conf->src[3] = t_cw; out_conf->src[4] = t_cb;
    test_set_op_i32(out_conf, 0, 38);

    ggml_tensor * out_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_span, B * L);
    out_mask->op = GGML_OP_INKLING2_OP;
    out_mask->src[0] = t_dm;
    test_set_op_i32(out_mask, 0, 39); test_set_op_i32(out_mask, 1, B);

    ggml_tensor * out_lc = ggml_inkling2_loop_ctrl_post(ctx, t_cur, t_msg, t_gf, t_nw, 1e-6f);
    out_lc->src[4] = t_flg;

    ggml_tensor * out_rt = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, top_k, T_f, 3);
    out_rt->op = GGML_OP_INKLING2_OP;
    out_rt->src[0] = t_rl; out_rt->src[1] = nullptr; out_rt->src[2] = t_rb; out_rt->src[3] = t_rflg;
    test_set_op_i32(out_rt, 0, 24); test_set_op_i32(out_rt, 1, top_k);
    test_set_op_f32(out_rt, 2, 1.0f); test_set_op_f32(out_rt, 3, 1e-20f); test_set_op_i32(out_rt, 4, 74);

    ggml_set_output(out_tok); ggml_set_output(out_conf); ggml_set_output(out_mask); ggml_set_output(out_lc); ggml_set_output(out_rt);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_tok);
    ggml_build_forward_expand(gf, out_conf);
    ggml_build_forward_expand(gf, out_mask);
    ggml_build_forward_expand(gf, out_lc);
    ggml_build_forward_expand(gf, out_rt);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(t_base, base.data(), 0, base.size() * sizeof(float));
    ggml_backend_tensor_set(t_prev, prev_tok, 0, sizeof(prev_tok));
    ggml_backend_tensor_set(t_w1, w1_f16.data(), 0, w1_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(t_w2, w2_q8.data(), 0, w2_q8.size());
    ggml_backend_tensor_set(t_hs, h_spec.data(), 0, h_spec.size() * sizeof(float));
    ggml_backend_tensor_set(t_cw, conf_w.data(), 0, conf_w.size() * sizeof(float));
    ggml_backend_tensor_set(t_cb, conf_b.data(), 0, sizeof(float));
    int32_t dmeta[3 + L] = { pref_len, L, S_b, 5, 0, 3 };
    ggml_backend_tensor_set(t_dm, dmeta, 0, sizeof(dmeta));
    ggml_backend_tensor_set(t_cur, cur_f.data(), 0, cur_f.size() * sizeof(float));
    ggml_backend_tensor_set(t_msg, msg_f.data(), 0, msg_f.size() * sizeof(float));
    ggml_backend_tensor_set(t_gf,  gf_f.data(),  0, gf_f.size() * sizeof(float));
    ggml_backend_tensor_set(t_nw,  nw_f.data(),  0, nw_f.size() * sizeof(float));
    ggml_backend_tensor_set(t_flg, lc_flags, 0, sizeof(lc_flags));
    ggml_backend_tensor_set(t_rl,  r_logits.data(), 0, r_logits.size() * sizeof(float));
    ggml_backend_tensor_set(t_rb,  r_bias.data(),   0, r_bias.size() * sizeof(float));
    ggml_backend_tensor_set(t_rflg, r_flags, 0, sizeof(r_flags));
    ggml_backend_graph_compute(backend, gf);

    int32_t got_tok[L] = {};
    float got_conf[L] = {};
    std::vector<ggml_fp16_t> got_mask((size_t) kv_span * B * L);
    std::vector<float> got_lc((size_t) H * T_f);
    std::vector<int32_t> got_rt((size_t) top_k * T_f * 3);
    ggml_backend_tensor_get(out_tok,  got_tok,  0, sizeof(got_tok));
    ggml_backend_tensor_get(out_conf, got_conf, 0, sizeof(got_conf));
    ggml_backend_tensor_get(out_mask, got_mask.data(), 0, got_mask.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(out_lc,   got_lc.data(),   0, got_lc.size() * sizeof(float));
    ggml_backend_tensor_get(out_rt,   got_rt.data(),   0, got_rt.size() * sizeof(int32_t));
    ggml_gallocr_free(ga);
    ggml_free(ctx);

    bool pad_rt_ok = true;
    for (int t = 2; t < T_f; ++t) {
        for (int k = 0; k < top_k; ++k) {
            if (got_rt[(size_t) t * top_k + k] != got_rt[k]) pad_rt_ok = false;
            if (got_rt[(size_t) (T_f + t) * top_k + k] != got_rt[(size_t) T_f * top_k + k]) pad_rt_ok = false;
        }
    }

    bool tok_ok = true;
    float conf_err = 0.0f;
    for (int l = 0; l < L; ++l) {
        if (got_tok[l] != ref_tok[l]) tok_ok = false;
        conf_err = std::max(conf_err, std::fabs(got_conf[l] - ref_conf[l]));
    }
    bool mask_ok = true;
    for (int l = 0; l < L; ++l) {
        const int slen = dmeta[3 + l];
        for (int k = 0; k < kv_span; ++k) {
            const bool exp_vis = (k < pref_len) ||
                                 (k >= pref_len + l * S_b && k < pref_len + l * S_b + slen) ||
                                 (k >= pref_len + L * S_b + l * B && k < pref_len + L * S_b + (l + 1) * B);
            const float mv = ggml_fp16_to_fp32(got_mask[(size_t) (l * B) * kv_span + k]);
            if (exp_vis && mv != 0.0f) mask_ok = false;
            if (!exp_vis && !std::isinf(mv)) mask_ok = false;
        }
    }
    // Check flagged loop_ctrl_post row 0 (pass-0: pure RMSNorm) and row 1 (pass-1: full blend + RMSNorm)
    float lc_err = 0.0f;
    for (int t = 0; t < T_f; ++t) {
        std::vector<float> y(H);
        float ss = 0.0f;
        for (int i = 0; i < H; ++i) {
            const float c = cur_f[(size_t) t * H + i];
            float v = c;
            if (lc_flags[t] & 1) {
                const float m = msg_f[(size_t) t * H + i];
                const float g = std::tanh(gf_f[(size_t) t * 2 * H + i]);
                const float f = std::tanh(gf_f[(size_t) t * 2 * H + H + i]);
                v = (c - f * c) + g * m;
            }
            y[i] = v;
            ss += v * v;
        }
        const float irms = 1.0f / std::sqrt(ss / (float) H + 1e-6f);
        for (int i = 0; i < H; ++i) {
            lc_err = std::max(lc_err, std::fabs(got_lc[(size_t) t * H + i] - y[i] * irms * nw_f[i]));
        }
    }

    std::printf("[test_inkling2_ops] 2dsd_markov_conf_mask: tok_ok=%d conf_err=%.3e mask_ok=%d lc_err=%.3e pad_rt=%d\n",
                tok_ok ? 1 : 0, conf_err, mask_ok ? 1 : 0, lc_err, pad_rt_ok ? 1 : 0);
    return tok_ok && conf_err < 1e-5f && mask_ok && lc_err < 1e-5f && pad_rt_ok;
}

// ─── 2DSD Unit Test 3: Direct Uncompressed PyTorch `.pt` ZIP Loader ──────
static bool test_2dsd_mask_embedding_pt() {
    namespace fs = std::filesystem;
    const fs::path tmp_dir = fs::temp_directory_path();
    const fs::path path_lfh = tmp_dir / "test_luce_mask_emb_lfh.pt";
    const fs::path path_cd  = tmp_dir / "test_luce_mask_emb_cd.pt";
    const int H = 16;
    std::vector<uint16_t> bf16(H);
    std::vector<float> ref(H);
    for (int i = 0; i < H; ++i) {
        float v = (float) (i - 7) * 0.125f;
        uint32_t u = 0;
        std::memcpy(&u, &v, sizeof(u));
        bf16[i] = (uint16_t) (u >> 16);
        uint32_t back = (uint32_t) bf16[i] << 16;
        std::memcpy(&ref[i], &back, sizeof(back));
    }
    const std::string pkl = "torch BFloat16Storage";
    auto push16 = [](std::vector<uint8_t> & buf, uint16_t v) {
        buf.push_back(v & 0xFF); buf.push_back((v >> 8) & 0xFF);
    };
    auto push32 = [](std::vector<uint8_t> & buf, uint32_t v) {
        for (int b = 0; b < 4; ++b) buf.push_back((v >> (8 * b)) & 0xFF);
    };
    auto write_lfh = [&](std::vector<uint8_t> & buf, const std::string & name, const void * data, uint32_t len, bool data_desc) {
        const uint32_t off = (uint32_t) buf.size();
        push32(buf, 0x04034b50u);
        push16(buf, 20);
        push16(buf, data_desc ? 0x0808u : 0u);
        push16(buf, 0); push16(buf, 0); push16(buf, 0);
        push32(buf, 0);
        push32(buf, data_desc ? 0u : len);
        push32(buf, data_desc ? 0u : len);
        push16(buf, (uint16_t) name.size()); push16(buf, 0);
        buf.insert(buf.end(), name.begin(), name.end());
        const auto * p = static_cast<const uint8_t *>(data);
        buf.insert(buf.end(), p, p + len);
        if (data_desc) {
            push32(buf, 0x08074b50u); push32(buf, 0); push32(buf, len); push32(buf, len);
        }
        return off;
    };
    // Case A: local-header-only ZIP (fallback path)
    {
        std::vector<uint8_t> zip;
        write_lfh(zip, "mask_embedding/data.pkl", pkl.data(), (uint32_t) pkl.size(), false);
        write_lfh(zip, "mask_embedding/data/0",   bf16.data(), (uint32_t) (H * sizeof(uint16_t)), false);
        std::ofstream ofs(path_lfh, std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(zip.data()), (std::streamsize) zip.size());
    }
    // Case B: PyTorch torch.save style ZIP with data-descriptor flag 0x0808 + Central Directory + EOCD
    {
        std::vector<uint8_t> zip;
        struct Rec { std::string name; uint32_t off; uint32_t len; };
        std::vector<Rec> recs;
        recs.push_back({ "mask_embedding/data.pkl", write_lfh(zip, "mask_embedding/data.pkl", pkl.data(), (uint32_t) pkl.size(), true), (uint32_t) pkl.size() });
        recs.push_back({ "mask_embedding/data/0",   write_lfh(zip, "mask_embedding/data/0",   bf16.data(), (uint32_t) (H * sizeof(uint16_t)), true), (uint32_t) (H * sizeof(uint16_t)) });
        const uint32_t cd_off = (uint32_t) zip.size();
        for (const auto & r : recs) {
            push32(zip, 0x02014b50u);
            push16(zip, 20); push16(zip, 20); push16(zip, 0x0808u); push16(zip, 0);
            push16(zip, 0); push16(zip, 0);
            push32(zip, 0); push32(zip, r.len); push32(zip, r.len);
            push16(zip, (uint16_t) r.name.size()); push16(zip, 0); push16(zip, 0);
            push16(zip, 0); push16(zip, 0); push32(zip, 0);
            push32(zip, r.off);
            zip.insert(zip.end(), r.name.begin(), r.name.end());
        }
        const uint32_t cd_sz = (uint32_t) zip.size() - cd_off;
        push32(zip, 0x06054b50u);
        push16(zip, 0); push16(zip, 0);
        push16(zip, (uint16_t) recs.size()); push16(zip, (uint16_t) recs.size());
        push32(zip, cd_sz); push32(zip, cd_off); push16(zip, 0);
        std::ofstream ofs(path_cd, std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(zip.data()), (std::streamsize) zip.size());
    }

    std::vector<float> got_lfh, got_cd;
    std::string err1, err2;
    const bool ok_lfh = luce::common::load_inkling2_mask_embedding_pt(path_lfh.string(), H, got_lfh, err1);
    const bool ok_cd  = luce::common::load_inkling2_mask_embedding_pt(path_cd.string(),  H, got_cd,  err2);
    const float diff_lfh = ok_lfh ? max_abs_diff(got_lfh, ref) : 1.0f;
    const float diff_cd  = ok_cd  ? max_abs_diff(got_cd,  ref) : 1.0f;
    std::error_code ec;
    fs::remove(path_lfh, ec);
    fs::remove(path_cd, ec);
    std::printf("[test_inkling2_ops] mask_embedding_pt: lfh_ok=%d err=%.3e cd_ok=%d err=%.3e\n",
                ok_lfh ? 1 : 0, diff_lfh, ok_cd ? 1 : 0, diff_cd);
    return ok_lfh && diff_lfh == 0.0f && ok_cd && diff_cd == 0.0f;
}

// ─── 2DSD Unit Test 4: Deterministic CPU Oracle for Inkling2TwoDSDExecutor ──
static bool test_2dsd_executor() {
    using namespace luce::common;
    auto hash_seq = [](const std::vector<int32_t> & seq, int salt) -> uint64_t {
        uint64_t h = 1469598103934665603ULL ^ (uint64_t) salt * 0x9e3779b97f4a7c15ULL;
        for (int32_t t : seq) {
            h ^= (uint64_t) (uint32_t) t;
            h *= 1099511628211ULL;
        }
        return h;
    };
    auto oracle_token = [&](const std::vector<int32_t> & seq, int depth) -> int32_t {
        const uint64_t h3 = hash_seq(seq, 3);
        const int32_t tok3 = (int32_t) (h3 % 64) + 1;
        if (depth == 3) return tok3;
        // Simulate realistic varloop early-exit agreement (~72% at d1=1, ~88% at d2=2)
        const uint64_t hd = hash_seq(seq, depth * 17);
        if (depth == 1 && (hd % 100) < 28) return (int32_t) ((hd / 100) % 64) + 1;
        if (depth == 2 && (hd % 100) < 12) return (int32_t) ((hd / 100) % 64) + 1;
        return tok3;
    };

    const std::vector<int32_t> prompt = { 10, 20, 30, 40 };
    const int n_gen = 96;
    std::vector<int32_t> ref_tokens;
    {
        std::vector<int32_t> hist = prompt;
        for (int i = 0; i < n_gen; ++i) {
            int32_t tok = oracle_token(hist, 3);
            ref_tokens.push_back(tok);
            hist.push_back(tok);
        }
    }

    class MockRunner final : public Inkling2TwoDSDRunner {
    public:
        MockRunner(std::vector<int32_t> pref, std::function<int32_t(const std::vector<int32_t> &, int)> ora)
            : prefix_(std::move(pref)), oracle_(std::move(ora)) {}
        int prefix_length() const override { return (int) prefix_.size(); }
        int max_nodes() const override { return 128; }
        int max_ctx() const override { return 1024; }
        bool advance_nodes(const std::vector<Inkling2TwoDSDNodeDesc> & nodes,
                           uint32_t readout_mask,
                           std::vector<int> & out_rows,
                           std::vector<int32_t> & out_argmax) override {
            out_rows.clear(); out_argmax.clear();
            for (int i = 0; i < (int) nodes.size(); ++i) {
                const auto & nd = nodes[i];
                slot_tok_[nd.slot] = nd.token;
                const int new_depth = nd.depth + 1;
                if (readout_mask & (1u << new_depth)) {
                    std::vector<int32_t> seq = prefix_;
                    for (int as : nd.ancestors) seq.push_back(slot_tok_.at(as));
                    seq.push_back(nd.token);
                    out_rows.push_back(i);
                    out_argmax.push_back(oracle_(seq, new_depth));
                }
            }
            return true;
        }
        bool commit_nodes(const int * slots, int n) override {
            for (int i = 0; i < n; ++i) prefix_.push_back(slot_tok_.at(slots[i]));
            return true;
        }
        bool draft_batch(const std::vector<Inkling2TwoDSDDraftReq> & reqs,
                         std::vector<Inkling2TwoDSDDraftOut> & out) override {
            out.resize(reqs.size());
            for (size_t i = 0; i < reqs.size(); ++i) {
                std::vector<int32_t> seq = prefix_;
                for (int as : reqs[i].span_slots) seq.push_back(slot_tok_.at(as));
                seq.push_back(reqs[i].head_token);
                out[i].tokens.resize(7);
                out[i].confidence.resize(7);
                for (int s = 0; s < 7; ++s) {
                    int32_t truth = oracle_(seq, 3);
                    // Draft matches truth on ~75% of early slots, decaying with slot index
                    const bool hit = ((truth + s * 7 + reqs[i].depth) % 10) < (8 - s / 2);
                    int32_t dt = hit ? truth : ((truth % 64) + 1);
                    out[i].tokens[s] = dt;
                    out[i].confidence[s] = hit ? (0.85f - 0.08f * s) : (0.35f - 0.03f * s);
                    seq.push_back(dt);
                }
            }
            return true;
        }
    private:
        std::vector<int32_t> prefix_;
        std::function<int32_t(const std::vector<int32_t> &, int)> oracle_;
        std::unordered_map<int, int32_t> slot_tok_;
    };

    struct Case { Inkling2TwoDSDStage st; float conf; const char * name; };
    const Case cases[] = {
        { Inkling2TwoDSDStage::Stage1,  0.0f,  "stage1" },
        { Inkling2TwoDSDStage::Stage2a, 0.0f,  "stage2a" },
        { Inkling2TwoDSDStage::Stage2b, 0.0f,  "stage2b" },
        { Inkling2TwoDSDStage::Stage2b, 0.50f, "stage2b_conf0.50" },
    };
    bool all_ok = true;
    for (const auto & cs : cases) {
        MockRunner runner(prompt, oracle_token);
        Inkling2TwoDSDConfig cfg;
        cfg.stage = cs.st;
        cfg.R = 3; cfg.d1 = 1; cfg.d2 = 2; cfg.k = 8;
        cfg.draft_confidence = cs.conf;
        std::vector<int32_t> got;
        Inkling2TwoDSDStats st;
        std::string err;
        const bool ok = inkling2_2dsd_run_greedy(
            runner, cfg, ref_tokens[0], n_gen,
            [&](int32_t & tok) -> bool {
                got.push_back(tok);
                return (int) got.size() < n_gen;
            },
            st, err);
        const bool exact = ok && (got == ref_tokens);
        const bool conf_cut_ok = (cs.conf <= 0.0f) || (st.draft_slots_cut > 0);
        std::printf("[test_inkling2_ops] 2dsd_exec(%s): exact=%d calls=%lld promo=%lld forks=%lld cut=%lld\n",
                    cs.name, exact ? 1 : 0, st.target_calls, st.promotions, st.forks, st.draft_slots_cut);
        if (!exact || !conf_cut_ok) all_ok = false;
    }
    return all_ok;
}

// ─── 2DSD Unit Test 5: P0 Fused Kernels (subops 40, 41, 42) ──────────────
static bool test_2dsd_p0_fused_ops(ggml_backend_t backend) {
    // 1. Subop 40: 2DSD_MARKOV_TOPK_STEP (paired with subop 18) for both Q8_0 and F16 markov_w2
    bool ok_40 = true;
    {
        const int R = 256, V = 4096, L = 4;
        std::mt19937 rng(4040);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> base((size_t) V * L), w1_f32((size_t) R * V), w2_f32((size_t) R * V);
        for (float & v : base)   v = dist(rng);
        for (float & v : w1_f32) v = dist(rng) * 0.4f;
        for (float & v : w2_f32) v = dist(rng) * 0.4f;
        // Inject sharp peaks in each leaf so top-1024 exact match against full-vocab is guaranteed
        for (int l = 0; l < L; ++l) {
            base[(size_t) l * V + (317 * (l + 1) % V)] += 10.0f;
            base[(size_t) l * V + (1993 * (l + 1) % V)] += 9.5f;
        }
        std::vector<ggml_fp16_t> w1_f16(w1_f32.size()), w2_f16(w2_f32.size());
        for (size_t i = 0; i < w1_f32.size(); ++i) {
            w1_f16[i] = ggml_fp32_to_fp16(w1_f32[i]);
            w2_f16[i] = ggml_fp32_to_fp16(w2_f32[i]);
        }
        std::vector<uint8_t> w2_q8((size_t) V * ggml_row_size(GGML_TYPE_Q8_0, R));
        ggml_quantize_chunk(GGML_TYPE_Q8_0, w2_f32.data(), w2_q8.data(), 0, V, R, nullptr);
        int32_t prev_tok[L] = { 5, 1021, 2048, 4091 };

        for (int use_q8 = 0; use_q8 <= 1; ++use_q8) {
            struct ggml_init_params gp = { 32 * ggml_tensor_overhead() + ggml_graph_overhead() + (1 << 20), nullptr, true };
            ggml_context * ctx = ggml_init(gp);
            ggml_tensor * t_base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, V, L);
            ggml_tensor * t_prev = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
            ggml_tensor * t_w1   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, R, V);
            ggml_tensor * t_w2   = ggml_new_tensor_2d(ctx, use_q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16, R, V);
            ggml_set_input(t_base); ggml_set_input(t_prev); ggml_set_input(t_w1); ggml_set_input(t_w2);

            ggml_tensor * cands = ggml_inkling2_dspark_topk_cands(ctx, t_base, V);
            ggml_tensor * out_fast = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
            out_fast->op = GGML_OP_INKLING2_OP;
            out_fast->src[0] = cands; out_fast->src[1] = t_base; out_fast->src[2] = t_prev;
            out_fast->src[3] = t_w1;  out_fast->src[4] = t_w2;
            test_set_op_i32(out_fast, 0, 40); test_set_op_i32(out_fast, 1, V);

            ggml_tensor * out_full = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
            out_full->op = GGML_OP_INKLING2_OP;
            out_full->src[0] = t_base; out_full->src[1] = t_prev; out_full->src[2] = t_w1; out_full->src[3] = t_w2;
            test_set_op_i32(out_full, 0, 37); test_set_op_i32(out_full, 1, V);

            ggml_set_output(out_fast); ggml_set_output(out_full);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out_fast);
            ggml_build_forward_expand(gf, out_full);

            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            ggml_gallocr_alloc_graph(ga, gf);
            ggml_backend_tensor_set(t_base, base.data(), 0, base.size() * sizeof(float));
            ggml_backend_tensor_set(t_prev, prev_tok, 0, sizeof(prev_tok));
            ggml_backend_tensor_set(t_w1, w1_f16.data(), 0, w1_f16.size() * sizeof(ggml_fp16_t));
            if (use_q8) ggml_backend_tensor_set(t_w2, w2_q8.data(), 0, w2_q8.size());
            else        ggml_backend_tensor_set(t_w2, w2_f16.data(), 0, w2_f16.size() * sizeof(ggml_fp16_t));
            ggml_backend_graph_compute(backend, gf);

            int32_t got_fast[L] = {}, got_full[L] = {};
            ggml_backend_tensor_get(out_fast, got_fast, 0, sizeof(got_fast));
            ggml_backend_tensor_get(out_full, got_full, 0, sizeof(got_full));
            for (int l = 0; l < L; ++l) {
                if (got_fast[l] != got_full[l]) ok_40 = false;
            }
            ggml_gallocr_free(ga);
            ggml_free(ctx);
        }
    }

    // 2. Subop 41: TREE_SCONV_KV_WRITE_KV (compare bit-exact vs two subop 34 calls for F16 & Q8_0)
    bool ok_41 = true;
    for (int use_q8 = 0; use_q8 <= 1; ++use_q8) {
        const int D = 128, Hk = 4, kv_sz = D * Hk, H = 512, s_ch = 2 * kv_sz + 2 * H;
        const int P = 3, max_nodes = 16, s_max = max_nodes + 1, T = 4, path_w = 8, m_stride = 11 + path_w;
        const ggml_type kv_t = use_q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
        std::mt19937 rng(4141 + use_q8);
        std::uniform_real_distribution<float> dist(-0.8f, 0.8f);

        std::vector<float> x_k((size_t) kv_sz * T), x_v((size_t) kv_sz * T), wk((size_t) 4 * kv_sz), wv((size_t) 4 * kv_sz);
        std::vector<float> pc((size_t) 3 * s_ch * P, 0.0f);
        for (float & v : x_k) v = dist(rng);
        for (float & v : x_v) v = dist(rng);
        for (float & v : wk)  v = dist(rng) * 0.25f;
        for (float & v : wv)  v = dist(rng) * 0.25f;
        for (float & v : pc)  v = dist(rng) * 0.2f;

        std::vector<int32_t> meta((size_t) T * m_stride, -1);
        const int slots[T] = { 2, 5, 7, 9 };
        const int hrow[T][3]  = { {-1,-1,-1}, {-1,-1,0}, {-1,0,1}, {-1,-1,0} };
        const int hslot[T][3] = { {-1,-2,-3}, {-2,-3,2}, {-3,2,5}, {-2,-3,2} };
        for (int r = 0; r < T; ++r) {
            int32_t * mt = meta.data() + (size_t) r * m_stride;
            mt[0] = slots[r]; mt[1] = 10 + r; mt[2] = 1; mt[3] = 0; mt[4] = 0;
            for (int j = 0; j < 3; ++j) { mt[5 + j] = hrow[r][j]; mt[8 + j] = hslot[r][j]; }
            mt[11] = slots[r];
        }

        struct ggml_init_params sp = { 16 * ggml_tensor_overhead() + 4096, nullptr, true };
        ggml_context * sctx = ggml_init(sp);
        ggml_tensor * tk_a = ggml_new_tensor_4d(sctx, kv_t, D, s_max, Hk, P);
        ggml_tensor * tv_a = ggml_new_tensor_4d(sctx, kv_t, D, s_max, Hk, P);
        ggml_tensor * tc_a = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, s_ch, s_max, P);
        ggml_tensor * tk_b = ggml_new_tensor_4d(sctx, kv_t, D, s_max, Hk, P);
        ggml_tensor * tv_b = ggml_new_tensor_4d(sctx, kv_t, D, s_max, Hk, P);
        ggml_tensor * tc_b = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, s_ch, s_max, P);
        ggml_tensor * t_pc = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, 3, s_ch, P);
        ggml_backend_buffer_t sbuf = ggml_backend_alloc_ctx_tensors(sctx, backend);
        ggml_backend_tensor_memset(tk_a, 0, 0, ggml_nbytes(tk_a));
        ggml_backend_tensor_memset(tv_a, 0, 0, ggml_nbytes(tv_a));
        ggml_backend_tensor_memset(tc_a, 0, 0, ggml_nbytes(tc_a));
        ggml_backend_tensor_memset(tk_b, 0, 0, ggml_nbytes(tk_b));
        ggml_backend_tensor_memset(tv_b, 0, 0, ggml_nbytes(tv_b));
        ggml_backend_tensor_memset(tc_b, 0, 0, ggml_nbytes(tc_b));
        ggml_backend_tensor_set(t_pc, pc.data(), 0, pc.size() * sizeof(float));

        struct ggml_init_params gp = { 32 * ggml_tensor_overhead() + ggml_graph_overhead() + (1 << 20), nullptr, true };
        ggml_context * ctx = ggml_init(gp);
        ggml_tensor * t_xk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_sz, T);
        ggml_tensor * t_xv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_sz, T);
        ggml_tensor * t_wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, kv_sz);
        ggml_tensor * t_wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, kv_sz);
        ggml_tensor * t_mt = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, m_stride, T);
        ggml_set_input(t_xk); ggml_set_input(t_xv); ggml_set_input(t_wk); ggml_set_input(t_wv); ggml_set_input(t_mt);

        // Reference: two subop 34 calls -> (tk_a, tv_a, tc_a)
        ggml_tensor * ref_k = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ref_k->op = GGML_OP_INKLING2_OP;
        ref_k->src[0] = t_xk; ref_k->src[1] = t_wk; ref_k->src[2] = t_pc; ref_k->src[3] = tc_a;
        ref_k->src[4] = t_mt; ref_k->src[5] = nullptr; ref_k->src[6] = tk_a; ref_k->src[7] = nullptr;
        test_set_op_i32(ref_k, 0, 34); test_set_op_i32(ref_k, 1, 0); test_set_op_i32(ref_k, 2, D);
        test_set_op_i32(ref_k, 3, Hk); test_set_op_i32(ref_k, 4, 0); test_set_op_i32(ref_k, 5, m_stride);
        test_set_op_f32(ref_k, 6, 4.0f); test_set_op_f32(ref_k, 7, 1e-6f);

        ggml_tensor * ref_v = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ref_v->op = GGML_OP_INKLING2_OP;
        ref_v->src[0] = t_xv; ref_v->src[1] = t_wv; ref_v->src[2] = t_pc; ref_v->src[3] = tc_a;
        ref_v->src[4] = t_mt; ref_v->src[5] = nullptr; ref_v->src[6] = tv_a; ref_v->src[7] = ref_k;
        test_set_op_i32(ref_v, 0, 34); test_set_op_i32(ref_v, 1, kv_sz); test_set_op_i32(ref_v, 2, D);
        test_set_op_i32(ref_v, 3, Hk); test_set_op_i32(ref_v, 4, 1); test_set_op_i32(ref_v, 5, m_stride);
        test_set_op_f32(ref_v, 6, 4.0f); test_set_op_f32(ref_v, 7, 1e-6f);

        // Fused subop 41 -> (tk_b, tv_b, tc_b)
        ggml_tensor * fused_kv = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        fused_kv->op = GGML_OP_INKLING2_OP;
        fused_kv->src[0] = t_xk; fused_kv->src[1] = t_xv; fused_kv->src[2] = t_wk; fused_kv->src[3] = t_wv;
        fused_kv->src[4] = t_pc; fused_kv->src[5] = tc_b; fused_kv->src[6] = t_mt; fused_kv->src[7] = tk_b; fused_kv->src[8] = tv_b;
        test_set_op_i32(fused_kv, 0, 41); test_set_op_i32(fused_kv, 1, D); test_set_op_i32(fused_kv, 2, Hk);
        test_set_op_i32(fused_kv, 3, m_stride); test_set_op_f32(fused_kv, 4, 4.0f); test_set_op_f32(fused_kv, 5, 1e-6f);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ref_v);
        ggml_build_forward_expand(gf, fused_kv);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_tensor_set(t_xk, x_k.data(), 0, x_k.size() * sizeof(float));
        ggml_backend_tensor_set(t_xv, x_v.data(), 0, x_v.size() * sizeof(float));
        ggml_backend_tensor_set(t_wk, wk.data(),  0, wk.size()  * sizeof(float));
        ggml_backend_tensor_set(t_wv, wv.data(),  0, wv.size()  * sizeof(float));
        ggml_backend_tensor_set(t_mt, meta.data(), 0, meta.size() * sizeof(int32_t));
        ggml_backend_graph_compute(backend, gf);

        std::vector<uint8_t> ba(ggml_nbytes(tk_a)), bb(ggml_nbytes(tk_b));
        ggml_backend_tensor_get(tk_a, ba.data(), 0, ba.size());
        ggml_backend_tensor_get(tk_b, bb.data(), 0, bb.size());
        if (ba != bb) ok_41 = false;
        ggml_backend_tensor_get(tv_a, ba.data(), 0, ba.size());
        ggml_backend_tensor_get(tv_b, bb.data(), 0, bb.size());
        if (ba != bb) ok_41 = false;
        std::vector<float> ca(ggml_nelements(tc_a)), cb(ggml_nelements(tc_b));
        ggml_backend_tensor_get(tc_a, ca.data(), 0, ggml_nbytes(tc_a));
        ggml_backend_tensor_get(tc_b, cb.data(), 0, ggml_nbytes(tc_b));
        if (max_abs_diff(ca, cb) > 1e-6f) ok_41 = false;

        ggml_gallocr_free(ga);
        ggml_free(ctx);
        ggml_backend_buffer_free(sbuf);
        ggml_free(sctx);
    }

    // 3. Subop 42: FUSED_MOE_MASKED_SUM_TREE_SCONV_ADD (single-GPU + dual-GPU F16 peer + same-call chain)
    float max_err_42 = 0.0f;
    for (int with_peer = 0; with_peer <= 1; ++with_peer) {
        const int H = 512, K = 8, T = 4, P = 3, max_nodes = 16, s_max = max_nodes + 1;
        const int chan_off = 256, s_ch = chan_off + H, path_w = 8, m_stride = 11 + path_w;
        std::mt19937 rng(4242 + with_peer);
        std::uniform_real_distribution<float> dist(-0.7f, 0.7f);

        std::vector<float> down((size_t) H * K * T), wsel((size_t) K * T), peer_f32((size_t) H * T);
        std::vector<int32_t> msel((size_t) K * T);
        std::vector<float> w_sc((size_t) 4 * H), addend((size_t) H * T), pc((size_t) 3 * s_ch * P);
        for (float & v : down)     v = dist(rng);
        for (float & v : wsel)     v = std::fabs(dist(rng)) * 0.3f;
        for (float & v : peer_f32) v = dist(rng) * 0.5f;
        for (size_t i = 0; i < msel.size(); ++i) msel[i] = (i % 3 == 0) ? -1 : (int32_t) (i % 128);
        for (float & v : w_sc)   v = dist(rng) * 0.25f;
        for (float & v : addend) v = dist(rng);
        for (float & v : pc)     v = dist(rng) * 0.2f;
        std::vector<ggml_fp16_t> peer_f16(peer_f32.size());
        for (size_t i = 0; i < peer_f32.size(); ++i) peer_f16[i] = ggml_fp32_to_fp16(peer_f32[i]);

        std::vector<int32_t> meta((size_t) T * m_stride, -1);
        const int slots[T] = { 1, 3, 6, 8 };
        const int hrow[T][3]  = { {-1,-1,-1}, {-1,-1,0}, {-1,0,1}, {-1,-1,0} };
        const int hslot[T][3] = { {-1,-2,-3}, {-2,-3,1}, {-3,1,3}, {-2,-3,1} };
        for (int r = 0; r < T; ++r) {
            int32_t * mt = meta.data() + (size_t) r * m_stride;
            mt[0] = slots[r]; mt[1] = 10 + r; mt[2] = 2; mt[3] = 0; mt[4] = 0;
            for (int j = 0; j < 3; ++j) { mt[5 + j] = hrow[r][j]; mt[8 + j] = hslot[r][j]; }
            mt[11] = slots[r];
        }

        struct ggml_init_params sp = { 8 * ggml_tensor_overhead() + 4096, nullptr, true };
        ggml_context * sctx = ggml_init(sp);
        ggml_tensor * tc_a = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, s_ch, s_max, P);
        ggml_tensor * tc_b = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, s_ch, s_max, P);
        ggml_tensor * t_pc = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, 3, s_ch, P);
        ggml_backend_buffer_t sbuf = ggml_backend_alloc_ctx_tensors(sctx, backend);
        ggml_backend_tensor_memset(tc_a, 0, 0, ggml_nbytes(tc_a));
        ggml_backend_tensor_memset(tc_b, 0, 0, ggml_nbytes(tc_b));
        ggml_backend_tensor_set(t_pc, pc.data(), 0, pc.size() * sizeof(float));

        struct ggml_init_params gp = { 32 * ggml_tensor_overhead() + ggml_graph_overhead() + (1 << 20), nullptr, true };
        ggml_context * ctx = ggml_init(gp);
        ggml_tensor * t_dn   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, K, T);
        ggml_tensor * t_ws   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
        ggml_tensor * t_ms   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K, T);
        ggml_tensor * t_peer = with_peer ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, H, T) : nullptr;
        ggml_tensor * t_wsc  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, H);
        ggml_tensor * t_add  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
        ggml_tensor * t_mt   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, m_stride, T);
        ggml_set_input(t_dn); ggml_set_input(t_ws); ggml_set_input(t_ms);
        if (t_peer) ggml_set_input(t_peer);
        ggml_set_input(t_wsc); ggml_set_input(t_add); ggml_set_input(t_mt);

        // Reference: subop 20 + subop 33
        ggml_tensor * ref_sum = ggml_inkling2_moe_masked_sum(ctx, t_dn, t_ws, t_ms, t_peer, 0);
        ggml_tensor * ref_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
        ref_out->op = GGML_OP_INKLING2_OP;
        ref_out->src[0] = ref_sum; ref_out->src[1] = t_wsc; ref_out->src[2] = t_pc; ref_out->src[3] = tc_a;
        ref_out->src[4] = t_mt; ref_out->src[5] = nullptr; ref_out->src[6] = t_add;
        test_set_op_i32(ref_out, 0, 33); test_set_op_i32(ref_out, 1, chan_off);
        test_set_op_i32(ref_out, 2, 0);  test_set_op_i32(ref_out, 3, m_stride);

        // Fused subop 42
        ggml_tensor * fus_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
        fus_out->op = GGML_OP_INKLING2_OP;
        fus_out->src[0] = t_dn; fus_out->src[1] = t_ws; fus_out->src[2] = t_ms; fus_out->src[3] = t_peer;
        fus_out->src[4] = t_wsc; fus_out->src[5] = t_pc; fus_out->src[6] = tc_b; fus_out->src[7] = t_mt; fus_out->src[8] = t_add;
        test_set_op_i32(fus_out, 0, 42); test_set_op_i32(fus_out, 1, chan_off); test_set_op_i32(fus_out, 2, m_stride);

        ggml_set_output(ref_out); ggml_set_output(fus_out);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ref_out);
        ggml_build_forward_expand(gf, fus_out);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_tensor_set(t_dn,  down.data(),   0, down.size()   * sizeof(float));
        ggml_backend_tensor_set(t_ws,  wsel.data(),   0, wsel.size()   * sizeof(float));
        ggml_backend_tensor_set(t_ms,  msel.data(),   0, msel.size()   * sizeof(int32_t));
        if (t_peer) ggml_backend_tensor_set(t_peer, peer_f16.data(), 0, peer_f16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(t_wsc, w_sc.data(),   0, w_sc.size()   * sizeof(float));
        ggml_backend_tensor_set(t_add, addend.data(), 0, addend.size() * sizeof(float));
        ggml_backend_tensor_set(t_mt,  meta.data(),   0, meta.size()   * sizeof(int32_t));
        ggml_backend_graph_compute(backend, gf);

        std::vector<float> ra((size_t) H * T), rb((size_t) H * T);
        ggml_backend_tensor_get(ref_out, ra.data(), 0, ra.size() * sizeof(float));
        ggml_backend_tensor_get(fus_out, rb.data(), 0, rb.size() * sizeof(float));
        max_err_42 = std::max(max_err_42, max_abs_diff(ra, rb));

        std::vector<float> ca(ggml_nelements(tc_a)), cb(ggml_nelements(tc_b));
        ggml_backend_tensor_get(tc_a, ca.data(), 0, ggml_nbytes(tc_a));
        ggml_backend_tensor_get(tc_b, cb.data(), 0, ggml_nbytes(tc_b));
        max_err_42 = std::max(max_err_42, max_abs_diff(ca, cb));

        ggml_gallocr_free(ga);
        ggml_free(ctx);
        ggml_backend_buffer_free(sbuf);
        ggml_free(sctx);
    }

    // 4. Subop 43: TREE_PREFILL_CLONE_CODA (F16 global + Q8_0 SWA ring wrap-around + prefix_conv)
    bool ok_43 = true;
    for (int swa = 0; swa <= 1; ++swa) {
        const int D = 128, Hk = 4, cap = swa ? 8 : 16, P = 3, s_ch = 64;
        const int pos0 = swa ? 6 : 3, n_tok = 5; // swa wraps: slots {6, 7, 0, 1, 2}
        const ggml_type kv_t = swa ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
        struct ggml_init_params sp = { 16 * ggml_tensor_overhead() + ggml_graph_overhead() + (1 << 20), nullptr, true };
        ggml_context * ctx = ggml_init(sp);
        ggml_tensor * pk = ggml_new_tensor_4d(ctx, kv_t, D, cap, Hk, P);
        ggml_tensor * pv = ggml_new_tensor_4d(ctx, kv_t, D, cap, Hk, P);
        ggml_tensor * pc = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, s_ch, P);
        ggml_tensor * dep = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_memset(pk, 0, 0, ggml_nbytes(pk));
        ggml_backend_tensor_memset(pv, 0, 0, ggml_nbytes(pv));
        ggml_backend_tensor_memset(pc, 0, 0, ggml_nbytes(pc));
        ggml_backend_tensor_memset(dep, 0, 0, ggml_nbytes(dep));

        // Populate plane 2 with non-zero distinct bytes
        std::vector<uint8_t> k_p2(pk->nb[3]), v_p2(pv->nb[3]);
        for (size_t i = 0; i < k_p2.size(); ++i) { k_p2[i] = (uint8_t) ((i * 13 + 7) & 0xFF); v_p2[i] = (uint8_t) ((i * 29 + 3) & 0xFF); }
        ggml_backend_tensor_set(pk, k_p2.data(), 2 * pk->nb[3], pk->nb[3]);
        ggml_backend_tensor_set(pv, v_p2.data(), 2 * pv->nb[3], pv->nb[3]);
        std::vector<float> c_p2((size_t) 3 * s_ch);
        for (size_t i = 0; i < c_p2.size(); ++i) c_p2[i] = (float) (i + 1) * 0.25f;
        ggml_backend_tensor_set(pc, c_p2.data(), 2 * pc->nb[2], pc->nb[2]);

        ggml_tensor * op = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        op->op = GGML_OP_INKLING2_OP;
        op->src[0] = pk; op->src[1] = pv; op->src[2] = pc; op->src[3] = dep;
        test_set_op_i32(op, 0, 43); test_set_op_i32(op, 1, pos0);
        test_set_op_i32(op, 2, n_tok); test_set_op_i32(op, 3, swa ? 4 : 0);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, op);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_graph_compute(backend, gf);

        const size_t rb = pk->nb[1];
        std::vector<uint8_t> r_ref(rb), r_got(rb);
        for (int t = 0; t < n_tok; ++t) {
            const int slot = swa ? ((pos0 + t) % cap) : (pos0 + t);
            for (int h = 0; h < Hk; ++h) {
                const size_t off2 = 2 * pk->nb[3] + (size_t) h * pk->nb[2] + (size_t) slot * rb;
                ggml_backend_tensor_get(pk, r_ref.data(), off2, rb);
                for (int p = 0; p < 2; ++p) {
                    ggml_backend_tensor_get(pk, r_got.data(), (size_t) p * pk->nb[3] + (size_t) h * pk->nb[2] + (size_t) slot * rb, rb);
                    if (r_got != r_ref) ok_43 = false;
                }
                ggml_backend_tensor_get(pv, r_ref.data(), off2, rb);
                for (int p = 0; p < 2; ++p) {
                    ggml_backend_tensor_get(pv, r_got.data(), (size_t) p * pv->nb[3] + (size_t) h * pv->nb[2] + (size_t) slot * rb, rb);
                    if (r_got != r_ref) ok_43 = false;
                }
            }
        }
        std::vector<float> c_got((size_t) 3 * s_ch);
        for (int p = 0; p < 2; ++p) {
            ggml_backend_tensor_get(pc, c_got.data(), (size_t) p * pc->nb[2], pc->nb[2]);
            if (max_abs_diff(c_got, c_p2) > 0.0f) ok_43 = false;
        }
        ggml_gallocr_free(ga);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    std::printf("[test_inkling2_ops] 2dsd_p0_fused: subop40=%d subop41=%d subop42_err=%.3e subop43=%d\n",
                ok_40 ? 1 : 0, ok_41 ? 1 : 0, max_err_42, ok_43 ? 1 : 0);
    return ok_40 && ok_41 && max_err_42 < 1e-5f && ok_43;
}

// ─── Test: Multi-Slot TREE_ATTN_REL (subop 35) per-plane prefix_len_buf[plane] ─
static bool test_2dsd_multislot_tree_attn_rel(ggml_backend_t backend) {
    const int D = 128, Hk = 2, GQA = 3, Hq = Hk * GQA;
    const int cap = 32, S = 8, P = 4, T = 4, rel_ext = 64, path_w = 8, m_stride = 11 + path_w;
    const int32_t pref_lens[P] = { 3, 9, 5, 14 };
    const int32_t q_planes[T]  = { 0, 1, 2, 3 };
    const int32_t q_extra[T]   = { 2, 1, 3, 2 }; // uncommitted tree ancestors + self

    std::mt19937 rng(3535);
    std::uniform_real_distribution<float> dist(-0.6f, 0.6f);

    std::vector<float> q((size_t) D * Hq * T);
    std::vector<ggml_fp16_t> pk((size_t) D * cap * Hk * P), pv((size_t) D * cap * Hk * P);
    std::vector<ggml_fp16_t> tk((size_t) D * S   * Hk * P), tv((size_t) D * S   * Hk * P);
    std::vector<float> rt((size_t) (rel_ext + 1) * Hq * T), gate((size_t) D * Hq * T);
    for (float & v : q)    v = dist(rng);
    for (float & v : rt)   v = dist(rng) * 0.3f;
    for (float & v : gate) v = dist(rng);
    for (auto & v : pk) v = ggml_fp32_to_fp16(dist(rng));
    for (auto & v : pv) v = ggml_fp32_to_fp16(dist(rng));
    for (auto & v : tk) v = ggml_fp32_to_fp16(dist(rng));
    for (auto & v : tv) v = ggml_fp32_to_fp16(dist(rng));

    std::vector<int32_t> meta((size_t) T * m_stride, -1);
    for (int t = 0; t < T; ++t) {
        int32_t * mt = meta.data() + (size_t) t * m_stride;
        const int p = q_planes[t];
        mt[0] = q_extra[t] - 1;
        mt[1] = pref_lens[p] + q_extra[t] - 1;
        mt[2] = p;
        mt[3] = 0;
        mt[4] = 0;
        for (int e = 0; e < q_extra[t]; ++e) mt[11 + e] = e;
    }

    auto run_one = [&](const int32_t * pl_ptr, int pl_n, int single_row) -> std::vector<float> {
        const int rows = (single_row >= 0) ? 1 : T;
        struct ggml_init_params gp = { 32 * ggml_tensor_overhead() + ggml_graph_overhead() + (2 << 20), nullptr, true };
        ggml_context * ctx = ggml_init(gp);
        ggml_tensor * t_q  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hq, rows);
        ggml_tensor * t_pk = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, cap, Hk, P);
        ggml_tensor * t_pv = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, cap, Hk, P);
        ggml_tensor * t_tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, S,   Hk, P);
        ggml_tensor * t_tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, S,   Hk, P);
        ggml_tensor * t_rt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rel_ext + 1, Hq, rows);
        ggml_tensor * t_mt = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, m_stride, rows);
        ggml_tensor * t_pl = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, pl_n);
        ggml_tensor * t_gt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D * Hq, rows);
        ggml_set_input(t_q); ggml_set_input(t_pk); ggml_set_input(t_pv);
        ggml_set_input(t_tk); ggml_set_input(t_tv); ggml_set_input(t_rt);
        ggml_set_input(t_mt); ggml_set_input(t_pl); ggml_set_input(t_gt);

        ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D * Hq, rows);
        dst->op = GGML_OP_INKLING2_OP;
        dst->src[0] = t_q;  dst->src[1] = t_pk; dst->src[2] = t_pv;
        dst->src[3] = t_tk; dst->src[4] = t_tv; dst->src[5] = t_rt;
        dst->src[6] = t_mt; dst->src[7] = t_pl; dst->src[8] = t_gt;
        test_set_op_i32(dst, 0, 35);
        test_set_op_i32(dst, 1, 0);
        test_set_op_i32(dst, 2, rel_ext);
        test_set_op_i32(dst, 3, m_stride);
        test_set_op_i32(dst, 4, path_w);
        ggml_set_output(dst);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dst);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(ga, gf);

        const int r0 = (single_row >= 0) ? single_row : 0;
        ggml_backend_tensor_set(t_q,  q.data()    + (size_t) r0 * D * Hq,           0, (size_t) rows * D * Hq * sizeof(float));
        ggml_backend_tensor_set(t_pk, pk.data(),                                    0, pk.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(t_pv, pv.data(),                                    0, pv.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(t_tk, tk.data(),                                    0, tk.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(t_tv, tv.data(),                                    0, tv.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(t_rt, rt.data()   + (size_t) r0 * (rel_ext + 1) * Hq, 0, (size_t) rows * (rel_ext + 1) * Hq * sizeof(float));
        ggml_backend_tensor_set(t_mt, meta.data() + (size_t) r0 * m_stride,         0, (size_t) rows * m_stride * sizeof(int32_t));
        ggml_backend_tensor_set(t_pl, pl_ptr,                                       0, (size_t) pl_n * sizeof(int32_t));
        ggml_backend_tensor_set(t_gt, gate.data() + (size_t) r0 * D * Hq,           0, (size_t) rows * D * Hq * sizeof(float));

        ggml_backend_graph_compute(backend, gf);
        std::vector<float> out((size_t) D * Hq * rows);
        ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
        ggml_gallocr_free(ga);
        ggml_free(ctx);
        return out;
    };

    const std::vector<float> batched = run_one(pref_lens, P, -1);
    float max_err = 0.0f;
    for (int t = 0; t < T; ++t) {
        const int32_t pl1 = pref_lens[q_planes[t]];
        const std::vector<float> ref = run_one(&pl1, 1, t);
        std::vector<float> got(batched.begin() + (ptrdiff_t) t * D * Hq,
                               batched.begin() + (ptrdiff_t) (t + 1) * D * Hq);
        max_err = std::max(max_err, max_abs_diff(ref, got));
    }
    std::printf("[test_inkling2_ops] 2dsd_multislot_attn_rel: max_err=%.3e\n", max_err);
    return max_err < 1e-6f;
}

// ─── Test: Non-greedy Seeded Parity & N=4 Concurrent Batched 2DSD Sessions ──
static bool test_2dsd_nongreedy_and_multislot_session() {
    using namespace luce::common;
    auto synth_logits = [](uint32_t h, int depth) -> std::array<float, 8> {
        std::array<float, 8> lg{};
        for (int v = 0; v < 8; ++v) {
            uint32_t x = h * 2654435761u + (uint32_t) (v * 97 + depth * 31);
            x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
            lg[(size_t) v] = ((float) (x & 1023u) / 1024.0f) * 3.0f - 1.5f;
        }
        return lg;
    };
    auto argmax8 = [](const std::array<float, 8> & lg) -> int32_t {
        int32_t best = 0;
        for (int i = 1; i < 8; ++i) if (lg[(size_t) i] > lg[(size_t) best]) best = i;
        return best;
    };
    auto sample8 = [](const float * lg, std::mt19937 & rng, float temp) -> int32_t {
        float mx = lg[0];
        for (int i = 1; i < 8; ++i) mx = std::max(mx, lg[i]);
        double probs[8], Z = 0.0;
        for (int i = 0; i < 8; ++i) { probs[i] = std::exp((lg[i] - mx) / temp); Z += probs[i]; }
        std::uniform_real_distribution<double> u01(0.0, Z);
        const double r = u01(rng);
        double acc = 0.0;
        for (int i = 0; i < 8; ++i) {
            acc += probs[i];
            if (r <= acc || i == 7) return i;
        }
        return 7;
    };
    auto seq_hash = [](const std::vector<int32_t> & seq) -> uint32_t {
        uint32_t h = 2166136261u;
        for (int32_t t : seq) { h ^= (uint32_t) t; h *= 16777619u; }
        return h;
    };

    // 1. Non-greedy seeded parity: 2DSD stage2b with stochastic hooks MUST match AR bit-for-bit!
    const float temp = 0.8f;
    const uint32_t seed = 424242u;
    const int n_gen = 32;
    const std::vector<int32_t> prompt = { 1, 5, 2, 7 };

    std::vector<int32_t> ar_toks;
    {
        std::mt19937 rng(seed);
        std::vector<int32_t> ctx = prompt;
        auto lg0 = synth_logits(seq_hash(ctx), 3);
        int32_t tok = sample8(lg0.data(), rng, temp);
        ar_toks.push_back(tok);
        for (int i = 1; i < n_gen; ++i) {
            ctx.push_back(tok);
            auto lg = synth_logits(seq_hash(ctx), 3);
            tok = sample8(lg.data(), rng, temp);
            ar_toks.push_back(tok);
        }
    }

    struct MockStochRunner final : public Inkling2TwoDSDRunner {
        std::vector<int32_t> committed;
        std::vector<int32_t> slot_tok;
        std::vector<std::array<float, 8>> last_readout_logits;
        decltype(synth_logits) & fn_lg;
        decltype(argmax8) & fn_am;
        decltype(seq_hash) & fn_sh;
        MockStochRunner(std::vector<int32_t> p, decltype(synth_logits) & fl, decltype(argmax8) & fa, decltype(seq_hash) & fs)
            : committed(std::move(p)), slot_tok(129, 0), fn_lg(fl), fn_am(fa), fn_sh(fs) {}
        int prefix_length() const override { return (int) committed.size(); }
        int max_nodes() const override { return 128; }
        int max_ctx() const override { return 256; }
        bool advance_nodes(const std::vector<Inkling2TwoDSDNodeDesc> & nodes, uint32_t readout_mask,
                           std::vector<int> & out_rows, std::vector<int32_t> & out_argmax) override {
            out_rows.clear(); out_argmax.clear(); last_readout_logits.clear();
            for (int i = 0; i < (int) nodes.size(); ++i) {
                const auto & nd = nodes[(size_t) i];
                slot_tok[(size_t) nd.slot] = nd.token;
                const int new_d = nd.depth + 1;
                if ((readout_mask & (1u << new_d)) != 0u) {
                    std::vector<int32_t> path = committed;
                    for (int anc : nd.ancestors) path.push_back(slot_tok[(size_t) anc]);
                    path.push_back(nd.token);
                    auto lg = fn_lg(fn_sh(path), new_d);
                    out_rows.push_back(i);
                    out_argmax.push_back(fn_am(lg));
                    last_readout_logits.push_back(lg);
                }
            }
            return true;
        }
        bool commit_nodes(const int * slots, int n) override {
            for (int i = 0; i < n; ++i) committed.push_back(slot_tok[(size_t) slots[i]]);
            return true;
        }
        bool draft_batch(const std::vector<Inkling2TwoDSDDraftReq> & reqs,
                         std::vector<Inkling2TwoDSDDraftOut> & out) override {
            out.clear();
            for (const auto & r : reqs) {
                Inkling2TwoDSDDraftOut o;
                std::vector<int32_t> path = committed;
                for (int anc : r.span_slots) path.push_back(slot_tok[(size_t) anc]);
                path.push_back(r.head_token);
                for (int s = 0; s < 5; ++s) {
                    auto lg = fn_lg(fn_sh(path), 3);
                    int32_t tk = fn_am(lg);
                    o.tokens.push_back(tk);
                    o.confidence.push_back(0.9f);
                    path.push_back(tk);
                }
                out.push_back(std::move(o));
            }
            return true;
        }
    };

    std::vector<int32_t> td_toks;
    {
        std::mt19937 rng(seed);
        MockStochRunner runner(prompt, synth_logits, argmax8, seq_hash);
        auto lg0 = synth_logits(seq_hash(prompt), 3);
        int32_t first_tok = sample8(lg0.data(), rng, temp);

        Inkling2TwoDSDSamplerHooks hooks;
        hooks.capture = [&](int r_idx, Inkling2TwoDSDDepthRSample & out) {
            const auto & lg = runner.last_readout_logits[(size_t) r_idx];
            out.logits.assign(lg.begin(), lg.end());
        };
        hooks.sample = [&](const Inkling2TwoDSDDepthRSample & payload, int32_t greedy_tok) -> int32_t {
            if (payload.logits.size() == 8) return sample8(payload.logits.data(), rng, temp);
            return greedy_tok;
        };

        Inkling2TwoDSDConfig cfg;
        cfg.stage = Inkling2TwoDSDStage::Stage2b;
        cfg.k = 6;
        Inkling2TwoDSDStats st;
        std::string err;
        auto on_commit = [&](int32_t & tok) -> bool {
            td_toks.push_back(tok);
            return (int) td_toks.size() < n_gen;
        };
        if (!inkling2_2dsd_run(runner, cfg, first_tok, n_gen, on_commit, &hooks, st, err)) {
            std::fprintf(stderr, "[test_inkling2_ops] non-greedy 2dsd failed: %s\n", err.c_str());
            return false;
        }
    }
    const bool ok_nongreedy = (ar_toks == td_toks);

    // 2. N=4 concurrent batched Inkling2TwoDSDSession parity vs independent AR
    constexpr int N = 4;
    std::vector<std::vector<int32_t>> prompts_n = {
        { 1, 2, 3 }, { 4, 5, 6, 7, 0 }, { 2, 2, 1, 6 }, { 7, 3, 5, 1, 4, 2 }
    };
    std::vector<std::vector<int32_t>> ref_n(N), got_n(N);
    for (int s = 0; s < N; ++s) {
        std::vector<int32_t> ctx = prompts_n[(size_t) s];
        int32_t tok = argmax8(synth_logits(seq_hash(ctx), 3));
        ref_n[(size_t) s].push_back(tok);
        for (int i = 1; i < 20; ++i) {
            ctx.push_back(tok);
            tok = argmax8(synth_logits(seq_hash(ctx), 3));
            ref_n[(size_t) s].push_back(tok);
        }
    }

    std::vector<std::unique_ptr<MockStochRunner>> runners(N);
    std::vector<Inkling2TwoDSDStats> stats_n(N);
    std::vector<std::unique_ptr<Inkling2TwoDSDSession>> sessions(N);
    Inkling2TwoDSDConfig conc_cfg = inkling2_2dsd_config_for_concurrency(3, 6, N);
    for (int s = 0; s < N; ++s) {
        runners[(size_t) s] = std::make_unique<MockStochRunner>(prompts_n[(size_t) s], synth_logits, argmax8, seq_hash);
        sessions[(size_t) s] = std::make_unique<Inkling2TwoDSDSession>(*runners[(size_t) s], conc_cfg, stats_n[(size_t) s], 256);
        int32_t first_tok = argmax8(synth_logits(seq_hash(prompts_n[(size_t) s]), 3));
        got_n[(size_t) s].push_back(first_tok);
        std::string err;
        if (!sessions[(size_t) s]->seed_cursor(first_tok, (int) prompts_n[(size_t) s].size(), err)) return false;
    }

    std::vector<bool> done_n(N, false);
    for (int iter = 0; iter < 100; ++iter) {
        bool any = false;
        for (int s = 0; s < N; ++s) if (!done_n[(size_t) s]) any = true;
        if (!any) break;

        std::vector<Inkling2TwoDSDNodeDesc> combined;
        std::vector<int> owner, local_pos;
        std::vector<std::vector<int>> active_ids(N);
        uint32_t mask = 0u;
        for (int s = 0; s < N; ++s) {
            if (done_n[(size_t) s]) continue;
            std::vector<Inkling2TwoDSDDraftReq> reqs;
            std::vector<int> ph, pw;
            sessions[(size_t) s]->collect_draft_reqs(reqs, ph, pw);
            if (!reqs.empty()) {
                std::vector<Inkling2TwoDSDDraftOut> outs;
                runners[(size_t) s]->draft_batch(reqs, outs);
                sessions[(size_t) s]->apply_draft_outs(ph, pw, outs);
            }
            std::vector<Inkling2TwoDSDNodeDesc> ds;
            std::string err;
            if (!sessions[(size_t) s]->collect_active(active_ids[(size_t) s], ds, err)) return false;
            mask |= sessions[(size_t) s]->readout_mask();
            for (int r = 0; r < (int) ds.size(); ++r) {
                ds[(size_t) r].seq_slot = s;
                combined.push_back(std::move(ds[(size_t) r]));
                owner.push_back(s);
                local_pos.push_back(r);
            }
        }

        // Batched target advance across all N slots together
        std::vector<std::vector<int>> per_r(N);
        std::vector<std::vector<int32_t>> per_am(N);
        for (int i = 0; i < (int) combined.size(); ++i) {
            const auto & nd = combined[(size_t) i];
            const int s = owner[(size_t) i];
            runners[(size_t) s]->slot_tok[(size_t) nd.slot] = nd.token;
            const int new_d = nd.depth + 1;
            if ((mask & (1u << new_d)) != 0u) {
                std::vector<int32_t> path = runners[(size_t) s]->committed;
                for (int anc : nd.ancestors) path.push_back(runners[(size_t) s]->slot_tok[(size_t) anc]);
                path.push_back(nd.token);
                per_r[(size_t) s].push_back(local_pos[(size_t) i]);
                per_am[(size_t) s].push_back(argmax8(synth_logits(seq_hash(path), new_d)));
            }
        }
        for (int s = 0; s < N; ++s) {
            if (done_n[(size_t) s]) continue;
            bool stopped = false;
            std::string err;
            auto on_commit = [&](int32_t & tok) -> bool {
                got_n[(size_t) s].push_back(tok);
                return (int) got_n[(size_t) s].size() < 20;
            };
            if (!sessions[(size_t) s]->apply_step_and_verify(active_ids[(size_t) s], per_r[(size_t) s], per_am[(size_t) s], on_commit, stopped, err)) {
                return false;
            }
            if (stopped || (int) got_n[(size_t) s].size() >= 20) done_n[(size_t) s] = true;
        }
    }
    bool ok_conc = true;
    for (int s = 0; s < N; ++s) {
        if (got_n[(size_t) s] != ref_n[(size_t) s]) ok_conc = false;
    }

    std::printf("[test_inkling2_ops] 2dsd_nongreedy_and_conc: nongreedy_parity=%d conc_n4_parity=%d\n",
                ok_nongreedy ? 1 : 0, ok_conc ? 1 : 0);
    return ok_nongreedy && ok_conc;
}

static bool test_2dsd_tree_paged_and_dual_split(ggml_backend_t backend, bool use_q8) {
    const int Hq = 24, Hk = 8, D = 128, P = 3, S_max = 16;
    const int block_size = 16;
    const int pref_len = 80; // 5 blocks
    const int n_blocks = 6;  // block 5 for post-commit growth
    const int pool_blocks = n_blocks + 3;
    const int prefix_cap = pool_blocks * block_size;
    const int path_w = 8;
    const int m_stride = 11 + path_w;
    const int rel_ext = 32;
    const int T = 4; // mixed planes {0, 1, 2, 2} with tree fork!

    std::mt19937 rng(7331 + (use_q8 ? 100 : 0));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Q((size_t) D * Hq * T), Gate((size_t) D * Hq * T);
    std::vector<float> rt((size_t) (rel_ext + 1) * Hq * T);
    for (float & v : Q) v = dist(rng);
    for (float & v : Gate) v = dist(rng);
    for (float & v : rt) v = dist(rng) * 0.2f;

    // Permuted full block table (with block 1 evicted = -1), plus disjoint Tier-0 (local) & Tier-1.5 (remote)
    std::vector<int32_t> bt_full   = { 6, -1, 4, 2, 7, 5 };
    std::vector<int32_t> bt_remote = { 6, -1, 4, -1, -1, -1 }; // Tier-1.5 warm blocks (Strix Halo)
    std::vector<int32_t> bt_local  = { -1, -1, -1, 2, 7, 5 };  // Tier-0 hot blocks (R9700)

    std::vector<float> K_pref((size_t) P * Hk * pref_len * D);
    std::vector<float> V_pref((size_t) P * Hk * pref_len * D);
    std::vector<float> K_tree((size_t) P * Hk * S_max * D);
    std::vector<float> V_tree((size_t) P * Hk * S_max * D);
    for (float & v : K_pref) v = dist(rng);
    for (float & v : V_pref) v = dist(rng);
    for (float & v : K_tree) v = dist(rng);
    for (float & v : V_tree) v = dist(rng);

    // 4 mixed-plane query rows: r0(plane 0, pos 80, path=[2]), r1(plane 1, pos 81, path=[2,5]),
    // r2(plane 2, pos 82, path=[2,5,7]), r3(plane 2, pos 81, fork path=[2,9]).
    const int planes[T] = { 0, 1, 2, 2 };
    const int q_pos[T]  = { 80, 81, 82, 81 };
    const std::vector<int> paths[T] = { {2}, {2, 5}, {2, 5, 7}, {2, 9} };
    std::vector<int32_t> meta((size_t) m_stride * T, -1);
    for (int t = 0; t < T; ++t) {
        int32_t * mt = &meta[(size_t) t * m_stride];
        mt[0] = paths[t].back();
        mt[1] = q_pos[t];
        mt[2] = planes[t];
        for (size_t e = 0; e < paths[t].size(); ++e) mt[11 + e] = paths[t][e];
    }

    // Exact CPU reference for single paged table (with block 1 evicted) + tree path
    std::vector<float> ref((size_t) D * Hq * T, 0.0f);
    const float scale = 1.0f / std::sqrt((float) D);
    for (int t = 0; t < T; ++t) {
        const int pl = planes[t], qp_i = q_pos[t];
        for (int h = 0; h < Hq; ++h) {
            const int kh = h / (Hq / Hk);
            const float * qp = &Q[((size_t) t * Hq + h) * D];
            const float * rtp = &rt[((size_t) t * Hq + h) * (rel_ext + 1)];
            std::vector<float> scores(qp_i + 1, -INFINITY);
            float mx = 0.0f; // zero-sink
            for (int kp = 0; kp <= qp_i; ++kp) {
                const float * kptr = nullptr;
                if (kp < pref_len) {
                    if (bt_full[kp / block_size] < 0) continue;
                    kptr = &K_pref[(((size_t) pl * Hk + kh) * pref_len + kp) * D];
                } else {
                    const int slot = paths[t][kp - pref_len];
                    kptr = &K_tree[(((size_t) pl * Hk + kh) * S_max + slot) * D];
                }
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += qp[d] * kptr[d];
                scores[kp] = dot * scale + rtp[std::min(qp_i - kp, rel_ext)];
                mx = std::max(mx, scores[kp]);
            }
            float sum = std::exp(0.0f - mx);
            std::vector<float> probs(qp_i + 1, 0.0f);
            for (int kp = 0; kp <= qp_i; ++kp) {
                if (std::isfinite(scores[kp])) { probs[kp] = std::exp(scores[kp] - mx); sum += probs[kp]; }
            }
            float * op = &ref[((size_t) t * Hq + h) * D];
            const float * gp = &Gate[((size_t) t * Hq + h) * D];
            for (int d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (int kp = 0; kp <= qp_i; ++kp) {
                    if (probs[kp] <= 0.0f) continue;
                    const float * vptr = (kp < pref_len)
                        ? &V_pref[(((size_t) pl * Hk + kh) * pref_len + kp) * D]
                        : &V_tree[(((size_t) pl * Hk + kh) * S_max + paths[t][kp - pref_len]) * D];
                    acc += (probs[kp] / sum) * vptr[d];
                }
                op[d] = acc / (1.0f + std::exp(-gp[d]));
            }
        }
    }

    const ggml_type kv_t = use_q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    const size_t row_b = ggml_row_size(kv_t, D);
    auto pack_row = [&](const float * src, uint8_t * dst) {
        if (use_q8) {
            ggml_quantize_chunk(GGML_TYPE_Q8_0, src, dst, 0, 1, D, nullptr);
        } else {
            auto * hp = reinterpret_cast<ggml_fp16_t *>(dst);
            for (int d = 0; d < D; ++d) hp[d] = ggml_fp32_to_fp16(src[d]);
        }
    };

    // Poison-fill prefix pool to detect any accidental read through -1 holes
    std::vector<uint8_t> pk_buf((size_t) P * Hk * prefix_cap * row_b, 0x7F);
    std::vector<uint8_t> pv_buf((size_t) P * Hk * prefix_cap * row_b, 0x7F);
    std::vector<uint8_t> tk_buf((size_t) P * Hk * S_max * row_b, 0);
    std::vector<uint8_t> tv_buf((size_t) P * Hk * S_max * row_b, 0);
    for (int pl = 0; pl < P; ++pl) {
        for (int h = 0; h < Hk; ++h) {
            for (int kp = 0; kp < pref_len; ++kp) {
                const int pb = bt_full[kp / block_size];
                if (pb < 0) continue;
                const int slot = pb * block_size + (kp % block_size);
                pack_row(&K_pref[(((size_t) pl * Hk + h) * pref_len + kp) * D],
                         pk_buf.data() + (((size_t) pl * Hk + h) * prefix_cap + slot) * row_b);
                pack_row(&V_pref[(((size_t) pl * Hk + h) * pref_len + kp) * D],
                         pv_buf.data() + (((size_t) pl * Hk + h) * prefix_cap + slot) * row_b);
            }
            for (int s = 0; s < S_max; ++s) {
                pack_row(&K_tree[(((size_t) pl * Hk + h) * S_max + s) * D],
                         tk_buf.data() + (((size_t) pl * Hk + h) * S_max + s) * row_b);
                pack_row(&V_tree[(((size_t) pl * Hk + h) * S_max + s) * D],
                         tv_buf.data() + (((size_t) pl * Hk + h) * S_max + s) * row_b);
            }
        }
    }

    auto make_pl_buf = [&](const std::vector<int32_t> & bt) {
        std::vector<int32_t> v(2 + bt.size());
        v[0] = pref_len;
        v[1] = 0;
        for (size_t i = 0; i < bt.size(); ++i) v[2 + i] = bt[i];
        return v;
    };
    const auto pl_full   = make_pl_buf(bt_full);
    const auto pl_remote = make_pl_buf(bt_remote);
    const auto pl_local  = make_pl_buf(bt_local);

    struct ggml_init_params gp = { 256 * ggml_tensor_overhead() + ggml_graph_overhead() + 8192, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * t_Q    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hq, T);
    ggml_tensor * t_pk   = ggml_new_tensor_4d(ctx, kv_t, D, prefix_cap, Hk, P);
    ggml_tensor * t_pv   = ggml_new_tensor_4d(ctx, kv_t, D, prefix_cap, Hk, P);
    ggml_tensor * t_tk   = ggml_new_tensor_4d(ctx, kv_t, D, S_max, Hk, P);
    ggml_tensor * t_tv   = ggml_new_tensor_4d(ctx, kv_t, D, S_max, Hk, P);
    ggml_tensor * t_rt   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rel_ext + 1, Hq, T);
    ggml_tensor * t_meta = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, m_stride, T);
    ggml_tensor * t_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D * Hq, T);
    ggml_tensor * t_plf  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) pl_full.size());
    ggml_tensor * t_plr  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) pl_remote.size());
    ggml_tensor * t_pll  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) pl_local.size());
    // Commit test tensors (commit 3 tokens [2, 5, 7] at pos 80..82 -> block 5 = phys 5)
    ggml_tensor * t_pc   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 16, P);
    ggml_tensor * t_tc   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, S_max, P);
    ggml_tensor * t_cs   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
    ggml_tensor * t_cm   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    ggml_tensor * t_bt   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_blocks);
    ggml_set_input(t_Q); ggml_set_input(t_pk); ggml_set_input(t_pv); ggml_set_input(t_tk); ggml_set_input(t_tv);
    ggml_set_input(t_rt); ggml_set_input(t_meta); ggml_set_input(t_gate);
    ggml_set_input(t_plf); ggml_set_input(t_plr); ggml_set_input(t_pll);
    ggml_set_input(t_pc); ggml_set_input(t_tc); ggml_set_input(t_cs); ggml_set_input(t_cm); ggml_set_input(t_bt);

    auto make_ta = [&](ggml_tensor * tk, ggml_tensor * tv, ggml_tensor * plb, ggml_tensor * g, int part, int ex_sink) {
        ggml_tensor * r = part
            ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (D + 2) * Hq, T)
            : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D * Hq, T);
        r->op = GGML_OP_INKLING2_OP;
        r->src[0] = t_Q; r->src[1] = t_pk; r->src[2] = t_pv;
        r->src[3] = tk;  r->src[4] = tv;   r->src[5] = t_rt;
        r->src[6] = t_meta; r->src[7] = plb; r->src[8] = g;
        test_set_op_i32(r, 0, 35);
        test_set_op_i32(r, 1, 0);
        test_set_op_i32(r, 2, rel_ext);
        test_set_op_i32(r, 3, m_stride);
        test_set_op_i32(r, 4, path_w);
        test_set_op_i32(r, 5, block_size);
        test_set_op_i32(r, 6, part);
        test_set_op_i32(r, 7, ex_sink);
        test_set_op_i32(r, 8, 0);
        test_set_op_i32(r, 9, 0);
        return r;
    };

    ggml_tensor * single = make_ta(t_tk, t_tv, t_plf, t_gate, 0, 0);
    // Zero-tree-copy Tier-1.5 (Strix Halo): tree_k=nullptr, tree_v=nullptr, include_sink=1!
    ggml_tensor * part_r = make_ta(nullptr, nullptr, t_plr, nullptr, 1, 0);
    // Tier-0 (R9700): local hot blocks + in-flight tree_k/v, exclude_sink=1!
    ggml_tensor * part_l = make_ta(t_tk, t_tv, t_pll, nullptr, 1, 1);
    ggml_tensor * merged = ggml_inkling2_attn_split_combine(ctx, part_r, part_l, t_gate, D, Hq);

    // Paged TREE_COMMIT (runs after attention reads)
    ggml_tensor * cmt = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    cmt->op = GGML_OP_INKLING2_OP;
    cmt->src[0] = t_pk; cmt->src[1] = t_pv; cmt->src[2] = t_tk; cmt->src[3] = t_tv;
    cmt->src[4] = t_pc; cmt->src[5] = t_tc; cmt->src[6] = t_cs; cmt->src[7] = t_cm;
    cmt->src[8] = t_bt;
    test_set_op_i32(cmt, 0, 36);
    test_set_op_i32(cmt, 1, 0);
    test_set_op_i32(cmt, 2, block_size);
    test_set_op_i32(cmt, 3, 0);
    test_set_op_i32(cmt, 4, 0);

    ggml_set_output(single); ggml_set_output(merged); ggml_set_output(cmt); ggml_set_output(t_pk); ggml_set_output(t_pv);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, single);
    ggml_build_forward_expand(gf, merged);
    ggml_build_forward_expand(gf, cmt);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);

    const int32_t cs[3] = { 2, 5, 7 };
    const int32_t cm[2] = { pref_len, 3 };
    ggml_backend_tensor_set(t_Q, Q.data(), 0, Q.size() * sizeof(float));
    ggml_backend_tensor_set(t_pk, pk_buf.data(), 0, pk_buf.size());
    ggml_backend_tensor_set(t_pv, pv_buf.data(), 0, pv_buf.size());
    ggml_backend_tensor_set(t_tk, tk_buf.data(), 0, tk_buf.size());
    ggml_backend_tensor_set(t_tv, tv_buf.data(), 0, tv_buf.size());
    ggml_backend_tensor_set(t_rt, rt.data(), 0, rt.size() * sizeof(float));
    ggml_backend_tensor_set(t_meta, meta.data(), 0, meta.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_gate, Gate.data(), 0, Gate.size() * sizeof(float));
    ggml_backend_tensor_set(t_plf, pl_full.data(), 0, pl_full.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_plr, pl_remote.data(), 0, pl_remote.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_pll, pl_local.data(), 0, pl_local.size() * sizeof(int32_t));
    ggml_backend_tensor_memset(t_pc, 0, 0, ggml_nbytes(t_pc));
    ggml_backend_tensor_memset(t_tc, 0, 0, ggml_nbytes(t_tc));
    ggml_backend_tensor_set(t_cs, cs, 0, sizeof(cs));
    ggml_backend_tensor_set(t_cm, cm, 0, sizeof(cm));
    ggml_backend_tensor_set(t_bt, bt_full.data(), 0, bt_full.size() * sizeof(int32_t));
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> got_s(ref.size()), got_m(ref.size());
    std::vector<uint8_t> pk_after(pk_buf.size()), pv_after(pv_buf.size());
    ggml_backend_tensor_get(single, got_s.data(), 0, got_s.size() * sizeof(float));
    ggml_backend_tensor_get(merged, got_m.data(), 0, got_m.size() * sizeof(float));
    ggml_backend_tensor_get(t_pk, pk_after.data(), 0, pk_after.size());
    ggml_backend_tensor_get(t_pv, pv_after.data(), 0, pv_after.size());
    ggml_gallocr_free(ga);
    ggml_free(ctx);

    const float ref_err   = max_abs_diff(got_s, ref);
    const float dual_err  = max_abs_diff(got_s, got_m);
    const float tol       = use_q8 ? 2e-2f : 2e-3f;
    bool cmt_ok = true;
    for (int pl = 0; pl < P; ++pl) {
        for (int h = 0; h < Hk; ++h) {
            for (int i = 0; i < 3; ++i) {
                const int pos = pref_len + i;
                const int dslot = bt_full[pos / block_size] * block_size + (pos % block_size);
                const uint8_t * dk = pk_after.data() + (((size_t) pl * Hk + h) * prefix_cap + dslot) * row_b;
                const uint8_t * dv = pv_after.data() + (((size_t) pl * Hk + h) * prefix_cap + dslot) * row_b;
                const uint8_t * sk = tk_buf.data()   + (((size_t) pl * Hk + h) * S_max + cs[i]) * row_b;
                const uint8_t * sv = tv_buf.data()   + (((size_t) pl * Hk + h) * S_max + cs[i]) * row_b;
                if (std::memcmp(dk, sk, row_b) != 0 || std::memcmp(dv, sv, row_b) != 0) cmt_ok = false;
            }
        }
    }
    std::printf("[test_inkling2_ops] 2dsd_tree_paged_dual(%s) ref_err=%.3e dual_err=%.3e cmt_kv=%s\n",
                use_q8 ? "Q8_0" : "F16", ref_err, dual_err, cmt_ok ? "EXACT" : "DIFF");
    return ref_err < tol && dual_err < 1e-4f && cmt_ok;
}

// ─── Step 2 Unit Test B: Last-Loop Shared-KV (W=16, P_storage=1, P_logical=3) + Paged Commit ──
static bool test_2dsd_tree_shared_kv(ggml_backend_t backend, bool use_q8) {
    const int Hq = 24, Hk = 8, D = 128, P_tree = 3, P_storage = 1, S_max = 16;
    const int block_size = 16;
    const int shared_w = 16;
    const int tail_cap = 32;
    const int pref_len = 64; // [0..47] shared r=R, [48..63] per-plane W-ring
    const int n_blocks = 5;  // block 4 for commit
    const int pool_blocks = 6;
    const int pool_cap = pool_blocks * block_size;
    const int prefix_cap = pool_cap + (P_tree - 1) * tail_cap;
    const int path_w = 8, m_stride = 11 + path_w, rel_ext = 32;
    const int T = 3; // planes 0, 1, 2 at pos 64, 65, 65

    std::mt19937 rng(8844 + (use_q8 ? 55 : 0));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> Q((size_t) D * Hq * T), Gate((size_t) D * Hq * T), rt((size_t) (rel_ext + 1) * Hq * T);
    for (float & v : Q) v = dist(rng);
    for (float & v : Gate) v = dist(rng);
    for (float & v : rt) v = dist(rng) * 0.2f;

    std::vector<int32_t> bt = { 4, -1, 2, 5, 3 }; // block 1 evicted (-1)
    std::vector<float> K_shared((size_t) Hk * pref_len * D), V_shared((size_t) Hk * pref_len * D);
    std::vector<float> K_early((size_t) (P_tree - 1) * Hk * tail_cap * D), V_early((size_t) (P_tree - 1) * Hk * tail_cap * D);
    std::vector<float> K_tree((size_t) P_tree * Hk * S_max * D), V_tree((size_t) P_tree * Hk * S_max * D);
    for (float & v : K_shared) v = dist(rng);
    for (float & v : V_shared) v = dist(rng);
    for (float & v : K_early)  v = dist(rng);
    for (float & v : V_early)  v = dist(rng);
    for (float & v : K_tree)   v = dist(rng);
    for (float & v : V_tree)   v = dist(rng);

    const int planes[T] = { 0, 1, 2 };
    const int q_pos[T]  = { 64, 65, 65 };
    const std::vector<int> paths[T] = { {1}, {1, 4}, {1, 4} };
    std::vector<int32_t> meta((size_t) m_stride * T, -1);
    for (int t = 0; t < T; ++t) {
        int32_t * mt = &meta[(size_t) t * m_stride];
        mt[0] = paths[t].back();
        mt[1] = q_pos[t];
        mt[2] = planes[t];
        for (size_t e = 0; e < paths[t].size(); ++e) mt[11 + e] = paths[t][e];
    }

    // Exact CPU reference: early planes < 2 read r=R pool for kp < 48 and early ring for 48 <= kp < 64
    std::vector<float> ref((size_t) D * Hq * T, 0.0f);
    const float scale = 1.0f / std::sqrt((float) D);
    for (int t = 0; t < T; ++t) {
        const int pl = planes[t], qp_i = q_pos[t];
        for (int h = 0; h < Hq; ++h) {
            const int kh = h / (Hq / Hk);
            const float * qp = &Q[((size_t) t * Hq + h) * D];
            const float * rtp = &rt[((size_t) t * Hq + h) * (rel_ext + 1)];
            std::vector<float> scores(qp_i + 1, -INFINITY);
            float mx = 0.0f;
            auto get_kv = [&](int kp, bool is_v) -> const float * {
                const auto & sh = is_v ? V_shared : K_shared;
                const auto & ea = is_v ? V_early  : K_early;
                const auto & tr = is_v ? V_tree   : K_tree;
                if (kp >= pref_len) return &tr[(((size_t) pl * Hk + kh) * S_max + paths[t][kp - pref_len]) * D];
                if (pl < P_tree - 1 && kp >= pref_len - shared_w) {
                    return &ea[(((size_t) pl * Hk + kh) * tail_cap + (kp % tail_cap)) * D];
                }
                if (bt[kp / block_size] < 0) return nullptr;
                return &sh[((size_t) kh * pref_len + kp) * D];
            };
            for (int kp = 0; kp <= qp_i; ++kp) {
                const float * kptr = get_kv(kp, false);
                if (!kptr) continue;
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += qp[d] * kptr[d];
                scores[kp] = dot * scale + rtp[std::min(qp_i - kp, rel_ext)];
                mx = std::max(mx, scores[kp]);
            }
            float sum = std::exp(0.0f - mx);
            std::vector<float> probs(qp_i + 1, 0.0f);
            for (int kp = 0; kp <= qp_i; ++kp) {
                if (std::isfinite(scores[kp])) { probs[kp] = std::exp(scores[kp] - mx); sum += probs[kp]; }
            }
            float * op = &ref[((size_t) t * Hq + h) * D];
            const float * gp = &Gate[((size_t) t * Hq + h) * D];
            for (int d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (int kp = 0; kp <= qp_i; ++kp) {
                    if (probs[kp] > 0.0f) acc += (probs[kp] / sum) * get_kv(kp, true)[d];
                }
                op[d] = acc / (1.0f + std::exp(-gp[d]));
            }
        }
    }

    const ggml_type kv_t = use_q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    const size_t row_b = ggml_row_size(kv_t, D);
    auto pack_row = [&](const float * src, uint8_t * dst) {
        if (use_q8) ggml_quantize_chunk(GGML_TYPE_Q8_0, src, dst, 0, 1, D, nullptr);
        else { auto * hp = reinterpret_cast<ggml_fp16_t *>(dst); for (int d = 0; d < D; ++d) hp[d] = ggml_fp32_to_fp16(src[d]); }
    };
    std::vector<uint8_t> pk_buf((size_t) P_storage * Hk * prefix_cap * row_b, 0x7F);
    std::vector<uint8_t> pv_buf((size_t) P_storage * Hk * prefix_cap * row_b, 0x7F);
    std::vector<uint8_t> tk_buf((size_t) P_tree * Hk * S_max * row_b, 0);
    std::vector<uint8_t> tv_buf((size_t) P_tree * Hk * S_max * row_b, 0);
    for (int h = 0; h < Hk; ++h) {
        for (int kp = 0; kp < pref_len; ++kp) {
            const int pb = bt[kp / block_size];
            if (pb < 0) continue;
            const int slot = pb * block_size + (kp % block_size);
            pack_row(&K_shared[((size_t) h * pref_len + kp) * D], pk_buf.data() + ((size_t) h * prefix_cap + slot) * row_b);
            pack_row(&V_shared[((size_t) h * pref_len + kp) * D], pv_buf.data() + ((size_t) h * prefix_cap + slot) * row_b);
        }
        for (int pl = 0; pl < P_tree - 1; ++pl) {
            for (int r = 0; r < tail_cap; ++r) {
                const int slot = pool_cap + pl * tail_cap + r;
                pack_row(&K_early[(((size_t) pl * Hk + h) * tail_cap + r) * D], pk_buf.data() + ((size_t) h * prefix_cap + slot) * row_b);
                pack_row(&V_early[(((size_t) pl * Hk + h) * tail_cap + r) * D], pv_buf.data() + ((size_t) h * prefix_cap + slot) * row_b);
            }
        }
        for (int pl = 0; pl < P_tree; ++pl) {
            for (int s = 0; s < S_max; ++s) {
                pack_row(&K_tree[(((size_t) pl * Hk + h) * S_max + s) * D], tk_buf.data() + (((size_t) pl * Hk + h) * S_max + s) * row_b);
                pack_row(&V_tree[(((size_t) pl * Hk + h) * S_max + s) * D], tv_buf.data() + (((size_t) pl * Hk + h) * S_max + s) * row_b);
            }
        }
    }

    std::vector<int32_t> plb(2 + bt.size());
    plb[0] = pref_len; plb[1] = shared_w;
    for (size_t i = 0; i < bt.size(); ++i) plb[2 + i] = bt[i];

    struct ggml_init_params gp = { 256 * ggml_tensor_overhead() + ggml_graph_overhead() + 8192, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * t_Q    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hq, T);
    ggml_tensor * t_pk   = ggml_new_tensor_4d(ctx, kv_t, D, prefix_cap, Hk, P_storage);
    ggml_tensor * t_pv   = ggml_new_tensor_4d(ctx, kv_t, D, prefix_cap, Hk, P_storage);
    ggml_tensor * t_tk   = ggml_new_tensor_4d(ctx, kv_t, D, S_max, Hk, P_tree);
    ggml_tensor * t_tv   = ggml_new_tensor_4d(ctx, kv_t, D, S_max, Hk, P_tree);
    ggml_tensor * t_rt   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rel_ext + 1, Hq, T);
    ggml_tensor * t_meta = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, m_stride, T);
    ggml_tensor * t_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D * Hq, T);
    ggml_tensor * t_plb  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) plb.size());
    ggml_tensor * t_pc   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 16, P_tree);
    ggml_tensor * t_tc   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, S_max, P_tree);
    ggml_tensor * t_cs   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    ggml_tensor * t_cm   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    ggml_tensor * t_bt   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_blocks);
    ggml_set_input(t_Q); ggml_set_input(t_pk); ggml_set_input(t_pv); ggml_set_input(t_tk); ggml_set_input(t_tv);
    ggml_set_input(t_rt); ggml_set_input(t_meta); ggml_set_input(t_gate); ggml_set_input(t_plb);
    ggml_set_input(t_pc); ggml_set_input(t_tc); ggml_set_input(t_cs); ggml_set_input(t_cm); ggml_set_input(t_bt);

    ggml_tensor * attn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D * Hq, T);
    attn->op = GGML_OP_INKLING2_OP;
    attn->src[0] = t_Q; attn->src[1] = t_pk; attn->src[2] = t_pv;
    attn->src[3] = t_tk; attn->src[4] = t_tv; attn->src[5] = t_rt;
    attn->src[6] = t_meta; attn->src[7] = t_plb; attn->src[8] = t_gate;
    test_set_op_i32(attn, 0, 35);
    test_set_op_i32(attn, 1, 0);
    test_set_op_i32(attn, 2, rel_ext);
    test_set_op_i32(attn, 3, m_stride);
    test_set_op_i32(attn, 4, path_w);
    test_set_op_i32(attn, 5, block_size);
    test_set_op_i32(attn, 6, 0);
    test_set_op_i32(attn, 7, 0);
    test_set_op_i32(attn, 8, shared_w);
    test_set_op_i32(attn, 9, tail_cap);

    ggml_tensor * cmt = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    cmt->op = GGML_OP_INKLING2_OP;
    cmt->src[0] = t_pk; cmt->src[1] = t_pv; cmt->src[2] = t_tk; cmt->src[3] = t_tv;
    cmt->src[4] = t_pc; cmt->src[5] = t_tc; cmt->src[6] = t_cs; cmt->src[7] = t_cm;
    cmt->src[8] = t_bt;
    test_set_op_i32(cmt, 0, 36);
    test_set_op_i32(cmt, 1, 0);
    test_set_op_i32(cmt, 2, block_size);
    test_set_op_i32(cmt, 3, shared_w);
    test_set_op_i32(cmt, 4, tail_cap);

    ggml_set_output(attn); ggml_set_output(cmt); ggml_set_output(t_pk); ggml_set_output(t_pv);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, attn);
    ggml_build_forward_expand(gf, cmt);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);

    const int32_t cs[2] = { 1, 4 };
    const int32_t cm[2] = { pref_len, 2 };
    ggml_backend_tensor_set(t_Q, Q.data(), 0, Q.size() * sizeof(float));
    ggml_backend_tensor_set(t_pk, pk_buf.data(), 0, pk_buf.size());
    ggml_backend_tensor_set(t_pv, pv_buf.data(), 0, pv_buf.size());
    ggml_backend_tensor_set(t_tk, tk_buf.data(), 0, tk_buf.size());
    ggml_backend_tensor_set(t_tv, tv_buf.data(), 0, tv_buf.size());
    ggml_backend_tensor_set(t_rt, rt.data(), 0, rt.size() * sizeof(float));
    ggml_backend_tensor_set(t_meta, meta.data(), 0, meta.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_gate, Gate.data(), 0, Gate.size() * sizeof(float));
    ggml_backend_tensor_set(t_plb, plb.data(), 0, plb.size() * sizeof(int32_t));
    ggml_backend_tensor_memset(t_pc, 0, 0, ggml_nbytes(t_pc));
    ggml_backend_tensor_memset(t_tc, 0, 0, ggml_nbytes(t_tc));
    ggml_backend_tensor_set(t_cs, cs, 0, sizeof(cs));
    ggml_backend_tensor_set(t_cm, cm, 0, sizeof(cm));
    ggml_backend_tensor_set(t_bt, bt.data(), 0, bt.size() * sizeof(int32_t));
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> got(ref.size());
    std::vector<uint8_t> pk_after(pk_buf.size()), pv_after(pv_buf.size());
    ggml_backend_tensor_get(attn, got.data(), 0, got.size() * sizeof(float));
    ggml_backend_tensor_get(t_pk, pk_after.data(), 0, pk_after.size());
    ggml_backend_tensor_get(t_pv, pv_after.data(), 0, pv_after.size());
    ggml_gallocr_free(ga);
    ggml_free(ctx);

    const float err = max_abs_diff(got, ref);
    const float tol = use_q8 ? 2e-2f : 2e-3f;
    bool cmt_ok = true;
    for (int h = 0; h < Hk; ++h) {
        for (int i = 0; i < 2; ++i) {
            const int pos = pref_len + i;
            // Plane 2 (r=R) -> paged pool block 4 (phys 3)
            const int r_slot = bt[pos / block_size] * block_size + (pos % block_size);
            if (std::memcmp(pk_after.data() + ((size_t) h * prefix_cap + r_slot) * row_b,
                            tk_buf.data() + (((size_t) 2 * Hk + h) * S_max + cs[i]) * row_b, row_b) != 0) cmt_ok = false;
            if (std::memcmp(pv_after.data() + ((size_t) h * prefix_cap + r_slot) * row_b,
                            tv_buf.data() + (((size_t) 2 * Hk + h) * S_max + cs[i]) * row_b, row_b) != 0) cmt_ok = false;
            // Planes 0 & 1 -> early W-rings packed at pool_cap + pl * tail_cap
            for (int pl = 0; pl < P_tree - 1; ++pl) {
                const int e_slot = pool_cap + pl * tail_cap + (pos % tail_cap);
                if (std::memcmp(pk_after.data() + ((size_t) h * prefix_cap + e_slot) * row_b,
                                tk_buf.data() + (((size_t) pl * Hk + h) * S_max + cs[i]) * row_b, row_b) != 0) cmt_ok = false;
                if (std::memcmp(pv_after.data() + ((size_t) h * prefix_cap + e_slot) * row_b,
                                tv_buf.data() + (((size_t) pl * Hk + h) * S_max + cs[i]) * row_b, row_b) != 0) cmt_ok = false;
            }
        }
    }
    std::printf("[test_inkling2_ops] 2dsd_tree_shared_kv(%s, W=%d) err=%.3e cmt_kv=%s\n",
                use_q8 ? "Q8_0" : "F16", shared_w, err, cmt_ok ? "EXACT" : "DIFF");
    return err < tol && cmt_ok;
}

// ─── Step 2 Unit Test C: Sliding 5-Layer Drafter Ring Mask at Active Cap ──
static bool test_2dsd_draft_sliding_ring_mask(ggml_backend_t backend) {
    const int ring_cap = 256, L_b = 4, S_b = 8, B = 8;
    const int kv_span = ((ring_cap + L_b * S_b + L_b * B + 255) / 256) * 256; // 512
    const int32_t dmeta[3 + L_b] = { ring_cap, L_b, S_b, 3, 0, 5, 1 };

    struct ggml_init_params gp = { 32 * ggml_tensor_overhead() + ggml_graph_overhead() + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * t_dm = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3 + L_b);
    ggml_set_input(t_dm);
    ggml_tensor * t_m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_span, (int64_t) B * L_b);
    t_m->op = GGML_OP_INKLING2_OP;
    t_m->src[0] = t_dm;
    test_set_op_i32(t_m, 0, 39);
    test_set_op_i32(t_m, 1, B);
    ggml_set_output(t_m);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_m);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(t_dm, dmeta, 0, sizeof(dmeta));
    ggml_backend_graph_compute(backend, gf);

    std::vector<ggml_fp16_t> got((size_t) kv_span * B * L_b);
    ggml_backend_tensor_get(t_m, got.data(), 0, got.size() * sizeof(ggml_fp16_t));
    ggml_gallocr_free(ga);
    ggml_free(ctx);

    bool ok = true;
    for (int l = 0; l < L_b; ++l) {
        const int slen = dmeta[3 + l];
        const int span0 = ring_cap + l * S_b;
        const int blk0  = ring_cap + L_b * S_b + l * B;
        for (int q = 0; q < B; ++q) {
            const ggml_fp16_t * row = &got[(size_t) (l * B + q) * kv_span];
            for (int k = 0; k < kv_span; ++k) {
                const bool want = (k < ring_cap)
                    || (k >= span0 && k < span0 + slen)
                    || (k >= blk0 && k < blk0 + B);
                const float v = ggml_fp16_to_fp32(row[k]);
                if (want ? (v != 0.0f) : std::isfinite(v)) ok = false;
            }
        }
    }
    std::printf("[test_inkling2_ops] 2dsd_draft_sliding_ring_mask(cap=%d, span=%d): %s\n",
                ring_cap, kv_span, ok ? "PASS" : "FAIL");
    return ok;
}

// ─── Paged Subop 43 (TREE_PREFILL_CLONE_CODA) & Multi-Slot Paged Dual-Tier Subop 35 ──
static bool test_2dsd_paged_clone_and_multislot_paged(ggml_backend_t backend) {
    // 1. Paged subop 43 (TREE_PREFILL_CLONE_CODA) with shuffled block_table & Q8_0 KV
    const int D = 128, Hk = 2, P = 3, block_size = 16, n_pool_blks = 4, cap = n_pool_blks * block_size;
    const int max_blocks = 6, pos0 = 32, n_tok = 24, s_ch = 64;
    const size_t row_b = ggml_row_size(GGML_TYPE_Q8_0, D);
    std::mt19937 rng(4343);
    std::uniform_int_distribution<int> byte_dist(1, 200);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);

    std::vector<uint8_t> pk_in((size_t) P * Hk * cap * row_b, 0x11);
    std::vector<uint8_t> pv_in((size_t) P * Hk * cap * row_b, 0x22);
    for (int h = 0; h < Hk; ++h) {
        for (int s = 0; s < cap; ++s) {
            uint8_t * kr = pk_in.data() + (((size_t) (P - 1) * Hk + h) * cap + s) * row_b;
            uint8_t * vr = pv_in.data() + (((size_t) (P - 1) * Hk + h) * cap + s) * row_b;
            for (size_t b = 0; b < row_b; ++b) { kr[b] = (uint8_t) byte_dist(rng); vr[b] = (uint8_t) byte_dist(rng); }
        }
    }
    std::vector<float> pc_in((size_t) P * s_ch * 3, 0.0f);
    for (int c = 0; c < s_ch * 3; ++c) pc_in[(size_t) (P - 1) * s_ch * 3 + c] = fdist(rng);
    // Shuffled block table: logical blks 2 -> phys 3, 3 -> phys 1; others -1 or untouched phys 0, 2
    const int32_t bt_arr[max_blocks] = { 0, -1, 3, 1, 2, -1 };

    bool clone_ok = true;
    {
        struct ggml_init_params gp = { 32 * ggml_tensor_overhead() + ggml_graph_overhead() + 4096, nullptr, true };
        ggml_context * ctx = ggml_init(gp);
        ggml_tensor * t_pk  = ggml_new_tensor_4d(ctx, GGML_TYPE_Q8_0, D, cap, Hk, P);
        ggml_tensor * t_pv  = ggml_new_tensor_4d(ctx, GGML_TYPE_Q8_0, D, cap, Hk, P);
        ggml_tensor * t_pc  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, s_ch, P);
        ggml_tensor * t_dep = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_tensor * t_bt  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_blocks);
        ggml_set_input(t_pk); ggml_set_input(t_pv); ggml_set_input(t_pc);
        ggml_set_input(t_dep); ggml_set_input(t_bt);
        ggml_tensor * op = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        op->op = GGML_OP_INKLING2_OP;
        op->src[0] = t_pk; op->src[1] = t_pv; op->src[2] = t_pc; op->src[3] = t_dep; op->src[4] = t_bt;
        test_set_op_i32(op, 0, 43);
        test_set_op_i32(op, 1, pos0);
        test_set_op_i32(op, 2, n_tok);
        test_set_op_i32(op, 3, 0);
        test_set_op_i32(op, 4, block_size);
        test_set_op_i32(op, 5, 0);
        test_set_op_i32(op, 6, 0);
        ggml_set_output(op);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, op);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(ga, gf);
        float dep_v = 0.0f;
        ggml_backend_tensor_set(t_pk, pk_in.data(), 0, pk_in.size());
        ggml_backend_tensor_set(t_pv, pv_in.data(), 0, pv_in.size());
        ggml_backend_tensor_set(t_pc, pc_in.data(), 0, pc_in.size() * sizeof(float));
        ggml_backend_tensor_set(t_dep, &dep_v, 0, sizeof(dep_v));
        ggml_backend_tensor_set(t_bt, bt_arr, 0, sizeof(bt_arr));
        ggml_backend_graph_compute(backend, gf);

        std::vector<uint8_t> pk_out(pk_in.size()), pv_out(pv_in.size());
        std::vector<float> pc_out(pc_in.size());
        ggml_backend_tensor_get(t_pk, pk_out.data(), 0, pk_out.size());
        ggml_backend_tensor_get(t_pv, pv_out.data(), 0, pv_out.size());
        ggml_backend_tensor_get(t_pc, pc_out.data(), 0, pc_out.size() * sizeof(float));
        ggml_gallocr_free(ga);
        ggml_free(ctx);

        std::vector<uint8_t> cloned_slot(cap, 0);
        for (int t = 0; t < n_tok; ++t) {
            const int pos = pos0 + t;
            const int sl  = bt_arr[pos / block_size] * block_size + (pos % block_size);
            cloned_slot[(size_t) sl] = 1;
        }
        for (int p = 0; p < P - 1; ++p) {
            for (int h = 0; h < Hk; ++h) {
                for (int s = 0; s < cap; ++s) {
                    const size_t doff = (((size_t) p * Hk + h) * cap + s) * row_b;
                    const size_t soff = (((size_t) (P - 1) * Hk + h) * cap + s) * row_b;
                    if (cloned_slot[(size_t) s]) {
                        if (std::memcmp(pk_out.data() + doff, pk_in.data() + soff, row_b) != 0) clone_ok = false;
                        if (std::memcmp(pv_out.data() + doff, pv_in.data() + soff, row_b) != 0) clone_ok = false;
                    } else {
                        if (pk_out[doff] != 0x11 || pv_out[doff] != 0x22) clone_ok = false;
                    }
                }
            }
            for (int c = 0; c < s_ch * 3; ++c) {
                if (std::fabs(pc_out[(size_t) p * s_ch * 3 + c] - pc_in[(size_t) (P - 1) * s_ch * 3 + c]) > 1e-6f) {
                    clone_ok = false;
                }
            }
        }
    }

    // 2. Multi-slot (n_seq=2, P=3 -> p_storage=6) paged dual-tier subop 35 parity vs single-slot CPU oracle
    const int n_seq = 2, P_tot = P * n_seq, Hq = Hk * 3, S_max = 8, rel_ext = 32, path_w = 8, m_stride = 11 + path_w;
    const int T = 4; // 2 rows for slot 0 (planes 0, 2) + 2 rows for slot 1 (planes 3, 5)
    const int pref_lens[2] = { 40, 56 }; // slot 0 has 3 blks (0..2), slot 1 has 4 blks (0..3)
    const int32_t bt_loc[2][max_blocks] = {
        { 0, -1, 2, -1, -1, -1 }, // slot 0: blk 0 & 2 in Tier 0, blk 1 in Tier 1.5
        { 1,  3, -1, 0, -1, -1 }, // slot 1: blk 0, 1, 3 in Tier 0, blk 2 in Tier 1.5
    };
    const int32_t bt_rem[2][max_blocks] = {
        { -1, 1, -1, -1, -1, -1 }, // slot 0: blk 1 in Tier 1.5 phys 1
        { -1, -1, 0, -1, -1, -1 }, // slot 1: blk 2 in Tier 1.5 phys 0
    };
    std::vector<float> q_in((size_t) D * Hq * T), rt_in((size_t) (rel_ext + 1) * Hq * T), gate_in((size_t) D * Hq * T);
    for (float & v : q_in) v = fdist(rng);
    for (float & v : rt_in) v = fdist(rng) * 0.25f;
    for (float & v : gate_in) v = fdist(rng);

    std::vector<float> l_kf((size_t) P_tot * Hk * cap * D), l_vf(l_kf.size());
    std::vector<float> r_kf((size_t) P_tot * Hk * cap * D), r_vf(r_kf.size());
    std::vector<float> t_kf((size_t) P_tot * Hk * S_max * D), t_vf(t_kf.size());
    for (float & v : l_kf) v = fdist(rng);
    for (float & v : l_vf) v = fdist(rng);
    for (float & v : r_kf) v = fdist(rng);
    for (float & v : r_vf) v = fdist(rng);
    for (float & v : t_kf) v = fdist(rng);
    for (float & v : t_vf) v = fdist(rng);

    auto to_f16 = [](const std::vector<float> & src) {
        std::vector<ggml_fp16_t> dst(src.size());
        ggml_fp32_to_fp16_row(src.data(), dst.data(), (int64_t) src.size());
        return dst;
    };
    auto lk16 = to_f16(l_kf), lv16 = to_f16(l_vf), rk16 = to_f16(r_kf), rv16 = to_f16(r_vf), tk16 = to_f16(t_kf), tv16 = to_f16(t_vf);

    const int row_planes[T] = { 0, 2, 3, 5 }; // seq0: p0, p2; seq1: p0, p2
    const int row_qpos[T]   = { 40, 41, 56, 57 };
    const int row_tslot[T]  = { 1, 2, 3, 4 };
    std::vector<int32_t> meta((size_t) T * m_stride, -1);
    for (int r = 0; r < T; ++r) {
        int32_t * mt = meta.data() + (size_t) r * m_stride;
        const int s = row_planes[r] / P;
        mt[0] = row_tslot[r];
        mt[1] = row_qpos[r];
        mt[2] = row_planes[r];
        mt[3] = 0; mt[4] = 0;
        const int n_tree = row_qpos[r] - pref_lens[s] + 1;
        for (int e = 0; e < n_tree; ++e) mt[11 + e] = (e + 1 == n_tree) ? row_tslot[r] : row_tslot[r - 1];
    }

    const int pl_cols = 2 + max_blocks;
    std::vector<int32_t> pl_loc((size_t) pl_cols * n_seq, -1), pl_rem((size_t) pl_cols * n_seq, -1);
    for (int s = 0; s < n_seq; ++s) {
        pl_loc[(size_t) s * pl_cols + 0] = pref_lens[s]; pl_loc[(size_t) s * pl_cols + 1] = 0;
        pl_rem[(size_t) s * pl_cols + 0] = pref_lens[s]; pl_rem[(size_t) s * pl_cols + 1] = 0;
        for (int b = 0; b < max_blocks; ++b) {
            pl_loc[(size_t) s * pl_cols + 2 + b] = bt_loc[s][b];
            pl_rem[(size_t) s * pl_cols + 2 + b] = bt_rem[s][b];
        }
    }

    std::vector<float> got((size_t) D * Hq * T, 0.0f);
    {
        struct ggml_init_params gp = { 64 * ggml_tensor_overhead() + ggml_graph_overhead() + 8192, nullptr, true };
        ggml_context * ctx = ggml_init(gp);
        ggml_tensor * t_Q    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hq, T);
        ggml_tensor * t_lk   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, cap, Hk, P_tot);
        ggml_tensor * t_lv   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, cap, Hk, P_tot);
        ggml_tensor * t_rk   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, cap, Hk, P_tot);
        ggml_tensor * t_rv   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, cap, Hk, P_tot);
        ggml_tensor * t_tk   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, S_max, Hk, P_tot);
        ggml_tensor * t_tv   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, S_max, Hk, P_tot);
        ggml_tensor * t_rt   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rel_ext + 1, Hq, T);
        ggml_tensor * t_meta = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, m_stride, T);
        ggml_tensor * t_pll  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, pl_cols, n_seq);
        ggml_tensor * t_plr  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, pl_cols, n_seq);
        ggml_tensor * t_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D * Hq, T);
        ggml_set_input(t_Q); ggml_set_input(t_lk); ggml_set_input(t_lv); ggml_set_input(t_rk); ggml_set_input(t_rv);
        ggml_set_input(t_tk); ggml_set_input(t_tv); ggml_set_input(t_rt); ggml_set_input(t_meta);
        ggml_set_input(t_pll); ggml_set_input(t_plr); ggml_set_input(t_gate);

        ggml_tensor * part_r = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (D + 2) * Hq, T);
        part_r->op = GGML_OP_INKLING2_OP;
        part_r->src[0] = t_Q; part_r->src[1] = t_rk; part_r->src[2] = t_rv;
        part_r->src[5] = t_rt; part_r->src[6] = t_meta; part_r->src[7] = t_plr;
        test_set_op_i32(part_r, 0, 35); test_set_op_i32(part_r, 1, 0); test_set_op_i32(part_r, 2, rel_ext);
        test_set_op_i32(part_r, 3, m_stride); test_set_op_i32(part_r, 4, path_w); test_set_op_i32(part_r, 5, block_size);
        test_set_op_i32(part_r, 6, 1); test_set_op_i32(part_r, 7, 0); test_set_op_i32(part_r, 8, 0);
        test_set_op_i32(part_r, 9, 0); test_set_op_i32(part_r, 10, P);

        ggml_tensor * part_l = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (D + 2) * Hq, T);
        part_l->op = GGML_OP_INKLING2_OP;
        part_l->src[0] = t_Q; part_l->src[1] = t_lk; part_l->src[2] = t_lv;
        part_l->src[3] = t_tk; part_l->src[4] = t_tv;
        part_l->src[5] = t_rt; part_l->src[6] = t_meta; part_l->src[7] = t_pll;
        test_set_op_i32(part_l, 0, 35); test_set_op_i32(part_l, 1, 0); test_set_op_i32(part_l, 2, rel_ext);
        test_set_op_i32(part_l, 3, m_stride); test_set_op_i32(part_l, 4, path_w); test_set_op_i32(part_l, 5, block_size);
        test_set_op_i32(part_l, 6, 1); test_set_op_i32(part_l, 7, 1); test_set_op_i32(part_l, 8, 0);
        test_set_op_i32(part_l, 9, 0); test_set_op_i32(part_l, 10, P);

        ggml_tensor * comb = ggml_inkling2_attn_split_combine(ctx, part_r, part_l, t_gate, D, Hq);
        ggml_set_output(comb);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, comb);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_tensor_set(t_Q, q_in.data(), 0, q_in.size() * sizeof(float));
        ggml_backend_tensor_set(t_lk, lk16.data(), 0, lk16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(t_lv, lv16.data(), 0, lv16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(t_rk, rk16.data(), 0, rk16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(t_rv, rv16.data(), 0, rv16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(t_tk, tk16.data(), 0, tk16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(t_tv, tv16.data(), 0, tv16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(t_rt, rt_in.data(), 0, rt_in.size() * sizeof(float));
        ggml_backend_tensor_set(t_meta, meta.data(), 0, meta.size() * sizeof(int32_t));
        ggml_backend_tensor_set(t_pll, pl_loc.data(), 0, pl_loc.size() * sizeof(int32_t));
        ggml_backend_tensor_set(t_plr, pl_rem.data(), 0, pl_rem.size() * sizeof(int32_t));
        ggml_backend_tensor_set(t_gate, gate_in.data(), 0, gate_in.size() * sizeof(float));
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_tensor_get(comb, got.data(), 0, got.size() * sizeof(float));
        ggml_gallocr_free(ga);
        ggml_free(ctx);
    }

    // Exact FP32 CPU reference
    const float inv_s = 1.0f / std::sqrt((float) D);
    std::vector<float> ref(got.size(), 0.0f);
    for (int r = 0; r < T; ++r) {
        const int pl = row_planes[r], s = pl / P, qp = row_qpos[r], plen = pref_lens[s];
        const int32_t * mt = meta.data() + (size_t) r * m_stride;
        for (int h = 0; h < Hq; ++h) {
            const int kh = h / 3;
            const float * qv = q_in.data() + ((size_t) r * Hq + h) * D;
            const float * rt = rt_in.data() + ((size_t) r * Hq + h) * (rel_ext + 1);
            const float * gp = gate_in.data() + ((size_t) r * Hq + h) * D;
            std::vector<float> sc = { 0.0f };
            std::vector<const float *> vp = { nullptr };
            for (int kp = 0; kp <= qp; ++kp) {
                const float * kptr = nullptr, * vptr = nullptr;
                if (kp < plen) {
                    const int lb = kp / block_size, bo = kp % block_size;
                    if (bt_loc[s][lb] >= 0) {
                        const int sl = bt_loc[s][lb] * block_size + bo;
                        kptr = l_kf.data() + (((size_t) pl * Hk + kh) * cap + sl) * D;
                        vptr = l_vf.data() + (((size_t) pl * Hk + kh) * cap + sl) * D;
                    } else if (bt_rem[s][lb] >= 0) {
                        const int sl = bt_rem[s][lb] * block_size + bo;
                        kptr = r_kf.data() + (((size_t) pl * Hk + kh) * cap + sl) * D;
                        vptr = r_vf.data() + (((size_t) pl * Hk + kh) * cap + sl) * D;
                    }
                } else {
                    const int ts = mt[11 + (kp - plen)];
                    kptr = t_kf.data() + (((size_t) pl * Hk + kh) * S_max + ts) * D;
                    vptr = t_vf.data() + (((size_t) pl * Hk + kh) * S_max + ts) * D;
                }
                if (!kptr) continue;
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += qv[d] * inv_s * kptr[d];
                sc.push_back(dot + rt[std::min(qp - kp, rel_ext)]);
                vp.push_back(vptr);
            }
            float mx = sc[0]; for (float v : sc) mx = std::max(mx, v);
            float dn = 0.0f;  for (float & v : sc) { v = std::exp(v - mx); dn += v; }
            float * op = ref.data() + ((size_t) r * Hq + h) * D;
            for (int d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (size_t i = 0; i < sc.size(); ++i) if (vp[i]) acc += (sc[i] / dn) * vp[i][d];
                op[d] = acc * (1.0f / (1.0f + std::exp(-gp[d])));
            }
        }
    }
    const float ms_err = max_abs_diff(got, ref);
    std::printf("[test_inkling2_ops] 2dsd_paged_clone_and_ms_dual: clone_ok=%d ms_dual_err=%.3e\n",
                clone_ok ? 1 : 0, ms_err);
    return clone_ok && ms_err < 2e-3f;
}

} // namespace

int main(int argc, char ** argv) {
    bool cpu_only = false;
    int gpu = 0;
    if (const char * e = std::getenv("LUCE_TEST_GPU")) gpu = std::atoi(e);
    if (argc > 1) {
        if (std::strcmp(argv[1], "--cpu-only") == 0) cpu_only = true;
        else gpu = std::atoi(argv[1]);
    }
    if (cpu_only) {
        bool ok = true;
        ok &= test_2dsd_mask_embedding_pt();
        ok &= test_2dsd_executor();
        ok &= test_2dsd_nongreedy_and_multislot_session();
        std::printf("[test_inkling2_ops] (cpu-only) %s\n", ok ? "ALL PASSED" : "FAILED");
        return ok ? 0 : 1;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(gpu);
    if (!backend) {
        std::fprintf(stderr, "[test_inkling2_ops] no GPU %d available, skipping\n", gpu);
        return 77;
    }

    bool ok = true;
    ok &= test_situ(backend);
    ok &= test_sconv(backend);
    ok &= test_attn_rel(backend, false, 1,  97);  // F16 decode T=1
    ok &= test_attn_rel(backend, true,  8,  97);  // Q8_0 verify T=8
    ok &= test_attn_rel(backend, false, 32, 32);  // F16 prefill T=32 (WMMA!)
    ok &= test_attn_rel(backend, true,  32, 32);  // Q8_0 prefill T=32 (WMMA!)
    ok &= test_attn_rel(backend, true,  64, 64);  // Q8_0 prefill T=64 (WMMA!)
    ok &= test_attn_rel_paged(backend, false, 1,  97, 16, false);
    ok &= test_attn_rel_paged(backend, true,  8,  97, 16, false);
    ok &= test_attn_rel_paged(backend, true,  8,  97, 64, true);
    ok &= test_attn_rel_paged(backend, false, 32, 64, 16, true);
    ok &= test_attn_rel_paged(backend, true,  64, 64, 16, true);
    ok &= test_attn_rel_paged_dual_split_combine(backend, false, 8,  97, false);
    ok &= test_attn_rel_paged_dual_split_combine(backend, true,  8,  97, false);
    ok &= test_attn_rel_paged_dual_split_combine(backend, true,  8,  97, true);
    ok &= test_attn_rel_paged_dual_split_combine(backend, false, 64, 64, false);
    ok &= test_attn_rel_paged_dual_split_combine(backend, true,  64, 64, false);
    ok &= test_attn_rel_paged_dual_split_combine(backend, true,  64, 64, true);
    ok &= test_paged_kv_writes(backend, false, 5);
    ok &= test_paged_kv_writes(backend, true,  5);
    ok &= test_paged_kv_writes(backend, false, 24);
    ok &= test_paged_kv_writes(backend, true,  24);
    ok &= test_sigmoid_topk(backend);
    ok &= test_dynamic_split_lut(backend);
    ok &= test_dspark_topk(backend);
    ok &= test_dspark_markov_argmax(backend);
    ok &= test_moe_masked_sum(backend);
    ok &= test_sconv_kv_write_and_clip_r(backend);
    ok &= test_2dsd_tree_ops(backend, false);
    ok &= test_2dsd_tree_ops(backend, true);
    ok &= test_2dsd_tree_paged_and_dual_split(backend, false);
    ok &= test_2dsd_tree_paged_and_dual_split(backend, true);
    ok &= test_2dsd_tree_shared_kv(backend, false);
    ok &= test_2dsd_tree_shared_kv(backend, true);
    ok &= test_2dsd_draft_sliding_ring_mask(backend);
    ok &= test_2dsd_drafter_and_flags(backend);
    ok &= test_2dsd_mask_embedding_pt();
    ok &= test_2dsd_executor();
    ok &= test_2dsd_p0_fused_ops(backend);
    ok &= test_2dsd_multislot_tree_attn_rel(backend);
    ok &= test_2dsd_paged_clone_and_multislot_paged(backend);
    ok &= test_2dsd_nongreedy_and_multislot_session();

    ggml_backend_free(backend);
    std::printf("[test_inkling2_ops] %s\n", ok ? "ALL PASSED" : "FAILED");
    return ok ? 0 : 1;
}
