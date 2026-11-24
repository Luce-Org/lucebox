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

} // namespace

int main(int argc, char ** argv) {
    int gpu = 0;
    if (const char * e = std::getenv("LUCE_TEST_GPU")) gpu = std::atoi(e);
    if (argc > 1) gpu = std::atoi(argv[1]);

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
    ok &= test_sigmoid_topk(backend);
    ok &= test_dynamic_split_lut(backend);
    ok &= test_dspark_topk(backend);
    ok &= test_dspark_markov_argmax(backend);
    ok &= test_moe_masked_sum(backend);
    ok &= test_sconv_kv_write_and_clip_r(backend);
    ok &= test_2dsd_tree_ops(backend, false);
    ok &= test_2dsd_tree_ops(backend, true);
    ok &= test_2dsd_drafter_and_flags(backend);
    ok &= test_2dsd_mask_embedding_pt();
    ok &= test_2dsd_executor();

    ggml_backend_free(backend);
    std::printf("[test_inkling2_ops] %s\n", ok ? "ALL PASSED" : "FAILED");
    return ok ? 0 : 1;
}
