// Unit test for all custom GGML_OP_INKLING2_OP GPU kernels against exact CPU FP32 reference.
// Run on any CUDA/HIP GPU (pass GPU index via LUCE_TEST_GPU or argv[1], default 0).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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

    ggml_backend_free(backend);
    std::printf("[test_inkling2_ops] %s\n", ok ? "ALL PASSED" : "FAILED");
    return ok ? 0 : 1;
}
