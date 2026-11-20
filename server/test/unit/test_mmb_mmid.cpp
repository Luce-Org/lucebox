// qwen4exp MoE expert GEMM (MUL_MAT_ID) under the gfx1151 profile, at batch sizes on both sides of every
// dispatch cutover (MMVQ, MMQ, MMB), against an exact CPU reference built from the dequantized weights.
// Covers the broadcast input of gate/up and the per-expert input of down. LUCE_MMID_TELEMETRY=1 logs the path.
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <vector>

static uint32_t g_state = 0x9e3779b9u;
static uint32_t rnd_u() { g_state = g_state * 1664525u + 1013904223u; return g_state >> 8; }
static float rnd() { return (float) (rnd_u() & 0xFFFFFF) / (float) 0x1000000 * 2.0f - 1.0f; }

struct Case { ggml_type type; int64_t K, M, E, n_used; bool per_expert; };

static bool gpu_mmid(ggml_backend_t gpu, const Case & c, int64_t T, const std::vector<uint8_t> & wq,
                     const std::vector<float> & x, const std::vector<int32_t> & ids, std::vector<float> & out) {
    ggml_init_params ip{ 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w   = ggml_new_tensor_3d(ctx, c.type, c.K, c.M, c.E);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, c.per_expert ? c.n_used : 1, T);
    ggml_tensor * id  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_used, T);
    ggml_tensor * dst = ggml_mul_mat_id(ctx, w, b, id);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    bool ok = buf != nullptr;
    if (ok) {
        ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
        ggml_backend_tensor_set(b, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(id, ids.data(), 0, ids.size() * sizeof(int32_t));
        ok = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
        if (ok) { out.resize(ggml_nelements(dst)); ggml_backend_tensor_get(dst, out.data(), 0, ggml_nbytes(dst)); }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) { std::fprintf(stderr, "backend init failed\n"); return 1; }
    if (!ggml_backend_cuda_qwen4exp_supported(gpu)) {
        std::fprintf(stderr, "[mmid] SKIP: qwen4exp profile is gfx1151 only\n");
        ggml_backend_free(gpu);
        return 77;
    }
    ggml_backend_cuda_set_qwen4exp_profile(GGML_CUDA_QWEN4EXP_DEFAULT);

    const Case cases[] = { { GGML_TYPE_Q4_K,   2560, 128, 16, 4, false },
                           { GGML_TYPE_IQ4_XS, 2560, 128, 16, 4, false },
                           { GGML_TYPE_Q6_K,    512, 512, 16, 4, true  } };
    const int64_t Ts[] = { 1, 5, 16, 17, 31, 32, 63, 64, 100, 127, 128, 129, 255, 300, 511, 512, 600 };
    int fails = 0, n = 0;
    for (const Case & c : cases) {
        const ggml_type_traits * tr = ggml_get_type_traits(c.type);
        const size_t row = ggml_row_size(c.type, c.K);
        std::vector<float> wf((size_t) c.K * c.M * c.E), tmp(c.K);
        std::vector<uint8_t> wq(row * c.M * c.E);
        for (auto & v : wf) v = rnd();
        for (int64_t r = 0; r < c.M * c.E; ++r) {
            tr->from_float_ref(wf.data() + r * c.K, wq.data() + r * row, c.K);
            tr->to_float(wq.data() + r * row, wf.data() + r * c.K, c.K);   // the reference reads what the GPU reads
        }
        for (int64_t T : Ts) {
            const int64_t nb = c.per_expert ? c.n_used : 1;
            std::vector<float> x((size_t) c.K * nb * T), out;
            for (auto & v : x) v = rnd();
            // Skewed routing with distinct experts per token: experts 0-3 take most tokens, some get none.
            std::vector<int32_t> ids((size_t) c.n_used * T);
            for (int64_t t = 0; t < T; ++t)
                for (int64_t u = 0; u < c.n_used; ++u) {
                    int32_t e; bool dup;
                    do {
                        e = (int32_t) (rnd_u() % 3 ? rnd_u() % 4 : rnd_u() % c.E);
                        dup = false;
                        for (int64_t v = 0; v < u; ++v) dup |= ids[t * c.n_used + v] == e;
                    } while (dup);
                    ids[t * c.n_used + u] = e;
                }
            if (!gpu_mmid(gpu, c, T, wq, x, ids, out)) { std::fprintf(stderr, "[mmid] compute failed\n"); return 1; }
            double se = 0, sr = 0, md = 0;
            for (int64_t t = 0; t < T; ++t)
                for (int64_t u = 0; u < c.n_used; ++u) {
                    const float * xr = x.data() + (t * nb + (c.per_expert ? u : 0)) * c.K;
                    const float * we = wf.data() + (size_t) ids[t * c.n_used + u] * c.M * c.K;
                    for (int64_t m = 0; m < c.M; ++m) {
                        double ref = 0;
                        for (int64_t k = 0; k < c.K; ++k) ref += (double) we[m * c.K + k] * xr[k];
                        const double d = out[(t * c.n_used + u) * c.M + m] - ref;
                        se += d * d; sr += ref * ref; md = std::fmax(md, std::fabs(d));
                    }
                }
            const double rms = std::sqrt(sr / (T * c.n_used * c.M)), rel = std::sqrt(se / sr);
            // Activation rounding (bf16 in MMB, q8_1 in MMVQ/MMQ) stays well under 1% rms; a wrong
            // expert, row or token is O(1).
            const bool ok = std::isfinite(rel) && rel < 0.01 && md < 0.05 * rms;
            fails += !ok; ++n;
            std::fprintf(stderr, "[mmid] %-6s K=%lld M=%lld %s T=%-3lld rel_rms=%.2e max=%.2e (rms %.2f) %s\n",
                ggml_type_name(c.type), (long long) c.K, (long long) c.M, c.per_expert ? "per-expert" : "broadcast ",
                (long long) T, rel, md, rms, ok ? "OK" : "FAIL");
        }
    }
    ggml_backend_free(gpu);
    std::fprintf(stderr, "mmb-mmid: %d cases, %s\n", n, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
