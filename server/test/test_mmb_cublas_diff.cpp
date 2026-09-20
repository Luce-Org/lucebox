// Semantics of a quantized MUL_MAT on the CUDA/HIP backend, checked under the
// mmb route (forced on): the 4-bit weight and the bf16 shadow that
// mmb_shadow_prepare builds from it must agree. blk.3.attn_output.weight is
// Q5_K at M=6144 K=2560. Usage: <M> <K> <T> [iq4|q6k|q5k] [sched].
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#if defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#endif

typedef void (*cast_fn)(const void *, void *, int64_t, void *);
cast_fn ggml_get_to_fp16_cuda(ggml_type type);
cast_fn ggml_get_to_bf16_cuda(ggml_type type);

static uint32_t g_state = 0x12345678u;
static float rnd() {
    g_state = g_state * 1664525u + 1013904223u;
    return (float) ((g_state >> 8) & 0xFFFFFF) / (float) 0x1000000 * 2.0f - 1.0f;
}

static bool run(ggml_backend_t backend, ggml_type wtype,
                const std::vector<uint8_t> & wq, const std::vector<float> & xf,
                int64_t M, int64_t K, int64_t T, bool use_sched, std::vector<float> & out) {
    ggml_init_params ip{};
    ip.mem_size = 16 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_tensor * w = ggml_new_tensor_2d(ctx, wtype, K, M);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    ggml_set_input(w); ggml_set_input(x);
    ggml_tensor * dst = ggml_mul_mat(ctx, w, x);
    ggml_set_output(dst);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));

    // The bf16 shadow is created by graph_optimize, which only the scheduler
    // calls. The weight already lives in a backend buffer, so the scheduler
    // leaves it in place and prepares the shadow before compute.
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_t cpu_fallback = nullptr;
    if (use_sched) {
        cpu_fallback = ggml_backend_cpu_init();
        ggml_backend_t backends[2] = { backend, cpu_fallback };
        ggml_backend_buffer_type_t bufts[2] = {
            ggml_backend_get_default_buffer_type(backend),
            ggml_backend_get_default_buffer_type(cpu_fallback) };
        sched = ggml_backend_sched_new(backends, bufts, 2, 4096, false, false);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            std::fprintf(stderr, "sched alloc failed\n");
            ggml_backend_sched_free(sched); ggml_backend_free(cpu_fallback);
            ggml_backend_buffer_free(buf); ggml_free(ctx);
            return false;
        }
    }

    const ggml_status st = sched
        ? ggml_backend_sched_graph_compute(sched, gf)
        : ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed (%d)\n", (int) st);
        if (sched) ggml_backend_sched_free(sched);
        if (cpu_fallback) ggml_backend_free(cpu_fallback);
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }
    out.resize((size_t) M * T);
    ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
    if (sched) ggml_backend_sched_free(sched);
    if (cpu_fallback) ggml_backend_free(cpu_fallback);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// Replicates mmb_shadow_prepare's K-quant conversion with the public cast entry
// points: dequantize to F16, then F16 -> BF16.
static bool build_shadow(ggml_backend_t gpu, ggml_type wtype, const std::vector<uint8_t> & wq,
                         int64_t M, int64_t K, std::vector<uint8_t> & shadow) {
    const int64_t n = K * M;
    ggml_init_params ip{};
    ip.mem_size = 16 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_tensor * w   = ggml_new_tensor_1d(ctx, wtype, n);
    ggml_tensor * tmp = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n);
    ggml_tensor * sh  = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, n);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    bool ok = buf && ggml_nbytes(w) == wq.size();
    if (ok) {
        const cast_fn to_f16  = ggml_get_to_fp16_cuda(wtype);
        const cast_fn to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F16);
        ok = to_f16 && to_bf16;
        if (ok) {
            ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
            void * stream = ggml_backend_cuda_get_stream(gpu);
            to_f16(w->data, tmp->data, n, stream);
            to_bf16(tmp->data, sh->data, n, stream);
            ggml_backend_synchronize(gpu);
            shadow.resize((size_t) n * 2);
            ggml_backend_tensor_get(sh, shadow.data(), 0, shadow.size());
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

static double maxdiff(const std::vector<float> & a, const std::vector<float> & b) {
    double md = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) md = std::fmax(md, std::fabs((double) a[i] - b[i]));
    return md;
}
static double rms(const std::vector<float> & a) {
    double s = 0;
    for (float v : a) s += (double) v * v;
    return a.empty() ? 0 : std::sqrt(s / a.size());
}

int main(int argc, char ** argv) {
    // The mmb route is opt-in; this differential exists to check it, so enable
    // it unless the caller explicitly disabled it.
#if defined(_WIN32)
    _putenv_s("GGML_CUDA_MMB", "1");
#else
    setenv("GGML_CUDA_MMB", "1", 0);
#endif
    const int64_t M = argc > 3 ? atoll(argv[1]) : 6144;
    const int64_t K = argc > 3 ? atoll(argv[2]) : 2560;
    const int64_t T = argc > 3 ? atoll(argv[3]) : 512;
    const char * tname = argc > 4 ? argv[4] : "q5k";
    ggml_type wtype = GGML_TYPE_IQ4_NL;
    for (const auto & e : { std::pair<const char *, ggml_type>{"iq4nl", GGML_TYPE_IQ4_NL},
                            {"iq4xs", GGML_TYPE_IQ4_XS}, {"iq2xxs", GGML_TYPE_IQ2_XXS},
                            {"iq2xs", GGML_TYPE_IQ2_XS},   {"iq2s", GGML_TYPE_IQ2_S},
                            {"iq3xxs", GGML_TYPE_IQ3_XXS}, {"iq3s", GGML_TYPE_IQ3_S},
                            {"iq1s", GGML_TYPE_IQ1_S},     {"q2_0", GGML_TYPE_Q2_0},
                            {"q4k", GGML_TYPE_Q4_K},       {"q4_0", GGML_TYPE_Q4_0},
                            {"q5_0", GGML_TYPE_Q5_0},      {"q5_1", GGML_TYPE_Q5_1},
                            {"q5k", GGML_TYPE_Q5_K},       {"q6k", GGML_TYPE_Q6_K} }) {
        if (std::strcmp(tname, e.first) == 0) wtype = e.second;
    }
    bool use_sched = false;
    for (int i = 4; i < argc; ++i) if (std::strcmp(argv[i], "sched") == 0) use_sched = true;
    const size_t row = ggml_row_size(wtype, K);

    std::vector<float>   wf((size_t) K * M), xf((size_t) K * T);
    std::vector<uint8_t> wq(row * M);
    for (auto & v : wf) v = rnd();
    for (auto & v : xf) v = rnd();
    const ggml_type_traits * tr = ggml_get_type_traits(wtype);
    for (int64_t m = 0; m < M; ++m) tr->from_float_ref(wf.data() + m * K, wq.data() + m * row, K);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!cpu || !gpu) { std::fprintf(stderr, "backend init failed\n"); return 1; }
    // The mmb route only engages on RDNA3.5/RDNA4; elsewhere the two GPU routes
    // differ by activation-quantization choice and the comparison is moot.
#if defined(GGML_USE_HIP)
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess ||
        (!std::strstr(prop.gcnArchName, "gfx1151") && !std::strstr(prop.gcnArchName, "gfx12"))) {
        std::fprintf(stderr, "[diff] SKIP: mmb is RDNA3.5/RDNA4 only\n");
        ggml_backend_free(cpu); ggml_backend_free(gpu);
        return 77;
    }
#endif

    std::vector<float> ref, out, sh_out;
    if (!run(cpu, wtype, wq, xf, M, K, T, false, ref)) return 1;
    if (!run(gpu, wtype, wq, xf, M, K, T, use_sched, out)) return 1;
    std::vector<uint8_t> shadow;
    if (!build_shadow(gpu, wtype, wq, M, K, shadow)) { std::fprintf(stderr, "shadow build failed\n"); return 1; }
    if (!run(gpu, GGML_TYPE_BF16, shadow, xf, M, K, T, false, sh_out)) return 1;
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);

    const double r = rms(out);
    // The CPU vec_dot quantizes activations (Q8_0/Q8_K), so it is only a sanity
    // bound for the mmb bf16 route, not an oracle.
    std::fprintf(stderr, "[diff] M=%lld K=%lld T=%lld type=%s%s cpu_maxdiff=%.6g cpu_rel=%.4g rms=%.4g\n",
        (long long) M, (long long) K, (long long) T, ggml_type_name(wtype), use_sched ? " sched" : "",
        maxdiff(out, ref), r > 0 ? maxdiff(out, ref) / r : 0.0, r);

    // The bf16 shadow must reproduce what the mmb kernel reads from the 4-bit
    // weight. Both routes are mmb here, so they share activation handling and
    // the only difference is the weight dequant.
    const bool ok = std::isfinite(r) && r > 0 && maxdiff(out, sh_out) < 0.02 * r;
    std::fprintf(stderr, "[diff] shadow_maxdiff=%.6g %s\n", maxdiff(out, sh_out), ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
