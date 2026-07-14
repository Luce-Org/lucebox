// CPU-parity test for the DeepSeek-V4 Hierarchical-Controller pre-mix kernels
// (deepseek4_hc_cuda.cu). This is the #500 pattern (rms_norm_hip) applied to the
// hand-written DS4 HC path: it drives deepseek4_cuda_hc_pre (the host entry used by
// the decode path) and checks its two kernels against a CPU reference on the
// device's actual wavefront width, so it validates whichever mode the runner's card
// uses (gfx1100 / gfx1151 are wave32; a wave64 card exercises the wider reduction).
// Scope: deepseek4_cuda_hc_pre_mix (a separate mix-only entry with its own
// hc_sumsq_kernel/hc_mix_kernel reduction pair) and the device-pointer variants
// (_device, _device_params) are out of scope here.
//
// deepseek4_cuda_hc_pre runs, in order:
//   1. hc_mix_norm_kernel  -- per-row RMS-normalized dot fn[row] . state, one
//      block per mix row, __syncthreads shared-mem tree reduction (wave-agnostic
//      by construction, but the reduction width follows blockDim, so parity here
//      is the wave-width backstop for this kernel).
//   2. hc_finish_kernel    -- sigmoid pre/post gates + a row-softmax followed by
//      NHC=4 Sinkhorn normalization of the comb matrix, then the gated combine
//      working[d] = sum_h pre[h] * state[h][d].
//
// The CPU reference reproduces all three outputs (working, post, comb) as the
// kernel computes them (same fp32 arithmetic, same +sinkhorn_eps guards, same half
// round-trip on fn). Pass criterion: max |gpu - cpu| within a tight fp32 tolerance
// whose only slack is reduction ordering + the fn half round-trip. Being derived
// from the same spec, it catches arithmetic / reduction / wave-width regressions,
// not a shared misreading of the spec (e.g. a comb src/dst convention both agree on).
//
// Written in CUDA spellings; the hip_compat/ shim maps them onto HIP exactly as
// the kernel and the rms_norm_hip / flashprefill tests do.

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "deepseek4/deepseek4_hc_cuda.h"

using dflash::common::deepseek4_cuda_hc_pre;

#define CK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while (0)

static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

int main() {
    // DS4-Flash HC config: n_hc = 4 (mix_dim = 2*4 + 4*4 = 24). NOTE: the kernel's
    // d_mix is fixed at kMixDim=24 while deepseek4_cuda_hc_pre still admits n_hc<=8
    // (mix_dim up to 80), so n_hc>4 overflows d_mix in the kernel today -- a latent
    // kernel bug, deliberately not exercised here. n_embd chosen > block(256) so
    // hc_dim spans several strided load iterations in both reductions.
    constexpr int N_HC          = 4;
    constexpr int N_EMBD        = 512;
    constexpr int HC_DIM        = N_HC * N_EMBD;          // 2048
    constexpr int MIX_DIM       = 2 * N_HC + N_HC * N_HC; // 24
    constexpr int SINKHORN_ITER = 4;
    constexpr float EPS         = 1e-5f;

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("[ds4-hc-test] device=%s warpSize=%d n_hc=%d n_embd=%d sinkhorn_iters=%d\n",
                prop.name, prop.warpSize, N_HC, N_EMBD, SINKHORN_ITER);

    std::mt19937 rng(20260711u);
    std::uniform_real_distribution<float> sdist(-1.0f, 1.0f);   // hc_state
    std::uniform_real_distribution<float> fdist(-0.05f, 0.05f); // fn weights
    std::uniform_real_distribution<float> bdist(-0.5f, 0.5f);   // base
    std::uniform_real_distribution<float> cdist(0.25f, 0.75f);  // scale[0..2]

    std::vector<float>  hc_state(HC_DIM);
    std::vector<float>  fn_f((size_t)MIX_DIM * HC_DIM);
    std::vector<__half> fn_h(fn_f.size());
    std::vector<float>  scale(MIX_DIM, 0.0f);
    std::vector<float>  base(MIX_DIM);

    for (auto & v : hc_state) v = sdist(rng);
    for (size_t i = 0; i < fn_f.size(); ++i) { fn_f[i] = fdist(rng); fn_h[i] = __float2half(fn_f[i]); }
    for (auto & v : base) v = bdist(rng);
    scale[0] = cdist(rng); // pre_scale
    scale[1] = cdist(rng); // post_scale
    scale[2] = cdist(rng); // comb_scale

    // fn must already live on the device: deepseek4_cuda_hc_pre passes it straight
    // to the kernel without an H2D copy.
    __half * d_fn = nullptr;
    CK(cudaMalloc(&d_fn, fn_h.size() * sizeof(__half)));
    CK(cudaMemcpy(d_fn, fn_h.data(), fn_h.size() * sizeof(__half), cudaMemcpyHostToDevice));

    std::vector<float> working(N_EMBD), post(N_HC), comb((size_t)N_HC * N_HC);
    if (!deepseek4_cuda_hc_pre(hc_state.data(), d_fn, scale.data(), base.data(),
                               N_EMBD, N_HC, SINKHORN_ITER, EPS,
                               working.data(), post.data(), comb.data())) {
        std::fprintf(stderr, "[ds4-hc-test] FAIL: deepseek4_cuda_hc_pre returned false\n");
        cudaFree(d_fn);
        return 1;
    }
    cudaFree(d_fn);

    // ---- CPU reference (mirrors hc_mix_norm_kernel + hc_finish_kernel) ----
    // inv_rms is over the whole hc_dim state and identical for every mix row.
    double sumsq = 0.0;
    for (int i = 0; i < HC_DIM; ++i) sumsq += (double)hc_state[i] * hc_state[i];
    const float inv_rms = 1.0f / std::sqrt((float)(sumsq / (double)HC_DIM) + EPS);

    std::vector<float> mix(MIX_DIM);
    for (int row = 0; row < MIX_DIM; ++row) {
        double dot = 0.0;
        const __half * w = fn_h.data() + (size_t)row * HC_DIM;
        for (int c = 0; c < HC_DIM; ++c) dot += (double)__half2float(w[c]) * hc_state[c];
        mix[row] = (float)dot * inv_rms;
    }

    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    const float sk_eps     = 1.0e-6f;

    std::vector<float> ref_pre(N_HC), ref_post(N_HC), ref_comb((size_t)N_HC * N_HC);
    for (int i = 0; i < N_HC; ++i) ref_pre[i]  = sigmoidf(mix[i] * pre_scale + base[i]) + sk_eps;
    for (int i = 0; i < N_HC; ++i) ref_post[i] = 2.0f * sigmoidf(mix[N_HC + i] * post_scale + base[N_HC + i]);

    // comb: per-dst row softmax over src, then one column norm, then the Sinkhorn
    // sweeps (dst-norm over src, src-norm over dst) for sinkhorn_iters-1 iters.
    std::vector<float> c((size_t)N_HC * N_HC);
    for (int dst = 0; dst < N_HC; ++dst) {
        float row_max = -1.0e30f;
        for (int src = 0; src < N_HC; ++src) {
            const int idx = src + dst * N_HC;
            const float v = mix[2 * N_HC + idx] * comb_scale + base[2 * N_HC + idx];
            c[idx] = v;
            row_max = v > row_max ? v : row_max;
        }
        float row_sum = 0.0f;
        for (int src = 0; src < N_HC; ++src) { const int idx = src + dst * N_HC; c[idx] = std::exp(c[idx] - row_max); row_sum += c[idx]; }
        const float inv = 1.0f / row_sum;
        for (int src = 0; src < N_HC; ++src) c[src + dst * N_HC] = c[src + dst * N_HC] * inv + sk_eps;
    }
    for (int src = 0; src < N_HC; ++src) {
        float sum = 0.0f;
        for (int dst = 0; dst < N_HC; ++dst) sum += c[src + dst * N_HC];
        const float inv = 1.0f / (sum + sk_eps);
        for (int dst = 0; dst < N_HC; ++dst) c[src + dst * N_HC] *= inv;
    }
    for (int iter = 1; iter < SINKHORN_ITER; ++iter) {
        for (int dst = 0; dst < N_HC; ++dst) {
            float sum = 0.0f;
            for (int src = 0; src < N_HC; ++src) sum += c[src + dst * N_HC];
            const float inv = 1.0f / (sum + sk_eps);
            for (int src = 0; src < N_HC; ++src) c[src + dst * N_HC] *= inv;
        }
        for (int src = 0; src < N_HC; ++src) {
            float sum = 0.0f;
            for (int dst = 0; dst < N_HC; ++dst) sum += c[src + dst * N_HC];
            const float inv = 1.0f / (sum + sk_eps);
            for (int dst = 0; dst < N_HC; ++dst) c[src + dst * N_HC] *= inv;
        }
    }
    ref_comb = c;

    std::vector<float> ref_working(N_EMBD);
    for (int d = 0; d < N_EMBD; ++d) {
        float acc = 0.0f;
        for (int h = 0; h < N_HC; ++h) acc += ref_pre[h] * hc_state[(size_t)h * N_EMBD + d];
        ref_working[d] = acc;
    }

    // ---- compare ----
    float max_abs = 0.0f, max_rel = 0.0f;
    auto acc_diff = [&](const std::vector<float> & got, const std::vector<float> & ref, const char * tag) {
        float ma = 0.0f;
        for (size_t i = 0; i < ref.size(); ++i) {
            const float ad = std::fabs(got[i] - ref[i]);
            ma = std::fmax(ma, ad);
            max_abs = std::fmax(max_abs, ad);
            max_rel = std::fmax(max_rel, ad / (std::fabs(ref[i]) + 1e-6f));
        }
        std::printf("[ds4-hc-test]   %-8s max_abs=%.3e\n", tag, ma);
    };
    acc_diff(working, ref_working, "working");
    acc_diff(post,    ref_post,    "post");
    acc_diff(comb,    ref_comb,    "comb");

    std::printf("[ds4-hc-test] max_abs_diff=%.3e max_rel_diff=%.3e\n", max_abs, max_rel);

    // fp32 on both sides; slack is reduction ordering over hc_dim=2048 plus the
    // fn half round-trip. Measured margin on gfx1151 wave32 is ~1.5e-7; 1e-4
    // matches the sibling rms_norm_hip test and leaves headroom for wave64
    // reduction ordering.
    const float TOL = 1e-4f;
    if (max_abs > TOL) {
        std::fprintf(stderr, "[ds4-hc-test] FAIL: max_abs_diff %.3e exceeds tol %.3e\n", max_abs, TOL);
        return 1;
    }
    std::printf("[ds4-hc-test] PASS\n");
    return 0;
}
