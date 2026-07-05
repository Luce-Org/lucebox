// CPU-parity test for the HIP per-token RMSNorm kernel (rms_norm_hip.cu).
//
// rms_norm_mul_w_f32_kernel reduces sum-of-squares across one wavefront, then
// across per-wavefront partials. That reduction was hardcoded to a 32-lane
// wavefront; AMD-2 generalized it to __AMDGCN_WAVEFRONT_SIZE so wave64 archs
// reduce all lanes. This test compares the GPU kernel against a CPU reference
// on the device's actual wavefront width, so it validates whichever mode the
// runner's card uses (gfx1100 / gfx1151 are wave32; a wave64 card exercises the
// generalized path). Written in CUDA spellings; the hip_compat/ shim maps them
// onto HIP exactly as the kernel and the flashprefill test do.
//
// Pass criterion: max |gpu - cpu| within a tight fp32 tolerance (the kernel
// accumulates in fp32, so the only slack is reduction ordering).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>

extern "C" void launch_rms_norm_mul_w_f32(
    const float * src, const float * w, float * dst,
    int n_tokens, int hidden, float eps,
    cudaStream_t stream);

#define CK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while (0)

int main() {
    constexpr int N_TOK  = 64;
    constexpr int HIDDEN = 2048;   // > block(256): forces the strided load loop
    constexpr float EPS  = 1e-5f;

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("[rmsnorm-test] device=%s warpSize=%d hidden=%d n_tok=%d\n",
                prop.name, prop.warpSize, HIDDEN, N_TOK);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> wdist(0.5f, 1.5f);

    std::vector<float> src((size_t)N_TOK * HIDDEN);
    std::vector<float> w(HIDDEN);
    for (auto & v : src) v = xdist(rng);
    for (auto & v : w)   v = wdist(rng);

    float *d_src, *d_w, *d_dst;
    CK(cudaMalloc(&d_src, src.size() * sizeof(float)));
    CK(cudaMalloc(&d_w,   w.size()   * sizeof(float)));
    CK(cudaMalloc(&d_dst, src.size() * sizeof(float)));
    CK(cudaMemcpy(d_src, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w,   w.data(),   w.size()   * sizeof(float), cudaMemcpyHostToDevice));

    launch_rms_norm_mul_w_f32(d_src, d_w, d_dst, N_TOK, HIDDEN, EPS, 0);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());

    std::vector<float> gpu(src.size());
    CK(cudaMemcpy(gpu.data(), d_dst, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference: inv = rsqrt(mean(x^2) + eps); out = x * inv * w.
    float max_abs = 0.0f, max_rel = 0.0f;
    for (int t = 0; t < N_TOK; ++t) {
        const float * row = src.data() + (size_t)t * HIDDEN;
        double sumsq = 0.0;
        for (int i = 0; i < HIDDEN; ++i) sumsq += (double)row[i] * row[i];
        const float inv = 1.0f / std::sqrt((float)(sumsq / HIDDEN) + EPS);
        for (int i = 0; i < HIDDEN; ++i) {
            const float ref = row[i] * inv * w[i];
            const float got = gpu[(size_t)t * HIDDEN + i];
            const float ad  = std::fabs(got - ref);
            max_abs = std::fmax(max_abs, ad);
            max_rel = std::fmax(max_rel, ad / (std::fabs(ref) + 1e-6f));
        }
    }

    cudaFree(d_src); cudaFree(d_w); cudaFree(d_dst);

    std::printf("[rmsnorm-test] max_abs_diff=%.3e max_rel_diff=%.3e\n", max_abs, max_rel);

    // fp32 accumulation on both sides; allow only reduction-ordering slack.
    const float TOL = 1e-4f;
    if (max_abs > TOL) {
        std::fprintf(stderr, "[rmsnorm-test] FAIL: max_abs_diff %.3e exceeds tol %.3e\n", max_abs, TOL);
        return 1;
    }
    std::printf("[rmsnorm-test] PASS\n");
    return 0;
}
