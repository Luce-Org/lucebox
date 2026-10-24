// Parity test for the GDN q/k L2 norm folded into the gfx1151 conv epilogue
// (gdn-conv.cu, gdn_conv_direct_norm_kernel) against the two-kernel chain it
// replaces: gdn_conv_direct_kernel (SiLU(conv) over all channels) followed by
// the fused RMS_NORM -> SCALE kernel rms_norm_scale_f32 on the q and k
// channels (LUCE_GDN_NORM_LEGACY=1 restores the chain).
//
// Claim under test: for the qwen4exp layout (C = 10240 channels = q 2048 |
// k 2048 | v 6144, 4-tap causal conv with a [3, C] history, D = 128) the
// fused launch writes bit-identical q/k channels (compared as floats, so
// +0/-0 agree) and bit-identical v channels, for token counts that cover a
// full tile, a ragged sub-tile and a prefill chunk.
//
// The launchers are declared with hipStream_t (the HIP build maps cudaStream_t
// to it), same as test_gdn_rowsplit_parity.

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern "C" int ggml_cuda_test_gdn_conv_direct(int norm, const float * saved_state, const float * x,
        const float * w, float * y, int C, int T, int state_stride, int qk_channels, float eps, float post_scale,
        float bias, void * stream);
extern "C" int ggml_cuda_test_rms_norm_scale(const float * x, float * dst, int ncols, int nrows,
        int nchannels, int nsamples, int64_t stride_row, int64_t stride_channel, int64_t stride_sample, float eps,
        float post_scale, float bias, void * stream);

namespace {

constexpr int D  = 128;
constexpr int Hk = 16;
constexpr int Hv = 48;
constexpr int QK = 2 * Hk * D;          // 4096 q|k channels
constexpr int C  = QK + Hv * D;         // 10240

struct Dev {
    float * p = nullptr;
    explicit Dev(size_t n) { if (hipMalloc(&p, n * sizeof(float)) != hipSuccess) p = nullptr; }
    ~Dev() { if (p) hipFree(p); }
};

bool run_case(int T) {
    std::mt19937 rng(4242 + T);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int state_stride = T + 3;     // the concat row pitch the fusion passes
    std::vector<float> state((size_t) C * state_stride), x((size_t) C * T), w((size_t) C * 4);
    for (float & v : state) v = nd(rng);
    for (float & v : x) v = nd(rng);
    for (float & v : w) v = 0.5f * nd(rng);
    const float eps = 1e-6f / (float) D, post_scale = 1.0f / std::sqrt((float) D), bias = 0.0f;

    Dev dstate(state.size()), dx(x.size()), dw(w.size()), dy((size_t) C * T), dref((size_t) QK * T);
    if (!dstate.p || !dx.p || !dw.p || !dy.p || !dref.p) { std::printf("  alloc failed\n"); return false; }
    (void) hipMemcpy(dstate.p, state.data(), state.size() * 4, hipMemcpyHostToDevice);
    (void) hipMemcpy(dx.p, x.data(), x.size() * 4, hipMemcpyHostToDevice);
    (void) hipMemcpy(dw.p, w.data(), w.size() * 4, hipMemcpyHostToDevice);

    // Reference: plain conv, then rms_norm_scale over the q/k heads ([128, 2*Hk, T] view, row pitch C).
    (void) hipMemset(dy.p, 0, (size_t) C * T * 4);
    if (!ggml_cuda_test_gdn_conv_direct(0, dstate.p, dx.p, dw.p, dy.p, C, T, state_stride, QK, eps, post_scale, bias, (hipStream_t) 0) ||
        !ggml_cuda_test_rms_norm_scale(dy.p, dref.p, D, 2 * Hk, T, 1, D, C, (int64_t) C * T, eps, post_scale, bias, (hipStream_t) 0) ||
        hipDeviceSynchronize() != hipSuccess) {
        std::printf("  reference launch failed\n");
        return false;
    }
    std::vector<float> plain((size_t) C * T), ref((size_t) QK * T), fused((size_t) C * T);
    (void) hipMemcpy(plain.data(), dy.p, plain.size() * 4, hipMemcpyDeviceToHost);
    (void) hipMemcpy(ref.data(), dref.p, ref.size() * 4, hipMemcpyDeviceToHost);

    (void) hipMemset(dy.p, 0, (size_t) C * T * 4);
    if (!ggml_cuda_test_gdn_conv_direct(1, dstate.p, dx.p, dw.p, dy.p, C, T, state_stride, QK, eps, post_scale, bias, (hipStream_t) 0) ||
        hipDeviceSynchronize() != hipSuccess) {
        std::printf("  fused launch failed\n");
        return false;
    }
    (void) hipMemcpy(fused.data(), dy.p, fused.size() * 4, hipMemcpyDeviceToHost);

    size_t qk_mis = 0, v_mis = 0, nonfinite = 0;
    double worst = 0.0;
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < C; ++c) {
            const float f = fused[(size_t) t * C + c];
            if (!std::isfinite(f)) ++nonfinite;
            if (c < QK) {
                const float r = ref[(size_t) t * QK + c];
                if (!(f == r)) {
                    if (qk_mis < 4) {
                        uint32_t uf, ur; std::memcpy(&uf, &f, 4); std::memcpy(&ur, &r, 4);
                        std::printf("    mismatch t=%d c=%d: fused %.9g (0x%08x) ref %.9g (0x%08x) plain-conv %.9g\n", t, c, f, uf, r, ur, plain[(size_t) t * C + c]);
                    }
                    ++qk_mis; worst = std::fmax(worst, std::fabs((double) f - r));
                }
            } else if (!(f == plain[(size_t) t * C + c])) {
                ++v_mis;
            }
        }
    }
    // The reference must be a real unit-norm row (rms_norm * 1/sqrt(D) -> ||row|| == 1).
    double nrm = 0.0;
    for (int i = 0; i < D; ++i) nrm += (double) ref[i] * ref[i];
    const bool unit = std::fabs(nrm - 1.0) < 1e-3;
    const bool ok = qk_mis == 0 && v_mis == 0 && nonfinite == 0 && unit;
    std::printf("[gdn-conv-qknorm-parity] T=%d: q/k mismatches %zu (worst |d| %.3g), v mismatches %zu, non-finite %zu, ref ||row0||^2 %.6f %s\n",
                T, qk_mis, worst, v_mis, nonfinite, nrm, ok ? "ok" : "FAIL");
    return ok;
}

}  // namespace

int main() {
#if defined(GGML_USE_HIP)
    int n_devices = 0;
    if (hipGetDeviceCount(&n_devices) != hipSuccess || n_devices == 0) {
        std::printf("[gdn-conv-qknorm-parity] SKIP: no HIP device\n");
        return 77;
    }
    bool ok = true;
    for (int T : { 256, 333, 4096 }) {
        ok = run_case(T) && ok;
    }
    std::printf("[gdn-conv-qknorm-parity] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
#else
    std::printf("[gdn-conv-qknorm-parity] SKIP: HIP-only\n");
    return 77;
#endif
}
