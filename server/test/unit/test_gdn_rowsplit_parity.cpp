// Parity test for the row-split GDN prefill recurrence (gated_delta_net.cu,
// gated_delta_net_rowsplit_cuda) against the legacy tiled kernel it replaces
// as the T>1 default on RDNA3.5.
//
// Claim under test: for the qwen4exp S_v=128 chain-mode call (contiguous
// q/k [128, Hk, T], v a strided view into the conv output, scalar g/beta per
// token and head), both kernels produce the same attention output and final
// state to within float32 reassociation noise, for token counts that cover
// a single token, a partial trailing window, and full prefill chunks.
//
// The launcher is declared with hipStream_t (the HIP build maps cudaStream_t
// to it, ggml/src/ggml-cuda/vendors/hip.h), same as test_mmid_bounded_parity.

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

void ggml_cuda_gdn_prefill_kernel_launch(
        bool legacy,
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * state_in, float * dst, float * state_out,
        int H, int Hk, int n_tokens,
        int64_t sq1, int64_t sq2, int64_t sv1, int64_t sv2, int64_t sb1, int64_t sb2,
        float scale, hipStream_t stream);

namespace {

constexpr int D  = 128;
constexpr int H  = 48;
constexpr int Hk = 16;
constexpr int CHANNELS = (2 * Hk + H) * D;  // conv output row: q | k | v
constexpr double OUT_TOL   = 5.0e-5;        // |a-b| / max(|a|, 1e-3)
constexpr double STATE_TOL = 2.0e-4;

struct Dev {
    float * p = nullptr;
    explicit Dev(size_t n) { if (hipMalloc(&p, n * sizeof(float)) != hipSuccess) p = nullptr; }
    ~Dev() { if (p) hipFree(p); }
};

// Max |a-b| / max(|a|, 1e-3); any non-finite element returns NaN (fmax
// would silently drop it).
double max_rel(const std::vector<float> & a, const std::vector<float> & b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return NAN;
        const double den = std::fmax(std::fabs((double) a[i]), 1e-3);
        worst = std::fmax(worst, std::fabs((double) a[i] - b[i]) / den);
    }
    return worst;
}

bool run_case(int T) {
    std::mt19937 rng(1234 + T);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);

    // q/k unit-normalised per (token, head) as rms_norm * 1/sqrt(D) produces.
    std::vector<float> conv((size_t) T * CHANNELS), q((size_t) T * Hk * D), k(q.size());
    for (float & x : conv) x = nd(rng);
    for (int t = 0; t < T; ++t) {
        for (int kh = 0; kh < 2 * Hk; ++kh) {
            float * row = conv.data() + (size_t) t * CHANNELS + kh * D;
            double ss = 0.0;
            for (int i = 0; i < D; ++i) ss += (double) row[i] * row[i];
            const float inv = 1.0f / std::sqrt((float) ss);
            for (int i = 0; i < D; ++i) row[i] *= inv;
        }
        std::memcpy(q.data() + (size_t) t * Hk * D, conv.data() + (size_t) t * CHANNELS, Hk * D * sizeof(float));
        std::memcpy(k.data() + (size_t) t * Hk * D, conv.data() + (size_t) t * CHANNELS + Hk * D, Hk * D * sizeof(float));
    }
    std::vector<float> g((size_t) T * H), beta(g.size()), state0((size_t) H * D * D);
    for (size_t i = 0; i < g.size(); ++i) {
        // log-decay: mostly mild, with occasional near-total forgetting (the
        // row-split kernel's scaled-state fold) and total forgetting, where
        // expf(g) underflows to exactly 0 as it does for real qwen4exp gates.
        const float r = ud(rng);
        g[i] = r < 0.01f ? -200.0f : r < 0.03f ? -30.0f : -0.3f * ud(rng);
        beta[i] = 1.0f / (1.0f + std::exp(-nd(rng)));
    }
    for (float & x : state0) x = 0.1f * nd(rng);

    const size_t out_elems = (size_t) T * H * D;
    Dev d_conv(conv.size()), d_q(q.size()), d_k(k.size()), d_g(g.size()), d_beta(beta.size());
    Dev d_state0(state0.size()), d_state(state0.size()), d_out(out_elems);
    if (!d_conv.p || !d_q.p || !d_k.p || !d_g.p || !d_beta.p || !d_state0.p || !d_state.p || !d_out.p) {
        std::printf("  alloc failed\n");
        return false;
    }
    hipMemcpy(d_conv.p, conv.data(), conv.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_q.p, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_k.p, k.data(), k.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_g.p, g.data(), g.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_beta.p, beta.data(), beta.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_state0.p, state0.data(), state0.size() * sizeof(float), hipMemcpyHostToDevice);

    const float scale = 1.0f / std::sqrt((float) D);
    std::vector<float> out[2], st[2];
    for (int arm = 0; arm < 2; ++arm) {
        hipMemset(d_out.p, 0, out_elems * sizeof(float));
        hipMemset(d_state.p, 0, state0.size() * sizeof(float));
        ggml_cuda_gdn_prefill_kernel_launch(
            /*legacy=*/arm == 0, d_q.p, d_k.p, d_conv.p + 2 * Hk * D, d_g.p, d_beta.p,
            d_state0.p, d_out.p, d_state.p, H, Hk, T,
            /*sq1=*/D, /*sq2=*/(int64_t) Hk * D, /*sv1=*/D, /*sv2=*/CHANNELS, /*sb1=*/1, /*sb2=*/H,
            scale, (hipStream_t) 0);
        if (hipDeviceSynchronize() != hipSuccess) {
            std::printf("  arm %d launch failed\n", arm);
            return false;
        }
        out[arm].resize(out_elems);
        st[arm].resize(state0.size());
        hipMemcpy(out[arm].data(), d_out.p, out_elems * sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(st[arm].data(), d_state.p, state0.size() * sizeof(float), hipMemcpyDeviceToHost);
    }
    const double out_err = max_rel(out[0], out[1]);
    const double st_err  = max_rel(st[0], st[1]);
    const bool ok = out_err <= OUT_TOL && st_err <= STATE_TOL && std::isfinite(out_err) && std::isfinite(st_err);
    std::printf("[gdn-rowsplit-parity] T=%d out max_rel %.3g (tol %.1g) state max_rel %.3g (tol %.1g) %s\n",
                T, out_err, OUT_TOL, st_err, STATE_TOL, ok ? "ok" : "FAIL");
    return ok;
}

}  // namespace

int main() {
#if defined(GGML_USE_HIP)
    int n_devices = 0;
    if (hipGetDeviceCount(&n_devices) != hipSuccess || n_devices == 0) {
        std::printf("[gdn-rowsplit-parity] SKIP: no HIP device\n");
        return 77;
    }
    bool ok = true;
    for (int T : { 1, 5, 33, 512, 4096 }) {
        ok = run_case(T) && ok;
    }
    std::printf("[gdn-rowsplit-parity] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
#else
    std::printf("[gdn-rowsplit-parity] SKIP: HIP-only\n");
    return 77;
#endif
}
