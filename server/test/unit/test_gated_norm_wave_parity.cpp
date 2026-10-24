// Parity test for the wave-per-row gated norm kernels (gated-norm.cu,
// gated_rms_norm_wave_kernel) against the block-per-row kernels they replace
// as the default (LUCE_GDN_NORM_LEGACY=1 restores the old family).
//
// Claim under test: for the qwen4exp GDN tail (ncols = 128, gamma per head,
// z the [ncols*nh, T] gate projection) both kernel families emit byte-identical
// F16 for every (head, token) row, including row counts that leave a partial
// block. (The 256-wide gamma == NULL gate stays on the block kernel.)
//
// The launcher is declared with hipStream_t (the HIP build maps cudaStream_t
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

extern "C" int ggml_cuda_test_gated_rms_norm_f16(int wave,
        const float * x, const float * gamma, const float * z, void * dst,
        int ncols, int nh, int T, int64_t x_s1, int64_t x_s2, int64_t z_sh, int64_t z_st, int gamma_rows, float eps,
        void * stream);

namespace {

struct Dev {
    void * p = nullptr;
    explicit Dev(size_t bytes) { if (hipMalloc(&p, bytes) != hipSuccess) p = nullptr; }
    ~Dev() { if (p) hipFree(p); }
};

// GDN tail shape: x contiguous [128, nh, T], gamma [128, gamma_rows], z [128*nh, T].
bool run_case(int nh, int T, int gamma_rows) {
    const int ncols = 128;
    std::mt19937 rng(99 + nh * 7 + T);
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t n = (size_t) ncols * nh * T;
    const int64_t z_sh = ncols, z_st = (int64_t) ncols * nh;
    std::vector<float> x(n), z((size_t) z_st * T), g((size_t) ncols * gamma_rows);
    for (float & v : x) v = 3.0f * nd(rng);
    for (float & v : z) v = 2.0f * nd(rng);
    for (float & v : g) v = 1.0f + 0.3f * nd(rng);
    x[0] = 0.0f;                       // exercise a zero column and a large-magnitude row
    for (int i = 0; i < ncols; ++i) x[n - 1 - i] = 250.0f * nd(rng);

    Dev dx(n * 4), dz(z.size() * 4), dg(g.size() * 4), dout(n * 2);
    if (!dx.p || !dz.p || !dg.p || !dout.p) { std::printf("  alloc failed\n"); return false; }
    (void) hipMemcpy(dx.p, x.data(), n * 4, hipMemcpyHostToDevice);
    (void) hipMemcpy(dz.p, z.data(), z.size() * 4, hipMemcpyHostToDevice);
    (void) hipMemcpy(dg.p, g.data(), g.size() * 4, hipMemcpyHostToDevice);

    std::vector<uint16_t> out[2];
    for (int arm = 0; arm < 2; ++arm) {
        (void) hipMemset(dout.p, 0xff, n * 2);
        if (!ggml_cuda_test_gated_rms_norm_f16(arm, (const float *) dx.p, (const float *) dg.p,
                (const float *) dz.p, dout.p, ncols, nh, T, ncols, (int64_t) ncols * nh, z_sh, z_st, gamma_rows,
                1e-6f, (hipStream_t) 0) || hipDeviceSynchronize() != hipSuccess) {
            std::printf("  arm %d launch failed\n", arm);
            return false;
        }
        out[arm].resize(n);
        (void) hipMemcpy(out[arm].data(), dout.p, n * 2, hipMemcpyDeviceToHost);
    }
    size_t mismatches = 0, first = n;
    for (size_t i = 0; i < n; ++i) {
        if (out[0][i] != out[1][i]) {
            if (first == n) first = i;
            if (mismatches < 4) {
                const size_t row = i / ncols, col = i % ncols;
                std::printf("    mismatch at row %zu (t=%zu h=%zu) col %zu: old 0x%04x new 0x%04x x=%.9g z=%.9g\n", row, row / nh, row % nh, col,
                            out[0][i], out[1][i], x[i], z[(row / nh) * (size_t) z_st + (row % nh) * (size_t) z_sh + col]);
            }
            ++mismatches;
        }
    }
    // Sanity on the reference itself: a finite, non-trivial row.
    bool finite = true;
    for (size_t i = 0; i < n; ++i) if ((out[0][i] & 0x7c00) == 0x7c00) finite = false;
    const bool ok = mismatches == 0 && finite;
    std::printf("[gated-norm-wave-parity] norm128 nh=%d T=%d gamma_rows=%d rows=%d: %zu/%zu mismatching F16 (first %zd)%s %s\n",
                nh, T, gamma_rows, nh * T, mismatches, n, first == n ? (ptrdiff_t) -1 : (ptrdiff_t) first,
                finite ? "" : " NON-FINITE", ok ? "ok" : "FAIL");
    return ok;
}

}  // namespace

int main() {
#if defined(GGML_USE_HIP)
    int n_devices = 0;
    if (hipGetDeviceCount(&n_devices) != hipSuccess || n_devices == 0) {
        std::printf("[gated-norm-wave-parity] SKIP: no HIP device\n");
        return 77;
    }
    bool ok = true;
    ok = run_case(48, 1, 1) && ok;       // decode row
    ok = run_case(48, 7, 1) && ok;       // partial block of waves
    ok = run_case(48, 333, 48) && ok;    // per-head gamma
    ok = run_case(48, 4096, 1) && ok;    // prefill chunk
    std::printf("[gated-norm-wave-parity] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
#else
    std::printf("[gated-norm-wave-parity] SKIP: HIP-only\n");
    return 77;
#endif
}
