// rocWMMA fragment layout probe (RDNA4, gfx1201).
//
// Isolates which rocWMMA operation diverges from the CUDA WMMA semantics the
// fattn-wmma kernel assumes: matrix_a row-major loads, matrix_b col-major
// loads from a row-major Q buffer (the transposed-Q trick), mma_sync with
// half and float accumulators, and col-major accumulator stores.
#include <rocwmma/rocwmma.hpp>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace wmma = rocwmma;

using half_t = _Float16;

#define HIP_CHECK(call)                                                       \
    do {                                                                      \
        const hipError_t _e = (call);                                         \
        if (_e != hipSuccess) {                                               \
            std::fprintf(stderr, "HIP error %d at %s:%d\n", (int) _e,        \
                         __FILE__, __LINE__);                                 \
            return 1;                                                         \
        }                                                                     \
    } while (0)

template <int LDM_A, int LDM_B, int LDM_C>
__global__ void probe_kernel(
        const half_t * __restrict__ A,   // 16x16 row-major (m x k), row stride LDM_A
        const half_t * __restrict__ B,   // 16x16 stored n-major with k contiguous (Q-style), n stride LDM_B
        half_t       * __restrict__ C,   // 16x16 col-major (k x n) per CUDA store contract, n stride LDM_C
        float        * __restrict__ C_f) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half_t, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half_t, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half_t> c_h;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;

    wmma::load_matrix_sync(a, A, LDM_A);
    wmma::load_matrix_sync(b, B, LDM_B);
    wmma::fill_fragment(c_h, (half_t) 0.0f);
    wmma::fill_fragment(c_f, 0.0f);
    wmma::mma_sync(c_h, a, b, c_h);
    wmma::mma_sync(c_f, a, b, c_f);

    // A is m x k and B is n x k stored as col-major k x n, so the product is
    // A * B^T. The host computes the same.
    wmma::store_matrix_sync(C, c_h, LDM_C, wmma::mem_col_major);
    wmma::store_matrix_sync(C_f, c_f, LDM_C, wmma::mem_col_major);
}

// A loaded as matrix_a col_major: element (m, k) at m + k*LDM_A. Mirrors the
// kernel's VKQ path (frag_a_V), where the fragment holds V^T from a V buffer
// addressed as V[s=k][d=m] at k*stride + m.
// Multi-warp chained f16 accumulation: 4 warps (blockDim.y = 4) each run the
// same chain into per-warp outputs, mirroring the fattn-wmma block geometry.
__global__ void probe_kernel_chain_multi(
        const half_t * __restrict__ A,   // 16 blocks of 16x16 row-major A tiles (m x k), ldm 16
        const half_t * __restrict__ B,   // 16 blocks of 16x16 Q-style B tiles, n stride 16
        half_t       * __restrict__ C_h,
        float        * __restrict__ C_f) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half_t, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half_t, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half_t> c_h;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;

    wmma::fill_fragment(c_h, (half_t) 0.0f);
    wmma::fill_fragment(c_f, 0.0f);
#pragma unroll
    for (int t = 0; t < 16; ++t) {
        wmma::load_matrix_sync(a, A + t * 256, 16);
        wmma::load_matrix_sync(b, B + t * 256, 16);
        wmma::mma_sync(c_h, a, b, c_h);
        wmma::mma_sync(c_f, a, b, c_f);
    }
    wmma::store_matrix_sync(C_h + threadIdx.y * 256, c_h, 16, wmma::mem_col_major);
    wmma::store_matrix_sync(C_f + threadIdx.y * 256, c_f, 16, wmma::mem_col_major);
}

// Chained f16 accumulation: the fattn-wmma kernel runs 16 sequential
// mma_sync into the same half accumulator for D=256. Compares the f16-acc
// chain against an f32-acc chain of identical structure.
__global__ void probe_kernel_chain(
        const half_t * __restrict__ A,   // 16 blocks of 16x16 row-major A tiles (m x k), ldm 16
        const half_t * __restrict__ B,   // 16 blocks of 16x16 Q-style B tiles, n stride 16
        half_t       * __restrict__ C_h,
        float        * __restrict__ C_f) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half_t, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half_t, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half_t> c_h;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;

    wmma::fill_fragment(c_h, (half_t) 0.0f);
    wmma::fill_fragment(c_f, 0.0f);
#pragma unroll
    for (int t = 0; t < 16; ++t) {
        wmma::load_matrix_sync(a, A + t * 256, 16);
        wmma::load_matrix_sync(b, B + t * 256, 16);
        wmma::mma_sync(c_h, a, b, c_h);
        wmma::mma_sync(c_f, a, b, c_f);
    }
    wmma::store_matrix_sync(C_h, c_h, 16, wmma::mem_col_major);
    wmma::store_matrix_sync(C_f, c_f, 16, wmma::mem_col_major);
}

template <int LDM_A, int LDM_C>
__global__ void probe_kernel_acol(
        const half_t * __restrict__ A,   // 16x16, col-major fragment source, (m,k) at m + k*LDM_A
        const half_t * __restrict__ B,   // 16x16 Q-style (k contiguous), n stride 16
        half_t       * __restrict__ C,
        float        * __restrict__ C_f) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half_t, wmma::col_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half_t, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half_t> c_h;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_f;

    wmma::load_matrix_sync(a, A, LDM_A);
    wmma::load_matrix_sync(b, B, 16);
    wmma::fill_fragment(c_h, (half_t) 0.0f);
    wmma::fill_fragment(c_f, 0.0f);
    wmma::mma_sync(c_h, a, b, c_h);
    wmma::mma_sync(c_f, a, b, c_f);
    wmma::store_matrix_sync(C, c_h, LDM_C, wmma::mem_col_major);
    wmma::store_matrix_sync(C_f, c_f, LDM_C, wmma::mem_col_major);
}

static bool probe_h2exp();

int main() {
#if defined(GGML_USE_HIP)
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, 0) != hipSuccess ||
        std::strncmp(props.gcnArchName, "gfx12", 5) != 0) {
        std::printf("[rocwmma-probe] SKIP: requires gfx12 (RDNA4)\n");
        return 77;
    }

    constexpr int N = 16;
    std::vector<float> A(N * N), Bmem(N * N);
    for (int i = 0; i < N * N; ++i) {
        A[i]    = (float) ((i * 7 + 3) % 17 - 8) / 8.0f;
        Bmem[i] = (float) ((i * 5 + 1) % 13 - 6) / 6.0f;
    }

    // Bmem is the Q-style buffer: element (i, j) at [j*N + i] (k contiguous).
    // Host reference: C[m][n] = sum_k A[m][k] * Bmem[n*N + k]  (A * B^T).
    std::vector<float> C_ref(N * N);
    for (int m = 0; m < N; ++m) {
        for (int n = 0; n < N; ++n) {
            float s = 0.0f;
            for (int k = 0; k < N; ++k) {
                s += A[m * N + k] * Bmem[n * N + k];
            }
            C_ref[m * N + n] = s;
        }
    }

    struct Variant { const char * name; int ldm_a; int ldm_b; int ldm_c; };
    const Variant variants[] = {
        { "ldm16",     16,  16,  16 },
        { "padded264", 256, 264, 264 },
    };

    bool all_pass = true;

    // Chained f16 accumulation probe.
    {
        std::vector<float> At(16 * N * N), Bt(16 * N * N);
        for (int t = 0; t < 16; ++t) {
            for (int i = 0; i < N * N; ++i) {
                At[t * N * N + i] = (float) ((i * 7 + t * 11 + 3) % 17 - 8) / 8.0f;
                Bt[t * N * N + i] = (float) ((i * 5 + t * 3 + 1) % 13 - 6) / 6.0f;
            }
        }
        // Reference: C[m][n] = sum_t sum_k A_t[m][k] * Bt_t[n][k] (A_t * B_t^T).
        std::vector<float> C_chain_ref(N * N, 0.0f);
        for (int t = 0; t < 16; ++t)
            for (int m = 0; m < N; ++m)
                for (int n = 0; n < N; ++n)
                    for (int k = 0; k < N; ++k)
                        C_chain_ref[m * N + n] += At[t * N * N + m * N + k] * Bt[t * N * N + n * N + k];

        std::vector<half_t> Ah(At.size()), Bh(Bt.size());
        for (size_t i = 0; i < At.size(); ++i) { Ah[i] = (half_t) At[i]; Bh[i] = (half_t) Bt[i]; }

        half_t *d_A, *d_B, *d_Ch;
        float  *d_Cf;
        HIP_CHECK(hipMalloc(&d_A, Ah.size() * sizeof(half_t)));
        HIP_CHECK(hipMalloc(&d_B, Bh.size() * sizeof(half_t)));
        HIP_CHECK(hipMalloc(&d_Ch, N * N * sizeof(half_t)));
        HIP_CHECK(hipMalloc(&d_Cf, N * N * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_A, Ah.data(), Ah.size() * sizeof(half_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, Bh.data(), Bh.size() * sizeof(half_t), hipMemcpyHostToDevice));

        probe_kernel_chain<<<1, 32>>>(d_A, d_B, d_Ch, d_Cf);
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<half_t> Ch(N * N);
        std::vector<float> Cf(N * N);
        HIP_CHECK(hipMemcpy(Ch.data(), d_Ch, N * N * sizeof(half_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(Cf.data(), d_Cf, N * N * sizeof(float), hipMemcpyDeviceToHost));

        float max_diff_h = 0.0f, max_diff_f = 0.0f;
        int worst_m = -1, worst_n = -1;
        for (int n = 0; n < N; ++n) {
            for (int m = 0; m < N; ++m) {
                const float ref = C_chain_ref[m * N + n];
                const float dh = std::fabs((float) Ch[m + n * N] - ref);
                const float df = std::fabs(Cf[m + n * N] - ref);
                if (dh > max_diff_h) { max_diff_h = dh; worst_m = m; worst_n = n; }
                if (df > max_diff_f) max_diff_f = df;
            }
        }
        const bool pass = max_diff_f < 1e-2f && max_diff_h < 5e-2f;
        all_pass = all_pass && pass;
        std::printf("[rocwmma-probe] chained_f16acc max_diff_f32acc=%0.6f max_diff_f16acc=%0.6f worst=(m=%d,n=%d) %s\n",
                    max_diff_f, max_diff_h, worst_m, worst_n, pass ? "PASS" : "FAIL");

        // Multi-warp variant: 4 warps write separate outputs.
        half_t *d_Ch4; float *d_Cf4;
        HIP_CHECK(hipMalloc(&d_Ch4, 4 * N * N * sizeof(half_t)));
        HIP_CHECK(hipMalloc(&d_Cf4, 4 * N * N * sizeof(float)));
        probe_kernel_chain_multi<<<1, dim3(32, 4)>>>(d_A, d_B, d_Ch4, d_Cf4);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<half_t> Ch4(4 * N * N);
        std::vector<float> Cf4(4 * N * N);
        HIP_CHECK(hipMemcpy(Ch4.data(), d_Ch4, 4 * N * N * sizeof(half_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(Cf4.data(), d_Cf4, 4 * N * N * sizeof(float), hipMemcpyDeviceToHost));
        float max_diff_mh = 0.0f, max_diff_mf = 0.0f;
        for (int w = 0; w < 4; ++w) {
            for (int n = 0; n < N; ++n) {
                for (int m = 0; m < N; ++m) {
                    const float ref = C_chain_ref[m * N + n];
                    max_diff_mh = std::max(max_diff_mh, std::fabs((float) Ch4[(size_t) w * 256 + m + n * 16] - ref));
                    max_diff_mf = std::max(max_diff_mf, std::fabs(Cf4[(size_t) w * 256 + m + n * 16] - ref));
                }
            }
        }
        const bool mpass = max_diff_mf < 1e-2f && max_diff_mh < 5e-2f;
        all_pass = all_pass && mpass;
        std::printf("[rocwmma-probe] chained_multi_warp max_diff_f32acc=%0.6f max_diff_f16acc=%0.6f %s\n",
                    max_diff_mf, max_diff_mh, mpass ? "PASS" : "FAIL");
        hipFree(d_A); hipFree(d_B); hipFree(d_Ch); hipFree(d_Cf); hipFree(d_Ch4); hipFree(d_Cf4);
    }

    // matrix_a col_major probe: A_frag[m][k] = V^T[m][k] = Bmem2[k][m] where
    // Bmem2 is the row-major 16x16 "V" matrix; memory (m,k) at m + k*ldm_a.
    {
        std::vector<float> Vmem(N * N);
        for (int i = 0; i < N * N; ++i) Vmem[i] = (float) ((i * 3 + 5) % 11 - 5) / 5.0f;
        // Host reference: C[m][n] = sum_k A_frag[m][k] * B[k][n]
        //   A_frag[m][k] = Vmem[k*N + m]; B[k][n] = Bmem[n*N + k] (Q-style).
        std::vector<float> C_acol_ref(N * N);
        for (int m = 0; m < N; ++m)
            for (int n = 0; n < N; ++n) {
                float s = 0.0f;
                for (int k = 0; k < N; ++k) s += Vmem[k * N + m] * Bmem[n * N + k];
                C_acol_ref[m * N + n] = s;
            }

        const int ldm_a = 16;
        std::vector<float> Apad((size_t) ldm_a * N);
        for (int m = 0; m < N; ++m)
            for (int k = 0; k < N; ++k) Apad[(size_t) m + (size_t) k * ldm_a] = Vmem[k * N + m];
        std::vector<half_t> Ah(Apad.size()), Bh(N * N);
        for (size_t i = 0; i < Apad.size(); ++i) Ah[i] = (half_t) Apad[i];
        for (int i = 0; i < N * N; ++i) Bh[i] = (half_t) Bmem[i];

        half_t *d_A, *d_B, *d_C;
        float  *d_Cf;
        HIP_CHECK(hipMalloc(&d_A, Ah.size() * sizeof(half_t)));
        HIP_CHECK(hipMalloc(&d_B, Bh.size() * sizeof(half_t)));
        HIP_CHECK(hipMalloc(&d_C, N * N * sizeof(half_t)));
        HIP_CHECK(hipMalloc(&d_Cf, N * N * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_A, Ah.data(), Ah.size() * sizeof(half_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, Bh.data(), Bh.size() * sizeof(half_t), hipMemcpyHostToDevice));

        probe_kernel_acol<16, 16><<<1, 32>>>(d_A, d_B, d_C, d_Cf);
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<half_t> Ch(N * N);
        std::vector<float> Cf(N * N);
        HIP_CHECK(hipMemcpy(Ch.data(), d_C, N * N * sizeof(half_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(Cf.data(), d_Cf, N * N * sizeof(float), hipMemcpyDeviceToHost));

        float max_diff_h = 0.0f, max_diff_f = 0.0f;
        int worst_m = -1, worst_n = -1;
        for (int n = 0; n < N; ++n) {
            for (int m = 0; m < N; ++m) {
                const float ref = C_acol_ref[m * N + n];
                const float dh = std::fabs((float) Ch[m + n * N] - ref);
                const float df = std::fabs(Cf[m + n * N] - ref);
                if (dh > max_diff_h) { max_diff_h = dh; worst_m = m; worst_n = n; }
                if (df > max_diff_f) max_diff_f = df;
            }
        }
        const bool pass = max_diff_f < 1e-3f;
        all_pass = all_pass && pass;
        std::printf("[rocwmma-probe] a_col_major max_diff_f32acc=%0.6f max_diff_f16acc=%0.6f worst=(m=%d,n=%d) %s\n",
                    max_diff_f, max_diff_h, worst_m, worst_n, pass ? "PASS" : "FAIL");
        hipFree(d_A); hipFree(d_B); hipFree(d_C); hipFree(d_Cf);
    }

    for (const Variant & v : variants) {
        // Lay out padded buffers: A rows at stride ldm_a, B cols at stride ldm_b.
        std::vector<float> Apad((size_t) v.ldm_a * N), Bpad((size_t) v.ldm_b * N);
        for (int m = 0; m < N; ++m)
            for (int k = 0; k < N; ++k) Apad[(size_t) m * v.ldm_a + k] = A[m * N + k];
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < N; ++k) Bpad[(size_t) n * v.ldm_b + k] = Bmem[n * N + k];

        std::vector<half_t> Ah(Apad.size()), Bh(Bpad.size());
        for (size_t i = 0; i < Apad.size(); ++i) Ah[i] = (half_t) Apad[i];
        for (size_t i = 0; i < Bpad.size(); ++i) Bh[i] = (half_t) Bpad[i];

        half_t *d_A, *d_B, *d_C;
        float  *d_Cf;
        HIP_CHECK(hipMalloc(&d_A, Ah.size() * sizeof(half_t)));
        HIP_CHECK(hipMalloc(&d_B, Bh.size() * sizeof(half_t)));
        HIP_CHECK(hipMalloc(&d_C, (size_t) v.ldm_c * N * sizeof(half_t)));
        HIP_CHECK(hipMalloc(&d_Cf, (size_t) v.ldm_c * N * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_A, Ah.data(), Ah.size() * sizeof(half_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, Bh.data(), Bh.size() * sizeof(half_t), hipMemcpyHostToDevice));

        // The kernel is templated on all three leading dimensions; select
        // the instantiation from the variant's own (ldm_a, ldm_b, ldm_c)
        // triple so a future variant with mixed strides cannot silently
        // run the wrong instantiation.
        if (v.ldm_a == 256 && v.ldm_b == 264 && v.ldm_c == 264) {
            probe_kernel<256, 264, 264><<<1, 32>>>(d_A, d_B, d_C, d_Cf);
        } else if (v.ldm_a == 16 && v.ldm_b == 16 && v.ldm_c == 16) {
            probe_kernel<16, 16, 16><<<1, 32>>>(d_A, d_B, d_C, d_Cf);
        } else {
            std::printf("[rocwmma-probe] %s SKIP: no instantiation for ldm_a=%d ldm_b=%d ldm_c=%d\n",
                        v.name, v.ldm_a, v.ldm_b, v.ldm_c);
            hipFree(d_A); hipFree(d_B); hipFree(d_C); hipFree(d_Cf);
            continue;
        }
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<half_t> Ch((size_t) v.ldm_c * N);
        std::vector<float> Cf((size_t) v.ldm_c * N);
        HIP_CHECK(hipMemcpy(Ch.data(), d_C, Ch.size() * sizeof(half_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(Cf.data(), d_Cf, Cf.size() * sizeof(float), hipMemcpyDeviceToHost));

        // CUDA contract: store col_major -> element (m, n) at m + n*ldm_c.
        float max_diff_h = 0.0f, max_diff_f = 0.0f;
        int worst_m = -1, worst_n = -1;
        for (int n = 0; n < N; ++n) {
            for (int m = 0; m < N; ++m) {
                const float ref = C_ref[m * N + n];
                const float vh = (float) Ch[(size_t) m + (size_t) n * v.ldm_c];
                const float vf = Cf[(size_t) m + (size_t) n * v.ldm_c];
                const float dh = std::fabs(vh - ref);
                const float df = std::fabs(vf - ref);
                if (dh > max_diff_h) { max_diff_h = dh; worst_m = m; worst_n = n; }
                if (df > max_diff_f) max_diff_f = df;
            }
        }

        const bool pass = max_diff_f < 1e-3f;
        all_pass = all_pass && pass;
        std::printf("[rocwmma-probe] %s max_diff_f32acc=%0.6f max_diff_f16acc=%0.6f worst=(m=%d,n=%d) %s\n",
                    v.name, max_diff_f, max_diff_h, worst_m, worst_n, pass ? "PASS" : "FAIL");

        hipFree(d_A); hipFree(d_B); hipFree(d_C); hipFree(d_Cf);
    }
    if (!probe_h2exp()) all_pass = false;
    return all_pass ? 0 : 1;
#else
    std::printf("[rocwmma-probe] SKIP: HIP-only\n");
    return 77;
#endif
}

// Quick h2exp sanity: __ocml_exp_2f16 vs host exp on a value sweep.
__global__ void h2exp_probe_kernel(const float * x, half2 * out) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = h2exp(make_half2(__float2half(x[i]), __float2half(x[i] - 0.5f)));
}

static bool probe_h2exp() {
    constexpr int n = 64;
    std::vector<float> x(n);
    // The kernel only feeds h2exp diffs <= 0 (x - running_max) and the
    // logit-softcap tanh path; positive values overflow half by design.
    for (int i = 0; i < n; ++i) x[i] = -36.0f * i / (n - 1);
    float *d_x; half2 *d_out;
    HIP_CHECK(hipMalloc(&d_x, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, n * sizeof(half2)));
    HIP_CHECK(hipMemcpy(d_x, x.data(), n * sizeof(float), hipMemcpyHostToDevice));
    h2exp_probe_kernel<<<1, n>>>(d_x, d_out);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<half2> out(n);
    HIP_CHECK(hipMemcpy(out.data(), d_out, n * sizeof(half2), hipMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float ref = std::exp(x[i]);
        max_err = std::max(max_err, std::fabs(__low2float(out[i]) - ref) / std::max(ref, 1e-3f));
        max_err = std::max(max_err, std::fabs(__high2float(out[i]) - std::exp(x[i] - 0.5f)) / std::max(std::exp(x[i] - 0.5f), 1e-3f));
    }
    const bool pass = max_err < 0.05f;
    std::printf("[rocwmma-probe] h2exp max_rel_err=%0.5f %s\n", max_err, pass ? "PASS" : "FAIL");
    hipFree(d_x); hipFree(d_out);
    return pass;
}
