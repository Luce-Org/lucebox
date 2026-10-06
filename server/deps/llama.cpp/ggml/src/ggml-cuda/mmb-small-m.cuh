#pragma once
// Tiny-M dense GEMM for prefill: out[t][m] = sum_k w[m][k] * x[t][k], M <= 8 (qwen4exp HC inject, 4 x 10240).
// Bandwidth-bound: one wave per token streams its activation row once (F32 or bf16), the M weight rows
// (F32 or bf16, L2-resident) are read alongside, and each lane keeps F32 FMA partials that a fixed xor-tree
// reduces (deterministic). Replaces a 128-row WMMA tile that wasted 124/128 rows and launched T/64 blocks.

#include <cstdint>

template <bool BF16> __device__ __forceinline__ void msm_load4(const void * p, const size_t i, float v[4]) {
    if constexpr (BF16) {
        const uint2 u = *(const uint2 *) ((const uint16_t *) p + i);
        v[0] = __uint_as_float(u.x << 16); v[1] = __uint_as_float(u.x & 0xffff0000u);
        v[2] = __uint_as_float(u.y << 16); v[3] = __uint_as_float(u.y & 0xffff0000u);
    } else {
        const float4 f = *(const float4 *) ((const float *) p + i);
        v[0] = f.x; v[1] = f.y; v[2] = f.z; v[3] = f.w;
    }
}

// TPW tokens per wave share each weight load (F32 weights are otherwise re-read from L2 once per token).
template <int MM, bool WBF16, bool XBF16, int TPW = 2>
__global__ void __launch_bounds__(256) mmb_small_m_kernel(const void * __restrict__ w, const void * __restrict__ x,
        float * __restrict__ d, const int T, const int M, const int K) {
    const int lane = threadIdx.x & 31, t0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * TPW;
    if (t0 >= T) return;
    float acc[TPW][MM];
#pragma unroll
    for (int j = 0; j < TPW; ++j)
#pragma unroll
        for (int m = 0; m < MM; ++m) acc[j][m] = 0.0f;
    // Four independent 128-element strips per iteration keep enough loads in flight (one strip ran at 33 GB/s).
    for (int k0 = lane * 4; k0 < K; k0 += 512) {
        float xv[TPW][4][4];
#pragma unroll
        for (int j = 0; j < TPW; ++j)
#pragma unroll
            for (int u = 0; u < 4; ++u) msm_load4<XBF16>(x, (size_t) min(t0 + j, T - 1) * K + k0 + u * 128, xv[j][u]);
#pragma unroll
        for (int m = 0; m < MM; ++m) {
            if (m < M) {
                float wv[4][4];
#pragma unroll
                for (int u = 0; u < 4; ++u) msm_load4<WBF16>(w, (size_t) m * K + k0 + u * 128, wv[u]);
#pragma unroll
                for (int j = 0; j < TPW; ++j)
#pragma unroll
                    for (int u = 0; u < 4; ++u)
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc[j][m] = fmaf(wv[u][e], xv[j][u][e], acc[j][m]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TPW; ++j)
#pragma unroll
        for (int m = 0; m < MM; ++m)
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc[j][m] += __shfl_xor(acc[j][m], o, 32);
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < TPW; ++j)
#pragma unroll
            for (int m = 0; m < MM; ++m) if (m < M && t0 + j < T) d[(size_t) (t0 + j) * M + m] = acc[j][m];
    }
}

template <int MM>
static void mmb_small_m_dispatch(const void * w, const bool wbf16, const void * x, const bool xbf16, float * d,
        const int T, const int M, const int K, hipStream_t stream) {
    const unsigned grid = (unsigned) ((T + 15) / 16);   // 8 waves x 2 tokens
    if (wbf16 && xbf16)  mmb_small_m_kernel<MM, true,  true ><<<grid, 256, 0, stream>>>(w, x, d, T, M, K);
    if (wbf16 && !xbf16) mmb_small_m_kernel<MM, true,  false><<<grid, 256, 0, stream>>>(w, x, d, T, M, K);
    if (!wbf16 && xbf16) mmb_small_m_kernel<MM, false, true ><<<grid, 256, 0, stream>>>(w, x, d, T, M, K);
    if (!wbf16 && !xbf16) mmb_small_m_kernel<MM, false, false><<<grid, 256, 0, stream>>>(w, x, d, T, M, K);
}

// w: F32 or bf16 [M][K]; x: F32 or bf16 [T][K]; K % 512 == 0, M <= 8. Returns false for other shapes.
static bool mmb_small_m_launch(const void * w, const bool wbf16, const void * x, const bool xbf16, float * d,
        const int T, const int M, const int K, hipStream_t stream) {
    if (M < 1 || M > 8 || K % 512 != 0 || T < 1) return false;
    if (M <= 4) mmb_small_m_dispatch<4>(w, wbf16, x, xbf16, d, T, M, K, stream);
    else        mmb_small_m_dispatch<8>(w, wbf16, x, xbf16, d, T, M, K, stream);
    return true;
}
