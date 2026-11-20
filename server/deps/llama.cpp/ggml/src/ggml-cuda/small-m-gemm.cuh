#pragma once
// Fast custom HIP Small-M Dense GEMM for RDNA3/3.5 (gfx1151) and RDNA4 (gfx1201).
// Eliminates pathological rocBLAS/hipBLASLt small-M latency on BF16/F16/F32 projections
// (e.g., MoE routers [K, 128/512], SSM alpha/beta [K, 48], HC inject [K, 4]) while
// avoiding F32->BF16 activation conversion and BF16->F32 output conversion round-trips.
// Computes dst[n * M + m] = sum_{k=0..K-1} src0[m * K + k] * src1[n * K + k] in F32.

#include "common.cuh"
#include "mmb.cuh"
#include <cstdlib>

#if defined(GGML_USE_HIP)

enum small_m_wtype {
    SMALL_M_W_F32  = 0,
    SMALL_M_W_BF16 = 1,
    SMALL_M_W_F16  = 2,
};

template <int WTYPE>
__device__ __forceinline__ void small_m_load4(const void * __restrict__ p, const size_t idx, float out[4]) {
    if constexpr (WTYPE == SMALL_M_W_BF16) {
        const uint2 u = *reinterpret_cast<const uint2 *>(reinterpret_cast<const uint16_t *>(p) + idx);
        out[0] = __uint_as_float(u.x << 16);
        out[1] = __uint_as_float(u.x & 0xffff0000u);
        out[2] = __uint_as_float(u.y << 16);
        out[3] = __uint_as_float(u.y & 0xffff0000u);
    } else if constexpr (WTYPE == SMALL_M_W_F16) {
        const half2 h0 = *reinterpret_cast<const half2 *>(reinterpret_cast<const half *>(p) + idx);
        const half2 h1 = *reinterpret_cast<const half2 *>(reinterpret_cast<const half *>(p) + idx + 2);
        const float2 f0 = __half22float2(h0);
        const float2 f1 = __half22float2(h1);
        out[0] = f0.x;
        out[1] = f0.y;
        out[2] = f1.x;
        out[3] = f1.y;
    } else {
        const float4 f = *reinterpret_cast<const float4 *>(reinterpret_cast<const float *>(p) + idx);
        out[0] = f.x;
        out[1] = f.y;
        out[2] = f.z;
        out[3] = f.w;
    }
}

// Path A: Ultra-skinny M <= 8 streaming kernel (bandwidth-bound, weights L2-resident).
template <int MM, int WTYPE, int TPW, int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32) small_m_skinny_kernel(
        const void * __restrict__ w,
        const float * __restrict__ x,
        float * __restrict__ d,
        const int M, const int N, const int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n0   = (blockIdx.x * NWARPS + warp) * TPW;
    if (n0 >= N) return;

    float acc[TPW][MM];
#pragma unroll
    for (int j = 0; j < TPW; ++j) {
#pragma unroll
        for (int m = 0; m < MM; ++m) acc[j][m] = 0.0f;
    }

    // Main loop: 4 strips of 128 elements (512 elements per iter) for high MLP.
    int k0 = lane * 4;
    for (; k0 + 384 < K; k0 += 512) {
        float xv[TPW][4][4];
#pragma unroll
        for (int j = 0; j < TPW; ++j) {
            const int nj = min(n0 + j, N - 1);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                small_m_load4<SMALL_M_W_F32>(x, (size_t) nj * K + k0 + u * 128, xv[j][u]);
            }
        }
#pragma unroll
        for (int m = 0; m < MM; ++m) {
            if (m < M) {
                float wv[4][4];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    small_m_load4<WTYPE>(w, (size_t) m * K + k0 + u * 128, wv[u]);
                }
#pragma unroll
                for (int j = 0; j < TPW; ++j) {
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            acc[j][m] = fmaf(wv[u][e], xv[j][u][e], acc[j][m]);
                        }
                    }
                }
            }
        }
    }
    // Tail loop: 128 elements per iter (handles any K % 4 == 0).
    for (; k0 < K; k0 += 128) {
        float xv[TPW][4];
#pragma unroll
        for (int j = 0; j < TPW; ++j) {
            const int nj = min(n0 + j, N - 1);
            small_m_load4<SMALL_M_W_F32>(x, (size_t) nj * K + k0, xv[j]);
        }
#pragma unroll
        for (int m = 0; m < MM; ++m) {
            if (m < M) {
                float wv[4];
                small_m_load4<WTYPE>(w, (size_t) m * K + k0, wv);
#pragma unroll
                for (int j = 0; j < TPW; ++j) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        acc[j][m] = fmaf(wv[e], xv[j][e], acc[j][m]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < TPW; ++j) {
#pragma unroll
        for (int m = 0; m < MM; ++m) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                acc[j][m] += __shfl_xor(acc[j][m], o, 32);
            }
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < TPW; ++j) {
            if (n0 + j < N) {
#pragma unroll
                for (int m = 0; m < MM; ++m) {
                    if (m < M) {
                        d[(size_t) (n0 + j) * M + m] = acc[j][m];
                    }
                }
            }
        }
    }
}

// Path B1: Large-N tiled F32 GEMM (BM=32, BN=64, BK=32, 128 threads, 4x4 per thread).
template <int WTYPE>
__global__ void __launch_bounds__(128, 4) small_m_tiled_32x64_kernel(
        const void * __restrict__ w,
        const float * __restrict__ x,
        float * __restrict__ d,
        const int M, const int N, const int K) {
    constexpr int BM = 32;
    constexpr int BN = 64;
    constexpr int BK = 32;
    // Padded strides (multiples of 4 for float4 alignment, odd*4 to avoid 32-bank LDS conflicts).
    __shared__ float As[BK][BM + 4];
    __shared__ float Bs[BK][BN + 4];

    const int tid = threadIdx.x; // 0..127
    const int tx  = tid & 7;     // 0..7  -> 4 rows of M (32 total)
    const int ty  = tid >> 3;    // 0..15 -> 4 cols of N (64 total)

    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;

    float acc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][i] = 0.0f;
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load As: 32 rows x 32 cols = 256 float4s -> 2 per thread.
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int q    = tid + s * 128;
            const int row  = q >> 3;         // 0..31
            const int col4 = (q & 7) << 2;   // 0, 4, ..., 28
            const int gm   = m0 + row;
            const int gk   = k0 + col4;
            float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (gm < M && gk < K) {
                small_m_load4<WTYPE>(w, (size_t) gm * K + gk, v);
            }
            As[col4 + 0][row] = v[0];
            As[col4 + 1][row] = v[1];
            As[col4 + 2][row] = v[2];
            As[col4 + 3][row] = v[3];
        }

        // Load Bs: 64 rows x 32 cols = 512 float4s -> 4 per thread.
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int q    = tid + s * 128;
            const int row  = q >> 3;         // 0..63
            const int col4 = (q & 7) << 2;   // 0, 4, ..., 28
            const int gn   = n0 + row;
            const int gk   = k0 + col4;
            float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (gn < N && gk < K) {
                small_m_load4<SMALL_M_W_F32>(x, (size_t) gn * K + gk, v);
            }
            Bs[col4 + 0][row] = v[0];
            Bs[col4 + 1][row] = v[1];
            Bs[col4 + 2][row] = v[2];
            Bs[col4 + 3][row] = v[3];
        }

        __syncthreads();

#pragma unroll 8
        for (int k = 0; k < BK; ++k) {
            const float4 a = *reinterpret_cast<const float4 *>(&As[k][tx * 4]);
            const float4 b = *reinterpret_cast<const float4 *>(&Bs[k][ty * 4]);
            acc[0][0] = fmaf(a.x, b.x, acc[0][0]);
            acc[0][1] = fmaf(a.y, b.x, acc[0][1]);
            acc[0][2] = fmaf(a.z, b.x, acc[0][2]);
            acc[0][3] = fmaf(a.w, b.x, acc[0][3]);

            acc[1][0] = fmaf(a.x, b.y, acc[1][0]);
            acc[1][1] = fmaf(a.y, b.y, acc[1][1]);
            acc[1][2] = fmaf(a.z, b.y, acc[1][2]);
            acc[1][3] = fmaf(a.w, b.y, acc[1][3]);

            acc[2][0] = fmaf(a.x, b.z, acc[2][0]);
            acc[2][1] = fmaf(a.y, b.z, acc[2][1]);
            acc[2][2] = fmaf(a.z, b.z, acc[2][2]);
            acc[2][3] = fmaf(a.w, b.z, acc[2][3]);

            acc[3][0] = fmaf(a.x, b.w, acc[3][0]);
            acc[3][1] = fmaf(a.y, b.w, acc[3][1]);
            acc[3][2] = fmaf(a.z, b.w, acc[3][2]);
            acc[3][3] = fmaf(a.w, b.w, acc[3][3]);
        }

        __syncthreads();
    }

    const int gm0 = m0 + tx * 4;
    const int gn0 = n0 + ty * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int gn = gn0 + j;
        if (gn < N) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int gm = gm0 + i;
                if (gm < M) {
                    d[(size_t) gn * M + gm] = acc[j][i];
                }
            }
        }
    }
}

// Path B2: Small-N tiled F32 GEMM (BM=16, BN=16, BK=32, 64 threads, 2x2 per thread) for high CU occupancy when N <= 128.
template <int WTYPE>
__global__ void __launch_bounds__(64, 8) small_m_tiled_16x16_kernel(
        const void * __restrict__ w,
        const float * __restrict__ x,
        float * __restrict__ d,
        const int M, const int N, const int K) {
    constexpr int BM = 16;
    constexpr int BN = 16;
    constexpr int BK = 32;
    __shared__ float As[BK][BM + 4];
    __shared__ float Bs[BK][BN + 4];

    const int tid = threadIdx.x; // 0..63
    const int tx  = tid & 7;     // 0..7 -> 2 rows of M (16 total)
    const int ty  = tid >> 3;    // 0..7 -> 2 cols of N (16 total)

    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;

    float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int q    = tid + s * 64;
            const int row  = q >> 3;       // 0..15
            const int col4 = (q & 7) << 2; // 0, 4, ..., 28
            const int gk   = k0 + col4;
            float va[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (m0 + row < M && gk < K) {
                small_m_load4<WTYPE>(w, (size_t) (m0 + row) * K + gk, va);
            }
            As[col4 + 0][row] = va[0];
            As[col4 + 1][row] = va[1];
            As[col4 + 2][row] = va[2];
            As[col4 + 3][row] = va[3];

            float vb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (n0 + row < N && gk < K) {
                small_m_load4<SMALL_M_W_F32>(x, (size_t) (n0 + row) * K + gk, vb);
            }
            Bs[col4 + 0][row] = vb[0];
            Bs[col4 + 1][row] = vb[1];
            Bs[col4 + 2][row] = vb[2];
            Bs[col4 + 3][row] = vb[3];
        }

        __syncthreads();

#pragma unroll 8
        for (int k = 0; k < BK; ++k) {
            const float2 a = *reinterpret_cast<const float2 *>(&As[k][tx * 2]);
            const float2 b = *reinterpret_cast<const float2 *>(&Bs[k][ty * 2]);
            acc[0][0] = fmaf(a.x, b.x, acc[0][0]);
            acc[0][1] = fmaf(a.y, b.x, acc[0][1]);
            acc[1][0] = fmaf(a.x, b.y, acc[1][0]);
            acc[1][1] = fmaf(a.y, b.y, acc[1][1]);
        }

        __syncthreads();
    }

    const int gm0 = m0 + tx * 2;
    const int gn0 = n0 + ty * 2;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int gn = gn0 + j;
        if (gn < N) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int gm = gm0 + i;
                if (gm < M) {
                    d[(size_t) gn * M + gm] = acc[j][i];
                }
            }
        }
    }
}

template <int WTYPE>
static void small_m_gemm_dispatch(
        const void * w, const float * x, float * d,
        const int M, const int N, const int K, hipStream_t stream) {
    if (M <= 8) {
        if (N <= 128) {
            const unsigned grid = (unsigned) ((N + 3) / 4); // 4 warps x 1 tok
            if (M <= 4) small_m_skinny_kernel<4, WTYPE, 1, 4><<<grid, 128, 0, stream>>>(w, x, d, M, N, K);
            else        small_m_skinny_kernel<8, WTYPE, 1, 4><<<grid, 128, 0, stream>>>(w, x, d, M, N, K);
        } else {
            const unsigned grid = (unsigned) ((N + 15) / 16); // 8 warps x 2 tok
            if (M <= 4) small_m_skinny_kernel<4, WTYPE, 2, 8><<<grid, 256, 0, stream>>>(w, x, d, M, N, K);
            else        small_m_skinny_kernel<8, WTYPE, 2, 8><<<grid, 256, 0, stream>>>(w, x, d, M, N, K);
        }
        return;
    }

    if (N <= 128) {
        const dim3 grid((unsigned) ((M + 15) / 16), (unsigned) ((N + 15) / 16), 1);
        small_m_tiled_16x16_kernel<WTYPE><<<grid, 64, 0, stream>>>(w, x, d, M, N, K);
    } else {
        const dim3 grid((unsigned) ((M + 31) / 32), (unsigned) ((N + 63) / 64), 1);
        small_m_tiled_32x64_kernel<WTYPE><<<grid, 128, 0, stream>>>(w, x, d, M, N, K);
    }
}

static bool ggml_cuda_small_m_gemm_supported(
        const int device, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const bool enabled = []() {
        const char * e = std::getenv("LUCE_SMALL_M_GEMM");
        return e ? std::atoi(e) != 0 : true;
    }();
    if (!enabled) return false;

    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_RDNA4(cc)) return false;

    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_BF16 && src0->type != GGML_TYPE_F16) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (ggml_cuda_mmb_bf16_src(src1) != nullptr || ggml_cuda_mmb_is_f16_only(src1)) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return false;
    if (src0->ne[0] % 4 != 0) return false;
    if (src0->ne[1] < 1 || src0->ne[1] > 512) return false;
    if (src1->ne[1] < 16) return false;
    return true;
}

static void ggml_cuda_small_m_gemm(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int K = (int) src0->ne[0];
    const int M = (int) src0->ne[1];
    const int N = (int) src1->ne[1];
    hipStream_t stream = ctx.stream();
    if (src0->type == GGML_TYPE_BF16) {
        small_m_gemm_dispatch<SMALL_M_W_BF16>(src0->data, (const float *) src1->data, (float *) dst->data, M, N, K, stream);
    } else if (src0->type == GGML_TYPE_F16) {
        small_m_gemm_dispatch<SMALL_M_W_F16>(src0->data, (const float *) src1->data, (float *) dst->data, M, N, K, stream);
    } else {
        small_m_gemm_dispatch<SMALL_M_W_F32>(src0->data, (const float *) src1->data, (float *) dst->data, M, N, K, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

#endif // defined(GGML_USE_HIP)
