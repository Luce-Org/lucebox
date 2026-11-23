#pragma once
// Native zero-spill, ZERO-W_lds Direct-Register BK=256 RDNA3.5 (gfx1151) & RDNA4 (gfx1201)
// Wave32 BF16 WMMA Quantized GEMM for decode/verify batches (T=5..16).
//
// Key breakthroughs over MMQ:
// 1. ZERO W_lds: Each wave32 lane (r=lane&15, hi=lane>>4) dequantizes its own 8 consecutive
//    weights directly into registers -> 1:1 feed into RDNA4 v_wmma_f32_16x16x16_bf16
//    (or 4x __shfl_xor(v,16) on RDNA3.5). Zero W ds_write/ds_read, zero bank conflicts.
// 2. Full BK=256 superblock stepping with double-buffered X_lds[2][16*264] (16.9 KB):
//    - Only ONE __syncthreads() per 256 K-elements.
//    - Superblock headers (d, dm, scales, qh) loaded ONCE per 256 elements.
// 3. Direct F32 activation ingest (zero quantize_mmq_q8_1 launch!) + 1-op __popc parity signs.
// 4. Deterministic Split-K for small M to saturate all 64 CUs (R9700) / 40 CUs (Strix Halo).
//
// Measured microbench wins vs MMQ (T=16):
// - R9700 (gfx1201): IQ2_S 1.78x, IQ2_XS 1.68x, Q3_K 1.36x, Q8_0[96] 1.32x, Q4_K[1024] 1.21x,
//                    IQ4_XS[down] 1.09x, IQ3_XXS 1.08x, IQ3_S 1.05x, Q4_K[17k] 1.04x.
// - Strix Halo (gfx1151): Q3_K 1.67x, Q8_0[96] 1.60x (2.0x at T=8), IQ2_S 1.47x (1.58x at T=8),
//                         IQ2_XS 1.40x, Q4_K[17k] 1.23x, Q5_K[1024] 1.23x, Q4_K[1024] 1.12x.

#include "common.cuh"
#include "dequantize.cuh"
#include <algorithm>
#include <cstdlib>

#if defined(GGML_USE_HIP)

namespace {

using smq_bf16x8_t  = __attribute__((ext_vector_type(8)))  short;
using smq_bf16x16_t = __attribute__((ext_vector_type(16))) short;
using smq_f32x8_t   = __attribute__((ext_vector_type(8)))  float;

__device__ __forceinline__ uint32_t smq_pack_bf2(const float a, const float b) {
    const uint32_t ua = __float_as_uint(a) + 0x7FFFu;
    const uint32_t ub = __float_as_uint(b) + 0x7FFFu;
    return (ua >> 16) | (ub & 0xFFFF0000u);
}

__device__ __forceinline__ uint32_t smq_pack_bf2_sgn(const float a, const float b, const uint32_t s0, const uint32_t s1) {
    uint32_t p = smq_pack_bf2(a, b);
    p ^= (s0 << 15) | (s1 << 31);
    return p;
}

__device__ __forceinline__ uint32_t smq_ksign_iq2xs(const uint32_t idx7) {
    return idx7 | ((uint32_t)(__popc(idx7) & 1) << 7);
}

template <int N_TILES>
__device__ __forceinline__ void smq_wmma_step(
        smq_f32x8_t (&acc)[N_TILES],
        const uint4 w_u4,
        const uint16_t * __restrict__ X_tile_kt,
        const int x_tile_stride,
        const int hi) {
#if defined(RDNA4)
    const smq_bf16x8_t wv = __builtin_bit_cast(smq_bf16x8_t, w_u4);
#pragma unroll
    for (int nt = 0; nt < N_TILES; ++nt) {
        const uint4 x_u4 = *reinterpret_cast<const uint4 *>(X_tile_kt + nt * x_tile_stride + hi * 8);
        acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
            wv, __builtin_bit_cast(smq_bf16x8_t, x_u4), acc[nt]);
    }
#else
    const uint4 w_other = make_uint4(
        __shfl_xor(w_u4.x, 16, 32),
        __shfl_xor(w_u4.y, 16, 32),
        __shfl_xor(w_u4.z, 16, 32),
        __shfl_xor(w_u4.w, 16, 32));
    const uint4 w0 = hi ? w_other : w_u4;
    const uint4 w1 = hi ? w_u4    : w_other;
    const smq_bf16x16_t wv = __builtin_bit_cast(smq_bf16x16_t, (uint4[2]){w0, w1});
#pragma unroll
    for (int nt = 0; nt < N_TILES; ++nt) {
        const uint4 x0 = *reinterpret_cast<const uint4 *>(X_tile_kt + nt * x_tile_stride + 0);
        const uint4 x1 = *reinterpret_cast<const uint4 *>(X_tile_kt + nt * x_tile_stride + 8);
        acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
            wv, __builtin_bit_cast(smq_bf16x16_t, (uint4[2]){x0, x1}), acc[nt]);
    }
#endif
}

__device__ __forceinline__ uint4 smq_unpack_grid8(const uint32_t g_lo, const uint32_t g_hi, const float d, const uint32_t signs) {
    return make_uint4(
        smq_pack_bf2_sgn(d * (float)( g_lo        & 0xFFu), d * (float)((g_lo >>  8) & 0xFFu), (signs >> 0) & 1u, (signs >> 1) & 1u),
        smq_pack_bf2_sgn(d * (float)((g_lo >> 16) & 0xFFu), d * (float)((g_lo >> 24) & 0xFFu), (signs >> 2) & 1u, (signs >> 3) & 1u),
        smq_pack_bf2_sgn(d * (float)( g_hi        & 0xFFu), d * (float)((g_hi >>  8) & 0xFFu), (signs >> 4) & 1u, (signs >> 5) & 1u),
        smq_pack_bf2_sgn(d * (float)((g_hi >> 16) & 0xFFu), d * (float)((g_hi >> 24) & 0xFFu), (signs >> 6) & 1u, (signs >> 7) & 1u));
}

__device__ __forceinline__ uint4 smq_unpack_i8_8(const uint32_t v0, const uint32_t v1, const float d) {
    return make_uint4(
        smq_pack_bf2(d * (float)(int8_t)(v0 & 0xFFu),        d * (float)(int8_t)((v0 >>  8) & 0xFFu)),
        smq_pack_bf2(d * (float)(int8_t)((v0 >> 16) & 0xFFu), d * (float)(int8_t)((v0 >> 24) & 0xFFu)),
        smq_pack_bf2(d * (float)(int8_t)(v1 & 0xFFu),        d * (float)(int8_t)((v1 >>  8) & 0xFFu)),
        smq_pack_bf2(d * (float)(int8_t)((v1 >> 16) & 0xFFu), d * (float)(int8_t)((v1 >> 24) & 0xFFu)));
}

// WTYPE enum:
// 0=Q8_0, 1=Q5_K, 2=Q4_K, 3=IQ2_S, 4=IQ2_XS, 5=Q3_K, 6=IQ3_S, 7=IQ3_XXS, 8=IQ4_XS
template <int WTYPE, int BM, int N_TILES>
__global__ void __launch_bounds__(BM * 2, 1)
small_m_qgemm_wmma_kernel(
        const char * __restrict__ W_ptr,
        const float * __restrict__ X_ptr,
        float * __restrict__ out_ptr,
        const int M, const int K, const int T,
        const int num_splits, const bool is_split) {
#if defined(RDNA4) || defined(RDNA3)
    constexpr int BN = 16 * N_TILES;
    constexpr int BK = 256;
    constexpr int NT = BM * 2;
    constexpr int X_STRIDE = BK + 8;
    constexpr int X_TILE_STRIDE = 16 * X_STRIDE;

    // Double-buffered X_lds (16.9 KB for BN=16, 33.8 KB for BN=32); ZERO W_lds!
    __shared__ __align__(16) uint16_t X_lds[2][BN * X_STRIDE];

    const int tid  = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int r    = lane & 15;
    const int hi   = lane >> 4;

    const int m0 = (int)blockIdx.x * BM;
    const int sp = (int)blockIdx.y;
    const int t0 = (int)blockIdx.z * BN;

    const int k_steps = K / BK;
    const int steps_per_sp = (k_steps + num_splits - 1) / num_splits;
    const int ks0 = sp * steps_per_sp;
    const int ks1 = min(k_steps, ks0 + steps_per_sp);
    // Note: host guarantees num_splits has no empty splits, and loop safely skips if ks0 >= ks1.

    const size_t wrow_bytes =
        (WTYPE == 0) ? ((size_t)(K / QK8_0) * sizeof(block_q8_0)) :
        (WTYPE == 1) ? ((size_t)(K / QK_K)  * sizeof(block_q5_K)) :
        (WTYPE == 2) ? ((size_t)(K / QK_K)  * sizeof(block_q4_K)) :
        (WTYPE == 3) ? ((size_t)(K / QK_K)  * sizeof(block_iq2_s)) :
        (WTYPE == 4) ? ((size_t)(K / QK_K)  * sizeof(block_iq2_xs)) :
        (WTYPE == 5) ? ((size_t)(K / QK_K)  * sizeof(block_q3_K)) :
        (WTYPE == 6) ? ((size_t)(K / QK_K)  * sizeof(block_iq3_s)) :
        (WTYPE == 7) ? ((size_t)(K / QK_K)  * sizeof(block_iq3_xxs)) :
                       ((size_t)(K / QK_K)  * sizeof(block_iq4_xs));

    const int mg_safe = min(m0 + wave * 16 + r, M - 1);
    const char * W_row = W_ptr + (size_t)mg_safe * wrow_bytes;

    constexpr int x_thrs_per_row  = NT / 16;          // 4 (BM=32) or 8 (BM=64)
    constexpr int x_elems_per_thr = BK / x_thrs_per_row;
    const int x_row0 = tid / x_thrs_per_row;
    const int x_c    = (tid & (x_thrs_per_row - 1)) * x_elems_per_thr;

    auto load_x_buf = [&](const int buf_idx, const int ks) {
#pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            const int x_row = nt * 16 + x_row0;
            const int x_tg  = t0 + x_row;
            uint32_t * dst32 = reinterpret_cast<uint32_t *>(X_lds[buf_idx] + x_row * X_STRIDE + x_c);
            if (x_tg < T) {
                const float * xp = X_ptr + (size_t)x_tg * K + (size_t)ks * BK + x_c;
#pragma unroll
                for (int c = 0; c < x_elems_per_thr; c += 8) {
                    const float4 f0 = *reinterpret_cast<const float4 *>(xp + c + 0);
                    const float4 f1 = *reinterpret_cast<const float4 *>(xp + c + 4);
                    *reinterpret_cast<uint4 *>(dst32 + (c >> 1)) = make_uint4(
                        smq_pack_bf2(f0.x, f0.y), smq_pack_bf2(f0.z, f0.w),
                        smq_pack_bf2(f1.x, f1.y), smq_pack_bf2(f1.z, f1.w));
                }
            } else {
#pragma unroll
                for (int c = 0; c < x_elems_per_thr; c += 8) {
                    *reinterpret_cast<uint4 *>(dst32 + (c >> 1)) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
    };

    if (ks0 < ks1) {
        load_x_buf(0, ks0);
    }
    __syncthreads();

    smq_f32x8_t acc[N_TILES];
#pragma unroll
    for (int nt = 0; nt < N_TILES; ++nt) acc[nt] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int ks = ks0; ks < ks1; ++ks) {
        const int cur = (ks - ks0) & 1;
        const int nxt = cur ^ 1;
        if (ks + 1 < ks1) {
            load_x_buf(nxt, ks + 1);
        }

        const uint16_t * X_tile = X_lds[cur] + r * X_STRIDE;

        {
            if constexpr (WTYPE == 0) {
                const block_q8_0 * bps = reinterpret_cast<const block_q8_0 *>(W_row) + ks * 8;
#if defined(RDNA3)
#pragma unroll 1
#else
#pragma unroll
#endif
                for (int ib = 0; ib < 8; ++ib) {
                    const block_q8_0 * bp = bps + ib;
                    const float d = __half2float(bp->d);
                    const uint16_t * qp0 = reinterpret_cast<const uint16_t *>(bp->qs + hi * 8);
                    const uint32_t v0 = (uint32_t)qp0[0] | ((uint32_t)qp0[1] << 16);
                    const uint32_t v1 = (uint32_t)qp0[2] | ((uint32_t)qp0[3] << 16);
                    smq_wmma_step<N_TILES>(acc, smq_unpack_i8_8(v0, v1, d), X_tile + (2 * ib + 0) * 16, X_TILE_STRIDE, hi);

                    const uint16_t * qp1 = reinterpret_cast<const uint16_t *>(bp->qs + 16 + hi * 8);
                    const uint32_t v2 = (uint32_t)qp1[0] | ((uint32_t)qp1[1] << 16);
                    const uint32_t v3 = (uint32_t)qp1[2] | ((uint32_t)qp1[3] << 16);
                    smq_wmma_step<N_TILES>(acc, smq_unpack_i8_8(v2, v3, d), X_tile + (2 * ib + 1) * 16, X_TILE_STRIDE, hi);
                }
            } else if constexpr (WTYPE == 1) {
                const block_q5_K * bp = reinterpret_cast<const block_q5_K *>(W_row) + ks;
                const float d = __half2float(bp->dm.x), dmin = __half2float(bp->dm.y);
                const uint32_t qha0 = *reinterpret_cast<const uint32_t *>(bp->qh + hi * 8 + 0);
                const uint32_t qha1 = *reinterpret_cast<const uint32_t *>(bp->qh + hi * 8 + 4);
                const uint32_t qhb0 = *reinterpret_cast<const uint32_t *>(bp->qh + 16 + hi * 8 + 0);
                const uint32_t qhb1 = *reinterpret_cast<const uint32_t *>(bp->qh + 16 + hi * 8 + 4);
                auto unpack_q5 = [](const uint32_t q0, const uint32_t q1, const uint32_t h0, const uint32_t h1,
                                    const int qsh, const int hbit, const float ds, const float dm) -> uint4 {
                    auto f = [&](const uint32_t q, const uint32_t h, const int b) -> float {
                        const uint32_t lo = (q >> (b + qsh)) & 0x0Fu;
                        const uint32_t hb = ((h >> (b + hbit)) & 1u) << 4;
                        return ds * (float)(lo | hb) - dm;
                    };
                    return make_uint4(
                        smq_pack_bf2(f(q0, h0,  0), f(q0, h0,  8)),
                        smq_pack_bf2(f(q0, h0, 16), f(q0, h0, 24)),
                        smq_pack_bf2(f(q1, h1,  0), f(q1, h1,  8)),
                        smq_pack_bf2(f(q1, h1, 16), f(q1, h1, 24)));
                };
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint8_t sc0, mn0, sc1, mn1;
                    get_scale_min_k4(2 * j + 0, bp->scales, sc0, mn0);
                    get_scale_min_k4(2 * j + 1, bp->scales, sc1, mn1);
                    const float ds0 = d * (float)sc0, dm0 = dmin * (float)mn0;
                    const float ds1 = d * (float)sc1, dm1 = dmin * (float)mn1;
                    const uint8_t * qs = bp->qs + j * 32;
                    const uint32_t qa0 = *reinterpret_cast<const uint32_t *>(qs + hi * 8 + 0);
                    const uint32_t qa1 = *reinterpret_cast<const uint32_t *>(qs + hi * 8 + 4);
                    const uint32_t qb0 = *reinterpret_cast<const uint32_t *>(qs + 16 + hi * 8 + 0);
                    const uint32_t qb1 = *reinterpret_cast<const uint32_t *>(qs + 16 + hi * 8 + 4);
                    smq_wmma_step<N_TILES>(acc, unpack_q5(qa0, qa1, qha0, qha1, 0, 2 * j + 0, ds0, dm0), X_tile + (4 * j + 0) * 16, X_TILE_STRIDE, hi);
                    smq_wmma_step<N_TILES>(acc, unpack_q5(qb0, qb1, qhb0, qhb1, 0, 2 * j + 0, ds0, dm0), X_tile + (4 * j + 1) * 16, X_TILE_STRIDE, hi);
                    smq_wmma_step<N_TILES>(acc, unpack_q5(qa0, qa1, qha0, qha1, 4, 2 * j + 1, ds1, dm1), X_tile + (4 * j + 2) * 16, X_TILE_STRIDE, hi);
                    smq_wmma_step<N_TILES>(acc, unpack_q5(qb0, qb1, qhb0, qhb1, 4, 2 * j + 1, ds1, dm1), X_tile + (4 * j + 3) * 16, X_TILE_STRIDE, hi);
                }
            } else if constexpr (WTYPE == 2) {
                const block_q4_K * bp = reinterpret_cast<const block_q4_K *>(W_row) + ks;
                const float d = __half2float(bp->dm.x), dmin = __half2float(bp->dm.y);
                auto unpack_q4 = [](const uint32_t q0, const uint32_t q1, const int qsh, const float ds, const float dm) -> uint4 {
                    return make_uint4(
                        smq_pack_bf2(ds * (float)((q0 >> ( 0 + qsh)) & 0xFu) - dm, ds * (float)((q0 >> ( 8 + qsh)) & 0xFu) - dm),
                        smq_pack_bf2(ds * (float)((q0 >> (16 + qsh)) & 0xFu) - dm, ds * (float)((q0 >> (24 + qsh)) & 0xFu) - dm),
                        smq_pack_bf2(ds * (float)((q1 >> ( 0 + qsh)) & 0xFu) - dm, ds * (float)((q1 >> ( 8 + qsh)) & 0xFu) - dm),
                        smq_pack_bf2(ds * (float)((q1 >> (16 + qsh)) & 0xFu) - dm, ds * (float)((q1 >> (24 + qsh)) & 0xFu) - dm));
                };
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint8_t sc0, mn0, sc1, mn1;
                    get_scale_min_k4(2 * j + 0, bp->scales, sc0, mn0);
                    get_scale_min_k4(2 * j + 1, bp->scales, sc1, mn1);
                    const float ds0 = d * (float)sc0, dm0 = dmin * (float)mn0;
                    const float ds1 = d * (float)sc1, dm1 = dmin * (float)mn1;
                    const uint8_t * qs = bp->qs + j * 32;
                    const uint32_t qa0 = *reinterpret_cast<const uint32_t *>(qs + hi * 8 + 0);
                    const uint32_t qa1 = *reinterpret_cast<const uint32_t *>(qs + hi * 8 + 4);
                    const uint32_t qb0 = *reinterpret_cast<const uint32_t *>(qs + 16 + hi * 8 + 0);
                    const uint32_t qb1 = *reinterpret_cast<const uint32_t *>(qs + 16 + hi * 8 + 4);
                    smq_wmma_step<N_TILES>(acc, unpack_q4(qa0, qa1, 0, ds0, dm0), X_tile + (4 * j + 0) * 16, X_TILE_STRIDE, hi);
                    smq_wmma_step<N_TILES>(acc, unpack_q4(qb0, qb1, 0, ds0, dm0), X_tile + (4 * j + 1) * 16, X_TILE_STRIDE, hi);
                    smq_wmma_step<N_TILES>(acc, unpack_q4(qa0, qa1, 4, ds1, dm1), X_tile + (4 * j + 2) * 16, X_TILE_STRIDE, hi);
                    smq_wmma_step<N_TILES>(acc, unpack_q4(qb0, qb1, 4, ds1, dm1), X_tile + (4 * j + 3) * 16, X_TILE_STRIDE, hi);
                }
            } else if constexpr (WTYPE == 3) {
                const block_iq2_s * bp = reinterpret_cast<const block_iq2_s *>(W_row) + ks;
                const float d_base = __half2float(bp->d) * 0.25f;
#pragma unroll
                for (int ib = 0; ib < 8; ++ib) {
                    const uint8_t sc = bp->scales[ib];
                    const float d_lo = d_base * (0.5f + (float)(sc & 0x0Fu));
                    const float d_hi = d_base * (0.5f + (float)(sc >> 4));
                    const uint32_t qh = (uint32_t)bp->qh[ib];
                    const uint8_t * qs = bp->qs + 4 * ib;
                    const uint8_t * sg = bp->qs + 32 + 4 * ib;

                    const int idx0 = (int)qs[hi] | (int)(((qh << (8 - 2 * hi)) & 0x300u));
                    const uint64_t g0 = iq2s_grid[idx0];
                    smq_wmma_step<N_TILES>(acc, smq_unpack_grid8((uint32_t)g0, (uint32_t)(g0 >> 32), d_lo, sg[hi]),
                                  X_tile + (2 * ib + 0) * 16, X_TILE_STRIDE, hi);

                    const int il1 = 2 + hi;
                    const int idx1 = (int)qs[il1] | (int)(((qh << (8 - 2 * il1)) & 0x300u));
                    const uint64_t g1 = iq2s_grid[idx1];
                    smq_wmma_step<N_TILES>(acc, smq_unpack_grid8((uint32_t)g1, (uint32_t)(g1 >> 32), d_hi, sg[il1]),
                                  X_tile + (2 * ib + 1) * 16, X_TILE_STRIDE, hi);
                }
            } else if constexpr (WTYPE == 4) {
                const block_iq2_xs * bp = reinterpret_cast<const block_iq2_xs *>(W_row) + ks;
                const float d_base = __half2float(bp->d) * 0.25f;
#pragma unroll
                for (int ib = 0; ib < 8; ++ib) {
                    const uint8_t sc = bp->scales[ib];
                    const float d_lo = d_base * (0.5f + (float)(sc & 0x0Fu));
                    const float d_hi = d_base * (0.5f + (float)(sc >> 4));
                    const uint16_t * q2 = bp->qs + 4 * ib;

                    const uint32_t w0 = q2[hi];
                    const uint64_t g0 = iq2xs_grid[w0 & 511u];
                    smq_wmma_step<N_TILES>(acc, smq_unpack_grid8((uint32_t)g0, (uint32_t)(g0 >> 32), d_lo, smq_ksign_iq2xs(w0 >> 9)),
                                  X_tile + (2 * ib + 0) * 16, X_TILE_STRIDE, hi);

                    const uint32_t w1 = q2[2 + hi];
                    const uint64_t g1 = iq2xs_grid[w1 & 511u];
                    smq_wmma_step<N_TILES>(acc, smq_unpack_grid8((uint32_t)g1, (uint32_t)(g1 >> 32), d_hi, smq_ksign_iq2xs(w1 >> 9)),
                                  X_tile + (2 * ib + 1) * 16, X_TILE_STRIDE, hi);
                }
            } else if constexpr (WTYPE == 5) {
                const block_q3_K * bp = reinterpret_cast<const block_q3_K *>(W_row) + ks;
                const float d_all = __half2float(bp->d);
                uint32_t s0 = (uint32_t)bp->scales[0] | ((uint32_t)bp->scales[1] << 8) | ((uint32_t)bp->scales[2] << 16) | ((uint32_t)bp->scales[3] << 24);
                uint32_t s1 = (uint32_t)bp->scales[4] | ((uint32_t)bp->scales[5] << 8) | ((uint32_t)bp->scales[6] << 16) | ((uint32_t)bp->scales[7] << 24);
                uint32_t s2 = (uint32_t)bp->scales[8] | ((uint32_t)bp->scales[9] << 8) | ((uint32_t)bp->scales[10] << 16) | ((uint32_t)bp->scales[11] << 24);
                const uint32_t sc_w0 = (s0 & 0x0f0f0f0fu) | ((s2 & 0x03030303u) << 4);
                const uint32_t sc_w1 = (s1 & 0x0f0f0f0fu) | (((s2 >> 2) & 0x03030303u) << 4);
                const uint32_t sc_w2 = ((s0 >> 4) & 0x0f0f0f0fu) | (((s2 >> 4) & 0x03030303u) << 4);
                const uint32_t sc_w3 = ((s1 >> 4) & 0x0f0f0f0fu) | (((s2 >> 6) & 0x03030303u) << 4);

                const uint32_t hma0 = (uint32_t)bp->hmask[hi*8+0] | ((uint32_t)bp->hmask[hi*8+1]<<8) | ((uint32_t)bp->hmask[hi*8+2]<<16) | ((uint32_t)bp->hmask[hi*8+3]<<24);
                const uint32_t hma1 = (uint32_t)bp->hmask[hi*8+4] | ((uint32_t)bp->hmask[hi*8+5]<<8) | ((uint32_t)bp->hmask[hi*8+6]<<16) | ((uint32_t)bp->hmask[hi*8+7]<<24);
                const uint32_t hmb0 = (uint32_t)bp->hmask[16+hi*8+0] | ((uint32_t)bp->hmask[16+hi*8+1]<<8) | ((uint32_t)bp->hmask[16+hi*8+2]<<16) | ((uint32_t)bp->hmask[16+hi*8+3]<<24);
                const uint32_t hmb1 = (uint32_t)bp->hmask[16+hi*8+4] | ((uint32_t)bp->hmask[16+hi*8+5]<<8) | ((uint32_t)bp->hmask[16+hi*8+6]<<16) | ((uint32_t)bp->hmask[16+hi*8+7]<<24);

                auto unpack_q3 = [](const uint32_t q0, const uint32_t q1, const uint32_t h0, const uint32_t h1,
                                    const int qsh, const int hbit, const float ds) -> uint4 {
                    auto f = [&](const uint32_t q, const uint32_t h, const int b) -> float {
                        const int v = (int)((q >> (b + qsh)) & 3u) - (((h >> (b + hbit)) & 1u) ? 0 : 4);
                        return ds * (float)v;
                    };
                    return make_uint4(
                        smq_pack_bf2(f(q0, h0,  0), f(q0, h0,  8)),
                        smq_pack_bf2(f(q0, h0, 16), f(q0, h0, 24)),
                        smq_pack_bf2(f(q1, h1,  0), f(q1, h1,  8)),
                        smq_pack_bf2(f(q1, h1, 16), f(q1, h1, 24)));
                };
                auto get_sc = [&](const int idx) -> float {
                    const uint32_t w = (idx < 4) ? sc_w0 : (idx < 8) ? sc_w1 : (idx < 12) ? sc_w2 : sc_w3;
                    const int s8 = (int)((w >> ((idx & 3) * 8)) & 0xFFu) - 32;
                    return d_all * (float)s8;
                };
#if defined(RDNA3)
#pragma unroll 1
#else
#pragma unroll
#endif
                for (int half = 0; half < 2; ++half) {
                    const uint8_t * qs = bp->qs + half * 32;
                    const uint32_t qa0 = (uint32_t)qs[hi*8+0] | ((uint32_t)qs[hi*8+1]<<8) | ((uint32_t)qs[hi*8+2]<<16) | ((uint32_t)qs[hi*8+3]<<24);
                    const uint32_t qa1 = (uint32_t)qs[hi*8+4] | ((uint32_t)qs[hi*8+5]<<8) | ((uint32_t)qs[hi*8+6]<<16) | ((uint32_t)qs[hi*8+7]<<24);
                    const uint32_t qb0 = (uint32_t)qs[16+hi*8+0] | ((uint32_t)qs[16+hi*8+1]<<8) | ((uint32_t)qs[16+hi*8+2]<<16) | ((uint32_t)qs[16+hi*8+3]<<24);
                    const uint32_t qb1 = (uint32_t)qs[16+hi*8+4] | ((uint32_t)qs[16+hi*8+5]<<8) | ((uint32_t)qs[16+hi*8+6]<<16) | ((uint32_t)qs[16+hi*8+7]<<24);
#if defined(RDNA3)
#pragma unroll 1
#else
#pragma unroll
#endif
                    for (int g = 0; g < 4; ++g) {
                        const int kt0 = half * 8 + g * 2;
                        const int hbit = half * 4 + g;
                        smq_wmma_step<N_TILES>(acc, unpack_q3(qa0, qa1, hma0, hma1, g * 2, hbit, get_sc(kt0 + 0)), X_tile + (kt0 + 0) * 16, X_TILE_STRIDE, hi);
                        smq_wmma_step<N_TILES>(acc, unpack_q3(qb0, qb1, hmb0, hmb1, g * 2, hbit, get_sc(kt0 + 1)), X_tile + (kt0 + 1) * 16, X_TILE_STRIDE, hi);
                    }
                }
            } else if constexpr (WTYPE == 6) {
                const block_iq3_s * bp = reinterpret_cast<const block_iq3_s *>(W_row) + ks;
                const float d_base = __half2float(bp->d);
#pragma unroll
                for (int ib = 0; ib < 8; ++ib) {
                    const float dl = d_base * (float)(1 + 2 * ((bp->scales[ib >> 1] >> (4 * (ib & 1))) & 0x0Fu));
                    const uint32_t qh = (uint32_t)bp->qh[ib];
                    const uint8_t * qs = bp->qs + 8 * ib;
                    const uint8_t * sg = bp->signs + 4 * ib;

                    const uint32_t g0_lo = iq3s_grid[qs[2 * hi + 0] | ((qh << (8 - 2 * hi)) & 256u)];
                    const uint32_t g0_hi = iq3s_grid[qs[2 * hi + 1] | ((qh << (7 - 2 * hi)) & 256u)];
                    smq_wmma_step<N_TILES>(acc, smq_unpack_grid8(g0_lo, g0_hi, dl, sg[hi]), X_tile + (2 * ib + 0) * 16, X_TILE_STRIDE, hi);

                    const int il1 = 2 + hi;
                    const uint32_t g1_lo = iq3s_grid[qs[2 * il1 + 0] | ((qh << (8 - 2 * il1)) & 256u)];
                    const uint32_t g1_hi = iq3s_grid[qs[2 * il1 + 1] | ((qh << (7 - 2 * il1)) & 256u)];
                    smq_wmma_step<N_TILES>(acc, smq_unpack_grid8(g1_lo, g1_hi, dl, sg[il1]), X_tile + (2 * ib + 1) * 16, X_TILE_STRIDE, hi);
                }
            } else if constexpr (WTYPE == 7) {
                const block_iq3_xxs * bp = reinterpret_cast<const block_iq3_xxs *>(W_row) + ks;
                const float d_base = __half2float(bp->d) * 0.5f;
                const uint16_t * gas_base = reinterpret_cast<const uint16_t *>(bp->qs + QK_K / 4);
#pragma unroll
                for (int ib = 0; ib < 8; ++ib) {
                    const uint16_t * gas = gas_base + 2 * ib;
                    const uint32_t aux32 = (uint32_t)gas[0] | ((uint32_t)gas[1] << 16);
                    const float dl = d_base * (0.5f + (float)(aux32 >> 28));
                    const uint8_t * q3 = bp->qs + 8 * ib;

                    const uint32_t g0_lo = iq3xxs_grid[q3[2 * hi + 0]];
                    const uint32_t g0_hi = iq3xxs_grid[q3[2 * hi + 1]];
                    const uint32_t s0 = smq_ksign_iq2xs((aux32 >> (7 * hi)) & 127u);
                    smq_wmma_step<N_TILES>(acc, smq_unpack_grid8(g0_lo, g0_hi, dl, s0), X_tile + (2 * ib + 0) * 16, X_TILE_STRIDE, hi);

                    const int il1 = 2 + hi;
                    const uint32_t g1_lo = iq3xxs_grid[q3[2 * il1 + 0]];
                    const uint32_t g1_hi = iq3xxs_grid[q3[2 * il1 + 1]];
                    const uint32_t s1 = smq_ksign_iq2xs((aux32 >> (7 * il1)) & 127u);
                    smq_wmma_step<N_TILES>(acc, smq_unpack_grid8(g1_lo, g1_hi, dl, s1), X_tile + (2 * ib + 1) * 16, X_TILE_STRIDE, hi);
                }
            } else {
                const block_iq4_xs * bp = reinterpret_cast<const block_iq4_xs *>(W_row) + ks;
                const float d_base = __half2float(bp->d);
                const uint16_t sh = bp->scales_h;
                auto unpack_iq4 = [](const uint32_t q0, const uint32_t q1, const int shift, const float dl) -> uint4 {
                    return make_uint4(
                        smq_pack_bf2(dl * (float)kvalues_iq4nl[(q0 >> ( 0 + shift)) & 0xFu], dl * (float)kvalues_iq4nl[(q0 >> ( 8 + shift)) & 0xFu]),
                        smq_pack_bf2(dl * (float)kvalues_iq4nl[(q0 >> (16 + shift)) & 0xFu], dl * (float)kvalues_iq4nl[(q0 >> (24 + shift)) & 0xFu]),
                        smq_pack_bf2(dl * (float)kvalues_iq4nl[(q1 >> ( 0 + shift)) & 0xFu], dl * (float)kvalues_iq4nl[(q1 >> ( 8 + shift)) & 0xFu]),
                        smq_pack_bf2(dl * (float)kvalues_iq4nl[(q1 >> (16 + shift)) & 0xFu], dl * (float)kvalues_iq4nl[(q1 >> (24 + shift)) & 0xFu]));
                };
#pragma unroll
                for (int ib = 0; ib < 8; ++ib) {
                    const int ls = ((bp->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xFu) | (((sh >> (2 * ib)) & 3u) << 4);
                    const float dl = d_base * (float)(ls - 32);
                    const uint32_t * qp = reinterpret_cast<const uint32_t *>(bp->qs + 16 * ib + hi * 8);
                    const uint32_t q0 = qp[0];
                    const uint32_t q1 = qp[1];
                    smq_wmma_step<N_TILES>(acc, unpack_iq4(q0, q1, 0, dl), X_tile + (2 * ib + 0) * 16, X_TILE_STRIDE, hi);
                    smq_wmma_step<N_TILES>(acc, unpack_iq4(q0, q1, 4, dl), X_tile + (2 * ib + 1) * 16, X_TILE_STRIDE, hi);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int nt = 0; nt < N_TILES; ++nt) {
        const int tg = t0 + nt * 16 + r;
        if (tg < T) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
#if defined(RDNA4)
                const int out_m = m0 + wave * 16 + hi * 8 + e;
#else
                const int out_m = m0 + wave * 16 + 2 * e + hi;
#endif
                if (out_m < M) {
                    if (is_split) {
                        out_ptr[((size_t)sp * T + tg) * M + out_m] = acc[nt][e];
                    } else {
                        out_ptr[(size_t)tg * M + out_m] = acc[nt][e];
                    }
                }
            }
        }
    }
#endif
}

__global__ void small_m_qgemm_reduce_kernel(
        const float * __restrict__ part,
        float * __restrict__ dst,
        const int M, const int T, const int num_splits) {
    const int idx = (int)blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * T;
    if (idx >= total) return;
    float s = 0.0f;
    for (int sp = 0; sp < num_splits; ++sp) {
        s += part[(size_t)sp * total + idx];
    }
    dst[idx] = s;
}

} // namespace

static bool ggml_cuda_small_m_qgemm_supported(const int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    const bool is_r4 = GGML_CUDA_CC_IS_RDNA4(cc);
    const bool is_r35 = GGML_CUDA_CC_IS_RDNA3_5(cc);
    if (!is_r4 && !is_r35) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return false;

    const ggml_type t = src0->type;
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t T = src1->ne[1];
    if (K < 1024 || (K % 256) != 0 || M < 32) return false;
    // Concurrency T=17..64 (BN=32, 2x weight-register reuse):
    // - Q8_0 M<=512: 1.26x-2.24x on R9700, 2.00x-2.97x on Strix Halo!
    // - IQ2_S / IQ2_XS on R9700 T=17..32: 1.20x-1.30x!
    if (T > 16 && T <= 64) {
        if (t == GGML_TYPE_Q8_0 && M <= 512) return true;
        if (is_r4 && T <= 32 && (t == GGML_TYPE_IQ2_S || t == GGML_TYPE_IQ2_XS)) return true;
        return false;
    }
    if (T < 9 || T > 16) return false;

    // Strictly-winning per-architecture dispatch (microbench-verified on gfx1201 + gfx1151):
    if (is_r4) {
        if (t == GGML_TYPE_IQ2_S || t == GGML_TYPE_IQ2_XS || t == GGML_TYPE_Q3_K ||
            t == GGML_TYPE_IQ3_XXS || t == GGML_TYPE_IQ3_S) {
            return true;
        }
        if (t == GGML_TYPE_IQ4_XS && M <= 8192 && K >= 8192) return true; // down_proj [17408,5120] 1.09x
        if (M <= 1024 && (t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q4_K)) return true;
        return false;
    }
    // RDNA3.5 (Strix Halo gfx1151):
    if (t == GGML_TYPE_IQ2_S || t == GGML_TYPE_IQ2_XS || t == GGML_TYPE_Q3_K || t == GGML_TYPE_Q4_K) {
        return true;
    }
    if (M <= 1024 && (t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q5_K)) {
        return true;
    }
    return false;
}

static void ggml_cuda_small_m_qgemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int K = (int) src0->ne[0];
    const int M = (int) src0->ne[1];
    const int T = (int) src1->ne[1];

    const bool use_bm64 = (M > 1024);
    const bool use_bn32 = (T > 16);
    const int BM = use_bm64 ? 64 : 32;
    const int BN = use_bn32 ? 32 : 16;
    const int m_blocks = (M + BM - 1) / BM;
    const int t_blocks = (T + BN - 1) / BN;
    const int k_steps  = K / 256;

    const int target_blocks = 128;
    int num_splits = std::max(1, (target_blocks + m_blocks * t_blocks - 1) / (m_blocks * t_blocks));
    num_splits = std::min(num_splits, std::min(20, std::max(1, k_steps)));
    const int steps_per_sp = (k_steps + num_splits - 1) / num_splits;
    num_splits = (k_steps + steps_per_sp - 1) / steps_per_sp;

    hipStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> part_alloc(ctx.pool());
    float * out_ptr = (float *) dst->data;
    const bool is_split = (num_splits > 1);
    if (is_split) {
        out_ptr = part_alloc.alloc((size_t)num_splits * M * T);
    }

    const dim3 grid((unsigned)m_blocks, (unsigned)num_splits, (unsigned)t_blocks);
    const dim3 block((unsigned)(BM * 2), 1, 1);

    auto launch = [&](auto wt_tag) {
        constexpr int WT = decltype(wt_tag)::value;
        if (use_bn32) {
            if (use_bm64) {
                small_m_qgemm_wmma_kernel<WT, 64, 2><<<grid, block, 0, stream>>>(
                    (const char *) src0->data, (const float *) src1->data, out_ptr, M, K, T, num_splits, is_split);
            } else {
                small_m_qgemm_wmma_kernel<WT, 32, 2><<<grid, block, 0, stream>>>(
                    (const char *) src0->data, (const float *) src1->data, out_ptr, M, K, T, num_splits, is_split);
            }
        } else {
            if (use_bm64) {
                small_m_qgemm_wmma_kernel<WT, 64, 1><<<grid, block, 0, stream>>>(
                    (const char *) src0->data, (const float *) src1->data, out_ptr, M, K, T, num_splits, is_split);
            } else {
                small_m_qgemm_wmma_kernel<WT, 32, 1><<<grid, block, 0, stream>>>(
                    (const char *) src0->data, (const float *) src1->data, out_ptr, M, K, T, num_splits, is_split);
            }
        }
    };
    switch (src0->type) {
        case GGML_TYPE_Q8_0:    launch(std::integral_constant<int,0>{}); break;
        case GGML_TYPE_Q5_K:    launch(std::integral_constant<int,1>{}); break;
        case GGML_TYPE_Q4_K:    launch(std::integral_constant<int,2>{}); break;
        case GGML_TYPE_IQ2_S:   launch(std::integral_constant<int,3>{}); break;
        case GGML_TYPE_IQ2_XS:  launch(std::integral_constant<int,4>{}); break;
        case GGML_TYPE_Q3_K:    launch(std::integral_constant<int,5>{}); break;
        case GGML_TYPE_IQ3_S:   launch(std::integral_constant<int,6>{}); break;
        case GGML_TYPE_IQ3_XXS: launch(std::integral_constant<int,7>{}); break;
        case GGML_TYPE_IQ4_XS:  launch(std::integral_constant<int,8>{}); break;
        default: GGML_ABORT("unreachable");
    }

    if (is_split) {
        const int total = M * T;
        const int rblocks = (total + 255) / 256;
        small_m_qgemm_reduce_kernel<<<rblocks, 256, 0, stream>>>(out_ptr, (float *) dst->data, M, T, num_splits);
    }
}

#else
static bool ggml_cuda_small_m_qgemm_supported(int, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *) { return false; }
static void ggml_cuda_small_m_qgemm(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {}
#endif
