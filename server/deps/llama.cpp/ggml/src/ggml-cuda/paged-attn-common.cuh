#pragma once

// Warp exchanges, Q8_0 dequantization and tile stores of the paged attention
// kernels, kept apart so test_paged_attn_lanes can check them bit for bit
// against the plain shuffles, the float dequantization and the row-major
// tile they replace.

#include "common.cuh"

#if defined(GGML_USE_HIP) && defined(RDNA)
// DPP row_xmask:m (control 0x160 | m) reads lane (lane ^ m) within each row of
// 16 lanes.
static constexpr int PAGED_ATTN_DPP_ROW_XMASK = 0x160;
// v_permlanex16 gives lane i of a row of 16 the value of lane sel[i] of the
// other row in its 32. These selectors map lanes 0-7 and 8-15 to themselves,
// so lane l reads lane l ^ 16.
static constexpr unsigned PAGED_ATTN_PERMLANEX16_SEL_LO = 0x76543210u;
static constexpr unsigned PAGED_ATTN_PERMLANEX16_SEL_HI = 0xfedcba98u;
#endif // defined(GGML_USE_HIP) && defined(RDNA)

// Lane src's copy of x, with src the same on every lane. RDNA runs wave32, so
// it reads the lane into a scalar register instead of routing every lane
// through a cross-lane shuffle. On wave64 targets two logical warps share a
// wave and a hardware lane number would name the other warp's lane.
static __device__ __forceinline__ int paged_attn_lane_value(int x, int src) {
#if defined(GGML_USE_HIP) && defined(RDNA)
    return __builtin_amdgcn_readlane(x, src);
#else
    return __shfl_sync(0xFFFFFFFF, x, src, WARP_SIZE);
#endif // defined(GGML_USE_HIP) && defined(RDNA)
}

static __device__ __forceinline__ float paged_attn_lane_value(float x, int src) {
    return __int_as_float(paged_attn_lane_value(__float_as_int(x), src));
}

// Lane (lane ^ offset)'s copy of x. RDNA moves it inside the VALU, with DPP
// within each row of 16 lanes and a row swap for offset 16, instead of a
// cross-lane shuffle through LDS.
static __device__ __forceinline__ float paged_attn_xor_lane(float x, int offset) {
#if defined(GGML_USE_HIP) && defined(RDNA)
    const int xi = __float_as_int(x);
    switch (offset) {
        case 1:  return __int_as_float(__builtin_amdgcn_update_dpp(
                     xi, xi, PAGED_ATTN_DPP_ROW_XMASK | 1, 0xF, 0xF, false));
        case 2:  return __int_as_float(__builtin_amdgcn_update_dpp(
                     xi, xi, PAGED_ATTN_DPP_ROW_XMASK | 2, 0xF, 0xF, false));
        case 4:  return __int_as_float(__builtin_amdgcn_update_dpp(
                     xi, xi, PAGED_ATTN_DPP_ROW_XMASK | 4, 0xF, 0xF, false));
        case 8:  return __int_as_float(__builtin_amdgcn_update_dpp(
                     xi, xi, PAGED_ATTN_DPP_ROW_XMASK | 8, 0xF, 0xF, false));
        case 16: return __int_as_float(__builtin_amdgcn_permlanex16(
                     xi, xi, PAGED_ATTN_PERMLANEX16_SEL_LO, PAGED_ATTN_PERMLANEX16_SEL_HI,
                     false, false));
        default: break;
    }
#endif // defined(GGML_USE_HIP) && defined(RDNA)
    return __shfl_xor_sync(0xFFFFFFFF, x, offset, WARP_SIZE);
}

// warp_reduce_sum and warp_reduce_max with the same butterfly, so the same
// additions and comparisons in the same order, through paged_attn_xor_lane.
static __device__ __forceinline__ float paged_attn_warp_sum(float x) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        x += paged_attn_xor_lane(x, offset);
    }
    return x;
}

static __device__ __forceinline__ float paged_attn_warp_max(float x) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, paged_attn_xor_lane(x, offset));
    }
    return x;
}

// Reduces the warp's partial sums of n tokens at once and returns the full
// sum of token lane / (WARP_SIZE/n). The first log2(n) butterfly levels
// exchange half of the remaining tokens instead of one whole sum, so every
// exchange carries a value the partner still needs: the lower lane of a pair
// keeps the lower half and the upper lane the upper half. Every token's sum
// still takes exactly the additions of paged_attn_warp_sum, in the same order.
template<int n>
static __device__ __forceinline__ float paged_attn_reduce_group(float (&sums)[n]) {
    static_assert(n >= 1 && n <= WARP_SIZE && (n & (n - 1)) == 0, "bad group size");
#pragma unroll
    for (int half = n / 2, offset = WARP_SIZE / 2; half >= 1; half /= 2, offset /= 2) {
        const bool upper = threadIdx.x & offset;
#pragma unroll
        for (int i = 0; i < half; ++i) {
            const float keep = upper ? sums[half + i] : sums[i];
            const float send = upper ? sums[i] : sums[half + i];
            sums[i] = keep + paged_attn_xor_lane(send, offset);
        }
    }
    float sum = sums[0];
#pragma unroll
    for (int offset = WARP_SIZE / (2 * n); offset > 0; offset >>= 1) {
        sum += paged_attn_xor_lane(sum, offset);
    }
    return sum;
}

// Four Q8_0 quants, little-endian in qs, times the block scale d as two half2
// words (quants 0, 1 and 2, 3). Under the high byte 0x64 of the f16 1024, the
// byte q ^ 0x80 = q + 128 makes the f16 1024 + (q + 128), so subtracting 1152
// leaves q exactly, and the f16 product with d rounds once, like
// __float2half(__half2float(d) * q). Only a zero product can differ: it keeps
// its IEEE sign, where ROCm's __float2half(-0.0f) returns +0. No attention
// output sees that sign, as every mma accumulator starts at +0.
// mmb-q8f16.cuh keeps its own copy of this construction on _Float16 vectors.
static __device__ __forceinline__ void paged_attn_q8_0_to_half2(
        uint32_t qs, half2 d, uint32_t (&h)[2]) {
    const uint32_t biased = qs ^ 0x80808080u;
#ifdef GGML_USE_HIP
    const uint32_t pair[2] = {
        __builtin_amdgcn_perm(biased, 0x64646464u, 0x01050004u),
        __builtin_amdgcn_perm(biased, 0x64646464u, 0x03070206u),
    };
#else
    const uint32_t pair[2] = {
        __byte_perm(biased, 0x64646464u, 0x4140),
        __byte_perm(biased, 0x64646464u, 0x4342),
    };
#endif // GGML_USE_HIP
    const half2 bias = __float2half2_rn(1152.0f);
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const half2 hv = __hmul2(__hsub2(*reinterpret_cast<const half2 *>(&pair[p]), bias), d);
        h[p] = *reinterpret_cast<const uint32_t *>(&hv);
    }
}

// Threads that stage one token row of a K/V tile: thread tid owns row
// tid / PAGED_ATTN_WMMA_ROW_THREADS, so the two rows of a token pair sit
// PAGED_ATTN_WMMA_ROW_THREADS lanes apart in one warp.
static constexpr int PAGED_ATTN_WMMA_ROW_THREADS = 4;

// Writes one thread's segment of `vec` half2s to a tokens x dims tile as
// vec/4 16-byte stores.
template <int vec>
static __device__ __forceinline__ void paged_attn_wmma_store_segment(
        uint32_t * __restrict__ out, const uint32_t (&h)[vec]) {
#pragma unroll
    for (int m = 0; m < vec / 4; ++m) {
        *reinterpret_cast<uint4 *>(out + 4 * m) =
            make_uint4(h[4*m], h[4*m+1], h[4*m+2], h[4*m+3]);
    }
}

// Column of token pair `pair` in row `dim` of a dims x tokens tile. The
// 16-byte chunks of 4 pairs are XOR-swizzled by dim/16: the lanes of one
// transposed store write rows of 8 different 16-dim groups, which the row
// stride alone maps to the same banks.
static __host__ __device__ __forceinline__ int paged_attn_wmma_pair_col(int dim, int pair) {
    return 4 * ((pair / 4) ^ (dim / 16 % 8)) + pair % 4;
}

// Writes one thread's segment of token row `row` into a dims x tokens tile,
// row dim holding the half2 (row & ~1, row | 1) at paged_attn_wmma_pair_col,
// which is how the V operand of the VKQ mma is laid out, so the fragments
// load without a transposing mma. Each thread of a token pair writes 16 of
// the segment's dims and trades the others with its partner.
template <int vec, int stride_tile>
static __device__ __forceinline__ void paged_attn_wmma_store_transposed(
        half2 * __restrict__ tile, int row, int sub, const uint32_t (&h)[vec]) {
    constexpr int half_vec = vec / 2;
    static_assert(vec == 16, "one 16-dim swizzle group per thread");
    static_assert((PAGED_ATTN_WMMA_ROW_THREADS & (PAGED_ATTN_WMMA_ROW_THREADS - 1)) == 0 &&
                  WARP_SIZE % (2 * PAGED_ATTN_WMMA_ROW_THREADS) == 0,
                  "a token pair must sit in one warp at a power-of-two distance");
    const bool odd = row & 1;
    const int first_dim = 2*sub*vec + (odd ? vec : 0);
    half2 * col = tile + first_dim * stride_tile + paged_attn_wmma_pair_col(first_dim, row / 2);
#pragma unroll
    for (int m = 0; m < half_vec; ++m) {
        const uint32_t mine = odd ? h[half_vec + m] : h[m];
        const uint32_t sent = odd ? h[m] : h[half_vec + m];
        const uint32_t other = __float_as_uint(
            paged_attn_xor_lane(__uint_as_float(sent), PAGED_ATTN_WMMA_ROW_THREADS));
        const uint32_t even_row = odd ? other : mine;
        const uint32_t odd_row  = odd ? mine : other;
        const half2 lo = *reinterpret_cast<const half2 *>(&even_row);
        const half2 hi = *reinterpret_cast<const half2 *>(&odd_row);
        col[(2*m    ) * stride_tile] = __lows2half2(lo, hi);
        col[(2*m + 1) * stride_tile] = __highs2half2(lo, hi);
    }
}
