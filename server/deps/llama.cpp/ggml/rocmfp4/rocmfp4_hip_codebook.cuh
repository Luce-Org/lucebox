#pragma once

#include "rocmfp4_hip_scale.cuh"

#include <cstdint>
#include <cstring>

#ifndef GGML_ROCMFP4_UNALIGNED_QS_DWORD_LOAD
#define GGML_ROCMFP4_UNALIGNED_QS_DWORD_LOAD 1
#endif

#if !defined(GGML_USE_HIP)
// Self-contained 16-entry table expander for the non-HIP fallback. This header
// is pulled into translation units that do not include vecdotq.cuh (where the
// generic get_int_from_table_16 lives) - e.g. fattn-chunked.cu - so relying on
// that symbol breaks the CUDA build on include order. This is the generic
// branch of get_int_from_table_16 verbatim, so results are bit-identical; only
// the HIP path (Strix) uses __builtin_amdgcn_perm and never reaches here.
static __device__ __forceinline__ int2 rocmfp4_table16_fallback(const int & q4, const int8_t * table) {
    const int      q0_32  = (q4 >> 0) & 0x0F0F0F0F;
    const int8_t * q0_8   = (const int8_t *) &q0_32;
    const char4    val0_8 = make_char4(
        table[q0_8[0]], table[q0_8[1]], table[q0_8[2]], table[q0_8[3]]);

    const int      q1_32  = (q4 >> 4) & 0x0F0F0F0F;
    const int8_t * q1_8   = (const int8_t *) &q1_32;
    const char4    val1_8 = make_char4(
        table[q1_8[0]], table[q1_8[1]], table[q1_8[2]], table[q1_8[3]]);

    return make_int2(*((const int *) &val0_8), *((const int *) &val1_8));
}
#endif

static __device__ __forceinline__ int rocmfp4_get_qs_i32(const void * x, const int & i32) {
#if defined(GGML_USE_HIP) && GGML_ROCMFP4_UNALIGNED_QS_DWORD_LOAD
    return *((const int *) ((const uint8_t *) x + 4*i32));
#else
    const uint8_t * x8 = (const uint8_t *) x;

    int x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;

    return x32;
#endif
}

// AMD-specific fast path for expanding eight packed ROCmFP4 nibbles into two
// int32 DP4A operands. This encodes the Codebook10 table directly as four
// 32-bit constants:
//   [0, 1, 2, 3], [4, 6, 8, 10], [0, -1, -2, -3], [-4, -6, -8, -10]
// Avoiding the table pointer keeps the ROCm/HIP MMVQ/MMQ hot path fully local
// to this format. Non-HIP builds still use llama.cpp's generic table expander.
static __device__ __forceinline__ int2 rocmfp4_get_int_from_codebook_16(const int & q4, const int8_t * fallback_table) {
#if defined(GGML_USE_HIP)
    constexpr uint32_t values0 = 0x03020100u;
    constexpr uint32_t values1 = 0x0a080604u;
    constexpr uint32_t values2 = 0xfdfeff00u;
    constexpr uint32_t values3 = 0xf6f8fafcu;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;

    const uint32_t v_even_low  = __builtin_amdgcn_perm(values1, values0, q_even & 0x07070707u);
    const uint32_t v_odd_low   = __builtin_amdgcn_perm(values1, values0, q_odd  & 0x07070707u);
    const uint32_t v_even_high = __builtin_amdgcn_perm(values3, values2, q_even & 0x07070707u);
    const uint32_t v_odd_high  = __builtin_amdgcn_perm(values3, values2, q_odd  & 0x07070707u);

    const uint32_t mask_even = 0x03020100u | ((q_even & 0x08080808u) >> 1);
    const uint32_t mask_odd  = 0x03020100u | ((q_odd  & 0x08080808u) >> 1);

    return make_int2(
        __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even),
        __builtin_amdgcn_perm(v_odd_high,  v_odd_low,  mask_odd));
#else
    return rocmfp4_table16_fallback(q4, fallback_table);
#endif
}

// Variant for call sites that already selected either the low or high nibble
// stream and only need one DP4A operand. This avoids the extra odd/even table
// expansion work in ROCmFP4 FlashAttention K/V decode.
static __device__ __forceinline__ int rocmfp4_get_low_int_from_codebook_16(const int & q4, const int8_t * fallback_table) {
#if defined(GGML_USE_HIP)
    constexpr uint32_t values0 = 0x03020100u;
    constexpr uint32_t values1 = 0x0a080604u;
    constexpr uint32_t values2 = 0xfdfeff00u;
    constexpr uint32_t values3 = 0xf6f8fafcu;

    const uint32_t q = q4;

    const uint32_t v_low  = __builtin_amdgcn_perm(values1, values0, q & 0x07070707u);
    const uint32_t v_high = __builtin_amdgcn_perm(values3, values2, q & 0x07070707u);
    const uint32_t mask   = 0x03020100u | ((q & 0x08080808u) >> 1);

    return __builtin_amdgcn_perm(v_high, v_low, mask);
#else
    return rocmfp4_table16_fallback(q4, fallback_table).x;
#endif
}
