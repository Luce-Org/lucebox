#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

static __device__ __forceinline__ float rocmfp4_u32_as_f32(uint32_t bits) {
#if defined(GGML_USE_HIP)
    return __uint_as_float(bits);
#else
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
#endif
}

// ROCmFP4 validates scale bytes before backend execution, so HIP/ROCm hot
// paths can decode finite unsigned E4M3 half-scales directly without the
// generic FP8 NaN handling used by other formats.
static __device__ __forceinline__ float rocmfp4_ue4m3_to_fp32_half_finite(uint8_t x) {
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;

    if (exp == 0) {
        return (float) man * (1.0f / 1024.0f);
    }

    const uint32_t bits = ((uint32_t) exp + 119u) << 23 | ((uint32_t) man << 20);
    return rocmfp4_u32_as_f32(bits);
}

static __device__ __forceinline__ float rocmfpx_ue4m3_to_fp32_finite(uint8_t x) {
    if (x > 0x7e) {
        return 0.0f;
    }

    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;

    if (exp == 0) {
        return (float) man * (1.0f / 1024.0f);
    }

    const uint32_t bits = ((uint32_t) exp + 119u) << 23 | ((uint32_t) man << 20);
    return rocmfp4_u32_as_f32(bits);
}

static __device__ __forceinline__ uint8_t rocmfpx_nearest_scale_ue4m3_cuda(float target_scale) {
    if (!(target_scale > 0.0f) || !isfinite(target_scale)) {
        return 0;
    }

    uint8_t lo = 1;
    uint8_t hi = 0x7e;
    while (lo < hi) {
        const uint8_t mid = lo + (hi - lo) / 2;
        if (rocmfpx_ue4m3_to_fp32_finite(mid) < target_scale) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if (lo == 1) {
        return 1;
    }

    const float hi_scale = rocmfpx_ue4m3_to_fp32_finite(lo);
    const float lo_scale = rocmfpx_ue4m3_to_fp32_finite((uint8_t) (lo - 1));
    return (target_scale - lo_scale <= hi_scale - target_scale) ? (uint8_t) (lo - 1) : lo;
}

static __device__ __forceinline__ int8_t rocmfp4_decode_i8(uint8_t q) {
    q &= 0x0f;
    const int mag3 = q & 0x07;
    const int mag = mag3 <= 4 ? mag3 : 2*mag3 - 4;
    return (q & 0x08) ? -mag : mag;
}
