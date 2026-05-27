// HIP compatibility shim for <cuda_bf16.h>
// Used by the ggml-hip target (vendored llama.cpp) when compiling files that
// include <cuda_bf16.h> directly (e.g. fattn-sparse.cu). Mirrors
// dflash/hip_compat/cuda_bf16.h but with two ROCm-7.2-friendly tweaks:
//   1. hipcc path delegates to <hip/hip_bf16.h> (native __float2bfloat16/_rn
//      and __bfloat162float on ROCm 7.2+).  Redefining them here would
//      conflict with the native overloads that return __hip_bfloat16.
//   2. __nv_bfloat16 aliases to __hip_bfloat16 (new type) instead of the
//      legacy hip_bfloat16 struct, so casts to/from the native helpers match.
#pragma once

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#  define __HIP_PLATFORM_AMD__
#endif

#include <hip/hip_bfloat16.h>
#include <cstring>

using __nv_bfloat16 = __hip_bfloat16;

#ifdef __HIPCC__
#  include <hip/hip_bf16.h>
#else
#include <cstdint>
namespace __hip_bf16_compat_detail {
    inline uint16_t float_to_bf16_bits_trunc(float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        return static_cast<uint16_t>(u >> 16);
    }
    inline uint16_t float_to_bf16_bits_rne(float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu))
            return static_cast<uint16_t>(u >> 16);
        uint32_t lsb = (u >> 16) & 1u;
        return static_cast<uint16_t>((u + 0x7fffu + lsb) >> 16);
    }
    inline float bf16_bits_to_float(uint16_t b) {
        uint32_t u = static_cast<uint32_t>(b) << 16;
        float f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    }
}
inline float __bfloat162float(hip_bfloat16 x) {
    return __hip_bf16_compat_detail::bf16_bits_to_float(x.data);
}
inline hip_bfloat16 __float2bfloat16(float x) {
    hip_bfloat16 r;
    r.data = __hip_bf16_compat_detail::float_to_bf16_bits_trunc(x);
    return r;
}
inline hip_bfloat16 __float2bfloat16_rn(float x) {
    hip_bfloat16 r;
    r.data = __hip_bf16_compat_detail::float_to_bf16_bits_rne(x);
    return r;
}
#endif
