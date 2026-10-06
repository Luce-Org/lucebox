#pragma once
// Device helpers shared by the qwen4exp fused kernels (HC, MMB, PLE/GDN conv, gated norm). Each fusion replaces a
// sequence of separate ops bit for bit, so it rounds where those ops round.

#include "common.cuh"

// One F32 multiply / add the compiler cannot contract into an FMA (the separate MUL and ADD ops round each step).
#if defined(__HIP_PLATFORM_AMD__)
static __device__ __forceinline__ float q4x_mul_rn(const float a, const float b) {
    float result;
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}

static __device__ __forceinline__ float q4x_add_rn(const float a, const float b) {
    float result;
    asm("v_add_f32_e32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}
#else
static __device__ __forceinline__ float q4x_mul_rn(const float a, const float b) {
    return __fmul_rn(a, b);
}

static __device__ __forceinline__ float q4x_add_rn(const float a, const float b) {
    return __fadd_rn(a, b);
}
#endif

// same expression as op_sigmoid in unary.cu
static __device__ __forceinline__ float q4x_sigmoid(const float x) {
    return 1.0f / (1.0f + expf(-x));
}

// BF16 bit patterns: exact widening, round-to-nearest-even narrowing.
static __device__ __forceinline__ float q4x_bf2f(const uint16_t h) { return __uint_as_float(((uint32_t) h) << 16); }
static __device__ __forceinline__ uint16_t q4x_f2bf(const float f) { uint32_t u = __float_as_uint(f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
static __device__ __forceinline__ uint32_t q4x_pack2(const float a, const float b) {
    return (uint32_t) q4x_f2bf(a) | ((uint32_t) q4x_f2bf(b) << 16);
}

// Depthwise-conv input concat [state (H columns) | x (T columns)] per channel: preserve the old state in the
// otherwise-unused concat columns, then materialize only the tail from tail_from that the state checkpoint reads.
template <int H>
static __global__ void q4x_conv_concat_tail(const float * __restrict__ state, const float * __restrict__ x, float * __restrict__ out,
                                            const int C, const int T, const int tail_from, const int row_stride) {
    const int ncols = T + H - tail_from;
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t) C * (H + ncols)) return;
    const int c = (int) (idx / (H + ncols)), col = (int) (idx % (H + ncols));
    if (col < H) {
        out[(size_t) c * row_stride + col] = state[c * H + col];
    } else {
        const int j = tail_from + col - H;
        out[(size_t) c * row_stride + j] = (j < H) ? state[c * H + j] : x[(size_t) (j - H) * C + c];
    }
}
