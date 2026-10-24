// Device helpers shared by the MOSS-Image kernels.
//
// The kernels target RDNA4 (gfx12) WMMA. Their bodies compile under MOSS_GFX12; the host pass sees a trap instead,
// and the server checks the device (device_supported in kernels.h) before it loads the model.

#pragma once

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#if defined(__gfx1200__) || defined(__gfx1201__)
#define MOSS_GFX12 1
#endif

namespace luce::moss_image {

constexpr int kBlock = 256;  // threads of the elementwise kernels

static inline unsigned elementwise_blocks(const int64_t n) {
    return (unsigned) ((n + kBlock - 1) / kBlock);
}

// The 32-lane butterfly sum and max (offsets 16 .. 1): every lane ends with the same value.
static __device__ __forceinline__ float wave_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor(x, offset, 32);
    }
    return x;
}

static __device__ __forceinline__ float wave_max(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor(x, offset, 32));
    }
    return x;
}

// ggml's block_reduce over block_size threads: the wave sums, then every wave adds the block's wave sums in lanes
// 0 .. block_size / 32 - 1 (0 in the other lanes). partial holds 32 floats; a second reduction through it needs a
// barrier first.
template <int block_size>
static __device__ __forceinline__ float block_sum(float x, float * partial) {
    x = wave_sum(x);
    if constexpr (block_size > 32) {
        if (threadIdx.x % 32 == 0) {
            partial[threadIdx.x / 32] = x;
        }
        __syncthreads();
        x = threadIdx.x % 32 < block_size / 32 ? partial[threadIdx.x % 32] : 0.0f;
        x = wave_sum(x);
    }
    return x;
}

template <int block_size>
static __device__ __forceinline__ float block_max(float x, float * partial) {
    x = wave_max(x);
    if constexpr (block_size > 32) {
        if (threadIdx.x % 32 == 0) {
            partial[threadIdx.x / 32] = x;
        }
        __syncthreads();
        x = threadIdx.x % 32 < block_size / 32 ? partial[threadIdx.x % 32] : -INFINITY;
        x = wave_max(x);
    }
    return x;
}

static __device__ __forceinline__ float bf16_to_f32(const uint16_t b) {
    return __uint_as_float((uint32_t) b << 16);
}

// Round to nearest even, as __float2bfloat16.
static __device__ __forceinline__ uint16_t to_bf16(const float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) != 0x7f800000u) {
        u += 0x7fffu + ((u >> 16) & 1u);
    } else if (u & 0xffffu) {
        u |= 0x10000u;
    }
    return (uint16_t) (u >> 16);
}

// Round to nearest even, as __float2half.
static __device__ __forceinline__ uint16_t to_f16(const float f) {
    return __builtin_bit_cast(uint16_t, (_Float16) f);
}

// MOSS-Image's FFN gate, cap_g tanh(gate / cap_g) sigmoid(gate) cap_u tanh(up / cap_u), with ggml's op order.
static __device__ __forceinline__ float situ(const float gate, const float up, const float gate_cap, const float up_cap) {
    const float a = tanhf(gate * (1.0f / gate_cap)) * gate_cap * (1.0f / (1.0f + expf(-gate)));
    return a * (tanhf(up * (1.0f / up_cap)) * up_cap);
}

} // namespace luce::moss_image
