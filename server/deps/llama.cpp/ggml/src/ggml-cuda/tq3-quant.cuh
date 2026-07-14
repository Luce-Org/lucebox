#pragma once

#include "ggml-common.h"

// TQ3_0: 3-bit Lloyd-Max codebook with FWHT rotation.
// Block size = 32 elements (14 bytes). Group size = 128 elements (4 blocks share norm + rotation).

static __constant__ float d_tq3_centroids[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

static __constant__ float d_tq3_mids[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f
};

static __constant__ float d_tq3_wht_signs1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};

static __constant__ float d_tq3_wht_signs2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f};

// ── Warp-cooperative FWHT (128 elements, 32 threads × 4 values) ─────
// Replaces single-thread tq3_fwht_128 which spills 128 registers.
// Each thread in a 32-thread warp holds 4 contiguous elements.
// Stages 0-1: local butterfly (within thread's 4 values)
// Stages 2-6: __shfl_xor_sync (across threads in warp)

static __device__ __forceinline__
void warp_fwht_128(float &v0, float &v1, float &v2, float &v3) {
    // Stage 0: stride=1
    {
        float a, b;
        a = v0; b = v1; v0 = a + b; v1 = a - b;
        a = v2; b = v3; v2 = a + b; v3 = a - b;
    }
    // Stage 1: stride=2
    {
        float a, b;
        a = v0; b = v2; v0 = a + b; v2 = a - b;
        a = v1; b = v3; v1 = a + b; v3 = a - b;
    }
    // Stages 2-6: stride=4,8,16,32,64 → lane_mask=1,2,4,8,16
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int lane_mask = 1; lane_mask <= 16; lane_mask <<= 1) {
        float s0 = __shfl_xor_sync(0xFFFFFFFF, v0, lane_mask);
        float s1 = __shfl_xor_sync(0xFFFFFFFF, v1, lane_mask);
        float s2 = __shfl_xor_sync(0xFFFFFFFF, v2, lane_mask);
        float s3 = __shfl_xor_sync(0xFFFFFFFF, v3, lane_mask);
        if ((lane & lane_mask) == 0) {
            v0 = v0 + s0; v1 = v1 + s1; v2 = v2 + s2; v3 = v3 + s3;
        } else {
            v0 = s0 - v0; v1 = s1 - v1; v2 = s2 - v2; v3 = s3 - v3;
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    v0 *= inv_sqrt_128; v1 *= inv_sqrt_128; v2 *= inv_sqrt_128; v3 *= inv_sqrt_128;
}

// Apply element-wise sign pattern. Thread t (within warp) applies signs at [4t..4t+3].
static __device__ __forceinline__
void warp_apply_signs(float &v0, float &v1, float &v2, float &v3,
                      const float * __restrict__ signs) {
    const int base = (threadIdx.x & 31) * 4;
    v0 *= signs[base + 0]; v1 *= signs[base + 1];
    v2 *= signs[base + 2]; v3 *= signs[base + 3];
}

// Forward rotation: signs1 → FWHT → signs2
static __device__ __forceinline__
void warp_tq3_rotate_forward(float &v0, float &v1, float &v2, float &v3) {
    warp_apply_signs(v0, v1, v2, v3, d_tq3_wht_signs1);
    warp_fwht_128(v0, v1, v2, v3);
    warp_apply_signs(v0, v1, v2, v3, d_tq3_wht_signs2);
}

// Inverse rotation: signs2 → FWHT → signs1
static __device__ __forceinline__
void warp_tq3_rotate_inverse(float &v0, float &v1, float &v2, float &v3) {
    warp_apply_signs(v0, v1, v2, v3, d_tq3_wht_signs2);
    warp_fwht_128(v0, v1, v2, v3);
    warp_apply_signs(v0, v1, v2, v3, d_tq3_wht_signs1);
}

static __device__ __forceinline__
void tq3_fwht_128(float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt_128;
}

static __device__ __forceinline__
void tq3_rotate_forward(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= d_tq3_wht_signs1[i];
    tq3_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= d_tq3_wht_signs2[i];
}

static __device__ __forceinline__
void tq3_rotate_inverse(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= d_tq3_wht_signs2[i];
    tq3_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= d_tq3_wht_signs1[i];
}

static __device__ __forceinline__
uint8_t tq3_find_nearest(float val) {
    if      (val < d_tq3_mids[0]) return 0;
    else if (val < d_tq3_mids[1]) return 1;
    else if (val < d_tq3_mids[2]) return 2;
    else if (val < d_tq3_mids[3]) return 3;
    else if (val < d_tq3_mids[4]) return 4;
    else if (val < d_tq3_mids[5]) return 5;
    else if (val < d_tq3_mids[6]) return 6;
    else                          return 7;
}

static __device__ __forceinline__
void dequantize_tq3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_tq3_0 * x = (const block_tq3_0 *)vx;
    const float norm = __half2float(x[ib].norm);
    {
        const int j = iqs;
        const uint8_t low2 = (x[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
        const uint8_t hi1  = (x[ib].signs[j/8] >> (j%8)) & 0x1;
        v.x = d_tq3_centroids[low2 | (hi1 << 2)] * norm;
    }
    {
        const int j = iqs + QK_TQ3_0/2;
        const uint8_t low2 = (x[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
        const uint8_t hi1  = (x[ib].signs[j/8] >> (j%8)) & 0x1;
        v.y = d_tq3_centroids[low2 | (hi1 << 2)] * norm;
    }
}
