#pragma once

#include "ggml-common.h"
#include "convert.cuh"
#include "tq3-quant.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

static __device__ void quantize_f32_tq3_0_group(const float * __restrict__ src, block_tq3_0 * __restrict__ dst) {
    float x[128];
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        x[j] = src[j];
        norm_sq += x[j] * x[j];
    }

    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < 128; j++) x[j] *= inv_norm;

    tq3_rotate_forward(x);

    float recon_norm_sq = 0.0f;
    for (int b = 0; b < 4; b++) {
        const int off = b * QK_TQ3_0;
        for (int j = 0; j < QK_TQ3_0 / 4; j++) dst[b].qs[j] = 0;
        for (int j = 0; j < QK_TQ3_0 / 8; j++) dst[b].signs[j] = 0;
        for (int j = 0; j < QK_TQ3_0; j++) {
            uint8_t idx = tq3_find_nearest(x[off + j]);
            dst[b].qs[j/4] |= (idx & 0x3) << ((j%4) * 2);
            if (idx & 0x4) dst[b].signs[j/8] |= (1 << (j%8));
            float c = d_tq3_centroids[idx];
            recon_norm_sq += c * c;
        }
    }

    float recon_norm = sqrtf(recon_norm_sq);
    float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    half h_norm = __float2half(corrected_norm);
    for (int b = 0; b < 4; b++) dst[b].norm = h_norm;
}

// Warp-cooperative replacement for quantize_f32_tq3_0_group.
//
// Each warp (32 threads) quantizes one 128-element group. The caller must
// launch with blockDim.x == 32 and one warp per group.
//
// Each lane holds 4 consecutive elements at indices [lane*4 .. lane*4+3];
// reductions use __shfl_xor_sync, the FWHT reuses warp_tq3_rotate_forward
// (already warp-cooperative), and the per-byte packing into qs/signs is
// conflict-free: qs is 4-elements-per-byte so each lane writes its own
// byte, signs is 8-elements-per-byte so paired lanes shfl-OR before the
// even lane in each pair writes.
//
// vs the single-thread version this replaces:
//   - 32x more parallel work per group (full warp instead of one thread)
//   - no float x[128] stack array (was spilling to local memory)
//   - reuses the already-warp-cooperative warp_tq3_rotate_forward
static __device__ __forceinline__ void warp_quantize_f32_tq3_0_group(
        const float * __restrict__ src,
        block_tq3_0 * __restrict__ dst) {
    const int lane = threadIdx.x & 31;

    float v0 = src[lane*4 + 0];
    float v1 = src[lane*4 + 1];
    float v2 = src[lane*4 + 2];
    float v3 = src[lane*4 + 3];

    // Group L2 norm via warp reduction.
    float norm_sq = v0*v0 + v1*v1 + v2*v2 + v3*v3;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        norm_sq += __shfl_xor_sync(0xFFFFFFFF, norm_sq, mask);
    }
    const float grp_norm = sqrtf(norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;
    v0 *= inv_norm; v1 *= inv_norm; v2 *= inv_norm; v3 *= inv_norm;

    // Forward FWHT rotation (already warp-cooperative).
    warp_tq3_rotate_forward(v0, v1, v2, v3);

    // Quantize each value to a 3-bit centroid index.
    const uint8_t idx0 = tq3_find_nearest(v0);
    const uint8_t idx1 = tq3_find_nearest(v1);
    const uint8_t idx2 = tq3_find_nearest(v2);
    const uint8_t idx3 = tq3_find_nearest(v3);

    // Reconstruction norm for the per-block correction factor.
    const float c0 = d_tq3_centroids[idx0];
    const float c1 = d_tq3_centroids[idx1];
    const float c2 = d_tq3_centroids[idx2];
    const float c3 = d_tq3_centroids[idx3];
    float recon_sq = c0*c0 + c1*c1 + c2*c2 + c3*c3;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        recon_sq += __shfl_xor_sync(0xFFFFFFFF, recon_sq, mask);
    }
    const float recon_norm = sqrtf(recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    const half h_norm = __float2half(corrected_norm);

    // Pack into block_tq3_0 layout.
    //   block index b   = lane / 8   (4 blocks of 32 elements per group)
    //   t_in_block      = lane & 7   (8 lanes per block)
    //   qs[t_in_block]      holds elements [4*t_in_block .. 4*t_in_block+3]
    //   signs[t_in_block/2] holds 8 elements; bits (t_in_block&1)*4 .. +3
    const int b          = lane >> 3;
    const int t_in_block = lane & 7;

    const uint8_t my_qs =
          (uint8_t)((idx0 & 0x3) << 0)
        | (uint8_t)((idx1 & 0x3) << 2)
        | (uint8_t)((idx2 & 0x3) << 4)
        | (uint8_t)((idx3 & 0x3) << 6);

    const int sign_bit_off = (t_in_block & 1) * 4;
    const uint8_t my_signs_partial =
          (uint8_t)(((idx0 >> 2) & 0x1) << (sign_bit_off + 0))
        | (uint8_t)(((idx1 >> 2) & 0x1) << (sign_bit_off + 1))
        | (uint8_t)(((idx2 >> 2) & 0x1) << (sign_bit_off + 2))
        | (uint8_t)(((idx3 >> 2) & 0x1) << (sign_bit_off + 3));

    // Combine the two halves of each signs byte across paired lanes.
    const uint8_t partner = (uint8_t)__shfl_xor_sync(0xFFFFFFFF, my_signs_partial, 1);
    const uint8_t my_signs_byte = my_signs_partial | partner;

    // Each lane writes its own qs byte (no conflict).
    dst[b].qs[t_in_block] = my_qs;

    // Only the even lane of each pair writes the signs byte.
    if ((t_in_block & 1) == 0) {
        dst[b].signs[t_in_block >> 1] = my_signs_byte;
    }

    // Only the first lane of each block writes the norm.
    if (t_in_block == 0) {
        dst[b].norm = h_norm;
    }
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
