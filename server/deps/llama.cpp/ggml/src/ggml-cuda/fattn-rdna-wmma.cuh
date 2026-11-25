#pragma once
// Native zero-spill RDNA3.5 (gfx1151) & RDNA4 (gfx1201) Wave32 WMMA FlashAttention-2 + FlashDecoding kernel (v4).
// - D=128: NWARPS=8, BM=128, BN=32, Q_reg[8] + O[8] in VGPRs (ZERO Q_lds, ZERO spill on gfx1201, 8.4 KB LDS).
// - D=256: Warp-Split-D! NWARPS=8 (4 warp-pairs, BM=64), BN=32, each warp owns half of D (128 dims):
//          Q_reg[8] (32 VGPRs) + O[8] (64 VGPRs) -> ZERO spill on gfx1201, 16.6 KB LDS.
// - GQA Head-Packing: packs (q_tok, head_in_gqa_group) into BM rows sharing ONE K/V load (up to 6x-8x less KV BW!).
// - FlashDecoding Split-KV: automatically splits S_kv across blocks when nq is small (DFlash2/MTP verify nq=4..32),
//   saturating all CUs on both R9700 & Strix Halo with a fast log-sum-exp combine kernel.
// - XOR-swizzled bank-conflict-free K_lds & V_lds with 4-row uint32-packed ds_write_b64 V-transpose.
// - Hardware exp2f (v_exp_f32) online softmax + zero-mask fast path skipping mask loads on ~99% of active tiles.

#include "common.cuh"
#include "fattn-common.cuh"
#include <cstdlib>
#include <cstring>
#include <algorithm>

#if defined(GGML_USE_HIP)

extern "C" void ggml_backend_cuda_record_fattn_mma256_launch(void);

namespace {

using fa_f16x8_t  = __attribute__((ext_vector_type(8)))  _Float16;
using fa_f16x16_t = __attribute__((ext_vector_type(16))) _Float16;
using fa_f32x8_t  = __attribute__((ext_vector_type(8)))  float;

static constexpr float FA_LOG2E = 1.44269504088896340736f;

__device__ __forceinline__ int fa_k_swiz_chunk(const int ki, const int chunk16) {
    return chunk16 ^ (ki & 7);
}

template <int BN>
__device__ __forceinline__ int fa_v_swiz_chunk(const int d, const int chunk16) {
    if constexpr (BN == 64) {
        return chunk16 ^ (d & 7) ^ ((d >> 3) & 7);
    } else {
        return chunk16 ^ ((d >> 1) & 3) ^ ((d >> 3) & 3);
    }
}

__device__ __forceinline__ uint32_t fa_pack_f16x2(const float a, const float b) {
    const __half2 h2 = __floats2half2_rn(a, b);
    return __builtin_bit_cast(uint32_t, h2);
}

__device__ __forceinline__ uint4 fa_dequant_q8_0_quad(const block_q8_0 * bp, const int sub) {
    const float d = __half2float(bp->d);
    uint64_t q64;
#ifdef __HIP_DEVICE_COMPILE__
    __builtin_memcpy(&q64, bp->qs + sub * 8, sizeof(q64));
#else
    std::memcpy(&q64, bp->qs + sub * 8, sizeof(q64));
#endif
    const uint32_t lo = (uint32_t) q64;
    const uint32_t hi = (uint32_t) (q64 >> 32);
    const float f0 = (float)(int8_t)(lo & 0xFFu),         f1 = (float)(int8_t)((lo >>  8) & 0xFFu);
    const float f2 = (float)(int8_t)((lo >> 16) & 0xFFu), f3 = (float)(int8_t)(lo >> 24);
    const float f4 = (float)(int8_t)(hi & 0xFFu),         f5 = (float)(int8_t)((hi >>  8) & 0xFFu);
    const float f6 = (float)(int8_t)((hi >> 16) & 0xFFu), f7 = (float)(int8_t)(hi >> 24);
    return make_uint4(
        fa_pack_f16x2(d * f0, d * f1),
        fa_pack_f16x2(d * f2, d * f3),
        fa_pack_f16x2(d * f4, d * f5),
        fa_pack_f16x2(d * f6, d * f7));
}

template <int BN>
__device__ __forceinline__ void fa_store_v_4rows(
        _Float16 * KV_lds, const int d0, const int g,
        const uint4 r0, const uint4 r1, const uint4 r2, const uint4 r3) {
    const int ch16 = g >> 1;
    const int sub4 = (g & 1) * 4;

    auto write_pair = [&](const int u_base, const uint32_t w0, const uint32_t w1, const uint32_t w2, const uint32_t w3) {
        const uint2 lo = make_uint2(
            (w0 & 0xFFFFu) | (w1 << 16),
            (w2 & 0xFFFFu) | (w3 << 16));
        const uint2 hi = make_uint2(
            (w0 >> 16) | (w1 & 0xFFFF0000u),
            (w2 >> 16) | (w3 & 0xFFFF0000u));
        const int d_lo = d0 + u_base;
        const int d_hi = d0 + u_base + 1;
        *reinterpret_cast<uint2 *>(KV_lds + d_lo * BN + fa_v_swiz_chunk<BN>(d_lo, ch16) * 8 + sub4) = lo;
        *reinterpret_cast<uint2 *>(KV_lds + d_hi * BN + fa_v_swiz_chunk<BN>(d_hi, ch16) * 8 + sub4) = hi;
    };

    write_pair(0, r0.x, r1.x, r2.x, r3.x);
    write_pair(2, r0.y, r1.y, r2.y, r3.y);
    write_pair(4, r0.z, r1.z, r2.z, r3.z);
    write_pair(6, r0.w, r1.w, r2.w, r3.w);
}

// SPLIT_D = 1 for D=128 (BM=128, 8 warps), SPLIT_D = 2 for D=256 (BM=64, 4 warp-pairs x 2 D-halves).
// When num_splits == 1: writes normalized output directly to dst_ptr (ml_part == nullptr).
// When num_splits > 1:  writes unnormalized partial O to dst_ptr and (m_max, l_sum) to ml_part.
template <int D, int BN, int SPLIT_D, bool KV_Q8_0, bool USE_SOFTCAP>
__global__ void __launch_bounds__(256, 1)
fattn_rdna_wmma_kernel(
        const char * __restrict__ Q_ptr,
        const char * __restrict__ K_ptr,
        const char * __restrict__ V_ptr,
        const char * __restrict__ mask_ptr,
        const float * __restrict__ sinks_ptr,
        float * __restrict__ dst_ptr,
        float2 * __restrict__ ml_part,
        const float scale,
        const float logit_softcap,
        const int nq, const int Hq, const int Hk, const int S_kv,
        const int num_splits,
        const int nb01, const int nb02, const int nb03,
        const int nb11, const int nb12, const int64_t nb13,
        const int nb21, const int nb22, const int64_t nb23,
        const int ne32, const int ne33,
        const int nb31, const int nb32, const int64_t nb33) {
#if defined(RDNA4) || defined(RDNA3)
    constexpr int NWARPS = 8;
    constexpr int BM     = (NWARPS / SPLIT_D) * 16; // 128 (D=128) or 64 (D=256)
    constexpr int D_SUB  = D / SPLIT_D;             // ALWAYS 128!
    constexpr int ND_SUB = D_SUB / 16;              // ALWAYS 8!
    constexpr int NT     = BN / 16;                 // 2 (BN=32)

    __shared__ __align__(16) _Float16 KV_lds[BN * D];

    const int lane = threadIdx.x;       // 0..31
    const int wave = threadIdx.y;       // 0..7
    const int tid  = wave * 32 + lane;  // 0..255
    const int r    = lane & 15;         // 0..15
    const int hi   = lane >> 4;         // 0 or 1

    const int q_wave = (SPLIT_D == 1) ? wave : (wave >> 1);
    const int dh     = (SPLIT_D == 1) ? 0    : (wave & 1);
    const int d_base = dh * D_SUB;

    const int gqa      = Hq / Hk;
    const int total_q  = nq * gqa;
    const int q_base   = blockIdx.x * BM;
    const int kv_head  = blockIdx.y;
    const int batch    = (num_splits == 1) ? (int)blockIdx.z : ((int)blockIdx.z / num_splits);
    const int split_id = (num_splits == 1) ? 0               : ((int)blockIdx.z - batch * num_splits);

    const int kv_tiles        = (S_kv + BN - 1) / BN;
    const int tiles_per_split = (kv_tiles + num_splits - 1) / num_splits;
    const int kv_begin        = split_id * tiles_per_split * BN;
    const int kv_end          = min(S_kv, (split_id + 1) * tiles_per_split * BN);

    const char * Q_b  = Q_ptr + (size_t)batch * nb03;
    const char * K_hb = K_ptr + (size_t)batch * nb13 + (size_t)kv_head * nb12;
    const char * V_hb = V_ptr + (size_t)batch * nb23 + (size_t)kv_head * nb22;
    const char * M_b  = mask_ptr ? (mask_ptr + (size_t)(batch % ne33) * nb33) : nullptr;

    // GQA-packed row mapping: row_idx = q_tok * gqa + h_sub
    const int q_local   = q_wave * 16 + r;
    const int row_idx   = q_base + q_local;
    const bool valid_q  = (row_idx < total_q);
    const int q_tok     = valid_q ? (row_idx / gqa) : 0;
    const int head      = valid_q ? (kv_head * gqa + (row_idx - q_tok * gqa)) : 0;
    const half * mask_row = (M_b && valid_q)
        ? reinterpret_cast<const half *>(M_b + (size_t)(head % ne32) * nb32 + (size_t)q_tok * nb31)
        : nullptr;

    // 1. Load Q[head, q_tok, d_base..+127] directly into Q_reg[8] (32 VGPRs)
    const float q_scale = USE_SOFTCAP ? (scale / logit_softcap) : (scale * FA_LOG2E);
    const float softcap_log2e = logit_softcap * FA_LOG2E;

    fa_f16x8_t Q_reg[ND_SUB];
    if (valid_q) {
        const char * Q_row = Q_b + (size_t)head * nb02 + (size_t)q_tok * nb01 + d_base * 4;
#pragma unroll
        for (int kt = 0; kt < ND_SUB; ++kt) {
            const float4 f0 = *reinterpret_cast<const float4 *>(Q_row + (kt * 16 + hi * 8 + 0) * 4);
            const float4 f1 = *reinterpret_cast<const float4 *>(Q_row + (kt * 16 + hi * 8 + 4) * 4);
            const uint4 qu = make_uint4(
                fa_pack_f16x2(f0.x * q_scale, f0.y * q_scale),
                fa_pack_f16x2(f0.z * q_scale, f0.w * q_scale),
                fa_pack_f16x2(f1.x * q_scale, f1.y * q_scale),
                fa_pack_f16x2(f1.z * q_scale, f1.w * q_scale));
            Q_reg[kt] = __builtin_bit_cast(fa_f16x8_t, qu);
            __builtin_amdgcn_sched_barrier(0);
        }
    } else {
#pragma unroll
        for (int kt = 0; kt < ND_SUB; ++kt) {
            Q_reg[kt] = {(_Float16)0.f, (_Float16)0.f, (_Float16)0.f, (_Float16)0.f,
                         (_Float16)0.f, (_Float16)0.f, (_Float16)0.f, (_Float16)0.f};
        }
    }

    // Output accumulator O[8] in F32 registers (64 VGPRs)
    fa_f32x8_t O[ND_SUB];
#pragma unroll
    for (int dt = 0; dt < ND_SUB; ++dt) {
        O[dt] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }
    float m_max = -INFINITY;
    float l_sum = 0.0f;

    // Precompute mask-scan row info per thread (avoids integer division inside kv0 loop!)
    constexpr int thrs_per_row  = 256 / BM;          // 2 (BM=128) or 4 (BM=64)
    constexpr int halfs_per_thr = BN / thrs_per_row; // 16 (BM=128,BN=32) or 8 (BM=64,BN=32)
    const int scan_qi    = tid / thrs_per_row;
    const int scan_row   = q_base + scan_qi;
    const bool scan_ok   = (M_b != nullptr) && (scan_row < total_q);
    const int scan_qtok  = scan_ok ? (scan_row / gqa) : 0;
    const int scan_head  = scan_ok ? (kv_head * gqa + (scan_row - scan_qtok * gqa)) : 0;
    const int scan_koff  = (tid & (thrs_per_row - 1)) * halfs_per_thr;
    const half * scan_mp_base = scan_ok
        ? reinterpret_cast<const half *>(M_b + (size_t)(scan_head % ne32) * nb32 + (size_t)scan_qtok * nb31)
        : nullptr;

    // 2. Main loop over KV blocks of BN within [kv_begin, kv_end)
    for (int kv0 = kv_begin; kv0 < kv_end; kv0 += BN) {
        bool tile_has_nonzero_mask = false;
        if (M_b) {
            int local_finite  = 0;
            int local_nonzero = 0;
            if (scan_mp_base) {
                const half * mp = scan_mp_base + kv0 + scan_koff;
                if (kv0 + scan_koff + halfs_per_thr <= kv_end) {
                    uint32_t nz_acc = 0u;
#pragma unroll
                    for (int ch = 0; ch < halfs_per_thr / 8; ++ch) {
                        const uint4 v = *reinterpret_cast<const uint4 *>(mp + ch * 8);
                        nz_acc |= (v.x | v.y | v.z | v.w);
                        if (((v.x & 0x7C00u) != 0x7C00u) || (((v.x >> 16) & 0x7C00u) != 0x7C00u) ||
                            ((v.y & 0x7C00u) != 0x7C00u) || (((v.y >> 16) & 0x7C00u) != 0x7C00u) ||
                            ((v.z & 0x7C00u) != 0x7C00u) || (((v.z >> 16) & 0x7C00u) != 0x7C00u) ||
                            ((v.w & 0x7C00u) != 0x7C00u) || (((v.w >> 16) & 0x7C00u) != 0x7C00u)) {
                            local_finite = 1;
                        }
                    }
                    if ((nz_acc & 0x7FFF7FFFu) != 0u) local_nonzero = 1;
                } else {
                    for (int u = 0; u < halfs_per_thr && kv0 + scan_koff + u < kv_end; ++u) {
                        const uint16_t h = *reinterpret_cast<const uint16_t *>(mp + u);
                        if ((h & 0x7C00u) != 0x7C00u) local_finite = 1;
                        if ((h & 0x7FFFu) != 0u) local_nonzero = 1;
                    }
                }
            }
            if (!__syncthreads_or(local_finite)) {
                continue;
            }
            tile_has_nonzero_mask = (__syncthreads_or(local_nonzero) != 0);
        }

        // 2a. Load K[kv0 .. kv0+BN-1, 0..D-1] into XOR-swizzled KV_lds[BN][D]
        {
            constexpr int chunks_per_row = D / 8;
            constexpr int total_chunks = BN * chunks_per_row;
#pragma unroll 1
            for (int idx = tid; idx < total_chunks; idx += 256) {
                const int ki = idx / chunks_per_row;
                const int c  = idx - ki * chunks_per_row;
                const int kg = kv0 + ki;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (kg < kv_end) {
                    if constexpr (!KV_Q8_0) {
                        v = *reinterpret_cast<const uint4 *>(K_hb + (size_t)kg * nb11 + c * 16);
                    } else {
                        const block_q8_0 * bp = reinterpret_cast<const block_q8_0 *>(K_hb + (size_t)kg * nb11) + (c >> 2);
                        v = fa_dequant_q8_0_quad(bp, c & 3);
                    }
                }
                const int sc = fa_k_swiz_chunk(ki, c);
                *reinterpret_cast<uint4 *>(KV_lds + ki * D + sc * 8) = v;
            }
        }
        __syncthreads();

        // 2b. S = K[:, d_base..+127] * Q_reg^T
        fa_f32x8_t S[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            S[t] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }

        const int c_base = d_base >> 3;
#pragma unroll
        for (int kt = 0; kt < ND_SUB; ++kt) {
#if defined(RDNA4)
            const fa_f16x8_t qb = Q_reg[kt];
            const int c_raw = c_base + kt * 2 + hi;
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const int ki = t * 16 + r;
                const int sc = fa_k_swiz_chunk(ki, c_raw);
                const uint4 ku = *reinterpret_cast<const uint4 *>(KV_lds + ki * D + sc * 8);
                const fa_f16x8_t kv = __builtin_bit_cast(fa_f16x8_t, ku);
                S[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kv, qb, S[t]);
            }
            __builtin_amdgcn_sched_barrier(0);
#else
            const uint4 q_self = __builtin_bit_cast(uint4, Q_reg[kt]);
            const uint4 q_peer = make_uint4(
                __shfl_xor(q_self.x, 16, 32),
                __shfl_xor(q_self.y, 16, 32),
                __shfl_xor(q_self.z, 16, 32),
                __shfl_xor(q_self.w, 16, 32));
            const uint4 q0 = hi ? q_peer : q_self;
            const uint4 q1 = hi ? q_self : q_peer;
            const fa_f16x16_t qb = __builtin_bit_cast(fa_f16x16_t, (uint4[2]){q0, q1});
            const int c0_raw = c_base + kt * 2 + 0;
            const int c1_raw = c_base + kt * 2 + 1;
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const int ki = t * 16 + r;
                const int sc0 = fa_k_swiz_chunk(ki, c0_raw);
                const int sc1 = fa_k_swiz_chunk(ki, c1_raw);
                const uint4 ku0 = *reinterpret_cast<const uint4 *>(KV_lds + ki * D + sc0 * 8);
                const uint4 ku1 = *reinterpret_cast<const uint4 *>(KV_lds + ki * D + sc1 * 8);
                const fa_f16x16_t kv = __builtin_bit_cast(fa_f16x16_t, (uint4[2]){ku0, ku1});
                S[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kv, qb, S[t]);
            }
            __builtin_amdgcn_sched_barrier(0);
#endif
        }

        if constexpr (SPLIT_D == 2) {
            __syncthreads();
            float * S_red = reinterpret_cast<float *>(KV_lds);
            const int my_base   = (q_wave * 2 + dh) * (NT * 256);
            const int peer_base = (q_wave * 2 + (dh ^ 1)) * (NT * 256);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    S_red[my_base + (t * 8 + e) * 32 + lane] = S[t][e];
                }
            }
            __syncthreads();
#pragma unroll
            for (int t = 0; t < NT; ++t) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    S[t][e] += S_red[peer_base + (t * 8 + e) * 32 + lane];
                }
            }
        }

        if constexpr (USE_SOFTCAP) {
#pragma unroll
            for (int t = 0; t < NT; ++t) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    S[t][e] = softcap_log2e * tanhf(S[t][e]);
                }
            }
        }

        if (tile_has_nonzero_mask || (kv0 + BN > kv_end)) {
#pragma unroll
            for (int t = 0; t < NT; ++t) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
#if defined(RDNA4)
                    const int kv_i = kv0 + t * 16 + hi * 8 + e;
#else
                    const int kv_i = kv0 + t * 16 + 2 * e + hi;
#endif
                    if (kv_i >= kv_end) {
                        S[t][e] = -INFINITY;
                    } else if (tile_has_nonzero_mask && mask_row) {
                        S[t][e] += __half2float(mask_row[kv_i]) * FA_LOG2E;
                    }
                }
            }
        }

        float m_tile = m_max;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                m_tile = fmaxf(m_tile, S[t][e]);
            }
        }
        m_tile = fmaxf(m_tile, __shfl_xor(m_tile, 16, 32));

        const bool valid_tile = isfinite(m_tile);
        const float alpha = (valid_tile && isfinite(m_max)) ? exp2f(m_max - m_tile) : 0.0f;
        if (valid_tile) {
            m_max = m_tile;
            l_sum *= alpha;
#pragma unroll
            for (int dt = 0; dt < ND_SUB; ++dt) {
#pragma unroll
                for (int e = 0; e < 8; ++e) O[dt][e] *= alpha;
            }
        }

        fa_f16x8_t P[NT];
        float p_sum = 0.0f;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float pv[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                pv[e] = (valid_tile && isfinite(S[t][e])) ? exp2f(S[t][e] - m_max) : 0.0f;
                p_sum += pv[e];
            }
            const uint4 pu = make_uint4(
                fa_pack_f16x2(pv[0], pv[1]),
                fa_pack_f16x2(pv[2], pv[3]),
                fa_pack_f16x2(pv[4], pv[5]),
                fa_pack_f16x2(pv[6], pv[7]));
            P[t] = __builtin_bit_cast(fa_f16x8_t, pu);
        }
        p_sum += __shfl_xor(p_sum, 16, 32);
        l_sum += p_sum;

        __syncthreads();

        // 2c. Load V[kv0 .. kv0+BN-1, 0..D-1] TRANSPOSED into XOR-swizzled KV_lds[D][BN]
        {
            constexpr int chunks_per_row = D / 8;
            constexpr int row_groups = BN / 4;
            constexpr int total_4row_tasks = row_groups * chunks_per_row;
#pragma unroll 1
            for (int idx = tid; idx < total_4row_tasks; idx += 256) {
                const int g   = idx / chunks_per_row;
                const int c   = idx - g * chunks_per_row;
                const int ki0 = g * 4;
                const int d0  = c * 8;

                uint4 r0 = make_uint4(0u,0u,0u,0u), r1 = make_uint4(0u,0u,0u,0u);
                uint4 r2 = make_uint4(0u,0u,0u,0u), r3 = make_uint4(0u,0u,0u,0u);
                if constexpr (!KV_Q8_0) {
                    if (kv0 + ki0 + 0 < kv_end) r0 = *reinterpret_cast<const uint4 *>(V_hb + (size_t)(kv0 + ki0 + 0) * nb21 + c * 16);
                    if (kv0 + ki0 + 1 < kv_end) r1 = *reinterpret_cast<const uint4 *>(V_hb + (size_t)(kv0 + ki0 + 1) * nb21 + c * 16);
                    if (kv0 + ki0 + 2 < kv_end) r2 = *reinterpret_cast<const uint4 *>(V_hb + (size_t)(kv0 + ki0 + 2) * nb21 + c * 16);
                    if (kv0 + ki0 + 3 < kv_end) r3 = *reinterpret_cast<const uint4 *>(V_hb + (size_t)(kv0 + ki0 + 3) * nb21 + c * 16);
                } else {
                    const int b = c >> 2, sub = c & 3;
                    if (kv0 + ki0 + 0 < kv_end) r0 = fa_dequant_q8_0_quad(reinterpret_cast<const block_q8_0 *>(V_hb + (size_t)(kv0 + ki0 + 0) * nb21) + b, sub);
                    if (kv0 + ki0 + 1 < kv_end) r1 = fa_dequant_q8_0_quad(reinterpret_cast<const block_q8_0 *>(V_hb + (size_t)(kv0 + ki0 + 1) * nb21) + b, sub);
                    if (kv0 + ki0 + 2 < kv_end) r2 = fa_dequant_q8_0_quad(reinterpret_cast<const block_q8_0 *>(V_hb + (size_t)(kv0 + ki0 + 2) * nb21) + b, sub);
                    if (kv0 + ki0 + 3 < kv_end) r3 = fa_dequant_q8_0_quad(reinterpret_cast<const block_q8_0 *>(V_hb + (size_t)(kv0 + ki0 + 3) * nb21) + b, sub);
                }
                fa_store_v_4rows<BN>(KV_lds, d0, g, r0, r1, r2, r3);
            }
        }
        __syncthreads();

        // 2d. O += V^T[d_base..+127, :] * P
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#if defined(RDNA4)
            const fa_f16x8_t pt = P[t];
#pragma unroll
            for (int dt = 0; dt < ND_SUB; ++dt) {
                const int d = d_base + dt * 16 + r;
                const int sc = fa_v_swiz_chunk<BN>(d, t * 2 + hi);
                const uint4 vu = *reinterpret_cast<const uint4 *>(KV_lds + d * BN + sc * 8);
                const fa_f16x8_t vv = __builtin_bit_cast(fa_f16x8_t, vu);
                O[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vv, pt, O[dt]);
                __builtin_amdgcn_sched_barrier(0);
            }
#else
            const uint4 p_self = __builtin_bit_cast(uint4, P[t]);
            const uint4 p_peer = make_uint4(
                __shfl_xor(p_self.x, 16, 32),
                __shfl_xor(p_self.y, 16, 32),
                __shfl_xor(p_self.z, 16, 32),
                __shfl_xor(p_self.w, 16, 32));
            const uint4 lo_s = hi ? p_peer : p_self;
            const uint4 hi_s = hi ? p_self : p_peer;
            auto interleave_lo = [](uint32_t a, uint32_t b) -> uint32_t { return (a & 0xFFFFu) | (b << 16); };
            auto interleave_hi = [](uint32_t a, uint32_t b) -> uint32_t { return (a >> 16) | (b & 0xFFFF0000u); };
            const uint4 p0 = make_uint4(
                interleave_lo(lo_s.x, hi_s.x), interleave_hi(lo_s.x, hi_s.x),
                interleave_lo(lo_s.y, hi_s.y), interleave_hi(lo_s.y, hi_s.y));
            const uint4 p1 = make_uint4(
                interleave_lo(lo_s.z, hi_s.z), interleave_hi(lo_s.z, hi_s.z),
                interleave_lo(lo_s.w, hi_s.w), interleave_hi(lo_s.w, hi_s.w));
            const fa_f16x16_t pt = __builtin_bit_cast(fa_f16x16_t, (uint4[2]){p0, p1});
#pragma unroll
            for (int dt = 0; dt < ND_SUB; ++dt) {
                const int d = d_base + dt * 16 + r;
                const int sc0 = fa_v_swiz_chunk<BN>(d, t * 2 + 0);
                const int sc1 = fa_v_swiz_chunk<BN>(d, t * 2 + 1);
                const uint4 v0 = *reinterpret_cast<const uint4 *>(KV_lds + d * BN + sc0 * 8);
                const uint4 v1 = *reinterpret_cast<const uint4 *>(KV_lds + d * BN + sc1 * 8);
                const fa_f16x16_t vv = __builtin_bit_cast(fa_f16x16_t, (uint4[2]){v0, v1});
                O[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vv, pt, O[dt]);
                __builtin_amdgcn_sched_barrier(0);
            }
#endif
        }
        __syncthreads();
    }

    // 3. Epilogue: either normalize directly -> dst_ptr (num_splits==1) or write partial O + (m, l)
    if (valid_q) {
        if (ml_part == nullptr) {
            if (sinks_ptr) {
                const float sink_val = sinks_ptr[head] * FA_LOG2E;
                const float m_new = fmaxf(m_max, sink_val);
                const float scale_o = isfinite(m_max) ? exp2f(m_max - m_new) : 0.0f;
                l_sum = l_sum * scale_o + exp2f(sink_val - m_new);
#pragma unroll
                for (int dt = 0; dt < ND_SUB; ++dt) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) O[dt][e] *= scale_o;
                }
            }
            const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
            float * dst_row = dst_ptr + ((size_t)(batch * nq + q_tok) * Hq + head) * D + d_base;
#pragma unroll
            for (int dt = 0; dt < ND_SUB; ++dt) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
#if defined(RDNA4)
                    const int d = dt * 16 + hi * 8 + e;
#else
                    const int d = dt * 16 + 2 * e + hi;
#endif
                    dst_row[d] = O[dt][e] * inv_l;
                }
            }
        } else {
            const size_t slot = ((size_t)(batch * num_splits + split_id) * nq + q_tok) * Hq + head;
            float * out_row = dst_ptr + slot * D + d_base;
#pragma unroll
            for (int dt = 0; dt < ND_SUB; ++dt) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
#if defined(RDNA4)
                    const int d = dt * 16 + hi * 8 + e;
#else
                    const int d = dt * 16 + 2 * e + hi;
#endif
                    out_row[d] = O[dt][e];
                }
            }
            if (dh == 0 && hi == 0) {
                ml_part[slot] = make_float2(m_max, l_sum);
            }
        }
    }
#endif // defined(RDNA4) || defined(RDNA3)
}

template <int D>
__global__ void fattn_rdna_splitk_reduce(
        const float * __restrict__ O_part,
        const float2 * __restrict__ ml_part,
        const float * __restrict__ sinks_ptr,
        float * __restrict__ dst_ptr,
        const int nq, const int Hq, const int num_splits) {
    const int d     = threadIdx.x; // 0..D-1
    const int head  = blockIdx.x;
    const int q_tok = blockIdx.y;
    const int batch = blockIdx.z;

    float m_global = sinks_ptr ? (sinks_ptr[head] * FA_LOG2E) : -INFINITY;
    for (int sp = 0; sp < num_splits; ++sp) {
        const size_t slot = ((size_t)(batch * num_splits + sp) * nq + q_tok) * Hq + head;
        m_global = fmaxf(m_global, ml_part[slot].x);
    }

    float acc = 0.0f;
    float l_total = 0.0f;
    if (isfinite(m_global)) {
        if (sinks_ptr) {
            l_total += exp2f(sinks_ptr[head] * FA_LOG2E - m_global);
        }
        for (int sp = 0; sp < num_splits; ++sp) {
            const size_t slot = ((size_t)(batch * num_splits + sp) * nq + q_tok) * Hq + head;
            const float2 ml = ml_part[slot];
            if (isfinite(ml.x) && ml.y > 0.0f) {
                const float w = exp2f(ml.x - m_global);
                l_total += ml.y * w;
                acc += O_part[slot * D + d] * w;
            }
        }
    }

    const float inv_l = (l_total > 0.0f) ? (1.0f / l_total) : 0.0f;
    dst_ptr[(((size_t)(batch * nq + q_tok) * Hq + head) * D) + d] = acc * inv_l;
}

} // namespace

static bool ggml_cuda_fattn_rdna_wmma_supported(const int device, const ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc) && !GGML_CUDA_CC_IS_RDNA3_5(cc)) return false;

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    if (!Q || !K || !V) return false;
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (K->type != V->type || (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0)) return false;
    if ((Q->ne[0] != 128 && Q->ne[0] != 256) || V->ne[0] != Q->ne[0]) return false;
    // Allow small nq (DFlash2/MTP verify) when packed GQA rows >= 16
    const int64_t gqa = Q->ne[2] / K->ne[2];
    if (Q->ne[1] * gqa < 16) return false;
    if (Q->ne[1] < 4) return false; // keep specialized fattn-vec for pure nq=1..3 decode
    if (GGML_CUDA_CC_IS_RDNA3_5(cc) && K->type == GGML_TYPE_F16 && Q->ne[1] < 16) return false;
    if (mask && mask->type != GGML_TYPE_F16) return false;
    if (dst->src[5] != nullptr) return false;
    if (Q->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || !ggml_is_contiguous(dst)) return false;
    if (K->type == GGML_TYPE_F16 && ((K->nb[1] & 15) != 0 || (V->nb[1] & 15) != 0)) return false;
    if ((Q->nb[1] & 15) != 0) return false;

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    if (max_bias != 0.0f) return false;
    return true;
}

static void ggml_cuda_flash_attn_ext_rdna_wmma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    float scale = 1.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int D     = (int) Q->ne[0];
    const int nq    = (int) Q->ne[1];
    const int Hq    = (int) Q->ne[2];
    const int batch = (int) Q->ne[3];
    const int S_kv  = (int) K->ne[1];
    const int Hk    = (int) K->ne[2];
    const int gqa   = Hq / Hk;

    if (D == 256 && (S_kv % 256 == 0)) {
        ggml_backend_cuda_record_fattn_mma256_launch();
    }

    const int BM = (D == 128) ? 128 : 64;
    const int n_qblocks = (nq * gqa + BM - 1) / BM;
    const int base_blocks = n_qblocks * Hk * batch;

    // Choose Split-KV count to saturate GPU CUs on small-nq verify (nq=4..64) with long S_kv
    int num_splits = 1;
    const int target_blocks = 128;
    if (base_blocks < target_blocks && S_kv >= 512) {
        int want = (target_blocks + base_blocks - 1) / base_blocks;
        int max_by_kv = std::max(1, S_kv / 256);
        num_splits = std::max(1, std::min({want, max_by_kv, 32}));
    }

    const dim3 grid((unsigned)n_qblocks, (unsigned)Hk, (unsigned)(batch * num_splits));
    const dim3 block(32, 8, 1);
    hipStream_t stream = ctx.stream();

    const int ne32 = mask ? (int) mask->ne[2] : 1;
    const int ne33 = mask ? (int) mask->ne[3] : 1;
    const int nb31 = mask ? (int) mask->nb[1] : 0;
    const int nb32 = mask ? (int) mask->nb[2] : 0;
    const int64_t nb33 = mask ? (int64_t) mask->nb[3] : 0;

    const bool q8 = (K->type == GGML_TYPE_Q8_0);
    const bool sc = (logit_softcap != 0.0f);

    ggml_cuda_pool_alloc<float>  o_part_alloc(ctx.pool());
    ggml_cuda_pool_alloc<float2> ml_part_alloc(ctx.pool());
    float * out_ptr = (float *) dst->data;
    float2 * ml_ptr = nullptr;
    if (num_splits > 1) {
        const size_t slots = (size_t)batch * num_splits * nq * Hq;
        out_ptr = o_part_alloc.alloc(slots * D);
        ml_ptr  = ml_part_alloc.alloc(slots);
    }

    auto launch = [&](auto D_tag, auto SD_tag, auto Q8_tag, auto SC_tag) {
        constexpr int DD = decltype(D_tag)::value;
        constexpr int SD = decltype(SD_tag)::value;
        constexpr bool QQ = decltype(Q8_tag)::value;
        constexpr bool SS = decltype(SC_tag)::value;
        fattn_rdna_wmma_kernel<DD, 32, SD, QQ, SS><<<grid, block, 0, stream>>>(
            (const char *) Q->data, (const char *) K->data, (const char *) V->data,
            mask ? (const char *) mask->data : nullptr,
            sinks ? (const float *) sinks->data : nullptr,
            out_ptr, ml_ptr,
            scale, logit_softcap,
            nq, Hq, Hk, S_kv, num_splits,
            (int) Q->nb[1], (int) Q->nb[2], (int) Q->nb[3],
            (int) K->nb[1], (int) K->nb[2], (int64_t) K->nb[3],
            (int) V->nb[1], (int) V->nb[2], (int64_t) V->nb[3],
            ne32, ne33, nb31, nb32, nb33);
    };

    if (D == 128) {
        if (!q8 && !sc) launch(std::integral_constant<int,128>{}, std::integral_constant<int,1>{}, std::false_type{}, std::false_type{});
        if (!q8 &&  sc) launch(std::integral_constant<int,128>{}, std::integral_constant<int,1>{}, std::false_type{}, std::true_type{});
        if ( q8 && !sc) launch(std::integral_constant<int,128>{}, std::integral_constant<int,1>{}, std::true_type{},  std::false_type{});
        if ( q8 &&  sc) launch(std::integral_constant<int,128>{}, std::integral_constant<int,1>{}, std::true_type{},  std::true_type{});
    } else {
        if (!q8 && !sc) launch(std::integral_constant<int,256>{}, std::integral_constant<int,2>{}, std::false_type{}, std::false_type{});
        if (!q8 &&  sc) launch(std::integral_constant<int,256>{}, std::integral_constant<int,2>{}, std::false_type{}, std::true_type{});
        if ( q8 && !sc) launch(std::integral_constant<int,256>{}, std::integral_constant<int,2>{}, std::true_type{},  std::false_type{});
        if ( q8 &&  sc) launch(std::integral_constant<int,256>{}, std::integral_constant<int,2>{}, std::true_type{},  std::true_type{});
    }

    if (num_splits > 1) {
        const dim3 rgrid((unsigned)Hq, (unsigned)nq, (unsigned)batch);
        if (D == 128) {
            fattn_rdna_splitk_reduce<128><<<rgrid, 128, 0, stream>>>(
                out_ptr, ml_ptr, sinks ? (const float *) sinks->data : nullptr,
                (float *) dst->data, nq, Hq, num_splits);
        } else {
            fattn_rdna_splitk_reduce<256><<<rgrid, 256, 0, stream>>>(
                out_ptr, ml_ptr, sinks ? (const float *) sinks->data : nullptr,
                (float *) dst->data, nq, Hq, num_splits);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

#endif // defined(GGML_USE_HIP)
