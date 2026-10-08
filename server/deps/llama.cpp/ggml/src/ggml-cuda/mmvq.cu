#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static thread_local size_t g_mmvq_launch_count = 0;
static thread_local size_t g_mmvq_mmid_grouped_launch_count = 0;

extern "C" size_t ggml_backend_cuda_get_mmvq_launch_count(void) {
    return g_mmvq_launch_count;
}

extern "C" size_t ggml_backend_cuda_get_mmvq_mmid_grouped_launch_count(void) {
    return g_mmvq_mmid_grouped_launch_count;
}

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q4_0_ROCMFP4:      return vec_dot_rocmfp4_q8_1;
        case GGML_TYPE_Q4_0_ROCMFP4_FAST: return vec_dot_rocmfp4_fast_q8_1;
        case GGML_TYPE_Q2_0_ROCMFP2:      return vec_dot_rocmfpx_fp2_q8_1;
        case GGML_TYPE_Q3_0_ROCMFPX:      return vec_dot_rocmfpx_fp3_q8_1;
        case GGML_TYPE_Q6_0_ROCMFPX:      return vec_dot_rocmfpx_fp6_q8_1;
        case GGML_TYPE_Q8_0_ROCMFPX:      return vec_dot_rocmfpx_fp8_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0_ROCMFP4:      return VDR_ROCMFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0_ROCMFP4_FAST: return VDR_ROCMFP4_FAST_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0_ROCMFP2:      return VDR_ROCMFP2_Q8_1_MMVQ;
        case GGML_TYPE_Q3_0_ROCMFPX:      return VDR_ROCMFP3_Q8_1_MMVQ;
        case GGML_TYPE_Q6_0_ROCMFPX:      return VDR_ROCMFP6_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0_ROCMFPX:      return VDR_ROCMFP8_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

// MMVQ uses VDR=2 and supplies even iqs values {0, 2, 4, 6}. Each lane
// therefore consumes one disjoint 24-bit quarter of the 96-bit FP3 payload.
// Loading only those three contiguous bytes avoids reconstructing the full
// 12-byte block independently in four lanes. The two DP4A operations and the
// final two-scale expression remain in exactly the same order as the generic
// path in vecdotq.cuh.
static __device__ __forceinline__ float vec_dot_rocmfpx_fp3_q8_1_packed24(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {

    static_assert(VDR_ROCMFP3_Q8_1_MMVQ == 2,
                  "packed FP3 MMVQ path requires VDR=2");
    const block_rocmfp3 * bq3 = (const block_rocmfp3 *) vbq + kbx;
    const int byte_offset = 3 * (iqs >> 1);
    const uint32_t packed24 =
        (uint32_t) bq3->qs[byte_offset + 0] |
        ((uint32_t) bq3->qs[byte_offset + 1] << 8) |
        ((uint32_t) bq3->qs[byte_offset + 2] << 16);

    const int val0 = rocmfpx_pack4_fp3_bits12_vec_cuda(packed24 & 0xFFFu);
    const int val1 = rocmfpx_pack4_fp3_bits12_vec_cuda(packed24 >> 12);
    const int u0 = get_int_b4(bq8_1->qs, iqs + 0);
    const int u1 = get_int_b4(bq8_1->qs, iqs + 1);

    int sumi0 = 0;
    int sumi1 = 0;
    if (iqs < QK_ROCMFP3/8) {
        sumi0 = ggml_cuda_dp4a(val0, u0, sumi0);
        sumi0 = ggml_cuda_dp4a(val1, u1, sumi0);
    } else {
        sumi1 = ggml_cuda_dp4a(val0, u0, sumi1);
        sumi1 = ggml_cuda_dp4a(val1, u1, sumi1);
    }

    const float db = __low2float(bq8_1->ds);
    return db * (rocmfpx_ue4m3_to_fp32_finite(bq3->e[0]) * sumi0 +
                 rocmfpx_ue4m3_to_fp32_finite(bq3->e[1]) * sumi1);
}

// FP2 MMVQ assigns one lane to each 16-weight half-block. Load that lane's
// four packed bytes with one unaligned-safe memcpy instead of four independent
// byte loads. Byte extraction, DP4A order, scale conversion, and the final
// floating-point multiplication remain identical to vecdotq.cuh.
static __device__ __forceinline__ float vec_dot_rocmfpx_fp2_q8_1_packed32(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {

    static_assert(VDR_ROCMFP2_Q8_1_MMVQ == 1,
                  "packed FP2 MMVQ path requires VDR=1");
    const block_rocmfp2 * bq2 = (const block_rocmfp2 *) vbq + kbx;
    uint32_t packed32;
    memcpy(&packed32, bq2->qs + 4*iqs, sizeof(packed32));

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t bits8 = (packed32 >> (8*j)) & 0xFFu;
        const int val_packed = rocmfpx_pack4_fp2_bits8_vec_cuda(bits8);
        const int u = get_int_b4(bq8_1->qs, 4*iqs + j);
        sumi = ggml_cuda_dp4a(val_packed, u, sumi);
    }

    const float db = __low2float(bq8_1->ds);
#ifdef ROCMFP2_AFFINE
    return rocmfpx_fp2_affine_dot(bq2, bq8_1, sumi, iqs, db);
#else
    return db * rocmfpx_fp2_half_scale_to_fp32_finite(bq2->e[iqs]) * sumi;
#endif
}

// vec_dot_rocmfpx_fp2_q8_1 on operands that are already in registers. Same
// byte extraction, DP4A order and final float expression, so the value is
// bit-identical to the memory-operand version.
static __device__ __forceinline__ float vec_dot_rocmfpx_fp2_q8_1_regs(
    const uint32_t packed32, const uint8_t e, const int (&u)[4],
    const float db) {

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t bits8 = (packed32 >> (8*j)) & 0xFFu;
        const int val_packed = rocmfpx_pack4_fp2_bits8_vec_cuda(bits8);
        sumi = ggml_cuda_dp4a(val_packed, u[j], sumi);
    }
    return db * rocmfpx_fp2_half_scale_to_fp32_finite(e) * sumi;
}

// ROCmFP2 MoE rows are latency bound in the generic MMVQ loop on gfx1151:
// every iteration waits for a row's scale byte before it issues the next
// row's loads, so a warp keeps one or two DRAM requests in flight. This loop
// first issues the weight, scale and activation loads of up to eight
// iterations, then runs the dot products. Each lane still adds the same
// vec_dot_rocmfpx_fp2_q8_1 terms to the same accumulator in the same kbx
// order, so tmp/tmp_gate are bit-identical to the generic loop.
template <int c_rows_per_block, bool has_fusion>
static __device__ __forceinline__ void mul_mat_vec_q_moe_fp2_prefetch(
        const void * __restrict__ vx, const void * __restrict__ vgate,
        const bool use_gate, const block_q8_1 * __restrict__ y,
        const int kbx_offset, const uint32_t stride_row_x,
        const int blocks_per_row_x,
        float (&tmp)[c_rows_per_block], float (&tmp_gate)[c_rows_per_block]) {
    constexpr int qk  = ggml_cuda_type_traits<GGML_TYPE_Q2_0_ROCMFP2>::qk;
    constexpr int qi  = ggml_cuda_type_traits<GGML_TYPE_Q2_0_ROCMFP2>::qi;
    constexpr int vdr = VDR_ROCMFP2_Q8_1_MMVQ;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_iter = vdr * warp_size / qi;
    constexpr int n_iter =
        c_rows_per_block * (has_fusion ? 2 : 1) <= 2 ? 8 : 4;
    static_assert(vdr == 1, "prefetching FP2 MMVQ requires VDR=1");

    const int iqs = threadIdx.x % (qi/vdr);
    const bool load_gate = has_fusion && use_gate;
    const block_rocmfp2 * x = (const block_rocmfp2 *) vx + kbx_offset;
    const block_rocmfp2 * g = load_gate
        ? (const block_rocmfp2 *) vgate + kbx_offset : x;

    for (int kbx0 = threadIdx.x / (qi/vdr); kbx0 < blocks_per_row_x;
         kbx0 += n_iter*blocks_per_iter) {
        uint32_t qx[n_iter][c_rows_per_block];
        uint8_t  ex[n_iter][c_rows_per_block];
        uint32_t qg[n_iter][c_rows_per_block];
        uint8_t  eg[n_iter][c_rows_per_block];
        int      yq[n_iter][4];
        half2    yds[n_iter];

#pragma unroll
        for (int it = 0; it < n_iter; ++it) {
            const int kbx = kbx0 + it*blocks_per_iter;
            if (kbx < blocks_per_row_x) {
                const block_q8_1 * by = y + kbx*(qk/QK8_1);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    yq[it][j] = get_int_b4(by->qs, 4*iqs + j);
                }
                yds[it] = by->ds;
#pragma unroll
                for (int i = 0; i < c_rows_per_block; ++i) {
                    const block_rocmfp2 * bx = x + i*stride_row_x + kbx;
                    memcpy(&qx[it][i], bx->qs + 4*iqs, sizeof(uint32_t));
                    ex[it][i] = bx->e[iqs];
                    if (load_gate) {
                        const block_rocmfp2 * bg = g + i*stride_row_x + kbx;
                        memcpy(&qg[it][i], bg->qs + 4*iqs, sizeof(uint32_t));
                        eg[it][i] = bg->e[iqs];
                    }
                }
            }
        }

#pragma unroll
        for (int it = 0; it < n_iter; ++it) {
            const int kbx = kbx0 + it*blocks_per_iter;
            if (kbx < blocks_per_row_x) {
                // Convert here, not in the load phase: an f16 convert next
                // to the loads makes the compiler wait for each batch.
                const float yd = __low2float(yds[it]);
#pragma unroll
                for (int i = 0; i < c_rows_per_block; ++i) {
                    tmp[i] += vec_dot_rocmfpx_fp2_q8_1_regs(
                        qx[it][i], ex[it][i], yq[it], yd);
                    if (load_gate) {
                        tmp_gate[i] += vec_dot_rocmfpx_fp2_q8_1_regs(
                            qg[it][i], eg[it][i], yq[it], yd);
                    }
                }
            }
        }
    }
}

// Dense q4 verification applies one weight row to four token activations.
// Decode the ROCmFP4 payload once and retain one independent integer
// accumulator per token.  Each token keeps the original DP4A and floating
// multiplication order, so this removes duplicate weight/codebook work
// without changing its result.
static __device__ __forceinline__ float4 vec_dot_rocmfp4_fast_q8_1_x4(
        const void * __restrict__ vbq,
        const block_q8_1 * __restrict__ bq8_0,
        const block_q8_1 * __restrict__ bq8_1,
        const block_q8_1 * __restrict__ bq8_2,
        const block_q8_1 * __restrict__ bq8_3,
        const int & kbx, const int & iqs) {
    const block_rocmfp4_fast * bq4 =
        (const block_rocmfp4_fast *) vbq + kbx;
    const int * q80 = (const int *) bq8_0->qs + iqs;
    const int * q81 = (const int *) bq8_1->qs + iqs;
    const int * q82 = (const int *) bq8_2->qs + iqs;
    const int * q83 = (const int *) bq8_3->qs + iqs;

    int sumi0 = 0;
    int sumi1 = 0;
    int sumi2 = 0;
    int sumi3 = 0;
#pragma unroll
    for (int l = 0; l < VDR_ROCMFP4_FAST_Q8_1_MMVQ; ++l) {
        const int aux_q4 = rocmfp4_get_qs_i32(bq4->qs, iqs + l);
        const int2 v =
            rocmfp4_get_int_from_codebook_16(aux_q4, kvalues_rocmfp4);
        sumi0 = ggml_cuda_dp4a(v.x, q80[l + 0], sumi0);
        sumi0 = ggml_cuda_dp4a(v.y, q80[l + 4], sumi0);
        sumi1 = ggml_cuda_dp4a(v.x, q81[l + 0], sumi1);
        sumi1 = ggml_cuda_dp4a(v.y, q81[l + 4], sumi1);
        sumi2 = ggml_cuda_dp4a(v.x, q82[l + 0], sumi2);
        sumi2 = ggml_cuda_dp4a(v.y, q82[l + 4], sumi2);
        sumi3 = ggml_cuda_dp4a(v.x, q83[l + 0], sumi3);
        sumi3 = ggml_cuda_dp4a(v.y, q83[l + 4], sumi3);
    }

    const float d = rocmfp4_ue4m3_to_fp32_half_finite(bq4->e);
    return make_float4(
        __low2float(bq8_0->ds) * d * sumi0,
        __low2float(bq8_1->ds) * d * sumi1,
        __low2float(bq8_2->ds) * d * sumi2,
        __low2float(bq8_3->ds) * d * sumi3);
}

template <ggml_type type, bool c_fp3_packed24>
static __device__ __forceinline__ float vec_dot_q_mmvq(
        const void * __restrict__ vbq,
        const block_q8_1 * __restrict__ bq8_1,
        const int & kbx, const int & iqs) {
    static_assert(!c_fp3_packed24 || type == GGML_TYPE_Q3_0_ROCMFPX,
                  "packed FP3 dispatch requires the ROCmFP3 type");
    if constexpr (c_fp3_packed24) {
        return vec_dot_rocmfpx_fp3_q8_1_packed24(
            vbq, bq8_1, kbx, iqs);
    } else {
        constexpr vec_dot_q_cuda_t vec_dot_q_cuda =
            get_vec_dot_q_cuda(type);
        return vec_dot_q_cuda(vbq, bq8_1, kbx, iqs);
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_MXFP8: return 0;
        case GGML_TYPE_Q3_1_ROCMFP3_MIX: return 0;
        case GGML_TYPE_Q2_1_ROCMFP2_MIX: return 0;
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_MXFP8: return 0;
        case GGML_TYPE_Q3_1_ROCMFP3_MIX: return 0;
        case GGML_TYPE_Q2_1_ROCMFP2_MIX: return 0;
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_MXFP8: return 0;
        case GGML_TYPE_Q3_1_ROCMFP3_MIX: return 0;
        case GGML_TYPE_Q2_1_ROCMFP2_MIX: return 0;
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        case GGML_TYPE_MXFP8: return 0;
        case GGML_TYPE_Q3_1_ROCMFP3_MIX: return 0;  // no MMVQ; dequant->cuBLAS
        case GGML_TYPE_Q2_1_ROCMFP2_MIX: return 0;  // no MMVQ; dequant->cuBLAS
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_MXFP8: return 0;
        case GGML_TYPE_Q3_1_ROCMFP3_MIX: return 0;  // no MMVQ; dequant->cuBLAS
        case GGML_TYPE_Q2_1_ROCMFP2_MIX: return 0;  // no MMVQ; dequant->cuBLAS
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_MXFP8: return 0;
        case GGML_TYPE_Q3_1_ROCMFP3_MIX: return 0;  // no MMVQ; dequant->cuBLAS
        case GGML_TYPE_Q2_1_ROCMFP2_MIX: return 0;  // no MMVQ; dequant->cuBLAS
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        case GGML_TYPE_MXFP8: return 0;
        case GGML_TYPE_Q3_1_ROCMFP3_MIX: return 0;  // no MMVQ; dequant->cuBLAS
        case GGML_TYPE_Q2_1_ROCMFP2_MIX: return 0;  // no MMVQ; dequant->cuBLAS
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}


// [TAG_MMID_GROUPED] grouped-expert MUL_MAT_ID for small speculative-verify
// batches (2..MMVQ_MAX_MOE_BATCH_SIZE tokens). Consecutive draft tokens route
// to heavily overlapping expert sets, but mul_mat_vec_q_moe reads each
// (token, slot) pair's expert weights independently. This path sorts the
// pairs by expert so that same-expert reads coalesce in cache, cutting expert
// weight traffic toward the union of routed experts. Bit-exact per
// (row, token) vs mul_mat_vec_q_moe (same vec_dot sequence and reduction).
// Model-agnostic: applies to any MoE with n_expert_used*n_tokens <= 256.
//   LUCE_MMID_GROUPED         1 = enable, 0 = disable
//                               (CUDA default on; HIP default off)
//   LUCE_MMID_GROUPED_TYPES   bitmask, 1 = Q4_K, 2 = Q6_K,
//                               4 = Q4_0/Q8_0/Q5_K, 8 = ROCmFP2/ROCmFP3,
//                               16 = ROCmFP4-fast, 32 = ROCmFP3 only,
//                               64 = Q5_0 on sm_86 with batch/projection-shape guards.
//                               Default 71; 7 disables Q5_0 while retaining
//                               the previously enabled formats.
//   LUCE_MMID_GROUPED_DEVICE  optional zero-based device index; unset/-1
//                               applies the path to every eligible device.
//                               Q6_K stays on its tuned MMQ route above 5
//                               tokens unless enabled. The ROCmFP formats are
//                               opt-in until qualified on each AMD target.
#define MMID_GROUPED_MAX_PAIRS 256
#define MMID_GROUPED_DEFAULT_TPG 2
#define MMID_GROUPED_FP3_TPG     4
#define MMID_META_NG 0
#define MMID_META_GE 1
#define MMID_META_GS (MMID_META_GE + MMID_GROUPED_MAX_PAIRS)
#define MMID_META_PT (MMID_META_GS + MMID_GROUPED_MAX_PAIRS + 1)
#define MMID_META_PS (MMID_META_PT + MMID_GROUPED_MAX_PAIRS)
#define MMID_META_INTS (MMID_META_PS + MMID_GROUPED_MAX_PAIRS)

// [TAG_MMID_ADAPTIVE_K] per-token expert budget. The graph builder tags the
// mmid ids tensor via ->extra with this struct; the prep kernel then drops
// low-mass slots (cumulative router weight >= tau keeps), sentinels their ids
// to -1 and renormalizes the kept weights in place. Idempotent across the
// gate/up/down calls of one layer (-1 markers). Requires the grouped path for
// every routed expert type (LUCE_MMID_GROUPED=1, LUCE_MMID_GROUPED_TYPES=7).
struct mmid_gate_extra {
    uint32_t magic;      // 0x4D474154 "MGAT"
    float tau;
    const ggml_tensor * weights;  // [n_used, n_tok] f32, combine weights
};
#define MMID_GATE_MAGIC 0x4D474154u

static bool mmid_grouped_env() {
    // Bit-exact and measured equal-or-faster on small MoE verify batches, so
    // enabled by default on CUDA; LUCE_MMID_GROUPED=0 is the kill switch.
    // HIP (RDNA3/RDNA4) stays opt-in and default-off. Generic quantized types
    // have correctness coverage on gfx1151; ROCmFP2/ROCmFP3 are separately
    // enabled by the type mask after platform-specific qualification.
    static const bool on = []() {
        const char * e = std::getenv("LUCE_MMID_GROUPED");
        if (e != nullptr) {
            return e[0] == '1' && e[1] == '\0';
        }
#ifdef GGML_USE_HIP
        return false;
#else
        return true;
#endif
    }();
    return on;
}

static bool mmvq_env_flag(const char * name, bool default_value = false) {
    const char * value = std::getenv(name);
    if (!value || !value[0]) return default_value;
    return std::strcmp(value, "0") != 0;
}

static bool mmid_grouped_type_ok(ggml_type type) {
    // bit0 = Q4_K, bit1 = Q6_K, bit2 = Q4_0/Q8_0/Q5_K,
    // bit3 = Q2_0_ROCMFP2/Q3_0_ROCMFPX, bit4 = ROCmFP4-fast,
    // bit5 = ROCmFP3 only, bit6 = Q5_0. Q5_0 has additional architecture,
    // batch and projection-shape guards below. LUCE_MMID_GROUPED_TYPES is an
    // experimental override; 7 restores the policy without Q5_0.
    static const int mask = []() {
        const char * e = std::getenv("LUCE_MMID_GROUPED_TYPES");
        if (e == nullptr || e[0] == '\0') {
            return 7 | 64;
        }
        return atoi(e);
    }();
    switch (type) {
        case GGML_TYPE_Q5_0:
            return (mask & 64) != 0;
        case GGML_TYPE_Q4_K:
            return (mask & 1) != 0;
        case GGML_TYPE_Q6_K:
            return (mask & 2) != 0;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_K:
            return (mask & 4) != 0;
        case GGML_TYPE_Q2_0_ROCMFP2:
            return (mask & 8) != 0;
        case GGML_TYPE_Q3_0_ROCMFPX:
            // Bit 8 preserves the original combined ROCmFP2/3 policy. Bit 32
            // allows gfx1151 profiles to select the independently qualified
            // ROCmFP3 path without also routing fused ROCmFP2 gate/up through
            // a schedule that is slower for that shape.
            return (mask & (8 | 32)) != 0;
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
            return (mask & 16) != 0;
        default:
            return false;
    }
}

// [TAG_MMID_GROUPED] LUCE_MMID_PREP_MEMO=0 recomputes the expert grouping for
// every MUL_MAT_ID (see the grouped path in ggml_cuda_mul_mat_vec_q).
static bool mmid_prep_memo_on() {
    static const bool on = [] {
        const char * e = std::getenv("LUCE_MMID_PREP_MEMO");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    return on;
}
// The memo entry kind of a cached expert grouping (q8_1 entries use a type or -1).
#define MMID_PREP_MEMO_KIND (-2)

static bool mmid_grouped_device_ok() {
    static const int selected = []() {
        const char * e = std::getenv("LUCE_MMID_GROUPED_DEVICE");
        return e == nullptr || e[0] == '\0' ? -1 : atoi(e);
    }();
    return selected < 0 || selected == ggml_cuda_get_device();
}

static bool mmid_grouped_arch_ok(int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING) ||
        GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc);
}

bool ggml_cuda_mmvq_mmid_grouped_enabled(
        ggml_type type, int cc, int64_t ncols_dst, int64_t routed_pairs) {
    // This shape-independent predicate also vetoes graph fusion. Q5_0 is
    // selected only at the kernel call site below, after fusion is decided,
    // so adding its shape-specific path does not disable existing fusions.
    if (type == GGML_TYPE_Q5_0) {
        return false;
    }
    return ncols_dst >= 2 && ncols_dst <= MMVQ_MAX_MOE_BATCH_SIZE &&
        routed_pairs <= MMID_GROUPED_MAX_PAIRS &&
        mmid_grouped_env() && mmid_grouped_type_ok(type) &&
        mmid_grouped_arch_ok(cc) && mmid_grouped_device_ok();
}

// Ordinary MMID admission, also used to select the invariant fallback below.
static int get_mmvq_mmid_max_batch_regular(ggml_type type, int cc) {
    // [TAG_MMID_GROUPED] the grouped kernel handles any supported type up to the
    // MoE batch ceiling; this also keeps CUDA graphs on for these batches.
    // RDNA3/RDNA4 (wave32) share the non-grouped kernel's wave-width warp_reduce.
    // IS_RDNA3/IS_RDNA4 are pure cc-range checks, safe above the IS_AMD guard below.
    // The HIP path remains opt-in until on-hardware parity and performance validation.
    // Q5_0 support is shape-specific and must not raise the MMVQ batch ceiling
    // for the shapes that still use the legacy kernel.
    if (type != GGML_TYPE_Q5_0 && mmid_grouped_env() && mmid_grouped_type_ok(type) &&
        mmid_grouped_arch_ok(cc) && mmid_grouped_device_ok()) {
        return MMVQ_MAX_MOE_BATCH_SIZE;
    }
    // Dedicated multi-token MoE kernel: extend the MUL_MAT_ID ceiling to 16
    // tokens on NVIDIA Turing+ for types whose base ceiling is already the
    // maximum. Types with tuned lower ceilings (per PR 20905) keep them.
    static const bool moe_kernel_enabled =
        mmvq_env_flag("LUCE_CUDA_MMVQ_MOE_KERNEL", true);
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return moe_kernel_enabled ? MMVQ_MAX_MOE_BATCH_SIZE : MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            const int base = get_mmvq_mmid_max_batch_turing_plus(type);
            if (moe_kernel_enabled && base >= MMVQ_MAX_BATCH_SIZE) {
                return MMVQ_MAX_MOE_BATCH_SIZE;
            }
            return base;
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    const int limit = get_mmvq_mmid_max_batch_regular(type, cc);
    // The RDNA3 Q4_K/Q5_K/Q6_K ceiling is four, even though dense invariant
    // MMVQ admits eight. Verification must not fall through to MMQ (different
    // activation quantization). Wider invariant MMID uses the tokenwise kernel,
    // not the MoE kernel whose compiled launch_bounds still impose that ceiling.
    return limit > 0 && ggml_cuda_mmvq_batch_invariant() ? std::max(limit, MMVQ_MAX_BATCH_SIZE) : limit;
}

static bool mmid_invariant_tokenwise(ggml_type type, int cc, int64_t ncols) {
    if (!ggml_cuda_mmvq_batch_invariant() || ncols > MMVQ_MAX_BATCH_SIZE) return false;
    const int limit = get_mmvq_mmid_max_batch_regular(type, cc);
    return limit > 0 && ncols > limit;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_MOE_BATCH_SIZE;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    // Mirror the host-side extension: full-cap types compile the MoE kernel
    // for up to MMVQ_MAX_MOE_BATCH_SIZE tokens (launch_bounds); the host
    // router decides at runtime whether to use the extended range.
    return get_mmvq_mmid_max_batch_turing_plus(type) >= MMVQ_MAX_BATCH_SIZE
        ? MMVQ_MAX_MOE_BATCH_SIZE
        : get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 8;
                case GGML_TYPE_IQ4_XS:
                    // gfx1201 decode peaks at 6-7 waves; prefer the smaller
                    // block after an end-to-end 4/5/6/7/8-wave sweep.
                    return 6;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

static bool is_gfx1151(const int cc) {
    return cc == GGML_CUDA_CC_OFFSET_AMD + 0x1151;
}

// Dense projections at two to sixteen columns use the same weight row for
// every column. Decode each packed ROCmFP4 fragment once while preserving the
// original per-column DP4A and floating-point accumulation order, so every
// column equals the single-column kernel bit for bit at any width.
#define ROCMFP4_REUSE_MAX_COLS 16
template <int ncols>
static __device__ __forceinline__ void vec_dot_rocmfp4_fast_q8_1_ncols(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ y,
        const uint32_t stride_col_y, const int kby, const int kbx, const int iqs,
        float (&result)[ncols]) {
    static_assert(ncols >= 2 && ncols <= ROCMFP4_REUSE_MAX_COLS,
                  "weight reuse covers two to sixteen dense columns");
    const block_rocmfp4_fast * bq4 = (const block_rocmfp4_fast *) vbq + kbx;

    int2 weights[VDR_ROCMFP4_FAST_Q8_1_MMVQ];
#pragma unroll
    for (int l = 0; l < VDR_ROCMFP4_FAST_Q8_1_MMVQ; ++l) {
        const int packed = rocmfp4_get_qs_i32(bq4->qs, iqs + l);
        weights[l] = rocmfp4_get_int_from_codebook_16(packed, kvalues_rocmfp4);
    }

    const float weight_scale = rocmfp4_ue4m3_to_fp32_half_finite(bq4->e);
    int sums[ncols] = {};
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        const block_q8_1 * bq8 = &y[j*stride_col_y + kby];
        const int * q8 = (const int *) bq8->qs + iqs;
#pragma unroll
        for (int l = 0; l < VDR_ROCMFP4_FAST_Q8_1_MMVQ; ++l) {
            sums[j] = ggml_cuda_dp4a(weights[l].x, q8[l + 0], sums[j]);
            sums[j] = ggml_cuda_dp4a(weights[l].y, q8[l + 4], sums[j]);
        }
        result[j] = __low2float(bq8->ds) * weight_scale * sums[j];
    }
}

static bool rocmfp3_packed24_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("LUCE_CUDA_MMVQ_FP3_PACKED24");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

static bool rocmfp4_x4_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("LUCE_CUDA_MMVQ_FP4_X4");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

static bool rocmfp4_q5_x4_plus1_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("LUCE_CUDA_MMVQ_FP4_Q5_X4_PLUS1");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

// __shfl_xor(v, off) in a wave32 without the LDS crossbar: off 16 is
// v_permlanex16 with identity selectors (lane n <- lane n^16), off < 16 is
// ROW_XMASK DPP (lane n <- lane n^off inside a row of 16), the exchanges
// gqh.cu verified on gfx1201. Needs a full EXEC mask.
template <int off>
static __device__ __forceinline__ int warp_shfl_xor_wave32(const int v) {
    static_assert(off > 0 && off <= 16, "a wave32 xor exchange");
#if defined(GGML_USE_HIP) && (defined(RDNA3) || defined(RDNA4))
    if constexpr (off == 16) {
        return __builtin_amdgcn_permlanex16(0, v, 0x76543210u, 0xfedcba98u, false, true);
    } else {
        return __builtin_amdgcn_update_dpp(0, v, 0x160 | off, 0xf, 0xf, false);
    }
#else
    return __shfl_xor_sync(0xffffffff, v, off, 32);
#endif // defined(GGML_USE_HIP) && (defined(RDNA3) || defined(RDNA4))
}

// warp_reduce_sum's xor butterfly on those exchanges: every lane adds the
// partner value __shfl_xor returns, so each level and the sum are
// bit-identical.
static __device__ __forceinline__ float warp_reduce_sum_wave32_dpp(float x) {
    x += __int_as_float(warp_shfl_xor_wave32<16>(__float_as_int(x)));
    x += __int_as_float(warp_shfl_xor_wave32< 8>(__float_as_int(x)));
    x += __int_as_float(warp_shfl_xor_wave32< 4>(__float_as_int(x)));
    x += __int_as_float(warp_shfl_xor_wave32< 2>(__float_as_int(x)));
    x += __int_as_float(warp_shfl_xor_wave32< 1>(__float_as_int(x)));
    return x;
}

// The shared-grid IQ types' lane dot from grid, their table in shared memory
// (uint2 IQ2_XXS entries, the level table with levels, or uint32 IQ3_XXS
// entries): the two factors d * sumi, or their product.
template <ggml_type type, bool levels>
static __device__ __forceinline__ void vec_dot_iq_shared_grid_parts(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs,
        const void * __restrict__ grid, float & d, int & sumi) {
    if constexpr (type == GGML_TYPE_IQ2_XXS) {
        vec_dot_iq2_xxs_q8_1_grid_parts<levels>(vbq, bq8_1, kbx, iqs, (const uint2 *) grid, d, sumi);
    } else {
        static_assert(type == GGML_TYPE_IQ3_XXS, "the shared-grid types are IQ2_XXS and IQ3_XXS");
        vec_dot_iq3_xxs_q8_1_grid_parts(vbq, bq8_1, kbx, iqs, (const uint32_t *) grid, d, sumi);
    }
}

template <ggml_type type, bool levels>
static __device__ __forceinline__ float vec_dot_iq_shared_grid(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs,
        const void * __restrict__ grid) {
    float d;
    int sumi;
    vec_dot_iq_shared_grid_parts<type, levels>(vbq, bq8_1, kbx, iqs, grid, d, sumi);
    return d * sumi;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false,
          int fixed_ncols_x = 0, bool unroll_k_loop_2 = false,
          bool reuse_rocmfp4_weights = false,
          bool c_fp3_packed24 = false, bool c_fp4_x4 = false,
          bool c_width_invariant = false, int c_row_warps = 1, int c_wave_rows = 1>
__launch_bounds__(calc_nwarps(type, c_width_invariant ? 1 : ncols_dst, get_device_table_id())*c_row_warps*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids, const ggml_cuda_mm_fusion_args_device fusion, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride, const bool ids_tokenwise_samples) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    // Width-invariant launches keep the single-column block shape, so each
    // column's K traversal and reduction match a one-column product exactly.
    constexpr int shape_cols = c_width_invariant ? 1 : ncols_dst;
    constexpr int nwarps = calc_nwarps(type, shape_cols, table_id);
    // c_row_warps > 1 stacks independent one-wave rows in one block: warp y
    // owns c_wave_rows consecutive rows starting at
    // c_wave_rows*(blockIdx.x*c_row_warps + y). Each row keeps the one-wave
    // schedule, its own accumulator and its own warp reduction; the rows of a
    // wave share the activation loads. The host launches it only where the
    // table picks one wave and one row.
    static_assert(c_wave_rows == 1 || c_row_warps > 1, "rows per wave need the multi-row block");
    constexpr int rows_per_cuda_block = c_row_warps > 1 ? c_wave_rows :
        calc_rows_per_block(shape_cols, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    if constexpr (c_row_warps > 1 && (nwarps != 1 || calc_rows_per_block(shape_cols, table_id, small_k, nwarps) != 1)) {
        NO_DEVICE_CODE;
        return;
    }
    // The multi-row block is a wave32 kernel (DPP / permlanex16 exchanges, 32-lane
    // row schedule); the host only launches it on wave32 parts, and a wave64
    // target (gfx9) compiles it to an inert stub.
    if constexpr (c_row_warps > 1 && warp_size != 32) {
        NO_DEVICE_CODE;
        return;
    }
#if defined(GGML_USE_HIP)
    // threadIdx.y is wave-uniform here; readfirstlane makes the row, and so
    // the weight and dst addressing, scalar.
    const     int warp_row = c_row_warps > 1 ? __builtin_amdgcn_readfirstlane(threadIdx.y) : 0;
#else
    const     int warp_row = c_row_warps > 1 ? (int) threadIdx.y : 0;
#endif // defined(GGML_USE_HIP)
    const     int tid = c_row_warps > 1 ? (int) threadIdx.x : (int) (warp_size*threadIdx.y + threadIdx.x);
    const     int row0 = rows_per_cuda_block*(c_row_warps*(int) blockIdx.x + warp_row);
    const     int blocks_per_row_x = (fixed_ncols_x > 0 ? fixed_ncols_x : ncols_x) / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    // Multi-row IQ2_XXS / IQ3_XXS blocks decode from a shared-memory grid
    // table (see the fixed-K loop) and reduce with warp_reduce_sum_wave32_dpp.
    constexpr bool iq_shared_grid = (type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ3_XXS) &&
        fixed_ncols_x > 0 && c_row_warps > 1 && warp_size == 32;
#if defined(GGML_USE_HIP)
    constexpr bool iq2xxs_levels = true;
#else
    constexpr bool iq2xxs_levels = false;
#endif // defined(GGML_USE_HIP)
    static_assert(!iq_shared_grid || ncols_dst == 1, "the IQ shared-grid path is a one-column kernel");

    static_assert(fixed_ncols_x == 0 ||
                  type == GGML_TYPE_Q3_0_ROCMFPX ||
                  type == GGML_TYPE_Q2_0_ROCMFP2 ||
                  type == GGML_TYPE_IQ2_XXS ||
                  type == GGML_TYPE_IQ3_XXS,
                  "fixed-K decode specialization is limited to profiled ROCmFP2/FP3/IQ2_XXS/IQ3_XXS types");
    static_assert(!unroll_k_loop_2 || type == GGML_TYPE_Q4_0_ROCMFP4_FAST,
                  "partial K-loop unrolling is limited to the profiled FP4FAST path");
    static_assert(!reuse_rocmfp4_weights ||
                  (type == GGML_TYPE_Q4_0_ROCMFP4_FAST &&
                   ncols_dst >= 2 && ncols_dst <= ROCMFP4_REUSE_MAX_COLS &&
                   fixed_ncols_x == 0 && !unroll_k_loop_2),
                  "weight reuse is limited to the two- to sixteen-column FP4FAST path");
    static_assert(!c_fp3_packed24 || type == GGML_TYPE_Q3_0_ROCMFPX,
                  "packed FP3 MMVQ specialization requires ROCmFP3 weights");
    static_assert(!c_fp4_x4 ||
                      (type == GGML_TYPE_Q4_0_ROCMFP4_FAST &&
                       (ncols_dst == 4 || ncols_dst == 5)),
                  "FP4 x4 MMVQ specialization requires ROCmFP4-fast q4/q5");
    const uint32_t channel_dst = blockIdx.y;

    const bool has_ids = ids != nullptr;
    const bool ids_multi_col = has_ids && ncols_dst > 1;

    const uint32_t sample_dst = blockIdx.z;
    const uint32_t ids_col    = ids_tokenwise_samples ? sample_dst : 0;
    const int32_t  channel_x_raw = has_ids ? ids[channel_dst + ids_col*ids_stride] : 0;
    const bool     channel_valid = !has_ids || channel_x_raw >= 0;
    // Negative MUL_MAT_ID entries are masked routes.  Clamp the address to a
    // valid expert before doing any pointer arithmetic; channel_valid then
    // suppresses every expert-weight/bias read and forces a zero result.
    const uint32_t channel_x  = has_ids ? (channel_valid ? (uint32_t) channel_x_raw : 0u)
                                        : fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y  = has_ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;

    uint32_t channel_xs[ncols_dst];
    bool channel_valids[ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        const int32_t raw = ids_multi_col ? ids[channel_dst + j*ids_stride] : channel_x_raw;
        channel_valids[j] = !has_ids || raw >= 0;
        channel_xs[j] = channel_valids[j] ? (uint32_t) raw : 0u;
        if (!has_ids) {
            channel_xs[j] = channel_x;
        }
    }

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    ggml_glu_op active_glu;
    float glu_param0 = 0.0f;
    float glu_param1 = 0.0f;
    float gate_value_scale = 1.0f;
    float x_value_scale = 1.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_param0    = fusion.glu_param0;
        glu_param1    = fusion.glu_param1;
        gate_value_scale = fusion.gate_value_scale;
        x_value_scale = fusion.x_value_scale;
    }


    float x_biases[ncols_dst]    = { 0.0f };
    float gate_biases[ncols_dst] = { 0.0f };
    if constexpr (has_fusion) {
        if (use_bias) {
            const float * x_bias_base = x_bias + sample_dst*stride_sample_dst + row0;
            // 1. Hide latency by prefetching bias and gate here
            // 2. load only on threads that won't die after partial sum calculation
            if (threadIdx.x < rows_per_cuda_block && (c_row_warps > 1 || threadIdx.y == 0) &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    if (!channel_valids[j]) {
                        continue;
                    }
                    const uint32_t channel_bias = has_ids ? channel_xs[j] : channel_dst;
                    const uint32_t col_offset   = ids_multi_col ? 0 : j*stride_col_dst;
                    x_biases[j] = x_bias_base[channel_bias*stride_channel_dst + col_offset + threadIdx.x];
                }
            }
        }
        if (use_gate_bias) {
            const float * gate_bias_base = gate_bias + sample_dst*stride_sample_dst + row0;
            if (threadIdx.x < rows_per_cuda_block && (c_row_warps > 1 || threadIdx.y == 0) &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    if (!channel_valids[j]) {
                        continue;
                    }
                    const uint32_t channel_bias = has_ids ? channel_xs[j] : channel_dst;
                    const uint32_t col_offset   = ids_multi_col ? 0 : j*stride_col_dst;
                    gate_biases[j] = gate_bias_base[channel_bias*stride_channel_dst + col_offset + threadIdx.x];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;

    if constexpr (reuse_rocmfp4_weights) {
        const int kbx_offset = sample_x*stride_sample_x +
                               channel_x*stride_channel_x + row0*stride_row_x;
        for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                float dots[ncols_dst];
                vec_dot_rocmfp4_fast_q8_1_ncols<ncols_dst>(
                    vx, y, stride_col_y, kby,
                    kbx_offset + i*stride_row_x + kbx, kqs, dots);
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    tmp[j][i] += dots[j];
                }
                if constexpr (has_fusion) {
                    if (use_gate) {
                        float gate_dots[ncols_dst];
                        vec_dot_rocmfp4_fast_q8_1_ncols<ncols_dst>(
                            vgate, y, stride_col_y, kby,
                            kbx_offset + i*stride_row_x + kbx, kqs, gate_dots);
#pragma unroll
                        for (int j = 0; j < ncols_dst; ++j) {
                            tmp_gate[j][i] += gate_dots[j];
                        }
                    }
                }
            }
        }
    } else if constexpr (fixed_ncols_x > 0) {
        static_assert(fixed_ncols_x % qk == 0, "fixed K must contain whole quant blocks");
        constexpr int fixed_blocks = fixed_ncols_x/qk;
        constexpr int fixed_iters = (fixed_blocks + blocks_per_iter - 1) / blocks_per_iter;

        // Multi-row IQ2_XXS / IQ3_XXS blocks copy the grid (2 KiB / 1 KiB) to
        // shared memory once: the lane-divergent lookups become LDS reads
        // instead of vector-cache gathers. On HIP the IQ2_XXS copy is the
        // level table (iq2_xxs_grid_levels); the decoded weights, and so every
        // dot product, are unchanged.
        const void * iq_grid_src = nullptr;
        if constexpr (iq_shared_grid) {
            // A masked route (negative id) makes the whole block's rows zero,
            // exactly as the loop below would, and validity is per block
            // (blockIdx.y), so every wave leaves before the barrier.
            if (!channel_valids[0]) {
                if (threadIdx.x < rows_per_cuda_block &&
                    (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
                    dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0 + threadIdx.x] = 0.0f;
                }
                return;
            }
            const int block_tid = warp_size*threadIdx.y + threadIdx.x;
            if constexpr (type == GGML_TYPE_IQ2_XXS) {
                // Each thread copies two adjacent entries: one 16-byte load.
                __shared__ uint2 iq2xxs_grid_shared[256];
                for (int l = 2*block_tid; l < 256; l += 2*c_row_warps*warp_size) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const uint2 grid = ((const uint2 *) iq2xxs_grid)[l + e];
#if defined(GGML_USE_HIP)
                        iq2xxs_grid_shared[l + e] = iq2_xxs_grid_levels(grid);
#else
                        iq2xxs_grid_shared[l + e] = grid;
#endif // defined(GGML_USE_HIP)
                    }
                }
                iq_grid_src = iq2xxs_grid_shared;
            } else {
                __shared__ uint32_t iq3xxs_grid_shared[256];
                for (int l = block_tid; l < 256; l += c_row_warps*warp_size) {
                    iq3xxs_grid_shared[l] = iq3xxs_grid[l];
                }
                iq_grid_src = iq3xxs_grid_shared;
            }
            __syncthreads();
        }
        GGML_UNUSED(iq_grid_src);

        // K = 2304 leaves one block for the last iteration, i.e. lanes 0-7 of
        // each row. With two rows per wave, lanes 8r..8r+7 decode row r's
        // part of it in one pass and lanes 0-7 take row 1's two factors
        // through ROW_XMASK DPP (lane n <- lane n^8), then form d*sumi and add
        // it themselves: every accumulator adds the same product in the same
        // order as before.
        constexpr bool iq_pack_tail = iq_shared_grid && !has_fusion &&
            rows_per_cuda_block == 2 && fixed_blocks % blocks_per_iter == 1;

#pragma unroll
        for (int iter = 0; iter < fixed_iters; ++iter) {
            if constexpr (iq_pack_tail) {
                if (iter == fixed_iters - 1) {
                    constexpr int lanes_per_block = qi/vdr;
                    const int tail_row = tid / lanes_per_block;
                    const int kqs = vdr * (tid % lanes_per_block);
                    const int kbx_offset = sample_x*stride_sample_x + channel_xs[0]*stride_channel_x + row0*stride_row_x;
                    float tail_d = 0.0f;
                    int tail_sumi = 0;
                    if (tail_row < rows_per_cuda_block) {
                        vec_dot_iq_shared_grid_parts<type, iq2xxs_levels>(
                            vx, &y[(fixed_blocks - 1)*(qk/QK8_1)],
                            kbx_offset + tail_row*stride_row_x + fixed_blocks - 1, kqs, iq_grid_src,
                            tail_d, tail_sumi);
                    }
                    const float tail_d1    = __int_as_float(warp_shfl_xor_wave32<lanes_per_block>(__float_as_int(tail_d)));
                    const int   tail_sumi1 = warp_shfl_xor_wave32<lanes_per_block>(tail_sumi);
                    if (tail_row == 0) {
                        tmp[0][0] += tail_d  * tail_sumi;
                        tmp[0][1] += tail_d1 * tail_sumi1;
                    }
                    continue;
                }
            }
            const int kbx = tid / (qi/vdr) + iter*blocks_per_iter;
            if constexpr (fixed_blocks % blocks_per_iter != 0) {
                if (kbx >= fixed_blocks) {
                    continue;
                }
            }
            const int kby = kbx * (qk/QK8_1);
            const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                if (!channel_valids[j]) {
                    continue;
                }
                const int kbx_offset = sample_x*stride_sample_x + channel_xs[j]*stride_channel_x + row0*stride_row_x;
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    if constexpr (iq_shared_grid) {
                        tmp[j][i] += vec_dot_iq_shared_grid<type, iq2xxs_levels>(
                            vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs, iq_grid_src);
                        if constexpr (has_fusion) {
                            if (use_gate) {
                                tmp_gate[j][i] += vec_dot_iq_shared_grid<type, iq2xxs_levels>(
                                    vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs,
                                    iq_grid_src);
                            }
                        }
                    } else {
                        tmp[j][i] += vec_dot_q_mmvq<type, c_fp3_packed24>(
                            vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                        if constexpr (has_fusion) {
                            if (use_gate) {
                                tmp_gate[j][i] += vec_dot_q_mmvq<type, c_fp3_packed24>(
                                    vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                            }
                        }
                    }
                }
            }
        }
    } else if constexpr (unroll_k_loop_2) {
#pragma unroll 2
        for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                if (!channel_valids[j]) {
                    continue;
                }
                const int kbx_offset = sample_x*stride_sample_x + channel_xs[j]*stride_channel_x + row0*stride_row_x;
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    tmp[j][i] += vec_dot_q_mmvq<type, c_fp3_packed24>(
                        vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_gate[j][i] += vec_dot_q_mmvq<type, c_fp3_packed24>(
                                vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                        }
                    }
                }
            }
        }
    } else {
        for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            const int kqs = vdr * (tid % (qi/vdr));

            if constexpr (c_fp4_x4) {
                const int kbx_offset = sample_x*stride_sample_x +
                    channel_xs[0]*stride_channel_x + row0*stride_row_x;
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const float4 dots = vec_dot_rocmfp4_fast_q8_1_x4(
                        vx,
                        &y[0*stride_col_y + kby],
                        &y[1*stride_col_y + kby],
                        &y[2*stride_col_y + kby],
                        &y[3*stride_col_y + kby],
                        kbx_offset + i*stride_row_x + kbx, kqs);
                    tmp[0][i] += dots.x;
                    tmp[1][i] += dots.y;
                    tmp[2][i] += dots.z;
                    tmp[3][i] += dots.w;
                    if constexpr (ncols_dst == 5) {
                        tmp[4][i] += vec_dot_q_mmvq<type, false>(
                            vx, &y[4*stride_col_y + kby],
                            kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            const float4 gate_dots =
                                vec_dot_rocmfp4_fast_q8_1_x4(
                                    vgate,
                                    &y[0*stride_col_y + kby],
                                    &y[1*stride_col_y + kby],
                                    &y[2*stride_col_y + kby],
                                    &y[3*stride_col_y + kby],
                                    kbx_offset + i*stride_row_x + kbx,
                                    kqs);
                            tmp_gate[0][i] += gate_dots.x;
                            tmp_gate[1][i] += gate_dots.y;
                            tmp_gate[2][i] += gate_dots.z;
                            tmp_gate[3][i] += gate_dots.w;
                            if constexpr (ncols_dst == 5) {
                                tmp_gate[4][i] +=
                                    vec_dot_q_mmvq<type, false>(
                                        vgate, &y[4*stride_col_y + kby],
                                        kbx_offset + i*stride_row_x + kbx,
                                        kqs);
                            }
                        }
                    }
                }
                continue;
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                if (!channel_valids[j]) {
                    continue;
                }
                const int kbx_offset = sample_x*stride_sample_x + channel_xs[j]*stride_channel_x + row0*stride_row_x;
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    tmp[j][i] += vec_dot_q_mmvq<type, c_fp3_packed24>(
                        vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_gate[j][i] += vec_dot_q_mmvq<type, c_fp3_packed24>(
                                vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                        }
                    }
                }
            }
        }
    }

    // A one-wave specialization has no cross-wave partials. Avoid reserving
    // shared memory and issuing a block barrier in that common decode path.
    // This does not change the per-lane accumulation or warp reduction order.
    if constexpr (nwarps > 1) {
        __shared__ float tmp_shared[nwarps-1][ncols_dst][rows_per_cuda_block][warp_size];
        __shared__ float tmp_shared_gate[has_fusion ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
        if constexpr (!has_fusion) {
            (void) tmp_shared_gate;
        } else if (!use_gate) {
            (void) tmp_shared_gate;
        }

        if (threadIdx.y > 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                        }
                    }
                }
            }
        }

        __syncthreads();
        if (threadIdx.y > 0) {
            return;
        }

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
                for (int l = 0; l < nwarps-1; ++l) {
                    tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                        }
                    }
                }
            }
        }
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            if constexpr (iq_shared_grid) {
                tmp[j][i] = warp_reduce_sum_wave32_dpp(tmp[j][i]);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] = warp_reduce_sum_wave32_dpp(tmp_gate[j][i]);
                    }
                }
            } else {
                tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                    }
                }
            }
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            float result = 0.0f;
            if (channel_valids[j]) {
                result = tmp[j][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_bias) {
                        result += x_biases[j];
                    }
                    if (use_gate) {
                        float gate_value = tmp_gate[j][threadIdx.x];
                        if (use_gate_bias) {
                            gate_value += gate_biases[j];
                        }
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI: {
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result, glu_param0, glu_param1);
                                break;
                            }
                            case GGML_GLU_OP_SWIGLU_DS4: {
                                if (gate_value_scale != 1.0f) {
                                    gate_value *= gate_value_scale;
                                }
                                if (x_value_scale != 1.0f) {
                                    result *= x_value_scale;
                                }
                                result = ggml_cuda_op_swiglu_ds4_single(gate_value, result, glu_param0);
                                break;
                            }
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
            }
            dst[j*stride_col_dst + threadIdx.x] = result;
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, active_glu, glu_param0, glu_param1, gate_bias, x_bias, tmp_gate);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, int c_warp_groups,
          bool has_fusion, bool sparse_warp_blocks,
          bool c_fp3_packed24 = false, bool c_fp2_packed32 = false,
          bool c_fp2_prefetch = false>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids,
        const ggml_cuda_mm_fusion_args_device fusion,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t nchannels_dst, const uint32_t ids_stride,
        const bool compact_masked_ids, const bool aligned_shared_ids) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);
    static_assert(c_warp_groups == 1 || c_warp_groups == 2,
                  "unsupported MoE warp-group count");

    // Hybrid expert ownership leaves negative IDs in either the hot or cold
    // graph.  The original (warp_size, n_tokens) block keeps invalid token
    // warps resident beside valid ones.  The sparse-owner variant gives every
    // token/route pair its own one-warp block so invalid routes retire without
    // consuming the valid block's occupancy.  The per-warp K traversal and
    // accumulation order are unchanged.
    const uint32_t token_idx = sparse_warp_blocks
        ? (uint32_t) blockIdx.y % ncols_dst
        : (uint32_t) threadIdx.y % ncols_dst;
    const uint32_t row_group = sparse_warp_blocks
        ? 0u
        : (uint32_t) threadIdx.y / ncols_dst;
    const int row0 = c_rows_per_block *
        ((int) blockIdx.x * c_warp_groups + (int) row_group);
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t compact_slot = sparse_warp_blocks
        ? (uint32_t) blockIdx.y / ncols_dst
        : (uint32_t) blockIdx.y;

    // The final grid block can contain a row group whose first output row is
    // already outside nrows_x.  This branch is uniform within that warp and
    // the kernel has no block-wide synchronization.
    if (token_idx >= ncols_dst || (uint32_t) row0 >= nrows_x) {
        return;
    }

    // Pack owner-valid routes into the low block slots without changing the
    // logical output layout.  In the heterogeneous graph each owner receives
    // the same six route positions with non-owned IDs masked to -1.  Leaving
    // those holes interleaved makes nearly every (route, q-token) block carry
    // at least one valid warp, so invalid warps occupy the Strix wave beside
    // useful work.  The stable permutation below executes valid routes first
    // but writes every result back to its original route position.  Down MMVQ
    // and route weighting therefore retain their original indexing and
    // floating-point reduction order.
    uint32_t channel_dst = compact_slot;
    int32_t channel_x_raw = -1;
    const int32_t encoded_id = ids[compact_slot + token_idx * ids_stride];
    const bool has_aligned_id = aligned_shared_ids &&
        (((uint32_t) encoded_id & 0x7f000000u) == 0x5a000000u);
    if (has_aligned_id) {
        channel_dst = ((uint32_t) encoded_id >> 16) & 0xffu;
        if (encoded_id >= 0) {
            channel_x_raw = encoded_id & 0xffff;
        } else {
            channel_x_raw = -1;
        }
    } else if (compact_masked_ids) {
        uint32_t n_valid = 0;
        for (uint32_t c = 0; c < nchannels_dst; ++c) {
            const int32_t id = ids[c + token_idx * ids_stride];
            n_valid += id >= 0;
        }

        const bool want_valid = compact_slot < n_valid;
        uint32_t rank = want_valid ? compact_slot : compact_slot - n_valid;
        for (uint32_t c = 0; c < nchannels_dst; ++c) {
            const int32_t id = ids[c + token_idx * ids_stride];
            const bool valid = id >= 0;
            if (valid == want_valid) {
                if (rank == 0) {
                    channel_dst = c;
                    channel_x_raw = valid ? id : -1;
                    break;
                }
                --rank;
            }
        }
    } else {
        channel_x_raw = ids[channel_dst + token_idx * ids_stride];
    }
    // Expert IDs alone cannot distinguish a zero-weight padding duplicate
    // from a valid repeated route. Never suppress a route without its weight.
    const bool channel_valid = channel_x_raw >= 0;
    const uint32_t channel_x = channel_valid ? (uint32_t) channel_x_raw : 0u;
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    ggml_glu_op active_glu;
    float glu_param0 = 0.0f;
    float glu_param1 = 0.0f;
    float gate_value_scale = 1.0f;
    float x_value_scale = 1.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_param0    = fusion.glu_param0;
        glu_param1    = fusion.glu_param1;
        gate_value_scale = fusion.gate_value_scale;
        x_value_scale = fusion.x_value_scale;
    }

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    if constexpr (type == GGML_TYPE_Q2_0_ROCMFP2 && c_fp2_prefetch) {
        if (channel_valid) {
            mul_mat_vec_q_moe_fp2_prefetch<c_rows_per_block, has_fusion>(
                vx, vgate, use_gate, y, kbx_offset, stride_row_x,
                blocks_per_row_x, tmp, tmp_gate);
        }
    } else if (channel_valid) {
        for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
            for (int i = 0; i < c_rows_per_block; ++i) {
                if constexpr (type == GGML_TYPE_Q2_0_ROCMFP2 &&
                              c_fp2_packed32) {
                    tmp[i] += vec_dot_rocmfpx_fp2_q8_1_packed32(
                        vx, &y[kby], kbx_offset + i*stride_row_x + kbx,
                        kqs);
                } else if constexpr (type == GGML_TYPE_Q3_0_ROCMFPX &&
                              c_fp3_packed24) {
                    tmp[i] += vec_dot_rocmfpx_fp3_q8_1_packed24(
                        vx, &y[kby], kbx_offset + i*stride_row_x + kbx,
                        kqs);
                } else {
                    tmp[i] += vec_dot_q_cuda(
                        vx, &y[kby], kbx_offset + i*stride_row_x + kbx,
                        kqs);
                }
                if constexpr (has_fusion) {
                    if (use_gate) {
                        if constexpr (type == GGML_TYPE_Q2_0_ROCMFP2 &&
                                      c_fp2_packed32) {
                            tmp_gate[i] += vec_dot_rocmfpx_fp2_q8_1_packed32(
                                vgate, &y[kby],
                                kbx_offset + i*stride_row_x + kbx, kqs);
                        } else if constexpr (type == GGML_TYPE_Q3_0_ROCMFPX &&
                                      c_fp3_packed24) {
                            tmp_gate[i] += vec_dot_rocmfpx_fp3_q8_1_packed24(
                                vgate, &y[kby],
                                kbx_offset + i*stride_row_x + kbx, kqs);
                        } else {
                            tmp_gate[i] += vec_dot_q_cuda(
                                vgate, &y[kby],
                                kbx_offset + i*stride_row_x + kbx, kqs);
                        }
                    }
                }
            }
        }
    }

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = 0.0f;
        if (channel_valid) {
            result = tmp[threadIdx.x];
            if constexpr (has_fusion) {
                if (use_bias) {
                    result += x_bias[channel_x*stride_channel_dst + row0 + threadIdx.x];
                }
                if (use_gate) {
                    float gate_value = tmp_gate[threadIdx.x];
                    if (use_gate_bias) {
                        gate_value += gate_bias[channel_x*stride_channel_dst + row0 + threadIdx.x];
                    }
                    switch (active_glu) {
                        case GGML_GLU_OP_SWIGLU:
                            result *= ggml_cuda_op_silu_single(gate_value);
                            break;
                        case GGML_GLU_OP_GEGLU:
                            result *= ggml_cuda_op_gelu_single(gate_value);
                            break;
                        case GGML_GLU_OP_SWIGLU_OAI:
                            result = ggml_cuda_op_swiglu_oai_single(gate_value, result, glu_param0, glu_param1);
                            break;
                        case GGML_GLU_OP_SWIGLU_DS4:
                            if (gate_value_scale != 1.0f) {
                                gate_value *= gate_value_scale;
                            }
                            if (x_value_scale != 1.0f) {
                                result *= x_value_scale;
                            }
                            result = ggml_cuda_op_swiglu_ds4_single(gate_value, result, glu_param0);
                            break;
                        default:
                            result = result * gate_value;
                            break;
                    }
                }
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, vgate, x_bias, gate_bias, active_glu, glu_param0, glu_param1, tmp_gate);
    }
}


// [TAG_MMID_GROUPED] prep: rank-sort the (token, slot) pairs of a MUL_MAT_ID
// batch by routed expert. meta holds, per sorted pair p: ge[p] = expert,
// pt[p] = token, ps[p] = slot. Warps processing consecutive sorted pairs then
// share an expert, which is what makes the grouped kernel dedup weight reads.
// Single block, O(np^2) rank sort: np <= 256, ~microseconds, capture-safe.
static __global__ void mmid_group_prep(
        int32_t * __restrict__ ids, int32_t * __restrict__ meta,
        const int n_slots, const int n_tok, const int ids_stride,
        float * __restrict__ gate_w, const int gate_w_stride, const float gate_tau) {
    __shared__ int sh_expert[MMID_GROUPED_MAX_PAIRS];
    const int np = n_slots*n_tok;
    const int i  = threadIdx.x;
    // [TAG_MMID_ADAPTIVE_K] one thread per token: keep slots until cumulative
    // combine weight >= tau, sentinel the rest, renormalize kept in place.
    // A row is "already gated" (second/third mmid of the layer) when a slot
    // weight is exactly 0.0f: `ids` is now a per-op scratch copy, so the
    // shared zeroed weights are the only marker that survives across ops.
    if (gate_w != nullptr && i < n_tok) {
        int32_t * idrow = ids + i*ids_stride;
        float   * wrow  = gate_w + i*gate_w_stride;
        bool gated = false;
        for (int j = 0; j < n_slots; ++j) {
            gated = gated || (idrow[j] < 0) || (wrow[j] == 0.0f);
        }
        if (!gated) {
            float cum = 0.0f;
            int   cut = n_slots;
            float total = 0.0f;
            for (int j = 0; j < n_slots; ++j) {
                total += wrow[j];
            }
            for (int j = 0; j < n_slots; ++j) {
                cum += wrow[j];
                if (cum >= gate_tau*total) { cut = j + 1; break; }
            }
            if (cut < n_slots) {
                const float inv = total/cum;
                for (int j = 0; j < n_slots; ++j) {
                    if (j < cut) {
                        wrow[j] *= inv;
                    } else {
                        idrow[j] = -1;
                        wrow[j]  = 0.0f;
                    }
                }
            }
        }
    }
    __syncthreads();
    int e = 0;
    if (i < np) {
        e = ids[(i % n_slots) + (i / n_slots)*ids_stride];
        sh_expert[i] = e < 0 ? 0x7FFFFFFF : e;  // sentinels sort last
        e = sh_expert[i];
    }
    __syncthreads();
    if (i < np) {
        int r = 0;
        for (int j = 0; j < np; ++j) {
            const int ej = sh_expert[j];
            r += (ej < e || (ej == e && j < i)) ? 1 : 0;
        }
        meta[MMID_META_GE + r] = e == 0x7FFFFFFF ? -1 : e;
        meta[MMID_META_PT + r] = i / n_slots;
        meta[MMID_META_PS + r] = i % n_slots;
    }
}

// [TAG_MMID_GROUPED] grouped MoE kernel. Identical launch shape and per-warp
// structure to mul_mat_vec_q_moe (block (warp_size, tokens_per_group),
// c_rows_per_block rows per warp, same vec_dot sequence and warp reduction),
// so results are bit-exact vs that kernel. The only difference: warp w
// handles the expert-SORTED pair blockIdx.y*tokens_per_group + w instead
// of (slot = blockIdx.y, token = w). Pairs routed to the same expert are
// adjacent after sorting, so the warps of a block mostly share one expert and
// their concurrent weight reads are served once from DRAM, then from L1/L2.
// Weight traffic approaches (union of routed experts) instead of
// (n_expert_used x n_tokens) expert-matrix reads.
template <ggml_type type, int c_rows_per_block, bool has_fusion,
          bool fp3_packed24 = false, bool fp2_packed32 = false,
          int tokens_per_group = MMID_GROUPED_DEFAULT_TPG>
__launch_bounds__(tokens_per_group*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe_grouped(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ meta,
        const ggml_cuda_mm_fusion_args_device fusion,
        float * __restrict__ dst,
        const uint32_t np, const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);
    static_assert(!fp3_packed24 || type == GGML_TYPE_Q3_0_ROCMFPX,
                  "packed FP3 grouped dispatch requires ROCmFP3 weights");
    static_assert(!fp2_packed32 || type == GGML_TYPE_Q2_0_ROCMFP2,
                  "packed FP2 grouped dispatch requires ROCmFP2 weights");

    const uint32_t p = blockIdx.y*tokens_per_group + threadIdx.y;
    if (p >= np) {
        return;
    }

    const int channel_x = meta[MMID_META_GE + p];
    const int tok       = meta[MMID_META_PT + p];
    const int slot      = meta[MMID_META_PS + p];

    const int row0             = c_rows_per_block*blockIdx.x;

    // [TAG_MMID_ADAPTIVE_K] dropped slot: contribute exact zeros (combine
    // weight is zeroed too, but dst must not hold garbage).
    if (channel_x < 0) {
        if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
            dst[slot*stride_channel_dst + tok*stride_col_dst + row0 + threadIdx.x] = 0.0f;
        }
        return;
    }
    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    const uint32_t channel_y = fastmodulo((uint32_t) slot, nchannels_y);
    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + tok*stride_col_y;
    const int kbx_offset = channel_x*stride_channel_x + row0*stride_row_x;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    ggml_glu_op active_glu;
    float glu_param0 = 0.0f;
    float glu_param1 = 0.0f;
    float gate_value_scale = 1.0f;
    float x_value_scale = 1.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_param0    = fusion.glu_param0;
        glu_param1    = fusion.glu_param1;
        gate_value_scale = fusion.gate_value_scale;
        x_value_scale = fusion.x_value_scale;
    }

    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            if constexpr (fp2_packed32) {
                tmp[i] += vec_dot_rocmfpx_fp2_q8_1_packed32(
                    vx, &y[kby],
                    kbx_offset + i*stride_row_x + kbx, kqs);
            } else if constexpr (fp3_packed24) {
                tmp[i] += vec_dot_rocmfpx_fp3_q8_1_packed24(
                    vx, &y[kby],
                    kbx_offset + i*stride_row_x + kbx, kqs);
            } else {
                tmp[i] += vec_dot_q_cuda(
                    vx, &y[kby],
                    kbx_offset + i*stride_row_x + kbx, kqs);
            }
            if constexpr (has_fusion) {
                if (use_gate) {
                    if constexpr (fp2_packed32) {
                        tmp_gate[i] += vec_dot_rocmfpx_fp2_q8_1_packed32(
                            vgate, &y[kby],
                            kbx_offset + i*stride_row_x + kbx, kqs);
                    } else if constexpr (fp3_packed24) {
                        tmp_gate[i] += vec_dot_rocmfpx_fp3_q8_1_packed24(
                            vgate, &y[kby],
                            kbx_offset + i*stride_row_x + kbx, kqs);
                    } else {
                        tmp_gate[i] += vec_dot_q_cuda(
                            vgate, &y[kby],
                            kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            if (use_bias) {
                result += x_bias[channel_x*stride_channel_dst + row0 + threadIdx.x];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if (use_gate_bias) {
                    gate_value += gate_bias[channel_x*stride_channel_dst + row0 + threadIdx.x];
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(
                            gate_value, result, glu_param0, glu_param1);
                        break;
                    case GGML_GLU_OP_SWIGLU_DS4:
                        if (gate_value_scale != 1.0f) {
                            gate_value *= gate_value_scale;
                        }
                        if (x_value_scale != 1.0f) {
                            result *= x_value_scale;
                        }
                        result = ggml_cuda_op_swiglu_ds4_single(
                            gate_value, result, glu_param0);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[slot*stride_channel_dst + tok*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, vgate, x_bias,
                         gate_bias, active_glu, glu_param0, glu_param1,
                         gate_value_scale, x_value_scale, tmp_gate);
    }
}

template <ggml_type type, bool fp3_packed24 = false,
          bool fp2_packed32 = false,
          int tokens_per_group = MMID_GROUPED_DEFAULT_TPG>
static void mul_mat_vec_q_moe_grouped_launch(
        const void * vx, const void * vy, const int32_t * meta, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const int np, const int warp_size, cudaStream_t stream) {

    constexpr int rows_per_block = 4;
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1)/rows_per_block;
    const dim3 block_nums(
        nblocks_rows, (np + tokens_per_group - 1)/tokens_per_group);
    const dim3 block_dims(warp_size, tokens_per_group);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if (has_fusion) {
        mul_mat_vec_q_moe_grouped<
            type, rows_per_block, true, fp3_packed24, fp2_packed32,
            tokens_per_group>
            <<<block_nums, block_dims, 0, stream>>>(
                vx, vy, meta, fusion, dst, (uint32_t) np, ncols_x,
                nchannels_y, nrows_x, stride_row_x, stride_col_y,
                stride_col_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst);
    } else {
        mul_mat_vec_q_moe_grouped<
            type, rows_per_block, false, fp3_packed24, fp2_packed32,
            tokens_per_group>
            <<<block_nums, block_dims, 0, stream>>>(
                vx, vy, meta, fusion, dst, (uint32_t) np, ncols_x,
                nchannels_y, nrows_x, stride_row_x, stride_col_y,
                stride_col_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst);
    }
}

static bool mul_mat_vec_q_grouped_dispatch(
        const ggml_type type, const void * vx, const void * vy, const int32_t * meta,
        const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int nchannels_y,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int max_groups, cudaStream_t stream) {

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const uint3 nchannels_y_fd = init_fastdiv_values((uint32_t) nchannels_y);
    switch (type) {
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_moe_grouped_launch<GGML_TYPE_Q5_0>(vx, vy, meta, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, max_groups, warp_size, stream);
            return true;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_moe_grouped_launch<GGML_TYPE_Q4_0>(vx, vy, meta, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, max_groups, warp_size, stream);
            return true;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_moe_grouped_launch<GGML_TYPE_Q8_0>(vx, vy, meta, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, max_groups, warp_size, stream);
            return true;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_moe_grouped_launch<GGML_TYPE_Q4_K>(vx, vy, meta, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, max_groups, warp_size, stream);
            return true;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_moe_grouped_launch<GGML_TYPE_Q5_K>(vx, vy, meta, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, max_groups, warp_size, stream);
            return true;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_moe_grouped_launch<GGML_TYPE_Q6_K>(vx, vy, meta, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, max_groups, warp_size, stream);
            return true;
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
            mul_mat_vec_q_moe_grouped_launch<
                GGML_TYPE_Q4_0_ROCMFP4_FAST>(
                    vx, vy, meta, fusion, dst, ncols_x, nchannels_y_fd,
                    nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                    stride_channel_x, stride_channel_y, stride_channel_dst,
                    max_groups, warp_size, stream);
            return true;
        case GGML_TYPE_Q2_0_ROCMFP2:
            if (mmvq_env_flag("LUCE_CUDA_MMVQ_MOE_FP2_PACKED32")) {
                mul_mat_vec_q_moe_grouped_launch<
                    GGML_TYPE_Q2_0_ROCMFP2, false, true>(
                        vx, vy, meta, fusion, dst, ncols_x, nchannels_y_fd,
                        nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst,
                        max_groups, warp_size, stream);
            } else {
                mul_mat_vec_q_moe_grouped_launch<
                    GGML_TYPE_Q2_0_ROCMFP2>(
                        vx, vy, meta, fusion, dst, ncols_x, nchannels_y_fd,
                        nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst,
                        max_groups, warp_size, stream);
            }
            return true;
        case GGML_TYPE_Q3_0_ROCMFPX: {
            const bool packed =
                mmvq_env_flag("LUCE_CUDA_MMVQ_MOE_FP3_PACKED24") &&
                std::getenv(
                    "LUCE_CUDA_MMVQ_MOE_FP3_PACKED24_RUNTIME_DISABLE") ==
                    nullptr;
            if (packed) {
                mul_mat_vec_q_moe_grouped_launch<
                    GGML_TYPE_Q3_0_ROCMFPX, true, false,
                    MMID_GROUPED_FP3_TPG>(
                        vx, vy, meta, fusion, dst, ncols_x, nchannels_y_fd,
                        nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst,
                        max_groups, warp_size, stream);
            } else {
                mul_mat_vec_q_moe_grouped_launch<
                    GGML_TYPE_Q3_0_ROCMFPX, false, false,
                    MMID_GROUPED_FP3_TPG>(
                        vx, vy, meta, fusion, dst, ncols_x, nchannels_y_fd,
                        nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst,
                        max_groups, warp_size, stream);
            }
            return true;
        }
        default:
            return false;
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false,
        const bool width_invariant = false) {
    const int shape_cols = width_invariant ? 1 : ncols_dst;
    const int nwarps = calc_nwarps(type, shape_cols, table_id);
    const int rpb = calc_rows_per_block(shape_cols, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false,
         int fixed_ncols_x = 0, bool unroll_k_loop_2 = false,
         bool reuse_rocmfp4_weights = false,
         bool fp3_packed24 = false, bool fp4_x4 = false,
         bool width_invariant = false, int row_warps = 1, int wave_rows = 1>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream, const bool ids_tokenwise_samples = false) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if (has_fusion) {
        mul_mat_vec_q<type, c_ncols_dst, true, small_k, fixed_ncols_x,
                      unroll_k_loop_2, reuse_rocmfp4_weights, fp3_packed24,
                      fp4_x4, width_invariant, row_warps, wave_rows><<<block_nums, block_dims, nbytes_shared, stream>>>
            (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
             channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
             sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, ids_tokenwise_samples);
        return;
    }

    mul_mat_vec_q<type, c_ncols_dst, false, small_k, fixed_ncols_x,
                  unroll_k_loop_2, reuse_rocmfp4_weights, fp3_packed24,
                  fp4_x4, width_invariant, row_warps, wave_rows><<<block_nums, block_dims, nbytes_shared, stream>>>
        (vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, ids_tokenwise_samples);
}

template <ggml_type type, int fixed_ncols_x, int row_warps = 1, int wave_rows = 1>
static void mul_mat_vec_rocmfpx_fixed_k_launch(
        const void * vx, const void * vy, const int32_t * ids,
        const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint3 nchannels_y,
        const uint32_t nchannels_dst, const uint32_t stride_row_x,
        const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t nsamples_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst, const uint32_t ids_stride,
        const int warp_size, cudaStream_t stream, const bool ids_tokenwise_samples = false) {
    static_assert(type == GGML_TYPE_Q3_0_ROCMFPX ||
                  type == GGML_TYPE_Q2_0_ROCMFP2 ||
                  type == GGML_TYPE_IQ2_XXS ||
                  type == GGML_TYPE_IQ3_XXS,
                  "fixed-K helper is limited to profiled ROCmFP2/FP3/IQ2_XXS/IQ3_XXS kernels");
    static_assert(fixed_ncols_x == 2048 || fixed_ncols_x == 4096 ||
                  fixed_ncols_x == 2304 || fixed_ncols_x == 5120,
                  "fixed-K helper compiled an unexpected DS4/DS4.1 shape");

    GGML_ASSERT(nrows_x % (row_warps*wave_rows) == 0);
    const dim3 block_nums(nrows_x / (row_warps*wave_rows), nchannels_dst, nsamples_dst);
    const dim3 block_dims(warp_size, row_warps, 1);
    mul_mat_vec_q_switch_fusion<type, 1, false, fixed_ncols_x,
                                false, false, false, false, false, row_warps, wave_rows>(
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x,
        stride_col_y, stride_col_dst, channel_ratio, stride_channel_x,
        stride_channel_y, stride_channel_dst, sample_ratio, stride_sample_x,
        stride_sample_y, stride_sample_dst, block_nums, block_dims, 0,
        ids_stride, stream, ids_tokenwise_samples);
}

// gfx1151 fixed-K single-column launches (profiled DS4 shapes), shared by the dispatcher below and
// mmvq_moe_matches_single_column.
static bool mmvq_rocmfp3_fixed_k(const int cc, const int ncols_x) {
    return is_gfx1151(cc) && ncols_x == 2048;
}

static bool mmvq_rocmfp2_fixed_k(const int cc, const int ncols_x) {
    return is_gfx1151(cc) && (ncols_x == 4096 || ncols_x == 2048);
}

static void mul_mat_vec_rocmfp4_unroll2_launch(
        const void * vx, const void * vy, const int32_t * ids,
        const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint3 nchannels_y,
        const uint32_t nchannels_dst, const uint32_t stride_row_x,
        const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t nsamples_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst, const uint32_t ids_stride,
        const int warp_size, cudaStream_t stream, const bool ids_tokenwise_samples = false) {
    const dim3 block_nums(nrows_x, nchannels_dst, nsamples_dst);
    const dim3 block_dims(warp_size, 1, 1);
    mul_mat_vec_q_switch_fusion<GGML_TYPE_Q4_0_ROCMFP4_FAST, 1, false, 0, true>(
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x,
        stride_col_y, stride_col_dst, channel_ratio, stride_channel_x,
        stride_channel_y, stride_channel_dst, sample_ratio, stride_sample_x,
        stride_sample_y, stride_sample_dst, block_nums, block_dims, 0,
        ids_stride, stream, ids_tokenwise_samples);
}

template <int ncols_dst>
static void mul_mat_vec_rocmfp4_reuse_launch(
        const void * vx, const void * vy,
        const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint3 nchannels_y,
        const uint32_t nchannels_dst, const uint32_t stride_row_x,
        const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t nsamples_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y,
        const uint32_t stride_sample_dst, const uint32_t ids_stride,
        const int warp_size, const mmvq_parameter_table_id table_id,
        cudaStream_t stream) {
    static_assert(ncols_dst >= 2 && ncols_dst <= ROCMFP4_REUSE_MAX_COLS,
                  "weight reuse covers two to sixteen dense columns");
    const auto dims = calc_launch_params<GGML_TYPE_Q4_0_ROCMFP4_FAST>(
        ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
    mul_mat_vec_q_switch_fusion<
        GGML_TYPE_Q4_0_ROCMFP4_FAST, ncols_dst, false, 0, false, true>(
            vx, vy, nullptr, fusion, dst, ncols_x, nchannels_y, stride_row_x,
            stride_col_y, stride_col_dst, channel_ratio, stride_channel_x,
            stride_channel_y, stride_channel_dst, sample_ratio, stride_sample_x,
            stride_sample_y, stride_sample_dst, dims.first, dims.second, 0,
            ids_stride, stream);
}

template <ggml_type type, int rows_per_block, int warp_groups = 1,
          bool fp3_packed24 = false, bool fp2_packed32 = false,
          bool fp2_prefetch = false>
static void mul_mat_vec_q_moe_launch_rpb(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream,
        const bool sparse_warp_blocks, const bool compact_masked_ids,
        const bool aligned_shared_ids) {

    static_assert(rows_per_block == 1 || rows_per_block == 2 ||
                  rows_per_block == 4 || rows_per_block == 8,
                  "unsupported MoE rows-per-block variant");
    static_assert(warp_groups == 1 || warp_groups == 2,
                  "unsupported MoE warp-group variant");
    const int active_warp_groups = sparse_warp_blocks ? 1 : warp_groups;
    const int64_t rows_per_grid_block =
        (int64_t) rows_per_block * active_warp_groups;
    const int64_t nblocks_rows =
        (nrows_x + rows_per_grid_block - 1) / rows_per_grid_block;
    const dim3 block_nums(
        nblocks_rows,
        sparse_warp_blocks ? nchannels_dst * ncols_dst : nchannels_dst);
    const dim3 block_dims(
        warp_size,
        sparse_warp_blocks ? 1 : ncols_dst * active_warp_groups);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if (has_fusion) {
        if (sparse_warp_blocks) {
            mul_mat_vec_q_moe<type, rows_per_block, 1, true, true,
                              fp3_packed24, fp2_packed32, fp2_prefetch><<<block_nums, block_dims, 0, stream>>>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, nchannels_dst, ids_stride,
                compact_masked_ids, aligned_shared_ids);
        } else {
            mul_mat_vec_q_moe<type, rows_per_block, warp_groups, true, false,
                              fp3_packed24, fp2_packed32, fp2_prefetch><<<block_nums, block_dims, 0, stream>>>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, nchannels_dst, ids_stride,
                compact_masked_ids, aligned_shared_ids);
        }
    } else {
        if (sparse_warp_blocks) {
            mul_mat_vec_q_moe<type, rows_per_block, 1, false, true,
                              fp3_packed24, fp2_packed32, fp2_prefetch><<<block_nums, block_dims, 0, stream>>>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, nchannels_dst, ids_stride,
                compact_masked_ids, aligned_shared_ids);
        } else {
            mul_mat_vec_q_moe<type, rows_per_block, warp_groups, false, false,
                              fp3_packed24, fp2_packed32, fp2_prefetch><<<block_nums, block_dims, 0, stream>>>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, nchannels_dst, ids_stride,
                compact_masked_ids, aligned_shared_ids);
        }
    }
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    static const bool sparse_warp_blocks = []() {
        const char * e = std::getenv("LUCE_CUDA_MMVQ_MOE_SPARSE_WARP_BLOCKS");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    static const bool compact_masked_ids = []() {
        const char * e = std::getenv("LUCE_CUDA_MMVQ_MOE_COMPACT_MASKED_IDS");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    static const bool aligned_shared_ids =
        mmvq_env_flag("LUCE_CUDA_MMVQ_MOE_ALIGN_SHARED_IDS");
    static const int configured_rows_per_block = []() {
        const char * e = std::getenv("LUCE_CUDA_MMVQ_MOE_ROWS_PER_BLOCK");
        if (!e || !e[0]) return 0;
        const int value = std::atoi(e);
        return value == 1 || value == 2 || value == 4 || value == 8
            ? value : 2;
    }();
    const bool gfx1151 = is_gfx1151(
        ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
    int tuned_rows_per_block =
        configured_rows_per_block > 0 ? configured_rows_per_block : 2;
    if (configured_rows_per_block == 0 && gfx1151 &&
        ncols_dst >= 2 && ncols_dst <= 5) {
        // Real DS4 shapes on wave32 prefer one output row per warp for fused
        // ROCmFP2 gate/up and unfused ROCmFP3 down.  Keep the opposite fusion
        // cases and unmeasured widths on the established two-row schedule.
        if constexpr (type == GGML_TYPE_Q2_0_ROCMFP2) {
            if (fusion.gate != nullptr) tuned_rows_per_block = 1;
        } else if constexpr (type == GGML_TYPE_Q3_0_ROCMFPX) {
            if (fusion.gate == nullptr) tuned_rows_per_block = 1;
        }
    }
    static const bool q2_warp_groups = []() {
        const char * e =
            std::getenv("LUCE_CUDA_MMVQ_MOE_Q2_WARP_GROUPS");
        return e && e[0] == '2' && e[1] == '\0';
    }();
    static const bool q4_warp_groups = []() {
        const char * e =
            std::getenv("LUCE_CUDA_MMVQ_MOE_Q4_WARP_GROUPS");
        return e && e[0] == '2' && e[1] == '\0';
    }();
    // Explicit LUCE_CUDA_MMVQ_MOE_FP3_PACKED24 wins; unset defaults to the
    // packed kernel on gfx1151 only.
    static const int fp3_packed24_setting = []() {
        const char * e =
            std::getenv("LUCE_CUDA_MMVQ_MOE_FP3_PACKED24");
        if (e == nullptr) return -1;
        return (e[0] == '1' && e[1] == '\0') ? 1 : 0;
    }();
    const bool fp3_packed24 =
        (fp3_packed24_setting >= 0 ? fp3_packed24_setting == 1
                                   : gfx1151) &&
        std::getenv("LUCE_CUDA_MMVQ_MOE_FP3_PACKED24_RUNTIME_DISABLE") == nullptr;
    static const bool fp2_packed32 = []() {
        const char * e =
            std::getenv("LUCE_CUDA_MMVQ_MOE_FP2_PACKED32");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    // Explicit LUCE_CUDA_MMVQ_MOE_FP2_PREFETCH wins; unset defaults to the
    // prefetching ROCmFP2 loop on gfx1151 only.
    static const int fp2_prefetch_setting = []() {
        const char * e =
            std::getenv("LUCE_CUDA_MMVQ_MOE_FP2_PREFETCH");
        if (e == nullptr) return -1;
        return (e[0] == '1' && e[1] == '\0') ? 1 : 0;
    }();
    const bool fp2_prefetch =
        fp2_prefetch_setting >= 0 ? fp2_prefetch_setting == 1 : gfx1151;
    GGML_UNUSED(fp2_prefetch);

#define GGML_MOE_LAUNCH_RPB(RPB) \
    mul_mat_vec_q_moe_launch_rpb<type, RPB, 1>( \
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, \
        stride_row_x, stride_col_y, stride_col_dst, \
        stride_channel_x, stride_channel_y, stride_channel_dst, \
        ncols_dst, ids_stride, warp_size, nchannels_dst, stream, \
        sparse_warp_blocks, compact_masked_ids, aligned_shared_ids)

#define GGML_MOE_LAUNCH_TWO_WARP_GROUPS() \
    mul_mat_vec_q_moe_launch_rpb<type, 2, 2>( \
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, \
        stride_row_x, stride_col_y, stride_col_dst, \
        stride_channel_x, stride_channel_y, stride_channel_dst, \
        ncols_dst, ids_stride, warp_size, nchannels_dst, stream, \
        sparse_warp_blocks, compact_masked_ids, aligned_shared_ids)

#define GGML_MOE_LAUNCH_FP3_PACKED24(RPB, WARP_GROUPS) \
    mul_mat_vec_q_moe_launch_rpb<type, RPB, WARP_GROUPS, true>( \
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, \
        stride_row_x, stride_col_y, stride_col_dst, \
        stride_channel_x, stride_channel_y, stride_channel_dst, \
        ncols_dst, ids_stride, warp_size, nchannels_dst, stream, \
        sparse_warp_blocks, compact_masked_ids, aligned_shared_ids)

#define GGML_MOE_LAUNCH_FP2_PACKED32(RPB, WARP_GROUPS) \
    mul_mat_vec_q_moe_launch_rpb<type, RPB, WARP_GROUPS, false, true>( \
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, \
        stride_row_x, stride_col_y, stride_col_dst, \
        stride_channel_x, stride_channel_y, stride_channel_dst, \
        ncols_dst, ids_stride, warp_size, nchannels_dst, stream, \
        sparse_warp_blocks, compact_masked_ids, aligned_shared_ids)

#define GGML_MOE_LAUNCH_FP2_PREFETCH(RPB, WARP_GROUPS) \
    mul_mat_vec_q_moe_launch_rpb<type, RPB, WARP_GROUPS, false, false, true>( \
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x, \
        stride_row_x, stride_col_y, stride_col_dst, \
        stride_channel_x, stride_channel_y, stride_channel_dst, \
        ncols_dst, ids_stride, warp_size, nchannels_dst, stream, \
        sparse_warp_blocks, compact_masked_ids, aligned_shared_ids)

    // The affine FP2 dot reads more than the packed word and scale byte, so
    // that layout keeps the generic loop.
#ifndef ROCMFP2_AFFINE
    if constexpr (type == GGML_TYPE_Q2_0_ROCMFP2) {
        if (fp2_prefetch &&
            (tuned_rows_per_block == 1 || tuned_rows_per_block == 2) &&
            !sparse_warp_blocks) {
            if (tuned_rows_per_block == 1) {
                GGML_MOE_LAUNCH_FP2_PREFETCH(1, 1);
            } else if ((q2_warp_groups && ncols_dst == 2) ||
                (q4_warp_groups && ncols_dst == 4)) {
                GGML_MOE_LAUNCH_FP2_PREFETCH(2, 2);
            } else {
                GGML_MOE_LAUNCH_FP2_PREFETCH(2, 1);
            }
            return;
        }
    }
#endif // ROCMFP2_AFFINE

    if constexpr (type == GGML_TYPE_Q2_0_ROCMFP2) {
        if (fp2_packed32 &&
            (tuned_rows_per_block == 1 || tuned_rows_per_block == 2) &&
            !sparse_warp_blocks) {
            if (tuned_rows_per_block == 1) {
                GGML_MOE_LAUNCH_FP2_PACKED32(1, 1);
            } else if ((q2_warp_groups && ncols_dst == 2) ||
                (q4_warp_groups && ncols_dst == 4)) {
                GGML_MOE_LAUNCH_FP2_PACKED32(2, 2);
            } else {
                GGML_MOE_LAUNCH_FP2_PACKED32(2, 1);
            }
            return;
        }
    }

    if constexpr (type == GGML_TYPE_Q3_0_ROCMFPX) {
        if (fp3_packed24 &&
            (tuned_rows_per_block == 1 || tuned_rows_per_block == 2) &&
            !sparse_warp_blocks) {
            if (tuned_rows_per_block == 1) {
                GGML_MOE_LAUNCH_FP3_PACKED24(1, 1);
            } else if ((q2_warp_groups && ncols_dst == 2) ||
                (q4_warp_groups && ncols_dst == 4)) {
                GGML_MOE_LAUNCH_FP3_PACKED24(2, 2);
            } else {
                GGML_MOE_LAUNCH_FP3_PACKED24(2, 1);
            }
            return;
        }
    }

    if constexpr (type == GGML_TYPE_Q2_0_ROCMFP2 ||
                  type == GGML_TYPE_Q3_0_ROCMFPX) {
        if (((q2_warp_groups && ncols_dst == 2) ||
             (q4_warp_groups && ncols_dst == 4)) &&
            tuned_rows_per_block == 2 && !sparse_warp_blocks) {
            GGML_MOE_LAUNCH_TWO_WARP_GROUPS();
        } else {
            switch (tuned_rows_per_block) {
                case 1: GGML_MOE_LAUNCH_RPB(1); break;
                case 4: GGML_MOE_LAUNCH_RPB(4); break;
                case 8: GGML_MOE_LAUNCH_RPB(8); break;
                default: GGML_MOE_LAUNCH_RPB(2); break;
            }
        }
    } else {
        GGML_MOE_LAUNCH_RPB(2);
    }
#undef GGML_MOE_LAUNCH_TWO_WARP_GROUPS
#undef GGML_MOE_LAUNCH_FP2_PACKED32
#undef GGML_MOE_LAUNCH_FP2_PREFETCH
#undef GGML_MOE_LAUNCH_FP3_PACKED24
#undef GGML_MOE_LAUNCH_RPB
}


#if defined(GGML_USE_HIP)
// RDNA4 Q8_0 dense MMVQ (HIP builds only).
//
// On RDNA4 the one-column Q8_0 kernel, and the batch-invariant multi-column
// launch that keeps its block shape (DSpark verify), run eight waves per
// output row: wave w, lane l takes quant block 8*w + l/4 (+64 per K step),
// quarter l%4, accumulates one FMA chain per lane, and the waves' partials
// are added in wave order before the lane butterfly.  For DS4.1 projections
// (K = 1280..8192) that is one to four K steps per eight-wave block, so the
// time goes into block launch, the LDS exchange and the barrier, and every
// column's activations are reloaded for every row.
//
// This kernel computes exactly the same arithmetic: each lane keeps the
// partial of every virtual wave it covers (c_vw physical waves share the
// eight virtual waves), sums them in virtual-wave order and runs the same
// butterfly.  Each physical wave covers c_rows rows that share one load of
// each activation fragment.  Every output is bit-identical to the eight-wave
// kernel at every width, so decode and verify stay consistent.
#define MMVQ_Q8_RDNA4_MAX_COLS 8

// glu: vx is the up matrix, vx_gate the gate matrix of the same shape; each
// product keeps the single kernel's partials and reduction, and dst gets
// ggml_cuda_op_swiglu_ds4_single(gate, up, glu_limit), the GLU kernel's value.
template <int ncols_dst, int c_rows, int c_vw, int c_row_groups, bool glu = false>
__launch_bounds__(32*c_vw*c_row_groups, 1)
static __global__ void mul_mat_vec_q8_0_rdna4(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y,
        const int stride_col_dst, const uint3 channel_ratio, const int stride_channel_x,
        const int stride_channel_y, const int stride_channel_dst, const uint3 sample_ratio,
        const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int kb_off, const void * __restrict__ vx_gate, const float glu_limit) {
    constexpr int nvirt = 8;               // waves of the reference block
    constexpr int nq    = nvirt / c_vw;    // virtual waves per physical wave
    static_assert(nvirt % c_vw == 0, "c_vw must divide eight");

    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int g    = wave % c_vw;
    const int rg   = wave / c_vw;
    const int quarter = lane & 3;
    const int boff    = lane >> 2;
    const int row_base = (blockIdx.x*c_row_groups + rg)*c_rows;
    const int nb = ncols_x / QK8_0;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    const block_q8_0 * x = ((const block_q8_0 *) vx) + sample_x*stride_sample_x + channel_x*stride_channel_x;

    const block_q8_0 * xg = glu ? ((const block_q8_0 *) vx_gate) + sample_x*stride_sample_x + channel_x*stride_channel_x : x;

    const block_q8_0 * xr[c_rows];
    const block_q8_0 * xgr[c_rows];
    bool row_ok[c_rows];
#pragma unroll
    for (int r = 0; r < c_rows; ++r) {
        const int row = row_base + r;
        row_ok[r] = row < nrows_x;
        xr[r]  = x  + (int64_t) (row_ok[r] ? row : 0)*stride_row_x;
        xgr[r] = xg + (int64_t) (row_ok[r] ? row : 0)*stride_row_x;
    }

    float part[c_rows][ncols_dst][nq];
    float partg[glu ? c_rows : 1][ncols_dst][nq];
#pragma unroll
    for (int r = 0; r < c_rows; ++r)
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j)
#pragma unroll
            for (int q = 0; q < nq; ++q) {
                part[r][j][q] = 0.0f;
                if constexpr (glu) partg[r][j][q] = 0.0f;
            }

    // kb_off > 0: the rows are a column range starting at block kb_off of
    // wider rows (ggml_backend_cuda_mul_mat_whole_row_lanes). Each block takes
    // the lane and K step it takes in the whole rows, so every partial adds
    // the same terms in the same order as the whole rows times a src1 that is
    // zero outside the range: those products only add exact zeros.
    for (int k0 = (kb_off/64)*64; k0 < kb_off + nb; k0 += 64) {
#pragma unroll
        for (int q = 0; q < nq; ++q) {
            const int kbx = k0 + 8*(g + c_vw*q) + boff - kb_off;
            if (kbx >= 0 && kbx < nb) {
                int   u0[ncols_dst];
                int   u1[ncols_dst];
                float dy[ncols_dst];
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const block_q8_1 * by = y + j*stride_col_y + kbx;
                    u0[j] = get_int_b4(by->qs, 2*quarter + 0);
                    u1[j] = get_int_b4(by->qs, 2*quarter + 1);
                    dy[j] = __low2half(by->ds);
                }
#pragma unroll
                for (int r = 0; r < c_rows; ++r) {
                    const block_q8_0 * bx = xr[r] + kbx;
                    const int   v0 = get_int_b2(bx->qs, 2*quarter + 0);
                    const int   v1 = get_int_b2(bx->qs, 2*quarter + 1);
                    const float dx = bx->d;
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        int sumi = 0;
                        sumi = ggml_cuda_dp4a(v0, u0[j], sumi);
                        sumi = ggml_cuda_dp4a(v1, u1[j], sumi);
                        part[r][j][q] += dx*dy[j] * ((float) sumi);
                    }
                    if constexpr (glu) {
                        const block_q8_0 * bg = xgr[r] + kbx;
                        const int   g0 = get_int_b2(bg->qs, 2*quarter + 0);
                        const int   g1 = get_int_b2(bg->qs, 2*quarter + 1);
                        const float dg = bg->d;
#pragma unroll
                        for (int j = 0; j < ncols_dst; ++j) {
                            int sumg = 0;
                            sumg = ggml_cuda_dp4a(g0, u0[j], sumg);
                            sumg = ggml_cuda_dp4a(g1, u1[j], sumg);
                            partg[r][j][q] += dg*dy[j] * ((float) sumg);
                        }
                    }
                }
            }
        }
    }

    float sum[c_rows][ncols_dst];
    float sumg[glu ? c_rows : 1][ncols_dst];
    if constexpr (c_vw == 1) {
#pragma unroll
        for (int r = 0; r < c_rows; ++r)
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                float t = part[r][j][0];
#pragma unroll
                for (int q = 1; q < nvirt; ++q) {
                    t += part[r][j][q];
                }
                sum[r][j] = t;
                if constexpr (glu) {
                    float tg = partg[r][j][0];
#pragma unroll
                    for (int q = 1; q < nvirt; ++q) {
                        tg += partg[r][j][q];
                    }
                    sumg[r][j] = tg;
                }
            }
    } else {
        __shared__ float sh[c_row_groups][c_vw][c_rows][ncols_dst][nq][32];
        __shared__ float shg[glu ? c_row_groups : 1][glu ? c_vw : 1][glu ? c_rows : 1][ncols_dst][glu ? nq : 1][32];
#pragma unroll
        for (int r = 0; r < c_rows; ++r)
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j)
#pragma unroll
                for (int q = 0; q < nq; ++q) {
                    sh[rg][g][r][j][q][lane] = part[r][j][q];
                    if constexpr (glu) shg[rg][g][r][j][q][lane] = partg[r][j][q];
                }
        __syncthreads();
        if (g != 0) {
            return;
        }
#pragma unroll
        for (int r = 0; r < c_rows; ++r)
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                float t = sh[rg][0][r][j][0][lane];
#pragma unroll
                for (int v = 1; v < nvirt; ++v) {
                    t += sh[rg][v % c_vw][r][j][v / c_vw][lane];
                }
                sum[r][j] = t;
                if constexpr (glu) {
                    float tg = shg[rg][0][r][j][0][lane];
#pragma unroll
                    for (int v = 1; v < nvirt; ++v) {
                        tg += shg[rg][v % c_vw][r][j][v / c_vw][lane];
                    }
                    sumg[r][j] = tg;
                }
            }
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst;
#pragma unroll
    for (int r = 0; r < c_rows; ++r) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const float t = warp_reduce_sum<32>(sum[r][j]);
            if constexpr (glu) {
                const float tg = warp_reduce_sum<32>(sumg[r][j]);
                if (lane == 0 && row_ok[r]) {
                    dst[j*stride_col_dst + row_base + r] = ggml_cuda_op_swiglu_ds4_single(tg, t, glu_limit);
                }
            } else if (lane == 0 && row_ok[r]) {
                dst[j*stride_col_dst + row_base + r] = t;
            }
        }
    }
}

static bool mmvq_q8_0_rdna4_enabled() {
    static const bool enabled = []() {
        const char * e = std::getenv("LUCE_CUDA_MMVQ_Q8_RDNA4");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    return enabled;
}

// A Q8_0 gate/up pair (two MUL_MATs on the same activations) whose SwiGLU-DS4
// reads exactly their outputs, at 2..8 columns of batch-invariant products on
// RDNA4: ggml_cuda_mul_mat_vec_q with the gate as fusion runs it as one launch
// of mul_mat_vec_q8_0_rdna4, bit-identical to the two products and the GLU
// kernel (the caller rules out split buffers). LUCE_CUDA_MMVQ_Q8_RDNA4_GLU=0
// keeps three launches.
bool ggml_cuda_mmvq_rdna4_glu_pair(const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * glu) {
    static const bool enabled = []() {
        const char * e = std::getenv("LUCE_CUDA_MMVQ_Q8_RDNA4_GLU");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    if (!enabled || !mmvq_q8_0_rdna4_enabled() || !ggml_cuda_mmvq_batch_invariant()) return false;
    const int device = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc) || ggml_cuda_info().devices[device].warp_size != 32 ||
        get_device_table_id(cc) != MMVQ_PARAMETERS_RDNA4 ||
        calc_nwarps(GGML_TYPE_Q8_0, 1, MMVQ_PARAMETERS_RDNA4) != 8) {
        return false;
    }
    if (gate->op != GGML_OP_MUL_MAT || up->op != GGML_OP_MUL_MAT) return false;
    const ggml_tensor * g0 = gate->src[0];
    const ggml_tensor * u0 = up->src[0];
    const ggml_tensor * x  = up->src[1];
    if (g0->type != GGML_TYPE_Q8_0 || u0->type != GGML_TYPE_Q8_0 || gate->src[1] != x ||
        !ggml_are_same_shape(g0, u0) || !ggml_are_same_stride(g0, u0) ||
        x->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 ||
        x->ne[2] != 1 || x->ne[3] != 1 || g0->ne[2] != 1 || g0->ne[3] != 1) {
        return false;
    }
    if (x->ne[1] < 2 || x->ne[1] > MMVQ_Q8_RDNA4_MAX_COLS) return false;
    if (glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU_DS4 ||
        glu->src[0] != gate || glu->src[1] != up || ggml_get_op_params_i32(glu, 1) != 0 ||
        glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu)) {
        return false;
    }
    return true;
}

template <int ncols_dst, int c_rows, int c_vw, int c_row_groups>
static void mul_mat_vec_q8_0_rdna4_launch_cfg(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y,
        const int stride_col_dst, const int nchannels_dst, const uint3 channel_ratio,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_dst, const uint3 sample_ratio, const int stride_sample_x,
        const int stride_sample_y, const int stride_sample_dst, const int kb_off, cudaStream_t stream) {
    constexpr int rows_per_block = c_rows*c_row_groups;
    const dim3 block_nums((nrows_x + rows_per_block - 1)/rows_per_block, nchannels_dst, nsamples_dst);
    const dim3 block_dims(32, c_vw*c_row_groups, 1);
    mul_mat_vec_q8_0_rdna4<ncols_dst, c_rows, c_vw, c_row_groups><<<block_nums, block_dims, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, kb_off, nullptr, 0.0f);
}

// The gate/up pair with SwiGLU-DS4 (ggml_cuda_mmvq_rdna4_glu_pair): one block
// shape, two rows per wave over four waves, whose cross-wave partials of both
// products fit the LDS at eight columns. Every shape is bit-exact, so the pair
// gives the value of the two table-shaped products and the GLU kernel.
template <int ncols_dst>
static void mul_mat_vec_q8_0_rdna4_launch_glu(
        const void * vx, const void * vx_gate, const void * vy, float * dst, const float glu_limit,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y,
        const int stride_col_dst, const int nchannels_dst, const uint3 channel_ratio,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_dst, const uint3 sample_ratio, const int stride_sample_x,
        const int stride_sample_y, const int stride_sample_dst, cudaStream_t stream) {
    constexpr int c_rows = 2, c_vw = 4, c_row_groups = 1;
    static_assert(2u*c_row_groups*c_vw*c_rows*ncols_dst*(8/c_vw)*32*sizeof(float) <= 65536,
                  "both products' partials must fit the LDS");
    constexpr int rows_per_block = c_rows*c_row_groups;
    const dim3 block_nums((nrows_x + rows_per_block - 1)/rows_per_block, nchannels_dst, nsamples_dst);
    const dim3 block_dims(32, c_vw*c_row_groups, 1);
    mul_mat_vec_q8_0_rdna4<ncols_dst, c_rows, c_vw, c_row_groups, true><<<block_nums, block_dims, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, 0, vx_gate, glu_limit);
}

template <int ncols_dst>
static void mul_mat_vec_q8_0_rdna4_launch_nc(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y,
        const int stride_col_dst, const int nchannels_dst, const uint3 channel_ratio,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_dst, const uint3 sample_ratio, const int stride_sample_x,
        const int stride_sample_y, const int stride_sample_dst, const int kb_off, cudaStream_t stream) {
#define Q8_RDNA4_LAUNCH(R, VW, G) \
    mul_mat_vec_q8_0_rdna4_launch_cfg<ncols_dst, R, VW, G>(vx, vy, dst, ncols_x, nrows_x, \
        stride_row_x, stride_col_y, stride_col_dst, nchannels_dst, channel_ratio, \
        stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_dst, sample_ratio, \
        stride_sample_x, stride_sample_y, stride_sample_dst, kb_off, stream)
    // Block shapes from the gfx1201 sweep of the DS4.1 dense projections
    // (server/test/bench/bench_ds41_q8_mmvq.cpp). Few rows (attn_kv, 512) or one
    // K step per row (attn_q_b, K=1280): four waves share a row pair's eight
    // virtual waves and two row pairs share a block, which keeps enough
    // loads in flight. Long rows over many rows: eight waves per row pair.
    const int nb = ncols_x / QK8_0;
    const int64_t rows = (int64_t) nrows_x * nchannels_dst * nsamples_dst;
    if (rows <= 1024 || nb <= 64) {
        Q8_RDNA4_LAUNCH(2, 4, 2);
    } else {
        Q8_RDNA4_LAUNCH(2, 8, 1);
    }
#undef Q8_RDNA4_LAUNCH
}

// vx_gate set: the gate/up pair (vx is the up matrix), 2..8 columns, no column range.
static void mul_mat_vec_q8_0_rdna4_launch(
        const void * vx, const void * vy, float * dst, const int ncols_dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y,
        const int stride_col_dst, const int nchannels_dst, const uint3 channel_ratio,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_dst, const uint3 sample_ratio, const int stride_sample_x,
        const int stride_sample_y, const int stride_sample_dst, const int kb_off,
        const void * vx_gate, const float glu_limit, cudaStream_t stream) {
    if (vx_gate) {
        GGML_ASSERT(kb_off == 0);
#define Q8_RDNA4_GLU_NC(NC) case NC: mul_mat_vec_q8_0_rdna4_launch_glu<NC>(vx, vx_gate, vy, dst, glu_limit, \
        ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, nchannels_dst, channel_ratio, \
        stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_dst, sample_ratio, \
        stride_sample_x, stride_sample_y, stride_sample_dst, stream); break
        switch (ncols_dst) {
            Q8_RDNA4_GLU_NC(2);
            Q8_RDNA4_GLU_NC(3);
            Q8_RDNA4_GLU_NC(4);
            Q8_RDNA4_GLU_NC(5);
            Q8_RDNA4_GLU_NC(6);
            Q8_RDNA4_GLU_NC(7);
            Q8_RDNA4_GLU_NC(8);
            default: GGML_ABORT("unreachable q8_0 rdna4 glu width");
        }
#undef Q8_RDNA4_GLU_NC
        return;
    }
#define Q8_RDNA4_NC(NC) case NC: mul_mat_vec_q8_0_rdna4_launch_nc<NC>(vx, vy, dst, ncols_x, nrows_x, \
        stride_row_x, stride_col_y, stride_col_dst, nchannels_dst, channel_ratio, stride_channel_x, \
        stride_channel_y, stride_channel_dst, nsamples_dst, sample_ratio, stride_sample_x, \
        stride_sample_y, stride_sample_dst, kb_off, stream); break
    switch (ncols_dst) {
        Q8_RDNA4_NC(1);
        Q8_RDNA4_NC(2);
        Q8_RDNA4_NC(3);
        Q8_RDNA4_NC(4);
        Q8_RDNA4_NC(5);
        Q8_RDNA4_NC(6);
        Q8_RDNA4_NC(7);
        Q8_RDNA4_NC(8);
        default: GGML_ABORT("unreachable q8_0 rdna4 width");
    }
#undef Q8_RDNA4_NC
}
#endif // defined(GGML_USE_HIP)

// True when the single-column launch for this type, device and K is the generic kernel with one wave per row:
// each lane walks K in the same order and one warp reduction sums the row, which is also what the multi-token
// MoE kernel does for every (token, row). Batch-invariant MUL_MAT_ID can then keep the MoE kernel, which serves
// all tokens in one launch with its expert prefetch, for batches within its per-type width (its launch bounds;
// batch-invariant dispatch also sends wider batches here). Multi-wave tables and the specialized single-column
// launches (gfx1151 fixed-K ROCmFP, the FP4 unroll, the multi-row IQ kernels on AMD) reduce in another order, so
// those take the tokenwise launch. test_ds41_mmid_width_invariance checks both cases per format and device.
template <ggml_type type>
static bool mmvq_moe_matches_single_column(const int cc, const int ncols_x, const int ncols_dst,
                                           const mmvq_parameter_table_id table_id) {
    if (ncols_dst > get_mmvq_mmid_max_batch(type, cc) ||
        calc_nwarps(type, 1, table_id) != 1 || calc_rows_per_block(1, table_id, false, 1) != 1) {
        return false;
    }
    switch (type) {
        case GGML_TYPE_Q4_0_ROCMFP4_FAST: return !is_gfx1151(cc);
        case GGML_TYPE_Q3_0_ROCMFPX:      return !mmvq_rocmfp3_fixed_k(cc, ncols_x);
        case GGML_TYPE_Q2_0_ROCMFP2:      return !mmvq_rocmfp2_fixed_k(cc, ncols_x);
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ3_XXS:           return !GGML_CUDA_CC_IS_AMD(cc);
        default:                          return true;
    }
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream, const bool ids_tokenwise_samples = false,
        const int kb_off = 0) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    // Tokens as samples (see the batch-invariant MUL_MAT_ID case below) are a single-column launch.
    GGML_ASSERT(!ids_tokenwise_samples || (ids != nullptr && ncols_dst == 1));

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    // The gfx1151 ROCmFP4 dense weight-reuse kernel accepts up to sixteen
    // columns; every other path keeps the generic MMVQ batch limit.
    const bool wide_rocmfp4_reuse =
        type == GGML_TYPE_Q4_0_ROCMFP4_FAST && ids == nullptr && is_gfx1151(cc);
    GGML_ASSERT(ncols_dst <= (ids ? MMVQ_MAX_MOE_BATCH_SIZE
                              : wide_rocmfp4_reuse ? ROCMFP4_REUSE_MAX_COLS
                                                   : MMVQ_MAX_BATCH_SIZE));

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    const bool has_ids = ids != nullptr;

    static const bool use_moe_kernel =
        mmvq_env_flag("LUCE_CUDA_MMVQ_MOE_KERNEL", true);

    // Batch-invariant MUL_MAT_ID (ggml_backend_cuda_set_mmvq_batch_invariant, set by DS4.1 verification): every
    // token must get the single-token path's arithmetic -- the decode kernel with its launch shape and reduction
    // order -- so a verify batch reproduces decode bit for bit. The MoE kernel already does where the decode launch
    // is the generic one-wave kernel (mmvq_moe_matches_single_column); elsewhere the multi-token kernels reduce in a
    // different order (test_ds41_mmid_width_invariance). For MUL_MAT_ID the columns are tokens.
    if (has_ids && ncols_dst > 1 && ggml_cuda_mmvq_batch_invariant() &&
        !(use_moe_kernel && mmvq_moe_matches_single_column<type>(cc, ncols_x, ncols_dst, table_id))) {
        // One launch: the tokens become the sample dimension of the single-token launch, and each block reads its
        // token's expert ids, activations and output row, so every block runs the decode kernel's arithmetic for
        // its token. Expert-bias fusions index the bias by sample, so they keep one launch per token.
        if (nsamples_x == 1 && nsamples_dst == 1 && fusion.x_bias == nullptr && fusion.gate_bias == nullptr) {
            mul_mat_vec_q_switch_ncols_dst<type>(
                vx, vy, ids, fusion, dst, ncols_x, nrows_x, 1, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                /*nsamples_x=*/1, /*nsamples_dst=*/ncols_dst, /*stride_sample_x=*/0,
                /*stride_sample_y=*/stride_col_y, /*stride_sample_dst=*/stride_col_dst, ids_stride, stream,
                /*ids_tokenwise_samples=*/true);
            return;
        }
        for (int t = 0; t < ncols_dst; ++t) {
            mul_mat_vec_q_switch_ncols_dst<type>(
                vx, (const block_q8_1 *) vy + (int64_t) t*stride_col_y, ids + (int64_t) t*ids_stride, fusion,
                dst + (int64_t) t*stride_col_dst, ncols_x, nrows_x, 1, stride_row_x, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
        }
        return;
    }

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
        constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
        constexpr int vdr                   = get_vdr_mmvq(type);
        const int     blocks_per_row_x      = ncols_x / qk;
        const int     blocks_per_iter_1warp = vdr * warp_size / qi;
        const int     nwarps                = calc_nwarps(type, c_ncols_dst, table_id);
        bool          use                   = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    static const bool use_tokenwise_mmid = []() {
        const char * e = std::getenv("LUCE_CUDA_MMVQ_MOE_TOKENWISE");
        return e && e[0] == '1' && e[1] == '\0';
    }();

    static const bool use_tokenwise_mm = []() {
        const char * e = std::getenv("LUCE_CUDA_MMVQ_TOKENWISE");
        return e && e[0] == '1' && e[1] == '\0';
    }();

    if (has_ids && ncols_dst > 1 && (use_tokenwise_mmid || mmid_invariant_tokenwise(type, cc, ncols_dst))) {
        constexpr int c_ncols_dst = 1;
        const bool use_small_k = should_use_small_k(c_ncols_dst);
        const uint3 token_sample_ratio_fd = init_fastdiv_values(ncols_dst);
        std::pair<dim3, dim3> dims = calc_launch_params<type>(
            c_ncols_dst, nrows_x, nchannels_dst, ncols_dst, warp_size, table_id, use_small_k);
        if (use_small_k) {
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, true>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd,
                stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                token_sample_ratio_fd, stride_sample_x, stride_col_y, stride_col_dst,
                dims.first, dims.second, 0, ids_stride, stream, true);
        } else {
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd,
                stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                token_sample_ratio_fd, stride_sample_x, stride_col_y, stride_col_dst,
                dims.first, dims.second, 0, ids_stride, stream, true);
        }
        return;
    }

    // Batch-invariant products (ggml_backend_cuda_set_mmvq_batch_invariant):
    // Q8_0 keeps the single-column block shape and reads each weight row once
    // for all columns; other types run the single-column kernel per column.
    const bool width_invariant = !has_ids && ncols_dst > 1 && ggml_cuda_mmvq_batch_invariant();
    // RDNA4 Q8_0 products on the eight-wave single-row block shape (one
    // column, or several columns under batch-invariant products): same
    // arithmetic, packed rows, see mul_mat_vec_q8_0_rdna4.
#if defined(GGML_USE_HIP)
    if constexpr (type == GGML_TYPE_Q8_0) {
        // A gate/up pair with SwiGLU-DS4 and nothing else fused (see
        // ggml_cuda_mmvq_rdna4_glu_pair) runs as one launch of this kernel.
        const bool rdna4_glu = has_fusion && fusion.gate != nullptr && fusion.x_bias == nullptr &&
            fusion.gate_bias == nullptr && fusion.glu_op == GGML_GLU_OP_SWIGLU_DS4 &&
            fusion.gate_value_scale == 1.0f && fusion.x_value_scale == 1.0f && kb_off == 0;
        if (!has_ids && (!has_fusion || rdna4_glu) && (ncols_dst == 1 || width_invariant) &&
            ncols_dst <= MMVQ_Q8_RDNA4_MAX_COLS && GGML_CUDA_CC_IS_RDNA4(cc) &&
            warp_size == 32 && table_id == MMVQ_PARAMETERS_RDNA4 &&
            calc_nwarps(type, 1, table_id) == 8 && mmvq_q8_0_rdna4_enabled()) {
            mul_mat_vec_q8_0_rdna4_launch(
                vx, vy, dst, ncols_dst, ncols_x, nrows_x, stride_row_x, stride_col_y,
                stride_col_dst, nchannels_dst, channel_ratio_fd, stride_channel_x,
                stride_channel_y, stride_channel_dst, nsamples_dst, sample_ratio_fd,
                stride_sample_x, stride_sample_y, stride_sample_dst, kb_off,
                rdna4_glu ? fusion.gate : nullptr, fusion.glu_param0, stream);
            return;
        }
    }
#endif // defined(GGML_USE_HIP)
    // The generic kernels fuse one column only; several columns are the RDNA4
    // Q8_0 gate/up pair above (ggml_cuda_mmvq_rdna4_glu_pair).
    GGML_ASSERT(!has_fusion || has_ids || ncols_dst == 1);
    if constexpr (type == GGML_TYPE_Q8_0) {
        if (width_invariant) {
#define GGML_MMVQ_INVARIANT_LAUNCH(NC) \
            case NC: { \
                std::pair<dim3, dim3> dims = calc_launch_params<type>( \
                    NC, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id, false, true); \
                mul_mat_vec_q_switch_fusion<type, NC, false, 0, false, false, false, false, true>( \
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, \
                    stride_row_x, stride_col_y, stride_col_dst, \
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, \
                    sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, \
                    dims.first, dims.second, 0, ids_stride, stream); \
            } break
            switch (ncols_dst) {
                GGML_MMVQ_INVARIANT_LAUNCH(2);
                GGML_MMVQ_INVARIANT_LAUNCH(3);
                GGML_MMVQ_INVARIANT_LAUNCH(4);
                GGML_MMVQ_INVARIANT_LAUNCH(5);
                GGML_MMVQ_INVARIANT_LAUNCH(6);
                GGML_MMVQ_INVARIANT_LAUNCH(7);
                GGML_MMVQ_INVARIANT_LAUNCH(8);
                default: GGML_ABORT("unreachable batch-invariant width");
            }
#undef GGML_MMVQ_INVARIANT_LAUNCH
            return;
        }
    }

    if ((use_tokenwise_mm || width_invariant) && !has_ids && ncols_dst > 1 && nsamples_dst == 1) {
        constexpr int c_ncols_dst = 1;
        const bool use_small_k = should_use_small_k(c_ncols_dst);
        const uint3 token_sample_ratio_fd = init_fastdiv_values(ncols_dst);
        std::pair<dim3, dim3> dims = calc_launch_params<type>(
            c_ncols_dst, nrows_x, nchannels_dst, ncols_dst, warp_size, table_id, use_small_k);
        if (use_small_k) {
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, true>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd,
                stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                token_sample_ratio_fd, stride_sample_x, stride_col_y, stride_col_dst,
                dims.first, dims.second, 0, ids_stride, stream);
        } else {
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd,
                stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                token_sample_ratio_fd, stride_sample_x, stride_col_y, stride_col_dst,
                dims.first, dims.second, 0, ids_stride, stream);
        }
        return;
    }

    if (has_ids && ncols_dst > 1 && (use_moe_kernel || ncols_dst > MMVQ_MAX_BATCH_SIZE)) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    if constexpr (type == GGML_TYPE_Q4_0_ROCMFP4_FAST) {
        if (!has_ids && ncols_dst == 4 && rocmfp4_x4_enabled()) {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(
                c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst,
                warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, 0,
                                        false, false, false, true>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd,
                stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y,
                stride_channel_dst, sample_ratio_fd, stride_sample_x,
                stride_sample_y, stride_sample_dst, dims.first, dims.second,
                0, ids_stride, stream);
            return;
        }
        if (!has_ids && ncols_dst == 5 && rocmfp4_x4_enabled() &&
            rocmfp4_q5_x4_plus1_enabled()) {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(
                c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst,
                warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, 0,
                                        false, false, false, true>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd,
                stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y,
                stride_channel_dst, sample_ratio_fd, stride_sample_x,
                stride_sample_y, stride_sample_dst, dims.first, dims.second,
                0, ids_stride, stream);
            return;
        }
    }

    // The generic q4 verifier kernels previously made every FP3 lane load the
    // complete 12-byte payload even though VDR=2 assigns that lane one
    // disjoint three-byte quarter.  Keep a separately instantiated, opt-in
    // ncols=4 path so the exact packed-dot A/B does not perturb q1 or the
    // dedicated multi-token expert kernel.
    if constexpr (type == GGML_TYPE_Q3_0_ROCMFPX) {
        if (!has_ids && ncols_dst == 4 && rocmfp3_packed24_enabled()) {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(
                c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst,
                warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, 0,
                                        false, false, true>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd,
                stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y,
                stride_channel_dst, sample_ratio_fd, stride_sample_x,
                stride_sample_y, stride_sample_dst, dims.first, dims.second,
                0, ids_stride, stream);
            return;
        }
    }

    // Decode-only gfx1151 specializations. Each one preserves the original
    // per-lane K traversal and accumulation order, so exact validated shapes
    // use the faster kernel without a serving-time tuning flag.
    if constexpr (type == GGML_TYPE_Q4_0_ROCMFP4_FAST) {
        // Two to sixteen dense columns share one decode of every weight
        // fragment. Two and three columns previously took the generic
        // per-column kernel; above the generic MMVQ limit a packed prompt
        // step previously had to split its projections into four-column
        // parts and re-read the weights per part.
        if (is_gfx1151(cc) && !has_ids && ncols_dst >= 2 &&
            ncols_dst <= ROCMFP4_REUSE_MAX_COLS) {
#define GGML_ROCMFP4_REUSE_LAUNCH(NC) \
            case NC: mul_mat_vec_rocmfp4_reuse_launch<NC>( \
                vx, vy, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, \
                nchannels_dst, stride_row_x, stride_col_y, stride_col_dst, \
                channel_ratio_fd, stride_channel_x, stride_channel_y, \
                stride_channel_dst, nsamples_dst, sample_ratio_fd, \
                stride_sample_x, stride_sample_y, stride_sample_dst, \
                ids_stride, warp_size, table_id, stream); break
            switch (ncols_dst) {
                GGML_ROCMFP4_REUSE_LAUNCH(2);
                GGML_ROCMFP4_REUSE_LAUNCH(3);
                GGML_ROCMFP4_REUSE_LAUNCH(4);
                GGML_ROCMFP4_REUSE_LAUNCH(5);
                GGML_ROCMFP4_REUSE_LAUNCH(6);
                GGML_ROCMFP4_REUSE_LAUNCH(7);
                GGML_ROCMFP4_REUSE_LAUNCH(8);
                GGML_ROCMFP4_REUSE_LAUNCH(9);
                GGML_ROCMFP4_REUSE_LAUNCH(10);
                GGML_ROCMFP4_REUSE_LAUNCH(11);
                GGML_ROCMFP4_REUSE_LAUNCH(12);
                GGML_ROCMFP4_REUSE_LAUNCH(13);
                GGML_ROCMFP4_REUSE_LAUNCH(14);
                GGML_ROCMFP4_REUSE_LAUNCH(15);
                GGML_ROCMFP4_REUSE_LAUNCH(16);
                default: GGML_ABORT("unreachable reuse width");
            }
#undef GGML_ROCMFP4_REUSE_LAUNCH
            return;
        }
        if (is_gfx1151(cc) && ncols_dst == 1) {
            mul_mat_vec_rocmfp4_unroll2_launch(
                vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd,
                nchannels_dst, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_dst, sample_ratio_fd,
                stride_sample_x, stride_sample_y, stride_sample_dst,
                ids_stride, warp_size, stream, ids_tokenwise_samples);
            return;
        }
    }

    if constexpr (type == GGML_TYPE_Q3_0_ROCMFPX) {
        if (ncols_dst == 1 && mmvq_rocmfp3_fixed_k(cc, ncols_x)) {
            mul_mat_vec_rocmfpx_fixed_k_launch<type, 2048>(
                vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd,
                nchannels_dst, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_dst, sample_ratio_fd,
                stride_sample_x, stride_sample_y, stride_sample_dst,
                ids_stride, warp_size, stream, ids_tokenwise_samples);
            return;
        }
    }

    if constexpr (type == GGML_TYPE_Q2_0_ROCMFP2) {
        if (ncols_dst == 1 && mmvq_rocmfp2_fixed_k(cc, ncols_x)) {
            if (ncols_x == 4096) {
                mul_mat_vec_rocmfpx_fixed_k_launch<type, 4096>(
                    vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd,
                    nchannels_dst, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y,
                    stride_channel_dst, nsamples_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst,
                    ids_stride, warp_size, stream, ids_tokenwise_samples);
                return;
            }
            if (ncols_x == 2048) {
                mul_mat_vec_rocmfpx_fixed_k_launch<type, 2048>(
                    vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd,
                    nchannels_dst, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y,
                    stride_channel_dst, nsamples_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst,
                    ids_stride, warp_size, stream, ids_tokenwise_samples);
                return;
            }
        }
    }

    // DS4.1 IQ2_XXS / IQ3_XXS experts (gate/up K=5120, down K=2304) on
    // one-wave AMD tables. A compile-time K fully unrolls the lane loop so
    // every weight and activation load issues before the first dot; each lane
    // still adds its blocks in the same order and the warp reduction is
    // unchanged. Both types have the same lane layout (qi 16, vdr 2).
    if constexpr (type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ3_XXS) {
        // Eight waves per block, two rows per wave: the two rows share every
        // activation load and the per-wave prologue, grid fill and barrier.
        constexpr int iq_row_warps = 8;
        constexpr int iq_wave_rows = 2;
        // Mirrors the kernel's own guards (wave32, one wave and one row per
        // table): a mismatch is declined here rather than launching a stub.
        if (GGML_CUDA_CC_IS_AMD(cc) && warp_size == 32 && ncols_dst == 1 &&
            nrows_x % (iq_row_warps*iq_wave_rows) == 0 &&
            calc_nwarps(type, 1, table_id) == 1 && calc_rows_per_block(1, table_id, false, 1) == 1 &&
            !should_use_small_k(1)) {
            if (ncols_x == 5120) {
                mul_mat_vec_rocmfpx_fixed_k_launch<type, 5120, iq_row_warps, iq_wave_rows>(
                    vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd,
                    nchannels_dst, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y,
                    stride_channel_dst, nsamples_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst,
                    ids_stride, warp_size, stream, ids_tokenwise_samples);
                return;
            }
            if (ncols_x == 2304) {
                mul_mat_vec_rocmfpx_fixed_k_launch<type, 2304, iq_row_warps, iq_wave_rows>(
                    vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd,
                    nchannels_dst, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y,
                    stride_channel_dst, nsamples_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst,
                    ids_stride, warp_size, stream, ids_tokenwise_samples);
                return;
            }
        }
    }

    switch (ncols_dst) {
        case 1: {
            constexpr int c_ncols_dst = 1;

            bool use_small_k = should_use_small_k(c_ncols_dst);

            if (use_small_k) {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id, true);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, true>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream, ids_tokenwise_samples);
            } else {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream, ids_tokenwise_samples);
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }

    GGML_UNUSED(has_fusion);
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream, const int kb_off = 0) {
    switch (type_x) {
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream,
                 /*ids_tokenwise_samples=*/false, kb_off);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0_ROCMFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0_ROCMFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0_ROCMFP4_FAST>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0_ROCMFP2:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0_ROCMFP2>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_0_ROCMFPX:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_0_ROCMFPX>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_0_ROCMFPX:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_0_ROCMFPX>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0_ROCMFPX:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0_ROCMFPX>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

bool ggml_cuda_luce_q8_memo_on() {
    static const bool on = []() {
        const char * e = getenv("LUCE_Q8_MEMO");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    return on;
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_Q8_1);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    ++g_mmvq_launch_count;

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_MOE_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        // Several columns only for the RDNA4 Q8_0 gate/up pair (a gate, no
        // biases: ggml_cuda_mmvq_rdna4_glu_pair), which switch_ncols_dst runs
        // on mul_mat_vec_q8_0_rdna4.
        GGML_ASSERT(  ids || dst->ne[1] == 1 ||
                    (src0->type == GGML_TYPE_Q8_0 && fusion->gate && !fusion->x_bias && !fusion->gate_bias));
        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_param0 = fusion->glu_param0;
        fusion_local.glu_param1 = fusion->glu_param1;
        fusion_local.gate_value_scale = fusion->gate_value_scale;
        fusion_local.x_value_scale = fusion->x_value_scale;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    // On by default: memoising the q8_1-quantized activations across the
    // matmuls of one token is a straight win on every backend measured, and
    // requiring an env var to get the tuned path invites running the slow one
    // by accident. LUCE_Q8_MEMO=0 opts out.
    static const bool luce_q8_memo_on = ggml_cuda_luce_q8_memo_on();
    const size_t q8_bytes = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    char * src1_q8_d = nullptr;
    if (src1->type == GGML_TYPE_Q8_1) {
        // Already the q8_1 rows read below (quantized on another device).
        GGML_ASSERT(ggml_is_contiguous(src1) && ne10 == ne10_padded);
        src1_q8_d = (char *) src1->data;
    }
    // The src1->q8_1 quantization depends only on src0->type and src1's dims/strides,
    // not on ids (ids only affects the matmul kernel's channel/dst strides below), so
    // the memo is valid for MUL_MAT_ID too: gate/up and the shared expert re-quantize
    // the same ffn_norm activation and can share one q8 buffer within an evaluation.
    // Synthetic tensors used inside a coarse fused operator do not own a
    // persistent graph allocation. Never retain their stack address as a
    // cross-node memo key.
    // The multi-stream graph optimizer runs sibling matmuls that share one
    // src1 on different streams with events only at fork and join: a memo
    // entry filled on one stream could be read on another while the
    // quantize kernel is still in flight. Skip the memo whenever concurrent
    // streams are active for this evaluation.
    const bool use_q8_memo = luce_q8_memo_on && src1->buffer != nullptr && src1->type == GGML_TYPE_F32 &&
                             ctx.stream_context().concurrent_events.empty();
    // quantize_row_q8_1_cuda ignores src0's type and writes rows in their
    // flattened order, so a contiguous src1 is keyed by the node it reshapes
    // (a reshape only reinterprets the same bytes), its address, its row
    // length and its row count: the routed experts' [n_embd, 1, T] input and
    // the shared expert's [n_embd, T] activations share one quantization.
    const ggml_tensor * q8_key_node = src1;
    int64_t q8_key_ne[4] = {ne10, ne11, ne12, ne13};
    int q8_key_type = (int) src0->type;
    if (ggml_is_contiguous(src1)) {
        // src[0], not view_src: view_src is the allocation root and would
        // skip an in-place node between the reshape and that root.
        while (q8_key_node->op == GGML_OP_RESHAPE && q8_key_node->src[0]) {
            q8_key_node = q8_key_node->src[0];
        }
        q8_key_ne[1] = ne11*ne12*ne13;
        q8_key_ne[2] = 1;
        q8_key_ne[3] = 1;
        q8_key_type = -1;
    }
    if (use_q8_memo) {
        for (const auto & e : ctx.luce_q8_memo) {
            if (e.src1_node == (const void *) q8_key_node && e.src1_data == (const void *) src1_d &&
                e.src0_type == q8_key_type &&
                e.ne[0] == q8_key_ne[0] && e.ne[1] == q8_key_ne[1] &&
                e.ne[2] == q8_key_ne[2] && e.ne[3] == q8_key_ne[3]) {
                src1_q8_d = e.buf->ptr;
                break;
            }
        }
    }
    if (src1_q8_d == nullptr) {
        char * q8_dst;
        if (use_q8_memo) {
            ggml_backend_cuda_context::luce_q8_memo_entry ent;
            ent.src1_node = (const void *) q8_key_node;
            ent.src1_data = (const void *) src1_d;
            ent.src0_type = q8_key_type;
            ent.ne[0] = q8_key_ne[0]; ent.ne[1] = q8_key_ne[1]; ent.ne[2] = q8_key_ne[2]; ent.ne[3] = q8_key_ne[3];
            ent.buf = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), q8_bytes);
            q8_dst = ent.buf->ptr;
            ctx.luce_q8_memo.push_back(std::move(ent));
        } else {
            src1_q8_1.alloc(q8_bytes);
            q8_dst = src1_q8_1.ptr;
        }
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, q8_dst, src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        src1_q8_d = q8_dst;
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    static const bool mmid_telemetry = []() {
        const char * value = std::getenv("LUCE_MMID_TELEMETRY");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();

    // [TAG_MMID_GROUPED] grouped-expert path for small MUL_MAT_ID batches.
    // Qualify Q5_0 for the Gemma 4 expert-down projection. A wider shape
    // screen found regressions, so other projections retain their fallback.
    // The grouped kernel reads complete four-row tiles; the qualified row
    // count is divisible by four. Broadcast and per-slot inputs are supported.
    const bool q5_grouped = src0->type == GGML_TYPE_Q5_0 && cc == 860 &&
        ncols_dst == 16 && nchannels_dst == 8 && ne00 == 704 && ne01 == 2816 &&
        (nchannels_y == 1 || nchannels_y == 8) &&
        fusion_local.gate == nullptr && fusion_local.x_bias == nullptr && fusion_local.gate_bias == nullptr &&
        mmid_grouped_env() && mmid_grouped_type_ok(src0->type) && mmid_grouped_device_ok();
    if (ids && (q5_grouped || ggml_cuda_mmvq_mmid_grouped_enabled(
            src0->type, cc, ncols_dst, nchannels_dst*ncols_dst))) {
        // Batches above MMID_GROUPED_MAX_PAIRS fall through to the legacy
        // per-expert kernel instead of aborting the request.
        const int np = (int) (nchannels_dst*ncols_dst);
        ggml_cuda_pool_alloc<int32_t> mmid_meta(ctx.pool());
        int32_t * meta_d = nullptr;
        float * gate_w = nullptr;
        int gate_w_stride = 0;
        float gate_tau = 0.0f;
        const mmid_gate_extra * gx = (const mmid_gate_extra *) ids->extra;  // [TAG_MMID_ADAPTIVE_K]
        if (gx != nullptr && gx->magic == MMID_GATE_MAGIC && gx->weights != nullptr && gx->weights->data != nullptr) {
            gate_w        = (float *) gx->weights->data;
            gate_w_stride = (int) (gx->weights->nb[1]/sizeof(float));
            gate_tau      = gx->tau;
        }
        // The gate, up and down products of an MoE layer read the same ids, so
        // with fixed top-k (nothing gates them) their expert grouping is the
        // same: it is computed once per evaluation and kept beside the
        // memoized q8_1 activations (same lifetime, same LIFO release), keyed
        // by the ids node like those are by their source node. Only when no
        // local pool block is outstanding, which the release order needs.
        const bool prep_memo = gate_w == nullptr && mmid_prep_memo_on() && src1_q8_1.ptr == nullptr &&
                               ids->buffer != nullptr && ctx.stream_context().concurrent_events.empty();
        if (prep_memo) {
            for (const auto & e : ctx.luce_q8_memo) {
                if (e.src0_type == MMID_PREP_MEMO_KIND && e.src1_node == (const void *) ids &&
                    e.src1_data == (const void *) ids_d && e.ne[0] == np && e.ne[1] == ncols_dst &&
                    e.ne[2] == nchannels_dst && e.ne[3] == ids_stride) {
                    meta_d = (int32_t *) e.buf->ptr;
                    break;
                }
            }
        }
        if (meta_d == nullptr) {
            if (prep_memo) {
                ggml_backend_cuda_context::luce_q8_memo_entry ent;
                ent.src1_node = (const void *) ids;
                ent.src1_data = (const void *) ids_d;
                ent.src0_type = MMID_PREP_MEMO_KIND;
                ent.ne[0] = np; ent.ne[1] = ncols_dst; ent.ne[2] = nchannels_dst; ent.ne[3] = ids_stride;
                ent.buf = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), MMID_META_INTS*sizeof(int32_t));
                meta_d = (int32_t *) ent.buf->ptr;
                ctx.luce_q8_memo.push_back(std::move(ent));
            } else {
                meta_d = mmid_meta.alloc(MMID_META_INTS);
            }
            // Adaptive-k writes -1 drop sentinels, so it must gate a scratch copy:
            // sibling weights can still take the legacy kernel, which interprets
            // ids as unsigned. Fixed top-k only reads ids and avoids both the
            // allocation and the tiny device-to-device copy on every expert op.
            ggml_cuda_pool_alloc<int32_t> ids_gated(ctx.pool());
            int32_t * prep_ids = const_cast<int32_t *>(ids_d);
            int prep_ids_stride = (int) ids_stride;
            if (gate_w != nullptr) {
                prep_ids = ids_gated.alloc((size_t) np);
                prep_ids_stride = (int) nchannels_dst;
                CUDA_CHECK(cudaMemcpy2DAsync(prep_ids, nchannels_dst*sizeof(int32_t),
                                             ids_d, ids_stride*sizeof(int32_t),
                                             nchannels_dst*sizeof(int32_t), ncols_dst,
                                             cudaMemcpyDeviceToDevice, stream));
            }
            // q4/top4 has just 16 pairs. Launch the smallest whole-warp block
            // instead of 256 threads; __syncthreads() still covers every pair.
            const int prep_warp = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
            const int prep_threads = ((np + prep_warp - 1)/prep_warp)*prep_warp;
            mmid_group_prep<<<1, prep_threads, 0, stream>>>(
                prep_ids, meta_d, (int) nchannels_dst, (int) ncols_dst, prep_ids_stride,
                gate_w, gate_w_stride, gate_tau);
            CUDA_CHECK(cudaGetLastError());
        }
        if (mul_mat_vec_q_grouped_dispatch(
                src0->type, src0->data, src1_q8_d, meta_d, fusion_local, dst_d,
                (int) ne00, (int) ne01, (int) nchannels_y,
                (int) s01, (int) stride_col_y, (int) stride_col_dst,
                (int) s02, (int) stride_channel_y, (int) stride_channel_dst,
                np, stream)) {
            ++g_mmvq_mmid_grouped_launch_count;
            if (mmid_telemetry) {
                std::fprintf(stderr,
                    "[dflash-mmid] event=mmvq type=%s width=%lld pairs=%d variant=grouped\n",
                    ggml_type_name(src0->type), (long long) ncols_dst, np);
            }
            return;
        }
    }

    if (mmid_telemetry && ids) {
        if (ncols_dst < 2) {
            // Single-token MUL_MAT_ID: the ordinary single-column MMVQ case, not
            // the multi-token legacy MoE launch the grouped path falls back to.
            std::fprintf(stderr,
                "[dflash-mmid] event=mmvq type=%s width=%lld pairs=%lld variant=single\n",
                ggml_type_name(src0->type), (long long) ncols_dst,
                (long long) (nchannels_dst*ncols_dst));
        } else {
            // Name the kernel mul_mat_vec_q_switch_type will actually run for this
            // ungrouped multi-token batch, so the label is not misreported when the
            // tokenwise or generic MMVQ modes are selected by env.
            static const bool tokenwise_mmid = []() {
                const char * e = std::getenv("LUCE_CUDA_MMVQ_MOE_TOKENWISE");
                return e && e[0] == '1' && e[1] == '\0';
            }();
            static const bool moe_kernel = []() {
                const char * e = std::getenv("LUCE_CUDA_MMVQ_MOE_KERNEL");
                return !(e && e[0] == '0' && e[1] == '\0');
            }();
            const char * variant =
                (tokenwise_mmid || mmid_invariant_tokenwise(src0->type, cc, ncols_dst)) ? "tokenwise" :
                (moe_kernel || ncols_dst > MMVQ_MAX_BATCH_SIZE) ? "moe" : "generic";
            const char * reason =
                ncols_dst > MMVQ_MAX_MOE_BATCH_SIZE ? "width_gt_16" :
                (int) (nchannels_dst*ncols_dst) > MMID_GROUPED_MAX_PAIRS ? "pairs_gt_256" :
                !mmid_grouped_env() ? "flag_off" :
                !mmid_grouped_type_ok(src0->type) ? "unsupported_type" :
                !mmid_grouped_arch_ok(cc) ? "unsupported_arch" :
                !mmid_grouped_device_ok() ? "device_filtered" : "dispatch_rejected";
            std::fprintf(stderr,
                "[dflash-mmid] event=mmvq type=%s width=%lld pairs=%lld variant=%s reason=%s\n",
                ggml_type_name(src0->type), (long long) ncols_dst,
                (long long) (nchannels_dst*ncols_dst), variant, reason);
        }
    }

    // A flagged column range of Q8_0 rows: its first block in the whole rows
    // (ggml_backend_cuda_mul_mat_whole_row_lanes).
    int kb_off = 0;
    if (!ids && src0->type == GGML_TYPE_Q8_0 && dst->op == GGML_OP_MUL_MAT &&
        dst->op_params[1] == GGML_CUDA_WHOLE_ROW_LANES &&
        src0->view_src != nullptr && src0->nb[1] == src0->view_src->nb[1]) {
        kb_off = (int) ((src0->view_offs % src0->nb[1]) / sizeof(block_q8_0));
    }
    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_d, ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream, kb_off);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
