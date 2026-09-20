#pragma once
#include "dequantize.cuh"
#include <type_traits>

struct mmb_quant_slice {
    uint16_t * dst;
    int begin;
    int offset = 0;
    struct element {
        uint16_t * dst;
        int index;
        __device__ void operator=(float value) const {
            if (index >= 0 && index < 64) dst[index] = mmb_f2bf(value);
        }
    };
    __device__ mmb_quant_slice operator+(int64_t n) const { return {dst, begin, offset + (int) n}; }
    __device__ element operator[](int64_t n) const { return {dst, offset + (int) n - begin}; }
};

template <ggml_type TYPE>
__device__ __forceinline__ void mmb_decode_slice(const void * row, const int k0, uint16_t * dst, const int lane) {
    constexpr int QK = ggml_cuda_type_traits<TYPE>::qk;
    if constexpr (TYPE == GGML_TYPE_Q2_0) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q2_0(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q4_0) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q4_0(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q4_1) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q4_1(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q5_0) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q5_0(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q5_1) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q5_1(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q8_0) {
        constexpr int QR = ggml_cuda_type_traits<TYPE>::qr;
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, qs = (pos % QK) / QR;
            float2 v;
            dequantize_q8_0(row, ib, qs, v);
            const int o = ib * QK + qs - k0;
            dst[o] = mmb_f2bf(v.x);
            dst[o + (QR == 1 ? 1 : QK / 2)] = mmb_f2bf(v.y);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_Q2_K) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 64; tid += 8) dequantize_q2_K<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_Q3_K) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 64; tid += 8) dequantize_q3_K<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_Q4_K) {
        const mmb_quant_slice out{dst, k0 % QK};
        const int tid = (k0 % QK) / 64 * 8 + lane;
        dequantize_q4_K<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_Q5_K) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 64; tid += 8) dequantize_q5_K<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_Q6_K) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 64; tid += 8) dequantize_q6_K<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ1_S) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 32; tid += 8) dequantize_iq1_s<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ1_M) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 32; tid += 8) dequantize_iq1_m<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ2_XXS) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 32; tid += 8) dequantize_iq2_xxs<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ2_XS) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 32; tid += 8) dequantize_iq2_xs<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ2_S) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 32; tid += 8) dequantize_iq2_s<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ3_XXS) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 32; tid += 8) dequantize_iq3_xxs<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ3_S) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 32; tid += 8) dequantize_iq3_s<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ4_XS) {
        const mmb_quant_slice out{dst, k0 % QK};
#pragma unroll
        for (int tid = lane; tid < 32; tid += 8) dequantize_iq4_xs<float>(row, k0 / QK, out, tid);
    }
    else if constexpr (TYPE == GGML_TYPE_IQ4_NL) {
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / 32, q = (pos % 32) / 2;
            const block_iq4_nl & b = ((const block_iq4_nl *) row)[ib];
            const int o = ib * 32 + q - k0;
            dst[o] = mmb_f2bf((float)b.d * kvalues_iq4nl[b.qs[q] & 15]);
            dst[o + 16] = mmb_f2bf((float)b.d * kvalues_iq4nl[b.qs[q] >> 4]);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_MXFP4) {
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, q = (pos % QK) / 2;
            const block_mxfp4 & b = ((const block_mxfp4 *) row)[ib];
            const float d = ggml_cuda_e8m0_to_fp32(b.e);
            const int o = ib * QK + q - k0;
            dst[o] = mmb_f2bf(d * kvalues_mxfp4[b.qs[q] & 15] * 0.5f);
            dst[o + QK / 2] = mmb_f2bf(d * kvalues_mxfp4[b.qs[q] >> 4] * 0.5f);
        }
    }
    else if constexpr (TYPE == GGML_TYPE_NVFP4) {
#pragma unroll
        for (int p = lane; p < 32; p += 8) {
            const int pos = k0 + 2 * p, ib = pos / QK, in = pos % QK;
            const int sub = in / QK_NVFP4_SUB, j = (in % QK_NVFP4_SUB) / 2;
            const block_nvfp4 & b = ((const block_nvfp4 *) row)[ib];
            const float d = ggml_cuda_ue4m3_to_fp32(b.d[sub]);
            const uint8_t q = b.qs[sub * (QK_NVFP4_SUB / 2) + j];
            const int o = ib * QK + sub * QK_NVFP4_SUB + j - k0;
            dst[o] = mmb_f2bf(d * kvalues_mxfp4[q & 15]);
            dst[o + QK_NVFP4_SUB / 2] = mmb_f2bf(d * kvalues_mxfp4[q >> 4]);
        }
    }
}

template <int WTYPE>
__host__ __device__ constexpr size_t mmb_row_bytes(int k) {
    if constexpr (WTYPE == 0) return (size_t)(k / 32) * 18;
    else if constexpr (WTYPE == 1) return (size_t)(k / 32) * 34;
    else if constexpr (WTYPE == 2) return (size_t)k * 2;
    else {
        using traits = ggml_cuda_type_traits<(ggml_type)(WTYPE - 32)>;
        return (size_t)(k / traits::qk) * traits::bs;
    }
}

template <int WTYPE, int BM, int STRIDE>
__device__ __forceinline__ void mmb_load_quant_tile(const uint8_t * weights, size_t row_bytes, int rows, int ks, uint16_t * tile) {
    for (int row = threadIdx.x / 8; row < BM; row += blockDim.x / 8) {
        uint16_t * dst = tile + row * STRIDE;
        const int lane = threadIdx.x % 8;
        if (row < rows) mmb_decode_slice<(ggml_type)(WTYPE - 32)>(weights + row * row_bytes, ks * 64, dst, lane);
        else {
#pragma unroll
            for (int k = lane; k < 64; k += 8) dst[k] = 0;
        }
    }
}

static bool mmb_quant_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            return true;
        default: return false;
    }
}

template <typename Fn>
static void mmb_dispatch_quant(ggml_type type, Fn fn) {
    switch (type) {
        case GGML_TYPE_Q2_0: fn(std::integral_constant<int, 32 + GGML_TYPE_Q2_0>{}); break;
        case GGML_TYPE_Q4_0: fn(std::integral_constant<int, 32 + GGML_TYPE_Q4_0>{}); break;
        case GGML_TYPE_Q4_1: fn(std::integral_constant<int, 32 + GGML_TYPE_Q4_1>{}); break;
        case GGML_TYPE_Q5_0: fn(std::integral_constant<int, 32 + GGML_TYPE_Q5_0>{}); break;
        case GGML_TYPE_Q5_1: fn(std::integral_constant<int, 32 + GGML_TYPE_Q5_1>{}); break;
        case GGML_TYPE_Q8_0: fn(std::integral_constant<int, 1>{}); break;
        case GGML_TYPE_Q2_K: fn(std::integral_constant<int, 32 + GGML_TYPE_Q2_K>{}); break;
        case GGML_TYPE_Q3_K: fn(std::integral_constant<int, 32 + GGML_TYPE_Q3_K>{}); break;
        case GGML_TYPE_Q4_K: fn(std::integral_constant<int, 32 + GGML_TYPE_Q4_K>{}); break;
        case GGML_TYPE_Q5_K: fn(std::integral_constant<int, 32 + GGML_TYPE_Q5_K>{}); break;
        case GGML_TYPE_Q6_K: fn(std::integral_constant<int, 32 + GGML_TYPE_Q6_K>{}); break;
        case GGML_TYPE_IQ1_S: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ1_S>{}); break;
        case GGML_TYPE_IQ1_M: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ1_M>{}); break;
        case GGML_TYPE_IQ2_XXS: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ2_XXS>{}); break;
        case GGML_TYPE_IQ2_XS: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ2_XS>{}); break;
        case GGML_TYPE_IQ2_S: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ2_S>{}); break;
        case GGML_TYPE_IQ3_XXS: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ3_XXS>{}); break;
        case GGML_TYPE_IQ3_S: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ3_S>{}); break;
        case GGML_TYPE_IQ4_XS: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ4_XS>{}); break;
        case GGML_TYPE_IQ4_NL: fn(std::integral_constant<int, 0>{}); break;
        case GGML_TYPE_MXFP4: fn(std::integral_constant<int, 32 + GGML_TYPE_MXFP4>{}); break;
        case GGML_TYPE_NVFP4: fn(std::integral_constant<int, 32 + GGML_TYPE_NVFP4>{}); break;
        default: GGML_ABORT("unsupported MMB quant type");
    }
}

__device__ __forceinline__ void mmb_dq_q4k_slice(uint4 q0, uint4 q1, uint4 meta, int slice, uint32_t * out) {
    const float d = mmb_h2f((uint16_t) meta.x), dm = mmb_h2f((uint16_t)(meta.x >> 16));
    auto byte = [&](int i) { const uint32_t word = i < 4 ? meta.y : i < 8 ? meta.z : meta.w; return (word >> (8 * (i & 3))) & 255; };
    auto scale_min = [&](int i, float & ds, float & ms) {
        const uint32_t sc = i < 4 ? byte(i) & 63 : (byte(i + 4) & 15) | ((byte(i - 4) >> 6) << 4);
        const uint32_t mn = i < 4 ? byte(i + 4) & 63 : (byte(i + 4) >> 4) | ((byte(i) >> 6) << 4);
        ds = d * sc; ms = dm * mn;
    };
    float d0, d1, m0, m1;
    scale_min(2 * slice, d0, m0); scale_min(2 * slice + 1, d1, m1);
    const uint32_t words[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t q = words[j];
        out[2*j] = mmb_pack2(d0 * (q & 15) - m0, d0 * ((q >> 8) & 15) - m0);
        out[2*j + 1] = mmb_pack2(d0 * ((q >> 16) & 15) - m0, d0 * ((q >> 24) & 15) - m0);
        out[16 + 2*j] = mmb_pack2(d1 * ((q >> 4) & 15) - m1, d1 * ((q >> 12) & 15) - m1);
        out[16 + 2*j + 1] = mmb_pack2(d1 * ((q >> 20) & 15) - m1, d1 * (q >> 28) - m1);
    }
}


__device__ __forceinline__ void mmb_dq_q51_pair(uint4 a, uint4 b, uint4 c, uint32_t * out) {
    const uint32_t words[12] = {a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w,c.x,c.y,c.z,c.w};
#pragma unroll
    for (int block = 0; block < 2; ++block) {
        const uint32_t dm = words[block * 6], high = words[block * 6 + 1];
        const float d = mmb_h2f((uint16_t)dm), m = mmb_h2f((uint16_t)(dm >> 16));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q = words[block * 6 + 2 + j];
            float lo[4], hi[4];
#pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
                const int low = ((q >> (8 * lane)) & 15) | (((high >> (4 * j + lane)) & 1) << 4);
                const int upper = ((q >> (8 * lane + 4)) & 15) | (((high >> (16 + 4 * j + lane)) & 1) << 4);
                lo[lane] = d * low + m; hi[lane] = d * upper + m;
            }
            out[block * 16 + 2*j] = mmb_pack2(lo[0],lo[1]);
            out[block * 16 + 2*j + 1] = mmb_pack2(lo[2],lo[3]);
            out[block * 16 + 8 + 2*j] = mmb_pack2(hi[0],hi[1]);
            out[block * 16 + 8 + 2*j + 1] = mmb_pack2(hi[2],hi[3]);
        }
    }
}
