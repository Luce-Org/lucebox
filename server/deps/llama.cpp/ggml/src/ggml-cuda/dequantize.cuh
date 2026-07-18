#include "common.cuh"
#include "tq3-quant.cuh"
#include "../../rocmfp4/rocmfp4_hip_scale.cuh"

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

static __device__ __forceinline__ void dequantize_rocmfp4(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_rocmfp4 * x = (const block_rocmfp4 *) vx;

    const int q = x[ib].qs[iqs];
    const float d0 = rocmfp4_ue4m3_to_fp32_half_finite(x[ib].e[0]);
    const float d1 = rocmfp4_ue4m3_to_fp32_half_finite(x[ib].e[1]);

    v.x = d0 * rocmfp4_decode_i8(q);
    v.y = d1 * rocmfp4_decode_i8(q >> 4);
}

static __device__ __forceinline__ void dequantize_rocmfp4_fast(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_rocmfp4_fast * x = (const block_rocmfp4_fast *) vx;

    const int q = x[ib].qs[iqs];
    const float d = rocmfp4_ue4m3_to_fp32_half_finite(x[ib].e);

    v.x = d * rocmfp4_decode_i8(q);
    v.y = d * rocmfp4_decode_i8(q >> 4);
}

template<int qs>
static __device__ __forceinline__ uint32_t rocmfpx_load_qs_window_cuda(const uint8_t * src, const int byte_pos) {
    uint32_t v = (uint32_t) src[byte_pos + 0];

    if (byte_pos + 1 < qs) {
        v |= (uint32_t) src[byte_pos + 1] << 8;
    }
    if (byte_pos + 2 < qs) {
        v |= (uint32_t) src[byte_pos + 2] << 16;
    }

    return v;
}

static __device__ __forceinline__ uint32_t rocmfpx_get_fp3_code_cuda(const uint8_t * src, const int i) {
    const int bit_pos  = i * 3;
    const int byte_pos = bit_pos >> 3;
    const int shift    = bit_pos & 7;
    return (rocmfpx_load_qs_window_cuda<QS_ROCMFP3>(src, byte_pos) >> shift) & 7u;
}

static __device__ __forceinline__ uint32_t rocmfpx_get_fp2_code_cuda(const uint8_t * src, const int i) {
    return (src[i >> 2] >> (2*(i & 3))) & 3u;
}

static __device__ __forceinline__ uint32_t rocmfpx_get_fp6_code_cuda(const uint8_t * src, const int i) {
    const int bit_pos  = i * 6;
    const int byte_pos = bit_pos >> 3;
    const int shift    = bit_pos & 7;
    return (rocmfpx_load_qs_window_cuda<QS_ROCMFP6>(src, byte_pos) >> shift) & 63u;
}

static __device__ __forceinline__ int rocmfpx_decode_fp3_code_cuda(const uint32_t code) {
    const uint32_t mag_code = code & 3u;
    const int mag = mag_code == 3u ? 4 : (int) mag_code;
    return (code & 4u) ? -mag : mag;
}

static __device__ __forceinline__ float rocmfpx_decode_fp2_code_cuda(const uint32_t code) {
    switch (code & 3u) {
        case 0: return (float) ROCMFP2_KVALUE_0_I8;
        case 1: return (float) ROCMFP2_KVALUE_1_I8;
        case 2: return (float) ROCMFP2_KVALUE_2_I8;
        default: return (float) ROCMFP2_KVALUE_3_I8;
    }
}

static __device__ __forceinline__ int rocmfpx_decode_fp6_code_cuda(const uint32_t code) {
    const int mag = (int) (code & 31u);
    return (code & 32u) ? -mag : mag;
}

static __device__ __forceinline__ void dequantize_rocmfpx_fp3(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_rocmfp3 * x = (const block_rocmfp3 *) vx;

    const int i0 = iqs + 0;
    const int i1 = iqs + 1;
    const float d0 = rocmfpx_ue4m3_to_fp32_finite(x[ib].e[i0 >= QK_ROCMFP3/2]);
    const float d1 = rocmfpx_ue4m3_to_fp32_finite(x[ib].e[i1 >= QK_ROCMFP3/2]);

    v.x = d0 * (float) rocmfpx_decode_fp3_code_cuda(rocmfpx_get_fp3_code_cuda(x[ib].qs, i0));
    v.y = d1 * (float) rocmfpx_decode_fp3_code_cuda(rocmfpx_get_fp3_code_cuda(x[ib].qs, i1));
}

static __device__ __forceinline__ void dequantize_rocmfpx_fp2(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_rocmfp2 * x = (const block_rocmfp2 *) vx;

    const int i0 = iqs + 0;
    const int i1 = iqs + 1;
    const float d0 = rocmfpx_ue4m3_to_fp32_finite(x[ib].e[i0 >= QK_ROCMFP2/2]);
    const float d1 = rocmfpx_ue4m3_to_fp32_finite(x[ib].e[i1 >= QK_ROCMFP2/2]);

    v.x = d0 * rocmfpx_decode_fp2_code_cuda(rocmfpx_get_fp2_code_cuda(x[ib].qs, i0));
    v.y = d1 * rocmfpx_decode_fp2_code_cuda(rocmfpx_get_fp2_code_cuda(x[ib].qs, i1));
}

static __device__ __forceinline__ void dequantize_rocmfpx_fp6(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_rocmfp6 * x = (const block_rocmfp6 *) vx;

    const int i0 = iqs + 0;
    const int i1 = iqs + 1;
    const float d0 = rocmfpx_ue4m3_to_fp32_finite(x[ib].e[i0 >= QK_ROCMFP6/2]);
    const float d1 = rocmfpx_ue4m3_to_fp32_finite(x[ib].e[i1 >= QK_ROCMFP6/2]);

    v.x = d0 * (float) rocmfpx_decode_fp6_code_cuda(rocmfpx_get_fp6_code_cuda(x[ib].qs, i0));
    v.y = d1 * (float) rocmfpx_decode_fp6_code_cuda(rocmfpx_get_fp6_code_cuda(x[ib].qs, i1));
}

static __device__ __forceinline__ void dequantize_rocmfpx_fp8(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_rocmfp8 * x = (const block_rocmfp8 *) vx;

    const float d = rocmfpx_ue4m3_to_fp32_finite(x[ib].e);
    v.x = d * (float) x[ib].qs[iqs + 0];
    v.y = d * (float) x[ib].qs[iqs + 1];
}
