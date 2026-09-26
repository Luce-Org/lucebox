#include "common.cuh"
#include "tq3-quant.cuh"
#include "convert.cuh"
#include "../../rocmfp4/rocmfp4_hip_scale.cuh"

static __device__ __forceinline__ void dequantize_q2_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q2_0 * x = (const block_q2_0 *) vx;

    const float d = x[ib].d;

    // Q2_0: 2 bits per element, 4 elements per byte.
    // Stored code c in {0,1,2,3} maps to symbol s = c - 1 in {-1, 0, +1, +2}.
    const int byte_index_0 = iqs / 4;
    const int bit_offset_0 = (iqs % 4) * 2;

    const int byte_index_1 = (iqs + 1) / 4;
    const int bit_offset_1 = ((iqs + 1) % 4) * 2;

    const int c0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 0x3;
    const int c1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 0x3;

    v.x = (c0 - 1) * d;
    v.y = (c1 - 1) * d;
}

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

#ifdef ROCMFP2_AFFINE
    // Affine qtype-107 stores one (scale, offset) pair per 32-weight block:
    // value = code*scale - offset, with code in [0, 3]. This wire format is
    // incompatible with the default two-scale ROCmFP2 layout.
    const float scale  = rocmfpx_ue4m3_to_fp32_finite(x[ib].e[0]);
    const float offset = rocmfpx_ue4m3_to_fp32_finite(x[ib].e[1]);

    v.x = (float) (rocmfpx_get_fp2_code_cuda(x[ib].qs, i0) & 3u) * scale - offset;
    v.y = (float) (rocmfpx_get_fp2_code_cuda(x[ib].qs, i1) & 3u) * scale - offset;
#else
    const float d0 = rocmfpx_ue4m3_to_fp32_finite(x[ib].e[i0 >= QK_ROCMFP2/2]);
    const float d1 = rocmfpx_ue4m3_to_fp32_finite(x[ib].e[i1 >= QK_ROCMFP2/2]);

    v.x = d0 * rocmfpx_decode_fp2_code_cuda(rocmfpx_get_fp2_code_cuda(x[ib].qs, i0));
    v.y = d1 * rocmfpx_decode_fp2_code_cuda(rocmfpx_get_fp2_code_cuda(x[ib].qs, i1));
#endif
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
//================================== k-quants

// Each call dequantizes one super-block of QK_K values into y using the
// thread layout of the caller: 32 threads for q4_K, 64 threads otherwise.

template<typename dst_t, typename dst_ptr_t>
static __device__ __forceinline__ void dequantize_q2_K(const void * vx, const int64_t ib, dst_ptr_t yy, const int tid) {
    const block_q2_K * x = (const block_q2_K *) vx;

    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;

    const uint8_t q = x[ib].qs[32*n + l];
    auto y = yy + 128*n;

    float dall = __low2half(x[ib].dm);
    float dmin = __high2half(x[ib].dm);
    y[l+ 0] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[ib].scales[is+0] >> 4));
    y[l+32] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[ib].scales[is+2] >> 4));
    y[l+64] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[ib].scales[is+4] >> 4));
    y[l+96] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[ib].scales[is+6] >> 4));
}

template<typename dst_t, typename dst_ptr_t>
static __device__ __forceinline__ void dequantize_q3_K(const void * vx, const int64_t ib, dst_ptr_t yy, const int tid) {
    const block_q3_K * x = (const block_q3_K *) vx;

    const int64_t r = tid/4;
    const int64_t t = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16*is0 + 4*(tid%4);
    const int64_t n = t / 4;
    const int64_t j = t - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[ib].scales[is-0] & 0xF) | (((x[ib].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[ib].scales[is-0] & 0xF) | (((x[ib].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[ib].scales[is-8] >>  4) | (((x[ib].scales[is+0] >> 4) & 3) << 4) :
                          (x[ib].scales[is-8] >>  4) | (((x[ib].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[ib].d;
    float dl = d_all * (us - 32);

    auto y = yy + 128*n + 32*j;
    const uint8_t * q = x[ib].qs + 32*n;
    const uint8_t * hm = x[ib].hmask;

    for (int l = l0; l < l0+4; ++l) {
        y[l] = ggml_cuda_cast<dst_t>(dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t, typename dst_ptr_t>
static __device__ __forceinline__ void dequantize_q4_K(const void * vx, const int64_t ib, dst_ptr_t yy, const int tid) {
    const block_q4_K * x = (const block_q4_K *) vx;

    // assume 32 threads
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;

    auto y = yy + 64*il + n*ir;

    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);

    const uint8_t * q = x[ib].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = ggml_cuda_cast<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l +32] = ggml_cuda_cast<dst_t>(d2 * (q[l] >>  4) - m2);
    }
}

template<typename dst_t, typename dst_ptr_t>
static __device__ __forceinline__ void dequantize_q5_K(const void * vx, const int64_t ib, dst_ptr_t yy, const int tid) {
    const block_q5_K * x = (const block_q5_K *) vx;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    auto y = yy + 64*il + 2*ir;

    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);

    const uint8_t * ql = x[ib].qs + 32*il + 2*ir;
    const uint8_t * qh = x[ib].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = ggml_cuda_cast<dst_t>(d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1);
    y[ 1] = ggml_cuda_cast<dst_t>(d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1);
    hm <<= 1;
    y[32] = ggml_cuda_cast<dst_t>(d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2);
    y[33] = ggml_cuda_cast<dst_t>(d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2);
}

template<typename dst_t, typename dst_ptr_t>
static __device__ __forceinline__ void dequantize_q6_K(const void * vx, const int64_t ib, dst_ptr_t yy, const int tid) {
    const block_q6_K * x = (const block_q6_K *) vx;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    auto y = yy + 128*ip + il;

    const float d = x[ib].d;

    const uint8_t * ql = x[ib].ql + 64*ip + il;
    const uint8_t   qh = x[ib].qh[32*ip + il];
    const int8_t  * sc = x[ib].scales + is;

    y[ 0] = ggml_cuda_cast<dst_t>(d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    y[32] = ggml_cuda_cast<dst_t>(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    y[64] = ggml_cuda_cast<dst_t>(d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
    y[96] = ggml_cuda_cast<dst_t>(d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32));
}

//================================== i-quants

// Each call dequantizes one super-block of QK_K values into y with 32
// threads; iq4_nl packs QK_K/QK4_NL sub-blocks per super-block.

template<typename dst_t, typename dst_ptr_t>
static __device__ __forceinline__ void dequantize_iq2_xxs(const void * vx, const int64_t ibs, dst_ptr_t yy, const int tid) {

    const block_iq2_xxs * x = (const block_iq2_xxs  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    auto y = yy + 32*ib + 8*il;
    const uint16_t * q2 = x[ibs].qs + 4*ib;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const uint8_t  * grid = (const uint8_t *)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float)x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
}

template<typename dst_t, typename dst_ptr_t>
static __device__ __forceinline__ void dequantize_iq2_xs(const void * vx, const int64_t ibs, dst_ptr_t yy, const int tid) {

    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    auto y = yy + 32*ib + 8*il;
    const uint16_t * q2 = x[ibs].qs + 4*ib;
    const uint8_t  * grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d = (float)x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
}

template<typename dst_t, typename dst_ptr_t>
static __device__ __forceinline__ void dequantize_iq2_s(const void * vx, const int64_t ibs, dst_ptr_t yy, const int tid) {

    const block_iq2_s * x = (const block_iq2_s *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    auto y = yy + 32*ib + 8*il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (x[ibs].qs[4*ib+il] | ((x[ibs].qh[ib] << (8-2*il)) & 0x300)));
    const float d = (float)x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K/8+4*ib+il];
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
}

template<typename dst_t, typename dst_ptr_t>
static __device__ __forceinline__ void dequantize_iq3_xxs(const void * vx, const int64_t ibs, dst_ptr_t yy, const int tid) {

    const block_iq3_xxs * x = (const block_iq3_xxs  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    auto y = yy + 32*ib + 8*il;
    const uint8_t  * q3 = x[ibs].qs + 8*ib;
    const uint16_t * gas = (const uint16_t *)(x[ibs].qs + QK_K/4) + 2*ib;
    const uint8_t  * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    const uint8_t  * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float)x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = ggml_cuda_cast<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f));
        y[j+4] = ggml_cuda_cast<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f));
    }
}

template<typename dst_t, typename dst_ptr_t>
static __device__ __forceinline__ void dequantize_iq3_s(const void * vx, const int64_t ibs, dst_ptr_t yy, const int tid) {

    const block_iq3_s * x = (const block_iq3_s *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    auto y = yy + 32*ib + 8*il;
    const uint8_t * qs = x[ibs].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[ibs].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[ibs].qh[ib] << (7-2*il)) & 256)));
    const float d = (float)x[ibs].d * (1 + 2*((x[ibs].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = ggml_cuda_cast<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f));
        y[j+4] = ggml_cuda_cast<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f));
    }
}

template<typename dst_t, typename dst_ptr_t>
static __device__ __forceinline__ void dequantize_iq1_s(const void * vx, const int64_t ibs, dst_ptr_t yy, const int tid) {

    const block_iq1_s * x = (const block_iq1_s  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    auto y = yy + 32*ib + 8*il;
    const float delta = x[ibs].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = (float)x[ibs].d * (2*((x[ibs].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4*ib+il] | (((x[ibs].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * (q[j] + delta));
    }
}

template<typename dst_t, typename dst_ptr_t>
static __device__ __forceinline__ void dequantize_iq1_m(const void * vx, const int64_t ibs, dst_ptr_t yy, const int tid) {

    const block_iq1_m * x = (const block_iq1_m  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    auto y = yy + 32*ib + 8*il;
    const uint16_t * sc = (const uint16_t *)x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    const float d = (float)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4*ib+il] | (((x[ibs].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * (q[j] + delta));
    }
}

template<typename dst_t, typename dst_ptr_t>
static __device__ __forceinline__ void dequantize_iq4_nl(const void * vx, const int64_t ibs, dst_ptr_t yy, const int tid) {

    const block_iq4_nl * x = (const block_iq4_nl *) vx + ibs*(QK_K/QK4_NL);

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    auto y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = (float)x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j+16] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] >>  4]);
    }
}

template<typename dst_t, typename dst_ptr_t>
static __device__ __forceinline__ void dequantize_iq4_xs(const void * vx, const int64_t ibs, dst_ptr_t yy, const int tid) {
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    auto y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ibs].qs + 16*ib + 4*il;
    const float d = (float)x[ibs].d * ((((x[ibs].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[ibs].scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j+16] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] >>  4]);
    }
}

template<typename dst_t, typename dst_ptr_t>
static __device__ __forceinline__ void dequantize_mxfp4(const void * vx, const int64_t ibs, dst_ptr_t yy, const int tid) {

    const block_mxfp4 * x = (const block_mxfp4 *) vx + ibs*(QK_K/QK_MXFP4);

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    auto y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = ggml_cuda_e8m0_to_fp32(x[ib].e);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q4[j] & 0xf]*0.5f);
        y[j+16] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q4[j] >>  4]*0.5f);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q2_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    dequantize_q2_K<dst_t, dst_t *>(vx, ib, yy, tid);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q3_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    dequantize_q3_K<dst_t, dst_t *>(vx, ib, yy, tid);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q4_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    dequantize_q4_K<dst_t, dst_t *>(vx, ib, yy, tid);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q5_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    dequantize_q5_K<dst_t, dst_t *>(vx, ib, yy, tid);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q6_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    dequantize_q6_K<dst_t, dst_t *>(vx, ib, yy, tid);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_xxs(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    dequantize_iq2_xxs<dst_t, dst_t *>(vx, ib, yy, tid);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_xs(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    dequantize_iq2_xs<dst_t, dst_t *>(vx, ib, yy, tid);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_s(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    dequantize_iq2_s<dst_t, dst_t *>(vx, ib, yy, tid);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq3_xxs(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    dequantize_iq3_xxs<dst_t, dst_t *>(vx, ib, yy, tid);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq3_s(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    dequantize_iq3_s<dst_t, dst_t *>(vx, ib, yy, tid);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq1_s(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    dequantize_iq1_s<dst_t, dst_t *>(vx, ib, yy, tid);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq1_m(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    dequantize_iq1_m<dst_t, dst_t *>(vx, ib, yy, tid);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_nl(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    dequantize_iq4_nl<dst_t, dst_t *>(vx, ib, yy, tid);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_xs(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    dequantize_iq4_xs<dst_t, dst_t *>(vx, ib, yy, tid);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_mxfp4(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    dequantize_mxfp4<dst_t, dst_t *>(vx, ib, yy, tid);
}
