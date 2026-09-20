#include "mmb.cuh"
#include "unary.cuh"
#include "convert.cuh"
#include <unordered_map>
#include <map>
#include <utility>
#include "mmid.cuh"
#include <cstdlib>
#include <vector>
#include <unordered_set>

namespace {

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f  __attribute__((ext_vector_type(8)));
constexpr int MMB_BK = 64, MMB_NT = 256, MMB_LDS_STRIDE = MMB_BK + 8;

__device__ __forceinline__ uint16_t mmb_f2bf(float f) { uint32_t u = __float_as_uint(f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
__device__ __forceinline__ uint32_t mmb_pack2(float a, float b) { return (uint32_t)mmb_f2bf(a) | ((uint32_t)mmb_f2bf(b) << 16); }
__constant__ int8_t mmb_kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__device__ __forceinline__ float mmb_h2f(uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }

__global__ void mmb_cvt_f32_bf16(const float * __restrict__ x, uint16_t * __restrict__ y, const size_t n) {
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i + 8 <= n) {
        const float4 a = *(const float4 *)(x + i), b = *(const float4 *)(x + i + 4);
        uint4 o; o.x = mmb_pack2(a.x, a.y); o.y = mmb_pack2(a.z, a.w); o.z = mmb_pack2(b.x, b.y); o.w = mmb_pack2(b.z, b.w);
        *(uint4 *)(y + i) = o;
    } else {
        for (; i < n; ++i) y[i] = mmb_f2bf(x[i]);
    }
}

}
#include "mmb-quant.cuh"
namespace {

// dequantize one weight row's two consecutive IQ4_NL blocks (36 bytes) into 64 bf16 in LDS.
// LUT held in registers as (kv + 128) bytes and applied with v_perm_b32 (4 nibbles per op pair) instead of a per-lane
// indexed constant array (which lowers to one scalar-byte memory load per element). The value kv*d is produced as
// fma(kv+128, d, -128*d): -128*d is exact, so the single rounding equals RN(kv*d) -> bitwise the same BF16 as before.
__device__ __forceinline__ void mmb_dq_row36(const uint4 w0, const uint4 w1, const uint32_t w2, uint32_t * arow) {
    const uint32_t ws[9] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2};
    const float d0 = mmb_h2f((uint16_t)(ws[0] & 0xffff)), d1 = mmb_h2f((uint16_t)(ws[4] >> 16));
    const uint32_t q0[4] = { (ws[0] >> 16) | (ws[1] << 16), (ws[1] >> 16) | (ws[2] << 16), (ws[2] >> 16) | (ws[3] << 16), (ws[3] >> 16) | (ws[4] << 16) };
    const uint32_t q1[4] = { ws[5], ws[6], ws[7], ws[8] };
    // kv + 128 = {1,24,45,63,79,93,106,118,129,141,153,166,181,197,217,241} packed little-endian, 4 per dword
    const uint32_t L0 = 0x3f2d1801u, L1 = 0x766a5d4fu, L2 = 0xa6998d81u, L3 = 0xf1d9c5b5u;
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
        const float d = blk ? d1 : d0; const float md = -128.0f * d; const uint32_t * q = blk ? q1 : q0; uint32_t * out = arow + blk * 16;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t v = q[w];
            const uint32_t nib[2] = { v & 0x0F0F0F0Fu, (v >> 4) & 0x0F0F0F0Fu };
            float x[2][4];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t n = nib[h];
                const uint32_t sel = n & 0x07070707u;
                const uint32_t pA = __builtin_amdgcn_perm(L1, L0, sel);   // entries 0..7
                const uint32_t pB = __builtin_amdgcn_perm(L3, L2, sel);   // entries 8..15
                const uint32_t m  = ((n >> 3) & 0x01010101u) * 0xFFu;      // 0xFF where the nibble >= 8
                const uint32_t u  = (pA & ~m) | (pB & m);
                x[h][0] = fmaf((float)(u & 0xFFu), d, md);
                x[h][1] = fmaf((float)((u >> 8) & 0xFFu), d, md);
                x[h][2] = fmaf((float)((u >> 16) & 0xFFu), d, md);
                x[h][3] = fmaf((float)(u >> 24), d, md);
            }
            out[2*w] = mmb_pack2(x[0][0], x[0][1]); out[2*w + 1] = mmb_pack2(x[0][2], x[0][3]);
            out[8 + 2*w] = mmb_pack2(x[1][0], x[1][1]); out[8 + 2*w + 1] = mmb_pack2(x[1][2], x[1][3]);
        }
    }
}

// dequantize one weight row's two consecutive Q8_0 blocks (68 bytes: d0 qs0[32] d1 qs1[32]) into 64 bf16 in LDS
__device__ __forceinline__ void mmb_dq_row68(const uint4 w0, const uint4 w1, const uint4 w2, const uint4 w3, const uint32_t w4, uint32_t * arow) {
    const uint32_t ws[17] = {w0.x,w0.y,w0.z,w0.w, w1.x,w1.y,w1.z,w1.w, w2.x,w2.y,w2.z,w2.w, w3.x,w3.y,w3.z,w3.w, w4};
    const float d0 = mmb_h2f((uint16_t)(ws[0] & 0xffff)), d1 = mmb_h2f((uint16_t)(ws[8] >> 16));
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
        const float d = blk ? d1 : d0; uint32_t * out = arow + blk * 16;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint32_t v = blk ? ws[9 + w] : ((ws[w] >> 16) | (ws[w + 1] << 16));
            const float e0 = d * (float)(int8_t)(v      ), e1 = d * (float)(int8_t)(v >>  8);
            const float e2 = d * (float)(int8_t)(v >> 16), e3 = d * (float)(int8_t)(v >> 24);
            out[2*w] = mmb_pack2(e0, e1); out[2*w + 1] = mmb_pack2(e2, e3);
        }
    }
}

template <typename DRowFn>
__device__ __forceinline__ void mmb_store_tile(const v8f & acc, float * __restrict__ stg, float * __restrict__ D, uint16_t * __restrict__ Dh,
        const bool store_f32, const int M, DRowFn drow, const int n_base, const int m_base, const int lane) {
    const int cm = lane & 15, cn = lane >> 4;
#pragma unroll
    for (int e = 0; e < 8; ++e) { stg[(2 * e + cn) * 16 + cm] = acc[e]; }
    __syncthreads();
    const int n = lane >> 1, half = lane & 1;
    const int dr = drow(n_base + n);
    if (dr >= 0) {
        const float4 v0 = *(const float4 *)(stg + n * 16 + half * 8), v1 = *(const float4 *)(stg + n * 16 + half * 8 + 4);
        const size_t base = (size_t)dr * M + m_base + half * 8;
        const bool full = m_base + 16 <= M;
        if (store_f32) {
            if (full && (M & 3) == 0) { *(float4 *)(D + base) = v0; *(float4 *)(D + base + 4) = v1; }
            else { const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
                   for (int k = 0; k < 8; ++k) { if (m_base + half * 8 + k < M) D[base + k] = vv[k]; } }
        }
        if (Dh) {
            if (full && (M & 7) == 0) { *(uint4 *)(Dh + base) = make_uint4(mmb_pack2(v0.x, v0.y), mmb_pack2(v0.z, v0.w), mmb_pack2(v1.x, v1.y), mmb_pack2(v1.z, v1.w)); }
            else { const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
                   for (int k = 0; k < 8; ++k) { if (m_base + half * 8 + k < M) Dh[base + k] = mmb_f2bf(vv[k]); } }
        }
    }
    __syncthreads();
}
template <int BM, int BN, int WTM, int WTN, int WTYPE, bool TAIL, typename XRowFn, typename DRowFn>
__device__ __forceinline__ void mmb_tile_gemm(const uint8_t * __restrict__ Wbase, const size_t wrow_bytes, const int a_rows,
        const uint16_t * __restrict__ Xh, const int K, XRowFn xrow, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, DRowFn drow, const int m0,
        const int n_cols, uint16_t * As, uint16_t * Bs) {
    constexpr int WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    constexpr int A_ITEMS = (BM + MMB_NT - 1) / MMB_NT;
    constexpr int B_ITEMS = (BN * 8) / MMB_NT;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;
    uint4 a0[A_ITEMS], a1[A_ITEMS], a3[A_ITEMS], a4[A_ITEMS], a5[A_ITEMS], a6[A_ITEMS], a7[A_ITEMS], a8[A_ITEMS]; uint32_t a2[A_ITEMS];
    uint4 bst[B_ITEMS];
    int brow[B_ITEMS];
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; brow[i] = xrow(c >> 3); }

    int weight_ks = 0;
    auto load_regs = [&](const int ks) {
        weight_ks = ks;
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) {
            const int row = tid + i * MMB_NT;
            if (row < BM && row < a_rows) {
                if constexpr (WTYPE == 0) { const uint8_t * p = Wbase + (size_t)row * wrow_bytes + (size_t)ks * 36;
                    a0[i] = *(const uint4 *)(p); a1[i] = *(const uint4 *)(p + 16); a2[i] = *(const uint32_t *)(p + 32); }
                else if constexpr (WTYPE == 1) { const uint8_t * p = Wbase + (size_t)row * wrow_bytes + (size_t)ks * 68;
                    a0[i] = *(const uint4 *)(p); a1[i] = *(const uint4 *)(p + 16); a3[i] = *(const uint4 *)(p + 32); a4[i] = *(const uint4 *)(p + 48); a2[i] = *(const uint32_t *)(p + 64); }
                else if constexpr (WTYPE == 32 + GGML_TYPE_Q5_1) {
                    const uint4 * p = (const uint4 *)(Wbase + (size_t)row * wrow_bytes + (size_t)ks * 48);
                    a0[i] = p[0]; a1[i] = p[1]; a3[i] = p[2];
                }
                else if constexpr (WTYPE == 2) { const uint4 * p = (const uint4 *)(Wbase + (size_t)row * wrow_bytes + (size_t)ks * 128);
                    a0[i] = p[0]; a1[i] = p[1]; a3[i] = p[2]; a4[i] = p[3]; a5[i] = p[4]; a6[i] = p[5]; a7[i] = p[6]; a8[i] = p[7]; }
            } else { a0[i] = make_uint4(0,0,0,0); a1[i] = make_uint4(0,0,0,0); a3[i] = make_uint4(0,0,0,0); a4[i] = make_uint4(0,0,0,0); a5[i] = a6[i] = a7[i] = a8[i] = make_uint4(0,0,0,0); a2[i] = 0; }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {
            const int c = tid + i * MMB_NT; const int off = (c & 7) * 8;
            bst[i] = (brow[i] >= 0) ? *(const uint4 *)(Xh + (size_t)brow[i] * K + ks * MMB_BK + off) : make_uint4(0,0,0,0);
        }
    };
    auto store_lds = [&]() {
        if constexpr (WTYPE >= 32 && WTYPE != 32 + GGML_TYPE_Q5_1) mmb_load_quant_tile<WTYPE, BM, MMB_LDS_STRIDE>(Wbase, wrow_bytes, a_rows, weight_ks, As);
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) { const int row = tid + i * MMB_NT; if (row < BM) {
            if constexpr (WTYPE == 0) mmb_dq_row36(a0[i], a1[i], a2[i], (uint32_t *)(As + row * MMB_LDS_STRIDE));
            else if constexpr (WTYPE == 1) mmb_dq_row68(a0[i], a1[i], a3[i], a4[i], a2[i], (uint32_t *)(As + row * MMB_LDS_STRIDE));
            else if constexpr (WTYPE == 32 + GGML_TYPE_Q5_1) mmb_dq_q51_pair(a0[i], a1[i], a3[i], (uint32_t *)(As + row * MMB_LDS_STRIDE));
            else if constexpr (WTYPE == 2) { uint4 * d = (uint4 *)(As + row * MMB_LDS_STRIDE); d[0] = a0[i]; d[1] = a1[i]; d[2] = a3[i]; d[3] = a4[i]; d[4] = a5[i]; d[5] = a6[i]; d[6] = a7[i]; d[7] = a8[i]; } } }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; *(uint4 *)(Bs + (c >> 3) * MMB_LDS_STRIDE + (c & 7) * 8) = bst[i]; }
    };

    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

    bool jact[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) jact[j] = !TAIL || (wn * WTN + j * 16) < n_cols;   // whole fragments past the valid columns are never stored
    const int nks = K / MMB_BK;
    load_regs(0); store_lds(); __syncthreads();
    for (int ks = 0; ks < nks; ++ks) {
        if (ks + 1 < nks) load_regs(ks + 1);
#pragma unroll
        for (int kk = 0; kk < MMB_BK; kk += 16) {
            v16s a[TM], b[TN]; const int r = lane & 15;
#pragma unroll
            for (int i = 0; i < TM; ++i) { const uint16_t * p = As + (wm * WTM + i * 16 + r) * MMB_LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); a[i] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
            for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; } const uint16_t * p = Bs + (wn * WTN + j * 16 + r) * MMB_LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); b[j] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; } acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], a[i], acc[i][j]); }
        }
        __syncthreads();
        if (ks + 1 < nks) store_lds();
        __syncthreads();
    }
    // epilogue through LDS (per-wave 1 KB stage in the now-free A/B tile area); tiles are 16 rows, a_rows is a
    // multiple of 32 in this model, so whole tiles are either valid or beyond a_rows
    float * stg = (float *)((BN * MMB_LDS_STRIDE * 2 >= MMB_NT / 32 * 1024) ? Bs : As) + wave * 256;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int ml = wm * WTM + i * 16; const bool ok = ml < a_rows;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            if (ok && jact[j]) mmb_store_tile(acc[i][j], stg, D, Dh, store_f32, M, drow, wn * WTN + j * 16, m0 + ml, lane);
            else { __syncthreads(); __syncthreads(); }
        }
    }
}

template <int BM, int BN, int WTM, int WTN, int WTYPE>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_dense_kernel(const uint8_t * __restrict__ W, const uint16_t * __restrict__ Xh, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, const int K, const int T) {
    __shared__ __align__(16) uint16_t As[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    const int m0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    const size_t wrow_bytes = mmb_row_bytes<WTYPE>(K);
    mmb_tile_gemm<BM, BN, WTM, WTN, WTYPE, false>(W + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (t0 + i < T) ? t0 + i : -1; }, D, Dh, store_f32, M, [&](int i) { return (t0 + i < T) ? t0 + i : -1; }, m0, T - t0, As, Bs);
}

// Small-N (N<=8) bf16 dense: out[n,t] = sum_k w[n,k]*x[k,t]. The generic mmb tile
// pads N up to 128 and re-reads x per warp; here every thread owns one (n,t)
// output, x is staged once per k-tile by the whole block, and w is L2-resident.
template <int NMAX, int TT, int KT>
__global__ void __launch_bounds__(256, 2)
mmb_small_n_bf16_kernel(const uint16_t * __restrict__ W, const uint16_t * __restrict__ X, float * __restrict__ D,
        const int N, const int K, const int T) {
    constexpr int NT = NMAX * TT;
    static_assert(NT == 256, "kernel assumes 256 threads");
    __shared__ uint16_t xs[KT][TT];
    __shared__ uint16_t ws[NMAX][KT];
    const int tid = threadIdx.x;
    const int t0  = blockIdx.x * TT;
    const int n   = tid / TT;
    const int tlc = tid % TT;
    const int t   = t0 + tlc;
    float acc = 0.0f;
    const int nks = K / KT;
    for (int ks = 0; ks < nks; ++ks) {
        for (int i = tid; i < KT * TT; i += NT) {
            const int c = i / KT, r = i % KT, tt = t0 + c;
            xs[r][c] = (tt < T) ? X[(size_t) tt * K + ks * KT + r] : (uint16_t) 0;
        }
        for (int i = tid; i < NMAX * KT; i += NT) {
            const int nn = i / KT, rr = i % KT;
            ws[nn][rr] = (nn < N) ? W[(size_t) nn * K + ks * KT + rr] : (uint16_t) 0;
        }
        __syncthreads();
        if (n < N) {
            #pragma unroll
            for (int r = 0; r < KT; ++r) {
                const float wv = __uint_as_float(((uint32_t) ws[n][r]) << 16);
                const float xv = __uint_as_float(((uint32_t) xs[r][tlc]) << 16);
                acc += wv * xv;
            }
        }
        __syncthreads();
    }
    if (t < T && n < N) D[(size_t) t * N + n] = acc;
}

#if defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float gm_mul_rn(const float a, const float b) { float r; asm("v_mul_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r; }
__device__ __forceinline__ float gm_add_rn(const float a, const float b) { float r; asm("v_add_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r; }
#else
__device__ __forceinline__ float gm_mul_rn(const float a, const float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float gm_add_rn(const float a, const float b) { return __fadd_rn(a, b); }
#endif
__device__ __forceinline__ float gm_sigmoid(const float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float gm_bf2f(const uint16_t h) { return __uint_as_float(((uint32_t) h) << 16); }

template <int HC, int WTYPE = 0>
__global__ void __launch_bounds__(MMB_NT, 2)
hc_gate_mix_kernel(const uint8_t * __restrict__ W, const uint16_t * __restrict__ Lo, const uint16_t * __restrict__ Xn, float * __restrict__ Out,
        uint16_t * __restrict__ OutH, const bool store_f32,
        const int E, const int K, const int T, const float scale, const float bias) {
    constexpr int CH = 32, BN = 128, BM = HC * CH;
    static_assert(BM <= MMB_NT, "one A row per thread");
    __shared__ __align__(16) uint16_t As[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave & 1, wn = wave >> 1;                 // wave: 16 channels (all HC streams) x 32 tokens
    const int e0 = blockIdx.x * CH, t0 = blockIdx.y * BN;
    const size_t wrow_bytes = mmb_row_bytes<WTYPE>(K);
    constexpr int B_ITEMS = (BN * 8) / MMB_NT;
    uint4 a0 = make_uint4(0,0,0,0), a1 = make_uint4(0,0,0,0); uint32_t a2 = 0; uint4 bst[B_ITEMS]; int brow[B_ITEMS];
    const uint8_t * arow = W;
    if (tid < BM) { const int c = tid / CH, i = tid - c * CH; arow = W + (size_t)(c * E + e0 + i) * wrow_bytes; }
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; const int t = t0 + (c >> 3); brow[i] = t < T ? t : -1; }
    int weight_ks = 0;
    auto load_regs = [&](const int ks) {
        weight_ks = ks;
        if constexpr (WTYPE == 0) if (tid < BM) { const uint8_t * p = arow + (size_t)ks * 36; a0 = *(const uint4 *)(p); a1 = *(const uint4 *)(p + 16); a2 = *(const uint32_t *)(p + 32); }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; const int off = (c & 7) * 8;
            bst[i] = (brow[i] >= 0) ? *(const uint4 *)(Lo + (size_t)brow[i] * K + ks * MMB_BK + off) : make_uint4(0,0,0,0); }
    };
    auto store_lds = [&]() {
        if constexpr (WTYPE == 0) { if (tid < BM) mmb_dq_row36(a0, a1, a2, (uint32_t *)(As + tid * MMB_LDS_STRIDE)); }
        else {
            constexpr int TYPE = WTYPE == 1 ? GGML_TYPE_Q8_0 : WTYPE - 32;
            for (int row = tid / 8; row < BM; row += MMB_NT / 8) {
                const int c = row / CH, ch = row % CH;
                mmb_decode_slice<(ggml_type)TYPE>(W + (size_t)(c * E + e0 + ch) * wrow_bytes, weight_ks * 64, As + row * MMB_LDS_STRIDE, tid % 8);
            }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; *(uint4 *)(Bs + (c >> 3) * MMB_LDS_STRIDE + (c & 7) * 8) = bst[i]; }
    };
    v8f acc[HC][2];
#pragma unroll
    for (int c = 0; c < HC; ++c)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[c][j][e] = 0.f;
    const int nks = K / MMB_BK;
    load_regs(0); store_lds(); __syncthreads();
    for (int ks = 0; ks < nks; ++ks) {
        if (ks + 1 < nks) load_regs(ks + 1);
#pragma unroll
        for (int kk = 0; kk < MMB_BK; kk += 16) {
            v16s a[HC], b[2]; const int r = lane & 15;
#pragma unroll
            for (int c = 0; c < HC; ++c) { const uint16_t * p = As + (c * CH + wm * 16 + r) * MMB_LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); a[c] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
            for (int j = 0; j < 2; ++j) { const uint16_t * p = Bs + (wn * 32 + j * 16 + r) * MMB_LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); b[j] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
            for (int c = 0; c < HC; ++c)
#pragma unroll
                for (int j = 0; j < 2; ++j) acc[c][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], a[c], acc[c][j]);
        }
        __syncthreads();
        if (ks + 1 < nks) store_lds();
        __syncthreads();
    }
    // epilogue: lane holds channel (lane & 15) of the wave's 16 and tokens 2e + (lane >> 4) of each 16-token fragment
    const int cm = lane & 15, cn = lane >> 4; const int ch = e0 + wm * 16 + cm;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int t = t0 + wn * 32 + j * 16 + 2 * e + cn;
            if (t >= T) continue;
            const uint16_t * xr = Xn + (size_t)t * ((size_t)HC * E) + ch;
            float s = 0.f;
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                const float g = __uint_as_float(((uint32_t) mmb_f2bf(acc[c][j][e])) << 16);   // the gate GEMM's BF16 epilogue rounding
                const float term = gm_mul_rn(gm_bf2f(xr[(size_t)c * E]), gm_sigmoid(g));
                s = (c == 0) ? term : gm_add_rn(s, term);
            }
            const float o = scale * s + bias;
            if (store_f32) Out[(size_t)t * E + ch] = o;
            if (OutH) OutH[(size_t)t * E + ch] = mmb_f2bf(o);
        }
    }
}

template <int BM, int BN, int WTM, int WTN, int WTYPE = 0>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_routed_kernel(const uint8_t * __restrict__ W, const size_t expert_bytes, const uint16_t * __restrict__ Xh, float * __restrict__ D,
        uint16_t * __restrict__ Dh, const bool store_f32,
        const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const uint32_t * __restrict__ desc, const int M, const int K) {
    __shared__ __align__(16) uint16_t As[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    const uint32_t dsc = desc[blockIdx.y];
    if (dsc == UINT32_MAX) return;   // uniform across the block, before any barrier
    const int e = dsc & 0xffff, jt = dsc >> 16;
    const int r0 = bounds[e] + jt * BN, cnt = bounds[e + 1] - r0;
    const int m0 = blockIdx.x * BM;
    const size_t wrow_bytes = mmb_row_bytes<WTYPE>(K);
    mmb_tile_gemm<BM, BN, WTM, WTN, WTYPE, true>(W + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (i < cnt) ? ids_src[r0 + i] : -1; }, D, Dh, store_f32, M, [&](int i) { return (i < cnt) ? ids_dst[r0 + i] : -1; }, m0, cnt, As, Bs);
}

template <int BM, int BN, int WTM, int WTN, int WTYPE, bool TAIL, typename XRowFn, typename DRowFn>
__device__ __forceinline__ void mmb_tile_gemm_glu(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t wrow_bytes, const int a_rows,
        const uint16_t * __restrict__ Xh, const int K, XRowFn xrow, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, DRowFn drow, const int m0,
        const int n_cols, uint16_t * Ag, uint16_t * Au, uint16_t * Bs) {
    constexpr int WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    constexpr int A_ITEMS = (BM + MMB_NT - 1) / MMB_NT;
    constexpr int B_ITEMS = (BN * 8) / MMB_NT;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;
    uint4 g0[A_ITEMS], g1[A_ITEMS], u0[A_ITEMS], u1[A_ITEMS]; uint32_t g2[A_ITEMS], u2[A_ITEMS];
    uint4 gm[A_ITEMS], um[A_ITEMS];
    uint4 bst[B_ITEMS];
    int brow[B_ITEMS];
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; brow[i] = xrow(c >> 3); }
    int weight_ks = 0;
    auto load_regs = [&](const int ks) {
        weight_ks = ks;
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) {
            const int row = tid + i * MMB_NT;
            if constexpr (WTYPE == 0) {
            if (row < BM && row < a_rows) {
                const uint8_t * pg = Wg + (size_t)row * wrow_bytes + (size_t)ks * 36;
                const uint8_t * pu = Wu + (size_t)row * wrow_bytes + (size_t)ks * 36;
                g0[i] = *(const uint4 *)(pg); g1[i] = *(const uint4 *)(pg + 16); g2[i] = *(const uint32_t *)(pg + 32);
                u0[i] = *(const uint4 *)(pu); u1[i] = *(const uint4 *)(pu + 16); u2[i] = *(const uint32_t *)(pu + 32);
            } else { g0[i] = g1[i] = u0[i] = u1[i] = make_uint4(0,0,0,0); g2[i] = u2[i] = 0; }
            } else if constexpr (WTYPE == 32 + GGML_TYPE_Q4_K) {
                if (row < BM && row < a_rows) {
                    const uint8_t * pg = Wg + (size_t)row * wrow_bytes + (size_t)(ks / 4) * sizeof(block_q4_K);
                    const uint8_t * pu = Wu + (size_t)row * wrow_bytes + (size_t)(ks / 4) * sizeof(block_q4_K);
                    gm[i] = *(const uint4 *)pg; um[i] = *(const uint4 *)pu;
                    const int offset = 16 + (ks & 3) * 32;
                    g0[i] = *(const uint4 *)(pg + offset); g1[i] = *(const uint4 *)(pg + offset + 16);
                    u0[i] = *(const uint4 *)(pu + offset); u1[i] = *(const uint4 *)(pu + offset + 16);
                } else { gm[i] = um[i] = g0[i] = g1[i] = u0[i] = u1[i] = make_uint4(0,0,0,0); }
            }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {
            const int c = tid + i * MMB_NT; const int off = (c & 7) * 8;
            bst[i] = (brow[i] >= 0) ? *(const uint4 *)(Xh + (size_t)brow[i] * K + ks * MMB_BK + off) : make_uint4(0,0,0,0);
        }
    };
    auto store_lds = [&]() {
        if constexpr (WTYPE != 0 && WTYPE != 32 + GGML_TYPE_Q4_K) {
            constexpr int LOAD_TYPE = WTYPE == 1 ? 32 + GGML_TYPE_Q8_0 : WTYPE;
            mmb_load_quant_tile<LOAD_TYPE, BM, MMB_LDS_STRIDE>(Wg, wrow_bytes, a_rows, weight_ks, Ag);
            mmb_load_quant_tile<LOAD_TYPE, BM, MMB_LDS_STRIDE>(Wu, wrow_bytes, a_rows, weight_ks, Au);
        }

#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) {
            const int row = tid + i * MMB_NT;
            if (row < BM) {
                if constexpr (WTYPE == 0) {
                    mmb_dq_row36(g0[i], g1[i], g2[i], (uint32_t *)(Ag + row * MMB_LDS_STRIDE));
                    mmb_dq_row36(u0[i], u1[i], u2[i], (uint32_t *)(Au + row * MMB_LDS_STRIDE));
                } else if constexpr (WTYPE == 32 + GGML_TYPE_Q4_K) {
                    mmb_dq_q4k_slice(g0[i], g1[i], gm[i], weight_ks & 3, (uint32_t *)(Ag + row * MMB_LDS_STRIDE));
                    mmb_dq_q4k_slice(u0[i], u1[i], um[i], weight_ks & 3, (uint32_t *)(Au + row * MMB_LDS_STRIDE));
                }
            }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; *(uint4 *)(Bs + (c >> 3) * MMB_LDS_STRIDE + (c & 7) * 8) = bst[i]; }
    };
    v8f accg[TM][TN], accu[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) { accg[i][j][e] = 0.f; accu[i][j][e] = 0.f; }
    bool jact[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) jact[j] = !TAIL || (wn * WTN + j * 16) < n_cols;
    const int nks = K / MMB_BK;
    load_regs(0); store_lds(); __syncthreads();
    for (int ks = 0; ks < nks; ++ks) {
        if (ks + 1 < nks) load_regs(ks + 1);
#pragma unroll
        for (int kk = 0; kk < MMB_BK; kk += 16) {
            v16s ag[TM], au[TM], b[TN]; const int r = lane & 15;
#pragma unroll
            for (int i = 0; i < TM; ++i) { const int off = (wm * WTM + i * 16 + r) * MMB_LDS_STRIDE + kk;
                ag[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Ag + off), *(const uint4 *)(Ag + off + 8)});
                au[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Au + off), *(const uint4 *)(Au + off + 8)}); }
#pragma unroll
            for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; } const uint16_t * p = Bs + (wn * WTN + j * 16 + r) * MMB_LDS_STRIDE + kk;
                b[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)p, *(const uint4 *)(p + 8)}); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    if constexpr (TAIL) { if (!jact[j]) continue; }
                    accg[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], ag[i], accg[i][j]);
                    accu[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], au[i], accu[i][j]);
                }
        }
        __syncthreads();
        if (ks + 1 < nks) store_lds();
        __syncthreads();
    }
    float * stg = (float *)((BN * MMB_LDS_STRIDE * 2 >= MMB_NT / 32 * 1024) ? Bs : Ag) + wave * 256;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int ml = wm * WTM + i * 16; const bool ok = ml < a_rows;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            if (ok && jact[j]) {
                v8f v;
#pragma unroll
                for (int e = 0; e < 8; ++e) { v[e] = ggml_cuda_op_silu_single(accg[i][j][e]) * accu[i][j][e]; }
                mmb_store_tile(v, stg, D, Dh, store_f32, M, drow, wn * WTN + j * 16, m0 + ml, lane);
            } else { __syncthreads(); __syncthreads(); }
        }
    }
}

template <int BM, int BN, int WTM, int WTN, int WTYPE = 0>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_routed_glu_kernel(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t expert_bytes, const uint16_t * __restrict__ Xh,
        float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32,
        const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const uint32_t * __restrict__ desc, const int M, const int K) {
    __shared__ __align__(16) uint16_t Ag[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Au[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    const uint32_t dsc = desc[blockIdx.y];
    if (dsc == UINT32_MAX) return;
    const int e = dsc & 0xffff, jt = dsc >> 16;
    const int r0 = bounds[e] + jt * BN, cnt = bounds[e + 1] - r0;
    const int m0 = blockIdx.x * BM;
    const size_t wrow_bytes = mmb_row_bytes<WTYPE>(K);
    mmb_tile_gemm_glu<BM, BN, WTM, WTN, WTYPE, true>(Wg + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, Wu + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (i < cnt) ? ids_src[r0 + i] : -1; }, D, Dh, store_f32, M, [&](int i) { return (i < cnt) ? ids_dst[r0 + i] : -1; }, m0, cnt, Ag, Au, Bs);
}

// F32 x F32 -> F32 GEMM with F32-equivalent precision on WMMA: each operand is split into F16 hi + F16 lo at tile load and
// the product is accumulated as hi*hi + hi*lo + lo*hi in F32 (lo*lo ~2^-22 relative, dropped). Used for the F32 MoE router.
__device__ __forceinline__ void mmb_split2(float x, uint16_t & hi, uint16_t & lo) {
    hi = __builtin_bit_cast(uint16_t, (_Float16) x); lo = __builtin_bit_cast(uint16_t, (_Float16) (x - (float) __builtin_bit_cast(_Float16, hi)));
}
template <int BM, int BN, int WTM, int WTN, bool TWO, bool X16 = false>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_f32split_kernel(const float * __restrict__ W, const void * __restrict__ Xv, float * __restrict__ D, const int M, const int K, const int T) {
    const float * __restrict__ X = (const float *) Xv;
    const uint16_t * __restrict__ Xh = (const uint16_t *) Xv;
    constexpr int BKs = 32, LS = BKs + 8, WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    __shared__ __align__(16) uint16_t Ah[BM * LS], Al[BM * LS], Bh[BN * LS], Bl[BN * LS];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, wm = wave % WAVES_M, wn = wave / WAVES_M;
    const int m0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;
    constexpr int A_CH = BM * BKs / 4, B_CH = BN * BKs / 4;
    for (int k0 = 0; k0 < K; k0 += BKs) {
        for (int idx = tid; idx < A_CH; idx += MMB_NT) { const int row = idx >> 3, c4 = (idx & 7) * 4; const int m = m0 + row;
            float4 v = make_float4(0.f,0.f,0.f,0.f); if (m < M) v = *(const float4 *)(W + (size_t) m * K + k0 + c4);
            uint16_t h[4], l[4]; mmb_split2(v.x,h[0],l[0]); mmb_split2(v.y,h[1],l[1]); mmb_split2(v.z,h[2],l[2]); mmb_split2(v.w,h[3],l[3]);
            *(uint2 *)(Ah + row * LS + c4) = make_uint2((uint32_t)h[0] | ((uint32_t)h[1] << 16), (uint32_t)h[2] | ((uint32_t)h[3] << 16));
            *(uint2 *)(Al + row * LS + c4) = make_uint2((uint32_t)l[0] | ((uint32_t)l[1] << 16), (uint32_t)l[2] | ((uint32_t)l[3] << 16)); }
        for (int idx = tid; idx < B_CH; idx += MMB_NT) { const int row = idx >> 3, c4 = (idx & 7) * 4; const int t = t0 + row;
            float4 v = make_float4(0.f,0.f,0.f,0.f);
            if (t < T) {
                if constexpr (X16) {
                    // src1 is bf16-only (in-place); widen before the f16 hi/lo split
                    const uint2 u = *reinterpret_cast<const uint2 *>(Xh + (size_t) t * K + k0 + c4);
                    v.x = __bfloat162float(__builtin_bit_cast(nv_bfloat16, (uint16_t) (u.x & 0xffff)));
                    v.y = __bfloat162float(__builtin_bit_cast(nv_bfloat16, (uint16_t) (u.x >> 16)));
                    v.z = __bfloat162float(__builtin_bit_cast(nv_bfloat16, (uint16_t) (u.y & 0xffff)));
                    v.w = __bfloat162float(__builtin_bit_cast(nv_bfloat16, (uint16_t) (u.y >> 16)));
                } else {
                    v = *(const float4 *)(X + (size_t) t * K + k0 + c4);
                }
            }
            uint16_t h[4], l[4]; mmb_split2(v.x,h[0],l[0]); mmb_split2(v.y,h[1],l[1]); mmb_split2(v.z,h[2],l[2]); mmb_split2(v.w,h[3],l[3]);
            *(uint2 *)(Bh + row * LS + c4) = make_uint2((uint32_t)h[0] | ((uint32_t)h[1] << 16), (uint32_t)h[2] | ((uint32_t)h[3] << 16));
            *(uint2 *)(Bl + row * LS + c4) = make_uint2((uint32_t)l[0] | ((uint32_t)l[1] << 16), (uint32_t)l[2] | ((uint32_t)l[3] << 16)); }
        __syncthreads();
        const int r = lane & 15;
#pragma unroll
        for (int kk = 0; kk < BKs; kk += 16) {
            v16s ah[TM], al[TM], bh[TN], bl[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) { const int off = (wm * WTM + i * 16 + r) * LS + kk;
                ah[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Ah + off), *(const uint4 *)(Ah + off + 8)});
                al[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Al + off), *(const uint4 *)(Al + off + 8)}); }
#pragma unroll
            for (int j = 0; j < TN; ++j) { const int off = (wn * WTN + j * 16 + r) * LS + kk;
                bh[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Bh + off), *(const uint4 *)(Bh + off + 8)});
                bl[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Bl + off), *(const uint4 *)(Bl + off + 8)}); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bh[j], ah[i], acc[i][j]);
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bl[j], ah[i], acc[i][j]);
                    if constexpr (!TWO) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(bh[j], al[i], acc[i][j]);
                }
        }
        __syncthreads();
    }
    const int cm = lane & 15, cn = lane >> 4;
#pragma unroll
    for (int i = 0; i < TM; ++i) { const int m = m0 + wm * WTM + i * 16 + cm; if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) { const int t = t0 + wn * WTN + j * 16 + 2 * e + cn; if (t < T) D[(size_t) t * M + m] = acc[i][j][e]; } }
}

// two tile classes: experts with >= thresh rows get BN_BIG-row tiles, the rest BN_SMALL-row tiles (fewer wasted rows on tiny experts)
__global__ void mmb_build_desc2(const int32_t * __restrict__ bounds, uint32_t * __restrict__ desc_big, uint32_t * __restrict__ desc_small,
        const int E, const int nbig_max, const int nsmall_max, const int BN_BIG, const int BN_SMALL, const int thresh) {
    __shared__ int sb[1024], ss[1024];
    const int e = threadIdx.x;
    for (int i = e; i < nbig_max;   i += blockDim.x) desc_big[i]   = UINT32_MAX;
    for (int i = e; i < nsmall_max; i += blockDim.x) desc_small[i] = UINT32_MAX;
    int cnt = (e < E) ? bounds[e + 1] - bounds[e] : 0;
    const bool big = cnt >= thresh;
    const int tb = big ? (cnt + BN_BIG - 1) / BN_BIG : 0;
    const int ts = big ? 0 : (cnt + BN_SMALL - 1) / BN_SMALL;
    sb[e] = tb; ss[e] = ts;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        const int vb = (e >= off) ? sb[e - off] : 0, vs = (e >= off) ? ss[e - off] : 0;
        __syncthreads();
        sb[e] += vb; ss[e] += vs;
        __syncthreads();
    }
    const int bb = sb[e] - tb, bs = ss[e] - ts;
    for (int jt = 0; jt < tb; ++jt) { const int idx = bb + jt; if (idx < nbig_max)   desc_big[idx]   = (uint32_t)e | ((uint32_t)jt << 16); }
    for (int jt = 0; jt < ts; ++jt) { const int idx = bs + jt; if (idx < nsmall_max) desc_small[idx] = (uint32_t)e | ((uint32_t)jt << 16); }
}

struct mmb_cache_entry { const ggml_tensor * root; const void * data; size_t n; ggml_cuda_pool_alloc<uint16_t> * buf; };
static std::vector<mmb_cache_entry> g_mmb_cache;
static mmb_cache_entry g_mmb_slots[4] = {{nullptr,nullptr,0,nullptr},{nullptr,nullptr,0,nullptr},{nullptr,nullptr,0,nullptr},{nullptr,nullptr,0,nullptr}};
static std::unordered_set<const ggml_tensor *> g_mmb_bf16_only;

static size_t mmb_cache_max() { return 4; }
static const ggml_tensor * mmb_root(const ggml_tensor * t) { return t->view_src ? t->view_src : t; }
static uint16_t * mmb_cache_insert(ggml_backend_cuda_context & ctx, const ggml_tensor * t, const size_t n) {
    if (g_mmb_cache.size() >= mmb_cache_max()) { delete g_mmb_cache.front().buf; g_mmb_cache.erase(g_mmb_cache.begin()); }
    auto * buf = new ggml_cuda_pool_alloc<uint16_t>(ctx.pool(), n);
    g_mmb_cache.push_back({mmb_root(t), t->data, n, buf});
    return buf->get();
}
static const uint16_t * mmb_bf16_activation(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const size_t n, cudaStream_t stream) {
    // A bf16-only tensor holds its bf16 form in place; never re-convert it.
    if (const uint16_t * b = ggml_cuda_mmb_bf16_src(src1)) return b;
    const ggml_tensor * root = mmb_root(src1);
    for (auto & e : g_mmb_slots) if (e.buf && e.root == root && e.data == src1->data && e.n == n) return e.buf->get();
    for (auto & e : g_mmb_cache) if (e.root == root && e.data == src1->data && e.n == n) return e.buf->get();
    uint16_t * buf = mmb_cache_insert(ctx, src1, n);
    mmb_cvt_f32_bf16<<<(unsigned)((n / 8 + 255) / 256), 256, 0, stream>>>((const float *) src1->data, buf, n);
    return buf;
}

// Shadow BF16 copies of IQ4_NL dense weights: dequantised once (same LUT*scale -> BF16 RNE as mmb_dq_row36, so the
// WMMA inputs are bitwise identical) so the dense GEMM runs the dequant-free WTYPE=2 path.
__global__ void mmb_dq_q6k_bf16_kernel(const uint8_t * __restrict__ W, uint16_t * __restrict__ out, const size_t nblocks) {
    const size_t b = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const uint8_t * p  = W + b * 210;
    const uint8_t * ql = p, * qh = p + 128;
    const int8_t  * sc = (const int8_t *) (p + 192);
    const float d = mmb_h2f(*(const uint16_t *) (p + 208));
    uint16_t * o = out + b * 256;
    for (int n = 0; n < 2; ++n) {
        const uint8_t * QL = ql + 64 * n; const uint8_t * QH = qh + 32 * n; const int8_t * S = sc + 8 * n; uint16_t * Y = o + 128 * n;
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = (int8_t)((QL[l +  0] & 0xF) | (((QH[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((QL[l + 32] & 0xF) | (((QH[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((QL[l +  0] >>  4) | (((QH[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((QL[l + 32] >>  4) | (((QH[l] >> 6) & 3) << 4)) - 32;
            Y[l +  0] = mmb_f2bf(d * S[is + 0] * q1);
            Y[l + 32] = mmb_f2bf(d * S[is + 2] * q2);
            Y[l + 64] = mmb_f2bf(d * S[is + 4] * q3);
            Y[l + 96] = mmb_f2bf(d * S[is + 6] * q4);
        }
    }
}

__global__ void mmb_dq_iq4nl_bf16_kernel(const uint8_t * __restrict__ W, uint16_t * __restrict__ out, const size_t nblocks) {
    const size_t b = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const uint8_t * p = W + b * 18;
    const float d = mmb_h2f(*(const uint16_t *) p);
    const uint32_t * q = (const uint32_t *) (p + 2);
    uint32_t * o = (uint32_t *) (out + b * 32);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const uint32_t v = q[w];
        const float l0 = d * mmb_kv_iq4nl[(v      ) & 0xF], h0 = d * mmb_kv_iq4nl[(v >>  4) & 0xF];
        const float l1 = d * mmb_kv_iq4nl[(v >>  8) & 0xF], h1 = d * mmb_kv_iq4nl[(v >> 12) & 0xF];
        const float l2 = d * mmb_kv_iq4nl[(v >> 16) & 0xF], h2 = d * mmb_kv_iq4nl[(v >> 20) & 0xF];
        const float l3 = d * mmb_kv_iq4nl[(v >> 24) & 0xF], h3 = d * mmb_kv_iq4nl[(v >> 28) & 0xF];
        o[2*w] = mmb_pack2(l0, l1); o[2*w + 1] = mmb_pack2(l2, l3); o[8 + 2*w] = mmb_pack2(h0, h1); o[8 + 2*w + 1] = mmb_pack2(h2, h3);
    }
}
static std::unordered_map<const void *, uint16_t *> g_mmb_shadow;
static std::map<std::pair<const void *, const void *>, uint16_t *> g_mmb_shadow_pair;   // concat(w0, w1) along rows -> BF16 copy
static size_t g_mmb_shadow_bytes = 0;
int    mmb_shadow_mode(){
    static const int m = []() { const char * e = getenv("DFLASH_MMB_SHADOW"); return e ? atoi(e) : 2; }();
    return m;
}
bool   mmb_shadow()    { return mmb_shadow_mode() != 0; }
bool   mmb_shadow_q6k(){ return mmb_shadow_mode() >= 1; }
size_t mmb_shadow_cap(){
    static const size_t c = []() {
        const char * e = getenv("DFLASH_MMB_SHADOW_CAP_MB");
        return (size_t) (e ? atoll(e) : 6144) << 20;
    }();
    return c;
}
static bool mmb_is_resident_q6k(const ggml_tensor * w) { return w && w->type == GGML_TYPE_Q6_K && w->op == GGML_OP_NONE && w->data && w->buffer && w->ne[2] == 1 && w->ne[3] == 1 && ggml_is_contiguous(w) && w->ne[0] % 256 == 0 && w->ne[1] <= 32768; }
static bool mmb_is_resident_iq4(const ggml_tensor * w) { return w && w->type == GGML_TYPE_IQ4_NL && w->op == GGML_OP_NONE && w->data && w->buffer && w->ne[2] == 1 && w->ne[3] == 1 && ggml_is_contiguous(w); }
// Q5_K is a handful of attn_output weights; shadowing them routes the GEMM to
// cuBLAS instead of the ~7 TFLOP/s hand-rolled Q5_K mmb path.
static bool mmb_is_resident_q5k(const ggml_tensor * w) { return w && w->type == GGML_TYPE_Q5_K && w->op == GGML_OP_NONE && w->data && w->buffer && w->ne[2] == 1 && w->ne[3] == 1 && ggml_is_contiguous(w) && w->ne[0] % 256 == 0 && w->ne[1] <= 32768; }
static bool mmb_is_row_concat(const ggml_tensor * w) {
    return w && w->op == GGML_OP_CONCAT && w->type == GGML_TYPE_IQ4_NL && ggml_get_op_params_i32(w, 0) == 1 && mmb_is_resident_iq4(w->src[0]) && mmb_is_resident_iq4(w->src[1]) &&
           w->src[0]->ne[0] == w->src[1]->ne[0] && w->ne[0] == w->src[0]->ne[0] && w->ne[1] == w->src[0]->ne[1] + w->src[1]->ne[1];
}
static const uint16_t * mmb_shadow_lookup(const ggml_tensor * w) {
    if (w->op == GGML_OP_CONCAT) { auto it = g_mmb_shadow_pair.find({w->src[0]->data, w->src[1]->data}); return it == g_mmb_shadow_pair.end() ? nullptr : it->second; }
    auto it = g_mmb_shadow.find(w->data); return it == g_mmb_shadow.end() ? nullptr : it->second;
}

// Off by default: opt in with GGML_CUDA_MMB=1. bf16 WMMA dequant changes the
// rounding of prefill GEMMs relative to the MMQ int path, so it must never
// silently alter existing architectures' numerics.
bool mmb_enabled() {
    static const bool requested = []() {
        const char * env = getenv("GGML_CUDA_MMB");
        return env && atoi(env) == 1;
    }();
    if (!requested) return false;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4(cc);
}
int  mmb_min_t()   { return 512; }
int  mmb_f32split_mode(){ return 2; }
bool mmb_f32split() { return true; }
bool mmb_bf16w()    { return true; }
bool mmb_hc16()    { return true; }
int  mmb_tall_mode(){ return 2; }
bool mmb_tall()    { return mmb_tall_mode() != 0; }
bool mmb_gatemix_flag() { return true; }
bool mmb_down16_flag() { return true; }
bool mmb_glu()     { return true; }

} // namespace

// bf16 weight shadow (mmb_shadow_prepare) as a raw pointer, so a GEMM can feed
// it to cuBLAS/hipBLASLt instead of running the hand-rolled mmb kernel.
const void * ggml_cuda_mmb_shadow_ptr(const ggml_tensor * w) {
    return (!w || !mmb_shadow()) ? nullptr : (const void *) mmb_shadow_lookup(w);
}

// The bf16-only mark is set on the xn view created by hc_combine_norm, but its
// consumers usually read a reshape/cont of that view, so a plain pointer test
// misses them. Resolve by shared root tensor + view offset + data pointer.
const uint16_t * ggml_cuda_mmb_bf16_src(const ggml_tensor * t) {
    if (!t) return nullptr;
    if (g_mmb_bf16_only.count(t) > 0) return (const uint16_t *) t->data;
    const ggml_tensor * tr = mmb_root(t);
    for (const ggml_tensor * m : g_mmb_bf16_only) {
        if (mmb_root(m) == tr && t->view_offs == m->view_offs && t->data == m->data) {
            return (const uint16_t *) t->data;
        }
    }
    return nullptr;
}

const uint16_t * ggml_cuda_mmb_cache_lookup(const ggml_tensor * t) {
    // A bf16-only tensor holds its form in place, so there is no cache slot to race.
    if (const uint16_t * b = ggml_cuda_mmb_bf16_src(t)) return b;
    const ggml_tensor * root = mmb_root(t);
    for (auto & e : g_mmb_slots) if (e.buf && e.root == root && e.data == t->data) return e.buf->get();
    for (auto & e : g_mmb_cache) if (e.root == root && e.data == t->data) return e.buf->get();
    return nullptr;
}
static size_t g_mmb_slot_cap[4] = {0, 0, 0};
uint16_t * ggml_cuda_mmb_slot_reserve(ggml_backend_cuda_context & ctx, int slot, const ggml_tensor * t, size_t n) {
    mmb_cache_entry & e = g_mmb_slots[slot];
    if (e.buf && g_mmb_slot_cap[slot] < n) { delete e.buf; e.buf = nullptr; }
    if (!e.buf) { e.buf = new ggml_cuda_pool_alloc<uint16_t>(ctx.pool(), n); g_mmb_slot_cap[slot] = n; }
    e.root = mmb_root(t); e.data = t->data; e.n = n;
    return e.buf->get();
}
void ggml_cuda_mmb_marks_clear() { g_mmb_bf16_only.clear(); }
size_t ggml_cuda_mmb_marks_count() { return g_mmb_bf16_only.size(); }
void ggml_cuda_mmb_mark_bf16_only(const ggml_tensor * t) { g_mmb_bf16_only.insert(t); }
bool ggml_cuda_mmb_is_bf16_only(const ggml_tensor * t) { return g_mmb_bf16_only.count(t) > 0; }
void ggml_cuda_mmb_begin_graph() { for (auto & e : g_mmb_cache) delete e.buf; g_mmb_cache.clear(); for (auto & e : g_mmb_slots) { e.root = nullptr; e.data = nullptr; e.n = 0; } g_mmb_bf16_only.clear(); }
void ggml_cuda_mmb_release_all() {
    ggml_cuda_mmb_begin_graph();
    for (int i = 0; i < 4; ++i) { if (g_mmb_slots[i].buf) delete g_mmb_slots[i].buf; g_mmb_slots[i].buf = nullptr; g_mmb_slot_cap[i] = 0; }
    // Shadow weights are raw cudaMalloc keyed by data pointer and held for the process lifetime.
    for (auto & e : g_mmb_shadow)      { if (e.second) cudaFree(e.second); }
    for (auto & e : g_mmb_shadow_pair) { if (e.second) cudaFree(e.second); }
    g_mmb_shadow.clear();
    g_mmb_shadow_pair.clear();
    g_mmb_shadow_bytes = 0;
}
uint16_t * ggml_cuda_mmb_cache_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * t, size_t n) {
    if (!mmb_enabled() || ggml_nrows(t) < mmb_min_t()) return nullptr;
    return ggml_cuda_mmb_slot_reserve(ctx, 0, t, n);
}

bool ggml_cuda_mmb_supported_mm(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (!mmb_enabled()) return false;
    const bool quant = mmb_quant_type(src0->type);
    const bool bf16w = src0->type == GGML_TYPE_BF16 && mmb_bf16w();
    const bool f32w  = src0->type == GGML_TYPE_F32 && mmb_f32split();
    if (quant && src0->ne[0] % ggml_blck_size(src0->type) != 0) return false;
    if ((!quant && !bf16w && !f32w) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return false;
    const int64_t K = src0->ne[0], M = src0->ne[1];
    if ((f32w ? K % 32 : K % 64) != 0 || src1->ne[0] != K || dst->ne[0] != M) return false;
    const int64_t T = src1->ne[1] * src1->ne[2] * src1->ne[3];
    if (T < mmb_min_t() || T > INT32_MAX / 4) return false;
    return ggml_nrows(dst) == T;
}

bool ggml_cuda_mmb_supported_mmid(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst) {
    if (!mmb_enabled()) return false;
    if (!mmb_quant_type(src0->type) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return false;
    const int64_t K = src0->ne[0], M = src0->ne[1], E = src0->ne[2];
    if (src0->ne[3] != 1 || K % 64 != 0 || E < 1 || E > 1024) return false;
    if (K % ggml_blck_size(src0->type) != 0) return false;
    const int64_t n_used = ids->ne[0], T = ids->ne[1];
    if (src1->ne[0] != K || src1->ne[3] != 1 || src1->ne[2] != T) return false;
    if (src1->ne[1] != 1 && src1->ne[1] != n_used) return false;
    if (dst->ne[0] != M || dst->ne[1] != n_used || dst->ne[2] != T || dst->ne[3] != 1) return false;
    if (ids->nb[0] != sizeof(int32_t) || ids->ne[2] != 1 || ids->ne[3] != 1) return false;
    if (T < mmb_min_t() || n_used > 64 || (T * n_used) >> 16 >= 1024) return false;   // tile index must fit in 16 bits per expert
    return true;
}

void ggml_cuda_mul_mat_mmb(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();
    const int K = (int) src0->ne[0], M = (int) src0->ne[1];
    const int T = (int) (src1->ne[1] * src1->ne[2] * src1->ne[3]);
    if (src0->type == GGML_TYPE_F32) {
        dim3 grid((M + 127) / 128, (T + 127) / 128);
        static const bool two = true;
        // A bf16-only src1 already holds its bf16 form; widen inside the tile load.
        const uint16_t * xb = ggml_cuda_mmb_bf16_src(src1);
        const void * xp = xb ? (const void *) xb : (const void *) src1->data;
        if (two) {
            if (xb) mmb_f32split_kernel<128, 128, 32, 64, true,  true ><<<grid, MMB_NT, 0, stream>>>((const float *) src0->data, xp, (float *) dst->data, M, K, T);
            else    mmb_f32split_kernel<128, 128, 32, 64, true,  false><<<grid, MMB_NT, 0, stream>>>((const float *) src0->data, xp, (float *) dst->data, M, K, T);
        } else {
            if (xb) mmb_f32split_kernel<128, 128, 32, 64, false, true ><<<grid, MMB_NT, 0, stream>>>((const float *) src0->data, xp, (float *) dst->data, M, K, T);
            else    mmb_f32split_kernel<128, 128, 32, 64, false, false><<<grid, MMB_NT, 0, stream>>>((const float *) src0->data, xp, (float *) dst->data, M, K, T);
        }
        CUDA_CHECK(cudaGetLastError()); return;
    }
    // A bf16-only tensor already lives in the mmb cache as bf16; reading it as f32 would reinterpret bytes.
    const uint16_t * xhp = ggml_cuda_mmb_bf16_src(src1);
    if (!xhp) xhp = mmb_bf16_activation(ctx, src1, (size_t) T * K, stream);
    if (src0->type == GGML_TYPE_BF16 && M <= 8 && (K % 128) == 0) {
        if (M <= 4) {
            // NMAX=4 keeps all 256 threads active for the 4-row HC inject.
            const int grid = (int) ((T + 63) / 64);
            mmb_small_n_bf16_kernel<4, 64, 128><<<grid, 256, 0, stream>>>(
                (const uint16_t *) src0->data, xhp, (float *) dst->data, (int) M, K, (int) T);
        } else {
            const int grid = (int) ((T + 31) / 32);
            mmb_small_n_bf16_kernel<8, 32, 128><<<grid, 256, 0, stream>>>(
                (const uint16_t *) src0->data, xhp, (float *) dst->data, (int) M, K, (int) T);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const uint8_t * W = (const uint8_t *) src0->data; float * D = (float *) dst->data;
    if (mmb_tall() && src0->type == GGML_TYPE_IQ4_NL && M <= 384 && K >= 4096 && T >= 2048) {   // tall-M tile: HC down|inject [10240 -> 324], activations read once
        static const int wide = mmb_tall_mode() >= 2;
        if (wide) { dim3 grid(1, (T + 63) / 64); mmb_dense_kernel<384, 64, 96, 32, 0><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, (uint16_t *) nullptr, true, M, K, T); }
        else      { dim3 grid(1, (T + 31) / 32); mmb_dense_kernel<384, 32, 96, 16, 0><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, (uint16_t *) nullptr, true, M, K, T); }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const uint16_t * shadow_pre = ((src0->type == GGML_TYPE_IQ4_NL && mmb_shadow()) || src0->type == GGML_TYPE_Q6_K || src0->type == GGML_TYPE_Q5_K) ? mmb_shadow_lookup(src0) : nullptr;
    const bool big = (M >= 6144 && K >= 2560) || (shadow_pre && K >= 2560 && T >= 4096);
    uint16_t * Dh = (mmb_hc16() && K == 320 && M == 10240) ? ggml_cuda_mmb_slot_reserve(ctx, 1, dst, (size_t) T * M) : nullptr;
    bool store_f32 = !(Dh && ggml_cuda_mmb_is_bf16_only(dst));
    if (ggml_cuda_mmb_blk16() && !Dh && ggml_cuda_mmb_is_bf16_only(dst) && (M & 7) == 0) {
        Dh = (uint16_t *) dst->data; store_f32 = false;
    }
    dim3 grid((M + 127) / 128, big ? (T + 255) / 256 : (T + 127) / 128);
    const uint16_t * shadow = shadow_pre;

    if (shadow) {
        if (big) mmb_dense_kernel<128, 256, 64, 64, 2><<<grid, MMB_NT, 0, stream>>>((const uint8_t *) shadow, xhp, D, Dh, store_f32, M, K, T);
        else     mmb_dense_kernel<128, 128, 32, 64, 2><<<grid, MMB_NT, 0, stream>>>((const uint8_t *) shadow, xhp, D, Dh, store_f32, M, K, T);
    } else if (src0->type == GGML_TYPE_IQ4_NL) {
        if (big) mmb_dense_kernel<128, 256, 64, 64, 0><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
        else     mmb_dense_kernel<128, 128, 32, 64, 0><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
    } else if (src0->type == GGML_TYPE_Q8_0) {
        if (big) mmb_dense_kernel<128, 256, 64, 64, 1><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
        else     mmb_dense_kernel<128, 128, 32, 64, 1><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
    } else if (mmb_quant_type(src0->type)) {
        mmb_dispatch_quant(src0->type, [&](auto tag) {
            constexpr int WT = decltype(tag)::value;
            if (big) mmb_dense_kernel<128, 256, 64, 64, WT><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
            else     mmb_dense_kernel<128, 128, 32, 64, WT><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
        });
    } else {
        if (big) mmb_dense_kernel<128, 256, 64, 64, 2><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
        else     mmb_dense_kernel<128, 128, 32, 64, 2><<<grid, MMB_NT, 0, stream>>>(W, xhp, D, Dh, store_f32, M, K, T);
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_mmb_gatemix() { return mmb_gatemix_flag(); }
bool ggml_cuda_mmb_down16() { return mmb_down16_flag(); }
bool ggml_cuda_mmb_blk16() { return true; }
bool ggml_cuda_mmb_res16()  { return true; }
bool ggml_cuda_hc_gate_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * lo, const ggml_tensor * xn, ggml_tensor * dst,
        const int hc, const float scale, const float bias) {
    if (!mmb_gatemix_flag() || hc != 4 || !mmb_quant_type(w->type) || lo->type != GGML_TYPE_F32 || !ggml_is_contiguous(lo) || !ggml_is_contiguous(dst)) return false;
    const int K = (int) w->ne[0], M = (int) w->ne[1], E = (int) dst->ne[0]; const int T = (int) ggml_nrows(dst);
    if (K % ggml_blck_size(w->type) != 0) return false;
    if (K % MMB_BK != 0 || M != hc * E || E % 32 != 0 || lo->ne[0] != K || ggml_nrows(lo) != T || xn->ne[0] != M || ggml_nrows(xn) != T || T < mmb_min_t()) return false;
    const uint16_t * xn16 = ggml_cuda_mmb_cache_lookup(xn);
    cudaStream_t stream = ctx.stream();
    // The HC stream is usually handed over by hc_combine_norm; convert on the fly here, the cache serves later HC ops.
    if (!xn16) xn16 = mmb_bf16_activation(ctx, xn, (size_t) T * M, stream);
    if (!xn16) return false;
    const uint16_t * lo16 = mmb_bf16_activation(ctx, lo, (size_t) T * K, stream);
    // A bf16-only mix output stores its bf16 form in place (first half of the f32
    // buffer); the f32 copy is then never written, same contract as xn.
    const bool marked = ggml_cuda_mmb_is_bf16_only(dst);
    uint16_t * outh = marked ? (uint16_t *) dst->data
                             : ggml_cuda_mmb_slot_reserve(ctx, 3, dst, (size_t) T * E);
    const bool store_f32 = !(outh && marked);
    dim3 grid(E / 32, (T + 127) / 128);
    mmb_dispatch_quant(w->type, [&](auto tag) {
        constexpr int WT = decltype(tag)::value;
    hc_gate_mix_kernel<4, WT><<<grid, MMB_NT, 0, stream>>>((const uint8_t *) w->data, lo16, xn16, (float *) dst->data, outh, store_f32, E, K, T, scale, bias);
    });
    CUDA_CHECK(cudaGetLastError());
    return true;
}

void ggml_cuda_mul_mat_id_mmb(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();
    const int K = (int) src0->ne[0], M = (int) src0->ne[1], E = (int) src0->ne[2];
    const int ne11 = (int) src1->ne[1], T = (int) src1->ne[2], n_used = (int) ids->ne[0];
    const int n_rows_x = ne11 * T, n_rows = n_used * T;
    constexpr int BN = 128;

    const uint16_t * xhp = mmb_bf16_activation(ctx, src1, (size_t) n_rows_x * K, stream);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), E + 1);
    const int si1  = (int) (ids->nb[1] / sizeof(int32_t));
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);
    if (!ggml_cuda_launch_mm_ids_bounded(ctx, (const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            E, T, n_used, ne11, si1, sis1, /*inverse=*/false, stream)) {
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            E, T, n_used, ne11, si1, sis1, /*write_inverse=*/false, stream);
    }
    constexpr int BN_SMALL = 32, THRESH = 128;
    const int nbig_max   = n_rows / BN + E + 1;
    const int nsmall_max = E * ((THRESH + BN_SMALL - 1) / BN_SMALL) + 1;
    ggml_cuda_pool_alloc<uint32_t> desc_big(ctx.pool(), nbig_max);
    ggml_cuda_pool_alloc<uint32_t> desc_small(ctx.pool(), nsmall_max);
    mmb_build_desc2<<<1, 1024, 0, stream>>>(bounds.get(), desc_big.get(), desc_small.get(), E, nbig_max, nsmall_max, BN, BN_SMALL, THRESH);

    const uint8_t * W = (const uint8_t *) src0->data; float * D = (float *) dst->data; const size_t eb = (size_t) src0->nb[2];
    uint16_t * Dh = (mmb_down16_flag() && ggml_cuda_mmb_is_bf16_only(dst)) ? (uint16_t *) dst->data : nullptr;
    const bool store_f32 = Dh == nullptr;
    dim3 gbig((M + 127) / 128, nbig_max), gsmall((M + 127) / 128, nsmall_max);
    mmb_dispatch_quant(src0->type, [&](auto tag) {
        constexpr int WT = decltype(tag)::value;
    mmb_routed_kernel<128, BN, 32, 64, WT><<<gbig, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
    mmb_routed_kernel<128, BN_SMALL, 32, 16, WT><<<gsmall, MMB_NT, 0, stream>>>(W, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
    });
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_mmb_supported_glu(const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * glu) {
    if (!mmb_enabled() || !mmb_glu() || !gw || !uw || !src1 || !ids || !glu) return false;
    if (!mmb_quant_type(gw->type) || uw->type != gw->type) return false;
    if (!ggml_are_same_shape(gw, uw) || gw->nb[1] != uw->nb[1] || gw->nb[2] != uw->nb[2]) return false;
    if (glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0) return false;
    if (glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu) || !glu->src[0] || !glu->src[1]) return false;
    if (glu->src[0]->op != GGML_OP_MUL_MAT_ID || glu->src[1]->op != GGML_OP_MUL_MAT_ID) return false;
    if (glu->src[0]->src[0] != gw || glu->src[1]->src[0] != uw || glu->src[0]->src[1] != src1 || glu->src[1]->src[1] != src1 || glu->src[0]->src[2] != ids || glu->src[1]->src[2] != ids) return false;
    if (ggml_nelements(glu) != ggml_nelements(glu->src[0]) || glu->ne[0] != gw->ne[1]) return false;
    return ggml_cuda_mmb_supported_mmid(gw, src1, ids, glu->src[0]) && ggml_cuda_mmb_supported_mmid(uw, src1, ids, glu->src[1]);
}

void ggml_cuda_mul_mat_id_mmb_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * glu) {
    cudaStream_t stream = ctx.stream();
    const int K = (int) gw->ne[0], M = (int) gw->ne[1], E = (int) gw->ne[2];
    const int ne11 = (int) src1->ne[1], T = (int) src1->ne[2], n_used = (int) ids->ne[0];
    const int n_rows_x = ne11 * T, n_rows = n_used * T;
    constexpr int BN = 128;
    const uint16_t * xhp = mmb_bf16_activation(ctx, src1, (size_t) n_rows_x * K, stream);
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), E + 1);
    const int si1  = (int) (ids->nb[1] / sizeof(int32_t));
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);
    if (!ggml_cuda_launch_mm_ids_bounded(ctx, (const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            E, T, n_used, ne11, si1, sis1, /*inverse=*/false, stream)) {
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            E, T, n_used, ne11, si1, sis1, /*write_inverse=*/false, stream);
    }
    constexpr int BN_SMALL = 32, THRESH = 128;
    const int nbig_max   = n_rows / BN + E + 1;
    const int nsmall_max = E * ((THRESH + BN_SMALL - 1) / BN_SMALL) + 1;
    ggml_cuda_pool_alloc<uint32_t> desc_big(ctx.pool(), nbig_max);
    ggml_cuda_pool_alloc<uint32_t> desc_small(ctx.pool(), nsmall_max);
    mmb_build_desc2<<<1, 1024, 0, stream>>>(bounds.get(), desc_big.get(), desc_small.get(), E, nbig_max, nsmall_max, BN, BN_SMALL, THRESH);
    uint16_t * Dh = ggml_cuda_mmb_slot_reserve(ctx, 2, glu, (size_t) n_rows * M);
    const bool store_f32 = !ggml_cuda_mmb_is_bf16_only(glu);
    const uint8_t * Wg = (const uint8_t *) gw->data, * Wu = (const uint8_t *) uw->data; float * D = (float *) glu->data; const size_t eb = (size_t) gw->nb[2];
    dim3 gbig((M + 63) / 64, nbig_max), gsmall((M + 63) / 64, nsmall_max);
    mmb_dispatch_quant(gw->type, [&](auto tag) {
        constexpr int WT = decltype(tag)::value;
    mmb_routed_glu_kernel<64, BN, 32, 32, WT><<<gbig, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
    mmb_routed_glu_kernel<64, BN_SMALL, 16, 16, WT><<<gsmall, MMB_NT, 0, stream>>>(Wg, Wu, eb, xhp, D, Dh, store_f32, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
    });
    CUDA_CHECK(cudaGetLastError());
}

// Called from graph_optimize (outside stream capture): create the shadow for an eligible IQ4_NL dense weight.
void ggml_cuda_mmb_shadow_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w) {
    if (!w) return;
    if (mmb_is_resident_q6k(w)) {
        if (!mmb_shadow_q6k() || g_mmb_shadow.count(w->data) > 0) return;
        const size_t n = (size_t) w->ne[0] * w->ne[1], bytes = n * 2;
        if (g_mmb_shadow_bytes + bytes > mmb_shadow_cap()) { GGML_LOG_INFO("MMB_SHADOW cap reached; %s stays Q6_K\n", w->name); return; }
        uint16_t * buf = nullptr;
        if (cudaMalloc((void **) &buf, bytes) != cudaSuccess) { GGML_LOG_WARN("MMB_SHADOW alloc failed (%zu bytes)\n", bytes); return; }
        mmb_dq_q6k_bf16_kernel<<<(unsigned) ((n / 256 + 255) / 256), 256, 0, ctx.stream()>>>((const uint8_t *) w->data, buf, n / 256);
        CUDA_CHECK(cudaGetLastError());
        g_mmb_shadow[w->data] = buf; g_mmb_shadow_bytes += bytes;
        return;
    }
    if (mmb_shadow_mode() != 1) return;               // mode 2: Q6_K only
    if (mmb_is_resident_q5k(w)) {
        if (!mmb_shadow() || g_mmb_shadow.count(w->data) > 0) return;
        const size_t n = (size_t) w->ne[0] * w->ne[1], bytes = n * 2;
        if (g_mmb_shadow_bytes + bytes > mmb_shadow_cap()) { GGML_LOG_INFO("MMB_SHADOW cap reached; %s stays Q5_K\n", w->name); return; }
        // No Q5_K->bf16 dequant; go via F16 (dequantize_row_q5_K) then F16->bf16.
        const to_fp16_cuda_t to_f16  = ggml_get_to_fp16_cuda(GGML_TYPE_Q5_K);
        const to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F16);
        if (!to_f16 || !to_bf16) return;
        uint16_t * buf = nullptr;
        half * tmp = nullptr;
        if (cudaMalloc((void **) &buf, bytes) != cudaSuccess) { GGML_LOG_WARN("MMB_SHADOW alloc failed (%zu bytes)\n", bytes); return; }
        if (cudaMalloc((void **) &tmp, bytes) != cudaSuccess) { cudaFree(buf); return; }
        to_f16((const void *) w->data, (half *) tmp, (int64_t) n, ctx.stream());
        to_bf16((const void *) tmp, (nv_bfloat16 *) buf, (int64_t) n, ctx.stream());
        cudaFree(tmp);
        CUDA_CHECK(cudaGetLastError());
        g_mmb_shadow[w->data] = buf; g_mmb_shadow_bytes += bytes;
        return;
    }
    const bool concat = mmb_is_row_concat(w);
    if (!concat && !mmb_is_resident_iq4(w)) return;
    if (concat ? g_mmb_shadow_pair.count({w->src[0]->data, w->src[1]->data}) > 0 : g_mmb_shadow.count(w->data) > 0) return;
    const size_t n = (size_t) w->ne[0] * w->ne[1];
    const size_t bytes = n * 2;
    if (g_mmb_shadow_bytes + bytes > mmb_shadow_cap()) { static bool warned = false; if (!warned) { GGML_LOG_INFO("MMB_SHADOW cap reached at %.1f MB; further weights stay IQ4_NL\n", g_mmb_shadow_bytes / 1048576.0); warned = true; } return; }
    uint16_t * buf = nullptr;
    if (cudaMalloc((void **) &buf, bytes) != cudaSuccess) { GGML_LOG_WARN("MMB_SHADOW alloc failed (%zu bytes)\n", bytes); return; }
    if (concat) {
        const size_t n0 = (size_t) w->src[0]->ne[0] * w->src[0]->ne[1], n1 = (size_t) w->src[1]->ne[0] * w->src[1]->ne[1];
        mmb_dq_iq4nl_bf16_kernel<<<(unsigned) ((n0 / 32 + 255) / 256), 256, 0, ctx.stream()>>>((const uint8_t *) w->src[0]->data, buf, n0 / 32);
        mmb_dq_iq4nl_bf16_kernel<<<(unsigned) ((n1 / 32 + 255) / 256), 256, 0, ctx.stream()>>>((const uint8_t *) w->src[1]->data, buf + n0, n1 / 32);
        g_mmb_shadow_pair[{w->src[0]->data, w->src[1]->data}] = buf;
    } else {
        mmb_dq_iq4nl_bf16_kernel<<<(unsigned) ((n / 32 + 255) / 256), 256, 0, ctx.stream()>>>((const uint8_t *) w->data, buf, n / 32);
        g_mmb_shadow[w->data] = buf;
    }
    CUDA_CHECK(cudaGetLastError());
    g_mmb_shadow_bytes += bytes;
}
