#pragma once
// Q8_0 x F16 dense GEMM on the gfx1151 F16 WMMA cores, for wide prefill projections.
// Weights are dequantized to F16 as they are committed to LDS (magic-number
// construction: exact code, one rounding for q*d), activations are F16 rows
// [T][K], and the matrix cores accumulate the whole K sweep in F32 with no
// per-block rescale. Block = BM weight rows x BN tokens, BK 32-element Q8_0
// blocks per LDS stage, 8 waves = WM row groups x WN token groups.
// Derived from gufo's DenseF16GEMMKernel (MIT, github.com/gufo-org/gufo b93bd5c); its license:
//
// MIT License
//
// Copyright (c) 2026 gufo contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Self-contained so the standalone microbench can include it.

#include <cstdint>

typedef _Float16 mq_v16h __attribute__((ext_vector_type(16)));
typedef _Float16 mq_h2   __attribute__((ext_vector_type(2)));
typedef float    mq_v8f  __attribute__((ext_vector_type(8)));

__device__ __forceinline__ uint16_t mq_f2bf(float f) { uint32_t u = __float_as_uint(f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
__device__ __forceinline__ _Float16 mq_f2h_sat(float v) { return (_Float16) (fabsf(v) > 65504.0f ? copysignf(65504.0f, v) : v); }

// XT: activation rows as stored by the graph, 0 = F16, 1 = bf16, 2 = F32. Non-F16 rows are converted to F16
// (saturating, RNE) while committing to LDS, so no separate conversion pass or scratch buffer is needed.
template <int BM, int BN, int BK, int WM, int WN, int XT = 0>
__global__ void __launch_bounds__(256) mmb_q8f16_kernel(const uint8_t * __restrict__ w, const void * __restrict__ xv,
        float * __restrict__ y, uint16_t * __restrict__ yh, const int T, const int M, const int K) {
    static_assert(WM * WN == 8, "256 threads = 8 waves");
    constexpr int RT = BM / 16 / WM, TT = BN / 16 / WN;       // row / token tiles per wave
    static_assert(RT % 2 == 0 && RT * WM * 16 == BM && TT * WN * 16 == BN, "wave tiling");
    constexpr int AU = BM * BK, BU = BN * BK, AP = (AU + 255) / 256, BP = (BU + 255) / 256;
    constexpr int STR = 36;                                    // epilogue floats per token (32 rows + pad)
    constexpr int LDS_MAIN = BK * (BM + BN) * 4, LDS_EPI = 8 * 16 * STR / 4;
    __shared__ __attribute__((aligned(16))) uint4 lds[LDS_MAIN > LDS_EPI ? LDS_MAIN : LDS_EPI];
    auto sa = reinterpret_cast<uint4 (*)[BM][4]>(lds);
    auto sb = reinterpret_cast<uint4 (*)[BN][4]>(lds + BK * BM * 4);
    // Permute a row's four 16-byte chunks so a fragment read (one row per lane, 64-byte stride) spans all banks.
    auto swz = [](int r, int c) { return c ^ ((r >> 1) & 3); };

    const int nkb = K / 32;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, sl = lane & 15, hl = lane >> 4;
    const int wr = wave / WN, wt = wave % WN;
    // Rasterize in groups of G token tiles: a group's F16 activations stay cache-resident while every row tile
    // sweeps them (at T=16384 a plain row-major walk re-streams ~84 MB of activations per row tile).
    constexpr int G = 16;
    const int gx = (T + BN - 1) / BN, gy = (M + BM - 1) / BM, b = blockIdx.x;
    const int grp = b / (G * gy), rem = b % (G * gy), gcur = min(G, gx - grp * G);
    const int r0 = (rem / gcur) * BM, t0 = (grp * G + rem % gcur) * BN;

    const uint8_t * ap[AP]; bool alive[AP];
#pragma unroll
    for (int p = 0; p < AP; ++p) {
        const int idx = p * 256 + tid, r = r0 + idx / BK;
        alive[p] = idx < AU && r < M;                          // dead rows read row M-1 with a zero scale
        ap[p] = w + (size_t) (alive[p] ? r : M - 1) * nkb * 34;
    }
    constexpr int ES = XT == 2 ? 4 : 2, NC = 32 * ES / 16;         // bytes per element, 16-byte chunks per K block
    const char * x = (const char *) xv;
    const char * bp[BP];
#pragma unroll
    for (int p = 0; p < BP; ++p) {
        const int idx = p * 256 + tid, t = t0 + idx / BK;
        bp[p] = idx < BU && t < T ? x + (size_t) t * K * ES : nullptr;
    }
    uint4 ac[AP][2]; uint32_t ad[AP]; uint4 bd[BP][NC];

    auto fetch = [&](const int kb0) {
#pragma unroll
        for (int p = 0; p < AP; ++p) {
            const int kb = kb0 + (p * 256 + tid) % BK;
            const bool live = alive[p] && kb < nkb;
            const uint8_t * blk = ap[p] + (size_t) (live ? kb : nkb - 1) * 34;
            ad[p] = live ? *(const uint16_t *) blk : 0u;
            __builtin_memcpy(&ac[p][0], blk + 2, 16);
            __builtin_memcpy(&ac[p][1], blk + 18, 16);
        }
#pragma unroll
        for (int p = 0; p < BP; ++p) {
            const int kb = kb0 + (p * 256 + tid) % BK;
            const bool live = bp[p] && kb < nkb;
            const uint4 * src = (const uint4 *) (live ? bp[p] + (size_t) kb * 32 * ES : x);
#pragma unroll
            for (int c = 0; c < NC; ++c) bd[p][c] = live ? src[c] : make_uint4(0, 0, 0, 0);
        }
    };
    const mq_h2 magic = {(_Float16) -1152.0f, (_Float16) -1152.0f};
    auto commit = [&]() {
#pragma unroll
        for (int p = 0; p < AP; ++p) {
            const int idx = p * 256 + tid;
            if (idx >= AU) break;
            const int row = idx / BK, kk = idx % BK;
            const uint32_t words[8] = {ac[p][0].x, ac[p][0].y, ac[p][0].z, ac[p][0].w, ac[p][1].x, ac[p][1].y, ac[p][1].z, ac[p][1].w};
            const _Float16 d = __builtin_bit_cast(_Float16, (uint16_t) ad[p]);
            const mq_h2 d2 = {d, d};
            uint32_t h[16];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                // q ^ 0x80 = q + 128 as the low byte of F16 1024 + (q + 128); -1152 leaves q exactly, then one rounding for q*d.
                const uint32_t c = words[i] ^ 0x80808080u;
                const uint32_t p0 = __builtin_amdgcn_perm(c, 0x64646464u, 0x01050004u);
                const uint32_t p1 = __builtin_amdgcn_perm(c, 0x64646464u, 0x03070206u);
                h[2 * i]     = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(mq_h2, p0) + magic) * d2);
                h[2 * i + 1] = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(mq_h2, p1) + magic) * d2);
            }
#pragma unroll
            for (int c = 0; c < 4; ++c) sa[kk][row][swz(row, c)] = make_uint4(h[4 * c], h[4 * c + 1], h[4 * c + 2], h[4 * c + 3]);
        }
#pragma unroll
        for (int p = 0; p < BP; ++p) {
            const int idx = p * 256 + tid;
            if (idx >= BU) break;
            const int t = idx / BK, kk = idx % BK;
            if constexpr (XT == 0) {
#pragma unroll
                for (int c = 0; c < 4; ++c) sb[kk][t][swz(t, c)] = bd[p][c];
            } else {
                const uint32_t * u = (const uint32_t *) bd[p];
                uint32_t h[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    float f0, f1;
                    if constexpr (XT == 1) { f0 = __uint_as_float(u[i] << 16); f1 = __uint_as_float(u[i] & 0xffff0000u); }
                    else                   { f0 = __uint_as_float(u[2 * i]);   f1 = __uint_as_float(u[2 * i + 1]); }
                    h[i] = (uint32_t) __builtin_bit_cast(uint16_t, mq_f2h_sat(f0)) | ((uint32_t) __builtin_bit_cast(uint16_t, mq_f2h_sat(f1)) << 16);
                }
#pragma unroll
                for (int c = 0; c < 4; ++c) sb[kk][t][swz(t, c)] = make_uint4(h[4 * c], h[4 * c + 1], h[4 * c + 2], h[4 * c + 3]);
            }
        }
    };

    mq_v8f acc[RT][TT];
#pragma unroll
    for (int i = 0; i < RT; ++i)
#pragma unroll
        for (int j = 0; j < TT; ++j) acc[i][j] = mq_v8f{0, 0, 0, 0, 0, 0, 0, 0};

    fetch(0);
    for (int kb0 = 0; kb0 < nkb; kb0 += BK) {
        commit();
        __syncthreads();
        if (kb0 + BK < nkb) fetch(kb0 + BK);
        // Keep the next stage's loads ahead of the matrix work (the compiler otherwise sinks them).
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int kb = 0; kb < BK; ++kb) {
            mq_v16h alo[RT], ahi[RT];
#pragma unroll
            for (int i = 0; i < RT; ++i) {
                const int row = (wr * RT + i) * 16 + sl;
                uint4 c[4];
#pragma unroll
                for (int q = 0; q < 4; ++q) c[q] = sa[kb][row][swz(row, q)];
                __builtin_memcpy(&alo[i], &c[0], 32); __builtin_memcpy(&ahi[i], &c[2], 32);
            }
#pragma unroll
            for (int j = 0; j < TT; ++j) {
                const int t = (wt * TT + j) * 16 + sl;
                uint4 c[4];
#pragma unroll
                for (int q = 0; q < 4; ++q) c[q] = sb[kb][t][swz(t, q)];
                mq_v16h blo, bhi;
                __builtin_memcpy(&blo, &c[0], 32); __builtin_memcpy(&bhi, &c[2], 32);
#pragma unroll
                for (int i = 0; i < RT; ++i) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(alo[i], blo, acc[i][j]);
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ahi[i], bhi, acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // Transpose two row tiles at a time through LDS so each store covers 32 consecutive rows of one token.
    float * ts = reinterpret_cast<float *>(lds) + wave * 16 * STR;
#pragma unroll
    for (int i = 0; i < RT; i += 2) {
#pragma unroll
        for (int j = 0; j < TT; ++j) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {           // lane: token sl, rows 2l + hl
                ts[sl * STR + 2 * l + hl]      = acc[i][j][l];
                ts[sl * STR + 16 + 2 * l + hl] = acc[i + 1][j][l];
            }
            __builtin_amdgcn_wave_barrier();
            const int rr = r0 + (wr * RT + i) * 16, tl = lane >> 1, rl = (lane & 1) * 16;
            const int tok = t0 + (wt * TT + j) * 16 + tl;
            const float * src = ts + tl * STR + rl;
            if (tok < T) {
                const size_t o = (size_t) tok * M + rr + rl;
                if (rr + 32 <= M && (M & 7) == 0) {
                    if (y) {
#pragma unroll
                        for (int q = 0; q < 4; ++q) *(float4 *) (y + o + 4 * q) = *(const float4 *) (src + 4 * q);
                    }
                    if (yh) {
                        uint32_t hw[8];
#pragma unroll
                        for (int q = 0; q < 8; ++q) hw[q] = (uint32_t) mq_f2bf(src[2 * q]) | ((uint32_t) mq_f2bf(src[2 * q + 1]) << 16);
                        *(uint4 *) (yh + o) = make_uint4(hw[0], hw[1], hw[2], hw[3]);
                        *(uint4 *) (yh + o + 8) = make_uint4(hw[4], hw[5], hw[6], hw[7]);
                    }
                } else {
#pragma unroll
                    for (int q = 0; q < 16; ++q) {
                        if (rr + rl + q < M) {
                            if (y) y[o + q] = src[q];
                            if (yh) yh[o + q] = mq_f2bf(src[q]);
                        }
                    }
                }
            }
            __builtin_amdgcn_wave_barrier();
        }
    }
}

// One plan: 256 rows x 128 tokens (BK=2, 8x1 waves), measured best or tied on every target shape at T=2048 and
// T=16384 (docs/performance/qwen4exp-q8-dense). Rejected there: 128x256 / 64x128 tiles for M <= 512 and a K split
// for the small-grid HC down (320x10240, ~17 TFLOPS: M=320 wastes 3/8 of the second row tile; gufo uses int8 there).
// xt: 0 = F16, 1 = bf16, 2 = F32 activation rows [T][K].
static bool mmb_q8f16_launch(const uint8_t * w, const void * x, const int xt, float * y, uint16_t * yh,
        const int T, const int M, const int K, hipStream_t stream) {
    if (K % 64 != 0 || T < 96 || M < 1 || xt < 0 || xt > 2) return false;
    if (M <= 1024) {   // few output rows (shared expert, k/v): 128-row x 256-token tiles, bit-exact (0.33 -> 0.22 ms at M=640)
        const unsigned grid_s = ((T + 255) / 256) * ((M + 127) / 128);
        if (xt == 0) mmb_q8f16_kernel<128, 256, 2, 2, 4, 0><<<grid_s, 256, 0, stream>>>(w, x, y, yh, T, M, K);
        if (xt == 1) mmb_q8f16_kernel<128, 256, 2, 2, 4, 1><<<grid_s, 256, 0, stream>>>(w, x, y, yh, T, M, K);
        if (xt == 2) mmb_q8f16_kernel<128, 256, 2, 2, 4, 2><<<grid_s, 256, 0, stream>>>(w, x, y, yh, T, M, K);
        return true;
    }
    const unsigned grid = ((T + 127) / 128) * ((M + 255) / 256);
    if (xt == 0) mmb_q8f16_kernel<256, 128, 2, 8, 1, 0><<<grid, 256, 0, stream>>>(w, x, y, yh, T, M, K);
    if (xt == 1) mmb_q8f16_kernel<256, 128, 2, 8, 1, 1><<<grid, 256, 0, stream>>>(w, x, y, yh, T, M, K);
    if (xt == 2) mmb_q8f16_kernel<256, 128, 2, 8, 1, 2><<<grid, 256, 0, stream>>>(w, x, y, yh, T, M, K);
    return true;
}
