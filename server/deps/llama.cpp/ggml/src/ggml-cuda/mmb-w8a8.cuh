#pragma once
// W8A8 GEMM on the gfx1151 int8 WMMA cores: Q8_0 weights x Q8 activation tiles, for the HC down projection
// (xn [T][4*2560] -> [T][320]). Per 32-K block the int32 WMMA result is rescaled by dw*dx in registers (no barrier
// per block) into a whole-K F32 accumulator. Activation tiles hold 16 tokens x 32 K in 576 bytes: codes
// [half][token][16], F32 scales at +512; they are written by hc_combine_norm while xn is still in registers.
// Ported from gufo's W8A8BlockedWmmaGEMMKernel (MIT, github.com/gufo-org/gufo b93bd5c).
// The kernel below is a substantial portion of that software; its license:
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

#include <cstdint>

constexpr int MW8_TILE_TOKENS = 16, MW8_TILE_BYTES = 576, MW8_SCALE_OFF = 512;
typedef int mw8_i4 __attribute__((__vector_size__(4 * sizeof(int))));
typedef int mw8_i8 __attribute__((__vector_size__(8 * sizeof(int))));

__device__ __forceinline__ mw8_i8 mw8_wmma(const mw8_i4 a, const mw8_i4 b, const mw8_i8 c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
}

// Bytes of Q8 activation tiles for T tokens and K columns, padded to the GEMM's 128-token tiles.
static size_t mmb_w8a8_tile_bytes(const int T, const int K) {
    return (size_t) ((T + 127) / 128) * (128 / MW8_TILE_TOKENS) * (K / 32) * MW8_TILE_BYTES;
}

template <int BM, int BN, int BK, int WM, int WN>
__global__ void __launch_bounds__(256) mmb_w8a8_kernel(const uint8_t * __restrict__ w, const int8_t * __restrict__ xq,
        float * __restrict__ y, const int T, const int M, const int K) {
    static_assert(WM * WN == 8, "256 threads = 8 waves");
    constexpr int RT = BM / 16, TTL = BN / 16, WRT = RT / WM, WTT = TTL / WN;
    static_assert(TTL <= 8 && WRT % 2 == 0, "one wave per token tile; the epilogue pairs row tiles");
    constexpr int A_BYTES = BK * RT * 32 * 16, DW_BYTES = BK * RT * 16 * 4, B_BYTES = BK * TTL * 32 * 16, DX_BYTES = BK * TTL * 16 * 4;
    constexpr int LDS = A_BYTES + DW_BYTES + B_BYTES + DX_BYTES;
    static_assert(LDS >= 8 * 16 * 32 * 4, "epilogue transposes 16 KB");
    __shared__ __attribute__((aligned(16))) uint8_t lds[LDS];
    auto s_a  = reinterpret_cast<mw8_i4 (*)[RT][32]>(lds);
    auto s_dw = reinterpret_cast<float (*)[RT][16]>(lds + A_BYTES);
    auto s_b  = reinterpret_cast<mw8_i4 (*)[TTL][32]>(lds + A_BYTES + DW_BYTES);
    auto s_dx = reinterpret_cast<float (*)[TTL][16]>(lds + A_BYTES + DW_BYTES + B_BYTES);

    const int nkb = K / 32;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, sl = lane & 15, hl = lane >> 4;
    const int wr = wave / WN, wt = wave % WN;
    const int r_block = blockIdx.y * BM, t_block = blockIdx.x * BN;

    float acc[WRT][WTT][8];
#pragma unroll
    for (int i = 0; i < WRT; ++i)
#pragma unroll
        for (int j = 0; j < WTT; ++j)
#pragma unroll
            for (int l = 0; l < 8; ++l) acc[i][j][l] = 0.0f;

    constexpr int PF = (BM * BK) / 256;
    mw8_i4 r_q0[PF], r_q1[PF]; float r_dw[PF];
    mw8_i4 r_b[BK]; float r_dx[BK];
    const uint8_t * w_row[PF]; float row_live[PF];
#pragma unroll
    for (int p = 0; p < PF; ++p) {
        const int r = r_block + (p * 256 + tid) / BK;
        w_row[p] = w + (size_t) (r < M ? r : M - 1) * nkb * 34;
        row_live[p] = r < M ? 1.0f : 0.0f;
    }
    const int8_t * b_base = xq + (size_t) (t_block / MW8_TILE_TOKENS + wave) * nkb * MW8_TILE_BYTES;   // one wave per token tile
    const bool b_live = wave < TTL;

    auto fetch = [&](const int kb0) {
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int kb = kb0 + (p * 256 + tid) % BK;
            const uint8_t * blk = w_row[p] + (size_t) (kb < nkb ? kb : nkb - 1) * 34;
            __builtin_memcpy(&r_q0[p], blk + 2, 16);
            __builtin_memcpy(&r_q1[p], blk + 18, 16);
            r_dw[p] = (float) __builtin_bit_cast(_Float16, *(const uint16_t *) blk) * (kb < nkb ? row_live[p] : 0.0f);
        }
        if (b_live) {
#pragma unroll
            for (int i = 0; i < BK; ++i) {
                const int kb = kb0 + i;
                const int8_t * tile = b_base + (size_t) (kb < nkb ? kb : nkb - 1) * MW8_TILE_BYTES;
                r_b[i]  = reinterpret_cast<const mw8_i4 *>(tile)[lane];
                r_dx[i] = (kb < nkb ? 1.0f : 0.0f) * reinterpret_cast<const float *>(tile + MW8_SCALE_OFF)[lane & 15];
            }
        }
    };
    auto commit = [&]() {
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int idx = p * 256 + tid, rr = idx / BK, kk = idx % BK, rs = rr / 16, rl = rr % 16;
            s_a[kk][rs][rl] = r_q0[p];
            s_a[kk][rs][16 + rl] = r_q1[p];
            s_dw[kk][rs][(rl % 2 == 0) ? rl / 2 : 8 + rl / 2] = r_dw[p];
        }
        if (b_live) {
#pragma unroll
            for (int i = 0; i < BK; ++i) {
                s_b[i][wave][lane] = r_b[i];
                if (lane < 16) s_dx[i][wave][lane] = r_dx[i];
            }
        }
    };

    fetch(0);
    for (int kb0 = 0; kb0 < nkb; kb0 += BK) {
        commit();
        __syncthreads();
        if (kb0 + BK < nkb) fetch(kb0 + BK);
#pragma unroll
        for (int kb = 0; kb < BK; ++kb) {
            mw8_i4 a0[WRT], a1[WRT]; float dw[WRT][8];
#pragma unroll
            for (int i = 0; i < WRT; ++i) {
                const int rs = wr * WRT + i;
                a0[i] = s_a[kb][rs][sl];
                a1[i] = s_a[kb][rs][16 + sl];
                const float4 lo = *reinterpret_cast<const float4 *>(&s_dw[kb][rs][hl * 8]);
                const float4 up = *reinterpret_cast<const float4 *>(&s_dw[kb][rs][hl * 8 + 4]);
                dw[i][0] = lo.x; dw[i][1] = lo.y; dw[i][2] = lo.z; dw[i][3] = lo.w;
                dw[i][4] = up.x; dw[i][5] = up.y; dw[i][6] = up.z; dw[i][7] = up.w;
            }
#pragma unroll
            for (int j = 0; j < WTT; ++j) {
                const int ts = wt * WTT + j;
                const mw8_i4 b0 = s_b[kb][ts][sl], b1 = s_b[kb][ts][16 + sl];
                const float dx = s_dx[kb][ts][sl];
#pragma unroll
                for (int i = 0; i < WRT; ++i) {
                    mw8_i8 c = {0, 0, 0, 0, 0, 0, 0, 0};
                    c = mw8_wmma(a0[i], b0, c);
                    c = mw8_wmma(a1[i], b1, c);
#pragma unroll
                    for (int l = 0; l < 8; ++l) acc[i][j][l] += (dw[i][l] * dx) * (float) c[l];
                }
            }
        }
        __syncthreads();
    }

    // Transpose two row tiles at a time through LDS so each store covers 32 consecutive rows of one token.
    __syncthreads();
    float * ts = reinterpret_cast<float *>(lds) + wave * 16 * 32;
#pragma unroll
    for (int i = 0; i < WRT; i += 2) {
#pragma unroll
        for (int j = 0; j < WTT; ++j) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                ts[sl * 32 + 2 * l + hl]      = acc[i][j][l];
                ts[sl * 32 + 16 + 2 * l + hl] = acc[i + 1][j][l];
            }
            __builtin_amdgcn_wave_barrier();
            const int r0 = r_block + (wr * WRT + i) * 16, t0 = t_block + (wt * WTT + j) * 16;
            const int tl = lane >> 1, rl = (lane & 1) * 16, tok = t0 + tl;
            const float * src = ts + tl * 32 + rl;
            if (tok < T) {
                if (r0 + 32 <= M) {
#pragma unroll
                    for (int q = 0; q < 4; ++q) *(float4 *) (y + (size_t) tok * M + r0 + rl + 4 * q) = *(const float4 *) (src + 4 * q);
                } else {
                    for (int q = 0; q < 16; ++q) if (r0 + rl + q < M) y[(size_t) tok * M + r0 + rl + q] = src[q];
                }
            }
            __builtin_amdgcn_wave_barrier();
        }
    }
}

// y[T][M] F32 = Q8_0 w[M][K] x Q8 tiles of x[T][K]. HC down (M = 320): 64 x 128 blocks, 4 K blocks per stage.
static bool mmb_w8a8_launch(const uint8_t * w, const int8_t * xq, float * y, const int T, const int M, const int K, hipStream_t stream) {
    if (K % 128 != 0 || M < 32 || T < 1) return false;
    const dim3 grid((T + 127) / 128, (M + 63) / 64);
    mmb_w8a8_kernel<64, 128, 4, 2, 4><<<grid, 256, 0, stream>>>(w, xq, y, T, M, K);
    return true;
}
