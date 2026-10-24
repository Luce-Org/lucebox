// MOSS-Image DiT and text-block kernels for RDNA4.
//
// The int8 path (convrot quantization, the int8 GEMMs with their epilogues, the FFN gate, attention) is the kernels
// of the measured stable-diffusion.cpp build without ggml's plumbing; their arithmetic is unchanged. The other
// kernels repeat the float operations of the ggml ops they replace in the same order, so the results are the same.
// HIP contracts a multiply feeding an add into an FMA (and the build relies on it), so code that stands for several
// separate ggml kernels turns contraction off and writes the FMAs those kernels make explicitly.

#include "kernels.h"

#include "device_common.cuh"

#include <cstring>

namespace luce::moss_image {
namespace {

// ---------------------------------------------------------------------------------------------------------------
// Convrot quantization with the row in the registers of one wave.
//
// A row of K = 512 P values is P pairs of 256-groups. For pair p, lane l holds group 2 p + l / 16, elements
// 64 q + 4 (l % 16) + i (q, i = 0..3): in base 4 the element's digits are (i, l % 4, l / 4 % 4, q), so the stride-1
// and stride-64 butterflies are in-lane and the stride-4 and stride-16 ones exchange with lanes l ^ 1..3 and l ^ 4,
// 8, 12 (DPP row_xmask). Every 16-byte load and 4-byte store of a half-wave is contiguous.

#if defined(MOSS_GFX12)
template <int ctrl>
__device__ __forceinline__ float dpp(const float x) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), ctrl, 0xf, 0xf, false));
}

// In-lane radix-4 butterfly: a, b, c, d are the elements whose digit is 0, 1, 2, 3.
__device__ __forceinline__ void rotate_r4(float & a, float & b, float & c, float & d) {
    const float a0 = a, b0 = b, c0 = c, d0 = d;
    a =  a0 + b0 + c0 - d0;
    b =  a0 + b0 - c0 + d0;
    c =  a0 - b0 + c0 + d0;
    d = -a0 + b0 + c0 + d0;
}

// The same butterfly across the 4 lanes l ^ (m * s), m = 0..3; r is this lane's digit. The lane with digit p holds
// v_p = x_{p ^ r}, and output r takes the minus sign on v_{3 - r}: ((v0 + v1) + v2) - v3 for r = 0, and so on.
template <int s>
__device__ __forceinline__ float rotate_r4_lanes(const float x0, const int r) {
    const float x1 = dpp<0x160 | (1 * s)>(x0);
    const float x2 = dpp<0x160 | (2 * s)>(x0);
    const float x3 = dpp<0x160 | (3 * s)>(x0);
    const bool  r1 = r & 1;
    const bool  r2 = r & 2;
    const float v0 = r2 ? (r1 ? x3 : x2) : (r1 ? x1 : x0);
    const float v1 = r2 ? (r1 ? x2 : x3) : (r1 ? x0 : x1);
    const float v2 = r2 ? (r1 ? x1 : x0) : (r1 ? x3 : x2);
    const float v3 = r2 ? (r1 ? x0 : x1) : (r1 ? x2 : x3);
    return (((r == 3 ? -v0 : v0) + (r == 2 ? -v1 : v1)) + (r == 1 ? -v2 : v2)) + (r == 0 ? -v3 : v3);
}

__device__ __forceinline__ int64_t convrot_index(const int p, const int q) {
    const int lane = threadIdx.x % 32;
    return (int64_t) (2 * p + lane / 16) * 256 + 64 * q + 4 * (lane % 16);
}

// The 256-point transform of the two groups of one pair, held as v[q][i], in place.
__device__ __forceinline__ void rotate_pair(float (&v)[4][4]) {
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        rotate_r4(v[q][0], v[q][1], v[q][2], v[q][3]);
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[q][i] = rotate_r4_lanes<1>(v[q][i], lane % 4);
        }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[q][i] = rotate_r4_lanes<4>(v[q][i], lane / 4 % 4);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        rotate_r4(v[0][i], v[1][i], v[2][i], v[3][i]);
    }
}

// The sum of squares in exactly the order of a 1024-thread block norm: its thread t = e % 1024 adds elements t,
// t + 1024, ..., then the 32-lane butterflies over t % 32 and t / 32. Thread t's elements are this lane's elements of
// the pairs p0, p0 + 2, ... (one pass, p0 = 0, 1), t % 32 = i + 4 (l % 8) and t / 32 = l / 8 % 2 + 2 q + 8 (l / 16)
// + 16 p0. A pass of J pairs, pair(j) its j-th, folded over t % 32 into s[q]:
template <int J, typename Pair>
__device__ __forceinline__ void sumsq_pass(const Pair & pair, float (&s)[4]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        float t[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            t[i] = 0.0f;
#pragma unroll
            for (int j = 0; j < J; ++j) {
                t[i] = __builtin_fmaf(pair(j)[q][i], pair(j)[q][i], t[i]);  // the block norm's loop is contracted
            }
            t[i] += dpp<0x160 | 4>(t[i]);  // 16, 8, 4 are lanes ^ 4, 2, 1
            t[i] += dpp<0x160 | 2>(t[i]);
            t[i] += dpp<0x160 | 1>(t[i]);
        }
        s[q] = (t[0] + t[2]) + (t[1] + t[3]);  // 2, 1 are i ^ 2, i ^ 1
    }
}

// ... and the two passes over t / 32: 16 is p0, 8 is lane ^ 16, 4 and 2 are q ^ 2, q ^ 1, 1 is lane ^ 8.
__device__ __forceinline__ float sumsq_combine(const float (&s0)[4], const float (&s1)[4]) {
    float z[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        z[q] = s0[q] + s1[q];
        z[q] += __shfl_xor(z[q], 16, 32);
    }
    const float z0 = z[0] + z[2];
    const float z1 = z[1] + z[3];
    float sumsq = z0 + z1;
    sumsq += dpp<0x160 | 8>(sumsq);
    return sumsq;
}

// A row of K = 512 P values -> with NORM rms_norm(row) * (norm_w * (1 + mod_scale)) + mod_shift first, then the
// convrot int8 values at dst_row and the scale (absolute max / 127). load(p, q) returns this lane's 4 values of
// sub-block q of pair p. The modulated weight is ggml's two ops (scale + 1, then the multiply), each rounded; the block
// stages it (wq) and the shift (bq) in LDS. Each pair is normalized right before its rotation, so only the row and one
// pair's modulation are live at once.
template <int P, bool NORM, typename Load>
__device__ __forceinline__ void convrot_row(const Load & load, const float * wq_s, const float * bq_s, const float eps,
                                            int8_t * dst_row, float * scale_out) {
    constexpr int K    = 512 * P;
    const int     lane = threadIdx.x % 32;
    float v[P][4][4];
#pragma unroll
    for (int j = 0; j < P; ++j) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float4 t = load(j, q);
            v[j][q][0] = t.x; v[j][q][1] = t.y; v[j][q][2] = t.z; v[j][q][3] = t.w;
        }
    }
    // The rotation and the absolute max of pair j.
    float      amax       = 0.0f;
    const auto rotate_max = [&](float (&pv)[4][4]) {
        rotate_pair(pv);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                amax = fmaxf(amax, fabsf(pv[q][i]));
            }
        }
    };
    if constexpr (NORM) {
        float s0[4];
        float s1[4];
        sumsq_pass<P / 2>([&](const int j) -> const float (&)[4][4] { return v[2 * j]; }, s0);
        sumsq_pass<P / 2>([&](const int j) -> const float (&)[4][4] { return v[2 * j + 1]; }, s1);
        const float scale = rsqrtf(sumsq_combine(s0, s1) / K + eps);
#pragma unroll
        for (int j = 0; j < P; ++j) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float4 wq = *(const float4 *) (wq_s + convrot_index(j, q));
                const float4 bq = *(const float4 *) (bq_s + convrot_index(j, q));
                v[j][q][0] = (scale * v[j][q][0] * wq.x + bq.x) * (1.0f / 16.0f);
                v[j][q][1] = (scale * v[j][q][1] * wq.y + bq.y) * (1.0f / 16.0f);
                v[j][q][2] = (scale * v[j][q][2] * wq.z + bq.z) * (1.0f / 16.0f);
                v[j][q][3] = (scale * v[j][q][3] * wq.w + bq.w) * (1.0f / 16.0f);
            }
            rotate_max(v[j]);
        }
    } else {
#pragma unroll
        for (int j = 0; j < P; ++j) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    v[j][q][i] *= 1.0f / 16.0f;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < P; ++j) {
            rotate_max(v[j]);
        }
    }
    amax = wave_max(amax);
    const float row_scale = amax / 127.0f;
    if (lane == 0) {
        *scale_out = row_scale;
    }
    const float inv_scale = row_scale == 0.0f ? 0.0f : 1.0f / row_scale;
    const auto  quantize  = [&](const float x) {
        const int q = row_scale == 0.0f ? 0 : __float2int_rn(x * inv_scale);
        return (uint32_t) (uint8_t) (int8_t) max(-127, min(127, q));
    };
#pragma unroll
    for (int j = 0; j < P; ++j) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            *(uint32_t *) (dst_row + convrot_index(j, q)) =
                quantize(v[j][q][0]) | (quantize(v[j][q][1]) << 8) | (quantize(v[j][q][2]) << 16) | (quantize(v[j][q][3]) << 24);
        }
    }
}

#endif // defined(MOSS_GFX12)

// Rows of K = 512 P f32 values -> convrot int8 rows and scales, one wave per row; with NORM the modulated norm first.
template <int P, bool NORM>
__global__ void __launch_bounds__(256) convrot_kernel(const float * x, const float * norm_w, const float * mod_scale,
                                                      const float * mod_shift, const float eps, const int64_t rows,
                                                      int8_t * dst, float * scales) {
#if defined(MOSS_GFX12)
    constexpr int K = 512 * P;
    __shared__ __attribute__((aligned(16))) float wq_s[NORM ? K : 1];
    __shared__ __attribute__((aligned(16))) float bq_s[NORM ? K : 1];
    if constexpr (NORM) {
        for (int c = threadIdx.x; c < K / 4; c += blockDim.x) {
            const float4 nw = ((const float4 *) norm_w)[c];
            const float4 ms = ((const float4 *) mod_scale)[c];
            ((float4 *) wq_s)[c] = make_float4(nw.x * (ms.x + 1.0f), nw.y * (ms.y + 1.0f), nw.z * (ms.z + 1.0f), nw.w * (ms.w + 1.0f));
            ((float4 *) bq_s)[c] = ((const float4 *) mod_shift)[c];
        }
        __syncthreads();
    }
    const int64_t row = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) {
        return;
    }
    const float * x_row = x + row * K;
    convrot_row<P, NORM>([&](const int p, const int q) { return *(const float4 *) (x_row + convrot_index(p, q)); },
                         wq_s, bq_s, eps, dst + row * K, scales + row);
#else
    __builtin_trap();
#endif // defined(MOSS_GFX12)
}

template <bool NORM>
void launch_convrot(const float * x, const int64_t rows, const int64_t k, const float * norm_w, const float * mod_scale,
                    const float * mod_shift, const float eps, int8_t * xq, float * xs, hipStream_t stream) {
    const dim3 grid((unsigned) ((rows + 7) / 8));
    switch (k) {
        case 2048: convrot_kernel<4, NORM><<<grid, 256, 0, stream>>>(x, norm_w, mod_scale, mod_shift, eps, rows, xq, xs); break;
        case 3072: convrot_kernel<6, NORM><<<grid, 256, 0, stream>>>(x, norm_w, mod_scale, mod_shift, eps, rows, xq, xs); break;
        case 4096: convrot_kernel<8, NORM><<<grid, 256, 0, stream>>>(x, norm_w, mod_scale, mod_shift, eps, rows, xq, xs); break;
        default:   break;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// rms_norm(x) * w then RoPE on 128-wide head rows, one wave per row: x[i] comes from load(i), pe_row holds the
// (cos, sin) of the 64 pairs (i, i + 64), out is contiguous. The multiplies and adds stay separate, as in the unfused
// ops. The sum of squares takes the order of the ggml kernel the row stands for.

// ggml's fused rms_norm_mul_rope row (the DiT): lane l adds elements l, l + 32, ... in order, then the wave sum.
template <typename Load>
__device__ __forceinline__ float rms_scale_wave(const Load & load, const float eps) {
    const int lane = threadIdx.x % 32;
    float sumsq = 0.0f;
    for (int i = lane; i < 128; i += 32) {
        const float v = load(i);
        sumsq += v * v;
    }
    sumsq = wave_sum(sumsq);
    return rsqrtf(sumsq / 128 + eps);
}

// ggml's rms_norm in 256-thread blocks (the text blocks, whose head norms are not fused): warp w holds element
// 32 w + lane, then lanes 0..7 of one warp add the 8 warp sums, of which 4..7 are 0.
template <typename Load>
__device__ __forceinline__ float rms_scale_block256(const Load & load, const float eps) {
#pragma clang fp contract(off)
    const int lane = threadIdx.x % 32;
    float w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float v = load(32 * i + lane);
        w[i] = wave_sum(__builtin_fmaf(v, v, 0.0f));  // the block kernel's contracted tmp += x * x
    }
    return rsqrtf(((w[0] + w[2]) + (w[1] + w[3])) / 128 + eps);
}

template <typename T, typename Load>
__device__ __forceinline__ void mul_rope_row(const Load & load, const float * w, const float * pe_row, const float scale,
                                             T * out) {
    const int lane = threadIdx.x % 32;
    for (int j = lane; j < 64; j += 32) {
        const int   i0 = j;
        const int   i1 = j + 64;
        const float a  = __fmul_rn(__fmul_rn(scale, load(i0)), w[i0]);
        const float b  = __fmul_rn(__fmul_rn(scale, load(i1)), w[i1]);
        const float c  = pe_row[2 * j];
        const float s  = pe_row[2 * j + 1];
        const float o0 = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
        const float o1 = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
        out[i0] = (T) o0;
        out[i1] = (T) o1;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Int8 GEMM for RDNA4 WMMA with the consumer fused into its epilogue, so the int32 sums never go to memory.
//
// Block tile 256 features x 128 tokens x 64 k on 8 waves (4 x 2 waves of 64 x 64, 4 x 4 WMMA 16x16x16 iu8 tiles
// each). Both operands go through double-buffered LDS with one barrier per k block; the next block's global loads are
// issued before the WMMAs and stored to LDS after them. In a 64-byte k block a lane feeds bytes 32 (lane / 16) .. + 31
// of its row (weights, A) and of its column (tokens, B) as four k steps of 8 bytes: WMMA k 16 s + 8 h + j is real k
// 32 h + 8 s + j for both operands, so every k is summed exactly once and each lane reads 16-byte LDS vectors. The
// fence is LDS-only (a __syncthreads fence invalidates L0 every k block in WGP mode), the k loop is branch-free, and
// G token blocks per feature block keep neighbours on the same L2 lines.

typedef int v2i_t __attribute__((ext_vector_type(2)));
typedef int v4i_t __attribute__((ext_vector_type(4)));
typedef int v8i_t __attribute__((ext_vector_type(8)));

namespace gemm_i8 {

constexpr int WM = 4, WN = 2, BM = 64 * WM, BN = 64 * WN, NT = 32 * WM * WN;

// A split tile's k ranges (block_tile) meet in the caller's workspace (gemm_i8_split_bytes): the int32 sums of up to
// SPLIT_TILES tiles (N = 9216, the qkv linear) in the accumulators' layout, then how many ranges of each have arrived.
constexpr int SPLIT = 4, SPLIT_TILES = 36, SPLIT_SUMS = SPLIT_TILES * NT * 128;

#if defined(MOSS_GFX12)
constexpr int BK = 64, WS = BK + 16, CPR = BK / 16, CA = BM * CPR / NT, CB = BN * CPR / NT, RPS = NT / CPR;
constexpr int LDS = 2 * (BM + BN) * WS;  // 61440 bytes: WS = 80 makes 16 lanes' 16-byte reads hit 64 different banks
constexpr int G = 4;

__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Thread threadIdx.x stages the 16 bytes at k a_col() of tile rows a_row() + RPS i, i < CA, of the weights and of
// tokens t0 + a_row() + RPS i, i < CB.
__device__ __forceinline__ int a_row() {
    return threadIdx.x / CPR;
}

__device__ __forceinline__ int a_col() {
    return 16 * (threadIdx.x % CPR);
}

// A wave's tile is TI x TJ WMMA tiles: 64 x 64 (TI = TJ = 4) on 4 x 2 waves, or 128 x 32 (TI = 8, TJ = 2) on 2 x 4
// waves, which gives a wave whole 128-feature rows. acc[i][j]: lane (h = lane / 16, r = lane % 16) of wave (wm, wn) =
// (wave % (BM / (16 TI)), wave / (BM / (16 TI))) holds tile rows 16 TI wm + 16 i + 8 h + e (e = 0..7) of token
// t0 + 16 TJ wn + 16 j + r. Token rows past T read row T - 1 and must be dropped by the caller. asrc is this thread's
// weight row for tile row a_row() at k a_col(), astep the distance to its next one (tile row + RPS). The loop runs nkb
// k blocks from k0, around the end (a start staggered by block runs the long k loop of the FFN down linear ~8% faster).
// IDLE: the wave only stages its share of the tiles and leaves acc zero (see mainloop_wave).
template <int TI, int TJ, bool IDLE = false>
__device__ __forceinline__ void mainloop(const int8_t * __restrict__ asrc, const size_t astep,
                                         const int8_t * __restrict__ X, const int T, const int K, const int t0,
                                         const int k0, const int nkb, int8_t * smem, v8i_t (&acc)[TI][TJ]) {
    static_assert(TI * TJ == 16, "a wave holds 16 WMMA tiles");
    constexpr int WMv = BM / (16 * TI);
    int8_t * sa = smem;                // [2][BM * WS]
    int8_t * sb = smem + 2 * BM * WS;  // [2][BN * WS]
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const int wm = wave % WMv, wn = wave / WMv, h = lane / 16, r = lane % 16;
    const int8_t * bsrc[CB];
#pragma unroll
    for (int i = 0; i < CB; ++i) {
        bsrc[i] = X + (size_t) min(t0 + a_row() + RPS * i, T - 1) * K + a_col();
    }
    const int ldst = a_row() * WS + a_col();
    const int nk   = K / BK;
    int       kn   = k0;  // the k block loaded next

#pragma unroll
    for (int i = 0; i < TI; ++i) {
#pragma unroll
        for (int j = 0; j < TJ; ++j) {
            acc[i][j] = v8i_t{0, 0, 0, 0, 0, 0, 0, 0};
        }
    }
    v4i_t ra[CA], rb[CB];
#pragma unroll
    for (int i = 0; i < CA; ++i) {
        ra[i] = *(const v4i_t *) (asrc + astep * i + BK * kn);
    }
#pragma unroll
    for (int i = 0; i < CB; ++i) {
        rb[i] = *(const v4i_t *) (bsrc[i] + BK * kn);
    }
#pragma unroll
    for (int i = 0; i < CA; ++i) {
        *(v4i_t *) &sa[ldst + RPS * i * WS] = ra[i];
    }
#pragma unroll
    for (int i = 0; i < CB; ++i) {
        *(v4i_t *) &sb[ldst + RPS * i * WS] = rb[i];
    }
    lds_barrier();

    for (int kb = 0; kb < nkb; ++kb) {
        const int buf = kb & 1;
        kn = kn + 1 == nk ? 0 : kn + 1;  // the last trip reloads the first block so the loop stays branch-free
#pragma unroll
        for (int i = 0; i < CA; ++i) {
            ra[i] = *(const v4i_t *) (asrc + astep * i + BK * kn);
        }
#pragma unroll
        for (int i = 0; i < CB; ++i) {
            rb[i] = *(const v4i_t *) (bsrc[i] + BK * kn);
        }
        __builtin_amdgcn_sched_barrier(0);
        const int8_t * ca = sa + buf * BM * WS;
        const int8_t * cb = sb + buf * BN * WS;
#pragma unroll
        for (int q = 0; q < (IDLE ? 0 : BK / 32); ++q) {
            v4i_t a[TI], b[TJ];
#pragma unroll
            for (int i = 0; i < TI; ++i) {
                a[i] = *(const v4i_t *) &ca[(16 * TI * wm + 16 * i + r) * WS + BK / 2 * h + 16 * q];
            }
#pragma unroll
            for (int j = 0; j < TJ; ++j) {
                b[j] = *(const v4i_t *) &cb[(16 * TJ * wn + 16 * j + r) * WS + BK / 2 * h + 16 * q];
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
#pragma unroll
                for (int i = 0; i < TI; ++i) {
#pragma unroll
                    for (int j = 0; j < TJ; ++j) {
                        const v2i_t af = s == 0 ? v2i_t{a[i].x, a[i].y} : v2i_t{a[i].z, a[i].w};
                        const v2i_t bf = s == 0 ? v2i_t{b[j].x, b[j].y} : v2i_t{b[j].z, b[j].w};
                        acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, af, true, bf, acc[i][j], false);
                    }
                }
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        int8_t * na = sa + (buf ^ 1) * BM * WS;
        int8_t * nb = sb + (buf ^ 1) * BN * WS;
#pragma unroll
        for (int i = 0; i < CA; ++i) {
            *(v4i_t *) &na[ldst + RPS * i * WS] = ra[i];
        }
#pragma unroll
        for (int i = 0; i < CB; ++i) {
            *(v4i_t *) &nb[ldst + RPS * i * WS] = rb[i];
        }
        lds_barrier();
    }
}

// The main loop of a wave whose tokens (16 TJ from t0 + 16 TJ wn) all lie past T, in a partial last token block, skips
// the products: its sums would be dropped.
template <int TI, int TJ>
__device__ __forceinline__ void mainloop_wave(const int8_t * __restrict__ asrc, const size_t astep,
                                              const int8_t * __restrict__ X, const int T, const int K, const int t0,
                                              const int k0, const int nkb, int8_t * smem, v8i_t (&acc)[TI][TJ]) {
    const int wn = __builtin_amdgcn_readfirstlane(threadIdx.x / 32 / (BM / (16 * TI)));
    if (t0 + 16 * TJ * wn >= T) {
        mainloop<TI, TJ, true>(asrc, astep, X, T, K, t0, k0, nkb, smem, acc);
    } else {
        mainloop<TI, TJ, false>(asrc, astep, X, T, K, t0, k0, nkb, smem, acc);
    }
}

// The plain weight rows of tile rows f0 + a_row() + RPS i.
__device__ __forceinline__ const int8_t * weight_rows(const int8_t * W, const int K, const int f0) {
    return W + (size_t) (f0 + a_row()) * K + a_col();
}

// This block's tile: weight rows f0 .., tokens t0 .., nkb k blocks from k0 (around the end); returns the tile's index
// when the block is one k range of a split tile (finished by split_reduce), else -1. Each feature block takes G token
// blocks at a time so that neighbours share L2 lines. A partial last token block (the text tokens past the whole
// blocks of image tokens) would hold the GPU for a whole round of tiles for a few tokens: with split, its tiles come
// after all the others as SPLIT k ranges each, which fill in as the other tiles finish.
__device__ __forceinline__ int block_tile(const int N, const int T, const int K, const int stagger, const bool split,
                                          int & f0, int & t0, int & k0, int & nkb) {
    const int nf = N / BM, nk = K / BK, nt = split ? T / BN : (T + BN - 1) / BN, regular = nf * nt;
    if ((int) blockIdx.x < regular) {
        const int per = G * nf, group = blockIdx.x / per, first = group * G, gs = min(nt - first, G), in = blockIdx.x % per;
        f0  = (in / gs) * BM;
        t0  = (first + in % gs) * BN;
        k0  = (int) (blockIdx.x * stagger % nk);
        nkb = nk;
        return -1;
    }
    const int i = blockIdx.x - regular;
    f0  = (i / SPLIT) * BM;
    t0  = nt * BN;
    nkb = nk / SPLIT;
    k0  = (i % SPLIT) * nkb;
    return i / SPLIT;
}

// The end of a split tile's k range: the ranges add their sums into the tile's part of the workspace `split` (int32
// sums are exact in any order) in the accumulators' own layout, so a wave's atomics are whole cache lines, for the
// 16-token columns with a token below T (the rest is dropped anyway). The range that arrives last takes the totals
// into acc, leaving the sums and the count zero for the next launch, and returns true to run the epilogue; the others
// return false.
template <int TI, int TJ>
__device__ __forceinline__ bool split_reduce(v8i_t (&acc)[TI][TJ], int32_t * split, const int tile, const int t0,
                                             const int T) {
    __shared__ int last;
    const int  lane = threadIdx.x % 32, wave = threadIdx.x / 32, wn = wave / (BM / (16 * TI));
    int32_t *  ws       = split + (size_t) tile * NT * 128 + wave * 32 * 128 + lane;  // + 32 ((i TJ + j) 8 + e)
    unsigned * arrivals = (unsigned *) (split + SPLIT_SUMS) + tile;
#pragma unroll
    for (int j = 0; j < TJ; ++j) {
        if (t0 + 16 * TJ * wn + 16 * j < T) {
#pragma unroll
            for (int i = 0; i < TI; ++i) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    atomicAdd(ws + 32 * ((i * TJ + j) * 8 + e), acc[i][j][e]);
                }
            }
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(arrivals, 1u) == SPLIT - 1;
    }
    __syncthreads();
    if (!last) {
        return false;
    }
    __threadfence();
#pragma unroll
    for (int j = 0; j < TJ; ++j) {
        if (t0 + 16 * TJ * wn + 16 * j < T) {
#pragma unroll
            for (int i = 0; i < TI; ++i) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    acc[i][j][e] = atomicExch(ws + 32 * ((i * TJ + j) * 8 + e), 0);
                }
            }
        }
    }
    if (threadIdx.x == 0) {
        *arrivals = 0;
    }
    return true;
}

// The int32 sum times the token's activation scale and the feature's weight scale. contract(off): the stand-alone
// dequantization these GEMMs replace keeps the two multiplies apart from what follows.
__device__ __forceinline__ float dequant(const int32_t sum, const float act, const float ws) {
#pragma clang fp contract(off)
    float value = (float) sum * act;
    value *= ws;
    return value;
}

// a * b + c rounded once to f16 (v_fma_mixlo_f16, f32 sources), the instruction the compiler makes of a contracted
// a * b + c cast to f16.
__device__ __forceinline__ _Float16 fma_mix_f16(const float a, const float b, const float c) {
    uint32_t d;
    asm("v_fma_mixlo_f16 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    return __builtin_bit_cast(_Float16, (uint16_t) d);
}

__device__ __forceinline__ void unpack8(const float * p, float (&v)[8]) {
    const float4 a = *(const float4 *) p;
    const float4 b = *(const float4 *) (p + 4);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
#endif // defined(MOSS_GFX12)

// Whether a launch splits its partial last token block (block_tile): when it has one, its feature blocks fit the
// workspace and its k blocks divide by SPLIT.
bool split_tail(const int64_t n, const int64_t rows, const int64_t k) {
    return rows % BN != 0 && n / BM <= SPLIT_TILES && k % (64 * SPLIT) == 0;
}

int64_t blocks(const int64_t n, const int64_t rows, const bool split = false) {
    return (n / BM) * (split ? rows / BN + SPLIT : (rows + BN - 1) / BN);
}

} // namespace gemm_i8

// The qkv linear and the attention prologue. The 256 features of a block are two heads of q, k or v (heads * 128 is a
// multiple of 256), and each wave holds one head's whole 128-feature rows for 32 tokens (128 x 32 wave tiles), so the
// epilogue needs no LDS: v rows are cast straight from the accumulators, and q and k rows get the fused row kernel's
// rms_norm and RoPE in registers. Lane (h, r) holds features 16 i + 8 h + e of its token; the row kernel's lane l, which
// adds elements l, l + 32, l + 64, l + 96, is (i0, h, e) = (l / 16, l / 8 % 2, l % 8) with tiles i0, i0 + 2, i0 + 4,
// i0 + 6, so its wave sum (lanes ^ 16, 8, 4, 2, 1) adds over i0 in the lane, then with lane ^ 16, then over e in the
// lane; a RoPE pair (p, p + 64) is tiles i and i + 4 of one lane.
__global__ void __launch_bounds__(gemm_i8::NT) gemm_i8_qkv_attention_kernel(
        const int8_t * __restrict__ W, const int8_t * __restrict__ X, const int T, const int K, int32_t * split,
        const float * __restrict__ act_scales, const float * __restrict__ weight_scales,
        const float * __restrict__ q_w, const float * __restrict__ k_w, const float eps, const float * __restrict__ pe,
        const int heads, float * __restrict__ q, _Float16 * __restrict__ k, _Float16 * __restrict__ v) {
#if defined(MOSS_GFX12)
    using namespace gemm_i8;
    constexpr int head_dim = 128, TI = 8, TJ = 2, WMv = BM / (16 * TI);
    __shared__ __attribute__((aligned(16))) int8_t smem[LDS];
    const int dim = heads * head_dim, N = 3 * dim;
    int f0, t0, k0, nkb;
    const int tile = block_tile(N, T, K, 0, split != nullptr, f0, t0, k0, nkb);
    v8i_t acc[TI][TJ];
    mainloop_wave(weight_rows(W, K, f0), (size_t) RPS * K, X, T, K, t0, k0, nkb, smem, acc);
    if (tile >= 0 && !split_reduce(acc, split, tile, t0, T)) {
        return;
    }

    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32, wm = wave % WMv, wn = wave / WMv, h = lane / 16, r = lane % 16;
    const int part = f0 / dim, head = (f0 - part * dim) / head_dim + wm;
    const float * ws_lane = weight_scales + f0 + head_dim * wm + 8 * h;  // + 16 i + e
    const auto swap16 = [](const float x) {
        return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, x), __builtin_bit_cast(int, x),
                                                                      0x76543210, 0xfedcba98, false, false));
    };

    if (part == 2) {
#pragma unroll
        for (int j = 0; j < TJ; ++j) {
            const int t = t0 + 32 * wn + 16 * j + r;
            if (t >= T) {
                continue;
            }
            const float act = act_scales[t];
#pragma unroll
            for (int i = 0; i < TI; ++i) {
                float wsv[8];
                unpack8(ws_lane + 16 * i, wsv);
                _Float16 o[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    o[e] = (_Float16) dequant(acc[i][j][e], act, wsv[e]);
                }
                *(v4i_t *) (v + ((size_t) head * T + t) * head_dim + 16 * i + 8 * h) = *(const v4i_t *) o;
            }
        }
        return;
    }
    const float * qk_w = (part == 0 ? q_w : k_w) + 8 * h;  // + 16 i + e
#pragma unroll
    for (int j = 0; j < TJ; ++j) {
        const int t  = t0 + 32 * wn + 16 * j + r;
        const int tc = min(t, T - 1);  // rows past T are computed (lane ^ 16 shares the token) but not stored
        const float act = act_scales[tc];
        float x[TI][8];
#pragma unroll
        for (int i = 0; i < TI; ++i) {
            float wsv[8];
            unpack8(ws_lane + 16 * i, wsv);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                x[i][e] = dequant(acc[i][j][e], act, wsv[e]);
            }
        }
        // rms_scale_wave: each row-kernel lane's sum in its order, then the wave sum
        float part_sum[2][8];
#pragma unroll
        for (int i0 = 0; i0 < 2; ++i0) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float sumsq = 0.0f;
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    sumsq = __builtin_fmaf(x[i0 + 2 * m][e], x[i0 + 2 * m][e], sumsq);  // the row loop's contracted sum
                }
                part_sum[i0][e] = sumsq;
            }
        }
        float sum8[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            sum8[e] = part_sum[0][e] + part_sum[1][e];  // lane ^ 16
            sum8[e] += swap16(sum8[e]);                  // lane ^ 8
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            sum8[e] += sum8[e + 4];  // lane ^ 4
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            sum8[e] += sum8[e + 2];  // lane ^ 2
        }
        const float sumsq = sum8[0] + sum8[1];  // lane ^ 1
        const float scale = rsqrtf(sumsq / 128 + eps);
        // mul_rope_row: pair p = 16 i + 8 h + e, i < 4, with p + 64
#pragma unroll
        for (int i = 0; i < TI / 2; ++i) {
            float w0[8], w1[8], cs0[8], cs1[8];
            unpack8(qk_w + 16 * i, w0);
            unpack8(qk_w + 16 * i + 64, w1);
            unpack8(pe + (size_t) tc * head_dim + 2 * (16 * i + 8 * h), cs0);
            unpack8(pe + (size_t) tc * head_dim + 2 * (16 * i + 8 * h) + 8, cs1);
            float a[8], b[8], c[8], sn[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                a[e]  = __fmul_rn(__fmul_rn(scale, x[i][e]), w0[e]);
                b[e]  = __fmul_rn(__fmul_rn(scale, x[i + 4][e]), w1[e]);
                c[e]  = e < 4 ? cs0[2 * e] : cs1[2 * e - 8];
                sn[e] = e < 4 ? cs0[2 * e + 1] : cs1[2 * e - 7];
            }
            if (t >= T) {
                continue;
            }
            const size_t row = ((size_t) head * T + t) * head_dim + 16 * i + 8 * h;
            if (part == 0) {
                // the row kernel's contracted (a c - b s, a s + b c): fma(a, c, -(b s)) and fma(b, c, a s)
                float o0[8], o1[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    o0[e] = __builtin_fmaf(a[e], c[e], -__fmul_rn(b[e], sn[e]));
                    o1[e] = __builtin_fmaf(b[e], c[e], __fmul_rn(a[e], sn[e]));
                }
                *(float4 *) (q + row)          = make_float4(o0[0], o0[1], o0[2], o0[3]);
                *(float4 *) (q + row + 4)      = make_float4(o0[4], o0[5], o0[6], o0[7]);
                *(float4 *) (q + row + 64)     = make_float4(o1[0], o1[1], o1[2], o1[3]);
                *(float4 *) (q + row + 64 + 4) = make_float4(o1[4], o1[5], o1[6], o1[7]);
            } else {
                // ... rounded once to f16, as the row kernel's v_fma_mixlo_f16
                _Float16 h0[8], h1[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    h0[e] = fma_mix_f16(a[e], c[e], -__fmul_rn(b[e], sn[e]));
                    h1[e] = fma_mix_f16(b[e], c[e], __fmul_rn(a[e], sn[e]));
                }
                *(v4i_t *) (k + row)      = *(const v4i_t *) h0;
                *(v4i_t *) (k + row + 64) = *(const v4i_t *) h1;
            }
        }
    }
#else
    __builtin_trap();
#endif // defined(MOSS_GFX12)
}

// A gated residual from the accumulators: x += dequantized sums * gate. Each lane reads the 8 residual values it
// writes, all four of a token's before any store.
__global__ void __launch_bounds__(gemm_i8::NT) gemm_i8_gated_residual_kernel(
        const int8_t * __restrict__ W, const int8_t * __restrict__ X, const int N, const int T, const int K, const int S,
        int32_t * split, const float * __restrict__ act_scales, const float * __restrict__ weight_scales,
        const float * __restrict__ gate, float * x) {
#if defined(MOSS_GFX12)
    using namespace gemm_i8;
    __shared__ __attribute__((aligned(16))) int8_t smem[LDS];
    int f0, t0, k0, nkb;
    const int tile = block_tile(N, T, K, S, split != nullptr, f0, t0, k0, nkb);
    v8i_t acc[4][4];
    mainloop_wave(weight_rows(W, K, f0), (size_t) RPS * K, X, T, K, t0, k0, nkb, smem, acc);
    if (tile >= 0 && !split_reduce(acc, split, tile, t0, T)) {
        return;
    }

    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32, wm = wave % WM, wn = wave / WM, h = lane / 16, r = lane % 16;
    // block constants in LDS (the main loop ends on a barrier): weight scale and gate per feature, act scale per token
    float * cw = (float *) smem;
    float * cg = cw + BM;
    float * ca = cg + BM;
    cw[threadIdx.x] = weight_scales[f0 + threadIdx.x];
    cg[threadIdx.x] = gate[f0 + threadIdx.x];
    if (threadIdx.x < BN) {
        ca[threadIdx.x] = act_scales[min(t0 + (int) threadIdx.x, T - 1)];
    }
    lds_barrier();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int t = t0 + 64 * wn + 16 * j + r;
        if (t >= T) {
            continue;
        }
        const float act = ca[64 * wn + 16 * j + r];
        float rv[4][8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            unpack8(x + (size_t) t * N + f0 + 64 * wm + 16 * i + 8 * h, rv[i]);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int fl = 64 * wm + 16 * i + 8 * h;
            float wsv[8], gv[8], o[8];
            unpack8(cw + fl, wsv);
            unpack8(cg + fl, gv);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                o[e] = __fadd_rn(rv[i][e], __fmul_rn(dequant(acc[i][j][e], act, wsv[e]), gv[e]));
            }
            float * out = x + (size_t) t * N + f0 + fl;
            *(float4 *) out       = make_float4(o[0], o[1], o[2], o[3]);
            *(float4 *) (out + 4) = make_float4(o[4], o[5], o[6], o[7]);
        }
    }
#else
    __builtin_trap();
#endif // defined(MOSS_GFX12)
}

// The plain int32 sums [T][N] of an int8 linear, for a consumer that needs a whole row of them (the FFN gate), stored
// non-temporal (they are read once, by the next kernel). Rows past T are not written.
__global__ void __launch_bounds__(gemm_i8::NT) gemm_i8_i32_kernel(const int8_t * __restrict__ W,
                                                                  const int8_t * __restrict__ X, const int N, const int T,
                                                                  const int K, int32_t * __restrict__ sums) {
#if defined(MOSS_GFX12)
    using namespace gemm_i8;
    __shared__ __attribute__((aligned(16))) int8_t smem[LDS];
    int f0, t0, k0, nkb;
    block_tile(N, T, K, 0, false, f0, t0, k0, nkb);
    v8i_t acc[4][4];
    mainloop_wave(weight_rows(W, K, f0), (size_t) RPS * K, X, T, K, t0, k0, nkb, smem, acc);

    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32, wm = wave % WM, wn = wave / WM, h = lane / 16, r = lane % 16;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int t = t0 + 64 * wn + 16 * j + r;
        if (t >= T) {
            continue;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int32_t * out = sums + (size_t) t * N + f0 + 64 * wm + 16 * i + 8 * h;
            __builtin_nontemporal_store(v4i_t{acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]}, (v4i_t *) out);
            __builtin_nontemporal_store(v4i_t{acc[i][j][4], acc[i][j][5], acc[i][j][6], acc[i][j][7]}, (v4i_t *) (out + 4));
        }
    }
#else
    __builtin_trap();
#endif // defined(MOSS_GFX12)
}

// ---------------------------------------------------------------------------------------------------------------
// The FFN gate (situ on the dequantized int32 sums of the up/gate linear) quantized with convrot for the down linear:
// 8 waves per row, each walking its G 256-groups one at a time and keeping the rotated values in registers until the
// row's absolute max is known.
//
// In a group, lane l holds elements i + 4 (l % 4) + 16 (l / 4 % 4) + 64 (j + 2 (l / 16)), i = 0..3, j = 0..1: the
// base-4 digits are (i, l % 4, l / 4 % 4, j + 2 (l / 16)), so the stride-1 butterfly is in-lane, stride 4 and 16 are
// DPP row_xmask exchanges, and stride 64 pairs each lane with lane l ^ 16 (permlanex16).
template <int G>
__global__ void __launch_bounds__(256) situ_quantize_kernel(const int32_t * sums, const float * act_scales,
                                                            const float * weight_scales, const int64_t gate_offset,
                                                            const int64_t up_offset, const float gate_cap,
                                                            const float up_cap, int8_t * dst, float * scales) {
#if defined(MOSS_GFX12)
    constexpr int W  = 8;            // waves per row
    constexpr int NC = 256 * G * W;  // output values per row
    __shared__ float maxima[W];
    const int     lane        = threadIdx.x % 32;
    const int     wave        = threadIdx.x / 32;
    const int     hi          = lane / 16;
    const int     lane_offset = 4 * (lane % 16) + 128 * hi;  // element (i, j) = (0, 0) of this lane in a group
    const int64_t row         = blockIdx.x;
    const int32_t * row_sums  = sums + row * 2 * NC;
    const float     act_scale = act_scales[row];
    const auto swap16 = [](const float x) {
        return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, x), __builtin_bit_cast(int, x),
                                                                      0x76543210, 0xfedcba98, false, false));
    };

    float keep[G][2][4];
    float amax = 0.0f;
#pragma unroll 1
    for (int g = 0; g < G; ++g) {
        const int64_t base = (int64_t) (wave * G + g) * 256 + lane_offset;
        float x[2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int64_t col = base + 64 * j;
            const int4    gs  = *(const int4 *) (row_sums + gate_offset + col);
            const int4    us  = *(const int4 *) (row_sums + up_offset + col);
            const float4  gw  = *(const float4 *) (weight_scales + gate_offset + col);
            const float4  uw  = *(const float4 *) (weight_scales + up_offset + col);
            const int32_t gsv[4] = { gs.x, gs.y, gs.z, gs.w };
            const int32_t usv[4] = { us.x, us.y, us.z, us.w };
            const float   gwv[4] = { gw.x, gw.y, gw.z, gw.w };
            const float   uwv[4] = { uw.x, uw.y, uw.z, uw.w };
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float gate = (float) gsv[i] * act_scale;
                float up   = (float) usv[i] * act_scale;
                gate *= gwv[i];
                up   *= uwv[i];
                x[j][i] = situ(gate, up, gate_cap, up_cap) * (1.0f / 16.0f);
            }
        }
        // strides 1, 4 and 16
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            rotate_r4(x[j][0], x[j][1], x[j][2], x[j][3]);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                x[j][i] = rotate_r4_lanes<1>(x[j][i], lane % 4);
            }
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                x[j][i] = rotate_r4_lanes<4>(x[j][i], lane / 4 % 4);
            }
        }
        // stride 64: digits 0, 1 (a, b) are the low half-wave's j, digits 2, 3 (c, d) the high one's; output digit r
        // takes the minus sign on digit 3 - r, as in ((a + b) + c) - d for r = 0
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o0 = swap16(x[0][i]);
            const float o1 = swap16(x[1][i]);
            const float a  = hi ? o0 : x[0][i];
            const float b  = hi ? o1 : x[1][i];
            const float c  = hi ? x[0][i] : o0;
            const float d  = hi ? x[1][i] : o1;
            x[0][i] = hi ? ((a - b) + c) + d : ((a + b) + c) - d;
            x[1][i] = hi ? ((-a + b) + c) + d : ((a + b) - c) + d;
        }
        // keep the groups in order with static register indices: shift down one slot, this group last
#pragma unroll
        for (int k = 0; k + 1 < G; ++k) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    keep[k][j][i] = keep[k + 1][j][i];
                }
            }
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                keep[G - 1][j][i] = x[j][i];
                amax = fmaxf(amax, fabsf(x[j][i]));
            }
        }
    }
    amax = wave_max(amax);
    if (lane == 0) {
        maxima[wave] = amax;
    }
    __syncthreads();
#pragma unroll
    for (int w = 0; w < W; ++w) {
        amax = fmaxf(amax, maxima[w]);
    }
    const float row_scale = amax / 127.0f;
    if (threadIdx.x == 0) {
        scales[row] = row_scale;
    }
    const float inv_scale = row_scale == 0.0f ? 0.0f : 1.0f / row_scale;
    const auto  quantize  = [&](const float v) {
        const int q = row_scale == 0.0f ? 0 : __float2int_rn(v * inv_scale);
        return (uint32_t) (uint8_t) (int8_t) max(-127, min(127, q));
    };
    int8_t * dst_row = dst + row * NC;
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const int64_t base = (int64_t) (wave * G + g) * 256 + lane_offset;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            *(uint32_t *) (dst_row + base + 64 * j) = quantize(keep[g][j][0]) | (quantize(keep[g][j][1]) << 8) |
                                                       (quantize(keep[g][j][2]) << 16) | (quantize(keep[g][j][3]) << 24);
        }
    }
#else
    __builtin_trap();
#endif // defined(MOSS_GFX12)
}

// ---------------------------------------------------------------------------------------------------------------
// Flash attention forward for RDNA4 WMMA, head size 128, no mask and no GQA.
//
// A block takes 192 query rows of one head (12 waves x 16 rows) and walks the keys in tiles of 64. The scores are
// computed transposed, S^T = K Q^T, because in the gfx12 WMMA result layout every lane then holds one query with 8
// consecutive keys: the row max needs one exchange with lane ^ 16, the probabilities are already laid out as the B
// operand of O^T += V^T P^T, and rescaling the output is per lane. V is transposed into LDS for that product. Scores
// are in the log2 domain (Q is pre-scaled by scale * log2(e)), probabilities in f16, all sums in f32.

namespace fattn {

typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef float    float8_t __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));

constexpr int WAVES    = 12;
constexpr int BQ       = 16 * WAVES;  // query rows per block
constexpr int THREADS  = 32 * WAVES;
#if defined(MOSS_GFX12)
constexpr int D        = 128;
constexpr int BK       = 64;          // key rows per tile
// Shared memory holds two tiles, the one being computed and the next one: a K tile (64 key rows of 256 bytes) then a
// V^T tile (128 head-dimension rows of 128 bytes), unpadded, 64 KB in all. The 16-byte chunks of a row are permuted
// (an XOR swizzle) so that the 16 rows a half wave reads or writes at once fall in different banks: chunk q of K row r
// is stored at chunk q ^ (r % 16), chunk q of V^T row d at chunk q ^ (d % 8) ^ (d / 8 % 2).
constexpr int K_BYTES    = BK * D * 2;
constexpr int TILE_BYTES = 2 * K_BYTES;

typedef short short8_t __attribute__((ext_vector_type(8)));

// global_load_tr_b128: each group of 8 lanes reads an 8 x 8 block of 16-bit values, lane i giving the address of row
// i, and lane j of the group receives column j.
__device__ __forceinline__ u32x4_t load_transposed(const _Float16 * src) {
    return __builtin_bit_cast(u32x4_t, __builtin_amdgcn_global_load_tr_b128_v8i16((short8_t *) src));
}

// A barrier that orders LDS only: a __syncthreads fence also invalidates the vector L0 cache at every tile.
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
#endif // defined(MOSS_GFX12)

struct params {
    const float *    Q;    // [D, Lq, H]
    const _Float16 * K;    // [D, Lk, H]
    const _Float16 * V;    // [D, Lk, H]
    float *          dst;  // [D, H, Lq], contiguous
    int              Lq;
    int              Lk;
    int              H;
    int64_t          q_s1, q_s2;  // element strides of tokens and heads
    int64_t          k_s1, k_s2;  // k_s1 = v_s1 = D: the kernel reads K and V tiles as contiguous rows
    int64_t          v_s1, v_s2;
    float            scale_log2;
};

__global__ void __launch_bounds__(THREADS) __attribute__((amdgpu_waves_per_eu(6, 6))) attention_kernel(const params p) {
#if defined(MOSS_GFX12)
    __shared__ __attribute__((aligned(16))) char smem[2 * TILE_BYTES];

    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;
    const int hs   = lane / 16;  // the 8-wide half of a 16-wide WMMA k step this lane holds

    // Full query tiles first, the partial last tile of every head at the end of the grid: its waves past the last query
    // skip the compute, so those half-empty blocks fill the last round of the grid instead of adding one.
    int tile, h;
    {
        const int full   = p.Lq / BQ;
        const int b      = blockIdx.x + gridDim.x * blockIdx.y;
        const int n_full = full * p.H;
        if (b < n_full) {
            tile = b % full;
            h    = b / full % p.H;
        } else {
            tile = full;
            h    = (b - n_full) % p.H;
        }
    }
    const int  qi     = tile * BQ + wave * 16 + lane % 16;  // this lane's query
    const bool active = tile * BQ + wave * 16 < p.Lq;

    // Q^T as the B operand of S^T = K Q^T: query qi, head dimensions 16 s + 8 hs + v.
    half8_t qb[D / 16];
    {
        const float * q_row = p.Q + (int64_t) min(qi, p.Lq - 1) * p.q_s1 + (int64_t) h * p.q_s2;
#pragma unroll
        for (int s = 0; s < D / 16; ++s) {
            const float4 a = *(const float4 *) (q_row + 16 * s + 8 * hs);
            const float4 b = *(const float4 *) (q_row + 16 * s + 8 * hs + 4);
            const float  c = p.scale_log2;
            qb[s] = half8_t{ (_Float16) (a.x * c), (_Float16) (a.y * c), (_Float16) (a.z * c), (_Float16) (a.w * c),
                             (_Float16) (b.x * c), (_Float16) (b.y * c), (_Float16) (b.z * c), (_Float16) (b.w * c) };
        }
    }

    float8_t o[D / 16] = {};  // O^T: query qi, head dimensions 16 t + 8 hs + v
    // A 1.0 the compiler cannot see: (_Float16) (e * one) is one v_fma_mix* per value, a pair filling one register,
    // where (_Float16) e takes a v_cvt_f16_f32 per value and a v_pack_b32_f16 per pair. e * 1 = e: the same rounding.
    float one = 1.0f;
    asm volatile("" : "+s"(one));
    float    m = -INFINITY;   // running max of row qi, the same in lanes qi and qi ^ 16
    float    l = 0.0f;        // running sum over this lane's half of the keys

    const _Float16 * k_head = p.K + (int64_t) h * p.k_s2;
    const _Float16 * v_head = p.V + (int64_t) h * p.v_s2;

    // A tile goes global -> registers -> shared memory, the next tile's loads issued before the current tile is
    // computed. Each thread moves 3 of the 1024 16-byte K chunks of a tile (task u = thread + 384 i: key row u / 16,
    // chunk u % 16) and 3 of the 1024 V^T rows of 8 keys: task u is row j = u % 8 of the 8 x 8 block b = u / 8 (keys
    // 8 (b / 16).., head dimensions 8 (b % 16)..), read transposed, so the lane gets head dimension 8 (b % 16) + j.
    // Tasks past 1023 repeat task 1023 and store nothing. A whole tile's 64 rows of 128 values are 16 contiguous KB of
    // K and of V: a full tile is read at fixed per-thread byte offsets from the tile's (uniform) start; the last,
    // partial tile repeats row Lk - 1. The staging registers are native vectors: an array of uint4 structs gets
    // promoted to LDS and copied through it.
    constexpr int TASKS          = BK * D / 8;
    constexpr int TASKS_PER_THREAD = (TASKS + THREADS - 1) / THREADS;
    int           task[TASKS_PER_THREAD];
    uint32_t      v_src[TASKS_PER_THREAD], k_dst[TASKS_PER_THREAD], v_dst[TASKS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < TASKS_PER_THREAD; ++i) {
        const int u = min((int) threadIdx.x + i * THREADS, TASKS - 1);
        const int b = u / 8, j = u % 8, d = 8 * (b % 16) + j;
        task[i]     = u;
        v_src[i]    = (uint32_t) (((8 * (b / 16) + j) * D + 8 * (b % 16)) * 2);
        k_dst[i]    = (uint32_t) (u / 16 * 256 + 16 * ((u % 16) ^ (u / 16 % 16)));
        v_dst[i]    = (uint32_t) (K_BYTES + d * 128 + 16 * ((b / 16) ^ (d % 8) ^ (d / 8 % 2)));
    }
    u32x4_t k_next[TASKS_PER_THREAD], v_next[TASKS_PER_THREAD];
    const auto load_tile = [&](const int kv0) {
        if (kv0 + BK <= p.Lk) {
            const char * k_tile = (const char *) (k_head + (int64_t) kv0 * D);
            const char * v_tile = (const char *) (v_head + (int64_t) kv0 * D);
#pragma unroll
            for (int i = 0; i < TASKS_PER_THREAD; ++i) {
                k_next[i] = *(const u32x4_t *) (k_tile + 16 * (uint32_t) task[i]);
                v_next[i] = load_transposed((const _Float16 *) (v_tile + v_src[i]));
            }
            return;
        }
#pragma unroll
        for (int i = 0; i < TASKS_PER_THREAD; ++i) {
            const int u  = task[i];
            const int kk = min(kv0 + u / 16, p.Lk - 1);
            const int kv = min(kv0 + 8 * (u / 128) + u % 8, p.Lk - 1);
            k_next[i]    = *(const u32x4_t *) (k_head + (int64_t) kk * D + 8 * (u % 16));
            v_next[i]    = load_transposed(v_head + (int64_t) kv * D + 8 * (u / 8 % 16));
        }
    };

    // One barrier per tile: tile n is stored to buffer n % 2, whose previous tile (n - 2) every wave finished before
    // the barrier of tile n - 1. The next tile's loads go out right after the stores, before the barrier.
    load_tile(0);
    for (int kv0 = 0, buf = 0; kv0 < p.Lk; kv0 += BK, buf ^= 1) {
        char * const tile = smem + buf * TILE_BYTES;
#pragma unroll
        for (int i = 0; i < TASKS_PER_THREAD; ++i) {
            if (THREADS * (i + 1) <= TASKS || (int) threadIdx.x + i * THREADS < TASKS) {
                *(u32x4_t *) (tile + k_dst[i]) = k_next[i];
                *(u32x4_t *) (tile + v_dst[i]) = v_next[i];
            }
        }
        if (kv0 + BK < p.Lk) {
            load_tile(kv0 + BK);
        }
        lds_barrier();

        if (!active) {
            continue;
        }

        // S^T in 4 chunks of 16 keys: this lane gets keys kv0 + 16 c + 8 hs + v of query qi. The WMMAs go s-major, so
        // consecutive ones write different chunks (each still sums s = 0..7 in order), and every step's K fragments
        // are read one step ahead; left to itself the compiler waits for each read right before its WMMA.
        float8_t st[BK / 16] = {};
        {
            // row 16 c + r (r = lane % 16), chunk 2 s + hs, at byte 4096 c + (k_lane ^ 32 s)
            const int      r      = lane % 16;
            const uint32_t k_lane = 256 * r + 32 * (r / 2) + 16 * (hs ^ (r % 2));
            half8_t        kf[2][BK / 16];
#pragma unroll
            for (int c = 0; c < BK / 16; ++c) {
                kf[0][c] = *(const half8_t *) (tile + 4096 * c + k_lane);
            }
#pragma unroll
            for (int s = 0; s < D / 16; ++s) {
                if (s + 1 < D / 16) {
#pragma unroll
                    for (int c = 0; c < BK / 16; ++c) {
                        kf[(s + 1) % 2][c] = *(const half8_t *) (tile + 4096 * c + (k_lane ^ (32 * (s + 1))));
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int c = 0; c < BK / 16; ++c) {
                    st[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kf[s % 2][c], qb[s], st[c]);
                }
                __builtin_amdgcn_sched_barrier(0);
            }
        }
        if (kv0 + BK > p.Lk) {
#pragma unroll
            for (int c = 0; c < BK / 16; ++c) {
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    if (kv0 + 16 * c + 8 * hs + v >= p.Lk) {
                        st[c][v] = -INFINITY;
                    }
                }
            }
        }

        float mx = m;
#pragma unroll
        for (int c = 0; c < BK / 16; ++c) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                mx = fmaxf(mx, st[c][v]);
            }
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float alpha = __builtin_amdgcn_exp2f(m - mx);
        m = mx;

        half8_t pb[BK / 16];  // P^T as the B operand of O^T += V^T P^T
        float   sum = 0.0f;
#pragma unroll
        for (int c = 0; c < BK / 16; ++c) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float e = __builtin_amdgcn_exp2f(st[c][v] - mx);
                sum += e;
                pb[c][v] = (_Float16) (e * one);
            }
        }
        l = l * alpha + sum;
        if (__builtin_amdgcn_ballot_w32(alpha != 1.0f)) {  // once the maxima settle alpha is 1: x * 1 = x
#pragma unroll
            for (int t = 0; t < D / 16; ++t) {
                o[t] *= alpha;
            }
        }
        // O^T += V^T P^T in 8 groups of 4 WMMAs: key chunk c = g / 2, head-dimension tiles t = 4 (g % 2) + i. Each
        // o[t] still sums c = 0..3 in order; every group's V^T fragments are read one group ahead.
        {
            // row 16 t + r, chunk 2 c + hs, at byte K_BYTES + 2048 t + (v_lane ^ 32 c)
            const int      r      = lane % 16;
            const uint32_t v_lane = 128 * r + 16 * (hs ^ (r % 8) ^ (r / 8));
            half8_t        vf[2][4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                vf[0][i] = *(const half8_t *) (tile + K_BYTES + 2048 * i + v_lane);
            }
#pragma unroll
            for (int g = 0; g < 2 * (BK / 16); ++g) {
                if (g + 1 < 2 * (BK / 16)) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int t = 4 * ((g + 1) % 2) + i;
                        vf[(g + 1) % 2][i] =
                            *(const half8_t *) (tile + K_BYTES + 2048 * t + (v_lane ^ (32 * ((g + 1) / 2))));
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int t = 4 * (g % 2) + i;
                    o[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vf[g % 2][i], pb[g / 2], o[t]);
                }
                __builtin_amdgcn_sched_barrier(0);
            }
        }
    }

    l += __shfl_xor(l, 16, 32);
    if (qi < p.Lq) {
        const float inv = 1.0f / l;
        float *     out = p.dst + ((int64_t) qi * p.H + h) * D;
#pragma unroll
        for (int t = 0; t < D / 16; ++t) {
            const float8_t r = o[t] * inv;
            *(float4 *) (out + 16 * t + 8 * hs)     = make_float4(r[0], r[1], r[2], r[3]);
            *(float4 *) (out + 16 * t + 8 * hs + 4) = make_float4(r[4], r[5], r[6], r[7]);
        }
    }
#else
    __builtin_trap();
#endif // defined(MOSS_GFX12)
}

} // namespace fattn

// ---------------------------------------------------------------------------------------------------------------
// Row kernels around the bf16 linears (text blocks, timestep path, output head). Each repeats the ggml kernels it
// replaces; where one kernel stands for several, contraction is off and the FMAs ggml makes are written out.

__global__ void f32_to_bf16_kernel(const float * x, const int64_t n, uint16_t * dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = to_bf16(x[i]);
    }
}

__global__ void add_bias_kernel(const float * mm, const float * bias, const int64_t n, const int64_t total, float * dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        dst[i] = mm[i] + bias[i % n];
    }
}

__global__ void add_bias_residual_kernel(const float * mm, const float * bias, const int64_t n, const int64_t total,
                                         float * x) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        x[i] = x[i] + (mm[i] + bias[i % n]);
    }
}

// ggml's rms_norm_f32<1024> and the weight: thread t adds the squares of columns t, t + 1024, ..., then the block sum;
// scale * x * w, rounded to bf16 for the linear after it.
__global__ void __launch_bounds__(1024) rms_norm_mul_bf16_kernel(const float * x, const float * w, const float eps,
                                                                 const int64_t dim, uint16_t * dst) {
    __shared__ float partial[32];
    const float * x_row   = x + (int64_t) blockIdx.x * dim;
    uint16_t *    dst_row = dst + (int64_t) blockIdx.x * dim;
    float sumsq = 0.0f;
    for (int64_t col = threadIdx.x; col < dim; col += 1024) {
        const float v = x_row[col];
        sumsq += v * v;
    }
    sumsq = block_sum<1024>(sumsq, partial);
    const float scale = rsqrtf(sumsq / dim + eps);
    for (int64_t col = threadIdx.x; col < dim; col += 1024) {
        dst_row[col] = to_bf16(scale * x_row[col] * w[col]);
    }
}

// The attention inputs from a text block's qkv linear, one wave per (head, token): q and k get the bias, the head norm
// and RoPE, v the bias and the f16 cast.
__global__ void __launch_bounds__(256) text_qkv_prologue_kernel(const float * raw, const float * bias, const float * q_w,
                                                                const float * k_w, const float eps, const float * pe,
                                                                const int64_t rows, const int heads, float * q,
                                                                _Float16 * k, _Float16 * v) {
    const int64_t wave_index = (int64_t) blockIdx.x * 8 + threadIdx.x / 32;
    if (wave_index >= heads * rows) {
        return;
    }
    const int     lane   = threadIdx.x % 32;
    const int64_t head   = wave_index % heads;
    const int64_t token  = wave_index / heads;
    const int64_t row    = head * rows + token;  // output row
    const int64_t dim    = (int64_t) heads * 128;
    const float * s      = raw + token * 3 * dim;
    const int64_t q0     = head * 128;
    const int64_t k0     = dim + q0;
    const int64_t v0     = 2 * dim + q0;
    const float * pe_row = pe + token * 128;
    const auto load_q = [&](const int i) { return s[q0 + i] + bias[q0 + i]; };
    const auto load_k = [&](const int i) { return s[k0 + i] + bias[k0 + i]; };
    mul_rope_row(load_q, q_w, pe_row, rms_scale_block256(load_q, eps), q + row * 128);
    mul_rope_row(load_k, k_w, pe_row, rms_scale_block256(load_k, eps), k + row * 128);
    for (int i = lane; i < 128; i += 32) {
        v[row * 128 + i] = (_Float16) (s[v0 + i] + bias[v0 + i]);
    }
}

// The text FFN gate: the up and gate biases (ggml's separate adds), situ with the caps as arguments (ggml's GLU
// kernel), rounded to bf16 for the down linear.
__global__ void text_situ_bf16_kernel(const float * mm, const float * bias_up, const float * bias_gate, const int64_t nc,
                                      const int64_t total, const float gate_cap, const float up_cap, uint16_t * dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const int64_t row  = i / nc;
    const int64_t col  = i - row * nc;
    const float * m    = mm + row * 2 * nc;
    const float   up   = m[col] + bias_up[col];
    const float   gate = m[nc + col] + bias_gate[col];
    dst[i] = to_bf16(situ(gate, up, gate_cap, up_cap));
}

// ggml's mul_mat_vec_f on a bf16 matrix and one f32 column (ncols_dst = 1), for up to 16 columns (rows of x) at once so
// the matrix is read once: thread t of output n's block takes the column pairs t, t + block_size, ... with one FMA per
// value, then the wave sums and, past one wave, the sum of the block's wave sums (unused slots 0). The bias add and
// the SiLU are ggml's separate ops.
constexpr int kMaxLinearRows = 16;

template <int block_size>
__global__ void __launch_bounds__(block_size) linear_bf16_rows_kernel(const uint16_t * w, const float * bias,
                                                                       const float * x, const int rows, const int64_t k,
                                                                       const bool silu, float * dst, const int64_t n) {
    __shared__ float partial[kMaxLinearRows][32];
    const int64_t    col = blockIdx.x;
    const int        tid = threadIdx.x;
    const uint32_t * w2  = (const uint32_t *) (w + col * k);
    if constexpr (block_size > 32) {
        if (tid < 32) {
#pragma unroll
            for (int j = 0; j < kMaxLinearRows; ++j) {
                partial[j][tid] = 0.0f;
            }
        }
        __syncthreads();
    }
    float sum[kMaxLinearRows] = {};
    for (int64_t c2 = tid; c2 < k / 2; c2 += block_size) {
        const uint32_t pair = w2[c2];
        const float    w0   = bf16_to_f32((uint16_t) pair);
        const float    w1   = bf16_to_f32((uint16_t) (pair >> 16));
#pragma unroll
        for (int j = 0; j < kMaxLinearRows; ++j) {
            if (j < rows) {
                const float2 xv = *(const float2 *) (x + j * k + 2 * c2);
                sum[j] += w0 * xv.x;
                sum[j] += w1 * xv.y;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < kMaxLinearRows; ++j) {
        sum[j] = wave_sum(sum[j]);
        if constexpr (block_size > 32) {
            if (tid % 32 == 0) {
                partial[j][tid / 32] = sum[j];
            }
        }
    }
    if constexpr (block_size > 32) {
        __syncthreads();
        if (tid >= 32) {
            return;
        }
#pragma unroll
        for (int j = 0; j < kMaxLinearRows; ++j) {
            sum[j] = wave_sum(partial[j][tid]);
        }
    }
    if (tid != 0) {
        return;
    }
#pragma unroll
    for (int j = 0; j < kMaxLinearRows; ++j) {
        if (j < rows) {
            float value = sum[j] + bias[col];
            if (silu) {
                value = value / (1.0f + expf(-value));
            }
            dst[j * n + col] = value;
        }
    }
}

// The DiT input rows of each sequence: the video embedder's image rows (+ its bias), then the sequence's text rows.
__global__ void assemble_tokens_kernel(const float * img, const float * img_bias, const int64_t n_img,
                                       const float * text_c, const int64_t n_text_c, const float * text_u,
                                       const int64_t dim, const int64_t total, float * x) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const int64_t row    = i / dim;
    const int64_t col    = i - row * dim;
    const int64_t rows_c = n_img + n_text_c;
    const bool    uncond = row >= rows_c;
    const int64_t r      = uncond ? row - rows_c : row;
    const float * text   = uncond ? text_u : text_c;
    x[i] = r < n_img ? img[r * dim + col] + img_bias[col] : text[(r - n_img) * dim + col];
}

// The output head's input row: final_norm (ggml's rms_norm_f32<1024>, then the weight), the head modulation as ggml's
// fused mul_add (x + x * scale, one FMA), the shift add, and the bf16 rounding of the linear after it.
__global__ void __launch_bounds__(1024) final_head_kernel(const float * x, const int64_t dim, const int64_t n_img,
                                                          const int64_t uncond_first, const float * norm_w,
                                                          const float * shift, const float * scale, const float eps,
                                                          uint16_t * dst) {
#pragma clang fp contract(off)
    __shared__ float partial[32];
    const int64_t r       = blockIdx.x;
    const float * x_row   = x + (r < n_img ? r : uncond_first + r - n_img) * dim;
    uint16_t *    dst_row = dst + r * dim;
    float sumsq = 0.0f;
    for (int64_t col = threadIdx.x; col < dim; col += 1024) {
        const float v = x_row[col];
        sumsq = __builtin_fmaf(v, v, sumsq);
    }
    sumsq = block_sum<1024>(sumsq, partial);
    const float rs = rsqrtf(sumsq / dim + eps);
    for (int64_t col = threadIdx.x; col < dim; col += 1024) {
        const float xn = rs * x_row[col] * norm_w[col];
        dst_row[col] = to_bf16(__builtin_fmaf(xn, scale[col], xn) + shift[col]);
    }
}

} // namespace

bool device_supported(const int device) {
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device) != hipSuccess) {
        return false;
    }
    return std::strncmp(prop.gcnArchName, "gfx1200", 7) == 0 || std::strncmp(prop.gcnArchName, "gfx1201", 7) == 0;
}

void f32_to_bf16(const float * x, const int64_t n, uint16_t * dst, hipStream_t stream) {
    f32_to_bf16_kernel<<<elementwise_blocks(n), kBlock, 0, stream>>>(x, n, dst);
}

void add_bias(const float * mm, const float * bias, const int64_t rows, const int64_t n, float * dst, hipStream_t stream) {
    add_bias_kernel<<<elementwise_blocks(rows * n), kBlock, 0, stream>>>(mm, bias, n, rows * n, dst);
}

void add_bias_residual(const float * mm, const float * bias, const int64_t rows, const int64_t n, float * x,
                       hipStream_t stream) {
    add_bias_residual_kernel<<<elementwise_blocks(rows * n), kBlock, 0, stream>>>(mm, bias, n, rows * n, x);
}

void linear_bf16_rows(const uint16_t * w, const float * bias, const float * x, const int rows, const int64_t n,
                      const int64_t k, const bool silu, float * dst, hipStream_t stream) {
    // ggml's block size: the fewest loop trips over 32, 64, ..., 256 threads, the smaller block on a tie
    int     block = 32;
    int64_t trips = (k + 63) / 64;
    for (int b = 64; b <= 256; b += 32) {
        const int64_t t = (k + 2 * b - 1) / (2 * b);
        if (t < trips) {
            trips = t;
            block = b;
        }
    }
    const auto launch = [&](auto kernel) {
        kernel<<<(unsigned) n, block, 0, stream>>>(w, bias, x, rows, k, silu, dst, n);
    };
    switch (block) {
        case 32:  launch(linear_bf16_rows_kernel<32>); break;
        case 64:  launch(linear_bf16_rows_kernel<64>); break;
        case 96:  launch(linear_bf16_rows_kernel<96>); break;
        case 128: launch(linear_bf16_rows_kernel<128>); break;
        case 160: launch(linear_bf16_rows_kernel<160>); break;
        case 192: launch(linear_bf16_rows_kernel<192>); break;
        case 224: launch(linear_bf16_rows_kernel<224>); break;
        default:  launch(linear_bf16_rows_kernel<256>); break;
    }
}

void rms_norm_mul_bf16(const float * x, const float * w, const float eps, const int64_t rows, const int64_t dim,
                       uint16_t * dst, hipStream_t stream) {
    rms_norm_mul_bf16_kernel<<<(unsigned) rows, 1024, 0, stream>>>(x, w, eps, dim, dst);
}

void text_qkv_prologue(const float * raw, const float * bias, const float * q_w, const float * k_w, const float eps,
                       const float * pe, const int64_t rows, const int heads, float * q, uint16_t * k, uint16_t * v,
                       hipStream_t stream) {
    text_qkv_prologue_kernel<<<(unsigned) ((heads * rows + 7) / 8), 256, 0, stream>>>(
        raw, bias, q_w, k_w, eps, pe, rows, heads, q, (_Float16 *) k, (_Float16 *) v);
}

void text_situ_bf16(const float * mm, const float * bias_up, const float * bias_gate, const int64_t rows,
                    const int64_t nc, uint16_t * dst, hipStream_t stream) {
    text_situ_bf16_kernel<<<elementwise_blocks(rows * nc), kBlock, 0, stream>>>(mm, bias_up, bias_gate, nc, rows * nc,
                                                                                4.0f, 25.0f, dst);
}

void assemble_tokens(const float * img, const float * img_bias, const int64_t n_img, const float * text_c,
                     const int64_t n_text_c, const float * text_u, const int64_t n_text_u, const int64_t dim, float * x,
                     hipStream_t stream) {
    const int64_t total = (n_img + n_text_c + (text_u != nullptr ? n_img + n_text_u : 0)) * dim;
    assemble_tokens_kernel<<<elementwise_blocks(total), kBlock, 0, stream>>>(img, img_bias, n_img, text_c, n_text_c,
                                                                             text_u, dim, total, x);
}

void norm_quantize_i8(const float * x, const int64_t rows, const int64_t k, const float * norm_w, const float * scale,
                      const float * shift, const float eps, int8_t * xq, float * xs, hipStream_t stream) {
    launch_convrot<true>(x, rows, k, norm_w, scale, shift, eps, xq, xs, stream);
}

void quantize_i8(const float * x, const int64_t rows, const int64_t k, int8_t * xq, float * xs, hipStream_t stream) {
    launch_convrot<false>(x, rows, k, nullptr, nullptr, nullptr, 0.0f, xq, xs, stream);
}

size_t gemm_i8_split_bytes() {
    return size_t(gemm_i8::SPLIT_SUMS) * sizeof(int32_t) + gemm_i8::SPLIT_TILES * sizeof(unsigned);
}

void gemm_i8_qkv_attention(const int8_t * w, const float * w_scales, const int8_t * xq, const float * xs,
                           const int64_t rows, const int64_t k, const int heads, const float * q_w, const float * k_w,
                           const float eps, const float * pe, float * q, uint16_t * k_out, uint16_t * v,
                           int32_t * split, hipStream_t stream) {
    const int64_t n  = 3 * heads * 128;
    int32_t *     ws = split != nullptr && gemm_i8::split_tail(n, rows, k) ? split : nullptr;
    gemm_i8_qkv_attention_kernel<<<(unsigned) gemm_i8::blocks(n, rows, ws != nullptr), gemm_i8::NT, 0, stream>>>(
        w, xq, (int) rows, (int) k, ws, xs, w_scales, q_w, k_w, eps, pe, heads, q, (_Float16 *) k_out, (_Float16 *) v);
}

void attention_d128(const float * q, const uint16_t * k, const uint16_t * v, const int64_t rows, const int64_t first,
                    const int64_t n, const int heads, const float scale, float * dst, hipStream_t stream) {
    const fattn::params p = {
        q + first * 128,
        (const _Float16 *) k + first * 128,
        (const _Float16 *) v + first * 128,
        dst + first * heads * 128,
        (int) n,
        (int) n,
        heads,
        128, rows * 128,
        128, rows * 128,
        128, rows * 128,
        scale * 1.4426950408889634f,
    };
    const dim3 grid((unsigned) ((n + fattn::BQ - 1) / fattn::BQ), (unsigned) heads);
    fattn::attention_kernel<<<grid, fattn::THREADS, 0, stream>>>(p);
}

void gemm_i8_gated_residual(const int8_t * w, const float * w_scales, const int8_t * xq, const float * xs,
                            const int64_t rows, const int64_t k, const int64_t n, const float * gate, float * x,
                            int32_t * split, hipStream_t stream) {
    const int stagger = k >= 8192 ? 5 : 0;  // k blocks staggered by 5 per block: only the long k loop gains
    int32_t * ws      = split != nullptr && gemm_i8::split_tail(n, rows, k) ? split : nullptr;
    gemm_i8_gated_residual_kernel<<<(unsigned) gemm_i8::blocks(n, rows, ws != nullptr), gemm_i8::NT, 0, stream>>>(
        w, xq, (int) n, (int) rows, (int) k, stagger, ws, xs, w_scales, gate, x);
}

void gemm_i8_i32(const int8_t * w, const int8_t * xq, const int64_t rows, const int64_t k, const int64_t n,
                 int32_t * sums, hipStream_t stream) {
    gemm_i8_i32_kernel<<<(unsigned) gemm_i8::blocks(n, rows), gemm_i8::NT, 0, stream>>>(w, xq, (int) n, (int) rows,
                                                                                       (int) k, sums);
}

void situ_quantize_i8(const int32_t * sums, const float * xs, const float * w_scales, const int64_t rows,
                      const int64_t nc, int8_t * dst, float * dst_scales, hipStream_t stream) {
    // MOSS-Image's situ caps; the gate is the second half of the fused up/gate linear
    const auto kernel = nc == 4096 ? situ_quantize_kernel<2> : situ_quantize_kernel<4>;
    kernel<<<(unsigned) rows, 256, 0, stream>>>(sums, xs, w_scales, nc, 0, 4.0f, 25.0f, dst, dst_scales);
}

void final_head_bf16(const float * x, const int64_t dim, const int64_t n_img, const int n_seq, const int64_t uncond_first,
                     const float * norm_w, const float * head_mod, const float eps, uint16_t * dst,
                     hipStream_t stream) {
    final_head_kernel<<<(unsigned) (n_seq * n_img), 1024, 0, stream>>>(x, dim, n_img, uncond_first, norm_w, head_mod,
                                                                       head_mod + dim, eps, dst);
}

} // namespace luce::moss_image
