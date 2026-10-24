// MOSS-Image VAE decoder kernels for RDNA4.
//
// The convolution and the GroupNorm are the kernels of the measured stable-diffusion.cpp build without ggml's plumbing
// and its batch dimension (one image); their arithmetic is unchanged. The other kernels repeat the float operations of
// the ggml ops and host steps they replace in the same order, with contraction off where those were separate
// operations, so the results are the same.

#include "kernels.h"

#include "device_common.cuh"

#include <initializer_list>
#include <type_traits>

namespace luce::moss_image {
namespace {

// ---------------------------------------------------------------------------------------------------------------
// 2D convolution (stride 1, dilation 1) as an implicit GEMM on RDNA4 WMMA:
// y[oc][p] = sum over taps t and input channels ci of w[oc][t][ci] * x[p + t][ci], f16 operands, f32 sums.
// The input is f16 channels-last and the weights [OC][KH * KW][IC], so both WMMA operands are rows of 8 consecutive
// channels (one 16-byte load per lane). A block computes 128 output channels x 128 output pixels with 8 waves of
// 32 x 64; the K loop takes 32 channels of one tap per step, double-buffered in LDS with one barrier per step. Pixels
// outside the input read as zero (the padding). The epilogue adds the bias, then the residual, and writes f32
// [OC][OH * OW]. With UP the input is a nearest 2x upscale read from its half-size source.

namespace conv {

constexpr int BM      = 128;  // output channels per block
constexpr int BN      = 128;  // output pixels per block
constexpr int THREADS = 256;
constexpr int SM_BM   = 16;   // output channels of the small-OC kernel (one WMMA tile)

struct params {
    const _Float16 * x;     // [IH * IW][IC], with UP [IH / 2 * IW / 2][IC]
    const _Float16 * w;     // [OC][KH * KW][IC]
    const float *    bias;  // [OC] or null
    const float *    res;   // [OC][OH * OW] or null
    float *          y;     // [OC][OH * OW]
    int              IW, IH, OW, OH, KW, KH, PX, PY, IC, OC;
};

#if defined(MOSS_GFX12)
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef float    float8_t __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));  // not uint4: an array of those structs lands in scratch

constexpr int BK     = 32;      // input channels per K step
constexpr int STRIDE = BK + 8;  // halves per LDS row, padded against bank conflicts

// A barrier that orders LDS only: __syncthreads also invalidates the vector L0 cache at every K step.
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// The K step a block loads next: tap (ky, kx) = tap index, channels 32 cs .., advanced incrementally (uniform).
struct kstep {
    int cs = 0, tap = 0, ky = 0, kx = 0;

    __device__ __forceinline__ void advance(const params & p, const int ci_steps) {
        if (++cs == ci_steps) {
            cs = 0;
            ++tap;
            if (++kx == p.KW) {
                kx = 0;
                ++ky;
            }
        }
    }
};

// One 16-byte chunk of B: channels lc .. lc + 7 of the step's 32 at output pixel (oy, ox), zero outside the input.
// The per-pixel part of the address is b_off (bytes, without UP), the step's part a uniform pointer.
template <bool UP>
__device__ __forceinline__ u32x4_t load_pixel(const params & p, const kstep & k, const bool pv, const int oy,
                                              const int ox, const uint32_t b_off, const int lc) {
    const int  iy = oy + k.ky - p.PY;
    const int  ix = ox + k.kx - p.PX;
    const bool in = pv && (unsigned) iy < (unsigned) p.IH && (unsigned) ix < (unsigned) p.IW;
    if (!in) {
        return u32x4_t{ 0, 0, 0, 0 };
    }
    if constexpr (UP) {
        // the nearest 2x upscale of a source of (IW / 2) x (IH / 2) pixels; every factor is below 2^24
        const uint32_t off = __umul24((uint32_t) iy >> 1, (uint32_t) (p.IW / 2 * p.IC) * 2) +
                             __umul24((uint32_t) ix >> 1, (uint32_t) p.IC * 2) + 2 * lc;
        return *(const u32x4_t *) ((const char *) (p.x + k.cs * BK) + off);
    } else {
        const _Float16 * xs = p.x + ((int64_t) (k.ky - p.PY) * p.IW + (k.kx - p.PX)) * p.IC + k.cs * BK;
        return *(const u32x4_t *) ((const char *) xs + b_off);
    }
}
#endif // defined(MOSS_GFX12)

template <bool UP>
__global__ void __launch_bounds__(THREADS) conv2d_kernel(const params p) {
#if defined(MOSS_GFX12)
    __shared__ _Float16 sA[2][BM * STRIDE];
    __shared__ _Float16 sB[2][BN * STRIDE];

    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;
    const int wm   = 32 * (wave / 2);  // first output channel of this wave in the block
    const int wn   = 64 * (wave % 2);  // first pixel of this wave in the block
    const int hs   = lane / 16;        // the 8-wide half of a 16-wide WMMA k step this lane holds

    const int OHW = p.OH * p.OW;
    const int p0  = blockIdx.x * BN;
    const int oc0 = blockIdx.y * BM;

    // Every thread moves two 16-byte chunks of A and of B per step: rows lr and lr + 64, channels lc .. lc + 7.
    const int lr = threadIdx.x / 4;
    const int lc = 8 * (threadIdx.x % 4);
    const int     ci_steps = p.IC / BK;
    const int     steps    = p.KH * p.KW * ci_steps;
    const int64_t w_row    = (int64_t) p.KH * p.KW * p.IC;
    // The per-thread parts of the load addresses, in bytes. They fit in 32 bits: the largest f16 map, 1024 x 1024
    // pixels of 256 channels, is 512 MiB. A row past OC loads row OC - 1: its sums are not stored.
    int           oy[2], ox[2];
    bool          pv[2];
    uint32_t      a_off[2], b_off[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int pix = p0 + lr + 64 * r;
        pv[r]         = pix < OHW;
        oy[r]         = pix / p.OW;
        ox[r]         = pix % p.OW;
        a_off[r]      = (uint32_t) (((int64_t) min(oc0 + lr + 64 * r, p.OC - 1) * w_row + lc) * 2);
        b_off[r]      = (uint32_t) ((((int64_t) oy[r] * p.IW + ox[r]) * p.IC + lc) * 2);
    }

    // Two register sets: a step's loads are issued two steps before its stores to shared memory, which wait for them
    // (the input comes from DRAM, slower than one step of WMMAs).
    u32x4_t ra[2][2], rb[2][2];
    kstep   next;
    auto    load = [&](const int set) {
        const char * w_step = (const char *) (p.w + next.tap * p.IC + next.cs * BK);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            ra[set][r] = *(const u32x4_t *) (w_step + a_off[r]);
            rb[set][r] = load_pixel<UP>(p, next, pv[r], oy[r], ox[r], b_off[r], lc);
        }
    };
    auto store = [&](const int set, const int buf) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            *(u32x4_t *) (sA[buf] + (lr + 64 * r) * STRIDE + lc) = ra[set][r];
            *(u32x4_t *) (sB[buf] + (lr + 64 * r) * STRIDE + lc) = rb[set][r];
        }
    };

    float8_t acc[2][4] = {};
    load(0);
    if (1 < steps) {
        next.advance(p, ci_steps);
    }
    load(1);
    store(0, 0);
    lds_barrier();
    // Step t: load step t + 2 into set t % 2 (stored at step t - 1), compute step t, store step t + 1 (set (t + 1) % 2).
    // The stores are unconditional (past the last step they write a reloaded last step to the unused buffer): behind
    // a branch the WMMAs get sunk below them, and they wait for the loads. Two steps per iteration keep the register
    // sets and buffers static.
    const auto step = [&](const int t, const auto buf_c) {
        constexpr int buf = decltype(buf_c)::value;
        if (t + 2 < steps) {
            next.advance(p, ci_steps);
        }
        load(buf);
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            half8_t a[2], b[4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                a[i] = *(const half8_t *) (sA[buf] + (wm + 16 * i + lane % 16) * STRIDE + ks + 8 * hs);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                b[j] = *(const half8_t *) (sB[buf] + (wn + 16 * j + lane % 16) * STRIDE + ks + 8 * hs);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], b[j], acc[i][j]);
                }
            }
        }
        // The WMMAs stay above the stores, which wait for their loads: left to itself the compiler moves the stores
        // above the WMMAs, or the WMMAs below the next step's branches.
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            asm volatile("" : : "v"(acc[i][0]), "v"(acc[i][1]), "v"(acc[i][2]), "v"(acc[i][3]));
        }
        __builtin_amdgcn_sched_barrier(0);
        store(buf ^ 1, buf ^ 1);
        lds_barrier();
    };
    int t = 0;
    for (; t + 1 < steps; t += 2) {
        step(t, std::integral_constant<int, 0>());
        step(t + 1, std::integral_constant<int, 1>());
    }
    if (t < steps) {
        step(t, std::integral_constant<int, 0>());
    }

    // Result layout: lane holds pixel lane % 16 of a 16-pixel tile and output channels 8 hs .. 8 hs + 7 of a 16-channel tile.
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int pix = p0 + wn + 16 * j + lane % 16;
        if (pix >= OHW) {
            continue;
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int oc = oc0 + wm + 16 * i + 8 * hs + v;
                if (oc < p.OC) {
                    float r = acc[i][j][v] + (p.bias != nullptr ? p.bias[oc] : 0.0f);
                    if (p.res != nullptr) {
                        r = r + p.res[(int64_t) oc * OHW + pix];
                    }
                    p.y[(int64_t) oc * OHW + pix] = r;
                }
            }
        }
    }
#else
    __builtin_trap();
#endif // defined(MOSS_GFX12)
}

// Convolutions with at most 16 output channels (conv_out has 3): one 16-row tile of output channels instead of 128,
// NT / 32 waves of 16 x (BP * 32 / NT) pixels. The K order is conv2d_kernel's (taps, 32 input channels per step, two
// 16-wide WMMA steps), so every output is summed the same way.
template <int NT, int BP>
__global__ void __launch_bounds__(NT) small_oc_kernel(const params p) {
#if defined(MOSS_GFX12)
    constexpr int WP = BP / (NT / 32);  // pixels per wave
    constexpr int JT = WP / 16;         // 16-pixel tiles per wave
    constexpr int RB = BP * 4 / NT;     // 16-byte B chunks per thread and step
    __shared__ _Float16 sA[2][SM_BM * STRIDE];
    __shared__ _Float16 sB[2][BP * STRIDE];

    const int lane = threadIdx.x % 32;
    const int wn   = WP * (threadIdx.x / 32);
    const int hs   = lane / 16;

    const int OHW = p.OH * p.OW;
    const int p0  = blockIdx.x * BP;

    // B: rows lr + (NT / 4) r; A: row lr of threads 0 .. 63.
    const int lr = threadIdx.x / 4;
    const int lc = 8 * (threadIdx.x % 4);
    const int     ci_steps = p.IC / BK;
    const int     steps    = p.KH * p.KW * ci_steps;
    const int64_t w_row    = (int64_t) p.KH * p.KW * p.IC;
    int           oy[RB], ox[RB];
    bool          pv[RB];
    uint32_t      b_off[RB];
#pragma unroll
    for (int r = 0; r < RB; ++r) {
        const int pix = p0 + lr + (NT / 4) * r;
        pv[r]         = pix < OHW;
        oy[r]         = pix / p.OW;
        ox[r]         = pix % p.OW;
        b_off[r]      = (uint32_t) ((((int64_t) oy[r] * p.IW + ox[r]) * p.IC + lc) * 2);
    }
    // rows past OC (and the threads past the 16 A rows) load row OC - 1: those sums are not stored
    const uint32_t a_off = (uint32_t) (((int64_t) min(lr, p.OC - 1) * w_row + lc) * 2);

    u32x4_t ra, rb[RB];
    kstep   next;
    auto    load = [&]() {
        ra = *(const u32x4_t *) ((const char *) (p.w + next.tap * p.IC + next.cs * BK) + a_off);
#pragma unroll
        for (int r = 0; r < RB; ++r) {
            rb[r] = load_pixel<false>(p, next, pv[r], oy[r], ox[r], b_off[r], lc);
        }
    };
    auto store = [&](const int buf) {
        if (lr < SM_BM) {
            *(u32x4_t *) (sA[buf] + lr * STRIDE + lc) = ra;
        }
#pragma unroll
        for (int r = 0; r < RB; ++r) {
            *(u32x4_t *) (sB[buf] + (lr + (NT / 4) * r) * STRIDE + lc) = rb[r];
        }
    };

    float8_t acc[JT] = {};
    load();
    store(0);
    lds_barrier();
    for (int t = 0; t < steps; ++t) {
        const int buf = t & 1;
        if (t + 1 < steps) {
            next.advance(p, ci_steps);
        }
        load();
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            const half8_t a = *(const half8_t *) (sA[buf] + (lane % 16) * STRIDE + ks + 8 * hs);
            half8_t       b[JT];
#pragma unroll
            for (int j = 0; j < JT; ++j) {
                b[j] = *(const half8_t *) (sB[buf] + (wn + 16 * j + lane % 16) * STRIDE + ks + 8 * hs);
            }
#pragma unroll
            for (int j = 0; j < JT; ++j) {
                acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b[j], acc[j]);
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        store(buf ^ 1);
        lds_barrier();
    }

#pragma unroll
    for (int j = 0; j < JT; ++j) {
        const int pix = p0 + wn + 16 * j + lane % 16;
        if (pix >= OHW) {
            continue;
        }
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int oc = 8 * hs + v;
            if (oc < p.OC) {
                float r = acc[j][v] + (p.bias != nullptr ? p.bias[oc] : 0.0f);
                if (p.res != nullptr) {
                    r = r + p.res[(int64_t) oc * OHW + pix];
                }
                p.y[(int64_t) oc * OHW + pix] = r;
            }
        }
    }
#else
    __builtin_trap();
#endif // defined(MOSS_GFX12)
}

} // namespace conv

// [C][HW] f32 -> [HW][C] f16 through a 64 x 64 tile in LDS.
__global__ void to_nhwc_kernel(const float * __restrict__ x, _Float16 * __restrict__ y, const int C, const int HW) {
    __shared__ float tile[64][65];
    const int p0 = blockIdx.x * 64;
    const int c0 = blockIdx.y * 64;
    for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
        const int c = i / 64, q = i % 64;
        tile[c][q]  = c0 + c < C && p0 + q < HW ? x[(int64_t) (c0 + c) * HW + p0 + q] : 0.0f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
        const int q = i / 64, c = i % 64;
        if (p0 + q < HW && c0 + c < C) {
            y[(int64_t) (p0 + q) * C + c0 + c] = (_Float16) tile[c][q];
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// GroupNorm, its per-channel affine and optionally SiLU, written as the next convolution's f16 channels-last input.
// The C / G channels of a group are one contiguous range of values. The stats pass splits every group into chunks of
// CHUNK values and leaves Welford partials (count, mean, M2) per chunk; the finalize pass merges them (Chan et al.)
// into the group's mean and 1 / std; the normalize pass writes y = (x - mean) * rstd * w[c] + b[c], then silu(y).

namespace gn {

constexpr int CHUNK   = 16384;  // values per stats block
constexpr int THREADS = 256;

struct welford {
    float n, mean, m2;
};

__device__ __forceinline__ welford welford_merge(const welford a, const welford b) {
    const float n = a.n + b.n;
    if (n == 0.0f) {
        return a;
    }
    const float d = b.mean - a.mean;
    const float f = b.n / n;
    return { n, a.mean + d * f, a.m2 + b.m2 + d * d * a.n * f };
}

__device__ __forceinline__ welford welford_wave_reduce(welford w) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const welford o = { __shfl_xor(w.n, off, 32), __shfl_xor(w.mean, off, 32), __shfl_xor(w.m2, off, 32) };
        w = welford_merge(w, o);
    }
    return w;
}

__global__ void __launch_bounds__(THREADS) stats_kernel(const float * __restrict__ x, welford * __restrict__ partial,
                                                        const int64_t group_size) {
    const int64_t group = blockIdx.y;
    const float * xg    = x + group * group_size;
    const int64_t end   = min(group_size, (int64_t) (blockIdx.x + 1) * CHUNK);
    welford       w     = { 0.0f, 0.0f, 0.0f };
    for (int64_t k = (int64_t) blockIdx.x * CHUNK + 4 * threadIdx.x; k < end; k += 4 * THREADS) {
        const float4 v = *(const float4 *) (xg + k);
        for (const float e : { v.x, v.y, v.z, v.w }) {
            w.n += 1.0f;
            const float d = e - w.mean;
            w.mean += d / w.n;
            w.m2 += d * (e - w.mean);
        }
    }
    w = welford_wave_reduce(w);
    __shared__ welford s[THREADS / 32];
    if (threadIdx.x % 32 == 0) {
        s[threadIdx.x / 32] = w;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < THREADS / 32; ++i) {
            w = welford_merge(w, s[i]);
        }
        partial[group * gridDim.x + blockIdx.x] = w;
    }
}

// One wave per group: the mean and 1 / std, merged from the chunk partials.
__global__ void finalize_kernel(const welford * __restrict__ partial, float2 * __restrict__ stats, const int chunks,
                                const float eps) {
    const int64_t group = blockIdx.y;
    welford       acc   = { 0.0f, 0.0f, 0.0f };
    for (int c = threadIdx.x; c < chunks; c += 32) {
        acc = welford_merge(acc, partial[group * chunks + c]);
    }
    acc = welford_wave_reduce(acc);
    if (threadIdx.x == 0) {
        stats[group] = make_float2(acc.mean, rsqrtf(acc.m2 / acc.n + eps));
    }
}

// The normalize pass through a 64 x 64 tile in LDS, rounded to f16 and written channels-last. A tile word holds the
// channel pair (2 k, 2 k + 1) of one pixel; 33 words per pixel keep both passes free of bank conflicts. hw % 4 == 0,
// so a float4 of pixels is all inside the map or all past its end, where the last tile stops.
template <bool silu>
__global__ void __launch_bounds__(THREADS) apply_nhwc_kernel(const float * __restrict__ x, const float2 * __restrict__ stats,
                                                             const float * __restrict__ w, const float * __restrict__ b,
                                                             _Float16 * __restrict__ y, const int C, const int64_t hw,
                                                             const int channels_per_group) {
    __shared__ uint32_t tile[64][33];
    const int64_t p0 = (int64_t) blockIdx.x * 64;
    const int     c0 = blockIdx.y * 64;
    const int     q  = 4 * (threadIdx.x % 16);
    for (int k = threadIdx.x / 16; k < 32; k += THREADS / 16) {
        float o[2][4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int    c    = c0 + 2 * k + h;
            const float2 st   = stats[c / channels_per_group];
            const float  mean = st.x;
            const float  rstd = st.y;
            const float  sc   = rstd * w[c];
            const float  bc   = b[c];
            const float4 v    = p0 + q < hw ? *(const float4 *) (x + (int64_t) c * hw + p0 + q)
                                            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            o[h][0] = (v.x - mean) * sc + bc;
            o[h][1] = (v.y - mean) * sc + bc;
            o[h][2] = (v.z - mean) * sc + bc;
            o[h][3] = (v.w - mean) * sc + bc;
            if constexpr (silu) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    o[h][i] = o[h][i] / (1.0f + expf(-o[h][i]));
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const _Float16 lo = (_Float16) o[0][i];
            const _Float16 hi = (_Float16) o[1][i];
            tile[q + i][k] = (uint32_t) __builtin_bit_cast(uint16_t, lo) | ((uint32_t) __builtin_bit_cast(uint16_t, hi) << 16);
        }
    }
    __syncthreads();
    const int k0     = 4 * (threadIdx.x % 8);  // channels 2 k0 .. 2 k0 + 7
    const int pixels = (int) min((int64_t) 64, hw - p0);
    for (int qq = threadIdx.x / 8; qq < pixels; qq += THREADS / 8) {
        const uint4 v = make_uint4(tile[qq][k0], tile[qq][k0 + 1], tile[qq][k0 + 2], tile[qq][k0 + 3]);
        *(uint4 *) (y + (p0 + qq) * C + c0 + 2 * k0) = v;
    }
}

int64_t chunks(const int C, const int64_t HW, const int groups) {
    return ((C / groups) * HW + CHUNK - 1) / CHUNK;
}

// The partials, rounded up to 16 bytes; the per-group stats follow them in the scratch.
size_t partial_bytes(const int64_t n_chunks, const int groups) {
    return (n_chunks * groups * sizeof(welford) + 15) / 16 * 16;
}

} // namespace gn

// ---------------------------------------------------------------------------------------------------------------
// The mid-block attention's softmax and the elementwise steps at the ends of the decoder.

// ggml's soft_max_f32 on one row of scores, without mask: block_size threads (the power of two from 32 that covers the
// row, at most 1024) take every block_size-th column. The scale is the ggml_scale before the softmax, one rounding of
// x * scale like ggml's x * scale + 0; the exponentials are computed again instead of cached, with the same values.
template <int block_size>
__global__ void __launch_bounds__(block_size) softmax_f16_kernel(const float * s, const int64_t cols, const float scale,
                                                                 uint16_t * p) {
#pragma clang fp contract(off)
    __shared__ float partial[32];
    const float * row = s + (int64_t) blockIdx.x * cols;
    uint16_t *    out = p + (int64_t) blockIdx.x * cols;
    float max_val = -INFINITY;
    for (int64_t col = threadIdx.x; col < cols; col += block_size) {
        max_val = fmaxf(max_val, __builtin_fmaf(row[col], scale, 0.0f));
    }
    max_val = block_max<block_size>(max_val, partial);
    float sum = 0.0f;
    for (int64_t col = threadIdx.x; col < cols; col += block_size) {
        sum += expf(__builtin_fmaf(row[col], scale, 0.0f) - max_val);
    }
    if constexpr (block_size > 32) {
        __syncthreads();  // partial is reused
    }
    sum = block_sum<block_size>(sum, partial);
    const float inv_sum = 1.0f / sum;
    for (int64_t col = threadIdx.x; col < cols; col += block_size) {
        out[col] = to_f16(expf(__builtin_fmaf(row[col], scale, 0.0f) - max_val) * inv_sum);
    }
}

template <int block_size>
void launch_softmax_f16(const float * s, const int64_t rows, const int64_t cols, const float scale, uint16_t * p,
                        hipStream_t stream) {
    softmax_f16_kernel<block_size><<<(unsigned) rows, block_size, 0, stream>>>(s, cols, scale, p);
}

__global__ void f32_to_f16_kernel(const float * x, const int64_t n, uint16_t * dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = to_f16(x[i]);
    }
}

// sd.cpp's diffusion_to_vae_latents on the DiT state, then the 2 x 2 unpatchify: latent[c][2 y + py][2 x + px] =
// state[y w + x][k] * std[k] + mean[k], k = 4 c + 2 py + px. The host steps round one at a time (the division by the
// scale factor 1 is exact).
__global__ void unpack_latent_kernel(const float * state, const float * mean, const float * std, const int h,
                                     const int w, float * latent) {
#pragma clang fp contract(off)
    const int64_t i     = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t plane = (int64_t) 4 * h * w;
    if (i >= 32 * plane) {
        return;
    }
    const int     c = (int) (i / plane);
    const int     y = (int) (i % plane / (2 * w));
    const int     x = (int) (i % (2 * w));
    const int     k = 4 * c + 2 * (y % 2) + x % 2;
    const int64_t t = (int64_t) (y / 2) * w + x / 2;
    latent[i] = state[t * 128 + k] * std[k] + mean[k];
}

// sd.cpp's decoded image to 8 bits on the host: (x + 1) * 0.5 clamped to [0, 1] (std::min, then std::max), then 0 and
// 255 at the ends and v * 255 + 0.5 truncated in between.
__global__ void to_rgb8_kernel(const float * x, const int64_t HW, uint8_t * rgb) {
#pragma clang fp contract(off)
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 3 * HW) {
        return;
    }
    const float v  = (x[i % 3 * HW + i / 3] + 1.0f) * 0.5f;
    const float lo = v < 1.0f ? v : 1.0f;
    const float u  = 0.0f < lo ? lo : 0.0f;
    rgb[i] = u <= 0.0f ? 0 : u >= 1.0f ? 255 : (uint8_t) (u * 255.0f + 0.5f);
}

} // namespace

void f32_to_f16(const float * x, const int64_t n, uint16_t * dst, hipStream_t stream) {
    f32_to_f16_kernel<<<elementwise_blocks(n), kBlock, 0, stream>>>(x, n, dst);
}

void unpack_latent(const float * state, const float * mean, const float * std, const int h, const int w, float * latent,
                   hipStream_t stream) {
    unpack_latent_kernel<<<elementwise_blocks((int64_t) 128 * h * w), kBlock, 0, stream>>>(state, mean, std, h, w,
                                                                                          latent);
}

void to_nhwc_f16(const float * x, const int C, const int64_t HW, uint16_t * y, hipStream_t stream) {
    const dim3 grid((unsigned) ((HW + 63) / 64), (unsigned) ((C + 63) / 64));
    to_nhwc_kernel<<<grid, 256, 0, stream>>>(x, (_Float16 *) y, C, (int) HW);
}

size_t group_norm_scratch_bytes(const int C, const int64_t HW, const int groups) {
    return gn::partial_bytes(gn::chunks(C, HW, groups), groups) + groups * sizeof(float2);
}

void group_norm_nhwc(const float * x, const int C, const int64_t HW, const int groups, const float eps,
                     const float * gamma, const float * beta, const bool silu, void * scratch, uint16_t * y,
                     hipStream_t stream) {
    const int64_t  n_chunks = gn::chunks(C, HW, groups);
    gn::welford *  partial  = (gn::welford *) scratch;
    float2 *       stats    = (float2 *) ((char *) scratch + gn::partial_bytes(n_chunks, groups));
    gn::stats_kernel<<<dim3((unsigned) n_chunks, (unsigned) groups), gn::THREADS, 0, stream>>>(
        x, partial, (C / groups) * HW);
    gn::finalize_kernel<<<dim3(1, (unsigned) groups), 32, 0, stream>>>(partial, stats, (int) n_chunks, eps);
    const dim3 grid((unsigned) ((HW + 63) / 64), (unsigned) (C / 64));
    if (silu) {
        gn::apply_nhwc_kernel<true><<<grid, gn::THREADS, 0, stream>>>(x, stats, gamma, beta, (_Float16 *) y, C, HW,
                                                                      C / groups);
    } else {
        gn::apply_nhwc_kernel<false><<<grid, gn::THREADS, 0, stream>>>(x, stats, gamma, beta, (_Float16 *) y, C, HW,
                                                                       C / groups);
    }
}

void conv2d(const uint16_t * x, const int IW, const int IH, const int IC, const bool up2x, const uint16_t * w,
            const int KW, const int KH, const int OC, const int px, const int py, const float * bias, const float * res,
            float * y, hipStream_t stream) {
    const int          OW = IW + 2 * px - KW + 1;
    const int          OH = IH + 2 * py - KH + 1;
    const conv::params p  = { (const _Float16 *) x, (const _Float16 *) w, bias, res, y, IW, IH, OW, OH, KW, KH, px, py,
                              IC, OC };
    if (OC <= conv::SM_BM && !up2x) {
        // 4 waves x 32 pixels per block: the fastest of 64-256 threads x 128-256 pixels for conv_out at 1024 x 1024
        conv::small_oc_kernel<128, 128><<<(unsigned) ((OH * OW + 127) / 128), 128, 0, stream>>>(p);
        return;
    }
    const dim3 grid((unsigned) ((OH * OW + conv::BN - 1) / conv::BN), (unsigned) ((OC + conv::BM - 1) / conv::BM));
    if (up2x) {
        conv::conv2d_kernel<true><<<grid, conv::THREADS, 0, stream>>>(p);
    } else {
        conv::conv2d_kernel<false><<<grid, conv::THREADS, 0, stream>>>(p);
    }
}

void softmax_f16(const float * s, const int64_t rows, const int64_t cols, const float scale, uint16_t * p,
                 hipStream_t stream) {
    if (cols <= 32) {
        launch_softmax_f16<32>(s, rows, cols, scale, p, stream);
    } else if (cols <= 64) {
        launch_softmax_f16<64>(s, rows, cols, scale, p, stream);
    } else if (cols <= 128) {
        launch_softmax_f16<128>(s, rows, cols, scale, p, stream);
    } else if (cols <= 256) {
        launch_softmax_f16<256>(s, rows, cols, scale, p, stream);
    } else if (cols <= 512) {
        launch_softmax_f16<512>(s, rows, cols, scale, p, stream);
    } else {
        launch_softmax_f16<1024>(s, rows, cols, scale, p, stream);
    }
}

void to_rgb8(const float * x, const int64_t HW, uint8_t * rgb, hipStream_t stream) {
    to_rgb8_kernel<<<elementwise_blocks(3 * HW), kBlock, 0, stream>>>(x, HW, rgb);
}

} // namespace luce::moss_image
