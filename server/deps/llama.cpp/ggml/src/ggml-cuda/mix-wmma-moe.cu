// F16 WMMA routed-expert GEMM for the ROCmFP2/ROCmFP3 MIX expert types
// (qtype 106 / 105) on RDNA3 (gfx11), used for large mul_mat_id batches
// (prefill). The MMQ path re-decodes every weight tile for each 64-route tile
// and runs the codebook unpack inside its int8 inner loop; here each K step
// decodes a 128x32 weight tile once into LDS as F16 (scale * codebook level,
// the same float product the dequantize kernels round to half) and eight
// waves accumulate it against 64 gathered F16 routes on the matrix cores.
//
// On by default on RDNA3.5 (gfx115x, the arch it is validated on;
// LUCE_MIX_WMMA_PREFILL=0 disables); other GPUs keep MMQ.
//
// The structure (dequantize a weight tile into padded LDS, then accumulate it
// on the WMMA units against gathered routed activations) follows Piotr
// Wilkin's bf16 WMMA dequant GEMM for large prefill batches (mmb.cu, in his
// Strix Halo llama.cpp work integrated into ROCmFPX), adapted here to F16 and
// to the ROCmFP2/FP3 MIX codebook formats.

#include "mix-wmma-moe.cuh"
#include "unary.cuh"
#include "mmid.cuh"
#include "rocmfp2_mix.cuh"
#include "rocmfp3_mix.cuh"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <type_traits>

#if defined(GGML_USE_HIP) && defined(__HIP_DEVICE_COMPILE__) && defined(RDNA3)
#define MIX_WMMA_DEVICE 1
#else
#define MIX_WMMA_DEVICE 0
#endif

namespace {

constexpr int kBM = 128;          // output rows per block
constexpr int kBN = 64;           // routes per block
constexpr int kThreads = 256;

template <int TYPE> struct MixFormat;
template <> struct MixFormat<GGML_TYPE_Q2_1_ROCMFP2_MIX> {
    static constexpr int kBlockBytes = 10, kCodeBytes = 8, kLevels = 4, kBits = 2;
};
template <> struct MixFormat<GGML_TYPE_Q3_1_ROCMFP3_MIX> {
    static constexpr int kBlockBytes = 14, kCodeBytes = 12, kLevels = 8, kBits = 3;
};

// Mode-0 fixed levels, code order (see mix_fp2_fixed / mix_fp3_fixed).
template <int TYPE>
__device__ __forceinline__ float mix_wmma_fixed(uint32_t code) {
    if constexpr (TYPE == GGML_TYPE_Q2_1_ROCMFP2_MIX) {
        return (float) ((int) code - 1);
    } else {
        const uint32_t m = code & 3u;
        const int mag = (m == 3u) ? 4 : (int) m;
        return (code & 4u) ? -(float) mag : (float) mag;
    }
}

// Expert-sorted route ranges -> compact (expert, first route) tiles. One
// block, one thread per expert (n_experts <= 1024).
__global__ void mix_wmma_build_tiles(const int32_t * __restrict__ expert_bounds, int n_experts,
                                     int2 * __restrict__ tiles, int * __restrict__ n_tiles, int bn) {
    __shared__ int s_scan[1024];
    const int e = threadIdx.x;
    int count = 0, lo = 0;
    if (e < n_experts) {
        lo = expert_bounds[e];
        count = (expert_bounds[e + 1] - lo + bn - 1) / bn;
    }
    s_scan[e] = count;
    __syncthreads();
    for (int off = 1; off < (int) blockDim.x; off <<= 1) {
        const int v = e >= off ? s_scan[e - off] : 0;
        __syncthreads();
        s_scan[e] += v;
        __syncthreads();
    }
    const int first = s_scan[e] - count;
    for (int i = 0; i < count; ++i) {
        tiles[first + i] = make_int2(e, lo + i * bn);
    }
    if (e == (int) blockDim.x - 1) {
        *n_tiles = s_scan[e];
    }
}

// X[r][k] = half(src1[ids_src1[r] * s11 + k]); 8 halves per thread.
__global__ void mix_wmma_gather(const float * __restrict__ src1, const int32_t * __restrict__ ids_src1,
                                half * __restrict__ x, int64_t n_routes, int k, int64_t s11) {
    const int64_t i = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const int64_t r = i / k;
    const int c = (int) (i % k);
    if (r >= n_routes) return;
    const float * src = src1 + (int64_t) ids_src1[r] * s11 + c;
    half2 * dst = reinterpret_cast<half2 *>(x + r * k + c);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        dst[j] = __floats2half2_rn(src[2 * j], src[2 * j + 1]);
    }
}

#if MIX_WMMA_DEVICE
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f  = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v16h mix_wmma_frag(const half * p) {
    union { v16h f; uint4 u[2]; } cvt;
    cvt.u[0] = *reinterpret_cast<const uint4 *>(p);
    cvt.u[1] = *reinterpret_cast<const uint4 *>(p + 8);
    return cvt.f;
}
#endif

#if MIX_WMMA_DEVICE
// UE4M3 from its bits: (8 + m) * 2^(e - 11) is the float with exponent field
// e + 119 and top mantissa bits m; e == 0 is m * 2^-10; codes above 0x7E are 0.
__device__ __forceinline__ float mix_wmma_scale(uint32_t e) {
    const uint32_t ex = e >> 3, mant = e & 7u;
    const float normal = __uint_as_float(((ex + 119u) << 23) | (mant << 20));
    const float sub = (float) mant * 0.0009765625f;
    return e > 0x7Eu ? 0.0f : (ex ? normal : sub);
}

__device__ __forceinline__ uint32_t mix_wmma_pack(float a, float b) {
    const half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t *>(&h);
}

// Pair of halves for codes c0 (low) and c1 (high): x = c0 | c1 << 16 selects
// bytes 2c, 2c+1 of the table {t1:t0} (perm bytes 0-3 are t0, 4-7 are t1).
__device__ __forceinline__ uint32_t mix_wmma_pick(uint32_t t0, uint32_t t1, uint32_t x) {
    return __builtin_amdgcn_perm(t1, t0, x * 0x202u + 0x01000100u);
}

// Decode one 16-weight half-block into eight packed half2 words.
template <int TYPE>
__device__ __forceinline__ void mix_wmma_decode_half(uint64_t codes, uint32_t meta, int mode,
                                                     const float * __restrict__ s_lut, uint32_t out[8]) {
    using F = MixFormat<TYPE>;
    float lev[F::kLevels];
    if (mode == 0) {
        const float s = mix_wmma_scale(meta);
#pragma unroll
        for (int c = 0; c < F::kLevels; ++c) lev[c] = s * mix_wmma_fixed<TYPE>((uint32_t) c);
    } else {
        const float s = mix_wmma_scale(meta & 0x7Fu);
        const float * l = s_lut + (meta >> 7) * F::kLevels;
#pragma unroll
        for (int c = 0; c < F::kLevels; ++c) lev[c] = s * l[c];
    }
    const uint32_t t0 = mix_wmma_pack(lev[0], lev[1]);
    const uint32_t t1 = mix_wmma_pack(lev[2], lev[3]);
    if constexpr (TYPE == GGML_TYPE_Q2_1_ROCMFP2_MIX) {
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const uint32_t c0 = (uint32_t) (codes >> (4 * p)) & 3u;
            const uint32_t c1 = (uint32_t) (codes >> (4 * p + 2)) & 3u;
            out[p] = mix_wmma_pick(t0, t1, c0 | (c1 << 16));
        }
    } else {
        const uint32_t t2 = mix_wmma_pack(lev[4], lev[5]);
        const uint32_t t3 = mix_wmma_pack(lev[6], lev[7]);
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const uint32_t c0 = (uint32_t) (codes >> (6 * p)) & 7u;
            const uint32_t c1 = (uint32_t) (codes >> (6 * p + 3)) & 7u;
            const uint32_t x = (c0 & 3u) | ((c1 & 3u) << 16);
            const uint32_t lo = mix_wmma_pick(t0, t1, x);
            const uint32_t hi = mix_wmma_pick(t2, t3, x);
            const uint32_t mask = ((c0 & 4u) ? 0x0000FFFFu : 0u) | ((c1 & 4u) ? 0xFFFF0000u : 0u);
            out[p] = (lo & ~mask) | (hi & mask);
        }
    }
}

// Raw bytes of one quant block (2-byte aligned: blocks are 10 or 14 bytes).
template <int TYPE> struct MixRaw { uint16_t v[(MixFormat<TYPE>::kBlockBytes + 1) / 2]; };

template <int TYPE>
__device__ __forceinline__ MixRaw<TYPE> mix_wmma_load_raw(const uint8_t * p) {
    MixRaw<TYPE> r;
    if constexpr (MixFormat<TYPE>::kBlockBytes % 2 == 0) {
        const uint16_t * q = reinterpret_cast<const uint16_t *>(p);
#pragma unroll
        for (int i = 0; i < MixFormat<TYPE>::kBlockBytes / 2; ++i) r.v[i] = q[i];
    } else {
        uint8_t * b = reinterpret_cast<uint8_t *>(r.v);
#pragma unroll
        for (int i = 0; i < MixFormat<TYPE>::kBlockBytes; ++i) b[i] = p[i];
    }
    return r;
}

#endif

constexpr int kBK2 = 64;                  // two quant blocks per row per step
constexpr int kLds2 = kBK2 + 8;

template <int TYPE, int BN = kBN>
__launch_bounds__(kThreads) __global__ void mix_wmma_moe_kernel(
        const uint8_t * __restrict__ w, int64_t nb02, int64_t nb01, int m, int k,
        const nv_bfloat16 * __restrict__ codebooks, const uint8_t * __restrict__ modes,
        const half * __restrict__ x, const int2 * __restrict__ tiles, const int * __restrict__ n_tiles,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst, int64_t s1,
        const float * __restrict__ glu_gate = nullptr, int64_t s_gate = 0, float glu_limit = 0.0f) {
#if MIX_WMMA_DEVICE
    using F = MixFormat<TYPE>;
    if ((int) blockIdx.y >= *n_tiles) return;
    const int2 tile = tiles[blockIdx.y];
    const int expert = tile.x;
    const int r0 = tile.y;
    const int r_end = min(r0 + BN, expert_bounds[expert + 1]);
    const int m0 = blockIdx.x * kBM;
    constexpr int kNF = BN / 32;          // 16-route fragments per wave (waves split BN in two)
    constexpr int kXLoads = BN * 8 / kThreads;   // uint4 activation loads per thread per step

    __shared__ __align__(16) half s_w[kBM * kLds2];
    __shared__ __align__(16) half s_x[BN * kLds2];
    __shared__ float s_lut[2 * F::kLevels];

    const int tid = threadIdx.x;
    if (tid < 2 * F::kLevels) {
        s_lut[tid] = __bfloat162float(codebooks[(int64_t) expert * 2 * F::kLevels + tid]);
    }
    const int mode = modes[expert];

    // Decode: row tid/2, quant block (tid&1) of the step's two.
    const int d_row = tid >> 1;
    const int d_blk = tid & 1;
    const uint8_t * w_row = w + (int64_t) expert * nb02 + (int64_t) (m0 + d_row) * nb01
                          + (int64_t) d_blk * F::kBlockBytes;
    // Activations: flat uint4 slot tid + i*kThreads -> route slot/8, halves (slot%8)*8.
    const half * x_src[kXLoads];
    int x_lds[kXLoads];
    bool x_live[kXLoads];
#pragma unroll
    for (int i = 0; i < kXLoads; ++i) {
        const int slot = tid + i * kThreads;
        const int row = slot >> 3, col = (slot & 7) * 8;
        x_live[i] = r0 + row < r_end;
        x_src[i] = x + (int64_t) (r0 + row) * k + col;
        x_lds[i] = row * kLds2 + col;
    }

    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int wm = (wave >> 1) * 32;
    const int wn = (wave & 1) * (BN / 2);
    const int sub = lane & 15;
    const int hl = lane >> 4;

    v8f acc[2][kNF] = {};
    const int nk = k / kBK2;

    MixRaw<TYPE> raw = mix_wmma_load_raw<TYPE>(w_row);
    uint4 xv[kXLoads];
#pragma unroll
    for (int i = 0; i < kXLoads; ++i) {
        xv[i] = x_live[i] ? *reinterpret_cast<const uint4 *>(x_src[i]) : make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();   // s_lut

    for (int kb = 0; kb < nk; ++kb) {
        {
            const uint8_t * rb = reinterpret_cast<const uint8_t *>(raw.v);
            uint32_t o0[8], o1[8];
            uint64_t lo = 0; uint32_t hi = 0;
            // Code bytes as one little-endian stream.
            uint8_t cb[16] = {};
#pragma unroll
            for (int i = 0; i < F::kCodeBytes; ++i) cb[i] = rb[i];
            std::memcpy(&lo, cb, 8);
            std::memcpy(&hi, cb + 8, 4);
            const uint32_t meta0 = rb[F::kCodeBytes], meta1 = rb[F::kCodeBytes + 1];
            if constexpr (TYPE == GGML_TYPE_Q2_1_ROCMFP2_MIX) {
                mix_wmma_decode_half<TYPE>(lo & 0xFFFFFFFFull, meta0, mode, s_lut, o0);
                mix_wmma_decode_half<TYPE>(lo >> 32, meta1, mode, s_lut, o1);
            } else {
                // 48 bits per half: bits 0..47 and 48..95 of the 96-bit stream.
                const uint64_t h1 = (lo >> 48) | ((uint64_t) hi << 16);
                mix_wmma_decode_half<TYPE>(lo & 0xFFFFFFFFFFFFull, meta0, mode, s_lut, o0);
                mix_wmma_decode_half<TYPE>(h1, meta1, mode, s_lut, o1);
            }
            uint4 * dw = reinterpret_cast<uint4 *>(&s_w[d_row * kLds2 + d_blk * 32]);
            dw[0] = make_uint4(o0[0], o0[1], o0[2], o0[3]);
            dw[1] = make_uint4(o0[4], o0[5], o0[6], o0[7]);
            dw[2] = make_uint4(o1[0], o1[1], o1[2], o1[3]);
            dw[3] = make_uint4(o1[4], o1[5], o1[6], o1[7]);
#pragma unroll
            for (int i = 0; i < kXLoads; ++i) *reinterpret_cast<uint4 *>(&s_x[x_lds[i]]) = xv[i];
        }
        __syncthreads();
        if (kb + 1 < nk) {
            raw = mix_wmma_load_raw<TYPE>(w_row + (int64_t) (kb + 1) * 2 * F::kBlockBytes);
#pragma unroll
            for (int i = 0; i < kXLoads; ++i) {
                if (x_live[i]) xv[i] = *reinterpret_cast<const uint4 *>(x_src[i] + (kb + 1) * kBK2);
            }
        }
#pragma unroll
        for (int ks = 0; ks < kBK2; ks += 16) {
            v16h a[2], b[kNF];
#pragma unroll
            for (int i = 0; i < 2; ++i) a[i] = mix_wmma_frag(&s_w[(wm + i * 16 + sub) * kLds2 + ks]);
#pragma unroll
            for (int j = 0; j < kNF; ++j) b[j] = mix_wmma_frag(&s_x[(wn + j * 16 + sub) * kLds2 + ks]);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < kNF; ++j)
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < kNF; ++j) {
        const int r = r0 + wn + j * 16 + sub;
        if (r >= r_end) continue;
        const int64_t row_id = ids_dst[r];
        float * out_row = dst + row_id * s1 + m0;
        if (glu_gate) {
            // Fused SwiGLU-DS4: this launch computes `up`; gate was written by
            // the previous launch. Same function as the standalone glu kernel.
            const float * gate_row = glu_gate + row_id * s_gate + m0;
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const int c = wm + i * 16 + 2 * v + hl;
                    out_row[c] = ggml_cuda_op_swiglu_ds4_single(gate_row[c], acc[i][j][v], glu_limit);
                }
        } else {
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int v = 0; v < 8; ++v) out_row[wm + i * 16 + 2 * v + hl] = acc[i][j][v];
        }
    }
#else
    GGML_UNUSED_VARS(w, nb02, nb01, m, k, codebooks, modes, x, tiles, n_tiles, expert_bounds, ids_dst, dst, s1, glu_gate, s_gate, glu_limit);
#endif
}

// Masked owner routes (negative expert ids) are compacted out of the tiles,
// so no tile writes their destination column (route r = token *
// n_expert_used + slot, as in ids_dst). MMQ clears all of dst for this;
// clearing only these columns keeps their contribution exactly zero.
__global__ void mix_wmma_zero_masked(const int32_t * __restrict__ ids, int si1, int n_expert_used,
                                     float * __restrict__ dst, int64_t s1, int m) {
    const int64_t r = blockIdx.x;
    const int64_t t = r / n_expert_used;
    const int j = (int) (r % n_expert_used);
    if (ids[t * si1 + j] >= 0) return;
    float * col = dst + r * s1;
    for (int i = threadIdx.x; i < m; i += blockDim.x) col[i] = 0.0f;
}

}  // namespace

// On by default; LUCE_MIX_WMMA_PREFILL=0 returns these batches to MMQ.
static bool mix_wmma_prefill_enabled() {
    static const bool enabled = [] {
        const char * v = std::getenv("LUCE_MIX_WMMA_PREFILL");
        return !(v && std::strcmp(v, "0") == 0);
    }();
    return enabled;
}

static std::atomic<uint64_t> g_mix_wmma_launches{0};

bool ggml_cuda_mix_wmma_moe_available(int device) {
    return mix_wmma_prefill_enabled() && device >= 0 && device < ggml_cuda_info().device_count &&
           GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[device].cc);
}

uint64_t ggml_cuda_mix_wmma_moe_launch_count() {
    return g_mix_wmma_launches.load(std::memory_order_relaxed);
}

bool ggml_cuda_mix_wmma_moe_enabled(const ggml_tensor * src0, const ggml_tensor * src1,
                                    const ggml_tensor * ids, int64_t n_tokens, int cc) {
    const bool enabled = mix_wmma_prefill_enabled();
    // A whole number >= 1; anything else keeps the default (0 would send
    // decode batches here).
    static const int min_tokens = [] {
        const char * v = std::getenv("LUCE_MIX_WMMA_MIN_TOKENS");
        char * end = nullptr;
        const long n = v && *v ? std::strtol(v, &end, 10) : 0;
        return (end && *end == '\0' && n >= 1 && n <= 1 << 20) ? (int) n : 64;
    }();
    if (!enabled || !GGML_CUDA_CC_IS_RDNA3_5(cc)) return false;
    if (src0->type != GGML_TYPE_Q2_1_ROCMFP2_MIX && src0->type != GGML_TYPE_Q3_1_ROCMFP3_MIX) return false;
    if (!(n_tokens >= min_tokens && src0->ne[0] % kBK2 == 0 && src0->ne[1] % kBM == 0 && src0->ne[2] <= 1024)) {
        return false;
    }
    // The gather reads each activation row as K contiguous floats.
    if (!ids || src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1)) return false;
    // One grid row per route tile: at most n_routes / 64 + n_experts tiles
    // (bn >= 64) must fit the 65535 grid-y limit; longer batches stay on MMQ.
    const int64_t max_tiles = (ids->ne[0] * n_tokens + 63) / 64 + src0->ne[2];
    return max_tiles <= 65535;
}

// Routes, tiles and the gathered F16 activations are shared by every weight
// that multiplies the same src1/ids (the gate/up pair); each weight then gets
// its own launch.
static void mix_wmma_moe_run(ggml_backend_cuda_context & ctx, const ggml_tensor * const * src0s,
                             ggml_tensor * const * dsts, int n_weights,
                             const ggml_tensor * src1, const ggml_tensor * ids,
                             const ggml_tensor * glu_gate = nullptr, float glu_limit = 0.0f) {
    cudaStream_t stream = ctx.stream();
    const ggml_tensor * src0 = src0s[0];
    const int64_t k = src0->ne[0];
    const int64_t m = src0->ne[1];
    const int64_t n_experts = src0->ne[2];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_routes = ne12 * n_expert_used;
    GGML_ASSERT(dsts[0]->ne[1] == n_expert_used);
    GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
    const bool fp2 = src0->type == GGML_TYPE_Q2_1_ROCMFP2_MIX;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_routes);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_routes);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), n_experts + 1);
    CUDA_CHECK(cudaMemsetAsync(ids_src1.get(), 0, n_routes * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(ids_dst.get(), 0, n_routes * sizeof(int32_t), stream));
    const int si1  = ids->nb[1] / ggml_element_size(ids);
    const int sis1 = src1->nb[2] / src1->nb[1];
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        (int) n_experts, (int) ne12, (int) n_expert_used, (int) ne11, si1, sis1, /*write_inverse=*/false, stream);

    // Route-tile width: decode work per FLOP falls as 1/bn, padding rises
    // with it. Pick from the mean routes per expert; LUCE_MIX_WMMA_BN forces.
    static const int forced_bn = [] {
        const char * v = std::getenv("LUCE_MIX_WMMA_BN");
        const int b = v && *v ? std::atoi(v) : 0;
        return (b == 64 || b == 96 || b == 128) ? b : 0;
    }();
    const int64_t mean_routes = n_routes / std::max<int64_t>(1, n_experts);
    const int bn = forced_bn ? forced_bn : mean_routes >= 128 ? 128 : 64;
    const int64_t max_tiles = (n_routes + bn - 1) / bn + n_experts;
    ggml_cuda_pool_alloc<int2> tiles(ctx.pool(), max_tiles);
    ggml_cuda_pool_alloc<int> n_tiles(ctx.pool(), 1);
    int tile_threads = 32;
    while (tile_threads < n_experts) tile_threads <<= 1;
    mix_wmma_build_tiles<<<1, tile_threads, 0, stream>>>(expert_bounds.get(), (int) n_experts, tiles.get(), n_tiles.get(), bn);

    ggml_cuda_pool_alloc<half> x(ctx.pool(), n_routes * k);
    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t gather_threads = n_routes * k / 8;
    mix_wmma_gather<<<(gather_threads + 255) / 256, 256, 0, stream>>>(
        (const float *) src1->data, ids_src1.get(), x.get(), n_routes, (int) k, s11);

    // Side data must stay registered until the kernels are enqueued.
    struct RegistryLock {
        bool fp2;
        explicit RegistryLock(bool f) : fp2(f) {
            if (fp2) ggml_cuda_rocmfp2_mix_registry_lock(); else ggml_cuda_rocmfp3_mix_registry_lock();
        }
        ~RegistryLock() {
            if (fp2) ggml_cuda_rocmfp2_mix_registry_unlock(); else ggml_cuda_rocmfp3_mix_registry_unlock();
        }
    } registry_lock(fp2);

    const dim3 grid((unsigned) (m / kBM), (unsigned) max_tiles);
    for (int wi = 0; wi < n_weights; ++wi) {
        const ggml_tensor * w = src0s[wi];
        ggml_tensor * dst = dsts[wi];
        const void * codebooks = nullptr;
        const uint8_t * modes = nullptr;
        GGML_ASSERT(fp2 ? ggml_cuda_rocmfp2_mix_mmq_info(w->data, &codebooks, &modes)
                        : ggml_cuda_rocmfp3_mix_mmq_info(w->data, &codebooks, &modes));
        const int64_t s1 = dst->nb[1] / sizeof(float);
        mix_wmma_zero_masked<<<(unsigned) n_routes, 256, 0, stream>>>(
            (const int32_t *) ids->data, si1, (int) n_expert_used, (float *) dst->data, s1, (int) m);
        // The last weight may apply SwiGLU-DS4 against an already computed gate.
        const bool fuse = glu_gate && wi == n_weights - 1;
        const float * gate_ptr = fuse ? (const float *) glu_gate->data : nullptr;
        const int64_t s_gate = fuse ? glu_gate->nb[1] / (int64_t) sizeof(float) : 0;
        const auto launch = [&](auto type_tag, auto bn_tag) {
            constexpr int TY = decltype(type_tag)::value;
            constexpr int BN = decltype(bn_tag)::value;
            mix_wmma_moe_kernel<TY, BN><<<grid, kThreads, 0, stream>>>(
                (const uint8_t *) w->data, w->nb[2], w->nb[1], (int) m, (int) k,
                (const nv_bfloat16 *) codebooks, modes, x.get(), tiles.get(), n_tiles.get(),
                expert_bounds.get(), ids_dst.get(), (float *) dst->data, s1,
                gate_ptr, s_gate, glu_limit);
        };
        using fp2_t = std::integral_constant<int, GGML_TYPE_Q2_1_ROCMFP2_MIX>;
        using fp3_t = std::integral_constant<int, GGML_TYPE_Q3_1_ROCMFP3_MIX>;
        using bn64 = std::integral_constant<int, 64>;
        using bn96 = std::integral_constant<int, 96>;
        using bn128 = std::integral_constant<int, 128>;
        if (fp2) {
            if (bn == 128) launch(fp2_t{}, bn128{}); else if (bn == 96) launch(fp2_t{}, bn96{}); else launch(fp2_t{}, bn64{});
        } else {
            if (bn == 128) launch(fp3_t{}, bn128{}); else if (bn == 96) launch(fp3_t{}, bn96{}); else launch(fp3_t{}, bn64{});
        }
    }
    CUDA_CHECK(cudaGetLastError());
    g_mix_wmma_launches.fetch_add(1, std::memory_order_relaxed);
}

void ggml_cuda_mix_wmma_moe(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                            const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    const ggml_tensor * w[1] = {src0};
    ggml_tensor * d[1] = {dst};
    mix_wmma_moe_run(ctx, w, d, 1, src1, ids);
}

void ggml_cuda_mix_wmma_moe_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * src0_a, const ggml_tensor * src0_b,
                                 const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst_a, ggml_tensor * dst_b) {
    GGML_ASSERT(src0_a->type == src0_b->type && ggml_are_same_shape(src0_a, src0_b));
    const ggml_tensor * w[2] = {src0_a, src0_b};
    ggml_tensor * d[2] = {dst_a, dst_b};
    mix_wmma_moe_run(ctx, w, d, 2, src1, ids);
}

void ggml_cuda_mix_wmma_moe_pair_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * w_up, const ggml_tensor * w_gate,
                                     const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * gate_dst,
                                     ggml_tensor * glu_dst, float limit) {
    GGML_ASSERT(w_up->type == w_gate->type && ggml_are_same_shape(w_up, w_gate));
    GGML_ASSERT(ggml_are_same_shape(gate_dst, glu_dst) && ggml_is_contiguous(glu_dst));
    const ggml_tensor * w[2] = {w_gate, w_up};
    ggml_tensor * d[2] = {gate_dst, glu_dst};
    mix_wmma_moe_run(ctx, w, d, 2, src1, ids, gate_dst, limit);
}
