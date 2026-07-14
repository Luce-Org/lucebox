// GPU top-K + log-prob extraction for DDTree draft distributions.
// See geometric_draft_topk_cuda.h for the contract. Mirrors extract_draft_topk (ddtree.cpp).

#include "geometric_draft_topk_cuda.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace dflash::common {

namespace {

constexpr int kMaxK    = 8;     // ddtree_K is 8 in practice; K>kMaxK → CPU fallback
constexpr int kBlock   = 256;   // threads per block (power of two for the reduction)
constexpr int kMaxSplit = 128;  // max vocab splits per position (combine-block cap)

// The workload is tiny in rows (n_positions ~ 15) but huge in vocab (~152k),
// so it is purely DRAM-bandwidth bound: ~9 MB to read per call. We 
// split each position's vocab scan across `split` blocks (a 2D grid of
// n_positions × split) so the whole device stays busy, then a cheap second
// kernel merges the `split` partials per position into the final top-K.

// Merge two sorted-descending top-K lists (av/ai, bv/bi) into dst (dv/di),
// keeping the K largest; ties resolve toward the lower id. dst may alias av/ai
// safely only via the temporary below, so callers pass a distinct dst when the
// inputs come from shared memory.
template <int K>
__device__ __forceinline__ void merge_topk(const float * av, const int * ai,
                                            const float * bv, const int * bi,
                                            float * dv, int * di) {
    int ia = 0, ib = 0;
#pragma unroll
    for (int k = 0; k < K; k++) {
        bool takeA;
        if      (ib >= K)         takeA = true;
        else if (ia >= K)         takeA = false;
        else if (av[ia] > bv[ib]) takeA = true;
        else if (av[ia] < bv[ib]) takeA = false;
        else                      takeA = (ai[ia] <= bi[ib]);  // tie → lower id
        if (takeA) { dv[k] = av[ia]; di[k] = ai[ia]; ia++; }
        else       { dv[k] = bv[ib]; di[k] = bi[ib]; ib++; }
    }
}

// Fold one logit (raw value `LRAW`, vocab id `ID`) into a thread's running
// online-logsumexp (lmax,lsum) and its register-resident sorted-descending
// top-K (topv,topi). K is a compile-time constant so the unrolled bubble uses
// only fixed indices — the arrays stay in registers instead of spilling to
// local memory the way a data-dependent insertion index would. The new value
// enters at slot K-1 and bubbles up only past strictly-smaller entries, so on
// ties the earlier (lower-id, since ids ascend within a thread) entry wins —
// matching the CPU heap's first-wins behaviour.
#define DFLASH_TOPK_CONSUME(LRAW, ID)                                          \
    do {                                                                       \
        const float _l = (LRAW) * inv_t;                                       \
        if (_l > lmax) { lsum = lsum * __expf(lmax - _l) + 1.0f; lmax = _l; }  \
        else           { lsum += __expf(_l - lmax); }                          \
        if (_l > topv[K - 1]) {                                                \
            topv[K - 1] = _l; topi[K - 1] = (ID);                              \
            _Pragma("unroll")                                                  \
            for (int _p = K - 1; _p > 0; --_p) {                               \
                if (topv[_p] > topv[_p - 1]) {                                 \
                    const float _tv = topv[_p];                                \
                    topv[_p] = topv[_p - 1]; topv[_p - 1] = _tv;               \
                    const int _ti = topi[_p];                                  \
                    topi[_p] = topi[_p - 1]; topi[_p - 1] = _ti;               \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } while (0)

// ---- pass 1: per-(position, split) partial logsumexp + local top-K ---------
// Grid: (n_positions, split). Block s of row `row` scans its contiguous vocab
// chunk and writes one partial (max, sum, sorted top-K) to part_* at index
// row*split + s. Contiguous chunks keep the strided per-warp reads coalesced.
// VEC selects 16-byte float4 loads (one coalesced transaction per 4 logits) —
// only used when every row's base pointer is 16-byte aligned (vocab % 4 == 0
// and an aligned tensor); otherwise the scalar path is used.
template <int K, bool VEC>
__global__ void geometric_draft_topk_partial(const float * __restrict__ logits,
                                   int vocab, float inv_t, int split,
                                   float * __restrict__ part_max,
                                   float * __restrict__ part_sum,
                                   float * __restrict__ part_v,
                                   int32_t * __restrict__ part_i) {
    const int row = blockIdx.x;
    const int s   = blockIdx.y;
    const int tid = threadIdx.x;
    const float * __restrict__ li = logits + (size_t)row * vocab;

    float lmax = -FLT_MAX;
    float lsum = 0.0f;
    float topv[K];
    int   topi[K];
#pragma unroll
    for (int k = 0; k < K; k++) { topv[k] = -FLT_MAX; topi[k] = -1; }

    if (VEC) {
        // Partition the float4s of the row into `split` contiguous chunks.
        const int vocab4 = vocab >> 2;                   // # of full float4s
        const int chunk4 = (vocab4 + split - 1) / split;
        const int b4 = s * chunk4;
        const int e4 = min(b4 + chunk4, vocab4);
        const float4 * __restrict__ li4 = reinterpret_cast<const float4 *>(li);
        for (int j4 = b4 + tid; j4 < e4; j4 += kBlock) {
            const float4 f = li4[j4];
            const int base = j4 << 2;
            DFLASH_TOPK_CONSUME(f.x, base + 0);
            DFLASH_TOPK_CONSUME(f.y, base + 1);
            DFLASH_TOPK_CONSUME(f.z, base + 2);
            DFLASH_TOPK_CONSUME(f.w, base + 3);
        }
        // Tail elements past the last full float4 (only when vocab % 4 != 0);
        // the last split owns them so no id is scanned twice.
        if (s == split - 1) {
            for (int j = (vocab4 << 2) + tid; j < vocab; j += kBlock)
                DFLASH_TOPK_CONSUME(li[j], j);
        }
    } else {
        const int chunk = (vocab + split - 1) / split;
        const int begin = s * chunk;
        const int end   = min(begin + chunk, vocab);
        for (int j = begin + tid; j < end; j += kBlock)
            DFLASH_TOPK_CONSUME(li[j], j);
    }

    // ---- block reduction over kBlock threads ------------------------------
    __shared__ float s_max[kBlock];
    __shared__ float s_sum[kBlock];
    __shared__ float s_topv[kBlock * K];
    __shared__ int32_t s_topi[kBlock * K];

    s_max[tid] = lmax;
    s_sum[tid] = lsum;
#pragma unroll
    for (int k = 0; k < K; k++) {
        s_topv[tid * K + k] = topv[k];
        s_topi[tid * K + k] = topi[k];
    }
    __syncthreads();

    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float am = s_max[tid],          as = s_sum[tid];
            const float bm = s_max[tid + stride], bs = s_sum[tid + stride];
            const float m = fmaxf(am, bm);
            s_sum[tid] = as * __expf(am - m) + bs * __expf(bm - m);
            s_max[tid] = m;

            float dv[K]; int di[K];
            merge_topk<K>(&s_topv[tid * K],            &s_topi[tid * K],
                          &s_topv[(tid + stride) * K], &s_topi[(tid + stride) * K],
                          dv, di);
#pragma unroll
            for (int k = 0; k < K; k++) {
                s_topv[tid * K + k] = dv[k];
                s_topi[tid * K + k] = di[k];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const int idx = row * split + s;
        part_max[idx] = s_max[0];
        part_sum[idx] = s_sum[0];
#pragma unroll
        for (int k = 0; k < K; k++) {
            part_v[(size_t)idx * K + k] = s_topv[k];
            part_i[(size_t)idx * K + k] = s_topi[k];
        }
    }
}

#undef DFLASH_TOPK_CONSUME

// ---- pass 2: merge the `split` partials per position into the final top-K --
// Grid: n_positions blocks of blockDim = pow2_ceil(split) threads. Thread t
// loads partial t (or an identity when t >= split), then a shared-memory tree
// reduction merges all partials. log_prob[k] = top_v[k] - log_z.
template <int K>
__global__ void geometric_draft_topk_combine(const float * __restrict__ part_max,
                                   const float * __restrict__ part_sum,
                                   const float * __restrict__ part_v,
                                   const int32_t * __restrict__ part_i,
                                   int split,
                                   float * __restrict__ out_lp,
                                   int32_t * __restrict__ out_ids) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;     // blockDim is pow2_ceil(split)

    __shared__ float s_max[kMaxSplit];
    __shared__ float s_sum[kMaxSplit];
    __shared__ float s_topv[kMaxSplit * K];
    __shared__ int32_t s_topi[kMaxSplit * K];

    if (tid < split) {
        const int idx = row * split + tid;
        s_max[tid] = part_max[idx];
        s_sum[tid] = part_sum[idx];
#pragma unroll
        for (int k = 0; k < K; k++) {
            s_topv[tid * K + k] = part_v[(size_t)idx * K + k];
            s_topi[tid * K + k] = part_i[(size_t)idx * K + k];
        }
    } else {
        s_max[tid] = -FLT_MAX;
        s_sum[tid] = 0.0f;
#pragma unroll
        for (int k = 0; k < K; k++) {
            s_topv[tid * K + k] = -FLT_MAX;
            s_topi[tid * K + k] = -1;
        }
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float am = s_max[tid],          as = s_sum[tid];
            const float bm = s_max[tid + stride], bs = s_sum[tid + stride];
            const float m = fmaxf(am, bm);
            s_sum[tid] = as * __expf(am - m) + bs * __expf(bm - m);
            s_max[tid] = m;

            float dv[K]; int di[K];
            merge_topk<K>(&s_topv[tid * K],            &s_topi[tid * K],
                          &s_topv[(tid + stride) * K], &s_topi[(tid + stride) * K],
                          dv, di);
#pragma unroll
            for (int k = 0; k < K; k++) {
                s_topv[tid * K + k] = dv[k];
                s_topi[tid * K + k] = di[k];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float log_z = s_max[0] + logf(s_sum[0]);
#pragma unroll
        for (int k = 0; k < K; k++) {
            out_lp[(size_t)row * K + k]  = s_topv[k] - log_z;
            out_ids[(size_t)row * K + k] = s_topi[k];
        }
    }
}

// Per-device scratch for the [n_positions × K] outputs plus the
// [n_positions × split × K] pass-1 partials, grown as needed. The decode loop
// is single-threaded, so a plain static cache is safe and avoids a
// cudaMalloc/cudaFree on every step.
struct Scratch {
    int       device   = -1;
    size_t    cap       = 0;       // output elements (n_positions*K)
    size_t    part_cap  = 0;       // partial-list elements (n_positions*split*kMaxK)
    float *   d_lp      = nullptr;
    int32_t * d_ids     = nullptr;
    float *   d_pmax    = nullptr; // [n_positions*split]
    float *   d_psum    = nullptr; // [n_positions*split]
    float *   d_pv      = nullptr; // [n_positions*split*kMaxK]
    int32_t * d_pi      = nullptr; // [n_positions*split*kMaxK]
};
Scratch g_scratch;

void free_scratch() {
    if (g_scratch.d_lp)   cudaFree(g_scratch.d_lp);
    if (g_scratch.d_ids)  cudaFree(g_scratch.d_ids);
    if (g_scratch.d_pmax) cudaFree(g_scratch.d_pmax);
    if (g_scratch.d_psum) cudaFree(g_scratch.d_psum);
    if (g_scratch.d_pv)   cudaFree(g_scratch.d_pv);
    if (g_scratch.d_pi)   cudaFree(g_scratch.d_pi);
    g_scratch = Scratch{};
}

// Allocate output + partial buffers. n = n_positions*K outputs;
// n_parts = n_positions*split partials (each carrying K entries).
bool ensure_scratch(int device, size_t n, size_t n_parts) {
    const size_t n_part_lists = n_parts * (size_t)kMaxK;  // upper bound on K
    if (g_scratch.device == device && g_scratch.cap >= n &&
        g_scratch.part_cap >= n_part_lists)
        return true;
    free_scratch();
    if (cudaMalloc(&g_scratch.d_lp,   n            * sizeof(float))   != cudaSuccess) goto fail;
    if (cudaMalloc(&g_scratch.d_ids,  n            * sizeof(int32_t)) != cudaSuccess) goto fail;
    if (cudaMalloc(&g_scratch.d_pmax, n_parts      * sizeof(float))   != cudaSuccess) goto fail;
    if (cudaMalloc(&g_scratch.d_psum, n_parts      * sizeof(float))   != cudaSuccess) goto fail;
    if (cudaMalloc(&g_scratch.d_pv,   n_part_lists * sizeof(float))   != cudaSuccess) goto fail;
    if (cudaMalloc(&g_scratch.d_pi,   n_part_lists * sizeof(int32_t)) != cudaSuccess) goto fail;
    g_scratch.device   = device;
    g_scratch.cap      = n;
    g_scratch.part_cap = n_part_lists;
    return true;
fail:
    free_scratch();
    return false;
}

inline int pow2_ceil(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

// Choose how many blocks to split each position's vocab scan across. The work
// is bandwidth bound, so we want enough total blocks (n_positions × split) to
// saturate the device while keeping each chunk large enough that the strided
// reads stay coalesced and per-block overhead stays amortized.
int pick_split(int vocab, int n_positions) {
    if (const char * v = std::getenv("DFLASH_TOPK_SPLIT")) {
        int s = std::atoi(v);
        if (s >= 1 && s <= kMaxSplit) return s;
    }
    // Aim for ~240 total blocks (≈3 waves on a ~80-SM device, measured sweet
    // spot for vocab~150k), but cap the chunk floor at ~2k elements so a block
    // never scans too little to be worth launching.
    int by_blocks = (240 + n_positions - 1) / n_positions;
    int by_chunk  = vocab / 2048;
    int split = by_blocks < by_chunk ? by_blocks : by_chunk;
    if (split < 1)         split = 1;
    if (split > kMaxSplit) split = kMaxSplit;
    return split;
}

}  // namespace

bool geometric_extract_draft_topk_cuda(const void * d_logits,
                             int n_positions, int vocab, int K,
                             float * out_log_probs,
                             int32_t * out_token_ids,
                             float temperature) {
    if (!d_logits || n_positions <= 0 || vocab <= 0 || K <= 0 || K > kMaxK) return false;

    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, d_logits) != cudaSuccess) {
        cudaGetLastError();  // clear the error so we don't poison the next CUDA call
        return false;
    }
    if (attr.type != cudaMemoryTypeDevice) return false;

    int prev = 0;
    cudaGetDevice(&prev);
    const int dev = attr.device;
    if (dev != prev) cudaSetDevice(dev);

    static const bool kProfile = std::getenv("DFLASH_TOPK_PROFILE") != nullptr;
    bool ok = false;
    const int    split   = pick_split(vocab, n_positions);
    const size_t n       = (size_t)n_positions * K;
    const size_t n_parts = (size_t)n_positions * split;
    if (ensure_scratch(dev, n, n_parts)) {
        const float inv_t = 1.0f / fmaxf(1e-3f, temperature);
        cudaEvent_t e_k0, e_k1, e_c1;
        if (kProfile) { cudaEventCreate(&e_k0); cudaEventCreate(&e_k1); cudaEventCreate(&e_c1); cudaEventRecord(e_k0); }

        const dim3 grid1(n_positions, split);
        const int  comb_block = pow2_ceil(split);
        const float * lp_in = static_cast<const float *>(d_logits);
        // float4 loads are safe only when every row base is 16-byte aligned:
        // the tensor base aligned and a vocab stride that is a multiple of 4.
        const bool use_vec = (vocab % 4 == 0) &&
                             (reinterpret_cast<uintptr_t>(lp_in) % 16 == 0);
        // K (and the vectorization flag) are compile-time template parameters
        // so the per-thread/per-partial top-K stays register-resident; dispatch
        // the runtime K to its instantiation. K>kMaxK is already rejected above.
#define DFLASH_TOPK_LAUNCH(KV, VEC)                                                             \
            geometric_draft_topk_partial<KV, VEC><<<grid1, kBlock>>>(                                     \
                lp_in, vocab, inv_t, split,                                                     \
                g_scratch.d_pmax, g_scratch.d_psum, g_scratch.d_pv, g_scratch.d_pi);           \
            geometric_draft_topk_combine<KV><<<n_positions, comb_block>>>(                                \
                g_scratch.d_pmax, g_scratch.d_psum, g_scratch.d_pv, g_scratch.d_pi,            \
                split, g_scratch.d_lp, g_scratch.d_ids);
#define DFLASH_TOPK_CASE(KV)                                                                    \
            case KV:                                                                            \
                if (use_vec) { DFLASH_TOPK_LAUNCH(KV, true) }                                   \
                else         { DFLASH_TOPK_LAUNCH(KV, false) }                                  \
                break;
        switch (K) {
            DFLASH_TOPK_CASE(1) DFLASH_TOPK_CASE(2) DFLASH_TOPK_CASE(3) DFLASH_TOPK_CASE(4)
            DFLASH_TOPK_CASE(5) DFLASH_TOPK_CASE(6) DFLASH_TOPK_CASE(7) DFLASH_TOPK_CASE(8)
            default: break;
        }
#undef DFLASH_TOPK_CASE
#undef DFLASH_TOPK_LAUNCH

        if (kProfile) cudaEventRecord(e_k1);
        if (cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess) {
            const cudaError_t e1 = cudaMemcpy(out_log_probs, g_scratch.d_lp,
                                              n * sizeof(float), cudaMemcpyDeviceToHost);
            const cudaError_t e2 = cudaMemcpy(out_token_ids, g_scratch.d_ids,
                                              n * sizeof(int32_t), cudaMemcpyDeviceToHost);
            ok = (e1 == cudaSuccess && e2 == cudaSuccess);
        }
        if (kProfile) {
            cudaEventRecord(e_c1); cudaEventSynchronize(e_c1);
            float k_ms = 0, c_ms = 0;
            cudaEventElapsedTime(&k_ms, e_k0, e_k1);
            cudaEventElapsedTime(&c_ms, e_k1, e_c1);
            std::fprintf(stderr, "[topk] kernels=%.3f ms  sync+copy=%.3f ms (n_pos=%d vocab=%d split=%d)\n",
                         k_ms, c_ms, n_positions, vocab, split);
            cudaEventDestroy(e_k0); cudaEventDestroy(e_k1); cudaEventDestroy(e_c1);
        }
    }
    if (!ok) cudaGetLastError();
    if (dev != prev) cudaSetDevice(prev);
    return ok;
}

}  // namespace dflash::common
