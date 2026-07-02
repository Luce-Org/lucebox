// CUDA port of the sample_logits chain. See geometric_sampler_cuda.h for the contract and
// the CPU/GPU split rationale, and sampler.cpp for the reference CPU chain this
// mirrors: rep_penalty -> freq/pres_penalty -> softmax(temp) -> draw.
// top_p (nucleus) is not implemented here — cfg.top_p in (0,1) falls back to
// the CPU chain (see geometric_sampler_cuda.h).
//
// The per-call workload is one logit row (vocab ~150k). That is small enough
// that a single thread block handles the whole row: it keeps every reduction
// and the inverse-CDF scan in shared memory with no cross-block
// synchronization, which is far simpler and — for one row per token — fast
// enough. (The bandwidth-bound multi-block split used by
// geometric_draft_topk_cuda.cu pays off only for many rows at once.)
//
// A few structural choices below are lifted from ggml-cuda's softmax kernel
// (ggml/src/ggml-cuda/softmax.cu):
//   * penalty application, greedy argmax, temp-sample draw, and (when
//     top_k/top_p force CPU-side truncation) raw probability emission are all
//     one kernel (geometric_sample_kernel, mode-selected) instead of separate
//     penalty/softmax kernels — pass 0 (penalties) and pass 1 (max/argmax)
//     are identical work regardless of what happens after, so duplicating
//     them across kernels bought nothing;
//   * penalty application is fused into pass 0 of that kernel instead of a
//     separate launch — the penalty set is tiny (<= rep_window ids), so
//     folding it in costs a few extra loop iterations for a handful of
//     threads, not a new pass over the vocab, and it removes a whole kernel
//     launch + sync boundary per token;
//   * block-wide max/sum/argmax reductions use warp-shuffle (__shfl_xor_sync)
//     first, falling back to a small (<=32-slot) shared-memory reduction only
//     across warps, mirroring ggml's warp_reduce_*/block reduction idiom.
//     This cuts block-wide __syncthreads() calls per reduction from
//     log2(nthreads) (10, for a 1024-thread block) down to a handful, and
//     drops the per-thread argmax-index shared array entirely (moved to a
//     fixed 32-slot static array), shrinking the dynamic shared-memory
//     footprint below.

#include "geometric_sampler_cuda.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <unordered_map>

namespace dflash::common {

namespace {

constexpr int kWarpSize = 32;

// Warp-level reductions via register shuffles — no shared memory, no
// __syncthreads(). Mirrors ggml-cuda's warp_reduce_sum/warp_reduce_max
// (ggml/src/ggml-cuda/common.cuh); we only need sum and argmax variants here
// (the kernel below always tracks argmax alongside max — see its pass 1).
__device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, off, kWarpSize);
    return x;
}
__device__ __forceinline__ void warp_reduce_argmax(float & val, int & idx) {
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, val, off, kWarpSize);
        const int   oi = __shfl_xor_sync(0xffffffffu, idx, off, kWarpSize);
        if (ov > val || (ov == val && oi < idx)) { val = ov; idx = oi; }
    }
}

// Block-wide reductions: each warp reduces internally via the shuffles above
// (no sync), lane 0 of each warp parks its partial in a small scratch array
// (one slot per warp, so <=32 slots for any block <=1024 threads), then the
// first warp finishes the reduction over those partials and broadcasts the
// result back through the same scratch slot. `*_scratch` must have >=32
// (or, for argmax, 32 float + 32 int) entries; callers pass a fixed-size
// __shared__ array so this needs no dynamic-shared-memory sizing. The
// trailing sync (after every thread has read the broadcast value) guards
// against a subsequent block_reduce_* call reusing the same scratch slots
// before all threads are done reading this one.
__device__ __forceinline__ float block_reduce_sum(float x, float * scratch) {
    const int lane   = threadIdx.x & (kWarpSize - 1);
    const int wid    = threadIdx.x / kWarpSize;
    const int nwarps = (blockDim.x + kWarpSize - 1) / kWarpSize;
    x = warp_reduce_sum(x);
    if (lane == 0) scratch[wid] = x;
    __syncthreads();
    x = (threadIdx.x < nwarps) ? scratch[threadIdx.x] : 0.0f;
    if (wid == 0) x = warp_reduce_sum(x);
    if (threadIdx.x == 0) scratch[0] = x;
    __syncthreads();
    const float result = scratch[0];
    __syncthreads();
    return result;
}
__device__ __forceinline__ void block_reduce_argmax(float & val, int & idx,
                                       float * val_scratch, int * idx_scratch) {
    const int lane   = threadIdx.x & (kWarpSize - 1);
    const int wid    = threadIdx.x / kWarpSize;
    const int nwarps = (blockDim.x + kWarpSize - 1) / kWarpSize;
    warp_reduce_argmax(val, idx);
    if (lane == 0) { val_scratch[wid] = val; idx_scratch[wid] = idx; }
    __syncthreads();
    if (threadIdx.x < nwarps) { val = val_scratch[threadIdx.x]; idx = idx_scratch[threadIdx.x]; }
    else                      { val = -FLT_MAX; idx = INT_MAX; }
    if (wid == 0) warp_reduce_argmax(val, idx);
    if (threadIdx.x == 0) { val_scratch[0] = val; idx_scratch[0] = idx; }
    __syncthreads();
    val = val_scratch[0];
    idx = idx_scratch[0];
    __syncthreads();
}

// geometric_sample_kernel's per-thread shared-memory footprint: two float
// arrays (sh, s_off), each sized [blockDim.x], used only by the inverse-CDF
// draw (each thread's chunk mass, then its exclusive prefix offset — a true
// per-thread array, not reducible to a scalar). The max/argmax/sum
// reductions above no longer need a per-thread shared array (they use the
// fixed 32-slot statics), so this is 2 floats/thread instead of the previous
// 2 floats + 1 int32. float, not double: the reductions sum ~vocab
// already-float terms (softmax probabilities in (0,1]), and float32 error is
// negligible for a stochastic sampling draw — while avoiding double-precision
// arithmetic, which on consumer GPUs (e.g. RTX 3090) runs at a small fraction
// of float32 throughput.
constexpr size_t kSampleShmemPerThread = 2 * sizeof(float);

// Per-device sample-kernel block size (threads per row), cached after the
// first call — same single-threaded-decode-loop assumption as Scratch below.
// Queried rather than hardcoded because maxThreadsPerBlock and
// sharedMemPerBlock vary across GPUs (e.g. older compute-capability parts cap
// at 512 threads and/or less shared memory than the 1024-thread/8-byte-per-
// thread shape that fits an RTX 3090).
struct BlockCfg {
    int device = -1;
    int block  = 0;
};
BlockCfg g_block_cfg;

int pick_block_size(int device) {
    if (g_block_cfg.device == device) return g_block_cfg.block;
    int    max_threads = 1024;
    size_t smem_cap    = 48 * 1024;  // conservative fallback if the query fails
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        max_threads = prop.maxThreadsPerBlock;
        smem_cap    = prop.sharedMemPerBlock;
    } else {
        cudaGetLastError();
    }
    int block = 1;
    while (block * 2 <= max_threads) block *= 2;  // largest power of two <= max_threads
    // Halve further if the reductions' shared-memory arrays wouldn't fit
    // (stays a power of two so the block-reduction loop below still works).
    while (block > 32 && (size_t)block * kSampleShmemPerThread > smem_cap) block /= 2;
    g_block_cfg = {device, block};
    return block;
}

// Applies the (already-uploaded) sparse penalty set to `work[]` in place —
// pass 0 of geometric_sample_kernel below (shared by all three modes).
// `m` is tiny (<= cfg.rep_window token ids), so a grid-stride loop over it
// costs a handful of extra iterations for a handful of threads, not a new
// pass over the vocab; folding it in here removes what used to be a separate
// geometric_apply_penalties launch + sync boundary before every sample/probs
// call. `rep_pen` is the raw multiplicative penalty (or 1.0 to disable);
// `add[i]` folds freq_pen*count + pres_pen for pen_ids[i]. Order matches the
// CPU chain: multiplicative repetition penalty first, then the additive
// frequency/presence subtraction. Callers must __syncthreads() after this
// (only if m>0) before any thread reads work[] for the reductions below, so
// every thread's penalty writes are visible.
__device__ __forceinline__ void apply_penalties_inplace(float * __restrict__ work,
                                const int32_t * __restrict__ pen_ids,
                                const float * __restrict__ pen_add,
                                int m, float rep_pen, int rep_active) {
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        const int id = pen_ids[i];
        float v = work[id];
        if (rep_active) v = (v > 0.0f) ? v / rep_pen : v * rep_pen;
        v -= pen_add[i];
        work[id] = v;
    }
}

// geometric_sample_kernel's three mutually-exclusive endings:
//   kModeGreedy    - write the argmax token id, nothing else (temp<=0).
//   kModeSample    - draw one token via inverse-CDF and write its id.
//   kModeEmitProbs - write the full normalized probability vector (used when
//                    top_k/top_p forces CPU-side truncation, so the CPU
//                    doesn't need to redo penalties+softmax itself).
// All three share pass 0 (penalty fusion) and pass 1 (max/argmax) below;
// kModeSample and kModeEmitProbs also share pass 2 (the Z reduction). Only
// kModeGreedy and kModeEmitProbs skip the inverse-CDF draw's per-thread `sh`/
// `s_off` array, so only kModeSample launches need to size dynamic shared
// memory for it (see the call sites) — kModeGreedy/kModeEmitProbs never touch
// `geometric_smem` and can launch with 0 dynamic shared memory. Note
// out_probs is written ONLY in kModeEmitProbs: the greedy/sample paths never
// materialize the full probability vector in global memory at all, since the
// draw happens entirely from registers/shared memory inside the kernel.
constexpr int kModeGreedy    = 0;
constexpr int kModeSample    = 1;
constexpr int kModeEmitProbs = 2;

// Dynamic shared memory for geometric_sample_kernel's kModeSample path,
// sized at launch to blockDim.x * kSampleShmemPerThread (see
// pick_block_size): two float arrays (sh, s_off), used only by the
// inverse-CDF draw (the block-wide reductions use the fixed 32-slot statics
// declared inside the kernel).
extern __shared__ unsigned char geometric_smem[];

// Single-block sampler/softmax kernel. `work[]` holds the pre-penalty logits
// (penalties are applied in place as pass 0, in-kernel — see
// apply_penalties_inplace). Behaviour selected by `mode` (see above); ties in
// the greedy argmax go to the lowest token id, matching the CPU manual-argmax
// and DFLASH_GPU_ARGMAX behaviour.
__global__ void geometric_sample_kernel(float * __restrict__ work, int vocab,
                              float inv_t, int mode,
                              double r_uniform,
                              int32_t * __restrict__ out_token,
                              float * __restrict__ out_probs,
                              const int32_t * __restrict__ pen_ids,
                              const float * __restrict__ pen_add,
                              int pen_m, float rep_pen, int rep_active) {
    const int t        = threadIdx.x;
    const int nthreads = blockDim.x;

    apply_penalties_inplace(work, pen_ids, pen_add, pen_m, rep_pen, rep_active);
    if (pen_m > 0) __syncthreads();

    // Vectorized, coalesced grid-stride layout over the row. Every O(vocab)
    // pass below walks `work` as float4: thread t reads groups v = t, t+nthreads,
    // ... so consecutive threads touch consecutive 16-byte groups (a coalesced
    // warp transaction).
    // `work` is g_scratch.d_work, a cudaMalloc allocation (>=256-byte aligned),
    // so the float4 reinterpret is safe. `nvec` full groups cover [0, tail); the
    // <=3 leftover ids in [tail, vocab) are handled by a scalar grid-stride tail.
    // Within a thread, group ids ascend (v grows, and x<y<z<w within a group)
    // and the tail's ids are all >= tail, so the strict-'>' argmax below keeps
    // the lowest id on ties without an explicit index compare.
    const int      nvec = vocab >> 2;
    const int      tail = nvec << 2;
    const float4 * __restrict__ w4 = reinterpret_cast<const float4 *>(work);

    float * sh    = reinterpret_cast<float *>(geometric_smem);
    float * s_off = sh + nthreads;
    __shared__ float s_val_scratch[kWarpSize];
    __shared__ int   s_idx_scratch[kWarpSize];

    // ---- pass 1: max logit + argmax (lowest id on ties), warp-shuffle first,
    // falling back to the <=32-slot statics only across warps (see
    // block_reduce_argmax) — far fewer __syncthreads() than a full
    // shared-memory tree reduction over all nthreads slots. Computed in every
    // mode (kModeEmitProbs doesn't need the index, but the extra shuffle over
    // an int alongside the float is negligible next to the O(vocab) scan
    // above it, and keeping one shared pass 1 avoids a second reduction
    // helper just to drop the index).
    float lmax    = -FLT_MAX;
    int   largmax = vocab;  // sentinel so a real id always wins
    for (int v = t; v < nvec; v += nthreads) {
        const float4 f    = w4[v];
        const int    base = v << 2;
        if (f.x > lmax) { lmax = f.x; largmax = base;     }
        if (f.y > lmax) { lmax = f.y; largmax = base + 1; }
        if (f.z > lmax) { lmax = f.z; largmax = base + 2; }
        if (f.w > lmax) { lmax = f.w; largmax = base + 3; }
    }
    for (int i = tail + t; i < vocab; i += nthreads) {
        const float v = work[i];
        if (v > lmax) { lmax = v; largmax = i; }
    }
    block_reduce_argmax(lmax, largmax, s_val_scratch, s_idx_scratch);
    const float xmax   = lmax * inv_t;
    const int   argmax = largmax;

    if (mode == kModeGreedy) {
        if (t == 0) *out_token = argmax;
        return;
    }

    // ---- pass 2: softmax denominator Z = sum exp(x_i - xmax) --------------
    // float throughout, not double: expf() (float) is ~19x faster than exp()
    // (double) on consumer GPUs (e.g. RTX 3090), where FP64 throughput is a
    // small fraction of FP32. The sums are ~vocab terms in (0,1], and float32
    // error is negligible for a stochastic sampling draw.
    float lz = 0.0f;
    for (int v = t; v < nvec; v += nthreads) {
        const float4 f = w4[v];
        lz += expf(f.x * inv_t - xmax) + expf(f.y * inv_t - xmax)
            + expf(f.z * inv_t - xmax) + expf(f.w * inv_t - xmax);
    }
    for (int i = tail + t; i < vocab; i += nthreads)
        lz += expf(work[i] * inv_t - xmax);
    const float Z = block_reduce_sum(lz, s_val_scratch);

    if (mode == kModeEmitProbs) {
        float4 * __restrict__ o4 = reinterpret_cast<float4 *>(out_probs);
        const float inv_z = 1.0f / Z;
        for (int v = t; v < nvec; v += nthreads) {
            const float4 f = w4[v];
            float4 o;
            o.x = expf(f.x * inv_t - xmax) * inv_z;
            o.y = expf(f.y * inv_t - xmax) * inv_z;
            o.z = expf(f.z * inv_t - xmax) * inv_z;
            o.w = expf(f.w * inv_t - xmax) * inv_z;
            o4[v] = o;
        }
        for (int i = tail + t; i < vocab; i += nthreads)
            out_probs[i] = expf(work[i] * inv_t - xmax) * inv_z;
        return;
    }

    // ---- kModeSample: multinomial inverse-CDF draw over the full distribution
    // The cumulative distribution is laid out as thread 0's ids (in its
    // grid-stride order), then thread 1's, ... — a fixed permutation of the pmf,
    // which is all inverse-CDF sampling needs (each id still occupies an interval
    // of width p_id, so target ~ U(0,Z) lands in it with probability p_id/Z; the
    // GPU draw need not be bit-identical to the CPU's contiguous ordering — see
    // test_gpu_sampler_cuda.cpp). A serial exclusive scan over the nthreads
    // per-thread masses gives each thread its CDF offset; the one whose
    // [offset, offset+mass) straddles the target re-scans its own ids, in the
    // same grid-stride order, for the crossing id. (This per-thread array can't
    // be collapsed into the warp-shuffle reductions above — thread 0 needs every
    // individual chunk mass, not just their sum, to compute the prefix offsets.)
    // Each thread's mass is exactly the partial sum it already accumulated for Z
    // in pass 2 — block_reduce_sum takes `lz` by value, so it is still intact
    // here. Reuse it instead of a second O(vocab) expf pass over the row.
    const float pm = lz;
    sh[t] = pm;
    __syncthreads();

    // Thread 0: seed the safety default (used only if fp rounding leaves no
    // straddling thread) and compute the exclusive prefix offsets.
    if (t == 0) {
        *out_token = argmax;
        float acc = 0.0f;
        for (int k = 0; k < nthreads; k++) { s_off[k] = acc; acc += sh[k]; }
    }
    __syncthreads();

    // r_uniform is a host-drawn double; keep this comparison in double (a
    // single scalar op, not a reduction, so it doesn't cost the FP64 penalty).
    const double targetv = r_uniform * (double)Z;
    if (targetv >= (double)s_off[t] && targetv < (double)s_off[t] + pm) {
        float acc = s_off[t];
        for (int v = t; v < nvec; v += nthreads) {
            const float4 f    = w4[v];
            const int    base = v << 2;
            acc += expf(f.x * inv_t - xmax); if (targetv < acc) { *out_token = base;     goto done; }
            acc += expf(f.y * inv_t - xmax); if (targetv < acc) { *out_token = base + 1; goto done; }
            acc += expf(f.z * inv_t - xmax); if (targetv < acc) { *out_token = base + 2; goto done; }
            acc += expf(f.w * inv_t - xmax); if (targetv < acc) { *out_token = base + 3; goto done; }
        }
        for (int i = tail + t; i < vocab; i += nthreads) {
            acc += expf(work[i] * inv_t - xmax);
            if (targetv < acc) { *out_token = i; goto done; }
        }
        done:;
    }
}

// Per-device persistent scratch. The decode loop is single-threaded, so a plain
// static cache avoids a cudaMalloc/cudaFree per token (mirrors geometric_draft_topk_cuda).
struct Scratch {
    int       device   = -1;
    int       vocab_cap = 0;
    int       pen_cap  = 0;
    float *   d_work   = nullptr;  // [vocab] mutable logit copy
    float *   d_probs  = nullptr;  // [vocab] softmax output (geometric_compute_probs_cuda)
    int32_t * d_pen_id = nullptr;  // [pen_cap]
    float *   d_pen_add = nullptr; // [pen_cap]
    int32_t * d_out    = nullptr;  // [1]
};
Scratch g_scratch;

void free_scratch() {
    if (g_scratch.d_work)   cudaFree(g_scratch.d_work);
    if (g_scratch.d_probs)  cudaFree(g_scratch.d_probs);
    if (g_scratch.d_pen_id) cudaFree(g_scratch.d_pen_id);
    if (g_scratch.d_pen_add) cudaFree(g_scratch.d_pen_add);
    if (g_scratch.d_out)    cudaFree(g_scratch.d_out);
    g_scratch = Scratch{};
}

bool ensure_scratch(int device, int vocab, int pen) {
    const bool ok = g_scratch.device == device &&
                    g_scratch.vocab_cap >= vocab &&
                    g_scratch.pen_cap >= pen;
    if (ok) return true;
    free_scratch();
    if (cudaMalloc(&g_scratch.d_out, sizeof(int32_t)) != cudaSuccess) goto fail;
    if (cudaMalloc(&g_scratch.d_work, (size_t)vocab * sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMalloc(&g_scratch.d_probs, (size_t)vocab * sizeof(float)) != cudaSuccess) goto fail;
    if (pen > 0) {
        if (cudaMalloc(&g_scratch.d_pen_id, (size_t)pen * sizeof(int32_t)) != cudaSuccess) goto fail;
        if (cudaMalloc(&g_scratch.d_pen_add, (size_t)pen * sizeof(float)) != cudaSuccess) goto fail;
    }
    g_scratch.device    = device;
    g_scratch.vocab_cap = vocab;
    g_scratch.pen_cap   = pen;
    return true;
fail:
    free_scratch();
    return false;
}

// Uploads `logits` (host or device) into g_scratch.d_work and the sparse
// penalty set into g_scratch.d_pen_id/d_pen_add — the shared prefix of
// geometric_sample_logits_cuda and geometric_compute_probs_cuda. Penalty
// *application* itself is no longer done here: it's fused into pass 0 of
// geometric_sample_kernel (see apply_penalties_inplace), so this function
// only stages data, no kernel launch. Writes the penalty count to `*out_m`.
// Assumes `dev` is already the current device. Returns false on any CUDA
// allocation/copy error.
bool stage_and_penalize(int dev, const float * logits, int vocab, const SamplerCfg & cfg,
               const std::vector<int32_t> & history, bool logits_on_device, int * out_m) {
    std::vector<int32_t> pen_id;
    std::vector<float>   pen_add;
    const bool rep_active  = cfg.rep_pen > 1.0f;
    const bool add_active  = (cfg.freq_pen != 0.0f || cfg.pres_pen != 0.0f);
    if ((rep_active || add_active) && !history.empty()) {
        const int win  = std::min((int)history.size(), cfg.rep_window);
        const int from = (int)history.size() - win;
        std::unordered_map<int, int> counts;
        for (int i = from; i < (int)history.size(); i++) counts[history[i]]++;
        pen_id.reserve(counts.size());
        pen_add.reserve(counts.size());
        for (const auto & kv : counts) {
            if (kv.first < 0 || kv.first >= vocab) continue;
            pen_id.push_back(kv.first);
            pen_add.push_back(add_active ? (cfg.freq_pen * kv.second + cfg.pres_pen) : 0.0f);
        }
    }
    const int m = (int)pen_id.size();

    if (!ensure_scratch(dev, vocab, m)) return false;
    if (cudaMemcpy(g_scratch.d_work, logits, (size_t)vocab * sizeof(float),
                   logits_on_device ? cudaMemcpyDeviceToDevice
                                    : cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    if (m > 0) {
        if (cudaMemcpy(g_scratch.d_pen_id, pen_id.data(), (size_t)m * sizeof(int32_t),
                       cudaMemcpyHostToDevice) != cudaSuccess) return false;
        if (cudaMemcpy(g_scratch.d_pen_add, pen_add.data(), (size_t)m * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) return false;
    }
    *out_m = m;
    return true;
}

}  // namespace

bool gpu_sampler_enabled() {
    static const bool on = []() {
        const char * v = std::getenv("DFLASH_GPU_SAMPLE");
        if (v == nullptr || v[0] == '\0') return true;  // on by default
        return v[0] != '0';                             // "0" (or "0...") opts out
    }();
    return on;
}

int geometric_sample_logits_cuda(const float * logits,
                       int vocab,
                       const SamplerCfg & cfg,
                       const std::vector<int32_t> & history,
                       double r_uniform,
                       bool logits_on_device) {
    if (!logits || vocab <= 0) return -1;
    // top_k stays on the CPU (a single-row partial_sort beats a per-token GPU
    // select); signal fallback.
    if (cfg.top_k > 0 && cfg.top_k < vocab) return -1;
    // top_p (nucleus) is not implemented on this kernel; signal fallback (see
    // geometric_sampler_cuda.h for why).
    if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) return -1;

    // Pick the device. For a device pointer, derive it from the allocation so we
    // run where the logits live; otherwise use the current device.
    int dev = 0;
    if (logits_on_device) {
        cudaPointerAttributes attr{};
        if (cudaPointerGetAttributes(&attr, logits) != cudaSuccess) { cudaGetLastError(); return -1; }
        if (attr.type != cudaMemoryTypeDevice) return -1;
        dev = attr.device;
    } else {
        cudaGetDevice(&dev);
    }
    int prev = 0;
    cudaGetDevice(&prev);
    if (dev != prev) cudaSetDevice(dev);

    int result = -1;
    int m = 0;
    if (stage_and_penalize(dev, logits, vocab, cfg, history, logits_on_device, &m)) {
        const int    mode         = (cfg.temp > 0.0f) ? kModeSample : kModeGreedy;
        const float  inv_t        = 1.0f / fmaxf(1e-3f, cfg.temp);
        const int    block        = pick_block_size(dev);
        // Only kModeSample's inverse-CDF draw touches geometric_smem; skip
        // sizing it for the (cheaper, no-shared-mem) greedy launch.
        const size_t shmem_bytes  = (mode == kModeSample) ? (size_t)block * kSampleShmemPerThread : 0;
        geometric_sample_kernel<<<1, block, shmem_bytes>>>(g_scratch.d_work, vocab, inv_t, mode,
                                     r_uniform, g_scratch.d_out, /*out_probs=*/nullptr,
                                     g_scratch.d_pen_id, g_scratch.d_pen_add, m,
                                     cfg.rep_pen, cfg.rep_pen > 1.0f ? 1 : 0);
        int32_t tok = -1;
        if (cudaGetLastError() == cudaSuccess &&
            cudaMemcpy(&tok, g_scratch.d_out, sizeof(int32_t),
                       cudaMemcpyDeviceToHost) == cudaSuccess) {
            result = tok;
        }
    }
    if (result < 0) cudaGetLastError();  // clear so we don't poison the next call
    if (dev != prev) cudaSetDevice(prev);
    return result;
}

bool geometric_compute_probs_cuda(const float * logits,
                        int vocab,
                        const SamplerCfg & cfg,
                        const std::vector<int32_t> & history,
                        float * out_probs,
                        bool logits_on_device) {
    if (!logits || vocab <= 0 || !out_probs || cfg.temp <= 0.0f) return false;

    int dev = 0;
    if (logits_on_device) {
        cudaPointerAttributes attr{};
        if (cudaPointerGetAttributes(&attr, logits) != cudaSuccess) { cudaGetLastError(); return false; }
        if (attr.type != cudaMemoryTypeDevice) return false;
        dev = attr.device;
    } else {
        cudaGetDevice(&dev);
    }
    int prev = 0;
    cudaGetDevice(&prev);
    if (dev != prev) cudaSetDevice(dev);

    bool ok = false;
    int m = 0;
    if (stage_and_penalize(dev, logits, vocab, cfg, history, logits_on_device, &m)) {
        const float inv_t = 1.0f / fmaxf(1e-3f, cfg.temp);
        const int   block = pick_block_size(dev);
        // kModeEmitProbs never touches geometric_smem (no inverse-CDF draw),
        // so no dynamic shared memory needed.
        geometric_sample_kernel<<<1, block>>>(g_scratch.d_work, vocab, inv_t, kModeEmitProbs,
                                     /*r_uniform=*/0.0, /*out_token=*/nullptr, g_scratch.d_probs,
                                     g_scratch.d_pen_id, g_scratch.d_pen_add, m,
                                     cfg.rep_pen, cfg.rep_pen > 1.0f ? 1 : 0);
        if (cudaGetLastError() == cudaSuccess &&
            cudaMemcpy(out_probs, g_scratch.d_probs, (size_t)vocab * sizeof(float),
                       cudaMemcpyDeviceToHost) == cudaSuccess) {
            ok = true;
        }
    }
    if (!ok) cudaGetLastError();  // clear so we don't poison the next call
    if (dev != prev) cudaSetDevice(prev);
    return ok;
}

}  // namespace dflash::common
