#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(GGML_CUDA_USE_CUB)

static int topk_next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

template<typename T>
static inline __device__ void topk_swap(T & a, T & b) {
    T tmp = a;
    a     = b;
    b     = tmp;
}

// Warp-local xor shuffle. For the bitonic inner stages where the stride j is
// smaller than the wavefront width, both partners of a compare-exchange live in
// the same wave, so the exchange can be done register-to-register via shuffle —
// no shared-memory round-trip and no __syncthreads() barrier.
#if defined(GGML_USE_HIP) || defined(__HIP_PLATFORM_AMD__)
// On RDNA `__shfl_xor(v, j)` lowers to `ds_bpermute_b32`, which first computes a
// per-lane address VGPR (`(lane ^ j) << 2`) and reads through the LDS return
// path. For an xor mask that stays inside a 32-lane group (j < 32) the same
// permutation is expressible as a single `ds_swizzle_b32` in bitmask mode
// (offset[14:10]=xor, [9:5]=or, [4:0]=and; new lane = ((lane & and) | or) ^ xor):
// with and=0x1f, or=0, xor=j the source lane is exactly `lane ^ j`, so no
// address VGPR is built and the crosslane op is dropped straight onto the
// dependent shuffle chain. These partial-bitonic kernels are latency-bound on
// exactly those register-shuffle chains, so trimming the address-compute off
// each stage shortens the critical path. Every intra-wave use here has j in
// {1,2,4,8,16}; a wave64 top tail (j == 32) crosses the 32-lane group and falls
// back to `__shfl_xor`. Each case passes a literal so the builtin's
// immediate-operand requirement holds for any (even runtime) j. (The encoding
// was verified bit-identical to `__shfl_xor` on gfx1201 and gfx1151 for all five
// masks — test-backend-ops -o TOP_K passes on both.)
#    define TOPK_DS_SWIZZLE_XOR(m) (((m) << 10) | 0x1f)
static __device__ __forceinline__ int topk_shfl_xor_i32(int v, int mask) {
    switch (mask) {
        case 1:
            return __builtin_amdgcn_ds_swizzle(v, TOPK_DS_SWIZZLE_XOR(1));
        case 2:
            return __builtin_amdgcn_ds_swizzle(v, TOPK_DS_SWIZZLE_XOR(2));
        case 4:
            return __builtin_amdgcn_ds_swizzle(v, TOPK_DS_SWIZZLE_XOR(4));
        case 8:
            return __builtin_amdgcn_ds_swizzle(v, TOPK_DS_SWIZZLE_XOR(8));
        case 16:
            return __builtin_amdgcn_ds_swizzle(v, TOPK_DS_SWIZZLE_XOR(16));
        default:
            return __shfl_xor(v, mask);
    }
}
static __device__ __forceinline__ int topk_shfl_xor(int v, int mask) {
    return topk_shfl_xor_i32(v, mask);
}
static __device__ __forceinline__ float topk_shfl_xor(float v, int mask) {
    return __int_as_float(topk_shfl_xor_i32(__float_as_int(v), mask));
}
#    define TOPK_SHFL_XOR(v, mask) topk_shfl_xor((v), (mask))
#else
#    define TOPK_SHFL_XOR(v, mask) __shfl_xor_sync(0xffffffffu, (v), (mask))
#endif

// Dedicated argmax for k == 1 (HIP / no-CUB builds).
//
// TOP_K with k == 1 is a pure argmax, and the op accepts *any* index holding the
// maximum value (with ties the test compares the values behind the indices, not
// the indices themselves). The generic bitonic path handles k == 1 as a kpad == 1
// merge-tree: log2(ncols_pad) reduction levels, each with two block barriers and a
// strided shared-memory round-trip. A direct wave-shuffle reduction replaces that
// with two barrier-free shuffle passes and a single __syncthreads(). Threads are
// rounded up to a whole number of waves so every lane in every wave participates
// in the shuffles (out-of-range lanes seed -inf and never win). Ties break toward
// the lower index purely to keep the reduction deterministic.
static __global__ void k_topk_argmax(const float * x, int * dst, const int ncols) {
    const int col = threadIdx.x;
    const int row = blockIdx.x;

    const float * x_row = x + (size_t) row * ncols;

    float v  = col < ncols ? x_row[col] : -INFINITY;
    int   id = col;

    // Intra-wave reduction.
    for (int j = warpSize >> 1; j > 0; j >>= 1) {
        const float pv = TOPK_SHFL_XOR(v, j);
        const int   pi = TOPK_SHFL_XOR(id, j);
        if (pv > v || (pv == v && pi < id)) {
            v  = pv;
            id = pi;
        }
    }

    const int lane   = col & (warpSize - 1);
    const int warp   = col / warpSize;
    const int nwarps = blockDim.x / warpSize;

    extern __shared__ int argmax_smem[];
    int *   s_idx = argmax_smem;
    float * s_val = (float *) (argmax_smem + nwarps);

    if (lane == 0) {
        s_idx[warp] = id;
        s_val[warp] = v;
    }
    __syncthreads();

    // Final reduction over the wave leaders, done entirely in wave 0 (nwarps is at
    // most ncols_pad/warpSize <= 1024/32 = 32 <= warpSize, so one wave covers them).
    if (warp == 0) {
        v  = lane < nwarps ? s_val[lane] : -INFINITY;
        id = lane < nwarps ? s_idx[lane] : 0;
        for (int j = warpSize >> 1; j > 0; j >>= 1) {
            const float pv = TOPK_SHFL_XOR(v, j);
            const int   pi = TOPK_SHFL_XOR(id, j);
            if (pv > v || (pv == v && pi < id)) {
                v  = pv;
                id = pi;
            }
        }
        if (lane == 0) {
            dst[row] = id;
        }
    }
}

// Dedicated block-resident top-k selection (HIP / no-CUB builds).
//
// TOP_K only requires the *set* of the k largest indices per row — the op's
// output order is explicitly irrelevant (the CPU reference even swaps the first
// two outputs, and the test compares results as an order-independent set). So
// instead of fully sorting the whole (padded) row and copying the first k, we
// do a partial bitonic top-k that only ever keeps a window of `kpad` elements:
//
//   Phase A: bitonic-sort each contiguous `kpad`-block of the row descending.
//   Phase B: merge-tree — repeatedly merge two adjacent descending `kpad`-blocks,
//            keeping only the top `kpad` of the pair (C[i] = max(X[i], Y[kpad-1-i])
//            is bitonic and holds the kpad largest; one bitonic merge re-sorts it
//            descending). After log2(ncols_pad/kpad) levels, block 0 holds the top
//            kpad descending, and its first k entries are the top-k set.
//
// Work scales with kpad rather than ncols_pad, so small-k rows (the common case)
// avoid the full O(n log^2 n) sort. Padding lanes are seeded with -inf so they can
// never enter the top-k (ncols >= k always holds for TOP_K), which lets the whole
// network compare values directly with no index/padding guards.
// KPAD (= topk_next_power_of_2(k)) and WARP (= the device warp size) are both
// compile-time template parameters. This is the same lever proven on the smallk
// (iter-21, +3.1%) and 2wave (iter-22, +4.6%) paths, applied to the last
// untemplated (k > 2*warpSize) path: with static KPAD/WARP the Phase-A sort
// network's `len`/`j` loops and every intra-wave shuffle merge tail have
// compile-time bounds and `#pragma unroll` to straight-line dependent shuffle
// chains, removing the per-stage loop-counter shift+compare+branch that sat on
// the (barrier-light) register-shuffle critical path. The barrier/LDS structure
// is preserved BYTE-FOR-BYTE: the cross-wave stages keep the same single
// writeback-per-`len` + in-place LDS compare-exchange + one barrier per stage,
// and the Phase-B `span` loop stays runtime (it is ncols_pad-dependent and
// barrier-bound, so unrolling it buys nothing). Only loop control is removed;
// the compare-exchange decisions are identical, so the produced top-k set is
// bit-identical to the runtime kernel.
template <int KPAD, int WARP>
static __global__ void k_topk_bitonic(const float * x,
                                       int *         dst,
                                       const int     ncols,
                                       const int     ncols_pad,
                                       const int     k) {
    const int col = threadIdx.x;
    const int row = blockIdx.x;

    const float * x_row = x + (size_t) row * ncols;

    extern __shared__ int smem[];
    int *   idx_row = smem;
    float * val_row = (float *) (smem + ncols_pad);

    // Phase A: sort every contiguous KPAD-block descending (blocks are
    // independent — partners of a compare-exchange stay within the block).
    // (v, id) stay register-resident; LDS is touched only for cross-wave stages
    // (stride >= WARP). A `len <= WARP` stage is fully intra-wave (every stride
    // j <= len/2 < WARP) and runs entirely via xor shuffle with no LDS/barrier —
    // that whole branch is dead-code-eliminated for the cross-wave `len`s.
    float v  = col < ncols ? x_row[col] : -INFINITY;
    int   id = col;
#pragma unroll
    for (int len = 2; len <= KPAD; len <<= 1) {
        if (len <= WARP) {
            // Fully intra-wave sort stage: register-to-register xor shuffle.
#pragma unroll
            for (int j = len >> 1; j > 0; j >>= 1) {
                const float pv = TOPK_SHFL_XOR(v, j);
                const int   pi = TOPK_SHFL_XOR(id, j);

                const bool  up       = (((col & (KPAD - 1)) & len) == 0);
                const bool  low      = (col & j) == 0;
                const float low_val  = low ? v : pv;
                const float high_val = low ? pv : v;
                const bool  do_swap  = up ? (low_val < high_val) : (low_val > high_val);
                if (do_swap) {
                    v  = pv;
                    id = pi;
                }
            }
        } else {
            // Cross-wave stages (stride >= WARP): exchange through LDS + barrier.
            // Flush the register-resident (v, id) to LDS first so partner lanes
            // read the current values.
            val_row[col] = v;
            idx_row[col] = id;
            __syncthreads();
#pragma unroll
            for (int j = len >> 1; j >= WARP; j >>= 1) {
                const int partner = col ^ j;
                if (partner > col) {
                    // Descending target for this comparator (uniform per block at
                    // the top stage; alternating within to build bitonic subseqs).
                    const bool up = (((col & (KPAD - 1)) & len) == 0);
                    const bool do_swap =
                        up ? (val_row[col] < val_row[partner]) : (val_row[col] > val_row[partner]);
                    if (do_swap) {
                        topk_swap(val_row[col], val_row[partner]);
                        topk_swap(idx_row[col], idx_row[partner]);
                    }
                }
                __syncthreads();
            }
            v  = val_row[col];
            id = idx_row[col];

            // Intra-wave tail (stride < WARP): xor shuffle, no LDS/barrier. Both
            // lanes of a pair reconstruct the same (low, high) view and the same
            // `up` direction, so the exchange matches the shared-memory path.
#pragma unroll
            for (int j = WARP >> 1; j > 0; j >>= 1) {
                const float pv = TOPK_SHFL_XOR(v, j);
                const int   pi = TOPK_SHFL_XOR(id, j);

                const bool  up       = (((col & (KPAD - 1)) & len) == 0);
                const bool  low      = (col & j) == 0;
                const float low_val  = low ? v : pv;
                const float high_val = low ? pv : v;
                const bool  do_swap  = up ? (low_val < high_val) : (low_val > high_val);
                if (do_swap) {
                    v  = pv;
                    id = pi;
                }
            }
        }
    }
    // Publish the sorted KPAD-blocks to LDS for Phase B's cross-wave merges.
    val_row[col] = v;
    idx_row[col] = id;
    __syncthreads();

    // Phase B: merge-tree. Merge the two descending KPAD-blocks at the head of
    // each 2*span group into a single descending top-KPAD block stored at the
    // group head. `span` doubles each level until one block spans the row.
    // Stays runtime: trip count log2(ncols_pad/KPAD) is ncols_pad-dependent and
    // this loop is barrier-bound, so unrolling it would only bloat code.
    for (int span = KPAD; span < ncols_pad; span <<= 1) {
        const int gs     = span << 1;
        const int i      = col & (gs - 1);       // position within the 2*span group
        const int gbase  = col - i;              // group head
        const bool active = i < KPAD;            // only the surviving KPAD lanes work
        // On the final merge level the whole row is one group and each col < k
        // lane emits only its OWN slot to dst. The post-merge LDS writeback (and
        // its ordering barrier + the trailing readback) then order a store that
        // nobody else reads, so on the last level we emit dst straight from the
        // register-resident merge result and skip the LDS round-trip entirely.
        const bool last = (span << 1) >= ncols_pad;

        if (active) {
            // C[i] = max(X[i], Y[KPAD-1-i]) is bitonic and holds the KPAD largest.
            // The pre-writeback barrier is unnecessary: each active lane reads its
            // own lower-half slot a_pos = gbase+i plus an upper-half slot b_pos in
            // [gbase+span, gbase+span+KPAD) (never written this level since
            // KPAD <= span), and writes back only to gbase+i. The write set
            // [gbase, gbase+KPAD) is disjoint from the cross-half read set and the
            // a_pos read-before-write is intra-thread ordered, so only the
            // post-writeback barrier is needed.
            const int   a_pos = gbase + i;
            const int   b_pos = gbase + span + (KPAD - 1 - i);
            const float av    = val_row[a_pos];
            const float bv    = val_row[b_pos];
            if (av >= bv) {
                val_row[gbase + i] = av;
                idx_row[gbase + i] = idx_row[a_pos];
            } else {
                val_row[gbase + i] = bv;
                idx_row[gbase + i] = idx_row[b_pos];
            }
        }
        __syncthreads();

        // Cross-wave merge stages via shared memory (j = KPAD/2 .. WARP).
#pragma unroll
        for (int j = KPAD >> 1; j >= WARP; j >>= 1) {
            const int partner = col ^ j;
            if (active && partner > col) {
                if (val_row[col] < val_row[partner]) {
                    topk_swap(val_row[col], val_row[partner]);
                    topk_swap(idx_row[col], idx_row[partner]);
                }
            }
            __syncthreads();
        }

        // Intra-wave merge tail (j = WARP/2 .. 1) via xor shuffle. Merge partners
        // (col ^ j, j < KPAD) stay within the aligned head KPAD-block, so an
        // active lane's partner is also active; inactive lanes shuffle harmlessly
        // but never write back. Uniform descending merge: low lane keeps the max.
        {
            float mv  = val_row[col];
            int   mid = idx_row[col];
#pragma unroll
            for (int j = WARP >> 1; j > 0; j >>= 1) {
                const float pv = TOPK_SHFL_XOR(mv, j);
                const int   pi = TOPK_SHFL_XOR(mid, j);

                const bool  low      = (col & j) == 0;
                const float low_val  = low ? mv : pv;
                const float high_val = low ? pv : mv;
                if (low_val < high_val) {
                    mv  = pv;
                    mid = pi;
                }
            }
            if (active) {
                if (last) {
                    // Last level: emit our own top-k slot straight from registers.
                    if (col < k) {
                        dst[(size_t) row * k + col] = mid;
                    }
                } else {
                    val_row[col] = mv;
                    idx_row[col] = mid;
                }
            }
            if (!last) {
                __syncthreads();
            }
        }
    }

    // If Phase B never ran (KPAD >= ncols_pad: Phase A already fully sorted the
    // row), emit the top-k from the published LDS. Otherwise the last merge level
    // above already wrote dst directly from registers.
    if (KPAD >= ncols_pad && col < k) {
        dst[(size_t) row * k + col] = idx_row[col];
    }
}

// Dedicated small-k top-k for kpad <= warpSize (HIP / no-CUB builds).
//
// Same partial-bitonic algorithm as k_topk_bitonic, specialised for the case
// where a whole kpad-block fits inside one wave. Two barriers/LDS-traffic cuts
// that are only valid in this regime:
//   * Phase A runs entirely in registers. With kpad <= warpSize every
//     compare-exchange partner (col ^ j, j < kpad) is in the same wave, so the
//     per-block sort is driven purely by xor shuffles — no LDS round-trip and no
//     __syncthreads() between the len stages (the shared kernel writes back to
//     LDS and barriers after every len).
//   * Phase B elides the C-step's pre-writeback barrier. Each active lane reads
//     its own lower-half slot (a_pos = gbase + i) plus an upper-half slot
//     (b_pos, never written this level) and writes back only to gbase + i, so no
//     lane's write aliases another lane's read — the read/write sets are disjoint
//     and the merge only needs the single post-writeback barrier.
// Both cuts are pure schedule changes; the compare-exchange decisions are
// byte-for-byte those of k_topk_bitonic, so the produced top-k set is identical.
// KPAD is a compile-time template parameter (a power of two <= warpSize). Making
// it static lets the compiler fully unroll Phase A's sort network and Phase B's
// intra-wave merge tail: on this path those are register-only, dependent
// shuffle chains that are NOT hidden behind barriers, so eliminating the loop
// counters / per-iteration shift+branch overhead shortens the critical path.
// The Phase-B `span` loop stays runtime (it depends on ncols_pad and is the
// barrier-bound part where unrolling buys nothing).
template <int KPAD>
static __global__ void k_topk_bitonic_smallk(const float * x,
                                             int *         dst,
                                             const int     ncols,
                                             const int     ncols_pad,
                                             const int     k) {
    const int col = threadIdx.x;
    const int row = blockIdx.x;

    const float * x_row = x + (size_t) row * ncols;

    extern __shared__ int smem[];
    int *   idx_row = smem;
    float * val_row = (float *) (smem + ncols_pad);

    // Phase A (register-only): sort each contiguous KPAD-block descending.
    float v  = col < ncols ? x_row[col] : -INFINITY;
    int   id = col;
#pragma unroll
    for (int len = 2; len <= KPAD; len <<= 1) {
#pragma unroll
        for (int j = len >> 1; j > 0; j >>= 1) {
            const float pv = TOPK_SHFL_XOR(v, j);
            const int   pi = TOPK_SHFL_XOR(id, j);

            const bool  up       = (((col & (KPAD - 1)) & len) == 0);
            const bool  low      = (col & j) == 0;
            const float low_val  = low ? v : pv;
            const float high_val = low ? pv : v;
            const bool  do_swap  = up ? (low_val < high_val) : (low_val > high_val);
            if (do_swap) {
                v  = pv;
                id = pi;
            }
        }
    }
    val_row[col] = v;
    idx_row[col] = id;
    __syncthreads();

    // Phase B: merge-tree. Both C-step barriers are elided here (kpad <= warpSize):
    //   * pre-writeback: lower-half write set [gbase,gbase+kpad) is disjoint from
    //     the upper-half cross-read since span >= kpad (see header note).
    //   * post-C-step: the merge is fully intra-wave (kpad <= warpSize), so it
    //     loads each lane's own slot exactly once (val_row[col], written by that
    //     lane itself in the C-step — intra-thread ordered) and then runs entirely
    //     in registers via xor shuffle. No lane reads another lane's LDS slot
    //     between the C-step write and the merge load, so the barrier that fed the
    //     load is unnecessary. A single post-merge-writeback barrier per level
    //     suffices to order the writeback ahead of the next level's C-step read.
    for (int span = KPAD; span < ncols_pad; span <<= 1) {
        const int  gs     = span << 1;
        const int  i      = col & (gs - 1);
        const int  gbase  = col - i;
        const bool active = i < KPAD;

        if (active) {
            const int   a_pos = gbase + i;
            const int   b_pos = gbase + span + (KPAD - 1 - i);
            const float av    = val_row[a_pos];
            const float bv    = val_row[b_pos];
            if (av >= bv) {
                val_row[gbase + i] = av;
                idx_row[gbase + i] = idx_row[a_pos];
            } else {
                val_row[gbase + i] = bv;
                idx_row[gbase + i] = idx_row[b_pos];
            }
        }

        // Bitonic merge the KPAD-wide block back to descending order. KPAD <=
        // warpSize, so the whole merge is intra-wave xor shuffle. Each active lane
        // loads only its own C-step write (no barrier needed, see above).
        v  = val_row[col];
        id = idx_row[col];
#pragma unroll
        for (int j = KPAD >> 1; j > 0; j >>= 1) {
            const float pv = TOPK_SHFL_XOR(v, j);
            const int   pi = TOPK_SHFL_XOR(id, j);

            const bool  low      = (col & j) == 0;
            const float low_val  = low ? v : pv;
            const float high_val = low ? pv : v;
            if (low_val < high_val) {
                v  = pv;
                id = pi;
            }
        }
        if (active) {
            val_row[col] = v;
            idx_row[col] = id;
        }
        __syncthreads();
    }

    if (col < k) {
        dst[(size_t) row * k + col] = idx_row[col];
    }
}

// Dedicated top-k for a two-wave kpad-block (warpSize < kpad <= 2*warpSize;
// HIP / no-CUB builds). kpad is a power of two, so this regime is exactly
// kpad == 2*warpSize (kpad=64 on wave32). Same partial-bitonic algorithm as
// k_topk_bitonic, but exploiting that a kpad-block spans only two waves:
//   * Phase A runs in registers across all `len` stages. Every compare-exchange
//     partner (col ^ j, j < kpad) with j < warpSize is in the same wave, and the
//     single j == warpSize top stage is the only cross-wave exchange — so instead
//     of the shared kernel's LDS write-back + __syncthreads() after every `len`,
//     the block sorts entirely in registers via xor shuffle and touches LDS just
//     once (for that one cross-wave exchange). The comparator (up/low/do_swap) is
//     byte-for-byte the shared kernel's shuffle tail.
//   * Phase B elides the C-step's pre-write-back barrier (see k_topk_bitonic_smallk
//     header): each active lane's write set [gbase, gbase+kpad) is disjoint from its
//     cross-half read slot b_pos in [gbase+span, gbase+span+kpad) since kpad <= span,
//     so only the post-write-back barrier is needed.
// Both cuts are pure schedule changes; the produced top-k set is identical to
// k_topk_bitonic. Isolated in its own kernel so the k > 2*warpSize path's codegen
// (k_topk_bitonic) is unchanged.
//
// KPAD is a compile-time template parameter. This path is entered only when
// kpad == 2*warpSize (kpad is a power of two with warpSize < kpad <= 2*warpSize),
// so warpSize == KPAD/2 is compile-time knowable (constexpr WARP below). Making
// KPAD static lets the compiler fully unroll Phase A's sort network and the
// Phase-B intra-wave merge tail — register-only dependent shuffle chains that are
// NOT hidden behind barriers, so removing the runtime loop counters / per-stage
// shift+compare+branch shortens their exposed critical path (the same +3.1% lever
// iter-21 applied to the smallk path). Because WARP is constexpr, the single
// cross-wave stage in each phase (j == WARP) resolves at compile time and every
// other stage's cross-wave branch is dead-code-eliminated to a straight-line
// shuffle. The Phase-B `span` loop stays runtime (it depends on ncols_pad and is
// the barrier-bound part where unrolling buys nothing). Comparator logic is
// byte-for-byte the runtime version, so the produced top-k set is identical.
template <int KPAD>
static __global__ void k_topk_bitonic_2wave(const float * x,
                                            int *         dst,
                                            const int     ncols,
                                            const int     ncols_pad,
                                            const int     k) {
    constexpr int WARP = KPAD >> 1;  // == warpSize on this path (kpad == 2*warpSize)

    const int col = threadIdx.x;
    const int row = blockIdx.x;

    const float * x_row = x + (size_t) row * ncols;

    extern __shared__ int smem[];
    int *   idx_row = smem;
    float * val_row = (float *) (smem + ncols_pad);

    // Phase A: sort every contiguous KPAD-block descending, register-resident.
    // Exactly one cross-wave stage exists (len == KPAD, j == WARP); every other
    // (len, j) has j < WARP and is intra-wave shuffle. Both bounds are compile-time
    // so the network unrolls and the j >= WARP branch survives in only that one copy.
    float v  = col < ncols ? x_row[col] : -INFINITY;
    int   id = col;
#pragma unroll
    for (int len = 2; len <= KPAD; len <<= 1) {
#pragma unroll
        for (int j = len >> 1; j > 0; j >>= 1) {
            const bool up = (((col & (KPAD - 1)) & len) == 0);
            if (j >= WARP) {
                // Cross-wave exchange (partner in the sibling wave) through LDS.
                val_row[col] = v;
                idx_row[col] = id;
                __syncthreads();
                const int   partner = col ^ j;
                const float pv       = val_row[partner];
                const int   pi       = idx_row[partner];
                __syncthreads();  // all cross-wave reads done before LDS reuse
                const bool  low      = (col & j) == 0;
                const float low_val  = low ? v : pv;
                const float high_val = low ? pv : v;
                const bool  do_swap  = up ? (low_val < high_val) : (low_val > high_val);
                if (do_swap) {
                    v  = pv;
                    id = pi;
                }
            } else {
                // Intra-wave: register-to-register via xor shuffle.
                const float pv = TOPK_SHFL_XOR(v, j);
                const int   pi = TOPK_SHFL_XOR(id, j);

                const bool  low      = (col & j) == 0;
                const float low_val  = low ? v : pv;
                const float high_val = low ? pv : v;
                const bool  do_swap  = up ? (low_val < high_val) : (low_val > high_val);
                if (do_swap) {
                    v  = pv;
                    id = pi;
                }
            }
        }
    }
    __syncthreads();  // ensure the last cross-wave LDS reads finished before overwrite
    val_row[col] = v;
    idx_row[col] = id;
    __syncthreads();

    // Phase B: merge-tree (C-step pre-write-back barrier elided, see header).
    for (int span = KPAD; span < ncols_pad; span <<= 1) {
        const int  gs     = span << 1;
        const int  i      = col & (gs - 1);
        const int  gbase  = col - i;
        const bool active = i < KPAD;

        if (active) {
            const int   a_pos = gbase + i;
            const int   b_pos = gbase + span + (KPAD - 1 - i);
            const float av    = val_row[a_pos];
            const float bv    = val_row[b_pos];
            if (av >= bv) {
                val_row[gbase + i] = av;
                idx_row[gbase + i] = idx_row[a_pos];
            } else {
                val_row[gbase + i] = bv;
                idx_row[gbase + i] = idx_row[b_pos];
            }
        }
        __syncthreads();

        // Single cross-wave merge stage (j == WARP == KPAD/2) via shared memory.
        {
            const int partner = col ^ WARP;
            if (active && partner > col) {
                if (val_row[col] < val_row[partner]) {
                    topk_swap(val_row[col], val_row[partner]);
                    topk_swap(idx_row[col], idx_row[partner]);
                }
            }
            __syncthreads();
        }

        // Intra-wave merge tail (j = WARP/2 .. 1), fully unrolled via xor shuffle.
        {
            float mv  = val_row[col];
            int   mid = idx_row[col];
#pragma unroll
            for (int j = WARP >> 1; j > 0; j >>= 1) {
                const float pv = TOPK_SHFL_XOR(mv, j);
                const int   pi = TOPK_SHFL_XOR(mid, j);

                const bool  low      = (col & j) == 0;
                const float low_val  = low ? mv : pv;
                const float high_val = low ? pv : mv;
                if (low_val < high_val) {
                    mv  = pv;
                    mid = pi;
                }
            }
            if (active) {
                val_row[col] = mv;
                idx_row[col] = mid;
            }
            __syncthreads();
        }
    }

    if (col < k) {
        dst[(size_t) row * k + col] = idx_row[col];
    }
}

// Dispatch the shared (kpad > 2*warpSize) partial-bitonic kernel over the
// compile-time (KPAD, WARP) ladder. kpad is a power of two in {128,256,512,1024}
// (kpad = next_pow2(k), k <= ncols <= 1024); WARP is the device warp size. Every
// k routing here gets its own fully-unrolled instantiation — nothing is keyed on
// the benchmark's ncols/k.
template <int WARP>
static void topk_bitonic_launch(const float * x,
                                int *         dst,
                                const int     ncols,
                                const int     ncols_pad,
                                const int     k,
                                const int     kpad,
                                const dim3    block_nums,
                                const dim3    block_dims,
                                const size_t  shared_mem,
                                cudaStream_t  stream) {
    switch (kpad) {
        case 128:
            k_topk_bitonic<128, WARP><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, k);
            break;
        case 256:
            k_topk_bitonic<256, WARP><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, k);
            break;
        case 512:
            k_topk_bitonic<512, WARP><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, k);
            break;
        case 1024:
            k_topk_bitonic<1024, WARP><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, k);
            break;
        default:
            GGML_ABORT("top-k shared: unexpected kpad %d", kpad);
    }
}

static void topk_bitonic_cuda(const float * x,
                              int *         dst,
                              const int     ncols,
                              const int     nrows,
                              const int     k,
                              cudaStream_t  stream) {
    const int ncols_pad = topk_next_power_of_2(ncols);
    const int kpad      = topk_next_power_of_2(k);

    const dim3   block_dims(ncols_pad, 1, 1);
    const dim3   block_nums(nrows, 1, 1);
    const size_t shared_mem = ncols_pad * (sizeof(int) + sizeof(float));

    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    if (warp_size == 64) {
        topk_bitonic_launch<64>(x, dst, ncols, ncols_pad, k, kpad, block_nums, block_dims, shared_mem, stream);
    } else {
        topk_bitonic_launch<32>(x, dst, ncols, ncols_pad, k, kpad, block_nums, block_dims, shared_mem, stream);
    }
}

static void topk_bitonic_smallk_cuda(const float * x,
                                     int *         dst,
                                     const int     ncols,
                                     const int     nrows,
                                     const int     k,
                                     cudaStream_t  stream) {
    const int ncols_pad = topk_next_power_of_2(ncols);
    const int kpad      = topk_next_power_of_2(k);

    const dim3   block_dims(ncols_pad, 1, 1);
    const dim3   block_nums(nrows, 1, 1);
    const size_t shared_mem = ncols_pad * (sizeof(int) + sizeof(float));

    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    // kpad is a power of two <= warpSize (this path's precondition). Dispatch to
    // the KPAD-templated kernel so its Phase-A sort and Phase-B merge unroll at
    // compile time. warpSize is 32 (wave32) on gfx1201 and 64 on CDNA/wave64, so
    // instantiate the full {2..64} ladder; the switch is exhaustive for any
    // power-of-two kpad in range.
    switch (kpad) {
        case 2:
            k_topk_bitonic_smallk<2><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, k);
            break;
        case 4:
            k_topk_bitonic_smallk<4><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, k);
            break;
        case 8:
            k_topk_bitonic_smallk<8><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, k);
            break;
        case 16:
            k_topk_bitonic_smallk<16><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, k);
            break;
        case 32:
            k_topk_bitonic_smallk<32><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, k);
            break;
        case 64:
            k_topk_bitonic_smallk<64><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, k);
            break;
        default:
            GGML_ABORT("top-k smallk: unexpected kpad %d", kpad);
    }
}

static void topk_bitonic_2wave_cuda(const float * x,
                                    int *         dst,
                                    const int     ncols,
                                    const int     nrows,
                                    const int     k,
                                    cudaStream_t  stream) {
    const int ncols_pad = topk_next_power_of_2(ncols);
    const int kpad      = topk_next_power_of_2(k);

    const dim3   block_dims(ncols_pad, 1, 1);
    const dim3   block_nums(nrows, 1, 1);
    const size_t shared_mem = ncols_pad * (sizeof(int) + sizeof(float));

    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    // This path is entered only when kpad == 2*warpSize (a power of two with
    // warpSize < kpad <= 2*warpSize). warpSize is 32 (wave32, gfx1201) or 64
    // (CDNA/wave64), so kpad is exactly 64 or 128; dispatch to the KPAD-templated
    // kernel so its Phase-A sort and Phase-B merge tail unroll at compile time.
    switch (kpad) {
        case 64:
            k_topk_bitonic_2wave<64><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, k);
            break;
        case 128:
            k_topk_bitonic_2wave<128><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, k);
            break;
        default:
            GGML_ABORT("top-k 2wave: unexpected kpad %d", kpad);
    }
}

static void topk_argmax_cuda(const float * x,
                             int *         dst,
                             const int     ncols,
                             const int     nrows,
                             cudaStream_t  stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // Round the thread count up to a whole number of waves so every shuffle lane
    // is a launched thread (ncols <= 1024 is guaranteed by supports_op).
    const int nthreads = ((ncols + warp_size - 1) / warp_size) * warp_size;
    const int nwarps   = nthreads / warp_size;

    const dim3   block_dims(nthreads, 1, 1);
    const dim3   block_nums(nrows, 1, 1);
    const size_t shared_mem = (size_t) nwarps * (sizeof(int) + sizeof(float));

    k_topk_argmax<<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols);
}

#endif  // !defined(GGML_CUDA_USE_CUB)

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    if (shared_mem > max_shared_mem || ncols > 1024) {
        argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    } else {
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    }
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#else                             // GGML_CUDA_USE_CUB
    // ncols > 1024 exceeds the single-block bitonic cap (1024 threads/block).
    // Route it through the device-wide sort (hipCUB / rocPRIM on ROCm) + slice
    // the first k, mirroring the CUB branch above; keep the fast partial-bitonic
    // top-k for ncols <= 1024 (the common decode/verify case).
    if (ncols > 1024) {
#ifdef GGML_CUDA_USE_HIPCUB
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#else
        GGML_UNUSED(pool);
        GGML_ABORT("top-k: ncols > 1024 requires CUB/hipCUB (rocPRIM)");
#endif  // GGML_CUDA_USE_HIPCUB
        return;
    }
    // Dedicated partial bitonic top-k: keeps only a kpad-wide window instead of
    // fully sorting the row then discarding all but the first k indices.
    GGML_UNUSED(pool);
    if (k == 1) {
        // k == 1 is a pure argmax — a direct wave-shuffle reduction beats the
        // kpad == 1 bitonic merge-tree (fewer barriers, no strided shared traffic).
        topk_argmax_cuda(src0_d, dst_d, ncols, nrows, stream);
    } else {
        const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
        const int kpad      = topk_next_power_of_2(k);
        if (kpad <= warp_size) {
            // A whole kpad-block fits in one wave: run Phase A fully in registers
            // (no LDS/barriers) and fuse the Phase-B C-step barriers. Isolated in
            // its own kernel so the k > warpSize path's codegen is unchanged.
            topk_bitonic_smallk_cuda(src0_d, dst_d, ncols, nrows, k, stream);
        } else if (kpad <= 2 * warp_size) {
            // A kpad-block spans exactly two waves: Phase A stays register-resident
            // (LDS only for the single j == warpSize cross-wave exchange) and the
            // Phase-B C-step pre-barrier is elided. Isolated so the k > 2*warpSize
            // path's codegen (k_topk_bitonic) is unchanged.
            topk_bitonic_2wave_cuda(src0_d, dst_d, ncols, nrows, k, stream);
        } else {
            topk_bitonic_cuda(src0_d, dst_d, ncols, nrows, k, stream);
        }
    }
#endif
}
