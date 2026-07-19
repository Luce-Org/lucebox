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

#elif defined(GGML_CUDA_USE_CUB) || defined(GGML_CUDA_USE_HIPCUB)  // CUB_TOP_K_AVAILABLE

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
// was verified bit-identical to `__shfl_xor` on gfx1201 and gfx1151 for all five masks.)
#    define TOPK_DS_SWIZZLE_XOR(m) (((m) << 10) | 0x1f)
// The intra-quad xor masks (1, 2) are expressible as a DPP quad_perm, which runs
// as a VALU cross-lane operand-gather (fused into the ALU pipe) rather than an
// `ds_swizzle_b32` through the LDS permute crossbar. On the latency-bound
// register-shuffle chains here (Phase-A sort tails, Phase-B merge tails, argmax
// butterfly) that shaves the crossbar issue off the two shortest-stride stages of
// every chain. quad_perm ctrl selects, per output lane j in a 4-lane quad, which
// quad lane (ctrl>>(2*j))&3 to read; xor^1 -> [1,0,3,2] = 0xB1, xor^2 ->
// [2,3,0,1] = 0x4E. All lanes are launched/active (out-of-range lanes seed -inf),
// so the gather is exact regardless of bound_ctrl. Masks 4/8/16 are not intra-quad
// and stay on ds_swizzle; the wave64 j==32 tail still falls back to __shfl_xor.
#    define TOPK_DPP_QUAD_XOR1 0xB1
#    define TOPK_DPP_QUAD_XOR2 0x4E
// The ^16 xor is the widest intra-wave stride and the ENTRY stage of every
// 32-lane shuffle chain here (the merge/sort tails run j = 16,8,4,2,1), so it
// sits on the exposed critical path. It swaps the two 16-lane halves of the
// wave — exactly what `v_permlanex16_b32` does in its identity-cross form: with
// selectors (0x76543210, 0xFEDCBA98) output lane j reads the opposite half's
// same-index lane (j<16 -> lane j+16 = j^16; j>=16 -> lane j-16 = j^16). Unlike
// `ds_swizzle_b32`, which issues through the LDS permute crossbar, permlanex16
// is a pure VALU cross-lane gather (fused into the ALU pipe) — the same
// "off the LDS crossbar onto VALU" move that DPP quad_perm won for ^1/^2, now
// applied to the largest intra-wave stride. bound_ctrl/fi are false; every lane
// is written by the full permutation so the passthrough `old` operand is dead.
// Verified bit-identical to `__shfl_xor(v, 16)` on gfx1201 and gfx1151. Masks 4/8 stay on
// ds_swizzle; the wave64 j==32 tail still falls back to __shfl_xor.
static __device__ __forceinline__ int topk_shfl_xor_i32(int v, int mask) {
    switch (mask) {
        case 1:
            return __builtin_amdgcn_mov_dpp(v, TOPK_DPP_QUAD_XOR1, 0xf, 0xf, false);
        case 2:
            return __builtin_amdgcn_mov_dpp(v, TOPK_DPP_QUAD_XOR2, 0xf, 0xf, false);
        case 4:
            return __builtin_amdgcn_ds_swizzle(v, TOPK_DS_SWIZZLE_XOR(4));
        case 8:
            return __builtin_amdgcn_ds_swizzle(v, TOPK_DS_SWIZZLE_XOR(8));
        case 16:
            return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xFEDCBA98u, false, false);
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
// in the shuffles (out-of-range lanes seed -inf and never win).
//
// The compare-select carries no index tiebreak: on a tie we keep the current
// lane's (v, id). Because (v, id) only ever advances to a partner when its value
// is *strictly* greater, the invariant x_row[id] == v holds at every stage, so
// after the butterfly every lane's v is the wave max (value-max is commutative /
// associative regardless of tie policy) and id is an index achieving it. Only
// lane 0's id is published, and it is always a valid argmax — which is all TOP_K
// with k == 1 needs (the op accepts any index holding the max; the test compares
// the values behind the indices, not the indices themselves). Dropping the
// `pv == v && pi < id` term shortens the dependent compare on each of this
// latency-bound reduction's shuffle stages; the only property lost is which of
// several tied maxima is returned, which the op leaves unspecified.
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
        if (pv > v) {
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
            if (pv > v) {
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
    // Phase A double-buffers the cross-wave LDS window (iter-51): each cross-wave
    // `len` uses one of two ncols_pad-wide (idx,val) regions, and the next len
    // (or the Phase-A publish) writes the OTHER region. The pure-WAR barrier that
    // used to separate a len's peel reads from the next len's writeback is thus
    // replaced by buffer separation and dropped. Consecutive uses of the SAME
    // region are always >= one intervening len apart (that len's own barriers
    // order them), so reuse stays race-free with no wave-lockstep assumption.
    int *   idx_buf[2] = { smem, smem + ncols_pad };
    float * val_buf[2] = { (float *) (smem + 2 * ncols_pad),
                           (float *) (smem + 3 * ncols_pad) };
    int     bsel       = 0;
    int *   idx_row    = idx_buf[0];
    float * val_row    = val_buf[0];

    // Phase A: sort every contiguous KPAD-block descending (blocks are
    // independent — partners of a compare-exchange stay within the block).
    // (v, id) stay register-resident; LDS is touched only for cross-wave stages
    // (stride >= WARP). A `len <= WARP` stage is fully intra-wave (every stride
    // j <= len/2 < WARP) and runs entirely via xor shuffle with no LDS/barrier —
    // that whole branch is dead-code-eliminated for the cross-wave `len`s.
    float v  = col < ncols ? x_row[col] : -INFINITY;
    int   id = col;
    // The final (widest) sort len == KPAD is peeled below so its middle
    // cross-wave strides can use the iter-53 group-merge fuse; this loop runs
    // the lower lens (len < KPAD) with the stock per-stride structure.
#pragma unroll
    for (int len = 2; len < KPAD; len <<= 1) {
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
            // Pick this len's LDS region (iter-51 double-buffer) and flush the
            // register-resident (v, id) to it so partner lanes read the current
            // values. The previous cross-wave len read the OTHER region, so this
            // writeback cannot clobber its still-in-flight peel reads.
            val_row = val_buf[bsel];
            idx_row = idx_buf[bsel];
            val_row[col] = v;
            idx_row[col] = id;
            __syncthreads();
            // Cross-wave strides via in-place LDS swap, down to but NOT including
            // j == WARP (peeled below as a register merge, iter-47). The final
            // barrier of this loop (or the writeback barrier above when the loop
            // runs zero iterations, i.e. len == 2*WARP) publishes the sibling-wave
            // slots the register merge reads.
#pragma unroll
            for (int j = len >> 1; j > WARP; j >>= 1) {
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

            // Peeled last cross-wave stride (j == WARP) as a register merge
            // (iter-47, the iter-46 Phase-B peel applied to every Phase-A
            // cross-wave len). partner == col ^ WARP is in the sibling wave; its
            // slot was published by the barrier above. Instead of the stock
            // "lower lane swaps both slots in LDS, then BOTH lanes reload
            // val_row[col]/idx_row[col]", each lane reads its own slot + its
            // partner's and computes its OWN result in registers: for the pair
            // (low = col&WARP==0, high) with the alternating bitonic direction
            // `up`, the low lane keeps the max and the high the min when up (and
            // the reverse when !up) — byte-for-byte what the stock conditional
            // swap produced. This drops the swap's conditional LDS writes AND the
            // full own-slot reload from the critical path. `up` is uniform across
            // the WARP-pair (they differ only in bit WARP < len, which does not
            // affect col & len), so both lanes agree on the direction.
            //
            // Barrier count is UNCHANGED: the __syncthreads() the stock loop ran
            // after the j == WARP swap is retained here as a WAR barrier — the
            // register merge is the last LDS reader of this len, and the next
            // len's writeback (or Phase A's publish writeback after len == KPAD)
            // overwrites val_row[col], which the partner lane read here. Keeping
            // that barrier makes the elision race-free (no reliance on wave
            // lockstep), so this is a pure LDS-traffic cut, not a barrier cut.
            {
                const int   partner = col ^ WARP;
                const float sv      = val_row[col];
                const float pv      = val_row[partner];
                const bool  up      = (((col & (KPAD - 1)) & len) == 0);
                const bool  low     = (col & WARP) == 0;
                const bool  take    = (up == low) ? (sv < pv) : (sv > pv);
                v  = take ? pv : sv;
                id = take ? idx_row[partner] : idx_row[col];
            }
            // iter-51: the WAR __syncthreads() that used to sit here is DROPPED.
            // The peel above read val_buf[bsel]; the next cross-wave len (or the
            // Phase-A publish) writes val_buf[bsel ^ 1], so there is no
            // write-after-read hazard on this region — buffer separation stands
            // in for the barrier. Advance to the other region for the next len.
            bsel ^= 1;

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

    // Peeled final Phase-A sort stage (len == KPAD), iter-53: apply the iter-52
    // Phase-B middle-cross-wave-stride group-merge fuse to Phase A. This is the
    // widest Phase-A sort len and is ALWAYS cross-wave on this path (KPAD >
    // 2*WARP), and it carries the most middle strides (j = KPAD/2 .. 2*WARP), so
    // it has the most barriers to collapse. The middle strides pair columns that
    // differ only in bits [log2(2*WARP) .. log2(KPAD/2)], so the MID_A =
    // KPAD/(2*WARP) columns { base + t*2*WARP } form a group CLOSED under all of
    // them. When MID_A >= 4 one owner lane per group (i & GMASK == 0) gathers its
    // MID_A slots into registers, runs the whole sub-sort there (group-index
    // strides MID_A/2..1 == full strides KPAD/2..2*WARP), and writes the results
    // back — replacing the M per-stride block barriers AND the M-1 intermediate
    // LDS write->readback round-trips with ONE barrier (same fuse class as
    // iter-50/52). For k=400 (KPAD=512, strides {256,128,64}) that is 3 barriers
    // -> 1.
    //
    // For len == KPAD the sort direction is UNIFORMLY descending: up = ((col &
    // (KPAD-1)) & KPAD) == 0 is always true (bit log2(KPAD) is masked out by
    // KPAD-1), so the group merge is a plain descending bitonic sub-sort with the
    // exact comparator iter-52 uses (low index keeps the max, swap only on strict
    // <; idx follows value). Race-free with no internal barrier: each owner
    // exclusively owns its MID_A group slots (disjoint groups tile the KPAD-block
    // in steps of 2*WARP) and reads ONLY those slots — all published by the
    // writeback barrier below, none written by any other lane this stage; reads
    // precede writes intra-thread. Bit-identical to the stock in-place loop. For
    // MID_A < 4 (KPAD == 2*WARP: 0-or-1 middle stride, no barrier to save) keep
    // the stock in-place loop byte-for-byte.
    {
        const int len = KPAD;
        val_row      = val_buf[bsel];
        idx_row      = idx_buf[bsel];
        val_row[col] = v;
        idx_row[col] = id;
        __syncthreads();

        constexpr int MID_A = KPAD / (2 * WARP);
        if (MID_A >= 4) {
            constexpr int GMASK = (MID_A - 1) * (2 * WARP);
            constexpr int GARR  = MID_A >= 4 ? MID_A : 1;  // avoid 0-length array in dead insts
            const int     i     = col & (KPAD - 1);
            const int     gbase = col - i;
            if ((i & GMASK) == 0) {
                float gv[GARR];
                int   gi[GARR];
#pragma unroll
                for (int t = 0; t < MID_A; ++t) {
                    const int p = gbase + i + t * (2 * WARP);
                    gv[t]       = val_row[p];
                    gi[t]       = idx_row[p];
                }
#pragma unroll
                for (int s = MID_A >> 1; s > 0; s >>= 1) {
#pragma unroll
                    for (int a = 0; a < MID_A; ++a) {
                        const int b = a ^ s;
                        if (b > a && gv[a] < gv[b]) {
                            topk_swap(gv[a], gv[b]);
                            topk_swap(gi[a], gi[b]);
                        }
                    }
                }
#pragma unroll
                for (int t = 0; t < MID_A; ++t) {
                    const int p = gbase + i + t * (2 * WARP);
                    val_row[p]  = gv[t];
                    idx_row[p]  = gi[t];
                }
            }
            __syncthreads();
        } else {
#pragma unroll
            for (int j = len >> 1; j > WARP; j >>= 1) {
                const int partner = col ^ j;
                if (partner > col) {
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
        }

        // Peeled j == WARP register merge (iter-47), unchanged from the loop body.
        {
            const int   partner = col ^ WARP;
            const float sv      = val_row[col];
            const float pv      = val_row[partner];
            const bool  up      = (((col & (KPAD - 1)) & len) == 0);
            const bool  low     = (col & WARP) == 0;
            const bool  take    = (up == low) ? (sv < pv) : (sv > pv);
            v  = take ? pv : sv;
            id = take ? idx_row[partner] : idx_row[col];
        }
        bsel ^= 1;

        // Intra-wave tail (stride < WARP), unchanged from the loop body.
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
    // Publish the sorted KPAD-blocks to LDS for Phase B's cross-wave merges.
    // The last Phase-A len read val_buf[bsel ^ 1]; the trailing `bsel ^= 1`
    // advanced to val_buf[bsel], so publishing here writes the OTHER region and
    // does not clobber that len's still-in-flight peel reads (buffer separation
    // again replaces the dropped WAR barrier). Phase B then operates in this
    // region for the rest of the kernel.
    val_row = val_buf[bsel];
    idx_row = idx_buf[bsel];
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

        // Fused C-step + first cross-wave merge stride (j == KPAD/2), iter-50.
        // The stock code ran the C-step (C[i] = max(X[i], Y[KPAD-1-i]) written to
        // gbase+i for every i < KPAD), a block __syncthreads(), then the first
        // cross-wave stride j == KPAD/2 as an in-place LDS compare-exchange. But the
        // C-step's per-slot output and the j == KPAD/2 compare only couple WITHIN a
        // single (i, i+KPAD/2) pair, so the two stages fuse: the low lane of each
        // pair (local i < KPAD/2) computes BOTH its C-values itself, straight from
        // the still-pristine published input blocks, and emits the descending-merge
        // result directly (max -> gbase+i, min -> gbase+i+KPAD/2). This removes one
        // full block __syncthreads() AND the j == KPAD/2 stage's LDS re-read+re-write
        // round-trip per Phase-B level from the barrier-bound k>2*warp shared path.
        //
        // Race-free without an internal barrier: each low lane exclusively owns
        // outputs gbase+i and gbase+i+KPAD/2 (no other lane writes them); its two
        // cross-block reads b0/b1 land in [gbase+span, gbase+span+KPAD), never
        // written this level (KPAD <= span); its two low-half reads a0/a1 are its own
        // outputs (intra-thread read-before-write). Bit-identical to the stock
        // C-step (a wins ties: av >= bv) + stock stride (low keeps its value on ties:
        // swap only when strictly <, i.e. keep max with cav >= cbv).
        const int half = KPAD >> 1;
        if (i < half) {
            const int   a0  = gbase + i;
            const int   b0  = gbase + span + (KPAD - 1 - i);
            const int   a1  = gbase + i + half;
            const int   b1  = gbase + span + (half - 1 - i);
            const float a0v = val_row[a0];
            const float b0v = val_row[b0];
            const float a1v = val_row[a1];
            const float b1v = val_row[b1];
            float cav;
            int   cai;
            if (a0v >= b0v) { cav = a0v; cai = idx_row[a0]; } else { cav = b0v; cai = idx_row[b0]; }
            float cbv;
            int   cbi;
            if (a1v >= b1v) { cbv = a1v; cbi = idx_row[a1]; } else { cbv = b1v; cbi = idx_row[b1]; }
            if (cav >= cbv) {
                val_row[a0] = cav; idx_row[a0] = cai;
                val_row[a1] = cbv; idx_row[a1] = cbi;
            } else {
                val_row[a0] = cbv; idx_row[a0] = cbi;
                val_row[a1] = cav; idx_row[a1] = cai;
            }
        }
        __syncthreads();

        // Remaining cross-wave merge strides (j = KPAD/4 .. 2*WARP; j == KPAD/2 was
        // fused above; j == WARP is peeled below as a register merge, so it is off
        // this range — guard j > WARP, not >=).
        //
        // iter-52: these strides all pair columns that differ only in the bits
        // [log2(2*WARP) .. log2(KPAD/4)], so the G = KPAD/(4*WARP) columns
        // { gbase+i, gbase+i+2W, ..., gbase+i+(G-1)*2W } form a group CLOSED under
        // every one of them. When there are >= 2 middle strides (G >= 4, i.e. KPAD in
        // {512,1024}) one owner lane per group (the low element, i & GMASK == 0)
        // gathers its G slots into registers, runs the whole sub-merge there
        // (group-index strides G/2..1 == full strides KPAD/4..2W), and writes the G
        // results back. This replaces the M per-stride block barriers AND the M-1
        // intermediate LDS write->readback round-trips with ONE barrier and zero
        // intermediate round-trips (same fuse class as iter-50, now spanning ALL the
        // middle strides at once). For k=400 (KPAD=512, strides {128,64}) that is
        // 2 barriers -> 1.
        //
        // Race-free with NO internal barrier: each owner exclusively owns its G
        // output slots (disjoint groups; GMASK-aligned bases tile [gbase,gbase+KPAD)
        // in steps of 2W), and it reads ONLY those same slots — all written by the
        // fused C-step above (published by the barrier at the top of this block), none
        // written by any other lane in THIS stage. Reads precede writes intra-thread.
        // Bit-exact to the stock loop: identical comparator (low index keeps the max,
        // swap only on strict <) applied in identical stride order; idx follows value.
        // For G <= 2 (KPAD in {128,256}: 0 or 1 middle stride, so no barrier to save)
        // keep the stock in-place loop byte-for-byte.
        constexpr int MID_G = KPAD / (4 * WARP);
        if (MID_G >= 4) {
            constexpr int GMASK = (MID_G - 1) * (2 * WARP);
            constexpr int GARR  = MID_G >= 4 ? MID_G : 1;  // avoid 0-length array in the dead (MID_G<4) instantiations
            if (active && (i & GMASK) == 0) {
                float gv[GARR];
                int   gi[GARR];
#pragma unroll
                for (int t = 0; t < MID_G; ++t) {
                    const int p = gbase + i + t * (2 * WARP);
                    gv[t] = val_row[p];
                    gi[t] = idx_row[p];
                }
#pragma unroll
                for (int s = MID_G >> 1; s > 0; s >>= 1) {
#pragma unroll
                    for (int a = 0; a < MID_G; ++a) {
                        const int b = a ^ s;
                        if (b > a && gv[a] < gv[b]) {
                            topk_swap(gv[a], gv[b]);
                            topk_swap(gi[a], gi[b]);
                        }
                    }
                }
#pragma unroll
                for (int t = 0; t < MID_G; ++t) {
                    const int p = gbase + i + t * (2 * WARP);
                    val_row[p] = gv[t];
                    idx_row[p] = gi[t];
                }
            }
            __syncthreads();
        } else {
#pragma unroll
            for (int j = KPAD >> 2; j > WARP; j >>= 1) {
                const int partner = col ^ j;
                if (active && partner > col) {
                    if (val_row[col] < val_row[partner]) {
                        topk_swap(val_row[col], val_row[partner]);
                        topk_swap(idx_row[col], idx_row[partner]);
                    }
                }
                __syncthreads();
            }
        }

        // Fused last cross-wave stride (j == WARP) + intra-wave merge tail
        // (j = WARP/2 .. 1) via xor shuffle. Merge partners (col ^ j, j < KPAD)
        // stay within the aligned head KPAD-block, so an active lane's partner is
        // also active; inactive lanes shuffle harmlessly but never write back.
        // Uniform descending merge: low lane keeps the max.
        //
        // j == WARP is the boundary between cross-wave (partner in a sibling wave,
        // via LDS) and intra-wave (via xor shuffle) strides. The stock loop above
        // did it as an in-place LDS swap (lower lane reads+writes both slots) gated
        // by a __syncthreads(), then the tail RELOADED val_row[col] to start the
        // shuffle chain. Instead, each lane reads its own slot + its sibling-wave
        // partner's slot (both synced by the last cross-wave barrier) and computes
        // its OWN merge result in registers: low lane keeps the max, high lane the
        // min — byte-for-byte what the swap produced (low slot <- max, high <- min).
        // No lane writes LDS during this stride, so the barrier after it and the
        // tail's reload both vanish: the whole tail runs register-resident straight
        // off the register merge. On the barrier-bound k=400 shared path (jstop=16
        // → the tail is a single j=16 stage) this removes one full block barrier
        // and the LDS write+readback round-trip per Phase-B level — the same
        // barrier/round-trip elision class iter-44 won with, now on the peeled
        // cross-wave/intra-wave boundary stage.
        //
        // k-aligned early termination on the OUTPUT (last) level: a bitonic merge
        // completed down to stride s leaves the window rank-sorted into size-s
        // blocks (block m holds ranks [m*s, (m+1)*s)); the smaller-stride stages
        // only reorder WITHIN a block. TOP_K's result is an unordered SET, so once
        // k is a multiple of the current stride s, positions [0, k) already hold
        // exactly the top-k set and the j < s stages are pure dead reordering of
        // in-set / out-of-set elements. Stop at jstop = the largest power of two
        // dividing k (k & -k). NON-last levels must stay fully sorted (their block
        // feeds the next level's C-step, which requires descending order), so jstop
        // is forced to 1 there. jstop is uniform across the block (k / last are
        // uniform), so the guard is a divergence-free branch and every lane runs
        // the surviving shuffle stages in lockstep. For the benchmark's k=400 case
        // jstop=16, cutting the last-level tail from 5 shuffle stages to 1.
        {
            // j == WARP cross-wave merge, register-resident (no LDS write, no
            // barrier). partner == col ^ WARP is in the sibling wave; its slot was
            // synced by the last cross-wave barrier above. low lane keeps the max,
            // high lane the min (uniform descending) — identical to the swap the
            // stock loop performed. idx_row[partner] is read only when the partner
            // wins (lazy cross-wave index read).
            float mv;
            int   mid;
            {
                const int   partner = col ^ WARP;
                const float sv      = val_row[col];
                const float pv      = val_row[partner];
                const bool  low     = (col & WARP) == 0;
                const bool  take    = low ? (sv < pv) : (sv > pv);
                mv  = take ? pv : sv;
                mid = take ? idx_row[partner] : idx_row[col];
            }
            const int jstop = last ? (k & (-k)) : 1;
#pragma unroll
            for (int j = WARP >> 1; j > 0; j >>= 1) {
                if (j < jstop) {
                    continue;
                }
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

        // C-step: C[i] = max(X[i], Y[KPAD-1-i]). Keep the winner in registers and
        // feed the intra-wave merge below directly, instead of storing it to
        // val_row[gbase+i] and immediately reloading val_row[col]. On this path the
        // merge is fully intra-wave xor shuffle (KPAD <= warpSize) and touches no
        // LDS, and no other lane reads gbase+i before the post-merge writeback
        // overwrites it (merge partners col^j, j<KPAD, stay in the aligned head
        // KPAD-block and exchange via registers), so that store+reload was a dead
        // same-slot LDS round-trip sitting on the latency-bound C-step->merge chain.
        // Inactive lanes (i >= KPAD) keep their prior registers; their shuffle
        // outputs never feed an active lane (an active lane's partner is always
        // active), so their contents are irrelevant.
        if (active) {
            // a_pos == gbase + i == col: the lane's OWN slot, which still holds the
            // register-resident (v, id). Every active lane wrote val_row[col]=v /
            // idx_row[col]=id at the previous level's writeback (Phase A for the
            // first level) and has not mutated v/id since — the merge below feeds
            // straight back into (v, id) (iter-33), and the Phase-B active set only
            // shrinks (col mod 4span >= col mod 2span, so once inactive always
            // inactive), so an active lane was active at every prior level and its
            // own-slot writeback chain kept val_row[col] === v. Only the cross-half
            // b-side lives in another (inactive) lane's slot, so only it is loaded
            // from LDS; the a-side reload of the lane's own just-written value was a
            // dead same-slot LDS round-trip sitting on the latency-bound
            // C-step->merge chain. Tie policy is unchanged: b wins only on a strict
            // bv > v (== the old `av >= bv` keeping a on ties).
            const int   b_pos = gbase + span + (KPAD - 1 - i);
            const float bv    = val_row[b_pos];
            if (bv > v) {
                v  = bv;
                id = idx_row[b_pos];
            }
        }

        // Bitonic merge the KPAD-wide block back to descending order. KPAD <=
        // warpSize, so the whole merge is intra-wave xor shuffle over the
        // register-resident C-step winner (no LDS reload, no barrier).
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
                // Decide the exchange from the value read (pv) — no barrier needed
                // for a register compare — so the partner-index LDS read below is
                // issued only on the lanes that actually swap. On a non-swapping
                // lane (~half, per bitonic stage) idx_row[partner] is never touched,
                // trimming one ds_read from that lane's cross-wave critical path.
                const bool  low      = (col & j) == 0;
                const float low_val  = low ? v : pv;
                const float high_val = low ? pv : v;
                const bool  do_swap  = up ? (low_val < high_val) : (low_val > high_val);
                const int   pi       = do_swap ? idx_row[partner] : id;
                __syncthreads();  // all cross-wave reads done before LDS reuse
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
    // No barrier needed before this writeback: Phase A has exactly one cross-wave
    // exchange (len == KPAD, j == WARP), and its post-read barrier (the second
    // __syncthreads() in the j >= WARP branch above) already guarantees every
    // block-wide LDS read of that exchange has completed. Every stage after it
    // (j = WARP/2 .. 1) is an intra-wave register xor shuffle that never touches
    // LDS, so nothing can still be reading val_row/idx_row when we overwrite each
    // lane's own slot here. Only the post-writeback barrier is required (to order
    // the writeback ahead of Phase B's first C-step read).
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
            // a-side reload elision (iter-36, the iter-35 lever applied to 2wave):
            // a_pos == gbase + i == col is the lane's OWN slot, and the registers
            // (v, id) already mirror val_row[col]. After Phase A every lane wrote
            // val_row[col] = v; at each Phase-B level the merge-tail writeback below
            // stores (mv, mid) to val_row[col] AND refreshes (v, id) = (mv, mid), and
            // the trailing __syncthreads() is the only thing between that store and
            // this C-step read — with no mutation of v/id in between — so
            // val_row[col] === v holds here. The Phase-B active set only SHRINKS
            // (active ⇔ (col mod 2span) < KPAD, and col mod 4span >= col mod 2span, so
            // once inactive always inactive), hence an active lane was active at every
            // prior level and its own-slot mirror chain is unbroken. Only the
            // cross-half b-side lives in another lane's slot, so only it is loaded
            // from LDS; the a-side own-slot reload was a dead round-trip on the
            // latency-bound C-step->merge chain. The C-step OUTPUT store stays: the
            // cross-wave stage below reads val_row[col] via the sibling (partner)
            // lane. Tie policy unchanged: `bv > v` keeps a on ties (== old av >= bv).
            const int   b_pos = gbase + span + (KPAD - 1 - i);
            const float bv    = val_row[b_pos];
            if (bv > v) {
                v  = bv;
                id = idx_row[b_pos];
            }
            val_row[gbase + i] = v;
            idx_row[gbase + i] = id;
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
                // Refresh the own-slot register mirror so next level's C-step can
                // take its a-side from (v, id) instead of reloading val_row[col].
                v  = mv;
                id = mid;
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
    // Double-buffered Phase-A LDS window (iter-51): two ncols_pad-wide (idx,val)
    // regions instead of one. ncols <= 1024 on this path, so this is <= 16 KB.
    const size_t shared_mem = 2 * ncols_pad * (sizeof(int) + sizeof(float));

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
#elif defined(GGML_CUDA_USE_CUB) || defined(GGML_CUDA_USE_HIPCUB)  // CUB_TOP_K_AVAILABLE
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
