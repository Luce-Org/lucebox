#include <algorithm>
#include <climits>
#include <cstdint>

#include "argmax.cuh"
#include "common.cuh"
#include "sum.cuh"

// The pick a host scan `if (x[i] > x[best]) best = i` from best = 0 makes: the lowest id of the row's maximum, -inf
// included, with NaNs never picked unless x[0] is one (then 0). Greedy decoding on the device then emits exactly the
// tokens the host scan would.
static __device__ __forceinline__ bool argmax_f32_better(const float val, const int col, const float maxval, const int argmax) {
    return val > maxval || (val == maxval && col < argmax);
}

static __global__ void argmax_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;

    float maxval = -INFINITY;
    int   argmax = INT_MAX;
    const float * rowx = x + row * ncols;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (argmax_f32_better(val, col, maxval, argmax)) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (argmax_f32_better(val, col, maxval, argmax)) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (argmax_f32_better(val, col, maxval, argmax)) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax == INT_MAX || isnan(rowx[0]) ? 0 : argmax;
    }
}

// Long rows (an LM head's logits: 248320 per verify token, 64000 per draft token) in two launches: blocks of a row each
// pick the best entry of one contiguous slice with the rule above, then one warp per row picks the best of those
// picks the same way. The rule is a total order on the entries it can pick (larger value, then lower index; a NaN is
// never picked), so the pick does not depend on how the row is cut: the same index as argmax_f32's one block per row,
// which reads the row with 1024 threads at a time and took 63 us for three verify rows.
constexpr int kArgmaxSliceThreads = 256;
constexpr int kArgmaxMaxSlices    = 64;
constexpr int64_t kArgmaxSliceCols = 2048;   // columns per slice, up to kArgmaxMaxSlices slices
constexpr int64_t kArgmaxSliceMinCols = 8192; // shorter rows keep one block per row

static __global__ void argmax_f32_slices(const float * __restrict__ x, float * __restrict__ pval, int * __restrict__ pcol,
                                         const int64_t ncols, const int nslices) {
    const int64_t row   = blockIdx.y;
    const int     slice = blockIdx.x;
    const int64_t per   = (ncols + nslices - 1) / nslices;
    const int64_t c0    = slice * per;
    const int64_t c1    = c0 + per < ncols ? c0 + per : ncols;
    const float * rowx  = x + row * ncols;

    float maxval = -INFINITY;
    int   argmax = INT_MAX;
    for (int64_t col = c0 + threadIdx.x; col < c1; col += blockDim.x) {
        const float val = rowx[col];
        if (argmax_f32_better(val, (int) col, maxval, argmax)) {
            maxval = val;
            argmax = (int) col;
        }
    }
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (argmax_f32_better(val, col, maxval, argmax)) {
            maxval = val;
            argmax = col;
        }
    }
    constexpr int n_warps = kArgmaxSliceThreads / WARP_SIZE;
    __shared__ float shared_maxval[n_warps];
    __shared__ int   shared_argmax[n_warps];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) {
        shared_maxval[warp_id] = maxval;
        shared_argmax[warp_id] = argmax;
    }
    __syncthreads();
    if (warp_id == 0) {
        maxval = lane_id < n_warps ? shared_maxval[lane_id] : -INFINITY;
        argmax = lane_id < n_warps ? shared_argmax[lane_id] : INT_MAX;
#pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
            const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
            if (argmax_f32_better(val, col, maxval, argmax)) {
                maxval = val;
                argmax = col;
            }
        }
        if (lane_id == 0) {
            pval[row * nslices + slice] = maxval;
            pcol[row * nslices + slice] = argmax;
        }
    }
}

static __global__ void argmax_f32_pick(const float * __restrict__ x, const float * __restrict__ pval,
                                       const int * __restrict__ pcol, int32_t * __restrict__ dst,
                                       const int64_t ncols, const int nslices) {
    const int64_t row = blockIdx.x;
    float maxval = -INFINITY;
    int   argmax = INT_MAX;
    for (int i = threadIdx.x; i < nslices; i += WARP_SIZE) {
        const float val = pval[row * nslices + i];
        const int   col = pcol[row * nslices + i];
        if (argmax_f32_better(val, col, maxval, argmax)) {
            maxval = val;
            argmax = col;
        }
    }
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (argmax_f32_better(val, col, maxval, argmax)) {
            maxval = val;
            argmax = col;
        }
    }
    if (threadIdx.x == 0) {
        dst[row] = argmax == INT_MAX || isnan(x[row * ncols]) ? 0 : argmax;
    }
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    if (ne00 >= kArgmaxSliceMinCols && ne00 <= INT_MAX && nrows <= 65535) {
        const int nslices = (int) std::min<int64_t>(kArgmaxMaxSlices, (ne00 + kArgmaxSliceCols - 1) / kArgmaxSliceCols);
        ggml_cuda_pool_alloc<float> pval(ctx.pool(), nrows * nslices);
        ggml_cuda_pool_alloc<int>   pcol(ctx.pool(), nrows * nslices);
        argmax_f32_slices<<<dim3(nslices, nrows, 1), kArgmaxSliceThreads, 0, stream>>>(src0_d, pval.get(), pcol.get(), ne00, nslices);
        argmax_f32_pick<<<nrows, WARP_SIZE, 0, stream>>>(src0_d, pval.get(), pcol.get(), dst_d, ne00, nslices);
        return;
    }

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);
}
