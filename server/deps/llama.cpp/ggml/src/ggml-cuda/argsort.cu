#include "argsort.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 1)
#        define STRIDED_ITERATOR_AVAILABLE
#    endif
using namespace cub;
#elif defined(GGML_CUDA_USE_HIPCUB)
// hipCUB exposes the CUB device-sort API (DeviceRadixSort / DeviceSegmentedSort /
// DeviceSegmentedRadixSort) over rocPRIM. No strided-iterator / CCCL support, so
// STRIDED_ITERATOR_AVAILABLE stays undefined and the init_offsets path is used.
#    include <hipcub/hipcub.hpp>
using namespace hipcub;
#endif  // GGML_CUDA_USE_CUB

static __global__ void init_indices(int * indices, const int ncols, const int nrows) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;

    if (col < ncols && row < nrows) {
        indices[row * ncols + col] = col;
    }
}

#ifndef STRIDED_ITERATOR_AVAILABLE
static __global__ void init_offsets(int * offsets, const int ncols, const int nrows) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= nrows) {
        offsets[idx] = idx * ncols;
    }
}
#endif  // STRIDED_ITERATOR_AVAILABLE

#if defined(GGML_CUDA_USE_CUB) || defined(GGML_CUDA_USE_HIPCUB)
void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                              const float *    x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              ggml_sort_order  order,
                              cudaStream_t     stream) {
    ggml_cuda_pool_alloc<int>   temp_indices_alloc(pool, ncols * nrows);
    ggml_cuda_pool_alloc<float> temp_keys_alloc(pool, ncols * nrows);

    int *   temp_indices = temp_indices_alloc.get();
    float * temp_keys    = temp_keys_alloc.get();

    static const int block_size = 256;
    const dim3 grid_size((ncols + block_size - 1) / block_size, nrows);
    init_indices<<<grid_size, block_size, 0, stream>>>(temp_indices, ncols, nrows);

#ifdef STRIDED_ITERATOR_AVAILABLE
    auto offset_iterator = cuda::make_strided_iterator(cuda::make_counting_iterator(0), ncols);
#else
    // offset_iterator needs to populate nrows + 1 elements, so we also have to ceildiv nrows + 1 by block_size
    const int                 nrows_offset = nrows + 1;
    ggml_cuda_pool_alloc<int> offsets_alloc(pool, nrows_offset);
    int *                     offset_iterator = offsets_alloc.get();
    const dim3                offset_grid((nrows_offset + block_size - 1) / block_size);
    init_offsets<<<offset_grid, block_size, 0, stream>>>(offset_iterator, ncols, nrows);
#endif
    CUDA_CHECK(cudaMemcpyAsync(temp_keys, x, ncols * nrows * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    size_t temp_storage_bytes = 0;

    bool is_capturing = false;
#ifdef USE_CUDA_GRAPH
    // Currently (confirmed for CCCL <= 3.2) DeviceSegmentedSort does not support stream capture, while DeviceSegmentedRadixSort does.
    // See https://github.com/NVIDIA/cccl/issues/5661#issuecomment-3229037149
    // TODO: constrain this to the CCCL versions that have this issue once it's resolved in a future CCCL release.
    cudaStreamCaptureStatus capture_status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    is_capturing = (capture_status != cudaStreamCaptureStatusNone);
#endif  // USE_CUDA_GRAPH

    if (order == GGML_SORT_ORDER_ASC) {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, temp_keys, temp_keys,  // keys (in-place)
                                                  temp_indices, dst,  // values (indices)
                                                  ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairs(
                nullptr, temp_storage_bytes, temp_keys, temp_keys,  // keys (in-place)
                temp_indices, dst,                                  // values (indices)
                ncols * nrows, nrows,                               // num items, num segments
                offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairs(nullptr, temp_storage_bytes, temp_keys,
                                                      temp_keys,             // keys (in-place)
                                                      temp_indices, dst,     // values (indices)
                                                      ncols * nrows, nrows,  // num items, num segments
                                                      offset_iterator, offset_iterator + 1, stream));
        }
    } else {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_bytes, temp_keys,
                                                            temp_keys,          // keys (in-place)
                                                            temp_indices, dst,  // values (indices)
                                                            ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairsDescending(
                nullptr, temp_storage_bytes, temp_keys, temp_keys, temp_indices, dst, ncols * nrows, nrows,
                offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairsDescending(nullptr, temp_storage_bytes, temp_keys, temp_keys,
                                                                temp_indices, dst, ncols * nrows, nrows,
                                                                offset_iterator, offset_iterator + 1, stream));
        }
    }

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    if (order == GGML_SORT_ORDER_ASC) {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys,
                                                  temp_keys,          // keys (in-place)
                                                  temp_indices, dst,  // values (indices)
                                                  ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
                                                           temp_indices, dst, ncols * nrows, nrows, offset_iterator,
                                                           offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairs(d_temp_storage, temp_storage_bytes, temp_keys, temp_keys,
                                                      temp_indices, dst, ncols * nrows, nrows, offset_iterator,
                                                      offset_iterator + 1, stream));
        }
    } else {
        if (nrows == 1) {
            CUDA_CHECK(DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, temp_keys,
                                                            temp_keys,          // keys (in-place)
                                                            temp_indices, dst,  // values (indices)
                                                            ncols, 0, sizeof(float) * 8, stream));
        } else if (is_capturing) {
            CUDA_CHECK(DeviceSegmentedRadixSort::SortPairsDescending(
                d_temp_storage, temp_storage_bytes, temp_keys, temp_keys, temp_indices, dst, ncols * nrows, nrows,
                offset_iterator, offset_iterator + 1, 0, sizeof(float) * 8, stream));
        } else {
            CUDA_CHECK(DeviceSegmentedSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, temp_keys,
                                                                temp_keys, temp_indices, dst, ncols * nrows, nrows,
                                                                offset_iterator, offset_iterator + 1, stream));
        }
    }
}
#endif  // GGML_CUDA_USE_CUB || GGML_CUDA_USE_HIPCUB

// Bitonic sort implementation
template<typename T>
static inline __device__ void ggml_cuda_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

// Warp-local xor shuffle. For the bitonic inner stages where the stride j is
// smaller than the wavefront width, both partners of a compare-exchange live in
// the same wave, so the exchange can be done register-to-register via shuffle —
// no shared-memory round-trip and no __syncthreads() barrier.
#if defined(GGML_USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#    define GGML_ARGSORT_SHFL_XOR(v, mask) __shfl_xor((v), (mask))
#else
#    define GGML_ARGSORT_SHFL_XOR(v, mask) __shfl_xor_sync(0xffffffffu, (v), (mask))
#endif

template<ggml_sort_order order>
static __global__ void k_argsort_f32_i32(const float * x, int * dst, const int ncols, int ncols_pad) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.x;

    if (col >= ncols_pad) {
        return;
    }

    const float * x_row = x + row * ncols;
    // Shared layout: [ncols_pad] indices followed by [ncols_pad] cached key
    // values. The bitonic network re-reads two keys per comparison across
    // ~log2(ncols_pad)^2 stages; caching the value that travels with each
    // index removes the repeated indirect global gathers from the inner loop.
    extern __shared__ int smem[];
    int *   dst_row = smem;
    float * val_row = (float *) (smem + ncols_pad);

    // initialize indices and cache the key each index points at (padding lanes
    // keep an untouched value; they are handled by the index-based checks below)
    dst_row[col] = col;
    if (col < ncols) {
        val_row[col] = x_row[col];
    }

    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        int j = k / 2;

        // Cross-wave stages (stride >= warpSize): partners live in different
        // waves, so the exchange must go through shared memory with a full
        // block barrier between stages.
        for (; j >= warpSize; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            val_row[col] > val_row[ixj] :
                            val_row[col] < val_row[ixj]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                        ggml_cuda_swap(val_row[col], val_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            val_row[col] < val_row[ixj] :
                            val_row[col] > val_row[ixj]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                        ggml_cuda_swap(val_row[col], val_row[ixj]);
                    }
                }
            }
            __syncthreads();
        }

        // Intra-wave tail (stride < warpSize): partner is in the same wave.
        // Pull (idx, val) into registers and drive the remaining stages with
        // xor shuffles — no LDS traffic, no barriers. Both lanes of a pair
        // reconstruct the same (low, high) view and compute an identical swap
        // decision, so the exchange stays consistent. Semantics match the
        // shared-memory path above byte-for-byte (col == low lane, ixj == high).
        if (j > 0) {
            int   my_idx = dst_row[col];
            float my_val = val_row[col];
            for (; j > 0; j /= 2) {
                const float p_val = GGML_ARGSORT_SHFL_XOR(my_val, j);
                const int   p_idx = GGML_ARGSORT_SHFL_XOR(my_idx, j);

                const bool  low      = (col & j) == 0;
                const int   low_idx  = low ? my_idx : p_idx;
                const int   high_idx = low ? p_idx : my_idx;
                const float low_val  = low ? my_val : p_val;
                const float high_val = low ? p_val : my_val;

                bool swap;
                if ((col & k) == 0) {
                    swap = low_idx >= ncols ||
                           (high_idx < ncols && (order == GGML_SORT_ORDER_ASC ?
                                low_val > high_val :
                                low_val < high_val));
                } else {
                    swap = high_idx >= ncols ||
                           (low_idx < ncols && (order == GGML_SORT_ORDER_ASC ?
                                low_val < high_val :
                                low_val > high_val));
                }
                if (swap) {
                    my_idx = p_idx;
                    my_val = p_val;
                }
            }
            dst_row[col] = my_idx;
            val_row[col] = my_val;
            __syncthreads();
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
        dst[row * ncols + col] = dst_row[col];
    }
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

void argsort_f32_i32_cuda_bitonic(const float *   x,
                                  int *           dst,
                                  const int       ncols,
                                  const int       nrows,
                                  ggml_sort_order order,
                                  cudaStream_t    stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    // indices (int) + cached key values (float), both ncols_pad entries
    const size_t shared_mem = ncols_pad * (sizeof(int) + sizeof(float));

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_ASC>
            <<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_DESC>
            <<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else {
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

#if defined(GGML_CUDA_USE_CUB) || defined(GGML_CUDA_USE_HIPCUB)
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    if (shared_mem > max_shared_mem || ncols > 1024) {
        ggml_cuda_pool & pool = ctx.pool();
        argsort_f32_i32_cuda_cub(pool, src0_d, (int *) dst_d, ncols, nrows, order, stream);
    } else {
        argsort_f32_i32_cuda_bitonic(src0_d, (int *) dst_d, ncols, nrows, order, stream);
    }
#else
    argsort_f32_i32_cuda_bitonic(src0_d, (int *) dst_d, ncols, nrows, order, stream);
#endif
}
