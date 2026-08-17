#include "concat.cuh"

#include <atomic>

static std::atomic<size_t> g_concat_transpose_f32_count{0};

extern "C" size_t ggml_backend_cuda_get_concat_transpose_f32_count(void) {
    return g_concat_transpose_f32_count.load(std::memory_order_relaxed);
}

// Fast path for concatenating a short contiguous prefix with a dense
// dim-0/dim-1 transpose.  The motivating layout is the SSM convolution input:
//
//   src0 [prefix, channels, sequences] (contiguous)
//   src1 [tokens, channels, sequences]  (a view of contiguous
//                                         [channels, tokens, sequences])
//   dst  [prefix + tokens, channels, sequences] (contiguous)
//
// concat_non_cont assigns one block to each channel and has the threads in a
// wave read adjacent logical tokens.  For src1 those reads are `channels`
// elements apart.  Transpose a 32x32 tile through shared memory instead so
// both the source reads and destination writes are coalesced.  Copy the small
// prefix in the token-tile-zero blocks to avoid a second launch.
constexpr int CUDA_CONCAT_TRANSPOSE_TILE = 32;
constexpr int CUDA_CONCAT_TRANSPOSE_ROWS = 8;

static __global__ void __launch_bounds__(CUDA_CONCAT_TRANSPOSE_TILE * CUDA_CONCAT_TRANSPOSE_ROWS)
concat_dim0_dense_transpose_f32(
        const float * __restrict__ src0,
        const float * __restrict__ src1,
              float * __restrict__ dst,
        int64_t prefix,
        int64_t tokens,
        int64_t channels) {
    __shared__ float tile[CUDA_CONCAT_TRANSPOSE_TILE][CUDA_CONCAT_TRANSPOSE_TILE + 1];

    const int64_t channel_base = (int64_t) blockIdx.x * CUDA_CONCAT_TRANSPOSE_TILE;
    const int64_t token_base   = (int64_t) blockIdx.y * CUDA_CONCAT_TRANSPOSE_TILE;
    const int64_t sequence     = blockIdx.z;
    const int64_t channel      = channel_base + threadIdx.x;

    const int64_t src0_sequence_stride = prefix * channels;
    const int64_t src1_sequence_stride = tokens * channels;
    const int64_t dst_sequence_stride  = (prefix + tokens) * channels;

    const float * src0_sequence = src0 + sequence * src0_sequence_stride;
    const float * src1_sequence = src1 + sequence * src1_sequence_stride;
    float * dst_sequence = dst + sequence * dst_sequence_stride;

    // src0 is small in the intended use (three Qwen conv-history rows).
    // Flatten its portion of this channel tile across the whole workgroup.
    if (blockIdx.y == 0) {
        const int64_t channels_in_tile =
            channels - channel_base < CUDA_CONCAT_TRANSPOSE_TILE
                ? channels - channel_base
                : CUDA_CONCAT_TRANSPOSE_TILE;
        const int thread = threadIdx.y * CUDA_CONCAT_TRANSPOSE_TILE + threadIdx.x;
        const int n_threads = CUDA_CONCAT_TRANSPOSE_TILE * CUDA_CONCAT_TRANSPOSE_ROWS;
        for (int64_t i = thread; i < channels_in_tile * prefix; i += n_threads) {
            const int64_t local_channel = i / prefix;
            const int64_t prefix_row    = i - local_channel * prefix;
            const int64_t dst_channel   = channel_base + local_channel;
            dst_sequence[dst_channel * (prefix + tokens) + prefix_row] =
                src0_sequence[dst_channel * prefix + prefix_row];
        }
    }

    // src1's physical layout is [token][channel], so waves read channels.
    for (int local_token = threadIdx.y;
         local_token < CUDA_CONCAT_TRANSPOSE_TILE;
         local_token += CUDA_CONCAT_TRANSPOSE_ROWS) {
        const int64_t token = token_base + local_token;
        if (channel < channels && token < tokens) {
            tile[local_token][threadIdx.x] =
                src1_sequence[token * channels + channel];
        }
    }
    __syncthreads();

    // Swap the in-tile coordinates.  The +1 shared-memory pitch avoids bank
    // conflicts for both CUDA warp32 and RDNA wave32 execution.
    const int64_t dst_token = token_base + threadIdx.x;
    for (int local_channel = threadIdx.y;
         local_channel < CUDA_CONCAT_TRANSPOSE_TILE;
         local_channel += CUDA_CONCAT_TRANSPOSE_ROWS) {
        const int64_t dst_channel = channel_base + local_channel;
        if (dst_channel < channels && dst_token < tokens) {
            dst_sequence[dst_channel * (prefix + tokens) + prefix + dst_token] =
                tile[threadIdx.x][local_channel];
        }
    }
}

static bool concat_dim0_dense_transpose_f32_cuda(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst,
        int dim) {
    constexpr size_t element_size = sizeof(float);

    const int64_t prefix    = src0->ne[0];
    const int64_t tokens    = src1->ne[0];
    const int64_t channels  = dst->ne[1];
    const int64_t sequences = dst->ne[2];

    // Keep this a strict layout specialization.  Anything padded, permuted in
    // more than the first two dimensions, unusually large, or non-F32 falls
    // through to the existing fully general implementation.
    const bool supported =
        dim == 0 &&
        src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        prefix > 0 && prefix <= CUDA_CONCAT_TRANSPOSE_TILE &&
        tokens > 0 && channels > 0 && sequences > 0 &&
        channels <= (int64_t) CUDA_CONCAT_TRANSPOSE_TILE * 65535 &&
        tokens   <= (int64_t) CUDA_CONCAT_TRANSPOSE_TILE * 65535 &&
        sequences <= 65535 &&
        src0->ne[1] == channels && src1->ne[1] == channels &&
        src0->ne[2] == sequences && src1->ne[2] == sequences &&
        src0->ne[3] == 1 && src1->ne[3] == 1 && dst->ne[3] == 1 &&
        dst->ne[0] == prefix + tokens &&
        src0->nb[0] == element_size &&
        src0->nb[1] == (size_t) prefix * element_size &&
        src0->nb[2] == (size_t) prefix * channels * element_size &&
        src1->nb[1] == element_size &&
        src1->nb[0] == (size_t) channels * element_size &&
        src1->nb[2] == (size_t) tokens * channels * element_size &&
        dst->nb[0] == element_size &&
        dst->nb[1] == (size_t) (prefix + tokens) * element_size &&
        dst->nb[2] == (size_t) (prefix + tokens) * channels * element_size;

    if (!supported) {
        return false;
    }

    const dim3 block(CUDA_CONCAT_TRANSPOSE_TILE, CUDA_CONCAT_TRANSPOSE_ROWS, 1);
    const dim3 grid(
        (channels + CUDA_CONCAT_TRANSPOSE_TILE - 1) / CUDA_CONCAT_TRANSPOSE_TILE,
        (tokens   + CUDA_CONCAT_TRANSPOSE_TILE - 1) / CUDA_CONCAT_TRANSPOSE_TILE,
        sequences);
    concat_dim0_dense_transpose_f32<<<grid, block, 0, ctx.stream()>>>(
        (const float *) src0->data,
        (const float *) src1->data,
        (float *) dst->data,
        prefix, tokens, channels);
    g_concat_transpose_f32_count.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// contiguous kernels
template <typename T>
static __global__ void concat_dim0(const T * x, const T * y, T * dst, const int ne0, const int ne00) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (nidx < ne00) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne00 +
            blockIdx.z * ne00 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            (nidx - ne00) +
            blockIdx.y * (ne0 - ne00) +
            blockIdx.z * (ne0 - ne00) * gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

template <typename T>
static __global__ void concat_dim1(const T * x, const T * y, T * dst, const int ne0, const int ne01) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.y < (unsigned)ne01) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * ne01;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            (blockIdx.y - ne01) * ne0 +
            blockIdx.z * ne0 * (gridDim.y - ne01);
        dst[offset_dst] = y[offset_src];
    }
}

template <typename T>
static __global__ void concat_dim2(const T * x, const T * y, T * dst, const int ne0, const int ne02) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.z < (unsigned)ne02) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            (blockIdx.z - ne02) * ne0 *  gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

template <typename T>
static void concat_cuda(const T * x, const T * y, T * dst, int ne00, int ne01, int ne02, int ne0, int ne1, int ne2, int dim, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2);
    if (dim == 0) {
        concat_dim0<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne00);
        return;
    }
    if (dim == 1) {
        concat_dim1<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne01);
        return;
    }
    concat_dim2<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne02);
}

// non-contiguous kernel (slow)
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3){
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const T * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const T *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const T *) (src1 + i3 * nb13 + i2 * nb12 + i1 * nb11 + (i0 - ne00) * nb10);
            } else if constexpr (dim == 1) {
                x = (const T *) (src1 + i3 * nb13 + i2 * nb12 + (i1 - ne01) * nb11 + i0 * nb10);
            } else if constexpr (dim == 2) {
                x = (const T *) (src1 + i3 * nb13 + (i2 - ne02) * nb12 + i1 * nb11 + i0 * nb10);
            } else if constexpr (dim == 3) {
                x = (const T *) (src1 + (i3 - ne03) * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);
            }
        }

        T * y = (T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

template <typename T>
static void concat_cuda_typed(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int dim) {
    cudaStream_t stream = ctx.stream();

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const T * src0_d = (const T *)src0->data;
        const T * src1_d = (const T *)src1->data;
        T * dst_d = (T *)dst->data;

        if (dim != 3) {
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_cuda(
                        src0_d + i3 * (src0->nb[3] / sizeof(T)),
                        src1_d + i3 * (src1->nb[3] / sizeof(T)),
                        dst_d + i3 * ( dst->nb[3] / sizeof(T)),
                        src0->ne[0], src0->ne[1], src0->ne[2],
                        dst->ne[0],  dst->ne[1],  dst->ne[2], dim, stream);
            }
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            CUDA_CHECK(cudaMemcpyAsync(dst_d,                         src0_d, size0, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync((char *)dst_d + size0,         src1_d, size1, cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            concat_non_cont<T, dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
                break;
        }
    }
}

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(src0->type == dst->type);

    switch (src0->type) {
        case GGML_TYPE_F32:
            // Check before the generic contiguous branch: when tokens == 1,
            // ggml correctly considers the transpose view contiguous, but the
            // legacy dim-0 kernel still launches one mostly-idle block per
            // channel and sequence.
            if (!concat_dim0_dense_transpose_f32_cuda(ctx, src0, src1, dst, dim)) {
                concat_cuda_typed<float>(ctx, src0, src1, dst, dim);
            }
            break;
        case GGML_TYPE_F16:
            concat_cuda_typed<half>(ctx, src0, src1, dst, dim);
            break;
        case GGML_TYPE_BF16:
            concat_cuda_typed<nv_bfloat16>(ctx, src0, src1, dst, dim);
            break;
        case GGML_TYPE_I8:
            concat_cuda_typed<int8_t>(ctx, src0, src1, dst, dim);
            break;
        case GGML_TYPE_I32:
            concat_cuda_typed<int32_t>(ctx, src0, src1, dst, dim);
            break;
        default:
            GGML_ABORT("unsupported concat type %s", ggml_type_name(src0->type));
    }
}
