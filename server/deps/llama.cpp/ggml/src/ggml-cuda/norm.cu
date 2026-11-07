#include "norm.cuh"
#include "mmq.cuh"  // block_q8_1_mmq, mmq_q8_1_ds_layout (the q8_1 MMQ rows below)
#include "unary.cuh"  // ggml_cuda_op_silu_single
#include <climits>
#include <cmath>
#include <cstdint>

// The q8_1 MMQ rows of one finished
// norm output row, byte for byte what quantize_mmq_q8_1<ds_layout>
// (quantize.cu) writes for an MMQ src1 of ne1 contiguous rows in one channel
// (no ids, no groups). `x` is the F32 row: ne00 values (a multiple of 4),
// padded with zeros to ne0 (a multiple of MATRIX_ROW_PADDING); i1 is its row.
//
// The loop body is quantize_mmq_q8_1's, statement for statement:
//   - each thread takes 4 consecutive values, and lane l of a warp takes the
//     values 4l..4l+3 of a 128-value block, as there (block_size is a
//     multiple of the warp, so every warp covers one whole block per
//     iteration and, 128 dividing ne0, is all inside ne0 or all past it);
//   - amax is the same fmaxf chain, then the same xor shuffles;
//   - the DS4 / D2S6 sum is x+y+z+w, then the same xor shuffles in the same
//     order;
//   - d_inv = 127/amax, q = roundf(x*d_inv), d = 1/d_inv, stored at the same
//     block, lane and scale slot.
// Its early `return`s are `continue`s here (one 4-value group per
// iteration). The caller wrote the row and synchronized the block first: the
// values are read back from memory, as quantize_mmq_q8_1 reads them, so the
// norm's products cannot be contracted into these sums.
template <int block_size, mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void norm_row_q8_1_mmq(
        const float * x, block_q8_1_mmq * y, const int64_t i1, const int64_t ne1,
        const int64_t ne00, const int64_t ne0) {
    static_assert(block_size % WARP_SIZE == 0, "a warp must take whole 128-value blocks");

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const float4 * x4 = (const float4 *) x;

    for (int64_t i0 = 4*(int64_t) threadIdx.x; i0 < ne0; i0 += 4*block_size) {
        const int64_t ib  = (i0 / (4*QK8_1))*ne1 + i1; // block index in channel
        const int64_t iqs = i0 % (4*QK8_1);            // quant index in block

        // Load 4 floats per thread and calculate max. abs. value between them.
        const float4 xi = i0 < ne00
            ? x4[i0/4]
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float amax = fabsf(xi.x);
        amax = fmaxf(amax, fabsf(xi.y));
        amax = fmaxf(amax, fabsf(xi.z));
        amax = fmaxf(amax, fabsf(xi.w));

        // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
        for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
        }

        float sum;
        if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
            sum = xi.x + xi.y + xi.z + xi.w;

            // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
            for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
                sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
            }
        }

        const float d_inv = 127.0f / amax;
        char4 q;
        q.x = roundf(xi.x*d_inv);
        q.y = roundf(xi.y*d_inv);
        q.z = roundf(xi.z*d_inv);
        q.w = roundf(xi.w*d_inv);

        // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
        char4 * yqs4 = (char4 *) y[ib].qs;
        yqs4[iqs/4] = q;

        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
            if (iqs % 16 != 0 || iqs >= 96) {
                continue;
            }

            y[ib].d2s6[2 + iqs/16] = sum;

            if (iqs % 64 != 0) {
                continue;
            }

            const float d = 1.0f / d_inv;

            y[ib].d2s6[iqs/64] = d;

            continue;
        }

        if (iqs % 32 != 0) {
            continue;
        }

        const float d = 1.0f / d_inv;

        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y[ib].ds4[iqs/32] = make_half2(d, sum);
        } else {
            y[ib].d4[iqs/32]  = d;
        }
    }
}

#if defined(GGML_HIP_DS4V_VISION)
bool ggml_hip_vision_norm_capable(int device) {
    const auto & info = ggml_cuda_info();
    return device >= 0 && device < info.device_count && info.devices[device].warp_size == 32;
}

bool ggml_hip_vision_norm_supported(int device, const ggml_tensor * op) {
    if (!ggml_hip_vision_norm_capable(device) || !op || op->op != GGML_OP_RMS_NORM_VISION_F32 ||
        !op->src[0] || op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) ||
        !ggml_are_same_shape(op, op->src[0]) || op->ne[0] != 1024 || op->ne[1] < 16 ||
        op->ne[1] > INT_MAX/1024 || op->ne[2] != 1 || op->ne[3] != 1) {
        return false;
    }
    float eps;
    memcpy(&eps, op->op_params, sizeof(eps));
    return std::isfinite(eps) && eps >= 0.0f;
}

// PyTorch 3d3aa833db84eed6b7f5595cb5f162c2f78300a4, aten/src/ATen/native/cuda/
// Reduce.cuh:499-557,655-668: four input accumulators and ascending ROCm shuffles.
// With contiguous width1024 and >=16 rows the source uses one wave32 per row.
static __global__ void rms_norm_vision_f32(const float * x, float * dst, int rows, float eps) {
    const int row = blockIdx.x * 16 + threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= rows) {
        return;
    }
    float a[4] = {0, 0, 0, 0};
    for (int k = 0; k < 8; ++k) {
        for (int j = 0; j < 4; ++j) {
            const float xi = x[row*1024 + 4*(lane + 32*k) + j];
            volatile float squared = xi*xi; // source materializes F32 square before mean
            a[j] = a[j] + squared;
        }
    }
    float sum = ((a[0] + a[1]) + a[2]) + a[3];
    for (int offset = 1; offset < 32; offset *= 2) {
        sum = sum + __shfl_down(sum, offset, 32);
    }
    sum = __shfl(sum, 0, 32);
    const float mean = sum * (1.0f/1024.0f);
    // Same pinned Torch UnaryOpsKernel.cu:78-80 calls global ::rsqrt. Its ROCm
    // overload promotes float to double; keep the verified promotion explicitly.
    const float scale = static_cast<float>(::rsqrt(static_cast<double>(mean + eps)));
    for (int col = lane; col < 1024; col += 32) {
        dst[row*1024 + col] = x[row*1024 + col] * scale;
    }
}

void ggml_hip_vision_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_hip_vision_norm_supported(ctx.device, dst));
    float eps;
    memcpy(&eps, dst->op_params, sizeof(eps));
    const int rows = static_cast<int>(dst->ne[1]);
    rms_norm_vision_f32<<<(rows + 15)/16, dim3(32, 16), 0, ctx.stream()>>>(
        static_cast<const float *>(dst->src[0]->data), static_cast<float *>(dst->data), rows, eps);
    CUDA_CHECK(cudaGetLastError());
    ++ctx.vision_norm_launches;
}
#endif

template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float2 mean_var = make_float2(0.0f, 0.0f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (x[col] - mean) * inv_std;
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

// q8_mmq_layout: an mmq_q8_1_ds_layout, or -1. With a layout,
// `q8` holds block_q8_1_mmq rows (as many bytes per row as block_q8_1 rows)
// and ncols_q8 is ncols padded to MATRIX_ROW_PADDING: the finished row is
// also written in the form quantize_mmq_q8_1 gives it (norm_row_q8_1_mmq).
template <int block_size, bool do_multiply = false, bool do_add = false, bool do_q8 = false, int q8_mmq_layout = -1>
static __global__ void rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    block_q8_1 *  q8                   = nullptr,
                                    const int     ncols_q8             = 0) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    if constexpr (do_q8) {
        // The output's q8_1 form as quantize_q8_1 writes it for MMVQ (rows
        // padded with zero blocks to ncols_q8): a warp holds one 32-wide
        // block, reduced in the same butterfly order.
        static_assert(do_multiply && !do_add, "q8 output is fused with rms_norm * w only");
        block_q8_1 * yrow = q8 + (int64_t) ((sample*nchannels + channel)*nrows + row) * (ncols_q8 / QK8_1);
        for (int col = tid; col < ncols_q8; col += block_size) {
            float v = 0.0f;
            if (col < ncols) {
                const int mul_col = fastmodulo(col, mul_ncols_packed);
                v                 = scale * x[col] * mul[mul_col];
                dst[col]          = v;
            }
            float amax = fabsf(v);
            float sum  = v;
            amax = warp_reduce_max<QK8_1>(amax);
            sum  = warp_reduce_sum<QK8_1>(sum);
            const float  d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(v / d);
            yrow[col / QK8_1].qs[col % QK8_1] = q;
            if (col % QK8_1 == 0) {
                yrow[col / QK8_1].ds = make_half2(d, sum);
            }
        }
        return;
    }
    for (int col = tid; col < ncols; col += block_size) {
        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col] + add[add_col];
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col];
        } else {
            dst[col] = scale * x[col];
        }
    }

    if constexpr (q8_mmq_layout >= 0) {
        static_assert(do_multiply && !do_add && !do_q8, "q8_1 MMQ output is fused with rms_norm * w only");
        // Every thread's part of the row is written: read it back whole.
        __syncthreads();
        const int64_t i1  = ((int64_t) sample*nchannels + channel)*nrows + row;
        const int64_t ne1 = (int64_t) nrows*nchannels*gridDim.z;
        norm_row_q8_1_mmq<block_size, (mmq_q8_1_ds_layout) q8_mmq_layout>(
            dst, (block_q8_1_mmq *) q8, i1, ne1, ncols, ncols_q8);
    }
}

// Preserve the two-op RMS_NORM -> SCALE rounding, including the intermediate
// F32 product. Algebraically collapsing this into rsqrt(sum + eps) changes bits.
static __global__ void rms_norm_scale_f32(
        const float * x, float * dst, int ncols,
        int64_t stride_row, int64_t stride_channel, int64_t stride_sample,
        float eps, float post_scale, float bias) {
    x += blockIdx.z * stride_sample + blockIdx.y * stride_channel + blockIdx.x * stride_row;
    dst += ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * (int64_t) ncols;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < ncols; col += 128) {
        const float xi = x[col];
        sum += xi * xi;
    }
    // The 128-wide input leaves four zero warps in the original 256-thread
    // kernel. Omitting those warps preserves its two-stage reduction tree.
    extern __shared__ float s_sum[];
    sum = block_reduce<block_reduce_method::SUM, 128>(sum, s_sum);
    const float scale = rsqrtf(sum / ncols + eps);
    for (int col = threadIdx.x; col < ncols; col += 128) {
        const float normalized = __fmul_rn(scale, x[col]);
        dst[col] = post_scale * normalized + bias;
    }
}

extern "C" { GGML_BACKEND_API int ggml_cuda_test_rms_norm_scale(const float * x, float * dst, int ncols, int nrows,
        int nchannels, int nsamples, int64_t stride_row, int64_t stride_channel, int64_t stride_sample, float eps,
        float post_scale, float bias, void * stream) {
    if (!x || !dst || ncols <= 0 || nrows <= 0 || nchannels <= 0 || nsamples <= 0) return 0;
    rms_norm_scale_f32<<<dim3(nrows, nchannels, nsamples), 128, 32 * sizeof(float), (cudaStream_t) stream>>>(
        x, dst, ncols, stride_row, stride_channel, stride_sample, eps, post_scale, bias);
    return cudaGetLastError() == cudaSuccess;
} }

void ggml_cuda_op_rms_norm_scale(ggml_backend_cuda_context & ctx, ggml_tensor * norm, ggml_tensor * scale) {
    const ggml_tensor * x = norm->src[0];
    float eps, params[2];
    memcpy(&eps, norm->op_params, sizeof(eps));
    memcpy(params, scale->op_params, sizeof(params));
    const dim3 grid(x->ne[1], x->ne[2], x->ne[3]);
    rms_norm_scale_f32<<<grid, 128, 32 * sizeof(float), ctx.stream()>>>(
        (const float *) x->data, (float *) scale->data, x->ne[0],
        x->nb[1] / sizeof(float), x->nb[2] / sizeof(float), x->nb[3] / sizeof(float),
        eps, params[0], params[1]);
}

// dflash: residual add fused into the following rms_norm * weight.
//   sum = a + b            (written to sum_out; it is the next residual)
//   dst = rms_norm(sum) * w
// All of a, b, sum_out, dst are contiguous [ncols, R]; w is [ncols].
// q8_mmq_layout: as for rms_norm_f32; with a layout, the
// finished dst row is also written to q8 as block_q8_1_mmq rows (ncols_q8 is
// ncols padded to MATRIX_ROW_PADDING, the R rows one channel).
template <int block_size, int q8_mmq_layout = -1>
static __global__ void add_rms_norm_mul_f32(const float * __restrict__ a,
                                            const float * __restrict__ b,
                                            float *       __restrict__ sum_out,
                                            float *       __restrict__ dst,
                                            const float * __restrict__ w,
                                            const int     ncols,
                                            const float   eps,
                                            block_q8_1_mmq * __restrict__ q8 = nullptr,
                                            const int     ncols_q8 = 0) {
    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.x;

    a       += row * ncols;
    b       += row * ncols;
    sum_out += row * ncols;
    dst     += row * ncols;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float s = a[col] + b[col];
        sum_out[col] = s;
        tmp += s * s;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * sum_out[col] * w[col];
    }

    if constexpr (q8_mmq_layout >= 0) {
        // Every thread's part of the row is written: read it back whole.
        __syncthreads();
        norm_row_q8_1_mmq<block_size, (mmq_q8_1_ds_layout) q8_mmq_layout>(
            dst, q8, row, gridDim.x, ncols, ncols_q8);
    }
}

template <int block_size>
static void add_rms_norm_mul_q8_mmq_f32_cuda(const float * a, const float * b, float * sum_out, float * dst,
                                             const float * w, const int ncols, const int64_t nrows,
                                             const float eps, cudaStream_t stream,
                                             block_q8_1_mmq * q8, const int q8_mmq_layout) {
    const dim3 blocks_num(nrows, 1, 1);
    const dim3 block_dims(block_size, 1, 1);
    const int  ncols_q8 = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    switch (q8_mmq_layout) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            add_rms_norm_mul_f32<block_size, MMQ_Q8_1_DS_LAYOUT_D4><<<blocks_num, block_dims, 32 * sizeof(float), stream>>>(
                a, b, sum_out, dst, w, ncols, eps, q8, ncols_q8);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            add_rms_norm_mul_f32<block_size, MMQ_Q8_1_DS_LAYOUT_DS4><<<blocks_num, block_dims, 32 * sizeof(float), stream>>>(
                a, b, sum_out, dst, w, ncols, eps, q8, ncols_q8);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            add_rms_norm_mul_f32<block_size, MMQ_Q8_1_DS_LAYOUT_D2S6><<<blocks_num, block_dims, 32 * sizeof(float), stream>>>(
                a, b, sum_out, dst, w, ncols, eps, q8, ncols_q8);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// q8_mmq: optional, the q8_1 MMQ rows of dst in layout
// q8_mmq_layout (an mmq_q8_1_ds_layout); null keeps the plain kernels.
static void add_rms_norm_mul_f32_cuda(const float * a, const float * b, float * sum_out, float * dst,
                                      const float * w, const int ncols, const int64_t nrows,
                                      const float eps, cudaStream_t stream,
                                      void * q8_mmq = nullptr, const int q8_mmq_layout = -1) {
    if (q8_mmq != nullptr) {
        GGML_ASSERT(ncols % 4 == 0);
        if (ncols < 1024) {
            add_rms_norm_mul_q8_mmq_f32_cuda<256>(a, b, sum_out, dst, w, ncols, nrows, eps, stream,
                                                  (block_q8_1_mmq *) q8_mmq, q8_mmq_layout);
        } else {
            add_rms_norm_mul_q8_mmq_f32_cuda<1024>(a, b, sum_out, dst, w, ncols, nrows, eps, stream,
                                                   (block_q8_1_mmq *) q8_mmq, q8_mmq_layout);
        }
        return;
    }
    const dim3 blocks_num(nrows, 1, 1);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        add_rms_norm_mul_f32<256><<<blocks_num, block_dims, 32 * sizeof(float), stream>>>(a, b, sum_out, dst, w, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        add_rms_norm_mul_f32<1024><<<blocks_num, block_dims, 32 * sizeof(float), stream>>>(a, b, sum_out, dst, w, ncols, eps);
    }
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size>
static __global__ void l2_norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col];
    }
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

static void rms_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        rms_norm_f32<256, false><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_f32<1024, false><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

// RMS_NORM * w that also writes the q8_1 MMQ rows of dst: the
// MMQ buffer travels in rms_norm_f32's q8 argument, ncols_q8 the padded row.
template <int block_size>
static void rms_norm_mul_q8_mmq_f32_cuda(const dim3 &   blocks_num,
                                         const float *  x,
                                         const float *  mul,
                                         float *        dst,
                                         const int      ncols,
                                         const int64_t  stride_row,
                                         const int64_t  stride_channel,
                                         const int64_t  stride_sample,
                                         const int64_t  mul_stride_row,
                                         const int64_t  mul_stride_channel,
                                         const int64_t  mul_stride_sample,
                                         const uint3    mul_ncols_packed,
                                         const uint3    mul_nrows_packed,
                                         const uint3    mul_nchannels_packed,
                                         const uint3    mul_nsamples_packed,
                                         const float    eps,
                                         cudaStream_t   stream,
                                         void *         q8_mmq,
                                         const int      q8_mmq_layout) {
    const dim3   block_dims(block_size, 1, 1);
    const size_t smem     = block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0;
    const uint3  z        = make_uint3(0, 0, 0);
    const int    ncols_q8 = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    block_q8_1 * q8       = (block_q8_1 *) q8_mmq;
    switch (q8_mmq_layout) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            rms_norm_f32<block_size, true, false, false, MMQ_Q8_1_DS_LAYOUT_D4><<<blocks_num, block_dims, smem, stream>>>(
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                nullptr, 0, 0, 0, z, z, z, z, q8, ncols_q8);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            rms_norm_f32<block_size, true, false, false, MMQ_Q8_1_DS_LAYOUT_DS4><<<blocks_num, block_dims, smem, stream>>>(
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                nullptr, 0, 0, 0, z, z, z, z, q8, ncols_q8);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            rms_norm_f32<block_size, true, false, false, MMQ_Q8_1_DS_LAYOUT_D2S6><<<blocks_num, block_dims, smem, stream>>>(
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                nullptr, 0, 0, 0, z, z, z, z, q8, ncols_q8);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static void rms_norm_mul_f32_cuda(const float *  x,
                                  const float *  mul,
                                  const float *  add,
                                  float *        dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream,
                                  block_q8_1 *   q8 = nullptr,
                                  const int      ncols_q8 = 0,
                                  void *         q8_mmq = nullptr,
                                  const int      q8_mmq_layout = -1) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (mul == nullptr) {
        rms_norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (add == nullptr) {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        const uint3 z = make_uint3(0, 0, 0);
        if (q8_mmq != nullptr) {
            // The q8_1 MMQ rows instead of the MMVQ ones.
            GGML_ASSERT(q8 == nullptr && ncols % 4 == 0);
            if (ncols < 1024) {
                rms_norm_mul_q8_mmq_f32_cuda<256>(blocks_num, x, mul, dst, ncols, stride_row, stride_channel,
                    stride_sample, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed,
                    mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, eps, stream, q8_mmq, q8_mmq_layout);
            } else {
                rms_norm_mul_q8_mmq_f32_cuda<1024>(blocks_num, x, mul, dst, ncols, stride_row, stride_channel,
                    stride_sample, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed,
                    mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, eps, stream, q8_mmq, q8_mmq_layout);
            }
            return;
        }
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            if (q8) {
                rms_norm_f32<256, true, false, true><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(
                    x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                    mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                    nullptr, 0, 0, 0, z, z, z, z, q8, ncols_q8);
            } else {
                rms_norm_f32<256, true><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(
                    x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                    mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
            }
        } else {
            const dim3 block_dims(1024, 1, 1);
            if (q8) {
                rms_norm_f32<1024, true, false, true><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(
                    x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                    mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                    nullptr, 0, 0, 0, z, z, z, z, q8, ncols_q8);
            } else {
                rms_norm_f32<1024, true><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(
                    x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                    mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
            }
        }
    } else {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            rms_norm_f32<256, true, true><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed);
        } else {
            const dim3 block_dims(1024, 1, 1);
            rms_norm_f32<1024, true, true><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed);
        }
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

static void l2_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        l2_norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        l2_norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor,
                                 void * q8_out, int ncols_q8, void * q8_mmq, int q8_mmq_layout) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float * src0_d = (const float *) rms_norm_src->data;
    const float * mul_d = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_d = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    if (q8_mmq != nullptr) {
        // One channel of contiguous rows: the MMQ src1 the q8_1 rows are for.
        GGML_ASSERT(q8_out == nullptr && ggml_is_contiguous(mul_tensor) && ne02 == 1 && ne03 == 1 && ne00 % 4 == 0);
    }

    rms_norm_mul_f32_cuda(src0_d, mul_d, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ 0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream, (block_q8_1 *) q8_out, ncols_q8, q8_mmq, q8_mmq_layout);
}

// dflash: ADD (residual) + RMS_NORM + MUL in one launch. `add_tensor` is the
// residual add node (its output is materialized), `rms_tensor` is elided,
// `mul_tensor` receives the normalized * weight result.
void ggml_cuda_op_add_rms_norm_mul_fused(ggml_backend_cuda_context & ctx,
                                         ggml_tensor *               add_tensor,
                                         ggml_tensor *               rms_tensor,
                                         ggml_tensor *               mul_tensor,
                                         void *                      q8_mmq,
                                         int                         q8_mmq_layout) {
    const ggml_tensor * a = add_tensor->src[0];
    const ggml_tensor * b = add_tensor->src[1];
    const ggml_tensor * w = (mul_tensor->src[0] == rms_tensor) ? mul_tensor->src[1] : mul_tensor->src[0];

    float eps = 0.0f;
    memcpy(&eps, rms_tensor->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32 && mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(add_tensor) && ggml_is_contiguous(mul_tensor));
    GGML_ASSERT(ggml_are_same_shape(a, b) && ggml_are_same_shape(a, add_tensor) && ggml_are_same_shape(a, mul_tensor));
    GGML_ASSERT(w->ne[0] == a->ne[0] && ggml_nelements(w) == a->ne[0]);

    const int     ncols = (int) a->ne[0];
    const int64_t nrows = ggml_nrows(a);
    if (q8_mmq != nullptr) {
        // One channel of rows: the MMQ src1 the q8_1 rows are for.
        GGML_ASSERT(mul_tensor->ne[2] == 1 && mul_tensor->ne[3] == 1 && ncols % 4 == 0);
    }

    add_rms_norm_mul_f32_cuda((const float *) a->data, (const float *) b->data,
                              (float *) add_tensor->data, (float *) mul_tensor->data,
                              (const float *) w->data, ncols, nrows, eps, ctx.stream(), q8_mmq, q8_mmq_layout);
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float *       src0_d  = (const float *) rms_norm_src->data;
    const float *       mul_d   = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d   = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_d   = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const float *       add_d   = nullptr;
    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_d   = (float *) add_tensor->src[1]->data;
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_d   = (float *) add_tensor->src[0]->data;
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float *      dst_d  = (float *) add_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d,add_d,dst_d,
                          ne00,ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    l2_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

// Parity tests: the two norm producers on raw device pointers,
// with (q8 non-null, the layout of weight type q8_type) or without their
// q8_1 MMQ rows. kind 0: ADD + RMS_NORM + MUL (sum_out = a + b, dst =
// rms_norm(sum_out) * w); kind 1: RMS_NORM + MUL (dst = rms_norm(a) * w; b
// and sum_out unused). Rows are contiguous [ncols, nrows], w is [ncols].
// Synchronizes `stream`; 1 on success.
extern "C" { GGML_BACKEND_API int ggml_cuda_test_norm_q8_mmq(int kind, const float * a, const float * b,
        float * sum_out, float * dst, const float * w, int ncols, int nrows, float eps, void * q8, int q8_type,
        void * stream) {
    if (!a || !dst || !w || ncols <= 0 || nrows <= 0 || ncols % 4 != 0 || (kind == 0 && (!b || !sum_out)) ||
        (kind != 0 && kind != 1)) {
        return 0;
    }
    const int layout = q8 ? (int) mmq_get_q8_1_ds_layout((ggml_type) q8_type) : -1;
    cudaStream_t s = (cudaStream_t) stream;
    if (kind == 0) {
        add_rms_norm_mul_f32_cuda(a, b, sum_out, dst, w, ncols, nrows, eps, s, q8, layout);
    } else {
        const int64_t plane = (int64_t) ncols * nrows;
        rms_norm_mul_f32_cuda(a, w, nullptr, dst, ncols, nrows, 1, 1,
                              /*s00*/ ncols, plane, plane,
                              /*mul_s00*/ ncols, ncols, ncols,
                              /*mul ne*/ (uint32_t) ncols, 1, 1, 1,
                              /*add_s00*/ 0, 0, 0,
                              0, 0, 0, 0,
                              eps, s, nullptr, 0, q8, layout);
    }
    if (cudaGetLastError() != cudaSuccess) {
        return 0;
    }
    return cudaStreamSynchronize(s) == cudaSuccess;
} }


// SWIGLU rows, op_silu(x) * g as
// unary_gated_op_kernel computes them (F32, contiguous [nc, nrows] output),
// that also write the q8_1 MMQ rows of their first MMQ reader, in its layout
// (norm_row_q8_1_mmq on the finished row, as the norm producers do): the reader
// and the later readers of that layout skip their quantize. One block per row.
template <int block_size, mmq_q8_1_ds_layout ds_layout>
static __global__ void swiglu_q8_mmq_kernel(const float * x, const float * g, float * dst, const int64_t nc,
                                            const int64_t o0, const int64_t o1, block_q8_1_mmq * q8,
                                            const int64_t ncols_q8) {
    const int64_t row = blockIdx.x;
    const float * xr = x + row*o0;
    const float * gr = g + row*o1;
    float * dr = dst + row*nc;
    for (int64_t i = threadIdx.x; i < nc; i += block_size) {
        dr[i] = ggml_cuda_op_silu_single(xr[i]) * gr[i];
    }
    __syncthreads();
    norm_row_q8_1_mmq<block_size, ds_layout>(dr, q8, row, gridDim.x, nc, ncols_q8);
}

bool ggml_cuda_op_swiglu_q8_mmq_supported(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (dst->op != GGML_OP_GLU || ggml_get_glu_op(dst) != GGML_GLU_OP_SWIGLU ||
        dst->type != GGML_TYPE_F32 || src0 == nullptr || src0->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(dst) || !ggml_is_contiguous_1(src0) || src0->nb[0] != sizeof(float)) {
        return false;
    }
    if (src1 != nullptr && (src1->type != GGML_TYPE_F32 || !ggml_is_contiguous_1(src1) ||
                            src1->nb[0] != sizeof(float))) {
        return false;
    }
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    return dst->ne[0] == nc && nc % 4 == 0 && ggml_nrows(dst) == ggml_nrows(src0) &&
           ggml_nrows(dst) <= INT_MAX;
}

void ggml_cuda_op_swiglu_q8_mmq(ggml_backend_cuda_context & ctx, ggml_tensor * dst, void * q8, const int layout) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    const float * x = (const float *) src0->data;
    const float * g = src1 ? (const float *) src1->data : (const float *) src0->data;
    if (!src1) {
        x += swapped ? nc : 0;
        g += swapped ? 0 : nc;
    }
    const int64_t o0 = src0->nb[1] / sizeof(float);
    const int64_t o1 = (src1 ? src1->nb[1] : src0->nb[1]) / sizeof(float);
    const int64_t nrows = ggml_nrows(dst);
    const int64_t ncols_q8 = GGML_PAD(nc, MATRIX_ROW_PADDING);
    constexpr int block_size = 1024;
    const dim3 blocks((unsigned) nrows, 1, 1);
    cudaStream_t stream = ctx.stream();
    block_q8_1_mmq * y = (block_q8_1_mmq *) q8;
    switch (layout) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            swiglu_q8_mmq_kernel<block_size, MMQ_Q8_1_DS_LAYOUT_D4><<<blocks, block_size, 0, stream>>>(
                x, g, (float *) dst->data, nc, o0, o1, y, ncols_q8);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            swiglu_q8_mmq_kernel<block_size, MMQ_Q8_1_DS_LAYOUT_DS4><<<blocks, block_size, 0, stream>>>(
                x, g, (float *) dst->data, nc, o0, o1, y, ncols_q8);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            swiglu_q8_mmq_kernel<block_size, MMQ_Q8_1_DS_LAYOUT_D2S6><<<blocks, block_size, 0, stream>>>(
                x, g, (float *) dst->data, nc, o0, o1, y, ncols_q8);
            break;
        default:
            GGML_ABORT("swiglu q8 mmq: bad layout");
    }
}
