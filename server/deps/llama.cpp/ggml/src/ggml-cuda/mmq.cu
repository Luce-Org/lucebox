#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"
#include "rocmfp2_mix.cuh"
#include "rocmfp3_mix.cuh"

static thread_local size_t g_mmq_launch_count = 0;

extern "C" size_t ggml_backend_cuda_get_mmq_launch_count(void) {
    return g_mmq_launch_count;
}

namespace {

class mix_registry_dispatch_guard {
public:
    explicit mix_registry_dispatch_guard(ggml_type type) : type_(type) {
        if (type_ == GGML_TYPE_Q2_1_ROCMFP2_MIX) {
            ggml_cuda_rocmfp2_mix_registry_lock();
        } else if (type_ == GGML_TYPE_Q3_1_ROCMFP3_MIX) {
            ggml_cuda_rocmfp3_mix_registry_lock();
        }
    }

    ~mix_registry_dispatch_guard() {
        if (type_ == GGML_TYPE_Q2_1_ROCMFP2_MIX) {
            ggml_cuda_rocmfp2_mix_registry_unlock();
        } else if (type_ == GGML_TYPE_Q3_1_ROCMFP3_MIX) {
            ggml_cuda_rocmfp3_mix_registry_unlock();
        }
    }

    mix_registry_dispatch_guard(const mix_registry_dispatch_guard &) = delete;
    mix_registry_dispatch_guard & operator=(
        const mix_registry_dispatch_guard &) = delete;

private:
    ggml_type type_;
};

}  // namespace

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    const bool is_mix_type =
        args.type_x == GGML_TYPE_Q2_1_ROCMFP2_MIX ||
        args.type_x == GGML_TYPE_Q3_1_ROCMFP3_MIX;
    GGML_ASSERT(!is_mix_type || (args.mix_codebooks && args.mix_modes));
    ++g_mmq_launch_count;
    switch (args.type_x) {
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
            mul_mat_q_case<GGML_TYPE_Q4_0_ROCMFP4_FAST>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0_ROCMFP2:
            mul_mat_q_case<GGML_TYPE_Q2_0_ROCMFP2>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_1_ROCMFP2_MIX:
            mul_mat_q_case<GGML_TYPE_Q2_1_ROCMFP2_MIX>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_1_ROCMFP3_MIX:
            mul_mat_q_case<GGML_TYPE_Q3_1_ROCMFP3_MIX>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_0_ROCMFPX:
            mul_mat_q_case<GGML_TYPE_Q3_0_ROCMFPX>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
#ifndef GGML_CUDA_BLACKWELL_CONSUMER
            if (args.type_x == GGML_TYPE_MXFP4) {
                mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            } else {
                mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            }
#else
            GGML_ABORT("FP4 quantization requires sm_120a, not supported on consumer Blackwell (SM 12.0)");
#endif
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_mul_mat_q_impl(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src0_pair,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst,
        ggml_tensor * dst_pair) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.
    GGML_ASSERT((src0_pair == nullptr) == (dst_pair == nullptr));
    if (src0_pair) {
        GGML_ASSERT(src0_pair->type == src0->type);
        GGML_ASSERT(dst_pair->type == dst->type);
        GGML_ASSERT(ggml_are_same_shape(src0_pair, src0));
        GGML_ASSERT(ggml_are_same_stride(src0_pair, src0));
        GGML_ASSERT(ggml_are_same_shape(dst_pair, dst));
        GGML_ASSERT(ggml_are_same_stride(dst_pair, dst));
    }

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // Keep mix side data alive until every MMQ launch using it is enqueued.
    // Registry teardown takes the same lock and drains the owning device before
    // freeing those buffers.
    mix_registry_dispatch_guard mix_guard(src0->type);
    const void * mix_codebooks_raw = nullptr;
    const uint8_t * mix_modes = nullptr;
    if (src0->type == GGML_TYPE_Q2_1_ROCMFP2_MIX) {
        GGML_ASSERT(ggml_cuda_rocmfp2_mix_mmq_info(
            src0->data, &mix_codebooks_raw, &mix_modes));
    } else if (src0->type == GGML_TYPE_Q3_1_ROCMFP3_MIX) {
        GGML_ASSERT(ggml_cuda_rocmfp3_mix_mmq_info(
            src0->data, &mix_codebooks_raw, &mix_modes));
    }
    const nv_bfloat16 * mix_codebooks =
        reinterpret_cast<const nv_bfloat16 *>(mix_codebooks_raw);

    // Temporary weight tensors may have an allocation tail read by tiled MMQ.
    // Clear it once for each projection before launching either multiply.
    const ggml_tensor * weights[] = {src0, src0_pair};
    for (const ggml_tensor * weight : weights) {
        if (!weight ||
            ggml_backend_buffer_get_usage(weight->buffer) !=
                GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            continue;
        }
        const size_t size_data  = ggml_nbytes(weight);
        const size_t size_alloc =
            ggml_backend_buffer_get_alloc_size(weight->buffer, weight);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(weight));
            GGML_ASSERT(!weight->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) weight->data + size_data, 0,
                                       size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    bool use_stream_k =
        (GGML_CUDA_CC_IS_NVIDIA(cc) &&
         ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc);
    // Keep the established small-batch tuning hook from #503. The default
    // remains stream-k; positive values opt small verify widths into the
    // lower-overhead data-parallel path.
    static const int luce_mmq_dp_max_ne1 = []() {
        const char * value = getenv("LUCE_MMQ_DP_MAX_NE1");
        return value ? atoi(value) : 0;
    }();
    if (use_stream_k && ne11 <= luce_mmq_dp_max_ne1) {
        use_stream_k = false;
    }

    // TODO: tighter pool buffer size vs q8 path
    const bool use_native_mxfp4 = blackwell_mma_available(cc) && src0->type == GGML_TYPE_MXFP4;
    const bool grouped_src = !ids && ggml_mul_mat_is_grouped_src(dst);
    const ggml_tensor * grouped_physical = grouped_src ? src1->view_src : nullptr;
    int64_t grouped_width = 0;
    int64_t grouped_row_stride = 0;
    int64_t grouped_plane_stride = 0;
    if (grouped_src) {
        const int64_t groups = ggml_mul_mat_grouped_src_groups(dst);
        GGML_ASSERT(groups > 1 && ne10 % groups == 0);
        grouped_width = ne10 / groups;

        if (grouped_physical) {
            GGML_ASSERT(grouped_physical->type == GGML_TYPE_F32);
            GGML_ASSERT(grouped_physical->ne[0] == grouped_width);
            GGML_ASSERT(grouped_physical->ne[1] == ne11);
            GGML_ASSERT(grouped_physical->ne[2] == groups);
            GGML_ASSERT(grouped_physical->ne[3] == 1);
            grouped_row_stride = grouped_physical->nb[1] / ts_src1;
            grouped_plane_stride = grouped_physical->nb[2] / ts_src1;
        } else {
            // A scheduler copy between unlike backends materializes the raw
            // bytes of the logical 2-D view as a standalone tensor. The bytes
            // retain the grouped physical ordering, while view_src metadata
            // cannot cross the allocation boundary. ggml_mul_mat_grouped_src
            // guarantees a contiguous [width, rows, groups] physical source,
            // so reconstruct those two strides from the op's group count.
            grouped_row_stride = grouped_width;
            grouped_plane_stride = grouped_width * ne11;
        }
        GGML_ASSERT(!use_native_mxfp4);
    }

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1 +
            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (grouped_src) {
                quantize_mmq_q8_1_grouped_cuda(
                    src1_d, src1_q8_1.get(), src0->type,
                    ne10, grouped_width,
                    grouped_row_stride, grouped_plane_stride,
                    ne10_padded, ne11, stream);
            } else if (use_native_mxfp4) {
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_mxfp4_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_mxfp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) /
                                    (8 * QK_MXFP4 * sizeof(int))  // block_fp4_mmq holds 256 values (8 blocks of 32)
                                :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr,
            mix_codebooks, mix_modes, dst_d,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            use_stream_k, ne1};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        if (src0_pair) {
            mmq_args pair_args = args;
            pair_args.x = (const char *) src0_pair->data;
            pair_args.dst = (float *) dst_pair->data;
            if (src0_pair->type == GGML_TYPE_Q2_1_ROCMFP2_MIX) {
                const void * pair_codebooks = nullptr;
                const uint8_t * pair_modes = nullptr;
                GGML_ASSERT(ggml_cuda_rocmfp2_mix_mmq_info(
                    src0_pair->data, &pair_codebooks, &pair_modes));
                pair_args.mix_codebooks =
                    reinterpret_cast<const nv_bfloat16 *>(pair_codebooks);
                pair_args.mix_modes = pair_modes;
            } else if (src0_pair->type == GGML_TYPE_Q3_1_ROCMFP3_MIX) {
                const void * pair_codebooks = nullptr;
                const uint8_t * pair_modes = nullptr;
                GGML_ASSERT(ggml_cuda_rocmfp3_mix_mmq_info(
                    src0_pair->data, &pair_codebooks, &pair_modes));
                pair_args.mix_codebooks =
                    reinterpret_cast<const nv_bfloat16 *>(pair_codebooks);
                pair_args.mix_modes = pair_modes;
            }
            ggml_cuda_mul_mat_q_switch_type(ctx, pair_args, stream);
        }
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    // Pad ids_dst by mmq_x to prevent OOB reads in the stream-k kernel.
    // The kernel loads a full mmq_x-wide tile from ids_dst (line 3709 in mmq.cuh)
    // without bounds-checking the load, only the write-back is bounded.
    const int64_t mmq_x_pad = (int64_t)get_mmq_x_max_host(cc);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows + mmq_x_pad);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    {
        // The sorter compacts negative (masked) owner routes out of the expert
        // ranges. Quantization still visits these fixed-capacity arrays, so
        // make the unused tail point at token zero instead of stale pool data.
        CUDA_CHECK(cudaMemsetAsync(ids_src1.get(), 0,
                                   ne_get_rows * sizeof(int32_t), stream));
        CUDA_CHECK(cudaMemsetAsync(ids_dst.get(), 0,
                                   (ne_get_rows + mmq_x_pad) * sizeof(int32_t),
                                   stream));
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    // The helper's fixed-capacity layout is part of MMQ's addressing contract.
    // A future compact route count must be produced and owned by the graph
    // builder; until then, quantize the complete initialized route array.
    const int64_t n_routes_quantized = ne_get_rows;
    const size_t nbytes_src1_q8_1 = n_routes_quantized*ne10_padded * sizeof(block_q8_1)/QK8_1 +
        get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    const int64_t ne11_flat = n_routes_quantized;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_mxfp4) {
            quantize_mmq_mxfp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                    ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t s12 = use_native_mxfp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (8 * QK_MXFP4 * sizeof(int)) :
                                           ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    const int64_t ncols_max = ne12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(),
        mix_codebooks, mix_modes, dst_d,
        ne00, ne01, ne_get_rows, s01, n_routes_quantized, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        use_stream_k, ncols_max};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
    if (src0_pair) {
        mmq_args pair_args = args;
        pair_args.x = (const char *) src0_pair->data;
        pair_args.dst = (float *) dst_pair->data;
        if (src0_pair->type == GGML_TYPE_Q2_1_ROCMFP2_MIX) {
            const void * pair_codebooks = nullptr;
            const uint8_t * pair_modes = nullptr;
            GGML_ASSERT(ggml_cuda_rocmfp2_mix_mmq_info(
                src0_pair->data, &pair_codebooks, &pair_modes));
            pair_args.mix_codebooks =
                reinterpret_cast<const nv_bfloat16 *>(pair_codebooks);
            pair_args.mix_modes = pair_modes;
        } else if (src0_pair->type == GGML_TYPE_Q3_1_ROCMFP3_MIX) {
            const void * pair_codebooks = nullptr;
            const uint8_t * pair_modes = nullptr;
            GGML_ASSERT(ggml_cuda_rocmfp3_mix_mmq_info(
                src0_pair->data, &pair_codebooks, &pair_modes));
            pair_args.mix_codebooks =
                reinterpret_cast<const nv_bfloat16 *>(pair_codebooks);
            pair_args.mix_modes = pair_modes;
        }
        ggml_cuda_mul_mat_q_switch_type(ctx, pair_args, stream);
    }
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst) {
    ggml_cuda_mul_mat_q_impl(ctx, src0, nullptr, src1, ids, dst, nullptr);
}

void ggml_cuda_mul_mat_q_pair(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0_a,
        const ggml_tensor * src0_b,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst_a,
        ggml_tensor * dst_b) {
    ggml_cuda_mul_mat_q_impl(
        ctx, src0_a, src0_b, src1, ids, dst_a, dst_b);
}

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;
    const int64_t stride01 = ne00 / ggml_blck_size(src0->type);

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    const bool use_stream_k = ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc))
                            && src1_ncols == ne11;
    mix_registry_dispatch_guard mix_guard(src0->type);
    const void * mix_codebooks_raw = nullptr;
    const uint8_t * mix_modes = nullptr;
    if (src0->type == GGML_TYPE_Q2_1_ROCMFP2_MIX) {
        GGML_ASSERT(ggml_cuda_rocmfp2_mix_mmq_info(
            src0_dd_i, &mix_codebooks_raw, &mix_modes));
    } else if (src0->type == GGML_TYPE_Q3_1_ROCMFP3_MIX) {
        GGML_ASSERT(ggml_cuda_rocmfp3_mix_mmq_info(
            src0_dd_i, &mix_codebooks_raw, &mix_modes));
    }
    const mmq_args args = {
        src0_dd_i, src0->type, (const int *) src1_ddq_i, nullptr, nullptr,
        reinterpret_cast<const nv_bfloat16 *>(mix_codebooks_raw), mix_modes,
        dst_dd_i,
        ne00, row_diff, src1_ncols, stride01, ne11, nrows_dst,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k, src1_ncols};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_padded_row_size);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

#ifdef ROCMFP2_AFFINE
    // The affine Q8_1 tile loader carries scale and -offset and uses the
    // activation sum correction. Keep it opt-in until model-level validation.
    if (type == GGML_TYPE_Q2_0_ROCMFP2 &&
        std::getenv("DFLASH_CUDA_MMQ_FP2_AFFINE") == nullptr) {
        return false;
    }
    // Batched expert MMQ is finite with the affine correction, but its
    // gather/quantize path is slower than grouped MMVQ at DS4 verify width.
    if (type == GGML_TYPE_Q2_0_ROCMFP2 && n_experts > 1) {
        return false;
    }
    if (type == GGML_TYPE_Q2_0_ROCMFP2) {
        const char * runtime_disable = std::getenv(
            "DFLASH_CUDA_MMQ_FP2_AFFINE_RUNTIME_DISABLE");
        if (runtime_disable && *runtime_disable &&
            std::strcmp(runtime_disable, "0") != 0) {
            return false;
        }
    }
    if (type == GGML_TYPE_Q2_0_ROCMFP2) {
        // Owner-isolation/qualification switch. AMD architecture codes are
        // accepted in decimal or 0x form (for example 0x1100 or 0x1151).
        static const int required_cc = [] {
            const char * raw = std::getenv(
                "DFLASH_CUDA_MMQ_FP2_AFFINE_CC");
            if (!raw || !*raw) return 0;
            char * end = nullptr;
            const long parsed = std::strtol(raw, &end, 0);
            return end && end != raw && *end == '\0' && parsed > 0 &&
                           parsed <= INT_MAX
                ? (int) parsed
                : 0;
        }();
        if (required_cc > 0 && cc != required_cc) {
            return false;
        }
    }
    if (type == GGML_TYPE_Q2_0_ROCMFP2) {
        static const int min_ncols = [] {
            const char * raw = std::getenv(
                "DFLASH_CUDA_MMQ_FP2_AFFINE_MIN_NCOLS");
            if (!raw || !*raw) return 0;
            char * end = nullptr;
            const long parsed = std::strtol(raw, &end, 10);
            return end && end != raw && *end == '\0' && parsed > 0 &&
                           parsed <= INT_MAX
                ? (int) parsed
                : 0;
        }();
        if (ne11 < min_ncols) {
            return false;
        }
    }
#endif // ROCMFP2_AFFINE

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
#ifdef GGML_CUDA_BLACKWELL_CONSUMER
            mmq_supported = false;
            break;
#endif
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        case GGML_TYPE_Q2_0_ROCMFP2:
        case GGML_TYPE_Q3_0_ROCMFPX:
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
            // ROCmFPX MMQ uses wave32 WMMA plus portable packed-int dot
            // products. Both gfx1151 (RDNA 3.5) and gfx12xx (RDNA 4) provide
            // that contract; the latter is required by the grouped DS4
            // prefill projection because no BLAS path understands its
            // strided [K/group, N, group] activation layout.
            mmq_supported = GGML_CUDA_CC_IS_RDNA3_5(cc) ||
                            GGML_CUDA_CC_IS_RDNA4(cc);
            break;
        case GGML_TYPE_Q2_1_ROCMFP2_MIX: {
            static const bool mix_mmq_enabled = []() {
                const char * value = getenv("DFLASH_DS4_MIX_MMQ_PREFILL");
                return value != nullptr && !(value[0] == '0' && value[1] == '\0');
            }();
            mmq_supported = mix_mmq_enabled &&
                (GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4(cc));
            break;
        }
        case GGML_TYPE_Q3_1_ROCMFP3_MIX: {
            static const bool mix_mmq_enabled = []() {
                const char * value = getenv("DFLASH_DS4_MIX_MMQ_PREFILL");
                return value != nullptr && !(value[0] == '0' && value[1] == '\0');
            }();
            mmq_supported = mix_mmq_enabled &&
                (GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4(cc));
            break;
        }
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                // Wide Q4_K batches on gfx1151 favor dequantization +
                // hipBLAS; keep MMQ for decode/small-prefill shapes and for
                // the unmeasured RDNA 3.0 family.
                case GGML_TYPE_Q4_K:
                    return !GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 256;
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
