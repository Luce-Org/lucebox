#include "paged-attn.cuh"

#include "fattn-common.cuh"

#include <cfloat>
#include <cstdlib>
#include <cstring>

static constexpr int PAGED_ATTN_MAX_PARTITIONS = 128;
static constexpr int PAGED_ATTN_BLOCKS_PER_PARTITION = 64;

// Each warp handles one (sequence, query head, context partition). Warps for
// query heads that share a GQA K/V head are grouped in the same block so their
// identical cache reads can be reused by the GPU caches. Keeping Q in the same
// register formats as fattn-vec lets the quantized K dot products and V
// dequantizers share their well-tested F16/Q4_0/Q8_0 implementations. Long
// contexts are split between blocks and merged below; the direct specialization
// avoids scratch for a single part.
template<int D, ggml_type type_K, ggml_type type_V, bool write_partials>
static __global__ void paged_attn_decode(
        const char    * __restrict__ q,
        const char    * __restrict__ k,
        const char    * __restrict__ v,
        const char    * __restrict__ block_table,
        const char    * __restrict__ kv_seq_lens,
        char          * __restrict__ dst,
        float         * __restrict__ partial_acc,
        float2        * __restrict__ partial_meta,
        int64_t q_nb1,   int64_t q_nb2,
        int64_t k_nb1,   int64_t k_nb2,
        int64_t v_nb1,   int64_t v_nb2,
        int64_t bt_nb0,  int64_t bt_nb1,
        int64_t ksl_nb0,
        int64_t dst_nb1, int64_t dst_nb2,
        int32_t n_seq,
        int32_t n_head,
        int32_t n_head_kv,
        int32_t pool_tokens,
        int32_t max_blocks,
        int32_t block_size,
        int32_t n_partitions,
        int32_t min_partitions,
        float scale) {
    constexpr int nthreads = WARP_SIZE;
    constexpr int values_per_load = 4;
    constexpr int values_per_lane = D / nthreads;
    static_assert(D % (nthreads * values_per_load) == 0, "unsupported head size");

    const int head =
        (int) blockIdx.x * (int) blockDim.y + (int) threadIdx.y;
    const int seq  = blockIdx.y;
    const int partition = blockIdx.z;
    const int lane = threadIdx.x;
    // The launcher uses exact sequence/partition extents and a head-group
    // size that divides n_head. Keep every warp live through the quantized-Q
    // barrier below; partial blocks would make an early bounds return unsafe.

    const float * q_row = (const float *) (q + (int64_t) seq * q_nb1 + (int64_t) head * q_nb2);
    float       * o_row =       (float *) (dst + (int64_t) seq * dst_nb1 + (int64_t) head * dst_nb2);

    const int32_t kv_seq_len_raw =
        *(const int32_t *) (kv_seq_lens + (int64_t) seq * ksl_nb0);
    const int64_t table_capacity =
        (int64_t) max_blocks * block_size;
    const int32_t kv_seq_len = kv_seq_len_raw <= 0
        ? 0
        : (kv_seq_len_raw < table_capacity
            ? kv_seq_len_raw
            : (int32_t) table_capacity);
    const int32_t n_logical_blocks =
        (kv_seq_len + block_size - 1) / block_size;
    const int32_t context_partitions =
        (n_logical_blocks + PAGED_ATTN_BLOCKS_PER_PARTITION - 1) /
        PAGED_ATTN_BLOCKS_PER_PARTITION;
    const int32_t requested_partitions =
        context_partitions > min_partitions
            ? context_partitions
            : min_partitions;
    const int32_t available_partitions =
        n_logical_blocks < n_partitions ? n_logical_blocks : n_partitions;
    const int32_t active_partitions =
        requested_partitions < available_partitions
            ? requested_partitions
            : available_partitions;
    const int64_t output_row = (int64_t) head * n_seq + seq;
    const int64_t partial_row =
        output_row * n_partitions + partition;

    if (partition >= active_partitions) {
        if constexpr (write_partials) {
            if (lane == 0) {
                partial_meta[partial_row] = make_float2(-FLT_MAX, 0.0f);
            }
        } else {
#pragma unroll
            for (int i = lane; i < D; i += nthreads) {
                o_row[i] = 0.0f;
            }
        }
        return;
    }

    const int32_t logical_block_begin =
        ((int64_t) n_logical_blocks * partition) / active_partitions;
    const int32_t logical_block_end =
        ((int64_t) n_logical_blocks * (partition + 1)) /
        active_partitions;

    constexpr bool quantize_q =
        type_K != GGML_TYPE_F16;
    constexpr int q_registers = (D / 2) / nthreads;

#ifdef V_DOT2_F32_F16_AVAILABLE
    half2 q_reg[q_registers];
#else
    float2 q_reg[q_registers];
#endif
    int    q_i32[D / (sizeof(int) * nthreads)];
    float2 q_ds [D / (sizeof(int) * nthreads)];

    extern __shared__ char q_shared[];
    int * q_i32_shared =
        (int *) q_shared +
        (int) threadIdx.y * (D / sizeof(int));
    float2 * q_ds_shared =
        (float2 *) ((int *) q_shared +
                    (int) blockDim.y * (D / sizeof(int))) +
        (int) threadIdx.y * (D / QK8_1);

    if constexpr (quantize_q) {
        // Quantizing scale*Q matches fattn-vec and avoids an extra multiply for
        // every cached K row.
#pragma unroll
        for (int i0 = 0; i0 < D / (int) sizeof(int); i0 += nthreads) {
            quantize_q8_1_to_shared<float2, nthreads>(
                q_row + i0 * sizeof(int), scale,
                q_i32_shared + i0, q_ds_shared + i0 / QI8_1);
        }
        __syncthreads();

#pragma unroll
        for (int i0 = 0; i0 < D / (int) sizeof(int); i0 += nthreads) {
            const int i = i0 + lane;
            q_i32[i0 / nthreads] = q_i32_shared[i];
            q_ds [i0 / nthreads] = q_ds_shared[i / QI8_1];
        }
    } else {
        constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
        constexpr int cpy_ne = cpy_nb / sizeof(float);
        const float2 * q_f2 = (const float2 *) q_row;

#pragma unroll
        for (int i0 = 0; i0 < D / 2; i0 += nthreads * cpy_ne) {
            const int i = i0 + lane * cpy_ne;
            __align__(16) float2 tmp[cpy_ne];
            ggml_cuda_memcpy_1<cpy_nb>(
                tmp, q_f2 + i);
            ggml_cuda_memcpy_1<cpy_nb>(
                tmp + cpy_ne / 2, q_f2 + i + cpy_ne / 2);
#pragma unroll
            for (int j = 0; j < cpy_ne; ++j) {
#ifdef V_DOT2_F32_F16_AVAILABLE
                q_reg[i0 / nthreads + j] =
                    make_half2(tmp[j].x * scale, tmp[j].y * scale);
#else
                q_reg[i0 / nthreads + j] =
                    make_float2(tmp[j].x * scale, tmp[j].y * scale);
#endif
            }
        }
    }

    constexpr vec_dot_KQ_t vec_dot_kq =
        get_vec_dot_KQ<type_K, D, nthreads>();
    constexpr dequantize_V_t dequantize_v =
        get_dequantize_V<type_V, float, values_per_load>();

    float acc[values_per_lane] = { 0.0f };
    float qk_max = -FLT_MAX;
    float qk_sum = 0.0f;

    const int32_t n_physical_blocks = pool_tokens / block_size;
    const int32_t kv_head = head / (n_head / n_head_kv);

    for (int32_t logical_block = logical_block_begin;
         logical_block < logical_block_end;
         ++logical_block) {
        const int32_t physical_block =
            *(const int32_t *) (block_table +
                (int64_t) logical_block * bt_nb0 +
                (int64_t) seq * bt_nb1);

        // Invalid entries are never expected from the allocator, but skipping
        // them prevents stale metadata from becoming an out-of-bounds read.
        if (physical_block < 0 || physical_block >= n_physical_blocks) {
            continue;
        }

        const int32_t block_begin = logical_block * block_size;
        const int32_t block_end =
            kv_seq_len < block_begin + block_size
                ? kv_seq_len
                : block_begin + block_size;

        for (int32_t token = block_begin; token < block_end; ++token) {
            const int32_t physical_token =
                physical_block * block_size + token - block_begin;
            const char * k_row =
                k + (int64_t) physical_token * k_nb1 +
                    (int64_t) kv_head * k_nb2;
            const char * v_row =
                v + (int64_t) physical_token * v_nb1 +
                    (int64_t) kv_head * v_nb2;

            float score = vec_dot_kq(k_row, q_reg, q_i32, q_ds);
            score = warp_reduce_sum(score);

            const float qk_max_new = fmaxf(qk_max, score);
            const float old_scale  = expf(qk_max - qk_max_new);
            const float weight     = expf(score - qk_max_new);
            qk_sum = qk_sum * old_scale + weight;
            qk_max = qk_max_new;

#pragma unroll
            for (int segment = 0;
                 segment < D / (nthreads * values_per_load);
                 ++segment) {
                float values[values_per_load];
                const int value0 =
                    segment * nthreads * values_per_load +
                    lane * values_per_load;
                dequantize_v(v_row, values, value0);
#pragma unroll
                for (int j = 0; j < values_per_load; ++j) {
                    const int ai = segment * values_per_load + j;
                    acc[ai] = acc[ai] * old_scale + weight * values[j];
                }
            }
        }
    }

    if constexpr (write_partials) {
#pragma unroll
        for (int segment = 0;
             segment < D / (nthreads * values_per_load);
             ++segment) {
            const int value0 =
                segment * nthreads * values_per_load +
                lane * values_per_load;
#pragma unroll
            for (int j = 0; j < values_per_load; ++j) {
                partial_acc[partial_row * D + value0 + j] =
                    acc[segment * values_per_load + j];
            }
        }
        if (lane == 0) {
            partial_meta[partial_row] = make_float2(qk_max, qk_sum);
        }
    } else {
        const float inv_sum = qk_sum > 0.0f ? 1.0f / qk_sum : 0.0f;
#pragma unroll
        for (int segment = 0;
             segment < D / (nthreads * values_per_load);
             ++segment) {
            const int value0 =
                segment * nthreads * values_per_load +
                lane * values_per_load;
#pragma unroll
            for (int j = 0; j < values_per_load; ++j) {
                o_row[value0 + j] =
                    acc[segment * values_per_load + j] * inv_sum;
            }
        }
    }
}

template<int D>
__launch_bounds__(D, 1)
static __global__ void paged_attn_combine(
        const float  * __restrict__ partial_acc,
        const float2 * __restrict__ partial_meta,
        char         * __restrict__ dst,
        int64_t dst_nb1,
        int64_t dst_nb2,
        int32_t n_seq,
        int32_t n_partitions) {
    const int head = blockIdx.x;
    const int seq  = blockIdx.y;
    const int tid  = threadIdx.x;
    // combine_grid is exactly [n_head, n_seq], so no bounds branch is needed
    // above the block-wide barrier below.

    const int64_t output_row = (int64_t) head * n_seq + seq;
    const int64_t partial_row = output_row * n_partitions;

    __shared__ float partition_scale[PAGED_ATTN_MAX_PARTITIONS];
    __shared__ float denominator;

    // Compute the stable merge weights once per output row rather than once
    // per output component. Empty or invalid partitions have a zero sum.
    if (tid == 0) {
        float global_max = -FLT_MAX;
        for (int partition = 0; partition < n_partitions; ++partition) {
            const float2 meta = partial_meta[partial_row + partition];
            if (meta.y > 0.0f) {
                global_max = fmaxf(global_max, meta.x);
            }
        }

        float global_sum = 0.0f;
        for (int partition = 0; partition < n_partitions; ++partition) {
            const float2 meta = partial_meta[partial_row + partition];
            const float weight =
                meta.y > 0.0f ? expf(meta.x - global_max) : 0.0f;
            partition_scale[partition] = weight;
            global_sum += weight * meta.y;
        }
        denominator = global_sum;
    }
    __syncthreads();

    float numerator = 0.0f;
    for (int partition = 0; partition < n_partitions; ++partition) {
        const float weight = partition_scale[partition];
        if (weight > 0.0f) {
            numerator +=
                weight *
                partial_acc[(partial_row + partition) * D + tid];
        }
    }

    float * o_row =
        (float *) (dst + (int64_t) seq * dst_nb1 +
                         (int64_t) head * dst_nb2);
    o_row[tid] = denominator > 0.0f ? numerator / denominator : 0.0f;
}

static bool paged_attn_type_supported(ggml_type type) {
    return type == GGML_TYPE_F16 ||
           type == GGML_TYPE_Q4_0 ||
           type == GGML_TYPE_Q8_0;
}

bool ggml_cuda_paged_attn_supported(const ggml_tensor * dst) {
    const ggml_tensor * q             = dst->src[0];
    const ggml_tensor * k             = dst->src[1];
    const ggml_tensor * v             = dst->src[2];
    const ggml_tensor * block_table   = dst->src[3];
    const ggml_tensor * kv_seq_lens   = dst->src[4];

    if (!q || !k || !v || !block_table || !kv_seq_lens) {
        return false;
    }

    const int32_t block_size = ggml_get_op_params_i32(dst, 1);
    return dst->type == GGML_TYPE_F32 &&
           q->type == GGML_TYPE_F32 &&
           paged_attn_type_supported(k->type) &&
           paged_attn_type_supported(v->type) &&
           block_table->type == GGML_TYPE_I32 &&
           kv_seq_lens->type == GGML_TYPE_I32 &&
           q->nb[0] == sizeof(float) &&
           k->nb[0] == ggml_type_size(k->type) &&
           v->nb[0] == ggml_type_size(v->type) &&
           block_table->nb[0] == sizeof(int32_t) &&
           kv_seq_lens->nb[0] == sizeof(int32_t) &&
           dst->nb[0] == sizeof(float) &&
           dst->ne[0] == q->ne[0] &&
           dst->ne[1] == q->ne[1] &&
           dst->ne[2] == q->ne[2] &&
           dst->ne[3] == q->ne[3] &&
           q->ne[0] == 256 &&
           q->ne[0] == k->ne[0] &&
           q->ne[0] == v->ne[0] &&
           k->ne[1] == v->ne[1] &&
           k->ne[2] > 0 &&
           k->ne[2] == v->ne[2] &&
           q->ne[2] % k->ne[2] == 0 &&
           q->ne[3] == 1 &&
           k->ne[3] == 1 &&
           v->ne[3] == 1 &&
           block_table->ne[0] > 0 &&
           block_table->ne[1] == q->ne[1] &&
           block_table->ne[2] == 1 &&
           block_table->ne[3] == 1 &&
           kv_seq_lens->ne[0] == q->ne[1] &&
           kv_seq_lens->ne[1] == 1 &&
           kv_seq_lens->ne[2] == 1 &&
           kv_seq_lens->ne[3] == 1 &&
           block_size > 0 &&
           k->ne[1] % block_size == 0;
}

template<ggml_type type_K, ggml_type type_V>
static void launch_paged_attn(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        float scale,
        int32_t block_size) {
    constexpr int D = 256;
    const ggml_tensor * q            = dst->src[0];
    const ggml_tensor * k            = dst->src[1];
    const ggml_tensor * v            = dst->src[2];
    const ggml_tensor * block_table  = dst->src[3];
    const ggml_tensor * kv_seq_lens = dst->src[4];

    const int32_t gqa_ratio =
        (int32_t) (q->ne[2] / k->ne[2]);
    int32_t heads_per_block =
        gqa_ratio <= 32 ? gqa_ratio : 1;

    int max_blocks_per_sm = 0;
    size_t q_shared_bytes = 0;
    // Deliberately query the write_partials instantiation: occupancy only
    // steers min_partitions, which is a partition count for that variant.
    // The direct variant launches solely when the count collapses to one,
    // where occupancy no longer influences the topology.
    //
    // Large GQA ratios can make this register-heavy kernel impossible to
    // launch with one warp per query head. Reduce the number of colocated
    // heads until the device reports a viable block size.
    while (true) {
        q_shared_bytes =
            type_K == GGML_TYPE_F16
                ? 0
                : (size_t) heads_per_block *
                    (D + (D / QK8_1) * sizeof(float2));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            paged_attn_decode<D, type_K, type_V, true>,
            WARP_SIZE * heads_per_block, q_shared_bytes));
        if (max_blocks_per_sm > 0 || heads_per_block == 1) {
            break;
        }
        do {
            heads_per_block /= 2;
        } while (heads_per_block > 1 &&
                 q->ne[2] % heads_per_block != 0);
    }
    if (max_blocks_per_sm == 0) {
        GGML_ABORT("paged attention kernel has zero occupancy");
    }

    GGML_ASSERT(q->ne[2] % heads_per_block == 0);
    const int32_t head_groups =
        (int32_t) q->ne[2] / heads_per_block;
    const dim3 block(WARP_SIZE, (unsigned int) heads_per_block, 1);

    const int64_t output_rows = q->ne[1] * q->ne[2];
    const int64_t work_groups = q->ne[1] * head_groups;
    const int64_t target_blocks =
        (int64_t) ggml_cuda_info().devices[ctx.device].nsm *
        max_blocks_per_sm;
    int32_t min_partitions = (int32_t)
        ((target_blocks + work_groups - 1) / work_groups);
    if (min_partitions < 1) {
        min_partitions = 1;
    }
    // A single sequence needs more context partitions to expose enough work.
    // With multiple sequences, additional partitions mostly repeat Q setup.
    const int32_t partition_limit =
        q->ne[1] == 1 ? PAGED_ATTN_MAX_PARTITIONS : 32;
    if (min_partitions > partition_limit) {
        min_partitions = partition_limit;
    }
    if (min_partitions > block_table->ne[0]) {
        min_partitions = (int32_t) block_table->ne[0];
    }

    // Reserve enough grid slots for a long sequence to subdivide itself using
    // its device-side context length. Capacity, not the current length, keeps
    // the launch topology stable across CUDA graph replays.
    const int32_t capacity_partitions =
        ((int32_t) block_table->ne[0] +
         PAGED_ATTN_BLOCKS_PER_PARTITION - 1) /
        PAGED_ATTN_BLOCKS_PER_PARTITION;
    int32_t n_partitions =
        capacity_partitions > min_partitions
            ? capacity_partitions
            : min_partitions;
    if (n_partitions > PAGED_ATTN_MAX_PARTITIONS) {
        n_partitions = PAGED_ATTN_MAX_PARTITIONS;
    }
    if (n_partitions > block_table->ne[0]) {
        n_partitions = (int32_t) block_table->ne[0];
    }

    // Test/debug override used to exercise both the direct and partials paths
    // independently of device-specific occupancy. Values outside the valid
    // grid range are ignored.
    if (const char * env =
            std::getenv("GGML_CUDA_PAGED_ATTN_FORCE_PARTITIONS")) {
        const int forced = std::atoi(env);
        if (forced >= 1 && forced <= PAGED_ATTN_MAX_PARTITIONS &&
            forced <= block_table->ne[0]) {
            min_partitions = forced;
            n_partitions = forced;
        }
    }

    const dim3 grid(
        (unsigned int) head_groups,
        (unsigned int) q->ne[1],
        (unsigned int) n_partitions);

    if (n_partitions == 1) {
        paged_attn_decode<D, type_K, type_V, false>
            <<<grid, block, q_shared_bytes, ctx.stream()>>>(
            (const char *) q->data,
            (const char *) k->data,
            (const char *) v->data,
            (const char *) block_table->data,
            (const char *) kv_seq_lens->data,
            (char *) dst->data,
            nullptr,
            nullptr,
            q->nb[1], q->nb[2],
            k->nb[1], k->nb[2],
            v->nb[1], v->nb[2],
            block_table->nb[0], block_table->nb[1],
            kv_seq_lens->nb[0],
            dst->nb[1], dst->nb[2],
            (int32_t) q->ne[1],
            (int32_t) q->ne[2],
            (int32_t) k->ne[2],
            (int32_t) k->ne[1],
            (int32_t) block_table->ne[0],
            block_size,
            n_partitions,
            min_partitions,
            scale);
        return;
    }

    const size_t partial_rows =
        (size_t) output_rows * n_partitions;
    const size_t partial_values = partial_rows * D;
    ggml_cuda_pool_alloc<float> scratch(
        ctx.pool(), partial_values + 2 * partial_rows);
    float * partial_acc = scratch.ptr;
    float2 * partial_meta =
        (float2 *) (scratch.ptr + partial_values);

    paged_attn_decode<D, type_K, type_V, true>
        <<<grid, block, q_shared_bytes, ctx.stream()>>>(
        (const char *) q->data,
        (const char *) k->data,
        (const char *) v->data,
        (const char *) block_table->data,
        (const char *) kv_seq_lens->data,
        (char *) dst->data,
        partial_acc,
        partial_meta,
        q->nb[1], q->nb[2],
        k->nb[1], k->nb[2],
        v->nb[1], v->nb[2],
        block_table->nb[0], block_table->nb[1],
        kv_seq_lens->nb[0],
        dst->nb[1], dst->nb[2],
        (int32_t) q->ne[1],
        (int32_t) q->ne[2],
        (int32_t) k->ne[2],
        (int32_t) k->ne[1],
        (int32_t) block_table->ne[0],
        block_size,
        n_partitions,
        min_partitions,
        scale);

    const dim3 combine_grid(
        (unsigned int) q->ne[2],
        (unsigned int) q->ne[1],
        1);
    paged_attn_combine<D>
        <<<combine_grid, dim3(D, 1, 1), 0, ctx.stream()>>>(
        partial_acc,
        partial_meta,
        (char *) dst->data,
        dst->nb[1],
        dst->nb[2],
        (int32_t) q->ne[1],
        n_partitions);
}

template<ggml_type type_K>
static void launch_paged_attn_v(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        float scale,
        int32_t block_size) {
    switch (dst->src[2]->type) {
        case GGML_TYPE_F16:
            launch_paged_attn<type_K, GGML_TYPE_F16>(ctx, dst, scale, block_size);
            break;
        case GGML_TYPE_Q4_0:
            launch_paged_attn<type_K, GGML_TYPE_Q4_0>(ctx, dst, scale, block_size);
            break;
        case GGML_TYPE_Q8_0:
            launch_paged_attn<type_K, GGML_TYPE_Q8_0>(ctx, dst, scale, block_size);
            break;
        default:
            GGML_ABORT("unsupported paged-attention V type: %s",
                       ggml_type_name(dst->src[2]->type));
    }
}

void ggml_cuda_paged_attn(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_paged_attn_supported(dst));

    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    const int32_t block_size = ggml_get_op_params_i32(dst, 1);

    switch (dst->src[1]->type) {
        case GGML_TYPE_F16:
            launch_paged_attn_v<GGML_TYPE_F16>(ctx, dst, scale, block_size);
            break;
        case GGML_TYPE_Q4_0:
            launch_paged_attn_v<GGML_TYPE_Q4_0>(ctx, dst, scale, block_size);
            break;
        case GGML_TYPE_Q8_0:
            launch_paged_attn_v<GGML_TYPE_Q8_0>(ctx, dst, scale, block_size);
            break;
        default:
            GGML_ABORT("unsupported paged-attention K type: %s",
                       ggml_type_name(dst->src[1]->type));
    }
    CUDA_CHECK(cudaGetLastError());
}
