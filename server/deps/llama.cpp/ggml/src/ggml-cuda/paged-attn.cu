#include "paged-attn.cuh"

#include "fattn-common.cuh"

#include <cfloat>
#include <cstring>

// Decode uses one logical warp per (sequence, query head). Keeping Q in the
// same register formats as fattn-vec lets the quantized K dot products and V
// dequantizers share their well-tested F16/Q4_0/Q8_0 implementations.
template<int D, ggml_type type_K, ggml_type type_V>
static __global__ void paged_attn_decode(
        const char    * __restrict__ q,
        const char    * __restrict__ k,
        const char    * __restrict__ v,
        const char    * __restrict__ block_table,
        const char    * __restrict__ kv_seq_lens,
        char          * __restrict__ dst,
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
        float scale) {
    constexpr int nthreads = WARP_SIZE;
    constexpr int values_per_load = 4;
    constexpr int values_per_lane = D / nthreads;
    static_assert(D % (nthreads * values_per_load) == 0, "unsupported head size");

    const int head = blockIdx.x;
    const int seq  = blockIdx.y;
    const int lane = threadIdx.x;
    if (seq >= n_seq || head >= n_head) {
        return;
    }

    const float * q_row = (const float *) (q + (int64_t) seq * q_nb1 + (int64_t) head * q_nb2);
    float       * o_row =       (float *) (dst + (int64_t) seq * dst_nb1 + (int64_t) head * dst_nb2);

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

    __shared__ int    q_i32_shared[D / sizeof(int)];
    __shared__ float2 q_ds_shared[D / QK8_1];

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
    const int32_t n_physical_blocks = pool_tokens / block_size;
    const int32_t kv_head = head / (n_head / n_head_kv);

    for (int32_t logical_block = 0; logical_block < n_logical_blocks; ++logical_block) {
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

    const dim3 grid((unsigned int) q->ne[2], (unsigned int) q->ne[1], 1);
    const dim3 block(WARP_SIZE, 1, 1);
    paged_attn_decode<D, type_K, type_V><<<grid, block, 0, ctx.stream()>>>(
        (const char *) q->data,
        (const char *) k->data,
        (const char *) v->data,
        (const char *) block_table->data,
        (const char *) kv_seq_lens->data,
        (char *) dst->data,
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
        scale);
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
