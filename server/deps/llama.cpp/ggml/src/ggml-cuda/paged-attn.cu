#include "paged-attn.cuh"

#include "fattn-common.cuh"

#include <atomic>
#include <cfloat>
#include <cstdlib>
#include <cstring>

static constexpr int PAGED_ATTN_MAX_PARTITIONS = 128;
static constexpr int PAGED_ATTN_BLOCKS_PER_PARTITION = 64;
static constexpr int PAGED_ATTN_HEAD_DIM = 256;
static constexpr int PAGED_ATTN_MAX_PACKED_WARPS = 8;

// Partition count for n_blocks of context: enough partitions to cover the
// blocks and to reach min_partitions for occupancy, but never more than the
// blocks themselves or the cap. Host and device must agree on this so the
// launched grid matches the per-sequence active partition count.
static __host__ __device__ __forceinline__ int32_t paged_attn_partitions(
        int32_t n_blocks, int32_t min_partitions, int32_t cap) {
    const int32_t context_partitions =
        (n_blocks + PAGED_ATTN_BLOCKS_PER_PARTITION - 1) /
        PAGED_ATTN_BLOCKS_PER_PARTITION;
    const int32_t requested =
        context_partitions > min_partitions
            ? context_partitions
            : min_partitions;
    const int32_t available = n_blocks < cap ? n_blocks : cap;
    return requested < available ? requested : available;
}

// All scores are computed in the log2 domain: log2(e) is folded into the same
// Q prescale that already carries the 1/sqrt(D) attention scale, so every
// softmax exponential uses the fast exp2f SFU path.
static constexpr float PAGED_ATTN_LOG2E = 1.44269504088896340736f;

// Each warp handles n_batch_heads query heads that share one GQA K/V head for
// one (sequence, context partition). Batching the heads into a single warp
// amortizes the K/V loads and dequantization across the whole group: each K
// row is read once and dotted against every batched Q, and each V row is
// dequantized once and accumulated with every batched weight. Warps covering
// the remaining heads of a K/V group (and further K/V groups) are colocated
// in the same block so the GPU caches can still dedupe overlapping reads.
//
// The token loop works on WARP_SIZE-token score tiles: lane t owns the score
// of tile token t, so the per-token dot/reduce chains are independent, the
// accumulator rescale runs once per tile instead of once per token, and it is
// skipped entirely when the tile does not raise the running maximum.
//
// Quantized Q is produced once per decode step by paged_attn_quantize_q and
// read back from global memory here, so context partitions do not repeat the
// quantization work. Long contexts are split between blocks and merged below;
// the direct specialization avoids scratch for a single partition.

// Batched K·Q dots: one K row is loaded once per lane and dotted against all
// n_batch_heads quantized Q vectors. These mirror the fattn-common vec_dot
// implementations for the three supported K cache types.

template<int D, int nbh>
static __device__ __forceinline__ void multi_vec_dot_kq_f16(
        const char * __restrict__ K_c,
#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 (&Q_reg)[nbh][(D/2)/WARP_SIZE],
#else
        const float2 (&Q_reg)[nbh][(D/2)/WARP_SIZE],
#endif
        float (&sum)[nbh]) {
    const half2 * K_h2 = (const half2 *) K_c;
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

#pragma unroll
    for (int k0 = 0; k0 < D/2; k0 += WARP_SIZE*cpy_ne) {
        __align__(16) half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k0 + threadIdx.x*cpy_ne);
#pragma unroll
        for (int k1 = 0; k1 < cpy_ne; ++k1) {
#pragma unroll
            for (int h = 0; h < nbh; ++h) {
#ifdef V_DOT2_F32_F16_AVAILABLE
                ggml_cuda_mad(sum[h],                 tmp[k1] , Q_reg[h][k0/WARP_SIZE + k1]);
#else
                ggml_cuda_mad(sum[h], __half22float2(tmp[k1]), Q_reg[h][k0/WARP_SIZE + k1]);
#endif
            }
        }
    }
}

template<int D, int nbh>
static __device__ __forceinline__ void multi_vec_dot_kq_q4_0(
        const char * __restrict__ K_c,
        const int    (&Q_q8)[nbh][D/(sizeof(int)*WARP_SIZE)],
        const float2 (&Q_ds)[nbh][D/(sizeof(int)*WARP_SIZE)],
        float (&sum)[nbh]) {
    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;

#pragma unroll
    for (int k0 = 0; k0 < int(D/sizeof(int)); k0 += WARP_SIZE) {
        const int k_KQ  = k0 + threadIdx.x;
        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const float K_d = __half2float(K_q4_0[ib].d);

#pragma unroll
        for (int h = 0; h < nbh; ++h) {
            const int sumi = ggml_cuda_dp4a(v, Q_q8[h][k0/WARP_SIZE], 0);
            const float2 ds = Q_ds[h][k0/WARP_SIZE];
            sum[h] += K_d * (sumi*ds.x - (8/QI8_1)*ds.y);
        }
    }
}

template<int D, int nbh>
static __device__ __forceinline__ void multi_vec_dot_kq_q8_0(
        const char * __restrict__ K_c,
        const int    (&Q_q8)[nbh][D/(sizeof(int)*WARP_SIZE)],
        const float2 (&Q_ds)[nbh][D/(sizeof(int)*WARP_SIZE)],
        float (&sum)[nbh]) {
    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;

#pragma unroll
    for (int k0 = 0; k0 < int(D/sizeof(int)); k0 += WARP_SIZE) {
        const int k_KQ = k0 + threadIdx.x;
        const int ib   = k_KQ / QI8_0;
        const int iqs  = k_KQ % QI8_0;

        int v;
        ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);
        const float K_d = __half2float(K_q8_0[ib].d);

#pragma unroll
        for (int h = 0; h < nbh; ++h) {
            const int sumi = ggml_cuda_dp4a(v, Q_q8[h][k0/WARP_SIZE], 0);
            sum[h] += K_d * Q_ds[h][k0/WARP_SIZE].x * sumi;
        }
    }
}

// Quantizes scale*log2(e)*Q to q8_1 once per decode step. The decode kernel
// used to requantize Q per (warp, partition) through shared memory; at long
// contexts that repeated the same work up to PAGED_ATTN_MAX_PARTITIONS times
// per head. One warp per (head, sequence) row.
template<int D>
static __global__ void paged_attn_quantize_q(
        const char   * __restrict__ q,
        int          * __restrict__ q_i32,
        float2       * __restrict__ q_ds,
        int64_t q_nb1, int64_t q_nb2,
        int32_t n_seq,
        float scale) {
    const int head = blockIdx.x;
    const int seq  = blockIdx.y;

    const float * q_row = (const float *) (q + (int64_t) seq * q_nb1 + (int64_t) head * q_nb2);
    const int64_t row = (int64_t) head * n_seq + seq;
    int    * yq32 = q_i32 + row * (D / (int) sizeof(int));
    float2 * yds  = q_ds  + row * (D / QK8_1);

#pragma unroll
    for (int i0 = 0; i0 < D / (int) sizeof(int); i0 += WARP_SIZE) {
        quantize_q8_1_to_shared<float2, WARP_SIZE>(
            q_row + i0 * sizeof(int), scale, yq32 + i0, yds + i0 / QI8_1);
    }
}

template<int D, ggml_type type_K, ggml_type type_V, int n_batch_heads, bool write_partials>
static __global__ void paged_attn_decode(
        const char    * __restrict__ q,
        const char    * __restrict__ k,
        const char    * __restrict__ v,
        const int     * __restrict__ q_i32_glob,
        const float2  * __restrict__ q_ds_glob,
        const char    * __restrict__ block_table,
        const char    * __restrict__ kv_seq_lens,
        const char    * __restrict__ active_slot_ids,
        const char    * __restrict__ query_positions,
        char          * __restrict__ dst,
        half          * __restrict__ partial_acc,
        float2        * __restrict__ partial_meta,
        int64_t q_nb1,   int64_t q_nb2,
        int64_t k_nb1,   int64_t k_nb2,
        int64_t v_nb1,   int64_t v_nb2,
        int64_t bt_nb0,  int64_t bt_nb1,
        int64_t ksl_nb0,
        int64_t asi_nb0, int64_t qpos_nb0,
        int64_t dst_nb1, int64_t dst_nb2,
        int32_t n_table_seq,
        int32_t n_head,
        int32_t n_head_kv,
        int32_t pool_tokens,
        int32_t max_blocks,
        int32_t block_size,
        int32_t min_partitions,
        float scale) {
    constexpr int nthreads = WARP_SIZE;
    constexpr int values_per_load = 4;
    constexpr int values_per_lane = D / nthreads;
    static_assert(D % (nthreads * values_per_load) == 0, "unsupported head size");

    // The launcher guarantees n_batch_heads divides the GQA ratio, the ratio
    // divided by n_batch_heads divides blockDim.y, and the grid covers exact
    // sequence/partition extents, so every warp maps to live heads and no
    // bounds return is needed.
    const int gqa_ratio       = n_head / n_head_kv;
    const int warps_per_group = gqa_ratio / n_batch_heads;
    const int warp            = threadIdx.y;
    const int kv_head =
        (int) blockIdx.x * ((int) blockDim.y / warps_per_group) +
        warp / warps_per_group;
    const int head0 =
        kv_head * gqa_ratio + (warp % warps_per_group) * n_batch_heads;
    const int seq       = blockIdx.y;
    const int partition = blockIdx.z;
    const int lane      = threadIdx.x;

    const int n_seq        = gridDim.y;
    const int n_partitions = gridDim.z;

    const int32_t physical_seq_raw = active_slot_ids
        ? *(const int32_t *) (active_slot_ids + (int64_t) seq * asi_nb0)
        : seq;
    const int32_t query_pos = query_positions
        ? *(const int32_t *) (query_positions + (int64_t) seq * qpos_nb0)
        : -1;
    // A row is live when its slot id selects a real block-table column and,
    // for ragged batches, its causal position is non-negative. Dead rows are
    // pinned to column 0 with kv_seq_len forced to 0, which routes every
    // partition through the existing zero-output early path; the block table
    // is then never read for them.
    const bool valid_query =
        physical_seq_raw >= 0 && physical_seq_raw < n_table_seq &&
        (!query_positions || query_pos >= 0);
    const int32_t physical_seq = valid_query ? physical_seq_raw : 0;
    int32_t kv_seq_len_raw = valid_query
        ? *(const int32_t *) (kv_seq_lens +
                              (int64_t) physical_seq * ksl_nb0)
        : 0;
    // The inclusive clamp IS the causal mask: this row attends tokens
    // [0, pos] only, and every downstream bound (partition count, token loop
    // extents) already derives from kv_seq_len.
    if (query_positions && query_pos < kv_seq_len_raw) {
        kv_seq_len_raw = query_pos + 1;
    }
    const int64_t table_capacity =
        (int64_t) max_blocks * block_size;
    const int32_t kv_seq_len = kv_seq_len_raw <= 0
        ? 0
        : (kv_seq_len_raw < table_capacity
            ? kv_seq_len_raw
            : (int32_t) table_capacity);
    const int32_t n_logical_blocks =
        (kv_seq_len + block_size - 1) / block_size;
    const int32_t active_partitions =
        paged_attn_partitions(n_logical_blocks, min_partitions, n_partitions);

    if (partition >= active_partitions) {
#pragma unroll
        for (int h = 0; h < n_batch_heads; ++h) {
            const int64_t output_row = (int64_t) (head0 + h) * n_seq + seq;
            if constexpr (write_partials) {
                if (lane == 0) {
                    partial_meta[output_row * n_partitions + partition] =
                        make_float2(-FLT_MAX, 0.0f);
                }
            } else {
                float * o_row =
                    (float *) (dst + (int64_t) seq * dst_nb1 +
                                     (int64_t) (head0 + h) * dst_nb2);
#pragma unroll
                for (int i = lane; i < D; i += nthreads) {
                    o_row[i] = 0.0f;
                }
            }
        }
        return;
    }

    const int32_t logical_block_begin =
        ((int64_t) n_logical_blocks * partition) / active_partitions;
    const int32_t logical_block_end =
        ((int64_t) n_logical_blocks * (partition + 1)) /
        active_partitions;
    const int32_t token_begin = logical_block_begin * block_size;
    const int32_t token_end_blocks = logical_block_end * block_size;
    const int32_t token_end =
        kv_seq_len < token_end_blocks ? kv_seq_len : token_end_blocks;

    constexpr bool quantize_q = type_K != GGML_TYPE_F16;
    constexpr int q_registers   = (D / 2) / nthreads;
    constexpr int q_i32_per_lane = D / (sizeof(int) * nthreads);

#ifdef V_DOT2_F32_F16_AVAILABLE
    half2  q_reg[n_batch_heads][q_registers];
#else
    float2 q_reg[n_batch_heads][q_registers];
#endif
    int    q_i32[n_batch_heads][q_i32_per_lane];
    float2 q_ds [n_batch_heads][q_i32_per_lane];

    if constexpr (quantize_q) {
        // Read back the q8_1 rows produced by paged_attn_quantize_q. The rows
        // are tiny and shared by every partition, so they stay L2-resident.
#pragma unroll
        for (int h = 0; h < n_batch_heads; ++h) {
            const int64_t row = (int64_t) (head0 + h) * n_seq + seq;
            const int    * yq32 = q_i32_glob + row * (D / (int) sizeof(int));
            const float2 * yds  = q_ds_glob  + row * (D / QK8_1);
#pragma unroll
            for (int i0 = 0; i0 < D / (int) sizeof(int); i0 += nthreads) {
                const int i = i0 + lane;
                q_i32[h][i0 / nthreads] = yq32[i];
                q_ds [h][i0 / nthreads] = yds[i / QI8_1];
            }
        }
    } else {
        constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
        constexpr int cpy_ne = cpy_nb / sizeof(float);
        // Fold log2(e) into the same prescale that carries the attention
        // scale so the softmax below can use exp2f throughout.
        const float scale_log2 = scale * PAGED_ATTN_LOG2E;

#pragma unroll
        for (int h = 0; h < n_batch_heads; ++h) {
            const float2 * q_f2 = (const float2 *)
                (q + (int64_t) seq * q_nb1 + (int64_t) (head0 + h) * q_nb2);
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
                    q_reg[h][i0 / nthreads + j] =
                        make_half2(tmp[j].x * scale_log2, tmp[j].y * scale_log2);
#else
                    q_reg[h][i0 / nthreads + j] =
                        make_float2(tmp[j].x * scale_log2, tmp[j].y * scale_log2);
#endif
                }
            }
        }
    }

    constexpr dequantize_V_t dequantize_v =
        get_dequantize_V<type_V, float, values_per_load>();

    float acc[n_batch_heads][values_per_lane] = {{0.0f}};
    float qk_max[n_batch_heads];
    float qk_sum[n_batch_heads];
#pragma unroll
    for (int h = 0; h < n_batch_heads; ++h) {
        qk_max[h] = -FLT_MAX;
        qk_sum[h] = 0.0f;
    }

    const int32_t n_physical_blocks = pool_tokens / block_size;

    for (int32_t tile_begin = token_begin;
         tile_begin < token_end;
         tile_begin += nthreads) {
        const int32_t tile_len =
            token_end - tile_begin < nthreads
                ? token_end - tile_begin
                : nthreads;

        // Lane t resolves tile token t once; both phases below fetch it from
        // the owning lane instead of re-reading the block table per token.
        // Invalid entries are never expected from the allocator, but mapping
        // them to -1 prevents stale metadata from becoming an out-of-bounds
        // read; their tokens contribute nothing, mirroring the CPU reference.
        int32_t phys_mine = -1;
        const int32_t my_token = tile_begin + lane;
        if (my_token < token_end) {
            const int32_t logical_block = my_token / block_size;
            const int32_t physical_block =
                *(const int32_t *) (block_table +
                    (int64_t) logical_block * bt_nb0 +
                    (int64_t) physical_seq * bt_nb1);
            if (physical_block >= 0 && physical_block < n_physical_blocks) {
                phys_mine =
                    physical_block * block_size + my_token % block_size;
            }
        }

        float score_mine[n_batch_heads];
#pragma unroll
        for (int h = 0; h < n_batch_heads; ++h) {
            score_mine[h] = -FLT_MAX;
        }

        for (int j = 0; j < tile_len; ++j) {
            const int32_t phys_j = __shfl_sync(0xFFFFFFFF, phys_mine, j, WARP_SIZE);
            if (phys_j < 0) {
                continue;
            }
            const char * k_row =
                k + (int64_t) phys_j * k_nb1 + (int64_t) kv_head * k_nb2;

            float sums[n_batch_heads] = {0.0f};
            if constexpr (type_K == GGML_TYPE_F16) {
                multi_vec_dot_kq_f16<D, n_batch_heads>(k_row, q_reg, sums);
            } else if constexpr (type_K == GGML_TYPE_Q4_0) {
                multi_vec_dot_kq_q4_0<D, n_batch_heads>(k_row, q_i32, q_ds, sums);
            } else {
                static_assert(type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_Q4_0 ||
                              type_K == GGML_TYPE_Q8_0, "unsupported K type");
                multi_vec_dot_kq_q8_0<D, n_batch_heads>(k_row, q_i32, q_ds, sums);
            }
#pragma unroll
            for (int h = 0; h < n_batch_heads; ++h) {
                const float score = warp_reduce_sum(sums[h]);
                score_mine[h] = lane == j ? score : score_mine[h];
            }
        }

        // Tile softmax update: one accumulator rescale per tile, skipped
        // entirely when the tile does not raise the running maximum (the
        // overwhelmingly common case, where the rescale would multiply by 1).
        float w_mine[n_batch_heads];
#pragma unroll
        for (int h = 0; h < n_batch_heads; ++h) {
            const float tile_max = warp_reduce_max(score_mine[h]);
            if (tile_max > qk_max[h]) {
                const float old_scale = exp2f(qk_max[h] - tile_max);
                qk_sum[h] *= old_scale;
#pragma unroll
                for (int i = 0; i < values_per_lane; ++i) {
                    acc[h][i] *= old_scale;
                }
                qk_max[h] = tile_max;
            }
            w_mine[h] = score_mine[h] > -FLT_MAX/2
                ? exp2f(score_mine[h] - qk_max[h])
                : 0.0f;
            qk_sum[h] += warp_reduce_sum(w_mine[h]);
        }

        for (int j = 0; j < tile_len; ++j) {
            const int32_t phys_j = __shfl_sync(0xFFFFFFFF, phys_mine, j, WARP_SIZE);
            if (phys_j < 0) {
                continue;
            }
            const char * v_row =
                v + (int64_t) phys_j * v_nb1 + (int64_t) kv_head * v_nb2;

            float weight[n_batch_heads];
#pragma unroll
            for (int h = 0; h < n_batch_heads; ++h) {
                weight[h] = __shfl_sync(0xFFFFFFFF, w_mine[h], j, WARP_SIZE);
            }

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
                for (int i = 0; i < values_per_load; ++i) {
                    const int ai = segment * values_per_load + i;
#pragma unroll
                    for (int h = 0; h < n_batch_heads; ++h) {
                        acc[h][ai] += weight[h] * values[i];
                    }
                }
            }
        }
    }

#pragma unroll
    for (int h = 0; h < n_batch_heads; ++h) {
        const int64_t output_row = (int64_t) (head0 + h) * n_seq + seq;
        const float inv_sum = qk_sum[h] > 0.0f ? 1.0f / qk_sum[h] : 0.0f;
        // Partials are stored normalized by the partition's qk_sum: it
        // keeps the values inside f16 range (halving scratch traffic) and
        // lets the combine kernel reuse its weight*qk_sum coefficient.
        const int64_t partial_row =
            output_row * n_partitions + partition;
        float * o_row =
            (float *) (dst + (int64_t) seq * dst_nb1 +
                             (int64_t) (head0 + h) * dst_nb2);
#pragma unroll
        for (int segment = 0;
             segment < D / (nthreads * values_per_load);
             ++segment) {
            const int value0 =
                segment * nthreads * values_per_load +
                lane * values_per_load;
#pragma unroll
            for (int i = 0; i < values_per_load; ++i) {
                const float value =
                    acc[h][segment * values_per_load + i] * inv_sum;
                if constexpr (write_partials) {
                    partial_acc[partial_row * D + value0 + i] =
                        __float2half(value);
                } else {
                    o_row[value0 + i] = value;
                }
            }
        }
        if constexpr (write_partials) {
            if (lane == 0) {
                partial_meta[partial_row] = make_float2(qk_max[h], qk_sum[h]);
            }
        }
    }
}

template<int D>
__launch_bounds__(D, 1)
static __global__ void paged_attn_combine(
        const half   * __restrict__ partial_acc,
        const float2 * __restrict__ partial_meta,
        char         * __restrict__ dst,
        int64_t dst_nb1,
        int64_t dst_nb2,
        int32_t n_partitions) {
    const int head  = blockIdx.x;
    const int seq   = blockIdx.y;
    const int n_seq = gridDim.y;
    const int tid   = threadIdx.x;
    // combine_grid is exactly [n_head, n_seq], so no bounds branch is needed
    // above the block-wide barrier below.

    const int64_t output_row = (int64_t) head * n_seq + seq;
    const int64_t partial_row = output_row * n_partitions;

    __shared__ float partition_scale[PAGED_ATTN_MAX_PARTITIONS];
    __shared__ float reduction[D / WARP_SIZE];

    // Load each partition's metadata once, then use all warps for the stable
    // max/sum reduction. Threads beyond n_partitions contribute sentinels.
    // meta.x is a log2-domain maximum, hence exp2f below.
    const float2 meta =
        tid < n_partitions
            ? partial_meta[partial_row + tid]
            : make_float2(-FLT_MAX, 0.0f);
    const float local_max = meta.y > 0.0f ? meta.x : -FLT_MAX;
    const float global_max =
        block_reduce<block_reduce_method::MAX, D>(local_max, reduction);
    __syncthreads();

    // The stored partials are normalized by their partition's qk_sum, so a
    // partition's combine coefficient is weight*qk_sum — the same product
    // that forms the denominator.
    const float coefficient =
        meta.y > 0.0f ? exp2f(meta.x - global_max) * meta.y : 0.0f;
    if (tid < n_partitions) {
        partition_scale[tid] = coefficient;
    }
    const float denominator =
        block_reduce<block_reduce_method::SUM, D>(coefficient, reduction);
    __syncthreads();

    float numerator = 0.0f;
    for (int partition = 0; partition < n_partitions; ++partition) {
        const float w = partition_scale[partition];
        if (w > 0.0f) {
            numerator +=
                w *
                __half2float(
                    partial_acc[(partial_row + partition) * D + tid]);
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

// Per-warp query-head batch widths, tried widest first at launch. The launch
// fallback chain and the support check share this ladder: every ratio the
// check accepts must have a width the chain can launch.
static constexpr int PAGED_ATTN_BATCH_HEADS[] = {6, 3, 1};

static bool paged_attn_batch_heads_viable(int64_t gqa_ratio, int n_batch_heads) {
    return gqa_ratio % n_batch_heads == 0 &&
           gqa_ratio / n_batch_heads <= WARP_SIZE;
}

static bool paged_attn_gqa_supported(int64_t n_head, int64_t n_head_kv) {
    if (n_head_kv <= 0 || n_head % n_head_kv != 0) {
        return false;
    }
    const int64_t gqa_ratio = n_head / n_head_kv;
    for (int n_batch_heads : PAGED_ATTN_BATCH_HEADS) {
        if (paged_attn_batch_heads_viable(gqa_ratio, n_batch_heads)) {
            return true;
        }
    }
    return false;
}

bool ggml_cuda_paged_attn_supported(const ggml_tensor * dst) {
    const ggml_tensor * q             = dst->src[0];
    const ggml_tensor * k             = dst->src[1];
    const ggml_tensor * v             = dst->src[2];
    const ggml_tensor * block_table   = dst->src[3];
    const ggml_tensor * kv_seq_lens   = dst->src[4];
    const ggml_tensor * active_slot_ids = dst->src[5];
    const ggml_tensor * query_positions = dst->src[6];

    if (!q || !k || !v || !block_table || !kv_seq_lens) {
        return false;
    }

    // Ragged causal positions require the explicit row -> column mapping.
    if (query_positions && !active_slot_ids) {
        return false;
    }

    if (dst->type != GGML_TYPE_F32 ||
        q->type   != GGML_TYPE_F32 ||
        !paged_attn_type_supported(k->type) ||
        !paged_attn_type_supported(v->type) ||
        block_table->type != GGML_TYPE_I32 ||
        kv_seq_lens->type != GGML_TYPE_I32 ||
        (active_slot_ids && active_slot_ids->type != GGML_TYPE_I32) ||
        (query_positions && query_positions->type != GGML_TYPE_I32)) {
        return false;
    }

    if (q->nb[0] != sizeof(float) ||
        k->nb[0] != ggml_type_size(k->type) ||
        v->nb[0] != ggml_type_size(v->type) ||
        block_table->nb[0] != sizeof(int32_t) ||
        kv_seq_lens->nb[0] != sizeof(int32_t) ||
        (active_slot_ids && active_slot_ids->nb[0] != sizeof(int32_t)) ||
        (query_positions && query_positions->nb[0] != sizeof(int32_t)) ||
        dst->nb[0] != sizeof(float)) {
        return false;
    }

    if (dst->ne[0] != q->ne[0] ||
        dst->ne[1] != q->ne[1] ||
        dst->ne[2] != q->ne[2] ||
        dst->ne[3] != q->ne[3]) {
        return false;
    }

    if (q->ne[0] != PAGED_ATTN_HEAD_DIM ||
        q->ne[0] != k->ne[0] ||
        q->ne[0] != v->ne[0] ||
        k->ne[1] != v->ne[1] ||
        k->ne[2] <= 0 ||
        k->ne[2] != v->ne[2] ||
        !paged_attn_gqa_supported(q->ne[2], k->ne[2]) ||
        q->ne[3] != 1 ||
        k->ne[3] != 1 ||
        v->ne[3] != 1) {
        return false;
    }

    if (block_table->ne[0] <= 0 ||
        block_table->ne[1] != kv_seq_lens->ne[0] ||
        block_table->ne[2] != 1 ||
        block_table->ne[3] != 1 ||
        kv_seq_lens->ne[1] != 1 ||
        kv_seq_lens->ne[2] != 1 ||
        kv_seq_lens->ne[3] != 1) {
        return false;
    }

    // Compacted batches address slots through active_slot_ids (one entry per
    // query token); dense batches require one table column per query token.
    if (active_slot_ids
            ? (active_slot_ids->ne[0] != q->ne[1] ||
               active_slot_ids->ne[1] != 1 ||
               active_slot_ids->ne[2] != 1 ||
               active_slot_ids->ne[3] != 1)
            : block_table->ne[1] != q->ne[1]) {
        return false;
    }
    if (query_positions &&
        (query_positions->ne[0] != q->ne[1] ||
         query_positions->ne[1] != 1 ||
         query_positions->ne[2] != 1 ||
         query_positions->ne[3] != 1)) {
        return false;
    }

    const int32_t block_size = ggml_get_op_params_i32(dst, 1);
    const int32_t max_kv_seq_len = ggml_get_op_params_i32(dst, 2);
    return block_size > 0 &&
           max_kv_seq_len > 0 &&
           max_kv_seq_len <= k->ne[1] &&
           k->ne[1] % block_size == 0;
}

// Cached max resident blocks/SM for this instantiation at the given block
// width; 0 when the device cannot launch it. Deliberately queries the
// write_partials variant: occupancy only steers min_partitions, which is a
// partition count for that variant. The direct variant launches solely when
// the count collapses to one, where occupancy no longer influences the
// topology.
template<ggml_type type_K, ggml_type type_V, int n_batch_heads>
static int paged_attn_cached_occupancy(int device, int warps_per_block) {
    // 0 means "not queried yet"; UNLAUNCHABLE records a block size this
    // device rejects, so the caller's fallback is decided once rather than
    // on every decode step.
    constexpr int UNLAUNCHABLE = -1;
    static std::atomic<int>
        occupancy[GGML_CUDA_MAX_DEVICES][WARP_SIZE + 1] = {};
    std::atomic<int> & cached = occupancy[device][warps_per_block];
    int probe = cached.load(std::memory_order_acquire);
    if (probe == 0) {
        // A block size above the kernel's maxThreadsPerBlock makes CUDA
        // return cudaErrorInvalidValue rather than a zero occupancy, so
        // the status has to drive the caller's fallback instead of aborting.
        const cudaError_t err =
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &probe,
                paged_attn_decode<PAGED_ATTN_HEAD_DIM, type_K, type_V,
                                  n_batch_heads, true>,
                WARP_SIZE * warps_per_block, 0);
        if (err != cudaSuccess) {
            probe = UNLAUNCHABLE;
            // Keep the rejected query out of the context's error state so
            // the next unrelated CUDA_CHECK does not inherit it.
            (void) cudaGetLastError();
        }
        cached.store(probe > 0 ? probe : UNLAUNCHABLE,
                     std::memory_order_release);
    }
    return probe > 0 ? probe : 0;
}

// Attempts a launch with n_batch_heads query heads per warp. Returns false
// when this width cannot reach a viable occupancy on the device (register
// pressure grows with the head batch), so the caller can fall back to a
// narrower instantiation.
template<ggml_type type_K, ggml_type type_V, int n_batch_heads>
static bool try_launch_paged_attn(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        float scale,
        int32_t block_size,
        int32_t max_kv_seq_len) {
    constexpr int D = PAGED_ATTN_HEAD_DIM;
    const ggml_tensor * q            = dst->src[0];
    const ggml_tensor * k            = dst->src[1];
    const ggml_tensor * v            = dst->src[2];
    const ggml_tensor * block_table  = dst->src[3];
    const ggml_tensor * kv_seq_lens = dst->src[4];
    const ggml_tensor * active_slot_ids = dst->src[5];
    const ggml_tensor * query_positions = dst->src[6];

    const int32_t n_head    = (int32_t) q->ne[2];
    const int32_t n_head_kv = (int32_t) k->ne[2];
    const int32_t gqa_ratio = n_head / n_head_kv;
    GGML_ASSERT(gqa_ratio % n_batch_heads == 0);
    const int32_t warps_per_group = gqa_ratio / n_batch_heads;

    // Colocate the warps of one K/V group, then pack further K/V groups into
    // the same block up to PAGED_ATTN_MAX_PACKED_WARPS warps so small groups
    // still fill a block.
    int32_t kv_heads_per_block = 1;
    for (int32_t candidate = 2; candidate <= n_head_kv; ++candidate) {
        if (n_head_kv % candidate == 0 &&
            warps_per_group * candidate <= PAGED_ATTN_MAX_PACKED_WARPS) {
            kv_heads_per_block = candidate;
        }
    }

    int max_blocks_per_sm = 0;
    while (true) {
        const int warps_per_block = warps_per_group * kv_heads_per_block;
        // The occupancy cache is indexed by the CUDA block's warp count.
        // Values above WARP_SIZE are both unlaunchable (>1024 threads) and
        // outside the cache's [0, WARP_SIZE] range, so reject this
        // specialization before touching the cache and let the caller try a
        // narrower head batch.
        if (warps_per_block > WARP_SIZE) {
            return false;
        }
        max_blocks_per_sm =
            paged_attn_cached_occupancy<type_K, type_V, n_batch_heads>(
                ctx.device, warps_per_block);
        if (max_blocks_per_sm > 0 || kv_heads_per_block == 1) {
            break;
        }
        do {
            kv_heads_per_block /= 2;
        } while (kv_heads_per_block > 1 &&
                 n_head_kv % kv_heads_per_block != 0);
    }
    if (max_blocks_per_sm == 0) {
        return false;
    }

    GGML_ASSERT(n_head_kv % kv_heads_per_block == 0);
    const int32_t head_groups = n_head_kv / kv_heads_per_block;
    const dim3 block(
        WARP_SIZE,
        (unsigned int) (warps_per_group * kv_heads_per_block),
        1);

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
    // Small batches need more context partitions each to expose enough work;
    // large batches already fill the device, where extra partitions mostly
    // repeat the per-partition fixed costs. Scale the cap so the total
    // partition count stays roughly constant across batch sizes.
    int32_t partition_limit =
        PAGED_ATTN_MAX_PARTITIONS / (int32_t) q->ne[1];
    if (partition_limit < 32) {
        partition_limit = 32;
    }
    if (min_partitions > partition_limit) {
        min_partitions = partition_limit;
    }
    if (min_partitions > block_table->ne[0]) {
        min_partitions = (int32_t) block_table->ne[0];
    }

    // Size the launch from the live maximum sequence length carried in the
    // graph op, not the block-table capacity. Ragged sequences still clamp
    // their own active partition count from kv_seq_lens on device.
    const int32_t live_blocks =
        (max_kv_seq_len + block_size - 1) / block_size;
    int32_t n_partitions = paged_attn_partitions(
        live_blocks, min_partitions, PAGED_ATTN_MAX_PARTITIONS);

    // Test/debug override used to exercise both the direct and partials paths
    // independently of device-specific occupancy. Values outside the valid
    // grid range are ignored. Read once: every full-attention layer launches
    // this on every decode token, so an environment scan per launch would sit
    // in the decode hot path.
    static const int forced_partitions = []() {
        const char * env =
            std::getenv("GGML_CUDA_PAGED_ATTN_FORCE_PARTITIONS");
        return env ? std::atoi(env) : 0;
    }();
    if (forced_partitions >= 1 &&
        forced_partitions <= PAGED_ATTN_MAX_PARTITIONS &&
        forced_partitions <= block_table->ne[0]) {
        min_partitions = forced_partitions;
        n_partitions = forced_partitions;
    }

    constexpr bool quantize_q = type_K != GGML_TYPE_F16;
    int    * q_i32_glob = nullptr;
    float2 * q_ds_glob  = nullptr;
    ggml_cuda_pool_alloc<int>    q_i32_alloc(ctx.pool());
    ggml_cuda_pool_alloc<float2> q_ds_alloc(ctx.pool());
    if (quantize_q) {
        q_i32_glob = q_i32_alloc.alloc(output_rows * (D / sizeof(int)));
        q_ds_glob  = q_ds_alloc.alloc(output_rows * (D / QK8_1));
        const dim3 quantize_grid(
            (unsigned int) q->ne[2], (unsigned int) q->ne[1], 1);
        paged_attn_quantize_q<D>
            <<<quantize_grid, dim3(WARP_SIZE, 1, 1), 0, ctx.stream()>>>(
            (const char *) q->data,
            q_i32_glob,
            q_ds_glob,
            q->nb[1], q->nb[2],
            (int32_t) q->ne[1],
            scale * PAGED_ATTN_LOG2E);
    }

    const dim3 grid(
        (unsigned int) head_groups,
        (unsigned int) q->ne[1],
        (unsigned int) n_partitions);

    ggml_cuda_pool_alloc<half>   acc_scratch(ctx.pool());
    ggml_cuda_pool_alloc<float2> meta_scratch(ctx.pool());
    half   * partial_acc  = nullptr;
    float2 * partial_meta = nullptr;
    if (n_partitions > 1) {
        const size_t partial_rows = (size_t) output_rows * n_partitions;
        partial_acc  = acc_scratch.alloc(partial_rows * D);
        partial_meta = meta_scratch.alloc(partial_rows);
    }

    auto * decode_kernel = n_partitions == 1
        ? paged_attn_decode<D, type_K, type_V, n_batch_heads, false>
        : paged_attn_decode<D, type_K, type_V, n_batch_heads, true>;
    decode_kernel<<<grid, block, 0, ctx.stream()>>>(
        (const char *) q->data,
        (const char *) k->data,
        (const char *) v->data,
        q_i32_glob,
        q_ds_glob,
        (const char *) block_table->data,
        (const char *) kv_seq_lens->data,
        active_slot_ids ? (const char *) active_slot_ids->data : nullptr,
        query_positions ? (const char *) query_positions->data : nullptr,
        (char *) dst->data,
        partial_acc,
        partial_meta,
        q->nb[1], q->nb[2],
        k->nb[1], k->nb[2],
        v->nb[1], v->nb[2],
        block_table->nb[0], block_table->nb[1],
        kv_seq_lens->nb[0],
        active_slot_ids ? active_slot_ids->nb[0] : 0,
        query_positions ? query_positions->nb[0] : 0,
        dst->nb[1], dst->nb[2],
        (int32_t) block_table->ne[1],
        n_head,
        n_head_kv,
        (int32_t) k->ne[1],
        (int32_t) block_table->ne[0],
        block_size,
        min_partitions,
        scale);

    if (n_partitions > 1) {
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
            n_partitions);
    }
    return true;
}

template<ggml_type type_K, ggml_type type_V>
static void launch_paged_attn(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        float scale,
        int32_t block_size,
        int32_t max_kv_seq_len) {
    const int32_t gqa_ratio =
        (int32_t) (dst->src[0]->ne[2] / dst->src[1]->ne[2]);

    // Widest head batch first: register pressure can make a wide batch
    // unlaunchable on some devices, so each width falls back to the next.
    if (paged_attn_batch_heads_viable(gqa_ratio, PAGED_ATTN_BATCH_HEADS[0]) &&
        try_launch_paged_attn<type_K, type_V, PAGED_ATTN_BATCH_HEADS[0]>(
            ctx, dst, scale, block_size, max_kv_seq_len)) {
        return;
    }
    if (paged_attn_batch_heads_viable(gqa_ratio, PAGED_ATTN_BATCH_HEADS[1]) &&
        try_launch_paged_attn<type_K, type_V, PAGED_ATTN_BATCH_HEADS[1]>(
            ctx, dst, scale, block_size, max_kv_seq_len)) {
        return;
    }
    if (paged_attn_batch_heads_viable(gqa_ratio, PAGED_ATTN_BATCH_HEADS[2]) &&
        try_launch_paged_attn<type_K, type_V, PAGED_ATTN_BATCH_HEADS[2]>(
            ctx, dst, scale, block_size, max_kv_seq_len)) {
        return;
    }
    GGML_ABORT("paged attention kernel has zero occupancy");
}

template<ggml_type type_K>
static void launch_paged_attn_v(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        float scale,
        int32_t block_size,
        int32_t max_kv_seq_len) {
    switch (dst->src[2]->type) {
        case GGML_TYPE_F16:
            launch_paged_attn<type_K, GGML_TYPE_F16>(
                ctx, dst, scale, block_size, max_kv_seq_len);
            break;
        case GGML_TYPE_Q4_0:
            launch_paged_attn<type_K, GGML_TYPE_Q4_0>(
                ctx, dst, scale, block_size, max_kv_seq_len);
            break;
        case GGML_TYPE_Q8_0:
            launch_paged_attn<type_K, GGML_TYPE_Q8_0>(
                ctx, dst, scale, block_size, max_kv_seq_len);
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
    const int32_t max_kv_seq_len = ggml_get_op_params_i32(dst, 2);

    switch (dst->src[1]->type) {
        case GGML_TYPE_F16:
            launch_paged_attn_v<GGML_TYPE_F16>(
                ctx, dst, scale, block_size, max_kv_seq_len);
            break;
        case GGML_TYPE_Q4_0:
            launch_paged_attn_v<GGML_TYPE_Q4_0>(
                ctx, dst, scale, block_size, max_kv_seq_len);
            break;
        case GGML_TYPE_Q8_0:
            launch_paged_attn_v<GGML_TYPE_Q8_0>(
                ctx, dst, scale, block_size, max_kv_seq_len);
            break;
        default:
            GGML_ABORT("unsupported paged-attention K type: %s",
                       ggml_type_name(dst->src[1]->type));
    }
    CUDA_CHECK(cudaGetLastError());
}
