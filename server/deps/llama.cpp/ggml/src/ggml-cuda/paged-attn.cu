#include "paged-attn.cuh"

#include "fattn-common.cuh"
#include "mma.cuh"

#include <atomic>
#include <cfloat>
#include <cstdlib>
#include <cstring>

static constexpr int PAGED_ATTN_MAX_PARTITIONS = 128;
static constexpr int PAGED_ATTN_BLOCKS_PER_PARTITION = 64;
static constexpr int PAGED_ATTN_HEAD_DIM = 256;
static constexpr int PAGED_ATTN_MAX_PACKED_WARPS = 8;

static __host__ __device__ __forceinline__ int32_t paged_attn_ceil_div(
        int64_t dividend, int32_t divisor) {
    return (int32_t) (dividend / divisor + (dividend % divisor != 0));
}

// Partition count for n_blocks of context: enough partitions to cover the
// blocks and to reach min_partitions for occupancy, but never more than the
// blocks themselves or the cap. Host and device must agree on this so the
// launched grid matches the per-sequence active partition count.
static __host__ __device__ __forceinline__ int32_t paged_attn_partitions(
        int32_t n_blocks, int32_t min_partitions, int32_t cap) {
    const int32_t context_partitions =
        paged_attn_ceil_div(n_blocks, PAGED_ATTN_BLOCKS_PER_PARTITION);
    const int32_t requested =
        context_partitions > min_partitions
            ? context_partitions
            : min_partitions;
    const int32_t available = n_blocks < cap ? n_blocks : cap;
    return requested < available ? requested : available;
}

// parent_ids is a sequence-major [tree_width, n_tree_seq] table. Parents
// precede children in the flat tree, and the root parent is -1.
static __device__ __forceinline__ bool paged_attn_tree_visible(
        const char * __restrict__ parent_ids,
        int64_t parent_nb0,
        int64_t parent_nb1,
        int32_t tree_seq,
        int32_t query_node,
        int32_t candidate,
        int32_t tree_size) {
    if (candidate < 0 || candidate >= tree_size ||
        query_node < 0 || query_node >= tree_size) {
        return false;
    }

    bool visible = false;
    int32_t current = query_node;
    for (int32_t depth = 0; depth < tree_size; ++depth) {
        if (current == candidate) {
            visible = true;
        }
        if (current < 0 || current >= tree_size) {
            return false;
        }
        const int32_t parent = *(const int32_t *) (
            parent_ids + (int64_t) current * parent_nb0 +
            (int64_t) tree_seq * parent_nb1);
        if (parent == -1) {
            return visible;
        }
        if (parent < 0 || parent >= current) {
            return false;
        }
        current = parent;
    }
    return false;
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
        const char    * __restrict__ parent_ids,
        const char    * __restrict__ tree_sizes,
        char          * __restrict__ dst,
        half          * __restrict__ partial_acc,
        float2        * __restrict__ partial_meta,
        int64_t q_nb1,   int64_t q_nb2,
        int64_t k_nb1,   int64_t k_nb2,
        int64_t v_nb1,   int64_t v_nb2,
        int64_t bt_nb0,  int64_t bt_nb1,
        int64_t ksl_nb0,
        int64_t asi_nb0, int64_t qpos_nb0,
        int64_t parent_nb0, int64_t parent_nb1, int64_t tree_size_nb0,
        int64_t dst_nb1, int64_t dst_nb2,
        int32_t n_table_seq,
        int32_t n_head,
        int32_t n_head_kv,
        int32_t pool_tokens,
        int32_t max_blocks,
        int32_t block_size,
        int32_t min_partitions,
        int32_t tree_width,
        int32_t tree_row_offset,
        int32_t tree_scratch_base,
        int32_t tree_scratch_stride,
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

    const bool tree_mode = parent_ids != nullptr;
    const int32_t physical_seq_raw = active_slot_ids
        ? *(const int32_t *) (active_slot_ids + (int64_t) seq * asi_nb0)
        : seq;
    const int32_t query_pos = query_positions
        ? *(const int32_t *) (query_positions + (int64_t) seq * qpos_nb0)
        : -1;
    const bool tree_query = tree_mode && seq >= tree_row_offset;
    const int32_t tree_seq = tree_query
        ? (seq - tree_row_offset) / tree_width : 0;
    const int32_t query_node = tree_query
        ? seq - tree_row_offset - tree_seq * tree_width : -1;
    const int32_t tree_size = tree_query
        ? *(const int32_t *) (
            tree_sizes + (int64_t) tree_seq * tree_size_nb0)
        : 0;
    // A row is live when its slot id selects a real block-table column and,
    // for ragged batches, its causal position is non-negative. Tree padding
    // rows are validated by tree_sizes. Dead rows are pinned to column 0 with
    // an empty virtual context, so the block table and scratch are never read.
    const bool valid_query =
        physical_seq_raw >= 0 && physical_seq_raw < n_table_seq &&
        (!query_positions || tree_query || query_pos >= 0) &&
        (!tree_query ||
         (tree_size >= 0 && tree_size <= tree_width &&
          query_node < tree_size));
    const int32_t physical_seq = valid_query ? physical_seq_raw : 0;
    int32_t kv_seq_len_raw = valid_query
        ? *(const int32_t *) (kv_seq_lens +
                              (int64_t) physical_seq * ksl_nb0)
        : 0;
    // The inclusive clamp IS the causal mask for non-tree ragged rows. Tree
    // rows always read the whole committed prefix carried by kv_seq_lens.
    if (query_positions && !tree_query && query_pos < kv_seq_len_raw) {
        kv_seq_len_raw = query_pos + 1;
    }
    const int64_t table_capacity =
        (int64_t) max_blocks * block_size;
    const int32_t kv_seq_len = kv_seq_len_raw <= 0
        ? 0
        : (kv_seq_len_raw < table_capacity
            ? kv_seq_len_raw
            : (int32_t) table_capacity);
    // Treat the candidate slab as a virtual tail of tree_width tokens. The
    // normal partition split then covers prefix and tree candidates in one
    // stable softmax; invisible siblings/padding resolve to no physical row.
    const int64_t virtual_tokens = valid_query
        ? (int64_t) kv_seq_len + (tree_query ? tree_width : 0)
        : 0;
    const int32_t n_logical_blocks =
        paged_attn_ceil_div(virtual_tokens, block_size);
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
        virtual_tokens < token_end_blocks ? virtual_tokens : token_end_blocks;

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

    // In tree mode the committed block table may address only the prefix
    // pool before tree_scratch_base. Candidate rows are addressed directly
    // below, keeping uncommitted nodes out of every sequence block table.
    const int32_t prefix_pool_tokens =
        tree_mode ? tree_scratch_base : pool_tokens;
    const int32_t n_physical_blocks = prefix_pool_tokens / block_size;

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
        if (my_token < token_end && my_token < kv_seq_len) {
            const int32_t logical_block = my_token / block_size;
            const int32_t physical_block =
                *(const int32_t *) (block_table +
                    (int64_t) logical_block * bt_nb0 +
                    (int64_t) physical_seq * bt_nb1);
            if (physical_block >= 0 && physical_block < n_physical_blocks) {
                phys_mine =
                    physical_block * block_size + my_token % block_size;
            }
        } else if (tree_query && my_token < token_end) {
            const int32_t candidate = my_token - kv_seq_len;
            if (paged_attn_tree_visible(
                    parent_ids, parent_nb0, parent_nb1,
                    tree_seq, query_node, candidate, tree_size)) {
                const int64_t physical =
                    (int64_t) tree_scratch_base +
                    (int64_t) physical_seq * tree_scratch_stride +
                    candidate;
                if (physical >= 0 && physical < pool_tokens) {
                    phys_mine = (int32_t) physical;
                }
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
    const ggml_tensor * parent_ids      = dst->src[7];
    const ggml_tensor * tree_sizes      = dst->src[8];

    if (!q || !k || !v || !block_table || !kv_seq_lens) {
        return false;
    }

    // Ragged causal positions require the explicit row -> column mapping.
    if (query_positions && !active_slot_ids) {
        return false;
    }
    const bool tree_mode = parent_ids || tree_sizes;
    if ((parent_ids == nullptr) != (tree_sizes == nullptr) ||
        (tree_mode && !active_slot_ids)) {
        return false;
    }

    if (dst->type != GGML_TYPE_F32 ||
        q->type   != GGML_TYPE_F32 ||
        !paged_attn_type_supported(k->type) ||
        !paged_attn_type_supported(v->type) ||
        block_table->type != GGML_TYPE_I32 ||
        kv_seq_lens->type != GGML_TYPE_I32 ||
        (active_slot_ids && active_slot_ids->type != GGML_TYPE_I32) ||
        (query_positions && query_positions->type != GGML_TYPE_I32) ||
        (parent_ids && parent_ids->type != GGML_TYPE_I32) ||
        (tree_sizes && tree_sizes->type != GGML_TYPE_I32)) {
        return false;
    }

    if (q->nb[0] != sizeof(float) ||
        k->nb[0] != ggml_type_size(k->type) ||
        v->nb[0] != ggml_type_size(v->type) ||
        block_table->nb[0] != sizeof(int32_t) ||
        kv_seq_lens->nb[0] != sizeof(int32_t) ||
        (active_slot_ids && active_slot_ids->nb[0] != sizeof(int32_t)) ||
        (query_positions && query_positions->nb[0] != sizeof(int32_t)) ||
        (parent_ids && parent_ids->nb[0] != sizeof(int32_t)) ||
        (tree_sizes && tree_sizes->nb[0] != sizeof(int32_t)) ||
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
    const int32_t tree_width = ggml_get_op_params_i32(dst, 3);
    const int32_t tree_scratch_base = ggml_get_op_params_i32(dst, 4);
    const int32_t tree_scratch_stride = ggml_get_op_params_i32(dst, 5);
    if (block_size <= 0 ||
        max_kv_seq_len <= 0 ||
        (int64_t) max_kv_seq_len + tree_width > INT32_MAX ||
        k->ne[1] % block_size != 0) {
        return false;
    }

    if (!tree_mode) {
        return tree_width == 0 &&
               tree_scratch_base == 0 &&
               tree_scratch_stride == 0;
    }

    if (tree_width <= 0 ||
        tree_scratch_base <= 0 ||
        tree_scratch_base % block_size != 0 ||
        tree_scratch_stride < tree_width ||
        !ggml_is_contiguous(parent_ids) ||
        !ggml_is_contiguous(tree_sizes) ||
        parent_ids->ne[0] != tree_width ||
        parent_ids->ne[1] <= 0 ||
        parent_ids->ne[1] != tree_sizes->ne[0] ||
        parent_ids->ne[2] != 1 ||
        parent_ids->ne[3] != 1 ||
        tree_sizes->ne[1] != 1 ||
        tree_sizes->ne[2] != 1 ||
        tree_sizes->ne[3] != 1 ||
        parent_ids->ne[1] > INT64_MAX / tree_width ||
        q->ne[1] < parent_ids->ne[1] * tree_width ||
        (!query_positions &&
         q->ne[1] != parent_ids->ne[1] * tree_width) ||
        (int64_t) max_kv_seq_len + tree_width > INT32_MAX) {
        return false;
    }

    const int64_t scratch_end =
        (int64_t) tree_scratch_base +
        (block_table->ne[1] - 1) * (int64_t) tree_scratch_stride +
        tree_width;
    return scratch_end <= k->ne[1];
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
    const ggml_tensor * parent_ids      = dst->src[7];
    const ggml_tensor * tree_sizes      = dst->src[8];

    const int32_t tree_width = ggml_get_op_params_i32(dst, 3);
    const int32_t tree_scratch_base = ggml_get_op_params_i32(dst, 4);
    const int32_t tree_scratch_stride = ggml_get_op_params_i32(dst, 5);

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
    const int32_t tree_blocks = paged_attn_ceil_div(tree_width, block_size);
    const int64_t partitionable_blocks =
        block_table->ne[0] + (parent_ids ? tree_blocks : 0);
    if (min_partitions > partitionable_blocks) {
        min_partitions = (int32_t) partitionable_blocks;
    }

    // Size the launch from the live maximum committed prefix plus the virtual
    // tree tail. Ragged/tree rows still clamp their own active partition count
    // from device metadata.
    const int64_t live_tokens =
        (int64_t) max_kv_seq_len + (parent_ids ? tree_width : 0);
    const int32_t live_blocks =
        paged_attn_ceil_div(live_tokens, block_size);
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
        forced_partitions <= partitionable_blocks) {
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
        parent_ids ? (const char *) parent_ids->data : nullptr,
        tree_sizes ? (const char *) tree_sizes->data : nullptr,
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
        parent_ids ? parent_ids->nb[0] : 0,
        parent_ids ? parent_ids->nb[1] : 0,
        tree_sizes ? tree_sizes->nb[0] : 0,
        dst->nb[1], dst->nb[2],
        (int32_t) block_table->ne[1],
        n_head,
        n_head_kv,
        (int32_t) k->ne[1],
        (int32_t) block_table->ne[0],
        block_size,
        min_partitions,
        tree_width,
        parent_ids
            ? (int32_t)(q->ne[1] - parent_ids->ne[1] * tree_width)
            : 0,
        tree_scratch_base,
        tree_scratch_stride,
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

// ─── WMMA head-256 paged attention (RDNA4, stage 1) ────────────────────
// Tensor-core sibling of paged_attn_decode for the packed prefill path.
// One block covers 4 query rows x 8 head slots of ONE context partition.
// The 4 rows may belong to different sequences: the block loops over the
// unique sequences (<= 4 passes) and stages the K/V tiles once per
// sequence, masking other columns, so a packed prefill chunk (rows of one
// sequence) pays one K/V read per token for 32 columns instead of one row
// sweep per query row. Mirrors fattn-mma's fragment machinery (f16 WMMA,
// f32 accumulators, 8 warps, np=4 K-split) with block-table gathers
// replacing the contiguous KV loaders, in the log2 domain of the existing
// paged kernels (scale*log2(e) prescale, exp2f softmax) so
// paged_attn_combine and the partial convention are reused verbatim.

static thread_local size_t g_paged_attn_wmma256_launch_count = 0;
extern "C" void ggml_backend_cuda_record_paged_attn_wmma256_launch(void) {
    ++g_paged_attn_wmma256_launch_count;
}

extern "C" size_t ggml_backend_cuda_get_paged_attn_wmma256_launch_count(void) {
    return g_paged_attn_wmma256_launch_count;
}

// Vectorized paged K/V gather for one sequence and one KV head. Each thread
// owns 16 contiguous half2s (64 B) of one token row, so the block-table entry
// is resolved once per thread instead of once per (row, half2) element: F16
// rows are read as four 16-byte chunks, Q8_0 rows (the engine config) as one
// 32-dim block_q8_0 (34 B) with 4-byte qs loads. Writes the same
// tile[i*stride_tile + kk] layout as the scalar version; tokens beyond
// k_VKQ_sup or outside the validated block table yield zeros.
// Write one thread's segment of `vec` half2s to the stage tile as vec/4
// 16-byte stores (both quant branches produce the segment in registers first,
// the zero path writes zeros the same way).
template <int vec>
static __device__ __forceinline__ void paged_attn_wmma_store_segment(
        uint32_t * __restrict__ out, const uint32_t (&h)[vec]) {
#pragma unroll
    for (int m = 0; m < vec / 4; ++m) {
        *reinterpret_cast<uint4 *>(out + 4 * m) =
            make_uint4(h[4*m], h[4*m+1], h[4*m+2], h[4*m+3]);
    }
}

template <ggml_type type, int stride_tile, int nbatch_h2>
static __device__ __forceinline__ void paged_attn_wmma_stage_tile(
        const char * __restrict__ kv,
        const char * __restrict__ block_table,
        int64_t bt_nb0, int64_t bt_nb1,
        int64_t kv_nb1,
        int32_t kv_head, int32_t seq_s, int32_t block_size, int32_t pool_tokens,
        int64_t kv_nb2,
        int32_t token0, int32_t k_VKQ_sup,
        int32_t dim0,
        half2 * __restrict__ tile) {
    constexpr int vec = nbatch_h2 / 4;          // half2s owned by one thread
    static_assert(nbatch_h2 % 4 == 0 && vec % 4 == 0, "bad stage width");
    constexpr int qk = 32;                      // dims per Q8_0/Q4_0 block
    static_assert(type == GGML_TYPE_F16 || 2 * vec == qk,
                  "Q8_0/Q4_0 need one 32-dim block per thread");
    // Bytes per quant block: qs plus the f16 scale (Q4_0 16+2, Q8_0 32+2).
    constexpr int kblock_bytes = type == GGML_TYPE_Q4_0 ? qk/2 + 2 : qk + 2;

    const int warp_size = 32;
    const int tid = threadIdx.x + threadIdx.y * warp_size;
    const int row = tid / 4;                    // token row within the chunk
    const int sub = tid % 4;                    // 16-half2 segment of the row
    const int32_t n_physical_blocks = pool_tokens / block_size;

    const char * rowbase = nullptr;
    const int32_t token = token0 + row;
    if (row < k_VKQ_sup) {
        const int32_t logical_block = token / block_size;
        const int32_t physical_block = *(const int32_t *) (
            block_table + (int64_t) logical_block * bt_nb0 +
            (int64_t) seq_s * bt_nb1);
        if (physical_block >= 0 && physical_block < n_physical_blocks) {
            const int32_t phys = physical_block * block_size + token % block_size;
            rowbase = kv + (int64_t) phys * kv_nb1 + (int64_t) kv_head * kv_nb2;
        }
    }

    uint32_t * out = reinterpret_cast<uint32_t *>(tile + row * stride_tile + sub * vec);
    if (rowbase == nullptr) {
        const uint32_t zeros[vec] = {};
        paged_attn_wmma_store_segment<vec>(out, zeros);
        return;
    }

    if constexpr (type == GGML_TYPE_F16) {
        const uint4 * src = reinterpret_cast<const uint4 *>(
            rowbase + (size_t) (dim0 + sub * vec) * sizeof(half2));
#pragma unroll
        for (int m = 0; m < vec / 4; ++m) {
            *reinterpret_cast<uint4 *>(out + 4 * m) = src[m];
        }
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        // Q4_0: this segment is exactly one block_q4_0 = {half d; uint8_t qs[16]}
        // = 18 B. ggml layout: dims 0..15 are the low nibbles of qs[0..15],
        // dims 16..31 the high nibbles, each offset by -8 and scaled by d.
        const int blk_idx = (dim0 + sub * vec) / (qk / 2);
        const char * blk = rowbase + (size_t) blk_idx * kblock_bytes;
        const float d = __half2float(*(const half *) blk);
        uint32_t w[qk / 8];
#pragma unroll
        for (int j = 0; j < qk / 8; ++j) {
            memcpy(&w[j], blk + 2 + 4 * j, 4);
        }
        uint32_t h[vec];
#pragma unroll
        for (int m = 0; m < vec; ++m) {
            const int t0 = 2 * m, t1 = t0 + 1;
            const int b0 = t0 & (qk/2 - 1), b1 = t1 & (qk/2 - 1);
            const int q0 = (w[b0 >> 2] >> ((b0 & 3) * 8 + (t0 >= qk/2 ? 4 : 0))) & 0x0F;
            const int q1 = (w[b1 >> 2] >> ((b1 & 3) * 8 + (t1 >= qk/2 ? 4 : 0))) & 0x0F;
            const half2 hv = make_half2(d * (float) (q0 - 8), d * (float) (q1 - 8));
            h[m] = *reinterpret_cast<const uint32_t *>(&hv);
        }
        paged_attn_wmma_store_segment<vec>(out, h);
    } else {
        // Q8_0: this segment is exactly one block_q8_0 = {half d; int8 qs[32]}.
        const int blk_idx = (dim0 + sub * vec) / (qk / 2);
        const char * blk = rowbase + (size_t) blk_idx * kblock_bytes;
        const float d = __half2float(*(const half *) blk);
        uint32_t w[qk / 4];
#pragma unroll
        for (int j = 0; j < qk / 4; ++j) {
            memcpy(&w[j], blk + 2 + 4 * j, 4);
        }
        uint32_t h[vec];
#pragma unroll
        for (int m = 0; m < qk / 2; ++m) {
            const int e0 = 2 * m;
            const int8_t v0 = (int8_t) ((w[e0 >> 2] >> ((e0 & 3) * 8)) & 0xff);
            const int8_t v1 = (int8_t) ((w[(e0 + 1) >> 2] >> (((e0 + 1) & 3) * 8)) & 0xff);
            const half2 hv = make_half2(d * (float) v0, d * (float) v1);
            h[m] = *reinterpret_cast<const uint32_t *>(&hv);
        }
        paged_attn_wmma_store_segment<vec>(out, h);
    }
}

template <ggml_type type_K, ggml_type type_V>
static __device__ __forceinline__ void paged_attn_wmma_iter(
        const char  * __restrict__ k,
        const char  * __restrict__ v,
        const char  * __restrict__ block_table,
        int64_t bt_nb0, int64_t bt_nb1,
        int64_t k_nb1, int64_t k_nb2,
        int64_t v_nb1, int64_t v_nb2,
        int32_t kv_head, int32_t seq_s, int32_t block_size, int32_t pool_tokens,
        int32_t token_begin,
        const int32_t (& row_seq)[4],
        const int32_t (& row_extent)[4],
        half2 * __restrict__ tile_K,
        half2 * __restrict__ tile_V,
        ggml_cuda_mma::tile<16, 8, half2> * __restrict__ Q_B,
        ggml_cuda_mma::tile<16, 8, half2> * __restrict__ VKQ_C,
        float * __restrict__ KQ_max,
        float * __restrict__ KQ_rowsum,
        const int32_t kb0,
        const int32_t k_VKQ_sup) {
    using namespace ggml_cuda_mma;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int DKQ = 256, DV = 256;
    constexpr int ncols1 = 4, ncols2 = 8, nwarps = 8;
    constexpr int nbatch_fa = 64, nbatch_K2 = 64, nbatch_V2 = 64;
    constexpr int cols_per_warp = 16, cols_per_thread = 1, np = 4;
    static_assert(ncols1 == 4 && ncols2 == 8 && nwarps == 8 && np == 4 &&
                  nbatch_fa == 64 && nbatch_K2 == 64 && nbatch_V2 == 64,
                  "gfx1201 WMMA tile and mma operand order are hardcoded");
    using T_A_KQ  = tile<16,  8, half2>;
    using T_B_KQ  = tile<16,  8, half2>;
    using T_C_KQ  = tile<16, 16, float>;
    using T_A_VKQ = tile<16,  8, half2>;
    using T_B_VKQ = tile<16,  8, half2>;
    using T_C_VKQ = tile<16,  8, half2>;
    constexpr int stride_tile_K = nbatch_K2 + 4;
    constexpr int stride_tile_V = nbatch_V2 + 4;
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)) || defined(AMD_MFMA_AVAILABLE)
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
#pragma unroll
    for (int i = 0; i < nbatch_fa/(np*T_C_KQ::J); ++i) {
#pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            KQ_C[i].x[l] = 0.0f;
        }
    }

    // KQ phase: stage the K tile for this chunk and accumulate the
    // score matrix with the Q fragments held in registers.
#pragma unroll
    for (int k0_start = (DKQ/2-1) - (DKQ/2-1) % nbatch_K2; k0_start >= 0; k0_start -= nbatch_K2) {
        const int k0_stop = k0_start + nbatch_K2 < DKQ/2 ? k0_start + nbatch_K2 : DKQ/2;

        {
            paged_attn_wmma_stage_tile<type_K, stride_tile_K, nbatch_K2>(
                k, block_table, bt_nb0, bt_nb1, k_nb1, kv_head, seq_s, block_size,
                pool_tokens, k_nb2, token_begin + kb0*nbatch_fa, k_VKQ_sup, k0_start, tile_K);
            __syncthreads();
        }

        {
#pragma unroll
            for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;
#pragma unroll
                for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                    T_A_KQ K_A;
                    load_ldmatrix(K_A, tile_K + i_KQ_0*stride_tile_K + (k_KQ_0 - k0_start), stride_tile_K);
                    mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
                }
            }
        }

        {
            __syncthreads(); // tile_V reuses the tile_K smem in the VKQ phase.
        }
    }

    float KQ_max_new[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};

    {
        // Per-row causal/sequence mask, computed inline (was a staged tile):
        // token pair (2i, 2i+1) is visible iff its seq matches and both the
        // row extent and the chunk bound cover the token.
        const int32_t chunk_tok0 = token_begin + kb0*nbatch_fa;
#pragma unroll
        for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::J) {
            const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::J;
#pragma unroll
            for (int l0 = 0; l0 < T_C_KQ::ne; l0 += 2) {
                const int i = (i0 + T_C_KQ::get_j(l0)) / 2;
                const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l0)) / ncols2;

                const int32_t e = (row_seq[j] == seq_s) ? row_extent[j] : -1;
                const int32_t t0 = chunk_tok0 + 2*i;
                const float2 tmp = make_float2(
                    (t0     < e && t0     < chunk_tok0 + k_VKQ_sup) ? 0.0f : -FLT_MAX,
                    (t0 + 1 < e && t0 + 1 < chunk_tok0 + k_VKQ_sup) ? 0.0f : -FLT_MAX);
                KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 0] += tmp.x;
                KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 1] += tmp.y;
            }
        }
    }

    // Softmax in the log2 domain: log2(e) is folded into the Q prescale.
    static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            if (k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
                KQ_max_new[0] = fmaxf(KQ_max_new[0], KQ_C[(k0/(np*T_C_KQ::J))].x[l]);
            }
        }
    }

#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], 16, warp_size));
    }

    static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
            if (k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
                KQ_C[(k0/(np*T_C_KQ::J))].x[l] = exp2f(KQ_C[(k0/(np*T_C_KQ::J))].x[l] - KQ_max_new[0]);
                KQ_rowsum_add[0] += KQ_C[(k0/(np*T_C_KQ::J))].x[l];
            } else {
                KQ_C[(k0/(np*T_C_KQ::J))].x[l] = 0.0f;
            }
        }
    }

    {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = exp2f(KQ_max_diff);
            KQ_max[col] = KQ_max_new[col];

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
        }

        const half2 KQ_max_scale_h2 = make_half2(
            KQ_max_scale[0], KQ_max_scale[0]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
    }

    // Convert KQ C tiles into B tiles for the VKQ calculation:
    T_B_VKQ B[nbatch_fa/(np*2*T_B_VKQ::J)];
    static_assert(nbatch_fa % (np*2*T_B_VKQ::J) == 0, "bad loop size");
    {
#pragma unroll
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_half2(KQ_C[k]);
        }
    }

#if defined(AMD_WMMA_AVAILABLE) && !defined(LDMATRIX_TRANS_AVAILABLE)
    T_A_VKQ A_identity;
    make_identity_mat(A_identity);
#endif // defined(AMD_WMMA_AVAILABLE) && !defined(LDMATRIX_TRANS_AVAILABLE)

    // VKQ phase: stage the V tile (reusing the K smem) and accumulate.
#pragma unroll
    for (int i0_start = 0; i0_start < DV; i0_start += 2*nbatch_V2) {
        static_assert(DV % (2*nbatch_V2) == 0, "bad loop size");
        const int i0_stop = i0_start + 2*nbatch_V2;

        {
            paged_attn_wmma_stage_tile<type_V, stride_tile_V, nbatch_V2>(
                v, block_table, bt_nb0, bt_nb1, v_nb1, kv_head, seq_s, block_size,
                pool_tokens, v_nb2, token_begin + kb0*nbatch_fa, k_VKQ_sup, i0_start/2, tile_V);
            __syncthreads();
        }
        const half2 * tile_V_i = tile_V;

        constexpr int i0_stride = 2*T_C_VKQ::J;
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += i0_stride) {
            static_assert((nbatch_fa/2) % (np*T_A_VKQ::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa/2; k00 += np*T_A_VKQ::J) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::J;

                T_A_VKQ A; // Transposed in SRAM but not in registers, gets transposed on load.
#if defined(LDMATRIX_TRANS_AVAILABLE)
                load_ldmatrix_trans(A, tile_V_i + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2, stride_tile_V);
#else
                // Use mma to transpose T_A_VKQ for RDNA.
                T_A_VKQ A_trans;
                load_ldmatrix(A_trans, tile_V_i + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2, stride_tile_V);
                mma(A, A_trans, A_identity);
#endif // defined(LDMATRIX_TRANS_AVAILABLE)

                mma(VKQ_C[i_VKQ_0/i0_stride], A, B[k00/(np*T_A_VKQ::J)]);
            }
        }

        {
            __syncthreads();
        }
    }

#else
    GGML_UNUSED_VARS(k, v, block_table, bt_nb0, bt_nb1, k_nb1, k_nb2, v_nb1, v_nb2,
        kv_head, seq_s, block_size, pool_tokens, token_begin, row_seq, row_extent,
        tile_K, tile_V, Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0, k_VKQ_sup);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)) || defined(AMD_MFMA_AVAILABLE)
}template <ggml_type type_K, ggml_type type_V>
__launch_bounds__(256, 2)
static __global__ void paged_attn_wmma(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const char * __restrict__ block_table,
        const char * __restrict__ kv_seq_lens,
        const char * __restrict__ active_slot_ids,
        const char * __restrict__ query_positions,
        char       * __restrict__ dst,
        half       * __restrict__ partial_acc,
        float2     * __restrict__ partial_meta,
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
        int32_t n_rows,
        int32_t min_partitions,
        int32_t n_partitions,
        int32_t write_partials,
        float scale) {
    using namespace ggml_cuda_mma;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int DKQ = 256, DV = 256;
    constexpr int ncols1 = 4, ncols2 = 8, nwarps = 8;
    constexpr int ncols = ncols1 * ncols2;
    constexpr int nbatch_fa = 64, nbatch_K2 = 64, nbatch_V2 = 64, nbatch_combine = 32;
    constexpr int cols_per_warp = 16, cols_per_thread = 1, np = 4;
    static_assert(ncols1 == 4 && ncols2 == 8 && nwarps == 8 && np == 4 &&
                  nbatch_fa == 64 && nbatch_K2 == 64 && nbatch_V2 == 64,
                  "gfx1201 WMMA tile and mma operand order are hardcoded");
    using T_B_KQ  = tile<16,  8, half2>;
    using T_C_KQ  = tile<16, 16, float>;
    using T_C_VKQ = tile<16,  8, half2>;
    constexpr int stride_tile_Q = DKQ/2 + 4;

    const int gqa_ratio = n_head / n_head_kv;
    const int kv_head   = blockIdx.x;
    const int partition = blockIdx.z;
    const int group_row0 = ncols1 * (int) blockIdx.y;
    const int group_rows = n_rows - group_row0 < ncols1 ? n_rows - group_row0 : ncols1;

    // ── Per-row metadata (mirrors paged_attn_decode :275-330) ──
    int32_t row_seq[4];
    int32_t row_extent[4];
    {
        const bool has_pos = query_positions != nullptr;
#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const int row = group_row0 + j;
            int32_t extent = -1;
            if (row < group_row0 + group_rows && row < n_rows) {
                // Dense batches pass no slot ids: the query row selects its own
                // block-table column, exactly like paged_attn_decode :276-301.
                const int32_t slot = active_slot_ids != nullptr
                    ? *(const int32_t *) (active_slot_ids + (int64_t) row * asi_nb0)
                    : row;
                row_seq[j] = slot;
                if (slot >= 0 && slot < n_table_seq) {
                    int32_t kv_len = *(const int32_t *) (kv_seq_lens + (int64_t) slot * ksl_nb0);
                    if (has_pos) {
                        const int32_t pos = *(const int32_t *) (query_positions + (int64_t) row * qpos_nb0);
                        if (pos < 0) {
                            // Ragged rows without a causal position are invalid
                            // outside tree mode (decode valid_query :295-300):
                            // dead row, empty context, zero output.
                            kv_len = 0;
                        } else if (pos < kv_len) {
                            kv_len = pos + 1;
                        }
                    }
                    const int32_t cap = max_blocks * block_size;
                    extent = kv_len > 0 ? (kv_len < cap ? kv_len : cap) : 0;
                }
            } else {
                row_seq[j] = -1;
            }
            row_extent[j] = extent;
        }
    }

    // Pin one (row, head) slot this block owns when it carries no work. The
    // combine treats meta.y > 0 as live, so a dead slot must be written
    // explicitly: (-FLT_MAX, 0) for the partials path, zeros for the direct
    // path (mirrors paged_attn_decode :329-348).
    auto pin_slot = [&](int j, int c) {
        const int row  = group_row0 + j;
        const int head = kv_head*gqa_ratio + c;
        if (row >= n_rows || head >= n_head) {
            return;
        }
        if (write_partials) {
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                const int64_t output_row = (int64_t) head * n_rows + row;
                partial_meta[output_row * n_partitions + partition] =
                    make_float2(-FLT_MAX, 0.0f);
            }
        } else {
            float * o_row = (float *) (dst + (int64_t) row * dst_nb1 + (int64_t) head * dst_nb2);
#pragma unroll
            for (int i = threadIdx.x; i < DKQ; i += nwarps * warp_size) {
                o_row[i] = 0.0f;
            }
        }
    };

    // ── Dead rows (extent <= 0) are skipped by the write-back guard, so pin
    //    them here for every partition this block covers ──
#pragma unroll
    for (int j = 0; j < ncols1; ++j) {
        if (row_extent[j] > 0) {
            continue;
        }
#pragma unroll
        for (int c = 0; c < gqa_ratio; ++c) {
            pin_slot(j, c);
        }
    }

    // ── Partition token range (mirrors paged_attn_decode :324-348) ──
    // The divisor is the ACTIVE partition count, not the grid size: small
    // contexts have fewer live partitions, and dividing by the full count
    // collapses the range to empty (every block then returned early and
    // the combine read uninitialized partial meta).
    int32_t max_extent = 0;
#pragma unroll
    for (int j = 0; j < ncols1; ++j) {
        if (row_extent[j] > max_extent) {
            max_extent = row_extent[j];
        }
    }
    if (max_extent <= 0) {
        return;
    }
    const int32_t n_logical_blocks = paged_attn_ceil_div(max_extent, block_size);
    const int32_t active_partitions =
        paged_attn_partitions(n_logical_blocks, min_partitions, n_partitions);

    if (partition >= active_partitions) {
        // Dead partition: pin the live rows' slots so the combine never
        // reads uninitialized pool memory.
#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            if (row_extent[j] <= 0) {
                continue;
            }
#pragma unroll
            for (int c = 0; c < gqa_ratio; ++c) {
                pin_slot(j, c);
            }
        }
        return;
    }

    const int32_t logical_block_begin =
        ((int64_t) n_logical_blocks * partition) / active_partitions;
    const int32_t logical_block_end =
        ((int64_t) n_logical_blocks * (partition + 1)) / active_partitions;
    const int32_t token_begin = logical_block_begin * block_size;
    const int32_t token_end_blocks = logical_block_end * block_size;
    const int32_t token_end =
        max_extent < token_end_blocks ? max_extent : token_end_blocks;
    const int32_t token_count = token_end - token_begin;
    if (token_count <= 0) {
        return;
    }
    const int32_t kb0_stop = (token_count + nbatch_fa - 1) / nbatch_fa;
    const int32_t kb0_start = 0;
    const float2 * Q_f2 = (const float2 *) (q + (int64_t) group_row0 * q_nb1 +
                                            (int64_t) kv_head * gqa_ratio * q_nb2);

    extern __shared__ half2 tile_Q[];
    half2 * tile_K    = tile_Q;                 // K reuses the Q smem (Q lives in registers)
    half2 * tile_V    = tile_K;                 // single stage: V reuses K smem

    T_B_KQ    Q_B[DKQ/(2*T_B_KQ::J)];
#if defined(TURING_MMA_AVAILABLE)
    T_C_VKQ VKQ_C[cols_per_warp == 8 ? DV/T_C_VKQ::I : DV/(2*T_C_VKQ::J)];
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
#else // Volta
    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)

    {
        constexpr int n_vkq = DV/(2*T_C_VKQ::J);
        for (int i = 0; i < n_vkq; ++i) {
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] = make_half2(0.0f, 0.0f);
            }
        }
    }

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    // Load Q data into tile_Q, either temporarily or permanently.
    // Q in registers is faster, but register pressure is the biggest bottleneck.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    // Log2 domain: fold log2(e) into the prescale exactly like the
    // decode kernel (scale * PAGED_ATTN_LOG2E), so the exp2f softmax
    // matches the reference softmax.
    const half2 scale_h2 = make_half2(scale * PAGED_ATTN_LOG2E, scale * PAGED_ATTN_LOG2E);
#pragma unroll
    for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
        const int k0_start  = stride_k == warp_size ? 0 : DKQ/2 - (DKQ/2) % (2*stride_k);
        const int k0_stop   =                             DKQ/2 - (DKQ/2) % (1*stride_k);
        const int stride_jc = warp_size / stride_k;

        if (k0_start == k0_stop) {
            continue;
        }

#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
            const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

            if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                break;
            }

            const int j = jc / ncols2;
            const int c = jc % ncols2;

            if (j < group_rows && c < gqa_ratio) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[j*(q_nb1/8) + c*(q_nb2/8) + k];
                    tile_Q[jc*stride_tile_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    tile_Q[jc*stride_tile_Q + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    __syncthreads();

    {
        const int j0 = (threadIdx.y / np) * cols_per_warp;

#pragma unroll
        for (int k0 = 0; k0 < DKQ/2; k0 += T_B_KQ::J) {
            load_ldmatrix(Q_B[k0/T_B_KQ::J], tile_Q + j0*stride_tile_Q + k0, stride_tile_Q);
        }
    }

    __syncthreads();

    // ── Per-sequence passes over the partition token range ──
    for (int pass = 0; pass < ncols1; ++pass) {
        const int32_t seq_s = row_seq[pass];
        if (seq_s < 0) {
            continue;
        }
        bool dup = false;
        int32_t seq_max_extent = -1;
#pragma unroll
        for (int jj = 0; jj < ncols1; ++jj) {
            if (row_seq[jj] == seq_s) {
                if (jj < pass) {
                    dup = true;
                }
                if (row_extent[jj] > seq_max_extent) {
                    seq_max_extent = row_extent[jj];
                }
            }
        }
        if (dup || seq_max_extent < token_begin) {
            continue;
        }

        for (int kb0 = kb0_start; kb0 < kb0_stop; ++kb0) {
        // Live tokens in this chunk, clamped to the partition's own token
        // range: without the clamp the extent mask alone lets a token inside
        // the extent but outside the partition leak into the neighbouring
        // partition's chunk (both stage it, the combine double-counts it).
        const int32_t chunk_start = token_begin + kb0*nbatch_fa;
        const int32_t k_VKQ_sup =
            min(nbatch_fa, max(0, token_end - chunk_start));
        paged_attn_wmma_iter<type_K, type_V>(
            k, v, block_table, bt_nb0, bt_nb1, k_nb1, k_nb2, v_nb1, v_nb2, kv_head, seq_s,
            block_size, pool_tokens, token_begin,
            row_seq, row_extent,
            tile_K, tile_V, Q_B, VKQ_C, KQ_max, KQ_rowsum,
            kb0, k_VKQ_sup);
    }

    }

    // Finally, sum up partial KQ rowsums.
    {
#if defined(TURING_MMA_AVAILABLE)
        // The partial sums are spread across 8/4 threads.
        constexpr int offset_first = cols_per_warp == 8 ? 16 : 2;
        constexpr int offset_last  = cols_per_warp == 8 ?  4 : 1;
#elif defined(AMD_MFMA_AVAILABLE)
        // The partial sums are spread across 4 threads (wavefront64, 16 cols).
        constexpr int offset_first = 32;
        constexpr int offset_last  = 16;
#elif defined(AMD_WMMA_AVAILABLE)
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 16;
        constexpr int offset_last  = 16;
#else // Volta
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 2;
        constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, warp_size);
            }
        }
    }


    // Combine VKQ accumulator values if np > 1.
    // It's also faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.
    // So also write VKQ accumulators to shared memory in column-major format if np == 1.

    constexpr int tile_stride = nbatch_combine + 4;
    static_assert((DV/2) % nbatch_combine == 0, "bad nbatch_combine");

    {
        // jc_cwm = jc combine write meta
        // KQ_cmr = KQ combine max rowsum
        // Use the 16 bytes of padding in each Q column to store the meta data: KQ max, KQ rowsum, KQ max scale.
#if defined(TURING_MMA_AVAILABLE)
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(threadIdx.x % 4);
        const float2 KQ_cmr = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]);
        const bool thread_should_write = threadIdx.x % 4 < cols_per_thread;
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(0);
        const float2 KQ_cmr = make_float2(KQ_max[0], KQ_rowsum[0]);
        const bool thread_should_write = threadIdx.x / 16 < cols_per_thread;
#else // Volta
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_KQ::get_i(threadIdx.x & 2);
        const float2 KQ_cmr = make_float2(KQ_max[(threadIdx.x & 2) / 2], KQ_rowsum[(threadIdx.x & 2) / 2]);
        const bool thread_should_write = T_C_KQ::J == 8 || T_C_KQ::get_j(threadIdx.x & 2) < 8;
#endif // defined(TURING_MMA_AVAILABLE)

        if (thread_should_write) {
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

    }

    if (np > 1 && threadIdx.y % np == 0) {
        // Combine the meta data for parallel warps via shared memory.
        // Warps with threadIdx.y % np != 0 must NOT return early.
        // All threads must return simultaneously to avoid race conditions with work on the next tile.

        constexpr int nmeta = np*cols_per_warp >= warp_size ? np*cols_per_warp/warp_size : 1;

        const int jc_meta = threadIdx.y*cols_per_warp + (np*cols_per_warp < warp_size ? threadIdx.x % (np*cols_per_warp) : threadIdx.x);
        float2 * const meta_ptr = ((float2 *) tile_Q) + jc_meta*(tile_stride/2) + nbatch_combine/2;
        float2 meta[nmeta];
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            meta[imeta] = meta_ptr[imeta * warp_size * tile_stride/2];
        }

        float KQ_cmn = meta[0].x; // KQ combine max new, max between all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_cmn = fmaxf(KQ_cmn, meta[imeta].x);
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < warp_size) {
                KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, warp_size));
            }
        }

        float KQ_cms[nmeta]; // KQ combine max scale per warp.
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            KQ_cms[imeta] = exp2f(meta[imeta].x - KQ_cmn);
        }

        float KQ_crs = KQ_cms[0]*meta[0].y; // KQ combine rowsum, scaled sum of all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_crs += KQ_cms[imeta]*meta[imeta].y;
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < warp_size) {
                KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, warp_size);
            }
        }

        __syncthreads();

        // Write back combined meta data:
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            if (np*cols_per_warp >= warp_size || threadIdx.x < np*cols_per_warp) {
                // Combined KQ max scale + rowsum.
                meta_ptr[imeta * warp_size * tile_stride/2] = make_float2(KQ_cms[imeta], KQ_crs);
            }
        }

        // Combined KQ max + rowsum for the partition merge (mirrors
        // fattn's dstk_fixup_meta write and paged_attn_decode :601).
        static_assert(cols_per_warp <= warp_size);
        if (write_partials && (cols_per_warp == warp_size || threadIdx.x < cols_per_warp)) {
            const int jc = (threadIdx.y / np)*cols_per_warp + threadIdx.x;
            const int row  = group_row0 + jc / ncols2;
            const int c    = jc % ncols2;
            const int head = kv_head*gqa_ratio + c;
            // c >= gqa_ratio columns are zero-Q padding; writing their stats
            // would race with the next kv_head block's real meta for the
            // aliased head slots ((kv_head+1)*gqa_ratio + {0,1}).
            if (row < n_rows && c < gqa_ratio && head < n_head) {
                const int64_t output_row = (int64_t) head * n_rows + row;
                partial_meta[output_row * n_partitions + partition] =
                    make_float2(KQ_cmn, KQ_crs);
            }
        }

    } else if (np > 1) {
        // Warps with threadIdx.y % np == 0 execute a __syncthreads() in the if branch.
        // Therefore, all other warps also need to execute a __syncthreads().
        // Otherwise the points at which warps synchronize with each other would become misaligned.
        __syncthreads();
    }

#pragma unroll
    for (int k00 = 0; k00 < DV/2; k00 += nbatch_combine) {
        {
            const int j0 = threadIdx.y*cols_per_warp;
#pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    const int j = j0 + T_C_VKQ::get_i(l);
                    const int k = k1 + T_C_VKQ::get_j(l);

                    tile_Q[j*tile_stride + k] = VKQ_C[(k00 + k1)/T_C_VKQ::J].x[l];
                }
            }
        }

        __syncthreads();

        if (np == 1 || threadIdx.y % np == 0) {
            // The combine strips hold one slice per warp: destination
            // column jc_dst maps to strip (jc_dst/cols_per_warp)*(np*cols_per_warp)
            // + jc_dst % cols_per_warp, with the np partner slices at
            // ip*cols_per_warp strip offsets (fattn-mma-f16.cuh write-back).
#pragma unroll
            for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
                const int k0_start  = stride_k == warp_size ? 0 : nbatch_combine - nbatch_combine % (2*stride_k);
                const int k0_stop   =                             nbatch_combine - nbatch_combine % (1*stride_k);
                const int stride_jc = warp_size / stride_k;

                if (k0_start == k0_stop) {
                    continue;
                }

#pragma unroll
                for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += (nwarps/np)*stride_jc) {
                    const int jc_dst = jc0_dst + (threadIdx.y/np)*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                    if (jc0_dst + (nwarps/np)*stride_jc > ncols && jc_dst >= ncols) {
                        break;
                    }

                    const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;

                    const int j = jc_dst / ncols2;
                    const int c = jc_dst % ncols2;

                    // Dead rows and out-of-range head slots were pinned by the
                    // sentinel path before the pass loop; skip them here.
                    if (j >= group_rows || c >= gqa_ratio || row_seq[j] < 0 || row_extent[j] < 0) {
                        continue;
                    }
                    const int row  = group_row0 + j;
                    const int head = kv_head*gqa_ratio + c;

                    const float * meta_j = (const float *) tile_Q + jc_tile_K*tile_stride + nbatch_combine;
                    const float qk_sum = meta_j[1];
                    float * o_row = (float *) (dst + (int64_t) row * dst_nb1 + (int64_t) head * dst_nb2);
                    half * pa_row = nullptr;
                    if (write_partials) {
                        const int64_t output_row = (int64_t) head * n_rows + row;
                        pa_row = partial_acc + output_row * (int64_t) n_partitions * 256 + (int64_t) partition * 256;
                    }
                    const float inv_sum = qk_sum > 0.0f ? 1.0f/qk_sum : 0.0f;
#pragma unroll
                    for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                        const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                        float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                        for (int ip = 0; ip < np; ++ip) {
                            const float KQ_crs_ip = np == 1 ? 1.0f : meta_j[ip*cols_per_warp * tile_stride + 0];
                            const float2 dstk_val_add = __half22float2(tile_Q[(jc_tile_K + ip*cols_per_warp) * tile_stride + k]);
                            dstk_val.x += dstk_val_add.x*KQ_crs_ip;
                            dstk_val.y += dstk_val_add.y*KQ_crs_ip;
                        }
                        // paged_attn_combine expects per-partition partials
                        // normalized by the partition's own qk_sum (decode
                        // kernel convention: numerator and denominator both
                        // carry the qk_sum weight).
                        dstk_val.x *= inv_sum;
                        dstk_val.y *= inv_sum;
                        const int dim0 = 2*(k00 + k);
                        if (write_partials) {
                            pa_row[dim0]     = __float2half(dstk_val.x);
                            pa_row[dim0 + 1] = __float2half(dstk_val.y);
                        } else {
                            o_row[dim0]     = dstk_val.x;
                            o_row[dim0 + 1] = dstk_val.y;
                        }
                    }
                }
            }
        }
        if (np > 1 || DV/2 > nbatch_combine) {
            __syncthreads();
        }
    }
}
// Launcher for the stage-1 WMMA paged kernel. Env-gated (DFLASH27B_PAGED_WMMA,
// default off); falls back to the V_DOT2 kernel when disabled or ineligible.
static bool try_launch_paged_attn_wmma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool enabled = []() {
        const char * e = getenv("DFLASH27B_PAGED_WMMA");
        return e && atoi(e) != 0;
    }();
    if (!enabled) {
        return false;
    }
    // The iter's WMMA fragment layout and mma operand order are RDNA4-specific
    // (RDNA4 is only defined in the device pass, so gate on the live device's
    // capability here): on other tensor-core targets the kernel would compile
    // and silently miscompute.
    if (!GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ctx.device].cc)) {
        return false;
    }
    const ggml_tensor * q  = dst->src[0];
    const ggml_tensor * k  = dst->src[1];
    const ggml_tensor * v  = dst->src[2];
    const ggml_tensor * block_table = dst->src[3];
    const ggml_tensor * kv_seq_lens = dst->src[4];
    const ggml_tensor * active_slot_ids = dst->src[5];
    const ggml_tensor * query_positions = dst->src[6];

    // Stage-1 gates: RDNA4 only, non-tree, F16/Q8_0/Q4_0 KV, GQA ratio that
    // fits the 8 head slots (ncols2), 16-token blocks.
    if (dst->src[7] != nullptr || dst->src[8] != nullptr) {
        return false;
    }
    if ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0 && k->type != GGML_TYPE_Q4_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0 && v->type != GGML_TYPE_Q4_0)) {
        return false;
    }
    const int32_t n_head    = (int32_t) q->ne[2];
    const int32_t n_head_kv = (int32_t) k->ne[2];
    if (n_head % n_head_kv != 0 || n_head / n_head_kv > 8) {
        return false;
    }
    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));
    int32_t block_size, max_kv_seq_len;
    memcpy(&block_size, (const char *) dst->op_params + sizeof(float), sizeof(int32_t));
    memcpy(&max_kv_seq_len, (const char *) dst->op_params + sizeof(float) + sizeof(int32_t), sizeof(int32_t));
    if (block_size != 16) {
        return false;
    }

    const int32_t n_rows = (int32_t) q->ne[1];
    if (active_slot_ids == nullptr && (int32_t) block_table->ne[1] != n_rows) {
        return false;
    }
    const int32_t n_logical_blocks = paged_attn_ceil_div(max_kv_seq_len, block_size);
    const int32_t partitionable_blocks = (int32_t) block_table->ne[0];

    static const int force_partitions = []() {
        const char * e = getenv("GGML_CUDA_PAGED_ATTN_FORCE_PARTITIONS");
        return e ? atoi(e) : 0;
    }();
    const int32_t work_groups = n_rows * n_head_kv;
    const int32_t nsm = ggml_cuda_info().devices[ctx.device].nsm;
    int32_t min_partitions = (nsm * 4 + work_groups - 1) / work_groups;
    if (min_partitions < 1) {
        min_partitions = 1;
    }
    // Small batches need more context partitions each to expose enough work;
    // large batches already fill the device, where extra partitions mostly
    // repeat the per-partition fixed costs (staging, barriers, partials,
    // combine). Mirror the decode launcher's inversely batch-scaled cap: the
    // WMMA kernel's per-partition overhead is higher still, so forcing 32
    // partitions onto prefill-shaped calls measurably loses to the decode
    // kernel at short contexts.
    int32_t partition_limit = PAGED_ATTN_MAX_PARTITIONS / n_rows;
    if (partition_limit < 32) {
        partition_limit = 32;
    }
    if (min_partitions > partition_limit) {
        min_partitions = partition_limit;
    }
    if (min_partitions > partitionable_blocks) {
        min_partitions = partitionable_blocks;
    }
    int32_t n_partitions =
        paged_attn_partitions(n_logical_blocks, min_partitions, PAGED_ATTN_MAX_PARTITIONS);
    // Test/debug override, applied after the occupancy floor like the decode
    // launcher: the floor above would otherwise silently raise any forced
    // value <= 32 back to 32 and defeat partition bisection experiments.
    if (force_partitions >= 1 &&
        force_partitions <= PAGED_ATTN_MAX_PARTITIONS &&
        force_partitions <= partitionable_blocks) {
        min_partitions = force_partitions;
        n_partitions = force_partitions;
    }

    const int32_t rows_per_block = 4;
    const dim3 grid(n_head_kv, (n_rows + rows_per_block - 1) / rows_per_block, n_partitions);
    const dim3 block(32, 8);
    // Tiles alias sequentially at smem offset 0: Q staging (32x132),
    // K/V staging (64x68), combine strips (128x36 half2s). Allocating the
    // phase max (128 strips x 36 half2s = 18 KiB) fits two blocks per CU
    // (64 KiB budget on gfx1201); mask is computed inline, no smem.
    const size_t smem = (size_t) 128*36 * sizeof(half2);

    const int64_t output_rows = (int64_t) n_rows * n_head;
    ggml_cuda_pool_alloc<half>   acc_scratch(ctx.pool());
    ggml_cuda_pool_alloc<float2> meta_scratch(ctx.pool());
    half   * partial_acc  = nullptr;
    float2 * partial_meta = nullptr;
    if (n_partitions > 1) {
        const size_t partial_rows = (size_t) output_rows * n_partitions;
        partial_acc  = acc_scratch.alloc(partial_rows * 256);
        partial_meta = meta_scratch.alloc(partial_rows);
    }

    const int write_partials = n_partitions > 1 ? 1 : 0;
    // Dispatch on the KV types; the kernel is templated on both. Three KV
    // types (F16, Q8_0, Q4_0) make nine instantiations, so the argument list
    // lives in one place.
#define PAGED_ATTN_WMMA_LAUNCH(KT, VT)                                              \
    paged_attn_wmma<KT, VT><<<grid, block, smem, ctx.stream()>>>(                   \
        (const char *) q->data, (const char *) k->data, (const char *) v->data,     \
        (const char *) block_table->data, (const char *) kv_seq_lens->data,         \
        active_slot_ids ? (const char *) active_slot_ids->data : nullptr,           \
        query_positions ? (const char *) query_positions->data : nullptr,           \
        (char *) dst->data, partial_acc, partial_meta,                              \
        q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2],                 \
        block_table->nb[0], block_table->nb[1], kv_seq_lens->nb[0],                 \
        active_slot_ids ? active_slot_ids->nb[0] : 0,                               \
        query_positions ? query_positions->nb[0] : 0,                               \
        dst->nb[1], dst->nb[2],                                                     \
        (int32_t) block_table->ne[1], n_head, n_head_kv,                            \
        (int32_t) k->ne[1], (int32_t) block_table->ne[0], block_size,               \
        n_rows, min_partitions, n_partitions, write_partials, scale)

    if (k->type == GGML_TYPE_F16) {
        if (v->type == GGML_TYPE_F16)       { PAGED_ATTN_WMMA_LAUNCH(GGML_TYPE_F16,  GGML_TYPE_F16);  }
        else if (v->type == GGML_TYPE_Q8_0) { PAGED_ATTN_WMMA_LAUNCH(GGML_TYPE_F16,  GGML_TYPE_Q8_0); }
        else                                { PAGED_ATTN_WMMA_LAUNCH(GGML_TYPE_F16,  GGML_TYPE_Q4_0); }
    } else if (k->type == GGML_TYPE_Q8_0) {
        if (v->type == GGML_TYPE_F16)       { PAGED_ATTN_WMMA_LAUNCH(GGML_TYPE_Q8_0, GGML_TYPE_F16);  }
        else if (v->type == GGML_TYPE_Q8_0) { PAGED_ATTN_WMMA_LAUNCH(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0); }
        else                                { PAGED_ATTN_WMMA_LAUNCH(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0); }
    } else {
        if (v->type == GGML_TYPE_F16)       { PAGED_ATTN_WMMA_LAUNCH(GGML_TYPE_Q4_0, GGML_TYPE_F16);  }
        else if (v->type == GGML_TYPE_Q8_0) { PAGED_ATTN_WMMA_LAUNCH(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0); }
        else                                { PAGED_ATTN_WMMA_LAUNCH(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0); }
    }
#undef PAGED_ATTN_WMMA_LAUNCH

    if (n_partitions > 1) {
        const dim3 combine_grid(
            (unsigned int) q->ne[2],
            (unsigned int) q->ne[1],
            1);
        paged_attn_combine<256>
            <<<combine_grid, dim3(256, 1, 1), 0, ctx.stream()>>>(
            partial_acc,
            partial_meta,
            (char *) dst->data,
            dst->nb[1],
            dst->nb[2],
            n_partitions);
    }
    // Surface launch-configuration failures (grid/smem limits) instead of
    // leaving a partially written dst to be read as if it were valid.
    CUDA_CHECK(cudaGetLastError());
    ggml_backend_cuda_record_paged_attn_wmma256_launch();
    return true;
}

void ggml_cuda_paged_attn(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_paged_attn_supported(dst));
    if (try_launch_paged_attn_wmma(ctx, dst)) {
        return;
    }

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
