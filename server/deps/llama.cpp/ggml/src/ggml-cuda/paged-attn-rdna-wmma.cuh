#pragma once
// Native zero-spill RDNA3.5 (gfx1151) & RDNA4 (gfx1201) Wave32 WMMA + Warp-Split-D
// Paged Attention kernel (D=256, block_size=16, F16 & Q8_0 KV, Split-KV).
//
// - SPLIT_D=4 (64 dims/warp, ND_SUB=4): Q_reg[4] (16 VGPRs) + O[4] (32 VGPRs) ->
//   ZERO register spill on BOTH gfx1151 (Strix Halo) AND gfx1201 (R9700)!
// - SPLIT_D=2 (128 dims/warp, ND_SUB=8) on RDNA4: packs up to 48 queries/KV load
//   (BM=64) or 24 queries/4-warp block (BM=32) with 0 B spill!
// - Precomputes 64-bit ancestor bitmask (tree_vis_mask) ONCE before the KV loop ->
//   1-instruction bit test in the hot loop instead of per-candidate parent walks.
// - Supports BOTH speculative tree/chain verify (parent_ids != nullptr) AND non-tree
//   paged decode / causal prefill, reusing existing paged_attn_combine<256>.

#include "common.cuh"
#include "paged-attn-common.cuh"
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <cstring>

#if defined(GGML_USE_HIP)

namespace {

static int32_t paged_attn_partials_cap(int64_t output_rows, int32_t head_dim) {
    constexpr int64_t budget = 256ll << 20;
    const int64_t per_partition =
        output_rows * ((int64_t) head_dim * (int64_t) sizeof(half) + (int64_t) sizeof(float2));
    const int64_t cap = per_partition > 0 ? budget / per_partition : PAGED_ATTN_MAX_PARTITIONS;
    return (int32_t) std::max<int64_t>(1, std::min<int64_t>(cap, PAGED_ATTN_MAX_PARTITIONS));
}

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

using pa_f16x8_t  = __attribute__((ext_vector_type(8)))  _Float16;
using pa_f16x16_t = __attribute__((ext_vector_type(16))) _Float16;
using pa_f32x8_t  = __attribute__((ext_vector_type(8)))  float;

__device__ __forceinline__ int pa_k_swiz_chunk(const int ki, const int chunk16) {
    return chunk16 ^ (ki & 7);
}

__device__ __forceinline__ int pa_v_swiz_chunk32(const int d, const int chunk16) {
    return chunk16 ^ ((d >> 1) & 3) ^ ((d >> 3) & 3);
}

__device__ __forceinline__ uint32_t pa_pack_f16x2(const float a, const float b) {
    const __half2 h2 = __floats2half2_rn(a, b);
    return __builtin_bit_cast(uint32_t, h2);
}

__device__ __forceinline__ uint4 pa_dequant_q8_0_quad(const block_q8_0 * bp, const int sub) {
    const float d = __half2float(bp->d);
    uint64_t q64;
#ifdef __HIP_DEVICE_COMPILE__
    __builtin_memcpy(&q64, bp->qs + sub * 8, sizeof(q64));
#else
    std::memcpy(&q64, bp->qs + sub * 8, sizeof(q64));
#endif
    const uint32_t lo = (uint32_t) q64;
    const uint32_t hi = (uint32_t) (q64 >> 32);
    const float f0 = (float)(int8_t)(lo & 0xFFu),         f1 = (float)(int8_t)((lo >>  8) & 0xFFu);
    const float f2 = (float)(int8_t)((lo >> 16) & 0xFFu), f3 = (float)(int8_t)(lo >> 24);
    const float f4 = (float)(int8_t)(hi & 0xFFu),         f5 = (float)(int8_t)((hi >>  8) & 0xFFu);
    const float f6 = (float)(int8_t)((hi >> 16) & 0xFFu), f7 = (float)(int8_t)(hi >> 24);
    return make_uint4(
        pa_pack_f16x2(d * f0, d * f1),
        pa_pack_f16x2(d * f2, d * f3),
        pa_pack_f16x2(d * f4, d * f5),
        pa_pack_f16x2(d * f6, d * f7));
}

__device__ __forceinline__ void pa_store_v_4rows32(
        _Float16 * KV_lds, const int d0, const int g,
        const uint4 r0, const uint4 r1, const uint4 r2, const uint4 r3) {
    constexpr int BN = 32;
    const int ch16 = g >> 1;
    const int sub4 = (g & 1) * 4;

    auto write_pair = [&](const int u_base, const uint32_t w0, const uint32_t w1, const uint32_t w2, const uint32_t w3) {
        const uint2 lo = make_uint2(
            (w0 & 0xFFFFu) | (w1 << 16),
            (w2 & 0xFFFFu) | (w3 << 16));
        const uint2 hi = make_uint2(
            (w0 >> 16) | (w1 & 0xFFFF0000u),
            (w2 >> 16) | (w3 & 0xFFFF0000u));
        const int d_lo = d0 + u_base;
        const int d_hi = d0 + u_base + 1;
        *reinterpret_cast<uint2 *>(KV_lds + d_lo * BN + pa_v_swiz_chunk32(d_lo, ch16) * 8 + sub4) = lo;
        *reinterpret_cast<uint2 *>(KV_lds + d_hi * BN + pa_v_swiz_chunk32(d_hi, ch16) * 8 + sub4) = hi;
    };

    write_pair(0, r0.x, r1.x, r2.x, r3.x);
    write_pair(2, r0.y, r1.y, r2.y, r3.y);
    write_pair(4, r0.z, r1.z, r2.z, r3.z);
    write_pair(6, r0.w, r1.w, r2.w, r3.w);
}

template <bool KV_Q8_0, int SPLIT_D, int NQ_GROUPS, bool IS_TREE>
__launch_bounds__(NQ_GROUPS * SPLIT_D * 32, 1)
__global__ void paged_attn_rdna_wmma_kernel(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const char * __restrict__ block_table,
        const char * __restrict__ kv_seq_lens,
        const char * __restrict__ active_slot_ids,
        const char * __restrict__ query_positions,
        const char * __restrict__ parent_ids,
        const char * __restrict__ tree_sizes,
        char       * __restrict__ dst,
        half       * __restrict__ partial_acc,
        float2     * __restrict__ partial_meta,
        int64_t q_nb1, int64_t q_nb2,
        int64_t k_nb1, int64_t k_nb2,
        int64_t v_nb1, int64_t v_nb2,
        int64_t bt_nb0, int64_t bt_nb1,
        int64_t ksl_nb0,
        int64_t asi_nb0, int64_t qpos_nb0,
        int64_t parent_nb0, int64_t parent_nb1, int64_t tree_size_nb0,
        int64_t dst_nb1, int64_t dst_nb2,
        int32_t n_table_seq,
        int32_t n_head,
        int32_t n_head_kv,
        int32_t pool_tokens,
        int32_t max_blocks,
        int32_t n_rows,
        int32_t rows_per_block,
        int32_t groups_per_tree,
        int32_t tree_width,
        int32_t tree_scratch_base,
        int32_t tree_scratch_stride,
        int32_t min_partitions,
        int32_t n_partitions,
        int32_t write_partials,
        float scale) {
#if defined(RDNA4) || defined(RDNA3)
    constexpr int D        = 256;
    constexpr int BN       = 32;
    constexpr int NT       = 2;
    constexpr int D_SUB    = D / SPLIT_D;   // 128 (SPLIT_D=2) or 64 (SPLIT_D=4)
    constexpr int ND_SUB   = D_SUB / 16;    // 8   (SPLIT_D=2) or 4  (SPLIT_D=4)
    constexpr int NWARPS   = NQ_GROUPS * SPLIT_D;
    constexpr int NTHREADS = NWARPS * 32;
    constexpr int BLOCK_SZ = 16;
    static_assert(NWARPS <= 8, "NWARPS must be <= 8 so S_red fits inside 16 KB KV_lds");

    __shared__ _Float16 KV_lds[BN * D];
    __shared__ int32_t  phys_lds[BN];
    __shared__ int32_t  tile_inv_lds;
    __shared__ int32_t  blk_seq_lds[32];
    __shared__ int32_t  blk_kv_lds[32];

    const int lane   = threadIdx.x;
    const int wave   = threadIdx.y;
    const int tid    = wave * 32 + lane;
    const int r      = lane & 15;
    const int hi     = lane >> 4;
    const int q_wave = (SPLIT_D == 2) ? (wave >> 1) : (wave >> 2);
    const int dh     = wave & (SPLIT_D - 1);
    const int d_base = dh * D_SUB;

    const int kv_head   = blockIdx.x;
    const int partition = blockIdx.z;
    const int gqa_ratio = n_head / n_head_kv;
    const int32_t n_physical_blocks = pool_tokens / BLOCK_SZ;
    const int64_t table_capacity    = (int64_t) max_blocks * BLOCK_SZ;

    int32_t tree_seq      = 0;
    int32_t group_in_tree = 0;
    int32_t row_base      = 0;
    int32_t row_limit     = 0;
    int32_t tree_size     = 0;

    if constexpr (IS_TREE) {
        tree_seq      = (int32_t) blockIdx.y / groups_per_tree;
        group_in_tree = (int32_t) blockIdx.y - tree_seq * groups_per_tree;
        row_base      = tree_seq * tree_width + group_in_tree * rows_per_block;
        const int32_t tree_end = (tree_seq + 1) * tree_width;
        row_limit     = row_base + rows_per_block < tree_end ? row_base + rows_per_block : tree_end;
        tree_size     = *(const int32_t *)(tree_sizes + (int64_t) tree_seq * tree_size_nb0);
    } else {
        row_base  = (int32_t) blockIdx.y * rows_per_block;
        row_limit = row_base + rows_per_block < n_rows ? row_base + rows_per_block : n_rows;
    }
    const int32_t blk_row_count = row_limit - row_base;

    const int qi        = q_wave * 16 + r;
    const int local_row = qi / gqa_ratio;
    const int gqa_h     = qi - local_row * gqa_ratio;
    const int my_row    = row_base + local_row;
    const int my_head   = kv_head * gqa_ratio + gqa_h;
    const bool in_output_slot = (local_row < blk_row_count) && (my_row < row_limit) && (gqa_h < gqa_ratio) && (my_head < n_head);

    int32_t my_phys_seq   = -1;
    int32_t my_kv_len     = 0;
    int32_t my_query_node = -1;
    bool    valid_q       = false;
    uint64_t tree_vis_mask = 0ull;

    if (in_output_slot) {
        const int32_t ps_raw = active_slot_ids
            ? *(const int32_t *)(active_slot_ids + (int64_t) my_row * asi_nb0)
            : my_row;
        if constexpr (IS_TREE) {
            my_query_node = group_in_tree * rows_per_block + local_row;
            if (ps_raw >= 0 && ps_raw < n_table_seq &&
                tree_size >= 0 && tree_size <= tree_width &&
                my_query_node < tree_size) {
                my_phys_seq = ps_raw;
                const int32_t ksl = *(const int32_t *)(kv_seq_lens + (int64_t) my_phys_seq * ksl_nb0);
                my_kv_len = ksl <= 0 ? 0 : (ksl < table_capacity ? ksl : (int32_t) table_capacity);
                valid_q = true;
                // Precompute 64-bit ancestor bitmask ONCE before the KV loop!
                if (tree_size <= 64) {
                    int32_t cur = my_query_node;
                    uint64_t m = 0ull;
                    for (int32_t depth = 0; depth < tree_size; ++depth) {
                        if (cur < 0 || cur >= tree_size) { m = 0ull; break; }
                        m |= (1ull << cur);
                        const int32_t par = *(const int32_t *)(
                            parent_ids + (int64_t) cur * parent_nb0 +
                            (int64_t) tree_seq * parent_nb1);
                        if (par == -1) break;
                        if (par < 0 || par >= cur) { m = 0ull; break; }
                        cur = par;
                    }
                    tree_vis_mask = m;
                }
            }
        } else {
            const int32_t qpos = query_positions
                ? *(const int32_t *)(query_positions + (int64_t) my_row * qpos_nb0)
                : -1;
            if (ps_raw >= 0 && ps_raw < n_table_seq && (!query_positions || qpos >= 0)) {
                my_phys_seq = ps_raw;
                int32_t ksl = *(const int32_t *)(kv_seq_lens + (int64_t) my_phys_seq * ksl_nb0);
                if (query_positions && qpos < ksl) {
                    ksl = qpos + 1;
                }
                my_kv_len = ksl <= 0 ? 0 : (ksl < table_capacity ? ksl : (int32_t) table_capacity);
                valid_q = (my_kv_len > 0);
            }
        }
    }

    if constexpr (!IS_TREE) {
        if (tid < blk_row_count) {
            const int32_t rr = row_base + tid;
            const int32_t ps_raw = active_slot_ids
                ? *(const int32_t *)(active_slot_ids + (int64_t) rr * asi_nb0)
                : rr;
            int32_t ps_out = -1;
            int32_t kv_out = 0;
            const int32_t qpos = query_positions
                ? *(const int32_t *)(query_positions + (int64_t) rr * qpos_nb0)
                : -1;
            if (ps_raw >= 0 && ps_raw < n_table_seq && (!query_positions || qpos >= 0)) {
                int32_t ksl = *(const int32_t *)(kv_seq_lens + (int64_t) ps_raw * ksl_nb0);
                if (query_positions && qpos < ksl) {
                    ksl = qpos + 1;
                }
                const int32_t kl = ksl <= 0 ? 0 : (ksl < table_capacity ? ksl : (int32_t) table_capacity);
                if (kl > 0) {
                    ps_out = ps_raw;
                    kv_out = kl;
                }
            }
            blk_seq_lds[tid] = ps_out;
            blk_kv_lds[tid]  = kv_out;
        }
        __syncthreads();
    }

    const float q_scale = scale * PAGED_ATTN_LOG2E;
    pa_f16x8_t Q_reg[ND_SUB];
    if (valid_q) {
        const char * Q_row = q + (int64_t) my_row * q_nb1 + (int64_t) my_head * q_nb2 + d_base * 4;
#pragma unroll
        for (int kt = 0; kt < ND_SUB; ++kt) {
            const float4 f0 = *reinterpret_cast<const float4 *>(Q_row + (kt * 16 + hi * 8 + 0) * 4);
            const float4 f1 = *reinterpret_cast<const float4 *>(Q_row + (kt * 16 + hi * 8 + 4) * 4);
            const uint4 qu = make_uint4(
                pa_pack_f16x2(f0.x * q_scale, f0.y * q_scale),
                pa_pack_f16x2(f0.z * q_scale, f0.w * q_scale),
                pa_pack_f16x2(f1.x * q_scale, f1.y * q_scale),
                pa_pack_f16x2(f1.z * q_scale, f1.w * q_scale));
            Q_reg[kt] = __builtin_bit_cast(pa_f16x8_t, qu);
            __builtin_amdgcn_sched_barrier(0);
        }
    } else {
#pragma unroll
        for (int kt = 0; kt < ND_SUB; ++kt) {
            Q_reg[kt] = {(_Float16)0.f, (_Float16)0.f, (_Float16)0.f, (_Float16)0.f,
                         (_Float16)0.f, (_Float16)0.f, (_Float16)0.f, (_Float16)0.f};
        }
    }

    pa_f32x8_t O[ND_SUB];
#pragma unroll
    for (int dt = 0; dt < ND_SUB; ++dt) {
        O[dt] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }
    float m_max = -INFINITY;
    float l_sum = 0.0f;

    const char * K_hb = k + (int64_t) kv_head * k_nb2;
    const char * V_hb = v + (int64_t) kv_head * v_nb2;

    const int n_seq_iters = IS_TREE ? 1 : blk_row_count;

    for (int u = 0; u < n_seq_iters; ++u) {
        int32_t cur_seq    = -1;
        int32_t cur_max_kv = 0;
        int32_t cur_min_kv = INT32_MAX;
        bool    all_same_seq = true;

        if constexpr (IS_TREE) {
            if (tree_size > 0 && tree_size <= tree_width) {
                const int32_t first_node = group_in_tree * rows_per_block;
                if (first_node < tree_size) {
                    const int32_t ps = active_slot_ids
                        ? *(const int32_t *)(active_slot_ids + (int64_t) row_base * asi_nb0)
                        : row_base;
                    if (ps >= 0 && ps < n_table_seq) {
                        cur_seq = ps;
                        const int32_t ksl = *(const int32_t *)(kv_seq_lens + (int64_t) cur_seq * ksl_nb0);
                        cur_max_kv = ksl <= 0 ? 0 : (ksl < table_capacity ? ksl : (int32_t) table_capacity);
                        cur_min_kv = cur_max_kv;
                    }
                }
            }
            if (cur_seq < 0) break;
        } else {
            cur_seq = blk_seq_lds[u];
            if (cur_seq < 0) continue;
            bool seen = false;
            for (int p = 0; p < u; ++p) {
                if (blk_seq_lds[p] == cur_seq) { seen = true; break; }
            }
            if (seen) continue;

            for (int j = 0; j < blk_row_count; ++j) {
                if (blk_seq_lds[j] == cur_seq) {
                    if (blk_kv_lds[j] > cur_max_kv) cur_max_kv = blk_kv_lds[j];
                    if (blk_kv_lds[j] < cur_min_kv) cur_min_kv = blk_kv_lds[j];
                } else if (blk_seq_lds[j] >= 0) {
                    all_same_seq = false;
                }
            }
        }

        const int64_t virtual_tokens = (int64_t) cur_max_kv + (IS_TREE ? (int64_t) tree_width : 0);
        const int32_t n_logical_blocks = paged_attn_ceil_div(virtual_tokens, BLOCK_SZ);
        const int32_t active_partitions = paged_attn_partitions(n_logical_blocks, min_partitions, n_partitions);
        if (partition >= active_partitions) {
            continue;
        }

        const int32_t logical_block_begin = ((int64_t) n_logical_blocks * partition) / active_partitions;
        const int32_t logical_block_end   = ((int64_t) n_logical_blocks * (partition + 1)) / active_partitions;
        const int32_t token_begin         = logical_block_begin * BLOCK_SZ;
        const int32_t token_end_blocks    = logical_block_end * BLOCK_SZ;
        const int32_t token_end           = virtual_tokens < token_end_blocks ? (int32_t) virtual_tokens : token_end_blocks;
        if (token_begin >= token_end) {
            continue;
        }

        const bool row_for_cur_seq = valid_q && (my_phys_seq == cur_seq);

        // Pre-resolve first tile phys indices
        if (tid < 32) {
            const int32_t tok = token_begin + tid;
            int32_t phys = -1;
            if (tok < token_end) {
                if (tok < cur_max_kv) {
                    const int32_t lb = tok >> 4;
                    const int32_t pb = *(const int32_t *)(block_table + (int64_t) lb * bt_nb0 + (int64_t) cur_seq * bt_nb1);
                    if (pb >= 0 && pb < n_physical_blocks) phys = pb * BLOCK_SZ + (tok & 15);
                } else if constexpr (IS_TREE) {
                    const int32_t cand = tok - cur_max_kv;
                    if (cand >= 0 && cand < tree_size) {
                        const int64_t p64 = (int64_t) tree_scratch_base + (int64_t) cur_seq * tree_scratch_stride + cand;
                        if (p64 >= 0 && p64 < pool_tokens) phys = (int32_t) p64;
                    }
                }
            }
            phys_lds[tid] = phys;
            const int inv = (__any(phys < 0) ? 1 : 0) | (__all(phys < 0) ? 2 : 0);
            if (tid == 0) tile_inv_lds = inv;
        }
        __syncthreads();

        for (int32_t kv0 = token_begin; kv0 < token_end; kv0 += BN) {
            const int inv_flags     = tile_inv_lds;
            const bool tile_has_inv = (inv_flags & 1) != 0;
            const bool tile_all_inv = (inv_flags & 2) != 0;

            if (!tile_all_inv) {
            // 1. Load K[kv0..kv0+31, 0..255] into XOR-swizzled KV_lds[32][256]
            {
                constexpr int chunks_per_row = D / 8;
                constexpr int total_chunks   = BN * chunks_per_row;
#pragma unroll 1
                for (int idx = tid; idx < total_chunks; idx += NTHREADS) {
                    const int ki = idx >> 5;
                    const int c  = idx & 31;
                    const int32_t phys = phys_lds[ki];
                    uint4 val = make_uint4(0u, 0u, 0u, 0u);
                    if (phys >= 0) {
                        if constexpr (!KV_Q8_0) {
                            val = *reinterpret_cast<const uint4 *>(K_hb + (int64_t) phys * k_nb1 + c * 16);
                        } else {
                            const block_q8_0 * bp = reinterpret_cast<const block_q8_0 *>(K_hb + (int64_t) phys * k_nb1) + (c >> 2);
                            val = pa_dequant_q8_0_quad(bp, c & 3);
                        }
                    }
                    const int sc = pa_k_swiz_chunk(ki, c);
                    *reinterpret_cast<uint4 *>(KV_lds + ki * D + sc * 8) = val;
                }
            }
            __syncthreads();

            // 2. S = K[:, d_base..+D_SUB-1] * Q_reg^T
            pa_f32x8_t S[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                S[t] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            }

            const int c_base = d_base >> 3;
#pragma unroll
            for (int kt = 0; kt < ND_SUB; ++kt) {
#if defined(RDNA4)
                const pa_f16x8_t qb = Q_reg[kt];
                const int c_raw = c_base + kt * 2 + hi;
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    const int ki = t * 16 + r;
                    const int sc = pa_k_swiz_chunk(ki, c_raw);
                    const uint4 ku = *reinterpret_cast<const uint4 *>(KV_lds + ki * D + sc * 8);
                    const pa_f16x8_t kv = __builtin_bit_cast(pa_f16x8_t, ku);
                    S[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kv, qb, S[t]);
                }
                __builtin_amdgcn_sched_barrier(0);
#else
                const uint4 q_self = __builtin_bit_cast(uint4, Q_reg[kt]);
                const uint4 q_peer = make_uint4(
                    __shfl_xor(q_self.x, 16, 32),
                    __shfl_xor(q_self.y, 16, 32),
                    __shfl_xor(q_self.z, 16, 32),
                    __shfl_xor(q_self.w, 16, 32));
                const uint4 q0 = hi ? q_peer : q_self;
                const uint4 q1 = hi ? q_self : q_peer;
                const pa_f16x16_t qb = __builtin_bit_cast(pa_f16x16_t, (uint4[2]){q0, q1});
                const int c0_raw = c_base + kt * 2 + 0;
                const int c1_raw = c_base + kt * 2 + 1;
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    const int ki = t * 16 + r;
                    const int sc0 = pa_k_swiz_chunk(ki, c0_raw);
                    const int sc1 = pa_k_swiz_chunk(ki, c1_raw);
                    const uint4 ku0 = *reinterpret_cast<const uint4 *>(KV_lds + ki * D + sc0 * 8);
                    const uint4 ku1 = *reinterpret_cast<const uint4 *>(KV_lds + ki * D + sc1 * 8);
                    const pa_f16x16_t kv = __builtin_bit_cast(pa_f16x16_t, (uint4[2]){ku0, ku1});
                    S[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kv, qb, S[t]);
                }
                __builtin_amdgcn_sched_barrier(0);
#endif
            }

            // 3. Warp-Split-D reduction of S across SPLIT_D warps via KV_lds
            {
                __syncthreads();
                float * S_red = reinterpret_cast<float *>(KV_lds);
                const int my_base = (q_wave * SPLIT_D + dh) * (NT * 256);
#pragma unroll
                for (int t = 0; t < NT; ++t) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        S_red[my_base + (t * 8 + e) * 32 + lane] = S[t][e];
                    }
                }
                __syncthreads();
                if constexpr (SPLIT_D == 2) {
                    const int peer_base = (q_wave * 2 + (dh ^ 1)) * (NT * 256);
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            S[t][e] += S_red[peer_base + (t * 8 + e) * 32 + lane];
                        }
                    }
                } else {
                    const int grp_base = (q_wave * 4) * (NT * 256);
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            const int off = (t * 8 + e) * 32 + lane;
                            S[t][e] = S_red[grp_base + 0 * 512 + off]
                                    + S_red[grp_base + 1 * 512 + off]
                                    + S_red[grp_base + 2 * 512 + off]
                                    + S_red[grp_base + 3 * 512 + off];
                        }
                    }
                }
            }

            // 4. Masking (fast-path skips completely on full interior tiles; 1-insn bit test for tree!)
            const bool need_mask = !row_for_cur_seq || tile_has_inv || (kv0 + BN > token_end) ||
                (IS_TREE ? (kv0 + BN > cur_max_kv) : (!all_same_seq || (kv0 + BN > cur_min_kv)));
            if (need_mask) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
#if defined(RDNA4)
                        const int kv_local = t * 16 + hi * 8 + e;
#else
                        const int kv_local = t * 16 + 2 * e + hi;
#endif
                        const int32_t tok = kv0 + kv_local;
                        bool masked = (!row_for_cur_seq || tok >= token_end || phys_lds[kv_local] < 0);
                        if (!masked) {
                            if constexpr (IS_TREE) {
                                if (tok >= my_kv_len) {
                                    const int32_t cand = tok - my_kv_len;
                                    if (tree_size <= 64) {
                                        masked = (cand >= 64) || (((tree_vis_mask >> cand) & 1ull) == 0ull);
                                    } else {
                                        masked = !paged_attn_tree_visible(
                                            parent_ids, parent_nb0, parent_nb1,
                                            tree_seq, my_query_node, cand, tree_size);
                                    }
                                }
                            } else {
                                if (tok >= my_kv_len) masked = true;
                            }
                        }
                        if (masked) {
                            S[t][e] = -INFINITY;
                        }
                    }
                }
            }

            // 5. Online softmax in log2 domain
            float m_tile = m_max;
#pragma unroll
            for (int t = 0; t < NT; ++t) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    m_tile = fmaxf(m_tile, S[t][e]);
                }
            }
            m_tile = fmaxf(m_tile, __shfl_xor(m_tile, 16, 32));

            const bool valid_tile = row_for_cur_seq && isfinite(m_tile);
            const float alpha = (valid_tile && isfinite(m_max)) ? exp2f(m_max - m_tile) : 0.0f;
            if (valid_tile) {
                m_max = m_tile;
                l_sum *= alpha;
#pragma unroll
                for (int dt = 0; dt < ND_SUB; ++dt) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) O[dt][e] *= alpha;
                }
            }

            pa_f16x8_t P[NT];
            float p_sum = 0.0f;
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                float pv[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    pv[e] = (valid_tile && isfinite(S[t][e])) ? exp2f(S[t][e] - m_max) : 0.0f;
                    p_sum += pv[e];
                }
                const uint4 pu = make_uint4(
                    pa_pack_f16x2(pv[0], pv[1]),
                    pa_pack_f16x2(pv[2], pv[3]),
                    pa_pack_f16x2(pv[4], pv[5]),
                    pa_pack_f16x2(pv[6], pv[7]));
                P[t] = __builtin_bit_cast(pa_f16x8_t, pu);
            }
            p_sum += __shfl_xor(p_sum, 16, 32);
            if (valid_tile) {
                l_sum += p_sum;
            }

            __syncthreads();

            // 6. Load V[kv0..kv0+31, 0..255] TRANSPOSED into XOR-swizzled KV_lds[256][32]
            {
                constexpr int chunks_per_row   = D / 8;
                constexpr int row_groups       = BN / 4;
                constexpr int total_4row_tasks = row_groups * chunks_per_row;
#pragma unroll 1
                for (int idx = tid; idx < total_4row_tasks; idx += NTHREADS) {
                    const int g   = idx >> 5;
                    const int c   = idx & 31;
                    const int ki0 = g * 4;
                    const int d0  = c * 8;

                    const int32_t p0 = phys_lds[ki0 + 0];
                    const int32_t p1 = phys_lds[ki0 + 1];
                    const int32_t p2 = phys_lds[ki0 + 2];
                    const int32_t p3 = phys_lds[ki0 + 3];

                    uint4 r0 = make_uint4(0u, 0u, 0u, 0u), r1 = make_uint4(0u, 0u, 0u, 0u);
                    uint4 r2 = make_uint4(0u, 0u, 0u, 0u), r3 = make_uint4(0u, 0u, 0u, 0u);
                    if constexpr (!KV_Q8_0) {
                        if (p0 >= 0) r0 = *reinterpret_cast<const uint4 *>(V_hb + (int64_t) p0 * v_nb1 + c * 16);
                        if (p1 >= 0) r1 = *reinterpret_cast<const uint4 *>(V_hb + (int64_t) p1 * v_nb1 + c * 16);
                        if (p2 >= 0) r2 = *reinterpret_cast<const uint4 *>(V_hb + (int64_t) p2 * v_nb1 + c * 16);
                        if (p3 >= 0) r3 = *reinterpret_cast<const uint4 *>(V_hb + (int64_t) p3 * v_nb1 + c * 16);
                    } else {
                        const int b = c >> 2, sub = c & 3;
                        if (p0 >= 0) r0 = pa_dequant_q8_0_quad(reinterpret_cast<const block_q8_0 *>(V_hb + (int64_t) p0 * v_nb1) + b, sub);
                        if (p1 >= 0) r1 = pa_dequant_q8_0_quad(reinterpret_cast<const block_q8_0 *>(V_hb + (int64_t) p1 * v_nb1) + b, sub);
                        if (p2 >= 0) r2 = pa_dequant_q8_0_quad(reinterpret_cast<const block_q8_0 *>(V_hb + (int64_t) p2 * v_nb1) + b, sub);
                        if (p3 >= 0) r3 = pa_dequant_q8_0_quad(reinterpret_cast<const block_q8_0 *>(V_hb + (int64_t) p3 * v_nb1) + b, sub);
                    }
                    pa_store_v_4rows32(KV_lds, d0, g, r0, r1, r2, r3);
                }
            }
            __syncthreads();

            // 7. O += V^T[d_base..+D_SUB-1, :] * P
#pragma unroll
            for (int t = 0; t < NT; ++t) {
#if defined(RDNA4)
                const pa_f16x8_t pt = P[t];
#pragma unroll
                for (int dt = 0; dt < ND_SUB; ++dt) {
                    const int d = d_base + dt * 16 + r;
                    const int sc = pa_v_swiz_chunk32(d, t * 2 + hi);
                    const uint4 vu = *reinterpret_cast<const uint4 *>(KV_lds + d * BN + sc * 8);
                    const pa_f16x8_t vv = __builtin_bit_cast(pa_f16x8_t, vu);
                    O[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vv, pt, O[dt]);
                    __builtin_amdgcn_sched_barrier(0);
                }
#else
                const uint4 p_self = __builtin_bit_cast(uint4, P[t]);
                const uint4 p_peer = make_uint4(
                    __shfl_xor(p_self.x, 16, 32),
                    __shfl_xor(p_self.y, 16, 32),
                    __shfl_xor(p_self.z, 16, 32),
                    __shfl_xor(p_self.w, 16, 32));
                const uint4 lo_s = hi ? p_peer : p_self;
                const uint4 hi_s = hi ? p_self : p_peer;
                auto interleave_lo = [](uint32_t a, uint32_t b) -> uint32_t { return (a & 0xFFFFu) | (b << 16); };
                auto interleave_hi = [](uint32_t a, uint32_t b) -> uint32_t { return (a >> 16) | (b & 0xFFFF0000u); };
                const uint4 p0 = make_uint4(
                    interleave_lo(lo_s.x, hi_s.x), interleave_hi(lo_s.x, hi_s.x),
                    interleave_lo(lo_s.y, hi_s.y), interleave_hi(lo_s.y, hi_s.y));
                const uint4 p1 = make_uint4(
                    interleave_lo(lo_s.z, hi_s.z), interleave_hi(lo_s.z, hi_s.z),
                    interleave_lo(lo_s.w, hi_s.w), interleave_hi(lo_s.w, hi_s.w));
                const pa_f16x16_t pt = __builtin_bit_cast(pa_f16x16_t, (uint4[2]){p0, p1});
#pragma unroll
                for (int dt = 0; dt < ND_SUB; ++dt) {
                    const int d = d_base + dt * 16 + r;
                    const int sc0 = pa_v_swiz_chunk32(d, t * 2 + 0);
                    const int sc1 = pa_v_swiz_chunk32(d, t * 2 + 1);
                    const uint4 v0 = *reinterpret_cast<const uint4 *>(KV_lds + d * BN + sc0 * 8);
                    const uint4 v1 = *reinterpret_cast<const uint4 *>(KV_lds + d * BN + sc1 * 8);
                    const pa_f16x16_t vv = __builtin_bit_cast(pa_f16x16_t, (uint4[2]){v0, v1});
                    O[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vv, pt, O[dt]);
                    __builtin_amdgcn_sched_barrier(0);
                }
#endif
            }

            } else {
                // Ensure every warp has read tile_inv_lds before next-tile pre-resolve overwrites it.
                __syncthreads();
            }

            // Pre-resolve next tile phys indices before end-of-tile barrier
            const int32_t next_kv0 = kv0 + BN;
            if (next_kv0 < token_end && tid < 32) {
                const int32_t tok = next_kv0 + tid;
                int32_t phys = -1;
                if (tok < token_end) {
                    if (tok < cur_max_kv) {
                        const int32_t lb = tok >> 4;
                        const int32_t pb = *(const int32_t *)(block_table + (int64_t) lb * bt_nb0 + (int64_t) cur_seq * bt_nb1);
                        if (pb >= 0 && pb < n_physical_blocks) phys = pb * BLOCK_SZ + (tok & 15);
                    } else if constexpr (IS_TREE) {
                        const int32_t cand = tok - cur_max_kv;
                        if (cand >= 0 && cand < tree_size) {
                            const int64_t p64 = (int64_t) tree_scratch_base + (int64_t) cur_seq * tree_scratch_stride + cand;
                            if (p64 >= 0 && p64 < pool_tokens) phys = (int32_t) p64;
                        }
                    }
                }
                phys_lds[tid] = phys;
                const int inv = (__any(phys < 0) ? 1 : 0) | (__all(phys < 0) ? 2 : 0);
                if (tid == 0) tile_inv_lds = inv;
            }
            __syncthreads();
        }
    }

    // 8. Epilogue: write normalized partials (compatible with paged_attn_combine<256>) or direct FP32 dst
    if (in_output_slot) {
        const bool has_out = valid_q && (l_sum > 0.0f) && isfinite(m_max);
        if (write_partials) {
            const int64_t output_row = (int64_t) my_head * n_rows + my_row;
            const int64_t part_idx   = output_row * n_partitions + partition;
            if (has_out) {
                const float inv_l = 1.0f / l_sum;
                half * p_row = partial_acc + part_idx * D + d_base;
#pragma unroll
                for (int dt = 0; dt < ND_SUB; ++dt) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
#if defined(RDNA4)
                        const int d = dt * 16 + hi * 8 + e;
#else
                        const int d = dt * 16 + 2 * e + hi;
#endif
                        p_row[d] = __float2half(O[dt][e] * inv_l);
                    }
                }
                if (dh == 0 && hi == 0) {
                    partial_meta[part_idx] = make_float2(m_max, l_sum);
                }
            } else {
                if (dh == 0 && hi == 0) {
                    partial_meta[part_idx] = make_float2(-FLT_MAX, 0.0f);
                }
            }
        } else {
            float * o_row = reinterpret_cast<float *>(dst + (int64_t) my_row * dst_nb1 + (int64_t) my_head * dst_nb2) + d_base;
            const float inv_l = has_out ? (1.0f / l_sum) : 0.0f;
#pragma unroll
            for (int dt = 0; dt < ND_SUB; ++dt) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
#if defined(RDNA4)
                    const int d = dt * 16 + hi * 8 + e;
#else
                    const int d = dt * 16 + 2 * e + hi;
#endif
                    o_row[d] = has_out ? (O[dt][e] * inv_l) : 0.0f;
                }
            }
        }
    }
#endif // defined(RDNA4) || defined(RDNA3)
}

} // anonymous namespace

static bool try_launch_paged_attn_rdna_wmma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool enabled = []() {
        const char * e = getenv("LUCE_PAGED_WMMA");
        return !(e && atoi(e) == 0);
    }();
    if (!enabled || ggml_cuda_full_batch_invariant()) {
        return false;
    }

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool is_r4 = GGML_CUDA_CC_IS_RDNA4(cc);
    const bool is_r3 = GGML_CUDA_CC_IS_RDNA3(cc);
    if (!is_r4 && !is_r3) {
        return false;
    }

    const ggml_tensor * q               = dst->src[0];
    const ggml_tensor * k               = dst->src[1];
    const ggml_tensor * v               = dst->src[2];
    const ggml_tensor * block_table     = dst->src[3];
    const ggml_tensor * kv_seq_lens     = dst->src[4];
    const ggml_tensor * active_slot_ids = dst->src[5];
    const ggml_tensor * query_positions = dst->src[6];
    const ggml_tensor * parent_ids      = dst->src[7];
    const ggml_tensor * tree_sizes      = dst->src[8];

    if (q->ne[0] != 256) return false;
    if (k->type != v->type) return false;
    if (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) return false;

    const int32_t n_head    = (int32_t) q->ne[2];
    const int32_t n_head_kv = (int32_t) k->ne[2];
    if (n_head_kv <= 0 || (n_head % n_head_kv) != 0) return false;
    const int32_t gqa_ratio = n_head / n_head_kv;
    if (gqa_ratio < 1 || gqa_ratio > 8) return false;

    float scale = 0.0f;
    int32_t block_size = 0, max_kv_seq_len = 0;
    memcpy(&scale,          dst->op_params,                                       sizeof(float));
    memcpy(&block_size,     (const char *) dst->op_params + sizeof(float),        sizeof(int32_t));
    memcpy(&max_kv_seq_len, (const char *) dst->op_params + sizeof(float) + 4,    sizeof(int32_t));
    if (block_size != 16 || max_kv_seq_len <= 0) return false;

    const int32_t n_rows = (int32_t) q->ne[1];
    if (n_rows <= 0) return false;
    if (active_slot_ids == nullptr && (int32_t) block_table->ne[1] != n_rows) return false;

    const bool is_tree = (parent_ids != nullptr && tree_sizes != nullptr);
    if ((parent_ids != nullptr) != (tree_sizes != nullptr)) return false;

    // Strictly-winning default dispatch (mode==1, microbench-verified on gfx1201 + gfx1151):
    // - R9700 (gfx1201): wins on ALL tree verify (1.55x-2.77x) AND non-tree decode (1.11x-1.17x)!
    //   For non-tree prefill (n_rows > 16), only wins on single-seq; legacy paged_attn_wmma is competitive.
    // - Strix Halo (gfx1151): wins on tree verify, F16 decode, and Q8_0 prefill (n_rows > 16, 2.01x)!
    //   For Q8_0 single-row decode (n_rows <= 16), DP4A paged_attn_decode is faster -> defer!
    if (!is_tree && is_r3 && k->type == GGML_TYPE_Q8_0 && n_rows <= 16) {
        return false;
    }
    if (!is_tree && is_r4 && n_rows > 16 && n_rows < 128) {
        return false;
    }

    int32_t tree_width = 0, tree_scratch_base = 0, tree_scratch_stride = 0;
    int32_t n_tree_seqs = 0;
    if (is_tree) {
        tree_width          = ggml_get_op_params_i32(dst, 3);
        tree_scratch_base   = ggml_get_op_params_i32(dst, 4);
        tree_scratch_stride = ggml_get_op_params_i32(dst, 5);
        n_tree_seqs         = (int32_t) parent_ids->ne[1];
        if (tree_width <= 0 || n_tree_seqs <= 0) return false;
        const int32_t tree_row_offset = n_rows - n_tree_seqs * tree_width;
        if (tree_row_offset != 0) return false;
    }

    int split_d = 4;
    int nq_groups = 1;
    int32_t rows_per_blk = 1;
    int32_t groups_per_tree = 1;
    int32_t grid_y = 1;

    if (is_tree) {
        const int32_t max_r1 = 16 / gqa_ratio;
        const int32_t max_r2 = 32 / gqa_ratio;
        const int32_t max_r4 = 64 / gqa_ratio;
        if (tree_width <= max_r1) {
            split_d      = 4;
            nq_groups    = 1;
            rows_per_blk = tree_width;
        } else if (tree_width <= max_r2) {
            // On R9700, 4-warp SPLIT_D=2 beats 8-warp SPLIT_D=4 at high B (n_tree_seqs >= 6)
            split_d      = (is_r4 && n_tree_seqs >= 6) ? 2 : 4;
            nq_groups    = 2;
            rows_per_blk = tree_width;
        } else if (is_r4) {
            split_d      = 2;
            nq_groups    = 4;
            rows_per_blk = std::min(tree_width, std::min(max_r4, 16));
        } else {
            split_d      = 4;
            nq_groups    = 2;
            rows_per_blk = std::min(tree_width, max_r2);
        }
        groups_per_tree = (tree_width + rows_per_blk - 1) / rows_per_blk;
        grid_y = n_tree_seqs * groups_per_tree;
    } else {
        if (n_rows <= 16) {
            split_d      = 4;
            nq_groups    = 1;
            rows_per_blk = 1;
        } else if (is_r4) {
            split_d      = 2;
            nq_groups    = 4;
            rows_per_blk = std::min<int32_t>(std::min<int32_t>(64 / gqa_ratio, 16), n_rows);
        } else {
            split_d      = 4;
            nq_groups    = 2;
            rows_per_blk = std::min<int32_t>(32 / gqa_ratio, n_rows);
        }
        grid_y = (n_rows + rows_per_blk - 1) / rows_per_blk;
    }

    const int64_t max_virtual_tokens = (int64_t) max_kv_seq_len + (is_tree ? (int64_t) tree_width : 0);
    const int32_t n_logical_blocks   = paged_attn_ceil_div(max_virtual_tokens, block_size);
    const int32_t partitionable_blocks = std::max<int32_t>(1, n_logical_blocks);

    static const int force_partitions = []() {
        const char * e = getenv("GGML_CUDA_PAGED_ATTN_FORCE_PARTITIONS");
        return e ? atoi(e) : 0;
    }();

    const int32_t work_groups = grid_y * n_head_kv;
    const int32_t nsm = ggml_cuda_info().devices[ctx.device].nsm;
    const int32_t target_blocks = (nq_groups * split_d >= 8) ? (nsm * 2) : (nsm * 3);
    int32_t min_partitions = (target_blocks + work_groups - 1) / work_groups;
    if (min_partitions < 1) min_partitions = 1;

    int32_t partition_limit = PAGED_ATTN_MAX_PARTITIONS / std::max<int32_t>(1, grid_y);
    if (partition_limit < 16) partition_limit = 16;
    if (min_partitions > partition_limit) min_partitions = partition_limit;

    const int32_t partition_cap = paged_attn_partials_cap((int64_t) n_rows * n_head, 256);
    if (min_partitions > partition_cap) min_partitions = partition_cap;
    if (min_partitions > partitionable_blocks) min_partitions = partitionable_blocks;

    int32_t n_partitions = paged_attn_partitions(n_logical_blocks, min_partitions, partition_cap);
    if (force_partitions >= 1 &&
        force_partitions <= PAGED_ATTN_MAX_PARTITIONS &&
        force_partitions <= partitionable_blocks) {
        min_partitions = force_partitions;
        n_partitions   = force_partitions;
    }

    const dim3 grid(n_head_kv, grid_y, n_partitions);
    const dim3 block(32, nq_groups * split_d);

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
    const int32_t write_partials = (n_partitions > 1) ? 1 : 0;

    auto launch = [&](auto q8_tag, auto sd_tag, auto ng_tag, auto tree_tag) {
        constexpr bool Q8   = decltype(q8_tag)::value;
        constexpr int  SD   = decltype(sd_tag)::value;
        constexpr int  NG   = decltype(ng_tag)::value;
        constexpr bool TREE = decltype(tree_tag)::value;
        paged_attn_rdna_wmma_kernel<Q8, SD, NG, TREE><<<grid, block, 0, ctx.stream()>>>(
            (const char *) q->data, (const char *) k->data, (const char *) v->data,
            (const char *) block_table->data, (const char *) kv_seq_lens->data,
            active_slot_ids ? (const char *) active_slot_ids->data : nullptr,
            query_positions ? (const char *) query_positions->data : nullptr,
            parent_ids      ? (const char *) parent_ids->data      : nullptr,
            tree_sizes      ? (const char *) tree_sizes->data      : nullptr,
            (char *) dst->data, partial_acc, partial_meta,
            q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2],
            block_table->nb[0], block_table->nb[1], kv_seq_lens->nb[0],
            active_slot_ids ? active_slot_ids->nb[0] : 0,
            query_positions ? query_positions->nb[0] : 0,
            parent_ids      ? parent_ids->nb[0]      : 0,
            parent_ids      ? parent_ids->nb[1]      : 0,
            tree_sizes      ? tree_sizes->nb[0]      : 0,
            dst->nb[1], dst->nb[2],
            (int32_t) block_table->ne[1], n_head, n_head_kv,
            (int32_t) k->ne[1], (int32_t) block_table->ne[0],
            n_rows, rows_per_blk, groups_per_tree,
            tree_width, tree_scratch_base, tree_scratch_stride,
            min_partitions, n_partitions, write_partials, scale);
    };

    auto dispatch_cfg = [&](auto q8_tag, auto tree_tag) {
        if (split_d == 4) {
            if (nq_groups == 1) launch(q8_tag, std::integral_constant<int, 4>{}, std::integral_constant<int, 1>{}, tree_tag);
            else                launch(q8_tag, std::integral_constant<int, 4>{}, std::integral_constant<int, 2>{}, tree_tag);
        } else {
            if (nq_groups == 2) launch(q8_tag, std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{}, tree_tag);
            else                launch(q8_tag, std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{}, tree_tag);
        }
    };

    if (k->type == GGML_TYPE_Q8_0) {
        if (is_tree) dispatch_cfg(std::true_type{},  std::true_type{});
        else         dispatch_cfg(std::true_type{},  std::false_type{});
    } else {
        if (is_tree) dispatch_cfg(std::false_type{}, std::true_type{});
        else         dispatch_cfg(std::false_type{}, std::false_type{});
    }

    if (n_partitions > 1) {
        const dim3 combine_grid((unsigned int) n_head, (unsigned int) n_rows, 1);
        paged_attn_combine<256><<<combine_grid, dim3(256, 1, 1), 0, ctx.stream()>>>(
            partial_acc, partial_meta, (char *) dst->data,
            dst->nb[1], dst->nb[2], n_partitions);
    }

    CUDA_CHECK(cudaGetLastError());
    ggml_backend_cuda_record_paged_attn_wmma256_launch();
    return true;
}

#else

static bool try_launch_paged_attn_rdna_wmma(ggml_backend_cuda_context &, ggml_tensor *) {
    return false;
}

#endif // defined(GGML_USE_HIP)
