#include "gated_delta_net.cuh"
#ifndef GGML_USE_HIP
#include <cuda_fp16.h>
#endif
#include <type_traits>

// Tree-mode parent index sentinel: a node whose parent is the pre-block state
// (i.e. a "root" node in the DFS-flattened tree) uses this value in
// parent_ids[]. Any value < 0 triggers a reload from curr_state.
#define GGML_GDN_TREE_ROOT_PARENT (-1)

// Intermediate-state load/store helpers. Allow the persistent intermediate
// buffer (for dflash27b_ggml tree rollback) to live in fp16 instead of fp32,
// halving its memory footprint and letting us fit larger DDTree budgets in
// the hybrid (GatedDeltaNet) target's state cache.
static __device__ __forceinline__ float load_inter_state(const float * p, int idx) {
    return p[idx];
}
static __device__ __forceinline__ float load_inter_state(const __half * p, int idx) {
    return __half2float(p[idx]);
}
static __device__ __forceinline__ void store_inter_state(float * p, int idx, float v) {
    p[idx] = v;
}
static __device__ __forceinline__ void store_inter_state(__half * p, int idx, float v) {
    p[idx] = __float2half(v);
}

__device__ __forceinline__ float gdn_subgroup_sum_lane0(float value, int width) {
    const int lane = threadIdx.x % width;
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        const int src_lane = lane < offset ? lane + offset : lane;
        const float other = __shfl_sync(0xffffffffU, value, src_lane, width);
        if (lane < offset) {
            value += other;
        }
    }
    return value;
}

__device__ __forceinline__ float gdn_subgroup_broadcast_lane0(float value, int width) {
    return __shfl_sync(0xffffffffU, value, 0, width);
}

template <int S_v, bool KDA, bool TREE_MODE, typename InterT = float>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     const int *   parent_ids,    // TREE_MODE only; else ignored
                                     InterT *      persist_inter, // optional external buffer for per-token intermediates
                                     bool          skip_intermediate,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    const int64_t final_state_elems = S_v * S_v * H * n_seqs;
    float *       attn_data        = dst;
    float *       state            = dst + attn_score_elems;
    // intermediate_states region: one S_v*S_v*H*n_seqs state per token. Written
    // inside the token loop below (one state per `t`) to enable spec-decode
    // rollback without a replay forward pass. See ggml.c::ggml_gated_delta_net.
    //
    // dflash27b_ggml: if persist_inter != nullptr, the kernel writes the
    // intermediate states DIRECTLY to that external buffer instead of the
    // embedded region inside dst. InterT selects the storage precision (float
    // or __half). f16 halves the memory footprint — enough to fit larger
    // DDtree budgets on the 24 GB 3090.
    // When persist_inter is null, InterT MUST be float (the embedded region
    // inside dst is f32).
    InterT * inter_states = persist_inter
        ? persist_inter
        : (InterT *)(dst + attn_score_elems + final_state_elems);
    const bool write_intermediate = !skip_intermediate || TREE_MODE || persist_inter != nullptr;

    const int64_t state_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_offset;
    curr_state += state_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;
    // Per-sequence per-head base for this block's intermediates, token t=0.
    // Advance by (H * S_v * S_v) each iteration.
    InterT * inter_base = inter_states + (sequence * n_tokens * H + h_idx) * S_v * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    // TREE_MODE: pointer base for parent lookups. Each sequence has its own
    // parent_ids[n_tokens] slice. At branch points (parent_t != t - 1), we
    // reload s_shard from the intermediate-state region instead of continuing
    // the recurrence sequentially. Ports sglang's
    // fused_sigmoid_gating_recurrent.py HAS_EAGLE_TREE_CUSTOM_ATTN_MASK logic
    // to CUDA.
    const int * parent_ids_seq = nullptr;
    if constexpr (TREE_MODE) {
        parent_ids_seq = parent_ids + sequence * n_tokens;
    }

    for (int t = 0; t < n_tokens; t++) {
        // Tree branch-point reload: if this token's parent in the DFS-flattened
        // tree isn't the previous token in processing order, pull its state
        // back from the intermediate-state region. Same-thread read-after-write
        // on global memory — no __syncthreads() needed because each lane writes
        // and reads its own (col, row) slots.
        if constexpr (TREE_MODE) {
            if (t > 0) {
                const int parent_t = parent_ids_seq[t];
                if (parent_t == GGML_GDN_TREE_ROOT_PARENT) {
                    // Root-level sibling: reset to the pre-block state.
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        s_shard[r] = curr_state[i];
                    }
                } else if (parent_t != t - 1) {
                    // Branch: this token's parent is somewhere earlier in the
                    // DFS traversal. Pull that state from the intermediate
                    // region. inter_states base is per-sequence, per-head;
                    // parent_t picks the slot, col/i picks the element. The
                    // load helper converts from InterT (f32 or f16) → float.
                    const InterT * parent_base = inter_states
                        + ((sequence * n_tokens + parent_t) * H + h_idx) * S_v * S_v;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        s_shard[r] = load_inter_state(parent_base, col * S_v + i);
                    }
                }
                // parent_t == t - 1: sequential, keep s_shard in registers.
            }
        }

        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        // Write the intermediate state for token t (same transposed layout as the
        // final-state write below). Used by dflash27b_ggml spec-decode rollback.
        // Plain chain prefill does not consume it, so qwen35 can opt out to
        // avoid large transient global writes.
        if (write_intermediate) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                store_inter_state(inter_base, col * S_v + i, s_shard[r]);
            }
        }
        inter_base += S_v * S_v * H;

        attn_data += S_v * H;
    }

    // Write state back to global memory (transposed layout)
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i          = r * warp_size + lane;
        state[col * S_v + i] = s_shard[r];
    }
}

template <int S_v, int COLS, int WIDTH, int WARP_THREADS, typename InterT = float>
__global__ void __launch_bounds__(WARP_THREADS * 8, 2)
gated_delta_net_cuda_grouped_cols(const float * q,
                                  const float * k,
                                  const float * v,
                                  const float * g,
                                  const float * beta,
                                  const float * curr_state,
                                  float *       dst,
                                  InterT *      persist_inter,
                                  bool          skip_intermediate,
                                  int64_t       H,
                                  int64_t       n_tokens,
                                  int64_t       n_seqs,
                                  int64_t       sq1,
                                  int64_t       sq2,
                                  int64_t       sq3,
                                  int64_t       sv1,
                                  int64_t       sv2,
                                  int64_t       sv3,
                                  int64_t       sb1,
                                  int64_t       sb2,
                                  int64_t       sb3,
                                  const uint3   neqk1_magic,
                                  const uint3   rq3_magic,
                                  float         scale) {
    static_assert(S_v == 128, "grouped GDN kernel is specialized for S_v=128");
    static_assert(WIDTH == 16, "grouped GDN kernel expects 16-lane subgroups");
    static_assert(COLS == 4, "grouped GDN kernel expects 4 columns per subgroup");
    static_assert(WARP_THREADS == 32 || WARP_THREADS == 64, "grouped GDN kernel expects 32- or 64-lane warps");

    constexpr int subgroups_per_warp = WARP_THREADS / WIDTH;
    constexpr int rows_per_lane      = (S_v + WIDTH - 1) / WIDTH;

    const int h_idx    = blockIdx.x;
    const int sequence = blockIdx.y;
    const int subgroup = threadIdx.x / WIDTH;
    const int lane     = threadIdx.x % WIDTH;

    const int group_base = (blockIdx.z * blockDim.y + threadIdx.y) * subgroups_per_warp + subgroup;
    const int col_base   = group_base * COLS;

    if (col_base >= S_v) {
        return;
    }

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    const int64_t final_state_elems = S_v * S_v * H * n_seqs;
    float *       attn_data        = dst;
    float *       state            = dst + attn_score_elems;
    InterT *      inter_states     = persist_inter
        ? persist_inter
        : (InterT *)(dst + attn_score_elems + final_state_elems);
    const bool write_intermediate = !skip_intermediate || persist_inter != nullptr;

    const int64_t state_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_offset;
    curr_state += state_offset;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;
    InterT * inter_base = inter_states + (sequence * n_tokens * H + h_idx) * S_v * S_v;

    float state_shard[COLS][rows_per_lane];

#pragma unroll
    for (int c = 0; c < COLS; ++c) {
        const int col = col_base + c;
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int row = r * WIDTH + lane;
            state_shard[c][r] = curr_state[col * S_v + row];
        }
    }

    for (int t = 0; t < n_tokens; ++t) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;

        float g_val = 0.0f;
        float beta_val = 0.0f;
        if (threadIdx.x == 0) {
            g_val = expf(g[gb_offset]);
            beta_val = beta[gb_offset];
        }
        g_val = __shfl_sync(0xffffffffU, g_val, 0);
        beta_val = __shfl_sync(0xffffffffU, beta_val, 0);

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
        float kv_partial[COLS];

#pragma unroll
        for (int c = 0; c < COLS; ++c) {
            kv_partial[c] = 0.0f;
        }

#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int row = r * WIDTH + lane;
            const float q_val = q_t[row];
            const float k_val = k_t[row];
            q_reg[r] = q_val;
            k_reg[r] = k_val;

#pragma unroll
            for (int c = 0; c < COLS; ++c) {
                kv_partial[c] += state_shard[c][r] * k_val;
            }
        }

        float delta[COLS];
#pragma unroll
        for (int c = 0; c < COLS; ++c) {
            const float kv_col = gdn_subgroup_sum_lane0(kv_partial[c], WIDTH);
            float delta_val = 0.0f;
            if (lane == 0) {
                delta_val = (v_t[col_base + c] - g_val * kv_col) * beta_val;
            }
            delta[c] = gdn_subgroup_broadcast_lane0(delta_val, WIDTH);
        }

        float attn_partial[COLS];
#pragma unroll
        for (int c = 0; c < COLS; ++c) {
            attn_partial[c] = 0.0f;
        }

#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
#pragma unroll
            for (int c = 0; c < COLS; ++c) {
                const float new_state = fmaf(k_reg[r], delta[c], g_val * state_shard[c][r]);
                state_shard[c][r] = new_state;
                attn_partial[c] += new_state * q_reg[r];
            }
        }

#pragma unroll
        for (int c = 0; c < COLS; ++c) {
            attn_partial[c] = gdn_subgroup_sum_lane0(attn_partial[c], WIDTH);
        }

        if (lane == 0) {
#pragma unroll
            for (int c = 0; c < COLS; ++c) {
                attn_data[col_base + c] = attn_partial[c] * scale;
            }
        }

        if (write_intermediate) {
#pragma unroll
            for (int c = 0; c < COLS; ++c) {
                const int col = col_base + c;
#pragma unroll
                for (int r = 0; r < rows_per_lane; ++r) {
                    const int row = r * WIDTH + lane;
                    store_inter_state(inter_base, col * S_v + row, state_shard[c][r]);
                }
            }
        }

        attn_data += S_v * H;
        inter_base += S_v * S_v * H;
    }

#pragma unroll
    for (int c = 0; c < COLS; ++c) {
        const int col = col_base + c;
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int row = r * WIDTH + lane;
            state[col * S_v + row] = state_shard[c][r];
        }
    }
}

template <bool KDA, bool TREE_MODE, typename InterT = float>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d,
        const int * parent_ids_d,
        InterT * persist_inter_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        bool skip_intermediate,
        float scale, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    switch (S_v) {
        case 16:
            gated_delta_net_cuda<16, KDA, TREE_MODE, InterT><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, parent_ids_d, persist_inter_d, skip_intermediate, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 32:
            gated_delta_net_cuda<32, KDA, TREE_MODE, InterT><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, parent_ids_d, persist_inter_d, skip_intermediate, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 64: {
            gated_delta_net_cuda<64, KDA, TREE_MODE, InterT><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, parent_ids_d, persist_inter_d, skip_intermediate, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        }
        case 128: {
            if constexpr (!KDA && !TREE_MODE) {
                if ((GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE) ||
                    GGML_CUDA_CC_IS_AMD(cc)) {
                    constexpr int cols = 4;
                    constexpr int width = 16;
                    constexpr int column_groups_per_block = 8;
                    const int groups = 128 / cols;
                    if (warp_size == 32) {
                        constexpr int groups_per_warp = 32 / width;
                        dim3 grouped_grid_dims(H, n_seqs, (groups + column_groups_per_block * groups_per_warp - 1) / (column_groups_per_block * groups_per_warp));
                        dim3 grouped_block_dims(32, column_groups_per_block, 1);
                        gated_delta_net_cuda_grouped_cols<128, cols, width, 32, InterT><<<grouped_grid_dims, grouped_block_dims, 0, stream>>>(
                            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, persist_inter_d, skip_intermediate, H,
                            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                    } else if (warp_size == 64) {
                        constexpr int groups_per_warp = 64 / width;
                        dim3 grouped_grid_dims(H, n_seqs, (groups + column_groups_per_block * groups_per_warp - 1) / (column_groups_per_block * groups_per_warp));
                        dim3 grouped_block_dims(64, column_groups_per_block, 1);
                        gated_delta_net_cuda_grouped_cols<128, cols, width, 64, InterT><<<grouped_grid_dims, grouped_block_dims, 0, stream>>>(
                            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, persist_inter_d, skip_intermediate, H,
                            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                    } else {
                        gated_delta_net_cuda<128, KDA, TREE_MODE, InterT><<<grid_dims, block_dims, 0, stream>>>(
                            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, parent_ids_d, persist_inter_d, skip_intermediate, H,
                            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                    }
                } else {
                    gated_delta_net_cuda<128, KDA, TREE_MODE, InterT><<<grid_dims, block_dims, 0, stream>>>(
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, parent_ids_d, persist_inter_d, skip_intermediate, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                }
            } else {
                gated_delta_net_cuda<128, KDA, TREE_MODE, InterT><<<grid_dims, block_dims, 0, stream>>>(
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, parent_ids_d, persist_inter_d, skip_intermediate, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            }
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];
    // Optional 7th source = parent_ids[n_seqs, n_tokens] int32, enabling
    // tree-mode recurrence (dflash27b_ggml extension). nullptr means chain mode.
    ggml_tensor * src_parent = dst->src[6];
    // Optional 8th source = persistent external intermediate-state buffer
    // (dflash27b_ggml extension). When non-null, the kernel writes per-token
    // intermediate states directly to persist_inter->data instead of the
    // embedded region inside dst, saving a downstream ggml_cpy.
    ggml_tensor * src_persist_inter = dst->src[7];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;
    const int *   parent_ids_d = src_parent
        ? (const int *) src_parent->data
        : nullptr;
    void *        persist_inter_d = src_persist_inter
        ? src_persist_inter->data
        : nullptr;
    const bool    persist_is_f16 =
        src_persist_inter && src_persist_inter->type == GGML_TYPE_F16;
    if (src_persist_inter) {
        GGML_ASSERT(src_persist_inter->type == GGML_TYPE_F32 ||
                    src_persist_inter->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(src_persist_inter));
    }

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));
    if (src_parent) {
        GGML_ASSERT(src_parent->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(src_parent));
        GGML_ASSERT(ggml_nelements(src_parent) == n_tokens * n_seqs);
    }

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    const bool tree_mode = (parent_ids_d != nullptr);
    const bool skip_intermediate = ggml_get_op_params_i32(dst, 0) != 0;

    // Macro to expand the 4 (KDA × TREE_MODE) cases for a given InterT.
    // The persist_is_f16 branch picks between __half and float instantiations.
    #define GDN_LAUNCH(INTER_T)                                                                 \
        do {                                                                                    \
            INTER_T * persist_typed = (INTER_T *)persist_inter_d;                               \
            if (kda) {                                                                          \
                if (tree_mode) {                                                                \
                    launch_gated_delta_net<true, true, INTER_T>(                                \
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, parent_ids_d, persist_typed,       \
                        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                 \
                        sb1, sb2, sb3, neqk1, rq3, skip_intermediate, scale, stream);            \
                } else {                                                                        \
                    launch_gated_delta_net<true, false, INTER_T>(                               \
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, nullptr, persist_typed,            \
                        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                 \
                        sb1, sb2, sb3, neqk1, rq3, skip_intermediate, scale, stream);            \
                }                                                                               \
            } else {                                                                            \
                if (tree_mode) {                                                                \
                    launch_gated_delta_net<false, true, INTER_T>(                               \
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, parent_ids_d, persist_typed,       \
                        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                 \
                        sb1, sb2, sb3, neqk1, rq3, skip_intermediate, scale, stream);            \
                } else {                                                                        \
                    launch_gated_delta_net<false, false, INTER_T>(                               \
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, nullptr, persist_typed,            \
                        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                 \
                        sb1, sb2, sb3, neqk1, rq3, skip_intermediate, scale, stream);            \
                }                                                                               \
            }                                                                                   \
        } while (0)

    if (persist_is_f16) {
        GDN_LAUNCH(__half);
    } else {
        GDN_LAUNCH(float);
    }
    #undef GDN_LAUNCH
}
