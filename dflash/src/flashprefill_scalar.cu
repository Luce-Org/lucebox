// Scalar F16 FlashPrefill kernels for NVIDIA Pascal (sm_60/61/62).
//
// Same algorithm as flashprefill_kernels.cu (bf16 WMMA, sm_80+) but:
//   - F16 (half) instead of __nv_bfloat16 — Pascal has no BF16 hardware
//   - Scalar F16×F16→F32 math instead of WMMA tensor cores
//   - Cooperative shared-memory loads instead of cp.async (Pascal has no async copy)
//   - __shfl / __shfl_down instead of __shfl_sync / __shfl_down_sync
//   - membar.gl instead of fence.acq_rel.gpu
//
// Dispatched from flashprefill.cpp when DFLASH27B_HAVE_PASCAL_FLASHPREFILL is set.
// The drafter's persistent buffers must be GGML_TYPE_F16.
//
// Guarded to compile only for sm_60-69 so Pascal-specific intrinsics
// (__shfl without _sync, etc.) don't affect sm_70+ codepaths.

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 600 && __CUDA_ARCH__ < 700)

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace dflash27b {
namespace flashprefill {

// =============================================================================
// Kernel 1: compute_mean_vector (F16, scalar)
// =============================================================================
// Each block of (D_HEAD threads, 1 K-head, 1 batch) reduces one K-block
// along the sequence dim, computing the mean per dim.

template <int BLOCK, int D_HEAD>
__global__ void compute_mean_vector_kernel_f16(
    const half * __restrict__ K,
    half       * __restrict__ mean_K,
    int batch, int seq_len, int n_kv_heads,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d)
{
    const int block_idx_n = blockIdx.x;
    const int zh = blockIdx.y;
    const int b = zh / n_kv_heads;
    const int h = zh % n_kv_heads;
    if (b >= batch) return;

    const int tid = threadIdx.x;
    const int dim = tid;
    if (dim >= D_HEAD) return;

    const half * Kp = K + (size_t)b * s_K_b + (size_t)h * s_K_h;
    half       * Mp = mean_K + (size_t)b * s_mK_b + (size_t)h * s_mK_h
                                     + (size_t)block_idx_n * s_mK_m;

    const int n_lo = block_idx_n * BLOCK;
    const int n_hi = min(n_lo + BLOCK, seq_len);
    const int count = n_hi - n_lo;
    if (count <= 0) return;

    float sum = 0.0f;
    for (int n = n_lo; n < n_hi; ++n) {
        sum += __half2float(Kp[(size_t)n * s_K_n + (size_t)dim * s_K_d]);
    }
    Mp[(size_t)dim * s_mK_d] = __float2half(sum / (float)count);
}

extern "C" void launch_compute_mean_vector_f16_pascal(
    const void * K, void * mean_K,
    int batch, int seq_len, int n_kv_heads, int head_dim, int block_size,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    cudaStream_t stream)
{
    const int n_k_blocks = (seq_len + block_size - 1) / block_size;
    dim3 grid(n_k_blocks, batch * n_kv_heads, 1);
    dim3 block(head_dim, 1, 1);
    if (head_dim == 128 && block_size == 128) {
        compute_mean_vector_kernel_f16<128, 128><<<grid, block, 0, stream>>>(
            (const half *)K, (half *)mean_K,
            batch, seq_len, n_kv_heads,
            s_K_b, s_K_n, s_K_h, s_K_d,
            s_mK_b, s_mK_m, s_mK_h, s_mK_d);
    }
}

// =============================================================================
// Kernel 2: compute_block_score (F16, scalar)
// =============================================================================
// Per (q_block, k_block), compute the attention score via Q · mean_K^T.

template <int BLOCK, int D_HEAD, int N_BLOCKS_TILE>
__global__ void compute_block_score_kernel_f16(
    const half * __restrict__ Q,
    const half * __restrict__ mean_K,
    float sm_scale,
    float * __restrict__ score,
    float * __restrict__ score_max,
    int batch, int n_q_heads, int n_k_heads,
    int q_block_idx_max,
    int k_block_idx_max,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    int s_M_b, int s_M_m, int s_M_n, int s_M_h)
{
    const int q_block_idx = blockIdx.x;
    const int zh = blockIdx.y;
    const int b = zh / n_q_heads;
    const int qh = zh % n_q_heads;
    if (b >= batch) return;

    const int kh = qh * n_k_heads / n_q_heads;
    const int tid = threadIdx.x;
    const int q_row_local = tid;
    const int q_row_global = q_block_idx * BLOCK + q_row_local;

    if (q_row_local >= BLOCK || q_row_global >= q_block_idx_max * BLOCK) return;

    const half * Qp = Q + (size_t)b * s_Q_b + (size_t)q_row_global * s_Q_n + (size_t)qh * s_Q_h;
    float q_reg[D_HEAD];
    #pragma unroll
    for (int d = 0; d < D_HEAD; ++d) {
        q_reg[d] = __half2float(Qp[(size_t)d * s_Q_d]);
    }

    extern __shared__ float smem[];

    for (int n = 0; n <= q_block_idx; ++n) {
        const half * mKp = mean_K + (size_t)b * s_mK_b + (size_t)n * s_mK_m + (size_t)kh * s_mK_h;
        float dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < D_HEAD; ++d) {
            dot += q_reg[d] * __half2float(mKp[(size_t)d * s_mK_d]);
        }
        dot *= sm_scale * 1.4426950408889634f;

        smem[tid] = dot;
        __syncthreads();
        for (int off = BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) smem[tid] = fmaxf(smem[tid], smem[tid + off]);
            __syncthreads();
        }
        float m_block = smem[0];
        __syncthreads();

        float p = exp2f(dot - m_block);
        smem[tid] = p;
        __syncthreads();
        for (int off = BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) smem[tid] += smem[tid + off];
            __syncthreads();
        }
        float p_sum = smem[0];
        __syncthreads();

        if (tid == 0) {
            score    [(size_t)b * s_S_b + (size_t)q_block_idx * s_S_m + (size_t)n * s_S_n + (size_t)qh * s_S_h] = p_sum;
            score_max[(size_t)b * s_M_b + (size_t)q_block_idx * s_M_m + (size_t)n * s_M_n + (size_t)qh * s_M_h] = m_block;
        }
    }
}

extern "C" void launch_compute_block_score_f16_pascal(
    const void * Q, const void * mean_K, float sm_scale,
    void * score, void * score_max,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int block_size,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    int s_M_b, int s_M_m, int s_M_n, int s_M_h,
    cudaStream_t stream)
{
    const int M = (seq_len + block_size - 1) / block_size;
    dim3 grid(M, batch * n_q_heads, 1);
    dim3 block(block_size, 1, 1);
    size_t smem = block_size * sizeof(float);
    if (head_dim == 128 && block_size == 128) {
        compute_block_score_kernel_f16<128, 128, 1><<<grid, block, smem, stream>>>(
            (const half *)Q, (const half *)mean_K, sm_scale,
            (float *)score, (float *)score_max,
            batch, n_q_heads, n_k_heads, M, M,
            s_Q_b, s_Q_n, s_Q_h, s_Q_d,
            s_mK_b, s_mK_m, s_mK_h, s_mK_d,
            s_S_b, s_S_m, s_S_n, s_S_h,
            s_M_b, s_M_m, s_M_n, s_M_h);
    }
}

// =============================================================================
// Kernel 4: sparse_flash_forward (F16, scalar — no tensor cores)
// =============================================================================
// FlashAttention-style online softmax over a selected set of K-blocks.
// Scalar F16×F16→F32 math, shared-memory tiled, no WMMA.

template <int Q_TILE, int K_TILE, int BLOCK, int D_HEAD>
__global__ void sparse_flash_forward_kernel_f16(
    const half * __restrict__ Q,
    const half * __restrict__ K,
    const half * __restrict__ V,
    half       * __restrict__ O,
    const int32_t * __restrict__ block_index,
    const int32_t * __restrict__ counts,
    float scale,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int M_blocks,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_V_b, int s_V_n, int s_V_h, int s_V_d,
    int s_O_b, int s_O_n, int s_O_h, int s_O_d,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h)
{
    // Scalar kernel: Q_TILE rows per CTA, each thread handles one Q row.
    // No WMMA — every QK/PV dot product is scalar F16×F16→F32.
    // Shared memory: Q tile + KV tile + P tile + row state.

    const int q_tile_idx = blockIdx.x;
    const int zh = blockIdx.y;
    const int b  = zh / n_q_heads;
    const int qh = zh % n_q_heads;
    if (b >= batch) return;
    const int kh = qh * n_k_heads / n_q_heads;
    const int q_block_idx = q_tile_idx; // 1 q_block per CTA for scalar

    const int tid = threadIdx.x;
    if (tid >= Q_TILE) return;

    // SMEM layout: Q[K_TILE][D_HEAD] + KV[K_TILE][D_HEAD] + P[K_TILE] + row_m + row_l
    extern __shared__ __align__(16) unsigned char smem_raw[];
    half    * Q_sh  = reinterpret_cast<half*>(smem_raw);
    half    * KV_sh = Q_sh  + (size_t)Q_TILE * D_HEAD;
    half    * P_sh  = KV_sh + (size_t)K_TILE * D_HEAD;
    float   * row_m = reinterpret_cast<float*>(P_sh + (size_t)Q_TILE * K_TILE);
    float   * row_l = row_m + Q_TILE;
    // row_l already ends at row_l + Q_TILE, no further smem needed

    // Per-thread accumulators for output: float[Q_TILE][D_HEAD] — too big for registers.
    // Instead: use a register accumulator per (q_row, d) and store to shared at end.
    // For D_HEAD=128: each thread accumulates 128 f32 values. That's 512 bytes — too
    // many registers. Instead, tile the D dimension: process D_HEAD in chunks.
    // Simpler approach: store O accumulator to shared memory between K-block iterations.

    // Initialize row state
    row_m[tid] = -INFINITY;
    row_l[tid] = 0.0f;

    // Load Q tile [Q_TILE, D_HEAD] — 1 thread per row
    {
        const half * Qp = Q + (size_t)b * s_Q_b + (size_t)qh * s_Q_h;
        int q_global = q_tile_idx * Q_TILE + tid;
        for (int d = 0; d < D_HEAD; ++d) {
            Q_sh[tid * D_HEAD + d] = (q_global < seq_len)
                ? Qp[(size_t)q_global * s_Q_n + (size_t)d * s_Q_d]
                : __float2half(0.0f);
        }
    }
    __syncthreads();

    // Pre-scale Q in shared
    {
        const float sm_scale = scale * 1.4426950408889634f;
        for (int d = 0; d < D_HEAD; ++d) {
            float v = __half2float(Q_sh[tid * D_HEAD + d]);
            Q_sh[tid * D_HEAD + d] = __float2half(v * sm_scale);
        }
    }
    __syncthreads();

    // O accumulator: store to shared memory after each K-block
    float o_acc[D_HEAD];
    #pragma unroll
    for (int d = 0; d < D_HEAD; ++d) o_acc[d] = 0.0f;

    const int hi = counts[(size_t)b * s_cnt_b + (size_t)q_block_idx * s_idx_m + (size_t)qh * s_cnt_h];

    for (int it = 0; it < hi; ++it) {
        int blk = block_index[(size_t)b * s_idx_b + (size_t)q_block_idx * s_idx_m + (size_t)it * s_idx_n + (size_t)qh * s_idx_h];
        if (blk < 0 || blk >= M_blocks) continue;
        const int k_lo = blk * BLOCK;
        const bool is_diag = (blk == q_block_idx);

        // Load K tile [BLOCK, D_HEAD] cooperatively into KV_sh
        {
            const half * Kp = K + (size_t)b * s_K_b + (size_t)kh * s_K_h;
            for (int idx = tid; idx < BLOCK * D_HEAD; idx += Q_TILE) {
                int row = idx / D_HEAD;
                int dim = idx % D_HEAD;
                int k_global = k_lo + row;
                KV_sh[idx] = (k_global < seq_len)
                    ? Kp[(size_t)k_global * s_K_n + (size_t)dim * s_K_d]
                    : __float2half(0.0f);
            }
        }
        __syncthreads();

        // QK dot + online softmax: each thread computes its own Q row against all K rows
        float local_max = -INFINITY;
        #pragma unroll
        for (int j = 0; j < BLOCK; ++j) {
            float dot = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < D_HEAD; ++d) {
                dot += __half2float(Q_sh[tid * D_HEAD + d]) * __half2float(KV_sh[j * D_HEAD + d]);
            }
            int k_global = k_lo + j;
            int q_global = q_tile_idx * Q_TILE + tid;
            bool valid = (k_global < seq_len);
            if (is_diag) valid = valid && (k_global <= q_global);
            if (!valid) dot = -INFINITY;
            P_sh[tid * BLOCK + j] = __float2half(dot);
            if (dot > local_max) local_max = dot;
        }

        // Block reduce max (smem-based tree reduction)
        P_sh[tid * BLOCK + 0] = __float2half(local_max); // reuse first P slot
        __syncthreads();
        float block_max = local_max;
        for (int other = 0; other < Q_TILE; ++other) {
            float vm = __half2float(P_sh[other * BLOCK + 0]);
            if (vm > block_max) block_max = vm;
        }
        __syncthreads();

        // Compute P = exp2(QK - block_max), write to P_sh
        float row_sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < BLOCK; ++j) {
            float qk = __half2float(P_sh[tid * BLOCK + j]);
            float p = (qk == -INFINITY) ? 0.0f : exp2f(qk - block_max);
            P_sh[tid * BLOCK + j] = __float2half(p);
            row_sum += p;
        }

        // Load V tile cooperatively (overwrites KV_sh)
        {
            const half * Vp = V + (size_t)b * s_V_b + (size_t)kh * s_V_h;
            for (int idx = tid; idx < BLOCK * D_HEAD; idx += Q_TILE) {
                int row = idx / D_HEAD;
                int dim = idx % D_HEAD;
                int k_global = k_lo + row;
                KV_sh[idx] = (k_global < seq_len)
                    ? Vp[(size_t)k_global * s_V_n + (size_t)dim * s_V_d]
                    : __float2half(0.0f);
            }
        }
        __syncthreads();

        // Online softmax update
        float m_old = row_m[tid];
        float m_new = fmaxf(m_old, block_max);
        float alpha = (m_old == -INFINITY) ? 0.0f : exp2f(m_old - m_new);
        float l_old = row_l[tid];

        // PV accumulation: O += P @ V
        #pragma unroll
        for (int d = 0; d < D_HEAD; ++d) {
            float pv = 0.0f;
            #pragma unroll 4
            for (int j = 0; j < BLOCK; ++j) {
                pv += __half2float(P_sh[tid * BLOCK + j]) * __half2float(KV_sh[j * D_HEAD + d]);
            }
            o_acc[d] = alpha * o_acc[d] + pv;
        }

        row_m[tid] = m_new;
        row_l[tid] = alpha * l_old + row_sum;
        __syncthreads();
    }

    // Write O = o_acc / l_i
    {
        float l = row_l[tid];
        float l_rcp = (l > 0.0f) ? (1.0f / l) : 1.0f;
        int q_global = q_tile_idx * Q_TILE + tid;
        if (q_global < seq_len) {
            half * Op = O + (size_t)b * s_O_b + (size_t)q_global * s_O_n + (size_t)qh * s_O_h;
            for (int d = 0; d < D_HEAD; ++d) {
                Op[(size_t)d * s_O_d] = __float2half(o_acc[d] * l_rcp);
            }
        }
    }
}

extern "C" void launch_sparse_flash_forward_f16_pascal(
    const void * Q, const void * K, const void * V, void * O,
    const int32_t * block_index, const int32_t * counts,
    float scale,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int q_tile, int block_size,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_V_b, int s_V_n, int s_V_h, int s_V_d,
    int s_O_b, int s_O_n, int s_O_h, int s_O_d,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h,
    cudaStream_t stream)
{
    const int M = (seq_len + block_size - 1) / block_size;
    const int q_tiles = (seq_len + q_tile - 1) / q_tile;
    dim3 grid(q_tiles, batch * n_q_heads, 1);
    dim3 block(q_tile, 1, 1);
    if (q_tile == 64 && block_size == 128 && head_dim == 128) {
        constexpr int Q_TILE = 64, K_TILE = 128, BLOCK = 128, D_HEAD = 128;
        size_t smem_bytes = sizeof(half) * (Q_TILE * D_HEAD)     // Q tile
                           + sizeof(half) * (K_TILE * D_HEAD)     // KV tile
                           + sizeof(half) * (Q_TILE * K_TILE)     // P tile
                           + sizeof(float) * (2 * Q_TILE);        // row_m + row_l
        sparse_flash_forward_kernel_f16<Q_TILE, K_TILE, BLOCK, D_HEAD><<<grid, block, smem_bytes, stream>>>(
            (const half *)Q, (const half *)K,
            (const half *)V, (half *)O,
            block_index, counts, scale,
            batch, n_q_heads, n_k_heads, seq_len, M,
            s_Q_b, s_Q_n, s_Q_h, s_Q_d,
            s_K_b, s_K_n, s_K_h, s_K_d,
            s_V_b, s_V_n, s_V_h, s_V_d,
            s_O_b, s_O_n, s_O_h, s_O_d,
            s_idx_b, s_idx_m, s_idx_n, s_idx_h,
            s_cnt_b, s_cnt_m, s_cnt_h);
    }
}

// =============================================================================
// Kernel 3: block_select (same algorithm, Pascal-compatible shuffles)
// =============================================================================

__global__ void block_select_kernel_pascal(
    const float * __restrict__ score,
    int B, int M, int N, int H,
    int attention_sink, int window, int last_n_full, float alpha,
    int s_b, int s_m, int s_n, int s_h,
    int idx_s_b, int idx_s_m, int idx_s_n, int idx_s_h,
    int cnt_s_b, int cnt_s_m, int cnt_s_h,
    int32_t * __restrict__ idx_out,
    int32_t * __restrict__ cnt_out)
{
    const int b = blockIdx.x;
    const int m = blockIdx.y;
    const int h = blockIdx.z;
    const int lane = threadIdx.x;

    if (b >= B || m >= M || h >= H) return;

    const float * sp = score + (size_t)b*s_b + (size_t)m*s_m + (size_t)h*s_h;
    int32_t * idxp = idx_out + (size_t)b*idx_s_b + (size_t)m*idx_s_m + (size_t)h*idx_s_h;

    const bool last_full = (m >= M - last_n_full);
    const float NEG_INF = -INFINITY;

    // Pass 1: max score
    float local_max = NEG_INF;
    for (int n_base = 0; n_base <= m; n_base += 32) {
        int n = n_base + lane;
        bool valid = (n <= m);
        float v = valid ? sp[(size_t)n * s_n] : NEG_INF;
        local_max = fmaxf(local_max, v);
    }
    // Pascal: __shfl_xor (no _sync suffix)
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    const float max_score = local_max;
    const float thresh = max_score * alpha;

    // Pass 2: predicate + compact
    int total = 0;
    for (int n_base = 0; n_base <= m; n_base += 32) {
        int n = n_base + lane;
        bool valid = (n <= m);
        bool keep = false;
        if (valid) {
            float v = sp[(size_t)n * s_n];
            keep = last_full
                || (n < attention_sink)
                || ((m - n) < window)
                || (v >= thresh);
        }
        unsigned mask = __ballot(keep);  // Pascal: __ballot (no _sync suffix)
        int rank = __popc(mask & ((1u << lane) - 1u));
        if (keep) {
            idxp[(size_t)(total + rank) * idx_s_n] = (int32_t)n;
        }
        total += __popc(mask);
    }

    for (int n = total + lane; n < N; n += 32) {
        idxp[(size_t)n * idx_s_n] = (int32_t)-1;
    }

    if (lane == 0) {
        cnt_out[(size_t)b*cnt_s_b + (size_t)m*cnt_s_m + (size_t)h*cnt_s_h] = (int32_t)total;
    }
}

extern "C" void launch_block_select_pascal(
    const float * score,
    int B, int M, int N, int H,
    int attention_sink, int window, int last_n_full, float alpha,
    int s_b, int s_m, int s_n, int s_h,
    int idx_s_b, int idx_s_m, int idx_s_n, int idx_s_h,
    int cnt_s_b, int cnt_s_m, int cnt_s_h,
    int32_t * idx_out, int32_t * cnt_out,
    cudaStream_t stream)
{
    dim3 grid(B, M, H);
    dim3 block(32, 1, 1);
    block_select_kernel_pascal<<<grid, block, 0, stream>>>(
        score, B, M, N, H,
        attention_sink, window, last_n_full, alpha,
        s_b, s_m, s_n, s_h,
        idx_s_b, idx_s_m, idx_s_n, idx_s_h,
        cnt_s_b, cnt_s_m, cnt_s_h,
        idx_out, cnt_out);
}

} // namespace flashprefill
} // namespace dflash27b

#endif // !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 600 && __CUDA_ARCH__ < 700)
