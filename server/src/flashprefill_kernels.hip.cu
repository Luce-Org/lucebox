// ROCm/HIP port of flashprefill_kernels.cu — all 4 kernels.
//
// Porting changes:
//   - cp.async16/commit/wait_all  → direct uint4 load + __syncthreads()
//   - nvcuda::wmma::              → rocwmma:: (same MxNxK=16x16x16 tile)
//   - __shfl_xor_sync(0xffff,v,o) → __shfl_xor(v, o)
//   - __shfl_sync(0xffff, v, l)   → __shfl(v, l)
//   - __ballot_sync(0xffff, p)    → __ballot(p)
//   - cudaStream_t                → hipStream_t
//   - cudaFuncSetAttribute        → hipFuncSetAttribute
//   - __nv_bfloat16               → hip_bfloat16
//   - frag.x[i]                   → frag[i]   (rocWMMA uses operator[])
//
// Accumulator fragment layout change (kernel 4 mask/softmax/rescale):
//   NVIDIA m16n16k16 float32 (warp32):  lane t, elem i → row = 2*(t/4)+(i>=4), col = 2*(t%4)+(i&1)+((i>>1&1)*8)
//   AMD RDNA3 Wave32 v_wmma_f32_16x16x16_bf16: lane t, elem i → row = t%16, col = (t/16)*8 + i
//   Consequence: NVIDIA needs alpha0/alpha1 (two rows per lane); AMD needs one alpha (single row per lane).
//   The causal mask, rowmax, rowsum, alpha-rescale sections are rewritten accordingly.

#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <rocwmma/rocwmma.hpp>

namespace dflash::common {
namespace flashprefill {

// ---- Kernel 1: compute_mean_vector ----
// Each block computes mean K over one K-block. One thread per D dimension.

template <int BLOCK, int D_HEAD>
__global__ void compute_mean_vector_kernel_bf16(
    const hip_bfloat16 * __restrict__ K,
    hip_bfloat16       * __restrict__ mean_K,
    int batch, int seq_len, int n_kv_heads,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d)
{
    const int block_idx_n = blockIdx.x;
    const int zh = blockIdx.y;
    const int b  = zh / n_kv_heads;
    const int h  = zh % n_kv_heads;
    if (b >= batch) return;

    const int dim = threadIdx.x;
    if (dim >= D_HEAD) return;

    const hip_bfloat16 * Kp = K + (size_t)b * s_K_b + (size_t)h * s_K_h;
    hip_bfloat16       * Mp = mean_K + (size_t)b * s_mK_b + (size_t)h * s_mK_h
                                     + (size_t)block_idx_n * s_mK_m;

    const int n_lo  = block_idx_n * BLOCK;
    const int n_hi  = min(n_lo + BLOCK, seq_len);
    const int count = n_hi - n_lo;
    if (count <= 0) return;

    float sum = 0.0f;
    for (int n = n_lo; n < n_hi; ++n)
        sum += static_cast<float>(Kp[(size_t)n * s_K_n + (size_t)dim * s_K_d]);
    Mp[(size_t)dim * s_mK_d] = hip_bfloat16(sum / (float)count);
}

extern "C" int launch_compute_mean_vector_bf16(
    const void * K, void * mean_K,
    int batch, int seq_len, int n_kv_heads, int head_dim, int block_size,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    hipStream_t stream)
{
    if (head_dim != 128 || block_size != 128) {
        fprintf(stderr, "[dflash] launch_compute_mean_vector_bf16: unsupported shape "
                "head_dim=%d block_size=%d (only 128×128 supported)\n",
                head_dim, block_size);
        return -1;
    }
    const int n_k_blocks = (seq_len + block_size - 1) / block_size;
    dim3 grid(n_k_blocks, batch * n_kv_heads, 1);
    dim3 block(head_dim, 1, 1);
    compute_mean_vector_kernel_bf16<128, 128><<<grid, block, 0, stream>>>(
        (const hip_bfloat16 *)K, (hip_bfloat16 *)mean_K,
        batch, seq_len, n_kv_heads,
        s_K_b, s_K_n, s_K_h, s_K_d,
        s_mK_b, s_mK_m, s_mK_h, s_mK_d);
    return 0;
}

// ---- Kernel 2: compute_block_score ----
// Per (q_block, k_block) score = sum_{q rows} exp2(Q · mean_K^T * scale - rowmax).

template <int BLOCK, int D_HEAD, int N_BLOCKS_TILE>
__global__ void compute_block_score_kernel_bf16(
    const hip_bfloat16 * __restrict__ Q,
    const hip_bfloat16 * __restrict__ mean_K,
    float sm_scale,
    float * __restrict__ score,
    float * __restrict__ score_max,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len,
    int q_block_idx_max,
    int k_block_idx_max,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    int s_M_b, int s_M_m, int s_M_n, int s_M_h)
{
    const int q_block_idx = blockIdx.x;
    const int zh = blockIdx.y;
    const int b  = zh / n_q_heads;
    const int qh = zh % n_q_heads;
    if (b >= batch) return;

    const int kh           = qh * n_k_heads / n_q_heads;
    const int tid          = threadIdx.x;
    const int q_row_global = q_block_idx * BLOCK + tid;
    // Threads beyond the real sequence on the last partial block must still
    // reach every __syncthreads() below; use a flag instead of early return.
    const bool active = (tid < BLOCK) && (q_row_global < seq_len);

    float q_reg[D_HEAD] = {};
    if (active) {
        const hip_bfloat16 * Qp = Q + (size_t)b * s_Q_b
                                     + (size_t)q_row_global * s_Q_n
                                     + (size_t)qh * s_Q_h;
        #pragma unroll
        for (int d = 0; d < D_HEAD; ++d)
            q_reg[d] = static_cast<float>(Qp[(size_t)d * s_Q_d]);
    }

    extern __shared__ float smem[];

    for (int n = 0; n <= q_block_idx; ++n) {
        float dot = 0.0f;
        if (active) {
            const hip_bfloat16 * mKp = mean_K + (size_t)b * s_mK_b
                                               + (size_t)n * s_mK_m
                                               + (size_t)kh * s_mK_h;
            #pragma unroll
            for (int d = 0; d < D_HEAD; ++d)
                dot += q_reg[d] * static_cast<float>(mKp[(size_t)d * s_mK_d]);
            dot *= sm_scale * 1.4426950408889634f;
        }

        // Inactive threads contribute -inf so they don't skew the max reduction.
        smem[tid] = active ? dot : -__FLT_MAX__;
        __syncthreads();
        for (int off = BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) smem[tid] = fmaxf(smem[tid], smem[tid + off]);
            __syncthreads();
        }
        float m_block = smem[0];
        __syncthreads();

        // Inactive threads contribute 0 to the sum reduction.
        smem[tid] = active ? exp2f(dot - m_block) : 0.0f;
        __syncthreads();
        for (int off = BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) smem[tid] += smem[tid + off];
            __syncthreads();
        }
        float p_sum = smem[0];
        __syncthreads();

        if (tid == 0) {
            score    [(size_t)b*s_S_b + (size_t)q_block_idx*s_S_m + (size_t)n*s_S_n + (size_t)qh*s_S_h] = p_sum;
            score_max[(size_t)b*s_M_b + (size_t)q_block_idx*s_M_m + (size_t)n*s_M_n + (size_t)qh*s_M_h] = m_block;
        }
    }
}

extern "C" int launch_compute_block_score_bf16(
    const void * Q, const void * mean_K, float sm_scale,
    void * score, void * score_max,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int block_size,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    int s_M_b, int s_M_m, int s_M_n, int s_M_h,
    hipStream_t stream)
{
    if (head_dim != 128 || block_size != 128) {
        fprintf(stderr, "[dflash] launch_compute_block_score_bf16: unsupported shape "
                "head_dim=%d block_size=%d (only 128×128 supported)\n",
                head_dim, block_size);
        return -1;
    }
    const int M    = (seq_len + block_size - 1) / block_size;
    dim3 grid(M, batch * n_q_heads, 1);
    dim3 block(block_size, 1, 1);
    size_t smem = block_size * sizeof(float);
    compute_block_score_kernel_bf16<128, 128, 1><<<grid, block, smem, stream>>>(
        (const hip_bfloat16 *)Q, (const hip_bfloat16 *)mean_K, sm_scale,
        (float *)score, (float *)score_max,
        batch, n_q_heads, n_k_heads, seq_len, M, M,
        s_Q_b, s_Q_n, s_Q_h, s_Q_d,
        s_mK_b, s_mK_m, s_mK_h, s_mK_d,
        s_S_b, s_S_m, s_S_n, s_S_h,
        s_M_b, s_M_m, s_M_n, s_M_h);
    return 0;
}

// ---- Kernel 2b (Phase 4): compute_block_score_gemm ----
//
// Replaces the O(M²) scalar kernel with a rocWMMA GEMM:
//   score[b,m,n,h] = (mean_Q[b,m,h,:] · mean_K[b,n,kh,:]) * sm_scale
//
// Grid: (ceil(M/16), ceil(M/16), B*H). One Wave32 per 16×16 output tile.
// Inner loop: D=128 → 8 rocWMMA mma_sync calls per tile.
// Upper triangle (n_tile > m_tile) is skipped (causal — block_select handles the rest).
//
// Load layout:
//   A (mean_Q): row_major [16×16] tile, ld = s_mQ_m (stride between Q-block rows)
//   B (mean_K): col_major [16×16] tile, ld = s_mK_m (cols of B^T = rows of mean_K)
//   C (score):  Wave32 write: lane t, elem i → row=t%16, col=(t/16)*8+i

template <int D_HEAD>
__global__ void compute_block_score_gemm_kernel(
    const hip_bfloat16* __restrict__ mean_Q,  // [B, M, H,  D]
    const hip_bfloat16* __restrict__ mean_K,  // [B, M, Hk, D]
    float*              __restrict__ score,    // [B, M, M,  H]
    float sm_scale,
    int M, int n_q_heads, int n_k_heads,
    int s_mQ_b, int s_mQ_m, int s_mQ_h,
    int s_mK_b, int s_mK_m, int s_mK_h,
    int s_S_b,  int s_S_m,  int s_S_n, int s_S_h)
{
    const int m_tile = blockIdx.x;
    const int n_tile = blockIdx.y;
    const int zh     = blockIdx.z;
    const int b      = zh / n_q_heads;
    const int qh     = zh % n_q_heads;
    const int kh     = qh * n_k_heads / n_q_heads;

    if (n_tile > m_tile) return;   // upper triangle — causal

    using namespace rocwmma;
    fragment<matrix_a, 16, 16, 16, hip_bfloat16, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, hip_bfloat16, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float>                 c_frag;
    fill_fragment(c_frag, 0.0f);

    // Base pointers for this (m_tile, n_tile, b, qh/kh)
    const hip_bfloat16* mQ = mean_Q + (size_t)b * s_mQ_b
                                    + (size_t)(m_tile * 16) * s_mQ_m
                                    + (size_t)qh * s_mQ_h;
    const hip_bfloat16* mK = mean_K + (size_t)b * s_mK_b
                                    + (size_t)(n_tile * 16) * s_mK_m
                                    + (size_t)kh * s_mK_h;

    // 8 mma_sync calls over D=128 in 16-wide chunks
    #pragma unroll
    for (int k = 0; k < D_HEAD; k += 16) {
        // A: row_major [16×16], rows of mean_Q separated by s_mQ_m
        load_matrix_sync(a_frag, mQ + k, s_mQ_m);
        // B: col_major [K×N]=[16×16]; column j = row j of mean_K (stride s_mK_m)
        // → computes A × B^T = mean_Q_tile · mean_K_tile^T
        load_matrix_sync(b_frag, mK + k, s_mK_m);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Write: AMD Wave32 layout  lane t, elem i → row = t%16, col = (t/16)*8 + i
    const int lane   = threadIdx.x;
    const int r_base = m_tile * 16 + (lane % 16);
    const int c_base = n_tile * 16 + (lane / 16) * 8;
    float* Sp = score + (size_t)b * s_S_b + (size_t)qh * s_S_h;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (r_base < M && c_base + i < M)
            Sp[(size_t)r_base * s_S_m + (size_t)(c_base + i) * s_S_n] = c_frag[i] * sm_scale;
    }
}

extern "C" int launch_compute_block_score_gemm_bf16(
    const void* mean_Q, const void* mean_K, float sm_scale,
    void* score,
    int batch, int n_q_heads, int n_k_heads,
    int M, int head_dim,
    int s_mQ_b, int s_mQ_m, int s_mQ_h,
    int s_mK_b, int s_mK_m, int s_mK_h,
    int s_S_b,  int s_S_m,  int s_S_n, int s_S_h,
    hipStream_t stream)
{
    if (head_dim != 128) {
        fprintf(stderr, "[dflash] launch_compute_block_score_gemm_bf16: unsupported "
                "head_dim=%d (only 128 supported)\n", head_dim);
        return -1;
    }
    const int M16 = (M + 15) / 16;
    dim3 grid(M16, M16, batch * n_q_heads);
    dim3 block(32, 1, 1);
    compute_block_score_gemm_kernel<128><<<grid, block, 0, stream>>>(
        (const hip_bfloat16*)mean_Q,
        (const hip_bfloat16*)mean_K,
        (float*)score, sm_scale,
        M, n_q_heads, n_k_heads,
        s_mQ_b, s_mQ_m, s_mQ_h,
        s_mK_b, s_mK_m, s_mK_h,
        s_S_b,  s_S_m,  s_S_n, s_S_h);
    return 0;
}

// ---- Kernel 4: sparse_flash_forward (rocWMMA) ----
//
// FlashAttention-style online softmax over selected K-blocks.
// Tiled in SMEM: Q_sh [Q_TILE×D], KV_sh [K_TILE×D] (aliased K then V), P_sh [Q_TILE×K_TILE].
//
// AMD RDNA3 Wave32 accumulator layout for m16n16k16 float32:
//   lane t (0..31), element i (0..7): row = t % 16, col = (t / 16)*8 + i
// This differs from NVIDIA (two row-pairs per lane). Consequence:
//   - One alpha per lane (not two)
//   - rowmax/rowsum reduced via __shfl_xor(val, 16) across the two lane-halves sharing a row

template <int Q_TILE, int K_TILE, int BLOCK, int D_HEAD>
__global__ void sparse_flash_forward_kernel_bf16(
    const hip_bfloat16 * __restrict__ Q,
    const hip_bfloat16 * __restrict__ K,
    const hip_bfloat16 * __restrict__ V,
    hip_bfloat16       * __restrict__ O,
    const int32_t      * __restrict__ block_index,
    const int32_t      * __restrict__ counts,
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
    using namespace rocwmma;
    constexpr int MMA_M = 16, MMA_N = 16, MMA_K = 16;
    constexpr int NDK     = D_HEAD / MMA_K;     // D tiles along head_dim
    constexpr int NNK     = K_TILE / MMA_N;     // N tiles along K_TILE
    constexpr int N_INNER = BLOCK / K_TILE;     // inner iters per selected block
    constexpr int NTHREADS = (Q_TILE / MMA_M) * 32;

    const int q_tile_idx = blockIdx.x;
    const int zh = blockIdx.y;
    const int b  = zh / n_q_heads;
    const int qh = zh % n_q_heads;
    if (b >= batch) return;
    const int kh          = qh * n_k_heads / n_q_heads;
    const int q_block_idx = q_tile_idx * Q_TILE / BLOCK;

    const int wid  = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int tid  = threadIdx.x;

    // AMD RDNA3 Wave32: lane t owns row = t%16, cols = (t/16)*8 + {0..7}
    const int row_in_warp = lane & 15;   // 0..15
    const int col_half    = lane >> 4;   // 0 = cols 0..7, 1 = cols 8..15

    extern __shared__ __align__(16) unsigned char smem_raw[];
    hip_bfloat16 * Q_sh  = reinterpret_cast<hip_bfloat16 *>(smem_raw);
    hip_bfloat16 * KV_sh = Q_sh  + (size_t)Q_TILE * D_HEAD;
    hip_bfloat16 * P_sh  = KV_sh + (size_t)K_TILE * D_HEAD;
    float        * row_m = reinterpret_cast<float *>(P_sh + (size_t)Q_TILE * K_TILE);
    float        * row_l = row_m + Q_TILE;

    if (tid < Q_TILE) { row_m[tid] = -INFINITY; row_l[tid] = 0.0f; }

    // Cooperatively load Q [Q_TILE, D_HEAD] into Q_sh
    {
        const hip_bfloat16 * Qp = Q + (size_t)b * s_Q_b + (size_t)qh * s_Q_h;
        for (int idx = tid; idx < Q_TILE * D_HEAD; idx += NTHREADS) {
            int row = idx / D_HEAD, dim = idx - row * D_HEAD;
            int q_global = q_tile_idx * Q_TILE + row;
            Q_sh[row * D_HEAD + dim] = (q_global < seq_len)
                ? Qp[(size_t)q_global * s_Q_n + (size_t)dim * s_Q_d]
                : hip_bfloat16(0.0f);
        }
    }
    __syncthreads();

    // Pre-scale Q by sm_scale = scale * log2(e)
    {
        const float sm_scale = scale * 1.4426950408889634f;
        for (int idx = tid; idx < Q_TILE * D_HEAD; idx += NTHREADS)
            Q_sh[idx] = hip_bfloat16(static_cast<float>(Q_sh[idx]) * sm_scale);
    }
    __syncthreads();

    // O accumulator: one fragment per D col-tile per warp
    fragment<accumulator, MMA_M, MMA_N, MMA_K, float> O_frag[NDK];
    #pragma unroll
    for (int d = 0; d < NDK; ++d) fill_fragment(O_frag[d], 0.0f);

    const int hi = counts[(size_t)b*s_cnt_b + (size_t)q_block_idx*s_cnt_m + (size_t)qh*s_cnt_h];

    for (int it = 0; it < hi; ++it) {
        int blk = block_index[(size_t)b*s_idx_b + (size_t)q_block_idx*s_idx_m
                              + (size_t)it*s_idx_n + (size_t)qh*s_idx_h];
        if (blk < 0 || blk >= M_blocks) continue;
        const int k_lo_block = blk * BLOCK;
        const bool is_diag   = (blk == q_block_idx);

        #pragma unroll
        for (int inner = 0; inner < N_INNER; ++inner) {
            const int k_lo = k_lo_block + inner * K_TILE;

            // ── Load K tile [K_TILE, D_HEAD] into KV_sh ──
            {
                const hip_bfloat16 * Kp = K + (size_t)b * s_K_b + (size_t)kh * s_K_h;
                if (s_K_d == 1) {
                    const int total8 = (K_TILE * D_HEAD) / 8;
                    for (int idx = tid; idx < total8; idx += NTHREADS) {
                        const int row8 = idx / (D_HEAD / 8);
                        const int d8   = idx - row8 * (D_HEAD / 8);
                        const int j    = k_lo + row8;
                        hip_bfloat16 * dst = KV_sh + row8 * D_HEAD + d8 * 8;
                        if (j < seq_len)
                            *reinterpret_cast<uint4*>(dst) =
                                *reinterpret_cast<const uint4*>(Kp + (size_t)j*s_K_n + (size_t)(d8*8));
                        else
                            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
                    }
                }
            }
            __syncthreads();

            // ── S = Q @ K^T in register fragments ──
            fragment<accumulator, MMA_M, MMA_N, MMA_K, float> S_frag[NNK];
            #pragma unroll
            for (int nt = 0; nt < NNK; ++nt) fill_fragment(S_frag[nt], 0.0f);
            {
                fragment<matrix_a, MMA_M, MMA_N, MMA_K, hip_bfloat16, row_major> Af;
                fragment<matrix_b, MMA_M, MMA_N, MMA_K, hip_bfloat16, col_major> Bf;
                #pragma unroll
                for (int dk = 0; dk < NDK; ++dk) {
                    load_matrix_sync(Af, Q_sh + (size_t)(wid*MMA_M)*D_HEAD + dk*MMA_K, D_HEAD);
                    #pragma unroll
                    for (int nt = 0; nt < NNK; ++nt) {
                        load_matrix_sync(Bf, KV_sh + (size_t)(nt*MMA_N)*D_HEAD + dk*MMA_K, D_HEAD);
                        mma_sync(S_frag[nt], Af, Bf, S_frag[nt]);
                    }
                }
            }

            // ── Causal mask + rowmax (AMD RDNA3 layout) ──
            // lane t: row = t%16, col = (t/16)*8 + elem_i
            const int row_g = q_tile_idx * Q_TILE + wid * MMA_M + row_in_warp;
            float lm = -INFINITY;
            #pragma unroll
            for (int nt = 0; nt < NNK; ++nt) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int col_g = k_lo + nt * MMA_N + col_half * 8 + i;
                    bool valid = (col_g < seq_len);
                    if (is_diag) valid = valid && (col_g <= row_g);
                    if (!valid) S_frag[nt][i] = -INFINITY;
                    lm = fmaxf(lm, S_frag[nt][i]);
                }
            }
            // Merge rowmax: lane t and t+16 own the same row; shfl_xor(16) swaps them.
            lm = fmaxf(lm, __shfl_xor(lm, 16));

            // ── Softmax numerics: read old state, compute alpha ──
            const int row_warp = wid * MMA_M + row_in_warp;
            const float m_old = row_m[row_warp];
            const float l_old = row_l[row_warp];
            const float m_new = fmaxf(m_old, lm);
            const float alpha  = (m_old == -INFINITY) ? 0.0f : exp2f(m_old - m_new);

            // ── Compute P = exp2(S - m_new), scatter to P_sh, accumulate rowsum ──
            float rs = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < NNK; ++nt) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float v = S_frag[nt][i];
                    const float p = (v == -INFINITY) ? 0.0f : exp2f(v - m_new);
                    rs += p;
                    const int col_k = nt * MMA_N + col_half * 8 + i;
                    P_sh[(size_t)(wid * MMA_M + row_in_warp) * K_TILE + col_k] = hip_bfloat16(p);
                }
            }
            // Merge rowsum across the two lane-halves sharing a row
            rs += __shfl_xor(rs, 16);

            // Only col_half==0 (lanes 0..15) writes SMEM to avoid redundant stores
            if (col_half == 0) {
                row_m[row_warp] = m_new;
                row_l[row_warp] = alpha * l_old + rs;
            }

            // Rescale O accumulator (all 8 elements per lane belong to the same row)
            #pragma unroll
            for (int d = 0; d < NDK; ++d)
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    O_frag[d][i] *= alpha;

            __syncthreads();  // P_sh writes visible before V load overwrites KV_sh

            // ── Load V tile [K_TILE, D_HEAD] into KV_sh ──
            {
                const hip_bfloat16 * Vp = V + (size_t)b * s_V_b + (size_t)kh * s_V_h;
                if (s_V_d == 1) {
                    const int total8 = (K_TILE * D_HEAD) / 8;
                    for (int idx = tid; idx < total8; idx += NTHREADS) {
                        const int row8 = idx / (D_HEAD / 8);
                        const int d8   = idx - row8 * (D_HEAD / 8);
                        const int j    = k_lo + row8;
                        hip_bfloat16 * dst = KV_sh + row8 * D_HEAD + d8 * 8;
                        if (j < seq_len)
                            *reinterpret_cast<uint4*>(dst) =
                                *reinterpret_cast<const uint4*>(Vp + (size_t)j*s_V_n + (size_t)(d8*8));
                        else
                            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
                    }
                }
            }
            __syncthreads();

            // ── O += P @ V via WMMA ──
            {
                fragment<matrix_a, MMA_M, MMA_N, MMA_K, hip_bfloat16, row_major> Af;
                fragment<matrix_b, MMA_M, MMA_N, MMA_K, hip_bfloat16, row_major> Bf;
                #pragma unroll
                for (int kk = 0; kk < NNK; ++kk) {
                    load_matrix_sync(Af, P_sh + (size_t)(wid*MMA_M)*K_TILE + kk*MMA_K, K_TILE);
                    #pragma unroll
                    for (int dt = 0; dt < NDK; ++dt) {
                        load_matrix_sync(Bf, KV_sh + (size_t)(kk*MMA_K)*D_HEAD + dt*MMA_N, D_HEAD);
                        mma_sync(O_frag[dt], Af, Bf, O_frag[dt]);
                    }
                }
            }
            __syncthreads();
        } // inner
    } // it

    // Write O = acc / l_i into global memory.
    // Reuse KV_sh as float scratch [Q_TILE, MMA_N] for one D col-tile at a time.
    float * stage_f32 = reinterpret_cast<float *>(KV_sh);
    #pragma unroll
    for (int d = 0; d < NDK; ++d) {
        __syncthreads();
        store_matrix_sync(stage_f32 + (size_t)(wid * MMA_M) * MMA_N,
                          O_frag[d], MMA_N, mem_row_major);
        __syncthreads();
        // Lanes 0..15 each handle one row of the warp's 16-row tile.
        if (lane < MMA_M) {
            const int row       = wid * MMA_M + lane;
            const int q_global  = q_tile_idx * Q_TILE + row;
            if (q_global < seq_len) {
                hip_bfloat16 * Op = O + (size_t)b*s_O_b + (size_t)q_global*s_O_n + (size_t)qh*s_O_h;
                const float l    = row_l[row];
                const float l_rec = (l > 0.0f) ? (1.0f / l) : 1.0f;
                const float * srow = stage_f32 + (size_t)row * MMA_N;
                #pragma unroll
                for (int dd = 0; dd < MMA_N; ++dd)
                    Op[(size_t)(d * MMA_N + dd) * s_O_d] = hip_bfloat16(srow[dd] * l_rec);
            }
        }
    }
}

extern "C" void launch_sparse_flash_forward_bf16(
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
    hipStream_t stream)
{
    const int M       = (seq_len + block_size - 1) / block_size;
    const int q_tiles = (seq_len + q_tile   - 1) / q_tile;
    dim3 grid(q_tiles, batch * n_q_heads, 1);
    if (q_tile == 64 && block_size == 128 && head_dim == 128) {
        constexpr int Q_TILE = 64, K_TILE = 64, BLOCK = 128, D_HEAD = 128;
        const size_t smem_bytes = sizeof(hip_bfloat16) * (Q_TILE * D_HEAD)
                                + sizeof(hip_bfloat16) * (K_TILE * D_HEAD)
                                + sizeof(hip_bfloat16) * (Q_TILE * K_TILE)
                                + sizeof(float)        * (2 * Q_TILE);
        dim3 block128(128, 1, 1);
        hipFuncSetAttribute(
            reinterpret_cast<const void *>(
                sparse_flash_forward_kernel_bf16<Q_TILE, K_TILE, BLOCK, D_HEAD>),
            hipFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes);
        sparse_flash_forward_kernel_bf16<Q_TILE, K_TILE, BLOCK, D_HEAD>
            <<<grid, block128, smem_bytes, stream>>>(
            (const hip_bfloat16 *)Q, (const hip_bfloat16 *)K,
            (const hip_bfloat16 *)V, (hip_bfloat16 *)O,
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

// ---- Kernel 4b: transpose_kv ----
// Transposes [B, S, H, D] → [B, H, S, D] so that per-head token reads
// inside sparse_flash_forward_kernel_bf16 are contiguous (s_K_n = D, not
// H*D).  Each warp handles one (b, h) row-tile of size TILE x D.
// Launched from launch_bsa_sparse_flash_forward_bf16 as a pre-pass.

template <int D_HEAD>
__global__ void transpose_kv_kernel(
    const hip_bfloat16 * __restrict__ src,  // [B, S, H, D]
    hip_bfloat16       * __restrict__ dst,  // [B, H, S, D]
    int B, int S, int H)
{
    const int zh   = blockIdx.y;
    const int b    = zh / H;
    const int h    = zh % H;
    if (b >= B) return;

    const int n_start = (int)blockIdx.x * blockDim.x + (int)threadIdx.x;
    const int stride  = gridDim.x * (int)blockDim.x;

    const hip_bfloat16 * sp = src + (size_t)b * S * H * D_HEAD + (size_t)h * D_HEAD;
    hip_bfloat16       * dp = dst + (size_t)b * H * S * D_HEAD + (size_t)h * S * D_HEAD;

    for (int n = n_start; n < S; n += stride) {
        // Copy D elements. D=128 → 8 uint4 loads per token.
        #pragma unroll
        for (int d8 = 0; d8 < D_HEAD / 8; ++d8) {
            reinterpret_cast<uint4 &>(dp[n * D_HEAD + d8 * 8]) =
                *reinterpret_cast<const uint4 *>(sp + (size_t)n * H * D_HEAD + d8 * 8);
        }
    }
}

extern "C" int launch_transpose_kv_bf16(
    const void * src, void * dst,
    int B, int S, int H, int head_dim,
    hipStream_t stream)
{
    if (head_dim != 128) {
        fprintf(stderr, "[dflash] launch_transpose_kv_bf16: unsupported head_dim=%d "
                "(only 128 supported)\n", head_dim);
        return -1;
    }
    const int threads = 256;
    const int blocks_n = (S + threads - 1) / threads;
    dim3 grid(blocks_n, B * H, 1);
    transpose_kv_kernel<128><<<grid, threads, 0, stream>>>(
        (const hip_bfloat16 *)src, (hip_bfloat16 *)dst, B, S, H);
    return 0;
}

// ---- Kernel 3: block_select ----
// One warp per (B, M, H). Warp ballot compacts indices in sorted order.

__global__ void block_select_kernel(
    const float * __restrict__ score,
    int B, int M, int N, int H,
    int attention_sink, int window, int last_n_full, float alpha,
    int s_b, int s_m, int s_n, int s_h,
    int idx_s_b, int idx_s_m, int idx_s_n, int idx_s_h,
    int cnt_s_b, int cnt_s_m, int cnt_s_h,
    int32_t * __restrict__ idx_out,
    int32_t * __restrict__ cnt_out)
{
    const int b    = blockIdx.x;
    const int m    = blockIdx.y;
    const int h    = blockIdx.z;
    const int lane = threadIdx.x;
    if (b >= B || m >= M || h >= H) return;

    const float * sp   = score   + (size_t)b*s_b   + (size_t)m*s_m   + (size_t)h*s_h;
    int32_t     * idxp = idx_out + (size_t)b*idx_s_b + (size_t)m*idx_s_m + (size_t)h*idx_s_h;

    const bool  last_full = (m >= M - last_n_full);
    const float NEG_INF   = -INFINITY;

    // Pass 1: warp-reduce max score in [0, m]
    float local_max = NEG_INF;
    for (int n_base = 0; n_base <= m; n_base += 32) {
        const int n = n_base + lane;
        local_max = fmaxf(local_max, (n <= m) ? sp[(size_t)n * s_n] : NEG_INF);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    const float thresh = local_max * alpha;

    // Pass 2: predicate + warp-ballot compact (output sorted by n)
    int total = 0;
    for (int n_base = 0; n_base <= m; n_base += 32) {
        const int n     = n_base + lane;
        const bool valid = (n <= m);
        bool keep = false;
        if (valid) {
            const float v = sp[(size_t)n * s_n];
            keep = last_full || (n < attention_sink) || ((m - n) < window) || (v >= thresh);
        }
        const unsigned mask = (unsigned)__ballot(keep);
        const int rank = __popc(mask & ((1u << lane) - 1u));
        if (keep) idxp[(size_t)(total + rank) * idx_s_n] = (int32_t)n;
        total += __popc(mask);
    }

    // Pad remaining slots with -1
    for (int n = total + lane; n < N; n += 32)
        idxp[(size_t)n * idx_s_n] = (int32_t)-1;

    if (lane == 0)
        cnt_out[(size_t)b*cnt_s_b + (size_t)m*cnt_s_m + (size_t)h*cnt_s_h] = (int32_t)total;
}

extern "C" void launch_block_select(
    const float * score,
    int B, int M, int N, int H,
    int attention_sink, int window, int last_n_full, float alpha,
    int s_b, int s_m, int s_n, int s_h,
    int idx_s_b, int idx_s_m, int idx_s_n, int idx_s_h,
    int cnt_s_b, int cnt_s_m, int cnt_s_h,
    int32_t * idx_out, int32_t * cnt_out,
    hipStream_t stream)
{
    block_select_kernel<<<dim3(B, M, H), dim3(32, 1, 1), 0, stream>>>(
        score, B, M, N, H,
        attention_sink, window, last_n_full, alpha,
        s_b, s_m, s_n, s_h,
        idx_s_b, idx_s_m, idx_s_n, idx_s_h,
        cnt_s_b, cnt_s_m, cnt_s_h,
        idx_out, cnt_out);
}

// launch_rms_norm_mul_w_f32 is defined in rms_norm_hip.cu (compiled for all HIP builds).

} // namespace flashprefill
} // namespace dflash::common
