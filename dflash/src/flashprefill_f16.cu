// F16 WMMA port of FlashPrefill kernels for NVIDIA Volta (sm_70).
//
// Same algorithm as flashprefill_kernels.cu (bf16 WMMA, sm_80+) but:
//   - F16 WMMA m32n8k16 instead of __nv_bfloat16 m16n16k16
//   - cooperative shared-memory loads instead of cp.async (Volta has no async copy)
//   - all tensor I/O in F16 (half), no __nv_bfloat16
//
// Dispatched from flashprefill.cpp when DFLASH27B_HAVE_VOLTA_FLASHPREFILL is set
// and the drafter's persistent buffers are GGML_TYPE_F16.
//
// WMMA intrinsics are only available on sm_70+.  In multi-arch fat binaries
// (e.g. 70;61) the kernels below must not be instantiated for sm_60-69.
// Guard with __CUDA_ARCH__ so nvcc silently skips them on Pascal.

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace dflash27b {
namespace flashprefill {

// ── Kernel 1: compute_mean_vector (F16) ──────────────────────────────
// Scalar reduction, same algorithm as bf16 version but reads/writes F16.

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

extern "C" void launch_compute_mean_vector_f16(
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

// ── Kernel 2: compute_block_score (F16) ──────────────────────────────
// Same algorithm as bf16 version. Reads F16 Q and F16 mean_K, converts
// to F32 in registers, does dot product + softmax in F32, writes F32 output.

template <int BLOCK, int D_HEAD>
__global__ void compute_block_score_kernel_f16(
    const half * __restrict__ Q,
    const half * __restrict__ mean_K,
    float sm_scale,
    float * __restrict__ score,
    float * __restrict__ score_max,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int block_size,
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
    const int q_row_global = q_block_idx * BLOCK + tid;
    const bool active = (tid < BLOCK && q_row_global < seq_len);

    // Load Q row into F32 registers (predicate to avoid OOB for tail threads)
    float q_reg[D_HEAD];
    const half * Qp = Q + (size_t)b * s_Q_b
                              + (size_t)(q_block_idx * BLOCK + min(tid, seq_len - 1)) * s_Q_n
                              + (size_t)qh * s_Q_h;
    #pragma unroll
    for (int d = 0; d < D_HEAD; ++d) {
        q_reg[d] = active ? __half2float(Qp[(size_t)d * s_Q_d]) : 0.0f;
    }

    extern __shared__ float smem[];

    for (int n = 0; n <= q_block_idx; ++n) {
        const half * mKp = mean_K + (size_t)b * s_mK_b
                                          + (size_t)n * s_mK_m
                                          + (size_t)kh * s_mK_h;
        float dot = active ? 0.0f : -INFINITY;
        if (active) {
            #pragma unroll
            for (int d = 0; d < D_HEAD; ++d) {
                dot += q_reg[d] * __half2float(mKp[(size_t)d * s_mK_d]);
            }
            dot *= sm_scale;
        }

        smem[tid] = dot;
        __syncthreads();
        for (int off = BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) smem[tid] = fmaxf(smem[tid], smem[tid + off]);
            __syncthreads();
        }
        float m_block = smem[0];
        __syncthreads();

        float p = expf(dot - m_block);
        smem[tid] = p;
        __syncthreads();
        for (int off = BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) smem[tid] += smem[tid + off];
            __syncthreads();
        }
        float p_sum = smem[0];
        __syncthreads();

        if (tid == 0) {
            score    [(size_t)b * s_S_b + (size_t)q_block_idx * s_S_m
                      + (size_t)n * s_S_n + (size_t)qh * s_S_h] = p_sum;
            score_max[(size_t)b * s_M_b + (size_t)q_block_idx * s_M_m
                      + (size_t)n * s_M_n + (size_t)qh * s_M_h] = m_block;
        }
    }
}

extern "C" void launch_compute_block_score_f16(
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
        compute_block_score_kernel_f16<128, 128><<<grid, block, smem, stream>>>(
            (const half *)Q, (const half *)mean_K, sm_scale,
            (float *)score, (float *)score_max,
            batch, n_q_heads, n_k_heads, seq_len, block_size,
            s_Q_b, s_Q_n, s_Q_h, s_Q_d,
            s_mK_b, s_mK_m, s_mK_h, s_mK_d,
            s_S_b, s_S_m, s_S_n, s_S_h,
            s_M_b, s_M_m, s_M_n, s_M_h);
    }
}

// ── Kernel 4: sparse_flash_forward (F16 WMMA) ───────────────────────
//
// FlashAttention-style online softmax over selected K-blocks.
// Uses Volta F16 WMMA (m32n8k16) for QK and PV matrix multiplications.
//
// WMMA accumulator layout (m32n8k16 F16):
//   Each thread owns 8 elements of the 32×8 output tile.
//   Fragment x[0..7] maps to specific row/col positions.
//   1 warp (32 threads) per 32×8 output tile.
//
// Tile config: Q_TILE=64, K_TILE=64, BLOCK=128, D_HEAD=128
//   - 2 warps per Q tile (64/32 = 2), 128 threads/block
//   - Shared memory: Q(16K) + KV(16K) + P(8K) + row_state(2.5KB) ≈ 43 KB
//     (Volta shared mem limit: 48 KB)

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
    using namespace nvcuda;
    // Volta F16 WMMA: m32n8k16
    constexpr int MMA_M = 32, MMA_N = 8, MMA_K = 16;
    constexpr int NDK = D_HEAD / MMA_K;       // 128/16 = 8 D-column tiles
    constexpr int NNK = K_TILE / MMA_K;       // 64/16 = 4 K-row tiles
    constexpr int N_INNER = BLOCK / K_TILE;   // 128/64 = 2 inner iters per selected block
    constexpr int WARPS_PER_QTILE = Q_TILE / MMA_M;  // 64/32 = 2 warps
    constexpr int NTHREADS = WARPS_PER_QTILE * 32;   // 2 * 32 = 64 threads

    const int q_tile_idx = blockIdx.x;
    const int zh = blockIdx.y;
    const int b  = zh / n_q_heads;
    const int qh = zh % n_q_heads;
    if (b >= batch) return;
    const int kh = qh * n_k_heads / n_q_heads;
    const int q_block_idx = q_tile_idx * Q_TILE / BLOCK;

    const int wid  = threadIdx.x / 32;        // 0..1 (warp within tile)
    const int lane = threadIdx.x & 31;        // 0..31 (lane within warp)

    // Shared memory: Q + KV (aliased) + P + row max/logsum
    extern __shared__ unsigned char smem_raw[];
    half * Q_sh  = reinterpret_cast<half*>(smem_raw);
    half * KV_sh = Q_sh + (size_t)Q_TILE * D_HEAD;
    half * P_sh  = KV_sh + (size_t)K_TILE * D_HEAD;
    float * row_m = reinterpret_cast<float*>(P_sh + (size_t)Q_TILE * K_TILE);
    float * row_l = row_m + Q_TILE;

    if (threadIdx.x < Q_TILE) {
        row_m[threadIdx.x] = -INFINITY;
        row_l[threadIdx.x] = 0.0f;
    }

    // ── Cooperative load Q [Q_TILE, D_HEAD] ──
    {
        const half * Qp = Q + (size_t)b * s_Q_b + (size_t)qh * s_Q_h;
        for (int idx = threadIdx.x; idx < Q_TILE * D_HEAD; idx += NTHREADS) {
            int row = idx / D_HEAD;
            int dim = idx - row * D_HEAD;
            int q_global = q_tile_idx * Q_TILE + row;
            Q_sh[row * D_HEAD + dim] = (q_global < seq_len)
                ? Qp[(size_t)q_global * s_Q_n + (size_t)dim * s_Q_d]
                : __float2half(0.0f);
        }
    }
    __syncthreads();

    // Pre-scale Q by sm_scale = scale * log2(e)
    {
        const float sm_scale = scale * 1.4426950408889634f;
        for (int idx = threadIdx.x; idx < Q_TILE * D_HEAD; idx += NTHREADS) {
            float v = __half2float(Q_sh[idx]);
            Q_sh[idx] = __float2half(v * sm_scale);
        }
    }
    __syncthreads();

    // O accumulator fragments (registers, one per D-column tile)
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> O_frag[NDK];
    #pragma unroll
    for (int d = 0; d < NDK; ++d) wmma::fill_fragment(O_frag[d], 0.0f);

    const int hi = counts[(size_t)b * s_cnt_b + (size_t)q_block_idx * s_cnt_m + (size_t)qh * s_cnt_h];

    for (int it = 0; it < hi; ++it) {
        int blk = block_index[(size_t)b * s_idx_b + (size_t)q_block_idx * s_idx_m
                              + (size_t)it * s_idx_n + (size_t)qh * s_idx_h];
        if (blk < 0 || blk >= M_blocks) continue;
        const int k_lo_block = blk * BLOCK;
        const bool is_diag = (blk == q_block_idx);

        #pragma unroll
        for (int inner = 0; inner < N_INNER; ++inner) {
            const int k_lo = k_lo_block + inner * K_TILE;

            // ── Cooperative load K tile [K_TILE, D_HEAD] into KV_sh ──
            {
                const half * Kp = K + (size_t)b * s_K_b + (size_t)kh * s_K_h;
                int total8 = (K_TILE * D_HEAD) / 8;
                for (int idx = threadIdx.x; idx < total8; idx += NTHREADS) {
                    int row8 = idx / (D_HEAD / 8);
                    int d8   = idx - row8 * (D_HEAD / 8);
                    int j = k_lo + row8;
                    half2 * dst = reinterpret_cast<half2*>(KV_sh + row8 * D_HEAD + d8 * 8);
                    if (j < seq_len) {
                        const half2 * src = reinterpret_cast<const half2*>(Kp)
                            + (size_t)j * s_K_n + (size_t)(d8 * 8);
                        dst[0] = src[0]; dst[1] = src[1];
                        dst[2] = src[2]; dst[3] = src[3];
                    } else {
                        half2 z = __floats2half2_rn(0.0f, 0.0f);
                        dst[0] = z; dst[1] = z; dst[2] = z; dst[3] = z;
                    }
                }
            }
            __syncthreads();

            // ── S = Q @ K^T via F16 WMMA (m32n8k16) ──
            // Loop-swap: dk-outer reuses Af across NNK nt-iters
            wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> S_frag[NNK];
            #pragma unroll
            for (int nt = 0; nt < NNK; ++nt) wmma::fill_fragment(S_frag[nt], 0.0f);
            {
                wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, half, wmma::row_major> Af;
                wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, half, wmma::col_major> Bf;
                #pragma unroll
                for (int dk = 0; dk < NDK; ++dk) {
                    wmma::load_matrix_sync(Af,
                        Q_sh + (size_t)(wid * MMA_M) * D_HEAD + dk * MMA_K, D_HEAD);
                    #pragma unroll
                    for (int nt = 0; nt < NNK; ++nt) {
                        wmma::load_matrix_sync(Bf,
                            KV_sh + (size_t)(nt * MMA_K) * D_HEAD + dk * MMA_K, D_HEAD);
                        wmma::mma_sync(S_frag[nt], Af, Bf, S_frag[nt]);
                    }
                }
            }

            // ── Causal mask + per-lane row max ──
            // m32n8k16 accumulator layout: each thread has x[0..7]
            // x[i] maps to a specific (row, col) in the 32×8 output
            // Row index for element i: (i % 4) * 2 + (i / 4) % 2 ... complex
            // Simplified: process all 8 elements, apply mask, find max per row

            // For m32n8k16, the fragment layout is documented as:
            // x[0..3]: first 4 columns, x[4..7]: next 4 columns
            // Each column has 4 rows from this thread
            // Row assignment varies per element - we process element-wise

            float lm[Q_TILE] = {}; // track max per row in this warp's 32 rows
            #pragma unroll
            for (int row = 0; row < MMA_M; ++row) lm[row] = -INFINITY;

            const int row_base = q_tile_idx * Q_TILE + wid * MMA_M;

            #pragma unroll
            for (int nt = 0; nt < NNK; ++nt) {
                // For each element in the fragment, determine its row/col
                // and apply causal mask
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    // m32n8k16 fragment layout (from NV docs):
                    // row = (lane % 4) * 8 + (e / 4) * 2 + (e % 2)
                    // col = (lane / 4) * 2 + (e / 2) % 2
                    int row_in_tile = (lane % 4) * 8 + (e / 4) * 2 + (e % 2);
                    int col_in_tile = (lane / 4) * 2 + (e / 2) % 2;

                    int col_g = k_lo + nt * MMA_K + col_in_tile;
                    int row_g = row_base + row_in_tile;

                    bool valid = (col_g < seq_len);
                    if (is_diag) valid = valid && (col_g <= row_g);
                    if (!valid) S_frag[nt].x[e] = -INFINITY;

                    lm[row_in_tile] = fmaxf(lm[row_in_tile], S_frag[nt].x[e]);
                }
            }

            // Row max reduction across threads in warp
            #pragma unroll
            for (int row = 0; row < MMA_M; ++row) {
                float val = lm[row];
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
                }
                lm[row] = val;
            }

            // Read old row state; group leader reads from SMEM
            float m_old[MMA_M], l_old[MMA_M];
            #pragma unroll
            for (int row = 0; row < MMA_M; ++row) {
                if (lane == (row % 32)) {  // each row has exactly one thread reading
                    m_old[row] = row_m[wid * MMA_M + row];
                    l_old[row] = row_l[wid * MMA_M + row];
                }
            }
            // Broadcast to all threads
            #pragma unroll
            for (int row = 0; row < MMA_M; ++row) {
                m_old[row] = __shfl_sync(0xffffffff, m_old[row], row % 32);
                l_old[row] = __shfl_sync(0xffffffff, l_old[row], row % 32);
            }

            // Compute P (softmax) and row sums, write F16 P to P_sh
            float rs[MMA_M] = {};
            float m_new[MMA_M];
            #pragma unroll
            for (int row = 0; row < MMA_M; ++row) {
                m_new[row] = fmaxf(m_old[row], lm[row]);
            }

            #pragma unroll
            for (int nt = 0; nt < NNK; ++nt) {
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    int row_in_tile = (lane % 4) * 8 + (e / 4) * 2 + (e % 2);
                    int col_in_tile = (lane / 4) * 2 + (e / 2) % 2;

                    float p = (S_frag[nt].x[e] == -INFINITY) ? 0.0f
                               : exp2f(S_frag[nt].x[e] - m_new[row_in_tile]);
                    rs[row_in_tile] += p;

                    int p_row = wid * MMA_M + row_in_tile;
                    P_sh[(size_t)p_row * K_TILE + nt * MMA_K + col_in_tile] = __float2half(p);
                }
            }

            // Row sum reduction
            #pragma unroll
            for (int row = 0; row < MMA_M; ++row) {
                float val = rs[row];
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    val += __shfl_xor_sync(0xffffffff, val, mask);
                }
                rs[row] = val;
            }

            // Write new row state
            #pragma unroll
            for (int row = 0; row < MMA_M; ++row) {
                float alpha = (m_old[row] == -INFINITY) ? 0.0f : exp2f(m_old[row] - m_new[row]);
                row_m[wid * MMA_M + row] = m_new[row];
                row_l[wid * MMA_M + row] = alpha * l_old[row] + rs[row];
            }

            // Rescale O accumulator - each frag has 8 elements
            // Need to figure out which row each element belongs to
            #pragma unroll
            for (int d = 0; d < NDK; ++d) {
                // Each element in the fragment needs its row's alpha
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    int row_in_tile = (lane % 4) * 8 + (e / 4) * 2 + (e % 2);
                    float alpha = (m_old[row_in_tile] == -INFINITY) ? 0.0f
                               : exp2f(m_old[row_in_tile] - m_new[row_in_tile]);
                    O_frag[d].x[e] *= alpha;
                }
            }

            __syncthreads();

            // ── Cooperative load V tile [K_TILE, D_HEAD] into KV_sh ──
            {
                const half * Vp = V + (size_t)b * s_V_b + (size_t)kh * s_V_h;
                int total8 = (K_TILE * D_HEAD) / 8;
                for (int idx = threadIdx.x; idx < total8; idx += NTHREADS) {
                    int row8 = idx / (D_HEAD / 8);
                    int d8   = idx - row8 * (D_HEAD / 8);
                    int j = k_lo + row8;
                    half2 * dst = reinterpret_cast<half2*>(KV_sh + row8 * D_HEAD + d8 * 8);
                    if (j < seq_len) {
                        const half2 * src = reinterpret_cast<const half2*>(Vp)
                            + (size_t)j * s_V_n + (size_t)(d8 * 8);
                        dst[0] = src[0]; dst[1] = src[1];
                        dst[2] = src[2]; dst[3] = src[3];
                    } else {
                        half2 z = __floats2half2_rn(0.0f, 0.0f);
                        dst[0] = z; dst[1] = z; dst[2] = z; dst[3] = z;
                    }
                }
            }
            __syncthreads();

            // ── O += P @ V via F16 WMMA (m32n8k16) ──
            // Loop-swap: kk-outer reuses Af across NDK dt-iters
            {
                wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, half, wmma::row_major> Af;
                wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, half, wmma::row_major> Bf;
                #pragma unroll
                for (int kk = 0; kk < NNK; ++kk) {
                    wmma::load_matrix_sync(Af,
                        P_sh + (size_t)(wid * MMA_M) * K_TILE + kk * MMA_K, K_TILE);
                    #pragma unroll
                    for (int dt = 0; dt < NDK; ++dt) {
                        wmma::load_matrix_sync(Bf,
                            KV_sh + (size_t)(kk * MMA_K) * D_HEAD + dt * MMA_N, D_HEAD);
                        wmma::mma_sync(O_frag[dt], Af, Bf, O_frag[dt]);
                    }
                }
            }
            __syncthreads();
        } // inner
    } // it

    // ── Write O = acc / l_i ──
    // Use KV_sh as F32 staging (16 KB available, need 32*8*4 = 1 KB per frag)
    float * stage_f32 = reinterpret_cast<float*>(KV_sh);
    #pragma unroll
    for (int d = 0; d < NDK; ++d) {
        __syncthreads();
        wmma::store_matrix_sync(stage_f32 + (size_t)(wid * MMA_M) * MMA_N,
                                O_frag[d], MMA_N, wmma::mem_row_major);
        __syncthreads();

        // Each thread writes its rows from the staging area
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            int row_in_tile = (lane % 4) * 8 + (e / 4) * 2 + (e % 2);
            int col_in_tile = (lane / 4) * 2 + (e / 2) % 2;
            int q_global = q_tile_idx * Q_TILE + wid * MMA_M + row_in_tile;
            int d_global = d * MMA_N + col_in_tile;

            if (q_global < seq_len) {
                half * Op = O + (size_t)b * s_O_b + (size_t)q_global * s_O_n + (size_t)qh * s_O_h;
                float l = row_l[wid * MMA_M + row_in_tile];
                float l_rec = (l > 0.0f) ? (1.0f / l) : 1.0f;
                Op[(size_t)d_global * s_O_d] = __float2half(stage_f32[(size_t)(wid * MMA_M + row_in_tile) * MMA_N + col_in_tile] * l_rec);
            }
        }
    }
}

extern "C" void launch_sparse_flash_forward_f16(
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
    if (q_tile == 64 && block_size == 128 && head_dim == 128) {
        constexpr int Q_TILE = 64, K_TILE = 64, BLOCK = 128, D_HEAD = 128;
        size_t smem_bytes = sizeof(half) * (Q_TILE * D_HEAD)
                           + sizeof(half) * (K_TILE * D_HEAD)
                           + sizeof(half) * (Q_TILE * K_TILE)
                           + sizeof(float) * (2 * Q_TILE);
        dim3 block64(64, 1, 1);
        cudaFuncSetAttribute(
            (const void*)sparse_flash_forward_kernel_f16<Q_TILE, K_TILE, BLOCK, D_HEAD>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes);
        sparse_flash_forward_kernel_f16<Q_TILE, K_TILE, BLOCK, D_HEAD><<<grid, block64, smem_bytes, stream>>>(
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

// ── Kernel 3: block_select (shared, architecture-independent) ──
//
// One warp per (B, M, H). Each warp scans n in [0, m] in chunks of 32, takes
// the max of scores[b,m,n,h] for the threshold, then re-scans applying the
// keep predicate (sink | window | last_n_full | score >= max*alpha).

__global__ void block_select_kernel_f16(
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
    const int lane = threadIdx.x;  // 0..31, single warp CTA

    if (b >= B || m >= M || h >= H) return;

    const float * sp = score + (size_t)b*s_b + (size_t)m*s_m + (size_t)h*s_h;
    int32_t * idxp = idx_out + (size_t)b*idx_s_b + (size_t)m*idx_s_m + (size_t)h*idx_s_h;

    const bool last_full = (m >= M - last_n_full);
    const float NEG_INF = -INFINITY;

    // Pass 1: max score in [0, m] (warp reduce).
    float local_max = NEG_INF;
    for (int n_base = 0; n_base <= m; n_base += 32) {
        int n = n_base + lane;
        bool valid = (n <= m);
        float v = valid ? sp[(size_t)n * s_n] : NEG_INF;
        local_max = fmaxf(local_max, v);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));
    const float max_score = local_max;
    const float thresh = max_score * alpha;

    // Pass 2: predicate + warp-ballot compact, sorted by n.
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
        unsigned mask = __ballot_sync(0xffffffff, keep);
        int rank = __popc(mask & ((1u << lane) - 1u));
        if (keep) {
            idxp[(size_t)(total + rank) * idx_s_n] = (int32_t)n;
        }
        total += __popc(mask);
    }

    // Tail pad with -1 across [total, N).
    for (int n = total + lane; n < N; n += 32) {
        idxp[(size_t)n * idx_s_n] = (int32_t)-1;
    }

    if (lane == 0) {
        cnt_out[(size_t)b*cnt_s_b + (size_t)m*cnt_s_m + (size_t)h*cnt_s_h] = (int32_t)total;
    }
}

extern "C" void launch_block_select_f16(
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
    block_select_kernel_f16<<<grid, block, 0, stream>>>(
        score, B, M, N, H,
        attention_sink, window, last_n_full, alpha,
        s_b, s_m, s_n, s_h,
        idx_s_b, idx_s_m, idx_s_n, idx_s_h,
        cnt_s_b, cnt_s_m, cnt_s_h,
        idx_out, cnt_out);
}

} // namespace flashprefill
} // namespace dflash27b

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
