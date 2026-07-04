#pragma once

// Chunked-prefill FlashAttention with online softmax and O(CHUNK) temp memory.
//
// Ported from Luce-Org-adjacent project dusterbloom-turboquant (fattn.cu:750-).
// Strips TBQ-specific dequant kernels and Walsh-Hadamard rotations; retains the
// KV-type-agnostic online softmax kernels and the generic chunked driver.
//
// Three kernels (identical to turboquant):
//   k_chunked_attn_init        — init O_acc/l_acc/m_acc
//   k_chunked_softmax_update   — online softmax over one KV chunk
//   k_chunked_attn_finalize    — normalize O by l, transpose to ggml layout
//
// The driver (ggml_cuda_flash_attn_ext_chunked) is provided in fattn-chunked.cu.
//
// Use when: Q->ne[1] > 1 (prefill) AND K->ne[1] > CHUNKED_PREFILL_THRESHOLD.
// KV must be dequantizable to f32 (f16, bf16, Q4_0, Q8_0). cuBLAS SGEMM is used
// for Q@K^T and P@V on FP32 temp buffers.

#include "common.cuh"
#include "fattn-common.cuh"
#include "tq3-quant.cuh"

// Adaptive chunk-size configuration
static constexpr int CHUNKED_PF_MAX = 8192;
static constexpr int CHUNKED_PF_MIN = 256;

// Compute largest power-of-2 chunk size that fits in available GPU memory.
// Accounts for S buffer (nh_q * nq * chunk * 4) and K/V temp buffers
// (2 * D * chunk * nh_kv * 4). Reserves 512 MB headroom.
static inline int chunked_pf_compute_chunk_size(size_t free_bytes,
                                                int64_t nh_q,
                                                int64_t nh_kv,
                                                int64_t nq,
                                                int64_t D) {
    const size_t headroom = 512ULL * 1024 * 1024;
    const size_t usable = (free_bytes > headroom) ? (free_bytes - headroom) : 0;
    const size_t per_chunk_token = (size_t)(nh_q * nq + 2 * D * nh_kv) * sizeof(float);
    if (per_chunk_token == 0) return CHUNKED_PF_MIN;
    size_t max_chunk = usable / per_chunk_token;
    int chunk = CHUNKED_PF_MAX;
    while (chunk > CHUNKED_PF_MIN && (size_t)chunk > max_chunk) {
        chunk >>= 1;
    }
    return chunk;
}

// Next power of 2 >= n (n >= 1).
static inline int chunked_pf_next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Kernel 1: Initialize accumulators.
// O_acc = 0, l_acc = 0, m_acc = -inf
// Grid: (nq_heads, 1, 1), blockDim.x = min(D, 1024).
static __global__ void k_chunked_attn_init(
        float * __restrict__ O_acc,
        float * __restrict__ l_acc,
        float * __restrict__ m_acc,
        const int64_t nq_heads,
        const int64_t D) {
    const int64_t hq  = (int64_t)blockIdx.x;
    const int     tid = (int)threadIdx.x;
    const int     bdx = (int)blockDim.x;
    if (hq >= nq_heads) return;
    if (tid == 0) {
        l_acc[hq] = 0.0f;
        m_acc[hq] = -INFINITY;
    }
    for (int64_t d = tid; d < D; d += bdx) {
        O_acc[hq * D + d] = 0.0f;
    }
}

// Kernel 2: Online softmax update.
// Processes one (head, query) pair per thread block.
// Shared memory layout: sm[0..chunk_pad-1] scores/exp, sm[chunk_pad..chunk_pad+1] alpha/beta.
// blockDim.x = min(chunk_pad, 1024). Binary-tree max-then-sum reduction.
// Rescales O_acc by alpha, writes P = beta * exp(S-m_chunk) back into S for the P@V GEMM.
static __global__ void k_chunked_softmax_update(
        float * __restrict__ S,          // [nh_q, q_batch, s_stride] — scores → P after kernel
        float * __restrict__ O_acc,      // [nh_q, nq_total, D]
        float * __restrict__ l_acc,      // [nh_q, nq_total]
        float * __restrict__ m_acc,      // [nh_q, nq_total]
        const int chunk_len,
        const int chunk_pad,
        const int64_t D,
        const int64_t nq_total,
        const int64_t q_batch,
        const int64_t q_start,
        const int64_t nh_q,
        const half  * __restrict__ mask, // [nq_total, nkv_total] f16 mask, or nullptr
        const int64_t mask_stride,
        const int64_t kv_start,
        const int s_stride) {
    const int64_t hq_idx  = (int64_t)blockIdx.x;
    const int64_t head    = hq_idx / q_batch;
    const int64_t q_local = hq_idx % q_batch;
    const int64_t q_pos   = q_start + q_local;
    if (head >= nh_q) return;

    const int tid = (int)threadIdx.x;
    const int bdx = (int)blockDim.x;
    extern __shared__ float sm[];  // (chunk_pad + 2) floats

    const int64_t s_base  = head * q_batch * s_stride + q_local * s_stride;
    const int64_t acc_idx = head * nq_total + q_pos;

    // Load scores + mask into sm[], pad with -inf.
    for (int c = tid; c < chunk_pad; c += bdx) {
        if (c < chunk_len) {
            float val = S[s_base + c];
            if (mask != nullptr) {
                val += __half2float(mask[q_pos * mask_stride + kv_start + c]);
            }
            sm[c] = val;
        } else {
            sm[c] = -INFINITY;
        }
    }
    __syncthreads();

    // Binary tree max reduction.
    for (int stride = chunk_pad >> 1; stride >= 1; stride >>= 1) {
        for (int c = tid; c < stride; c += bdx) {
            sm[c] = fmaxf(sm[c], sm[c + stride]);
        }
        __syncthreads();
    }
    const float m_chunk = sm[0];
    __syncthreads();
    const bool chunk_empty = (m_chunk == -INFINITY);

    // Compute exp(score - m_chunk), pad with 0.
    for (int c = tid; c < chunk_pad; c += bdx) {
        if (c < chunk_len) {
            float val = S[s_base + c];
            if (mask != nullptr) {
                val += __half2float(mask[q_pos * mask_stride + kv_start + c]);
            }
            sm[c] = __expf(val - m_chunk);
        } else {
            sm[c] = 0.0f;
        }
    }
    __syncthreads();

    // Binary tree sum reduction.
    for (int stride = chunk_pad >> 1; stride >= 1; stride >>= 1) {
        for (int c = tid; c < stride; c += bdx) {
            sm[c] += sm[c + stride];
        }
        __syncthreads();
    }
    const float l_chunk = sm[0];
    __syncthreads();

    // Update m_acc, l_acc and broadcast alpha/beta via sm[chunk_pad..+1].
    if (tid == 0) {
        if (!chunk_empty) {
            const float m_old = m_acc[acc_idx];
            const float m_new = fmaxf(m_old, m_chunk);
            const float alpha = (m_old > -INFINITY) ? __expf(m_old - m_new) : 0.0f;
            const float beta  = __expf(m_chunk - m_new);
            sm[chunk_pad]     = alpha;
            sm[chunk_pad + 1] = beta;
            l_acc[acc_idx] = alpha * l_acc[acc_idx] + beta * l_chunk;
            m_acc[acc_idx] = m_new;
        } else {
            sm[chunk_pad]     = 1.0f;
            sm[chunk_pad + 1] = 0.0f;
        }
    }
    __syncthreads();

    const float alpha = sm[chunk_pad];
    const float beta  = sm[chunk_pad + 1];

    // Rescale O_acc.
    for (int64_t d = tid; d < D; d += bdx) {
        O_acc[acc_idx * D + d] *= alpha;
    }
    __syncthreads();

    // Write P = beta * exp(S - m_chunk) back to S; zero padding.
    for (int c = tid; c < chunk_len; c += bdx) {
        if (chunk_empty) {
            S[s_base + c] = 0.0f;
        } else {
            float val = S[s_base + c];
            if (mask != nullptr) {
                val += __half2float(mask[q_pos * mask_stride + kv_start + c]);
            }
            S[s_base + c] = beta * __expf(val - m_chunk);
        }
    }
    for (int c = tid + chunk_len; c < s_stride; c += bdx) {
        S[s_base + c] = 0.0f;
    }
}

// Kernel 3: Finalize. Reads head-major O_acc, writes token-major dst / l.
static __global__ void k_chunked_attn_finalize(
        const float * __restrict__ O_acc, // [nh_q, nq, D] head-major
        const float * __restrict__ l_acc, // [nh_q * nq]
        float       * __restrict__ dst,   // ggml layout [D, nh_q, nq] token-major
        const int64_t nq,
        const int64_t nh_q,
        const int64_t D) {
    const int64_t hq_idx = (int64_t)blockIdx.x;
    const int64_t head   = hq_idx / nq;
    const int64_t q_pos  = hq_idx % nq;
    const int64_t d      = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;

    const float l = fmaxf(l_acc[hq_idx], 1e-30f);
    dst[q_pos * nh_q * D + head * D + d] = O_acc[hq_idx * D + d] / l;
}

// Apply TQ3 FWHT rotation in-place to a flat [nh, nq, D] fp32 buffer.
// One warp (32 threads) per (head, token, group-of-128) tuple.
// direction == 0 → forward, 1 → inverse.
static __global__ void k_tq3_rotate_inplace_f32(
        float * __restrict__ data,
        int nh, int nq, int D,
        int direction)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int g = blockIdx.z;
    if (t >= nq || h >= nh) return;
    float * row = data + (h * nq + t) * D + g * 128;
    const int lane = threadIdx.x & 31;
    float v0 = row[lane * 4 + 0];
    float v1 = row[lane * 4 + 1];
    float v2 = row[lane * 4 + 2];
    float v3 = row[lane * 4 + 3];
    if (direction == 0) {
        warp_tq3_rotate_forward(v0, v1, v2, v3);
    } else {
        warp_tq3_rotate_inverse(v0, v1, v2, v3);
    }
    row[lane * 4 + 0] = v0;
    row[lane * 4 + 1] = v1;
    row[lane * 4 + 2] = v2;
    row[lane * 4 + 3] = v3;
}

// ─── Strided chunk dequant kernels ─────────────────────────────────────────
//
// Extract a [chunk_len, nh_kv] window from a ggml KV tensor (layout
// [D, kv_len, nh_kv] with byte strides nb1, nb2) into a contiguous
// [nh_kv, chunk_len, D] FP32 destination. Each thread block handles one
// (kv_local, head) cell; threadIdx.x cooperates across the head_dim.
//
// Grid: (chunk_len, nh_kv, 1). blockDim.x = up to D (capped to 256).
//
// This mirrors turboquant's k_tbq*_dequant_f32 pattern but for the native
// ggml block types (Q4_0, Q8_0) and plain half/bfloat16 layouts — nothing
// compressed-domain, no SRHT.

static __global__ void k_chunked_dequant_f16_f32(
        const char * __restrict__ src_base,
        float      * __restrict__ dst,
        const int64_t D,
        const int64_t chunk_len,
        const size_t  nb1,
        const size_t  nb2,
        const int64_t kv_start) {
    const int64_t kv_local = blockIdx.x;
    const int64_t h        = blockIdx.y;
    if (kv_local >= chunk_len) return;
    const half * src = (const half *)(src_base + h * nb2 + (kv_start + kv_local) * nb1);
    float      * out = dst + (h * chunk_len + kv_local) * D;
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        out[d] = __half2float(src[d]);
    }
}

static __global__ void k_chunked_dequant_bf16_f32(
        const char * __restrict__ src_base,
        float      * __restrict__ dst,
        const int64_t D,
        const int64_t chunk_len,
        const size_t  nb1,
        const size_t  nb2,
        const int64_t kv_start) {
    const int64_t kv_local = blockIdx.x;
    const int64_t h        = blockIdx.y;
    if (kv_local >= chunk_len) return;
    const nv_bfloat16 * src = (const nv_bfloat16 *)(src_base + h * nb2 + (kv_start + kv_local) * nb1);
    float             * out = dst + (h * chunk_len + kv_local) * D;
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        out[d] = __bfloat162float(src[d]);
    }
}

// Q4_0: one token has D/QK4_0 blocks; each block has a half scale + 16 bytes
// of packed 4-bit quants (32 elements / block). Per-thread we dequant one
// whole block to amortize the scale load.
static __global__ void k_chunked_dequant_q4_0_f32(
        const char * __restrict__ src_base,
        float      * __restrict__ dst,
        const int64_t D,
        const int64_t chunk_len,
        const size_t  nb1,
        const size_t  nb2,
        const int64_t kv_start) {
    const int64_t kv_local = blockIdx.x;
    const int64_t h        = blockIdx.y;
    if (kv_local >= chunk_len) return;
    const char * src_token = src_base + h * nb2 + (kv_start + kv_local) * nb1;
    float      * out       = dst + (h * chunk_len + kv_local) * D;
    const int n_blocks = D / QK4_0;
    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        const block_q4_0 * blk = (const block_q4_0 *)(src_token + b * sizeof(block_q4_0));
        const float scale = __half2float(blk->d);
        float * out_blk = out + b * QK4_0;
        #pragma unroll
        for (int i = 0; i < QK4_0 / 2; i++) {
            const uint8_t packed = blk->qs[i];
            const int lo = (int)(packed & 0x0F) - 8;
            const int hi = (int)(packed >> 4)   - 8;
            out_blk[i]             = lo * scale;
            out_blk[i + QK4_0 / 2] = hi * scale;
        }
    }
}

// Q8_0: one token has D/QK8_0 blocks; each block has a half scale + 32 int8
// quants (32 elements / block).
static __global__ void k_chunked_dequant_q8_0_f32(
        const char * __restrict__ src_base,
        float      * __restrict__ dst,
        const int64_t D,
        const int64_t chunk_len,
        const size_t  nb1,
        const size_t  nb2,
        const int64_t kv_start) {
    const int64_t kv_local = blockIdx.x;
    const int64_t h        = blockIdx.y;
    if (kv_local >= chunk_len) return;
    const char * src_token = src_base + h * nb2 + (kv_start + kv_local) * nb1;
    float      * out       = dst + (h * chunk_len + kv_local) * D;
    const int n_blocks = D / QK8_0;
    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        const block_q8_0 * blk = (const block_q8_0 *)(src_token + b * sizeof(block_q8_0));
        const float scale = __half2float(blk->d);
        float * out_blk = out + b * QK8_0;
        #pragma unroll
        for (int i = 0; i < QK8_0; i++) {
            out_blk[i] = ((int)blk->qs[i]) * scale;
        }
    }
}

// One thread per output element (not one thread per block).
//
// The previous version used `for (b = threadIdx.x; b < n_blocks; b += blockDim.x)`,
// which at the launch config (blockDim.x = min(D, 256), n_blocks = D/32) meant
// only 8 of 256 threads did real work for D=256 (3% block utilization). With
// thread-per-element, all 256 threads load and write coalesced.
//
// Output writes are 32-thread coalesced (one cache line per warp). Input reads
// hit the small TQ3 codebook in __constant__ memory — same-norm reads across
// the warp broadcast, centroid lookups serialize on different indices but the
// table is only 8 entries so the serial cost is at worst 8 cycles per warp.
static __global__ void k_chunked_dequant_tq3_0_f32(
        const char * __restrict__ src_base,
        float      * __restrict__ dst,
        const int64_t D,
        const int64_t chunk_len,
        const size_t  nb1,
        const size_t  nb2,
        const int64_t kv_start) {
    const int64_t kv_local = blockIdx.x;
    const int64_t h        = blockIdx.y;
    if (kv_local >= chunk_len) return;
    const char * src_token = src_base + h * nb2 + (kv_start + kv_local) * nb1;
    float      * out       = dst + (h * chunk_len + kv_local) * D;

    for (int j = (int)threadIdx.x; j < (int)D; j += (int)blockDim.x) {
        const int b           = j >> 5;     // j / QK_TQ3_0
        const int idx_in_blk  = j & 31;     // j % QK_TQ3_0
        const block_tq3_0 * blk = (const block_tq3_0 *)(src_token + b * sizeof(block_tq3_0));
        const float norm = __half2float(blk->norm);
        const uint8_t low2 = (blk->qs   [idx_in_blk / 4] >> ((idx_in_blk % 4) * 2)) & 0x3;
        const uint8_t hi1  = (blk->signs[idx_in_blk / 8] >> ( idx_in_blk % 8))      & 0x1;
        out[j] = d_tq3_centroids[low2 | (hi1 << 2)] * norm;
    }
}

// Launch helper: dispatch by ggml type. Returns false if unsupported.
static inline bool chunked_dequant_launch(
        ggml_type    type,
        const char * src_base,
        float      * dst,
        int64_t      D,
        int64_t      chunk_len,
        int64_t      nh_kv,
        size_t       nb1,
        size_t       nb2,
        int64_t      kv_start,
        cudaStream_t stream) {
    const dim3 grid((int)chunk_len, (int)nh_kv, 1);
    const int threads = (int)std::min(D, (int64_t)256);
    switch (type) {
        case GGML_TYPE_F16:
            k_chunked_dequant_f16_f32<<<grid, threads, 0, stream>>>(
                src_base, dst, D, chunk_len, nb1, nb2, kv_start);
            return true;
        case GGML_TYPE_BF16:
            k_chunked_dequant_bf16_f32<<<grid, threads, 0, stream>>>(
                src_base, dst, D, chunk_len, nb1, nb2, kv_start);
            return true;
        case GGML_TYPE_Q4_0:
            k_chunked_dequant_q4_0_f32<<<grid, threads, 0, stream>>>(
                src_base, dst, D, chunk_len, nb1, nb2, kv_start);
            return true;
        case GGML_TYPE_Q8_0:
            k_chunked_dequant_q8_0_f32<<<grid, threads, 0, stream>>>(
                src_base, dst, D, chunk_len, nb1, nb2, kv_start);
            return true;
        case GGML_TYPE_TQ3_0:
            k_chunked_dequant_tq3_0_f32<<<grid, threads, 0, stream>>>(
                src_base, dst, D, chunk_len, nb1, nb2, kv_start);
            return true;
        default:
            return false;
    }
}

// Entry point (implemented in fattn-chunked.cu). Enabled via
// BEST_FATTN_KERNEL_CHUNKED from the main dispatcher.
void ggml_cuda_flash_attn_ext_chunked(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
