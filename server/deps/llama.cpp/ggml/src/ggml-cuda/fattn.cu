#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn-chunked.cuh"
#include "fattn.cuh"

#if defined(GGML_USE_HIP)

__device__ static float ds4_fa_block_sum(float v) {
    __shared__ float smem[256];
    const int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    return smem[0];
}

__device__ static float ds4_fa_block_max(float v) {
    __shared__ float smem[256];
    const int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
    return smem[0];
}

template <typename KV, typename Mask>
__device__ static __forceinline__ float ds4_fa_load(const KV * ptr) {
    return (float) *ptr;
}

template <>
__device__ __forceinline__ float ds4_fa_load<half, half>(const half * ptr) {
    return __half2float(*ptr);
}

template <typename KV>
__device__ static __forceinline__ void ds4_fa_load_pair(
        const KV * ptr, float & v0, float & v1) {
    v0 = (float) ptr[0];
    v1 = (float) ptr[1];
}

template <>
__device__ __forceinline__ void ds4_fa_load_pair<float>(
        const float * ptr, float & v0, float & v1) {
    const float2 pair = *reinterpret_cast<const float2 *>(ptr);
    v0 = pair.x;
    v1 = pair.y;
}

template <>
__device__ __forceinline__ void ds4_fa_load_pair<half>(
        const half * ptr, float & v0, float & v1) {
    const half2 pair = *reinterpret_cast<const half2 *>(ptr);
    const float2 unpacked = __half22float2(pair);
    v0 = unpacked.x;
    v1 = unpacked.y;
}

template <typename KV>
__device__ static __forceinline__ void ds4_fa_load_quad(
        const KV * ptr, float & v0, float & v1, float & v2, float & v3) {
    v0 = (float) ptr[0];
    v1 = (float) ptr[1];
    v2 = (float) ptr[2];
    v3 = (float) ptr[3];
}

template <>
__device__ __forceinline__ void ds4_fa_load_quad<float>(
        const float * ptr, float & v0, float & v1, float & v2, float & v3) {
    const float4 values = *reinterpret_cast<const float4 *>(ptr);
    v0 = values.x;
    v1 = values.y;
    v2 = values.z;
    v3 = values.w;
}

template <>
__device__ __forceinline__ void ds4_fa_load_quad<half>(
        const half * ptr, float & v0, float & v1, float & v2, float & v3) {
    const float2 lo = __half22float2(
        *reinterpret_cast<const half2 *>(ptr + 0));
    const float2 hi = __half22float2(
        *reinterpret_cast<const half2 *>(ptr + 2));
    v0 = lo.x;
    v1 = lo.y;
    v2 = hi.x;
    v3 = hi.y;
}

struct ds4_inverse_rope_params {
    int   enabled;
    int   forward_q_enabled;
    int   kv_start;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float corr_low;
    float corr_high;
    float theta_scale;
};

// Keep these expressions aligned with rope.cu. The attention result is first
// stored in shared F32, matching the standalone attention-output store/load
// boundary, before the pair is rotated.
__device__ static __forceinline__ float ds4_rope_theta_fp64(
        int32_t p, float theta_scale, int exp_int) {
    const double tau = 6.2831853071795864769;
    double angle = exp_int == 0
        ? (double) p
        : (double) p * pow((double) theta_scale, (double) exp_int);
    angle -= tau * floor(angle * (1.0 / tau));
    return (float) angle;
}

__device__ static __forceinline__ void ds4_inverse_rope_coefficients(
        int pair, int token,
        const ds4_inverse_rope_params & p,
        float & cos_theta, float & sin_theta) {
    const int i0 = 2 * pair;
    const float theta_extrap = ds4_rope_theta_fp64(
        -(p.kv_start + token), p.theta_scale, pair);
    const float theta_interp = p.freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = p.attn_factor;
    if (p.ext_factor != 0.0f) {
        const float ramp_y = (i0 / 2 - p.corr_low) /
            max(0.001f, p.corr_high - p.corr_low);
        const float ramp_mix =
            (1.0f - min(1.0f, max(0.0f, ramp_y))) * p.ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) +
                theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / p.freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
}

// Forward counterpart of ds4_inverse_rope_coefficients. Keep the expressions
// aligned with rope_norm<true> in rope.cu; unlike the inverse path, position is
// positive. Compressed-layer YaRN interpolation means the inverse coefficients
// cannot safely be recovered by merely negating sin(theta).
__device__ static __forceinline__ void ds4_forward_rope_coefficients(
        int pair, int token,
        const ds4_inverse_rope_params & p,
        float & cos_theta, float & sin_theta) {
    const int i0 = 2 * pair;
    const float theta_extrap = ds4_rope_theta_fp64(
        p.kv_start + token, p.theta_scale, pair);
    const float theta_interp = p.freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = p.attn_factor;
    if (p.ext_factor != 0.0f) {
        const float ramp_y = (i0 / 2 - p.corr_low) /
            max(0.001f, p.corr_high - p.corr_low);
        const float ramp_mix =
            (1.0f - min(1.0f, max(0.0f, ramp_y))) * p.ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) +
                theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / p.freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
}

__device__ static __forceinline__ void ds4_apply_inverse_rope_pair(
        float x0, float x1, float cos_theta, float sin_theta,
        float & y0, float & y1) {
    y0 = x0 * cos_theta - x1 * sin_theta;
    y1 = x0 * sin_theta + x1 * cos_theta;
}

// RoPE coefficients depend on token position and pair, not on the query head.
// Materialize them once per attention call instead of recomputing FP64 pow,
// floor, cos and sin in every head block. F32 storage preserves the same
// explicit coefficient rounding used by the original in-kernel calculation.
__global__ static void ds4_inverse_rope_coefficients_kernel(
        float * coefficients,
        int n_tokens,
        ds4_inverse_rope_params inverse_rope) {
    const int index = (int) blockIdx.x * (int) blockDim.x +
                      (int) threadIdx.x;
    const int count = n_tokens * 32;
    if (index >= count) return;
    const int token = index / 32;
    const int pair = index % 32;
    float cos_theta;
    float sin_theta;
    ds4_inverse_rope_coefficients(
        pair, token, inverse_rope, cos_theta, sin_theta);
    coefficients[2 * index + 0] = cos_theta;
    coefficients[2 * index + 1] = sin_theta;
}

__global__ static void ds4_forward_rope_coefficients_kernel(
        float * coefficients,
        int n_tokens,
        ds4_inverse_rope_params rope) {
    const int index = (int) blockIdx.x * (int) blockDim.x +
                      (int) threadIdx.x;
    const int count = n_tokens * 32;
    if (index >= count) return;
    const int token = index / 32;
    const int pair = index % 32;
    float cos_theta;
    float sin_theta;
    ds4_forward_rope_coefficients(
        pair, token, rope, cos_theta, sin_theta);
    coefficients[2 * index + 0] = cos_theta;
    coefficients[2 * index + 1] = sin_theta;
}

// One mean latent-key vector per compressed-cache block. Raw SWA/current rows
// deliberately stay outside this summary and are always evaluated exactly.
template <typename KV>
__global__ static void ds4_fa_mean_comp_blocks_kernel(
        const KV * k,
        float    * mean_k,
        int        n_kv,
        int        raw_rows,
        int        block_size,
        int        n_blocks) {
    constexpr int D = 512;
    const int b = (int) blockIdx.x;
    if (b >= n_blocks) return;
    const int begin = raw_rows + b * block_size;
    const int end = min(n_kv, begin + block_size);
    const float inv = 1.0f / (float) max(1, end - begin);
    for (int d = (int) threadIdx.x; d < D; d += (int) blockDim.x) {
        float sum = 0.0f;
        for (int r = begin; r < end; ++r) {
            sum += ds4_fa_load<KV, KV>(k + (size_t) r * D + d);
        }
        mean_k[(size_t) b * D + d] = sum * inv;
    }
}

// Find the visible envelope in the raw and non-raw regions once per query
// token. Attention blocks for all query-head groups reuse these four bounds.
// Internal masked rows remain inside the envelope and are still evaluated as
// masked, so this changes storage only, not attention semantics.
template <typename Mask>
__global__ static void ds4_fa_visibility_bounds_kernel(
        const Mask * mask,
        int        * bounds,
        int          n_tokens,
        int          n_kv,
        int          raw_rows) {
    const int t = (int) blockIdx.x;
    const int lane = (int) threadIdx.x;
    if (t >= n_tokens || lane >= warpSize) return;

    int raw_first = raw_rows;
    int raw_last = -1;
    int comp_first = n_kv;
    int comp_last = -1;
    const Mask * token_mask = mask + (size_t) t * n_kv;

    for (int base = 0; base < raw_rows; base += warpSize) {
        const int r = base + lane;
        const unsigned long long active = __ballot(
            r < raw_rows &&
            ds4_fa_load<Mask, Mask>(token_mask + r) > -1.0e20f);
        if (lane == 0 && active != 0) {
            if (raw_first == raw_rows) {
                raw_first = base + __ffsll(active) - 1;
            }
            raw_last = base + 63 - __clzll(active);
        }
    }
    for (int base = raw_rows; base < n_kv; base += warpSize) {
        const int r = base + lane;
        const unsigned long long active = __ballot(
            r < n_kv &&
            ds4_fa_load<Mask, Mask>(token_mask + r) > -1.0e20f);
        if (lane == 0 && active != 0) {
            if (comp_first == n_kv) {
                comp_first = base + __ffsll(active) - 1;
            }
            comp_last = base + 63 - __clzll(active);
        }
    }
    if (lane == 0) {
        int * token_bounds = bounds + (size_t) t * 4;
        token_bounds[0] = raw_first;
        token_bounds[1] = raw_last;
        token_bounds[2] = comp_first;
        token_bounds[3] = comp_last;
    }
}

// Convert an externally selected compressed-row mask into exact lookup tables.
// selected_rows preserves ascending physical-row order for the value pass.
// owner_offsets/owner_ranks group those ascending ranks by the thread that
// owned the physical row in the original r = tid + 256*k traversal. The hot
// score and softmax passes can therefore visit only selected rows while
// retaining every thread's original accumulation order and reduction leaf.
template <typename Mask>
__global__ static void ds4_fa_indexed_rows_kernel(
        const Mask * mask,
        int        * selected_rows,
        int        * selected_counts,
        int        * owner_offsets,
        int        * owner_ranks,
        int          n_tokens,
        int          n_kv,
        int          raw_rows,
        int          capacity) {
    const int t = (int) blockIdx.x;
    const int tid = (int) threadIdx.x;
    if (t >= n_tokens) return;

    constexpr int N_THREADS = 256;
    __shared__ int owner_write[N_THREADS];
    const int n_comp_rows = n_kv - raw_rows;
    int * token_rows = selected_rows + (size_t) t * capacity;
    int * token_owner_offsets = owner_offsets + (size_t) t * (N_THREADS + 1);
    int * token_owner_ranks = owner_ranks + (size_t) t * capacity;
    token_owner_offsets[tid] = 0;
    if (tid == 0) token_owner_offsets[N_THREADS] = 0;
    __syncthreads();

    if (tid == 0) {
        const Mask * token_mask = mask + (size_t) t * n_kv;
        int count = 0;
        for (int c = 0; c < n_comp_rows; ++c) {
            if (ds4_fa_load<Mask, Mask>(token_mask + raw_rows + c) <= -1.0e20f) {
                continue;
            }
            if (count < capacity) {
                const int r = raw_rows + c;
                token_rows[count] = r;
                ++token_owner_offsets[(r & (N_THREADS - 1)) + 1];
            }
            ++count;
        }
        count = min(count, capacity);
        selected_counts[t] = count;

        int prefix = 0;
        for (int owner = 0; owner < N_THREADS; ++owner) {
            const int owner_count = token_owner_offsets[owner + 1];
            token_owner_offsets[owner] = prefix;
            prefix += owner_count;
        }
        token_owner_offsets[N_THREADS] = prefix;
    }
    __syncthreads();

    owner_write[tid] = token_owner_offsets[tid];
    __syncthreads();

    if (tid == 0) {
        const int count = selected_counts[t];
        for (int rank = 0; rank < count; ++rank) {
            const int owner = token_rows[rank] & (N_THREADS - 1);
            token_owner_ranks[owner_write[owner]++] = rank;
        }
    }
}

template <typename KV, typename Mask>
__global__ static void ds4_flash_attn_d512_shared_kv_kernel(
        float       * dst,
        const float * q,
        size_t        q_stride_token,
        size_t        q_stride_head,
        const KV    * k,
        const KV    * v,
        const Mask  * mask,
        const float * sinks,
        const float * mean_k,
        int           n_tokens,
        int           n_heads,
        int           n_kv,
        float         scale,
        int           raw_rows,
        int           sparse_keep_rows,
        int           sparse_block_size,
        int           n_comp_blocks,
        ds4_inverse_rope_params inverse_rope,
        const float * inverse_rope_coefficients,
        const float * forward_rope_coefficients) {
    constexpr int D = 512;
    const int t = (int) blockIdx.x;
    const int h = (int) blockIdx.y;
    const int tid = (int) threadIdx.x;
    if (t >= n_tokens || h >= n_heads) return;

    extern __shared__ float scratch[];
    float * scores = scratch;
    float * block_scores = scores + n_kv;
    float * block_keep = block_scores + n_comp_blocks;
    float * q_rope_tail = block_keep + n_comp_blocks;
    const float * qh = q + (size_t) t * q_stride_token +
                       (size_t) h * q_stride_head;

    if (inverse_rope.forward_q_enabled) {
        for (int pair = tid; pair < 32; pair += (int) blockDim.x) {
            const float x0 = qh[D - 64 + 2 * pair + 0];
            const float x1 = qh[D - 64 + 2 * pair + 1];
            const size_t coefficient_index =
                ((size_t) t * 32 + (size_t) pair) * 2;
            const float cos_theta = forward_rope_coefficients[
                coefficient_index + 0];
            const float sin_theta = forward_rope_coefficients[
                coefficient_index + 1];
            ds4_apply_inverse_rope_pair(
                x0, x1, cos_theta, sin_theta,
                q_rope_tail[2 * pair + 0], q_rope_tail[2 * pair + 1]);
        }
        __syncthreads();
    }

    const int n_comp_rows = n_kv - raw_rows;
    const bool sparse = mean_k && n_comp_blocks > 0 &&
                        sparse_keep_rows > 0 &&
                        sparse_keep_rows < n_comp_rows;
    if (sparse) {
        for (int b = tid; b < n_comp_blocks; b += (int) blockDim.x) {
            const int first_row = raw_rows + b * sparse_block_size;
            const float mask_v = mask
                ? ds4_fa_load<Mask, Mask>(
                    mask + (size_t) t * n_kv + first_row)
                : 0.0f;
            const float * kb = mean_k + (size_t) b * D;
            float dot = -3.402823466e38f;
            if (mask_v > -1.0e20f) {
                dot = 0.0f;
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    const float qv = inverse_rope.forward_q_enabled && d >= D - 64
                        ? q_rope_tail[d - (D - 64)] : qh[d];
                    dot += qv * kb[d];
                }
                dot *= scale;
            }
            block_scores[b] = dot;
        }
        __syncthreads();

        const int keep_blocks = min(n_comp_blocks,
            (sparse_keep_rows + sparse_block_size - 1) / sparse_block_size);
        for (int b = tid; b < n_comp_blocks; b += (int) blockDim.x) {
            const float score = block_scores[b];
            int rank = 0;
            for (int j = 0; j < n_comp_blocks; ++j) {
                const float other = block_scores[j];
                rank += (other > score || (other == score && j < b)) ? 1 : 0;
            }
            block_keep[b] = rank < keep_blocks ? 1.0f : 0.0f;
        }
        __syncthreads();
    }

    // Keep the original row-to-thread mapping and reduction order. The mask
    // remains the sole visibility authority for dense attention; duplicating
    // its DS4 policy here risks changing semantics. Masked rows already avoid
    // the expensive D=512 dot product. Sparse mode additionally drops selected
    // compressed blocks by design.
    float local_max = sinks ? sinks[h] : -3.402823466e38f;
    for (int r = tid; r < n_kv; r += blockDim.x) {
        bool keep = true;
        if (sparse && r >= raw_rows) {
            const int b = (r - raw_rows) / sparse_block_size;
            keep = b < n_comp_blocks && block_keep[b] != 0.0f;
        }
        const float mask_v = mask
            ? ds4_fa_load<Mask, Mask>(mask + (size_t) t * n_kv + r)
            : 0.0f;
        float s = -3.402823466e38f;
        if (keep && mask_v > -1.0e20f) {
            const KV * kr = k + (size_t) r * D;
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < D; ++d) {
                const float qv = inverse_rope.forward_q_enabled && d >= D - 64
                    ? q_rope_tail[d - (D - 64)] : qh[d];
                dot += qv * ds4_fa_load<KV, Mask>(kr + d);
            }
            s = dot * scale + mask_v;
        }
        scores[r] = s;
        local_max = fmaxf(local_max, s);
    }
    const float max_score = ds4_fa_block_max(local_max);

    float local_sum = 0.0f;
    for (int r = tid; r < n_kv; r += blockDim.x) {
        const float w = expf(scores[r] - max_score);
        scores[r] = w;
        local_sum += w;
    }
    if (tid == 0 && sinks) {
        local_sum += expf(sinks[h] - max_score);
    }
    const float denom = ds4_fa_block_sum(local_sum);
    const float inv_denom = 1.0f / denom;

    // One wave locates the non-zero envelope independently in the raw and
    // compressed spans. Ballots inspect a whole wave of scores at once and
    // require only one block barrier, unlike a four-reduction implementation.
    // Values inside each envelope retain their original order, including any
    // internal zero, so floating-point accumulation is unchanged.
    __shared__ int value_bounds[4];
    if (tid < warpSize) {
        const int lane = tid;
        if (lane == 0) {
            value_bounds[0] = raw_rows;
            value_bounds[1] = -1;
            value_bounds[2] = n_kv;
            value_bounds[3] = -1;
        }
        for (int base = 0; base < raw_rows; base += warpSize) {
            const int r = base + lane;
            const unsigned long long active = __ballot(
                r < raw_rows && scores[r] != 0.0f);
            if (lane == 0 && active != 0) {
                if (value_bounds[0] == raw_rows) {
                    value_bounds[0] = base + __ffsll(active) - 1;
                }
                value_bounds[1] = base + 63 - __clzll(active);
            }
        }
        for (int base = raw_rows; base < n_kv; base += warpSize) {
            const int r = base + lane;
            const unsigned long long active = __ballot(
                r < n_kv && scores[r] != 0.0f);
            if (lane == 0 && active != 0) {
                if (value_bounds[2] == n_kv) {
                    value_bounds[2] = base + __ffsll(active) - 1;
                }
                value_bounds[3] = base + 63 - __clzll(active);
            }
        }
    }
    __syncthreads();
    const int raw_first = value_bounds[0];
    const int raw_last = value_bounds[1];
    const int comp_first = value_bounds[2];
    const int comp_last = value_bounds[3];

    // Scoring is complete here, so the forward-RoPE tail scratch can be
    // reused for inverse-RoPE output. Do not alias scores: different waves
    // accumulate value dimensions concurrently, and tail writers would race
    // readers of scores[0..63].
    float * rope_tail = q_rope_tail;
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int r = raw_first; r <= raw_last; ++r) {
            acc += scores[r] * ds4_fa_load<KV, Mask>(
                v + (size_t) r * D + d);
        }
        for (int r = comp_first; r <= comp_last; ++r) {
            acc += scores[r] * ds4_fa_load<KV, Mask>(
                v + (size_t) r * D + d);
        }
        const float value = acc * inv_denom;
        if (inverse_rope.enabled && d >= D - 64) {
            rope_tail[d - (D - 64)] = value;
        } else {
            dst[((size_t) t * (size_t) n_heads + (size_t) h) * D + d] = value;
        }
    }
    if (inverse_rope.enabled) {
        __syncthreads();
        if (tid < 32) {
            const float x0 = rope_tail[2 * tid + 0];
            const float x1 = rope_tail[2 * tid + 1];
            const size_t coefficient_index =
                ((size_t) t * 32 + (size_t) tid) * 2;
            const float cos_theta = inverse_rope_coefficients[
                coefficient_index + 0];
            const float sin_theta = inverse_rope_coefficients[
                coefficient_index + 1];
            float y0;
            float y1;
            ds4_apply_inverse_rope_pair(
                x0, x1, cos_theta, sin_theta, y0, y1);
            float * out = dst +
                ((size_t) t * (size_t) n_heads + (size_t) h) * D + D - 64;
            out[2 * tid + 0] = y0;
            out[2 * tid + 1] = y1;
        }
    }
}

// DS4 MLA uses one latent K/V head for every query head. Grouping query heads
// in one block lets them share each K/V load while retaining the reference
// kernel's row-to-thread mapping, per-head reduction tree, and accumulation
// order. The grouped path is dense-only; experimental sparse selection keeps
// using the single-head kernel above.
template <typename KV, typename Mask, int HEADS_PER_BLOCK>
__global__ static void ds4_flash_attn_d512_shared_kv_grouped_kernel(
        float       * dst,
        const float * q,
        size_t        q_stride_token,
        size_t        q_stride_head,
        const KV    * k,
        const KV    * v,
        const Mask  * mask,
        const float * sinks,
        int           n_tokens,
        int           n_heads,
        int           n_kv,
        float         scale,
        int           raw_rows,
        ds4_inverse_rope_params inverse_rope,
        const float * inverse_rope_coefficients,
        const float * forward_rope_coefficients) {
    constexpr int D = 512;
    constexpr int N_THREADS = 256;
    const int t = (int) blockIdx.x;
    const int h_begin = (int) blockIdx.y * HEADS_PER_BLOCK;
    const int tid = (int) threadIdx.x;
    if (t >= n_tokens || h_begin >= n_heads) return;

    extern __shared__ float scratch[];
    float * scores = scratch;
    float * reduction = scores + (size_t) HEADS_PER_BLOCK * n_kv;
    int * value_bounds = reinterpret_cast<int *>(
        reduction + (size_t) HEADS_PER_BLOCK * N_THREADS);
    float * q_rope_tail = reinterpret_cast<float *>(
        value_bounds + (size_t) HEADS_PER_BLOCK * 4);

    const float * qh[HEADS_PER_BLOCK];
#pragma unroll
    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
        const int h = h_begin + j;
        qh[j] = q + (size_t) t * q_stride_token +
                (size_t) h * q_stride_head;
    }

    if (inverse_rope.forward_q_enabled) {
        for (int index = tid; index < HEADS_PER_BLOCK * 32;
             index += (int) blockDim.x) {
            const int j = index / 32;
            const int pair = index % 32;
            const float x0 = qh[j][D - 64 + 2 * pair + 0];
            const float x1 = qh[j][D - 64 + 2 * pair + 1];
            const size_t coefficient_index =
                ((size_t) t * 32 + (size_t) pair) * 2;
            const float cos_theta = forward_rope_coefficients[
                coefficient_index + 0];
            const float sin_theta = forward_rope_coefficients[
                coefficient_index + 1];
            ds4_apply_inverse_rope_pair(
                x0, x1, cos_theta, sin_theta,
                q_rope_tail[(size_t) j * 64 + 2 * pair + 0],
                q_rope_tail[(size_t) j * 64 + 2 * pair + 1]);
        }
        __syncthreads();
    }

    float local_max[HEADS_PER_BLOCK];
#pragma unroll
    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
        const int h = h_begin + j;
        local_max[j] = h < n_heads && sinks
            ? sinks[h] : -3.402823466e38f;
    }
    // A thread owns exactly the same rows as in the single-head kernel. Four
    // independent dot-product chains consume one shared K value per feature.
    for (int r = tid; r < n_kv; r += blockDim.x) {
        const float mask_v = mask
            ? ds4_fa_load<Mask, Mask>(mask + (size_t) t * n_kv + r)
            : 0.0f;
        const bool visible = mask_v > -1.0e20f;
        float dot[HEADS_PER_BLOCK] = {};
        if (visible) {
            const KV * kr = k + (size_t) r * D;
#pragma unroll
            for (int d = 0; d < D; ++d) {
                const float kv = ds4_fa_load<KV, Mask>(kr + d);
#pragma unroll
                for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
                    const float qv =
                        inverse_rope.forward_q_enabled && d >= D - 64
                            ? q_rope_tail[(size_t) j * 64 + d - (D - 64)]
                            : qh[j][d];
                    dot[j] += qv * kv;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
            const int h = h_begin + j;
            const float s = h < n_heads && visible
                ? dot[j] * scale + mask_v : -3.402823466e38f;
            scores[(size_t) j * n_kv + r] = s;
            local_max[j] = fmaxf(local_max[j], s);
        }
    }

    // Match ds4_fa_block_max independently for every grouped head.
#pragma unroll
    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
        reduction[(size_t) j * N_THREADS + tid] = local_max[j];
    }
    __syncthreads();
    for (int stride = N_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
                float * row = reduction + (size_t) j * N_THREADS;
                row[tid] = fmaxf(row[tid], row[tid + stride]);
            }
        }
        __syncthreads();
    }

    float max_score[HEADS_PER_BLOCK];
    float local_sum[HEADS_PER_BLOCK] = {};
#pragma unroll
    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
        max_score[j] = reduction[(size_t) j * N_THREADS];
    }
    for (int r = tid; r < n_kv; r += blockDim.x) {
#pragma unroll
        for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
            float * score = scores + (size_t) j * n_kv + r;
            const float weight = expf(*score - max_score[j]);
            *score = weight;
            local_sum[j] += weight;
        }
    }
    if (tid == 0 && sinks) {
#pragma unroll
        for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
            local_sum[j] += expf(sinks[h_begin + j] - max_score[j]);
        }
    }

    // Match ds4_fa_block_sum independently for every grouped head.
#pragma unroll
    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
        reduction[(size_t) j * N_THREADS + tid] = local_sum[j];
    }
    __syncthreads();
    for (int stride = N_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
                float * row = reduction + (size_t) j * N_THREADS;
                row[tid] += row[tid + stride];
            }
        }
        __syncthreads();
    }

    float inv_denom[HEADS_PER_BLOCK];
#pragma unroll
    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
        inv_denom[j] = 1.0f / reduction[(size_t) j * N_THREADS];
    }

    // Underflow can make the non-zero envelope differ by head, so retain one
    // pair of raw/compressed bounds per head. Ballot order does not affect any
    // arithmetic result.
    if (tid < warpSize) {
        const int lane = tid;
#pragma unroll
        for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
            int * bounds = value_bounds + 4 * j;
            if (lane == 0) {
                bounds[0] = raw_rows;
                bounds[1] = -1;
                bounds[2] = n_kv;
                bounds[3] = -1;
            }
            for (int base = 0; base < raw_rows; base += warpSize) {
                const int r = base + lane;
                const unsigned long long active = __ballot(
                    r < raw_rows &&
                    scores[(size_t) j * n_kv + r] != 0.0f);
                if (lane == 0 && active != 0) {
                    if (bounds[0] == raw_rows) {
                        bounds[0] = base + __ffsll(active) - 1;
                    }
                    bounds[1] = base + 63 - __clzll(active);
                }
            }
            for (int base = raw_rows; base < n_kv; base += warpSize) {
                const int r = base + lane;
                const unsigned long long active = __ballot(
                    r < n_kv && scores[(size_t) j * n_kv + r] != 0.0f);
                if (lane == 0 && active != 0) {
                    if (bounds[2] == n_kv) {
                        bounds[2] = base + __ffsll(active) - 1;
                    }
                    bounds[3] = base + 63 - __clzll(active);
                }
            }
        }
    }
    __syncthreads();

    int raw_first = raw_rows;
    int raw_last = -1;
    int comp_first = n_kv;
    int comp_last = -1;
#pragma unroll
    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
        const int * bounds = value_bounds + 4 * j;
        raw_first = min(raw_first, bounds[0]);
        raw_last = max(raw_last, bounds[1]);
        comp_first = min(comp_first, bounds[2]);
        comp_last = max(comp_last, bounds[3]);
    }

    float * rope_tail = reduction;
    for (int d = tid; d < D; d += blockDim.x) {
        float acc[HEADS_PER_BLOCK] = {};
        for (int r = raw_first; r <= raw_last; ++r) {
            const float vv = ds4_fa_load<KV, Mask>(v + (size_t) r * D + d);
#pragma unroll
            for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
                const int * bounds = value_bounds + 4 * j;
                if (r >= bounds[0] && r <= bounds[1]) {
                    acc[j] += scores[(size_t) j * n_kv + r] * vv;
                }
            }
        }
        for (int r = comp_first; r <= comp_last; ++r) {
            const float vv = ds4_fa_load<KV, Mask>(v + (size_t) r * D + d);
#pragma unroll
            for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
                const int * bounds = value_bounds + 4 * j;
                if (r >= bounds[2] && r <= bounds[3]) {
                    acc[j] += scores[(size_t) j * n_kv + r] * vv;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
            const int h = h_begin + j;
            if (h < n_heads) {
                const float value = acc[j] * inv_denom[j];
                if (inverse_rope.enabled && d >= D - 64) {
                    rope_tail[(size_t) j * 64 + d - (D - 64)] = value;
                } else {
                    dst[((size_t) t * (size_t) n_heads + (size_t) h) * D + d] =
                        value;
                }
            }
        }
    }
    if (inverse_rope.enabled) {
        __syncthreads();
        if (tid < HEADS_PER_BLOCK * 32) {
            const int j = tid / 32;
            const int pair = tid % 32;
            const float x0 = rope_tail[(size_t) j * 64 + 2 * pair + 0];
            const float x1 = rope_tail[(size_t) j * 64 + 2 * pair + 1];
            const size_t coefficient_index =
                ((size_t) t * 32 + (size_t) pair) * 2;
            const float cos_theta = inverse_rope_coefficients[
                coefficient_index + 0];
            const float sin_theta = inverse_rope_coefficients[
                coefficient_index + 1];
            float y0;
            float y1;
            ds4_apply_inverse_rope_pair(
                x0, x1, cos_theta, sin_theta, y0, y1);
            const int h = h_begin + j;
            float * out = dst +
                ((size_t) t * (size_t) n_heads + (size_t) h) * D + D - 64;
            out[2 * pair + 0] = y0;
            out[2 * pair + 1] = y1;
        }
    }
}

// Dense-prefill variant of the grouped kernel with compact score storage.
// The mask-derived envelopes only change the address used to retain a score;
// every visible row keeps its original owner thread, dot-product order,
// reduction tree, softmax order, and value-accumulation position.
template <typename KV, typename Mask, int HEADS_PER_BLOCK, bool INDEXED_MASK,
          int VALUES_PER_THREAD>
__global__ static void ds4_flash_attn_d512_shared_kv_grouped_compact_kernel(
        float       * dst,
        const float * q,
        size_t        q_stride_token,
        size_t        q_stride_head,
        const KV    * k,
        const KV    * v,
        const Mask  * mask,
        const float * sinks,
        int           n_tokens,
        int           n_heads,
        int           n_kv,
        float         scale,
        int           raw_rows,
        int           raw_score_capacity,
        int           score_stride,
        const int   * visibility_bounds,
        const int   * indexed_rows,
        const int   * indexed_counts,
        const int   * indexed_owner_offsets,
        const int   * indexed_owner_ranks,
        int           indexed_capacity,
        ds4_inverse_rope_params inverse_rope,
        const float * inverse_rope_coefficients,
        const float * forward_rope_coefficients) {
    constexpr int D = 512;
    constexpr int N_THREADS = 256;
    static_assert(VALUES_PER_THREAD == 2 || VALUES_PER_THREAD == 4);
    const int t = (int) blockIdx.x;
    const int h_begin = (int) blockIdx.y * HEADS_PER_BLOCK;
    const int tid = (int) threadIdx.x;
    if (t >= n_tokens || h_begin >= n_heads) return;

    extern __shared__ float scratch[];
    float * scores = scratch;
    float * reduction = scores + (size_t) HEADS_PER_BLOCK * score_stride;
    int * value_bounds = reinterpret_cast<int *>(
        reduction + (size_t) HEADS_PER_BLOCK * N_THREADS);
    float * q_rope_tail = reinterpret_cast<float *>(
        value_bounds + (size_t) HEADS_PER_BLOCK * 4);

    const int * token_visibility = visibility_bounds + (size_t) t * 4;
    const int mask_raw_first = token_visibility[0];
    const int mask_raw_last = token_visibility[1];
    const int mask_comp_first = token_visibility[2];
    const int mask_comp_last = token_visibility[3];
    const int * token_indexed_rows = nullptr;
    const int * token_owner_offsets = nullptr;
    const int * token_owner_ranks = nullptr;
    int indexed_count = 0;
    if constexpr (INDEXED_MASK) {
        token_indexed_rows = indexed_rows + (size_t) t * indexed_capacity;
        token_owner_offsets = indexed_owner_offsets + (size_t) t * (N_THREADS + 1);
        token_owner_ranks = indexed_owner_ranks + (size_t) t * indexed_capacity;
        indexed_count = indexed_counts[t];
    }

    const float * qh[HEADS_PER_BLOCK];
#pragma unroll
    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
        const int h = h_begin + j;
        qh[j] = q + (size_t) t * q_stride_token +
                (size_t) h * q_stride_head;
    }

    if (inverse_rope.forward_q_enabled) {
        for (int index = tid; index < HEADS_PER_BLOCK * 32;
             index += (int) blockDim.x) {
            const int j = index / 32;
            const int pair = index % 32;
            const float x0 = qh[j][D - 64 + 2 * pair + 0];
            const float x1 = qh[j][D - 64 + 2 * pair + 1];
            const size_t coefficient_index =
                ((size_t) t * 32 + (size_t) pair) * 2;
            const float cos_theta = forward_rope_coefficients[
                coefficient_index + 0];
            const float sin_theta = forward_rope_coefficients[
                coefficient_index + 1];
            ds4_apply_inverse_rope_pair(
                x0, x1, cos_theta, sin_theta,
                q_rope_tail[(size_t) j * 64 + 2 * pair + 0],
                q_rope_tail[(size_t) j * 64 + 2 * pair + 1]);
        }
        __syncthreads();
    }

    float local_max[HEADS_PER_BLOCK];
#pragma unroll
    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
        const int h = h_begin + j;
        local_max[j] = h < n_heads && sinks
            ? sinks[h] : -3.402823466e38f;
    }
    // Build the full kernel's exact per-head non-zero envelope while the
    // softmax weights are emitted. This avoids rescanning every context row
    // after softmax without changing the subsequent V accumulation interval.
    if (tid < HEADS_PER_BLOCK * 4) {
        const int slot = tid & 3;
        value_bounds[tid] = slot == 0
            ? raw_rows
            : slot == 1
                ? -1
                : slot == 2
                    ? (INDEXED_MASK ? indexed_count : n_kv)
                    : -1;
    }

    // Preserve each thread's original r = tid + 256*k order. In indexed mode,
    // owner_ranks is the exact selected subsequence of that traversal, so the
    // hot passes no longer scan every unselected compressed row.
    const int raw_owner_first = mask_raw_first +
        ((tid - (mask_raw_first & (N_THREADS - 1)) + N_THREADS) &
         (N_THREADS - 1));
    const int raw_iteration_count = raw_owner_first <= mask_raw_last
        ? 1 + (mask_raw_last - raw_owner_first) / N_THREADS : 0;
    const int comp_owner_first = mask_comp_first +
        ((tid - (mask_comp_first & (N_THREADS - 1)) + N_THREADS) &
         (N_THREADS - 1));
    const int comp_iteration_count = comp_owner_first <= mask_comp_last
        ? 1 + (mask_comp_last - comp_owner_first) / N_THREADS : 0;
    int owner_begin = 0;
    int owner_count = 0;
    int score_iteration_count = raw_iteration_count + comp_iteration_count;
    if constexpr (INDEXED_MASK) {
        owner_begin = token_owner_offsets[tid];
        owner_count = token_owner_offsets[tid + 1] - owner_begin;
        score_iteration_count = raw_iteration_count + owner_count;
    }
    for (int iteration = 0; iteration < score_iteration_count; ++iteration) {
        int r;
        int score_index;
        if constexpr (INDEXED_MASK) {
            if (iteration < raw_iteration_count) {
                r = raw_owner_first + iteration * N_THREADS;
                score_index = r - mask_raw_first;
            } else {
                const int rank = token_owner_ranks[
                    owner_begin + iteration - raw_iteration_count];
                r = token_indexed_rows[rank];
                score_index = raw_score_capacity + rank;
            }
        } else {
            if (iteration < raw_iteration_count) {
                r = raw_owner_first + iteration * N_THREADS;
                score_index = r - mask_raw_first;
            } else {
                r = comp_owner_first +
                    (iteration - raw_iteration_count) * N_THREADS;
                score_index = raw_score_capacity + r - mask_comp_first;
            }
        }

        const float mask_v = ds4_fa_load<Mask, Mask>(
            mask + (size_t) t * n_kv + r);
        const bool visible = mask_v > -1.0e20f;
        float dot[HEADS_PER_BLOCK] = {};
        if (visible) {
            const KV * kr = k + (size_t) r * D;
#pragma unroll
            for (int d = 0; d < D; ++d) {
                const float kv = ds4_fa_load<KV, Mask>(kr + d);
#pragma unroll
                for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
                    const float qv =
                        inverse_rope.forward_q_enabled && d >= D - 64
                            ? q_rope_tail[(size_t) j * 64 + d - (D - 64)]
                            : qh[j][d];
                    dot[j] += qv * kv;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
            const int h = h_begin + j;
            const float s = h < n_heads && visible
                ? dot[j] * scale + mask_v : -3.402823466e38f;
            scores[(size_t) j * score_stride + score_index] = s;
            local_max[j] = fmaxf(local_max[j], s);
        }
    }

#pragma unroll
    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
        reduction[(size_t) j * N_THREADS + tid] = local_max[j];
    }
    __syncthreads();
    for (int stride = N_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
                float * row = reduction + (size_t) j * N_THREADS;
                row[tid] = fmaxf(row[tid], row[tid + stride]);
            }
        }
        __syncthreads();
    }

    float max_score[HEADS_PER_BLOCK];
    float local_sum[HEADS_PER_BLOCK] = {};
#pragma unroll
    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
        max_score[j] = reduction[(size_t) j * N_THREADS];
    }
    for (int iteration = 0; iteration < score_iteration_count; ++iteration) {
        int score_index;
        int bound_value;
        const bool raw_value = iteration < raw_iteration_count;
        if constexpr (INDEXED_MASK) {
            if (raw_value) {
                const int r = raw_owner_first + iteration * N_THREADS;
                score_index = r - mask_raw_first;
                bound_value = r;
            } else {
                const int rank = token_owner_ranks[
                    owner_begin + iteration - raw_iteration_count];
                score_index = raw_score_capacity + rank;
                bound_value = rank;
            }
        } else {
            if (raw_value) {
                const int r = raw_owner_first + iteration * N_THREADS;
                score_index = r - mask_raw_first;
                bound_value = r;
            } else {
                const int r = comp_owner_first +
                    (iteration - raw_iteration_count) * N_THREADS;
                score_index = raw_score_capacity + r - mask_comp_first;
                bound_value = r;
            }
        }
#pragma unroll
        for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
            float * score = scores + (size_t) j * score_stride + score_index;
            const float weight = expf(*score - max_score[j]);
            *score = weight;
            local_sum[j] += weight;
            if (weight != 0.0f) {
                int * bounds = value_bounds + 4 * j + (raw_value ? 0 : 2);
                atomicMin(bounds + 0, bound_value);
                atomicMax(bounds + 1, bound_value);
            }
        }
    }
    if (tid == 0 && sinks) {
#pragma unroll
        for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
            local_sum[j] += expf(sinks[h_begin + j] - max_score[j]);
        }
    }

#pragma unroll
    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
        reduction[(size_t) j * N_THREADS + tid] = local_sum[j];
    }
    __syncthreads();
    for (int stride = N_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
                float * row = reduction + (size_t) j * N_THREADS;
                row[tid] += row[tid + stride];
            }
        }
        __syncthreads();
    }

    float inv_denom[HEADS_PER_BLOCK];
#pragma unroll
    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
        inv_denom[j] = 1.0f / reduction[(size_t) j * N_THREADS];
    }

    // Finish the shared envelope before the value phase reads it.
    __syncthreads();

    int raw_first = raw_rows;
    int raw_last = -1;
    int comp_first = INDEXED_MASK ? indexed_count : n_kv;
    int comp_last = -1;
    int head_raw_first[HEADS_PER_BLOCK];
    int head_raw_last[HEADS_PER_BLOCK];
    int head_comp_first[HEADS_PER_BLOCK];
    int head_comp_last[HEADS_PER_BLOCK];
#pragma unroll
    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
        const int * bounds = value_bounds + 4 * j;
        head_raw_first[j] = bounds[0];
        head_raw_last[j] = bounds[1];
        head_comp_first[j] = bounds[2];
        head_comp_last[j] = bounds[3];
        raw_first = min(raw_first, head_raw_first[j]);
        raw_last = max(raw_last, head_raw_last[j]);
        comp_first = min(comp_first, head_comp_first[j]);
        comp_last = max(comp_last, head_comp_last[j]);
    }

    // One active thread owns adjacent value dimensions. This retains each
    // dimension's ascending row accumulation order while sharing score loads,
    // row-loop control, and a naturally aligned vector V load across the group.
    float * rope_tail = reduction;
    const int d0 = VALUES_PER_THREAD * tid;
    const int d1 = d0 + 1;
    const int d2 = d0 + 2;
    const int d3 = d0 + 3;
    float acc0[HEADS_PER_BLOCK] = {};
    float acc1[HEADS_PER_BLOCK] = {};
    float acc2[HEADS_PER_BLOCK] = {};
    float acc3[HEADS_PER_BLOCK] = {};
    if (d0 < D) {
        for (int r = raw_first; r <= raw_last; ++r) {
            const int score_index = r - mask_raw_first;
            float vv0;
            float vv1;
            float vv2 = 0.0f;
            float vv3 = 0.0f;
            if constexpr (VALUES_PER_THREAD == 2) {
                ds4_fa_load_pair<KV>(
                    v + (size_t) r * D + d0, vv0, vv1);
            } else {
                ds4_fa_load_quad<KV>(
                    v + (size_t) r * D + d0, vv0, vv1, vv2, vv3);
            }
#pragma unroll
            for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
                if (r >= head_raw_first[j] && r <= head_raw_last[j]) {
                    const float weight =
                        scores[(size_t) j * score_stride + score_index];
                    acc0[j] += weight * vv0;
                    acc1[j] += weight * vv1;
                    if constexpr (VALUES_PER_THREAD == 4) {
                        acc2[j] += weight * vv2;
                        acc3[j] += weight * vv3;
                    }
                }
            }
        }
        if constexpr (INDEXED_MASK) {
            for (int rank = comp_first; rank <= comp_last; ++rank) {
                const int r = token_indexed_rows[rank];
                const int score_index = raw_score_capacity + rank;
                bool any_nonzero = false;
#pragma unroll
                for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
                    any_nonzero = any_nonzero ||
                        scores[(size_t) j * score_stride + score_index] != 0.0f;
                }
                if (!any_nonzero) continue;
                float vv0;
                float vv1;
                float vv2 = 0.0f;
                float vv3 = 0.0f;
                if constexpr (VALUES_PER_THREAD == 2) {
                    ds4_fa_load_pair<KV>(
                        v + (size_t) r * D + d0, vv0, vv1);
                } else {
                    ds4_fa_load_quad<KV>(
                        v + (size_t) r * D + d0, vv0, vv1, vv2, vv3);
                }
#pragma unroll
                for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
                    if (rank >= head_comp_first[j] && rank <= head_comp_last[j]) {
                        const float weight =
                            scores[(size_t) j * score_stride + score_index];
                        acc0[j] += weight * vv0;
                        acc1[j] += weight * vv1;
                        if constexpr (VALUES_PER_THREAD == 4) {
                            acc2[j] += weight * vv2;
                            acc3[j] += weight * vv3;
                        }
                    }
                }
            }
        } else {
            for (int r = comp_first; r <= comp_last; ++r) {
                const int score_index =
                    raw_score_capacity + r - mask_comp_first;
                bool any_nonzero = false;
#pragma unroll
                for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
                    any_nonzero = any_nonzero ||
                        scores[(size_t) j * score_stride + score_index] != 0.0f;
                }
                if (!any_nonzero) continue;
                float vv0;
                float vv1;
                float vv2 = 0.0f;
                float vv3 = 0.0f;
                if constexpr (VALUES_PER_THREAD == 2) {
                    ds4_fa_load_pair<KV>(
                        v + (size_t) r * D + d0, vv0, vv1);
                } else {
                    ds4_fa_load_quad<KV>(
                        v + (size_t) r * D + d0, vv0, vv1, vv2, vv3);
                }
#pragma unroll
                for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
                    if (r >= head_comp_first[j] && r <= head_comp_last[j]) {
                        const float weight =
                            scores[(size_t) j * score_stride + score_index];
                        acc0[j] += weight * vv0;
                        acc1[j] += weight * vv1;
                        if constexpr (VALUES_PER_THREAD == 4) {
                            acc2[j] += weight * vv2;
                            acc3[j] += weight * vv3;
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
            const int h = h_begin + j;
            if (h < n_heads) {
                const float value0 = acc0[j] * inv_denom[j];
                const float value1 = acc1[j] * inv_denom[j];
                if (inverse_rope.enabled && d0 >= D - 64) {
                    rope_tail[(size_t) j * 64 + d0 - (D - 64)] = value0;
                    rope_tail[(size_t) j * 64 + d1 - (D - 64)] = value1;
                    if constexpr (VALUES_PER_THREAD == 4) {
                        const float value2 = acc2[j] * inv_denom[j];
                        const float value3 = acc3[j] * inv_denom[j];
                        rope_tail[(size_t) j * 64 + d2 - (D - 64)] = value2;
                        rope_tail[(size_t) j * 64 + d3 - (D - 64)] = value3;
                    }
                } else {
                    float * out = dst +
                        ((size_t) t * (size_t) n_heads + (size_t) h) * D + d0;
                    out[0] = value0;
                    out[1] = value1;
                    if constexpr (VALUES_PER_THREAD == 4) {
                        out[2] = acc2[j] * inv_denom[j];
                        out[3] = acc3[j] * inv_denom[j];
                    }
                }
            }
        }
    }
    if (inverse_rope.enabled) {
        __syncthreads();
        if (tid < HEADS_PER_BLOCK * 32) {
            const int j = tid / 32;
            const int pair = tid % 32;
            const float x0 = rope_tail[(size_t) j * 64 + 2 * pair + 0];
            const float x1 = rope_tail[(size_t) j * 64 + 2 * pair + 1];
            const size_t coefficient_index =
                ((size_t) t * 32 + (size_t) pair) * 2;
            const float cos_theta = inverse_rope_coefficients[
                coefficient_index + 0];
            const float sin_theta = inverse_rope_coefficients[
                coefficient_index + 1];
            float y0;
            float y1;
            ds4_apply_inverse_rope_pair(
                x0, x1, cos_theta, sin_theta, y0, y1);
            const int h = h_begin + j;
            float * out = dst +
                ((size_t) t * (size_t) n_heads + (size_t) h) * D + D - 64;
            out[2 * pair + 0] = y0;
            out[2 * pair + 1] = y1;
        }
    }
}

template <int HEADS_PER_BLOCK>
static bool ds4_launch_flash_attn_d512_grouped(
        ggml_tensor       * dst,
        const ggml_tensor * Q,
        const ggml_tensor * K,
        const ggml_tensor * V,
        const ggml_tensor * mask,
        const ggml_tensor * sinks,
        bool                kv_f16,
        bool                kv_f32,
        int                 n_tokens,
        int                 n_heads,
        int                 n_kv,
        float               scale,
        int                 raw_rows,
        size_t              q_stride_token,
        size_t              q_stride_head,
        ds4_inverse_rope_params inverse_rope,
        const float        * inverse_rope_coefficients,
        const float        * forward_rope_coefficients,
        size_t              shmem,
        cudaStream_t        stream) {
    dim3 grid(
        (unsigned) n_tokens,
        (unsigned) (n_heads / HEADS_PER_BLOCK), 1);
    if (kv_f16 && (!mask || mask->type == GGML_TYPE_F16)) {
        ds4_flash_attn_d512_shared_kv_grouped_kernel<
            half, half, HEADS_PER_BLOCK>
            <<<grid, 256, shmem, stream>>>(
                (float *) dst->data, (const float *) Q->data,
                q_stride_token, q_stride_head,
                (const half *) K->data, (const half *) V->data,
                mask ? (const half *) mask->data : nullptr,
                sinks ? (const float *) sinks->data : nullptr,
                n_tokens, n_heads, n_kv, scale, raw_rows, inverse_rope,
                inverse_rope_coefficients, forward_rope_coefficients);
    } else if (kv_f32 && (!mask || mask->type == GGML_TYPE_F32)) {
        ds4_flash_attn_d512_shared_kv_grouped_kernel<
            float, float, HEADS_PER_BLOCK>
            <<<grid, 256, shmem, stream>>>(
                (float *) dst->data, (const float *) Q->data,
                q_stride_token, q_stride_head,
                (const float *) K->data, (const float *) V->data,
                mask ? (const float *) mask->data : nullptr,
                sinks ? (const float *) sinks->data : nullptr,
                n_tokens, n_heads, n_kv, scale, raw_rows, inverse_rope,
                inverse_rope_coefficients, forward_rope_coefficients);
    } else if (kv_f32 && mask && mask->type == GGML_TYPE_F16) {
        ds4_flash_attn_d512_shared_kv_grouped_kernel<
            float, half, HEADS_PER_BLOCK>
            <<<grid, 256, shmem, stream>>>(
                (float *) dst->data, (const float *) Q->data,
                q_stride_token, q_stride_head,
                (const float *) K->data, (const float *) V->data,
                (const half *) mask->data,
                sinks ? (const float *) sinks->data : nullptr,
                n_tokens, n_heads, n_kv, scale, raw_rows, inverse_rope,
                inverse_rope_coefficients, forward_rope_coefficients);
    } else {
        return false;
    }
    return true;
}

template <int HEADS_PER_BLOCK, bool INDEXED_MASK, int VALUES_PER_THREAD>
static bool ds4_launch_flash_attn_d512_grouped_compact(
        ggml_tensor       * dst,
        const ggml_tensor * Q,
        const ggml_tensor * K,
        const ggml_tensor * V,
        const ggml_tensor * mask,
        const ggml_tensor * sinks,
        bool                kv_f16,
        bool                kv_f32,
        int                 n_tokens,
        int                 n_heads,
        int                 n_kv,
        float               scale,
        int                 raw_rows,
        int                 raw_score_capacity,
        int                 score_stride,
        const int         * visibility_bounds,
        const int         * indexed_rows,
        const int         * indexed_counts,
        const int         * indexed_owner_offsets,
        const int         * indexed_owner_ranks,
        int                 indexed_capacity,
        size_t              q_stride_token,
        size_t              q_stride_head,
        ds4_inverse_rope_params inverse_rope,
        const float        * inverse_rope_coefficients,
        const float        * forward_rope_coefficients,
        size_t              shmem,
        cudaStream_t        stream) {
    GGML_ASSERT(mask && visibility_bounds);
    if constexpr (INDEXED_MASK) {
        GGML_ASSERT(indexed_rows && indexed_counts &&
                    indexed_owner_offsets && indexed_owner_ranks);
    }
    dim3 grid(
        (unsigned) n_tokens,
        (unsigned) (n_heads / HEADS_PER_BLOCK), 1);
    if (kv_f16 && mask->type == GGML_TYPE_F16) {
        ds4_flash_attn_d512_shared_kv_grouped_compact_kernel<
            half, half, HEADS_PER_BLOCK, INDEXED_MASK, VALUES_PER_THREAD>
            <<<grid, 256, shmem, stream>>>(
                (float *) dst->data, (const float *) Q->data,
                q_stride_token, q_stride_head,
                (const half *) K->data, (const half *) V->data,
                (const half *) mask->data,
                sinks ? (const float *) sinks->data : nullptr,
                n_tokens, n_heads, n_kv, scale, raw_rows,
                raw_score_capacity, score_stride, visibility_bounds,
                indexed_rows, indexed_counts,
                indexed_owner_offsets, indexed_owner_ranks, indexed_capacity,
                inverse_rope, inverse_rope_coefficients,
                forward_rope_coefficients);
    } else if (kv_f32 && mask->type == GGML_TYPE_F32) {
        ds4_flash_attn_d512_shared_kv_grouped_compact_kernel<
            float, float, HEADS_PER_BLOCK, INDEXED_MASK, VALUES_PER_THREAD>
            <<<grid, 256, shmem, stream>>>(
                (float *) dst->data, (const float *) Q->data,
                q_stride_token, q_stride_head,
                (const float *) K->data, (const float *) V->data,
                (const float *) mask->data,
                sinks ? (const float *) sinks->data : nullptr,
                n_tokens, n_heads, n_kv, scale, raw_rows,
                raw_score_capacity, score_stride, visibility_bounds,
                indexed_rows, indexed_counts,
                indexed_owner_offsets, indexed_owner_ranks, indexed_capacity,
                inverse_rope, inverse_rope_coefficients,
                forward_rope_coefficients);
    } else if (kv_f32 && mask->type == GGML_TYPE_F16) {
        ds4_flash_attn_d512_shared_kv_grouped_compact_kernel<
            float, half, HEADS_PER_BLOCK, INDEXED_MASK, VALUES_PER_THREAD>
            <<<grid, 256, shmem, stream>>>(
                (float *) dst->data, (const float *) Q->data,
                q_stride_token, q_stride_head,
                (const float *) K->data, (const float *) V->data,
                (const half *) mask->data,
                sinks ? (const float *) sinks->data : nullptr,
                n_tokens, n_heads, n_kv, scale, raw_rows,
                raw_score_capacity, score_stride, visibility_bounds,
                indexed_rows, indexed_counts,
                indexed_owner_offsets, indexed_owner_ranks, indexed_capacity,
                inverse_rope, inverse_rope_coefficients,
                forward_rope_coefficients);
    } else {
        return false;
    }
    return true;
}

static bool ggml_cuda_ds4_flash_attn_d512_f32_supported(const ggml_tensor * dst) {
    if (!ggml_flash_attn_ext_is_ds4(dst)) {
        return false;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const bool kv_f32 = K && V && K->type == GGML_TYPE_F32 &&
                        V->type == GGML_TYPE_F32;
    const bool kv_f16 = K && V && K->type == GGML_TYPE_F16 &&
                        V->type == GGML_TYPE_F16;
    const bool mask_ok = !mask || mask->type == GGML_TYPE_F16 ||
                         (kv_f32 && mask->type == GGML_TYPE_F32);
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (!Q || !K || !V ||
        Q->type != GGML_TYPE_F32 || (!kv_f32 && !kv_f16) || !mask_ok ||
        dst->type != GGML_TYPE_F32 ||
        Q->ne[0] != 512 || K->ne[0] != 512 || V->ne[0] != 512 ||
        K->ne[1] != V->ne[1] ||
        K->ne[2] != 1 || V->ne[2] != 1 ||
        Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 ||
        dst->ne[0] != 512 || dst->ne[1] != Q->ne[2] ||
        dst->ne[2] != Q->ne[1] || dst->ne[3] != 1 ||
        Q->nb[0] != (int64_t) sizeof(float) ||
        !ggml_is_contiguous(dst) ||
        max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    const size_t kv_esz = kv_f16 ? sizeof(half) : sizeof(float);
    if (K->nb[0] != kv_esz || V->nb[0] != kv_esz ||
        K->nb[1] != (size_t) K->ne[0] * kv_esz ||
        V->nb[1] != (size_t) V->ne[0] * kv_esz ||
        Q->nb[1] % sizeof(float) != 0 ||
        Q->nb[2] % sizeof(float) != 0 ||
        (mask && (mask->ne[0] != K->ne[1] ||
                  mask->ne[1] != Q->ne[1] ||
                  mask->ne[2] != 1 || mask->ne[3] != 1 ||
                  mask->nb[0] != ggml_type_size(mask->type) ||
                  !ggml_is_contiguous(mask)))) {
        return false;
    }
    if (sinks && (sinks->type != GGML_TYPE_F32 ||
                  sinks->ne[0] != Q->ne[2] || sinks->ne[1] != 1 ||
                  sinks->ne[2] != 1 || sinks->ne[3] != 1 ||
                  !ggml_is_contiguous(sinks))) {
        return false;
    }

    const int n_tokens = (int) Q->ne[1];
    const int n_heads = (int) Q->ne[2];
    const int n_kv = (int) K->ne[1];
    if (n_tokens <= 0 || n_heads <= 0 || n_kv <= 0) {
        return false;
    }

    const int raw_rows = ggml_get_op_params_i32(dst, 4);
    const int sparse_keep_rows = ggml_get_op_params_i32(dst, 5);
    const unsigned int ds4_layout =
        (unsigned int) ggml_get_op_params_i32(dst, 6);
    const int raw_window = (int) (ds4_layout >> 16);
    const int sparse_block_size = (int) (ds4_layout & 0xffffu);
    const int rope_flags = ggml_get_op_params_i32(dst, 7);
    if (raw_rows < 0 || raw_rows > n_kv ||
        (ds4_layout != 0 && (raw_window <= 0 || sparse_block_size <= 0)) ||
        sparse_keep_rows == INT_MIN ||
        (sparse_keep_rows != 0 && !mask) ||
        (rope_flags & ~3) != 0 ||
        ((rope_flags & 2) != 0 && (rope_flags & 1) == 0)) {
        return false;
    }

    return true;
}

static bool ggml_cuda_ds4_flash_attn_d512_f32(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!ggml_cuda_ds4_flash_attn_d512_f32_supported(dst)) {
        return false;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const bool kv_f32 = K->type == GGML_TYPE_F32;
    const bool kv_f16 = K->type == GGML_TYPE_F16;
    const int n_tokens = (int) Q->ne[1];
    const int n_heads = (int) Q->ne[2];
    const int n_kv = (int) K->ne[1];
    const size_t q_stride_token = Q->nb[1] / sizeof(float);
    const size_t q_stride_head = Q->nb[2] / sizeof(float);

    int raw_rows = ggml_get_op_params_i32(dst, 4);
    int sparse_keep_rows = ggml_get_op_params_i32(dst, 5);
    const unsigned int ds4_layout =
        (unsigned int) ggml_get_op_params_i32(dst, 6);
    int raw_window = (int) (ds4_layout >> 16);
    int sparse_block_size = (int) (ds4_layout & 0xffffu);
    raw_rows = max(0, min(raw_rows, n_kv));
    if (raw_window <= 0) raw_window = raw_rows;
    raw_window = max(1, min(raw_window, raw_rows));
    if (sparse_block_size <= 0) sparse_block_size = 32;
    const int n_comp_rows = n_kv - raw_rows;
    const int n_comp_blocks = (n_comp_rows + sparse_block_size - 1) /
                              sparse_block_size;
    const bool sparse = sparse_keep_rows > 0 &&
                        sparse_keep_rows < n_comp_rows &&
                        n_comp_blocks > 0;
    const bool indexed_mask = sparse_keep_rows < 0 && n_comp_rows > 0;
    const int indexed_capacity = indexed_mask
        ? min(-sparse_keep_rows, n_comp_rows) : 0;

    ds4_inverse_rope_params inverse_rope{};
    const int rope_flags = ggml_get_op_params_i32(dst, 7);
    inverse_rope.enabled = rope_flags & 1;
    inverse_rope.forward_q_enabled = (rope_flags & 2) != 0;
    if (rope_flags != 0) {
        inverse_rope.kv_start = ggml_get_op_params_i32(dst, 8);
        const float freq_base = ggml_get_op_params_f32(dst, 9);
        inverse_rope.freq_scale = ggml_get_op_params_f32(dst, 10);
        inverse_rope.ext_factor = ggml_get_op_params_f32(dst, 11);
        inverse_rope.attn_factor = ggml_get_op_params_f32(dst, 12);
        const float beta_fast = ggml_get_op_params_f32(dst, 13);
        const float beta_slow = ggml_get_op_params_f32(dst, 14);
        const int n_ctx_orig = ggml_get_op_params_i32(dst, 15);
        float corr_dims[2];
        ggml_rope_yarn_corr_dims(
            64, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        inverse_rope.corr_low = corr_dims[0];
        inverse_rope.corr_high = corr_dims[1];
        inverse_rope.theta_scale = powf(freq_base, -2.0f / 64.0f);
    }

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> inverse_rope_coefficients_alloc(ctx.pool());
    float * inverse_rope_coefficients = nullptr;
    if (inverse_rope.enabled) {
        inverse_rope_coefficients = inverse_rope_coefficients_alloc.alloc(
            (size_t) n_tokens * 32 * 2);
        const int coefficient_count = n_tokens * 32;
        ds4_inverse_rope_coefficients_kernel<<<
            (coefficient_count + 255) / 256, 256, 0, stream>>>(
                inverse_rope_coefficients, n_tokens, inverse_rope);
    }
    ggml_cuda_pool_alloc<float> forward_rope_coefficients_alloc(ctx.pool());
    float * forward_rope_coefficients = nullptr;
    if (inverse_rope.forward_q_enabled) {
        forward_rope_coefficients = forward_rope_coefficients_alloc.alloc(
            (size_t) n_tokens * 32 * 2);
        const int coefficient_count = n_tokens * 32;
        ds4_forward_rope_coefficients_kernel<<<
            (coefficient_count + 255) / 256, 256, 0, stream>>>(
                forward_rope_coefficients, n_tokens, inverse_rope);
    }
    ggml_cuda_pool_alloc<float> mean_k_alloc(ctx.pool());
    float * mean_k = nullptr;
    if (sparse) {
        mean_k = mean_k_alloc.alloc((size_t) n_comp_blocks * 512);
        if (kv_f16) {
            ds4_fa_mean_comp_blocks_kernel<half>
                <<<n_comp_blocks, 256, 0, stream>>>(
                    (const half *) K->data, mean_k, n_kv, raw_rows,
                    sparse_block_size, n_comp_blocks);
        } else {
            ds4_fa_mean_comp_blocks_kernel<float>
                <<<n_comp_blocks, 256, 0, stream>>>(
                    (const float *) K->data, mean_k, n_kv, raw_rows,
                    sparse_block_size, n_comp_blocks);
        }
    }
    dim3 grid((unsigned) n_tokens, (unsigned) n_heads, 1);
    const bool needs_rope_tail = inverse_rope.enabled ||
                                 inverse_rope.forward_q_enabled;
    const size_t shmem =
        (size_t) (n_kv + 2 * n_comp_blocks +
                  (needs_rope_tail ? 64 : 0)) * sizeof(float);
    float params[3] = {};
    memcpy(params, dst->op_params, sizeof(params));
    const float scale = params[0];

    constexpr int group4 = 4;
    constexpr int group2 = 2;
    const size_t group4_shmem =
        ((size_t) group4 * n_kv + (size_t) group4 * 256) * sizeof(float) +
        (size_t) group4 * 4 * sizeof(int) +
        (inverse_rope.forward_q_enabled ? (size_t) group4 * 64 * sizeof(float) : 0);
    const size_t group2_shmem =
        ((size_t) group2 * n_kv + (size_t) group2 * 256) * sizeof(float) +
        (size_t) group2 * 4 * sizeof(int) +
        (inverse_rope.forward_q_enabled ? (size_t) group2 * 64 * sizeof(float) : 0);
    const int compact_score_stride = raw_window +
        (indexed_mask ? indexed_capacity : n_comp_rows);
    const size_t compact_group4_shmem =
        ((size_t) group4 * compact_score_stride + (size_t) group4 * 256) * sizeof(float) +
        (size_t) group4 * 4 * sizeof(int) +
        (inverse_rope.forward_q_enabled ? (size_t) group4 * 64 * sizeof(float) : 0);
    // Four heads win while two blocks can remain resident in 48 KiB of LDS.
    // Beyond that point, two-head grouping trades some K/V reuse for higher
    // occupancy; larger working sets fall back to the single-head kernel.
    if (!sparse && n_heads % group4 == 0 && group4_shmem <= 24 * 1024) {
        return ds4_launch_flash_attn_d512_grouped<group4>(
            dst, Q, K, V, mask, sinks, kv_f16, kv_f32,
            n_tokens, n_heads, n_kv, scale, raw_rows,
            q_stride_token, q_stride_head,
            inverse_rope,
            inverse_rope_coefficients,
            forward_rope_coefficients,
            group4_shmem, stream);
    }
    // Long causal-prefill chunks can have thousands of physical raw rows but
    // at most raw_window visible rows for any one token. Compacting only the
    // score storage lets the same exact four-head kernel remain at two-block
    // occupancy. Keep the ordinary path for shapes that already fit, avoiding
    // a bounds-scan launch where it cannot improve grouping.
    const bool compact_group4 =
        !sparse && mask && n_heads % group4 == 0 &&
        raw_rows > raw_window && group4_shmem > 24 * 1024 &&
        compact_group4_shmem <= 24 * 1024;
    if (compact_group4) {
        ggml_cuda_pool_alloc<int> visibility_bounds_alloc(ctx.pool());
        int * visibility_bounds = visibility_bounds_alloc.alloc(
            (size_t) n_tokens * 4);
        ggml_cuda_pool_alloc<int> indexed_rows_alloc(ctx.pool());
        ggml_cuda_pool_alloc<int> indexed_counts_alloc(ctx.pool());
        ggml_cuda_pool_alloc<int> indexed_owner_offsets_alloc(ctx.pool());
        ggml_cuda_pool_alloc<int> indexed_owner_ranks_alloc(ctx.pool());
        int * indexed_rows = nullptr;
        int * indexed_counts = nullptr;
        int * indexed_owner_offsets = nullptr;
        int * indexed_owner_ranks = nullptr;
        if (indexed_mask) {
            indexed_rows = indexed_rows_alloc.alloc(
                (size_t) n_tokens * indexed_capacity);
            indexed_counts = indexed_counts_alloc.alloc((size_t) n_tokens);
            indexed_owner_offsets = indexed_owner_offsets_alloc.alloc(
                (size_t) n_tokens * 257);
            indexed_owner_ranks = indexed_owner_ranks_alloc.alloc(
                (size_t) n_tokens * indexed_capacity);
            if (mask->type == GGML_TYPE_F16) {
                ds4_fa_indexed_rows_kernel<half><<<n_tokens, 256, 0, stream>>>(
                    (const half *) mask->data, indexed_rows, indexed_counts,
                    indexed_owner_offsets, indexed_owner_ranks,
                    n_tokens, n_kv, raw_rows,
                    indexed_capacity);
            } else {
                ds4_fa_indexed_rows_kernel<float><<<n_tokens, 256, 0, stream>>>(
                    (const float *) mask->data, indexed_rows, indexed_counts,
                    indexed_owner_offsets, indexed_owner_ranks,
                    n_tokens, n_kv, raw_rows,
                    indexed_capacity);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        if (mask->type == GGML_TYPE_F16) {
            ds4_fa_visibility_bounds_kernel<half><<<n_tokens, 64, 0, stream>>>(
                (const half *) mask->data, visibility_bounds,
                n_tokens, n_kv, raw_rows);
        } else {
            ds4_fa_visibility_bounds_kernel<float><<<n_tokens, 64, 0, stream>>>(
                (const float *) mask->data, visibility_bounds,
                n_tokens, n_kv, raw_rows);
        }
        CUDA_CHECK(cudaGetLastError());
        if (indexed_mask) {
            return ds4_launch_flash_attn_d512_grouped_compact<
                group4, true, 4>(
                dst, Q, K, V, mask, sinks, kv_f16, kv_f32,
                n_tokens, n_heads, n_kv, scale, raw_rows,
                raw_window, compact_score_stride, visibility_bounds,
                indexed_rows, indexed_counts,
                indexed_owner_offsets, indexed_owner_ranks, indexed_capacity,
                q_stride_token, q_stride_head,
                inverse_rope,
                inverse_rope_coefficients,
                forward_rope_coefficients,
                compact_group4_shmem, stream);
        }
        return ds4_launch_flash_attn_d512_grouped_compact<
            group4, false, 4>(
            dst, Q, K, V, mask, sinks, kv_f16, kv_f32,
            n_tokens, n_heads, n_kv, scale, raw_rows,
            raw_window, compact_score_stride, visibility_bounds,
            nullptr, nullptr, nullptr, nullptr, 0,
            q_stride_token, q_stride_head,
            inverse_rope,
            inverse_rope_coefficients,
            forward_rope_coefficients,
            compact_group4_shmem, stream);
    }
    if (!sparse && n_heads % group2 == 0 && group2_shmem <= 48 * 1024) {
        return ds4_launch_flash_attn_d512_grouped<group2>(
            dst, Q, K, V, mask, sinks, kv_f16, kv_f32,
            n_tokens, n_heads, n_kv, scale, raw_rows,
            q_stride_token, q_stride_head,
            inverse_rope,
            inverse_rope_coefficients,
            forward_rope_coefficients,
            group2_shmem, stream);
    }

    if (kv_f16 && (!mask || mask->type == GGML_TYPE_F16)) {
        ds4_flash_attn_d512_shared_kv_kernel<half, half>
            <<<grid, 256, shmem, stream>>>(
                (float *) dst->data, (const float *) Q->data,
                q_stride_token, q_stride_head,
                (const half *) K->data, (const half *) V->data,
                mask ? (const half *) mask->data : nullptr,
                sinks ? (const float *) sinks->data : nullptr,
                mean_k, n_tokens, n_heads, n_kv, scale, raw_rows,
                sparse_keep_rows, sparse_block_size, n_comp_blocks,
                inverse_rope, inverse_rope_coefficients,
                forward_rope_coefficients);
    } else if (kv_f32 && (!mask || mask->type == GGML_TYPE_F32)) {
        ds4_flash_attn_d512_shared_kv_kernel<float, float>
            <<<grid, 256, shmem, stream>>>(
                (float *) dst->data, (const float *) Q->data,
                q_stride_token, q_stride_head,
                (const float *) K->data, (const float *) V->data,
                mask ? (const float *) mask->data : nullptr,
                sinks ? (const float *) sinks->data : nullptr,
                mean_k, n_tokens, n_heads, n_kv, scale, raw_rows,
                sparse_keep_rows, sparse_block_size, n_comp_blocks,
                inverse_rope, inverse_rope_coefficients,
                forward_rope_coefficients);
    } else if (kv_f32 && mask && mask->type == GGML_TYPE_F16) {
        ds4_flash_attn_d512_shared_kv_kernel<float, half>
            <<<grid, 256, shmem, stream>>>(
                (float *) dst->data, (const float *) Q->data,
                q_stride_token, q_stride_head,
                (const float *) K->data, (const float *) V->data,
                (const half *) mask->data,
                sinks ? (const float *) sinks->data : nullptr,
                mean_k, n_tokens, n_heads, n_kv, scale, raw_rows,
                sparse_keep_rows, sparse_block_size, n_comp_blocks,
                inverse_rope, inverse_rope_coefficients,
                forward_rope_coefficients);
    } else {
        return false;
    }
    return true;
}

#endif // defined(GGML_USE_HIP)

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if ((turing_mma_available(cc) || amd_wmma_available(cc)) && Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || amd_wmma_available(cc) || Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        if (use_gqa_opt && gqa_ratio > 1) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }

        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)

 #ifndef GGML_USE_HIP
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TQ3_0, GGML_TYPE_TQ3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_TQ3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_TQ3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_TQ3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_TQ3_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TQ3_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TQ3_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TQ3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TQ3_0, GGML_TYPE_BF16)
#endif // GGML_USE_HIP
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
#ifndef GGML_USE_HIP
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TQ3_0, GGML_TYPE_TQ3_0)
#endif // GGML_USE_HIP
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_WMMA_F16 = 300,
    BEST_FATTN_KERNEL_MMA_F16  = 400,
    BEST_FATTN_KERNEL_CHUNKED  = 500,   // chunked long-context prefill (fattn-chunked.cu)
};

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

    switch (K->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return BEST_FATTN_KERNEL_NONE;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_TQ3_0:
        case GGML_TYPE_BF16:
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // Chunked long-context prefill. Routes to fattn-chunked.cu which uses
    // cuBLAS SGEMM + online softmax with adaptive chunk sizing for O(CHUNK)
    // temp memory. Intended for prefill (Q->ne[1] > 1) at contexts where the
    // MMA kernel's O(nq_chunk * kv_len * D) memory pressure dominates.
    //
    // TQ3_0 has no MMA kernel support and must always use chunked.
    // For other K/V types MMA is faster, so the threshold-based forcing is
    // off by default (DFLASH27B_CHUNKED_THRESHOLD=0). Set the env var to a
    // positive value (e.g. 8192) to opt in when MMA's temp memory becomes
    // the bottleneck on a memory-tight card.
    {
        static const int64_t chunked_threshold = [] {
            const char * e = getenv("DFLASH27B_CHUNKED_THRESHOLD");
            if (e) return (int64_t)atoll(e);
            return (int64_t)0;
        }();
        const bool kv_supported =
            (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_BF16 ||
             K->type == GGML_TYPE_Q4_0 || K->type == GGML_TYPE_Q8_0 ||
             K->type == GGML_TYPE_TQ3_0) &&
            (V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_BF16 ||
             V->type == GGML_TYPE_Q4_0 || V->type == GGML_TYPE_Q8_0 ||
             V->type == GGML_TYPE_TQ3_0);
        // Route TQ3_0 through CHUNKED except for the narrow CUDA VEC case below.
        // CHUNKED has the general TQ3 contract: dequant K/V to f32 in compressed
        // (rotated) domain, attend with graph-rotated Q, return rotated O for
        // the graph to inverse-rotate.
        const bool tq3_any = (K->type == GGML_TYPE_TQ3_0 || V->type == GGML_TYPE_TQ3_0);
        // VEC dispatch can handle TQ3 only at SWA-shaped decode and only when
        // the actual vector-kernel constraints hold. Everything else still
        // routes to CHUNKED.
#ifndef GGML_USE_HIP
        const bool tq3_can_vec = (Q->ne[1] == 1) && (Q->ne[0] <= 256) &&
            (Q->ne[0] % 64 == 0) && (K->ne[1] % FATTN_KQ_STRIDE == 0);
#else
        const bool tq3_can_vec = false;
#endif // GGML_USE_HIP
        const bool tq3_needs_chunked = tq3_any && !tq3_can_vec;
        if ((chunked_threshold > 0 && K->ne[1] > chunked_threshold) || tq3_needs_chunked) {
            if (Q->type == GGML_TYPE_F32 && kv_supported && mask != nullptr) {
                return BEST_FATTN_KERNEL_CHUNKED;
            }
        }
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0;
    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        int gqa_ratio_eff = 1;
        const int ncols2_max = Q->ne[0] == 576 ? 16 : 8;
        while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
            gqa_ratio_eff *= 2;
        }
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use the WMMA kernel if possible:
    if (ggml_cuda_should_use_wmma_fattn(cc) && K->ne[1] % FATTN_KQ_STRIDE == 0 && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 512 && Q->ne[0] != 576) {
        if (can_use_vector_kernel && Q->ne[1] <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        return BEST_FATTN_KERNEL_WMMA_F16;
    }

    if (amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) && gqa_opt_applies && Q->ne[0] <= 128 && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (Q->ne[1] == 1) {
                    if (!gqa_opt_applies) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            } else {
                if (Q->ne[1] <= 2) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        }
        int gqa_ratio_eff = 1;
        const int ncols2_max = Q->ne[0] == 576 ? 16 : 8;
        while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
            gqa_ratio_eff *= 2;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 8) {
            return BEST_FATTN_KERNEL_TILE; // AMD WMMA is only faster if the full tile width of 16 can be utilized.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use MFMA flash attention for CDNA (MI100+):
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 256 && Q->ne[0] != 512 && Q->ne[0] != 576) {
        const int64_t eff_nq = Q->ne[1] * (gqa_opt_applies ? gqa_ratio : 1);
        // MMA vs tile crossover benchmarked on MI300X @ d32768:
        //   hsk=64  (gqa=4): MMA wins at eff >= 128 (+11%)
        //   hsk=128 (gqa=4): MMA wins at eff >= 128 (+4%)
        if (eff_nq >= (GGML_CUDA_CC_IS_CDNA1(cc) && Q->ne[0] == 64 ? 64 : 128)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        // Fall through to tile kernel for small effective batch sizes.
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    if (ggml_flash_attn_ext_is_ds4(dst)) {
#if defined(GGML_USE_HIP)
        if (!ggml_cuda_ds4_flash_attn_d512_f32(ctx, dst)) {
            GGML_ABORT("unsupported DeepSeek4 D=512 flash-attention contract");
        }
        return;
#else
        GGML_ABORT("DeepSeek4 D=512 flash attention is only available on HIP");
#endif // defined(GGML_USE_HIP)
    }
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_WMMA_F16:
            ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_CHUNKED:
            ggml_cuda_flash_attn_ext_chunked(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    if (ggml_flash_attn_ext_is_ds4(dst)) {
#if defined(GGML_USE_HIP)
        return ggml_cuda_ds4_flash_attn_d512_f32_supported(dst);
#else
        return false;
#endif // defined(GGML_USE_HIP)
    }
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
