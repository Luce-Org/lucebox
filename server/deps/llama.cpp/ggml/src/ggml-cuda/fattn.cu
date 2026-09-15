#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn-chunked.cuh"
#include "fattn.cuh"
#include "ds4-env.cuh"
#include "ds4-causal.h"

#include <type_traits>

static thread_local size_t g_mla_stream_topk_launch_count = 0;

extern "C" size_t ggml_backend_cuda_get_mla_stream_topk_launch_count(void) {
    return g_mla_stream_topk_launch_count;
}

// Bumped from the template instantiated in the fattn-mma-f16 instance TUs;
// see the extern "C" declaration in fattn-mma-f16.cuh.
static thread_local size_t g_fattn_mma256_launch_count = 0;

extern "C" void ggml_backend_cuda_record_fattn_mma256_launch(void) {
    ++g_fattn_mma256_launch_count;
}

extern "C" size_t ggml_backend_cuda_get_fattn_mma256_launch_count(void) {
    return g_fattn_mma256_launch_count;
}

// Same pattern for the rocWMMA head-size-256 kernel (fattn-wmma-f16.cu).
static thread_local size_t g_fattn_wmma256_launch_count = 0;

extern "C" void ggml_backend_cuda_record_fattn_wmma256_launch(void) {
    ++g_fattn_wmma256_launch_count;
}

extern "C" size_t ggml_backend_cuda_get_fattn_wmma256_launch_count(void) {
    return g_fattn_wmma256_launch_count;
}

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
    const int32_t * positions;
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
// Return the unreduced angle. Frequency scaling and YaRN interpolation must
// happen before modulo reduction; reducing here changes the angle whenever
// freq_scale is not an integer (the production DS4 configuration uses YaRN).
__device__ static __forceinline__ double ds4_rope_theta_fp64(
        int32_t p, float theta_scale, int exp_int) {
    return exp_int == 0
        ? (double) p
        : (double) p * pow((double) theta_scale, (double) exp_int);
}

__device__ static __forceinline__ void ds4_rope_coefficients_at_position(
        int pair, int32_t position,
        const ds4_inverse_rope_params & p,
        float & cos_theta, float & sin_theta) {
    const int i0 = 2 * pair;
    const double theta_extrap = ds4_rope_theta_fp64(
        position, p.theta_scale, pair);
    const double theta_interp = (double) p.freq_scale * theta_extrap;
    double theta = theta_interp;
    float mscale = p.attn_factor;
    if (p.ext_factor != 0.0f) {
        const float ramp_y = (i0 / 2 - p.corr_low) /
            max(0.001f, p.corr_high - p.corr_low);
        const float ramp_mix =
            (1.0f - min(1.0f, max(0.0f, ramp_y))) * p.ext_factor;
        theta = theta_interp * (1.0 - (double) ramp_mix) +
                theta_extrap * (double) ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / p.freq_scale);
    }

    // Match rope_yarn(): preserve FP64 precision through all scaling and
    // blend operations, then reduce only at the trig boundary.
    const double tau = 6.2831853071795864769;
    theta -= tau * floor(theta * (1.0 / tau));
    cos_theta = cosf((float) theta) * mscale;
    sin_theta = sinf((float) theta) * mscale;
}

__device__ static __forceinline__ void ds4_inverse_rope_coefficients(
        int pair, int token,
        const ds4_inverse_rope_params & p,
        float & cos_theta, float & sin_theta) {
    ds4_rope_coefficients_at_position(
        pair, -(p.positions ? p.positions[token] : p.kv_start + token),
        p, cos_theta, sin_theta);
}

// Forward counterpart of ds4_inverse_rope_coefficients. Keep the expressions
// aligned with rope_norm<true> in rope.cu; unlike the inverse path, position is
// positive. Compressed-layer YaRN interpolation means the inverse coefficients
// cannot safely be recovered by merely negating sin(theta).
__device__ static __forceinline__ void ds4_forward_rope_coefficients(
        int pair, int token,
        const ds4_inverse_rope_params & p,
        float & cos_theta, float & sin_theta) {
    ds4_rope_coefficients_at_position(
        pair, p.positions ? p.positions[token] : p.kv_start + token,
        p, cos_theta, sin_theta);
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

// Exact DS4 ratio-4 layer-major visibility without a materialized mask.
// Physical raw rows are [prior chronological SWA | current chunk]. A
// compressed row becomes visible only after its four source tokens complete.
__global__ static void ds4_fa_ratio4_causal_bounds_kernel(
        int * bounds,
        int   n_tokens,
        int   n_kv,
        int   raw_rows,
        int   raw_window,
        int   kv_start) {
    const int t = (int) blockIdx.x * (int) blockDim.x +
                  (int) threadIdx.x;
    if (t >= n_tokens) return;

    const auto visible = ds4_ratio4_causal_visibility(
        t, n_tokens, raw_rows, n_kv - raw_rows, raw_window, kv_start);
    int * token_bounds = bounds + (size_t) t * 4;
    token_bounds[0] = visible.raw_first;
    token_bounds[1] = visible.raw_last;
    token_bounds[2] = visible.comp_first;
    token_bounds[3] = visible.comp_last;
}

// Exact causal bounds for a layer-major batch. Physical rows are
// [chronological rows from before the batch | every row in this batch |
// contiguous compressed rows]. Encoding both monotonic frontiers directly
// avoids materializing the quadratic [KV,batch] mask.
__global__ static void ds4_fa_contiguous_causal_bounds_kernel(
        int * bounds,
        int   n_tokens,
        int   n_kv,
        int   raw_rows,
        int   raw_window,
        int   kv_start,
        int   compression_ratio) {
    const int t = (int) blockIdx.x * (int) blockDim.x +
                  (int) threadIdx.x;
    if (t >= n_tokens) return;

    const int prior_rows = raw_rows - n_tokens;
    const int n_comp_rows = n_kv - raw_rows;
    const int visible_comp = compression_ratio > 1
        ? min(n_comp_rows, (kv_start + t + 1) / compression_ratio)
        : 0;
    int * token_bounds = bounds + (size_t) t * 4;
    token_bounds[0] = max(0, prior_rows + t - raw_window + 1);
    token_bounds[1] = prior_rows + t;
    token_bounds[2] = visible_comp > 0 ? raw_rows : n_kv;
    token_bounds[3] = visible_comp > 0
        ? raw_rows + visible_comp - 1 : -1;
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

// Long contexts used to compact the exact indexer mask on thread 0, scanning
// every compressed row serially for every token and layer. Compact chunks in
// physical-row order with warp ballots instead. The generated selected_rows
// and per-owner rank lists are deliberately identical to the serial kernel so
// the attention accumulation order and logits remain unchanged.
template <typename Mask>
__global__ static void ds4_fa_indexed_rows_parallel_kernel(
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
    constexpr int MAX_WARPS = N_THREADS / 32;
    __shared__ int warp_offsets[MAX_WARPS];
    __shared__ int owner_counts[N_THREADS];
    __shared__ int chunk_base;
    __shared__ int total_selected;

    const int n_comp_rows = n_kv - raw_rows;
    const Mask * token_mask = mask + (size_t) t * n_kv;
    int * token_rows = selected_rows + (size_t) t * capacity;
    int * token_owner_offsets = owner_offsets + (size_t) t * (N_THREADS + 1);
    int * token_owner_ranks = owner_ranks + (size_t) t * capacity;

    owner_counts[tid] = 0;
    if (tid == 0) total_selected = 0;
    __syncthreads();

    const int lane = tid % warpSize;
    const int warp = tid / warpSize;
    const int n_warps = N_THREADS / warpSize;
    for (int base = 0; base < n_comp_rows; base += N_THREADS) {
        const int c = base + tid;
        const bool selected = c < n_comp_rows &&
            ds4_fa_load<Mask, Mask>(token_mask + raw_rows + c) > -1.0e20f;
        const unsigned long long selected_bits = __ballot(selected);
        if (lane == 0) {
            warp_offsets[warp] = __popcll(selected_bits);
        }
        __syncthreads();

        if (tid == 0) {
            int prefix = 0;
            for (int w = 0; w < n_warps; ++w) {
                const int count = warp_offsets[w];
                warp_offsets[w] = prefix;
                prefix += count;
            }
            chunk_base = total_selected;
            total_selected += prefix;
        }
        __syncthreads();

        const unsigned long long lower_lanes = lane == 0
            ? 0ULL
            : ((1ULL << lane) - 1ULL);
        const int rank = chunk_base + warp_offsets[warp] +
            __popcll(selected_bits & lower_lanes);
        if (selected && rank < capacity) {
            token_rows[rank] = raw_rows + c;
        }
        __syncthreads();
    }

    if (tid == 0) {
        selected_counts[t] = min(total_selected, capacity);
    }
    __syncthreads();

    const int count = selected_counts[t];
    for (int rank = tid; rank < count; rank += N_THREADS) {
        const int owner = token_rows[rank] & (N_THREADS - 1);
        atomicAdd(owner_counts + owner, 1);
    }
    __syncthreads();

    if (tid == 0) {
        int prefix = 0;
        for (int owner = 0; owner < N_THREADS; ++owner) {
            token_owner_offsets[owner] = prefix;
            prefix += owner_counts[owner];
        }
        token_owner_offsets[N_THREADS] = prefix;
    }
    __syncthreads();

    // One thread owns each original attention reduction lane. Walking the
    // already sorted rows preserves the serial kernel's rank order per owner.
    int write = token_owner_offsets[tid];
    for (int rank = 0; rank < count; ++rank) {
        if ((token_rows[rank] & (N_THREADS - 1)) == tid) {
            token_owner_ranks[write++] = rank;
        }
    }
}

// The indexer already returns the exact compressed-row set, ordered by score.
// Convert it directly into the lookup tables consumed by compact attention.
// A shared-memory bitonic sort restores ascending physical-row order, matching
// the old top-k -> mask -> physical scan path and therefore preserving each
// reduction lane's accumulation order exactly.
template <typename Mask, int SORT_WIDTH, bool RATIO4_CAUSAL = false>
__global__ static void ds4_fa_indexed_rows_topk_kernel(
        const Mask    * mask,
        const int32_t * topk,
        int           * selected_rows,
        int           * selected_counts,
        int           * owner_offsets,
        int           * owner_ranks,
        int             n_tokens,
        int             n_kv,
        int             raw_rows,
        int             capacity,
        int             kv_start = 0) {
    const int t = (int) blockIdx.x;
    const int tid = (int) threadIdx.x;
    if (t >= n_tokens) return;

    constexpr int N_OWNERS = 256;
    constexpr int INVALID_ROW = 0x7fffffff;
    __shared__ int sorted_rows[SORT_WIDTH];
    __shared__ int owner_counts[N_OWNERS];
    __shared__ int count;

    const int n_comp_rows = n_kv - raw_rows;
    const Mask * token_mask = RATIO4_CAUSAL
        ? nullptr : mask + (size_t) t * n_kv;
    const int32_t * token_topk = topk + (size_t) t * capacity;
    int * token_rows = selected_rows + (size_t) t * capacity;
    int * token_owner_offsets = owner_offsets + (size_t) t * (N_OWNERS + 1);
    int * token_owner_ranks = owner_ranks + (size_t) t * capacity;

    int row = INVALID_ROW;
    if (tid < capacity) {
        const int comp = token_topk[tid];
        const int physical = raw_rows + comp;
        bool visible = comp >= 0 && comp < n_comp_rows;
        if constexpr (RATIO4_CAUSAL) {
            visible = visible && comp < (kv_start + t + 1) / 4;
        } else {
            visible = visible &&
                ds4_fa_load<Mask, Mask>(token_mask + physical) > -1.0e20f;
        }
        if (visible) {
            row = physical;
        }
    }
    sorted_rows[tid] = row;
    if (tid < N_OWNERS) owner_counts[tid] = 0;
    __syncthreads();

    for (int width = 2; width <= SORT_WIDTH; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            const int peer = tid ^ stride;
            if (peer > tid) {
                const int lhs = sorted_rows[tid];
                const int rhs = sorted_rows[peer];
                const bool ascending = (tid & width) == 0;
                if ((lhs > rhs) == ascending) {
                    sorted_rows[tid] = rhs;
                    sorted_rows[peer] = lhs;
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        int valid = 0;
        while (valid < capacity && sorted_rows[valid] != INVALID_ROW) {
            ++valid;
        }
        count = valid;
        selected_counts[t] = valid;
    }
    __syncthreads();

    if (tid < count) {
        token_rows[tid] = sorted_rows[tid];
        atomicAdd(owner_counts + (sorted_rows[tid] & (N_OWNERS - 1)), 1);
    }
    __syncthreads();

    if (tid == 0) {
        int prefix = 0;
        for (int owner = 0; owner < N_OWNERS; ++owner) {
            token_owner_offsets[owner] = prefix;
            prefix += owner_counts[owner];
        }
        token_owner_offsets[N_OWNERS] = prefix;
    }
    __syncthreads();

    if (tid < N_OWNERS) {
        int write = token_owner_offsets[tid];
        for (int rank = 0; rank < count; ++rank) {
            if ((token_rows[rank] & (N_OWNERS - 1)) == tid) {
                token_owner_ranks[write++] = rank;
            }
        }
    }
}

template <typename Mask>
static void ds4_launch_indexed_rows_topk(
        const Mask * mask, const int32_t * topk,
        int * selected_rows, int * selected_counts,
        int * owner_offsets, int * owner_ranks,
        int n_tokens, int n_kv, int raw_rows, int capacity,
        cudaStream_t stream) {
    // The learned top-512 stays on its original launch. A batched verifier
    // appends a small saved-raw suffix and needs the next sorting bucket.
    if (capacity <= 512) {
        ds4_fa_indexed_rows_topk_kernel<Mask, 512><<<n_tokens, 512, 0, stream>>>(
            mask, topk, selected_rows, selected_counts, owner_offsets, owner_ranks,
            n_tokens, n_kv, raw_rows, capacity);
    } else {
        ds4_fa_indexed_rows_topk_kernel<Mask, 1024><<<n_tokens, 1024, 0, stream>>>(
            mask, topk, selected_rows, selected_counts, owner_offsets, owner_ranks,
            n_tokens, n_kv, raw_rows, capacity);
    }
}

template <typename KV, typename Mask, bool ANALYTIC_CAUSAL = false>
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
        int           raw_window,
        int           sparse_keep_rows,
        int           sparse_block_size,
        int           n_comp_blocks,
        int           kv_start,
        int           compression_ratio,
        ds4_inverse_rope_params inverse_rope,
        const float * inverse_rope_coefficients,
        const float * forward_rope_coefficients,
        bool          skip_sparse_value_gaps) {
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
    const int prior_rows = raw_rows - n_tokens;
    const int causal_raw_first = ANALYTIC_CAUSAL
        ? max(0, prior_rows + t - raw_window + 1) : 0;
    const int causal_raw_last = ANALYTIC_CAUSAL
        ? prior_rows + t : raw_rows - 1;
    const int causal_comp_rows = ANALYTIC_CAUSAL && compression_ratio > 1
        ? min(n_comp_rows, (kv_start + t + 1) / compression_ratio)
        : n_comp_rows;
    const bool sparse = mean_k && n_comp_blocks > 0 &&
                        sparse_keep_rows > 0 &&
                        sparse_keep_rows < n_comp_rows;
    if (sparse) {
        for (int b = tid; b < n_comp_blocks; b += (int) blockDim.x) {
            const int first_row = raw_rows + b * sparse_block_size;
            float mask_v = 0.0f;
            if constexpr (ANALYTIC_CAUSAL) {
                mask_v = first_row < raw_rows + causal_comp_rows
                    ? 0.0f : -3.402823466e38f;
            } else if (mask) {
                mask_v = ds4_fa_load<Mask, Mask>(
                    mask + (size_t) t * n_kv + first_row);
            }
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

    // Keep the original r % 256 owner for every visible row so max/sum
    // reductions remain byte-stable. With analytic causal bounds, selected-
    // block mode can omit invisible raw rows and unselected compressed rows
    // entirely; the old loop still backs the kill switch and explicit masks.
    const bool compact_analytic = ANALYTIC_CAUSAL &&
        (!sparse || skip_sparse_value_gaps);
    float local_max = sinks ? sinks[h] : -3.402823466e38f;
    if (compact_analytic) {
        constexpr int N_THREADS = 256;
        const int raw_owner_first = causal_raw_first +
            ((tid - (causal_raw_first & (N_THREADS - 1)) + N_THREADS) &
             (N_THREADS - 1));
        for (int r = raw_owner_first; r <= causal_raw_last;
             r += N_THREADS) {
            const KV * kr = k + (size_t) r * D;
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < D; ++d) {
                const float qv = inverse_rope.forward_q_enabled && d >= D - 64
                    ? q_rope_tail[d - (D - 64)] : qh[d];
                dot += qv * ds4_fa_load<KV, Mask>(kr + d);
            }
            const float s = dot * scale;
            scores[r] = s;
            local_max = fmaxf(local_max, s);
        }
        const int comp_begin = raw_rows;
        const int comp_end = raw_rows + causal_comp_rows - 1;
        const int comp_owner_first = comp_begin +
            ((tid - (comp_begin & (N_THREADS - 1)) + N_THREADS) &
             (N_THREADS - 1));
        for (int r = comp_owner_first; r <= comp_end; r += N_THREADS) {
            const int b = (r - raw_rows) / sparse_block_size;
            if (sparse &&
                (b >= n_comp_blocks || block_keep[b] == 0.0f)) {
                continue;
            }
            const KV * kr = k + (size_t) r * D;
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < D; ++d) {
                const float qv = inverse_rope.forward_q_enabled && d >= D - 64
                    ? q_rope_tail[d - (D - 64)] : qh[d];
                dot += qv * ds4_fa_load<KV, Mask>(kr + d);
            }
            const float s = dot * scale;
            scores[r] = s;
            local_max = fmaxf(local_max, s);
        }
    } else {
        for (int r = tid; r < n_kv; r += blockDim.x) {
            bool keep = true;
            if (sparse && r >= raw_rows) {
                const int b = (r - raw_rows) / sparse_block_size;
                keep = b < n_comp_blocks && block_keep[b] != 0.0f;
            }
            float mask_v = 0.0f;
            if constexpr (ANALYTIC_CAUSAL) {
                const bool visible = r < raw_rows
                    ? r >= causal_raw_first && r <= causal_raw_last
                    : r < raw_rows + causal_comp_rows;
                mask_v = visible ? 0.0f : -3.402823466e38f;
            } else if (mask) {
                mask_v = ds4_fa_load<Mask, Mask>(
                    mask + (size_t) t * n_kv + r);
            }
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
    }
    const float max_score = ds4_fa_block_max(local_max);

    float local_sum = 0.0f;
    if (compact_analytic) {
        constexpr int N_THREADS = 256;
        const int raw_owner_first = causal_raw_first +
            ((tid - (causal_raw_first & (N_THREADS - 1)) + N_THREADS) &
             (N_THREADS - 1));
        for (int r = raw_owner_first; r <= causal_raw_last;
             r += N_THREADS) {
            const float w = expf(scores[r] - max_score);
            scores[r] = w;
            local_sum += w;
        }
        const int comp_begin = raw_rows;
        const int comp_end = raw_rows + causal_comp_rows - 1;
        const int comp_owner_first = comp_begin +
            ((tid - (comp_begin & (N_THREADS - 1)) + N_THREADS) &
             (N_THREADS - 1));
        for (int r = comp_owner_first; r <= comp_end; r += N_THREADS) {
            const int b = (r - raw_rows) / sparse_block_size;
            if (sparse &&
                (b >= n_comp_blocks || block_keep[b] == 0.0f)) {
                continue;
            }
            const float w = expf(scores[r] - max_score);
            scores[r] = w;
            local_sum += w;
        }
    } else {
        for (int r = tid; r < n_kv; r += blockDim.x) {
            const float w = expf(scores[r] - max_score);
            scores[r] = w;
            local_sum += w;
        }
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
    if (compact_analytic) {
        if (tid == 0) {
            value_bounds[0] = causal_raw_first;
            value_bounds[1] = causal_raw_last;
            value_bounds[2] = raw_rows;
            value_bounds[3] = raw_rows + causal_comp_rows - 1;
        }
    } else if (tid < warpSize) {
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
        if (sparse && skip_sparse_value_gaps) {
            // The selector keeps whole compressed blocks. Visit those blocks
            // in ascending row order instead of loading the zero-weight gaps
            // between the first and last selected block. This preserves the
            // order of every non-zero accumulation while bounding value-side
            // traffic by sparse_keep_rows rather than total context length.
            for (int b = 0; b < n_comp_blocks; ++b) {
                if (block_keep[b] == 0.0f) {
                    continue;
                }
                const int first = raw_rows + b * sparse_block_size;
                int last = min(n_kv, first + sparse_block_size);
                if constexpr (ANALYTIC_CAUSAL) {
                    last = min(last, raw_rows + causal_comp_rows);
                }
                for (int r = first; r < last; ++r) {
                    acc += scores[r] * ds4_fa_load<KV, Mask>(
                        v + (size_t) r * D + d);
                }
            }
        } else {
            for (int r = comp_first; r <= comp_last; ++r) {
                acc += scores[r] * ds4_fa_load<KV, Mask>(
                    v + (size_t) r * D + d);
            }
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
    // Retire all reads of the maxima before reusing reduction for sums.
    __syncthreads();
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
          bool MASKLESS_CAUSAL, int VALUES_PER_THREAD, bool QUAD_DOT = false>
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
    // Reserve the exact per-head nonzero envelopes. Scalar paths update them
    // while emitting weights; the eight-head path scans bounded scores later.
    // Both preserve the full kernel's value-accumulation interval and order.
    if (tid < HEADS_PER_BLOCK * 4) {
        const int slot = tid & 3;
        if constexpr (MASKLESS_CAUSAL) {
            // Every iterated row is visible by construction, so seed the
            // exact analytic interval instead of rediscovering it with the
            // shared-memory atomics below (skipped for this specialization).
            // Underflowed weights stay zero in the value pass, so the wider
            // interval is exact: it only adds 0*v terms.
            if (slot == 0) {
                value_bounds[tid] = mask_raw_first;
            } else if (slot == 1) {
                value_bounds[tid] = mask_raw_last;
            } else if (slot == 2) {
                value_bounds[tid] = INDEXED_MASK
                    ? (indexed_count > 0 ? 0 : indexed_count)
                    : mask_comp_first;
            } else {
                value_bounds[tid] = INDEXED_MASK
                    ? indexed_count - 1 : mask_comp_last;
            }
        } else {
            value_bounds[tid] = slot == 0
                ? raw_rows
                : slot == 1
                    ? -1
                    : slot == 2
                        ? (INDEXED_MASK ? indexed_count : n_kv)
                        : -1;
        }
    }
    __syncthreads();

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

        float mask_v = 0.0f;
        if constexpr (!MASKLESS_CAUSAL) {
            mask_v = ds4_fa_load<Mask, Mask>(
                mask + (size_t) t * n_kv + r);
        }
        const bool visible = mask_v > -1.0e20f;
        float dot[HEADS_PER_BLOCK] = {};
        if (visible) {
            const KV * kr = k + (size_t) r * D;
            if constexpr (QUAD_DOT) {
                // Bound code expansion for the eight-head HIP path while
                // preserving the original sequential FMAs in each head.
#pragma unroll 4
                for (int d = 0; d < D; d += 4) {
                    float k0, k1, k2, k3;
                    ds4_fa_load_quad<KV>(kr + d, k0, k1, k2, k3);
#pragma unroll
                    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
                        const float * qr = inverse_rope.forward_q_enabled && d >= D - 64
                            ? q_rope_tail + (size_t) j * 64 + d - (D - 64)
                            : qh[j] + d;
                        // Keep the four FMAs in the original dimension order.
                        dot[j] += qr[0] * k0;
                        dot[j] += qr[1] * k1;
                        dot[j] += qr[2] * k2;
                        dot[j] += qr[3] * k3;
                    }
                }
            } else {
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
    // Retire all reads of the maxima before reusing reduction for sums.
    __syncthreads();
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
            if constexpr (!QUAD_DOT && !MASKLESS_CAUSAL) {
                if (weight != 0.0f) {
                    int * bounds = value_bounds + 4 * j + (raw_value ? 0 : 2);
                    atomicMin(bounds + 0, bound_value);
                    atomicMax(bounds + 1, bound_value);
                }
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

    if constexpr (QUAD_DOT) {
        static_assert(INDEXED_MASK && HEADS_PER_BLOCK == 8);
        // One wave scans each head's completed weights. Integer min/max
        // replaces contended per-weight atomics without touching the score,
        // softmax, or value arithmetic. The sum reduction above has already
        // made every weight visible to every wave in the block.
        constexpr int WAVE = 32;
        const int head = tid / WAVE;
        const int lane = tid % WAVE;
        const float * head_scores = scores + (size_t) head * score_stride;
        int first_raw = raw_rows, last_raw = -1;
        int first_comp = indexed_count, last_comp = -1;
        for (int r = mask_raw_first + lane; r <= mask_raw_last; r += WAVE) {
            if (head_scores[r - mask_raw_first] != 0.0f) {
                first_raw = min(first_raw, r);
                last_raw = max(last_raw, r);
            }
        }
        for (int rank = lane; rank < indexed_count; rank += WAVE) {
            if (head_scores[raw_score_capacity + rank] != 0.0f) {
                first_comp = min(first_comp, rank);
                last_comp = max(last_comp, rank);
            }
        }
#pragma unroll
        for (int delta = WAVE / 2; delta > 0; delta >>= 1) {
            first_raw = min(first_raw, __shfl_xor_sync(0xffffffffu, first_raw, delta, WAVE));
            last_raw = max(last_raw, __shfl_xor_sync(0xffffffffu, last_raw, delta, WAVE));
            first_comp = min(first_comp, __shfl_xor_sync(0xffffffffu, first_comp, delta, WAVE));
            last_comp = max(last_comp, __shfl_xor_sync(0xffffffffu, last_comp, delta, WAVE));
        }
        if (lane == 0) {
            value_bounds[4 * head + 0] = first_raw;
            value_bounds[4 * head + 1] = last_raw;
            value_bounds[4 * head + 2] = first_comp;
            value_bounds[4 * head + 3] = last_comp;
        }
    }

    float inv_denom[HEADS_PER_BLOCK];
#pragma unroll
    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
        inv_denom[j] = 1.0f / reduction[(size_t) j * N_THREADS];
    }

    int raw_first = raw_rows;
    int raw_last = -1;
    int comp_first = INDEXED_MASK ? indexed_count : n_kv;
    int comp_last = -1;
    int head_raw_first[HEADS_PER_BLOCK];
    int head_raw_last[HEADS_PER_BLOCK];
    int head_comp_first[HEADS_PER_BLOCK];
    int head_comp_last[HEADS_PER_BLOCK];
    // Finish the shared envelope before the value phase reads it.
    __syncthreads();
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

// Streaming indexed MLA for long prefill.  The compact grouped kernel above
// stores every score and then reloads V.  Once the trained indexer has reduced
// the compressed history to a bounded top-k set, that extra traffic is no
// longer necessary: stage one latent row in LDS, share it across the heads in
// a block, and update online-softmax state while the row is resident. Sixteen
// wave32 heads amortize each LDS load best on gfx1151. The contract is
// backend-generic (D512 MQA with K == V and direct indexed rows); model policy
// remains in the graph/backend layer.
template <typename KV, typename Mask, int HEADS_PER_BLOCK = 16,
          int KEYS_PER_STAGE = 16, bool STAGE_F32 = false,
          bool FAST_EXP = false, bool MASKLESS_CAUSAL = false>
__global__ static void ds4_flash_attn_d512_streaming_topk_kernel(
        float       * dst,
        const float * q,
        size_t        q_stride_token,
        size_t        q_stride_head,
        const KV    * kv,
        const Mask  * mask,
        const float * sinks,
        int           n_tokens,
        int           n_heads,
        int           n_kv,
        float         scale,
        const int   * visibility_bounds,
        const int   * indexed_rows,
        const int   * indexed_counts,
        int           indexed_capacity,
        ds4_inverse_rope_params inverse_rope,
        const float * inverse_rope_coefficients,
        const float * forward_rope_coefficients) {
    constexpr int D = 512;
    constexpr int WAVE = 32;
    constexpr int N_THREADS = HEADS_PER_BLOCK * WAVE;
    constexpr int VALUES_PER_LANE = D / WAVE;
    static_assert(HEADS_PER_BLOCK == 16);
    static_assert(KEYS_PER_STAGE == 16);
    static_assert(N_THREADS == 512);

    const int token = (int) blockIdx.x;
    const int head_begin = (int) blockIdx.y * HEADS_PER_BLOCK;
    const int tid = (int) threadIdx.x;
    const int wave = tid / WAVE;
    const int lane = tid & (WAVE - 1);
    const int head = head_begin + wave;

    // The launch gate makes the grid exact.  Keeping every thread live is
    // required because each stage has workgroup-wide barriers.
    if (token >= n_tokens || head_begin + HEADS_PER_BLOCK > n_heads) return;

    using stage_type = std::conditional_t<STAGE_F32, float, KV>;
    __shared__ __align__(16) stage_type staged_kv[KEYS_PER_STAGE * D];
    __shared__ int staged_rows[KEYS_PER_STAGE];
    __shared__ float staged_masks[KEYS_PER_STAGE];

    const int * token_visibility = visibility_bounds + (size_t) token * 4;
    const int raw_first = token_visibility[0];
    const int raw_last = token_visibility[1];
    const int raw_count = raw_last >= raw_first
        ? raw_last - raw_first + 1 : 0;
    const int indexed_count = indexed_counts[token];
    const int total_rows = raw_count + indexed_count;
    const int * token_rows = indexed_rows +
        (size_t) token * indexed_capacity;
    const Mask * token_mask = MASKLESS_CAUSAL
        ? nullptr : mask + (size_t) token * n_kv;

    const float * qh = q + (size_t) token * q_stride_token +
        (size_t) head * q_stride_head;
    float q_values[VALUES_PER_LANE];
    float accum[VALUES_PER_LANE] = {};
#pragma unroll
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int dim = lane + i * WAVE;
        float qv = qh[dim];
        if (inverse_rope.forward_q_enabled && dim >= D - 64) {
            const int pair = (dim - (D - 64)) / 2;
            const float x0 = qh[D - 64 + 2 * pair + 0];
            const float x1 = qh[D - 64 + 2 * pair + 1];
            const size_t coefficient_index =
                ((size_t) token * 32 + (size_t) pair) * 2;
            float y0;
            float y1;
            ds4_apply_inverse_rope_pair(
                x0, x1,
                forward_rope_coefficients[coefficient_index + 0],
                forward_rope_coefficients[coefficient_index + 1],
                y0, y1);
            qv = (dim & 1) == 0 ? y0 : y1;
        }
        q_values[i] = qv;
    }

    float row_max = -3.402823466e38f;
    float row_sum = 0.0f;
    for (int row_base = 0; row_base < total_rows;
         row_base += KEYS_PER_STAGE) {
        if (tid < KEYS_PER_STAGE) {
            const int selected = row_base + tid;
            int row = -1;
            if (selected < raw_count) {
                row = raw_first + selected;
            } else if (selected < total_rows) {
                row = token_rows[selected - raw_count];
            }
            staged_rows[tid] = row;
            if constexpr (MASKLESS_CAUSAL) {
                staged_masks[tid] = row >= 0 && row < n_kv
                    ? 0.0f : -3.402823466e38f;
            } else {
                staged_masks[tid] = row >= 0 && row < n_kv
                    ? ds4_fa_load<Mask, Mask>(token_mask + row)
                    : -3.402823466e38f;
            }
        }
        __syncthreads();

        // Convert aligned half2 pairs once while loading them. Every head in
        // the block then consumes the same F32 LDS values without repeating
        // half conversion in both the score and value passes.
        if constexpr (STAGE_F32 && std::is_same_v<KV, half>) {
            constexpr int PAIRS_PER_ROW = D / 2;
            for (int pair_index = tid;
                 pair_index < KEYS_PER_STAGE * PAIRS_PER_ROW;
                 pair_index += N_THREADS) {
                const int slot = pair_index / PAIRS_PER_ROW;
                const int pair = pair_index - slot * PAIRS_PER_ROW;
                const int row = staged_rows[slot];
                float2 unpacked = make_float2(0.0f, 0.0f);
                if (row >= 0 && row < n_kv) {
                    ds4_fa_load_pair(
                        kv + (size_t) row * D + 2 * pair,
                        unpacked.x, unpacked.y);
                }
                reinterpret_cast<float2 *>(staged_kv)[pair_index] = unpacked;
            }
        } else {
            for (int index = tid; index < KEYS_PER_STAGE * D;
                 index += N_THREADS) {
                const int slot = index / D;
                const int dim = index - slot * D;
                const int row = staged_rows[slot];
                staged_kv[index] = row >= 0 && row < n_kv
                    ? static_cast<stage_type>(kv[(size_t) row * D + dim])
                    : stage_type{};
            }
        }
        __syncthreads();

#pragma unroll
        for (int slot = 0; slot < KEYS_PER_STAGE; ++slot) {
            const int selected = row_base + slot;
            if (selected >= total_rows) continue;
            const float mask_value = staged_masks[slot];
            if (mask_value <= -1.0e20f) continue;

            float partial = 0.0f;
#pragma unroll
            for (int i = 0; i < VALUES_PER_LANE; ++i) {
                const int dim = lane + i * WAVE;
                partial += q_values[i] *
                    ds4_fa_load<stage_type, stage_type>(
                        staged_kv + slot * D + dim);
            }
            partial = warp_reduce_sum<WAVE>(partial);
            // XOR reduction can associate operands differently in each lane.
            // Broadcast lane zero so every output dimension advances one
            // identical softmax state.
            partial = __shfl_sync(0xffffffffu, partial, 0, WAVE);

            const float score = partial * scale + mask_value;
            const float next_max = fmaxf(row_max, score);
            float old_scale = 0.0f;
            float value_scale = 0.0f;
            if (lane == 0) {
                old_scale = row_sum == 0.0f
                    ? 0.0f
                    : (FAST_EXP
                        ? __expf(row_max - next_max)
                        : expf(row_max - next_max));
                value_scale = FAST_EXP
                    ? __expf(score - next_max)
                    : expf(score - next_max);
            }
            old_scale = __shfl_sync(0xffffffffu, old_scale, 0, WAVE);
            value_scale = __shfl_sync(0xffffffffu, value_scale, 0, WAVE);
            row_sum = row_sum * old_scale + value_scale;
            row_max = next_max;
#pragma unroll
            for (int i = 0; i < VALUES_PER_LANE; ++i) {
                const int dim = lane + i * WAVE;
                const float value = ds4_fa_load<stage_type, stage_type>(
                    staged_kv + slot * D + dim);
                accum[i] = accum[i] * old_scale + value_scale * value;
            }
        }
        __syncthreads();
    }

    if (sinks) {
        const float sink = sinks[head];
        const float next_max = fmaxf(row_max, sink);
        float old_scale = 0.0f;
        float sink_scale = 0.0f;
        if (lane == 0) {
            old_scale = row_sum == 0.0f
                ? 0.0f
                : (FAST_EXP
                    ? __expf(row_max - next_max)
                    : expf(row_max - next_max));
            sink_scale = FAST_EXP
                ? __expf(sink - next_max)
                : expf(sink - next_max);
        }
        old_scale = __shfl_sync(0xffffffffu, old_scale, 0, WAVE);
        sink_scale = __shfl_sync(0xffffffffu, sink_scale, 0, WAVE);
        row_sum = row_sum * old_scale + sink_scale;
#pragma unroll
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            accum[i] *= old_scale;
        }
    }

    const float inv_sum = row_sum == 0.0f ? 0.0f : 1.0f / row_sum;
    float * out = dst +
        ((size_t) token * (size_t) n_heads + (size_t) head) * D;
#pragma unroll
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int dim = lane + i * WAVE;
        float value = accum[i] * inv_sum;
        if (inverse_rope.enabled && dim >= D - 64) {
            const float partner = __shfl_xor_sync(
                0xffffffffu, value, 1, WAVE);
            const float x0 = (dim & 1) == 0 ? value : partner;
            const float x1 = (dim & 1) == 0 ? partner : value;
            const int pair = (dim - (D - 64)) / 2;
            const size_t coefficient_index =
                ((size_t) token * 32 + (size_t) pair) * 2;
            float y0;
            float y1;
            ds4_apply_inverse_rope_pair(
                x0, x1,
                inverse_rope_coefficients[coefficient_index + 0],
                inverse_rope_coefficients[coefficient_index + 1],
                y0, y1);
            value = (dim & 1) == 0 ? y0 : y1;
        }
        out[dim] = value;
    }
}

// Matrix-core D512 MLA for RDNA wave32. One workgroup evaluates one or two
// groups of 16 query heads against 16 latent rows at a time. Rows may come
// from an index or from contiguous visibility bounds. The waves split the
// 512-wide reduction, then reuse the staged latent tile for the value product.
// Softmax state and the final output stay in FP32; the exponentials use
// __expf like the qualified streaming top-k default on gfx1151.
template <int HEAD_GROUPS = 1, bool MASKLESS_CAUSAL = false,
          bool INDEXED_ROWS = true>
__global__ static void ds4_flash_attn_d512_streaming_wmma_kernel(
        float       * dst,
        const float * q,
        size_t        q_stride_token,
        size_t        q_stride_head,
        const half  * kv,
        const half  * mask,
        const float * sinks,
        int           n_tokens,
        int           n_heads,
        int           n_kv,
        float         scale,
        const int   * visibility_bounds,
        const int   * indexed_rows,
        const int   * indexed_counts,
        int           indexed_capacity,
        ds4_inverse_rope_params inverse_rope,
        const float * inverse_rope_coefficients,
        const float * forward_rope_coefficients) {
    constexpr int D = 512;
    constexpr int WAVE = 32;
    constexpr int HEADS = 16;
    constexpr int TOTAL_HEADS = HEAD_GROUPS * HEADS;
    constexpr int KEYS = 16;
    constexpr int NWAVES = 16;
    constexpr int N_THREADS = WAVE * NWAVES;
    constexpr int OUTPUT_TILES_PER_WAVE = 2;
    static_assert(HEAD_GROUPS == 1 || HEAD_GROUPS == 2);
    constexpr auto input_layout = get_input_data_layout();
    using input_tile = tile<16, 8, half2, input_layout>;
    using accum_tile = tile<16, 16, float, DATA_LAYOUT_J_MAJOR>;

    const int token = (int) blockIdx.x;
    const int head_begin = (int) blockIdx.y * TOTAL_HEADS;
    const int lane = (int) threadIdx.x;
    const int wave = (int) threadIdx.y;
    const int tid = wave * WAVE + lane;
    if (token >= n_tokens || head_begin + TOTAL_HEADS > n_heads) return;

    __shared__ __align__(16) half staged_rows_major[KEYS * D];
    __shared__ float partial_scores[NWAVES * KEYS * HEADS];
    __shared__ half softmax_weights[HEADS * KEYS];
    __shared__ int selected_row_ids[KEYS];
    __shared__ float selected_masks[KEYS];
    __shared__ float row_max[TOTAL_HEADS];
    __shared__ float row_sum[TOTAL_HEADS];
    __shared__ float old_scale[TOTAL_HEADS];

    accum_tile output_acc[HEAD_GROUPS][OUTPUT_TILES_PER_WAVE];
    if (tid < TOTAL_HEADS) {
        row_max[tid] = -3.402823466e38f;
        row_sum[tid] = 0.0f;
        old_scale[tid] = 0.0f;
    }
    __syncthreads();

    const int * token_visibility = visibility_bounds + (size_t) token * 4;
    const int raw_first = token_visibility[0];
    const int raw_last = token_visibility[1];
    const int raw_count = raw_last >= raw_first
        ? raw_last - raw_first + 1 : 0;
    const int comp_first = token_visibility[2];
    const int comp_last = token_visibility[3];
    const int comp_count = comp_last >= comp_first
        ? comp_last - comp_first + 1 : 0;
    const int selected_count = INDEXED_ROWS
        ? indexed_counts[token] : comp_count;
    const int total_rows = raw_count + selected_count;
    const int * token_rows = INDEXED_ROWS
        ? indexed_rows + (size_t) token * indexed_capacity : nullptr;
    const half * token_mask = MASKLESS_CAUSAL
        ? nullptr : mask + (size_t) token * n_kv;
    const float * token_q = q + (size_t) token * q_stride_token;

    // Q is invariant across every selected-row stage. Keep the WMMA
    // fragments resident instead of reloading and reconverting it roughly
    // forty times for a 640-row sparse window.
    input_tile query_tiles[HEAD_GROUPS][OUTPUT_TILES_PER_WAVE];
#pragma unroll
    for (int head_group = 0; head_group < HEAD_GROUPS; ++head_group) {
#pragma unroll
        for (int slice = 0; slice < OUTPUT_TILES_PER_WAVE; ++slice) {
            const int dim_base =
                wave * OUTPUT_TILES_PER_WAVE * 16 + slice * 16;
#pragma unroll
            for (int element = 0; element < input_tile::ne; ++element) {
                const int local_head = input_tile::get_i(element);
                const int pair = input_tile::get_j(element);
                const int dim = dim_base + 2 * pair;
                const float * qh = token_q +
                    (size_t) (head_begin + head_group * HEADS + local_head) *
                    q_stride_head;
                float q0 = qh[dim + 0];
                float q1 = qh[dim + 1];
                if (inverse_rope.forward_q_enabled && dim >= D - 64) {
                    const int rope_pair = (dim - (D - 64)) / 2;
                    const size_t coefficient_index =
                        ((size_t) token * 32 + (size_t) rope_pair) * 2;
                    ds4_apply_inverse_rope_pair(
                        q0, q1,
                        forward_rope_coefficients[coefficient_index + 0],
                        forward_rope_coefficients[coefficient_index + 1],
                        q0, q1);
                }
                query_tiles[head_group][slice].x[element] =
                    __floats2half2_rn(q0, q1);
            }
        }
    }

    for (int row_base = 0; row_base < total_rows; row_base += KEYS) {
        if (tid < KEYS) {
            const int selected = row_base + tid;
            int row = -1;
            if (selected < raw_count) {
                row = raw_first + selected;
            } else if (selected < total_rows) {
                if constexpr (INDEXED_ROWS) {
                    row = token_rows[selected - raw_count];
                } else {
                    row = comp_first + selected - raw_count;
                }
            }
            selected_row_ids[tid] = row;
            if constexpr (MASKLESS_CAUSAL) {
                selected_masks[tid] = row >= 0 && row < n_kv
                    ? 0.0f : -3.402823466e38f;
            } else {
                selected_masks[tid] = row >= 0 && row < n_kv
                    ? __half2float(token_mask[row])
                    : -3.402823466e38f;
            }
        }
        __syncthreads();

        for (int index = tid; index < KEYS * D; index += N_THREADS) {
            const int key = index / D;
            const int dim = index - key * D;
            const int row = selected_row_ids[key];
            staged_rows_major[index] = row >= 0 && row < n_kv
                ? kv[(size_t) row * D + dim] : half{};
        }
        __syncthreads();

        for (int head_group = 0; head_group < HEAD_GROUPS; ++head_group) {
            input_tile key_tile;
            accum_tile score_acc;
#pragma unroll
            for (int slice = 0; slice < OUTPUT_TILES_PER_WAVE; ++slice) {
                const int dim_base =
                    wave * OUTPUT_TILES_PER_WAVE * 16 + slice * 16;
                load_generic(
                    key_tile,
                    reinterpret_cast<const half2 *>(staged_rows_major) +
                        dim_base / 2,
                    D / 2);
                mma(score_acc, key_tile, query_tiles[head_group][slice]);
            }
#pragma unroll
            for (int element = 0; element < accum_tile::ne; ++element) {
                const int key = accum_tile::get_i(element);
                const int local_head = accum_tile::get_j(element);
                partial_scores[(wave * KEYS + key) * HEADS + local_head] =
                    score_acc.x[element];
            }
            __syncthreads();

            if (tid < KEYS * HEADS) {
                const int key = tid / HEADS;
                const int local_head = tid - key * HEADS;
                float score = 0.0f;
#pragma unroll
                for (int source_wave = 0; source_wave < NWAVES;
                     ++source_wave) {
                    score += partial_scores[
                        (source_wave * KEYS + key) * HEADS + local_head];
                }
                partial_scores[key * HEADS + local_head] =
                    score * scale + selected_masks[key];
            }
            __syncthreads();

            if (tid < HEADS) {
                const int local_head = tid;
                const int state = head_group * HEADS + local_head;
                float next_max = row_max[state];
#pragma unroll
                for (int key = 0; key < KEYS; ++key) {
                    next_max = fmaxf(
                        next_max,
                        partial_scores[key * HEADS + local_head]);
                }
                const float rescale = row_sum[state] == 0.0f
                    ? 0.0f
                    : __expf(row_max[state] - next_max);
                float sum = row_sum[state] * rescale;
#pragma unroll
                for (int key = 0; key < KEYS; ++key) {
                    const float score =
                        partial_scores[key * HEADS + local_head];
                    const float weight = score <= -1.0e20f
                        ? 0.0f
                        : __expf(score - next_max);
                    softmax_weights[local_head * KEYS + key] =
                        __float2half(weight);
                    sum += weight;
                }
                old_scale[state] = rescale;
                row_max[state] = next_max;
                row_sum[state] = sum;
            }
            __syncthreads();

            input_tile weight_tile;
            load_generic(
                weight_tile,
                reinterpret_cast<const half2 *>(softmax_weights),
                KEYS / 2);
#pragma unroll
            for (int output_slice = 0;
                 output_slice < OUTPUT_TILES_PER_WAVE; ++output_slice) {
#pragma unroll
                for (int element = 0; element < accum_tile::ne; ++element) {
                    const int local_head = accum_tile::get_i(element);
                    const int state = head_group * HEADS + local_head;
                    output_acc[head_group][output_slice].x[element] *=
                        old_scale[state];
                }
                input_tile value_tile;
                const int dim_base = wave * OUTPUT_TILES_PER_WAVE * 16 +
                    output_slice * 16;
                // Gather the value tile straight from the row-major stage;
                // a dims-major copy costs 16 KiB of LDS for no measured gain.
#pragma unroll
                for (int element = 0; element < input_tile::ne;
                     ++element) {
                    const int local_dim = input_tile::get_i(element);
                    const int key_pair = input_tile::get_j(element);
                    const int dim = dim_base + local_dim;
                    value_tile.x[element] = __halves2half2(
                        staged_rows_major[(2 * key_pair + 0) * D + dim],
                        staged_rows_major[(2 * key_pair + 1) * D + dim]);
                }
                mma(output_acc[head_group][output_slice],
                    weight_tile, value_tile);
            }
            __syncthreads();
        }
    }

    if (sinks && tid < TOTAL_HEADS) {
        const int state = tid;
        const float sink = sinks[head_begin + state];
        const float next_max = fmaxf(row_max[state], sink);
        const float rescale = row_sum[state] == 0.0f
            ? 0.0f
            : __expf(row_max[state] - next_max);
        const float sink_weight = __expf(sink - next_max);
        row_sum[state] = row_sum[state] * rescale + sink_weight;
        row_max[state] = next_max;
        old_scale[state] = rescale;
    } else if (!sinks && tid < TOTAL_HEADS) {
        old_scale[tid] = 1.0f;
    }
    __syncthreads();

#pragma unroll
    for (int head_group = 0; head_group < HEAD_GROUPS; ++head_group) {
#pragma unroll
        for (int output_slice = 0;
             output_slice < OUTPUT_TILES_PER_WAVE; ++output_slice) {
#pragma unroll
            for (int element = 0; element < accum_tile::ne; ++element) {
                const int local_head = accum_tile::get_i(element);
                const int state = head_group * HEADS + local_head;
                const int local_dim = accum_tile::get_j(element);
                const int dim = wave * OUTPUT_TILES_PER_WAVE * 16 +
                    output_slice * 16 + local_dim;
                float value = row_sum[state] > 0.0f
                    ? output_acc[head_group][output_slice].x[element] *
                      old_scale[state] / row_sum[state]
                    : 0.0f;
                if (inverse_rope.enabled && dim >= D - 64) {
                    const float partner = __shfl_xor_sync(
                        0xffffffffu, value, 1, WAVE);
                    const float x0 = (dim & 1) == 0 ? value : partner;
                    const float x1 = (dim & 1) == 0 ? partner : value;
                    const int rope_pair = (dim - (D - 64)) / 2;
                    const size_t coefficient_index =
                        ((size_t) token * 32 + (size_t) rope_pair) * 2;
                    float y0;
                    float y1;
                    ds4_apply_inverse_rope_pair(
                        x0, x1,
                        inverse_rope_coefficients[coefficient_index + 0],
                        inverse_rope_coefficients[coefficient_index + 1],
                        y0, y1);
                    value = (dim & 1) == 0 ? y0 : y1;
                }
                dst[((size_t) token * n_heads + head_begin + state) * D +
                    dim] = value;
            }
        }
    }
}

template <typename KV>
static __device__ __forceinline__ const KV * ds4_fa_segmented_row(
        const KV * raw,
        const KV * compressed,
        const KV * preserved_tail,
        int        row,
        int        raw_rows,
        int        compressed_rows) {
    constexpr int D = 512;
    if (row < raw_rows) {
        return raw + (size_t) row * D;
    }
    row -= raw_rows;
    if (row < compressed_rows) {
        return compressed + (size_t) row * D;
    }
    return preserved_tail + (size_t) (row - compressed_rows) * D;
}

// Split-KV decode for indexed MLA.  A single grouped block leaves most of a
// wide RDNA GPU idle when q is small: DS4 has 64 heads, so the four-head
// kernel exposes only 16 blocks per layer.  Split the bounded raw+top-k row
// set across four blocks, retain online-softmax state per split, then combine
// the states in a second kernel.  The implementation is expressed in terms of
// the D512 latent-attention contract rather than model weights, so future MLA
// models using the same layout can reuse it.
template <typename KV, typename Mask, int HEADS_PER_BLOCK, int N_SPLITS>
__global__ static void ds4_flash_attn_d512_indexed_split_stage1_kernel(
        float       * partial,
        const float * q,
        size_t        q_stride_token,
        size_t        q_stride_head,
        const KV    * k,
        const KV    * v,
        const KV    * compressed_kv,
        const KV    * preserved_tail_kv,
        const Mask  * mask,
        const float * sinks,
        int           n_tokens,
        int           n_heads,
        int           n_kv,
        int           materialized_kv_rows,
        int           compressed_kv_rows,
        float         scale,
        int           raw_rows,
        int           split_stride,
        const int   * visibility_bounds,
        const int   * indexed_rows,
        const int   * indexed_counts,
        int           indexed_capacity,
        ds4_inverse_rope_params inverse_rope,
        const float * forward_rope_coefficients) {
    constexpr int D = 512;
    constexpr int N_THREADS = 256;
    constexpr int VALUES_PER_THREAD = 4;

    const int token = (int) blockIdx.x;
    const int head_begin = (int) blockIdx.y * HEADS_PER_BLOCK;
    const int split = (int) blockIdx.z;
    const int tid = (int) threadIdx.x;
    if (token >= n_tokens || head_begin >= n_heads) return;

    extern __shared__ float scratch[];
    float * scores = scratch;
    float * reduction = scores + (size_t) HEADS_PER_BLOCK * split_stride;
    float * q_rope_tail = reduction + (size_t) HEADS_PER_BLOCK * N_THREADS;

    const int raw_first = visibility_bounds
        ? visibility_bounds[(size_t) token * 4 + 0] : 0;
    const int raw_last = visibility_bounds
        ? visibility_bounds[(size_t) token * 4 + 1] : raw_rows - 1;
    const int raw_count = raw_last >= raw_first
        ? raw_last - raw_first + 1 : 0;
    const int indexed_count = indexed_counts[token];
    const int total_rows = raw_count + indexed_count;
    const int rows_per_split = (total_rows + N_SPLITS - 1) / N_SPLITS;
    const int split_begin = split * rows_per_split;
    const int split_end = min(total_rows, split_begin + rows_per_split);
    const int * token_indexed_rows = indexed_rows +
        (size_t) token * indexed_capacity;

    const float * qh[HEADS_PER_BLOCK];
#pragma unroll
    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
        qh[j] = q + (size_t) token * q_stride_token +
                (size_t) (head_begin + j) * q_stride_head;
    }

    if (inverse_rope.forward_q_enabled) {
        for (int index = tid; index < HEADS_PER_BLOCK * 64;
             index += N_THREADS) {
            const int j = index / 64;
            const int tail_d = index % 64;
            const int pair = tail_d >> 1;
            const float x0 = qh[j][D - 64 + 2 * pair + 0];
            const float x1 = qh[j][D - 64 + 2 * pair + 1];
            const size_t coefficient =
                ((size_t) token * 32 + (size_t) pair) * 2;
            const float cos_theta = forward_rope_coefficients[coefficient + 0];
            const float sin_theta = forward_rope_coefficients[coefficient + 1];
            q_rope_tail[(size_t) j * 64 + tail_d] = (tail_d & 1)
                ? x0 * sin_theta + x1 * cos_theta
                : x0 * cos_theta - x1 * sin_theta;
        }
        __syncthreads();
    }

    float local_max[HEADS_PER_BLOCK];
#pragma unroll
    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
        local_max[j] = split == 0 && sinks
            ? sinks[head_begin + j] : -3.402823466e38f;
    }

    for (int slot = split_begin + tid; slot < split_end;
         slot += N_THREADS) {
        const int row = slot < raw_count
            ? raw_first + slot
            : token_indexed_rows[slot - raw_count];
        const float mask_v = ds4_fa_load<Mask, Mask>(
            mask + (size_t) token * n_kv + row);
        const KV * kr = compressed_kv
            ? ds4_fa_segmented_row(
                  k, compressed_kv, preserved_tail_kv, row,
                  materialized_kv_rows, compressed_kv_rows)
            : k + (size_t) row * D;
        float dot[HEADS_PER_BLOCK] = {};
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
        const int score_index = slot - split_begin;
#pragma unroll
        for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
            const float score = dot[j] * scale + mask_v;
            scores[(size_t) j * split_stride + score_index] = score;
            local_max[j] = fmaxf(local_max[j], score);
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

    float split_max[HEADS_PER_BLOCK];
    float local_sum[HEADS_PER_BLOCK] = {};
#pragma unroll
    for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
        split_max[j] = reduction[(size_t) j * N_THREADS];
    }
    // Every wave must finish reading the maxima before the scratch rows are
    // reused for sums. A faster wave can otherwise overwrite another's max.
    __syncthreads();
    for (int slot = split_begin + tid; slot < split_end;
         slot += N_THREADS) {
        const int score_index = slot - split_begin;
#pragma unroll
        for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
            float * score = scores + (size_t) j * split_stride + score_index;
            const float weight = expf(*score - split_max[j]);
            *score = weight;
            local_sum[j] += weight;
        }
    }
    if (tid == 0 && split == 0 && sinks) {
#pragma unroll
        for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
            local_sum[j] += expf(sinks[head_begin + j] - split_max[j]);
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

    const int d0 = VALUES_PER_THREAD * tid;
    if (d0 < D) {
        float acc0[HEADS_PER_BLOCK] = {};
        float acc1[HEADS_PER_BLOCK] = {};
        float acc2[HEADS_PER_BLOCK] = {};
        float acc3[HEADS_PER_BLOCK] = {};
        for (int slot = split_begin; slot < split_end; ++slot) {
            const int row = slot < raw_count
                ? raw_first + slot
                : token_indexed_rows[slot - raw_count];
            float vv0, vv1, vv2, vv3;
            const KV * vr = compressed_kv
                ? ds4_fa_segmented_row(
                      v, compressed_kv, preserved_tail_kv, row,
                      materialized_kv_rows, compressed_kv_rows)
                : v + (size_t) row * D;
            ds4_fa_load_quad<KV>(vr + d0,
                                 vv0, vv1, vv2, vv3);
            const int score_index = slot - split_begin;
#pragma unroll
            for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
                const float weight =
                    scores[(size_t) j * split_stride + score_index];
                acc0[j] += weight * vv0;
                acc1[j] += weight * vv1;
                acc2[j] += weight * vv2;
                acc3[j] += weight * vv3;
            }
        }
#pragma unroll
        for (int j = 0; j < HEADS_PER_BLOCK; ++j) {
            float * out = partial +
                (((size_t) token * n_heads + head_begin + j) * N_SPLITS +
                 split) * (D + 2);
            out[d0 + 0] = acc0[j];
            out[d0 + 1] = acc1[j];
            out[d0 + 2] = acc2[j];
            out[d0 + 3] = acc3[j];
        }
    }
    if (tid < HEADS_PER_BLOCK) {
        float * out = partial +
            (((size_t) token * n_heads + head_begin + tid) * N_SPLITS +
             split) * (D + 2);
        out[D + 0] = split_max[tid];
        out[D + 1] = reduction[(size_t) tid * N_THREADS];
    }
}

template <int N_SPLITS>
__global__ static void ds4_flash_attn_d512_indexed_split_stage2_kernel(
        float       * dst,
        const float * partial,
        int           n_tokens,
        int           n_heads,
        ds4_inverse_rope_params inverse_rope,
        const float * inverse_rope_coefficients) {
    constexpr int D = 512;
    const int token = (int) blockIdx.x;
    const int head = (int) blockIdx.y;
    const int tid = (int) threadIdx.x;
    if (token >= n_tokens || head >= n_heads) return;

    __shared__ float global_max;
    __shared__ float inverse_denom;
    const float * head_partial = partial +
        ((size_t) token * n_heads + head) * N_SPLITS * (D + 2);
    if (tid == 0) {
        float max_value = -3.402823466e38f;
#pragma unroll
        for (int split = 0; split < N_SPLITS; ++split) {
            max_value = fmaxf(max_value,
                head_partial[(size_t) split * (D + 2) + D]);
        }
        float denom = 0.0f;
#pragma unroll
        for (int split = 0; split < N_SPLITS; ++split) {
            const float * state = head_partial + (size_t) split * (D + 2);
            denom += state[D + 1] * expf(state[D] - max_value);
        }
        global_max = max_value;
        inverse_denom = 1.0f / denom;
    }
    __syncthreads();

    const int d0 = 2 * tid;
    float x0 = 0.0f;
    float x1 = 0.0f;
#pragma unroll
    for (int split = 0; split < N_SPLITS; ++split) {
        const float * state = head_partial + (size_t) split * (D + 2);
        const float rescale = expf(state[D] - global_max);
        x0 += state[d0 + 0] * rescale;
        x1 += state[d0 + 1] * rescale;
    }
    x0 *= inverse_denom;
    x1 *= inverse_denom;

    float * out = dst + ((size_t) token * n_heads + head) * D + d0;
    if (inverse_rope.enabled && d0 >= D - 64) {
        const int pair = (d0 - (D - 64)) / 2;
        const size_t coefficient =
            ((size_t) token * 32 + (size_t) pair) * 2;
        const float cos_theta = inverse_rope_coefficients[coefficient + 0];
        const float sin_theta = inverse_rope_coefficients[coefficient + 1];
        float y0;
        float y1;
        ds4_apply_inverse_rope_pair(
            x0, x1, cos_theta, sin_theta, y0, y1);
        out[0] = y0;
        out[1] = y1;
    } else {
        out[0] = x0;
        out[1] = x1;
    }
}

template <typename KV, typename Mask, int N_SPLITS>
static void ds4_launch_flash_attn_d512_indexed_split(
        float             * dst,
        float             * partial,
        const float       * q,
        size_t              q_stride_token,
        size_t              q_stride_head,
        const KV          * k,
        const KV          * v,
        const KV          * compressed_kv,
        const KV          * preserved_tail_kv,
        const Mask        * mask,
        const float       * sinks,
        int                 n_tokens,
        int                 n_heads,
        int                 n_kv,
        int                 materialized_kv_rows,
        int                 compressed_kv_rows,
        float               scale,
        int                 raw_rows,
        int                 split_stride,
        const int         * visibility_bounds,
        const int         * indexed_rows,
        const int         * indexed_counts,
        int                 indexed_capacity,
        ds4_inverse_rope_params inverse_rope,
        const float       * inverse_rope_coefficients,
        const float       * forward_rope_coefficients,
        cudaStream_t        stream) {
    constexpr int HEADS_PER_BLOCK = 4;
    const size_t shmem =
        ((size_t) HEADS_PER_BLOCK * split_stride +
         (size_t) HEADS_PER_BLOCK * 256 +
         (inverse_rope.forward_q_enabled
             ? (size_t) HEADS_PER_BLOCK * 64 : 0)) * sizeof(float);
    const dim3 stage1_grid(
        (unsigned) n_tokens,
        (unsigned) (n_heads / HEADS_PER_BLOCK),
        (unsigned) N_SPLITS);
    ds4_flash_attn_d512_indexed_split_stage1_kernel<
        KV, Mask, HEADS_PER_BLOCK, N_SPLITS>
        <<<stage1_grid, 256, shmem, stream>>>(
            partial, q, q_stride_token, q_stride_head, k, v,
            compressed_kv, preserved_tail_kv, mask, sinks,
            n_tokens, n_heads, n_kv, materialized_kv_rows,
            compressed_kv_rows, scale, raw_rows, split_stride,
            visibility_bounds, indexed_rows, indexed_counts,
            indexed_capacity, inverse_rope, forward_rope_coefficients);
    const dim3 stage2_grid((unsigned) n_tokens, (unsigned) n_heads, 1);
    ds4_flash_attn_d512_indexed_split_stage2_kernel<N_SPLITS>
        <<<stage2_grid, 256, 0, stream>>>(
            dst, partial, n_tokens, n_heads, inverse_rope,
            inverse_rope_coefficients);
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

template <int HEADS_PER_BLOCK, bool INDEXED_MASK, bool MASKLESS_CAUSAL,
          int VALUES_PER_THREAD>
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
    GGML_ASSERT((MASKLESS_CAUSAL || mask) && visibility_bounds);
    if constexpr (INDEXED_MASK) {
        GGML_ASSERT(indexed_rows && indexed_counts &&
                    indexed_owner_offsets && indexed_owner_ranks);
    }
    dim3 grid(
        (unsigned) n_tokens,
        (unsigned) (n_heads / HEADS_PER_BLOCK), 1);
    if (kv_f16 && (MASKLESS_CAUSAL || mask->type == GGML_TYPE_F16)) {
        ds4_flash_attn_d512_shared_kv_grouped_compact_kernel<
            half, half, HEADS_PER_BLOCK, INDEXED_MASK, MASKLESS_CAUSAL,
            VALUES_PER_THREAD>
            <<<grid, 256, shmem, stream>>>(
                (float *) dst->data, (const float *) Q->data,
                q_stride_token, q_stride_head,
                (const half *) K->data, (const half *) V->data,
                mask ? (const half *) mask->data : nullptr,
                sinks ? (const float *) sinks->data : nullptr,
                n_tokens, n_heads, n_kv, scale, raw_rows,
                raw_score_capacity, score_stride, visibility_bounds,
                indexed_rows, indexed_counts,
                indexed_owner_offsets, indexed_owner_ranks, indexed_capacity,
                inverse_rope, inverse_rope_coefficients,
                forward_rope_coefficients);
    } else if (kv_f32 &&
               (MASKLESS_CAUSAL || mask->type == GGML_TYPE_F32)) {
        ds4_flash_attn_d512_shared_kv_grouped_compact_kernel<
            float, float, HEADS_PER_BLOCK, INDEXED_MASK, MASKLESS_CAUSAL,
            VALUES_PER_THREAD>
            <<<grid, 256, shmem, stream>>>(
                (float *) dst->data, (const float *) Q->data,
                q_stride_token, q_stride_head,
                (const float *) K->data, (const float *) V->data,
                mask ? (const float *) mask->data : nullptr,
                sinks ? (const float *) sinks->data : nullptr,
                n_tokens, n_heads, n_kv, scale, raw_rows,
                raw_score_capacity, score_stride, visibility_bounds,
                indexed_rows, indexed_counts,
                indexed_owner_offsets, indexed_owner_ranks, indexed_capacity,
                inverse_rope, inverse_rope_coefficients,
                forward_rope_coefficients);
    } else if (!MASKLESS_CAUSAL && kv_f32 &&
               mask->type == GGML_TYPE_F16) {
        ds4_flash_attn_d512_shared_kv_grouped_compact_kernel<
            float, half, HEADS_PER_BLOCK, INDEXED_MASK, MASKLESS_CAUSAL,
            VALUES_PER_THREAD>
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

// The rocWMMA D512 kernels, their gfx1151 defaults and the split-KV decode
// schedule were qualified on Strix Halo only (wave32, ROCm 7.2). Widen this
// to a device class once another RDNA 3.5 part has been measured.
static bool ds4_fa_is_gfx1151(const int cc) {
    return cc == GGML_CUDA_CC_OFFSET_AMD + 0x1151;
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
    const ggml_tensor * indexer_topk = dst->src[5];
    // src[6] carries the optional DS4 rope positions
    // (ggml_flash_attn_ext_set_ds4_rope_positions); the optional segmented-KV
    // operands set by ggml_flash_attn_ext_set_ds4_kv_segments are src[7]/src[8].
    const ggml_tensor * kv_compressed = dst->src[7];
    const ggml_tensor * kv_preserved_tail = dst->src[8];
    const bool segmented_kv = kv_compressed && kv_preserved_tail;
    const bool ratio4_causal = indexer_topk && !mask;
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
    if ((kv_compressed == nullptr) != (kv_preserved_tail == nullptr) ||
        !Q || !K || !V ||
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
    if (segmented_kv &&
        (!kv_f16 || K != V || kv_compressed->type != K->type ||
         kv_preserved_tail->type != K->type ||
         kv_compressed->ne[0] != K->ne[0] ||
         kv_preserved_tail->ne[0] != K->ne[0] ||
         kv_compressed->ne[1] <= 0 || kv_preserved_tail->ne[1] <= 0 ||
         kv_compressed->ne[2] != 1 || kv_compressed->ne[3] != 1 ||
         kv_preserved_tail->ne[2] != 1 ||
         kv_preserved_tail->ne[3] != 1 ||
         !ggml_is_contiguous(kv_compressed) ||
         !ggml_is_contiguous(kv_preserved_tail))) {
        return false;
    }
    const size_t kv_esz = kv_f16 ? sizeof(half) : sizeof(float);
    if (K->nb[0] != kv_esz || V->nb[0] != kv_esz ||
        K->nb[1] != (size_t) K->ne[0] * kv_esz ||
        V->nb[1] != (size_t) V->ne[0] * kv_esz ||
        Q->nb[1] % sizeof(float) != 0 ||
        Q->nb[2] % sizeof(float) != 0 ||
        (mask && (mask->ne[0] !=
                    K->ne[1] +
                        (segmented_kv ? kv_compressed->ne[1] +
                                            kv_preserved_tail->ne[1]
                                      : 0) ||
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
    const int materialized_kv_rows = (int) K->ne[1];
    const int n_kv = materialized_kv_rows +
        (segmented_kv
             ? (int) (kv_compressed->ne[1] + kv_preserved_tail->ne[1])
             : 0);
    if (n_tokens <= 0 || n_heads <= 0 || n_kv <= 0) {
        return false;
    }

    const int raw_rows = ggml_get_op_params_i32(dst, 4);
    const int sparse_keep_rows = ggml_get_op_params_i32(dst, 5);
    const unsigned int ds4_layout =
        (unsigned int) ggml_get_op_params_i32(dst, 6);
    const int raw_window = (int) (ds4_layout >> 16);
    const int sparse_block_size = (int) (ds4_layout & 0xffffu);
    const uint32_t packed_rope_flags =
        (uint32_t) ggml_get_op_params_i32(dst, 7);
    const int rope_flags = (int) (packed_rope_flags & 0xffffu);
    const int causal_ratio = (int) (packed_rope_flags >> 16);
    const ggml_tensor * rope_positions = dst->src[6];
    if (rope_positions &&
        ((rope_flags & 1) == 0 || causal_ratio != 0 ||
         rope_positions->type != GGML_TYPE_I32 ||
         rope_positions->ne[0] != n_tokens || rope_positions->ne[1] != 1 ||
         rope_positions->ne[2] != 1 || rope_positions->ne[3] != 1 ||
         !ggml_is_contiguous(rope_positions))) {
        return false;
    }
    if (sparse_keep_rows == INT_MIN) {
        return false;
    }
    if (segmented_kv &&
        (!mask || n_tokens > 8 || sparse_keep_rows >= 0 ||
         raw_rows != materialized_kv_rows)) {
        return false;
    }
    if (raw_rows < 0 || raw_rows > n_kv ||
        (ds4_layout != 0 && (raw_window <= 0 || sparse_block_size <= 0)) ||
        (sparse_keep_rows != 0 && !mask && !ratio4_causal &&
         causal_ratio == 0) ||
        (rope_flags & ~3) != 0 ||
        ((rope_flags & 2) != 0 && (rope_flags & 1) == 0)) {
        return false;
    }
    const int n_comp_rows = n_kv - raw_rows;
    if (causal_ratio > 0) {
        const int kv_start = ggml_get_op_params_i32(dst, 8);
        const int prior_rows = raw_rows - n_tokens;
        if (mask || indexer_topk || n_tokens <= raw_window ||
            prior_rows < 0 ||
            prior_rows != std::min(kv_start, raw_window) ||
            (n_comp_rows == 0 && causal_ratio != 1) ||
            (n_comp_rows > 0 &&
             (causal_ratio <= 1 || (rope_flags & 1) == 0 ||
              n_comp_rows != (kv_start + n_tokens) / causal_ratio))) {
            return false;
        }
    }
    if (indexer_topk &&
        (sparse_keep_rows >= 0 || -sparse_keep_rows > 1024 ||
         -sparse_keep_rows > n_comp_rows ||
         indexer_topk->type != GGML_TYPE_I32 ||
         indexer_topk->ne[0] != -sparse_keep_rows ||
         indexer_topk->ne[1] != Q->ne[1] ||
         indexer_topk->ne[2] != 1 || indexer_topk->ne[3] != 1 ||
         !ggml_is_contiguous(indexer_topk))) {
        return false;
    }
    if (indexer_topk) {
        constexpr int group4 = 4;
        const int indexed_capacity = -sparse_keep_rows;
        const int requested_raw_window =
            raw_window > 0 ? raw_window : raw_rows;
        const int effective_raw_window =
            std::max(1, std::min(requested_raw_window, raw_rows));
        const int compact_score_stride =
            effective_raw_window + indexed_capacity;
        const size_t compact_group4_shmem =
            ((size_t) group4 * compact_score_stride +
             (size_t) group4 * 256) * sizeof(float) +
            (size_t) group4 * 4 * sizeof(int) +
            ((rope_flags & 2) != 0
                ? (size_t) group4 * 64 * sizeof(float) : 0);
        if (raw_rows <= 0 || n_comp_rows <= 0 ||
            n_heads % group4 != 0 || compact_group4_shmem > 24 * 1024) {
            return false;
        }
        if (ratio4_causal) {
            const int kv_start = ggml_get_op_params_i32(dst, 8);
            const int prior_rows = raw_rows - n_tokens;
            if ((rope_flags & 1) == 0 || rope_positions ||
                kv_start < 0 || raw_window <= 0 || indexed_capacity > 512 ||
                n_tokens <= raw_window ||
                prior_rows != std::min(kv_start, raw_window) ||
                n_comp_rows != (kv_start + n_tokens) / 4) {
                return false;
            }
        }
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
    const ggml_tensor * indexer_topk = dst->src[5];
    // Segmented KV lives in src[7]/src[8]; src[6] is the rope positions.
    const ggml_tensor * kv_compressed = dst->src[7];
    const ggml_tensor * kv_preserved_tail = dst->src[8];
    const bool segmented_kv = kv_compressed && kv_preserved_tail;
    const bool ratio4_causal = indexer_topk && !mask;
    const bool kv_f32 = K->type == GGML_TYPE_F32;
    const bool kv_f16 = K->type == GGML_TYPE_F16;
    const int n_tokens = (int) Q->ne[1];
    const int n_heads = (int) Q->ne[2];
    const int materialized_kv_rows = (int) K->ne[1];
    const int compressed_kv_rows = segmented_kv
        ? (int) kv_compressed->ne[1] : 0;
    const int preserved_tail_rows = segmented_kv
        ? (int) kv_preserved_tail->ne[1] : 0;
    const int n_kv = materialized_kv_rows + compressed_kv_rows +
        preserved_tail_rows;
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
    const bool sparse_requested = sparse_keep_rows > 0 &&
                                  sparse_keep_rows < n_comp_rows &&
                                  n_comp_blocks > 0;
    const bool indexed_mask = sparse_keep_rows < 0 && n_comp_rows > 0;
    const int indexed_capacity = indexed_mask
        ? -sparse_keep_rows : 0;
    const uint32_t packed_rope_flags =
        (uint32_t) ggml_get_op_params_i32(dst, 7);
    const int rope_flags = (int) (packed_rope_flags & 0xffffu);
    const int causal_ratio = (int) (packed_rope_flags >> 16);
    const int causal_kv_start = ggml_get_op_params_i32(dst, 8);
    const int prior_rows = raw_rows - n_tokens;
    const bool contiguous_causal = !mask && !indexer_topk &&
        causal_ratio > 0 && n_tokens > raw_window &&
        prior_rows >= 0 && prior_rows == min(causal_kv_start, raw_window) &&
        ((n_comp_rows == 0 && causal_ratio == 1) ||
         (n_comp_rows > 0 && causal_ratio > 1 &&
          (rope_flags & 1) != 0 &&
          n_comp_rows ==
              (causal_kv_start + n_tokens) / causal_ratio));
    // The GGML_CUDA_MLA_* switches are read per launch on purpose: the unit
    // tests toggle them between graph computes inside one process. The two
    // dense switches default off here and are enabled by the gfx1151 device
    // profile, so =0 is their kill switch.
    const bool dense_high_ratio =
        ds4_env_flag_enabled("GGML_CUDA_MLA_DENSE_HIGH_RATIO");
    const bool dense_wmma_enabled =
        ds4_env_flag_enabled("GGML_CUDA_MLA_DENSE_WMMA");
    const auto & device_info =
        ggml_cuda_info().devices[ggml_cuda_get_device()];
    const bool bypass_sparse_selector = sparse_requested &&
        dense_high_ratio && dense_wmma_enabled && contiguous_causal &&
        causal_ratio >= 64 && kv_f16 && K->data == V->data &&
        n_heads % 32 == 0 && n_tokens >= 64 &&
        device_info.warp_size == 32 &&
        ds4_fa_is_gfx1151(device_info.cc);
    const bool sparse = sparse_requested && !bypass_sparse_selector;

    ds4_inverse_rope_params inverse_rope{};
    inverse_rope.enabled = rope_flags & 1;
    inverse_rope.forward_q_enabled = (rope_flags & 2) != 0;
    if (rope_flags != 0) {
        inverse_rope.kv_start = ggml_get_op_params_i32(dst, 8);
        inverse_rope.positions = dst->src[6]
            ? static_cast<const int32_t *>(dst->src[6]->data) : nullptr;
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
    // Default on for gfx1151; GGML_CUDA_MLA_SPARSE_VALUE_SKIP=0 disables it.
    const bool skip_sparse_value_gaps =
        getenv("GGML_CUDA_MLA_SPARSE_VALUE_SKIP")
            ? ds4_env_flag_enabled("GGML_CUDA_MLA_SPARSE_VALUE_SKIP")
            : ds4_fa_is_gfx1151(device_info.cc);

    // High-compression histories can outgrow the grouped kernel's score LDS,
    // but the streaming WMMA kernel stores no context-sized score matrix.
    // Dispatch it before the compact-score gate so this path remains bounded
    // at contexts beyond the current DS4 128K limit as well.
    if (bypass_sparse_selector) {
        ggml_cuda_pool_alloc<int> visibility_bounds_alloc(ctx.pool());
        int * visibility_bounds = visibility_bounds_alloc.alloc(
            (size_t) n_tokens * 4);
        ds4_fa_contiguous_causal_bounds_kernel<<<
            (n_tokens + 255) / 256, 256, 0, stream>>>(
                visibility_bounds, n_tokens, n_kv, raw_rows, raw_window,
                causal_kv_start, causal_ratio);
        constexpr int wmma_heads = 16;
        constexpr int head_groups = 2;
        const dim3 wmma_grid(
            (unsigned) n_tokens,
            (unsigned) (n_heads / (head_groups * wmma_heads)), 1);
        const dim3 wmma_block(32, 16, 1);
        ds4_flash_attn_d512_streaming_wmma_kernel<
            head_groups, true, false>
            <<<wmma_grid, wmma_block, 0, stream>>>(
                (float *) dst->data, (const float *) Q->data,
                q_stride_token, q_stride_head,
                (const half *) K->data, nullptr,
                sinks ? (const float *) sinks->data : nullptr,
                n_tokens, n_heads, n_kv, scale,
                visibility_bounds, nullptr, nullptr, 0,
                inverse_rope, inverse_rope_coefficients,
                forward_rope_coefficients);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

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
    if (!sparse && !indexer_topk && !contiguous_causal &&
        n_heads % group4 == 0 &&
        group4_shmem <= 24 * 1024) {
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
    // at most raw_window visible rows for any one token. Indexed decode has a
    // full physical raw ring plus a bounded set of selected compressed rows.
    // Compacting score storage lets both shapes keep the four-head kernel at
    // two-block occupancy. Ordinary dense shapes avoid the extra bounds scan.
    const bool compact_group4 =
        !sparse && (mask || ratio4_causal || contiguous_causal) &&
        n_heads % group4 == 0 &&
        (raw_rows > raw_window || indexed_mask) &&
        (contiguous_causal || indexed_mask ||
         group4_shmem > 24 * 1024) &&
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
            const bool parallel_index_scan = n_comp_rows > 512 &&
                getenv("GGML_DS4_FA_SERIAL_INDEX_SCAN") == nullptr;
            if (ratio4_causal) {
                ds4_fa_indexed_rows_topk_kernel<half, 512, true>
                    <<<n_tokens, 512, 0, stream>>>(
                        nullptr, (const int32_t *) indexer_topk->data,
                        indexed_rows, indexed_counts,
                        indexed_owner_offsets, indexed_owner_ranks,
                        n_tokens, n_kv, raw_rows, indexed_capacity,
                        inverse_rope.kv_start);
            } else if (mask->type == GGML_TYPE_F16) {
                if (indexer_topk) {
                    ds4_launch_indexed_rows_topk<half>(
                        (const half *) mask->data,
                        (const int32_t *) indexer_topk->data,
                        indexed_rows, indexed_counts,
                        indexed_owner_offsets, indexed_owner_ranks,
                        n_tokens, n_kv, raw_rows, indexed_capacity, stream);
                } else if (parallel_index_scan) {
                    ds4_fa_indexed_rows_parallel_kernel<half><<<n_tokens, 256, 0, stream>>>(
                        (const half *) mask->data, indexed_rows, indexed_counts,
                        indexed_owner_offsets, indexed_owner_ranks,
                        n_tokens, n_kv, raw_rows,
                        indexed_capacity);
                } else {
                    ds4_fa_indexed_rows_kernel<half><<<n_tokens, 256, 0, stream>>>(
                        (const half *) mask->data, indexed_rows, indexed_counts,
                        indexed_owner_offsets, indexed_owner_ranks,
                        n_tokens, n_kv, raw_rows,
                        indexed_capacity);
                }
            } else {
                if (indexer_topk) {
                    ds4_launch_indexed_rows_topk<float>(
                        (const float *) mask->data,
                        (const int32_t *) indexer_topk->data,
                        indexed_rows, indexed_counts,
                        indexed_owner_offsets, indexed_owner_ranks,
                        n_tokens, n_kv, raw_rows, indexed_capacity, stream);
                } else if (parallel_index_scan) {
                    ds4_fa_indexed_rows_parallel_kernel<float><<<n_tokens, 256, 0, stream>>>(
                        (const float *) mask->data, indexed_rows, indexed_counts,
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
            }
            CUDA_CHECK(cudaGetLastError());
        }
        if (ratio4_causal) {
            ds4_fa_ratio4_causal_bounds_kernel<<<
                (n_tokens + 255) / 256, 256, 0, stream>>>(
                    visibility_bounds, n_tokens, n_kv, raw_rows,
                    raw_window, inverse_rope.kv_start);
        } else if (contiguous_causal) {
            ds4_fa_contiguous_causal_bounds_kernel<<<
                (n_tokens + 255) / 256, 256, 0, stream>>>(
                    visibility_bounds, n_tokens, n_kv, raw_rows,
                    raw_window, causal_kv_start, causal_ratio);
        } else if (mask->type == GGML_TYPE_F16) {
            ds4_fa_visibility_bounds_kernel<half><<<n_tokens, 64, 0, stream>>>(
                (const half *) mask->data, visibility_bounds,
                n_tokens, n_kv, raw_rows);
        } else {
            ds4_fa_visibility_bounds_kernel<float><<<n_tokens, 64, 0, stream>>>(
                (const float *) mask->data, visibility_bounds,
                n_tokens, n_kv, raw_rows);
        }
        CUDA_CHECK(cudaGetLastError());
        // The same D512 matrix-core kernel also applies when every visible
        // compressed row participates in attention.  This is the common
        // non-indexed MLA case: visibility_bounds describes two contiguous
        // raw/compressed intervals, so no row-index materialization is
        // needed.  The gfx1151 profile enables it (GGML_CUDA_MLA_DENSE_WMMA);
        // =0 falls back to the scalar kernels below.
        if (dense_wmma_enabled && !indexed_mask && !indexer_topk &&
            kv_f16 &&
            (contiguous_causal ||
             (mask && mask->type == GGML_TYPE_F16)) &&
            K->data == V->data && n_heads % 32 == 0 &&
            device_info.warp_size == 32 && n_tokens >= 64 &&
            ds4_fa_is_gfx1151(device_info.cc)) {
            constexpr int wmma_heads = 16;
            constexpr int head_groups = 2;
            const dim3 wmma_grid(
                (unsigned) n_tokens,
                (unsigned) (n_heads / (head_groups * wmma_heads)), 1);
            const dim3 wmma_block(32, 16, 1);
            const auto launch_dense_wmma = [&](auto maskless_tag) {
                constexpr bool maskless =
                    decltype(maskless_tag)::value;
                ds4_flash_attn_d512_streaming_wmma_kernel<
                    head_groups, maskless, false>
                    <<<wmma_grid, wmma_block, 0, stream>>>(
                        (float *) dst->data,
                        (const float *) Q->data,
                        q_stride_token, q_stride_head,
                        (const half *) K->data,
                        maskless ? nullptr : (const half *) mask->data,
                        sinks ? (const float *) sinks->data : nullptr,
                        n_tokens, n_heads, n_kv, scale,
                        visibility_bounds, nullptr, nullptr, 0,
                        inverse_rope, inverse_rope_coefficients,
                        forward_rope_coefficients);
            };
            if (contiguous_causal) {
                launch_dense_wmma(std::true_type{});
            } else {
                launch_dense_wmma(std::false_type{});
            }
            CUDA_CHECK(cudaGetLastError());
            return true;
        }
        // The ROCm 7.2 gfx1151 toolchain currently produces nondeterministic
        // output for the grouped scalar specialization with an analytic mask.
        // The qualified matrix-core path above is both faster and correct on
        // that device. Retain a portable scalar fallback through the tested
        // single-head analytic kernel when WMMA is not selected.
        if (contiguous_causal) {
            if (kv_f16) {
                ds4_flash_attn_d512_shared_kv_kernel<half, half, true>
                    <<<grid, 256, shmem, stream>>>(
                        (float *) dst->data, (const float *) Q->data,
                        q_stride_token, q_stride_head,
                        (const half *) K->data, (const half *) V->data,
                        nullptr,
                        sinks ? (const float *) sinks->data : nullptr,
                        mean_k, n_tokens, n_heads, n_kv, scale, raw_rows,
                        raw_window, sparse_keep_rows, sparse_block_size,
                        n_comp_blocks, causal_kv_start, causal_ratio,
                        inverse_rope, inverse_rope_coefficients,
                        forward_rope_coefficients,
                        skip_sparse_value_gaps);
            } else {
                ds4_flash_attn_d512_shared_kv_kernel<float, float, true>
                    <<<grid, 256, shmem, stream>>>(
                        (float *) dst->data, (const float *) Q->data,
                        q_stride_token, q_stride_head,
                        (const float *) K->data, (const float *) V->data,
                        nullptr,
                        sinks ? (const float *) sinks->data : nullptr,
                        mean_k, n_tokens, n_heads, n_kv, scale, raw_rows,
                        raw_window, sparse_keep_rows, sparse_block_size,
                        n_comp_blocks, causal_kv_start, causal_ratio,
                        inverse_rope, inverse_rope_coefficients,
                        forward_rope_coefficients,
                        skip_sparse_value_gaps);
            }
            CUDA_CHECK(cudaGetLastError());
            return true;
        }
        // Long ratio-4 prefill selects this path structurally. Preserve the
        // existing overrides and the opt-in policy for other indexed shapes.
        // HIP F32 preserves compact arithmetic with eight-head grouping;
        // F16 and CUDA keep the existing wave32 online-softmax policy.
        // Streaming top-k, F32 staging and fast exp default on for ratio-4
        // prefill on every device and for every indexed shape on gfx1151.
        const bool gfx1151_stream_defaults =
            ds4_fa_is_gfx1151(device_info.cc);
        const char * streaming_topk_env =
            getenv("GGML_CUDA_MLA_STREAM_TOPK");
        if (!streaming_topk_env) {
            streaming_topk_env = getenv("GGML_DS4_FA_STREAM_TOPK");
        }
        const bool streaming_topk_enabled = streaming_topk_env
            ? streaming_topk_env[0] != '\0' &&
              strcmp(streaming_topk_env, "0") != 0
            : (ratio4_causal || gfx1151_stream_defaults);
        constexpr int streaming_min_tokens = 64;
        const int active_row_upper_bound = raw_window + indexed_capacity;
        const int device_warp_size = device_info.warp_size;
        if (streaming_topk_enabled && indexed_mask && indexer_topk &&
            (ratio4_causal || mask->type == GGML_TYPE_F16) &&
            K->data == V->data && n_heads % 16 == 0 &&
            device_warp_size == 32 && n_tokens >= streaming_min_tokens &&
            active_row_upper_bound > 0 &&
            n_kv >= 3 * active_row_upper_bound) {
#if defined(GGML_USE_HIP)
            if (kv_f32) {
                // Eight heads reuse each K/V load. Four adjacent values per
                // thread share score loads and row-loop control while each
                // dimension retains its original accumulation order.
                const auto launch_group8 = [&](auto maskless) {
                    ds4_flash_attn_d512_shared_kv_grouped_compact_kernel<
                        float, half, 8, true, decltype(maskless)::value, 4, true>
                        <<<dim3(n_tokens, n_heads / 8), 256,
                           2 * compact_group4_shmem, stream>>>(
                        (float *) dst->data, (const float *) Q->data,
                        q_stride_token, q_stride_head, (const float *) K->data,
                        (const float *) V->data,
                        mask ? (const half *) mask->data : nullptr,
                        sinks ? (const float *) sinks->data : nullptr,
                        n_tokens, n_heads, n_kv, scale,
                        raw_rows, raw_window, compact_score_stride,
                        visibility_bounds, indexed_rows, indexed_counts,
                        indexed_owner_offsets, indexed_owner_ranks,
                        indexed_capacity, inverse_rope,
                        inverse_rope_coefficients, forward_rope_coefficients);
                };
                if (ratio4_causal) launch_group8(std::true_type{});
                else              launch_group8(std::false_type{});
                CUDA_CHECK(cudaGetLastError());
                ++g_mla_stream_topk_launch_count;
                return true;
            }
#endif
            // Off unless the gfx1151 profile sets GGML_CUDA_MLA_STREAM_WMMA;
            // =0 is the kill switch back to the scalar streaming kernel.
            const bool use_wmma = kv_f16 &&
                ds4_env_flag_enabled("GGML_CUDA_MLA_STREAM_WMMA") &&
                ds4_fa_is_gfx1151(device_info.cc);
            if (use_wmma) {
                constexpr int wmma_heads = 16;
                const char * head_groups_env =
                    getenv("GGML_CUDA_MLA_STREAM_WMMA_HEAD_GROUPS");
                const bool use_two_head_groups = head_groups_env &&
                    strcmp(head_groups_env, "2") == 0 &&
                    n_heads % (2 * wmma_heads) == 0;
                const auto launch_wmma = [&](auto head_groups_tag,
                                             auto maskless_tag) {
                    constexpr int head_groups =
                        decltype(head_groups_tag)::value;
                    constexpr bool maskless =
                        decltype(maskless_tag)::value;
                    const dim3 wmma_grid(
                        (unsigned) n_tokens,
                        (unsigned) (n_heads / (head_groups * wmma_heads)), 1);
                    const dim3 wmma_block(32, 16, 1);
                    ds4_flash_attn_d512_streaming_wmma_kernel<
                        head_groups, maskless, true>
                        <<<wmma_grid, wmma_block, 0, stream>>>(
                                (float *) dst->data,
                                (const float *) Q->data,
                                q_stride_token, q_stride_head,
                                (const half *) K->data,
                                maskless ? nullptr
                                         : (const half *) mask->data,
                                sinks ? (const float *) sinks->data : nullptr,
                                n_tokens, n_heads, n_kv, scale,
                                visibility_bounds, indexed_rows,
                                indexed_counts, indexed_capacity,
                                inverse_rope, inverse_rope_coefficients,
                                forward_rope_coefficients);
                };
                if (ratio4_causal) {
                    if (use_two_head_groups) {
                        launch_wmma(std::integral_constant<int, 2>{},
                                    std::true_type{});
                    } else {
                        launch_wmma(std::integral_constant<int, 1>{},
                                    std::true_type{});
                    }
                } else {
                    if (use_two_head_groups) {
                        launch_wmma(std::integral_constant<int, 2>{},
                                    std::false_type{});
                    } else {
                        launch_wmma(std::integral_constant<int, 1>{},
                                    std::false_type{});
                    }
                }
                CUDA_CHECK(cudaGetLastError());
                return true;
            }
            const char * f32_stage_env =
                getenv("GGML_CUDA_MLA_STREAM_F32_STAGE");
            const bool f32_stage = f32_stage_env
                ? f32_stage_env[0] != '\0' && strcmp(f32_stage_env, "0") != 0
                : (ratio4_causal || gfx1151_stream_defaults);
            const char * fast_exp_env =
                getenv("GGML_CUDA_MLA_STREAM_FAST_EXP");
            const bool fast_exp = fast_exp_env
                ? fast_exp_env[0] != '\0' && strcmp(fast_exp_env, "0") != 0
                : (ratio4_causal || gfx1151_stream_defaults);
            constexpr int streaming_heads = 16;
            const dim3 streaming_grid(
                (unsigned) n_tokens,
                (unsigned) (n_heads / streaming_heads), 1);
            const auto launch_streaming = [&](auto kv_type, auto stage_f32, auto use_fast_exp,
                                              auto maskless) {
                using KV = decltype(kv_type);
                ds4_flash_attn_d512_streaming_topk_kernel<
                    KV, half, streaming_heads, 16,
                    decltype(stage_f32)::value, decltype(use_fast_exp)::value,
                    decltype(maskless)::value>
                    <<<streaming_grid, streaming_heads * 32, 0, stream>>>(
                        (float *) dst->data, (const float *) Q->data,
                        q_stride_token, q_stride_head,
                        (const KV *) K->data,
                        mask ? (const half *) mask->data : nullptr,
                        sinks ? (const float *) sinks->data : nullptr,
                        n_tokens, n_heads, n_kv, scale,
                        visibility_bounds, indexed_rows, indexed_counts,
                        indexed_capacity, inverse_rope,
                        inverse_rope_coefficients,
                        forward_rope_coefficients);
            };
            const auto dispatch_streaming = [&](auto kv_type, auto maskless) {
                if (f32_stage && fast_exp) {
                    launch_streaming(kv_type, std::true_type{}, std::true_type{}, maskless);
                } else if (f32_stage) {
                    launch_streaming(kv_type, std::true_type{}, std::false_type{}, maskless);
                } else {
                    launch_streaming(kv_type, std::false_type{}, std::false_type{}, maskless);
                }
            };
            if (ratio4_causal) {
                if (kv_f16) dispatch_streaming(half{}, std::true_type{});
                else        dispatch_streaming(float{}, std::true_type{});
            } else {
                if (kv_f16) dispatch_streaming(half{}, std::false_type{});
                else        dispatch_streaming(float{}, std::false_type{});
            }
            CUDA_CHECK(cudaGetLastError());
            ++g_mla_stream_topk_launch_count;
            return true;
        }
        // AITER-style split-KV schedule, implemented directly in the native HIP
        // backend. Matched Strix Halo profiling showed a bit-identical output,
        // about -58% attention time and +2-3% decode throughput, so make it the
        // gfx1151 default. Other devices remain opt-in until measured.
        constexpr int split_kv_max_decode_tokens = 8;
        const bool split_kv_default = ds4_fa_is_gfx1151(device_info.cc);
        // Segmented KV is consumed only by the split stage-1 kernel; every
        // other kernel reads K/V as one contiguous [n_kv, 512] buffer, so the
        // GGML_CUDA_MLA_NO_SPLIT_KV kill switch must not apply to it.
        if (indexed_mask && !ratio4_causal &&
            n_tokens <= split_kv_max_decode_tokens &&
            (segmented_kv || ds4_mla_split_kv_enabled(split_kv_default))) {
            constexpr int split_count = 4;
            const int split_stride =
                (raw_window + indexed_capacity + split_count - 1) /
                split_count;
            ggml_cuda_pool_alloc<float> partial_alloc(ctx.pool());
            float * partial = partial_alloc.alloc(
                (size_t) n_tokens * n_heads * split_count * (512 + 2));
            if (kv_f16 && mask->type == GGML_TYPE_F16) {
                ds4_launch_flash_attn_d512_indexed_split<
                    half, half, split_count>(
                    (float *) dst->data, partial, (const float *) Q->data,
                    q_stride_token, q_stride_head,
                    (const half *) K->data, (const half *) V->data,
                    segmented_kv
                        ? (const half *) kv_compressed->data : nullptr,
                    segmented_kv
                        ? (const half *) kv_preserved_tail->data : nullptr,
                    (const half *) mask->data,
                    sinks ? (const float *) sinks->data : nullptr,
                    n_tokens, n_heads, n_kv, materialized_kv_rows,
                    compressed_kv_rows, scale, raw_rows, split_stride,
                    visibility_bounds, indexed_rows, indexed_counts,
                    indexed_capacity, inverse_rope,
                    inverse_rope_coefficients, forward_rope_coefficients,
                    stream);
            } else if (kv_f32 && mask->type == GGML_TYPE_F32) {
                ds4_launch_flash_attn_d512_indexed_split<
                    float, float, split_count>(
                    (float *) dst->data, partial, (const float *) Q->data,
                    q_stride_token, q_stride_head,
                    (const float *) K->data, (const float *) V->data,
                    nullptr, nullptr,
                    (const float *) mask->data,
                    sinks ? (const float *) sinks->data : nullptr,
                    n_tokens, n_heads, n_kv, materialized_kv_rows,
                    compressed_kv_rows, scale, raw_rows, split_stride,
                    visibility_bounds, indexed_rows, indexed_counts,
                    indexed_capacity, inverse_rope,
                    inverse_rope_coefficients, forward_rope_coefficients,
                    stream);
            } else if (kv_f32 && mask->type == GGML_TYPE_F16) {
                ds4_launch_flash_attn_d512_indexed_split<
                    float, half, split_count>(
                    (float *) dst->data, partial, (const float *) Q->data,
                    q_stride_token, q_stride_head,
                    (const float *) K->data, (const float *) V->data,
                    nullptr, nullptr,
                    (const half *) mask->data,
                    sinks ? (const float *) sinks->data : nullptr,
                    n_tokens, n_heads, n_kv, materialized_kv_rows,
                    compressed_kv_rows, scale, raw_rows, split_stride,
                    visibility_bounds, indexed_rows, indexed_counts,
                    indexed_capacity, inverse_rope,
                    inverse_rope_coefficients, forward_rope_coefficients,
                    stream);
            } else {
                return false;
            }
            CUDA_CHECK(cudaGetLastError());
            return true;
        }
        // No kernel below understands the segmented layout (see above).
        GGML_ASSERT(!segmented_kv);
        if (indexed_mask) {
            if (ratio4_causal) {
                return ds4_launch_flash_attn_d512_grouped_compact<
                    group4, true, true, 4>(
                    dst, Q, K, V, mask, sinks, kv_f16, kv_f32,
                    n_tokens, n_heads, n_kv, scale, raw_rows,
                    raw_window, compact_score_stride, visibility_bounds,
                    indexed_rows, indexed_counts,
                    indexed_owner_offsets, indexed_owner_ranks,
                    indexed_capacity, q_stride_token, q_stride_head,
                    inverse_rope, inverse_rope_coefficients,
                    forward_rope_coefficients,
                    compact_group4_shmem, stream);
            }
            return ds4_launch_flash_attn_d512_grouped_compact<
                group4, true, false, 4>(
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
        if (contiguous_causal) {
            return ds4_launch_flash_attn_d512_grouped_compact<
                group4, false, true, 4>(
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
        return ds4_launch_flash_attn_d512_grouped_compact<
            group4, false, false, 4>(
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
    // Direct top-k indices are consumed only by the compact four-head path.
    // Never let an unsupported shape silently fall through to dense attention.
    if (indexer_topk) {
        return false;
    }
    // Segmented KV is handled only by the split-KV schedule inside the compact
    // four-head block; the dense kernels below index K/V past their buffer.
    GGML_ASSERT(!segmented_kv);
    if (!sparse && !contiguous_causal && n_heads % group2 == 0 &&
        group2_shmem <= 48 * 1024) {
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
        if (contiguous_causal) {
            ds4_flash_attn_d512_shared_kv_kernel<half, half, true>
                <<<grid, 256, shmem, stream>>>(
                    (float *) dst->data, (const float *) Q->data,
                    q_stride_token, q_stride_head,
                    (const half *) K->data, (const half *) V->data, nullptr,
                    sinks ? (const float *) sinks->data : nullptr,
                    mean_k, n_tokens, n_heads, n_kv, scale, raw_rows,
                    raw_window, sparse_keep_rows, sparse_block_size,
                    n_comp_blocks, causal_kv_start, causal_ratio,
                    inverse_rope, inverse_rope_coefficients,
                    forward_rope_coefficients,
                    skip_sparse_value_gaps);
        } else {
            ds4_flash_attn_d512_shared_kv_kernel<half, half, false>
                <<<grid, 256, shmem, stream>>>(
                    (float *) dst->data, (const float *) Q->data,
                    q_stride_token, q_stride_head,
                    (const half *) K->data, (const half *) V->data,
                    mask ? (const half *) mask->data : nullptr,
                    sinks ? (const float *) sinks->data : nullptr,
                    mean_k, n_tokens, n_heads, n_kv, scale, raw_rows,
                    raw_window, sparse_keep_rows, sparse_block_size,
                    n_comp_blocks, causal_kv_start, causal_ratio,
                    inverse_rope, inverse_rope_coefficients,
                    forward_rope_coefficients,
                    skip_sparse_value_gaps);
        }
    } else if (kv_f32 && (!mask || mask->type == GGML_TYPE_F32)) {
        if (contiguous_causal) {
            ds4_flash_attn_d512_shared_kv_kernel<float, float, true>
                <<<grid, 256, shmem, stream>>>(
                    (float *) dst->data, (const float *) Q->data,
                    q_stride_token, q_stride_head,
                    (const float *) K->data, (const float *) V->data,
                    nullptr,
                    sinks ? (const float *) sinks->data : nullptr,
                    mean_k, n_tokens, n_heads, n_kv, scale, raw_rows,
                    raw_window, sparse_keep_rows, sparse_block_size,
                    n_comp_blocks, causal_kv_start, causal_ratio,
                    inverse_rope, inverse_rope_coefficients,
                    forward_rope_coefficients,
                    skip_sparse_value_gaps);
        } else {
            ds4_flash_attn_d512_shared_kv_kernel<float, float, false>
                <<<grid, 256, shmem, stream>>>(
                    (float *) dst->data, (const float *) Q->data,
                    q_stride_token, q_stride_head,
                    (const float *) K->data, (const float *) V->data,
                    mask ? (const float *) mask->data : nullptr,
                    sinks ? (const float *) sinks->data : nullptr,
                    mean_k, n_tokens, n_heads, n_kv, scale, raw_rows,
                    raw_window, sparse_keep_rows, sparse_block_size,
                    n_comp_blocks, causal_kv_start, causal_ratio,
                    inverse_rope, inverse_rope_coefficients,
                    forward_rope_coefficients,
                    skip_sparse_value_gaps);
        }
    } else if (kv_f32 && mask && mask->type == GGML_TYPE_F16) {
        ds4_flash_attn_d512_shared_kv_kernel<float, half, false>
            <<<grid, 256, shmem, stream>>>(
                (float *) dst->data, (const float *) Q->data,
                q_stride_token, q_stride_head,
                (const float *) K->data, (const float *) V->data,
                (const half *) mask->data,
                sinks ? (const float *) sinks->data : nullptr,
                mean_k, n_tokens, n_heads, n_kv, scale, raw_rows,
                raw_window, sparse_keep_rows, sparse_block_size,
                n_comp_blocks, causal_kv_start, causal_ratio,
                inverse_rope, inverse_rope_coefficients,
                forward_rope_coefficients,
                skip_sparse_value_gaps);
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
    // On RDNA4 the rocWMMA kernel is not qualified (fragment layouts do not
    // match the hand-rolled softmax reductions), so it is reachable only
    // through the env-gated head-256 block below.
    if (ggml_cuda_should_use_wmma_fattn(cc) && !GGML_CUDA_CC_IS_RDNA4(cc) && K->ne[1] % FATTN_KQ_STRIDE == 0 && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 512 && Q->ne[0] != 576) {
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

    // Head-size-256 tensor-core FA on RDNA4. The gate above caps the MMA
    // path at head 128 and the WMMA gate below skips head 256, so
    // Qwen3.5/3.6/3.8 dense-hybrid targets (head_dim=256) fall to the
    // generic tile kernel, which leaves the WMMA units idle and dominates
    // long-context prefill wall time. The raw-MMA kernel is qualified by
    // test_fattn_mma256 (max diff vs the CPU reference enforced at 1e-3
    // f16 / 2e-3 q8_0 KV) and is the default; DFLASH27B_FA256_MMA=0 opts
    // out. DFLASH27B_FA256_WMMA=1 additionally enables the unqualified
    // rocWMMA kernel for A/B work.
    static const auto env_int64 = [](const char * name, int64_t def) -> int64_t {
        const char * e = getenv(name);
        return e ? atoll(e) : def;
    };
    static const bool fa256_tc          = env_int64("DFLASH27B_FA256_MMA", 1) != 0;
    static const bool fa256_wmma        = env_int64("DFLASH27B_FA256_WMMA", 0) != 0;
    // KV length above which the raw-MMA kernel takes over from rocWMMA in
    // flag builds. Measured on gfx1201 (q8_0 KV, nq=512): rocWMMA wins at
    // 8K (4.24 vs 5.32 ms) and 16K (8.55 vs 10.17), raw-MMA wins at 32K
    // (19.45 vs 19.77), 64K (37.87 vs 39.77) and 131K (74.80 vs 79.00).
    // The crossover lies in the unmeasured 16-32K band; override with
    // DFLASH27B_FA256_WMMA_MAX_KV for A/B.
    static const int64_t fa256_wmma_max_kv = env_int64("DFLASH27B_FA256_WMMA_MAX_KV", 32768);
    if ((fa256_tc || fa256_wmma) && amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) &&
        gqa_opt_applies && Q->ne[0] == 256 && V->ne[0] == 256) {
        // Same effective-GQA computation as the RDNA4 head<=128 gate above.
        int gqa_ratio_eff = 1;
        while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < 8) {
            gqa_ratio_eff *= 2;
        }
        if (fa256_wmma && ggml_cuda_should_use_wmma_fattn(cc) && K->ne[1] % FATTN_KQ_STRIDE == 0) {
            return BEST_FATTN_KERNEL_WMMA_F16;
        }
        if (fa256_wmma && !ggml_cuda_should_use_wmma_fattn(cc)) {
            static bool wmma_without_build_warned = false;
            if (!wmma_without_build_warned) {
                fprintf(stderr, "DFLASH27B_FA256_WMMA=1 set but this build has no rocWMMA kernel; using the raw-MMA kernel.\n");
                wmma_without_build_warned = true;
            }
        }
        if (fa256_tc && Q->ne[1] * gqa_ratio_eff > 32) {
            if (ggml_cuda_should_use_wmma_fattn(cc) && K->ne[1] < fa256_wmma_max_kv) {
                return BEST_FATTN_KERNEL_WMMA_F16;
            }
            return BEST_FATTN_KERNEL_MMA_F16;
        }
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
