#include "ds4-indexer.cuh"
#include "ds4-env.cuh"

#if defined(GGML_USE_HIP)
// rocWMMA 1.x rejects RDNA4/gfx1151 at compile time. Use the optimized path
// only with rocWMMA 2.x or newer; older or header-less ROCm installations
// retain the scalar implementation below.
#    if __has_include(<rocwmma/rocwmma-version.hpp>)
#        include <rocwmma/rocwmma-version.hpp>
#    endif
#    if defined(ROCWMMA_VERSION_MAJOR) && ROCWMMA_VERSION_MAJOR > 1
#        include <rocwmma/rocwmma.hpp>
namespace ds4_wmma = rocwmma;
#        define DS4_INDEXER_WMMA_AVAILABLE 1
#    else
#        define DS4_INDEXER_WMMA_AVAILABLE 0
#    endif
#elif !defined(GGML_USE_MUSA)
#    include <mma.h>
namespace ds4_wmma = nvcuda::wmma;
#    define DS4_INDEXER_WMMA_AVAILABLE 1
#else
#    define DS4_INDEXER_WMMA_AVAILABLE 0
#endif

#if DS4_INDEXER_WMMA_AVAILABLE
#    if defined(GGML_USE_HIP) && HIP_VERSION >= 60500000
using ds4_indexer_wmma_half = _Float16;
#    else
using ds4_indexer_wmma_half = half;
#    endif
#endif

// Keep this operation bit-for-bit aligned with the official DeepSeek V4
// graph and antirez/ds4's dsv4_indexer_qat implementation. Indexer query and
// compressed-key rows both pass through an orthonormal Hadamard-128 followed
// by one UE4M3-scaled E2M1 activation-simulation block per 32 values.

static __device__ __forceinline__ float ds4_indexer_e2m1_value(int i) {
    switch (i & 7) {
        case 0: return 0.0f;
        case 1: return 0.5f;
        case 2: return 1.0f;
        case 3: return 1.5f;
        case 4: return 2.0f;
        case 5: return 3.0f;
        case 6: return 4.0f;
        default: return 6.0f;
    }
}

static __device__ __forceinline__ float ds4_indexer_e2m1_round(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - ds4_indexer_e2m1_value(0));
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        const float diff = fabsf(ax - ds4_indexer_e2m1_value(i));
        // Round ties to the even E2M1 code, matching the converter/reference.
        if (diff < best_diff ||
            (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * ds4_indexer_e2m1_value(best);
}

static __global__ void ds4_indexer_qat_kernel(
        float       * dst,
        const float * src,
        int64_t       n_rows,
        int64_t       src_row_stride,
        int64_t       dst_row_stride) {
    constexpr int WIDTH = 128;
    constexpr float HADAMARD_SCALE = 0.08838834764831845f;
    const int64_t row = (int64_t) blockIdx.x;
    const int tid = (int) threadIdx.x;
    if (row >= n_rows || tid >= WIDTH) return;

    __shared__ float values[WIDTH];
    __shared__ float abs_values[WIDTH];
    const float * src_row = src + row * src_row_stride;
    float * dst_row = dst + row * dst_row_stride;
    values[tid] = src_row[tid];
    __syncthreads();

    for (int stride = 1; stride < WIDTH; stride <<= 1) {
        if ((tid & stride) == 0) {
            const int base =
                (tid & ~(2 * stride - 1)) + (tid & (stride - 1));
            const float a = values[base];
            const float b = values[base + stride];
            values[base] = a + b;
            values[base + stride] = a - b;
        }
        __syncthreads();
    }

    const float value = values[tid] * HADAMARD_SCALE;
    const int block = tid >> 5;
    const int lane = tid & 31;
    const int block_base = block * 32;
    abs_values[tid] = fabsf(value);
    __syncthreads();

    for (int stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) {
            abs_values[block_base + lane] = fmaxf(
                abs_values[block_base + lane],
                abs_values[block_base + lane + stride]);
        }
        __syncthreads();
    }

    const float amax = fmaxf(
        abs_values[block_base], 7.052966104933725e-38f);
    const float scale = exp2f(ceilf(log2f(amax / 6.0f)));
    const float normalized = fminf(6.0f, fmaxf(-6.0f, value / scale));
    dst_row[tid] = ds4_indexer_e2m1_round(normalized) * scale;
}

void ggml_cuda_op_ds4_indexer_qat(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src && src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src->ne[0] == 128 && dst->ne[0] == 128);
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_rows = ggml_nrows(src);
    const int64_t src_row_stride = src->nb[1] / sizeof(float);
    const int64_t dst_row_stride = dst->nb[1] / sizeof(float);
    GGML_ASSERT(src_row_stride >= 128 && dst_row_stride >= 128);

    cudaStream_t stream = ctx.stream();
    ds4_indexer_qat_kernel<<<(unsigned) n_rows, 128, 0, stream>>>(
        static_cast<float *>(dst->data),
        static_cast<const float *>(src->data),
        n_rows, src_row_stride, dst_row_stride);
    CUDA_CHECK(cudaGetLastError());
}

// Compute 16 query tokens against 128 compressed rows per block. QAT values
// are powers-of-two-scaled E2M1 and therefore exactly representable as F16 in
// the model's operating range; the compressed cache is already F16. WMMA
// removes the otherwise enormous [n_comp,64,n_tokens] intermediate while the
// ReLU, head weighting and reduction remain F32.
#if DS4_INDEXER_WMMA_AVAILABLE
// Staged query rows are padded from 128 to 136 halves so the 16-wide fragment
// loads of consecutive rows do not land on the same LDS bank.
constexpr int DS4_INDEXER_QUERY_STRIDE = 136;

static __global__ void ds4_indexer_score_wmma_kernel(
        float       * scores,
        const float * q,
        const float * weights,
        const half  * index_comp,
        const float * visibility_mask,
        int           n_comp,
        int           n_tokens,
        int           kv_start,
        int           n_head,
        int           ratio) {
    const int tile_c = (int) blockIdx.x * 128;
    const int tile_t = (int) blockIdx.y * 16;
    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;

    __shared__ half a_sh[16 * 128];
    __shared__ half b_sh[128 * 128];
    __shared__ float c_sh[8 * 16 * 16];

    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;

    for (int i = tid; i < 128 * 128; i += 256) {
        const int c = i >> 7;
        const int d = i & 127;
        const int comp = tile_c + c;
        b_sh[d + c * 128] = comp < n_comp
            ? index_comp[(size_t) comp * 128 + d]
            : __float2half(0.0f);
    }
    __syncthreads();

    for (int h = 0; h < n_head; ++h) {
        for (int pair = tid; pair < 16 * 64; pair += 256) {
            const int row = pair >> 6;
            const int d = (pair & 63) * 2;
            const int token = tile_t + row;
            half2 value = __float2half2_rn(0.0f);
            if (token < n_tokens) {
                const float2 q_value = *reinterpret_cast<const float2 *>(
                    q + ((size_t) token * n_head + h) * 128 + d);
                value = __floats2half2_rn(q_value.x, q_value.y);
            }
            *reinterpret_cast<half2 *>(a_sh + row * 128 + d) = value;
        }
        __syncthreads();

        ds4_wmma::fragment<ds4_wmma::matrix_a, 16, 16, 16,
                           ds4_indexer_wmma_half,
                           ds4_wmma::row_major> a_frag;
        ds4_wmma::fragment<ds4_wmma::matrix_b, 16, 16, 16,
                           ds4_indexer_wmma_half,
                           ds4_wmma::col_major> b_frag;
        ds4_wmma::fragment<ds4_wmma::accumulator, 16, 16, 16,
                           float> c_frag;
        ds4_wmma::fill_fragment(c_frag, 0.0f);
        const int col0 = warp * 16;
        for (int k0 = 0; k0 < 128; k0 += 16) {
            const ds4_indexer_wmma_half * a_wmma =
                reinterpret_cast<const ds4_indexer_wmma_half *>(a_sh);
            const ds4_indexer_wmma_half * b_wmma =
                reinterpret_cast<const ds4_indexer_wmma_half *>(b_sh);
            ds4_wmma::load_matrix_sync(a_frag, a_wmma + k0, 128);
            ds4_wmma::load_matrix_sync(
                b_frag, b_wmma + col0 * 128 + k0, 128);
            ds4_wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        ds4_wmma::store_matrix_sync(
            c_sh + warp * 16 * 16, c_frag, 16,
            ds4_wmma::mem_row_major);
        __syncthreads();

        const int token_for_lane = tile_t + (tid >> 4);
        const float head_weight = token_for_lane < n_tokens
            ? weights[(size_t) token_for_lane * n_head + h]
            : 0.0f;
        int slot = 0;
        for (int i = tid; i < 8 * 16 * 16; i += 256, ++slot) {
            acc[slot] += fmaxf(c_sh[i], 0.0f) * head_weight;
        }
        __syncthreads();
    }

    int slot = 0;
    for (int i = tid; i < 8 * 16 * 16; i += 256, ++slot) {
        const int wtile = i >> 8;
        const int local = i & 255;
        const int row = local >> 4;
        const int col = local & 15;
        const int token = tile_t + row;
        const int comp = tile_c + wtile * 16 + col;
        if (token < n_tokens && comp < n_comp) {
            const int visible = (kv_start + token + 1) / ratio;
            const bool row_visible = visibility_mask
                ? visibility_mask[(size_t) token * n_comp + comp] > -1.0e20f
                : comp < visible;
            scores[(size_t) token * n_comp + comp] =
                row_visible ? acc[slot] : -1.0e30f;
        }
    }
}

// Decode has one token but 64 indexer heads. Treat 16 heads as WMMA rows and
// 128 compressed keys as columns. This performs only useful dot products;
// the general 16-token kernel above spends 15/16 of its WMMA work on zero rows
// when n_tokens == 1, which made decode scale linearly with context length.
static __global__ void ds4_indexer_score_decode_wmma_kernel(
        float       * scores,
        const float * q,
        const float * weights,
        const half  * index_comp,
        const float * visibility_mask,
        int           n_comp,
        int           kv_start,
        int           n_head,
        int           ratio) {
    const int tile_c = (int) blockIdx.x * 128;
    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;

    constexpr int QUERY_STRIDE = DS4_INDEXER_QUERY_STRIDE;
    __shared__ half a_sh[16 * QUERY_STRIDE];
    __shared__ half b_sh[128 * 128];
    __shared__ float c_sh[8 * 16 * 16];

    for (int i = tid; i < 128 * 128; i += 256) {
        const int c = i >> 7;
        const int d = i & 127;
        const int comp = tile_c + c;
        b_sh[d + c * 128] = comp < n_comp
            ? index_comp[(size_t) comp * 128 + d]
            : __float2half(0.0f);
    }
    __syncthreads();

    float score = 0.0f;
    for (int head0 = 0; head0 < n_head; head0 += 16) {
        for (int i = tid; i < 16 * 128; i += 256) {
            const int local_head = i >> 7;
            const int d = i & 127;
            const int head = head0 + local_head;
            a_sh[(size_t) local_head * QUERY_STRIDE + d] = head < n_head
                ? __float2half(q[(size_t) head * 128 + d])
                : __float2half(0.0f);
        }
        __syncthreads();

        ds4_wmma::fragment<ds4_wmma::matrix_a, 16, 16, 16,
                           ds4_indexer_wmma_half,
                           ds4_wmma::row_major> a_frag;
        ds4_wmma::fragment<ds4_wmma::matrix_b, 16, 16, 16,
                           ds4_indexer_wmma_half,
                           ds4_wmma::col_major> b_frag;
        ds4_wmma::fragment<ds4_wmma::accumulator, 16, 16, 16,
                           float> c_frag;
        ds4_wmma::fill_fragment(c_frag, 0.0f);
        const int col0 = warp * 16;
        for (int k0 = 0; k0 < 128; k0 += 16) {
            const ds4_indexer_wmma_half * a_wmma =
                reinterpret_cast<const ds4_indexer_wmma_half *>(a_sh);
            const ds4_indexer_wmma_half * b_wmma =
                reinterpret_cast<const ds4_indexer_wmma_half *>(b_sh);
            ds4_wmma::load_matrix_sync(
                a_frag, a_wmma + k0, QUERY_STRIDE);
            ds4_wmma::load_matrix_sync(
                b_frag, b_wmma + col0 * 128 + k0, 128);
            ds4_wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        ds4_wmma::store_matrix_sync(
            c_sh + warp * 16 * 16, c_frag, 16,
            ds4_wmma::mem_row_major);
        __syncthreads();

        if (tid < 128) {
            const int local_warp = tid >> 4;
            const int col = tid & 15;
#pragma unroll
            for (int h = 0; h < 16; ++h) {
                const int head = head0 + h;
                if (head < n_head) {
                    score += fmaxf(c_sh[local_warp * 256 + h * 16 + col], 0.0f) *
                             weights[head];
                }
            }
        }
        __syncthreads();
    }

    if (tid < 128) {
        const int comp = tile_c + tid;
        if (comp < n_comp) {
            const int visible = (kv_start + 1) / ratio;
            const bool row_visible = visibility_mask
                ? visibility_mask[comp] > -1.0e20f
                : comp < visible;
            scores[comp] = row_visible ? score : -1.0e30f;
        }
    }
}

// The speculative verifier scores only a few query tokens. The general WMMA
// kernel places them in a 16-row tile and executes the unused rows for every
// head. Pack consecutive heads into the tile instead:
// row = N_TOKENS*head_in_group + token. The post-WMMA loop still accumulates
// heads in their original order, preserving the established F32 numerical
// topology. This is the HIP equivalent of the Vulkan small-CM dispatch.
template<int N_TOKENS>
static __global__ void ds4_indexer_score_wmma_small_kernel(
        float       * scores,
        const float * q,
        const float * weights,
        const half  * index_comp,
        const float * visibility_mask,
        int           n_comp,
        int           kv_start,
        int           n_head,
        int           ratio) {
    const int tile_c = (int) blockIdx.x * 128;
    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;

    __shared__ half a_sh[16 * 128];
    __shared__ half b_sh[128 * 128];
    __shared__ float c_sh[8 * 16 * 16];
    __shared__ float weight_sh[16];

    static_assert(N_TOKENS >= 2 && N_TOKENS <= 5,
                  "small-CM kernel is specialized for verifier widths 2..5");
    constexpr int HEADS_PER_TILE = 16 / N_TOKENS;
    constexpr int USED_ROWS = HEADS_PER_TILE * N_TOKENS;
    constexpr int ACC_SLOTS = (N_TOKENS + 1) / 2;
    float acc[ACC_SLOTS];
#pragma unroll
    for (int slot = 0; slot < ACC_SLOTS; ++slot) acc[slot] = 0.0f;

    for (int i = tid; i < 128 * 128; i += 256) {
        const int c = i >> 7;
        const int d = i & 127;
        const int comp = tile_c + c;
        b_sh[d + c * 128] = comp < n_comp
            ? index_comp[(size_t) comp * 128 + d]
            : __float2half(0.0f);
    }
    __syncthreads();

    for (int head_base = 0; head_base < n_head;
         head_base += HEADS_PER_TILE) {
        for (int pair = tid; pair < 16 * 64; pair += 256) {
            const int row = pair >> 6;
            const int d = (pair & 63) * 2;
            half2 value = __float2half2_rn(0.0f);
            if (row < USED_ROWS) {
                const int token = row % N_TOKENS;
                const int head = head_base + row / N_TOKENS;
                if (head < n_head) {
                    const float2 q_value =
                        *reinterpret_cast<const float2 *>(
                            q + ((size_t) token * n_head + head) * 128 + d);
                    value = __floats2half2_rn(q_value.x, q_value.y);
                }
            }
            *reinterpret_cast<half2 *>(a_sh + row * 128 + d) = value;
        }
        if (tid < 16) {
            const int token = tid % N_TOKENS;
            const int head = head_base + tid / N_TOKENS;
            weight_sh[tid] = tid < USED_ROWS && head < n_head
                ? weights[(size_t) token * n_head + head]
                : 0.0f;
        }
        __syncthreads();

        ds4_wmma::fragment<ds4_wmma::matrix_a, 16, 16, 16,
                           ds4_indexer_wmma_half,
                           ds4_wmma::row_major> a_frag;
        ds4_wmma::fragment<ds4_wmma::matrix_b, 16, 16, 16,
                           ds4_indexer_wmma_half,
                           ds4_wmma::col_major> b_frag;
        ds4_wmma::fragment<ds4_wmma::accumulator, 16, 16, 16,
                           float> c_frag;
        ds4_wmma::fill_fragment(c_frag, 0.0f);
        const int col0 = warp * 16;
        for (int k0 = 0; k0 < 128; k0 += 16) {
            const ds4_indexer_wmma_half * a_wmma =
                reinterpret_cast<const ds4_indexer_wmma_half *>(a_sh);
            const ds4_indexer_wmma_half * b_wmma =
                reinterpret_cast<const ds4_indexer_wmma_half *>(b_sh);
            ds4_wmma::load_matrix_sync(a_frag, a_wmma + k0, 128);
            ds4_wmma::load_matrix_sync(
                b_frag, b_wmma + col0 * 128 + k0, 128);
            ds4_wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        ds4_wmma::store_matrix_sync(
            c_sh + warp * 16 * 16, c_frag, 16,
            ds4_wmma::mem_row_major);
        __syncthreads();

        int slot = 0;
        for (int output = tid; output < N_TOKENS * 128;
             output += 256, ++slot) {
            const int token = output >> 7;
            const int local_comp = output & 127;
            const int comp_tile = local_comp >> 4;
            const int comp_col = local_comp & 15;
#pragma unroll
            for (int head_in_group = 0;
                 head_in_group < HEADS_PER_TILE;
                 ++head_in_group) {
                const int row = N_TOKENS * head_in_group + token;
                const float dot = c_sh[
                    comp_tile * 16 * 16 + row * 16 + comp_col];
                acc[slot] += fmaxf(dot, 0.0f) * weight_sh[row];
            }
        }
        __syncthreads();
    }

    int slot = 0;
    for (int output = tid; output < N_TOKENS * 128;
         output += 256, ++slot) {
        const int token = output >> 7;
        const int comp = tile_c + (output & 127);
        if (comp < n_comp) {
            const int visible = (kv_start + token + 1) / ratio;
            const bool row_visible = visibility_mask
                ? visibility_mask[(size_t) token * n_comp + comp] > -1.0e20f
                : comp < visible;
            scores[(size_t) token * n_comp + comp] =
                row_visible ? acc[slot] : -1.0e30f;
        }
    }
}

#if defined(GGML_USE_HIP)
// Pack two 16-row query groups into one 32x16 rocWMMA fragment. This halves
// duplicate compressed-key fragment loads. Full compressed-row tiles read B
// directly from device memory to avoid a 32 KiB LDS allocation; a staged tail
// kernel preserves bounds safety without changing score bits.
template<int N_TOKENS, bool DIRECT_B, bool CACHE_B, bool F16_Q>
static __global__ void ds4_indexer_score_wmma_m32_kernel(
        float       * scores,
        const void  * q,
        const float * weights,
        const half  * index_comp,
        const float * visibility_mask,
        int           n_comp,
        int           kv_start,
        int           n_head,
        int           ratio,
        int           comp_offset) {
    static_assert(N_TOKENS >= 2 && N_TOKENS <= 8,
                  "M32 indexer covers verifier widths 2..8");
    constexpr int COMP_TILE = 128;
    constexpr int HEADS_PER_ROW_TILE = 16 / N_TOKENS;
    constexpr int USED_ROWS = HEADS_PER_ROW_TILE * N_TOKENS;
    constexpr int HEADS_PER_ITER = 2 * HEADS_PER_ROW_TILE;
    constexpr int ACC_SLOTS = (N_TOKENS * COMP_TILE + 255) / 256;
    constexpr int QUERY_STRIDE = DS4_INDEXER_QUERY_STRIDE;

    const int tile_c = comp_offset + (int) blockIdx.x * COMP_TILE;
    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;

    __shared__ half a_sh[32 * QUERY_STRIDE];
    __shared__ __align__(16) half b_sh[DIRECT_B ? 1 : COMP_TILE * 128];
    __shared__ float c_sh[8 * 32 * 16];
    __shared__ float weight_sh[32];

    float acc[ACC_SLOTS];
#pragma unroll
    for (int slot = 0; slot < ACC_SLOTS; ++slot) acc[slot] = 0.0f;

    if constexpr (!DIRECT_B) {
        for (int i = tid; i < COMP_TILE * 16; i += 256) {
            const int c = i >> 4;
            const int d = (i & 15) * 8;
            const int comp = tile_c + c;
            float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (comp < n_comp) {
                value = *reinterpret_cast<const float4 *>(
                    index_comp + (size_t) comp * 128 + d);
            }
            *reinterpret_cast<float4 *>(b_sh + c * 128 + d) = value;
        }
        __syncthreads();
    }

    using b_fragment = ds4_wmma::fragment<
        ds4_wmma::matrix_b, 32, 16, 16, ds4_indexer_wmma_half,
        ds4_wmma::col_major>;
    b_fragment cached_b[8];
    if constexpr (CACHE_B) {
#pragma unroll
        for (int k_tile = 0; k_tile < 8; ++k_tile) {
            const ds4_indexer_wmma_half * b_wmma =
                reinterpret_cast<const ds4_indexer_wmma_half *>(
                    DIRECT_B ? index_comp + (size_t) tile_c * 128 : b_sh);
            ds4_wmma::load_matrix_sync(
                cached_b[k_tile],
                b_wmma + warp * 16 * 128 + k_tile * 16, 128);
        }
    }

    for (int head_base = 0; head_base < n_head;
         head_base += HEADS_PER_ITER) {
        for (int pair = tid; pair < 32 * 64; pair += 256) {
            const int row = pair >> 6;
            const int local_row = row & 15;
            const int d = (pair & 63) * 2;
            half2 value = __float2half2_rn(0.0f);
            if (local_row < USED_ROWS) {
                const int token = local_row % N_TOKENS;
                const int head = head_base + (row >> 4) * HEADS_PER_ROW_TILE +
                                 local_row / N_TOKENS;
                if (head < n_head) {
                    const size_t q_offset =
                        ((size_t) token * n_head + head) * 128 + d;
                    if constexpr (F16_Q) {
                        value = *reinterpret_cast<const half2 *>(
                            static_cast<const half *>(q) + q_offset);
                    } else {
                        const float2 q_value =
                            *reinterpret_cast<const float2 *>(
                                static_cast<const float *>(q) + q_offset);
                        value = __floats2half2_rn(q_value.x, q_value.y);
                    }
                }
            }
            *reinterpret_cast<half2 *>(
                a_sh + row * QUERY_STRIDE + d) = value;
        }
        if (tid < 32) {
            const int local_row = tid & 15;
            const int token = local_row % N_TOKENS;
            const int head = head_base + (tid >> 4) * HEADS_PER_ROW_TILE +
                             local_row / N_TOKENS;
            weight_sh[tid] = local_row < USED_ROWS && head < n_head
                ? weights[(size_t) token * n_head + head]
                : 0.0f;
        }
        __syncthreads();

        ds4_wmma::fragment<ds4_wmma::matrix_a, 32, 16, 16,
                           ds4_indexer_wmma_half,
                           ds4_wmma::row_major> a_frag;
        b_fragment b_frag;
        ds4_wmma::fragment<ds4_wmma::accumulator, 32, 16, 16,
                           float> c_frag;
        ds4_wmma::fill_fragment(c_frag, 0.0f);
        for (int k0 = 0; k0 < 128; k0 += 16) {
            const ds4_indexer_wmma_half * a_wmma =
                reinterpret_cast<const ds4_indexer_wmma_half *>(a_sh);
            const ds4_indexer_wmma_half * b_wmma =
                reinterpret_cast<const ds4_indexer_wmma_half *>(
                    DIRECT_B ? index_comp + (size_t) tile_c * 128 : b_sh);
            ds4_wmma::load_matrix_sync(
                a_frag, a_wmma + k0, QUERY_STRIDE);
            if constexpr (CACHE_B) {
                ds4_wmma::mma_sync(
                    c_frag, a_frag, cached_b[k0 / 16], c_frag);
            } else {
                ds4_wmma::load_matrix_sync(
                    b_frag, b_wmma + warp * 16 * 128 + k0, 128);
                ds4_wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }
        ds4_wmma::store_matrix_sync(
            c_sh + warp * 32 * 16, c_frag, 16,
            ds4_wmma::mem_row_major);
        __syncthreads();

        int slot = 0;
        for (int output = tid; output < N_TOKENS * COMP_TILE;
             output += 256, ++slot) {
            const int token = output / COMP_TILE;
            const int local_comp = output % COMP_TILE;
            const int comp_tile = local_comp >> 4;
            const int comp_col = local_comp & 15;
#pragma unroll
            for (int query_tile = 0; query_tile < 2; ++query_tile) {
#pragma unroll
                for (int head_in_group = 0;
                     head_in_group < HEADS_PER_ROW_TILE;
                     ++head_in_group) {
                    const int row = query_tile * 16 +
                                    N_TOKENS * head_in_group + token;
                    const float dot = c_sh[
                        comp_tile * 32 * 16 + row * 16 + comp_col];
                    acc[slot] += fmaxf(dot, 0.0f) * weight_sh[row];
                }
            }
        }
        // The next head_base iteration rewrites weight_sh and c_sh: drain the
        // reads above before any warp reaches them. The small kernel's
        // equivalent barrier is at :474; without it the staged M32 route races
        // and nondeterministically changes score bits.
        __syncthreads();
    }

    int slot = 0;
    for (int output = tid; output < N_TOKENS * COMP_TILE;
         output += 256, ++slot) {
        const int token = output / COMP_TILE;
        const int comp = tile_c + output % COMP_TILE;
        if (comp < n_comp) {
            const int visible = (kv_start + token + 1) / ratio;
            const bool row_visible = visibility_mask
                ? visibility_mask[(size_t) token * n_comp + comp] > -1.0e20f
                : comp < visible;
            scores[(size_t) token * n_comp + comp] =
                row_visible ? acc[slot] : -1.0e30f;
        }
    }
}

// Process two adjacent 16-token tiles in one rocWMMA M32 fragment. The
// established prefill kernel stages a 32 KiB compressed-key tile and can keep
// only one workgroup resident on gfx1151. Full tiles read B directly, leaving
// 24 KiB of LDS for A and C so two workgroups can remain resident. Head
// contributions are still accumulated in their original ascending order.
template<bool DIRECT_B, bool CACHE_B, bool F16_Q>
static __global__ void ds4_indexer_score_wmma_m32_prefill_kernel(
        float       * scores,
        const void  * q,
        const float * weights,
        const half  * index_comp,
        const float * visibility_mask,
        int           n_comp,
        int           n_tokens,
        int           kv_start,
        int           n_head,
        int           ratio,
        int           comp_offset) {
    constexpr int TOKEN_TILE = 32;
    constexpr int COMP_TILE = 128;
    constexpr int ACC_SLOTS = TOKEN_TILE * COMP_TILE / 256;
    constexpr int HEADS_PER_TILE = 1;
    constexpr int QUERY_STRIDE = DS4_INDEXER_QUERY_STRIDE;
    constexpr int SCORE_STRIDE = 16;

    const int tile_c = comp_offset + (int) blockIdx.x * COMP_TILE;
    const int tile_t = (int) blockIdx.y * TOKEN_TILE;
    const int tid = (int) threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ half a_sh[HEADS_PER_TILE * TOKEN_TILE * QUERY_STRIDE];
    __shared__ __align__(16) half b_sh[DIRECT_B ? 1 : COMP_TILE * 128];
    __shared__ float c_sh[8 * TOKEN_TILE * SCORE_STRIDE];
    __shared__ float weight_sh[HEADS_PER_TILE * TOKEN_TILE];

    float acc[ACC_SLOTS];
#pragma unroll
    for (int slot = 0; slot < ACC_SLOTS; ++slot) acc[slot] = 0.0f;

    if constexpr (!DIRECT_B) {
        for (int i = tid; i < COMP_TILE * 16; i += 256) {
            const int c = i >> 4;
            const int d = (i & 15) * 8;
            const int comp = tile_c + c;
            float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (comp < n_comp) {
                value = *reinterpret_cast<const float4 *>(
                    index_comp + (size_t) comp * 128 + d);
            }
            *reinterpret_cast<float4 *>(b_sh + c * 128 + d) = value;
        }
        __syncthreads();
    }

    // The compressed-key tile is invariant across all 64 indexer heads. A
    // direct-B launch otherwise fetches the same eight fragments from global
    // memory once per head. Keep those fragments wave-local when the device
    // has enough registers; the uncached specialization remains available for
    // architecture-specific tuning.
    using b_fragment = ds4_wmma::fragment<
        ds4_wmma::matrix_b, 32, 16, 16, ds4_indexer_wmma_half,
        ds4_wmma::col_major>;
    b_fragment cached_b[8];
    if constexpr (CACHE_B) {
#pragma unroll
        for (int k_tile = 0; k_tile < 8; ++k_tile) {
            const ds4_indexer_wmma_half * b_wmma =
                reinterpret_cast<const ds4_indexer_wmma_half *>(
                    DIRECT_B ? index_comp + (size_t) tile_c * 128 : b_sh);
            ds4_wmma::load_matrix_sync(
                cached_b[k_tile],
                b_wmma + warp * 16 * 128 + k_tile * 16, 128);
        }
    }

    for (int head_base = 0; head_base < n_head;
         head_base += HEADS_PER_TILE) {
        for (int pair = tid;
             pair < HEADS_PER_TILE * TOKEN_TILE * 64;
             pair += 256) {
            const int head_local = pair / (TOKEN_TILE * 64);
            const int head_pair = pair % (TOKEN_TILE * 64);
            const int row = head_pair >> 6;
            const int d = (head_pair & 63) * 2;
            const int token = tile_t + row;
            const int head = head_base + head_local;
            half2 value = __float2half2_rn(0.0f);
            if (token < n_tokens && head < n_head) {
                const size_t q_offset =
                    ((size_t) token * n_head + head) * 128 + d;
                if constexpr (F16_Q) {
                    value = *reinterpret_cast<const half2 *>(
                        static_cast<const half *>(q) + q_offset);
                } else {
                    const float2 q_value =
                        *reinterpret_cast<const float2 *>(
                            static_cast<const float *>(q) + q_offset);
                    value = __floats2half2_rn(q_value.x, q_value.y);
                }
            }
            *reinterpret_cast<half2 *>(
                a_sh + ((size_t) head_local * TOKEN_TILE + row) *
                    QUERY_STRIDE + d) =
                    value;
        }
        if (tid < HEADS_PER_TILE * TOKEN_TILE) {
            const int head_local = tid / TOKEN_TILE;
            const int row = tid % TOKEN_TILE;
            const int token = tile_t + row;
            const int head = head_base + head_local;
            weight_sh[tid] = token < n_tokens && head < n_head
                ? weights[(size_t) token * n_head + head]
                : 0.0f;
        }
        __syncthreads();

        // Every wave owns one 16-column compressed-key tile. Wave-scoped
        // synchronization is sufficient while a wave stores and consumes its
        // private score tile; only the query-storage handoff between heads
        // needs a full workgroup barrier.
#pragma unroll
        for (int head_local = 0; head_local < HEADS_PER_TILE; ++head_local) {
            ds4_wmma::fragment<ds4_wmma::matrix_a, 32, 16, 16,
                               ds4_indexer_wmma_half,
                               ds4_wmma::row_major> a_frag;
            b_fragment b_frag;
            ds4_wmma::fragment<ds4_wmma::accumulator, 32, 16, 16,
                               float> c_frag;
            ds4_wmma::fill_fragment(c_frag, 0.0f);
            for (int k0 = 0; k0 < 128; k0 += 16) {
                const ds4_indexer_wmma_half * a_wmma =
                    reinterpret_cast<const ds4_indexer_wmma_half *>(a_sh) +
                    (size_t) head_local * TOKEN_TILE * QUERY_STRIDE;
                const ds4_indexer_wmma_half * b_wmma =
                    reinterpret_cast<const ds4_indexer_wmma_half *>(
                        DIRECT_B
                            ? index_comp + (size_t) tile_c * 128
                            : b_sh);
                ds4_wmma::load_matrix_sync(
                    a_frag, a_wmma + k0, QUERY_STRIDE);
                if constexpr (CACHE_B) {
                    ds4_wmma::mma_sync(
                        c_frag, a_frag, cached_b[k0 / 16], c_frag);
                } else {
                    ds4_wmma::load_matrix_sync(
                        b_frag, b_wmma + warp * 16 * 128 + k0, 128);
                    ds4_wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
            }
            ds4_wmma::store_matrix_sync(
                c_sh + warp * TOKEN_TILE * SCORE_STRIDE,
                c_frag, SCORE_STRIDE,
                ds4_wmma::mem_row_major);
            __syncwarp();

            int slot = 0;
            for (int output = lane; output < TOKEN_TILE * 16;
                 output += 32, ++slot) {
                const int row = output >> 4;
                const int col = output & 15;
                acc[slot] += fmaxf(
                    c_sh[(warp * TOKEN_TILE + row) * SCORE_STRIDE + col],
                    0.0f) *
                    weight_sh[head_local * TOKEN_TILE + row];
            }
            __syncwarp();
        }
        __syncthreads();
    }

    int slot = 0;
    for (int output = lane; output < TOKEN_TILE * 16;
         output += 32, ++slot) {
        const int row = output >> 4;
        const int col = output & 15;
        const int token = tile_t + row;
        const int comp = tile_c + warp * 16 + col;
        if (token < n_tokens && comp < n_comp) {
            const int visible = (kv_start + token + 1) / ratio;
            const bool row_visible = visibility_mask
                ? visibility_mask[(size_t) token * n_comp + comp] > -1.0e20f
                : comp < visible;
            scores[(size_t) token * n_comp + comp] =
                row_visible ? acc[slot] : -1.0e30f;
        }
    }
}

#endif

#endif

template<bool F16_Q>
static __global__ void ds4_indexer_score_scalar_kernel(
        float       * scores,
        const void  * q,
        const float * weights,
        const half  * index_comp,
        const float * visibility_mask,
        int           n_comp,
        int           n_tokens,
        int           kv_start,
        int           n_head,
        int           ratio) {
    const int comp = (int) blockIdx.x;
    const int token = (int) blockIdx.y;
    const int tid = (int) threadIdx.x;
    if (comp >= n_comp || token >= n_tokens) return;
    const int visible = (kv_start + token + 1) / ratio;
    const bool row_visible = visibility_mask
        ? visibility_mask[(size_t) token * n_comp + comp] > -1.0e20f
        : comp < visible;
    if (!row_visible) {
        if (tid == 0) {
            scores[(size_t) token * n_comp + comp] = -1.0e30f;
        }
        return;
    }

    __shared__ float partial[256];
    float total = 0.0f;
    const half * k = index_comp + (size_t) comp * 128;
    for (int h = 0; h < n_head; ++h) {
        const size_t q_offset = ((size_t) token * n_head + h) * 128;
        float dot = 0.0f;
        for (int d = tid; d < 128; d += 256) {
            const float q_value = F16_Q
                ? __half2float(static_cast<const half *>(q)[q_offset + d])
                : static_cast<const float *>(q)[q_offset + d];
            dot += q_value * __half2float(k[d]);
        }
        partial[tid] = dot;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            total += fmaxf(partial[0], 0.0f) *
                     weights[(size_t) token * n_head + h];
        }
        __syncthreads();
    }
    if (tid == 0) scores[(size_t) token * n_comp + comp] = total;
}

void ggml_cuda_op_ds4_indexer_score(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * weights = dst->src[1];
    const ggml_tensor * comp = dst->src[2];
    const ggml_tensor * visibility_mask = dst->src[3];
    GGML_ASSERT(q && weights && comp);
    GGML_ASSERT((q->type == GGML_TYPE_F32 || q->type == GGML_TYPE_F16) &&
                q->ne[0] == 128);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(comp->type == GGML_TYPE_F16 && comp->ne[0] == 128);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(ggml_is_contiguous(comp));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(!visibility_mask ||
                (visibility_mask->type == GGML_TYPE_F32 &&
                 ggml_is_contiguous(visibility_mask)));

    const int n_head = (int) q->ne[1];
    const int n_tokens = (int) q->ne[2];
    const int n_comp = (int) comp->ne[1];
    const int kv_start = ggml_get_op_params_i32(dst, 0);
    const int ratio = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(weights->ne[0] == n_head && weights->ne[1] == n_tokens);
    GGML_ASSERT(dst->ne[0] == n_comp && dst->ne[1] == n_tokens);

    cudaStream_t stream = ctx.stream();
    const int device = ggml_cuda_get_device();
    const auto & device_info = ggml_cuda_info().devices[device];
    const int warp_size = device_info.warp_size;
    const bool wmma_capable =
        warp_size == 32 &&
        (!GGML_CUDA_CC_IS_NVIDIA(device_info.cc) ||
         device_info.cc >= GGML_CUDA_CC_VOLTA);
    const char * packed_small_name = "GGML_DS4_INDEXER_PACK_SMALL";
    const char * packed_small_env = std::getenv(packed_small_name);
    if (!packed_small_env) {
        // Backward-compatible alias for the original q=4-only prototype.
        packed_small_name = "GGML_DS4_INDEXER_PACK_Q4";
        packed_small_env = std::getenv(packed_small_name);
    }
    const bool use_packed_small = packed_small_env
        ? ds4_env_flag_enabled(packed_small_name)
        : GGML_CUDA_CC_IS_RDNA3_5(device_info.cc) ||
          GGML_CUDA_CC_IS_RDNA4(device_info.cc);
#if DS4_INDEXER_WMMA_AVAILABLE
#if defined(GGML_USE_HIP)
    // The M32 switches are read per call on purpose: the unit tests toggle
    // them between graph computes inside one process. M32 defaults on for
    // RDNA 3.5 (=0 is the kill switch); CACHE_B is enabled by the gfx1151
    // device profile; PREFILL and DIRECT_B override measured crossovers and
    // let the tests reach every specialization.
    const char * m32_name = "GGML_DS4_INDEXER_M32";
    const char * m32_env = std::getenv(m32_name);
    const bool m32_enabled = m32_env
        ? ds4_env_flag_enabled(m32_name)
        : GGML_CUDA_CC_IS_RDNA3_5(device_info.cc);
    const bool use_m32 =
        (q->type == GGML_TYPE_F32 || q->type == GGML_TYPE_F16) &&
        n_tokens >= 2 && n_tokens <= 8 && m32_enabled;
    const char * m32_prefill_name = "GGML_DS4_INDEXER_M32_PREFILL";
    const char * m32_prefill_env = std::getenv(m32_prefill_name);
    // Small tail batches do not provide enough parallel work to amortize the
    // larger M32 fragment. Layer-major prefill normally submits 2K-10K rows;
    // 256 is the measured gfx1151 crossover that also protects a 129-row
    // first-chunk edge case.
    constexpr int m32_prefill_min_tokens = 256;
    const bool use_m32_prefill = n_tokens >= 16 && m32_enabled &&
        (m32_prefill_env
             ? ds4_env_flag_enabled(m32_prefill_name)
             : n_tokens >= m32_prefill_min_tokens);
    // Full 128-row tiles read B straight from device memory. Below this row
    // count the staged tail launch is a large share of the work, so a call
    // with a partial tile stays on the staged kernel (measured gfx1151
    // crossover).
    constexpr int m32_direct_b_min_rows = 6144;
    const char * direct_b_name = "GGML_DS4_INDEXER_M32_DIRECT_B";
    const char * direct_b_env = std::getenv(direct_b_name);
    const bool direct_b_shape = n_comp % 128 == 0 ||
                                n_comp >= m32_direct_b_min_rows;
    const bool use_m32_direct_b = direct_b_env
        ? ds4_env_flag_enabled(direct_b_name)
        : direct_b_shape;
    const bool use_m32_prefill_direct_b = direct_b_env
        ? use_m32_direct_b
        : true;
    const char * cache_b_name = "GGML_DS4_INDEXER_M32_CACHE_B";
    const char * cache_b_env = std::getenv(cache_b_name);
    const bool use_m32_cache_b = cache_b_env &&
        ds4_env_flag_enabled(cache_b_name);
    if (wmma_capable && use_m32) {
#define DS4_LAUNCH_M32_INDEXER(N, DB, CB, F16Q)                        \
        ds4_indexer_score_wmma_m32_kernel<N, DB, CB, F16Q>              \
            <<<grid, 256, 0, stream>>>(                                \
            static_cast<float *>(dst->data),                           \
            q->data,                                                    \
            static_cast<const float *>(weights->data),                 \
            static_cast<const half *>(comp->data),                     \
            visibility_mask                                            \
                ? static_cast<const float *>(visibility_mask->data)     \
                : nullptr,                                              \
            n_comp, kv_start, n_head, ratio, comp_offset)
#define DS4_DISPATCH_M32_TYPED(DB, CB, F16Q)                           \
            switch (n_tokens) {                                       \
                case 2: DS4_LAUNCH_M32_INDEXER(2, DB, CB, F16Q); break; \
                case 3: DS4_LAUNCH_M32_INDEXER(3, DB, CB, F16Q); break; \
                case 4: DS4_LAUNCH_M32_INDEXER(4, DB, CB, F16Q); break; \
                case 5: DS4_LAUNCH_M32_INDEXER(5, DB, CB, F16Q); break; \
                case 6: DS4_LAUNCH_M32_INDEXER(6, DB, CB, F16Q); break; \
                case 7: DS4_LAUNCH_M32_INDEXER(7, DB, CB, F16Q); break; \
                case 8: DS4_LAUNCH_M32_INDEXER(8, DB, CB, F16Q); break; \
                default: GGML_ABORT("unreachable M32 indexer width");  \
            }
#define DS4_DISPATCH_M32(DB, CB)                                       \
            do {                                                       \
                if (q->type == GGML_TYPE_F16) {                        \
                    DS4_DISPATCH_M32_TYPED(DB, CB, true);              \
                } else {                                               \
                    DS4_DISPATCH_M32_TYPED(DB, CB, false);             \
                }                                                      \
            } while (false)
        if (use_m32_direct_b) {
            const int full_tiles = n_comp / 128;
            if (full_tiles > 0) {
                const dim3 grid((unsigned) full_tiles, 1, 1);
                const int comp_offset = 0;
                if (use_m32_cache_b) {
                    DS4_DISPATCH_M32(true, true);
                } else {
                    DS4_DISPATCH_M32(true, false);
                }
            }
            if (full_tiles * 128 < n_comp) {
                const dim3 grid(1, 1, 1);
                const int comp_offset = full_tiles * 128;
                if (use_m32_cache_b) {
                    DS4_DISPATCH_M32(false, true);
                } else {
                    DS4_DISPATCH_M32(false, false);
                }
            }
        } else {
            const dim3 grid((unsigned) ((n_comp + 127) / 128), 1, 1);
            const int comp_offset = 0;
            if (use_m32_cache_b) {
                DS4_DISPATCH_M32(false, true);
            } else {
                DS4_DISPATCH_M32(false, false);
            }
        }
#undef DS4_DISPATCH_M32
#undef DS4_DISPATCH_M32_TYPED
#undef DS4_LAUNCH_M32_INDEXER
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (wmma_capable && use_m32_prefill) {
#define DS4_LAUNCH_M32_PREFILL_INDEXER(DB, CB, F16Q)                    \
        ds4_indexer_score_wmma_m32_prefill_kernel<DB, CB, F16Q>         \
            <<<grid, 256, 0, stream>>>(                                \
            static_cast<float *>(dst->data),                           \
            q->data,                                                    \
            static_cast<const float *>(weights->data),                 \
            static_cast<const half *>(comp->data),                     \
            visibility_mask                                            \
                ? static_cast<const float *>(visibility_mask->data)     \
                : nullptr,                                              \
            n_comp, n_tokens, kv_start, n_head, ratio, comp_offset)
        const bool f16_q = q->type == GGML_TYPE_F16;
        const unsigned int token_tiles =
            (unsigned int) ((n_tokens + 31) / 32);
#define DS4_DISPATCH_M32_PREFILL(DB, CB)                                \
        do {                                                            \
            if (f16_q) {                                                \
                DS4_LAUNCH_M32_PREFILL_INDEXER(DB, CB, true);           \
            } else {                                                    \
                DS4_LAUNCH_M32_PREFILL_INDEXER(DB, CB, false);          \
            }                                                           \
        } while (false)
        if (use_m32_prefill_direct_b) {
            const int full_tiles = n_comp / 128;
            if (full_tiles > 0) {
                const dim3 grid(
                    (unsigned) full_tiles, token_tiles, 1);
                const int comp_offset = 0;
                if (use_m32_cache_b) {
                    DS4_DISPATCH_M32_PREFILL(true, true);
                } else {
                    DS4_DISPATCH_M32_PREFILL(true, false);
                }
            }
            if (full_tiles * 128 < n_comp) {
                const dim3 grid(1, token_tiles, 1);
                const int comp_offset = full_tiles * 128;
                if (use_m32_cache_b) {
                    DS4_DISPATCH_M32_PREFILL(false, true);
                } else {
                    DS4_DISPATCH_M32_PREFILL(false, false);
                }
            }
        } else {
            const dim3 grid((unsigned) ((n_comp + 127) / 128),
                            token_tiles, 1);
            const int comp_offset = 0;
            if (use_m32_cache_b) {
                DS4_DISPATCH_M32_PREFILL(false, true);
            } else {
                DS4_DISPATCH_M32_PREFILL(false, false);
            }
        }
#undef DS4_DISPATCH_M32_PREFILL
#undef DS4_LAUNCH_M32_PREFILL_INDEXER
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#endif
    if (wmma_capable && q->type == GGML_TYPE_F32 && n_tokens == 1) {
        const dim3 grid((unsigned) ((n_comp + 127) / 128), 1, 1);
        ds4_indexer_score_decode_wmma_kernel<<<grid, 256, 0, stream>>>(
            static_cast<float *>(dst->data),
            static_cast<const float *>(q->data),
            static_cast<const float *>(weights->data),
            static_cast<const half *>(comp->data),
            visibility_mask
                ? static_cast<const float *>(visibility_mask->data) : nullptr,
            n_comp, kv_start, n_head, ratio);
    } else if (wmma_capable && q->type == GGML_TYPE_F32 &&
               n_tokens == 2 && use_packed_small) {
        const dim3 grid((unsigned) ((n_comp + 127) / 128), 1, 1);
        ds4_indexer_score_wmma_small_kernel<2><<<grid, 256, 0, stream>>>(
            static_cast<float *>(dst->data),
            static_cast<const float *>(q->data),
            static_cast<const float *>(weights->data),
            static_cast<const half *>(comp->data),
            visibility_mask
                ? static_cast<const float *>(visibility_mask->data) : nullptr,
            n_comp, kv_start, n_head, ratio);
    } else if (wmma_capable && q->type == GGML_TYPE_F32 &&
               n_tokens == 3 && use_packed_small) {
        const dim3 grid((unsigned) ((n_comp + 127) / 128), 1, 1);
        ds4_indexer_score_wmma_small_kernel<3><<<grid, 256, 0, stream>>>(
            static_cast<float *>(dst->data),
            static_cast<const float *>(q->data),
            static_cast<const float *>(weights->data),
            static_cast<const half *>(comp->data),
            visibility_mask
                ? static_cast<const float *>(visibility_mask->data) : nullptr,
            n_comp, kv_start, n_head, ratio);
    } else if (wmma_capable && q->type == GGML_TYPE_F32 &&
               n_tokens == 4 && use_packed_small) {
        const dim3 grid((unsigned) ((n_comp + 127) / 128), 1, 1);
        ds4_indexer_score_wmma_small_kernel<4><<<grid, 256, 0, stream>>>(
            static_cast<float *>(dst->data),
            static_cast<const float *>(q->data),
            static_cast<const float *>(weights->data),
            static_cast<const half *>(comp->data),
            visibility_mask
                ? static_cast<const float *>(visibility_mask->data) : nullptr,
            n_comp, kv_start, n_head, ratio);
    } else if (wmma_capable && q->type == GGML_TYPE_F32 &&
               n_tokens == 5 && use_packed_small) {
        const dim3 grid((unsigned) ((n_comp + 127) / 128), 1, 1);
        ds4_indexer_score_wmma_small_kernel<5><<<grid, 256, 0, stream>>>(
            static_cast<float *>(dst->data),
            static_cast<const float *>(q->data),
            static_cast<const float *>(weights->data),
            static_cast<const half *>(comp->data),
            visibility_mask
                ? static_cast<const float *>(visibility_mask->data) : nullptr,
            n_comp, kv_start, n_head, ratio);
    } else if (wmma_capable && q->type == GGML_TYPE_F32) {
        const dim3 grid((unsigned) ((n_comp + 127) / 128),
                        (unsigned) ((n_tokens + 15) / 16), 1);
        ds4_indexer_score_wmma_kernel<<<grid, 256, 0, stream>>>(
            static_cast<float *>(dst->data),
            static_cast<const float *>(q->data),
            static_cast<const float *>(weights->data),
            static_cast<const half *>(comp->data),
            visibility_mask
                ? static_cast<const float *>(visibility_mask->data) : nullptr,
            n_comp, n_tokens, kv_start, n_head, ratio);
    } else
#endif
    {
        (void) wmma_capable;
        const dim3 grid((unsigned) n_comp, (unsigned) n_tokens, 1);
        if (q->type == GGML_TYPE_F16) {
            ds4_indexer_score_scalar_kernel<true><<<grid, 256, 0, stream>>>(
                static_cast<float *>(dst->data), q->data,
                static_cast<const float *>(weights->data),
                static_cast<const half *>(comp->data),
                visibility_mask
                    ? static_cast<const float *>(visibility_mask->data)
                    : nullptr,
                n_comp, n_tokens, kv_start, n_head, ratio);
        } else {
            ds4_indexer_score_scalar_kernel<false><<<grid, 256, 0, stream>>>(
                static_cast<float *>(dst->data), q->data,
                static_cast<const float *>(weights->data),
                static_cast<const half *>(comp->data),
                visibility_mask
                    ? static_cast<const float *>(visibility_mask->data)
                    : nullptr,
                n_comp, n_tokens, kv_start, n_head, ratio);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void ds4_indexer_mask_init_kernel(
        float       * dst,
        const float * base,
        int64_t       n_elements,
        int           n_attn,
        int           raw_rows) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_elements) return;
    const int row = (int) (index % n_attn);
    dst[index] = row < raw_rows ? base[index] : -1.0e30f;
}

static __global__ void ds4_indexer_mask_scatter_kernel(
        float         * dst,
        const float   * base,
        const int32_t * selected,
        int64_t         n_selected,
        int             n_attn,
        int             raw_rows,
        int             n_comp,
        int             top_k) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_selected) return;
    const int token = (int) (index / top_k);
    const int comp = selected[index];
    if (comp < 0 || comp >= n_comp) return;
    const int64_t output_index =
        (int64_t) token * n_attn + raw_rows + comp;
    // Preserve the base causal mask: when a row is not yet visible, top-k may
    // still return it only as an -inf filler for tokens with < k live rows.
    dst[output_index] = base[output_index];
}

void ggml_cuda_op_ds4_indexer_mask(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    const ggml_tensor * base = dst->src[0];
    const ggml_tensor * selected = dst->src[1];
    GGML_ASSERT(base && selected);
    GGML_ASSERT(base->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(selected->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(base, dst));
    GGML_ASSERT(ggml_is_contiguous(base) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(selected));

    const int n_attn = (int) base->ne[0];
    const int top_k = (int) selected->ne[0];
    const int n_tokens = (int) ggml_nrows(base);
    const int raw_rows = ggml_get_op_params_i32(dst, 0);
    const int n_comp = n_attn - raw_rows;
    GGML_ASSERT(raw_rows >= 0 && n_comp >= 0);
    GGML_ASSERT(ggml_nrows(selected) == n_tokens);

    const int64_t n_elements = ggml_nelements(base);
    const int64_t n_selected = ggml_nelements(selected);
    cudaStream_t stream = ctx.stream();
    ds4_indexer_mask_init_kernel<<<
        (unsigned) ((n_elements + 255) / 256), 256, 0, stream>>>(
            static_cast<float *>(dst->data),
            static_cast<const float *>(base->data),
            n_elements, n_attn, raw_rows);
    ds4_indexer_mask_scatter_kernel<<<
        (unsigned) ((n_selected + 255) / 256), 256, 0, stream>>>(
            static_cast<float *>(dst->data),
            static_cast<const float *>(base->data),
            static_cast<const int32_t *>(selected->data),
            n_selected, n_attn, raw_rows, n_comp, top_k);
    CUDA_CHECK(cudaGetLastError());
}
