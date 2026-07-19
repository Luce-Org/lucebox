#include "ds4-indexer.cuh"

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
static __global__ void ds4_indexer_score_wmma_kernel(
        float       * scores,
        const float * q,
        const float * weights,
        const half  * index_comp,
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
            scores[(size_t) token * n_comp + comp] =
                comp < visible ? acc[slot] : -1.0e30f;
        }
    }
}
#endif

static __global__ void ds4_indexer_score_scalar_kernel(
        float       * scores,
        const float * q,
        const float * weights,
        const half  * index_comp,
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
    if (comp >= visible) {
        if (tid == 0) {
            scores[(size_t) token * n_comp + comp] = -1.0e30f;
        }
        return;
    }

    __shared__ float partial[256];
    float total = 0.0f;
    const half * k = index_comp + (size_t) comp * 128;
    for (int h = 0; h < n_head; ++h) {
        const float * qh = q + ((size_t) token * n_head + h) * 128;
        float dot = 0.0f;
        for (int d = tid; d < 128; d += 256) {
            dot += qh[d] * __half2float(k[d]);
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
    GGML_ASSERT(q && weights && comp);
    GGML_ASSERT(q->type == GGML_TYPE_F32 && q->ne[0] == 128);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(comp->type == GGML_TYPE_F16 && comp->ne[0] == 128);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(ggml_is_contiguous(comp));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int n_head = (int) q->ne[1];
    const int n_tokens = (int) q->ne[2];
    const int n_comp = (int) comp->ne[1];
    const int kv_start = ggml_get_op_params_i32(dst, 0);
    const int ratio = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(weights->ne[0] == n_head && weights->ne[1] == n_tokens);
    GGML_ASSERT(dst->ne[0] == n_comp && dst->ne[1] == n_tokens);

    cudaStream_t stream = ctx.stream();
    const int warp_size =
        ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
#if DS4_INDEXER_WMMA_AVAILABLE
    if (warp_size == 32) {
        const dim3 grid((unsigned) ((n_comp + 127) / 128),
                        (unsigned) ((n_tokens + 15) / 16), 1);
        ds4_indexer_score_wmma_kernel<<<grid, 256, 0, stream>>>(
            static_cast<float *>(dst->data),
            static_cast<const float *>(q->data),
            static_cast<const float *>(weights->data),
            static_cast<const half *>(comp->data),
            n_comp, n_tokens, kv_start, n_head, ratio);
    } else
#endif
    {
        const dim3 grid((unsigned) n_comp, (unsigned) n_tokens, 1);
        ds4_indexer_score_scalar_kernel<<<grid, 256, 0, stream>>>(
            static_cast<float *>(dst->data),
            static_cast<const float *>(q->data),
            static_cast<const float *>(weights->data),
            static_cast<const half *>(comp->data),
            n_comp, n_tokens, kv_start, n_head, ratio);
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
