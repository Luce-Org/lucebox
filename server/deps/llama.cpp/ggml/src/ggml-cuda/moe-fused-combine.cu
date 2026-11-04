#include "common.cuh"
#include "moe-fused-combine.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Fused post-MMID route reduction and shared-expert add. The down projection
// remains a separate operation and materializes down_e before this kernel.
// Computes dst[t, d] = (shared_out ? shared_out[t, d] : 0) + sum_{e=0}^{n_used-1} (down_e[t, e, d] * weights[t, e])
// Uses vectorized float4 128-bit memory transactions and sequential non-FMA FP32 accumulation
// to preserve the legacy route-order reduction.

static __global__ void moe_fused_combine_shared_kernel_f32(
        const float4 * __restrict__ down_e,
        const float  * __restrict__ weights,
        const float4 * __restrict__ shared_out,
        const float  * __restrict__ shared_logit,
        float4       * __restrict__ output,
        const int n_embd_vec4,
        const int n_used,
        const int n_tokens,
        const size_t down_nb1,
        const size_t down_nb2,
        const size_t weights_nb1,
        const size_t shared_nb1,
        const size_t logit_nb1,
        const size_t out_nb1) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_embd_vec4 * n_tokens;
    if (idx >= total) return;

    const int h4 = idx % n_embd_vec4;
    const int t  = idx / n_embd_vec4;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    for (int e = 0; e < n_used; ++e) {
        const float  w = weights[e + t * weights_nb1];
        // Expert-major owners encode routes assigned to the other device as
        // weight zero and ID -1. Do not read those MMID output lanes: a masked
        // lane is semantically zero, and stale NaN/Inf multiplied by zero would
        // otherwise poison the reduction.
        if (w == 0.0f) {
            continue;
        }
        const float4 v = down_e[h4 + e * down_nb1 + t * down_nb2];

        const float p0 = __fmul_rn(v.x, w);
        const float p1 = __fmul_rn(v.y, w);
        const float p2 = __fmul_rn(v.z, w);
        const float p3 = __fmul_rn(v.w, w);

        // Start from +0 and add every active route, matching the legacy
        // sum_rows reduction even for a first product of -0.0f.
        sum0 = __fadd_rn(sum0, p0);
        sum1 = __fadd_rn(sum1, p1);
        sum2 = __fadd_rn(sum2, p2);
        sum3 = __fadd_rn(sum3, p3);
    }

    if (shared_out != nullptr) {
        float4 sh = shared_out[h4 + t * shared_nb1];
        if (shared_logit != nullptr) {
            // The shared expert's sigmoid gate as unary.cu's op_sigmoid, times the output as binbcast.cu's op_mul.
            const float g = 1.0f / (1.0f + expf(-shared_logit[t * logit_nb1]));
            sh = make_float4(__fmul_rn(sh.x, g), __fmul_rn(sh.y, g), __fmul_rn(sh.z, g), __fmul_rn(sh.w, g));
        }
        sum0 = __fadd_rn(sh.x, sum0);
        sum1 = __fadd_rn(sh.y, sum1);
        sum2 = __fadd_rn(sh.z, sum2);
        sum3 = __fadd_rn(sh.w, sum3);
    }

    output[h4 + t * out_nb1] = make_float4(sum0, sum1, sum2, sum3);
}

void ggml_cuda_op_ds4_moe_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                  const ggml_tensor * shared_ungated, const ggml_tensor * shared_logit) {
    const ggml_tensor * down_e     = dst->src[0];
    const ggml_tensor * weights    = dst->src[1];
    const ggml_tensor * shared_out = shared_logit ? shared_ungated : dst->src[2];
    if (shared_logit) {
        GGML_ASSERT(shared_out != nullptr && shared_logit->type == GGML_TYPE_F32 && shared_logit->ne[0] == 1 &&
                    shared_logit->ne[1] == down_e->ne[2]);
    }

    GGML_ASSERT(down_e->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int n_embd   = (int) down_e->ne[0];
    const int n_used   = (int) down_e->ne[1];
    const int n_tokens = (int) down_e->ne[2];

    static const bool trace_enabled = []() {
        const char * raw = std::getenv("LUCE_MOE_FUSED_COMBINE_TRACE");
        return raw && *raw && std::strcmp(raw, "0") != 0;
    }();
    if (trace_enabled) {
        static std::atomic<unsigned long long> launch_count{0};
        int device = -1;
        CUDA_CHECK(cudaGetDevice(&device));
        const unsigned long long launch =
            launch_count.fetch_add(1, std::memory_order_relaxed) + 1;
        std::fprintf(stderr,
            "[moe-fused-combine] launch=%llu device=%d tokens=%d routes=%d shared=%d\n",
            launch, device, n_tokens, n_used, shared_out != nullptr ? 1 : 0);
    }

    GGML_ASSERT(n_embd % 4 == 0 && "n_embd must be a multiple of 4 for float4 vectorization");
    GGML_ASSERT(reinterpret_cast<uintptr_t>(down_e->data) % 16 == 0 && "down_e->data must be 16-byte aligned");
    GGML_ASSERT(reinterpret_cast<uintptr_t>(dst->data) % 16 == 0 && "dst->data must be 16-byte aligned");
    GGML_ASSERT(down_e->nb[0] == sizeof(float) && "down_e must be contiguous in dimension 0");
    GGML_ASSERT(down_e->nb[1] % sizeof(float4) == 0 && "down_e->nb[1] must be divisible by sizeof(float4)");
    GGML_ASSERT(down_e->nb[2] % sizeof(float4) == 0 && "down_e->nb[2] must be divisible by sizeof(float4)");
    GGML_ASSERT(dst->nb[0] == sizeof(float) && "dst must be contiguous in dimension 0");
    GGML_ASSERT(dst->nb[1] % sizeof(float4) == 0 && "dst->nb[1] must be divisible by sizeof(float4)");
    if (shared_out != nullptr) {
        GGML_ASSERT(reinterpret_cast<uintptr_t>(shared_out->data) % 16 == 0 && "shared_out->data must be 16-byte aligned");
        GGML_ASSERT(shared_out->nb[0] == sizeof(float) && "shared_out must be contiguous in dimension 0");
        GGML_ASSERT(shared_out->nb[1] % sizeof(float4) == 0 && "shared_out->nb[1] must be divisible by sizeof(float4)");
    }

    const int n_embd_vec4 = n_embd / 4;
    const int total_threads = n_embd_vec4 * n_tokens;

    const int block_size = 256;
    const int grid_size = (total_threads + block_size - 1) / block_size;

    const size_t down_nb1 = down_e->nb[1] / sizeof(float4);
    const size_t down_nb2 = down_e->nb[2] / sizeof(float4);
    const size_t weights_nb1 = weights->nb[1] / sizeof(float);
    const size_t shared_nb1 = shared_out ? (shared_out->nb[1] / sizeof(float4)) : 0;
    const size_t logit_nb1 = shared_logit ? (shared_logit->nb[1] / sizeof(float)) : 0;
    const size_t out_nb1 = dst->nb[1] / sizeof(float4);

    cudaStream_t stream = ctx.stream();

    moe_fused_combine_shared_kernel_f32<<<grid_size, block_size, 0, stream>>>(
        (const float4 *) down_e->data,
        (const float *)  weights->data,
        shared_out ? (const float4 *) shared_out->data : nullptr,
        shared_logit ? (const float *) shared_logit->data : nullptr,
        (float4 *) dst->data,
        n_embd_vec4,
        n_used,
        n_tokens,
        down_nb1,
        down_nb2,
        weights_nb1,
        shared_nb1,
        logit_nb1,
        out_nb1
    );
    CUDA_CHECK(cudaGetLastError());
}

// The renormalized weights of each token's top-k routes, [k, T] from the router probabilities [n_expert, T] and the
// picks [k, T]: GET_ROWS, SUM_ROWS, CLAMP and DIV in one launch with their arithmetic. One thread per token; the row
// sum follows reduce_rows_f32 on a k-wide row (lane l holds element l, then the 32-lane xor butterfly; the block's
// second stage adds zeros only).
static __global__ void moe_route_weights_norm_kernel(
        const char * __restrict__ probs, const size_t probs_nb1, const size_t probs_nb2,
        const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
        const int k, const int n_tokens, const float clamp_min, const float clamp_max, float * __restrict__ dst) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tokens) return;
    float w[32];
    float v[32];
#pragma unroll
    for (int l = 0; l < 32; ++l) {
        w[l] = 0.0f;
        if (l < k) {
            const int32_t e = *(const int32_t *) (ids + (size_t) l * ids_nb0 + (size_t) t * ids_nb1);
            w[l] = *(const float *) (probs + (size_t) e * probs_nb1 + (size_t) t * probs_nb2);
        }
        v[l] = w[l];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float n[32];
#pragma unroll
        for (int l = 0; l < 32; ++l) n[l] = v[l] + v[l ^ off];
#pragma unroll
        for (int l = 0; l < 32; ++l) v[l] = n[l];
    }
    const float sum = fminf(fmaxf(v[0], clamp_min), clamp_max);   // clamp.cu's op_clamp
    for (int i = 0; i < k; ++i) {
        dst[(size_t) t * k + i] = w[i] / sum;                       // binbcast.cu's op_div
    }
}

void ggml_cuda_moe_route_weights_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * probs3,
                                      const ggml_tensor * ids, float clamp_min, float clamp_max, ggml_tensor * dst) {
    const int k = (int) ids->ne[0];
    const int n_tokens = (int) ids->ne[1];
    GGML_ASSERT(k <= 32 && probs3->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_nelements(dst) == (int64_t) k * n_tokens);
    const int block = 64;
    moe_route_weights_norm_kernel<<<(n_tokens + block - 1) / block, block, 0, ctx.stream()>>>(
        (const char *) probs3->data, probs3->nb[1], probs3->nb[2], (const char *) ids->data, ids->nb[0], ids->nb[1],
        k, n_tokens, clamp_min, clamp_max, (float *) dst->data);
    CUDA_CHECK(cudaGetLastError());
}
