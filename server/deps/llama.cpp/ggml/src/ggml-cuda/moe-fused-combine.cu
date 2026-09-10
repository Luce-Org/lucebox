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
        float4       * __restrict__ output,
        const int n_embd_vec4,
        const int n_used,
        const int n_tokens,
        const size_t down_nb1,
        const size_t down_nb2,
        const size_t weights_nb1,
        const size_t shared_nb1,
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
        const float4 sh = shared_out[h4 + t * shared_nb1];
        sum0 = __fadd_rn(sh.x, sum0);
        sum1 = __fadd_rn(sh.y, sum1);
        sum2 = __fadd_rn(sh.z, sum2);
        sum3 = __fadd_rn(sh.w, sum3);
    }

    output[h4 + t * out_nb1] = make_float4(sum0, sum1, sum2, sum3);
}

void ggml_cuda_op_ds4_moe_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * down_e     = dst->src[0];
    const ggml_tensor * weights    = dst->src[1];
    const ggml_tensor * shared_out = dst->src[2];

    GGML_ASSERT(down_e->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int n_embd   = (int) down_e->ne[0];
    const int n_used   = (int) down_e->ne[1];
    const int n_tokens = (int) down_e->ne[2];

    static const bool trace_enabled = []() {
        const char * raw = std::getenv("DFLASH_MOE_FUSED_COMBINE_TRACE");
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
    const size_t out_nb1 = dst->nb[1] / sizeof(float4);

    cudaStream_t stream = ctx.stream();

    moe_fused_combine_shared_kernel_f32<<<grid_size, block_size, 0, stream>>>(
        (const float4 *) down_e->data,
        (const float *)  weights->data,
        shared_out ? (const float4 *) shared_out->data : nullptr,
        (float4 *) dst->data,
        n_embd_vec4,
        n_used,
        n_tokens,
        down_nb1,
        down_nb2,
        weights_nb1,
        shared_nb1,
        out_nb1
    );
    CUDA_CHECK(cudaGetLastError());
}
