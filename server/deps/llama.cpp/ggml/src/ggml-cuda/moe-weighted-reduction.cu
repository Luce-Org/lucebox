#include "moe-weighted-reduction.cuh"
#include "mmb.cuh"
#include <cstdlib>

__device__ __forceinline__ float moe_bf2f(const uint16_t h) { return __uint_as_float(((uint32_t) h) << 16); }


static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}


// float4 variant: 4 consecutive columns per thread, same per-element accumulation order as the scalar kernel (bit-identical)
static __global__ void moe_weighted_reduction_f32_v4(const float * __restrict__ experts,
                                                     const float * __restrict__ expert_scale,
                                                     const float * __restrict__ weights,
                                                     float * __restrict__ dst,
                                                     const int64_t n_embd,
                                                     const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col4  = ((int64_t) blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (col4 >= n_embd) return;
    const uint64_t first_row = (uint64_t) token * n_expert_used;
    const float first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    const float w0 = weights[first_row];
    const float4 e0 = *(const float4 *)(experts + first_row * n_embd + col4);
    float4 sum; sum.x = (e0.x * first_scale) * w0; sum.y = (e0.y * first_scale) * w0; sum.z = (e0.z * first_scale) * w0; sum.w = (e0.w * first_scale) * w0;
    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row = first_row + expert;
        const float scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        const float w = weights[row];
        const float4 e = *(const float4 *)(experts + row * n_embd + col4);
        sum.x += (e.x * scale) * w; sum.y += (e.y * scale) * w; sum.z += (e.z * scale) * w; sum.w += (e.w * scale) * w;
    }
    *(float4 *)(dst + token * n_embd + col4) = sum;
}

// BF16 expert outputs: identical arithmetic, the inputs are the BF16-rounded down-GEMM results
static __global__ void moe_weighted_reduction_bf16_v4(const uint16_t * __restrict__ experts,
                                                      const float * __restrict__ expert_scale,
                                                      const float * __restrict__ weights,
                                                      float * __restrict__ dst,
                                                      const int64_t n_embd,
                                                      const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col4  = ((int64_t) blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (col4 >= n_embd) return;
    const uint64_t first_row = (uint64_t) token * n_expert_used;
    const float first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    const float w0 = weights[first_row];
    const ushort4 h0 = *(const ushort4 *)(experts + first_row * n_embd + col4);
    float4 sum; sum.x = (moe_bf2f(h0.x) * first_scale) * w0; sum.y = (moe_bf2f(h0.y) * first_scale) * w0;
    sum.z = (moe_bf2f(h0.z) * first_scale) * w0; sum.w = (moe_bf2f(h0.w) * first_scale) * w0;
    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row = first_row + expert;
        const float scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        const float w = weights[row];
        const ushort4 h = *(const ushort4 *)(experts + row * n_embd + col4);
        sum.x += (moe_bf2f(h.x) * scale) * w; sum.y += (moe_bf2f(h.y) * scale) * w;
        sum.z += (moe_bf2f(h.z) * scale) * w; sum.w += (moe_bf2f(h.w) * scale) * w;
    }
    *(float4 *)(dst + token * n_embd + col4) = sum;
}

__device__ __forceinline__ uint16_t moe_f2bf(const float f) { uint32_t u = __float_as_uint(f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }

// BF16 in, BF16 out: the only consumer is the fused HC combine, which reads block_out as BF16
static __global__ void moe_weighted_reduction_bf16_v4_out(const uint16_t * __restrict__ experts,
                                                          const float * __restrict__ expert_scale,
                                                          const float * __restrict__ weights,
                                                          uint16_t * __restrict__ dst,
                                                          const float * __restrict__ merge,
                                                          const int64_t n_embd, const int n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col4  = ((int64_t) blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (col4 >= n_embd) return;
    const uint64_t first_row = (uint64_t) token * n_expert_used;
    const float first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    const float w0 = weights[first_row];
    const ushort4 h0 = *(const ushort4 *)(experts + first_row * n_embd + col4);
    float4 sum; sum.x = (moe_bf2f(h0.x) * first_scale) * w0; sum.y = (moe_bf2f(h0.y) * first_scale) * w0;
    sum.z = (moe_bf2f(h0.z) * first_scale) * w0; sum.w = (moe_bf2f(h0.w) * first_scale) * w0;
    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row = first_row + expert;
        const float scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        const float w = weights[row];
        const ushort4 h = *(const ushort4 *)(experts + row * n_embd + col4);
        sum.x += (moe_bf2f(h.x) * scale) * w; sum.y += (moe_bf2f(h.y) * scale) * w;
        sum.z += (moe_bf2f(h.z) * scale) * w; sum.w += (moe_bf2f(h.w) * scale) * w;
    }
    if (merge) { const float4 m = *(const float4 *)(merge + token * n_embd + col4); sum.x += m.x; sum.y += m.y; sum.z += m.z; sum.w += m.w; }
    *(ushort4 *)(dst + token * n_embd + col4) = make_ushort4(moe_f2bf(sum.x), moe_f2bf(sum.y), moe_f2bf(sum.z), moe_f2bf(sum.w));
}

static __global__ void moe_weighted_reduction_f32in_bf16out_v4(const float * __restrict__ experts,
                                                              const float * __restrict__ expert_scale,
                                                              const float * __restrict__ weights,
                                                              uint16_t * __restrict__ dst,
                                                              const float * __restrict__ merge,
                                                              const int64_t n_embd, const int n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col4  = ((int64_t) blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (col4 >= n_embd) return;
    const uint64_t first_row = (uint64_t) token * n_expert_used;
    const float first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    const float w0 = weights[first_row];
    const float4 e0 = *(const float4 *)(experts + first_row * n_embd + col4);
    float4 sum; sum.x = (e0.x * first_scale) * w0; sum.y = (e0.y * first_scale) * w0;
    sum.z = (e0.z * first_scale) * w0; sum.w = (e0.w * first_scale) * w0;
    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row = first_row + expert;
        const float scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        const float w = weights[row];
        const float4 e = *(const float4 *)(experts + row * n_embd + col4);
        sum.x += (e.x * scale) * w; sum.y += (e.y * scale) * w; sum.z += (e.z * scale) * w; sum.w += (e.w * scale) * w;
    }
    if (merge) { const float4 m = *(const float4 *)(merge + token * n_embd + col4); sum.x += m.x; sum.y += m.y; sum.z += m.z; sum.w += m.w; }
    *(ushort4 *)(dst + token * n_embd + col4) = make_ushort4(moe_f2bf(sum.x), moe_f2bf(sum.y), moe_f2bf(sum.z), moe_f2bf(sum.w));
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    static const bool use_v4 = true;
    if (use_v4 && n_embd % 4 == 0 && ((uintptr_t) experts % 16) == 0 && ((uintptr_t) dst % 16) == 0) {
        const dim3 blocks(n_tokens, (n_embd / 4 + threads - 1) / threads, 1);
        moe_weighted_reduction_f32_v4<<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
        return;
    }
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst,
                                         const ggml_tensor *         merge) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    const float * mrg = merge ? (const float *) merge->data : nullptr;
    if (ggml_cuda_mmb_blk16() && ggml_cuda_mmb_is_bf16_only(dst)) {
        GGML_ASSERT(n_embd % 4 == 0);
        const bool ein = ggml_cuda_mmb_is_bf16_only(experts);
        constexpr int threads = 256;
        const dim3 blocks(n_tokens, (n_embd / 4 + threads - 1) / threads, 1);
        if (ein) moe_weighted_reduction_bf16_v4_out<<<blocks, threads, 0, stream>>>((const uint16_t *) experts->data,
            expert_scale ? (const float *) expert_scale->data : nullptr, (const float *) weights->data,
            (uint16_t *) dst->data, mrg, n_embd, (int) n_expert_used);
        else     moe_weighted_reduction_f32in_bf16out_v4<<<blocks, threads, 0, stream>>>((const float *) experts->data,
            expert_scale ? (const float *) expert_scale->data : nullptr, (const float *) weights->data,
            (uint16_t *) dst->data, mrg, n_embd, (int) n_expert_used);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    GGML_ASSERT(merge == nullptr && "shared-expert merge is only fused on the BF16 output path");
    if (ggml_cuda_mmb_down16() && ggml_cuda_mmb_is_bf16_only(experts)) {
        GGML_ASSERT(n_embd % 4 == 0 && ((uintptr_t) experts->data % 16) == 0 && ((uintptr_t) dst->data % 16) == 0);
        constexpr int threads = 256;
        const dim3 blocks(n_tokens, (n_embd / 4 + threads - 1) / threads, 1);
        moe_weighted_reduction_bf16_v4<<<blocks, threads, 0, stream>>>((const uint16_t *) experts->data,
            expert_scale ? (const float *) expert_scale->data : nullptr, (const float *) weights->data,
            (float *) dst->data, n_embd, (int) n_expert_used);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}
