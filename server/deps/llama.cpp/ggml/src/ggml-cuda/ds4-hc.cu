#include "ds4-hc.cuh"

// Fused DeepSeek4 hyper-connection ops.
//
// mode 0 (pre):  src0 = mix [mix_dim,n_tokens] (f32, from fn @ rms_norm(hc_state))
//                src1 = base [mix_dim]       (f32)
//                src2 = hc_state [n_embd*n_hc,n_tokens] (f32, raw residual streams)
//                dst  = [n_embd + mix_dim,n_tokens]:
//                       dst[0..n_embd)          = working vector (pre-mixed input)
//                       dst[n_embd..n_embd+mix) = split = {pre[n_hc], post[n_hc], comb[n_hc*n_hc]}
//                Math matches cpu_hc_sinkhorn + finish_hc_pre_from_mix_into in
//                deepseek4_graph.cpp (sigmoid gates + Sinkhorn-normalized combine).
//
// mode 1 (post): src0 = residual hc_state [n_embd*n_hc,n_tokens]
//                src1 = block_out [n_embd,n_tokens]
//                src2 = split [mix_dim,n_tokens] (view of a mode-0 dst tail)
//                dst  = new hc_state [n_embd*n_hc,n_tokens]:
//                       dst[h*n_embd+d] = post[h]*block_out[d]
//                                       + sum_src comb[h + src*n_hc] * residual[src*n_embd+d]
//
// mode 2 (out):  src0 = mix [n_hc,n_tokens]
//                src1 = base [n_hc]
//                src2 = hc_state [n_embd*n_hc,n_tokens]
//                dst  = [n_embd,n_tokens]: weights[h] = sigmoid(mix[h]*s0+base[h]) + 1e-6;
//                       dst[d] = sum_h weights[h]*hc_state[h*n_embd+d]

#define DS4_HC_SINKHORN_EPS 1.0e-6f
#define DS4_HC_MAX_HC 8
#define DS4_HC_MAX_MIX (2*DS4_HC_MAX_HC + DS4_HC_MAX_HC*DS4_HC_MAX_HC)

static __device__ __forceinline__ float ds4_hc_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __device__ void ds4_hc_sinkhorn_split(
        const float * mix,
        const float * base,
        float         pre_scale,
        float         post_scale,
        float         comb_scale,
        int           n_hc,
        int           iters,
        float       * split) {
    for (int i = 0; i < n_hc; ++i) {
        split[i] = ds4_hc_sigmoid(mix[i] * pre_scale + base[i]) + DS4_HC_SINKHORN_EPS;
    }
    for (int i = 0; i < n_hc; ++i) {
        split[n_hc + i] = 2.0f * ds4_hc_sigmoid(mix[n_hc + i] * post_scale + base[n_hc + i]);
    }

    float c[DS4_HC_MAX_HC * DS4_HC_MAX_HC];
    for (int dst_i = 0; dst_i < n_hc; ++dst_i) {
        float row_max = -1.0e30f;
        for (int src_i = 0; src_i < n_hc; ++src_i) {
            const int idx = src_i + dst_i * n_hc;
            const float v = mix[2 * n_hc + idx] * comb_scale + base[2 * n_hc + idx];
            c[idx] = v;
            row_max = v > row_max ? v : row_max;
        }
        float row_sum = 0.0f;
        for (int src_i = 0; src_i < n_hc; ++src_i) {
            const int idx = src_i + dst_i * n_hc;
            c[idx] = expf(c[idx] - row_max);
            row_sum += c[idx];
        }
        const float inv = 1.0f / row_sum;
        for (int src_i = 0; src_i < n_hc; ++src_i) {
            c[src_i + dst_i * n_hc] = c[src_i + dst_i * n_hc] * inv + DS4_HC_SINKHORN_EPS;
        }
    }
    for (int src_i = 0; src_i < n_hc; ++src_i) {
        float sum = 0.0f;
        for (int dst_i = 0; dst_i < n_hc; ++dst_i) sum += c[src_i + dst_i * n_hc];
        const float inv = 1.0f / (sum + DS4_HC_SINKHORN_EPS);
        for (int dst_i = 0; dst_i < n_hc; ++dst_i) c[src_i + dst_i * n_hc] *= inv;
    }
    for (int iter = 1; iter < iters; ++iter) {
        for (int dst_i = 0; dst_i < n_hc; ++dst_i) {
            float sum = 0.0f;
            for (int src_i = 0; src_i < n_hc; ++src_i) sum += c[src_i + dst_i * n_hc];
            const float inv = 1.0f / (sum + DS4_HC_SINKHORN_EPS);
            for (int src_i = 0; src_i < n_hc; ++src_i) c[src_i + dst_i * n_hc] *= inv;
        }
        for (int src_i = 0; src_i < n_hc; ++src_i) {
            float sum = 0.0f;
            for (int dst_i = 0; dst_i < n_hc; ++dst_i) sum += c[src_i + dst_i * n_hc];
            const float inv = 1.0f / (sum + DS4_HC_SINKHORN_EPS);
            for (int dst_i = 0; dst_i < n_hc; ++dst_i) c[src_i + dst_i * n_hc] *= inv;
        }
    }
    for (int i = 0; i < n_hc * n_hc; ++i) {
        split[2 * n_hc + i] = c[i];
    }
}

// Fully-unrolled variant: with compile-time NHC the c[] matrix lives in
// registers. The generic version's runtime-bound loops force c[] into scratch
// (private, VRAM-backed) memory, making the serial sinkhorn ~20x slower
// (97us vs 5us measured on gfx1151).
template<int NHC>
static __device__ void ds4_hc_sinkhorn_split_t(
        const float * mix,
        const float * base,
        float         pre_scale,
        float         post_scale,
        float         comb_scale,
        int           iters,
        float       * split) {
    #pragma unroll
    for (int i = 0; i < NHC; ++i) {
        split[i] = ds4_hc_sigmoid(mix[i] * pre_scale + base[i]) + DS4_HC_SINKHORN_EPS;
    }
    #pragma unroll
    for (int i = 0; i < NHC; ++i) {
        split[NHC + i] = 2.0f * ds4_hc_sigmoid(mix[NHC + i] * post_scale + base[NHC + i]);
    }
    float c[NHC * NHC];
    #pragma unroll
    for (int dst_i = 0; dst_i < NHC; ++dst_i) {
        float row_max = -1.0e30f;
        #pragma unroll
        for (int src_i = 0; src_i < NHC; ++src_i) {
            const int idx = src_i + dst_i * NHC;
            const float v = mix[2 * NHC + idx] * comb_scale + base[2 * NHC + idx];
            c[idx] = v;
            row_max = v > row_max ? v : row_max;
        }
        float row_sum = 0.0f;
        #pragma unroll
        for (int src_i = 0; src_i < NHC; ++src_i) {
            const int idx = src_i + dst_i * NHC;
            c[idx] = expf(c[idx] - row_max);
            row_sum += c[idx];
        }
        const float inv = 1.0f / row_sum;
        #pragma unroll
        for (int src_i = 0; src_i < NHC; ++src_i) {
            c[src_i + dst_i * NHC] = c[src_i + dst_i * NHC] * inv + DS4_HC_SINKHORN_EPS;
        }
    }
    #pragma unroll
    for (int src_i = 0; src_i < NHC; ++src_i) {
        float sum = 0.0f;
        #pragma unroll
        for (int dst_i = 0; dst_i < NHC; ++dst_i) sum += c[src_i + dst_i * NHC];
        const float inv = 1.0f / (sum + DS4_HC_SINKHORN_EPS);
        #pragma unroll
        for (int dst_i = 0; dst_i < NHC; ++dst_i) c[src_i + dst_i * NHC] *= inv;
    }
    for (int iter = 1; iter < iters; ++iter) {
        #pragma unroll
        for (int dst_i = 0; dst_i < NHC; ++dst_i) {
            float sum = 0.0f;
            #pragma unroll
            for (int src_i = 0; src_i < NHC; ++src_i) sum += c[src_i + dst_i * NHC];
            const float inv = 1.0f / (sum + DS4_HC_SINKHORN_EPS);
            #pragma unroll
            for (int src_i = 0; src_i < NHC; ++src_i) c[src_i + dst_i * NHC] *= inv;
        }
        #pragma unroll
        for (int src_i = 0; src_i < NHC; ++src_i) {
            float sum = 0.0f;
            #pragma unroll
            for (int dst_i = 0; dst_i < NHC; ++dst_i) sum += c[src_i + dst_i * NHC];
            const float inv = 1.0f / (sum + DS4_HC_SINKHORN_EPS);
            #pragma unroll
            for (int dst_i = 0; dst_i < NHC; ++dst_i) c[src_i + dst_i * NHC] *= inv;
        }
    }
    #pragma unroll
    for (int i = 0; i < NHC * NHC; ++i) {
        split[2 * NHC + i] = c[i];
    }
}

template<int NHC>
static __global__ void ds4_hc_pre_kernel_t(
        const float * __restrict__ mix,
        const float * __restrict__ base,
        const float * __restrict__ hc_state,
        float       * __restrict__ dst,
        int   n_embd,
        int   iters,
        float pre_scale,
        float post_scale,
        float comb_scale,
        size_t mix_stride,
        size_t hc_stride,
        size_t dst_stride) {
    const int token = (int) blockIdx.y;
    mix     += (size_t) token * mix_stride;
    hc_state += (size_t) token * hc_stride;
    dst     += (size_t) token * dst_stride;

    __shared__ float split[DS4_HC_MAX_MIX];
    __shared__ float s_mix[DS4_HC_MAX_MIX];
    __shared__ float s_base[DS4_HC_MAX_MIX];
    constexpr int mix_dim = 2 * NHC + NHC * NHC;
    const int tid = threadIdx.x;

    if (tid < mix_dim) {
        s_mix[tid]  = mix[tid];
        s_base[tid] = base[tid];
    }
    __syncthreads();

    if (tid == 0) {
        ds4_hc_sinkhorn_split_t<NHC>(s_mix, s_base, pre_scale, post_scale, comb_scale, iters, split);
        if (blockIdx.x == 0) {
            #pragma unroll
            for (int i = 0; i < mix_dim; ++i) {
                dst[n_embd + i] = split[i];
            }
        }
    }
    __syncthreads();

    const int d = (int) blockIdx.x * blockDim.x + tid;
    if (d < n_embd) {
        float acc = 0.0f;
        #pragma unroll
        for (int h = 0; h < NHC; ++h) {
            acc += split[h] * hc_state[(size_t) h * n_embd + d];
        }
        dst[d] = acc;
    }
}

// Large prefill batches already expose thousands of token blocks, so they do
// not need every embedding tile to recompute the same serial Sinkhorn. Split
// it into one deterministic solve per token followed by the parallel mixing
// pass. The small-token path keeps the fused kernel above to avoid an extra
// launch during decode and speculative verification.
template<int NHC>
static __global__ void ds4_hc_pre_split_kernel_t(
        const float * __restrict__ mix,
        const float * __restrict__ base,
        float       * __restrict__ dst,
        int   n_embd,
        int   iters,
        float pre_scale,
        float post_scale,
        float comb_scale,
        size_t mix_stride,
        size_t dst_stride) {
    const int token = (int) blockIdx.x;
    mix += (size_t) token * mix_stride;
    dst += (size_t) token * dst_stride;

    __shared__ float split[DS4_HC_MAX_MIX];
    __shared__ float s_mix[DS4_HC_MAX_MIX];
    __shared__ float s_base[DS4_HC_MAX_MIX];
    constexpr int mix_dim = 2 * NHC + NHC * NHC;
    const int tid = (int) threadIdx.x;

    if (tid < mix_dim) {
        s_mix[tid] = mix[tid];
        s_base[tid] = base[tid];
    }
    __syncthreads();

    if (tid == 0) {
        ds4_hc_sinkhorn_split_t<NHC>(
            s_mix, s_base, pre_scale, post_scale, comb_scale, iters, split);
#pragma unroll
        for (int i = 0; i < mix_dim; ++i) {
            dst[n_embd + i] = split[i];
        }
    }
}

template<int NHC>
static __global__ void ds4_hc_pre_mix_kernel_t(
        const float * __restrict__ hc_state,
        float       * __restrict__ dst,
        int    n_embd,
        size_t hc_stride,
        size_t dst_stride) {
    const int token = (int) blockIdx.y;
    hc_state += (size_t) token * hc_stride;
    dst += (size_t) token * dst_stride;

    __shared__ float pre[NHC];
    const int tid = (int) threadIdx.x;
    if (tid < NHC) {
        pre[tid] = dst[n_embd + tid];
    }
    __syncthreads();

    const int d = (int) blockIdx.x * (int) blockDim.x + tid;
    if (d < n_embd) {
        float acc = 0.0f;
#pragma unroll
        for (int h = 0; h < NHC; ++h) {
            acc += pre[h] * hc_state[(size_t) h * n_embd + d];
        }
        dst[d] = acc;
    }
}

static __global__ void ds4_hc_pre_kernel(
        const float * __restrict__ mix,
        const float * __restrict__ base,
        const float * __restrict__ hc_state,
        float       * __restrict__ dst,
        int   n_embd,
        int   n_hc,
        int   iters,
        float pre_scale,
        float post_scale,
        float comb_scale,
        size_t mix_stride,
        size_t hc_stride,
        size_t dst_stride) {
    const int token = (int) blockIdx.y;
    mix     += (size_t) token * mix_stride;
    hc_state += (size_t) token * hc_stride;
    dst     += (size_t) token * dst_stride;

    __shared__ float split[DS4_HC_MAX_MIX];
    __shared__ float s_mix[DS4_HC_MAX_MIX];
    __shared__ float s_base[DS4_HC_MAX_MIX];
    const int mix_dim = 2 * n_hc + n_hc * n_hc;
    const int tid = threadIdx.x;

    // Stage mix/base cooperatively: base lives in managed (UMA) memory where
    // serial scalar loads cost ~2us each; one parallel coalesced load instead.
    if (tid < mix_dim) {
        s_mix[tid]  = mix[tid];
        s_base[tid] = base[tid];
    }
    __syncthreads();

    // Each block redoes the (tiny) sinkhorn into shared memory so the mix
    // loop below can spread across the whole GPU instead of one CU.
    if (tid == 0) {
        ds4_hc_sinkhorn_split(s_mix, s_base, pre_scale, post_scale, comb_scale, n_hc, iters, split);
        if (blockIdx.x == 0) {
            for (int i = 0; i < mix_dim; ++i) {
                dst[n_embd + i] = split[i];
            }
        }
    }
    __syncthreads();

    const int d = (int) blockIdx.x * blockDim.x + tid;
    if (d < n_embd) {
        float acc = 0.0f;
        for (int h = 0; h < n_hc; ++h) {
            acc += split[h] * hc_state[(size_t) h * n_embd + d];
        }
        dst[d] = acc;
    }
}

static __global__ void ds4_hc_post_kernel(
        const float * __restrict__ residual,
        const float * __restrict__ block_out,
        const float * __restrict__ split,
        float       * __restrict__ dst,
        int n_embd,
        int n_hc,
        size_t residual_stride,
        size_t block_out_stride,
        size_t split_stride,
        size_t dst_stride) {
    const int token = (int) blockIdx.y;
    residual += (size_t) token * residual_stride;
    block_out += (size_t) token * block_out_stride;
    split += (size_t) token * split_stride;
    dst += (size_t) token * dst_stride;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_embd * n_hc;
    if (i >= total) {
        return;
    }
    const int h = i / n_embd;
    const int d = i - h * n_embd;
    const float * post = split + n_hc;
    const float * comb = split + 2 * n_hc;
    float acc = block_out[d] * post[h];
    for (int src = 0; src < n_hc; ++src) {
        acc += comb[h + src * n_hc] * residual[(size_t) src * n_embd + d];
    }
    dst[i] = acc;
}

static __global__ void ds4_hc_out_kernel(
        const float * __restrict__ mix,
        const float * __restrict__ base,
        const float * __restrict__ hc_state,
        float       * __restrict__ dst,
        int   n_embd,
        int   n_hc,
        float pre_scale,
        size_t mix_stride,
        size_t hc_stride,
        size_t dst_stride) {
    const int token = (int) blockIdx.y;
    mix += (size_t) token * mix_stride;
    hc_state += (size_t) token * hc_stride;
    dst += (size_t) token * dst_stride;

    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) {
        return;
    }
    float acc = 0.0f;
    for (int h = 0; h < n_hc; ++h) {
        const float wgt = ds4_hc_sigmoid(mix[h] * pre_scale + base[h]) + DS4_HC_SINKHORN_EPS;
        acc += wgt * hc_state[(size_t) h * n_embd + d];
    }
    dst[d] = acc;
}

void ggml_cuda_op_ds4_hc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0 && src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1 && src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2 && src2->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int mode   = ggml_get_op_params_i32(dst, 0);
    const int n_embd = ggml_get_op_params_i32(dst, 1);
    const int n_hc   = ggml_get_op_params_i32(dst, 2);
    const int n_tokens = (int) dst->ne[1];

    GGML_ASSERT(n_hc > 0 && n_hc <= DS4_HC_MAX_HC);
    GGML_ASSERT(n_tokens > 0);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src2->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[0] == sizeof(float));

    cudaStream_t stream = ctx.stream();

    switch (mode) {
        case 0: {
            const int   iters      = ggml_get_op_params_i32(dst, 3);
            const float pre_scale  = ggml_get_op_params_f32(dst, 4);
            const float post_scale = ggml_get_op_params_f32(dst, 5);
            const float comb_scale = ggml_get_op_params_f32(dst, 6);
            const int pre_blocks = (n_embd + 255) / 256;
            const dim3 grid(pre_blocks, n_tokens, 1);
            if (n_hc == 4 && n_tokens >= 64) {
                ds4_hc_pre_split_kernel_t<4><<<n_tokens, 256, 0, stream>>>(
                    (const float *) src0->data, (const float *) src1->data,
                    (float *) dst->data,
                    n_embd, iters, pre_scale, post_scale, comb_scale,
                    src0->nb[1] / sizeof(float), dst->nb[1] / sizeof(float));
                ds4_hc_pre_mix_kernel_t<4><<<grid, 256, 0, stream>>>(
                    (const float *) src2->data, (float *) dst->data,
                    n_embd, src2->nb[1] / sizeof(float),
                    dst->nb[1] / sizeof(float));
            } else if (n_hc == 4) {
                ds4_hc_pre_kernel_t<4><<<grid, 256, 0, stream>>>(
                    (const float *) src0->data, (const float *) src1->data,
                    (const float *) src2->data, (float *) dst->data,
                    n_embd, iters, pre_scale, post_scale, comb_scale,
                    src0->nb[1] / sizeof(float), src2->nb[1] / sizeof(float),
                    dst->nb[1] / sizeof(float));
            } else {
                ds4_hc_pre_kernel<<<grid, 256, 0, stream>>>(
                    (const float *) src0->data, (const float *) src1->data,
                    (const float *) src2->data, (float *) dst->data,
                    n_embd, n_hc, iters, pre_scale, post_scale, comb_scale,
                    src0->nb[1] / sizeof(float), src2->nb[1] / sizeof(float),
                    dst->nb[1] / sizeof(float));
            }
        } break;
        case 1: {
            const int total = n_embd * n_hc;
            const int blocks = (total + 255) / 256;
            const dim3 grid(blocks, n_tokens, 1);
            ds4_hc_post_kernel<<<grid, 256, 0, stream>>>(
                (const float *) src0->data, (const float *) src1->data,
                (const float *) src2->data, (float *) dst->data,
                n_embd, n_hc,
                src0->nb[1] / sizeof(float), src1->nb[1] / sizeof(float),
                src2->nb[1] / sizeof(float), dst->nb[1] / sizeof(float));
        } break;
        case 2: {
            const float pre_scale = ggml_get_op_params_f32(dst, 4);
            const int blocks = (n_embd + 255) / 256;
            const dim3 grid(blocks, n_tokens, 1);
            ds4_hc_out_kernel<<<grid, 256, 0, stream>>>(
                (const float *) src0->data, (const float *) src1->data,
                (const float *) src2->data, (float *) dst->data,
                n_embd, n_hc, pre_scale,
                src0->nb[1] / sizeof(float), src2->nb[1] / sizeof(float),
                dst->nb[1] / sizeof(float));
        } break;
        default:
            GGML_ABORT("ds4_hc: unknown mode");
    }
}
