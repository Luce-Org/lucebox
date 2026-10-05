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
//
// mode 3 (post split): mode 1 with src1 = main block_out, src3 = peer
//                block_out. The kernel evaluates peer[d] + main[d] before
//                multiplying by post[h], matching the eliminated GGML add.

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

static __global__ void ds4_hc_post_split_kernel(
        const float * __restrict__ residual,
        const float * __restrict__ main_block,
        const float * __restrict__ peer_block,
        const float * __restrict__ split,
        float       * __restrict__ dst,
        int n_embd,
        int n_hc,
        size_t residual_stride,
        size_t main_block_stride,
        size_t peer_block_stride,
        size_t split_stride,
        size_t dst_stride) {
    const int token = (int) blockIdx.y;
    residual += (size_t) token * residual_stride;
    main_block += (size_t) token * main_block_stride;
    peer_block += (size_t) token * peer_block_stride;
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
    // Keep the established reduction order: the graph being replaced uses
    // ggml_add(peer, main), followed by HC post multiplication.
    const float block_out = peer_block[d] + main_block[d];
    float acc = block_out * post[h];
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

// Modes 4/5: the DS4 router (ggml_ds4_router_select / _weights). Same expressions as op_softplus / op_sqrt /
// op_add / op_clamp / op_div / scale_f32 and the same bitonic network as
// k_argsort_f32_i32 (descending), so the selection and weights are identical
// (test_ds4_fused_ops_cuda). No expression here may contract to an FMA:
// the one product-plus-sum uses explicit rounding intrinsics.
static __device__ __forceinline__ float ds4_router_prob(float x) {
    const float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
    return sqrtf(sp);
}

template <int NPAD>
static __device__ void ds4_router_sort_desc(int * idx, float * val, int n, int col) {
    for (int kk = 2; kk <= NPAD; kk *= 2) {
        for (int j = kk / 2; j > 0; j /= 2) {
            const int ixj = col ^ j;
            if (ixj > col) {
                if ((col & kk) == 0) {
                    if (idx[col] >= n || (idx[ixj] < n && val[col] < val[ixj])) {
                        const int ti = idx[col]; idx[col] = idx[ixj]; idx[ixj] = ti;
                        const float tv = val[col]; val[col] = val[ixj]; val[ixj] = tv;
                    }
                } else {
                    if (idx[ixj] >= n || (idx[col] < n && val[col] > val[ixj])) {
                        const int ti = idx[col]; idx[col] = idx[ixj]; idx[ixj] = ti;
                        const float tv = val[col]; val[col] = val[ixj]; val[ixj] = tv;
                    }
                }
            }
            __syncthreads();
        }
    }
}

template <int NPAD>
static __global__ void ds4_router_select_kernel(
        const float * __restrict__ logits, size_t ld, const float * __restrict__ bias,
        const float * __restrict__ native_bias, const int32_t * __restrict__ protected_mask,
        int n_expert, int k, int32_t * __restrict__ dst) {
    __shared__ int idx[NPAD];
    __shared__ float val[NPAD];
    __shared__ float probs[NPAD];
    __shared__ int native_top[32];
    __shared__ int keep_native;
    const int t = blockIdx.x;
    const int col = threadIdx.x;
    if (col < n_expert) probs[col] = ds4_router_prob(logits[(size_t) t * ld + col]);
    if (col == 0) keep_native = 0;
    __syncthreads();
    if (native_bias) {
        idx[col] = col;
        if (col < n_expert) val[col] = probs[col] + native_bias[col];
        __syncthreads();
        ds4_router_sort_desc<NPAD>(idx, val, n_expert, col);
        if (col < k) {
            native_top[col] = idx[col];
            const int e = idx[col];
            if (e >= 0 && e < n_expert && protected_mask[e] != 0) atomicOr(&keep_native, 1);
        }
        __syncthreads();
    }
    idx[col] = col;
    if (col < n_expert) val[col] = probs[col] + bias[col];
    __syncthreads();
    ds4_router_sort_desc<NPAD>(idx, val, n_expert, col);
    if (col < k) dst[(size_t) t * k + col] = keep_native ? native_top[col] : idx[col];
}

static __global__ void ds4_router_weights_kernel(
        const float * __restrict__ logits, size_t ld, const int32_t * __restrict__ ids,
        int k, float clamp_min, float scale, int apply_scale, float * __restrict__ dst) {
    // One thread per token: k <= 32 values, reduced in reduce_rows' order.
    const int t = blockIdx.x;
    float w[32];
    float v[32];
#pragma unroll
    for (int l = 0; l < 32; ++l) {
        w[l] = l < k ? ds4_router_prob(logits[(size_t) t * ld + ids[(size_t) t * k + l]]) : 0.0f;
        v[l] = w[l];
    }
    // reduce_rows_f32 on a k-wide row: lane l holds element l, then the
    // 32-lane xor butterfly (the second block stage adds zeros only).
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float n[32];
#pragma unroll
        for (int l = 0; l < 32; ++l) n[l] = v[l] + v[l ^ off];
#pragma unroll
        for (int l = 0; l < 32; ++l) v[l] = n[l];
    }
    float sum = v[0];
    sum = fminf(fmaxf(sum, clamp_min), INFINITY);
    for (int i = 0; i < k; ++i) {
        float r = w[i] / sum;
        if (apply_scale) r = __fadd_rn(__fmul_rn(scale, r), 0.0f);  // scale_f32: scale * x + 0
        dst[(size_t) t * k + i] = r;
    }
}

// Mode 6: the staggered HC collapse. sum_rows over a 4-wide row of products
// is one wave32 butterfly, (x0 + x2) + (x1 + x3); other widths replay the
// whole butterfly with zeros past n_hc. A product must not fuse into the
// following add: __fmul_rn keeps it a separate rounding on CUDA too, where
// nvcc ignores the clang pragma and builds with fast math.
static __global__ void ds4_hc_collapse_kernel(
        const float * __restrict__ hc, const float * __restrict__ pre, float * __restrict__ dst,
        int n_embd, int n_hc, int n_tokens, size_t hc_stride, size_t pre_stride, size_t dst_stride) {
#pragma clang fp contract(off)
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    // grid.y is capped at 65535: a longer batch strides over its tokens.
    for (int t = blockIdx.y; t < n_tokens; t += gridDim.y) {
        const float * h = hc + (size_t) t * hc_stride + d;
        const float * p = pre + (size_t) t * pre_stride;
        if (n_hc == 4) {
            const float v0 = __fmul_rn(h[0], p[0]);
            const float v1 = __fmul_rn(h[(size_t) n_embd], p[1]);
            const float v2 = __fmul_rn(h[2 * (size_t) n_embd], p[2]);
            const float v3 = __fmul_rn(h[3 * (size_t) n_embd], p[3]);
            dst[(size_t) t * dst_stride + d] = (v0 + v2) + (v1 + v3);
            continue;
        }
        float v[32];
#pragma unroll
        for (int l = 0; l < 32; ++l) v[l] = l < n_hc ? __fmul_rn(h[(size_t) l * n_embd], p[l]) : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float n[32];
#pragma unroll
            for (int l = 0; l < 32; ++l) n[l] = v[l] + v[l ^ off];
#pragma unroll
            for (int l = 0; l < 32; ++l) v[l] = n[l];
        }
        dst[(size_t) t * dst_stride + d] = v[0];
    }
}

// GGML_OP_DS4_HC modes (op_params[0]): 0 hc_pre, 1 hc_post, 2 hc_out,
// 3 hc_post_split, 4 router_select, 5 router_weights, 6 hc_collapse.
void ggml_cuda_op_ds4_hc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    if (ggml_get_op_params_i32(dst, 0) == 4) {
        const ggml_tensor * logits = dst->src[0];
        const int k = ggml_get_op_params_i32(dst, 1);
        const int n_expert = (int) logits->ne[0];
        const int n_tokens = (int) logits->ne[1];
        const float * nb = dst->src[2] ? (const float *) dst->src[2]->data : nullptr;
        const int32_t * pm = dst->src[3] ? (const int32_t *) dst->src[3]->data : nullptr;
        if (n_expert <= 512) {
            ds4_router_select_kernel<512><<<n_tokens, 512, 0, ctx.stream()>>>(
                (const float *) logits->data, logits->nb[1] / sizeof(float),
                (const float *) dst->src[1]->data, nb, pm, n_expert, k, (int32_t *) dst->data);
        } else {
            ds4_router_select_kernel<1024><<<n_tokens, 1024, 0, ctx.stream()>>>(
                (const float *) logits->data, logits->nb[1] / sizeof(float),
                (const float *) dst->src[1]->data, nb, pm, n_expert, k, (int32_t *) dst->data);
        }
        return;
    }
    if (ggml_get_op_params_i32(dst, 0) == 6) {
        const ggml_tensor * hc = dst->src[0];
        const ggml_tensor * pre = dst->src[1];
        const int n_embd = ggml_get_op_params_i32(dst, 1);
        const int n_hc = ggml_get_op_params_i32(dst, 2);
        const int n_tokens = (int) dst->ne[1];
        const dim3 grid((n_embd + 255) / 256, (unsigned) (n_tokens < 65535 ? n_tokens : 65535), 1);
        ds4_hc_collapse_kernel<<<grid, 256, 0, ctx.stream()>>>(
            (const float *) hc->data, (const float *) pre->data, (float *) dst->data,
            n_embd, n_hc, n_tokens, hc->nb[1] / sizeof(float), pre->nb[1] / sizeof(float),
            dst->nb[1] / sizeof(float));
        return;
    }
    if (ggml_get_op_params_i32(dst, 0) == 5) {
        const ggml_tensor * logits = dst->src[0];
        const ggml_tensor * ids = dst->src[1];
        const float clamp_min = ggml_get_op_params_f32(dst, 4);
        const float scale = ggml_get_op_params_f32(dst, 5);
        ds4_router_weights_kernel<<<(int) ids->ne[1], 1, 0, ctx.stream()>>>(
            (const float *) logits->data, logits->nb[1] / sizeof(float), (const int32_t *) ids->data,
            (int) ids->ne[0], clamp_min, scale, scale != 1.0f ? 1 : 0, (float *) dst->data);
        return;
    }

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
        case 3: {
            const ggml_tensor * src3 = dst->src[3];
            GGML_ASSERT(src3 && src3->type == GGML_TYPE_F32);
            const int total = n_embd * n_hc;
            const int blocks = (total + 255) / 256;
            const dim3 grid(blocks, n_tokens, 1);
            ds4_hc_post_split_kernel<<<grid, 256, 0, stream>>>(
                (const float *) src0->data, (const float *) src1->data,
                (const float *) src3->data, (const float *) src2->data,
                (float *) dst->data, n_embd, n_hc,
                src0->nb[1] / sizeof(float), src1->nb[1] / sizeof(float),
                src3->nb[1] / sizeof(float), src2->nb[1] / sizeof(float),
                dst->nb[1] / sizeof(float));
        } break;
        default:
            GGML_ABORT("ds4_hc: unknown mode");
    }
}
