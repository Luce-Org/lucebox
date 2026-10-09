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

// The weights of one token's selected routes: ds4_router_weights_kernel's
// arithmetic for token t (one thread).
static __device__ __forceinline__ void ds4_router_weights_token(
        const float * __restrict__ logits, size_t ld, const int32_t * __restrict__ ids,
        int k, float clamp_min, float scale, int apply_scale, float * __restrict__ dst, int t) {
    // One thread per token: k <= 32 values, reduced in reduce_rows' order.
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

// WITH_WEIGHTS: thread 0 then writes the routes' weights as
// ds4_router_weights_kernel would (the router's two launches in one).
template <int NPAD, bool WITH_WEIGHTS = false>
static __global__ void ds4_router_select_kernel(
        const float * __restrict__ logits, size_t ld, const float * __restrict__ bias,
        const float * __restrict__ native_bias, const int32_t * __restrict__ protected_mask,
        int n_expert, int k, int32_t * __restrict__ dst,
        float clamp_min = 0.0f, float scale = 1.0f, float * __restrict__ weights_dst = nullptr) {
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
    if constexpr (WITH_WEIGHTS) {
        __syncthreads();
        __threadfence_block();
        if (col == 0) {
            ds4_router_weights_token(logits, ld, dst, k, clamp_min, scale, scale != 1.0f ? 1 : 0,
                                     weights_dst, t);
        }
    }
}

static __global__ void ds4_router_weights_kernel(
        const float * __restrict__ logits, size_t ld, const int32_t * __restrict__ ids,
        int k, float clamp_min, float scale, int apply_scale, float * __restrict__ dst) {
    ds4_router_weights_token(logits, ld, ids, k, clamp_min, scale, apply_scale, dst, (int) blockIdx.x);
}

// The router's select and weights ops (modes 4 and 5) over the same logits,
// in one launch.
void ggml_cuda_op_ds4_router_fused(ggml_backend_cuda_context & ctx, ggml_tensor * select, ggml_tensor * weights) {
    const ggml_tensor * logits = select->src[0];
    const int k = ggml_get_op_params_i32(select, 1);
    const int n_expert = (int) logits->ne[0];
    const int n_tokens = (int) logits->ne[1];
    const float * nb = select->src[2] ? (const float *) select->src[2]->data : nullptr;
    const int32_t * pm = select->src[3] ? (const int32_t *) select->src[3]->data : nullptr;
    const float clamp_min = ggml_get_op_params_f32(weights, 4);
    const float scale = ggml_get_op_params_f32(weights, 5);
    if (n_expert <= 512) {
        ds4_router_select_kernel<512, true><<<n_tokens, 512, 0, ctx.stream()>>>(
            (const float *) logits->data, logits->nb[1] / sizeof(float),
            (const float *) select->src[1]->data, nb, pm, n_expert, k, (int32_t *) select->data,
            clamp_min, scale, (float *) weights->data);
    } else {
        ds4_router_select_kernel<1024, true><<<n_tokens, 1024, 0, ctx.stream()>>>(
            (const float *) logits->data, logits->nb[1] / sizeof(float),
            (const float *) select->src[1]->data, nb, pm, n_expert, k, (int32_t *) select->data,
            clamp_min, scale, (float *) weights->data);
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

// ── One HC sub-block boundary per launch ────────────────────────────────
// The graph builds the end of a staggered (V4.1) sub-block X and the start of
// the next one as six launches:
//   n1   = rms_norm(R)                       R: the sub-block's HC input streams
//   mix  = fn @ n1                           F16 weights, MMVF
//   pre  = ds4_hc_pre(mix, base, R)          Sinkhorn split (+ working vector)
//   R'   = ds4_hc_post(R, block_out, split)  the next HC streams
//   in   = ds4_hc_collapse(R', pre[0:n_hc])  the next sub-block's input
//   out  = rms_norm(in) * w                  ... normalized for its projections
// ds4_hc_boundary_kernel does all six for one token per block. Each phase
// repeats the arithmetic of the kernel it replaces in the same order:
// rms_norm_f32<1024> (thread col mapping and block reduction), MMVF
// <half, half acc, MMV_BLOCK> (per-thread half2 partial sums, warp then
// cross-warp reduction), ds4_hc_pre_kernel_t, ds4_hc_post_kernel,
// ds4_hc_collapse_kernel (contraction off) and rms_norm_f32<1024, mul>, so
// the outputs are bit-identical (test_ds4_fused_ops_cuda).

// The collapse of one embedding column: ds4_hc_collapse_kernel's arithmetic.
template <int NHC>
static __device__ __forceinline__ float ds4_hc_collapse_value(const float * v, const float * p) {
#pragma clang fp contract(off)
    if constexpr (NHC == 4) {
        const float v0 = __fmul_rn(v[0], p[0]);
        const float v1 = __fmul_rn(v[1], p[1]);
        const float v2 = __fmul_rn(v[2], p[2]);
        const float v3 = __fmul_rn(v[3], p[3]);
        return (v0 + v2) + (v1 + v3);
    } else {
        float w[32];
#pragma unroll
        for (int l = 0; l < 32; ++l) w[l] = l < NHC ? __fmul_rn(v[l], p[l]) : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float n[32];
#pragma unroll
            for (int l = 0; l < 32; ++l) n[l] = w[l] + w[l ^ off];
#pragma unroll
            for (int l = 0; l < 32; ++l) w[l] = n[l];
        }
        return w[0];
    }
}

#define DS4_HC_BOUNDARY_MAX_COLS 8   // n_embd <= 8 * 1024

// An F32 value the compiler must materialize as it stands: the replaced
// kernels store it to memory before the next one reads it, so it must not be
// fused into a following conversion (v_fma_mix) or FMA.
static __device__ __forceinline__ float ds4_hc_f32(float x) {
#if defined(GGML_USE_HIP)
    asm volatile("" : "+v"(x));
#else
    asm volatile("" : "+f"(x));
#endif
    return x;
}

// Every block of the grid waits here for the others (one block per token
// reaches it). The barrier keeps its state on the device only: the last block
// to arrive zeroes the count and advances the generation the others wait on,
// so it needs no per-launch target from the host and a CUDA graph can replay
// the launch. All the grid's blocks must be resident at once.
static __device__ __forceinline__ void ds4_hc_grid_barrier(unsigned int * barrier, const unsigned int n_blocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int * count = barrier;
        unsigned int * generation = barrier + 1;
        const unsigned int gen = atomicAdd(generation, 0u);
        __threadfence();
        if (atomicAdd(count, 1u) == n_blocks - 1) {
            atomicExch(count, 0u);
            __threadfence();
            atomicAdd(generation, 1u);
        } else {
            while (atomicAdd(generation, 0u) == gen) {
#if defined(GGML_USE_HIP)
                __builtin_amdgcn_s_sleep(1);
#endif
            }
        }
        __threadfence();
    }
    __syncthreads();
}

template <int NHC, int MMV_BLOCK>
static __global__ void __launch_bounds__(1024) ds4_hc_boundary_kernel(
        const float * __restrict__ R,         int64_t R_stride,
        const half  * __restrict__ fn,        int64_t fn_stride,
        const float * __restrict__ base,
        const float * __restrict__ block_out, int64_t bo_stride,
        const float * __restrict__ w,
        float       * __restrict__ pre_dst,   int64_t pre_stride,
        float       * __restrict__ R_next,    int64_t Rn_stride,
        float       *              out,       int64_t out_stride,
        const int n_embd, const float eps_hc, const float eps_out, const int iters,
        const float pre_scale, const float post_scale, const float comb_scale,
        const int write_pre,
        unsigned int * barrier,
        block_q8_1 * __restrict__ q8, const int64_t q8_row_blocks) {
    constexpr int block_size = 1024;
    constexpr int mix_dim = 2 * NHC + NHC * NHC;
    constexpr int n_groups = block_size / MMV_BLOCK;
    constexpr int warps_per_group = MMV_BLOCK / WARP_SIZE;
    const int token = (int) blockIdx.x;
    const int tid = (int) threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int ncols_hc = n_embd * NHC;

    R         += (int64_t) token * R_stride;
    block_out += (int64_t) token * bo_stride;
    R_next    += (int64_t) token * Rn_stride;
    out       += (int64_t) token * out_stride;
    if (write_pre) pre_dst += (int64_t) token * pre_stride;

    extern __shared__ half2 s_y2[];               // normalized R, as MMVF reads it
    __shared__ float s_sum[32];
    __shared__ float s_sum_out[32];
    __shared__ float s_part[mix_dim][warps_per_group];
    __shared__ float s_mix[DS4_HC_MAX_MIX];
    __shared__ float s_base[DS4_HC_MAX_MIX];
    __shared__ float split[DS4_HC_MAX_MIX];

    // n1 = rms_norm(R): rms_norm_f32<1024>.
    float tmp = 0.0f;
    for (int col = tid; col < ncols_hc; col += block_size) {
        const float xi = R[col];
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float mean  = tmp / ncols_hc;
    const float scale = rsqrtf(mean + eps_hc);
    // MMVF reads F32 pairs and rounds each to half (make_half2 of two floats).
    const float2 * R2 = (const float2 *) R;
    for (int c2 = tid; c2 < ncols_hc / 2; c2 += block_size) {
        const float2 x = R2[c2];
        // rms_norm_f32 stores scale * x as F32; MMVF then rounds it to half.
        const float y0 = ds4_hc_f32(scale * x.x);
        const float y1 = ds4_hc_f32(scale * x.y);
        s_y2[c2] = make_half2(y0, y1);
    }
    __syncthreads();

    // mix = fn @ n1: each group of MMV_BLOCK threads is one MMVF block.
    {
        const int g  = tid / MMV_BLOCK;
        const int lt = tid % MMV_BLOCK;
        const int ncols2 = ncols_hc / 2;
        for (int r = g; r < mix_dim; r += n_groups) {
            const half2 * x2 = (const half2 *) (fn + (int64_t) r * fn_stride);
            half2 sumh2 = make_half2(0.0f, 0.0f);
#pragma unroll 8
            for (int col2 = lt; col2 < ncols2; col2 += MMV_BLOCK) {
                const half2 tmpx = x2[col2];
                sumh2 += tmpx * s_y2[col2];
            }
            float sumf = __low2float(sumh2) + __high2float(sumh2);
            sumf = warp_reduce_sum<WARP_SIZE>(sumf);
            if (lane == 0) {
                s_part[r][lt / WARP_SIZE] = sumf;
            }
        }
    }
    __syncthreads();
    if (tid / WARP_SIZE < mix_dim) {
        // MMVF's cross-warp stage: the warp sums in lanes 0..warps-1, zeros above.
        const int r = tid / WARP_SIZE;
        float v = lane < warps_per_group ? s_part[r][lane] : 0.0f;
        v = warp_reduce_sum<WARP_SIZE>(v);
        if (lane == 0) {
            s_mix[r] = v;
        }
    }
    if (tid < mix_dim) {
        s_base[tid] = base[tid];
    }
    __syncthreads();

    // pre = ds4_hc_pre(mix, base, R): ds4_hc_pre_kernel_t<NHC>.
    if (tid == 0) {
        ds4_hc_sinkhorn_split_t<NHC>(s_mix, s_base, pre_scale, post_scale, comb_scale, iters, split);
        if (write_pre) {
#pragma unroll
            for (int i = 0; i < mix_dim; ++i) {
                pre_dst[n_embd + i] = split[i];
            }
        }
    }
    __syncthreads();
    if (write_pre) {
        for (int d = tid; d < n_embd; d += block_size) {
            float acc = 0.0f;
#pragma unroll
            for (int h = 0; h < NHC; ++h) {
                acc += split[h] * R[(int64_t) h * n_embd + d];
            }
            pre_dst[d] = acc;
        }
    }

    // R' = ds4_hc_post(R, block_out, split); in = collapse(R', pre). The
    // column mapping is rms_norm_f32<1024>'s over n_embd, and in stays in
    // registers: out may share memory with R or block_out (the unfused
    // allocation frees both before out), so it is written only after every
    // block has finished reading them.
    const float * post = split + NHC;
    const float * comb = split + 2 * NHC;
    float xin[DS4_HC_BOUNDARY_MAX_COLS];
    float tmp_out = 0.0f;
#pragma unroll
    for (int k = 0; k < DS4_HC_BOUNDARY_MAX_COLS; ++k) {
        const int d = tid + k * block_size;
        xin[k] = 0.0f;
        if (d < n_embd) {
            float rn[NHC];
#pragma unroll
            for (int h = 0; h < NHC; ++h) {
                // ds4_hc_post_kernel compiles to one multiply, then an FMA
                // per source copy accumulating into it: spelled out here so
                // the unrolled loop cannot contract the other way.
                float acc = __fmul_rn(block_out[d], post[h]);
                for (int src = 0; src < NHC; ++src) {
                    acc = __fmaf_rn(comb[h + src * NHC], R[(int64_t) src * n_embd + d], acc);
                }
                rn[h] = acc;
                R_next[(int64_t) h * n_embd + d] = acc;
            }
            const float xi = ds4_hc_collapse_value<NHC>(rn, split);
            xin[k] = xi;
            tmp_out += xi * xi;
        }
    }
    // out = rms_norm(in) * w: rms_norm_f32<1024, mul>.
    tmp_out = block_reduce<block_reduce_method::SUM, block_size>(tmp_out, s_sum_out);
    const float mean_out  = tmp_out / n_embd;
    const float scale_out = rsqrtf(mean_out + eps_out);

    ds4_hc_grid_barrier(barrier, gridDim.x);
#pragma unroll
    for (int k = 0; k < DS4_HC_BOUNDARY_MAX_COLS; ++k) {
        const int d = tid + k * block_size;
        if (d < n_embd) {
            const float xo = scale_out * xin[k] * w[d];
            out[d] = xo;
            if (q8) {
                // quantize_q8_1 of out: a warp holds one 32-wide block (d is
                // warp-contiguous), reduced in the same butterfly order.
                float amax = fabsf(xo);
                float sum = xo;
                amax = warp_reduce_max<QK8_1>(amax);
                sum  = warp_reduce_sum<QK8_1>(sum);
                const float  dq = amax / 127.0f;
                const int8_t q  = amax == 0.0f ? 0 : roundf(xo / dq);
                block_q8_1 * yb = q8 + (int64_t) token * q8_row_blocks + d / QK8_1;
                const int iqs = d % QK8_1;
                yb->qs[iqs] = q;
                if (iqs == 0) {
                    yb->ds = make_half2(dq, sum);
                }
            }
        }
    }
}

template <int NHC, int MMV_BLOCK, int NSPLIT>
static __global__ void __launch_bounds__(1024) ds4_hc_boundary_split_kernel(
        const float * __restrict__ R,         int64_t R_stride,
        const half  * __restrict__ fn,        int64_t fn_stride,
        const float * __restrict__ base,
        const float * __restrict__ block_out, int64_t bo_stride,
        const float * __restrict__ w,
        float       * __restrict__ pre_dst,   int64_t pre_stride,
        float       * __restrict__ R_next,    int64_t Rn_stride,
        float       *              out,       int64_t out_stride,
        const int n_embd, const float eps_hc, const float eps_out, const int iters,
        const float pre_scale, const float post_scale, const float comb_scale,
        const int write_pre,
        unsigned int * barrier,
        block_q8_1 * __restrict__ q8, const int64_t q8_row_blocks,
        float * __restrict__ mix_scratch, unsigned int * __restrict__ rows_done) {
    constexpr int block_size = 1024;
    constexpr int mix_dim = 2 * NHC + NHC * NHC;
    constexpr int n_groups = block_size / MMV_BLOCK;
    constexpr int warps_per_group = MMV_BLOCK / WARP_SIZE;
    const int token = (int) blockIdx.x;
    const int tid = (int) threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int ncols_hc = n_embd * NHC;

    R         += (int64_t) token * R_stride;
    block_out += (int64_t) token * bo_stride;
    R_next    += (int64_t) token * Rn_stride;
    out       += (int64_t) token * out_stride;
    if (write_pre) pre_dst += (int64_t) token * pre_stride;

    extern __shared__ half2 s_y2[];               // normalized R, as MMVF reads it
    __shared__ float s_sum[32];
    __shared__ float s_sum_out[32];
    __shared__ float s_part[n_groups][warps_per_group];
    __shared__ float s_mix[DS4_HC_MAX_MIX];
    __shared__ float s_base[DS4_HC_MAX_MIX];
    __shared__ float split[DS4_HC_MAX_MIX];

    // n1 = rms_norm(R): rms_norm_f32<1024>.
    float tmp = 0.0f;
    for (int col = tid; col < ncols_hc; col += block_size) {
        const float xi = R[col];
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float mean  = tmp / ncols_hc;
    const float scale = rsqrtf(mean + eps_hc);
    // MMVF reads F32 pairs and rounds each to half (make_half2 of two floats).
    const float2 * R2 = (const float2 *) R;
    for (int c2 = tid; c2 < ncols_hc / 2; c2 += block_size) {
        const float2 x = R2[c2];
        // rms_norm_f32 stores scale * x as F32; MMVF then rounds it to half.
        const float y0 = ds4_hc_f32(scale * x.x);
        const float y1 = ds4_hc_f32(scale * x.y);
        s_y2[c2] = make_half2(y0, y1);
    }
    __syncthreads();

    // mix = fn @ n1: block (token, split) computes rows split*n_groups + g,
    // one per MMVF group, with MMVF's per-row arithmetic.
    static_assert(NSPLIT * n_groups >= mix_dim, "the split blocks must cover every mix row");
    const int split_id = (int) blockIdx.y;
    {
        const int g  = tid / MMV_BLOCK;
        const int lt = tid % MMV_BLOCK;
        const int ncols2 = ncols_hc / 2;
        const int r = split_id * n_groups + g;
        if (r < mix_dim) {
            const half2 * x2 = (const half2 *) (fn + (int64_t) r * fn_stride);
            half2 sumh2 = make_half2(0.0f, 0.0f);
#pragma unroll 8
            for (int col2 = lt; col2 < ncols2; col2 += MMV_BLOCK) {
                const half2 tmpx = x2[col2];
                sumh2 += tmpx * s_y2[col2];
            }
            float sumf = __low2float(sumh2) + __high2float(sumh2);
            sumf = warp_reduce_sum<WARP_SIZE>(sumf);
            if (lane == 0) {
                s_part[g][lt / WARP_SIZE] = sumf;
            }
        }
    }
    __syncthreads();
    if (tid / WARP_SIZE < n_groups) {
        // MMVF's cross-warp stage for this block's rows.
        const int g = tid / WARP_SIZE;
        const int r = split_id * n_groups + g;
        if (r < mix_dim) {
            float v = lane < warps_per_group ? s_part[g][lane] : 0.0f;
            v = warp_reduce_sum<WARP_SIZE>(v);
            if (lane == 0) {
                mix_scratch[(int64_t) token * 32 + r] = v;
                __threadfence();   // the row is visible before this block counts itself done
            }
        }
    }
    // The token's last block to finish its rows runs the rest of the boundary.
    // Every launch adds NSPLIT to each of its tokens' counts.
    __shared__ int s_last;
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const unsigned int done = atomicAdd(&rows_done[token], 1u) + 1u;
        s_last = (done % NSPLIT) == 0;
        __threadfence();
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    if (tid < mix_dim) {
        s_mix[tid] = ((volatile const float *) mix_scratch)[(int64_t) token * 32 + tid];
        s_base[tid] = base[tid];
    }
    __syncthreads();

    // pre = ds4_hc_pre(mix, base, R): ds4_hc_pre_kernel_t<NHC>.
    if (tid == 0) {
        ds4_hc_sinkhorn_split_t<NHC>(s_mix, s_base, pre_scale, post_scale, comb_scale, iters, split);
        if (write_pre) {
#pragma unroll
            for (int i = 0; i < mix_dim; ++i) {
                pre_dst[n_embd + i] = split[i];
            }
        }
    }
    __syncthreads();
    if (write_pre) {
        for (int d = tid; d < n_embd; d += block_size) {
            float acc = 0.0f;
#pragma unroll
            for (int h = 0; h < NHC; ++h) {
                acc += split[h] * R[(int64_t) h * n_embd + d];
            }
            pre_dst[d] = acc;
        }
    }

    // R' = ds4_hc_post(R, block_out, split); in = collapse(R', pre). The
    // column mapping is rms_norm_f32<1024>'s over n_embd, and in stays in
    // registers: out may share memory with R or block_out (the unfused
    // allocation frees both before out), so it is written only after every
    // block has finished reading them.
    const float * post = split + NHC;
    const float * comb = split + 2 * NHC;
    float xin[DS4_HC_BOUNDARY_MAX_COLS];
    float tmp_out = 0.0f;
#pragma unroll
    for (int k = 0; k < DS4_HC_BOUNDARY_MAX_COLS; ++k) {
        const int d = tid + k * block_size;
        xin[k] = 0.0f;
        if (d < n_embd) {
            float rn[NHC];
#pragma unroll
            for (int h = 0; h < NHC; ++h) {
                // ds4_hc_post_kernel compiles to one multiply, then an FMA
                // per source copy accumulating into it: spelled out here so
                // the unrolled loop cannot contract the other way.
                float acc = __fmul_rn(block_out[d], post[h]);
                for (int src = 0; src < NHC; ++src) {
                    acc = __fmaf_rn(comb[h + src * NHC], R[(int64_t) src * n_embd + d], acc);
                }
                rn[h] = acc;
                R_next[(int64_t) h * n_embd + d] = acc;
            }
            const float xi = ds4_hc_collapse_value<NHC>(rn, split);
            xin[k] = xi;
            tmp_out += xi * xi;
        }
    }
    // out = rms_norm(in) * w: rms_norm_f32<1024, mul>.
    tmp_out = block_reduce<block_reduce_method::SUM, block_size>(tmp_out, s_sum_out);
    const float mean_out  = tmp_out / n_embd;
    const float scale_out = rsqrtf(mean_out + eps_out);

    ds4_hc_grid_barrier(barrier, gridDim.x);
#pragma unroll
    for (int k = 0; k < DS4_HC_BOUNDARY_MAX_COLS; ++k) {
        const int d = tid + k * block_size;
        if (d < n_embd) {
            const float xo = scale_out * xin[k] * w[d];
            out[d] = xo;
            if (q8) {
                // quantize_q8_1 of out: a warp holds one 32-wide block (d is
                // warp-contiguous), reduced in the same butterfly order.
                float amax = fabsf(xo);
                float sum = xo;
                amax = warp_reduce_max<QK8_1>(amax);
                sum  = warp_reduce_sum<QK8_1>(sum);
                const float  dq = amax / 127.0f;
                const int8_t q  = amax == 0.0f ? 0 : roundf(xo / dq);
                block_q8_1 * yb = q8 + (int64_t) token * q8_row_blocks + d / QK8_1;
                const int iqs = d % QK8_1;
                yb->qs[iqs] = q;
                if (iqs == 0) {
                    yb->ds = make_half2(dq, sum);
                }
            }
        }
    }
}

bool ggml_cuda_ds4_hc_boundary_supported(int n_hc, int n_embd, int mmv_block) {
    // n_embd % 1024: every warp's columns are one whole q8_1 block of out.
    return n_hc == 4 && n_embd >= 1024 && n_embd % 1024 == 0 &&
           n_embd <= DS4_HC_BOUNDARY_MAX_COLS * 1024 &&
           (n_embd * n_hc) % 2 == 0 &&
           (size_t) n_embd * n_hc / 2 * sizeof(half2) <= 48u * 1024u &&
           (mmv_block == 64 || mmv_block == 128 || mmv_block == 256);
}

void ggml_cuda_ds4_hc_boundary(
        ggml_backend_cuda_context & ctx, int n_tokens, int n_hc, int mmv_block,
        const float * R, int64_t R_stride, const half * fn, int64_t fn_stride, const float * base,
        const float * block_out, int64_t bo_stride, const float * w,
        float * pre_dst, int64_t pre_stride, float * R_next, int64_t Rn_stride,
        float * out, int64_t out_stride,
        int n_embd, float eps_hc, float eps_out, int iters,
        float pre_scale, float post_scale, float comb_scale, bool write_pre,
        void * q8_out) {
    GGML_ASSERT(ggml_cuda_ds4_hc_boundary_supported(n_hc, n_embd, mmv_block));
    GGML_ASSERT(n_tokens >= 1 && n_tokens <= 64);
    // The output barrier's count and generation (ds4_hc_grid_barrier).
    if (!ctx.ds4_hc_barrier) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMalloc((void **) &ctx.ds4_hc_barrier, 2 * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemsetAsync(ctx.ds4_hc_barrier, 0, 2 * sizeof(unsigned int), ctx.stream()));
    }
    const size_t shmem = (size_t) n_embd * n_hc / 2 * sizeof(half2);
    cudaStream_t stream = ctx.stream();
    // LUCE_DS4_HC_BOUNDARY_SPLIT=1: the mix over several blocks per token
    // (one token's block reads the whole fn matrix alone otherwise). Only
    // while every block fits at once beside the others, since the finishing
    // blocks wait for each other at the output barrier.
    static const bool split_mix = [] {
        const char * v = getenv("LUCE_DS4_HC_BOUNDARY_SPLIT");
        return v && *v && strcmp(v, "0") != 0;
    }();
    if (split_mix && n_tokens * 6 <= 48) {
        if (!ctx.ds4_hc_split_scratch) {
            ggml_cuda_set_device(ctx.device);
            CUDA_CHECK(cudaMalloc(&ctx.ds4_hc_split_scratch, 64 * 32 * sizeof(float) + 64 * sizeof(unsigned int)));
            CUDA_CHECK(cudaMemsetAsync(ctx.ds4_hc_split_scratch, 0, 64 * 32 * sizeof(float) + 64 * sizeof(unsigned int), stream));
        }
        float * mix_scratch = (float *) ctx.ds4_hc_split_scratch;
        unsigned int * rows_done = (unsigned int *) (mix_scratch + 64 * 32);
#define DS4_HC_BOUNDARY_SPLIT_LAUNCH(B, NS) \
        ds4_hc_boundary_split_kernel<4, B, NS><<<dim3(n_tokens, NS), 1024, shmem, stream>>>( \
            R, R_stride, fn, fn_stride, base, block_out, bo_stride, w, \
            pre_dst, pre_stride, R_next, Rn_stride, out, out_stride, \
            n_embd, eps_hc, eps_out, iters, pre_scale, post_scale, comb_scale, write_pre ? 1 : 0, \
            ctx.ds4_hc_barrier, \
            (block_q8_1 *) q8_out, (int64_t) GGML_PAD(n_embd, MATRIX_ROW_PADDING) / QK8_1, \
            mix_scratch, rows_done)
        switch (mmv_block) {
            case  64: DS4_HC_BOUNDARY_SPLIT_LAUNCH(64, 2);  break;
            case 128: DS4_HC_BOUNDARY_SPLIT_LAUNCH(128, 3); break;
            default:  DS4_HC_BOUNDARY_SPLIT_LAUNCH(256, 6); break;
        }
#undef DS4_HC_BOUNDARY_SPLIT_LAUNCH
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#define DS4_HC_BOUNDARY_LAUNCH(B) \
    ds4_hc_boundary_kernel<4, B><<<n_tokens, 1024, shmem, stream>>>( \
        R, R_stride, fn, fn_stride, base, block_out, bo_stride, w, \
        pre_dst, pre_stride, R_next, Rn_stride, out, out_stride, \
        n_embd, eps_hc, eps_out, iters, pre_scale, post_scale, comb_scale, write_pre ? 1 : 0, \
        ctx.ds4_hc_barrier, \
        (block_q8_1 *) q8_out, (int64_t) GGML_PAD(n_embd, MATRIX_ROW_PADDING) / QK8_1)
    switch (mmv_block) {
        case  64: DS4_HC_BOUNDARY_LAUNCH(64);  break;
        case 128: DS4_HC_BOUNDARY_LAUNCH(128); break;
        default:  DS4_HC_BOUNDARY_LAUNCH(256); break;
    }
#undef DS4_HC_BOUNDARY_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

// GGML_OP_DS4_HC modes (op_params[0]): 0 hc_pre, 1 hc_post, 2 hc_out,
// 3 hc_post_split, 4 router_select, 5 router_weights, 6 hc_collapse.
// Mode 7: the column argmax of a vocabulary slice (argmax_f32's scan: first
// occurrence per thread, then the butterfly) as one rank's exchange slot.
static __global__ void ds4_argmax_pair_kernel(const float * __restrict__ x, float * __restrict__ dst,
                                              const int64_t nrows, const int row_offset,
                                              const int slot, const int n_slots) {
    const int64_t t = blockIdx.x;
    const float * col = x + t * nrows;
    float maxval = -FLT_MAX;
    int argmax = -1;
    for (int64_t r = threadIdx.x; r < nrows; r += blockDim.x) {
        const float v = col[r];
        if (v > maxval) { maxval = v; argmax = (int) r; }
    }
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float v = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   c = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (v > maxval || (v == maxval && c >= 0 && (argmax < 0 || c < argmax))) { maxval = v; argmax = c; }
    }
    __shared__ float s_val[32];
    __shared__ int s_arg[32];
    const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
    if (lane == 0) { s_val[warp] = maxval; s_arg[warp] = argmax; }
    __syncthreads();
    float * out = dst + t * (int64_t) (8 * n_slots);
    if (threadIdx.x < 8 * n_slots) out[threadIdx.x] = 0.0f;
    __syncthreads();
    if (threadIdx.x == 0) {
        const int n_warps = blockDim.x / WARP_SIZE;
        float bv = s_val[0]; int bi = s_arg[0];
        for (int w = 1; w < n_warps; ++w) {
            if (s_val[w] > bv || (s_val[w] == bv && s_arg[w] >= 0 && (bi < 0 || s_arg[w] < bi))) { bv = s_val[w]; bi = s_arg[w]; }
        }
        const unsigned int vb = __float_as_uint(bv);
        const unsigned int ib = (unsigned int) (bi < 0 ? 0 : bi + row_offset);
        float * o = out + 8 * slot;
        for (int b = 0; b < 4; ++b) {
            o[b]     = (float) ((vb >> (8 * b)) & 0xFFu);
            o[4 + b] = (float) ((ib >> (8 * b)) & 0xFFu);
        }
    }
}

// Mode 8: the winner of the summed slots, ties to the lower index.
static __global__ void ds4_argmax_pick_kernel(const float * __restrict__ pairs, int32_t * __restrict__ dst,
                                              const int n_tokens, const int n_slots) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tokens) return;
    const float * p = pairs + (int64_t) t * (8 * n_slots);
    float bv = 0.0f; int bi = -1;
    for (int s = 0; s < n_slots; ++s) {
        unsigned int vb = 0, ib = 0;
        for (int b = 0; b < 4; ++b) {
            vb |= ((unsigned int) p[8 * s + b]) << (8 * b);
            ib |= ((unsigned int) p[8 * s + 4 + b]) << (8 * b);
        }
        const float v = __uint_as_float(vb);
        if (bi < 0 || v > bv || (v == bv && (int) ib < bi)) { bv = v; bi = (int) ib; }
    }
    dst[t] = bi;
}

void ggml_cuda_op_ds4_hc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_get_op_params_i32(dst, 0) == 7) {
        const ggml_tensor * logits = dst->src[0];
        ds4_argmax_pair_kernel<<<(unsigned) logits->ne[1], 1024, 0, ctx.stream()>>>(
            (const float *) logits->data, (float *) dst->data, logits->ne[0],
            ggml_get_op_params_i32(dst, 1), ggml_get_op_params_i32(dst, 2), ggml_get_op_params_i32(dst, 3));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (ggml_get_op_params_i32(dst, 0) == 8) {
        const ggml_tensor * pairs = dst->src[0];
        const int n_tokens = (int) pairs->ne[1];
        ds4_argmax_pick_kernel<<<(n_tokens + 63) / 64, 64, 0, ctx.stream()>>>(
            (const float *) pairs->data, (int32_t *) dst->data, n_tokens, ggml_get_op_params_i32(dst, 3));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
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
