#include "hc-cn.cuh"
#include <cstdlib>

#if defined(__HIP_PLATFORM_AMD__)
static __device__ __forceinline__ float hc_mul_rn(const float a, const float b) {
    float result;
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}

static __device__ __forceinline__ float hc_add_rn(const float a, const float b) {
    float result;
    asm("v_add_f32_e32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}
#else
static __device__ __forceinline__ float hc_mul_rn(const float a, const float b) {
    return __fmul_rn(a, b);
}

static __device__ __forceinline__ float hc_add_rn(const float a, const float b) {
    return __fadd_rn(a, b);
}
#endif

// same expression as op_sigmoid in unary.cu
static __device__ __forceinline__ float hc_sigmoid(const float x) {
    return 1.0f / (1.0f + expf(-x));
}

#define HC_CN_MAX_EMB 3072

__device__ __forceinline__ float hc_bf2f32(const uint16_t h) { return __uint_as_float(((uint32_t) h) << 16); }
__device__ __forceinline__ uint16_t hc_f2bf32(const float f) { uint32_t u = __float_as_uint(f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }

#define HC_CN_BLOCK2 256

// Q8 activation tiles for the W8A8 HC down projection (mmb-w8a8.cuh layout: 16 tokens x 32 K per 576 bytes, codes
// [half][token][16], F32 scales at +512). The 16 threads holding a 32-wide block share its absmax; each writes its two
// codes. kk = stream * n_embd + col is the K index of the pair, nkb = hc * n_embd / 32.
static __device__ __forceinline__ void hc_q8_pair(int8_t * q8, const int nkb, const int t, const int kk, const float v0, const float v1) {
    float m = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off, 32));
    const float d = m / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    const int q0 = (int) roundf(v0 * id), q1 = (int) roundf(v1 * id);
    int8_t * tile = q8 + ((size_t) (t / 16) * nkb + kk / 32) * 576;
    const int pos = kk % 32, tl = t % 16;
    *(uint16_t *) (tile + (pos / 16) * 256 + tl * 16 + (pos % 16)) = (uint16_t) ((q0 & 0xff) | ((q1 & 0xff) << 8));
    if ((threadIdx.x & 15) == 0) *(float *) (tile + 512 + tl * 4) = d;
}
__device__ __forceinline__ float hc_lo(const uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hc_hi(const uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ uint32_t hc_pack2(const float a, const float b) {
    return (uint32_t) hc_f2bf32(a) | ((uint32_t) hc_f2bf32(b) << 16);
}

// two elements per thread per iteration, packed 32-bit accesses
static __global__ void __launch_bounds__(HC_CN_BLOCK2, 4) hc_combine_norm_f32_b256(
        const float * inject, const float * residual,
        const float * block_out, const float * gamma,
        float * out_res, float * out_xn, uint16_t * out_xn_bf16, const bool store_xn_f32,
        const uint16_t * res_in_bf16, uint16_t * res_out_bf16, const uint16_t * blk_in_bf16,
        const int n_embd, const float s1, const float b1, const float s2, const float b2, const float eps, int8_t * q8 = nullptr) {
    __shared__ float s_sum[32];
    const int c = blockIdx.x, t = blockIdx.y, hc = gridDim.x, tid = threadIdx.x;
    const float x1 = s1 * inject[(int64_t) t * hc + c] + b1;
    const float x2 = hc_sigmoid(x1);
    const float w  = s2 * x2 + b2;
    const int64_t row = (int64_t) t * hc + c;
    const float *    res   = residual + row * n_embd;
    const uint16_t * res16 = res_in_bf16  ? res_in_bf16  + row * n_embd : nullptr;
    float *          dst   = out_res      + row * n_embd;
    uint16_t *       dst16 = res_out_bf16 ? res_out_bf16 + row * n_embd : nullptr;
    const float *    blk   = block_out + (int64_t) t * n_embd;
    const uint16_t * blk16 = blk_in_bf16 ? blk_in_bf16 + (int64_t) t * n_embd : nullptr;
    constexpr int KP = (HC_CN_MAX_EMB / 2 + HC_CN_BLOCK2 - 1) / HC_CN_BLOCK2;
    float xs[2 * KP];
    float tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < KP; ++k) {
        const int col = (tid + k * HC_CN_BLOCK2) * 2;
        xs[2 * k] = 0.0f; xs[2 * k + 1] = 0.0f;
        if (col + 1 < n_embd) {
            float r0, r1, m0, m1;
            if (res16) { const uint32_t u = *(const uint32_t *)(res16 + col); r0 = hc_lo(u); r1 = hc_hi(u); }
            else       { const float2   v = *(const float2 *)(res + col);     r0 = v.x;      r1 = v.y; }
            if (blk16) { const uint32_t u = *(const uint32_t *)(blk16 + col); m0 = hc_lo(u); m1 = hc_hi(u); }
            else       { const float2   v = *(const float2 *)(blk + col);     m0 = v.x;      m1 = v.y; }
            const float a0 = hc_add_rn(r0, hc_mul_rn(m0, w));
            const float a1 = hc_add_rn(r1, hc_mul_rn(m1, w));
            if (dst16) *(uint32_t *)(dst16 + col) = hc_pack2(a0, a1);
            else       *(float2 *)(dst + col)     = make_float2(a0, a1);
            xs[2 * k] = a0; xs[2 * k + 1] = a1;
            tmp += a0 * a0 + a1 * a1;
        } else if (col < n_embd) {
            const float r0 = res16 ? hc_bf2f32(res16[col]) : res[col];
            const float m0 = blk16 ? hc_bf2f32(blk16[col]) : blk[col];
            const float a0 = hc_add_rn(r0, hc_mul_rn(m0, w));
            if (dst16) dst16[col] = hc_f2bf32(a0); else dst[col] = a0;
            xs[2 * k] = a0;
            tmp += a0 * a0;
        }
    }
    tmp = block_reduce<block_reduce_method::SUM, HC_CN_BLOCK2>(tmp, s_sum);
    const float mean  = tmp / n_embd;
    const float scale = rsqrtf(mean + eps);
    const float * g  = gamma  + (int64_t) c * n_embd;
    float *       xn = out_xn + row * n_embd;
    uint16_t *    xh = out_xn_bf16 ? out_xn_bf16 + row * n_embd : nullptr;
#pragma unroll
    for (int k = 0; k < KP; ++k) {
        const int col = (tid + k * HC_CN_BLOCK2) * 2;
        if (col + 1 < n_embd) {
            const float2 gv = *(const float2 *)(g + col);
            const float v0 = scale * xs[2 * k] * gv.x, v1 = scale * xs[2 * k + 1] * gv.y;
            if (store_xn_f32) *(float2 *)(xn + col) = make_float2(v0, v1);
            if (xh) *(uint32_t *)(xh + col) = hc_pack2(v0, v1);
            if (q8) hc_q8_pair(q8, gridDim.x * n_embd / 32, t, c * n_embd + col, v0, v1);
        } else if (col < n_embd) {
            const float v0 = scale * xs[2 * k] * g[col];
            if (store_xn_f32) xn[col] = v0;
            if (xh) xh[col] = hc_f2bf32(v0);
        }
    }
}

// MoE mode: one block per token builds the MoE block output for its columns once, then runs the per-stream body of
// hc_combine_norm_f32_b256 (same thread mapping, expressions and reduction) for every stream. Bit-exact against
// DS4_MOE_COMBINE + HC_COMBINE_NORM.
template <bool DOWN_F16>
static __global__ void __launch_bounds__(HC_CN_BLOCK2, 4) hc_combine_norm_moe_f32_b256(
        const float * inject, const float * residual, const float * gamma,
        const void * down, const float * moe_w, const float * shared, const float * sh_logit,
        float * out_res, float * out_xn, uint16_t * out_xn_bf16, const bool store_xn_f32,
        const int n_embd, const int hc, const int n_used, const int64_t down_s1, const int64_t down_s2,
        const int64_t w_s1, const int64_t sh_s1,
        const float s1, const float b1, const float s2, const float b2, const float eps, int8_t * q8) {
    __shared__ float s_sum[32];
    const int t = blockIdx.x, tid = threadIdx.x;
    constexpr int KP = (HC_CN_MAX_EMB / 2 + HC_CN_BLOCK2 - 1) / HC_CN_BLOCK2;
    float ms[2 * KP];
    const float sig = hc_sigmoid(sh_logit[t]);
    const float * wr = moe_w + (int64_t) t * w_s1;
    // Routes outermost: every route issues all of this thread's column loads at once (the per-element add order is
    // still route order, so the result is unchanged).
#pragma unroll
    for (int i = 0; i < 2 * KP; ++i) ms[i] = 0.0f;
    for (int e = 0; e < n_used; ++e) {
        const float w = wr[e];
        if (w == 0.0f) continue;
        const int64_t rb = (int64_t) t * down_s2 + (int64_t) e * down_s1;
#pragma unroll
        for (int k = 0; k < KP; ++k) {
            const int col = (tid + k * HC_CN_BLOCK2) * 2;
            if (col + 1 < n_embd) {
                float v0, v1;
                if constexpr (DOWN_F16) {
                    const __half2 h = *(const __half2 *) ((const half *) down + rb + col);
                    v0 = __low2float(h); v1 = __high2float(h);
                } else {
                    const float2 f = *(const float2 *) ((const float *) down + rb + col);
                    v0 = f.x; v1 = f.y;
                }
                ms[2 * k]     = __fadd_rn(ms[2 * k],     __fmul_rn(v0, w));
                ms[2 * k + 1] = __fadd_rn(ms[2 * k + 1], __fmul_rn(v1, w));
            } else if (col < n_embd) {
                const float v0 = DOWN_F16 ? __half2float(((const half *) down)[rb + col]) : ((const float *) down)[rb + col];
                ms[2 * k] = __fadd_rn(ms[2 * k], __fmul_rn(v0, w));
            }
        }
    }
#pragma unroll
    for (int k = 0; k < KP; ++k) {
        const int col = (tid + k * HC_CN_BLOCK2) * 2;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            if (col + j < n_embd) {
                const float sh = shared[(int64_t) t * sh_s1 + col + j] * sig;
                ms[2 * k + j] = __fadd_rn(sh, ms[2 * k + j]);
            }
        }
    }
    for (int c = 0; c < hc; ++c) {
        const float x1 = s1 * inject[(int64_t) t * hc + c] + b1;
        const float x2 = hc_sigmoid(x1);
        const float w  = s2 * x2 + b2;
        const int64_t row = (int64_t) t * hc + c;
        const float * res = residual + row * n_embd;
        float *       dst = out_res  + row * n_embd;
        float xs[2 * KP];
        float tmp = 0.0f;
#pragma unroll
        for (int k = 0; k < KP; ++k) {
            const int col = (tid + k * HC_CN_BLOCK2) * 2;
            xs[2 * k] = 0.0f; xs[2 * k + 1] = 0.0f;
            if (col + 1 < n_embd) {
                const float2 v = *(const float2 *)(res + col);
                const float r0 = v.x, r1 = v.y, m0 = ms[2 * k], m1 = ms[2 * k + 1];
                const float a0 = hc_add_rn(r0, hc_mul_rn(m0, w));
                const float a1 = hc_add_rn(r1, hc_mul_rn(m1, w));
                *(float2 *)(dst + col) = make_float2(a0, a1);
                xs[2 * k] = a0; xs[2 * k + 1] = a1;
                tmp += a0 * a0 + a1 * a1;
            } else if (col < n_embd) {
                const float r0 = res[col], m0 = ms[2 * k];
                const float a0 = hc_add_rn(r0, hc_mul_rn(m0, w));
                dst[col] = a0;
                xs[2 * k] = a0;
                tmp += a0 * a0;
            }
        }
        tmp = block_reduce<block_reduce_method::SUM, HC_CN_BLOCK2>(tmp, s_sum);
        const float mean  = tmp / n_embd;
        const float scale = rsqrtf(mean + eps);
        const float * g  = gamma  + (int64_t) c * n_embd;
        float *       xn = out_xn + row * n_embd;
        uint16_t *    xh = out_xn_bf16 ? out_xn_bf16 + row * n_embd : nullptr;
#pragma unroll
        for (int k = 0; k < KP; ++k) {
            const int col = (tid + k * HC_CN_BLOCK2) * 2;
            if (col + 1 < n_embd) {
                const float2 gv = *(const float2 *)(g + col);
                const float v0 = scale * xs[2 * k] * gv.x, v1 = scale * xs[2 * k + 1] * gv.y;
                if (store_xn_f32) *(float2 *)(xn + col) = make_float2(v0, v1);
                if (xh) *(uint32_t *)(xh + col) = hc_pack2(v0, v1);
                if (q8) hc_q8_pair(q8, hc * n_embd / 32, t, c * n_embd + col, v0, v1);
            } else if (col < n_embd) {
                const float v0 = scale * xs[2 * k] * g[col];
                if (store_xn_f32) xn[col] = v0;
                if (xh) xh[col] = hc_f2bf32(v0);
            }
        }
        __syncthreads();   // s_sum is reused by the next stream's reduction
    }
}

bool ggml_cuda_hc_combine_norm_supported(const ggml_cuda_hc_combine_norm_args & a, const int warp_size) {
    const int64_t n_embd = a.out_res->ne[0];
    const int64_t hc     = a.out_res->ne[1];
    return warp_size == 32 && n_embd >= 1024 && n_embd <= HC_CN_MAX_EMB && hc >= 1 && hc <= 16 &&
        ggml_nelements(a.gamma) == n_embd * hc;
}

void ggml_cuda_hc_combine_norm_ptrs(ggml_backend_cuda_context & ctx,
        const ggml_tensor * inject, const ggml_tensor * residual, const ggml_tensor * block_out,
        const ggml_tensor * gamma, float * out_res, float * out_xn,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        float s1, float b1, float s2, float b2, float eps) {
    hc_combine_norm_f32_b256<<<dim3((unsigned) hc, (unsigned) n_tokens, 1), HC_CN_BLOCK2, 0, ctx.stream()>>>(
        (const float *) inject->data, (const float *) residual->data,
        (const float *) block_out->data, (const float *) gamma->data,
        out_res, out_xn, nullptr, true, nullptr, nullptr, nullptr,
        (int) n_embd, s1, b1, s2, b2, eps);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_hc_combine_norm(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_combine_norm_args & a) {
    const int64_t n_embd   = a.out_res->ne[0];
    const int64_t hc       = a.out_res->ne[1];
    const int64_t n_tokens = a.out_res->ne[2] * a.out_res->ne[3];

    if (a.moe_down) {
        GGML_ASSERT(n_embd <= HC_CN_MAX_EMB && a.out_res->ne[3] == 1 && !a.res_in_bf16 && !a.res_out_bf16);
        GGML_ASSERT(a.moe_down->ne[0] == n_embd && a.moe_down->ne[2] == n_tokens && a.moe_down->nb[0] == sizeof(float));
        GGML_ASSERT(a.moe_w->ne[0] == a.moe_down->ne[1] && ggml_nrows(a.moe_w) == n_tokens);
        GGML_ASSERT(a.moe_shared->ne[0] == n_embd && ggml_nrows(a.moe_shared) == n_tokens && ggml_nelements(a.moe_sh_logit) == n_tokens);
        // moe_down_f16: the routed down rows hold F16 in place (element strides unchanged, 2 bytes each).
        const size_t esz = a.moe_down_f16 ? sizeof(half) : sizeof(float);
        auto kern = a.moe_down_f16 ? hc_combine_norm_moe_f32_b256<true> : hc_combine_norm_moe_f32_b256<false>;
        kern<<<dim3((unsigned) n_tokens, 1, 1), HC_CN_BLOCK2, 0, ctx.stream()>>>(
            (const float *) a.inject->data, (const float *) a.residual->data, (const float *) a.gamma->data,
            (const void *) a.moe_down->data, (const float *) a.moe_w->data, (const float *) a.moe_shared->data,
            (const float *) a.moe_sh_logit->data,
            (float *) a.out_res->data, (float *) a.out_xn->data, a.out_xn_bf16, a.store_xn_f32,
            (int) n_embd, (int) hc, (int) a.moe_down->ne[1],
            (int64_t) (a.moe_down->nb[1] / sizeof(float)), (int64_t) (a.moe_down->nb[2] / sizeof(float)),   // element strides
            (int64_t) (a.moe_w->nb[1] / sizeof(float)), (int64_t) (a.moe_shared->nb[1] / sizeof(float)),
            a.s1, a.b1, a.s2, a.b2, a.eps, a.out_q8);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    GGML_ASSERT(a.out_res->ne[3] == 1);
    GGML_ASSERT(ggml_nelements(a.out_res) == ggml_nelements(a.out_xn) && ggml_are_same_shape(a.out_res, a.residual));
    GGML_ASSERT(ggml_nelements(a.block_out) == n_embd * n_tokens && ggml_nelements(a.gamma) == n_embd * hc &&
                ggml_nelements(a.inject) == hc * n_tokens);

    hc_combine_norm_f32_b256<<<dim3((unsigned) hc, (unsigned) n_tokens, 1), HC_CN_BLOCK2, 0, ctx.stream()>>>(
        (const float *) a.inject->data, (const float *) a.residual->data,
        (const float *) a.block_out->data, (const float *) a.gamma->data,
        (float *) a.out_res->data, (float *) a.out_xn->data, a.out_xn_bf16, a.store_xn_f32,
        a.res_in_bf16, a.res_out_bf16, a.blk_in_bf16,
        (int) n_embd, a.s1, a.b1, a.s2, a.b2, a.eps, a.out_q8);
    CUDA_CHECK(cudaGetLastError());
}
