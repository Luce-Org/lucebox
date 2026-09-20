#include "hc-mix.cuh"
#include "mmb.cuh"
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

// mixed[e,t] = scale * ( ((xn*sig)[0] + (xn*sig)[1]) + ... ) + bias, streams summed in order 0..hc-1
static __global__ void hc_mix_reduce_f32(
        const float * __restrict__ xn, const float * __restrict__ gate, float * __restrict__ dst,
        const int64_t n_embd, const int64_t n_tokens, const int hc, const float scale, const float bias) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n_embd * n_tokens) {
        return;
    }

    const int64_t t = index / n_embd;
    const int64_t e = index - t * n_embd;
    const int64_t hc_dim = n_embd * hc;
    const int64_t base = t * hc_dim + e;

    float acc = hc_mul_rn(xn[base], hc_sigmoid(gate[base]));
    for (int c = 1; c < hc; ++c) {
        const int64_t i = base + int64_t(c) * n_embd;
        acc = hc_add_rn(acc, hc_mul_rn(xn[i], hc_sigmoid(gate[i])));
    }

    dst[index] = scale * acc + bias;
}

static __global__ void hc_mix_reduce_f32_hc4_parallel(
        const float * __restrict__ xn, const float * __restrict__ gate, float * __restrict__ dst,
        const int64_t n_embd, const int64_t n_tokens, const float scale, const float bias) {
    const int lane = threadIdx.x % 32;
    const int stream = threadIdx.x / 32;
    const int64_t index = int64_t(blockIdx.x) * 32 + lane;
    const int64_t count = n_embd * n_tokens;
    __shared__ float products[4][32];
    float value = 0.0f;
    if (index < count) {
        const int64_t token = index / n_embd;
        const int64_t embd = index - token * n_embd;
        const int64_t offset = token * (4 * n_embd) + stream * n_embd + embd;
        value = hc_mul_rn(xn[offset], hc_sigmoid(gate[offset]));
    }
    products[stream][lane] = value;
    __syncthreads();
    if (stream == 0 && index < count) {
        float sum = products[0][lane];
        sum = hc_add_rn(sum, products[1][lane]);
        sum = hc_add_rn(sum, products[2][lane]);
        sum = hc_add_rn(sum, products[3][lane]);
        dst[index] = scale * sum + bias;
    }
}

__device__ __forceinline__ float hc_bf2f(const uint16_t h) { return __uint_as_float(((uint32_t) h) << 16); }
// same reduction as hc_mix_reduce_f32, reading BF16 copies of xn and gate
static __global__ void hc_mix_reduce_bf16(
        const uint16_t * __restrict__ xn, const uint16_t * __restrict__ gate, float * __restrict__ dst,
        const int64_t n_embd, const int64_t n_tokens, const int hc, const float scale, const float bias) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n_embd * n_tokens) return;
    const int64_t t = index / n_embd, e = index - t * n_embd, base = t * (int64_t) n_embd * hc + e;
    float acc = hc_mul_rn(hc_bf2f(xn[base]), hc_sigmoid(hc_bf2f(gate[base])));
    for (int c = 1; c < hc; ++c) { const int64_t i = base + int64_t(c) * n_embd; acc = hc_add_rn(acc, hc_mul_rn(hc_bf2f(xn[i]), hc_sigmoid(hc_bf2f(gate[i])))); }
    dst[index] = scale * acc + bias;
}

static bool hc_ranges_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t av = (uintptr_t) a->data;
    const uintptr_t bv = (uintptr_t) b->data;
    return av <= bv ? bv-av < ggml_nbytes(a) : av-bv < ggml_nbytes(b);
}

void ggml_cuda_op_hc_mix_reduce(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_mix_args & args) {
    const ggml_tensor * xn   = args.xn;
    const ggml_tensor * gate = args.gate;
    ggml_tensor *       dst  = args.dst;

    GGML_ASSERT(xn->type == GGML_TYPE_F32 && gate->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(xn) && ggml_is_contiguous(gate) && ggml_is_contiguous(dst));
    GGML_ASSERT(args.hc > 0);

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_tokens = ggml_nrows(dst);
    GGML_ASSERT(xn->ne[0] == n_embd * args.hc && ggml_nrows(xn) == n_tokens);
    GGML_ASSERT(ggml_are_same_shape(xn, gate));

    const bool stage = hc_ranges_overlap(dst, xn) || hc_ranges_overlap(dst, gate);

    ggml_cuda_pool_alloc<float> staged(ctx.pool());
    float * out = stage ? staged.alloc(n_embd * n_tokens) : (float *) dst->data;

    const int64_t n_items = n_embd * n_tokens;
    static const bool hc16 = true;
    const uint16_t * xn16 = hc16 ? ggml_cuda_mmb_cache_lookup(xn) : nullptr;
    const uint16_t * g16  = hc16 ? ggml_cuda_mmb_cache_lookup(gate) : nullptr;
    if (xn16 && g16) {
        const int threads = 256; const int blocks = (int) ((n_items + threads - 1) / threads);
        hc_mix_reduce_bf16<<<blocks, threads, 0, ctx.stream()>>>(xn16, g16, out, n_embd, n_tokens, args.hc, args.scale, args.bias);
    } else if (args.hc == 4 && n_tokens <= 32) {
        const int blocks = (int) ((n_items + 31) / 32);
        hc_mix_reduce_f32_hc4_parallel<<<blocks, 128, 0, ctx.stream()>>>(
            (const float *) xn->data, (const float *) gate->data, out, n_embd, n_tokens, args.scale, args.bias);
    } else {
        const int threads = 256;
        const int blocks = (int) ((n_items + threads - 1) / threads);
        hc_mix_reduce_f32<<<blocks, threads, 0, ctx.stream()>>>(
            (const float *) xn->data, (const float *) gate->data, out, n_embd, n_tokens, args.hc, args.scale, args.bias);
    }
    CUDA_CHECK(cudaGetLastError());

    if (stage) {
        CUDA_CHECK(cudaMemcpyAsync(dst->data, out, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, ctx.stream()));
    }
}
