// Per-token RMSNorm + weight multiply kernel — compiled for all HIP builds
// (not gated on DFLASH27B_HIP_SM80_EQUIV) so that the HIP chunk-B graph
// path in qwen3_graph.cpp can call it even in the baseline (q8-fallback) build.

#include <hip/hip_runtime.h>

__global__ void rms_norm_mul_w_f32_kernel(
    const float * __restrict__ src,
    const float * __restrict__ w,
    float       * __restrict__ dst,
    int hidden, float eps)
{
    const int tok = blockIdx.x;
    const float * row = src + (size_t)tok * hidden;
    float       * out = dst + (size_t)tok * hidden;

    extern __shared__ float smem[];

    float sumsq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = row[i];
        sumsq += v * v;
    }

    for (int off = 16; off > 0; off >>= 1)
        sumsq += __shfl_xor(sumsq, off);

    if ((threadIdx.x & 31) == 0)
        smem[threadIdx.x >> 5] = sumsq;
    __syncthreads();

    const int n_warps = blockDim.x >> 5;
    if (threadIdx.x < 32) {
        sumsq = (threadIdx.x < n_warps) ? smem[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            sumsq += __shfl_xor(sumsq, off);
        if (threadIdx.x == 0)
            smem[0] = sumsq;
    }
    __syncthreads();

    const float inv = rsqrtf(smem[0] / (float)hidden + eps);

    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        out[i] = row[i] * inv * w[i];
}

extern "C" void launch_rms_norm_mul_w_f32(
    const float * src, const float * w, float * dst,
    int n_tokens, int hidden, float eps,
    hipStream_t stream)
{
    const int block = 256;
    const size_t smem = (size_t)(block >> 5) * sizeof(float);
    rms_norm_mul_w_f32_kernel<<<n_tokens, block, smem, stream>>>(
        src, w, dst, hidden, eps);
}
