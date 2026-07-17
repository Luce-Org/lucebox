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
    // HIP exposes the target-specific width through a device builtin. Unlike
    // preprocessor spellings of the AMDGCN builtin, this works across ROCm
    // releases and remains correct in multi-architecture builds.
    const int wave = warpSize;

    const int tok = blockIdx.x;
    const float * row = src + (size_t)tok * hidden;
    float       * out = dst + (size_t)tok * hidden;

    extern __shared__ float smem[];

    float sumsq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = row[i];
        sumsq += v * v;
    }

    #pragma unroll
    for (int off = wave / 2; off > 0; off >>= 1)
        sumsq += __shfl_xor(sumsq, off);

    if ((threadIdx.x & (wave - 1)) == 0)
        smem[threadIdx.x / wave] = sumsq;
    __syncthreads();

    // Final reduce across per-wavefront partials in a single wavefront. Valid
    // while n_warps <= wave, which holds for the fixed block=256 launch below
    // (8 warps on wave32, 4 on wave64).
    const int n_warps = blockDim.x / wave;
    if (threadIdx.x < wave) {
        sumsq = (threadIdx.x < n_warps) ? smem[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int off = wave / 2; off > 0; off >>= 1)
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
    // One float per wavefront partial. Host code can't see __AMDGCN_WAVEFRONT_SIZE,
    // so size for the wave32 (max-warp) case: block/32 floats is a safe upper
    // bound that also covers wave64 (block/64 partials).
    const size_t smem = (size_t)(block >> 5) * sizeof(float);
    rms_norm_mul_w_f32_kernel<<<n_tokens, block, smem, stream>>>(
        src, w, dst, hidden, eps);
}
