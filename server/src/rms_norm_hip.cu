// Per-token RMSNorm + weight multiply kernel — compiled for all HIP builds
// (not gated on DFLASH27B_HIP_SM80_EQUIV) so that the HIP chunk-B graph
// path in qwen3_graph.cpp can call it even in the baseline (q8-fallback) build.

#include <hip/hip_runtime.h>

// Wavefront width of the target arch: 32 on RDNA (gfx10/gfx11), 64 on GCN/CDNA.
// The reduction below must span exactly one wavefront, so hardcoding 32 leaves
// lanes 32..63 out of the reduce on wave64 archs. __AMDGCN_WAVEFRONT_SIZE(__) is
// a compile-time builtin defined only in the device pass. The 64 fallback is
// host-only (WAVE is never used off-device); a device pass without the builtin
// is a hard error rather than a silent wrong-width guess.
#ifndef DFLASH_WAVE_SIZE
#  if defined(__AMDGCN_WAVEFRONT_SIZE__)
#    define DFLASH_WAVE_SIZE __AMDGCN_WAVEFRONT_SIZE__
#  elif defined(__AMDGCN_WAVEFRONT_SIZE)
#    define DFLASH_WAVE_SIZE __AMDGCN_WAVEFRONT_SIZE
#  elif !defined(__HIP_DEVICE_COMPILE__)
#    define DFLASH_WAVE_SIZE 64  // host pass only; WAVE is unused off-device
#  else
#    error "rms_norm_hip: missing wavefront-size builtin in device pass; set DFLASH_WAVE_SIZE"
#  endif
#endif

__global__ void rms_norm_mul_w_f32_kernel(
    const float * __restrict__ src,
    const float * __restrict__ w,
    float       * __restrict__ dst,
    int hidden, float eps)
{
    constexpr int WAVE = DFLASH_WAVE_SIZE;

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
    for (int off = WAVE / 2; off > 0; off >>= 1)
        sumsq += __shfl_xor(sumsq, off);

    if ((threadIdx.x & (WAVE - 1)) == 0)
        smem[threadIdx.x / WAVE] = sumsq;
    __syncthreads();

    // Final reduce across per-wavefront partials in a single wavefront. Valid
    // while n_warps <= WAVE, which holds for the fixed block=256 launch below
    // (8 warps on wave32, 4 on wave64).
    const int n_warps = blockDim.x / WAVE;
    if (threadIdx.x < WAVE) {
        sumsq = (threadIdx.x < n_warps) ? smem[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int off = WAVE / 2; off > 0; off >>= 1)
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
