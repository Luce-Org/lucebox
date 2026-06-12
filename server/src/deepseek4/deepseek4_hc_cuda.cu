#include "deepseek4_hc_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <mutex>
#include <vector>

namespace dflash::common {
namespace {

constexpr int kMixDim = 24;
constexpr int kThreads = 256;
constexpr int kSums = 64;

struct HcCudaScratch {
    float * d_state = nullptr;
    float * d_sums = nullptr;
    float * d_mix = nullptr;
    size_t state_cap = 0;

    ~HcCudaScratch() {
        if (d_state) cudaFree(d_state);
        if (d_sums) cudaFree(d_sums);
        if (d_mix) cudaFree(d_mix);
    }

    bool ensure(size_t hc_dim) {
        if (!d_sums && cudaMalloc(&d_sums, sizeof(float) * kSums) != cudaSuccess) return false;
        if (!d_mix && cudaMalloc(&d_mix, sizeof(float) * kMixDim) != cudaSuccess) return false;
        if (state_cap < hc_dim) {
            if (d_state) {
                cudaFree(d_state);
                d_state = nullptr;
                state_cap = 0;
            }
            if (cudaMalloc(&d_state, sizeof(float) * hc_dim) != cudaSuccess) return false;
            state_cap = hc_dim;
        }
        return true;
    }
};

std::mutex g_mu;
HcCudaScratch g_scratch;

__global__ void hc_sumsq_kernel(const float * x, int n, float * sums) {
    __shared__ float smem[kThreads];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    float acc = 0.0f;
    for (int i = bid * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) {
        const float v = x[i];
        acc += v * v;
    }
    smem[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) sums[bid] = smem[0];
}

__global__ void hc_mix_kernel(const float * x,
                              const __half * fn,
                              int cols,
                              float inv_rms,
                              float * mix) {
    __shared__ float smem[kThreads];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    float acc = 0.0f;
    const __half * w = fn + (size_t)row * (size_t)cols;
    for (int c = tid; c < cols; c += blockDim.x) {
        acc += __half2float(w[c]) * (x[c] * inv_rms);
    }
    smem[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) mix[row] = smem[0];
}

} // namespace

bool deepseek4_cuda_hc_pre_mix(const float * hc_state_host,
                               const void *  fn_device,
                               int           n_embd,
                               int           n_hc,
                               float         eps,
                               float *       mix_host) {
    if (!hc_state_host || !fn_device || !mix_host || n_embd <= 0 || n_hc <= 0) {
        return false;
    }
    const int hc_dim = n_embd * n_hc;
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_scratch.ensure((size_t)hc_dim)) {
        return false;
    }
    if (cudaMemcpy(g_scratch.d_state, hc_state_host, sizeof(float) * (size_t)hc_dim,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    hc_sumsq_kernel<<<kSums, kThreads>>>(g_scratch.d_state, hc_dim, g_scratch.d_sums);
    if (cudaGetLastError() != cudaSuccess) return false;
    std::vector<float> sums(kSums);
    if (cudaMemcpy(sums.data(), g_scratch.d_sums, sizeof(float) * sums.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    float ss = 0.0f;
    for (float v : sums) ss += v;
    const float inv_rms = rsqrtf(ss / (float)hc_dim + eps);
    hc_mix_kernel<<<kMixDim, kThreads>>>(g_scratch.d_state,
                                         static_cast<const __half *>(fn_device),
                                         hc_dim,
                                         inv_rms,
                                         g_scratch.d_mix);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (cudaMemcpy(mix_host, g_scratch.d_mix, sizeof(float) * kMixDim,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

} // namespace dflash::common
