#include "deepseek4_hc_cuda.h"

#include "common/gpu_runtime_compat.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <mutex>
#include <vector>

namespace dflash::common {
namespace {

constexpr int kMixDim = 24;
constexpr int kThreads = 256;
constexpr int kSums = 64;
constexpr int kMaxHc = 8;
constexpr int kMaxMixDim = 2 * kMaxHc + kMaxHc * kMaxHc;

struct HcCudaScratch {
    float * d_state = nullptr;
    float * d_sums = nullptr;
    float * d_mix = nullptr;
    float * d_scale = nullptr;
    float * d_base = nullptr;
    float * d_working = nullptr;
    float * d_post = nullptr;
    float * d_comb = nullptr;
    size_t state_cap = 0;

    ~HcCudaScratch() {
        if (d_state) cudaFree(d_state);
        if (d_sums) cudaFree(d_sums);
        if (d_mix) cudaFree(d_mix);
        if (d_scale) cudaFree(d_scale);
        if (d_base) cudaFree(d_base);
        if (d_working) cudaFree(d_working);
        if (d_post) cudaFree(d_post);
        if (d_comb) cudaFree(d_comb);
    }

    bool ensure(size_t hc_dim, size_t n_embd, size_t n_hc) {
        if (!d_sums && cudaMalloc(&d_sums, sizeof(float) * kSums) != cudaSuccess) return false;
        if (!d_mix && cudaMalloc(&d_mix, sizeof(float) * kMixDim) != cudaSuccess) return false;
        if (!d_scale && cudaMalloc(&d_scale, sizeof(float) * kMaxMixDim) != cudaSuccess) return false;
        if (!d_base && cudaMalloc(&d_base, sizeof(float) * kMaxMixDim) != cudaSuccess) return false;
        if (!d_working && cudaMalloc(&d_working, sizeof(float) * n_embd) != cudaSuccess) return false;
        if (!d_post && cudaMalloc(&d_post, sizeof(float) * n_hc) != cudaSuccess) return false;
        if (!d_comb && cudaMalloc(&d_comb, sizeof(float) * n_hc * n_hc) != cudaSuccess) return false;
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

void hc_log_cuda_error(const char * label, cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[deepseek4-hc-direct] %s: %s\n", label, cudaGetErrorString(err));
    }
}

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

__device__ float hc_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void hc_mix_norm_kernel(const float * x,
                                   const __half * fn,
                                   int cols,
                                   int rows,
                                   float eps,
                                   float * mix) {
    __shared__ float smem[kThreads];
    __shared__ float inv_rms;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    float sumsq = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        const float v = x[c];
        sumsq += v * v;
    }
    smem[tid] = sumsq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        inv_rms = rsqrtf(smem[0] / (float) cols + eps);
    }
    __syncthreads();

    const __half * w = fn + (size_t) row * (size_t) cols;
    float dot = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        dot += __half2float(w[c]) * x[c];
    }
    smem[tid] = dot * inv_rms;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        mix[row] = smem[0];
    }
}

__global__ void hc_finish_kernel(const float * hc_state,
                                 const float * mix,
                                 const float * scale,
                                 const float * base,
                                 int n_embd,
                                 int n_hc,
                                 int sinkhorn_iters,
                                 float * working,
                                 float * post,
                                 float * comb) {
    __shared__ float split[kMaxMixDim];
    const int tid = threadIdx.x;

    if (tid == 0) {
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];
        const float sinkhorn_eps = 1.0e-6f;

        for (int i = 0; i < n_hc; ++i) {
            split[i] = hc_sigmoid(mix[i] * pre_scale + base[i]) + sinkhorn_eps;
        }
        for (int i = 0; i < n_hc; ++i) {
            split[n_hc + i] = 2.0f * hc_sigmoid(mix[n_hc + i] * post_scale + base[n_hc + i]);
            post[i] = split[n_hc + i];
        }

        float c[kMaxHc * kMaxHc];
        for (int dst = 0; dst < n_hc; ++dst) {
            float row_max = -1.0e30f;
            for (int src = 0; src < n_hc; ++src) {
                const int idx = src + dst * n_hc;
                const float v = mix[2 * n_hc + idx] * comb_scale + base[2 * n_hc + idx];
                c[idx] = v;
                row_max = v > row_max ? v : row_max;
            }
            float row_sum = 0.0f;
            for (int src = 0; src < n_hc; ++src) {
                const int idx = src + dst * n_hc;
                c[idx] = expf(c[idx] - row_max);
                row_sum += c[idx];
            }
            const float inv = 1.0f / row_sum;
            for (int src = 0; src < n_hc; ++src) {
                c[src + dst * n_hc] = c[src + dst * n_hc] * inv + sinkhorn_eps;
            }
        }

        for (int src = 0; src < n_hc; ++src) {
            float sum = 0.0f;
            for (int dst = 0; dst < n_hc; ++dst) sum += c[src + dst * n_hc];
            const float inv = 1.0f / (sum + sinkhorn_eps);
            for (int dst = 0; dst < n_hc; ++dst) c[src + dst * n_hc] *= inv;
        }
        for (int iter = 1; iter < sinkhorn_iters; ++iter) {
            for (int dst = 0; dst < n_hc; ++dst) {
                float sum = 0.0f;
                for (int src = 0; src < n_hc; ++src) sum += c[src + dst * n_hc];
                const float inv = 1.0f / (sum + sinkhorn_eps);
                for (int src = 0; src < n_hc; ++src) c[src + dst * n_hc] *= inv;
            }
            for (int src = 0; src < n_hc; ++src) {
                float sum = 0.0f;
                for (int dst = 0; dst < n_hc; ++dst) sum += c[src + dst * n_hc];
                const float inv = 1.0f / (sum + sinkhorn_eps);
                for (int dst = 0; dst < n_hc; ++dst) c[src + dst * n_hc] *= inv;
            }
        }

        for (int i = 0; i < n_hc * n_hc; ++i) {
            split[2 * n_hc + i] = c[i];
            comb[i] = c[i];
        }
    }
    __syncthreads();

    for (int d = tid; d < n_embd; d += blockDim.x) {
        float acc = 0.0f;
        for (int h = 0; h < n_hc; ++h) {
            acc += split[h] * hc_state[(size_t) h * n_embd + d];
        }
        working[d] = acc;
    }
}

bool hc_pre_device_locked(const void * hc_state_device,
                          const void * fn_device,
                          const void * scale_device,
                          const void * base_device,
                          int          n_embd,
                          int          n_hc,
                          int          sinkhorn_iters,
                          float        eps,
                          void *       working_device,
                          void *       post_device,
                          void *       comb_device,
                          bool         log_errors) {
    const int hc_dim = n_embd * n_hc;
    const int mix_dim = 2 * n_hc + n_hc * n_hc;
    auto fail = [log_errors](const char * label, cudaError_t err) {
        if (log_errors) {
            hc_log_cuda_error(label, err);
        }
        return false;
    };

    if (hc_state_device != g_scratch.d_state) {
        cudaError_t err = cudaMemcpy(g_scratch.d_state, hc_state_device,
                                     sizeof(float) * (size_t) hc_dim,
                                     cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            return fail("copy state d2d", err);
        }
    }

    hc_mix_norm_kernel<<<mix_dim, kThreads>>>(
        g_scratch.d_state,
        static_cast<const __half *>(fn_device),
        hc_dim,
        mix_dim,
        eps,
        g_scratch.d_mix);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return fail("mix kernel", err);
    }

    hc_finish_kernel<<<1, kThreads>>>(
        g_scratch.d_state,
        g_scratch.d_mix,
        static_cast<const float *>(scale_device),
        static_cast<const float *>(base_device),
        n_embd,
        n_hc,
        sinkhorn_iters,
        g_scratch.d_working,
        g_scratch.d_post,
        g_scratch.d_comb);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return fail("finish kernel", err);
    }
#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return fail("device sync", err);
    }
#endif

    if (working_device != g_scratch.d_working) {
        err = cudaMemcpy(working_device, g_scratch.d_working,
                         sizeof(float) * (size_t) n_embd,
                         cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            return fail("copy working d2d", err);
        }
    }
    if (post_device != g_scratch.d_post) {
        err = cudaMemcpy(post_device, g_scratch.d_post,
                         sizeof(float) * (size_t) n_hc,
                         cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            return fail("copy post d2d", err);
        }
    }
    if (comb_device != g_scratch.d_comb) {
        err = cudaMemcpy(comb_device, g_scratch.d_comb,
                         sizeof(float) * (size_t) n_hc * (size_t) n_hc,
                         cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            return fail("copy comb d2d", err);
        }
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return fail("post-copy sync", err);
    }

    return true;
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
    if (!g_scratch.ensure((size_t)hc_dim, (size_t)n_embd, (size_t)n_hc)) {
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
    const float inv_rms = 1.0f / std::sqrt(ss / (float)hc_dim + eps);
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
    return true;
}

bool deepseek4_cuda_hc_pre(const float * hc_state_host,
                           const void *  fn_device,
                           const float * scale_host,
                           const float * base_host,
                           int           n_embd,
                           int           n_hc,
                           int           sinkhorn_iters,
                           float         eps,
                           float *       working_host,
                           float *       post_host,
                           float *       comb_host) {
    if (!hc_state_host || !fn_device || !scale_host || !base_host ||
        !working_host || !post_host || !comb_host ||
        n_embd <= 0 || n_hc <= 0 || n_hc > kMaxHc) {
        return false;
    }
    const int hc_dim = n_embd * n_hc;
    const int mix_dim = 2 * n_hc + n_hc * n_hc;
    if (mix_dim > kMaxMixDim) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_scratch.ensure((size_t) hc_dim, (size_t) n_embd, (size_t) n_hc)) {
        return false;
    }
    if (cudaMemcpy(g_scratch.d_state, hc_state_host, sizeof(float) * (size_t) hc_dim,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(g_scratch.d_scale, scale_host, sizeof(float) * (size_t) mix_dim,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(g_scratch.d_base, base_host, sizeof(float) * (size_t) mix_dim,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    if (!hc_pre_device_locked(
            g_scratch.d_state,
            fn_device,
            g_scratch.d_scale,
            g_scratch.d_base,
            n_embd,
            n_hc,
            sinkhorn_iters,
            eps,
            g_scratch.d_working,
            g_scratch.d_post,
            g_scratch.d_comb,
            false)) {
        return false;
    }

    if (cudaMemcpy(working_host, g_scratch.d_working, sizeof(float) * (size_t) n_embd,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(post_host, g_scratch.d_post, sizeof(float) * (size_t) n_hc,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(comb_host, g_scratch.d_comb, sizeof(float) * (size_t) n_hc * (size_t) n_hc,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }

    return true;
}

bool deepseek4_cuda_hc_pre_device(const void * hc_state_device,
                                  const void * fn_device,
                                  const void * scale_device,
                                  const void * base_device,
                                  int          n_embd,
                                  int          n_hc,
                                  int          sinkhorn_iters,
                                  float        eps,
                                  void *       working_device,
                                  void *       post_device,
                                  void *       comb_device) {
    if (!hc_state_device || !fn_device || !scale_device || !base_device ||
        !working_device || !post_device || !comb_device ||
        n_embd <= 0 || n_hc <= 0 || n_hc > kMaxHc) {
        return false;
    }
    const int hc_dim = n_embd * n_hc;
    const int mix_dim = 2 * n_hc + n_hc * n_hc;
    if (mix_dim > kMaxMixDim) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_scratch.ensure((size_t) hc_dim, (size_t) n_embd, (size_t) n_hc)) {
        return false;
    }
    return hc_pre_device_locked(hc_state_device,
                                fn_device,
                                scale_device,
                                base_device,
                                n_embd,
                                n_hc,
                                sinkhorn_iters,
                                eps,
                                working_device,
                                post_device,
                                comb_device,
                                false);
}

bool deepseek4_cuda_hc_pre_device_params(const void * hc_state_device,
                                         const void * fn_device,
                                         const float * scale_host,
                                         const float * base_host,
                                         int           n_embd,
                                         int           n_hc,
                                         int           sinkhorn_iters,
                                         float         eps,
                                         void *        working_device,
                                         void *        post_device,
                                         void *        comb_device) {
    if (!hc_state_device || !fn_device || !scale_host || !base_host ||
        !working_device || !post_device || !comb_device ||
        n_embd <= 0 || n_hc <= 0 || n_hc > kMaxHc) {
        return false;
    }
    const int hc_dim = n_embd * n_hc;
    const int mix_dim = 2 * n_hc + n_hc * n_hc;
    if (mix_dim > kMaxMixDim) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_scratch.ensure((size_t) hc_dim, (size_t) n_embd, (size_t) n_hc)) {
        std::fprintf(stderr, "[deepseek4-hc-direct] ensure failed\n");
        return false;
    }
    if (cudaMemcpy(g_scratch.d_state, hc_state_device, sizeof(float) * (size_t) hc_dim,
                   cudaMemcpyDeviceToDevice) != cudaSuccess) {
        hc_log_cuda_error("copy state d2d", cudaGetLastError());
        return false;
    }
    if (cudaMemcpy(g_scratch.d_scale, scale_host, sizeof(float) * (size_t) mix_dim,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        hc_log_cuda_error("copy scale", cudaGetLastError());
        return false;
    }
    if (cudaMemcpy(g_scratch.d_base, base_host, sizeof(float) * (size_t) mix_dim,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        hc_log_cuda_error("copy base", cudaGetLastError());
        return false;
    }
    return hc_pre_device_locked(g_scratch.d_state,
                                fn_device,
                                g_scratch.d_scale,
                                g_scratch.d_base,
                                n_embd,
                                n_hc,
                                sinkhorn_iters,
                                eps,
                                working_device,
                                post_device,
                                comb_device,
                                true);
}

} // namespace dflash::common
