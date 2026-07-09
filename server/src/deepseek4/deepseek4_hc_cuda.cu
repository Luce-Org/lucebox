#include "deepseek4_hc_cuda.h"

#include "common/gpu_runtime_compat.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
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
    float * d_inv_rms = nullptr;
    float * d_mix = nullptr;
    float * d_scale = nullptr;
    float * d_base = nullptr;
    float * d_working = nullptr;
    float * d_post = nullptr;
    float * d_comb = nullptr;
    float * d_pre = nullptr;
    size_t state_cap = 0;
    size_t sums_cap = 0;
    size_t inv_rms_cap = 0;
    size_t mix_cap = 0;
    size_t working_cap = 0;
    size_t post_cap = 0;
    size_t comb_cap = 0;
    size_t pre_cap = 0;

    ~HcCudaScratch() {
        if (d_state) cudaFree(d_state);
        if (d_sums) cudaFree(d_sums);
        if (d_inv_rms) cudaFree(d_inv_rms);
        if (d_mix) cudaFree(d_mix);
        if (d_scale) cudaFree(d_scale);
        if (d_base) cudaFree(d_base);
        if (d_working) cudaFree(d_working);
        if (d_post) cudaFree(d_post);
        if (d_comb) cudaFree(d_comb);
        if (d_pre) cudaFree(d_pre);
    }

    bool ensure(size_t hc_dim, size_t n_embd, size_t n_hc, size_t n_tokens = 1) {
        auto ensure_buffer = [](float *& ptr, size_t & cap, size_t elems) {
            if (cap >= elems) {
                return true;
            }
            if (ptr) {
                cudaFree(ptr);
                ptr = nullptr;
                cap = 0;
            }
            if (cudaMalloc(&ptr, sizeof(float) * elems) != cudaSuccess) {
                return false;
            }
            cap = elems;
            return true;
        };
        if (!ensure_buffer(d_sums, sums_cap, (size_t) kSums * n_tokens)) return false;
        if (!ensure_buffer(d_inv_rms, inv_rms_cap, n_tokens)) return false;
        if (!ensure_buffer(d_mix, mix_cap, (size_t) kMaxMixDim * n_tokens)) return false;
        if (!d_scale && cudaMalloc(&d_scale, sizeof(float) * kMaxMixDim) != cudaSuccess) return false;
        if (!d_base && cudaMalloc(&d_base, sizeof(float) * kMaxMixDim) != cudaSuccess) return false;
        if (!ensure_buffer(d_working, working_cap, n_embd * n_tokens)) return false;
        if (!ensure_buffer(d_post, post_cap, n_hc * n_tokens)) return false;
        if (!ensure_buffer(d_comb, comb_cap, n_hc * n_hc * n_tokens)) return false;
        if (!ensure_buffer(d_pre, pre_cap, n_hc * n_tokens)) return false;
        if (!ensure_buffer(d_state, state_cap, hc_dim)) return false;
        return true;
    }
};

std::mutex g_mu;
std::unordered_map<void *, std::unique_ptr<HcCudaScratch>> g_scratch_by_stream;

HcCudaScratch & hc_scratch_for_stream(cudaStream_t stream) {
    void * key = reinterpret_cast<void *>(stream);
    auto it = g_scratch_by_stream.find(key);
    if (it == g_scratch_by_stream.end()) {
        auto inserted = g_scratch_by_stream.emplace(key, std::make_unique<HcCudaScratch>());
        it = inserted.first;
    }
    return *it->second;
}

void hc_log_cuda_error(const char * label, cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[deepseek4-hc-direct] %s: %s\n", label, cudaGetErrorString(err));
    }
}

bool hc_env_flag(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] && std::strcmp(value, "0") != 0;
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

__global__ void hc_finalize_inv_rms_kernel(const float * sums,
                                           int n_sums,
                                           int cols,
                                           float eps,
                                           float * inv_rms_out) {
    __shared__ float smem[kThreads];
    const int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < n_sums; i += blockDim.x) {
        acc += sums[i];
    }
    smem[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        inv_rms_out[0] = rsqrtf(smem[0] / (float) cols + eps);
    }
}

__global__ void hc_sumsq_batch_kernel(const float * x,
                                      int cols,
                                      float * sums) {
    __shared__ float smem[kThreads];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int token = blockIdx.y;
    const float * x_token = x + (size_t) token * (size_t) cols;
    float acc = 0.0f;
    for (int i = bid * blockDim.x + tid; i < cols; i += gridDim.x * blockDim.x) {
        const float v = x_token[i];
        acc += v * v;
    }
    smem[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        sums[(size_t) token * kSums + bid] = smem[0];
    }
}

__global__ void hc_finalize_inv_rms_batch_kernel(const float * sums,
                                                 int n_sums,
                                                 int cols,
                                                 float eps,
                                                 float * inv_rms_out) {
    __shared__ float smem[kThreads];
    const int tid = threadIdx.x;
    const int token = blockIdx.x;
    const float * sums_token = sums + (size_t) token * (size_t) n_sums;
    float acc = 0.0f;
    for (int i = tid; i < n_sums; i += blockDim.x) {
        acc += sums_token[i];
    }
    smem[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        inv_rms_out[token] = rsqrtf(smem[0] / (float) cols + eps);
    }
}

__global__ void hc_mix_kernel_device_inv(const float * x,
                                         const __half * fn,
                                         int cols,
                                         const float * inv_rms_ptr,
                                         float * mix) {
    __shared__ float smem[kThreads];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float inv_rms = inv_rms_ptr[0];
    float acc = 0.0f;
    const __half * w = fn + (size_t) row * (size_t) cols;
    for (int c = tid; c < cols; c += blockDim.x) {
        acc += __half2float(w[c]) * (x[c] * inv_rms);
    }
    smem[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        mix[row] = smem[0];
    }
}

__global__ void hc_mix_batch_kernel_device_inv(const float * x,
                                               const __half * fn,
                                               int cols,
                                               const float * inv_rms_ptr,
                                               int mix_dim,
                                               float * mix) {
    __shared__ float smem[kThreads];
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    const int tid = threadIdx.x;
    const float * x_token = x + (size_t) token * (size_t) cols;
    const float inv_rms = inv_rms_ptr[token];
    float acc = 0.0f;
    const __half * w = fn + (size_t) row * (size_t) cols;
    for (int c = tid; c < cols; c += blockDim.x) {
        acc += __half2float(w[c]) * (x_token[c] * inv_rms);
    }
    smem[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        mix[(size_t) token * (size_t) mix_dim + row] = smem[0];
    }
}

__device__ float hc_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
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

__global__ void hc_post_kernel(const float * residual_hc,
                               const float * block_out,
                               const float * post,
                               const float * comb,
                               int n_embd,
                               int n_hc,
                               float * out_hc) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_embd * n_hc;
    if (idx >= total) {
        return;
    }

    const int d = idx % n_embd;
    const int dst = idx / n_embd;
    float acc = block_out[d] * post[dst];
    for (int src = 0; src < n_hc; ++src) {
        acc += comb[dst + src * n_hc] *
               residual_hc[(size_t) src * (size_t) n_embd + (size_t) d];
    }
    out_hc[(size_t) dst * (size_t) n_embd + (size_t) d] = acc;
}

__global__ void hc_finish_batch_kernel(const float * hc_state,
                                       const float * mix,
                                       const float * scale,
                                       const float * base,
                                       int n_tokens,
                                       int n_embd,
                                       int n_hc,
                                       int sinkhorn_iters,
                                       float * working,
                                       float * post,
                                       float * comb,
                                       float * pre) {
    (void) n_tokens;
    __shared__ float split[kMaxMixDim];
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const int mix_dim = 2 * n_hc + n_hc * n_hc;
    const float * hc_state_token =
        hc_state + (size_t) token * (size_t) n_embd * (size_t) n_hc;
    const float * mix_token = mix + (size_t) token * (size_t) mix_dim;
    float * working_token = working + (size_t) token * (size_t) n_embd;
    float * post_token = post + (size_t) token * (size_t) n_hc;
    float * comb_token = comb + (size_t) token * (size_t) n_hc * (size_t) n_hc;
    float * pre_token = pre ? pre + (size_t) token * (size_t) n_hc : nullptr;

    if (tid == 0) {
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];
        const float sinkhorn_eps = 1.0e-6f;

        for (int i = 0; i < n_hc; ++i) {
            split[i] = hc_sigmoid(mix_token[i] * pre_scale + base[i]) + sinkhorn_eps;
            if (pre_token) {
                pre_token[i] = split[i];
            }
        }
        for (int i = 0; i < n_hc; ++i) {
            split[n_hc + i] =
                2.0f * hc_sigmoid(mix_token[n_hc + i] * post_scale + base[n_hc + i]);
            post_token[i] = split[n_hc + i];
        }

        float c[kMaxHc * kMaxHc];
        for (int dst = 0; dst < n_hc; ++dst) {
            float row_max = -1.0e30f;
            for (int src = 0; src < n_hc; ++src) {
                const int idx = src + dst * n_hc;
                const float v =
                    mix_token[2 * n_hc + idx] * comb_scale + base[2 * n_hc + idx];
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
            for (int dst = 0; dst < n_hc; ++dst) {
                sum += c[src + dst * n_hc];
            }
            const float inv = 1.0f / (sum + sinkhorn_eps);
            for (int dst = 0; dst < n_hc; ++dst) {
                c[src + dst * n_hc] *= inv;
            }
        }
        for (int iter = 1; iter < sinkhorn_iters; ++iter) {
            for (int dst = 0; dst < n_hc; ++dst) {
                float sum = 0.0f;
                for (int src = 0; src < n_hc; ++src) {
                    sum += c[src + dst * n_hc];
                }
                const float inv = 1.0f / (sum + sinkhorn_eps);
                for (int src = 0; src < n_hc; ++src) {
                    c[src + dst * n_hc] *= inv;
                }
            }
            for (int src = 0; src < n_hc; ++src) {
                float sum = 0.0f;
                for (int dst = 0; dst < n_hc; ++dst) {
                    sum += c[src + dst * n_hc];
                }
                const float inv = 1.0f / (sum + sinkhorn_eps);
                for (int dst = 0; dst < n_hc; ++dst) {
                    c[src + dst * n_hc] *= inv;
                }
            }
        }

        for (int i = 0; i < n_hc * n_hc; ++i) {
            split[2 * n_hc + i] = c[i];
            comb_token[i] = c[i];
        }
    }
    __syncthreads();

    for (int d = tid; d < n_embd; d += blockDim.x) {
        float acc = 0.0f;
        for (int h = 0; h < n_hc; ++h) {
            acc += split[h] * hc_state_token[(size_t) h * n_embd + d];
        }
        working_token[d] = acc;
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
                          HcCudaScratch & scratch,
                          cudaStream_t stream,
                          bool         sync_device,
                          bool         log_errors) {
    const int hc_dim = n_embd * n_hc;
    const int mix_dim = 2 * n_hc + n_hc * n_hc;
    auto fail = [log_errors](const char * label, cudaError_t err) {
        if (log_errors) {
            hc_log_cuda_error(label, err);
        }
        return false;
    };
    const float * hc_state_ptr = static_cast<const float *>(hc_state_device);

    hc_sumsq_kernel<<<kSums, kThreads, 0, stream>>>(
        hc_state_ptr,
        hc_dim,
        scratch.d_sums);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return fail("sumsq kernel", err);
    }

    hc_finalize_inv_rms_kernel<<<1, kThreads, 0, stream>>>(
        scratch.d_sums,
        kSums,
        hc_dim,
        eps,
        scratch.d_sums);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return fail("inv_rms kernel", err);
    }

    hc_mix_kernel_device_inv<<<mix_dim, kThreads, 0, stream>>>(
        hc_state_ptr,
        static_cast<const __half *>(fn_device),
        hc_dim,
        scratch.d_sums,
        scratch.d_mix);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return fail("mix kernel", err);
    }

    hc_finish_kernel<<<1, kThreads, 0, stream>>>(
        hc_state_ptr,
        scratch.d_mix,
        static_cast<const float *>(scale_device),
        static_cast<const float *>(base_device),
        n_embd,
        n_hc,
        sinkhorn_iters,
        static_cast<float *>(working_device),
        static_cast<float *>(post_device),
        static_cast<float *>(comb_device));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return fail("finish kernel", err);
    }
    if (sync_device) {
        err = stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            return fail("finish sync", err);
        }
    }

    return true;
}

bool hc_pre_batch_device_locked(const void * hc_state_device,
                                const void * fn_device,
                                const void * scale_device,
                                const void * base_device,
                                int          n_tokens,
                                int          n_embd,
                                int          n_hc,
                                int          sinkhorn_iters,
                                float        eps,
                                void *       working_device,
                                void *       post_device,
                                void *       comb_device,
                                void *       pre_device,
                                HcCudaScratch & scratch,
                                cudaStream_t stream,
                                bool         sync_device,
                                bool         log_errors) {
    const int hc_dim = n_embd * n_hc;
    const int mix_dim = 2 * n_hc + n_hc * n_hc;
    auto fail = [log_errors](const char * label, cudaError_t err) {
        if (log_errors) {
            hc_log_cuda_error(label, err);
        }
        return false;
    };
    if (mix_dim > kMixDim) {
        if (log_errors) {
            std::fprintf(stderr,
                         "[deepseek4-hc-direct] batch mix dim too large: %d > %d\n",
                         mix_dim, kMixDim);
        }
        return false;
    }
    const float * hc_state_ptr = static_cast<const float *>(hc_state_device);

    hc_sumsq_batch_kernel<<<dim3(kSums, n_tokens), kThreads, 0, stream>>>(
        hc_state_ptr,
        hc_dim,
        scratch.d_sums);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return fail("batch sumsq kernel", err);
    }

    hc_finalize_inv_rms_batch_kernel<<<n_tokens, kThreads, 0, stream>>>(
        scratch.d_sums,
        kSums,
        hc_dim,
        eps,
        scratch.d_inv_rms);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return fail("batch inv_rms kernel", err);
    }

    hc_mix_batch_kernel_device_inv<<<dim3(mix_dim, n_tokens), kThreads, 0, stream>>>(
        hc_state_ptr,
        static_cast<const __half *>(fn_device),
        hc_dim,
        scratch.d_inv_rms,
        mix_dim,
        scratch.d_mix);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return fail("batch mix kernel", err);
    }

    hc_finish_batch_kernel<<<n_tokens, kThreads, 0, stream>>>(
        hc_state_ptr,
        scratch.d_mix,
        static_cast<const float *>(scale_device),
        static_cast<const float *>(base_device),
        n_tokens,
        n_embd,
        n_hc,
        sinkhorn_iters,
        static_cast<float *>(working_device),
        static_cast<float *>(post_device),
        static_cast<float *>(comb_device),
        static_cast<float *>(pre_device));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return fail("batch finish kernel", err);
    }
    if (sync_device) {
        err = stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            return fail("batch finish sync", err);
        }
    }

    return true;
}

static cudaStream_t hc_backend_stream_or_default(ggml_backend_t backend) {
    if (!backend || !backend->context) {
        return nullptr;
    }
    if (!backend->device) {
        return nullptr;
    }
    const auto dev_type = ggml_backend_dev_type(backend->device);
    if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
        dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        return nullptr;
    }
    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    return cuda_ctx ? cuda_ctx->stream() : nullptr;
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
    HcCudaScratch & scratch = hc_scratch_for_stream(nullptr);
    if (!scratch.ensure((size_t)hc_dim, (size_t)n_embd, (size_t)n_hc)) {
        return false;
    }
    if (cudaMemcpy(scratch.d_state, hc_state_host, sizeof(float) * (size_t)hc_dim,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    hc_sumsq_kernel<<<kSums, kThreads>>>(scratch.d_state, hc_dim, scratch.d_sums);
    if (cudaGetLastError() != cudaSuccess) return false;
    std::vector<float> sums(kSums);
    if (cudaMemcpy(sums.data(), scratch.d_sums, sizeof(float) * sums.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    float ss = 0.0f;
    for (float v : sums) ss += v;
    const float inv_rms = 1.0f / std::sqrt(ss / (float)hc_dim + eps);
    hc_mix_kernel<<<kMixDim, kThreads>>>(scratch.d_state,
                                         static_cast<const __half *>(fn_device),
                                         hc_dim,
                                         inv_rms,
                                         scratch.d_mix);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (cudaMemcpy(mix_host, scratch.d_mix, sizeof(float) * kMixDim,
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
    HcCudaScratch & scratch = hc_scratch_for_stream(nullptr);
    if (!scratch.ensure((size_t) hc_dim, (size_t) n_embd, (size_t) n_hc)) {
        return false;
    }
    if (cudaMemcpy(scratch.d_state, hc_state_host, sizeof(float) * (size_t) hc_dim,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(scratch.d_scale, scale_host, sizeof(float) * (size_t) mix_dim,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(scratch.d_base, base_host, sizeof(float) * (size_t) mix_dim,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    if (!hc_pre_device_locked(
            scratch.d_state,
            fn_device,
            scratch.d_scale,
            scratch.d_base,
            n_embd,
            n_hc,
            sinkhorn_iters,
            eps,
            scratch.d_working,
            scratch.d_post,
            scratch.d_comb,
            scratch,
            nullptr,
            true,
            false)) {
        return false;
    }

    if (cudaMemcpy(working_host, scratch.d_working, sizeof(float) * (size_t) n_embd,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(post_host, scratch.d_post, sizeof(float) * (size_t) n_hc,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(comb_host, scratch.d_comb, sizeof(float) * (size_t) n_hc * (size_t) n_hc,
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
    HcCudaScratch & scratch = hc_scratch_for_stream(nullptr);
    if (!scratch.ensure((size_t) hc_dim, (size_t) n_embd, (size_t) n_hc)) {
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
                                scratch,
                                nullptr,
                                true,
                                false);
}

bool deepseek4_cuda_hc_pre_device_on_backend(ggml_backend_t backend,
                                             const void * hc_state_device,
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

    cudaStream_t stream = hc_backend_stream_or_default(backend);
    std::lock_guard<std::mutex> lock(g_mu);
    HcCudaScratch & scratch = hc_scratch_for_stream(stream);
    if (!scratch.ensure((size_t) hc_dim, (size_t) n_embd, (size_t) n_hc)) {
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
                                scratch,
                                stream,
                                true,
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
    HcCudaScratch & scratch = hc_scratch_for_stream(nullptr);
    if (!scratch.ensure((size_t) hc_dim, (size_t) n_embd, (size_t) n_hc)) {
        std::fprintf(stderr, "[deepseek4-hc-direct] ensure failed\n");
        return false;
    }
    if (cudaMemcpy(scratch.d_scale, scale_host, sizeof(float) * (size_t) mix_dim,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        hc_log_cuda_error("copy scale", cudaGetLastError());
        return false;
    }
    if (cudaMemcpy(scratch.d_base, base_host, sizeof(float) * (size_t) mix_dim,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        hc_log_cuda_error("copy base", cudaGetLastError());
        return false;
    }
    return hc_pre_device_locked(hc_state_device,
                                fn_device,
                                scratch.d_scale,
                                scratch.d_base,
                                n_embd,
                                n_hc,
                                sinkhorn_iters,
                                eps,
                                working_device,
                                post_device,
                                comb_device,
                                scratch,
                                nullptr,
                                true,
                                true);
}

bool deepseek4_cuda_hc_pre_device_params_on_backend(ggml_backend_t backend,
                                                    const void * hc_state_device,
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

    cudaStream_t stream = hc_backend_stream_or_default(backend);
    std::lock_guard<std::mutex> lock(g_mu);
    HcCudaScratch & scratch = hc_scratch_for_stream(stream);
    if (!scratch.ensure((size_t) hc_dim, (size_t) n_embd, (size_t) n_hc)) {
        std::fprintf(stderr, "[deepseek4-hc-direct] ensure failed\n");
        return false;
    }
    if (cudaMemcpy(scratch.d_scale, scale_host, sizeof(float) * (size_t) mix_dim,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        hc_log_cuda_error("copy scale", cudaGetLastError());
        return false;
    }
    if (cudaMemcpy(scratch.d_base, base_host, sizeof(float) * (size_t) mix_dim,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        hc_log_cuda_error("copy base", cudaGetLastError());
        return false;
    }
    return hc_pre_device_locked(hc_state_device,
                                fn_device,
                                scratch.d_scale,
                                scratch.d_base,
                                n_embd,
                                n_hc,
                                sinkhorn_iters,
                                eps,
                                working_device,
                                post_device,
                                comb_device,
                                scratch,
                                stream,
                                true,
                                true);
}

bool deepseek4_cuda_hc_post_device_on_backend(ggml_backend_t backend,
                                              const void * residual_hc_device,
                                              const void * block_out_device,
                                              const void * post_device,
                                              const void * comb_device,
                                              int          n_embd,
                                              int          n_hc,
                                              void *       out_hc_device) {
    if (!residual_hc_device || !block_out_device || !post_device ||
        !comb_device || !out_hc_device || n_embd <= 0 || n_hc <= 0 ||
        n_hc > kMaxHc) {
        return false;
    }

    cudaStream_t stream = hc_backend_stream_or_default(backend);
    const int total = n_embd * n_hc;
    const int blocks = (total + kThreads - 1) / kThreads;
    hc_post_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const float *>(residual_hc_device),
        static_cast<const float *>(block_out_device),
        static_cast<const float *>(post_device),
        static_cast<const float *>(comb_device),
        n_embd,
        n_hc,
        static_cast<float *>(out_hc_device));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        hc_log_cuda_error("hc_post kernel", err);
        return false;
    }
    return true;
}

bool deepseek4_cuda_hc_pre_batch_device_on_backend(ggml_backend_t backend,
                                                   const void * hc_state_device,
                                                   const void * fn_device,
                                                   const void * scale_device,
                                                   const void * base_device,
                                                   int          n_tokens,
                                                   int          n_embd,
                                                   int          n_hc,
                                                   int          sinkhorn_iters,
                                                   float        eps,
                                                   void *       working_device,
                                                   void *       post_device,
                                                   void *       comb_device,
                                                   void *       pre_device) {
    if (!hc_state_device || !fn_device || !scale_device || !base_device ||
        !working_device || !post_device || !comb_device ||
        n_tokens <= 0 || n_embd <= 0 || n_hc <= 0 || n_hc > kMaxHc) {
        return false;
    }
    const int hc_dim = n_embd * n_hc;
    const int mix_dim = 2 * n_hc + n_hc * n_hc;
    if (mix_dim > kMaxMixDim) {
        return false;
    }

    const bool use_default_stream =
        hc_env_flag("DFLASH_DS4_BATCH_HC_PRE_DIRECT_DEFAULT_STREAM");
    cudaStream_t stream = use_default_stream
        ? nullptr
        : hc_backend_stream_or_default(backend);
    std::lock_guard<std::mutex> lock(g_mu);
    HcCudaScratch & scratch = hc_scratch_for_stream(stream);
    if (!scratch.ensure((size_t) hc_dim, (size_t) n_embd, (size_t) n_hc, (size_t) n_tokens)) {
        return false;
    }
    return hc_pre_batch_device_locked(hc_state_device,
                                      fn_device,
                                      scale_device,
                                      base_device,
                                      n_tokens,
                                      n_embd,
                                      n_hc,
                                      sinkhorn_iters,
                                      eps,
                                      working_device,
                                      post_device,
                                      comb_device,
                                      pre_device,
                                      scratch,
                                      stream,
                                      !use_default_stream,
                                      false);
}

bool deepseek4_cuda_hc_pre_batch_device_params_on_backend(ggml_backend_t backend,
                                                          const void * hc_state_device,
                                                          const void * fn_device,
                                                          const float * scale_host,
                                                          const float * base_host,
                                                          int           n_tokens,
                                                          int           n_embd,
                                                          int           n_hc,
                                                          int           sinkhorn_iters,
                                                          float         eps,
                                                          void *        working_device,
                                                          void *        post_device,
                                                          void *        comb_device,
                                                          void *        pre_device) {
    if (!hc_state_device || !fn_device || !scale_host || !base_host ||
        !working_device || !post_device || !comb_device ||
        n_tokens <= 0 || n_embd <= 0 || n_hc <= 0 || n_hc > kMaxHc) {
        return false;
    }
    const int hc_dim = n_embd * n_hc;
    const int mix_dim = 2 * n_hc + n_hc * n_hc;
    if (mix_dim > kMaxMixDim) {
        return false;
    }

    const bool use_default_stream =
        hc_env_flag("DFLASH_DS4_BATCH_HC_PRE_DIRECT_DEFAULT_STREAM");
    cudaStream_t stream = use_default_stream
        ? nullptr
        : hc_backend_stream_or_default(backend);
    std::lock_guard<std::mutex> lock(g_mu);
    HcCudaScratch & scratch = hc_scratch_for_stream(stream);
    if (!scratch.ensure((size_t) hc_dim, (size_t) n_embd, (size_t) n_hc, (size_t) n_tokens)) {
        std::fprintf(stderr, "[deepseek4-hc-direct] ensure failed\n");
        return false;
    }
    cudaError_t err = cudaMemcpyAsync(scratch.d_scale,
                                      scale_host,
                                      sizeof(float) * (size_t) mix_dim,
                                      cudaMemcpyHostToDevice,
                                      stream);
    if (err != cudaSuccess) {
        hc_log_cuda_error("batch copy scale", err);
        return false;
    }
    err = cudaMemcpyAsync(scratch.d_base,
                          base_host,
                          sizeof(float) * (size_t) mix_dim,
                          cudaMemcpyHostToDevice,
                          stream);
    if (err != cudaSuccess) {
        hc_log_cuda_error("batch copy base", err);
        return false;
    }
    return hc_pre_batch_device_locked(hc_state_device,
                                      fn_device,
                                      scratch.d_scale,
                                      scratch.d_base,
                                      n_tokens,
                                      n_embd,
                                      n_hc,
                                      sinkhorn_iters,
                                      eps,
                                      working_device,
                                      post_device,
                                      comb_device,
                                      pre_device,
                                      scratch,
                                      stream,
                                      !use_default_stream,
                                      true);
}

} // namespace dflash::common
