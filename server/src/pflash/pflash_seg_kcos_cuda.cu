// Adjacent-segment mean-key cosine GPU kernels (shared CUDA/HIP via hip_compat).
// See pflash_seg_kcos.h.

#include "pflash_seg_kcos.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace luce::common {

namespace {

constexpr int kBlock = 256;

// Grid: N blocks (one block per segment), blockDim = 256.
// Accumulates mean key vector `k_bar` (and optionally mass-weighted key vector `k_att`)
// for segment i in a single pass over `keys`, L2-normalizing in-place.
__global__ void pflash_seg_mean_keys_kernel(
        const float * __restrict__ keys,
        const int32_t * __restrict__ bounds,
        float * __restrict__ seg_keys,
        const float * __restrict__ token_mass,
        float * __restrict__ seg_katt,
        int n_segments,
        int k_dim) {
    const int seg = blockIdx.x;
    if (seg >= n_segments) return;
    const int begin = bounds[2 * seg];
    const int end   = bounds[2 * seg + 1];
    const int len   = end > begin ? (end - begin) : 0;
    const float inv_len = len > 0 ? (1.0f / (float) len) : 0.0f;
    const bool do_katt = (token_mass != nullptr && seg_katt != nullptr);

    float inv_sum_m = 0.0f;
    if (do_katt && len > 0) {
        double sm = 0.0;
        for (int t = begin; t < end; ++t) {
            const float mt = token_mass[t];
            if (mt > 0.0f) sm += (double) mt;
        }
        if (sm > 1e-18 && isfinite(sm)) {
            inv_sum_m = (float) (1.0 / sm);
        }
    }

    float * dst = seg_keys + (size_t) seg * (size_t) k_dim;
    float * dst_att = do_katt ? (seg_katt + (size_t) seg * (size_t) k_dim) : nullptr;
    double local_sq = 0.0;
    double local_sq_att = 0.0;
    for (int d = threadIdx.x; d < k_dim; d += kBlock) {
        float acc = 0.0f;
        float acc_att = 0.0f;
        if (do_katt && inv_sum_m > 0.0f) {
            for (int t = begin; t < end; ++t) {
                const float kv = keys[(size_t) t * (size_t) k_dim + (size_t) d];
                acc += kv;
                const float mt = token_mass[t];
                if (mt > 0.0f) {
                    acc_att += (mt * inv_sum_m) * kv;
                }
            }
        } else {
            for (int t = begin; t < end; ++t) {
                acc += keys[(size_t) t * (size_t) k_dim + (size_t) d];
            }
        }
        const float m = acc * inv_len;
        dst[d] = m;
        local_sq += (double) m * (double) m;
        if (do_katt) {
            dst_att[d] = acc_att;
            local_sq_att += (double) acc_att * (double) acc_att;
        }
    }

    __shared__ double s_reduce[kBlock];
    s_reduce[threadIdx.x] = local_sq;
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const double norm_sq = s_reduce[0];
    if (norm_sq > 1e-16 && isfinite(norm_sq)) {
        const float inv_norm = (float) (1.0 / sqrt(norm_sq));
        for (int d = threadIdx.x; d < k_dim; d += kBlock) {
            dst[d] *= inv_norm;
        }
    } else {
        for (int d = threadIdx.x; d < k_dim; d += kBlock) {
            dst[d] = 0.0f;
        }
    }

    if (do_katt) {
        __syncthreads();
        s_reduce[threadIdx.x] = local_sq_att;
        __syncthreads();
        for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
            }
            __syncthreads();
        }
        const double norm_sq_att = s_reduce[0];
        if (inv_sum_m > 0.0f && norm_sq_att > 1e-16 && isfinite(norm_sq_att)) {
            const float inv_norm_att = (float) (1.0 / sqrt(norm_sq_att));
            for (int d = threadIdx.x; d < k_dim; d += kBlock) {
                dst_att[d] *= inv_norm_att;
            }
        } else {
            for (int d = threadIdx.x; d < k_dim; d += kBlock) {
                dst_att[d] = dst[d];
            }
        }
    }
}

// Grid: N blocks (one block per segment), blockDim = 256.
// Writes clamped cosine similarity between normalized seg_keys[i-1] and seg_keys[i] to kcos_out[i].
__global__ void pflash_adj_kcos_kernel(
        const float * __restrict__ seg_keys,
        float * __restrict__ kcos_out,
        int n_segments,
        int k_dim) {
    const int seg = blockIdx.x;
    if (seg >= n_segments) return;
    if (seg == 0) {
        if (threadIdx.x == 0) kcos_out[0] = 0.0f;
        return;
    }
    const float * a = seg_keys + (size_t) (seg - 1) * (size_t) k_dim;
    const float * b = seg_keys + (size_t) seg * (size_t) k_dim;
    float local_dot = 0.0f;
    for (int d = threadIdx.x; d < k_dim; d += kBlock) {
        local_dot += a[d] * b[d];
    }

    __shared__ float s_reduce[kBlock];
    s_reduce[threadIdx.x] = local_dot;
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float c = s_reduce[0];
        if (!isfinite(c) || c < 0.0f) c = 0.0f;
        else if (c > 1.0f) c = 1.0f;
        kcos_out[seg] = c;
    }
}

} // namespace

bool pflash_adjacent_segment_key_cosine_device(
        const float * keys_dev,
        const int32_t * bounds_dev,
        float * seg_keys_dev,
        float * kcos_dev,
        int n_segments,
        int k_dim,
        void * stream_ptr,
        const float * token_mass_dev,
        float * seg_katt_dev) {
    if (!keys_dev || !bounds_dev || !seg_keys_dev || !kcos_dev ||
        n_segments <= 0 || k_dim <= 0) {
        return false;
    }
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    (void)cudaGetLastError();  // discard any unrelated prior status
    pflash_seg_mean_keys_kernel<<<n_segments, kBlock, 0, stream>>>(
        keys_dev, bounds_dev, seg_keys_dev, token_mass_dev, seg_katt_dev, n_segments, k_dim);
    if (cudaGetLastError() != cudaSuccess) return false;
    pflash_adj_kcos_kernel<<<n_segments, kBlock, 0, stream>>>(
        seg_keys_dev, kcos_dev, n_segments, k_dim);
    if (cudaGetLastError() != cudaSuccess) return false;
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

} // namespace luce::common
