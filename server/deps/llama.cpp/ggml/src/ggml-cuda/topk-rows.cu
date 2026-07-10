// [TAG_TOPK_ROWS] Small-k top-k + softmax probabilities over the rows of a
// device-resident contiguous F32 tensor. Serves draft-head candidate
// extraction for speculative decoding (adaptive verify width, draft trees):
// k <= 8 candidates out of a ~100k-entry vocab distribution per drafted
// position, ~KB of readback instead of the full logits. Not a graph op: call
// after the producing graph_compute has returned; runs on the default stream.
// CONTRACT: callers must produce `logits` with the synchronous
// ggml_backend_graph_compute() (which drains the backend stream) - the
// async variant would race this helper's default-stream reads.
#include "common.cuh"
#include "ggml-cuda.h"

#if defined(GGML_USE_HIP)
#ifndef cudaPointerAttributes
#define cudaPointerAttributes    hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaMemoryTypeDevice     hipMemoryTypeDevice
#define cudaMemoryTypeManaged    hipMemoryTypeManaged
#endif
#endif

#define TOPK_ROWS_K        8
#define TOPK_ROWS_MAX_ROWS 64
#define TOPK_ROWS_BLOCK    256

// One block per row. Each thread keeps a sorted local top-K over its strided
// columns; thread 0 merges the block's candidates. Deterministic: ties break
// toward the lower column index, and the softmax denominator uses a fixed
// shared-memory tree reduction.
static __global__ void topk_rows_kernel(const float * __restrict__ x, const int ncols,
                                        float * __restrict__ out_vals,
                                        int32_t * __restrict__ out_ids) {
    constexpr int K = TOPK_ROWS_K;
    const float * xr = x + (size_t) blockIdx.x * (size_t) ncols;

    float tv[K];
    int   ti[K];
#pragma unroll
    for (int j = 0; j < K; ++j) { tv[j] = -INFINITY; ti[j] = 0x7fffffff; }
    for (int c = threadIdx.x; c < ncols; c += blockDim.x) {
        const float v = xr[c];
        if (v > tv[K-1] || (v == tv[K-1] && c < ti[K-1])) {
            int j = K - 1;
            while (j > 0 && (v > tv[j-1] || (v == tv[j-1] && c < ti[j-1]))) {
                tv[j] = tv[j-1]; ti[j] = ti[j-1]; --j;
            }
            tv[j] = v; ti[j] = c;
        }
    }

    __shared__ float sv[TOPK_ROWS_BLOCK * K];
    __shared__ int   si[TOPK_ROWS_BLOCK * K];
#pragma unroll
    for (int j = 0; j < K; ++j) {
        sv[threadIdx.x * K + j] = tv[j];
        si[threadIdx.x * K + j] = ti[j];
    }
    __syncthreads();

    __shared__ float s_val[K];
    if (threadIdx.x == 0) {
        float bv[K]; int bi[K];
#pragma unroll
        for (int j = 0; j < K; ++j) { bv[j] = -INFINITY; bi[j] = 0x7fffffff; }
        for (int e = 0; e < TOPK_ROWS_BLOCK * K; ++e) {
            const float v = sv[e]; const int idx = si[e];
            if (idx == 0x7fffffff) continue;
            if (v > bv[K-1] || (v == bv[K-1] && idx < bi[K-1])) {
                int j = K - 1;
                while (j > 0 && (v > bv[j-1] || (v == bv[j-1] && idx < bi[j-1]))) {
                    bv[j] = bv[j-1]; bi[j] = bi[j-1]; --j;
                }
                bv[j] = v; bi[j] = idx;
            }
        }
#pragma unroll
        for (int j = 0; j < K; ++j) {
            s_val[j] = bv[j];
            out_ids[blockIdx.x * K + j] = bi[j] == 0x7fffffff ? -1 : bi[j];
        }
    }
    __syncthreads();

    const float mx = s_val[0];
    float lsum = 0.0f;
    for (int c = threadIdx.x; c < ncols; c += blockDim.x) {
        lsum += expf(xr[c] - mx);
    }
    __shared__ float ssum[TOPK_ROWS_BLOCK];
    ssum[threadIdx.x] = lsum;
    __syncthreads();
    for (int s = TOPK_ROWS_BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float z = ssum[0];
#pragma unroll
        for (int j = 0; j < K; ++j) {
            out_vals[blockIdx.x * K + j] = s_val[j] == -INFINITY ? 0.0f : expf(s_val[j] - mx) / z;
        }
    }
}

bool ggml_backend_cuda_topk_rows(const struct ggml_tensor * logits, int k,
                                 float * probs_out, int32_t * ids_out) {
    if (logits == nullptr || probs_out == nullptr || ids_out == nullptr) return false;
    if (logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(logits)) return false;
    const int64_t ncols = logits->ne[0];
    const int64_t nrows = ggml_nrows(logits);
    if (k < 1 || k > TOPK_ROWS_K || nrows < 1 ||
        nrows > TOPK_ROWS_MAX_ROWS || ncols < TOPK_ROWS_K) {
        return false;
    }
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, logits->data) != cudaSuccess ||
        (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged)) {
        return false;
    }
    ggml_cuda_set_device(attr.device);

    // Tiny per-device scratch; decode is single-threaded per process (same
    // assumption as the persistent verify-graph slots in the server).
    static float   * d_vals = nullptr;
    static int32_t * d_ids  = nullptr;
    static int       d_dev  = -1;
    if (d_dev != attr.device) {
        if (d_vals != nullptr) { CUDA_CHECK(cudaFree(d_vals)); d_vals = nullptr; }
        if (d_ids  != nullptr) { CUDA_CHECK(cudaFree(d_ids));  d_ids  = nullptr; }
        CUDA_CHECK(cudaMalloc(&d_vals, sizeof(float)   * TOPK_ROWS_MAX_ROWS * TOPK_ROWS_K));
        CUDA_CHECK(cudaMalloc(&d_ids,  sizeof(int32_t) * TOPK_ROWS_MAX_ROWS * TOPK_ROWS_K));
        d_dev = attr.device;
    }

    topk_rows_kernel<<<(int) nrows, TOPK_ROWS_BLOCK>>>(
        (const float *) logits->data, (int) ncols, d_vals, d_ids);
    float   h_vals[TOPK_ROWS_MAX_ROWS * TOPK_ROWS_K];
    int32_t h_ids [TOPK_ROWS_MAX_ROWS * TOPK_ROWS_K];
    CUDA_CHECK(cudaMemcpy(h_vals, d_vals, sizeof(float)   * (size_t) nrows * TOPK_ROWS_K, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ids,  d_ids,  sizeof(int32_t) * (size_t) nrows * TOPK_ROWS_K, cudaMemcpyDeviceToHost));
    for (int64_t r = 0; r < nrows; ++r) {
        for (int j = 0; j < k; ++j) {
            probs_out[r * k + j] = h_vals[r * TOPK_ROWS_K + j];
            ids_out  [r * k + j] = h_ids [r * TOPK_ROWS_K + j];
        }
    }
    return true;
}
