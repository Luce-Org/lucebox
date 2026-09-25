#include "copy-batch.cuh"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kCopyBatchMax = ggml_cuda_copy_batch_max;

struct copy_batch_params {
    const char * src[kCopyBatchMax];
    char *       dst[kCopyBatchMax];
    uint64_t     nbytes[kCopyBatchMax];
};

// blockIdx.y selects the descriptor; the x blocks stride over its bytes,
// 16 at a time when both ends are 16-byte aligned.
__global__ void k_copy_batch(const copy_batch_params p) {
    const int i = blockIdx.y;
    const char * src = p.src[i];
    char * dst = p.dst[i];
    const uint64_t n = p.nbytes[i];
    const uint64_t tid = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t) gridDim.x * blockDim.x;
    uint64_t done = 0;
    if ((((uintptr_t) src | (uintptr_t) dst) & 15) == 0) {
        const uint64_t n16 = n / 16;
        const int4 * s4 = (const int4 *) src;
        int4 * d4 = (int4 *) dst;
        for (uint64_t j = tid; j < n16; j += stride) {
            d4[j] = s4[j];
        }
        done = n16 * 16;
    }
    for (uint64_t j = done + tid; j < n; j += stride) {
        dst[j] = src[j];
    }
}

} // namespace

void ggml_cuda_copy_batch(const ggml_cuda_copy_desc * descs, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    for (int base = 0; base < n; base += kCopyBatchMax) {
        const int count = std::min(kCopyBatchMax, n - base);
        copy_batch_params p = {};
        uint64_t max_bytes = 0;
        int used = 0;
        for (int i = 0; i < count; ++i) {
            const ggml_cuda_copy_desc & d = descs[base + i];
            if (d.nbytes == 0) {
                continue;
            }
            p.src[used] = (const char *) d.src;
            p.dst[used] = (char *) d.dst;
            p.nbytes[used] = d.nbytes;
            max_bytes = std::max<uint64_t>(max_bytes, d.nbytes);
            ++used;
        }
        if (used == 0) {
            continue;
        }
        // Enough blocks for the largest copy to stream at full bandwidth;
        // smaller descriptors leave their surplus blocks idle.
        const uint64_t per_block = (uint64_t) threads * 16;
        const int blocks_x = (int) std::min<uint64_t>(512, (max_bytes + per_block - 1) / per_block);
        k_copy_batch<<<dim3(std::max(1, blocks_x), used), threads, 0, stream>>>(p);
        CUDA_CHECK(cudaGetLastError());
    }
}

void ggml_backend_cuda_copy_batch_async(ggml_backend_t backend,
                                        const ggml_cuda_copy_desc * descs,
                                        int n) {
    if (n <= 0) {
        return;
    }
    GGML_ASSERT(ggml_backend_is_cuda(backend));
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(ctx->device);
    ggml_cuda_copy_batch(descs, n, ctx->stream());
}
