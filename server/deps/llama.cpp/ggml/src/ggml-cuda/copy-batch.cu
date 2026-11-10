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

// k_copy_batch's loop over the first n descriptors of a fixed-size table, for a graph's tail (ggml_cuda_copy_tail_*).
struct copy_tail_params {
    const char * src[kCopyBatchMax];
    char *       dst[kCopyBatchMax];
    uint64_t     nbytes[kCopyBatchMax];
    int          n;
};

__global__ void k_copy_tail(const copy_tail_params p) {
    const int i = blockIdx.y;
    if (i >= p.n) {
        return;
    }
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

static_assert(sizeof(copy_tail_params) <= sizeof(ggml_cuda_copy_tail_storage) - sizeof(void *), "tail storage");

// Fills p with the non-empty descriptors; returns the launch's grid.
static dim3 copy_tail_fill(const ggml_cuda_copy_desc * descs, int n, copy_tail_params & p) {
    GGML_ASSERT(n >= 0 && n <= kCopyBatchMax);
    p = {};
    uint64_t max_bytes = 0;
    for (int i = 0; i < n; ++i) {
        if (descs[i].nbytes == 0) {
            continue;
        }
        p.src[p.n] = (const char *) descs[i].src;
        p.dst[p.n] = (char *) descs[i].dst;
        p.nbytes[p.n] = descs[i].nbytes;
        max_bytes = std::max<uint64_t>(max_bytes, descs[i].nbytes);
        ++p.n;
    }
    const uint64_t per_block = (uint64_t) 256 * 16;
    const int blocks_x = (int) std::min<uint64_t>(512, (max_bytes + per_block - 1) / per_block);
    return dim3(std::max(1, blocks_x), std::max(1, p.n), 1);
}

// Every block strides over every descriptor in turn, four 16-byte loads in flight per thread when both ends are
// 16-byte aligned: a handful of blocks keeps enough reads in flight to stream.
__global__ void k_copy_batch_lean(const copy_batch_params p, const int used) {
    const uint64_t tid = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t) gridDim.x * blockDim.x;
    for (int i = 0; i < used; ++i) {
        const char * src = p.src[i];
        char * dst = p.dst[i];
        const uint64_t n = p.nbytes[i];
        uint64_t done = 0;
        if ((((uintptr_t) src | (uintptr_t) dst) & 15) == 0) {
            const uint64_t n16 = n / 16;
            const int4 * s4 = (const int4 *) src;
            int4 * d4 = (int4 *) dst;
            uint64_t j = tid;
            for (; j + 3*stride < n16; j += 4*stride) {
                const int4 a = s4[j], b = s4[j + stride], c = s4[j + 2*stride], d = s4[j + 3*stride];
                d4[j] = a; d4[j + stride] = b; d4[j + 2*stride] = c; d4[j + 3*stride] = d;
            }
            for (; j < n16; j += stride) {
                d4[j] = s4[j];
            }
            done = n16 * 16;
        }
        for (uint64_t j = done + tid; j < n; j += stride) {
            dst[j] = src[j];
        }
    }
}

} // namespace

void ggml_cuda_copy_tail_launch(const ggml_cuda_copy_desc * descs, int n, cudaStream_t stream) {
    copy_tail_params p;
    const dim3 grid = copy_tail_fill(descs, n, p);
    k_copy_tail<<<grid, 256, 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_copy_tail_set(const ggml_cuda_copy_desc * descs, int n, ggml_cuda_copy_tail_storage * storage,
                             cudaKernelNodeParams * params) {
    copy_tail_params * p = (copy_tail_params *) storage;
    const dim3 grid = copy_tail_fill(descs, n, *p);
    storage->args[0] = p;
    *params = {};
    params->func = (void *) k_copy_tail;
    params->gridDim = grid;
    params->blockDim = dim3(256, 1, 1);
    params->sharedMemBytes = 0;
    params->kernelParams = storage->args;
    params->extra = nullptr;
}

const void * ggml_cuda_copy_tail_fn() {
    return (const void *) k_copy_tail;
}

void ggml_cuda_copy_batch_lean(const ggml_cuda_copy_desc * descs, int n, int blocks, cudaStream_t stream) {
    constexpr int threads = 256;
    for (int base = 0; base < n; base += kCopyBatchMax) {
        const int count = std::min(kCopyBatchMax, n - base);
        copy_batch_params p = {};
        int used = 0;
        for (int i = 0; i < count; ++i) {
            const ggml_cuda_copy_desc & d = descs[base + i];
            if (d.nbytes == 0) {
                continue;
            }
            p.src[used] = (const char *) d.src;
            p.dst[used] = (char *) d.dst;
            p.nbytes[used] = d.nbytes;
            ++used;
        }
        if (used > 0) {
            k_copy_batch_lean<<<std::max(1, blocks), threads, 0, stream>>>(p, used);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

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
    ggml_cuda_copy_batch(descs, n, ggml_cuda_enter_stream(ctx));
}

void ggml_backend_cuda_copy_batch_side(ggml_backend_t backend,
                                       const ggml_cuda_copy_desc * descs,
                                       int n) {
    if (n <= 0) {
        return;
    }
    GGML_ASSERT(ggml_backend_is_cuda(backend));
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    const cudaStream_t stream = ggml_cuda_enter_stream(ctx);
    ggml_cuda_side_stream_init(ctx);
    CUDA_CHECK(cudaEventRecord(ctx->side_copy_ready, stream));
    CUDA_CHECK(cudaStreamWaitEvent(ctx->side_copy_stream, ctx->side_copy_ready, 0));
    // 8 blocks of 256 threads stream the verify rollback (113 MiB) beside the first draft step's kernels. At 32
    // blocks the copy took ~530 GB/s and those kernels ran 3-5x slower; 8 spreads it thinner and still lands
    // in time (2 / 4 / 8 / 16 / 32 blocks 80.45 / 80.61 / 80.65 / 80.56 / 80.54 tok/s, the same text).
    ggml_cuda_copy_batch_lean(descs, n, 8, ctx->side_copy_stream);
    CUDA_CHECK(cudaEventRecord(ctx->side_copy_done, ctx->side_copy_stream));
    ctx->side_batch_pending = true;
}

void ggml_backend_cuda_set_side_overlap(ggml_backend_t backend, bool on) {
    if (!ggml_backend_is_cuda(backend)) {
        return;
    }
    ((ggml_backend_cuda_context *) backend->context)->side_overlap = on;
}
