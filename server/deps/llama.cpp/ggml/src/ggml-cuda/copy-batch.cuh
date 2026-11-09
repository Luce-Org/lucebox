#pragma once

#include "common.cuh"
#include "ggml-cuda.h"

// Descriptors per kernel launch.
constexpr int ggml_cuda_copy_batch_max = 48;

// Stream-ordered copies of independent {src, dst, nbytes} descriptors, one
// kernel launch per 48 descriptors. Destinations must not overlap each other
// or any source in the same call.
void ggml_cuda_copy_batch(const ggml_cuda_copy_desc * descs, int n, cudaStream_t stream);

// A hop's copies as the last kernel of the producer's captured graph. One launch of up to
// ggml_cuda_copy_batch_max descriptors (n may be 0: the launch then does nothing), the same copy loop as
// ggml_cuda_copy_batch; ggml_cuda_copy_tail_set fills a graph kernel node's parameters for the same launch (the
// storage must outlive the call that applies them), ggml_cuda_copy_tail_fn names the kernel.
struct ggml_cuda_copy_tail_storage {
    const char * src[ggml_cuda_copy_batch_max];
    char *       dst[ggml_cuda_copy_batch_max];
    uint64_t     nbytes[ggml_cuda_copy_batch_max];
    int          n;
    void *       args[1];
};
void ggml_cuda_copy_tail_launch(const ggml_cuda_copy_desc * descs, int n, cudaStream_t stream);
void ggml_cuda_copy_tail_set(const ggml_cuda_copy_desc * descs, int n, ggml_cuda_copy_tail_storage * storage,
                             cudaKernelNodeParams * params);
const void * ggml_cuda_copy_tail_fn();

// The same copies on `blocks` blocks that stride over every descriptor in turn, for a side stream beside other
// kernels.
void ggml_cuda_copy_batch_lean(const ggml_cuda_copy_desc * descs, int n, int blocks, cudaStream_t stream);
