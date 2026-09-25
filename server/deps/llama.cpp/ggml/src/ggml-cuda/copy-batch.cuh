#pragma once

#include "common.cuh"
#include "ggml-cuda.h"

// Stream-ordered copies of independent {src, dst, nbytes} descriptors, one
// kernel launch per 48 descriptors. Destinations must not overlap each other
// or any source in the same call.
void ggml_cuda_copy_batch(const ggml_cuda_copy_desc * descs, int n, cudaStream_t stream);
