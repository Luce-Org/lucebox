#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define C10_CUDA_CHECK(EXPR) \
  do { cudaError_t _e = (EXPR); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); std::abort(); } } while(0)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaPeekAtLastError())
