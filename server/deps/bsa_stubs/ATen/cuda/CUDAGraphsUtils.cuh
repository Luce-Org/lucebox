#pragma once
#include "CUDAGeneratorImpl.h"
#include <cuda_runtime.h>
namespace at { namespace cuda { namespace philox {
__device__ __forceinline__ std::tuple<uint64_t, uint64_t> unpack(at::PhiloxCudaState arg) {
    return std::make_tuple(arg.seed_, arg.offset_);
}
} } }
