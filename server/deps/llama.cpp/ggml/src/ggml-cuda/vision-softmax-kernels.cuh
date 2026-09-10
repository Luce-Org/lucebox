#pragma once

// F32, wave32 adaptation of PyTorch 3d3aa833db84eed6b7f5595cb5f162c2f78300a4:
// aten/src/ATen/native/cuda/{PersistentSoftmax.cuh,SoftMax.cu,block_reduce.cuh}.
// Preserve source traversal, reduction, exp, and division order. Build without
// fast math or floating-point contraction. See the full upstream license below.
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace ggml_vision_softmax {

template<int LOG2_WIDTH>
static __global__ void persistent(const float * input, float * output, int width, int rows) {
    constexpr int power = 1 << LOG2_WIDTH;
    constexpr int lanes = power < 32 ? power : 32;
    constexpr int iterations = power / lanes;
    constexpr int batch = power <= 128 ? 2 : 1;
    const int first = (int(blockDim.y) * int(blockIdx.x) + int(threadIdx.y)) * batch;
    const int lane = int(threadIdx.x);
    float values[batch][iterations];
    float maximum[batch];
    float sums[batch];
    static_assert(std::is_same<decltype(std::exp(float{})), float>::value, "float exp required");
#pragma unroll
    for (int b = 0; b < batch; ++b) {
#pragma unroll
        for (int it = 0; it < iterations; ++it) {
            const int col = lane + it * lanes;
            values[b][it] = first + b < rows && col < width ?
                input[(first + b) * width + col] : -std::numeric_limits<float>::infinity();
        }
        maximum[b] = values[b][0];
#pragma unroll
        for (int it = 0; it < iterations; ++it) {
            maximum[b] = maximum[b] > values[b][it] ? maximum[b] : values[b][it];
        }
    }
#pragma unroll
    for (int offset = lanes / 2; offset > 0; offset /= 2) {
#pragma unroll
        for (int b = 0; b < batch; ++b) {
            const float other = __shfl_xor(maximum[b], offset, lanes);
            maximum[b] = maximum[b] < other ? other : maximum[b];
        }
    }
#pragma unroll
    for (int b = 0; b < batch; ++b) {
        sums[b] = 0.0f;
#pragma unroll
        for (int it = 0; it < iterations; ++it) {
            values[b][it] = std::exp(values[b][it] - maximum[b]);
            sums[b] += values[b][it];
        }
    }
#pragma unroll
    for (int offset = lanes / 2; offset > 0; offset /= 2) {
#pragma unroll
        for (int b = 0; b < batch; ++b) {
            sums[b] += __shfl_xor(sums[b], offset, lanes);
        }
    }
#pragma unroll
    for (int b = 0; b < batch; ++b) {
        if (first + b >= rows) {
            break;
        }
#pragma unroll
        for (int it = 0; it < iterations; ++it) {
            const int col = lane + it * lanes;
            if (col < width) {
                output[(first + b) * width + col] = values[b][it] / sums[b];
            }
        }
    }
}

template<bool SUM>
static __device__ __forceinline__ float combine(float a, float b) {
    if constexpr (SUM) {
        return a + b;
    } else {
        return a < b ? b : a;
    }
}

template<bool SUM>
static __device__ __forceinline__ float block_reduce(float value, float * shared) {
    const int tid = int(threadIdx.x), lane = tid % 32, warp = tid / 32;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        value = combine<SUM>(value, __shfl_down(value, offset, 32));
    }
    __syncthreads();
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    value = tid < 16 ? shared[lane] : (SUM ? 0.0f : -std::numeric_limits<float>::max());
    if (warp == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            value = combine<SUM>(value, __shfl_down(value, offset, 32));
        }
    }
    if (tid == 0) {
        if constexpr (SUM) {
            shared[0] = 1.0f / value;
        } else {
            shared[0] = value;
        }
    }
    __syncthreads();
    return shared[0];
}

struct alignas(16) vector4 { float val[4]; };

template<bool SUM>
static __device__ __forceinline__ float accumulate(float acc, float value, float maximum) {
    if constexpr (SUM) {
        return acc + __expf(value - maximum);
    } else {
        return ::max(acc, value);
    }
}

template<bool SUM, bool DIVISIBLE>
static __device__ __forceinline__ float local_reduce(const float * data, int size, float maximum) {
    float result = SUM ? 0.0f : -std::numeric_limits<float>::max();
    int offset = int(threadIdx.x);
    if constexpr (DIVISIBLE) {
        for (; offset * 4 < size; offset += 512) {
            const vector4 values = reinterpret_cast<const vector4 *>(data)[offset];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                result = accumulate<SUM>(result, values.val[j], maximum);
            }
        }
    } else {
        const int shift = int(uintptr_t(data) % 16) / int(sizeof(float));
        if (shift > 0) {
            data -= shift;
            size += shift;
            if (offset >= shift && offset < size) {
                result = accumulate<SUM>(result, data[offset], maximum);
            }
            size -= size < 512 ? size : 512;
            data += 512;
        }
        const int last = size % (4 * 512);
        for (; offset * 4 < size - last; offset += 512) {
            const vector4 values = reinterpret_cast<const vector4 *>(data)[offset];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                result = accumulate<SUM>(result, values.val[j], maximum);
            }
        }
        for (offset = size - last + int(threadIdx.x); offset < size; offset += 512) {
            result = accumulate<SUM>(result, data[offset], maximum);
        }
    }
    return result;
}

template<bool DIVISIBLE>
static __global__ void fast(const float * input, float * output, int width) {
    extern __shared__ float shared[];
    input += int(blockIdx.x) * width;
    output += int(blockIdx.x) * width;
    const float maximum = block_reduce<false>(local_reduce<false, DIVISIBLE>(input, width, 0.0f), shared);
    const float inverse = block_reduce<true>(local_reduce<true, DIVISIBLE>(input, width, maximum), shared);
    if constexpr (DIVISIBLE) {
        for (int offset = int(threadIdx.x); offset * 4 < width; offset += 512) {
            const vector4 values = reinterpret_cast<const vector4 *>(input)[offset];
            vector4 result;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                result.val[j] = __expf(values.val[j] - maximum) * inverse;
            }
            reinterpret_cast<vector4 *>(output)[offset] = result;
        }
    } else {
        for (int col = int(threadIdx.x); col < width; col += 512) {
            output[col] = __expf(input[col] - maximum) * inverse;
        }
    }
}

template<int LOG2_WIDTH>
static inline void launch_persistent(const float * input, float * output, int width, int rows, hipStream_t stream) {
    constexpr int power = 1 << LOG2_WIDTH;
    constexpr int lanes = power < 32 ? power : 32;
    constexpr int batch = power <= 128 ? 2 : 1;
    constexpr int groups = 128 / lanes;
    persistent<LOG2_WIDTH><<<(rows + groups * batch - 1) / (groups * batch), dim3(lanes, groups), 0, stream>>>(
        input, output, width, rows);
}

// Preconditions: F32, contiguous, aligned base, width in [16,4096], rows > 0,
// wave32, and total byte span <= INT_MAX. One launch, no allocation.
static inline void launch(const float * input, float * output, int width, int rows, hipStream_t stream) {
    if (width > 2048) {
        if (width % 4 == 0) {
            fast<true><<<rows, 512, 16 * sizeof(float), stream>>>(input, output, width);
        } else {
            fast<false><<<rows, 512, 16 * sizeof(float), stream>>>(input, output, width);
        }
        return;
    }
    int log2_width = 4;
    while ((1 << log2_width) < width) {
        ++log2_width;
    }
    switch (log2_width) {
#define GGML_VISION_SOFTMAX_CASE(N) case N: launch_persistent<N>(input, output, width, rows, stream); break
        GGML_VISION_SOFTMAX_CASE(4);
        GGML_VISION_SOFTMAX_CASE(5);
        GGML_VISION_SOFTMAX_CASE(6);
        GGML_VISION_SOFTMAX_CASE(7);
        GGML_VISION_SOFTMAX_CASE(8);
        GGML_VISION_SOFTMAX_CASE(9);
        GGML_VISION_SOFTMAX_CASE(10);
        GGML_VISION_SOFTMAX_CASE(11);
#undef GGML_VISION_SOFTMAX_CASE
    }
}

} // namespace ggml_vision_softmax

/*
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
