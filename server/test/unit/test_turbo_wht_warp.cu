#include "common.cuh"
#include "tq3-quant.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define TEST_CUDA_CHECK(expr) do { \
    const cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

static __global__ void scalar_reference(
        const float * input, float * output, int64_t groups, int direction) {
    const int64_t group = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) {
        return;
    }
    float values[128];
    for (int i = 0; i < 128; ++i) {
        values[i] = input[group * 128 + i];
    }
    if (direction == 0) {
        tq3_rotate_forward(values);
    } else {
        tq3_rotate_inverse(values);
    }
    for (int i = 0; i < 128; ++i) {
        output[group * 128 + i] = values[i];
    }
}

static __global__ void warp_candidate(
        const float * input, float * output, int64_t groups, int direction) {
    constexpr int warp_size = 32;
    const int warp = threadIdx.x / warp_size;
    const int lane = threadIdx.x & (warp_size - 1);
    const int64_t group = (int64_t) blockIdx.x * (blockDim.x / warp_size) + warp;
    if (group >= groups) {
        return;
    }
    const int64_t base = group * 128 + lane * 4;
    float v0 = input[base + 0];
    float v1 = input[base + 1];
    float v2 = input[base + 2];
    float v3 = input[base + 3];
    if (direction == 0) {
        warp_tq3_rotate_forward(v0, v1, v2, v3);
    } else {
        warp_tq3_rotate_inverse(v0, v1, v2, v3);
    }
    output[base + 0] = v0;
    output[base + 1] = v1;
    output[base + 2] = v2;
    output[base + 3] = v3;
}

static bool run_case(int64_t groups, int direction) {
    const int64_t count = groups * 128;
    std::vector<float> input((size_t) count);
    for (int64_t i = 0; i < count; ++i) {
        input[(size_t) i] = (float) ((i * 37 + 11) % 257 - 128) / 64.0f;
    }

    float * d_input = nullptr;
    float * d_reference = nullptr;
    float * d_candidate = nullptr;
    TEST_CUDA_CHECK(cudaMalloc(&d_input, count * sizeof(float)));
    TEST_CUDA_CHECK(cudaMalloc(&d_reference, count * sizeof(float)));
    TEST_CUDA_CHECK(cudaMalloc(&d_candidate, count * sizeof(float)));
    TEST_CUDA_CHECK(cudaMemcpy(d_input, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));

    scalar_reference<<<(groups + 127) / 128, 128>>>(d_input, d_reference, groups, direction);
    warp_candidate<<<(groups + 3) / 4, 128>>>(d_input, d_candidate, groups, direction);
    TEST_CUDA_CHECK(cudaGetLastError());
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> reference((size_t) count);
    std::vector<float> candidate((size_t) count);
    TEST_CUDA_CHECK(cudaMemcpy(reference.data(), d_reference, count * sizeof(float), cudaMemcpyDeviceToHost));
    TEST_CUDA_CHECK(cudaMemcpy(candidate.data(), d_candidate, count * sizeof(float), cudaMemcpyDeviceToHost));

    int64_t mismatches = 0;
    for (int64_t i = 0; i < count; ++i) {
        if (reference[(size_t) i] != candidate[(size_t) i]) {
            ++mismatches;
            if (mismatches <= 4) {
                std::fprintf(stderr, "groups=%lld direction=%d i=%lld ref=%a got=%a\n",
                    (long long) groups, direction, (long long) i,
                    reference[(size_t) i], candidate[(size_t) i]);
            }
        }
    }

    cudaFree(d_candidate);
    cudaFree(d_reference);
    cudaFree(d_input);
    std::printf("[%s] groups=%lld direction=%d mismatches=%lld\n",
        mismatches == 0 ? "PASS" : "FAIL", (long long) groups,
        direction, (long long) mismatches);
    return mismatches == 0;
}

static double benchmark_case(int64_t groups, bool warp_kernel, int iterations) {
    const int64_t count = groups * 128;
    std::vector<float> input((size_t) count, 0.125f);
    float * d_input = nullptr;
    float * d_output = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    TEST_CUDA_CHECK(cudaMalloc(&d_input, count * sizeof(float)));
    TEST_CUDA_CHECK(cudaMalloc(&d_output, count * sizeof(float)));
    TEST_CUDA_CHECK(cudaMemcpy(d_input, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(cudaEventCreate(&start));
    TEST_CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < 100; ++i) {
        if (warp_kernel) {
            warp_candidate<<<(groups + 3) / 4, 128>>>(d_input, d_output, groups, 0);
        } else {
            scalar_reference<<<(groups + 127) / 128, 128>>>(d_input, d_output, groups, 0);
        }
    }
    TEST_CUDA_CHECK(cudaDeviceSynchronize());
    TEST_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        if (warp_kernel) {
            warp_candidate<<<(groups + 3) / 4, 128>>>(d_input, d_output, groups, 0);
        } else {
            scalar_reference<<<(groups + 127) / 128, 128>>>(d_input, d_output, groups, 0);
        }
    }
    TEST_CUDA_CHECK(cudaEventRecord(stop));
    TEST_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    TEST_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(d_output);
    cudaFree(d_input);
    return (double) elapsed_ms * 1000.0 / iterations;
}

int main() {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status == cudaErrorNoDevice) {
        std::puts("SKIP: no CUDA device");
        return 0;
    }
    TEST_CUDA_CHECK(device_status);
    if (device_count == 0) {
        std::puts("SKIP: no CUDA device");
        return 0;
    }
    TEST_CUDA_CHECK(cudaSetDevice(0));

    const int64_t group_counts[] = {1, 4, 32, 128, 384};
    int failures = 0;
    for (const int direction : {0, 1}) {
        for (const int64_t groups : group_counts) {
            if (!run_case(groups, direction)) {
                ++failures;
            }
        }
    }
    if (failures != 0) {
        std::fprintf(stderr, "FAILED: %d cases\n", failures);
        return 1;
    }
    std::puts("ALL PASS: scalar and warp FWHT outputs are bit-identical");
    for (const int64_t groups : {128LL, 384LL}) {
        const double scalar_us = benchmark_case(groups, false, 10000);
        const double warp_us = benchmark_case(groups, true, 10000);
        std::printf("BENCH groups=%lld scalar=%.3f us warp=%.3f us speedup=%.2fx reduction=%.1f%%\n",
            (long long) groups, scalar_us, warp_us, scalar_us / warp_us,
            100.0 * (1.0 - warp_us / scalar_us));
    }
    return 0;
}
