#include <hip/hip_runtime.h>

#include "ggml.h"
#include "rocmfp4.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using to_fp32_cuda_t = void (*)(const void *, float *, int64_t, hipStream_t);
to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

static void hip_check(hipError_t status, const char * operation) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, hipGetErrorString(status));
        std::exit(1);
    }
}

static std::vector<float> make_input(int64_t size) {
    std::vector<float> input((size_t) size);
    for (int64_t i = 0; i < size; ++i) {
        input[(size_t) i] = ((float) ((i*37) % 101) - 50.0f)*0.03125f;
    }
    return input;
}

template<typename Block>
static void check_layout(
        ggml_type type,
        int64_t size,
        void (*quantize)(const float *, Block *, int64_t),
        void (*dequantize)(const Block *, float *, int64_t),
        const char * label) {
    const std::vector<float> input = make_input(size);
    std::vector<Block> quantized((size_t) (size / QK_ROCMFP4));
    std::vector<float> expected((size_t) size);
    quantize(input.data(), quantized.data(), size);
    dequantize(quantized.data(), expected.data(), size);

    constexpr int64_t guard_count = 256;
    constexpr float guard_value = 12345.25f;
    std::vector<float> actual((size_t) (size + guard_count), guard_value);

    void * device_quantized = nullptr;
    float * device_actual = nullptr;
    hip_check(hipMalloc(&device_quantized, quantized.size()*sizeof(Block)), "hipMalloc input");
    hip_check(hipMalloc(&device_actual, actual.size()*sizeof(float)), "hipMalloc output");
    hip_check(
        hipMemcpy(
            device_quantized,
            quantized.data(),
            quantized.size()*sizeof(Block),
            hipMemcpyHostToDevice),
        "hipMemcpy input");
    hip_check(
        hipMemcpy(
            device_actual,
            actual.data(),
            actual.size()*sizeof(float),
            hipMemcpyHostToDevice),
        "hipMemcpy output");

    const to_fp32_cuda_t convert = ggml_get_to_fp32_cuda(type);
    if (convert == nullptr) {
        std::fprintf(stderr, "%s size=%lld: missing converter\n", label, (long long) size);
        std::exit(1);
    }

    convert(device_quantized, device_actual, size, nullptr);
    hip_check(hipGetLastError(), "ROCmFP4 converter launch");
    hip_check(hipDeviceSynchronize(), "ROCmFP4 converter synchronize");
    hip_check(
        hipMemcpy(
            actual.data(),
            device_actual,
            actual.size()*sizeof(float),
            hipMemcpyDeviceToHost),
        "hipMemcpy result");

    if (std::memcmp(actual.data(), expected.data(), (size_t) size*sizeof(float)) != 0) {
        for (int64_t i = 0; i < size; ++i) {
            if (std::memcmp(&actual[(size_t) i], &expected[(size_t) i], sizeof(float)) != 0) {
                std::fprintf(
                    stderr,
                    "%s size=%lld mismatch at %lld: gpu=%a cpu=%a\n",
                    label,
                    (long long) size,
                    (long long) i,
                    actual[(size_t) i],
                    expected[(size_t) i]);
                break;
            }
        }
        std::exit(1);
    }

    for (int64_t i = size; i < size + guard_count; ++i) {
        if (actual[(size_t) i] != guard_value) {
            std::fprintf(
                stderr,
                "%s size=%lld overwrote guard at +%lld\n",
                label,
                (long long) size,
                (long long) (i - size));
            std::exit(1);
        }
    }

    hip_check(hipFree(device_actual), "hipFree output");
    hip_check(hipFree(device_quantized), "hipFree input");
    std::printf("%s size=%lld: byte-identical, guard intact\n", label, (long long) size);
}

int main() {
    const int64_t sizes[] = { 32, 64, 96, 224, 256, 288 };
    for (const int64_t size : sizes) {
        check_layout<block_rocmfp4>(
            GGML_TYPE_Q4_0_ROCMFP4,
            size,
            rocmfp4_quantize_row_q4_0_ref,
            rocmfp4_dequantize_row_q4_0,
            "dual");
        check_layout<block_rocmfp4_fast>(
            GGML_TYPE_Q4_0_ROCMFP4_FAST,
            size,
            rocmfp4_quantize_row_q4_0_fast_ref,
            rocmfp4_dequantize_row_q4_0_fast,
            "fast");
    }
    return 0;
}
