#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "rocmfp4.h"
#include "rocmfpx.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using quantize_fn = size_t (*)(const float *, void *, int64_t, int64_t, const float *);
using dequantize_fn = void (*)(const void *, float *, int64_t);

struct QuantCase {
    ggml_type type;
    const char * label;
    quantize_fn quantize;
    dequantize_fn dequantize;
};

struct Shape {
    int64_t k;
    int64_t m;
    int64_t n;
    const char * label;
};

enum class DispatchPath {
    MMVQ,
    MMQ,
};

static const char * dispatch_path_name(DispatchPath path) {
    return path == DispatchPath::MMVQ ? "MMVQ" : "MMQ";
}

static std::vector<float> make_values(size_t count, int stride, float scale) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        const int centered = (int) ((i * (size_t) stride + 17) % 113) - 56;
        values[i] = (float) centered * scale + 0.125f * std::sin((float) i * 0.017f);
    }
    return values;
}

static void dequantize_fp2(const void * src, float * dst, int64_t size) {
    rocmfpx_dequantize_row_fp2((const block_rocmfp2 *) src, dst, size);
}

static void dequantize_fp3(const void * src, float * dst, int64_t size) {
    rocmfpx_dequantize_row_fp3((const block_rocmfp3 *) src, dst, size);
}

static void dequantize_fp4_fast(const void * src, float * dst, int64_t size) {
    rocmfp4_dequantize_row_q4_0_fast((const block_rocmfp4_fast *) src, dst, size);
}

static bool run_backend(
        ggml_backend_t backend,
        ggml_type type,
        const Shape & shape,
        const std::vector<uint8_t> & weights_data,
        const std::vector<float> & input_data,
        std::vector<float> & output_data,
        DispatchPath expected_path) {
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        return false;
    }

    ggml_tensor * weights = ggml_new_tensor_2d(ctx, type, shape.k, shape.m);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape.k, shape.n);
    ggml_tensor * output = ggml_mul_mat(ctx, weights, input);
    ggml_set_input(weights);
    ggml_set_input(input);
    ggml_set_output(output);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::fprintf(stderr, "backend tensor allocation failed\n");
        ggml_free(ctx);
        return false;
    }

    bool ok = true;
    if (weights_data.size() != ggml_nbytes(weights)) {
        std::fprintf(
            stderr,
            "weight byte mismatch: quantized=%zu tensor=%zu\n",
            weights_data.size(),
            ggml_nbytes(weights));
        ok = false;
    }

    if (ok) {
        ggml_backend_tensor_set(weights, weights_data.data(), 0, weights_data.size());
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

        const size_t mmvq_before = ggml_backend_cuda_get_mmvq_launch_count();
        const size_t mmq_before = ggml_backend_cuda_get_mmq_launch_count();
        const int previous_mmvq_max =
            ggml_backend_cuda_set_mmvq_max_ncols_override(
                expected_path == DispatchPath::MMVQ ? 8 : 1);
        const bool previous_graphs_disabled =
            ggml_backend_cuda_set_graphs_disabled_override(true);
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        ggml_backend_cuda_set_graphs_disabled_override(previous_graphs_disabled);
        ggml_backend_cuda_set_mmvq_max_ncols_override(previous_mmvq_max);
        const size_t mmvq_delta =
            ggml_backend_cuda_get_mmvq_launch_count() - mmvq_before;
        const size_t mmq_delta =
            ggml_backend_cuda_get_mmq_launch_count() - mmq_before;
        if (!ok) {
            std::fprintf(stderr, "backend graph compute failed\n");
        } else {
            const bool dispatch_matches =
                expected_path == DispatchPath::MMVQ
                    ? mmvq_delta == 1 && mmq_delta == 0
                    : mmvq_delta == 0 && mmq_delta == 1;
            if (!dispatch_matches) {
                std::fprintf(
                    stderr,
                    "%s: expected %s dispatch, observed MMVQ=%zu MMQ=%zu\n",
                    shape.label,
                    dispatch_path_name(expected_path),
                    mmvq_delta,
                    mmq_delta);
                ok = false;
            }
        }
    }

    if (ok) {
        output_data.resize((size_t) ggml_nelements(output));
        ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

static std::vector<float> reference_mul_mat(
        const QuantCase & quant,
        const Shape & shape,
        const std::vector<uint8_t> & weights_data,
        const std::vector<float> & input_data) {
    std::vector<float> weights_f32((size_t) shape.k * shape.m);
    quant.dequantize(weights_data.data(), weights_f32.data(), shape.k * shape.m);

    std::vector<float> output((size_t) shape.m * shape.n, 0.0f);
    for (int64_t column = 0; column < shape.n; ++column) {
        for (int64_t row = 0; row < shape.m; ++row) {
            float sum = 0.0f;
            for (int64_t i = 0; i < shape.k; ++i) {
                sum += weights_f32[(size_t) row * shape.k + i] *
                       input_data[(size_t) column * shape.k + i];
            }
            output[(size_t) column * shape.m + row] = sum;
        }
    }
    return output;
}

static bool compare_outputs(
        const QuantCase & quant,
        const Shape & shape,
        const std::vector<float> & expected,
        const std::vector<float> & actual) {
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "%s/%s: output size mismatch\n", quant.label, shape.label);
        return false;
    }

    double squared_error = 0.0;
    double reference_power = 0.0;
    float max_abs_error = 0.0f;
    bool finite = true;

    for (size_t i = 0; i < actual.size(); ++i) {
        finite = finite && std::isfinite(actual[i]) && std::isfinite(expected[i]);
        const float error = actual[i] - expected[i];
        squared_error += (double) error * error;
        reference_power += (double) expected[i] * expected[i];
        max_abs_error = std::max(max_abs_error, std::fabs(error));
    }

    const double nmse = squared_error / std::max(reference_power, 1e-30);
    constexpr double max_nmse = 5e-4;
    const bool pass = finite && nmse <= max_nmse;
    std::printf(
        "%s/%s k=%lld m=%lld n=%lld: nmse=%.8g max_abs=%.8g %s\n",
        quant.label,
        shape.label,
        (long long) shape.k,
        (long long) shape.m,
        (long long) shape.n,
        nmse,
        max_abs_error,
        pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_case(
        ggml_backend_t hip_backend,
        const QuantCase & quant,
        const Shape & shape,
        DispatchPath expected_path) {
    const std::vector<float> weights_f32 =
        make_values((size_t) shape.k * shape.m, 37, 0.015625f);
    const std::vector<float> input_f32 =
        make_values((size_t) shape.k * shape.n, 29, 0.0078125f);

    const size_t row_size = ggml_row_size(quant.type, shape.k);
    std::vector<uint8_t> weights_quantized(row_size * (size_t) shape.m);
    const size_t quantized_bytes = quant.quantize(
        weights_f32.data(),
        weights_quantized.data(),
        shape.m,
        shape.k,
        nullptr);
    if (quantized_bytes != weights_quantized.size()) {
        std::fprintf(
            stderr,
            "%s/%s: quantizer wrote %zu bytes, expected %zu\n",
            quant.label,
            shape.label,
            quantized_bytes,
            weights_quantized.size());
        return false;
    }

    const std::vector<float> expected =
        reference_mul_mat(quant, shape, weights_quantized, input_f32);
    std::vector<float> actual;
    if (!run_backend(
            hip_backend, quant.type, shape, weights_quantized, input_f32, actual,
            expected_path)) {
        std::fprintf(
            stderr,
            "%s/%s: HIP %s run failed\n",
            quant.label,
            shape.label,
            dispatch_path_name(expected_path));
        return false;
    }
    return compare_outputs(quant, shape, expected, actual);
}

int main() {
    setenv("DFLASH_CUDA_MMVQ_FP2_AFFINE", "1", 1);
    setenv("DFLASH_CUDA_MMQ_FP2_AFFINE", "1", 1);
    setenv("DFLASH_CUDA_MMQ_FP2_AFFINE_GENERAL", "1", 1);
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) {
        std::fprintf(stderr, "failed to query HIP device 0\n");
        return 1;
    }
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7) != 0 &&
        std::strncmp(properties.gcnArchName, "gfx12", 5) != 0) {
        std::printf("SKIP: ROCmFPX dispatch test expects gfx1151/gfx12xx (found %s)\n",
                    properties.gcnArchName);
        return 0;
    }

    ggml_backend_t hip_backend = ggml_backend_cuda_init(0);
    if (!hip_backend) {
        std::fprintf(stderr, "failed to initialize HIP backend\n");
        return 1;
    }

    const QuantCase quant_cases[] = {
        {GGML_TYPE_Q2_0_ROCMFP2, "rocmfp2", rocmfpx_quantize_fp2, dequantize_fp2},
        {GGML_TYPE_Q3_0_ROCMFPX, "rocmfp3", rocmfpx_quantize_fp3, dequantize_fp3},
        {GGML_TYPE_Q4_0_ROCMFP4_FAST, "rocmfp4_fast", rocmfp4_quantize_q4_0_fast,
         dequantize_fp4_fast},
    };
    const Shape shapes[] = {
        {256, 64, 64, "full"},
        {288, 70, 67, "tail"},
        {4096, 64, 7, "expert_gate_up_n7"},
        {4096, 64, 16, "expert_gate_up_n16"},
        {4096, 64, 31, "expert_gate_up_n31"},
        {4096, 64, 33, "expert_gate_up_n33"},
        {2048, 64, 7, "expert_down_n7"},
        {2048, 64, 16, "expert_down_n16"},
        {2048, 64, 31, "expert_down_n31"},
        {2048, 64, 33, "expert_down_n33"},
        {4096, 4096, 31, "actual_gate_up_n31"},
        {2048, 4096, 31, "actual_down_n31"},
    };
    const char * shape_filter = std::getenv("DFLASH_TEST_SHAPE");
    bool matched_shape = false;

    bool ok = true;
    for (const QuantCase & quant : quant_cases) {
        for (const Shape & shape : shapes) {
            if (shape_filter && std::strcmp(shape.label, shape_filter) != 0) {
                continue;
            }
            matched_shape = true;
            if (std::strncmp(shape.label, "actual_", 7) == 0 &&
                quant.type != GGML_TYPE_Q2_0_ROCMFP2) {
                continue;
            }
            const DispatchPath expected_path =
                shape.n <= 8 ? DispatchPath::MMVQ : DispatchPath::MMQ;
            ok = test_case(hip_backend, quant, shape, expected_path) && ok;
        }
    }
    if (shape_filter && !matched_shape) {
        std::fprintf(stderr, "DFLASH_TEST_SHAPE matched no shape: %s\n",
                     shape_filter);
        ok = false;
    }

    ggml_backend_free(hip_backend);
    std::printf("%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
