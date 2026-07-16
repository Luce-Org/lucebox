// CUDA correctness coverage for the IQ4_XS MMQ stream-k scheduler.
//
// These five shapes reproduce the scheduler transitions validated on an
// 82-SM sm_86 GPU while backporting llama.cpp #22298:
//   - high-efficiency tiling without fixup
//   - a partial output-row tile
//   - stream-k with fixup
//   - the exact two-wave boundary
//   - a deeper K dimension
//
// On CUDA devices with a different SM count the exact scheduling decisions can
// differ, but the cases remain useful MMQ correctness coverage. Each case uses
// the same deterministic quantized weights and F32 activations on the CPU and
// CUDA backends, rejects non-finite output, and applies GGML's MUL_MAT oracle
// threshold of NMSE <= 5e-4.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double kMaxNmse = 5e-4;

struct Shape {
    const char * name;
    int64_t m;
    int64_t n;
    int64_t k;
};

bool compute_mul_mat(
        ggml_backend_t backend,
        const Shape & shape,
        const std::vector<uint8_t> & weights,
        const std::vector<float> & activations,
        std::vector<float> & output,
        std::string & error) {
    constexpr size_t kContextSize = 1024 * 1024;
    ggml_init_params params = {kContextSize, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        error = "ggml_init failed";
        return false;
    }

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_XS, shape.k, shape.m);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape.k, shape.n);
    ggml_set_name(a, "iq4_xs_weights");
    ggml_set_name(b, "f32_activations");
    ggml_set_input(a);
    ggml_set_input(b);

    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_set_name(out, "mul_mat_output");
    ggml_set_output(out);

    if (!ggml_backend_supports_op(backend, out)) {
        error = "backend does not support IQ4_XS MUL_MAT";
        ggml_free(ctx);
        return false;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (alloc == nullptr) {
        error = "ggml_gallocr_new failed";
        ggml_free(ctx);
        return false;
    }
    if (!ggml_gallocr_alloc_graph(alloc, graph)) {
        error = "ggml_gallocr_alloc_graph failed";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    if (weights.size() != ggml_nbytes(a) ||
        activations.size() * sizeof(float) != ggml_nbytes(b)) {
        error = "host data size does not match graph inputs";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(a, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(b, activations.data(), 0, activations.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        error = "ggml_backend_graph_compute failed with status " + std::to_string(status);
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_synchronize(backend);

    output.resize(static_cast<size_t>(shape.m * shape.n));
    if (output.size() * sizeof(float) != ggml_nbytes(out)) {
        error = "unexpected MUL_MAT output size";
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

bool run_case(ggml_backend_t cpu, ggml_backend_t cuda, const Shape & shape, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    std::vector<float> weights_f32(static_cast<size_t>(shape.k * shape.m));
    std::vector<float> activations(static_cast<size_t>(shape.k * shape.n));
    for (float & value : weights_f32) {
        value = distribution(rng);
    }
    for (float & value : activations) {
        value = distribution(rng);
    }

    const size_t quantized_size = ggml_row_size(GGML_TYPE_IQ4_XS, shape.k) * shape.m;
    std::vector<uint8_t> weights_iq4_xs(quantized_size);
    const size_t written = ggml_quantize_chunk(
        GGML_TYPE_IQ4_XS,
        weights_f32.data(),
        weights_iq4_xs.data(),
        0,
        shape.m,
        shape.k,
        nullptr);
    if (written != quantized_size) {
        std::fprintf(stderr,
            "[FAIL] %s: IQ4_XS quantizer wrote %zu bytes, expected %zu\n",
            shape.name, written, quantized_size);
        return false;
    }

    std::vector<float> cpu_output;
    std::vector<float> cuda_output;
    std::string error;
    if (!compute_mul_mat(cpu, shape, weights_iq4_xs, activations, cpu_output, error)) {
        std::fprintf(stderr, "[FAIL] %s: CPU: %s\n", shape.name, error.c_str());
        return false;
    }
    if (!compute_mul_mat(cuda, shape, weights_iq4_xs, activations, cuda_output, error)) {
        std::fprintf(stderr, "[FAIL] %s: CUDA: %s\n", shape.name, error.c_str());
        return false;
    }

    double squared_error = 0.0;
    double reference_energy = 0.0;
    double max_abs_error = 0.0;
    bool finite = true;
    for (size_t i = 0; i < cpu_output.size(); ++i) {
        const double reference = cpu_output[i];
        const double candidate = cuda_output[i];
        if (!std::isfinite(reference) || !std::isfinite(candidate)) {
            finite = false;
            break;
        }
        const double difference = reference - candidate;
        squared_error += difference * difference;
        reference_energy += reference * reference;
        max_abs_error = std::fmax(max_abs_error, std::fabs(difference));
    }

    const double nmse = reference_energy > 0.0
        ? squared_error / reference_energy
        : INFINITY;
    const bool pass = finite && nmse <= kMaxNmse;
    std::printf(
        "[%s] %-20s m=%lld n=%lld k=%lld finite=%s nmse=%.6e max_abs=%.6e\n",
        pass ? "PASS" : "FAIL",
        shape.name,
        static_cast<long long>(shape.m),
        static_cast<long long>(shape.n),
        static_cast<long long>(shape.k),
        finite ? "yes" : "no",
        nmse,
        max_abs_error);
    return pass;
}

} // namespace

int main() {
    const int device_count = ggml_backend_cuda_get_device_count();
    if (device_count == 0) {
        std::printf("SKIP: no CUDA device available\n");
        return 0;
    }

    char description[256] = {};
    ggml_backend_cuda_get_device_description(0, description, sizeof(description));
    std::printf("IQ4_XS MMQ stream-k correctness: device=%s, NMSE limit=%.1e\n",
        description, kMaxNmse);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (cpu == nullptr || cuda == nullptr) {
        std::fprintf(stderr, "FAIL: unable to initialize CPU and CUDA backends\n");
        if (cpu != nullptr) {
            ggml_backend_free(cpu);
        }
        if (cuda != nullptr) {
            ggml_backend_free(cuda);
        }
        return 1;
    }
    ggml_backend_cpu_set_n_threads(cpu, 4);

    const Shape shapes[] = {
        {"no_fixup",         5120, 512,  256},
        {"partial_rows",     5000, 512,  256},
        {"streamk_fixup",    4096, 512,  256},
        {"two_wave_boundary", 5121, 512, 256},
        {"deeper_k",         5120, 512, 1024},
    };

    int failures = 0;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i) {
        if (!run_case(cpu, cuda, shapes[i], 0x4D4D5100u + static_cast<uint32_t>(i))) {
            ++failures;
        }
    }

    ggml_backend_free(cuda);
    ggml_backend_free(cpu);
    ggml_quantize_free();

    if (failures != 0) {
        std::fprintf(stderr, "FAILED: %d/%zu IQ4_XS MMQ cases\n",
            failures, sizeof(shapes) / sizeof(shapes[0]));
        return 1;
    }
    std::printf("ALL PASS: %zu/%zu IQ4_XS MMQ cases\n",
        sizeof(shapes) / sizeof(shapes[0]),
        sizeof(shapes) / sizeof(shapes[0]));
    return 0;
}
