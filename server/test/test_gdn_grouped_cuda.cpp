// Numerical parity for the classic and grouped-column fused Gated DeltaNet
// kernels. Each arm gets a separate graph and a freshly restored recurrent
// state so in-place state updates cannot hide a divergence.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr int64_t kStateDim = 128;
constexpr int64_t kQkHeads = 16;
constexpr int64_t kValueHeads = 48;
constexpr double kMaxNmse = 1e-6;

struct Inputs {
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> g;
    std::vector<float> beta;
    std::vector<float> state;
};

struct Result {
    std::vector<float> output;
    std::vector<float> state;
    bool ok = false;
};

struct ErrorStats {
    double nmse = 0.0;
    double max_abs = 0.0;
    bool finite = true;
};

void normalize_qk(std::vector<float> & values, int64_t n_tokens) {
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t head = 0; head < kQkHeads; ++head) {
            float * row = values.data() +
                (token * kQkHeads + head) * kStateDim;
            double sum_sq = 0.0;
            for (int64_t i = 0; i < kStateDim; ++i) {
                sum_sq += static_cast<double>(row[i]) * row[i];
            }
            const float inverse_norm =
                1.0f / std::sqrt(static_cast<float>(sum_sq) + 1e-6f);
            for (int64_t i = 0; i < kStateDim; ++i) {
                row[i] *= inverse_norm;
            }
        }
    }
}

Inputs make_inputs(int64_t n_tokens, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::uniform_real_distribution<float> gate(-0.10f, -0.001f);
    std::uniform_real_distribution<float> beta_dist(0.05f, 0.95f);

    Inputs inputs;
    inputs.q.resize(static_cast<size_t>(kStateDim * kQkHeads * n_tokens));
    inputs.k.resize(inputs.q.size());
    inputs.v.resize(static_cast<size_t>(kStateDim * kValueHeads * n_tokens));
    inputs.g.resize(static_cast<size_t>(kValueHeads * n_tokens));
    inputs.beta.resize(inputs.g.size());
    inputs.state.resize(static_cast<size_t>(kStateDim * kStateDim * kValueHeads));
    for (float & value : inputs.q) value = unit(rng);
    for (float & value : inputs.k) value = unit(rng);
    for (float & value : inputs.v) value = 0.5f * unit(rng);
    for (float & value : inputs.g) value = gate(rng);
    for (float & value : inputs.beta) value = beta_dist(rng);
    for (float & value : inputs.state) value = 0.1f * unit(rng);
    normalize_qk(inputs.q, n_tokens);
    normalize_qk(inputs.k, n_tokens);
    return inputs;
}

Result run_kernel(
        ggml_backend_t backend,
        const Inputs & inputs,
        int64_t n_tokens,
        bool grouped) {
    if (grouped) {
        unsetenv("DFLASH_GDN_NO_GROUPED_COLS");
        setenv("DFLASH_GDN_FORCE_GROUPED_COLS", "1", 1);
    } else {
        unsetenv("DFLASH_GDN_FORCE_GROUPED_COLS");
        setenv("DFLASH_GDN_NO_GROUPED_COLS", "1", 1);
    }

    ggml_init_params params = {32 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    Result result;
    if (ctx == nullptr) {
        return result;
    }

    ggml_tensor * q = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, kStateDim, kQkHeads, n_tokens, 1);
    ggml_tensor * k = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, kStateDim, kQkHeads, n_tokens, 1);
    ggml_tensor * v = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, kStateDim, kValueHeads, n_tokens, 1);
    ggml_tensor * g = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, 1, kValueHeads, n_tokens, 1);
    ggml_tensor * beta = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, 1, kValueHeads, n_tokens, 1);
    ggml_tensor * state = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, kStateDim, kStateDim, kValueHeads, 1);
    for (ggml_tensor * input : {q, k, v, g, beta, state}) {
        ggml_set_input(input);
    }

    ggml_tensor * packed = ggml_gated_delta_net(ctx, q, k, v, g, beta, state);
    ggml_gated_delta_net_set_skip_intermediate(packed, true);
    const size_t element_size = ggml_element_size(packed);
    ggml_tensor * output = ggml_view_4d(
        ctx, packed,
        kStateDim, kValueHeads, n_tokens, 1,
        kStateDim * element_size,
        kStateDim * kValueHeads * element_size,
        kStateDim * kValueHeads * n_tokens * element_size,
        0);
    ggml_tensor * final_state = ggml_view_4d(
        ctx, packed,
        kStateDim, kStateDim, kValueHeads, 1,
        kStateDim * element_size,
        kStateDim * kStateDim * element_size,
        kStateDim * kStateDim * kValueHeads * element_size,
        kStateDim * kValueHeads * n_tokens * element_size);
    output = ggml_cont(ctx, output);
    final_state = ggml_cont(ctx, final_state);
    ggml_set_output(output);
    ggml_set_output(final_state);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(graph, output);
    ggml_build_forward_expand(graph, final_state);
    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    bool ok = alloc != nullptr && ggml_gallocr_alloc_graph(alloc, graph);
    if (ok) {
        ggml_backend_tensor_set(q, inputs.q.data(), 0, inputs.q.size() * sizeof(float));
        ggml_backend_tensor_set(k, inputs.k.data(), 0, inputs.k.size() * sizeof(float));
        ggml_backend_tensor_set(v, inputs.v.data(), 0, inputs.v.size() * sizeof(float));
        ggml_backend_tensor_set(g, inputs.g.data(), 0, inputs.g.size() * sizeof(float));
        ggml_backend_tensor_set(
            beta, inputs.beta.data(), 0, inputs.beta.size() * sizeof(float));
        ggml_backend_tensor_set(
            state, inputs.state.data(), 0, inputs.state.size() * sizeof(float));
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    }

    result.output.resize(
        static_cast<size_t>(kStateDim * kValueHeads * n_tokens));
    result.state.resize(inputs.state.size());
    if (ok) {
        ggml_backend_tensor_get(
            output, result.output.data(), 0, result.output.size() * sizeof(float));
        ggml_backend_tensor_get(
            final_state, result.state.data(), 0, result.state.size() * sizeof(float));
    }
    result.ok = ok;

    if (alloc != nullptr) ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return result;
}

ErrorStats compare(
        const std::vector<float> & expected,
        const std::vector<float> & actual) {
    ErrorStats stats;
    double squared_error = 0.0;
    double expected_energy = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) {
            stats.finite = false;
            break;
        }
        const double difference =
            static_cast<double>(expected[i]) - static_cast<double>(actual[i]);
        squared_error += difference * difference;
        expected_energy += static_cast<double>(expected[i]) * expected[i];
        stats.max_abs = std::fmax(stats.max_abs, std::fabs(difference));
    }
    stats.nmse = expected_energy > 0.0 ? squared_error / expected_energy : INFINITY;
    return stats;
}

bool run_case(ggml_backend_t backend, int64_t n_tokens, uint32_t seed) {
    const Inputs inputs = make_inputs(n_tokens, seed);
    const Result classic = run_kernel(backend, inputs, n_tokens, false);
    const Result grouped = run_kernel(backend, inputs, n_tokens, true);
    const ErrorStats output_error = compare(classic.output, grouped.output);
    const ErrorStats state_error = compare(classic.state, grouped.state);
    const bool pass = classic.ok && grouped.ok &&
        output_error.finite && state_error.finite &&
        output_error.nmse <= kMaxNmse && state_error.nmse <= kMaxNmse;
    std::printf(
        "[%s] tokens=%lld output_nmse=%.6e output_max=%.6e "
        "state_nmse=%.6e state_max=%.6e\n",
        pass ? "PASS" : "FAIL",
        static_cast<long long>(n_tokens),
        output_error.nmse,
        output_error.max_abs,
        state_error.nmse,
        state_error.max_abs);
    return pass;
}

} // namespace

int main() {
    if (ggml_backend_cuda_get_device_count() == 0) {
        std::printf("SKIP: no CUDA device available\n");
        return 0;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "FAIL: unable to initialize CUDA backend\n");
        return 1;
    }

    int failures = 0;
    const int64_t cases[] = {1, 221, 477, 512, 768};
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        if (!run_case(backend, cases[i], 0x47444E00u + static_cast<uint32_t>(i))) {
            ++failures;
        }
    }
    unsetenv("DFLASH_GDN_FORCE_GROUPED_COLS");
    unsetenv("DFLASH_GDN_NO_GROUPED_COLS");
    ggml_backend_free(backend);

    if (failures != 0) {
        std::fprintf(stderr, "FAILED: %d grouped GDN parity case(s)\n", failures);
        return 1;
    }
    std::printf("ALL PASS: grouped GDN matches classic for Qwen3.5 shapes\n");
    return 0;
}
