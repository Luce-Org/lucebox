#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr int kEmbeddings = 260;

struct CombineCase {
    int top_k;
    int tokens;
    bool include_shared;
};

struct Inputs {
    std::vector<float> down;
    std::vector<float> weights;
    std::vector<float> shared;
    std::vector<float> expected;
    int zero_weight_routes = 0;
};

Inputs make_inputs(const CombineCase & test, bool poison_masked = true) {
    Inputs result;
    result.down.resize((size_t) kEmbeddings * test.top_k * test.tokens);
    result.weights.resize((size_t) test.top_k * test.tokens);
    if (test.include_shared) {
        result.shared.resize((size_t) kEmbeddings * test.tokens);
    }
    result.expected.resize((size_t) kEmbeddings * test.tokens);

    for (int token = 0; token < test.tokens; ++token) {
        for (int expert = 0; expert < test.top_k; ++expert) {
            const size_t weight_offset = (size_t) token * test.top_k + expert;
            const bool masked = (token * 3 + expert) % 4 == 0;
            result.weights[weight_offset] = masked
                ? 0.0f
                : 0.125f * (float) (expert + 1);
            result.zero_weight_routes += masked ? 1 : 0;

            for (int embedding = 0; embedding < kEmbeddings; ++embedding) {
                const size_t down_offset =
                    (size_t) token * test.top_k * kEmbeddings +
                    (size_t) expert * kEmbeddings + embedding;
                result.down[down_offset] = masked
                    ? (poison_masked
                        ? std::numeric_limits<float>::quiet_NaN()
                        : 0.03125f * (float) ((embedding % 7) + 1))
                    : 0.03125f * (float) ((embedding % 11) - 5) *
                          (float) (expert + 1) +
                      0.015625f * (float) token;
            }
        }
    }

    for (int token = 0; token < test.tokens; ++token) {
        for (int embedding = 0; embedding < kEmbeddings; ++embedding) {
            const size_t output_offset = (size_t) token * kEmbeddings + embedding;
            const float shared_value = test.include_shared
                ? 0.0625f * (float) ((embedding + token) % 13 - 6)
                : 0.0f;
            if (test.include_shared) {
                result.shared[output_offset] = shared_value;
            }
            float sum = 0.0f;
            for (int expert = 0; expert < test.top_k; ++expert) {
                const size_t route = (size_t) token * test.top_k + expert;
                if (result.weights[route] == 0.0f) {
                    continue;
                }
                const size_t down_offset =
                    (size_t) token * test.top_k * kEmbeddings +
                    (size_t) expert * kEmbeddings + embedding;
                const float product = result.down[down_offset] * result.weights[route];
                sum += product;
            }
            sum += shared_value;
            result.expected[output_offset] = sum;
        }
    }

    return result;
}

bool run_backend(
        ggml_backend_t backend,
        const CombineCase & test,
        const Inputs & inputs,
        bool fused,
        std::vector<float> * output) {
    ggml_init_params params{};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "[ds4-moe-combine] ggml_init failed\n");
        return false;
    }

    ggml_tensor * down = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, kEmbeddings, test.top_k, test.tokens);
    ggml_tensor * weights = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, test.top_k, test.tokens);
    ggml_tensor * shared = test.include_shared
        ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kEmbeddings, test.tokens)
        : nullptr;
    ggml_set_input(down);
    ggml_set_input(weights);
    if (shared) {
        ggml_set_input(shared);
    }
    ggml_tensor * combined = nullptr;
    if (fused) {
        combined = ggml_ds4_moe_fused_combine_shared(
            ctx, down, weights, shared);
    } else {
        ggml_tensor * weights_3d = ggml_reshape_3d(
            ctx, weights, 1, test.top_k, test.tokens);
        ggml_tensor * routed = ggml_mul(ctx, down, weights_3d);
        routed = ggml_cont(ctx, ggml_permute(ctx, routed, 1, 0, 2, 3));
        routed = ggml_sum_rows(ctx, routed);
        routed = ggml_reshape_2d(ctx, routed, kEmbeddings, test.tokens);
        combined = shared ? ggml_add(ctx, routed, shared) : routed;
    }
    ggml_set_output(combined);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, combined);
    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph)) {
        std::fprintf(stderr, "[ds4-moe-combine] graph allocation failed\n");
        if (alloc) {
            ggml_gallocr_free(alloc);
        }
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(down, inputs.down.data(), 0, ggml_nbytes(down));
    ggml_backend_tensor_set(weights, inputs.weights.data(), 0, ggml_nbytes(weights));
    if (shared) {
        ggml_backend_tensor_set(shared, inputs.shared.data(), 0, ggml_nbytes(shared));
    }

    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_synchronize(backend);
        output->resize(inputs.expected.size());
        ggml_backend_tensor_get(
            combined, output->data(), 0, output->size() * sizeof(float));
    } else {
        std::fprintf(stderr, "[ds4-moe-combine] graph compute failed\n");
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}

bool equal_bytes(const std::vector<float> & expected,
                 const std::vector<float> & actual,
                 const char * label,
                 const CombineCase & test) {
    if (expected.size() == actual.size() &&
        std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)) == 0) {
        return true;
    }

    size_t first = 0;
    while (first < expected.size() && first < actual.size() &&
           std::memcmp(&expected[first], &actual[first], sizeof(float)) == 0) {
        ++first;
    }
    std::fprintf(stderr,
                 "[ds4-moe-combine] %s mismatch top_k=%d tokens=%d shared=%d index=%zu "
                 "expected=%g actual=%g\n",
                 label, test.top_k, test.tokens, test.include_shared ? 1 : 0, first,
                 first < expected.size() ? expected[first] : 0.0f,
                 first < actual.size() ? actual[first] : 0.0f);
    return false;
}

bool finite_output(const std::vector<float> & output, const CombineCase & test) {
    for (size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i])) {
            std::fprintf(stderr,
                         "[ds4-moe-combine] poisoned masked route reached output "
                         "top_k=%d tokens=%d shared=%d index=%zu value=%g\n",
                         test.top_k, test.tokens, test.include_shared ? 1 : 0,
                         i, output[i]);
            return false;
        }
    }
    return true;
}

Inputs make_signed_zero_inputs(const CombineCase & test) {
    Inputs result;
    result.down.assign((size_t) kEmbeddings * test.top_k * test.tokens, 1.0f);
    result.weights.assign((size_t) test.top_k * test.tokens, 0.0f);
    result.expected.assign((size_t) kEmbeddings * test.tokens, 0.0f);
    for (int token = 0; token < test.tokens; ++token) {
        result.weights[(size_t) token * test.top_k] = 1.0f;
        for (int embedding = 0; embedding < kEmbeddings; ++embedding) {
            result.down[(size_t) token * test.top_k * kEmbeddings + embedding] = -0.0f;
        }
    }
    return result;
}

bool benchmark_path(ggml_backend_t backend, int tokens, bool fused,
                    double * median_ms, double * mad_ms) {
    constexpr int n_embd = 4096;
    constexpr int top_k = 6;
    constexpr int warmups = 2;
    constexpr int samples = 7;

    ggml_init_params params{};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * down = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, n_embd, top_k, tokens);
    ggml_tensor * weights = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, top_k, tokens);
    ggml_tensor * shared = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_embd, tokens);
    ggml_set_input(down);
    ggml_set_input(weights);
    ggml_set_input(shared);

    ggml_tensor * output = nullptr;
    if (fused) {
        output = ggml_ds4_moe_fused_combine_shared(
            ctx, down, weights, shared);
    } else {
        ggml_tensor * weights_3d = ggml_reshape_3d(
            ctx, weights, 1, top_k, tokens);
        ggml_tensor * routed = ggml_mul(ctx, down, weights_3d);
        routed = ggml_cont(ctx, ggml_permute(ctx, routed, 1, 0, 2, 3));
        routed = ggml_sum_rows(ctx, routed);
        routed = ggml_reshape_2d(ctx, routed, n_embd, tokens);
        output = ggml_add(ctx, routed, shared);
    }
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph)) {
        if (alloc) {
            ggml_gallocr_free(alloc);
        }
        ggml_free(ctx);
        return false;
    }

    std::vector<float> down_h((size_t) n_embd * top_k * tokens, 0.03125f);
    std::vector<float> weights_h((size_t) top_k * tokens, 0.125f);
    std::vector<float> shared_h((size_t) n_embd * tokens, -0.0625f);
    ggml_backend_tensor_set(down, down_h.data(), 0, ggml_nbytes(down));
    ggml_backend_tensor_set(weights, weights_h.data(), 0, ggml_nbytes(weights));
    ggml_backend_tensor_set(shared, shared_h.data(), 0, ggml_nbytes(shared));

    std::vector<double> timings;
    timings.reserve(samples);
    bool ok = true;
    for (int i = 0; i < warmups + samples; ++i) {
        const auto start = std::chrono::steady_clock::now();
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS && ok;
        ggml_backend_synchronize(backend);
        const auto end = std::chrono::steady_clock::now();
        if (i >= warmups) {
            timings.push_back(std::chrono::duration<double, std::milli>(
                end - start).count());
        }
    }

    std::sort(timings.begin(), timings.end());
    *median_ms = timings[timings.size() / 2];
    std::vector<double> deviations;
    deviations.reserve(timings.size());
    for (double value : timings) {
        deviations.push_back(std::fabs(value - *median_ms));
    }
    std::sort(deviations.begin(), deviations.end());
    *mad_ms = deviations[deviations.size() / 2];

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    if (ggml_backend_cuda_get_device_count() <= 0) {
        std::puts("[ds4-moe-combine] SKIP: HIP device unavailable");
        return 77;
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t hip = ggml_backend_cuda_init(0);
    if (!cpu || !hip) {
        std::fprintf(stderr, "[ds4-moe-combine] backend initialization failed\n");
        if (hip) {
            ggml_backend_free(hip);
        }
        if (cpu) {
            ggml_backend_free(cpu);
        }
        return 1;
    }
    ggml_backend_cpu_set_n_threads(cpu, 1);

    const CombineCase cases[] = {
        {4, 1, false}, {4, 3, true}, {4, 33, false}, {4, 401, true},
        {6, 1, true},  {6, 3, false}, {6, 33, true},  {6, 401, false},
    };

    bool ok = true;
    for (const CombineCase & test : cases) {
        const Inputs inputs = make_inputs(test);
        std::vector<float> cpu_output;
        std::vector<float> hip_output;
        ok = run_backend(cpu, test, inputs, true, &cpu_output) && ok;
        ok = run_backend(hip, test, inputs, true, &hip_output) && ok;
        ok = finite_output(cpu_output, test) && finite_output(hip_output, test) && ok;
        ok = equal_bytes(inputs.expected, cpu_output, "CPU reference", test) && ok;
        ok = equal_bytes(cpu_output, hip_output, "HIP exact parity", test) && ok;

        const Inputs finite_inputs = make_inputs(test, false);
        std::vector<float> legacy_cpu_output;
        std::vector<float> legacy_hip_output;
        std::vector<float> fused_cpu_output;
        std::vector<float> fused_hip_output;
        ok = run_backend(cpu, test, finite_inputs, false, &legacy_cpu_output) && ok;
        ok = run_backend(hip, test, finite_inputs, false, &legacy_hip_output) && ok;
        ok = run_backend(cpu, test, finite_inputs, true, &fused_cpu_output) && ok;
        ok = run_backend(hip, test, finite_inputs, true, &fused_hip_output) && ok;
        ok = equal_bytes(legacy_cpu_output, fused_cpu_output,
                         "CPU legacy differential", test) && ok;
        ok = equal_bytes(legacy_hip_output, fused_hip_output,
                         "HIP legacy differential", test) && ok;
        std::printf("[ds4-moe-combine] top_k=%d tokens=%d shared=%d zero_routes=%d %s\n",
                    test.top_k, test.tokens, test.include_shared ? 1 : 0,
                    inputs.zero_weight_routes, ok ? "PASS" : "FAIL");
    }

    const CombineCase signed_zero_case{6, 3, false};
    const Inputs signed_zero_inputs = make_signed_zero_inputs(signed_zero_case);
    std::vector<float> signed_zero_legacy;
    std::vector<float> signed_zero_fused;
    ok = run_backend(hip, signed_zero_case, signed_zero_inputs, false,
                     &signed_zero_legacy) && ok;
    ok = run_backend(hip, signed_zero_case, signed_zero_inputs, true,
                     &signed_zero_fused) && ok;
    ok = equal_bytes(signed_zero_legacy, signed_zero_fused,
                     "HIP signed-zero legacy differential", signed_zero_case) && ok;
    ok = equal_bytes(signed_zero_inputs.expected, signed_zero_fused,
                     "HIP signed-zero +0 result", signed_zero_case) && ok;

    if (argc == 2 && std::strcmp(argv[1], "--benchmark") == 0) {
        for (int tokens : {401, 2048}) {
            double legacy_ms = 0.0;
            double legacy_mad = 0.0;
            double fused_ms = 0.0;
            double fused_mad = 0.0;
            ok = benchmark_path(
                hip, tokens, false, &legacy_ms, &legacy_mad) && ok;
            ok = benchmark_path(
                hip, tokens, true, &fused_ms, &fused_mad) && ok;
            std::printf(
                "[ds4-moe-combine-bench] tokens=%d legacy_ms=%.6f "
                "legacy_mad=%.6f fused_ms=%.6f fused_mad=%.6f speedup=%.6fx\n",
                tokens, legacy_ms, legacy_mad, fused_ms, fused_mad,
                fused_ms > 0.0 ? legacy_ms / fused_ms : 0.0);
        }
    }

    ggml_backend_free(hip);
    ggml_backend_free(cpu);
    return ok ? 0 : 1;
}
