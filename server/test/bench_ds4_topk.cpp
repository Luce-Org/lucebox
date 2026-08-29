#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

struct RunResult {
    std::vector<int32_t> indices;
    double milliseconds = 0.0;
};

std::vector<float> make_scores(int ncols, int nrows) {
    std::vector<float> scores((size_t) ncols * nrows);
    for (int row = 0; row < nrows; ++row) {
        for (int col = 0; col < ncols; ++col) {
            // An odd multiplier is a permutation modulo 2^24, so every score
            // in a row is distinct and exactly representable as float.
            const uint32_t value =
                ((uint32_t) col * 2654435761u + (uint32_t) row * 2246822519u) &
                0x00ffffffu;
            scores[(size_t) row * ncols + col] =
                (float) ((int32_t) value - 0x00800000);
        }
    }
    return scores;
}

bool compute_topk(
        ggml_backend_t backend,
        int ncols,
        int nrows,
        int k,
        bool block_radix,
        int warmup,
        int iterations,
        RunResult & result) {
    ggml_init_params params{};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        return false;
    }

    ggml_tensor * scores =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * selected = ggml_top_k(ctx, scores, k);
    ggml_set_input(scores);
    ggml_set_output(selected);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, selected);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::fprintf(stderr, "backend tensor allocation failed\n");
        ggml_free(ctx);
        return false;
    }

    const std::vector<float> host_scores = make_scores(ncols, nrows);
    ggml_backend_tensor_set(
        scores, host_scores.data(), 0, host_scores.size() * sizeof(float));
    setenv("GGML_DS4_TOPK_BLOCK_RADIX", block_radix ? "1" : "0", 1);

    bool ok = true;
    for (int i = 0; i < warmup; ++i) {
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        if (!ok) {
            break;
        }
    }
    ggml_backend_synchronize(backend);

    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; ok && i < iterations; ++i) {
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    }
    ggml_backend_synchronize(backend);
    const auto end = std::chrono::steady_clock::now();
    result.milliseconds =
        std::chrono::duration<double, std::milli>(end - begin).count() /
        iterations;

    if (ok) {
        result.indices.resize((size_t) nrows * k);
        ggml_backend_tensor_get(
            selected,
            result.indices.data(),
            0,
            result.indices.size() * sizeof(int32_t));
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

bool same_selected_set(
        const std::vector<int32_t> & reference,
        const std::vector<int32_t> & candidate,
        int ncols,
        int nrows,
        int k) {
    if (reference.size() != candidate.size()) {
        return false;
    }
    for (int row = 0; row < nrows; ++row) {
        const auto begin = (size_t) row * k;
        std::vector<int32_t> expected(
            reference.begin() + begin, reference.begin() + begin + k);
        std::vector<int32_t> actual(
            candidate.begin() + begin, candidate.begin() + begin + k);
        if (std::any_of(actual.begin(), actual.end(), [ncols](int32_t index) {
                return index < 0 || index >= ncols;
            })) {
            return false;
        }
        std::sort(expected.begin(), expected.end());
        std::sort(actual.begin(), actual.end());
        if (expected != actual) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "ggml_backend_cuda_init failed\n");
        return 1;
    }

    const bool previous_graphs_disabled =
        ggml_backend_cuda_set_graphs_disabled_override(true);
    constexpr int k = 512;
    constexpr int nrows = 4;
    const int shapes[] = {
        5121, 8192, 8193, 12288, 12289, 16384, 28673, 30720, 32768
    };
    bool all_ok = true;

    for (const int ncols : shapes) {
        RunResult full_sort;
        RunResult hierarchical;
        const int iterations = ncols == 30720 ? 100 : 5;
        const bool ran =
            compute_topk(
                backend, ncols, nrows, k, false, 2, iterations, full_sort) &&
            compute_topk(
                backend, ncols, nrows, k, true, 2, iterations, hierarchical);
        const bool exact = ran && same_selected_set(
            full_sort.indices, hierarchical.indices, ncols, nrows, k);
        const double speedup = hierarchical.milliseconds > 0.0
            ? full_sort.milliseconds / hierarchical.milliseconds
            : 0.0;
        std::printf(
            "ncols=%d rows=%d k=%d full_sort=%.3f ms hierarchical=%.3f ms "
            "speedup=%.2fx exact_set=%s\n",
            ncols,
            nrows,
            k,
            full_sort.milliseconds,
            hierarchical.milliseconds,
            speedup,
            exact ? "yes" : "NO");
        all_ok = all_ok && exact;
    }

    unsetenv("GGML_DS4_TOPK_BLOCK_RADIX");
    ggml_backend_cuda_set_graphs_disabled_override(previous_graphs_disabled);
    ggml_backend_free(backend);
    return all_ok ? 0 : 1;
}
