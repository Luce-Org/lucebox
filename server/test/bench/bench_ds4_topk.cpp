#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

// Performance-only sweep of the plain k = 512 top-k over rows and columns. The
// ctest-registered test_deepseek4_unit checks exact selected-set parity and
// padded -infinity ties across these shapes. The kernel is the backend's choice
// for each shape; GGML_DS4_TOPK_BLOCK_RADIX=0/1, set before the run (the
// backend reads it once), forces the device-wide sort or the block radix top-k.
//
//   bench_ds4_topk [rows,...]   (default 1,4,8,64,4096)
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct RunResult {
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


    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}


}  // namespace

int main(int argc, char ** argv) {
    std::vector<int> rows;
    for (std::string list = argc > 1 ? argv[1] : "1,4,8,64,4096"; !list.empty();) {
        const size_t comma = list.find(',');
        rows.push_back(std::atoi(list.substr(0, comma).c_str()));
        list = comma == std::string::npos ? "" : list.substr(comma + 1);
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "ggml_backend_cuda_init failed\n");
        return 1;
    }

    const bool previous_graphs_disabled =
        ggml_backend_cuda_set_graphs_disabled_override(true);
    constexpr int k = 512;
    const int shapes[] = {2048, 5120, 8192, 16384, 32768};
    const char * forced = std::getenv("GGML_DS4_TOPK_BLOCK_RADIX");
    bool all_ok = true;

    for (const int nrows : rows) {
        for (const int ncols : shapes) {
            if ((size_t) nrows * ncols > ((size_t) 1 << 27)) continue;   // 512 MB of scores
            RunResult run;
            const int iterations = nrows * (size_t) ncols > ((size_t) 1 << 22) ? 5 : 50;
            const bool ran = compute_topk(backend, ncols, nrows, k, 3, iterations, run);
            std::printf("kernel=%s rows=%d ncols=%d k=%d ms=%.4f status=%s\n",
                        forced ? (std::strcmp(forced, "1") == 0 ? "block_radix" : "device_sort") : "default",
                        nrows, ncols, k, run.milliseconds, ran ? "ok" : "FAILED");
            all_ok = all_ok && ran;
        }
    }

    ggml_backend_cuda_set_graphs_disabled_override(previous_graphs_disabled);
    ggml_backend_free(backend);
    return all_ok ? 0 : 1;
}
