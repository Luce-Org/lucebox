#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if !defined(GGML_TEST_GRAPHS_ENABLED) || GGML_TEST_GRAPHS_ENABLED != 1
#error "Build this regression through its graph-enabled CMake target"
#endif

struct GraphStats {
    void * key = nullptr;
    int nodes = 0;
    unsigned long long total = 0;
    unsigned long long replay = 0;
    unsigned long long capture = 0;
    unsigned long long eager = 0;
    int enabled = 0;
    unsigned int records = 0;
};

static void capture_stats(enum ggml_log_level, const char * text, void * user_data) {
    std::fputs(text, stderr);
    const char * line = std::strstr(text, "[cuda-graph-stats]");
    if (!line) {
        return;
    }
    auto & stats = *static_cast<GraphStats *>(user_data);
    GGML_ASSERT(std::sscanf(line,
        "[cuda-graph-stats] key=%p n_nodes=%d total=%llu replay=%llu capture=%llu eager=%llu enabled=%d",
        &stats.key, &stats.nodes, &stats.total, &stats.replay, &stats.capture,
        &stats.eager, &stats.enabled) == 7);
    ++stats.records;
}

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    GGML_ASSERT(_putenv_s(name, value) == 0);
#else
    GGML_ASSERT(setenv(name, value, 1) == 0);
#endif
}

static void check_stats(const GraphStats & stats, const void * key, unsigned int total,
                        unsigned int replay, unsigned int capture, unsigned int eager) {
    GGML_ASSERT(stats.records == total);
    GGML_ASSERT(stats.key == key && stats.nodes == 1 && stats.enabled == 1);
    GGML_ASSERT(stats.total == total && stats.replay == replay);
    GGML_ASSERT(stats.capture == capture && stats.eager == eager);
}

static int run_arm(int device, bool force_skip, GraphStats & stats) {
    stats = {};
    ggml_backend_t backend = ggml_backend_cuda_init(device);
    GGML_ASSERT(backend != nullptr);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(backend, 1024 * 1024);
    GGML_ASSERT(buffer != nullptr);
    auto * base = static_cast<char *>(ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(base != nullptr);

    constexpr size_t n = 16;
    constexpr size_t bytes = n * sizeof(float);
    const size_t alignment = ggml_backend_buft_get_alignment(
        ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(alignment != 0);
    const size_t output_offset = (bytes + alignment - 1) / alignment * alignment;
    GGML_ASSERT(output_offset + bytes <= ggml_backend_buffer_get_size(buffer));

    std::vector<uint8_t> arena(1024 * 1024);
    const void * graph_key = nullptr;
    const bool previous_skip = ggml_backend_cuda_set_skip_props_check(false);
    for (int generation = 0; generation < 2; ++generation) {
        ggml_init_params params = {arena.size(), arena.data(), true};
        ggml_context * ctx = ggml_init(params);
        GGML_ASSERT(ctx != nullptr);
        ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_set_input(input);
        ggml_tensor * output = generation == 0 ? ggml_neg(ctx, input) : ggml_sqr(ctx, input);
        ggml_set_output(output);
        GGML_ASSERT(ggml_backend_tensor_alloc(buffer, input, base) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(ggml_backend_tensor_alloc(buffer, output, base + output_offset) == GGML_STATUS_SUCCESS);

        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, output);
        GGML_ASSERT(graph->uid == 0 && graph->n_nodes == 1);
        if (generation == 0) {
            graph_key = graph->nodes[0];
        }
        GGML_ASSERT(graph_key == graph->nodes[0]);
        ggml_backend_cuda_set_skip_props_check(generation == 1 && force_skip);

        for (unsigned int submission = 0; submission < 3; ++submission) {
            std::array<float, n> values;
            std::array<float, n> result;
            for (size_t i = 0; i < n; ++i) {
                values[i] = static_cast<float>(i + (generation == 0 ? 2 : 3) + 10 * submission);
            }
            result.fill(-99999.0f);
            ggml_backend_tensor_set(output, result.data(), 0, bytes);
            ggml_backend_tensor_set(input, values.data(), 0, bytes);
            GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(output, result.data(), 0, bytes);
            // Graph support is required at build time. Missing telemetry is a
            // failure; only an explicitly disabled runtime graph may skip here.
            if (generation == 0 && submission == 0 && stats.records == 1 && !stats.enabled) {
                std::fprintf(stderr, "SKIP: native graphs disabled for device %d or by environment\n", device);
                ggml_backend_synchronize(backend);
                ggml_free(ctx);
                ggml_backend_cuda_set_skip_props_check(previous_skip);
                ggml_backend_free(backend);
                ggml_backend_buffer_free(buffer);
                return 77;
            }
            for (size_t i = 0; i < n; ++i) {
                const float expected = generation == 0 ? -values[i] : values[i] * values[i];
                if (result[i] != expected) {
                    std::fprintf(stderr,
                        "FAIL skip=%d generation=%d submission=%u index=%zu: got %g expected %g\n",
                        force_skip, generation, submission, i, result[i], expected);
                    GGML_ABORT("stale or incorrect native graph output");
                }
            }
            check_stats(stats, graph_key, 3 * generation + submission + 1,
                        generation + (submission == 2),
                        generation + (submission >= 1), generation + 1);
        }
        ggml_backend_synchronize(backend);
        ggml_free(ctx);
    }
    ggml_backend_cuda_set_skip_props_check(previous_skip);
    ggml_backend_free(backend);
    ggml_backend_buffer_free(buffer);
    std::fprintf(stderr, "PASS device=%d forced_skip=%d: exact outputs, rebuild, capture and replay\n",
                 device, force_skip);
    return 0;
}

int main(int argc, char ** argv) {
    int device = 0;
    if (argc != 1) {
        char * end = nullptr;
        const long parsed = argc == 3 ? std::strtol(argv[2], &end, 10) : -1;
        if (argc != 3 || std::strcmp(argv[1], "--device") != 0 ||
            end == argv[2] || *end != '\0' || parsed < 0 || parsed >= GGML_CUDA_MAX_DEVICES) {
            std::fprintf(stderr, "Usage: %s [--device N]\n", argv[0]);
            return 1;
        }
        device = static_cast<int>(parsed);
    }
    const char * cursor = std::getenv("GGML_CUDA_DISABLE_GRAPHS_DEVICES");
    while (cursor && *cursor) {
        char * end = nullptr;
        const long disabled_device = std::strtol(cursor, &end, 10);
        if (end == cursor) {
            ++cursor;
            continue;
        }
        if (disabled_device == device) {
            std::fprintf(stderr, "SKIP: native graphs explicitly disabled for device %d\n", device);
            return 77;
        }
        cursor = end;
    }
    set_env("GGML_CUDA_GRAPH_STATS", "1");
    set_env("GGML_CUDA_GRAPH_STATS_EVERY", "1");
    GraphStats stats;
    ggml_log_set(capture_stats, &stats);
    if (ggml_backend_cuda_get_device_count() <= device) {
        std::fprintf(stderr, "SKIP: CUDA/HIP device %d unavailable\n", device);
        ggml_log_set(nullptr, nullptr);
        return 77;
    }
    char description[256] = {};
    ggml_backend_cuda_get_device_description(device, description, sizeof(description));
    std::fprintf(stderr, "Testing %s device %d: %s\n", GGML_CUDA_NAME, device, description);
    int status = run_arm(device, false, stats);
    if (status == 0) {
        status = run_arm(device, true, stats);
    }
    ggml_log_set(nullptr, nullptr);
    return status;
}
