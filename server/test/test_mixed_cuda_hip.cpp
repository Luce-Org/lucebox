#include "common/dynamic_backend.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using dflash::common::PlacementBackend;
using dflash::common::backend_pair_capabilities;
using dflash::common::init_placement_backend;
using dflash::common::placement_backend_of;

namespace {

bool run_scale(ggml_backend_t backend, const char * label) {
    constexpr int64_t n = 4096;
    ggml_init_params params{};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_input(input);
    ggml_tensor * output = ggml_scale(ctx, input, 2.0f);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph)) {
        if (alloc) ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> host((size_t)n);
    for (int64_t i = 0; i < n; ++i) host[(size_t)i] = (float)i / 128.0f;
    ggml_backend_tensor_set(input, host.data(), 0, ggml_nbytes(input));
    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    std::vector<float> result((size_t)n);
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(output, result.data(), 0, ggml_nbytes(output));
    }

    bool ok = status == GGML_STATUS_SUCCESS;
    for (int64_t i = 0; ok && i < n; ++i) {
        ok = std::fabs(result[(size_t)i] - 2.0f * host[(size_t)i]) < 1.0e-5f;
    }
    std::printf("mixed-backend %s scale: %s\n", label, ok ? "ok" : "FAILED");
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}

bool run_cross_copy(ggml_backend_t src_backend,
                    ggml_backend_t dst_backend,
                    const char * label) {
    constexpr int64_t n = 4096;
    ggml_init_params params{};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * src_ctx = ggml_init(params);
    ggml_context * dst_ctx = ggml_init(params);
    if (!src_ctx || !dst_ctx) {
        if (src_ctx) ggml_free(src_ctx);
        if (dst_ctx) ggml_free(dst_ctx);
        return false;
    }

    ggml_tensor * src = ggml_new_tensor_1d(src_ctx, GGML_TYPE_F32, n);
    ggml_tensor * dst = ggml_new_tensor_1d(dst_ctx, GGML_TYPE_F32, n);
    ggml_backend_buffer_t src_buf = ggml_backend_alloc_ctx_tensors(src_ctx, src_backend);
    ggml_backend_buffer_t dst_buf = ggml_backend_alloc_ctx_tensors(dst_ctx, dst_backend);
    if (!src_buf || !dst_buf) {
        if (src_buf) ggml_backend_buffer_free(src_buf);
        if (dst_buf) ggml_backend_buffer_free(dst_buf);
        ggml_free(src_ctx);
        ggml_free(dst_ctx);
        return false;
    }

    std::vector<float> input((size_t)n);
    std::vector<float> output((size_t)n, 0.0f);
    for (int64_t i = 0; i < n; ++i) input[(size_t)i] = (float)(i * 17 - 31);
    ggml_backend_tensor_set(src, input.data(), 0, ggml_nbytes(src));
    ggml_backend_tensor_copy_async(src_backend, dst_backend, src, dst);
    ggml_backend_synchronize(dst_backend);
    ggml_backend_tensor_get(dst, output.data(), 0, ggml_nbytes(dst));
    const bool ok = input == output;
    std::printf("mixed-backend %s copy: %s\n", label, ok ? "ok" : "FAILED");

    ggml_backend_buffer_free(src_buf);
    ggml_backend_buffer_free(dst_buf);
    ggml_free(src_ctx);
    ggml_free(dst_ctx);
    return ok;
}

bool run_cross_graph(ggml_backend_t first,
                     ggml_backend_t second,
                     const char * label,
                     bool batch_split_copies) {
    constexpr int64_t n = 4096;
    ggml_init_params params{};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!ctx || !cpu) {
        if (cpu) ggml_backend_free(cpu);
        if (ctx) ggml_free(ctx);
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_input(input);
    ggml_tensor * first_head = ggml_scale(ctx, input, 2.0f);
    ggml_tensor * first_sibling = ggml_scale(ctx, input, 3.0f);
    ggml_tensor * second_middle = ggml_add(ctx, first_head, first_sibling);
    ggml_tensor * first_tail = ggml_scale(ctx, second_middle, 4.0f);
    ggml_set_output(first_tail);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, first_tail);

    ggml_backend_t backends[] = { first, second, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 3, 64, false, true);
    bool ok = sched != nullptr;
    if (ok) {
        ggml_backend_sched_set_tensor_backend(sched, input, first);
        ggml_backend_sched_set_tensor_backend(sched, first_head, first);
        ggml_backend_sched_set_tensor_backend(sched, first_sibling, first);
        ggml_backend_sched_set_tensor_backend(sched, second_middle, second);
        ggml_backend_sched_set_tensor_backend(sched, first_tail, first);
        ggml_backend_sched_set_batch_split_copies(
            sched, batch_split_copies);
        ok = ggml_backend_sched_alloc_graph(sched, graph);
    }

    std::vector<float> host((size_t)n);
    std::vector<float> result((size_t)n, 0.0f);
    for (int iteration = 0; ok && iteration < 3; ++iteration) {
        for (int64_t i = 0; i < n; ++i) {
            host[(size_t)i] = ((float)i + 17.0f * iteration) / 128.0f;
        }
        ggml_backend_tensor_set(input, host.data(), 0, ggml_nbytes(input));
        ok = ggml_backend_sched_graph_compute(sched, graph) ==
             GGML_STATUS_SUCCESS;
        if (ok) {
            ggml_backend_tensor_get(
                first_tail, result.data(), 0, ggml_nbytes(first_tail));
        }
        for (int64_t i = 0; ok && i < n; ++i) {
            ok = std::fabs(result[(size_t)i] - 20.0f * host[(size_t)i]) <
                 1.0e-4f;
        }
    }
    std::printf("mixed-backend %s graph batch=%d: %s\n", label,
                batch_split_copies ? 1 : 0,
                ok ? "ok" : "FAILED");

    if (sched) ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return ok;
}

bool run_noncontiguous_cross_graph(ggml_backend_t first,
                                   ggml_backend_t second,
                                   const char * label,
                                   bool transpose) {
    constexpr int64_t columns = 13;
    constexpr int64_t rows = 7;
    constexpr int64_t padded_columns = 19;
    const int64_t input_columns = transpose ? columns : padded_columns;

    ggml_init_params params{};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!ctx || !cpu) {
        if (cpu) ggml_backend_free(cpu);
        if (ctx) ggml_free(ctx);
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, input_columns, rows);
    ggml_set_input(input);
    ggml_tensor * scaled = ggml_scale(ctx, input, 2.0f);
    ggml_tensor * logical = transpose
        ? ggml_transpose(ctx, scaled)
        : ggml_view_2d(ctx, scaled, columns, rows, scaled->nb[1], 0);
    ggml_tensor * packed = ggml_cont(ctx, logical);
    ggml_tensor * output = ggml_scale(ctx, packed, 3.0f);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_t backends[] = { first, second, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 3, 128, false, true);
    bool ok = sched != nullptr;
    if (ok) {
        ggml_backend_sched_set_tensor_backend(sched, input, first);
        ggml_backend_sched_set_tensor_backend(sched, scaled, first);
        ggml_backend_sched_set_tensor_backend(sched, logical, first);
        ggml_backend_sched_set_tensor_backend(sched, packed, second);
        ggml_backend_sched_set_tensor_backend(sched, output, second);
        ggml_backend_sched_set_batch_split_copies(sched, true);
        ok = ggml_backend_sched_alloc_graph(sched, graph);
    }

    const size_t input_size = (size_t) input_columns * (size_t) rows;
    const size_t output_size = (size_t) columns * (size_t) rows;
    std::vector<float> host(input_size);
    std::vector<float> result(output_size, 0.0f);
    for (int iteration = 0; ok && iteration < 2; ++iteration) {
        for (size_t i = 0; i < host.size(); ++i) {
            host[i] = ((float) i + 11.0f * iteration) / 32.0f;
        }
        ggml_backend_tensor_set(input, host.data(), 0, ggml_nbytes(input));
        ok = ggml_backend_sched_graph_compute(sched, graph) ==
             GGML_STATUS_SUCCESS;
        if (ok) {
            ggml_backend_tensor_get(
                output, result.data(), 0, ggml_nbytes(output));
        }
        if (transpose) {
            for (int64_t original_column = 0;
                 ok && original_column < columns; ++original_column) {
                for (int64_t original_row = 0;
                     ok && original_row < rows; ++original_row) {
                    const size_t output_index =
                        (size_t) original_column * (size_t) rows +
                        (size_t) original_row;
                    const size_t input_index =
                        (size_t) original_row * (size_t) columns +
                        (size_t) original_column;
                    ok = std::fabs(
                        result[output_index] - 6.0f * host[input_index]) <
                        1.0e-5f;
                }
            }
        } else {
            for (int64_t row = 0; ok && row < rows; ++row) {
                for (int64_t column = 0; ok && column < columns; ++column) {
                    const size_t output_index =
                        (size_t) row * (size_t) columns + (size_t) column;
                    const size_t input_index =
                        (size_t) row * (size_t) padded_columns +
                        (size_t) column;
                    ok = std::fabs(
                        result[output_index] - 6.0f * host[input_index]) <
                        1.0e-5f;
                }
            }
        }
    }

    std::printf("mixed-backend %s %s graph: %s\n", label,
                transpose ? "permuted" : "strided",
                ok ? "ok" : "FAILED");
    if (sched) ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return ok;
}

bool reject_invalid_compiled_device() {
    std::string error;
    ggml_backend_t backend = init_placement_backend(
        dflash::common::compiled_placement_backend(),
        std::numeric_limits<int>::max(), &error);
    const bool ok = backend == nullptr &&
        error.find("is out of range (found ") != std::string::npos;
    if (backend) ggml_backend_free(backend);
    std::printf("mixed-backend compiled device bounds: %s\n",
                ok ? "ok" : "FAILED");
    return ok;
}

struct SchedulerGraphCase {
    ggml_context * ctx = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * first = nullptr;
    ggml_tensor * second = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    int64_t elements = 0;
};

SchedulerGraphCase make_resize_graph(int64_t elements) {
    SchedulerGraphCase result;
    result.elements = elements;
    ggml_init_params params{};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    result.ctx = ggml_init(params);
    if (!result.ctx) return result;

    result.input = ggml_new_tensor_1d(
        result.ctx, GGML_TYPE_F32, elements);
    ggml_set_input(result.input);
    result.first = ggml_scale(result.ctx, result.input, 2.0f);
    result.second = ggml_scale(result.ctx, result.first, 3.0f);
    result.output = ggml_scale(result.ctx, result.second, 4.0f);
    ggml_set_output(result.output);
    result.graph = ggml_new_graph(result.ctx);
    ggml_build_forward_expand(result.graph, result.output);
    return result;
}

bool pin_resize_graph(ggml_backend_sched_t sched,
                      const SchedulerGraphCase & graph,
                      ggml_backend_t first,
                      ggml_backend_t second) {
    if (!sched || !graph.ctx || !graph.input || !graph.first ||
        !graph.second || !graph.output || !graph.graph) {
        return false;
    }
    ggml_backend_sched_set_tensor_backend(sched, graph.input, first);
    ggml_backend_sched_set_tensor_backend(sched, graph.first, first);
    ggml_backend_sched_set_tensor_backend(sched, graph.second, second);
    ggml_backend_sched_set_tensor_backend(sched, graph.output, first);
    return ggml_backend_sched_alloc_graph(sched, graph.graph);
}

bool set_and_check_resize_graph(ggml_backend_sched_t sched,
                                const SchedulerGraphCase & graph,
                                float bias,
                                bool async) {
    std::vector<float> input((size_t) graph.elements);
    std::vector<float> output((size_t) graph.elements, 0.0f);
    for (int64_t i = 0; i < graph.elements; ++i) {
        input[(size_t) i] = ((float) i + bias) / 256.0f;
    }
    ggml_backend_tensor_set(
        graph.input, input.data(), 0, ggml_nbytes(graph.input));
    const enum ggml_status status = async
        ? ggml_backend_sched_graph_compute_async(sched, graph.graph)
        : ggml_backend_sched_graph_compute(sched, graph.graph);
    if (status != GGML_STATUS_SUCCESS) return false;
    if (async) return true;

    ggml_backend_tensor_get(
        graph.output, output.data(), 0, ggml_nbytes(graph.output));
    for (int64_t i = 0; i < graph.elements; ++i) {
        if (std::fabs(output[(size_t) i] - 24.0f * input[(size_t) i]) >=
            1.0e-4f) {
            return false;
        }
    }
    return true;
}

bool run_async_reset_resize_and_free(ggml_backend_t first,
                                     ggml_backend_t second,
                                     const char * label) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    SchedulerGraphCase small = make_resize_graph(1024);
    SchedulerGraphCase large = make_resize_graph(32768);
    ggml_backend_t backends[] = { first, second, cpu };
    ggml_backend_sched_t sched = cpu
        ? ggml_backend_sched_new(backends, nullptr, 3, 64, false, true)
        : nullptr;
    bool ok = sched && small.ctx && large.ctx;
    if (ok) {
        ggml_backend_sched_set_batch_split_copies(sched, true);
        ok = pin_resize_graph(sched, small, first, second) &&
             set_and_check_resize_graph(
                 sched, small, 7.0f, /*async=*/true);
    }
    if (ok) {
        // Reset while the small graph is still in flight, then grow both the
        // graph allocation and staging arena. Allocation must quiesce old host
        // pages before replacing them.
        ggml_backend_sched_reset(sched);
        ok = pin_resize_graph(sched, large, first, second) &&
             set_and_check_resize_graph(
                 sched, large, 11.0f, /*async=*/false);
    }
    if (ok) {
        // Leave one final submission in flight. Scheduler teardown owns the
        // wait required before freeing graph buffers and staging pages.
        ok = set_and_check_resize_graph(
            sched, large, 19.0f, /*async=*/true);
    }
    if (sched) ggml_backend_sched_free(sched);
    if (cpu) ggml_backend_free(cpu);
    if (large.ctx) ggml_free(large.ctx);
    if (small.ctx) ggml_free(small.ctx);
    std::printf("mixed-backend %s async-reset-resize-free: %s\n",
                label, ok ? "ok" : "FAILED");
    return ok;
}

bool run_multi_source_graph(ggml_backend_t first,
                            ggml_backend_t second,
                            const char * label) {
    constexpr int64_t n = 8192;
    ggml_init_params params{};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!ctx || !cpu) {
        if (cpu) ggml_backend_free(cpu);
        if (ctx) ggml_free(ctx);
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_input(input);
    ggml_tensor * first_head = ggml_scale(ctx, input, 2.0f);
    ggml_tensor * cpu_head = ggml_scale(ctx, input, 5.0f);
    ggml_tensor * second_head = ggml_scale(ctx, input, 3.0f);
    ggml_tensor * second_join = ggml_add(ctx, first_head, cpu_head);
    ggml_tensor * output = ggml_add(ctx, second_join, second_head);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_t backends[] = { first, second, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 3, 64, false, true);
    bool ok = sched != nullptr;
    if (ok) {
        ggml_backend_sched_set_tensor_backend(sched, input, first);
        ggml_backend_sched_set_tensor_backend(sched, first_head, first);
        ggml_backend_sched_set_tensor_backend(sched, cpu_head, cpu);
        ggml_backend_sched_set_tensor_backend(sched, second_head, second);
        ggml_backend_sched_set_tensor_backend(sched, second_join, second);
        ggml_backend_sched_set_tensor_backend(sched, output, first);
        ggml_backend_sched_set_batch_split_copies(sched, true);
        ok = ggml_backend_sched_alloc_graph(sched, graph);
    }

    std::vector<float> host((size_t) n);
    std::vector<float> result((size_t) n, 0.0f);
    for (int iteration = 0; ok && iteration < 3; ++iteration) {
        for (int64_t i = 0; i < n; ++i) {
            host[(size_t) i] = ((float) i + 13.0f * iteration) / 128.0f;
        }
        ggml_backend_tensor_set(input, host.data(), 0, ggml_nbytes(input));
        ok = ggml_backend_sched_graph_compute(sched, graph) ==
             GGML_STATUS_SUCCESS;
        if (ok) {
            ggml_backend_tensor_get(
                output, result.data(), 0, ggml_nbytes(output));
        }
        for (int64_t i = 0; ok && i < n; ++i) {
            ok = std::fabs(result[(size_t) i] - 10.0f * host[(size_t) i]) <
                 1.0e-4f;
        }
    }
    std::printf("mixed-backend %s multi-source graph: %s\n",
                label, ok ? "ok" : "FAILED");
    if (sched) ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return ok;
}

}  // namespace

int main() {
    std::string error;
    bool ok = reject_invalid_compiled_device();
    ggml_backend_t cuda = init_placement_backend(PlacementBackend::Cuda, 0, &error);
    if (!cuda) {
        std::fprintf(stderr, "CUDA initialization failed: %s\n", error.c_str());
        return 1;
    }
    ggml_backend_t hip = init_placement_backend(PlacementBackend::Hip, 0, &error);
    if (!hip) {
        std::fprintf(stderr, "HIP initialization failed: %s\n", error.c_str());
        ggml_backend_free(cuda);
        return 1;
    }

    const auto pair = backend_pair_capabilities(cuda, hip);
    ok = placement_backend_of(cuda) == PlacementBackend::Cuda &&
         placement_backend_of(hip) == PlacementBackend::Hip &&
         !pair.same_runtime && !pair.native_gpu_handoff && ok;
    ok = run_scale(cuda, "CUDA") && ok;
    ok = run_scale(hip, "HIP") && ok;
    ok = run_cross_copy(cuda, hip, "CUDA->HIP") && ok;
    ok = run_cross_copy(hip, cuda, "HIP->CUDA") && ok;
    ok = run_cross_graph(cuda, hip, "CUDA->HIP->CUDA", false) && ok;
    ok = run_cross_graph(cuda, hip, "CUDA->HIP->CUDA", true) && ok;
    ok = run_cross_graph(hip, cuda, "HIP->CUDA->HIP", false) && ok;
    ok = run_cross_graph(hip, cuda, "HIP->CUDA->HIP", true) && ok;
    ok = run_noncontiguous_cross_graph(
        cuda, hip, "CUDA->HIP", false) && ok;
    ok = run_noncontiguous_cross_graph(
        cuda, hip, "CUDA->HIP", true) && ok;
    ok = run_noncontiguous_cross_graph(
        hip, cuda, "HIP->CUDA", false) && ok;
    ok = run_noncontiguous_cross_graph(
        hip, cuda, "HIP->CUDA", true) && ok;
    ok = run_multi_source_graph(cuda, hip, "CUDA+CPU->HIP") && ok;
    ok = run_multi_source_graph(hip, cuda, "HIP+CPU->CUDA") && ok;
    ok = run_async_reset_resize_and_free(
        cuda, hip, "CUDA->HIP->CUDA") && ok;
    ok = run_async_reset_resize_and_free(
        hip, cuda, "HIP->CUDA->HIP") && ok;

    ggml_backend_free(hip);
    ggml_backend_free(cuda);
    return ok ? 0 : 1;
}
