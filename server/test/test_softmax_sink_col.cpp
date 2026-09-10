// ggml_soft_max_ext_sink_col regression test.
//
// DeepSeek4 attention adds one learned per-head sink logit to the softmax
// denominator. The fused op treats the sink as a virtual last column with the
// query scale folded in, replacing scale -> concat(sink) -> soft_max -> view.
// This test compares both forms bit for bit on the HIP and CPU backends over
// column counts across block-size steps and the shared-memory fallback boundary.
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

uint32_t lcg_state = 0x12345678u;
float lcg_uniform() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (float) ((lcg_state >> 8) & 0xFFFFFF) / (float) 0x1000000 * 2.0f - 1.0f;
}

bool run(ggml_backend_t backend, bool fused, int ncols, int rows, float scale,
         const std::vector<float> & scores, const std::vector<float> & sinks, std::vector<float> & out) {
    ggml_init_params params{};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    ggml_tensor * st = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, rows);
    ggml_tensor * kt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rows);
    ggml_set_input(st);
    ggml_set_input(kt);
    ggml_tensor * probs = nullptr;
    if (fused) {
        probs = ggml_soft_max_ext_sink_col(ctx, st, kt, scale);
    } else {
        ggml_tensor * scaled = ggml_scale(ctx, st, scale);
        ggml_tensor * sink2d = ggml_reshape_2d(ctx, kt, 1, rows);
        ggml_tensor * with_sink = ggml_concat(ctx, scaled, sink2d, 0);
        ggml_tensor * full = ggml_soft_max(ctx, with_sink);
        // The old path views the first ncols columns; materialize the view so
        // both forms hand back a contiguous [ncols, rows] tensor.
        probs = ggml_cont(ctx, ggml_view_2d(ctx, full, ncols, rows, full->nb[1], 0));
    }
    ggml_set_output(probs);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, probs);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(st, scores.data(), 0, scores.size() * sizeof(float));
    ggml_backend_tensor_set(kt, sinks.data(), 0, sinks.size() * sizeof(float));
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(backend);
    if (ok) {
        out.resize((size_t) ncols * rows);
        ggml_backend_tensor_get(probs, out.data(), 0, out.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

size_t bit_mismatches(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return a.size() + b.size();
    }
    size_t mm = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            ++mm;
        }
    }
    return mm;
}

} // namespace

int main() {
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) {
        std::fprintf(stderr, "failed to query HIP device 0\n");
        return 1;
    }
    ggml_backend_t hip = ggml_backend_cuda_init(0);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!hip || !cpu) {
        std::fprintf(stderr, "failed to initialize backends\n");
        return 1;
    }
    const bool previous_graphs = ggml_backend_cuda_set_graphs_disabled_override(true);
    const int rows = 64;                       // n_head
    const float scale = 1.0f / 22.627417f;     // 1/sqrt(512)
    bool ok = true;
    std::vector<int> widths = {1, 20, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200, 255, 256, 257, 511, 512, 513, 700, 1023, 1024, 1025, 1500, 2047, 2048, 3000};
    // Shared scratch includes one warp of reduction storage and the virtual
    // sink column. Cross its capacity to exercise output-backed scratch, where
    // caching the virtual column would overwrite the next row (or the buffer).
    const int shared_cols = (int) (properties.sharedMemPerBlock / sizeof(float)) - properties.warpSize;
    widths.insert(widths.end(), {shared_cols - 1, shared_cols, shared_cols + 1, 2 * shared_cols + 1});
    for (int ncols : widths) {
        std::vector<float> scores((size_t) ncols * rows), sinks(rows);
        lcg_state = 0xabc123u ^ (uint32_t) ncols;
        for (float & v : scores) {
            v = lcg_uniform() * 40.0f;   // pre-scale logits of realistic magnitude
        }
        for (float & v : sinks) {
            v = lcg_uniform() * 3.0f;
        }
        // Include sink-dominated and negligible-sink rows so large widths still
        // expose an omitted sink or an incorrect per-row sink index.
        sinks[0] = 80.0f;
        sinks[1] = -80.0f;
        sinks[2] = 0.0f;
        for (auto backend : {hip, cpu}) {
            const char * name = backend == hip ? "hip" : "cpu";
            std::vector<float> ref, fused;
            if (!run(backend, false, ncols, rows, scale, scores, sinks, ref) ||
                !run(backend, true, ncols, rows, scale, scores, sinks, fused)) {
                std::printf("FAIL %s ncols=%d: compute failed\n", name, ncols);
                ok = false;
                continue;
            }
            const size_t mm = bit_mismatches(ref, fused);
            std::printf("%s %s ncols=%d: %zu values, %zu bit mismatches vs scale-concat-softmax\n",
                        mm == 0 ? "PASS" : "FAIL", name, ncols, ref.size(), mm);
            ok = ok && mm == 0;
        }
    }
    ggml_backend_cuda_set_graphs_disabled_override(previous_graphs);
    ggml_backend_free(hip);
    ggml_backend_free(cpu);
    std::printf("%s\n", ok ? "PASS softmax sink column" : "FAIL softmax sink column");
    return ok ? 0 : 1;
}
