#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    const int n_tokens = argc > 1 ? std::max(1, std::atoi(argv[1])) : 64;
    const bool initialize_peer = argc <= 2 || std::atoi(argv[2]) != 0;
    const int device = argc > 3 ? std::max(0, std::atoi(argv[3])) : 0;
    constexpr int n_embd = 4096;
    constexpr float eps = 1.0e-6f;

    ggml_backend_t backend = ggml_backend_cuda_init(device);
    if (!backend) {
        std::fprintf(stderr, "failed to initialize GPU %d\n", device);
        return 2;
    }

    // Leave a second HIP device initialized, matching the heterogeneous
    // server's process state. The backend must still execute this graph on 0.
    ggml_backend_t peer = nullptr;
    if (initialize_peer && ggml_backend_cuda_get_device_count() > 1) {
        peer = ggml_backend_cuda_init(device == 0 ? 1 : 0);
    }

    ggml_init_params params{};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return 3;

    ggml_tensor * input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * weight = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F32, n_embd);
    ggml_set_input(input);
    ggml_set_input(weight);
    ggml_tensor * output = ggml_mul(
        ctx, ggml_rms_norm(ctx, input, eps), weight);
    ggml_set_output(output);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph)) {
        std::fprintf(stderr, "graph allocation failed\n");
        return 4;
    }

    std::vector<float> input_data((size_t)n_embd * n_tokens);
    std::vector<float> weight_data(n_embd);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = ((int)(i % 31) - 15) * 0.01f;
    }
    for (int i = 0; i < n_embd; ++i) {
        weight_data[(size_t)i] = 0.75f + (i % 17) * 0.01f;
    }
    ggml_backend_tensor_set(input, input_data.data(), 0,
                            input_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0,
                            weight_data.size() * sizeof(float));

    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed: %d\n", (int)status);
        return 5;
    }

    std::vector<float> output_data(input_data.size());
    ggml_backend_tensor_get(output, output_data.data(), 0,
                            output_data.size() * sizeof(float));
    for (float value : output_data) {
        if (!std::isfinite(value)) {
            std::fprintf(stderr, "non-finite output\n");
            return 6;
        }
    }
    std::printf("PASS device=%d peer=%s tokens=%d embd=%d\n",
                device, peer ? "initialized" : "absent", n_tokens, n_embd);

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    if (peer) ggml_backend_free(peer);
    ggml_backend_free(backend);
    return 0;
}
