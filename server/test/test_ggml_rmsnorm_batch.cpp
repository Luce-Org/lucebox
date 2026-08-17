#include "CppUnitTestFramework.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace CppUnitTestFramework;

namespace {
struct GgmlRmsnormBatchFixture : CommonFixture {
    using CommonFixture::CommonFixture;
};
}

TEST_CASE(GgmlRmsnormBatchFixture, rmsnorm_batch) {
    constexpr int n_tokens = 64;
    constexpr bool initialize_peer = true;
    constexpr int device = 0;
    constexpr int n_embd = 4096;
    constexpr float eps = 1.0e-6f;

    ggml_backend_t backend = ggml_backend_cuda_init(device);
    if (!backend) {
        SKIP("CUDA/HIP backend unavailable");
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
    if (!ctx) {
        if (peer) ggml_backend_free(peer);
        ggml_backend_free(backend);
        REQUIRE(ctx != nullptr);
    }

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
        if (alloc) ggml_gallocr_free(alloc);
        ggml_free(ctx);
        if (peer) ggml_backend_free(peer);
        ggml_backend_free(backend);
        REQUIRE(alloc != nullptr);
        REQUIRE(false);
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
    }

    std::vector<float> output_data(input_data.size());
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(output, output_data.data(), 0,
                                output_data.size() * sizeof(float));
    }
    bool finite = status == GGML_STATUS_SUCCESS;
    for (float value : output_data) {
        if (!std::isfinite(value)) {
            std::fprintf(stderr, "non-finite output\n");
            finite = false;
            break;
        }
    }
    std::printf("PASS device=%d peer=%s tokens=%d embd=%d\n",
                device, peer ? "initialized" : "absent", n_tokens, n_embd);

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    if (peer) ggml_backend_free(peer);
    ggml_backend_free(backend);
    REQUIRE(finite);
}
