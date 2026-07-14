#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
#if defined(_WIN32)
    _putenv_s("LUCE_Q8_MEMO", "1");
#else
    setenv("LUCE_Q8_MEMO", "1", 1);
#endif

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "failed to initialize CUDA/HIP backend\n");
        return 1;
    }

    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return 1;
    }

    constexpr int64_t k = 256;
    constexpr int64_t n = 256;
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k, n);
    ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
    ggml_tensor * output  = ggml_mul_mat(ctx, weights, input);
    ggml_set_input(input);
    ggml_set_output(output);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<uint8_t> weights_data(ggml_nbytes(weights), 0);
    std::vector<float> input_data((size_t) k, 1.0f);
    ggml_backend_tensor_set(weights, weights_data.data(), 0, weights_data.size());
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_free(backend);
        return 1;
    }

    // LUCE_Q8_MEMO intentionally retains the pool allocation after compute.
    // Backend teardown must release that allocation before destroying its pool.
    ggml_backend_free(backend);
    return 0;
}
