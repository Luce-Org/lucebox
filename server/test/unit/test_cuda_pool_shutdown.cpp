#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
struct CudaPoolShutdownFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;

    void exercise_pool_trim(
        ggml_backend_t backend,
        ggml_backend_t expected_pool_backend = nullptr);
};

ggml_backend_meta_split_state mirrored_split_state(
        const ggml_tensor *, void *) {
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, 1, {1}};
}

void CudaPoolShutdownFixture::exercise_pool_trim(
        ggml_backend_t backend,
        ggml_backend_t expected_pool_backend) {
    REQUIRE(backend != nullptr);

    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

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
    REQUIRE(buffer != nullptr);

    std::vector<uint8_t> weights_data(ggml_nbytes(weights), 0);
    std::vector<float> input_data((size_t) k, 1.0f);
    ggml_backend_tensor_set(weights, weights_data.data(), 0, weights_data.size());
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    auto compute = [&]() {
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        return status;
    };

    // Two stable evaluations warm and capture graph replay when enabled.
    REQUIRE(compute() == GGML_STATUS_SUCCESS);
    REQUIRE(compute() == GGML_STATUS_SUCCESS);

    // LUCE_Q8_MEMO intentionally retains a pool allocation after compute.
    // Trimming must first release that memo, then return its cached block to
    // the driver rather than retaining VRAM indefinitely.
    const bool expected_has_legacy_pool = ggml_backend_cuda_has_legacy_pool(
        expected_pool_backend ? expected_pool_backend : backend);
    const bool has_legacy_pool = ggml_backend_cuda_has_legacy_pool(backend);
    REQUIRE(has_legacy_pool == expected_has_legacy_pool);
    const size_t trimmed = ggml_backend_cuda_trim_pool(backend);
    if (expected_has_legacy_pool) {
        REQUIRE(trimmed > 0);
    } else {
        REQUIRE(trimmed == 0);
    }

    // A legacy trim must retire graph executables that captured the released
    // memo address. Recomputing the same graph must recapture rather than
    // replaying a stale pointer. VMM pools safely keep the existing replay.
    REQUIRE(compute() == GGML_STATUS_SUCCESS);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}
}

TEST_CASE(CudaPoolShutdownFixture, backend_pool_shutdown) {
    const luce_test::ScopedEnvVar q8_memo("LUCE_Q8_MEMO", "1");

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "skip: no CUDA/HIP backend available\n");
        return;
    }

    exercise_pool_trim(backend);
    ggml_backend_free(backend);
    REQUIRE_TRUE(true);
}

TEST_CASE(CudaPoolShutdownFixture, meta_backend_pool_shutdown) {
    const luce_test::ScopedEnvVar q8_memo("LUCE_Q8_MEMO", "1");

    ggml_backend_reg_t cuda_reg = ggml_backend_cuda_reg();
    if (!cuda_reg || ggml_backend_reg_dev_count(cuda_reg) == 0) {
        std::fprintf(stderr, "skip: no CUDA/HIP device available\n");
        return;
    }

    ggml_backend_dev_t device = ggml_backend_reg_dev_get(cuda_reg, 0);
    ggml_backend_dev_t meta_device = ggml_backend_meta_device(
        &device, 1, mirrored_split_state, nullptr);
    ggml_backend_t backend = meta_device
        ? ggml_backend_dev_init(meta_device, nullptr)
        : nullptr;
    if (!backend) {
        std::fprintf(stderr, "skip: no CUDA/HIP meta backend available\n");
        return;
    }

    ggml_backend_t rank_backend = ggml_backend_meta_simple_backend(backend, 0);
    REQUIRE(rank_backend != nullptr);
    exercise_pool_trim(backend, rank_backend);
    ggml_backend_free(backend);
    REQUIRE_TRUE(true);
}
