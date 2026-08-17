#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "CppUnitTestFramework.hpp"
using CppUnitTestFramework::CommonFixture;

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static bool parse_test_devices(const char * value, int & first, int & second) {
    char trailing = '\0';
    return value != nullptr &&
        std::sscanf(value, "%d,%d%c", &first, &second, &trailing) == 2;
}

static bool run_allreduce_test(int first, int second) {
    const int device_count = ggml_backend_cuda_get_device_count();
    if (first < 0 || second < 0 || first >= device_count ||
        second >= device_count || first == second) {
        std::fprintf(stderr,
            "DFLASH_TP_TEST_DEVICES must name two distinct visible CUDA devices "
            "(got %d,%d; visible=%d)\n",
            first, second, device_count);
        return false;
    }

    ggml_backend_t backends[2] = {
        ggml_backend_cuda_init(first),
        ggml_backend_cuda_init(second),
    };
    ggml_context * contexts[2] = {nullptr, nullptr};
    ggml_backend_buffer_t buffers[2] = {nullptr, nullptr};
    ggml_tensor * tensors[2] = {nullptr, nullptr};

    auto cleanup = [&]() {
        for (int i = 0; i < 2; ++i) {
            if (buffers[i]) ggml_backend_buffer_free(buffers[i]);
            if (contexts[i]) ggml_free(contexts[i]);
            if (backends[i]) ggml_backend_free(backends[i]);
        }
    };

    if (!backends[0] || !backends[1]) {
        std::fputs("failed to initialize selected CUDA backends\n", stderr);
        cleanup();
        return false;
    }

    const std::array<std::array<float, 4>, 2> input = {{
        {{1.0f, 2.0f, 3.0f, 4.0f}},
        {{10.0f, 20.0f, 30.0f, 40.0f}},
    }};
    const std::array<float, 4> expected = {{11.0f, 22.0f, 33.0f, 44.0f}};

    for (int i = 0; i < 2; ++i) {
        ggml_init_params params{};
        params.mem_size = 16 * 1024;
        params.no_alloc = true;
        contexts[i] = ggml_init(params);
        if (!contexts[i]) {
            std::fputs("failed to initialize GGML test context\n", stderr);
            cleanup();
            return false;
        }
        tensors[i] = ggml_new_tensor_1d(contexts[i], GGML_TYPE_F32, 4);
        tensors[i]->flags |= GGML_TENSOR_FLAG_COMPUTE;
        buffers[i] = ggml_backend_alloc_ctx_tensors(contexts[i], backends[i]);
        if (!buffers[i]) {
            std::fputs("failed to allocate CUDA all-reduce tensor\n", stderr);
            cleanup();
            return false;
        }
        ggml_backend_tensor_set(
            tensors[i], input[i].data(), 0, sizeof(input[i]));
    }

    if (!ggml_backend_cuda_allreduce_tensor(backends, tensors, 2)) {
        std::fputs(
            "selected-device all-reduce failed (ensure this build has NCCL)\n",
            stderr);
        cleanup();
        return false;
    }

    bool ok = true;
    for (int i = 0; i < 2; ++i) {
        ggml_backend_synchronize(backends[i]);
        std::array<float, 4> output{};
        ggml_backend_tensor_get(tensors[i], output.data(), 0, sizeof(output));
        for (size_t j = 0; j < output.size(); ++j) {
            if (std::fabs(output[j] - expected[j]) > 1e-6f) {
                std::fprintf(stderr,
                    "all-reduce mismatch on CUDA%d[%zu]: got %.8g expected %.8g\n",
                    i == 0 ? first : second, j, output[j], expected[j]);
                ok = false;
            }
        }
    }
    cleanup();
    return ok;
}

namespace {
struct CudaCommApiFixture : CommonFixture {
    using CommonFixture::CommonFixture;
};
}

TEST_CASE(CudaCommApiFixture, cuda_communicator_api) {
    if (ggml_backend_cuda_get_device_count() == 0) {
        SKIP("CUDA device unavailable");
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fputs("failed to initialize CUDA backend 0\n", stderr);
        REQUIRE_TRUE(false);
    }

    // Calling the public wrapper keeps its declaration and exported symbol
    // covered. Passing one CUDA device twice also exercises the duplicate-rank
    // guard without entering NCCL communicator creation, where this topology
    // previously hung.
    ggml_backend_t backends[] = {backend, backend};
    ggml_tensor * tensors[] = {nullptr, nullptr};
    const bool result =
        ggml_backend_cuda_allreduce_tensor(backends, tensors, 2);
    ggml_backend_free(backend);

    if (result) {
        std::fputs("duplicate CUDA devices unexpectedly initialized a communicator\n", stderr);
        REQUIRE_TRUE(false);
    }

    std::puts("CUDA communicator API compatibility test passed");
}

TEST_CASE(CudaCommApiFixture, selected_device_nccl_allreduce) {
    if (ggml_backend_cuda_get_device_count() == 0) {
        SKIP("CUDA device unavailable");
    }

    const char * selected = std::getenv("DFLASH_TP_TEST_DEVICES");
    if (!selected || !selected[0]) {
        SKIP("DFLASH_TP_TEST_DEVICES is unset");
    }

    int first = -1;
    int second = -1;
    if (!parse_test_devices(selected, first, second)) {
        std::fprintf(stderr,
            "bad DFLASH_TP_TEST_DEVICES=%s (expected e.g. 1,2)\n", selected);
        REQUIRE_TRUE(false);
    }
    REQUIRE_TRUE(run_allreduce_test(first, second));

    std::printf("selected-device NCCL all-reduce passed on CUDA%d,CUDA%d\n",
                first, second);
}
