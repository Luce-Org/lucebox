#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>

int main() {
    if (ggml_backend_cuda_get_device_count() == 0) {
        std::puts("CUDA device unavailable; skipping communicator API test");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fputs("failed to initialize CUDA backend 0\n", stderr);
        return 1;
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
        return 1;
    }

    std::puts("CUDA communicator API compatibility test passed");
    return 0;
}
