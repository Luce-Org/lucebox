// Smoke test: load the z-lab DFlash draft safetensors into a CUDA ggml
// context. Prints tensor count, total bytes, and a checksum-ish spot check
// on one tensor. Exit 0 on success, nonzero on any failure.
//
// Usage: smoke_load_draft <path/to/model.safetensors>

#include "dflash27b.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace dflash::common;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.safetensors>\n", argv[0]);
        return 2;
    }
    const char * path = argv[1];

    // Initialize CUDA backend
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "ggml_backend_cuda_init(0) failed\n");
        return 1;
    }
    std::printf("cuda backend: %s\n", ggml_backend_name(backend));

    DraftWeights w;
    if (!load_draft_safetensors(path, backend, w)) {
        std::fprintf(stderr, "load_draft_safetensors failed: %s\n",
                     dflash27b_last_error());
        ggml_backend_free(backend);
        return 1;
    }

    // Count tensors and total bytes
    size_t n_tensors = 0;
    size_t total_bytes = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(w.ctx); t != nullptr;
         t = ggml_get_next_tensor(w.ctx, t)) {
        n_tensors++;
        total_bytes += ggml_nbytes(t);
    }
    std::printf("loaded %zu tensors, total %.2f GiB\n",
                n_tensors, total_bytes / (1024.0 * 1024.0 * 1024.0));

    // Spot check: the `fc` tensor should be [25600 -> 5120] (in ggml ne[0]=25600 ne[1]=5120).
    std::printf("fc: ne=[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] type=%s nbytes=%zu\n",
                w.fc->ne[0], w.fc->ne[1], w.fc->ne[2], w.fc->ne[3],
                ggml_type_name(w.fc->type), ggml_nbytes(w.fc));
    std::printf("hidden_norm: ne[0]=%" PRId64 " type=%s\n",
                w.hidden_norm->ne[0], ggml_type_name(w.hidden_norm->type));
    std::printf("layers[0].wq: ne=[%" PRId64 ", %" PRId64 "] type=%s\n",
                w.layers[0].wq->ne[0], w.layers[0].wq->ne[1],
                ggml_type_name(w.layers[0].wq->type));

    // Pull a few bytes of `norm.weight` back from CUDA and print a few values,
    // as a proof of end-to-end data path (file → mmap → CUDA → host).
    std::vector<uint16_t> hn(w.hidden_norm->ne[0]);
    ggml_backend_tensor_get(w.hidden_norm, hn.data(), 0, sizeof(uint16_t) * hn.size());
    auto bf16_to_f32 = [](uint16_t u) {
        uint32_t bits = ((uint32_t)u) << 16;
        float f;
        std::memcpy(&f, &bits, 4);
        return f;
    };
    std::printf("hidden_norm.weight first 8 values: ");
    for (int i = 0; i < 8 && i < (int)hn.size(); i++) {
        std::printf("%.4f ", bf16_to_f32(hn[i]));
    }
    std::printf("\n");

    free_draft_weights(w);
    ggml_backend_free(backend);
    std::printf("OK\n");
    return 0;
}
