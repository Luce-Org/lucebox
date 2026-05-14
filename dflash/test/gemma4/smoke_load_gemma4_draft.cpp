// Smoke test: load Gemma4 DFlash draft weights from a safetensors directory.
//
// Usage: smoke_load_gemma4_draft <draft_dir>

#include "internal.h"
#include "gemma4.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace dflash27b;

static void fail(const char * msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <draft_dir>\n", argv[0]);
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "ggml_backend_cuda_init(0) failed\n");
        return 1;
    }
    std::printf("cuda backend: %s\n", ggml_backend_name(backend));

    GemmaDraftWeights dw;
    if (!load_gemma4_draft_safetensors(argv[1], backend, dw)) {
        std::fprintf(stderr, "load_gemma4_draft_safetensors failed: %s\n",
                     dflash27b_last_error());
        ggml_backend_free(backend);
        return 1;
    }

    // Print loaded metadata
    std::printf("n_layer=%d n_head=%d n_head_kv=%d head_dim=%d n_embd=%d n_ff=%d n_vocab=%d\n",
                dw.n_layer, dw.n_head, dw.n_head_kv, dw.head_dim,
                dw.n_embd, dw.n_ff, dw.n_vocab);
    std::printf("n_target_layers=%d target_hidden=%d logit_softcap=%.1f\n",
                dw.n_target_layers, dw.target_hidden, dw.logit_softcap);

    // Assert expected draft topology
    if (dw.n_layer != 5) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "n_layer=%d expected 5", dw.n_layer);
        fail(buf);
    }
    if (dw.n_vocab != 262144) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "n_vocab=%d expected 262144", dw.n_vocab);
        fail(buf);
    }
    if (!dw.fc) fail("fc is null");

    // Validate fc shape: ne[0] = 6*target_hidden (input features), ne[1] = draft_hidden (output)
    // In ggml convention: ne[0] is the fast (inner) dimension of matrix multiply,
    // so fc has ne[0]=6*target_hidden and ne[1]=draft_hidden.
    const int64_t expected_fc_ne0 = (int64_t)dw.n_target_layers * dw.target_hidden;
    std::printf("fc: ne=[%" PRId64 ", %" PRId64 "] type=%s\n",
                dw.fc->ne[0], dw.fc->ne[1],
                ggml_type_name(dw.fc->type));
    if (dw.fc->ne[0] != expected_fc_ne0) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "fc->ne[0]=%" PRId64 " expected %" PRId64 " (n_target_layers=%d * target_hidden=%d)",
            dw.fc->ne[0], expected_fc_ne0, dw.n_target_layers, dw.target_hidden);
        fail(buf);
    }

    // Assert layers vector size
    if ((int)dw.layers.size() != dw.n_layer) {
        char buf[64];
        std::snprintf(buf, sizeof(buf),
            "layers.size()=%zu expected %d", dw.layers.size(), dw.n_layer);
        fail(buf);
    }

    // Spot-check layer 0 key tensors
    if (!dw.layers[0].wq)     fail("layers[0].wq is null");
    if (!dw.layers[0].wk)     fail("layers[0].wk is null");
    if (!dw.layers[0].w_gate) fail("layers[0].w_gate is null");

    // Print layer 0 shape as spot check
    std::printf("layers[0].wq: ne=[%" PRId64 ", %" PRId64 "] type=%s\n",
                dw.layers[0].wq->ne[0], dw.layers[0].wq->ne[1],
                ggml_type_name(dw.layers[0].wq->type));

    // Validate hidden_norm and out_norm
    if (!dw.hidden_norm) fail("hidden_norm is null");
    if (!dw.out_norm)    fail("out_norm is null");
    // tok_embd is NOT loaded from the draft safetensors; it is injected at
    // runtime from the target model's token embedding table.
    if (dw.tok_embd) fail("tok_embd should be null after loading draft (shared with target)");

    std::printf("hidden_norm: ne[0]=%" PRId64 " type=%s\n",
                dw.hidden_norm->ne[0], ggml_type_name(dw.hidden_norm->type));

    free_gemma4_draft_weights(dw);
    ggml_backend_free(backend);
    std::printf("PASS\n");
    return 0;
}
