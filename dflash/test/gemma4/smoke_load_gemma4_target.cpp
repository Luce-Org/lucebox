// Smoke test: load a Gemma4 target GGUF, validate metadata and tensor shapes.
//
// Usage: smoke_load_gemma4_target <gemma4.gguf>

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
        std::fprintf(stderr, "usage: %s <gemma4.gguf>\n", argv[0]);
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    GemmaTargetWeights w;
    if (!load_gemma4_target_gguf(argv[1], backend, w)) {
        std::fprintf(stderr, "load_gemma4_target_gguf failed: %s\n", dflash27b_last_error());
        ggml_backend_free(backend);
        return 1;
    }

    // Print architecture metadata
    std::printf("hparams: n_layer=%d n_embd=%d n_head=%d n_head_kv=%d head_dim=%d "
                "n_vocab=%d n_ff=%d\n",
                w.n_layer, w.n_embd, w.n_head, w.n_head_kv, w.head_dim,
                w.n_vocab, w.n_ff);

    // Count SWA vs full-attention layers
    int n_swa = 0, n_full = 0;
    for (int il = 0; il < w.n_layer; il++) {
        if (il < (int)w.swa_layers.size() && w.swa_layers[il]) n_swa++;
        else n_full++;
    }
    std::printf("swa_layers: swa=%d full=%d (total=%d)\n", n_swa, n_full, w.n_layer);

    // Print KV-sharing config
    std::printf("kv_sharing: n_kv_shared_layers=%d n_layer_kv=%d\n",
                w.n_kv_shared_layers, w.n_layer_kv);

    // Print Per-Layer Embedding dimension
    std::printf("n_embd_per_layer=%d\n", w.n_embd_per_layer);

    // Print MoE config (0 for dense)
    std::printf("moe: n_expert=%d n_expert_used=%d\n", w.n_expert, w.n_expert_used);

    // Print attention config
    std::printf("logit_softcap=%.2f attn_scale=%.4f rope_theta=%.0f\n",
                w.logit_softcap, w.attn_scale, w.rope_theta);

    // Print captured layer IDs for the DFlash draft
    std::printf("capture_layer_ids:");
    for (int i = 0; i < w.n_capture_layers; i++) {
        std::printf(" %d", w.capture_layer_ids[i]);
    }
    std::printf("\n");

    // Assertions
    if (w.n_vocab != 262144) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "n_vocab=%d expected 262144", w.n_vocab);
        fail(buf);
    }
    if (w.logit_softcap != 30.0f) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "logit_softcap=%.2f expected 30.0", w.logit_softcap);
        fail(buf);
    }
    if (w.n_layer_kv <= 0) {
        fail("n_layer_kv must be > 0");
    }
    if (w.n_layer_kv > w.n_layer) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "n_layer_kv=%d > n_layer=%d", w.n_layer_kv, w.n_layer);
        fail(buf);
    }

    // Spot-check layer 0 tensors
    if (!w.layers[0].wq) fail("layers[0].wq is null");
    if (!w.layers[0].wo) fail("layers[0].wo is null");
    if (!w.layers[0].w_gate) fail("layers[0].w_gate is null");

    // Spot-check tok_embd and output
    if (!w.tok_embd) fail("tok_embd is null");
    if (!w.output)   fail("output (lm_head) is null");
    if (!w.out_norm) fail("out_norm is null");

    std::printf("tok_embd: ne=[%" PRId64 ", %" PRId64 "] type=%s nbytes=%.2f MiB\n",
                w.tok_embd->ne[0], w.tok_embd->ne[1],
                ggml_type_name(w.tok_embd->type),
                ggml_nbytes(w.tok_embd) / (1024.0 * 1024.0));

    free_gemma4_target_weights(w);
    ggml_backend_free(backend);
    std::printf("PASS\n");
    return 0;
}
