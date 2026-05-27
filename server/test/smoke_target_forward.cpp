// Smoke test for the qwen35 target forward graph.
//
// Loads Qwen3.5-27B from GGUF, creates a target cache, builds the forward
// graph for a single-token decode, runs it on CUDA, and prints the top-k
// token IDs from the resulting logits distribution.
//
// Does NOT validate numerics — we just want the graph to COMPUTE without
// asserting. Numerics matching vs. llama.cpp is follow-up work.
//
// Usage: smoke_target_forward <qwen35.gguf>

#include "dflash27b.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace dflash::common;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen35.gguf>\n", argv[0]);
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    // Load target weights
    TargetWeights w;
    if (!load_target_gguf(argv[1], backend, w)) {
        std::fprintf(stderr, "load_target_gguf: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[target] %s\n", dflash27b_last_error());

    // Create target state cache
    TargetCache cache;
    const int max_ctx = 64;
    if (!create_target_cache(w, max_ctx, /*max_verify_tokens=*/0, backend, cache)) {
        std::fprintf(stderr, "create_target_cache: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[cache] attn_k=%zu attn_v=%zu ssm=%zu conv=%zu\n",
        cache.attn_k.size(), cache.attn_v.size(),
        cache.ssm_state.size(), cache.conv_state.size());

    // Graph context
    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) { std::fprintf(stderr, "ggml_init graph failed\n"); return 1; }

    // Input tensors: one token (placeholder id=1). The embedding is computed
    // on CPU and uploaded via inp_embed, so tok_embd never lives on GPU.
    // M-RoPE needs 4 position values per token (one per axis).
    const int n_tokens = 1;
    const int hidden   = w.n_embd;
    ggml_tensor * inp_embed = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(inp_embed, "inp_embed");
    ggml_set_name(positions, "positions");
    ggml_set_input(inp_embed);
    ggml_set_input(positions);

    QwenGraphInputs gi{};
    gi.inp_embed = inp_embed;
    gi.positions = positions;
    gi.n_tokens  = n_tokens;
    gi.kv_start  = 0;

    // Pre-sized graph: 64 layers × ~60 nodes + overhead = ~4096
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 8192, false);

    QwenGraphOutputs go = build_qwen35_graph(gctx, gf, w, cache, gi);
    if (!go.logits) { std::fprintf(stderr, "build_qwen35_graph returned null\n"); return 1; }
    ggml_set_output(go.logits);
    ggml_build_forward_expand(gf, go.logits);
    std::printf("[graph] nodes=%d\n", ggml_graph_n_nodes(gf));

    // Allocate graph
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "ggml_gallocr_alloc_graph failed\n");
        return 1;
    }

    // Fill inputs: embed token id=1 on CPU, upload to GPU
    int32_t tok_ids[1] = { 1 };
    std::vector<float> embed_buf(hidden * n_tokens);
    if (!w.embedder.embed(tok_ids, n_tokens, embed_buf.data())) {
        std::fprintf(stderr, "cpu embedder failed\n");
        return 1;
    }
    ggml_backend_tensor_set(inp_embed, embed_buf.data(), 0, sizeof(float) * embed_buf.size());

    int32_t pos4[4] = { 0, 0, 0, 0 };
    ggml_backend_tensor_set(positions, pos4, 0, sizeof(int32_t) * 4);

    // Compute
    auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed: %d\n", (int)status);
        return 1;
    }
    std::printf("[compute] OK\n");

    // Read logits
    const int64_t vocab = w.n_vocab;
    std::vector<float> logits(vocab);
    ggml_backend_tensor_get(go.logits, logits.data(), 0, sizeof(float) * vocab);

    // Quick stats
    int n_nan = 0, n_inf = 0;
    float vmin = 1e30f, vmax = -1e30f;
    for (auto v : logits) {
        if (std::isnan(v)) n_nan++;
        else if (std::isinf(v)) n_inf++;
        else { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
    }
    std::printf("[logits] vocab=%" PRId64 " nan=%d inf=%d min=%.4g max=%.4g\n",
        vocab, n_nan, n_inf, vmin, vmax);

    // Top 5 token IDs
    std::vector<std::pair<float, int>> sorted;
    sorted.reserve(vocab);
    for (int i = 0; i < vocab; i++) sorted.emplace_back(logits[i], i);
    std::partial_sort(sorted.begin(), sorted.begin() + 5, sorted.end(),
        [](const auto & a, const auto & b) { return a.first > b.first; });
    std::printf("[top 5] ");
    for (int i = 0; i < 5; i++) {
        std::printf("id=%d l=%.3f  ", sorted[i].second, sorted[i].first);
    }
    std::printf("\n");

    ggml_gallocr_free(alloc);
    ggml_free(gctx);
    free_target_cache(cache);
    free_target_weights(w);
    ggml_backend_free(backend);
    std::printf("OK\n");
    return 0;
}
