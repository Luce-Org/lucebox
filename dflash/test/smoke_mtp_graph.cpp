// Smoke test for the native Qwen35 NextN/MTP graph.
//
// Loads a GGUF with embedded nextn tensors, creates the MTP KV cache, runs a
// single-token MTP forward from caller-provided token embedding + synthetic
// target pre-norm hidden, and checks the resulting logits for NaN/Inf.
//
// Usage: smoke_mtp_graph <qwen35-mtp.gguf> [cuda_gpu]

#include "dflash27b.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace dflash27b;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen35-mtp.gguf>\n", argv[0]);
        return 2;
    }

    int gpu = 0;
    if (argc >= 3) gpu = std::atoi(argv[2]);

    ggml_backend_t backend = ggml_backend_cuda_init(gpu);
    if (!backend) {
        std::fprintf(stderr, "cuda init failed\n");
        return 1;
    }

    TargetWeights w;
    if (!load_target_gguf(argv[1], backend, w)) {
        std::fprintf(stderr, "load_target_gguf: %s\n", dflash27b_last_error());
        ggml_backend_free(backend);
        return 1;
    }
    std::printf("[target] %s\n", dflash27b_last_error());
    if (w.mtp_layers.empty()) {
        std::fprintf(stderr, "model has no MTP/nextn layers\n");
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    TargetMtpCache mtp_cache;
    if (!create_target_mtp_cache(w, /*max_ctx=*/64, backend, mtp_cache)) {
        std::fprintf(stderr, "create_target_mtp_cache: %s\n", dflash27b_last_error());
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }
    std::printf("[mtp-cache] layers=%zu max_ctx=%d kv=%s/%s\n",
        mtp_cache.attn_k.size(), mtp_cache.max_ctx,
        ggml_type_name(mtp_cache.kv_k_type), ggml_type_name(mtp_cache.kv_v_type));

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) {
        std::fprintf(stderr, "ggml_init graph failed\n");
        free_target_mtp_cache(mtp_cache);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    const int n_tokens = 1;
    ggml_tensor * token_embed = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_tensor * hidden      = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_tensor * positions   = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(token_embed, "mtp_token_embed");
    ggml_set_name(hidden, "target_pre_norm_hidden");
    ggml_set_name(positions, "positions");
    ggml_set_input(token_embed);
    ggml_set_input(hidden);
    ggml_set_input(positions);

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 2048, false);
    QwenMtpGraphInputs gi{};
    gi.token_embed = token_embed;
    gi.pre_norm_hidden = hidden;
    gi.positions = positions;
    gi.n_tokens = n_tokens;
    gi.kv_start = 0;

    QwenMtpGraphOutputs go = build_qwen35_mtp_graph(gctx, gf, w, mtp_cache, gi);
    if (!go.logits) {
        std::fprintf(stderr, "build_qwen35_mtp_graph: %s\n", dflash27b_last_error());
        ggml_free(gctx);
        free_target_mtp_cache(mtp_cache);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }
    ggml_set_output(go.logits);
    ggml_build_forward_expand(gf, go.logits);
    std::printf("[graph] nodes=%d\n", ggml_graph_n_nodes(gf));

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "ggml_gallocr_alloc_graph failed\n");
        ggml_free(gctx);
        free_target_mtp_cache(mtp_cache);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    int32_t tok_ids[1] = { 1 };
    std::vector<float> embed_buf((size_t)w.n_embd * n_tokens);
    if (!w.embedder.embed(tok_ids, n_tokens, embed_buf.data())) {
        std::fprintf(stderr, "cpu embedder failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(gctx);
        free_target_mtp_cache(mtp_cache);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }
    std::vector<float> hidden_buf((size_t)w.n_embd * n_tokens, 0.0f);
    int32_t pos4[4] = { 0, 0, 0, 0 };
    ggml_backend_tensor_set(token_embed, embed_buf.data(), 0, sizeof(float) * embed_buf.size());
    ggml_backend_tensor_set(hidden, hidden_buf.data(), 0, sizeof(float) * hidden_buf.size());
    ggml_backend_tensor_set(positions, pos4, 0, sizeof(pos4));

    auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed: %d\n", (int)status);
        ggml_gallocr_free(alloc);
        ggml_free(gctx);
        free_target_mtp_cache(mtp_cache);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    const int64_t vocab = go.logits->ne[0];
    std::vector<float> logits((size_t)vocab);
    ggml_backend_tensor_get(go.logits, logits.data(), 0, sizeof(float) * logits.size());

    int n_nan = 0, n_inf = 0;
    float vmin = 1e30f, vmax = -1e30f;
    for (float v : logits) {
        if (std::isnan(v)) n_nan++;
        else if (std::isinf(v)) n_inf++;
        else {
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
    }
    std::printf("[mtp-logits] vocab=%lld nan=%d inf=%d min=%.4g max=%.4g\n",
        (long long)vocab, n_nan, n_inf, vmin, vmax);

    ggml_gallocr_free(alloc);
    ggml_free(gctx);
    free_target_mtp_cache(mtp_cache);
    free_target_weights(w);
    ggml_backend_free(backend);
    std::printf("OK\n");
    return (n_nan == 0 && n_inf == 0) ? 0 : 1;
}
