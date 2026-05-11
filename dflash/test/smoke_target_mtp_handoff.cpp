// Smoke test for the DFlash target -> native MTP handoff.
//
// Builds one graph containing:
//   target forward with expose_pre_norm_hidden=true
//   MTP/NextN forward fed by that target_pre_norm_hidden tensor
//
// This validates the C++ tensor handoff required by the real speculative loop.
//
// Usage: smoke_target_mtp_handoff <qwen35-mtp.gguf> [cuda_gpu]

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

static int check_logits(ggml_tensor * logits, const char * label) {
    const int64_t vocab = logits->ne[0];
    std::vector<float> buf((size_t)vocab);
    ggml_backend_tensor_get(logits, buf.data(), 0, sizeof(float) * buf.size());
    int n_nan = 0, n_inf = 0;
    float vmin = 1e30f, vmax = -1e30f;
    for (float v : buf) {
        if (std::isnan(v)) n_nan++;
        else if (std::isinf(v)) n_inf++;
        else {
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
    }
    std::printf("[%s] vocab=%lld nan=%d inf=%d min=%.4g max=%.4g\n",
        label, (long long)vocab, n_nan, n_inf, vmin, vmax);
    return (n_nan == 0 && n_inf == 0) ? 0 : 1;
}

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

    TargetCache target_cache;
    if (!create_target_cache(w, /*max_ctx=*/64, /*max_verify_tokens=*/0, backend, target_cache)) {
        std::fprintf(stderr, "create_target_cache: %s\n", dflash27b_last_error());
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }
    TargetMtpCache mtp_cache;
    if (!create_target_mtp_cache(w, /*max_ctx=*/64, backend, mtp_cache)) {
        std::fprintf(stderr, "create_target_mtp_cache: %s\n", dflash27b_last_error());
        free_target_cache(target_cache);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_init_params ip{};
    ip.mem_size   = 768 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) {
        std::fprintf(stderr, "ggml_init graph failed\n");
        free_target_mtp_cache(mtp_cache);
        free_target_cache(target_cache);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    const int n_tokens = 1;
    ggml_tensor * inp_embed = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, w.n_embd, n_tokens, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(inp_embed, "inp_embed");
    ggml_set_name(positions, "positions");
    ggml_set_input(inp_embed);
    ggml_set_input(positions);

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 8192, false);
    QwenGraphInputs target_in{};
    target_in.inp_embed = inp_embed;
    target_in.positions = positions;
    target_in.n_tokens = n_tokens;
    target_in.kv_start = 0;
    target_in.expose_pre_norm_hidden = true;

    QwenGraphOutputs target_out = build_qwen35_graph(gctx, gf, w, target_cache, target_in);
    if (!target_out.logits || !target_out.pre_norm_hidden) {
        std::fprintf(stderr, "build_qwen35_graph did not expose target hidden: %s\n", dflash27b_last_error());
        ggml_free(gctx);
        free_target_mtp_cache(mtp_cache);
        free_target_cache(target_cache);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    QwenMtpGraphInputs mtp_in{};
    mtp_in.token_embed = ggml_reshape_2d(gctx, inp_embed, w.n_embd, n_tokens);
    mtp_in.pre_norm_hidden = target_out.pre_norm_hidden;
    mtp_in.positions = positions;
    mtp_in.n_tokens = n_tokens;
    mtp_in.kv_start = 0;

    QwenMtpGraphOutputs mtp_out = build_qwen35_mtp_graph(gctx, gf, w, mtp_cache, mtp_in);
    if (!mtp_out.logits || !mtp_out.hidden) {
        std::fprintf(stderr, "build_qwen35_mtp_graph failed: %s\n", dflash27b_last_error());
        ggml_free(gctx);
        free_target_mtp_cache(mtp_cache);
        free_target_cache(target_cache);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }
    ggml_set_output(target_out.logits);
    ggml_set_output(mtp_out.logits);
    ggml_build_forward_expand(gf, mtp_out.logits);
    std::printf("[graph] nodes=%d\n", ggml_graph_n_nodes(gf));

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "ggml_gallocr_alloc_graph failed\n");
        ggml_free(gctx);
        free_target_mtp_cache(mtp_cache);
        free_target_cache(target_cache);
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
        free_target_cache(target_cache);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }
    int32_t pos4[4] = { 0, 0, 0, 0 };
    ggml_backend_tensor_set(inp_embed, embed_buf.data(), 0, sizeof(float) * embed_buf.size());
    ggml_backend_tensor_set(positions, pos4, 0, sizeof(pos4));

    auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed: %d\n", (int)status);
        ggml_gallocr_free(alloc);
        ggml_free(gctx);
        free_target_mtp_cache(mtp_cache);
        free_target_cache(target_cache);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    const int bad_target = check_logits(target_out.logits, "target-logits");
    const int bad_mtp = check_logits(mtp_out.logits, "mtp-logits");

    ggml_gallocr_free(alloc);
    ggml_free(gctx);
    free_target_mtp_cache(mtp_cache);
    free_target_cache(target_cache);
    free_target_weights(w);
    ggml_backend_free(backend);
    std::printf("OK\n");
    return (bad_target || bad_mtp) ? 1 : 0;
}
