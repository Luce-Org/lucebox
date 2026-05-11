// Minimal integrated DFlash + native MTP decode smoke.
//
// This is not the optimized multi-token speculative loop yet. It proves the
// functional contract end-to-end:
//   1. target DFlash consumes the committed token and exposes pre-norm hidden
//   2. native MTP/NextN consumes that hidden in the same graph and drafts
//   3. greedy target logits accept or correct the MTP draft token
//   4. the chosen token becomes the next committed token
//
// Usage:
//   smoke_mtp_integrated_decode <qwen35-mtp.gguf> [n_gen] [seed_token_id] [cuda_gpu]

#include "dflash27b.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace dflash27b;

static bool run_integrated_step(const TargetWeights & w,
                                TargetCache & target_cache,
                                TargetMtpCache & mtp_cache,
                                ggml_backend_t backend,
                                int32_t token,
                                int kv_start,
                                int32_t & target_next,
                                int32_t & mtp_next) {
    ggml_init_params ip{};
    ip.mem_size   = 768 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init graph failed\n");
        return false;
    }

    const int n_tokens = 1;
    ggml_tensor * inp_embed = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w.n_embd, n_tokens, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(inp_embed, "inp_embed");
    ggml_set_name(positions, "positions");
    ggml_set_input(inp_embed);
    ggml_set_input(positions);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    QwenGraphInputs target_in{};
    target_in.inp_embed = inp_embed;
    target_in.positions = positions;
    target_in.n_tokens = n_tokens;
    target_in.kv_start = kv_start;
    target_in.expose_pre_norm_hidden = true;

    QwenGraphOutputs target_out = build_qwen35_graph(ctx, gf, w, target_cache, target_in);
    if (!target_out.logits || !target_out.pre_norm_hidden) {
        std::fprintf(stderr, "build_qwen35_graph failed: %s\n", dflash27b_last_error());
        ggml_free(ctx);
        return false;
    }

    QwenMtpGraphInputs mtp_in{};
    mtp_in.token_embed = ggml_reshape_2d(ctx, inp_embed, w.n_embd, n_tokens);
    mtp_in.pre_norm_hidden = target_out.pre_norm_hidden;
    mtp_in.positions = positions;
    mtp_in.n_tokens = n_tokens;
    mtp_in.kv_start = kv_start;

    QwenMtpGraphOutputs mtp_out = build_qwen35_mtp_graph(ctx, gf, w, mtp_cache, mtp_in);
    if (!mtp_out.logits) {
        std::fprintf(stderr, "build_qwen35_mtp_graph failed: %s\n", dflash27b_last_error());
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * target_argmax = ggml_argmax(ctx, target_out.logits);
    ggml_set_name(target_argmax, "target_argmax");
    ggml_set_output(target_argmax);
    ggml_build_forward_expand(gf, target_argmax);

    ggml_tensor * mtp_argmax = ggml_argmax(ctx, mtp_out.logits);
    ggml_set_name(mtp_argmax, "mtp_argmax");
    ggml_set_output(mtp_argmax);
    ggml_build_forward_expand(gf, mtp_argmax);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "ggml_gallocr_alloc_graph failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> embed_buf((size_t)w.n_embd);
    if (!w.embedder.embed(&token, 1, embed_buf.data())) {
        std::fprintf(stderr, "cpu embedder failed for token %d\n", (int)token);
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }
    int32_t pos4[4] = { kv_start, kv_start, kv_start, kv_start };
    ggml_backend_tensor_set(inp_embed, embed_buf.data(), 0, sizeof(float) * embed_buf.size());
    ggml_backend_tensor_set(positions, pos4, 0, sizeof(pos4));

    auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed: %d\n", (int)status);
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(target_argmax, &target_next, 0, sizeof(target_next));
    ggml_backend_tensor_get(mtp_argmax, &mtp_next, 0, sizeof(mtp_next));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen35-mtp.gguf> [n_gen] [seed_token_id] [cuda_gpu]\n", argv[0]);
        return 2;
    }
    const int n_gen = argc >= 3 ? std::max(1, std::atoi(argv[2])) : 8;
    int32_t last_tok = argc >= 4 ? (int32_t)std::atoi(argv[3]) : 1;
    const int gpu = argc >= 5 ? std::atoi(argv[4]) : 0;

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
    if (!create_target_cache(w, /*max_ctx=*/std::max(64, n_gen + 8),
                             /*max_verify_tokens=*/0, backend, target_cache)) {
        std::fprintf(stderr, "create_target_cache: %s\n", dflash27b_last_error());
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    TargetMtpCache mtp_cache;
    if (!create_target_mtp_cache(w, /*max_ctx=*/std::max(64, n_gen + 8),
                                 backend, mtp_cache)) {
        std::fprintf(stderr, "create_target_mtp_cache: %s\n", dflash27b_last_error());
        free_target_cache(target_cache);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<int32_t> generated;
    generated.reserve((size_t)n_gen);
    int draft_n = 0;
    int accepted = 0;
    int corrected = 0;

    auto t0 = std::chrono::steady_clock::now();
    for (int pos = 0; pos < n_gen; pos++) {
        int32_t target_next = -1;
        int32_t mtp_next = -1;
        if (!run_integrated_step(w, target_cache, mtp_cache, backend,
                                 last_tok, pos, target_next, mtp_next)) {
            free_target_mtp_cache(mtp_cache);
            free_target_cache(target_cache);
            free_target_weights(w);
            ggml_backend_free(backend);
            return 1;
        }

        draft_n++;
        const bool ok = (mtp_next == target_next);
        if (ok) {
            accepted++;
        } else {
            corrected++;
        }
        const int32_t chosen = ok ? mtp_next : target_next;
        generated.push_back(chosen);
        target_cache.cur_pos = pos + 1;
        target_cache.last_tok = chosen;
        mtp_cache.cur_pos = pos + 1;

        std::printf("[mtp-decode step=%d] input=%d mtp=%d target=%d %s chosen=%d\n",
            pos, (int)last_tok, (int)mtp_next, (int)target_next,
            ok ? "ACCEPT" : "CORRECT", (int)chosen);
        last_tok = chosen;
    }
    auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();

    std::printf("[mtp-decode] generated=%d draft_n=%d accepted=%d corrected=%d acceptance=%.1f%% tok/s=%.2f\n",
        n_gen, draft_n, accepted, corrected,
        draft_n > 0 ? 100.0 * accepted / draft_n : 0.0,
        n_gen / std::max(1e-9, seconds));
    std::printf("[mtp-decode ids]");
    for (int32_t t : generated) std::printf(" %d", (int)t);
    std::printf("\nOK\n");

    free_target_mtp_cache(mtp_cache);
    free_target_cache(target_cache);
    free_target_weights(w);
    ggml_backend_free(backend);
    return 0;
}
