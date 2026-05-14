// Smoke test: load Gemma4 DFlash draft weights, build a forward graph with
// synthetic inputs, run on CUDA, and validate logits.
//
// Usage: smoke_gemma4_draft_forward <draft_dir> <target.gguf>

#include "internal.h"
#include "gemma4.h"

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
#include <random>
#include <vector>

using namespace dflash27b;

static void fail(const char * msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <draft_dir> <target.gguf>\n", argv[0]);
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    GemmaDraftWeights dw;
    if (!load_gemma4_draft_safetensors(argv[1], backend, dw)) {
        std::fprintf(stderr, "load_gemma4_draft_safetensors: %s\n", dflash27b_last_error());
        ggml_backend_free(backend);
        return 1;
    }

    // Load target to get tok_embd (shared between target and draft for LM head).
    // tok_embd is not in the draft safetensors — it must come from the target at runtime.
    // The target loader keeps tok_embd CPU-side (CpuEmbedder / mmap) to avoid uploading
    // ~400 MiB to VRAM for every inference.  For this smoke test we upload it once.
    GemmaTargetWeights tw;
    if (!load_gemma4_target_gguf(argv[2], backend, tw)) {
        std::fprintf(stderr, "load_gemma4_target_gguf: %s\n", dflash27b_last_error());
        free_gemma4_draft_weights(dw);
        ggml_backend_free(backend);
        return 1;
    }

    // tw.tok_embd is metadata-only (data = nullptr); actual bytes live in tw.embedder.
    // Allocate a dedicated GPU tensor for tok_embd and upload the quantized bytes.
    ggml_context * tok_embd_ctx = nullptr;
    ggml_backend_buffer_t tok_embd_buf = nullptr;
    {
        ggml_init_params ep{};
        ep.mem_size   = ggml_tensor_overhead() * 2;
        ep.mem_buffer = nullptr;
        ep.no_alloc   = true;
        tok_embd_ctx = ggml_init(ep);
        if (!tok_embd_ctx) {
            std::fprintf(stderr, "ggml_init for tok_embd failed\n");
            free_gemma4_target_weights(tw);
            free_gemma4_draft_weights(dw);
            ggml_backend_free(backend);
            return 1;
        }

        const ggml_type emb_type = tw.embedder.tok_embd_type;
        const int64_t   n_embd_t = tw.embedder.n_embd;
        const int64_t   n_vocab_t = tw.embedder.n_vocab;

        // ggml convention: ne[0] = n_embd (fast axis), ne[1] = n_vocab
        ggml_tensor * te = ggml_new_tensor_2d(tok_embd_ctx, emb_type, n_embd_t, n_vocab_t);
        ggml_set_name(te, "tok_embd_gpu");

        tok_embd_buf = ggml_backend_alloc_ctx_tensors(tok_embd_ctx, backend);
        if (!tok_embd_buf) {
            std::fprintf(stderr, "ggml_backend_alloc_ctx_tensors for tok_embd failed\n");
            ggml_free(tok_embd_ctx);
            free_gemma4_target_weights(tw);
            free_gemma4_draft_weights(dw);
            ggml_backend_free(backend);
            return 1;
        }

        const size_t emb_bytes = (size_t)tw.embedder.row_bytes * (size_t)n_vocab_t;
        ggml_backend_tensor_set(te, tw.embedder.tok_embd_bytes, 0, emb_bytes);
        std::printf("[tok_embd] uploaded %.1f MiB to GPU (%s [%" PRId64 ", %" PRId64 "])\n",
                    (double)emb_bytes / (1024.0 * 1024.0),
                    ggml_type_name(emb_type), n_embd_t, n_vocab_t);

        dw.tok_embd = te;
        dw.n_vocab  = (int)n_vocab_t;
    }

    std::printf("[draft] n_layer=%d n_head=%d n_embd=%d n_vocab=%d target_hidden=%d\n",
                dw.n_layer, dw.n_head, dw.n_embd, dw.n_vocab, dw.target_hidden);

    const int n_tokens      = 16;                                    // one block
    const int target_feat_w = dw.n_target_layers * dw.target_hidden; // 6*4096 = 24576
    const int draft_hidden  = dw.n_embd;
    const int n_vocab       = dw.n_vocab;
    const int kq_mask_pad   = 32;

    auto align_up = [](int x, int a) { return ((x + a - 1) / a) * a; };

    // Allocate draft KV cache
    GemmaTargetCache cache;
    cache.backend = backend;
    if (!create_draft_kv_cache(dw, backend, cache)) {
        std::fprintf(stderr, "create_draft_kv_cache failed\n");
        return 1;
    }
    std::printf("[draft kv] cap=%d\n", cache.draft_kv_cap);

    // ── Step 1: Prefill draft KV with synthetic target features ──────
    // Simulate n_tokens context positions with random target features
    {
        ggml_init_params ip{};
        ip.mem_size   = 256 * 1024 * 1024;
        ip.no_alloc   = true;
        ggml_context * pctx = ggml_init(ip);
        if (!pctx) { fail("ggml_init for prefill failed"); }

        ggml_tensor * pf_target_feat = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, target_feat_w, n_tokens);
        ggml_tensor * pf_positions   = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, n_tokens);
        ggml_set_name(pf_target_feat, "pf_target_feat");
        ggml_set_name(pf_positions,   "pf_positions");
        ggml_set_input(pf_target_feat);
        ggml_set_input(pf_positions);

        ggml_cgraph * pf_gf = ggml_new_graph_custom(pctx, 4096, false);
        build_draft_kv_prefill_graph(pctx, pf_gf, dw, cache,
                                      pf_target_feat, pf_positions, n_tokens);

        ggml_gallocr_t pf_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(pf_alloc, pf_gf)) { fail("prefill alloc failed"); }

        std::mt19937 rng_pf(42);
        std::uniform_real_distribution<float> u_pf(-0.05f, 0.05f);
        {
            std::vector<float> data((size_t)target_feat_w * n_tokens);
            for (auto & v : data) v = u_pf(rng_pf);
            ggml_backend_tensor_set(pf_target_feat, data.data(), 0, sizeof(float) * data.size());
        }
        {
            std::vector<int32_t> pos(n_tokens);
            for (int i = 0; i < n_tokens; i++) pos[i] = i;
            ggml_backend_tensor_set(pf_positions, pos.data(), 0, sizeof(int32_t) * n_tokens);
        }

        auto st = ggml_backend_graph_compute(backend, pf_gf);
        if (st != GGML_STATUS_SUCCESS) { fail("prefill compute failed"); }
        cache.draft_kv_pos = n_tokens;
        std::printf("[prefill] KV materialized for %d positions\n", n_tokens);

        ggml_gallocr_free(pf_alloc);
        ggml_free(pctx);
    }

    // ── Step 2: Draft forward with KV cache ──────────────────────────
    const int kv_start = cache.draft_kv_pos;  // context length = n_tokens

    ggml_init_params ip{};
    ip.mem_size   = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) { std::fprintf(stderr, "ggml_init failed\n"); return 1; }

    ggml_tensor * draft_embed = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, draft_hidden, n_tokens);
    ggml_tensor * positions   = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_tokens);
    const int kv_len = kv_start + n_tokens;
    const int kv_pad = align_up(kv_len, kq_mask_pad);
    const int q_pad  = align_up(n_tokens, kq_mask_pad);
    ggml_tensor * attn_mask   = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, kv_pad, q_pad);

    ggml_set_name(draft_embed, "draft_embed");
    ggml_set_name(positions,   "positions");
    ggml_set_name(attn_mask,   "attn_mask");
    ggml_set_input(draft_embed);
    ggml_set_input(positions);
    ggml_set_input(attn_mask);

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 8192, false);
    ggml_tensor * logits = build_gemma4_draft_graph(
        gctx, gf, dw, cache,
        draft_embed, positions, attn_mask,
        n_tokens, kv_start);
    if (!logits) { std::fprintf(stderr, "build_gemma4_draft_graph returned null\n"); return 1; }
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    std::printf("[graph] nodes=%d\n", ggml_graph_n_nodes(gf));

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "ggml_gallocr_alloc_graph failed\n");
        return 1;
    }

    // Fill inputs with deterministic random data
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-0.05f, 0.05f);

    // draft_embed: [draft_hidden, 16] f32
    {
        std::vector<float> data((size_t)draft_hidden * n_tokens);
        for (auto & v : data) v = u(rng);
        ggml_backend_tensor_set(draft_embed, data.data(), 0, sizeof(float) * data.size());
    }
    // positions: [kv_start, kv_start+1, ..., kv_start+15]
    {
        std::vector<int32_t> pos(n_tokens);
        for (int i = 0; i < n_tokens; i++) pos[i] = kv_start + i;
        ggml_backend_tensor_set(positions, pos.data(), 0, sizeof(int32_t) * n_tokens);
    }
    // attn_mask: causal over full kv_len, block queries attend to all context + causal within block
    {
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t ninf_h = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask((size_t)kv_pad * q_pad, ninf_h);
        for (int q = 0; q < n_tokens; q++) {
            int max_kv = kv_start + q;  // attend to all context + block[0..q]
            for (int k = 0; k <= max_kv; k++) {
                mask[(size_t)q * kv_pad + k] = zero_h;
            }
        }
        ggml_backend_tensor_set(attn_mask, mask.data(), 0, sizeof(ggml_fp16_t) * mask.size());
    }

    // Compute
    auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ggml_backend_graph_compute failed: %d\n", (int)status);
        return 1;
    }
    std::printf("[compute] OK\n");

    // Validate expected output shape
    if (logits->ne[0] != (int64_t)n_vocab || logits->ne[1] != (int64_t)n_tokens) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "logits shape [%" PRId64 ", %" PRId64 "] expected [%d, %d]",
            logits->ne[0], logits->ne[1], n_vocab, n_tokens);
        fail(buf);
    }
    std::printf("[logits] shape: [%" PRId64 ", %" PRId64 "]\n",
                logits->ne[0], logits->ne[1]);

    // Read logits for position 0
    std::vector<float> logit_buf((size_t)n_vocab * n_tokens);
    ggml_backend_tensor_get(logits, logit_buf.data(), 0, sizeof(float) * logit_buf.size());

    // Check for NaN and softcap bounds across all positions
    int n_nan = 0, n_oob = 0;
    float vmin = 1e30f, vmax = -1e30f;
    for (auto v : logit_buf) {
        if (std::isnan(v)) { n_nan++; continue; }
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        if (v < -30.0f || v > 30.0f) n_oob++;
    }
    std::printf("[logits] nan=%d oob=%d min=%.4g max=%.4g\n",
                n_nan, n_oob, vmin, vmax);

    if (n_nan > 0) fail("NaN values in logits");
    if (n_oob > 0) {
        char buf[64];
        std::snprintf(buf, sizeof(buf),
            "%d logit values outside [-30, 30] softcap bounds", n_oob);
        fail(buf);
    }

    // Top-5 tokens for position 0
    const float * pos0_logits = logit_buf.data();
    std::vector<std::pair<float, int>> sorted;
    sorted.reserve((size_t)n_vocab);
    for (int i = 0; i < n_vocab; i++) sorted.emplace_back(pos0_logits[i], i);
    std::partial_sort(sorted.begin(), sorted.begin() + 5, sorted.end(),
        [](const auto & a, const auto & b) { return a.first > b.first; });
    std::printf("[top 5 pos=0]");
    for (int i = 0; i < 5; i++) {
        std::printf("  id=%d l=%.3f", sorted[i].second, sorted[i].first);
    }
    std::printf("\n");

    ggml_gallocr_free(alloc);
    ggml_free(gctx);
    free_draft_kv_cache(cache);
    // dw.tok_embd points into tok_embd_ctx/buf — null it before freeing the draft
    // so free_gemma4_draft_weights doesn't double-free or access freed memory.
    dw.tok_embd = nullptr;
    free_gemma4_draft_weights(dw);
    // Free tok_embd GPU allocation (must outlive the compute graph).
    if (tok_embd_buf) ggml_backend_buffer_free(tok_embd_buf);
    if (tok_embd_ctx) ggml_free(tok_embd_ctx);
    // Target weights own the mmap that backs tok_embd_bytes; free after GPU upload.
    free_gemma4_target_weights(tw);
    ggml_backend_free(backend);
    std::printf("PASS\n");
    return 0;
}
