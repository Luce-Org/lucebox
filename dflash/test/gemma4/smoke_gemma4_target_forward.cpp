// Smoke test: load Gemma4 target, run a single-token forward pass, validate logits.
//
// Usage: smoke_gemma4_target_forward <gemma4.gguf>

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
#include <limits>
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

    // Load target weights
    GemmaTargetWeights w;
    if (!load_gemma4_target_gguf(argv[1], backend, w)) {
        std::fprintf(stderr, "load_gemma4_target_gguf: %s\n", dflash27b_last_error());
        ggml_backend_free(backend);
        return 1;
    }
    std::printf("[target] n_layer=%d n_embd=%d n_vocab=%d\n",
                w.n_layer, w.n_embd, w.n_vocab);

    // Create target cache
    GemmaTargetCache cache;
    const int max_ctx = 512;
    if (!create_gemma4_cache(w, max_ctx, backend, cache)) {
        std::fprintf(stderr, "create_gemma4_cache: %s\n", dflash27b_last_error());
        free_gemma4_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }
    std::printf("[cache] created max_ctx=%d kv_layers=%zu\n",
                cache.max_ctx, cache.attn_k.size());

    // Build graph context
    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) { std::fprintf(stderr, "ggml_init failed\n"); return 1; }

    // Input tensors for a single token at position 0
    const int n_tokens = 1;
    const int hidden   = w.n_embd;
    const int kv_start = 0;

    // Gemma4 uses 1D positions (not M-RoPE with 4 values like Qwen)
    ggml_tensor * inp_embed = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_embed, "inp_embed");
    ggml_set_name(positions, "positions");
    ggml_set_input(inp_embed);
    ggml_set_input(positions);

    // CUDA flash attention for head_dim>=512 (Gemma4-26B has head_dim=512 on full-attn
    // layers) requires a non-null mask so the GQA optimisation path is taken.
    // Provide a causal attention mask: shape [kv_len_padded, n_tokens], F32.
    // Entries are 0.0 for positions we attend to and -INF for positions we don't.
    const int kv_len        = kv_start + n_tokens;           // 1
    const int kv_len_padded = ((kv_len + 255) / 256) * 256;  // 256
    ggml_tensor * attn_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, kv_len_padded, n_tokens);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    // SWA mask sized to match the SWA layers' view of cache.swa_K.
    // Without this, gemma4_target_graph falls back to attn_mask for SWA layers
    // (gemma4_target_graph.cpp:972), which is sized for the full-attn view
    // and lets FA read past the populated region — catastrophic with TQ3_0/Q4_0
    // KV. Mirror the same wiring used by gemma4_runtime_helpers.cpp:131-137.
    const int swa_kv_pad = ((cache.swa_ctx_alloc + 255) / 256) * 256;
    ggml_tensor * swa_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, swa_kv_pad, n_tokens);
    ggml_set_name(swa_mask, "swa_mask");
    ggml_set_input(swa_mask);

    GemmaGraphInputs gi{};
    gi.inp_embed      = inp_embed;
    gi.positions      = positions;
    gi.attn_mask      = attn_mask;
    gi.swa_mask       = swa_mask;
    gi.n_tokens       = n_tokens;
    gi.kv_start       = kv_start;
    gi.capture_layers = true;

    // Build graph
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 16384, false);
    GemmaGraphOutputs go = build_gemma4_graph(gctx, gf, w, cache, gi);
    if (!go.logits) { std::fprintf(stderr, "build_gemma4_graph returned null logits\n"); return 1; }
    ggml_set_output(go.logits);
    ggml_build_forward_expand(gf, go.logits);
    std::printf("[graph] nodes=%d\n", ggml_graph_n_nodes(gf));

    // Allocate graph memory
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "ggml_gallocr_alloc_graph failed\n");
        return 1;
    }

    // Fill causal attention mask (F16).
    // mask[k, q] = 0.0  if k <= q  (position k is visible from query q)
    //            = -INF otherwise   (masked out / padding)
    {
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t ninf_h = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask_data((size_t)kv_len_padded * n_tokens, ninf_h);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = 0; k <= kv_start + q; k++) {
                mask_data[(size_t)q * kv_len_padded + k] = zero_h;
            }
        }
        ggml_backend_tensor_set(attn_mask, mask_data.data(), 0,
                                sizeof(ggml_fp16_t) * mask_data.size());
    }

    // Fill SWA mask the same way (only k=kv_start..kv_start+q visible per query;
    // rest -INF). Cache positions past current pos are zero-initialised garbage
    // and must be masked out.
    {
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t ninf_h = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask_data((size_t)swa_kv_pad * n_tokens, ninf_h);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = 0; k <= kv_start + q; k++) {
                mask_data[(size_t)q * swa_kv_pad + k] = zero_h;
            }
        }
        ggml_backend_tensor_set(swa_mask, mask_data.data(), 0,
                                sizeof(ggml_fp16_t) * mask_data.size());
    }

    // Embed token id=2 (BOS) using the CPU embedder
    int32_t bos_id = 2;
    std::vector<float> embed_buf((size_t)hidden * n_tokens);
    if (!w.embedder.embed(&bos_id, n_tokens, embed_buf.data())) {
        std::fprintf(stderr, "embedder.embed failed\n");
        return 1;
    }
    ggml_backend_tensor_set(inp_embed, embed_buf.data(), 0, sizeof(float) * embed_buf.size());

    // Position 0
    int32_t pos0 = 0;
    ggml_backend_tensor_set(positions, &pos0, 0, sizeof(int32_t));

    // Compute
    auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ggml_backend_graph_compute failed: %d\n", (int)status);
        return 1;
    }
    std::printf("[compute] OK\n");

    // Read logits back
    const int64_t vocab = w.n_vocab;
    std::vector<float> logits((size_t)vocab);
    ggml_backend_tensor_get(go.logits, logits.data(), 0, sizeof(float) * vocab);

    // Check for NaN / Inf and validate softcap bounds
    int n_nan = 0, n_inf = 0, n_oob = 0;
    float vmin = 1e30f, vmax = -1e30f;
    for (auto v : logits) {
        if (std::isnan(v)) { n_nan++; continue; }
        if (std::isinf(v)) { n_inf++; continue; }
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        // Logit softcap = 30.0 means values are in (-30, 30)
        if (v < -30.0f || v > 30.0f) n_oob++;
    }
    std::printf("[logits] vocab=%" PRId64 " nan=%d inf=%d oob=%d min=%.4g max=%.4g\n",
                vocab, n_nan, n_inf, n_oob, vmin, vmax);

    if (n_nan > 0) fail("NaN values in logits");
    if (n_inf > 0) fail("Inf values in logits");
    if (n_oob > 0) {
        char buf[64];
        std::snprintf(buf, sizeof(buf),
            "%d logit values outside [-30, 30] softcap bounds", n_oob);
        fail(buf);
    }

    // Print top-5 tokens
    std::vector<std::pair<float, int>> sorted;
    sorted.reserve((size_t)vocab);
    for (int i = 0; i < (int)vocab; i++) sorted.emplace_back(logits[i], i);
    std::partial_sort(sorted.begin(), sorted.begin() + 5, sorted.end(),
        [](const auto & a, const auto & b) { return a.first > b.first; });
    std::printf("[top 5]");
    for (int i = 0; i < 5; i++) {
        std::printf("  id=%d l=%.3f", sorted[i].second, sorted[i].first);
    }
    std::printf("\n");

    ggml_gallocr_free(alloc);
    ggml_free(gctx);
    free_gemma4_cache(cache);
    free_gemma4_target_weights(w);
    ggml_backend_free(backend);
    std::printf("PASS\n");
    return 0;
}
