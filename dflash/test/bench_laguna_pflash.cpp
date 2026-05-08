// End-to-end PFlash + Laguna TTFT bench. Mirrors the qwen3.6-27B PFlash flow:
//
//   1. Tokenize input (synthetic in DRAFTER vocab for the bench)
//   2. Drafter (Qwen3-0.6B BF16) score_and_compress  -> surviving Qwen3 IDs
//   3. Cross-tokenizer mapping Qwen3 IDs -> Laguna IDs (NOT plumbed yet; we
//      use a fake target token for compute-time-only measurement)
//   4. Laguna build_laguna_graph dense prefill on the COMPRESSED sequence
//   5. Report drafter time + target time + total TTFT and the compression
//      ratio achieved.
//
// Usage:
//   bench_laguna_pflash <laguna.gguf> <drafter.gguf> <N> [keep_ratio=0.10] [chunk=2048]

#include "laguna_internal.h"
#include "internal.h"
#include "qwen3_drafter.h"
#include "dflash27b.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cuda.h"

using namespace dflash27b;

// Chunked prefill loop on top of the shared laguna_step() helper. Reports
// total prefill time and the argmax / logit at the LAST chunk.
static bool laguna_chunked_prefill(
    ggml_backend_t backend,
    const LagunaTargetWeights & w,
    LagunaTargetCache & cache,
    const std::vector<float> & embed_full,
    int n_tokens,
    int chunk,
    bool no_mask,
    double * out_pf_s,
    int * out_argmax,
    float * out_logit)
{
    *out_pf_s = 0.0;
    *out_argmax = -1;
    *out_logit  = 0.0f;
    const int n_chunks = (n_tokens + chunk - 1) / chunk;
    std::vector<float> last_logits, scratch;
    for (int c = 0; c < n_chunks; ++c) {
        const int kv_start = c * chunk;
        const int n_tok    = std::min(chunk, n_tokens - c * chunk);
        std::vector<float> & sink = (c == n_chunks - 1) ? last_logits : scratch;
        auto tA = std::chrono::steady_clock::now();
        if (!laguna_step(backend, w, cache,
                          embed_full.data() + (size_t)kv_start * w.n_embd,
                          n_tok, kv_start, no_mask, sink)) {
            std::fprintf(stderr, "laguna_step chunk=%d failed\n", c);
            return false;
        }
        auto tB = std::chrono::steady_clock::now();
        *out_pf_s += std::chrono::duration<double>(tB - tA).count();
    }
    if (!last_logits.empty()) {
        int best = 0; float bv = last_logits[0];
        for (size_t i = 1; i < last_logits.size(); ++i) {
            if (last_logits[i] > bv) { bv = last_logits[i]; best = (int)i; }
        }
        *out_argmax = best;
        *out_logit  = bv;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <laguna.gguf> <drafter.gguf> <N> [keep_ratio=0.10] [chunk=2048]\n", argv[0]);
        return 2;
    }
    const std::string laguna_path  = argv[1];
    const std::string drafter_path = argv[2];
    const int N           = std::atoi(argv[3]);
    const float keep_r    = (argc >= 5) ? std::atof(argv[4]) : 0.10f;
    const int chunk_arg   = (argc >= 6) ? std::atoi(argv[5]) : 2048;
    const int32_t fake_q  = 1972;  // any non-special drafter id
    const int32_t fake_l  = 1972;  // dummy laguna id (cross-tokenizer skipped)
    const bool no_mask    = (std::getenv("DFLASH_NO_MASK") != nullptr);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    // ---- Phase 1: drafter only. Loads ~1.2 GB BF16 weights, runs compress
    //               with full VRAM available, then frees so target gets max
    //               headroom for activations + cache.
    DrafterContext drafter;
    auto td0 = std::chrono::steady_clock::now();
    if (!load_drafter(drafter_path, /*gpu_layers=*/-1, drafter)) {
        std::fprintf(stderr, "load_drafter failed: %s\n", dflash27b_last_error());
        return 1;
    }
    auto td1 = std::chrono::steady_clock::now();
    std::printf("[pflash] drafter loaded in %.2fs vocab=%d\n",
                std::chrono::duration<double>(td1 - td0).count(), drafter.weights.n_vocab);

    std::vector<int32_t> input(N, fake_q);
    auto tc0 = std::chrono::steady_clock::now();
    std::vector<int32_t> compressed = drafter_score_and_compress(
        drafter, input, keep_r, /*chunk_size=*/32, /*n_lookahead=*/8, /*pool_kernel=*/13);
    auto tc1 = std::chrono::steady_clock::now();
    if (compressed.empty()) {
        std::fprintf(stderr, "drafter compress failed: %s\n", dflash27b_last_error());
        free_drafter(drafter); return 1;
    }
    const int M = (int)compressed.size();
    const double drafter_s = std::chrono::duration<double>(tc1 - tc0).count();
    std::printf("[pflash] drafter compress N=%d -> M=%d ratio=%.4f in %.3fs\n",
                N, M, (double)M / N, drafter_s);

    // Free drafter so its 1.2 GB BF16 weights + scratch buffers don't compete
    // with target cache + activations.
    free_drafter(drafter);

    // ---- Phase 2: load Laguna target now that drafter VRAM is free ----
    LagunaTargetWeights w;
    if (!load_target_gguf_laguna(laguna_path, backend, w)) {
        std::fprintf(stderr, "load_laguna failed: %s\n", dflash27b_last_error());
        return 1;
    }

    LagunaTargetCache cache;
    if (const char * kv_t = std::getenv("DFLASH_KV_TYPE")) {
        const std::string s = kv_t;
        if      (s == "q4_0") { cache.kv_k_type = GGML_TYPE_Q4_0; cache.kv_v_type = GGML_TYPE_Q4_0; }
        else if (s == "q5_0") { cache.kv_k_type = GGML_TYPE_Q5_0; cache.kv_v_type = GGML_TYPE_Q5_0; }
        else if (s == "q8_0") { cache.kv_k_type = GGML_TYPE_Q8_0; cache.kv_v_type = GGML_TYPE_Q8_0; }
        else if (s == "f16")  { cache.kv_k_type = GGML_TYPE_F16;  cache.kv_v_type = GGML_TYPE_F16;  }
    }
    // Cache sized for COMPRESSED length M (much smaller than raw N).
    if (!create_laguna_target_cache(w, M, backend, cache)) {
        std::fprintf(stderr, "create_laguna_target_cache: %s\n", dflash27b_last_error());
        free_laguna_target_weights(w); return 1;
    }
    std::printf("[pflash] laguna cache max_ctx=%d KV=%s/%s\n", M,
                ggml_type_name(cache.kv_k_type), ggml_type_name(cache.kv_v_type));

    // ---- Embed compressed token (single fake) for Laguna --------------
    std::vector<int32_t> laguna_ids(M, fake_l);
    std::vector<float> embed_full((size_t)M * w.n_embd);
    if (!w.embedder.embed(laguna_ids.data(), M, embed_full.data())) {
        std::fprintf(stderr, "laguna embed failed at M=%d\n", M);
        // drafter was already freed above before loading the laguna target;
        // do NOT free it again here.
        free_laguna_target_cache(cache); free_laguna_target_weights(w);
        return 1;
    }

    // ---- Laguna chunked prefill on COMPRESSED sequence -----------------
    const int chunk = std::min(M, chunk_arg);
    double tgt_s = 0.0; int argmax = 0; float logit = 0.0f;
    auto tt0 = std::chrono::steady_clock::now();
    if (!laguna_chunked_prefill(backend, w, cache, embed_full,
                                 M, chunk, no_mask,
                                 &tgt_s, &argmax, &logit)) {
        std::fprintf(stderr, "laguna prefill failed\n");
        free_laguna_target_cache(cache); free_laguna_target_weights(w);
        return 1;
    }
    auto tt1 = std::chrono::steady_clock::now();
    const double tgt_total_s = std::chrono::duration<double>(tt1 - tt0).count();
    const double total_ttft  = drafter_s + tgt_s;

    std::printf("[pflash] laguna prefill M=%d in %.3fs (graph %.3fs, %.1f tok/s effective on N=%d)\n",
                M, tgt_total_s, tgt_s, N / std::max(1e-9, total_ttft), N);
    std::printf("[pflash] === SUMMARY ===\n");
    std::printf("[pflash] N=%d  M=%d  compress=%.4f  drafter=%.3fs  target=%.3fs  TTFT=%.3fs  effective=%.1f tok/s\n",
                N, M, (double)M / N, drafter_s, tgt_s, total_ttft,
                N / std::max(1e-9, total_ttft));
    std::printf("[pflash] argmax=%d logit=%.3f\n", argmax, logit);

    free_laguna_target_cache(cache);
    free_laguna_target_weights(w);
    ggml_backend_free(backend);
    return 0;
}
