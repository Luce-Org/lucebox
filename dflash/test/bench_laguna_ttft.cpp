// TTFT (time-to-first-token) bench for Laguna in dflash. Loads Laguna Q4_K_M,
// allocates a cache sized for the longest prefill, then for each context
// length: builds a prefill graph for N tokens (synthetic input), runs it on
// CUDA, measures wall time. Reports TTFT @ each length.
//
// Usage:
//   bench_laguna_ttft <laguna.gguf> ["4096,16384,32768"]
//
// The synthetic input uses token id 1972 repeated N times (avoids BOS
// special-casing; any non-special id works, the bench measures wall time not
// generation quality).
//
// On RTX 3090 24 GB the practical ceiling without KV bit-reduction:
//   Q8_0 KV  + 18.77 GiB weights -> ~32K context
//   For 64K+ need Q4_0 KV (planned, not in this bench).

#include "laguna_internal.h"
#include "internal.h"
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

static std::vector<int> parse_csv(const std::string & s, std::vector<int> dflt) {
    if (s.empty()) return dflt;
    std::vector<int> out;
    size_t start = 0;
    while (start < s.size()) {
        size_t comma = s.find(',', start);
        std::string tok = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

// Forward step lives in src/laguna_target_graph.cpp::laguna_step(); this
// bench just times chunked prefill on top of it.


int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <laguna.gguf> [\"4096,16384,32768\"]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const std::string lens_csv = (argc >= 3) ? argv[2] : "";
    std::vector<int> ctx_lens = parse_csv(lens_csv, {1024, 4096, 16384});
    int max_len = 0;
    for (int n : ctx_lens) if (n > max_len) max_len = n;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    LagunaTargetWeights w;
    if (!load_target_gguf_laguna(path, backend, w)) {
        std::fprintf(stderr, "load failed: %s\n", dflash27b_last_error());
        ggml_backend_free(backend); return 1;
    }

    LagunaTargetCache cache;
    // Override KV cache dtype via env (Q4_0 fits 128K on a 24 GB GPU; Q8_0 caps near 32K).
    if (const char * kv_t = std::getenv("DFLASH_KV_TYPE")) {
        const std::string s = kv_t;
        if      (s == "q4_0" || s == "Q4_0") { cache.kv_k_type = GGML_TYPE_Q4_0; cache.kv_v_type = GGML_TYPE_Q4_0; }
        else if (s == "q5_0" || s == "Q5_0") { cache.kv_k_type = GGML_TYPE_Q5_0; cache.kv_v_type = GGML_TYPE_Q5_0; }
        else if (s == "q8_0" || s == "Q8_0") { cache.kv_k_type = GGML_TYPE_Q8_0; cache.kv_v_type = GGML_TYPE_Q8_0; }
        else if (s == "f16")                  { cache.kv_k_type = GGML_TYPE_F16;  cache.kv_v_type = GGML_TYPE_F16;  }
    }
    std::printf("[bench] KV K=%s V=%s\n",
                ggml_type_name(cache.kv_k_type), ggml_type_name(cache.kv_v_type));
    if (!create_laguna_target_cache(w, max_len, backend, cache)) {
        std::fprintf(stderr, "cache failed: %s\n", dflash27b_last_error());
        free_laguna_target_weights(w); ggml_backend_free(backend); return 1;
    }
    std::printf("[bench] cache max_ctx=%d  KV bytes/layer ~ %.1f MiB\n",
                max_len, (2.0 * w.head_dim * max_len * w.n_head_kv) / (1024.0 * 1024.0));

    const int32_t fake_tok = 1972;  // "hello" or whatever; just a non-special id

    // Chunked prefill: at large N a single forward exceeds the 24 GB activation
    // budget on RTX 3090 (MoE intermediate [n_embd, n_used, n_tokens] = 1 GB at
    // n_tokens=16K). Split N into CHUNK chunks, advance kv_start per chunk.
    int chunk_env = 0;
    if (const char * c = std::getenv("DFLASH_CHUNK")) chunk_env = std::atoi(c);
    const int CHUNK = chunk_env > 0 ? chunk_env : 4096;

    const bool no_mask = (std::getenv("DFLASH_NO_MASK") != nullptr);

    for (int N : ctx_lens) {
        if (N > max_len) { std::printf("[bench] skip N=%d > max_len=%d\n", N, max_len); continue; }
        reset_laguna_target_cache(cache);

        std::vector<int32_t> ids((size_t)N, fake_tok);
        std::vector<float> embed_full((size_t)N * w.n_embd);
        if (!w.embedder.embed(ids.data(), N, embed_full.data())) {
            std::fprintf(stderr, "embed failed at N=%d\n", N);
            continue;
        }

        const int chunk = std::min(N, CHUNK);
        const int n_chunks = (N + chunk - 1) / chunk;

        // Time the SUM of all chunks as TTFT. The first chunk pays the
        // gallocr planning cost; subsequent chunks reuse the static gallocr
        // inside laguna_step(), so steady-state per-chunk timing emerges
        // after the first call.
        bool ok = true;
        double total_pf_s = 0.0;
        std::vector<float> last_logits;
        for (int c = 0; c < n_chunks && ok; ++c) {
            const int kv_start = c * chunk;
            const int n_tok    = std::min(chunk, N - c * chunk);
            std::vector<float> tmp;
            auto tA = std::chrono::steady_clock::now();
            const bool step_ok = laguna_step(
                backend, w, cache,
                embed_full.data() + (size_t)kv_start * w.n_embd,
                n_tok, kv_start, no_mask,
                (c == n_chunks - 1) ? last_logits : tmp);
            auto tB = std::chrono::steady_clock::now();
            if (!step_ok) {
                std::fprintf(stderr, "laguna_step N=%d chunk=%d failed\n", N, c);
                ok = false; break;
            }
            total_pf_s += std::chrono::duration<double>(tB - tA).count();
        }
        if (!ok) continue;

        int best = 0; float bv = last_logits.empty() ? 0.0f : last_logits[0];
        int n_nan = 0, n_inf = 0;
        for (size_t i = 0; i < last_logits.size(); ++i) {
            const float v = last_logits[i];
            if (std::isnan(v)) ++n_nan;
            if (std::isinf(v)) ++n_inf;
            if (v > bv) { bv = v; best = (int)i; }
        }
        std::printf("[bench] N=%6d  TTFT=%8.3f s  (%6.1f tok/s) chunks=%d  argmax=%d  logit=%.3f  nan=%d inf=%d\n",
                    N, total_pf_s, N / std::max(1e-9, total_pf_s),
                    n_chunks, best, bv, n_nan, n_inf);
    }

    free_laguna_target_cache(cache);
    free_laguna_target_weights(w);
    ggml_backend_free(backend);
    return 0;
}
