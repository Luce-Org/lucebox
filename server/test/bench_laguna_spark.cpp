// Laguna Spark decode bench: drives LagunaBackend (the REAL hybrid/spark
// path used by dflash_server), honoring all DFLASH_* env knobs:
//   DFLASH_LAGUNA_HOTNESS=<csv>     calibrated placement
//   DFLASH_EXPERT_BUDGET_PCT=60     pinned-hot fraction
//   DFLASH_LAGUNA_CACHE_SLOTS=16    cache ring slots/layer
//   DFLASH_LAGUNA_PROFILE=1         cold-experts/token profiling
//   DFLASH_LAGUNA_NO_SINGLE_GRAPH=1 per-layer fallback (for trace capture)
//   DFLASH_LAGUNA_PREGATE_TRACE=<f> pregate trace capture (fallback path)
//
// Usage:
//   bench_laguna_spark <laguna.gguf> [prompt_N=128] [n_gen=256]
//   bench_laguna_spark <laguna.gguf> --draft <draft.gguf> --ddtree --ddtree-budget 22

#include "laguna_backend.h"
#include "dflash27b.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dflash::common;

static bool env_true(const char * name) {
    const char * v = std::getenv(name);
    return v && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

static int env_int(const char * name, int fallback) {
    const char * v = std::getenv(name);
    return v ? std::atoi(v) : fallback;
}

static float env_float(const char * name, float fallback) {
    const char * v = std::getenv(name);
    return v ? (float)std::atof(v) : fallback;
}

static bool parse_kv_type(const char * s, ggml_type & out) {
    if (!s) return false;
    if (std::strcmp(s, "q4_0") == 0) { out = GGML_TYPE_Q4_0; return true; }
    if (std::strcmp(s, "q5_0") == 0) { out = GGML_TYPE_Q5_0; return true; }
    if (std::strcmp(s, "q8_0") == 0) { out = GGML_TYPE_Q8_0; return true; }
    if (std::strcmp(s, "f16")  == 0) { out = GGML_TYPE_F16;  return true; }
    return false;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <laguna.gguf> [prompt_N=128] [n_gen=256] "
                     "[--draft draft.gguf] [--ddtree] [--ddtree-budget N]\n",
                     argv[0]);
        return 2;
    }

    std::vector<const char *> positional;
    std::string draft_path = std::getenv("DFLASH_DRAFT") ? std::getenv("DFLASH_DRAFT") : "";
    bool ddtree_mode = env_true("DFLASH_DDTREE") || env_true("DFLASH_LAGUNA_DDTREE");
    int ddtree_budget = env_int("DFLASH_DDTREE_BUDGET", env_int("DFLASH_LAGUNA_DDTREE_BUDGET", 22));
    float ddtree_temp = env_float("DFLASH_DDTREE_TEMP", env_float("DFLASH_LAGUNA_DDTREE_TEMP", 1.0f));
    int verify_width = env_int("DFLASH_VERIFY_WIDTH", 0);
    int max_ctx_arg = env_int("DFLASH_MAX_CTX", 0);
    int draft_gpu = env_int("DFLASH_DRAFT_GPU", -1);
    int draft_ctx_max = env_int("DFLASH_DRAFT_CTX_MAX", 4096);
    ggml_type kv_type = GGML_TYPE_Q8_0;
    parse_kv_type(std::getenv("DFLASH_KV_TYPE"), kv_type);

    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        if (std::strcmp(a, "--draft") == 0 && i + 1 < argc) {
            draft_path = argv[++i];
        } else if (std::strcmp(a, "--ddtree") == 0) {
            ddtree_mode = true;
        } else if (std::strcmp(a, "--ddtree-budget") == 0 && i + 1 < argc) {
            ddtree_budget = std::max(1, std::atoi(argv[++i]));
        } else if (std::strncmp(a, "--ddtree-budget=", 16) == 0) {
            ddtree_budget = std::max(1, std::atoi(a + 16));
        } else if (std::strcmp(a, "--ddtree-temp") == 0 && i + 1 < argc) {
            ddtree_temp = (float)std::atof(argv[++i]);
        } else if (std::strncmp(a, "--ddtree-temp=", 14) == 0) {
            ddtree_temp = (float)std::atof(a + 14);
        } else if (std::strcmp(a, "--verify-width") == 0 && i + 1 < argc) {
            verify_width = std::max(0, std::atoi(argv[++i]));
        } else if (std::strncmp(a, "--verify-width=", 15) == 0) {
            verify_width = std::max(0, std::atoi(a + 15));
        } else if (std::strcmp(a, "--max-ctx") == 0 && i + 1 < argc) {
            max_ctx_arg = std::max(1, std::atoi(argv[++i]));
        } else if (std::strncmp(a, "--max-ctx=", 10) == 0) {
            max_ctx_arg = std::max(1, std::atoi(a + 10));
        } else if (std::strcmp(a, "--draft-gpu") == 0 && i + 1 < argc) {
            draft_gpu = std::atoi(argv[++i]);
        } else if (std::strncmp(a, "--draft-gpu=", 12) == 0) {
            draft_gpu = std::atoi(a + 12);
        } else if (std::strcmp(a, "--draft-ctx-max") == 0 && i + 1 < argc) {
            draft_ctx_max = std::max(1, std::atoi(argv[++i]));
        } else if (std::strncmp(a, "--draft-ctx-max=", 16) == 0) {
            draft_ctx_max = std::max(1, std::atoi(a + 16));
        } else if (std::strcmp(a, "--kv") == 0 && i + 1 < argc) {
            if (!parse_kv_type(argv[++i], kv_type)) {
                std::fprintf(stderr, "unknown --kv type: %s\n", argv[i]);
                return 2;
            }
        } else if (std::strncmp(a, "--kv=", 5) == 0) {
            if (!parse_kv_type(a + 5, kv_type)) {
                std::fprintf(stderr, "unknown --kv type: %s\n", a + 5);
                return 2;
            }
        } else if (a[0] == '-') {
            std::fprintf(stderr, "unknown option: %s\n", a);
            return 2;
        } else {
            positional.push_back(a);
        }
    }

    if (positional.empty()) {
        std::fprintf(stderr, "missing laguna.gguf\n");
        return 2;
    }
    const int prompt_N = std::max(1, positional.size() >= 2 ? std::atoi(positional[1]) : 128);
    const int n_gen    = std::max(1, positional.size() >= 3 ? std::atoi(positional[2]) : 256);

    LagunaBackendArgs args;
    args.target_path    = positional[0];
    args.draft_path     = draft_path;
    args.draft_gpu      = draft_gpu;
    args.draft_ctx_max  = draft_ctx_max;
    args.ddtree_mode    = ddtree_mode;
    args.ddtree_budget  = ddtree_budget;
    args.ddtree_temp    = ddtree_temp > 0.0f ? ddtree_temp : 1.0f;
    args.verify_width   = verify_width;
    args.max_ctx        = max_ctx_arg > 0 ? max_ctx_arg : prompt_N + n_gen + 64;
    args.kv_type        = kv_type;

    std::printf("[spark-bench] target=%s draft=%s ddtree=%d budget=%d temp=%.2f "
                "verify_width=%d max_ctx=%d kv=%s\n",
                args.target_path.c_str(),
                args.draft_path.empty() ? "(none)" : args.draft_path.c_str(),
                (int)args.ddtree_mode, args.ddtree_budget, args.ddtree_temp,
                args.verify_width, args.max_ctx, ggml_type_name(args.kv_type));

    LagunaBackend be(args);
    if (!be.init()) {
        std::fprintf(stderr, "backend init failed\n");
        return 1;
    }
    be.print_ready_banner();

    // BOS + fake tokens (same seeding as bench_laguna_generate so the
    // routing trajectory is comparable across configs). DFLASH_BENCH_MIX=1
    // uses a deterministic varied prompt instead (non-degenerate continuation,
    // for exactness comparisons between decode paths).
    GenerateRequest req;
    req.prompt.resize((size_t)prompt_N, 1972);
    req.prompt[0] = 2;  // laguna bos
    if (std::getenv("DFLASH_BENCH_MIX")) {
        int64_t seed = 1;
        if (const char * s = std::getenv("DFLASH_BENCH_SEED")) seed = std::atoll(s);
        for (int i = 1; i < prompt_N; ++i)
            req.prompt[(size_t)i] = 1000 + (int32_t)((((int64_t)i + seed * 7919) * 2654435761LL) % 50000);
    }
    req.n_gen = n_gen;
    req.stream = false;

    DaemonIO io{};
    GenerateResult r = be.generate(req, io);
    if (!r.ok) {
        std::fprintf(stderr, "generate failed: %s\n", r.error.c_str());
        return 1;
    }
    const int nd = (int)r.tokens.size();
    std::printf("[spark-bench] prefill N=%d in %.3fs (%.1f tok/s)\n",
                prompt_N, r.prefill_s, prompt_N / std::max(1e-9, r.prefill_s));
    std::printf("[spark-bench] decoded %d tokens in %.3fs (%.1f tok/s)\n",
                nd, r.decode_s, nd / std::max(1e-9, r.decode_s));
    std::printf("[spark-bench] first ids:");
    for (int i = 0; i < nd && i < 16; ++i) std::printf(" %d", r.tokens[(size_t)i]);
    std::printf("\n");
    // FNV-1a over the full generated sequence: exactness fingerprint.
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < nd; ++i) {
        uint32_t v = (uint32_t)r.tokens[(size_t)i];
        for (int b = 0; b < 4; ++b) { h ^= (v >> (8*b)) & 0xff; h *= 1099511628211ULL; }
    }
    std::printf("[spark-bench] ids_hash=%016llx n=%d\n", (unsigned long long)h, nd);
    return 0;
}
