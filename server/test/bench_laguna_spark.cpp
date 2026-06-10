// Laguna Spark decode bench: drives LagunaBackend (the REAL hybrid/spark
// path used by dflash_server), honoring all DFLASH_* env knobs:
//   DFLASH_LAGUNA_HOTNESS=<csv>     calibrated placement
//   DFLASH_EXPERT_BUDGET_PCT=60     pinned-hot fraction
//   DFLASH_LAGUNA_CACHE_SLOTS=16    cache ring slots/layer
//   DFLASH_LAGUNA_PROFILE=1         cold-experts/token profiling
//   DFLASH_LAGUNA_NO_SINGLE_GRAPH=1 per-layer fallback (for trace capture)
//   DFLASH_LAGUNA_PREGATE_TRACE=<f> pregate trace capture (fallback path)
//
// Usage: bench_laguna_spark <laguna.gguf> [prompt_N=128] [n_gen=256]

#include "laguna_backend.h"
#include "dflash27b.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dflash::common;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <laguna.gguf> [prompt_N=128] [n_gen=256]\n", argv[0]);
        return 2;
    }
    const int prompt_N = std::max(1, (argc >= 3) ? std::atoi(argv[2]) : 128);
    const int n_gen    = std::max(1, (argc >= 4) ? std::atoi(argv[3]) : 256);

    LagunaBackendArgs args;
    args.target_path = argv[1];
    args.max_ctx     = prompt_N + n_gen + 64;

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
