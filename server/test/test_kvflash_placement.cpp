// Unit test for the KVFlash placement KV-reservation decision (no GPU).
//
// Behaviour under test: when KVFlash is active, the MoE expert placement must
// reserve KV for the resident POOL, not max_ctx — otherwise a large max_ctx
// reservation forces experts cold even though KVFlash bounds the resident KV.
// The decision also reports whether the model is all-hot with the FULL max_ctx
// KV (i.e. KVFlash is redundant) so the gate can disable the pool when unneeded.
#include "../src/common/kvflash_placement.h"

#include <cstdio>
#include <cstdlib>

using namespace dflash::common;

static void expect(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

int main() {
    // qwen3.6-35B-A3B-like budget on a 24 GiB card:
    //   ~80 KiB/token KV  (5 GiB @ 65536, 10 GiB @ 131072)
    //   experts ~13.19 GiB, core ~3.12 GiB, draft ~1.2 GiB present.
    const uint64_t MiB = 1024ull * 1024;
    const uint64_t GiB = 1024ull * MiB;
    const uint64_t kv_per_tok = 80 * 1024;            // bytes/token
    const uint64_t gpu       = 24 * GiB;
    const uint64_t core      = 3 * GiB + 122 * MiB;   // ~3.12 GiB
    const uint64_t experts   = 13 * GiB + 195 * MiB;  // ~13.19 GiB
    const uint64_t warm      = 200 * MiB;
    const uint64_t safety    = 256 * MiB;             // reduced when draft present
    const uint64_t draft     = 1200 * MiB;

    // Case 1 — max_ctx 65536, NO kvflash: reserve full ctx, fits all-hot.
    {
        auto d = kvflash_placement_decision(kv_per_tok, 65536, /*pool=*/0,
                                            gpu, core, experts, warm, safety, draft);
        expect(d.kv_ctx == 65536, "C1: no-kvflash reserves full ctx");
        expect(d.all_hot_full_kv, "C1: 65536 full KV fits all experts hot");
        expect(!d.pool_reduced,   "C1: no pool reduction");
    }

    // Case 2 — max_ctx 65536 + pool 49152: full KV still fits all-hot, so KVFlash
    // is redundant — do NOT reduce to the pool (the gate will disable it).
    {
        auto d = kvflash_placement_decision(kv_per_tok, 65536, /*pool=*/49152,
                                            gpu, core, experts, warm, safety, draft);
        expect(d.all_hot_full_kv, "C2: 65536 full KV still fits -> kvflash redundant");
        expect(d.kv_ctx == 65536, "C2: keeps full ctx (gate disables kvflash)");
        expect(!d.pool_reduced,   "C2: no pool reduction when redundant");
    }

    // Case 3 (THE FIX) — max_ctx 131072 + pool 49152: full KV (10 GiB) forces
    // experts cold, so reserve for the POOL -> experts stay hot.
    {
        auto d = kvflash_placement_decision(kv_per_tok, 131072, /*pool=*/49152,
                                            gpu, core, experts, warm, safety, draft);
        expect(!d.all_hot_full_kv, "C3: 131072 full KV forces experts cold");
        expect(d.kv_ctx == 49152,  "C3: reserves POOL ctx, not max_ctx");
        expect(d.pool_reduced,     "C3: pool reduction engaged");
        expect(d.kv_total == kv_per_tok * 49152ull, "C3: kv_total sized to pool");
        expect(d.kv_total < kv_per_tok * 131072ull,  "C3: pool reservation < full");
    }

    // Case 4 — max_ctx 131072, NO kvflash: full ctx, cold cliff.
    {
        auto d = kvflash_placement_decision(kv_per_tok, 131072, /*pool=*/0,
                                            gpu, core, experts, warm, safety, draft);
        expect(d.kv_ctx == 131072, "C4: no-kvflash reserves full ctx");
        expect(!d.all_hot_full_kv, "C4: 131072 no-kvflash -> cold cliff");
    }

    // Case 5 — pool >= max_ctx: pool can't exceed ctx, no reduction.
    {
        auto d = kvflash_placement_decision(kv_per_tok, 32768, /*pool=*/49152,
                                            gpu, core, experts, warm, safety, draft);
        expect(!d.pool_reduced,    "C5: pool>=ctx -> no reduction");
        expect(d.kv_ctx == 32768,  "C5: keeps full ctx");
    }

    std::printf("PASS: kvflash placement decision (5 cases)\n");
    return 0;
}
