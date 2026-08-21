// Pure unit test for kvflash_pool_from_env (kvflash_pager.h). No ggml, no GPU.
//
// Guards the MoE placement bug (PR #428): placement called pool sizing with
// bare max_ctx (no budget, scorer_expected=false) and got the max_ctx/2
// fallback, while runtime passed a real VRAM budget + scorer policy and got a
// speed-capped value. Placement then over-reserved KV and starved experts.
// These asserts pin the two behaviours so a future caller can't silently
// reintroduce the divergence.
#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"
#include "../src/common/kvflash_pager.h"

#include <cstdio>
#include <cstdlib>

using namespace dflash::common;

namespace {
struct KvflashPoolSizingFixture {};
}

TEST_CASE(KvflashPoolSizingFixture, kvflash_pool_sizing_suite) {
    REQUIRE(!kvflash_pool_requested(nullptr));
    REQUIRE(!kvflash_pool_requested("0"));
    REQUIRE(kvflash_pool_requested("auto"));
    REQUIRE(kvflash_pool_requested("4096"));

    REQUIRE(!kvflash_fixed_pool_requested(nullptr));
    REQUIRE(!kvflash_fixed_pool_requested("0"));
    REQUIRE(!kvflash_fixed_pool_requested("auto"));
    REQUIRE(kvflash_fixed_pool_requested("4096"));

    {
        const luce_test::ScopedEnvVar kvflash_off("DFLASH_KVFLASH", "0");
        REQUIRE(kvflash_pool_from_env(131072) == 0);
    }
    const luce_test::ScopedEnvVar kvflash_mode("DFLASH_KVFLASH", "auto");
    const luce_test::ScopedEnvVar max_pool("DFLASH_KVFLASH_MAX_POOL", nullptr);
    const int max_ctx = 131072;

    // No budget supplied -> fallback fraction of max_ctx (the buggy placement
    // path). scorer_expected toggles 1/2 vs 1/4.
    REQUIRE(kvflash_pool_from_env(max_ctx, KvFlashConfig{}, false) == max_ctx / 2);
    REQUIRE(kvflash_pool_from_env(max_ctx, KvFlashConfig{}, true) == max_ctx / 4);

    // Real budget with ample VRAM -> capped at the speed point (16384), far
    // below max_ctx/2. This is what runtime actually allocates; placement must
    // pass the same budget so it reserves this, not 65536.
    KvFlashAutoBudget budget;
    budget.free_bytes        = 12LL * 1024 * 1024 * 1024;  // 12 GiB free
    budget.reserve_bytes     = 1LL * 1024 * 1024 * 1024;
    budget.bytes_per_token   = 80 * 1024;                  // ~qwen35moe density
    budget.speed_cap_tokens  = 16384;
    const int with_budget = kvflash_pool_from_env(max_ctx, KvFlashConfig{}, true, budget);
    REQUIRE(with_budget == 16384);
    REQUIRE(with_budget < max_ctx / 2);

    // Tight VRAM -> budget binds below the cap, still well under max_ctx/2.
    budget.free_bytes = 2LL * 1024 * 1024 * 1024;          // 2 GiB free
    const int tight = kvflash_pool_from_env(max_ctx, KvFlashConfig{}, true, budget);
    REQUIRE(tight > 0 && tight <= 16384);

    std::printf("OK test_kvflash_pool_sizing\n");
}
