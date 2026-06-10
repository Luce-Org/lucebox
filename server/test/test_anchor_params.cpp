// Unit tests for resolve_anchor_params() — no GPU, no model files.
//
// Build: cmake --build build --target test_anchor_params -j
// Run:   ./build/test_anchor_params

#include "qwen3/anchor_params.h"

#include <cstdio>

using namespace dflash::common;

static int failures = 0;
static int checks   = 0;

#define CHECK_EQ(a, b) do { \
    checks++; \
    if ((a) != (b)) { \
        failures++; \
        std::fprintf(stderr, "  FAIL %s:%d: %d != %d\n", __FILE__, __LINE__, (a), (b)); \
    } \
} while (0)

static void test_tier_boundaries() {
    // tier 0: n_chunks < 1024 -> {2,8}
    { auto p = resolve_anchor_params(0,    -1,-1, -1,-1); CHECK_EQ(p.radius,2); CHECK_EQ(p.max_hits,8); }
    { auto p = resolve_anchor_params(1,    -1,-1, -1,-1); CHECK_EQ(p.radius,2); CHECK_EQ(p.max_hits,8); }
    { auto p = resolve_anchor_params(1023, -1,-1, -1,-1); CHECK_EQ(p.radius,2); CHECK_EQ(p.max_hits,8); }
    // tier 1: 1024 <= n_chunks < 2048 -> {4,16}
    { auto p = resolve_anchor_params(1024, -1,-1, -1,-1); CHECK_EQ(p.radius,4); CHECK_EQ(p.max_hits,16); }
    { auto p = resolve_anchor_params(2047, -1,-1, -1,-1); CHECK_EQ(p.radius,4); CHECK_EQ(p.max_hits,16); }
    // tier 2: n_chunks >= 2048 -> {8,32}
    { auto p = resolve_anchor_params(2048, -1,-1, -1,-1); CHECK_EQ(p.radius,8); CHECK_EQ(p.max_hits,32); }
    { auto p = resolve_anchor_params(4096, -1,-1, -1,-1); CHECK_EQ(p.radius,8); CHECK_EQ(p.max_hits,32); }
}

static void test_legacy_env_overrides_tier() {
    // legacy (DFLASH) env wins over tier default
    { auto p = resolve_anchor_params(500, -1,-1, 5, 20); CHECK_EQ(p.radius,5); CHECK_EQ(p.max_hits,20); }
    { auto p = resolve_anchor_params(2048, -1,-1, 1, 4); CHECK_EQ(p.radius,1); CHECK_EQ(p.max_hits,4); }
}

static void test_pflash_env_wins_over_legacy() {
    // PFLASH env (env_*) wins over legacy (legacy_*) which wins over tier
    { auto p = resolve_anchor_params(500, 10, 40, 5, 20); CHECK_EQ(p.radius,10); CHECK_EQ(p.max_hits,40); }
    { auto p = resolve_anchor_params(2048, 3, 12, 1, 4); CHECK_EQ(p.radius,3); CHECK_EQ(p.max_hits,12); }
}

static void test_sentinel_minus1_falls_through_to_tier() {
    // -1 means unset; both sentinels -> tier
    { auto p = resolve_anchor_params(1023, -1,-1, -1,-1); CHECK_EQ(p.radius,2); CHECK_EQ(p.max_hits,8); }
    { auto p = resolve_anchor_params(1024, -1,-1, -1,-1); CHECK_EQ(p.radius,4); CHECK_EQ(p.max_hits,16); }
}

static void test_zero_is_valid_override() {
    // 0 is a valid override (>= 0), not sentinel
    { auto p = resolve_anchor_params(2048, 0, 0, -1,-1); CHECK_EQ(p.radius,0); CHECK_EQ(p.max_hits,0); }
}

int main() {
    std::fprintf(stderr, "test_anchor_params\n");
    test_tier_boundaries();
    test_legacy_env_overrides_tier();
    test_pflash_env_wins_over_legacy();
    test_sentinel_minus1_falls_through_to_tier();
    test_zero_is_valid_override();
    if (failures == 0)
        std::fprintf(stdout, "PASS  %d checks\n", checks);
    else
        std::fprintf(stdout, "FAIL  %d/%d checks failed\n", failures, checks);
    return failures ? 1 : 0;
}
