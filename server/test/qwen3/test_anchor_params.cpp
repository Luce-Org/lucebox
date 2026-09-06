// Unit tests for resolve_anchor_params() — no GPU, no model files.

#include "CppUnitTestFramework.hpp"
#include "qwen3/anchor_params.h"

using namespace dflash::common;

namespace {
struct AnchorParamsFixture {};
}

TEST_CASE(AnchorParamsFixture, test_tier_boundaries) {
    // tier 0: n_chunks < 1024 -> {2,8}
    { auto p = resolve_anchor_params(0,    -1,-1, -1,-1); CHECK_EQUAL(p.radius, 2); CHECK_EQUAL(p.max_hits, 8); }
    { auto p = resolve_anchor_params(1,    -1,-1, -1,-1); CHECK_EQUAL(p.radius, 2); CHECK_EQUAL(p.max_hits, 8); }
    { auto p = resolve_anchor_params(1023, -1,-1, -1,-1); CHECK_EQUAL(p.radius, 2); CHECK_EQUAL(p.max_hits, 8); }
    // tier 1: 1024 <= n_chunks < 2048 -> {4,16}
    { auto p = resolve_anchor_params(1024, -1,-1, -1,-1); CHECK_EQUAL(p.radius, 4); CHECK_EQUAL(p.max_hits, 16); }
    { auto p = resolve_anchor_params(2047, -1,-1, -1,-1); CHECK_EQUAL(p.radius, 4); CHECK_EQUAL(p.max_hits, 16); }
    // tier 2: n_chunks >= 2048 -> {8,32}
    { auto p = resolve_anchor_params(2048, -1,-1, -1,-1); CHECK_EQUAL(p.radius, 8); CHECK_EQUAL(p.max_hits, 32); }
    { auto p = resolve_anchor_params(4096, -1,-1, -1,-1); CHECK_EQUAL(p.radius, 8); CHECK_EQUAL(p.max_hits, 32); }
}

TEST_CASE(AnchorParamsFixture, test_legacy_env_overrides_tier) {
    // legacy (DFLASH) env wins over tier default
    { auto p = resolve_anchor_params(500, -1,-1, 5, 20); CHECK_EQUAL(p.radius, 5); CHECK_EQUAL(p.max_hits, 20); }
    { auto p = resolve_anchor_params(2048, -1,-1, 1, 4); CHECK_EQUAL(p.radius, 1); CHECK_EQUAL(p.max_hits, 4); }
}

TEST_CASE(AnchorParamsFixture, test_pflash_env_wins_over_legacy) {
    // PFLASH env (env_*) wins over legacy (legacy_*) which wins over tier
    { auto p = resolve_anchor_params(500, 10, 40, 5, 20); CHECK_EQUAL(p.radius, 10); CHECK_EQUAL(p.max_hits, 40); }
    { auto p = resolve_anchor_params(2048, 3, 12, 1, 4); CHECK_EQUAL(p.radius, 3); CHECK_EQUAL(p.max_hits, 12); }
}

TEST_CASE(AnchorParamsFixture, test_sentinel_minus1_falls_through_to_tier) {
    // -1 means unset; both sentinels -> tier
    { auto p = resolve_anchor_params(1023, -1,-1, -1,-1); CHECK_EQUAL(p.radius, 2); CHECK_EQUAL(p.max_hits, 8); }
    { auto p = resolve_anchor_params(1024, -1,-1, -1,-1); CHECK_EQUAL(p.radius, 4); CHECK_EQUAL(p.max_hits, 16); }
}

TEST_CASE(AnchorParamsFixture, test_zero_is_valid_override) {
    // 0 is a valid override (>= 0), not sentinel
    { auto p = resolve_anchor_params(2048, 0, 0, -1,-1); CHECK_EQUAL(p.radius, 0); CHECK_EQUAL(p.max_hits, 0); }
}
