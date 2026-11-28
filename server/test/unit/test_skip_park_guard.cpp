// Unit tests for skip_park_allowed — pure, GPU-free.

#include "CppUnitTestFramework.hpp"
#include "placement/skip_park_guard.h"

namespace {
struct SkipParkGuardFixture {};
}

static constexpr size_t GiB = 1024ull * 1024 * 1024;

TEST_CASE(SkipParkGuardFixture, T1_not_requested_stays_off) {
    CHECK(!luce::common::skip_park_allowed(false, 24 * GiB, 32768));
}

TEST_CASE(SkipParkGuardFixture, T2_big_card_any_ctx) {
    CHECK(luce::common::skip_park_allowed(true, 32 * GiB, 131072));
    // Real 32 GiB GPUs report ~31.84 GiB (RTX 5090: 32607 MiB, R9700: 32624 MiB).
    CHECK(luce::common::skip_park_allowed(true, 32607ull * 1024 * 1024, 131072));
}

TEST_CASE(SkipParkGuardFixture, T3_small_card_small_ctx_allowed) {
    CHECK(luce::common::skip_park_allowed(true, 24 * GiB, 65536));
}

TEST_CASE(SkipParkGuardFixture, T5_boundary_ctx_one_over) {
    CHECK(!luce::common::skip_park_allowed(true, 24 * GiB, 65537));
}

TEST_CASE(SkipParkGuardFixture, T6_boundary_vram_just_under_30g) {
    CHECK(luce::common::skip_park_allowed(true, 30 * GiB, 131072));
    CHECK(!luce::common::skip_park_allowed(true, 30 * GiB - 1, 131072));
}
