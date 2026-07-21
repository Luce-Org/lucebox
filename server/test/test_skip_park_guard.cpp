// Unit tests for skip_park_allowed — pure, GPU-free.

#include "CppUnitTestFramework.hpp"
#include "placement/skip_park_guard.h"

namespace {
struct SkipParkGuardFixture {};
}

static constexpr size_t GiB = 1024ull * 1024 * 1024;

TEST_CASE(SkipParkGuardFixture, T1_not_requested_stays_off) {
    CHECK(!dflash::common::skip_park_allowed(false, 24 * GiB, 32768));
}

TEST_CASE(SkipParkGuardFixture, T2_big_card_any_ctx) {
    CHECK(dflash::common::skip_park_allowed(true, 32 * GiB, 131072));
}

TEST_CASE(SkipParkGuardFixture, T3_small_card_small_ctx_allowed) {
    CHECK(dflash::common::skip_park_allowed(true, 24 * GiB, 65536));
}

TEST_CASE(SkipParkGuardFixture, T4_small_card_big_ctx_downgraded) {
    CHECK(!dflash::common::skip_park_allowed(true, 24 * GiB, 131072));
}

TEST_CASE(SkipParkGuardFixture, T5_boundary_ctx_one_over) {
    CHECK(!dflash::common::skip_park_allowed(true, 24 * GiB, 65537));
}

TEST_CASE(SkipParkGuardFixture, T6_boundary_vram_just_under_32g) {
    CHECK(!dflash::common::skip_park_allowed(true, 32 * GiB - 1, 131072));
}
