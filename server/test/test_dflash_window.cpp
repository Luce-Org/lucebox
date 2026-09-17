#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"
#include "../src/common/dflash_feature_ring.h"

namespace {
struct DflashWindowFixture {};
}

TEST_CASE(DflashWindowFixture, trained_window_capped_by_max_ctx) {
    using dflash::common::dflash_drafter_window;
    using dflash::common::DFLASH_DRAFTER_TRAINED_CTX;
    luce_test::ScopedEnvVar ring("DFLASH_FEAT_RING_CAP", nullptr);
    CHECK(dflash_drafter_window(131072) == DFLASH_DRAFTER_TRAINED_CTX);
    CHECK(dflash_drafter_window(16384) == 16384);
}

TEST_CASE(DflashWindowFixture, env_valve_lowers_the_window) {
    using dflash::common::dflash_drafter_window;
    using dflash::common::DFLASH_DRAFTER_TRAINED_CTX;
    {
        luce_test::ScopedEnvVar ring("DFLASH_FEAT_RING_CAP", "8192");
        CHECK(dflash_drafter_window(131072) == 8192);
        CHECK(dflash_drafter_window(4096) == 4096);   // max_ctx still clamps
    }
    // unset/0/negative/greater-than-trained all fall back to the trained window
    const char * const fallback_values[] = {nullptr, "0", "-5", "999999"};
    for (const char * v : fallback_values) {
        luce_test::ScopedEnvVar ring("DFLASH_FEAT_RING_CAP", v);
        CHECK(dflash_drafter_window(131072) == DFLASH_DRAFTER_TRAINED_CTX);
    }
}
