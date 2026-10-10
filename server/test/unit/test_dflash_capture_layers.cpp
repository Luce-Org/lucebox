// Unit tests for luce::common::adopt_drafter_capture_layers — no GPU, no
// model files.

#include "CppUnitTestFramework.hpp"
#include "common/dflash_capture.h"
#include "luce.h"

#include <vector>

using namespace luce::common;

namespace {
struct DFlashCaptureLayersFixture {};
}

// z-lab Qwen3.5-9B DFlash captures 8 target layers: more than the 5-slot
// default, so the table must take the drafter's count, not keep 5.
TEST_CASE(DFlashCaptureLayersFixture, adopts_drafter_count_beyond_default) {
    int ids[LUCE_DRAFT_MAX_TARGET_LAYERS] = {1, 8, 15, 22, 29};
    int n = LUCE_DRAFT_N_TARGET_LAYERS;
    const std::vector<int> drafter = {1, 5, 9, 13, 17, 21, 25, 29};
    CHECK(adopt_drafter_capture_layers(drafter, /*target_n_layer=*/32,
                                       LUCE_DRAFT_MAX_TARGET_LAYERS, ids, n));
    CHECK(n == 8);
    CHECK(std::vector<int>(ids, ids + n) == drafter);
    CHECK(target_capture_index(ids, n, 29) == 7);
}

TEST_CASE(DFlashCaptureLayersFixture, rejects_invalid_lists_untouched) {
    int ids[LUCE_DRAFT_MAX_TARGET_LAYERS] = {1, 8, 15, 22, 29};
    int n = LUCE_DRAFT_N_TARGET_LAYERS;
    // Layer 32 does not exist in a 32-layer target.
    CHECK(!adopt_drafter_capture_layers({1, 5, 32}, 32,
                                        LUCE_DRAFT_MAX_TARGET_LAYERS, ids, n));
    // More layers than the table holds.
    CHECK(!adopt_drafter_capture_layers(std::vector<int>(3, 1), 32,
                                        /*max_slots=*/2, ids, n));
    CHECK(!adopt_drafter_capture_layers({}, 32,
                                        LUCE_DRAFT_MAX_TARGET_LAYERS, ids, n));
    CHECK(n == LUCE_DRAFT_N_TARGET_LAYERS);
    CHECK(ids[4] == 29);
}
