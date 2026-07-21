// Unit tests for dflash::common::compute_score_range().
// SCORE_LAYERS is relative to fwd_layer_limit: ee7+sl7 → [0,7), not phantom-empty [7,7).

#include "CppUnitTestFramework.hpp"
#include "score_range.h"

#include <cstdio>
#include <cstdlib>

using dflash::common::ScoreRange;
using dflash::common::compute_score_range;

namespace {
struct DrafterEarlyExitScoreRangeFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;

    void t1_bug_scenario() {
        ScoreRange r = compute_score_range(/*n_layer=*/28,
                                           /*score_layers=*/7,
                                           /*fwd_layer_limit=*/7);
        REQUIRE(r.start == 0 && "score_layer_start must be 0");
        REQUIRE(r.end   == 7 && "score_layer_end must equal fwd_layer_limit");
        REQUIRE(!r.empty()   && "range must be non-empty");
        REQUIRE(r.count() == 7);
        printf("T1 pass: early_exit_n=7 score_layers=7 n_layer=28 -> [%d,%d)\n",
               r.start, r.end);
    }

    void t2_no_early_exit() {
        ScoreRange r = compute_score_range(28, 7, 28);
        REQUIRE(r.start == 21);
        REQUIRE(r.end   == 28);
        REQUIRE(!r.empty());
        REQUIRE(r.count() == 7);
        printf("T2 pass: no early exit score_layers=7 -> [%d,%d)\n", r.start, r.end);
    }

    void t3_all_layers_no_exit() {
        ScoreRange r = compute_score_range(28, -1, 28);
        REQUIRE(r.start == 0);
        REQUIRE(r.end   == 28);
        REQUIRE(!r.empty());
        printf("T3 pass: score_layers=-1 no exit -> [%d,%d)\n", r.start, r.end);
    }

    void t4_all_layers_with_exit() {
        ScoreRange r = compute_score_range(28, -1, 14);
        REQUIRE(r.start == 0);
        REQUIRE(r.end   == 14);
        REQUIRE(!r.empty());
        printf("T4 pass: score_layers=-1 early_exit=14 -> [%d,%d)\n", r.start, r.end);
    }

    void t5_score_layers_exceeds_exit() {
        ScoreRange r = compute_score_range(28, 14, 7);
        REQUIRE(r.start == 0);
        REQUIRE(r.end   == 7);
        REQUIRE(!r.empty());
        printf("T5 pass: score_layers=14 early_exit=7 -> [%d,%d)\n", r.start, r.end);
    }

    void t6_score_layers_equals_n_layer() {
        ScoreRange r = compute_score_range(28, 28, 28);
        REQUIRE(r.start == 0);
        REQUIRE(r.end   == 28);
        REQUIRE(!r.empty());
        printf("T6 pass: score_layers=n_layer=28 -> [%d,%d)\n", r.start, r.end);
    }

    void t7_partial_exit_partial_score() {
        ScoreRange r = compute_score_range(28, 7, 14);
        REQUIRE(r.start == 7);
        REQUIRE(r.end   == 14);
        REQUIRE(!r.empty());
        REQUIRE(r.count() == 7);
        printf("T7 pass: early_exit=14 score_layers=7 -> [%d,%d)\n", r.start, r.end);
    }
};
}

TEST_CASE(DrafterEarlyExitScoreRangeFixture, score_range_suite) {
    t1_bug_scenario();
    t2_no_early_exit();
    t3_all_layers_no_exit();
    t4_all_layers_with_exit();
    t5_score_layers_exceeds_exit();
    t6_score_layers_equals_n_layer();
    t7_partial_exit_partial_score();
    printf("\nAll score_range tests passed.\n");
}
