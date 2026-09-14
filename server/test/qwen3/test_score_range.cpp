#include "CppUnitTestFramework.hpp"
#include "score_range.h"

using dflash::common::compute_score_range;

namespace {
struct ScoreRangeFixture {};
}

TEST_CASE(ScoreRangeFixture, scoring_uses_last_layers_actually_forwarded) {
    // Includes ee7+sl7: scoring must not become the phantom-empty [7,7).
    const struct { int layers, score, forwarded, start, end; } cases[] = {
        {28, 7, 7, 0, 7},
        {28, 7, 28, 21, 28},
        {28, 7, 14, 7, 14},
        {28, 14, 7, 0, 7},
        {28, -1, 28, 0, 28},
        {28, -1, 14, 0, 14},
        {28, 28, 28, 0, 28},
        {28, 29, 14, 0, 14},
        {28, 0, 14, 0, 14},
        {1, 1, 1, 0, 1},
    };
    for (const auto & c : cases) {
        const auto range = compute_score_range(c.layers, c.score, c.forwarded);
        CHECK(range.start == c.start);
        CHECK(range.end == c.end);
        CHECK(range.count() == c.end - c.start);
        CHECK(!range.empty());
    }
}

TEST_CASE(ScoreRangeFixture, no_forwarded_layers_has_no_scores) {
    for (int score : {-1, 0, 7, 28}) {
        const auto range = compute_score_range(28, score, 0);
        CHECK(range.start == 0);
        CHECK(range.end == 0);
        CHECK(range.count() == 0);
        CHECK(range.empty());
    }
    CHECK(compute_score_range(0, -1, 0).empty());
}
