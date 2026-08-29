#include "CppUnitTestFramework.hpp"
#include "common/adaptive_spec_width.h"

#include <cmath>

using namespace dflash::common;

namespace {
struct AdaptiveSpecWidthFixture {};

bool near(float lhs, float rhs, float tolerance = 1e-6f) {
    return std::fabs(lhs - rhs) <= tolerance;
}
} // namespace

TEST_CASE(AdaptiveSpecWidthFixture, starts_with_two_candidates) {
    AdaptiveSpecWidth width(16);
    CHECK(width.next_width() == 3);
}

TEST_CASE(AdaptiveSpecWidthFixture, clean_drafts_probe_upward) {
    AdaptiveSpecWidth width(8);
    CHECK(width.next_width() == 3);
    width.observe(3, 3);
    CHECK(width.next_width() == 4);
    width.observe(4, 4);
    CHECK(width.next_width() == 5);
}

TEST_CASE(AdaptiveSpecWidthFixture, censored_sample_is_not_averaged_down) {
    AdaptiveSpecWidth width(8, 2, true, 5.0f);
    width.observe(3, 3);
    CHECK(near(width.accepted_candidates_ema(), 6.0f));
    CHECK(width.next_width() == 7);
}

TEST_CASE(AdaptiveSpecWidthFixture, rejection_backs_off_gently) {
    AdaptiveSpecWidth width(8, 2, true, 4.0f);
    width.observe(2, 6); // one accepted candidate, then a rejection
    CHECK(near(width.accepted_candidates_ema(), 3.25f));
    CHECK(width.next_width() == 4);
}

TEST_CASE(AdaptiveSpecWidthFixture, respects_model_and_feedback_caps) {
    AdaptiveSpecWidth width(8, 2, true, 5.0f);
    CHECK(width.next_width(4) == 4);
    CHECK(width.next_width(99) == 6);
    CHECK(width.next_width(1) == 1);
}

TEST_CASE(AdaptiveSpecWidthFixture, respects_minimum_width) {
    AdaptiveSpecWidth width(16, 6, true, 0.0f);
    CHECK(width.next_width() == 6);
    width.observe(1, 6);
    CHECK(width.next_width() == 6);
}

TEST_CASE(AdaptiveSpecWidthFixture, disabled_controller_preserves_proposal) {
    AdaptiveSpecWidth width(16, 2, false);
    CHECK(width.next_width() == 16);
    CHECK(width.next_width(7) == 7);
    width.observe(1, 16);
    CHECK(width.next_width() == 16);
}

TEST_CASE(AdaptiveSpecWidthFixture, reset_restores_initial_estimate) {
    AdaptiveSpecWidth width(8);
    width.observe(1, 3);
    CHECK(!near(width.accepted_candidates_ema(), 2.0f));
    width.reset();
    CHECK(near(width.accepted_candidates_ema(), 2.0f));
    CHECK(width.next_width() == 3);
}
