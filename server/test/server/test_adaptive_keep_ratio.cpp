// Unit tests for AdaptiveKeepRatioState + HttpServerSessions — no GPU, no model files.

#include "CppUnitTestFramework.hpp"
#include "server/adaptive_keep_ratio.h"

#include <cmath>
#include <string>

using namespace dflash::common;

namespace {
struct AdaptiveKeepRatioFixture {};
}

static inline bool approx_eq(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

TEST_CASE(AdaptiveKeepRatioFixture, default_construction) {
    AdaptiveKeepRatioState s{};
    CHECK(approx_eq(s.ema, 0.0f));
    CHECK(approx_eq(s.last_keep, 0.10f));
    CHECK(s.turn_count == 0);
}

TEST_CASE(AdaptiveKeepRatioFixture, first_turn_sets_ema_to_observed) {
    AdaptiveKeepRatioState s{};
    auto next = step_adaptive_keep_ratio(s, 0.82f);
    CHECK(approx_eq(next.ema, 0.82f));
    CHECK(next.turn_count == 1);
}

TEST_CASE(AdaptiveKeepRatioFixture, high_accept_decreases_keep) {
    AdaptiveKeepRatioState s{};
    s.turn_count = 1;
    s.ema = 0.88f;
    s.last_keep = 0.10f;
    auto next = step_adaptive_keep_ratio(s, 0.88f);
    CHECK(next.last_keep < s.last_keep);
}

TEST_CASE(AdaptiveKeepRatioFixture, low_accept_increases_keep) {
    AdaptiveKeepRatioState s{};
    s.turn_count = 1;
    s.ema = 0.65f;
    s.last_keep = 0.10f;
    auto next = step_adaptive_keep_ratio(s, 0.65f);
    CHECK(next.last_keep > s.last_keep);
}

TEST_CASE(AdaptiveKeepRatioFixture, in_band_no_change) {
    AdaptiveKeepRatioState s{};
    s.turn_count = 1;
    s.ema = 0.80f;
    s.last_keep = 0.10f;
    auto next = step_adaptive_keep_ratio(s, 0.80f);
    CHECK(approx_eq(next.last_keep, s.last_keep));
}

TEST_CASE(AdaptiveKeepRatioFixture, respects_lower_bound) {
    AdaptiveKeepRatioState s{};
    s.turn_count = 5;
    s.ema = 0.95f;
    s.last_keep = kBanditKeepMin;
    auto next = step_adaptive_keep_ratio(s, 0.99f);
    CHECK(approx_eq(next.last_keep, kBanditKeepMin));
}

TEST_CASE(AdaptiveKeepRatioFixture, respects_upper_bound) {
    AdaptiveKeepRatioState s{};
    s.turn_count = 5;
    s.ema = 0.40f;
    s.last_keep = kBanditKeepMax;
    auto next = step_adaptive_keep_ratio(s, 0.40f);
    CHECK(approx_eq(next.last_keep, kBanditKeepMax));
}

TEST_CASE(AdaptiveKeepRatioFixture, ten_turn_convergence_high_accept) {
    AdaptiveKeepRatioState s{};
    float prev_keep = s.last_keep;
    bool monotone = true;
    for (int i = 0; i < 10; ++i) {
        s = step_adaptive_keep_ratio(s, 0.90f);
        if (s.last_keep > prev_keep + 1e-6f) {
            monotone = false;
            break;
        }
        prev_keep = s.last_keep;
    }
    CHECK(monotone);
    CHECK(s.last_keep < 0.10f);
}

TEST_CASE(AdaptiveKeepRatioFixture, escalation_far_outside_band) {
    AdaptiveKeepRatioState s{};
    s.turn_count = 1;
    s.ema = 0.92f;
    s.last_keep = 0.10f;
    auto next = step_adaptive_keep_ratio(s, 0.92f);
    float drop = s.last_keep - next.last_keep;
    CHECK(approx_eq(drop, kBanditStepLarge, 1e-4f));
}
