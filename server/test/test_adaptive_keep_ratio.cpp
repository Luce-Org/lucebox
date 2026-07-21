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

TEST_CASE(AdaptiveKeepRatioFixture, sessions_isolated) {
    HttpServerSessions mgr;
    mgr.update("s1", 0.90f);
    mgr.update("s2", 0.50f);
    float k1 = mgr.get_keep_ratio("s1");
    float k2 = mgr.get_keep_ratio("s2");
    CHECK(k1 < k2);
    CHECK(mgr.turn_count("s1") == 1);
    CHECK(mgr.turn_count("s2") == 1);
    CHECK(mgr.size() == 2);
}

TEST_CASE(AdaptiveKeepRatioFixture, unknown_session_returns_default) {
    HttpServerSessions mgr;
    float k = mgr.get_keep_ratio("no-such-session");
    CHECK(approx_eq(k, AdaptiveKeepRatioState{}.last_keep));
    CHECK(mgr.turn_count("no-such-session") == 0);
}

TEST_CASE(AdaptiveKeepRatioFixture, get_ema_reflects_post_update_value) {
    HttpServerSessions mgr;
    CHECK(approx_eq(mgr.get_ema("s1"), 0.0f));
    mgr.update("s1", 0.80f);
    CHECK(approx_eq(mgr.get_ema("s1"), 0.80f));
    mgr.update("s1", 0.60f);
    float expected = kBanditEmaAlpha * 0.80f + (1.0f - kBanditEmaAlpha) * 0.60f;
    CHECK(approx_eq(mgr.get_ema("s1"), expected));
}

TEST_CASE(AdaptiveKeepRatioFixture, lru_eviction_bounds_map_size) {
    HttpServerSessions mgr;

    const std::size_t over = kMaxSessions + 100;
    for (std::size_t i = 0; i < over; ++i) {
        mgr.update("sess-" + std::to_string(i), 0.80f);
    }
    CHECK(mgr.size() <= kMaxSessions);

    float k0 = mgr.get_keep_ratio("sess-0");
    CHECK(mgr.size() <= kMaxSessions);
    (void) k0;

    const std::string pinned = "sess-" + std::to_string(over - 1);
    for (int t = 0; t < 3; ++t) {
        mgr.update(pinned, 0.80f);
    }
    for (std::size_t i = over; i < over + 200; ++i) {
        mgr.update("wave2-" + std::to_string(i), 0.80f);
    }

    CHECK(mgr.size() <= kMaxSessions);
    CHECK(mgr.turn_count(pinned) >= 3);
}
