// Unit tests for session state, eviction, and request session-ID parsing.

#include "CppUnitTestFramework.hpp"
#include "server/http_server.h"
#include "server/adaptive_keep_ratio.h"

#include <cmath>
#include <string>

using namespace dflash::common;

namespace {
struct BanditIntegrationFixture {};
struct AdaptiveKeepRatioFixture {};
}

static inline bool approx_eq(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

TEST_CASE(BanditIntegrationFixture, three_turn_session_evolves_keep_ratio) {
    HttpServerSessions sessions;
    float k0 = sessions.get_keep_ratio("s1");
    CHECK(approx_eq(k0, AdaptiveKeepRatioState{}.last_keep));

    sessions.update("s1", 0.95f);
    float k1 = sessions.get_keep_ratio("s1");
    sessions.update("s1", 0.95f);
    float k2 = sessions.get_keep_ratio("s1");
    sessions.update("s1", 0.95f);
    float k3 = sessions.get_keep_ratio("s1");

    CHECK(k1 < k0);
    CHECK(k2 <= k1);
    CHECK(k3 <= k2);
    CHECK(sessions.turn_count("s1") == 3);
}

TEST_CASE(BanditIntegrationFixture, no_session_id_uses_static_default) {
    HttpServerSessions sessions;
    CHECK(sessions.size() == 0);
    float k = sessions.get_keep_ratio("");
    CHECK(approx_eq(k, AdaptiveKeepRatioState{}.last_keep));
}

TEST_CASE(BanditIntegrationFixture, isolated_sessions) {
    HttpServerSessions sessions;
    sessions.update("high_accept", 0.95f);
    sessions.update("low_accept", 0.50f);

    float k_high = sessions.get_keep_ratio("high_accept");
    float k_low = sessions.get_keep_ratio("low_accept");
    CHECK(k_high < k_low);
    CHECK(sessions.turn_count("high_accept") == 1);
    CHECK(sessions.turn_count("low_accept") == 1);
    CHECK(sessions.size() == 2);
}

TEST_CASE(BanditIntegrationFixture, multi_turn_reaches_lower_bound) {
    HttpServerSessions sessions;
    for (int i = 0; i < 100; ++i) {
        sessions.update("s_hi", 1.0f);
    }
    float k = sessions.get_keep_ratio("s_hi");
    CHECK(approx_eq(k, kBanditKeepMin));
}

TEST_CASE(BanditIntegrationFixture, multi_turn_reaches_upper_bound) {
    HttpServerSessions sessions;
    for (int i = 0; i < 100; ++i) {
        sessions.update("s_lo", 0.0f);
    }
    float k = sessions.get_keep_ratio("s_lo");
    CHECK(approx_eq(k, kBanditKeepMax));
}

TEST_CASE(BanditIntegrationFixture, zero_accept_drives_keep_up) {
    HttpServerSessions sessions;
    float k0 = sessions.get_keep_ratio("s1");
    sessions.update("s1", 0.0f);
    float k1 = sessions.get_keep_ratio("s1");

    CHECK(k1 >= kBanditKeepMin && k1 <= kBanditKeepMax);
    CHECK(k1 > k0);
    CHECK(sessions.turn_count("s1") == 1);
}

TEST_CASE(BanditIntegrationFixture, non_string_session_id_integer_extra_body) {
    json body = {{"extra_body", {{"session_id", 42}}}};
    std::string sid = parse_session_id_from_body(body);
    CHECK(sid.empty());
}

TEST_CASE(BanditIntegrationFixture, non_string_session_id_null_top_level) {
    json body = {{"session_id", nullptr}};
    std::string sid = parse_session_id_from_body(body);
    CHECK(sid.empty());
}

TEST_CASE(BanditIntegrationFixture, non_string_session_id_array_extra_body) {
    json body = {{"extra_body", {{"session_id", json::array({"a", "b"})}}}};
    std::string sid = parse_session_id_from_body(body);
    CHECK(sid.empty());
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
    CHECK(approx_eq(mgr.get_ema("s1"), 0.74f));
}

TEST_CASE(AdaptiveKeepRatioFixture, lru_eviction_bounds_map_size) {
    HttpServerSessions mgr;

    const std::size_t over = kMaxSessions + 100;
    for (std::size_t i = 0; i < over; ++i) {
        mgr.update("sess-" + std::to_string(i), 0.80f);
    }
    CHECK(mgr.size() == kMaxSessions);

    float k0 = mgr.get_keep_ratio("sess-0");
    CHECK(mgr.size() == kMaxSessions);
    CHECK(approx_eq(k0, AdaptiveKeepRatioState{}.last_keep));
    CHECK(mgr.turn_count("sess-0") == 0);

    const std::string pinned = "sess-" + std::to_string(over - 1);
    for (int t = 0; t < 3; ++t) {
        mgr.update(pinned, 0.80f);
    }
    for (std::size_t i = over; i < over + 200; ++i) {
        mgr.update("wave2-" + std::to_string(i), 0.80f);
    }

    CHECK(mgr.size() == kMaxSessions);
    CHECK(mgr.turn_count(pinned) == 4);
}

TEST_CASE(AdaptiveKeepRatioFixture, each_read_accessor_refreshes_lru) {
    for (int accessor = 0; accessor < 3; ++accessor) {
        HttpServerSessions sessions;
        for (size_t i = 0; i < kMaxSessions; ++i) {
            sessions.update(std::to_string(i), 0.8f);
        }
        // Touch the oldest entry through each public read path, then force
        // eviction. The next-oldest entry must go, not the one just read.
        if (accessor == 0) sessions.get_keep_ratio("0");
        if (accessor == 1) sessions.get_ema("0");
        if (accessor == 2) sessions.turn_count("0");
        sessions.update("new", 0.8f);
        CHECK(sessions.size() == kMaxSessions);
        CHECK(sessions.turn_count("0") == 1);
        CHECK(sessions.turn_count("1") == 0);
        CHECK(sessions.turn_count("new") == 1);
    }
}

TEST_CASE(AdaptiveKeepRatioFixture, missing_reads_do_not_allocate_sessions) {
    HttpServerSessions sessions;
    CHECK(approx_eq(sessions.get_keep_ratio("missing"), AdaptiveKeepRatioState{}.last_keep));
    CHECK(sessions.get_ema("missing") == 0.0f);
    CHECK(sessions.turn_count("missing") == 0);
    CHECK(sessions.size() == 0);
}
