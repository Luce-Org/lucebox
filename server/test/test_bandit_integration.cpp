// Integration tests: adaptive bandit wired into HttpServer request path.

#include "CppUnitTestFramework.hpp"
#include "server/http_server.h"
#include "server/adaptive_keep_ratio.h"

#include <cmath>
#include <string>

using namespace luce::common;

namespace {
struct BanditIntegrationFixture {};
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

TEST_CASE(BanditIntegrationFixture, multi_turn_reaches_lower_bound) {
    HttpServerSessions sessions;
    for (int i = 0; i < 100; ++i) {
        sessions.update("s_hi", 1.0f);
    }
    float k = sessions.get_keep_ratio("s_hi");
    CHECK(k >= kBanditKeepMin - 1e-5f);
}

TEST_CASE(BanditIntegrationFixture, multi_turn_reaches_upper_bound) {
    HttpServerSessions sessions;
    for (int i = 0; i < 100; ++i) {
        sessions.update("s_lo", 0.0f);
    }
    float k = sessions.get_keep_ratio("s_lo");
    CHECK(k <= kBanditKeepMax + 1e-5f);
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
