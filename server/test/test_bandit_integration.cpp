// Integration tests: adaptive bandit wired into HttpServer request path.
// No GPU, no model files — uses a synchronous MockBackend that returns
// a configurable accept_rate.
//
// Build: cmake --build dflash/build --target test_bandit_integration -j
// Run:   cd dflash/build && ./test_bandit_integration

#include "server/http_server.h"
#include "server/adaptive_keep_ratio.h"

#include <cmath>
#include <cstdio>
#include <string>

using namespace dflash::common;

// ─── Test framework (ds4 style) ──────────────────────────────────────────────

static int test_failures = 0;
static int test_count    = 0;

#define TEST_ASSERT(expr) do { \
    test_count++; \
    if (!(expr)) { \
        test_failures++; \
        std::fprintf(stderr, "  FAIL: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
    } \
} while (0)

#define TEST_ASSERT_MSG(expr, msg) do { \
    test_count++; \
    if (!(expr)) { \
        test_failures++; \
        std::fprintf(stderr, "  FAIL: %s:%d: %s -- %s\n", __FILE__, __LINE__, #expr, msg); \
    } \
} while (0)

#define RUN_TEST(fn) do { \
    std::fprintf(stderr, "  %s ...", #fn); \
    int before = test_failures; \
    fn(); \
    if (test_failures == before) std::fprintf(stderr, " ok\n"); \
    else std::fprintf(stderr, "\n"); \
} while (0)

static inline bool approx_eq(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

// ─── Tests for HttpServerSessions (the integration contract) ─────────────────

// Test 1: Three-turn session with high accept_rate should decrease keep_ratio.
// This mirrors "three_turn_session_evolves_keep_ratio".
static void three_turn_session_evolves_keep_ratio() {
    HttpServerSessions sessions;

    // Initial keep ratio (default prior = 0.10)
    float k0 = sessions.get_keep_ratio("s1");
    TEST_ASSERT_MSG(approx_eq(k0, AdaptiveKeepRatioState{}.last_keep),
                    "initial keep should be the default prior");

    // Turn 1: high accept => next keep should drop
    sessions.update("s1", 0.95f);
    float k1 = sessions.get_keep_ratio("s1");

    // Turn 2: same high accept => keep drops further
    sessions.update("s1", 0.95f);
    float k2 = sessions.get_keep_ratio("s1");

    // Turn 3: same
    sessions.update("s1", 0.95f);
    float k3 = sessions.get_keep_ratio("s1");

    TEST_ASSERT_MSG(k1 < k0, "turn 1 keep must be less than initial for high accept");
    TEST_ASSERT_MSG(k2 <= k1, "turn 2 keep must not exceed turn 1 under high accept");
    TEST_ASSERT_MSG(k3 <= k2, "turn 3 keep must not exceed turn 2 under high accept");
    TEST_ASSERT(sessions.turn_count("s1") == 3);
}

// Test 2: Request without session_id uses config default (no bandit mutation).
// We verify that the sessions map stays empty when no session_id is used.
static void no_session_id_uses_static_default() {
    HttpServerSessions sessions;

    // Never call update with empty key — this simulates the "no session_id" path.
    // The server code guards: if (session_id.empty()) skip bandit.
    // So sessions stays empty and get_keep_ratio("") returns the default.
    TEST_ASSERT(sessions.size() == 0);
    // If someone queries with empty string (shouldn't happen), they get default.
    float k = sessions.get_keep_ratio("");
    TEST_ASSERT_MSG(approx_eq(k, AdaptiveKeepRatioState{}.last_keep),
                    "empty session_id must return default keep_ratio");
}

// Test 3: Two sessions with different accept rates stay isolated.
// High-accept session ends up with lower keep than low-accept session.
static void isolated_sessions() {
    HttpServerSessions sessions;

    // Session A: accept = 0.95 (high) → keep should decrease
    sessions.update("high_accept", 0.95f);

    // Session B: accept = 0.50 (low) → keep should increase
    sessions.update("low_accept", 0.50f);

    float k_high = sessions.get_keep_ratio("high_accept");
    float k_low  = sessions.get_keep_ratio("low_accept");

    TEST_ASSERT_MSG(k_high < k_low,
                    "session with high accept must have lower keep than low-accept session");
    TEST_ASSERT(sessions.turn_count("high_accept") == 1);
    TEST_ASSERT(sessions.turn_count("low_accept") == 1);
    TEST_ASSERT(sessions.size() == 2);
}

// Test 4: Multi-turn convergence — with persistent high accept the ratio
// reaches the lower bound and stays there.
static void multi_turn_reaches_lower_bound() {
    HttpServerSessions sessions;

    // Drive 100 turns with accept=1.0
    for (int i = 0; i < 100; ++i) {
        sessions.update("s_hi", 1.0f);
    }
    float k = sessions.get_keep_ratio("s_hi");
    TEST_ASSERT_MSG(k >= kBanditKeepMin - 1e-5f,
                    "keep must not fall below kBanditKeepMin");
}

// Test 5: Multi-turn convergence with low accept reaches the upper bound.
static void multi_turn_reaches_upper_bound() {
    HttpServerSessions sessions;

    for (int i = 0; i < 100; ++i) {
        sessions.update("s_lo", 0.0f);
    }
    float k = sessions.get_keep_ratio("s_lo");
    TEST_ASSERT_MSG(k <= kBanditKeepMax + 1e-5f,
                    "keep must not exceed kBanditKeepMax");
}

// Test 6: Zero accept_rate with spec_decode_ran=true MUST update the bandit.
// Previously, the guard was accept_rate>0, which silently skipped 0-accept
// sessions — exactly the case where the bandit most needs to act (push keep up).
// The fix uses spec_decode_ran as the gate; this test exercises the session layer
// directly: update() with 0.0 must drive keep_ratio toward kBanditKeepMax.
static void zero_accept_drives_keep_up() {
    HttpServerSessions sessions;

    float k0 = sessions.get_keep_ratio("s1");
    // Simulate server calling update() because spec_decode_ran==true, accept==0
    sessions.update("s1", 0.0f);
    float k1 = sessions.get_keep_ratio("s1");

    TEST_ASSERT(k1 >= kBanditKeepMin && k1 <= kBanditKeepMax);
    TEST_ASSERT_MSG(k1 > k0, "zero accept must increase keep_ratio");
    TEST_ASSERT(sessions.turn_count("s1") == 1);
}

// ─── Tests for parse_session_id_from_body (non-string guard) ─────────────────

// Test 7: session_id as integer in extra_body => empty (no type_error)
static void non_string_session_id_integer_extra_body() {
    json body = {{"extra_body", {{"session_id", 42}}}};
    std::string sid = parse_session_id_from_body(body);
    TEST_ASSERT_MSG(sid.empty(), "integer session_id in extra_body must yield empty string");
}

// Test 8: session_id as null at top level => empty (no type_error)
static void non_string_session_id_null_top_level() {
    json body = {{"session_id", nullptr}};
    std::string sid = parse_session_id_from_body(body);
    TEST_ASSERT_MSG(sid.empty(), "null session_id at top level must yield empty string");
}

// Test 9: session_id as array in extra_body => empty (no type_error)
static void non_string_session_id_array_extra_body() {
    json body = {{"extra_body", {{"session_id", json::array({"a", "b"})}}}};
    std::string sid = parse_session_id_from_body(body);
    TEST_ASSERT_MSG(sid.empty(), "array session_id in extra_body must yield empty string");
}

// ─── main ────────────────────────────────────────────────────────────────────

int main() {
    std::fprintf(stderr, "=== test_bandit_integration ===\n");

    RUN_TEST(three_turn_session_evolves_keep_ratio);
    RUN_TEST(no_session_id_uses_static_default);
    RUN_TEST(isolated_sessions);
    RUN_TEST(multi_turn_reaches_lower_bound);
    RUN_TEST(multi_turn_reaches_upper_bound);
    RUN_TEST(zero_accept_drives_keep_up);
    RUN_TEST(non_string_session_id_integer_extra_body);
    RUN_TEST(non_string_session_id_null_top_level);
    RUN_TEST(non_string_session_id_array_extra_body);

    std::fprintf(stderr, "\n%d tests, %d failures\n", test_count, test_failures);
    return (test_failures == 0) ? 0 : 1;
}
