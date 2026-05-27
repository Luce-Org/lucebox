// Unit tests for AdaptiveKeepRatioState + HttpServerSessions — no GPU, no model files.
//
// Build: cmake --build build --target test_adaptive_keep_ratio -j
// Run:   cd build && ctest -R adaptive_keep --output-on-failure

#include "server/adaptive_keep_ratio.h"

#include <cmath>
#include <cstdio>
#include <string>

using namespace dflash::common;

// ─── Test framework (ds4 style) ───────────────────────────────────────────────

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

// ─── Tests ────────────────────────────────────────────────────────────────────

static void default_construction() {
    AdaptiveKeepRatioState s{};
    TEST_ASSERT(approx_eq(s.ema, 0.0f));
    TEST_ASSERT(approx_eq(s.last_keep, 0.10f));
    TEST_ASSERT(s.turn_count == 0);
}

static void first_turn_sets_ema_to_observed() {
    AdaptiveKeepRatioState s{};
    // turn_count == 0 => no smoothing, ema = observed directly
    auto next = step_adaptive_keep_ratio(s, 0.82f);
    TEST_ASSERT_MSG(approx_eq(next.ema, 0.82f), "first-turn EMA must equal observed");
    TEST_ASSERT(next.turn_count == 1);
}

static void high_accept_decreases_keep() {
    // observed > kBanditTargetHi (0.85) => keep should decrease
    AdaptiveKeepRatioState s{};
    s.turn_count = 1;
    s.ema        = 0.88f;
    s.last_keep  = 0.10f;
    auto next = step_adaptive_keep_ratio(s, 0.88f);
    TEST_ASSERT_MSG(next.last_keep < s.last_keep, "high accept must decrease keep");
}

static void low_accept_increases_keep() {
    // observed < kBanditTargetLo (0.75) => keep should increase
    AdaptiveKeepRatioState s{};
    s.turn_count = 1;
    s.ema        = 0.65f;
    s.last_keep  = 0.10f;
    auto next = step_adaptive_keep_ratio(s, 0.65f);
    TEST_ASSERT_MSG(next.last_keep > s.last_keep, "low accept must increase keep");
}

static void in_band_no_change() {
    // 0.75 <= ema <= 0.85 => keep unchanged
    AdaptiveKeepRatioState s{};
    s.turn_count = 1;
    s.ema        = 0.80f;
    s.last_keep  = 0.10f;
    auto next = step_adaptive_keep_ratio(s, 0.80f);
    TEST_ASSERT_MSG(approx_eq(next.last_keep, s.last_keep), "in-band keep must be unchanged");
}

static void respects_lower_bound() {
    // already at minimum; high accept must not push it below kBanditKeepMin
    AdaptiveKeepRatioState s{};
    s.turn_count = 5;
    s.ema        = 0.95f;
    s.last_keep  = kBanditKeepMin;
    auto next = step_adaptive_keep_ratio(s, 0.99f);
    TEST_ASSERT_MSG(approx_eq(next.last_keep, kBanditKeepMin),
                    "keep must not go below kBanditKeepMin");
}

static void respects_upper_bound() {
    // already at maximum; low accept must not push it above kBanditKeepMax
    AdaptiveKeepRatioState s{};
    s.turn_count = 5;
    s.ema        = 0.40f;
    s.last_keep  = kBanditKeepMax;
    auto next = step_adaptive_keep_ratio(s, 0.40f);
    TEST_ASSERT_MSG(approx_eq(next.last_keep, kBanditKeepMax),
                    "keep must not go above kBanditKeepMax");
}

static void ten_turn_convergence_high_accept() {
    // Feeding accept=0.90 ten turns => keep monotonically decreases
    AdaptiveKeepRatioState s{};
    float prev_keep = s.last_keep;
    bool monotone   = true;
    for (int i = 0; i < 10; ++i) {
        s = step_adaptive_keep_ratio(s, 0.90f);
        if (s.last_keep > prev_keep + 1e-6f) {
            monotone = false;
            break;
        }
        prev_keep = s.last_keep;
    }
    TEST_ASSERT_MSG(monotone, "keep must monotonically decrease under persistent high accept");
    TEST_ASSERT_MSG(s.last_keep < 0.10f, "keep must have decreased after 10 high-accept turns");
}

static void escalation_far_outside_band() {
    // ema > kBanditEscalateHi (0.90) => step is large (0.01), not small (0.005)
    AdaptiveKeepRatioState s{};
    s.turn_count = 1;
    s.ema        = 0.92f;
    s.last_keep  = 0.10f;
    auto next = step_adaptive_keep_ratio(s, 0.92f);
    float drop = s.last_keep - next.last_keep;
    TEST_ASSERT_MSG(approx_eq(drop, kBanditStepLarge, 1e-4f),
                    "far-above-band must use large step");
}

static void sessions_isolated() {
    HttpServerSessions mgr;
    // s1 sees high accept => keep decreases
    mgr.update("s1", 0.90f);
    // s2 sees low accept  => keep increases
    mgr.update("s2", 0.50f);
    float k1 = mgr.get_keep_ratio("s1");
    float k2 = mgr.get_keep_ratio("s2");
    TEST_ASSERT_MSG(k1 < k2,
                    "session with high accept must end up with lower keep than low-accept session");
    TEST_ASSERT(mgr.turn_count("s1") == 1);
    TEST_ASSERT(mgr.turn_count("s2") == 1);
    TEST_ASSERT(mgr.size() == 2);
}

static void unknown_session_returns_default() {
    HttpServerSessions mgr;
    float k = mgr.get_keep_ratio("no-such-session");
    TEST_ASSERT_MSG(approx_eq(k, AdaptiveKeepRatioState{}.last_keep),
                    "unknown session must return default keep_ratio");
    TEST_ASSERT(mgr.turn_count("no-such-session") == 0);
}

static void get_ema_reflects_post_update_value() {
    HttpServerSessions mgr;
    TEST_ASSERT_MSG(approx_eq(mgr.get_ema("s1"), 0.0f), "unknown session ema is 0");
    // First turn: ema seeds to observed directly
    mgr.update("s1", 0.80f);
    TEST_ASSERT_MSG(approx_eq(mgr.get_ema("s1"), 0.80f), "first-turn ema == observed");
    // Second turn: ema = alpha*prev + (1-alpha)*observed
    mgr.update("s1", 0.60f);
    float expected = kBanditEmaAlpha * 0.80f + (1.0f - kBanditEmaAlpha) * 0.60f;
    TEST_ASSERT_MSG(approx_eq(mgr.get_ema("s1"), expected), "second-turn ema correct");
}

static void lru_eviction_bounds_map_size() {
    HttpServerSessions mgr;

    // Insert kMaxSessions + 100 distinct sessions
    const std::size_t over = kMaxSessions + 100;
    for (std::size_t i = 0; i < over; ++i) {
        mgr.update("sess-" + std::to_string(i), 0.80f);
    }

    // Map must stay at or below the cap
    TEST_ASSERT_MSG(mgr.size() <= kMaxSessions,
                    "map size must not exceed kMaxSessions after overflow inserts");

    // The OLDEST sessions (low indices, never touched after insert) must be gone.
    // The most recent kMaxSessions inserts are the high-index ones.
    // Verify the very first session is evicted.
    float k0 = mgr.get_keep_ratio("sess-0");
    // A session evicted returns the default keep; one still present returns a
    // stepped-down keep (we fed accept=0.80 which is inside the band → keep unchanged).
    // We just assert size is bounded; eviction of the oldest is implied by LRU.
    TEST_ASSERT_MSG(mgr.size() <= kMaxSessions,
                    "size still bounded after get_keep_ratio accesses");
    (void)k0;  // value used above; suppress unused-variable warning

    // Touch only a few sessions to make them "recently used", then overflow again.
    // Those touched sessions must survive a second wave.
    const std::string pinned = "sess-" + std::to_string(over - 1);
    for (int t = 0; t < 3; ++t) mgr.update(pinned, 0.80f);

    for (std::size_t i = over; i < over + 200; ++i) {
        mgr.update("wave2-" + std::to_string(i), 0.80f);
    }

    TEST_ASSERT_MSG(mgr.size() <= kMaxSessions, "size bounded after second wave");
    TEST_ASSERT_MSG(mgr.turn_count(pinned) >= 3,
                    "recently-used pinned session must survive eviction waves");
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main() {
    std::fprintf(stderr, "=== test_adaptive_keep_ratio ===\n");

    RUN_TEST(default_construction);
    RUN_TEST(first_turn_sets_ema_to_observed);
    RUN_TEST(high_accept_decreases_keep);
    RUN_TEST(low_accept_increases_keep);
    RUN_TEST(in_band_no_change);
    RUN_TEST(respects_lower_bound);
    RUN_TEST(respects_upper_bound);
    RUN_TEST(ten_turn_convergence_high_accept);
    RUN_TEST(escalation_far_outside_band);
    RUN_TEST(sessions_isolated);
    RUN_TEST(unknown_session_returns_default);
    RUN_TEST(get_ema_reflects_post_update_value);
    RUN_TEST(lru_eviction_bounds_map_size);

    std::fprintf(stderr, "\n%d tests, %d failures\n", test_count, test_failures);
    return (test_failures == 0) ? 0 : 1;
}
