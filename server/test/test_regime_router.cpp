// Unit tests for the pflash regime router v2 — pure function, no GPU.
//
// Tests kept: t8 (deploy-noop), t10 (agentic-throttle), t11 (retrieval-full),
//             t12 (below-threshold), t14 (degenerate), t18 (detect_request_type).
//
// Tests removed:
//   t1-t7  — v1 R-router (decide_regime), refuted (ρ=-0.27), deleted.
//   t9     — sparse_prompt_guard, validated zero-sum, deleted.
//   t13    — recency_floor_invariant, deleted with recency floor feature.
//   t15-t17 — recency_floor_for, deleted with recency floor feature.
//
// Build (standalone, from repo root):
//   g++-11 -std=gnu++17 -O2 -Wall -Wextra -Werror -I server/src/common
//     -o /tmp/test_regime_router server/test/test_regime_router.cpp
// CMake:
//   cmake --build build --target test_regime_router -j
//   ctest -R regime_router --output-on-failure

#include "regime_router.h"

#include <cmath>
#include <cstdio>
#include <string>

using namespace dflash::common;

// ─── Minimal test framework ───────────────────────────────────────────────────

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
        std::fprintf(stderr, "  FAIL: %s:%d: %s -- %s\n", \
                     __FILE__, __LINE__, #expr, msg); \
    } \
} while (0)

#define RUN_TEST(fn) do { \
    std::fprintf(stderr, "  %s ...", #fn); \
    int before = test_failures; \
    fn(); \
    if (test_failures == before) std::fprintf(stderr, " ok\n"); \
    else std::fprintf(stderr, "\n"); \
} while (0)

static inline bool approx_eq(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

static RouterPolicyV2 default_v2_policy() { return {}; }

static RouterPolicyV2 enabled_v2_policy() {
    RouterPolicyV2 p;
    p.enabled = true;
    return p;
}

static RequestFeatures make_features(bool is_agentic, int prompt_tokens) {
    return { is_agentic, prompt_tokens };
}

// ─── T8: DEPLOY-NO-OP ────────────────────────────────────────────────────────
// enabled=false → SAFE for every input, including is_agentic=true and huge prompts.

static void t8_v2_deploy_noop() {
    RouterPolicyV2 p = default_v2_policy();   // enabled=false

    {
        auto d = decide_v2(make_features(true, 100000), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T8a: disabled->keep_target must be full_keep_target");
        TEST_ASSERT_MSG(d.cascade, "T8a: disabled->cascade must be true");
        TEST_ASSERT_MSG(std::string(d.reason) == "disabled_noop",
                        "T8a: disabled->reason must be 'disabled_noop'");
    }
    // Sweep all combinations of is_agentic and prompt sizes.
    for (int i = 0; i < 4; ++i) {
        bool agentic = (i & 1) != 0;
        int  prompt  = (i & 2) ? 100000 : 500;
        auto d = decide_v2(make_features(agentic, prompt), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T8-sweep: disabled->keep_target must be full_keep_target");
        TEST_ASSERT_MSG(d.cascade, "T8-sweep: disabled->cascade must be true");
    }
    // Explicitly: is_agentic=true, large prompt — must be SAFE.
    {
        auto d = decide_v2(make_features(true, 200000), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T8b: disabled, agentic, huge prompt -> SAFE");
        TEST_ASSERT_MSG(d.cascade, "T8b: disabled -> cascade=true");
    }
}

// ─── T10: AGENTIC-THROTTLE ───────────────────────────────────────────────────
// enabled, is_agentic=true, prompt > threshold
// → keep_target=agentic_keep_target, cascade=false.

static void t10_agentic_throttle() {
    RouterPolicyV2 p = enabled_v2_policy();

    {
        auto d = decide_v2(make_features(true, 40000), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.agentic_keep_target),
                        "T10a: agentic throttle -> keep_target=agentic_keep_target");
        TEST_ASSERT_MSG(!d.cascade, "T10a: agentic throttle -> cascade=false");
        TEST_ASSERT_MSG(std::string(d.reason) == "agentic_throttle",
                        "T10a: reason must be 'agentic_throttle'");
    }
    // Custom agentic_keep_target.
    {
        RouterPolicyV2 p2 = p;
        p2.agentic_keep_target = 0.30;
        auto d = decide_v2(make_features(true, 60000), p2);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, 0.30),
                        "T10b: custom agentic_keep_target propagated");
        TEST_ASSERT_MSG(!d.cascade, "T10b: agentic -> cascade=false");
    }
}

// ─── T11: RETRIEVAL-FULL ─────────────────────────────────────────────────────
// enabled, is_agentic=false, prompt > threshold
// → cascade=true, keep_target=full_keep_target.

static void t11_retrieval_full() {
    RouterPolicyV2 p = enabled_v2_policy();

    {
        auto d = decide_v2(make_features(false, 40000), p);
        TEST_ASSERT_MSG(d.cascade, "T11a: retrieval -> cascade=true");
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T11a: retrieval -> keep_target=full_keep_target");
        TEST_ASSERT_MSG(std::string(d.reason) == "retrieval_full",
                        "T11a: reason must be 'retrieval_full'");
    }
    // Custom full_keep_target.
    {
        RouterPolicyV2 p2   = p;
        p2.full_keep_target = 0.80;
        auto d = decide_v2(make_features(false, 50000), p2);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, 0.80),
                        "T11b: custom full_keep_target propagated");
        TEST_ASSERT_MSG(d.cascade, "T11b: retrieval -> cascade=true");
    }
}

// ─── T12: BELOW-THRESHOLD ────────────────────────────────────────────────────
// prompt_tokens < threshold_tokens → SAFE regardless of is_agentic.

static void t12_v2_below_threshold() {
    RouterPolicyV2 p = enabled_v2_policy();

    // Agentic, just below threshold.
    {
        auto d = decide_v2(make_features(true, p.threshold_tokens - 1), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T12a: agentic, below threshold -> SAFE");
        TEST_ASSERT_MSG(d.cascade, "T12a: below threshold -> cascade=true");
        TEST_ASSERT_MSG(std::string(d.reason) == "below_threshold",
                        "T12a: reason must be 'below_threshold'");
    }
    // Non-agentic, just below threshold.
    {
        auto d = decide_v2(make_features(false, p.threshold_tokens - 1), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T12b: non-agentic, below threshold -> SAFE");
    }
    // Custom threshold.
    {
        RouterPolicyV2 p2   = p;
        p2.threshold_tokens = 10000;
        auto d = decide_v2(make_features(true, 9999), p2);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p2.full_keep_target),
                        "T12c: custom threshold, below it -> SAFE");
        TEST_ASSERT_MSG(std::string(d.reason) == "below_threshold",
                        "T12c: reason must be 'below_threshold'");
    }
}

// ─── T14: DEGENERATE ─────────────────────────────────────────────────────────
// prompt_tokens <= 0 → SAFE (no crash, no garbage).

static void t14_v2_degenerate() {
    RouterPolicyV2 p = enabled_v2_policy();

    // prompt_tokens = 0
    {
        auto d = decide_v2(make_features(true, 0), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T14a: prompt_tokens=0 -> SAFE");
        TEST_ASSERT_MSG(d.cascade, "T14a: degenerate -> cascade=true");
        TEST_ASSERT_MSG(std::string(d.reason) == "degenerate",
                        "T14a: reason must be 'degenerate'");
    }
    // prompt_tokens < 0
    {
        auto d = decide_v2(make_features(false, -1), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T14b: negative prompt_tokens -> SAFE");
        TEST_ASSERT_MSG(std::string(d.reason) == "degenerate",
                        "T14b: reason must be 'degenerate'");
    }
    // Both degenerate
    {
        auto d = decide_v2(make_features(true, -5), p);
        TEST_ASSERT_MSG(approx_eq(d.keep_target, p.full_keep_target),
                        "T14c: negative agentic -> SAFE");
    }
}

// ─── T18: detect_request_type — bool truth-table ─────────────────────────────
//
// Exhaustive 3-bit truth table: any true → Agentic, all false → Retrieval.
// No JSON dependency; the caller extracts bools at the handler boundary.

static void t18_detect_request_type() {
    // All-false → Retrieval (safe default).
    {
        auto type = detect_request_type(false, false, false);
        TEST_ASSERT_MSG(type == RequestType::Retrieval,
                        "T18a: all false -> Retrieval");
    }
    // has_tools only → Agentic.
    {
        auto type = detect_request_type(true, false, false);
        TEST_ASSERT_MSG(type == RequestType::Agentic,
                        "T18b: has_tools=true -> Agentic");
    }
    // has_tool_use_blocks only → Agentic.
    {
        auto type = detect_request_type(false, true, false);
        TEST_ASSERT_MSG(type == RequestType::Agentic,
                        "T18c: has_tool_use_blocks=true -> Agentic");
    }
    // has_tool_calls only → Agentic.
    {
        auto type = detect_request_type(false, false, true);
        TEST_ASSERT_MSG(type == RequestType::Agentic,
                        "T18d: has_tool_calls=true -> Agentic");
    }
    // has_tools + has_tool_use_blocks → Agentic.
    {
        auto type = detect_request_type(true, true, false);
        TEST_ASSERT_MSG(type == RequestType::Agentic,
                        "T18e: has_tools + has_tool_use_blocks -> Agentic");
    }
    // has_tools + has_tool_calls → Agentic.
    {
        auto type = detect_request_type(true, false, true);
        TEST_ASSERT_MSG(type == RequestType::Agentic,
                        "T18f: has_tools + has_tool_calls -> Agentic");
    }
    // has_tool_use_blocks + has_tool_calls → Agentic.
    {
        auto type = detect_request_type(false, true, true);
        TEST_ASSERT_MSG(type == RequestType::Agentic,
                        "T18g: has_tool_use_blocks + has_tool_calls -> Agentic");
    }
    // All true → Agentic.
    {
        auto type = detect_request_type(true, true, true);
        TEST_ASSERT_MSG(type == RequestType::Agentic,
                        "T18h: all true -> Agentic");
    }
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main() {
    std::fprintf(stderr, "=== test_regime_router ===\n");

    RUN_TEST(t8_v2_deploy_noop);
    RUN_TEST(t10_agentic_throttle);
    RUN_TEST(t11_retrieval_full);
    RUN_TEST(t12_v2_below_threshold);
    RUN_TEST(t14_v2_degenerate);

    std::fprintf(stderr, "--- detect_request_type ---\n");
    RUN_TEST(t18_detect_request_type);

    std::fprintf(stderr, "\n%d tests, %d failures\n", test_count, test_failures);
    return (test_failures == 0) ? 0 : 1;
}
