// Unit tests for skip_park_allowed — pure, GPU-free.
// Build: /usr/bin/c++ -std=gnu++17 -O0 -I server/src -o /tmp/test_skip_park_guard server/test/test_skip_park_guard.cpp && /tmp/test_skip_park_guard
#include "placement/skip_park_guard.h"

#include <cstdio>

static int test_failures = 0;
static int test_count    = 0;

#define TEST_ASSERT(expr) do {                                  \
    test_count++;                                               \
    if (!(expr)) {                                              \
        test_failures++;                                        \
        std::fprintf(stderr, "  FAIL: %s:%d: %s\n",            \
                     __FILE__, __LINE__, #expr);                \
    }                                                           \
} while (0)

#define RUN_TEST(fn) do {                                       \
    std::fprintf(stderr, "  %s ...", #fn);                      \
    int before = test_failures;                                 \
    fn();                                                       \
    std::fprintf(stderr, (test_failures == before) ? " ok\n" : "\n"); \
} while (0)

static constexpr size_t GiB = 1024ull * 1024 * 1024;

// not_requested stays off regardless of card size or ctx
static void T1_not_requested_stays_off() {
    TEST_ASSERT(dflash::common::skip_park_allowed(false, 24 * GiB, 32768) == false);
}

// >=32GB card: safe at any ctx
static void T2_big_card_any_ctx() {
    TEST_ASSERT(dflash::common::skip_park_allowed(true, 32 * GiB, 131072) == true);
}

// <32GB card, max_ctx<=65536: proven safe
static void T3_small_card_small_ctx_allowed() {
    TEST_ASSERT(dflash::common::skip_park_allowed(true, 24 * GiB, 65536) == true);
}

// <32GB card, max_ctx=131072: tonight's crash cell — must downgrade
static void T4_small_card_big_ctx_downgraded() {
    TEST_ASSERT(dflash::common::skip_park_allowed(true, 24 * GiB, 131072) == false);
}

// <32GB card, max_ctx=65537: one over the proven-safe boundary
static void T5_boundary_ctx_one_over() {
    TEST_ASSERT(dflash::common::skip_park_allowed(true, 24 * GiB, 65537) == false);
}

// just under 32GB: still counts as small card
static void T6_boundary_vram_just_under_32g() {
    TEST_ASSERT(dflash::common::skip_park_allowed(true, 32 * GiB - 1, 131072) == false);
}

int main() {
    std::fprintf(stderr, "=== test_skip_park_guard ===\n");
    RUN_TEST(T1_not_requested_stays_off);
    RUN_TEST(T2_big_card_any_ctx);
    RUN_TEST(T3_small_card_small_ctx_allowed);
    RUN_TEST(T4_small_card_big_ctx_downgraded);
    RUN_TEST(T5_boundary_ctx_one_over);
    RUN_TEST(T6_boundary_vram_just_under_32g);
    std::fprintf(stderr, "\n%d tests, %d failures\n", test_count, test_failures);
    return (test_failures == 0) ? 0 : 1;
}
