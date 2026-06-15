// Unit tests for should_reject_oversized — pure, GPU-free.
// Reject iff prompt+max_output > max_ctx AND compression is NOT enabled.
// Build: /usr/bin/g++-11 -std=gnu++17 -O0 -I server/src -o /tmp/test_admission server/test/test_admission.cpp && /tmp/test_admission
#include "server/admission.h"

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

// 100+100 <= 1024, no compression -> accept
static void test_small_prompt_no_compression_accepts() {
    TEST_ASSERT(!should_reject_oversized(100, 100, 1024, false));
}

// 900+200 > 1024, no compression -> reject (hard gate preserved)
static void test_oversized_no_compression_rejects() {
    TEST_ASSERT(should_reject_oversized(900, 200, 1024, false));
}

// 167000+2048 > 65536, compression enabled -> accept (post-compress check is gate)
static void test_oversized_with_compression_accepts() {
    TEST_ASSERT(!should_reject_oversized(167000, 2048, 65536, true));
}

// prompt+max_output == max_ctx is not oversized -> accept
static void test_exactly_at_limit_accepts() {
    TEST_ASSERT(!should_reject_oversized(1024, 0, 1024, false));
    TEST_ASSERT(!should_reject_oversized(512, 512, 1024, false));
}

// 1025 > 1024, no compression -> reject
static void test_one_over_limit_no_compression_rejects() {
    TEST_ASSERT(should_reject_oversized(1025, 0, 1024, false));
}

// 1025 > 1024, compression enabled -> accept
static void test_one_over_limit_with_compression_accepts() {
    TEST_ASSERT(!should_reject_oversized(1025, 0, 1024, true));
}

// ── effective_prompt_overflows tests ───────────────────────────────────────

// (a) FlowKV-compressed request, effective_tokens already within budget → no reject.
static void test_effective_overflows_compressed_within_budget() {
    // raw=50000, after FlowKV effective=5000, max_output=2048, max_ctx=65536
    TEST_ASSERT(!effective_prompt_overflows(5000, 0, 2048, 65536));
}

// (b) BUG-B: raw-oversized request that is a pFlash full-cache hit (served_from_cache
//     tokens=800 which fits) must NOT be rejected.
// This is THE BUG: current code uses effective_tokens (raw=70000) and rejects.
static void test_effective_overflows_full_cache_hit_uses_served_size() {
    // raw prompt = 70000 tokens, but full-cache hit stores only 800 compressed tokens.
    // max_output=2048, max_ctx=65536.
    // Served size 800 + 2048 = 2848 <= 65536 → must NOT overflow.
    // BUG: current implementation ignores served size → returns true (false reject).
    TEST_ASSERT(!effective_prompt_overflows(70000, 800, 2048, 65536));
}

// (c) Genuinely oversized post-compress, no cache hit (-1 sentinel) → reject.
static void test_effective_overflows_post_compress_genuinely_oversized() {
    // effective=60000, max_output=10000, max_ctx=65536 → 70000 > 65536 → reject.
    TEST_ASSERT(effective_prompt_overflows(60000, -1, 10000, 65536));
}

// (d) Verbatim turn-1 within budget → no reject.
static void test_effective_overflows_verbatim_within_budget() {
    // effective=1000, no cache, max_output=2048, max_ctx=65536 → accept.
    TEST_ASSERT(!effective_prompt_overflows(1000, -1, 2048, 65536));
}

// (f) Degenerate zero-length cache hit must be treated as a hit, not as no-hit.
static void test_effective_overflows_zero_length_hit_is_a_hit() {
    // served=0 (valid hit), max_output=2048 → 2048 <= 65536 → accept.
    TEST_ASSERT(!effective_prompt_overflows(70000, 0, 2048, 65536));
}

// (e) Full-cache hit but served size + max_output itself overflows → reject.
static void test_effective_overflows_full_cache_hit_still_too_large() {
    // served=60000, max_output=10000, max_ctx=65536 → 70000 > 65536 → reject.
    TEST_ASSERT(effective_prompt_overflows(200000, 60000, 10000, 65536));
}

int main() {
    std::fprintf(stderr, "=== test_admission ===\n");
    RUN_TEST(test_small_prompt_no_compression_accepts);
    RUN_TEST(test_oversized_no_compression_rejects);
    RUN_TEST(test_oversized_with_compression_accepts);
    RUN_TEST(test_exactly_at_limit_accepts);
    RUN_TEST(test_one_over_limit_no_compression_rejects);
    RUN_TEST(test_one_over_limit_with_compression_accepts);
    RUN_TEST(test_effective_overflows_compressed_within_budget);
    RUN_TEST(test_effective_overflows_full_cache_hit_uses_served_size);
    RUN_TEST(test_effective_overflows_post_compress_genuinely_oversized);
    RUN_TEST(test_effective_overflows_verbatim_within_budget);
    RUN_TEST(test_effective_overflows_full_cache_hit_still_too_large);
    RUN_TEST(test_effective_overflows_zero_length_hit_is_a_hit);
    std::fprintf(stderr, "\n%d tests, %d failures\n", test_count, test_failures);
    return (test_failures == 0) ? 0 : 1;
}
