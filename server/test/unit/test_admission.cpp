// Unit tests for the admission gate, reply fitting and the concurrent PFlash reserve — pure, GPU-free.

#include "CppUnitTestFramework.hpp"
#include "server/admission.h"

namespace {
struct AdmissionFixture {};
}

TEST_CASE(AdmissionFixture, test_small_prompt_no_compression_accepts) {
    CHECK(!should_reject_oversized(100, 100, 1024, false));
}

TEST_CASE(AdmissionFixture, test_oversized_no_compression_rejects) {
    CHECK(should_reject_oversized(900, 200, 1024, false));
}

TEST_CASE(AdmissionFixture, test_reply_over_limit_with_compression_accepts) {
    CHECK(!should_reject_oversized(65000, 2048, 65536, true));
}

TEST_CASE(AdmissionFixture, test_exactly_at_limit_accepts) {
    CHECK(!should_reject_oversized(1024, 0, 1024, false));
    CHECK(!should_reject_oversized(512, 512, 1024, false));
}

TEST_CASE(AdmissionFixture, test_one_over_limit_no_compression_rejects) {
    CHECK(should_reject_oversized(1025, 0, 1024, false));
}

TEST_CASE(AdmissionFixture, test_one_over_limit_with_compression_rejects) {
    CHECK(should_reject_oversized(1025, 0, 1024, true));
}

TEST_CASE(AdmissionFixture, test_effective_overflows_compressed_within_budget) {
    CHECK(!effective_prompt_overflows(5000, 0, 2048, 65536));
}

TEST_CASE(AdmissionFixture, test_effective_overflows_full_cache_hit_uses_served_size) {
    CHECK(!effective_prompt_overflows(70000, 800, 2048, 65536));
}

TEST_CASE(AdmissionFixture, test_effective_overflows_post_compress_genuinely_oversized) {
    CHECK(effective_prompt_overflows(60000, -1, 10000, 65536));
}

TEST_CASE(AdmissionFixture, test_effective_overflows_verbatim_within_budget) {
    CHECK(!effective_prompt_overflows(1000, -1, 2048, 65536));
}

TEST_CASE(AdmissionFixture, test_effective_overflows_zero_length_hit_is_a_hit) {
    CHECK(!effective_prompt_overflows(70000, 0, 2048, 65536));
}

TEST_CASE(AdmissionFixture, test_effective_overflows_full_cache_hit_still_too_large) {
    CHECK(effective_prompt_overflows(200000, 60000, 10000, 65536));
}

TEST_CASE(AdmissionFixture, test_reply_fits_what_the_prompt_leaves) {
    // Claude Code asks for 32000 whatever the prompt.
    CHECK(fit_reply_to_context(110000, 32000, 131072) == 21072);
    CHECK(fit_reply_to_context(1000, 32000, 131072) == 32000);
    CHECK(fit_reply_to_context(131072, 32000, 131072) == 0);
    CHECK(fit_reply_to_context(140000, 32000, 131072) == 0);
    // Trimmed, the request is no longer oversized; a prompt that fills the
    // context still is.
    CHECK(!should_reject_oversized(110000, fit_reply_to_context(110000, 32000, 131072),
                                   131072, false));
    CHECK(should_reject_oversized(140000, fit_reply_to_context(140000, 32000, 131072),
                                  131072, false));
}

TEST_CASE(AdmissionFixture, test_images_reserve_at_most_the_reply_budget) {
    // DeepSeek V4.1's card asks 262144 by default, more than its 204,800-token
    // context: the images keep room for the reply budget, and the reply is then
    // trimmed to what they leave.
    CHECK(image_output_reserve(262144, 4096) == 4096);
    CHECK(image_output_reserve(1024, 4096) == 1024);
    CHECK(image_output_reserve(262144, 0) == 0);
    CHECK(image_output_reserve(-1, 4096) == 0);
    CHECK(fit_reply_to_context(200000, 262144, 204800) == 4800);
}

TEST_CASE(AdmissionFixture, test_a_default_reply_is_trimmed_even_when_routed) {
    CHECK(trims_reply(false, true, false));
    CHECK(trims_reply(false, false, false));
    // The router sends a sized reply whole to a model with room for it; the
    // card's default may fit no model whole.
    CHECK(!trims_reply(true, true, false));
    CHECK(trims_reply(true, false, false));
    CHECK(!trims_reply(false, false, true));
}

// With compression on, --max-ctx is the longest raw prompt PFlash takes (the drafter scores it whole): a prompt
// that fills the context passes even when its reply would not fit, since compression makes room; a longer one is
// refused.
TEST_CASE(AdmissionFixture, test_oversized_with_compression_respects_max_ctx) {
    CHECK(!should_reject_oversized(204800, 1, 204800, true));
    CHECK(!should_reject_oversized(200000, 32000, 204800, true));
    CHECK(should_reject_oversized(204801, 0, 204800, true));
    CHECK(should_reject_oversized(200000, 1, 139264, true));
    CHECK(should_reject_oversized(204800, 1, 204800, false));
    CHECK(!should_reject_oversized(1000, 1, 139264, false));
}

// The drafter's reserve covers its measured peak (Qwen3.5-0.8B BF16 drafter, R9700, peak minus the floor before the
// request) at every prompt length that was measured.
TEST_CASE(AdmissionFixture, test_pflash_concurrent_reserve_covers_measured_drafter_peaks) {
    const long long drafter_bytes = 1516744736;  // Qwen3.5-0.8B-BF16.gguf
    const long long gib = 1LL << 30;
    const struct { int tokens; double peak_gib; } measured[] = {
        {33513, 2.26}, {57775, 3.43}, {83928, 4.83}, {136684, 7.38}, {206205, 11.04}};
    for (const auto & m : measured) {
        CHECK(pflash_concurrent_reserve_bytes(drafter_bytes, m.tokens) >= (long long)(m.peak_gib * gib));
    }
    CHECK(pflash_concurrent_reserve_bytes(drafter_bytes, 139264) ==
          drafter_bytes + 139264LL * kPFlashScoringBytesPerToken);
    CHECK(pflash_concurrent_reserve_bytes(-1, -1) == 0);
}
