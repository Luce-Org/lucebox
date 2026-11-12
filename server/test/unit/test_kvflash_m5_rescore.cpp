// Unit test for the mode-5 rescore decision (no GPU): when the drafter scoring is skipped.

#include "CppUnitTestFramework.hpp"

#include "kvflash_m5_rescore.h"

#include <vector>

using namespace luce::common;

namespace {
struct KvflashM5RescoreFixture {};
std::vector<int32_t> seq(int n, int32_t base = 0) { std::vector<int32_t> v((size_t)n); for (int i = 0; i < n; i++) v[(size_t)i] = base + i; return v; }
KvFlashM5Last last_at(uint64_t key, const std::vector<int32_t> & ids) { KvFlashM5Last l; l.valid = true; l.query_key = key; l.ids = ids; return l; }
std::vector<int32_t> extend(std::vector<int32_t> v, int n) { const int32_t b = 1000000; for (int i = 0; i < n; i++) v.push_back(b + i); return v; }
}

TEST_CASE(KvflashM5RescoreFixture, no_previous_score_scores) {
    CHECK(kvflash_m5_should_score(KvFlashM5Last{}, 7, seq(1000), 4096));
}
TEST_CASE(KvflashM5RescoreFixture, same_query_extending_prompt_under_budget_skips) {
    CHECK(!kvflash_m5_should_score(last_at(7, seq(1000)), 7, extend(seq(1000), 500), 4096));
    CHECK(!kvflash_m5_should_score(last_at(7, seq(1000)), 7, seq(1000), 4096));
}
TEST_CASE(KvflashM5RescoreFixture, a_rerendered_tail_within_the_slack_still_skips) {
    // the last scored prompt ended with the generation prompt; the next one renders the assistant turn there
    std::vector<int32_t> next = seq(1000);
    for (int i = 1000 - kKvFlashM5TailSlack; i < 1000; i++) next[(size_t)i] = -1;
    CHECK(!kvflash_m5_should_score(last_at(7, seq(1000)), 7, extend(next, 300), 4096));
}
TEST_CASE(KvflashM5RescoreFixture, a_divergence_before_the_slack_scores) {
    std::vector<int32_t> next = seq(1000);
    next[(size_t)(1000 - kKvFlashM5TailSlack - 1)] = -1;
    CHECK(kvflash_m5_should_score(last_at(7, seq(1000)), 7, extend(next, 300), 4096));
}
TEST_CASE(KvflashM5RescoreFixture, changed_query_scores) {
    CHECK(kvflash_m5_should_score(last_at(7, seq(1000)), 8, extend(seq(1000), 100), 4096));
}
TEST_CASE(KvflashM5RescoreFixture, shorter_prompt_scores) {
    CHECK(kvflash_m5_should_score(last_at(7, seq(1000)), 7, seq(900), 4096));
}
TEST_CASE(KvflashM5RescoreFixture, growth_past_the_common_prefix_at_or_over_budget_scores) {
    CHECK(!kvflash_m5_should_score(last_at(7, seq(1000)), 7, extend(seq(1000), 4095), 4096));
    CHECK(kvflash_m5_should_score(last_at(7, seq(1000)), 7, extend(seq(1000), 4096), 4096));
}
TEST_CASE(KvflashM5RescoreFixture, zero_or_negative_budget_always_scores) {
    CHECK(kvflash_m5_should_score(last_at(7, seq(1000)), 7, extend(seq(1000), 1), 0));
    CHECK(kvflash_m5_should_score(last_at(7, seq(1000)), 7, extend(seq(1000), 1), -5));
}
TEST_CASE(KvflashM5RescoreFixture, query_key_hashes_the_span_only) {
    const std::vector<int32_t> a = {1, 2, 3, 4, 5, 6}, b = {9, 9, 3, 4, 5, 8};
    CHECK(kvflash_m5_query_key(a, 2, 5) == kvflash_m5_query_key(b, 2, 5));
    CHECK(kvflash_m5_query_key(a, 2, 5) != kvflash_m5_query_key(a, 2, 6));
    CHECK(kvflash_m5_query_key(a, 1, 5) != kvflash_m5_query_key(a, 2, 5));
}

TEST_CASE(KvflashM5RescoreFixture, reused_scores_pad_new_chunks_with_the_lowest_score) {
    const std::vector<float> last = {0.5f, -2.0f, 3.0f};
    const std::vector<float> r = luce::common::kvflash_m5_reuse_scores(last, 5);
    CHECK_EQUAL(5, (int)r.size());
    CHECK_EQUAL(3.0f, r[2]);
    CHECK_EQUAL(-2.0f, r[3]);
    CHECK_EQUAL(-2.0f, r[4]);
    CHECK_EQUAL(2, (int)luce::common::kvflash_m5_reuse_scores(last, 2).size());
    CHECK_EQUAL(0, (int)luce::common::kvflash_m5_reuse_scores({}, 4).size());
}
