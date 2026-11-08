// Unit test for the mode-5 rescore decision (no GPU): when the drafter scoring is skipped.

#include "CppUnitTestFramework.hpp"

#include "kvflash_m5_rescore.h"

#include <vector>

using namespace luce::common;

namespace {
struct KvflashM5RescoreFixture {};
KvFlashM5Last last_at(uint64_t key, int end, uint64_t ph) { KvFlashM5Last l; l.valid = true; l.query_key = key; l.end = end; l.prefix_hash = ph; return l; }
}

TEST_CASE(KvflashM5RescoreFixture, no_previous_score_scores) {
    CHECK(kvflash_m5_should_score(KvFlashM5Last{}, 7, 1000, 0, 4096));
}
TEST_CASE(KvflashM5RescoreFixture, same_query_extending_prompt_under_budget_skips) {
    CHECK(!kvflash_m5_should_score(last_at(7, 1000, 99), 7, 1500, 99, 4096));
    CHECK(!kvflash_m5_should_score(last_at(7, 1000, 99), 7, 1000, 99, 4096));
}
TEST_CASE(KvflashM5RescoreFixture, changed_query_scores) {
    CHECK(kvflash_m5_should_score(last_at(7, 1000, 99), 8, 1100, 99, 4096));
}
TEST_CASE(KvflashM5RescoreFixture, diverged_prefix_scores) {
    CHECK(kvflash_m5_should_score(last_at(7, 1000, 99), 7, 1100, 98, 4096));
}
TEST_CASE(KvflashM5RescoreFixture, shorter_prompt_scores) {
    CHECK(kvflash_m5_should_score(last_at(7, 1000, 99), 7, 900, 99, 4096));
}
TEST_CASE(KvflashM5RescoreFixture, growth_at_or_over_budget_scores) {
    CHECK(!kvflash_m5_should_score(last_at(7, 1000, 99), 7, 5095, 99, 4096));
    CHECK(kvflash_m5_should_score(last_at(7, 1000, 99), 7, 5096, 99, 4096));
    CHECK(kvflash_m5_should_score(last_at(7, 1000, 99), 7, 9000, 99, 4096));
}
TEST_CASE(KvflashM5RescoreFixture, zero_or_negative_budget_always_scores) {
    CHECK(kvflash_m5_should_score(last_at(7, 1000, 99), 7, 1001, 99, 0));
    CHECK(kvflash_m5_should_score(last_at(7, 1000, 99), 7, 1001, 99, -5));
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
