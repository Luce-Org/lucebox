#include "CppUnitTestFramework.hpp"
#include "common/prompt_lookup.h"

#include <cstdint>
#include <vector>

using namespace luce::common;

namespace {
struct PromptLookupFixture {};

int propose(const std::vector<int32_t> & seq, int max_k, std::vector<int32_t> & out, int & match) {
    PromptLookup lookup;
    lookup.append(seq.data(), seq.size());
    out.assign((size_t) max_k, -1);
    const int k = lookup.propose(max_k, out.data(), match);
    out.resize((size_t) k);
    return k;
}
} // namespace

TEST_CASE(PromptLookupFixture, proposes_the_continuation_of_the_earlier_copy) {
    std::vector<int32_t> out;
    int match = 0;
    CHECK(propose({1, 2, 3, 4, 5, 6, 7, 1, 2, 3}, 3, out, match) == 3);
    CHECK((out == std::vector<int32_t>{4, 5, 6}));
    CHECK(match == 3);
}

TEST_CASE(PromptLookupFixture, longest_match_wins_then_most_recent) {
    std::vector<int32_t> out;
    int match = 0;
    // [9 1 2 3] matches 4 tokens back, the more recent [8 1 2 3] only 3.
    CHECK(propose({9, 1, 2, 3, 40, 41, 8, 1, 2, 3, 50, 51, 9, 1, 2, 3}, 2, out, match) == 2);
    CHECK((out == std::vector<int32_t>{40, 41}));
    CHECK(match == 4);
    // Equal matches: the most recent copy.
    CHECK(propose({1, 2, 3, 40, 7, 1, 2, 3, 50, 8, 1, 2, 3}, 1, out, match) == 1);
    CHECK(out[0] == 50);
    CHECK(match == 3);
}

TEST_CASE(PromptLookupFixture, proposes_nothing_without_a_trigram_match) {
    std::vector<int32_t> out;
    int match = 0;
    CHECK(propose({1, 2, 3, 4, 5, 6}, 4, out, match) == 0);
    CHECK(match == 0);
    CHECK(propose({1, 2, 3, 4, 9, 2, 3}, 4, out, match) == 0);   // only two tokens match
    CHECK(propose({1, 2, 3}, 4, out, match) == 0);               // too short
    CHECK(propose({1, 2, 3, 4, 1, 2, 3}, 0, out, match) == 0);   // max_k 0
}

TEST_CASE(PromptLookupFixture, proposal_stops_at_the_end_of_the_history) {
    std::vector<int32_t> out;
    int match = 0;
    CHECK(propose({1, 2, 3, 1, 2, 3}, 7, out, match) == 3);
    CHECK((out == std::vector<int32_t>{1, 2, 3}));
}

TEST_CASE(PromptLookupFixture, incremental_appends_keep_the_most_recent_ways) {
    PromptLookup lookup;
    const std::vector<int32_t> seq{5, 6, 7, 10, 5, 6, 7, 11, 5, 6, 7, 12, 5, 6, 7, 13, 5, 6, 7, 14, 5, 6, 7};
    for (const int32_t t : seq) lookup.append(&t, 1);
    int32_t out[4];
    int match = 0;
    CHECK(lookup.propose(1, out, match) == 1);
    CHECK(out[0] == 14);
    CHECK(lookup.size() == seq.size());
}

TEST_CASE(PromptLookupFixture, priors_take_long_matches_only) {
    PromptLookupPolicy policy(8);
    // E per cost with q = 0.96 on the shape 45 + 20 (w - 1): 0.03766 at width 7, 0.03765 at 8; the draft model's
    // 3.1 tokens / 105 = 0.0295 at width 4.
    PromptLookupPolicy::Pick pick = policy.choose(4, 7, 30);
    CHECK(pick.lookup);
    CHECK(pick.width == 7);
    // A short match (q prior 0.75) neither wins nor is confident enough to probe.
    pick = policy.choose(4, 7, 3);
    CHECK(!pick.lookup);
    CHECK(pick.width == 4);
    // No proposal: the draft model's width as given.
    pick = policy.choose(3, 0, 0);
    CHECK(!pick.lookup);
    CHECK(pick.width == 3);
}

TEST_CASE(PromptLookupFixture, learns_acceptance_per_match_length) {
    PromptLookupPolicy policy(8);
    // Draft-model rounds of width 4 commit 3.5 tokens in 76 ms; lookup rounds of width 8 take 90 ms.
    for (int i = 0; i < 200; ++i) policy.observe(false, 4, i % 2 == 0 ? 3 : 2, 0, 76.0);
    CHECK(policy.model_tokens(4) > 3.3);
    CHECK(policy.model_tokens(4) < 3.7);
    // Long matches whose drafts are all rejected: q falls and the policy stops taking them.
    for (int i = 0; i < 60; ++i) policy.observe(true, 8, 0, 30, 90.0);
    CHECK(policy.lookup_rate(30) < 0.2);
    CHECK(!policy.choose(4, 7, 30).lookup);
    // Long matches accepted in full at the measured lookup cost: taken again.
    for (int i = 0; i < 200; ++i) policy.observe(true, 8, 7, 30, 90.0);
    CHECK(policy.lookup_rate(30) > 0.9);
    CHECK(policy.choose(4, 7, 30).lookup);
    // The short-match bucket kept its prior.
    CHECK(policy.lookup_rate(3) > 0.74);
    CHECK(policy.lookup_rate(3) < 0.76);
}

TEST_CASE(PromptLookupFixture, costs_come_from_measured_rounds) {
    PromptLookupPolicy policy(8);
    // An unmeasured lookup width borrows the draft-model round's cost of that width, then the shape.
    policy.observe(false, 4, 3, 0, 80.0);
    CHECK(policy.cost_ms(true, 4) == 80.0);
    CHECK(policy.cost_ms(false, 5) > 80.0);
    // A measured lookup width uses its own, cheaper time (no draft model).
    policy.observe(true, 4, 3, 30, 70.0);
    CHECK(policy.cost_ms(true, 4) == 70.0);
    CHECK(policy.cost_ms(false, 4) == 80.0);
    // Untimed rounds (a graph build) teach acceptance but not cost.
    policy.observe(true, 6, 5, 30, -1.0);
    CHECK(policy.cost_ms(true, 4) == 70.0);
    // Width-1 rounds carry no drafts and are ignored.
    policy.observe(false, 1, 0, 0, 40.0);
    CHECK(policy.model_tokens(1) == 1.0);
}
