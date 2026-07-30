#include "common/spec_commit.h"

#include <cassert>
#include <cstdint>
#include <vector>

using dflash::common::SpecCommitDecision;

static void test_greedy_mismatch_commits_bonus() {
    const std::vector<int32_t> draft = {10, 11, 12, 13};
    const std::vector<int32_t> target = {11, 99, 13, 14};
    const auto decision =
        SpecCommitDecision::greedy(draft, target, 4, 4);

    assert(decision.accepted_count() == 2);
    assert(decision.valid());
    assert(decision.commit_count() == 3);
    assert(decision.has_bonus());
    assert(decision.commits_bonus());
    assert(decision.bonus_token() == 99);

    int32_t token = -1;
    assert(decision.token_at(0, draft, token) && token == 10);
    assert(decision.token_at(1, draft, token) && token == 11);
    assert(decision.token_at(2, draft, token) && token == 99);
    assert(!decision.token_at(3, draft, token));

    std::vector<int32_t> committed;
    assert(decision.materialize(draft, committed));
    assert((committed == std::vector<int32_t>{10, 11, 99}));
}

static void test_all_matches_have_no_bonus() {
    const std::vector<int32_t> draft = {20, 21, 22};
    const std::vector<int32_t> target = {21, 22, 23};
    const auto decision =
        SpecCommitDecision::greedy(draft, target, 3, 8);

    assert(decision.accepted_count() == 3);
    assert(decision.valid());
    assert(decision.commit_count() == 3);
    assert(!decision.has_bonus());
    assert(!decision.commits_bonus());
}

static void test_budget_clips_bonus_and_prefix() {
    const auto no_bonus =
        SpecCommitDecision::precomputed(3, 4, true, 77, 3);
    assert(no_bonus.accepted_count() == 3);
    assert(no_bonus.commit_count() == 3);
    assert(no_bonus.has_bonus());
    assert(!no_bonus.commits_bonus());

    const auto clipped_prefix =
        SpecCommitDecision::precomputed(3, 4, true, 77, 2);
    assert(clipped_prefix.accepted_count() == 3);
    assert(clipped_prefix.commit_count() == 2);
    assert(clipped_prefix.has_bonus());
    assert(!clipped_prefix.commits_bonus());
}

static void test_invalid_inputs_are_safe() {
    const auto empty =
        SpecCommitDecision::greedy(nullptr, 0, nullptr, 0, 0, 4);
    assert(!empty.valid());
    assert(empty.accepted_count() == 0);
    assert(empty.commit_count() == 0);

    const std::vector<int32_t> short_draft = {1};
    int32_t token = -1;
    const auto malformed =
        SpecCommitDecision::precomputed(2, 2, false, 0, 2);
    assert(malformed.valid());
    assert(!malformed.token_at(1, short_draft, token));
    std::vector<int32_t> committed;
    assert(!malformed.materialize(short_draft, committed));
    assert(committed.empty());

    const auto no_budget =
        SpecCommitDecision::precomputed(1, 2, true, 2, -1);
    assert(no_budget.valid());
    assert(no_budget.commit_count() == 0);
    assert(no_budget.has_bonus());
    assert(!no_budget.commits_bonus());

    assert(!SpecCommitDecision::precomputed(0, 2, true, 2, 2).valid());
    assert(!SpecCommitDecision::precomputed(3, 2, false, 0, 2).valid());
    assert(!SpecCommitDecision::precomputed(2, 2, true, 2, 2).valid());
    const auto strict_without_bonus =
        SpecCommitDecision::precomputed(1, 2, false, -1, 2);
    assert(strict_without_bonus.valid());
    assert(strict_without_bonus.accepted_count() == 1);
    assert(strict_without_bonus.commit_count() == 1);
    assert(!strict_without_bonus.has_bonus());
    std::vector<int32_t> sentinel_prefix;
    assert(strict_without_bonus.materialize(
        std::vector<int32_t>{42, 43}, sentinel_prefix));
    assert((sentinel_prefix == std::vector<int32_t>{42}));
    assert(!SpecCommitDecision::precomputed(1, 2, true, -1, 2).valid());

    const std::vector<int32_t> invalid_draft = {1, -1};
    const auto invalid_token =
        SpecCommitDecision::precomputed(2, 2, false, 0, 2);
    assert(!invalid_token.materialize(invalid_draft, committed));

    const std::vector<int32_t> mismatched_draft = {1, 2};
    const std::vector<int32_t> invalid_target = {-1, 0};
    const auto greedy_sentinel =
        SpecCommitDecision::greedy(mismatched_draft, invalid_target, 2, 2);
    assert(greedy_sentinel.valid());
    assert(greedy_sentinel.accepted_count() == 1);
    assert(greedy_sentinel.commit_count() == 1);
    assert(!greedy_sentinel.has_bonus());
    assert(greedy_sentinel.materialize(mismatched_draft, committed));
    assert((committed == std::vector<int32_t>{1}));
}

int main() {
    test_greedy_mismatch_commits_bonus();
    test_all_matches_have_no_bonus();
    test_budget_clips_bonus_and_prefix();
    test_invalid_inputs_are_safe();
    return 0;
}
