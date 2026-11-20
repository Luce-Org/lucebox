#include "CppUnitTestFramework.hpp"
#include "common/spec_acceptance.h"

using namespace luce::common;

namespace {
struct SpecAcceptanceFixture {};
}

TEST_CASE(SpecAcceptanceFixture, empty_history_has_zero_rate) {
    SpecAcceptanceStats stats;
    CHECK(stats.accepted() == 0);
    CHECK(stats.offered() == 0);
    CHECK(stats.rate() == 0.0f);
}

TEST_CASE(SpecAcceptanceFixture, final_budget_does_not_penalize_full_acceptance) {
    SpecAcceptanceStats stats;
    stats.record_chain(4, 4, 4, 5);
    stats.record_chain(6, 6, 1, 1);
    CHECK(stats.accepted() == 5);
    CHECK(stats.offered() == 5);
    CHECK(stats.rate() == 1.0f);
}

TEST_CASE(SpecAcceptanceFixture, rejection_bonus_is_not_an_accepted_draft) {
    SpecAcceptanceStats stats;
    stats.record_chain(2, 6, 3, 3); // seed + match + rejection bonus
    CHECK(stats.accepted() == 2);
    CHECK(stats.offered() == 3);
    CHECK(std::fabs(stats.rate() - 2.0f / 3.0f) < 1e-6f);
}

TEST_CASE(SpecAcceptanceFixture, early_stop_counts_only_emitted_accepts) {
    SpecAcceptanceStats stats;
    stats.record_chain(4, 4, 2, 20); // EOS/cancel stops an otherwise accepted step
    CHECK(stats.accepted() == 2);
    CHECK(stats.offered() == 4);
    CHECK(stats.rate() == 0.5f);
}

TEST_CASE(SpecAcceptanceFixture, tree_only_counts_actual_topology_with_root) {
    SpecAcceptanceStats stats;
    stats.record_tree(4, 7, 20); // four emitted positions out of root + seven nodes
    stats.record_tree(2, 3, 16); // padded verify capacity must not enter this API
    CHECK(stats.accepted() == 6);
    CHECK(stats.offered() == 12);
    CHECK(stats.rate() == 0.5f);
}

TEST_CASE(SpecAcceptanceFixture, mixed_tree_and_chain_share_one_denominator) {
    SpecAcceptanceStats stats;
    stats.record_tree(3, 4, 20);
    stats.record_chain(2, 3, 3, 20);
    CHECK(stats.accepted() == 5);
    CHECK(stats.offered() == 8);
    CHECK(stats.rate() == 0.625f);
}

TEST_CASE(SpecAcceptanceFixture, final_tree_budget_matches_chain_accounting) {
    SpecAcceptanceStats stats;
    stats.record_tree(3, 7, 3);
    CHECK(stats.accepted() == 3);
    CHECK(stats.offered() == 3);
    CHECK(stats.rate() == 1.0f);
    stats.record_tree(1, 15, 1); // only the root can be emitted
    CHECK(stats.accepted() == 4);
    CHECK(stats.offered() == 4);
    CHECK(stats.rate() == 1.0f);
}
