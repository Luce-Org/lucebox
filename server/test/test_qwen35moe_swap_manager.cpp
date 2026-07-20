#include "CppUnitTestFramework.hpp"
#include "../src/common/moe_hybrid_swap_manager.h"
#include "../src/common/moe_hybrid_placement.h"
#include "../src/common/moe_hybrid_routing_stats.h"

#include <cstdio>
#include <cstdlib>

using namespace dflash::common;

namespace {
struct Qwen35MoeSwapManagerFixture {};
}

TEST_CASE(Qwen35MoeSwapManagerFixture, moe_swap_manager_suite) {
    MoeHybridRoutingStats stats;
    stats.n_layer = 2;
    stats.n_expert = 4;
    stats.n_expert_used = 2;
    stats.counts = {
        100, 10, 90, 5,   // layer 0
        50, 49, 48, 47    // layer 1
    };
    stats.layer_totals = {205, 194};

    MoeHybridPlacement placement;
    placement.n_layer = 2;
    placement.n_expert = 4;
    placement.n_expert_used = 2;
    placement.total_hot = 2;
    placement.hot_counts = {1, 1};
    placement.hot_expert_ids = {{1}, {0}};

    MoeHybridSwapPolicy policy;
    policy.max_swaps_total = 1;
    policy.min_promote_gain = 5;

    MoeHybridSwapPlan plan;
    std::string err;
    REQUIRE(build_moe_hybrid_swap_plan(placement, stats, policy, plan, &err));
    REQUIRE(plan.actions.size() == 1);
    REQUIRE(plan.actions[0].layer_idx == 0);
    REQUIRE(plan.actions[0].evict_expert == 1);
    REQUIRE(plan.actions[0].promote_expert == 0);
    REQUIRE(plan.next_placement.is_hot(0, 0));
    REQUIRE(!plan.next_placement.is_hot(0, 1));
    REQUIRE(plan.next_placement.is_hot(1, 0));

    std::printf("OK\n");
}
