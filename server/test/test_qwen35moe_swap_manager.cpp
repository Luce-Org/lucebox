#include "CppUnitTestFramework.hpp"
#include "../src/common/moe_hybrid_swap_manager.h"
#include "../src/common/moe_hybrid_placement.h"
#include "../src/common/moe_hybrid_routing_stats.h"

#include <cstdio>
#include <cstdlib>

using namespace dflash::common;

#define TEST_ASSERT(cond) do { \
    auto _cpputf_exception = CppUnitTestFramework::Assert::IsTrue(static_cast<bool>(cond), #cond); \
    if (_cpputf_exception) { \
        throw *_cpputf_exception; \
    } \
} while (0)
static void expect(bool cond, const char * msg) { (void) msg; TEST_ASSERT(cond); }

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
    expect(build_moe_hybrid_swap_plan(placement, stats, policy, plan, &err), err.c_str());
    expect(plan.actions.size() == 1, "one swap planned");
    expect(plan.actions[0].layer_idx == 0, "layer0 swap");
    expect(plan.actions[0].evict_expert == 1, "evict weakest hot");
    expect(plan.actions[0].promote_expert == 0, "promote best cold");
    expect(plan.next_placement.is_hot(0, 0), "layer0 expert0 hot after plan");
    expect(!plan.next_placement.is_hot(0, 1), "layer0 expert1 evicted");
    expect(plan.next_placement.is_hot(1, 0), "layer1 unchanged");

    std::printf("OK\n");
}
