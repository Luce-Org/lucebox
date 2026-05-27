#include "qwen35moe_swap_manager.h"

#include <cstdio>
#include <cstdlib>

using namespace dflash::common;

static void expect(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

int main() {
    Qwen35MoeRoutingStats stats;
    stats.n_layer = 2;
    stats.n_expert = 4;
    stats.n_expert_used = 2;
    stats.counts = {
        100, 10, 90, 5,   // layer 0
        50, 49, 48, 47    // layer 1
    };
    stats.layer_totals = {205, 194};

    Qwen35MoeExpertPlacement placement;
    placement.n_layer = 2;
    placement.n_expert = 4;
    placement.n_expert_used = 2;
    placement.total_hot = 2;
    placement.hot_counts = {1, 1};
    placement.hot_expert_ids = {{1}, {0}};

    Qwen35MoeSwapPolicy policy;
    policy.max_swaps_total = 1;
    policy.min_promote_gain = 5;

    Qwen35MoeSwapPlan plan;
    std::string err;
    expect(build_qwen35moe_swap_plan(placement, stats, policy, plan, &err), err.c_str());
    expect(plan.actions.size() == 1, "one swap planned");
    expect(plan.actions[0].layer_idx == 0, "layer0 swap");
    expect(plan.actions[0].evict_expert == 1, "evict weakest hot");
    expect(plan.actions[0].promote_expert == 0, "promote best cold");
    expect(plan.next_placement.is_hot(0, 0), "layer0 expert0 hot after plan");
    expect(!plan.next_placement.is_hot(0, 1), "layer0 expert1 evicted");
    expect(plan.next_placement.is_hot(1, 0), "layer1 unchanged");

    std::printf("OK\n");
    return 0;
}
