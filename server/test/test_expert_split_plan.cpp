#include "../src/common/expert_split_plan.h"
#include "../src/common/moe_hybrid_placement.h"
#include "../src/common/moe_hybrid_routing_stats.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dflash::common;

static void expect(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

static void test_ordered_targets_and_cpu_fallback() {
    ExpertSplitConfig cfg;
    cfg.n_layer = 1;
    cfg.n_expert = 5;

    const std::vector<ExpertSplitTarget> targets = {
        {"cuda:0", "cuda", 0, 2, 0, false},
        {"hip:0",  "hip",  0, 1, 0, false},
    };
    const std::vector<ExpertSplitUnit> units = {
        {0, 0, 1, 50.0},
        {0, 1, 1, 40.0},
        {0, 2, 1, 30.0},
        {0, 3, 1, 20.0},
        {0, 4, 1, 10.0},
    };

    ExpertSplitPlan plan;
    std::string err;
    expect(build_expert_split_plan(cfg, targets, units, plan, &err), err.c_str());
    expect(plan.matches(1, 5), "plan matches dims");
    expect((int) plan.targets.size() == 3, "implicit cpu fallback added");
    expect(plan.at(0, 0).target_index == 0, "best expert on first target");
    expect(plan.at(0, 1).target_index == 0, "second expert on first target");
    expect(plan.at(0, 2).target_index == 1, "third expert on second target");
    expect(plan.at(0, 3).target_index == 2, "overflow expert on cpu");
    expect(plan.at(0, 4).target_index == 2, "tail expert on cpu");
}

static void test_hybrid_placement_uses_expert_split_planner() {
    MoeHybridRoutingStats stats;
    stats.n_layer = 2;
    stats.n_expert = 4;
    stats.n_expert_used = 2;
    stats.counts = {
        100, 80, 60, 40,
        100, 1, 1, 1,
    };
    stats.layer_totals = {280, 103};

    const std::vector<uint64_t> layer_bytes = {1, 1};
    MoeHybridPlacement placement;
    std::string err;
    expect(MoeHybridPlacement::build_from_stats_with_layer_bytes(
               stats, layer_bytes, /*total_hot_budget_bytes=*/4,
               /*min_hot_per_layer=*/1, placement, &err),
           err.c_str());
    expect(placement.hot_counts.size() == 2, "hot counts size");
    expect(placement.hot_counts[0] == 3, "layer0 got extra hot slots");
    expect(placement.hot_counts[1] == 1, "layer1 kept minimum hot slot");
    expect(placement.is_hot(0, 0), "layer0 expert0 hot");
    expect(placement.is_hot(0, 1), "layer0 expert1 hot");
    expect(placement.is_hot(0, 2), "layer0 expert2 hot");
    expect(!placement.is_hot(0, 3), "layer0 expert3 cold");
    expect(placement.is_hot(1, 0), "layer1 expert0 hot");
}

int main() {
    test_ordered_targets_and_cpu_fallback();
    test_hybrid_placement_uses_expert_split_planner();
    std::printf("OK\n");
    return 0;
}
