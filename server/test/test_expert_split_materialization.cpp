#include "../src/common/expert_split_materialization.h"
#include "../src/common/expert_split_plan.h"
#include "../src/common/expert_split_runtime.h"

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

static ExpertSplitPlan build_multi_target_plan() {
    ExpertSplitConfig cfg;
    cfg.n_layer = 2;
    cfg.n_expert = 4;

    const std::vector<ExpertSplitTarget> targets = {
        {"cuda:0", "cuda", 0, 4, 0, false},
        {"hip:0", "hip", 0, 2, 0, false},
    };

    const std::vector<ExpertSplitUnit> units = {
        {0, 0, 1, 100.0}, {0, 1, 1, 90.0}, {0, 2, 1, 80.0}, {0, 3, 1, 70.0},
        {1, 0, 1, 60.0},  {1, 1, 1, 50.0}, {1, 2, 1, 40.0}, {1, 3, 1, 30.0},
    };

    ExpertSplitPlan plan;
    std::string err;
    expect(build_expert_split_plan(cfg, targets, units, plan, &err), err.c_str());
    return plan;
}

static void test_materialization_builds_ordered_cold_union() {
    const ExpertSplitPlan plan = build_multi_target_plan();
    ExpertSplitRuntime runtime;
    std::string err;
    expect(build_expert_split_runtime(plan, runtime, &err), err.c_str());

    ExpertSplitMaterialization materialization;
    expect(build_expert_split_materialization(runtime, /*n_expert_used=*/2,
                                              materialization, &err),
           err.c_str());

    expect(materialization.matches(2, 4, 2), "materialization dims");
    expect(materialization.ordered_cold_union, "ordered union enabled");
    expect(materialization.targets.size() == 3, "targets include cpu fallback");
    expect(materialization.primary_placement.total_hot == 4, "primary hot total");

    const auto & l0_hot = materialization.primary_placement.hot_expert_ids[0];
    const auto & l1_hot = materialization.primary_placement.hot_expert_ids[1];
    expect(l0_hot.size() == 4, "layer0 primary hot count");
    expect(l1_hot.empty(), "layer1 primary hot empty");
    expect(l0_hot[0] == 0 && l0_hot[1] == 1 && l0_hot[2] == 2 && l0_hot[3] == 3,
           "layer0 primary hot order");

    const auto & l0_cold = materialization.cold_expert_ids_by_layer[0];
    const auto & l1_cold = materialization.cold_expert_ids_by_layer[1];
    expect(l0_cold.empty(), "layer0 cold empty");
    expect(l1_cold.size() == 4, "layer1 cold count");
    expect(l1_cold[0] == 0 && l1_cold[1] == 1 && l1_cold[2] == 2 &&
           l1_cold[3] == 3, "layer1 cold preserves target order");
}

static void test_materialization_single_explicit_gpu_keeps_single_primary_shape() {
    ExpertSplitConfig cfg;
    cfg.n_layer = 1;
    cfg.n_expert = 4;

    const std::vector<ExpertSplitTarget> targets = {
        {"cuda:0", "cuda", 0, 2, 0, false},
    };
    const std::vector<ExpertSplitUnit> units = {
        {0, 0, 1, 40.0}, {0, 1, 1, 30.0}, {0, 2, 1, 20.0}, {0, 3, 1, 10.0},
    };

    ExpertSplitPlan plan;
    std::string err;
    expect(build_expert_split_plan(cfg, targets, units, plan, &err), err.c_str());

    ExpertSplitRuntime runtime;
    expect(build_expert_split_runtime(plan, runtime, &err), err.c_str());

    ExpertSplitMaterialization materialization;
    expect(build_expert_split_materialization(runtime, /*n_expert_used=*/2,
                                              materialization, &err),
           err.c_str());
    expect(materialization.matches(1, 4, 2), "single-primary materialization dims");
    expect(!materialization.ordered_cold_union, "single-primary path keeps existing cold order");
    expect(materialization.cold_expert_ids_by_layer.empty(), "no cold override");
}

int main() {
    test_materialization_builds_ordered_cold_union();
    test_materialization_single_explicit_gpu_keeps_single_primary_shape();
    std::printf("OK\n");
    return 0;
}
