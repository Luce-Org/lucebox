#include "../src/common/expert_split_plan.h"
#include "../src/common/expert_split_compute_runtime.h"
#include "../src/common/expert_split_materialization.h"
#include "../src/common/expert_split_runtime.h"
#include "../src/common/expert_split_state.h"

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

static ExpertSplitPlan build_sample_plan() {
    ExpertSplitConfig cfg;
    cfg.n_layer = 2;
    cfg.n_expert = 4;

    const std::vector<ExpertSplitTarget> targets = {
        {"cuda:0", "cuda", 0, 4, 0, false},
        {"hip:0", "hip", 0, 2, 0, false},
    };
    const std::vector<ExpertSplitUnit> units = {
        {0, 0, 1, 100.0},
        {0, 1, 1, 90.0},
        {0, 2, 1, 80.0},
        {0, 3, 1, 70.0},
        {1, 0, 1, 60.0},
        {1, 1, 1, 50.0},
        {1, 2, 1, 40.0},
        {1, 3, 1, 30.0},
    };

    ExpertSplitPlan plan;
    std::string err;
    expect(build_expert_split_plan(cfg, targets, units, plan, &err), err.c_str());
    return plan;
}

static void test_runtime_materializes_targets_and_cpu_fallback() {
    const ExpertSplitPlan plan = build_sample_plan();
    ExpertSplitRuntime runtime;
    std::string err;
    expect(build_expert_split_runtime(plan, runtime, &err), err.c_str());

    expect(runtime.matches(2, 4), "runtime dims");
    expect((int) runtime.targets.size() == 3, "runtime keeps cpu fallback target");
    expect(runtime.targets[0].target.name == "cuda:0", "target0 name");
    expect(runtime.targets[1].target.name == "hip:0", "target1 name");
    expect(runtime.targets[2].target.backend == "cpu", "target2 cpu fallback");

    expect(runtime.target_index(0, 0) == 0, "layer0 expert0 on target0");
    expect(runtime.target_index(0, 1) == 0, "layer0 expert1 on target0");
    expect(runtime.target_index(0, 2) == 0, "layer0 expert2 on target0");
    expect(runtime.target_index(0, 3) == 0, "layer0 expert3 on target0");
    expect(runtime.target_index(1, 0) == 1, "layer1 expert0 on target1");
    expect(runtime.target_index(1, 1) == 1, "layer1 expert1 on target1");
    expect(runtime.target_index(1, 2) == 2, "layer1 expert2 on cpu");
    expect(runtime.target_index(1, 3) == 2, "layer1 expert3 on cpu");

    expect(runtime.targets[0].used_bytes == 4, "target0 bytes");
    expect(runtime.targets[1].used_bytes == 2, "target1 bytes");
    expect(runtime.targets[2].used_bytes == 2, "target2 bytes");
}

static void test_runtime_local_order_is_score_stable() {
    const ExpertSplitPlan plan = build_sample_plan();
    ExpertSplitRuntime runtime;
    std::string err;
    expect(build_expert_split_runtime(plan, runtime, &err), err.c_str());

    const ExpertSplitLayerTarget * layer0_t0 = runtime.layer_target_ptr(0, 0);
    expect(layer0_t0 != nullptr, "layer0 target0 exists");
    expect(layer0_t0->global_expert_ids.size() == 4, "layer0 target0 expert count");
    expect(layer0_t0->global_expert_ids[0] == 0, "layer0 target0 local0");
    expect(layer0_t0->global_expert_ids[1] == 1, "layer0 target0 local1");
    expect(layer0_t0->global_expert_ids[2] == 2, "layer0 target0 local2");
    expect(layer0_t0->global_expert_ids[3] == 3, "layer0 target0 local3");
    expect(runtime.local_index(0, 0) == 0, "layer0 expert0 local");
    expect(runtime.local_index(0, 3) == 3, "layer0 expert3 local");

    const ExpertSplitLayerTarget * layer1_t2 = runtime.layer_target_ptr(2, 1);
    expect(layer1_t2 != nullptr, "layer1 cpu exists");
    expect(layer1_t2->global_expert_ids.size() == 2, "layer1 cpu expert count");
    expect(layer1_t2->global_expert_ids[0] == 2, "layer1 cpu local0 by score");
    expect(layer1_t2->global_expert_ids[1] == 3, "layer1 cpu local1 by score");
    expect(runtime.local_index(1, 2) == 0, "layer1 expert2 local");
    expect(runtime.local_index(1, 3) == 1, "layer1 expert3 local");
}

static void test_target_placement_roundtrip() {
    const ExpertSplitPlan plan = build_sample_plan();
    ExpertSplitRuntime runtime;
    std::string err;
    expect(build_expert_split_runtime(plan, runtime, &err), err.c_str());

    MoeHybridPlacement t0;
    expect(build_expert_split_target_placement(runtime, /*n_expert_used=*/2, 0, t0, &err),
           err.c_str());
    expect(t0.matches(2, 4, 2), "target0 placement dims");
    expect(t0.total_hot == 4, "target0 total hot");
    expect(t0.hot_counts[0] == 4, "target0 layer0 count");
    expect(t0.hot_counts[1] == 0, "target0 layer1 count");
    expect(t0.hot_expert_ids[0].size() == 4, "target0 layer0 ids");
    expect(t0.hot_expert_ids[1].empty(), "target0 layer1 empty");

    MoeHybridPlacement t2;
    expect(build_expert_split_target_placement(runtime, /*n_expert_used=*/2, 2, t2, &err),
           err.c_str());
    expect(t2.total_hot == 2, "target2 total hot");
    expect(t2.hot_counts[0] == 0, "target2 layer0 count");
    expect(t2.hot_counts[1] == 2, "target2 layer1 count");
    expect(t2.hot_expert_ids[1].size() == 2, "target2 layer1 ids");
    expect(t2.hot_expert_ids[1][0] == 2, "target2 layer1 first");
    expect(t2.hot_expert_ids[1][1] == 3, "target2 layer1 second");
}

static void test_all_target_placements_materialize() {
    const ExpertSplitPlan plan = build_sample_plan();
    ExpertSplitRuntime runtime;
    std::string err;
    expect(build_expert_split_runtime(plan, runtime, &err), err.c_str());

    std::vector<ExpertSplitTargetPlacement> placements;
    expect(build_all_expert_split_target_placements(runtime, /*n_expert_used=*/2,
                                                    placements, &err),
           err.c_str());
    expect(placements.size() == 3, "all target placements count");
    expect(placements[0].target_index == 0, "placement0 index");
    expect(placements[1].target_index == 1, "placement1 index");
    expect(placements[2].target_index == 2, "placement2 index");
    expect(placements[0].placement.total_hot == 4, "placement0 total hot");
    expect(placements[1].placement.total_hot == 2, "placement1 total hot");
    expect(placements[2].placement.total_hot == 2, "placement2 total hot");
}

static void test_compute_runtime_materializes_target_views() {
    const ExpertSplitPlan plan = build_sample_plan();
    ExpertSplitRuntime runtime;
    std::string err;
    expect(build_expert_split_runtime(plan, runtime, &err), err.c_str());

    ExpertSplitComputeRuntime compute_runtime;
    expect(build_expert_split_compute_runtime(runtime, /*n_expert_used=*/2,
                                              compute_runtime, &err),
           err.c_str());
    expect(compute_runtime.matches(2, 4, 2), "compute runtime dims");
    expect(compute_runtime.targets.size() == 3, "compute runtime targets");
    expect(compute_runtime.targets[0].target.name == "cuda:0", "compute target0");
    expect(compute_runtime.targets[1].target.name == "hip:0", "compute target1");
    expect(compute_runtime.targets[2].target.backend == "cpu", "compute target2");
    expect(compute_runtime.targets[0].placement.total_hot == 4, "compute target0 hot");
    expect(compute_runtime.targets[1].placement.total_hot == 2, "compute target1 hot");
    expect(compute_runtime.targets[2].placement.total_hot == 2, "compute target2 hot");
    expect(compute_runtime.targets[0].placement.hot_expert_ids[0][0] == 0, "compute t0 l0 e0");
    expect(compute_runtime.targets[1].placement.hot_expert_ids[1][0] == 0, "compute t1 l1 e0");
    expect(compute_runtime.targets[1].placement.hot_expert_ids[1][1] == 1, "compute t1 l1 e1");
    expect(compute_runtime.targets[2].placement.hot_expert_ids[1][0] == 2, "compute t2 l1 e2");
    expect(compute_runtime.targets[2].placement.hot_expert_ids[1][1] == 3, "compute t2 l1 e3");
    expect(compute_runtime.target_index(1, 2) == 2, "compute target lookup l1 e2");
    expect(compute_runtime.local_index(1, 2) == 0, "compute local lookup l1 e2");
    expect(compute_runtime.target_index(1, 3) == 2, "compute target lookup l1 e3");
    expect(compute_runtime.local_index(1, 3) == 1, "compute local lookup l1 e3");
    expect(compute_runtime.target_index(-1, 0) == -1, "compute target lookup rejects layer");
    expect(compute_runtime.local_index(0, 4) == -1, "compute local lookup rejects expert");
}

static void test_compute_runtime_preserves_target_local_order() {
    const ExpertSplitPlan plan = build_sample_plan();
    ExpertSplitRuntime runtime;
    std::string err;
    expect(build_expert_split_runtime(plan, runtime, &err), err.c_str());

    ExpertSplitComputeRuntime compute_runtime;
    expect(build_expert_split_compute_runtime(runtime, /*n_expert_used=*/2,
                                              compute_runtime, &err),
           err.c_str());

    const auto & hip_layer = compute_runtime.targets[1].placement.hot_expert_ids[1];
    const auto & cpu_layer = compute_runtime.targets[2].placement.hot_expert_ids[1];
    expect(hip_layer.size() == 2, "hip layer count");
    expect(cpu_layer.size() == 2, "cpu layer count");
    expect(hip_layer[0] == 0 && hip_layer[1] == 1, "hip local order follows score");
    expect(cpu_layer[0] == 2 && cpu_layer[1] == 3, "cpu local order follows score");
}

static ExpertSplitPlan build_reordered_target_plan() {
    ExpertSplitPlan plan;
    plan.n_layer = 1;
    plan.n_expert = 4;
    plan.targets = {
        {"cuda:0", "cuda", 0, 2, 0, false},
        {"hip:0", "hip", 0, 2, 0, false},
    };
    plan.target_used_bytes = {2, 2};
    plan.assignments = {
        {0, 1, 10.0},
        {1, 1, 9.0},
        {0, 1, 8.0},
        {1, 1, 7.0},
    };
    return plan;
}

static void test_compute_runtime_fingerprint_changes_with_assignment() {
    ExpertSplitRuntime runtime_a;
    ExpertSplitRuntime runtime_b;
    std::string err;

    expect(build_expert_split_runtime(build_sample_plan(), runtime_a, &err), err.c_str());
    expect(build_expert_split_runtime(build_reordered_target_plan(), runtime_b, &err), err.c_str());

    ExpertSplitComputeRuntime compute_a;
    ExpertSplitComputeRuntime compute_b;
    expect(build_expert_split_compute_runtime(runtime_a, /*n_expert_used=*/2,
                                              compute_a, &err),
           err.c_str());
    expect(build_expert_split_compute_runtime(runtime_b, /*n_expert_used=*/2,
                                              compute_b, &err),
           err.c_str());

    const uint64_t fp_a = expert_split_compute_runtime_fingerprint(compute_a);
    const uint64_t fp_b = expert_split_compute_runtime_fingerprint(compute_b);
    expect(fp_a != fp_b, "compute runtime fingerprint changes when placement changes");
}

static void test_materialization_builds_ordered_cold_union() {
    const ExpertSplitPlan plan = build_sample_plan();
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

static void test_builds_shared_expert_split_state_components() {
    ExpertSplitConfig cfg;
    cfg.n_layer = 2;
    cfg.n_expert = 3;

    const std::vector<ExpertSplitTarget> targets = {
        {"cuda:0", "cuda", 0, 3, 0, false},
        {"hip:0", "hip", 0, 1, 0, false},
    };
    const std::vector<ExpertSplitUnit> units = {
        {0, 0, 1, 100.0},
        {0, 1, 1, 90.0},
        {0, 2, 1, 80.0},
        {1, 0, 1, 70.0},
        {1, 1, 1, 60.0},
        {1, 2, 1, 50.0},
    };

    ExpertSplitStateComponents state;
    std::string err;
    expect(build_expert_split_state(cfg, targets, units, /*n_expert_used=*/2,
                                    state, &err),
           err.c_str());
    expect(state.matches(2, 3, 2), "shared state matches dims");
    expect(state.plan.at(0, 0).target_index == 0, "plan target lookup");
    expect(state.runtime.target_index(1, 2) == 2, "runtime includes cpu fallback");
    expect(state.compute_runtime.local_index(1, 2) == 1, "compute runtime local id");
    expect(state.materialization.ordered_cold_union, "materialization ordered cold union");
}

static void test_builds_generic_layer_mapping() {
    ExpertSplitLayerMapping mapping;
    std::string err;
    expect(build_expert_split_layer_mapping(
               /*n_total_layer=*/5, {1, 3}, mapping, &err),
           err.c_str());
    expect(mapping.n_total_layer == 5, "mapping total layers");
    expect(mapping.split_layer_count() == 2, "mapping split count");
    expect(mapping.physical_layer_by_split_layer[0] == 1, "mapping split0");
    expect(mapping.physical_layer_by_split_layer[1] == 3, "mapping split1");
    expect(mapping.split_layer_for_physical(1) == 0, "mapping reverse 1");
    expect(mapping.split_layer_for_physical(3) == 1, "mapping reverse 3");
    expect(mapping.split_layer_for_physical(0) == -1, "mapping dense lead");
    expect(mapping.is_split_layer(3), "mapping is split layer");
    expect(!mapping.is_split_layer(4), "mapping rejects dense layer");
}

static void test_builds_contiguous_layer_mapping() {
    ExpertSplitLayerMapping mapping;
    std::string err;
    expect(build_contiguous_expert_split_layer_mapping(
               /*n_total_layer=*/4, /*first_split_layer=*/1,
               /*n_split_layer=*/3, mapping, &err),
           err.c_str());
    expect(mapping.physical_layer_by_split_layer.size() == 3, "contiguous mapping size");
    expect(mapping.physical_layer_by_split_layer[0] == 1, "contiguous split0");
    expect(mapping.physical_layer_by_split_layer[1] == 2, "contiguous split1");
    expect(mapping.physical_layer_by_split_layer[2] == 3, "contiguous split2");
    expect(mapping.split_layer_for_physical(1) == 0, "contiguous reverse1");
    expect(mapping.split_layer_for_physical(2) == 1, "contiguous reverse2");
    expect(mapping.split_layer_for_physical(3) == 2, "contiguous reverse3");
}

static void test_runtime_rejects_invalid_plan() {
    ExpertSplitPlan plan;
    plan.n_layer = 1;
    plan.n_expert = 1;
    plan.targets = {{"cpu", "cpu", -1, 0, 0, true}};
    plan.target_used_bytes = {0};
    plan.assignments = {{-1, 0, 0.0}};

    ExpertSplitRuntime runtime;
    std::string err;
    expect(!build_expert_split_runtime(plan, runtime, &err), "invalid plan rejected");
    expect(!err.empty(), "invalid plan error");
}

int main() {
    test_runtime_materializes_targets_and_cpu_fallback();
    test_runtime_local_order_is_score_stable();
    test_target_placement_roundtrip();
    test_all_target_placements_materialize();
    test_compute_runtime_materializes_target_views();
    test_compute_runtime_preserves_target_local_order();
    test_compute_runtime_fingerprint_changes_with_assignment();
    test_materialization_builds_ordered_cold_union();
    test_materialization_single_explicit_gpu_keeps_single_primary_shape();
    test_builds_shared_expert_split_state_components();
    test_builds_generic_layer_mapping();
    test_builds_contiguous_layer_mapping();
    test_runtime_rejects_invalid_plan();
    std::printf("OK\n");
    return 0;
}
