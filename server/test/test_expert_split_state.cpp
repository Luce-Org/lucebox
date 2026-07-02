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

int main() {
    test_builds_shared_expert_split_state_components();
    test_builds_generic_layer_mapping();
    test_builds_contiguous_layer_mapping();
    std::printf("OK\n");
    return 0;
}
