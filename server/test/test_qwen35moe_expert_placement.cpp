#include "CppUnitTestFramework.hpp"
#include "../src/common/moe_hybrid_placement.h"
#include "../src/common/moe_hybrid_routing_stats.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

using namespace dflash::common;

#define TEST_ASSERT(cond) do { \
    auto _cpputf_exception = CppUnitTestFramework::Assert::IsTrue(static_cast<bool>(cond), #cond); \
    if (_cpputf_exception) { \
        throw *_cpputf_exception; \
    } \
} while (0)
static void expect(bool cond, const char * msg) { (void) msg; TEST_ASSERT(cond); }

namespace {
struct Qwen35MoeExpertPlacementFixture {};
}

TEST_CASE(Qwen35MoeExpertPlacementFixture, moe_expert_placement_suite) {
    MoeHybridRoutingStats stats;
    stats.n_layer = 2;
    stats.n_expert = 4;
    stats.n_expert_used = 2;
    stats.counts = {
        100, 80, 60, 40,   // layer 0
        100, 1, 1, 1       // layer 1
    };
    stats.layer_totals = {280, 103};

    MoeHybridPlacement placement;
    std::string err;
    expect(MoeHybridPlacement::build_from_stats(stats, /*total_hot_budget=*/4,
                                                /*min_hot_per_layer=*/1,
                                                placement, &err), err.c_str());
    expect(placement.n_layer == 2, "n_layer");
    expect(placement.hot_counts.size() == 2, "hot_counts size");
    expect(placement.hot_counts[0] == 3, "layer0 got extra hot slots");
    expect(placement.hot_counts[1] == 1, "layer1 kept minimum hot slot");
    expect(placement.is_hot(0, 0), "layer0 expert0 hot");
    expect(placement.is_hot(0, 1), "layer0 expert1 hot");
    expect(placement.is_hot(0, 2), "layer0 expert2 hot");
    expect(!placement.is_hot(0, 3), "layer0 expert3 cold");
    expect(placement.is_hot(1, 0), "layer1 expert0 hot");
    expect(!placement.is_hot(1, 1), "layer1 expert1 cold");

    expect(placement.matches(2, 4, 2), "placement matches dims");

    const auto tmp = std::filesystem::temp_directory_path() / "moe-hybrid-placement-test.json";
    expect(placement.save_json(tmp.string(), "moe_hybrid", &err), err.c_str());
    MoeHybridPlacement loaded;
    expect(MoeHybridPlacement::load_json(tmp.string(), loaded, &err), err.c_str());
    expect(loaded.hot_counts == placement.hot_counts, "loaded hot counts");
    expect(loaded.hot_expert_ids == placement.hot_expert_ids, "loaded hot ids");
    std::filesystem::remove(tmp);

    std::printf("OK\n");
}
