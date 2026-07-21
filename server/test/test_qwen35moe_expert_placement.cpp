#include "CppUnitTestFramework.hpp"
#include "../src/common/moe_hybrid_placement.h"
#include "../src/common/moe_hybrid_routing_stats.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

using namespace dflash::common;

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
    REQUIRE(MoeHybridPlacement::build_from_stats(stats, /*total_hot_budget=*/4,
                                                 /*min_hot_per_layer=*/1,
                                                 placement, &err));
    REQUIRE(placement.n_layer == 2);
    REQUIRE(placement.hot_counts.size() == 2);
    REQUIRE(placement.hot_counts[0] == 3);
    REQUIRE(placement.hot_counts[1] == 1);
    REQUIRE(placement.is_hot(0, 0));
    REQUIRE(placement.is_hot(0, 1));
    REQUIRE(placement.is_hot(0, 2));
    REQUIRE(!placement.is_hot(0, 3));
    REQUIRE(placement.is_hot(1, 0));
    REQUIRE(!placement.is_hot(1, 1));

    REQUIRE(placement.matches(2, 4, 2));

    const auto tmp = std::filesystem::temp_directory_path() / "moe-hybrid-placement-test.json";
    REQUIRE(placement.save_json(tmp.string(), "moe_hybrid", &err));
    MoeHybridPlacement loaded;
    REQUIRE(MoeHybridPlacement::load_json(tmp.string(), loaded, &err));
    REQUIRE(loaded.hot_counts == placement.hot_counts);
    REQUIRE(loaded.hot_expert_ids == placement.hot_expert_ids);
    std::filesystem::remove(tmp);

    std::printf("OK\n");
}
