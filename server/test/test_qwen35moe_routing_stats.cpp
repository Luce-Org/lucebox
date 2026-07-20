#include "CppUnitTestFramework.hpp"
#include "../src/common/moe_hybrid_routing_stats.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {
struct Qwen35MoeRoutingStatsFixture {};
}

TEST_CASE(Qwen35MoeRoutingStatsFixture, moe_routing_stats_suite) {
    MoeHybridRoutingStats stats;
    REQUIRE(stats.init(2, 4, 2));
    REQUIRE(stats.matches(2, 4, 2));

    const int32_t layer0_a[] = {2, 1};
    const int32_t layer0_b[] = {2, 3};
    const int32_t layer1_a[] = {0, 0};

    REQUIRE(stats.observe(0, layer0_a, 2));
    REQUIRE(stats.observe(0, layer0_b, 2));
    REQUIRE(stats.observe(1, layer1_a, 2));

    REQUIRE(stats.count(0, 2) == 2);
    REQUIRE(stats.count(0, 1) == 1);
    REQUIRE(stats.count(0, 3) == 1);
    REQUIRE(stats.count(1, 0) == 2);
    REQUIRE(stats.layer_totals[0] == 4);
    REQUIRE(stats.layer_totals[1] == 2);

    std::vector<int> ranked0 = stats.ranked_experts(0);
    REQUIRE(ranked0.size() == 4);
    REQUIRE(ranked0[0] == 2);

    std::vector<int> hot0 = stats.hot_experts(0, 2);
    REQUIRE(hot0.size() == 2);
    REQUIRE(hot0[0] == 2);

    const auto tmp = std::filesystem::temp_directory_path() / "moe-hybrid-routing-stats-test.csv";
    std::string err;
    REQUIRE(stats.save_csv(tmp.string(), &err));

    MoeHybridRoutingStats loaded;
    REQUIRE(MoeHybridRoutingStats::load_csv(tmp.string(), loaded, &err));
    REQUIRE(loaded.matches(2, 4, 2));
    REQUIRE(loaded.count(0, 2) == 2);
    REQUIRE(loaded.layer_totals[1] == 2);

    std::filesystem::remove(tmp);
    std::printf("OK\n");
}
