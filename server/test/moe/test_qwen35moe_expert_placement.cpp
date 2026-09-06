#include "CppUnitTestFramework.hpp"
#include "common/moe_hybrid_placement.h"
#include "common/moe_hybrid_routing_stats.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
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

    MoeHybridPlacement malformed = placement;
    malformed.hot_expert_ids[0][1] = malformed.hot_expert_ids[0][0];
    REQUIRE(!malformed.valid(&err));
    REQUIRE(err == "placement contains a duplicate expert at layer 0");

    {
        std::ofstream stream(tmp);
        stream << R"({
            "n_layer": 1,
            "n_expert": 4,
            "n_expert_used": 2,
            "total_hot": 2,
            "hot_counts": [2],
            "hot_expert_ids": [[1, 1]]
        })";
    }
    REQUIRE(!MoeHybridPlacement::load_json(tmp.string(), loaded, &err));
    REQUIRE(err == "placement contains a duplicate expert at layer 0");
    std::filesystem::remove(tmp);
    // Aggregate hit-rate placement can overfeed a highly skewed layer while a
    // flat layer remains peer-bound. The critical-path model stops at each
    // layer's branch crossover and may deliberately leave spare memory unused.
    MoeHybridRoutingStats balance_stats;
    balance_stats.n_layer = 2;
    balance_stats.n_expert = 4;
    balance_stats.n_expert_used = 2;
    balance_stats.counts = {
        100, 100, 100, 100,  // flat: needs three hot experts
        400,   1,   1,   1,  // skewed: one hot expert is sufficient
    };
    balance_stats.layer_totals = {400, 403};

    MoeHybridCriticalPathConfig balance_cfg;
    balance_cfg.active_experts = 2;
    balance_cfg.main_to_peer_rate = 3.0;
    MoeHybridPlacement balanced;
    REQUIRE(MoeHybridPlacement::build_critical_path_balanced_from_stats(
        balance_stats,
        /*layer_expert_bytes=*/{100, 100},
        /*layer_main_fixed_bytes=*/{100, 100},
        /*total_hot_budget_bytes=*/600,
        balance_cfg, balanced, &err));
    REQUIRE(balanced.hot_counts == std::vector<int>({3, 1}));
    REQUIRE(balanced.total_hot == 4);
    REQUIRE(balanced.is_hot(0, 0));
    REQUIRE(balanced.is_hot(0, 1));
    REQUIRE(balanced.is_hot(0, 2));
    REQUIRE(balanced.is_hot(1, 0));

    // Ratio-greedy selection picks the four-byte layer first and strands two
    // bytes. The exact frontier correctly spends all six bytes on the larger
    // improvement from layer 0.
    MoeHybridRoutingStats unequal_size_stats;
    unequal_size_stats.n_layer = 2;
    unequal_size_stats.n_expert = 2;
    unequal_size_stats.n_expert_used = 1;
    unequal_size_stats.counts = {
        80, 20,
        90, 10,
    };
    unequal_size_stats.layer_totals = {100, 100};
    MoeHybridCriticalPathConfig unequal_size_cfg;
    unequal_size_cfg.active_experts = 1;
    unequal_size_cfg.main_to_peer_rate = 100.0;
    MoeHybridPlacement unequal_size;
    REQUIRE(MoeHybridPlacement::build_critical_path_balanced_from_stats(
        unequal_size_stats,
        /*layer_expert_bytes=*/{6, 4},
        /*layer_main_fixed_bytes=*/{0, 0},
        /*total_hot_budget_bytes=*/6,
        unequal_size_cfg, unequal_size, &err));
    REQUIRE(unequal_size.hot_counts == std::vector<int>({1, 0}));
    REQUIRE(unequal_size.is_hot(0, 0));

    balance_cfg.max_hot_per_layer = 2;
    MoeHybridPlacement capped;
    REQUIRE(MoeHybridPlacement::build_critical_path_balanced_from_stats(
        balance_stats, {100, 100}, {100, 100}, 600,
        balance_cfg, capped, &err));
    REQUIRE(capped.hot_counts == std::vector<int>({2, 1}));
    balance_cfg.max_hot_per_layer = 0;

    // A phase-specific decode placement remains intact while a second profile
    // spends otherwise-unused bytes on prefill residency.
    MoeHybridRoutingStats residency_stats;
    residency_stats.n_layer = 2;
    residency_stats.n_expert = 4;
    residency_stats.n_expert_used = 2;
    residency_stats.counts = {
        1, 2, 3, 500,
        1, 300, 400, 1000,
    };
    residency_stats.layer_totals = {506, 1701};
    MoeHybridPlacement expanded = balanced;
    expanded.total_hot = 999;  // Expansion must derive this aggregate from IDs.
    REQUIRE(MoeHybridPlacement::expand_from_stats_with_layer_bytes(
        residency_stats, {100, 100}, 600, expanded, &err));
    REQUIRE(expanded.total_hot == 6);
    REQUIRE(expanded.hot_counts == std::vector<int>({4, 2}));
    REQUIRE(expanded.is_hot(0, 0));
    REQUIRE(expanded.is_hot(0, 1));
    REQUIRE(expanded.is_hot(0, 2));
    REQUIRE(expanded.is_hot(0, 3));
    REQUIRE(expanded.is_hot(1, 0));
    REQUIRE(expanded.is_hot(1, 3));
    REQUIRE(!expanded.is_hot(1, 2));

    MoeHybridPlacement over_budget = balanced;
    REQUIRE(!MoeHybridPlacement::expand_from_stats_with_layer_bytes(
        residency_stats, {100, 100}, 300, over_budget, &err));

    MoeHybridRoutingStats overflow_stats;
    overflow_stats.n_layer = 1;
    overflow_stats.n_expert = 2;
    overflow_stats.n_expert_used = 1;
    overflow_stats.counts = {
        std::numeric_limits<uint64_t>::max(), 1,
    };
    overflow_stats.layer_totals = {0};
    REQUIRE(!MoeHybridPlacement::build_critical_path_balanced_from_stats(
        overflow_stats, {100}, {100}, 200,
        balance_cfg, balanced, &err));
    REQUIRE(err == "routing profile count overflow at layer 0");

    const auto overflow_csv = std::filesystem::temp_directory_path() /
        "moe-hybrid-routing-overflow.csv";
    {
        std::ofstream stream(overflow_csv);
        stream << "# hotness table: n_layer=1 n_expert=2 n_expert_used=1\n"
               << std::numeric_limits<uint64_t>::max() << ",1\n";
    }
    MoeHybridRoutingStats loaded_overflow;
    REQUIRE(!MoeHybridRoutingStats::load_csv(
        overflow_csv.string(), loaded_overflow, &err));
    REQUIRE(err == "routing profile count overflow at layer 0");
    std::filesystem::remove(overflow_csv);
    balance_cfg.main_to_peer_rate = 0.0;
    REQUIRE(!MoeHybridPlacement::build_critical_path_balanced_from_stats(
        balance_stats, {100, 100}, {100, 100}, 600,
        balance_cfg, balanced, &err));
    std::printf("OK\n");
}
