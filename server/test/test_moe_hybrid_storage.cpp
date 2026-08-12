#include "CppUnitTestFramework.hpp"
#include "../src/common/heterogeneous_stage_planner.h"
#include "../src/common/moe_hybrid_storage.h"

#include <cstdint>
#include <limits>
#include <vector>

using namespace dflash::common;

namespace {
struct MoeHybridStorageFixture {};
}

TEST_CASE(MoeHybridStorageFixture, expert_residency_tracks_model_sized_expert_sets) {
    MoeHybridLayerStorage storage;
    storage.reset_expert_vram_mask(320);

    storage.set_expert_hot(0);
    storage.set_expert_hot(255);
    storage.set_expert_hot(256);
    storage.set_expert_hot(319);

    REQUIRE(storage.is_expert_hot(0));
    REQUIRE(storage.is_expert_hot(255));
    REQUIRE(storage.is_expert_hot(256));
    REQUIRE(storage.is_expert_hot(319));
    REQUIRE(!storage.is_expert_hot(320));

    const std::vector<int32_t> all_hot = {0, 256, 319, -1};
    REQUIRE(storage.all_routed_are_hot(all_hot.data(), (int)all_hot.size()));

    const std::vector<int32_t> includes_cold = {0, 257};
    REQUIRE(!storage.all_routed_are_hot(includes_cold.data(), (int)includes_cold.size()));

    storage.clear_expert_hot(256);
    REQUIRE(!storage.is_expert_hot(256));
    REQUIRE(!storage.all_routed_are_hot(all_hot.data(), (int)all_hot.size()));
}

TEST_CASE(MoeHybridStorageFixture, heterogeneous_stage_split_rejects_invalid_inputs) {
    const HeterogeneousStagePlan unaligned =
        plan_balanced_heterogeneous_stage_width(2050, 32, 3.0, 1.0);
    const HeterogeneousStagePlan invalid_alignment =
        plan_balanced_heterogeneous_stage_width(2048, 0, 3.0, 1.0);
    const HeterogeneousStagePlan empty_balanced =
        plan_balanced_heterogeneous_stage_width(0, 32, 3.0, 1.0);

    REQUIRE(unaligned.valid());
    REQUIRE(unaligned.peer_width == 0);
    REQUIRE(unaligned.main_width == 2050);
    REQUIRE(invalid_alignment.valid());
    REQUIRE(invalid_alignment.peer_width == 0);
    REQUIRE(invalid_alignment.main_width == 2048);
    REQUIRE(!empty_balanced.valid());
}

TEST_CASE(MoeHybridStorageFixture, heterogeneous_stage_balance_compares_aligned_candidates) {
    const HeterogeneousStagePlan plan =
        plan_balanced_heterogeneous_stage_width(100, 10, 17.0, 3.0);

    REQUIRE(plan.valid());
    REQUIRE(plan.main_width == 90);
    REQUIRE(plan.peer_width == 10);
}

TEST_CASE(MoeHybridStorageFixture, heterogeneous_stage_balance_handles_extreme_values) {
    const double largest = std::numeric_limits<double>::max();
    const HeterogeneousStagePlan one_busy_owner =
        plan_balanced_heterogeneous_stage_width(
            100, 10, largest, largest, largest, 0.0);
    const HeterogeneousStagePlan both_busy =
        plan_balanced_heterogeneous_stage_width(
            100, 10, largest, largest, largest, largest);
    const HeterogeneousStagePlan negligible_main =
        plan_balanced_heterogeneous_stage_width(
            100, 10, std::numeric_limits<double>::denorm_min(), largest);

    REQUIRE(one_busy_owner.valid());
    REQUIRE(one_busy_owner.main_width == 0);
    REQUIRE(one_busy_owner.peer_width == 100);
    REQUIRE(both_busy.valid());
    REQUIRE(both_busy.main_width == 50);
    REQUIRE(both_busy.peer_width == 50);
    REQUIRE(negligible_main.main_width == 0);
    REQUIRE(negligible_main.peer_width == 100);
}

TEST_CASE(MoeHybridStorageFixture, heterogeneous_stage_balance_accounts_for_fixed_work) {
    const HeterogeneousStagePlan rate_only =
        plan_balanced_heterogeneous_stage_width(2048, 32, 3.0, 1.0);
    const HeterogeneousStagePlan main_already_busy =
        plan_balanced_heterogeneous_stage_width(
            2048, 32, 3.0, 1.0, 128.0, 0.0);

    REQUIRE(rate_only.valid());
    REQUIRE(rate_only.main_width > 0);
    REQUIRE(rate_only.peer_width == 512);
    REQUIRE(main_already_busy.valid());
    REQUIRE(main_already_busy.main_width > 0);
    REQUIRE(main_already_busy.peer_width == 544);
}

TEST_CASE(MoeHybridStorageFixture, heterogeneous_route_balance_scales_with_model_top_k) {
    const HeterogeneousStagePlan top4 =
        plan_balanced_heterogeneous_stage_width(4 * 4, 1, 4.4, 1.0);
    const HeterogeneousStagePlan top6 =
        plan_balanced_heterogeneous_stage_width(4 * 6, 1, 4.4, 1.0);

    REQUIRE(top4.main_width == 13);
    REQUIRE(top4.peer_width == 3);
    REQUIRE(top6.main_width == 20);
    REQUIRE(top6.peer_width == 4);
}
