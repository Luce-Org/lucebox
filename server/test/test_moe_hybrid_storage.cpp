#include "CppUnitTestFramework.hpp"
#include "../src/common/heterogeneous_stage_planner.h"
#include "../src/common/moe_hybrid_storage.h"

#include <cstdint>
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

TEST_CASE(MoeHybridStorageFixture, heterogeneous_stage_split_is_aligned_and_complete) {
    const HeterogeneousStagePlan plan =
        plan_heterogeneous_stage_width(2048, 32, 0.23);

    REQUIRE(plan.split());
    REQUIRE(plan.main_width == 1568);
    REQUIRE(plan.peer_width == 480);
    REQUIRE(plan.main_width + plan.peer_width == 2048);
    REQUIRE(plan.main_width % 32 == 0);
    REQUIRE(plan.peer_width % 32 == 0);
}

TEST_CASE(MoeHybridStorageFixture, heterogeneous_stage_split_rejects_unsafe_shapes) {
    const HeterogeneousStagePlan unaligned =
        plan_heterogeneous_stage_width(2050, 32, 0.25);
    const HeterogeneousStagePlan too_narrow =
        plan_heterogeneous_stage_width(32, 32, 0.5);
    const HeterogeneousStagePlan empty_balanced =
        plan_balanced_heterogeneous_stage_width(0, 32, 3.0, 1.0);

    REQUIRE(!unaligned.split());
    REQUIRE(unaligned.main_width == 2050);
    REQUIRE(!too_narrow.split());
    REQUIRE(too_narrow.main_width == 32);
    REQUIRE(!empty_balanced.valid());
}

TEST_CASE(MoeHybridStorageFixture, heterogeneous_stage_split_keeps_both_owners_for_interior_fraction) {
    const HeterogeneousStagePlan small_peer =
        plan_heterogeneous_stage_width(2048, 32, 0.001);
    const HeterogeneousStagePlan small_main =
        plan_heterogeneous_stage_width(2048, 32, 0.999);

    REQUIRE(small_peer.split());
    REQUIRE(small_peer.peer_width == 32);
    REQUIRE(small_main.split());
    REQUIRE(small_main.main_width == 32);
}

TEST_CASE(MoeHybridStorageFixture, heterogeneous_stage_partition_supports_whole_stage_ownership) {
    const HeterogeneousStagePlan main =
        plan_heterogeneous_stage_width(2048, 32, 0.0);
    const HeterogeneousStagePlan peer =
        plan_heterogeneous_stage_width(2048, 32, 1.0);

    REQUIRE(main.valid());
    REQUIRE(!main.uses_peer());
    REQUIRE(main.main_width == 2048);
    REQUIRE(peer.valid());
    REQUIRE(peer.uses_peer());
    REQUIRE(!peer.split());
    REQUIRE(peer.main_width == 0);
    REQUIRE(peer.peer_width == 2048);
}

TEST_CASE(MoeHybridStorageFixture, heterogeneous_stage_balance_accounts_for_owner_rates_and_fixed_work) {
    const HeterogeneousStagePlan rate_only =
        plan_balanced_heterogeneous_stage_width(
            2048, 32, 3.0, 1.0);
    const HeterogeneousStagePlan main_already_busy =
        plan_balanced_heterogeneous_stage_width(
            2048, 32, 3.0, 1.0, 128.0, 0.0);

    REQUIRE(rate_only.split());
    REQUIRE(rate_only.peer_width == 512);
    REQUIRE(main_already_busy.split());
    REQUIRE(main_already_busy.peer_width == 544);
}
