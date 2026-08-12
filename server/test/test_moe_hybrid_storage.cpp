#include "CppUnitTestFramework.hpp"
#include "../src/common/moe_hybrid_ffn_eval.h"
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

TEST_CASE(MoeHybridStorageFixture, heterogeneous_route_balance_scales_with_model_top_k) {
    REQUIRE(moe_balanced_main_slots_x4(4, 4.4) == 13);
    REQUIRE(moe_balanced_main_slots_x4(6, 4.4) == 20);
    REQUIRE(moe_balanced_main_slots_x4(0, 4.4) == 0);
    REQUIRE(moe_balanced_main_slots_x4(6, 0.0) == 0);
}
