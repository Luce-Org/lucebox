#include "CppUnitTestFramework.hpp"
#include "../src/common/moe_hybrid_ffn_eval.h"
#include "../src/common/moe_hybrid_storage.h"

#include <algorithm>
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

TEST_CASE(MoeHybridStorageFixture, dynamic_route_balance_uses_physical_owner_maps) {
    MoeHybridLayerStorage storage;
    storage.hot_local_by_global = {0, -1, 1, -1};
    storage.cold_local_by_global = {0, 1, 2, 3};
    storage.decode_hot_local_by_global = {-1, -1, 1, -1};
    storage.decode_cold_local_by_global = {0, 1, -1, 3};

    const MoeHybridOwnerMapView static_maps =
        moe_hybrid_owner_maps(storage, false);
    REQUIRE(static_maps.main == &storage.decode_hot_local_by_global);
    REQUIRE(static_maps.peer == &storage.decode_cold_local_by_global);

    const MoeHybridOwnerMapView dynamic_maps =
        moe_hybrid_owner_maps(storage, true);
    REQUIRE(dynamic_maps.main == &storage.hot_local_by_global);
    REQUIRE(dynamic_maps.peer == &storage.cold_local_by_global);
    REQUIRE(std::none_of(
        dynamic_maps.peer->begin(), dynamic_maps.peer->end(),
        [](int32_t local) { return local < 0; }));
}

TEST_CASE(MoeHybridStorageFixture, fractional_route_quota_rounds_over_the_batch) {
    ggml_init_params params{
        /*mem_size=*/1024 * 1024,
        /*mem_buffer=*/nullptr,
        /*no_alloc=*/true,
    };
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 6, 5);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6, 5);
    ggml_tensor * local_lut =
        ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 1, 8, 5, 1);
    ggml_tensor * candidate_lut =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 8, 5, 1);
    REQUIRE(ids && weights && local_lut && candidate_lut);

    // top-k 6 at a 3:1 owner rate is 4.5 main routes per token. Across a
    // five-token verifier batch the exact quota is 22.5, which rounds to 23.
    ggml_tensor * owner_ids = ggml_ds4_moe_balanced_owner_ids(
        ctx, ids, weights, local_lut, candidate_lut,
        /*main_slots_x4=*/18, /*main_owner=*/true);
    REQUIRE(owner_ids != nullptr);
    const int32_t main_quota = owner_ids->op_params[1];
    REQUIRE(main_quota == 23);

    ggml_free(ctx);
}
