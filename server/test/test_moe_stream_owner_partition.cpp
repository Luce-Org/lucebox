#include "CppUnitTestFramework.hpp"
#include "../src/common/moe_hybrid_stream.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

struct MoeStreamOwnerPartitionFixture {};

MoeStreamRouteBatch make_batch(const std::vector<int32_t> & ids,
                               const std::vector<float> & weights) {
    MoeStreamRouteBatch batch;
    batch.layer = 0;
    batch.n_expert = 8;
    batch.top_k = 4;
    batch.n_tokens = 2;
    batch.selected_ids = ids.data();
    batch.selected_weights = weights.data();
    return batch;
}

bool is_exact_partition(const std::vector<float> & weights,
                        const std::vector<float> & primary,
                        const std::vector<float> & secondary) {
    if (primary.size() != weights.size() ||
        secondary.size() != weights.size()) return false;
    for (size_t i = 0; i < weights.size(); ++i) {
        if (primary[i] + secondary[i] != weights[i] ||
            (primary[i] != 0.0f && secondary[i] != 0.0f)) return false;
    }
    return true;
}

}  // namespace

TEST_CASE(MoeStreamOwnerPartitionFixture,
          hash_policy_is_exact_deterministic_and_uses_both_owners) {
    const std::vector<int32_t> ids = {0, 1, 2, 3, 1, 4, 5, 6};
    const std::vector<float> weights = {
        0.40f, 0.30f, 0.20f, 0.10f,
        0.35f, 0.30f, 0.20f, 0.15f};
    const MoeStreamRouteBatch batch = make_batch(ids, weights);
    MoeStreamDualOwnerPolicy policy;
    policy.primary_share_per_mille = 500;

    std::vector<float> primary;
    std::vector<float> secondary;
    MoeStreamDualOwnerStats stats;
    std::string error;
    REQUIRE(partition_moe_stream_routes(
        batch, policy, primary, secondary, &stats, &error));
    REQUIRE(is_exact_partition(weights, primary, secondary));
    REQUIRE(stats.primary_experts > 0);
    REQUIRE(stats.secondary_experts > 0);

    // Expert 1 appears in both tokens and must retain one cache owner.
    REQUIRE((primary[1] != 0.0f) == (primary[4] != 0.0f));

    std::vector<float> primary_again;
    std::vector<float> secondary_again;
    REQUIRE(partition_moe_stream_routes(
        batch, policy, primary_again, secondary_again, nullptr, &error));
    REQUIRE(primary_again == primary);
    REQUIRE(secondary_again == secondary);
}

TEST_CASE(MoeStreamOwnerPartitionFixture,
          explicit_placement_is_authoritative) {
    const std::vector<int32_t> ids = {0, 1, 2, 3, 1, 4, 5, 6};
    const std::vector<float> weights = {
        0.40f, 0.30f, 0.20f, 0.10f,
        0.35f, 0.30f, 0.20f, 0.15f};
    const MoeStreamRouteBatch batch = make_batch(ids, weights);

    MoeHybridPlacement placement;
    placement.n_layer = 1;
    placement.n_expert = 8;
    placement.n_expert_used = 4;
    placement.total_hot = 2;
    placement.hot_counts = {2};
    placement.hot_expert_ids = {{1, 5}};

    MoeStreamDualOwnerPolicy policy;
    policy.primary_placement = &placement;
    policy.primary_share_per_mille = 500;

    std::vector<float> primary;
    std::vector<float> secondary;
    MoeStreamDualOwnerStats stats;
    std::string error;
    REQUIRE(partition_moe_stream_routes(
        batch, policy, primary, secondary, &stats, &error));
    REQUIRE(is_exact_partition(weights, primary, secondary));
    for (size_t i = 0; i < ids.size(); ++i) {
        const bool expected_primary = ids[i] == 1 || ids[i] == 5;
        REQUIRE((primary[i] != 0.0f) == expected_primary);
    }
    REQUIRE(stats.primary_experts == 2);
    REQUIRE(stats.secondary_experts == 5);
}

TEST_CASE(MoeStreamOwnerPartitionFixture,
          endpoint_shares_and_shape_mismatch_fail_closed) {
    const std::vector<int32_t> ids = {0, 1, 2, 3, 1, 4, 5, 6};
    const std::vector<float> weights(ids.size(), 0.25f);
    const MoeStreamRouteBatch batch = make_batch(ids, weights);
    std::vector<float> primary;
    std::vector<float> secondary;
    MoeStreamDualOwnerStats stats;
    std::string error;

    MoeStreamDualOwnerPolicy policy;
    policy.primary_share_per_mille = 0;
    REQUIRE(partition_moe_stream_routes(
        batch, policy, primary, secondary, &stats, &error));
    REQUIRE(stats.primary_routes == 0);
    REQUIRE(stats.secondary_routes == 8);
    REQUIRE(is_exact_partition(weights, primary, secondary));

    policy.primary_share_per_mille = 1000;
    REQUIRE(partition_moe_stream_routes(
        batch, policy, primary, secondary, &stats, &error));
    REQUIRE(stats.primary_routes == 8);
    REQUIRE(stats.secondary_routes == 0);
    REQUIRE(is_exact_partition(weights, primary, secondary));

    MoeHybridPlacement mismatch;
    mismatch.n_layer = 1;
    mismatch.n_expert = 7;
    policy.primary_placement = &mismatch;
    REQUIRE(!partition_moe_stream_routes(
        batch, policy, primary, secondary, nullptr, &error));
}
