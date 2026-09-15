#include "CppUnitTestFramework.hpp"

#include "qwen3/qwen3_buffer_plan.h"

#include <cstddef>
#include <cstdint>

using dflash::common::qwen3_drafter_buffer_plan;

namespace {
struct Qwen3BufferPlanFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;

    void nope_tail_reuses_current_layer_kv() {
        const auto plan = qwen3_drafter_buffer_plan(true, 28);
        REQUIRE(plan.rope_k_buffers == (size_t)1);
        REQUIRE(plan.value_buffers == (size_t)1);
        REQUIRE(plan.rope_q_tail_buffers == (size_t)0);
        REQUIRE(plan.layer_cache_index(0) == (size_t)0);
        REQUIRE(plan.layer_cache_index(1) == (size_t)0);
        REQUIRE(plan.layer_cache_index(27) == (size_t)0);
    }

    void legacy_rope_scoring_retains_per_layer_state() {
        const auto plan = qwen3_drafter_buffer_plan(false, 28);
        REQUIRE(plan.rope_k_buffers == (size_t)28);
        REQUIRE(plan.value_buffers == (size_t)1);
        REQUIRE(plan.rope_q_tail_buffers == (size_t)28);
        REQUIRE(plan.layer_cache_index(0) == (size_t)0);
        REQUIRE(plan.layer_cache_index(1) == (size_t)1);
        REQUIRE(plan.layer_cache_index(27) == (size_t)27);
    }

    void empty_model_has_no_layer_buffers() {
        const auto plan = qwen3_drafter_buffer_plan(true, 0);
        REQUIRE(plan.rope_k_buffers == (size_t)0);
        REQUIRE(plan.value_buffers == (size_t)0);
        REQUIRE(plan.rope_q_tail_buffers == (size_t)0);
    }

    void single_layer_mapping_is_in_bounds() {
        const auto nope_plan = qwen3_drafter_buffer_plan(true, 1);
        const auto legacy_plan = qwen3_drafter_buffer_plan(false, 1);
        REQUIRE(nope_plan.layer_cache_index(0) == (size_t)0);
        REQUIRE(legacy_plan.layer_cache_index(0) == (size_t)0);
    }

    void nope_tail_removes_reported_per_layer_allocation_growth() {
        constexpr size_t heads_kv = 8;
        constexpr size_t head_dim = 128;
        constexpr size_t bf16_bytes = 2;
        const auto bytes_per_kv = [](size_t seq_len) {
            return seq_len * heads_kv * head_dim * bf16_bytes;
        };

        REQUIRE(bytes_per_kv(179262) == (size_t)367128576);
        REQUIRE(bytes_per_kv(199530) == (size_t)408637440);

        const auto plan = qwen3_drafter_buffer_plan(true, 28);
        REQUIRE(plan.rope_k_buffers + plan.value_buffers == (size_t)2);
    }
};
}

TEST_CASE(Qwen3BufferPlanFixture, allocation_policy) {
    nope_tail_reuses_current_layer_kv();
    legacy_rope_scoring_retains_per_layer_state();
    empty_model_has_no_layer_buffers();
    single_layer_mapping_is_in_bounds();
    nope_tail_removes_reported_per_layer_allocation_growth();
}
