#include "CppUnitTestFramework.hpp"
#include "qwen3/qwen3_buffer_plan.h"

using dflash::common::qwen3_drafter_buffer_plan;

namespace {
struct Qwen3BufferPlanFixture {};
}

TEST_CASE(Qwen3BufferPlanFixture, nope_tail_reuses_current_layer_kv) {
    for (int layers : {1, 28}) {
        const auto plan = qwen3_drafter_buffer_plan(true, layers);
        CHECK(plan.rope_k_buffers == 1);
        CHECK(plan.value_buffers == 1);
        CHECK(plan.rope_q_tail_buffers == 0);
        for (int layer = 0; layer < layers; ++layer) {
            CHECK(plan.layer_cache_index(layer) == 0);
        }
    }
}

TEST_CASE(Qwen3BufferPlanFixture, legacy_rope_scoring_retains_per_layer_state) {
    for (int layers : {1, 28}) {
        const auto plan = qwen3_drafter_buffer_plan(false, layers);
        CHECK(plan.rope_k_buffers == (size_t)layers);
        CHECK(plan.value_buffers == 1);
        CHECK(plan.rope_q_tail_buffers == (size_t)layers);
        for (int layer = 0; layer < layers; ++layer) {
            CHECK(plan.layer_cache_index(layer) == (size_t)layer);
        }
    }
}

TEST_CASE(Qwen3BufferPlanFixture, empty_model_has_no_layer_buffers) {
    for (bool nope_tail : {false, true}) {
        const auto plan = qwen3_drafter_buffer_plan(nope_tail, 0);
        CHECK(plan.rope_k_buffers == 0);
        CHECK(plan.value_buffers == 0);
        CHECK(plan.rope_q_tail_buffers == 0);
    }
}
