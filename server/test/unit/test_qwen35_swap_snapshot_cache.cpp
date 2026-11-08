// SWAP_SNAPSHOT: the live SSM tensors trade
// storage with their snapshot partners (gdn_single_commit.h exchange_pairs).
// Tensor identity is kept, so identity-based copies follow the exchange, and
// every cache reset puts the pairs back before it clears base_buf: a request
// never starts from the previous request's state. Synthetic one-GDN-layer
// cache on the CPU backend.

#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"
#include "qwen35_tiny_gdn_target.h"
#include "gdn_single_commit.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <vector>

using luce::common::TargetCache;
using luce_test::TinyGdnTarget;
namespace gp = luce::common::gdn_commit;

namespace {

struct Qwen35SwapSnapshotCache {};

std::vector<float> filled(const ggml_tensor * t, float v) {
    return std::vector<float>((size_t)ggml_nelements(t), v);
}

void set_all(ggml_tensor * t, float v) {
    const std::vector<float> values = filled(t, v);
    ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
}

std::vector<float> get_all(const ggml_tensor * t) {
    std::vector<float> values((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, values.data(), 0, ggml_nbytes(t));
    return values;
}

}  // namespace

TEST_CASE(Qwen35SwapSnapshotCache, exchange_keeps_identity_and_reset_restores_layout) {
    const luce_test::ScopedEnvVar kv_f16("LUCE_KV_F16", "1");
    TinyGdnTarget t(ggml_backend_cpu_init(), /*state_dim=*/16, /*raw_gates=*/false);
    REQUIRE(t.ok);
    TargetCache & c = t.cache;
    ggml_tensor * live = c.ssm_state[0];
    ggml_tensor * spare = c.ssm_state_snap[0];
    REQUIRE(live != nullptr);
    REQUIRE(spare != nullptr);
    REQUIRE(live->buffer == c.base_buf);
    REQUIRE(spare->buffer == c.rollback_buf);
    void * const live_data = live->data;
    void * const spare_data = spare->data;

    set_all(live, 1.0f);
    set_all(spare, 2.0f);
    gp::exchange_pairs(c.ssm_state, c.ssm_state_snap, c.ssm_state_swapped);
    CHECK(c.ssm_state_swapped);
    CHECK(c.ssm_state[0] == live);
    CHECK(live->buffer == c.rollback_buf);
    CHECK(live->data == spare_data);
    CHECK(spare->data == live_data);
    // The contents follow the storage.
    CHECK(get_all(live) == filled(live, 2.0f));
    CHECK(get_all(spare) == filled(spare, 1.0f));
    // Identity-based writers and readers (prefix cache, kvflash) follow too.
    set_all(c.ssm_state[0], 3.0f);
    CHECK(get_all(live) == filled(live, 3.0f));

    // Request start: the base_buf clear must zero the live state.
    luce::common::reset_recurrent_state(c);
    CHECK(!c.ssm_state_swapped);
    CHECK(live->buffer == c.base_buf);
    CHECK(live->data == live_data);
    CHECK(spare->data == spare_data);
    CHECK(get_all(live) == filled(live, 0.0f));

    // reset_target_cache zeroes both and leaves the allocation layout.
    set_all(spare, 4.0f);
    gp::exchange_pairs(c.ssm_state, c.ssm_state_snap, c.ssm_state_swapped);
    luce::common::reset_target_cache(c);
    CHECK(!c.ssm_state_swapped);
    CHECK(live->data == live_data);
    CHECK(get_all(live) == filled(live, 0.0f));
    CHECK(get_all(spare) == filled(spare, 0.0f));

    // free_target_cache hands each buffer its own tensors back first.
    gp::exchange_pairs(c.ssm_state, c.ssm_state_snap, c.ssm_state_swapped);
    luce::common::free_target_cache(c);
    CHECK(!c.ssm_state_swapped);
    CHECK(c.ssm_state.empty());
}

TEST_CASE(Qwen35SwapSnapshotCache, unswapped_reset_is_unchanged) {
    const luce_test::ScopedEnvVar kv_f16("LUCE_KV_F16", "1");
    TinyGdnTarget t(ggml_backend_cpu_init(), /*state_dim=*/16, /*raw_gates=*/false);
    REQUIRE(t.ok);
    TargetCache & c = t.cache;
    ggml_tensor * live = c.ssm_state[0];
    void * const live_data = live->data;
    set_all(live, 5.0f);
    set_all(c.ssm_state_snap[0], 6.0f);
    luce::common::reset_recurrent_state(c);
    CHECK(!c.ssm_state_swapped);
    CHECK(live->data == live_data);
    CHECK(get_all(live) == filled(live, 0.0f));
    // The parent's reset clears base_buf only: the snapshot keeps its bytes.
    CHECK(get_all(c.ssm_state_snap[0]) == filled(live, 6.0f));
}
