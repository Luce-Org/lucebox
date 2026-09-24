#include "CppUnitTestFramework.hpp"
#include "../src/common/moe_hybrid_ffn_eval.h"
#include "../src/common/moe_hybrid_storage.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace luce::common;

namespace {
struct MoeHybridStorageFixture {};
struct ScopedFfnGraph : CachedFfnGraph {
    ~ScopedFfnGraph() { free(); }
};
}

TEST_CASE(MoeHybridStorageFixture, single_token_cached_graphs_keep_mixed_mmq_policy) {
    auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(
        ggml_backend_cpu_init(), ggml_backend_free);
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
        ggml_init({1u << 20, nullptr, true}), ggml_free);
    REQUIRE(backend && ctx);
    auto * gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 4, 8, 2);
    auto * up = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 4, 8, 2);
    auto * down = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 8, 4, 2);
    MoeLayerDesc desc;
    desc.ffn_gate_shexp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 8);
    desc.ffn_up_shexp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 8);
    desc.ffn_down_shexp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 8, 4);
    auto weights = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    REQUIRE(weights != nullptr);
    for (auto policy : {GGML_MIXED_MMQ_DEFAULT, GGML_MIXED_MMQ_ENABLED, GGML_MIXED_MMQ_DISABLED}) {
        ScopedFfnGraph hot, cold;
        REQUIRE(build_cached_hot_graph(hot, backend.get(), gate, up, down, nullptr,
            1, 1, 1, 1, desc, 4, 8, 1, CachedHotGraphOptions{0, false, 0, policy}));
        REQUIRE(build_cached_cold_graph(cold, backend.get(), gate, up, down, nullptr,
            1, 1, 1, 1, 4, 8, 1, 0, policy));
        for (auto * graph : {hot.gf, cold.gf}) {
            int routed = 0;
            int shared = 0;
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto * op = ggml_graph_node(graph, i);
                if (op->op == GGML_OP_MUL_MAT_ID) {
                    CHECK(ggml_mul_mat_get_mixed_mmq(op) == policy);
                    ++routed;
                } else if (op->op == GGML_OP_MUL_MAT) {
                    CHECK(ggml_mul_mat_get_mixed_mmq(op) == GGML_MIXED_MMQ_DEFAULT);
                    ++shared;
                }
            }
            CHECK(routed == 3);
            CHECK(shared == (graph == hot.gf ? 3 : 0));
        }
    }
}

TEST_CASE(MoeHybridStorageFixture, storage_identity_includes_mixed_mmq_policy) {
    MoeHybridConfig cfg;
    MoeHybridStorage storage;
    cfg.n_layer = storage.placement.n_layer = 1;
    cfg.n_expert = storage.placement.n_expert = 2;
    cfg.n_expert_used = storage.placement.n_expert_used = 1;
    storage.placement.hot_counts = {0};
    storage.placement.hot_expert_ids = {{}};
    storage.layers.resize(1);
    REQUIRE(storage.matches(cfg));
    cfg.mixed_mmq_policy = GGML_MIXED_MMQ_ENABLED;
    REQUIRE(!storage.matches(cfg));
    storage.mixed_mmq_policy = cfg.mixed_mmq_policy;
    REQUIRE(storage.matches(cfg));
    cfg.mixed_mmq_policy = GGML_MIXED_MMQ_DISABLED;
    REQUIRE(!storage.matches(cfg));
}

TEST_CASE(MoeHybridStorageFixture, cold_owner_none_implies_no_cold_materialization) {
    MoeHybridConfig cfg;
    cfg.cold_expert_backend = MoeHybridColdBackend::None;
    // The flag keeps its default; None must not need it cleared by hand.
    REQUIRE(cfg.materialize_cold_experts);
    REQUIRE(!cfg.materializes_cold_experts());

    MoeHybridStorage storage;
    cfg.n_layer = storage.placement.n_layer = 1;
    cfg.n_expert = storage.placement.n_expert = 2;
    cfg.n_expert_used = storage.placement.n_expert_used = 1;
    storage.placement.hot_counts = {0};
    storage.placement.hot_expert_ids = {{}};
    storage.layers.resize(1);
    storage.cold_backend_kind = MoeHybridColdBackend::None;
    // Storage built for None records no cold materialization, and a config
    // that kept the default flag still identifies it.
    storage.materialized_cold_experts = true;
    REQUIRE(!storage.matches(cfg));
    storage.materialized_cold_experts = false;
    REQUIRE(storage.matches(cfg));

    cfg.cold_expert_backend = MoeHybridColdBackend::Gpu;
    REQUIRE(cfg.materializes_cold_experts());
}

TEST_CASE(MoeHybridStorageFixture, cold_owner_none_refuses_a_shared_expert_in_the_routed_partial) {
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
        ggml_init({1u << 16, nullptr, true}), ggml_free);
    REQUIRE(ctx != nullptr);
    MoeHybridConfig cfg;
    cfg.n_embd = 4;
    cfg.n_expert = 2;
    cfg.n_expert_used = 1;
    cfg.cold_expert_backend = MoeHybridColdBackend::None;
    MoeHybridLayerStorage storage;
    storage.cold_backend_kind = MoeHybridColdBackend::None;
    MoeLayerDesc desc;
    desc.ffn_gate_shexp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 8);
    desc.ffn_up_shexp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 8);
    desc.ffn_down_shexp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 8, 4);

    // Every owner would add the replicated shared expert into the partial the
    // caller sums across owners. The evaluator refuses before touching a
    // backend instead of double counting.
    const float cur[8] = {};
    const int32_t ids[2] = {0, 1};
    const float weights[2] = {1.0f, 1.0f};
    std::vector<float> out;
    std::string err;
    CHECK(!eval_moe_hybrid_ffn_batched(nullptr, nullptr, cfg, desc, storage,
                                       cur, ids, weights, 2, out, &err));
    CHECK(err.find("shared expert") != std::string::npos);
    err.clear();
    CHECK(!eval_moe_hybrid_ffn_single(nullptr, cfg, desc, storage, nullptr,
                                      cur, ids, weights, 1, out, nullptr, &err));
    CHECK(err.find("shared expert") != std::string::npos);

    // A non-positive batch is empty, not a huge allocation.
    out.assign(3, 1.0f);
    CHECK(eval_moe_shared_expert_batched(nullptr, cfg, desc, storage, cur, -1, out));
    CHECK(out.empty());
    CHECK(!eval_moe_shared_expert_batched(nullptr, cfg, desc, storage, cur, 2, out, &err));
    CHECK(err.find("GPU backend") != std::string::npos);
}

TEST_CASE(MoeHybridStorageFixture, cold_owner_none_does_not_stream_cold_experts) {
    MoeHybridStorage storage;
    storage.materialized_cold_experts = false;
    storage.cold_backend_kind = MoeHybridColdBackend::Gpu;
    CHECK(storage.streams_cold_experts());
    storage.cold_backend_kind = MoeHybridColdBackend::Cpu;
    CHECK(storage.streams_cold_experts());
    storage.cold_backend_kind = MoeHybridColdBackend::None;
    CHECK(!storage.streams_cold_experts());
    storage.materialized_cold_experts = true;
    storage.cold_backend_kind = MoeHybridColdBackend::Gpu;
    CHECK(!storage.streams_cold_experts());
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
