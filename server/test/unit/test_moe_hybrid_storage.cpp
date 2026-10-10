#include "CppUnitTestFramework.hpp"
#include "../../src/common/moe_hybrid_ffn_eval.h"
#include "../../src/common/moe_hybrid_storage.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

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

#if !defined(_WIN32)
namespace {

// A two-layer expert file, mapped read-only from offset zero as the loaders
// map a GGUF: per layer the gate, up and down stacks of six F32 experts.
// Every float encodes (layer, tensor, expert, element), so a stack's bytes
// say which experts it holds and in which order.
struct ExpertFile {
    static constexpr int n_layer = 2, n_expert = 6, n_embd = 4, n_ff = 8;
    static constexpr size_t expert_floats = (size_t) n_embd * n_ff;
    static constexpr size_t expert_bytes = expert_floats * sizeof(float);
    static constexpr size_t tensor_bytes = expert_bytes * n_expert;

    bool ok = false;
    std::string path;
    ggml_context * meta = nullptr;
    std::vector<MoeLayerDesc> descs;

    static float value(int layer, int tensor, int expert, size_t i) {
        return (float) (layer * 1000 + tensor * 100 + expert) + (float) i / 64.0f;
    }
    static size_t offset(int layer, int tensor) { return ((size_t) layer * 3 + (size_t) tensor) * tensor_bytes; }

    ExpertFile() {
        char name[] = "/tmp/moe-hybrid-primary-XXXXXX";
        const int fd = mkstemp(name);
        if (fd < 0) return;
        path = name;
        std::vector<float> data((size_t) n_layer * 3 * n_expert * expert_floats);
        for (int il = 0; il < n_layer; ++il)
            for (int t = 0; t < 3; ++t)
                for (int e = 0; e < n_expert; ++e)
                    for (size_t i = 0; i < expert_floats; ++i)
                        data[(offset(il, t) + (size_t) e * expert_bytes) / sizeof(float) + i] = value(il, t, e, i);
        const bool written =
            ::write(fd, data.data(), data.size() * sizeof(float)) == (ssize_t) (data.size() * sizeof(float));
        ::close(fd);
        meta = ggml_init({64 * ggml_tensor_overhead(), nullptr, true});
        if (!written || !meta) return;
        for (int il = 0; il < n_layer; ++il) {
            MoeLayerDesc d;
            d.ffn_gate_exps = ggml_new_tensor_3d(meta, GGML_TYPE_F32, n_embd, n_ff, n_expert);
            d.ffn_up_exps = ggml_new_tensor_3d(meta, GGML_TYPE_F32, n_embd, n_ff, n_expert);
            d.ffn_down_exps = ggml_new_tensor_3d(meta, GGML_TYPE_F32, n_ff, n_embd, n_expert);
            descs.push_back(d);
        }
        ok = true;
    }
    ~ExpertFile() {
        if (meta) ggml_free(meta);
        if (!path.empty()) std::remove(path.c_str());
    }

    // Builds storage over a fresh mapping (the storage unmaps it).
    bool build(const std::vector<std::vector<int32_t>> & hot,
               const std::vector<std::vector<int32_t>> & secondary,
               ggml_backend_t primary, ggml_backend_t peer, int cache_slots,
               MoeHybridStorage & out, std::string * err) const {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        const size_t size = offset(n_layer, 0);
        void * map = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if (map == MAP_FAILED) return false;
        MoeHybridConfig cfg;
        cfg.n_layer = n_layer;
        cfg.n_expert = n_expert;
        cfg.n_expert_used = 2;
        cfg.n_embd = n_embd;
        cfg.n_ff_exp = n_ff;
        cfg.cold_expert_backend = MoeHybridColdBackend::Gpu;
        cfg.cold_expert_ids = secondary;
        MoeHybridPlacement placement;
        placement.n_layer = n_layer;
        placement.n_expert = n_expert;
        placement.n_expert_used = 2;
        placement.hot_expert_ids = hot;
        for (const auto & ids : hot) {
            placement.hot_counts.push_back((int) ids.size());
            placement.total_hot += (int) ids.size();
        }
        std::vector<LayerExpertFileData> files((size_t) n_layer);
        const auto * base = static_cast<const uint8_t *>(map);
        for (int il = 0; il < n_layer; ++il) {
            files[(size_t) il].gate_exps = {base + offset(il, 0), tensor_bytes};
            files[(size_t) il].up_exps = {base + offset(il, 1), tensor_bytes};
            files[(size_t) il].down_exps = {base + offset(il, 2), tensor_bytes};
        }
        const bool ok = build_moe_hybrid_storage_from_file_with_mmap(
            cfg, primary, placement, descs, files, map, size, out, err, cache_slots, peer);
        if (!ok) ::munmap(map, size);
        return ok;
    }

    // What a stack holding `experts` of (layer, tensor), in that order, must contain.
    static std::vector<float> expected(int layer, int tensor, const std::vector<int32_t> & experts) {
        std::vector<float> out;
        for (int32_t e : experts)
            for (size_t i = 0; i < expert_floats; ++i) out.push_back(value(layer, tensor, e, i));
        return out;
    }
};

std::vector<float> tensor_floats(const ggml_tensor * t) {
    std::vector<float> out(ggml_nbytes(t) / sizeof(float));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    return out;
}

}  // namespace

// A partial eviction frees the primary expert stacks and later rebuilds them
// around the retained secondary stack: same rows in the same order, every
// owner map unchanged, the secondary stack neither reallocated nor reloaded.
TEST_CASE(MoeHybridStorageFixture, primary_experts_rebuild_in_place_around_the_secondary_stack) {
    const ExpertFile file;
    REQUIRE(file.ok);
    auto primary = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(
        ggml_backend_cpu_init(), ggml_backend_free);
    auto peer = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(
        ggml_backend_cpu_init(), ggml_backend_free);
    REQUIRE(primary && peer);
    // Unsorted hot ids: the stack row order is the list order.
    const std::vector<std::vector<int32_t>> hot = {{4, 1}, {0, 5, 2}};
    const std::vector<std::vector<int32_t>> secondary = {{0, 2}, {3}};
    MoeHybridStorage storage;
    std::string err;
    REQUIRE(file.build(hot, secondary, primary.get(), peer.get(), 0, storage, &err));
    const MoeHybridOwnershipRecord placed = MoeHybridOwnershipRecord::capture(storage);
    const uint64_t primary_bytes = storage.primary_upload_bytes;
    const uint64_t secondary_bytes = storage.secondary_upload_bytes;
    REQUIRE(primary_bytes == 5 * 3 * ExpertFile::expert_bytes);
    REQUIRE(secondary_bytes == 3 * 3 * ExpertFile::expert_bytes);
    std::vector<ggml_backend_buffer_t> cold_bufs;
    for (const auto & layer : storage.layers) cold_bufs.push_back(layer.cold_buf);

    REQUIRE(storage.release_primary_experts(&err));
    REQUIRE(storage.release_primary_experts(&err));  // idempotent
    for (size_t il = 0; il < storage.layers.size(); ++il) {
        const auto & layer = storage.layers[il];
        CHECK(!layer.hot_buf && !layer.gate_hot && !layer.up_hot && !layer.down_hot);
        CHECK(!layer.is_expert_hot(hot[il][0]));  // not resident while released
        CHECK(layer.cold_buf == cold_bufs[il]);
    }
    REQUIRE(placed.matches(storage, &err));

    REQUIRE(storage.rebuild_primary_experts(primary.get(), file.path, &err));
    for (int il = 0; il < ExpertFile::n_layer; ++il) {
        const auto & layer = storage.layers[(size_t) il];
        CHECK(tensor_floats(layer.gate_hot) == ExpertFile::expected(il, 0, hot[(size_t) il]));
        CHECK(tensor_floats(layer.up_hot) == ExpertFile::expected(il, 1, hot[(size_t) il]));
        CHECK(tensor_floats(layer.down_hot) == ExpertFile::expected(il, 2, hot[(size_t) il]));
        CHECK(tensor_floats(layer.gate_cold) == ExpertFile::expected(il, 0, secondary[(size_t) il]));
        CHECK(layer.cold_buf == cold_bufs[(size_t) il]);
        for (int e = 0; e < ExpertFile::n_expert; ++e) {
            const bool is_hot = std::count(hot[(size_t) il].begin(), hot[(size_t) il].end(), e) > 0;
            CHECK(layer.is_expert_hot(e) == is_hot);
        }
    }
    CHECK(placed.matches(storage, &err));
    CHECK(storage.primary_upload_bytes == 2 * primary_bytes);
    CHECK(storage.secondary_upload_bytes == secondary_bytes);

    // A map that no longer matches its ordered ids (a recomputed placement)
    // is refused; the storage stays released rather than half rebuilt.
    REQUIRE(storage.release_primary_experts(&err));
    std::swap(storage.layers[0].hot_local_by_global[4], storage.layers[0].hot_local_by_global[1]);
    err.clear();
    CHECK(!storage.rebuild_primary_experts(primary.get(), file.path, &err));
    CHECK(err.find("layer 0") != std::string::npos);
    CHECK(storage.primary_experts_released());
    CHECK(!storage.layers[0].hot_buf && !storage.layers[1].hot_buf);
    CHECK(!placed.matches(storage));
}

// A storage whose primary population is mutable (spare cache slots that
// swap experts in) cannot be rebuilt as it was, so it is not released, and a
// different placement never passes for the recorded one.
TEST_CASE(MoeHybridStorageFixture, primary_release_refuses_spare_cache_slots) {
    const ExpertFile file;
    REQUIRE(file.ok);
    auto primary = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(
        ggml_backend_cpu_init(), ggml_backend_free);
    auto peer = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(
        ggml_backend_cpu_init(), ggml_backend_free);
    REQUIRE(primary && peer);
    MoeHybridStorage fixed, cached;
    std::string err;
    REQUIRE(file.build({{4, 1}, {0, 5, 2}}, {}, primary.get(), peer.get(), 0, fixed, &err));
    REQUIRE(file.build({{1, 4}, {0, 5, 2}}, {}, primary.get(), peer.get(), 1, cached, &err));
    CHECK(!MoeHybridOwnershipRecord::capture(fixed).matches(cached));

    err.clear();
    CHECK(!cached.release_primary_experts(&err));
    CHECK(err.find("spare") != std::string::npos);
    CHECK(!cached.primary_experts_released());
    CHECK(cached.layers[0].hot_buf != nullptr);
}
#endif
