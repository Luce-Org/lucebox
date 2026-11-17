#include "CppUnitTestFramework.hpp"
#include "../../src/common/moe_hybrid_expert_cache.h"
#include "../../src/common/moe_hybrid_ffn_eval.h"
#include "../../src/common/moe_hybrid_storage.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#if !defined(_WIN32)
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

// Streamed expert slots are shared by layers of different quant types, each
// reading them through a view whose expert stride the kernels divide into
// whole blocks: the stride must divide exactly.
TEST_CASE(MoeHybridStorageFixture, expert_slot_stride_divides_every_shared_type) {
    // DS4.1 gate/up experts, 2304 x 5120: IQ2_XXS, IQ2_XS and IQ3_XXS layers share the gate slots.
    const size_t rows = 2304, blocks = 5120 / 256;
    std::vector<size_t> sizes;
    size_t largest = 0;
    for (ggml_type t : {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS}) {
        sizes.push_back(ggml_type_size(t));
        largest = std::max(largest, rows * blocks * ggml_type_size(t));
    }
    // The unrounded stride is not whole IQ2_XXS blocks.
    REQUIRE(largest % ggml_type_size(GGML_TYPE_IQ2_XXS) != 0);
    const size_t stride = moe_expert_slot_stride(largest, sizes);
    CHECK(stride >= largest);
    CHECK(stride - largest < (size_t) 2 * 2 * 3 * 7 * 7 * 11 * 37);  // < one lcm unit of slack
    for (size_t ts : sizes) CHECK(stride % ts == 0);

    // One type: the largest expert itself.
    const size_t q2 = 5120 * 9 * ggml_type_size(GGML_TYPE_Q2_K);
    CHECK(moe_expert_slot_stride(q2, {ggml_type_size(GGML_TYPE_Q2_K)}) == q2);

    // Down experts, 5120 x 2304, five types.
    std::vector<size_t> down;
    size_t down_largest = 0;
    for (ggml_type t : {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_Q2_K,
                        GGML_TYPE_IQ3_XXS, GGML_TYPE_Q3_K}) {
        down.push_back(ggml_type_size(t));
        down_largest = std::max(down_largest, (size_t) 5120 * 9 * ggml_type_size(t));
    }
    const size_t down_stride = moe_expert_slot_stride(down_largest, down);
    for (size_t ts : down) CHECK(down_stride % ts == 0);
}

#if !defined(_WIN32)
namespace {

// A one-layer model file of three F32 expert stacks (gate, up, down), each
// expert's bytes set to a value naming its stack and expert.
struct TinyExpertFile {
    static constexpr int kExperts = 4;
    static constexpr int kIn = 4, kOut = 8;
    static constexpr size_t kExpertBytes = sizeof(float) * kIn * kOut;
    static constexpr size_t kStackBytes = kExpertBytes * kExperts;
    std::string path;
    void * mapping = nullptr;
    size_t size = 3 * kStackBytes;

    static uint8_t value(int stack, int expert) { return (uint8_t) (16 * (stack + 1) + expert); }

    bool create() {
        char name[] = "/tmp/luce_moe_hot_XXXXXX";
        const int fd = ::mkstemp(name);
        if (fd < 0) return false;
        path = name;
        std::vector<uint8_t> bytes(size);
        for (int stack = 0; stack < 3; ++stack) {
            for (int e = 0; e < kExperts; ++e) {
                std::fill_n(bytes.begin() + (ptrdiff_t) (stack * kStackBytes + e * kExpertBytes),
                            kExpertBytes, value(stack, e));
            }
        }
        const bool written = ::write(fd, bytes.data(), bytes.size()) == (ssize_t) bytes.size();
        mapping = written ? ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0) : MAP_FAILED;
        ::close(fd);
        if (mapping == MAP_FAILED) mapping = nullptr;
        return mapping != nullptr;
    }
    ~TinyExpertFile() {
        if (!path.empty()) ::unlink(path.c_str());
    }
    ExpertTensorFileData stack(int i) const {
        return {static_cast<const uint8_t *>(mapping) + i * kStackBytes, kStackBytes};
    }
};

// Every expert of `stack` holds the bytes of the file's experts `ids`, in order.
bool stack_holds(ggml_tensor * stack, int file_stack, const std::vector<int32_t> & ids) {
    if (!stack) return false;
    std::vector<uint8_t> got(TinyExpertFile::kExpertBytes);
    for (size_t i = 0; i < ids.size(); ++i) {
        ggml_backend_tensor_get(stack, got.data(), i * TinyExpertFile::kExpertBytes, got.size());
        const uint8_t want = TinyExpertFile::value(file_stack, ids[i]);
        if (!std::all_of(got.begin(), got.end(), [want](uint8_t b) { return b == want; })) return false;
    }
    return true;
}

}  // namespace

TEST_CASE(MoeHybridStorageFixture, hot_experts_come_back_after_a_release_and_the_cold_stack_stays) {
    TinyExpertFile file;
    REQUIRE(file.create());
    auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(
        ggml_backend_cpu_init(), ggml_backend_free);
    auto meta = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
        ggml_init({1u << 16, nullptr, true}), ggml_free);
    REQUIRE(backend && meta);
    MoeLayerDesc desc;
    desc.ffn_gate_exps = ggml_new_tensor_3d(meta.get(), GGML_TYPE_F32, TinyExpertFile::kIn, TinyExpertFile::kOut, 4);
    desc.ffn_up_exps = ggml_new_tensor_3d(meta.get(), GGML_TYPE_F32, TinyExpertFile::kIn, TinyExpertFile::kOut, 4);
    desc.ffn_down_exps = ggml_new_tensor_3d(meta.get(), GGML_TYPE_F32, TinyExpertFile::kIn, TinyExpertFile::kOut, 4);
    MoeHybridConfig cfg;
    cfg.n_embd = TinyExpertFile::kIn;
    cfg.n_ff_exp = TinyExpertFile::kOut;
    cfg.n_expert = TinyExpertFile::kExperts;
    cfg.n_expert_used = 2;
    cfg.n_layer = 1;
    MoeHybridPlacement placement;
    placement.n_layer = 1;
    placement.n_expert = TinyExpertFile::kExperts;
    placement.n_expert_used = 2;
    placement.hot_counts = {2};
    placement.hot_expert_ids = {{3, 1}};
    placement.total_hot = 2;
    std::vector<LayerExpertFileData> file_data(1);
    file_data[0].gate_exps = file.stack(0);
    file_data[0].up_exps = file.stack(1);
    file_data[0].down_exps = file.stack(2);

    MoeHybridStorage storage;
    std::string err;
    REQUIRE(build_moe_hybrid_storage_from_file_with_mmap(cfg, backend.get(), placement, {desc}, file_data,
                                                         file.mapping, file.size, storage, &err,
                                                         /*cache_slots=*/1));
    file.mapping = nullptr;  // the storage owns the mapping now
    MoeHybridLayerStorage & layer = storage.layers[0];
    REQUIRE((layer.cold_expert_ids == std::vector<int32_t>{0, 2}));
    REQUIRE(stack_holds(layer.gate_hot, 0, {3, 1}));
    REQUIRE(stack_holds(layer.down_cold, 2, {0, 2}));
    // Cold expert 0 swapped into the spare slot, after the two placed ones.
    REQUIRE(layer.cache_slots == 1);
    REQUIRE(moe_hybrid_cache_swap_in(layer, 0, backend.get()) == 2);
    ggml_backend_synchronize(backend.get());
    REQUIRE(stack_holds(layer.gate_hot, 0, {3, 1, 0}));

    ggml_backend_buffer_t cold_buf = layer.cold_buf;
    storage.release_hot_experts();
    CHECK(layer.hot_buf == nullptr);
    CHECK(layer.gate_hot == nullptr);
    CHECK(!layer.is_expert_hot(3));
    CHECK(layer.hot_local_by_global[3] == 0);  // the maps stay for the reload
    CHECK(layer.hot_local_by_global[0] == -1);  // the swapped-in expert went with the stack
    CHECK(layer.spare_global[0] == -1);
    CHECK(layer.cold_buf == cold_buf);
    CHECK(stack_holds(layer.up_cold, 1, {0, 2}));

    // Back from the retained mapping: copied out of it, then read from the
    // file on the reader's threads.
    REQUIRE(rematerialize_moe_hybrid_hot_experts(backend.get(), {desc}, storage, &err));
    CHECK(stack_holds(layer.gate_hot, 0, {3, 1}));
    CHECK(stack_holds(layer.up_hot, 1, {3, 1}));
    CHECK(stack_holds(layer.down_hot, 2, {3, 1}));
    CHECK(layer.is_expert_hot(3) && layer.is_expert_hot(1) && !layer.is_expert_hot(0));
    // Swapped in again, it is copied again rather than read from a stale slot.
    REQUIRE(moe_hybrid_cache_swap_in(layer, 0, backend.get()) == 2);
    ggml_backend_synchronize(backend.get());
    CHECK(stack_holds(layer.gate_hot, 0, {3, 1, 0}));
    CHECK(!rematerialize_moe_hybrid_hot_experts(backend.get(), {desc}, storage, &err));  // already resident

    storage.release_hot_experts();
    TensorFileReader reader;
    REQUIRE(reader.open(file.path, &err, TensorFileReader::Mode::Buffered));
    REQUIRE(rematerialize_moe_hybrid_hot_experts(backend.get(), {desc}, storage, &err, &reader));
    CHECK(stack_holds(layer.gate_hot, 0, {3, 1}));
    CHECK(stack_holds(layer.down_hot, 2, {3, 1}));
    CHECK(layer.cold_buf == cold_buf);
    CHECK(stack_holds(layer.gate_cold, 0, {0, 2}));
}
#endif
