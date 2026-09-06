#include "CppUnitTestFramework.hpp"
#include "internal.h"
#include "qwen35/graph_builders.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstdio>
#include <vector>

using namespace CppUnitTestFramework;

using dflash::common::TargetCache;
using dflash::common::TargetPagedTreeGraphKey;
using dflash::common::StepGraph;
using dflash::common::restore_ssm_state;
using dflash::common::snapshot_ssm_state;

namespace {
struct RecurrentSnapshotFixture : CommonFixture {
    using CommonFixture::CommonFixture;
};
}

static void set_tensor(ggml_tensor * tensor, const std::vector<float> & values) {
    ggml_backend_tensor_set(tensor, values.data(), 0,
                            values.size() * sizeof(float));
}

static std::vector<float> get_tensor(const ggml_tensor * tensor) {
    std::vector<float> values((size_t)ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0,
                            values.size() * sizeof(float));
    return values;
}

TEST_CASE(RecurrentSnapshotFixture, validates_paged_tree_capacity_and_uploads) {
    size_t graph_capacity = 0;
    CHECK(dflash::common::detail::
              target_graph_capacity_for_parallel_segments(
                  0, graph_capacity) && graph_capacity == 16384);
    CHECK(dflash::common::detail::
              target_graph_capacity_for_parallel_segments(
                  8, graph_capacity) && graph_capacity == 16384);
    CHECK(dflash::common::detail::
              target_graph_capacity_for_parallel_segments(
                  16, graph_capacity) && graph_capacity == 32768);
    CHECK(dflash::common::detail::
              target_graph_capacity_for_parallel_segments(
                  64, graph_capacity) && graph_capacity == 131072);
    CHECK(!dflash::common::detail::
               target_graph_capacity_for_parallel_segments(
                   65, graph_capacity));
    CHECK(dflash::common::detail::target_paged_tree_graph_capacity(
              16, 16, graph_capacity) && graph_capacity == 32768);
    CHECK(dflash::common::detail::target_paged_tree_graph_capacity(
              8, 1, graph_capacity, 10) && graph_capacity == 32768);
    CHECK(!dflash::common::detail::target_paged_tree_graph_capacity(
               17, 16, graph_capacity));

    {
        ggml_backend_t tree_backend = ggml_backend_cpu_init();
        CHECK(tree_backend != nullptr);
        ggml_init_params marker_params{};
        marker_params.mem_size = 4 * ggml_tensor_overhead();
        marker_params.no_alloc = true;
        ggml_context * marker_ctx = ggml_init(marker_params);
        ggml_init_params live_params{};
        live_params.mem_size = 16 * ggml_tensor_overhead();
        live_params.no_alloc = true;
        ggml_context * live_ctx = ggml_init(live_params);
        CHECK(marker_ctx != nullptr);
        CHECK(live_ctx != nullptr);
        if (tree_backend && marker_ctx && live_ctx) {
            StepGraph tree;
            tree.active_slot_ids =
                ggml_new_tensor_1d(marker_ctx, GGML_TYPE_I32, 2);
            ggml_tensor * unallocated_state_ids =
                ggml_new_tensor_1d(marker_ctx, GGML_TYPE_I32, 2);
            tree.inp_embed = ggml_new_tensor_2d(
                live_ctx, GGML_TYPE_F32, 4, 4);
            tree.positions =
                ggml_new_tensor_1d(live_ctx, GGML_TYPE_I32, 16);
            tree.parent_ids =
                ggml_new_tensor_2d(live_ctx, GGML_TYPE_I32, 2, 2);
            tree.tree_sizes =
                ggml_new_tensor_1d(live_ctx, GGML_TYPE_I32, 2);
            tree.state_slot_ids =
                ggml_new_tensor_1d(live_ctx, GGML_TYPE_I32, 2);
            tree.paged_query_seq_ids =
                ggml_new_tensor_1d(live_ctx, GGML_TYPE_I32, 4);
            tree.kv_write_rows =
                ggml_new_tensor_2d(live_ctx, GGML_TYPE_I64, 4, 1);
            ggml_backend_buffer_t live_buffer =
                ggml_backend_alloc_ctx_tensors(live_ctx, tree_backend);
            CHECK(live_buffer != nullptr);
            if (live_buffer) {
                CHECK(tree.active_slot_ids->buffer == nullptr);
                CHECK(dflash::common::detail::
                          target_paged_tree_uploads_ready(tree));
                CHECK(!dflash::common::detail::
                           target_paged_tree_active_slots_need_upload(tree));

                const int32_t state_ids[] = {0, 1};
                ggml_backend_tensor_set(tree.state_slot_ids, state_ids, 0,
                                        sizeof(state_ids));
                tree.state_slot_ids = unallocated_state_ids;
                CHECK(!dflash::common::detail::
                           target_paged_tree_uploads_ready(tree));
                ggml_backend_buffer_free(live_buffer);
            }
        }
        if (live_ctx) ggml_free(live_ctx);
        if (marker_ctx) ggml_free(marker_ctx);
        if (tree_backend) ggml_backend_free(tree_backend);
    }

}

TEST_CASE(RecurrentSnapshotFixture, validates_paged_tree_layout) {
    {
        ggml_init_params shape_params{};
        shape_params.mem_size = 8 * ggml_tensor_overhead();
        shape_params.no_alloc = true;
        ggml_context * shape_ctx = ggml_init(shape_params);
        CHECK(shape_ctx != nullptr);
        if (shape_ctx) {
            TargetCache shape_cache;
            shape_cache.n_seq_slots = 2;
            shape_cache.paged_block_table =
                ggml_new_tensor_2d(shape_ctx, GGML_TYPE_I32, 4, 2);
            shape_cache.paged_kv_seq_lens =
                ggml_new_tensor_1d(shape_ctx, GGML_TYPE_I32, 2);
            shape_cache.attn_k = {
                ggml_new_tensor_4d(shape_ctx, GGML_TYPE_F16, 4, 64, 1, 1),
            };
            CHECK(dflash::common::detail::validate_target_paged_tree_layout(
                shape_cache, 8, 2, 4096, 32, 16));
            CHECK(!dflash::common::detail::validate_target_paged_tree_layout(
                shape_cache, 8, 2, 4096, 48, 16));
            CHECK(!dflash::common::detail::validate_target_paged_tree_layout(
                shape_cache, 8, 5, 4096, 32, 16));

            const dflash::common::QwenPrefillSegment segments[] = {
                {0, 3, 1},
                {3, 2, 0},
            };
            int direct_rows = -1;
            CHECK(dflash::common::detail::validate_target_paged_tree_prefix(
                shape_cache, 5, segments, 2, 1, direct_rows));
            CHECK(direct_rows == 6);
            const dflash::common::QwenPrefillSegment gap[] = {
                {0, 3, 1},
                {4, 2, 0},
            };
            CHECK(!dflash::common::detail::validate_target_paged_tree_prefix(
                shape_cache, 5, gap, 2, 1, direct_rows));
            CHECK(!dflash::common::detail::validate_target_paged_tree_prefix(
                shape_cache, 5, segments, 2, 3, direct_rows));
            ggml_free(shape_ctx);
        }
    }

}

TEST_CASE(RecurrentSnapshotFixture, invalidates_paged_tree_graph_cache_key) {
    StepGraph graph;
    graph.paged_tree_meta_arena.reset(
        new uint8_t[1], std::default_delete<uint8_t[]>());
    graph.paged_tree_key = TargetPagedTreeGraphKey{
        nullptr, nullptr, nullptr, 8, 4, 256, 4096, 16, 0, 18,
        {1, 3, 0, 2}};
    const TargetPagedTreeGraphKey other_slot{
        nullptr, nullptr, nullptr, 8, 4, 256, 4096, 16, 0, 18,
        {0, 3, 1, 2}};
    const TargetPagedTreeGraphKey other_logits{
        nullptr, nullptr, nullptr, 8, 4, 256, 4096, 16, 0, 19,
        {1, 3, 0, 2}};
    CHECK(*graph.paged_tree_key != other_slot);
    CHECK(*graph.paged_tree_key != other_logits);
    step_graph_free(graph);
    CHECK(!graph.paged_tree_key);
    CHECK(graph.paged_tree_meta_arena);
    step_graph_destroy(graph);
    CHECK(!graph.paged_tree_meta_arena);
}

TEST_CASE(RecurrentSnapshotFixture, snapshot_and_restore_recurrent_state) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr);
    if (!backend) SKIP("CPU backend is unavailable");

    {
        size_t graph_capacity = 0;
        CHECK(dflash::common::detail::target_paged_tree_graph_capacity(
            16, 16, graph_capacity));
        ggml_init_params graph_params{};
        graph_params.mem_size = 32 * 1024 * 1024;
        graph_params.no_alloc = true;
        ggml_context * graph_ctx = ggml_init(graph_params);
        CHECK(graph_ctx != nullptr);
        if (graph_ctx) {
            ggml_tensor * input =
                ggml_new_tensor_1d(graph_ctx, GGML_TYPE_F32, 1);
            ggml_set_input(input);
            ggml_cgraph * graph = ggml_new_graph_custom(
                graph_ctx, graph_capacity, false);
            for (int i = 0; i < 16385; ++i) {
                ggml_build_forward_expand(
                    graph, ggml_dup(graph_ctx, input));
            }
            CHECK(ggml_graph_n_nodes(graph) == 16385);
            ggml_gallocr_t graph_alloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend));
            CHECK(graph_alloc != nullptr);
            CHECK(graph_alloc &&
                  ggml_gallocr_alloc_graph(graph_alloc, graph));
            if (graph_alloc) ggml_gallocr_free(graph_alloc);
            ggml_free(graph_ctx);
        }
    }

    ggml_init_params params{};
    params.mem_size = 8 * ggml_tensor_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);
    if (!ctx) {
        ggml_backend_free(backend);
        SKIP("could not initialize ggml context");
    }

    ggml_tensor * ssm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_tensor * ssm_snap = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_tensor * conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 2);
    ggml_tensor * conv_snap = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 2);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buffer != nullptr);
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        SKIP("could not allocate CPU backend tensors");
    }

    TargetCache cache;
    cache.ssm_state = {ssm, nullptr};
    cache.ssm_state_snap = {ssm_snap, nullptr};
    cache.conv_state = {conv, nullptr};
    cache.conv_state_snap = {conv_snap, nullptr};

    const std::vector<float> ssm_original = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    };
    const std::vector<float> conv_original = {
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    };
    const std::vector<float> ssm_mutated(ssm_original.size(), -1.0f);
    const std::vector<float> conv_mutated(conv_original.size(), -2.0f);

    CHECK(ggml_nelements(ssm) == (int64_t) ssm_original.size());
    CHECK(ggml_nelements(conv) == (int64_t) conv_original.size());
    set_tensor(ssm, ssm_original);
    set_tensor(conv, conv_original);
    CHECK(snapshot_ssm_state(cache, backend));
    set_tensor(ssm, ssm_mutated);
    set_tensor(conv, conv_mutated);
    CHECK(restore_ssm_state(cache, backend));
    CHECK(get_tensor(ssm) == ssm_original);
    CHECK(get_tensor(conv) == conv_original);

    // A partial shard may omit a complete recurrent-state quartet, but an
    // asymmetric quartet must fail validation before any copy is queued.
    set_tensor(ssm, ssm_mutated);
    cache.conv_state_snap[0] = nullptr;
    CHECK(!snapshot_ssm_state(cache, backend));
    CHECK(get_tensor(ssm_snap) == ssm_original);
    cache.conv_state_snap[0] = conv_snap;

    CHECK(!snapshot_ssm_state(cache, nullptr));
    cache.ssm_state_snap.pop_back();
    CHECK(!restore_ssm_state(cache, backend));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}
