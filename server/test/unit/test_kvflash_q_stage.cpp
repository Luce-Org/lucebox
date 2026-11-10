// The verify query stage: the per-layer view must be allocatable on a
// tensor-parallel (meta) backend, and a staged row must read back per layer.

#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_query_capture.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <memory>
#include <string>
#include <vector>

namespace {
struct KvflashQStageFixture {};

using ContextPtr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
constexpr int kHd = 4, kHeads = 4, kLayers = 2, kRows = 16;

// Heads split in two: axis 1 of q_stage (qwen35_tensor_parallel.cpp), axis 2 of
// the query q_fa [head_dim, n_tokens, n_head].
ggml_backend_meta_split_state heads_split(const ggml_tensor * t, void *) {
    ggml_backend_meta_split_state s{};
    s.n_segments = 1;
    s.nr[0] = 1;
    const std::string name = t->name;
    const int axis = name == "q_stage" ? 1 : name == "q_fa" ? 2 : -1;
    if (axis < 0) { s.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED; return s; }
    s.axis = (ggml_backend_meta_split_axis) axis;
    s.ne[0] = t->ne[axis] / 2;
    s.ne[1] = t->ne[axis] - s.ne[0];
    return s;
}

// The graph's three stage lines: permute q_fa, view the stage, copy.
ggml_cgraph * stage_graph(ggml_context * g, ggml_tensor * q_stage, ggml_tensor * q_fa,
                          int fa_idx, int n_tokens) {
    ggml_tensor * qp = ggml_permute(g, q_fa, 0, 2, 1, 3);
    ggml_tensor * qd = luce::common::kvflash_q_stage_view(g, q_stage, fa_idx, n_tokens);
    ggml_cgraph * gf = ggml_new_graph(g);
    ggml_build_forward_expand(gf, ggml_cpy(g, qp, qd));
    return gf;
}
}

// Regression: a layer-major stage aborted graph allocation on the meta backend
// from 2 rows up ("fatal error" in ggml-backend-meta.cpp).
TEST_CASE(KvflashQStageFixture, view_allocates_on_a_head_split_backend) {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    REQUIRE(cpu != nullptr);
    ggml_backend_dev_t devs[2] = {cpu, cpu};
    ggml_backend_t backend = ggml_backend_dev_init(
        ggml_backend_meta_device(devs, 2, heads_split, nullptr), nullptr);
    REQUIRE(backend != nullptr);
    for (int n_tokens : {1, 2, 16}) {
        for (int fa_idx = 0; fa_idx < kLayers; ++fa_idx) {
            ContextPtr sctx(ggml_init({ggml_tensor_overhead() * 8, nullptr, true}), ggml_free);
            ggml_tensor * q_stage = ggml_new_tensor_4d(sctx.get(), GGML_TYPE_F32, kHd, kHeads, kRows, kLayers);
            ggml_set_name(q_stage, "q_stage");
            ggml_tensor * q_fa = ggml_new_tensor_3d(sctx.get(), GGML_TYPE_F32, kHd, n_tokens, kHeads);
            ggml_set_name(q_fa, "q_fa");
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(sctx.get(), backend);
            REQUIRE(buf != nullptr);
            ContextPtr gctx(ggml_init({ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true}), ggml_free);
            ggml_cgraph * gf = stage_graph(gctx.get(), q_stage, q_fa, fa_idx, n_tokens);
            ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            CHECK(ggml_gallocr_alloc_graph(galloc, gf));
            ggml_gallocr_free(galloc);
            ggml_backend_buffer_free(buf);
        }
    }
    ggml_backend_free(backend);
}

// 3 rows x 2 layers on the CPU backend: row r of layer l reads back as the
// query that layer staged for token r.
TEST_CASE(KvflashQStageFixture, staged_row_reads_back_per_layer) {
    const int n_tokens = 3;
    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);
    ContextPtr sctx(ggml_init({ggml_tensor_overhead() * 8, nullptr, true}), ggml_free);
    ggml_tensor * q_stage = ggml_new_tensor_4d(sctx.get(), GGML_TYPE_F32, kHd, kHeads, kRows, kLayers);
    ggml_tensor * q_fa[kLayers];
    for (int l = 0; l < kLayers; ++l) {
        q_fa[l] = ggml_new_tensor_3d(sctx.get(), GGML_TYPE_F32, kHd, n_tokens, kHeads);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(sctx.get(), backend);
    REQUIRE(buf != nullptr);
    // q_fa[l](d, t, h) = 1000 * l + 100 * t + 10 * h + d
    auto value = [](int l, int t, int h, int d) { return (float)(1000 * l + 100 * t + 10 * h + d); };
    for (int l = 0; l < kLayers; ++l) {
        std::vector<float> v((size_t)kHd * n_tokens * kHeads);
        for (int h = 0; h < kHeads; ++h)
            for (int t = 0; t < n_tokens; ++t)
                for (int d = 0; d < kHd; ++d) v[((size_t)h * n_tokens + t) * kHd + d] = value(l, t, h, d);
        ggml_backend_tensor_set(q_fa[l], v.data(), 0, v.size() * sizeof(float));
    }
    for (int l = 0; l < kLayers; ++l) {
        ContextPtr gctx(ggml_init({ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true}), ggml_free);
        ggml_cgraph * gf = stage_graph(gctx.get(), q_stage, q_fa[l], l, n_tokens);
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        REQUIRE(ggml_gallocr_alloc_graph(galloc, gf));
        REQUIRE(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
        ggml_gallocr_free(galloc);
    }
    const size_t plane = q_stage->nb[2];   // one q_cap plane: [head_dim, n_head]
    CHECK(plane == (size_t)kHd * kHeads * sizeof(float));
    for (int r = 0; r < n_tokens; ++r) {
        for (int l = 0; l < kLayers; ++l) {
            std::vector<float> got(kHd * kHeads);
            ggml_backend_tensor_get(q_stage, got.data(),
                                    luce::common::kvflash_q_stage_offset(q_stage, l, r), plane);
            bool same = true;
            for (int h = 0; h < kHeads; ++h)
                for (int d = 0; d < kHd; ++d) same = same && got[(size_t)h * kHd + d] == value(l, r, h, d);
            CHECK(same);
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
}
