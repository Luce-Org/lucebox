// SINGLE_REPLAY: the keep-base chain verify graph
// with a one-slot state map. Against the KEEP_BASE graph it swaps exactly the
// DeltaNet op for a mapped verify over the map (no checkpoint buffer, no final
// state), and adds one view node: the replay log, read in place from the GDN
// result, which stays a graph output. The conv history write and the conv
// window capture stay. Builds the graphs on a synthetic one-GDN-layer model
// on the CPU backend; no compute.

#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"
#include "qwen35_tiny_gdn_target.h"
#include "graph_builders.h"
#include "step_graph.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <array>
#include <cstdint>

using luce::common::StepGraph;
using luce::common::build_target_step;
using luce::common::step_graph_destroy;
using luce_test::TinyGdnTarget;

namespace {

struct Qwen35SingleReplayGraph {};

struct GdnGraph {
    int nodes = 0;
    std::array<int, GGML_OP_COUNT> ops{};
    int ssm_state = 0;      // cpy into the persistent SSM state
    int conv_state = 0;     // cpy into the persistent conv history
    int conv_input = 0;     // cpy of the conv window into its capture buffer
    const ggml_tensor * gdn = nullptr;
    int gdn_ops = 0;
};

// ggml_get_op_params_i32 lives in ggml-impl.h; op_params is public.
int32_t op_param(const ggml_tensor * t, int i) {
    return reinterpret_cast<const int32_t *>(t->op_params)[i];
}

bool writes_into(const ggml_tensor * node, const ggml_tensor * target) {
    if (node->op != GGML_OP_CPY || !node->src[1]) return false;
    const ggml_tensor * dst = node->src[1];
    return dst == target || dst->view_src == target;
}

GdnGraph inspect(const StepGraph & sg, const TinyGdnTarget & t) {
    GdnGraph g;
    g.nodes = ggml_graph_n_nodes(sg.gf);
    for (int i = 0; i < g.nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node(sg.gf, i);
        g.ops[(size_t)node->op]++;
        if (writes_into(node, t.cache.ssm_state[0])) g.ssm_state++;
        if (writes_into(node, t.cache.conv_state[0])) g.conv_state++;
        if (writes_into(node, t.cache.conv_input_cache[0])) g.conv_input++;
        if (node->op == GGML_OP_GATED_DELTA_NET) {
            g.gdn = node;
            g.gdn_ops++;
        }
    }
    return g;
}

bool build_verify(TinyGdnTarget & t, StepGraph & sg, int n, bool keep_base,
                  ggml_tensor * slot_ids) {
    return build_target_step(sg, t.w, t.cache, t.backend, /*kv_start=*/8, n,
                             /*with_mask=*/true, /*capture=*/true,
                             /*capture_delta_intermediate=*/true,
                             /*fa_window=*/0, /*logits_tail_rows=*/0,
                             luce::common::KQ_MASK_PAD, /*capture_moe_router=*/false,
                             /*kvflash_mask=*/false, /*capture_qk=*/false,
                             /*paged_attention=*/false, /*n_seqs=*/1,
                             /*seq_slot=*/0, /*paged_max_kv_len=*/0,
                             /*n_prefill_tokens=*/0,
                             /*prefill_segments=*/nullptr,
                             /*n_prefill_segments=*/0, /*n_logits_rows=*/0,
                             /*compact_slots=*/false, /*top_n=*/0, keep_base, slot_ids);
}

// The one-slot map Qwen35DFlashTarget allocates (I32 [1] = 0).
struct SlotMap {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * ids = nullptr;
    SlotMap(ggml_backend_t backend, ggml_type type) {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        ids = ggml_new_tensor_1d(ctx, type, 1);
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        const int32_t zero = 0;
        if (buf && type == GGML_TYPE_I32) ggml_backend_tensor_set(ids, &zero, 0, sizeof(zero));
    }
    ~SlotMap() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

}  // namespace

TEST_CASE(Qwen35SingleReplayGraph, mapped_verify_replaces_the_checkpoint_writes) {
    const luce_test::ScopedEnvVar kv_f16("LUCE_KV_F16", "1");
    TinyGdnTarget t(ggml_backend_cpu_init(), /*state_dim=*/16, /*raw_gates=*/false);
    REQUIRE(t.ok);
    REQUIRE(t.cache.ssm_intermediate[0] != nullptr);
    SlotMap map(t.backend, GGML_TYPE_I32);
    REQUIRE(map.buf != nullptr);

    StepGraph sg;
    for (int n : {2, 16}) {
        REQUIRE(build_verify(t, sg, n, /*keep_base=*/true, nullptr));
        const GdnGraph kept = inspect(sg, t);
        REQUIRE(kept.gdn != nullptr);
        CHECK(kept.gdn->src[7] == t.cache.ssm_intermediate[0]);
        CHECK_EQUAL(op_param(kept.gdn, 11), 0);
        CHECK(sg.delta_captures[0].replay_log == nullptr);

        REQUIRE(build_verify(t, sg, n, /*keep_base=*/true, map.ids));
        const GdnGraph replay = inspect(sg, t);
        REQUIRE(replay.gdn != nullptr);
        CHECK_EQUAL(replay.gdn_ops, 1);

        // The recurrence: a mapped verify (op_params[11]) reading slab 0 of
        // the live state through the map, no checkpoint buffer in src[7].
        CHECK_EQUAL(op_param(replay.gdn, 11), 1);
        CHECK(replay.gdn->src[8] == map.ids);
        CHECK(replay.gdn->src[7] == nullptr);
        CHECK(replay.gdn->src[5] == t.cache.ssm_state[0] ||
              replay.gdn->src[5]->view_src == t.cache.ssm_state[0]);
        CHECK(replay.gdn->flags & GGML_TENSOR_FLAG_OUTPUT);

        // Its replay log: [2*S_v + 1, H, n, 1] read in place from the result.
        const luce::common::DeltaNetCapture & cap = sg.delta_captures[0];
        REQUIRE(cap.replay_log != nullptr);
        CHECK(cap.replay_log->view_src == replay.gdn);
        CHECK(cap.replay_log->op == GGML_OP_VIEW);
        CHECK(cap.replay_log->flags & GGML_TENSOR_FLAG_OUTPUT);
        CHECK(cap.replay_log->data != nullptr);
        CHECK_EQUAL(cap.replay_log->ne[0], (int64_t)(2 * t.kState + 1));
        CHECK_EQUAL(cap.replay_log->ne[1], (int64_t)TinyGdnTarget::kVHeads);
        CHECK_EQUAL(cap.replay_log->ne[2], (int64_t)n);
        CHECK_EQUAL(cap.replay_log->ne[3], (int64_t)1);
        CHECK(ggml_is_contiguous(cap.replay_log));
        CHECK(cap.ssm_intermediate_states == nullptr);
        CHECK(cap.conv_input == t.cache.conv_input_cache[0]);

        // Everything else is the KEEP_BASE graph: the same ops, one more view
        // node (the log), no SSM state write, the same conv writes.
        CHECK_EQUAL(replay.nodes, kept.nodes + 1);
        for (int op = 0; op < GGML_OP_COUNT; ++op) {
            const int added = op == GGML_OP_VIEW ? 1 : 0;
            CHECK_EQUAL(replay.ops[(size_t)op], kept.ops[(size_t)op] + added);
        }
        CHECK_EQUAL(kept.ssm_state, 0);
        CHECK_EQUAL(replay.ssm_state, 0);
        CHECK_EQUAL(replay.conv_state, kept.conv_state);
        CHECK_EQUAL(replay.conv_input, kept.conv_input);
        CHECK_EQUAL(replay.conv_input, 1);
    }
    step_graph_destroy(sg);
}

TEST_CASE(Qwen35SingleReplayGraph, single_replay_requires_keep_base_and_an_i32_map) {
    const luce_test::ScopedEnvVar kv_f16("LUCE_KV_F16", "1");
    TinyGdnTarget t(ggml_backend_cpu_init(), /*state_dim=*/16, /*raw_gates=*/false);
    REQUIRE(t.ok);
    SlotMap map(t.backend, GGML_TYPE_I32);
    SlotMap wrong(t.backend, GGML_TYPE_F32);
    REQUIRE(map.buf != nullptr);
    REQUIRE(wrong.buf != nullptr);
    StepGraph sg;
    CHECK(!build_verify(t, sg, 4, /*keep_base=*/false, map.ids));
    CHECK(!build_verify(t, sg, 4, /*keep_base=*/true, wrong.ids));
    CHECK(build_verify(t, sg, 4, /*keep_base=*/true, map.ids));
    CHECK(build_verify(t, sg, 4, /*keep_base=*/false, nullptr));
    CHECK(sg.delta_captures[0].replay_log == nullptr);
    CHECK(sg.delta_captures[0].ssm_intermediate_states == t.cache.ssm_intermediate[0]);
    step_graph_destroy(sg);
}
