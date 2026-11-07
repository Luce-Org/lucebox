// KEEP_BASE: the chain capture verify graph drops
// exactly the DeltaNet final-state write into ssm_state (its cpy node and the
// view of the GDN result's final-state slot that the cpy read). The conv
// history write, the conv-window capture and the intermediate-state capture
// (the GDN op's persistent src[7]) stay. Builds the verify graph the target
// runs on a synthetic one-GDN-layer model on the CPU backend; no compute.

#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"
#include "qwen35_tiny_gdn_target.h"
#include "graph_builders.h"
#include "step_graph.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <array>

using luce::common::StepGraph;
using luce::common::build_target_step;
using luce::common::step_graph_destroy;
using luce_test::TinyGdnTarget;

namespace {

struct Qwen35KeepBaseGraph {};

struct GraphWrites {
    int nodes = 0;
    std::array<int, GGML_OP_COUNT> ops{};   // node count per op
    int ssm_state = 0;     // cpy into the persistent SSM state
    int conv_state = 0;    // cpy into the persistent conv history
    int conv_input = 0;    // cpy of the conv window into its capture buffer
    int gdn_persist = 0;   // GDN ops writing the F32 checkpoints directly
};

bool writes_into(const ggml_tensor * node, const ggml_tensor * target) {
    if (node->op != GGML_OP_CPY || !node->src[1]) return false;
    const ggml_tensor * dst = node->src[1];
    return dst == target || dst->view_src == target;
}

GraphWrites count_writes(const StepGraph & sg, const TinyGdnTarget & t) {
    GraphWrites g;
    g.nodes = ggml_graph_n_nodes(sg.gf);
    for (int i = 0; i < g.nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node(sg.gf, i);
        g.ops[(size_t)node->op]++;
        if (writes_into(node, t.cache.ssm_state[0])) g.ssm_state++;
        if (writes_into(node, t.cache.conv_state[0])) g.conv_state++;
        if (writes_into(node, t.cache.conv_input_cache[0])) g.conv_input++;
        if (node->op == GGML_OP_GATED_DELTA_NET &&
            node->src[7] == t.cache.ssm_intermediate[0]) {
            g.gdn_persist++;
        }
    }
    return g;
}

// The build Qwen35DFlashTarget::verify_batch runs for a chain verify block.
bool build_verify(TinyGdnTarget & t, StepGraph & sg, int kv_start, int n,
                  bool capture_delta, bool keep_base) {
    return build_target_step(sg, t.w, t.cache, t.backend, kv_start, n,
                             /*with_mask=*/true, /*capture=*/true,
                             /*capture_delta_intermediate=*/capture_delta,
                             /*fa_window=*/0, /*logits_tail_rows=*/0,
                             luce::common::KQ_MASK_PAD, /*capture_moe_router=*/false,
                             /*kvflash_mask=*/false, /*capture_qk=*/false,
                             /*paged_attention=*/false, /*n_seqs=*/1,
                             /*seq_slot=*/0, /*paged_max_kv_len=*/0,
                             /*n_prefill_tokens=*/0,
                             /*prefill_segments=*/nullptr,
                             /*n_prefill_segments=*/0, /*n_logits_rows=*/0,
                             /*compact_slots=*/false, /*top_n=*/0, keep_base);
}

}  // namespace

TEST_CASE(Qwen35KeepBaseGraph, keep_base_drops_only_the_ssm_state_write) {
    const luce_test::ScopedEnvVar kv_f16("LUCE_KV_F16", "1");
    TinyGdnTarget t(ggml_backend_cpu_init(), /*state_dim=*/16, /*raw_gates=*/false);
    REQUIRE(t.ok);
    REQUIRE(t.cache.ssm_intermediate[0] != nullptr);
    REQUIRE(t.cache.ssm_intermediate[0]->type == GGML_TYPE_F32);

    StepGraph sg;
    for (int n : {2, 16}) {
        REQUIRE(build_verify(t, sg, 8, n, /*capture_delta=*/true, /*keep_base=*/false));
        const GraphWrites parent = count_writes(sg, t);
        REQUIRE(build_verify(t, sg, 8, n, /*capture_delta=*/true, /*keep_base=*/true));
        const GraphWrites kept = count_writes(sg, t);

        CHECK_EQUAL(parent.ssm_state, 1);
        CHECK_EQUAL(kept.ssm_state, 0);
        CHECK_EQUAL(kept.nodes, parent.nodes - 2);
        for (int op = 0; op < GGML_OP_COUNT; ++op) {
            const int dropped = (op == GGML_OP_CPY || op == GGML_OP_VIEW) ? 1 : 0;
            CHECK_EQUAL(kept.ops[(size_t)op], parent.ops[(size_t)op] - dropped);
        }
        CHECK_EQUAL(kept.conv_state, parent.conv_state);
        CHECK_EQUAL(kept.conv_input, parent.conv_input);
        CHECK_EQUAL(parent.conv_input, 1);
        CHECK_EQUAL(parent.gdn_persist, 1);
        CHECK_EQUAL(kept.gdn_persist, 1);
    }
    step_graph_destroy(sg);
}

TEST_CASE(Qwen35KeepBaseGraph, keep_base_requires_a_capture_verify) {
    const luce_test::ScopedEnvVar kv_f16("LUCE_KV_F16", "1");
    TinyGdnTarget t(ggml_backend_cpu_init(), /*state_dim=*/16, /*raw_gates=*/false);
    REQUIRE(t.ok);
    StepGraph sg;
    // A replay (no delta capture) has no checkpoints to commit from.
    CHECK(!build_verify(t, sg, 8, 4, /*capture_delta=*/false, /*keep_base=*/true));
    CHECK(build_verify(t, sg, 8, 4, /*capture_delta=*/false, /*keep_base=*/false));
    step_graph_destroy(sg);
}
