// Glue fusions host logic, GPU-free.
//
// the Q8 memo (ggml-cuda/glue-q8-memo.h): which later matmuls of
// a graph may read the q8_1 activation one MMQ call quantized. The nodes are
// built with ggml on a no-alloc context and given fake, non-overlapping
// addresses, so the scan sees the same tensors the CUDA loop would.
// test_glue_fusions_gpu checks the memo itself on the GPU.
//
// Strided gates (qwen35/glue_fusions.h): which copies of stacked
// projection slices the qwen35 graph drops on which path, and the inert
// reasons where a copy stays.
//
// Gate|up stack (qwen35/glue_fusions.h): when the FFN runs one
// GEMM over the stacked ffn_gate|ffn_up, and when it says it is inert.

#include "CppUnitTestFramework.hpp"

#include "../../deps/llama.cpp/ggml/src/ggml-cuda/glue-q8-memo.h"
#include "qwen35/glue_fusions.h"

#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

struct GlueQ8Memo {};

// D4 / DS4 / D2S6 as mmq_get_q8_1_ds_layout assigns them; -1: no MMQ path.
int layout_of(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q6_K:
            return 0;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            return 1;
        case GGML_TYPE_Q2_K:
            return 2;
        default:
            return -1;
    }
}

constexpr int64_t K = 256;   // row length
constexpr int64_t N = 16;    // columns (verify width)

// A no-alloc context whose tensors get fake addresses: each new allocation
// takes the next 1 MiB slot, so distinct tensors never overlap unless a test
// places one over another on purpose.
struct FakeGraph {
    ggml_context * ctx = nullptr;
    uintptr_t next = 0x10000000;
    std::vector<ggml_tensor *> nodes;

    FakeGraph() {
        ggml_init_params ip{};
        ip.mem_size = 256 * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
    }
    ~FakeGraph() { ggml_free(ctx); }

    ggml_backend_buffer * fake_buffer() { return (ggml_backend_buffer *) (uintptr_t) 0x1; }

    // Give `t` its own address (or `at`) and mark it as a buffer tensor.
    ggml_tensor * place(ggml_tensor * t, uintptr_t at = 0) {
        if (at == 0) {
            at = next;
            next += 1u << 20;
        }
        t->data = (void *) at;
        t->buffer = fake_buffer();
        return t;
    }
    // A graph node the CUDA loop launches.
    ggml_tensor * node(ggml_tensor * t, uintptr_t at = 0) {
        if (t->view_src == nullptr) {
            place(t, at);
        } else {
            t->buffer = fake_buffer();
        }
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        nodes.push_back(t);
        return t;
    }
    ggml_tensor * weight(ggml_type type, int64_t rows) {
        return place(ggml_new_tensor_2d(ctx, type, K, rows));
    }
    ggml_tensor * activation() { return place(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N)); }
    int index_of(const ggml_tensor * t) const {
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i] == t) return (int) i;
        }
        return -1;
    }
    glue_q8_plan scan(const ggml_tensor * from, int window = 0) const {
        const int at = index_of(from);
        return glue_q8_scan(nodes.data(), (int) nodes.size(), at, from->src[1],
                            layout_of(from->src[0]->type), layout_of, window);
    }
};

}  // namespace

TEST_CASE(GlueQ8Memo, same_layout_readers_share_until_the_last_one) {
    FakeGraph g;
    ggml_tensor * x = g.node(ggml_rms_norm(g.ctx, g.activation(), 1e-6f));
    ggml_tensor * a = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), x));
    ggml_tensor * b = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_Q8_0, 96), x));    // D4 too
    ggml_tensor * c = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_Q4_K, 512), x));   // DS4
    ggml_tensor * d = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_Q6_K, 512), x));   // D4
    const glue_q8_plan p = g.scan(a);
    CHECK_EQUAL(p.last_reader, g.index_of(d));
    CHECK_EQUAL(p.clean_until, g.index_of(d));
    // The DS4 reader has no later DS4 reader of its own.
    CHECK_EQUAL(g.scan(c).last_reader, -1);
    (void) b;
}

TEST_CASE(GlueQ8Memo, no_reader_of_another_layout_or_without_mmq) {
    FakeGraph g;
    ggml_tensor * x = g.node(ggml_rms_norm(g.ctx, g.activation(), 1e-6f));
    ggml_tensor * a = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_Q4_K, 512), x));
    g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_Q2_K, 512), x));    // D2S6
    g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), x));  // D4
    g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_F16, 512), x));     // not quantized
    CHECK_EQUAL(g.scan(a).last_reader, -1);
}

TEST_CASE(GlueQ8Memo, an_in_place_write_ends_the_window) {
    FakeGraph g;
    ggml_tensor * x = g.node(ggml_rms_norm(g.ctx, g.activation(), 1e-6f));
    ggml_tensor * a = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), x));
    ggml_tensor * b = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), x));
    ggml_tensor * w = g.node(ggml_scale_inplace(g.ctx, x, 2.0f));  // rewrites x's bytes
    g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), x));
    const glue_q8_plan p = g.scan(a);
    CHECK_EQUAL(p.last_reader, g.index_of(b));
    CHECK_EQUAL(p.clean_until, g.index_of(w) - 1);
}

TEST_CASE(GlueQ8Memo, a_node_placed_over_the_bytes_ends_the_window) {
    FakeGraph g;
    ggml_tensor * x = g.node(ggml_rms_norm(g.ctx, g.activation(), 1e-6f));
    ggml_tensor * a = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), x));
    // gallocr reuses x's memory for an unrelated node, partly overlapping it.
    ggml_tensor * other = g.node(ggml_rms_norm(g.ctx, g.activation(), 1e-6f),
                                 (uintptr_t) x->data + 64);
    g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), x));
    const glue_q8_plan p = g.scan(a);
    CHECK_EQUAL(p.last_reader, -1);
    CHECK_EQUAL(p.clean_until, g.index_of(other) - 1);
}

TEST_CASE(GlueQ8Memo, a_non_matmul_reader_counts_as_a_writer) {
    FakeGraph g;
    ggml_tensor * x = g.node(ggml_rms_norm(g.ctx, g.activation(), 1e-6f));
    ggml_tensor * a = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), x));
    ggml_tensor * r = g.node(ggml_add(g.ctx, x, g.activation()));  // may write a source in place
    g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), x));
    const glue_q8_plan p = g.scan(a);
    CHECK_EQUAL(p.last_reader, -1);
    CHECK_EQUAL(p.clean_until, g.index_of(r) - 1);
}

TEST_CASE(GlueQ8Memo, a_matmul_reading_a_view_of_the_bytes_does_not_end_the_window) {
    FakeGraph g;
    ggml_tensor * x = g.node(ggml_rms_norm(g.ctx, g.activation(), 1e-6f));
    ggml_tensor * a = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), x));
    ggml_tensor * half = g.node(ggml_view_2d(g.ctx, x, K, N / 2, x->nb[1], 0));  // launches nothing
    g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), half));  // other ne: not a reader
    ggml_tensor * b = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), x));
    const glue_q8_plan p = g.scan(a);
    CHECK_EQUAL(p.last_reader, g.index_of(b));
    CHECK_EQUAL(p.clean_until, g.index_of(b));
}

TEST_CASE(GlueQ8Memo, a_reshape_of_the_input_is_the_same_input) {
    FakeGraph g;
    ggml_tensor * x = g.node(ggml_rms_norm(g.ctx, g.activation(), 1e-6f));
    ggml_tensor * a = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), x));
    ggml_tensor * flat = g.node(ggml_reshape_3d(g.ctx, x, K, N, 1));     // same ne after padding
    ggml_tensor * b = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), flat));
    ggml_tensor * split = g.node(ggml_reshape_3d(g.ctx, x, K, N / 2, 2)); // other ne: another layout
    g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), split));
    CHECK_TRUE(glue_q8_same_input(x, flat));
    CHECK_FALSE(glue_q8_same_input(x, split));
    CHECK_EQUAL(g.scan(a).last_reader, g.index_of(b));
}

TEST_CASE(GlueQ8Memo, nodes_the_loop_skips_write_nothing) {
    FakeGraph g;
    ggml_tensor * x = g.node(ggml_rms_norm(g.ctx, g.activation(), 1e-6f));
    ggml_tensor * a = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), x));
    ggml_tensor * dead = g.node(ggml_scale_inplace(g.ctx, x, 2.0f));
    dead->flags &= ~GGML_TENSOR_FLAG_COMPUTE;   // not computed: the loop skips it
    ggml_tensor * b = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), x));
    CHECK_EQUAL(g.scan(a).last_reader, g.index_of(b));
}

TEST_CASE(GlueQ8Memo, the_window_bounds_the_scan) {
    FakeGraph g;
    ggml_tensor * x = g.node(ggml_rms_norm(g.ctx, g.activation(), 1e-6f));
    ggml_tensor * a = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), x));
    for (int i = 0; i < 6; ++i) {
        g.node(ggml_rms_norm(g.ctx, g.activation(), 1e-6f));
    }
    ggml_tensor * b = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), x));
    CHECK_EQUAL(g.scan(a, 7).last_reader, g.index_of(b));
    const glue_q8_plan p = g.scan(a, 6);
    CHECK_EQUAL(p.last_reader, -1);
    CHECK_EQUAL(p.clean_until, g.index_of(a) + 6);
}

TEST_CASE(GlueQ8Memo, only_contiguous_f32_inputs_in_a_buffer_are_keyed) {
    FakeGraph g;
    ggml_tensor * x = g.node(ggml_rms_norm(g.ctx, g.activation(), 1e-6f));
    ggml_tensor * t = g.node(ggml_transpose(g.ctx, ggml_reshape_2d(g.ctx, x, N, K)));
    CHECK_TRUE(glue_q8_keyable(x));
    CHECK_FALSE(glue_q8_keyable(t));
    ggml_tensor * a = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), t));
    g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_IQ4_XS, 512), t));
    CHECK_EQUAL(g.scan(a).last_reader, -1);
    ggml_tensor * synthetic = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, N);
    synthetic->data = (void *) (uintptr_t) 0x7000000;   // no buffer: a stack tensor in a fusion
    CHECK_FALSE(glue_q8_keyable(synthetic));
}

TEST_CASE(GlueQ8Memo, overlap_is_half_open) {
    CHECK_TRUE(glue_q8_overlap((void *) 0x1000, 0x100, (void *) 0x10ff, 1));
    CHECK_FALSE(glue_q8_overlap((void *) 0x1000, 0x100, (void *) 0x1100, 1));
    CHECK_FALSE(glue_q8_overlap((void *) 0x1100, 1, (void *) 0x1000, 0x100));
    CHECK_FALSE(glue_q8_overlap((void *) 0x1000, 0, (void *) 0x1000, 0x100));
}

// The first MMQ reader of a producer's output
// (glue_q8_first_reader), whose layout the norm writes its q8_1 rows in.
namespace {

int first_reader(const FakeGraph & g, const ggml_tensor * producer, bool (*takes_mmq)(const ggml_tensor *)) {
    return glue_q8_first_reader(g.nodes.data(), (int) g.nodes.size(), g.index_of(producer), producer, layout_of,
                                takes_mmq, 0);
}

bool every_matmul_takes_mmq(const ggml_tensor *) { return true; }
bool q8_0_takes_mmvq(const ggml_tensor * mm) { return mm->src[0]->type != GGML_TYPE_Q8_0; }

ggml_tensor * norm_node(FakeGraph & g) {
    ggml_tensor * rms = g.node(ggml_rms_norm(g.ctx, g.activation(), 1e-6f));
    return g.node(ggml_mul(g.ctx, rms, g.activation()));
}

}  // namespace

TEST_CASE(GlueQ8Memo, the_norm_writes_the_layout_of_its_first_mmq_reader) {
    FakeGraph g;
    ggml_tensor * n = norm_node(g);
    g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_F16, 512), n));                      // not quantized
    ggml_tensor * a = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_Q8_0, 512), n));   // D4
    ggml_tensor * b = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_Q4_K, 512), n));   // DS4
    ggml_tensor * c = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_Q5_K, 256), n));   // DS4
    CHECK_EQUAL(first_reader(g, n, every_matmul_takes_mmq), g.index_of(a));
    // A reader that runs MMVQ quantizes its own form: the next one decides.
    CHECK_EQUAL(first_reader(g, n, q8_0_takes_mmvq), g.index_of(b));
    // Planned from the norm in that layout, the entry serves both DS4 readers.
    const glue_q8_plan p = glue_q8_scan(g.nodes.data(), (int) g.nodes.size(), g.index_of(n), n,
                                        layout_of(GGML_TYPE_Q4_K), layout_of, 0);
    CHECK_EQUAL(p.last_reader, g.index_of(c));
}

TEST_CASE(GlueQ8Memo, no_norm_q8_when_its_output_is_rewritten_first) {
    FakeGraph g;
    ggml_tensor * n = norm_node(g);
    g.node(ggml_scale_inplace(g.ctx, n, 2.0f));   // rewrites the norm's bytes before any reader
    g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_Q8_0, 512), n));
    CHECK_EQUAL(first_reader(g, n, every_matmul_takes_mmq), -1);
}

TEST_CASE(GlueQ8Memo, no_norm_q8_for_a_reader_of_other_ne_or_without_a_layout) {
    FakeGraph g;
    ggml_tensor * n = norm_node(g);
    ggml_tensor * split = g.node(ggml_reshape_3d(g.ctx, n, K, N / 2, 2));   // another MMQ layout of the bytes
    g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_Q8_0, 512), split));
    g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_Q4_0, 512), n));           // layout_of -1 in this table
    CHECK_EQUAL(first_reader(g, n, every_matmul_takes_mmq), -1);
    ggml_tensor * flat = g.node(ggml_reshape_3d(g.ctx, n, K, N, 1));        // the same ne: the same input
    ggml_tensor * r = g.node(ggml_mul_mat(g.ctx, g.weight(GGML_TYPE_Q6_K, 512), flat));
    CHECK_EQUAL(first_reader(g, n, every_matmul_takes_mmq), g.index_of(r));
}

// ── Strided gates ──────────────────────────────────────────

namespace {

struct StridedGdnPlanCase {};

using luce::qwen35::glue::StridedGdnFacts;
using luce::qwen35::glue::StridedGdnPlan;
using luce::qwen35::glue::strided_gdn_plan;

// The single-request chain verify of Qwen3.8 at T = 16 on HIP: both stacks,
// both copies, fused chain kernels with raw gates.
StridedGdnFacts chain_verify() {
    StridedGdnFacts f;
    f.on = true;
    f.stacked_ba = true;
    f.ba_copy = true;
    f.fused_chain = true;
    f.raw_gates = true;
    f.stacked_z = true;
    f.z_copy = true;
    f.cuda_backend = true;
    f.fusion_enabled = true;
    return f;
}

}  // namespace

TEST_CASE(StridedGdnPlanCase, turned_off_changes_nothing) {
    StridedGdnFacts f = chain_verify();
    f.on = false;
    const StridedGdnPlan p = strided_gdn_plan(f);
    CHECK_FALSE(p.gates);
    CHECK_FALSE(p.z);
    CHECK_NULL(p.inert_why);
}

TEST_CASE(StridedGdnPlanCase, chain_verify_drops_all_three_copies) {
    const StridedGdnPlan p = strided_gdn_plan(chain_verify());
    CHECK_TRUE(p.gates);
    CHECK_TRUE(p.z);
    CHECK_NULL(p.inert_why);
}

TEST_CASE(StridedGdnPlanCase, nothing_to_drop_where_the_parent_copies_nothing) {
    StridedGdnFacts f = chain_verify();
    f.ba_copy = false;   // T = 1: the slices are contiguous
    f.z_copy = false;
    const StridedGdnPlan p = strided_gdn_plan(f);
    CHECK_FALSE(p.gates);
    CHECK_FALSE(p.z);
    CHECK_NULL(p.inert_why);
    f = chain_verify();
    f.stacked_ba = false;   // separate ssm_beta / ssm_alpha GEMMs: contiguous already
    f.ba_copy = false;
    f.stacked_z = false;    // the 19 layers whose attn_gate is its own GEMM
    f.z_copy = false;
    const StridedGdnPlan q = strided_gdn_plan(f);
    CHECK_FALSE(q.gates);
    CHECK_FALSE(q.z);
    CHECK_NULL(q.inert_why);
}

TEST_CASE(StridedGdnPlanCase, mapped_verify_reads_z_in_place) {
    StridedGdnFacts f = chain_verify();
    f.fused_chain = false;
    f.mapped_verify = true;
    StridedGdnPlan p = strided_gdn_plan(f);
    CHECK_FALSE(p.gates);   // its own views and kernel; nothing for the fusion to do
    CHECK_TRUE(p.z);        // the same contiguous GDN output rows as the chain
    CHECK_NULL(p.inert_why);
    f.fusion_enabled = false;   // the unfused silu needs a contiguous input
    p = strided_gdn_plan(f);
    CHECK_FALSE(p.z);
    CHECK_NOT_NULL(p.inert_why);
}

// Every segment that is not the dense single-sequence chain on the fused
// sequential kernels keeps both copies, z included: tree verify, compact
// multi-sequence decode (T = 1, S > 1: z is a strided [n_z, S] slice), ragged
// prefill, chunked, SpecLA, LUCE_QWEN35_NO_FUSED_KERNELS. The graph builder
// reports all of them as fused_chain = false.
TEST_CASE(StridedGdnPlanCase, other_paths_keep_both_copies) {
    for (int raw = 0; raw < 2; ++raw) {
        StridedGdnFacts f = chain_verify();
        f.fused_chain = false;
        f.raw_gates = raw != 0;
        const StridedGdnPlan p = strided_gdn_plan(f);
        CHECK_FALSE(p.gates);
        CHECK_FALSE(p.z);
        CHECK_NOT_NULL(p.inert_why);
        if (p.inert_why) {
            CHECK_TRUE(std::string(p.inert_why).find("compact multi-sequence decode") != std::string::npos);
        }
    }
    // z alone (separate ssm_beta / ssm_alpha GEMMs) stays too.
    StridedGdnFacts f = chain_verify();
    f.fused_chain = false;
    f.stacked_ba = false;
    f.ba_copy = false;
    const StridedGdnPlan p = strided_gdn_plan(f);
    CHECK_FALSE(p.z);
    CHECK_NOT_NULL(p.inert_why);
}

TEST_CASE(StridedGdnPlanCase, missing_raw_gate_kernel_keeps_the_gate_copies) {
    StridedGdnFacts f = chain_verify();
    f.raw_gates = false;     // no ssm_gate_ba: the gates are computed in the graph
    const StridedGdnPlan p = strided_gdn_plan(f);
    CHECK_FALSE(p.gates);
    CHECK_TRUE(p.z);         // still the dense chain: z reads the stack
    CHECK_NOT_NULL(p.inert_why);
}

TEST_CASE(StridedGdnPlanCase, z_needs_the_fused_cuda_kernel) {
    StridedGdnFacts f = chain_verify();
    f.fusion_enabled = false;
    StridedGdnPlan p = strided_gdn_plan(f);
    CHECK_TRUE(p.gates);
    CHECK_FALSE(p.z);
    CHECK_NOT_NULL(p.inert_why);
    f = chain_verify();
    f.cuda_backend = false;
    f.fused_chain = false;   // a CPU backend has no fused chain kernel either
    p = strided_gdn_plan(f);
    CHECK_FALSE(p.gates);
    CHECK_FALSE(p.z);
    CHECK_NOT_NULL(p.inert_why);
    if (p.inert_why) {
        CHECK_EQUAL(std::string(p.inert_why), std::string("copies stay: not a CUDA/HIP backend"));
    }
}

TEST_CASE(StridedGdnPlanCase, both_fusions_are_on_by_default) {
    const luce::qwen35::glue::Fusions defaults;
    CHECK(defaults.strided_gates && defaults.gate_up_stack);
    CHECK(luce::qwen35::glue::strided_gates_on() && luce::qwen35::glue::gate_up_stack_on());
}

// ── Gate|up stack ─────────────────────────────────────────

namespace {

struct GateUpPlanCase {};

using luce::qwen35::glue::GateUpFacts;
using luce::qwen35::glue::GateUpPlan;
using luce::qwen35::glue::gate_up_plan;

GateUpFacts stacked_exact() {
    GateUpFacts f;
    f.on = true;
    f.stacked = true;
    f.unit_scales = true;
    f.backend_inexact = nullptr;
    return f;
}

}  // namespace

TEST_CASE(GateUpPlanCase, turned_off_keeps_two_gemms) {
    GateUpFacts f = stacked_exact();
    f.on = false;
    const GateUpPlan p = gate_up_plan(f);
    CHECK_FALSE(p.stack);
    CHECK_NULL(p.inert_why);
}

TEST_CASE(GateUpPlanCase, exact_backend_stacks) {
    const GateUpPlan p = gate_up_plan(stacked_exact());
    CHECK_TRUE(p.stack);
    CHECK_NULL(p.inert_why);
}

TEST_CASE(GateUpPlanCase, layers_without_a_stack_are_silent) {
    GateUpFacts f = stacked_exact();
    f.stacked = false;              // gate and up of different types: no alias
    f.backend_inexact = "ignored";
    const GateUpPlan p = gate_up_plan(f);
    CHECK_FALSE(p.stack);
    CHECK_NULL(p.inert_why);
}

TEST_CASE(GateUpPlanCase, inexact_backend_or_scales_are_inert) {
    GateUpFacts f = stacked_exact();
    f.backend_inexact = "MMVQ width (decode keeps its gate/up kernels)";
    GateUpPlan p = gate_up_plan(f);
    CHECK_FALSE(p.stack);
    CHECK_EQUAL(std::string(p.inert_why), std::string(f.backend_inexact));
    f = stacked_exact();
    f.unit_scales = false;          // NVFP4 per-tensor scales apply per GEMM
    p = gate_up_plan(f);
    CHECK_FALSE(p.stack);
    CHECK_NOT_NULL(p.inert_why);
}
