// qwen35 AR decode must upload its step inputs into the graph that computes.
//
// build_target_step frees the step graph (step_graph_free: inp_embed and
// positions = nullptr), creates fresh input tensors in a new context, and lets
// the gallocr place them. Inputs written before the build land in the previous
// graph's tensors and the step computes on whatever bytes the gallocr put
// under the new ones.
//
// Same-shape rebuilds reuse the previous placement (ggml_gallocr_alloc_graph
// skips reserve when nodes, leafs and sizes match), so AR step -> AR step is
// harmless. The first AR step after a prefill chunk or a verify block rebuilds
// from a different shape, which re-plans the buffer.
//
// These cases call build_ar_step_with_inputs, the function do_ar_decode runs,
// after a prefill-shaped and a verify-shaped build on a tiny synthetic
// single-attention-layer model, and read the inputs back from the graph the
// step would compute on. No model file, no compute.
#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"
#include "internal.h"
#include "graph_builders.h"
#include "step_graph.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef LUCE_BACKEND_CUDA
#include "ggml-cuda.h"
#endif

#include <cstdint>
#include <cstdio>
#include <vector>

using luce::common::StepGraph;
using luce::common::TargetCache;
using luce::common::TargetLayer;
using luce::common::TargetWeights;
using luce::common::build_ar_step_with_inputs;
using luce::common::build_target_step;
using luce::common::step_graph_destroy;

namespace {

struct Qwen35ArInputOrderFixture {};

// One full-attention layer, dense SwiGLU FFN, f32 weights on `be`.
// Takes ownership of the backend.
struct TinyTarget {
    static constexpr int kHidden   = 32;
    static constexpr int kHeadDim  = 16;
    static constexpr int kHeads    = 2;
    static constexpr int kHeadsKv  = 1;
    static constexpr int kFf       = 64;
    static constexpr int kVocab    = 64;
    static constexpr int kMaxCtx   = 64;

    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    TargetWeights w;
    TargetCache cache;
    bool ok = false;

    explicit TinyTarget(ggml_backend_t be) : backend(be) {
        if (!backend) return;

        w.n_layer = 1;
        w.full_attention_interval = 1;   // layer 0 is the attention layer
        w.n_embd = kHidden;
        w.n_embd_head_k = kHeadDim;
        w.n_embd_head_v = kHeadDim;
        w.n_head = kHeads;
        w.n_head_kv = kHeadsKv;
        w.n_ff = kFf;
        w.n_vocab = kVocab;
        w.rope_dimension_count = 8;
        w.rope_sections[0] = 2; w.rope_sections[1] = 1;
        w.rope_sections[2] = 1; w.rope_sections[3] = 0;
        w.n_capture_layers = 0;
        w.is_moe = false;
        w.ssm_d_inner = 32; w.ssm_d_state = 1; w.ssm_dt_rank = 1;
        w.ssm_n_group = 1;  w.ssm_d_conv = 2;
        w.layers.resize(1);

        ggml_init_params ip{};
        ip.mem_size = 32 * ggml_tensor_overhead();
        ip.no_alloc = true;
        wctx = ggml_init(ip);
        if (!wctx) return;

        const int q_dim = kHeadDim * kHeads;
        const int kv_dim = kHeadDim * kHeadsKv;
        TargetLayer & L = w.layers[0];
        L.attn_norm      = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHidden);
        L.attn_post_norm = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHidden);
        L.wq     = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kHidden, 2 * q_dim);
        L.wk     = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kHidden, kv_dim);
        L.wv     = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kHidden, kv_dim);
        L.wo     = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, q_dim, kHidden);
        L.q_norm = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHeadDim);
        L.k_norm = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHeadDim);
        L.w_gate = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kHidden, kFf);
        L.w_up   = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kHidden, kFf);
        L.w_down = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kFf, kHidden);
        w.out_norm = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHidden);
        w.output   = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, kHidden, kVocab);

        wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
        if (!wbuf) return;
        for (ggml_tensor * t = ggml_get_first_tensor(wctx); t;
             t = ggml_get_next_tensor(wctx, t)) {
            std::vector<float> ones((size_t)ggml_nelements(t), 1.0f);
            ggml_backend_tensor_set(t, ones.data(), 0, ggml_nbytes(t));
        }

        // No recurrent layers, so no rollback cache (an empty rollback
        // context cannot be allocated); attention-only needs the base cache.
        ok = luce::common::create_target_cache_partial(
            w, kMaxCtx, /*max_verify_tokens=*/8, backend, cache,
            /*prefill_only=*/true, /*layer_begin=*/0, /*layer_end=*/1,
            /*allocate_target_feat=*/false, /*ctx_alloc=*/0,
            /*f32_ssm_intermediates=*/false, /*paged_attention=*/false,
            /*n_seq_slots=*/1, /*concurrent_tree=*/false,
            GGML_TYPE_F16, GGML_TYPE_F16);
        if (!ok) std::fprintf(stderr, "[tiny-target] cache: %s\n", luce_last_error());
    }

    ~TinyTarget() {
        if (ok) luce::common::free_target_cache(cache);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
        if (backend) ggml_backend_free(backend);
    }
};

// The build the backend's prefill runs for a prompt chunk
// (qwen35_backend.cpp do_prefill: with_mask for n_tokens > 1, capture=true,
// logits_tail_rows=0 on the final chunk).
bool build_prefill_chunk(TinyTarget & t, StepGraph & sg, int kv_start, int n) {
    return build_target_step(sg, t.w, t.cache, t.backend, kv_start, n,
                             /*with_mask=*/n > 1, /*capture=*/true,
                             /*capture_delta_intermediate=*/false,
                             /*fa_window=*/0, /*logits_tail_rows=*/0);
}

// The build Qwen35DFlashTarget::verify_batch runs for a chain verify block
// without a pager (mask for n_tokens > 1, capture=true, delta-intermediate
// capture on under the default fast-rollback policy).
bool build_verify_block(TinyTarget & t, StepGraph & sg, int kv_start, int n) {
    return build_target_step(sg, t.w, t.cache, t.backend, kv_start, n,
                             /*with_mask=*/n > 1, /*capture=*/true,
                             /*capture_delta_intermediate=*/true,
                             /*fa_window=*/0, /*logits_tail_rows=*/0);
}

// The build do_ar_decode runs for one decode token without --kvflash
// (with_mask only for bailingmoe3, no capture, no pool mask).
bool build_ar_step(TinyTarget & t, StepGraph & sg, int kv_start) {
    return build_target_step(sg, t.w, t.cache, t.backend, kv_start, 1,
                             /*with_mask=*/false, /*capture=*/false,
                             /*capture_delta_intermediate=*/false,
                             /*fa_window=*/0, /*logits_tail_rows=*/0);
}

size_t buffer_offset(const ggml_tensor * t) {
    return (size_t)((const char *)t->data -
                    (const char *)ggml_backend_buffer_get_base(t->buffer));
}

// Runs do_ar_decode's input path for the step at `committed`
// (build_ar_step_with_inputs: build the step graph, upload the token's
// embedding and positions) and reads the inputs back from the graph the step
// would compute on. Returns true when they hold the uploaded values. `tag`
// names the case in the diagnostics.
bool ar_step_inputs_survive_build(TinyTarget & t, StepGraph & sg, int committed,
                                  const char * tag) {
    const int hidden = TinyTarget::kHidden;
    std::vector<float> embed((size_t)hidden);
    for (int i = 0; i < hidden; ++i) embed[(size_t)i] = 1.0f + (float)i;

    const size_t old_embed_off = buffer_offset(sg.inp_embed);
    const size_t old_pos_off   = buffer_offset(sg.positions);

    if (!build_ar_step_with_inputs(sg, [&] { return build_ar_step(t, sg, committed); },
                                   embed.data(), hidden, committed)) {
        std::fprintf(stderr, "[%s] AR build failed\n", tag);
        return false;
    }

    std::vector<float> embed_back((size_t)hidden, 0.0f);
    int32_t pos_back[4] = {-1, -1, -1, -1};
    ggml_backend_tensor_get(sg.inp_embed, embed_back.data(), 0, sizeof(float) * hidden);
    ggml_backend_tensor_get(sg.positions, pos_back, 0, sizeof(int32_t) * 4);

    const bool embed_ok = embed_back == embed;
    const bool pos_ok = pos_back[0] == committed && pos_back[1] == committed &&
                        pos_back[2] == committed && pos_back[3] == 0;
    std::fprintf(stderr,
        "[%s] inp_embed offset %zu -> %zu (%s), positions offset %zu -> %zu (%s): "
        "positions read back {%d,%d,%d,%d}, wanted {%d,%d,%d,0}\n",
        tag, old_embed_off, buffer_offset(sg.inp_embed), embed_ok ? "intact" : "LOST",
        old_pos_off, buffer_offset(sg.positions), pos_ok ? "intact" : "LOST",
        pos_back[0], pos_back[1], pos_back[2], pos_back[3],
        committed, committed, committed);
    return embed_ok && pos_ok;
}

}  // namespace

// Control: an AR step after an AR step rebuilds the same shape and the
// gallocr keeps the placement, so writing before the build goes unnoticed.
TEST_CASE(Qwen35ArInputOrderFixture, same_shape_ar_step_keeps_inputs) {
    const luce_test::ScopedEnvVar kv_f16("LUCE_KV_F16", "1");
    TinyTarget t(ggml_backend_cpu_init());
    REQUIRE(t.ok);
    StepGraph sg;
    REQUIRE(build_ar_step(t, sg, 8));
    CHECK(ar_step_inputs_survive_build(t, sg, 9, "ar->ar cpu"));
    step_graph_destroy(sg);
}

// The first AR step of a request follows the prompt's final prefill chunk.
TEST_CASE(Qwen35ArInputOrderFixture, first_ar_step_after_prefill_keeps_inputs) {
    const luce_test::ScopedEnvVar kv_f16("LUCE_KV_F16", "1");
    TinyTarget t(ggml_backend_cpu_init());
    REQUIRE(t.ok);
    StepGraph sg;
    REQUIRE(build_prefill_chunk(t, sg, 0, 8));
    CHECK(ar_step_inputs_survive_build(t, sg, 8, "prefill->ar cpu"));
    step_graph_destroy(sg);
}

// Spec decode falls back to do_ar_decode after a verify block (AR hop,
// pool exhaustion, end of the spec loop); the same graph object is shared.
TEST_CASE(Qwen35ArInputOrderFixture, first_ar_step_after_verify_block_keeps_inputs) {
    const luce_test::ScopedEnvVar kv_f16("LUCE_KV_F16", "1");
    TinyTarget t(ggml_backend_cpu_init());
    REQUIRE(t.ok);
    StepGraph sg;
    REQUIRE(build_prefill_chunk(t, sg, 0, 8));
    REQUIRE(build_verify_block(t, sg, 8, 4));
    CHECK(ar_step_inputs_survive_build(t, sg, 12, "verify->ar cpu"));
    step_graph_destroy(sg);
}

#ifdef LUCE_BACKEND_CUDA
// Same sequence on the backend the server runs: the gallocr plan is
// backend-independent apart from alignment, so the placement moves here too.
TEST_CASE(Qwen35ArInputOrderFixture, first_ar_step_after_prefill_keeps_inputs_cuda) {
    if (ggml_backend_cuda_get_device_count() < 1) SKIP("no CUDA device");
    const luce_test::ScopedEnvVar kv_f16("LUCE_KV_F16", "1");
    TinyTarget t(ggml_backend_cuda_init(0));
    REQUIRE(t.ok);
    StepGraph sg;
    REQUIRE(build_prefill_chunk(t, sg, 0, 8));
    CHECK(ar_step_inputs_survive_build(t, sg, 8, "prefill->ar cuda"));
    step_graph_destroy(sg);
}
#endif
