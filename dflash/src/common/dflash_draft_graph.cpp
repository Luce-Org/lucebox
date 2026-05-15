#include "dflash_draft_graph.h"
#include "draft/draft_graph.h"  // DraftGraphInputs, DraftGraphOutputs, build_draft_graph

#include "ggml-alloc.h"

#include <cstdio>

namespace dflash27b {

bool build_draft_step(
    StepGraph & sg,
    const DraftWeights & dw,
    ggml_tensor * lm_head,
    ggml_backend_t backend,
    int ctx_len,
    const DraftFeatureMirror * mirror,
    int committed) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size   = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = dw.n_embd;
    const int q_len  = dw.block_size;
    const int fc_in  = dw.n_target_layers * hidden;

    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, q_len, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    int mirror_slot0 = 0;
    if (mirror && draft_feature_mirror_can_view(*mirror, committed, ctx_len, mirror_slot0)) {
        const size_t stride = mirror->target_feat->nb[1];
        sg.target_hidden_cat = ggml_view_3d(
            sg.ctx,
            mirror->target_feat,
            fc_in, ctx_len, 1,
            stride,
            stride * (size_t)ctx_len,
            (size_t)mirror_slot0 * stride);
    } else {
        sg.target_hidden_cat = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, fc_in, ctx_len, 1);
        ggml_set_input(sg.target_hidden_cat);
    }
    ggml_set_name(sg.target_hidden_cat, "target_hidden_cat");

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, q_len);
    ggml_set_name(sg.positions, "positions_q");
    ggml_set_input(sg.positions);

    sg.positions_k = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, ctx_len + q_len);
    ggml_set_name(sg.positions_k, "positions_k");
    ggml_set_input(sg.positions_k);

    sg.gf = ggml_new_graph_custom(sg.ctx, 4096, false);

    DraftGraphInputs gi{};
    gi.ctx_len           = ctx_len;
    gi.noise_embed       = sg.inp_embed;
    gi.target_hidden_cat = sg.target_hidden_cat;
    gi.positions_q       = sg.positions;
    gi.positions_k       = sg.positions_k;
    gi.lm_head           = lm_head;
    DraftGraphOutputs go = build_draft_graph(sg.ctx, dw, gi);
    sg.hidden_states = go.hidden_states;
    sg.logits = go.logits;
    if (!sg.hidden_states) {
        std::fprintf(stderr, "draft graph missing hidden_states\n");
        return false;
    }
    if (sg.logits) {
        sg.argmax_tokens = ggml_argmax(sg.ctx, sg.logits);
        ggml_set_name(sg.argmax_tokens, "argmax_tokens");
        ggml_set_output(sg.argmax_tokens);
        ggml_build_forward_expand(sg.gf, sg.argmax_tokens);
    } else {
        ggml_set_output(sg.hidden_states);
        ggml_build_forward_expand(sg.gf, sg.hidden_states);
    }

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

}  // namespace dflash27b
