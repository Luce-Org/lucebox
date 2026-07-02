#include "dflash_draft_graph.h"
#include "draft/draft_graph.h"  // DraftGraphInputs, DraftGraphOutputs, build_draft_graph

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace dflash::common {

// Minimum alignment required by ggml flash_attn_ext for mask rows.
static constexpr int MASK_KV_PAD = 32;

static inline int mask_align_up(int x, int a) { return ((x + a - 1) / a) * a; }

// Check whether any layer in the draft is SWA.
static bool draft_has_swa_layers(const DraftWeights & dw) {
    for (int i = 0; i < dw.n_layer; i++)
        if (dw.layers[i].is_swa) return true;
    return false;
}

// Build draft graph at a given ctx_len into sg. Does NOT touch sg.alloc.
// mirror_view: if true, uses a view into mirror->target_feat at slot0.
// ctx_alloc: allocation/topology size of the ctx dimension (>= ctx_len).
// When ctx_alloc > 0 and differs from the legacy behavior, a full-layer pad
// mask input is created so the graph topology stays stable while ctx_len
// grows (CUDA-graph replay for the draft forward).
static bool build_draft_graph_internal(
    StepGraph & sg,
    const DraftWeights & dw,
    ggml_tensor * lm_head,
    int ctx_len,
    const DraftFeatureMirror * mirror,
    int mirror_slot0,
    bool mirror_view,
    bool pad_masked = false) {

    const size_t arena_sz = 32u * 1024 * 1024;
    if (sg.meta_arena.size() < arena_sz) sg.meta_arena.resize(arena_sz);
    ggml_init_params ip{};
    ip.mem_size   = sg.meta_arena.size();
    ip.mem_buffer = sg.meta_arena.data();
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = dw.n_embd;
    const int q_len  = dw.block_size;
    const int fc_in  = dw.n_target_layers * hidden;

    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, q_len, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    if (mirror_view) {
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

    // Causal mask for SWA layers (if any).
    // Shape: [kv_pad, q_len] F16 (directly, no cast needed — matches attn_masks.h pattern).
    sg.attn_mask = nullptr;
    const bool has_swa = draft_has_swa_layers(dw);
    if (has_swa) {
        // SWA layers' effective KV length (windowed or full ctx)
        const bool swa_active = dw.swa_window > 0 && ctx_len > dw.swa_window;
        const int eff_ctx = swa_active ? dw.swa_window : ctx_len;
        const int eff_total_k = eff_ctx + q_len;
        const int kv_pad = mask_align_up(eff_total_k, MASK_KV_PAD);
        sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_len);
        ggml_set_name(sg.attn_mask, "causal_mask_swa");
        ggml_set_input(sg.attn_mask);
    }

    bool any_full_layer = false;
    for (int i = 0; i < dw.n_layer; i++)
        if (!dw.layers[i].is_swa) { any_full_layer = true; break; }
    sg.pad_mask_full = nullptr;
    if (pad_masked && any_full_layer) {
        const int total_k = ctx_len + q_len;
        const int kv_pad = mask_align_up(total_k, MASK_KV_PAD);
        sg.pad_mask_full = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_len);
        ggml_set_name(sg.pad_mask_full, "pad_mask_full");
        ggml_set_input(sg.pad_mask_full);
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 4096, false);

    DraftGraphInputs gi{};
    gi.ctx_len           = ctx_len;
    gi.noise_embed       = sg.inp_embed;
    gi.target_hidden_cat = sg.target_hidden_cat;
    gi.positions_q       = sg.positions;
    gi.positions_k       = sg.positions_k;
    gi.lm_head           = lm_head;
    gi.causal_mask_swa   = sg.attn_mask;
    gi.pad_mask_full     = sg.pad_mask_full;
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
    return true;
}

bool build_draft_step(
    StepGraph & sg,
    const DraftWeights & dw,
    ggml_tensor * lm_head,
    ggml_backend_t backend,
    int ctx_len,
    const DraftFeatureMirror * mirror,
    int committed,
    int /*ctx_len_max*/,
    bool pad_ctx) {
    step_graph_free(sg);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }

    // Padded-ctx mode: build the graph at the 64-aligned ctx size and mask the
    // pad keys, so the topology (and gallocr layout) stays IDENTICAL across
    // ~64 tokens of context growth and ggml-cuda can replay the draft forward
    // as a CUDA graph. Requires masking, so only usable when the draft has no
    // SWA layers (SWA windowing would slide into the pad region).
    // Padding is safe as long as no layer does actual SWA WINDOWING at this
    // context size (windowing would slide the K view into the pad region).
    // Layers flagged is_swa below the window size just mean "causal noise
    // mask" and pad fine with the pad rows masked out.
    const int ctx_pad_cand = (ctx_len + 63) & ~63;
    const bool swa_windowing = draft_has_swa_layers(dw) && dw.swa_window > 0 &&
                               ctx_pad_cand > dw.swa_window;
    const bool do_pad = pad_ctx && !swa_windowing;
    const int ctx_alloc = do_pad ? ctx_pad_cand : ctx_len;
    static bool s_pad_logged = false;
    if (!s_pad_logged) {
        s_pad_logged = true;
        std::fprintf(stderr, "[draft-pad] pad_ctx=%d has_swa=%d do_pad=%d ctx_len=%d ctx_alloc=%d\n",
                     (int)pad_ctx, (int)draft_has_swa_layers(dw), (int)do_pad, ctx_len, ctx_alloc);
    }

    int mirror_slot0 = 0;
    bool use_view = mirror &&
        draft_feature_mirror_can_view(*mirror, committed, ctx_len, mirror_slot0);
    if (use_view && do_pad &&
        mirror_slot0 + ctx_alloc > mirror->cap) {
        use_view = false;  // padded view would run past the ring
    }

    // If ctx_len exceeds our cached reserve, re-reserve at next 64 boundary.
    // This makes all subsequent alloc_graph calls within the 64-token window
    // a no-op (no CUDA free+alloc).
    const int ctx_padded = (ctx_len + 63) & ~63;
    if (ctx_padded > sg.alloc_reserved_ctx) {
        // Build a dummy graph at ctx_padded just for sizing.
        // Use non-view path for reserve (view tensors don't need allocation).
        if (!build_draft_graph_internal(sg, dw, lm_head, ctx_padded,
                                        nullptr, 0, false, do_pad)) {
            return false;
        }
        ggml_gallocr_reserve(sg.alloc, sg.gf);
        sg.alloc_reserved_ctx = ctx_padded;
        step_graph_free(sg);
    }

    // Build real graph. Padded mode: topology at ctx_alloc, real rows = ctx_len.
    if (!build_draft_graph_internal(sg, dw, lm_head,
                                    do_pad ? ctx_alloc : ctx_len,
                                    mirror, mirror_slot0, use_view, do_pad)) {
        return false;
    }
    sg.ctx_alloc = do_pad ? ctx_alloc : 0;

    if (!ggml_gallocr_alloc_graph(sg.alloc, sg.gf)) {
        return false;
    }

    if (do_pad) {
        const int q_len = dw.block_size;
        const int total_k = ctx_alloc + q_len;
        const int kv_pad = mask_align_up(total_k, MASK_KV_PAD);
        // Full-layer mask: real ctx keys + all noise keys visible (the DFlash
        // block is non-causal on full layers), pad keys and alignment columns
        // -inf.
        static constexpr uint16_t ZERO = 0x0000;
        static constexpr uint16_t NEG_INF = 0xFC00;
        std::vector<uint16_t> mask_data((size_t)kv_pad * q_len, NEG_INF);
        for (int q = 0; q < q_len; q++) {
            for (int k = 0; k < ctx_len; k++)
                mask_data[(size_t)q * kv_pad + k] = ZERO;
            for (int j = 0; j < q_len; j++)
                mask_data[(size_t)q * kv_pad + (ctx_alloc + j)] = ZERO;
        }
        if (sg.pad_mask_full) {
            ggml_backend_tensor_set(sg.pad_mask_full, mask_data.data(), 0,
                                    sizeof(uint16_t) * mask_data.size());
        }

        // The pad rows of the ctx features must be FINITE (they are masked
        // out, but NaN/Inf would still poison flash-attn). Zero them.
        if (ctx_alloc > ctx_len) {
            if (use_view) {
                const size_t row_bytes = (size_t)mirror->target_feat->nb[1];
                std::vector<uint8_t> zeros((size_t)(ctx_alloc - ctx_len) * row_bytes, 0);
                ggml_backend_tensor_set(mirror->target_feat, zeros.data(),
                                        (size_t)(mirror_slot0 + ctx_len) * row_bytes,
                                        zeros.size());
            } else {
                const size_t row_bytes = (size_t)sg.target_hidden_cat->nb[1];
                std::vector<uint8_t> zeros((size_t)(ctx_alloc - ctx_len) * row_bytes, 0);
                ggml_backend_tensor_set(sg.target_hidden_cat, zeros.data(),
                                        (size_t)ctx_len * row_bytes,
                                        zeros.size());
            }
        }
    }

    // Fill causal mask data for SWA layers (after allocation gives memory to the tensor).
    if (sg.attn_mask) {
        const int q_len = dw.block_size;
        const bool swa_active = !do_pad && dw.swa_window > 0 && ctx_len > dw.swa_window;
        // Padded mode: keys span ctx_alloc rows; only the first ctx_len are
        // real (visible), the pad rows stay -inf. Noise keys sit at ctx_alloc.
        const int eff_ctx = do_pad ? ctx_alloc : (swa_active ? dw.swa_window : ctx_len);
        const int vis_ctx = do_pad ? ctx_len : eff_ctx;
        const int eff_total_k = eff_ctx + q_len;
        const int kv_pad = mask_align_up(eff_total_k, MASK_KV_PAD);

        // Build causal mask in F16 directly (same pattern as attn_masks.h):
        // Context keys (k < eff_ctx): always visible.
        // Noise keys (k = eff_ctx + j): visible if j <= q (causal).
        static constexpr uint16_t ZERO = 0x0000;
        static constexpr uint16_t NEG_INF = 0xFC00;
        std::vector<uint16_t> mask_data((size_t)kv_pad * q_len, NEG_INF);
        for (int q = 0; q < q_len; q++) {
            for (int k = 0; k < vis_ctx; k++)
                mask_data[(size_t)q * kv_pad + k] = ZERO;
            for (int j = 0; j <= q; j++)
                mask_data[(size_t)q * kv_pad + (eff_ctx + j)] = ZERO;
        }
        ggml_backend_tensor_set(sg.attn_mask, mask_data.data(), 0,
                                sizeof(uint16_t) * mask_data.size());
    }

    return true;
}

}  // namespace dflash::common
