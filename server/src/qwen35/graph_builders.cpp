#include "graph_builders.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cstdio>

namespace dflash::common {

// ── build_layer_step ────────────────────────────────────────────

bool build_layer_step(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int layer_idx,
    ggml_tensor * act_in,
    ggml_tensor * act_out,
    int chunk_start,
    int n_tokens,
    int kv_start,
    bool with_mask,
    bool capture,
    int fa_window,
    int kq_stride_pad,
    bool kvflash,
    bool tree_mode) {
    if (kvflash) with_mask = true;
    step_graph_free(sg);

    const bool is_attn = (((layer_idx + 1) % w.full_attention_interval) == 0);

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = w.n_embd;

    sg.inp_embed = ggml_view_2d(sg.ctx, act_in,
        hidden, n_tokens,
        act_in->nb[1], (size_t)chunk_start * act_in->nb[1]);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    if (is_attn) {
        sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
        ggml_set_name(sg.positions, "positions");
        ggml_set_input(sg.positions);

        if (with_mask) {
            int phys_ctx = cache.max_ctx;
            if (kvflash) {
                for (ggml_tensor * t : cache.attn_k) {
                    if (t) { phys_ctx = std::min(phys_ctx, (int)t->ne[1]); break; }
                }
            }
            // Size from the fixed physical capacity so gallocr doesn't grow
            // as kv_start advances. Under kvflash this is the resident pool.
            const int max_win_len = phys_ctx + n_tokens;
            const int kv_pad = align_up(max_win_len, kq_stride_pad);
            const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
            sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
            ggml_set_name(sg.attn_mask, "attn_mask");
            ggml_set_input(sg.attn_mask);
        }
        if (kvflash) {
            sg.kv_write_rows = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_I64,
                                                  n_tokens, w.n_head_kv);
            ggml_set_name(sg.kv_write_rows, "kv_write_rows");
            ggml_set_input(sg.kv_write_rows);
        }
    }

    if (tree_mode && !is_attn) {
        sg.parent_ids = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_name(sg.parent_ids, "parent_ids");
        ggml_set_input(sg.parent_ids);
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    ggml_tensor * layer_out = build_qwen35_layer(
        sg.ctx, sg.gf, w, cache, layer_idx,
        sg.inp_embed, sg.positions, sg.attn_mask,
        kv_start, n_tokens, capture, fa_window,
        /*q_tail_capture=*/nullptr, /*q_tail_start=*/0,
        sg.kv_write_rows, sg.parent_ids);
    if (!layer_out) return false;

    ggml_tensor * out_view = ggml_view_2d(sg.ctx, act_out,
        hidden, n_tokens,
        act_out->nb[1], (size_t)chunk_start * act_out->nb[1]);
    ggml_build_forward_expand(sg.gf, ggml_cpy(sg.ctx, layer_out, out_view));

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

bool build_layer_prefn_step(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int layer_idx,
    int kv_start,
    int n_tokens,
    bool with_mask,
    int fa_window,
    int kq_stride_pad,
    bool kvflash) {
    if (kvflash) with_mask = true;   // slot-space masking is mandatory on the pool
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = w.n_embd;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    const bool is_attn = (((layer_idx + 1) % w.full_attention_interval) == 0);
    if (is_attn) {
        sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
        ggml_set_name(sg.positions, "positions");
        ggml_set_input(sg.positions);
        if (with_mask) {
            // Mask width follows the PHYSICAL tensor capacity (pool-sized
            // under kvflash) so it agrees with the FA span clamp inside
            // build_full_attn_block.
            int phys_ctx = cache.max_ctx;
            for (ggml_tensor * t : cache.attn_k) {
                if (t) { phys_ctx = std::min(phys_ctx, (int)t->ne[1]); break; }
            }
            const int max_win_len = phys_ctx + n_tokens;
            const int kv_pad = align_up(max_win_len, kq_stride_pad);
            const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
            sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
            ggml_set_name(sg.attn_mask, "attn_mask");
            ggml_set_input(sg.attn_mask);
        }
        if (kvflash) {
            sg.kv_write_rows = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_I64,
                                                  n_tokens, w.n_head_kv);
            ggml_set_name(sg.kv_write_rows, "kv_write_rows");
            ggml_set_input(sg.kv_write_rows);
        }
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);
    QwenLayerPrefnOutputs go = build_qwen35_layer_prefn(
        sg.ctx, sg.gf, w, cache, layer_idx,
        sg.inp_embed, sg.positions, sg.attn_mask,
        kv_start, n_tokens, fa_window,
        sg.kv_write_rows,
        /*skip_gdn_intermediate=*/true);
    if (!go.residual || !go.post) return false;
    sg.ffn_residual = go.residual;
    sg.ffn_post = go.post;
    sg.moe_weights = go.moe_weights;
    if (go.moe_selected) {
        sg.moe_selected.assign((size_t)w.n_layer, nullptr);
        sg.moe_selected[(size_t)layer_idx] = go.moe_selected;
        ggml_set_output(go.moe_selected);
        ggml_build_forward_expand(sg.gf, go.moe_selected);
    }
    if (go.moe_weights) {
        ggml_set_output(go.moe_weights);
        ggml_build_forward_expand(sg.gf, go.moe_weights);
    }
    ggml_set_output(go.residual);
    ggml_build_forward_expand(sg.gf, go.residual);
    ggml_set_output(go.post);
    ggml_build_forward_expand(sg.gf, go.post);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// Full-layer graph for hybrid decode: pre-FFN (attention/DeltaNet + router) +
// MoE FFN (all selected experts via ggml_mul_mat_id) + shared FFN + residual.
// Outputs: sg.logits = layer_output, sg.moe_selected[layer_idx] = router picks.
// This is 1 graph compute per layer instead of 2 (pre-FFN + fused hot+shared).
bool build_hybrid_full_layer_step(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int layer_idx,
    int kv_start,
    int n_tokens,
    bool with_mask,
    int fa_window,
    int kq_stride_pad) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = w.n_embd;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    const bool is_attn = (((layer_idx + 1) % w.full_attention_interval) == 0);
    if (is_attn) {
        sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
        ggml_set_name(sg.positions, "positions");
        ggml_set_input(sg.positions);
        if (with_mask) {
            const int max_win_len = cache.max_ctx + n_tokens;
            const int kv_pad = align_up(max_win_len, kq_stride_pad);
            const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
            sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
            ggml_set_name(sg.attn_mask, "attn_mask");
            ggml_set_input(sg.attn_mask);
        }
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    ggml_tensor * moe_selected = nullptr;
    ggml_tensor * layer_out = build_qwen35_layer(
        sg.ctx, sg.gf, w, cache, layer_idx,
        sg.inp_embed, sg.positions, sg.attn_mask,
        kv_start, n_tokens, /*capture=*/false, fa_window,
        /*q_tail_capture=*/nullptr, /*q_tail_start=*/0,
        &moe_selected);
    if (!layer_out) return false;

    // Use hidden_input as the layer output tensor (repurpose field)
    sg.hidden_input = layer_out;
    ggml_set_output(layer_out);
    ggml_build_forward_expand(sg.gf, layer_out);

    if (moe_selected) {
        sg.moe_selected.assign((size_t)w.n_layer, nullptr);
        sg.moe_selected[(size_t)layer_idx] = moe_selected;
        ggml_set_output(moe_selected);
        ggml_build_forward_expand(sg.gf, moe_selected);
    }

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// ── build_target_step ───────────────────────────────────────────

bool build_target_step(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start,
    int n_tokens,
    bool with_mask,
    bool capture,
    bool capture_delta_intermediate,
    int fa_window,
    bool last_token_logits_only,
    int kq_stride_pad,
    bool capture_moe_router,
    bool kvflash_mask,
    bool capture_qk) {
    step_graph_free(sg);

    // Persistent thread_local arena: rebuilt step graphs land at identical
    // addresses, keeping the ggml-cuda CUDA-graph cache key (nodes[0]) and
    // every node property stable across AR decode steps -> captured graph
    // replays instead of re-launching every kernel. Pairs with the
    // step-invariant set_rows KV write (kv_write_rows) below.
    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    static thread_local std::vector<uint8_t> g_step_arena;
    if (g_step_arena.size() < ip.mem_size) g_step_arena.resize(ip.mem_size);
    ip.mem_buffer = g_step_arena.data();
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = w.n_embd;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(sg.positions, "positions");
    ggml_set_input(sg.positions);

    if (with_mask) {
        // Use max_ctx for mask allocation so the gallocr buffer never needs to
        // grow as kv_start increases during generation.  The actual mask is
        // filled only up to kv_start + n_tokens; the excess is don't-care.
        // kvflash mode: the physical span is the (smaller) pool capacity of
        // the attention tensors, so size the mask from those instead.
        int phys_ctx = cache.max_ctx;
        for (auto * t : cache.attn_k) {
            if (t) { phys_ctx = std::min(phys_ctx, (int)t->ne[1]); break; }
        }
        const int max_win_len = phys_ctx + n_tokens;
        const int kv_pad = align_up(max_win_len, kq_stride_pad);
        const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
        sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
        ggml_set_name(sg.attn_mask, "attn_mask");
        ggml_set_input(sg.attn_mask);
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    // Step-invariant KV write: only when topology can't vary per step.
    // DFLASH_QWEN35_NO_KVPAD=1 restores the legacy cpy append + exact-length
    // FA span (per-step node properties -> no CUDA-graph replay).
    static const bool g_no_kvpad = (std::getenv("DFLASH_QWEN35_NO_KVPAD") != nullptr);
    // kvflash_mask: kvflash mode. The mask carries pool slot validity
    // (uploaded by the caller before EVERY compute — the input's buffer
    // region is reused by graph execution) and set_rows carries per-token
    // physical slots, so the slot-mapped write stays active for masked,
    // multi-token, and feature-capturing forwards (decode AND spec verify).
    const bool use_kv_write_rows =
        !g_no_kvpad && !capture_delta_intermediate &&
        (kvflash_mask
             ? (fa_window == 0)
             : (n_tokens == 1 && fa_window == 0 && !with_mask && !capture));
    if (use_kv_write_rows) {
        sg.kv_write_rows = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_I64,
                                              n_tokens, w.n_head_kv);
        ggml_set_name(sg.kv_write_rows, "kv_write_rows");
        ggml_set_input(sg.kv_write_rows);
    }

    QwenGraphInputs gi{};
    gi.inp_embed                  = sg.inp_embed;
    gi.positions                  = sg.positions;
    gi.attn_mask                  = sg.attn_mask;
    gi.n_tokens                   = n_tokens;
    gi.kv_start                   = kv_start;
    gi.capture_layers             = capture;
    gi.capture_delta_intermediate = capture_delta_intermediate;
    gi.capture_moe_router         = capture_moe_router;
    gi.fa_window                  = fa_window;
    gi.last_token_logits_only     = last_token_logits_only;
    gi.kv_write_rows              = sg.kv_write_rows;
    gi.q_capture                  = capture_qk;

    QwenGraphOutputs go = build_qwen35_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    sg.logits = go.logits;
    sg.delta_captures = std::move(go.delta_captures);
    sg.moe_selected = std::move(go.moe_selected);
    ggml_set_output(sg.logits);

    sg.argmax_tokens = ggml_argmax(sg.ctx, sg.logits);
    ggml_set_name(sg.argmax_tokens, "chain_verify_argmax");
    ggml_set_output(sg.argmax_tokens);
    ggml_build_forward_expand(sg.gf, sg.argmax_tokens);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// ── build_target_step_tree ──────────────────────────────────────

bool build_target_step_tree(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start,
    int n_tokens,
    int fa_window,
    int kq_stride_pad) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = w.n_embd;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(sg.positions, "positions");
    ggml_set_input(sg.positions);

    const int max_win_len = cache.max_ctx + n_tokens;
    const int kv_pad = align_up(max_win_len, kq_stride_pad);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
    ggml_set_name(sg.attn_mask, "attn_mask");
    ggml_set_input(sg.attn_mask);

    sg.parent_ids = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(sg.parent_ids, "parent_ids");
    ggml_set_input(sg.parent_ids);

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    QwenGraphInputs gi{};
    gi.inp_embed                  = sg.inp_embed;
    gi.positions                  = sg.positions;
    gi.attn_mask                  = sg.attn_mask;
    gi.n_tokens                   = n_tokens;
    gi.kv_start                   = kv_start;
    gi.fa_window                  = fa_window;
    gi.capture_layers             = true;
    gi.capture_delta_intermediate = true;
    gi.parent_ids                 = sg.parent_ids;

    QwenGraphOutputs go = build_qwen35_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    sg.logits = go.logits;
    sg.delta_captures = std::move(go.delta_captures);
    ggml_set_output(sg.logits);

    sg.argmax_tokens = ggml_argmax(sg.ctx, sg.logits);
    ggml_set_name(sg.argmax_tokens, "tree_verify_argmax");
    ggml_set_output(sg.argmax_tokens);
    ggml_build_forward_expand(sg.gf, sg.argmax_tokens);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}


// ── build_lm_head_projection_step ───────────────────────────────

bool build_lm_head_projection_step(
    StepGraph & sg,
    const TargetWeights & w,
    ggml_backend_t backend,
    int n_tokens) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size   = 64 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = w.n_embd;
    sg.hidden_input = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_set_name(sg.hidden_input, "draft_hidden_for_lm_head");
    ggml_set_input(sg.hidden_input);

    sg.gf = ggml_new_graph_custom(sg.ctx, 1024, false);
    sg.logits = ggml_mul_mat(sg.ctx, w.output, sg.hidden_input);
    ggml_set_name(sg.logits, "draft_projected_logits");
    ggml_set_output(sg.logits);
    sg.argmax_tokens = ggml_argmax(sg.ctx, sg.logits);
    ggml_set_name(sg.argmax_tokens, "draft_projected_argmax");
    ggml_set_output(sg.argmax_tokens);
    ggml_build_forward_expand(sg.gf, sg.argmax_tokens);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

}  // namespace dflash::common
