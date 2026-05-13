// Gemma4 runtime helpers — chunked-prefill graph builder, token embedders,
// DFlash draft graph wrappers, and AdaptiveDraftMax controller.
//
// Ported from feature/gemma4-support dflash/test/gemma4/test_gemma4_dflash.cpp
// (ranges 213-275, 280-450, 342-425, 525-612, 690-770 per the source digests).
// Namespace adjusted to dflash27b; local sampler types and driver-only
// scaffolding stripped.

#include "gemma4_runtime_helpers.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

namespace dflash27b {

// ─── step_graph_free ────────────────────────────────────────────────────────

void step_graph_free(StepGraph & sg) {
    if (sg.ctx) { ggml_free(sg.ctx); sg.ctx = nullptr; }
    if (sg.alloc) { ggml_gallocr_free(sg.alloc); sg.alloc = nullptr; }
    sg.gf        = nullptr;
    sg.inp_embed = nullptr;
    sg.positions = nullptr;
    sg.attn_mask = nullptr;
    sg.swa_mask  = nullptr;
    sg.logits    = nullptr;
}

// ─── build_causal_mask ──────────────────────────────────────────────────────

void build_causal_mask(std::vector<uint16_t> & out,
                       int kv_len, int n_tokens, int kv_start) {
    const int kv_pad = align_up(kv_len, g_kq_stride_pad);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    for (int q = 0; q < n_tokens; q++) {
        const int abs_q = kv_start + q;
        for (int k = 0; k <= abs_q && k < kv_len; k++) {
            out[(size_t)q * kv_pad + k] = F16_ZERO;
        }
    }
}

// ─── build_swa_causal_mask ──────────────────────────────────────────────────
//
// Non-monotonic ring mask.  The K view is always the full ring (ring_size slots,
// ring_win_start==0).  Slot k_view maps to absolute position via:
//   latest_slot = (kv_end - 1) % ring_size
//   offset_back = (latest_slot - k_view + ring_size) % ring_size
//   abs_k       = (kv_end - 1) - offset_back
//
// mask[q_idx][k_view_idx] = 0 (attend) iff:
//   abs_k >= (abs_q - swa_window + 1) AND abs_k <= abs_q AND abs_k >= 0
// else -inf.

void build_swa_causal_mask(std::vector<uint16_t> & out,
                            int kv_start,
                            int n_tokens,
                            int swa_window,
                            int ring_size,
                            int kv_end) {
    const int kv_pad = align_up(ring_size, g_kq_stride_pad);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    const int latest_slot = ((kv_end - 1) % ring_size + ring_size) % ring_size;
    for (int q = 0; q < n_tokens; q++) {
        const int abs_q = kv_start + q;
        const int q_lo  = std::max(0, abs_q - swa_window + 1);
        for (int k_view = 0; k_view < ring_size; k_view++) {
            const int offset_back = (latest_slot - k_view + ring_size) % ring_size;
            const int abs_k       = (kv_end - 1) - offset_back;
            const bool valid = (abs_k >= q_lo && abs_k <= abs_q && abs_k >= 0);
            if (valid) {
                out[(size_t)q * kv_pad + k_view] = F16_ZERO;
            }
        }
    }
}

// ─── build_gemma4_step ──────────────────────────────────────────────────────

bool build_gemma4_step(StepGraph & sg,
                       const GemmaTargetWeights & w,
                       GemmaTargetCache & cache,
                       ggml_backend_t backend,
                       int kv_start,
                       int n_tokens,
                       bool with_mask,
                       bool capture,
                       bool use_pflash,
                       float pflash_alpha,
                       int fa_window,
                       bool last_token_logits_only) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, w.n_embd, n_tokens, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(sg.positions, "positions");
    ggml_set_input(sg.positions);

    if (with_mask) {
        const int kv_len = kv_start + n_tokens;
        const int kv_pad = align_up(kv_len, g_kq_stride_pad);
        const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);

        sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
        ggml_set_name(sg.attn_mask, "attn_mask");
        ggml_set_input(sg.attn_mask);
        ggml_set_output(sg.attn_mask);  // force gallocr to allocate even if no op references it

        // SWA mask is required for every SWA dispatch — including single-token
        // decode (n_tokens==1). When swa_mask is null, gemma4_target_graph falls
        // back to attn_mask, which is sized for kv_len rather than the SWA window;
        // the resulting dimension mismatch lets FA read past the populated cache
        // region and corrupts attention. Catastrophic with TQ3_0 KV (it amplifies
        // uninitialized-cache noise into a fixed-point repetition loop), benign
        // but technically wrong with Q8_0 KV.
        const SwaView swa_view = compute_swa_view(kv_start, n_tokens,
                                                   w.swa_window, cache.swa_ctx_alloc);
        const int swa_kv_pad = align_up(swa_view.effective_win_len, g_kq_stride_pad);
        sg.swa_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, swa_kv_pad, q_pad);
        ggml_set_name(sg.swa_mask, "swa_mask");
        ggml_set_input(sg.swa_mask);
        ggml_set_output(sg.swa_mask);  // force gallocr to allocate even if no op references it
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    GemmaGraphInputs gi{};
    gi.inp_embed              = sg.inp_embed;
    gi.positions              = sg.positions;
    gi.attn_mask              = sg.attn_mask;
    gi.swa_mask               = sg.swa_mask;
    gi.n_tokens               = n_tokens;
    gi.kv_start               = kv_start;
    gi.capture_layers         = capture;
    gi.fa_window              = fa_window;
    gi.use_pflash             = use_pflash;
    gi.pflash_alpha           = pflash_alpha;
    gi.last_token_logits_only = last_token_logits_only;

    GemmaGraphOutputs go = build_gemma4_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    sg.logits = go.logits;
    ggml_set_output(sg.logits);
    ggml_build_forward_expand(sg.gf, sg.logits);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// ─── embed_token ────────────────────────────────────────────────────────────

bool embed_token(const GemmaTargetWeights & w,
                 int32_t tok,
                 ggml_tensor * inp_embed,
                 ggml_backend_t backend) {
    const int hidden = w.n_embd;
    std::vector<float> emb((size_t)hidden);
    if (!w.embedder.embed(&tok, 1, emb.data())) {
        std::fprintf(stderr, "[embed_token] failed for tok=%d\n", tok);
        return false;
    }
    ggml_backend_tensor_set(inp_embed, emb.data(), 0, sizeof(float) * hidden);
    (void)backend;
    return true;
}

// ─── embed_tokens_batch ─────────────────────────────────────────────────────

bool embed_tokens_batch(const GemmaTargetWeights & w,
                        const int32_t * ids,
                        int n,
                        ggml_tensor * inp_embed,
                        ggml_backend_t backend) {
    const int hidden = w.n_embd;
    std::vector<float> emb((size_t)hidden * n);
    if (!w.embedder.embed(ids, n, emb.data())) {
        std::fprintf(stderr, "[embed_batch] failed for %d tokens\n", n);
        return false;
    }
    ggml_backend_tensor_set(inp_embed, emb.data(), 0, sizeof(float) * hidden * n);
    (void)backend;
    return true;
}

// ─── draft_kv_prefill_create ────────────────────────────────────────────────
//
// Builds and allocates a DraftKVPrefillGraph for n_tokens positions.
// Wraps build_draft_kv_prefill_graph (internal.h:783).
// Ported from test_gemma4_dflash.cpp Range E (static build_draft_kv_prefill).

bool draft_kv_prefill_create(DraftKVPrefillGraph & pkg,
                             const GemmaDraftWeights & dw,
                             GemmaTargetCache & cache,
                             ggml_backend_t backend,
                             int n_tokens) {
    // Free any previous state
    if (pkg.alloc) { ggml_gallocr_free(pkg.alloc); pkg.alloc = nullptr; }
    if (pkg.ctx)   { ggml_free(pkg.ctx);   pkg.ctx   = nullptr; }
    pkg.gf          = nullptr;
    pkg.target_feat = nullptr;
    pkg.positions   = nullptr;

    const int target_feat_w = dw.n_target_layers * dw.target_hidden;

    ggml_init_params ip{};
    ip.mem_size   = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    pkg.ctx = ggml_init(ip);
    if (!pkg.ctx) return false;

    pkg.target_feat = ggml_new_tensor_2d(pkg.ctx, GGML_TYPE_F32, target_feat_w, n_tokens);
    ggml_set_name(pkg.target_feat, "prefill_target_feat");
    ggml_set_input(pkg.target_feat);

    pkg.positions = ggml_new_tensor_1d(pkg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(pkg.positions, "prefill_positions");
    ggml_set_input(pkg.positions);

    pkg.gf = ggml_new_graph_custom(pkg.ctx, 4096, false);

    build_draft_kv_prefill_graph(pkg.ctx, pkg.gf, dw, cache,
                                 pkg.target_feat, pkg.positions, n_tokens);

    if (!pkg.alloc) {
        pkg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!ggml_gallocr_alloc_graph(pkg.alloc, pkg.gf)) {
        ggml_gallocr_free(pkg.alloc); pkg.alloc = nullptr;
        ggml_free(pkg.ctx); pkg.ctx = nullptr;
        pkg.gf = nullptr; pkg.target_feat = nullptr; pkg.positions = nullptr;
        return false;
    }
    return true;
}

// ─── draft_kv_prefill_destroy ───────────────────────────────────────────────

void draft_kv_prefill_destroy(DraftKVPrefillGraph & pkg) {
    if (pkg.alloc) { ggml_gallocr_free(pkg.alloc); pkg.alloc = nullptr; }
    if (pkg.ctx)   { ggml_free(pkg.ctx);   pkg.ctx   = nullptr; }
    pkg.gf          = nullptr;
    pkg.target_feat = nullptr;
    pkg.positions   = nullptr;
}

// ─── draft_step_build ───────────────────────────────────────────────────────
//
// Builds and allocates a DraftStepGraph for q_len tokens at KV offset kv_start.
// Wraps build_gemma4_draft_graph (internal.h:790).
// Ported from test_gemma4_dflash.cpp Range E (static build_draft_step).

bool draft_step_build(DraftStepGraph & dsg,
                      const GemmaDraftWeights & dw,
                      GemmaTargetCache & cache,
                      ggml_backend_t backend,
                      int q_len,
                      int kv_start) {
    // Free any previous state
    if (dsg.alloc) { ggml_gallocr_free(dsg.alloc); dsg.alloc = nullptr; }
    if (dsg.ctx)   { ggml_free(dsg.ctx);   dsg.ctx   = nullptr; }
    dsg.gf          = nullptr;
    dsg.draft_embed = nullptr;
    dsg.positions   = nullptr;
    dsg.attn_mask   = nullptr;
    dsg.logits      = nullptr;

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    dsg.ctx = ggml_init(ip);
    if (!dsg.ctx) return false;

    const int kv_len  = kv_start + q_len;
    const int kv_pad  = align_up(kv_len, g_kq_stride_pad);
    const int q_pad   = align_up(q_len, KQ_MASK_PAD);

    dsg.draft_embed = ggml_new_tensor_2d(dsg.ctx, GGML_TYPE_F32, dw.n_embd, q_len);
    ggml_set_name(dsg.draft_embed, "draft_embed");
    ggml_set_input(dsg.draft_embed);

    dsg.positions = ggml_new_tensor_1d(dsg.ctx, GGML_TYPE_I32, q_len);
    ggml_set_name(dsg.positions, "draft_positions");
    ggml_set_input(dsg.positions);

    dsg.attn_mask = ggml_new_tensor_2d(dsg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
    ggml_set_name(dsg.attn_mask, "draft_attn_mask");
    ggml_set_input(dsg.attn_mask);
    ggml_set_output(dsg.attn_mask);

    dsg.gf = ggml_new_graph_custom(dsg.ctx, 4096, false);

    ggml_tensor * logits_out = build_gemma4_draft_graph(dsg.ctx, dsg.gf, dw, cache,
                                                         dsg.draft_embed, dsg.positions,
                                                         dsg.attn_mask, q_len, kv_start);
    if (!logits_out) {
        ggml_free(dsg.ctx); dsg.ctx = nullptr;
        return false;
    }
    dsg.logits = logits_out;
    ggml_set_output(dsg.logits);
    ggml_build_forward_expand(dsg.gf, dsg.logits);

    if (!dsg.alloc) {
        dsg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!ggml_gallocr_alloc_graph(dsg.alloc, dsg.gf)) {
        ggml_gallocr_free(dsg.alloc); dsg.alloc = nullptr;
        ggml_free(dsg.ctx); dsg.ctx = nullptr;
        dsg.gf = nullptr; dsg.draft_embed = nullptr;
        dsg.positions = nullptr; dsg.attn_mask = nullptr; dsg.logits = nullptr;
        return false;
    }
    return true;
}

// ─── draft_step_free ────────────────────────────────────────────────────────

void draft_step_free(DraftStepGraph & dsg) {
    if (dsg.alloc) { ggml_gallocr_free(dsg.alloc); dsg.alloc = nullptr; }
    if (dsg.ctx)   { ggml_free(dsg.ctx);   dsg.ctx   = nullptr; }
    dsg.gf          = nullptr;
    dsg.draft_embed = nullptr;
    dsg.positions   = nullptr;
    dsg.attn_mask   = nullptr;
    dsg.logits      = nullptr;
}

// copy_target_feat_bf16_to_f32: Copy n_tokens columns from the bf16 ring buffer
// src_bf16 (shape [target_feat_w, ring_cap]) starting at slot start_slot into
// the f32 tensor dst_f32 (shape [target_feat_w, n_tokens]).
//
// Uses ggml_cpy with ggml_view_2d for type conversion on the GPU backend —
// ring-wrap is handled with two graph copies (pre-wrap + post-wrap segments).
// On-device, no host roundtrip.

void copy_target_feat_bf16_to_f32(ggml_backend_t backend,
                                   ggml_tensor * src_bf16,
                                   ggml_tensor * dst_f32,
                                   int start_slot,
                                   int n_tokens,
                                   int target_feat_w) {
    const int cap    = (int)src_bf16->ne[1];
    const int pre_n  = std::min(n_tokens, cap - start_slot);
    const int post_n = n_tokens - pre_n;

    ggml_init_params ip{};
    ip.mem_size   = 256 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * tmp_ctx = ggml_init(ip);

    ggml_cgraph * gf = ggml_new_graph(tmp_ctx);

    // Pre-wrap segment: rows [start_slot .. start_slot+pre_n-1] → dst rows [0..pre_n-1]
    {
        ggml_tensor * s = ggml_view_2d(tmp_ctx, src_bf16, target_feat_w, pre_n,
                                       src_bf16->nb[1],
                                       (size_t)start_slot * src_bf16->nb[1]);
        ggml_tensor * d = ggml_view_2d(tmp_ctx, dst_f32, target_feat_w, pre_n,
                                       dst_f32->nb[1], 0);
        ggml_build_forward_expand(gf, ggml_cpy(tmp_ctx, s, d));
    }
    // Post-wrap segment: rows [0..post_n-1] → dst rows [pre_n..pre_n+post_n-1]
    if (post_n > 0) {
        ggml_tensor * s = ggml_view_2d(tmp_ctx, src_bf16, target_feat_w, post_n,
                                       src_bf16->nb[1], 0);
        ggml_tensor * d = ggml_view_2d(tmp_ctx, dst_f32, target_feat_w, post_n,
                                       dst_f32->nb[1],
                                       (size_t)pre_n * dst_f32->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(tmp_ctx, s, d));
    }

    ggml_backend_graph_compute(backend, gf);
    ggml_free(tmp_ctx);
}

}  // namespace dflash27b
