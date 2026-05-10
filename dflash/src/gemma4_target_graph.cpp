// Forward pass of Gemma4 (pure attention) in pure ggml.
//
// Supports both Gemma4-31B (dense, 60 layers) and Gemma4-26B-A4B (MoE, 30 layers).
// All model dimensions are read from GGUF at load time via GemmaTargetWeights.
// No llama.cpp runtime is linked — only ggml ops.
//
// Architecture highlights:
//   - ALL layers are attention (no DeltaNet/SSM) — simpler than Qwen3.5 hybrid
//   - Two layer types interleaved per swa_layers[]:
//       SWA (sliding window): standard RoPE (rope_theta_swa), windowed FA
//       Full (global):        proportional RoPE via per-layer rope_freqs, full FA
//   - Attention scale = 1.0 (self.scaling = 1.0, not 1/sqrt(head_dim))
//   - Logit softcapping: output = softcap * tanh(output / softcap), softcap=30
//   - Per-Layer Embeddings (PLE): gated embedding added to residual each layer
//   - Shared KV cache: some layers reuse an earlier layer's KV slot
//   - MoE FFN (26B-A4B): shared_expert + routed experts (top-K)
//
// State (persisted in GemmaTargetCache across calls):
//   - attn_k, attn_v   : KV cache for non-shared KV layers
//   - layer_to_kv_idx  : maps layer index -> KV slot index (-1 = shared)
//   - layer_to_donor_kv: maps layer index -> donor slot for shared layers

#include "internal.h"
#include "kv_quant.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace dflash27b {

// ─── File-local constants ────────────────────────────────────────────────────

static constexpr float EPS = GEMMA4_RMS_EPS;

// ─── Helpers ─────────────────────────────────────────────────────────────────

static ggml_tensor * rms_norm_mul(ggml_context * ctx, ggml_tensor * x,
                                  ggml_tensor * weight, float eps) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, weight);
}

// GeGLU FFN: w_down @ (gelu(w_gate @ x) * (w_up @ x))
static ggml_tensor * build_geglu_ffn(ggml_context * ctx,
                                     ggml_tensor * cur,
                                     const GemmaTargetLayer & L) {
    ggml_tensor * gate = ggml_mul_mat(ctx, L.w_gate, cur);
    ggml_tensor * up   = ggml_mul_mat(ctx, L.w_up,   cur);
    ggml_tensor * gu   = ggml_geglu_split(ctx, gate, up);
    return ggml_mul_mat(ctx, L.w_down, gu);
}

// MoE FFN — shared expert + softmax-gated routed experts.
// Matches Gemma4-26B-A4B architecture:
//   shared_out  = w_down @ (gelu(w_gate @ x) * (w_up @ x))
//   shared_out  = rms_norm(shared_out) * ffn_post_norm_1
//   router_in   = rms_norm(inpSA) / sqrt(n_embd) * ffn_gate_inp_s  (bare rms_norm)
//   logits      = ffn_gate_inp @ router_in             [n_expert, n_tokens]
//   probs       = softmax(logits)
//   top_ids     = argsort_top_k(probs, n_expert_used)  [n_expert_used, n_tokens] i32
//   weights     = get_rows(probs, top_ids)             [1, n_expert_used, n_tokens]
//   weights     = weights / sum(weights)               (normalize to 1.0)
//   gate_up_out = mul_mat_id(ffn_gate_up_exps, x, top_ids) → gelu+mul → weighted
//   expert_out  = mul_mat_id(ffn_down_exps, act, top_ids) [n_embd, n_expert_used, n_tokens]
//   expert_out  = sum over expert dim                  [n_embd, n_tokens]
//   expert_out  = rms_norm(expert_out) * ffn_post_norm_2
//   result      = shared_out + expert_out
static ggml_tensor * build_moe_ffn(ggml_context * ctx,
                                   ggml_cgraph *  gf,
                                   const GemmaTargetWeights & w,
                                   const GemmaTargetLayer & L,
                                   ggml_tensor * cur_shared_ffn,
                                   ggml_tensor * cur_moe_ffn,
                                   ggml_tensor * cur_for_router,
                                   int n_tokens) {
    const int n_embd        = w.n_embd;
    const int n_expert_used = w.n_expert_used;
    const int n_expert      = w.n_expert;
    const int n_ff_exp      = w.n_ff_exp;

    // ── Shared expert (always active) ──────────────────────────────────────────
    ggml_tensor * shared_out = nullptr;
    if (L.w_gate && L.w_up && L.w_down) {
        ggml_tensor * sg  = ggml_mul_mat(ctx, L.w_gate, cur_shared_ffn);
        ggml_tensor * su  = ggml_mul_mat(ctx, L.w_up,   cur_shared_ffn);
        ggml_tensor * sgu = ggml_geglu_split(ctx, sg, su);
        shared_out = ggml_mul_mat(ctx, L.w_down, sgu);
        if (L.ffn_post_norm_1) {
            shared_out = rms_norm_mul(ctx, shared_out, L.ffn_post_norm_1, EPS);
        }
    }

    // ── Router ─────────────────────────────────────────────────────────────────
    // router_in = rms_norm(inpSA) / sqrt(n_embd) * ffn_gate_inp_s (bare rms_norm, no weight)
    ggml_tensor * router_in = ggml_rms_norm(ctx, cur_for_router, EPS);
    router_in = ggml_scale(ctx, router_in, 1.0f / std::sqrt((float)n_embd));
    if (L.ffn_gate_inp_s) {
        router_in = ggml_mul(ctx, router_in, L.ffn_gate_inp_s);
    }
    // logits: [n_expert, n_tokens]
    ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, router_in);

    // Softmax gating
    ggml_tensor * probs = ggml_soft_max(ctx, logits);  // [n_expert, n_tokens]

    // Top-K selection — returns i32 index tensor [n_expert_used, n_tokens]
    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx, probs, n_expert_used);

    // Routing weights: gather probs at selected indices [1, n_expert_used, n_tokens]
    ggml_tensor * probs_3d   = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);
    ggml_tensor * weights    = ggml_get_rows(ctx, probs_3d, selected_experts);
    // weights: [1, n_expert_used, n_tokens] → normalize to sum=1.0
    {
        ggml_tensor * w2d = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);
        ggml_tensor * wsum = ggml_sum_rows(ctx, w2d);
        wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
        w2d = ggml_div(ctx, w2d, wsum);
        weights = ggml_reshape_3d(ctx, w2d, 1, n_expert_used, n_tokens);
    }

    // ── Routed experts via ggml_mul_mat_id ─────────────────────────────────────
    ggml_tensor * expert_out = nullptr;
    if (L.ffn_gate_up_exps && L.ffn_down_exps) {
        // cur_moe_ffn is [n_embd, n_tokens]; mul_mat_id expects [n_embd, 1, n_tokens]
        ggml_tensor * x = ggml_reshape_3d(ctx, cur_moe_ffn, n_embd, 1, n_tokens);

        // Gate+up projection: ffn_gate_up_exps [2*n_ff_exp, n_embd, n_expert]
        // Result: [2*n_ff_exp, n_expert_used, n_tokens]
        ggml_tensor * gate_up = ggml_mul_mat_id(ctx, L.ffn_gate_up_exps,
                                                x, selected_experts);

        const size_t elt = ggml_element_size(gate_up);
        // gate half: first n_ff_exp rows
        ggml_tensor * g_half = ggml_view_3d(ctx, gate_up,
            n_ff_exp, n_expert_used, n_tokens,
            (size_t)n_ff_exp * 2 * elt,
            (size_t)n_ff_exp * 2 * n_expert_used * elt,
            0);
        // up half: second n_ff_exp rows
        ggml_tensor * u_half = ggml_view_3d(ctx, gate_up,
            n_ff_exp, n_expert_used, n_tokens,
            (size_t)n_ff_exp * 2 * elt,
            (size_t)n_ff_exp * 2 * n_expert_used * elt,
            (size_t)n_ff_exp * elt);

        // GeGLU activation (views are non-contiguous; ggml_gelu requires contiguous)
        g_half = ggml_cont(ctx, g_half);
        u_half = ggml_cont(ctx, u_half);
        ggml_tensor * activated = ggml_mul(ctx, ggml_gelu(ctx, g_half), u_half);

        // Scale by routing weights [1, n_expert_used, n_tokens]
        activated = ggml_mul(ctx, activated, weights);

        // Down projection: ffn_down_exps [n_embd, n_ff_exp, n_expert]
        // activated: [n_ff_exp, n_expert_used, n_tokens]
        ggml_tensor * down_out = ggml_mul_mat_id(ctx, L.ffn_down_exps,
                                                  activated, selected_experts);
        // down_out: [n_embd, n_expert_used, n_tokens]

        // Optional down-projection scale (ffn_down_exps_s is a per-column scale)
        if (L.ffn_down_exps_s) {
            down_out = ggml_mul(ctx, down_out, L.ffn_down_exps_s);
        }

        // Sum over n_expert_used to get [n_embd, n_tokens].
        // down_out: [n_embd, n_expert_used, n_tokens]
        // Use the proven llama.cpp pattern: ggml_build_forward_expand the full
        // tensor then sum slice views with ggml_add in a loop over n_expert_used.
        ggml_build_forward_expand(gf, down_out);
        expert_out = ggml_view_2d(ctx, down_out,
                                   n_embd, n_tokens,
                                   down_out->nb[2],
                                   0);
        ggml_build_forward_expand(gf, expert_out);
        for (int ei = 1; ei < n_expert_used; ++ei) {
            ggml_tensor * slice = ggml_view_2d(ctx, down_out,
                                               n_embd, n_tokens,
                                               down_out->nb[2],
                                               (size_t)ei * down_out->nb[1]);
            ggml_build_forward_expand(gf, slice);
            expert_out = ggml_add(ctx, expert_out, slice);
            ggml_build_forward_expand(gf, expert_out);
        }

        if (L.ffn_post_norm_2) {
            expert_out = rms_norm_mul(ctx, expert_out, L.ffn_post_norm_2, EPS);
        }
    }

    // ── Combine shared + routed experts ────────────────────────────────────────
    if (shared_out && expert_out) {
        return ggml_add(ctx, shared_out, expert_out);
    } else if (shared_out) {
        return shared_out;
    } else if (expert_out) {
        return expert_out;
    }
    // Fallback: should not happen with a correctly loaded MoE model
    return cur_shared_ffn;
}

// ─── SWA view geometry helper ────────────────────────────────────────────────
//
// Compute the (abs_win_start, effective_win_len, ring_win_start) triple for a
// chunk at position kv_start with n_tokens query tokens, given swa_window and
// the ring-buffer size (swa_ctx_alloc).  This is the single source of truth for
// the K/V view passed to FA and for the host-side causal mask.
SwaView compute_swa_view(int kv_start, int n_tokens,
                          int swa_window, int swa_ctx_alloc)
{
    SwaView v;
    v.abs_win_start    = (swa_window > 0 && kv_start > swa_window)
                          ? (kv_start - swa_window) : 0;
    // K view is ALWAYS the full ring; the host-built mask handles the
    // non-monotonic ring layout via abs_pos(slot) computation.
    v.effective_win_len = swa_ctx_alloc;
    v.ring_win_start    = 0;
    return v;
}

// Sliding-Window Attention block.
// Uses standard RoPE (rope_theta_swa) and a windowed view of the KV cache.
static ggml_tensor * build_swa_attn_block(
    ggml_context *             ctx,
    ggml_cgraph *              gf,
    const GemmaTargetWeights & w,
    const GemmaTargetLayer &   L,
    ggml_tensor *              cur,
    ggml_tensor *              positions,
    ggml_tensor *              cache_k,
    ggml_tensor *              cache_v,
    ggml_tensor *              attn_mask,
    int                        kv_start,
    int                        n_tokens,
    ggml_type                  kv_k_type,
    ggml_type                  kv_v_type,
    bool                       write_kv,
    int                        il)
{
    // SWA layers use the SWA head_dim (may be smaller than full-attn head_dim)
    const int head_dim  = w.head_dim_swa;
    const int n_head    = w.n_head;
    const int n_head_kv = (il >= 0 && il < (int)w.head_kv_per_layer.size())
                              ? w.head_kv_per_layer[il] : w.n_head_kv;
    const int q_dim     = n_head * head_dim;

    // Q projection
    ggml_tensor * Qcur = ggml_mul_mat(ctx, L.wq, cur);
    Qcur = ggml_reshape_3d(ctx, Qcur, head_dim, n_head, n_tokens);
    Qcur = rms_norm_mul(ctx, Qcur, L.q_norm, EPS);

    ggml_tensor * Kcur = nullptr;
    ggml_tensor * Vcur = nullptr;
    if (write_kv) {
        Kcur = ggml_mul_mat(ctx, L.wk, cur);
        Kcur = ggml_reshape_3d(ctx, Kcur, head_dim, n_head_kv, n_tokens);

        Vcur = ggml_mul_mat(ctx, L.wv, cur);
        Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_head_kv, n_tokens);

        if (L.k_norm) {
            Kcur = rms_norm_mul(ctx, Kcur, L.k_norm, EPS);
        }
        Vcur = ggml_rms_norm(ctx, Vcur, EPS);
    }

    // Standard RoPE (SWA uses rope_theta_swa, no freq_factors)
    Qcur = ggml_rope_ext(ctx, Qcur, positions, /*freq_factors=*/nullptr,
                         head_dim, GGML_ROPE_TYPE_NEOX, /*n_ctx_orig=*/0,
                         w.rope_theta_swa, /*freq_scale=*/1.0f,
                         /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                         /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
    if (Kcur) {
        Kcur = ggml_rope_ext(ctx, Kcur, positions, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             w.rope_theta_swa, 1.0f,
                             0.0f, 1.0f, 0.0f, 0.0f);
    }

    // SWA ring-buffer: derive the ring size from the tensor's actual slot count.
    // When swa_ctx_alloc < max_ctx (long contexts), writes use kv_start % ring_size
    // so the tensor is never exceeded.
    const int ring_size = cache_k ? (int)cache_k->ne[1] : (kv_start + n_tokens);

    // Write K/V into cache using ring-buffer position.
    // Split-on-wrap: when write_pos + n_tokens > ring_size the chunk straddles
    // the ring boundary, so we issue two ggml_cpy ops (pre-wrap and post-wrap).
    if (write_kv && cache_k && cache_v && Kcur && Vcur) {
        ggml_tensor * Kcur_T = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
        ggml_tensor * Vcur_T = ggml_permute(ctx, Vcur, 0, 2, 1, 3);

        const int write_pos = kv_start % ring_size;
        const int pre_n  = std::min(n_tokens, ring_size - write_pos);
        const int post_n = n_tokens - pre_n;

        // First slice: [write_pos .. write_pos+pre_n)
        {
            ggml_tensor * k_slot = ggml_view_3d(ctx, cache_k,
                head_dim, pre_n, n_head_kv,
                cache_k->nb[1], cache_k->nb[2],
                cache_k->nb[1] * write_pos);
            ggml_tensor * v_slot = ggml_view_3d(ctx, cache_v,
                head_dim, pre_n, n_head_kv,
                cache_v->nb[1], cache_v->nb[2],
                cache_v->nb[1] * write_pos);
            ggml_tensor * k_src = ggml_view_3d(ctx, Kcur_T,
                head_dim, pre_n, n_head_kv,
                Kcur_T->nb[1], Kcur_T->nb[2], 0);
            ggml_tensor * v_src = ggml_view_3d(ctx, Vcur_T,
                head_dim, pre_n, n_head_kv,
                Vcur_T->nb[1], Vcur_T->nb[2], 0);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, k_src, k_slot));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, v_src, v_slot));
        }

        // Second slice (wrap-around): [0 .. post_n)
        if (post_n > 0) {
            ggml_tensor * k_slot = ggml_view_3d(ctx, cache_k,
                head_dim, post_n, n_head_kv,
                cache_k->nb[1], cache_k->nb[2],
                0);
            ggml_tensor * v_slot = ggml_view_3d(ctx, cache_v,
                head_dim, post_n, n_head_kv,
                cache_v->nb[1], cache_v->nb[2],
                0);
            ggml_tensor * k_src = ggml_view_3d(ctx, Kcur_T,
                head_dim, post_n, n_head_kv,
                Kcur_T->nb[1], Kcur_T->nb[2],
                Kcur_T->nb[1] * pre_n);
            ggml_tensor * v_src = ggml_view_3d(ctx, Vcur_T,
                head_dim, post_n, n_head_kv,
                Vcur_T->nb[1], Vcur_T->nb[2],
                Vcur_T->nb[1] * pre_n);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, k_src, k_slot));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, v_src, v_slot));
        }
    }

    // Determine window for SWA reads using the shared geometry helper.
    // ring_win_start is always 0 (full-ring read); correctness comes from the
    // host-built mask which uses abs_pos(slot) arithmetic for ring geometry.
    const SwaView swa_view = compute_swa_view(kv_start, n_tokens,
                                               w.swa_window, ring_size);
    const int effective_win_len = swa_view.effective_win_len;
    const int ring_win_start    = swa_view.ring_win_start;  // always 0

    // swa_ctx_alloc is already aligned to fattn_stride (set in create_gemma4_cache),
    // so win_len_padded == effective_win_len == ring_size. No further snap needed.
    const bool need_256_pad = (kv_k_type == GGML_TYPE_TQ3_0 || kv_v_type == GGML_TYPE_TQ3_0
                               || head_dim >= 512);
    const int fattn_stride = need_256_pad ? 256 : 1;
    const int win_len_padded = ((effective_win_len + fattn_stride - 1) / fattn_stride) * fattn_stride;

    const bool q_rotate   = (kv_k_type == GGML_TYPE_TQ3_0);
    const bool out_rotate = (kv_v_type == GGML_TYPE_TQ3_0);
    // TQ3 contract: caller pre-rotates Q forward, post-rotates FA output
    // backward. ggml_turbo_wht's kernel writes dst using src strides
    // (turbo-wht.cu:20-21), so non-contiguous input scatters writes and
    // corrupts the result — wrap with ggml_cont before rotating, never after
    // permute alone. (Regression-fix vs. c15f93a; see EVIDENCE.md TQ3 thread.)
    ggml_tensor * Qfa = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));
    if (q_rotate) {
        Qfa = ggml_turbo_wht(ctx, Qfa, 0);
    }

    ggml_tensor * Kfa = ggml_view_3d(ctx, cache_k,
        head_dim, win_len_padded, n_head_kv,
        cache_k->nb[1], cache_k->nb[2],
        cache_k->nb[1] * ring_win_start);
    ggml_tensor * Vfa = ggml_view_3d(ctx, cache_v,
        head_dim, win_len_padded, n_head_kv,
        cache_v->nb[1], cache_v->nb[2],
        cache_v->nb[1] * ring_win_start);
    // Gemma4: attn_scale = 1.0 (self.scaling = 1.0, no 1/sqrt(head_dim))
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa, attn_mask,
                                             1.0f, 0.0f, 0.0f);

    if (out_rotate) {
        attn = ggml_cont(ctx, attn);
        attn = ggml_turbo_wht(ctx, attn, 1);
    }

    attn = ggml_reshape_2d(ctx, attn, q_dim, n_tokens);
    attn = ggml_mul_mat(ctx, L.wo, attn);
    return attn;
}

// Full (Global) Attention block.
// Uses proportional RoPE via per-layer rope_freqs (freq_factors) and full context.
// When use_pflash is true, uses ggml_flash_attn_sparse (block-sparse) instead of
// ggml_flash_attn_ext for the attention computation.
static ggml_tensor * build_full_attn_block(
    ggml_context *             ctx,
    ggml_cgraph *              gf,
    const GemmaTargetWeights & w,
    const GemmaTargetLayer &   L,
    ggml_tensor *              cur,
    ggml_tensor *              positions,
    ggml_tensor *              cache_k,
    ggml_tensor *              cache_v,
    ggml_tensor *              attn_mask,
    int                        kv_start,
    int                        n_tokens,
    ggml_type                  kv_k_type,
    ggml_type                  kv_v_type,
    bool                       write_kv,
    int                        fa_window,
    int                        il,
    bool                       use_pflash,
    float                      pflash_alpha)
{
    // Full-attention layers use the full head_dim
    const int head_dim  = w.head_dim;
    const int n_head    = w.n_head;
    const int n_head_kv = (il >= 0 && il < (int)w.head_kv_per_layer.size())
                              ? w.head_kv_per_layer[il] : w.n_head_kv;
    const int q_dim     = n_head * head_dim;

    // Q projection
    ggml_tensor * Qcur = ggml_mul_mat(ctx, L.wq, cur);
    Qcur = ggml_reshape_3d(ctx, Qcur, head_dim, n_head, n_tokens);
    Qcur = rms_norm_mul(ctx, Qcur, L.q_norm, EPS);

    ggml_tensor * Kcur = nullptr;
    ggml_tensor * Vcur = nullptr;
    if (write_kv) {
        Kcur = ggml_mul_mat(ctx, L.wk, cur);
        Kcur = ggml_reshape_3d(ctx, Kcur, head_dim, n_head_kv, n_tokens);

        // V = K (pre-norm) when wv absent, else separate projection
        if (L.wv == L.wk) {
            Vcur = Kcur;
        } else {
            Vcur = ggml_mul_mat(ctx, L.wv, cur);
            Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_head_kv, n_tokens);
        }

        // K gets weighted RMSNorm, V gets bare RMSNorm (no learned weights)
        if (L.k_norm) {
            Kcur = rms_norm_mul(ctx, Kcur, L.k_norm, EPS);
        }
        Vcur = ggml_rms_norm(ctx, Vcur, EPS);
    }

    // Proportional RoPE for full-attention layers (uses per-layer rope_freqs)
    Qcur = ggml_rope_ext(ctx, Qcur, positions, L.rope_freqs,
                         head_dim, GGML_ROPE_TYPE_NEOX, /*n_ctx_orig=*/0,
                         w.rope_theta, /*freq_scale=*/1.0f,
                         /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                         /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
    if (Kcur) {
        Kcur = ggml_rope_ext(ctx, Kcur, positions, L.rope_freqs,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             w.rope_theta, 1.0f,
                             0.0f, 1.0f, 0.0f, 0.0f);
    }

    // Write K/V into cache
    if (write_kv && cache_k && cache_v && Kcur && Vcur) {
        ggml_tensor * Kcur_T = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
        ggml_tensor * Vcur_T = ggml_permute(ctx, Vcur, 0, 2, 1, 3);

        ggml_tensor * k_slot = ggml_view_3d(ctx, cache_k,
            head_dim, n_tokens, n_head_kv,
            cache_k->nb[1], cache_k->nb[2],
            cache_k->nb[1] * kv_start);
        ggml_tensor * v_slot = ggml_view_3d(ctx, cache_v,
            head_dim, n_tokens, n_head_kv,
            cache_v->nb[1], cache_v->nb[2],
            cache_v->nb[1] * kv_start);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur_T, k_slot));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur_T, v_slot));
    }

    // For full-attention layers: optional windowed FA for long-context efficiency
    const int win_start = (fa_window > 0 && kv_start > fa_window)
                              ? (kv_start - fa_window) : 0;
    const int kv_len  = kv_start + n_tokens;
    const int win_len = kv_len - win_start;

    const bool need_256_pad = (kv_k_type == GGML_TYPE_TQ3_0 || kv_v_type == GGML_TYPE_TQ3_0
                               || head_dim >= 512);
    const int fattn_stride = need_256_pad ? 256 : 1;
    const int win_len_padded = ((win_len + fattn_stride - 1) / fattn_stride) * fattn_stride;

    const bool q_rotate   = (kv_k_type == GGML_TYPE_TQ3_0);
    const bool out_rotate = (kv_v_type == GGML_TYPE_TQ3_0);
    // See SWA block above for the contiguity rationale (turbo-wht.cu:20-21).
    ggml_tensor * Qfa = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));
    if (q_rotate) {
        Qfa = ggml_turbo_wht(ctx, Qfa, 0);
    }

    ggml_tensor * Kfa = ggml_view_3d(ctx, cache_k,
        head_dim, win_len_padded, n_head_kv,
        cache_k->nb[1], cache_k->nb[2],
        cache_k->nb[1] * win_start);
    ggml_tensor * Vfa = ggml_view_3d(ctx, cache_v,
        head_dim, win_len_padded, n_head_kv,
        cache_v->nb[1], cache_v->nb[2],
        cache_v->nb[1] * win_start);

    // pFlash sparse path supports F16, Q8_0, and Q4_0 K/V — the CUDA dispatch layer
    // dequantizes to F16 before the S<->H BF16 transpose for these types.
    // TQ3_0 is excluded because sparse FA does not consume rotated TQ3 K/V
    // directly; fall back to dense FA for TQ3_0 and other types.
    auto pflash_supports = [](enum ggml_type t) {
        return t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0;
    };
    const bool can_pflash = use_pflash &&
                            pflash_supports(Kfa->type) &&
                            pflash_supports(Vfa->type);

    // Gemma4: attn_scale = 1.0 (self.scaling = 1.0, no 1/sqrt(head_dim))
    ggml_tensor * attn;
    if (can_pflash) {
        attn = ggml_flash_attn_sparse(ctx, Qfa, Kfa, Vfa, 1.0f, pflash_alpha);
    } else {
        attn = ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa, attn_mask, 1.0f, 0.0f, 0.0f);
    }

    if (out_rotate) {
        attn = ggml_cont(ctx, attn);
        attn = ggml_turbo_wht(ctx, attn, 1);
    }

    attn = ggml_reshape_2d(ctx, attn, q_dim, n_tokens);
    attn = ggml_mul_mat(ctx, L.wo, attn);
    return attn;
}

// ─── GemmaTargetCache allocation ─────────────────────────────────────────────

bool create_gemma4_cache(const GemmaTargetWeights & w,
                         int max_ctx,
                         ggml_backend_t backend,
                         GemmaTargetCache & out,
                         const std::vector<int> & extra_q8_layers,
                         int target_feat_cap_hint,
                         bool enable_dflash_capture_overrides) {
    out.backend = backend;
    out.max_ctx = max_ctx;
    out.cur_pos = 0;

    // Resolve KV types from environment
    ggml_type kv_k_type = GGML_TYPE_Q8_0;
    ggml_type kv_v_type = GGML_TYPE_Q8_0;
    dflash::resolve_kv_types(kv_k_type, kv_v_type);
    out.kv_k_type = kv_k_type;
    out.kv_v_type = kv_v_type;

    // TQ3_0 and head_dim>=512 (CUDA FA FATTN_KQ_STRIDE) require 256-alignment
    const bool need_256_align = (kv_k_type == GGML_TYPE_TQ3_0 || kv_v_type == GGML_TYPE_TQ3_0
                                 || w.head_dim >= 512);
    const int align_stride = need_256_align ? 256 : 1;
    const int max_ctx_alloc = need_256_align
        ? ((max_ctx + 255) / 256) * 256
        : max_ctx;

    // SWA layers only need swa_window slots (ring-buffer). Allocate
    // min(max_ctx_alloc, swa_window_padded) for SWA layers, saving ~50% VRAM
    // at long contexts. swa_ctx_alloc must be strictly > swa_window so the
    // decode window (win_len = swa_window + n_tokens) fits within one view.
    // We pad swa_window to the same alignment stride and add one alignment
    // block as headroom so contiguous views always work for n_tokens=1 decode.
    const int swa_window_padded = (w.swa_window > 0)
        ? ((w.swa_window + align_stride - 1) / align_stride) * align_stride
        : max_ctx_alloc;
    // Ring sized to hold last R = 2*swa_window keys (= 2 chunks worth, since
    // chunk_size <= swa_window). Combined with a non-monotonic mask in the
    // test driver's build_swa_causal_mask, this lets the K view be the full
    // ring while correctness comes from the mask filtering by abs_pos.
    const int swa_ring_target = 2 * swa_window_padded;
    const int swa_ctx_alloc = (w.swa_window > 0)
        ? std::min(max_ctx_alloc, swa_ring_target)
        : max_ctx_alloc;
    out.swa_ctx_alloc = swa_ctx_alloc;

    // Build layer -> KV index mappings.
    // Gemma4 can share KV caches across layers. The weight loader sets wk=nullptr
    // for shared layers. We detect this and point them at the most recent
    // non-shared layer's KV slot.
    out.layer_to_kv_idx.assign(w.n_layer, -1);
    out.layer_to_donor_kv.assign(w.n_layer, -1);

    int n_kv_slots = 0;
    for (int il = 0; il < w.n_layer; il++) {
        if (w.layers[il].wk != nullptr) {
            out.layer_to_kv_idx[il] = n_kv_slots++;
        }
    }

    // For shared layers, find the most recent layer that owns a KV slot
    int last_kv_slot = -1;
    for (int il = 0; il < w.n_layer; il++) {
        if (out.layer_to_kv_idx[il] >= 0) {
            last_kv_slot = out.layer_to_kv_idx[il];
        } else {
            out.layer_to_donor_kv[il] = last_kv_slot;
        }
    }

    if (n_kv_slots == 0) {
        set_last_error("create_gemma4_cache: no KV-owning layers found");
        return false;
    }

    // Per-layer KV types.
    //
    // The upstream FA dispatch (deps/llama.cpp/.../fattn.cu:441) routes
    // TQ3 + (Q->ne[0] > 256 || Q->ne[1] > 1) to the slow CHUNKED kernel.
    // On Dense Gemma4 31B with full-attn head_dim=512, every chunked
    // prefill / draft-verify hits this trap.
    //
    // Narrow workaround (Codex pattern, mirrors vLLM's kv-cache-dtype-skip-layers):
    // when the DFlash draft is wired up, force Q8_0 KV on the small subset of
    // full-attn layers whose hidden states are CAPTURED for the draft (the
    // "target_feat" ring at gemma4_target_graph.cpp:971 — drafter consumes
    // these in build_gemma4_draft_graph). This unblocks the pflash sparse
    // fast path for the layers the draft actually depends on, without
    // touching the other 8/10 full-attn layers (avoids the MoE regression
    // we saw when forcing ALL full-attn -> Q8).
    out.kv_k_type_per_layer.assign(w.n_layer, kv_k_type);
    out.kv_v_type_per_layer.assign(w.n_layer, kv_v_type);

    const bool gate = (kv_k_type == GGML_TYPE_TQ3_0 || kv_v_type == GGML_TYPE_TQ3_0)
                    && (w.head_dim > 256)
                    && enable_dflash_capture_overrides
                    && (w.n_capture_layers > 0);

    if (gate) {
        int n_overridden = 0;
        for (int ci = 0; ci < w.n_capture_layers; ci++) {
            const int captured_il = w.capture_layer_ids[ci];
            if (captured_il < 0 || captured_il >= w.n_layer) continue;
            const bool is_swa = (captured_il < (int)w.swa_layers.size())
                                && w.swa_layers[captured_il];
            if (is_swa) continue;  // SWA layers don't hit the trap
            if (kv_k_type == GGML_TYPE_TQ3_0) {
                out.kv_k_type_per_layer[captured_il] = GGML_TYPE_Q8_0;
            }
            if (kv_v_type == GGML_TYPE_TQ3_0) {
                out.kv_v_type_per_layer[captured_il] = GGML_TYPE_Q8_0;
            }
            n_overridden++;
        }
        // Count total full-attn layers for the log message
        int n_full_attn = 0;
        for (int il = 0; il < w.n_layer; il++) {
            const bool is_swa = (il < (int)w.swa_layers.size()) && w.swa_layers[il];
            if (!is_swa && out.layer_to_kv_idx[il] >= 0) n_full_attn++;
        }
        std::fprintf(stderr,
            "[cache] narrow asymmetric: forced Q8_0 on %d captured full-attn layer(s) "
            "(remaining %d full-attn keep TQ3)\n",
            n_overridden, n_full_attn - n_overridden);
    }

    // Extra override: force Q8_0 on caller-specified layer indices (e.g. MTP donor layers).
    // These layers must NOT use TQ3_0 because MTP cross-attention reads them via ggml_cast
    // (no FWHT inverse applied), so TQ3_0 FWHT-domain values would corrupt attention scores.
    if (!extra_q8_layers.empty() &&
        (kv_k_type == GGML_TYPE_TQ3_0 || kv_v_type == GGML_TYPE_TQ3_0)) {
        int n_mtp_overridden = 0;
        for (int il : extra_q8_layers) {
            if (il < 0 || il >= w.n_layer) continue;
            if (kv_k_type == GGML_TYPE_TQ3_0) out.kv_k_type_per_layer[il] = GGML_TYPE_Q8_0;
            if (kv_v_type == GGML_TYPE_TQ3_0) out.kv_v_type_per_layer[il] = GGML_TYPE_Q8_0;
            n_mtp_overridden++;
        }
        if (n_mtp_overridden > 0) {
            std::fprintf(stderr,
                "[cache] MTP donor override: forced Q8_0 on %d layer(s) to avoid TQ3/FWHT cross-attn mismatch\n",
                n_mtp_overridden);
        }
    }

    // (head_dim and n_head_kv are resolved per-layer in the allocation loop below)

    const int n_capture_layers = w.n_capture_layers;
    const int n_embd            = w.n_embd;

    // Tensor count: 2 (K+V) per KV slot + 1 target_feat
    const int n_tensors = 2 * n_kv_slots + 1;
    ggml_init_params ip{};
    ip.mem_size   = (size_t)(n_tensors + 16) * ggml_tensor_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    out.base_ctx = ggml_init(ip);
    if (!out.base_ctx) {
        set_last_error("create_gemma4_cache: ggml_init failed");
        return false;
    }

    out.attn_k.assign(n_kv_slots, nullptr);
    out.attn_v.assign(n_kv_slots, nullptr);

    // Create KV tensors — iterate layers to preserve name <-> layer correlation.
    // Each layer's KV slot uses the head_dim and n_head_kv appropriate to its
    // attention type (SWA vs full-attention may have different dimensions).
    for (int il = 0; il < w.n_layer; il++) {
        const int kv_idx = out.layer_to_kv_idx[il];
        if (kv_idx < 0) continue;

        const bool is_swa_layer = (il < (int)w.swa_layers.size()) && w.swa_layers[il];
        const int layer_head_dim  = is_swa_layer ? w.head_dim_swa : w.head_dim;
        const int layer_n_head_kv = (il < (int)w.head_kv_per_layer.size())
                                        ? w.head_kv_per_layer[il] : w.n_head_kv;

        // SWA layers use a ring buffer of swa_ctx_alloc slots; full-attn layers
        // need the full max_ctx_alloc to cover the entire context.
        const int layer_ctx_alloc = is_swa_layer ? swa_ctx_alloc : max_ctx_alloc;

        const ggml_type layer_kv_k_type = out.kv_k_type_per_layer[il];
        const ggml_type layer_kv_v_type = out.kv_v_type_per_layer[il];
        ggml_tensor * K = ggml_new_tensor_3d(out.base_ctx, layer_kv_k_type,
                                             layer_head_dim, layer_ctx_alloc, layer_n_head_kv);
        ggml_tensor * V = ggml_new_tensor_3d(out.base_ctx, layer_kv_v_type,
                                             layer_head_dim, layer_ctx_alloc, layer_n_head_kv);
        char name[64];
        std::snprintf(name, sizeof(name), "gemma4_cache_k_%d", il);
        ggml_set_name(K, name);
        std::snprintf(name, sizeof(name), "gemma4_cache_v_%d", il);
        ggml_set_name(V, name);
        out.attn_k[kv_idx] = K;
        out.attn_v[kv_idx] = V;
    }

    // target_feat ring buffer: [n_capture_layers * n_embd, cap] bf16
    constexpr int TARGET_FEAT_CAP_DEFAULT = 4096;
    const int target_feat_cap_req = std::max(TARGET_FEAT_CAP_DEFAULT, target_feat_cap_hint);
    out.target_feat_cap = std::min(max_ctx, target_feat_cap_req);
    {
        const int fc_in = n_capture_layers * n_embd;
        out.target_feat = ggml_new_tensor_2d(out.base_ctx, GGML_TYPE_BF16,
                                             fc_in, out.target_feat_cap);
        ggml_set_name(out.target_feat, "gemma4_target_feat");
    }

    out.base_buf = ggml_backend_alloc_ctx_tensors(out.base_ctx, backend);
    if (!out.base_buf) {
        set_last_error("create_gemma4_cache: ggml_backend_alloc_ctx_tensors failed");
        ggml_free(out.base_ctx);
        out.base_ctx = nullptr;
        return false;
    }

    // Count full-attn vs SWA KV-owning layers for VRAM savings log.
    int n_full_kv = 0, n_swa_kv = 0;
    for (int il = 0; il < w.n_layer; il++) {
        if (out.layer_to_kv_idx[il] < 0) continue;
        const bool is_swa = (il < (int)w.swa_layers.size()) && w.swa_layers[il];
        if (is_swa) n_swa_kv++; else n_full_kv++;
    }
    const float full_slots = (float)n_full_kv  * max_ctx_alloc;
    const float swa_slots  = (float)n_swa_kv   * swa_ctx_alloc;
    const float old_slots  = (float)(n_full_kv + n_swa_kv) * max_ctx_alloc;
    const float saved_pct  = old_slots > 0.0f
        ? 100.0f * (1.0f - (full_slots + swa_slots) / old_slots)
        : 0.0f;
    // Find a representative SWA layer index and a representative full-attn layer index
    // for the diagnostic log (first of each kind that owns a KV slot).
    int repr_swa_il = -1, repr_full_il = -1;
    for (int il = 0; il < w.n_layer; il++) {
        if (out.layer_to_kv_idx[il] < 0) continue;
        const bool is_swa = (il < (int)w.swa_layers.size()) && w.swa_layers[il];
        if (is_swa  && repr_swa_il  < 0) repr_swa_il  = il;
        if (!is_swa && repr_full_il < 0) repr_full_il = il;
        if (repr_swa_il >= 0 && repr_full_il >= 0) break;
    }
    const char * swa_k_name  = (repr_swa_il  >= 0)
        ? ggml_type_name(out.kv_k_type_per_layer[repr_swa_il])  : "n/a";
    const char * full_k_name = (repr_full_il >= 0)
        ? ggml_type_name(out.kv_k_type_per_layer[repr_full_il]) : "n/a";
    std::fprintf(stderr,
        "[cache] created max_ctx=%d (full_attn=%d, swa=%d), kv_layers=%d, saved %.1f%%\n",
        max_ctx, max_ctx_alloc, swa_ctx_alloc, n_kv_slots, saved_pct);
    std::fprintf(stderr, "[cache] kv types: SWA=%s, full=%s\n", swa_k_name, full_k_name);

    // Zero-initialize all tensors
    std::vector<uint8_t> zeros(1 * 1024 * 1024, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(out.base_ctx); t != nullptr;
         t = ggml_get_next_tensor(out.base_ctx, t)) {
        size_t nb  = ggml_nbytes(t);
        size_t off = 0;
        while (off < nb) {
            size_t chunk = std::min(nb - off, zeros.size());
            ggml_backend_tensor_set(t, zeros.data(), off, chunk);
            off += chunk;
        }
    }

    return true;
}

void free_gemma4_cache(GemmaTargetCache & c) {
    free_draft_kv_cache(c);
    if (c.base_buf) { ggml_backend_buffer_free(c.base_buf); c.base_buf = nullptr; }
    if (c.base_ctx) { ggml_free(c.base_ctx);                c.base_ctx = nullptr; }
    c.attn_k.clear();
    c.attn_v.clear();
    c.layer_to_kv_idx.clear();
    c.layer_to_donor_kv.clear();
    c.target_feat     = nullptr;
    c.cur_pos         = 0;
    c.last_tok        = -1;
    c.swa_ctx_alloc   = 0;
}

void reset_gemma4_cache(GemmaTargetCache & c) {
    c.cur_pos      = 0;
    c.last_tok     = -1;
    c.draft_kv_pos = 0;
    std::vector<uint8_t> zeros(1 * 1024 * 1024, 0);
    if (!c.base_ctx) return;
    for (ggml_tensor * t = ggml_get_first_tensor(c.base_ctx); t != nullptr;
         t = ggml_get_next_tensor(c.base_ctx, t)) {
        size_t nb  = ggml_nbytes(t);
        size_t off = 0;
        while (off < nb) {
            size_t chunk = std::min(nb - off, zeros.size());
            ggml_backend_tensor_set(t, zeros.data(), off, chunk);
            off += chunk;
        }
    }
}

// ─── Draft KV cache allocation ───────────────────────────────────────────────

bool create_draft_kv_cache(const GemmaDraftWeights & dw,
                           ggml_backend_t backend,
                           GemmaTargetCache & cache,
                           int cap_override) {
    // Capacity: sliding window + one block + headroom
    const int default_cap = dw.sliding_window + dw.block_size + 32;
    const int draft_kv_cap = cap_override > 0 ? cap_override : default_cap;
    if (draft_kv_cap < dw.block_size + 1) {
        set_last_error("create_draft_kv_cache: cap_override is smaller than block_size+1");
        return false;
    }

    const size_t n_tensors = (size_t)(2 * dw.n_layer);  // K + V per layer
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * n_tensors + 256;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    cache.draft_kv_ctx = ggml_init(ip);
    if (!cache.draft_kv_ctx) {
        set_last_error("create_draft_kv_cache: ggml_init failed");
        return false;
    }

    cache.draft_k.reserve((size_t)dw.n_layer);
    cache.draft_v.reserve((size_t)dw.n_layer);

    for (int il = 0; il < dw.n_layer; il++) {
        ggml_tensor * K = ggml_new_tensor_3d(cache.draft_kv_ctx, GGML_TYPE_F32,
                                             dw.head_dim, dw.n_head_kv, draft_kv_cap);
        ggml_tensor * V = ggml_new_tensor_3d(cache.draft_kv_ctx, GGML_TYPE_F32,
                                             dw.head_dim, dw.n_head_kv, draft_kv_cap);
        char name[64];
        std::snprintf(name, sizeof(name), "draft_k_%d", il);
        ggml_set_name(K, name);
        std::snprintf(name, sizeof(name), "draft_v_%d", il);
        ggml_set_name(V, name);
        cache.draft_k.push_back(K);
        cache.draft_v.push_back(V);
    }

    cache.draft_kv_buf = ggml_backend_alloc_ctx_tensors(cache.draft_kv_ctx, backend);
    if (!cache.draft_kv_buf) {
        set_last_error("create_draft_kv_cache: ggml_backend_alloc_ctx_tensors failed");
        ggml_free(cache.draft_kv_ctx);
        cache.draft_kv_ctx = nullptr;
        cache.draft_k.clear();
        cache.draft_v.clear();
        return false;
    }

    cache.draft_kv_cap = draft_kv_cap;
    cache.draft_kv_pos = 0;

    ggml_backend_buffer_clear(cache.draft_kv_buf, 0);

    return true;
}

void free_draft_kv_cache(GemmaTargetCache & cache) {
    if (cache.draft_kv_buf) {
        ggml_backend_buffer_free(cache.draft_kv_buf);
        cache.draft_kv_buf = nullptr;
    }
    if (cache.draft_kv_ctx) {
        ggml_free(cache.draft_kv_ctx);
        cache.draft_kv_ctx = nullptr;
    }
    cache.draft_k.clear();
    cache.draft_v.clear();
    cache.draft_kv_cap = 0;
    cache.draft_kv_pos = 0;
}

// ─── Main graph builder ───────────────────────────────────────────────────────

GemmaGraphOutputs build_gemma4_graph(
    ggml_context *              ctx,
    ggml_cgraph *               gf,
    const GemmaTargetWeights &  w,
    GemmaTargetCache &          cache,
    const GemmaGraphInputs &    in)
{
    const int n_tokens = in.n_tokens;
    const int kv_start = in.kv_start;
    const int n_embd   = w.n_embd;

    // CUDA FA for head_dim>=512 requires a non-null mask to enable the GQA
    // optimization path (gqa_opt_applies=true).  Auto-create a causal mask
    // when the caller did not supply one so that full-attention layers don't
    // hit BEST_FATTN_KERNEL_NONE → abort.
    ggml_tensor * attn_mask = in.attn_mask;
    if (!attn_mask && w.head_dim >= 512) {
        const int kv_len        = kv_start + n_tokens;
        // Pad to 256 — required by FATTN_KQ_STRIDE for TQ3 / large head_dim.
        const int kv_len_padded = ((kv_len + 255) / 256) * 256;
        attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len_padded, n_tokens);
        ggml_set_name(attn_mask, "auto_causal_mask");
        ggml_set_input(attn_mask);
    }

    ggml_tensor * inpL = in.inp_embed;  // [n_embd, n_tokens] f32

    // Gemma4 scales embeddings by sqrt(n_embd) (matches HF Gemma4TextScaledWordEmbedding)
    inpL = ggml_scale(ctx, inpL, std::sqrt((float)n_embd));

    for (int il = 0; il < w.n_layer; il++) {
        const GemmaTargetLayer & L = w.layers[il];
        const bool is_swa = (il < (int)w.swa_layers.size()) ? w.swa_layers[il] : true;

        // ── a) Pre-attention RMSNorm ────────────────────────────────────────────
        ggml_tensor * inpSA = inpL;
        ggml_tensor * cur   = rms_norm_mul(ctx, inpL, L.attn_norm, EPS);

        // ── b-f) Attention (SWA or Full) ───────────────────────────────────────
        const int kv_idx = cache.layer_to_kv_idx[il];
        const bool write_kv = (kv_idx >= 0);

        // Determine which KV cache buffers to use for reading
        const int read_kv_idx = write_kv ? kv_idx : cache.layer_to_donor_kv[il];
        ggml_tensor * cache_k = (read_kv_idx >= 0) ? cache.attn_k[read_kv_idx] : nullptr;
        ggml_tensor * cache_v = (read_kv_idx >= 0) ? cache.attn_v[read_kv_idx] : nullptr;

        // Resolve per-layer KV types (asymmetric: TQ3 on SWA, Q8 on full-attn).
        const ggml_type layer_kv_k = !cache.kv_k_type_per_layer.empty()
            ? cache.kv_k_type_per_layer[il] : cache.kv_k_type;
        const ggml_type layer_kv_v = !cache.kv_v_type_per_layer.empty()
            ? cache.kv_v_type_per_layer[il] : cache.kv_v_type;

        if (is_swa) {
            ggml_tensor * effective_mask = in.swa_mask ? in.swa_mask : attn_mask;
            cur = build_swa_attn_block(ctx, gf, w, L, cur, in.positions,
                                       cache_k, cache_v, effective_mask,
                                       kv_start, n_tokens,
                                       layer_kv_k, layer_kv_v,
                                       write_kv, il);
        } else {
            cur = build_full_attn_block(ctx, gf, w, L, cur, in.positions,
                                        cache_k, cache_v, attn_mask,
                                        kv_start, n_tokens,
                                        layer_kv_k, layer_kv_v,
                                        write_kv, in.fa_window, il,
                                        in.use_pflash, in.pflash_alpha);
        }

        // ── g) Output projection already done inside attn block ────────────────

        // ── h) Post-attention norm + residual ──────────────────────────────────
        if (L.attn_post_norm) {
            cur = rms_norm_mul(ctx, cur, L.attn_post_norm, EPS);
        }
        // NOTE: out_scale is applied AFTER the full layer (after FFN), not here
        ggml_tensor * inpSA_post = ggml_add(ctx, cur, inpSA);

        // ── i) FFN ─────────────────────────────────────────────────────────────
        ggml_tensor * ffn_residual = inpSA_post;
        ggml_tensor * ffn_in = rms_norm_mul(ctx, inpSA_post, L.ffn_norm, EPS);

        ggml_tensor * ffn_out = nullptr;
        if (L.ffn_gate_inp != nullptr) {
            // MoE path (26B-A4B): shared expert uses ffn_norm, routed use ffn_pre_norm_2
            ggml_tensor * moe_in = L.ffn_pre_norm_2
                ? rms_norm_mul(ctx, inpSA_post, L.ffn_pre_norm_2, EPS)
                : ffn_in;
            ffn_out = build_moe_ffn(ctx, gf, w, L,
                                    ffn_in, moe_in, inpSA_post,
                                    n_tokens);
        } else {
            // Dense path (31B)
            ffn_out = build_geglu_ffn(ctx, ffn_in, L);
        }

        // Post-FFN norm
        if (L.ffn_post_norm) {
            ffn_out = rms_norm_mul(ctx, ffn_out, L.ffn_post_norm, EPS);
        }

        cur = ggml_add(ctx, ffn_out, ffn_residual);

        // ── layer_output_scale: applied after full layer (attn + FFN residuals) ─
        // Matches HF: hidden_states = layer_scalar * (attn_residual + ffn_residual)
        if (L.out_scale) {
            cur = ggml_mul(ctx, cur, L.out_scale);
        }

        // ── j) Per-Layer Embedding (PLE) ───────────────────────────────────────
        if (in.per_layer_inp && L.ple_inp_gate && L.ple_proj) {
            // ple_inp_gate: gate projection
            ggml_tensor * ple_gate = ggml_mul_mat(ctx, L.ple_inp_gate, cur);
            ple_gate = ggml_gelu(ctx, ple_gate);

            // per_layer_inp is [n_embd_per_layer, n_tokens, n_layer] or similar;
            // we select the slice for this layer along axis 2.
            // Assuming per_layer_inp is [n_embd_per_layer, n_tokens] for this layer
            // (caller pre-selects by layer index) — or it is [n_embd_per_layer, n_layer]
            // shaped with the layer axis being dim 1.
            // Use a view to extract the il-th column if per_layer_inp has n_layer cols.
            const int n_embd_per_layer = w.n_embd_per_layer > 0 ? w.n_embd_per_layer
                                                                  : (int)in.per_layer_inp->ne[0];
            ggml_tensor * ple_emb;
            if (ggml_n_dims(in.per_layer_inp) >= 3 || (int)in.per_layer_inp->ne[1] == w.n_layer) {
                // Shape [n_embd_per_layer, n_layer] or [n_embd_per_layer, n_tokens, n_layer]
                ple_emb = ggml_view_2d(ctx, in.per_layer_inp,
                    n_embd_per_layer, n_tokens,
                    in.per_layer_inp->nb[1],
                    (size_t)il * n_tokens * in.per_layer_inp->nb[1]);
            } else {
                // Already sliced per-layer by caller
                ple_emb = in.per_layer_inp;
            }

            ggml_tensor * ple = ggml_mul(ctx, ple_gate, ple_emb);
            ple = ggml_mul_mat(ctx, L.ple_proj, ple);
            if (L.ple_post_norm) {
                ple = rms_norm_mul(ctx, ple, L.ple_post_norm, EPS);
            }
            cur = ggml_add(ctx, cur, ple);
        }

        // ── k) Target feature capture ──────────────────────────────────────────
        if (in.capture_layers && cache.target_feat) {
            int capture_idx = -1;
            for (int k = 0; k < w.n_capture_layers; k++) {
                if (w.capture_layer_ids[k] == il) { capture_idx = k; break; }
            }
            if (capture_idx >= 0) {
                const size_t elt        = ggml_element_size(cache.target_feat);
                const size_t col_stride = cache.target_feat->nb[1];
                const int    cap        = cache.target_feat_cap;
                const int    slot_start = kv_start % cap;
                const int    pre_n      = std::min(n_tokens, cap - slot_start);
                const int    post_n     = n_tokens - pre_n;

                ggml_tensor * cur_2d = ggml_reshape_2d(ctx, cur, n_embd, n_tokens);

                // First slice: [slot_start..slot_start+pre_n) in the ring
                {
                    const size_t offset =
                        (size_t)slot_start * col_stride +
                        (size_t)capture_idx * n_embd * elt;
                    ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                        n_embd, pre_n, col_stride, offset);
                    ggml_tensor * src  = ggml_view_2d(ctx, cur_2d,
                        n_embd, pre_n, cur_2d->nb[1], 0);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
                }

                // Second slice: wrap-around at [0..post_n) if needed
                if (post_n > 0) {
                    const size_t offset =
                        (size_t)capture_idx * n_embd * elt;
                    ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                        n_embd, post_n, col_stride, offset);
                    ggml_tensor * src  = ggml_view_2d(ctx, cur_2d,
                        n_embd, post_n, cur_2d->nb[1],
                        (size_t)pre_n * cur_2d->nb[1]);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
                }
            }
        }

        // ── l) Advance residual stream ──────────────────────────────────────────
        inpL = cur;
    }

    // ── Final norm ─────────────────────────────────────────────────────────────
    ggml_tensor * out = rms_norm_mul(ctx, inpL, w.out_norm, EPS);

    // ── MTP h_prev capture (post-output-norm, last token) ──────────────────────
    // h_prev must be the backbone hidden AFTER final RMSNorm — the same vector
    // fed to lm_head — so the MTP draft head sees the same representation as
    // the target's token prediction.  Capturing inside the layer loop (pre-norm)
    // caused accept_rate=0 because the draft head was trained on post-norm hiddens.
    // Source: vLLM PR #41745:569-621 + llama.cpp #22738.
    if (cache.mtp_h_prev_enabled && cache.mtp_h_prev) {
        const int n_embd_hp = (int)cache.mtp_h_prev->ne[0];
        ggml_tensor * h_prev_src = out;
        if (n_tokens > 1) {
            h_prev_src = ggml_view_2d(ctx, out,
                n_embd_hp, 1,
                ggml_row_size(out->type, n_embd_hp),
                ggml_row_size(out->type, n_embd_hp) * (n_tokens - 1));
        }
        if (h_prev_src->type != GGML_TYPE_F32) {
            h_prev_src = ggml_cast(ctx, h_prev_src, GGML_TYPE_F32);
        }
        h_prev_src = ggml_reshape_2d(ctx, h_prev_src, n_embd_hp, 1);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, h_prev_src, cache.mtp_h_prev));
    }

    // ── last_token_logits_only: slice to the final token before lm_head ────────
    // During chunked prefill we only need the last token's logits to seed decode.
    // Slicing here reduces lm_head compute from O(n_tokens) to O(1) and avoids
    // allocating a [vocab, n_tokens] output tensor (saves ~1 GB for chunk_size=1024).
    if (in.last_token_logits_only && n_tokens > 1) {
        out = ggml_view_2d(ctx, out,
            n_embd, 1,
            ggml_row_size(out->type, n_embd),
            ggml_row_size(out->type, n_embd) * (n_tokens - 1));
    }

    // ── LM head ────────────────────────────────────────────────────────────────
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, out);

    // ── Logit softcapping: logits = softcap * tanh(logits / softcap) ──────────
    if (w.logit_softcap > 0.0f) {
        logits = ggml_scale(ctx, logits, 1.0f / w.logit_softcap);
        logits = ggml_tanh(ctx, logits);
        logits = ggml_scale(ctx, logits, w.logit_softcap);
    }

    ggml_set_name(logits, "logits");
    ggml_build_forward_expand(gf, logits);

    GemmaGraphOutputs og{};
    og.logits = logits;
    return og;
}

} // namespace dflash27b
