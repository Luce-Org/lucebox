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
//   - Attention scale = 1.0 (no sqrt(head_dim) division)
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

// Standard SwiGLU FFN: w_down @ (silu(w_gate @ x) * (w_up @ x))
static ggml_tensor * build_swiglu_ffn(ggml_context * ctx,
                                      ggml_tensor * cur,
                                      const GemmaTargetLayer & L) {
    ggml_tensor * gate = ggml_mul_mat(ctx, L.w_gate, cur);
    ggml_tensor * up   = ggml_mul_mat(ctx, L.w_up,   cur);
    ggml_tensor * gu   = ggml_swiglu_split(ctx, gate, up);
    return ggml_mul_mat(ctx, L.w_down, gu);
}

// MoE FFN — shared expert + softmax-gated routed experts.
// Matches Gemma4-26B-A4B architecture:
//   shared_out  = w_down @ (silu(w_gate @ x) * (w_up @ x))
//   shared_out  = rms_norm(shared_out) * ffn_post_norm_1
//   router_in   = rms_norm(inpSA) * ffn_pre_norm_2 / sqrt(n_embd)
//   router_in   = router_in * ffn_gate_inp_s          (per-channel scale)
//   logits      = ffn_gate_inp @ router_in             [n_expert, n_tokens]
//   probs       = softmax(logits)
//   top_ids     = argsort_top_k(probs, n_expert_used)  [n_expert_used, n_tokens] i32
//   weights     = get_rows(probs, top_ids)             [1, n_expert_used, n_tokens]
//   gate_up_out = mul_mat_id(ffn_gate_up_exps, x, top_ids) → silu+mul → weighted
//   expert_out  = mul_mat_id(ffn_down_exps, act, top_ids) [n_embd, n_expert_used, n_tokens]
//   expert_out  = sum over expert dim                  [n_embd, n_tokens]
//   expert_out  = rms_norm(expert_out) * ffn_post_norm_2
//   result      = shared_out + expert_out
static ggml_tensor * build_moe_ffn(ggml_context * ctx,
                                   ggml_cgraph *  gf,
                                   const GemmaTargetWeights & w,
                                   const GemmaTargetLayer & L,
                                   ggml_tensor * cur_pre_ffn,
                                   ggml_tensor * cur_for_router,
                                   int n_tokens) {
    const int n_embd        = w.n_embd;
    const int n_expert_used = w.n_expert_used;
    const int n_expert      = w.n_expert;
    const int n_ff_exp      = w.n_ff_exp;

    // ── Shared expert (always active) ──────────────────────────────────────────
    ggml_tensor * shared_out = nullptr;
    if (L.w_gate && L.w_up && L.w_down) {
        ggml_tensor * sg  = ggml_mul_mat(ctx, L.w_gate, cur_pre_ffn);
        ggml_tensor * su  = ggml_mul_mat(ctx, L.w_up,   cur_pre_ffn);
        ggml_tensor * sgu = ggml_swiglu_split(ctx, sg, su);
        shared_out = ggml_mul_mat(ctx, L.w_down, sgu);
        if (L.ffn_post_norm_1) {
            shared_out = rms_norm_mul(ctx, shared_out, L.ffn_post_norm_1, EPS);
        }
    }

    // ── Router ─────────────────────────────────────────────────────────────────
    // router_in = rms_norm(inpSA) * ffn_pre_norm_2 / sqrt(n_embd)
    ggml_tensor * router_in = cur_for_router;
    if (L.ffn_pre_norm_2) {
        router_in = rms_norm_mul(ctx, router_in, L.ffn_pre_norm_2, EPS);
    }
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
    // weights: [1, n_expert_used, n_tokens]

    // ── Routed experts via ggml_mul_mat_id ─────────────────────────────────────
    ggml_tensor * expert_out = nullptr;
    if (L.ffn_gate_up_exps && L.ffn_down_exps) {
        // cur_pre_ffn is [n_embd, n_tokens]; mul_mat_id expects [n_embd, 1, n_tokens]
        ggml_tensor * x = ggml_reshape_3d(ctx, cur_pre_ffn, n_embd, 1, n_tokens);

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

        // SwiGLU activation (views are non-contiguous; ggml_silu requires contiguous)
        g_half = ggml_cont(ctx, g_half);
        u_half = ggml_cont(ctx, u_half);
        ggml_tensor * activated = ggml_mul(ctx, ggml_silu(ctx, g_half), u_half);

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
    return cur_pre_ffn;
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
        if (L.k_norm) {
            Kcur = rms_norm_mul(ctx, Kcur, L.k_norm, EPS);
        }
        Vcur = ggml_mul_mat(ctx, L.wv, cur);
        Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_head_kv, n_tokens);
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

    // Determine window start for SWA
    const int win_start = (w.swa_window > 0 && kv_start > w.swa_window)
                              ? (kv_start - w.swa_window) : 0;
    const int kv_len  = kv_start + n_tokens;
    const int win_len = kv_len - win_start;

    const bool need_256_pad = (kv_k_type == GGML_TYPE_TQ3_0 || kv_v_type == GGML_TYPE_TQ3_0
                               || head_dim >= 512);
    const int fattn_stride = need_256_pad ? 256 : 1;
    const int win_len_padded = ((win_len + fattn_stride - 1) / fattn_stride) * fattn_stride;

    ggml_tensor * Qfa = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
    Qfa = ggml_cont(ctx, Qfa);

    const bool q_rotate   = (kv_k_type == GGML_TYPE_TQ3_0);
    const bool out_rotate = (kv_v_type == GGML_TYPE_TQ3_0);
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

    // Gemma4: attn_scale = 1/sqrt(head_dim) (matches HF head_dim**-0.5)
    const float attn_scale_swa = 1.0f / std::sqrt((float)head_dim);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa, attn_mask,
                                             attn_scale_swa, 0.0f, 0.0f);

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
    int                        il)
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
        if (L.k_norm) {
            Kcur = rms_norm_mul(ctx, Kcur, L.k_norm, EPS);
        }
        if (L.wv == L.wk) {
            // Gemma4 full-attention: V = K (post-norm, pre-RoPE) — attention_k_eq_v=True
            Vcur = Kcur;
        } else {
            Vcur = ggml_mul_mat(ctx, L.wv, cur);
            Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_head_kv, n_tokens);
        }
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

    ggml_tensor * Qfa = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
    Qfa = ggml_cont(ctx, Qfa);

    const bool q_rotate   = (kv_k_type == GGML_TYPE_TQ3_0);
    const bool out_rotate = (kv_v_type == GGML_TYPE_TQ3_0);
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

    // Gemma4: attn_scale = 1/sqrt(head_dim) (matches HF head_dim**-0.5)
    const float attn_scale_full = 1.0f / std::sqrt((float)head_dim);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa, attn_mask,
                                             attn_scale_full, 0.0f, 0.0f);

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
                         GemmaTargetCache & out) {
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
    const int max_ctx_alloc = need_256_align
        ? ((max_ctx + 255) / 256) * 256
        : max_ctx;

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

        ggml_tensor * K = ggml_new_tensor_3d(out.base_ctx, kv_k_type,
                                             layer_head_dim, max_ctx_alloc, layer_n_head_kv);
        ggml_tensor * V = ggml_new_tensor_3d(out.base_ctx, kv_v_type,
                                             layer_head_dim, max_ctx_alloc, layer_n_head_kv);
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
    out.target_feat_cap = std::min(max_ctx, TARGET_FEAT_CAP_DEFAULT);
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
    if (c.base_buf) { ggml_backend_buffer_free(c.base_buf); c.base_buf = nullptr; }
    if (c.base_ctx) { ggml_free(c.base_ctx);                c.base_ctx = nullptr; }
    c.attn_k.clear();
    c.attn_v.clear();
    c.layer_to_kv_idx.clear();
    c.layer_to_donor_kv.clear();
    c.target_feat     = nullptr;
    c.cur_pos         = 0;
    c.last_tok        = -1;
}

void reset_gemma4_cache(GemmaTargetCache & c) {
    c.cur_pos  = 0;
    c.last_tok = -1;
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

        if (is_swa) {
            cur = build_swa_attn_block(ctx, gf, w, L, cur, in.positions,
                                       cache_k, cache_v, attn_mask,
                                       kv_start, n_tokens,
                                       cache.kv_k_type, cache.kv_v_type,
                                       write_kv, il);
        } else {
            cur = build_full_attn_block(ctx, gf, w, L, cur, in.positions,
                                        cache_k, cache_v, attn_mask,
                                        kv_start, n_tokens,
                                        cache.kv_k_type, cache.kv_v_type,
                                        write_kv, in.fa_window, il);
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
            // MoE path (26B-A4B)
            ffn_out = build_moe_ffn(ctx, gf, w, L,
                                    ffn_in, inpSA_post,
                                    n_tokens);
        } else {
            // Dense path (31B)
            ffn_out = build_swiglu_ffn(ctx, ffn_in, L);
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
