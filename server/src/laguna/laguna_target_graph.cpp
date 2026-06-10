// Hand-rolled CUDA forward graph for Poolside Laguna-XS.2 in dflash.
//
// Mirrors qwen35_target_graph.cpp's structure but for the laguna arch:
//   - 40 layers, hybrid iSWA: every 4th layer is FULL attention, rest are SWA(512)
//   - Per-layer head count: 48 (full) / 64 (SWA), 8 KV heads always, head_dim=128
//   - Q-norm + K-norm RMSNorm at head_dim level (Qwen3-style)
//   - Per-head SOFTPLUS attention gate: g_proj : hidden -> n_head, broadcast over head_dim
//   - Per-layer-type partial RoPE:
//       FULL: YaRN (theta=500K, factor=32, partial=0.5  -> n_rot=64)
//       SWA:  default (theta=10K, partial=1.0  -> n_rot=128)
//   - Layer 0: dense SwiGLU MLP (n_ff=8192). Layers 1..39: sparse MoE (256 top-8,
//     sigmoid router with score-correction bias, sum-normalize, scale=2.5) +
//     always-on shared expert SwiGLU (intermediate=512).
//
// Phase 2 status: cache lifecycle + attention blocks (full + SWA) + dense MLP
// + MoE block + layer dispatcher + full graph. Forward parity vs HF reference
// is tested against our llama.cpp build_laguna (already verified to match HF
// for 30+ tokens on B-tree prompt; see Lucebox/Laguna-XS.2-GGUF README).

#include "laguna_internal.h"
#include "../common/moe_hybrid_storage.h"
#include "common/ggml_graph_precision.h"
#include "internal.h"
#include "dflash27b.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <algorithm>

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"

namespace dflash::common {

static constexpr float LAGUNA_EPS = 1e-6f;

// ---- Cache lifecycle ----------------------------------------------------

bool create_laguna_target_cache(const LagunaTargetWeights & w,
                                 int max_ctx,
                                 ggml_backend_t backend,
                                 LagunaTargetCache & out) {
    return create_laguna_target_cache_partial(
        w, max_ctx, backend, /*layer_begin=*/0, /*layer_end=*/w.n_layer, out);
}

bool create_laguna_target_cache_partial(const LagunaTargetWeights & w,
                                         int max_ctx,
                                         ggml_backend_t backend,
                                         int layer_begin,
                                         int layer_end,
                                         LagunaTargetCache & out) {
    if (layer_begin < 0) layer_begin = 0;
    if (layer_end < 0) layer_end = w.n_layer;
    if (layer_begin > layer_end || layer_end > w.n_layer) {
        set_last_error("laguna cache: invalid layer range");
        return false;
    }

    out.backend  = backend;
    out.max_ctx  = max_ctx;
    out.cur_pos  = 0;
    out.last_tok = -1;
    // KV cache: per-layer, ALL 40 layers (full + SWA). Layout matches qwen35:
    //   [head_dim, max_ctx, n_head_kv]
    // dtype Q8_0 to halve VRAM vs F16.
    const ggml_type k_type = out.kv_k_type;
    const ggml_type v_type = out.kv_v_type;

    const size_t n_tensors_per_layer = 2;
    const size_t need_tensors = (size_t)w.n_layer * n_tensors_per_layer;

    ggml_init_params ip{};
    // Each tensor descriptor + overhead. Be generous.
    ip.mem_size = ggml_tensor_overhead() * (need_tensors + 16) + 4096;
    ip.no_alloc = true;
    out.base_ctx = ggml_init(ip);
    if (!out.base_ctx) { set_last_error("laguna cache: ggml_init failed"); return false; }

    out.attn_k.resize(w.n_layer, nullptr);
    out.attn_v.resize(w.n_layer, nullptr);
    for (int il = 0; il < w.n_layer; ++il) {
        if (il < layer_begin || il >= layer_end) continue;
        char nm[32];
        std::snprintf(nm, sizeof(nm), "k_l%d", il);
        ggml_tensor * k = ggml_new_tensor_3d(out.base_ctx, k_type, w.head_dim, max_ctx, w.n_head_kv);
        ggml_set_name(k, nm);
        std::snprintf(nm, sizeof(nm), "v_l%d", il);
        ggml_tensor * v = ggml_new_tensor_3d(out.base_ctx, v_type, w.head_dim, max_ctx, w.n_head_kv);
        ggml_set_name(v, nm);
        out.attn_k[il] = k;
        out.attn_v[il] = v;
    }

    out.base_buf = ggml_backend_alloc_ctx_tensors(out.base_ctx, backend);
    if (!out.base_buf) {
        set_last_error("laguna cache: ggml_backend_alloc_ctx_tensors failed");
        ggml_free(out.base_ctx); out.base_ctx = nullptr;
        return false;
    }

    // Zero-init KV (so reads before any write don't see garbage).
    const size_t buf_sz = ggml_backend_buffer_get_size(out.base_buf);
    std::vector<uint8_t> zeros(std::min<size_t>(buf_sz, 64 * 1024 * 1024), 0);
    for (int il = 0; il < w.n_layer; ++il) {
        for (auto * t : { out.attn_k[il], out.attn_v[il] }) {
            if (!t) continue;
            const size_t sz = ggml_nbytes(t);
            for (size_t off = 0; off < sz; off += zeros.size()) {
                const size_t chunk = std::min(zeros.size(), sz - off);
                ggml_backend_tensor_set(t, zeros.data(), off, chunk);
            }
        }
    }
    return true;
}

// ---- Cache snapshot helpers (prefix-cache slots) ------------------------
//
// laguna_snapshot_alloc: build a parallel set of K/V tensors RIGHT-SIZED to
// snap_pos positions (not full max_ctx). Position dimension is ne[1].
bool laguna_snapshot_alloc(const LagunaTargetCache & cache,
                            ggml_backend_t            backend,
                            int                       n_layer,
                            int                       snap_pos,
                            int                       n_head_kv,
                            int                       head_dim,
                            LagunaCacheSnapshot &     out) {
    if (out.ctx) return true;
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (size_t)(n_layer * 2 + 16) + 4096;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) { set_last_error("snapshot: ggml_init failed"); return false; }
    out.attn_k.assign((size_t)n_layer, nullptr);
    out.attn_v.assign((size_t)n_layer, nullptr);
    for (int il = 0; il < n_layer; ++il) {
        if (!cache.attn_k[il] || !cache.attn_v[il]) continue;
        char nm[32];
        std::snprintf(nm, sizeof(nm), "snap_k_l%d", il);
        // Right-sized: [head_dim, snap_pos, n_head_kv]
        ggml_tensor * k = ggml_new_tensor_3d(out.ctx, cache.kv_k_type, head_dim, snap_pos, n_head_kv);
        ggml_set_name(k, nm);
        std::snprintf(nm, sizeof(nm), "snap_v_l%d", il);
        ggml_tensor * v = ggml_new_tensor_3d(out.ctx, cache.kv_v_type, head_dim, snap_pos, n_head_kv);
        ggml_set_name(v, nm);
        out.attn_k[il] = k;
        out.attn_v[il] = v;
    }
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        set_last_error("snapshot: ggml_backend_alloc_ctx_tensors failed");
        ggml_free(out.ctx); out.ctx = nullptr;
        return false;
    }
    out.cur_pos = 0;
    out.used    = false;
    return true;
}

void laguna_snapshot_free(LagunaCacheSnapshot & snap) {
    if (snap.buf) { ggml_backend_buffer_free(snap.buf); snap.buf = nullptr; }
    if (snap.ctx) { ggml_free(snap.ctx); snap.ctx = nullptr; }
    snap.attn_k.clear();
    snap.attn_v.clear();
    snap.cur_pos = 0;
    snap.used    = false;
}

// Save cache → snapshot. Handles alloc/realloc internally.
bool laguna_snapshot_save(const LagunaTargetCache & cache,
                           ggml_backend_t            backend,
                           int                       n_layer,
                           int                       n_head_kv,
                           int                       head_dim,
                           LagunaCacheSnapshot &     snap) {
    const int snap_pos = cache.cur_pos;
    if (snap_pos <= 0) {
        set_last_error("snapshot_save: cur_pos <= 0");
        return false;
    }

    // Realloc if shapes don't match (different cur_pos).
    if (snap.ctx && snap.cur_pos != snap_pos) {
        laguna_snapshot_free(snap);
    }
    if (!snap.ctx) {
        if (!laguna_snapshot_alloc(cache, backend, n_layer, snap_pos, n_head_kv, head_dim, snap)) {
            return false;
        }
    }

    // Copy KV strip-by-strip (right-sized snapshot, position dim = ne[1]).
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * sk = cache.attn_k[il];
        ggml_tensor * dk = snap.attn_k[il];
        ggml_tensor * sv = cache.attn_v[il];
        ggml_tensor * dv = snap.attn_v[il];
        if (!sk || !dk || !sv || !dv) continue;
        const size_t k_strip = (size_t)snap_pos * sk->nb[1];
        const size_t v_strip = (size_t)snap_pos * sv->nb[1];
        for (int kh = 0; kh < n_head_kv; kh++) {
            size_t src_off = (size_t)kh * sk->nb[2];
            size_t dst_off = (size_t)kh * dk->nb[2];
            ggml_backend_tensor_get(sk, (char *)dk->data + dst_off, src_off, k_strip);
        }
        for (int kh = 0; kh < n_head_kv; kh++) {
            size_t src_off = (size_t)kh * sv->nb[2];
            size_t dst_off = (size_t)kh * dv->nb[2];
            ggml_backend_tensor_get(sv, (char *)dv->data + dst_off, src_off, v_strip);
        }
    }
    snap.cur_pos = snap_pos;
    snap.used    = true;
    return true;
}

bool laguna_snapshot_restore(const LagunaCacheSnapshot & snap,
                              LagunaTargetCache &         cache) {
    if (!snap.used || snap.attn_k.size() != cache.attn_k.size()) {
        set_last_error("snapshot_restore: snapshot unused or layer count mismatch");
        return false;
    }
    const int snap_pos = snap.cur_pos;
    // Copy right-sized snapshot back into full-size cache, strip-by-strip.
    for (size_t il = 0; il < cache.attn_k.size(); ++il) {
        ggml_tensor * sk = snap.attn_k[il];
        ggml_tensor * dk = cache.attn_k[il];
        ggml_tensor * sv = snap.attn_v[il];
        ggml_tensor * dv = cache.attn_v[il];
        if (!sk || !dk || !sv || !dv) continue;
        const size_t k_strip = (size_t)snap_pos * sk->nb[1];
        const size_t v_strip = (size_t)snap_pos * sv->nb[1];
        for (int kh = 0; kh < (int)sk->ne[2]; kh++) {
            size_t src_off = (size_t)kh * sk->nb[2];
            size_t dst_off = (size_t)kh * dk->nb[2];
            ggml_backend_tensor_set(dk, (const char *)sk->data + src_off, dst_off, k_strip);
        }
        for (int kh = 0; kh < (int)sv->ne[2]; kh++) {
            size_t src_off = (size_t)kh * sv->nb[2];
            size_t dst_off = (size_t)kh * dv->nb[2];
            ggml_backend_tensor_set(dv, (const char *)sv->data + src_off, dst_off, v_strip);
        }
    }
    cache.cur_pos = snap_pos;
    return true;
}

void free_laguna_target_cache(LagunaTargetCache & c) {
    if (c.base_buf) { ggml_backend_buffer_free(c.base_buf); c.base_buf = nullptr; }
    if (c.base_ctx) { ggml_free(c.base_ctx);                c.base_ctx = nullptr; }
    c.attn_k.clear();
    c.attn_v.clear();
}

void reset_laguna_target_cache(LagunaTargetCache & c) {
    c.cur_pos  = 0;
    c.last_tok = -1;
    if (!c.base_ctx) return;
    std::vector<uint8_t> zeros(64 * 1024 * 1024, 0);
    for (auto * t : c.attn_k) {
        if (!t) continue;
        const size_t sz = ggml_nbytes(t);
        for (size_t off = 0; off < sz; off += zeros.size()) {
            const size_t chunk = std::min(zeros.size(), sz - off);
            ggml_backend_tensor_set(t, zeros.data(), off, chunk);
        }
    }
    for (auto * t : c.attn_v) {
        if (!t) continue;
        const size_t sz = ggml_nbytes(t);
        for (size_t off = 0; off < sz; off += zeros.size()) {
            const size_t chunk = std::min(zeros.size(), sz - off);
            ggml_backend_tensor_set(t, zeros.data(), off, chunk);
        }
    }
}

// ---- Helpers ------------------------------------------------------------

static ggml_tensor * laguna_rms_norm_mul(ggml_context * ctx, ggml_tensor * x,
                                          ggml_tensor * weight, float eps = LAGUNA_EPS) {
    x = rms_norm_input_f32(ctx, x);
    weight = graph_tensor_f32(ctx, weight);
    ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, weight);
}

static ggml_tensor * build_laguna_dense_ffn(ggml_context * ctx, ggml_tensor * cur,
                                              const LagunaTargetLayer & L) {
    // SwiGLU: down( silu(gate(x)) * up(x) )
    ggml_tensor * gate = ggml_mul_mat(ctx, L.w_gate, cur);   // [n_ff, n_tokens]
    ggml_tensor * up   = ggml_mul_mat(ctx, L.w_up,   cur);   // [n_ff, n_tokens]
    ggml_tensor * gu   = ggml_swiglu_split(ctx, gate, up);
    return ggml_mul_mat(ctx, L.w_down, gu);                  // [n_embd, n_tokens]
}

// Forward decl for the full MoE block (defined further down).
static ggml_tensor * build_laguna_moe_block_full(ggml_context * ctx, ggml_tensor * cur,
                                                  const LagunaTargetWeights & w,
                                                  const LagunaTargetLayer & L);
// Forward decl for the hybrid (offload) MoE block (defined further down).
static ggml_tensor * build_laguna_moe_block_hybrid(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * cur,
                                                   const LagunaTargetWeights & w,
                                                   const LagunaTargetLayer & L,
                                                   const MoeHybridLayerStorage & hot,
                                                   ggml_tensor * lut_all, ggml_tensor * vld_all,
                                                   ggml_tensor * sel_all, int moe_idx);

// MoE block: sigmoid router with score-correction bias, sum-normalize selected
// weights, scale routed combine by expert_weights_scale (=2.5 for Laguna),
// add always-on shared expert SwiGLU. Matches modeling_laguna.LagunaSparseMoeBlock.
//
// `cur` shape [n_embd, n_tokens]. Returns [n_embd, n_tokens].
// PHASE 2.0 STUB: returns ONLY the shared expert (no routed dispatch). Lets us
// validate attention path + dense MLP path on layer 0 + final norm + lm_head.
// Numerically wrong (drops routed MoE contribution = ~80% of MLP signal) but
// graph builds + executes. Routed dispatch (sigmoid router + score-correction
// bias + sum-norm + ggml_mul_mat_id) is Phase 2.1.
// DEBUG SWITCH: env DFLASH_LAGUNA_MOE_STUB=1 routes to shared-only stub.
// Default: full MoE.
static ggml_tensor * build_laguna_moe_block(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * cur,
                                             const LagunaTargetWeights & w,
                                             const LagunaTargetLayer & L,
                                             const LagunaHybridMoe * hyb = nullptr, int il = 0) {
    static const bool stub = (std::getenv("DFLASH_LAGUNA_MOE_STUB") != nullptr);
    if (stub) {
        ggml_tensor * sh_gate = ggml_mul_mat(ctx, L.ffn_gate_shexp, cur);
        ggml_tensor * sh_up   = ggml_mul_mat(ctx, L.ffn_up_shexp,   cur);
        ggml_tensor * sh_gu   = ggml_swiglu_split(ctx, sh_gate, sh_up);
        return ggml_mul_mat(ctx, L.ffn_down_shexp, sh_gu);
    }
    if (hyb && hyb->storage) {
        return build_laguna_moe_block_hybrid(ctx, gf, cur, w, L,
            hyb->storage->layers[(size_t)il], hyb->lut_all, hyb->vld_all, hyb->sel_all,
            il - hyb->dense_lead);
    }
    return build_laguna_moe_block_full(ctx, cur, w, L);
}

// Phase 2.1: full MoE dispatch (sigmoid + score-correction bias + sum-norm +
// scale 2.5 + always-on shared expert). Mirrors llama.cpp's build_moe_ffn for
// the SIGMOID + WEIGHTS_NORM + EXP_PROBS_B configuration that Laguna uses.
static ggml_tensor * build_laguna_moe_block_full(ggml_context * ctx, ggml_tensor * cur,
                                                  const LagunaTargetWeights & w,
                                                  const LagunaTargetLayer & L) {
    const int n_tokens = (int)cur->ne[1];
    const int n_expert = w.n_expert;
    const int n_used   = w.n_expert_used;
    const int n_embd   = w.n_embd;

    // Router logits + sigmoid
    ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, cur);  // [n_expert, n_tokens]
    ggml_tensor * probs  = ggml_sigmoid(ctx, logits);

    // Add score-correction bias for SELECTION (not for combine weights).
    ggml_tensor * scores_sel = ggml_add(ctx, probs, L.ffn_exp_probs_b);

    // Top-k selection: indices [n_used, n_tokens] i32.
    ggml_tensor * selected = ggml_top_k(ctx, scores_sel, n_used);

    // Gather ORIGINAL probs (no bias) at the selected indices for combine weights.
    // Trick: reshape probs to [1, n_expert, n_tokens] so ggml_get_rows treats
    // the expert axis as the row axis. Output shape [1, n_used, n_tokens].
    ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);
    ggml_tensor * weights  = ggml_get_rows(ctx, probs_3d, selected);
    weights = ggml_reshape_2d(ctx, weights, n_used, n_tokens);

    // Sum-normalize selected weights (Laguna sets expert_weights_norm=true).
    ggml_tensor * w_sum = ggml_sum_rows(ctx, weights);  // [1, n_tokens]
    weights = ggml_div(ctx, weights, w_sum);

    // Scale routed combine.
    if (w.expert_weights_scale != 1.0f) {
        weights = ggml_scale(ctx, weights, w.expert_weights_scale);
    }

    // Per-expert SwiGLU via mul_mat_id.
    //   ffn_gate_exps: [n_embd, n_ff_exp, n_expert]
    //   ffn_up_exps:   [n_embd, n_ff_exp, n_expert]
    //   selected:      [n_used, n_tokens] i32  (ids->ne = {n_used, n_tokens, 1, 1})
    // ggml_mul_mat_id requires b->ne[2] == ids->ne[1]. cur is [n_embd, n_tokens]
    // (ne[2]=1) so reshape to [n_embd, 1, n_tokens] (ne[2]=n_tokens). Same trick
    // llama.cpp's build_moe_ffn uses.
    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
    ggml_tensor * gate_e = ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur_3d, selected);
    ggml_tensor * up_e   = ggml_mul_mat_id(ctx, L.ffn_up_exps,   cur_3d, selected);
    ggml_tensor * gu     = ggml_swiglu_split(ctx, gate_e, up_e);
    //   ffn_down_exps: [n_ff_exp, n_embd, n_expert]
    //   gu:            [n_ff_exp, n_used, n_tokens]   (ne[2] = n_tokens)
    //   experts out:   [n_embd, n_used, n_tokens]
    ggml_tensor * experts = ggml_mul_mat_id(ctx, L.ffn_down_exps, gu, selected);

    // Multiply per-expert outputs by their routing weights.
    //   experts: [n_embd, n_used, n_tokens]
    //   weights: [n_used, n_tokens] -> view as [1, n_used, n_tokens] for broadcast
    ggml_tensor * w_view = ggml_reshape_3d(ctx, weights, 1, n_used, n_tokens);
    experts = ggml_mul(ctx, experts, w_view);

    // Sum across the n_used axis: explicit slice + add loop (matches llama.cpp
    // pattern; ggml_sum_rows would sum over dim 0 which is n_embd, wrong).
    ggml_tensor * routed = nullptr;
    for (int i = 0; i < n_used; ++i) {
        ggml_tensor * slice = ggml_view_2d(ctx, experts,
            n_embd, n_tokens,
            experts->nb[2],
            (size_t)i * experts->nb[1]);
        routed = (i == 0) ? slice : ggml_add(ctx, routed, slice);
    }

    // Always-on shared expert (SwiGLU).
    ggml_tensor * sh_gate = ggml_mul_mat(ctx, L.ffn_gate_shexp, cur);
    ggml_tensor * sh_up   = ggml_mul_mat(ctx, L.ffn_up_shexp,   cur);
    ggml_tensor * sh_gu   = ggml_swiglu_split(ctx, sh_gate, sh_up);
    ggml_tensor * shared  = ggml_mul_mat(ctx, L.ffn_down_shexp, sh_gu);

    return ggml_add(ctx, routed, shared);
}

// Hybrid MoE block: same router as build_laguna_moe_block_full, but the routed
// experts are served from the bounded hot stack via a global->local LUT, and a
// validity mask drops (zeroes the combine weight of) any selected expert that
// is not currently resident. `lut`/`vld` are graph inputs set once per token by
// laguna_step_hybrid from the cache residency state.
static ggml_tensor * build_laguna_moe_block_hybrid(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * cur,
                                                    const LagunaTargetWeights & w,
                                                    const LagunaTargetLayer & L,
                                                    const MoeHybridLayerStorage & hot,
                                                    ggml_tensor * lut_all,
                                                    ggml_tensor * vld_all,
                                                    ggml_tensor * sel_all,
                                                    int moe_idx) {
    const int n_tokens = (int)cur->ne[1];
    const int n_expert = w.n_expert;
    const int n_used   = w.n_expert_used;
    const int n_embd   = w.n_embd;

    ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, cur);
    ggml_tensor * probs  = ggml_sigmoid(ctx, logits);
    ggml_tensor * scores_sel = ggml_add(ctx, probs, L.ffn_exp_probs_b);
    ggml_tensor * selected = ggml_top_k(ctx, scores_sel, n_used);  // [n_used, n_tokens] global ids
    {   // batched readback: write this layer's selection into column moe_idx of sel_all
        ggml_tensor * sel_col = ggml_view_2d(ctx, sel_all, n_used, n_tokens,
                                             sel_all->nb[1], (size_t)moe_idx * sel_all->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, selected, sel_col));
    }

    ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);
    ggml_tensor * weights  = ggml_get_rows(ctx, probs_3d, selected);
    weights = ggml_reshape_2d(ctx, weights, n_used, n_tokens);
    ggml_tensor * w_sum = ggml_sum_rows(ctx, weights);
    weights = ggml_div(ctx, weights, w_sum);
    if (w.expert_weights_scale != 1.0f) {
        weights = ggml_scale(ctx, weights, w.expert_weights_scale);
    }

    // Per-layer residency LUT/valid = column moe_idx of the shared input tensors.
    ggml_tensor * lut = ggml_reshape_2d(ctx, ggml_view_1d(ctx, lut_all, n_expert, (size_t)moe_idx * lut_all->nb[1]), 1, n_expert);
    ggml_tensor * vld = ggml_reshape_2d(ctx, ggml_view_1d(ctx, vld_all, n_expert, (size_t)moe_idx * vld_all->nb[1]), 1, n_expert);

    ggml_tensor * lid = ggml_get_rows(ctx, lut, selected);          // [1, n_used, n_tokens]
    ggml_tensor * ids = ggml_cont(ctx, ggml_reshape_2d(ctx, lid, n_used, n_tokens));
    ggml_tensor * vm  = ggml_get_rows(ctx, vld, selected);          // [1, n_used, n_tokens]
    vm = ggml_reshape_2d(ctx, vm, n_used, n_tokens);
    weights = ggml_mul(ctx, weights, vm);                          // drop non-resident experts

    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
    ggml_tensor * gate_e = ggml_mul_mat_id(ctx, hot.gate_hot, cur_3d, ids);
    ggml_tensor * up_e   = ggml_mul_mat_id(ctx, hot.up_hot,   cur_3d, ids);
    ggml_tensor * gu     = ggml_swiglu_split(ctx, gate_e, up_e);
    ggml_tensor * experts = ggml_mul_mat_id(ctx, hot.down_hot, gu, ids);

    ggml_tensor * w_view = ggml_reshape_3d(ctx, weights, 1, n_used, n_tokens);
    experts = ggml_mul(ctx, experts, w_view);

    ggml_tensor * routed = nullptr;
    for (int i = 0; i < n_used; ++i) {
        ggml_tensor * slice = ggml_view_2d(ctx, experts, n_embd, n_tokens,
            experts->nb[2], (size_t)i * experts->nb[1]);
        routed = (i == 0) ? slice : ggml_add(ctx, routed, slice);
    }

    ggml_tensor * sh_gate = ggml_mul_mat(ctx, L.ffn_gate_shexp, cur);
    ggml_tensor * sh_up   = ggml_mul_mat(ctx, L.ffn_up_shexp,   cur);
    ggml_tensor * sh_gu   = ggml_swiglu_split(ctx, sh_gate, sh_up);
    ggml_tensor * shared  = ggml_mul_mat(ctx, L.ffn_down_shexp, sh_gu);

    return ggml_add(ctx, routed, shared);
}

static ggml_tensor * build_laguna_moe_block_legacy(ggml_context * ctx, ggml_tensor * cur,
                                                    const LagunaTargetWeights & w,
                                                    const LagunaTargetLayer & L) {
    // ggml_build_moe is the standard helper used by deepseek/qwen3 MoE in
    // llama.cpp. We need an equivalent. ggml's ggml_mul_mat_id implements the
    // grouped per-expert matmul; the router/topk logic must be done via op
    // composition. Match llama.cpp's build_moe_ffn semantics.
    //
    //   logits = ffn_gate_inp @ x                    # [n_expert, n_tokens]
    //   probs  = sigmoid(logits)                     # SIGMOID gating
    //   sel_scores = probs + exp_probs_b             # bias-corrected for SELECTION
    //   topk = argtopk(sel_scores, n_expert_used)    # indices [topk, n_tokens]
    //   weights = gather(probs, topk)                # ORIGINAL probs (no bias) for combine
    //   weights = weights / sum(weights, axis=-2, keepdim=True)
    //   y_routed = sum_e weights[e] * down_e(silu(gate_e(x)) * up_e(x))
    //   y_routed = y_routed * scale
    //   y_shared = down_sh(silu(gate_sh(x)) * up_sh(x))
    //   return y_routed + y_shared
    //
    // ggml provides this composition via ggml_top_k + ggml_mul_mat_id. To keep
    // the file self-contained and avoid extending ggml here, route through
    // existing ops:

    // Router logits: [n_expert, n_tokens]
    ggml_tensor * router_logits = ggml_mul_mat(ctx, L.ffn_gate_inp, cur);
    // Sigmoid (Laguna router uses sigmoid, not softmax)
    ggml_tensor * probs = ggml_sigmoid(ctx, router_logits);

    // For selection only, add the score-correction bias. Don't use bias-added
    // values for the combine weights.
    //   bias is [n_expert]; broadcast over n_tokens.
    ggml_tensor * scores_for_sel = ggml_add(ctx, probs, L.ffn_exp_probs_b);

    // Top-k indices: [n_expert_used, n_tokens] i32.
    // ggml_top_k returns argmax indices into the n_expert axis.
    ggml_tensor * selected = ggml_top_k(ctx, scores_for_sel, w.n_expert_used);

    // Gather the ORIGINAL probs at the selected indices for combine weights.
    // ggml_get_rows would treat probs's first dim as rows; here we want to
    // index into the n_expert axis per-token, which matches the standard MoE
    // pattern in llama.cpp's build_moe_ffn (see deepseek path).
    ggml_tensor * weights = ggml_get_rows(ctx, probs, selected);
    // Reshape to [n_expert_used, n_tokens] (ggml_get_rows yields [..., 1]; squeeze).
    weights = ggml_reshape_2d(ctx, weights, w.n_expert_used, ggml_nelements(weights) / w.n_expert_used);

    // Sum-normalize selected weights along expert axis.
    ggml_tensor * w_sum = ggml_sum_rows(ctx, weights);  // [1, n_tokens]
    weights = ggml_div(ctx, weights, w_sum);

    // Apply the routed scaling factor.
    if (w.expert_weights_scale != 1.0f) {
        weights = ggml_scale(ctx, weights, w.expert_weights_scale);
    }

    // Routed expert gate+up (fused SwiGLU): use mul_mat_id to dispatch.
    //   ffn_gate_exps: [n_embd, n_ff_exp, n_expert]
    //   ffn_up_exps:   [n_embd, n_ff_exp, n_expert]
    //   ffn_down_exps: [n_ff_exp, n_embd, n_expert]
    // ggml_mul_mat_id expects ids tensor [n_expert_used, n_tokens] (we have it).
    ggml_tensor * gate_e = ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur, selected);
    ggml_tensor * up_e   = ggml_mul_mat_id(ctx, L.ffn_up_exps,   cur, selected);
    ggml_tensor * gu     = ggml_swiglu_split(ctx, gate_e, up_e);
    ggml_tensor * routed = ggml_mul_mat_id(ctx, L.ffn_down_exps, gu, selected);

    // Multiply per-expert outputs by their routing weights and combine.
    // mul_mat_id gives [n_embd, n_expert_used, n_tokens]. Broadcast multiply
    // by weights [n_expert_used, n_tokens] -> need shape match. Reshape weights
    // to [1, n_expert_used, n_tokens] via view.
    ggml_tensor * w_view = ggml_reshape_3d(ctx, weights, 1, w.n_expert_used,
                                            ggml_nelements(weights) / w.n_expert_used);
    routed = ggml_mul(ctx, routed, w_view);
    routed = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_permute(ctx, routed, 0, 2, 1, 3)));
    // Now [n_embd, n_tokens, 1]. Reshape to [n_embd, n_tokens].
    routed = ggml_reshape_2d(ctx, routed, w.n_embd, ggml_nelements(routed) / w.n_embd);

    // Shared expert (always on).
    ggml_tensor * sh_gate = ggml_mul_mat(ctx, L.ffn_gate_shexp, cur);
    ggml_tensor * sh_up   = ggml_mul_mat(ctx, L.ffn_up_shexp,   cur);
    ggml_tensor * sh_gu   = ggml_swiglu_split(ctx, sh_gate, sh_up);
    ggml_tensor * shared  = ggml_mul_mat(ctx, L.ffn_down_shexp, sh_gu);

    return ggml_add(ctx, routed, shared);
}

// Attention block. Handles BOTH full and SWA layers; the only differences are
//   - n_head[il] (per-layer)
//   - rope_freq_base + n_rot (per layer-type)
//   - YaRN params (full only) vs default rope (swa)
//   - sliding window mask (swa only; passed in via attn_mask)
//
// Per-head softplus gate is applied AFTER the FA output (broadcast over head_dim)
// and BEFORE the o_proj.
static ggml_tensor * build_laguna_attn_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const LagunaTargetWeights & w,
    const LagunaTargetLayer & L,
    int il,
    ggml_tensor * cur,
    ggml_tensor * positions,
    ggml_tensor * cache_k,
    ggml_tensor * cache_v,
    ggml_tensor * attn_mask,
    ggml_tensor * attn_mask_swa,
    int kv_start,
    int n_tokens,
    bool is_full,
    int kv_pad = 0,
    ggml_tensor * kv_idx = nullptr)
{
    const int head_dim   = w.head_dim;
    const int n_head     = w.n_head_arr[il];
    const int n_head_kv  = w.n_head_kv;
    const int q_dim      = n_head * head_dim;

    // ---- Q/K/V projections ---
    ggml_tensor * Qcur = ggml_mul_mat(ctx, L.wq, cur);  // [q_dim, n_tokens]
    ggml_tensor * Kcur = ggml_mul_mat(ctx, L.wk, cur);  // [n_head_kv*head_dim, n_tokens]
    ggml_tensor * Vcur = ggml_mul_mat(ctx, L.wv, cur);

    Qcur = ggml_reshape_3d(ctx, Qcur, head_dim, n_head,    n_tokens);
    Kcur = ggml_reshape_3d(ctx, Kcur, head_dim, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_head_kv, n_tokens);

    // ---- Per-head Q/K RMSNorm (norm over head_dim) ---
    Qcur = laguna_rms_norm_mul(ctx, Qcur, L.q_norm);
    Kcur = laguna_rms_norm_mul(ctx, Kcur, L.k_norm);

    // ---- Per-head softplus attention gate ---
    // wqkv_gate : [n_embd, n_head]; gate_proj output [n_head, n_tokens] f32.
    ggml_tensor * gate_logits = ggml_mul_mat(ctx, L.wqkv_gate, cur); // [n_head, n_tokens]
    // Cast to f32 to match HF reference (computed in fp32, then cast back).
    gate_logits = ggml_cast(ctx, gate_logits, GGML_TYPE_F32);
    ggml_tensor * gate = ggml_softplus(ctx, gate_logits);            // [n_head, n_tokens] f32

    // ---- Partial RoPE (NeoX layout: rotate first n_rot dims, leave the rest) ---
    const int n_rot     = is_full ? w.n_rot_full : w.n_rot_swa;
    const float rope_th = is_full ? w.rope_freq_base_full : w.rope_freq_base_swa;
    // YaRN params: only on full layers (sliding uses default rope, attn_factor=0
    // makes ggml_rope_ext fall back to plain math).
    const float ext_factor  = is_full ? 1.0f : 0.0f;
    const float attn_factor = is_full ? 1.0f : 1.0f;
    const float beta_fast   = is_full ? w.yarn_beta_fast : 32.0f;
    const float beta_slow   = is_full ? w.yarn_beta_slow :  1.0f;
    const int   n_ctx_orig  = is_full ? w.yarn_orig_ctx  : 0;
    const float freq_scale  = is_full ? (1.0f / w.yarn_factor) : 1.0f;

    Qcur = ggml_rope_ext(ctx, Qcur, positions, /*freq_factors=*/nullptr,
                          n_rot, /*mode=*/GGML_ROPE_TYPE_NEOX,
                          n_ctx_orig, rope_th, freq_scale,
                          ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_ext(ctx, Kcur, positions, nullptr,
                          n_rot, GGML_ROPE_TYPE_NEOX,
                          n_ctx_orig, rope_th, freq_scale,
                          ext_factor, attn_factor, beta_fast, beta_slow);

    // ---- Write K/V to cache slot ---
    // All layers (full + SWA) use a uniform max_ctx-sized cache. SWA layers
    // pay the memory cost but the FA call still only reads `sliding_window`
    // entries via the windowed view below. Per-layer-size optimization (SWA
    // ring buffer to halve KV memory) requires careful chunk sizing and is
    // deferred (see git history for an in-progress version).
    ggml_tensor * Kcur_T = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
    ggml_tensor * Vcur_T = ggml_permute(ctx, Vcur, 0, 2, 1, 3);

    if (kv_idx) {
        // CUDA-graph-stable append: the destination is the WHOLE cache tensor
        // (stable data pointer) and the row index is a graph input whose DATA
        // changes per step but whose pointer doesn't. A kv_start-offset view
        // (below) changes node properties every step, which resets the
        // ggml-cuda CUDA-graph warmup and forfeits replay.
        // kv_idx [n_tokens] broadcasts over the n_head_kv dim (ggml_set_rows
        // requires b->ne[2] % c->ne[1] == 0).
        ggml_tensor * Krows = ggml_cont(ctx, Kcur_T);  // set_rows needs contiguous rows
        ggml_tensor * Vrows = ggml_cont(ctx, Vcur_T);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_k, Krows, kv_idx));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_v, Vrows, kv_idx));
    } else {
        ggml_tensor * k_slot = ggml_view_3d(ctx, cache_k,
            head_dim, n_tokens, n_head_kv,
            cache_k->nb[1], cache_k->nb[2],
            cache_k->nb[1] * (size_t)kv_start);
        ggml_tensor * v_slot = ggml_view_3d(ctx, cache_v,
            head_dim, n_tokens, n_head_kv,
            cache_v->nb[1], cache_v->nb[2],
            cache_v->nb[1] * (size_t)kv_start);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur_T, k_slot));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur_T, v_slot));
    }

    // ---- Flash attention ---
    // BOTH full and SWA layers read the FULL kv_len of K/V from the cache.
    // For SWA layers, the per-row sliding-window causal constraint is enforced
    // by attn_mask_swa (built by the caller). This is correct for the early-
    // token rows (which need to see KV positions [0..p+1)) and the late-token
    // rows (which need [p-sw+1..p+1)).
    const int kv_len   = kv_start + n_tokens;
    const int win_start = 0;
    // kv_pad > 0: read a stride-rounded fixed span so the view shape (and thus
    // every downstream FA node's properties) stays constant across decode
    // steps; the mask carries -inf for [kv_len, kv_pad) and the cache buffer
    // is zero-initialised, so the padded tail contributes exactly nothing.
    const int win_len   = kv_pad > 0 ? kv_pad : kv_len;

    ggml_tensor * Qfa = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
    Qfa = ggml_cont(ctx, Qfa);

    ggml_tensor * Kfa = ggml_view_3d(ctx, cache_k,
        head_dim, win_len, n_head_kv,
        cache_k->nb[1], cache_k->nb[2], cache_k->nb[1] * (size_t)win_start);
    ggml_tensor * Vfa = ggml_view_3d(ctx, cache_v,
        head_dim, win_len, n_head_kv,
        cache_v->nb[1], cache_v->nb[2], cache_v->nb[1] * (size_t)win_start);

    const float kq_scale = 1.0f / std::sqrt((float)head_dim);
    // FULL -> attn_mask (causal). SWA -> attn_mask_swa (causal + sliding-window).
    ggml_tensor * use_mask = is_full ? attn_mask : attn_mask_swa;
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa, use_mask,
                                              kq_scale, 0.0f, 0.0f);
    (void)win_start; (void)win_len;
    // attn: [head_dim, n_head, n_tokens]

    // Per-head softplus gate broadcast over head_dim:
    //   attn[d, h, t] *= gate[h, t]
    // Reshape gate to [1, n_head, n_tokens] so broadcast works.
    ggml_tensor * gate_b = ggml_reshape_3d(ctx, gate, 1, n_head, n_tokens);
    // Cast gate back to attn dtype to keep mul homogeneous.
    gate_b = ggml_cast(ctx, gate_b, attn->type);
    attn = ggml_mul(ctx, attn, gate_b);

    attn = ggml_reshape_2d(ctx, attn, q_dim, n_tokens);

    // ---- Output projection ---
    return ggml_mul_mat(ctx, L.wo, attn);  // [n_embd, n_tokens]
}

// ---- Layer dispatch ----------------------------------------------------

static ggml_tensor * build_laguna_layer(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const LagunaTargetWeights & w,
    LagunaTargetCache & cache,
    int il,
    ggml_tensor * inp,
    ggml_tensor * positions,
    ggml_tensor * attn_mask,
    int kv_start,
    int n_tokens,
    ggml_tensor * attn_mask_swa,
    const LagunaHybridMoe * hyb = nullptr,
    int kv_pad = 0,
    ggml_tensor * kv_idx = nullptr)
{
    const LagunaTargetLayer & L = w.layers[il];
    ggml_tensor * inp_f32 = graph_tensor_f32(ctx, inp);

    // Pre-attn norm
    ggml_tensor * cur = laguna_rms_norm_mul(ctx, inp_f32, L.attn_norm);

    // Attention
    const bool is_full = laguna_is_full_attn_layer(w, il);
    cur = build_laguna_attn_block(ctx, gf, w, L, il, cur,
                                    positions, cache.attn_k[il], cache.attn_v[il],
                                    attn_mask, attn_mask_swa, kv_start, n_tokens, is_full,
                                    kv_pad, kv_idx);

    // Residual
    ggml_tensor * ffn_inp = ggml_add(ctx, cur, inp_f32);

    // Pre-FFN norm
    cur = laguna_rms_norm_mul(ctx, ffn_inp, L.ffn_norm);

    // Dense MLP (layer 0) or sparse MoE+shared (layers 1..n)
    const bool is_dense = (il < w.n_layer_dense_lead);
    cur = is_dense ? build_laguna_dense_ffn(ctx, cur, L)
                   : build_laguna_moe_block(ctx, gf, cur, w, L, hyb, il);

    return ggml_add(ctx, cur, ffn_inp);
}

void laguna_layer_step_graph_free(LagunaLayerStepGraph & sg) {
    if (sg.ctx) {
        ggml_free(sg.ctx);
        sg.ctx = nullptr;
    }
    sg.gf = nullptr;
    sg.positions = nullptr;
    sg.attn_mask = nullptr;
    sg.attn_mask_swa = nullptr;
}

void laguna_layer_step_graph_destroy(LagunaLayerStepGraph & sg) {
    if (sg.alloc) {
        ggml_gallocr_free(sg.alloc);
        sg.alloc = nullptr;
    }
    laguna_layer_step_graph_free(sg);
}

bool build_laguna_layer_step(
    LagunaLayerStepGraph & sg,
    const LagunaTargetWeights & w,
    LagunaTargetCache & cache,
    ggml_backend_t backend,
    int layer_idx,
    ggml_tensor * act_in,
    ggml_tensor * act_out,
    int chunk_start,
    int n_tokens,
    int kv_start) {
    laguna_layer_step_graph_free(sg);
    if (layer_idx < 0 || layer_idx >= w.n_layer) return false;
    if (!cache.attn_k[layer_idx] || !cache.attn_v[layer_idx]) return false;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() + 16 * 1024 * 1024;
    ip.no_alloc = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;
    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    ggml_tensor * inp = ggml_view_2d(
        sg.ctx, act_in, w.n_embd, n_tokens,
        act_in->nb[1], (size_t)chunk_start * act_in->nb[1]);
    ggml_set_input(inp);

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(sg.positions);

    const int kv_len = kv_start + n_tokens;
    sg.attn_mask = ggml_new_tensor_4d(sg.ctx, GGML_TYPE_F32, kv_len, n_tokens, 1, 1);
    ggml_set_input(sg.attn_mask);
    ggml_tensor * mask_full_f16 = ggml_cast(sg.ctx, sg.attn_mask, GGML_TYPE_F16);

    sg.attn_mask_swa = ggml_new_tensor_4d(sg.ctx, GGML_TYPE_F32, kv_len, n_tokens, 1, 1);
    ggml_set_input(sg.attn_mask_swa);
    ggml_tensor * mask_swa_f16 = ggml_cast(sg.ctx, sg.attn_mask_swa, GGML_TYPE_F16);

    ggml_tensor * layer_out = build_laguna_layer(
        sg.ctx, sg.gf, w, cache, layer_idx, inp, sg.positions,
        mask_full_f16, kv_start, n_tokens, mask_swa_f16);
    if (!layer_out) return false;

    ggml_tensor * out_view = ggml_view_2d(
        sg.ctx, act_out, w.n_embd, n_tokens,
        act_out->nb[1], (size_t)chunk_start * act_out->nb[1]);
    ggml_build_forward_expand(sg.gf, ggml_cpy(sg.ctx, layer_out, out_view));

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

bool compute_laguna_split_projection(
    ggml_backend_t backend,
    const LagunaTargetWeights & w,
    ggml_tensor * act,
    int token_offset,
    int n_tokens,
    std::vector<int32_t> * out_argmax,
    std::vector<float> * out_logits) {
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead() + 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * act_view = ggml_view_2d(
        ctx, act, w.n_embd, n_tokens, act->nb[1],
        (size_t)token_offset * act->nb[1]);
    ggml_tensor * cur = laguna_rms_norm_mul(ctx, act_view, w.out_norm);
    cur = ggml_mul_mat(ctx, w.output, cur);
    ggml_tensor * logits = cur;
    ggml_tensor * argmax = nullptr;
    if (out_logits) {
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
    }
    if (out_argmax) {
        argmax = ggml_argmax(ctx, logits);
        ggml_set_output(argmax);
        ggml_build_forward_expand(gf, argmax);
    }

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
        if (alloc) ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }
    if (out_argmax) {
        out_argmax->resize((size_t)n_tokens);
        ggml_backend_tensor_get(argmax, out_argmax->data(), 0,
                                sizeof(int32_t) * (size_t)n_tokens);
    }
    if (out_logits) {
        const int vocab = (int)w.embedder.n_vocab;
        out_logits->resize((size_t)vocab * (size_t)n_tokens);
        ggml_backend_tensor_get(logits, out_logits->data(), 0,
                                sizeof(float) * (size_t)vocab * (size_t)n_tokens);
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

bool compute_laguna_split_argmax(
    ggml_backend_t backend,
    const LagunaTargetWeights & w,
    ggml_tensor * act,
    int token_offset,
    int n_tokens,
    std::vector<int32_t> & out_argmax) {
    return compute_laguna_split_projection(
        backend, w, act, token_offset, n_tokens, &out_argmax, nullptr);
}

LagunaGraphOutputs build_laguna_graph(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const LagunaTargetWeights & w,
    LagunaTargetCache & cache,
    const LagunaGraphInputs & in)
{
    LagunaGraphOutputs out{};

    ggml_tensor * cur = in.inp_embed;  // [n_embd, n_tokens, 1] f32 (CPU-embedded)
    if (cur->ne[2] == 1) {
        cur = ggml_reshape_2d(ctx, cur, w.n_embd, in.n_tokens);
    }

    for (int il = 0; il < w.n_layer; ++il) {
        cur = build_laguna_layer(ctx, gf, w, cache, il, cur,
                                  in.positions, in.attn_mask, in.kv_start, in.n_tokens,
                                  in.attn_mask_swa, in.hybrid, in.kv_pad, in.kv_idx);
    }

    // Final norm + lm_head
    cur = laguna_rms_norm_mul(ctx, cur, w.out_norm);

    if (in.output_hidden_states) {
        out.hidden_states = cur;
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
    }

    if (in.output_logits) {
        ggml_tensor * head_in = cur;  // [n_embd, n_tokens]
        if (in.output_last_only && in.n_tokens > 1) {
            head_in = ggml_view_2d(ctx, cur, w.n_embd, 1,
                                    cur->nb[1],
                                    (size_t)(in.n_tokens - 1) * cur->nb[1]);
        }
        out.logits = ggml_mul_mat(ctx, w.output, head_in);  // [vocab, 1] or [vocab, n_tokens]
        ggml_set_output(out.logits);
        ggml_build_forward_expand(gf, out.logits);
    }

    return out;
}

// ---- Public turnkey forward step ----------------------------------------
//
// Allocates a fresh ggml_context + cgraph each call (cheap relative to the
// CUDA forward), wires the FULL + SWA causal masks, runs the backend graph,
// and returns last-token logits on the host. Updates cache.cur_pos.
//
// Reuses a single static gallocr across calls so the per-step allocation
// overhead amortises after the first warmup.
bool laguna_step(
    ggml_backend_t              backend,
    const LagunaTargetWeights & w,
    LagunaTargetCache &         cache,
    const float *               embed,
    int                         n_tok,
    int                         kv_start,
    bool                        no_mask,
    std::vector<float> &        out_logits)
{
    // Same CUDA-graph-replay treatment as laguna_step_hybrid: persistent
    // arena (stable node addresses -> stable graph key), stride-padded KV
    // span, and set_rows K/V append (index is an input, so node properties
    // are bit-identical across decode steps and the captured graph replays).
    const size_t arena_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() + 16 * 1024 * 1024;
    // thread_local: decode is single-threaded per process today (the static
    // gallocr below makes the same assumption), but a second decode thread
    // must not share the arena — each thread gets its own stable addresses.
    static thread_local std::vector<uint8_t> g_arena;
    if (g_arena.size() < arena_size) g_arena.resize(arena_size);
    ggml_init_params ip{};
    ip.mem_size = arena_size;
    ip.mem_buffer = g_arena.data();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);

    ggml_tensor * ie = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w.n_embd, n_tok, 1);
    ggml_set_input(ie);
    ggml_tensor * pp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tok);
    ggml_set_input(pp);

    const int kv_len = kv_start + n_tok;
    static const bool g_no_kvpad = (std::getenv("DFLASH_LAGUNA_NO_KVPAD") != nullptr);
    static const bool g_pad_cpy = (std::getenv("DFLASH_LAGUNA_PAD_CPY") != nullptr);
    int kv_cap = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        if (cache.attn_k[(size_t)il]) { kv_cap = (int)cache.attn_k[(size_t)il]->ne[1]; break; }
    }
    const int kv_pad = (!g_no_kvpad && kv_cap > 0)
        ? std::min((kv_len + 255) & ~255, kv_cap) : 0;
    const int mk_w = kv_pad > 0 ? kv_pad : kv_len;

    ggml_tensor * kvi = nullptr;
    if (kv_pad > 0 && !g_pad_cpy) {
        kvi = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tok);
        ggml_set_input(kvi);
    }

    ggml_tensor * mk_full = nullptr, * mk_full_cnv = nullptr;
    ggml_tensor * mk_swa  = nullptr, * mk_swa_cnv  = nullptr;
    if (!no_mask) {
        mk_full = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, mk_w, n_tok, 1, 1);
        ggml_set_input(mk_full);
        mk_full_cnv = ggml_cast(ctx, mk_full, GGML_TYPE_F16);
        mk_swa = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, mk_w, n_tok, 1, 1);
        ggml_set_input(mk_swa);
        mk_swa_cnv = ggml_cast(ctx, mk_swa, GGML_TYPE_F16);
    }

    LagunaGraphInputs gi{};
    gi.inp_embed     = ie;
    gi.positions     = pp;
    gi.attn_mask     = mk_full_cnv;
    gi.attn_mask_swa = mk_swa_cnv;
    gi.n_tokens      = n_tok;
    gi.kv_start      = kv_start;
    gi.kv_pad        = kv_pad;
    gi.kv_idx        = kvi;
    gi.output_last_only = true;

    LagunaGraphOutputs go = build_laguna_graph(ctx, gf, w, cache, gi);
    ggml_set_output(go.logits);

    static ggml_gallocr_t galloc = nullptr;
    if (!galloc) galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "laguna_step: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(ie, embed, 0, ggml_nbytes(ie));
    std::vector<int32_t> pos((size_t)n_tok);
    for (int i = 0; i < n_tok; ++i) pos[i] = kv_start + i;
    ggml_backend_tensor_set(pp, pos.data(), 0, ggml_nbytes(pp));
    if (kvi) {
        ggml_backend_tensor_set(kvi, pos.data(), 0, ggml_nbytes(kvi));
    }

    if (!no_mask) {
        // Width mk_w (= kv_pad when padding): [kv_len, mk_w) stays -inf so the
        // zero-initialised padded cache tail contributes nothing.
        std::vector<float> mfull((size_t)mk_w * n_tok, -INFINITY);
        for (int q = 0; q < n_tok; ++q) {
            const int abs_q = kv_start + q;
            for (int k = 0; k <= abs_q && k < kv_len; ++k) {
                mfull[(size_t)q * mk_w + k] = 0.0f;
            }
        }
        ggml_backend_tensor_set(mk_full, mfull.data(), 0, ggml_nbytes(mk_full));

        std::vector<float> mswa((size_t)mk_w * n_tok, -INFINITY);
        const int W = w.sliding_window;
        for (int q = 0; q < n_tok; ++q) {
            const int abs_q = kv_start + q;
            const int win_lo = std::max(0, abs_q - W + 1);
            for (int k = win_lo; k <= abs_q && k < kv_len; ++k) {
                mswa[(size_t)q * mk_w + k] = 0.0f;
            }
        }
        ggml_backend_tensor_set(mk_swa, mswa.data(), 0, ggml_nbytes(mk_swa));
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "laguna_step: graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }

    out_logits.resize((size_t)w.embedder.n_vocab);
    ggml_backend_tensor_get(go.logits, out_logits.data(), 0,
                             out_logits.size() * sizeof(float));

    cache.cur_pos = kv_len;
    ggml_free(ctx);
    return true;
}


// ---- Single-graph hybrid decode forward -----------------------------------
bool laguna_step_hybrid(
    ggml_backend_t              backend,
    const LagunaTargetWeights & w,
    LagunaTargetCache &         cache,
    const float *               embed,
    int                         n_tok,
    int                         kv_start,
    bool                        no_mask,
    const MoeHybridStorage &    hyb,
    std::vector<float> &        out_logits,
    std::vector<int32_t> *      out_selected)
{
    // Persistent arena: rebuilt graphs land at IDENTICAL addresses every step.
    // The ggml-cuda CUDA-graph cache is keyed on nodes[0] and memcmps node
    // properties (incl. src data pointers); address stability across steps is
    // what lets the captured CUDA graph replay instead of re-launching ~1.4k
    // kernels per token.
    const size_t arena_size = ggml_tensor_overhead() * 32768 + ggml_graph_overhead() + 32 * 1024 * 1024;
    // thread_local: decode is single-threaded per process today (the static
    // gallocr below makes the same assumption), but a second decode thread
    // must not share the arena — each thread gets its own stable addresses.
    static thread_local std::vector<uint8_t> g_arena;
    if (g_arena.size() < arena_size) g_arena.resize(arena_size);
    ggml_init_params ip{};
    ip.mem_size = arena_size;
    ip.mem_buffer = g_arena.data();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);

    ggml_tensor * ie = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w.n_embd, n_tok, 1);
    ggml_set_input(ie);
    ggml_tensor * pp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tok);
    ggml_set_input(pp);

    // Stride-pad the KV span so the graph topology only changes when decode
    // crosses a 256-token boundary (clamped to cache capacity). Within a
    // window the masks gate validity and the K/V append uses ggml_set_rows,
    // so every node's properties are bit-identical step to step.
    const int kv_len = kv_start + n_tok;
    static const bool g_no_kvpad = (std::getenv("DFLASH_LAGUNA_NO_KVPAD") != nullptr);
    int kv_cap = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        if (cache.attn_k[(size_t)il]) { kv_cap = (int)cache.attn_k[(size_t)il]->ne[1]; break; }
    }
    const int kv_pad = (!g_no_kvpad && kv_cap > 0)
        ? std::min((kv_len + 255) & ~255, kv_cap) : 0;
    const int mk_w = kv_pad > 0 ? kv_pad : kv_len;

    // Decomposition knob: pad the FA span but keep the legacy cpy append
    // (kv_idx=null). No CUDA-graph replay; isolates pad-rounding vs set_rows.
    static const bool g_pad_cpy = (std::getenv("DFLASH_LAGUNA_PAD_CPY") != nullptr);
    ggml_tensor * kvi = nullptr;
    if (kv_pad > 0 && !g_pad_cpy) {
        kvi = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tok);
        ggml_set_input(kvi);
    }

    ggml_tensor * mk_full = nullptr, * mk_full_cnv = nullptr;
    ggml_tensor * mk_swa  = nullptr, * mk_swa_cnv  = nullptr;
    if (!no_mask) {
        mk_full = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, mk_w, n_tok, 1, 1);
        ggml_set_input(mk_full);
        mk_full_cnv = ggml_cast(ctx, mk_full, GGML_TYPE_F16);
        mk_swa = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, mk_w, n_tok, 1, 1);
        ggml_set_input(mk_swa);
        mk_swa_cnv = ggml_cast(ctx, mk_swa, GGML_TYPE_F16);
    }

    const int n_expert = w.n_expert;
    const int n_used   = w.n_expert_used;
    const int n_moe    = w.n_layer - w.n_layer_dense_lead;
    LagunaHybridMoe hybm{};
    hybm.storage    = &hyb;
    hybm.dense_lead = w.n_layer_dense_lead;
    hybm.lut_all = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert, n_moe); ggml_set_input(hybm.lut_all);
    hybm.vld_all = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_expert, n_moe); ggml_set_input(hybm.vld_all);
    hybm.sel_all = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used,  n_moe);  ggml_set_output(hybm.sel_all);

    // Reuse the shared graph builder (same attention / norm / precision path as
    // the all-GPU decode); the hybrid descriptor swaps MoE layers to the hot
    // stack via the per-layer LUTs.
    LagunaGraphInputs gi{};
    gi.inp_embed        = ie;
    gi.positions        = pp;
    gi.attn_mask        = mk_full_cnv;
    gi.attn_mask_swa    = mk_swa_cnv;
    gi.n_tokens         = n_tok;
    gi.kv_start         = kv_start;
    gi.kv_pad           = kv_pad;
    gi.kv_idx           = kvi;
    gi.output_last_only = true;
    gi.hybrid           = &hybm;
    LagunaGraphOutputs go = build_laguna_graph(ctx, gf, w, cache, gi);
    ggml_tensor * logits = go.logits;

    static ggml_gallocr_t galloc_h = nullptr;
    if (!galloc_h) galloc_h = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc_h, gf)) {
        std::fprintf(stderr, "laguna_step_hybrid: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(ie, embed, 0, ggml_nbytes(ie));
    std::vector<int32_t> pos((size_t)n_tok);
    for (int i = 0; i < n_tok; ++i) pos[i] = kv_start + i;
    ggml_backend_tensor_set(pp, pos.data(), 0, ggml_nbytes(pp));
    if (kvi) {
        // set_rows row indices = absolute cache positions of this step's tokens
        ggml_backend_tensor_set(kvi, pos.data(), 0, ggml_nbytes(kvi));
    }

    if (!no_mask) {
        // Width mk_w (= kv_pad when padding): columns in [kv_len, mk_w) stay
        // -inf so the zero-initialised padded cache tail contributes nothing.
        std::vector<float> mfull((size_t)mk_w * n_tok, -INFINITY);
        for (int q = 0; q < n_tok; ++q) {
            const int abs_q = kv_start + q;
            for (int k = 0; k <= abs_q && k < kv_len; ++k) mfull[(size_t)q * mk_w + k] = 0.0f;
        }
        ggml_backend_tensor_set(mk_full, mfull.data(), 0, ggml_nbytes(mk_full));
        std::vector<float> mswa((size_t)mk_w * n_tok, -INFINITY);
        const int Wsw = w.sliding_window;
        for (int q = 0; q < n_tok; ++q) {
            const int abs_q = kv_start + q;
            const int lo = std::max(0, abs_q - Wsw + 1);
            for (int k = lo; k <= abs_q && k < kv_len; ++k) mswa[(size_t)q * mk_w + k] = 0.0f;
        }
        ggml_backend_tensor_set(mk_swa, mswa.data(), 0, ggml_nbytes(mk_swa));
    }

    // Set ALL residency LUTs in two batched H2D copies from the hot stack mapping.
    std::vector<int32_t> lutbuf((size_t)n_expert * (size_t)n_moe);
    std::vector<float>   vldbuf((size_t)n_expert * (size_t)n_moe);
    for (int il = w.n_layer_dense_lead; il < w.n_layer; ++il) {
        const int mi = il - w.n_layer_dense_lead;
        const MoeHybridLayerStorage & st = hyb.layers[(size_t)il];
        for (int g = 0; g < n_expert; ++g) {
            int loc = (g < (int)st.hot_local_by_global.size()) ? st.hot_local_by_global[(size_t)g] : -1;
            lutbuf[(size_t)mi * n_expert + g] = loc >= 0 ? loc : 0;
            vldbuf[(size_t)mi * n_expert + g] = loc >= 0 ? 1.0f : 0.0f;
        }
    }
    ggml_backend_tensor_set(hybm.lut_all, lutbuf.data(), 0, sizeof(int32_t) * lutbuf.size());
    ggml_backend_tensor_set(hybm.vld_all, vldbuf.data(), 0, sizeof(float) * vldbuf.size());

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "laguna_step_hybrid: graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }

    out_logits.resize((size_t)w.embedder.n_vocab);
    ggml_backend_tensor_get(logits, out_logits.data(), 0, out_logits.size() * sizeof(float));

    if (out_selected) {
        std::vector<int32_t> selbuf((size_t)n_used * (size_t)n_moe);
        ggml_backend_tensor_get(hybm.sel_all, selbuf.data(), 0, sizeof(int32_t) * selbuf.size());
        out_selected->assign((size_t)w.n_layer * (size_t)n_used, -1);
        for (int il = w.n_layer_dense_lead; il < w.n_layer; ++il) {
            const int mi = il - w.n_layer_dense_lead;
            for (int k = 0; k < n_used; ++k)
                (*out_selected)[(size_t)il * n_used + k] = selbuf[(size_t)mi * n_used + k];
        }
    }
    ggml_free(ctx);
    return true;
}

} // namespace dflash::common
