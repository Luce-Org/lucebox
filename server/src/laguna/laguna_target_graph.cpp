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
#include "../common/moe_router_graph.h"
#include "../common/kvflash_pager.h"
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

static bool laguna_tensor_set_checked(const char * label,
                                      ggml_tensor * t,
                                      const void * data,
                                      size_t offset,
                                      size_t size,
                                      bool required = true) {
    if (!t || !t->buffer) {
        if (required) {
            std::fprintf(stderr, "[laguna-graph] input tensor not allocated: %s\n", label);
            set_last_error("laguna graph input tensor not allocated");
            return false;
        }
        return true;
    }
    ggml_backend_tensor_set(t, data, offset, size);
    return true;
}

// ---- Cache lifecycle ----------------------------------------------------

bool create_laguna_target_cache(const LagunaTargetWeights & w,
                                 int max_ctx,
                                 ggml_backend_t backend,
                                 LagunaTargetCache & out,
                                 int ctx_alloc) {
    return create_laguna_target_cache_partial(
        w, max_ctx, backend, /*layer_begin=*/0, /*layer_end=*/w.n_layer, out,
        ctx_alloc);
}

bool create_laguna_target_cache_partial(const LagunaTargetWeights & w,
                                         int max_ctx,
                                         ggml_backend_t backend,
                                         int layer_begin,
                                         int layer_end,
                                         LagunaTargetCache & out,
                                         int ctx_alloc) {
    if (layer_begin < 0) layer_begin = 0;
    if (layer_end < 0) layer_end = w.n_layer;
    if (layer_begin > layer_end || layer_end > w.n_layer) {
        set_last_error("laguna cache: invalid layer range");
        return false;
    }

    // Keep the physical KV span 256-aligned for ggml-cuda FA GQA kernels.
    // The logical bound remains max_ctx; extra rows are masked and zero-filled.
    constexpr int kKvFaPad = 256;
    const int ctx_phys_raw = (ctx_alloc > 0 && ctx_alloc < max_ctx) ? ctx_alloc : max_ctx;
    const int ctx_phys = std::max(ctx_phys_raw, ((ctx_phys_raw + kKvFaPad - 1) / kKvFaPad) * kKvFaPad);

    out.backend  = backend;
    out.max_ctx  = max_ctx;
    out.cur_pos  = 0;
    out.last_tok = -1;
    out.kv_head_major = std::getenv("DFLASH_LAGUNA_KV_HEAD_MAJOR") != nullptr;
    // KV cache: per-layer, ALL 40 layers (full + SWA). Layout matches qwen35:
    //   legacy:     [head_dim, max_ctx, n_head_kv]
    //   head-major: [head_dim*n_head_kv, max_ctx]
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
        ggml_tensor * k = out.kv_head_major
            ? ggml_new_tensor_2d(out.base_ctx, k_type, w.head_dim * w.n_head_kv, ctx_phys)
            : ggml_new_tensor_3d(out.base_ctx, k_type, w.head_dim, ctx_phys, w.n_head_kv);
        ggml_set_name(k, nm);
        std::snprintf(nm, sizeof(nm), "v_l%d", il);
        ggml_tensor * v = out.kv_head_major
            ? ggml_new_tensor_2d(out.base_ctx, v_type, w.head_dim * w.n_head_kv, ctx_phys)
            : ggml_new_tensor_3d(out.base_ctx, v_type, w.head_dim, ctx_phys, w.n_head_kv);
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
    const int n_feat_tensors = (cache.target_feat && cache.target_feat_cap > 0) ? 1 : 0;
    ip.mem_size = ggml_tensor_overhead() * (size_t)(n_layer * 2 + n_feat_tensors + 16) + 4096;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) { set_last_error("snapshot: ggml_init failed"); return false; }
    out.attn_k.assign((size_t)n_layer, nullptr);
    out.attn_v.assign((size_t)n_layer, nullptr);
    for (int il = 0; il < n_layer; ++il) {
        if (!cache.attn_k[il] || !cache.attn_v[il]) continue;
        char nm[32];
        std::snprintf(nm, sizeof(nm), "snap_k_l%d", il);
        ggml_tensor * k = cache.kv_head_major
            ? ggml_new_tensor_2d(out.ctx, cache.kv_k_type, head_dim * n_head_kv, snap_pos)
            : ggml_new_tensor_3d(out.ctx, cache.kv_k_type, head_dim, snap_pos, n_head_kv);
        ggml_set_name(k, nm);
        std::snprintf(nm, sizeof(nm), "snap_v_l%d", il);
        ggml_tensor * v = cache.kv_head_major
            ? ggml_new_tensor_2d(out.ctx, cache.kv_v_type, head_dim * n_head_kv, snap_pos)
            : ggml_new_tensor_3d(out.ctx, cache.kv_v_type, head_dim, snap_pos, n_head_kv);
        ggml_set_name(v, nm);
        out.attn_k[il] = k;
        out.attn_v[il] = v;
    }
    out.feat_snap = nullptr;
    out.feat_cap = 0;
    if (cache.target_feat && cache.target_feat_cap > 0) {
        const int feat_len = std::min(snap_pos, cache.target_feat_cap);
        out.feat_snap = ggml_new_tensor_2d(out.ctx, cache.target_feat->type,
                                           cache.target_feat->ne[0], feat_len);
        out.feat_cap = cache.target_feat_cap;
    }
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        set_last_error("snapshot: ggml_backend_alloc_ctx_tensors failed");
        ggml_free(out.ctx); out.ctx = nullptr;
        out.feat_snap = nullptr;
        out.feat_cap = 0;
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
    snap.feat_snap = nullptr;
    snap.feat_cap  = 0;
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
    const bool needs_feat = cache.target_feat && cache.target_feat_cap > 0;
    if (snap.ctx && (snap.cur_pos != snap_pos ||
                     (needs_feat && !snap.feat_snap) ||
                     (!needs_feat && snap.feat_snap))) {
        laguna_snapshot_free(snap);
    }
    if (!snap.ctx) {
        if (!laguna_snapshot_alloc(cache, backend, n_layer, snap_pos, n_head_kv, head_dim, snap)) {
            return false;
        }
    }

    // Copy KV strip-by-strip for legacy layout. Head-major layout stores each
    // position as one contiguous row [head_dim*n_head_kv], so one contiguous
    // prefix copy is enough.
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * sk = cache.attn_k[il];
        ggml_tensor * dk = snap.attn_k[il];
        ggml_tensor * sv = cache.attn_v[il];
        ggml_tensor * dv = snap.attn_v[il];
        if (!sk || !dk || !sv || !dv) continue;
        if (cache.kv_head_major) {
            const size_t k_bytes = (size_t)snap_pos * sk->nb[1];
            const size_t v_bytes = (size_t)snap_pos * sv->nb[1];
            ggml_backend_tensor_get(sk, dk->data, 0, k_bytes);
            ggml_backend_tensor_get(sv, dv->data, 0, v_bytes);
            continue;
        }
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

    if (snap.feat_snap && cache.target_feat) {
        const size_t feat_nbytes = ggml_nbytes(snap.feat_snap);
        ggml_backend_tensor_get(cache.target_feat, snap.feat_snap->data, 0, feat_nbytes);
    }
    return true;
}

bool laguna_snapshot_restore(const LagunaCacheSnapshot & snap,
                              LagunaTargetCache &         cache) {
    if (!snap.used || snap.attn_k.size() != cache.attn_k.size()) {
        set_last_error("snapshot_restore: snapshot unused or layer count mismatch");
        return false;
    }
    const int snap_pos = snap.cur_pos;
    // Copy right-sized snapshot back into full-size cache.
    for (size_t il = 0; il < cache.attn_k.size(); ++il) {
        ggml_tensor * sk = snap.attn_k[il];
        ggml_tensor * dk = cache.attn_k[il];
        ggml_tensor * sv = snap.attn_v[il];
        ggml_tensor * dv = cache.attn_v[il];
        if (!sk || !dk || !sv || !dv) continue;
        if (cache.kv_head_major) {
            const size_t k_bytes = (size_t)snap_pos * sk->nb[1];
            const size_t v_bytes = (size_t)snap_pos * sv->nb[1];
            ggml_backend_tensor_set(dk, sk->data, 0, k_bytes);
            ggml_backend_tensor_set(dv, sv->data, 0, v_bytes);
            continue;
        }
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
    if (snap.feat_snap && cache.target_feat) {
        const size_t feat_nbytes = ggml_nbytes(snap.feat_snap);
        ggml_backend_tensor_set(cache.target_feat, snap.feat_snap->data, 0, feat_nbytes);
    }
    cache.cur_pos = snap_pos;
    return true;
}

void free_laguna_target_cache(LagunaTargetCache & c) {
    free_laguna_target_feat(c);
    if (c.base_buf) { ggml_backend_buffer_free(c.base_buf); c.base_buf = nullptr; }
    if (c.base_ctx) { ggml_free(c.base_ctx);                c.base_ctx = nullptr; }
    c.attn_k.clear();
    c.attn_v.clear();
    c.max_ctx = 0;
    c.cur_pos = 0;
    c.last_tok = -1;
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
static ggml_tensor * build_laguna_moe_block_full(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * cur,
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
    return build_laguna_moe_block_full(ctx, gf, cur, w, L);
}

// Phase 2.1: full MoE dispatch (sigmoid + score-correction bias + sum-norm +
// scale 2.5 + always-on shared expert). Mirrors llama.cpp's build_moe_ffn for
// the SIGMOID + WEIGHTS_NORM + EXP_PROBS_B configuration that Laguna uses.
static ggml_tensor * build_laguna_moe_block_full(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * cur,
                                                  const LagunaTargetWeights & w,
                                                  const LagunaTargetLayer & L) {
    const int n_tokens = (int)cur->ne[1];
    const int n_expert = w.n_expert;
    const int n_used   = w.n_expert_used;
    const int n_embd   = w.n_embd;
    static const bool fused_combine = []() {
        const char * e = std::getenv("DFLASH_LAGUNA_MOE_FUSED_COMBINE");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();

    ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, cur);  // [n_expert, n_tokens]
    TopKMoeRouterResult router = build_sigmoid_topk_moe_router(
        ctx, gf, logits, L.ffn_exp_probs_b, n_expert, n_used, n_tokens,
        /*normalize_weights=*/true, w.expert_weights_scale,
        /*expand_weights=*/true);
    ggml_tensor * selected   = router.selected;
    ggml_tensor * weights_2d = router.weights_2d;
    ggml_tensor * weights_3d = router.weights_3d;

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

    ggml_tensor * routed = nullptr;
    if (fused_combine) {
        routed = ggml_laguna_moe_combine(ctx, experts, weights_2d);
    } else {
        experts = ggml_mul(ctx, experts, weights_3d);

        // Sum across the n_used axis: explicit slice + add loop (matches llama.cpp
        // pattern; ggml_sum_rows would sum over dim 0 which is n_embd, wrong).
        for (int i = 0; i < n_used; ++i) {
            ggml_tensor * slice = ggml_view_2d(ctx, experts,
                n_embd, n_tokens,
                experts->nb[2],
                (size_t)i * experts->nb[1]);
            routed = (i == 0) ? slice : ggml_add(ctx, routed, slice);
        }
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
    TopKMoeRouterResult router = build_sigmoid_topk_moe_router(
        ctx, gf, logits, L.ffn_exp_probs_b, n_expert, n_used, n_tokens,
        /*normalize_weights=*/true, w.expert_weights_scale,
        /*expand_weights=*/false);
    ggml_tensor * selected = router.selected;  // [n_used, n_tokens] global ids
    {   // batched readback: write this layer's selection into column moe_idx of sel_all
        ggml_tensor * sel_col = ggml_view_2d(ctx, sel_all, n_used, n_tokens,
                                             sel_all->nb[1], (size_t)moe_idx * sel_all->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, selected, sel_col));
    }

    ggml_tensor * weights = router.weights_2d;

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
    ggml_tensor * kv_idx = nullptr,
    ggml_tensor * tree_parent_ids = nullptr)
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
    const bool cache_head_major = cache_k && cache_k->ne[0] == head_dim * n_head_kv;
    ggml_tensor * Kcur_rows = nullptr;
    ggml_tensor * Vcur_rows = nullptr;
    if (cache_head_major) {
        const int64_t n_embd_kv = (int64_t)head_dim * n_head_kv;
        const bool k_merge_view =
            (size_t)Kcur->nb[1] == ggml_row_size(Kcur->type, head_dim) &&
            (size_t)Kcur->nb[2] == ggml_row_size(Kcur->type, n_embd_kv);
        const bool v_merge_view =
            (size_t)Vcur->nb[1] == ggml_row_size(Vcur->type, head_dim) &&
            (size_t)Vcur->nb[2] == ggml_row_size(Vcur->type, n_embd_kv);
        Kcur_rows = k_merge_view
            ? ggml_view_2d(ctx, Kcur, n_embd_kv, n_tokens, Kcur->nb[2], 0)
            : ggml_cont_2d(ctx, Kcur, n_embd_kv, n_tokens);
        Vcur_rows = v_merge_view
            ? ggml_view_2d(ctx, Vcur, n_embd_kv, n_tokens, Vcur->nb[2], 0)
            : ggml_cont_2d(ctx, Vcur, n_embd_kv, n_tokens);
    } else {
        Kcur_rows = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
        Vcur_rows = ggml_permute(ctx, Vcur, 0, 2, 1, 3);
    }

    if (kv_idx) {
        // CUDA-graph-stable append: the destination is the WHOLE cache tensor
        // (stable data pointer) and the row index is a graph input whose DATA
        // changes per step but whose pointer doesn't. A kv_start-offset view
        // (below) changes node properties every step, which resets the
        // ggml-cuda CUDA-graph warmup and forfeits replay.
        // Legacy layout broadcasts kv_idx over n_head_kv. Head-major layout
        // stores all KV heads for a token in one row [head_dim*n_head_kv].
        ggml_tensor * Krows = cache_head_major ? Kcur_rows : ggml_cont(ctx, Kcur_rows);
        ggml_tensor * Vrows = cache_head_major ? Vcur_rows : ggml_cont(ctx, Vcur_rows);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_k, Krows, kv_idx));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_v, Vrows, kv_idx));
    } else {
        if (cache_head_major) {
            ggml_tensor * k_slot = ggml_view_2d(ctx, cache_k,
                head_dim * n_head_kv, n_tokens,
                cache_k->nb[1], cache_k->nb[1] * (size_t)kv_start);
            ggml_tensor * v_slot = ggml_view_2d(ctx, cache_v,
                head_dim * n_head_kv, n_tokens,
                cache_v->nb[1], cache_v->nb[1] * (size_t)kv_start);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur_rows, k_slot));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur_rows, v_slot));
        } else {
            ggml_tensor * k_slot = ggml_view_3d(ctx, cache_k,
                head_dim, n_tokens, n_head_kv,
                cache_k->nb[1], cache_k->nb[2],
                cache_k->nb[1] * (size_t)kv_start);
            ggml_tensor * v_slot = ggml_view_3d(ctx, cache_v,
                head_dim, n_tokens, n_head_kv,
                cache_v->nb[1], cache_v->nb[2],
                cache_v->nb[1] * (size_t)kv_start);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur_rows, k_slot));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur_rows, v_slot));
        }
    }

    // ---- Flash attention ---
    // BOTH full and SWA layers read the FULL kv_len of K/V from the cache.
    // For SWA layers, the per-row sliding-window causal constraint is enforced
    // by attn_mask_swa (built by the caller). This is correct for the early-
    // token rows (which need to see KV positions [0..p+1)) and the late-token
    // rows (which need [p-sw+1..p+1)).
    const int kv_len = kv_start + n_tokens;
    // kv_pad > 0: read a stride-rounded fixed span so the view shape (and thus
    // every downstream FA node's properties) stays constant across decode
    // steps; the mask carries -inf for [kv_len, kv_pad) and the cache buffer
    // is zero-initialised, so the padded tail contributes exactly nothing.
    const int win_start = 0;
    const int win_len = kv_pad > 0 ? kv_pad : kv_len;

    ggml_tensor * Qfa = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
    Qfa = ggml_cont(ctx, Qfa);

    ggml_tensor * Kfa = nullptr;
    ggml_tensor * Vfa = nullptr;
    if (cache_head_major) {
        ggml_tensor * Kview = ggml_view_3d(ctx, cache_k,
            head_dim, n_head_kv, win_len,
            ggml_row_size(cache_k->type, head_dim), cache_k->nb[1],
            cache_k->nb[1] * (size_t)win_start);
        ggml_tensor * Vview = ggml_view_3d(ctx, cache_v,
            head_dim, n_head_kv, win_len,
            ggml_row_size(cache_v->type, head_dim), cache_v->nb[1],
            cache_v->nb[1] * (size_t)win_start);
        Kfa = ggml_permute(ctx, Kview, 0, 2, 1, 3);
        Vfa = ggml_permute(ctx, Vview, 0, 2, 1, 3);
    } else {
        Kfa = ggml_view_3d(ctx, cache_k,
            head_dim, win_len, n_head_kv,
            cache_k->nb[1], cache_k->nb[2], cache_k->nb[1] * (size_t)win_start);
        Vfa = ggml_view_3d(ctx, cache_v,
            head_dim, win_len, n_head_kv,
            cache_v->nb[1], cache_v->nb[2], cache_v->nb[1] * (size_t)win_start);
    }

    const float kq_scale = 1.0f / std::sqrt((float)head_dim);
    // FULL -> attn_mask (causal). SWA -> attn_mask_swa (causal + sliding-window).
    ggml_tensor * use_mask = is_full ? attn_mask : attn_mask_swa;
    static const bool g_tree_attn_op =
        (std::getenv("DFLASH_LAGUNA_TREE_ATTN_OP") != nullptr);
    ggml_tensor * attn = (g_tree_attn_op && tree_parent_ids && !is_full)
        ? ggml_flash_attn_tree(ctx, Qfa, Kfa, Vfa, use_mask,
                               tree_parent_ids, positions, kq_scale)
        : ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa, use_mask,
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
    ggml_tensor * kv_idx = nullptr,
    ggml_tensor * tree_parent_ids = nullptr)
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
                                    kv_pad, kv_idx, tree_parent_ids);

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
    sg.kv_idx = nullptr;
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
    int kv_start,
    const KvFlashPager * kvflash) {
    laguna_layer_step_graph_free(sg);
    if (layer_idx < 0 || layer_idx >= w.n_layer) return false;
    if (!cache.attn_k[layer_idx] || !cache.attn_v[layer_idx]) return false;
    if (kvflash && std::getenv("DFLASH_LAGUNA_NO_KVPAD")) return false;

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
    const int kv_cap = (int)cache.attn_k[(size_t)layer_idx]->ne[1];
    const int kv_pad = std::min((kv_len + 255) & ~255, kv_cap);
    const int mk_w = kvflash ? kv_pad : kv_len;
    sg.attn_mask = ggml_new_tensor_4d(sg.ctx, GGML_TYPE_F32, mk_w, n_tokens, 1, 1);
    ggml_set_input(sg.attn_mask);
    ggml_tensor * mask_full_f16 = ggml_cast(sg.ctx, sg.attn_mask, GGML_TYPE_F16);

    sg.attn_mask_swa = ggml_new_tensor_4d(sg.ctx, GGML_TYPE_F32, mk_w, n_tokens, 1, 1);
    ggml_set_input(sg.attn_mask_swa);
    ggml_tensor * mask_swa_f16 = ggml_cast(sg.ctx, sg.attn_mask_swa, GGML_TYPE_F16);

    sg.kv_idx = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(sg.kv_idx);

    ggml_tensor * layer_out = build_laguna_layer(
        sg.ctx, sg.gf, w, cache, layer_idx, inp, sg.positions,
        mask_full_f16, kv_start, n_tokens, mask_swa_f16,
        /*hyb=*/nullptr, kvflash ? kv_pad : 0, sg.kv_idx);
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

    const bool capture_with_rows =
        in.capture_features && cache.target_feat && in.target_feat_rows;
    std::vector<ggml_tensor *> capture_slices;
    if (capture_with_rows) {
        capture_slices.assign((size_t)cache.n_capture_layers, nullptr);
    }

    for (int il = 0; il < w.n_layer; ++il) {
        cur = build_laguna_layer(ctx, gf, w, cache, il, cur,
                                  in.positions, in.attn_mask, in.kv_start, in.n_tokens,
                                  in.attn_mask_swa, in.hybrid, in.kv_pad, in.kv_idx,
                                  in.tree_parent_ids);

        // Feature capture for DFlash spec-decode: write residual-stream layer
        // outputs into the BF16 target feature ring.
        if (in.capture_features && cache.target_feat) {
            int cap_idx = -1;
            for (int k = 0; k < cache.n_capture_layers; k++) {
                if (cache.capture_layer_ids[(size_t)k] == il) { cap_idx = k; break; }
            }
            if (cap_idx >= 0) {
                const int hidden = w.n_embd;
                ggml_tensor * cur_2d = ggml_reshape_2d(ctx, cur, hidden, in.n_tokens);
                if (capture_with_rows) {
                    capture_slices[(size_t)cap_idx] = cur_2d;
                    continue;
                }

                const int cap = cache.target_feat_cap;
                const size_t elt = ggml_element_size(cache.target_feat);
                const size_t col_stride = cache.target_feat->nb[1];
                const int slot_start = in.kv_start % cap;
                const int pre_n = std::min(in.n_tokens, cap - slot_start);
                const int post_n = in.n_tokens - pre_n;

                {
                    const size_t offset =
                        (size_t)slot_start * col_stride +
                        (size_t)cap_idx * hidden * elt;
                    ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                        hidden, pre_n, col_stride, offset);
                    ggml_tensor * src = ggml_view_2d(ctx, cur_2d,
                        hidden, pre_n, cur_2d->nb[1], 0);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
                }

                if (post_n > 0) {
                    const size_t offset =
                        (size_t)cap_idx * hidden * elt;
                    ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                        hidden, post_n, col_stride, offset);
                    ggml_tensor * src = ggml_view_2d(ctx, cur_2d,
                        hidden, post_n, cur_2d->nb[1],
                        (size_t)pre_n * cur_2d->nb[1]);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
                }
            }
        }
    }

    if (capture_with_rows && !capture_slices.empty()) {
        bool have_all = true;
        for (ggml_tensor * t : capture_slices) {
            if (!t) { have_all = false; break; }
        }
        if (have_all) {
            ggml_tensor * feat_cat = capture_slices[0];
            for (int k = 1; k < (int)capture_slices.size(); ++k) {
                feat_cat = ggml_concat(ctx, feat_cat, capture_slices[(size_t)k], 0);
            }
            feat_cat = ggml_cont(ctx, feat_cat);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache.target_feat,
                                                        feat_cat, in.target_feat_rows));
        }
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
        if (in.logits_are_output) {
            ggml_set_output(out.logits);
            ggml_build_forward_expand(gf, out.logits);
        }
    }

    return out;
}

// ---- Public turnkey forward step ----------------------------------------
//
// Wires the FULL + SWA causal masks, runs the backend graph, and returns
// last-token logits and/or a GPU-computed argmax on the host. Updates
// cache.cur_pos. The common greedy decode path reuses a per-thread graph while
// the generic path rebuilds for prefill, capture, kvflash, or logits readback.
bool laguna_step(
    ggml_backend_t              backend,
    const LagunaTargetWeights & w,
    LagunaTargetCache &         cache,
    const float *               embed,
    int                         n_tok,
    int                         kv_start,
    bool                        no_mask,
    std::vector<float> &        out_logits,
    const KvFlashPager *        kvflash,
    bool                        capture,
    int32_t *                   out_argmax,
    bool                        read_logits)
{
    if (kvflash && no_mask) {
        std::fprintf(stderr, "laguna_step: kvflash requires masks (slots are "
                             "relocated; position-implicit masking is invalid)\n");
        return false;
    }

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

    const bool can_reuse_step_graph =
        n_tok == 1 && !kvflash && !no_mask && !capture && kv_pad > 0 &&
        !g_pad_cpy && out_argmax && !read_logits;
    if (can_reuse_step_graph) {
        struct CachedStepGraph {
            const LagunaTargetWeights * w_ptr = nullptr;
            LagunaTargetCache * cache_ptr = nullptr;
            ggml_backend_t backend = nullptr;
            int mk_w = 0;
            int kv_pad = 0;
            std::vector<uint8_t> arena;
            ggml_context * ctx = nullptr;
            ggml_cgraph * gf = nullptr;
            ggml_gallocr_t alloc = nullptr;
            ggml_tensor * inp_embed = nullptr;
            ggml_tensor * positions = nullptr;
            ggml_tensor * kv_idx = nullptr;
            ggml_tensor * mask_full = nullptr;
            ggml_tensor * mask_swa = nullptr;
            ggml_tensor * argmax = nullptr;
            std::vector<float> full_mask;
            std::vector<float> swa_mask;

            void clear() {
                if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
                if (ctx) { ggml_free(ctx); ctx = nullptr; }
                gf = nullptr;
                inp_embed = positions = kv_idx = mask_full = mask_swa = argmax = nullptr;
                w_ptr = nullptr;
                cache_ptr = nullptr;
                backend = nullptr;
                mk_w = 0;
                kv_pad = 0;
            }
        };
        static thread_local CachedStepGraph cached;

        const bool rebuild =
            cached.ctx == nullptr || cached.w_ptr != &w || cached.cache_ptr != &cache ||
            cached.backend != backend || cached.mk_w != mk_w || cached.kv_pad != kv_pad;
        if (rebuild) {
            cached.clear();
            const size_t arena_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() + 16 * 1024 * 1024;
            cached.arena.resize(arena_size);

            ggml_init_params ip{};
            ip.mem_size = arena_size;
            ip.mem_buffer = cached.arena.data();
            ip.no_alloc = true;
            cached.ctx = ggml_init(ip);
            if (!cached.ctx) return false;
            cached.gf = ggml_new_graph_custom(cached.ctx, 16384, false);

            cached.inp_embed = ggml_new_tensor_3d(cached.ctx, GGML_TYPE_F32, w.n_embd, 1, 1);
            ggml_set_input(cached.inp_embed);
            cached.positions = ggml_new_tensor_1d(cached.ctx, GGML_TYPE_I32, 1);
            ggml_set_input(cached.positions);
            cached.kv_idx = ggml_new_tensor_1d(cached.ctx, GGML_TYPE_I32, 1);
            ggml_set_input(cached.kv_idx);
            cached.mask_full = ggml_new_tensor_4d(cached.ctx, GGML_TYPE_F32, mk_w, 1, 1, 1);
            ggml_set_input(cached.mask_full);
            ggml_tensor * mask_full_cnv = ggml_cast(cached.ctx, cached.mask_full, GGML_TYPE_F16);
            cached.mask_swa = ggml_new_tensor_4d(cached.ctx, GGML_TYPE_F32, mk_w, 1, 1, 1);
            ggml_set_input(cached.mask_swa);
            ggml_tensor * mask_swa_cnv = ggml_cast(cached.ctx, cached.mask_swa, GGML_TYPE_F16);

            LagunaGraphInputs gi{};
            gi.inp_embed     = cached.inp_embed;
            gi.positions     = cached.positions;
            gi.attn_mask     = mask_full_cnv;
            gi.attn_mask_swa = mask_swa_cnv;
            gi.n_tokens      = 1;
            gi.kv_start      = 0;
            gi.kv_pad        = kv_pad;
            gi.kv_idx        = cached.kv_idx;
            gi.capture_features = false;
            gi.output_last_only = true;
            gi.output_logits = true;
            gi.logits_are_output = false;
            gi.output_hidden_states = false;

            LagunaGraphOutputs go = build_laguna_graph(cached.ctx, cached.gf, w, cache, gi);
            cached.argmax = ggml_argmax(cached.ctx, go.logits);
            ggml_set_output(cached.argmax);
            ggml_build_forward_expand(cached.gf, cached.argmax);

            cached.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!cached.alloc || !ggml_gallocr_alloc_graph(cached.alloc, cached.gf)) {
                std::fprintf(stderr, "laguna_step: cached gallocr_alloc_graph failed\n");
                cached.clear();
                return false;
            }

            cached.w_ptr = &w;
            cached.cache_ptr = &cache;
            cached.backend = backend;
            cached.mk_w = mk_w;
            cached.kv_pad = kv_pad;
            cached.full_mask.resize((size_t)mk_w);
            cached.swa_mask.resize((size_t)mk_w);
        }

        if (!laguna_tensor_set_checked("laguna_step_cached.ie", cached.inp_embed,
                                       embed, 0, ggml_nbytes(cached.inp_embed))) {
            return false;
        }
        int32_t pos_val = kv_start;
        if (!laguna_tensor_set_checked("laguna_step_cached.positions", cached.positions,
                                       &pos_val, 0, sizeof(pos_val))) {
            return false;
        }
        if (!laguna_tensor_set_checked("laguna_step_cached.kv_idx", cached.kv_idx,
                                       &pos_val, 0, sizeof(pos_val))) {
            return false;
        }

        std::fill(cached.full_mask.begin(), cached.full_mask.end(), -INFINITY);
        for (int k = 0; k <= kv_start && k < kv_len && k < mk_w; ++k) {
            cached.full_mask[(size_t)k] = 0.0f;
        }
        if (!laguna_tensor_set_checked("laguna_step_cached.mask_full", cached.mask_full,
                                       cached.full_mask.data(), 0,
                                       ggml_nbytes(cached.mask_full))) {
            return false;
        }

        std::fill(cached.swa_mask.begin(), cached.swa_mask.end(), -INFINITY);
        const int W = w.sliding_window;
        const int win_lo = std::max(0, kv_start - W + 1);
        for (int k = win_lo; k <= kv_start && k < kv_len && k < mk_w; ++k) {
            cached.swa_mask[(size_t)k] = 0.0f;
        }
        if (!laguna_tensor_set_checked("laguna_step_cached.mask_swa", cached.mask_swa,
                                       cached.swa_mask.data(), 0,
                                       ggml_nbytes(cached.mask_swa))) {
            return false;
        }

        if (ggml_backend_graph_compute(backend, cached.gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "laguna_step: cached graph_compute failed\n");
            return false;
        }

        ggml_backend_tensor_get(cached.argmax, out_argmax, 0, sizeof(int32_t));
        out_logits.clear();

        cache.cur_pos = kv_len;
        cache.last_tok = *out_argmax;
        return true;
    }

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
    gi.capture_features = capture;
    gi.output_last_only = true;
    gi.output_logits = read_logits || out_argmax;
    gi.logits_are_output = read_logits;
    gi.output_hidden_states = !gi.output_logits;

    LagunaGraphOutputs go = build_laguna_graph(ctx, gf, w, cache, gi);
    ggml_tensor * argmax = nullptr;
    if (out_argmax) {
        argmax = ggml_argmax(ctx, go.logits);
        ggml_set_output(argmax);
        ggml_build_forward_expand(gf, argmax);
    }

    static ggml_gallocr_t galloc = nullptr;
    if (!galloc) galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "laguna_step: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    if (!laguna_tensor_set_checked("laguna_step.ie", ie, embed, 0, ggml_nbytes(ie))) {
        ggml_free(ctx);
        return false;
    }
    std::vector<int32_t> pos((size_t)n_tok);
    for (int i = 0; i < n_tok; ++i) pos[i] = kv_start + i;
    if (!laguna_tensor_set_checked("laguna_step.positions", pp, pos.data(), 0, ggml_nbytes(pp))) {
        ggml_free(ctx);
        return false;
    }

    if (kvflash) {
        if (!kvi) {
            std::fprintf(stderr, "laguna_step: kvflash requires the kv_pad "
                                 "set_rows path (NO_KVPAD / PAD_CPY are incompatible)\n");
            ggml_free(ctx);
            return false;
        }
        std::vector<int32_t> rows;
        std::vector<float> mfull, mswa;
        if (!kvflash_fill_rows_and_masks(*kvflash, kv_start, n_tok, mk_w,
                                         w.sliding_window, rows, &mfull, &mswa)) {
            ggml_free(ctx);
            return false;
        }
        if (!laguna_tensor_set_checked("laguna_step.kv_rows", kvi, rows.data(), 0, ggml_nbytes(kvi)) ||
            !laguna_tensor_set_checked("laguna_step.mask_full", mk_full, mfull.data(), 0, ggml_nbytes(mk_full)) ||
            !laguna_tensor_set_checked("laguna_step.mask_swa", mk_swa, mswa.data(), 0, ggml_nbytes(mk_swa))) {
            ggml_free(ctx);
            return false;
        }
    } else {
        if (kvi) {
            if (!laguna_tensor_set_checked("laguna_step.kv_idx", kvi, pos.data(), 0, ggml_nbytes(kvi))) {
                ggml_free(ctx);
                return false;
            }
        }

        if (!no_mask) {
            // Width mk_w (= kv_pad when padding): [kv_len, mk_w) stays -inf so
            // the zero-initialised padded cache tail contributes nothing.
            std::vector<float> mfull((size_t)mk_w * n_tok, -INFINITY);
            for (int q = 0; q < n_tok; ++q) {
                const int abs_q = kv_start + q;
                for (int k = 0; k <= abs_q && k < kv_len; ++k) {
                    mfull[(size_t)q * mk_w + k] = 0.0f;
                }
            }
            if (!laguna_tensor_set_checked("laguna_step.mask_full", mk_full, mfull.data(), 0,
                                           ggml_nbytes(mk_full))) {
                ggml_free(ctx);
                return false;
            }

            std::vector<float> mswa((size_t)mk_w * n_tok, -INFINITY);
            const int W = w.sliding_window;
            for (int q = 0; q < n_tok; ++q) {
                const int abs_q = kv_start + q;
                const int win_lo = std::max(0, abs_q - W + 1);
                for (int k = win_lo; k <= abs_q && k < kv_len; ++k) {
                    mswa[(size_t)q * mk_w + k] = 0.0f;
                }
            }
            if (!laguna_tensor_set_checked("laguna_step.mask_swa", mk_swa, mswa.data(), 0,
                                           ggml_nbytes(mk_swa))) {
                ggml_free(ctx);
                return false;
            }
        }
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "laguna_step: graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }

    if (out_argmax) {
        ggml_backend_tensor_get(argmax, out_argmax, 0, sizeof(int32_t));
    }
    if (read_logits) {
        out_logits.resize((size_t)w.embedder.n_vocab);
        ggml_backend_tensor_get(go.logits, out_logits.data(), 0,
                                out_logits.size() * sizeof(float));
    } else {
        out_logits.clear();
    }

    cache.cur_pos = kv_len;
    if (out_argmax) cache.last_tok = *out_argmax;
    ggml_free(ctx);
    return true;
}

bool laguna_verify_batch(
    ggml_backend_t              backend,
    const LagunaTargetWeights & w,
    LagunaTargetCache &         cache,
    const float *               embed,
    const int32_t *             token_ids,
    int                         n_tokens,
    int                         kv_start,
    std::vector<int32_t> &      out_argmax,
    const KvFlashPager *        kvflash,
    std::vector<float> *        out_logits)
{
    (void)token_ids;
    if (n_tokens <= 0) return false;

    const int kv_len = kv_start + n_tokens;
    static const bool g_no_kvpad = (std::getenv("DFLASH_LAGUNA_NO_KVPAD") != nullptr);
    static const bool g_pad_cpy = (std::getenv("DFLASH_LAGUNA_PAD_CPY") != nullptr);
    int kv_cap = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        if (cache.attn_k[(size_t)il]) { kv_cap = (int)cache.attn_k[(size_t)il]->ne[1]; break; }
    }
    const int kv_pad = (!g_no_kvpad && kv_cap > 0)
        ? std::min((kv_len + 255) & ~255, kv_cap) : 0;
    const int mk_w = kv_pad > 0 ? kv_pad : kv_len;

    static const bool g_verify_cache_disabled =
        (std::getenv("DFLASH_LAGUNA_VERIFY_CACHE_DISABLE") != nullptr);
    const bool can_reuse_verify_graph =
        !g_verify_cache_disabled && n_tokens <= 16 && !kvflash && out_logits == nullptr &&
        kv_pad > 0 && !g_pad_cpy;
    if (can_reuse_verify_graph) {
        struct CachedVerifyGraph {
            const LagunaTargetWeights * w_ptr = nullptr;
            LagunaTargetCache * cache_ptr = nullptr;
            ggml_backend_t backend = nullptr;
            ggml_tensor * target_feat = nullptr;
            int n_tokens = 0;
            int mk_w = 0;
            int kv_pad = 0;
            bool capture_features = false;
            std::vector<uint8_t> arena;
            ggml_context * ctx = nullptr;
            ggml_cgraph * gf = nullptr;
            ggml_gallocr_t alloc = nullptr;
            ggml_tensor * inp_embed = nullptr;
            ggml_tensor * positions = nullptr;
            ggml_tensor * kv_idx = nullptr;
            ggml_tensor * mask_full = nullptr;
            ggml_tensor * mask_swa = nullptr;
            ggml_tensor * feat_rows = nullptr;
            ggml_tensor * argmax = nullptr;
            std::vector<int32_t> pos;
            std::vector<int32_t> feat_idx;
            std::vector<float> full_mask;
            std::vector<float> swa_mask;

            void clear() {
                if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
                if (ctx) { ggml_free(ctx); ctx = nullptr; }
                gf = nullptr;
                inp_embed = positions = kv_idx = mask_full = mask_swa = feat_rows = argmax = nullptr;
                w_ptr = nullptr;
                cache_ptr = nullptr;
                backend = nullptr;
                target_feat = nullptr;
                n_tokens = 0;
                mk_w = 0;
                kv_pad = 0;
                capture_features = false;
            }
        };
        static thread_local CachedVerifyGraph cached_by_width[17];

        CachedVerifyGraph & cached = cached_by_width[n_tokens];
        const bool capture_features = cache.target_feat && cache.target_feat_cap > 0;
        const bool rebuild =
            cached.ctx == nullptr || cached.w_ptr != &w || cached.cache_ptr != &cache ||
            cached.backend != backend || cached.target_feat != cache.target_feat ||
            cached.n_tokens != n_tokens || cached.mk_w != mk_w ||
            cached.kv_pad != kv_pad || cached.capture_features != capture_features;
        if (rebuild) {
            cached.clear();
            const size_t arena_size =
                ggml_tensor_overhead() * 16384 + ggml_graph_overhead() + 16 * 1024 * 1024;
            cached.arena.resize(arena_size);

            ggml_init_params ip{};
            ip.mem_size = arena_size;
            ip.mem_buffer = cached.arena.data();
            ip.no_alloc = true;
            cached.ctx = ggml_init(ip);
            if (!cached.ctx) return false;
            cached.gf = ggml_new_graph_custom(cached.ctx, 16384, false);

            cached.inp_embed = ggml_new_tensor_3d(cached.ctx, GGML_TYPE_F32,
                                                  w.n_embd, n_tokens, 1);
            ggml_set_input(cached.inp_embed);
            cached.positions = ggml_new_tensor_1d(cached.ctx, GGML_TYPE_I32, n_tokens);
            ggml_set_input(cached.positions);
            cached.kv_idx = ggml_new_tensor_1d(cached.ctx, GGML_TYPE_I32, n_tokens);
            ggml_set_input(cached.kv_idx);
            cached.mask_full = ggml_new_tensor_4d(cached.ctx, GGML_TYPE_F32,
                                                  mk_w, n_tokens, 1, 1);
            ggml_set_input(cached.mask_full);
            ggml_tensor * mask_full_cnv = ggml_cast(cached.ctx, cached.mask_full, GGML_TYPE_F16);
            cached.mask_swa = ggml_new_tensor_4d(cached.ctx, GGML_TYPE_F32,
                                                 mk_w, n_tokens, 1, 1);
            ggml_set_input(cached.mask_swa);
            ggml_tensor * mask_swa_cnv = ggml_cast(cached.ctx, cached.mask_swa, GGML_TYPE_F16);

            if (capture_features) {
                cached.feat_rows = ggml_new_tensor_1d(cached.ctx, GGML_TYPE_I32, n_tokens);
                ggml_set_input(cached.feat_rows);
            }

            LagunaGraphInputs gi{};
            gi.inp_embed        = cached.inp_embed;
            gi.positions        = cached.positions;
            gi.attn_mask        = mask_full_cnv;
            gi.attn_mask_swa    = mask_swa_cnv;
            gi.n_tokens         = n_tokens;
            gi.kv_start         = 0;
            gi.kv_pad           = kv_pad;
            gi.kv_idx           = cached.kv_idx;
            gi.output_last_only = false;
            gi.output_logits    = true;
            gi.logits_are_output = false;
            gi.capture_features = capture_features;
            gi.target_feat_rows = cached.feat_rows;
            gi.hybrid           = nullptr;

            LagunaGraphOutputs go = build_laguna_graph(cached.ctx, cached.gf, w, cache, gi);
            cached.argmax = ggml_argmax(cached.ctx, go.logits);
            ggml_set_output(cached.argmax);
            ggml_build_forward_expand(cached.gf, cached.argmax);

            cached.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!cached.alloc || !ggml_gallocr_alloc_graph(cached.alloc, cached.gf)) {
                std::fprintf(stderr, "laguna_verify_batch: cached gallocr_alloc_graph failed\n");
                cached.clear();
                return false;
            }

            cached.w_ptr = &w;
            cached.cache_ptr = &cache;
            cached.backend = backend;
            cached.target_feat = cache.target_feat;
            cached.n_tokens = n_tokens;
            cached.mk_w = mk_w;
            cached.kv_pad = kv_pad;
            cached.capture_features = capture_features;
            cached.pos.resize((size_t)n_tokens);
            cached.feat_idx.resize((size_t)n_tokens);
            cached.full_mask.resize((size_t)mk_w * (size_t)n_tokens);
            cached.swa_mask.resize((size_t)mk_w * (size_t)n_tokens);
        }

        if (!laguna_tensor_set_checked("laguna_verify_cached.ie", cached.inp_embed,
                                       embed, 0, ggml_nbytes(cached.inp_embed))) {
            return false;
        }

        for (int i = 0; i < n_tokens; ++i) {
            cached.pos[(size_t)i] = kv_start + i;
        }
        if (!laguna_tensor_set_checked("laguna_verify_cached.positions", cached.positions,
                                       cached.pos.data(), 0, ggml_nbytes(cached.positions)) ||
            !laguna_tensor_set_checked("laguna_verify_cached.kv_idx", cached.kv_idx,
                                       cached.pos.data(), 0, ggml_nbytes(cached.kv_idx))) {
            return false;
        }

        if (cached.feat_rows) {
            for (int i = 0; i < n_tokens; ++i) {
                cached.feat_idx[(size_t)i] = (kv_start + i) % cache.target_feat_cap;
            }
        if (!laguna_tensor_set_checked("laguna_verify_cached.feat_rows", cached.feat_rows,
                                       cached.feat_idx.data(), 0,
                                       ggml_nbytes(cached.feat_rows), false)) {
            return false;
        }
        }

        std::fill(cached.full_mask.begin(), cached.full_mask.end(), -INFINITY);
        for (int q = 0; q < n_tokens; ++q) {
            const int abs_q = kv_start + q;
            for (int k = 0; k <= abs_q && k < kv_len && k < mk_w; ++k) {
                cached.full_mask[(size_t)q * (size_t)mk_w + (size_t)k] = 0.0f;
            }
        }
        if (!laguna_tensor_set_checked("laguna_verify_cached.mask_full", cached.mask_full,
                                       cached.full_mask.data(), 0,
                                       ggml_nbytes(cached.mask_full))) {
            return false;
        }

        std::fill(cached.swa_mask.begin(), cached.swa_mask.end(), -INFINITY);
        const int W = w.sliding_window;
        for (int q = 0; q < n_tokens; ++q) {
            const int abs_q = kv_start + q;
            const int win_lo = std::max(0, abs_q - W + 1);
            for (int k = win_lo; k <= abs_q && k < kv_len && k < mk_w; ++k) {
                cached.swa_mask[(size_t)q * (size_t)mk_w + (size_t)k] = 0.0f;
            }
        }
        if (!laguna_tensor_set_checked("laguna_verify_cached.mask_swa", cached.mask_swa,
                                       cached.swa_mask.data(), 0,
                                       ggml_nbytes(cached.mask_swa))) {
            return false;
        }

        if (ggml_backend_graph_compute(backend, cached.gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "laguna_verify_batch: cached graph_compute failed\n");
            return false;
        }

        out_argmax.resize((size_t)n_tokens);
        ggml_backend_tensor_get(cached.argmax, out_argmax.data(), 0,
                                sizeof(int32_t) * (size_t)n_tokens);
        cache.cur_pos = kv_len;
        cache.last_tok = out_argmax.empty() ? -1 : out_argmax.back();
        return true;
    }

    const size_t arena_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() + 16 * 1024 * 1024;
    static thread_local std::vector<uint8_t> g_arena_block;
    static thread_local std::vector<uint8_t> g_arena_bonus;
    std::vector<uint8_t> & g_arena = (n_tokens == 1) ? g_arena_bonus : g_arena_block;
    if (g_arena.size() < arena_size) g_arena.resize(arena_size);
    ggml_init_params ip{};
    ip.mem_size = arena_size;
    ip.mem_buffer = g_arena.data();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);

    ggml_tensor * ie = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w.n_embd, n_tokens, 1);
    ggml_set_input(ie);
    ggml_tensor * pp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(pp);

    ggml_tensor * kvi = nullptr;
    if (kv_pad > 0 && !g_pad_cpy) {
        kvi = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(kvi);
    }

    ggml_tensor * mk_full = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, mk_w, n_tokens, 1, 1);
    ggml_set_input(mk_full);
    ggml_tensor * mk_full_cnv = ggml_cast(ctx, mk_full, GGML_TYPE_F16);
    ggml_tensor * mk_swa = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, mk_w, n_tokens, 1, 1);
    ggml_set_input(mk_swa);
    ggml_tensor * mk_swa_cnv = ggml_cast(ctx, mk_swa, GGML_TYPE_F16);

    ggml_tensor * feat_rows = nullptr;
    if (cache.target_feat && cache.target_feat_cap > 0) {
        feat_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(feat_rows);
    }

    LagunaGraphInputs gi{};
    gi.inp_embed        = ie;
    gi.positions        = pp;
    gi.attn_mask        = mk_full_cnv;
    gi.attn_mask_swa    = mk_swa_cnv;
    gi.n_tokens         = n_tokens;
    gi.kv_start         = kv_start;
    gi.kv_pad           = kv_pad;
    gi.kv_idx           = kvi;
    gi.output_last_only = false;
    gi.output_logits    = true;
    gi.logits_are_output = out_logits != nullptr;
    gi.capture_features = true;
    gi.target_feat_rows = feat_rows;
    gi.hybrid           = nullptr;

    LagunaGraphOutputs go = build_laguna_graph(ctx, gf, w, cache, gi);
    ggml_tensor * argmax = ggml_argmax(ctx, go.logits);
    ggml_set_output(argmax);
    ggml_build_forward_expand(gf, argmax);

    static ggml_gallocr_t galloc_verify_block = nullptr;
    static ggml_gallocr_t galloc_verify_bonus = nullptr;
    ggml_gallocr_t & galloc_verify = (n_tokens == 1) ? galloc_verify_bonus : galloc_verify_block;
    if (!galloc_verify) galloc_verify = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc_verify, gf)) {
        std::fprintf(stderr, "laguna_verify_batch: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    if (!laguna_tensor_set_checked("laguna_verify.ie", ie, embed, 0, ggml_nbytes(ie))) {
        ggml_free(ctx);
        return false;
    }
    std::vector<int32_t> pos((size_t)n_tokens);
    for (int i = 0; i < n_tokens; ++i) pos[(size_t)i] = kv_start + i;
    if (!laguna_tensor_set_checked("laguna_verify.positions", pp, pos.data(), 0, ggml_nbytes(pp))) {
        ggml_free(ctx);
        return false;
    }
    if (feat_rows) {
        std::vector<int32_t> feat_idx((size_t)n_tokens);
        for (int i = 0; i < n_tokens; ++i) {
            feat_idx[(size_t)i] = (kv_start + i) % cache.target_feat_cap;
        }
        if (!laguna_tensor_set_checked("laguna_verify.feat_rows", feat_rows,
                                       feat_idx.data(), 0, ggml_nbytes(feat_rows), false)) {
            ggml_free(ctx);
            return false;
        }
    }

    if (kvflash) {
        if (!kvi) {
            std::fprintf(stderr, "laguna_verify_batch: kvflash requires the kv_pad "
                                 "set_rows path (NO_KVPAD / PAD_CPY are incompatible)\n");
            ggml_free(ctx);
            return false;
        }
        std::vector<int32_t> rows;
        std::vector<float> mfull, mswa;
        if (!kvflash_fill_rows_and_masks(*kvflash, kv_start, n_tokens, mk_w,
                                         w.sliding_window, rows, &mfull, &mswa)) {
            ggml_free(ctx);
            return false;
        }
        if (!laguna_tensor_set_checked("laguna_verify.kv_rows", kvi, rows.data(), 0, ggml_nbytes(kvi)) ||
            !laguna_tensor_set_checked("laguna_verify.mask_full", mk_full, mfull.data(), 0, ggml_nbytes(mk_full)) ||
            !laguna_tensor_set_checked("laguna_verify.mask_swa", mk_swa, mswa.data(), 0, ggml_nbytes(mk_swa))) {
            ggml_free(ctx);
            return false;
        }
    } else {
        if (kvi) {
            if (!laguna_tensor_set_checked("laguna_verify.kv_idx", kvi, pos.data(), 0, ggml_nbytes(kvi))) {
                ggml_free(ctx);
                return false;
            }
        }

        std::vector<float> mfull((size_t)mk_w * n_tokens, -INFINITY);
        for (int q = 0; q < n_tokens; ++q) {
            const int abs_q = kv_start + q;
            for (int k = 0; k <= abs_q && k < kv_len; ++k) {
                mfull[(size_t)q * mk_w + k] = 0.0f;
            }
        }
        if (!laguna_tensor_set_checked("laguna_verify.mask_full", mk_full, mfull.data(), 0,
                                       ggml_nbytes(mk_full))) {
            ggml_free(ctx);
            return false;
        }

        std::vector<float> mswa((size_t)mk_w * n_tokens, -INFINITY);
        const int W = w.sliding_window;
        for (int q = 0; q < n_tokens; ++q) {
            const int abs_q = kv_start + q;
            const int win_lo = std::max(0, abs_q - W + 1);
            for (int k = win_lo; k <= abs_q && k < kv_len; ++k) {
                mswa[(size_t)q * mk_w + k] = 0.0f;
            }
        }
        if (!laguna_tensor_set_checked("laguna_verify.mask_swa", mk_swa, mswa.data(), 0,
                                       ggml_nbytes(mk_swa))) {
            ggml_free(ctx);
            return false;
        }
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "laguna_verify_batch: graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }

    out_argmax.resize((size_t)n_tokens);
    ggml_backend_tensor_get(argmax, out_argmax.data(), 0,
                            sizeof(int32_t) * (size_t)n_tokens);
    if (out_logits) {
        const int vocab = (int)w.embedder.n_vocab;
        out_logits->resize((size_t)vocab * (size_t)n_tokens);
        ggml_backend_tensor_get(go.logits, out_logits->data(), 0,
                                sizeof(float) * out_logits->size());
    }

    cache.cur_pos = kv_len;
    cache.last_tok = out_argmax.empty() ? -1 : out_argmax.back();
    ggml_free(ctx);
    return true;
}

bool laguna_project_hidden(
    ggml_backend_t              backend,
    const LagunaTargetWeights & w,
    const float *               hidden,
    int                         n_tokens,
    std::vector<int32_t> &      out_tokens)
{
    if (n_tokens <= 0) return false;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead() + 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_set_input(inp);

    ggml_tensor * cur = ggml_mul_mat(ctx, w.output, inp);  // [vocab, n_tokens]
    cur = ggml_argmax(ctx, cur);                           // [n_tokens]
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    static ggml_gallocr_t galloc_proj = nullptr;
    if (!galloc_proj) galloc_proj = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc_proj, gf)) {
        std::fprintf(stderr, "laguna_project_hidden: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(inp, hidden, 0,
                            sizeof(float) * (size_t)n_tokens * (size_t)w.n_embd);

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "laguna_project_hidden: graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }

    out_tokens.resize((size_t)n_tokens);
    ggml_backend_tensor_get(cur, out_tokens.data(), 0,
                            sizeof(int32_t) * (size_t)n_tokens);

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
    std::vector<int32_t> *      out_selected,
    const KvFlashPager *        kvflash)
{
    if (kvflash && no_mask) {
        std::fprintf(stderr, "laguna_step_hybrid: kvflash requires masks (slots "
                             "are relocated; position-implicit masking is invalid)\n");
        return false;
    }
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

    if (!laguna_tensor_set_checked("laguna_step_hybrid.ie", ie, embed, 0, ggml_nbytes(ie))) {
        ggml_free(ctx);
        return false;
    }
    std::vector<int32_t> pos((size_t)n_tok);
    for (int i = 0; i < n_tok; ++i) pos[i] = kv_start + i;
    if (!laguna_tensor_set_checked("laguna_step_hybrid.positions", pp, pos.data(), 0, ggml_nbytes(pp))) {
        ggml_free(ctx);
        return false;
    }

    if (kvflash) {
        if (!kvi) {
            std::fprintf(stderr, "laguna_step_hybrid: kvflash requires the kv_pad "
                                 "set_rows path (NO_KVPAD / PAD_CPY are incompatible)\n");
            ggml_free(ctx);
            return false;
        }
        std::vector<int32_t> rows;
        std::vector<float> mfull, mswa;
        if (!kvflash_fill_rows_and_masks(*kvflash, kv_start, n_tok, mk_w,
                                         w.sliding_window, rows, &mfull, &mswa)) {
            ggml_free(ctx);
            return false;
        }
        if (!laguna_tensor_set_checked("laguna_step_hybrid.kv_rows", kvi, rows.data(), 0, ggml_nbytes(kvi)) ||
            !laguna_tensor_set_checked("laguna_step_hybrid.mask_full", mk_full, mfull.data(), 0, ggml_nbytes(mk_full)) ||
            !laguna_tensor_set_checked("laguna_step_hybrid.mask_swa", mk_swa, mswa.data(), 0, ggml_nbytes(mk_swa))) {
            ggml_free(ctx);
            return false;
        }
    } else {
    if (kvi) {
        // set_rows row indices = absolute cache positions of this step's tokens
        if (!laguna_tensor_set_checked("laguna_step_hybrid.kv_idx", kvi, pos.data(), 0, ggml_nbytes(kvi))) {
            ggml_free(ctx);
            return false;
        }
    }

    if (!no_mask) {
        // Width mk_w (= kv_pad when padding): columns in [kv_len, mk_w) stay
        // -inf so the zero-initialised padded cache tail contributes nothing.
        std::vector<float> mfull((size_t)mk_w * n_tok, -INFINITY);
        for (int q = 0; q < n_tok; ++q) {
            const int abs_q = kv_start + q;
            for (int k = 0; k <= abs_q && k < kv_len; ++k) mfull[(size_t)q * mk_w + k] = 0.0f;
        }
        if (!laguna_tensor_set_checked("laguna_step_hybrid.mask_full", mk_full, mfull.data(), 0,
                                       ggml_nbytes(mk_full))) {
            ggml_free(ctx);
            return false;
        }
        std::vector<float> mswa((size_t)mk_w * n_tok, -INFINITY);
        const int Wsw = w.sliding_window;
        for (int q = 0; q < n_tok; ++q) {
            const int abs_q = kv_start + q;
            const int lo = std::max(0, abs_q - Wsw + 1);
            for (int k = lo; k <= abs_q && k < kv_len; ++k) mswa[(size_t)q * mk_w + k] = 0.0f;
        }
        if (!laguna_tensor_set_checked("laguna_step_hybrid.mask_swa", mk_swa, mswa.data(), 0,
                                       ggml_nbytes(mk_swa))) {
            ggml_free(ctx);
            return false;
        }
    }
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
    if (!laguna_tensor_set_checked("laguna_step_hybrid.lut_all", hybm.lut_all,
                                   lutbuf.data(), 0, sizeof(int32_t) * lutbuf.size()) ||
        !laguna_tensor_set_checked("laguna_step_hybrid.vld_all", hybm.vld_all,
                                   vldbuf.data(), 0, sizeof(float) * vldbuf.size())) {
        ggml_free(ctx);
        return false;
    }

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
