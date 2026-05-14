// Single-step MTP (Multi-Token Prediction) graph builder for Gemma4.
//
// Builds a ggml compute graph that, given one token id and the target's last
// full-attention hidden state h_prev, produces:
//   - out_logits  : F32 [n_vocab, 1]  full vocabulary row
//   - out_h_post  : F32 [n_embd_backbone, 1]  next h_prev for the γ chain
//   - out_argmax  : I32 [1]  greedy draft token (4-byte host pull per step)
//
// Architecture (mirrors atomicbot's gemma4-assistant.cpp lines 28-256):
//   1. Token embedding from target.tok_embd, scaled by sqrt(n_embd_backbone).
//   2. Concat [tok_emb, h_prev] → pre_projection → [n_embd, 1].
//   3. 4 transformer blocks (cross-attention into target KV):
//      RMSNorm → Q proj → Q-norm → RoPE → cross-attn (reads donor K/V) →
//      wo → post_attn_norm → residual → ffn_norm → GELU FFN → post_ffn_norm →
//      residual → optional out_scale.
//   4. output_norm → post_projection → h_post  [n_embd_backbone, 1].
//   5. LM head: dense (tied tok_embd) or centroid-routed for ordered embeddings.
//   6. In-graph argmax.
//
// Cross-attention contract:
//   - Each MTP layer reads K/V from w.layers[il].donor_target_layer in the
//     target KV cache (resolved at load time as the LAST target layer whose
//     SWA type matches this MTP layer).
//   - V is ALWAYS read from the cache (use_k_as_v=false): per HF Gemma4 the
//     V slot stores rms-normed non-rotated vectors, distinct from post-RoPE K.
//   - The K/V view covers [0, attn_pos) = all committed target positions.
//     attn_pos is passed in via the in_pos tensor (caller sets it to
//     cache.cur_pos before each step).
//   - KV mask is not needed: all committed positions ≤ attn_pos are uniformly
//     admitted (step position > attn_pos, so every cell is in the causal cone).
//     We pass nullptr to ggml_flash_attn_ext for the mask argument.
//
// Centroid LM head (use_ordered_embeddings=true, always active for Dense 31B):
//   cent_logits = mul_mat(mtp_centroids, h_inner)
//   top_k_ids   = ggml_top_k(cent_logits, centroid_top_k)
//   sel_ids     = get_rows(token_ordering_view, top_k_ids)
//   sel_logits  = mul_mat(get_rows(tok_embd, flat_sel_ids), h_inner)
//   full_row    = scatter sel_logits into [-1e30 fill] via ggml_set_rows
//
// When use_ordered_embeddings is false (fallback, unlikely for 31B assistant):
//   out_logits = mul_mat(tok_embd, h_inner)  — dense tied head.

#include "internal.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>

namespace dflash27b {

static constexpr float MTP_RMS_EPS = GEMMA4_RMS_EPS;

// ─── Helpers ─────────────────────────────────────────────────────────────────

static ggml_tensor * mtp_rms_norm_mul(ggml_context * ctx,
                                       ggml_tensor  * x,
                                       ggml_tensor  * weight) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, MTP_RMS_EPS);
    return ggml_mul(ctx, n, weight);
}

// GELU FFN with SwiGLU-like gate: w_down @ (gelu(w_gate @ x) * (w_up @ x))
static ggml_tensor * mtp_gelu_ffn(ggml_context        * ctx,
                                   ggml_tensor         * cur,
                                   const MtpLayerWeights & L) {
    ggml_tensor * gate = ggml_mul_mat(ctx, L.ffn_gate, cur);
    ggml_tensor * up   = ggml_mul_mat(ctx, L.ffn_up,   cur);
    ggml_tensor * gu   = ggml_geglu_split(ctx, gate, up);
    return ggml_mul_mat(ctx, L.ffn_down, gu);
}

// ─── Public graph builder ─────────────────────────────────────────────────────

bool build_mtp_step_graph(const MtpDrafterWeights  & w,
                          const GemmaTargetCache   & target_cache,
                          const GemmaTargetWeights & target,
                          MtpStepGraph             & out,
                          int                        attn_pos) {
    // ── Validate prerequisites ────────────────────────────────────────────────
    if (!w.pre_projection || !w.post_projection || !w.output_norm) {
        set_last_error("build_mtp_step_graph: MtpDrafterWeights missing pre/post projection or output_norm");
        return false;
    }
    if ((int)w.layers.size() == 0) {
        set_last_error("build_mtp_step_graph: no MTP layers");
        return false;
    }
    if (!target.tok_embd) {
        set_last_error("build_mtp_step_graph: target.tok_embd is null");
        return false;
    }
    if (w.n_embd == 0 || w.n_embd_backbone == 0) {
        set_last_error("build_mtp_step_graph: n_embd or n_embd_backbone is 0");
        return false;
    }

    const int n_embd_backbone = w.n_embd_backbone;
    const int n_layer         = (int)w.layers.size();
    const int n_vocab         = (int)target.tok_embd->ne[1];

    // Donor KV slot validation runs inside the per-layer loop below; layer 0
    // is no special case (the loop iterates il=0..n_layer-1). The previously
    // hoisted layer-0 pre-check was identical to the il==0 iteration and
    // only ran one bounds check earlier, which is not worth duplicating.

    // ── Allocate ggml context ─────────────────────────────────────────────────
    // Conservative ggml-tensor overhead estimate. Each MTP layer emits
    // roughly the following named ops (≈80 total in the worst case):
    //   RMS-norm (in, out)               : 6
    //   Q proj + Q-norm + RoPE           : 5
    //   K/V cache views (with TQ3 wrap)  : 8
    //   K/V casts (TQ3 → F16 → F32)      : 6
    //   GQA block-broadcast view + cont  : 4
    //   Q permute + cont                 : 2
    //   FA inputs + FA + FA mask         : 6
    //   attn output, wo                  : 3
    //   post-attn-norm + residual        : 4
    //   FFN (gate, up, geglu, down)      : 6
    //   ffn-norm + residual              : 4
    //   set_name / set_input ceremony    : ≈25
    //   Reserve / rounding               : ≈3
    // Plus 3 inputs (in_tok_embd, in_h_prev, in_pos) and ~20 ops for the
    // pre-projection, output_norm, post-projection, LM head, argmax tail.
    // Final term `(n_layer * 80 + 20 + 3)`. Over-allocation is benign;
    // under-allocation aborts in ggml_init.
    const size_t n_tensors_est = (size_t)(3 + n_layer * 80 + 20);
    ggml_init_params ip{};
    ip.mem_size   = n_tensors_est * ggml_tensor_overhead() + 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        set_last_error("build_mtp_step_graph: ggml_init failed");
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);

    // ── Input tensors ─────────────────────────────────────────────────────────
    ggml_tensor * in_tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(in_tok);
    ggml_set_name(in_tok, "mtp_in_tok");

    // in_tok_embd: pre-dequantised token embedding supplied by caller.
    // Caller must call target.embedder.embed(&tok, 1, buf) and tensor_set before compute.
    // This avoids ggml_get_rows on a k-quant (Q4_K) source which the CUDA backend
    // does not support in this llama.cpp revision.
    ggml_tensor * in_tok_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd_backbone, 1);
    ggml_set_input(in_tok_embd);
    ggml_set_name(in_tok_embd, "mtp_in_tok_embd");

    ggml_tensor * in_h_prev = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd_backbone, 1);
    ggml_set_input(in_h_prev);
    ggml_set_name(in_h_prev, "mtp_in_h_prev");

    // in_pos: absolute target position for this draft step's RoPE.
    // Caller sets this to (cache.cur_pos + step_offset).
    ggml_tensor * in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(in_pos);
    ggml_set_name(in_pos, "mtp_in_pos");

    // ── 1. Token embedding from target (shared weight) ────────────────────────
    // Embedding is passed in pre-dequantised by the caller via in_tok_embd.
    // This bypasses ggml_get_rows on a potentially quantised target.tok_embd
    // (CUDA backend in this revision only supports F16/F32/Q8_0 for get_rows;
    // Q4_K targets would abort at compute time).
    ggml_tensor * tok_e = in_tok_embd;
    ggml_set_name(tok_e, "mtp_tok_embd");

    // Gemma4 scales token embeddings by sqrt(n_embd_backbone) at input pipeline
    const float tok_scale = std::sqrt((float)n_embd_backbone);
    tok_e = ggml_scale(ctx, tok_e, tok_scale);
    ggml_set_name(tok_e, "mtp_tok_embd_scaled");

    // ── 2. Concat [tok_e, h_prev] and project to n_embd ──────────────────────
    // Both are [n_embd_backbone, 1]; concat on axis 0 → [2*n_embd_backbone, 1]
    ggml_tensor * inp_cat = ggml_concat(ctx, tok_e, in_h_prev, 0);
    ggml_set_name(inp_cat, "mtp_concat");

    // pre_projection: [2*n_embd_backbone, n_embd] (ggml ne[0]=2*n_bb, ne[1]=n_embd)
    // mul_mat(A, x): A->ne[0] must == x->ne[0]; output ne[0]=A->ne[1]
    ggml_tensor * inpL = ggml_mul_mat(ctx, w.pre_projection, inp_cat);
    ggml_set_name(inpL, "mtp_pre_proj_out");

    // ── 3. Transformer blocks ─────────────────────────────────────────────────
    // Single FA mask shared across every layer that needs one. First need-mask
    // layer creates the input tensor; later layers reuse it. We require every
    // need-mask layer to want the same (width, kv_seq_len) — short contexts
    // satisfy this because SWA cap >= attn_pos. Divergence in long contexts
    // trips an error and the builder must be extended to per-layer masks.
    ggml_tensor * shared_fa_mask           = nullptr;
    int64_t       shared_fa_mask_width     = 0;
    int64_t       shared_fa_mask_kv_seq_len = 0;
    for (int il = 0; il < n_layer; ++il) {
        const MtpLayerWeights & L = w.layers[il];
        const bool is_swa = L.is_swa;

        // Resolve donor KV slot
        const int32_t donor_il = L.donor_target_layer;
        if (donor_il < 0 || donor_il >= (int)target_cache.layer_to_kv_idx.size()) {
            set_last_error("build_mtp_step_graph: invalid donor_target_layer");
            ggml_free(ctx);
            return false;
        }
        const int kv_slot = target_cache.layer_to_kv_idx[donor_il];
        const int kv_read_slot = (kv_slot >= 0) ? kv_slot
            : ((donor_il < (int)target_cache.layer_to_donor_kv.size())
                ? target_cache.layer_to_donor_kv[donor_il] : -1);
        if (kv_read_slot < 0 || kv_read_slot >= (int)target_cache.attn_k.size()) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "build_mtp_step_graph: donor KV slot unresolvable for MTP layer %d", il);
            set_last_error(buf);
            ggml_free(ctx);
            return false;
        }
        ggml_tensor * cache_k = target_cache.attn_k[kv_read_slot];
        ggml_tensor * cache_v = target_cache.attn_v[kv_read_slot];
        if (!cache_k || !cache_v) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "build_mtp_step_graph: null KV cache for MTP layer %d donor slot %d", il, kv_read_slot);
            set_last_error(buf);
            ggml_free(ctx);
            return false;
        }

        // KV cache layout: [head_dim_kv, max_ctx, n_head_kv]
        const int64_t head_dim_kv = cache_k->ne[0];
        const int64_t n_head_kv   = cache_k->ne[2];
        // Q dimensions: derive from wq output size and attn_q_norm shape.
        // wq:         [n_embd, q_out_dim]  where q_out_dim = n_head_norm * head_dim_norm
        // attn_q_norm:[head_dim_norm]       per-head norm weight from the MTP model's own hparams
        //
        // head_dim_norm may differ from head_dim_kv (the target KV cache head_dim).
        // Dense 31B example: MTP trained with head_dim_norm=256, target K stored at 128.
        // For flash_attn Q @ K^T to succeed, Q.ne[0] must equal K.ne[0].
        // Fix: norm and RoPE run at head_dim_norm; before FA, reshape Q to [head_dim_kv, ...]
        // so the dot-product dimension matches K.  q_out_dim is preserved throughout.
        const int64_t q_out_dim    = L.wq->ne[1];
        const int64_t head_dim_norm = L.attn_q_norm->ne[0];  // MTP model's per-head norm dim
        const int64_t n_head_norm   = q_out_dim / head_dim_norm;
        // FA head_dim must match target K; use head_dim_kv (from cache_k->ne[0]).
        const int64_t head_dim_fa  = head_dim_kv;
        const int64_t n_head_fa    = q_out_dim / head_dim_fa;

        // a) RMSNorm
        ggml_tensor * cur = mtp_rms_norm_mul(ctx, inpL, L.attn_norm);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_attn_norm_%d", il);
            ggml_set_name(cur, name);
        }

        // b) Q projection: [n_embd, 1] → [q_out_dim, 1], reshape to [head_dim_norm, n_head_norm, 1]
        ggml_tensor * Qcur = ggml_mul_mat(ctx, L.wq, cur);
        Qcur = ggml_reshape_3d(ctx, Qcur, head_dim_norm, n_head_norm, 1);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_Qcur_%d", il);
            ggml_set_name(Qcur, name);
        }

        // c) Q-norm: per-head RMSNorm at head_dim_norm (attn_q_norm shape: [head_dim_norm])
        Qcur = mtp_rms_norm_mul(ctx, Qcur, L.attn_q_norm);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_Qcur_normed_%d", il);
            ggml_set_name(Qcur, name);
        }

        // d) RoPE on Q at head_dim_norm
        // Use the target's rope_theta (SWA layers) or the full-attn layer's rope_freqs.
        // For MTP cross-attention: SWA layers use rope_theta_swa, full layers use rope_theta
        // (with per-layer freq_factors from the donor layer).
        // We use the target's SWA/full rope parameters mirroring atomicbot.
        ggml_tensor * rope_freq_factors = nullptr;
        float rope_theta_val = target.rope_theta_swa;
        if (!is_swa) {
            rope_theta_val = target.rope_theta;
            // For full-attention MTP layers: prefer assistant's OWN rope_freqs
            // (top-level "rope_freqs.weight" in assistant GGUF — the assistant
            // was trained with its own per-dim freq factors). Fall back to
            // target's per-layer rope_freqs only if the assistant didn't ship
            // one (legacy GGUFs).
            if (w.rope_freqs) {
                rope_freq_factors = w.rope_freqs;
            } else if (donor_il >= 0 && donor_il < (int)target.layers.size()) {
                rope_freq_factors = target.layers[donor_il].rope_freqs;
            }
        }
        Qcur = ggml_rope_ext(ctx, Qcur, in_pos,
                              rope_freq_factors,
                              (int)head_dim_norm, GGML_ROPE_TYPE_NEOX,
                              /*n_ctx_orig=*/0,
                              rope_theta_val, /*freq_scale=*/1.0f,
                              /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                              /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_Qcur_pos_%d", il);
            ggml_set_name(Qcur, name);
        }

        // e) Cross-attention (manual: Q@K^T → scale → softmax → @V)
        // Sidesteps ggml_flash_attn_ext CUDA kernel shape restrictions for MTP.
        //
        // Make Qcur contiguous before reshape — ggml_rope_ext returns a non-contiguous
        // view; ggml_reshape_3d requires a contiguous source.
        Qcur = ggml_cont(ctx, Qcur);
        // Reshape Q from [head_dim_norm, n_head_norm, 1] to [head_dim_fa, n_head_fa, 1]
        // so Q.ne[0] == K.ne[0] == head_dim_kv.
        // When head_dim_norm == head_dim_fa this is a no-op reshape.
        Qcur = ggml_reshape_3d(ctx, Qcur, head_dim_fa, n_head_fa, 1);

        // K/V view from the target KV cache.
        // Full-attention donors read [0, attn_pos). SWA donors use a ring buffer:
        // slice only the keys admitted by atomicbot's STANDARD SWA mask for an MTP
        // query at pos=attn_pos, then the remaining mask is an all-zero bias.
        int64_t kv_seq_len = (int64_t)attn_pos;
        int64_t kv_start_slot = 0;
        bool kv_wraps = false;
        int64_t kv_first_len = 0;
        if (is_swa) {
            const int64_t ring_len = std::min(cache_k->ne[1], cache_v->ne[1]);
            const int64_t swa_prev = target.swa_window > 0
                ? std::max<int64_t>((int64_t)target.swa_window - 1, 0) : ring_len;
            kv_seq_len = std::min<int64_t>((int64_t)attn_pos, std::min(swa_prev, ring_len));
            if (kv_seq_len > 0) {
                const int64_t first_abs = (int64_t)attn_pos - kv_seq_len;
                kv_start_slot = first_abs % ring_len;
                const int64_t kv_end_slot = kv_start_slot + kv_seq_len;
                kv_wraps = kv_end_slot > ring_len;
                kv_first_len = kv_wraps ? (ring_len - kv_start_slot) : kv_seq_len;
            }
        } else if ((int64_t)attn_pos > cache_k->ne[1] || (int64_t)attn_pos > cache_v->ne[1]) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "build_mtp_step_graph: attn_pos %d exceeds donor KV cache length (K=%lld V=%lld) for MTP layer %d",
                attn_pos, (long long)cache_k->ne[1], (long long)cache_v->ne[1], il);
            set_last_error(buf);
            ggml_free(ctx);
            return false;
        }
        // Pad to 1 minimum to avoid zero-size tensors when attn_pos==0.
        const int64_t kv_view_len = std::max(kv_seq_len, (int64_t)1);

        // For head_dim==512 with any K type, ggml_flash_attn_ext requires
        // K->ne[1] % 256 == 0 for gqa_opt_applies to be true (and returns
        // BEST_FATTN_KERNEL_NONE otherwise). Pad the K/V view to the next 256
        // multiple; the padding rows contain stale cache data but are masked
        // out by the caller-provided fa_mask with -inf bias on those positions.
        // This only applies to the non-wrap path (head_dim=512 layers are full-attn
        // with monotone KV so no wrap occurs).
        // FATTN_KQ_STRIDE alignment: TQ3_0 K is stored in blocks along ne[1] and
        // the FA kernels (chunked + vec) iterate KV in 256-position groups; an
        // unaligned ne[1] reads past the valid window into stale cache cells. We
        // pad the view to 256 and exclude the tail with a -inf mask.
        // This matches gemma4_target_graph.cpp:352-355's `need_256_pad` policy.
        const bool kv_cache_is_tq3 =
            (cache_k->type == GGML_TYPE_TQ3_0 || cache_v->type == GGML_TYPE_TQ3_0);
        if (kv_wraps &&
            (cache_k->type == GGML_TYPE_TQ3_0 || cache_v->type == GGML_TYPE_TQ3_0)) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "build_mtp_step_graph: refusing wrapped TQ3 donor attention for MTP layer %d donor=%d; force donor KV to Q8_0",
                il, donor_il);
            set_last_error(buf);
            ggml_free(ctx);
            return false;
        }
        const bool needs_kv_pad = (kv_cache_is_tq3 || head_dim_fa >= 512)
                                  && !kv_wraps && (kv_view_len % 256 != 0);
        const int64_t kv_view_len_padded = needs_kv_pad
            ? ((kv_view_len + 255) / 256) * 256
            : kv_view_len;

        auto view_kv = [&](ggml_tensor * cache, int64_t start, int64_t len) {
            return ggml_view_3d(ctx, cache,
                head_dim_kv, len, n_head_kv,
                cache->nb[1], cache->nb[2],
                cache->nb[1] * (size_t)start);
        };

        ggml_tensor * Kview = nullptr;
        ggml_tensor * Vview = nullptr;
        if (kv_wraps) {
            // ggml_concat on CUDA requires F32 src. Direct TQ3_0→F32 is unsupported
            // by cpy.cu (it only does TQ3_0→F16 and F16↔F32). So go via F16 first
            // when the cache is TQ3, else cast directly.
            auto to_f32 = [&](ggml_tensor * v) {
                if (v->type == GGML_TYPE_TQ3_0) {
                    v = ggml_cast(ctx, v, GGML_TYPE_F16);
                }
                if (v->type != GGML_TYPE_F32) {
                    v = ggml_cast(ctx, v, GGML_TYPE_F32);
                }
                return v;
            };
            const int64_t kv_second_len = kv_view_len - kv_first_len;
            ggml_tensor * k1 = to_f32(view_kv(cache_k, kv_start_slot, kv_first_len));
            ggml_tensor * k2 = to_f32(view_kv(cache_k, 0,             kv_second_len));
            ggml_tensor * v1 = to_f32(view_kv(cache_v, kv_start_slot, kv_first_len));
            ggml_tensor * v2 = to_f32(view_kv(cache_v, 0,             kv_second_len));
            Kview = ggml_concat(ctx, k1, k2, 1);
            Vview = ggml_concat(ctx, v1, v2, 1);
        } else {
            // Use padded length for the K/V view when required.
            Kview = view_kv(cache_k, kv_start_slot, kv_view_len_padded);
            Vview = view_kv(cache_v, kv_start_slot, kv_view_len_padded);
        }
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_Kview_%d", il);
            ggml_set_name(Kview, name);
            std::snprintf(name, sizeof(name), "mtp_Vview_%d", il);
            ggml_set_name(Vview, name);
        }

        // Detect if K/V is in TQ3_0 (FWHT-domain). Graph-level FWHT keeps the
        // FA backends on a single contract: pre-rotate Q for TQ3 K, inverse-
        // rotate output for TQ3 V, and pass the native K/V views into FA.
        const bool k_is_tq3 = (Kview->type == GGML_TYPE_TQ3_0);
        const bool v_is_tq3 = (Vview->type == GGML_TYPE_TQ3_0);

        // Cross-attention via ggml_flash_attn_ext.
        //
        // Layout for ggml_flash_attn_ext:
        //   Q: [head_dim, n_tokens=1, n_head_q]
        //   K: [head_dim, kv_len,     n_head_kv]  (GQA: n_head_q % n_head_kv == 0)
        //   V: [head_dim, kv_len,     n_head_kv]
        //   output: [head_dim, n_tokens=1, n_head_q]  (reshaped to [q_out_dim, 1])
        //
        // Benefits over manual matmul attention:
        //   - Handles GQA directly without broadcasting K/V.
        //   - Graph-level FWHT correction keeps TQ3 K/V in their native cache domain.
        //
        // For TQ3_0 + head_dim > 256 + n_tokens=1 (decode), the CUDA dispatch
        // requires a non-null mask to select the CHUNKED kernel path. We create
        // an all-zero (fully-admitted) mask in that case.
        //
        // Permute Q from [head_dim_fa, n_head_fa, 1] → [head_dim_fa, 1, n_head_fa]
        // so it matches the FA expected layout.
        // ggml_turbo_wht's CUDA kernel writes dst using src strides
        // (turbo-wht.cu:20-21); non-contiguous input scatters writes and
        // corrupts Q. Always make Q contiguous BEFORE rotating.
        ggml_tensor * Qfa = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));
        if (k_is_tq3) {
            Qfa = ggml_turbo_wht(ctx, Qfa, 0);
        }
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_Qfa_%d", il);
            ggml_set_name(Qfa, name);
        }

        // K/V for FA: pass the original Kview/Vview (TQ3_0, Q8_0, or concat-F32)
        // directly to ggml_flash_attn_ext. Graph-level FWHT correction above/below
        // accounts for TQ3_0 K/V without stripping the tensor type tag.
        // For the wrap case (kv_wraps=true), Kview is already F32 (from to_f32 + concat).
        ggml_tensor * Kfa = Kview;  // original type (TQ3_0, Q8_0, or concat-F32)
        ggml_tensor * Vfa = Vview;
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_Kfa_%d", il);
            ggml_set_name(Kfa, name);
            std::snprintf(name, sizeof(name), "mtp_Vfa_%d", il);
            ggml_set_name(Vfa, name);
        }
        // For head_dim==512 (any K type): the MMA dispatcher requires
        // gqa_opt_applies, which requires BOTH K->ne[1] % 256 == 0 AND
        // mask != nullptr. Without mask, BEST_FATTN_KERNEL_NONE → abort
        // even when K is properly aligned. Always provide the mask.
        // We padded K/V to kv_view_len_padded above when needs_kv_pad is true;
        // when not padding, mask width == kv_view_len (all positions admitted).
        // The caller fills: positions [0..kv_seq_len-1] = 0.0 (admit),
        //                   positions [kv_seq_len..mask_width-1] = -inf (exclude padding).
        //
        // For head_dim==256 (SWA) with TQ3_0 K (non-wrap): VEC kernel handles it
        // without mask UNLESS needs_kv_pad triggers (KV unaligned); then mask is
        // needed to exclude the padding tail.
        // For wrap case (F32 K/V after concat): no TQ3_0 issues, no mask needed.
        const bool need_mask = head_dim_fa >= 512 || needs_kv_pad;
        const int64_t fa_mask_width = (needs_kv_pad ? kv_view_len_padded : kv_view_len);
        ggml_tensor * fa_mask = nullptr;
        if (need_mask) {
            if (shared_fa_mask == nullptr) {
                shared_fa_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, fa_mask_width, 1);
                ggml_set_name(shared_fa_mask, "mtp_fa_mask");
                ggml_set_input(shared_fa_mask);
                shared_fa_mask_width      = fa_mask_width;
                shared_fa_mask_kv_seq_len = kv_view_len;
            } else if (shared_fa_mask_width != fa_mask_width
                       || shared_fa_mask_kv_seq_len != kv_view_len) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "build_mtp_step_graph: per-layer FA masks diverge "
                    "(layer %d wants width=%lld kv_seq=%lld; existing %lld/%lld). "
                    "Long-context SWA cap mismatch — extend builder to per-layer masks.",
                    il, (long long)fa_mask_width, (long long)kv_view_len,
                    (long long)shared_fa_mask_width, (long long)shared_fa_mask_kv_seq_len);
                set_last_error(buf);
                ggml_free(ctx);
                return false;
            }
            fa_mask = shared_fa_mask;
        }

        // Gemma4 MTP: f_attention_scale = 1.0 (no pre-softmax scaling).
        ggml_tensor * attn_out = ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa, fa_mask,
                                                      1.0f, 0.0f, 0.0f);
        if (v_is_tq3) {
            attn_out = ggml_cont(ctx, attn_out);
            attn_out = ggml_turbo_wht(ctx, attn_out, 1);
        }
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_fa_out_%d", il);
            ggml_set_name(attn_out, name);
        }

        // FA output: [head_dim_fa, 1, n_head_fa]. Flatten to [q_out_dim, 1].
        // Flatten heads: [head_dim_fa, 1, n_head_fa] → [q_out_dim, 1].
        // ggml_flash_attn_ext returns a contiguous tensor by spec; only the
        // v_is_tq3 branch above (turbo_wht in-place) can break contiguity, so
        // restrict the cont kernel to that case.
        ggml_tensor * attn = (v_is_tq3 ? ggml_cont(ctx, attn_out) : attn_out);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_attn_out_%d", il);
            ggml_set_name(attn, name);
        }

        // Reshape: [q_out_dim, 1] then output projection
        // head_dim_fa * n_head_fa == q_out_dim == head_dim_norm * n_head_norm
        attn = ggml_reshape_2d(ctx, attn, q_out_dim, 1);
        cur = ggml_mul_mat(ctx, L.wo, attn);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_attn_proj_%d", il);
            ggml_set_name(cur, name);
        }

        // f) Post-attention norm
        cur = mtp_rms_norm_mul(ctx, cur, L.attn_post_norm);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_attn_post_norm_%d", il);
            ggml_set_name(cur, name);
        }

        // g) Attention residual
        ggml_tensor * attn_residual = ggml_add(ctx, cur, inpL);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_attn_residual_%d", il);
            ggml_set_name(attn_residual, name);
        }

        // h) FFN norm
        ggml_tensor * ffn_in = mtp_rms_norm_mul(ctx, attn_residual, L.ffn_norm);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_ffn_norm_%d", il);
            ggml_set_name(ffn_in, name);
        }

        // i) GELU FFN
        ggml_tensor * ffn_out = mtp_gelu_ffn(ctx, ffn_in, L);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_ffn_out_%d", il);
            ggml_set_name(ffn_out, name);
        }

        // j) Post-FFN norm
        ffn_out = mtp_rms_norm_mul(ctx, ffn_out, L.ffn_post_norm);
        {
            char name[64]; std::snprintf(name, sizeof(name), "mtp_ffn_post_norm_%d", il);
            ggml_set_name(ffn_out, name);
        }

        // k) FFN residual
        cur = ggml_add(ctx, ffn_out, attn_residual);

        // l) Optional per-layer output scale
        if (L.out_scale) {
            cur = ggml_mul(ctx, cur, L.out_scale);
            {
                char name[64]; std::snprintf(name, sizeof(name), "mtp_out_scaled_%d", il);
                ggml_set_name(cur, name);
            }
        }

        inpL = cur;
    }

    // ── 4. Output norm ────────────────────────────────────────────────────────
    ggml_tensor * h_inner = mtp_rms_norm_mul(ctx, inpL, w.output_norm);
    ggml_set_name(h_inner, "mtp_result_norm");

    // ── 5. Post-projection → h_post (next h_prev) ─────────────────────────────
    // post_projection: [n_embd, n_embd_backbone] (ggml ne[0]=n_embd, ne[1]=n_embd_backbone)
    ggml_tensor * h_post = ggml_mul_mat(ctx, w.post_projection, h_inner);
    ggml_set_name(h_post, "mtp_post_proj_out");

    // ── 6. LM head ────────────────────────────────────────────────────────────
    ggml_tensor * logits = nullptr;

    // Determine whether tok_embd supports ggml_get_rows on CUDA.
    // This backend (custom llama.cpp fork) only supports F32/F16/BF16/Q4_0/Q4_1/
    // Q5_0/Q5_1/Q8_0/TQ3_0 for get_rows; K-quant types (Q4_K, Q5_K, Q6_K) are not.
    // When tok_embd is a K-quant, the centroid sparse path can't use get_rows;
    // fall back to dense mul_mat for logit computation instead.
    const bool tok_embd_get_rows_ok =
        (w.tok_embd &&
         (w.tok_embd->type == GGML_TYPE_F32  ||
          w.tok_embd->type == GGML_TYPE_F16  ||
          w.tok_embd->type == GGML_TYPE_BF16 ||
          w.tok_embd->type == GGML_TYPE_Q4_0 ||
          w.tok_embd->type == GGML_TYPE_Q4_1 ||
          w.tok_embd->type == GGML_TYPE_Q5_0 ||
          w.tok_embd->type == GGML_TYPE_Q5_1 ||
          w.tok_embd->type == GGML_TYPE_Q8_0));

    if (w.use_ordered_embeddings && w.centroids && w.n_centroids > 0 && tok_embd_get_rows_ok) {
        // Centroid-routed LM head (matches atomicbot lines 190-235).
        // All mul_mat ops use h_inner [n_embd, 1] (MTP's own hidden space, n_embd=1024).
        // The embedding source is the MTP model's own tok_embd [n_embd, n_vocab] (w.tok_embd),
        // NOT the target's tok_embd (which is in backbone space and used only in step 1).
        if (!w.tok_embd) {
            set_last_error("build_mtp_step_graph: use_ordered_embeddings=true but w.tok_embd is null (token_embd.weight missing from GGUF)");
            ggml_free(ctx);
            return false;
        }

        const int64_t n_c   = (int64_t)w.n_centroids;
        const int64_t top_k = (int64_t)w.centroid_top_k;
        // Validate centroid-head shape and index invariants before any arithmetic.
        GGML_ASSERT(n_vocab > 0   && "centroid LM head: n_vocab must be > 0");
        GGML_ASSERT(n_c > 0       && "centroid LM head: n_centroids must be > 0");
        GGML_ASSERT(n_vocab % n_c == 0
                    && "centroid LM head: n_vocab must be divisible by n_centroids");
        GGML_ASSERT(top_k > 0 && top_k <= n_c
                    && "centroid LM head: top_k must be in [1, n_centroids]");
        // vsc: tokens per centroid slot
        const int64_t vsc   = (int64_t)n_vocab / n_c;

        // centroid_logits = mul_mat(centroids, h_inner) → [n_centroids, 1]
        // centroids: [n_embd, n_centroids] (ne[0]=n_embd, ne[1]=n_centroids)
        ggml_tensor * centroid_logits = ggml_mul_mat(ctx, w.centroids, h_inner);
        ggml_set_name(centroid_logits, "mtp_centroid_logits");

        // top-k centroid indices
        ggml_tensor * topk_idx = ggml_top_k(ctx, centroid_logits, (int)top_k);
        ggml_set_name(topk_idx, "mtp_centroid_topk_idx");

        // View token_ordering as [vsc, n_centroids] (I32)
        const size_t ordering_row_bytes = ggml_row_size(GGML_TYPE_I32, vsc);
        ggml_tensor * ordering = ggml_view_2d(ctx, w.token_ordering,
            vsc, n_c, ordering_row_bytes, /*offset=*/0);
        ggml_set_name(ordering, "mtp_token_ordering_view");

        // Gather candidate token ids for top-k centroids: [vsc, top_k, 1]
        ggml_tensor * sel_ids = ggml_get_rows(ctx, ordering, topk_idx);
        ggml_set_name(sel_ids, "mtp_selected_token_ids");

        // Flatten to 1D for embedding lookup
        const int64_t n_sel = top_k * vsc;
        ggml_tensor * flat_ids = ggml_reshape_1d(ctx, sel_ids, n_sel);
        ggml_set_name(flat_ids, "mtp_selected_token_ids_flat");

        // Gather embeddings for selected tokens from MTP's own tok_embd [n_embd, n_vocab].
        // get_rows selects n_sel rows → [n_embd, n_sel]
        ggml_tensor * sel_emb = ggml_get_rows(ctx, w.tok_embd, flat_ids);
        ggml_set_name(sel_emb, "mtp_selected_embd");

        // Sparse logits: mul_mat(sel_emb, h_inner):
        // sel_emb [n_embd, n_sel], h_inner [n_embd, 1] → [n_sel, 1]
        ggml_tensor * sel_logits = ggml_mul_mat(ctx, sel_emb, h_inner);
        ggml_set_name(sel_logits, "mtp_selected_logits");
        ggml_tensor * sel_logits_f32 = ggml_cast(ctx, sel_logits, GGML_TYPE_F32);
        ggml_set_name(sel_logits_f32, "mtp_selected_logits_f32");

        // Build full vocab row pre-filled with -1e30
        ggml_tensor * logits_full = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, 1);
        logits_full = ggml_fill_inplace(ctx, logits_full, -1e30f);
        ggml_set_name(logits_full, "mtp_logits_masked_base");

        // Scatter selected logits into full row
        ggml_tensor * scatter_dst = ggml_cont_2d(ctx, logits_full, 1, (int64_t)n_vocab);
        ggml_tensor * scatter_src = ggml_cont_2d(ctx, sel_logits_f32, 1, n_sel);
        logits = ggml_set_rows(ctx, scatter_dst, scatter_src, flat_ids);
        logits = ggml_reshape_2d(ctx, logits, n_vocab, 1);
        ggml_set_name(logits, "mtp_logits_full");
    } else if (w.use_ordered_embeddings && w.tok_embd) {
        // Dense fallback for ordered-embeddings models when tok_embd type does not
        // support CUDA get_rows (e.g. K-quants in this llama.cpp fork).
        // mul_mat supports Q4_K/Q5_K/Q6_K on CUDA; produces exact logits
        // (not the centroid approximation) which is fine for greedy/low-temp decoding.
        logits = ggml_mul_mat(ctx, w.tok_embd, h_inner);
        ggml_set_name(logits, "mtp_logits_dense_fallback");
    } else {
        // Dense tied LM head: mul_mat(tok_embd, h_post) → [n_vocab, 1]
        // For non-ordered-embeddings models (n_embd == n_embd_backbone), use h_post
        // (post-projected to n_embd_backbone) so dimensions match target.tok_embd.
        // Prefer w.tok_embd (MTP's own, in n_embd space) if available, else
        // fall back to target.tok_embd (in n_embd_backbone space) with h_post.
        if (w.tok_embd) {
            // MTP has its own tied LM head in n_embd space
            logits = ggml_mul_mat(ctx, w.tok_embd, h_inner);
        } else {
            // Fallback: use target's tok_embd against the backbone-projected hidden
            logits = ggml_mul_mat(ctx, target.tok_embd, h_post);
        }
        ggml_set_name(logits, "mtp_logits_dense");
    }

    // Optional logit softcapping (matches target's softcap=30)
    if (target.logit_softcap > 0.0f) {
        logits = ggml_scale(ctx, logits, 1.0f / target.logit_softcap);
        logits = ggml_tanh(ctx, logits);
        logits = ggml_scale(ctx, logits, target.logit_softcap);
        ggml_set_name(logits, "mtp_logits_softcapped");
    }

    // ── 7. In-graph argmax ─────────────────────────────────────────────────────
    ggml_tensor * argmax = ggml_argmax(ctx, logits);
    ggml_set_name(argmax, "mtp_argmax");

    // Expand all outputs into the graph
    ggml_build_forward_expand(gf, argmax);
    ggml_build_forward_expand(gf, h_post);
    // Note: logits is already in argmax's DAG, but mark it as output for diagnostic reads.
    ggml_set_output(logits);
    ggml_set_output(h_post);
    ggml_set_output(argmax);

    // ── Populate output struct ────────────────────────────────────────────────
    out.ctx          = ctx;
    out.gf           = gf;
    out.in_tok       = in_tok;
    out.in_tok_embd  = in_tok_embd;
    out.in_h_prev    = in_h_prev;
    out.in_pos       = in_pos;
    out.fa_mask             = shared_fa_mask;
    out.fa_mask_kv_seq_len  = shared_fa_mask_kv_seq_len;
    out.out_logits   = logits;
    out.out_h_post   = h_post;
    out.out_argmax   = argmax;

    return true;
}

void free_mtp_step_graph(MtpStepGraph & g) {
    if (g.ctx) {
        ggml_free(g.ctx);
        g.ctx = nullptr;
    }
    g.gf           = nullptr;
    g.in_tok       = nullptr;
    g.in_tok_embd  = nullptr;
    g.in_h_prev    = nullptr;
    g.in_pos       = nullptr;
    g.fa_mask             = nullptr;
    g.fa_mask_kv_seq_len  = 0;
    g.out_logits   = nullptr;
    g.out_h_post   = nullptr;
    g.out_argmax   = nullptr;
}

} // namespace dflash27b
