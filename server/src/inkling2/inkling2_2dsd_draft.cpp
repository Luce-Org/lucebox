// 2DSD Loop-Context 5-Layer Qwen3-DFlash + Vanilla DSpark Markov + Confidence Head Drafter.

#include "inkling2_2dsd_draft.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace luce::common {
namespace {

static inline void set_op_i32(ggml_tensor * t, int idx, int32_t v) {
    std::memcpy(&t->op_params[idx], &v, sizeof(v));
}

static ggml_tensor * row_view_2d(ggml_context * c, ggml_tensor * src, int64_t row_off, int64_t n_rows) {
    return ggml_view_2d(c, src, n_rows, src->ne[1], src->nb[1], (size_t) row_off * src->nb[0]);
}

static ggml_tensor * rms_norm_w(ggml_context * c, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
}

// Subop 37: batched vanilla DSpark Markov argmax across L_b leaves (V = 200058, no softcap).
static ggml_tensor * inkling2_2dsd_markov_step(
        ggml_context * c,
        ggml_tensor * base_logits, // F32 [V, L_b]
        ggml_tensor * prev_tok,    // I32 [L_b]
        ggml_tensor * markov_w1,   // F16 [R, V]
        ggml_tensor * markov_w2,   // Q8_0/F16 [R, V]
        int n_vocab) {
    ggml_tensor * res = ggml_new_tensor_1d(c, GGML_TYPE_I32, prev_tok->ne[0]);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = base_logits;
    res->src[1] = prev_tok;
    res->src[2] = markov_w1;
    res->src[3] = markov_w2;
    set_op_i32(res, 0, 37);
    set_op_i32(res, 1, n_vocab);
    return res;
}

// Subop 40: single-launch batched top-1024 candidate vanilla DSpark Markov Top-2 argmax across L_b leaves.
// Returns I32 [L_b, 2]: row 0 = best1 tokens, row 1 = best2 tokens (0 extra VRAM reads!).
static ggml_tensor * inkling2_2dsd_markov_topk_step(
        ggml_context * c,
        ggml_tensor * cand_ids,    // I32 [1024, L_b]
        ggml_tensor * base_logits, // F32 [V, L_b]
        ggml_tensor * prev_tok,    // I32 [L_b]
        ggml_tensor * markov_w1,   // F16 [R, V]
        ggml_tensor * markov_w2,   // Q8_0/F16 [R, V]
        int n_vocab) {
    ggml_tensor * res = ggml_new_tensor_2d(c, GGML_TYPE_I32, prev_tok->ne[0], 2);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = cand_ids;
    res->src[1] = base_logits;
    res->src[2] = prev_tok;
    res->src[3] = markov_w1;
    res->src[4] = markov_w2;
    set_op_i32(res, 0, 40);
    set_op_i32(res, 1, n_vocab);
    return res;
}

// Subop 38: vectorized confidence head over M = 7 * L_b slots.
static ggml_tensor * inkling2_2dsd_conf_head(
        ggml_context * c,
        ggml_tensor * h_spec,      // F32 [H, M]
        ggml_tensor * before_toks, // I32 [M]
        ggml_tensor * markov_w1,   // F16 [R, V]
        ggml_tensor * conf_w,      // F32 [H + R]
        ggml_tensor * conf_b) {    // F32 [1]
    ggml_tensor * res = ggml_new_tensor_1d(c, GGML_TYPE_F32, h_spec->ne[1]);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = h_spec;
    res->src[1] = before_toks;
    res->src[2] = markov_w1;
    res->src[3] = conf_w;
    res->src[4] = conf_b;
    set_op_i32(res, 0, 38);
    return res;
}

// Subop 39: on-device multi-leaf loop-context bidirectional F16 mask [kv_span, B * L_b].
static ggml_tensor * inkling2_2dsd_draft_mask(
        ggml_context * c,
        ggml_tensor * draft_meta, // I32 [3 + L_b]
        int64_t kv_span,
        int B,
        int L_b) {
    ggml_tensor * res = ggml_new_tensor_2d(c, GGML_TYPE_F16, kv_span, (int64_t) B * L_b);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = draft_meta;
    set_op_i32(res, 0, 39);
    set_op_i32(res, 1, B);
    return res;
}

static int bucket_leaf(int n) {
    if (n <= 1) return 1;
    if (n <= 2) return 2;
    if (n <= 3) return 3;
    if (n <= 4) return 4;
    return 8;
}

static int bucket_span(int m) {
    if (m <= 0) return 0;
    if (m <= 4) return 4;
    if (m <= 8) return 8;
    return 16;
}

// Project context hidden [H, T] (already after fc + hidden_norm) into layer i's K/V
// using the layer's shared wqkv (rows [q_sz .. q_sz + 2*kv_sz)), apply k_norm + RoPE,
// and write to draft_k/v[i] at `wpos`.
static void project_ctx_kv_into_draft_cache(
        ggml_context * c,
        ggml_cgraph * gf,
        const Inkling2TwoDSDWeights & td,
        const Inkling2TreeCache & tc,
        ggml_tensor * ctx_h,     // F32 [H, T]
        ggml_tensor * rope_pos,  // I32 [T]
        ggml_tensor * write_pos, // I32 [T]
        std::vector<ggml_tensor *> * out_k_upd = nullptr,
        std::vector<ggml_tensor *> * out_v_upd = nullptr) {
    const int q_sz  = td.n_head * td.head_dim;
    const int kv_sz = td.n_head_kv * td.head_dim;
    const int T     = (int) ctx_h->ne[1];
    if (out_k_upd) out_k_upd->resize(td.n_layer, nullptr);
    if (out_v_upd) out_v_upd->resize(td.n_layer, nullptr);

    for (int i = 0; i < td.n_layer; ++i) {
        const auto & L = td.layers[i];
        // Shared K/V rows in fused wqkv: offset q_sz, count 2*kv_sz
        ggml_tensor * wkv = ggml_view_2d(
            c, L.wqkv, L.wqkv->ne[0], 2 * kv_sz, L.wqkv->nb[1], (size_t) q_sz * L.wqkv->nb[1]);
        ggml_tensor * kv    = ggml_mul_mat(c, wkv, ctx_h); // F32 [2*kv_sz, T]
        ggml_tensor * k_raw = ggml_cont(c, row_view_2d(c, kv, 0, kv_sz));
        ggml_tensor * v_raw = ggml_cont(c, row_view_2d(c, kv, kv_sz, kv_sz));
        ggml_tensor * k3    = ggml_reshape_3d(c, k_raw, td.head_dim, td.n_head_kv, T);
        k3 = ggml_mul(c, ggml_rms_norm(c, k3, td.rms_eps), L.k_norm);
        k3 = ggml_rope_ext(c, k3, rope_pos, nullptr, td.head_dim, GGML_ROPE_TYPE_NEOX,
                           0, td.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * v3 = ggml_reshape_3d(c, v_raw, td.head_dim, td.n_head_kv, T);
        ggml_tensor * kp = ggml_permute(c, k3, 0, 2, 1, 3); // [D, T, Hk]
        ggml_tensor * vp = ggml_permute(c, v3, 0, 2, 1, 3);
        ggml_tensor * dk = ggml_set_rows(c, tc.draft_k[i], kp, write_pos);
        ggml_tensor * dv = ggml_set_rows(c, tc.draft_v[i], vp, write_pos);
        if (gf) {
            ggml_build_forward_expand(gf, dk);
            ggml_build_forward_expand(gf, dv);
        }
        if (out_k_upd) (*out_k_upd)[i] = dk;
        if (out_v_upd) (*out_v_upd)[i] = dv;
    }
}

} // namespace

void inkling2_2dsd_append_prefix_kv_graph(
        ggml_context * c,
        ggml_cgraph * gf,
        const Inkling2TwoDSDWeights & td,
        const Inkling2TreeCache & tc,
        ggml_tensor * feats_dev,
        ggml_tensor * rope_pos,
        ggml_tensor * write_pos) {
    if (!td.loaded || !tc.active || !feats_dev || !rope_pos || !write_pos) return;
    ggml_tensor * in_f32 = (feats_dev->type == GGML_TYPE_F32) ? feats_dev : ggml_cast(c, feats_dev, GGML_TYPE_F32);
    ggml_tensor * ctx_h  = rms_norm_w(c, ggml_mul_mat(c, td.fc, in_f32), td.hidden_norm, td.rms_eps);
    project_ctx_kv_into_draft_cache(c, gf, td, tc, ctx_h, rope_pos, write_pos);
}

bool inkling2_2dsd_commit_prefix_kv_from_feats(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        ggml_tensor * feats_dev,
        int pos0,
        int n_tokens) {
    if (!w.twodsd.loaded || !cache.tree.active || !feats_dev || n_tokens <= 0) return false;
    const auto & td = w.twodsd;
    const int F = 5 * td.n_embd;

    const size_t max_nodes = 512;
    struct ggml_init_params gp = { max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false) + (1 << 20), nullptr, true };
    ggml_context * sc = ggml_init(gp);
    ggml_cgraph * gf  = ggml_new_graph_custom(sc, max_nodes, false);

    ggml_tensor * t_in  = ggml_new_tensor_2d(sc, feats_dev->type, F, n_tokens);
    ggml_tensor * t_pos = ggml_new_tensor_1d(sc, GGML_TYPE_I32, n_tokens);
    ggml_set_input(t_in);
    ggml_set_input(t_pos);

    ggml_tensor * in_f32 = (t_in->type == GGML_TYPE_F32) ? t_in : ggml_cast(sc, t_in, GGML_TYPE_F32);
    ggml_tensor * ctx_h  = rms_norm_w(sc, ggml_mul_mat(sc, td.fc, in_f32), td.hidden_norm, td.rms_eps);
    project_ctx_kv_into_draft_cache(sc, gf, td, cache.tree, ctx_h, t_pos, t_pos);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
    if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) {
        if (ga) ggml_gallocr_free(ga);
        ggml_free(sc);
        return false;
    }

    ggml_backend_tensor_copy(feats_dev, t_in);
    std::vector<int32_t> pos(n_tokens);
    for (int i = 0; i < n_tokens; ++i) pos[i] = pos0 + i;
    ggml_backend_tensor_set(t_pos, pos.data(), 0, (size_t) n_tokens * sizeof(int32_t));

    const bool ok = (ggml_backend_graph_compute(target_backend, gf) == GGML_STATUS_SUCCESS);
    ggml_gallocr_free(ga);
    ggml_free(sc);
    if (ok) cache.dspark_ctx_pos = pos0 + n_tokens;
    return ok;
}

bool inkling2_2dsd_commit_prefix_kv_from_slots(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        const int * commit_slots,
        int n_commit,
        int pos0) {
    if (!w.twodsd.loaded || !cache.tree.active || !commit_slots || n_commit <= 0) return false;
    if (n_commit > kInkling2TwoDSDMaxCommit) {
        // Chunk large commits safely
        for (int off = 0; off < n_commit; off += kInkling2TwoDSDMaxCommit) {
            const int m = std::min(kInkling2TwoDSDMaxCommit, n_commit - off);
            if (!inkling2_2dsd_commit_prefix_kv_from_slots(target_backend, w, cache, commit_slots + off, m, pos0 + off)) {
                return false;
            }
        }
        return true;
    }

    const auto & td = w.twodsd;
    auto & tc = cache.tree;
    auto & g  = tc.prefix_store_graphs[n_commit];
    const int S = tc.max_nodes + 1;

    if (!g.built) {
        const size_t max_nodes = 512;
        struct ggml_init_params gp = { max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false) + (1 << 20), nullptr, true };
        g.ctx = ggml_init(gp);
        g.gf  = ggml_new_graph_custom(g.ctx, max_nodes, false);
        g.feats = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_commit); // flat index = 2*S + slot
        g.pos   = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_commit); // true absolute RoPE pos
        g.wpos  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_commit); // compact write slot
        ggml_set_input(g.feats);
        ggml_set_input(g.pos);
        ggml_set_input(g.wpos);

        ggml_tensor * gathered = ggml_get_rows(g.ctx, tc.node_features, g.feats); // F32 [5*H, n_commit]
        ggml_tensor * ctx_h    = rms_norm_w(g.ctx, ggml_mul_mat(g.ctx, td.fc, gathered), td.hidden_norm, td.rms_eps);
        project_ctx_kv_into_draft_cache(g.ctx, g.gf, td, tc, ctx_h, g.pos, g.wpos);

        g.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
        if (!g.alloc || !ggml_gallocr_alloc_graph(g.alloc, g.gf)) {
            if (g.alloc) { ggml_gallocr_free(g.alloc); g.alloc = nullptr; }
            ggml_free(g.ctx); g.ctx = nullptr;
            g.gf = nullptr;
            return false;
        }
        g.built = true;
    }

    // Depth-R (depth_idx = n_depths - 1 = 2) features sit at rows [2*S .. 3*S - 1].
    // Negative commit_slots[i] encodes a varloop early-exited d2 slot (depth_idx = 1, actual_slot = -(s+1)).
    const int depth_r_base  = (tc.n_depths - 1) * S;
    const int depth_d2_base = std::min(1, tc.n_depths - 1) * S;
    int32_t idx[kInkling2TwoDSDMaxCommit];
    int32_t pos[kInkling2TwoDSDMaxCommit];
    int32_t wpos[kInkling2TwoDSDMaxCommit];
    for (int i = 0; i < n_commit; ++i) {
        const int rel    = std::max(0, (pos0 - tc.draft_pref_start) + i);
        const int raw    = commit_slots[i];
        const bool from_d2 = (raw < 0);
        const int actual = from_d2 ? -(raw + 1) : raw;
        idx[i]  = (from_d2 ? depth_d2_base : depth_r_base) + actual;
        pos[i]  = pos0 + i;
        wpos[i] = (tc.draft_ring_cap > 0) ? (rel % tc.draft_ring_cap) : rel;
    }
    ggml_backend_tensor_set(g.feats, idx,  0, (size_t) n_commit * sizeof(int32_t));
    ggml_backend_tensor_set(g.pos,   pos,  0, (size_t) n_commit * sizeof(int32_t));
    ggml_backend_tensor_set(g.wpos,  wpos, 0, (size_t) n_commit * sizeof(int32_t));

    const bool ok = (ggml_backend_graph_compute_async(target_backend, g.gf) == GGML_STATUS_SUCCESS);
    if (ok) cache.dspark_ctx_pos = pos0 + n_commit;
    return ok;
}

bool inkling2_2dsd_draft_batch(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        const std::vector<Inkling2TwoDSDDraftReq> & requests,
        std::vector<Inkling2TwoDSDDraftOut> & out) {
    out.clear();
    if (!w.twodsd.loaded || !cache.tree.active || requests.empty()) return false;

    const auto & td = w.twodsd;
    auto & tc = cache.tree;
    const int B = td.block_size;       // 8
    const int n_spec = B - 1;          // 7
    const int n_req = (int) requests.size();

    // If more than 8 leaves were queued in one step, split cleanly into chunks of 8.
    if (n_req > 8) {
        out.resize(n_req);
        for (int off = 0; off < n_req; off += 8) {
            const int cnt = std::min(8, n_req - off);
            std::vector<Inkling2TwoDSDDraftReq> sub(requests.begin() + off, requests.begin() + off + cnt);
            std::vector<Inkling2TwoDSDDraftOut> sub_out;
            if (!inkling2_2dsd_draft_batch(target_backend, w, cache, sub, sub_out)) return false;
            for (int i = 0; i < cnt; ++i) out[off + i] = std::move(sub_out[i]);
        }
        return true;
    }

    int max_span = 0;
    for (const auto & r : requests) {
        max_span = std::max(max_span, (int) r.span_slots.size());
    }
    if (max_span > 16) {
        std::fprintf(stderr, "[inkling2-2dsd] span length %d exceeds max 16\n", max_span);
        return false;
    }

    const int L_b = bucket_leaf(n_req);
    const int S_b = bucket_span(max_span);
    const int pref_len    = tc.prefix_len;
    const int rel_pref    = std::max(0, tc.prefix_len - tc.draft_pref_start);
    const int pref_len_kv = (tc.draft_ring_cap > 0) ? std::min(rel_pref, tc.draft_ring_cap) : rel_pref;
    const int kv_used     = pref_len_kv + L_b * S_b + L_b * B;
    if (kv_used + 1 > tc.draft_cap) {
        std::fprintf(stderr, "[inkling2-2dsd] draft KV overflow (used=%d cap=%d)\n", kv_used, tc.draft_cap);
        return false;
    }
    const int64_t kv_span = std::min<int64_t>(tc.draft_cap, ((kv_used + 255) / 256) * 256);

    const auto key = std::make_tuple(L_b, S_b, kv_span);
    auto & g = tc.draft_graphs[key];

    if (!g.built) {
        const size_t max_nodes = 4096;
        struct ggml_init_params gp = { max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false) + (8 << 20), nullptr, true };
        g.ctx = ggml_init(gp);
        g.gf  = ggml_new_graph_custom(g.ctx, max_nodes, false);
        ggml_context * c = g.ctx;

        g.inp_heads      = ggml_new_tensor_1d(c, GGML_TYPE_I32, L_b);
        g.inp_block_pos  = ggml_new_tensor_1d(c, GGML_TYPE_I32, B * L_b);
        g.inp_block_wpos = ggml_new_tensor_1d(c, GGML_TYPE_I32, B * L_b);
        g.inp_draft_meta = ggml_new_tensor_1d(c, GGML_TYPE_I32, 3 + L_b);
        ggml_set_input(g.inp_heads);
        ggml_set_input(g.inp_block_pos);
        ggml_set_input(g.inp_block_wpos);
        ggml_set_input(g.inp_draft_meta);

        std::vector<ggml_tensor *> span_k_upd(td.n_layer, nullptr);
        std::vector<ggml_tensor *> span_v_upd(td.n_layer, nullptr);
        if (S_b > 0) {
            const int n_span_tot = S_b * L_b;
            g.inp_span_idx  = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_span_tot);
            g.inp_span_pos  = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_span_tot);
            g.inp_span_wpos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_span_tot);
            ggml_set_input(g.inp_span_idx);
            ggml_set_input(g.inp_span_pos);
            ggml_set_input(g.inp_span_wpos);

            ggml_tensor * span_f = ggml_get_rows(c, tc.node_features, g.inp_span_idx); // F32 [5*H, S_b*L_b]
            ggml_tensor * ctx_h  = rms_norm_w(c, ggml_mul_mat(c, td.fc, span_f), td.hidden_norm, td.rms_eps);
            project_ctx_kv_into_draft_cache(c, nullptr, td, tc, ctx_h, g.inp_span_pos, g.inp_span_wpos,
                                            &span_k_upd, &span_v_upd);
        }

        // Build [anchor_normed, 7 * mask_embed] per leaf -> [H, B * L_b] in ONE 3D concat!
        ggml_tensor * anch_all = rms_norm_w(c, ggml_get_rows(c, w.tok_embd, g.inp_heads), w.tok_norm, w.rms_eps); // [H, L_b]
        ggml_tensor * anch_3d  = ggml_reshape_3d(c, anch_all, td.n_embd, 1, L_b);
        ggml_tensor * masks_3d = ggml_repeat_4d(c, ggml_reshape_3d(c, td.mask_embed, td.n_embd, 1, 1),
                                                td.n_embd, B - 1, L_b, 1);
        ggml_tensor * x = ggml_reshape_2d(c, ggml_concat(c, anch_3d, masks_3d, 1), td.n_embd, B * L_b);
        // On-device multi-leaf loop-context mask [kv_span, B * L_b]
        ggml_tensor * attn_mask = inkling2_2dsd_draft_mask(c, g.inp_draft_meta, kv_span, B, L_b);

        const int q_sz  = td.n_head * td.head_dim;
        const int kv_sz = td.n_head_kv * td.head_dim;
        const int Q_tot = B * L_b;
        const float attn_scale = 1.0f / std::sqrt((float) td.head_dim);

        for (int i = 0; i < td.n_layer; ++i) {
            const auto & L = td.layers[i];
            ggml_tensor * xn  = rms_norm_w(c, x, L.inp_norm, td.rms_eps);
            ggml_tensor * qkv = ggml_mul_mat(c, L.wqkv, xn); // [q_sz + 2*kv_sz, Q_tot]
            ggml_tensor * q3  = ggml_reshape_3d(c, ggml_cont(c, row_view_2d(c, qkv, 0, q_sz)), td.head_dim, td.n_head, Q_tot);
            ggml_tensor * k3  = ggml_reshape_3d(c, ggml_cont(c, row_view_2d(c, qkv, q_sz, kv_sz)), td.head_dim, td.n_head_kv, Q_tot);
            ggml_tensor * v3  = ggml_reshape_3d(c, ggml_cont(c, row_view_2d(c, qkv, q_sz + kv_sz, kv_sz)), td.head_dim, td.n_head_kv, Q_tot);

            q3 = ggml_mul(c, ggml_rms_norm(c, q3, td.rms_eps), L.q_norm);
            k3 = ggml_mul(c, ggml_rms_norm(c, k3, td.rms_eps), L.k_norm);
            q3 = ggml_rope_ext(c, q3, g.inp_block_pos, nullptr, td.head_dim, GGML_ROPE_TYPE_NEOX,
                               0, td.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k3 = ggml_rope_ext(c, k3, g.inp_block_pos, nullptr, td.head_dim, GGML_ROPE_TYPE_NEOX,
                               0, td.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            ggml_tensor * kp = ggml_permute(c, k3, 0, 2, 1, 3);
            ggml_tensor * vp = ggml_permute(c, v3, 0, 2, 1, 3);
            ggml_tensor * k_base = span_k_upd[i] ? span_k_upd[i] : tc.draft_k[i];
            ggml_tensor * v_base = span_v_upd[i] ? span_v_upd[i] : tc.draft_v[i];
            ggml_tensor * k_upd = ggml_set_rows(c, k_base, kp, g.inp_block_wpos);
            ggml_tensor * v_upd = ggml_set_rows(c, v_base, vp, g.inp_block_wpos);

            ggml_tensor * k_view = ggml_view_3d(c, k_upd, td.head_dim, kv_span, td.n_head_kv, k_upd->nb[1], k_upd->nb[2], 0);
            ggml_tensor * v_view = ggml_view_3d(c, v_upd, td.head_dim, kv_span, td.n_head_kv, v_upd->nb[1], v_upd->nb[2], 0);
            ggml_tensor * qp     = ggml_permute(c, q3, 0, 2, 1, 3); // [D, Q_tot, Hq]
            ggml_tensor * attn   = ggml_flash_attn_ext(c, qp, k_view, v_view, attn_mask, attn_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
            attn = ggml_reshape_2d(c, attn, q_sz, Q_tot);

            x = ggml_add(c, x, ggml_mul_mat(c, L.wo, attn));
            ggml_tensor * mn  = rms_norm_w(c, x, L.post_norm, td.rms_eps);
            ggml_tensor * gu  = ggml_mul_mat(c, L.w_gate_up, mn);
            ggml_tensor * ffn = ggml_mul_mat(c, L.w_down, ggml_swiglu(c, gu));
            x = ggml_add(c, x, ffn);
        }

        // Final norm -> gather slots 1..7 for each leaf, ordered slot-major [H, L_b, 7]
        // in ONE single permute+cont kernel (replaces up to 55 sequential ggml_concat launches!).
        ggml_tensor * h_all = rms_norm_w(c, x, td.norm, td.rms_eps); // [H, B * L_b]
        ggml_tensor * h_3d  = ggml_reshape_3d(c, h_all, td.n_embd, B, L_b);
        ggml_tensor * h_sp  = ggml_view_3d(c, h_3d, td.n_embd, n_spec, L_b, h_3d->nb[1], h_3d->nb[2], h_3d->nb[1]);
        ggml_tensor * h_slot_major = ggml_reshape_2d(
            c, ggml_cont(c, ggml_permute(c, h_sp, 0, 2, 1, 3)), td.n_embd, L_b * n_spec); // [H, L_b * 7]

        // Base logits (NO softcap! Unpadded V = 200058)
        ggml_tensor * draft_lm = td.lm_head_q4
            ? td.lm_head_q4
            : ggml_view_2d(c, w.output, td.n_embd, td.vocab_size, w.output->nb[1], 0);
        ggml_tensor * base_logits = ggml_mul_mat(c, draft_lm, h_slot_major); // F32 [V, L_b * 7]

        // Unrolled 7-step batched vanilla DSpark Markov argmax chain across all L_b leaves!
        // Default fast path: 1-pass top-1024 candidate extraction (subop 18) + 1-launch top-1024
        // Markov scoring per step (subop 40, ~195x less weight bandwidth than full 200k scan).
        // Set LUCE_INKLING2_2DSD_FULL_MARKOV=1 to force exact full-200k scan (subop 37).
        const bool use_full_markov = (std::getenv("LUCE_INKLING2_2DSD_FULL_MARKOV") != nullptr &&
                                      std::atoi(std::getenv("LUCE_INKLING2_2DSD_FULL_MARKOV")) != 0);
        ggml_tensor * topk_cands = (!use_full_markov && td.vocab_size >= 1024)
            ? ggml_inkling2_dspark_topk_cands(c, base_logits, td.vocab_size) // I32 [1024, L_b * 7]
            : nullptr;

        ggml_tensor * prev = g.inp_heads; // I32 [L_b]
        ggml_tensor * before_all = nullptr;
        ggml_tensor * tokens_all = nullptr;
        ggml_tensor * alts_all   = nullptr;
        for (int s = 0; s < n_spec; ++s) {
            before_all = (s == 0) ? prev : ggml_concat(c, before_all, prev, 0); // I32 [L_b * 7]
            ggml_tensor * base_s = ggml_view_2d(
                c, base_logits, td.vocab_size, L_b, base_logits->nb[1],
                (size_t) (s * L_b) * base_logits->nb[1]);
            ggml_tensor * alt_s = nullptr;
            if (topk_cands) {
                ggml_tensor * cands_s = ggml_view_2d(
                    c, topk_cands, 1024, L_b, topk_cands->nb[1],
                    (size_t) (s * L_b) * topk_cands->nb[1]);
                ggml_tensor * top2 = inkling2_2dsd_markov_topk_step(c, cands_s, base_s, prev, td.markov_w1, td.markov_w2, td.vocab_size);
                prev  = ggml_view_1d(c, top2, L_b, 0);
                alt_s = ggml_view_1d(c, top2, L_b, top2->nb[1]);
            } else {
                prev  = inkling2_2dsd_markov_step(c, base_s, prev, td.markov_w1, td.markov_w2, td.vocab_size);
                alt_s = prev;
            }
            tokens_all = (s == 0) ? prev  : ggml_concat(c, tokens_all, prev,  0); // I32 [L_b * 7]
            alts_all   = (s == 0) ? alt_s : ggml_concat(c, alts_all,   alt_s, 0); // I32 [L_b * 7]
        }

        // Vectorized confidence head over all 7 * L_b slots in ONE kernel!
        g.out_tokens     = tokens_all;
        g.out_alt_tokens = alts_all;
        g.out_conf       = inkling2_2dsd_conf_head(c, h_slot_major, before_all, td.markov_w1, td.conf_w, td.conf_b);
        ggml_set_output(g.out_tokens);
        ggml_set_output(g.out_alt_tokens);
        ggml_set_output(g.out_conf);
        ggml_build_forward_expand(g.gf, g.out_tokens);
        ggml_build_forward_expand(g.gf, g.out_alt_tokens);
        ggml_build_forward_expand(g.gf, g.out_conf);

        g.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
        if (!g.alloc || !ggml_gallocr_alloc_graph(g.alloc, g.gf)) {
            std::fprintf(stderr, "[inkling2-2dsd] draft graph alloc failed (L=%d S=%d kv=%lld)\n",
                         L_b, S_b, (long long) kv_span);
            if (g.alloc) { ggml_gallocr_free(g.alloc); g.alloc = nullptr; }
            ggml_free(g.ctx); g.ctx = nullptr;
            g.gf = nullptr;
            return false;
        }
        g.built = true;
    }

    // Pack tiny runtime inputs
    const int S_pool = tc.max_nodes + 1;
    const int scratch_kv = tc.draft_cap - 1;
    std::vector<int32_t> heads(L_b, requests[0].head_token);
    std::vector<int32_t> blk_pos(B * L_b, 0);
    std::vector<int32_t> blk_wpos(B * L_b, scratch_kv);
    std::vector<int32_t> dmeta(3 + L_b, 0);
    dmeta[0] = pref_len_kv;
    dmeta[1] = L_b;
    dmeta[2] = S_b;

    std::vector<int32_t> span_idx, span_pos, span_wpos;
    if (S_b > 0) {
        span_idx.assign((size_t) S_b * L_b, tc.max_nodes); // scratch slot in depth 0
        span_pos.assign((size_t) S_b * L_b, 0);
        span_wpos.assign((size_t) S_b * L_b, scratch_kv);
    }

    for (int l = 0; l < L_b; ++l) {
        const auto & req = (l < n_req) ? requests[l] : requests[0];
        const int slen   = (l < n_req) ? (int) req.span_slots.size() : 0;
        heads[l]     = req.head_token;
        dmeta[3 + l] = slen;

        const int depth_idx = std::clamp(req.depth - 1, 0, tc.n_depths - 1);
        for (int j = 0; j < slen; ++j) {
            const int flat = l * S_b + j;
            span_idx[flat]  = depth_idx * S_pool + req.span_slots[j];
            span_pos[flat]  = pref_len + j;                // true absolute RoPE pos
            span_wpos[flat] = pref_len_kv + l * S_b + j;   // compact KV write slot
        }
        for (int j = 0; j < B; ++j) {
            const int flat = l * B + j;
            blk_pos[flat]  = req.anchor_pos + j;
            blk_wpos[flat] = pref_len_kv + L_b * S_b + l * B + j;
        }
    }

    ggml_backend_tensor_set(g.inp_heads,      heads.data(),    0, heads.size()    * sizeof(int32_t));
    ggml_backend_tensor_set(g.inp_block_pos,  blk_pos.data(),  0, blk_pos.size()  * sizeof(int32_t));
    ggml_backend_tensor_set(g.inp_block_wpos, blk_wpos.data(), 0, blk_wpos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(g.inp_draft_meta, dmeta.data(),    0, dmeta.size()    * sizeof(int32_t));
    if (S_b > 0) {
        ggml_backend_tensor_set(g.inp_span_idx,  span_idx.data(),  0, span_idx.size()  * sizeof(int32_t));
        ggml_backend_tensor_set(g.inp_span_pos,  span_pos.data(),  0, span_pos.size()  * sizeof(int32_t));
        ggml_backend_tensor_set(g.inp_span_wpos, span_wpos.data(), 0, span_wpos.size() * sizeof(int32_t));
    }

    if (ggml_backend_graph_compute(target_backend, g.gf) != GGML_STATUS_SUCCESS) {
        return false;
    }

    std::vector<int32_t> all_toks((size_t) n_spec * L_b);
    std::vector<int32_t> all_alts((size_t) n_spec * L_b);
    std::vector<float>   all_conf((size_t) n_spec * L_b);
    ggml_backend_tensor_get(g.out_tokens,     all_toks.data(), 0, all_toks.size() * sizeof(int32_t));
    ggml_backend_tensor_get(g.out_alt_tokens, all_alts.data(), 0, all_alts.size() * sizeof(int32_t));
    ggml_backend_tensor_get(g.out_conf,       all_conf.data(), 0, all_conf.size() * sizeof(float));

    out.resize(n_req);
    for (int l = 0; l < n_req; ++l) {
        out[l].tokens.resize(n_spec);
        out[l].alt_tokens.resize(n_spec);
        out[l].confidence.resize(n_spec);
        for (int s = 0; s < n_spec; ++s) {
            out[l].tokens[s]     = all_toks[(size_t) s * L_b + l];
            out[l].alt_tokens[s] = all_alts[(size_t) s * L_b + l];
            out[l].confidence[s] = all_conf[(size_t) s * L_b + l];
        }
    }
    return true;
}

} // namespace luce::common
