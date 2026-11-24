// Mixed-depth static-bucketed 2DSD target runner (Prelude / Fused Multi-Pass Core /
// Fused Multi-Depth Coda / Commit) and 60-plane multi-depth Prefill for Inkling 2 Mini.

#include "inkling2_2dsd_graph.h"
#include "inkling2_2dsd_draft.h"
#include "common/cuda_graph_overrides.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace luce::common {
namespace {

static constexpr size_t kInkling2TwoDSDPrefetchBytes = (size_t) 56 << 20;

static inline void set_op_i32(ggml_tensor * t, int idx, int32_t v) {
    std::memcpy(&t->op_params[idx], &v, sizeof(v));
}
static inline void set_op_f32(ggml_tensor * t, int idx, float v) {
    std::memcpy(&t->op_params[idx], &v, sizeof(v));
}

static ggml_tensor * row_view_2d(ggml_context * c, ggml_tensor * src, int64_t row_off, int64_t n_rows) {
    return ggml_view_2d(c, src, n_rows, src->ne[1], src->nb[1], (size_t) row_off * src->nb[0]);
}

static ggml_tensor * rms_norm_w(ggml_context * c, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
}

// Reused Inkling2 fused op builders (matching inkling2_graph.cpp)
static ggml_tensor * make_clip_qr_rel_table(
        ggml_context * c,
        ggml_tensor * q_raw, ggml_tensor * q_clip_w,
        ggml_tensor * r_raw, ggml_tensor * r_clip_w,
        ggml_tensor * pos_buf, ggml_tensor * rel_bias, ggml_tensor * sink,
        int head_dim, int d_rel, int n_head,
        float q_thresh, float r_thresh, float eps, int log_floor, float log_alpha,
        ggml_tensor ** out_q, ggml_tensor ** out_rt) {
    const int64_t rel_extent = rel_bias->ne[1];
    const int64_t T = q_raw->ne[1];
    const int64_t q_elems  = (int64_t) head_dim * n_head * T;
    const int64_t rt_elems = (rel_extent + 1) * n_head * T;
    ggml_tensor * buf = ggml_new_tensor_1d(c, GGML_TYPE_F32, q_elems + rt_elems);
    buf->op = GGML_OP_INKLING2_OP;
    buf->src[0] = q_raw;
    buf->src[1] = q_clip_w;
    buf->src[2] = r_raw;
    buf->src[3] = r_clip_w;
    buf->src[4] = pos_buf;
    buf->src[5] = rel_bias;
    buf->src[6] = sink;
    set_op_i32(buf, 0, 31);
    set_op_i32(buf, 1, head_dim);
    set_op_i32(buf, 2, d_rel);
    set_op_i32(buf, 3, n_head);
    set_op_f32(buf, 4, q_thresh);
    set_op_f32(buf, 5, r_thresh);
    set_op_f32(buf, 6, eps);
    set_op_i32(buf, 7, log_floor);
    set_op_f32(buf, 8, log_alpha);

    *out_q = ggml_view_3d(c, buf, head_dim, n_head, T,
                          (size_t) head_dim * sizeof(float),
                          (size_t) head_dim * n_head * sizeof(float), 0);
    *out_rt = ggml_view_3d(c, buf, rel_extent + 1, n_head, T,
                           (size_t) (rel_extent + 1) * sizeof(float),
                           (size_t) (rel_extent + 1) * n_head * sizeof(float),
                           (size_t) q_elems * sizeof(float));
    return buf;
}

static ggml_tensor * make_fused_router_q8_topk(
        ggml_context * c, ggml_tensor * w_q8, ggml_tensor * cur, ggml_tensor * bias,
        ggml_tensor * flags, int top_k, float route_scale, float eps, int target_pct) {
    const int64_t T = cur->ne[1];
    ggml_tensor * logits = ggml_mul_mat(c, w_q8, cur);
    ggml_tensor * res = ggml_new_tensor_3d(c, GGML_TYPE_I32, top_k, T, target_pct > 0 ? 3 : 2);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = logits;
    res->src[1] = nullptr;
    res->src[2] = bias;
    res->src[3] = flags;
    set_op_i32(res, 0, 24);
    set_op_i32(res, 1, top_k);
    set_op_f32(res, 2, route_scale);
    set_op_f32(res, 3, eps);
    set_op_i32(res, 4, target_pct);
    return res;
}

static ggml_tensor * make_loop_ctrl_post_flagged(
        ggml_context * c,
        ggml_tensor * cur, ggml_tensor * msg, ggml_tensor * gf_out,
        ggml_tensor * norm_w, ggml_tensor * flags, float eps) {
    ggml_tensor * res = ggml_inkling2_loop_ctrl_post(c, cur, msg, gf_out, norm_w, eps);
    res->src[4] = flags;
    return res;
}

static ggml_tensor * make_tree_sconv(
        ggml_context * c,
        ggml_tensor * x, ggml_tensor * w_static,
        ggml_tensor * prefix_conv, ggml_tensor * tree_conv,
        ggml_tensor * meta, ggml_tensor * dyn_taps, ggml_tensor * addend,
        int chan_offset, int head_dim_dyn, int m_stride) {
    ggml_tensor * res = ggml_new_tensor_2d(c, GGML_TYPE_F32, x->ne[0], x->ne[1]);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = x;
    res->src[1] = w_static;
    res->src[2] = prefix_conv;
    res->src[3] = tree_conv;
    res->src[4] = meta;
    res->src[5] = dyn_taps;
    res->src[6] = addend;
    set_op_i32(res, 0, 33);
    set_op_i32(res, 1, chan_offset);
    set_op_i32(res, 2, head_dim_dyn);
    set_op_i32(res, 3, m_stride);
    return res;
}

static ggml_tensor * make_tree_sconv_kv_write(
        ggml_context * c,
        ggml_tensor * x, ggml_tensor * w_static,
        ggml_tensor * prefix_conv, ggml_tensor * tree_conv,
        ggml_tensor * meta, ggml_tensor * dyn_taps,
        ggml_tensor * tree_kv, ggml_tensor * prev_dep,
        int chan_offset, int head_dim, int n_head_kv, int is_v,
        int m_stride, float threshold, float eps) {
    ggml_tensor * res = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = x;
    res->src[1] = w_static;
    res->src[2] = prefix_conv;
    res->src[3] = tree_conv;
    res->src[4] = meta;
    res->src[5] = dyn_taps;
    res->src[6] = tree_kv;
    res->src[7] = prev_dep;
    set_op_i32(res, 0, 34);
    set_op_i32(res, 1, chan_offset);
    set_op_i32(res, 2, head_dim);
    set_op_i32(res, 3, n_head_kv);
    set_op_i32(res, 4, is_v);
    set_op_i32(res, 5, m_stride);
    set_op_f32(res, 6, threshold);
    set_op_f32(res, 7, eps);
    return res;
}

static ggml_tensor * make_tree_attn_rel(
        ggml_context * c,
        ggml_tensor * q,
        ggml_tensor * prefix_k, ggml_tensor * prefix_v,
        ggml_tensor * tree_k,   ggml_tensor * tree_v,
        ggml_tensor * rel_table, ggml_tensor * meta, ggml_tensor * prefix_len_buf,
        ggml_tensor * gate, ggml_tensor * dep_kv,
        int swa_window, int rel_extent, int m_stride, int path_w,
        int block_size = 0, int partial_out = 0, int exclude_sink = 0,
        int shared_kv_window = 0, int shared_tail_cap = 0, int planes_per_seq = 0) {
    const int64_t Hq = q->ne[1];
    const int64_t T  = q->ne[2];
    const int64_t out_cols = partial_out ? ((q->ne[0] + 2) * Hq) : (q->ne[0] * Hq);
    ggml_tensor * res = ggml_new_tensor_2d(c, GGML_TYPE_F32, out_cols, T);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = q;
    res->src[1] = prefix_k;
    res->src[2] = prefix_v;
    res->src[3] = tree_k;
    res->src[4] = tree_v;
    res->src[5] = rel_table;
    res->src[6] = meta;
    res->src[7] = prefix_len_buf;
    res->src[8] = gate;
    res->src[9] = dep_kv;
    set_op_i32(res, 0, 35);
    set_op_i32(res, 1, swa_window);
    set_op_i32(res, 2, rel_extent);
    set_op_i32(res, 3, m_stride);
    set_op_i32(res, 4, path_w);
    set_op_i32(res, 5, block_size);
    set_op_i32(res, 6, partial_out);
    set_op_i32(res, 7, exclude_sink);
    set_op_i32(res, 8, shared_kv_window);
    set_op_i32(res, 9, shared_tail_cap);
    set_op_i32(res, 10, planes_per_seq);
    return res;
}

static ggml_tensor * make_tree_prefill_clone_coda(
        ggml_context * c,
        ggml_tensor * prefix_k, ggml_tensor * prefix_v, ggml_tensor * prefix_conv,
        ggml_tensor * dep, int pos0, int n_tokens, int swa_window,
        ggml_tensor * block_table = nullptr, int block_size = 0,
        int shared_kv_window = 0, int shared_tail_cap = 0) {
    ggml_tensor * res = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = prefix_k;
    res->src[1] = prefix_v;
    res->src[2] = prefix_conv;
    res->src[3] = dep;
    res->src[4] = block_table;
    set_op_i32(res, 0, 43);
    set_op_i32(res, 1, pos0);
    set_op_i32(res, 2, n_tokens);
    set_op_i32(res, 3, swa_window);
    set_op_i32(res, 4, block_size);
    set_op_i32(res, 5, shared_kv_window);
    set_op_i32(res, 6, shared_tail_cap);
    return res;
}

static bool prefill_share_coda_enabled() {
    const char * e = std::getenv("LUCE_INKLING2_2DSD_PREFILL_SHARE_CODA");
    return (e == nullptr || std::atoi(e) != 0);
}

static int dual_2dsd_min_rows() {
    if (const char * e = std::getenv("LUCE_INKLING2_2DSD_DUAL_MIN_ROWS")) {
        const int v = std::atoi(e);
        if (v >= 1 && v <= 64) return v;
    }
    return 6;
}

static int dual_2dsd_phases_mask() {
    if (const char * e = std::getenv("LUCE_INKLING2_2DSD_DUAL_PHASES")) {
        return std::atoi(e);
    }
    return 7; // bit 0 = prelude, bit 1 = core, bit 2 = coda
}

static int dual_2dsd_split_pct(const Inkling2Weights & w) {
    if (const char * e = std::getenv("LUCE_INKLING2_2DSD_SPLIT_PCT")) {
        const int v = std::atoi(e);
        if (v >= 50 && v <= 99) return v;
    }
    return w.split_pct;
}

static bool dual_2dsd_f16_transport() {
    const char * e = std::getenv("LUCE_INKLING2_2DSD_DUAL_F16");
    return (e != nullptr && std::atoi(e) != 0);
}

static int dual_2dsd_lm_split_max() {
    if (const char * e = std::getenv("LUCE_INKLING2_2DSD_LM_SPLIT_MAX")) {
        const int v = std::atoi(e);
        if (v >= 0 && v <= 64) return v;
    }
    // Keep 2DSD Coda's Q8_0 LM head on target_backend (R9700): even at row_b=4..8,
    // copying 0.6-1.1 MB of F32 tail logits over PCIe + ggml_concat costs +0.30-0.37 ms
    // vs running w.output directly on R9700, while Coda's 6 MoE layers still split cleanly.
    return 0;
}

static bool dual_2dsd_batch_copies() {
    const char * e = std::getenv("LUCE_INKLING2_2DSD_BATCH_COPIES");
    return (e != nullptr && std::atoi(e) != 0);
}

static bool twodsd_profile_enabled() {
    static const bool on = (std::getenv("LUCE_INKLING2_2DSD_PROFILE") != nullptr &&
                            std::atoi(std::getenv("LUCE_INKLING2_2DSD_PROFILE")) != 0);
    return on;
}

static bool batch_invariant_2dsd_enabled() {
    const char * e = std::getenv("LUCE_BATCH_INVARIANT");
    return (e != nullptr && std::atoi(e) != 0);
}

static ggml_tensor * make_exact_gpu_top_cands(ggml_context * c, ggml_tensor * logits, float temp) {
    ggml_tensor * res = ggml_new_tensor_2d(c, GGML_TYPE_I32, 516, logits->ne[1]);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = logits;
    set_op_i32(res, 0, 32);
    set_op_f32(res, 1, temp);
    return res;
}

static ggml_tensor * make_tree_commit(
        ggml_context * c,
        ggml_tensor * prefix_k, ggml_tensor * prefix_v,
        ggml_tensor * tree_k,   ggml_tensor * tree_v,
        ggml_tensor * prefix_conv, ggml_tensor * tree_conv,
        ggml_tensor * commit_slots, ggml_tensor * commit_meta,
        int swa_window,
        ggml_tensor * block_table = nullptr, int block_size = 0,
        int shared_kv_window = 0, int shared_tail_cap = 0) {
    ggml_tensor * res = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = prefix_k;
    res->src[1] = prefix_v;
    res->src[2] = tree_k;
    res->src[3] = tree_v;
    res->src[4] = prefix_conv;
    res->src[5] = tree_conv;
    res->src[6] = commit_slots;
    res->src[7] = commit_meta;
    res->src[8] = block_table;
    set_op_i32(res, 0, 36);
    set_op_i32(res, 1, swa_window);
    set_op_i32(res, 2, block_size);
    set_op_i32(res, 3, shared_kv_window);
    set_op_i32(res, 4, shared_tail_cap);
    return res;
}

static ggml_tensor * make_fused_loop_ctrl_pre_down(
        ggml_context * c, ggml_tensor * cur, ggml_tensor * msg, ggml_tensor * w_q8, float eps) {
    ggml_tensor * res = (msg->ne[2] > 1)
        ? ggml_new_tensor_3d(c, GGML_TYPE_F32, w_q8->ne[1], msg->ne[1], msg->ne[2])
        : ggml_new_tensor_2d(c, GGML_TYPE_F32, w_q8->ne[1], msg->ne[1]);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = cur;
    res->src[1] = msg;
    res->src[2] = w_q8;
    set_op_i32(res, 0, 25);
    set_op_f32(res, 1, eps);
    return res;
}

static ggml_tensor * make_tree_sconv_kv_write_kv(
        ggml_context * c,
        ggml_tensor * x_k, ggml_tensor * x_v,
        ggml_tensor * wk_static, ggml_tensor * wv_static,
        ggml_tensor * prefix_conv, ggml_tensor * tree_conv,
        ggml_tensor * meta,
        ggml_tensor * tree_k, ggml_tensor * tree_v,
        int head_dim, int n_head_kv, int m_stride, float threshold, float eps) {
    ggml_tensor * res = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = x_k;
    res->src[1] = x_v;
    res->src[2] = wk_static;
    res->src[3] = wv_static;
    res->src[4] = prefix_conv;
    res->src[5] = tree_conv;
    res->src[6] = meta;
    res->src[7] = tree_k;
    res->src[8] = tree_v;
    set_op_i32(res, 0, 41);
    set_op_i32(res, 1, head_dim);
    set_op_i32(res, 2, n_head_kv);
    set_op_i32(res, 3, m_stride);
    set_op_f32(res, 4, threshold);
    set_op_f32(res, 5, eps);
    return res;
}

static ggml_tensor * make_moe_masked_sum_tree_sconv_add(
        ggml_context * c,
        ggml_tensor * down, ggml_tensor * wsel, ggml_tensor * mask_sel, ggml_tensor * peer_sum,
        ggml_tensor * w_static, ggml_tensor * prefix_conv, ggml_tensor * tree_conv,
        ggml_tensor * meta, ggml_tensor * addend,
        int chan_offset, int m_stride) {
    ggml_tensor * res = ggml_new_tensor_2d(c, GGML_TYPE_F32, down->ne[0], down->ne[2]);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = down;
    res->src[1] = wsel;
    res->src[2] = mask_sel;
    res->src[3] = peer_sum;
    res->src[4] = w_static;
    res->src[5] = prefix_conv;
    res->src[6] = tree_conv;
    res->src[7] = meta;
    res->src[8] = addend;
    set_op_i32(res, 0, 42);
    set_op_i32(res, 1, chan_offset);
    set_op_i32(res, 2, m_stride);
    return res;
}

// Prefetch next physical layer's attention & router weights into R9700's 64 MB Infinity Cache
// while waiting for Strix Halo's cold expert picks in Dual-GPU split mode.
static void prefetch_next_tree_layer(
        ggml_context * c, ggml_cgraph * gf,
        ggml_tensor * after, int ip, int phase_id, bool has_later,
        const Inkling2Weights & w) {
    if (!w.buf) return;
    std::vector<ggml_tensor *> next;
    int next_ip = ip + 1;
    if (phase_id == 0 && next_ip >= w.core_start) next_ip = w.core_start;
    if (phase_id == 1 && next_ip >= w.core_end)   next_ip = w.core_end;
    if (next_ip < w.n_layer_phys) {
        const auto & NL = w.layers[next_ip];
        const bool next_same_qkv = NL.is_swa && !(phase_id == 1 && next_ip < w.core_end && has_later);
        if (next_same_qkv && NL.wqkvgrs) {
            next = { NL.wqkvgrs, NL.wo, NL.ffn_gate_inp_q8 ? NL.ffn_gate_inp_q8 : NL.ffn_gate_inp };
        } else if (NL.wqgrs) {
            next = { NL.wqgrs, NL.wk_dyn ? NL.wk_dyn : NL.wk, NL.wv_dyn ? NL.wv_dyn : NL.wv, NL.wo };
        } else {
            next = { NL.wq, NL.wk, NL.wv, NL.wqkv_gate, NL.wo };
        }
    } else {
        next = { w.output };
    }
    ggml_tensor * srcs[GGML_MAX_SRC - 1];
    int n = 0;
    size_t bytes = 0;
    ggml_backend_buffer_type_t tbuft = ggml_backend_buffer_get_type(w.buf);
    ggml_backend_buffer_type_t dbuft = w.derived_buf ? ggml_backend_buffer_get_type(w.derived_buf) : nullptr;
    for (ggml_tensor * t : next) {
        if (!t || !t->buffer || !ggml_is_contiguous(t)) continue;
        ggml_backend_buffer_type_t b = ggml_backend_buffer_get_type(t->buffer);
        if (b != tbuft && b != dbuft) continue;
        if (n == GGML_MAX_SRC - 1 || bytes >= kInkling2TwoDSDPrefetchBytes) break;
        srcs[n++] = t;
        bytes += ggml_nbytes(t);
    }
    if (n > 0) {
        ggml_build_forward_expand(gf, ggml_moe_fused_prefetch(c, after, srcs, n, kInkling2TwoDSDPrefetchBytes));
    }
}

static int bucket_rows(int n) {
    static constexpr int kBuckets[] = { 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 64 };
    for (int b : kBuckets) if (n <= b) return b;
    return ((n + 31) / 32) * 32;
}

static int bucket_path(int w) {
    static constexpr int kBuckets[] = { 8, 16, 32, 64 };
    for (int b : kBuckets) if (w <= b) return b;
    return ((w + 31) / 32) * 32;
}

// Create a zero-copy 3D/2D view of a single plane inside a 4D/3D tree.prefix_* tensor (for prefill).
static ggml_tensor * plane_view_kv(ggml_context * c, ggml_tensor * kv4d, int plane) {
    return ggml_view_3d(c, kv4d, kv4d->ne[0], kv4d->ne[1], kv4d->ne[2],
                        kv4d->nb[1], kv4d->nb[2], (size_t) plane * kv4d->nb[3]);
}
static ggml_tensor * plane_view_conv(ggml_context * c, ggml_tensor * conv3d, int plane) {
    return ggml_view_2d(c, conv3d, conv3d->ne[0], conv3d->ne[1],
                        conv3d->nb[1], (size_t) plane * conv3d->nb[2]);
}

// Build one physical layer inside a 2DSD phase graph (Prelude / Core / Coda).
static ggml_tensor * build_tree_layer(
        ggml_context * c,
        ggml_cgraph * gf,
        const Inkling2Weights & w,
        Inkling2TreeCache & tc,
        Inkling2TwoDSDPhaseGraph & g,
        int ip,
        int phase_id,
        ggml_tensor * cur,
        std::vector<ggml_tensor *> & blocks,
        const std::vector<ggml_tensor *> * bank_blocks,
        ggml_tensor * bank_out,
        bool has_later,
        bool use_split,
        int row_b,
        int path_b,
        int kf_block_size,
        bool use_tier15) {
    const auto & L = w.layers[ip];
    const int Hq = w.n_head, Hk = w.n_head_kv, D = w.head_dim;
    const int q_sz = Hq * D, kv_sz = Hk * D, r_sz = w.d_rel * Hq;
    const int m_stride = 11 + path_b;
    const int swa = L.is_swa ? w.swa_window : 0;

    ggml_tensor * inpSA = cur;

    // Cross-pass messages for core layers when any row has pass > 0
    ggml_tensor * messages = nullptr;
    const Inkling2LoopLayer * loop = nullptr;
    if (has_later && bank_out != nullptr) {
        loop = &w.loop_layers[ip - w.core_start];
        messages = ggml_inkling2_attn_res_mix(
            c, bank_out, loop->msg_query, nullptr,
            (bank_blocks && !bank_blocks->empty()) ? bank_blocks->data() : nullptr,
            bank_blocks ? (int) bank_blocks->size() : 0,
            w.rms_eps); // [H, 4, T]
    }
    auto get_msg = [&](int ctrl) -> ggml_tensor * {
        return ggml_view_2d(c, messages, w.n_embd, row_b,
                            messages->nb[2], (size_t) ctrl * messages->nb[1]);
    };

    // Attention input mix + optional branchless per-row loop controller (FLAG_LATER)
    ggml_tensor * inputs[3] = { cur, cur, cur };
    const bool same_qkv_input = L.is_swa && (messages == nullptr);

    if (messages == nullptr) {
        if (L.attn_res_query == nullptr) {
            inputs[0] = rms_norm_w(c, cur, L.attn_norm, w.rms_eps);
            inputs[1] = inputs[2] = inputs[0];
        } else if (L.is_swa) {
            inputs[0] = ggml_inkling2_attn_res_mix(
                c, cur, L.attn_res_query, L.attn_norm,
                blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
            inputs[1] = inputs[2] = inputs[0];
        } else {
            ggml_tensor * mixed3 = ggml_inkling2_attn_res_mix(
                c, cur, L.attn_res_query, L.attn_qkv_norm,
                blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
            for (int j = 0; j < 3; ++j) {
                inputs[j] = ggml_view_2d(c, mixed3, w.n_embd, row_b,
                                         mixed3->nb[2], (size_t) j * mixed3->nb[1]);
            }
        }
    } else {
        ggml_tensor * pre_qkv = cur;
        if (L.attn_res_query) {
            ggml_tensor * mixed = ggml_inkling2_attn_res_mix(
                c, cur, L.attn_res_query, nullptr,
                blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
            pre_qkv = L.is_swa ? mixed : ggml_permute(c, mixed, 0, 2, 1, 3);
        }
        ggml_tensor * msg_qkv_v = ggml_view_3d(c, messages, w.n_embd, 3, row_b, messages->nb[1], messages->nb[2], 0);
        ggml_tensor * msg_qkv   = ggml_permute(c, msg_qkv_v, 0, 2, 1, 3); // [H, T, 3]
        ggml_tensor * nw_qkv    = L.is_swa ? L.attn_norm : L.attn_qkv_norm;
        ggml_tensor * gate      = (loop->ctrl_down_qkv->type == GGML_TYPE_Q8_0 && row_b <= 32)
            ? make_fused_loop_ctrl_pre_down(c, pre_qkv, msg_qkv, loop->ctrl_down_qkv, w.rms_eps)
            : ggml_mul_mat(c, loop->ctrl_down_qkv, ggml_inkling2_loop_ctrl_pre(c, pre_qkv, msg_qkv, w.rms_eps));
        ggml_tensor * gf_o      = ggml_mul_mat(c, loop->ctrl_gf_qkv, gate);
        ggml_tensor * qkv_in    = make_loop_ctrl_post_flagged(c, pre_qkv, msg_qkv, gf_o, nw_qkv, g.inp_flags, w.rms_eps);
        for (int j = 0; j < 3; ++j) {
            inputs[j] = ggml_view_2d(c, qkv_in, w.n_embd, row_b, qkv_in->nb[1], (size_t) j * qkv_in->nb[2]);
        }
    }

    // Fused attention projections (Q|K|V|Gate|R|Sink)
    ggml_tensor * Q_raw = nullptr, * K_raw = nullptr, * V_raw = nullptr;
    ggml_tensor * Gate  = nullptr, * R_raw = nullptr, * Sink  = nullptr;
    ggml_tensor * k_dyn = nullptr, * v_dyn = nullptr;

    if (same_qkv_input && L.wqkvgrs) {
        ggml_tensor * proj = ggml_mul_mat(c, L.wqkvgrs, inputs[0]);
        Q_raw = row_view_2d(c, proj, 0, q_sz);
        Gate  = row_view_2d(c, proj, q_sz, q_sz);
        R_raw = row_view_2d(c, proj, 2 * q_sz, r_sz);
        Sink  = row_view_2d(c, proj, 2 * q_sz + r_sz, Hq);
        K_raw = row_view_2d(c, proj, 2 * q_sz + r_sz + Hq, kv_sz);
        V_raw = row_view_2d(c, proj, 2 * q_sz + r_sz + Hq + kv_sz, kv_sz);
    } else {
        if (L.wqgrs) {
            ggml_tensor * proj = ggml_mul_mat(c, L.wqgrs, inputs[0]);
            Q_raw = row_view_2d(c, proj, 0, q_sz);
            Gate  = row_view_2d(c, proj, q_sz, q_sz);
            R_raw = row_view_2d(c, proj, 2 * q_sz, r_sz);
            Sink  = row_view_2d(c, proj, 2 * q_sz + r_sz, Hq);
        } else {
            Q_raw = ggml_mul_mat(c, L.wq, inputs[0]);
            Gate  = ggml_mul_mat(c, L.wqkv_gate, inputs[0]);
            R_raw = ggml_mul_mat(c, L.attn_rel, inputs[0]);
            Sink  = ggml_mul_mat(c, L.attn_sink_proj, inputs[0]);
        }
        if (!L.is_swa && L.wk_dyn && L.wv_dyn) {
            ggml_tensor * kd = ggml_mul_mat(c, L.wk_dyn, inputs[1]);
            K_raw = row_view_2d(c, kd, 0, kv_sz);
            k_dyn = row_view_2d(c, kd, kv_sz, (int64_t) w.sconv_kernel * Hk);
            ggml_tensor * vd = ggml_mul_mat(c, L.wv_dyn, inputs[2]);
            V_raw = row_view_2d(c, vd, 0, kv_sz);
            v_dyn = row_view_2d(c, vd, kv_sz, (int64_t) w.sconv_kernel * Hk);
        } else {
            K_raw = ggml_mul_mat(c, L.wk, inputs[1]);
            V_raw = ggml_mul_mat(c, L.wv, inputs[2]);
            if (!L.is_swa) {
                k_dyn = ggml_mul_mat(c, L.attn_k_conv_proj, inputs[1]);
                v_dyn = ggml_mul_mat(c, L.attn_v_conv_proj, inputs[2]);
            }
        }
    }

    ggml_tensor * Q = nullptr, * rt = nullptr;
    make_clip_qr_rel_table(
        c, Q_raw, L.attn_q_clip, R_raw, L.attn_rel_clip,
        L.is_swa ? nullptr : g.inp_pos, L.attn_rel_bias, Sink,
        D, w.d_rel, Hq, w.qk_clip, w.rel_clip, w.rms_eps,
        L.is_swa ? 0 : w.log_scale_floor, L.is_swa ? 0.0f : w.log_scale_alpha,
        &Q, &rt);

    // Tree K/V ShortConv + KV write:
    // SWA layers (20/24) fuse K+V into 1 launch (subop 41); Global layers keep 2 launches (dynamic taps).
    ggml_tensor * dep_kv = nullptr;
    if (L.is_swa) {
        dep_kv = make_tree_sconv_kv_write_kv(
            c, K_raw, V_raw, L.attn_k_conv, L.attn_v_conv,
            tc.prefix_conv[ip], tc.tree_conv[ip], g.inp_meta,
            tc.tree_k[ip], tc.tree_v[ip],
            D, Hk, m_stride, w.qk_clip, w.rms_eps);
    } else {
        ggml_tensor * dep_k = make_tree_sconv_kv_write(
            c, K_raw, L.attn_k_conv, tc.prefix_conv[ip], tc.tree_conv[ip], g.inp_meta, k_dyn,
            tc.tree_k[ip], nullptr, 0, D, Hk, 0, m_stride, w.qk_clip, w.rms_eps);
        dep_kv = make_tree_sconv_kv_write(
            c, V_raw, L.attn_v_conv, tc.prefix_conv[ip], tc.tree_conv[ip], g.inp_meta, v_dyn,
            tc.tree_v[ip], dep_k, kv_sz, D, Hk, 1, m_stride, w.qk_clip, w.rms_eps);
    }

    const bool shared_global = (!L.is_swa && tc.shared_kv_window > 0 && tc.planes[ip] > 1);
    ggml_tensor * attn_out = nullptr;
    if (!L.is_swa && kf_block_size > 0 && use_tier15 &&
        ip < (int) tc.remote_prefix_k.size() && tc.remote_prefix_k[ip] && g.inp_prefix_len_r) {
        // Dual-GPU 3-Tier Split-KV: Tier 1.5 Strix Halo attends warm prefix blocks ONLY (tree_k=nullptr!),
        // Tier 0 R9700 attends hot prefix blocks + W-tail rings + in-flight tree_k/v, merged via split_combine!
        ggml_tensor * part_r = make_tree_attn_rel(
            c, Q, tc.remote_prefix_k[ip], tc.remote_prefix_v[ip], nullptr, nullptr,
            rt, g.inp_meta, g.inp_prefix_len_r, nullptr, nullptr, 0, w.rel_extent, m_stride, path_b,
            /*block_size=*/kf_block_size, /*partial_out=*/1, /*exclude_sink=*/0,
            /*shared_kv_window=*/0, /*shared_tail_cap=*/0, /*planes_per_seq=*/tc.planes[ip]);
        g.expert_nodes.push_back(part_r);
        ggml_tensor * part_l = make_tree_attn_rel(
            c, Q, tc.prefix_k[ip], tc.prefix_v[ip], tc.tree_k[ip], tc.tree_v[ip],
            rt, g.inp_meta, g.inp_prefix_len, nullptr, dep_kv, 0, w.rel_extent, m_stride, path_b,
            /*block_size=*/kf_block_size, /*partial_out=*/1, /*exclude_sink=*/1,
            shared_global ? tc.shared_kv_window : 0,
            shared_global ? tc.shared_tail_cap : 0,
            /*planes_per_seq=*/tc.planes[ip]);
        attn_out = ggml_inkling2_attn_split_combine(c, part_r, part_l, Gate, D, Hq);
    } else {
        attn_out = make_tree_attn_rel(
            c, Q, tc.prefix_k[ip], tc.prefix_v[ip], tc.tree_k[ip], tc.tree_v[ip],
            rt, g.inp_meta, g.inp_prefix_len, Gate, dep_kv, swa, w.rel_extent, m_stride, path_b,
            /*block_size=*/(!L.is_swa && kf_block_size > 0) ? kf_block_size : 0,
            /*partial_out=*/0, /*exclude_sink=*/0,
            shared_global ? tc.shared_kv_window : 0,
            shared_global ? tc.shared_tail_cap : 0,
            /*planes_per_seq=*/tc.planes[ip]);
    }

    // Output proj + tree attn_post ShortConv
    cur = ggml_mul_mat(c, L.wo, attn_out);
    if (ip % w.attn_res_block == 0) {
        cur = make_tree_sconv(c, cur, L.attn_post_conv, tc.prefix_conv[ip], tc.tree_conv[ip],
                              g.inp_meta, nullptr, nullptr, 2 * kv_sz, 0, m_stride);
        blocks.push_back(inpSA);
    } else {
        cur = make_tree_sconv(c, cur, L.attn_post_conv, tc.prefix_conv[ip], tc.tree_conv[ip],
                              g.inp_meta, nullptr, inpSA, 2 * kv_sz, 0, m_stride);
    }
    ggml_tensor * ffn_inp = cur;

    // FFN AttnRes mix + optional flagged loop controller
    if (messages == nullptr) {
        cur = ggml_inkling2_attn_res_mix(
            c, cur, L.ffn_res_query, L.ffn_norm,
            blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
    } else {
        cur = ggml_inkling2_attn_res_mix(
            c, cur, L.ffn_res_query, nullptr,
            blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
        ggml_tensor * mf   = get_msg(Inkling2LoopLayer::CTRL_FFN);
        ggml_tensor * cd_q8 = loop->ctrl_down[Inkling2LoopLayer::CTRL_FFN];
        ggml_tensor * cd_w  = loop->ctrl_down_f16[Inkling2LoopLayer::CTRL_FFN]
            ? loop->ctrl_down_f16[Inkling2LoopLayer::CTRL_FFN]
            : cd_q8;
        ggml_tensor * gate = (cd_q8 && cd_q8->type == GGML_TYPE_Q8_0 && row_b <= 32)
            ? make_fused_loop_ctrl_pre_down(c, cur, mf, cd_q8, w.rms_eps)
            : ggml_mul_mat(c, cd_w, ggml_inkling2_loop_ctrl_pre(c, cur, mf, w.rms_eps));
        ggml_tensor * gf_o = ggml_mul_mat(c, loop->ctrl_gf[Inkling2LoopLayer::CTRL_FFN], gate);
        cur = make_loop_ctrl_post_flagged(c, cur, mf, gf_o, L.ffn_norm, g.inp_flags, w.rms_eps);
    }

    // Fused Q8_0 router (with FLAG_PADDING -> copies row 0 experts!) + MoE SITU (Single or Dual-GPU)
    const bool do_split = use_split && w.expert_backend && w.split_exp > 0 && w.split_exp < w.n_expert &&
                          L.exp_gate_up_tail && row_b <= 64 && w.n_expert == 128 && w.n_expert_used == 8;
    ggml_tensor * w_router = L.ffn_gate_inp_q8 ? L.ffn_gate_inp_q8 : L.ffn_gate_inp;

    if (do_split) {
        const int spct = dual_2dsd_split_pct(w);
        ggml_tensor * packed = make_fused_router_q8_topk(
            c, w_router, cur, L.ffn_exp_probs_b, g.inp_flags, w.n_expert_used, w.route_scale, 1e-20f, spct);
        ggml_tensor * hot_sel  = ggml_view_2d(c, packed, w.n_expert_used, row_b, packed->nb[1], 0);
        ggml_tensor * cold_sel = ggml_view_2d(c, packed, w.n_expert_used, row_b, packed->nb[1], packed->nb[2]);
        ggml_tensor * wsel     = ggml_view_2d(c, packed, w.n_expert_used, row_b, packed->nb[1], 2 * packed->nb[2]);
        ggml_build_forward_expand(gf, hot_sel);
        ggml_build_forward_expand(gf, cold_sel);
        ggml_build_forward_expand(gf, wsel);

        ggml_tensor * x3 = ggml_reshape_3d(c, cur, w.n_embd, 1, row_b);
        ggml_tensor * hot_gu   = ggml_mul_mat_id(c, L.ffn_gate_up_exps, x3, hot_sel);
        ggml_tensor * hot_act  = ggml_inkling2_situ(c, hot_gu, w.situ_beta, w.situ_linear_beta);
        ggml_tensor * hot_down = ggml_mul_mat_id(c, L.ffn_down_exps, hot_act, hot_sel);

        const bool use_f16_x = dual_2dsd_f16_transport();
        const int cold_sum_f16 = (use_f16_x && row_b >= 8) || (row_b > 32) ? 1 : 0;
        ggml_tensor * cold_in  = ((use_f16_x && row_b >= 8) || row_b > 32) ? ggml_cast(c, cur, GGML_TYPE_F16) : cur;
        ggml_tensor * cold_f32 = (cold_in->type == GGML_TYPE_F16) ? ggml_cast(c, cold_in, GGML_TYPE_F32) : cold_in;
        if (cold_f32 != cold_in) g.expert_nodes.push_back(cold_f32);
        ggml_tensor * cold_x3   = ggml_reshape_3d(c, cold_f32, w.n_embd, 1, row_b);
        ggml_tensor * cold_gu   = ggml_mul_mat_id(c, L.exp_gate_up_tail, cold_x3, cold_sel);
        ggml_tensor * cold_act  = ggml_inkling2_situ(c, cold_gu, w.situ_beta, w.situ_linear_beta);
        ggml_tensor * cold_down = ggml_mul_mat_id(c, L.exp_down_tail, cold_act, cold_sel);
        ggml_tensor * cold_sum  = ggml_inkling2_moe_masked_sum(c, cold_down, wsel, cold_sel, nullptr, cold_sum_f16);
        g.expert_nodes.insert(g.expert_nodes.end(), { cold_gu, cold_act, cold_down, cold_sum });
        ggml_build_forward_expand(gf, cold_sum);
        ggml_build_forward_expand(gf, hot_down);
        // Optional MALL prefetch (off by default matching inkling2_graph.cpp: reading ~35 MB/layer
        // unconditionally blocks the join kernel and adds ~1.3 ms/step on R9700!).
        const bool do_pf = (std::getenv("LUCE_INKLING2_PREFETCH") != nullptr &&
                            std::atoi(std::getenv("LUCE_INKLING2_PREFETCH")) != 0);
        if (do_pf) {
            prefetch_next_tree_layer(c, gf, hot_down, ip, phase_id, has_later, w);
        }

        // Fused MoE masked sum + peer join + tree ffn_post ShortConv + residual add (subop 42)
        cur = make_moe_masked_sum_tree_sconv_add(
            c, hot_down, wsel, hot_sel, cold_sum,
            L.ffn_post_conv, tc.prefix_conv[ip], tc.tree_conv[ip], g.inp_meta, ffn_inp,
            2 * kv_sz + w.n_embd, m_stride);
    } else {
        ggml_tensor * packed = make_fused_router_q8_topk(
            c, w_router, cur, L.ffn_exp_probs_b, g.inp_flags, w.n_expert_used, w.route_scale, 1e-20f, 0);
        ggml_tensor * sel  = ggml_view_2d(c, packed, w.n_expert_used, row_b, packed->nb[1], 0);
        ggml_tensor * wsel = ggml_view_2d(c, packed, w.n_expert_used, row_b, packed->nb[1], packed->nb[2]);

        ggml_tensor * x3   = ggml_reshape_3d(c, cur, w.n_embd, 1, row_b);
        ggml_tensor * gu   = ggml_mul_mat_id(c, L.ffn_gate_up_exps, x3, sel);
        ggml_tensor * act  = ggml_inkling2_situ(c, gu, w.situ_beta, w.situ_linear_beta);
        ggml_tensor * down = ggml_mul_mat_id(c, L.ffn_down_exps, act, sel);
        // Fused MoE masked sum + tree ffn_post ShortConv + residual add (subop 42)
        cur = make_moe_masked_sum_tree_sconv_add(
            c, down, wsel, sel, nullptr,
            L.ffn_post_conv, tc.prefix_conv[ip], tc.tree_conv[ip], g.inp_meta, ffn_inp,
            2 * kv_sz + w.n_embd, m_stride);
    }
    return cur;
}

static bool alloc_phase_graph(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        Inkling2TwoDSDPhaseGraph & g,
        size_t max_nodes,
        bool use_split,
        bool use_tier15) {
    const bool need_sched = (use_split || use_tier15) && !g.expert_nodes.empty() && w.expert_backend && cache.split_cpu;
    if (need_sched) {
        ggml_backend_t backends[3] = { target_backend, w.expert_backend, cache.split_cpu };
        g.sched = ggml_backend_sched_new(backends, nullptr, 3, max_nodes, true, false);
        if (!g.sched) return false;
        ggml_backend_sched_set_late_cross_input_split(g.sched, true);
        if (dual_2dsd_batch_copies()) {
            ggml_backend_sched_set_batch_split_copies(g.sched, true);
        }
        ggml_backend_cuda_set_copy_tail(target_backend, true);
        ggml_backend_cuda_set_copy_tail(w.expert_backend, true);
        ggml_backend_sched_reset(g.sched);
        const std::unordered_set<const ggml_tensor *> on_exp(g.expert_nodes.begin(), g.expert_nodes.end());
        for (int i = 0; i < ggml_graph_n_nodes(g.gf); ++i) {
            ggml_tensor * node = ggml_graph_node(g.gf, i);
            for (ggml_tensor * src : node->src) {
                for (ggml_tensor * t = src; t; t = t->view_src) {
                    if ((t->flags & GGML_TENSOR_FLAG_INPUT) && !t->buffer) {
                        ggml_backend_t pin_be = (t == g.inp_prefix_len_r) ? w.expert_backend : target_backend;
                        ggml_backend_sched_set_tensor_backend(g.sched, t, pin_be);
                    }
                }
            }
            const bool is_view = node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW ||
                                 node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE ||
                                 node->op == GGML_OP_TRANSPOSE;
            if (on_exp.count(node)) {
                ggml_backend_sched_set_tensor_backend(g.sched, node, w.expert_backend);
            } else if (!is_view && ggml_backend_supports_op(target_backend, node)) {
                ggml_backend_sched_set_tensor_backend(g.sched, node, target_backend);
            }
        }
        return ggml_backend_sched_alloc_graph(g.sched, g.gf);
    }
    g.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
    return g.alloc && ggml_gallocr_alloc_graph(g.alloc, g.gf);
}

static bool build_phase_graph(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        Inkling2TwoDSDPhaseGraph & g,
        int phase_id, // 0=prelude, 1=core, 2=coda
        int row_b,
        int path_b,
        bool has_later,
        bool use_split,
        int kf_bs,
        int kf_max_blocks,
        bool use_tier15) {
    auto & tc = cache.tree;
    const int m_stride = 11 + path_b;
    const size_t max_nodes = (phase_id == 1) ? 12288 : 6144;
    struct ggml_init_params gp = { max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false) + (12 << 20), nullptr, true };
    g.ctx = ggml_init(gp);
    if (!g.ctx) return false;
    g.gf = ggml_new_graph_custom(g.ctx, max_nodes, false);
    ggml_context * c = g.ctx;
    g.expert_nodes.clear();

    const int n_planes_phase = (phase_id == 0) ? 1 : w.n_passes;
    const bool use_paged = (kf_bs > 0 && kf_max_blocks > 0);
    const int pref_elems = use_paged ? (2 + kf_max_blocks) : ((tc.n_seq_slots > 1) ? (n_planes_phase * tc.n_seq_slots) : 1);

    g.inp_meta = ggml_new_tensor_2d(c, GGML_TYPE_I32, m_stride, row_b);
    if (use_paged && tc.n_seq_slots > 1) {
        g.inp_prefix_len = ggml_new_tensor_2d(c, GGML_TYPE_I32, pref_elems, tc.n_seq_slots);
    } else {
        g.inp_prefix_len = ggml_new_tensor_1d(c, GGML_TYPE_I32, pref_elems);
    }
    g.inp_prefix_len_r = nullptr;
    if (use_paged && use_tier15) {
        g.inp_prefix_len_r = (tc.n_seq_slots > 1)
            ? ggml_new_tensor_2d(c, GGML_TYPE_I32, pref_elems, tc.n_seq_slots)
            : ggml_new_tensor_1d(c, GGML_TYPE_I32, pref_elems);
        ggml_set_input(g.inp_prefix_len_r);
    }
    g.inp_pos        = ggml_new_tensor_1d(c, GGML_TYPE_I32, row_b);
    g.inp_flags      = ggml_new_tensor_1d(c, GGML_TYPE_I32, row_b);
    g.inp_slot       = ggml_new_tensor_1d(c, GGML_TYPE_I32, row_b);
    ggml_set_input(g.inp_meta);
    ggml_set_input(g.inp_prefix_len);
    ggml_set_input(g.inp_pos);
    ggml_set_input(g.inp_flags);
    ggml_set_input(g.inp_slot);

    if (phase_id == 0) {
        // ── Prelude (L0..core_start-1) ─────────────────────────────────
        g.inp_tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, row_b);
        ggml_set_input(g.inp_tokens);

        ggml_tensor * cur = rms_norm_w(c, ggml_get_rows(c, w.tok_embd, g.inp_tokens), w.tok_norm, w.rms_eps);
        std::vector<ggml_tensor *> blocks;
        ggml_tensor * tap_l2 = nullptr;

        for (int ip = 0; ip < w.core_start; ++ip) {
            if (ip == w.twodsd.feature_layers[0]) tap_l2 = cur;
            cur = build_tree_layer(c, g.gf, w, tc, g, ip, 0, cur, blocks, nullptr, nullptr, false, use_split, row_b, path_b, kf_bs, use_tier15);
        }
        for (size_t b = 0; b < tc.pre_blocks.size() && b < blocks.size(); ++b) {
            ggml_build_forward_expand(g.gf, ggml_set_rows(c, tc.pre_blocks[b], blocks[b], g.inp_slot));
        }
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, tc.pre_partial, cur, g.inp_slot));
        if (tap_l2) {
            ggml_build_forward_expand(g.gf, ggml_set_rows(c, tc.pre_tap, tap_l2, g.inp_slot));
        }
    } else if (phase_id == 1) {
        // ── Fused Multi-Pass Core (L6..17 across passes 0/1/2 together!) ──
        g.inp_flat_cur = ggml_new_tensor_1d(c, GGML_TYPE_I32, row_b);
        ggml_set_input(g.inp_flat_cur);
        if (has_later) {
            g.inp_flat_prev = ggml_new_tensor_1d(c, GGML_TYPE_I32, row_b);
            ggml_set_input(g.inp_flat_prev);
        }

        std::vector<ggml_tensor *> blocks;
        blocks.reserve(tc.pre_blocks.size() + tc.core_blocks.size());
        for (ggml_tensor * pb : tc.pre_blocks) {
            blocks.push_back(ggml_get_rows(c, pb, g.inp_slot));
        }
        ggml_tensor * cur = ggml_get_rows(c, tc.pre_partial, g.inp_slot);

        std::vector<ggml_tensor *> bank_blocks;
        ggml_tensor * bank_out = nullptr;
        if (has_later) {
            // Bank skips core_blocks[0] (= pre_partial closed at L6), keeping core_blocks[1..] + core_partial[prev]
            for (size_t b = 1; b < tc.core_blocks.size(); ++b) {
                bank_blocks.push_back(ggml_get_rows(c, tc.core_blocks[b], g.inp_flat_prev));
            }
            bank_out = ggml_get_rows(c, tc.core_partial, g.inp_flat_prev);
        }

        ggml_tensor * core_tap_out[3] = { nullptr, nullptr, nullptr };
        for (int ip = w.core_start; ip < w.core_end; ++ip) {
            for (int ti = 0; ti < 3; ++ti) {
                if (ip == w.twodsd.feature_layers[1 + ti]) core_tap_out[ti] = cur;
            }
            cur = build_tree_layer(c, g.gf, w, tc, g, ip, 1, cur, blocks,
                                   has_later ? &bank_blocks : nullptr, bank_out,
                                   has_later, use_split, row_b, path_b, kf_bs, use_tier15);
        }

        const size_t pre_n = tc.pre_blocks.size();
        for (size_t b = 0; b < tc.core_blocks.size() && (pre_n + b) < blocks.size(); ++b) {
            ggml_build_forward_expand(g.gf, ggml_set_rows(c, tc.core_blocks[b], blocks[pre_n + b], g.inp_flat_cur));
        }
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, tc.core_partial, cur, g.inp_flat_cur));
        for (int ti = 0; ti < 3; ++ti) {
            if (core_tap_out[ti]) {
                ggml_build_forward_expand(g.gf, ggml_set_rows(c, tc.core_taps[ti], core_tap_out[ti], g.inp_flat_cur));
            }
        }
    } else {
        // ── Fused Multi-Depth Coda (L18..23 across depths {1, 2, 3}!) ──
        g.inp_flat_cur = ggml_new_tensor_1d(c, GGML_TYPE_I32, row_b); // depth_idx * S + slot
        ggml_set_input(g.inp_flat_cur);

        std::vector<ggml_tensor *> blocks;
        blocks.reserve(tc.pre_blocks.size() + tc.core_blocks.size() + 2);
        for (ggml_tensor * pb : tc.pre_blocks) {
            blocks.push_back(ggml_get_rows(c, pb, g.inp_slot));
        }
        for (ggml_tensor * cb : tc.core_blocks) {
            blocks.push_back(ggml_get_rows(c, cb, g.inp_flat_cur));
        }
        ggml_tensor * cur = ggml_get_rows(c, tc.core_partial, g.inp_flat_cur);

        ggml_tensor * tap_l22 = nullptr;
        for (int ip = w.core_end; ip < w.n_layer_phys; ++ip) {
            if (ip == w.twodsd.feature_layers[4]) tap_l22 = cur;
            cur = build_tree_layer(c, g.gf, w, tc, g, ip, 2, cur, blocks, nullptr, nullptr, false, use_split, row_b, path_b, kf_bs, use_tier15);
        }

        // Assemble 5-tap depth features [5*H, row_b] -> write into node_features[depth_idx*S + slot] (F16)
        ggml_tensor * t0 = ggml_get_rows(c, tc.pre_tap, g.inp_slot);
        ggml_tensor * t1 = ggml_get_rows(c, tc.core_taps[0], g.inp_flat_cur);
        ggml_tensor * t2 = ggml_get_rows(c, tc.core_taps[1], g.inp_flat_cur);
        ggml_tensor * t3 = ggml_get_rows(c, tc.core_taps[2], g.inp_flat_cur);
        ggml_tensor * f01  = ggml_concat(c, t0, t1, 0);
        ggml_tensor * f23  = ggml_concat(c, t2, t3, 0);
        ggml_tensor * f03  = ggml_concat(c, f01, f23, 0);
        ggml_tensor * f04  = ggml_concat(c, f03, tap_l22, 0); // F32 [5*H, row_b]
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, tc.node_features, f04, g.inp_flat_cur));

        // Final AttnRes mix + output_norm + LM head (+ optional 79/21 split) + softcap + argmax
        ggml_tensor * out_norm = ggml_inkling2_attn_res_mix(
            c, cur, w.output_res_query, w.output_norm,
            blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);

        ggml_tensor * raw_logits = nullptr;
        if (use_split && w.expert_backend && w.output_tail && row_b <= dual_2dsd_lm_split_max()) {
            const int64_t head_rows = w.output->ne[1] - w.output_tail->ne[1];
            ggml_tensor * head_w = ggml_view_2d(c, w.output, w.output->ne[0], head_rows, w.output->nb[1], 0);
            ggml_tensor * head   = ggml_mul_mat(c, head_w, out_norm);
            ggml_tensor * tail   = ggml_mul_mat(c, w.output_tail, out_norm);
            ggml_backend_cuda_mul_mat_rdna4_rows(head);
            ggml_backend_cuda_mul_mat_rdna4_rows(tail);
            g.expert_nodes.push_back(tail);
            raw_logits = ggml_concat(c, head, tail, 0);
        } else {
            raw_logits = ggml_mul_mat(c, w.output, out_norm);
        }
        ggml_tensor * logits = ggml_inkling2_softcap_slice(c, raw_logits, w.n_vocab_unpadded, w.logit_softcap);
        g.out_logits = logits;
        ggml_set_output(g.out_logits);
        g.out_greedy = ggml_argmax(c, logits);
        ggml_set_output(g.out_greedy);
        ggml_build_forward_expand(g.gf, g.out_greedy);
    }

    if (!alloc_phase_graph(target_backend, w, cache, g, max_nodes, use_split, use_tier15)) {
        if (g.sched) { ggml_backend_sched_free(g.sched); g.sched = nullptr; }
        if (g.alloc) { ggml_gallocr_free(g.alloc); g.alloc = nullptr; }
        ggml_free(g.ctx); g.ctx = nullptr;
        g.gf = nullptr;
        return false;
    }
    g.built = true;
    return true;
}

struct StagedPhaseInputs {
    int row_b  = 0;
    int path_b = 0;
    std::vector<int32_t> meta;
    std::vector<int32_t> pos;
    std::vector<int32_t> flags;
    std::vector<int32_t> tokens;
    std::vector<int32_t> slot;
    std::vector<int32_t> flat_cur;
    std::vector<int32_t> flat_prev;
};

static StagedPhaseInputs stage_phase_rows(
        const std::vector<const Inkling2TwoDSDNodeDesc *> & rows,
        const std::vector<int> & planes,
        const std::vector<int> & flags_in,
        const std::vector<int> & extra_in,
        int max_nodes,
        const int * prefix_lens,
        int n_seq_slots,
        int n_planes_phase) {
    const int n = (int) rows.size();
    int max_path = 1;
    for (const auto * nd : rows) {
        max_path = std::max(max_path, (int) nd->ancestors.size() + 1);
    }
    const int row_b  = bucket_rows(n);
    const int path_b = bucket_path(max_path);
    const int m_stride = 11 + path_b;
    const int S = max_nodes + 1;
    const int scratch = max_nodes;
    const int pad_prefix_len = (prefix_lens && n_seq_slots > 0) ? prefix_lens[0] : 0;

    StagedPhaseInputs st;
    st.row_b  = row_b;
    st.path_b = path_b;
    st.meta.assign((size_t) row_b * m_stride, -1);
    st.pos.assign(row_b, pad_prefix_len);
    st.flags.assign(row_b, kInkling2TwoDSDFlagsPadding);
    st.tokens.assign(row_b, rows[0]->token);
    st.slot.assign(row_b, scratch);
    st.flat_cur.assign(row_b, scratch);
    st.flat_prev.assign(row_b, scratch);

    // Map ((seq_slot, slot) -> row index in this call) for same-call ShortConv hist_row resolution
    std::unordered_map<int64_t, int> row_of;
    row_of.reserve((size_t) n * 2);
    for (int r = 0; r < n; ++r) {
        const int64_t key = (int64_t) rows[r]->seq_slot * S + rows[r]->slot;
        row_of[key] = r;
    }

    // Initialize padding rows safely: scratch slot, plane 0, prefix history (-1,-2,-3), self in path[0]
    for (int r = n; r < row_b; ++r) {
        int32_t * mt = st.meta.data() + (size_t) r * m_stride;
        mt[0] = scratch;
        mt[1] = pad_prefix_len;
        mt[2] = 0;
        mt[3] = kInkling2TwoDSDFlagsPadding;
        mt[4] = 0;
        mt[5] = -1; mt[6] = -1; mt[7] = -1;
        mt[8] = -1; mt[9] = -2; mt[10] = -3;
        mt[11] = scratch;
    }

    for (int r = 0; r < n; ++r) {
        const auto & nd = *rows[r];
        const int ss        = (n_seq_slots > 1) ? nd.seq_slot : 0;
        const int plane     = planes[r];
        const int plane_eff = ss * n_planes_phase + plane;
        const int flg       = flags_in[r];
        const int ext       = extra_in[r];
        int32_t * mt = st.meta.data() + (size_t) r * m_stride;

        mt[0] = nd.slot;
        mt[1] = nd.pos;
        mt[2] = plane_eff;
        mt[3] = flg;
        mt[4] = ext;

        const int n_anc = (int) nd.ancestors.size();
        for (int j = 0; j < 3; ++j) {
            const int behind = n_anc - 3 + j;
            if (behind >= 0) {
                const int anc_slot = nd.ancestors[behind];
                const int64_t anc_key = (int64_t) ss * S + anc_slot;
                auto it = row_of.find(anc_key);
                if (it != row_of.end() && planes[it->second] == plane) {
                    mt[5 + j] = it->second; // same-call input row (same sequence & plane)
                    mt[8 + j] = anc_slot;
                } else {
                    mt[5 + j] = -1;
                    mt[8 + j] = anc_slot;   // tree_conv[plane_eff, anc_slot]
                }
            } else {
                mt[5 + j] = -1;
                // prefix_conv index = 3 + behind in {0, 1, 2}, encoded as -(idx + 1) in {-1, -2, -3}
                const int pidx = 3 + behind;
                mt[8 + j] = -(pidx + 1);
            }
        }
        for (int e = 0; e < n_anc; ++e) mt[11 + e] = nd.ancestors[e];
        mt[11 + n_anc] = nd.slot;

        st.pos[r]       = nd.pos;
        st.flags[r]     = flg;
        st.tokens[r]    = nd.token;
        st.slot[r]      = ss * S + nd.slot;
        st.flat_cur[r]  = (ss * 3 + plane) * S + nd.slot;
        st.flat_prev[r] = (ss * 3 + std::max(0, plane - 1)) * S + nd.slot;
    }
    return st;
}

static bool run_phase(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        int phase_id,
        const StagedPhaseInputs & st,
        const int * prefix_lens,
        int n_seq_slots,
        int n_real,
        bool has_later,
        bool allow_split,
        std::vector<int32_t> * out_greedy = nullptr,
        Inkling2TwoDSDAdvanceSampleOut * out_sample = nullptr,
        const Inkling2KvFlash * const * slot_kfs = nullptr) {
    auto & tc = cache.tree;
    const Inkling2KvFlash * any_kf = nullptr;
    bool any_t15_resident = false;
    if (slot_kfs) {
        for (int s = 0; s < n_seq_slots; ++s) {
            if (slot_kfs[s] && slot_kfs[s]->active()) {
                if (!any_kf) any_kf = slot_kfs[s];
                if (slot_kfs[s]->tier15_active() && slot_kfs[s]->tier15_resident_blocks() > 0) {
                    any_t15_resident = true;
                }
            }
        }
    } else if (cache.kvflash && cache.kvflash->active()) {
        any_kf = cache.kvflash.get();
        if (any_kf->tier15_active() && any_kf->tier15_resident_blocks() > 0) {
            any_t15_resident = true;
        }
    }
    const bool use_paged  = (any_kf != nullptr);
    const bool use_tier15 = (use_paged && any_t15_resident);
    const int  kf_bs      = use_paged ? any_kf->block_size() : 0;
    const int  kf_max_blk = use_paged ? any_kf->max_blocks() : 0;

    const bool batch_inv = batch_invariant_2dsd_enabled();
    const int min_dual = dual_2dsd_min_rows();
    const bool phase_split_ok = ((dual_2dsd_phases_mask() & (1 << phase_id)) != 0);
    const bool use_split = !batch_inv && allow_split && phase_split_ok && w.expert_backend && w.split_exp > 0 &&
                           st.row_b >= min_dual && st.row_b <= 64;
    const int split_flags = (use_split ? 1 : 0) | (use_tier15 ? 2 : 0);
    const auto key = std::make_tuple(phase_id, st.row_b, st.path_b, has_later ? 1 : 0, split_flags);
    auto & g = tc.phase_graphs[key];
    if (!g.built) {
        if (!build_phase_graph(target_backend, w, cache, g, phase_id, st.row_b, st.path_b,
                               has_later, use_split, kf_bs, kf_max_blk, use_tier15)) {
            return false;
        }
    }

    const bool prof = twodsd_profile_enabled();
    const auto t0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    const int n_planes_phase = (phase_id == 0) ? 1 : w.n_passes;
    ggml_backend_tensor_set(g.inp_meta, st.meta.data(), 0, st.meta.size() * sizeof(int32_t));
    if (use_paged) {
        const int row_elems = 2 + kf_max_blk;
        if (tc.n_seq_slots > 1) {
            std::vector<int32_t> pl_buf((size_t) row_elems * tc.n_seq_slots, -1);
            std::vector<int32_t> tmp;
            for (int s = 0; s < tc.n_seq_slots; ++s) {
                const int32_t pl = (prefix_lens && s < n_seq_slots) ? prefix_lens[s] : 0;
                const Inkling2KvFlash * skf = (slot_kfs && s < n_seq_slots) ? slot_kfs[s] : nullptr;
                int32_t * dst = pl_buf.data() + (size_t) s * row_elems;
                if (skf && skf->active()) {
                    skf->fill_tree_prefix_buf(pl, /*remote=*/false, tmp);
                    std::memcpy(dst, tmp.data(), (size_t) row_elems * sizeof(int32_t));
                } else {
                    dst[0] = pl;
                    dst[1] = 0;
                }
            }
            ggml_backend_tensor_set(g.inp_prefix_len, pl_buf.data(), 0, pl_buf.size() * sizeof(int32_t));
            if (use_tier15 && g.inp_prefix_len_r) {
                std::fill(pl_buf.begin(), pl_buf.end(), -1);
                for (int s = 0; s < tc.n_seq_slots; ++s) {
                    const int32_t pl = (prefix_lens && s < n_seq_slots) ? prefix_lens[s] : 0;
                    const Inkling2KvFlash * skf = (slot_kfs && s < n_seq_slots) ? slot_kfs[s] : nullptr;
                    int32_t * dst = pl_buf.data() + (size_t) s * row_elems;
                    if (skf && skf->active()) {
                        skf->fill_tree_prefix_buf(pl, /*remote=*/true, tmp);
                        std::memcpy(dst, tmp.data(), (size_t) row_elems * sizeof(int32_t));
                    } else {
                        dst[0] = pl;
                        dst[1] = 0;
                    }
                }
                ggml_backend_tensor_set(g.inp_prefix_len_r, pl_buf.data(), 0, pl_buf.size() * sizeof(int32_t));
            }
        } else {
            const int32_t pref_len = (prefix_lens && n_seq_slots > 0) ? prefix_lens[0] : tc.prefix_len;
            std::vector<int32_t> pbuf;
            any_kf->fill_tree_prefix_buf(pref_len, /*remote=*/false, pbuf);
            ggml_backend_tensor_set(g.inp_prefix_len, pbuf.data(), 0, pbuf.size() * sizeof(int32_t));
            if (use_tier15 && g.inp_prefix_len_r) {
                any_kf->fill_tree_prefix_buf(pref_len, /*remote=*/true, pbuf);
                ggml_backend_tensor_set(g.inp_prefix_len_r, pbuf.data(), 0, pbuf.size() * sizeof(int32_t));
            }
        }
    } else if (tc.n_seq_slots > 1) {
        std::vector<int32_t> pl_buf((size_t) n_planes_phase * tc.n_seq_slots, 0);
        for (int s = 0; s < tc.n_seq_slots; ++s) {
            const int32_t pl = (prefix_lens && s < n_seq_slots) ? prefix_lens[s] : 0;
            for (int p = 0; p < n_planes_phase; ++p) {
                pl_buf[(size_t) s * n_planes_phase + p] = pl;
            }
        }
        ggml_backend_tensor_set(g.inp_prefix_len, pl_buf.data(), 0, pl_buf.size() * sizeof(int32_t));
    } else {
        const int32_t pref_len = (prefix_lens && n_seq_slots > 0) ? prefix_lens[0] : tc.prefix_len;
        ggml_backend_tensor_set(g.inp_prefix_len, &pref_len, 0, sizeof(pref_len));
    }
    ggml_backend_tensor_set(g.inp_pos,        st.pos.data(),   0, st.pos.size()   * sizeof(int32_t));
    ggml_backend_tensor_set(g.inp_flags,      st.flags.data(), 0, st.flags.size() * sizeof(int32_t));
    ggml_backend_tensor_set(g.inp_slot,       st.slot.data(),  0, st.slot.size()  * sizeof(int32_t));
    if (g.inp_tokens)    ggml_backend_tensor_set(g.inp_tokens,    st.tokens.data(),    0, st.tokens.size()    * sizeof(int32_t));
    if (g.inp_flat_cur)  ggml_backend_tensor_set(g.inp_flat_cur,  st.flat_cur.data(),  0, st.flat_cur.size()  * sizeof(int32_t));
    if (g.inp_flat_prev) ggml_backend_tensor_set(g.inp_flat_prev, st.flat_prev.data(), 0, st.flat_prev.size() * sizeof(int32_t));

    ggml_status status = GGML_STATUS_SUCCESS;
    {
        const ScopedCudaGraphOverrides inv_guard(false, 0, false, 0, /*mmvq_batch_invariant=*/batch_inv);
        if (g.sched) {
            status = ggml_backend_sched_graph_compute_async(g.sched, g.gf);
            // Only issue Strix idle hold on Coda (or when explicitly synced) to keep Strix warm across
            // the ~2.6 ms drafter window without injecting a hold barrier between Prelude -> Core -> Coda!
            if (status == GGML_STATUS_SUCCESS && w.expert_backend && out_greedy) {
                ggml_backend_cuda_hold(w.expert_backend, 8000);
            }
            if (status == GGML_STATUS_SUCCESS && (out_greedy || out_sample || prof)) {
                ggml_backend_sched_synchronize(g.sched);
            }
        } else {
            // Chain Prelude/Core asynchronously on the same GPU stream when no host readback is requested;
            // Coda (with out_greedy) performs the single end-of-step sync!
            status = (out_greedy || out_sample || prof)
                ? ggml_backend_graph_compute(target_backend, g.gf)
                : ggml_backend_graph_compute_async(target_backend, g.gf);
        }
    }
    if (status != GGML_STATUS_SUCCESS) return false;

    const int n_out = std::min(std::max(n_real, 0), st.row_b);
    if (out_greedy && g.out_greedy) {
        out_greedy->resize((size_t) n_out);
        if (n_out > 0) {
            ggml_backend_tensor_get(g.out_greedy, out_greedy->data(), 0, (size_t) n_out * sizeof(int32_t));
        }
    }
    if (out_sample && g.out_logits && n_out > 0) {
        out_sample->logits_dev = g.out_logits;
        out_sample->n_vocab    = w.n_vocab_unpadded;
        out_sample->top_cands.clear();
        if (out_sample->fast_temp > 0.0f) {
            // Rebuild cached 1-node top-cand extractor if temp or active row count changed
            const bool need_build = !g.sample_gf || !g.sample_cands ||
                                    g.sample_cands->ne[1] != n_out ||
                                    g.sample_temp != out_sample->fast_temp;
            if (need_build) {
                if (g.sample_alloc) { ggml_gallocr_free(g.sample_alloc); g.sample_alloc = nullptr; }
                if (g.sample_ctx)   { ggml_free(g.sample_ctx); g.sample_ctx = nullptr; }
                g.sample_gf = nullptr;
                g.sample_logits = nullptr;
                g.sample_cands = nullptr;
                struct ggml_init_params sgp = { 16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false) + 4096, nullptr, true };
                g.sample_ctx = ggml_init(sgp);
                if (!g.sample_ctx) return false;
                g.sample_gf = ggml_new_graph_custom(g.sample_ctx, 16, false);
                // Leaf alias spanning ONLY the `n_out` active Coda rows (skips padding rows!)
                g.sample_logits = ggml_new_tensor_2d(g.sample_ctx, GGML_TYPE_F32, w.n_vocab_unpadded, n_out);
                g.sample_logits->data   = g.out_logits->data;
                g.sample_logits->buffer = g.out_logits->buffer;
                g.sample_cands = make_exact_gpu_top_cands(g.sample_ctx, g.sample_logits, out_sample->fast_temp);
                ggml_set_output(g.sample_cands);
                ggml_build_forward_expand(g.sample_gf, g.sample_cands);
                g.sample_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
                if (!g.sample_alloc || !ggml_gallocr_alloc_graph(g.sample_alloc, g.sample_gf)) {
                    if (g.sample_alloc) { ggml_gallocr_free(g.sample_alloc); g.sample_alloc = nullptr; }
                    ggml_free(g.sample_ctx); g.sample_ctx = nullptr;
                    g.sample_gf = nullptr; g.sample_logits = nullptr; g.sample_cands = nullptr;
                    return false;
                }
                g.sample_temp = out_sample->fast_temp;
            }
            g.sample_logits->data   = g.out_logits->data;
            g.sample_logits->buffer = g.out_logits->buffer;
            if (ggml_backend_graph_compute(target_backend, g.sample_gf) != GGML_STATUS_SUCCESS) {
                return false;
            }
            out_sample->top_cands.resize((size_t) 516 * n_out);
            ggml_backend_tensor_get(g.sample_cands, out_sample->top_cands.data(), 0,
                                    out_sample->top_cands.size() * sizeof(int32_t));
        }
    }
    if (prof) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        static const char * kNames[3] = { "Prelude", "Core", "Coda" };
        std::fprintf(stderr, "[2dsd-prof] %-7s row_b=%d path_b=%d split=%d ms=%.3f\n",
                     kNames[phase_id], st.row_b, st.path_b, use_split ? 1 : 0, ms);
    }
    return true;
}

} // namespace

// ─── 60-Plane Multi-Depth Prefill Chunk ─────────────────────────────────
bool inkling2_2dsd_prefill_chunk(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        const int32_t * tokens,
        int n_tokens,
        int pos0,
        bool is_last_chunk,
        int32_t & out_greedy_tok,
        std::vector<float> * out_last_logits,
        int capture_q_rows) {
    if (!w.twodsd.loaded || !cache.tree.active || n_tokens <= 0) return false;
    auto & tc = cache.tree;
    const int Hq = w.n_head, Hk = w.n_head_kv, D = w.head_dim;
    const int q_sz = Hq * D, kv_sz = Hk * D, r_sz = w.d_rel * Hq;

    const bool use_paged = (cache.kvflash && cache.kvflash->active());
    const int  kf_bs     = use_paged ? cache.kvflash->block_size() : 0;
    const int  q_cap_n   = (use_paged && capture_q_rows > 0) ? std::min(capture_q_rows, n_tokens) : 0;

    if (use_paged) {
        if (!cache.kvflash->prepare_span(pos0, n_tokens)) return false;
    }
    const bool use_tier15 = (use_paged && cache.kvflash->tier15_active() &&
                             cache.kvflash->tier15_resident_blocks() > 0);
    const bool share_coda = prefill_share_coda_enabled();

    // Map (phys layer ip, logical plane) -> global plane index [0 .. L_global-1] matching create_inkling2_cache!
    std::vector<std::array<int, 4>> global_plane_idx((size_t) w.n_layer_phys, {-1, -1, -1, -1});
    int n_global_planes = 0;
    if (use_paged) {
        for (int ip = 0; ip < w.n_layer_phys; ++ip) {
            if (w.layers[ip].is_swa) continue;
            const int p_st = (int) tc.prefix_k[ip]->ne[3];
            const int P    = tc.planes[ip];
            if (p_st == 1 && P > 1) {
                global_plane_idx[(size_t) ip][P - 1] = n_global_planes++;
            } else {
                for (int p = 0; p < P && p < 4; ++p) {
                    global_plane_idx[(size_t) ip][p] = n_global_planes++;
                }
            }
        }
    }

    const size_t max_nodes = 24576;
    struct ggml_init_params gp = { max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false) + (24 << 20), nullptr, true };
    ggml_context * c = ggml_init(gp);
    if (!c) return false;
    ggml_cgraph * gf = ggml_new_graph_custom(c, max_nodes, false);

    ggml_tensor * inp_tok       = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
    ggml_tensor * inp_pos       = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
    ggml_tensor * inp_meta      = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
    ggml_tensor * inp_meta_r    = nullptr;
    ggml_tensor * inp_bt_local  = nullptr;
    ggml_tensor * inp_bt_remote = nullptr;
    ggml_tensor * q_cap_out     = nullptr;
    std::vector<ggml_tensor *> expert_nodes;
    ggml_set_input(inp_tok);
    ggml_set_input(inp_pos);
    ggml_set_input(inp_meta);
    if (use_paged) {
        inp_bt_local = ggml_new_tensor_1d(c, GGML_TYPE_I32, cache.kvflash->max_blocks());
        ggml_set_input(inp_bt_local);
        if (use_tier15) {
            inp_meta_r    = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
            inp_bt_remote = ggml_new_tensor_1d(c, GGML_TYPE_I32, cache.kvflash->max_blocks());
            ggml_set_input(inp_meta_r);
            ggml_set_input(inp_bt_remote);
        }
        if (q_cap_n > 0 && n_global_planes > 0) {
            q_cap_out = ggml_new_tensor_4d(c, GGML_TYPE_F32, D, Hq, n_global_planes, q_cap_n);
            ggml_set_output(q_cap_out);
        }
    }

    auto run_prefill_layer = [&](int ip, int plane, ggml_tensor * cur,
                                 std::vector<ggml_tensor *> & blocks,
                                 const std::vector<ggml_tensor *> * bank,
                                 ggml_tensor * bank_out) -> ggml_tensor * {
        const auto & L = w.layers[ip];
        ggml_tensor * inpSA = cur;
        const bool shared_global = (!L.is_swa && tc.shared_kv_window > 0 && tc.planes[ip] > 1);
        const int  kv_plane      = (tc.prefix_k[ip]->ne[3] == 1) ? 0 : plane;
        ggml_tensor * k_p = shared_global
            ? ggml_view_3d(c, tc.prefix_k[ip], tc.prefix_k[ip]->ne[0], tc.global_pool_cap, tc.prefix_k[ip]->ne[2],
                           tc.prefix_k[ip]->nb[1], tc.prefix_k[ip]->nb[2], 0)
            : plane_view_kv(c, tc.prefix_k[ip], kv_plane);
        ggml_tensor * v_p = shared_global
            ? ggml_view_3d(c, tc.prefix_v[ip], tc.prefix_v[ip]->ne[0], tc.global_pool_cap, tc.prefix_v[ip]->ne[2],
                           tc.prefix_v[ip]->nb[1], tc.prefix_v[ip]->nb[2], 0)
            : plane_view_kv(c, tc.prefix_v[ip], kv_plane);
        ggml_tensor * st  = plane_view_conv(c, tc.prefix_conv[ip], plane);

        ggml_tensor * messages = nullptr;
        const Inkling2LoopLayer * loop = nullptr;
        if (bank_out != nullptr) {
            loop = &w.loop_layers[ip - w.core_start];
            messages = ggml_inkling2_attn_res_mix(
                c, bank_out, loop->msg_query, nullptr,
                (bank && !bank->empty()) ? bank->data() : nullptr,
                bank ? (int) bank->size() : 0, w.rms_eps);
        }
        auto get_msg = [&](int ctrl) -> ggml_tensor * {
            return ggml_view_2d(c, messages, w.n_embd, n_tokens,
                                messages->nb[2], (size_t) ctrl * messages->nb[1]);
        };

        ggml_tensor * inputs[3] = { cur, cur, cur };
        const bool same_qkv = L.is_swa && (messages == nullptr);
        if (messages == nullptr) {
            if (L.attn_res_query == nullptr) {
                inputs[0] = rms_norm_w(c, cur, L.attn_norm, w.rms_eps);
                inputs[1] = inputs[2] = inputs[0];
            } else if (L.is_swa) {
                inputs[0] = ggml_inkling2_attn_res_mix(
                    c, cur, L.attn_res_query, L.attn_norm,
                    blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
                inputs[1] = inputs[2] = inputs[0];
            } else {
                ggml_tensor * m3 = ggml_inkling2_attn_res_mix(
                    c, cur, L.attn_res_query, L.attn_qkv_norm,
                    blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
                for (int j = 0; j < 3; ++j)
                    inputs[j] = ggml_view_2d(c, m3, w.n_embd, n_tokens, m3->nb[2], (size_t) j * m3->nb[1]);
            }
        } else {
            ggml_tensor * pre_qkv = cur;
            if (L.attn_res_query) {
                ggml_tensor * mx = ggml_inkling2_attn_res_mix(
                    c, cur, L.attn_res_query, nullptr,
                    blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
                pre_qkv = L.is_swa ? mx : ggml_permute(c, mx, 0, 2, 1, 3);
            }
            ggml_tensor * mqv = ggml_view_3d(c, messages, w.n_embd, 3, n_tokens, messages->nb[1], messages->nb[2], 0);
            ggml_tensor * mq  = ggml_permute(c, mqv, 0, 2, 1, 3);
            ggml_tensor * nw  = L.is_swa ? L.attn_norm : L.attn_qkv_norm;
            ggml_tensor * gt  = ggml_mul_mat(c, loop->ctrl_down_qkv, ggml_inkling2_loop_ctrl_pre(c, pre_qkv, mq, w.rms_eps));
            ggml_tensor * gfo = ggml_mul_mat(c, loop->ctrl_gf_qkv, gt);
            ggml_tensor * qi  = ggml_inkling2_loop_ctrl_post(c, pre_qkv, mq, gfo, nw, w.rms_eps);
            for (int j = 0; j < 3; ++j)
                inputs[j] = ggml_view_2d(c, qi, w.n_embd, n_tokens, qi->nb[1], (size_t) j * qi->nb[2]);
        }

        ggml_tensor * Q_raw, * K_raw, * V_raw, * Gate, * R_raw, * Sink;
        ggml_tensor * k_dyn = nullptr, * v_dyn = nullptr;
        if (same_qkv && L.wqkvgrs) {
            ggml_tensor * proj = ggml_mul_mat(c, L.wqkvgrs, inputs[0]);
            Q_raw = row_view_2d(c, proj, 0, q_sz);
            Gate  = row_view_2d(c, proj, q_sz, q_sz);
            R_raw = row_view_2d(c, proj, 2 * q_sz, r_sz);
            Sink  = row_view_2d(c, proj, 2 * q_sz + r_sz, Hq);
            K_raw = row_view_2d(c, proj, 2 * q_sz + r_sz + Hq, kv_sz);
            V_raw = row_view_2d(c, proj, 2 * q_sz + r_sz + Hq + kv_sz, kv_sz);
        } else {
            if (L.wqgrs) {
                ggml_tensor * proj = ggml_mul_mat(c, L.wqgrs, inputs[0]);
                Q_raw = row_view_2d(c, proj, 0, q_sz);
                Gate  = row_view_2d(c, proj, q_sz, q_sz);
                R_raw = row_view_2d(c, proj, 2 * q_sz, r_sz);
                Sink  = row_view_2d(c, proj, 2 * q_sz + r_sz, Hq);
            } else {
                Q_raw = ggml_mul_mat(c, L.wq, inputs[0]);
                Gate  = ggml_mul_mat(c, L.wqkv_gate, inputs[0]);
                R_raw = ggml_mul_mat(c, L.attn_rel, inputs[0]);
                Sink  = ggml_mul_mat(c, L.attn_sink_proj, inputs[0]);
            }
            if (!L.is_swa && L.wk_dyn && L.wv_dyn) {
                ggml_tensor * kd = ggml_mul_mat(c, L.wk_dyn, inputs[1]);
                K_raw = row_view_2d(c, kd, 0, kv_sz);
                k_dyn = row_view_2d(c, kd, kv_sz, (int64_t) w.sconv_kernel * Hk);
                ggml_tensor * vd = ggml_mul_mat(c, L.wv_dyn, inputs[2]);
                V_raw = row_view_2d(c, vd, 0, kv_sz);
                v_dyn = row_view_2d(c, vd, kv_sz, (int64_t) w.sconv_kernel * Hk);
            } else {
                K_raw = ggml_mul_mat(c, L.wk, inputs[1]);
                V_raw = ggml_mul_mat(c, L.wv, inputs[2]);
                if (!L.is_swa) {
                    k_dyn = ggml_mul_mat(c, L.attn_k_conv_proj, inputs[1]);
                    v_dyn = ggml_mul_mat(c, L.attn_v_conv_proj, inputs[2]);
                }
            }
        }

        const int swa = L.is_swa ? w.swa_window : 0;
        ggml_tensor * Q = nullptr, * rt = nullptr;
        make_clip_qr_rel_table(
            c, Q_raw, L.attn_q_clip, R_raw, L.attn_rel_clip,
            L.is_swa ? nullptr : inp_pos, L.attn_rel_bias, Sink,
            D, w.d_rel, Hq, w.qk_clip, w.rel_clip, w.rms_eps,
            L.is_swa ? 0 : w.log_scale_floor, L.is_swa ? 0.0f : w.log_scale_alpha,
            &Q, &rt);

        if (!L.is_swa && q_cap_out && plane >= 0 && plane < 4) {
            ggml_tensor * q_tail = ggml_view_3d(
                c, Q, D, Hq, q_cap_n, Q->nb[1], Q->nb[2],
                (size_t) (n_tokens - q_cap_n) * Q->nb[2]);
            const bool clone_coda_q = (share_coda && ip >= w.core_end && plane == w.n_passes - 1);
            const int p_beg = clone_coda_q ? 0 : plane;
            const int p_end = clone_coda_q ? tc.planes[ip] : (plane + 1);
            for (int tp = p_beg; tp < p_end && tp < 4; ++tp) {
                const int gidx = global_plane_idx[(size_t) ip][tp];
                if (gidx >= 0) {
                    ggml_tensor * q_dst = ggml_view_3d(
                        c, q_cap_out, D, Hq, q_cap_n, q_cap_out->nb[1], q_cap_out->nb[3],
                        (size_t) gidx * q_cap_out->nb[2]);
                    ggml_build_forward_expand(gf, ggml_cpy(c, q_tail, q_dst));
                }
            }
        }

        ggml_tensor * kd_dep = nullptr, * vd_dep = nullptr;
        if (shared_global && plane < tc.planes[ip] - 1) {
            const size_t tail_off = (size_t) (tc.global_pool_cap + plane * tc.shared_tail_cap) * tc.prefix_k[ip]->nb[1];
            ggml_tensor * k_tail = ggml_view_3d(
                c, tc.prefix_k[ip], tc.prefix_k[ip]->ne[0], tc.shared_tail_cap, tc.prefix_k[ip]->ne[2],
                tc.prefix_k[ip]->nb[1], tc.prefix_k[ip]->nb[2], tail_off);
            ggml_tensor * v_tail = ggml_view_3d(
                c, tc.prefix_v[ip], tc.prefix_v[ip]->ne[0], tc.shared_tail_cap, tc.prefix_v[ip]->ne[2],
                tc.prefix_v[ip]->nb[1], tc.prefix_v[ip]->nb[2], tail_off);
            ggml_tensor * kc = ggml_inkling2_sconv(c, K_raw, L.attn_k_conv, st, st, k_dyn, 0, D, 1);
            ggml_tensor * vc = ggml_inkling2_sconv(c, V_raw, L.attn_v_conv, st, st, v_dyn, kv_sz, D, 1);
            if (use_paged) {
                kd_dep = ggml_inkling2_kv_write_paged(c, kc, k_p, inp_pos, inp_bt_local, kf_bs, D, Hk, 0, w.qk_clip, w.rms_eps);
                vd_dep = ggml_inkling2_kv_write_paged(c, vc, v_p, inp_pos, inp_bt_local, kf_bs, D, Hk, 1, w.qk_clip, w.rms_eps);
            } else {
                kd_dep = ggml_inkling2_kv_write(c, kc, k_p, inp_pos, D, Hk, 0, 0, w.qk_clip, w.rms_eps);
                vd_dep = ggml_inkling2_kv_write(c, vc, v_p, inp_pos, D, Hk, 1, 0, w.qk_clip, w.rms_eps);
            }
            const int m_tail    = std::min(n_tokens, tc.shared_tail_cap);
            const int tail_skip = n_tokens - m_tail;
            ggml_tensor * kc_tail  = (tail_skip > 0)
                ? ggml_view_2d(c, kc, kc->ne[0], m_tail, kc->nb[1], (size_t) tail_skip * kc->nb[1])
                : kc;
            ggml_tensor * vc_tail  = (tail_skip > 0)
                ? ggml_view_2d(c, vc, vc->ne[0], m_tail, vc->nb[1], (size_t) tail_skip * vc->nb[1])
                : vc;
            ggml_tensor * pos_tail = (tail_skip > 0)
                ? ggml_view_1d(c, inp_pos, m_tail, (size_t) tail_skip * inp_pos->nb[0])
                : inp_pos;
            ggml_tensor * kt_dep = ggml_inkling2_kv_write(c, kc_tail, k_tail, pos_tail, D, Hk, 0, tc.shared_tail_cap, w.qk_clip, w.rms_eps);
            ggml_tensor * vt_dep = ggml_inkling2_kv_write(c, vc_tail, v_tail, pos_tail, D, Hk, 1, tc.shared_tail_cap, w.qk_clip, w.rms_eps);
            ggml_build_forward_expand(gf, kt_dep);
            ggml_build_forward_expand(gf, vt_dep);
        } else if (!L.is_swa && use_paged) {
            kd_dep = ggml_inkling2_sconv_kv_write_paged(
                c, K_raw, L.attn_k_conv, st, st, k_dyn, k_p, inp_pos,
                inp_bt_local, kf_bs, 0, D, Hk, 0, w.qk_clip, w.rms_eps, 1);
            vd_dep = ggml_inkling2_sconv_kv_write_paged(
                c, V_raw, L.attn_v_conv, st, st, v_dyn, v_p, inp_pos,
                inp_bt_local, kf_bs, kv_sz, D, Hk, 1, w.qk_clip, w.rms_eps, 1);
        } else {
            kd_dep = ggml_inkling2_sconv_kv_write(
                c, K_raw, L.attn_k_conv, st, st, k_dyn, k_p, inp_pos,
                0, D, Hk, 0, swa, w.qk_clip, w.rms_eps, 1);
            vd_dep = ggml_inkling2_sconv_kv_write(
                c, V_raw, L.attn_v_conv, st, st, v_dyn, v_p, inp_pos,
                kv_sz, D, Hk, 1, swa, w.qk_clip, w.rms_eps, 1);
        }

        ggml_tensor * ao = nullptr;
        if (!L.is_swa && use_paged) {
            if (use_tier15 && ip < (int) tc.remote_prefix_k.size() && tc.remote_prefix_k[ip]) {
                ggml_tensor * rk_p = plane_view_kv(c, tc.remote_prefix_k[ip], kv_plane);
                ggml_tensor * rv_p = plane_view_kv(c, tc.remote_prefix_v[ip], kv_plane);
                ggml_tensor * part_r = ggml_inkling2_attn_rel_paged(
                    c, Q, rk_p, rv_p, rt, inp_meta_r, nullptr, nullptr, nullptr,
                    inp_bt_remote, kf_bs, w.rel_extent, /*partial=*/1, /*include_sink=*/1);
                expert_nodes.push_back(part_r);
                ggml_tensor * part_l = ggml_inkling2_attn_rel_paged(
                    c, Q, k_p, v_p, rt, inp_meta, kd_dep, vd_dep, nullptr,
                    inp_bt_local, kf_bs, w.rel_extent, /*partial=*/1, /*include_sink=*/0);
                ao = ggml_inkling2_attn_split_combine(c, part_r, part_l, Gate, D, Hq);
            } else {
                ao = ggml_inkling2_attn_rel_paged(
                    c, Q, k_p, v_p, rt, inp_meta, kd_dep, vd_dep, Gate,
                    inp_bt_local, kf_bs, w.rel_extent, /*partial=*/0, /*include_sink=*/1);
            }
        } else {
            ao = ggml_inkling2_attn_rel(
                c, Q, k_p, v_p, rt, inp_meta, kd_dep, vd_dep, Gate, swa, w.rel_extent, 0, 1);
        }

        cur = ggml_mul_mat(c, L.wo, ao);
        if (ip % w.attn_res_block == 0) {
            cur = ggml_inkling2_sconv(c, cur, L.attn_post_conv, st, st, nullptr, 2 * kv_sz, 0, 1);
            blocks.push_back(inpSA);
        } else {
            cur = ggml_inkling2_sconv_add(c, cur, L.attn_post_conv, st, st, nullptr, inpSA, 2 * kv_sz, 0, 1);
        }
        ggml_tensor * ffn_inp = cur;
        if (messages == nullptr) {
            cur = ggml_inkling2_attn_res_mix(
                c, cur, L.ffn_res_query, L.ffn_norm,
                blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
        } else {
            cur = ggml_inkling2_attn_res_mix(
                c, cur, L.ffn_res_query, nullptr,
                blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
            ggml_tensor * mf   = get_msg(Inkling2LoopLayer::CTRL_FFN);
            ggml_tensor * cd_w = loop->ctrl_down_f16[Inkling2LoopLayer::CTRL_FFN]
                ? loop->ctrl_down_f16[Inkling2LoopLayer::CTRL_FFN]
                : loop->ctrl_down[Inkling2LoopLayer::CTRL_FFN];
            ggml_tensor * gt  = ggml_mul_mat(c, cd_w, ggml_inkling2_loop_ctrl_pre(c, cur, mf, w.rms_eps));
            ggml_tensor * gfo = ggml_mul_mat(c, loop->ctrl_gf[Inkling2LoopLayer::CTRL_FFN], gt);
            cur = ggml_inkling2_loop_ctrl_post(c, cur, mf, gfo, L.ffn_norm, w.rms_eps);
        }

        ggml_tensor * rl   = ggml_mul_mat(c, L.ffn_gate_inp_q8 ? L.ffn_gate_inp_q8 : L.ffn_gate_inp, cur);
        ggml_tensor * sel  = ggml_inkling2_sigmoid_topk_ids(c, rl, L.ffn_exp_probs_b, w.n_expert_used);
        ggml_tensor * wsel = ggml_inkling2_sigmoid_topk_wts(c, rl, sel, w.route_scale, 1e-20f);
        ggml_tensor * x3   = ggml_reshape_3d(c, cur, w.n_embd, 1, n_tokens);
        ggml_tensor * gu   = ggml_mul_mat_id(c, L.ffn_gate_up_exps, x3, sel);
        ggml_tensor * act  = ggml_inkling2_situ(c, gu, w.situ_beta, w.situ_linear_beta);
        ggml_tensor * dn   = ggml_mul_mat_id(c, L.ffn_down_exps, act, sel);
        cur = ggml_ds4_moe_fused_combine_shared(c, dn, wsel, nullptr);
        cur = ggml_inkling2_sconv_add(c, cur, L.ffn_post_conv, st, st, nullptr, ffn_inp, 2 * kv_sz + w.n_embd, 0, 1);
        return cur;
    };

    // 1. Prelude (L0..5, plane 0)
    ggml_tensor * cur = rms_norm_w(c, ggml_get_rows(c, w.tok_embd, inp_tok), w.tok_norm, w.rms_eps);
    std::vector<ggml_tensor *> prelude_blocks;
    ggml_tensor * tap_l2 = nullptr;
    for (int ip = 0; ip < w.core_start; ++ip) {
        if (ip == w.twodsd.feature_layers[0]) tap_l2 = cur;
        cur = run_prefill_layer(ip, 0, cur, prelude_blocks, nullptr, nullptr);
    }
    ggml_tensor * prelude_cur = cur;

    // 2. For each pass p in 0..n_passes-1: run Core (L6..17, plane p).
    // When prefill_share_coda_enabled() (default ON): run Coda (L18..23) ONLY on final depth-3
    // pass (48 layers = exact AR cost!) and clone depth-3 prefix KV/conv -> planes 0 & 1 (~40 us!).
    std::vector<ggml_tensor *> bank;
    ggml_tensor * bank_out = nullptr;
    ggml_tensor * loop3_taps[5] = { tap_l2, nullptr, nullptr, nullptr, nullptr };
    ggml_tensor * final_out_norm = nullptr;
    ggml_tensor * prev_coda_out  = nullptr;

    for (int pass = 0; pass < w.n_passes; ++pass) {
        std::vector<ggml_tensor *> blocks = prelude_blocks;
        ggml_tensor * p_cur = prelude_cur;

        for (int ip = w.core_start; ip < w.core_end; ++ip) {
            if (pass == w.n_passes - 1) {
                for (int ti = 0; ti < 3; ++ti) {
                    if (ip == w.twodsd.feature_layers[1 + ti]) loop3_taps[1 + ti] = p_cur;
                }
            }
            p_cur = run_prefill_layer(ip, pass, p_cur, blocks, (pass > 0) ? &bank : nullptr, (pass > 0) ? bank_out : nullptr);
        }

        // Update bank for next pass BEFORE coda modifies `blocks`
        std::vector<ggml_tensor *> next_bank(blocks.begin() + (ptrdiff_t) prelude_blocks.size() + 1, blocks.end());
        ggml_tensor * next_bank_out = p_cur;

        if (!share_coda || pass == w.n_passes - 1) {
            std::vector<ggml_tensor *> coda_blocks = blocks;
            ggml_tensor * c_cur = p_cur;
            if (prev_coda_out && tc.shared_kv_window > 0 && use_tier15) {
                ggml_tensor * z = ggml_scale(c, ggml_view_2d(c, prev_coda_out, 1, 1, prev_coda_out->nb[1], 0), 0.0f);
                c_cur = ggml_add(c, c_cur, z);
            }
            for (int ip = w.core_end; ip < w.n_layer_phys; ++ip) {
                if (pass == w.n_passes - 1 && ip == w.twodsd.feature_layers[4]) {
                    loop3_taps[4] = c_cur;
                }
                c_cur = run_prefill_layer(ip, pass, c_cur, coda_blocks, nullptr, nullptr);
                if (share_coda && pass == w.n_passes - 1) {
                    const bool is_swa = w.layers[ip].is_swa;
                    const int swa = is_swa ? w.swa_window : 0;
                    const bool sh_g = (!is_swa && tc.shared_kv_window > 0 && tc.planes[ip] > 1);
                    ggml_build_forward_expand(gf, make_tree_prefill_clone_coda(
                        c, tc.prefix_k[ip], tc.prefix_v[ip], tc.prefix_conv[ip], c_cur, pos0, n_tokens, swa,
                        (!is_swa && use_paged) ? inp_bt_local : nullptr,
                        (!is_swa && use_paged) ? kf_bs : 0,
                        sh_g ? tc.shared_kv_window : 0,
                        sh_g ? tc.shared_tail_cap : 0));
                }
            }
            prev_coda_out = c_cur;
            if (pass == w.n_passes - 1) {
                final_out_norm = ggml_inkling2_attn_res_mix(
                    c, c_cur, w.output_res_query, w.output_norm,
                    coda_blocks.empty() ? nullptr : coda_blocks.data(), (int) coda_blocks.size(), w.rms_eps);
            } else {
                ggml_build_forward_expand(gf, c_cur);
            }
        }

        bank = std::move(next_bank);
        bank_out = next_bank_out;
    }

    // Fused in-graph tail-only drafter prefix KV projection (zero extra gallocr / copy / sync!)
    const int tail_beg = std::max(0, tc.draft_pref_start - pos0);
    const int tail_n   = std::max(0, n_tokens - tail_beg);
    ggml_tensor * inp_dr_pos  = nullptr;
    ggml_tensor * inp_dr_wpos = nullptr;
    if (tail_n > 0) {
        auto slice_tail = [&](ggml_tensor * t) -> ggml_tensor * {
            if (tail_beg == 0 && tail_n == n_tokens) return t;
            return ggml_view_2d(c, t, t->ne[0], tail_n, t->nb[1], (size_t) tail_beg * t->nb[1]);
        };
        ggml_tensor * f01 = ggml_concat(c, slice_tail(loop3_taps[0]), slice_tail(loop3_taps[1]), 0);
        ggml_tensor * f23 = ggml_concat(c, slice_tail(loop3_taps[2]), slice_tail(loop3_taps[3]), 0);
        ggml_tensor * feats_tail = ggml_concat(c, ggml_concat(c, f01, f23, 0), slice_tail(loop3_taps[4]), 0);
        inp_dr_pos  = ggml_new_tensor_1d(c, GGML_TYPE_I32, tail_n);
        inp_dr_wpos = ggml_new_tensor_1d(c, GGML_TYPE_I32, tail_n);
        ggml_set_input(inp_dr_pos);
        ggml_set_input(inp_dr_wpos);
        inkling2_2dsd_append_prefix_kv_graph(c, gf, w.twodsd, tc, feats_tail, inp_dr_pos, inp_dr_wpos);
    }

    ggml_tensor * greedy_t = nullptr;
    ggml_tensor * soft_t   = nullptr;
    if (is_last_chunk) {
        ggml_tensor * lm_in = ggml_view_2d(c, final_out_norm, w.n_embd, 1, final_out_norm->nb[1],
                                           (size_t) (n_tokens - 1) * final_out_norm->nb[1]);
        ggml_tensor * raw_l = ggml_mul_mat(c, w.output, lm_in);
        soft_t  = ggml_inkling2_softcap_slice(c, raw_l, w.n_vocab_unpadded, w.logit_softcap);
        if (out_last_logits) ggml_set_output(soft_t);
        greedy_t = ggml_argmax(c, soft_t);
        ggml_set_output(greedy_t);
        ggml_build_forward_expand(gf, greedy_t);
    } else {
        ggml_build_forward_expand(gf, final_out_norm);
    }

    const bool need_sched = use_tier15 && !expert_nodes.empty() && w.expert_backend && cache.split_cpu;
    ggml_backend_sched_t sched = nullptr;
    ggml_gallocr_t ga = nullptr;
    if (need_sched) {
        ggml_backend_t backends[3] = { target_backend, w.expert_backend, cache.split_cpu };
        sched = ggml_backend_sched_new(backends, nullptr, 3, max_nodes, true, false);
        if (!sched) { ggml_free(c); return false; }
        ggml_backend_sched_set_late_cross_input_split(sched, true);
        ggml_backend_sched_reset(sched);
        const std::unordered_set<const ggml_tensor *> on_exp(expert_nodes.begin(), expert_nodes.end());
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            for (ggml_tensor * src : node->src) {
                for (ggml_tensor * t = src; t; t = t->view_src) {
                    if ((t->flags & GGML_TENSOR_FLAG_INPUT) && !t->buffer) {
                        ggml_backend_t pin_be = (t == inp_meta_r || t == inp_bt_remote) ? w.expert_backend : target_backend;
                        ggml_backend_sched_set_tensor_backend(sched, t, pin_be);
                    }
                }
            }
            const bool is_view = node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW ||
                                 node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE ||
                                 node->op == GGML_OP_TRANSPOSE;
            if (on_exp.count(node)) {
                ggml_backend_sched_set_tensor_backend(sched, node, w.expert_backend);
            } else if (!is_view && ggml_backend_supports_op(target_backend, node)) {
                ggml_backend_sched_set_tensor_backend(sched, node, target_backend);
            }
        }
        if (q_cap_out && !q_cap_out->buffer) {
            ggml_backend_sched_set_tensor_backend(sched, q_cap_out, target_backend);
        }
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            ggml_backend_sched_free(sched);
            ggml_free(c);
            return false;
        }
    } else {
        ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
        if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) {
            if (ga) ggml_gallocr_free(ga);
            ggml_free(c);
            return false;
        }
    }

    std::vector<int32_t> pos(n_tokens);
    for (int i = 0; i < n_tokens; ++i) pos[i] = pos0 + i;
    int32_t meta[4] = { pos0, pos0 + n_tokens, 0, pos0 + n_tokens };
    ggml_backend_tensor_set(inp_tok,  tokens,     0, (size_t) n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(inp_pos,  pos.data(), 0, (size_t) n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(inp_meta, meta,       0, sizeof(meta));
    if (use_paged && inp_bt_local) {
        const auto & btl = cache.kvflash->bt_local();
        ggml_backend_tensor_set(inp_bt_local, btl.data(), 0, btl.size() * sizeof(int32_t));
        if (use_tier15 && inp_meta_r && inp_bt_remote) {
            const auto & btr = cache.kvflash->bt_remote();
            ggml_backend_tensor_set(inp_meta_r,    meta,       0, sizeof(meta));
            ggml_backend_tensor_set(inp_bt_remote, btr.data(), 0, btr.size() * sizeof(int32_t));
        }
    }
    if (tail_n > 0) {
        std::vector<int32_t> dr_pos(tail_n), dr_wpos(tail_n);
        for (int i = 0; i < tail_n; ++i) {
            const int raw_wpos = (pos0 + tail_beg - tc.draft_pref_start) + i;
            dr_pos[i]  = pos0 + tail_beg + i;
            dr_wpos[i] = (tc.draft_ring_cap > 0) ? (raw_wpos % tc.draft_ring_cap) : raw_wpos;
        }
        ggml_backend_tensor_set(inp_dr_pos,  dr_pos.data(),  0, (size_t) tail_n * sizeof(int32_t));
        ggml_backend_tensor_set(inp_dr_wpos, dr_wpos.data(), 0, (size_t) tail_n * sizeof(int32_t));
    }

    const bool ok = sched
        ? (ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS)
        : (ggml_backend_graph_compute(target_backend, gf) == GGML_STATUS_SUCCESS);
    if (ok) {
        if (use_paged && q_cap_out && q_cap_n > 0) {
            std::vector<float> q_host((size_t) ggml_nelements(q_cap_out));
            ggml_backend_tensor_get(q_cap_out, q_host.data(), 0, ggml_nbytes(q_cap_out));
            cache.kvflash->append_queries(q_host.data(), q_cap_n);
        }
        if (is_last_chunk && greedy_t) {
            ggml_backend_tensor_get(greedy_t, &out_greedy_tok, 0, sizeof(int32_t));
        }
        if (is_last_chunk && out_last_logits && soft_t) {
            out_last_logits->resize((size_t) w.n_vocab_unpadded);
            ggml_backend_tensor_get(soft_t, out_last_logits->data(), 0, (size_t) w.n_vocab_unpadded * sizeof(float));
        }
        tc.prefix_len = pos0 + n_tokens;
        cache.cur_pos = pos0 + n_tokens;
        if (tail_n > 0) cache.dspark_ctx_pos = pos0 + n_tokens;
    }
    if (sched) ggml_backend_sched_free(sched);
    if (ga)    ggml_gallocr_free(ga);
    ggml_free(c);
    return ok;
}

// ─── 3-Pass Wavefront 60-Plane Prefill (60 -> 24 MoE Weight Reads / Chunk!) ─
bool inkling2_2dsd_prefill_wavefront(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        const int32_t * tokens,
        int n_tokens,
        int pos0,
        int chunk_size,
        int32_t & out_greedy_tok,
        std::vector<float> * out_last_logits,
        int capture_q_rows,
        bool is_last_chunk) {
    if (!w.twodsd.loaded || !cache.tree.active || n_tokens <= 0) return false;
    auto & tc = cache.tree;
    const bool use_paged = (cache.kvflash && cache.kvflash->active());
    const int  kf_bs     = use_paged ? cache.kvflash->block_size() : 0;

    // Shared-KV shares plane 0 across passes; use sequential chunk prefill so passes execute strictly 0 -> 1 -> 2.
    if (use_paged) {
        bool fallback_chunk = (tc.shared_kv_window > 0);
        if (!fallback_chunk) {
            // Ensure all 3 in-flight wavefront chunks fit comfortably inside Tier-0 non-sink pool
            // so every block evicted by prepare_span(chunks[step]) is already 100% finished on all 10 global planes!
            const int avail = cache.kvflash->tier0_pool_tokens() - cache.kvflash->config().sink_chunks * kf_bs;
            const int max_wf_chunk = ((avail / 3) / kf_bs) * kf_bs;
            if (max_wf_chunk < 256) {
                fallback_chunk = true;
            } else {
                chunk_size = std::min(chunk_size, max_wf_chunk);
            }
        }
        if (fallback_chunk) {
            int off = 0;
            while (off < n_tokens) {
                const int len = std::min(chunk_size, n_tokens - off);
                const bool last = (off + len == n_tokens);
                const int cap_q = (last && capture_q_rows > 0) ? capture_q_rows : 0;
                if (!inkling2_2dsd_prefill_chunk(target_backend, w, cache, tokens + off, len, pos0 + off,
                                                 last && is_last_chunk, out_greedy_tok,
                                                 (last && is_last_chunk) ? out_last_logits : nullptr, cap_q)) {
                    return false;
                }
                off += len;
            }
            return true;
        }
    }

    if (n_tokens <= chunk_size || !w.is_loop || w.n_passes != 3) {
        return inkling2_2dsd_prefill_chunk(target_backend, w, cache, tokens, n_tokens, pos0,
                                           is_last_chunk, out_greedy_tok,
                                           is_last_chunk ? out_last_logits : nullptr, capture_q_rows);
    }

    const int Hq = w.n_head, Hk = w.n_head_kv, D = w.head_dim;
    const int q_sz = Hq * D, kv_sz = Hk * D, r_sz = w.d_rel * Hq;
    const int blk_div = std::max(1, w.attn_res_block);
    const int max_blocks_core = (w.core_end / blk_div) + 2;
    const int max_bank = ((w.core_end - w.core_start) / blk_div) + 2;

    struct ChunkInfo { int off; int len; int pos; };
    std::vector<ChunkInfo> chunks;
    for (int off = 0; off < n_tokens; off += chunk_size) {
        const int len = std::min(chunk_size, n_tokens - off);
        chunks.push_back({ off, len, pos0 + off });
    }
    const int M = (int) chunks.size();
    const bool share_coda = prefill_share_coda_enabled();

    // Optional Mode-4 query tail capture across the last chunk's global layers
    std::vector<std::array<int, 4>> global_plane_idx((size_t) w.n_layer_phys, {-1, -1, -1, -1});
    int n_global_planes = 0;
    if (use_paged) {
        for (int ip = 0; ip < w.n_layer_phys; ++ip) {
            if (w.layers[ip].is_swa) continue;
            const int p_st = (int) tc.prefix_k[ip]->ne[3];
            const int P    = tc.planes[ip];
            if (p_st == 1 && P > 1) {
                global_plane_idx[(size_t) ip][P - 1] = n_global_planes++;
            } else {
                for (int p = 0; p < P && p < 4; ++p) {
                    global_plane_idx[(size_t) ip][p] = n_global_planes++;
                }
            }
        }
    }
    const int q_cap_n = (use_paged && capture_q_rows > 0 && n_global_planes > 0)
        ? std::min(capture_q_rows, chunks[M - 1].len) : 0;
    std::vector<float> q_cap_host;
    if (q_cap_n > 0) {
        q_cap_host.assign((size_t) q_cap_n * n_global_planes * Hq * D, 0.0f);
    }
    auto scatter_q_cap = [&](ggml_tensor * t_qcap, int gidx) {
        if (!t_qcap || gidx < 0 || q_cap_n <= 0) return;
        const size_t row_elems = (size_t) Hq * D;
        std::vector<float> tmp((size_t) q_cap_n * row_elems);
        ggml_backend_tensor_get(t_qcap, tmp.data(), 0, tmp.size() * sizeof(float));
        for (int q = 0; q < q_cap_n; ++q) {
            float * dst = q_cap_host.data() + ((size_t) q * n_global_planes + gidx) * row_elems;
            std::memcpy(dst, tmp.data() + (size_t) q * row_elems, row_elems * sizeof(float));
        }
    };

    // Helper: allocate a stage graph with pure ggml_gallocr when single-GPU, or ggml_backend_sched when Tier 1.5 active!
    auto alloc_stage_graph = [&](ggml_cgraph * gf, size_t max_nodes, bool stage_t15,
                                 const std::vector<ggml_tensor *> & exp_nodes,
                                 const std::unordered_set<const ggml_tensor *> & exp_inputs,
                                 ggml_backend_sched_t & out_sched,
                                 ggml_gallocr_t & out_ga) -> bool {
        out_sched = nullptr;
        out_ga    = nullptr;
        const bool need_sched = stage_t15 && !exp_nodes.empty() && w.expert_backend && cache.split_cpu;
        if (need_sched) {
            ggml_backend_t backends[3] = { target_backend, w.expert_backend, cache.split_cpu };
            out_sched = ggml_backend_sched_new(backends, nullptr, 3, max_nodes, true, false);
            if (!out_sched) return false;
            ggml_backend_sched_set_late_cross_input_split(out_sched, true);
            ggml_backend_sched_reset(out_sched);
            const std::unordered_set<const ggml_tensor *> on_exp(exp_nodes.begin(), exp_nodes.end());
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                ggml_tensor * node = ggml_graph_node(gf, i);
                for (ggml_tensor * src : node->src) {
                    for (ggml_tensor * t = src; t; t = t->view_src) {
                        if ((t->flags & GGML_TENSOR_FLAG_INPUT) && !t->buffer) {
                            ggml_backend_t pin_be = exp_inputs.count(t) ? w.expert_backend : target_backend;
                            ggml_backend_sched_set_tensor_backend(out_sched, t, pin_be);
                        }
                    }
                }
                const bool is_view = node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW ||
                                     node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE ||
                                     node->op == GGML_OP_TRANSPOSE;
                if (on_exp.count(node)) {
                    ggml_backend_sched_set_tensor_backend(out_sched, node, w.expert_backend);
                } else if (!is_view && ggml_backend_supports_op(target_backend, node)) {
                    ggml_backend_sched_set_tensor_backend(out_sched, node, target_backend);
                }
            }
            if (!ggml_backend_sched_alloc_graph(out_sched, gf)) {
                ggml_backend_sched_free(out_sched);
                out_sched = nullptr;
                return false;
            }
            return true;
        }
        out_ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
        if (!out_ga || !ggml_gallocr_alloc_graph(out_ga, gf)) {
            if (out_ga) { ggml_gallocr_free(out_ga); out_ga = nullptr; }
            return false;
        }
        return true;
    };

    struct ChunkState {
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        ggml_tensor * prelude_cur = nullptr;
        ggml_tensor * core_cur    = nullptr;
        ggml_tensor * bank_out    = nullptr;
        ggml_tensor * tap_l2      = nullptr;
        ggml_tensor * core_taps[3] = {};
        std::vector<ggml_tensor *> blocks;
        std::vector<ggml_tensor *> bank;
        int len = 0;
        int n_prelude_blocks = 0;
        int n_blocks = 0;
        int n_bank = 0;
    };
    ChunkState ring[3];

    auto free_slot = [&](int s) {
        if (ring[s].buf) { ggml_backend_buffer_free(ring[s].buf); ring[s].buf = nullptr; }
        if (ring[s].ctx) { ggml_free(ring[s].ctx); ring[s].ctx = nullptr; }
        ring[s].blocks.clear();
        ring[s].bank.clear();
        ring[s].len = 0;
    };
    auto free_ring = [&]() { for (int s = 0; s < 3; ++s) free_slot(s); };
    auto alloc_slot = [&](int s, int len) -> bool {
        if (ring[s].len == len && ring[s].buf) return true;
        free_slot(s);
        auto & cs = ring[s];
        const size_t n_t = (size_t) (max_blocks_core + max_bank + 10);
        struct ggml_init_params gp = { n_t * ggml_tensor_overhead() + 4096, nullptr, true };
        cs.ctx = ggml_init(gp);
        if (!cs.ctx) return false;
        cs.prelude_cur = ggml_new_tensor_2d(cs.ctx, GGML_TYPE_F32, w.n_embd, len);
        cs.core_cur    = ggml_new_tensor_2d(cs.ctx, GGML_TYPE_F32, w.n_embd, len);
        cs.bank_out    = ggml_new_tensor_2d(cs.ctx, GGML_TYPE_F32, w.n_embd, len);
        cs.tap_l2      = ggml_new_tensor_2d(cs.ctx, GGML_TYPE_F32, w.n_embd, len);
        for (int ti = 0; ti < 3; ++ti) cs.core_taps[ti] = ggml_new_tensor_2d(cs.ctx, GGML_TYPE_F32, w.n_embd, len);
        cs.blocks.resize(max_blocks_core, nullptr);
        for (int b = 0; b < max_blocks_core; ++b) cs.blocks[b] = ggml_new_tensor_2d(cs.ctx, GGML_TYPE_F32, w.n_embd, len);
        cs.bank.resize(max_bank, nullptr);
        for (int b = 0; b < max_bank; ++b) cs.bank[b] = ggml_new_tensor_2d(cs.ctx, GGML_TYPE_F32, w.n_embd, len);
        cs.buf = ggml_backend_alloc_ctx_tensors(cs.ctx, target_backend);
        cs.len = len;
        return cs.buf != nullptr;
    };

    // Wavefront schedule: step = 0 .. M + 1
    for (int step = 0; step <= M + 1; ++step) {
        if (step < M && use_paged) {
            if (!cache.kvflash->prepare_span(chunks[step].pos, chunks[step].len)) {
                free_ring();
                return false;
            }
        }
        const bool stage_t15 = (use_paged && cache.kvflash->tier15_active() &&
                                cache.kvflash->tier15_resident_blocks() > 0);

        // ── Stage 1: Prelude (L0..5, plane 0) for chunk `step` ──────────
        if (step < M) {
            const auto & ci = chunks[step];
            const int s_slot = step % 3;
            if (!alloc_slot(s_slot, ci.len)) { free_ring(); return false; }
            auto & cs = ring[s_slot];

            const size_t max_nodes = 4096;
            struct ggml_init_params gp = { max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false) + (4 << 20), nullptr, true };
            ggml_context * c = ggml_init(gp);
            if (!c) { free_ring(); return false; }
            ggml_cgraph * gf = ggml_new_graph_custom(c, max_nodes, false);

            ggml_tensor * inp_tok       = ggml_new_tensor_1d(c, GGML_TYPE_I32, ci.len);
            ggml_tensor * inp_pos       = ggml_new_tensor_1d(c, GGML_TYPE_I32, ci.len);
            ggml_tensor * inp_meta      = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
            ggml_tensor * inp_meta_r    = nullptr;
            ggml_tensor * inp_bt_local  = nullptr;
            ggml_tensor * inp_bt_remote = nullptr;
            std::vector<ggml_tensor *> exp_nodes;
            std::unordered_set<const ggml_tensor *> exp_inputs;
            std::vector<std::pair<ggml_tensor *, int>> qcap_list;
            ggml_set_input(inp_tok);
            ggml_set_input(inp_pos);
            ggml_set_input(inp_meta);
            if (use_paged) {
                inp_bt_local = ggml_new_tensor_1d(c, GGML_TYPE_I32, cache.kvflash->max_blocks());
                ggml_set_input(inp_bt_local);
                if (stage_t15) {
                    inp_meta_r    = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
                    inp_bt_remote = ggml_new_tensor_1d(c, GGML_TYPE_I32, cache.kvflash->max_blocks());
                    ggml_set_input(inp_meta_r);
                    ggml_set_input(inp_bt_remote);
                    exp_inputs.insert(inp_meta_r);
                    exp_inputs.insert(inp_bt_remote);
                }
            }

            ggml_tensor * cur = rms_norm_w(c, ggml_get_rows(c, w.tok_embd, inp_tok), w.tok_norm, w.rms_eps);
            std::vector<ggml_tensor *> blocks;
            ggml_tensor * tap_l2 = nullptr;

            for (int ip = 0; ip < w.core_start; ++ip) {
                if (ip == w.twodsd.feature_layers[0]) tap_l2 = cur;
                const auto & L = w.layers[ip];
                ggml_tensor * inpSA = cur;
                ggml_tensor * k_p = plane_view_kv(c, tc.prefix_k[ip], 0);
                ggml_tensor * v_p = plane_view_kv(c, tc.prefix_v[ip], 0);
                ggml_tensor * st  = plane_view_conv(c, tc.prefix_conv[ip], 0);

                ggml_tensor * inputs[3] = { cur, cur, cur };
                if (L.attn_res_query == nullptr) {
                    inputs[0] = rms_norm_w(c, cur, L.attn_norm, w.rms_eps);
                    inputs[1] = inputs[2] = inputs[0];
                } else if (L.is_swa) {
                    inputs[0] = ggml_inkling2_attn_res_mix(
                        c, cur, L.attn_res_query, L.attn_norm,
                        blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
                    inputs[1] = inputs[2] = inputs[0];
                } else {
                    ggml_tensor * m3 = ggml_inkling2_attn_res_mix(
                        c, cur, L.attn_res_query, L.attn_qkv_norm,
                        blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
                    for (int j = 0; j < 3; ++j)
                        inputs[j] = ggml_view_2d(c, m3, w.n_embd, ci.len, m3->nb[2], (size_t) j * m3->nb[1]);
                }

                ggml_tensor * Q_raw, * K_raw, * V_raw, * Gate, * R_raw, * Sink;
                ggml_tensor * k_dyn = nullptr, * v_dyn = nullptr;
                if (L.is_swa && L.wqkvgrs) {
                    ggml_tensor * proj = ggml_mul_mat(c, L.wqkvgrs, inputs[0]);
                    Q_raw = row_view_2d(c, proj, 0, q_sz);
                    Gate  = row_view_2d(c, proj, q_sz, q_sz);
                    R_raw = row_view_2d(c, proj, 2 * q_sz, r_sz);
                    Sink  = row_view_2d(c, proj, 2 * q_sz + r_sz, Hq);
                    K_raw = row_view_2d(c, proj, 2 * q_sz + r_sz + Hq, kv_sz);
                    V_raw = row_view_2d(c, proj, 2 * q_sz + r_sz + Hq + kv_sz, kv_sz);
                } else {
                    if (L.wqgrs) {
                        ggml_tensor * proj = ggml_mul_mat(c, L.wqgrs, inputs[0]);
                        Q_raw = row_view_2d(c, proj, 0, q_sz);
                        Gate  = row_view_2d(c, proj, q_sz, q_sz);
                        R_raw = row_view_2d(c, proj, 2 * q_sz, r_sz);
                        Sink  = row_view_2d(c, proj, 2 * q_sz + r_sz, Hq);
                    } else {
                        Q_raw = ggml_mul_mat(c, L.wq, inputs[0]);
                        Gate  = ggml_mul_mat(c, L.wqkv_gate, inputs[0]);
                        R_raw = ggml_mul_mat(c, L.attn_rel, inputs[0]);
                        Sink  = ggml_mul_mat(c, L.attn_sink_proj, inputs[0]);
                    }
                    if (!L.is_swa && L.wk_dyn && L.wv_dyn) {
                        ggml_tensor * kd = ggml_mul_mat(c, L.wk_dyn, inputs[1]);
                        K_raw = row_view_2d(c, kd, 0, kv_sz);
                        k_dyn = row_view_2d(c, kd, kv_sz, (int64_t) w.sconv_kernel * Hk);
                        ggml_tensor * vd = ggml_mul_mat(c, L.wv_dyn, inputs[2]);
                        V_raw = row_view_2d(c, vd, 0, kv_sz);
                        v_dyn = row_view_2d(c, vd, kv_sz, (int64_t) w.sconv_kernel * Hk);
                    } else {
                        K_raw = ggml_mul_mat(c, L.wk, inputs[1]);
                        V_raw = ggml_mul_mat(c, L.wv, inputs[2]);
                        if (!L.is_swa) {
                            k_dyn = ggml_mul_mat(c, L.attn_k_conv_proj, inputs[1]);
                            v_dyn = ggml_mul_mat(c, L.attn_v_conv_proj, inputs[2]);
                        }
                    }
                }

                const int swa = L.is_swa ? w.swa_window : 0;
                ggml_tensor * Q = nullptr, * rt = nullptr;
                make_clip_qr_rel_table(
                    c, Q_raw, L.attn_q_clip, R_raw, L.attn_rel_clip,
                    L.is_swa ? nullptr : inp_pos, L.attn_rel_bias, Sink,
                    D, w.d_rel, Hq, w.qk_clip, w.rel_clip, w.rms_eps,
                    L.is_swa ? 0 : w.log_scale_floor, L.is_swa ? 0.0f : w.log_scale_alpha,
                    &Q, &rt);
                if (!L.is_swa && q_cap_n > 0 && step == M - 1) {
                    const int gidx = global_plane_idx[(size_t) ip][0];
                    if (gidx >= 0) {
                        ggml_tensor * qt = ggml_cont(c, ggml_view_3d(
                            c, Q, D, Hq, q_cap_n, Q->nb[1], Q->nb[2],
                            (size_t) (ci.len - q_cap_n) * Q->nb[2]));
                        ggml_set_output(qt);
                        ggml_build_forward_expand(gf, qt);
                        qcap_list.push_back({ qt, gidx });
                    }
                }
                ggml_tensor * kd_dep = nullptr, * vd_dep = nullptr, * ao = nullptr;
                if (!L.is_swa && use_paged) {
                    kd_dep = ggml_inkling2_sconv_kv_write_paged(
                        c, K_raw, L.attn_k_conv, st, st, k_dyn, k_p, inp_pos,
                        inp_bt_local, kf_bs, 0, D, Hk, 0, w.qk_clip, w.rms_eps, 1);
                    vd_dep = ggml_inkling2_sconv_kv_write_paged(
                        c, V_raw, L.attn_v_conv, st, st, v_dyn, v_p, inp_pos,
                        inp_bt_local, kf_bs, kv_sz, D, Hk, 1, w.qk_clip, w.rms_eps, 1);
                    if (stage_t15 && ip < (int) tc.remote_prefix_k.size() && tc.remote_prefix_k[ip]) {
                        ggml_tensor * rk_p = plane_view_kv(c, tc.remote_prefix_k[ip], 0);
                        ggml_tensor * rv_p = plane_view_kv(c, tc.remote_prefix_v[ip], 0);
                        ggml_tensor * part_r = ggml_inkling2_attn_rel_paged(
                            c, Q, rk_p, rv_p, rt, inp_meta_r, nullptr, nullptr, nullptr,
                            inp_bt_remote, kf_bs, w.rel_extent, /*partial=*/1, /*include_sink=*/1);
                        exp_nodes.push_back(part_r);
                        ggml_tensor * part_l = ggml_inkling2_attn_rel_paged(
                            c, Q, k_p, v_p, rt, inp_meta, kd_dep, vd_dep, nullptr,
                            inp_bt_local, kf_bs, w.rel_extent, /*partial=*/1, /*include_sink=*/0);
                        ao = ggml_inkling2_attn_split_combine(c, part_r, part_l, Gate, D, Hq);
                    } else {
                        ao = ggml_inkling2_attn_rel_paged(
                            c, Q, k_p, v_p, rt, inp_meta, kd_dep, vd_dep, Gate,
                            inp_bt_local, kf_bs, w.rel_extent, /*partial=*/0, /*include_sink=*/1);
                    }
                } else {
                    kd_dep = ggml_inkling2_sconv_kv_write(
                        c, K_raw, L.attn_k_conv, st, st, k_dyn, k_p, inp_pos,
                        0, D, Hk, 0, swa, w.qk_clip, w.rms_eps, 1);
                    vd_dep = ggml_inkling2_sconv_kv_write(
                        c, V_raw, L.attn_v_conv, st, st, v_dyn, v_p, inp_pos,
                        kv_sz, D, Hk, 1, swa, w.qk_clip, w.rms_eps, 1);
                    ao = ggml_inkling2_attn_rel(
                        c, Q, k_p, v_p, rt, inp_meta, kd_dep, vd_dep, Gate, swa, w.rel_extent, 0, 1);
                }

                cur = ggml_mul_mat(c, L.wo, ao);
                if (ip % w.attn_res_block == 0) {
                    cur = ggml_inkling2_sconv(c, cur, L.attn_post_conv, st, st, nullptr, 2 * kv_sz, 0, 1);
                    blocks.push_back(inpSA);
                } else {
                    cur = ggml_inkling2_sconv_add(c, cur, L.attn_post_conv, st, st, nullptr, inpSA, 2 * kv_sz, 0, 1);
                }
                ggml_tensor * ffn_inp = cur;
                cur = ggml_inkling2_attn_res_mix(
                    c, cur, L.ffn_res_query, L.ffn_norm,
                    blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
                ggml_tensor * rl   = ggml_mul_mat(c, L.ffn_gate_inp_q8 ? L.ffn_gate_inp_q8 : L.ffn_gate_inp, cur);
                ggml_tensor * sel  = ggml_inkling2_sigmoid_topk_ids(c, rl, L.ffn_exp_probs_b, w.n_expert_used);
                ggml_tensor * wsel = ggml_inkling2_sigmoid_topk_wts(c, rl, sel, w.route_scale, 1e-20f);
                ggml_tensor * x3   = ggml_reshape_3d(c, cur, w.n_embd, 1, ci.len);
                ggml_tensor * gu   = ggml_mul_mat_id(c, L.ffn_gate_up_exps, x3, sel);
                ggml_tensor * act  = ggml_inkling2_situ(c, gu, w.situ_beta, w.situ_linear_beta);
                ggml_tensor * dn   = ggml_mul_mat_id(c, L.ffn_down_exps, act, sel);
                cur = ggml_ds4_moe_fused_combine_shared(c, dn, wsel, nullptr);
                cur = ggml_inkling2_sconv_add(c, cur, L.ffn_post_conv, st, st, nullptr, ffn_inp, 2 * kv_sz + w.n_embd, 0, 1);
            }

            ggml_build_forward_expand(gf, ggml_cpy(c, cur, cs.prelude_cur));
            cs.n_prelude_blocks = (int) blocks.size();
            for (int b = 0; b < cs.n_prelude_blocks; ++b) {
                ggml_build_forward_expand(gf, ggml_cpy(c, blocks[b], cs.blocks[b]));
            }
            if (tap_l2) {
                ggml_build_forward_expand(gf, ggml_cpy(c, tap_l2, cs.tap_l2));
            }

            ggml_backend_sched_t sched = nullptr;
            ggml_gallocr_t ga = nullptr;
            if (!alloc_stage_graph(gf, max_nodes, stage_t15, exp_nodes, exp_inputs, sched, ga)) {
                ggml_free(c);
                free_ring();
                return false;
            }
            std::vector<int32_t> pos_v(ci.len);
            for (int i = 0; i < ci.len; ++i) pos_v[i] = ci.pos + i;
            int32_t meta[4] = { ci.pos, ci.pos + ci.len, 0, ci.pos + ci.len };
            ggml_backend_tensor_set(inp_tok,  tokens + ci.off, 0, (size_t) ci.len * sizeof(int32_t));
            ggml_backend_tensor_set(inp_pos,  pos_v.data(),    0, (size_t) ci.len * sizeof(int32_t));
            ggml_backend_tensor_set(inp_meta, meta,            0, sizeof(meta));
            if (use_paged && inp_bt_local) {
                const auto & btl = cache.kvflash->bt_local();
                ggml_backend_tensor_set(inp_bt_local, btl.data(), 0, btl.size() * sizeof(int32_t));
                if (stage_t15 && inp_meta_r && inp_bt_remote) {
                    const auto & btr = cache.kvflash->bt_remote();
                    ggml_backend_tensor_set(inp_meta_r,    meta,       0, sizeof(meta));
                    ggml_backend_tensor_set(inp_bt_remote, btr.data(), 0, btr.size() * sizeof(int32_t));
                }
            }
            const bool ok = sched
                ? (ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS)
                : (ggml_backend_graph_compute(target_backend, gf) == GGML_STATUS_SUCCESS);
            if (ok) {
                for (const auto & qc : qcap_list) scatter_q_cap(qc.first, qc.second);
            }
            if (sched) ggml_backend_sched_free(sched);
            if (ga)    ggml_gallocr_free(ga);
            ggml_free(c);
            if (!ok) { free_ring(); return false; }
        }

        // Active wavefront streams at `step`: (pass=2, cidx=step-2), (pass=1, cidx=step-1), (pass=0, cidx=step)
        struct StreamDesc { int chunk_idx; int ring_slot; int pass; int len; int pos; int tok_off; };
        std::vector<StreamDesc> streams;
        int T_tot = 0;
        for (int pass = 2; pass >= 0; --pass) {
            const int cidx = step - pass;
            if (cidx >= 0 && cidx < M) {
                streams.push_back({ cidx, cidx % 3, pass, chunks[cidx].len, chunks[cidx].pos, T_tot });
                T_tot += chunks[cidx].len;
            }
        }
        if (streams.empty()) continue;
        const int n_s = (int) streams.size();

        // ── Stage 2: Fused 3-Pass Loop Core (L6..17 ONCE for up to 3 chunks!) ──
        {
            const size_t max_nodes = 16384;
            struct ggml_init_params gp = { max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false) + (16 << 20), nullptr, true };
            ggml_context * c = ggml_init(gp);
            if (!c) { free_ring(); return false; }
            ggml_cgraph * gf = ggml_new_graph_custom(c, max_nodes, false);

            ggml_tensor * s_pos[3]      = {};
            ggml_tensor * s_meta[3]     = {};
            ggml_tensor * s_meta_r[3]   = {};
            ggml_tensor * s_cur[3]      = {};
            ggml_tensor * s_bank_out[3] = {};
            ggml_tensor * inp_bt_local  = nullptr;
            ggml_tensor * inp_bt_remote = nullptr;
            std::vector<ggml_tensor *> exp_nodes;
            std::unordered_set<const ggml_tensor *> exp_inputs;
            std::vector<std::pair<ggml_tensor *, int>> qcap_list;
            std::vector<ggml_tensor *> s_blocks[3];
            std::vector<ggml_tensor *> s_bank[3];

            if (use_paged) {
                inp_bt_local = ggml_new_tensor_1d(c, GGML_TYPE_I32, cache.kvflash->max_blocks());
                ggml_set_input(inp_bt_local);
                if (stage_t15) {
                    inp_bt_remote = ggml_new_tensor_1d(c, GGML_TYPE_I32, cache.kvflash->max_blocks());
                    ggml_set_input(inp_bt_remote);
                    exp_inputs.insert(inp_bt_remote);
                }
            }
            for (int si = 0; si < n_s; ++si) {
                const auto & sd = streams[si];
                const auto & cs = ring[sd.ring_slot];
                s_pos[si]  = ggml_new_tensor_1d(c, GGML_TYPE_I32, sd.len);
                s_meta[si] = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
                ggml_set_input(s_pos[si]);
                ggml_set_input(s_meta[si]);
                if (stage_t15) {
                    s_meta_r[si] = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
                    ggml_set_input(s_meta_r[si]);
                    exp_inputs.insert(s_meta_r[si]);
                }

                s_cur[si] = cs.prelude_cur;
                for (int b = 0; b < cs.n_prelude_blocks; ++b) s_blocks[si].push_back(cs.blocks[b]);
                if (sd.pass > 0) {
                    s_bank_out[si] = cs.bank_out;
                    for (int b = 0; b < cs.n_bank; ++b) s_bank[si].push_back(cs.bank[b]);
                }
            }

            auto concat_s = [&](ggml_tensor * arr[3]) -> ggml_tensor * {
                ggml_tensor * r = arr[0];
                for (int si = 1; si < n_s; ++si) r = ggml_concat(c, r, arr[si], 1);
                return r;
            };
            auto slice_s = [&](ggml_tensor * big, int si) -> ggml_tensor * {
                if (n_s == 1) return big;
                return ggml_view_2d(c, big, big->ne[0], streams[si].len, big->nb[1], (size_t) streams[si].tok_off * big->nb[1]);
            };

            ggml_tensor * s_core_taps[3][3] = {};
            for (int ip = w.core_start; ip < w.core_end; ++ip) {
                const auto & L    = w.layers[ip];
                const auto & loop = w.loop_layers[ip - w.core_start];

                ggml_tensor * inpSA[3] = {};
                ggml_tensor * in_q[3] = {}, * in_k[3] = {}, * in_v[3] = {};
                ggml_tensor * msgs[3] = {};

                for (int si = 0; si < n_s; ++si) {
                    const auto & sd = streams[si];
                    if (sd.pass == 2) {
                        for (int ti = 0; ti < 3; ++ti) {
                            if (ip == w.twodsd.feature_layers[1 + ti]) s_core_taps[si][ti] = s_cur[si];
                        }
                    }
                    inpSA[si] = s_cur[si];
                    if (sd.pass > 0) {
                        msgs[si] = ggml_inkling2_attn_res_mix(
                            c, s_bank_out[si], loop.msg_query, nullptr,
                            s_bank[si].empty() ? nullptr : s_bank[si].data(), (int) s_bank[si].size(), w.rms_eps);
                    }
                    if (sd.pass == 0) {
                        if (L.is_swa) {
                            in_q[si] = ggml_inkling2_attn_res_mix(
                                c, s_cur[si], L.attn_res_query, L.attn_norm,
                                s_blocks[si].empty() ? nullptr : s_blocks[si].data(), (int) s_blocks[si].size(), w.rms_eps);
                            in_k[si] = in_v[si] = in_q[si];
                        } else {
                            ggml_tensor * m3 = ggml_inkling2_attn_res_mix(
                                c, s_cur[si], L.attn_res_query, L.attn_qkv_norm,
                                s_blocks[si].empty() ? nullptr : s_blocks[si].data(), (int) s_blocks[si].size(), w.rms_eps);
                            in_q[si] = ggml_view_2d(c, m3, w.n_embd, sd.len, m3->nb[2], 0);
                            in_k[si] = ggml_view_2d(c, m3, w.n_embd, sd.len, m3->nb[2], m3->nb[1]);
                            in_v[si] = ggml_view_2d(c, m3, w.n_embd, sd.len, m3->nb[2], 2 * m3->nb[1]);
                        }
                    } else {
                        ggml_tensor * mx = ggml_inkling2_attn_res_mix(
                            c, s_cur[si], L.attn_res_query, nullptr,
                            s_blocks[si].empty() ? nullptr : s_blocks[si].data(), (int) s_blocks[si].size(), w.rms_eps);
                        ggml_tensor * pre_qkv   = L.is_swa ? mx : ggml_permute(c, mx, 0, 2, 1, 3);
                        ggml_tensor * msg_qkv_v = ggml_view_3d(c, msgs[si], w.n_embd, 3, sd.len, msgs[si]->nb[1], msgs[si]->nb[2], 0);
                        ggml_tensor * msg_qkv   = ggml_permute(c, msg_qkv_v, 0, 2, 1, 3);
                        ggml_tensor * nw_qkv    = L.is_swa ? L.attn_norm : L.attn_qkv_norm;
                        ggml_tensor * cpre      = ggml_inkling2_loop_ctrl_pre(c, pre_qkv, msg_qkv, w.rms_eps);
                        ggml_tensor * gate      = ggml_mul_mat(c, loop.ctrl_down_qkv, cpre);
                        ggml_tensor * gfo       = ggml_mul_mat(c, loop.ctrl_gf_qkv, gate);
                        ggml_tensor * qkv_in    = ggml_inkling2_loop_ctrl_post(c, pre_qkv, msg_qkv, gfo, nw_qkv, w.rms_eps);
                        in_q[si] = ggml_view_2d(c, qkv_in, w.n_embd, sd.len, qkv_in->nb[1], 0);
                        in_k[si] = ggml_view_2d(c, qkv_in, w.n_embd, sd.len, qkv_in->nb[1], qkv_in->nb[2]);
                        in_v[si] = ggml_view_2d(c, qkv_in, w.n_embd, sd.len, qkv_in->nb[1], 2 * qkv_in->nb[2]);
                    }
                }

                // ONE FUSED Q/Gate/R/Sink, K, V GEMM ACROSS ALL 3 LOOP PASSES!
                ggml_tensor * q_in_cat = concat_s(in_q);
                ggml_tensor * k_in_cat = concat_s(in_k);
                ggml_tensor * v_in_cat = concat_s(in_v);
                ggml_tensor * qgrs_all = L.wqgrs ? ggml_mul_mat(c, L.wqgrs, q_in_cat) : nullptr;
                ggml_tensor * q_sep = qgrs_all ? nullptr : ggml_mul_mat(c, L.wq, q_in_cat);
                ggml_tensor * g_sep = qgrs_all ? nullptr : ggml_mul_mat(c, L.wqkv_gate, q_in_cat);
                ggml_tensor * r_sep = qgrs_all ? nullptr : ggml_mul_mat(c, L.attn_rel, q_in_cat);
                ggml_tensor * s_sep = qgrs_all ? nullptr : ggml_mul_mat(c, L.attn_sink_proj, q_in_cat);
                const bool use_kv_dyn = (!L.is_swa && L.wk_dyn && L.wv_dyn);
                ggml_tensor * k_all = ggml_mul_mat(c, use_kv_dyn ? L.wk_dyn : L.wk, k_in_cat);
                ggml_tensor * v_all = ggml_mul_mat(c, use_kv_dyn ? L.wv_dyn : L.wv, v_in_cat);
                ggml_tensor * kd_sep = (!L.is_swa && !use_kv_dyn) ? ggml_mul_mat(c, L.attn_k_conv_proj, k_in_cat) : nullptr;
                ggml_tensor * vd_sep = (!L.is_swa && !use_kv_dyn) ? ggml_mul_mat(c, L.attn_v_conv_proj, v_in_cat) : nullptr;

                ggml_tensor * ao_s[3] = {};
                const int swa = L.is_swa ? w.swa_window : 0;
                for (int si = 0; si < n_s; ++si) {
                    const auto & sd = streams[si];
                    ggml_tensor * k_p = plane_view_kv(c, tc.prefix_k[ip], sd.pass);
                    ggml_tensor * v_p = plane_view_kv(c, tc.prefix_v[ip], sd.pass);
                    ggml_tensor * st  = plane_view_conv(c, tc.prefix_conv[ip], sd.pass);

                    ggml_tensor * Q_raw, * Gate, * R_raw, * Sink;
                    if (qgrs_all) {
                        ggml_tensor * pq = slice_s(qgrs_all, si);
                        Q_raw = row_view_2d(c, pq, 0, q_sz);
                        Gate  = row_view_2d(c, pq, q_sz, q_sz);
                        R_raw = row_view_2d(c, pq, 2 * q_sz, r_sz);
                        Sink  = row_view_2d(c, pq, 2 * q_sz + r_sz, Hq);
                    } else {
                        Q_raw = slice_s(q_sep, si);
                        Gate  = slice_s(g_sep, si);
                        R_raw = slice_s(r_sep, si);
                        Sink  = slice_s(s_sep, si);
                    }
                    ggml_tensor * pk = slice_s(k_all, si);
                    ggml_tensor * pv = slice_s(v_all, si);
                    ggml_tensor * K_raw = use_kv_dyn ? row_view_2d(c, pk, 0, kv_sz) : pk;
                    ggml_tensor * k_dyn = use_kv_dyn ? row_view_2d(c, pk, kv_sz, (int64_t) w.sconv_kernel * Hk)
                                                     : (kd_sep ? slice_s(kd_sep, si) : nullptr);
                    ggml_tensor * V_raw = use_kv_dyn ? row_view_2d(c, pv, 0, kv_sz) : pv;
                    ggml_tensor * v_dyn = use_kv_dyn ? row_view_2d(c, pv, kv_sz, (int64_t) w.sconv_kernel * Hk)
                                                     : (vd_sep ? slice_s(vd_sep, si) : nullptr);

                    ggml_tensor * Q = nullptr, * rt = nullptr;
                    make_clip_qr_rel_table(
                        c, Q_raw, L.attn_q_clip, R_raw, L.attn_rel_clip,
                        L.is_swa ? nullptr : s_pos[si], L.attn_rel_bias, Sink,
                        D, w.d_rel, Hq, w.qk_clip, w.rel_clip, w.rms_eps,
                        L.is_swa ? 0 : w.log_scale_floor, L.is_swa ? 0.0f : w.log_scale_alpha,
                        &Q, &rt);
                    if (!L.is_swa && q_cap_n > 0 && sd.chunk_idx == M - 1) {
                        const int gidx = global_plane_idx[(size_t) ip][sd.pass];
                        if (gidx >= 0) {
                            ggml_tensor * qt = ggml_cont(c, ggml_view_3d(
                                c, Q, D, Hq, q_cap_n, Q->nb[1], Q->nb[2],
                                (size_t) (sd.len - q_cap_n) * Q->nb[2]));
                            ggml_set_output(qt);
                            ggml_build_forward_expand(gf, qt);
                            qcap_list.push_back({ qt, gidx });
                        }
                    }
                    if (!L.is_swa && use_paged) {
                        ggml_tensor * kd_dep = ggml_inkling2_sconv_kv_write_paged(
                            c, K_raw, L.attn_k_conv, st, st, k_dyn, k_p, s_pos[si],
                            inp_bt_local, kf_bs, 0, D, Hk, 0, w.qk_clip, w.rms_eps, 1);
                        ggml_tensor * vd_dep = ggml_inkling2_sconv_kv_write_paged(
                            c, V_raw, L.attn_v_conv, st, st, v_dyn, v_p, s_pos[si],
                            inp_bt_local, kf_bs, kv_sz, D, Hk, 1, w.qk_clip, w.rms_eps, 1);
                        if (stage_t15 && ip < (int) tc.remote_prefix_k.size() && tc.remote_prefix_k[ip]) {
                            ggml_tensor * rk_p = plane_view_kv(c, tc.remote_prefix_k[ip], sd.pass);
                            ggml_tensor * rv_p = plane_view_kv(c, tc.remote_prefix_v[ip], sd.pass);
                            ggml_tensor * part_r = ggml_inkling2_attn_rel_paged(
                                c, Q, rk_p, rv_p, rt, s_meta_r[si], nullptr, nullptr, nullptr,
                                inp_bt_remote, kf_bs, w.rel_extent, /*partial=*/1, /*include_sink=*/1);
                            exp_nodes.push_back(part_r);
                            ggml_tensor * part_l = ggml_inkling2_attn_rel_paged(
                                c, Q, k_p, v_p, rt, s_meta[si], kd_dep, vd_dep, nullptr,
                                inp_bt_local, kf_bs, w.rel_extent, /*partial=*/1, /*include_sink=*/0);
                            ao_s[si] = ggml_inkling2_attn_split_combine(c, part_r, part_l, Gate, D, Hq);
                        } else {
                            ao_s[si] = ggml_inkling2_attn_rel_paged(
                                c, Q, k_p, v_p, rt, s_meta[si], kd_dep, vd_dep, Gate,
                                inp_bt_local, kf_bs, w.rel_extent, /*partial=*/0, /*include_sink=*/1);
                        }
                    } else {
                        ggml_tensor * kd_dep = ggml_inkling2_sconv_kv_write(
                            c, K_raw, L.attn_k_conv, st, st, k_dyn, k_p, s_pos[si],
                            0, D, Hk, 0, swa, w.qk_clip, w.rms_eps, 1);
                        ggml_tensor * vd_dep = ggml_inkling2_sconv_kv_write(
                            c, V_raw, L.attn_v_conv, st, st, v_dyn, v_p, s_pos[si],
                            kv_sz, D, Hk, 1, swa, w.qk_clip, w.rms_eps, 1);
                        ao_s[si] = ggml_inkling2_attn_rel(
                            c, Q, k_p, v_p, rt, s_meta[si], kd_dep, vd_dep, Gate, swa, w.rel_extent, 0, 1);
                    }
                }

                // ONE FUSED O-PROJ GEMM ACROSS ALL 3 LOOP PASSES!
                ggml_tensor * wo_all = ggml_mul_mat(c, L.wo, concat_s(ao_s));

                ggml_tensor * ffn_inp[3]  = {};
                ggml_tensor * ffn_in_s[3] = {};
                for (int si = 0; si < n_s; ++si) {
                    const auto & sd = streams[si];
                    ggml_tensor * st = plane_view_conv(c, tc.prefix_conv[ip], sd.pass);
                    ggml_tensor * x  = nullptr;
                    if (ip % w.attn_res_block == 0) {
                        x = ggml_inkling2_sconv(c, slice_s(wo_all, si), L.attn_post_conv, st, st, nullptr, 2 * kv_sz, 0, 1);
                        s_blocks[si].push_back(inpSA[si]);
                    } else {
                        x = ggml_inkling2_sconv_add(c, slice_s(wo_all, si), L.attn_post_conv, st, st, nullptr, inpSA[si], 2 * kv_sz, 0, 1);
                    }
                    ffn_inp[si] = x;

                    if (sd.pass == 0) {
                        ffn_in_s[si] = ggml_inkling2_attn_res_mix(
                            c, x, L.ffn_res_query, L.ffn_norm,
                            s_blocks[si].empty() ? nullptr : s_blocks[si].data(), (int) s_blocks[si].size(), w.rms_eps);
                    } else {
                        ggml_tensor * mx = ggml_inkling2_attn_res_mix(
                            c, x, L.ffn_res_query, nullptr,
                            s_blocks[si].empty() ? nullptr : s_blocks[si].data(), (int) s_blocks[si].size(), w.rms_eps);
                        ggml_tensor * mf = ggml_view_2d(c, msgs[si], w.n_embd, sd.len, msgs[si]->nb[2],
                                                        (size_t) Inkling2LoopLayer::CTRL_FFN * msgs[si]->nb[1]);
                        ggml_tensor * cpre = ggml_inkling2_loop_ctrl_pre(c, mx, mf, w.rms_eps);
                        ggml_tensor * cd_w = loop.ctrl_down_f16[Inkling2LoopLayer::CTRL_FFN]
                            ? loop.ctrl_down_f16[Inkling2LoopLayer::CTRL_FFN]
                            : loop.ctrl_down[Inkling2LoopLayer::CTRL_FFN];
                        ggml_tensor * gt   = ggml_mul_mat(c, cd_w, cpre);
                        ggml_tensor * gfo  = ggml_mul_mat(c, loop.ctrl_gf[Inkling2LoopLayer::CTRL_FFN], gt);
                        ffn_in_s[si] = ggml_inkling2_loop_ctrl_post(c, mx, mf, gfo, L.ffn_norm, w.rms_eps);
                    }
                }

                // ONE FUSED ROUTER + 128-EXPERT MoE PASS ACROSS ALL 3 LOOP PASSES!
                ggml_tensor * ffn_cat = concat_s(ffn_in_s);
                ggml_tensor * rl   = ggml_mul_mat(c, L.ffn_gate_inp_q8 ? L.ffn_gate_inp_q8 : L.ffn_gate_inp, ffn_cat);
                ggml_tensor * sel  = ggml_inkling2_sigmoid_topk_ids(c, rl, L.ffn_exp_probs_b, w.n_expert_used);
                ggml_tensor * wsel = ggml_inkling2_sigmoid_topk_wts(c, rl, sel, w.route_scale, 1e-20f);
                ggml_tensor * x3   = ggml_reshape_3d(c, ffn_cat, w.n_embd, 1, T_tot);
                ggml_tensor * gu   = ggml_mul_mat_id(c, L.ffn_gate_up_exps, x3, sel);
                ggml_tensor * act  = ggml_inkling2_situ(c, gu, w.situ_beta, w.situ_linear_beta);
                ggml_tensor * dn   = ggml_mul_mat_id(c, L.ffn_down_exps, act, sel);
                ggml_tensor * moe_all = ggml_ds4_moe_fused_combine_shared(c, dn, wsel, nullptr);

                for (int si = 0; si < n_s; ++si) {
                    const auto & sd = streams[si];
                    ggml_tensor * st = plane_view_conv(c, tc.prefix_conv[ip], sd.pass);
                    s_cur[si] = ggml_inkling2_sconv_add(
                        c, slice_s(moe_all, si), L.ffn_post_conv, st, st, nullptr, ffn_inp[si], 2 * kv_sz + w.n_embd, 0, 1);
                }
            }

            // Persist each stream's end-of-pass state for immediate Coda (Stage 3) and next Core pass
            for (int si = 0; si < n_s; ++si) {
                const auto & sd = streams[si];
                auto & cs = ring[sd.ring_slot];
                ggml_build_forward_expand(gf, ggml_cpy(c, s_cur[si], cs.core_cur));
                cs.n_blocks = (int) s_blocks[si].size();
                for (int b = cs.n_prelude_blocks; b < cs.n_blocks; ++b) {
                    ggml_build_forward_expand(gf, ggml_cpy(c, s_blocks[si][b], cs.blocks[b]));
                }
                if (sd.pass < 2) {
                    ggml_build_forward_expand(gf, ggml_cpy(c, s_cur[si], cs.bank_out));
                    cs.n_bank = std::max(0, cs.n_blocks - cs.n_prelude_blocks - 1);
                    for (int b = 0; b < cs.n_bank; ++b) {
                        ggml_build_forward_expand(gf, ggml_cpy(c, s_blocks[si][cs.n_prelude_blocks + 1 + b], cs.bank[b]));
                    }
                } else {
                    for (int ti = 0; ti < 3; ++ti) {
                        if (s_core_taps[si][ti]) {
                            ggml_build_forward_expand(gf, ggml_cpy(c, s_core_taps[si][ti], cs.core_taps[ti]));
                        }
                    }
                }
            }

            ggml_backend_sched_t sched = nullptr;
            ggml_gallocr_t ga = nullptr;
            if (!alloc_stage_graph(gf, max_nodes, stage_t15, exp_nodes, exp_inputs, sched, ga)) {
                ggml_free(c);
                free_ring();
                return false;
            }
            if (use_paged && inp_bt_local) {
                const auto & btl = cache.kvflash->bt_local();
                ggml_backend_tensor_set(inp_bt_local, btl.data(), 0, btl.size() * sizeof(int32_t));
                if (stage_t15 && inp_bt_remote) {
                    const auto & btr = cache.kvflash->bt_remote();
                    ggml_backend_tensor_set(inp_bt_remote, btr.data(), 0, btr.size() * sizeof(int32_t));
                }
            }
            for (int si = 0; si < n_s; ++si) {
                const auto & sd = streams[si];
                std::vector<int32_t> pos_v(sd.len);
                for (int i = 0; i < sd.len; ++i) pos_v[i] = sd.pos + i;
                int32_t meta[4] = { sd.pos, sd.pos + sd.len, 0, sd.pos + sd.len };
                ggml_backend_tensor_set(s_pos[si],  pos_v.data(), 0, (size_t) sd.len * sizeof(int32_t));
                ggml_backend_tensor_set(s_meta[si], meta,         0, sizeof(meta));
                if (stage_t15 && s_meta_r[si]) {
                    ggml_backend_tensor_set(s_meta_r[si], meta, 0, sizeof(meta));
                }
            }
            const bool ok = sched
                ? (ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS)
                : (ggml_backend_graph_compute(target_backend, gf) == GGML_STATUS_SUCCESS);
            if (ok) {
                for (const auto & qc : qcap_list) scatter_q_cap(qc.first, qc.second);
            }
            if (sched) ggml_backend_sched_free(sched);
            if (ga)    ggml_gallocr_free(ga);
            ggml_free(c);
            if (!ok) { free_ring(); return false; }
        }

        // ── Stage 3: Coda (L18..23) + Fused In-Graph Tail Drafter KV Commit ──
        std::vector<StreamDesc> c_streams;
        int C_tot = 0;
        for (const auto & sd : streams) {
            if (!share_coda || sd.pass == 2) {
                StreamDesc cs_d = sd;
                cs_d.tok_off = C_tot;
                c_streams.push_back(cs_d);
                C_tot += sd.len;
            }
        }
        if (!c_streams.empty()) {
            const int c_n = (int) c_streams.size();
            const size_t max_nodes = 12288;
            struct ggml_init_params gp = { max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false) + (12 << 20), nullptr, true };
            ggml_context * c = ggml_init(gp);
            if (!c) { free_ring(); return false; }
            ggml_cgraph * gf = ggml_new_graph_custom(c, max_nodes, false);

            ggml_tensor * c_pos[3]      = {};
            ggml_tensor * c_meta[3]     = {};
            ggml_tensor * c_meta_r[3]   = {};
            ggml_tensor * c_cur[3]      = {};
            ggml_tensor * inp_bt_local  = nullptr;
            ggml_tensor * inp_bt_remote = nullptr;
            std::vector<ggml_tensor *> exp_nodes;
            std::unordered_set<const ggml_tensor *> exp_inputs;
            std::vector<std::pair<ggml_tensor *, std::vector<int>>> qcap_list;
            std::vector<ggml_tensor *> c_blocks[3];
            int pass2_si = -1;

            if (use_paged) {
                inp_bt_local = ggml_new_tensor_1d(c, GGML_TYPE_I32, cache.kvflash->max_blocks());
                ggml_set_input(inp_bt_local);
                if (stage_t15) {
                    inp_bt_remote = ggml_new_tensor_1d(c, GGML_TYPE_I32, cache.kvflash->max_blocks());
                    ggml_set_input(inp_bt_remote);
                    exp_inputs.insert(inp_bt_remote);
                }
            }
            for (int si = 0; si < c_n; ++si) {
                const auto & sd = c_streams[si];
                const auto & cs = ring[sd.ring_slot];
                c_pos[si]  = ggml_new_tensor_1d(c, GGML_TYPE_I32, sd.len);
                c_meta[si] = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
                ggml_set_input(c_pos[si]);
                ggml_set_input(c_meta[si]);
                if (stage_t15) {
                    c_meta_r[si] = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
                    ggml_set_input(c_meta_r[si]);
                    exp_inputs.insert(c_meta_r[si]);
                }
                c_cur[si] = cs.core_cur;
                for (int b = 0; b < cs.n_blocks; ++b) c_blocks[si].push_back(cs.blocks[b]);
                if (sd.pass == 2) pass2_si = si;
            }

            auto concat_c = [&](ggml_tensor * arr[3]) -> ggml_tensor * {
                ggml_tensor * r = arr[0];
                for (int si = 1; si < c_n; ++si) r = ggml_concat(c, r, arr[si], 1);
                return r;
            };
            auto slice_c = [&](ggml_tensor * big, int si) -> ggml_tensor * {
                if (c_n == 1) return big;
                return ggml_view_2d(c, big, big->ne[0], c_streams[si].len, big->nb[1], (size_t) c_streams[si].tok_off * big->nb[1]);
            };

            ggml_tensor * tap_l22 = nullptr;
            for (int ip = w.core_end; ip < w.n_layer_phys; ++ip) {
                const auto & L = w.layers[ip];
                if (pass2_si >= 0 && ip == w.twodsd.feature_layers[4]) {
                    tap_l22 = c_cur[pass2_si];
                }

                ggml_tensor * inpSA[3] = {};
                ggml_tensor * in_q[3] = {}, * in_k[3] = {}, * in_v[3] = {};
                for (int si = 0; si < c_n; ++si) {
                    const auto & sd = c_streams[si];
                    inpSA[si] = c_cur[si];
                    if (L.is_swa) {
                        in_q[si] = ggml_inkling2_attn_res_mix(
                            c, c_cur[si], L.attn_res_query, L.attn_norm,
                            c_blocks[si].empty() ? nullptr : c_blocks[si].data(), (int) c_blocks[si].size(), w.rms_eps);
                        in_k[si] = in_v[si] = in_q[si];
                    } else {
                        ggml_tensor * m3 = ggml_inkling2_attn_res_mix(
                            c, c_cur[si], L.attn_res_query, L.attn_qkv_norm,
                            c_blocks[si].empty() ? nullptr : c_blocks[si].data(), (int) c_blocks[si].size(), w.rms_eps);
                        in_q[si] = ggml_view_2d(c, m3, w.n_embd, sd.len, m3->nb[2], 0);
                        in_k[si] = ggml_view_2d(c, m3, w.n_embd, sd.len, m3->nb[2], m3->nb[1]);
                        in_v[si] = ggml_view_2d(c, m3, w.n_embd, sd.len, m3->nb[2], 2 * m3->nb[1]);
                    }
                }

                // Fused QKV/Gate/R/Sink GEMM across active Coda stream(s)
                ggml_tensor * proj_all = nullptr;
                ggml_tensor * qgrs_all = nullptr;
                ggml_tensor * q_sep = nullptr, * g_sep = nullptr, * r_sep = nullptr, * s_sep = nullptr;
                ggml_tensor * k_all = nullptr, * v_all = nullptr, * kd_sep = nullptr, * vd_sep = nullptr;
                const bool use_kv_dyn = (!L.is_swa && L.wk_dyn && L.wv_dyn);
                if (L.is_swa && L.wqkvgrs) {
                    proj_all = ggml_mul_mat(c, L.wqkvgrs, concat_c(in_q));
                } else {
                    ggml_tensor * q_in_cat = concat_c(in_q);
                    ggml_tensor * k_in_cat = concat_c(in_k);
                    ggml_tensor * v_in_cat = concat_c(in_v);
                    if (L.wqgrs) {
                        qgrs_all = ggml_mul_mat(c, L.wqgrs, q_in_cat);
                    } else {
                        q_sep = ggml_mul_mat(c, L.wq, q_in_cat);
                        g_sep = ggml_mul_mat(c, L.wqkv_gate, q_in_cat);
                        r_sep = ggml_mul_mat(c, L.attn_rel, q_in_cat);
                        s_sep = ggml_mul_mat(c, L.attn_sink_proj, q_in_cat);
                    }
                    k_all = ggml_mul_mat(c, use_kv_dyn ? L.wk_dyn : L.wk, k_in_cat);
                    v_all = ggml_mul_mat(c, use_kv_dyn ? L.wv_dyn : L.wv, v_in_cat);
                    if (!L.is_swa && !use_kv_dyn) {
                        kd_sep = ggml_mul_mat(c, L.attn_k_conv_proj, k_in_cat);
                        vd_sep = ggml_mul_mat(c, L.attn_v_conv_proj, v_in_cat);
                    }
                }

                ggml_tensor * ao_s[3] = {};
                const int swa = L.is_swa ? w.swa_window : 0;
                for (int si = 0; si < c_n; ++si) {
                    const auto & sd = c_streams[si];
                    ggml_tensor * k_p = plane_view_kv(c, tc.prefix_k[ip], sd.pass);
                    ggml_tensor * v_p = plane_view_kv(c, tc.prefix_v[ip], sd.pass);
                    ggml_tensor * st  = plane_view_conv(c, tc.prefix_conv[ip], sd.pass);

                    ggml_tensor * Q_raw, * K_raw, * V_raw, * Gate, * R_raw, * Sink;
                    ggml_tensor * k_dyn = nullptr, * v_dyn = nullptr;
                    if (proj_all) {
                        ggml_tensor * pp = slice_c(proj_all, si);
                        Q_raw = row_view_2d(c, pp, 0, q_sz);
                        Gate  = row_view_2d(c, pp, q_sz, q_sz);
                        R_raw = row_view_2d(c, pp, 2 * q_sz, r_sz);
                        Sink  = row_view_2d(c, pp, 2 * q_sz + r_sz, Hq);
                        K_raw = row_view_2d(c, pp, 2 * q_sz + r_sz + Hq, kv_sz);
                        V_raw = row_view_2d(c, pp, 2 * q_sz + r_sz + Hq + kv_sz, kv_sz);
                    } else {
                        if (qgrs_all) {
                            ggml_tensor * pq = slice_c(qgrs_all, si);
                            Q_raw = row_view_2d(c, pq, 0, q_sz);
                            Gate  = row_view_2d(c, pq, q_sz, q_sz);
                            R_raw = row_view_2d(c, pq, 2 * q_sz, r_sz);
                            Sink  = row_view_2d(c, pq, 2 * q_sz + r_sz, Hq);
                        } else {
                            Q_raw = slice_c(q_sep, si);
                            Gate  = slice_c(g_sep, si);
                            R_raw = slice_c(r_sep, si);
                            Sink  = slice_c(s_sep, si);
                        }
                        ggml_tensor * pk = slice_c(k_all, si);
                        ggml_tensor * pv = slice_c(v_all, si);
                        K_raw = use_kv_dyn ? row_view_2d(c, pk, 0, kv_sz) : pk;
                        k_dyn = use_kv_dyn ? row_view_2d(c, pk, kv_sz, (int64_t) w.sconv_kernel * Hk)
                                           : (kd_sep ? slice_c(kd_sep, si) : nullptr);
                        V_raw = use_kv_dyn ? row_view_2d(c, pv, 0, kv_sz) : pv;
                        v_dyn = use_kv_dyn ? row_view_2d(c, pv, kv_sz, (int64_t) w.sconv_kernel * Hk)
                                           : (vd_sep ? slice_c(vd_sep, si) : nullptr);
                    }

                    ggml_tensor * Q = nullptr, * rt = nullptr;
                    make_clip_qr_rel_table(
                        c, Q_raw, L.attn_q_clip, R_raw, L.attn_rel_clip,
                        L.is_swa ? nullptr : c_pos[si], L.attn_rel_bias, Sink,
                        D, w.d_rel, Hq, w.qk_clip, w.rel_clip, w.rms_eps,
                        L.is_swa ? 0 : w.log_scale_floor, L.is_swa ? 0.0f : w.log_scale_alpha,
                        &Q, &rt);
                    if (!L.is_swa && q_cap_n > 0 && sd.chunk_idx == M - 1) {
                        std::vector<int> gidxs;
                        if (share_coda && sd.pass == 2) {
                            for (int tp = 0; tp < tc.planes[ip] && tp < 4; ++tp) {
                                if (global_plane_idx[(size_t) ip][tp] >= 0) gidxs.push_back(global_plane_idx[(size_t) ip][tp]);
                            }
                        } else if (global_plane_idx[(size_t) ip][sd.pass] >= 0) {
                            gidxs.push_back(global_plane_idx[(size_t) ip][sd.pass]);
                        }
                        if (!gidxs.empty()) {
                            ggml_tensor * qt = ggml_cont(c, ggml_view_3d(
                                c, Q, D, Hq, q_cap_n, Q->nb[1], Q->nb[2],
                                (size_t) (sd.len - q_cap_n) * Q->nb[2]));
                            ggml_set_output(qt);
                            ggml_build_forward_expand(gf, qt);
                            qcap_list.push_back({ qt, std::move(gidxs) });
                        }
                    }
                    if (!L.is_swa && use_paged) {
                        ggml_tensor * kd_dep = ggml_inkling2_sconv_kv_write_paged(
                            c, K_raw, L.attn_k_conv, st, st, k_dyn, k_p, c_pos[si],
                            inp_bt_local, kf_bs, 0, D, Hk, 0, w.qk_clip, w.rms_eps, 1);
                        ggml_tensor * vd_dep = ggml_inkling2_sconv_kv_write_paged(
                            c, V_raw, L.attn_v_conv, st, st, v_dyn, v_p, c_pos[si],
                            inp_bt_local, kf_bs, kv_sz, D, Hk, 1, w.qk_clip, w.rms_eps, 1);
                        if (stage_t15 && ip < (int) tc.remote_prefix_k.size() && tc.remote_prefix_k[ip]) {
                            ggml_tensor * rk_p = plane_view_kv(c, tc.remote_prefix_k[ip], sd.pass);
                            ggml_tensor * rv_p = plane_view_kv(c, tc.remote_prefix_v[ip], sd.pass);
                            ggml_tensor * part_r = ggml_inkling2_attn_rel_paged(
                                c, Q, rk_p, rv_p, rt, c_meta_r[si], nullptr, nullptr, nullptr,
                                inp_bt_remote, kf_bs, w.rel_extent, /*partial=*/1, /*include_sink=*/1);
                            exp_nodes.push_back(part_r);
                            ggml_tensor * part_l = ggml_inkling2_attn_rel_paged(
                                c, Q, k_p, v_p, rt, c_meta[si], kd_dep, vd_dep, nullptr,
                                inp_bt_local, kf_bs, w.rel_extent, /*partial=*/1, /*include_sink=*/0);
                            ao_s[si] = ggml_inkling2_attn_split_combine(c, part_r, part_l, Gate, D, Hq);
                        } else {
                            ao_s[si] = ggml_inkling2_attn_rel_paged(
                                c, Q, k_p, v_p, rt, c_meta[si], kd_dep, vd_dep, Gate,
                                inp_bt_local, kf_bs, w.rel_extent, /*partial=*/0, /*include_sink=*/1);
                        }
                    } else {
                        ggml_tensor * kd_dep = ggml_inkling2_sconv_kv_write(
                            c, K_raw, L.attn_k_conv, st, st, k_dyn, k_p, c_pos[si],
                            0, D, Hk, 0, swa, w.qk_clip, w.rms_eps, 1);
                        ggml_tensor * vd_dep = ggml_inkling2_sconv_kv_write(
                            c, V_raw, L.attn_v_conv, st, st, v_dyn, v_p, c_pos[si],
                            kv_sz, D, Hk, 1, swa, w.qk_clip, w.rms_eps, 1);
                        ao_s[si] = ggml_inkling2_attn_rel(
                            c, Q, k_p, v_p, rt, c_meta[si], kd_dep, vd_dep, Gate, swa, w.rel_extent, 0, 1);
                    }
                }

                // O-proj across active Coda stream(s)
                ggml_tensor * wo_all = ggml_mul_mat(c, L.wo, concat_c(ao_s));

                ggml_tensor * ffn_inp[3]  = {};
                ggml_tensor * ffn_in_s[3] = {};
                for (int si = 0; si < c_n; ++si) {
                    const auto & sd = c_streams[si];
                    ggml_tensor * st = plane_view_conv(c, tc.prefix_conv[ip], sd.pass);
                    ggml_tensor * x  = nullptr;
                    if (ip % w.attn_res_block == 0) {
                        x = ggml_inkling2_sconv(c, slice_c(wo_all, si), L.attn_post_conv, st, st, nullptr, 2 * kv_sz, 0, 1);
                        c_blocks[si].push_back(inpSA[si]);
                    } else {
                        x = ggml_inkling2_sconv_add(c, slice_c(wo_all, si), L.attn_post_conv, st, st, nullptr, inpSA[si], 2 * kv_sz, 0, 1);
                    }
                    ffn_inp[si]  = x;
                    ffn_in_s[si] = ggml_inkling2_attn_res_mix(
                        c, x, L.ffn_res_query, L.ffn_norm,
                        c_blocks[si].empty() ? nullptr : c_blocks[si].data(), (int) c_blocks[si].size(), w.rms_eps);
                }

                // Router + 128-expert MoE across active Coda stream(s)
                ggml_tensor * ffn_cat = concat_c(ffn_in_s);
                ggml_tensor * rl   = ggml_mul_mat(c, L.ffn_gate_inp_q8 ? L.ffn_gate_inp_q8 : L.ffn_gate_inp, ffn_cat);
                ggml_tensor * sel  = ggml_inkling2_sigmoid_topk_ids(c, rl, L.ffn_exp_probs_b, w.n_expert_used);
                ggml_tensor * wsel = ggml_inkling2_sigmoid_topk_wts(c, rl, sel, w.route_scale, 1e-20f);
                ggml_tensor * x3   = ggml_reshape_3d(c, ffn_cat, w.n_embd, 1, C_tot);
                ggml_tensor * gu   = ggml_mul_mat_id(c, L.ffn_gate_up_exps, x3, sel);
                ggml_tensor * act  = ggml_inkling2_situ(c, gu, w.situ_beta, w.situ_linear_beta);
                ggml_tensor * dn   = ggml_mul_mat_id(c, L.ffn_down_exps, act, sel);
                ggml_tensor * moe_all = ggml_ds4_moe_fused_combine_shared(c, dn, wsel, nullptr);

                for (int si = 0; si < c_n; ++si) {
                    const auto & sd = c_streams[si];
                    ggml_tensor * st = plane_view_conv(c, tc.prefix_conv[ip], sd.pass);
                    c_cur[si] = ggml_inkling2_sconv_add(
                        c, slice_c(moe_all, si), L.ffn_post_conv, st, st, nullptr, ffn_inp[si], 2 * kv_sz + w.n_embd, 0, 1);
                    if (share_coda && sd.pass == 2) {
                        ggml_build_forward_expand(gf, make_tree_prefill_clone_coda(
                            c, tc.prefix_k[ip], tc.prefix_v[ip], tc.prefix_conv[ip], c_cur[si], sd.pos, sd.len, swa,
                            (!L.is_swa && use_paged) ? inp_bt_local : nullptr,
                            (!L.is_swa && use_paged) ? kf_bs : 0));
                    }
                }
            }

            for (int si = 0; si < c_n; ++si) {
                ggml_build_forward_expand(gf, c_cur[si]);
            }

            ggml_tensor * greedy_t    = nullptr;
            ggml_tensor * soft_t      = nullptr;
            ggml_tensor * inp_dr_pos  = nullptr;
            ggml_tensor * inp_dr_wpos = nullptr;
            int tail_beg = 0, tail_n = 0;
            const bool is_last_pass2 = (pass2_si >= 0 && c_streams[pass2_si].chunk_idx == M - 1);
            if (pass2_si >= 0) {
                const auto & sd2 = c_streams[pass2_si];
                const auto & cs2 = ring[sd2.ring_slot];
                tail_beg = std::max(0, tc.draft_pref_start - sd2.pos);
                tail_n   = std::max(0, sd2.len - tail_beg);
                if (tail_n > 0) {
                    auto slice_tail = [&](ggml_tensor * t) -> ggml_tensor * {
                        if (tail_beg == 0 && tail_n == sd2.len) return t;
                        return ggml_view_2d(c, t, t->ne[0], tail_n, t->nb[1], (size_t) tail_beg * t->nb[1]);
                    };
                    ggml_tensor * f01 = ggml_concat(c, slice_tail(cs2.tap_l2), slice_tail(cs2.core_taps[0]), 0);
                    ggml_tensor * f23 = ggml_concat(c, slice_tail(cs2.core_taps[1]), slice_tail(cs2.core_taps[2]), 0);
                    ggml_tensor * feats_tail = ggml_concat(c, ggml_concat(c, f01, f23, 0), slice_tail(tap_l22), 0);
                    inp_dr_pos  = ggml_new_tensor_1d(c, GGML_TYPE_I32, tail_n);
                    inp_dr_wpos = ggml_new_tensor_1d(c, GGML_TYPE_I32, tail_n);
                    ggml_set_input(inp_dr_pos);
                    ggml_set_input(inp_dr_wpos);
                    inkling2_2dsd_append_prefix_kv_graph(c, gf, w.twodsd, tc, feats_tail, inp_dr_pos, inp_dr_wpos);
                }

                if (is_last_pass2 && is_last_chunk) {
                    ggml_tensor * final_out_norm = ggml_inkling2_attn_res_mix(
                        c, c_cur[pass2_si], w.output_res_query, w.output_norm,
                        c_blocks[pass2_si].empty() ? nullptr : c_blocks[pass2_si].data(),
                        (int) c_blocks[pass2_si].size(), w.rms_eps);
                    ggml_tensor * lm_in = ggml_view_2d(
                        c, final_out_norm, w.n_embd, 1, final_out_norm->nb[1],
                        (size_t) (sd2.len - 1) * final_out_norm->nb[1]);
                    ggml_tensor * raw_l = ggml_mul_mat(c, w.output, lm_in);
                    soft_t = ggml_inkling2_softcap_slice(c, raw_l, w.n_vocab_unpadded, w.logit_softcap);
                    if (out_last_logits) ggml_set_output(soft_t);
                    greedy_t = ggml_argmax(c, soft_t);
                    ggml_set_output(greedy_t);
                    ggml_build_forward_expand(gf, greedy_t);
                }
            }

            ggml_backend_sched_t sched = nullptr;
            ggml_gallocr_t ga = nullptr;
            if (!alloc_stage_graph(gf, max_nodes, stage_t15, exp_nodes, exp_inputs, sched, ga)) {
                ggml_free(c);
                free_ring();
                return false;
            }
            if (use_paged && inp_bt_local) {
                const auto & btl = cache.kvflash->bt_local();
                ggml_backend_tensor_set(inp_bt_local, btl.data(), 0, btl.size() * sizeof(int32_t));
                if (stage_t15 && inp_bt_remote) {
                    const auto & btr = cache.kvflash->bt_remote();
                    ggml_backend_tensor_set(inp_bt_remote, btr.data(), 0, btr.size() * sizeof(int32_t));
                }
            }
            for (int si = 0; si < c_n; ++si) {
                const auto & sd = c_streams[si];
                std::vector<int32_t> pos_v(sd.len);
                for (int i = 0; i < sd.len; ++i) pos_v[i] = sd.pos + i;
                int32_t meta[4] = { sd.pos, sd.pos + sd.len, 0, sd.pos + sd.len };
                ggml_backend_tensor_set(c_pos[si],  pos_v.data(), 0, (size_t) sd.len * sizeof(int32_t));
                ggml_backend_tensor_set(c_meta[si], meta,         0, sizeof(meta));
                if (stage_t15 && c_meta_r[si]) {
                    ggml_backend_tensor_set(c_meta_r[si], meta, 0, sizeof(meta));
                }
            }
            if (pass2_si >= 0 && tail_n > 0) {
                const auto & sd2 = c_streams[pass2_si];
                std::vector<int32_t> dr_pos(tail_n), dr_wpos(tail_n);
                for (int i = 0; i < tail_n; ++i) {
                    const int raw_wpos = (sd2.pos + tail_beg - tc.draft_pref_start) + i;
                    dr_pos[i]  = sd2.pos + tail_beg + i;
                    dr_wpos[i] = (tc.draft_ring_cap > 0) ? (raw_wpos % tc.draft_ring_cap) : raw_wpos;
                }
                ggml_backend_tensor_set(inp_dr_pos,  dr_pos.data(),  0, (size_t) tail_n * sizeof(int32_t));
                ggml_backend_tensor_set(inp_dr_wpos, dr_wpos.data(), 0, (size_t) tail_n * sizeof(int32_t));
            }
            const bool ok = sched
                ? (ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS)
                : (ggml_backend_graph_compute(target_backend, gf) == GGML_STATUS_SUCCESS);
            if (ok) {
                for (const auto & qc : qcap_list) {
                    for (int gidx : qc.second) scatter_q_cap(qc.first, gidx);
                }
                if (pass2_si >= 0) {
                    const auto & sd2 = c_streams[pass2_si];
                    if (tail_n > 0) cache.dspark_ctx_pos = sd2.pos + sd2.len;
                    if (is_last_pass2 && is_last_chunk && greedy_t) {
                        ggml_backend_tensor_get(greedy_t, &out_greedy_tok, 0, sizeof(int32_t));
                    }
                    if (is_last_pass2 && is_last_chunk && out_last_logits && soft_t) {
                        out_last_logits->resize((size_t) w.n_vocab_unpadded);
                        ggml_backend_tensor_get(soft_t, out_last_logits->data(), 0, (size_t) w.n_vocab_unpadded * sizeof(float));
                    }
                }
            }
            if (sched) ggml_backend_sched_free(sched);
            if (ga)    ggml_gallocr_free(ga);
            ggml_free(c);
            if (!ok) { free_ring(); return false; }
        }
    }

    if (use_paged && q_cap_n > 0 && !q_cap_host.empty()) {
        cache.kvflash->append_queries(q_cap_host.data(), q_cap_n);
    }
    tc.prefix_len = pos0 + n_tokens;
    cache.cur_pos = pos0 + n_tokens;
    free_ring();
    return true;
}

// ─── Advance Active Tree Nodes by +1 Loop (Single-Slot & Multi-Slot Batched) ─
bool inkling2_2dsd_advance_nodes_batched(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & multi_cache,
        const std::vector<Inkling2TwoDSDNodeDesc> & nodes,
        const int * prefix_lens,
        int n_seq_slots,
        uint32_t readout_mask,
        bool allow_split,
        std::vector<int> & out_readout_rows,
        std::vector<int32_t> & out_readout_argmax,
        Inkling2TwoDSDAdvanceSampleOut * out_sample,
        const Inkling2KvFlash * const * slot_kfs) {
    out_readout_rows.clear();
    out_readout_argmax.clear();
    if (out_sample) {
        out_sample->top_cands.clear();
        out_sample->logits_dev = nullptr;
        out_sample->n_vocab = w.n_vocab_unpadded;
    }
    if (!multi_cache.tree.active || nodes.empty() || n_seq_slots <= 0) return false;
    auto & tc = multi_cache.tree;

    // 1. Prelude for fresh nodes (depth == 0)
    std::vector<const Inkling2TwoDSDNodeDesc *> fresh_rows;
    std::vector<int> fresh_planes, fresh_flags, fresh_extra;
    for (const auto & nd : nodes) {
        if (nd.depth == 0) {
            fresh_rows.push_back(&nd);
            fresh_planes.push_back(0);
            fresh_flags.push_back(0);
            fresh_extra.push_back(nd.token);
        }
    }
    if (!fresh_rows.empty()) {
        auto st = stage_phase_rows(fresh_rows, fresh_planes, fresh_flags, fresh_extra,
                                   tc.max_nodes, prefix_lens, n_seq_slots, /*n_planes_phase=*/1);
        if (!run_phase(target_backend, w, multi_cache, /*phase_id=*/0, st,
                       prefix_lens, n_seq_slots, (int) fresh_rows.size(), /*has_later=*/false, allow_split,
                       nullptr, nullptr, slot_kfs)) {
            return false;
        }
    }

    // 2. Fused Multi-Pass Core for ALL active nodes (passes 0, 1, 2 together!)
    std::vector<const Inkling2TwoDSDNodeDesc *> all_rows;
    std::vector<int> all_planes, all_flags, all_extra;
    all_rows.reserve(nodes.size());
    all_planes.reserve(nodes.size());
    all_flags.reserve(nodes.size());
    all_extra.reserve(nodes.size());
    bool any_later = false;
    for (const auto & nd : nodes) {
        all_rows.push_back(&nd);
        all_planes.push_back(nd.depth); // pass 0, 1, or 2
        const int flg = (nd.depth > 0) ? kInkling2TwoDSDFlagsLater : 0;
        if (nd.depth > 0) any_later = true;
        all_flags.push_back(flg);
        all_extra.push_back(std::max(0, nd.depth - 1));
    }
    {
        auto st = stage_phase_rows(all_rows, all_planes, all_flags, all_extra,
                                   tc.max_nodes, prefix_lens, n_seq_slots, /*n_planes_phase=*/w.n_passes);
        if (!run_phase(target_backend, w, multi_cache, /*phase_id=*/1, st,
                       prefix_lens, n_seq_slots, (int) all_rows.size(), any_later, allow_split,
                       nullptr, nullptr, slot_kfs)) {
            return false;
        }
    }

    // 3. Fused Multi-Depth Coda for nodes reaching a readout depth (depth + 1 in readout_mask)
    std::vector<const Inkling2TwoDSDNodeDesc *> read_rows;
    std::vector<int> read_planes, read_flags, read_extra;
    for (int i = 0; i < (int) nodes.size(); ++i) {
        const int new_depth = nodes[i].depth + 1;
        if ((readout_mask & (1u << new_depth)) != 0u) {
            out_readout_rows.push_back(i);
            read_rows.push_back(&nodes[i]);
            read_planes.push_back(new_depth - 1); // 0, 1, or 2
            read_flags.push_back(0);
            read_extra.push_back(new_depth - 1);
        }
    }
    if (!read_rows.empty()) {
        auto st = stage_phase_rows(read_rows, read_planes, read_flags, read_extra,
                                   tc.max_nodes, prefix_lens, n_seq_slots, /*n_planes_phase=*/tc.n_depths);
        std::vector<int32_t> greedy_out;
        if (!run_phase(target_backend, w, multi_cache, /*phase_id=*/2, st,
                       prefix_lens, n_seq_slots, (int) read_rows.size(),
                       /*has_later=*/false, allow_split, &greedy_out, out_sample, slot_kfs)) {
            return false;
        }
        out_readout_argmax.assign(greedy_out.begin(), greedy_out.end());
    }
    return true;
}

bool inkling2_2dsd_advance_nodes(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        const std::vector<Inkling2TwoDSDNodeDesc> & nodes,
        uint32_t readout_mask,
        bool allow_split,
        std::vector<int> & out_readout_rows,
        std::vector<int32_t> & out_readout_argmax,
        Inkling2TwoDSDAdvanceSampleOut * out_sample) {
    const int pref_len = cache.tree.prefix_len;
    const Inkling2KvFlash * kf0 = (cache.kvflash && cache.kvflash->active()) ? cache.kvflash.get() : nullptr;
    const Inkling2KvFlash * const * kfs_ptr = kf0 ? &kf0 : nullptr;
    return inkling2_2dsd_advance_nodes_batched(
        target_backend, w, cache, nodes, &pref_len, 1, readout_mask, allow_split,
        out_readout_rows, out_readout_argmax, out_sample, kfs_ptr);
}

// ─── Commit Verified Path from Tree -> 60-Plane Prefix & 5-Layer Drafter KV ──
bool inkling2_2dsd_commit_nodes(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        const int * commit_slots,
        int n_commit) {
    if (!cache.tree.active || !commit_slots || n_commit <= 0) return false;
    if (n_commit > kInkling2TwoDSDMaxCommit) {
        for (int off = 0; off < n_commit; off += kInkling2TwoDSDMaxCommit) {
            const int m = std::min(kInkling2TwoDSDMaxCommit, n_commit - off);
            if (!inkling2_2dsd_commit_nodes(target_backend, w, cache, commit_slots + off, m)) return false;
        }
        return true;
    }

    auto & tc = cache.tree;
    const bool use_paged = (cache.kvflash && cache.kvflash->active());
    const int  kf_bs     = use_paged ? cache.kvflash->block_size() : 0;
    if (use_paged) {
        if (!cache.kvflash->prepare_span(tc.prefix_len, n_commit)) return false;
    }

    auto & cg = tc.commit_graph;
    if (!cg.built) {
        const size_t max_nodes = (size_t) w.n_layer_phys * 16 + 64;
        struct ggml_init_params gp = { max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false) + 65536, nullptr, true };
        cg.ctx = ggml_init(gp);
        cg.gf  = ggml_new_graph_custom(cg.ctx, max_nodes, false);
        cg.slots = ggml_new_tensor_1d(cg.ctx, GGML_TYPE_I32, kInkling2TwoDSDMaxCommit);
        cg.meta  = ggml_new_tensor_1d(cg.ctx, GGML_TYPE_I32, 2);
        cg.block_table = use_paged ? ggml_new_tensor_1d(cg.ctx, GGML_TYPE_I32, cache.kvflash->max_blocks()) : nullptr;
        ggml_set_input(cg.slots);
        ggml_set_input(cg.meta);
        if (cg.block_table) ggml_set_input(cg.block_table);

        for (int ip = 0; ip < w.n_layer_phys; ++ip) {
            const int swa = w.layers[ip].is_swa ? w.swa_window : 0;
            const bool shared_global = (!w.layers[ip].is_swa && tc.shared_kv_window > 0 && tc.planes[ip] > 1);
            ggml_tensor * op = make_tree_commit(
                cg.ctx, tc.prefix_k[ip], tc.prefix_v[ip], tc.tree_k[ip], tc.tree_v[ip],
                tc.prefix_conv[ip], tc.tree_conv[ip], cg.slots, cg.meta, swa,
                (!w.layers[ip].is_swa && use_paged) ? cg.block_table : nullptr,
                (!w.layers[ip].is_swa && use_paged) ? kf_bs : 0,
                shared_global ? tc.shared_kv_window : 0,
                shared_global ? tc.shared_tail_cap : 0);
            ggml_build_forward_expand(cg.gf, op);
        }
        cg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
        if (!cg.alloc || !ggml_gallocr_alloc_graph(cg.alloc, cg.gf)) {
            if (cg.alloc) { ggml_gallocr_free(cg.alloc); cg.alloc = nullptr; }
            ggml_free(cg.ctx); cg.ctx = nullptr;
            cg.gf = nullptr;
            return false;
        }
        cg.built = true;
    }

    int32_t slots_pad[kInkling2TwoDSDMaxCommit] = {};
    for (int i = 0; i < n_commit; ++i) slots_pad[i] = commit_slots[i];
    int32_t meta[2] = { tc.prefix_len, n_commit };
    ggml_backend_tensor_set(cg.slots, slots_pad, 0, sizeof(slots_pad));
    ggml_backend_tensor_set(cg.meta,  meta,      0, sizeof(meta));
    if (use_paged && cg.block_table) {
        const auto & btl = cache.kvflash->bt_local();
        ggml_backend_tensor_set(cg.block_table, btl.data(), 0, btl.size() * sizeof(int32_t));
    }

    if (ggml_backend_graph_compute_async(target_backend, cg.gf) != GGML_STATUS_SUCCESS) {
        return false;
    }
    if (!inkling2_2dsd_commit_prefix_kv_from_slots(target_backend, w, cache, commit_slots, n_commit, tc.prefix_len)) {
        return false;
    }

    tc.prefix_len += n_commit;
    cache.cur_pos  = tc.prefix_len;
    return true;
}

} // namespace luce::common
