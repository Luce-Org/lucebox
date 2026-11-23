// Inkling2-Loop forward, verify-rollback, and bundled DSpark draft graphs.
//
// Exploits:
//   - Load-time fused attention projections (Q|K|V|Gate|R|Sink & Q|Gate|R|Sink)
//   - Zero-kq_b inline relative-position + zero-sink Wave32 FlashAttention / Split-KV
//   - Fused Block AttnRes mix, Loop Cross-Pass controller, 4-tap (+dynamic) ShortConv,
//     SITU MoE activation, and Sigmoid Top-K router (all HIP-graph safe!)
//   - Dual-GPU R9700 + Strix Halo dynamic 74/26 parallel MoE experts + batched copy_tail +
//     79/21 LM-head split + Split-KV Global Attention offload
//   - 3-Pass Wavefront Loop-Fused Prefill (auto chunk=1024) + O(1) parallel ShortConv
//   - Native bundled DSpark (`mtp.safetensors`): 3-layer DFlare bidirectional block-8
//     transformer + exact-n_spec fused F16 RNN Markov chain + on-device masks in ONE HIP graph!

#include "inkling2_graph.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace luce::common {
namespace {

static constexpr size_t kInkling2PrefetchBytes = (size_t) 56 << 20;

static ggml_tensor * row_view_2d(ggml_context * c, ggml_tensor * src, int64_t row_off, int64_t n_rows) {
    return ggml_view_2d(c, src, n_rows, src->ne[1], src->nb[1], (size_t) row_off * src->nb[0]);
}

static ggml_tensor * rms_norm_w(ggml_context * c, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
}

static inline void set_op_i32(ggml_tensor * t, int idx, int32_t v) {
    std::memcpy(&t->op_params[idx], &v, sizeof(v));
}
static inline void set_op_f32(ggml_tensor * t, int idx, float v) {
    std::memcpy(&t->op_params[idx], &v, sizeof(v));
}

// Fused Q8_0 router (exact same MMVQ/MMQ Q8_1 arithmetic for T==1 and T>1 -> 100% AR/DSpark parity!)
// + zero-spill sigmoid top-K (+ optional Dual-GPU split!) (subop 24):
// target_pct == 0: returns I32 [top_k, T, 2] -> slice 0 = ids, slice 1 = __float_as_int(wts).
// target_pct >  0: returns I32 [top_k, T, 3] -> slice 0 = hot_sel, slice 1 = cold_sel, slice 2 = __float_as_int(wts).
static ggml_tensor * inkling2_fused_router_q8_topk(
        ggml_context * c, ggml_tensor * w_q8, ggml_tensor * cur, ggml_tensor * bias,
        int top_k, float route_scale, float eps, int target_pct = 0) {
    const int64_t T = cur->ne[1];
    ggml_tensor * logits = ggml_mul_mat(c, w_q8, cur);
    ggml_tensor * res = ggml_new_tensor_3d(c, GGML_TYPE_I32, top_k, T, target_pct > 0 ? 3 : 2);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = logits;
    res->src[1] = nullptr;
    res->src[2] = bias;
    set_op_i32(res, 0, 24);
    set_op_i32(res, 1, top_k);
    set_op_f32(res, 2, route_scale);
    set_op_f32(res, 3, eps);
    set_op_i32(res, 4, target_pct);
    return res;
}

// Fused Q-clip + R-clip + rel_table in ONE launch (subop 31, eliminates 48 launches/step across AR/DSpark/Prefill!)
static void inkling2_clip_qr_rel_table(
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

    *out_q = ggml_view_3d(
        c, buf, head_dim, n_head, T,
        (size_t) head_dim * sizeof(float),
        (size_t) head_dim * n_head * sizeof(float), 0);
    *out_rt = ggml_view_3d(
        c, buf, rel_extent + 1, n_head, T,
        (size_t) (rel_extent + 1) * sizeof(float),
        (size_t) (rel_extent + 1) * n_head * sizeof(float),
        (size_t) q_elems * sizeof(float));
}

// Fused K+V SConv + K-clip/V-RMSNorm + Direct KV Cache Write in 1 launch (subop 27, SWA layers)!
static ggml_tensor * inkling2_sconv_kv_write_kv(
        ggml_context * c,
        ggml_tensor * x_k, ggml_tensor * x_v,
        ggml_tensor * wk_static, ggml_tensor * wv_static,
        ggml_tensor * state_in, ggml_tensor * state_out,
        ggml_tensor * k_cache, ggml_tensor * v_cache,
        ggml_tensor * pos_buf,
        int head_dim, int n_head_kv, int swa_window, float threshold, float eps, int write_mode) {
    ggml_tensor * res = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = x_k;
    res->src[1] = x_v;
    res->src[2] = wk_static;
    res->src[3] = wv_static;
    res->src[4] = state_in;
    res->src[5] = state_out;
    res->src[6] = k_cache;
    res->src[7] = v_cache;
    res->src[8] = pos_buf;
    set_op_i32(res, 0, 27);
    set_op_i32(res, 1, head_dim);
    set_op_i32(res, 2, n_head_kv);
    set_op_i32(res, 3, swa_window);
    set_op_f32(res, 4, threshold);
    set_op_f32(res, 5, eps);
    set_op_i32(res, 6, write_mode);
    return res;
}

// Fused MoE Masked Sum (+ optional peer join) + FFN Post-Conv + Residual Add (subop 28)!
static ggml_tensor * inkling2_moe_masked_sum_sconv_add(
        ggml_context * c,
        ggml_tensor * down, ggml_tensor * wsel, ggml_tensor * mask_sel, ggml_tensor * peer_sum,
        ggml_tensor * w_static, ggml_tensor * state_in, ggml_tensor * state_out, ggml_tensor * addend,
        int state_chan_offset, int write_mode) {
    ggml_tensor * res = ggml_new_tensor_2d(c, GGML_TYPE_F32, down->ne[0], down->ne[2]);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = down;
    res->src[1] = wsel;
    res->src[2] = mask_sel;
    res->src[3] = peer_sum;
    res->src[4] = w_static;
    res->src[5] = state_in;
    res->src[6] = state_out;
    res->src[7] = addend;
    set_op_i32(res, 0, 28);
    set_op_i32(res, 1, state_chan_offset);
    set_op_i32(res, 2, write_mode);
    return res;
}

// Fused DSpark Markov RNN Step: get_rows(w1)+concat+F16 gemv+add(ctx)+bf16 gate in 1 kernel (subop 29)!
static ggml_tensor * inkling2_dspark_markov_rnn_step(
        ggml_context * c,
        ggml_tensor * prev_tok, ggml_tensor * state,
        ggml_tensor * markov_w1, ggml_tensor * rnn_proj_step, ggml_tensor * rnn_ctx_s) {
    ggml_tensor * res = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2 * state->ne[0], 1);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = prev_tok;
    res->src[1] = state;
    res->src[2] = markov_w1;
    res->src[3] = rnn_proj_step;
    res->src[4] = rnn_ctx_s;
    set_op_i32(res, 0, 29);
    return res;
}

// On-device DSpark global + local SWA mask builder (subop 30, zero CPU fill / zero PCIe upload at 262K!)
static ggml_tensor * inkling2_dspark_build_masks(
        ggml_context * c, ggml_tensor * draft_pos, int64_t kv_span, int B, int swa_window) {
    ggml_tensor * res = ggml_new_tensor_3d(c, GGML_TYPE_F16, kv_span, B, 2);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = draft_pos;
    set_op_i32(res, 0, 30);
    set_op_i32(res, 1, swa_window);
    return res;
}

// Fused 64-CU loop controller pre-norm + Q8_0 down-projection (subop 25):
// Replaces loop_ctrl_pre + quantize_mmq + 2-CU ctrl_down with ONE 64-CU kernel!
static ggml_tensor * inkling2_fused_loop_ctrl_pre_down(
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

// Fused Dual-GPU dynamic split LUT + hot/cold route masking (subop 26):
// Returns I32 [top_k, T, 2] where slice 0 = hot_sel, slice 1 = cold_sel (replaces 3 launches with 1!).
static ggml_tensor * inkling2_dynamic_split_routes(
        ggml_context * c, ggml_tensor * sel, int n_expert, int target_pct) {
    ggml_tensor * res = ggml_new_tensor_3d(c, GGML_TYPE_I32, sel->ne[0], sel->ne[1], 2);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = sel;
    set_op_i32(res, 0, 26);
    set_op_i32(res, 1, n_expert);
    set_op_i32(res, 2, target_pct);
    return res;
}

// Prefetch next physical layer's attention & router weights into R9700's 64 MB Infinity Cache
// while R9700 waits for Strix Halo's cold expert picks.
static void prefetch_next_layer_weights(ggml_context * c, ggml_cgraph * gf,
                                        ggml_tensor * after, int visit,
                                        const Inkling2Weights & w) {
    if (!w.buf) return;
    std::vector<ggml_tensor *> next;
    if (visit + 1 < w.n_layer_eff) {
        const auto & nv = w.schedule[visit + 1];
        const auto & NL = w.layers[nv.phys];
        if (nv.pass == 0 && NL.wqkvgrs) {
            next = { NL.wqkvgrs, NL.wo, NL.ffn_gate_inp };
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
        if (n == GGML_MAX_SRC - 1 || bytes >= kInkling2PrefetchBytes) break;
        srcs[n++] = t;
        bytes += ggml_nbytes(t);
    }
    if (n > 0) {
        ggml_build_forward_expand(gf, ggml_moe_fused_prefetch(c, after, srcs, n, kInkling2PrefetchBytes));
    }
}

static bool build_inkling2_target_graph(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        Inkling2GraphWorkspace & ws,
        int n_tokens,
        bool verify,
        bool use_split,
        bool use_split_kv,
        bool capture_taps) {
    free_inkling2_workspace(ws);
    ws.n_tokens     = n_tokens;
    ws.verify       = verify;
    ws.split        = use_split;
    ws.split_kv     = use_split_kv;
    ws.capture_taps = capture_taps;

    const size_t max_nodes = 16384;
    const size_t ctx_bytes = max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false) + (16 << 20);
    struct ggml_init_params gp = { ctx_bytes, nullptr, true };
    ws.ctx = ggml_init(gp);
    if (!ws.ctx) return false;

    ggml_context * c = ws.ctx;
    ggml_cgraph * gf = ggml_new_graph_custom(c, max_nodes, false);
    ws.gf = gf;

    ws.inp_tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
    ws.inp_pos    = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
    ws.inp_meta   = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
    ggml_set_input(ws.inp_tokens);
    ggml_set_input(ws.inp_pos);
    ggml_set_input(ws.inp_meta);
    if (use_split_kv) {
        ws.inp_meta_r = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
        ggml_set_input(ws.inp_meta_r);
    }

    const int write_mode = verify ? 2 : 1; // 2 = write verify conv ring; 1 = update live conv state
    const int Hq = w.n_head;
    const int Hk = w.n_head_kv;
    const int D  = w.head_dim;
    const int q_sz  = Hq * D;
    const int kv_sz = Hk * D;
    const int r_sz  = w.d_rel * Hq;

    // Input token embedding + tok_norm
    ggml_tensor * cur = ggml_get_rows(c, w.tok_embd, ws.inp_tokens);
    cur = rms_norm_w(c, cur, w.tok_norm, w.rms_eps);

    std::vector<ggml_tensor *> blocks;
    blocks.reserve(8);
    ggml_tensor * prelude = nullptr;
    std::vector<ggml_tensor *> prelude_blocks;

    std::vector<ggml_tensor *> bank;
    ggml_tensor * bank_out = nullptr;

    ggml_tensor * captured_taps[4] = {nullptr, nullptr, nullptr, nullptr};
    int cur_pass = -1;

    for (int v = 0; v < w.n_layer_eff; ++v) {
        const auto & vi = w.schedule[v];
        const int il = vi.slot;
        const int ip = vi.phys;
        const int pass = vi.pass;
        const auto & L = w.layers[ip];

        // Save prelude state right before core pass 0 starts
        if (w.is_loop && v == w.core_start) {
            prelude = cur;
            prelude_blocks = blocks;
            cur_pass = 0;
        }
        // Reset residual stream at the start of loop pass 1 and pass 2
        if (w.is_loop && vi.is_core && pass > 0 && pass != cur_pass) {
            bank.assign(blocks.begin() + (ptrdiff_t) prelude_blocks.size() + 1, blocks.end());
            bank_out = cur;
            cur = prelude;
            blocks = prelude_blocks;
            cur_pass = pass;
        }

        // Capture DSpark target taps (target_layer_ids = [5, 23, 41, 48])
        if (capture_taps) {
            for (int ti = 0; ti < 3 && ti < (int) w.dspark.target_layer_ids.size(); ++ti) {
                if (v == w.dspark.target_layer_ids[ti]) {
                    captured_taps[ti] = cur;
                }
            }
        }

        ggml_tensor * inpSA = cur;
        ggml_tensor * state_live = cache.conv_state[il];
        ggml_tensor * state_dst  = verify ? cache.conv_ring[il] : cache.conv_state[il];

        // Cross-pass messages on loop pass > 0
        ggml_tensor * messages = nullptr;
        const Inkling2LoopLayer * loop = nullptr;
        if (pass > 0) {
            loop = &w.loop_layers[ip - w.core_start];
            messages = ggml_inkling2_attn_res_mix(
                c, bank_out, loop->msg_query, nullptr,
                bank.empty() ? nullptr : bank.data(), (int) bank.size(), w.rms_eps);
            // messages shape: [n_embd, 4, T]
        }
        auto get_msg = [&](int ctrl) -> ggml_tensor * {
            return ggml_view_2d(c, messages, w.n_embd, n_tokens,
                                messages->nb[2], (size_t) ctrl * messages->nb[1]);
        };

        // ─── Attention Input Mix + Loop Controller + RMSNorm ────────────
        ggml_tensor * inputs[3] = {cur, cur, cur};
        const bool same_qkv_input = L.is_swa && (messages == nullptr);

        if (messages == nullptr) {
            if (L.attn_res_query == nullptr) {
                // Layer 0: no prior blocks, just attn_norm
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
                    inputs[j] = ggml_view_2d(c, mixed3, w.n_embd, n_tokens,
                                             mixed3->nb[2], (size_t) j * mixed3->nb[1]);
                }
            }
        } else {
            // Pass > 0: mix without norm, apply batched 3-way Q/K/V loop controller + fused norm in 4 launches (instead of 12)!
            ggml_tensor * pre_qkv = cur;
            if (L.attn_res_query) {
                ggml_tensor * mixed = ggml_inkling2_attn_res_mix(
                    c, cur, L.attn_res_query, nullptr,
                    blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
                pre_qkv = L.is_swa ? mixed : ggml_permute(c, mixed, 0, 2, 1, 3); // [H, T, 1] or [H, T, 3]
            }
            ggml_tensor * msg_qkv_v = ggml_view_3d(c, messages, w.n_embd, 3, n_tokens, messages->nb[1], messages->nb[2], 0);
            ggml_tensor * msg_qkv   = ggml_permute(c, msg_qkv_v, 0, 2, 1, 3); // [H, T, 3]
            ggml_tensor * nw_qkv    = L.is_swa ? L.attn_norm : L.attn_qkv_norm; // [H] or [H, 3]

            ggml_tensor * gate = (loop->ctrl_down_qkv->type == GGML_TYPE_Q8_0 && n_tokens <= 16)
                ? inkling2_fused_loop_ctrl_pre_down(c, pre_qkv, msg_qkv, loop->ctrl_down_qkv, w.rms_eps)
                : ggml_mul_mat(c, loop->ctrl_down_qkv, ggml_inkling2_loop_ctrl_pre(c, pre_qkv, msg_qkv, w.rms_eps)); // [64, T, 3]
            ggml_tensor * gf_o = ggml_mul_mat(c, loop->ctrl_gf_qkv, gate);                     // [2H, T, 3]
            ggml_tensor * qkv_in = ggml_inkling2_loop_ctrl_post(c, pre_qkv, msg_qkv, gf_o, nw_qkv, w.rms_eps); // [H, T, 3]
            for (int j = 0; j < 3; ++j) {
                inputs[j] = ggml_view_2d(c, qkv_in, w.n_embd, n_tokens, qkv_in->nb[1], (size_t) j * qkv_in->nb[2]);
            }
        }

        // ─── Self-Attention Projections (Fused Q|K|V|Gate|R|Sink) ───────
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
                k_dyn = row_view_2d(c, kd, kv_sz, w.sconv_kernel * Hk);
                ggml_tensor * vd = ggml_mul_mat(c, L.wv_dyn, inputs[2]);
                V_raw = row_view_2d(c, vd, 0, kv_sz);
                v_dyn = row_view_2d(c, vd, kv_sz, w.sconv_kernel * Hk);
            } else {
                K_raw = ggml_mul_mat(c, L.wk, inputs[1]);
                V_raw = ggml_mul_mat(c, L.wv, inputs[2]);
                if (!L.is_swa) {
                    k_dyn = ggml_mul_mat(c, L.attn_k_conv_proj, inputs[1]);
                    v_dyn = ggml_mul_mat(c, L.attn_v_conv_proj, inputs[2]);
                }
            }
        }

        // Fused Q-clip + R-clip + inline relative-position table in ONE launch (-48 launches/step!)
        ggml_tensor * Q = nullptr;
        ggml_tensor * rel_table = nullptr;
        inkling2_clip_qr_rel_table(
            c, Q_raw, L.attn_q_clip, R_raw, L.attn_rel_clip,
            L.is_swa ? nullptr : ws.inp_pos, L.attn_rel_bias, Sink,
            D, w.d_rel, Hq, w.qk_clip, w.rel_clip, w.rms_eps,
            L.is_swa ? 0 : w.log_scale_floor, L.is_swa ? 0.0f : w.log_scale_alpha,
            &Q, &rel_table);

        // K/V causal short conv + K-clip/V-RMSNorm + direct KV cache write + Inline Relative-Bias Attention
        const int swa = L.is_swa ? w.swa_window : 0;
        const int gidx = w.slot_to_global_idx[il];
        ggml_tensor * attn_out = nullptr;

        if (!L.is_swa && use_split_kv && gidx >= 0 && cache.remote_backend && cache.remote_k[gidx]) {
            ggml_tensor * K_conv = ggml_inkling2_sconv(
                c, K_raw, L.attn_k_conv, state_live, state_dst, k_dyn, 0, D, write_mode);
            ggml_tensor * V_conv = ggml_inkling2_sconv(
                c, V_raw, L.attn_v_conv, state_live, state_dst, v_dyn, kv_sz, D, write_mode);
            // Dual-GPU Split-KV Global Attention:
            ggml_tensor * k_dep_r = ggml_inkling2_kv_write(
                c, K_conv, cache.remote_k[gidx], ws.inp_pos, D, Hk, 0, 0, w.qk_clip, w.rms_eps);
            ggml_tensor * v_dep_r = ggml_inkling2_kv_write(
                c, V_conv, cache.remote_v[gidx], ws.inp_pos, D, Hk, 1, 0, w.qk_clip, w.rms_eps);
            ggml_tensor * part_r = ggml_inkling2_attn_rel(
                c, Q, cache.remote_k[gidx], cache.remote_v[gidx], rel_table, ws.inp_meta_r,
                k_dep_r, v_dep_r, nullptr, 0, w.rel_extent, /*partial=*/1, /*include_sink=*/1);
            ws.expert_nodes.insert(ws.expert_nodes.end(), {k_dep_r, v_dep_r, part_r});

            ggml_tensor * k_dep_l = ggml_inkling2_kv_write(
                c, K_conv, cache.k_cache[il], ws.inp_pos, D, Hk, 0, -cache.remote_split_pos, w.qk_clip, w.rms_eps);
            ggml_tensor * v_dep_l = ggml_inkling2_kv_write(
                c, V_conv, cache.v_cache[il], ws.inp_pos, D, Hk, 1, -cache.remote_split_pos, w.qk_clip, w.rms_eps);
            ggml_tensor * part_l = ggml_inkling2_attn_rel(
                c, Q, cache.k_cache[il], cache.v_cache[il], rel_table, ws.inp_meta,
                k_dep_l, v_dep_l, nullptr, 0, w.rel_extent, /*partial=*/1, /*include_sink=*/0);
            attn_out = ggml_inkling2_attn_split_combine(c, part_r, part_l, Gate, D, Hq);
        } else if (L.is_swa && !k_dyn && !v_dyn && n_tokens <= 8) {
            // Fused K+V SConv + K-clip/V-RMSNorm + Direct KV Write in ONE launch (-40 launches/step!)
            ggml_tensor * kv_dep = inkling2_sconv_kv_write_kv(
                c, K_raw, V_raw, L.attn_k_conv, L.attn_v_conv, state_live, state_dst,
                cache.k_cache[il], cache.v_cache[il], ws.inp_pos,
                D, Hk, swa, w.qk_clip, w.rms_eps, write_mode);
            attn_out = ggml_inkling2_attn_rel(
                c, Q, cache.k_cache[il], cache.v_cache[il], rel_table, ws.inp_meta,
                kv_dep, kv_dep, Gate, swa, w.rel_extent, /*partial=*/0, /*include_sink=*/1);
        } else {
            // Fused SConv + K-clip/V-RMSNorm + direct KV write (eliminates 96 launches/tok + zero K_conv/V_conv!)
            ggml_tensor * k_dep = ggml_inkling2_sconv_kv_write(
                c, K_raw, L.attn_k_conv, state_live, state_dst, k_dyn, cache.k_cache[il], ws.inp_pos,
                0, D, Hk, 0, swa, w.qk_clip, w.rms_eps, write_mode);
            ggml_tensor * v_dep = ggml_inkling2_sconv_kv_write(
                c, V_raw, L.attn_v_conv, state_live, state_dst, v_dyn, cache.v_cache[il], ws.inp_pos,
                kv_sz, D, Hk, 1, swa, w.qk_clip, w.rms_eps, write_mode);
            attn_out = ggml_inkling2_attn_rel(
                c, Q, cache.k_cache[il], cache.v_cache[il], rel_table, ws.inp_meta,
                k_dep, v_dep, Gate, swa, w.rel_extent, /*partial=*/0, /*include_sink=*/1);
        }

        // Output proj + attention post-conv (fused with residual add when not starting a new block!)
        cur = ggml_mul_mat(c, L.wo, attn_out);
        if (ip % w.attn_res_block == 0) {
            cur = ggml_inkling2_sconv(
                c, cur, L.attn_post_conv, state_live, state_dst, nullptr, 2 * kv_sz, 0, write_mode);
            blocks.push_back(inpSA);
        } else {
            cur = ggml_inkling2_sconv_add(
                c, cur, L.attn_post_conv, state_live, state_dst, nullptr, inpSA, 2 * kv_sz, 0, write_mode);
        }

        ggml_tensor * ffn_inp = cur;

        // ─── MoE FFN Input Mix + Optional Loop Controller + RMSNorm ─────
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
            ggml_tensor * gate = (cd_w->type == GGML_TYPE_Q8_0 && n_tokens <= 16)
                ? inkling2_fused_loop_ctrl_pre_down(c, cur, mf, cd_w, w.rms_eps)
                : ggml_mul_mat(c, cd_w, ggml_inkling2_loop_ctrl_pre(c, cur, mf, w.rms_eps));
            ggml_tensor * gf_o = ggml_mul_mat(c, loop->ctrl_gf[Inkling2LoopLayer::CTRL_FFN], gate);
            cur = ggml_inkling2_loop_ctrl_post(c, cur, mf, gf_o, L.ffn_norm, w.rms_eps);
        }

        // ─── Sigmoid Top-8 Router + MoE SITU FFN (Single or Dual-GPU Parallel) ─
        const bool do_split = use_split && w.expert_backend && w.split_exp > 0 && w.split_exp < w.n_expert && L.exp_gate_up_tail;
        const bool use_dyn_split = do_split && (n_tokens <= 16) &&
            (std::getenv("LUCE_INKLING2_DYNAMIC_SPLIT") == nullptr ||
             std::atoi(std::getenv("LUCE_INKLING2_DYNAMIC_SPLIT")) != 0);
        const bool use_fused_router = (L.ffn_gate_inp_q8 != nullptr && n_tokens <= 16);
        const bool use_fused_router_split = use_fused_router && use_dyn_split && w.n_expert == 128 && w.n_expert_used == 8;
        // Separate parallel-T moe_masked_sum (3.2 us) + sconv_add (3.4 us) beats serial-T fused kernel on T>1;
        // fuse ONLY for T==1 AR where there is zero T-parallelism to lose (-48 launches/tok!).
        const bool use_fused_moe_sconv = use_fused_router && n_tokens == 1 && w.n_expert_used <= 8;

        ggml_tensor * sel  = nullptr;
        ggml_tensor * wsel = nullptr;
        ggml_tensor * hot_sel  = nullptr;
        ggml_tensor * cold_sel = nullptr;
        if (use_fused_router_split) {
            // Fused SDOT4 router + zero-spill Top-8 + Dual-GPU O(1) histogram/ballot split (-48 launches/step!)
            ggml_tensor * packed = inkling2_fused_router_q8_topk(
                c, L.ffn_gate_inp_q8, cur, L.ffn_exp_probs_b, w.n_expert_used, w.route_scale, 1e-20f, w.split_pct);
            hot_sel  = ggml_view_2d(c, packed, w.n_expert_used, n_tokens, packed->nb[1], 0);
            cold_sel = ggml_view_2d(c, packed, w.n_expert_used, n_tokens, packed->nb[1], packed->nb[2]);
            wsel     = ggml_view_2d(c, packed, w.n_expert_used, n_tokens, packed->nb[1], 2 * packed->nb[2]);
        } else if (use_fused_router) {
            ggml_tensor * packed = inkling2_fused_router_q8_topk(
                c, L.ffn_gate_inp_q8, cur, L.ffn_exp_probs_b, w.n_expert_used, w.route_scale, 1e-20f, 0);
            sel  = ggml_view_2d(c, packed, w.n_expert_used, n_tokens, packed->nb[1], 0);
            wsel = ggml_view_2d(c, packed, w.n_expert_used, n_tokens, packed->nb[1], packed->nb[2]);
        } else {
            ggml_tensor * r_logits = ggml_mul_mat(c, L.ffn_gate_inp_q8 ? L.ffn_gate_inp_q8 : L.ffn_gate_inp, cur);
            sel  = ggml_inkling2_sigmoid_topk_ids(c, r_logits, L.ffn_exp_probs_b, w.n_expert_used);
            wsel = ggml_inkling2_sigmoid_topk_wts(c, r_logits, sel, w.route_scale, 1e-20f);
        }

        if (do_split) {
            // Dual-GPU R9700 + Strix Halo concurrent expert execution!
            if (!use_fused_router_split) {
                if (use_dyn_split) {
                    ggml_tensor * routes = inkling2_dynamic_split_routes(c, sel, w.n_expert, w.split_pct);
                    hot_sel  = ggml_view_2d(c, routes, w.n_expert_used, n_tokens, routes->nb[1], 0);
                    cold_sel = ggml_view_2d(c, routes, w.n_expert_used, n_tokens, routes->nb[1], routes->nb[2]);
                } else {
                    hot_sel  = ggml_moe_hot_routes(c, sel, w.split_hot_lut, nullptr, w.n_expert_used, true);
                    cold_sel = ggml_moe_hot_routes(c, sel, w.split_hot_lut, nullptr, w.n_expert_used, false);
                }
            }
            ggml_build_forward_expand(gf, hot_sel);
            ggml_build_forward_expand(gf, cold_sel);
            ggml_build_forward_expand(gf, wsel);

            ggml_tensor * x3 = ggml_reshape_3d(c, cur, w.n_embd, 1, n_tokens);
            // Hot lanes on R9700
            ggml_tensor * hot_gu   = ggml_mul_mat_id(c, L.ffn_gate_up_exps, x3, hot_sel);
            ggml_tensor * hot_act  = ggml_inkling2_situ(c, hot_gu, w.situ_beta, w.situ_linear_beta);
            ggml_tensor * hot_down = ggml_mul_mat_id(c, L.ffn_down_exps, hot_act, hot_sel);

            // Cold picks on Strix Halo (concurrent!): reduce cold_down -> 73 KB F32 (verify, bit-exact!)
            // or 36 KB F16 (prefill chunks) on Strix! Eliminates copying 589 KB hot_down R9700->Strix!
            const int cold_sum_f16 = (n_tokens > 16) ? 1 : 0;
            ggml_tensor * cold_in = (n_tokens > 32) ? ggml_cast(c, cur, GGML_TYPE_F16) : cur;
            ggml_tensor * cold_f32 = (cold_in->type == GGML_TYPE_F16) ? ggml_cast(c, cold_in, GGML_TYPE_F32) : cold_in;
            if (cold_f32 != cold_in) ws.expert_nodes.push_back(cold_f32);
            ggml_tensor * cold_x3 = ggml_reshape_3d(c, cold_f32, w.n_embd, 1, n_tokens);
            ggml_tensor * cold_gu   = ggml_mul_mat_id(c, L.exp_gate_up_tail, cold_x3, cold_sel);
            ggml_tensor * cold_act  = ggml_inkling2_situ(c, cold_gu, w.situ_beta, w.situ_linear_beta);
            ggml_tensor * cold_down = ggml_mul_mat_id(c, L.exp_down_tail, cold_act, cold_sel);
            ggml_tensor * cold_sum  = ggml_inkling2_moe_masked_sum(c, cold_down, wsel, cold_sel, nullptr, cold_sum_f16);
            ws.expert_nodes.insert(ws.expert_nodes.end(), {cold_gu, cold_act, cold_down, cold_sum});

            // Expand cold_sum FIRST so Strix starts immediately, while R9700 runs hot_down concurrently!
            ggml_build_forward_expand(gf, cold_sum);
            ggml_build_forward_expand(gf, hot_down);

            // Optional R9700 64 MB Infinity Cache prefetch during wait (off by default unless LUCE_INKLING2_PREFETCH=1)
            const bool do_pf = (std::getenv("LUCE_INKLING2_PREFETCH") != nullptr &&
                                std::atoi(std::getenv("LUCE_INKLING2_PREFETCH")) != 0);
            if (do_pf) prefetch_next_layer_weights(c, gf, hot_down, v, w);

            // Join on R9700 (+ fused FFN post-conv + residual add in ONE kernel for T<=8!)
            if (use_fused_moe_sconv) {
                cur = inkling2_moe_masked_sum_sconv_add(
                    c, hot_down, wsel, hot_sel, cold_sum,
                    L.ffn_post_conv, state_live, state_dst, ffn_inp, 2 * kv_sz + w.n_embd, write_mode);
            } else {
                cur = ggml_inkling2_moe_masked_sum(c, hot_down, wsel, hot_sel, cold_sum, /*out_f16=*/0);
                cur = ggml_inkling2_sconv_add(
                    c, cur, L.ffn_post_conv, state_live, state_dst, nullptr, ffn_inp, 2 * kv_sz + w.n_embd, 0, write_mode);
            }
        } else {
            ggml_tensor * x3   = ggml_reshape_3d(c, cur, w.n_embd, 1, n_tokens);
            ggml_tensor * gu   = ggml_mul_mat_id(c, L.ffn_gate_up_exps, x3, sel);
            ggml_tensor * act  = ggml_inkling2_situ(c, gu, w.situ_beta, w.situ_linear_beta);
            ggml_tensor * down = ggml_mul_mat_id(c, L.ffn_down_exps, act, sel);
            if (use_fused_moe_sconv) {
                cur = inkling2_moe_masked_sum_sconv_add(
                    c, down, wsel, sel, nullptr,
                    L.ffn_post_conv, state_live, state_dst, ffn_inp, 2 * kv_sz + w.n_embd, write_mode);
            } else {
                cur = use_fused_router
                    ? ggml_inkling2_moe_masked_sum(c, down, wsel, sel, nullptr, /*out_f16=*/0)
                    : ggml_ds4_moe_fused_combine_shared(c, down, wsel, nullptr);
                cur = ggml_inkling2_sconv_add(
                    c, cur, L.ffn_post_conv, state_live, state_dst, nullptr, ffn_inp, 2 * kv_sz + w.n_embd, 0, write_mode);
            }
        }
    }

    // ─── Final AttnRes Mix + Output Norm + Capture Tap 3 ────────────────
    ggml_tensor * out_norm = ggml_inkling2_attn_res_mix(
        c, cur, w.output_res_query, w.output_norm,
        blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);

    if (capture_taps) {
        captured_taps[3] = out_norm;
        // Pack the 4 captured taps into [n_embd, 4, T]
        ggml_tensor * t0 = ggml_reshape_3d(c, captured_taps[0], w.n_embd, 1, n_tokens);
        ggml_tensor * t1 = ggml_reshape_3d(c, captured_taps[1], w.n_embd, 1, n_tokens);
        ggml_tensor * t2 = ggml_reshape_3d(c, captured_taps[2], w.n_embd, 1, n_tokens);
        ggml_tensor * t3 = ggml_reshape_3d(c, captured_taps[3], w.n_embd, 1, n_tokens);
        ws.taps_out = ggml_concat(c, ggml_concat(c, t0, t1, 1), ggml_concat(c, t2, t3, 1), 1);
        ggml_set_output(ws.taps_out);
        ggml_build_forward_expand(gf, ws.taps_out);
    }

    // For prefill (!verify && n_tokens > 1), only project LM head for the final token
    ggml_tensor * lm_in = (!verify && n_tokens > 1)
        ? ggml_view_2d(c, out_norm, w.n_embd, 1, out_norm->nb[1], (size_t) (n_tokens - 1) * out_norm->nb[1])
        : out_norm;
    const int n_out_rows = (int) lm_in->ne[1];

    // LM head (with optional 79/21 R9700 + Strix row split on decode/verify!)
    ggml_tensor * raw_logits = nullptr;
    if (use_split && w.expert_backend && w.output_tail && n_out_rows <= 8) {
        const int64_t head_rows = w.output->ne[1] - w.output_tail->ne[1];
        ggml_tensor * head_w = ggml_view_2d(c, w.output, w.output->ne[0], head_rows, w.output->nb[1], 0);
        ggml_tensor * head = ggml_mul_mat(c, head_w, lm_in);
        ggml_tensor * tail = ggml_mul_mat(c, w.output_tail, lm_in);
        ggml_backend_cuda_mul_mat_rdna4_rows(head);
        ggml_backend_cuda_mul_mat_rdna4_rows(tail);
        ws.expert_nodes.push_back(tail);
        raw_logits = ggml_concat(c, head, tail, 0);
    } else {
        raw_logits = ggml_mul_mat(c, w.output, lm_in);
    }

    ws.logits = ggml_inkling2_softcap_slice(c, raw_logits, w.n_vocab_unpadded, w.logit_softcap);
    ws.greedy = ggml_argmax(c, ws.logits);
    ggml_set_output(ws.logits);
    ggml_set_output(ws.greedy);
    ggml_build_forward_expand(gf, ws.logits);
    ggml_build_forward_expand(gf, ws.greedy);

    // Allocate graph (single-device gallocr or true parallel multi-GPU scheduler!)
    const bool need_sched = (use_split || use_split_kv) && !ws.expert_nodes.empty() && w.expert_backend != nullptr && cache.split_cpu != nullptr;
    if (need_sched) {
        ggml_backend_t backends[3] = { target_backend, w.expert_backend, cache.split_cpu };
        // parallel=true, op_offload=false: per-copy GPU events keep R9700 and Strix running concurrently!
        ws.sched = ggml_backend_sched_new(backends, nullptr, 3, max_nodes, true, false);
        if (!ws.sched) {
            free_inkling2_workspace(ws);
            return false;
        }
        ggml_backend_sched_set_late_cross_input_split(ws.sched, true);
        // Batched peer copies run as the tail of the producer's captured HIP graph (eliminates serial copy-wait gaps!)
        ggml_backend_cuda_set_copy_tail(target_backend, true);
        ggml_backend_cuda_set_copy_tail(w.expert_backend, true);
        ggml_backend_sched_reset(ws.sched);
        // Pin EVERY non-expert node & graph input strictly to target_backend, and ONLY ws.expert_nodes to
        // expert_backend! Left unpinned, ggml_backend_sched pulls adjacent target nodes (loop_ctrl_pre,
        // dynamic_split_lut, hot_act, etc.) onto expert_backend and ping-pongs tensors across PCIe!
        {
            const std::unordered_set<const ggml_tensor *> on_expert(ws.expert_nodes.begin(), ws.expert_nodes.end());
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                ggml_tensor * node = ggml_graph_node(gf, i);
                for (ggml_tensor * src : node->src) {
                    for (ggml_tensor * t = src; t; t = t->view_src) {
                        if ((t->flags & GGML_TENSOR_FLAG_INPUT) && !t->buffer) {
                            ggml_backend_sched_set_tensor_backend(ws.sched, t, target_backend);
                        }
                    }
                }
                const bool is_view = node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW ||
                                     node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE ||
                                     node->op == GGML_OP_TRANSPOSE;
                if (on_expert.count(node)) {
                    ggml_backend_sched_set_tensor_backend(ws.sched, node, w.expert_backend);
                } else if (!is_view && ggml_backend_supports_op(target_backend, node)) {
                    ggml_backend_sched_set_tensor_backend(ws.sched, node, target_backend);
                }
            }
        }
        if (!ggml_backend_sched_alloc_graph(ws.sched, gf)) {
            std::fprintf(stderr, "[inkling2] ggml_backend_sched_alloc_graph failed (T=%d)\n", n_tokens);
            free_inkling2_workspace(ws);
            return false;
        }
    } else {
        ws.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
        if (!ws.alloc || !ggml_gallocr_alloc_graph(ws.alloc, gf)) {
            std::fprintf(stderr, "[inkling2] ggml_gallocr_alloc_graph failed (T=%d)\n", n_tokens);
            free_inkling2_workspace(ws);
            return false;
        }
    }

    ws.built = true;
    return true;
}

} // namespace

Inkling2ForwardOut inkling2_forward(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        const int32_t * tokens,
        int n_tokens,
        int pos0,
        bool verify,
        bool need_logits,
        bool capture_taps_for_dspark,
        bool allow_split,
        float fast_sample_temp) {
    Inkling2ForwardOut out;
    if (n_tokens <= 0 || pos0 < 0 || pos0 + n_tokens > cache.max_ctx) return out;

    const bool can_capture = capture_taps_for_dspark && w.dspark.loaded;
    const bool use_split_kv = (cache.remote_backend != nullptr && cache.remote_split_pos > 0);

    // Choose single-GPU vs Dual-GPU split:
    //   - T=1 decode: single-GPU on R9700 (zero PCIe hops) unless LUCE_INKLING2_SPLIT_T1=1
    //   - T>=2 verify: Dual-GPU parallel experts + LM head when expert_backend is loaded & healthy!
    bool use_split = allow_split && (w.expert_backend != nullptr && w.split_exp > 0);
    if (n_tokens == 1 && !verify) {
        const char * s1 = std::getenv("LUCE_INKLING2_SPLIT_T1");
        use_split = (s1 && std::atoi(s1) != 0 && w.expert_backend != nullptr && w.split_exp > 0);
    }

    Inkling2GraphWorkspace temp_ws;
    Inkling2GraphWorkspace * ws = nullptr;
    if (!verify && n_tokens == 1) {
        ws = &cache.decode_ws;
    } else if (verify && n_tokens >= 2 && n_tokens <= kInkling2MaxVerify) {
        ws = &cache.verify_ws[n_tokens];
    } else {
        ws = &temp_ws;
    }

    if (!ws->built || ws->n_tokens != n_tokens || ws->verify != verify ||
        ws->split != use_split || ws->split_kv != use_split_kv ||
        ws->capture_taps != can_capture) {
        if (!build_inkling2_target_graph(
                target_backend, w, cache, *ws, n_tokens, verify, use_split, use_split_kv, can_capture)) {
            return out;
        }
    }

    // Upload tiny runtime inputs (tokens, positions, kv_meta)
    std::vector<int32_t> pos(n_tokens);
    for (int i = 0; i < n_tokens; ++i) pos[i] = pos0 + i;

    const int kv_len = pos0 + n_tokens;
    int32_t meta[4] = { pos0, kv_len, use_split_kv ? cache.remote_split_pos : 0, kv_len };
    ggml_backend_tensor_set(ws->inp_tokens, tokens, 0, (size_t) n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(ws->inp_pos, pos.data(), 0, (size_t) n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(ws->inp_meta, meta, 0, sizeof(meta));
    if (use_split_kv && ws->inp_meta_r) {
        int32_t meta_r[4] = { pos0, kv_len, 0, std::min(kv_len, cache.remote_split_pos) };
        ggml_backend_tensor_set(ws->inp_meta_r, meta_r, 0, sizeof(meta_r));
    }

    ggml_status st = GGML_STATUS_SUCCESS;
    if (ws->sched) {
        st = ggml_backend_sched_graph_compute_async(ws->sched, ws->gf);
        if (st == GGML_STATUS_SUCCESS && w.expert_backend && n_tokens <= kInkling2MaxVerify) {
            // Keep Strix Halo awake between verify steps (~4 ms draft window) so layers 0..2 don't wake cold!
            ggml_backend_cuda_hold(w.expert_backend, 8000);
        }
        if (st == GGML_STATUS_SUCCESS) {
            ggml_backend_sched_synchronize(ws->sched);
        }
    } else {
        st = ggml_backend_graph_compute(target_backend, ws->gf);
    }
    if (st != GGML_STATUS_SUCCESS) {
        if (ws == &temp_ws) free_inkling2_workspace(temp_ws);
        return out;
    }

    const int n_out = (int) ws->greedy->ne[0];
    out.greedy.resize(n_out);
    ggml_backend_tensor_get(ws->greedy, out.greedy.data(), 0, (size_t) n_out * sizeof(int32_t));
    out.logits_dev = (ws != &temp_ws) ? ws->logits : nullptr;

    if (need_logits && fast_sample_temp > 0.0f && ws != &temp_ws) {
        // Cached 1-node exact GPU top-candidate extractor (~20 us on GPU, 2 KB/row PCIe vs 804 KB/row!)
        if (!ws->sample_gf) {
            struct ggml_init_params sgp = { 16 * ggml_tensor_overhead() + ggml_graph_overhead() + 4096, nullptr, true };
            ws->sample_ctx = ggml_init(sgp);
            ws->sample_gf  = ggml_new_graph_custom(ws->sample_ctx, 16, false);
            // Leaf alias pointing directly to ws->logits device memory (prevents walking the 48-layer graph!)
            ws->sample_logits = ggml_new_tensor_2d(ws->sample_ctx, GGML_TYPE_F32, w.n_vocab_unpadded, n_out);
            ws->sample_logits->data   = ws->logits->data;
            ws->sample_logits->buffer = ws->logits->buffer;
            ws->sample_cands = ggml_new_tensor_2d(ws->sample_ctx, GGML_TYPE_I32, 516, n_out);
            ws->sample_cands->op = GGML_OP_INKLING2_OP;
            ws->sample_cands->src[0] = ws->sample_logits;
            set_op_i32(ws->sample_cands, 0, 32);
            set_op_f32(ws->sample_cands, 1, fast_sample_temp);
            ggml_set_output(ws->sample_cands);
            ggml_build_forward_expand(ws->sample_gf, ws->sample_cands);
            ws->sample_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
            if (!ws->sample_alloc || !ggml_gallocr_alloc_graph(ws->sample_alloc, ws->sample_gf)) {
                if (ws->sample_alloc) { ggml_gallocr_free(ws->sample_alloc); ws->sample_alloc = nullptr; }
                if (ws->sample_ctx)   { ggml_free(ws->sample_ctx); ws->sample_ctx = nullptr; }
                ws->sample_gf = nullptr;
            }
        }
        if (ws->sample_gf) {
            ws->sample_logits->data   = ws->logits->data;
            ws->sample_logits->buffer = ws->logits->buffer;
            set_op_f32(ws->sample_cands, 1, fast_sample_temp);
            if (ggml_backend_graph_compute(target_backend, ws->sample_gf) == GGML_STATUS_SUCCESS) {
                out.top_cands.resize((size_t) 516 * n_out);
                ggml_backend_tensor_get(ws->sample_cands, out.top_cands.data(), 0, out.top_cands.size() * sizeof(int32_t));
            }
        }
        if (out.top_cands.empty()) {
            out.logits.resize((size_t) w.n_vocab_unpadded * n_out);
            ggml_backend_tensor_get(ws->logits, out.logits.data(), 0, out.logits.size() * sizeof(float));
        }
    } else if (need_logits) {
        out.logits.resize((size_t) w.n_vocab_unpadded * n_out);
        ggml_backend_tensor_get(ws->logits, out.logits.data(), 0, out.logits.size() * sizeof(float));
    }

    // For non-verify forwards (prefill / T=1 decode), immediately commit captured taps to DSpark KV cache
    if (!verify && can_capture && ws->taps_out) {
        inkling2_dspark_commit_taps(target_backend, w, cache, ws->taps_out, pos0, n_tokens);
    }

    if (!verify) {
        cache.cur_pos = pos0 + n_tokens;
    }

    if (ws == &temp_ws) free_inkling2_workspace(temp_ws);
    out.ok = true;
    return out;
}

// ─── 3-Pass Wavefront Loop-Fused Prefill (3x Less Core Weight Traffic, Bit-Exact!) ──
// Pipelines chunks C_{s-2}@Pass2, C_{s-1}@Pass1, and C_s@Pass0 through each physical
// core layer (blk.6..17) in ONE batched GEMM / mul_mat_id pass -> 3.0x less core weight
// reads AND 3x larger MoE expert batching straight into RDNA4 WMMA!
Inkling2ForwardOut inkling2_prefill_wavefront(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        const int32_t * tokens,
        int n_tokens,
        int pos0,
        int chunk_size,
        bool need_logits,
        bool capture_taps_for_dspark) {
    Inkling2ForwardOut out;
    if (!w.is_loop || w.n_passes != 3 || chunk_size < 32 || n_tokens < chunk_size * 2) {
        return out;
    }

    const int Hq = w.n_head, Hk = w.n_head_kv, D = w.head_dim;
    const int q_sz = Hq * D, kv_sz = Hk * D, r_sz = w.d_rel * Hq;
    const bool can_capture = capture_taps_for_dspark && w.dspark.loaded;

    // Split [pos0, pos0 + n_tokens) into M causal chunks
    struct ChunkInfo { int off; int len; int pos; };
    std::vector<ChunkInfo> chunks;
    for (int off = 0; off < n_tokens; off += chunk_size) {
        const int len = std::min(chunk_size, n_tokens - off);
        chunks.push_back({off, len, pos0 + off});
    }
    const int M = (int) chunks.size();

    // Persistent GPU state per in-flight chunk (ring of 3 slots -> only ~95 MiB total VRAM!)
    const int max_blocks_core = 4; // prelude(1) + core(2) + margin
    struct ChunkState {
        ggml_context *        ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        ggml_tensor * prelude_cur = nullptr;
        ggml_tensor * cur         = nullptr;
        ggml_tensor * blocks[4]   = {};
        ggml_tensor * bank[2]     = {};
        ggml_tensor * taps[3]     = {};
        int n_prelude_blocks = 0;
        int n_blocks = 0;
        int n_bank   = 0;
        int len      = 0;
    };
    ChunkState ring[3];
    auto free_ring = [&]() {
        for (int s = 0; s < 3; ++s) {
            if (ring[s].buf) { ggml_backend_buffer_free(ring[s].buf); ring[s].buf = nullptr; }
            if (ring[s].ctx) { ggml_free(ring[s].ctx); ring[s].ctx = nullptr; }
        }
    };
    auto alloc_slot = [&](int slot_i, int len) -> bool {
        auto & cs = ring[slot_i];
        if (cs.buf && cs.len == len) return true;
        if (cs.buf) { ggml_backend_buffer_free(cs.buf); cs.buf = nullptr; }
        if (cs.ctx) { ggml_free(cs.ctx); cs.ctx = nullptr; }
        struct ggml_init_params gp = { 32 * ggml_tensor_overhead() + 4096, nullptr, true };
        cs.ctx = ggml_init(gp);
        cs.prelude_cur = ggml_new_tensor_2d(cs.ctx, GGML_TYPE_F32, w.n_embd, len);
        cs.cur         = ggml_new_tensor_2d(cs.ctx, GGML_TYPE_F32, w.n_embd, len);
        for (int b = 0; b < max_blocks_core; ++b) cs.blocks[b] = ggml_new_tensor_2d(cs.ctx, GGML_TYPE_F32, w.n_embd, len);
        for (int b = 0; b < 2; ++b)               cs.bank[b]   = ggml_new_tensor_2d(cs.ctx, GGML_TYPE_F32, w.n_embd, len);
        for (int t = 0; t < 3; ++t)               cs.taps[t]   = ggml_new_tensor_2d(cs.ctx, GGML_TYPE_F32, w.n_embd, len);
        cs.buf = ggml_backend_alloc_ctx_tensors(cs.ctx, target_backend);
        cs.len = len;
        return cs.buf != nullptr;
    };

    // Helper: run one non-core physical layer (Prelude or Coda) on a single chunk
    auto run_single_layer = [&](ggml_context * c, ggml_tensor * cur, std::vector<ggml_tensor *> & blocks,
                                ggml_tensor * inp_pos, ggml_tensor * inp_meta, int il, int ip,
                                ggml_tensor * taps_cap[3], int v_idx) -> ggml_tensor * {
        const auto & L = w.layers[ip];
        if (can_capture) {
            for (int ti = 0; ti < 3; ++ti) if (v_idx == w.dspark.target_layer_ids[ti]) taps_cap[ti] = cur;
        }
        ggml_tensor * inpSA = cur;
        ggml_tensor * st = cache.conv_state[il];
        ggml_tensor * inputs[3] = {cur, cur, cur};
        const int T_c = (int) cur->ne[1];
        if (L.attn_res_query == nullptr) {
            inputs[0] = rms_norm_w(c, cur, L.attn_norm, w.rms_eps);
            inputs[1] = inputs[2] = inputs[0];
        } else if (L.is_swa) {
            inputs[0] = ggml_inkling2_attn_res_mix(c, cur, L.attn_res_query, L.attn_norm,
                blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
            inputs[1] = inputs[2] = inputs[0];
        } else {
            ggml_tensor * mixed3 = ggml_inkling2_attn_res_mix(c, cur, L.attn_res_query, L.attn_qkv_norm,
                blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
            for (int j = 0; j < 3; ++j)
                inputs[j] = ggml_view_2d(c, mixed3, w.n_embd, T_c, mixed3->nb[2], (size_t) j * mixed3->nb[1]);
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
            ggml_tensor * proj = ggml_mul_mat(c, L.wqgrs, inputs[0]);
            Q_raw = row_view_2d(c, proj, 0, q_sz);
            Gate  = row_view_2d(c, proj, q_sz, q_sz);
            R_raw = row_view_2d(c, proj, 2 * q_sz, r_sz);
            Sink  = row_view_2d(c, proj, 2 * q_sz + r_sz, Hq);
            if (!L.is_swa && L.wk_dyn && L.wv_dyn) {
                ggml_tensor * kd = ggml_mul_mat(c, L.wk_dyn, inputs[1]);
                K_raw = row_view_2d(c, kd, 0, kv_sz); k_dyn = row_view_2d(c, kd, kv_sz, w.sconv_kernel * Hk);
                ggml_tensor * vd = ggml_mul_mat(c, L.wv_dyn, inputs[2]);
                V_raw = row_view_2d(c, vd, 0, kv_sz); v_dyn = row_view_2d(c, vd, kv_sz, w.sconv_kernel * Hk);
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
        inkling2_clip_qr_rel_table(
            c, Q_raw, L.attn_q_clip, R_raw, L.attn_rel_clip,
            L.is_swa ? nullptr : inp_pos, L.attn_rel_bias, Sink,
            D, w.d_rel, Hq, w.qk_clip, w.rel_clip, w.rms_eps,
            L.is_swa ? 0 : w.log_scale_floor, L.is_swa ? 0.0f : w.log_scale_alpha,
            &Q, &rt);
        ggml_tensor * kd_dep = ggml_inkling2_sconv_kv_write(c, K_raw, L.attn_k_conv, st, st, k_dyn, cache.k_cache[il], inp_pos,
            0, D, Hk, 0, swa, w.qk_clip, w.rms_eps, 1);
        ggml_tensor * vd_dep = ggml_inkling2_sconv_kv_write(c, V_raw, L.attn_v_conv, st, st, v_dyn, cache.v_cache[il], inp_pos,
            kv_sz, D, Hk, 1, swa, w.qk_clip, w.rms_eps, 1);
        ggml_tensor * ao = ggml_inkling2_attn_rel(c, Q, cache.k_cache[il], cache.v_cache[il], rt, inp_meta,
            kd_dep, vd_dep, Gate, swa, w.rel_extent, 0, 1);
        cur = ggml_mul_mat(c, L.wo, ao);
        if (ip % w.attn_res_block == 0) {
            cur = ggml_inkling2_sconv(c, cur, L.attn_post_conv, st, st, nullptr, 2 * kv_sz, 0, 1);
            blocks.push_back(inpSA);
        } else {
            cur = ggml_inkling2_sconv_add(c, cur, L.attn_post_conv, st, st, nullptr, inpSA, 2 * kv_sz, 0, 1);
        }
        ggml_tensor * ffn_inp = cur;
        cur = ggml_inkling2_attn_res_mix(c, cur, L.ffn_res_query, L.ffn_norm,
            blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
        ggml_tensor * rl = ggml_mul_mat(c, L.ffn_gate_inp_q8 ? L.ffn_gate_inp_q8 : L.ffn_gate_inp, cur);
        ggml_tensor * sel = ggml_inkling2_sigmoid_topk_ids(c, rl, L.ffn_exp_probs_b, w.n_expert_used);
        ggml_tensor * wsel = ggml_inkling2_sigmoid_topk_wts(c, rl, sel, w.route_scale, 1e-20f);
        ggml_tensor * x3 = ggml_reshape_3d(c, cur, w.n_embd, 1, T_c);
        ggml_tensor * gu = ggml_mul_mat_id(c, L.ffn_gate_up_exps, x3, sel);
        ggml_tensor * act = ggml_inkling2_situ(c, gu, w.situ_beta, w.situ_linear_beta);
        ggml_tensor * dn = ggml_mul_mat_id(c, L.ffn_down_exps, act, sel);
        cur = ggml_ds4_moe_fused_combine_shared(c, dn, wsel, nullptr);
        cur = ggml_inkling2_sconv_add(c, cur, L.ffn_post_conv, st, st, nullptr, ffn_inp, 2 * kv_sz + w.n_embd, 0, 1);
        return cur;
    };

    // Wavefront schedule: step = 0 .. M + 1
    for (int step = 0; step <= M + 1; ++step) {
        // ── Stage 1: Prelude (L0..5) for chunk `step` ───────────────────
        if (step < M) {
            const auto & ci = chunks[step];
            const int s_slot = step % 3;
            if (!alloc_slot(s_slot, ci.len)) { free_ring(); return out; }
            auto & cs = ring[s_slot];

            const size_t max_nodes = 4096;
            struct ggml_init_params gp = { max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false) + (4 << 20), nullptr, true };
            ggml_context * c = ggml_init(gp);
            ggml_cgraph * gf = ggml_new_graph_custom(c, max_nodes, false);
            ggml_tensor * inp_tok  = ggml_new_tensor_1d(c, GGML_TYPE_I32, ci.len);
            ggml_tensor * inp_pos  = ggml_new_tensor_1d(c, GGML_TYPE_I32, ci.len);
            ggml_tensor * inp_meta = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
            ggml_set_input(inp_tok); ggml_set_input(inp_pos); ggml_set_input(inp_meta);

            ggml_tensor * cur = rms_norm_w(c, ggml_get_rows(c, w.tok_embd, inp_tok), w.tok_norm, w.rms_eps);
            std::vector<ggml_tensor *> blocks;
            ggml_tensor * taps_cap[3] = {};
            for (int il = 0; il < w.core_start; ++il) {
                cur = run_single_layer(c, cur, blocks, inp_pos, inp_meta, il, il, taps_cap, il);
            }
            ggml_build_forward_expand(gf, ggml_cpy(c, cur, cs.prelude_cur));
            ggml_build_forward_expand(gf, ggml_cpy(c, cur, cs.cur));
            cs.n_prelude_blocks = (int) blocks.size();
            cs.n_blocks = cs.n_prelude_blocks;
            for (int b = 0; b < cs.n_prelude_blocks; ++b) {
                ggml_build_forward_expand(gf, ggml_cpy(c, blocks[b], cs.blocks[b]));
            }
            if (can_capture && taps_cap[0]) {
                ggml_build_forward_expand(gf, ggml_cpy(c, taps_cap[0], cs.taps[0]));
            }

            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
            if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) { if (ga) ggml_gallocr_free(ga); ggml_free(c); free_ring(); return out; }
            std::vector<int32_t> pos_v(ci.len);
            for (int i = 0; i < ci.len; ++i) pos_v[i] = ci.pos + i;
            int32_t meta[4] = { ci.pos, ci.pos + ci.len, 0, ci.pos + ci.len };
            ggml_backend_tensor_set(inp_tok,  tokens + ci.off, 0, (size_t) ci.len * sizeof(int32_t));
            ggml_backend_tensor_set(inp_pos,  pos_v.data(),    0, (size_t) ci.len * sizeof(int32_t));
            ggml_backend_tensor_set(inp_meta, meta,            0, sizeof(meta));
            const bool ok = (ggml_backend_graph_compute(target_backend, gf) == GGML_STATUS_SUCCESS);
            ggml_gallocr_free(ga);
            ggml_free(c);
            if (!ok) { free_ring(); return out; }
        }

        // ── Stage 2: Fused 3-Pass Loop Core (blk.6..17 ONCE for up to 3 chunks!) ──
        struct StreamDesc { int chunk_idx; int ring_slot; int pass; int len; int pos; int tok_off; };
        std::vector<StreamDesc> streams;
        int T_tot = 0;
        for (int pass = 2; pass >= 0; --pass) {
            const int cidx = step - pass;
            if (cidx >= 0 && cidx < M) {
                streams.push_back({cidx, cidx % 3, pass, chunks[cidx].len, chunks[cidx].pos, T_tot});
                T_tot += chunks[cidx].len;
            }
        }
        if (!streams.empty()) {
            const int n_s = (int) streams.size();
            const size_t max_nodes = 16384;
            struct ggml_init_params gp = { max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false) + (16 << 20), nullptr, true };
            ggml_context * c = ggml_init(gp);
            ggml_cgraph * gf = ggml_new_graph_custom(c, max_nodes, false);

            ggml_tensor * s_pos[3]  = {};
            ggml_tensor * s_meta[3] = {};
            ggml_tensor * s_cur[3]  = {};
            ggml_tensor * s_bank_out[3] = {};
            std::vector<ggml_tensor *> s_blocks[3];
            std::vector<ggml_tensor *> s_bank[3];

            for (int si = 0; si < n_s; ++si) {
                const auto & sd = streams[si];
                const auto & cs = ring[sd.ring_slot];
                s_pos[si]  = ggml_new_tensor_1d(c, GGML_TYPE_I32, sd.len);
                s_meta[si] = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
                ggml_set_input(s_pos[si]);
                ggml_set_input(s_meta[si]);

                if (sd.pass == 0) {
                    s_cur[si] = cs.prelude_cur;
                    for (int b = 0; b < cs.n_prelude_blocks; ++b) s_blocks[si].push_back(cs.blocks[b]);
                } else {
                    s_bank_out[si] = cs.cur;
                    for (int b = 0; b < cs.n_bank; ++b) s_bank[si].push_back(cs.bank[b]);
                    s_cur[si] = cs.prelude_cur;
                    for (int b = 0; b < cs.n_prelude_blocks; ++b) s_blocks[si].push_back(cs.blocks[b]);
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

            // R9700 Wavefront Core (chunk=1024) hits 3,600+ tok/s single-GPU without heating Strix Halo;
            // Dual-GPU core split during prefill is opt-in via LUCE_INKLING2_DUAL_PREFILL=1.
            const bool use_dual_core = (w.expert_backend != nullptr && w.split_exp > 0 &&
                                        w.split_exp < w.n_expert && cache.split_cpu != nullptr &&
                                        T_tot >= 256 &&
                                        std::getenv("LUCE_INKLING2_DUAL_PREFILL") != nullptr &&
                                        std::atoi(std::getenv("LUCE_INKLING2_DUAL_PREFILL")) != 0);
            std::vector<ggml_tensor *> core_expert_nodes;

            ggml_tensor * s_taps_cap[3][3] = {};
            for (int ip = w.core_start; ip < w.core_end; ++ip) {
                const auto & L = w.layers[ip];
                const auto & loop = w.loop_layers[ip - w.core_start];

                ggml_tensor * inpSA[3] = {};
                ggml_tensor * in_q[3] = {}, * in_k[3] = {}, * in_v[3] = {};
                ggml_tensor * msgs[3] = {};

                for (int si = 0; si < n_s; ++si) {
                    const auto & sd = streams[si];
                    const int v_idx = w.core_start + sd.pass * (w.core_end - w.core_start) + (ip - w.core_start);
                    if (can_capture) {
                        for (int ti = 0; ti < 3; ++ti)
                            if (v_idx == w.dspark.target_layer_ids[ti]) s_taps_cap[si][ti] = s_cur[si];
                    }
                    inpSA[si] = s_cur[si];
                    if (sd.pass > 0) {
                        msgs[si] = ggml_inkling2_attn_res_mix(c, s_bank_out[si], loop.msg_query, nullptr,
                            s_bank[si].empty() ? nullptr : s_bank[si].data(), (int) s_bank[si].size(), w.rms_eps);
                    }
                    if (sd.pass == 0) {
                        if (L.is_swa) {
                            in_q[si] = ggml_inkling2_attn_res_mix(c, s_cur[si], L.attn_res_query, L.attn_norm,
                                s_blocks[si].empty() ? nullptr : s_blocks[si].data(), (int) s_blocks[si].size(), w.rms_eps);
                            in_k[si] = in_v[si] = in_q[si];
                        } else {
                            ggml_tensor * m3 = ggml_inkling2_attn_res_mix(c, s_cur[si], L.attn_res_query, L.attn_qkv_norm,
                                s_blocks[si].empty() ? nullptr : s_blocks[si].data(), (int) s_blocks[si].size(), w.rms_eps);
                            in_q[si] = ggml_view_2d(c, m3, w.n_embd, sd.len, m3->nb[2], 0);
                            in_k[si] = ggml_view_2d(c, m3, w.n_embd, sd.len, m3->nb[2], m3->nb[1]);
                            in_v[si] = ggml_view_2d(c, m3, w.n_embd, sd.len, m3->nb[2], 2 * m3->nb[1]);
                        }
                    } else {
                        // 3D-batched Q/K/V loop controller (4 launches instead of 12 per pass!)
                        ggml_tensor * mx = ggml_inkling2_attn_res_mix(c, s_cur[si], L.attn_res_query, nullptr,
                            s_blocks[si].empty() ? nullptr : s_blocks[si].data(), (int) s_blocks[si].size(), w.rms_eps);
                        ggml_tensor * pre_qkv = L.is_swa ? mx : ggml_permute(c, mx, 0, 2, 1, 3);
                        ggml_tensor * msg_qkv_v = ggml_view_3d(c, msgs[si], w.n_embd, 3, sd.len, msgs[si]->nb[1], msgs[si]->nb[2], 0);
                        ggml_tensor * msg_qkv   = ggml_permute(c, msg_qkv_v, 0, 2, 1, 3); // [H, T, 3]
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
                ggml_tensor * qgrs_all = ggml_mul_mat(c, L.wqgrs, concat_s(in_q));
                ggml_tensor * k_all    = ggml_mul_mat(c, (!L.is_swa && L.wk_dyn) ? L.wk_dyn : L.wk, concat_s(in_k));
                ggml_tensor * v_all    = ggml_mul_mat(c, (!L.is_swa && L.wv_dyn) ? L.wv_dyn : L.wv, concat_s(in_v));

                ggml_tensor * ao_s[3] = {};
                const int swa = L.is_swa ? w.swa_window : 0;
                for (int si = 0; si < n_s; ++si) {
                    const auto & sd = streams[si];
                    const int il = (sd.pass == 0) ? ip : (w.n_layer_phys + (sd.pass - 1) * (w.core_end - w.core_start) + (ip - w.core_start));
                    ggml_tensor * st = cache.conv_state[il];
                    ggml_tensor * pq = slice_s(qgrs_all, si);
                    ggml_tensor * Q_raw = row_view_2d(c, pq, 0, q_sz);
                    ggml_tensor * Gate  = row_view_2d(c, pq, q_sz, q_sz);
                    ggml_tensor * R_raw = row_view_2d(c, pq, 2 * q_sz, r_sz);
                    ggml_tensor * Sink  = row_view_2d(c, pq, 2 * q_sz + r_sz, Hq);
                    ggml_tensor * pk = slice_s(k_all, si);
                    ggml_tensor * pv = slice_s(v_all, si);
                    ggml_tensor * K_raw = (!L.is_swa && L.wk_dyn) ? row_view_2d(c, pk, 0, kv_sz) : pk;
                    ggml_tensor * k_dyn = (!L.is_swa && L.wk_dyn) ? row_view_2d(c, pk, kv_sz, w.sconv_kernel * Hk) : nullptr;
                    ggml_tensor * V_raw = (!L.is_swa && L.wv_dyn) ? row_view_2d(c, pv, 0, kv_sz) : pv;
                    ggml_tensor * v_dyn = (!L.is_swa && L.wv_dyn) ? row_view_2d(c, pv, kv_sz, w.sconv_kernel * Hk) : nullptr;

                    ggml_tensor * Q = nullptr, * rt = nullptr;
                    inkling2_clip_qr_rel_table(
                        c, Q_raw, L.attn_q_clip, R_raw, L.attn_rel_clip,
                        L.is_swa ? nullptr : s_pos[si], L.attn_rel_bias, Sink,
                        D, w.d_rel, Hq, w.qk_clip, w.rel_clip, w.rms_eps,
                        L.is_swa ? 0 : w.log_scale_floor, L.is_swa ? 0.0f : w.log_scale_alpha,
                        &Q, &rt);
                    ggml_tensor * kd_dep = ggml_inkling2_sconv_kv_write(c, K_raw, L.attn_k_conv, st, st, k_dyn, cache.k_cache[il], s_pos[si],
                        0, D, Hk, 0, swa, w.qk_clip, w.rms_eps, 1);
                    ggml_tensor * vd_dep = ggml_inkling2_sconv_kv_write(c, V_raw, L.attn_v_conv, st, st, v_dyn, cache.v_cache[il], s_pos[si],
                        kv_sz, D, Hk, 1, swa, w.qk_clip, w.rms_eps, 1);
                    ao_s[si] = ggml_inkling2_attn_rel(c, Q, cache.k_cache[il], cache.v_cache[il], rt, s_meta[si],
                        kd_dep, vd_dep, Gate, swa, w.rel_extent, 0, 1);
                }

                // ONE FUSED O-PROJ GEMM ACROSS ALL 3 LOOP PASSES!
                ggml_tensor * wo_all = ggml_mul_mat(c, L.wo, concat_s(ao_s));

                ggml_tensor * ffn_inp[3] = {};
                ggml_tensor * ffn_in_s[3] = {};
                for (int si = 0; si < n_s; ++si) {
                    const auto & sd = streams[si];
                    const int il = (sd.pass == 0) ? ip : (w.n_layer_phys + (sd.pass - 1) * (w.core_end - w.core_start) + (ip - w.core_start));
                    ggml_tensor * st = cache.conv_state[il];
                    ggml_tensor * x = nullptr;
                    if (ip % w.attn_res_block == 0) {
                        x = ggml_inkling2_sconv(c, slice_s(wo_all, si), L.attn_post_conv, st, st, nullptr, 2 * kv_sz, 0, 1);
                        s_blocks[si].push_back(inpSA[si]);
                    } else {
                        x = ggml_inkling2_sconv_add(c, slice_s(wo_all, si), L.attn_post_conv, st, st, nullptr, inpSA[si], 2 * kv_sz, 0, 1);
                    }
                    ffn_inp[si] = x;

                    if (sd.pass == 0) {
                        ffn_in_s[si] = ggml_inkling2_attn_res_mix(c, x, L.ffn_res_query, L.ffn_norm,
                            s_blocks[si].empty() ? nullptr : s_blocks[si].data(), (int) s_blocks[si].size(), w.rms_eps);
                    } else {
                        ggml_tensor * mx = ggml_inkling2_attn_res_mix(c, x, L.ffn_res_query, nullptr,
                            s_blocks[si].empty() ? nullptr : s_blocks[si].data(), (int) s_blocks[si].size(), w.rms_eps);
                        ggml_tensor * mf = ggml_view_2d(c, msgs[si], w.n_embd, sd.len, msgs[si]->nb[2], (size_t) Inkling2LoopLayer::CTRL_FFN * msgs[si]->nb[1]);
                        ggml_tensor * cpre = ggml_inkling2_loop_ctrl_pre(c, mx, mf, w.rms_eps);
                        ggml_tensor * gt   = ggml_mul_mat(c, loop.ctrl_down_f16[Inkling2LoopLayer::CTRL_FFN]
                            ? loop.ctrl_down_f16[Inkling2LoopLayer::CTRL_FFN] : loop.ctrl_down[Inkling2LoopLayer::CTRL_FFN], cpre);
                        ggml_tensor * gfo  = ggml_mul_mat(c, loop.ctrl_gf[Inkling2LoopLayer::CTRL_FFN], gt);
                        ffn_in_s[si] = ggml_inkling2_loop_ctrl_post(c, mx, mf, gfo, L.ffn_norm, w.rms_eps);
                    }
                }

                // ONE FUSED ROUTER + 128-EXPERT MoE PASS ACROSS ALL 3 LOOP PASSES (3.0x LESS EXPERT BANDWIDTH!)
                ggml_tensor * ffn_cat = concat_s(ffn_in_s);
                ggml_tensor * rl   = ggml_mul_mat(c, L.ffn_gate_inp_q8 ? L.ffn_gate_inp_q8 : L.ffn_gate_inp, ffn_cat);
                ggml_tensor * sel  = ggml_inkling2_sigmoid_topk_ids(c, rl, L.ffn_exp_probs_b, w.n_expert_used);
                ggml_tensor * wsel = ggml_inkling2_sigmoid_topk_wts(c, rl, sel, w.route_scale, 1e-20f);
                ggml_tensor * moe_all = nullptr;
                if (use_dual_core && L.exp_gate_up_tail) {
                    // Dual-GPU Wavefront Core MoE: R9700 (70%) + Strix Halo (30%) concurrently in F16 transport!
                    ggml_tensor * hot_sel  = ggml_moe_hot_routes(c, sel, w.split_hot_lut, nullptr, w.n_expert_used, true);
                    ggml_tensor * cold_sel = ggml_moe_hot_routes(c, sel, w.split_hot_lut, nullptr, w.n_expert_used, false);
                    ggml_build_forward_expand(gf, hot_sel);
                    ggml_build_forward_expand(gf, cold_sel);
                    ggml_build_forward_expand(gf, wsel);

                    ggml_tensor * x3 = ggml_reshape_3d(c, ffn_cat, w.n_embd, 1, T_tot);
                    ggml_tensor * hot_gu   = ggml_mul_mat_id(c, L.ffn_gate_up_exps, x3, hot_sel);
                    ggml_tensor * hot_act  = ggml_inkling2_situ(c, hot_gu, w.situ_beta, w.situ_linear_beta);
                    ggml_tensor * hot_down = ggml_mul_mat_id(c, L.ffn_down_exps, hot_act, hot_sel);

                    ggml_tensor * cold_in  = ggml_cast(c, ffn_cat, GGML_TYPE_F16);
                    ggml_tensor * cold_f32 = ggml_cast(c, cold_in, GGML_TYPE_F32);
                    ggml_tensor * cold_x3  = ggml_reshape_3d(c, cold_f32, w.n_embd, 1, T_tot);
                    ggml_tensor * cold_gu   = ggml_mul_mat_id(c, L.exp_gate_up_tail, cold_x3, cold_sel);
                    ggml_tensor * cold_act  = ggml_inkling2_situ(c, cold_gu, w.situ_beta, w.situ_linear_beta);
                    ggml_tensor * cold_down = ggml_mul_mat_id(c, L.exp_down_tail, cold_act, cold_sel);
                    ggml_tensor * cold_sum  = ggml_inkling2_moe_masked_sum(c, cold_down, wsel, cold_sel, nullptr, /*out_f16=*/1);
                    core_expert_nodes.insert(core_expert_nodes.end(), {cold_f32, cold_gu, cold_act, cold_down, cold_sum});

                    ggml_build_forward_expand(gf, cold_sum);
                    ggml_build_forward_expand(gf, hot_down);
                    moe_all = ggml_inkling2_moe_masked_sum(c, hot_down, wsel, hot_sel, cold_sum, /*out_f16=*/0);
                } else {
                    ggml_tensor * x3   = ggml_reshape_3d(c, ffn_cat, w.n_embd, 1, T_tot);
                    ggml_tensor * gu   = ggml_mul_mat_id(c, L.ffn_gate_up_exps, x3, sel);
                    ggml_tensor * act  = ggml_inkling2_situ(c, gu, w.situ_beta, w.situ_linear_beta);
                    ggml_tensor * dn   = ggml_mul_mat_id(c, L.ffn_down_exps, act, sel);
                    moe_all = ggml_ds4_moe_fused_combine_shared(c, dn, wsel, nullptr);
                }

                for (int si = 0; si < n_s; ++si) {
                    const auto & sd = streams[si];
                    const int il = (sd.pass == 0) ? ip : (w.n_layer_phys + (sd.pass - 1) * (w.core_end - w.core_start) + (ip - w.core_start));
                    ggml_tensor * st = cache.conv_state[il];
                    s_cur[si] = ggml_inkling2_sconv_add(c, slice_s(moe_all, si), L.ffn_post_conv, st, st, nullptr, ffn_inp[si], 2 * kv_sz + w.n_embd, 0, 1);
                }
            }

            // Persist each stream's end-of-pass state into its ring slot
            for (int si = 0; si < n_s; ++si) {
                const auto & sd = streams[si];
                auto & cs = ring[sd.ring_slot];
                ggml_build_forward_expand(gf, ggml_cpy(c, s_cur[si], cs.cur));
                if (sd.pass < 2) {
                    // Save bank for next pass: blocks after (prelude_blocks + 1)
                    cs.n_bank = std::max(0, (int) s_blocks[si].size() - cs.n_prelude_blocks - 1);
                    for (int b = 0; b < cs.n_bank; ++b) {
                        ggml_build_forward_expand(gf, ggml_cpy(c, s_blocks[si][cs.n_prelude_blocks + 1 + b], cs.bank[b]));
                    }
                } else {
                    // Pass 2 done -> save all blocks for Coda!
                    cs.n_blocks = (int) s_blocks[si].size();
                    for (int b = cs.n_prelude_blocks; b < cs.n_blocks; ++b) {
                        ggml_build_forward_expand(gf, ggml_cpy(c, s_blocks[si][b], cs.blocks[b]));
                    }
                }
                if (can_capture) {
                    for (int ti = 0; ti < 3; ++ti) {
                        if (s_taps_cap[si][ti]) ggml_build_forward_expand(gf, ggml_cpy(c, s_taps_cap[si][ti], cs.taps[ti]));
                    }
                }
            }

            bool ok = false;
            if (use_dual_core && !core_expert_nodes.empty()) {
                ggml_backend_t backends[3] = { target_backend, w.expert_backend, cache.split_cpu };
                ggml_backend_sched_t sc = ggml_backend_sched_new(backends, nullptr, 3, max_nodes, true, false);
                if (!sc) { ggml_free(c); free_ring(); return out; }
                ggml_backend_sched_set_late_cross_input_split(sc, true);
                ggml_backend_sched_reset(sc);
                const std::unordered_set<const ggml_tensor *> on_exp(core_expert_nodes.begin(), core_expert_nodes.end());
                for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                    ggml_tensor * node = ggml_graph_node(gf, i);
                    for (ggml_tensor * src : node->src) {
                        for (ggml_tensor * t = src; t; t = t->view_src) {
                            if ((t->flags & GGML_TENSOR_FLAG_INPUT) && !t->buffer) {
                                ggml_backend_sched_set_tensor_backend(sc, t, target_backend);
                            }
                        }
                    }
                    const bool is_view = node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW ||
                                         node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE ||
                                         node->op == GGML_OP_TRANSPOSE;
                    if (on_exp.count(node)) {
                        ggml_backend_sched_set_tensor_backend(sc, node, w.expert_backend);
                    } else if (!is_view && ggml_backend_supports_op(target_backend, node)) {
                        ggml_backend_sched_set_tensor_backend(sc, node, target_backend);
                    }
                }
                if (ggml_backend_sched_alloc_graph(sc, gf)) {
                    for (int si = 0; si < n_s; ++si) {
                        const auto & sd = streams[si];
                        std::vector<int32_t> pos_v(sd.len);
                        for (int i = 0; i < sd.len; ++i) pos_v[i] = sd.pos + i;
                        int32_t meta[4] = { sd.pos, sd.pos + sd.len, 0, sd.pos + sd.len };
                        ggml_backend_tensor_set(s_pos[si],  pos_v.data(), 0, (size_t) sd.len * sizeof(int32_t));
                        ggml_backend_tensor_set(s_meta[si], meta,         0, sizeof(meta));
                    }
                    ok = (ggml_backend_sched_graph_compute(sc, gf) == GGML_STATUS_SUCCESS);
                }
                ggml_backend_sched_free(sc);
                ggml_free(c);
            } else {
                ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
                if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) { if (ga) ggml_gallocr_free(ga); ggml_free(c); free_ring(); return out; }
                for (int si = 0; si < n_s; ++si) {
                    const auto & sd = streams[si];
                    std::vector<int32_t> pos_v(sd.len);
                    for (int i = 0; i < sd.len; ++i) pos_v[i] = sd.pos + i;
                    int32_t meta[4] = { sd.pos, sd.pos + sd.len, 0, sd.pos + sd.len };
                    ggml_backend_tensor_set(s_pos[si],  pos_v.data(), 0, (size_t) sd.len * sizeof(int32_t));
                    ggml_backend_tensor_set(s_meta[si], meta,         0, sizeof(meta));
                }
                ok = (ggml_backend_graph_compute(target_backend, gf) == GGML_STATUS_SUCCESS);
                ggml_gallocr_free(ga);
                ggml_free(c);
            }
            if (!ok) { free_ring(); return out; }
        }

        // ── Stage 3: Coda (L18..23) + Output Norm + DSpark taps (+ LM head on last chunk) ──
        if (step >= 2) {
            const int cidx = step - 2;
            const auto & ci = chunks[cidx];
            auto & cs = ring[cidx % 3];
            const bool is_last = (cidx == M - 1);

            const size_t max_nodes = 4096;
            struct ggml_init_params gp = { max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false) + (4 << 20), nullptr, true };
            ggml_context * c = ggml_init(gp);
            ggml_cgraph * gf = ggml_new_graph_custom(c, max_nodes, false);
            ggml_tensor * inp_pos  = ggml_new_tensor_1d(c, GGML_TYPE_I32, ci.len);
            ggml_tensor * inp_meta = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
            ggml_set_input(inp_pos); ggml_set_input(inp_meta);

            ggml_tensor * cur = cs.cur;
            std::vector<ggml_tensor *> blocks;
            for (int b = 0; b < cs.n_blocks; ++b) blocks.push_back(cs.blocks[b]);
            ggml_tensor * taps_cap[3] = { cs.taps[0], cs.taps[1], cs.taps[2] };

            const int coda_v0 = w.core_start + w.n_passes * (w.core_end - w.core_start);
            for (int il = w.core_end; il < w.n_layer_phys; ++il) {
                cur = run_single_layer(c, cur, blocks, inp_pos, inp_meta, il, il, taps_cap, coda_v0 + (il - w.core_end));
            }
            ggml_tensor * out_norm = ggml_inkling2_attn_res_mix(c, cur, w.output_res_query, w.output_norm,
                blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);

            ggml_tensor * taps_out = nullptr;
            if (can_capture) {
                ggml_tensor * t0 = ggml_reshape_3d(c, taps_cap[0], w.n_embd, 1, ci.len);
                ggml_tensor * t1 = ggml_reshape_3d(c, taps_cap[1], w.n_embd, 1, ci.len);
                ggml_tensor * t2 = ggml_reshape_3d(c, taps_cap[2], w.n_embd, 1, ci.len);
                ggml_tensor * t3 = ggml_reshape_3d(c, out_norm,    w.n_embd, 1, ci.len);
                taps_out = ggml_concat(c, ggml_concat(c, t0, t1, 1), ggml_concat(c, t2, t3, 1), 1);
                ggml_set_output(taps_out);
                ggml_build_forward_expand(gf, taps_out);
            }

            ggml_tensor * logits_t = nullptr, * greedy_t = nullptr;
            if (is_last) {
                ggml_tensor * lm_in = ggml_view_2d(c, out_norm, w.n_embd, 1, out_norm->nb[1], (size_t) (ci.len - 1) * out_norm->nb[1]);
                ggml_tensor * raw_l = ggml_mul_mat(c, w.output, lm_in);
                logits_t = ggml_inkling2_softcap_slice(c, raw_l, w.n_vocab_unpadded, w.logit_softcap);
                greedy_t = ggml_argmax(c, logits_t);
                ggml_set_output(logits_t);
                ggml_set_output(greedy_t);
                ggml_build_forward_expand(gf, logits_t);
                ggml_build_forward_expand(gf, greedy_t);
            } else if (!can_capture) {
                ggml_build_forward_expand(gf, out_norm);
            }

            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
            if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) { if (ga) ggml_gallocr_free(ga); ggml_free(c); free_ring(); return out; }
            std::vector<int32_t> pos_v(ci.len);
            for (int i = 0; i < ci.len; ++i) pos_v[i] = ci.pos + i;
            int32_t meta[4] = { ci.pos, ci.pos + ci.len, 0, ci.pos + ci.len };
            ggml_backend_tensor_set(inp_pos,  pos_v.data(), 0, (size_t) ci.len * sizeof(int32_t));
            ggml_backend_tensor_set(inp_meta, meta,         0, sizeof(meta));
            const bool ok = (ggml_backend_graph_compute(target_backend, gf) == GGML_STATUS_SUCCESS);
            if (ok && can_capture && taps_out) {
                inkling2_dspark_commit_taps(target_backend, w, cache, taps_out, ci.pos, ci.len);
            }
            if (ok && is_last) {
                out.greedy.resize(1);
                ggml_backend_tensor_get(greedy_t, out.greedy.data(), 0, sizeof(int32_t));
                if (need_logits) {
                    out.logits.resize((size_t) w.n_vocab_unpadded);
                    ggml_backend_tensor_get(logits_t, out.logits.data(), 0, out.logits.size() * sizeof(float));
                }
            }
            ggml_gallocr_free(ga);
            ggml_free(c);
            if (!ok) { free_ring(); return out; }
            cache.cur_pos = ci.pos + ci.len;
        }
    }

    free_ring();
    out.ok = true;
    return out;
}

bool inkling2_verify_rollback(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        int pos0,
        int retained) {
    if (retained < 1 || retained > kInkling2MaxVerify) return false;
    auto & dw = cache.dspark_ws;
    if (!dw.rb_built) {
        const size_t n_nodes = (size_t) w.n_layer_eff * 2 + 16;
        struct ggml_init_params gp = { n_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_nodes, false) + 4096, nullptr, true };
        dw.ctx_rb = ggml_init(gp);
        dw.gf_rb = ggml_new_graph_custom(dw.ctx_rb, n_nodes, false);
        dw.rb_slot = ggml_new_tensor_1d(dw.ctx_rb, GGML_TYPE_I32, 1);
        ggml_set_input(dw.rb_slot);
        for (int il = 0; il < w.n_layer_eff; ++il) {
            ggml_tensor * r = ggml_inkling2_state_restore(dw.ctx_rb, cache.conv_ring[il], cache.conv_state[il], dw.rb_slot);
            ggml_build_forward_expand(dw.gf_rb, r);
        }
        dw.alloc_rb = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
        if (!dw.alloc_rb || !ggml_gallocr_alloc_graph(dw.alloc_rb, dw.gf_rb)) {
            if (dw.alloc_rb) { ggml_gallocr_free(dw.alloc_rb); dw.alloc_rb = nullptr; }
            if (dw.ctx_rb)   { ggml_free(dw.ctx_rb); dw.ctx_rb = nullptr; }
            dw.gf_rb = nullptr;
            dw.rb_slot = nullptr;
            return false;
        }
        dw.rb_built = true;
    }
    const int32_t slot = retained - 1;
    ggml_backend_tensor_set(dw.rb_slot, &slot, 0, sizeof(slot));
    if (ggml_backend_graph_compute_async(target_backend, dw.gf_rb) != GGML_STATUS_SUCCESS) return false;
    cache.cur_pos = pos0 + retained;
    return true;
}

// ─── DSpark Context KV Store ────────────────────────────────────────────
bool inkling2_dspark_commit_taps(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        ggml_tensor * taps_dev, // F32 [n_embd, 4, T_total]
        int pos0,
        int retained) {
    if (!w.dspark.loaded || !taps_dev || retained <= 0) return false;
    const auto & ds = w.dspark;
    const int kv_sz = ds.n_head_kv * ds.head_dim;

    // Zero-host-copy device-to-device copy of the last accepted token's 4 taps -> cache.last_taps_dev
    const size_t one_tok_bytes = (size_t) ds.n_embd * 4 * sizeof(float);
    {
        ggml_tensor src_view = *taps_dev;
        src_view.ne[0] = ds.n_embd;
        src_view.ne[1] = 4;
        src_view.ne[2] = 1;
        src_view.ne[3] = 1;
        src_view.nb[0] = sizeof(float);
        src_view.nb[1] = (size_t) ds.n_embd * sizeof(float);
        src_view.nb[2] = one_tok_bytes;
        src_view.nb[3] = one_tok_bytes;
        src_view.data  = (char *) taps_dev->data + (size_t) (retained - 1) * one_tok_bytes;
        ggml_backend_tensor_copy(&src_view, cache.last_taps_dev);
    }

    // Build / reuse per-`retained` store graph (1..8) -> ZERO graph rebuilds & ZERO HIP graph churn!
    auto & ws = cache.dspark_ws;
    ggml_context * sc = nullptr;
    ggml_gallocr_t sa = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * t_in = nullptr;
    ggml_tensor * t_pos = nullptr;
    const bool reuse = (retained >= 1 && retained <= kInkling2MaxVerify);
    if (reuse && ws.store[retained].built) {
        gf = ws.store[retained].gf;
        t_in = ws.store[retained].taps;
        t_pos = ws.store[retained].pos;
    } else {
        const size_t max_nodes = 512;
        struct ggml_init_params gp = { max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false) + (1 << 20), nullptr, true };
        sc = ggml_init(gp);
        gf = ggml_new_graph_custom(sc, max_nodes, false);
        t_in  = ggml_new_tensor_3d(sc, GGML_TYPE_F32, ds.n_embd, 4, retained);
        t_pos = ggml_new_tensor_1d(sc, GGML_TYPE_I32, retained);
        ggml_set_input(t_in);
        ggml_set_input(t_pos);

        for (int i = 0; i < ds.n_layer; ++i) {
            const auto & L = ds.layers[i];
            ggml_tensor * ctx_h = ggml_inkling2_dspark_mix_norm(sc, t_in, ds.fuse_alpha[i], L.dflare_norm, ds.rms_eps);
            ggml_tensor * kv    = ggml_mul_mat(sc, L.wkv_ctx, ctx_h); // [2*kv_sz, retained]
            ggml_tensor * k_raw = ggml_cont(sc, row_view_2d(sc, kv, 0, kv_sz));
            ggml_tensor * v_raw = ggml_cont(sc, row_view_2d(sc, kv, kv_sz, kv_sz));
            ggml_tensor * k3    = ggml_reshape_3d(sc, k_raw, ds.head_dim, ds.n_head_kv, retained);
            k3 = ggml_mul(sc, ggml_rms_norm(sc, k3, ds.rms_eps), L.k_norm_ctx);
            k3 = ggml_rope_ext(sc, k3, t_pos, nullptr, ds.head_dim, GGML_ROPE_TYPE_NEOX, 0, ds.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            ggml_tensor * v3    = ggml_reshape_3d(sc, v_raw, ds.head_dim, ds.n_head_kv, retained);
            ggml_tensor * kp = ggml_permute(sc, k3, 0, 2, 1, 3);
            ggml_tensor * vp = ggml_permute(sc, v3, 0, 2, 1, 3);
            ggml_tensor * dst_k = ggml_set_rows(sc, cache.dspark_k[i], kp, t_pos);
            ggml_tensor * dst_v = ggml_set_rows(sc, cache.dspark_v[i], vp, t_pos);
            ggml_build_forward_expand(gf, dst_k);
            ggml_build_forward_expand(gf, dst_v);
        }
        sa = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
        if (!sa || !ggml_gallocr_alloc_graph(sa, gf)) {
            if (sa) ggml_gallocr_free(sa);
            ggml_free(sc);
            return false;
        }
        if (reuse) {
            auto & sl = ws.store[retained];
            sl.ctx   = sc;
            sl.alloc = sa;
            sl.gf    = gf;
            sl.taps  = t_in;
            sl.pos   = t_pos;
            sl.built = true;
        }
    }

    // Zero-host-copy device-to-device copy of taps[0..retained-1] into t_in
    {
        ggml_tensor src_slice = *taps_dev;
        src_slice.ne[0] = ds.n_embd;
        src_slice.ne[1] = 4;
        src_slice.ne[2] = retained;
        src_slice.ne[3] = 1;
        src_slice.nb[0] = sizeof(float);
        src_slice.nb[1] = (size_t) ds.n_embd * sizeof(float);
        src_slice.nb[2] = one_tok_bytes;
        src_slice.nb[3] = one_tok_bytes * (size_t) retained;
        ggml_backend_tensor_copy(&src_slice, t_in);
    }
    std::vector<int32_t> pos(retained);
    for (int i = 0; i < retained; ++i) pos[i] = pos0 + i;
    ggml_backend_tensor_set(t_pos, pos.data(), 0, (size_t) retained * sizeof(int32_t));

    const bool ok = reuse
        ? (ggml_backend_graph_compute_async(target_backend, gf) == GGML_STATUS_SUCCESS)
        : (ggml_backend_graph_compute(target_backend, gf) == GGML_STATUS_SUCCESS);
    if (!reuse) {
        ggml_gallocr_free(sa);
        ggml_free(sc);
    }
    if (ok) cache.dspark_ctx_pos = pos0 + retained;
    return ok;
}

// ─── DSpark Block-8 + 7-Step Fused RNN Markov Chain Graph ───────────────
bool inkling2_dspark_draft(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        int32_t anchor_token,
        int anchor_pos,
        int n_draft,
        std::vector<int32_t> & out_draft_tokens,
        int max_draft) {
    out_draft_tokens.clear();
    if (!w.dspark.loaded || n_draft <= 0) return false;
    const auto & ds = w.dspark;
    const int B = ds.block_size; // 8 (1 anchor + 7 masks)
    const int n_spec = std::min(n_draft, B - 1);
    const int build_n_spec = (max_draft > n_spec) ? std::min(max_draft, B - 1) : n_spec;
    if (anchor_pos + B > cache.max_ctx + B) return false;

    // Bucket draft KV span to multiples of 256 so the draft HIP graph replays without rebuilds!
    const int kv_total = anchor_pos + B;
    const int64_t kv_span = std::min<int64_t>(cache.dspark_k[0]->ne[1], ((kv_total + 255) / 256) * 256);

    auto & ws = cache.dspark_ws;
    if (!ws.draft_built || ws.draft_kv_span != kv_span || ws.draft_n_spec < build_n_spec) {
        if (ws.alloc_draft) { ggml_gallocr_free(ws.alloc_draft); ws.alloc_draft = nullptr; }
        if (ws.ctx_draft)   { ggml_free(ws.ctx_draft); ws.ctx_draft = nullptr; }
        ws.draft_built = false;

        const size_t max_nodes = 2048;
        struct ggml_init_params gp = { max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false) + (4 << 20), nullptr, true };
        ws.ctx_draft = ggml_init(gp);
        ws.gf_draft  = ggml_new_graph_custom(ws.ctx_draft, max_nodes, false);
        ggml_context * c = ws.ctx_draft;

        ws.draft_anchor     = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
        ws.draft_pos        = ggml_new_tensor_1d(c, GGML_TYPE_I32, B);
        ws.draft_zero_state = ggml_new_tensor_2d(c, GGML_TYPE_F32, ds.markov_rank, 1);
        ggml_set_input(ws.draft_anchor);
        ggml_set_input(ws.draft_pos);
        ggml_set_input(ws.draft_zero_state);

        // Build global & local SWA masks directly on GPU (zero CPU fill & zero PCIe upload at 262K!)
        ggml_tensor * masks_both = inkling2_dspark_build_masks(c, ws.draft_pos, kv_span, B, ds.swa_window);
        ws.draft_mask       = ggml_view_2d(c, masks_both, kv_span, B, masks_both->nb[1], 0);
        ws.draft_mask_local = ggml_view_2d(c, masks_both, kv_span, B, masks_both->nb[1], masks_both->nb[2]);

        // Anchor embedding (normed) + 7 copies of mask_embed -> [n_embd, 8]
        ggml_tensor * anch = rms_norm_w(c, ggml_get_rows(c, w.tok_embd, ws.draft_anchor), w.tok_norm, w.rms_eps);
        ggml_tensor * masks = ggml_repeat_4d(c, ggml_reshape_2d(c, ds.mask_embed, ds.n_embd, 1),
                                             ds.n_embd, B - 1, 1, 1);
        ggml_tensor * x = ggml_concat(c, anch, masks, 1); // [n_embd, 8]

        const int q_sz  = ds.n_head * ds.head_dim;
        const int kv_sz = ds.n_head_kv * ds.head_dim;
        const float attn_scale = 1.0f / std::sqrt((float) ds.head_dim);

        for (int i = 0; i < ds.n_layer; ++i) {
            const auto & L = ds.layers[i];
            ggml_tensor * xn = rms_norm_w(c, x, L.inp_norm, ds.rms_eps);
            ggml_tensor * qkv = ggml_mul_mat(c, L.wqkv, xn); // [q_sz + 2*kv_sz, 8]
            ggml_tensor * q3 = ggml_reshape_3d(c, ggml_cont(c, row_view_2d(c, qkv, 0, q_sz)), ds.head_dim, ds.n_head, B);
            ggml_tensor * k3 = ggml_reshape_3d(c, ggml_cont(c, row_view_2d(c, qkv, q_sz, kv_sz)), ds.head_dim, ds.n_head_kv, B);
            ggml_tensor * v3 = ggml_reshape_3d(c, ggml_cont(c, row_view_2d(c, qkv, q_sz + kv_sz, kv_sz)), ds.head_dim, ds.n_head_kv, B);

            q3 = ggml_mul(c, ggml_rms_norm(c, q3, ds.rms_eps), L.q_norm);
            k3 = ggml_mul(c, ggml_rms_norm(c, k3, ds.rms_eps), L.k_norm);
            q3 = ggml_rope_ext(c, q3, ws.draft_pos, nullptr, ds.head_dim, GGML_ROPE_TYPE_NEOX, 0, ds.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k3 = ggml_rope_ext(c, k3, ws.draft_pos, nullptr, ds.head_dim, GGML_ROPE_TYPE_NEOX, 0, ds.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            // Write the block's own K/V at draft_pos[0..7] into draft KV cache
            ggml_tensor * kp = ggml_permute(c, k3, 0, 2, 1, 3);
            ggml_tensor * vp = ggml_permute(c, v3, 0, 2, 1, 3);
            ggml_tensor * k_upd = ggml_set_rows(c, cache.dspark_k[i], kp, ws.draft_pos);
            ggml_tensor * v_upd = ggml_set_rows(c, cache.dspark_v[i], vp, ws.draft_pos);

            // Bidirectional FlashAttention over [0 .. kv_span) with local SWA (layers 0,2) or global (layer 1) mask
            ggml_tensor * k_view = ggml_view_3d(c, k_upd, ds.head_dim, kv_span, ds.n_head_kv, k_upd->nb[1], k_upd->nb[2], 0);
            ggml_tensor * v_view = ggml_view_3d(c, v_upd, ds.head_dim, kv_span, ds.n_head_kv, v_upd->nb[1], v_upd->nb[2], 0);
            ggml_tensor * qp = ggml_permute(c, q3, 0, 2, 1, 3); // [D, B, Hq]
            ggml_tensor * mask_i = L.is_local ? ws.draft_mask_local : ws.draft_mask;
            ggml_tensor * attn = ggml_flash_attn_ext(c, qp, k_view, v_view, mask_i, attn_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
            attn = ggml_reshape_2d(c, attn, q_sz, B);

            x = ggml_add(c, x, ggml_mul_mat(c, L.wo, attn));
            ggml_tensor * mn = rms_norm_w(c, x, L.post_norm, ds.rms_eps);
            ggml_tensor * gu = ggml_mul_mat(c, L.w_gate_up, mn);
            ggml_tensor * ffn = ggml_mul_mat(c, L.w_down, ggml_swiglu(c, gu));
            x = ggml_add(c, x, ffn);
        }

        // Final norm -> take mask positions 1..build_n_spec ([n_embd, build_n_spec])
        ggml_tensor * h_all = rms_norm_w(c, x, ds.norm, ds.rms_eps);
        ggml_tensor * h_spec = ggml_cont(c, ggml_view_2d(c, h_all, ds.n_embd, build_n_spec, h_all->nb[1], h_all->nb[1]));

        // Base logits for build_n_spec positions in ONE lm_head matmul (using fast Q4_0 draft head when loaded)!
        ggml_tensor * draft_lm = ds.lm_head_q4 ? ds.lm_head_q4 : w.output;
        ggml_tensor * base_logits = ggml_mul_mat(c, draft_lm, h_spec); // [V, build_n_spec]

        // Recurrent Markov head context for build_n_spec steps in ONE matmul!
        ggml_tensor * f_rnn = ggml_inkling2_dspark_mix_norm(
            c, cache.last_taps_dev, ds.fuse_alpha[3], ds.markov_dflare_norm, ds.rms_eps); // [n_embd, 1]
        ggml_tensor * f_rep = (build_n_spec > 1)
            ? ggml_repeat_4d(c, f_rnn, ds.n_embd, build_n_spec, 1, 1)
            : f_rnn;
        ggml_tensor * ctx_in = ggml_concat(c, h_spec, f_rep, 0); // [2*n_embd, build_n_spec]
        ggml_tensor * rnn_ctx = ggml_add(c, ggml_mul_mat(c, ds.rnn_proj_ctx, ctx_in), ds.rnn_bias); // [3*R, build_n_spec]

        // Full 201k-vocab exact Markov argmax (100% full acceptance + Q8_0 MALL-resident <0.06 ms/step!).
        const bool use_topk_markov = (std::getenv("LUCE_INKLING2_DSPARK_TOPK") != nullptr &&
                                      std::atoi(std::getenv("LUCE_INKLING2_DSPARK_TOPK")) != 0);
        ggml_tensor * topk_cands = (use_topk_markov && ds.fold_weight->type == GGML_TYPE_F16)
            ? ggml_inkling2_dspark_topk_cands(c, base_logits, w.n_vocab_unpadded) // I32 [1024, build_n_spec]
            : nullptr;

        // Unrolled build_n_spec-step fused RNN Markov chain IN ONE GRAPH (zero host round-trips!)
        ggml_tensor * state = ws.draft_zero_state;
        ggml_tensor * prev  = ws.draft_anchor; // I32 [1]

        for (int step = 0; step < build_n_spec; ++step) {
            ggml_tensor * ctx_s = ggml_view_2d(c, rnn_ctx, 3 * ds.markov_rank, 1,
                                               rnn_ctx->nb[1], (size_t) step * rnn_ctx->nb[1]);
            ggml_tensor * st_lat = inkling2_dspark_markov_rnn_step(
                c, prev, state, ds.markov_w1, ds.rnn_proj_step, ctx_s); // [2*R, 1]
            state = ggml_view_2d(c, st_lat, ds.markov_rank, 1, st_lat->nb[1], 0);
            ggml_tensor * latent = ggml_view_2d(c, st_lat, ds.markov_rank, 1, st_lat->nb[1], (size_t) ds.markov_rank * sizeof(float));

            ggml_tensor * base_s = ggml_view_2d(c, base_logits, w.n_vocab_padded, 1,
                                                base_logits->nb[1], (size_t) step * base_logits->nb[1]);
            if (topk_cands) {
                ggml_tensor * cands_s = ggml_view_1d(c, topk_cands, 1024, (size_t) step * topk_cands->nb[1]);
                prev = ggml_inkling2_dspark_topk_score(c, cands_s, base_s, latent, ds.fold_weight, ds.fold_bias, w.logit_softcap);
            } else {
                prev = ggml_inkling2_dspark_markov_argmax(c, base_s, latent, ds.fold_weight, ds.fold_bias, w.n_vocab_unpadded);
            }
            ws.draft_out_ids[step] = prev;
        }
        // Pack all build_n_spec drafted token IDs into ONE contiguous [build_n_spec] tensor -> 1 single D2H readback!
        ggml_tensor * packed_ids = ws.draft_out_ids[0];
        for (int step = 1; step < build_n_spec; ++step) {
            packed_ids = ggml_concat(c, packed_ids, ws.draft_out_ids[step], 0);
        }
        ws.draft_out_all = packed_ids;
        ggml_set_output(ws.draft_out_all);
        ggml_build_forward_expand(ws.gf_draft, ws.draft_out_all);

        ws.alloc_draft = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
        if (!ws.alloc_draft || !ggml_gallocr_alloc_graph(ws.alloc_draft, ws.gf_draft)) {
            std::fprintf(stderr, "[inkling2] DSpark draft graph alloc failed\n");
            if (ws.alloc_draft) { ggml_gallocr_free(ws.alloc_draft); ws.alloc_draft = nullptr; }
            if (ws.ctx_draft)   { ggml_free(ws.ctx_draft); ws.ctx_draft = nullptr; }
            ws.gf_draft = nullptr;
            return false;
        }
        ggml_backend_tensor_memset(ws.draft_zero_state, 0, 0, ggml_nbytes(ws.draft_zero_state));
        ws.draft_kv_span = kv_span;
        ws.draft_n_spec  = build_n_spec;
        ws.draft_built   = true;
    }

    // Set tiny runtime inputs ONLY: anchor token (4B) + block positions (32B)! Masks are built on-device!
    int32_t pos[8];
    for (int i = 0; i < B; ++i) pos[i] = anchor_pos + i;
    ggml_backend_tensor_set(ws.draft_anchor, &anchor_token, 0, sizeof(anchor_token));
    ggml_backend_tensor_set(ws.draft_pos, pos, 0, sizeof(pos));

    if (ggml_backend_graph_compute(target_backend, ws.gf_draft) != GGML_STATUS_SUCCESS) {
        return false;
    }

    int32_t all_ids[8] = {};
    ggml_backend_tensor_get(ws.draft_out_all, all_ids, 0, (size_t) ws.draft_n_spec * sizeof(int32_t));
    out_draft_tokens.assign(all_ids, all_ids + n_spec);
    return true;
}

} // namespace luce::common
