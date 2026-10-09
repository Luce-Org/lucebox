// Qwen4Exp forward graph — see qwen4exp_graph.h.
// embed -> hc_init -> per-layer ([PLE], hc_mix(attn) -> linear|QSA -> hc_combine,
// hc_mix(ffn) -> MoE -> hc_combine) -> hc_mix(output) -> lm_head. Single sequence.

#include "qwen4exp_graph.h"
#include "qwen4exp_chunk.h"

#include "common/cuda_graph_overrides.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <utility>
#include <vector>

namespace luce::common {
namespace {

// Node budget of one forward graph: its metadata context, its graph and a split scheduler's hash set.
constexpr int kQwen4ExpGraphNodes = 200000;

size_t ring_align_up(size_t value) {
    const size_t remainder = value % 256;
    return remainder == 0 ? value : value + 256 - remainder;
}

static void graph_memory(ggml_backend_t backend, ggml_context * ctx, ggml_cgraph * gf,
                         bool gfx1151, Qwen4ExpGraphMemory & memory) {
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve_n_size(alloc, gf, nullptr, nullptr, &memory.graph);
    ggml_gallocr_free(alloc);
    memory.metadata = ggml_get_mem_size(ctx);
    size_t activation = 0, largest = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const auto * node = ggml_graph_node(gf, i);
        largest = std::max(largest, ggml_nbytes(node));
        if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
            activation = std::max(activation, (size_t) ggml_nelements(node->src[1]));
        }
    }
    // MMB retains four BF16, four F16 and four producer buffers. Account
    // for those plus one Q8 activation and two largest-node scratch buffers
    // (conversion/attention and a retired pool allocation during growth).
    memory.scratch = gfx1151 ? (4 + 4 + 4) * 2 * activation + activation + 2 * largest : 2 * largest;
}

// Grow-only: slots keep their addresses across forwards, so captured CUDA graphs stay valid.
bool qwen4exp_input_ring_reserve(Qwen4ExpInputRing & ring,
                                 ggml_backend_buffer_type_t host_buft,
                                 size_t embd_bytes, size_t pos_bytes,
                                 size_t ple_bytes, size_t mask_bytes) {
    if (ring.buf != nullptr &&
        embd_bytes <= ring.embd_cap && pos_bytes <= ring.pos_cap &&
        ple_bytes <= ring.ple_cap && mask_bytes <= ring.mask_cap) {
        return true;
    }

    ring.embd_cap = std::max(ring.embd_cap, ring_align_up(embd_bytes));
    ring.pos_cap  = std::max(ring.pos_cap,  ring_align_up(pos_bytes));
    ring.ple_cap  = std::max(ring.ple_cap,  ring_align_up(ple_bytes));
    ring.mask_cap = std::max(ring.mask_cap, ring_align_up(mask_bytes));
    ring.embd_off = 0;
    ring.pos_off  = ring.embd_off + ring.embd_cap;
    ring.ple_off  = ring.pos_off + ring.pos_cap;
    ring.mask_off = ring.ple_off + ring.ple_cap;
    ring.slot_bytes = ring.mask_off + ring.mask_cap;

    ggml_backend_buffer_t buf =
        ggml_backend_buft_alloc_buffer(host_buft, 2 * ring.slot_bytes);
    if (buf == nullptr || ggml_backend_buffer_get_base(buf) == nullptr) {
        if (buf != nullptr) ggml_backend_buffer_free(buf);
        if (ring.buf != nullptr) ggml_backend_buffer_free(ring.buf);
        ring.buf = nullptr;
        ring.base = nullptr;
        ring.enabled = false;
        std::fprintf(stderr,
            "[qwen4exp] pinned input ring allocation failed (%zu bytes); "
            "falling back to staged graph inputs\n",
            2 * ring.slot_bytes);
        return false;
    }
    if (ring.buf != nullptr) ggml_backend_buffer_free(ring.buf);
    ring.buf = buf;
    ring.base = static_cast<char *>(ggml_backend_buffer_get_base(buf));
    return true;
}

ggml_tensor * mm(ggml_context * c, ggml_tensor * w, ggml_tensor * x, float s = 1.0f) {
    ggml_tensor * y = ggml_mul_mat(c, w, x);
    return s == 1.0f ? y : ggml_scale(c, y, s);
}

// One op: the hc-combine-norm matcher requires GGML_OP_REPEAT, not a concat chain.
ggml_tensor * repeat_dim1(ggml_context * c, ggml_tensor * x, int64_t hc) {
    return ggml_repeat_4d(c, x, x->ne[0], hc, x->ne[2], x->ne[3]);
}

// Shared by the standalone draft/oracle and the device-only prefill slices.
static ggml_tensor * mtp_input(ggml_context * ctx, const Qwen4ExpWeights & w,
                               ggml_tensor * inp_emb, ggml_tensor * inp_h) {
    const int64_t H = w.n_embd, hc = w.n_hc, T = inp_emb->ne[1];
    // nextn front: stream s of the new residual is eh_proj [enorm(e) ; hnorm(h)_s]; hnorm spans all HC streams.
    ggml_tensor * e = ggml_mul(ctx, ggml_rms_norm(ctx, inp_emb, w.rms_eps), w.mtp_enorm);
    ggml_tensor * h = ggml_mul(ctx, ggml_rms_norm(ctx, inp_h, w.rms_eps), w.mtp_hnorm);
    ggml_tensor * x = ggml_concat(ctx, repeat_dim1(ctx, ggml_reshape_3d(ctx, e, H, 1, T), hc),
                                  ggml_reshape_3d(ctx, h, H, hc, T), 0);                        // [2H, hc, T]
    return ggml_reshape_3d(ctx, mm(ctx, w.mtp_eh_proj, ggml_reshape_2d(ctx, x, 2 * H, hc * T)), H, hc, T);
}

// ── Hyper-connections ───────────────────────────────────────────────────

static ggml_tensor * hc_mix_body(ggml_context * c, ggml_tensor * xn, ggml_tensor * w_down,
                     ggml_tensor * w_up, ggml_tensor * w_inject, ggml_tensor ** inject,
                     int64_t n_embd, int64_t hc) {
    const int64_t nt = xn->ne[1];

    ggml_tensor * lo = mm(c, w_down, xn);                 // [hc_lr, nt]
    lo = ggml_silu(c, ggml_scale(c, lo, 1.0f / (float) hc));
    ggml_tensor * gate = ggml_sigmoid(c, mm(c, w_up, lo));// [hc_dim, nt]

    ggml_tensor * gated = ggml_mul(c, xn, gate);
    gated = ggml_reshape_3d(c, gated, n_embd, hc, nt);

    const size_t row = ggml_row_size(gated->type, n_embd);
    ggml_tensor * mixed = ggml_cont(c, ggml_view_2d(c, gated, n_embd, nt, row * hc, 0));
    for (int64_t s = 1; s < hc; ++s) {
        ggml_tensor * v = ggml_view_2d(c, gated, n_embd, nt, row * hc, row * s);
        mixed = ggml_add(c, mixed, v);
    }
    mixed = ggml_scale(c, mixed, 1.0f / (float) hc);

    if (inject) {
        *inject = mm(c, w_inject, xn);                    // [hc, nt]
    }
    return mixed;
}

ggml_tensor * hc_mix(ggml_context * c, ggml_tensor * x, ggml_tensor * w_norm,
                     ggml_tensor * w_down, ggml_tensor * w_up, ggml_tensor * w_inject,
                     ggml_tensor ** inject, int64_t n_embd, int64_t hc, float eps) {
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    ggml_tensor * xn = ggml_rms_norm(c, x, eps);          // per (hc, token)
    xn = ggml_reshape_2d(c, xn, hc_dim, nt);
    xn = ggml_mul(c, xn, w_norm);                         // [hc_dim, nt] gamma

    return hc_mix_body(c, xn, w_down, w_up, w_inject, inject, n_embd, hc);
}

static ggml_tensor * hc_mix_from_xn(ggml_context * c, ggml_tensor * xn3, ggml_tensor * w_down,
                     ggml_tensor * w_up, ggml_tensor * w_inject, ggml_tensor ** inject,
                     int64_t n_embd, int64_t hc) {
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = xn3->ne[2];
    ggml_tensor * xn = ggml_reshape_2d(c, xn3, hc_dim, nt);
    return hc_mix_body(c, xn, w_down, w_up, w_inject, inject, n_embd, hc);
}

ggml_tensor * hc_combine(ggml_context * c,
                         ggml_tensor * residual, ggml_tensor * block_out, ggml_tensor * inject,
                         int64_t n_embd, int64_t hc, int64_t nt) {
    ggml_tensor * w = ggml_sigmoid(c, ggml_scale(c, inject, 1.0f / (float) hc));
    w = ggml_scale(c, w, 2.0f);
    w = ggml_reshape_3d(c, w, 1, hc, nt);

    ggml_tensor * b = repeat_dim1(c, ggml_reshape_3d(c, block_out, n_embd, 1, nt), hc);

    return ggml_add(c, residual, ggml_mul(c, b, w));
}

// Packed [n_embd, hc, nt, 2]: channel 0 is the new residual, channel 1 the normalized stream.
static ggml_tensor * hc_combine_norm(ggml_context * c, ggml_tensor * inject, ggml_tensor * residual,
                         ggml_tensor * block_out, ggml_tensor * gamma,
                         int64_t n_embd, int64_t hc, int64_t nt, float eps) {
    return ggml_hc_combine_norm(c, inject, residual,
        ggml_reshape_3d(c, block_out, n_embd, 1, nt), gamma,
        1.0f / (float) hc, 0.0f, 2.0f, 0.0f, eps);
}

static ggml_tensor * hc_norm_res(ggml_context * c, ggml_tensor * fused,
                         int64_t n_embd, int64_t hc, int64_t nt) {
    return ggml_view_4d(c, fused, n_embd, hc, nt, 1, fused->nb[1], fused->nb[2], fused->nb[3], 0);
}

static ggml_tensor * hc_norm_xn(ggml_context * c, ggml_tensor * fused,
                         int64_t n_embd, int64_t hc, int64_t nt) {
    return ggml_view_4d(c, fused, n_embd, hc, nt, 1, fused->nb[1], fused->nb[2], fused->nb[3],
        (size_t) n_embd * hc * nt * sizeof(float));
}

// ── MoE FFN: 512 experts top-10 (softmax), gated shared expert ──────────

// Un-combined MoE outputs, for folding the combine into the next HC_COMBINE_NORM (ggml_hc_combine_norm_moe).
struct Qwen4ExpMoeParts {
    ggml_tensor * down         = nullptr;   // [n_embd, n_used, T]
    ggml_tensor * weights      = nullptr;   // [n_used, T]
    ggml_tensor * shared       = nullptr;   // [n_embd, T], before the sigmoid gate
    ggml_tensor * shared_logit = nullptr;   // [1, T]
};

// Split mode: the router, the gated shared expert and the expert input stay on
// the target; build_moe_routed runs the routed experts on the expert device.
struct Qwen4ExpMoeRoute {
    ggml_tensor * xin    = nullptr;   // expert input, F16 for prompt chunks (half the link bytes)
    ggml_tensor * sel    = nullptr;   // [n_used, T], the expert device's picks (hot ones -1)
    ggml_tensor * hot    = nullptr;   // [n_used, T], hot-stack slots on the target (cold ones -1)
    ggml_tensor * wsel   = nullptr;   // [n_used, T]
    ggml_tensor * shared = nullptr;   // gated shared expert, [n_embd, T]
};

// The trunk layer index of L, or -1 for a layer outside the trunk (the MTP draft layer).
static int layer_index(const Qwen4ExpLayer & L, const Qwen4ExpWeights & w) {
    const ptrdiff_t il = &L - w.layers.data();
    return il >= 0 && il < (ptrdiff_t) w.layers.size() ? (int) il : -1;
}

// Split mode: forwards of up to this many rows (decode, verify, short batches)
// share one scheduler and keep F32 activations on the link; longer prompt chunks
// cross it as F16.
constexpr int64_t kSplitShortRows = 64;

// The router: the top-k picks and their renormalized weights, [n_used, T] each.
static void moe_router(ggml_context * c, ggml_tensor * cur, const Qwen4ExpLayer & L, const Qwen4ExpWeights & w,
                       ggml_tensor ** sel, ggml_tensor ** wsel) {
    const int64_t n_tokens = cur->ne[1];
    ggml_tensor * logits = mm(c, L.ffn_gate_inp, cur);      // [n_expert, T]
    ggml_tensor * probs  = ggml_soft_max(c, logits);
    *sel = ggml_argsort_top_k(c, probs, (int) w.n_expert_used);
    ggml_tensor * probs3 = ggml_reshape_3d(c, probs, 1, w.n_expert, n_tokens);
    *wsel = ggml_reshape_2d(c, ggml_get_rows(c, probs3, *sel), w.n_expert_used, n_tokens);
    *wsel = ggml_div(c, *wsel, ggml_clamp(c, ggml_sum_rows(c, *wsel), 6.103515625e-5f, INFINITY));
}

// The shared expert, [n_embd, T], and its gate logit, [1, T], before the sigmoid.
static void moe_shared(ggml_context * c, ggml_tensor * cur, const Qwen4ExpLayer & L,
                       ggml_tensor ** shared, ggml_tensor ** shared_logit) {
    ggml_tensor * sh_gate = mm(c, L.ffn_gate_shexp, cur);
    ggml_tensor * sh_up   = mm(c, L.ffn_up_shexp, cur);
    ggml_tensor * sh_gu   = ggml_swiglu_split(c, sh_gate, sh_up);
    *shared       = mm(c, L.ffn_down_shexp, sh_gu);
    *shared_logit = mm(c, L.ffn_gate_inp_shexp, cur);
}

// Gate, up and down of the routed experts `ids` picks: [n_embd, n_used, T].
static ggml_tensor * moe_experts(ggml_context * c, ggml_tensor * gate_exps, ggml_tensor * up_exps,
                                 ggml_tensor * down_exps, ggml_tensor * x3, ggml_tensor * ids,
                                 std::vector<ggml_tensor *> * nodes = nullptr) {
    ggml_tensor * gate = ggml_mul_mat_id(c, gate_exps, x3, ids);
    ggml_tensor * up   = ggml_mul_mat_id(c, up_exps,   x3, ids);
    ggml_tensor * gu   = ggml_swiglu_split(c, gate, up);
    ggml_tensor * down = ggml_mul_mat_id(c, down_exps, gu, ids);
    if (nodes) nodes->insert(nodes->end(), {gate, up, gu, down});
    return down;
}

static Qwen4ExpMoeRoute build_moe_route(ggml_context * c, ggml_tensor * cur,
                                        const Qwen4ExpLayer & L, const Qwen4ExpWeights & w) {
    const int64_t n_tokens = cur->ne[1];
    Qwen4ExpMoeRoute r;
    const int il = layer_index(L, w);
    moe_router(c, cur, L, w, &r.sel, &r.wsel);
    ggml_tensor * shared_logit = nullptr;
    moe_shared(c, cur, L, &r.shared, &shared_logit);
    r.shared = ggml_mul(c, r.shared, ggml_sigmoid(c, shared_logit));
    // The expert kernels convert their activations to F16 anyway.
    r.xin = n_tokens > kSplitShortRows ? ggml_cast(c, cur, GGML_TYPE_F16) : cur;
    // Decode and verify steps with hot experts: each pick runs on one device,
    // masked (-1) on the other. Single-token steps take the same route as
    // verify rows, so every generated token computes its hot picks with the
    // same tokenwise kernel whatever the verify width, and greedy text does not
    // depend on the adaptive width. Prompt chunks keep every pick on the expert
    // device: only the matvec kernels take masked ids on gfx1151 (MMQ faults
    // on them).
    if (n_tokens <= QWEN4EXP_MTP_MAX_VERIFY && il >= 0 && il < (int) w.hot_lut.size() && w.hot_lut[il]) {
        ggml_tensor * flat = ggml_reshape_1d(c, ggml_is_contiguous(r.sel) ? r.sel : ggml_cont(c, r.sel),
                                             w.n_expert_used * n_tokens);
        r.hot = ggml_reshape_2d(c, ggml_get_rows(c, w.hot_lut[il], flat), w.n_expert_used, n_tokens);
        r.sel = ggml_reshape_2d(c, ggml_get_rows(c, w.cold_lut[il], flat), w.n_expert_used, n_tokens);
    }
    return r;
}

// The hot experts' weighted sum plus the shared expert, on the target while
// the expert device runs the cold picks.
static ggml_tensor * build_moe_hot(ggml_context * c, const Qwen4ExpMoeRoute & r,
                                   const MoeHybridLayerStorage & hot, const Qwen4ExpWeights & w) {
    ggml_tensor * x3 = ggml_reshape_3d(c, r.xin, w.n_embd, 1, r.hot->ne[1]);
    ggml_tensor * down = moe_experts(c, hot.gate_hot, hot.up_hot, hot.down_hot, x3, r.hot);
    return ggml_ds4_moe_fused_combine_shared(c, down, r.wsel, r.shared);
}

// The routed experts and their weighted sum, on the expert device: only the
// [n_embd, T] sum crosses back, not [n_embd, n_used, T].
static ggml_tensor * build_moe_routed(ggml_context * c, const Qwen4ExpMoeRoute & r,
                                      const Qwen4ExpLayer & L, const Qwen4ExpWeights & w,
                                      std::vector<ggml_tensor *> & expert_nodes) {
    const int64_t n_tokens = r.sel->ne[1];
    ggml_tensor * xin = r.xin->type == GGML_TYPE_F16 ? ggml_cast(c, r.xin, GGML_TYPE_F32) : r.xin;
    if (xin != r.xin) expert_nodes.push_back(xin);
    ggml_tensor * cur3 = ggml_reshape_3d(c, xin, w.n_embd, 1, n_tokens);
    ggml_tensor * down = moe_experts(c, L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps, cur3, r.sel, &expert_nodes);
    ggml_tensor * routed = ggml_ds4_moe_fused_combine_shared(c, down, r.wsel, nullptr);
    expert_nodes.push_back(routed);
    if (n_tokens > kSplitShortRows) {   // prompt chunks: the sum crosses back as F16 too
        routed = ggml_cast(c, routed, GGML_TYPE_F16);
        expert_nodes.push_back(routed);
    }
    return routed;
}

// A layer whose routed expert stacks were placed on the expert device (the MTP
// draft layer's stay on the target).
static bool qwen4exp_split_layer(const Qwen4ExpLayer & L, const Qwen4ExpWeights & w) {
    return w.expert_buf && L.ffn_gate_exps && L.ffn_gate_exps->buffer == w.expert_buf;
}

// The routed sum (F16 for prompt chunks) joins the shared expert on the target.
static ggml_tensor * moe_join(ggml_context * c, ggml_tensor * routed, ggml_tensor * shared) {
    return ggml_add(c, routed->type == GGML_TYPE_F32 ? routed : ggml_cast(c, routed, GGML_TYPE_F32), shared);
}

ggml_tensor * build_moe(ggml_context * c, ggml_tensor * cur,
                        const Qwen4ExpLayer & L, const Qwen4ExpWeights & w,
                        Qwen4ExpMoeParts * parts = nullptr,
                        std::vector<ggml_tensor *> * expert_nodes = nullptr) {
    if (expert_nodes && qwen4exp_split_layer(L, w)) {
        const Qwen4ExpMoeRoute r = build_moe_route(c, cur, L, w);
        ggml_tensor * routed = build_moe_routed(c, r, L, w, *expert_nodes);
        return moe_join(c, routed, r.hot ? build_moe_hot(c, r, w.hot->layers[layer_index(L, w)], w) : r.shared);
    }
    ggml_tensor * sel = nullptr, * wsel = nullptr;
    moe_router(c, cur, L, w, &sel, &wsel);
    ggml_tensor * cur3 = ggml_reshape_3d(c, cur, w.n_embd, 1, cur->ne[1]);
    ggml_tensor * down = moe_experts(c, L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps, cur3, sel);
    ggml_tensor * shared = nullptr, * shared_logit = nullptr;
    moe_shared(c, cur, L, &shared, &shared_logit);
    if (parts) {   // the caller folds the combine into the next HC_COMBINE_NORM
        *parts = { down, wsel, shared, shared_logit };
        return nullptr;
    }
    ggml_tensor * shared_gate = ggml_sigmoid(c, shared_logit);
    shared = ggml_mul(c, shared, shared_gate);   // [n_embd,T] * [1,T] broadcasts over dim 0
    return ggml_ds4_moe_fused_combine_shared(c, down, wsel, shared);
}

// ── Linear attention: gated delta net (36 layers) ───────────────────────

// spec_states / spec_conv (verify forward): receive the recurrent state after every token and the conv history after
// every token, so any accepted prefix can be retained.
ggml_tensor * build_linear_attn(ggml_context * c, ggml_cgraph * gf, ggml_tensor * cur,
                                const Qwen4ExpLayer & L, const Qwen4ExpWeights & w,
                                ggml_tensor * ssm_state, ggml_tensor * conv_state,
                                ggml_tensor * spec_states = nullptr, ggml_tensor * spec_conv = nullptr) {
    const int64_t D      = w.ssm_d_state;              // 128
    const int64_t Hk     = w.ssm_n_group;              // 16
    const int64_t Hv     = w.linear_value_heads;       // 48
    const int64_t d_in   = w.ssm_d_inner;              // 6144
    const int64_t T      = cur->ne[1];
    const int64_t kernel = w.ssm_d_conv;
    const int64_t conv_channels = 2 * Hk * D + d_in;   // 10240
    const float   eps    = w.rms_eps;

    ggml_tensor * qkv = mm(c, L.attn_qkv, cur);        // [conv_channels, T]
    ggml_tensor * z   = mm(c, L.attn_gate, cur);       // [d_in, T]

    // Decode, draft and verify steps (up to 8 rows) let the recurrence apply sigmoid(beta) and softplus(alpha + dt_bias)
    // * A to the raw projections, the same expressions as the ops below. Prompt chunks on gfx1151 need the ops: the
    // tiled recurrence takes finished gates.
    const bool step = T <= QWEN4EXP_MTP_MAX_VERIFY;
    const bool raw_gates = step && L.ssm_gate_ba;
    ggml_tensor * beta = ggml_reshape_4d(c, mm(c, L.ssm_beta, cur), 1, Hv, T, 1);
    ggml_tensor * gate;
    if (raw_gates) {
        gate = ggml_reshape_4d(c, mm(c, L.ssm_alpha, cur), 1, Hv, T, 1);
    } else {
        beta = ggml_sigmoid(c, beta);
        ggml_tensor * alpha = ggml_reshape_3d(c, mm(c, L.ssm_alpha, cur), Hv, T, 1);
        alpha = ggml_softplus(c, ggml_add(c, alpha, L.ssm_dt_bias));
        gate = ggml_reshape_4d(c, ggml_mul(c, alpha, L.ssm_a), 1, Hv, T, 1);
    }

    ggml_tensor * hist = ggml_reshape_3d(c, conv_state, kernel - 1, conv_channels, 1);
    // Keep the transpose as a view: the fused concat+transpose kernel keys off src1->nb[1] == sizeof(float).
    ggml_tensor * qkv_t = ggml_transpose(c, ggml_reshape_2d(c, qkv, conv_channels, T));
    ggml_tensor * conv_input = ggml_concat(c, hist, qkv_t, 0);

    // Steps write the conv history with one strided copy each: the new history, and in a verify step every token's
    // window at once (window t starts t+1 columns in, so the windows overlap and the source view is widened after it
    // is made). Prompt chunks keep the CONT copies the gfx1151 conv fusion matches.
    // nb[0] is the element size, so the tail offset is T*nb[0], NOT T*nb[1].
    ggml_tensor * new_hist = ggml_view_3d(c, conv_input, kernel - 1, conv_channels, 1,
        conv_input->nb[1], conv_input->nb[2], (size_t) T * conv_input->nb[0]);
    ggml_build_forward_expand(gf, ggml_cpy(c, step ? new_hist : ggml_cont(c, new_hist),
        ggml_reshape_3d(c, conv_state, kernel - 1, conv_channels, 1)));
    if (spec_conv && step) {
        ggml_tensor * windows = ggml_view_3d(c, conv_input, kernel - 1, conv_channels, 1,
            conv_input->nb[1], conv_input->nb[2], conv_input->nb[0]);
        windows->ne[2] = T;
        windows->nb[2] = conv_input->nb[0];
        windows->nb[3] = windows->nb[2] * T;
        ggml_build_forward_expand(gf, ggml_cpy(c, windows, ggml_view_3d(c, spec_conv, kernel - 1, conv_channels, T,
            spec_conv->nb[1], spec_conv->nb[2], 0)));
    }
    for (int64_t t = 0; spec_conv && !step && t < T; ++t) {
        ggml_build_forward_expand(gf, ggml_cpy(c, ggml_cont(c, ggml_view_3d(c, conv_input, kernel - 1, conv_channels, 1,
            conv_input->nb[1], conv_input->nb[2], (t + 1) * conv_input->nb[0])),
            ggml_view_3d(c, spec_conv, kernel - 1, conv_channels, 1,
                spec_conv->nb[1], spec_conv->nb[2], t * spec_conv->nb[2])));
    }

    ggml_tensor * conv_op = ggml_ssm_conv(c, conv_input, L.ssm_conv1d);
    // The gfx1151 fusion reads CONCAT's input at this later node. src[3] is
    // unused by ordinary SSM_CONV and gives the allocator the real lifetime.
    conv_op->src[3] = qkv_t;
    ggml_tensor * conv = ggml_silu(c, conv_op);

    const size_t esz     = ggml_element_size(conv);
    const size_t tstride = (size_t) conv_channels * esz;
    // Upstream build_gdn_l2_norm is rms_norm(x, eps/n) * (1/sqrt(n)) == x/sqrt(sum(x^2)+eps).
    // ggml_l2_norm instead rounds x*rsqrtf(max(sum(x^2), eps^2)), a ~1e-6 relative difference
    // that seeds the GDN recurrence and flips MoE routing.
    ggml_tensor * q_c;
    ggml_tensor * k_c;
    if (step) {   // q and k are adjacent in the conv output: one norm and one scale over both, the same per-row math
        ggml_tensor * qk_c = ggml_scale(c, ggml_rms_norm(c, ggml_view_3d(c, conv, D, 2 * Hk, T, D * esz, tstride, 0),
            eps / (float) D), 1.0f / sqrtf((float) D));
        q_c = ggml_view_3d(c, qk_c, D, Hk, T, qk_c->nb[1], qk_c->nb[2], 0);
        k_c = ggml_view_3d(c, qk_c, D, Hk, T, qk_c->nb[1], qk_c->nb[2], Hk * qk_c->nb[1]);
    } else {
        ggml_tensor * q_raw = ggml_view_3d(c, conv, D, Hk, T, D * esz, tstride, 0);
        ggml_tensor * k_raw = ggml_view_3d(c, conv, D, Hk, T, D * esz, tstride, D * Hk * esz);
        q_c = ggml_scale(c, ggml_rms_norm(c, q_raw, eps / (float) D), 1.0f / sqrtf((float) D));
        k_c = ggml_scale(c, ggml_rms_norm(c, k_raw, eps / (float) D), 1.0f / sqrtf((float) D));
    }
    ggml_tensor * v_c = ggml_view_3d(c, conv, D, Hv, T, D * esz, tstride, 2 * D * Hk * esz);

    ggml_tensor * state4 = ggml_reshape_4d(c, ssm_state, D, D, Hv, 1);
    // Steps update the recurrent state in place (each kernel block owns its state columns); prompt chunks keep the
    // packed final state and its copy, which the gfx1151 tiled recurrence (16+ rows) writes.
    ggml_tensor * gdn = step ? ggml_gated_delta_net_inplace(c, q_c, k_c, v_c, gate, beta, state4)
                             : ggml_gated_delta_net(c, q_c, k_c, v_c, gate, beta, state4);
    if (raw_gates) ggml_gated_delta_net_set_raw_gates(gdn, L.ssm_gate_ba);
    // Only speculative rollback needs per-token intermediate states; skipping keeps the packed result allocatable.
    ggml_gated_delta_net_set_skip_intermediate(gdn, true);
    // The kernel writes them straight to spec_states (same F32 transposed layout as the state, token-major); the
    // packed result stays compact because skip was set first.
    if (spec_states) gdn->src[7] = spec_states;

    // packed: [ attn S_v*H_v*T | final_state S_v*S_v*H_v (prompt chunks) ]
    ggml_tensor * attn = ggml_view_4d(c, gdn, D, Hv, T, 1,
        ggml_row_size(gdn->type, D),
        ggml_row_size(gdn->type, D * Hv),
        ggml_row_size(gdn->type, D * Hv * T), 0);
    if (!step) {
        ggml_tensor * new_state = ggml_view_4d(c, gdn, D, D, Hv, 1,
            ggml_row_size(gdn->type, D),
            ggml_row_size(gdn->type, D * D),
            ggml_row_size(gdn->type, D * D * Hv),
            ggml_row_size(gdn->type, D * Hv * T));
        ggml_build_forward_expand(gf, ggml_cpy(c, new_state, state4));
    }

    // Gated norm written as F16 in one pass, read directly by ssm_out's Q8_0 -> F16 GEMM (same arithmetic as the
    // chain below, so bit-exact).
    if (w.gfx1151 && ggml_backend_cuda_mmb_f16_input_ok(L.ssm_out, T)) {
        ggml_tensor * lin_raw = mm(c, L.ssm_out, ggml_gated_rms_norm_f16(c, attn, L.ssm_norm, z, eps));
        return ggml_reshape_2d(c, lin_raw, w.n_embd, T);
    }
    ggml_tensor * normed = ggml_mul(c, ggml_rms_norm(c, attn, eps), L.ssm_norm);
    ggml_tensor * out = ggml_mul(c, normed, ggml_sigmoid(c, ggml_reshape_4d(c, z, D, Hv, T, 1)));

    ggml_tensor * final = ggml_reshape_3d(c, out, d_in, T, 1);
    ggml_tensor * lin_raw = mm(c, L.ssm_out, final);
    return ggml_reshape_2d(c, lin_raw, w.n_embd, T);
}

// ── Full attention: dense GQA (12 layers) ───────────────────────────────

static ggml_tensor * qsa_pack_keys(ggml_context * c, ggml_tensor * keys) {
    ggml_tensor * cont = ggml_is_contiguous(keys) ? keys : ggml_cont(c, keys);
    ggml_tensor * blocks = ggml_reshape_4d(c, cont, 16, 16, 4,
        keys->ne[1] / 4 * keys->ne[2]);
    return ggml_cont(c, ggml_permute(c, blocks, 0, 2, 1, 3));
}

static ggml_tensor * qsa_pack_values(ggml_context * c, ggml_tensor * values) {
    ggml_tensor * cont = ggml_is_contiguous(values) ? values : ggml_cont(c, values);
    ggml_tensor * blocks = ggml_reshape_3d(c, cont, 256, 4,
        values->ne[1] / 4 * values->ne[2]);
    return ggml_cont(c, ggml_permute(c, blocks, 1, 0, 2, 3));
}

}  // namespace

ggml_tensor * qwen4exp_pool_blocks(ggml_context * c, ggml_tensor * keys, int64_t r) {
    const int64_t idim = keys->ne[0], nb = keys->ne[1] / r;
    ggml_tensor * k3 = ggml_reshape_3d(c, ggml_is_contiguous(keys) ? keys : ggml_cont(c, keys), idim, r, nb);
    ggml_tensor * sum = nullptr;
    for (int64_t i = 0; i < r; ++i) {
        ggml_tensor * tok = ggml_view_2d(c, k3, idim, nb, k3->nb[2], (size_t) i * k3->nb[1]);   // token r*b+i of every block
        sum = sum ? ggml_add(c, sum, tok) : ggml_cont(c, tok);
    }
    return ggml_scale(c, sum, 1.0f / (float) r);
}

int64_t qwen4exp_kv_bucket(int64_t max_ctx, int64_t kv_len, bool qsa) {
    return std::min(qsa ? std::min(max_ctx, kQwen4ExpStableQsaMaxCtx) : max_ctx, (kv_len + 255) / 256 * 256);
}

int64_t qwen4exp_stable_kv_span(int64_t & base, int64_t max_ctx, int64_t kv_len) {
    if (base == 0) base = qwen4exp_kv_bucket(max_ctx, kv_len + 256, false);   // one 256-token window past kv_len
    return kv_len <= base ? base : std::min<int64_t>(max_ctx, base + (kv_len - base + 511) / 512 * 512);
}

namespace {

static ggml_tensor * qsa_pool_norm_rope(ggml_context * c, const Qwen4ExpLayer & L,
        const Qwen4ExpWeights & w, ggml_tensor * pooled, ggml_tensor * positions) {
    const int64_t n = pooled->ne[1];
    pooled = ggml_mul(c, ggml_rms_norm(c, pooled, w.rms_eps), L.indexer_k_norm);
    int sections[4] = { w.rope_sections[0], w.rope_sections[1], w.rope_sections[2], w.rope_sections[3] };
    pooled = ggml_rope_multi(c, ggml_reshape_3d(c, pooled, w.indexer_head_size, 1, n), positions, nullptr,
        w.rope_dimension_count, sections, GGML_ROPE_TYPE_MROPE, 0, w.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    return ggml_reshape_2d(c, pooled, w.indexer_head_size, n);
}

// Pooled keys of the complete blocks [0, n_after) for QSA scoring. Blocks [n_pooled, n_after) are pooled here from
// raw keys -- tokens before pos0 from indexer_raw (written by earlier forwards), the rest from this forward's `kraw` --
// then normed, M-RoPE'd at their first token's position and stored in indexer_k (reference: Qwen4ExpTextQSAIndexer).
static ggml_tensor * qsa_pooled_keys(ggml_context * c, ggml_cgraph * gf, const Qwen4ExpLayer & L,
        const Qwen4ExpWeights & w, ggml_tensor * indexer_k, ggml_tensor * indexer_raw, ggml_tensor * kraw,
        int64_t pos0, int64_t r, int64_t n_pooled, int64_t n_after) {
    const int64_t idim = w.indexer_head_size;
    ggml_tensor * fresh = nullptr;
    if (n_after > n_pooled) {
        const int64_t t0 = r * n_pooled, t1 = r * n_after, n_new = n_after - n_pooled;
        const int64_t cached = std::min(t1, pos0) - t0;
        ggml_tensor * span = cached > 0
            ? ggml_view_2d(c, indexer_raw, idim, cached, indexer_raw->nb[1], (size_t) t0 * indexer_raw->nb[1])
            : nullptr;
        if (t1 > pos0) {
            ggml_tensor * now = ggml_view_2d(c, kraw, idim, t1 - pos0, kraw->nb[1], 0);
            span = span ? ggml_concat(c, span, now, 1) : now;
        }
        fresh = qwen4exp_pool_blocks(c, span, r);
        ggml_tensor * bp = ggml_scale(c, ggml_arange(c, (float) n_pooled, (float) n_after, 1.0f), (float) r);
        ggml_tensor * bp_i = ggml_cast(c, bp, GGML_TYPE_I32);
        ggml_tensor * bp_z = ggml_cast(c, ggml_scale(c, bp, 0.0f), GGML_TYPE_I32);
        ggml_tensor * bpos = ggml_concat(c, ggml_concat(c, bp_i, bp_i, 0), ggml_concat(c, bp_i, bp_z, 0), 0);
        fresh = qsa_pool_norm_rope(c, L, w, fresh, bpos);
        ggml_build_forward_expand(gf, ggml_cpy(c, fresh,
            ggml_view_2d(c, indexer_k, idim, n_new, indexer_k->nb[1], (size_t) n_pooled * indexer_k->nb[1])));
    }
    ggml_tensor * prefix = n_pooled > 0 ? ggml_view_2d(c, indexer_k, idim, n_pooled, indexer_k->nb[1], 0) : nullptr;
    ggml_tensor * all = prefix && fresh ? ggml_concat(c, prefix, fresh, 1) : (prefix ? prefix : fresh);
    return ggml_reshape_3d(c, all, idim, n_after, 1);
}

// The raw append is a graph dependency. Recompute the last complete block in
// the original token-add order; only completion steps write an authoritative
// row. Other steps write the permanent scratch row, outside every score view.
// A forward of T rows has ceil(T/4) such slots, one per block it can complete.
static ggml_tensor * qsa_pooled_keys_stable(ggml_context * c, const Qwen4ExpLayer & L,
        const Qwen4ExpWeights & w, ggml_tensor * indexer_k, ggml_tensor * raw_written,
        const Qwen4ExpDecodeWorkspace & ws, int64_t T) {
    ggml_tensor * written = indexer_k;
    for (int64_t s = 0; s < (T + 3) / 4; ++s) {
        const size_t o = (size_t) 10 * s * sizeof(int32_t);
        ggml_tensor * rows = ggml_view_1d(c, ws.qsa_params, 4, o + sizeof(int32_t));
        ggml_tensor * row = ggml_view_1d(c, ws.qsa_params, 1, o + 5 * sizeof(int32_t));
        ggml_tensor * pos = ggml_view_1d(c, ws.qsa_params, 4, o + 6 * sizeof(int32_t));
        ggml_tensor * span = ggml_get_rows(c, raw_written, rows);
        ggml_tensor * fresh = qsa_pool_norm_rope(c, L, w, qwen4exp_pool_blocks(c, span, 4), pos);
        written = ggml_set_rows(c, written, fresh, row);
    }
    return ggml_view_3d(c, written, w.indexer_head_size, ws.qsa_blocks, 1,
        written->nb[1], written->nb[2], 0);
}

// Row t's runtime QSA inputs in a stable graph of T rows: its visibility over the
// fixed score capacity and its valid block count. The graph serves every pos0 with
// pos0 + T in its 256-token bucket, so each row sees at least
// (bucket - 255 - T) / 4 blocks: the runtime-count top-k's lower bound.
struct Qwen4ExpQsaRow {
    ggml_tensor * visibility = nullptr;
    ggml_tensor * valid      = nullptr;
    int           min_valid  = 0;
};

static Qwen4ExpQsaRow qsa_row(ggml_context * c, const Qwen4ExpDecodeWorkspace & ws, int64_t t, int64_t T) {
    return { ggml_view_1d(c, ws.qsa_visibility, ws.qsa_blocks, (size_t) t * ws.qsa_blocks * sizeof(float)),
             ggml_view_1d(c, ws.qsa_params, 1, (size_t) 10 * t * sizeof(int32_t)),
             (int) std::max<int64_t>(513, (ws.kv_bucket - 255 - T) / 4) };
}

// Every row's QSA inputs at once: [qsa_blocks, T] visibility and the T valid counts (stride: one group).
static Qwen4ExpQsaRow qsa_rows_all(ggml_context * c, const Qwen4ExpDecodeWorkspace & ws, int64_t T) {
    return { ggml_view_2d(c, ws.qsa_visibility, ws.qsa_blocks, T, ws.qsa_blocks * sizeof(float), 0),
             ggml_view_2d(c, ws.qsa_params, 1, T, 10 * sizeof(int32_t), 0),
             (int) std::max<int64_t>(513, (ws.kv_bucket - 255 - T) / 4) };
}

static ggml_tensor * build_qsa_attn(ggml_context * c, ggml_tensor * cur,
        ggml_tensor * Q, ggml_tensor * Kf, ggml_tensor * Vf,
        const Qwen4ExpLayer & L, const Qwen4ExpWeights & w, int64_t ratio,
        ggml_tensor * positions, int64_t kv_start,
        ggml_tensor * pooled, int64_t nb, bool packed, const Qwen4ExpQsaRow * ws = nullptr) {
    const int64_t idim   = w.indexer_head_size;
    const int64_t nih    = w.indexer_n_head;
    const int64_t r      = ratio;
    const int64_t T      = cur->ne[1];
    const int64_t budget = w.indexer_top_k / r;
    const float   eps    = w.rms_eps;
    const float   qscale = 1.0f / std::sqrt((float) w.n_embd_head_k);
    int sections[4] = { w.rope_sections[0], w.rope_sections[1], w.rope_sections[2], w.rope_sections[3] };

    ggml_tensor * blocks;
    if (nb <= budget) {
        // All complete blocks fit: no scoring or pooling is needed. Reuse
        // QSA's causal IDs and F32 accumulation even in the dense regime.
        // With <r tokens, dummy block 0 is invalid; the remainder supplies
        // the visible tokens. ggml tensors cannot have a zero-sized axis.
        const int64_t count = std::max<int64_t>(1, nb);
        blocks = ggml_cast(c, ggml_repeat_4d(c, ggml_arange(c, 0.0f, (float) count, 1.0f),
            count, T, 1, 1), GGML_TYPE_I32);
    } else {
        ggml_tensor * qi = mm(c, L.indexer_q_proj, cur);              // [idim*nih, T]
        qi = ggml_reshape_3d(c, qi, idim, nih, T);
        qi = ggml_mul(c, ggml_rms_norm(c, qi, eps), L.indexer_q_norm);
        qi = ggml_rope_multi(c, qi, positions, nullptr, w.rope_dimension_count, sections,
            GGML_ROPE_TYPE_MROPE, 0, w.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        if (!ggml_is_contiguous(qi)) qi = ggml_cont(c, qi);

        ggml_tensor * comp16 = ggml_cast(c,
            ggml_reshape_2d(c, ggml_cont(c, pooled), idim, nb), GGML_TYPE_F16);
        // Every head weighs 1.0: the loaded ones constant, or the same values built in the graph.
        ggml_tensor * hw = w.qsa_ones && w.qsa_ones->ne[0] == nih && T <= w.qsa_ones->ne[1]
            ? ggml_view_2d(c, w.qsa_ones, nih, T, w.qsa_ones->nb[1], 0)
            : ggml_reshape_2d(c, ggml_scale_bias(c, ggml_scale(c, ggml_arange(c, 0.0f, (float) (nih * T), 1.0f), 0.0f),
                                                 0.0f, 1.0f), nih, T);
        ggml_tensor * summed = ws
            ? ggml_ds4_indexer_score_tokenwise(c, qi, hw, comp16, ws->visibility, (int) r)
            : ggml_ds4_indexer_score(c, qi, hw, comp16, (int) kv_start, (int) r);
        blocks = ws ? ggml_top_k_qsa(c, summed, ws->valid, ws->min_valid) : ggml_top_k(c, summed, (int) budget);
    }
    // The loader admits ratio 4 only and the cache keeps positions below 2^24,
    // the exact domain of the integer cell-id kernel.
    ggml_tensor * ids = ggml_qsa_decode_ids(c, blocks, ggml_view_1d(c, positions, T, 0), (int) r);

    ggml_tensor * q3 = ggml_cont(c, ggml_permute(c, Q, 0, 2, 1, 3));       // [D, T, Hq]
    // Only the packed prefill kernel needs contiguous K/V; the per-query kernel reads the cache through its strides,
    // so decode must not copy the visible prefix every token.
    ggml_tensor * Kc = packed && !ggml_is_contiguous(Kf) ? ggml_cont(c, Kf) : Kf;
    ggml_tensor * Vc = packed && !ggml_is_contiguous(Vf) ? ggml_cont(c, Vf) : Vf;
    ggml_tensor * attn = ggml_flash_attn_ext(c, q3, Kc, Vc, nullptr, qscale, 0.0f, 0.0f);
    attn->src[5] = ids;
    if (packed) {   // prefill kernel (T >= 128); the per-query decode kernel reads K/V rows by id
        attn->src[6] = qsa_pack_keys(c, Kc);
        attn->src[7] = qsa_pack_values(c, Vc);
    }
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    return attn;
}

// QSA block ratio of the full-attention layers (0 when the model has no indexer).
static int64_t qsa_ratio(const Qwen4ExpWeights & w) {
    for (int il = 0; il < w.n_layer && il < (int) w.compress_ratios.size(); ++il) {
        if (w.layers[il].is_full_attention && w.compress_ratios[il] > 1) return w.compress_ratios[il];
    }
    return 0;
}

enum Qwen4ExpQsaMode { QSA_DENSE = 0, QSA_PREFILL = 1, QSA_DECODE = 2 };

// Multi-row prompt calls use F32-accumulating QSA even below the selection budget.
// Call with T=1 for verify, regardless of its batch width. T=1 keeps
// the original dense decode/stable graph there, and sparse decode beyond it.
// Verify rows follow the T=1 path, including its dense/sparse boundary and accumulation.
static Qwen4ExpQsaMode qsa_mode(const Qwen4ExpWeights & w, const Qwen4ExpCache & cache, int64_t T, int64_t pos0,
                                bool enabled) {
    const int64_t r = qsa_ratio(w);
    if (!enabled || r <= 1 || cache.indexer_raw.empty() || !cache.indexer_raw[0]) return QSA_DENSE;
    if (w.indexer_head_size != 128 || w.indexer_n_head <= 0 || w.indexer_top_k % r != 0) return QSA_DENSE;
    const int64_t budget = w.indexer_top_k / r;
    if (budget * r + (r - 1) > 2560 || w.n_head != 12 * w.n_head_kv) return QSA_DENSE;
    if (T == 1 && (pos0 + T) / r <= budget) return QSA_DENSE;
    if (T < 128) return QSA_DECODE;
    const int64_t step = (r % 4 == 0) ? r : (r % 2 == 0 ? r * 2 : r * 4);
    const int64_t kv_pad = (pos0 + T + step - 1) / step * step;
    return kv_pad <= cache.attn_k[0]->ne[1] ? QSA_PREFILL : QSA_DENSE;
}

// Verify forward: row t attends exactly as a T=1 decode at pos0 + t would -- dense over that step's stable span with
// its mask, or QSA over the blocks complete at that position -- so a verified token keeps plain decode's numerics.
// A stable verify graph (qsa_ws) reads QSA rows through the T=1 graph's runtime inputs, one group per row.
struct Qwen4ExpAttnRow {
    Qwen4ExpQsaMode qsa       = QSA_DENSE;
    int64_t         span      = 0;         // dense: K/V span of the stable T=1 graph
    ggml_tensor *   mask      = nullptr;   // dense: [span, 1]
    ggml_tensor *   positions = nullptr;   // QSA: this row's M-RoPE positions
};

// A full-attention layer's prologue on the projections of its input rows: wq's interleaved [q | gate] halves of
// `qfull` (q RMS-normed per head), the RMS-normed keys and the values, with M-RoPE on q and the keys at `positions`.
// The single-sequence layer and the multi-slot rows share it.
struct Qwen4ExpAttnQkv {
    ggml_tensor * q = nullptr, * gate = nullptr, * k = nullptr, * v = nullptr;
};

static Qwen4ExpAttnQkv attn_qkv(ggml_context * c, const Qwen4ExpLayer & L, const Qwen4ExpWeights & w,
                                ggml_tensor * qfull, ggml_tensor * kraw, ggml_tensor * vraw, ggml_tensor * positions) {
    const int64_t D = w.n_embd_head_k, Hq = w.n_head, Hk = w.n_head_kv, T = qfull->ne[1];
    const size_t qe = ggml_element_size(qfull);
    Qwen4ExpAttnQkv r;
    r.q = ggml_mul(c, ggml_rms_norm(c,
        ggml_view_3d(c, qfull, D, Hq, T, 2 * D * qe, 2 * D * Hq * qe, 0), w.rms_eps), L.q_norm);
    r.gate = ggml_cont_2d(c,
        ggml_view_3d(c, qfull, D, Hq, T, 2 * D * qe, 2 * D * Hq * qe, D * qe), D * Hq, T);
    r.k = ggml_mul(c, ggml_rms_norm(c, ggml_reshape_3d(c, kraw, D, Hk, T), w.rms_eps), L.k_norm);
    r.v = ggml_reshape_3d(c, vraw, D, Hk, T);
    int sections[4] = { w.rope_sections[0], w.rope_sections[1],
                        w.rope_sections[2], w.rope_sections[3] };
    r.q = ggml_rope_multi(c, r.q, positions, nullptr, w.rope_dimension_count, sections,
                          GGML_ROPE_TYPE_MROPE, 0, w.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    r.k = ggml_rope_multi(c, r.k, positions, nullptr, w.rope_dimension_count, sections,
                          GGML_ROPE_TYPE_MROPE, 0, w.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    return r;
}

ggml_tensor * build_full_attn(ggml_context * c, ggml_cgraph * gf, ggml_tensor * cur,
                              const Qwen4ExpLayer & L, const Qwen4ExpWeights & w,
                              ggml_tensor * k_cache, ggml_tensor * v_cache,
                              ggml_tensor * indexer_k, ggml_tensor * indexer_raw,
                              ggml_tensor * positions, ggml_tensor * mask, ggml_tensor * kv_row,
                              int64_t kv_len, int64_t pos0, int64_t ratio,
                              int64_t n_pooled, Qwen4ExpQsaMode qsa,
                              const std::vector<Qwen4ExpAttnRow> * rows = nullptr,
                              const Qwen4ExpDecodeWorkspace * qsa_ws = nullptr, bool kv_only = false,
                              bool last_only = false, int draft_window = 0) {
    const int64_t D      = w.n_embd_head_k;   // 256
    const int64_t Hq     = w.n_head;          // 24
    const int64_t Hk     = w.n_head_kv;       // 2
    int64_t T            = cur->ne[1];

    // wq holds [q | gate] interleaved per head
    ggml_tensor * qfull = mm(c, L.wq, cur);   // [2*D*Hq, T]
    const size_t  qe    = ggml_element_size(qfull);
    ggml_tensor * k_proj = mm(c, L.wk, cur);
    const Qwen4ExpAttnQkv qkv = attn_qkv(c, L, w, qfull, k_proj, mm(c, L.wv, cur), positions);
    ggml_tensor * Q = qkv.q, * gate = qkv.gate, * K = qkv.k, * V = qkv.v;

    if (kv_row) {
        // Graph-stable append: the destination and graph topology stay fixed;
        // only the device input row changes between decode steps.
        ggml_tensor * Krows = ggml_cont(c, ggml_permute(c, K, 0, 2, 1, 3));
        ggml_tensor * Vrows = ggml_cont(c, ggml_permute(c, V, 0, 2, 1, 3));
        ggml_tensor * Kwrite = ggml_set_rows(c, k_cache, Krows, kv_row);
        ggml_tensor * Vwrite = ggml_set_rows(c, v_cache, Vrows, kv_row);
        ggml_build_forward_expand(gf, Kwrite);
        ggml_build_forward_expand(gf, Vwrite);
        if (qsa != QSA_DENSE) {
            k_cache = Kwrite;
            v_cache = Vwrite;
        }
    } else {
        ggml_tensor * Kt = ggml_permute(c, ggml_cast(c, K, k_cache->type), 0, 2, 1, 3);
        ggml_tensor * Vt = ggml_permute(c, ggml_cast(c, V, v_cache->type), 0, 2, 1, 3);
        ggml_build_forward_expand(gf, ggml_cpy(c, Kt,
            ggml_view_3d(c, k_cache, D, T, Hk, k_cache->nb[1], k_cache->nb[2],
                         k_cache->nb[1] * (size_t) pos0)));
        ggml_build_forward_expand(gf, ggml_cpy(c, Vt,
            ggml_view_3d(c, v_cache, D, T, Hk, v_cache->nb[1], v_cache->nb[2],
                         v_cache->nb[1] * (size_t) pos0)));
    }

    // A single MTP layer needs only these writes for prompt catch-up. Q/gate and
    // everything after attention have no cache consumers and are not in gf yet.
    if (kv_only) return nullptr;
    // A last-only mask is the final query's own row (a retained draft graph's K/V bucket).
    GGML_ASSERT(!last_only || !mask || mask->ne[1] == 1);
    if (last_only && T > 1) {
        // All retained pairs keep their original K/V projection width above.
        // Only the final query feeds the next proposal; its prefix is causal.
        GGML_ASSERT(qsa == QSA_DENSE && !rows && !indexer_raw);
        Q = ggml_view_3d(c, Q, D, Hq, 1, Q->nb[1], Q->nb[2], (T - 1) * Q->nb[2]);
        gate = ggml_view_2d(c, gate, D * Hq, 1, gate->nb[1], (T - 1) * gate->nb[1]);
        qfull = ggml_view_2d(c, qfull, qfull->ne[0], 1, qfull->nb[1], (T - 1) * qfull->nb[1]);
        T = 1;
    }
    // Session arm only: keep absolute RoPE/cache writes and restrict just the
    // draft query's visible K/V. Target calls always leave draft_window zero.
    const int64_t kv_start = draft_window > 0 ? std::max<int64_t>(0, kv_len - draft_window) : 0;
    GGML_ASSERT(!draft_window || (last_only && T == 1 && !mask && qsa == QSA_DENSE));

    ggml_tensor * K_full = ggml_view_3d(c, k_cache, D, kv_len - kv_start, Hk,
        k_cache->nb[1], k_cache->nb[2], kv_start * k_cache->nb[1]);
    ggml_tensor * V_full = ggml_view_3d(c, v_cache, D, kv_len - kv_start, Hk,
        v_cache->nb[1], v_cache->nb[2], kv_start * v_cache->nb[1]);

    // Every token's raw indexer key goes to the cache, so blocks can be pooled whenever QSA first needs them.
    ggml_tensor * kraw = nullptr;
    if (indexer_raw) {
        kraw = mm(c, L.indexer_k_proj, cur);   // [idim, T]
        ggml_tensor * raw_write = kv_row
            ? ggml_set_rows(c, indexer_raw, kraw, kv_row)
            : ggml_cpy(c, kraw, ggml_view_2d(c, indexer_raw, kraw->ne[0], T, indexer_raw->nb[1],
                                             (size_t) pos0 * indexer_raw->nb[1]));
        ggml_build_forward_expand(gf, raw_write);
        if (qsa_ws) indexer_raw = raw_write;
    }
    ggml_tensor * attn = nullptr;
    if (rows) {
        GGML_ASSERT((int64_t) rows->size() == T);
        int64_t n_after = 0;   // complete blocks visible to the last QSA row
        for (int64_t t = 0; t < T; ++t) if ((*rows)[t].qsa != QSA_DENSE) n_after = (pos0 + t + 1) / ratio;
        GGML_ASSERT(n_after == 0 || (kraw && indexer_k && ratio > 1));
        ggml_tensor * pooled = n_after == 0 ? nullptr : qsa_ws
            ? qsa_pooled_keys_stable(c, L, w, indexer_k, indexer_raw, *qsa_ws, T)
            : qsa_pooled_keys(c, gf, L, w, indexer_k, indexer_raw, kraw, pos0, ratio, n_pooled, n_after);
        const bool at_once = qsa_ws && (*rows)[0].qsa != QSA_DENSE;
        if (at_once) {
            // Every row is QSA (rows only move away from the dense prefix): score, select and attend for all
            // rows at once. Each row keeps its T=1 arithmetic -- the batch-invariant indexer projection,
            // one-token scoring, the one-row top-k and the per-query attention kernel.
            const int64_t nb = qsa_ws->qsa_blocks;
            const Qwen4ExpQsaRow in = qsa_rows_all(c, *qsa_ws, T);
            ggml_tensor * Kr = ggml_view_3d(c, k_cache, D, qsa_ws->kv_bucket, Hk, k_cache->nb[1], k_cache->nb[2], 0);
            ggml_tensor * Vr = ggml_view_3d(c, v_cache, D, qsa_ws->kv_bucket, Hk, v_cache->nb[1], v_cache->nb[2], 0);
            attn = build_qsa_attn(c, cur, Q, Kr, Vr, L, w, ratio, positions, pos0,
                ggml_view_3d(c, pooled, pooled->ne[0], nb, 1, pooled->nb[1], pooled->nb[2], 0), nb, false, &in);
        }
        for (int64_t t = 0; !at_once && t < T; ++t) {
            const Qwen4ExpAttnRow & row = (*rows)[t];
            ggml_tensor * q = ggml_view_3d(c, Q, D, Hq, 1, Q->nb[1], Q->nb[2], (size_t) t * Q->nb[2]);
            const int64_t span = row.qsa == QSA_DENSE ? row.span : qsa_ws ? qsa_ws->kv_bucket : pos0 + t + 1;
            ggml_tensor * Kr = ggml_view_3d(c, k_cache, D, span, Hk, k_cache->nb[1], k_cache->nb[2], 0);
            ggml_tensor * Vr = ggml_view_3d(c, v_cache, D, span, Hk, v_cache->nb[1], v_cache->nb[2], 0);
            ggml_tensor * a;
            if (row.qsa == QSA_DENSE) {
                a = ggml_flash_attn_ext(c, ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3)), Kr, Vr, row.mask,
                    1.0f / std::sqrt((float) D), 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);
            } else {
                const int64_t nb = qsa_ws ? qsa_ws->qsa_blocks : span / ratio;
                const Qwen4ExpQsaRow in = qsa_ws ? qsa_row(c, *qsa_ws, t, T) : Qwen4ExpQsaRow{};
                a = build_qsa_attn(c, ggml_view_2d(c, cur, cur->ne[0], 1, cur->nb[1], (size_t) t * cur->nb[1]), q, Kr, Vr,
                    L, w, ratio, row.positions, pos0 + t,
                    ggml_view_3d(c, pooled, pooled->ne[0], nb, 1, pooled->nb[1], pooled->nb[2], 0), nb, false,
                    qsa_ws ? &in : nullptr);
            }
            attn = attn ? ggml_concat(c, attn, a, 2) : a;
        }
    } else if (T > 1 && mask == nullptr && qsa == QSA_DENSE) {
        std::fprintf(stderr,
            "[qwen4exp] dense attention without a causal mask (T=%lld pos0=%lld)\n",
            (long long) T, (long long) pos0);
        std::abort();
    } else if (qsa != QSA_DENSE) {
        GGML_ASSERT(kraw && indexer_k && ratio > 1);
        const int64_t n_after = (pos0 + T) / ratio;   // complete blocks visible to the last query
        ggml_tensor * pooled = qsa_ws
            ? qsa_pooled_keys_stable(c, L, w, indexer_k, indexer_raw, *qsa_ws, T)
            : n_after > w.indexer_top_k / ratio
                ? qsa_pooled_keys(c, gf, L, w, indexer_k, indexer_raw, kraw, pos0, ratio, n_pooled, n_after)
                : nullptr;
        if (qsa == QSA_PREFILL) {
            // Pad K/V to a multiple of lcm(4, ratio) for the packed prefill kernel.
            const int64_t qsa_step = (ratio % 4 == 0) ? ratio : (ratio % 2 == 0 ? ratio * 2 : ratio * 4);
            const int64_t kv_pad   = (kv_len + qsa_step - 1) / qsa_step * qsa_step;
            if (kv_pad != kv_len) {
                // WMMA loads whole four-key blocks: masked probabilities do
                // not protect against an unwritten NaN value (0 * NaN).
                const int64_t pad = kv_pad - kv_len;
                ggml_tensor * zero = ggml_reshape_3d(c, ggml_scale(c,
                    ggml_arange(c, 0.0f, (float) (D * pad * Hk), 1.0f), 0.0f), D, pad, Hk);
                for (auto * cache : { k_cache, v_cache }) {
                    ggml_build_forward_expand(gf, ggml_cpy(c, zero,
                        ggml_view_3d(c, cache, D, pad, Hk, cache->nb[1], cache->nb[2],
                                     (size_t) kv_len * cache->nb[1])));
                }
            }
            ggml_tensor * K_pad = (kv_pad != kv_len)
                ? ggml_view_3d(c, k_cache, D, kv_pad, Hk, k_cache->nb[1], k_cache->nb[2], 0) : K_full;
            ggml_tensor * V_pad = (kv_pad != kv_len)
                ? ggml_view_3d(c, v_cache, D, kv_pad, Hk, v_cache->nb[1], v_cache->nb[2], 0) : V_full;
            attn = build_qsa_attn(c, cur, Q, K_pad, V_pad, L, w, ratio, positions,
                                  pos0, pooled, n_after, true);
        } else {
            const Qwen4ExpQsaRow in = qsa_ws ? qsa_row(c, *qsa_ws, 0, T) : Qwen4ExpQsaRow{};
            attn = build_qsa_attn(c, cur, Q, K_full, V_full, L, w, ratio, positions,
                                  pos0, pooled, qsa_ws ? qsa_ws->qsa_blocks : n_after, false, qsa_ws ? &in : nullptr);
        }
    } else {
        // The padded cache tail is zero-initialized and excluded by the causal mask,
        // including during single-token decode.
        if (mask && mask->ne[0] != kv_len) {
            GGML_ASSERT(mask->ne[0] <= k_cache->ne[1] && mask->ne[0] <= v_cache->ne[1]);
            K_full = ggml_view_3d(c, k_cache, D, mask->ne[0], Hk, k_cache->nb[1], k_cache->nb[2], 0);
            V_full = ggml_view_3d(c, v_cache, D, mask->ne[0], Hk, v_cache->nb[1], v_cache->nb[2], 0);
        }
        ggml_tensor * Qfa = ggml_cont(c, ggml_permute(c, Q, 0, 2, 1, 3));  // [D, T, Hq]
        attn = ggml_flash_attn_ext(c, Qfa, K_full, V_full, mask,
            1.0f / std::sqrt((float) D), 0.0f, 0.0f);                       // [D, Hq, T, 1]
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);                  // match upstream's F32 acc
    }

    // sigmoid(gate) * attn written as F16 in one pass straight from the wq view (no CONT), read directly by wo's
    // Q8_0 -> F16 GEMM. Same product as below, so bit-exact.
    if (w.gfx1151 && ggml_backend_cuda_mmb_f16_input_ok(L.wo, T)) {
        ggml_tensor * gate3 = ggml_view_3d(c, qfull, D, Hq, T, 2 * D * qe, 2 * D * Hq * qe, D * qe);
        return mm(c, L.wo, ggml_gated_f16(c, attn, gate3));
    }
    ggml_tensor * attn2 = ggml_reshape_2d(c, attn, D * Hq, T);
    attn2 = ggml_mul(c, attn2, ggml_sigmoid(c, gate));
    return mm(c, L.wo, attn2);
}

// ── Per-layer n-gram embedding (PLE) ────────────────────────────────────

// spec_state (verify forward): receives the conv history after each token.
ggml_tensor * build_ple(ggml_context * c, ggml_cgraph * gf, ggml_tensor * hidden,
                        ggml_tensor * ple_emb, const Qwen4ExpLayer & L,
                        const Qwen4ExpWeights & w, ggml_tensor * ple_conv_state,
                        ggml_tensor * spec_state = nullptr) {
    const int64_t n_embd = w.n_embd;
    const int64_t hc     = w.n_hc;
    const int64_t hc_dim = hc * n_embd;
    const int64_t T      = hidden->ne[2];
    const float   eps    = w.rms_eps;

    ggml_tensor * key   = mm(c, L.ple_key,   ple_emb);   // [hc_dim, T]
    ggml_tensor * value = mm(c, L.ple_value, ple_emb);   // [n_embd, T]

    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * nw) {
        ggml_tensor * t = ggml_reshape_3d(c, x, n_embd, hc, T);
        t = ggml_rms_norm(c, t, eps);
        t = ggml_reshape_2d(c, t, hc_dim, T);
        t = ggml_mul(c, t, nw);
        return ggml_reshape_3d(c, t, n_embd, hc, T);
    };

    key = grouped_norm(key, L.ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, L.ple_norm_query);

    ggml_tensor * s = ggml_sum_rows(c, ggml_mul(c, key, query));   // [1, hc, T]
    s = ggml_scale(c, s, 1.0f / std::sqrt((float) n_embd));
    ggml_tensor * mag = ggml_sqrt(c, ggml_clamp(c, ggml_abs(c, s), 1e-6f, INFINITY));
    ggml_tensor * ple_gate = ggml_sigmoid(c, ggml_mul(c, ggml_sgn(c, s), mag));

    ggml_tensor * v3 = repeat_dim1(c,
        ggml_reshape_3d(c, value, n_embd, 1, T), hc);
    ggml_tensor * gated = ggml_mul(c, v3, ple_gate);

    ggml_tensor * normalized = ggml_reshape_2d(c,
        grouped_norm(ggml_reshape_2d(c, gated, hc_dim, T), L.ple_norm_conv), hc_dim, T);

    const int64_t kern = w.ple_conv_kernel;
    const int64_t dil  = w.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    ggml_tensor * norm_t = ggml_transpose(c, ggml_reshape_2d(c, normalized, hc_dim, T));
    ggml_tensor * ple_state = ggml_cont(c, ggml_reshape_3d(c, ple_conv_state, hist, hc_dim, 1));
    ggml_tensor * padded = ggml_concat(c, ple_state, norm_t, 0);

    ggml_build_forward_expand(gf, ggml_cpy(c,
        ggml_cont(c, ggml_view_3d(c, padded, hist, hc_dim, 1, padded->nb[1], padded->nb[2],
                                  (size_t) T * padded->nb[0])),
        ggml_reshape_3d(c, ple_conv_state, hist, hc_dim, 1)));
    for (int64_t t = 0; spec_state && t < T; ++t) {
        ggml_build_forward_expand(gf, ggml_cpy(c,
            ggml_cont(c, ggml_view_3d(c, padded, hist, hc_dim, 1, padded->nb[1], padded->nb[2], (t + 1) * padded->nb[0])),
            ggml_view_3d(c, spec_state, hist, hc_dim, 1, spec_state->nb[1], spec_state->nb[2],
                t * spec_state->nb[2])));
    }

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        const int64_t start = hist - (kern - 1 - k) * dil;
        ggml_tensor * shifted = ggml_cont(c, ggml_transpose(c,
            ggml_view_3d(c, padded, T, hc_dim, 1, padded->nb[1], padded->nb[2],
                         ggml_row_size(padded->type, start))));
        // The fused kernel runs at the first CONT and reads norm_t directly.
        // CONT ignores src[1], so use it as the allocator dependency edge.
        if (k == 0) shifted->src[1] = norm_t;
        ggml_tensor * wk = ggml_cont(c,
            ggml_view_2d(c, L.ple_conv1d, 1, hc_dim, L.ple_conv1d->nb[1],
                         k * L.ple_conv1d->nb[0]));
        wk = ggml_reshape_1d(c, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) wk = ggml_cast(c, wk, GGML_TYPE_F32);
        ggml_tensor * term = ggml_mul(c, shifted, wk);
        conv_out = conv_out ? ggml_add(c, conv_out, term) : term;
    }
    conv_out = ggml_reshape_3d(c, ggml_cont(c, ggml_silu(c, conv_out)), n_embd, hc, T);

    ggml_tensor * ple_out = ggml_add(c, hidden, ggml_add(c, gated, conv_out));
    // Expand now so the conv matcher sees a contiguous concat -> state -> taps -> silu subgraph.
    ggml_build_forward_expand(gf, ple_out);
    return ple_out;
}

static ggml_tensor * column(ggml_context * c, ggml_tensor * x, int s) {
    GGML_ASSERT(s >= 0 && s < x->ne[1]);
    return ggml_view_2d(c, x, x->ne[0], 1, x->nb[1], (size_t) s * x->nb[1]);
}

static ggml_tensor * build_linear_attn_projected(ggml_context * c, ggml_cgraph * gf,
        ggml_tensor * qkv, ggml_tensor * z, ggml_tensor * beta, ggml_tensor * alpha,
        const Qwen4ExpLayer & L, const Qwen4ExpWeights & w,
        ggml_tensor * ssm_state, ggml_tensor * conv_state) {
    const int64_t D = w.ssm_d_state, Hk = w.ssm_n_group;
    const int64_t Hv = w.linear_value_heads, d_in = w.ssm_d_inner;
    const int64_t kernel = w.ssm_d_conv;
    const int64_t conv_channels = 2 * Hk * D + d_in;
    const float eps = w.rms_eps;

    beta = ggml_sigmoid(c, ggml_reshape_4d(c, beta, 1, Hv, 1, 1));
    alpha = ggml_reshape_3d(c, alpha, Hv, 1, 1);
    alpha = ggml_softplus(c, ggml_add(c, alpha, L.ssm_dt_bias));
    ggml_tensor * gate = ggml_reshape_4d(c,
        ggml_mul(c, alpha, L.ssm_a), 1, Hv, 1, 1);

    ggml_tensor * hist = ggml_reshape_3d(c, conv_state, kernel - 1, conv_channels, 1);
    ggml_tensor * qkv_t = ggml_transpose(c, qkv);
    ggml_tensor * conv_input = ggml_concat(c, hist, qkv_t, 0);
    ggml_tensor * new_hist = ggml_cont(c, ggml_view_3d(c, conv_input,
        kernel - 1, conv_channels, 1, conv_input->nb[1], conv_input->nb[2],
        conv_input->nb[0]));
    ggml_build_forward_expand(gf, ggml_cpy(c, new_hist,
        ggml_reshape_3d(c, conv_state, kernel - 1, conv_channels, 1)));

    ggml_tensor * conv_op = ggml_ssm_conv(c, conv_input, L.ssm_conv1d);
    // Preserve the fusion allocator dependency used by the single-sequence path.
    conv_op->src[3] = qkv_t;
    ggml_tensor * conv = ggml_silu(c, conv_op);
    const size_t esz = ggml_element_size(conv);
    const size_t tstride = (size_t) conv_channels * esz;
    ggml_tensor * q_raw = ggml_view_3d(c, conv, D, Hk, 1, D * esz, tstride, 0);
    ggml_tensor * k_raw = ggml_view_3d(c, conv, D, Hk, 1, D * esz, tstride, D * Hk * esz);
    ggml_tensor * q_c = ggml_scale(c, ggml_rms_norm(c, q_raw, eps / (float) D),
                                   1.0f / sqrtf((float) D));
    ggml_tensor * k_c = ggml_scale(c, ggml_rms_norm(c, k_raw, eps / (float) D),
                                   1.0f / sqrtf((float) D));
    ggml_tensor * v_c = ggml_view_3d(c, conv, D, Hv, 1, D * esz, tstride,
                                     2 * D * Hk * esz);
    ggml_tensor * state4 = ggml_reshape_4d(c, ssm_state, D, D, Hv, 1);
    ggml_tensor * gdn = ggml_gated_delta_net(c, q_c, k_c, v_c, gate, beta, state4);
    ggml_gated_delta_net_set_skip_intermediate(gdn, true);
    ggml_tensor * attn = ggml_view_4d(c, gdn, D, Hv, 1, 1,
        ggml_row_size(gdn->type, D), ggml_row_size(gdn->type, D * Hv),
        ggml_row_size(gdn->type, D * Hv), 0);
    ggml_tensor * new_state = ggml_view_4d(c, gdn, D, D, Hv, 1,
        ggml_row_size(gdn->type, D), ggml_row_size(gdn->type, D * D),
        ggml_row_size(gdn->type, D * D * Hv), ggml_row_size(gdn->type, D * Hv));
    ggml_build_forward_expand(gf, ggml_cpy(c, new_state, state4));
    ggml_tensor * normed = ggml_mul(c, ggml_rms_norm(c, attn, eps), L.ssm_norm);
    ggml_tensor * out = ggml_mul(c, normed,
        ggml_sigmoid(c, ggml_reshape_4d(c, z, D, Hv, 1, 1)));
    return ggml_reshape_2d(c, out, d_in, 1);
}

// Every slot's one-token recurrence at once (slots with stacked recurrent
// state, see Qwen4ExpSlotStates): the conv history is gathered and written
// back by slot id and the recurrence updates each slot's slab in place. The
// gates, norms and output gate run over all rows; each row's arithmetic is
// build_linear_attn_projected's.
static ggml_tensor * build_linear_attn_slots(ggml_context * c, ggml_cgraph * gf,
        ggml_tensor * qkv, ggml_tensor * z, ggml_tensor * beta, ggml_tensor * alpha,
        const Qwen4ExpLayer & L, const Qwen4ExpWeights & w,
        ggml_tensor * ssm_states, ggml_tensor * conv_states, ggml_tensor * slot_ids) {
    const int64_t D = w.ssm_d_state, Hk = w.ssm_n_group;
    const int64_t Hv = w.linear_value_heads, d_in = w.ssm_d_inner;
    const int64_t kernel = w.ssm_d_conv;
    const int64_t conv_channels = 2 * Hk * D + d_in;
    const int64_t S = qkv->ne[1];
    const float eps = w.rms_eps;

    beta = ggml_sigmoid(c, ggml_reshape_4d(c, beta, 1, Hv, 1, S));
    alpha = ggml_softplus(c, ggml_add(c, ggml_reshape_3d(c, alpha, Hv, 1, S), L.ssm_dt_bias));
    ggml_tensor * gate = ggml_reshape_4d(c, ggml_mul(c, alpha, L.ssm_a), 1, Hv, 1, S);

    const int64_t slab = (kernel - 1) * conv_channels;
    ggml_tensor * conv_all = ggml_reshape_2d(c, conv_states, slab, conv_states->ne[2]);
    ggml_tensor * hist = ggml_reshape_3d(c, ggml_get_rows(c, conv_all, slot_ids), kernel - 1, conv_channels, S);
    ggml_tensor * qkv_t = ggml_transpose(c, ggml_reshape_3d(c, qkv, conv_channels, 1, S));
    ggml_tensor * conv_input = ggml_concat(c, hist, qkv_t, 0);
    ggml_tensor * new_hist = ggml_cont(c, ggml_view_3d(c, conv_input, kernel - 1, conv_channels, S,
        conv_input->nb[1], conv_input->nb[2], conv_input->nb[0]));
    ggml_build_forward_expand(gf, ggml_set_rows(c, conv_all, ggml_reshape_2d(c, new_hist, slab, S), slot_ids));

    ggml_tensor * conv = ggml_silu(c, ggml_ssm_conv(c, conv_input, L.ssm_conv1d));   // [C, 1, S]
    const size_t esz = ggml_element_size(conv);
    ggml_tensor * q_raw = ggml_view_4d(c, conv, D, Hk, 1, S, D * esz, conv->nb[1], conv->nb[2], 0);
    ggml_tensor * k_raw = ggml_view_4d(c, conv, D, Hk, 1, S, D * esz, conv->nb[1], conv->nb[2], D * Hk * esz);
    ggml_tensor * v_c = ggml_view_4d(c, conv, D, Hv, 1, S, D * esz, conv->nb[1], conv->nb[2], 2 * D * Hk * esz);
    ggml_tensor * q_c = ggml_scale(c, ggml_rms_norm(c, q_raw, eps / (float) D), 1.0f / sqrtf((float) D));
    ggml_tensor * k_c = ggml_scale(c, ggml_rms_norm(c, k_raw, eps / (float) D), 1.0f / sqrtf((float) D));
    ggml_tensor * gdn = ggml_gated_delta_net_active_inplace(c, q_c, k_c, v_c, gate, beta, ssm_states, slot_ids);
    ggml_gated_delta_net_set_skip_intermediate(gdn, true);
    ggml_build_forward_expand(gf, gdn);
    ggml_tensor * attn = ggml_view_4d(c, gdn, D, Hv, 1, S,
        ggml_row_size(gdn->type, D), ggml_row_size(gdn->type, D * Hv),
        ggml_row_size(gdn->type, D * Hv), 0);
    ggml_tensor * normed = ggml_mul(c, ggml_rms_norm(c, attn, eps), L.ssm_norm);
    ggml_tensor * out = ggml_mul(c, normed, ggml_sigmoid(c, ggml_reshape_4d(c, z, D, Hv, 1, S)));
    return ggml_reshape_2d(c, out, d_in, S);
}

// Every slot's one-token attention against its own cache. The norms, M-RoPE,
// indexer projection and output gate run once over all rows (row-wise, so
// each row's arithmetic is the single-row graph's); the K/V writes and the
// attention itself stay per slot. `positions` holds the rows' M-RoPE sections,
// `kv_rows` each slot's K/V row, so the graph is stable across steps. A slot
// with `qsa[s]` attends like a stable T=1 QSA decode (its runtime inputs, a
// K/V bucket and its own [4] positions); the
// others attend densely over `spans[s]` behind `masks[s]`.
static ggml_tensor * build_full_attn_slots(ggml_context * c, ggml_cgraph * gf,
        ggml_tensor * qfull, ggml_tensor * kraw, ggml_tensor * vraw, ggml_tensor * cur,
        ggml_tensor * positions, ggml_tensor * const * masks, ggml_tensor * const * kv_rows,
        const int64_t * spans, const Qwen4ExpLayer & L, const Qwen4ExpWeights & w,
        Qwen4ExpCache * const * caches, int fi, bool indexer, const Qwen4ExpDecodeWorkspace * const * qsa) {
    const int64_t D = w.n_embd_head_k, Hq = w.n_head, Hk = w.n_head_kv;
    const int64_t S = qfull->ne[1];
    // Like dense solo, retain raw keys and leave pooling/indexer_blocks lazy until QSA runs.
    ggml_tensor * keys = indexer && caches[0]->indexer_raw[(size_t) fi]
        ? mm(c, L.indexer_k_proj, cur) : nullptr;
    const Qwen4ExpAttnQkv qkv = attn_qkv(c, L, w, qfull, kraw, vraw, positions);
    ggml_tensor * q = qkv.q, * gate = qkv.gate, * k = qkv.k, * v = qkv.v;
    ggml_tensor * qfa = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));   // [D, S, Hq]
    ggml_tensor * rows = nullptr;
    for (int s = 0; s < S; ++s) {
        const Qwen4ExpCache & cache = *caches[s];
        ggml_tensor * k_cache = cache.attn_k[(size_t) fi];
        ggml_tensor * v_cache = cache.attn_v[(size_t) fi];
        ggml_tensor * raw = keys ? ggml_set_rows(c, cache.indexer_raw[(size_t) fi], column(c, keys, s), kv_rows[s]) : nullptr;
        if (raw) ggml_build_forward_expand(gf, raw);
        // One row's [D, Hk] is already the [D, 1, Hk] layout the cache rows take.
        ggml_tensor * krow = ggml_reshape_3d(c, ggml_view_2d(c, k, D, Hk, k->nb[1], s * k->nb[2]), D, 1, Hk);
        ggml_tensor * vrow = ggml_reshape_3d(c, ggml_view_2d(c, v, D, Hk, v->nb[1], s * v->nb[2]), D, 1, Hk);
        ggml_tensor * kw = ggml_set_rows(c, k_cache, krow, kv_rows[s]);
        ggml_tensor * vw = ggml_set_rows(c, v_cache, vrow, kv_rows[s]);
        ggml_build_forward_expand(gf, kw);
        ggml_build_forward_expand(gf, vw);
        ggml_tensor * attn;
        if (qsa && qsa[s]) {
            // As the stable T=1 QSA graph: the writes are the attention's and pooling's sources.
            const Qwen4ExpDecodeWorkspace & ws = *qsa[s];
            GGML_ASSERT(raw && cache.indexer_k[(size_t) fi]);
            const Qwen4ExpQsaRow in = qsa_row(c, ws, 0, 1);
            ggml_tensor * pooled = qsa_pooled_keys_stable(c, L, w, cache.indexer_k[(size_t) fi], raw, ws, 1);
            attn = build_qsa_attn(c, column(c, cur, s),
                ggml_view_3d(c, q, D, Hq, 1, q->nb[1], q->nb[2], s * q->nb[2]),
                ggml_view_3d(c, kw, D, ws.kv_bucket, Hk, kw->nb[1], kw->nb[2], 0),
                ggml_view_3d(c, vw, D, ws.kv_bucket, Hk, vw->nb[1], vw->nb[2], 0),
                L, w, qsa_ratio(w), ws.positions, 0, pooled, ws.qsa_blocks, false, &in);
        } else {
            const int64_t span = spans[s];
            ggml_tensor * kfull = ggml_view_3d(c, k_cache, D, span, Hk, k_cache->nb[1], k_cache->nb[2], 0);
            ggml_tensor * vfull = ggml_view_3d(c, v_cache, D, span, Hk, v_cache->nb[1], v_cache->nb[2], 0);
            ggml_tensor * qs = ggml_view_3d(c, qfa, D, 1, Hq, qfa->nb[1], qfa->nb[2], s * qfa->nb[1]);
            attn = ggml_flash_attn_ext(c, qs, kfull, vfull, masks[s], 1.0f / sqrtf((float) D), 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        }
        attn = ggml_reshape_2d(c, attn, D * Hq, 1);
        rows = rows ? ggml_concat(c, rows, attn, 1) : attn;
    }
    return ggml_mul(c, rows, ggml_sigmoid(c, gate));
}

static ggml_tensor * build_ple_row(ggml_context * c, ggml_cgraph * gf,
        ggml_tensor * hidden, ggml_tensor * key, ggml_tensor * value,
        const Qwen4ExpLayer & L, const Qwen4ExpWeights & w, ggml_tensor * state) {
    const int64_t n_embd = w.n_embd, hc = w.n_hc, hc_dim = n_embd * hc;
    const float eps = w.rms_eps;
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * nw) {
        ggml_tensor * t = ggml_rms_norm(c, ggml_reshape_3d(c, x, n_embd, hc, 1), eps);
        return ggml_reshape_3d(c, ggml_mul(c, ggml_reshape_2d(c, t, hc_dim, 1), nw),
                               n_embd, hc, 1);
    };
    key = grouped_norm(key, L.ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, L.ple_norm_query);
    ggml_tensor * score = ggml_scale(c, ggml_sum_rows(c, ggml_mul(c, key, query)),
                                     1.0f / sqrtf((float) n_embd));
    ggml_tensor * mag = ggml_sqrt(c, ggml_clamp(c, ggml_abs(c, score), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(c, ggml_mul(c, ggml_sgn(c, score), mag));
    ggml_tensor * gated = ggml_mul(c, ggml_repeat_4d(c,
        ggml_reshape_3d(c, value, n_embd, 1, 1), n_embd, hc, 1, 1), gate);
    ggml_tensor * normalized = grouped_norm(
        ggml_reshape_2d(c, gated, hc_dim, 1), L.ple_norm_conv);
    const int64_t kern = w.ple_conv_kernel, dil = w.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;
    ggml_tensor * norm_t = ggml_transpose(c, ggml_reshape_2d(c, normalized, hc_dim, 1));
    ggml_tensor * padded = ggml_concat(c,
        ggml_cont(c, ggml_reshape_2d(c, state, hist, hc_dim)), norm_t, 0);
    ggml_build_forward_expand(gf, ggml_cpy(c,
        ggml_cont(c, ggml_view_2d(c, padded, hist, hc_dim, padded->nb[1], padded->nb[0])),
        ggml_reshape_2d(c, state, hist, hc_dim)));
    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        const int64_t start = hist - (kern - 1 - k) * dil;
        // `padded` is [time, features] with time contiguous. Mirror the solo
        // path's [T, features] view then transpose so each feature reads the
        // selected time tap; a direct [features, 1] view would walk adjacent
        // time values instead of the strided feature row.
        ggml_tensor * shifted = ggml_cont(c, ggml_transpose(c,
            ggml_view_3d(c, padded, 1, hc_dim, 1, padded->nb[1], padded->nb[2],
                         ggml_row_size(padded->type, start))));
        if (k == 0) shifted->src[1] = norm_t;
        ggml_tensor * wk = ggml_reshape_1d(c, ggml_cont(c,
            ggml_view_2d(c, L.ple_conv1d, 1, hc_dim, L.ple_conv1d->nb[1],
                         k * L.ple_conv1d->nb[0])), hc_dim);
        if (wk->type != GGML_TYPE_F32) wk = ggml_cast(c, wk, GGML_TYPE_F32);
        ggml_tensor * term = ggml_mul(c, shifted, wk);
        conv_out = conv_out ? ggml_add(c, conv_out, term) : term;
    }
    conv_out = ggml_reshape_3d(c, ggml_cont(c, ggml_silu(c, conv_out)), n_embd, hc, 1);
    return ggml_add(c, hidden, ggml_add(c, gated, conv_out));
}

}  // namespace

Qwen4ExpInputs qwen4exp_prepare_inputs(const Qwen4ExpWeights & w,
        const int32_t * tokens, int n_tokens, const std::vector<int32_t> & ple_prev) {
    Qwen4ExpInputs res;
    if (!tokens || n_tokens <= 0) return res;
    auto & emb = res.emb;
    emb.resize((size_t) w.n_embd * n_tokens);
    if (!w.embedder.embed(tokens, n_tokens, emb.data())) {
        std::fprintf(stderr, "[qwen4exp] cpu embedding failed\n");
        return res;
    }

    const bool has_ple = !w.ple_layer_ids.empty() && w.ple_reader.available();
    const int64_t ple_heads = w.ple_n_heads;
    std::vector<int32_t> ple_rows(has_ple ? (size_t) ple_heads * n_tokens : 0);
    auto & ple_data = res.ple;
    ple_data.resize(has_ple ? (size_t) w.ple_head_dim * ple_heads * n_tokens : 0);
    if (has_ple) {
        const int64_t ng = w.ple_ngram_size;
        std::vector<int32_t> seq = ple_prev;
        seq.insert(seq.end(), tokens, tokens + n_tokens);
        const int64_t base = (int64_t) ple_prev.size();
        for (int64_t i = 0; i < n_tokens; ++i) {
            const int64_t pos = base + i;
            std::vector<uint64_t> ctx(ng);
            ctx[0] = (uint64_t) tokens[i];
            bool cut = false;
            for (int64_t s = 1; s < ng; ++s) {
                if (cut || pos - s < 0) { ctx[s] = (uint64_t) w.ple_eos_token_id; cut = true; }
                else {
                    const int32_t t = seq[(size_t) (pos - s)];
                    if (t < 0 || t == w.ple_eos_token_id) cut = true;
                    ctx[s] = cut ? (uint64_t) w.ple_eos_token_id : (uint64_t) t;
                }
            }
            for (int64_t n = 2; n <= ng; ++n) {
                uint64_t mixed = ctx[0] * w.ple_layer_multipliers[0];
                for (int64_t j = 1; j < n; ++j) {
                    mixed ^= ctx[j] * w.ple_layer_multipliers[(size_t) j];
                }
                const int64_t head_base = (n - 2) * w.ple_heads_per_ngram;
                for (int64_t q = 0; q < w.ple_heads_per_ngram; ++q) {
                    const int64_t h = head_base + q;
                    const int64_t row = (int64_t) (mixed % (uint64_t) w.ple_head_vocab_sizes[h]) +
                                        w.ple_head_offsets[h];
                    ple_rows[(size_t) (i * ple_heads + h)] = (int32_t) row;
                }
            }
        }
        if (!w.ple_reader.gather(ple_rows.data(), (int64_t) ple_rows.size(), ple_data.data())) {
            std::fprintf(stderr, "[qwen4exp] PLE gather failed\n");
            return res;
        }
        const size_t keep = (size_t) std::min<int64_t>(ng - 1, n_tokens + (int64_t) ple_prev.size());
        std::vector<int32_t> next;
        next.reserve(keep);
        const size_t total = ple_prev.size() + (size_t) n_tokens;
        for (size_t k = total - keep; k < total; ++k) {
            next.push_back(k < ple_prev.size() ? ple_prev[k] : tokens[k - ple_prev.size()]);
        }
        res.ple_prev = std::move(next);
    }

    res.ok = true;
    return res;
}

// Split mode placement: the routed experts on the expert device, every other
// node and the graph inputs on the target. Left unpinned, the scheduler places
// a node after its graph-order neighbour, often an expert split, and keeps graph
// inputs on the CPU, copying them to each split that reads them; a strided view
// of an input (MTP prefill's per-block positions) cannot be copied as one span.
static void qwen4exp_pin_split(ggml_backend_sched_t sched, ggml_cgraph * gf, ggml_backend_t target,
                               ggml_backend_t expert, const std::vector<ggml_tensor *> & expert_nodes) {
    for (ggml_tensor * t : expert_nodes) ggml_backend_sched_set_tensor_backend(sched, t, expert);
    const std::unordered_set<const ggml_tensor *> on_expert(expert_nodes.begin(), expert_nodes.end());
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        for (ggml_tensor * src : node->src) {
            for (ggml_tensor * t = src; t; t = t->view_src) {
                if ((t->flags & GGML_TENSOR_FLAG_INPUT) && !t->buffer) {
                    ggml_backend_sched_set_tensor_backend(sched, t, target);
                }
            }
        }
        const bool view = node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
                          node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE;
        if (!view && !on_expert.count(node) && ggml_backend_supports_op(target, node)) {
            ggml_backend_sched_set_tensor_backend(sched, node, target);
        }
    }
}

// Pipelined split prefill: a prompt chunk as consecutive sub-chunks (streams),
// each built exactly like a chunk of its own, whose layers interleave in the
// graph so the expert device runs one stream's routed experts while the target
// runs the other's attention.
struct Qwen4ExpStream {
    int64_t T = 0, pos0 = 0, layer_T = 0;
    Qwen4ExpQsaMode qsa = QSA_DENSE;
    int n_pooled = 0;
    ggml_tensor * inp_emb = nullptr, * positions = nullptr, * mask = nullptr, * ple_in = nullptr;
    ggml_tensor * res_hc = nullptr, * xn_next = nullptr, * inject = nullptr, * routed = nullptr;
    Qwen4ExpMoeRoute route;
    bool done = false;   // the first stream skips the last layer's FFN (no output rows)
    // The last layer's attention output, every row, for the MTP hidden.
    ggml_tensor * last_cur = nullptr, * last_inject = nullptr, * last_res = nullptr;
};

// The attention inputs of a run of layers: the whole forward's, or one stream's.
struct Qwen4ExpLayerInputs {
    int64_t pos0 = 0, kv_len = 0;
    int n_pooled = 0;
    Qwen4ExpQsaMode qsa = QSA_DENSE;
    ggml_tensor * positions = nullptr, * mask = nullptr, * ple_in = nullptr;
};

// A layer's attention output, injection and residual, every row: the MTP hidden's FFN half runs on them.
struct Qwen4ExpBranch {
    ggml_tensor * cur = nullptr, * inject = nullptr, * res = nullptr;
};

// The per-layer pieces of a forward graph. The single-stream layers and each stream of a pipelined prompt chunk are
// built from them.
struct Qwen4ExpLayerBuilder {
    ggml_context * ctx;
    ggml_cgraph * gf;
    const Qwen4ExpWeights & w;
    Qwen4ExpCache & cache;
    bool verify, split, has_ple;
    std::vector<ggml_tensor *> & expert_nodes;          // split mode: pinned to the expert device
    ggml_tensor * kv_row = nullptr;                     // stable graphs: the K/V rows this forward writes
    const std::vector<Qwen4ExpAttnRow> * rows = nullptr;   // verify: each row's T=1 attention inputs
    const Qwen4ExpDecodeWorkspace * qsa_ws = nullptr;   // stable QSA: the graph's runtime QSA inputs
    std::vector<int> lin_idx, full_idx, ple_idx;        // per layer: its index among the linear, full and PLE layers

    Qwen4ExpLayerBuilder(ggml_context * ctx, ggml_cgraph * gf, const Qwen4ExpWeights & w, Qwen4ExpCache & cache,
                         bool verify, bool split, bool has_ple, std::vector<ggml_tensor *> & expert_nodes)
        : ctx(ctx), gf(gf), w(w), cache(cache), verify(verify), split(split), has_ple(has_ple),
          expert_nodes(expert_nodes), lin_idx(w.n_layer, -1), full_idx(w.n_layer, -1), ple_idx(w.n_layer, -1) {
        for (size_t i = 0; i < cache.linear_layer_ids.size(); ++i) lin_idx[cache.linear_layer_ids[i]] = (int) i;
        for (size_t i = 0; i < cache.full_layer_ids.size(); ++i)   full_idx[cache.full_layer_ids[i]] = (int) i;
        for (size_t i = 0; i < cache.ple_layer_ids.size(); ++i)    ple_idx[cache.ple_layer_ids[i]] = (int) i;
    }

    // Layer il + 1 starts with PLE, so layer il's output cannot be fused with its attention norm.
    bool next_is_ple(int il) const { return il + 1 < w.n_layer && w.layers[il + 1].is_ple && has_ple; }

    // The HC mix into layer il's attention input, from the previous layer's fused norm when xn_next holds it
    // (consumed). `inject` receives the mix's injection.
    ggml_tensor * attn_input(int il, ggml_tensor * res_hc, ggml_tensor *& xn_next, ggml_tensor *& inject) {
        const Qwen4ExpLayer & L = w.layers[il];
        ggml_tensor * cur = xn_next != nullptr
            ? hc_mix_from_xn(ctx, xn_next, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject, &inject, w.n_embd, w.n_hc)
            : hc_mix(ctx, res_hc, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject, &inject,
                     w.n_embd, w.n_hc, w.rms_eps);
        xn_next = nullptr;
        return cur;
    }

    // Attention half of layer il: PLE, the HC mix into the attention input (attn_input), and the attention.
    ggml_tensor * attn_half(int il, const Qwen4ExpLayerInputs & b, ggml_tensor *& res_hc, ggml_tensor *& xn_next,
                            ggml_tensor *& inject) {
        const Qwen4ExpLayer & L = w.layers[il];
        if (L.is_ple && has_ple) {
            res_hc = build_ple(ctx, gf, res_hc, b.ple_in, L, w, cache.ple_conv_state[ple_idx[il]],
                               verify ? cache.spec_ple : nullptr);
            xn_next = nullptr;   // PLE changed the residual; the norm must rerun
        }
        ggml_tensor * cur = attn_input(il, res_hc, xn_next, inject);
        if (L.is_full_attention) {
            const int fi = full_idx[il];
            return build_full_attn(ctx, gf, cur, L, w,
                                   cache.attn_k[fi], cache.attn_v[fi], cache.indexer_k[fi],
                                   w.qsa ? cache.indexer_raw[fi] : nullptr,
                                   b.positions, b.mask, kv_row, b.kv_len, b.pos0,
                                   il < (int) w.compress_ratios.size() ? w.compress_ratios[il] : 0,
                                   b.n_pooled, b.qsa, rows, qsa_ws);
        }
        const int li = lin_idx[il];
        return build_linear_attn(ctx, gf, cur, L, w, cache.ssm_state[li], cache.conv_state[li],
                                 verify ? cache.spec_ssm[li] : nullptr, verify ? cache.spec_conv[li] : nullptr);
    }

    // The last layer's FFN runs on the output row only. Upstream selects output rows before the final HC/FFN, so its
    // quantized matmuls dispatch with one token (MMV rather than MMQ); this reuses that selection after all attention
    // cache writes. Earlier rows have no remaining stateful consumers.
    void last_row(ggml_tensor *& cur, ggml_tensor *& inject, ggml_tensor *& res_hc, int64_t rows_T) {
        cur = ggml_view_2d(ctx, cur, w.n_embd, 1, cur->nb[1], (rows_T - 1)*cur->nb[1]);
        inject = ggml_view_2d(ctx, inject, inject->ne[0], 1, inject->nb[1], (rows_T - 1)*inject->nb[1]);
        res_hc = ggml_view_3d(ctx, res_hc, w.n_embd, w.n_hc, 1,
                              res_hc->nb[1], res_hc->nb[2], (rows_T - 1)*res_hc->nb[2]);
    }

    // HC combine of the attention output into the FFN norm, and the HC mix into the MoE input.
    ggml_tensor * ffn_input(int il, ggml_tensor * cur, ggml_tensor *& inject, ggml_tensor *& res_hc, int64_t layer_T) {
        const Qwen4ExpLayer & L = w.layers[il];
        ggml_tensor * ffn_fused = hc_combine_norm(ctx, inject, res_hc, cur,
            L.hc_ffn_norm, w.n_embd, w.n_hc, layer_T, w.rms_eps);
        res_hc = hc_norm_res(ctx, ffn_fused, w.n_embd, w.n_hc, layer_T);
        return hc_mix_from_xn(ctx, hc_norm_xn(ctx, ffn_fused, w.n_embd, w.n_hc, layer_T),
                              L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, &inject,
                              w.n_embd, w.n_hc);
    }

    // HC combine of the MoE output into the residual, fused with the next layer's attention norm (left in xn_next)
    // unless that layer starts with PLE. Returns the new residual.
    ggml_tensor * ffn_output(int il, ggml_tensor * moe_out, ggml_tensor * inject, ggml_tensor * res_hc,
                             int64_t layer_T, ggml_tensor *& xn_next) {
        if (next_is_ple(il)) {
            xn_next = nullptr;
            return hc_combine(ctx, res_hc, moe_out, inject, w.n_embd, w.n_hc, layer_T);
        }
        ggml_tensor * gamma = (il + 1 < w.n_layer) ? w.layers[il + 1].hc_attn_norm : w.output_hc_norm;
        ggml_tensor * f = hc_combine_norm(ctx, inject, res_hc, moe_out,
            gamma, w.n_embd, w.n_hc, layer_T, w.rms_eps);
        xn_next = hc_norm_xn(ctx, f, w.n_embd, w.n_hc, layer_T);
        return hc_norm_res(ctx, f, w.n_embd, w.n_hc, layer_T);
    }

    // FFN half of layer il: ffn_input, the MoE and ffn_output. Returns the new residual.
    ggml_tensor * ffn_half(int il, ggml_tensor * cur, ggml_tensor * inject, ggml_tensor * res_hc, int64_t layer_T,
                           ggml_tensor *& xn_next) {
        const Qwen4ExpLayer & L = w.layers[il];
        cur = ffn_input(il, cur, inject, res_hc, layer_T);
        // Prefill: the MoE combine runs inside the next HC_COMBINE_NORM (one kernel fewer; last-bit numerics change
        // from FMA contraction in the new kernel, covered by the long-prompt quality gate). Not at T=1: it cost ~1.7% decode.
        Qwen4ExpMoeParts moe_parts;
        const bool fold = !split && !next_is_ple(il) && ggml_backend_cuda_mmb_prefill(layer_T);
        cur = build_moe(ctx, cur, L, w, fold ? &moe_parts : nullptr, split ? &expert_nodes : nullptr);
        if (!fold) return ffn_output(il, cur, inject, res_hc, layer_T, xn_next);
        ggml_tensor * gamma = (il + 1 < w.n_layer)
            ? w.layers[il + 1].hc_attn_norm : w.output_hc_norm;
        ggml_tensor * f = ggml_hc_combine_norm_moe(ctx, inject, res_hc, moe_parts.down, moe_parts.weights,
            moe_parts.shared, moe_parts.shared_logit, gamma, 1.0f / (float) w.n_hc, 0.0f, 2.0f, 0.0f, w.rms_eps);
        xn_next = hc_norm_xn(ctx, f, w.n_embd, w.n_hc, layer_T);
        return hc_norm_res(ctx, f, w.n_embd, w.n_hc, layer_T);
    }

    // The output head's HC mix, from the last layer's fused norm when xn_next holds it.
    ggml_tensor * output_mix(ggml_tensor * res_hc, ggml_tensor * xn_next) const {
        return xn_next != nullptr
            ? hc_mix_from_xn(ctx, xn_next, w.output_hc_down, w.output_hc_up, nullptr, nullptr, w.n_embd, w.n_hc)
            : hc_mix(ctx, res_hc, w.output_hc_norm, w.output_hc_down, w.output_hc_up, nullptr, nullptr,
                     w.n_embd, w.n_hc, w.rms_eps);
    }
};

// Emits the layers of units whose work alternates between the target and the expert device (a split prompt chunk's
// streams, a multi-slot decode step's slot groups) in the order build_pipelined_layers describes: per layer,
// R(0) C(n-1, l-1) S(0)  R(1) C(0) S(1) ... Needs two or more units, so that a unit's C(l) precedes its R(l + 1).
template <class Unit, class R, class S, class C>
static void interleave_layers(int n_layer, std::vector<Unit> & units, R && stage_r, S && stage_s, C && stage_c) {
    for (int il = 0; il < n_layer; ++il) {
        for (size_t i = 0; i < units.size(); ++i) {
            stage_r(units[i], il);
            if (i > 0) {
                stage_c(units[i - 1], il);
            } else if (il > 0) {
                stage_c(units.back(), il - 1);
            }
            stage_s(units[i], il);
        }
    }
    stage_c(units.back(), n_layer - 1);
}

// A split prompt chunk as n streams of about 4096 rows (2..4, multiples of 256; the last takes the rest). Per layer
// the graph is emitted as
//   R(0) C(n-1,l-1) S(0)  R(1) C(0) S(1)  ...  R(n-1) C(n-2) S(n-1),
// R = target work (attention, HC, router, shared expert), S = the routed experts on the expert device, C = the
// target-side combine. A stream's own chain per layer (R, copy, S, copy, C) is longer than either device's share of
// the layer, so two streams leave the copies on the critical path; more streams keep both devices busy.
// Creates the streams and their inputs, leaves the last stream's residual and fused norm in res_hc / xn_next and
// returns the QSA blocks pooled once the chunk computes.
static int build_pipelined_layers(Qwen4ExpLayerBuilder & lb, int64_t pos0, int64_t T, bool want_hidden,
                                  std::vector<Qwen4ExpStream> & streams, ggml_tensor *& emb_rows,
                                  ggml_tensor *& res_hc, ggml_tensor *& xn_next) {
    const Qwen4ExpWeights & w = lb.w;
    ggml_context * ctx = lb.ctx;
    ggml_cgraph * gf = lb.gf;
    int pipeline_blocks = lb.cache.indexer_blocks;
    const int64_t r = std::max<int64_t>(1, qsa_ratio(w));
    const int n_streams = (int) std::clamp<int64_t>(T / kQwen4ExpPipelineStreamRows, 2, 4);
    streams.resize(n_streams);
    for (int i = 0; i + 1 < n_streams; ++i) {
        streams[i].T = std::max<int64_t>(256, (T / n_streams) / 256 * 256);
    }
    streams.back().T = T;
    for (int i = 0; i + 1 < n_streams; ++i) streams.back().T -= streams[i].T;
    emb_rows = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, T);
    ggml_set_input(emb_rows);
    int64_t start = pos0;
    for (Qwen4ExpStream & st : streams) {
        st.pos0 = start;
        start += st.T;
        st.qsa = qsa_mode(w, lb.cache, st.T, st.pos0, w.qsa);
        st.n_pooled = pipeline_blocks;   // what the previous stream commits, as for sequential chunks
        if (st.qsa != QSA_DENSE && (st.pos0 + st.T) / r > w.indexer_top_k / r) {
            pipeline_blocks = (int) ((st.pos0 + st.T) / r);
        }
        st.inp_emb = ggml_view_2d(ctx, emb_rows, w.n_embd, st.T, emb_rows->nb[1], (st.pos0 - pos0) * emb_rows->nb[1]);
        st.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * st.T);
        ggml_set_input(st.positions);
        if (st.qsa == QSA_DENSE) {
            st.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, st.pos0 + st.T, st.T);
            ggml_set_input(st.mask);
        }
        if (lb.has_ple) {
            st.ple_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.ple_head_dim * w.ple_n_heads, st.T);
            ggml_set_input(st.ple_in);
        }
        st.res_hc = repeat_dim1(ctx, ggml_reshape_3d(ctx, st.inp_emb, w.n_embd, 1, st.T), w.n_hc);
    }
    auto stage_r = [&](Qwen4ExpStream & st, int il) {
        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = lb.attn_half(il, { st.pos0, st.pos0 + st.T, st.n_pooled, st.qsa, st.positions, st.mask,
                                               st.ple_in }, st.res_hc, st.xn_next, inject);
        st.layer_T = st.T;
        if (il == w.n_layer - 1) {
            if (want_hidden) {
                st.last_cur = cur;
                st.last_inject = inject;
                st.last_res = st.res_hc;
            }
            if (&st != &streams.back()) {   // only its cache writes are needed
                ggml_build_forward_expand(gf, cur);
                st.done = true;
                return;
            }
            if (st.T > 1) {
                lb.last_row(cur, inject, st.res_hc, st.T);
                st.layer_T = 1;
            }
        }
        st.inject = inject;
        cur = lb.ffn_input(il, cur, st.inject, st.res_hc, st.layer_T);
        st.route = build_moe_route(ctx, cur, w.layers[il], w);
        for (ggml_tensor * t : { st.route.xin, st.route.sel, st.route.wsel, st.route.shared, st.inject, st.res_hc }) {
            ggml_build_forward_expand(gf, t);
        }
    };
    auto stage_s = [&](Qwen4ExpStream & st, int il) {
        if (st.done) return;
        st.routed = build_moe_routed(ctx, st.route, w.layers[il], w, lb.expert_nodes);
        ggml_build_forward_expand(gf, st.routed);
    };
    auto stage_c = [&](Qwen4ExpStream & st, int il) {
        if (st.done) return;
        st.res_hc = lb.ffn_output(il, moe_join(ctx, st.routed, st.route.shared), st.inject, st.res_hc,
                                  st.layer_T, st.xn_next);
        if (st.xn_next) ggml_build_forward_expand(gf, st.xn_next);
        ggml_build_forward_expand(gf, st.res_hc);
    };
    interleave_layers(w.n_layer, streams, stage_r, stage_s, stage_c);
    res_hc = streams.back().res_hc;
    xn_next = streams.back().xn_next;
    return pipeline_blocks;
}

// The streams' inputs: embeddings, M-RoPE positions (section-major), the causal masks over each stream's prefix and
// the PLE rows.
static void upload_pipeline_inputs(const std::vector<Qwen4ExpStream> & streams, const Qwen4ExpWeights & w,
                                   const std::vector<float> & emb, const std::vector<float> & ple) {
    int64_t t0 = 0;
    for (const Qwen4ExpStream & st : streams) {
        std::vector<int32_t> sp((size_t) 4 * st.T, 0);
        for (int64_t i = 0; i < st.T; ++i) {
            sp[(size_t) i] = sp[(size_t) (st.T + i)] = sp[(size_t) (2 * st.T + i)] = (int32_t) (st.pos0 + i);
        }
        ggml_backend_tensor_set(st.inp_emb, emb.data() + (size_t) t0 * w.n_embd, 0, sizeof(float) * w.n_embd * st.T);
        ggml_backend_tensor_set(st.positions, sp.data(), 0, sizeof(int32_t) * sp.size());
        if (st.mask) {
            const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
            const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
            const int64_t cols = st.pos0 + st.T;
            std::vector<ggml_fp16_t> sm((size_t) cols * st.T);
            for (int64_t row = 0; row < st.T; ++row) {
                for (int64_t col = 0; col < cols; ++col) {
                    sm[(size_t) (row * cols + col)] = col <= st.pos0 + row ? zero : ninf;
                }
            }
            ggml_backend_tensor_set(st.mask, sm.data(), 0, sizeof(ggml_fp16_t) * sm.size());
        }
        if (st.ple_in) {
            const size_t row = (size_t) w.ple_head_dim * w.ple_n_heads;
            ggml_backend_tensor_set(st.ple_in, ple.data() + (size_t) t0 * row, 0, sizeof(float) * row * st.T);
        }
        t0 += st.T;
    }
}

// True when QSA rows can attend inside a retained graph: ratio-4 indexers with a 2048-key budget, a context within
// the runtime-count top-k's range, and a backend whose top-k has that variant. No allocation or host readback: the
// backend is asked about the op. In particular, nondeterministic CUDA DeviceTopK is excluded.
static bool stable_qsa_supported(ggml_backend_t backend, const Qwen4ExpWeights & w, const Qwen4ExpCache & cache,
                                 int64_t kv_end) {
    if (qsa_ratio(w) != 4 || w.indexer_top_k != 2048 || kv_end > kQwen4ExpStableQsaMaxCtx) return false;
    ggml_tensor scores{}, valid{}, selection{};
    scores.type = GGML_TYPE_F32;
    scores.ne[0] = qwen4exp_kv_bucket(cache.max_ctx, kv_end, true) / 4;
    scores.ne[1] = scores.ne[2] = scores.ne[3] = 1;
    scores.nb[0] = sizeof(float);
    scores.nb[1] = scores.nb[2] = scores.nb[3] = scores.ne[0] * sizeof(float);
    valid.type = GGML_TYPE_I32;
    valid.ne[0] = valid.ne[1] = valid.ne[2] = valid.ne[3] = 1;
    selection.op = GGML_OP_TOP_K;
    selection.type = GGML_TYPE_I32;
    selection.ne[0] = 512;
    selection.ne[1] = selection.ne[2] = selection.ne[3] = 1;
    selection.src[0] = &scores;
    selection.src[1] = &valid;
    if (!ggml_backend_supports_op(backend, &selection)) return false;
    for (int il = 0; il < w.n_layer; ++il) {
        if (w.layers[il].is_full_attention &&
            (il >= (int) w.compress_ratios.size() || w.compress_ratios[il] != 4)) return false;
    }
    return true;
}

// A retained graph built for another backend, model or context cannot replay.
static bool stable_workspace_stale(const Qwen4ExpDecodeWorkspace & ws, ggml_backend_t backend,
                                   const Qwen4ExpWeights & w, int max_ctx) {
    return ws.backend != backend || ws.model != &w || ws.max_ctx != max_ctx;
}

// One forward's inputs to a retained (stable) graph, and the host copies that must outlive their asynchronous upload
// until the graph computes and its outputs are read.
struct Qwen4ExpStableInputs {
    ggml_backend_t backend = nullptr;
    bool split = false;   // the scheduler may place an input in another backend's buffer: upload synchronously
    bool qsa = false;     // the graph attends QSA rows through runtime inputs
    int64_t pos0 = 0, T = 0, kv_len = 0;
    int max_ctx = 0;
    const std::vector<float> * emb = nullptr, * ple = nullptr;
    std::vector<float> qsa_visibility;
    std::vector<int32_t> qsa_params, i32;
    std::vector<ggml_fp16_t> mask;

    // Async on the target stream; synchronous in split mode.
    void set(ggml_tensor * t, const void * data, size_t size) const {
        if (split) ggml_backend_tensor_set(t, data, 0, size);
        else       ggml_backend_tensor_set_async(backend, t, data, 0, size);
    }
};

// Rows pos0..pos0+T-1 (layout: Qwen4ExpDecodeWorkspace::qsa_params): row t sees its (pos0+t+1)/4 complete blocks.
// Slot s pools block pos0/4+s when this forward completes it; otherwise it recomputes the last block complete before
// the rows.
static void upload_stable_qsa(Qwen4ExpStableInputs & in, const Qwen4ExpDecodeWorkspace & ws) {
    if (!in.qsa) return;
    const int64_t T = in.T, pos0 = in.pos0;
    const int64_t blocks = ws.qsa_blocks, last = (pos0 + 1) / 4 - 1;
    in.qsa_visibility.assign((size_t) (blocks * T), -INFINITY);
    in.qsa_params.assign((size_t) (10 * T), 0);
    for (int64_t t = 0; t < T; ++t) {
        const int64_t n = (pos0 + t + 1) / 4;
        std::fill_n(in.qsa_visibility.begin() + t * blocks, n, 0.0f);
        in.qsa_params[10 * t] = (int32_t) n;
    }
    for (int64_t s = 0; s < (T + 3) / 4; ++s) {
        const int64_t block = pos0 / 4 + s;
        const bool done = block < in.kv_len / 4;
        int32_t * p = &in.qsa_params[10 * s];
        for (int i = 0; i < 4; ++i) p[1 + i] = (int32_t) (4 * (done ? block : last) + i);
        p[5] = done ? (int32_t) block : (in.max_ctx + 3) / 4;
        p[6] = p[7] = p[8] = p[1];
    }
    in.set(ws.qsa_visibility, in.qsa_visibility.data(), in.qsa_visibility.size() * sizeof(float));
    in.set(ws.qsa_params, in.qsa_params.data(), in.qsa_params.size() * sizeof(int32_t));
}

// Every position-dependent input of a stable graph: the M-RoPE positions (section-major: 0..2 carry the position, 3
// is zero), the K/V rows, the dense masks (T=1: over the bucket; verify: each dense row's over its span), each QSA
// row's positions, QSA.
static void upload_stable_inputs(Qwen4ExpStableInputs & in, const Qwen4ExpDecodeWorkspace & ws) {
    const int64_t T = in.T, pos0 = in.pos0;
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    in.i32.assign((size_t) 9 * T, 0);
    int32_t * pos = in.i32.data(), * rows = pos + 4 * T, * row_pos = rows + T;
    for (int64_t i = 0; i < T; ++i) {
        pos[i] = pos[T + i] = pos[2 * T + i] = rows[i] = (int32_t) (pos0 + i);
        row_pos[4 * i] = row_pos[4 * i + 1] = row_pos[4 * i + 2] = (int32_t) (pos0 + i);
    }
    int64_t mask_elems = ws.mask ? ws.kv_bucket : 0;
    for (int64_t span : ws.row_spans) mask_elems += span;
    in.mask.assign((size_t) mask_elems, ninf);
    ggml_fp16_t * m = in.mask.data();
    in.set(ws.inp_emb, in.emb->data(), sizeof(float) * in.emb->size());
    in.set(ws.positions, pos, sizeof(int32_t) * 4 * T);
    in.set(ws.kv_row, rows, sizeof(int32_t) * T);
    if (ws.ple_in) in.set(ws.ple_in, in.ple->data(), sizeof(float) * in.ple->size());
    if (ws.mask) {
        std::fill_n(m, in.kv_len, zero);
        in.set(ws.mask, m, sizeof(ggml_fp16_t) * ws.kv_bucket);
        m += ws.kv_bucket;
    }
    for (size_t t = 0; t < ws.row_spans.size(); ++t) {
        if (ws.row_spans[t] == 0) {
            if (ws.row_inputs[t]) in.set(ws.row_inputs[t], row_pos + 4 * t, 4 * sizeof(int32_t));
            continue;
        }
        std::fill_n(m, pos0 + t + 1, zero);
        in.set(ws.row_inputs[t], m, sizeof(ggml_fp16_t) * ws.row_spans[t]);
        m += ws.row_spans[t];
    }
    upload_stable_qsa(in, ws);
}

// Replay a retained graph with this forward's inputs and read its logits (and hidden rows). A failed compute drops
// the graph.
static bool run_stable_graph(Qwen4ExpStableInputs & in, Qwen4ExpDecodeWorkspace & ws, ggml_backend_sched_t short_sched,
                             bool verify, std::vector<float> & out_logits, std::vector<float> * out_hidden) {
    upload_stable_inputs(in, ws);
    ggml_status status;
    {   // verify: as built, with batch-invariant matmul columns (see forward_impl)
        ScopedCudaGraphOverrides invariant(false, 0, false, 0, /*mmvq_batch_invariant=*/verify);
        status = in.split ? ggml_backend_sched_graph_compute(ws.sched ? ws.sched : short_sched, ws.gf)
                          : ggml_backend_graph_compute(in.backend, ws.gf);
    }
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[qwen4exp] stable graph compute failed\n");
        clear_qwen4exp_decode_workspace(ws);
        return false;
    }
    out_logits.resize((size_t) ggml_nelements(ws.logits));   // n_vocab, per row when verifying
    ggml_backend_tensor_get(ws.logits, out_logits.data(), 0, ggml_nbytes(ws.logits));
    if (out_hidden && ws.hidden) {
        out_hidden->resize((size_t) ggml_nelements(ws.hidden));
        ggml_backend_tensor_get(ws.hidden, out_hidden->data(), 0, ggml_nbytes(ws.hidden));
    }
    return true;
}

// Dense decode may not have maintained the pooled QSA prefix: pool the already-cached complete blocks before pos0
// once. The stable graph handles this forward's own blocks.
static bool bootstrap_qsa_prefix(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache, int pos0) {
    ggml_context * bc = ggml_init({8 * 1024 * 1024, nullptr, true});
    if (!bc) return false;
    ggml_cgraph * bg = ggml_new_graph_custom(bc, 8192, false);
    for (size_t fi = 0; fi < cache.full_layer_ids.size(); ++fi) {
        const auto & L = w.layers[cache.full_layer_ids[fi]];
        qsa_pooled_keys(bc, bg, L, w, cache.indexer_k[fi], cache.indexer_raw[fi], nullptr,
                        pos0, 4, cache.indexer_blocks, pos0 / 4);
    }
    ggml_gallocr_t ba = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    const bool ok = ba && ggml_gallocr_alloc_graph(ba, bg) &&
        ggml_backend_graph_compute(backend, bg) == GGML_STATUS_SUCCESS;
    ggml_backend_cuda_graph_invalidate_range(backend, ggml_get_mem_buffer(bc), ggml_get_mem_size(bc));
    if (ba) ggml_gallocr_free(ba);
    ggml_free(bc);
    if (ok) cache.indexer_blocks = pos0 / 4;
    return ok;
}

// The graph's metadata context: a decode or verify workspace's own (`pool`), reset for a rebuild, or a fresh one.
static ggml_context * graph_context(Qwen4ExpDecodeWorkspace * pool, ggml_backend_t backend, const Qwen4ExpWeights & w,
                                   int max_ctx, bool verify) {
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * kQwen4ExpGraphNodes +
                  ggml_graph_overhead_custom(kQwen4ExpGraphNodes, false) + (1u << 20);
    ip.no_alloc = true;
    if (!pool) return ggml_init(ip);
    if (pool->ctx == nullptr) {
        pool->ctx = ggml_init(ip);
    } else {
        // Retire captures before recycling metadata/allocator addresses. A verify
        // width rebuilds the same topology at the same addresses for a new span, so
        // its captures stay and are updated at the next capture; every replay still
        // checks node properties (as the expert device's captures always have).
        // Retiring them cost ~30 ms per rebuild on ROCm, one capture per split.
        if (!verify) {
            ggml_backend_cuda_graph_invalidate_range(backend,
                ggml_get_mem_buffer(pool->ctx), ggml_get_mem_size(pool->ctx));
        }
        ggml_reset(pool->ctx);
    }
    pool->gf = nullptr;
    pool->backend = backend;
    pool->model = &w;
    pool->max_ctx = max_ctx;
    return pool->ctx;
}

// Split mode: one scheduler for decode, verify and short batches, one for prompt chunks; a verify width keeps its
// own short-batch scheduler (and allocation) beside its graph (`pool`). The chunk planner measures the live
// split-input copies through ggml_backend_sched_reserve_size.
// The cache whose split schedulers `c` uses (Qwen4ExpCache::split_owner).
static Qwen4ExpCache & split_owner(Qwen4ExpCache & c) { return c.split_owner ? *c.split_owner : c; }

static ggml_backend_sched_t split_scheduler(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache,
                                            Qwen4ExpDecodeWorkspace * pool, bool verify, bool short_batch) {
    Qwen4ExpCache & owner = split_owner(cache);
    ggml_backend_sched_t & sched = verify && pool ? pool->sched
                                 : short_batch ? owner.split_sched_short : owner.split_sched;
    if (!sched) {
        if (!owner.split_cpu) owner.split_cpu = ggml_backend_cpu_init();
        ggml_backend_t backends[3] = { backend, w.expert_backend, owner.split_cpu };
        // Both are parallel: per-copy events make a split wait for its
        // cross-device inputs on the GPU; a single-copy scheduler instead
        // synchronizes the receiving device from the host before every such
        // split, which serializes the two GPUs. Split-input copies live for
        // the whole graph, so the prompt scheduler keeps one set.
        sched = ggml_backend_sched_new(backends, nullptr, 3, kQwen4ExpGraphNodes, true, false);
        // End a split before a node that brings in a new cross-device input,
        // so independent target work is queued before that wait and runs
        // beside the expert device: the other stream's attention in a prompt
        // chunk; the shared and hot experts in a decode or verify step.
        ggml_backend_sched_set_late_cross_input_split(sched, true);
        if (!short_batch) ggml_backend_sched_set_n_copies(sched, 1);   // events without four copy sets
    }
    return sched;
}

// MTP prompt catch-up: the MTP layer's K/V for every (embedding, hidden) pair of the chunk, and the chunk's last
// hidden row carried into cache.mtp_prev_hidden (`hidden` becomes that row). Returns the pairs' positions input.
static ggml_tensor * build_mtp_prefill(ggml_context * ctx, ggml_cgraph * gf, const Qwen4ExpWeights & w,
                                       Qwen4ExpCache & cache, ggml_tensor *& hidden, ggml_tensor * emb_rows,
                                       int64_t pos0, int64_t T) {
    const int64_t H = w.n_embd, hc = w.n_hc, hd = H * hc;
    const int64_t first = pos0 == 0 ? 1 : 0, pairs = T - first, start = pos0 - (1 - first);
    ggml_tensor * h = ggml_reshape_2d(ctx, hidden, hd, T);
    if (pos0 > 0) {
        h = ggml_concat(ctx, cache.mtp_prev_hidden,
            ggml_view_2d(ctx, h, hd, T - 1, h->nb[1], 0), 1);
    }
    ggml_tensor * mtp_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * pairs);
    ggml_set_input(mtp_positions);
    // One submission, but exactly the old 512-row GEMMs, including the first
    // chunk's short final slice. A single wider GEMM changes the K/V bytes.
    for (int64_t i = 0; i < pairs; i += 512) {
        const int64_t n = std::min<int64_t>(512, pairs - i);
        ggml_tensor * e = ggml_view_2d(ctx, emb_rows, H, n, emb_rows->nb[1], (i + first) * emb_rows->nb[1]);
        ggml_tensor * hi = ggml_view_2d(ctx, h, hd, n, h->nb[1], i * h->nb[1]);
        ggml_tensor * p = ggml_reshape_1d(ctx, ggml_cont(ctx,
            ggml_view_2d(ctx, mtp_positions, n, 4, pairs * sizeof(int32_t), i * sizeof(int32_t))), 4 * n);
        ggml_tensor * r = mtp_input(ctx, w, e, hi);
        const auto & L = w.mtp;
        ggml_tensor * cur = hc_mix(ctx, r, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up,
            nullptr, nullptr, H, hc, w.rms_eps);
        build_full_attn(ctx, gf, cur, L, w, cache.mtp_k, cache.mtp_v, nullptr, nullptr,
            p, nullptr, nullptr, start + i + n, start + i, 4, 0, QSA_DENSE,
            nullptr, nullptr, /*kv_only=*/true);
    }
    // Carry one authoritative hidden row across chunks; only this row is
    // read back for the first decode step. All prompt pairs stay on device.
    hidden = ggml_view_2d(ctx, hidden, hd, 1, hidden->nb[2], (T - 1) * hidden->nb[2]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, hidden, cache.mtp_prev_hidden));
    return mtp_positions;
}

// The MTP pairs' M-RoPE positions: pair i sits at the position before its embedding's token.
static void upload_mtp_positions(ggml_tensor * mtp_positions, int64_t pos0, int64_t T) {
    const int64_t pairs = T - (pos0 == 0 ? 1 : 0), start = pos0 == 0 ? 0 : pos0 - 1;
    std::vector<int32_t> p((size_t) 4 * pairs, 0);
    for (int64_t i = 0; i < pairs; ++i) p[i] = p[pairs + i] = p[2 * pairs + i] = (int32_t) (start + i);
    ggml_backend_tensor_set(mtp_positions, p.data(), 0, ggml_nbytes(mtp_positions));
}

// The target's share of the split scheduler, copies included. The planner probes shapes no request has built (up to
// 32768 rows at the end of the context); a graph with a node no device runs keeps the single-device estimate, as the
// scheduler could not place it.
static void measure_split_graph(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache,
                                ggml_cgraph * gf, const std::vector<ggml_tensor *> & expert_nodes, int64_t T, int pos0,
                                bool verify, Qwen4ExpGraphMemory & measure) {
    ggml_backend_sched_t sched = split_scheduler(backend, w, cache, nullptr, verify, T <= kSplitShortRows);
    const ggml_backend_t devices[3] = { backend, w.expert_backend, split_owner(cache).split_cpu };
    const ggml_tensor * unplaced = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(gf) && !unplaced; ++i) {
        const ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) continue;
        if (std::none_of(devices, devices + 3, [&](ggml_backend_t b) { return b && ggml_backend_supports_op(b, node); })) {
            unplaced = node;
        }
    }
    if (unplaced) {
        std::fprintf(stderr, "[qwen4exp] split plan T=%lld pos0=%d: %s (%s) runs on no device; single-device estimate\n",
                     (long long) T, pos0, ggml_op_desc(unplaced), unplaced->name);
        return;
    }
    ggml_backend_sched_reset(sched);
    qwen4exp_pin_split(sched, gf, backend, w.expert_backend, expert_nodes);
    size_t sizes[3] = {};
    ggml_backend_sched_reserve_size(sched, gf, sizes);
    ggml_backend_sched_reset(sched);
    if (T <= kSplitShortRows) ++split_owner(cache).split_short_gen;   // a retained stable graph must allocate again
    measure.graph = sizes[0];
}

// Point a prompt chunk's inputs at this call's slot of the pinned host ring before allocation, so the allocator leaves
// them alone. False when the ring cannot take them (they are then allocated and uploaded as usual).
static bool place_inputs_in_ring(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache, int64_t T,
                                 int64_t mask_len, bool has_ple, ggml_tensor * inp_emb, ggml_tensor * positions,
                                 ggml_tensor * mask, ggml_tensor * ple_in) {
    const size_t embd_need = static_cast<size_t>(w.n_embd) * T * sizeof(float);
    const size_t pos_need  = static_cast<size_t>(4) * T * sizeof(int32_t);
    const size_t ple_need  = has_ple
        ? static_cast<size_t>(w.ple_head_dim) * w.ple_n_heads * T * sizeof(float) : 0;
    const size_t mask_need = mask
        ? static_cast<size_t>(mask_len) * T * sizeof(ggml_fp16_t) : 0;
    ggml_backend_buffer_type_t host_buft =
        ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
    if (host_buft == nullptr ||
        !qwen4exp_input_ring_reserve(cache.input_ring, host_buft, embd_need, pos_need, ple_need, mask_need)) {
        return false;
    }
    // Wait for the graph that last read this slot; the logits read already synchronizes each forward.
    if (cache.input_ring.writes >= 2) {
        ggml_backend_synchronize(backend);
    }
    const size_t slot = static_cast<size_t>(cache.input_ring.next_slot);
    cache.input_ring.next_slot = (cache.input_ring.next_slot + 1) % 2;
    cache.input_ring.writes++;
    char * slot_base = cache.input_ring.base + slot * cache.input_ring.slot_bytes;
    ggml_backend_tensor_alloc(cache.input_ring.buf, inp_emb, slot_base + cache.input_ring.embd_off);
    ggml_backend_tensor_alloc(cache.input_ring.buf, positions, slot_base + cache.input_ring.pos_off);
    if (mask) {
        ggml_backend_tensor_alloc(cache.input_ring.buf, mask, slot_base + cache.input_ring.mask_off);
    }
    if (ple_in) {
        ggml_backend_tensor_alloc(cache.input_ring.buf, ple_in, slot_base + cache.input_ring.ple_off);
    }
    return true;
}

// The CUDA pools keep every block they free, so the growing temporaries of a long prompt can hold the memory a graph
// allocation needs. Release the devices' cached blocks (ggml_backend_cuda_trim_pool); true when any came back.
static bool release_cached_pools(ggml_backend_t backend, const Qwen4ExpWeights & w) {
    size_t freed = ggml_backend_cuda_trim_pool(backend);
    if (w.expert_backend) freed += ggml_backend_cuda_trim_pool(w.expert_backend);
    if (freed) std::fprintf(stderr, "[qwen4exp] released %zu MiB of cached pool memory for a graph\n", freed >> 20);
    return freed > 0;
}

// Allocate the graph: on the split scheduler (weights decide placement, see qwen4exp_pin_split), or with the
// workspace's retained allocator (`pool`), or a fresh one (`galloc`, the caller frees it). A failed allocation
// retries once with the cached pool memory released. On failure the caller frees its context unless the workspace
// owns it.
static bool allocate_graph(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache,
                           Qwen4ExpDecodeWorkspace * pool, bool verify, ggml_cgraph * gf,
                           const std::vector<ggml_tensor *> & expert_nodes, int64_t T, int64_t kv_len,
                           ggml_backend_sched_t & split_sched, ggml_gallocr_t & galloc) {
    if (w.expert_backend) {
        split_sched = split_scheduler(backend, w, cache, pool, verify, T <= kSplitShortRows);
        auto alloc = [&] {
            ggml_backend_sched_reset(split_sched);
            qwen4exp_pin_split(split_sched, gf, backend, w.expert_backend, expert_nodes);
            return ggml_backend_sched_alloc_graph(split_sched, gf);
        };
        if (!alloc() && !(release_cached_pools(backend, w) && alloc())) {
            std::fprintf(stderr, "[qwen4exp] split graph alloc failed (T=%lld kv_len=%lld)\n",
                         (long long) T, (long long) kv_len);
            return false;
        }
        if (split_sched == split_owner(cache).split_sched_short) ++split_owner(cache).split_short_gen;
        return true;
    }
    if (pool) {
        if (pool->alloc == nullptr) {
            pool->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        }
        galloc = pool->alloc;
    } else {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!galloc) {
        std::fprintf(stderr, "[qwen4exp] graph allocator creation failed\n");
        return false;
    }
    // T=1 graphs keep the same broad shape but advancing KV views can change
    // lifetimes. Recompute assignments while retaining the allocator buffers;
    // reusing the old index-wise plan produced incorrect tokens.
    auto alloc = [&] {
        return (!pool || !pool->planned || ggml_gallocr_reserve(galloc, gf)) && ggml_gallocr_alloc_graph(galloc, gf);
    };
    if (!alloc() && !(release_cached_pools(backend, w) && alloc())) {
        std::fprintf(stderr, "[qwen4exp] graph alloc failed (T=%lld kv_len=%lld)\n",
                     (long long) T, (long long) kv_len);
        if (pool) {
            pool->planned = false;
        } else {
            ggml_gallocr_free(galloc);
            galloc = nullptr;
        }
        return false;
    }
    if (pool) pool->planned = true;
    return true;
}

// A rebuilt (not retained) graph's inputs: the embeddings, the M-RoPE positions (section-major [s*T + i]: 0..2 carry
// the position, 3 is zero), the causal mask over mask_len keys, the PLE rows (into the host ring when `ring` placed
// them there), and each verify row's mask or positions.
static void upload_graph_inputs(ggml_tensor * inp_emb, ggml_tensor * positions, ggml_tensor * mask,
                                ggml_tensor * ple_in, const std::vector<float> & emb, const std::vector<float> & ple,
                                int64_t pos0, int64_t T, int64_t mask_len, bool ring,
                                const std::vector<Qwen4ExpAttnRow> & rows) {
    std::vector<int32_t> pos((size_t) 4 * T, 0);
    for (int64_t i = 0; i < T; ++i) {
        const int32_t p = (int32_t) (pos0 + i);
        pos[(size_t) (0 * T + i)] = p;
        pos[(size_t) (1 * T + i)] = p;
        pos[(size_t) (2 * T + i)] = p;
        pos[(size_t) (3 * T + i)] = 0;
    }
    std::vector<ggml_fp16_t> m;
    if (mask) {
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
        m.resize((size_t) mask_len * T);
        for (int64_t row = 0; row < T; ++row) {
            const int64_t vis = pos0 + row;
            for (int64_t col = 0; col < mask_len; ++col) {
                m[(size_t) (row * mask_len + col)] = (col <= vis) ? zero : ninf;
            }
        }
    }
    if (ring) {
        std::memcpy(inp_emb->data, emb.data(), sizeof(float) * emb.size());
        std::memcpy(positions->data, pos.data(), sizeof(int32_t) * pos.size());
        if (mask) {
            std::memcpy(mask->data, m.data(), sizeof(ggml_fp16_t) * m.size());
        }
        if (ple_in) {
            std::memcpy(ple_in->data, ple.data(), sizeof(float) * ple.size());
        }
    } else {
        ggml_backend_tensor_set(inp_emb, emb.data(), 0, sizeof(float) * emb.size());
        ggml_backend_tensor_set(positions, pos.data(), 0, sizeof(int32_t) * pos.size());
        if (mask) {
            ggml_backend_tensor_set(mask, m.data(), 0, sizeof(ggml_fp16_t) * m.size());
        }
        if (ple_in) {
            ggml_backend_tensor_set(ple_in, ple.data(), 0, sizeof(float) * ple.size());
        }
    }
    for (int64_t t = 0; t < (int64_t) rows.size(); ++t) {
        const int32_t p = (int32_t) (pos0 + t);
        if (rows[t].mask) {
            std::vector<ggml_fp16_t> row_mask((size_t) rows[t].span, ggml_fp32_to_fp16(-INFINITY));
            std::fill(row_mask.begin(), row_mask.begin() + p + 1, ggml_fp32_to_fp16(0.0f));
            ggml_backend_tensor_set(rows[t].mask, row_mask.data(), 0, ggml_nbytes(rows[t].mask));
        } else {
            const int32_t row_pos[4] = { p, p, p, 0 };
            ggml_backend_tensor_set(rows[t].positions, row_pos, 0, sizeof row_pos);
        }
    }
}

static Qwen4ExpForwardResult forward_impl(ggml_backend_t backend,
                                       const Qwen4ExpWeights & w,
                                       Qwen4ExpCache & cache,
                                       const int32_t * tokens,
                                       int n_tokens,
                                       int pos0,
                                       std::vector<float> & out_logits, std::vector<float> * out_hidden,
                                       bool verify, bool mtp_prefill,
                                       const Qwen4ExpInputs * inputs, Qwen4ExpGraphMemory * measure) {
    Qwen4ExpForwardResult res;
    if (n_tokens <= 0 || pos0 < 0 || (!tokens && !measure)) return res;
    const Qwen4ExpCudaScope profile(w.gfx1151 || w.expert_gfx1151);
    const bool split = w.expert_backend != nullptr;
    if (verify && (n_tokens < 2 || n_tokens > cache.mtp_draft + 1 || !qwen4exp_verify_supported(cache))) return res;
    if (mtp_prefill && (n_tokens <= 1 || verify || !out_hidden ||
        !qwen4exp_verify_supported(cache) || !cache.mtp_prev_hidden ||
        (!measure && pos0 > 0 && cache.mtp_prev_pos != pos0 - 1))) return res;
    if (!measure) cache.spec_tokens = 0;
    const Qwen4ExpQsaMode qsa = qsa_mode(w, cache, verify ? 1 : n_tokens, pos0, w.qsa);
    const bool reuse_ws = n_tokens == 1;
    // Verify: row t attends as a T=1 decode at pos0 + t (see Qwen4ExpAttnRow), over that
    // step's stable dense span, or 0 for QSA. The spans key this width's retained graph.
    std::vector<int64_t> row_spans;
    for (int t = 0; verify && t < n_tokens; ++t) {
        row_spans.push_back(qsa_mode(w, cache, 1, pos0 + t, w.qsa) != QSA_DENSE ? 0
            : qwen4exp_stable_kv_span(cache.kv_bucket_base, cache.max_ctx, int64_t(pos0) + t + 1));
    }
    const int64_t kv_end = int64_t(pos0) + n_tokens;
    const bool qsa_rows = reuse_ws ? qsa == QSA_DECODE : verify && row_spans.back() == 0;
    const bool stable_qsa = qsa_rows && stable_qsa_supported(backend, w, cache, kv_end);
    // T=1 decode and verify replay a retained graph, unless QSA rows lack the runtime-count path.
    const bool use_stable_graph = (reuse_ws || verify) && (!qsa_rows || stable_qsa);
    // Context and allocator reused across calls: the T=1 decode workspace, or this verify width's.
    Qwen4ExpDecodeWorkspace * pool = measure ? nullptr : reuse_ws ? &cache.decode_workspace
                                   : verify ? &cache.verify_workspace[n_tokens] : nullptr;
    if (pos0 > cache.max_ctx || n_tokens > cache.max_ctx - pos0) {
        std::fprintf(stderr, "[qwen4exp] context overflow: %d + %d > %d\n",
                     pos0, n_tokens, cache.max_ctx);
        return res;
    }

    const bool has_ple = !cache.ple_layer_ids.empty() && w.ple_reader.available();
    const int64_t ple_heads = w.ple_n_heads;
    Qwen4ExpInputs local_inputs;
    if (!measure && !inputs) {
        local_inputs = qwen4exp_prepare_inputs(w, tokens, n_tokens, cache.ple_prev);
        inputs = &local_inputs;
    }
    if (!measure && (!inputs->ok || inputs->emb.size() != (size_t) w.n_embd * n_tokens ||
        inputs->ple.size() != (has_ple ? (size_t) w.ple_head_dim * ple_heads * n_tokens : 0))) return res;
    if (verify && !measure) {
        auto prev = cache.ple_prev;
        for (int t = 0; t < n_tokens; ++t) {
            if (has_ple) {
                prev.push_back(tokens[t]);
                if ((int) prev.size() >= w.ple_ngram_size) prev.erase(prev.begin());
            }
            cache.spec_ple_prev[t] = prev;
        }
    }
    const auto & emb = measure ? local_inputs.emb : inputs->emb;
    const auto & ple_data = measure ? local_inputs.ple : inputs->ple;

    const int64_t T = n_tokens;
    const int64_t kv_len = pos0 + n_tokens;
    // Split prompt chunks without an MTP hidden capture (see Qwen4ExpStream).
    const bool pipeline = split && !verify && T >= 1024;
    // Give a new stable graph at least one full 256-token generation window.
    // The fixed mask excludes its padded tail, while the stable K/V views and
    // set_rows index keep every graph pointer and property unchanged.
    Qwen4ExpDecodeWorkspace measured_ws;
    Qwen4ExpDecodeWorkspace & stable_ws = measure ? measured_ws : verify ? cache.verify_workspace[T] : cache.decode_workspace;
    if (!measure && cache.decode_workspace.ctx &&
        (!reuse_ws || stable_workspace_stale(cache.decode_workspace, backend, w, cache.max_ctx))) {
        clear_qwen4exp_decode_workspace(cache.decode_workspace);
    }
    if (!measure && verify && stable_ws.ctx && stable_workspace_stale(stable_ws, backend, w, cache.max_ctx)) {
        clear_qwen4exp_decode_workspace(stable_ws);
    }
    const int64_t stable_kv_bucket = stable_qsa
        ? qwen4exp_kv_bucket(cache.max_ctx, kv_len, true)
        : use_stable_graph && !verify
            ? qwen4exp_stable_kv_span(cache.kv_bucket_base, cache.max_ctx, kv_len)
            : 0;
    Qwen4ExpStableInputs stable_in;
    stable_in.backend = backend;
    stable_in.split = split;
    stable_in.qsa = stable_qsa;
    stable_in.pos0 = pos0;
    stable_in.T = T;
    stable_in.kv_len = kv_len;
    stable_in.max_ctx = cache.max_ctx;
    stable_in.emb = &emb;
    stable_in.ple = &ple_data;

    if (!measure && stable_qsa && cache.indexer_blocks < pos0 / 4 && !bootstrap_qsa_prefix(backend, w, cache, pos0)) {
        clear_qwen4exp_decode_workspace(stable_ws);
        return res;
    }
    // In split mode the T=1 graph replays only while it still owns the short scheduler's
    // allocation: the scheduler rewrote its cross-device sources, so it is never split again.
    if (use_stable_graph && stable_ws.gf && (stable_ws.qsa_blocks >= 0) == stable_qsa &&
        (stable_ws.hidden != nullptr) == (out_hidden != nullptr) &&
        (!split || stable_ws.sched || stable_ws.split_gen == split_owner(cache).split_short_gen) &&
        (verify ? stable_ws.row_spans == row_spans : kv_len <= stable_ws.kv_bucket && stable_ws.next_pos == pos0) &&
        (!stable_qsa || (stable_ws.kv_bucket == stable_kv_bucket &&
                        stable_ws.qsa_budget == w.indexer_top_k / 4 && cache.indexer_blocks == pos0 / 4))) {
        if (!run_stable_graph(stable_in, stable_ws, split_owner(cache).split_sched_short, verify, out_logits, out_hidden)) {
            return res;
        }
        stable_ws.next_pos = (int) kv_len;
        cache.cur_pos = (int) kv_len;
        ++stable_ws.replays;
        if (stable_qsa) cache.indexer_blocks = (int) (kv_len / 4);
        if (verify) { cache.spec_pos = pos0; cache.spec_tokens = n_tokens; }
        cache.ple_prev = inputs->ple_prev;
        res.ok = true;
        return res;
    }
    ggml_context * ctx = graph_context(pool, backend, w, cache.max_ctx, verify);
    if (!ctx) return res;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, kQwen4ExpGraphNodes, false);
    std::vector<ggml_tensor *> expert_nodes;   // split mode: pinned to the expert device
    std::vector<Qwen4ExpAttnRow> rows;         // verify: each row's T=1 attention inputs
    Qwen4ExpLayerBuilder lb(ctx, gf, w, cache, verify, split, has_ple, expert_nodes);
    lb.rows = verify ? &rows : nullptr;
    lb.qsa_ws = stable_qsa ? &stable_ws : nullptr;

    const int64_t graph_kv_len = use_stable_graph && !verify ? stable_kv_bucket : kv_len;
    const int64_t mask_len = graph_kv_len;

    ggml_tensor * inp_emb = nullptr, * positions = nullptr, * mask = nullptr, * ple_in = nullptr;
    ggml_tensor * emb_rows = nullptr;   // every row's embedding (the MTP pairs read it)
    ggml_tensor * res_hc = nullptr, * xn_next = nullptr;
    std::vector<Qwen4ExpStream> streams;
    int pipeline_blocks = cache.indexer_blocks;
    Qwen4ExpBranch branch;   // single stream: the last layer's every-row attention output, for the MTP hidden

    if (pipeline) {
        pipeline_blocks = build_pipelined_layers(lb, pos0, T, out_hidden != nullptr, streams, emb_rows, res_hc, xn_next);
    } else {
        inp_emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, T);
        ggml_set_input(inp_emb);
        emb_rows = inp_emb;
        positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * T);
        ggml_set_input(positions);
        if (use_stable_graph) {
            lb.kv_row = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
            ggml_set_input(lb.kv_row);
        }
        if (stable_qsa) {
            stable_ws.kv_bucket = stable_kv_bucket;
            stable_ws.qsa_blocks = stable_kv_bucket / 4;
            stable_ws.qsa_visibility = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, stable_ws.qsa_blocks * T);
            stable_ws.qsa_params = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 10 * T);
            ggml_set_input(stable_ws.qsa_visibility);
            ggml_set_input(stable_ws.qsa_params);
        }
        // Prompt QSA derives visibility itself; verify owns a separate mask per dense row.
        if ((T > 1 || use_stable_graph) && qsa == QSA_DENSE && !verify) {
            mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, mask_len, T);
            ggml_set_input(mask);
        }
        // A stable graph whose rows are all QSA attends them in one pass over the graph's positions.
        const bool qsa_rows_at_once = stable_qsa && verify && row_spans.front() == 0;
        for (int64_t t = 0; verify && t < T; ++t) {
            Qwen4ExpAttnRow row;
            row.qsa = row_spans[t] > 0 ? QSA_DENSE : QSA_DECODE;
            row.span = row_spans[t];
            if (row.span == 0 && qsa_rows_at_once) {
                rows.push_back(row);
                continue;
            }
            ggml_tensor *& in = row.span > 0 ? row.mask : row.positions;
            in = row.span > 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, row.span, 1) : ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
            ggml_set_input(in);
            rows.push_back(row);
        }
        if (has_ple) {
            ple_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.ple_head_dim * ple_heads, T);
            ggml_set_input(ple_in);
        }

        res_hc = repeat_dim1(ctx,
            ggml_reshape_3d(ctx, inp_emb, w.n_embd, 1, T), w.n_hc);

        const Qwen4ExpLayerInputs layer_in = { pos0, graph_kv_len, cache.indexer_blocks, qsa, positions, mask, ple_in };
        for (int il = 0; il < w.n_layer; ++il) {
            ggml_tensor * inject = nullptr;
            ggml_tensor * cur = lb.attn_half(il, layer_in, res_hc, xn_next, inject);
            int64_t layer_T = T;
            if (!verify && il == w.n_layer - 1 && T > 1) {
                if (out_hidden) branch = { cur, inject, res_hc };   // every row's FFN half, for the MTP hidden only
                lb.last_row(cur, inject, res_hc, T);
                layer_T = 1;
            }
            res_hc = lb.ffn_half(il, cur, inject, res_hc, layer_T, xn_next);
        }
    }

    ggml_tensor * final = lb.output_mix(res_hc, xn_next);
    ggml_tensor * last = final->ne[1] > 1 && !verify
        ? ggml_view_2d(ctx, final, w.n_embd, 1, final->nb[1], (size_t) (final->ne[1] - 1) * final->nb[1])
        : final;
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, last);
    ggml_set_output(logits);
    ggml_set_name(logits, "logits");
    ggml_build_forward_expand(gf, logits);
    // The final HC residual of every row feeds the MTP draft head. Appended after the logits so the logits path keeps
    // its node order; a row-selected prefill builds the other rows' FFN half on the side.
    ggml_tensor * hidden = nullptr;
    if (out_hidden) {
        ggml_tensor * unused = nullptr;
        if (pipeline) {
            for (Qwen4ExpStream & st : streams) {
                ggml_tensor * h = lb.ffn_half(w.n_layer - 1, st.last_cur, st.last_inject, st.last_res, st.T, unused);
                hidden = hidden ? ggml_concat(ctx, hidden, h, 2) : h;
            }
        } else {
            hidden = branch.cur ? lb.ffn_half(w.n_layer - 1, branch.cur, branch.inject, branch.res, T, unused) : res_hc;
        }
        for (ggml_tensor * t = hidden; t; t = t->view_src) ggml_set_output(t);
        ggml_build_forward_expand(gf, hidden);
    }
    ggml_tensor * mtp_positions = mtp_prefill
        ? build_mtp_prefill(ctx, gf, w, cache, hidden, emb_rows, pos0, T) : nullptr;

    if (measure) {
        graph_memory(backend, ctx, gf, w.gfx1151, *measure);
        if (split) measure_split_graph(backend, w, cache, gf, expert_nodes, T, pos0, verify, *measure);
        // Graph inputs are included above. Reserve BOTH grow-only UMA ring slots
        // as well (conservative: the normal allocator excludes their tensors).
        const size_t input_bytes = inp_emb == nullptr ? 0 :   // the pipeline's inputs are per stream (no ring)
            ring_align_up(ggml_nbytes(inp_emb)) + ring_align_up(ggml_nbytes(positions)) +
            (ple_in ? ring_align_up(ggml_nbytes(ple_in)) : 0);
        measure->inputs = cache.input_ring.enabled ? 2 * input_bytes : 0;
        measure->mask = cache.input_ring.enabled && mask ? 2 * ring_align_up(ggml_nbytes(mask)) : 0;
        // Current and lookahead host embeddings/PLE, sorted row indices + read scratch.
        measure->host = 2 * ((size_t) w.n_embd * T * sizeof(float) +
            (has_ple ? (size_t) ple_heads * T * (w.ple_head_dim * sizeof(float) +
                3 * sizeof(int32_t) + w.ple_reader.row_bytes()) : 0));
        ggml_free(ctx);
        res.ok = true;
        return res;
    }

    const bool ring = !use_stable_graph && !verify && !pipeline && cache.input_ring.enabled &&
        place_inputs_in_ring(backend, w, cache, T, mask_len, has_ple, inp_emb, positions, mask, ple_in);
    ggml_gallocr_t galloc = nullptr;
    ggml_backend_sched_t split_sched = nullptr;
    if (!allocate_graph(backend, w, cache, pool, verify, gf, expert_nodes, T, kv_len, split_sched, galloc)) {
        if (!pool) ggml_free(ctx);
        return res;
    }

    if (use_stable_graph) {
        stable_ws.expert_nodes = expert_nodes;
        stable_ws.split_gen = split_owner(cache).split_short_gen;
        stable_ws.gf = gf;
        stable_ws.inp_emb = inp_emb;
        stable_ws.positions = positions;
        stable_ws.mask = mask;
        stable_ws.ple_in = ple_in;
        stable_ws.kv_row = lb.kv_row;
        stable_ws.logits = logits;
        stable_ws.hidden = hidden;
        stable_ws.kv_bucket = stable_kv_bucket;
        stable_ws.qsa_blocks = stable_qsa ? stable_kv_bucket / 4 : -1;
        stable_ws.row_spans = row_spans;
        stable_ws.row_inputs.clear();
        for (const Qwen4ExpAttnRow & row : rows) stable_ws.row_inputs.push_back(row.span > 0 ? row.mask : row.positions);
        ++stable_ws.builds;
        stable_ws.qsa_budget = stable_qsa ? w.indexer_top_k / 4 : 0;
        stable_ws.next_pos = (int) kv_len;
    }

    if (mtp_positions) upload_mtp_positions(mtp_positions, pos0, T);
    if (pipeline) {
        upload_pipeline_inputs(streams, w, emb, ple_data);
    } else if (use_stable_graph) {
        upload_stable_inputs(stable_in, stable_ws);
    } else {
        upload_graph_inputs(inp_emb, positions, mask, ple_in, emb, ple_data, pos0, T, mask_len, ring, rows);
    }

    ggml_status status;
    {   // verify: every matmul column equals its single-token product (see ggml_backend_cuda_set_mmvq_batch_invariant)
        // Split prompt chunks: no HIP graph capture (long kernels gain nothing, and
        // capture fails on allocating kernels such as the CUB segmented sort).
        ScopedCudaGraphOverrides invariant(split && T > kSplitShortRows, 0, false, 0, /*mmvq_batch_invariant=*/verify);
        status = split ? ggml_backend_sched_graph_compute(split_sched, gf) : ggml_backend_graph_compute(backend, gf);
    }
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[qwen4exp] graph compute failed\n");
        if (pool) {
            clear_qwen4exp_decode_workspace(*pool);
        } else {
            if (galloc) ggml_gallocr_free(galloc);
            ggml_free(ctx);
        }
        return res;
    }
    // A split prompt chunk's temporaries grow with its position, and the target's pool keeps every block it frees:
    // a long prompt would leave each chunk's behind until the device runs out. Release them after the chunk.
    if (split && T > kSplitShortRows) ggml_backend_cuda_trim_pool(backend);
    cache.ple_prev = inputs->ple_prev;
    // Commit only after the graph computed: a failed compute must not mark blocks the kernel never pooled.
    cache.cur_pos = (int) kv_len;
    if (mtp_prefill) cache.mtp_prev_pos = pos0 + n_tokens - 1;
    if (verify) { cache.spec_pos = pos0; cache.spec_tokens = n_tokens; }
    // The all-keys branch did not pool anything. Leaving this prefix at zero
    // makes the first selected-attention call pool the earlier raw keys too.
    if (pipeline) {
        cache.indexer_blocks = pipeline_blocks;
    } else if ((qsa != QSA_DENSE || (verify && rows.back().qsa != QSA_DENSE)) &&
        (pos0 + T) / qsa_ratio(w) > w.indexer_top_k / qsa_ratio(w)) {
        cache.indexer_blocks = (int) ((pos0 + T) / qsa_ratio(w));
    }

    out_logits.resize((size_t) ggml_nelements(logits));   // n_vocab, per row when verifying
    ggml_backend_tensor_get(logits, out_logits.data(), 0, ggml_nbytes(logits));
    if (out_hidden && hidden) {
        out_hidden->resize((size_t) ggml_nelements(hidden));
        ggml_backend_tensor_get(hidden, out_hidden->data(), 0, ggml_nbytes(hidden));
    }

    if (!pool) {
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(ctx);
    }

    res.ok = true;
    return res;
}

bool qwen4exp_verify_supported(const Qwen4ExpCache & cache) {
    // The PLE rollback snapshot covers one PLE layer (every published GGUF has one).
    return cache.mtp_k && !cache.spec_ssm.empty() && cache.ple_conv_state.size() <= 1;
}

bool qwen4exp_verify_rollback(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache,
                              int pos0, int retained) {
    const Qwen4ExpCudaScope profile(w.gfx1151);
    if (pos0 != cache.spec_pos || retained < 1 || retained > cache.spec_tokens ||
        cache.spec_ssm_rows.size() != cache.ssm_state.size() ||
        cache.spec_conv_rows.size() != cache.conv_state.size()) return false;
    // Device copies queued on the backend stream. All views were allocated at
    // cache creation; rollback never allocates or copies state through the host.
    const int row = retained - 1;
    if (retained < cache.spec_tokens) {
        for (size_t i = 0; i < cache.ssm_state.size(); ++i) {
            ggml_backend_tensor_copy_async(backend, backend, cache.spec_ssm_rows[i][row], cache.ssm_state[i]);
            ggml_backend_tensor_copy_async(backend, backend, cache.spec_conv_rows[i][row], cache.conv_state[i]);
        }
        if (cache.spec_ple) ggml_backend_tensor_copy_async(backend, backend, cache.spec_ple_rows[row], cache.ple_conv_state[0]);
    }
    cache.ple_prev = cache.spec_ple_prev[row];
    cache.cur_pos = pos0 + retained;
    cache.indexer_blocks = qwen4exp_mtp_retained_blocks(cache.indexer_blocks, cache.cur_pos, (int) qsa_ratio(w));
    // Stale KV/raw/pooled suffix rows are outside all logical views and will be
    // overwritten before becoming visible, including partially retained blocks.
    cache.spec_tokens = 0;
    return true;
}

namespace {

// One MTP batch on the target. A device draft rank replays a retained graph (rank 0 per catch-up width, later
// ranks per rank): the positions, the K/V rows and the final query's mask over a 256-token K/V bucket are its
// inputs, and it rebuilds only when the bucket changes. Draft ranks run asynchronously in stream order (the
// caller synchronizes once). Other batches rebuild per call (their K/V span grows) on a reused context and
// allocator, as does the window-limited session draft.
bool mtp_forward_batch(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache,
                       const int32_t * tokens, const float * hidden, int n, int pos0,
                       std::vector<float> & out_logits, std::vector<float> * out_hidden, bool kv_only,
                       bool last_only = false, int draft_rank = -1, Qwen4ExpGraphMemory * measure = nullptr) {
    const Qwen4ExpCudaScope profile(w.gfx1151);
    const int64_t H = w.n_embd, hc = w.n_hc, T = n, kv_len = pos0 + n;
    const bool device_draft = draft_rank >= 0, chain = draft_rank > 0;
    const bool stable = device_draft && cache.mtp_window == 0;
    const int64_t span = stable ? qwen4exp_kv_bucket(cache.max_ctx, kv_len, false) : kv_len;
    std::vector<float> emb(measure || chain ? 0 : (size_t) H * T);
    if (!measure && !chain && !w.embedder.embed(tokens, n, emb.data())) return false;

    Qwen4ExpDecodeWorkspace local_ws;
    Qwen4ExpDecodeWorkspace & ws = measure ? local_ws : !stable ? cache.mtp_workspace
                                 : chain ? cache.mtp_rank_workspace[draft_rank] : cache.mtp_catchup_workspace[n];
    if (stable && ws.ctx && (ws.backend != backend || ws.model != &w || ws.max_ctx != cache.max_ctx)) {
        clear_qwen4exp_decode_workspace(ws);
    }
    auto run = [&]() {
        std::vector<int32_t> ints((size_t) 5 * T, 0);   // M-RoPE section-major (section 3 stays zero), K/V rows
        for (int64_t i = 0; i < T; ++i) {
            ints[(size_t) i] = ints[(size_t) (T + i)] = ints[(size_t) (2 * T + i)] = ints[(size_t) (4 * T + i)] = (int32_t) (pos0 + i);
        }
        if (!chain) {
            ggml_backend_tensor_set(ws.inp_emb, emb.data(), 0, ggml_nbytes(ws.inp_emb));
            ggml_backend_tensor_set(ws.hidden_in, hidden, 0, ggml_nbytes(ws.hidden_in));
        }
        ggml_backend_tensor_set(ws.positions, ints.data(), 0, ggml_nbytes(ws.positions));
        if (ws.kv_row) ggml_backend_tensor_set(ws.kv_row, ints.data() + 4 * T, 0, ggml_nbytes(ws.kv_row));
        if (ws.mask) {
            // The final query sees [0, kv_len) of the bucket; a full batch is causal over kv_len.
            const int64_t cols = ws.mask->ne[0], rows = ws.mask->ne[1];
            std::vector<ggml_fp16_t> m((size_t) (cols * rows), ggml_fp32_to_fp16(-INFINITY));
            for (int64_t row = 0; row < rows; ++row) {
                std::fill_n(m.begin() + row * cols, rows == 1 ? kv_len : pos0 + row + 1, ggml_fp32_to_fp16(0.0f));
            }
            ggml_backend_tensor_set(ws.mask, m.data(), 0, ggml_nbytes(ws.mask));
        }
        return stable ? ggml_backend_graph_compute_async(backend, ws.gf) == GGML_STATUS_SUCCESS
                      : ggml_backend_graph_compute(backend, ws.gf) == GGML_STATUS_SUCCESS;
    };
    if (stable && !measure && ws.gf && ws.kv_bucket == span) {
        if (!run()) { clear_qwen4exp_decode_workspace(ws); return false; }
        ++ws.replays;
        return true;
    }

    if (ws.ctx == nullptr) {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false) + (1u << 16);
        ip.no_alloc = true;
        ws.ctx = ggml_init(ip);
        if (!ws.ctx) return false;
    } else {
        // A draft rank rebuilds the same topology at the same addresses for a new
        // bucket: its captures stay and are updated at the next capture.
        if (!stable) ggml_backend_cuda_graph_invalidate_range(backend, ggml_get_mem_buffer(ws.ctx), ggml_get_mem_size(ws.ctx));
        ggml_reset(ws.ctx);
    }
    ws.gf = nullptr;
    ws.backend = backend;
    ws.model = &w;
    ws.max_ctx = cache.max_ctx;
    ggml_context * ctx = ws.ctx;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    ws.inp_emb = chain ? ggml_get_rows(ctx, w.mtp_embd,
        ggml_view_1d(ctx, cache.mtp_chain_ids, 1, (draft_rank - 1) * sizeof(int32_t)))
        : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ws.hidden_in = chain ? cache.mtp_chain_hidden : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H * hc, T);
    ws.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * T);
    ws.kv_row = stable ? ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T) : nullptr;
    ws.mask = stable ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, span, 1)
            : !kv_only && !last_only && T > 1 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len, T) : nullptr;
    if (!chain) { ggml_set_input(ws.inp_emb); ggml_set_input(ws.hidden_in); }
    for (ggml_tensor * t : {ws.positions, ws.kv_row, ws.mask}) if (t) ggml_set_input(t);

    ggml_tensor * res = mtp_input(ctx, w, ws.inp_emb, ws.hidden_in);

    // One trunk-style layer (upstream-order HC helpers; dense attention on the draft layer's own K/V).
    const Qwen4ExpLayer & L = w.mtp;
    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = hc_mix(ctx, res, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject, &inject,
                               H, hc, w.rms_eps);
    // Decode still attends densely over the entire prompt; prefill only fills its K/V.
    cur = build_full_attn(ctx, gf, cur, L, w, cache.mtp_k, cache.mtp_v, /*indexer_k=*/nullptr, /*indexer_raw=*/nullptr,
                          ws.positions, ws.mask, ws.kv_row, span, pos0, /*ratio=*/4, /*n_pooled=*/0, QSA_DENSE,
                          nullptr, nullptr, kv_only, last_only, device_draft ? cache.mtp_window : 0);
    ggml_tensor * draft_hidden = nullptr, * logits = nullptr;
    if (!kv_only) {
        const int64_t output_T = last_only ? 1 : T;
        if (last_only && T > 1) {
            res = ggml_view_3d(ctx, res, H, hc, 1, res->nb[1], res->nb[2], (T - 1) * res->nb[2]);
            inject = ggml_view_2d(ctx, inject, hc, 1, inject->nb[1], (T - 1) * inject->nb[1]);
        }
        res = hc_combine(ctx, res, cur, inject, H, hc, output_T);
        cur = hc_mix(ctx, res, L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, &inject, H, hc, w.rms_eps);
        res = hc_combine(ctx, res, build_moe(ctx, cur, L, w), inject, H, hc, output_T);

        ggml_tensor * last = ggml_view_3d(ctx, res, H, hc, 1, res->nb[1], res->nb[2], (size_t) (output_T - 1) * res->nb[2]);
        draft_hidden = out_hidden ? ggml_cont(ctx, last) : nullptr;
        if (draft_hidden) { ggml_set_output(draft_hidden); ggml_build_forward_expand(gf, draft_hidden); }
        ggml_tensor * head = hc_mix(ctx, last, w.mtp_head_norm, w.mtp_head_down, w.mtp_head_up, nullptr, nullptr,
                                    H, hc, w.rms_eps);
        logits = ggml_mul_mat(ctx, device_draft ? w.mtp_output : w.output, head);
        if (device_draft) {
            ggml_tensor * best = ggml_argmax(ctx, logits);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, best,
                ggml_view_1d(ctx, cache.mtp_chain_ids, 1, draft_rank * sizeof(int32_t))));
            // All readers of the preceding rank's residual precede this copy.
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, last, H * hc, 1), cache.mtp_chain_hidden));
        } else {
            ggml_set_output(logits);
            ggml_build_forward_expand(gf, logits);
        }
    }

    if (measure) {
        graph_memory(backend, ctx, gf, w.gfx1151, *measure);
        measure->host = (size_t) T * (H * (hc + 1) * sizeof(float) + 4 * sizeof(int32_t)) +
                        (device_draft ? sizeof(int32_t) * cache.mtp_draft : (size_t) w.n_vocab * sizeof(float));
        ggml_free(ctx);
        return true;
    }
    if (ws.alloc == nullptr) ws.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    // Re-plan every build (the K/V views move), keeping the allocator's buffer.
    bool ok = ws.alloc && (!ws.planned || ggml_gallocr_reserve(ws.alloc, gf)) && ggml_gallocr_alloc_graph(ws.alloc, gf);
    ws.planned = ok;
    ws.gf = gf;
    ws.kv_bucket = span;
    ok = ok && run();
    if (ok) {
        if (stable) ++ws.builds;
        out_logits.resize(kv_only || device_draft ? 0 : (size_t) w.n_vocab);
        if (logits && !device_draft) ggml_backend_tensor_get(logits, out_logits.data(), 0, sizeof(float) * w.n_vocab);
        if (out_hidden) {
            out_hidden->resize((size_t) H * hc);
            ggml_backend_tensor_get(draft_hidden, out_hidden->data(), 0, out_hidden->size() * sizeof(float));
        }
    } else {
        clear_qwen4exp_decode_workspace(ws);
    }
    return ok;
}

}  // namespace

bool qwen4exp_mtp_forward(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache,
                          const int32_t * tokens, const float * hidden, int n, int pos0,
                          std::vector<float> & out_logits, std::vector<float> * out_hidden, bool kv_only, bool last_only) {
    if (!w.mtp_eh_proj || !cache.mtp_k || !tokens || !hidden || n <= 0 || pos0 < 0 || pos0 + n > cache.max_ctx ||
        (kv_only && (out_hidden || last_only))) {
        return false;
    }
    // Keep the original slices even for K/V-only prefill: matmul dispatch/numerics depend on batch width.
    constexpr int slice = 512;
    const size_t hd = (size_t) w.n_embd * w.n_hc;
    for (int i = 0; i < n; i += slice) {
        const int m = std::min(slice, n - i);
        if (!mtp_forward_batch(backend, w, cache, tokens + i, hidden + (size_t) i * hd, m, pos0 + i,
                               out_logits, i + m == n ? out_hidden : nullptr, kv_only, last_only)) {
            return false;
        }
    }
    return true;
}

bool qwen4exp_mtp_draft(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache,
                        const int32_t * tokens, const float * hidden, int n, int pos0, int k,
                        std::vector<int32_t> & drafts) {
    if (k < 1 || k > cache.mtp_draft || n < 1 || n > cache.mtp_draft + 1 || pos0 < 0 ||
        pos0 + n + k - 1 > cache.max_ctx || !tokens || !hidden || !w.mtp_output || !w.mtp_embd ||
        !cache.mtp_chain_hidden || !cache.mtp_chain_ids) return false;
    drafts.clear();
    std::vector<float> unused;
    // First rank replaces every retained authoritative K/V pair. Later ranks
    // read the preceding GPU argmax and HC residual directly. MTP has no PLE.
    for (int rank = 0; rank < k; ++rank) {
        if (!mtp_forward_batch(backend, w, cache, tokens, hidden, rank ? 1 : n,
                                rank ? pos0 + n + rank - 1 : pos0, unused, nullptr,
                                false, true, rank)) return false;
    }
    drafts.resize(k);
    ggml_backend_synchronize(backend);   // the ranks ran asynchronously
    ggml_backend_tensor_get(cache.mtp_chain_ids, drafts.data(), 0, k * sizeof(int32_t));
    for (int32_t & id : drafts) {
        if (id < 0 || (size_t) id >= w.mtp_vocab_ids.size()) return false;
        id = w.mtp_vocab_ids[id];
    }
    return true;
}

bool qwen4exp_mtp_catch_up(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache,
                           Qwen4ExpMtpPending & pending, size_t keep, int max_rows) {
    const size_t hd = (size_t) w.n_embd * w.n_hc;
    std::vector<float> unused;
    while (pending.tok.size() > keep) {
        const int n = (int) std::min(pending.tok.size() - keep, (size_t) std::max(1, max_rows));
        if (!qwen4exp_mtp_forward(backend, w, cache, pending.tok.data(), pending.h.data(), n, pending.pos,
                                  unused, nullptr, /*kv_only=*/true)) return false;
        pending.tok.erase(pending.tok.begin(), pending.tok.begin() + n);
        pending.h.erase(pending.h.begin(), pending.h.begin() + (std::ptrdiff_t) ((size_t) n * hd));
        pending.pos += n;
    }
    return true;
}

bool qwen4exp_mtp_step(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache, int pos,
                       int32_t token, int max_k, AdaptiveSpecWidth * width, Qwen4ExpMtpPending * pending,
                       const std::function<bool(int32_t, const float *, int32_t &)> & sample_row,
                       std::vector<float> & logits, Qwen4ExpMtpStep & step,
                       const std::function<void(std::vector<int32_t> &)> & on_drafts) {
    step = {};
    const size_t hd = (size_t) w.n_embd * w.n_hc;
    auto graph_builds = [&cache] {
        uint64_t n = 0;
        for (const auto & ws : cache.verify_workspace) n += ws.builds;
        for (const auto & ws : cache.mtp_catchup_workspace) n += ws.builds;
        for (const auto & ws : cache.mtp_rank_workspace) n += ws.builds;
        return n;
    };
    const int n_width = width ? width->next_width_cost_aware({}, cache.mtp_draft + 1) : cache.mtp_draft + 1;
    const int k = pending ? std::max(0, std::min(n_width - 1, max_k)) : 0;
    const bool verify = k > 0;
    step.k = k;
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t builds0 = verify ? graph_builds() : 0;
    std::vector<int32_t> drafts;
    if (verify) {
        // After a stretch of plain decode, fold all but the last pair first: a draft catches up at most a verify's.
        if (pending->tok.size() > (size_t) cache.mtp_draft + 1 && !qwen4exp_mtp_catch_up(backend, w, cache, *pending, 1)) {
            step.error = "qwen4exp MTP catch-up failed";
            return false;
        }
        if (!qwen4exp_mtp_draft(backend, w, cache, pending->tok.data(), pending->h.data(), (int) pending->tok.size(),
                                pending->pos, k, drafts)) {
            step.error = "qwen4exp MTP draft failed";
            return false;
        }
        if (on_drafts) on_drafts(drafts);
    }
    const auto t_draft = std::chrono::steady_clock::now();
    std::array<int32_t, QWEN4EXP_MTP_MAX_VERIFY> in{}, samples{};
    in[0] = token;
    std::copy(drafts.begin(), drafts.end(), in.begin() + 1);
    std::vector<float> hidden;
    if (!qwen4exp_forward(backend, w, cache, in.data(), k + 1, pos, logits, pending ? &hidden : nullptr, verify).ok) {
        step.error = "qwen4exp decode forward failed";
        return false;
    }
    step.draft_s = std::chrono::duration<double>(t_draft - t0).count();
    step.verify_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_draft).count();
    // Rows are sampled lazily, each after its own fed token: no draws for rows past a rejection.
    Qwen4ExpMtpAcceptance & decision = step.decision;
    for (int i = 0; i <= k; ++i) {
        step.more = sample_row(in[(size_t) i], logits.data() + (size_t) i * w.n_vocab, samples[(size_t) i]);
        decision = qwen4exp_mtp_accept(drafts.data(), k, samples.data(), i + 1);
        if (!step.more || decision.n_accepted != i + 1) break;
    }
    const int retained = decision.n_emitted;
    if (verify) {
        // Replace every predicted MTP hidden/KV row with the corresponding
        // trunk pair on next catch-up, including after a partial accept.
        pending->h.assign(hidden.begin(), hidden.begin() + (std::ptrdiff_t) ((size_t) retained * hd));
        pending->tok.assign(decision.emitted.begin(), decision.emitted.begin() + retained);
        pending->pos = pos;
        if (!qwen4exp_verify_rollback(backend, w, cache, pos, retained)) {
            step.error = "qwen4exp verify rollback failed";
            return false;
        }
    } else if (pending) {
        pending->h.insert(pending->h.end(), hidden.begin(), hidden.end());
        pending->tok.push_back(decision.emitted[(size_t) retained - 1]);
    }
    // Only a step that replayed its verify and draft graphs times the width; one
    // that built a graph is slower and counts for acceptance alone.
    if (verify && width) width->observe(decision.n_accepted + 1, k + 1, graph_builds() != builds0 ? -1.0f :
        (float) std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

Qwen4ExpForwardResult qwen4exp_forward(ggml_backend_t backend, const Qwen4ExpWeights & w,
        Qwen4ExpCache & cache, const int32_t * tokens, int n_tokens, int pos0,
        std::vector<float> & logits, std::vector<float> * out_hidden,
        bool verify, bool mtp_prefill, const Qwen4ExpInputs * inputs) {
    return forward_impl(backend, w, cache, tokens, n_tokens, pos0, logits, out_hidden,
                        verify, mtp_prefill, inputs, nullptr);
}

Qwen4ExpGraphMemory qwen4exp_graph_memory(ggml_backend_t backend, const Qwen4ExpWeights & w,
        Qwen4ExpCache & cache, int n_tokens, int pos0, bool verify) {
    Qwen4ExpGraphMemory memory;
    std::vector<float> unused;
    const int blocks = cache.indexer_blocks;
    const int64_t bucket_base = cache.kv_bucket_base;
    const bool mtp = qwen4exp_verify_supported(cache);
    const int64_t ratio = std::max<int64_t>(1, qsa_ratio(w));
    cache.indexer_blocks = pos0 / ratio <= w.indexer_top_k / ratio ? 0 : (int) (pos0 / ratio);
    const bool ok = forward_impl(backend, w, cache, nullptr, n_tokens, pos0, unused,
                                mtp ? &unused : nullptr, verify, mtp && n_tokens > 1 && !verify,
                                nullptr, &memory).ok;
    cache.indexer_blocks = blocks;
    cache.kv_bucket_base = bucket_base;
    if (!ok) memory.graph = SIZE_MAX;
    return memory;
}

Qwen4ExpGraphMemory qwen4exp_mtp_graph_memory(ggml_backend_t backend, const Qwen4ExpWeights & w,
        Qwen4ExpCache & cache, int n_tokens, int pos0) {
    Qwen4ExpGraphMemory memory;
    std::vector<float> unused;
    if (!mtp_forward_batch(backend, w, cache, nullptr, nullptr, n_tokens, pos0,
                           unused, nullptr, false, true, 0, &memory)) memory.graph = SIZE_MAX;
    return memory;
}

bool qwen4exp_can_batch(const Qwen4ExpWeights & w,
        const Qwen4ExpForwardSegment * segments, int n_segments, bool use_qsa, bool qsa_rows) {
    if (n_segments < 1 || n_segments > kQwen4ExpMaxBatchedSlots) return false;
    for (int s = 0; s < n_segments; ++s) {
        const auto & segment = segments[s];
        if (segment.n_tokens != 1) return false;
        const Qwen4ExpQsaMode mode = qsa_mode(w, *segment.cache, 1, segment.pos0, use_qsa);
        if (mode != QSA_DENSE && !(qsa_rows && mode == QSA_DECODE)) return false;
    }
    return true;
}

static Qwen4ExpForwardResult forward_sequential(ggml_backend_t backend,
        const Qwen4ExpWeights & w, const Qwen4ExpForwardSegment * segments, int n_segments,
        std::vector<std::vector<float>> & out_logits, int hidden_slot = -1, std::vector<float> * out_hidden = nullptr) {
    Qwen4ExpForwardResult result;
    out_logits.resize((size_t) n_segments);
    for (int s = 0; s < n_segments; ++s) {
        const auto & segment = segments[s];
        if (!qwen4exp_forward(backend, w, *segment.cache, segment.tokens,
                segment.n_tokens, segment.pos0, out_logits[s], s == hidden_slot ? out_hidden : nullptr).ok) {
            out_logits.clear();
            return result;
        }
    }
    result.ok = true;
    return result;
}

// A slot of the multi-slot graph attends densely over spans[s] > 0 keys behind a mask (one 256-token window past
// its kv_len, as a stable T=1 graph's first span), or, with spans[s] < 0, like a stable T=1 QSA decode over a K/V
// bucket of -spans[s].
static int64_t batched_span(const Qwen4ExpCache & cache, int32_t pos, bool qsa) {
    const int64_t kv_len = (int64_t) pos + 1;
    return qsa ? -qwen4exp_kv_bucket(cache.max_ctx, kv_len, true) : qwen4exp_kv_bucket(cache.max_ctx, kv_len + 256, false);
}

// The multi-slot decode graph's nodes and inputs, in `ctx` and `ws`; nothing is allocated. Each slot's rows run the
// layer pieces a solo decode step runs (Qwen4ExpLayerBuilder), with the attention, the recurrence and PLE's
// convolution against the slot's own cache.
static bool build_batched_decode_nodes(ggml_context * ctx, const Qwen4ExpWeights & w,
        Qwen4ExpCache * const * caches, int n_slots, bool has_ple, bool use_qsa,
        const std::vector<int64_t> & spans, bool with_hidden, Qwen4ExpBatchedDecodeWorkspace & ws) {
    const bool split = w.expert_backend != nullptr;
    const int64_t T = n_slots;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, kQwen4ExpGraphNodes, false);
    if (!gf) return false;
    ws.expert_nodes.clear();
    Qwen4ExpLayerBuilder lb(ctx, gf, w, *caches[0], /*verify=*/false, split, has_ple, ws.expert_nodes);
    ws.inp_emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, T);
    ggml_set_input(ws.inp_emb);
    ws.ple_in = nullptr;
    if (has_ple) {
        ws.ple_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.ple_head_dim * w.ple_n_heads, T);
        ggml_set_input(ws.ple_in);
    }
    ws.masks.clear();
    ws.kv_rows.clear();
    // Slots whose recurrent states are slabs of one Qwen4ExpSlotStates.
    const Qwen4ExpSlotStates * slot_states = caches[0]->slot_states;
    for (int s = 1; s < n_slots; ++s) if (caches[s]->slot_states != slot_states) slot_states = nullptr;
    ws.slot_ids = nullptr;
    if (slot_states) {
        ws.slot_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        ggml_set_input(ws.slot_ids);
    }
    // QSA slots: each one's runtime inputs (a stable T=1 graph's, see Qwen4ExpDecodeWorkspace::qsa_params) and M-RoPE
    // position are views of three shared inputs, so a split scheduler still sees few graph inputs.
    ws.slot_qsa.assign((size_t) n_slots, {});
    std::vector<const Qwen4ExpDecodeWorkspace *> qsa((size_t) n_slots, nullptr);
    int64_t qsa_blocks = 0;
    for (const int64_t span : spans) if (span < 0) qsa_blocks += -span / 4;
    ws.qsa_params = ws.qsa_visibility = ws.qsa_positions = nullptr;
    if (qsa_blocks > 0) {
        ws.qsa_params = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 10 * T);
        ws.qsa_visibility = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, qsa_blocks);
        ws.qsa_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * T);
        ggml_set_input(ws.qsa_params);
        ggml_set_input(ws.qsa_visibility);
        ggml_set_input(ws.qsa_positions);
        int64_t offset = 0;
        for (int s = 0; s < n_slots; ++s) {
            if (spans[(size_t) s] >= 0) continue;
            Qwen4ExpDecodeWorkspace & slot = ws.slot_qsa[(size_t) s];
            slot.kv_bucket = -spans[(size_t) s];
            slot.qsa_blocks = slot.kv_bucket / 4;
            slot.qsa_params = ggml_view_1d(ctx, ws.qsa_params, 10, (size_t) 10 * s * sizeof(int32_t));
            slot.qsa_visibility = ggml_view_1d(ctx, ws.qsa_visibility, slot.qsa_blocks, (size_t) offset * sizeof(float));
            slot.positions = ggml_view_1d(ctx, ws.qsa_positions, 4, (size_t) 4 * s * sizeof(int32_t));
            offset += slot.qsa_blocks;
            qsa[(size_t) s] = &slot;
        }
    }

    // Split mode: the slots form two groups whose layers interleave in the graph like a prompt chunk's streams (see
    // interleave_layers). The matmuls are batch-invariant, so a row computes the same in either group.
    struct SlotGroup {
        int s0 = 0, n = 0;
        ggml_tensor * res_hc = nullptr, * xn_next = nullptr, * inject = nullptr;
        ggml_tensor * positions = nullptr, * ids = nullptr, * routed = nullptr;
        Qwen4ExpMoeRoute route;
    };
    const int n_groups = split ? 2 : 1;
    std::vector<SlotGroup> groups((size_t) n_groups);
    for (int g = 0; g < n_groups; ++g) {
        SlotGroup & G = groups[(size_t) g];
        G.s0 = g == 0 ? 0 : (n_slots + 1) / 2;
        G.n = n_groups == 1 ? n_slots : g == 0 ? (n_slots + 1) / 2 : n_slots / 2;
        ggml_tensor * emb = ggml_view_2d(ctx, ws.inp_emb, w.n_embd, G.n, ws.inp_emb->nb[1],
                                         (size_t) G.s0 * ws.inp_emb->nb[1]);
        G.res_hc = repeat_dim1(ctx, ggml_reshape_3d(ctx, emb, w.n_embd, 1, G.n), w.n_hc);
        if (ws.slot_ids) G.ids = ggml_view_1d(ctx, ws.slot_ids, G.n, (size_t) G.s0 * ws.slot_ids->nb[0]);
    }

    // A group's attention half on the target, and its FFN: the router and shared expert when the routed experts run
    // on the expert device, else the whole FFN half.
    auto stage_r = [&](SlotGroup & G, int il) {
        const Qwen4ExpLayer & L = w.layers[il];
        const int64_t n = G.n;
        if (L.is_ple && has_ple) {
            // PLE table projection weights are read once for the group's rows;
            // only the stateful convolution windows branch per sequence.
            ggml_tensor * ple = ggml_view_2d(ctx, ws.ple_in, ws.ple_in->ne[0], n, ws.ple_in->nb[1],
                                             (size_t) G.s0 * ws.ple_in->nb[1]);
            ggml_tensor * key = mm(ctx, L.ple_key, ple);
            ggml_tensor * value = mm(ctx, L.ple_value, ple);
            ggml_tensor * rows = nullptr;
            for (int j = 0; j < n; ++j) {
                ggml_tensor * hidden = ggml_view_3d(ctx, G.res_hc, w.n_embd, w.n_hc, 1,
                    G.res_hc->nb[1], G.res_hc->nb[2], (size_t) j * G.res_hc->nb[2]);
                ggml_tensor * row = build_ple_row(ctx, gf, hidden,
                    column(ctx, key, j), column(ctx, value, j), L, w,
                    caches[G.s0 + j]->ple_conv_state[(size_t) lb.ple_idx[il]]);
                rows = rows ? ggml_concat(ctx, rows, row, 2) : row;
            }
            G.res_hc = rows;
            G.xn_next = nullptr;
        }
        ggml_tensor * cur = lb.attn_input(il, G.res_hc, G.xn_next, G.inject);

        if (L.is_full_attention) {
            ggml_tensor * qfull = mm(ctx, L.wq, cur);
            ggml_tensor * kraw = mm(ctx, L.wk, cur);
            ggml_tensor * vraw = mm(ctx, L.wv, cur);
            if (ws.kv_rows.empty()) {
                // One mask and K/V row per slot and one M-RoPE position input per group, shared by
                // every attention layer (a split scheduler tracks at most GGML_SCHED_MAX_SPLIT_INPUTS).
                for (int s = 0; s < n_slots; ++s) {
                    const int64_t span = spans[(size_t) s];
                    ws.masks.push_back(span > 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, span, 1) : nullptr);
                    ws.kv_rows.push_back(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1));
                    if (ws.masks.back()) ggml_set_input(ws.masks.back());
                    ggml_set_input(ws.kv_rows.back());
                }
            }
            if (!G.positions) {
                G.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n);
                ggml_set_input(G.positions);
            }
            cur = mm(ctx, L.wo, build_full_attn_slots(ctx, gf, qfull, kraw, vraw, cur, G.positions,
                ws.masks.data() + G.s0, ws.kv_rows.data() + G.s0, spans.data() + G.s0, L, w,
                caches + G.s0, lb.full_idx[il], use_qsa, qsa.data() + G.s0));
        } else {
            // Dense input projections are batched; with stacked slot states the conv and
            // recurrence run once for the group, else each slot's sees one row and its own state.
            const int li = lb.lin_idx[il];
            ggml_tensor * qkv = mm(ctx, L.attn_qkv, cur);
            ggml_tensor * z = mm(ctx, L.attn_gate, cur);
            ggml_tensor * beta = mm(ctx, L.ssm_beta, cur);
            ggml_tensor * alpha = mm(ctx, L.ssm_alpha, cur);
            ggml_tensor * rows = nullptr;
            if (G.ids) {
                rows = build_linear_attn_slots(ctx, gf, qkv, z, beta, alpha, L, w,
                    slot_states->ssm[(size_t) li], slot_states->conv[(size_t) li], G.ids);
            }
            for (int j = 0; !G.ids && j < n; ++j) {
                ggml_tensor * row = build_linear_attn_projected(ctx, gf,
                    column(ctx, qkv, j), column(ctx, z, j), column(ctx, beta, j),
                    column(ctx, alpha, j), L, w,
                    caches[G.s0 + j]->ssm_state[(size_t) li], caches[G.s0 + j]->conv_state[(size_t) li]);
                rows = rows ? ggml_concat(ctx, rows, row, 1) : row;
            }
            cur = mm(ctx, L.ssm_out, rows);
        }
        if (!split || !qwen4exp_split_layer(L, w)) {
            G.res_hc = lb.ffn_half(il, cur, G.inject, G.res_hc, n, G.xn_next);
            return;
        }
        cur = lb.ffn_input(il, cur, G.inject, G.res_hc, n);
        G.route = build_moe_route(ctx, cur, L, w);
        for (ggml_tensor * t : { G.route.xin, G.route.sel, G.route.wsel, G.route.shared, G.inject, G.res_hc }) {
            ggml_build_forward_expand(gf, t);
        }
    };
    // The group's routed experts, on the expert device.
    auto stage_s = [&](SlotGroup & G, int il) {
        if (!G.route.sel) return;
        G.routed = build_moe_routed(ctx, G.route, w.layers[il], w, ws.expert_nodes);
        ggml_build_forward_expand(gf, G.routed);
    };
    // The MoE output joins the residual, normed for the next layer, on the target.
    auto stage_c = [&](SlotGroup & G, int il) {
        if (!G.route.sel) return;
        G.res_hc = lb.ffn_output(il, moe_join(ctx, G.routed, G.route.shared), G.inject, G.res_hc, G.n, G.xn_next);
        G.route = {};
        if (G.xn_next) ggml_build_forward_expand(gf, G.xn_next);
        ggml_build_forward_expand(gf, G.res_hc);
    };
    if (n_groups > 1) {
        interleave_layers(w.n_layer, groups, stage_r, stage_s, stage_c);
    } else {
        for (int il = 0; il < w.n_layer; ++il) stage_r(groups[0], il);
    }

    ggml_tensor * xn_next = nullptr, * res_hc = nullptr;
    for (const SlotGroup & G : groups) {
        xn_next = !G.xn_next ? nullptr : xn_next ? ggml_concat(ctx, xn_next, G.xn_next, 2) : G.xn_next;
        res_hc = res_hc ? ggml_concat(ctx, res_hc, G.res_hc, 2) : G.res_hc;
    }
    ws.logits = ggml_mul_mat(ctx, w.output, lb.output_mix(res_hc, xn_next));
    ggml_set_output(ws.logits);
    ggml_build_forward_expand(gf, ws.logits);
    // Appended after the logits, which keep their node order: the slots' final HC residuals.
    ws.hidden = nullptr;
    if (with_hidden) {
        ws.hidden = ggml_cont(ctx, res_hc);
        ggml_set_output(ws.hidden);
        ggml_build_forward_expand(gf, ws.hidden);
    }
    ws.groups.clear();
    for (const SlotGroup & G : groups) ws.groups.push_back({G.positions, G.s0, G.n});
    ws.gf = gf;
    return true;
}

// Builds and allocates the multi-slot decode graph into `ws`; the caller uploads its inputs and runs it. Like a verify
// width's graph, it keeps its own split scheduler beside it.
static bool build_batched_decode_graph(ggml_backend_t backend, const Qwen4ExpWeights & w,
        Qwen4ExpCache * const * caches, int n_slots, bool has_ple, bool use_qsa,
        const std::vector<int64_t> & spans, bool with_hidden, Qwen4ExpBatchedDecodeWorkspace & ws) {
    ggml_context * ctx = graph_context(&ws, backend, w, caches[0]->max_ctx, /*verify=*/true);
    if (!ctx || !build_batched_decode_nodes(ctx, w, caches, n_slots, has_ple, use_qsa, spans, with_hidden, ws)) {
        return false;
    }
    int64_t kv_len = 0;
    for (const int64_t span : spans) kv_len = std::max<int64_t>(kv_len, std::abs(span));
    ggml_backend_sched_t sched = nullptr;
    ggml_gallocr_t galloc = nullptr;
    if (!allocate_graph(backend, w, *caches[0], &ws, /*verify=*/true, ws.gf, ws.expert_nodes, n_slots, kv_len,
                        sched, galloc)) {
        ws.gf = nullptr;
        return false;
    }
    ws.caches.assign(caches, caches + n_slots);
    ws.spans = spans;
    return true;
}

Qwen4ExpGraphMemory qwen4exp_batched_graph_memory(ggml_backend_t backend, const Qwen4ExpWeights & w,
        Qwen4ExpCache & cache, int n_slots, int pos, bool qsa) {
    Qwen4ExpGraphMemory memory;
    memory.graph = SIZE_MAX;
    if (n_slots < 2 || n_slots > kQwen4ExpMaxBatchedSlots || pos < 0 || pos >= cache.max_ctx) return memory;
    const Qwen4ExpCudaScope profile(w.gfx1151 || w.expert_gfx1151);
    ggml_context * ctx = graph_context(nullptr, backend, w, cache.max_ctx, false);
    if (!ctx) return memory;
    // An allocation plan only: the slots share one cache and nothing runs.
    std::vector<Qwen4ExpCache *> caches((size_t) n_slots, &cache);
    Qwen4ExpBatchedDecodeWorkspace plan;
    if (build_batched_decode_nodes(ctx, w, caches.data(), n_slots,
            w.ple_reader.available() && !cache.ple_layer_ids.empty(), w.qsa,
            std::vector<int64_t>((size_t) n_slots, batched_span(cache, pos, qsa)), /*with_hidden=*/true, plan)) {
        memory.graph = 0;
        graph_memory(backend, ctx, plan.gf, w.gfx1151, memory);
    }
    ggml_free(ctx);
    return memory;
}

Qwen4ExpForwardResult qwen4exp_forward_batched(
        ggml_backend_t backend, const Qwen4ExpWeights & w,
        Qwen4ExpCache * const * caches, const int32_t * tokens,
        const int32_t * positions, int n_slots,
        Qwen4ExpBatchedDecodeWorkspace & workspace,
        std::vector<std::vector<float>> & out_logits, int hidden_slot, std::vector<float> * out_hidden) {
    Qwen4ExpForwardResult result;
    if (!backend || !caches || !tokens || !positions || n_slots <= 0) return result;
    if (hidden_slot < 0 || hidden_slot >= n_slots) out_hidden = nullptr;
    const bool split = w.expert_backend != nullptr;
    const Qwen4ExpCudaScope profile(w.gfx1151 || w.expert_gfx1151);
    out_logits.clear();

    // Preserve the established single-sequence arithmetic and state transitions
    // exactly. The new graph is only used for true multi-slot calls.
    if (n_slots == 1) {
        if (!caches[0]) return result;
        std::vector<float> logits;
        result = qwen4exp_forward(backend, w, *caches[0], tokens, 1, positions[0], logits, out_hidden);
        if (result.ok) {
            out_logits.assign(1, std::move(logits));
        }
        return result;
    }
    std::vector<Qwen4ExpForwardSegment> segments((size_t) n_slots);
    for (int s = 0; s < n_slots; ++s) {
        if (!caches[s] || positions[s] < 0 || positions[s] >= caches[s]->max_ctx ||
            caches[s]->linear_layer_ids != caches[0]->linear_layer_ids ||
            caches[s]->full_layer_ids != caches[0]->full_layer_ids ||
            caches[s]->ple_layer_ids != caches[0]->ple_layer_ids) return result;
        for (int prev = 0; prev < s; ++prev) if (caches[prev] == caches[s]) return result;
        segments[s] = {caches[s], tokens + s, 1, positions[s]};
    }
    // A slot past the QSA boundary attends as a stable T=1 QSA decode inside the graph
    // when its backend serves that graph; otherwise every slot runs its solo forward.
    std::vector<bool> qsa_slot((size_t) n_slots, false);
    bool qsa_rows = true;
    for (int s = 0; s < n_slots; ++s) {
        qsa_slot[(size_t) s] = qsa_mode(w, *caches[s], 1, positions[s], w.qsa) == QSA_DECODE;
        if (qsa_slot[(size_t) s] && !stable_qsa_supported(backend, w, *caches[s], (int64_t) positions[s] + 1))
            qsa_rows = false;
    }
    if (!qwen4exp_can_batch(w, segments.data(), n_slots, w.qsa, qsa_rows))
        return forward_sequential(backend, w, segments.data(), n_slots, out_logits, hidden_slot, out_hidden);

    const bool has_ple = w.ple_reader.available() && !caches[0]->ple_layer_ids.empty();
    const int64_t ple_heads = w.ple_n_heads;
    const int64_t ple_row_size = w.ple_head_dim * ple_heads;

    std::vector<float> emb((size_t) w.n_embd * n_slots);
    if (!w.embedder.embed(tokens, n_slots, emb.data())) {
        std::fprintf(stderr, "[qwen4exp] batched CPU embedding failed\n");
        return result;
    }

    std::vector<float> ple_data(has_ple ? (size_t) ple_row_size * n_slots : 0);
    std::vector<std::vector<int32_t>> next_prev((size_t) n_slots);
    if (has_ple) {
        const int64_t ng = w.ple_ngram_size;
        std::vector<int32_t> ple_rows((size_t) ple_heads * n_slots);
        for (int s = 0; s < n_slots; ++s) {
            const std::vector<int32_t> & prev = caches[s]->ple_prev;
            std::vector<int32_t> seq = prev;
            seq.push_back(tokens[s]);
            const int64_t pos = (int64_t) prev.size();
            std::vector<uint64_t> ctx((size_t) ng);
            ctx[0] = (uint64_t) tokens[s];
            bool cut = false;
            for (int64_t j = 1; j < ng; ++j) {
                if (cut || pos - j < 0) { ctx[j] = (uint64_t) w.ple_eos_token_id; cut = true; }
                else {
                    const int32_t id = seq[(size_t) (pos - j)];
                    if (id < 0 || id == w.ple_eos_token_id) cut = true;
                    ctx[j] = cut ? (uint64_t) w.ple_eos_token_id : (uint64_t) id;
                }
            }
            for (int64_t n = 2; n <= ng; ++n) {
                uint64_t mixed = ctx[0] * w.ple_layer_multipliers[0];
                for (int64_t j = 1; j < n; ++j) mixed ^= ctx[j] * w.ple_layer_multipliers[(size_t) j];
                const int64_t base = (n - 2) * w.ple_heads_per_ngram;
                for (int64_t q = 0; q < w.ple_heads_per_ngram; ++q) {
                    const int64_t h = base + q;
                    ple_rows[(size_t) s * ple_heads + h] = (int32_t)
                        ((mixed % (uint64_t) w.ple_head_vocab_sizes[h]) + w.ple_head_offsets[h]);
                }
            }
            const size_t keep = (size_t) std::min<int64_t>(ng - 1, prev.size() + 1);
            const size_t total = prev.size() + 1;
            for (size_t k = total - keep; k < total; ++k)
                next_prev[s].push_back(k < prev.size() ? prev[k] : tokens[s]);
        }
        if (!w.ple_reader.gather(ple_rows.data(), (int64_t) ple_rows.size(), ple_data.data())) {
            std::fprintf(stderr, "[qwen4exp] batched PLE gather failed\n");
            return result;
        }
    }

    std::vector<int64_t> spans((size_t) n_slots);
    for (int s = 0; s < n_slots; ++s) {
        // A QSA slot's graph pools as it goes, from the already-pooled prefix.
        if (qsa_slot[(size_t) s] && caches[s]->indexer_blocks < positions[s] / 4 &&
            !bootstrap_qsa_prefix(backend, w, *caches[s], positions[s])) return result;
        spans[(size_t) s] = batched_span(*caches[s], positions[s], qsa_slot[(size_t) s]);
    }
    const bool with_hidden = out_hidden != nullptr;
    const bool replay = workspace.gf && !stable_workspace_stale(workspace, backend, w, caches[0]->max_ctx) &&
        workspace.spans == spans && (workspace.hidden != nullptr) == with_hidden &&
        workspace.caches == std::vector<const Qwen4ExpCache *>(caches, caches + n_slots);
    if (!replay && !build_batched_decode_graph(backend, w, caches, n_slots, has_ple, w.qsa,
                                               spans, with_hidden, workspace)) return result;
    ++(replay ? workspace.replays : workspace.builds);

    ggml_backend_tensor_set(workspace.inp_emb, emb.data(), 0, emb.size() * sizeof(float));
    if (workspace.ple_in) ggml_backend_tensor_set(workspace.ple_in, ple_data.data(), 0, ple_data.size() * sizeof(float));
    for (const auto & G : workspace.groups) {
        if (!G.positions) continue;
        std::vector<int32_t> p((size_t) 4 * G.n, 0);
        for (int j = 0; j < G.n; ++j)
            for (int section = 0; section < 3; ++section) p[(size_t) section * G.n + j] = positions[G.s0 + j];
        ggml_backend_tensor_set(G.positions, p.data(), 0, p.size() * sizeof(int32_t));
    }
    if (workspace.slot_ids) {
        std::vector<int32_t> ids((size_t) n_slots);
        for (int s = 0; s < n_slots; ++s) ids[(size_t) s] = caches[s]->state_slot;
        ggml_backend_tensor_set(workspace.slot_ids, ids.data(), 0, ids.size() * sizeof(int32_t));
    }
    for (size_t s = 0; s < workspace.kv_rows.size(); ++s) {
        ggml_backend_tensor_set(workspace.kv_rows[s], &positions[s], 0, sizeof(int32_t));
        if (!workspace.masks[s]) continue;
        std::vector<ggml_fp16_t> mask((size_t) spans[s], ggml_fp32_to_fp16(-INFINITY));
        std::fill(mask.begin(), mask.begin() + positions[s] + 1, ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(workspace.masks[s], mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }
    // Each QSA slot's inputs, as its stable T=1 graph uploads them. The host copies outlive the uploads.
    std::vector<Qwen4ExpStableInputs> qsa_inputs((size_t) n_slots);
    if (workspace.qsa_positions) {
        std::vector<int32_t> qsa_pos((size_t) 4 * n_slots, 0);
        for (int s = 0; s < n_slots; ++s) {
            if (spans[(size_t) s] >= 0) continue;
            Qwen4ExpStableInputs & in = qsa_inputs[(size_t) s];
            in.backend = backend;
            in.split = split;
            in.qsa = true;
            in.pos0 = positions[s];
            in.T = 1;
            in.kv_len = (int64_t) positions[s] + 1;
            in.max_ctx = caches[s]->max_ctx;
            upload_stable_qsa(in, workspace.slot_qsa[(size_t) s]);
            qsa_pos[(size_t) 4 * s] = qsa_pos[(size_t) 4 * s + 1] = qsa_pos[(size_t) 4 * s + 2] = positions[s];
        }
        ggml_backend_tensor_set(workspace.qsa_positions, qsa_pos.data(), 0, qsa_pos.size() * sizeof(int32_t));
    }
    // Match solo MMVQ arithmetic regardless of the number of active slots.
    const ScopedCudaGraphOverrides overrides(false, 0, false, 0, /*mmvq_batch_invariant=*/true);
    const ggml_status status = split ? ggml_backend_sched_graph_compute(workspace.sched, workspace.gf)
                                     : ggml_backend_graph_compute(backend, workspace.gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[qwen4exp] batched graph compute failed\n");
        clear_qwen4exp_batched_decode_workspace(workspace);
        return result;
    }
    std::vector<float> packed((size_t) w.n_vocab * n_slots);
    ggml_backend_tensor_get(workspace.logits, packed.data(), 0, packed.size() * sizeof(float));
    out_logits.resize((size_t) n_slots);
    for (int s = 0; s < n_slots; ++s) {
        out_logits[(size_t) s].assign(packed.begin() + (size_t) s * w.n_vocab,
                                      packed.begin() + (size_t) (s + 1) * w.n_vocab);
    }
    if (out_hidden) {
        const size_t hd = (size_t) w.n_embd * w.n_hc;
        if ((size_t) ggml_nelements(workspace.hidden) != hd * (size_t) n_slots) {
            out_hidden->clear();
        } else {
            out_hidden->resize(hd);
            ggml_backend_tensor_get(workspace.hidden, out_hidden->data(), (size_t) hidden_slot * hd * sizeof(float),
                                    hd * sizeof(float));
        }
    }
    if (has_ple) {
        for (int s = 0; s < n_slots; ++s) caches[s]->ple_prev = std::move(next_prev[s]);
    }
    for (int s = 0; s < n_slots; ++s) {
        if (qsa_slot[(size_t) s]) caches[s]->indexer_blocks = (positions[s] + 1) / 4;
    }
    result.ok = true;
    return result;
}

}  // namespace luce::common
