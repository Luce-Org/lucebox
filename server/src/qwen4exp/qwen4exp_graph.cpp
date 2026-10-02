// Qwen4Exp forward graph — see qwen4exp_graph.h.
// embed -> hc_init -> per-layer ([PLE], hc_mix(attn) -> linear|QSA -> hc_combine,
// hc_mix(ffn) -> MoE -> hc_combine) -> hc_mix(output) -> lm_head. Single sequence.

#include "qwen4exp_graph.h"

#include "delta_net_chunked.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace luce::common {
namespace {

static double prof_now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// QWEN4EXP_PROF=1: per-call phase timings, forward-call timestamps and graph batch widths.
static bool prof_enabled() {
    static const bool on = std::getenv("QWEN4EXP_PROF") != nullptr;
    return on;
}

// F16 activation paths for gfx1151 MMB prefill (gated-norm tail, attention gate, MoE fold). Default on; QWEN4EXP_F16=0
// is the kill switch. Off under QWEN4EXP_UPSTREAM (byte-exact reference) and QWEN4EXP_DUMP (per-node dumps).
static bool f16_paths() {
    static const bool on = [] {
        const char * u = getenv("QWEN4EXP_UPSTREAM"), * f = getenv("QWEN4EXP_F16");
        return !(u && std::atoi(u) != 0) && getenv("QWEN4EXP_DUMP") == nullptr && !(f && std::atoi(f) == 0);
    }();
    return on;
}

static double mono_now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void trace_batch_widths(ggml_cgraph * gf, const char * phase,
                               int rows, int max_kv) {
    if (!prof_enabled() || !gf) return;

    std::map<int, int> matmul_widths, mmid_widths;
    std::map<std::string, int> state_ops;
    std::vector<std::string> width_one;
    int matmul_count = 0, mmid_count = 0, max_matmul_width = 0, dense_t1_count = 0;
    const int node_count = ggml_graph_n_nodes(gf);
    for (int i = 0; i < node_count; ++i) {
        const ggml_tensor * node = ggml_graph_node(gf, i);
        if (!node) continue;
        if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) &&
            node->src[0] && node->src[1]) {
            const bool is_id = node->op == GGML_OP_MUL_MAT_ID;
            const int width = is_id
                ? (int) node->src[1]->ne[2]
                : node->src[1]->ne[0] > 0
                    ? (int) (ggml_nelements(node->src[1]) / node->src[1]->ne[0])
                    : 0;
            if (is_id) {
                ++mmid_count;
                ++mmid_widths[width];
            } else {
                ++matmul_count;
                ++matmul_widths[width];
                max_matmul_width = std::max(max_matmul_width, width);
                if (width == 1) {
                    ++dense_t1_count;
                    if (width_one.size() < 12) width_one.emplace_back(
                        std::string(node->src[0]->name) +
                        "[" + std::to_string(node->src[0]->ne[0]) + "x" +
                        std::to_string(node->src[0]->ne[1]) + "]");
                }
            }
        } else if (node->op == GGML_OP_FLASH_ATTN_EXT ||
                   node->op == GGML_OP_GATED_DELTA_NET ||
                   node->op == GGML_OP_SSM_CONV) {
            ++state_ops[ggml_op_name(node->op)];
        }
    }
    auto widths = [](const std::map<int, int> & hist) {
        std::string out;
        for (const auto & [width, count] : hist) {
            if (!out.empty()) out += ",";
            out += std::to_string(width) + ":" + std::to_string(count);
        }
        return out.empty() ? std::string("-") : out;
    };
    std::string state;
    for (const auto & [op, count] : state_ops) {
        if (!state.empty()) state += ",";
        state += op + ":" + std::to_string(count);
    }
    std::string t1;
    for (const std::string & name : width_one) {
        if (!t1.empty()) t1 += ",";
        t1 += name;
    }
    static uint64_t call = 0;
    std::fprintf(stderr,
        "[qwen4exp-batch] call=%llu phase=%s rows=%d max_kv=%d nodes=%d "
        "MUL_MAT=%d{width:count:%s} MMID=%d{width:count:%s} "
        "max_dense_T=%d state={%s} dense_T1=%d names=%d[%s]\n",
        (unsigned long long) ++call, phase, rows, max_kv, node_count,
        matmul_count, widths(matmul_widths).c_str(), mmid_count,
        widths(mmid_widths).c_str(), max_matmul_width, state.c_str(),
        dense_t1_count, (int) width_one.size(), t1.empty() ? "-" : t1.c_str());
}

size_t ring_align_up(size_t value) {
    const size_t remainder = value % 256;
    return remainder == 0 ? value : value + 256 - remainder;
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

[[maybe_unused]] ggml_tensor * hc_mix(ggml_context * c, ggml_tensor * x, ggml_tensor * w_norm,
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

[[maybe_unused]] ggml_tensor * hc_combine(ggml_context * c,
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

[[maybe_unused]] ggml_tensor * build_moe(ggml_context * c, ggml_tensor * cur,
                        const Qwen4ExpLayer & L, const Qwen4ExpWeights & w, int il,
                        const std::function<void(ggml_tensor *, const char *)> & dump_mark = {},
                        Qwen4ExpMoeParts * parts = nullptr) {
    const int64_t n_embd   = w.n_embd;
    const int64_t n_tokens = cur->ne[1];
    const int64_t n_expert = w.n_expert;
    const int64_t n_used   = w.n_expert_used;

    char dlab[32];
    auto dmark = [&](ggml_tensor * t, const char * tag) {
        if (dump_mark && t) { std::snprintf(dlab, sizeof dlab, "L%02d.%s", il, tag); dump_mark(t, dlab); }
    };

    ggml_tensor * logits = mm(c, L.ffn_gate_inp, cur);      // [n_expert, T]
    dmark(logits, "mlogit");
    ggml_tensor * probs  = ggml_soft_max(c, logits);
    dmark(probs, "mprob");
    ggml_tensor * sel    = ggml_argsort_top_k(c, probs, (int) n_used);  // [n_used, T]
    dmark(sel, "mid");
    dmark(logits, "rlogit");

    ggml_tensor * probs3 = ggml_reshape_3d(c, probs, 1, n_expert, n_tokens);
    ggml_tensor * wsel   = ggml_reshape_2d(c, ggml_get_rows(c, probs3, sel), n_used, n_tokens);
    wsel = ggml_div(c, wsel, ggml_clamp(c, ggml_sum_rows(c, wsel), 6.103515625e-5f, INFINITY));
    dmark(wsel, "mwt");

    ggml_tensor * cur3 = ggml_reshape_3d(c, cur, n_embd, 1, n_tokens);

    ggml_tensor * gate = ggml_mul_mat_id(c, L.ffn_gate_exps, cur3, sel);
    dmark(gate, "mgate");
    ggml_tensor * up   = ggml_mul_mat_id(c, L.ffn_up_exps,   cur3, sel);
    dmark(up, "mup");
    ggml_tensor * gu   = ggml_swiglu_split(c, gate, up);
    dmark(gu, "mgu");

    ggml_tensor * down = ggml_mul_mat_id(c, L.ffn_down_exps, gu, sel);   // [n_embd, n_used, T]
    dmark(down, "mdown");

    ggml_tensor * sh_gate = mm(c, L.ffn_gate_shexp, cur);
    ggml_tensor * sh_up   = mm(c, L.ffn_up_shexp, cur);
    ggml_tensor * sh_gu   = ggml_swiglu_split(c, sh_gate, sh_up);
    ggml_tensor * shared  = mm(c, L.ffn_down_shexp, sh_gu);

    ggml_tensor * shared_logit = mm(c, L.ffn_gate_inp_shexp, cur);
    if (parts) {   // the caller folds the combine into the next HC_COMBINE_NORM
        *parts = { down, wsel, shared, shared_logit };
        return nullptr;
    }
    ggml_tensor * shared_gate = ggml_sigmoid(c, shared_logit);
    shared = ggml_mul(c, shared, shared_gate);   // [n_embd,T] * [1,T] broadcasts over dim 0
    dmark(shared, "msh");

    ggml_tensor * moe_out;
    // Keep the unfused form for upstream differential checks.
    static const bool moe_fused = [] {
        const char * v = getenv("QWEN4EXP_UPSTREAM");
        return !(v && std::atoi(v) != 0);
    }();
    if (!moe_fused) {
        // Upstream aggregate: weight every route (broadcast mul), then sequential
        // per-route view adds in argsort order, then add the gated shared expert.
        ggml_tensor * wexp = ggml_mul(c, down, ggml_reshape_3d(c, wsel, 1, n_used, n_tokens));
        ggml_tensor * acc = ggml_view_3d(c, wexp, n_embd, 1, n_tokens,
                                         wexp->nb[1], wexp->nb[2], 0);
        for (int64_t i = 1; i < n_used; ++i) {
            acc = ggml_add(c, acc, ggml_view_3d(c, wexp, n_embd, 1, n_tokens,
                                                wexp->nb[1], wexp->nb[2], i * wexp->nb[1]));
        }
        moe_out = ggml_add(c, ggml_reshape_2d(c, acc, n_embd, n_tokens), shared);
    } else {
        moe_out = ggml_ds4_moe_fused_combine_shared(c, down, wsel, shared);
    }
    dmark(moe_out, "mout");
    return moe_out;
}

// ── Linear attention: gated delta net (36 layers) ───────────────────────

ggml_tensor * build_linear_attn(ggml_context * c, ggml_cgraph * gf, ggml_tensor * cur,
                                const Qwen4ExpLayer & L, const Qwen4ExpWeights & w,
                                ggml_tensor * ssm_state, ggml_tensor * conv_state, int il,
                                const std::function<void(ggml_tensor *, const char *)> & dump_mark = {}) {
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

    ggml_tensor * beta = ggml_sigmoid(c,
        ggml_reshape_4d(c, mm(c, L.ssm_beta, cur), 1, Hv, T, 1));
    ggml_tensor * alpha = ggml_reshape_3d(c, mm(c, L.ssm_alpha, cur), Hv, T, 1);
    alpha = ggml_softplus(c, ggml_add(c, alpha, L.ssm_dt_bias));
    ggml_tensor * gate = ggml_reshape_4d(c, ggml_mul(c, alpha, L.ssm_a), 1, Hv, T, 1);

    ggml_tensor * hist = ggml_reshape_3d(c, conv_state, kernel - 1, conv_channels, 1);
    // Keep the transpose as a view: the fused concat+transpose kernel keys off src1->nb[1] == sizeof(float).
    ggml_tensor * qkv_t = ggml_transpose(c, ggml_reshape_2d(c, qkv, conv_channels, T));
    ggml_tensor * conv_input = ggml_concat(c, hist, qkv_t, 0);

    // nb[0] is the element size, so the tail offset is T*nb[0], NOT T*nb[1].
    ggml_tensor * new_hist = ggml_cont(c, ggml_view_3d(c, conv_input, kernel - 1, conv_channels, 1,
        conv_input->nb[1], conv_input->nb[2], (size_t) T * conv_input->nb[0]));
    ggml_build_forward_expand(gf, ggml_cpy(c, new_hist,
        ggml_reshape_3d(c, conv_state, kernel - 1, conv_channels, 1)));

    ggml_tensor * conv_op = ggml_ssm_conv(c, conv_input, L.ssm_conv1d);
    // The gfx1151 fusion reads CONCAT's input at this later node. src[3] is
    // unused by ordinary SSM_CONV and gives the allocator the real lifetime.
    conv_op->src[3] = qkv_t;
    ggml_tensor * conv = ggml_silu(c, conv_op);
    if (dump_mark) {
        char dlab[32];
        std::snprintf(dlab, sizeof dlab, "L%02d.conv", il);
        dump_mark(conv, dlab);
    }

    const size_t esz     = ggml_element_size(conv);
    const size_t tstride = (size_t) conv_channels * esz;
    ggml_tensor * q_raw = ggml_view_3d(c, conv, D, Hk, T, D * esz, tstride, 0);
    ggml_tensor * k_raw = ggml_view_3d(c, conv, D, Hk, T, D * esz, tstride, D * Hk * esz);
    // Upstream build_gdn_l2_norm is rms_norm(x, eps/n) * (1/sqrt(n)) == x/sqrt(sum(x^2)+eps).
    // ggml_l2_norm instead rounds x*rsqrtf(max(sum(x^2), eps^2)), a ~1e-6 relative difference
    // that seeds the GDN recurrence and flips MoE routing.
    ggml_tensor * q_c = ggml_scale(c, ggml_rms_norm(c, q_raw, eps / (float) D), 1.0f / sqrtf((float) D));
    ggml_tensor * k_c = ggml_scale(c, ggml_rms_norm(c, k_raw, eps / (float) D), 1.0f / sqrtf((float) D));
    ggml_tensor * v_c = ggml_view_3d(c, conv, D, Hv, T, D * esz, tstride, 2 * D * Hk * esz);

    ggml_tensor * state4 = ggml_reshape_4d(c, ssm_state, D, D, Hv, 1);
    ggml_tensor * gdn = ggml_gated_delta_net(c, q_c, k_c, v_c, gate, beta, state4);
    // Only speculative rollback needs per-token intermediate states; skipping keeps the packed result allocatable.
    ggml_gated_delta_net_set_skip_intermediate(gdn, true);

    // packed: [ attn S_v*H_v*T | final_state S_v*S_v*H_v ]
    ggml_tensor * attn = ggml_view_4d(c, gdn, D, Hv, T, 1,
        ggml_row_size(gdn->type, D),
        ggml_row_size(gdn->type, D * Hv),
        ggml_row_size(gdn->type, D * Hv * T), 0);
    ggml_tensor * new_state = ggml_view_4d(c, gdn, D, D, Hv, 1,
        ggml_row_size(gdn->type, D),
        ggml_row_size(gdn->type, D * D),
        ggml_row_size(gdn->type, D * D * Hv),
        ggml_row_size(gdn->type, D * Hv * T));
    ggml_build_forward_expand(gf, ggml_cpy(c, new_state, state4));

    // Gated norm written as F16 in one pass, read directly by ssm_out's Q8_0 -> F16 GEMM (same arithmetic as the
    // chain below, so bit-exact).
    if (f16_paths() && ggml_backend_cuda_mmb_f16_input_ok(L.ssm_out, T)) {
        ggml_tensor * lin_raw = mm(c, L.ssm_out, ggml_gated_rms_norm_f16(c, attn, L.ssm_norm, z, eps));
        return ggml_reshape_2d(c, lin_raw, w.n_embd, T);
    }
    ggml_tensor * normed = ggml_mul(c, ggml_rms_norm(c, attn, eps), L.ssm_norm);
    ggml_tensor * out = ggml_mul(c, normed, ggml_sigmoid(c, ggml_reshape_4d(c, z, D, Hv, T, 1)));
    if (dump_mark) {
        char dlab[32];
        std::snprintf(dlab, sizeof dlab, "L%02d.gnorm", il);
        dump_mark(out, dlab);
    }

    ggml_tensor * final = ggml_reshape_3d(c, out, d_in, T, 1);
    // lin is a reshape view; dump the contiguous GEMM output so the buffer is output-protected.
    ggml_tensor * lin_raw = mm(c, L.ssm_out, final);
    if (dump_mark) {
        char dlab[32];
        std::snprintf(dlab, sizeof dlab, "L%02d.lin", il);
        dump_mark(lin_raw, dlab);
    }
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

namespace {

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
        fresh = ggml_mul(c, ggml_rms_norm(c, fresh, w.rms_eps), L.indexer_k_norm);
        int sections[4] = { w.rope_sections[0], w.rope_sections[1], w.rope_sections[2], w.rope_sections[3] };
        ggml_tensor * bp = ggml_scale(c, ggml_arange(c, (float) n_pooled, (float) n_after, 1.0f), (float) r);
        ggml_tensor * bp_i = ggml_cast(c, bp, GGML_TYPE_I32);
        ggml_tensor * bp_z = ggml_cast(c, ggml_scale(c, bp, 0.0f), GGML_TYPE_I32);
        ggml_tensor * bpos = ggml_concat(c, ggml_concat(c, bp_i, bp_i, 0), ggml_concat(c, bp_i, bp_z, 0), 0);
        // rope indexes its tokens on ne[2], so put the block axis there.
        fresh = ggml_rope_multi(c, ggml_reshape_3d(c, fresh, idim, 1, n_new), bpos, nullptr,
            w.rope_dimension_count, sections, GGML_ROPE_TYPE_MROPE, 0, w.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        fresh = ggml_reshape_2d(c, fresh, idim, n_new);
        ggml_build_forward_expand(gf, ggml_cpy(c, fresh,
            ggml_view_2d(c, indexer_k, idim, n_new, indexer_k->nb[1], (size_t) n_pooled * indexer_k->nb[1])));
    }
    ggml_tensor * prefix = n_pooled > 0 ? ggml_view_2d(c, indexer_k, idim, n_pooled, indexer_k->nb[1], 0) : nullptr;
    ggml_tensor * all = prefix && fresh ? ggml_concat(c, prefix, fresh, 1) : (prefix ? prefix : fresh);
    return ggml_reshape_3d(c, all, idim, n_after, 1);
}

// Retain the original float graph outside the exact integer domain of ratio-4 QSA.
static ggml_tensor * qsa_cell_ids(ggml_context * c, ggml_tensor * blocks, ggml_tensor * positions,
        int64_t r, int64_t kv_start, int64_t nb) {
    const int64_t budget = blocks->ne[0];
    const int64_t T = blocks->ne[1];
    if (r == 4 && budget <= 1024 && nb < (1 << 22) && kv_start + T < (1 << 24)) {
        return ggml_qsa_decode_ids(c, blocks, ggml_view_1d(c, positions, T, 0), (int) r);
    }
    blocks = ggml_cont(c, blocks);
    ggml_tensor * order = ggml_argsort(c, ggml_cast(c, blocks, GGML_TYPE_F32), GGML_SORT_ORDER_ASC);
    blocks = ggml_reshape_3d(c, ggml_get_rows(c,
        ggml_view_4d(c, blocks, 1, budget, T, 1, blocks->nb[0], blocks->nb[1], blocks->nb[2], 0),
        order), budget, T, 1);

    ggml_tensor * shape3 = ggml_new_tensor_3d(c, GGML_TYPE_F32, r, budget, T);
    ggml_tensor * tv = ggml_repeat(c,
        ggml_reshape_3d(c, ggml_scale_bias(c, ggml_arange(c, 0.0f, (float) T, 1.0f), 1.0f, (float) kv_start),
                        1, 1, T), shape3);
    ggml_tensor * bf = ggml_cast(c, blocks, GGML_TYPE_F32);                        // [budget,T,1]
    ggml_tensor * bf3 = ggml_repeat(c, ggml_reshape_3d(c, bf, 1, budget, T), shape3);
    ggml_tensor * lim = ggml_scale_bias(c, ggml_scale(c, bf3, (float) r), 1.0f, (float) (r - 1));
    ggml_tensor * valid = ggml_step(c, ggml_scale_bias(c, ggml_sub(c, tv, lim), 1.0f, 1.0f));

    ggml_tensor * iv  = ggml_repeat(c, ggml_reshape_3d(c,
        ggml_arange(c, 0.0f, (float) r, 1.0f), r, 1, 1), shape3);
    ggml_tensor * cf  = ggml_scale_bias(c, ggml_add(c, ggml_scale(c, bf3, (float) r), iv), 1.0f, 1.0f);
    cf = ggml_scale_bias(c, ggml_mul(c, cf, valid), 1.0f, -1.0f);                  // (r*b+i+1)*valid - 1
    ggml_tensor * cells = ggml_reshape_4d(c, ggml_cast(c, cf, GGML_TYPE_I32), budget * r, T, 1, 1);

    ggml_tensor * tvt = ggml_scale_bias(c, ggml_arange(c, 0.0f, (float) T, 1.0f), 1.0f, (float) kv_start);
    ggml_tensor * br  = ggml_scale(c, ggml_floor(c, ggml_scale(c,
        ggml_scale_bias(c, ggml_arange(c, 1.0f, (float) (T + 1), 1.0f), 1.0f, (float) kv_start),
        1.0f / (float) r)), (float) r);
    ggml_tensor * rows = nullptr;
    for (int64_t i = 0; i < r - 1; ++i) {
        ggml_tensor * cell = ggml_scale_bias(c, br, 1.0f, (float) i);
        ggml_tensor * v = ggml_step(c, ggml_scale_bias(c, ggml_sub(c, tvt, cell), 1.0f, 1.0f));
        ggml_tensor * val = ggml_scale_bias(c, ggml_mul(c, v, ggml_scale_bias(c, cell, 1.0f, 1.0f)), 1.0f, -1.0f);
        ggml_tensor * row = ggml_reshape_2d(c, ggml_cast(c, val, GGML_TYPE_I32), 1, T);
        rows = rows ? ggml_concat(c, rows, row, 0) : row;
    }
    return ggml_reshape_2d(c,
        ggml_concat(c, cells, ggml_reshape_3d(c, rows, r - 1, T, 1), 0),
        budget * r + (r - 1), T);
}

static ggml_tensor * build_qsa_attn(ggml_context * c, ggml_tensor * cur,
        ggml_tensor * Q, ggml_tensor * Kf, ggml_tensor * Vf,
        const Qwen4ExpLayer & L, const Qwen4ExpWeights & w, int64_t ratio,
        ggml_tensor * positions, int64_t kv_pad, int64_t kv_len, int64_t kv_start,
        ggml_tensor * pooled, int64_t nb, bool packed) {
    const int64_t idim   = w.indexer_head_size;
    const int64_t nih    = w.indexer_n_head;
    const int64_t r      = ratio;
    const int64_t T      = cur->ne[1];
    const int64_t budget = w.indexer_top_k / r;
    const float   eps    = w.rms_eps;
    const float   qscale = 1.0f / std::sqrt((float) w.n_embd_head_k);
    int sections[4] = { w.rope_sections[0], w.rope_sections[1], w.rope_sections[2], w.rope_sections[3] };

    ggml_tensor * qi = mm(c, L.indexer_q_proj, cur);              // [idim*nih, T]
    qi = ggml_reshape_3d(c, qi, idim, nih, T);
    qi = ggml_mul(c, ggml_rms_norm(c, qi, eps), L.indexer_q_norm);
    qi = ggml_rope_multi(c, qi, positions, nullptr, w.rope_dimension_count, sections,
        GGML_ROPE_TYPE_MROPE, 0, w.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    if (!ggml_is_contiguous(qi)) qi = ggml_cont(c, qi);

    ggml_tensor * comp16 = ggml_cast(c,
        ggml_reshape_2d(c, ggml_cont(c, pooled), idim, nb), GGML_TYPE_F16);
    ggml_tensor * hw = ggml_reshape_2d(c,
        ggml_scale_bias(c, ggml_scale(c, ggml_arange(c, 0.0f, (float) (nih * T), 1.0f), 0.0f), 0.0f, 1.0f),
        nih, T);
    ggml_tensor * summed = ggml_ds4_indexer_score(c, qi, hw, comp16, (int) kv_start, (int) r);

    ggml_tensor * blocks = ggml_top_k(c, summed, (int) budget);   // Keep selection and its tie behavior unchanged.
    ggml_tensor * ids = qsa_cell_ids(c, blocks, positions, r, kv_start, nb);

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
    ggml_flash_attn_ext_set_n_kv_max(attn, (int32_t) ids->ne[0]);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    return attn;
}

static bool qsa_enabled() {
    static const bool on = [] { const char * v = getenv("QWEN4EXP_QSA"); return v && std::atoi(v) != 0; }();
    return on;
}

// QSA block ratio of the full-attention layers (0 when the model has no indexer).
static int64_t qsa_ratio(const Qwen4ExpWeights & w) {
    for (int il = 0; il < w.n_layer && il < (int) w.compress_ratios.size(); ++il) {
        if (w.layers[il].is_full_attention && w.compress_ratios[il] > 1) return w.compress_ratios[il];
    }
    return 0;
}

enum Qwen4ExpQsaMode { QSA_DENSE = 0, QSA_PREFILL = 1, QSA_DECODE = 2 };

// Selected attention is exactly dense while every query sees at most `budget` complete blocks, so it only runs past
// that point: the packed prefill kernel for T >= 128, the per-query decode kernel below that. CUDA builds and the
// upstream reference stay dense.
static Qwen4ExpQsaMode qsa_mode(const Qwen4ExpWeights & w, const Qwen4ExpCache & cache, int64_t T, int64_t pos0,
                                bool upstream) {
    const int64_t r = qsa_ratio(w);
    if (!qsa_enabled() || upstream || r <= 1 || cache.indexer_raw.empty() || !cache.indexer_raw[0]) return QSA_DENSE;
    if (w.indexer_head_size != 128 || w.indexer_n_head <= 0 || w.indexer_top_k % r != 0) return QSA_DENSE;
    const int64_t budget = w.indexer_top_k / r;
    if (budget * r + (r - 1) > 2560 || w.n_head != 12 * w.n_head_kv) return QSA_DENSE;
    if ((pos0 + T) / r <= budget) return QSA_DENSE;
    if (T < 128) return QSA_DECODE;
    const int64_t step = (r % 4 == 0) ? r : (r % 2 == 0 ? r * 2 : r * 4);
    const int64_t kv_pad = (pos0 + T + step - 1) / step * step;
    return kv_pad <= cache.attn_k[0]->ne[1] ? QSA_PREFILL : QSA_DENSE;
}

ggml_tensor * build_full_attn(ggml_context * c, ggml_cgraph * gf, ggml_tensor * cur,
                              const Qwen4ExpLayer & L, const Qwen4ExpWeights & w,
                              ggml_tensor * k_cache, ggml_tensor * v_cache,
                              ggml_tensor * indexer_k, ggml_tensor * indexer_raw,
                              ggml_tensor * positions, ggml_tensor * mask, ggml_tensor * kv_row,
                              int64_t kv_len, int64_t pos0, int64_t ratio,
                              int64_t n_pooled, Qwen4ExpQsaMode qsa, int il,
                              const std::function<void(ggml_tensor *, const char *)> & dump_mark = {}) {
    const int64_t D      = w.n_embd_head_k;   // 256
    const int64_t Hq     = w.n_head;          // 24
    const int64_t Hk     = w.n_head_kv;       // 2
    const int64_t T      = cur->ne[1];
    const float   eps    = w.rms_eps;

    if (dump_mark) {
        char dlab[32];
        std::snprintf(dlab, sizeof dlab, "L%02d.cur", il);
        dump_mark(cur, dlab);
    }

    // wq holds [q | gate] interleaved per head
    ggml_tensor * qfull = mm(c, L.wq, cur);   // [2*D*Hq, T]
    const size_t  qe    = ggml_element_size(qfull);
    ggml_tensor * Q = ggml_rms_norm(c,
        ggml_view_3d(c, qfull, D, Hq, T, 2 * D * qe, 2 * D * Hq * qe, 0), eps);
    Q = ggml_mul(c, Q, L.q_norm);
    ggml_tensor * gate = ggml_cont_2d(c,
        ggml_view_3d(c, qfull, D, Hq, T, 2 * D * qe, 2 * D * Hq * qe, D * qe), D * Hq, T);

    ggml_tensor * K = ggml_rms_norm(c,
        ggml_reshape_3d(c, mm(c, L.wk, cur), D, Hk, T), eps);
    K = ggml_mul(c, K, L.k_norm);
    // V is a view; dump the contiguous pre-reshape GEMM so the buffer is output-protected.
    ggml_tensor * Vraw = mm(c, L.wv, cur);
    ggml_tensor * V    = ggml_reshape_3d(c, Vraw, D, Hk, T);

    if (dump_mark) {
        char dlab[32];
        std::snprintf(dlab, sizeof dlab, "L%02d.qnorm", il);
        dump_mark(Q, dlab);
        std::snprintf(dlab, sizeof dlab, "L%02d.knorm", il);
        dump_mark(K, dlab);
        std::snprintf(dlab, sizeof dlab, "L%02d.vraw", il);
        dump_mark(Vraw, dlab);
    }

    int sections[4] = { w.rope_sections[0], w.rope_sections[1],
                        w.rope_sections[2], w.rope_sections[3] };
    Q = ggml_rope_multi(c, Q, positions, nullptr, w.rope_dimension_count, sections,
                        GGML_ROPE_TYPE_MROPE, 0, w.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_multi(c, K, positions, nullptr, w.rope_dimension_count, sections,
                        GGML_ROPE_TYPE_MROPE, 0, w.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    if (dump_mark) {
        char dlab[32];
        std::snprintf(dlab, sizeof dlab, "L%02d.Q", il);
        dump_mark(Q, dlab);
        std::snprintf(dlab, sizeof dlab, "L%02d.K", il);
        dump_mark(K, dlab);
    }

    if (kv_row) {
        // Graph-stable append: the destination and graph topology stay fixed;
        // only the device input row changes between decode steps.
        ggml_tensor * Krows = ggml_cont(c, ggml_permute(c, K, 0, 2, 1, 3));
        ggml_tensor * Vrows = ggml_cont(c, ggml_permute(c, V, 0, 2, 1, 3));
        ggml_build_forward_expand(gf, ggml_set_rows(c, k_cache, Krows, kv_row));
        ggml_build_forward_expand(gf, ggml_set_rows(c, v_cache, Vrows, kv_row));
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

    ggml_tensor * K_full = ggml_view_3d(c, k_cache, D, kv_len, Hk,
        k_cache->nb[1], k_cache->nb[2], 0);
    ggml_tensor * V_full = ggml_view_3d(c, v_cache, D, kv_len, Hk,
        v_cache->nb[1], v_cache->nb[2], 0);

    // Every token's raw indexer key goes to the cache, so blocks can be pooled whenever QSA first needs them.
    ggml_tensor * kraw = nullptr;
    if (indexer_raw) {
        kraw = mm(c, L.indexer_k_proj, cur);   // [idim, T]
        ggml_build_forward_expand(gf, kv_row
            ? ggml_set_rows(c, indexer_raw, kraw, kv_row)
            : ggml_cpy(c, kraw, ggml_view_2d(c, indexer_raw, kraw->ne[0], T, indexer_raw->nb[1],
                                             (size_t) pos0 * indexer_raw->nb[1])));
    }
    // qwen4exp_forward elides the dense causal mask only when every full layer takes QSA; a dense fallback without one must fail loudly.
    if (T > 1 && mask == nullptr && qsa != QSA_PREFILL) {
        std::fprintf(stderr,
            "[qwen4exp] dense attention without a causal mask (T=%lld pos0=%lld)\n",
            (long long) T, (long long) pos0);
        std::abort();
    }

    ggml_tensor * attn;
    if (qsa != QSA_DENSE) {
        GGML_ASSERT(kraw && indexer_k && ratio > 1);
        const int64_t n_after = (pos0 + T) / ratio;   // complete blocks visible to the last query
        ggml_tensor * pooled = qsa_pooled_keys(c, gf, L, w, indexer_k, indexer_raw, kraw, pos0, ratio, n_pooled, n_after);
        if (qsa == QSA_PREFILL) {
            // Pad K/V to a multiple of lcm(4, ratio) for the packed prefill kernel.
            const int64_t qsa_step = (ratio % 4 == 0) ? ratio : (ratio % 2 == 0 ? ratio * 2 : ratio * 4);
            const int64_t kv_pad   = (kv_len + qsa_step - 1) / qsa_step * qsa_step;
            ggml_tensor * K_pad = (kv_pad != kv_len)
                ? ggml_view_3d(c, k_cache, D, kv_pad, Hk, k_cache->nb[1], k_cache->nb[2], 0) : K_full;
            ggml_tensor * V_pad = (kv_pad != kv_len)
                ? ggml_view_3d(c, v_cache, D, kv_pad, Hk, v_cache->nb[1], v_cache->nb[2], 0) : V_full;
            attn = build_qsa_attn(c, cur, Q, K_pad, V_pad, L, w, ratio, positions,
                                  kv_pad, kv_len, pos0, pooled, n_after, true);
        } else {
            attn = build_qsa_attn(c, cur, Q, K_full, V_full, L, w, ratio, positions,
                                  kv_len, kv_len, pos0, pooled, n_after, false);
        }
        if (dump_mark) {   // the last query's selected cells
            ggml_tensor * ids = attn->src[5];
            char dlab[32];
            std::snprintf(dlab, sizeof dlab, "L%02d.qsaids", il);
            dump_mark(ggml_view_1d(c, ids, ids->ne[0], (size_t) (T - 1) * ids->nb[1]), dlab);
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

    if (dump_mark) {
        char dlab[32];
        std::snprintf(dlab, sizeof dlab, "L%02d.faout", il);
        dump_mark(attn, dlab);
        std::snprintf(dlab, sizeof dlab, "L%02d.gate", il);
        dump_mark(gate, dlab);
    }

    // sigmoid(gate) * attn written as F16 in one pass straight from the wq view (no CONT), read directly by wo's
    // Q8_0 -> F16 GEMM. Same product as below, so bit-exact.
    if (f16_paths() && ggml_backend_cuda_mmb_f16_input_ok(L.wo, T)) {
        ggml_tensor * gate3 = ggml_view_3d(c, qfull, D, Hq, T, 2 * D * qe, 2 * D * Hq * qe, D * qe);
        return mm(c, L.wo, ggml_gated_f16(c, attn, gate3));
    }
    ggml_tensor * attn2 = ggml_reshape_2d(c, attn, D * Hq, T);
    attn2 = ggml_mul(c, attn2, ggml_sigmoid(c, gate));
    if (dump_mark) {
        char dlab[32];
        std::snprintf(dlab, sizeof dlab, "L%02d.gated", il);
        dump_mark(attn2, dlab);
    }
    return mm(c, L.wo, attn2);
}

// ── Per-layer n-gram embedding (PLE) ────────────────────────────────────

ggml_tensor * build_ple(ggml_context * c, ggml_cgraph * gf, ggml_tensor * hidden,
                        ggml_tensor * ple_emb, const Qwen4ExpLayer & L,
                        const Qwen4ExpWeights & w, ggml_tensor * ple_conv_state,
                        const std::function<void(ggml_tensor *, const char *)> & dump_mark) {
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

    dump_mark(ple_emb, "ple.inp");
    dump_mark(key, "ple.key");
    dump_mark(value, "ple.value");
    key = grouped_norm(key, L.ple_norm_key);
    dump_mark(key, "ple.knorm");
    ggml_tensor * query = grouped_norm(hidden, L.ple_norm_query);
    dump_mark(query, "ple.qnorm");

    ggml_tensor * s = ggml_sum_rows(c, ggml_mul(c, key, query));   // [1, hc, T]
    s = ggml_scale(c, s, 1.0f / std::sqrt((float) n_embd));
    ggml_tensor * mag = ggml_sqrt(c, ggml_clamp(c, ggml_abs(c, s), 1e-6f, INFINITY));
    ggml_tensor * ple_gate = ggml_sigmoid(c, ggml_mul(c, ggml_sgn(c, s), mag));

    ggml_tensor * v3 = repeat_dim1(c,
        ggml_reshape_3d(c, value, n_embd, 1, T), hc);
    dump_mark(ple_gate, "ple.gate");
    ggml_tensor * gated = ggml_mul(c, v3, ple_gate);
    dump_mark(gated, "ple.gated");

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

    dump_mark(conv_out, "ple.conv");
    ggml_tensor * ple_out = ggml_add(c, hidden, ggml_add(c, gated, conv_out));
    dump_mark(ple_out, "ple.out");
    // Expand now so the conv matcher sees a contiguous concat -> state -> taps -> silu subgraph.
    ggml_build_forward_expand(gf, ple_out);
    return ple_out;
}

static ggml_tensor * column_span(ggml_context * c, ggml_tensor * x,
                                 int start, int count) {
    GGML_ASSERT(x && start >= 0 && count > 0 && start + count <= x->ne[1]);
    return ggml_view_2d(c, x, x->ne[0], count, x->nb[1],
                        (size_t) start * x->nb[1]);
}

static ggml_tensor * token_span_3d(ggml_context * c, ggml_tensor * x,
                                   int start, int count) {
    GGML_ASSERT(x && start >= 0 && count > 0 && start + count <= x->ne[2]);
    return ggml_view_3d(c, x, x->ne[0], x->ne[1], count,
                        x->nb[1], x->nb[2], (size_t) start * x->nb[2]);
}

static ggml_tensor * build_ple_projected(ggml_context * c, ggml_cgraph * gf,
        ggml_tensor * hidden, ggml_tensor * key, ggml_tensor * value,
        const Qwen4ExpLayer & L, const Qwen4ExpWeights & w,
        ggml_tensor * ple_conv_state) {
    const int64_t n_embd = w.n_embd, hc = w.n_hc, hc_dim = n_embd * hc;
    const int64_t T = hidden->ne[2];
    const float eps = w.rms_eps;
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * nw) {
        ggml_tensor * t = ggml_reshape_3d(c, x, n_embd, hc, T);
        t = ggml_rms_norm(c, t, eps);
        t = ggml_reshape_2d(c, t, hc_dim, T);
        t = ggml_mul(c, t, nw);
        return ggml_reshape_3d(c, t, n_embd, hc, T);
    };

    key = grouped_norm(key, L.ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, L.ple_norm_query);
    ggml_tensor * score = ggml_scale(c, ggml_sum_rows(c, ggml_mul(c, key, query)),
                                     1.0f / std::sqrt((float) n_embd));
    ggml_tensor * mag = ggml_sqrt(c,
        ggml_clamp(c, ggml_abs(c, score), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(c, ggml_mul(c, ggml_sgn(c, score), mag));
    ggml_tensor * repeated_value = repeat_dim1(c,
        ggml_reshape_3d(c, value, n_embd, 1, T), hc);
    ggml_tensor * gated = ggml_mul(c, repeated_value, gate);
    ggml_tensor * normalized = ggml_reshape_2d(c,
        grouped_norm(ggml_reshape_2d(c, gated, hc_dim, T), L.ple_norm_conv),
        hc_dim, T);

    const int64_t kern = w.ple_conv_kernel, dil = w.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;
    ggml_tensor * norm_t = ggml_transpose(c, ggml_reshape_2d(c, normalized, hc_dim, T));
    ggml_tensor * state = ggml_cont(c,
        ggml_reshape_3d(c, ple_conv_state, hist, hc_dim, 1));
    ggml_tensor * padded = ggml_concat(c, state, norm_t, 0);
    ggml_build_forward_expand(gf, ggml_cpy(c,
        ggml_cont(c, ggml_view_3d(c, padded, hist, hc_dim, 1,
            padded->nb[1], padded->nb[2], (size_t) T * padded->nb[0])),
        ggml_reshape_3d(c, ple_conv_state, hist, hc_dim, 1)));

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        const int64_t start = hist - (kern - 1 - k) * dil;
        ggml_tensor * shifted = ggml_cont(c, ggml_transpose(c,
            ggml_view_3d(c, padded, T, hc_dim, 1, padded->nb[1],
                         padded->nb[2], ggml_row_size(padded->type, start))));
        if (k == 0) shifted->src[1] = norm_t;
        ggml_tensor * wk = ggml_cont(c,
            ggml_view_2d(c, L.ple_conv1d, 1, hc_dim, L.ple_conv1d->nb[1],
                         k * L.ple_conv1d->nb[0]));
        wk = ggml_reshape_1d(c, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) wk = ggml_cast(c, wk, GGML_TYPE_F32);
        ggml_tensor * term = ggml_mul(c, shifted, wk);
        conv_out = conv_out ? ggml_add(c, conv_out, term) : term;
    }
    conv_out = ggml_reshape_3d(c, ggml_cont(c, ggml_silu(c, conv_out)),
                               n_embd, hc, T);
    ggml_tensor * out = ggml_add(c, hidden, ggml_add(c, gated, conv_out));
    ggml_build_forward_expand(gf, out);
    return out;
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

static ggml_tensor * build_linear_attn_projected_sequence(
        ggml_context * c, ggml_cgraph * gf, ggml_tensor * qkv,
        ggml_tensor * z, ggml_tensor * beta, ggml_tensor * alpha,
        const Qwen4ExpLayer & L, const Qwen4ExpWeights & w,
        ggml_tensor * ssm_state, ggml_tensor * conv_state) {
    const int64_t D = w.ssm_d_state, Hk = w.ssm_n_group;
    const int64_t Hv = w.linear_value_heads, d_in = w.ssm_d_inner;
    const int64_t T = qkv->ne[1], kernel = w.ssm_d_conv;
    const int64_t conv_channels = 2 * Hk * D + d_in;
    const float eps = w.rms_eps;

    beta = ggml_sigmoid(c, ggml_reshape_4d(c, beta, 1, Hv, T, 1));
    alpha = ggml_reshape_3d(c, alpha, Hv, T, 1);
    alpha = ggml_softplus(c, ggml_add(c, alpha, L.ssm_dt_bias));
    ggml_tensor * gate = ggml_reshape_4d(c,
        ggml_mul(c, alpha, L.ssm_a), 1, Hv, T, 1);

    ggml_tensor * hist = ggml_reshape_3d(c, conv_state, kernel - 1, conv_channels, 1);
    ggml_tensor * qkv_t = ggml_transpose(c, qkv);
    ggml_tensor * conv_input = ggml_concat(c, hist, qkv_t, 0);
    ggml_tensor * new_hist = ggml_cont(c, ggml_view_3d(c, conv_input,
        kernel - 1, conv_channels, 1, conv_input->nb[1], conv_input->nb[2],
        (size_t) T * conv_input->nb[0]));
    ggml_build_forward_expand(gf, ggml_cpy(c, new_hist,
        ggml_reshape_3d(c, conv_state, kernel - 1, conv_channels, 1)));

    ggml_tensor * conv_op = ggml_ssm_conv(c, conv_input, L.ssm_conv1d);
    // The fused kernel's source is a graph dependency, so qkv stays live
    // through the CONCAT-tail write under allocator reuse.
    conv_op->src[3] = qkv_t;
    ggml_tensor * conv = ggml_silu(c, conv_op);
    const size_t esz = ggml_element_size(conv);
    const size_t tstride = (size_t) conv_channels * esz;
    ggml_tensor * q_raw = ggml_view_3d(c, conv, D, Hk, T, D * esz,
                                       tstride, 0);
    ggml_tensor * k_raw = ggml_view_3d(c, conv, D, Hk, T, D * esz,
                                       tstride, D * Hk * esz);
    ggml_tensor * q_c = ggml_scale(c, ggml_rms_norm(c, q_raw, eps / (float) D),
                                   1.0f / sqrtf((float) D));
    ggml_tensor * k_c = ggml_scale(c, ggml_rms_norm(c, k_raw, eps / (float) D),
                                   1.0f / sqrtf((float) D));
    ggml_tensor * v_c = ggml_view_3d(c, conv, D, Hv, T, D * esz,
                                     tstride, 2 * D * Hk * esz);
    ggml_tensor * state4 = ggml_reshape_4d(c, ssm_state, D, D, Hv, 1);
    ggml_tensor * gdn = ggml_gated_delta_net(c, q_c, k_c, v_c, gate, beta, state4);
    ggml_gated_delta_net_set_skip_intermediate(gdn, true);
    ggml_tensor * attn = ggml_view_4d(c, gdn, D, Hv, T, 1,
        ggml_row_size(gdn->type, D),
        ggml_row_size(gdn->type, D * Hv),
        ggml_row_size(gdn->type, D * Hv * T), 0);
    ggml_tensor * new_state = ggml_view_4d(c, gdn, D, D, Hv, 1,
        ggml_row_size(gdn->type, D),
        ggml_row_size(gdn->type, D * D),
        ggml_row_size(gdn->type, D * D * Hv),
        ggml_row_size(gdn->type, D * Hv * T));
    ggml_build_forward_expand(gf, ggml_cpy(c, new_state, state4));
    ggml_tensor * normed = ggml_mul(c, ggml_rms_norm(c, attn, eps), L.ssm_norm);
    ggml_tensor * out = ggml_mul(c, normed,
        ggml_sigmoid(c, ggml_reshape_4d(c, z, D, Hv, T, 1)));
    return ggml_reshape_2d(c, out, d_in, T);
}

static ggml_tensor * build_full_attn_projected(ggml_context * c, ggml_cgraph * gf,
        ggml_tensor * qfull, ggml_tensor * kraw, ggml_tensor * vraw,
        ggml_tensor * positions, ggml_tensor * mask, const Qwen4ExpLayer & L,
        const Qwen4ExpWeights & w, ggml_tensor * k_cache, ggml_tensor * v_cache,
        int pos0, int64_t kv_view_len) {
    const int64_t D = w.n_embd_head_k, Hq = w.n_head, Hk = w.n_head_kv;
    const int64_t T = qfull->ne[1];
    const float eps = w.rms_eps;
    ggml_tensor * q = ggml_rms_norm(c,
        ggml_view_3d(c, qfull, D, Hq, T, 2 * D * ggml_element_size(qfull),
                     2 * D * Hq * ggml_element_size(qfull), 0), eps);
    q = ggml_mul(c, q, L.q_norm);
    ggml_tensor * gate = ggml_cont_2d(c,
        ggml_view_3d(c, qfull, D, Hq, T, 2 * D * ggml_element_size(qfull),
                     2 * D * Hq * ggml_element_size(qfull), D * ggml_element_size(qfull)),
        D * Hq, T);
    ggml_tensor * k = ggml_mul(c,
        ggml_rms_norm(c, ggml_reshape_3d(c, kraw, D, Hk, T), eps), L.k_norm);
    ggml_tensor * v = ggml_reshape_3d(c, vraw, D, Hk, T);
    int sections[4] = { w.rope_sections[0], w.rope_sections[1],
                        w.rope_sections[2], w.rope_sections[3] };
    q = ggml_rope_multi(c, q, positions, nullptr, w.rope_dimension_count, sections,
        GGML_ROPE_TYPE_MROPE, 0, w.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_multi(c, k, positions, nullptr, w.rope_dimension_count, sections,
        GGML_ROPE_TYPE_MROPE, 0, w.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * kt = ggml_permute(c, ggml_cast(c, k, k_cache->type), 0, 2, 1, 3);
    ggml_tensor * vt = ggml_permute(c, ggml_cast(c, v, v_cache->type), 0, 2, 1, 3);
    ggml_build_forward_expand(gf, ggml_cpy(c, kt,
        ggml_view_3d(c, k_cache, D, T, Hk, k_cache->nb[1],
                     k_cache->nb[2], (size_t) pos0 * k_cache->nb[1])));
    ggml_build_forward_expand(gf, ggml_cpy(c, vt,
        ggml_view_3d(c, v_cache, D, T, Hk, v_cache->nb[1],
                     v_cache->nb[2], (size_t) pos0 * v_cache->nb[1])));
    ggml_tensor * kfull = ggml_view_3d(c, k_cache, D, kv_view_len, Hk,
                                       k_cache->nb[1], k_cache->nb[2], 0);
    ggml_tensor * vfull = ggml_view_3d(c, v_cache, D, kv_view_len, Hk,
                                       v_cache->nb[1], v_cache->nb[2], 0);
    ggml_tensor * qfa = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
    ggml_tensor * attn = ggml_flash_attn_ext(c, qfa, kfull, vfull, mask,
        1.0f / sqrtf((float) D), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    return ggml_mul(c, ggml_reshape_2d(c, attn, D * Hq, T), ggml_sigmoid(c, gate));
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

Qwen4ExpForwardResult qwen4exp_forward(ggml_backend_t backend,
                                       const Qwen4ExpWeights & w,
                                       Qwen4ExpCache & cache,
                                       const int32_t * tokens,
                                       int n_tokens,
                                       int pos0,
                                       std::vector<float> & out_logits) {
    Qwen4ExpForwardResult res;
    if (n_tokens <= 0 || pos0 < 0 || !tokens) return res;
    // QWEN4EXP_DUMP=1 materialises per-layer activations and prints finite/absmax/mean.
    static const bool dump = getenv("QWEN4EXP_DUMP") != nullptr;
    // The fused reduction can differ from upstream's ggml_rms_norm below one
    // ulp; keep the unfused form for upstream differential checks.
    static const bool upstream = [] { const char * v = getenv("QWEN4EXP_UPSTREAM"); return v && std::atoi(v) != 0; }();
    const bool hc_fused = !upstream;
    // T=1 decode reuses one context/allocator and a stable bucketed graph.
    // Past the QSA block budget the selected-cell graph changes shape as blocks complete, so it is rebuilt per step.
    const Qwen4ExpQsaMode qsa = qsa_mode(w, cache, n_tokens, pos0, upstream);
    // T=1 decode reuses one context/allocator; below the QSA budget it also keeps a stable bucketed graph.
    const bool reuse_ws = !upstream && n_tokens == 1 && !dump;
    const bool use_stable_graph = reuse_ws && qsa == QSA_DENSE;
    std::vector<std::pair<ggml_tensor *, std::string>> dump_t;
    auto dump_mark = [&](ggml_tensor * t, const char * label) {
        if (t && dump) {
            // A view output must keep its owning allocation alive until the dump.
            for (ggml_tensor * base = t; base; base = base->view_src) ggml_set_output(base);
            ggml_set_name(t, label);
            dump_t.emplace_back(t, label);
        }
    };
    if (pos0 + n_tokens > cache.max_ctx) {
        std::fprintf(stderr, "[qwen4exp] context overflow: %d + %d > %d\n",
                     pos0, n_tokens, cache.max_ctx);
        return res;
    }

    std::vector<int> lin_idx(w.n_layer, -1);
    std::vector<int> full_idx(w.n_layer, -1);
    for (size_t i = 0; i < cache.linear_layer_ids.size(); ++i) lin_idx[cache.linear_layer_ids[i]] = (int) i;
    for (size_t i = 0; i < cache.full_layer_ids.size(); ++i)   full_idx[cache.full_layer_ids[i]] = (int) i;

    std::vector<float> emb((size_t) w.n_embd * n_tokens);
    if (!w.embedder.embed(tokens, n_tokens, emb.data())) {
        std::fprintf(stderr, "[qwen4exp] cpu embedding failed\n");
        return res;
    }

    const bool has_ple = !cache.ple_layer_ids.empty() && w.ple_reader.available();
    const int64_t ple_heads = w.ple_n_heads;
    std::vector<int32_t> ple_rows(has_ple ? (size_t) ple_heads * n_tokens : 0);
    std::vector<float>   ple_data(has_ple ? (size_t) w.ple_head_dim * ple_heads * n_tokens : 0);
    std::vector<int32_t> ple_prev = cache.ple_prev;   // oldest first, size <= ng-1
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
        cache.ple_prev = std::move(next);
    }

    const int64_t T = n_tokens;
    const int64_t kv_len = pos0 + n_tokens;
    const bool fa_pad256 = upstream;   // the upstream reference pads K/V to 256
    // Give a new stable graph at least one full 256-token generation window.
    // The fixed mask excludes its padded tail, while the stable K/V views and
    // set_rows index keep every graph pointer and property unchanged.
    const int64_t stable_kv_bucket = use_stable_graph
        ? std::min<int64_t>(cache.max_ctx, ((kv_len + 511) / 256) * 256)
        : 0;

    auto run_stable = [&](Qwen4ExpDecodeWorkspace & ws) -> bool {
        int32_t pos[4] = { pos0, pos0, pos0, 0 };
        const int32_t kv_row = pos0;
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask_data((size_t) ws.kv_bucket, ninf);
        std::fill(mask_data.begin(), mask_data.begin() + kv_len, zero);

        ggml_backend_tensor_set_async(backend, ws.inp_emb, emb.data(), 0,
                                      sizeof(float) * emb.size());
        ggml_backend_tensor_set_async(backend, ws.positions, pos, 0, sizeof(pos));
        ggml_backend_tensor_set_async(backend, ws.kv_row, &kv_row, 0, sizeof(kv_row));
        ggml_backend_tensor_set_async(backend, ws.mask, mask_data.data(), 0,
                                      sizeof(ggml_fp16_t) * mask_data.size());
        if (ws.ple_in) {
            ggml_backend_tensor_set_async(backend, ws.ple_in, ple_data.data(), 0,
                                          sizeof(float) * ple_data.size());
        }
        if (ggml_backend_graph_compute(backend, ws.gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[qwen4exp] stable graph compute failed\n");
            return false;
        }
        out_logits.resize((size_t) w.n_vocab);
        ggml_backend_tensor_get(ws.logits, out_logits.data(), 0, sizeof(float) * w.n_vocab);
        return true;
    };

    Qwen4ExpDecodeWorkspace & decode_ws = cache.decode_workspace;
    if (use_stable_graph && decode_ws.gf && kv_len <= decode_ws.kv_bucket) {
        if (!run_stable(decode_ws)) return res;
        res.ok = true;
        res.n_tokens = n_tokens;
        res.pos0 = pos0;
        return res;
    }
    if (use_stable_graph && decode_ws.ctx) {
        clear_qwen4exp_decode_workspace(decode_ws);
    }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 200000 +
                  ggml_graph_overhead_custom(200000, false) + (1u << 20);
    ip.no_alloc = true;
    ggml_context * ctx = nullptr;
    if (reuse_ws) {
        if (cache.decode_workspace.ctx == nullptr) {
            cache.decode_workspace.ctx = ggml_init(ip);
        } else {
            ggml_reset(cache.decode_workspace.ctx);
        }
        ctx = cache.decode_workspace.ctx;
        if (!use_stable_graph) {   // the reset just freed the stable graph's tensors
            decode_ws.gf = nullptr;
            decode_ws.kv_bucket = 0;
        }
    } else {
        ctx = ggml_init(ip);
    }
    if (!ctx) return res;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 200000, false);

    const int64_t graph_kv_len = use_stable_graph ? stable_kv_bucket : kv_len;
    const int64_t mask_len = use_stable_graph ? stable_kv_bucket
        : (fa_pad256 ? (kv_len + 255)/256*256 : kv_len);

    ggml_tensor * inp_emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, T);
    ggml_set_input(inp_emb);
    dump_mark(inp_emb, "L00.inp");
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * T);
    ggml_set_input(positions);
    ggml_tensor * kv_row = nullptr;
    if (use_stable_graph) {
        kv_row = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_input(kv_row);
    }
    ggml_tensor * mask = nullptr;
    // QSA derives its own complete-block visibility: the prefill kernel needs no dense mask.
    const bool qsa_all = qsa == QSA_PREFILL;
    if ((T > 1 || fa_pad256 || use_stable_graph) && !qsa_all) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, mask_len, T);
        ggml_set_input(mask);
    }
    ggml_tensor * ple_in = nullptr;
    if (has_ple) {
        ple_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.ple_head_dim * ple_heads, T);
        ggml_set_input(ple_in);
    }

    ggml_tensor * res_hc = repeat_dim1(ctx,
        ggml_reshape_3d(ctx, inp_emb, w.n_embd, 1, T), w.n_hc);
    dump_mark(res_hc, "L00.emb");

    ggml_tensor * xn_next = nullptr;

    for (int il = 0; il < w.n_layer; ++il) {
        const Qwen4ExpLayer & L = w.layers[il];

        if (L.is_ple && has_ple) {
            res_hc = build_ple(ctx, gf, res_hc, ple_in, L, w,
                               cache.ple_conv_state.empty() ? nullptr :
                                   cache.ple_conv_state[0], dump_mark);
            xn_next = nullptr;   // PLE changed the residual; the norm must rerun
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur;
        char dlab[32];
        if (hc_fused && xn_next != nullptr) {
            cur = hc_mix_from_xn(ctx, xn_next, L.hc_attn_down, L.hc_attn_up,
                                 L.hc_attn_inject, &inject, w.n_embd, w.n_hc);
            xn_next = nullptr;
        } else {
            cur = hc_mix(ctx, res_hc, L.hc_attn_norm, L.hc_attn_down,
                         L.hc_attn_up, L.hc_attn_inject, &inject,
                         w.n_embd, w.n_hc, w.rms_eps);
            xn_next = nullptr;
        }
        std::snprintf(dlab, sizeof dlab, "L%02d.hcmix", il);
        dump_mark(cur, dlab);
        if (L.is_full_attention) {
            const int fi = full_idx[il];
            cur = build_full_attn(ctx, gf, cur, L, w,
                                  cache.attn_k[fi], cache.attn_v[fi], cache.indexer_k[fi],
                                  qsa_enabled() && !upstream ? cache.indexer_raw[fi] : nullptr,
                                  positions, mask, kv_row, graph_kv_len, pos0,
                                  il < (int) w.compress_ratios.size() ? w.compress_ratios[il] : 0,
                                  cache.indexer_blocks, qsa, il, dump_mark);
        } else {
            const int li = lin_idx[il];
            cur = build_linear_attn(ctx, gf, cur, L, w,
                                    cache.ssm_state[li], cache.conv_state[li], il, dump_mark);
        }
        std::snprintf(dlab, sizeof dlab, "L%02d.att", il);
        dump_mark(cur, dlab);
        int64_t layer_T = T;
        if (il == w.n_layer - 1 && T > 1) {
            // Upstream selects output rows before the final HC/FFN, so its
            // quantized matmuls dispatch with one token (MMV rather than MMQ).
            // The default path reuses that selection after all attention cache
            // writes. Earlier rows have no remaining stateful consumers.
            cur = ggml_view_2d(ctx, cur, w.n_embd, 1, cur->nb[1], (T - 1)*cur->nb[1]);
            inject = ggml_view_2d(ctx, inject, inject->ne[0], 1, inject->nb[1], (T - 1)*inject->nb[1]);
            res_hc = ggml_view_3d(ctx, res_hc, w.n_embd, w.n_hc, 1,
                                 res_hc->nb[1], res_hc->nb[2], (T - 1)*res_hc->nb[2]);
            layer_T = 1;
        }
        if (hc_fused) {
            ggml_tensor * ffn_fused = hc_combine_norm(ctx, inject, res_hc, cur,
                L.hc_ffn_norm, w.n_embd, w.n_hc, layer_T, w.rms_eps);
            res_hc = hc_norm_res(ctx, ffn_fused, w.n_embd, w.n_hc, layer_T);
            std::snprintf(dlab, sizeof dlab, "L%02d.hcA", il);
            dump_mark(ffn_fused, dlab);
            ggml_tensor * ffn_xn = hc_norm_xn(ctx, ffn_fused, w.n_embd, w.n_hc, layer_T);
            std::snprintf(dlab, sizeof dlab, "L%02d.ffnxn", il);
            dump_mark(ffn_xn, dlab);
            cur = hc_mix_from_xn(ctx, ffn_xn,
                                 L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, &inject,
                                 w.n_embd, w.n_hc);
        } else {
            // Upstream build_hc_combine + build_hc_mix: the grouped RMSNorm runs
            // as its own ggml_rms_norm (1024-thread reduction), bit-matching upstream.
            res_hc = hc_combine(ctx, res_hc, cur, inject, w.n_embd, w.n_hc, layer_T);
            cur = hc_mix(ctx, res_hc, L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up,
                         L.hc_ffn_inject, &inject, w.n_embd, w.n_hc, w.rms_eps);
        }
        std::snprintf(dlab, sizeof dlab, "L%02d.fmix", il);
        dump_mark(cur, dlab);
        const bool next_ple = (il + 1 < w.n_layer) && w.layers[il + 1].is_ple && has_ple;
        // Prefill: the MoE combine runs inside the next HC_COMBINE_NORM (one kernel fewer; last-bit numerics change
        // from FMA contraction in the new kernel, covered by the long-prompt quality gate). Not at T=1: it cost ~1.7% decode.
        Qwen4ExpMoeParts moe_parts;
        const bool fold = f16_paths() && hc_fused && !next_ple && ggml_backend_cuda_mmb_prefill(layer_T);
        cur = build_moe(ctx, cur, L, w, il, dump_mark, fold ? &moe_parts : nullptr);
        std::snprintf(dlab, sizeof dlab, "L%02d.moe", il);
        dump_mark(cur, dlab);

        if (fold) {
            ggml_tensor * gamma = (il + 1 < w.n_layer)
                ? w.layers[il + 1].hc_attn_norm : w.output_hc_norm;
            ggml_tensor * f = ggml_hc_combine_norm_moe(ctx, inject, res_hc, moe_parts.down, moe_parts.weights,
                moe_parts.shared, moe_parts.shared_logit, gamma, 1.0f / (float) w.n_hc, 0.0f, 2.0f, 0.0f, w.rms_eps);
            res_hc = hc_norm_res(ctx, f, w.n_embd, w.n_hc, layer_T);
            xn_next = hc_norm_xn(ctx, f, w.n_embd, w.n_hc, layer_T);
        } else if (next_ple) {
            res_hc = hc_combine(ctx, res_hc, cur, inject, w.n_embd, w.n_hc, layer_T);
            xn_next = nullptr;
            std::snprintf(dlab, sizeof dlab, "L%02d.res", il);
            dump_mark(res_hc, dlab);
        } else if (hc_fused) {
            ggml_tensor * gamma = (il + 1 < w.n_layer)
                ? w.layers[il + 1].hc_attn_norm : w.output_hc_norm;
            ggml_tensor * f = hc_combine_norm(ctx, inject, res_hc, cur,
                gamma, w.n_embd, w.n_hc, layer_T, w.rms_eps);
            res_hc = hc_norm_res(ctx, f, w.n_embd, w.n_hc, layer_T);
            xn_next = hc_norm_xn(ctx, f, w.n_embd, w.n_hc, layer_T);
            std::snprintf(dlab, sizeof dlab, "L%02d.res", il);
            dump_mark(f, dlab);
        } else {
            res_hc = hc_combine(ctx, res_hc, cur, inject, w.n_embd, w.n_hc, layer_T);
            xn_next = nullptr;
            std::snprintf(dlab, sizeof dlab, "L%02d.res", il);
            dump_mark(res_hc, dlab);
        }
    }

    ggml_tensor * final = (xn_next != nullptr)
        ? hc_mix_from_xn(ctx, xn_next, w.output_hc_down, w.output_hc_up,
                         nullptr, nullptr, w.n_embd, w.n_hc)
        : hc_mix(ctx, res_hc, w.output_hc_norm, w.output_hc_down,
                 w.output_hc_up, nullptr, nullptr,
                 w.n_embd, w.n_hc, w.rms_eps);
    dump_mark(final, "final");
    ggml_tensor * last = final->ne[1] > 1
        ? ggml_view_2d(ctx, final, w.n_embd, 1, final->nb[1], (size_t) (final->ne[1] - 1) * final->nb[1])
        : final;
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, last);
    ggml_set_output(logits);
    ggml_set_name(logits, "logits");
    dump_mark(logits, "logits");
    ggml_build_forward_expand(gf, logits);

    // Point input tensors at this call's pinned ring slot before allocation so the gallocr leaves them alone.
    char * ring_embd = nullptr;
    char * ring_pos  = nullptr;
    char * ring_mask = nullptr;
    char * ring_ple  = nullptr;
    if (!use_stable_graph && cache.input_ring.enabled) {
        const size_t embd_need = static_cast<size_t>(w.n_embd) * T * sizeof(float);
        const size_t pos_need  = static_cast<size_t>(4) * T * sizeof(int32_t);
        const size_t ple_need  = has_ple
            ? static_cast<size_t>(w.ple_head_dim) * ple_heads * T * sizeof(float) : 0;
        const size_t mask_need = mask
            ? static_cast<size_t>(mask_len) * T * sizeof(ggml_fp16_t) : 0;
        ggml_backend_buffer_type_t host_buft =
            ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
        if (host_buft != nullptr &&
            qwen4exp_input_ring_reserve(cache.input_ring, host_buft,
                                        embd_need, pos_need, ple_need, mask_need)) {
            // Wait for the graph that last read this slot; the logits read already synchronizes each forward.
            if (cache.input_ring.writes >= 2) {
                ggml_backend_synchronize(backend);
            }
            const size_t slot = static_cast<size_t>(cache.input_ring.next_slot);
            cache.input_ring.next_slot = (cache.input_ring.next_slot + 1) % 2;
            cache.input_ring.writes++;
            char * slot_base = cache.input_ring.base + slot * cache.input_ring.slot_bytes;
            ring_embd = slot_base + cache.input_ring.embd_off;
            ring_pos  = slot_base + cache.input_ring.pos_off;
            ring_ple  = slot_base + cache.input_ring.ple_off;
            ring_mask = slot_base + cache.input_ring.mask_off;
            ggml_backend_tensor_alloc(cache.input_ring.buf, inp_emb, ring_embd);
            ggml_backend_tensor_alloc(cache.input_ring.buf, positions, ring_pos);
            if (mask) {
                ggml_backend_tensor_alloc(cache.input_ring.buf, mask, ring_mask);
            }
            if (ple_in) {
                ggml_backend_tensor_alloc(cache.input_ring.buf, ple_in, ring_ple);
            }
        }
    }

    ggml_gallocr_t galloc = nullptr;
    if (reuse_ws) {
        if (cache.decode_workspace.alloc == nullptr) {
            cache.decode_workspace.alloc =
                ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        }
        galloc = cache.decode_workspace.alloc;
    } else {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!galloc) {
        std::fprintf(stderr, "[qwen4exp] graph allocator creation failed\n");
        if (!reuse_ws) ggml_free(ctx);
        return res;
    }
    // T=1 graphs keep the same broad shape but advancing KV views can change
    // lifetimes. Recompute assignments while retaining the allocator buffers;
    // reusing the old index-wise plan produced incorrect tokens.
    const bool reserve_ok = !reuse_ws ||
        !cache.decode_workspace.planned || ggml_gallocr_reserve(galloc, gf);
    if (!reserve_ok || !ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "[qwen4exp] graph alloc failed (T=%lld kv_len=%lld)\n",
                     (long long) T, (long long) kv_len);
        if (reuse_ws) cache.decode_workspace.planned = false;
        if (!reuse_ws) {
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
        }
        return res;
    }
    if (reuse_ws) cache.decode_workspace.planned = true;

    if (use_stable_graph) {
        decode_ws.gf = gf;
        decode_ws.inp_emb = inp_emb;
        decode_ws.positions = positions;
        decode_ws.mask = mask;
        decode_ws.ple_in = ple_in;
        decode_ws.kv_row = kv_row;
        decode_ws.logits = logits;
        decode_ws.kv_bucket = stable_kv_bucket;
    }

    // M-RoPE sections are section-major [s*T + i]: 0..2 carry the position, 3 is zero.
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
    const int32_t kv_row_value = pos0;
    if (use_stable_graph) {
        ggml_backend_tensor_set_async(backend, inp_emb, emb.data(), 0, sizeof(float) * emb.size());
        ggml_backend_tensor_set_async(backend, positions, pos.data(), 0, sizeof(int32_t) * pos.size());
        ggml_backend_tensor_set_async(backend, kv_row, &kv_row_value, 0, sizeof(kv_row_value));
        ggml_backend_tensor_set_async(backend, mask, m.data(), 0, sizeof(ggml_fp16_t) * m.size());
        if (ple_in) {
            ggml_backend_tensor_set_async(backend, ple_in, ple_data.data(), 0,
                                          sizeof(float) * ple_data.size());
        }
    } else if (ring_embd != nullptr) {
        std::memcpy(inp_emb->data, emb.data(), sizeof(float) * emb.size());
        std::memcpy(positions->data, pos.data(), sizeof(int32_t) * pos.size());
        if (mask) {
            std::memcpy(mask->data, m.data(), sizeof(ggml_fp16_t) * m.size());
        }
        if (ple_in) {
            std::memcpy(ple_in->data, ple_data.data(), sizeof(float) * ple_data.size());
        }
    } else {
        ggml_backend_tensor_set(inp_emb, emb.data(), 0, sizeof(float) * emb.size());
        ggml_backend_tensor_set(positions, pos.data(), 0, sizeof(int32_t) * pos.size());
        if (mask) {
            ggml_backend_tensor_set(mask, m.data(), 0, sizeof(ggml_fp16_t) * m.size());
        }
        if (ple_in) {
            ggml_backend_tensor_set(ple_in, ple_data.data(), 0, sizeof(float) * ple_data.size());
        }
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[qwen4exp] graph compute failed\n");
        if (reuse_ws) {
            clear_qwen4exp_decode_workspace(decode_ws);
        } else {
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
        }
        return res;
    }
    // Commit only after the graph computed: a failed compute must not mark blocks the kernel never pooled.
    if (qsa != QSA_DENSE) cache.indexer_blocks = (int) ((pos0 + T) / qsa_ratio(w));

    out_logits.resize((size_t) w.n_vocab);
    ggml_backend_tensor_get(logits, out_logits.data(), 0, sizeof(float) * w.n_vocab);

    if (dump) {
        for (auto & dt : dump_t) {
            ggml_tensor * t = dt.first;
            const size_t n = (size_t) ggml_nelements(t);
            if (t->type == GGML_TYPE_I32) {
                // sel is a non-contiguous view; tensor_get asserts on raw spans across stride gaps.
                std::vector<int32_t> iv(n);
                const int64_t ne0 = t->ne[0];
                const size_t  row_bytes = (size_t) ne0 * sizeof(int32_t);
                for (int64_t r = 0; r < (int64_t) (n / (size_t) ne0); ++r) {
                    ggml_backend_tensor_get(t, iv.data() + r * ne0, (size_t) r * t->nb[1], row_bytes);
                }
                std::fprintf(stderr, "[dump] %-10s n=%-7zu finite=1 absmax=0 mean=0 sumsq=0 ids:",
                    dt.second.c_str(), n);
                for (size_t i = 0; i < n && i < 2048; ++i) std::fprintf(stderr, " %d", iv[i]);
                std::fprintf(stderr, "\n");
                continue;
            }
            std::vector<float> v(n);
            ggml_backend_tensor_get(t, v.data(), 0, n * sizeof(float));
            static const char * dump_bin = getenv("QWEN4EXP_DUMP_BIN");
            if (dump_bin && dump_bin[0] && n_tokens > 1) {
                std::string names = dump_bin, want;
                size_t pos = 0;
                while (pos <= names.size()) {
                    size_t comma = names.find(',', pos);
                    want = names.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                    if (want == dt.second) {
                        char path[256];
                        std::snprintf(path, sizeof path, "/tmp/our_%s.bin", dt.second.c_str());
                        FILE * f = std::fopen(path, "wb");
                        if (f) { std::fwrite(v.data(), sizeof(float), n, f); std::fclose(f); }
                        break;
                    }
                    if (comma == std::string::npos) break;
                    pos = comma + 1;
                }
            }
            bool finite = true; double amax = 0, sum = 0, sumsq = 0;
            size_t argmax = 0;
            for (size_t i = 0; i < n; ++i) {
                const float x = v[i];
                if (!std::isfinite(x)) finite = false;
                const double a = std::fabs((double) x);
                if (a > amax) amax = a;
                if (x > v[argmax]) argmax = i;
                sum += x;
                sumsq += (double) x * (double) x;
            }
            std::fprintf(stderr, "[dump] %-10s n=%-7zu finite=%d absmax=%-12.5g mean=%-12.5g sumsq=%-18.10g argmax=%zu\n",
                dt.second.c_str(), n, (int) finite, amax, n ? sum / (double) n : 0.0, sumsq, argmax);
            if (n > 1000) {
                for (int r = 0; r < 5; ++r) {
                    size_t bi = 0; float bv = -1e30f;
                    for (size_t i = 0; i < n; ++i) if (v[i] > bv) { bv = v[i]; bi = i; }
                    std::fprintf(stderr, "[dump]   top%d idx=%zu val=%.5f\n", r, bi, bv);
                    v[bi] = -1e30f;
                }
            }
        }
    }

    if (!reuse_ws) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
    }

    res.ok = true;
    res.n_tokens = n_tokens;
    res.pos0 = pos0;
    return res;
}

Qwen4ExpForwardResult qwen4exp_forward_batched(
        ggml_backend_t backend, const Qwen4ExpWeights & w,
        Qwen4ExpCache * const * caches, const int32_t * tokens,
        const int32_t * positions, int n_slots,
        Qwen4ExpBatchedDecodeWorkspace & workspace,
        std::vector<std::vector<float>> & out_logits) {
    Qwen4ExpForwardResult result;
    const char * enabled = std::getenv("QWEN4EXP_BATCHED_DECODE");
    const char * upstream_env = std::getenv("QWEN4EXP_UPSTREAM");
    if (!enabled || std::atoi(enabled) == 0 ||
        (upstream_env && std::atoi(upstream_env) != 0) ||
        !backend || !caches || !tokens || !positions || n_slots <= 0) return result;
    out_logits.clear();

    // Preserve the established single-sequence arithmetic and state transitions
    // exactly. The new graph is only used for true multi-slot calls.
    if (n_slots == 1) {
        if (!caches[0]) return result;
        std::vector<float> logits;
        result = qwen4exp_forward(backend, w, *caches[0], tokens, 1, positions[0], logits);
        if (result.ok) {
            out_logits.assign(1, std::move(logits));
        }
        return result;
    }
    if (n_slots > 8) return result;
    for (int s = 0; s < n_slots; ++s) {
        if (!caches[s] || positions[s] < 0 || positions[s] >= caches[s]->max_ctx ||
            caches[s]->kv_type != GGML_TYPE_F16 ||
            caches[s]->linear_layer_ids != caches[0]->linear_layer_ids ||
            caches[s]->full_layer_ids != caches[0]->full_layer_ids ||
            caches[s]->ple_layer_ids != caches[0]->ple_layer_ids) return result;
        for (int prev = 0; prev < s; ++prev) if (caches[prev] == caches[s]) return result;
    }

    static const bool hc_fused = [] {
        const char * v = std::getenv("QWEN4EXP_UPSTREAM");
        return !(v && std::atoi(v) != 0);
    }();
    const int64_t T = n_slots;
    const bool has_ple = w.ple_reader.available() && !caches[0]->ple_layer_ids.empty();
    const int64_t ple_heads = w.ple_n_heads;
    const int64_t ple_row_size = w.ple_head_dim * ple_heads;
    std::vector<int> lin_idx(w.n_layer, -1), full_idx(w.n_layer, -1), ple_idx(w.n_layer, -1);
    for (int i = 0; i < (int) caches[0]->linear_layer_ids.size(); ++i)
        lin_idx[caches[0]->linear_layer_ids[i]] = i;
    for (int i = 0; i < (int) caches[0]->full_layer_ids.size(); ++i)
        full_idx[caches[0]->full_layer_ids[i]] = i;
    for (int i = 0; i < (int) caches[0]->ple_layer_ids.size(); ++i)
        ple_idx[caches[0]->ple_layer_ids[i]] = i;

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

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 400000 +
                  ggml_graph_overhead_custom(400000, false) + (2u << 20);
    ip.no_alloc = true;
    if (!workspace.ctx) workspace.ctx = ggml_init(ip);
    else ggml_reset(workspace.ctx);
    ggml_context * ctx = workspace.ctx;
    if (!ctx) return result;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 400000, false);
    if (!gf) return result;
    ggml_tensor * inp_emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, T);
    ggml_set_input(inp_emb);
    ggml_tensor * ple_in = nullptr;
    if (has_ple) {
        ple_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ple_row_size, T);
        ggml_set_input(ple_in);
    }
    ggml_tensor * res_hc = repeat_dim1(ctx,
        ggml_reshape_3d(ctx, inp_emb, w.n_embd, 1, T), w.n_hc);
    ggml_tensor * xn_next = nullptr;
    std::vector<ggml_tensor *> position_inputs;
    std::vector<ggml_tensor *> mask_inputs;
    std::vector<int64_t> kv_view_lens((size_t) n_slots, 0);

    for (int il = 0; il < w.n_layer; ++il) {
        const Qwen4ExpLayer & L = w.layers[il];
        if (L.is_ple && has_ple) {
            // PLE table projection weights are read once for the whole row batch;
            // only the stateful convolution windows branch per sequence.
            ggml_tensor * key = mm(ctx, L.ple_key, ple_in);
            ggml_tensor * value = mm(ctx, L.ple_value, ple_in);
            ggml_tensor * rows = nullptr;
            for (int s = 0; s < n_slots; ++s) {
                ggml_tensor * hidden = ggml_view_3d(ctx, res_hc, w.n_embd, w.n_hc, 1,
                    res_hc->nb[1], res_hc->nb[2], (size_t) s * res_hc->nb[2]);
                ggml_tensor * row = build_ple_row(ctx, gf, hidden,
                    column(ctx, key, s), column(ctx, value, s), L, w,
                    caches[s]->ple_conv_state[(size_t) ple_idx[il]]);
                rows = rows ? ggml_concat(ctx, rows, row, 2) : row;
            }
            res_hc = rows;
            xn_next = nullptr;
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = hc_fused
            ? hc_mix_from_xn(ctx, xn_next ? xn_next : [&]() {
                ggml_tensor * xn = ggml_rms_norm(ctx, res_hc, w.rms_eps);
                xn = ggml_reshape_2d(ctx, xn, w.n_embd * w.n_hc, T);
                    xn = ggml_mul(ctx, xn, L.hc_attn_norm);
                    return ggml_reshape_3d(ctx, xn, w.n_embd, w.n_hc, T);
                }(), L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject,
                &inject, w.n_embd, w.n_hc)
            : hc_mix(ctx, res_hc, L.hc_attn_norm, L.hc_attn_down,
                L.hc_attn_up, L.hc_attn_inject, &inject,
                w.n_embd, w.n_hc, w.rms_eps);
        xn_next = nullptr;

        if (L.is_full_attention) {
            const int fi = full_idx[il];
            ggml_tensor * qfull = mm(ctx, L.wq, cur);
            ggml_tensor * kraw = mm(ctx, L.wk, cur);
            ggml_tensor * vraw = mm(ctx, L.wv, cur);
            ggml_tensor * attn_rows = nullptr;
            for (int s = 0; s < n_slots; ++s) {
                ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
                ggml_set_input(pos);
                position_inputs.push_back(pos);
                const int64_t kv_len = (int64_t) positions[s] + 1;
                const int64_t kv_view_len =
                    std::min<int64_t>(caches[s]->max_ctx, ((kv_len + 511) / 256) * 256);
                kv_view_lens[(size_t) s] = kv_view_len;
                ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_view_len, 1);
                ggml_set_input(mask);
                mask_inputs.push_back(mask);
                // The host fills one independent M-RoPE position vector per row.
                ggml_tensor * attn = build_full_attn_projected(ctx, gf,
                    column(ctx, qfull, s), column(ctx, kraw, s), column(ctx, vraw, s),
                    pos, mask, L, w, caches[s]->attn_k[(size_t) fi],
                    caches[s]->attn_v[(size_t) fi], positions[s], kv_view_len);
                attn_rows = attn_rows ? ggml_concat(ctx, attn_rows, attn, 1) : attn;
            }
            cur = mm(ctx, L.wo, attn_rows);
        } else {
            // Dense input projections are batched; each recurrent/conv op sees
            // exactly one row and its own slot state.
            const int li = lin_idx[il];
            ggml_tensor * qkv = mm(ctx, L.attn_qkv, cur);
            ggml_tensor * z = mm(ctx, L.attn_gate, cur);
            ggml_tensor * beta = mm(ctx, L.ssm_beta, cur);
            ggml_tensor * alpha = mm(ctx, L.ssm_alpha, cur);
            ggml_tensor * rows = nullptr;
            for (int s = 0; s < n_slots; ++s) {
                ggml_tensor * row = build_linear_attn_projected(ctx, gf,
                    column(ctx, qkv, s), column(ctx, z, s), column(ctx, beta, s),
                    column(ctx, alpha, s), L, w,
                    caches[s]->ssm_state[(size_t) li], caches[s]->conv_state[(size_t) li]);
                rows = rows ? ggml_concat(ctx, rows, row, 1) : row;
            }
            cur = mm(ctx, L.ssm_out, rows);
        }
        ggml_tensor * ffn_fused = nullptr;
        if (hc_fused) {
            ffn_fused = hc_combine_norm(ctx, inject, res_hc, cur,
                L.hc_ffn_norm, w.n_embd, w.n_hc, T, w.rms_eps);
            res_hc = hc_norm_res(ctx, ffn_fused, w.n_embd, w.n_hc, T);
            ggml_tensor * ffn_xn = hc_norm_xn(ctx, ffn_fused, w.n_embd, w.n_hc, T);
            cur = hc_mix_from_xn(ctx, ffn_xn,
                L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, &inject, w.n_embd, w.n_hc);
        } else {
            res_hc = hc_combine(ctx, res_hc, cur, inject, w.n_embd, w.n_hc, T);
            cur = hc_mix(ctx, res_hc, L.hc_ffn_norm, L.hc_ffn_down,
                L.hc_ffn_up, L.hc_ffn_inject, &inject, w.n_embd, w.n_hc, w.rms_eps);
        }
        cur = build_moe(ctx, cur, L, w, il);
        const bool next_ple = (il + 1 < w.n_layer) && w.layers[il + 1].is_ple && has_ple;
        if (next_ple) {
            // Match the single-sequence graph's PLE boundary: do not fold the
            // next layer's grouped norm into a residual that PLE will change.
            res_hc = hc_combine(ctx, res_hc, cur, inject, w.n_embd, w.n_hc, T);
            xn_next = nullptr;
        } else if (hc_fused) {
            ggml_tensor * gamma = (il + 1 < w.n_layer)
                ? w.layers[il + 1].hc_attn_norm : w.output_hc_norm;
            ggml_tensor * f = hc_combine_norm(ctx, inject, res_hc, cur,
                gamma, w.n_embd, w.n_hc, T, w.rms_eps);
            res_hc = hc_norm_res(ctx, f, w.n_embd, w.n_hc, T);
            xn_next = hc_norm_xn(ctx, f, w.n_embd, w.n_hc, T);
        } else {
            res_hc = hc_combine(ctx, res_hc, cur, inject, w.n_embd, w.n_hc, T);
            xn_next = nullptr;
        }
    }

    ggml_tensor * final = xn_next
        ? hc_mix_from_xn(ctx, xn_next, w.output_hc_down, w.output_hc_up,
                         nullptr, nullptr, w.n_embd, w.n_hc)
        : hc_mix(ctx, res_hc, w.output_hc_norm, w.output_hc_down,
                 w.output_hc_up, nullptr, nullptr, w.n_embd, w.n_hc, w.rms_eps);
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, final);
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    int max_kv = 0;
    for (int s = 0; s < n_slots; ++s)
        max_kv = std::max(max_kv, positions[s] + 1);
    trace_batch_widths(gf, "batched", n_slots, max_kv);

    if (!workspace.alloc)
        workspace.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!workspace.alloc || !ggml_gallocr_reserve(workspace.alloc, gf) ||
        !ggml_gallocr_alloc_graph(workspace.alloc, gf)) {
        workspace.planned = false;
        std::fprintf(stderr, "[qwen4exp] batched graph allocation failed (N=%d)\n", n_slots);
        return result;
    }
    workspace.planned = true;

    ggml_backend_tensor_set(inp_emb, emb.data(), 0, emb.size() * sizeof(float));
    if (ple_in) ggml_backend_tensor_set(ple_in, ple_data.data(), 0, ple_data.size() * sizeof(float));
    for (size_t i = 0; i < position_inputs.size(); ++i) {
        const int s = (int) (i % (size_t) n_slots);
        const int32_t p[4] = { positions[s], positions[s], positions[s], 0 };
        ggml_backend_tensor_set(position_inputs[i], p, 0, sizeof(p));
    }
    for (size_t s = 0; s < mask_inputs.size(); ++s) {
        const int slot = (int) (s % (size_t) n_slots);
        const int64_t kv_len = (int64_t) positions[slot] + 1;
        std::vector<ggml_fp16_t> mask((size_t) kv_view_lens[(size_t) slot],
            ggml_fp32_to_fp16(-INFINITY));
        std::fill(mask.begin(), mask.begin() + kv_len, ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(mask_inputs[s], mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }
    const bool batch_telemetry = prof_enabled();
    const auto compute_start = batch_telemetry
        ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const double compute_begin_s = batch_telemetry ? mono_now_s() : 0.0;
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        workspace.planned = false;
        std::fprintf(stderr, "[qwen4exp] batched graph compute failed\n");
        return result;
    }
    if (batch_telemetry) {
        const double compute_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - compute_start).count();
        std::fprintf(stderr,
            "[qwen4exp-batch-time] rows=%d compute_ms=%.3f begin_abs=%.9f done_abs=%.9f\n",
            n_slots, compute_ms, compute_begin_s, mono_now_s());
    }
    std::vector<float> packed((size_t) w.n_vocab * n_slots);
    ggml_backend_tensor_get(logits, packed.data(), 0, packed.size() * sizeof(float));
    out_logits.resize((size_t) n_slots);
    for (int s = 0; s < n_slots; ++s) {
        out_logits[(size_t) s].assign(packed.begin() + (size_t) s * w.n_vocab,
                                      packed.begin() + (size_t) (s + 1) * w.n_vocab);
    }
    if (has_ple) {
        for (int s = 0; s < n_slots; ++s) caches[s]->ple_prev = std::move(next_prev[s]);
    }
    result.ok = true;
    result.n_tokens = n_slots;
    result.pos0 = positions[0];
    return result;
}

Qwen4ExpForwardResult qwen4exp_forward_packed(
        ggml_backend_t backend, const Qwen4ExpWeights & w,
        const Qwen4ExpForwardSegment * segments, int n_segments,
        Qwen4ExpBatchedDecodeWorkspace & workspace,
        std::vector<std::vector<float>> & out_logits) {
    Qwen4ExpForwardResult result;
    const char * enabled = std::getenv("QWEN4EXP_BATCHED_DECODE");
    const char * upstream = std::getenv("QWEN4EXP_UPSTREAM");
    if (!enabled || std::atoi(enabled) == 0 ||
        (upstream && std::atoi(upstream) != 0) ||
        !backend || !segments || n_segments <= 0 || n_segments > 8) return result;
    out_logits.clear();
    if (!segments[0].cache) return result;

    if (n_segments == 1) {
        const Qwen4ExpForwardSegment & segment = segments[0];
        if (!segment.cache || !segment.tokens || segment.n_tokens <= 0) return result;
        std::vector<float> logits;
        result = qwen4exp_forward(backend, w, *segment.cache, segment.tokens,
                                  segment.n_tokens, segment.pos0, logits);
        if (result.ok) out_logits.push_back(std::move(logits));
        return result;
    }

    static const bool hc_fused = [] {
        const char * value = std::getenv("QWEN4EXP_UPSTREAM");
        return !(value && std::atoi(value) != 0);
    }();

    const int64_t ple_heads = w.ple_n_heads;
    const int64_t ple_row_size = w.ple_head_dim * ple_heads;
    const bool has_ple = w.ple_reader.available() &&
                         !segments[0].cache->ple_layer_ids.empty();
    std::vector<int> lin_idx(w.n_layer, -1), full_idx(w.n_layer, -1),
                     ple_idx(w.n_layer, -1);
    for (int i = 0; i < (int) segments[0].cache->linear_layer_ids.size(); ++i)
        lin_idx[segments[0].cache->linear_layer_ids[(size_t) i]] = i;
    for (int i = 0; i < (int) segments[0].cache->full_layer_ids.size(); ++i)
        full_idx[segments[0].cache->full_layer_ids[(size_t) i]] = i;
    for (int i = 0; i < (int) segments[0].cache->ple_layer_ids.size(); ++i)
        ple_idx[segments[0].cache->ple_layer_ids[(size_t) i]] = i;

    std::vector<int> offsets((size_t) n_segments);
    int total_rows = 0;
    std::vector<int32_t> all_tokens;
    std::vector<int64_t> kv_view_lens((size_t) n_segments);
    for (int s = 0; s < n_segments; ++s) {
        const Qwen4ExpForwardSegment & segment = segments[s];
        if (!segment.cache || !segment.tokens || segment.n_tokens <= 0 ||
            segment.pos0 < 0 ||
            (int64_t) segment.pos0 + segment.n_tokens > segment.cache->max_ctx ||
            segment.cache->kv_type != GGML_TYPE_F16 ||
            segment.cache->linear_layer_ids != segments[0].cache->linear_layer_ids ||
            segment.cache->full_layer_ids != segments[0].cache->full_layer_ids ||
            segment.cache->ple_layer_ids != segments[0].cache->ple_layer_ids ||
            segment.cache->cur_pos != segment.pos0 ||
            total_rows + segment.n_tokens > 2048) return result;
        for (int i = 0; i < segment.n_tokens; ++i)
            if (segment.tokens[i] < 0 || segment.tokens[i] >= w.n_vocab)
                return result;
        for (int prev = 0; prev < s; ++prev)
            if (segments[prev].cache == segment.cache) return result;
        offsets[(size_t) s] = total_rows;
        total_rows += segment.n_tokens;
        all_tokens.insert(all_tokens.end(), segment.tokens,
                          segment.tokens + segment.n_tokens);
        const int64_t kv_len = (int64_t) segment.pos0 + segment.n_tokens;
        kv_view_lens[(size_t) s] = kv_len;
    }

    std::vector<float> embeddings((size_t) w.n_embd * total_rows);
    if (!w.embedder.embed(all_tokens.data(), total_rows, embeddings.data())) {
        std::fprintf(stderr, "[qwen4exp] packed CPU embedding failed\n");
        return result;
    }

    std::vector<float> ple_data(has_ple ? (size_t) ple_row_size * total_rows : 0);
    std::vector<std::vector<int32_t>> next_prev((size_t) n_segments);
    if (has_ple) {
        const int64_t ng = w.ple_ngram_size;
        std::vector<int32_t> ple_rows((size_t) total_rows * ple_heads);
        for (int s = 0; s < n_segments; ++s) {
            const Qwen4ExpForwardSegment & segment = segments[s];
            const std::vector<int32_t> & prev = segment.cache->ple_prev;
            std::vector<int32_t> history = prev;
            history.insert(history.end(), segment.tokens,
                           segment.tokens + segment.n_tokens);
            for (int i = 0; i < segment.n_tokens; ++i) {
                const int64_t at = (int64_t) prev.size() + i;
                std::vector<uint64_t> ctx((size_t) ng);
                ctx[0] = (uint64_t) segment.tokens[i];
                bool cut = false;
                for (int64_t j = 1; j < ng; ++j) {
                    if (cut || at - j < 0) {
                        ctx[(size_t) j] = (uint64_t) w.ple_eos_token_id;
                        cut = true;
                    } else {
                        const int32_t token = history[(size_t) (at - j)];
                        if (token < 0 || token == w.ple_eos_token_id) cut = true;
                        ctx[(size_t) j] = cut
                            ? (uint64_t) w.ple_eos_token_id : (uint64_t) token;
                    }
                }
                for (int64_t order = 2; order <= ng; ++order) {
                    uint64_t mixed = ctx[0] * w.ple_layer_multipliers[0];
                    for (int64_t j = 1; j < order; ++j)
                        mixed ^= ctx[(size_t) j] *
                                 w.ple_layer_multipliers[(size_t) j];
                    const int64_t base = (order - 2) * w.ple_heads_per_ngram;
                    for (int64_t q = 0; q < w.ple_heads_per_ngram; ++q) {
                        const int64_t h = base + q;
                        const size_t row =
                            ((size_t) offsets[(size_t) s] + (size_t) i) *
                                (size_t) ple_heads + (size_t) h;
                        ple_rows[row] = (int32_t)
                            ((mixed % (uint64_t) w.ple_head_vocab_sizes[(size_t) h]) +
                             w.ple_head_offsets[(size_t) h]);
                    }
                }
            }
            const size_t keep = (size_t) std::min<int64_t>(
                ng - 1, (int64_t) history.size());
            next_prev[(size_t) s].assign(history.end() - keep, history.end());
        }
        if (!w.ple_reader.gather(ple_rows.data(),
                (int64_t) ple_rows.size(), ple_data.data())) {
            std::fprintf(stderr, "[qwen4exp] packed PLE gather failed\n");
            return result;
        }
    }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 400000 +
                  ggml_graph_overhead_custom(400000, false) + (2u << 20);
    ip.no_alloc = true;
    if (!workspace.ctx) workspace.ctx = ggml_init(ip);
    else ggml_reset(workspace.ctx);
    ggml_context * ctx = workspace.ctx;
    if (!ctx) return result;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 400000, false);
    if (!gf) return result;

    ggml_tensor * inp_emb = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, w.n_embd, total_rows);
    ggml_set_input(inp_emb);
    ggml_tensor * ple_in = nullptr;
    if (has_ple) {
        ple_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                    ple_row_size, total_rows);
        ggml_set_input(ple_in);
    }
    ggml_tensor * res_hc = repeat_dim1(ctx,
        ggml_reshape_3d(ctx, inp_emb, w.n_embd, 1, total_rows), w.n_hc);
    ggml_tensor * xn_next = nullptr;

    std::vector<ggml_tensor *> position_inputs((size_t) n_segments, nullptr);
    std::vector<ggml_tensor *> mask_inputs((size_t) n_segments, nullptr);
    std::vector<std::vector<int32_t>> positions_host((size_t) n_segments);
    std::vector<std::vector<ggml_fp16_t>> masks_host((size_t) n_segments);
    for (int s = 0; s < n_segments; ++s) {
        const Qwen4ExpForwardSegment & segment = segments[s];
        const int count = segment.n_tokens;
        positions_host[(size_t) s].assign((size_t) 4 * count, 0);
        for (int i = 0; i < count; ++i) {
            const int32_t pos = segment.pos0 + i;
            for (int axis = 0; axis < 3; ++axis)
                positions_host[(size_t) s][(size_t) axis * count + i] = pos;
        }
        ggml_tensor * position = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I32, (int64_t) 4 * count);
        ggml_set_input(position);
        position_inputs[(size_t) s] = position;

        const int64_t kv_len = (int64_t) segment.pos0 + count;
        const int64_t mask_len = kv_view_lens[(size_t) s];
        if (count > 1) {
            ggml_tensor * mask = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F16, mask_len, count);
            ggml_set_input(mask);
            mask_inputs[(size_t) s] = mask;
            auto & host = masks_host[(size_t) s];
            host.resize((size_t) mask_len * count);
            const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
            const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
            for (int row = 0; row < count; ++row) {
                const int64_t visible = (int64_t) segment.pos0 + row;
                for (int64_t col = 0; col < mask_len; ++col)
                    host[(size_t) row * mask_len + (size_t) col] =
                        col <= visible ? zero : ninf;
            }
            (void) kv_len;
        }
    }

    for (int il = 0; il < w.n_layer; ++il) {
        const Qwen4ExpLayer & L = w.layers[il];
        if (L.is_ple && has_ple) {
            ggml_tensor * key = mm(ctx, L.ple_key, ple_in);
            ggml_tensor * value = mm(ctx, L.ple_value, ple_in);
            ggml_tensor * rows = nullptr;
            for (int s = 0; s < n_segments; ++s) {
                const int start = offsets[(size_t) s];
                const int count = segments[s].n_tokens;
                ggml_tensor * hidden = token_span_3d(ctx, res_hc, start, count);
                ggml_tensor * row = build_ple_projected(ctx, gf, hidden,
                    column_span(ctx, key, start, count),
                    column_span(ctx, value, start, count), L, w,
                    segments[s].cache->ple_conv_state[(size_t) ple_idx[(size_t) il]]);
                rows = rows ? ggml_concat(ctx, rows, row, 2) : row;
            }
            res_hc = rows;
            xn_next = nullptr;
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = hc_fused
            ? hc_mix_from_xn(ctx, xn_next ? xn_next : [&]() {
                ggml_tensor * xn = ggml_rms_norm(ctx, res_hc, w.rms_eps);
                xn = ggml_reshape_2d(ctx, xn, w.n_embd * w.n_hc, total_rows);
                xn = ggml_mul(ctx, xn, L.hc_attn_norm);
                return ggml_reshape_3d(ctx, xn, w.n_embd, w.n_hc, total_rows);
            }(), L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject,
                &inject, w.n_embd, w.n_hc)
            : hc_mix(ctx, res_hc, L.hc_attn_norm, L.hc_attn_down,
                L.hc_attn_up, L.hc_attn_inject, &inject,
                w.n_embd, w.n_hc, w.rms_eps);
        xn_next = nullptr;

        if (L.is_full_attention) {
            ggml_tensor * qfull = mm(ctx, L.wq, cur);
            ggml_tensor * kraw = mm(ctx, L.wk, cur);
            ggml_tensor * vraw = mm(ctx, L.wv, cur);
            ggml_tensor * attention_rows = nullptr;
            const int fi = full_idx[(size_t) il];
            for (int s = 0; s < n_segments; ++s) {
                const int start = offsets[(size_t) s];
                const int count = segments[s].n_tokens;
                ggml_tensor * attn = build_full_attn_projected(ctx, gf,
                    column_span(ctx, qfull, start, count),
                    column_span(ctx, kraw, start, count),
                    column_span(ctx, vraw, start, count),
                    position_inputs[(size_t) s], mask_inputs[(size_t) s],
                    L, w, segments[s].cache->attn_k[(size_t) fi],
                    segments[s].cache->attn_v[(size_t) fi],
                    segments[s].pos0, kv_view_lens[(size_t) s]);
                attention_rows = attention_rows
                    ? ggml_concat(ctx, attention_rows, attn, 1) : attn;
            }
            cur = mm(ctx, L.wo, attention_rows);
        } else {
            const int li = lin_idx[(size_t) il];
            ggml_tensor * qkv = mm(ctx, L.attn_qkv, cur);
            ggml_tensor * z = mm(ctx, L.attn_gate, cur);
            ggml_tensor * beta = mm(ctx, L.ssm_beta, cur);
            ggml_tensor * alpha = mm(ctx, L.ssm_alpha, cur);
            ggml_tensor * linear_rows = nullptr;
            for (int s = 0; s < n_segments; ++s) {
                const int start = offsets[(size_t) s];
                const int count = segments[s].n_tokens;
                ggml_tensor * row = build_linear_attn_projected_sequence(ctx, gf,
                    column_span(ctx, qkv, start, count),
                    column_span(ctx, z, start, count),
                    column_span(ctx, beta, start, count),
                    column_span(ctx, alpha, start, count), L, w,
                    segments[s].cache->ssm_state[(size_t) li],
                    segments[s].cache->conv_state[(size_t) li]);
                linear_rows = linear_rows
                    ? ggml_concat(ctx, linear_rows, row, 1) : row;
            }
            cur = mm(ctx, L.ssm_out, linear_rows);
        }

        int64_t layer_rows = total_rows;
        if (il == w.n_layer - 1 && total_rows > n_segments) {
            ggml_tensor * cur_tail = nullptr;
            ggml_tensor * inject_tail = nullptr;
            ggml_tensor * residual_tail = nullptr;
            for (int s = 0; s < n_segments; ++s) {
                const int tail = offsets[(size_t) s] + segments[s].n_tokens - 1;
                ggml_tensor * c_row = ggml_view_2d(ctx, cur, cur->ne[0], 1,
                    cur->nb[1], (size_t) tail * cur->nb[1]);
                ggml_tensor * i_row = ggml_view_2d(ctx, inject, inject->ne[0], 1,
                    inject->nb[1], (size_t) tail * inject->nb[1]);
                ggml_tensor * r_row = ggml_view_3d(ctx, res_hc,
                    res_hc->ne[0], res_hc->ne[1], 1, res_hc->nb[1],
                    res_hc->nb[2], (size_t) tail * res_hc->nb[2]);
                cur_tail = cur_tail ? ggml_concat(ctx, cur_tail, c_row, 1) : c_row;
                inject_tail = inject_tail
                    ? ggml_concat(ctx, inject_tail, i_row, 1) : i_row;
                residual_tail = residual_tail
                    ? ggml_concat(ctx, residual_tail, r_row, 2) : r_row;
            }
            cur = cur_tail;
            inject = inject_tail;
            res_hc = residual_tail;
            layer_rows = n_segments;
        }

        if (hc_fused) {
            ggml_tensor * ffn_fused = hc_combine_norm(ctx, inject, res_hc, cur,
                L.hc_ffn_norm, w.n_embd, w.n_hc, layer_rows, w.rms_eps);
            res_hc = hc_norm_res(ctx, ffn_fused, w.n_embd, w.n_hc, layer_rows);
            ggml_tensor * ffn_xn = hc_norm_xn(ctx, ffn_fused,
                w.n_embd, w.n_hc, layer_rows);
            cur = hc_mix_from_xn(ctx, ffn_xn,
                L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject,
                &inject, w.n_embd, w.n_hc);
        } else {
            res_hc = hc_combine(ctx, res_hc, cur, inject,
                                w.n_embd, w.n_hc, layer_rows);
            cur = hc_mix(ctx, res_hc, L.hc_ffn_norm,
                L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject,
                &inject, w.n_embd, w.n_hc, w.rms_eps);
        }
        cur = build_moe(ctx, cur, L, w, il);
        const bool next_ple = il + 1 < w.n_layer &&
                              w.layers[il + 1].is_ple && has_ple;
        if (next_ple) {
            res_hc = hc_combine(ctx, res_hc, cur, inject, w.n_embd, w.n_hc,
                                layer_rows);
            xn_next = nullptr;
        } else if (hc_fused) {
            ggml_tensor * gamma = il + 1 < w.n_layer
                ? w.layers[il + 1].hc_attn_norm : w.output_hc_norm;
            ggml_tensor * f = hc_combine_norm(ctx, inject, res_hc, cur,
                gamma, w.n_embd, w.n_hc, layer_rows, w.rms_eps);
            res_hc = hc_norm_res(ctx, f, w.n_embd, w.n_hc, layer_rows);
            xn_next = hc_norm_xn(ctx, f, w.n_embd, w.n_hc, layer_rows);
        } else {
            res_hc = hc_combine(ctx, res_hc, cur, inject,
                                w.n_embd, w.n_hc, layer_rows);
            xn_next = nullptr;
        }
    }

    ggml_tensor * final = xn_next
        ? hc_mix_from_xn(ctx, xn_next, w.output_hc_down, w.output_hc_up,
                         nullptr, nullptr, w.n_embd, w.n_hc)
        : hc_mix(ctx, res_hc, w.output_hc_norm, w.output_hc_down,
                 w.output_hc_up, nullptr, nullptr,
                 w.n_embd, w.n_hc, w.rms_eps);
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, final);
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    int max_kv = 0;
    for (int s = 0; s < n_segments; ++s)
        max_kv = std::max(max_kv,
            segments[s].pos0 + segments[s].n_tokens);
    trace_batch_widths(gf, "packed", total_rows, max_kv);

    if (!workspace.alloc)
        workspace.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!workspace.alloc || !ggml_gallocr_reserve(workspace.alloc, gf) ||
        !ggml_gallocr_alloc_graph(workspace.alloc, gf)) {
        workspace.planned = false;
        std::fprintf(stderr, "[qwen4exp] packed graph allocation failed (segments=%d rows=%d)\n",
                     n_segments, total_rows);
        return result;
    }
    workspace.planned = true;
    ggml_backend_tensor_set(inp_emb, embeddings.data(), 0,
                            embeddings.size() * sizeof(float));
    if (ple_in)
        ggml_backend_tensor_set(ple_in, ple_data.data(), 0,
                                ple_data.size() * sizeof(float));
    for (int s = 0; s < n_segments; ++s) {
        ggml_backend_tensor_set(position_inputs[(size_t) s],
            positions_host[(size_t) s].data(), 0,
            positions_host[(size_t) s].size() * sizeof(int32_t));
        if (mask_inputs[(size_t) s])
            ggml_backend_tensor_set(mask_inputs[(size_t) s],
                masks_host[(size_t) s].data(), 0,
                masks_host[(size_t) s].size() * sizeof(ggml_fp16_t));
    }
    const bool batch_telemetry = prof_enabled();
    const double compute_begin_s = batch_telemetry ? mono_now_s() : 0.0;
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        workspace.planned = false;
        std::fprintf(stderr, "[qwen4exp] packed graph compute failed\n");
        return result;
    }
    if (batch_telemetry)
        std::fprintf(stderr,
            "[qwen4exp-packed-time] segments=%d rows=%d begin_abs=%.9f done_abs=%.9f\n",
            n_segments, total_rows, compute_begin_s, mono_now_s());

    std::vector<float> packed_logits((size_t) w.n_vocab * n_segments);
    ggml_backend_tensor_get(logits, packed_logits.data(), 0,
                            packed_logits.size() * sizeof(float));
    out_logits.resize((size_t) n_segments);
    for (int s = 0; s < n_segments; ++s) {
        const auto begin = packed_logits.begin() + (size_t) s * w.n_vocab;
        out_logits[(size_t) s].assign(begin, begin + w.n_vocab);
        segments[s].cache->ple_prev = std::move(next_prev[(size_t) s]);
    }
    result.ok = true;
    result.n_tokens = total_rows;
    result.pos0 = segments[0].pos0;
    return result;
}

}  // namespace luce::common
