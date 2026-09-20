// Qwen4Exp forward graph — see qwen4exp_graph.h.
// embed -> hc_init -> per-layer ([PLE], hc_mix(attn) -> linear|QSA -> hc_combine,
// hc_mix(ffn) -> MoE -> hc_combine) -> hc_mix(output) -> lm_head. Single sequence.

#include "qwen4exp_graph.h"

#include "delta_net_chunked.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace dflash::common {
namespace {

static double prof_now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
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

[[maybe_unused]] ggml_tensor * build_moe(ggml_context * c, ggml_tensor * cur,
                        const Qwen4ExpLayer & L, const Qwen4ExpWeights & w, int il,
                        const std::function<void(ggml_tensor *, const char *)> & dump_mark = {}) {
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

    ggml_tensor * shared_gate = ggml_sigmoid(c, mm(c, L.ffn_gate_inp_shexp, cur));
    shared = ggml_mul(c, shared, shared_gate);   // [n_embd,T] * [1,T] broadcasts over dim 0
    dmark(shared, "msh");

    ggml_tensor * moe_out;
    // The fork's fused combine is the validated default (HE 10/10, ~985 t/s prefill);
    // QWEN4EXP_MOE_UNFUSED=1 falls back to upstream's separate rounded multiply over
    // route views, which is ~25% slower on this model.
    static const bool moe_fused = [] {
        const char * v = getenv("QWEN4EXP_UPSTREAM");
        return !(v && std::atoi(v) != 0) && getenv("QWEN4EXP_MOE_UNFUSED") == nullptr;
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

    ggml_tensor * conv = ggml_silu(c, ggml_ssm_conv(c, conv_input, L.ssm_conv1d));
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
    // that seeds the GDN recurrence and flips MoE routing; keep it only behind an env gate.
    static const bool gdnl2_legacy = getenv("QWEN4EXP_GDNL2_LEGACY") != nullptr;
    ggml_tensor * q_c;
    ggml_tensor * k_c;
    if (gdnl2_legacy) {
        q_c = ggml_l2_norm(c, q_raw, eps);
        k_c = ggml_l2_norm(c, k_raw, eps);
    } else {
        q_c = ggml_scale(c, ggml_rms_norm(c, q_raw, eps / (float) D), 1.0f / sqrtf((float) D));
        k_c = ggml_scale(c, ggml_rms_norm(c, k_raw, eps / (float) D), 1.0f / sqrtf((float) D));
    }
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

// The zero-padded tail block (T % r != 0) is never selectable: complete-block visibility masks its cells.
static ggml_tensor * build_indexer_pooled(ggml_context * c, ggml_tensor * cur,
        const Qwen4ExpLayer & L, const Qwen4ExpWeights & w, int64_t pos0, int64_t r) {
    const int64_t idim = w.indexer_head_size;
    const int64_t T    = cur->ne[1];
    const int64_t nb   = (T + r - 1) / r;
    const float   eps  = w.rms_eps;
    int sections[4] = { w.rope_sections[0], w.rope_sections[1], w.rope_sections[2], w.rope_sections[3] };

    ggml_tensor * kraw = mm(c, L.indexer_k_proj, cur);            // [idim, T]
    if (nb * r > T) {
        ggml_tensor * z = ggml_reshape_2d(c,
            ggml_scale_bias(c, ggml_arange(c, 0.0f, (float) (idim * (nb * r - T)), 1.0f), 0.0f, 0.0f),
            idim, nb * r - T);
        kraw = ggml_concat(c, kraw, z, 1);
    }
    kraw = ggml_reshape_3d(c, kraw, idim, nb, r);
    ggml_tensor * pooled = nullptr;
    for (int64_t i = 0; i < r; ++i) {
        ggml_tensor * sl = ggml_cont(c, ggml_view_3d(c, kraw, idim, nb, 1,
            kraw->nb[1], kraw->nb[2], i * kraw->nb[2]));
        pooled = pooled ? ggml_add(c, pooled, sl) : sl;
    }
    pooled = ggml_scale(c, pooled, 1.0f / (float) r);
    pooled = ggml_mul(c, ggml_rms_norm(c, pooled, eps), L.indexer_k_norm);

    ggml_tensor * bp = ggml_scale_bias(c,
        ggml_scale(c, ggml_arange(c, 0.0f, (float) nb, 1.0f), (float) r), 1.0f, (float) pos0);
    ggml_tensor * bp_i = ggml_cast(c, bp, GGML_TYPE_I32);
    ggml_tensor * bp_z = ggml_cast(c, ggml_scale(c, bp, 0.0f), GGML_TYPE_I32);
    ggml_tensor * bpos = ggml_concat(c, ggml_concat(c, bp_i, bp_i, 0),
                                        ggml_concat(c, bp_i, bp_z, 0), 0);
    // rope indexes its tokens on ne[2], so put the block axis there.
    pooled = ggml_rope_multi(c, ggml_reshape_3d(c, pooled, idim, 1, nb), bpos, nullptr,
        w.rope_dimension_count, sections, GGML_ROPE_TYPE_MROPE, 0, w.rope_theta,
        1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    return ggml_reshape_3d(c, pooled, idim, nb, 1);
}

static ggml_tensor * build_qsa_attn(ggml_context * c, ggml_tensor * cur,
        ggml_tensor * Q, ggml_tensor * Kf, ggml_tensor * Vf,
        const Qwen4ExpLayer & L, const Qwen4ExpWeights & w, int64_t ratio,
        ggml_tensor * positions, int64_t kv_pad, int64_t kv_len, int64_t kv_start,
        ggml_tensor * pooled, int64_t nb) {
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

    ggml_tensor * blocks = ggml_cont(c, ggml_top_k(c, summed, (int) budget));   // [budget, T]
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
    ggml_tensor * ids = ggml_reshape_2d(c,
        ggml_concat(c, cells, ggml_reshape_3d(c, rows, r - 1, T, 1), 0),
        budget * r + (r - 1), T);

    ggml_tensor * q3 = ggml_cont(c, ggml_permute(c, Q, 0, 2, 1, 3));       // [D, T, Hq]
    ggml_tensor * Kc = ggml_is_contiguous(Kf) ? Kf : ggml_cont(c, Kf);
    ggml_tensor * Vc = ggml_is_contiguous(Vf) ? Vf : ggml_cont(c, Vf);
    ggml_tensor * attn = ggml_flash_attn_ext(c, q3, Kc, Vc, nullptr, qscale, 0.0f, 0.0f);
    attn->src[5] = ids;
    attn->src[6] = qsa_pack_keys(c, Kc);
    attn->src[7] = qsa_pack_values(c, Vc);
    ggml_flash_attn_ext_set_n_kv_max(attn, (int32_t) ids->ne[0]);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    return attn;
}

static bool indexer_store_ok(ggml_tensor * indexer_k, int64_t T, int64_t pos0, int64_t ratio) {
    static const bool use_qsa = getenv("QWEN4EXP_QSA") != nullptr;
    if (!use_qsa || T < 128 || ratio <= 1 || ratio % 4 != 0 || pos0 % ratio != 0) return false;
    const int64_t off    = pos0 / ratio;
    const int64_t nb_cur = (T + ratio - 1) / ratio;
    return indexer_k != nullptr && off + nb_cur <= indexer_k->ne[1];
}

// indexer_written bounds the populated prefix; selected-block scoring must never read uninitialised columns.
static bool qsa_layer_ok(const Qwen4ExpWeights & w, ggml_tensor * k_cache,
                         ggml_tensor * indexer_k, int64_t T, int64_t kv_len,
                         int64_t pos0, int64_t ratio, int64_t Hq, int64_t Hk,
                         int64_t indexer_written) {
    static const bool use_qsa = getenv("QWEN4EXP_QSA") != nullptr;
    if (!use_qsa || T < 128 || ratio <= 1) return false;
    if (w.indexer_head_size != 128 || w.indexer_n_head <= 0 ||
        w.indexer_top_k % ratio != 0) return false;
    if ((w.indexer_top_k / ratio) * ratio + (ratio - 1) > 2560 || Hq != 12 * Hk) return false;
    const int64_t qsa_step = (ratio % 4 == 0) ? ratio : (ratio % 2 == 0 ? ratio * 2 : ratio * 4);
    const int64_t kv_pad   = (kv_len + qsa_step - 1) / qsa_step * qsa_step;
    if (kv_pad > k_cache->ne[1]) return false;
    if (!indexer_store_ok(indexer_k, T, pos0, ratio)) return false;
    const int64_t budget = w.indexer_top_k / ratio;
    const int64_t nb_cur = (T + ratio - 1) / ratio;
    const int64_t off    = pos0 / ratio;
    if (off + nb_cur < budget) return false;
    if (off > indexer_written) return false;   // prefix not fully populated
    return true;
}

ggml_tensor * build_full_attn(ggml_context * c, ggml_cgraph * gf, ggml_tensor * cur,
                              const Qwen4ExpLayer & L, const Qwen4ExpWeights & w,
                              ggml_tensor * k_cache, ggml_tensor * v_cache,
                              ggml_tensor * indexer_k,
                              ggml_tensor * positions, ggml_tensor * mask, ggml_tensor * kv_row,
                              int64_t kv_len, int64_t pos0, int64_t ratio,
                              int64_t indexer_written, int il,
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

    // Pad K/V to a multiple of lcm(4, ratio) so QSA runs at any prompt length instead of falling back to dense.
    const int64_t qsa_step = (ratio % 4 == 0) ? ratio : (ratio % 2 == 0 ? ratio * 2 : ratio * 4);
    const int64_t kv_pad   = (kv_len + qsa_step - 1) / qsa_step * qsa_step;

    const int64_t nb_cur = (T + ratio - 1) / ratio;   // blocks in this chunk
    const int64_t off    = pos0 / ratio;              // cached blocks before it
    const bool store = indexer_store_ok(indexer_k, T, pos0, ratio);
    const bool qsa_ok = qsa_layer_ok(w, k_cache, indexer_k, T, kv_len, pos0, ratio, Hq, Hk, indexer_written);
    // qwen4exp_forward elides the dense causal mask only when every full layer takes QSA; a dense fallback without one must fail loudly.
    if (T > 1 && mask == nullptr && !qsa_ok) {
        std::fprintf(stderr,
            "[qwen4exp] dense attention without a causal mask (T=%lld pos0=%lld)\n",
            (long long) T, (long long) pos0);
        std::abort();
    }

    // Persist this chunk's indexer keys even if QSA is not selected: later chunks read blocks [0, off).
    ggml_tensor * pooled_cur = nullptr;
    if (store) {
        pooled_cur = build_indexer_pooled(c, cur, L, w, pos0, ratio);
        ggml_build_forward_expand(gf, ggml_cpy(c,
            ggml_reshape_2d(c, pooled_cur, w.indexer_head_size, nb_cur),
            ggml_view_2d(c, indexer_k, w.indexer_head_size, nb_cur,
                         indexer_k->nb[1], (size_t) off * indexer_k->nb[1])));
    }

    ggml_tensor * attn;
    if (qsa_ok) {
        GGML_ASSERT(pooled_cur != nullptr);
        ggml_tensor * pooled = pooled_cur;
        if (off > 0) {
            ggml_tensor * prefix = ggml_view_2d(c, indexer_k, w.indexer_head_size, off,
                                                indexer_k->nb[1], 0);
            pooled = ggml_reshape_3d(c, ggml_concat(c, prefix,
                ggml_reshape_2d(c, pooled_cur, w.indexer_head_size, nb_cur), 1),
                w.indexer_head_size, off + nb_cur, 1);
        }

        ggml_tensor * K_pad = (kv_pad != kv_len)
            ? ggml_view_3d(c, k_cache, D, kv_pad, Hk, k_cache->nb[1], k_cache->nb[2], 0) : K_full;
        ggml_tensor * V_pad = (kv_pad != kv_len)
            ? ggml_view_3d(c, v_cache, D, kv_pad, Hk, v_cache->nb[1], v_cache->nb[2], 0) : V_full;
        attn = build_qsa_attn(c, cur, Q, K_pad, V_pad, L, w, ratio, positions,
                              kv_pad, kv_len, pos0, pooled, off + nb_cur);
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
    static const bool prof = getenv("QWEN4EXP_PROF") != nullptr;
    // QWEN4EXP_DUMP=1 materialises per-layer activations and prints finite/absmax/mean.
    static const bool dump = getenv("QWEN4EXP_DUMP") != nullptr;
    // The fork's fused hc-combine-norm is the validated default (HE 10/10, ~985 t/s
    // prefill); its 256-thread sum-of-squares reduction differs sub-ulp from
    // upstream's ggml_rms_norm. QWEN4EXP_HC_UNFUSED=1 selects the upstream two-op
    // sequence, which is ~25% slower on this model.
    static const bool upstream = [] { const char * v = getenv("QWEN4EXP_UPSTREAM"); return v && std::atoi(v) != 0; }();
    static const bool hc_fused = !upstream && getenv("QWEN4EXP_HC_UNFUSED") == nullptr;
    static const bool decode_reuse = [] {
        const char * value = getenv("QWEN4EXP_DECODE_REUSE");
        return !(value && std::atoi(value) == 0);
    }();
    static const bool stable_graph = [] {
        const char * value = getenv("QWEN4EXP_DECODE_STABLEGRAPH");
        return value && std::atoi(value) != 0;
    }();
    static const bool stable_telemetry = [] {
        const char * value = getenv("QWEN4EXP_STABLEGRAPH_TELEMETRY");
        return value && std::atoi(value) != 0;
    }();
    const bool reuse_decode_workspace = decode_reuse && !upstream && n_tokens == 1 && !dump;
    const bool use_stable_graph = stable_graph && reuse_decode_workspace;
    std::vector<std::pair<ggml_tensor *, std::string>> dump_t;
    auto dump_mark = [&](ggml_tensor * t, const char * label) {
        if (dump && t) {
            // A view output must keep its owning allocation alive until the dump.
            for (ggml_tensor * base = t; base; base = base->view_src) ggml_set_output(base);
            ggml_set_name(t, label);
            dump_t.emplace_back(t, label);
        }
    };
    const double t0 = prof ? prof_now_ms() : 0.0;
    double t_emb = t0, t_ple = t0, t_alloc = t0, t_upload = t0, t_compute = t0;
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
    if (prof) t_emb = prof_now_ms();

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

    if (prof) t_ple = prof_now_ms();
    const int64_t T = n_tokens;
    const int64_t kv_len = pos0 + n_tokens;
    static const bool fa_pad256 = upstream || [] { const char * v = getenv("QWEN4EXP_FA_PAD256"); return v && std::atoi(v) != 0; }();
    static const bool dev_mask = getenv("QWEN4EXP_KQ_MASK_DEV") != nullptr;
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
        if (prof) t_upload = prof_now_ms();
        if (ggml_backend_graph_compute(backend, ws.gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[qwen4exp] stable graph compute failed\n");
            return false;
        }
        out_logits.resize((size_t) w.n_vocab);
        ggml_backend_tensor_get(ws.logits, out_logits.data(), 0, sizeof(float) * w.n_vocab);
        ws.stable_calls++;
        if (stable_telemetry && (ws.stable_calls <= 4 || ws.stable_calls % 128 == 0)) {
            std::fprintf(stderr, "[qwen4exp-stable] event=reuse bucket=%lld calls=%llu\n",
                         (long long) ws.kv_bucket, (unsigned long long) ws.stable_calls);
        }
        if (prof) {
            t_compute = prof_now_ms();
            std::fprintf(stderr,
                "[qwen4exp-prof] T=1 stable=1 embed=%.1fms ple=%.1fms build+alloc=0.0ms mask+upload=%.1fms compute=%.1fms get=%.1fms total=%.1fms\n",
                t_emb - t0, t_ple - t_emb, t_upload - t_ple,
                t_compute - t_upload, prof_now_ms() - t_compute, prof_now_ms() - t0);
        }
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
        if (stable_telemetry) {
            std::fprintf(stderr, "[qwen4exp-stable] event=rebuild old_bucket=%lld new_bucket=%lld\n",
                         (long long) decode_ws.kv_bucket, (long long) stable_kv_bucket);
        }
        clear_qwen4exp_decode_workspace(decode_ws);
    }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 200000 +
                  ggml_graph_overhead_custom(200000, false) + (1u << 20);
    ip.no_alloc = true;
    ggml_context * ctx = nullptr;
    if (reuse_decode_workspace) {
        if (cache.decode_workspace.ctx == nullptr) {
            cache.decode_workspace.ctx = ggml_init(ip);
        } else {
            ggml_reset(cache.decode_workspace.ctx);
        }
        ctx = cache.decode_workspace.ctx;
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
    // QSA derives its own complete-block visibility; when every full layer takes it the dense [kv_len, T] mask is never read.
    int full0 = -1;
    for (int il = 0; il < w.n_layer && full0 < 0; ++il) {
        if (w.layers[il].is_full_attention) full0 = il;
    }
    bool qsa_all = false;
    // Capture before advancing so the per-layer guard bounds the prefix this chunk reads.
    const int indexer_written = cache.indexer_blocks;
    // Commit only after the graph computes: a failed alloc/compute must not mark columns the kernel never wrote.
    int next_indexer_blocks = indexer_written;
    if (T > 1 && full0 >= 0) {
        const int fi = full_idx[full0];
        const int64_t ratio = full0 < (int) w.compress_ratios.size() ? w.compress_ratios[full0] : 0;
        qsa_all = qsa_layer_ok(w, cache.attn_k[fi], cache.indexer_k[fi],
                               T, kv_len, pos0, ratio, w.n_head, w.n_head_kv, indexer_written);
        if (indexer_store_ok(cache.indexer_k[fi], T, pos0, ratio)) {
            // Only extend on a contiguous chunk; a gap leaves the cache unusable so later chunks fall back to dense.
            const int off = (int) (pos0 / ratio);
            const int end = (int) (pos0 / ratio + (T + ratio - 1) / ratio);
            next_indexer_blocks = (indexer_written == off) ? end : -1;
        }
    }
    if ((T > 1 || fa_pad256 || use_stable_graph) && !qsa_all) {
        if (!dev_mask || use_stable_graph) {
            mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, mask_len, T);
            ggml_set_input(mask);
        } else {
            ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mask_len, T);
            m = ggml_fill_inplace(ctx, m, 0.0f);
            m = ggml_diag_mask_inf_inplace(ctx, m, (int) pos0);
            mask = ggml_cast(ctx, m, GGML_TYPE_F16);
        }
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
                                  positions, mask, kv_row, graph_kv_len, pos0,
                                  il < (int) w.compress_ratios.size() ? w.compress_ratios[il] : 0,
                                  indexer_written, il, dump_mark);
        } else {
            const int li = lin_idx[il];
            cur = build_linear_attn(ctx, gf, cur, L, w,
                                    cache.ssm_state[li], cache.conv_state[li], il, dump_mark);
        }
        std::snprintf(dlab, sizeof dlab, "L%02d.att", il);
        dump_mark(cur, dlab);
        int64_t layer_T = T;
        static const bool last_token_ffn = [] {
            const char * value = getenv("QWEN4EXP_LAST_TOKEN_FFN");
            return !(value && std::atoi(value) == 0);
        }();
        if ((upstream || last_token_ffn) && il == w.n_layer - 1 && T > 1) {
            // Upstream selects output rows before the final HC/FFN, so its
            // quantized matmuls dispatch with one token (MMV rather than MMQ).
            // The performance opt-in reuses that selection after all attention
            // cache writes. Earlier rows have no remaining stateful consumers.
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
        cur = build_moe(ctx, cur, L, w, il, dump_mark);
        std::snprintf(dlab, sizeof dlab, "L%02d.moe", il);
        dump_mark(cur, dlab);

        const bool next_ple = (il + 1 < w.n_layer) && w.layers[il + 1].is_ple && has_ple;
        if (next_ple) {
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
        const size_t mask_need = (!dev_mask && mask)
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
            if (mask && !dev_mask) {
                ggml_backend_tensor_alloc(cache.input_ring.buf, mask, ring_mask);
            }
            if (ple_in) {
                ggml_backend_tensor_alloc(cache.input_ring.buf, ple_in, ring_ple);
            }
        }
    }

    ggml_gallocr_t galloc = nullptr;
    if (reuse_decode_workspace) {
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
        if (!reuse_decode_workspace) ggml_free(ctx);
        return res;
    }
    // T=1 graphs keep the same broad shape but advancing KV views can change
    // lifetimes. Recompute assignments while retaining the allocator buffers;
    // reusing the old index-wise plan produced incorrect tokens.
    const bool reserve_ok = !reuse_decode_workspace ||
        !cache.decode_workspace.planned || ggml_gallocr_reserve(galloc, gf);
    if (!reserve_ok || !ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "[qwen4exp] graph alloc failed (T=%lld kv_len=%lld)\n",
                     (long long) T, (long long) kv_len);
        if (reuse_decode_workspace) cache.decode_workspace.planned = false;
        if (!reuse_decode_workspace) {
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
        }
        return res;
    }
    if (reuse_decode_workspace) cache.decode_workspace.planned = true;

    if (use_stable_graph) {
        decode_ws.gf = gf;
        decode_ws.inp_emb = inp_emb;
        decode_ws.positions = positions;
        decode_ws.mask = mask;
        decode_ws.ple_in = ple_in;
        decode_ws.kv_row = kv_row;
        decode_ws.logits = logits;
        decode_ws.kv_bucket = stable_kv_bucket;
        decode_ws.stable_calls = 0;
        if (stable_telemetry) {
            std::fprintf(stderr, "[qwen4exp-stable] event=build bucket=%lld nodes=%d\n",
                         (long long) stable_kv_bucket, ggml_graph_n_nodes(gf));
        }
    }

    if (prof) t_alloc = prof_now_ms();
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
    if ((!dev_mask || use_stable_graph) && mask) {
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
        if (mask && !dev_mask) {
            std::memcpy(mask->data, m.data(), sizeof(ggml_fp16_t) * m.size());
        }
        if (ple_in) {
            std::memcpy(ple_in->data, ple_data.data(), sizeof(float) * ple_data.size());
        }
    } else {
        ggml_backend_tensor_set(inp_emb, emb.data(), 0, sizeof(float) * emb.size());
        ggml_backend_tensor_set(positions, pos.data(), 0, sizeof(int32_t) * pos.size());
        if (!dev_mask && mask) {
            ggml_backend_tensor_set(mask, m.data(), 0, sizeof(ggml_fp16_t) * m.size());
        }
        if (ple_in) {
            ggml_backend_tensor_set(ple_in, ple_data.data(), 0, sizeof(float) * ple_data.size());
        }
    }

    if (prof) t_upload = prof_now_ms();
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[qwen4exp] graph compute failed\n");
        if (use_stable_graph) {
            clear_qwen4exp_decode_workspace(decode_ws);
        } else if (!reuse_decode_workspace) {
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
        }
        return res;
    }
    cache.indexer_blocks = next_indexer_blocks;

    out_logits.resize((size_t) w.n_vocab);
    ggml_backend_tensor_get(logits, out_logits.data(), 0, sizeof(float) * w.n_vocab);
    if (use_stable_graph) {
        decode_ws.stable_calls++;
        if (stable_telemetry) {
            std::fprintf(stderr, "[qwen4exp-stable] event=first bucket=%lld calls=%llu\n",
                         (long long) decode_ws.kv_bucket,
                         (unsigned long long) decode_ws.stable_calls);
        }
    }
    if (prof) {
        t_compute = prof_now_ms();
        std::fprintf(stderr,
            "[qwen4exp-prof] T=%d embed=%.1fms ple=%.1fms build+alloc=%.1fms mask+upload=%.1fms compute=%.1fms get=%.1fms total=%.1fms\n",
            n_tokens, t_emb - t0, t_ple - t_emb, t_alloc - t_ple, t_upload - t_alloc,
            t_compute - t_upload, prof_now_ms() - t_compute, prof_now_ms() - t0);
    }

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

    if (!reuse_decode_workspace) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
    }

    res.ok = true;
    res.n_tokens = n_tokens;
    res.pos0 = pos0;
    return res;
}

}  // namespace dflash::common
