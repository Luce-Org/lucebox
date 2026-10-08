// Qwen4Exp forward graph — see qwen4exp_graph.h.
// embed -> hc_init -> per-layer ([PLE], hc_mix(attn) -> linear|QSA -> hc_combine,
// hc_mix(ffn) -> MoE -> hc_combine) -> hc_mix(output) -> lm_head. Single sequence.

#include "qwen4exp_graph.h"

#include "common/cuda_graph_overrides.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace luce::common {
namespace {

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

ggml_tensor * build_moe(ggml_context * c, ggml_tensor * cur,
                        const Qwen4ExpLayer & L, const Qwen4ExpWeights & w,
                        Qwen4ExpMoeParts * parts = nullptr) {
    const int64_t n_embd   = w.n_embd;
    const int64_t n_tokens = cur->ne[1];
    const int64_t n_expert = w.n_expert;
    const int64_t n_used   = w.n_expert_used;

    ggml_tensor * logits = mm(c, L.ffn_gate_inp, cur);      // [n_expert, T]
    ggml_tensor * probs  = ggml_soft_max(c, logits);
    ggml_tensor * sel    = ggml_argsort_top_k(c, probs, (int) n_used);  // [n_used, T]

    ggml_tensor * probs3 = ggml_reshape_3d(c, probs, 1, n_expert, n_tokens);
    ggml_tensor * wsel   = ggml_reshape_2d(c, ggml_get_rows(c, probs3, sel), n_used, n_tokens);
    wsel = ggml_div(c, wsel, ggml_clamp(c, ggml_sum_rows(c, wsel), 6.103515625e-5f, INFINITY));

    ggml_tensor * cur3 = ggml_reshape_3d(c, cur, n_embd, 1, n_tokens);

    ggml_tensor * gate = ggml_mul_mat_id(c, L.ffn_gate_exps, cur3, sel);
    ggml_tensor * up   = ggml_mul_mat_id(c, L.ffn_up_exps,   cur3, sel);
    ggml_tensor * gu   = ggml_swiglu_split(c, gate, up);

    ggml_tensor * down = ggml_mul_mat_id(c, L.ffn_down_exps, gu, sel);   // [n_embd, n_used, T]

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
    for (int64_t t = 0; spec_conv && t < T; ++t) {
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
    // The kernel writes them straight to spec_states (same F32 transposed layout as the state, token-major); the
    // packed result stays compact because skip was set first.
    if (spec_states) gdn->src[7] = spec_states;

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
    if (ggml_backend_cuda_mmb_f16_input_ok(L.ssm_out, T)) {
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

int64_t qwen4exp_stable_kv_span(int64_t & base, int64_t max_ctx, int64_t kv_len) {
    if (base == 0) base = std::min<int64_t>(max_ctx, ((kv_len + 511) / 256) * 256);
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
static ggml_tensor * qsa_pooled_keys_stable(ggml_context * c, const Qwen4ExpLayer & L,
        const Qwen4ExpWeights & w, ggml_tensor * indexer_k, ggml_tensor * raw_written,
        const Qwen4ExpDecodeWorkspace & ws) {
    ggml_tensor * rows = ggml_view_1d(c, ws.qsa_params, 4, sizeof(int32_t));
    ggml_tensor * row = ggml_view_1d(c, ws.qsa_params, 1, 5 * sizeof(int32_t));
    ggml_tensor * pos = ggml_view_1d(c, ws.qsa_params, 4, 6 * sizeof(int32_t));
    ggml_tensor * span = ggml_get_rows(c, raw_written, rows);
    ggml_tensor * fresh = qsa_pool_norm_rope(c, L, w, qwen4exp_pool_blocks(c, span, 4), pos);
    ggml_tensor * written = ggml_set_rows(c, indexer_k, fresh, row);
    return ggml_view_3d(c, written, w.indexer_head_size, ws.qsa_blocks, 1,
        written->nb[1], written->nb[2], 0);
}

static ggml_tensor * build_qsa_attn(ggml_context * c, ggml_tensor * cur,
        ggml_tensor * Q, ggml_tensor * Kf, ggml_tensor * Vf,
        const Qwen4ExpLayer & L, const Qwen4ExpWeights & w, int64_t ratio,
        ggml_tensor * positions, int64_t kv_start,
        ggml_tensor * pooled, int64_t nb, bool packed, const Qwen4ExpDecodeWorkspace * ws = nullptr) {
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
        ggml_tensor * hw = ggml_reshape_2d(c,
            ggml_scale_bias(c, ggml_scale(c, ggml_arange(c, 0.0f, (float) (nih * T), 1.0f), 0.0f), 0.0f, 1.0f),
            nih, T);
        ggml_tensor * summed = ws
            ? ggml_ds4_indexer_score_masked(c, qi, hw, comp16, ws->qsa_visibility, 0, (int) r)
            : ggml_ds4_indexer_score(c, qi, hw, comp16, (int) kv_start, (int) r);
        blocks = ws
            ? ggml_top_k_qsa(c, summed, ggml_view_1d(c, ws->qsa_params, 1, 0),
                            (int) std::max<int64_t>(513, (ws->kv_bucket - 256) / 4))
            : ggml_top_k(c, summed, (int) budget);
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
struct Qwen4ExpAttnRow {
    Qwen4ExpQsaMode qsa       = QSA_DENSE;
    int64_t         span      = 0;         // dense: K/V span of the stable T=1 graph
    ggml_tensor *   mask      = nullptr;   // dense: [span, 1]
    ggml_tensor *   positions = nullptr;   // QSA: this row's M-RoPE positions
};

static ggml_tensor * write_indexer_keys(ggml_context * c, ggml_cgraph * gf,
        ggml_tensor * cur, const Qwen4ExpLayer & L, ggml_tensor * raw,
        int64_t pos0, ggml_tensor * kv_row = nullptr) {
    if (!raw) return nullptr;
    ggml_tensor * keys = mm(c, L.indexer_k_proj, cur);
    ggml_build_forward_expand(gf, kv_row
        ? ggml_set_rows(c, raw, keys, kv_row)
        : ggml_cpy(c, keys, ggml_view_2d(c, raw, keys->ne[0], keys->ne[1], raw->nb[1],
                                        (size_t) pos0 * raw->nb[1])));
    return keys;
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
    const float   eps    = w.rms_eps;

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
    ggml_tensor * V = ggml_reshape_3d(c, mm(c, L.wv, cur), D, Hk, T);

    int sections[4] = { w.rope_sections[0], w.rope_sections[1],
                        w.rope_sections[2], w.rope_sections[3] };
    Q = ggml_rope_multi(c, Q, positions, nullptr, w.rope_dimension_count, sections,
                        GGML_ROPE_TYPE_MROPE, 0, w.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_multi(c, K, positions, nullptr, w.rope_dimension_count, sections,
                        GGML_ROPE_TYPE_MROPE, 0, w.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

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
    if (last_only && T > 1) {
        // All retained pairs keep their original K/V projection width above.
        // Only the final query feeds the next proposal; its prefix is causal.
        GGML_ASSERT(qsa == QSA_DENSE && !rows && !indexer_raw && !kv_row);
        Q = ggml_view_3d(c, Q, D, Hq, 1, Q->nb[1], Q->nb[2], (T - 1) * Q->nb[2]);
        gate = ggml_view_2d(c, gate, D * Hq, 1, gate->nb[1], (T - 1) * gate->nb[1]);
        qfull = ggml_view_2d(c, qfull, qfull->ne[0], 1, qfull->nb[1], (T - 1) * qfull->nb[1]);
        T = 1;
        mask = nullptr;
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
        ggml_tensor * pooled = n_after > 0
            ? qsa_pooled_keys(c, gf, L, w, indexer_k, indexer_raw, kraw, pos0, ratio, n_pooled, n_after) : nullptr;
        for (int64_t t = 0; t < T; ++t) {
            const Qwen4ExpAttnRow & row = (*rows)[t];
            ggml_tensor * q = ggml_view_3d(c, Q, D, Hq, 1, Q->nb[1], Q->nb[2], (size_t) t * Q->nb[2]);
            const int64_t span = row.qsa == QSA_DENSE ? row.span : pos0 + t + 1;
            ggml_tensor * Kr = ggml_view_3d(c, k_cache, D, span, Hk, k_cache->nb[1], k_cache->nb[2], 0);
            ggml_tensor * Vr = ggml_view_3d(c, v_cache, D, span, Hk, v_cache->nb[1], v_cache->nb[2], 0);
            ggml_tensor * a;
            if (row.qsa == QSA_DENSE) {
                a = ggml_flash_attn_ext(c, ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3)), Kr, Vr, row.mask,
                    1.0f / std::sqrt((float) D), 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);
            } else {
                const int64_t nb = span / ratio;
                a = build_qsa_attn(c, ggml_view_2d(c, cur, cur->ne[0], 1, cur->nb[1], (size_t) t * cur->nb[1]), q, Kr, Vr,
                    L, w, ratio, row.positions, pos0 + t,
                    ggml_view_3d(c, pooled, pooled->ne[0], nb, 1, pooled->nb[1], pooled->nb[2], 0), nb, false);
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
            ? qsa_pooled_keys_stable(c, L, w, indexer_k, indexer_raw, *qsa_ws)
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
            attn = build_qsa_attn(c, cur, Q, K_full, V_full, L, w, ratio, positions,
                                  pos0, pooled, qsa_ws ? qsa_ws->qsa_blocks : n_after, false, qsa_ws);
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
    if (ggml_backend_cuda_mmb_f16_input_ok(L.wo, T)) {
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

static ggml_tensor * build_full_attn_projected(ggml_context * c, ggml_cgraph * gf,
        ggml_tensor * qfull, ggml_tensor * kraw, ggml_tensor * vraw,
        ggml_tensor * positions, ggml_tensor * mask, const Qwen4ExpLayer & L,
        const Qwen4ExpWeights & w, ggml_tensor * k_cache, ggml_tensor * v_cache,
        int pos0, int64_t kv_view_len, ggml_tensor * cur, ggml_tensor * indexer_raw) {
    // Like dense solo, retain raw keys and leave pooling/indexer_blocks lazy until QSA runs.
    write_indexer_keys(c, gf, cur, L, indexer_raw, pos0);
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
    const Qwen4ExpCudaScope profile(w.gfx1151);
    if (verify && (n_tokens < 2 || n_tokens > cache.mtp_draft + 1 || !qwen4exp_verify_supported(cache))) return res;
    if (mtp_prefill && (n_tokens <= 1 || verify || !out_hidden ||
        !qwen4exp_verify_supported(cache) || !cache.mtp_prev_hidden ||
        (!measure && pos0 > 0 && cache.mtp_prev_pos != pos0 - 1))) return res;
    if (!measure) cache.spec_tokens = 0;
    const Qwen4ExpQsaMode qsa = qsa_mode(w, cache, verify ? 1 : n_tokens, pos0, profile.optimized);
    const bool reuse_ws = n_tokens == 1;
    const int64_t logical_blocks = qsa == QSA_DENSE ? -1 : (int64_t(pos0) + n_tokens) / qsa_ratio(w);
    bool stable_qsa = reuse_ws && qsa == QSA_DECODE &&
        qsa_ratio(w) == 4 && w.indexer_top_k == 2048 && pos0 < 262144;
    if (stable_qsa) {
        // No allocation/host readback: ask the backend about the runtime-count
        // variant. In particular, nondeterministic CUDA DeviceTopK is excluded.
        ggml_tensor scores{}, valid{}, selection{};
        scores.type = GGML_TYPE_F32;
        scores.ne[0] = std::min<int64_t>(cache.max_ctx, ((int64_t(pos0) + 256) / 256) * 256) / 4;
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
        stable_qsa = ggml_backend_supports_op(backend, &selection);
    }
    if (stable_qsa) {
        for (int il = 0; il < w.n_layer; ++il) {
            if (w.layers[il].is_full_attention &&
                (il >= (int) w.compress_ratios.size() || w.compress_ratios[il] != 4)) stable_qsa = false;
        }
    }
    const bool use_stable_graph = reuse_ws && (qsa == QSA_DENSE || stable_qsa);
    // Context and allocator reused across calls: the T=1 decode workspace, or the verify forward's own.
    Qwen4ExpDecodeWorkspace * pool = measure ? nullptr : reuse_ws ? &cache.decode_workspace : verify ? &cache.verify_workspace : nullptr;
    if (pos0 > cache.max_ctx || n_tokens > cache.max_ctx - pos0) {
        std::fprintf(stderr, "[qwen4exp] context overflow: %d + %d > %d\n",
                     pos0, n_tokens, cache.max_ctx);
        return res;
    }

    std::vector<int> lin_idx(w.n_layer, -1);
    std::vector<int> full_idx(w.n_layer, -1);
    for (size_t i = 0; i < cache.linear_layer_ids.size(); ++i) lin_idx[cache.linear_layer_ids[i]] = (int) i;
    for (size_t i = 0; i < cache.full_layer_ids.size(); ++i)   full_idx[cache.full_layer_ids[i]] = (int) i;
    std::vector<int> ple_idx(w.n_layer, -1);
    for (size_t i = 0; i < cache.ple_layer_ids.size(); ++i)    ple_idx[cache.ple_layer_ids[i]] = (int) i;

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
    // Give a new stable graph at least one full 256-token generation window.
    // The fixed mask excludes its padded tail, while the stable K/V views and
    // set_rows index keep every graph pointer and property unchanged.
    Qwen4ExpDecodeWorkspace measured_decode_ws;
    Qwen4ExpDecodeWorkspace & decode_ws = measure ? measured_decode_ws : cache.decode_workspace;
    if (!measure && decode_ws.ctx && (!reuse_ws || decode_ws.backend != backend || decode_ws.model != &w ||
                          decode_ws.max_ctx != cache.max_ctx)) {
        clear_qwen4exp_decode_workspace(decode_ws);
    }
    const int64_t bucket_limit = stable_qsa ? std::min(cache.max_ctx, 262144) : cache.max_ctx;
    const int64_t stable_kv_bucket = stable_qsa
        ? std::min<int64_t>(bucket_limit, ((kv_len + 255) / 256) * 256)
        : use_stable_graph
            ? qwen4exp_stable_kv_span(cache.kv_bucket_base, cache.max_ctx, kv_len)
            : 0;

    // Storage survives until graph_compute/get completes the asynchronous uploads.
    std::vector<float> qsa_visibility;
    int32_t qsa_params[10] = {};
    auto upload_qsa = [&](Qwen4ExpDecodeWorkspace & ws) {
        if (!stable_qsa) return;
        const int n = (pos0 + 1) / 4, first = 4 * (n - 1);
        qsa_visibility.assign((size_t) ws.qsa_blocks, -INFINITY);
        std::fill_n(qsa_visibility.begin(), n, 0.0f);
        qsa_params[0] = n;
        for (int i = 0; i < 4; ++i) qsa_params[1 + i] = first + i;
        qsa_params[5] = (pos0 + 1) % 4 == 0 ? n - 1 : (cache.max_ctx + 3) / 4;
        qsa_params[6] = qsa_params[7] = qsa_params[8] = first;
        ggml_backend_tensor_set_async(backend, ws.qsa_visibility, qsa_visibility.data(), 0,
                                      qsa_visibility.size() * sizeof(float));
        ggml_backend_tensor_set_async(backend, ws.qsa_params, qsa_params, 0, sizeof(qsa_params));
    };

    auto run_stable = [&](Qwen4ExpDecodeWorkspace & ws) -> bool {
        int32_t pos[4] = { pos0, pos0, pos0, 0 };
        const int32_t kv_row = pos0;
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask_data;
        if (ws.mask) {
            mask_data.assign((size_t) ws.kv_bucket, ninf);
            std::fill(mask_data.begin(), mask_data.begin() + kv_len, zero);
        }

        ggml_backend_tensor_set_async(backend, ws.inp_emb, emb.data(), 0,
                                      sizeof(float) * emb.size());
        ggml_backend_tensor_set_async(backend, ws.positions, pos, 0, sizeof(pos));
        ggml_backend_tensor_set_async(backend, ws.kv_row, &kv_row, 0, sizeof(kv_row));
        if (ws.mask) ggml_backend_tensor_set_async(backend, ws.mask, mask_data.data(), 0,
                                                  sizeof(ggml_fp16_t) * mask_data.size());
        if (ws.ple_in) {
            ggml_backend_tensor_set_async(backend, ws.ple_in, ple_data.data(), 0,
                                          sizeof(float) * ple_data.size());
        }
        upload_qsa(ws);
        if (ggml_backend_graph_compute(backend, ws.gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[qwen4exp] stable graph compute failed\n");
            clear_qwen4exp_decode_workspace(ws);
            return false;
        }
        out_logits.resize((size_t) w.n_vocab);
        ggml_backend_tensor_get(ws.logits, out_logits.data(), 0, sizeof(float) * w.n_vocab);
        if (out_hidden && ws.hidden) {
            out_hidden->resize((size_t) ggml_nelements(ws.hidden));
            ggml_backend_tensor_get(ws.hidden, out_hidden->data(), 0, ggml_nbytes(ws.hidden));
        }
        return true;
    };

    if (use_stable_graph && decode_ws.gf && kv_len <= decode_ws.kv_bucket &&
        (decode_ws.qsa_blocks >= 0) == stable_qsa && decode_ws.next_pos == pos0 &&
        (decode_ws.hidden != nullptr) == (out_hidden != nullptr) &&
        (!stable_qsa || (decode_ws.kv_bucket == stable_kv_bucket &&
                        decode_ws.qsa_budget == w.indexer_top_k / 4 && cache.indexer_blocks == pos0 / 4))) {
        if (!run_stable(decode_ws)) return res;
        decode_ws.next_pos = (int) kv_len;
        cache.cur_pos = (int) kv_len;
        ++decode_ws.replays;
        if (qsa != QSA_DENSE) cache.indexer_blocks = (int) logical_blocks;
        cache.ple_prev = inputs->ple_prev;
        res.ok = true;
        return res;
    }
    // Dense decode may not have maintained the pooled prefix. Bootstrap only
    // already-cached complete blocks once; the stable graph handles this token.
    if (!measure && stable_qsa && cache.indexer_blocks < pos0 / 4) {
        ggml_context * bc = ggml_init({8 * 1024 * 1024, nullptr, true});
        if (!bc) return res;
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
        if (!ok) { clear_qwen4exp_decode_workspace(decode_ws); return res; }
        cache.indexer_blocks = pos0 / 4;
    }
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 200000 +
                  ggml_graph_overhead_custom(200000, false) + (1u << 20);
    ip.no_alloc = true;
    ggml_context * ctx = nullptr;
    if (pool) {
        if (pool->ctx == nullptr) {
            pool->ctx = ggml_init(ip);
        } else {
            // Retire captures before recycling metadata/allocator addresses.
            ggml_backend_cuda_graph_invalidate_range(backend,
                ggml_get_mem_buffer(pool->ctx), ggml_get_mem_size(pool->ctx));
            ggml_reset(pool->ctx);
        }
        ctx = pool->ctx;
        pool->gf = nullptr;
        pool->backend = backend;
        pool->model = &w;
        pool->max_ctx = cache.max_ctx;
    } else {
        ctx = ggml_init(ip);
    }
    if (!ctx) return res;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 200000, false);

    const int64_t graph_kv_len = use_stable_graph ? stable_kv_bucket : kv_len;
    const int64_t mask_len = use_stable_graph ? stable_kv_bucket : kv_len;

    ggml_tensor * inp_emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, T);
    ggml_set_input(inp_emb);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * T);
    ggml_set_input(positions);
    ggml_tensor * kv_row = nullptr;
    if (use_stable_graph) {
        kv_row = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_input(kv_row);
    }
    if (stable_qsa) {
        decode_ws.kv_bucket = stable_kv_bucket;
        decode_ws.qsa_blocks = stable_kv_bucket / 4;
        decode_ws.qsa_visibility = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, decode_ws.qsa_blocks);
        decode_ws.qsa_params = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 10);
        ggml_set_input(decode_ws.qsa_visibility);
        ggml_set_input(decode_ws.qsa_params);
    }
    ggml_tensor * mask = nullptr;
    // Prompt QSA derives visibility itself; verify owns a separate mask per dense row.
    if ((T > 1 || use_stable_graph) && qsa == QSA_DENSE && !verify) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, mask_len, T);
        ggml_set_input(mask);
    }
    std::vector<Qwen4ExpAttnRow> rows;   // verify: each row's T=1 attention inputs
    for (int64_t t = 0; verify && t < T; ++t) {
        Qwen4ExpAttnRow row;
        row.qsa = qsa_mode(w, cache, 1, pos0 + t, profile.optimized);
        if (row.qsa == QSA_DENSE) {
            row.span = qwen4exp_stable_kv_span(cache.kv_bucket_base, cache.max_ctx, pos0 + t + 1);
            row.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, row.span, 1);
            ggml_set_input(row.mask);
        } else {
            row.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
            ggml_set_input(row.positions);
        }
        rows.push_back(row);
    }
    ggml_tensor * ple_in = nullptr;
    if (has_ple) {
        ple_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.ple_head_dim * ple_heads, T);
        ggml_set_input(ple_in);
    }

    ggml_tensor * res_hc = repeat_dim1(ctx,
        ggml_reshape_3d(ctx, inp_emb, w.n_embd, 1, T), w.n_hc);

    ggml_tensor * xn_next = nullptr;

    // FFN half of layer il: HC combine into the FFN norm, MoE, and the HC combine fused with the next layer's norm
    // (left in xn_next). Returns the new residual.
    auto ffn_half = [&](int il, ggml_tensor * cur, ggml_tensor * inject, ggml_tensor * res_hc, int64_t layer_T,
                        ggml_tensor *& xn_next) -> ggml_tensor * {
        const Qwen4ExpLayer & L = w.layers[il];
        ggml_tensor * ffn_fused = hc_combine_norm(ctx, inject, res_hc, cur,
            L.hc_ffn_norm, w.n_embd, w.n_hc, layer_T, w.rms_eps);
        res_hc = hc_norm_res(ctx, ffn_fused, w.n_embd, w.n_hc, layer_T);
        cur = hc_mix_from_xn(ctx, hc_norm_xn(ctx, ffn_fused, w.n_embd, w.n_hc, layer_T),
                             L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, &inject,
                             w.n_embd, w.n_hc);
        const bool next_ple = (il + 1 < w.n_layer) && w.layers[il + 1].is_ple && has_ple;
        // Prefill: the MoE combine runs inside the next HC_COMBINE_NORM (one kernel fewer; last-bit numerics change
        // from FMA contraction in the new kernel, covered by the long-prompt quality gate). Not at T=1: it cost ~1.7% decode.
        Qwen4ExpMoeParts moe_parts;
        const bool fold = !next_ple && ggml_backend_cuda_mmb_prefill(layer_T);
        cur = build_moe(ctx, cur, L, w, fold ? &moe_parts : nullptr);

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
        } else {
            ggml_tensor * gamma = (il + 1 < w.n_layer)
                ? w.layers[il + 1].hc_attn_norm : w.output_hc_norm;
            ggml_tensor * f = hc_combine_norm(ctx, inject, res_hc, cur,
                gamma, w.n_embd, w.n_hc, layer_T, w.rms_eps);
            res_hc = hc_norm_res(ctx, f, w.n_embd, w.n_hc, layer_T);
            xn_next = hc_norm_xn(ctx, f, w.n_embd, w.n_hc, layer_T);
        }
        return res_hc;
    };
    struct { ggml_tensor * cur = nullptr, * inject = nullptr, * res = nullptr; } branch;

    for (int il = 0; il < w.n_layer; ++il) {
        const Qwen4ExpLayer & L = w.layers[il];

        if (L.is_ple && has_ple) {
            res_hc = build_ple(ctx, gf, res_hc, ple_in, L, w, cache.ple_conv_state[ple_idx[il]],
                               verify ? cache.spec_ple : nullptr);
            xn_next = nullptr;   // PLE changed the residual; the norm must rerun
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur;
        if (xn_next != nullptr) {
            cur = hc_mix_from_xn(ctx, xn_next, L.hc_attn_down, L.hc_attn_up,
                                 L.hc_attn_inject, &inject, w.n_embd, w.n_hc);
        } else {
            cur = hc_mix(ctx, res_hc, L.hc_attn_norm, L.hc_attn_down,
                         L.hc_attn_up, L.hc_attn_inject, &inject,
                         w.n_embd, w.n_hc, w.rms_eps);
        }
        xn_next = nullptr;
        if (L.is_full_attention) {
            const int fi = full_idx[il];
            cur = build_full_attn(ctx, gf, cur, L, w,
                                  cache.attn_k[fi], cache.attn_v[fi], cache.indexer_k[fi],
                                  profile.optimized ? cache.indexer_raw[fi] : nullptr,
                                  positions, mask, kv_row, graph_kv_len, pos0,
                                  il < (int) w.compress_ratios.size() ? w.compress_ratios[il] : 0,
                                  cache.indexer_blocks, qsa, verify ? &rows : nullptr, stable_qsa ? &decode_ws : nullptr);
        } else {
            const int li = lin_idx[il];
            cur = build_linear_attn(ctx, gf, cur, L, w, cache.ssm_state[li], cache.conv_state[li],
                                    verify ? cache.spec_ssm[li] : nullptr, verify ? cache.spec_conv[li] : nullptr);
        }
        int64_t layer_T = T;
        if (!verify && il == w.n_layer - 1 && T > 1) {
            if (out_hidden) branch = { cur, inject, res_hc };   // every row's FFN half, for the MTP hidden only
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
        res_hc = ffn_half(il, cur, inject, res_hc, layer_T, xn_next);
    }

    ggml_tensor * final = (xn_next != nullptr)
        ? hc_mix_from_xn(ctx, xn_next, w.output_hc_down, w.output_hc_up,
                         nullptr, nullptr, w.n_embd, w.n_hc)
        : hc_mix(ctx, res_hc, w.output_hc_norm, w.output_hc_down,
                 w.output_hc_up, nullptr, nullptr,
                 w.n_embd, w.n_hc, w.rms_eps);
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
        hidden = branch.cur ? ffn_half(w.n_layer - 1, branch.cur, branch.inject, branch.res, T, unused) : res_hc;
        for (ggml_tensor * t = hidden; t; t = t->view_src) ggml_set_output(t);
        ggml_build_forward_expand(gf, hidden);
    }

    ggml_tensor * mtp_positions = nullptr;
    if (mtp_prefill) {
        const int64_t H = w.n_embd, hc = w.n_hc, hd = H * hc;
        const int64_t first = pos0 == 0 ? 1 : 0, pairs = T - first, start = pos0 - (1 - first);
        ggml_tensor * h = ggml_reshape_2d(ctx, hidden, hd, T);
        if (pos0 > 0) {
            h = ggml_concat(ctx, cache.mtp_prev_hidden,
                ggml_view_2d(ctx, h, hd, T - 1, h->nb[1], 0), 1);
        }
        mtp_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * pairs);
        ggml_set_input(mtp_positions);
        // One submission, but exactly the old 512-row GEMMs, including the first
        // chunk's short final slice. A single wider GEMM changes the K/V bytes.
        for (int64_t i = 0; i < pairs; i += 512) {
            const int64_t n = std::min<int64_t>(512, pairs - i);
            ggml_tensor * e = ggml_view_2d(ctx, inp_emb, H, n, inp_emb->nb[1], (i + first) * inp_emb->nb[1]);
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
    }

    if (measure) {
        graph_memory(backend, ctx, gf, w.gfx1151, *measure);
        // Graph inputs are included above. Reserve BOTH grow-only UMA ring slots
        // as well (conservative: the normal allocator excludes their tensors).
        const size_t input_bytes = ring_align_up(ggml_nbytes(inp_emb)) +
            ring_align_up(ggml_nbytes(positions)) +
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

    // Point input tensors at this call's pinned ring slot before allocation so the gallocr leaves them alone.
    char * ring_embd = nullptr;
    char * ring_pos  = nullptr;
    char * ring_mask = nullptr;
    char * ring_ple  = nullptr;
    if (!use_stable_graph && !verify && cache.input_ring.enabled) {
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
        if (!pool) ggml_free(ctx);
        return res;
    }
    // T=1 graphs keep the same broad shape but advancing KV views can change
    // lifetimes. Recompute assignments while retaining the allocator buffers;
    // reusing the old index-wise plan produced incorrect tokens.
    const bool reserve_ok = !pool || !pool->planned || ggml_gallocr_reserve(galloc, gf);
    if (!reserve_ok || !ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "[qwen4exp] graph alloc failed (T=%lld kv_len=%lld)\n",
                     (long long) T, (long long) kv_len);
        if (pool) {
            pool->planned = false;
        } else {
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
        }
        return res;
    }
    if (pool) pool->planned = true;

    if (use_stable_graph) {
        decode_ws.gf = gf;
        decode_ws.inp_emb = inp_emb;
        decode_ws.positions = positions;
        decode_ws.mask = mask;
        decode_ws.ple_in = ple_in;
        decode_ws.kv_row = kv_row;
        decode_ws.logits = logits;
        decode_ws.hidden = hidden;
        decode_ws.kv_bucket = stable_kv_bucket;
        decode_ws.qsa_blocks = stable_qsa ? stable_kv_bucket / 4 : -1;
        ++decode_ws.builds;
        decode_ws.qsa_budget = stable_qsa ? w.indexer_top_k / 4 : 0;
        decode_ws.next_pos = (int) kv_len;
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
    if (mtp_positions) {
        const int64_t pairs = T - (pos0 == 0 ? 1 : 0), start = pos0 == 0 ? 0 : pos0 - 1;
        std::vector<int32_t> p((size_t) 4 * pairs, 0);
        for (int64_t i = 0; i < pairs; ++i) p[i] = p[pairs + i] = p[2 * pairs + i] = (int32_t) (start + i);
        ggml_backend_tensor_set(mtp_positions, p.data(), 0, ggml_nbytes(mtp_positions));
    }
    const int32_t kv_row_value = pos0;
    upload_qsa(decode_ws);
    if (use_stable_graph) {
        ggml_backend_tensor_set_async(backend, inp_emb, emb.data(), 0, sizeof(float) * emb.size());
        ggml_backend_tensor_set_async(backend, positions, pos.data(), 0, sizeof(int32_t) * pos.size());
        ggml_backend_tensor_set_async(backend, kv_row, &kv_row_value, 0, sizeof(kv_row_value));
        if (mask) ggml_backend_tensor_set_async(backend, mask, m.data(), 0, sizeof(ggml_fp16_t) * m.size());
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

    ggml_status status;
    {   // verify: every matmul column equals its single-token product (see ggml_backend_cuda_set_mmvq_batch_invariant)
        ScopedCudaGraphOverrides invariant(false, 0, false, 0, /*mmvq_batch_invariant=*/verify);
        status = ggml_backend_graph_compute(backend, gf);
    }
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[qwen4exp] graph compute failed\n");
        if (pool) {
            clear_qwen4exp_decode_workspace(*pool);
        } else {
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
        }
        return res;
    }
    cache.ple_prev = inputs->ple_prev;
    // Commit only after the graph computed: a failed compute must not mark blocks the kernel never pooled.
    cache.cur_pos = (int) kv_len;
    if (mtp_prefill) cache.mtp_prev_pos = pos0 + n_tokens - 1;
    if (verify) { cache.spec_pos = pos0; cache.spec_tokens = n_tokens; }
    // The all-keys branch did not pool anything. Leaving this prefix at zero
    // makes the first selected-attention call pool the earlier raw keys too.
    if ((qsa != QSA_DENSE || (verify && rows.back().qsa != QSA_DENSE)) &&
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
        ggml_gallocr_free(galloc);
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

// One MTP batch: the graph rebuilds per call (its K/V span grows) on a reused context and allocator.
bool mtp_forward_batch(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache,
                       const int32_t * tokens, const float * hidden, int n, int pos0,
                       std::vector<float> & out_logits, std::vector<float> * out_hidden, bool kv_only,
                       bool last_only = false, int draft_rank = -1, Qwen4ExpGraphMemory * measure = nullptr) {
    const Qwen4ExpCudaScope profile(w.gfx1151);
    const int64_t H = w.n_embd, hc = w.n_hc, T = n, kv_len = pos0 + n;
    const bool device_draft = draft_rank >= 0, chain = draft_rank > 0;
    std::vector<float> emb(measure || chain ? 0 : (size_t) H * T);
    if (!measure && !chain && !w.embedder.embed(tokens, n, emb.data())) return false;

    Qwen4ExpDecodeWorkspace local_ws;
    Qwen4ExpDecodeWorkspace & ws = measure ? local_ws : cache.mtp_workspace;
    if (ws.ctx == nullptr) {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false) + (1u << 16);
        ip.no_alloc = true;
        ws.ctx = ggml_init(ip);
        if (!ws.ctx) return false;
    } else {
        ggml_backend_cuda_graph_invalidate_range(backend, ggml_get_mem_buffer(ws.ctx), ggml_get_mem_size(ws.ctx));
        ggml_reset(ws.ctx);
    }
    ws.backend = backend;
    ggml_context * ctx = ws.ctx;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    ggml_tensor * inp_emb = chain ? ggml_get_rows(ctx, w.mtp_embd,
        ggml_view_1d(ctx, cache.mtp_chain_ids, 1, (draft_rank - 1) * sizeof(int32_t)))
        : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_tensor * inp_h = chain ? cache.mtp_chain_hidden : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H * hc, T);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * T);
    ggml_tensor * mask = !kv_only && !last_only && T > 1 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len, T) : nullptr;
    if (!chain) { ggml_set_input(inp_emb); ggml_set_input(inp_h); }
    for (ggml_tensor * t : {positions, mask}) if (t) ggml_set_input(t);

    ggml_tensor * res = mtp_input(ctx, w, inp_emb, inp_h);

    // One trunk-style layer (upstream-order HC helpers; dense attention on the draft layer's own K/V).
    const Qwen4ExpLayer & L = w.mtp;
    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = hc_mix(ctx, res, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject, &inject,
                               H, hc, w.rms_eps);
    // Decode still attends densely over the entire prompt; prefill only fills its K/V.
    cur = build_full_attn(ctx, gf, cur, L, w, cache.mtp_k, cache.mtp_v, /*indexer_k=*/nullptr, /*indexer_raw=*/nullptr,
                          positions, mask, nullptr, kv_len, pos0, /*ratio=*/4, /*n_pooled=*/0, QSA_DENSE,
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
    // Re-plan every call (the K/V views move), keeping the allocator's buffer.
    bool ok = ws.alloc && (!ws.planned || ggml_gallocr_reserve(ws.alloc, gf)) && ggml_gallocr_alloc_graph(ws.alloc, gf);
    ws.planned = ok;
    if (ok) {
        std::vector<int32_t> pos((size_t) 4 * T, 0);   // M-RoPE, section-major; section 3 stays zero
        for (int64_t i = 0; i < T; ++i) {
            pos[(size_t) i] = pos[(size_t) (T + i)] = pos[(size_t) (2 * T + i)] = (int32_t) (pos0 + i);
        }
        if (!chain) {
            ggml_backend_tensor_set(inp_emb, emb.data(), 0, ggml_nbytes(inp_emb));
            ggml_backend_tensor_set(inp_h, hidden, 0, ggml_nbytes(inp_h));
        }
        ggml_backend_tensor_set(positions, pos.data(), 0, ggml_nbytes(positions));
        if (mask) {
            std::vector<ggml_fp16_t> m((size_t) kv_len * T);
            for (int64_t row = 0; row < T; ++row) {
                for (int64_t col = 0; col < kv_len; ++col) {
                    m[(size_t) (row * kv_len + col)] = ggml_fp32_to_fp16(col <= pos0 + row ? 0.0f : -INFINITY);
                }
            }
            ggml_backend_tensor_set(mask, m.data(), 0, ggml_nbytes(mask));
        }
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
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
    ggml_backend_tensor_get(cache.mtp_chain_ids, drafts.data(), 0, k * sizeof(int32_t));
    for (int32_t & id : drafts) {
        if (id < 0 || (size_t) id >= w.mtp_vocab_ids.size()) return false;
        id = w.mtp_vocab_ids[id];
    }
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
        const Qwen4ExpForwardSegment * segments, int n_segments, bool use_qsa) {
    // RDNA3 MMID supports at most four batch-invariant decode rows.
    if (n_segments < 1 || n_segments > 4) return false;
    for (int s = 0; s < n_segments; ++s) {
        const auto & segment = segments[s];
        if (segment.n_tokens != 1 ||
            qsa_mode(w, *segment.cache, 1, segment.pos0, use_qsa) != QSA_DENSE)
            return false;
    }
    return true;
}

static Qwen4ExpForwardResult forward_sequential(ggml_backend_t backend,
        const Qwen4ExpWeights & w, const Qwen4ExpForwardSegment * segments, int n_segments,
        std::vector<std::vector<float>> & out_logits) {
    Qwen4ExpForwardResult result;
    out_logits.resize((size_t) n_segments);
    for (int s = 0; s < n_segments; ++s) {
        const auto & segment = segments[s];
        if (!qwen4exp_forward(backend, w, *segment.cache, segment.tokens,
                segment.n_tokens, segment.pos0, out_logits[s]).ok) {
            out_logits.clear();
            return result;
        }
    }
    result.ok = true;
    return result;
}

Qwen4ExpForwardResult qwen4exp_forward_batched(
        ggml_backend_t backend, const Qwen4ExpWeights & w,
        Qwen4ExpCache * const * caches, const int32_t * tokens,
        const int32_t * positions, int n_slots,
        Qwen4ExpBatchedDecodeWorkspace & workspace,
        std::vector<std::vector<float>> & out_logits) {
    Qwen4ExpForwardResult result;
    if (!backend || !caches || !tokens || !positions || n_slots <= 0) return result;
    const Qwen4ExpCudaScope profile(w.gfx1151);
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
    Qwen4ExpForwardSegment segments[8];
    for (int s = 0; s < n_slots; ++s) {
        if (!caches[s] || positions[s] < 0 || positions[s] >= caches[s]->max_ctx ||
            caches[s]->linear_layer_ids != caches[0]->linear_layer_ids ||
            caches[s]->full_layer_ids != caches[0]->full_layer_ids ||
            caches[s]->ple_layer_ids != caches[0]->ple_layer_ids) return result;
        for (int prev = 0; prev < s; ++prev) if (caches[prev] == caches[s]) return result;
        segments[s] = {caches[s], tokens + s, 1, positions[s]};
    }
    // ponytail: dense batching only; per-slot QSA graphs can replace this if throughput warrants it.
    if (!qwen4exp_can_batch(w, segments, n_slots, profile.optimized))
        return forward_sequential(backend, w, segments, n_slots, out_logits);

    const bool hc_fused = true;
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
                    caches[s]->attn_v[(size_t) fi], positions[s], kv_view_len,
                    column(ctx, cur, s), profile.optimized ? caches[s]->indexer_raw[(size_t) fi] : nullptr);
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
        cur = build_moe(ctx, cur, L, w);
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
    // Match solo MMVQ arithmetic regardless of the number of active slots.
    const ScopedCudaGraphOverrides overrides(false, 0, false, 0, /*mmvq_batch_invariant=*/true);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        workspace.planned = false;
        std::fprintf(stderr, "[qwen4exp] batched graph compute failed\n");
        return result;
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
    return result;
}

}  // namespace luce::common
