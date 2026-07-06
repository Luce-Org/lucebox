// DeepSeek V4 Flash ggml compute graph builder.
//
// Implements the full forward pass using ggml ops:
//   1. HC pre (Sinkhorn-normalized residual stream mixing)
//   2. MLA attention (low-rank Q, single KV head, grouped output)
//   3. KV compression (learned gate+kv pooling, RoPE on compressed rows)
//   4. Indexer (top-k selective attention over compressed KV)
//   5. HC post (update residual streams)
//   6. MoE FFN (hash routing + top-k + shared expert + clamped SwiGLU)

#include "deepseek4_internal.h"
#include "deepseek4_hc_cuda.h"
#include "internal.h"
#include "../common/step_graph.h"
#include "../common/moe_hybrid_ffn_eval.h"
#include "../common/moe_hybrid_routing_stats.h"
#include "../common/moe_hybrid_stream.h"
#include "../common/moe_hybrid_types.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace dflash::common {

namespace {
using Ds4TimingClock = std::chrono::steady_clock;

static uint64_t ds4_elapsed_us(Ds4TimingClock::time_point start,
                               Ds4TimingClock::time_point end) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

static bool ds4_env_flag(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] && std::strcmp(value, "0") != 0;
}

template <typename Fn>
static void ds4_parallel_for_tokens(int n_tokens, int min_parallel_tokens, Fn && fn) {
    if (n_tokens <= min_parallel_tokens) {
        fn(0, n_tokens);
        return;
    }

    unsigned nth = std::thread::hardware_concurrency();
    if (nth == 0) nth = 4;
    if (nth > 8) nth = 8;
    if ((int)nth <= 1) {
        fn(0, n_tokens);
        return;
    }

    const int chunk = std::max(1, (n_tokens + (int)nth - 1) / (int)nth);
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    pool.reserve(nth);
    for (unsigned i = 0; i < nth; ++i) {
        pool.emplace_back([&]() {
            for (;;) {
                const int begin = next.fetch_add(chunk);
                if (begin >= n_tokens) {
                    break;
                }
                const int end = std::min(begin + chunk, n_tokens);
                fn(begin, end);
            }
        });
    }
    for (auto & th : pool) {
        th.join();
    }
}

static void add_ffn_telemetry(DeepSeek4StepTelemetry * dst,
                              const MoeHybridFfnTelemetry & src) {
    if (!dst) return;
    dst->ffn_hot_us += src.hot_us;
    dst->ffn_cold_us += src.cold_us;
    dst->ffn_combine_us += src.combine_us;
    dst->ffn_partition_us += src.partition_us;
    dst->ffn_hot_graph_builds += src.hot_graph_builds;
    dst->ffn_hot_graph_hits += src.hot_graph_hits;
    dst->ffn_cold_graph_builds += src.cold_graph_builds;
    dst->ffn_cold_graph_hits += src.cold_graph_hits;
    dst->hot_selected += src.hot_selected;
    dst->cold_selected += src.cold_selected;
}

} // namespace

struct DeepSeek4I32InputBinding {
    ggml_tensor * tensor = nullptr;
    int32_t       value  = 0;
};

struct DeepSeek4I32ArrayBinding {
    ggml_tensor *          tensor = nullptr;
    std::vector<int32_t>   values;
};

struct DeepSeek4I64ArrayBinding {
    ggml_tensor *          tensor = nullptr;
    std::vector<int64_t>   values;
};

static ggml_tensor * build_rms_norm(ggml_context * ctx, ggml_tensor * x,
                                     ggml_tensor * weight, float eps);
static ggml_tensor * build_clamped_swiglu(ggml_context * ctx,
                                           ggml_tensor * gate,
                                           ggml_tensor * up,
                                           float clamp);
static ggml_tensor * build_shared_ffn(ggml_context * ctx,
                                       ggml_tensor * cur,
                                       const DeepSeek4Weights & w,
                                       const DeepSeek4Layer & L);
static ggml_tensor * build_moe_ffn(ggml_context * ctx,
                                    ggml_tensor * cur,
                                    const DeepSeek4Weights & w,
                                    const DeepSeek4Layer & L,
                                    int layer_idx,
                                    int n_tokens);

struct DeepSeek4CachedDecodeFfnGraph {
    const ggml_context * owner_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    int layer_idx = -1;
    int n_tokens = 0;
    bool hash_routed = false;
    StepGraph sg;
    ggml_tensor * hash_ids = nullptr;

    bool valid() const {
        return owner_ctx && backend && layer_idx >= 0 && n_tokens > 0 &&
               sg.ctx && sg.gf && sg.alloc && sg.inp_embed && sg.hidden_states &&
               (!hash_routed || hash_ids);
    }

    void free() {
        hash_ids = nullptr;
        step_graph_destroy(sg);
        owner_ctx = nullptr;
        backend = nullptr;
        layer_idx = -1;
        n_tokens = 0;
        hash_routed = false;
    }
};

struct DeepSeek4CachedDecodeOutputGraph {
    const ggml_context * owner_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    int n_tokens = 0;
    StepGraph sg;

    bool valid() const {
        return owner_ctx && backend && n_tokens > 0 &&
               sg.ctx && sg.gf && sg.alloc && sg.hidden_input && sg.logits;
    }

    void free() {
        step_graph_destroy(sg);
        owner_ctx = nullptr;
        backend = nullptr;
        n_tokens = 0;
    }
};

struct DeepSeek4AttentionGraphInputs {
    ggml_tensor * rope_pos = nullptr;
    ggml_tensor * neg_pos = nullptr;
    ggml_tensor * raw_kv_rows = nullptr;
    ggml_tensor * attn_ape_row = nullptr;
    ggml_tensor * attn_state_rows = nullptr;
    ggml_tensor * attn_comp_rows = nullptr;
    ggml_tensor * attn_comp_pos = nullptr;
    ggml_tensor * index_ape_row = nullptr;
    ggml_tensor * index_state_rows = nullptr;
    ggml_tensor * index_comp_rows = nullptr;
    ggml_tensor * index_comp_pos = nullptr;
};

struct DeepSeek4CachedDecodeAttnGraph {
    const ggml_context * owner_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    int layer_idx = -1;
    int n_tokens = 0;
    int n_raw = 0;
    int n_comp_attn = 0;
    int n_index_comp = 0;
    bool compressed = false;
    bool indexed = false;
    StepGraph sg;
    DeepSeek4AttentionGraphInputs inputs;

    bool valid() const {
        return owner_ctx && backend && layer_idx >= 0 && n_tokens == 1 &&
               n_raw > 0 && n_comp_attn >= 0 && n_index_comp >= 0 &&
               sg.ctx && sg.gf && sg.alloc && sg.inp_embed && sg.hidden_states &&
               inputs.rope_pos && inputs.neg_pos && inputs.raw_kv_rows &&
               (!compressed || (inputs.attn_ape_row &&
                                inputs.attn_state_rows && inputs.attn_comp_rows && inputs.attn_comp_pos)) &&
               (!indexed || (inputs.index_ape_row &&
                             inputs.index_state_rows && inputs.index_comp_rows && inputs.index_comp_pos));
    }

    void free() {
        step_graph_destroy(sg);
        inputs = {};
        owner_ctx = nullptr;
        backend = nullptr;
        layer_idx = -1;
        n_tokens = 0;
        n_raw = 0;
        n_comp_attn = 0;
        n_index_comp = 0;
        compressed = false;
        indexed = false;
    }
};

struct DeepSeek4CachedLayerAlloc {
    const ggml_context * owner_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t alloc = nullptr;

    bool valid() const {
        return owner_ctx && backend && alloc;
    }

    void free() {
        if (alloc) {
            ggml_gallocr_free(alloc);
            alloc = nullptr;
        }
        owner_ctx = nullptr;
        backend = nullptr;
    }
};

struct DeepSeek4LayerRangeScratch {
    const ggml_context * owner_ctx = nullptr;
    int n_tokens = 0;
    int n_embd = 0;
    int n_hc = 0;
    int n_expert_used = 0;
    std::vector<float> cur;
    std::vector<float> ffn_working;
    std::vector<float> hc_post;
    std::vector<float> hc_comb;
    std::vector<float> next_hc;
    std::vector<float> attn_out_host;
    std::vector<float> ffn_out_host;
    std::vector<float> final_embd;
    std::vector<int32_t> hash_expert_ids;

    void ensure(const ggml_context * ctx,
                int tokens,
                int embd,
                int hc,
                int expert_used) {
        owner_ctx = ctx;
        n_tokens = tokens;
        n_embd = embd;
        n_hc = hc;
        n_expert_used = expert_used;
        const size_t embd_count = (size_t) tokens * (size_t) embd;
        const size_t hc_count = embd_count * (size_t) hc;
        cur.resize(embd_count);
        ffn_working.resize(embd_count);
        hc_post.resize((size_t) tokens * (size_t) hc);
        hc_comb.resize((size_t) tokens * (size_t) hc * (size_t) hc);
        next_hc.resize(hc_count);
        attn_out_host.resize(embd_count);
        ffn_out_host.resize(embd_count);
        final_embd.resize(embd_count);
        hash_expert_ids.resize((size_t) tokens * (size_t) expert_used);
    }
};

static bool build_cached_decode_ffn_graph(
        DeepSeek4CachedDecodeFfnGraph & out,
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L,
        int layer_idx,
        int n_tokens,
        bool hash_routed) {
    out.free();

    const size_t ctx_size = 16 * 1024 * 1024;
    ggml_init_params params{};
    params.mem_size = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    out.sg.ctx = ggml_init(params);
    if (!out.sg.ctx) {
        return false;
    }

    out.sg.inp_embed = ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_set_input(out.sg.inp_embed);
    out.sg.gf = ggml_new_graph_custom(out.sg.ctx, 2048, false);

    ggml_tensor * ffn_normed = build_rms_norm(out.sg.ctx, out.sg.inp_embed, L.ffn_norm, w.rms_eps);
    ggml_tensor * ffn_out = nullptr;
    if (hash_routed) {
        out.hash_ids = ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_I32, w.n_expert_used, n_tokens);
        ggml_set_input(out.hash_ids);

        ggml_tensor * shared_out = build_shared_ffn(out.sg.ctx, ffn_normed, w, L);
        ggml_tensor * logits = ggml_mul_mat(out.sg.ctx, L.ffn_gate_inp, ffn_normed);
        ggml_tensor * probs = ggml_sqrt(out.sg.ctx, ggml_softplus(out.sg.ctx, logits));

        const int n_used = w.n_expert_used;
        const int n_ff_exp = w.n_ff_exp;
        ggml_tensor * cur_3d = ggml_reshape_3d(out.sg.ctx, ffn_normed, w.n_embd, 1, n_tokens);
        ggml_tensor * gate_e = ggml_mul_mat_id(out.sg.ctx, L.ffn_gate_exps, cur_3d, out.hash_ids);
        ggml_tensor * up_e = ggml_mul_mat_id(out.sg.ctx, L.ffn_up_exps, cur_3d, out.hash_ids);
        gate_e = ggml_reshape_3d(out.sg.ctx, gate_e, n_ff_exp, n_used, n_tokens);
        up_e = ggml_reshape_3d(out.sg.ctx, up_e, n_ff_exp, n_used, n_tokens);
        ggml_tensor * mid_e = build_clamped_swiglu(out.sg.ctx, gate_e, up_e, w.swiglu_clamp_exp);
        ggml_tensor * down_e = ggml_mul_mat_id(out.sg.ctx, L.ffn_down_exps, mid_e, out.hash_ids);
        down_e = ggml_reshape_3d(out.sg.ctx, down_e, w.n_embd, n_used, n_tokens);

        ggml_tensor * probs_3d = ggml_reshape_3d(out.sg.ctx, probs, 1, w.n_expert, n_tokens);
        ggml_tensor * weights = ggml_get_rows(out.sg.ctx, probs_3d, out.hash_ids);
        weights = ggml_reshape_2d(out.sg.ctx, weights, n_used, n_tokens);
        ggml_tensor * w_sum = ggml_sum_rows(out.sg.ctx, weights);
        w_sum = ggml_clamp(out.sg.ctx, w_sum, 6.103515625e-5f, INFINITY);
        weights = ggml_div(out.sg.ctx, weights, w_sum);
        if (w.expert_weight_scale != 1.0f) {
            weights = ggml_scale(out.sg.ctx, weights, w.expert_weight_scale);
        }

        ggml_tensor * weights_3d = ggml_reshape_3d(out.sg.ctx, weights, 1, n_used, n_tokens);
        ggml_tensor * routed_out = ggml_mul(out.sg.ctx, down_e, weights_3d);
        routed_out = ggml_cont(out.sg.ctx, ggml_permute(out.sg.ctx, routed_out, 1, 0, 2, 3));
        routed_out = ggml_sum_rows(out.sg.ctx, routed_out);
        routed_out = ggml_reshape_2d(out.sg.ctx, routed_out, w.n_embd, n_tokens);

        ffn_out = ggml_add(out.sg.ctx, shared_out, routed_out);
    } else {
        ffn_out = build_moe_ffn(out.sg.ctx, ffn_normed, w, L, layer_idx, n_tokens);
    }

    if (!ffn_out) {
        out.free();
        return false;
    }

    out.sg.hidden_states = ffn_out;
    ggml_set_output(out.sg.hidden_states);
    ggml_build_forward_expand(out.sg.gf, out.sg.hidden_states);

    out.sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(out.sg.alloc, out.sg.gf)) {
        out.free();
        return false;
    }

    out.owner_ctx = w.ctx;
    out.backend = backend;
    out.layer_idx = layer_idx;
    out.n_tokens = n_tokens;
    out.hash_routed = hash_routed;
    return true;
}

static bool build_cached_decode_output_graph(
        DeepSeek4CachedDecodeOutputGraph & out,
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        int n_tokens) {
    out.free();

    const size_t ctx_size = 16 * 1024 * 1024;
    ggml_init_params params{};
    params.mem_size = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    out.sg.ctx = ggml_init(params);
    if (!out.sg.ctx) {
        return false;
    }

    out.sg.hidden_input = ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_set_input(out.sg.hidden_input);
    ggml_tensor * normed = build_rms_norm(out.sg.ctx, out.sg.hidden_input, w.out_norm, w.rms_eps);
    out.sg.logits = ggml_mul_mat(out.sg.ctx, w.output, normed);
    ggml_set_output(out.sg.logits);
    out.sg.gf = ggml_new_graph_custom(out.sg.ctx, 1024, false);
    ggml_build_forward_expand(out.sg.gf, out.sg.logits);

    out.sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(out.sg.alloc, out.sg.gf)) {
        out.free();
        return false;
    }

    out.owner_ctx = w.ctx;
    out.backend = backend;
    out.n_tokens = n_tokens;
    return true;
}

// ─── Helper: RMSNorm ────────────────────────────────────────────────────

static ggml_tensor * build_rms_norm(ggml_context * ctx, ggml_tensor * x,
                                     ggml_tensor * weight, float eps) {
    ggml_tensor * normed = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, normed, weight);
}

// ─── Helper: Clamped SwiGLU ─────────────────────────────────────────────

static ggml_tensor * build_clamped_swiglu(ggml_context * ctx,
                                           ggml_tensor * gate,
                                           ggml_tensor * up,
                                           float clamp) {
    // DS4 clamps only the upper side of gate, but both sides of up.
    gate = ggml_clamp(ctx, gate, -INFINITY, clamp);
    up   = ggml_clamp(ctx, up,   -clamp, clamp);
    // silu(gate) * up
    gate = ggml_silu(ctx, gate);
    return ggml_mul(ctx, gate, up);
}

// ─── Helper: Partial RoPE (tail rotation) ───────────────────────────────
// DS4 applies RoPE only to the last n_rot dimensions of each head.
// ggml_rope_ext applies to the first n_dims, so we split, rope the tail, concat.
//
// x: [head_dim, n_heads, n_tokens] (3D) — applies tail RoPE to each head.
// pos: [n_tokens] I32 — position for each token.
// Returns: [head_dim, n_heads, n_tokens] with last n_rot dims rotated.

static ggml_tensor * build_tail_rope_3d(ggml_context * ctx,
                                         ggml_tensor * x,
                                         ggml_tensor * pos,
                                         int n_rot,
                                         int head_dim,
                                         int n_heads,
                                         int n_tokens,
                                         float freq_base,
                                         float freq_scale,
                                         float ext_factor,
                                         float attn_factor,
                                         float beta_fast,
                                         float beta_slow,
                                         int n_ctx_orig) {
    const int n_nope = head_dim - n_rot;
    // Split: nope [n_nope, n_heads, n_tokens], tail [n_rot, n_heads, n_tokens]
    ggml_tensor * nope = ggml_view_3d(ctx, x, n_nope, n_heads, n_tokens,
                                       x->nb[1], x->nb[2], 0);
    ggml_tensor * tail = ggml_view_3d(ctx, x, n_rot, n_heads, n_tokens,
                                       x->nb[1], x->nb[2],
                                       (size_t)n_nope * ggml_type_size(x->type));
    // tail is non-contiguous (stride between heads = head_dim, not n_rot)
    tail = ggml_cont(ctx, tail);
    // Apply rope to the contiguous tail: [n_rot, n_heads, n_tokens]
    // DS4 uses standard sequential pairs (i, i+1), which is GGML_ROPE_TYPE_NORMAL
    tail = ggml_rope_ext(ctx, tail, pos, nullptr,
                         n_rot, GGML_ROPE_TYPE_NORMAL, n_ctx_orig,
                         freq_base, freq_scale,
                         ext_factor, attn_factor, beta_fast, beta_slow);
    // Concat nope + tail along dim 0 → [head_dim, n_heads, n_tokens]
    return ggml_concat(ctx, ggml_cont(ctx, nope), tail, 0);
}

// For KV (single head): x is [head_dim, n_tokens]
static ggml_tensor * build_tail_rope_2d(ggml_context * ctx,
                                         ggml_tensor * x,
                                         ggml_tensor * pos,
                                         int n_rot,
                                         int head_dim,
                                         int n_tokens,
                                         float freq_base,
                                         float freq_scale,
                                         float ext_factor,
                                         float attn_factor,
                                         float beta_fast,
                                         float beta_slow,
                                         int n_ctx_orig) {
    // Reshape to 3D with n_heads=1 for the shared rope function
    ggml_tensor * x3d = ggml_reshape_3d(ctx, x, head_dim, 1, n_tokens);
    ggml_tensor * result = build_tail_rope_3d(ctx, x3d, pos, n_rot, head_dim, 1, n_tokens,
                                              freq_base, freq_scale, ext_factor, attn_factor,
                                              beta_fast, beta_slow, n_ctx_orig);
    return ggml_reshape_2d(ctx, result, head_dim, n_tokens);
}

// ─── KV Compressor Step ────────────────────────────────────────────────

static void build_compressor_step(
        ggml_context * ctx,
        ggml_cgraph * gf,
        ggml_tensor * cur_last,      // [n_embd, 1]
        ggml_tensor * ape,
        ggml_tensor * kv_proj,
        ggml_tensor * gate_proj,
        ggml_tensor * norm_weight,
        DeepSeek4CompressorState & state,
        ggml_tensor * comp_cache,
        int ratio,
        int head_dim,
        int token_pos,
        int n_rot,
        float rms_eps,
        float compress_rope_freq_base,
        float rope_scale_factor,
        float rope_yarn_beta_fast,
        float rope_yarn_beta_slow,
        int rope_orig_ctx,
        ggml_tensor * ape_row_inp,
        ggml_tensor * state_rows_inp,
        ggml_tensor * comp_rows_inp,
        ggml_tensor * comp_pos_inp,
        std::vector<DeepSeek4I64ArrayBinding> & i64_array_inputs,
        std::vector<DeepSeek4I32ArrayBinding> & i32_array_inputs) {
    if (!gf || !cur_last || !ape || !kv_proj || !gate_proj || !norm_weight ||
        !state.state_kv || !state.state_score || !comp_cache || ratio <= 0) {
        return;
    }

    // DS4 compression: internal width = coff * head_dim (2x for ratio-4, 1x for ratio-128)
    const int coff = (ratio == 4) ? 2 : 1;
    const int comp_width = coff * head_dim;
    const int pos_mod = token_pos % ratio;
    // For ratio-4: write into second half of state (rows ratio..2*ratio-1)
    const int row = (ratio == 4) ? (ratio + pos_mod) : pos_mod;

    ggml_tensor * kv_cur = ggml_mul_mat(ctx, kv_proj, cur_last);
    ggml_tensor * sc_cur = ggml_mul_mat(ctx, gate_proj, cur_last);

    ggml_tensor * ape_col = nullptr;
    if (ape_row_inp) {
        ape_col = ggml_get_rows(ctx, ape, ape_row_inp);
        ape_col = ggml_reshape_2d(ctx, ape_col, comp_width, 1);
    } else {
        ape_col = ggml_view_2d(
            ctx, ape, comp_width, 1, ape->nb[1], (size_t)pos_mod * ape->nb[1]);
        ape_col = ggml_cast(ctx, ape_col, GGML_TYPE_F32);
    }
    sc_cur = ggml_add(ctx, sc_cur, ape_col);

    if (state_rows_inp) {
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, state.state_kv, kv_cur, state_rows_inp));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, state.state_score, sc_cur, state_rows_inp));
    } else {
        ggml_tensor * kv_slot = ggml_view_2d(
            ctx, state.state_kv, comp_width, 1, state.state_kv->nb[1],
            (size_t)row * state.state_kv->nb[1]);
        ggml_tensor * sc_slot = ggml_view_2d(
            ctx, state.state_score, comp_width, 1, state.state_score->nb[1],
            (size_t)row * state.state_score->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, kv_cur, kv_slot));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, sc_cur, sc_slot));
    }

    if (((token_pos + 1) % ratio) != 0) {
        return;
    }

    // ── Pooling: per-dim softmax-weighted average across state rows ──
    // For ratio-128: straight per-dim softmax over all 128 rows
    // For ratio-4: interleaved across prev/current windows (complex, simplified here)
    //
    // state_kv: [comp_width, n_state_rows]
    // state_score: [comp_width, n_state_rows]
    // For ratio-128: n_state_rows = ratio = 128, all rows used directly
    // For ratio-4: n_state_rows = 2*ratio = 8 (prev 4 + current 4)
    //   Correct interleaving would select prev[j] and current[head_dim+j] alternately.
    //   Simplified: use all rows, take first head_dim of result.

    ggml_tensor * sv_kv = nullptr;
    ggml_tensor * sv_sc = nullptr;
    int n_state_rows = ratio;
    if (ratio == 4) {
        const size_t hi_off_kv = (size_t)ratio * state.state_kv->nb[1] +
                                 (size_t)head_dim * state.state_kv->nb[0];
        const size_t hi_off_sc = (size_t)ratio * state.state_score->nb[1] +
                                 (size_t)head_dim * state.state_score->nb[0];
        ggml_tensor * prev_kv = ggml_view_2d(ctx, state.state_kv, head_dim, ratio,
                                             state.state_kv->nb[1], 0);
        ggml_tensor * cur_kv_hi = ggml_view_2d(ctx, state.state_kv, head_dim, ratio,
                                               state.state_kv->nb[1], hi_off_kv);
        ggml_tensor * prev_sc = ggml_view_2d(ctx, state.state_score, head_dim, ratio,
                                             state.state_score->nb[1], 0);
        ggml_tensor * cur_sc_hi = ggml_view_2d(ctx, state.state_score, head_dim, ratio,
                                               state.state_score->nb[1], hi_off_sc);
        sv_kv = ggml_concat(ctx, prev_kv, cur_kv_hi, 1);
        sv_sc = ggml_concat(ctx, prev_sc, cur_sc_hi, 1);
        n_state_rows = 2 * ratio;
    } else {
        sv_kv = ggml_view_2d(ctx, state.state_kv, comp_width, n_state_rows,
                             state.state_kv->nb[1], 0);
        sv_sc = ggml_view_2d(ctx, state.state_score, comp_width, n_state_rows,
                             state.state_score->nb[1], 0);
    }
    // Transpose to [n_state_rows, comp_width] so softmax operates per-dimension
    ggml_tensor * sc_T = ggml_cont(ctx, ggml_transpose(ctx, sv_sc));
    ggml_tensor * kv_T = ggml_cont(ctx, ggml_transpose(ctx, sv_kv));
    // Softmax over ne[0] = n_state_rows for each of comp_width dims
    ggml_tensor * probs_T = ggml_soft_max(ctx, sc_T);
    // Element-wise: probs * kv
    ggml_tensor * weighted_T = ggml_mul(ctx, probs_T, kv_T);
    // Sum over ne[0] = n_state_rows → [1, comp_width]
    ggml_tensor * pooled_sum = ggml_sum_rows(ctx, weighted_T);
    ggml_tensor * pooled = ggml_reshape_1d(ctx, pooled_sum, head_dim);
    pooled = ggml_cont(ctx, pooled);
    pooled = build_rms_norm(ctx, pooled, norm_weight, rms_eps);
    pooled = ggml_reshape_2d(ctx, pooled, head_dim, 1);

    ggml_tensor * comp_pos = comp_pos_inp;
    if (!comp_pos) {
        comp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_input(comp_pos);
        i32_array_inputs.push_back({comp_pos, {token_pos + 1 - ratio}});
    }
    const float rope_scale = rope_scale_factor > 0.0f ? (1.0f / rope_scale_factor) : 1.0f;
    float rope_attn = 1.0f;
    if (rope_scale > 0.0f) {
        rope_attn /= (1.0f + 0.1f * logf(1.0f / rope_scale));
    }
    pooled = build_tail_rope_2d(ctx, pooled, comp_pos, n_rot, head_dim, 1,
                                compress_rope_freq_base, rope_scale, 1.0f, rope_attn,
                                rope_yarn_beta_fast, rope_yarn_beta_slow, rope_orig_ctx);

    ggml_tensor * pooled_f16 = ggml_cast(ctx, pooled, GGML_TYPE_F16);
    const int comp_row = token_pos / ratio;
    if (comp_row >= (int) comp_cache->ne[1]) {
        return;
    }

    if (comp_rows_inp) {
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, comp_cache, pooled, comp_rows_inp));
    } else {
        ggml_tensor * comp_slot = ggml_view_2d(
            ctx, comp_cache, head_dim, 1, comp_cache->nb[1],
            (size_t)comp_row * comp_cache->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, pooled_f16, comp_slot));
    }

    if (ratio == 4) {
        for (int r = 0; r < ratio; ++r) {
            ggml_tensor * src_kv = ggml_view_2d(ctx, state.state_kv, comp_width, 1,
                                                state.state_kv->nb[1],
                                                (size_t)(ratio + r) * state.state_kv->nb[1]);
            ggml_tensor * dst_kv = ggml_view_2d(ctx, state.state_kv, comp_width, 1,
                                                state.state_kv->nb[1],
                                                (size_t)r * state.state_kv->nb[1]);
            ggml_tensor * src_sc = ggml_view_2d(ctx, state.state_score, comp_width, 1,
                                                state.state_score->nb[1],
                                                (size_t)(ratio + r) * state.state_score->nb[1]);
            ggml_tensor * dst_sc = ggml_view_2d(ctx, state.state_score, comp_width, 1,
                                                state.state_score->nb[1],
                                                (size_t)r * state.state_score->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, src_kv, dst_kv));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, src_sc, dst_sc));
            ggml_tensor * dup_kv = ggml_view_2d(ctx, state.state_kv, comp_width, 1,
                                                state.state_kv->nb[1],
                                                (size_t)(ratio + r) * state.state_kv->nb[1]);
            ggml_tensor * dup_sc = ggml_view_2d(ctx, state.state_score, comp_width, 1,
                                                state.state_score->nb[1],
                                                (size_t)(ratio + r) * state.state_score->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, dst_kv, dup_kv));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, dst_sc, dup_sc));
        }
    }
}

static void build_indexer_compressor_step(
        ggml_context * ctx,
        ggml_cgraph * gf,
        ggml_tensor * cur_last,
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L,
        DeepSeek4LayerCache & lc,
        int token_pos,
        ggml_tensor * ape_row_inp,
        ggml_tensor * state_rows_inp,
        ggml_tensor * comp_rows_inp,
        ggml_tensor * comp_pos_inp,
        std::vector<DeepSeek4I64ArrayBinding> & i64_array_inputs,
        std::vector<DeepSeek4I32ArrayBinding> & i32_array_inputs) {
    build_compressor_step(ctx, gf, cur_last,
                          L.indexer_compressor_ape,
                          L.indexer_compressor_kv,
                          L.indexer_compressor_gate,
                          L.indexer_compressor_norm,
                          lc.indexer_compressor,
                          lc.index_comp_kv,
                          4,
                          w.n_indexer_head_dim,  // indexer head_dim = 128
                          token_pos,
                          w.n_rot,
                          w.rms_eps,
                          w.compress_rope_freq_base,
                          w.rope_scale_factor,
                          w.rope_yarn_beta_fast,
                          w.rope_yarn_beta_slow,
                          (int)w.rope_orig_ctx,
                          ape_row_inp,
                          state_rows_inp,
                          comp_rows_inp,
                          comp_pos_inp,
                          i64_array_inputs,
                          i32_array_inputs);
}

static int ds4_comp_rows_used(const ggml_tensor * comp_cache, int n_cached, int ratio, int token_pos) {
    if (!comp_cache || ratio <= 0) {
        return 0;
    }
    const int grew_this_step = ((token_pos + 1) % ratio) == 0 ? 1 : 0;
    return std::min(n_cached + grew_this_step, (int) comp_cache->ne[1]);
}

static ggml_tensor * build_indexer_score(
        ggml_context * ctx,
        ggml_tensor * qr_norm_last,   // [n_lora_q, 1]
        ggml_tensor * cur_last,       // [n_embd, 1]
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L,
        const DeepSeek4LayerCache & lc,
        int token_pos,
        std::vector<DeepSeek4I32InputBinding> & i32_inputs) {
    const int n_comp = ds4_comp_rows_used(lc.index_comp_kv, lc.n_index_comp, 4, token_pos);
    if (!qr_norm_last || !cur_last || !L.indexer_attn_q_b || !L.indexer_proj ||
        !lc.index_comp_kv || n_comp <= 0) {
        return nullptr;
    }

    const int n_indexer_head = w.n_indexer_head;
    const int head_dim = w.n_indexer_head_dim;

    // DS4 indexer decode scoring mirrors ds4.c::indexer_allowed_decode_one():
    //   1. Build an indexer query from qr_norm (after q_a + RMSNorm, before q_b).
    //   2. Apply full-dim RoPE in indexer head space.
    //   3. Project per-head scalar weights from the current hidden state.
    //   4. Score every compressed row with ReLU(dot(key_h, query_h)) * weight_h.
    //   5. Return the top-k compressed-row indices.
    ggml_tensor * index_q = ggml_mul_mat(ctx, L.indexer_attn_q_b, qr_norm_last);
    index_q = ggml_reshape_3d(ctx, index_q, head_dim, n_indexer_head, 1);

    // TODO: RoPE on indexer query (same gallocr issue as compressor RoPE)
    // Skipping for now — correctness deferred.
    index_q = ggml_reshape_2d(ctx, index_q, head_dim, n_indexer_head);

    ggml_tensor * head_weights = ggml_mul_mat(ctx, L.indexer_proj, cur_last);
    head_weights = ggml_scale(ctx, head_weights,
                              1.0f / std::sqrt((float) head_dim * (float) n_indexer_head));

    // index_comp_kv: [n_indexer_head_dim, comp_cap] — each row is 128-dim
    // Score each compressed row against all query heads via broadcast
    ggml_tensor * comp_view = ggml_view_2d(ctx, lc.index_comp_kv,
                                           head_dim, n_comp,
                                           lc.index_comp_kv->nb[1], 0);
    comp_view = ggml_cast(ctx, comp_view, GGML_TYPE_F32);
    // comp_view: [head_dim, n_comp] → [head_dim, 1, n_comp] for broadcast
    comp_view = ggml_reshape_3d(ctx, comp_view, head_dim, 1, n_comp);

    // index_q: [head_dim, n_indexer_head, 1] → repeat to [head_dim, n_indexer_head, n_comp]
    // But ggml_mul needs same shapes, so use matmul approach:
    // Reshape q: [head_dim, n_indexer_head] → used directly as A in matmul
    // comp: [head_dim, n_comp]
    // matmul: A^T @ B = [n_indexer_head, n_comp] dot scores
    ggml_tensor * comp_2d = ggml_reshape_2d(ctx, comp_view, head_dim, n_comp);
    // mul_mat(index_q, comp_2d): A=[head_dim, n_indexer_head], B=[head_dim, n_comp]
    // → result=[n_indexer_head, n_comp]
    ggml_tensor * dots = ggml_mul_mat(ctx, index_q, comp_2d);
    dots = ggml_relu(ctx, dots);

    // Weight each head's contribution: dots[n_indexer_head, n_comp] * weights[n_indexer_head, 1]
    ggml_tensor * weight_rep = ggml_repeat(ctx, head_weights, dots);
    ggml_tensor * weighted = ggml_mul(ctx, dots, weight_rep);
    // Sum across heads (ne[0]) → [1, n_comp]
    ggml_tensor * scores = ggml_sum_rows(ctx, weighted);
    scores = ggml_cont(ctx, scores);
    scores = ggml_reshape_2d(ctx, scores, n_comp, 1);

    return ggml_top_k(ctx, scores, std::min(n_comp, w.n_indexer_top_k));
}

static ggml_tensor * build_selected_comp_context(
        ggml_context * ctx,
        ggml_tensor * selected_rows,  // [head_dim, n_selected]
        ggml_tensor * query_seed,     // [head_dim, 1]
        ggml_tensor * q_template,     // [head_dim, n_head, n_tokens]
        int head_dim) {
    if (!selected_rows || !query_seed || !q_template || selected_rows->ne[1] <= 0) {
        return nullptr;
    }

    ggml_tensor * score = ggml_mul_mat(ctx, selected_rows, query_seed);
    ggml_tensor * probs = ggml_soft_max(ctx, score);
    ggml_tensor * rows_t = ggml_cont(ctx, ggml_transpose(ctx, selected_rows));
    ggml_tensor * context = ggml_mul_mat(ctx, rows_t, probs);
    context = ggml_reshape_3d(ctx, context, head_dim, 1, 1);
    return ggml_repeat(ctx, context, q_template);
}

// ─── MLA Attention Block ────────────────────────────────────────────────

static ggml_tensor * build_mla_attention(
        ggml_context * ctx,
        ggml_cgraph * gf,
        ggml_tensor * cur,           // [n_embd, n_tokens]
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L,
        DeepSeek4LayerCache & lc,
        int layer_idx,
        int kv_start,
        int n_tokens,
        const DeepSeek4AttentionGraphInputs * cached_inputs,
        std::vector<DeepSeek4I32InputBinding> & i32_inputs,
        std::vector<DeepSeek4I32ArrayBinding> & i32_array_inputs,
        std::vector<DeepSeek4I64ArrayBinding> & i64_array_inputs) {

    const int n_embd    = w.n_embd;
    const int head_dim  = w.head_dim;
    const int n_head    = w.n_head;
    const int n_lora_q  = w.n_lora_q;
    const int n_rot     = w.n_rot;
    const int n_out_group = w.n_out_group;
    const int n_lora_o  = w.n_lora_o;
    const int ratio     = w.compress_ratios[layer_idx];

    // ── Q path: cur → q_a → norm → q_b → per-head norm ─────────────
    // q_a: [n_embd, n_tokens] → [n_lora_q, n_tokens]
    ggml_tensor * qr = ggml_mul_mat(ctx, L.attn_q_a, cur);
    // qr_norm is reused by the ratio-4 indexer before the main q_b projection.
    qr = build_rms_norm(ctx, qr, L.attn_q_a_norm, w.rms_eps);
    // q_b: [n_lora_q, n_tokens] → [n_head * head_dim, n_tokens]
    ggml_tensor * q = ggml_mul_mat(ctx, L.attn_q_b, qr);
    // Reshape to [head_dim, n_head, n_tokens] for per-head ops
    q = ggml_reshape_3d(ctx, q, head_dim, n_head, n_tokens);
    // Reference DS4 applies unweighted RMSNorm independently to every Q head.
    q = ggml_rms_norm(ctx, q, w.rms_eps);

    // ── KV path: cur → kv → norm ───────────────────────────────────
    // kv: [n_embd, n_tokens] → [head_dim, n_tokens]
    ggml_tensor * kv = ggml_mul_mat(ctx, L.attn_kv, cur);
    kv = build_rms_norm(ctx, kv, L.attn_kv_a_norm, w.rms_eps);

    // ── RoPE on Q and KV (tail rotation on last n_rot dims) ────────
    // DS4 uses per-layer RoPE params: compressed layers get YaRN scaling.
    const bool compressed = (ratio > 0);
    const float rope_freq = compressed ? w.compress_rope_freq_base : w.rope_freq_base;
    const float rope_scale = compressed ? (1.0f / w.rope_scale_factor) : 1.0f;
    const float rope_ext = compressed ? 1.0f : 0.0f;
    // For YaRN: attn_factor cancels the magnitude scaling in rope_yarn
    float rope_attn = 1.0f;
    if (rope_ext != 0.0f && rope_scale > 0.0f) {
        rope_attn /= (1.0f + 0.1f * logf(1.0f / rope_scale));
    }

    // Position tensor for this token batch
    ggml_tensor * rope_pos = cached_inputs ? cached_inputs->rope_pos : nullptr;
    if (!rope_pos) {
        rope_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(rope_pos);
        std::vector<int32_t> pos_vals(n_tokens);
        for (int i = 0; i < n_tokens; i++) pos_vals[i] = kv_start + i;
        i32_array_inputs.push_back({rope_pos, std::move(pos_vals)});
    }

    // n_ctx_orig is critical for YaRN correction on compressed layers
    const int rope_n_ctx_orig = (int)w.rope_orig_ctx;  // 65536

    q = build_tail_rope_3d(ctx, q, rope_pos, n_rot, head_dim, n_head, n_tokens,
                           rope_freq, rope_scale, rope_ext, rope_attn,
                           w.rope_yarn_beta_fast, w.rope_yarn_beta_slow, rope_n_ctx_orig);
    kv = build_tail_rope_2d(ctx, kv, rope_pos, n_rot, head_dim, n_tokens,
                            rope_freq, rope_scale, rope_ext, rope_attn,
                            w.rope_yarn_beta_fast, w.rope_yarn_beta_slow, rope_n_ctx_orig);

    // ── Store ALL KV rows in the raw SWA ring ─────────────────────
    // For decode (n_tokens=1): write single row. For prefill: write all rows.
    if (cached_inputs && cached_inputs->raw_kv_rows) {
        ggml_tensor * kv_f32 = ggml_is_contiguous(kv) ? kv : ggml_cont(ctx, kv);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, lc.raw_kv, kv_f32, cached_inputs->raw_kv_rows));
    } else {
        for (int ti = 0; ti < n_tokens; ti++) {
            const int pos_ti = kv_start + ti;
            ggml_tensor * kv_row = ggml_view_2d(
                ctx, kv, head_dim, 1, kv->nb[1], (size_t)ti * kv->nb[1]);
            ggml_tensor * kv_slot = ggml_view_2d(
                ctx, lc.raw_kv, head_dim, 1, lc.raw_kv->nb[1],
                (size_t)(pos_ti % w.n_swa) * lc.raw_kv->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_cast(ctx, kv_row, GGML_TYPE_F16), kv_slot));
        }
    }
    const int token_pos = kv_start + n_tokens - 1;

    // ── Learned compression update ──────────────────────────────────
    ggml_tensor * cur_last = ggml_view_2d(
        ctx, cur, n_embd, 1, cur->nb[1], (size_t)(n_tokens - 1) * cur->nb[1]);
    ggml_tensor * qr_last = ggml_view_2d(
        ctx, qr, n_lora_q, 1, qr->nb[1], (size_t)(n_tokens - 1) * qr->nb[1]);
    if (ratio > 0 && L.attn_compressor_kv) {
        build_compressor_step(ctx, gf, cur_last,
                              L.attn_compressor_ape,
                              L.attn_compressor_kv,
                              L.attn_compressor_gate,
                              L.attn_compressor_norm,
                              lc.attn_compressor,
                              lc.comp_kv,
                              ratio,
                              head_dim,
                              token_pos,
                              w.n_rot,
                              w.rms_eps,
                              w.compress_rope_freq_base,
                              w.rope_scale_factor,
                              w.rope_yarn_beta_fast,
                              w.rope_yarn_beta_slow,
                              (int)w.rope_orig_ctx,
                              cached_inputs ? cached_inputs->attn_ape_row : nullptr,
                              cached_inputs ? cached_inputs->attn_state_rows : nullptr,
                              cached_inputs ? cached_inputs->attn_comp_rows : nullptr,
                              cached_inputs ? cached_inputs->attn_comp_pos : nullptr,
                              i64_array_inputs,
                              i32_array_inputs);
    }

    if (ratio == 4 && L.indexer_compressor_kv) {
        build_indexer_compressor_step(ctx, gf, cur_last, w, L, lc, token_pos,
                                      cached_inputs ? cached_inputs->index_ape_row : nullptr,
                                      cached_inputs ? cached_inputs->index_state_rows : nullptr,
                                      cached_inputs ? cached_inputs->index_comp_rows : nullptr,
                                      cached_inputs ? cached_inputs->index_comp_pos : nullptr,
                                      i64_array_inputs,
                                      i32_array_inputs);
        (void)build_indexer_score(ctx, qr_last, cur_last, w, L, lc, token_pos, i32_inputs);
    }

    // ── MLA Dot-Product Attention (SWA + compressed KV) ────────────
    // q: [head_dim, n_head, n_tokens] (after RoPE)
    // raw_kv: [head_dim, n_swa] F16 persistent ring buffer (single KV head, shared)
    // comp_kv: [head_dim, comp_cap] F16 compressed rows.
    // n_raw = min(kv_start + n_tokens, n_swa)
    const int n_raw = std::min(kv_start + n_tokens, w.n_swa);
    const int n_comp_attn = (ratio > 0) ? ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio, token_pos) : 0;
    const int n_attn = n_raw + n_comp_attn;
    const float kq_scale = 1.0f / sqrtf((float)head_dim);

    // Get valid KV rows. For single-token decode/prefill, include the current
    // in-graph KV row directly; otherwise attention can race the side-effecting
    // cache write and see the previous contents of the raw KV slot.
    ggml_tensor * kv_f32 = nullptr;
    if (n_tokens == 1) {
        ggml_tensor * cur_kv_f32 = ggml_cast(ctx, kv, GGML_TYPE_F32);
        if (n_raw > 1) {
            ggml_tensor * prev_f32 = ggml_cast(ctx,
                ggml_view_2d(ctx, lc.raw_kv, head_dim, n_raw - 1,
                             lc.raw_kv->nb[1], 0),
                GGML_TYPE_F32);
            kv_f32 = ggml_concat(ctx, prev_f32, cur_kv_f32, 1);
        } else {
            kv_f32 = cur_kv_f32;
        }
    } else {
        kv_f32 = ggml_cast(ctx, ggml_view_2d(ctx, lc.raw_kv, head_dim, n_raw,
                                             lc.raw_kv->nb[1], 0), GGML_TYPE_F32);
    }
    if (n_comp_attn > 0 && lc.comp_kv) {
        ggml_tensor * comp_f32 = ggml_cast(ctx,
            ggml_view_2d(ctx, lc.comp_kv, head_dim, n_comp_attn, lc.comp_kv->nb[1], 0),
            GGML_TYPE_F32);
        kv_f32 = ggml_concat(ctx, kv_f32, comp_f32, 1);
    }
    // kv_f32: [head_dim, n_attn]

    // Flatten q to [head_dim, n_head*n_tokens] for batched matmul
    ggml_tensor * q_flat = ggml_reshape_2d(ctx, q, head_dim, n_head * n_tokens);

    // Scores: mul_mat(kv_f32, q_flat) = kv_f32^T[n_attn, head_dim] @ q_flat[head_dim, n_head*n_tokens]
    //       → [n_attn, n_head*n_tokens]
    ggml_tensor * scores = ggml_mul_mat(ctx, kv_f32, q_flat);
    scores = ggml_scale(ctx, scores, kq_scale);

    // Sink-aware softmax: DS4 adds one learned per-head sink logit to the
    // denominator, but the sink contributes no value vector.
    ggml_tensor * probs = nullptr;
    if (L.attn_sinks) {
        ggml_tensor * sink_scores = ggml_reshape_2d(ctx, L.attn_sinks, 1, n_head);
        if (n_tokens > 1) {
            ggml_tensor * sink_shape = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_head * n_tokens);
            sink_scores = ggml_repeat(ctx, sink_scores, sink_shape);
        }
        ggml_tensor * scores_with_sink = ggml_concat(ctx, scores, sink_scores, 0);
        ggml_tensor * probs_with_sink = ggml_soft_max(ctx, scores_with_sink);
        probs = ggml_view_2d(ctx, probs_with_sink, n_attn, n_head * n_tokens,
                             probs_with_sink->nb[1], 0);
    } else {
        probs = ggml_soft_max(ctx, scores);
    }
    // probs: [n_attn, n_head*n_tokens]

    // Context: kv_T^T[head_dim, n_attn] @ probs[n_attn, n_head*n_tokens] → [head_dim, n_head*n_tokens]
    // i.e. mul_mat(kv_T, probs) where kv_T = cont(transpose(kv_f32)) = [n_raw, head_dim]
    ggml_tensor * kv_T = ggml_cont(ctx, ggml_transpose(ctx, kv_f32));
    ggml_tensor * context = ggml_mul_mat(ctx, kv_T, probs);
    // context: [head_dim, n_head*n_tokens]

    // Reshape back to [head_dim, n_head, n_tokens]
    context = ggml_reshape_3d(ctx, context, head_dim, n_head, n_tokens);

    // ── Inverse tail RoPE on attention output ───────────────────────
    ggml_tensor * neg_pos = cached_inputs ? cached_inputs->neg_pos : nullptr;
    if (!neg_pos) {
        neg_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(neg_pos);
        std::vector<int32_t> neg_vals(n_tokens);
        for (int i = 0; i < n_tokens; i++) neg_vals[i] = -(kv_start + i);
        i32_array_inputs.push_back({neg_pos, std::move(neg_vals)});
    }
    context = build_tail_rope_3d(ctx, context, neg_pos, n_rot, head_dim, n_head, n_tokens,
                                 rope_freq, rope_scale, rope_ext, rope_attn,
                                 w.rope_yarn_beta_fast, w.rope_yarn_beta_slow, rope_n_ctx_orig);

    // Flatten to [head_dim*n_head, n_tokens] for output projection
    ggml_tensor * attn_out = ggml_reshape_2d(ctx, context, head_dim * n_head, n_tokens);

    // ── Grouped output projection ──────────────────────────────────
    // DS4 output uses grouped low-rank projection:
    //   attn_out: [head_dim*n_head, n_tokens] → reshape [group_dim, n_tokens, n_groups]
    //   out_a: [group_dim, n_groups*n_lora_o] → reshape [group_dim, n_lora_o, n_groups]
    //   batched matmul over n_groups: → [n_lora_o, n_tokens, n_groups]
    //   → reshape [n_lora_o*n_groups, n_tokens]
    //   out_b: [n_lora_o*n_groups, n_embd] → final: [n_embd, n_tokens]
    const int group_dim = head_dim * (n_head / n_out_group);  // 512 * 8 = 4096
    // Reshape attn_out: [32768, n_tokens] → [4096, 8, n_tokens] → permute to [4096, n_tokens, 8]
    attn_out = ggml_reshape_3d(ctx, attn_out, group_dim, n_out_group, n_tokens);
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
    // attn_out is now [group_dim, n_tokens, n_out_group]
    ggml_tensor * out_a_3d = ggml_reshape_3d(ctx, L.attn_output_a, group_dim, n_lora_o, n_out_group);
    // out_a_3d: [group_dim, n_lora_o, n_out_group] — ne[2] matches
    ggml_tensor * attn_low = ggml_mul_mat(ctx, out_a_3d, attn_out);
    // attn_low: [n_lora_o, n_tokens, n_out_group]
    // Permute back to [n_lora_o, n_out_group, n_tokens] then flatten
    attn_low = ggml_cont(ctx, ggml_permute(ctx, attn_low, 0, 2, 1, 3));
    attn_low = ggml_reshape_2d(ctx, attn_low, n_lora_o * n_out_group, n_tokens);
    ggml_tensor * out = ggml_mul_mat(ctx, L.attn_output_b, attn_low);

    return out;
}

struct DeepSeek4CachedDecodeHcPreGraph {
    const ggml_context * owner_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    int layer_idx = -1;
    bool ffn = false;
    StepGraph sg;
    ggml_tensor * post = nullptr;
    ggml_tensor * comb = nullptr;

    bool valid() const {
        return owner_ctx && backend && layer_idx >= 0 &&
               sg.ctx && sg.gf && sg.alloc && sg.inp_embed && sg.hidden_states &&
               post && comb;
    }

    void free() {
        step_graph_destroy(sg);
        owner_ctx = nullptr;
        backend = nullptr;
        layer_idx = -1;
        ffn = false;
        post = nullptr;
        comb = nullptr;
    }
};

struct DeepSeek4CachedDecodeHcPostGraph {
    const ggml_context * owner_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    StepGraph sg;
    ggml_tensor * residual_hc = nullptr;
    ggml_tensor * block_out = nullptr;
    ggml_tensor * post = nullptr;
    ggml_tensor * comb = nullptr;

    bool valid() const {
        return owner_ctx && backend &&
               sg.ctx && sg.gf && sg.alloc && sg.hidden_states &&
               residual_hc && block_out && post && comb;
    }

    void free() {
        step_graph_destroy(sg);
        owner_ctx = nullptr;
        backend = nullptr;
        residual_hc = nullptr;
        block_out = nullptr;
        post = nullptr;
        comb = nullptr;
    }
};

static bool build_cached_decode_attn_graph(
        DeepSeek4CachedDecodeAttnGraph & out,
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L,
        DeepSeek4LayerCache & lc,
        int layer_idx,
        int kv_start,
        int raw_attn_count,
        int comp_attn_count,
        int index_comp_count) {
    out.free();

    const size_t ctx_size = 48 * 1024 * 1024;
    ggml_init_params params{};
    params.mem_size = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    out.sg.ctx = ggml_init(params);
    if (!out.sg.ctx) {
        return false;
    }

    const int ratio = w.compress_ratios[layer_idx];
    out.n_tokens = 1;
    out.n_raw = raw_attn_count;
    out.n_comp_attn = comp_attn_count;
    out.n_index_comp = index_comp_count;
    out.compressed = ratio > 0;
    out.indexed = ratio == 4;

    out.sg.inp_embed = ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_F32, w.n_embd, 1);
    ggml_set_input(out.sg.inp_embed);
    out.sg.gf = ggml_new_graph_custom(out.sg.ctx, 2048, false);

    out.inputs.rope_pos = ggml_new_tensor_1d(out.sg.ctx, GGML_TYPE_I32, 1);
    out.inputs.neg_pos = ggml_new_tensor_1d(out.sg.ctx, GGML_TYPE_I32, 1);
    ggml_set_input(out.inputs.rope_pos);
    ggml_set_input(out.inputs.neg_pos);

    out.inputs.raw_kv_rows = ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_I64, 1, 1);
    ggml_set_input(out.inputs.raw_kv_rows);
    if (ratio > 0) {
        out.inputs.attn_ape_row = ggml_new_tensor_1d(out.sg.ctx, GGML_TYPE_I32, 1);
        out.inputs.attn_comp_pos = ggml_new_tensor_1d(out.sg.ctx, GGML_TYPE_I32, 1);
        out.inputs.attn_state_rows = ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_I64, 1, 1);
        out.inputs.attn_comp_rows = ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_I64, 1, 1);
        ggml_set_input(out.inputs.attn_ape_row);
        ggml_set_input(out.inputs.attn_comp_pos);
        ggml_set_input(out.inputs.attn_state_rows);
        ggml_set_input(out.inputs.attn_comp_rows);
    }
    if (ratio == 4) {
        out.inputs.index_ape_row = ggml_new_tensor_1d(out.sg.ctx, GGML_TYPE_I32, 1);
        out.inputs.index_comp_pos = ggml_new_tensor_1d(out.sg.ctx, GGML_TYPE_I32, 1);
        out.inputs.index_state_rows = ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_I64, 1, 1);
        out.inputs.index_comp_rows = ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_I64, 1, 1);
        ggml_set_input(out.inputs.index_ape_row);
        ggml_set_input(out.inputs.index_comp_pos);
        ggml_set_input(out.inputs.index_state_rows);
        ggml_set_input(out.inputs.index_comp_rows);
    }

    std::vector<DeepSeek4I32InputBinding> i32_inputs;
    std::vector<DeepSeek4I32ArrayBinding> i32_array_inputs;
    std::vector<DeepSeek4I64ArrayBinding> i64_array_inputs;
    ggml_tensor * normed = build_rms_norm(out.sg.ctx, out.sg.inp_embed, L.attn_norm, w.rms_eps);
    out.sg.hidden_states = build_mla_attention(out.sg.ctx, out.sg.gf, normed, w, L, lc, layer_idx,
                                               kv_start, 1, &out.inputs,
                                               i32_inputs, i32_array_inputs, i64_array_inputs);
    if (!out.sg.hidden_states) {
        out.free();
        return false;
    }
    ggml_set_output(out.sg.hidden_states);
    ggml_build_forward_expand(out.sg.gf, out.sg.hidden_states);

    out.sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(out.sg.alloc, out.sg.gf)) {
        out.free();
        return false;
    }

    out.owner_ctx = w.ctx;
    out.backend = backend;
    out.layer_idx = layer_idx;
    return true;
}

static ggml_tensor * ds4_hc_row_normalize(ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * sums = ggml_sum_rows(ctx, x);
    return ggml_div(ctx, x, ggml_repeat(ctx, sums, x));
}

static ggml_tensor * ds4_hc_col_normalize(ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, x));
    xt = ds4_hc_row_normalize(ctx, xt);
    return ggml_cont(ctx, ggml_transpose(ctx, xt));
}

static bool ds4_backend_is_hip(ggml_backend_t backend) {
    const char * name = ggml_backend_name(backend);
    return name &&
        (std::strstr(name, "HIP") != nullptr ||
         std::strstr(name, "ROCm") != nullptr);
}

static bool ds4_backend_is_cuda(ggml_backend_t backend) {
    const char * name = ggml_backend_name(backend);
    return name && std::strstr(name, "CUDA") != nullptr;
}

static bool ds4_backend_is_gpu(ggml_backend_t backend) {
    return ds4_backend_is_hip(backend) || ds4_backend_is_cuda(backend);
}

static bool ds4_try_gpu_hc_pre(float * working,
                               float * post,
                               float * comb,
                               const float * hc_state,
                               const float * scale_data,
                               const float * base_data,
                               ggml_tensor * fn_tensor,
                               int n_embd,
                               int n_hc,
                               int sinkhorn_iters,
                               float hc_eps) {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    if (!fn_tensor || !fn_tensor->data) {
        return false;
    }
    return deepseek4_cuda_hc_pre(hc_state,
                                 fn_tensor->data,
                                 scale_data,
                                 base_data,
                                 n_embd,
                                 n_hc,
                                 sinkhorn_iters,
                                 hc_eps,
                                 working,
                                 post,
                                 comb);
#else
    (void) working;
    (void) post;
    (void) comb;
    (void) hc_state;
    (void) scale_data;
    (void) base_data;
    (void) fn_tensor;
    (void) n_embd;
    (void) n_hc;
    (void) sinkhorn_iters;
    (void) hc_eps;
    return false;
#endif
}

static bool ds4_try_gpu_hc_pre_device(ggml_tensor * working,
                                      ggml_tensor * post,
                                      ggml_tensor * comb,
                                      ggml_backend_t backend,
                                      int layer_idx,
                                      bool ffn,
                                      ggml_tensor * hc_state,
                                      ggml_tensor * fn_tensor,
                                      ggml_tensor * scale_tensor,
                                      ggml_tensor * base_tensor,
                                      const float * scale_data,
                                      const float * base_data,
                                      int n_embd,
                                      int n_hc,
                                      int sinkhorn_iters,
                                      float hc_eps) {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    if (!working || !post || !comb || !hc_state || !fn_tensor || !scale_data || !base_data ||
        !working->data || !post->data || !comb->data || !hc_state->data || !fn_tensor->data) {
        return false;
    }
    const bool can_use_device_params =
        ds4_backend_is_cuda(backend) &&
        scale_tensor && base_tensor &&
        scale_tensor->data && base_tensor->data &&
        scale_tensor->buffer && base_tensor->buffer &&
        !ggml_backend_buffer_is_host(scale_tensor->buffer) &&
        !ggml_backend_buffer_is_host(base_tensor->buffer);
    if (can_use_device_params) {
        return deepseek4_cuda_hc_pre_device(hc_state->data,
                                            fn_tensor->data,
                                            scale_tensor->data,
                                            base_tensor->data,
                                            n_embd,
                                            n_hc,
                                            sinkhorn_iters,
                                            hc_eps,
                                            working->data,
                                            post->data,
                                            comb->data);
    }
    return deepseek4_cuda_hc_pre_device_params(hc_state->data,
                                               fn_tensor->data,
                                               scale_data,
                                               base_data,
                                               n_embd,
                                               n_hc,
                                               sinkhorn_iters,
                                               hc_eps,
                                               working->data,
                                               post->data,
                                               comb->data);
#else
    (void) working;
    (void) post;
    (void) comb;
    (void) backend;
    (void) layer_idx;
    (void) ffn;
    (void) hc_state;
    (void) fn_tensor;
    (void) scale_tensor;
    (void) base_tensor;
    (void) scale_data;
    (void) base_data;
    (void) n_embd;
    (void) n_hc;
    (void) sinkhorn_iters;
    (void) hc_eps;
    return false;
#endif
}

static bool build_cached_decode_hc_pre_graph(
        DeepSeek4CachedDecodeHcPreGraph & out,
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L,
        const float * scale_data,
        int layer_idx,
        bool ffn) {
    out.free();

    const size_t ctx_size = 4 * 1024 * 1024;
    ggml_init_params params{};
    params.mem_size = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    out.sg.ctx = ggml_init(params);
    if (!out.sg.ctx) {
        return false;
    }

    const int hc_dim = w.n_hc * w.n_embd;
    ggml_tensor * hc_fn = ffn ? L.hc_ffn_fn : L.hc_attn_fn;
    ggml_tensor * hc_base = ffn ? L.hc_ffn_base : L.hc_attn_base;

    out.sg.inp_embed = ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_F32, hc_dim, 1);
    ggml_set_input(out.sg.inp_embed);
    out.sg.gf = ggml_new_graph_custom(out.sg.ctx, 2048, false);

    ggml_tensor * flat = ggml_rms_norm(out.sg.ctx, out.sg.inp_embed, w.hc_eps);
    ggml_tensor * mix = ggml_mul_mat(out.sg.ctx, hc_fn, flat);

    ggml_tensor * pre_mix = ggml_reshape_2d(out.sg.ctx,
        ggml_view_1d(out.sg.ctx, mix, w.n_hc, 0), w.n_hc, 1);
    ggml_tensor * post_mix = ggml_reshape_2d(out.sg.ctx,
        ggml_view_1d(out.sg.ctx, mix, w.n_hc, (size_t) w.n_hc * mix->nb[0]), w.n_hc, 1);
    ggml_tensor * comb_mix = ggml_reshape_2d(out.sg.ctx,
        ggml_view_1d(out.sg.ctx, mix, w.n_hc * w.n_hc, (size_t) (2 * w.n_hc) * mix->nb[0]),
        w.n_hc, w.n_hc);

    ggml_tensor * pre_base = ggml_reshape_2d(out.sg.ctx,
        ggml_view_1d(out.sg.ctx, hc_base, w.n_hc, 0), w.n_hc, 1);
    ggml_tensor * post_base = ggml_reshape_2d(out.sg.ctx,
        ggml_view_1d(out.sg.ctx, hc_base, w.n_hc, (size_t) w.n_hc * hc_base->nb[0]), w.n_hc, 1);
    ggml_tensor * comb_base = ggml_reshape_2d(out.sg.ctx,
        ggml_view_1d(out.sg.ctx, hc_base, w.n_hc * w.n_hc, (size_t) (2 * w.n_hc) * hc_base->nb[0]),
        w.n_hc, w.n_hc);

    ggml_tensor * pre = ggml_sigmoid(out.sg.ctx,
        ggml_add(out.sg.ctx,
                 ggml_scale(out.sg.ctx, pre_mix, scale_data[0]),
                 pre_base));
    ggml_tensor * post = ggml_scale(out.sg.ctx,
        ggml_sigmoid(out.sg.ctx,
                     ggml_add(out.sg.ctx,
                              ggml_scale(out.sg.ctx, post_mix, scale_data[1]),
                              post_base)),
        2.0f);

    ggml_tensor * comb = ggml_add(out.sg.ctx,
        ggml_scale(out.sg.ctx, comb_mix, scale_data[2]),
        comb_base);
    comb = ggml_soft_max(out.sg.ctx, comb);
    comb = ds4_hc_col_normalize(out.sg.ctx, comb);
    for (int iter = 1; iter < w.n_hc_sinkhorn_iter; ++iter) {
        comb = ds4_hc_row_normalize(out.sg.ctx, comb);
        comb = ds4_hc_col_normalize(out.sg.ctx, comb);
    }

    ggml_tensor * hc_state_2d = ggml_reshape_2d(out.sg.ctx, out.sg.inp_embed, w.n_embd, w.n_hc);
    ggml_tensor * hc_state_t = ggml_cont(out.sg.ctx, ggml_transpose(out.sg.ctx, hc_state_2d));
    ggml_tensor * working = ggml_mul_mat(out.sg.ctx, hc_state_t, pre);

    out.sg.hidden_states = working;
    out.post = post;
    out.comb = comb;
    ggml_set_output(out.sg.hidden_states);
    ggml_set_output(out.post);
    ggml_set_output(out.comb);
    ggml_build_forward_expand(out.sg.gf, out.sg.hidden_states);
    ggml_build_forward_expand(out.sg.gf, out.post);
    ggml_build_forward_expand(out.sg.gf, out.comb);

    out.sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(out.sg.alloc, out.sg.gf)) {
        out.free();
        return false;
    }

    out.owner_ctx = w.ctx;
    out.backend = backend;
    out.layer_idx = layer_idx;
    out.ffn = ffn;
    return true;
}

static bool build_cached_decode_hc_post_graph(
        DeepSeek4CachedDecodeHcPostGraph & out,
        ggml_backend_t backend,
        const DeepSeek4Weights & w) {
    out.free();

    const size_t ctx_size = 2 * 1024 * 1024;
    ggml_init_params params{};
    params.mem_size = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    out.sg.ctx = ggml_init(params);
    if (!out.sg.ctx) {
        return false;
    }

    const int hc_dim = w.n_embd * w.n_hc;
    out.residual_hc = ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_F32, hc_dim, 1);
    out.block_out = ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_F32, w.n_embd, 1);
    out.post = ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_F32, w.n_hc, 1);
    out.comb = ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_F32, w.n_hc, w.n_hc);
    ggml_set_input(out.residual_hc);
    ggml_set_input(out.block_out);
    ggml_set_input(out.post);
    ggml_set_input(out.comb);

    out.sg.gf = ggml_new_graph_custom(out.sg.ctx, 256, false);

    ggml_tensor * residual_2d = ggml_reshape_2d(out.sg.ctx, out.residual_hc, w.n_embd, w.n_hc);
    ggml_tensor * residual_t = ggml_cont(out.sg.ctx, ggml_transpose(out.sg.ctx, residual_2d));
    ggml_tensor * comb_t = ggml_cont(out.sg.ctx, ggml_transpose(out.sg.ctx, out.comb));
    ggml_tensor * mixed_t = ggml_mul_mat(out.sg.ctx, comb_t, residual_t);
    ggml_tensor * mixed = ggml_cont(out.sg.ctx, ggml_transpose(out.sg.ctx, mixed_t));
    ggml_tensor * post_t = ggml_cont(out.sg.ctx, ggml_transpose(out.sg.ctx, out.post));
    ggml_tensor * block_rep = ggml_repeat(out.sg.ctx, out.block_out, mixed);
    ggml_tensor * post_rep = ggml_repeat(out.sg.ctx, post_t, mixed);
    out.sg.hidden_states = ggml_reshape_2d(
        out.sg.ctx,
        ggml_add(out.sg.ctx, mixed, ggml_mul(out.sg.ctx, block_rep, post_rep)),
        hc_dim, 1);

    ggml_set_output(out.sg.hidden_states);
    ggml_build_forward_expand(out.sg.gf, out.sg.hidden_states);

    out.sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(out.sg.alloc, out.sg.gf)) {
        out.free();
        return false;
    }

    out.owner_ctx = w.ctx;
    out.backend = backend;
    return true;
}

// ─── MoE FFN Block ──────────────────────────────────────────────────────

struct Ds4MoeRouting {
    ggml_tensor * selected = nullptr;
    ggml_tensor * weights = nullptr;
};

static MoeHybridConfig make_ds4_moe_hybrid_config(const DeepSeek4Weights & w) {
    MoeHybridConfig cfg;
    cfg.n_embd = w.n_embd;
    cfg.n_expert = w.n_expert;
    cfg.n_expert_used = w.n_expert_used;
    cfg.n_ff_exp = w.n_ff_exp;
    cfg.n_ff_shexp = w.n_ff_exp;
    cfg.n_layer = w.n_layer;
    cfg.first_moe_layer = 0;
    cfg.swiglu_clamp = w.swiglu_clamp_exp;
    return cfg;
}

static MoeLayerDesc make_ds4_moe_layer_desc(const DeepSeek4Layer & L) {
    MoeLayerDesc desc;
    desc.ffn_gate_exps = L.ffn_gate_exps;
    desc.ffn_up_exps = L.ffn_up_exps;
    desc.ffn_down_exps = L.ffn_down_exps;
    desc.ffn_gate_up_exps = nullptr;
    desc.ffn_gate_shexp = L.ffn_gate_shexp;
    desc.ffn_up_shexp = L.ffn_up_shexp;
    desc.ffn_down_shexp = L.ffn_down_shexp;
    desc.ffn_gate_inp_shexp = nullptr;
    return desc;
}

static ggml_tensor * build_shared_ffn(
        ggml_context * ctx,
        ggml_tensor * cur,
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L) {
    ggml_tensor * gate_sh = ggml_mul_mat(ctx, L.ffn_gate_shexp, cur);
    ggml_tensor * up_sh = ggml_mul_mat(ctx, L.ffn_up_shexp, cur);
    ggml_tensor * mid_sh = build_clamped_swiglu(ctx, gate_sh, up_sh, w.swiglu_clamp_exp);
    return ggml_mul_mat(ctx, L.ffn_down_shexp, mid_sh);
}

static bool eval_ds4_hybrid(
        ggml_backend_t backend,
        ggml_backend_t cpu_backend,
        const MoeHybridConfig & hybrid_cfg,
        const MoeLayerDesc & desc,
        const MoeHybridStorage * hybrid_owner,
        MoeHybridLayerStorage & storage,
        MoeHybridStreamEngine * stream_engine,
        int layer,
        int n_embd,
        int n_expert_used,
        const float * ffn_normed_host,
        const int32_t * selected_host,
        const float * weights_host,
        int n_tokens,
        std::vector<float> & ffn_out_host,
        ggml_gallocr_t * hot_alloc,
        ggml_gallocr_t * cold_alloc,
        DeepSeek4StepTelemetry * step_tel) {
    const auto ffn_t0 = Ds4TimingClock::now();
    if (!storage.down_cold && !storage.gate_up_cold) {
        if (!hybrid_owner || !stream_engine || !stream_engine->is_ready() ||
            !hybrid_owner->has_mmap() ||
            layer < 0 || layer >= (int) hybrid_owner->layer_regions.size()) {
            std::fprintf(stderr,
                         "[deepseek4] layer %d requires cold-expert streaming but it is unavailable\n",
                         layer);
            return false;
        }

        const LayerExpertRegions & regions = hybrid_owner->layer_regions[(size_t) layer];
        ffn_out_host.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
        std::vector<int32_t> hot_selected;
        std::vector<float> hot_weights;
        std::vector<float> hot_out;
        std::vector<float> cold_out;
        for (int ti = 0; ti < n_tokens; ++ti) {
            const float * token_inp = ffn_normed_host + (size_t)ti * (size_t)n_embd;
            const int32_t * token_selected = selected_host + (size_t)ti * (size_t)n_expert_used;
            const float * token_weights = weights_host + (size_t)ti * (size_t)n_expert_used;
            hot_selected.clear();
            hot_weights.clear();
            bool has_cold = false;
            for (int ei = 0; ei < n_expert_used; ++ei) {
                const int32_t gid = token_selected[ei];
                if (gid < 0 || gid >= (int32_t) storage.hot_local_by_global.size()) {
                    std::fprintf(stderr,
                                 "[deepseek4] layer %d selected expert id out of range: %d\n",
                                 layer, (int) gid);
                    return false;
                }
                if (storage.hot_local_by_global[(size_t)gid] >= 0) {
                    hot_selected.push_back(gid);
                    hot_weights.push_back(token_weights[ei]);
                } else {
                    has_cold = true;
                }
            }

            std::string err;
            MoeHybridFfnTelemetry single_tel;
            if (!eval_moe_hybrid_ffn_single(
                    backend, hybrid_cfg, desc, storage, cpu_backend,
                    token_inp,
                    hot_selected.empty() ? nullptr : hot_selected.data(),
                    hot_weights.empty() ? nullptr : hot_weights.data(),
                    (int) hot_selected.size(),
                    hot_out,
                    step_tel ? &single_tel : nullptr,
                    &err)) {
                std::fprintf(stderr,
                             "[deepseek4] layer %d hot/shared eval failed: %s\n",
                             layer, err.c_str());
                return false;
            }
            add_ffn_telemetry(step_tel, single_tel);

            cold_out.assign((size_t)n_embd, 0.0f);
            if (has_cold) {
                if (!eval_moe_cold_experts_streaming(
                        *stream_engine, backend,
                        hybrid_owner->mmap_data, hybrid_owner->mmap_size,
                        hybrid_cfg, desc, regions, storage,
                        token_inp, token_selected, token_weights, 1,
                        cold_out, &err)) {
                    std::fprintf(stderr,
                                 "[deepseek4] layer %d cold streaming eval failed: %s\n",
                                 layer, err.c_str());
                    return false;
                }
            }

            float * dst = ffn_out_host.data() + (size_t)ti * (size_t)n_embd;
            for (int i = 0; i < n_embd; ++i) {
                dst[i] = hot_out[(size_t)i] + cold_out[(size_t)i];
            }
        }
        if (step_tel) step_tel->ffn_eval_us += ds4_elapsed_us(ffn_t0, Ds4TimingClock::now());
        return true;
    }

    MoeHybridFfnTelemetry ffn_tel;
    bool ffn_ok = eval_moe_hybrid_ffn_batched(
        backend, cpu_backend, hybrid_cfg, desc, storage,
        ffn_normed_host, selected_host, weights_host,
        n_tokens, ffn_out_host, nullptr, hot_alloc, cold_alloc,
        nullptr, nullptr,
        step_tel ? &ffn_tel : nullptr);
    if (ffn_ok) {
        if (step_tel) {
            step_tel->ffn_eval_us += ds4_elapsed_us(ffn_t0, Ds4TimingClock::now());
            add_ffn_telemetry(step_tel, ffn_tel);
        }
        return true;
    }

    ffn_out_host.assign((size_t)n_embd * (size_t)n_tokens, 0.0f);
    std::vector<float> single_out;
    for (int ti = 0; ti < n_tokens; ++ti) {
        MoeHybridFfnTelemetry single_tel;
        if (!eval_moe_hybrid_ffn_single(
                backend, hybrid_cfg, desc, storage, cpu_backend,
                ffn_normed_host + (size_t)ti * (size_t)n_embd,
                selected_host + (size_t)ti * (size_t)n_expert_used,
                weights_host + (size_t)ti * (size_t)n_expert_used,
                n_expert_used, single_out,
                step_tel ? &single_tel : nullptr)) {
            return false;
        }
        add_ffn_telemetry(step_tel, single_tel);
        std::memcpy(ffn_out_host.data() + (size_t)ti * (size_t)n_embd,
                    single_out.data(), sizeof(float) * (size_t)n_embd);
    }
    if (step_tel) step_tel->ffn_eval_us += ds4_elapsed_us(ffn_t0, Ds4TimingClock::now());
    return true;
}

static Ds4MoeRouting build_moe_routing(
        ggml_context * ctx,
        ggml_tensor * cur,
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L,
        int n_tokens) {
    Ds4MoeRouting out;
    ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, cur);

    // DS4 routes with sqrt(softplus(logit)). Optional bias affects only the
    // top-k expert selection, while expert weights come from the unbiased
    // router probabilities and are normalized after selection.
    ggml_tensor * probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));
    ggml_tensor * selection = probs;
    if (L.ffn_exp_probs_b) {
        selection = ggml_add(ctx, selection, L.ffn_exp_probs_b);
    }

    out.selected = ggml_top_k(ctx, selection, w.n_expert_used);
    ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, w.n_expert, n_tokens);
    out.weights = ggml_get_rows(ctx, probs_3d, out.selected);
    out.weights = ggml_reshape_2d(ctx, out.weights, w.n_expert_used, n_tokens);

    ggml_tensor * w_sum = ggml_sum_rows(ctx, out.weights);
    w_sum = ggml_clamp(ctx, w_sum, 6.103515625e-5f, INFINITY);
    out.weights = ggml_div(ctx, out.weights, w_sum);
    if (w.expert_weight_scale != 1.0f) {
        out.weights = ggml_scale(ctx, out.weights, w.expert_weight_scale);
    }
    return out;
}

static ggml_tensor * build_moe_ffn(
        ggml_context * ctx,
        ggml_tensor * cur,
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L,
        int layer_idx,
        int n_tokens) {

    const int n_embd = w.n_embd;
    const int n_used = w.n_expert_used;
    const int n_ff_exp = w.n_ff_exp;
    ggml_tensor * shared_out = build_shared_ffn(ctx, cur, w, L);
    ggml_tensor * routed_out = nullptr;

    if (layer_idx < w.n_hash_layer && L.ffn_gate_tid2eid) {
        routed_out = ggml_scale(ctx, cur, 0.0f);
    } else {
        Ds4MoeRouting routing = build_moe_routing(ctx, cur, w, L, n_tokens);
        ggml_tensor * cur_3d = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
        ggml_tensor * gate_e = ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur_3d, routing.selected);
        ggml_tensor * up_e = ggml_mul_mat_id(ctx, L.ffn_up_exps, cur_3d, routing.selected);

        gate_e = ggml_reshape_3d(ctx, gate_e, n_ff_exp, n_used, n_tokens);
        up_e = ggml_reshape_3d(ctx, up_e, n_ff_exp, n_used, n_tokens);
        ggml_tensor * mid_e = build_clamped_swiglu(ctx, gate_e, up_e, w.swiglu_clamp_exp);

        ggml_tensor * down_e = ggml_mul_mat_id(ctx, L.ffn_down_exps, mid_e, routing.selected);
        down_e = ggml_reshape_3d(ctx, down_e, n_embd, n_used, n_tokens);

        ggml_tensor * weights_3d = ggml_reshape_3d(ctx, routing.weights, 1, n_used, n_tokens);
        routed_out = ggml_mul(ctx, down_e, weights_3d);
        // Sum over dim-1 (n_used experts): permute [n_embd,n_used,n_tokens] -> [n_used,n_embd,n_tokens],
        // then sum_rows reduces dim-0, yielding [1,n_embd,n_tokens], reshape to [n_embd,n_tokens].
        routed_out = ggml_cont(ctx, ggml_permute(ctx, routed_out, 1, 0, 2, 3));
        routed_out = ggml_sum_rows(ctx, routed_out);
        routed_out = ggml_reshape_2d(ctx, routed_out, n_embd, n_tokens);
    }

    return ggml_add(ctx, shared_out, routed_out);
}

// ─── HC (Hierarchical Controller) Pre ───────────────────────────────────
// Mixes n_hc residual streams into a single working vector via Sinkhorn.

static ggml_tensor * build_hc_pre(
        ggml_context * ctx,
        ggml_tensor * hc_state,      // [n_hc * n_embd] persistent residual
        const DeepSeek4Weights & w,
        ggml_tensor * hc_fn,         // [n_hc * n_embd, hc_mix_dim]
        ggml_tensor * hc_scale,      // [3]
        ggml_tensor * hc_base,       // [n_hc]
        int n_tokens) {

    const int n_embd = w.n_embd;
    const int n_hc   = w.n_hc;
    (void)n_tokens;

    // RMSNorm over each HC stream independently
    ggml_tensor * flat = ggml_rms_norm(ctx, hc_state, w.hc_eps);

    // Mix projection: flat → [hc_mix_dim]
    // hc_mix_dim = 2*n_hc + n_hc*n_hc (pre weights + post gates + combine matrix)
    ggml_tensor * mix = ggml_mul_mat(ctx, hc_fn, flat);

    // Placeholder: return first HC stream as the working vector
    ggml_tensor * out = ggml_view_1d(ctx, hc_state, n_embd, 0);

    (void)mix; (void)hc_scale; (void)hc_base; (void)n_hc;
    return out;
}

// ─── CPU-side HC for hybrid path ────────────────────────────────────────
// HC involves Sinkhorn normalization (iterative, 4×4 matrix) which doesn't
// map well to ggml ops. For the hybrid path (per-layer graph execution),
// we implement HC entirely on CPU between layer graphs.

struct HcPreResult {
    std::vector<float> working;   // [n_embd] — input to sublayer
    float post[4];                // post gates
    float comb[16];               // combine matrix [4×4]
};

// Per-layer CPU-side HC weight cache (read from GPU once for CPU fallback and
// CUDA HC scalar parameters).
struct HcWeightsCpu {
    std::vector<uint16_t> fn_data;   // [hc_dim * mix_dim] F16
    std::vector<float> scale_data;   // [3]
    std::vector<float> base_data;    // [2*n_hc + n_hc*n_hc]
    bool loaded = false;
};

struct HcLayerWeightsCpu {
    HcWeightsCpu attn;
    HcWeightsCpu ffn;
};

struct HashRoutingTableCpu {
    std::vector<int32_t> ids;  // [n_vocab, n_expert_used]
    bool loaded = false;
};

static void cpu_rms_norm(float * out, const float * x, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    const float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale;
}

static float cpu_dot_f16_row(const uint16_t * row, const float * x, int cols) {
    float acc = 0.0f;
    for (int c = 0; c < cols; c++) {
        acc += ggml_fp16_to_fp32(row[c]) * x[c];
    }
    return acc;
}

static void cpu_matvec_f16_serial(float * out, const uint16_t * mat, const float * x, int rows, int cols) {
    // mat: [cols, rows] in row-major F16 (ggml layout: ne[0]=cols, ne[1]=rows)
    // out[r] = dot(mat_row_r, x) for r in [0, rows)
    for (int r = 0; r < rows; r++) {
        const uint16_t * row = mat + (size_t)r * cols;
        out[r] = cpu_dot_f16_row(row, x, cols);
    }
}

static void cpu_matvec_f16(float * out, const uint16_t * mat, const float * x, int rows, int cols) {
    const int64_t ops = (int64_t)rows * (int64_t)cols;
    const int min_parallel_rows = ops >= 262144 ? 1 : 512;
    ds4_parallel_for_tokens(rows, min_parallel_rows, [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            const uint16_t * row = mat + (size_t)r * cols;
            out[r] = cpu_dot_f16_row(row, x, cols);
        }
    });
}

static void cpu_hc_sinkhorn(float * out, const float * mix, const float * scale,
                             const float * base, int n_hc, int iters, float eps) {
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    // Pre weights: sigmoid(mix[i] * pre_scale + base[i]) + eps
    for (int i = 0; i < n_hc; i++) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }
    // Post gates: 2 * sigmoid(mix[n_hc+i] * post_scale + base[n_hc+i])
    for (int i = 0; i < n_hc; i++) {
        const float z = mix[n_hc + i] * post_scale + base[n_hc + i];
        out[n_hc + i] = 2.0f / (1.0f + expf(-z));
    }

    // Combine matrix: Sinkhorn normalization on [n_hc × n_hc]
    float c[16];
    for (int dst = 0; dst < n_hc; dst++) {
        float row_max = -1e30f;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            const float v = mix[2 * n_hc + idx] * comb_scale + base[2 * n_hc + idx];
            c[idx] = v;
            if (v > row_max) row_max = v;
        }
        float row_sum = 0.0f;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            c[idx] = expf(c[idx] - row_max);
            row_sum += c[idx];
        }
        const float inv = 1.0f / row_sum;
        for (int src = 0; src < n_hc; src++) {
            c[src + dst * n_hc] = c[src + dst * n_hc] * inv + eps;
        }
    }
    // Column normalization
    for (int src = 0; src < n_hc; src++) {
        float sum = 0.0f;
        for (int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];
        const float inv = 1.0f / (sum + eps);
        for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
    }
    // Additional Sinkhorn iterations
    for (int iter = 1; iter < iters; iter++) {
        for (int dst = 0; dst < n_hc; dst++) {
            float sum = 0.0f;
            for (int src = 0; src < n_hc; src++) sum += c[src + dst * n_hc];
            const float inv = 1.0f / (sum + eps);
            for (int src = 0; src < n_hc; src++) c[src + dst * n_hc] *= inv;
        }
        for (int src = 0; src < n_hc; src++) {
            float sum = 0.0f;
            for (int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];
            const float inv = 1.0f / (sum + eps);
            for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
        }
    }
    for (int i = 0; i < n_hc * n_hc; i++) out[2 * n_hc + i] = c[i];
}

static void finish_hc_pre_from_mix_into(float * working,
                                        float * post,
                                        float * comb,
                                        const float * hc_state,
                                        const float * mix,
                                        const float * scale_data,
                                        const float * base_data,
                                        int n_embd,
                                        int n_hc,
                                        int sinkhorn_iters) {
    // Sinkhorn split
    float split[24];  // 2*4 + 4*4 = 24
    cpu_hc_sinkhorn(split, mix, scale_data, base_data, n_hc, sinkhorn_iters, 1.0e-6f);

    // Weighted sum: out[d] = Σ_h split[h] * hc_state[h*n_embd + d]
    for (int d = 0; d < n_embd; d++) {
        float acc = 0.0f;
        for (int h = 0; h < n_hc; h++) {
            acc += split[h] * hc_state[(size_t)h * n_embd + d];
        }
        working[d] = acc;
    }

    memcpy(post, split + n_hc, (size_t)n_hc * sizeof(float));
    memcpy(comb, split + 2 * n_hc, (size_t)n_hc * n_hc * sizeof(float));
}

static HcPreResult finish_hc_pre_from_mix(const float * hc_state,
                                          const float * mix,
                                          const float * scale_data,
                                          const float * base_data,
                                          int n_embd,
                                          int n_hc,
                                          int sinkhorn_iters) {
    HcPreResult result;
    result.working.resize(n_embd);
    finish_hc_pre_from_mix_into(result.working.data(), result.post, result.comb,
                                hc_state, mix, scale_data, base_data,
                                n_embd, n_hc, sinkhorn_iters);
    return result;
}

static void cpu_hc_pre_into(float * working,
                            float * post,
                            float * comb,
                            const float * hc_state,
                            const uint16_t * fn_data,
                            const float * scale_data,
                            const float * base_data,
                            int n_embd,
                            int n_hc,
                            int sinkhorn_iters,
                            float hc_eps,
                            float * flat,
                            float * mix,
                            bool serial_fn) {
    const int hc_dim = n_hc * n_embd;
    const int mix_dim = 2 * n_hc + n_hc * n_hc;

    // RMSNorm over full HC state
    cpu_rms_norm(flat, hc_state, hc_dim, hc_eps);

    // Matmul: fn^T @ flat → mix[mix_dim]
    // fn is [hc_dim, mix_dim] F16 (ggml layout: ne[0]=hc_dim, ne[1]=mix_dim)
    if (serial_fn) {
        cpu_matvec_f16_serial(mix, fn_data, flat, mix_dim, hc_dim);
    } else {
        cpu_matvec_f16(mix, fn_data, flat, mix_dim, hc_dim);
    }
    finish_hc_pre_from_mix_into(working, post, comb, hc_state, mix,
                                scale_data, base_data,
                                n_embd, n_hc, sinkhorn_iters);
}

static HcPreResult cpu_hc_pre(const float * hc_state, const uint16_t * fn_data,
                               const float * scale_data, const float * base_data,
                               int n_embd, int n_hc, int sinkhorn_iters, float hc_eps) {
    HcPreResult result;
    result.working.resize(n_embd);
    std::vector<float> flat((size_t)n_hc * (size_t)n_embd);
    float mix[24];
    cpu_hc_pre_into(result.working.data(), result.post, result.comb,
                    hc_state, fn_data, scale_data, base_data,
                    n_embd, n_hc, sinkhorn_iters, hc_eps, flat.data(), mix, false);
    return result;
}

static bool ds4_hc_cuda_enabled() {
#if defined(DFLASH27B_BACKEND_CUDA)
    return true;
#else
    return false;
#endif
}

static HcPreResult hc_pre_auto(const float * hc_state,
                               const HcWeightsCpu & weights,
                               ggml_tensor * fn_tensor,
                               int n_embd,
                               int n_hc,
                               int sinkhorn_iters,
                               float hc_eps) {
#if defined(DFLASH27B_BACKEND_CUDA)
    if (ds4_hc_cuda_enabled() && fn_tensor && fn_tensor->data) {
        float mix[24];
        if (deepseek4_cuda_hc_pre_mix(hc_state, fn_tensor->data,
                                      n_embd, n_hc, hc_eps, mix)) {
            return finish_hc_pre_from_mix(hc_state, mix,
                                          weights.scale_data.data(),
                                          weights.base_data.data(),
                                          n_embd, n_hc, sinkhorn_iters);
        }
    }
#else
    (void)fn_tensor;
#endif
    return cpu_hc_pre(hc_state, weights.fn_data.data(),
                      weights.scale_data.data(), weights.base_data.data(),
                      n_embd, n_hc, sinkhorn_iters, hc_eps);
}

static void hc_pre_auto_into(float * working,
                             float * post,
                             float * comb,
                             const float * hc_state,
                             const HcWeightsCpu & weights,
                             ggml_tensor * fn_tensor,
                             int n_embd,
                             int n_hc,
                             int sinkhorn_iters,
                             float hc_eps,
                             float * flat,
                             float * mix_scratch) {
#if defined(DFLASH27B_BACKEND_CUDA)
    if (ds4_hc_cuda_enabled() && fn_tensor && fn_tensor->data) {
        float mix[24];
        if (deepseek4_cuda_hc_pre_mix(hc_state, fn_tensor->data,
                                      n_embd, n_hc, hc_eps, mix)) {
            finish_hc_pre_from_mix_into(working, post, comb, hc_state, mix,
                                        weights.scale_data.data(),
                                        weights.base_data.data(),
                                        n_embd, n_hc, sinkhorn_iters);
            return;
        }
    }
#else
    (void)fn_tensor;
#endif
    cpu_hc_pre_into(working, post, comb,
                    hc_state, weights.fn_data.data(),
                    weights.scale_data.data(), weights.base_data.data(),
                    n_embd, n_hc, sinkhorn_iters, hc_eps, flat, mix_scratch, true);
}

static void hc_pre_batch(std::vector<float> & working,
                         std::vector<float> & post,
                         std::vector<float> & comb,
                         const float * hc_state,
                         const HcWeightsCpu & weights,
                         ggml_tensor * fn_tensor,
                         int n_tokens,
                         int n_embd,
                         int n_hc,
                         int sinkhorn_iters,
                         float hc_eps) {
    const size_t hc_dim = (size_t)n_embd * (size_t)n_hc;
    working.resize((size_t)n_tokens * (size_t)n_embd);
    post.resize((size_t)n_tokens * (size_t)n_hc);
    comb.resize((size_t)n_tokens * (size_t)n_hc * (size_t)n_hc);

    ds4_parallel_for_tokens(n_tokens, 8, [&](int t0, int t1) {
        std::vector<float> flat(hc_dim);
        float mix[24];
        for (int t = t0; t < t1; ++t) {
            hc_pre_auto_into(working.data() + (size_t)t * n_embd,
                             post.data() + (size_t)t * n_hc,
                             comb.data() + (size_t)t * n_hc * (size_t)n_hc,
                             hc_state + (size_t)t * hc_dim,
                             weights,
                             fn_tensor,
                             n_embd,
                             n_hc,
                             sinkhorn_iters,
                             hc_eps,
                             flat.data(),
                             mix);
        }
    });
}

static void cpu_hc_post(float * out_hc, const float * block_out,
                         const float * residual_hc, const float * post,
                         const float * comb, int n_embd, int n_hc) {
    for (int dst = 0; dst < n_hc; dst++) {
        for (int d = 0; d < n_embd; d++) {
            float acc = block_out[d] * post[dst];
            for (int src = 0; src < n_hc; src++) {
                acc += comb[dst + src * n_hc] * residual_hc[(size_t)src * n_embd + d];
            }
            out_hc[(size_t)dst * n_embd + d] = acc;
        }
    }
}

static void hc_post_batch(std::vector<float> & out_hc,
                          const float * block_out,
                          const float * residual_hc,
                          const float * post,
                          const float * comb,
                          int n_tokens,
                          int n_embd,
                          int n_hc) {
    const size_t hc_dim = (size_t)n_embd * (size_t)n_hc;
    out_hc.resize((size_t)n_tokens * hc_dim);
    ds4_parallel_for_tokens(n_tokens, 8, [&](int t0, int t1) {
        for (int t = t0; t < t1; ++t) {
            cpu_hc_post(out_hc.data() + (size_t)t * hc_dim,
                        block_out + (size_t)t * n_embd,
                        residual_hc + (size_t)t * hc_dim,
                        post + (size_t)t * n_hc,
                        comb + (size_t)t * n_hc * (size_t)n_hc,
                        n_embd,
                        n_hc);
        }
    });
}

static void hc_output_batch(std::vector<float> & final_embd,
                            const float * hc_state,
                            const HcWeightsCpu & weights,
                            int n_tokens,
                            int n_embd,
                            int n_hc,
                            float hc_eps) {
    const size_t hc_dim = (size_t)n_embd * (size_t)n_hc;
    final_embd.resize((size_t)n_tokens * (size_t)n_embd);
    ds4_parallel_for_tokens(n_tokens, 8, [&](int t0, int t1) {
        std::vector<float> flat(hc_dim);
        std::vector<float> pre((size_t)n_hc);
        std::vector<float> hc_weights((size_t)n_hc);
        for (int t = t0; t < t1; ++t) {
            const float * token_hc = hc_state + (size_t)t * hc_dim;
            float * out = final_embd.data() + (size_t)t * n_embd;
            cpu_rms_norm(flat.data(), token_hc, (int)hc_dim, hc_eps);
            cpu_matvec_f16_serial(pre.data(), weights.fn_data.data(), flat.data(), n_hc, (int)hc_dim);
            for (int i = 0; i < n_hc; ++i) {
                const float z = pre[(size_t)i] * weights.scale_data[0] +
                                weights.base_data[(size_t)i];
                hc_weights[(size_t)i] = 1.0f / (1.0f + expf(-z)) + 1.0e-6f;
            }
            for (int d = 0; d < n_embd; ++d) {
                float acc = 0.0f;
                for (int h = 0; h < n_hc; ++h) {
                    acc += hc_weights[(size_t)h] * token_hc[(size_t)h * n_embd + d];
                }
                out[(size_t)d] = acc;
            }
        }
    });
}

static void load_hc_weights_cpu(HcWeightsCpu & dst, ggml_tensor * fn,
                                 ggml_tensor * scale, ggml_tensor * base) {
    if (!fn || !scale || !base || dst.loaded) return;
    dst.fn_data.resize(ggml_nelements(fn));
    dst.scale_data.resize(ggml_nelements(scale));
    dst.base_data.resize(ggml_nelements(base));
    ggml_backend_tensor_get(fn, dst.fn_data.data(), 0, ggml_nbytes(fn));
    ggml_backend_tensor_get(scale, dst.scale_data.data(), 0, ggml_nbytes(scale));
    ggml_backend_tensor_get(base, dst.base_data.data(), 0, ggml_nbytes(base));
    dst.loaded = true;
}

static bool load_hash_routing_cpu(HashRoutingTableCpu & dst, ggml_tensor * table) {
    if (dst.loaded) return true;
    if (!table) return false;
    dst.ids.resize(ggml_nelements(table));
    ggml_backend_tensor_get(table, dst.ids.data(), 0, ggml_nbytes(table));
    dst.loaded = true;
    return true;
}

static bool deepseek4_step_hybrid(
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        DeepSeek4Cache & cache,
        MoeHybridStorage & moe_hybrid,
        const float * embed,
        int n_tokens,
        int kv_start,
        std::vector<float> & out_logits,
        const int32_t * token_ids,
        MoeHybridStreamEngine * stream_engine,
        DeepSeek4StepTelemetry * telemetry,
        MoeHybridRoutingStats * routing_stats) {
    const auto step_t0 = Ds4TimingClock::now();
    const int n_embd = w.n_embd;
    const int n_hc = w.n_hc;
    const int hc_dim = n_hc * n_embd;
    ggml_backend_t cpu_backend = moe_hybrid.cpu_backend;
    ggml_gallocr_t hot_alloc = nullptr;
    ggml_gallocr_t cold_alloc = nullptr;

    // HC state: 4 streams, each n_embd. Initialize to copies of embedding.
    // For n_tokens=1 (decode), embed is [n_embd].
    std::vector<float> hc_state((size_t)hc_dim * (size_t)n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        for (int h = 0; h < n_hc; h++) {
            memcpy(hc_state.data() + (size_t)t * hc_dim + (size_t)h * n_embd,
                   embed + (size_t)t * n_embd, (size_t)n_embd * sizeof(float));
        }
    }

    // Lazy-loaded per-layer HC weights on CPU
    static std::vector<HcLayerWeightsCpu> hc_layer_weights;
    static HcWeightsCpu hc_output_weights;
    static std::vector<HashRoutingTableCpu> hash_routing_tables;
    if (hc_layer_weights.empty()) {
        hc_layer_weights.resize((size_t)w.n_layer);
        hash_routing_tables.resize((size_t)w.n_layer);
        for (int il = 0; il < w.n_layer; il++) {
            const DeepSeek4Layer & L = w.layers[(size_t)il];
            load_hc_weights_cpu(hc_layer_weights[il].attn, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base);
            load_hc_weights_cpu(hc_layer_weights[il].ffn, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base);
            if (il < w.n_hash_layer && L.ffn_gate_tid2eid) {
                load_hash_routing_cpu(hash_routing_tables[(size_t)il], L.ffn_gate_tid2eid);
            }
        }
        load_hc_weights_cpu(hc_output_weights, w.output_hc_fn, w.output_hc_scale, w.output_hc_base);
    }

    for (int il = 0; il < w.n_layer; ++il) {
        const DeepSeek4Layer & L = w.layers[(size_t) il];
        DeepSeek4LayerCache & lc = cache.layers[(size_t) il];
        const HcLayerWeightsCpu & hc_lw = hc_layer_weights[(size_t)il];

        // ── HC pre (attention) ──────────────────────────────────────
        // For decode (n_tokens=1): compute working vector from HC state
        const auto hc_pre_attn_t0 = Ds4TimingClock::now();
        std::vector<float> cur((size_t)n_embd * (size_t)n_tokens);
        HcPreResult hc_attn_result;
        if (hc_lw.attn.loaded && n_tokens == 1) {
            hc_attn_result = hc_pre_auto(hc_state.data(), hc_lw.attn, L.hc_attn_fn,
                                         n_embd, n_hc, w.n_hc_sinkhorn_iter, w.hc_eps);
            memcpy(cur.data(), hc_attn_result.working.data(), (size_t)n_embd * sizeof(float));
        } else {
            // Fallback: use first HC stream
            memcpy(cur.data(), hc_state.data(), (size_t)n_embd * (size_t)n_tokens * sizeof(float));
        }
        if (telemetry) telemetry->hc_pre_attn_us += ds4_elapsed_us(hc_pre_attn_t0, Ds4TimingClock::now());

        // ── Build attention graph ───────────────────────────────────
        const auto attn_build_t0 = Ds4TimingClock::now();
        const size_t ctx_size = 48 * 1024 * 1024;
        ggml_init_params params{};
        params.mem_size = ctx_size;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            if (hot_alloc) ggml_gallocr_free(hot_alloc);
            if (cold_alloc) ggml_gallocr_free(cold_alloc);
            return false;
        }

        ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp);
        std::vector<DeepSeek4I32InputBinding> i32_inputs;
        std::vector<DeepSeek4I32ArrayBinding> i32_array_inputs;
        std::vector<DeepSeek4I64ArrayBinding> i64_array_inputs;
        ggml_cgraph * gf = ggml_new_graph(ctx);

        ggml_tensor * normed = build_rms_norm(ctx, inp, L.attn_norm, w.rms_eps);
        ggml_tensor * attn_out = build_mla_attention(ctx, gf, normed, w, L, lc, il,
                                                     kv_start, n_tokens, nullptr,
                                                     i32_inputs, i32_array_inputs,
                                                     i64_array_inputs);
        // Output just attn_out (HC post handles the residual mixing)
        ggml_build_forward_expand(gf, attn_out);
        ggml_gallocr_t attn_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(attn_alloc, gf)) {
            ggml_gallocr_free(attn_alloc);
            ggml_free(ctx);
            if (hot_alloc) ggml_gallocr_free(hot_alloc);
            if (cold_alloc) ggml_gallocr_free(cold_alloc);
            return false;
        }
        if (telemetry) telemetry->attn_build_us += ds4_elapsed_us(attn_build_t0, Ds4TimingClock::now());
        ggml_backend_tensor_set(inp, cur.data(), 0, sizeof(float) * cur.size());
        for (const DeepSeek4I32InputBinding & binding : i32_inputs) {
            ggml_backend_tensor_set(binding.tensor, &binding.value, 0, sizeof(binding.value));
        }
        for (const DeepSeek4I32ArrayBinding & binding : i32_array_inputs) {
            ggml_backend_tensor_set(binding.tensor, binding.values.data(), 0,
                                    sizeof(int32_t) * binding.values.size());
        }
        for (const DeepSeek4I64ArrayBinding & binding : i64_array_inputs) {
            ggml_backend_tensor_set(binding.tensor, binding.values.data(), 0,
                                    sizeof(int64_t) * binding.values.size());
        }
        const auto attn_compute_t0 = Ds4TimingClock::now();
        bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        if (telemetry) telemetry->attn_compute_us += ds4_elapsed_us(attn_compute_t0, Ds4TimingClock::now());
        std::vector<float> attn_out_host((size_t)n_embd * (size_t)n_tokens);
        if (ok) {
            const auto attn_read_t0 = Ds4TimingClock::now();
            ggml_backend_tensor_get(attn_out, attn_out_host.data(), 0, sizeof(float) * attn_out_host.size());
            if (telemetry) telemetry->attn_read_us += ds4_elapsed_us(attn_read_t0, Ds4TimingClock::now());
        }
        ggml_gallocr_free(attn_alloc);
        ggml_free(ctx);
        if (!ok) {
            if (hot_alloc) ggml_gallocr_free(hot_alloc);
            if (cold_alloc) ggml_gallocr_free(cold_alloc);
            return false;
        }

        // ── HC post (attention) ─────────────────────────────────────
        const auto hc_post_attn_t0 = Ds4TimingClock::now();
        if (hc_lw.attn.loaded && n_tokens == 1) {
            std::vector<float> new_hc((size_t)hc_dim);
            cpu_hc_post(new_hc.data(), attn_out_host.data(), hc_state.data(),
                        hc_attn_result.post, hc_attn_result.comb, n_embd, n_hc);
            memcpy(hc_state.data(), new_hc.data(), (size_t)hc_dim * sizeof(float));
        } else {
            for (int i = 0; i < n_embd * n_tokens; i++) {
                hc_state[(size_t)i] += attn_out_host[(size_t)i];
            }
        }
        if (telemetry) telemetry->hc_post_attn_us += ds4_elapsed_us(hc_post_attn_t0, Ds4TimingClock::now());

        // ── HC pre (FFN) ────────────────────────────────────────────
        const auto hc_pre_ffn_t0 = Ds4TimingClock::now();
        std::vector<float> ffn_working((size_t)n_embd * (size_t)n_tokens);
        HcPreResult hc_ffn_result;
        if (hc_lw.ffn.loaded && n_tokens == 1) {
            hc_ffn_result = hc_pre_auto(hc_state.data(), hc_lw.ffn, L.hc_ffn_fn,
                                        n_embd, n_hc, w.n_hc_sinkhorn_iter, w.hc_eps);
            memcpy(ffn_working.data(), hc_ffn_result.working.data(), (size_t)n_embd * sizeof(float));
        } else {
            memcpy(ffn_working.data(), hc_state.data(), (size_t)n_embd * (size_t)n_tokens * sizeof(float));
        }
        if (telemetry) telemetry->hc_pre_ffn_us += ds4_elapsed_us(hc_pre_ffn_t0, Ds4TimingClock::now());

        // ── FFN ─────────────────────────────────────────────────────
        std::vector<float> ffn_out_host((size_t)n_embd * (size_t)n_tokens, 0.0f);

        if (il < w.n_hash_layer && L.ffn_gate_tid2eid) {
            // Hash-routed layers: selected experts come from token_id -> expert_ids,
            // while weights still come from router probabilities for those experts.
            if (!token_ids || !hash_routing_tables[(size_t)il].loaded) {
                std::fprintf(stderr, "[deepseek4] missing token ids/hash table for layer %d\n", il);
                if (hot_alloc) ggml_gallocr_free(hot_alloc);
                if (cold_alloc) ggml_gallocr_free(cold_alloc);
                return false;
            }
            ggml_init_params ffn_params{};
            const auto route_build_t0 = Ds4TimingClock::now();
            ffn_params.mem_size = 16 * 1024 * 1024;
            ffn_params.mem_buffer = nullptr;
            ffn_params.no_alloc = true;
            ggml_context * ffn_ctx = ggml_init(ffn_params);
            if (!ffn_ctx) {
                if (hot_alloc) ggml_gallocr_free(hot_alloc);
                if (cold_alloc) ggml_gallocr_free(cold_alloc);
                return false;
            }
            ggml_tensor * ffn_inp = ggml_new_tensor_2d(ffn_ctx, GGML_TYPE_F32, n_embd, n_tokens);
            ggml_set_input(ffn_inp);
            ggml_tensor * ffn_normed = build_rms_norm(ffn_ctx, ffn_inp, L.ffn_norm, w.rms_eps);
            ggml_tensor * router_logits = ggml_mul_mat(ffn_ctx, L.ffn_gate_inp, ffn_normed);
            ggml_tensor * router_probs = ggml_sqrt(ffn_ctx, ggml_softplus(ffn_ctx, router_logits));
            ggml_cgraph * ffn_gf = ggml_new_graph(ffn_ctx);
            ggml_build_forward_expand(ffn_gf, ffn_normed);
            ggml_build_forward_expand(ffn_gf, router_probs);
            ggml_gallocr_t ffn_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(ffn_alloc, ffn_gf)) {
                ggml_gallocr_free(ffn_alloc); ggml_free(ffn_ctx);
                if (hot_alloc) ggml_gallocr_free(hot_alloc);
                if (cold_alloc) ggml_gallocr_free(cold_alloc);
                return false;
            }
            if (telemetry) telemetry->route_build_us += ds4_elapsed_us(route_build_t0, Ds4TimingClock::now());
            ggml_backend_tensor_set(ffn_inp, ffn_working.data(), 0, sizeof(float) * ffn_working.size());
            const auto route_compute_t0 = Ds4TimingClock::now();
            ok = ggml_backend_graph_compute(backend, ffn_gf) == GGML_STATUS_SUCCESS;
            if (telemetry) telemetry->route_compute_us += ds4_elapsed_us(route_compute_t0, Ds4TimingClock::now());
            std::vector<float> ffn_normed_host((size_t)n_embd * (size_t)n_tokens);
            std::vector<float> probs_host((size_t)w.n_expert * (size_t)n_tokens);
            if (ok) {
                const auto route_read_t0 = Ds4TimingClock::now();
                ggml_backend_tensor_get(ffn_normed, ffn_normed_host.data(), 0, sizeof(float) * ffn_normed_host.size());
                ggml_backend_tensor_get(router_probs, probs_host.data(), 0, sizeof(float) * probs_host.size());
                if (telemetry) telemetry->route_read_us += ds4_elapsed_us(route_read_t0, Ds4TimingClock::now());
            }
            ggml_gallocr_free(ffn_alloc);
            ggml_free(ffn_ctx);
            if (!ok) {
                if (hot_alloc) ggml_gallocr_free(hot_alloc);
                if (cold_alloc) ggml_gallocr_free(cold_alloc);
                return false;
            }

            std::vector<int32_t> selected_host((size_t)w.n_expert_used * (size_t)n_tokens);
            std::vector<float> weights_host((size_t)w.n_expert_used * (size_t)n_tokens);
            const auto route_select_t0 = Ds4TimingClock::now();
            const auto & hash_table = hash_routing_tables[(size_t)il].ids;
            for (int ti = 0; ti < n_tokens; ++ti) {
                const int32_t tok = token_ids[ti];
                if (tok < 0 || tok >= w.n_vocab) {
                    std::fprintf(stderr, "[deepseek4] token id %d outside hash table for layer %d\n", tok, il);
                    if (hot_alloc) ggml_gallocr_free(hot_alloc);
                    if (cold_alloc) ggml_gallocr_free(cold_alloc);
                    return false;
                }
                const int32_t * row = hash_table.data() + (size_t)tok * (size_t)w.n_expert_used;
                float sum = 0.0f;
                for (int ei = 0; ei < w.n_expert_used; ++ei) {
                    const int32_t expert = row[ei];
                    selected_host[(size_t)ti * (size_t)w.n_expert_used + (size_t)ei] = expert;
                    float prob = 0.0f;
                    if (expert >= 0 && expert < w.n_expert) {
                        prob = probs_host[(size_t)ti * (size_t)w.n_expert + (size_t)expert];
                    }
                    weights_host[(size_t)ti * (size_t)w.n_expert_used + (size_t)ei] = prob;
                    sum += prob;
                }
                sum = std::max(sum, 6.103515625e-5f);
                for (int ei = 0; ei < w.n_expert_used; ++ei) {
                    float & weight = weights_host[(size_t)ti * (size_t)w.n_expert_used + (size_t)ei];
                    weight = weight / sum * w.expert_weight_scale;
                }
            }
            if (telemetry) telemetry->route_select_us += ds4_elapsed_us(route_select_t0, Ds4TimingClock::now());
            if (routing_stats) {
                for (int ti = 0; ti < n_tokens; ++ti) {
                    routing_stats->observe(il,
                        selected_host.data() + (size_t)ti * (size_t)w.n_expert_used,
                        w.n_expert_used);
                }
            }

            MoeHybridConfig hybrid_cfg = make_ds4_moe_hybrid_config(w);
            MoeLayerDesc desc = make_ds4_moe_layer_desc(L);
            auto & storage = moe_hybrid.layers[(size_t) il];
            if (!eval_ds4_hybrid(
                    backend, cpu_backend, hybrid_cfg, desc, &moe_hybrid, storage, stream_engine,
                    il, n_embd, w.n_expert_used,
                    ffn_normed_host.data(), selected_host.data(), weights_host.data(),
                    n_tokens, ffn_out_host, &hot_alloc, &cold_alloc, telemetry)) {
                if (hot_alloc) ggml_gallocr_free(hot_alloc);
                if (cold_alloc) ggml_gallocr_free(cold_alloc);
                return false;
            }
        } else {
            // MoE layers: compute routing on GPU, experts via hybrid
            const auto route_build_t0 = Ds4TimingClock::now();
            ggml_init_params ffn_params{};
            ffn_params.mem_size = 16 * 1024 * 1024;
            ffn_params.mem_buffer = nullptr;
            ffn_params.no_alloc = true;
            ggml_context * ffn_ctx = ggml_init(ffn_params);
            if (!ffn_ctx) {
                if (hot_alloc) ggml_gallocr_free(hot_alloc);
                if (cold_alloc) ggml_gallocr_free(cold_alloc);
                return false;
            }
            ggml_tensor * ffn_inp = ggml_new_tensor_2d(ffn_ctx, GGML_TYPE_F32, n_embd, n_tokens);
            ggml_set_input(ffn_inp);
            ggml_tensor * ffn_normed = build_rms_norm(ffn_ctx, ffn_inp, L.ffn_norm, w.rms_eps);
            ggml_tensor * router_logits = ggml_mul_mat(ffn_ctx, L.ffn_gate_inp, ffn_normed);
            ggml_tensor * router_probs = ggml_sqrt(ffn_ctx, ggml_softplus(ffn_ctx, router_logits));
            ggml_cgraph * ffn_gf = ggml_new_graph(ffn_ctx);
            ggml_build_forward_expand(ffn_gf, ffn_normed);
            ggml_build_forward_expand(ffn_gf, router_probs);
            ggml_gallocr_t ffn_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(ffn_alloc, ffn_gf)) {
                ggml_gallocr_free(ffn_alloc); ggml_free(ffn_ctx);
                if (hot_alloc) ggml_gallocr_free(hot_alloc);
                if (cold_alloc) ggml_gallocr_free(cold_alloc);
                return false;
            }
            if (telemetry) telemetry->route_build_us += ds4_elapsed_us(route_build_t0, Ds4TimingClock::now());
            ggml_backend_tensor_set(ffn_inp, ffn_working.data(), 0, sizeof(float) * ffn_working.size());
            const auto route_compute_t0 = Ds4TimingClock::now();
            ok = ggml_backend_graph_compute(backend, ffn_gf) == GGML_STATUS_SUCCESS;
            if (telemetry) telemetry->route_compute_us += ds4_elapsed_us(route_compute_t0, Ds4TimingClock::now());
            if (!ok) {
                ggml_gallocr_free(ffn_alloc); ggml_free(ffn_ctx);
                if (hot_alloc) ggml_gallocr_free(hot_alloc);
                if (cold_alloc) ggml_gallocr_free(cold_alloc);
                return false;
            }

            std::vector<float> ffn_normed_host((size_t)n_embd * (size_t)n_tokens);
            std::vector<float> probs_host((size_t)w.n_expert * (size_t)n_tokens);
            std::vector<int32_t> selected_host((size_t)w.n_expert_used * (size_t)n_tokens);
            std::vector<float> weights_host((size_t)w.n_expert_used * (size_t)n_tokens);
            const auto route_read_t0 = Ds4TimingClock::now();
            ggml_backend_tensor_get(ffn_normed, ffn_normed_host.data(), 0, sizeof(float) * ffn_normed_host.size());
            ggml_backend_tensor_get(router_probs, probs_host.data(), 0, sizeof(float) * probs_host.size());
            if (telemetry) telemetry->route_read_us += ds4_elapsed_us(route_read_t0, Ds4TimingClock::now());
            ggml_gallocr_free(ffn_alloc);
            ggml_free(ffn_ctx);

            std::vector<float> bias_host;
            const auto route_select_t0 = Ds4TimingClock::now();
            if (L.ffn_exp_probs_b) {
                bias_host.resize((size_t)w.n_expert);
                ggml_backend_tensor_get(L.ffn_exp_probs_b, bias_host.data(), 0,
                                        sizeof(float) * bias_host.size());
            }
            for (int ti = 0; ti < n_tokens; ++ti) {
                const float * probs = probs_host.data() + (size_t)ti * (size_t)w.n_expert;
                std::vector<int32_t> top((size_t)w.n_expert_used, -1);
                for (int expert = 0; expert < w.n_expert; ++expert) {
                    const float score = probs[expert] +
                        (!bias_host.empty() ? bias_host[(size_t)expert] : 0.0f);
                    for (int slot = 0; slot < w.n_expert_used; ++slot) {
                        const int32_t cur_expert = top[(size_t)slot];
                        const float cur_score = cur_expert >= 0
                            ? probs[cur_expert] +
                                (!bias_host.empty() ? bias_host[(size_t)cur_expert] : 0.0f)
                            : -INFINITY;
                        if (cur_expert < 0 || score > cur_score) {
                            for (int m = w.n_expert_used - 1; m > slot; --m) {
                                top[(size_t)m] = top[(size_t)m - 1];
                            }
                            top[(size_t)slot] = expert;
                            break;
                        }
                    }
                }
                float sum = 0.0f;
                for (int slot = 0; slot < w.n_expert_used; ++slot) {
                    const int32_t expert = top[(size_t)slot];
                    selected_host[(size_t)ti * (size_t)w.n_expert_used + (size_t)slot] = expert;
                    const float weight = expert >= 0 ? probs[expert] : 0.0f;
                    weights_host[(size_t)ti * (size_t)w.n_expert_used + (size_t)slot] = weight;
                    sum += weight;
                }
                sum = std::max(sum, 6.103515625e-5f);
                for (int slot = 0; slot < w.n_expert_used; ++slot) {
                    float & weight = weights_host[(size_t)ti * (size_t)w.n_expert_used + (size_t)slot];
                    weight = weight / sum * w.expert_weight_scale;
                }
            }
            if (telemetry) telemetry->route_select_us += ds4_elapsed_us(route_select_t0, Ds4TimingClock::now());
            if (routing_stats) {
                for (int ti = 0; ti < n_tokens; ++ti) {
                    routing_stats->observe(il,
                        selected_host.data() + (size_t)ti * (size_t)w.n_expert_used,
                        w.n_expert_used);
                }
            }

            MoeHybridConfig hybrid_cfg = make_ds4_moe_hybrid_config(w);
            MoeLayerDesc desc = make_ds4_moe_layer_desc(L);
            auto & storage = moe_hybrid.layers[(size_t) il];
            if (!eval_ds4_hybrid(
                    backend, cpu_backend, hybrid_cfg, desc, &moe_hybrid, storage, stream_engine,
                    il, n_embd, w.n_expert_used,
                    ffn_normed_host.data(), selected_host.data(), weights_host.data(),
                    n_tokens, ffn_out_host, &hot_alloc, &cold_alloc, telemetry)) {
                if (hot_alloc) ggml_gallocr_free(hot_alloc);
                if (cold_alloc) ggml_gallocr_free(cold_alloc);
                return false;
            }
        }

        // ── HC post (FFN) ───────────────────────────────────────────
        const auto hc_post_ffn_t0 = Ds4TimingClock::now();
        if (hc_lw.ffn.loaded && n_tokens == 1) {
            std::vector<float> new_hc((size_t)hc_dim);
            cpu_hc_post(new_hc.data(), ffn_out_host.data(), hc_state.data(),
                        hc_ffn_result.post, hc_ffn_result.comb, n_embd, n_hc);
            memcpy(hc_state.data(), new_hc.data(), (size_t)hc_dim * sizeof(float));
        } else {
            for (int i = 0; i < n_embd * n_tokens; i++) {
                hc_state[(size_t)i] += ffn_out_host[(size_t)i];
            }
        }
        if (telemetry) telemetry->hc_post_ffn_us += ds4_elapsed_us(hc_post_ffn_t0, Ds4TimingClock::now());
    }

    if (hot_alloc) ggml_gallocr_free(hot_alloc);
    if (cold_alloc) ggml_gallocr_free(cold_alloc);

    // ── Output HC pre → norm → logits ───────────────────────────────────
    const auto output_t0 = Ds4TimingClock::now();
    std::vector<float> final_embd((size_t)n_embd * (size_t)n_tokens);
    if (hc_output_weights.loaded && n_tokens == 1) {
        std::vector<float> flat((size_t)hc_dim);
        cpu_rms_norm(flat.data(), hc_state.data(), hc_dim, w.hc_eps);
        std::vector<float> pre(n_hc);
        cpu_matvec_f16(pre.data(), hc_output_weights.fn_data.data(), flat.data(), n_hc, hc_dim);
        float hc_weights[4];
        for (int i = 0; i < n_hc; i++) {
            const float z = pre[i] * hc_output_weights.scale_data[0] + hc_output_weights.base_data[i];
            hc_weights[i] = 1.0f / (1.0f + expf(-z)) + 1.0e-6f;
        }
        for (int d = 0; d < n_embd; d++) {
            float acc = 0.0f;
            for (int h = 0; h < n_hc; h++) {
                acc += hc_weights[h] * hc_state[(size_t)h * n_embd + d];
            }
            final_embd[d] = acc;
        }
    } else {
        memcpy(final_embd.data(), hc_state.data(), (size_t)n_embd * (size_t)n_tokens * sizeof(float));
    }

    const size_t final_ctx_size = 16 * 1024 * 1024;
    ggml_init_params params2{};
    params2.mem_size = final_ctx_size;
    params2.mem_buffer = nullptr;
    params2.no_alloc = true;
    ggml_context * ctx2 = ggml_init(params2);
    if (!ctx2) return false;

    ggml_tensor * final_inp = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(final_inp);
    ggml_tensor * normed_out = build_rms_norm(ctx2, final_inp, w.out_norm, w.rms_eps);
    ggml_tensor * logits = ggml_mul_mat(ctx2, w.output, normed_out);
    ggml_cgraph * final_gf = ggml_new_graph(ctx2);
    ggml_build_forward_expand(final_gf, logits);
    ggml_gallocr_t final_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(final_alloc, final_gf)) {
        ggml_gallocr_free(final_alloc);
        ggml_free(ctx2);
        return false;
    }
    ggml_backend_tensor_set(final_inp, final_embd.data(), 0, sizeof(float) * final_embd.size());
    bool final_ok = ggml_backend_graph_compute(backend, final_gf) == GGML_STATUS_SUCCESS;
    if (final_ok) {
        out_logits.resize((size_t)w.n_vocab);
        const size_t logits_offset = (size_t)(n_tokens - 1) * (size_t)w.n_vocab * sizeof(float);
        ggml_backend_tensor_get(logits, out_logits.data(), logits_offset,
                                sizeof(float) * (size_t)w.n_vocab);
    }
    ggml_gallocr_free(final_alloc);
    ggml_free(ctx2);
    if (!final_ok) return false;
    if (telemetry) {
        telemetry->output_us += ds4_elapsed_us(output_t0, Ds4TimingClock::now());
        telemetry->total_us += ds4_elapsed_us(step_t0, Ds4TimingClock::now());
    }

    cache.cur_pos = kv_start + n_tokens;
    return true;
}

// ─── Full forward step ──────────────────────────────────────────────────

bool deepseek4_step(
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        DeepSeek4Cache & cache,
        const float * embed,
        int n_tokens,
        int kv_start,
        std::vector<float> & out_logits,
        MoeHybridStorage * moe_hybrid,
        const int32_t * token_ids,
        MoeHybridStreamEngine * stream_engine,
        DeepSeek4StepTelemetry * telemetry,
        MoeHybridRoutingStats * routing_stats) {

    if (w.moe_hybrid && moe_hybrid != nullptr) {
        return deepseek4_step_hybrid(backend, w, cache, *moe_hybrid,
                                     embed, n_tokens, kv_start, out_logits,
                                     token_ids, stream_engine, telemetry, routing_stats);
    }

    const int n_embd = w.n_embd;
    const int n_layer = w.n_layer;

    // Create compute graph context — need large budget for MoE layers
    const size_t ctx_size = ggml_tensor_overhead() * 65536 + 16 * 1024 * 1024;
    ggml_init_params params{};
    params.mem_size = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    // Input embeddings
    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(inp, "inp_embed");
    ggml_set_input(inp);

    ggml_tensor * cur = inp;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);
    std::vector<DeepSeek4I32InputBinding> i32_inputs;
    std::vector<DeepSeek4I32ArrayBinding> i32_array_inputs;
    std::vector<DeepSeek4I64ArrayBinding> i64_array_inputs;

    // Layer loop
    for (int il = 0; il < n_layer; il++) {
        const DeepSeek4Layer & L = w.layers[il];
        DeepSeek4LayerCache & lc = cache.layers[il];

        // ── HC pre (attention) ──────────────────────────────────────
        // TODO: Full HC implementation. For now, pass cur through directly.
        ggml_tensor * attn_in = cur;

        // ── Attention norm ──────────────────────────────────────────
        ggml_tensor * normed = build_rms_norm(ctx, attn_in, L.attn_norm, w.rms_eps);

        // ── MLA attention ───────────────────────────────────────────
        ggml_tensor * attn_out = build_mla_attention(ctx, gf, normed, w, L, lc,
                                                      il, kv_start, n_tokens,
                                                      nullptr, i32_inputs, i32_array_inputs,
                                                      i64_array_inputs);

        // ── Residual ────────────────────────────────────────────────
        cur = ggml_add(ctx, cur, attn_out);

        // ── HC pre (FFN) ────────────────────────────────────────────
        ggml_tensor * ffn_in = cur;

        // ── FFN norm ────────────────────────────────────────────────
        ggml_tensor * ffn_normed = build_rms_norm(ctx, ffn_in, L.ffn_norm, w.rms_eps);

        // ── MoE FFN ─────────────────────────────────────────────────
        ggml_tensor * ffn_out = build_moe_ffn(ctx, ffn_normed, w, L, il, n_tokens);

        // ── Residual ────────────────────────────────────────────────
        cur = ggml_add(ctx, cur, ffn_out);
    }

    // ── Output head ─────────────────────────────────────────────────────
    // TODO: HC output pre (merge residual streams for final projection)

    // Final RMSNorm
    cur = build_rms_norm(ctx, cur, w.out_norm, w.rms_eps);

    // lm_head projection
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    // ── Build and run graph ─────────────────────────────────────────────
    ggml_build_forward_expand(gf, logits);

    // Allocate
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "[deepseek4] graph allocation failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    // Set input data
    ggml_backend_tensor_set(inp, embed, 0, n_embd * n_tokens * sizeof(float));
    for (const DeepSeek4I32InputBinding & binding : i32_inputs) {
        ggml_backend_tensor_set(binding.tensor, &binding.value, 0, sizeof(binding.value));
    }
    for (const DeepSeek4I32ArrayBinding & binding : i32_array_inputs) {
        ggml_backend_tensor_set(binding.tensor, binding.values.data(), 0,
                                sizeof(int32_t) * binding.values.size());
    }
    for (const DeepSeek4I64ArrayBinding & binding : i64_array_inputs) {
        ggml_backend_tensor_set(binding.tensor, binding.values.data(), 0,
                                sizeof(int64_t) * binding.values.size());
    }

    // Compute
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[deepseek4] graph compute failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    // Read logits (only last token for generation)
    out_logits.resize(w.n_vocab);
    const size_t logits_offset = (size_t)(n_tokens - 1) * w.n_vocab * sizeof(float);
    ggml_backend_tensor_get(logits, out_logits.data(), logits_offset,
                            w.n_vocab * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    const int next_pos = kv_start + n_tokens;
    for (int il = 0; il < n_layer; ++il) {
        const uint32_t ratio = w.compress_ratios[il];
        if (ratio <= 0 || (next_pos % (int) ratio) != 0) {
            continue;
        }
        cache.layers[il].n_comp = std::max(cache.layers[il].n_comp, next_pos / (int) ratio);
        if (ratio == 4) {
            cache.layers[il].n_index_comp = std::max(cache.layers[il].n_index_comp,
                                                     next_pos / (int) ratio);
        }
    }

    cache.cur_pos = next_pos;
    return true;
}

bool deepseek4_step_layer_range(
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        DeepSeek4Cache & cache,
        std::vector<float> & hc_state,
        const float * embed,
        int n_tokens,
        int kv_start,
        int layer_begin,
        int layer_end,
        std::vector<float> * out_logits,
        const int32_t * token_ids,
        DeepSeek4StepTelemetry * telemetry) {
    const auto step_t0 = Ds4TimingClock::now();

    // NOTE: The old deepseek4_step() lacks HC implementation.
    // Always use the HC-enabled layer_range path below.

    // ── Partial layer-range forward with HC ─────────────────────────────
    const int n_embd = w.n_embd;
    const int n_hc = w.n_hc;
    const int hc_dim = n_hc * n_embd;
    const bool is_last_shard = (layer_end >= w.n_layer);

    // Initialize HC state.
    // First shard (layer_begin=0): embed is token embeddings [n_embd × n_tokens],
    //   replicate into n_hc streams.
    // Later shards: embed is full HC state [hc_dim × n_tokens] from previous shard.
    if (hc_state.size() != (size_t)hc_dim * (size_t)n_tokens) {
        hc_state.resize((size_t)hc_dim * (size_t)n_tokens);
    }
    if (layer_begin == 0) {
        // First shard: replicate embedding into all HC streams
        for (int t = 0; t < n_tokens; t++) {
            for (int h = 0; h < n_hc; h++) {
                memcpy(hc_state.data() + (size_t)t * hc_dim + (size_t)h * n_embd,
                       embed + (size_t)t * n_embd, (size_t)n_embd * sizeof(float));
            }
        }
    } else {
        // Later shard: embed contains full HC state from previous shard
        memcpy(hc_state.data(), embed, sizeof(float) * (size_t)hc_dim * (size_t)n_tokens);
    }

    // Lazy-load per-layer HC weights on CPU (static to avoid reloading)
    static std::vector<HcLayerWeightsCpu> hc_layer_weights_range;
    static HcWeightsCpu hc_output_weights_range;
    static std::vector<HashRoutingTableCpu> hash_routing_tables_range;
    static std::vector<DeepSeek4CachedLayerAlloc> cached_attn_allocs;
    static std::vector<DeepSeek4CachedDecodeHcPreGraph> cached_decode_attn_hc_pre_graphs;
    static std::vector<DeepSeek4CachedDecodeHcPreGraph> cached_decode_ffn_hc_pre_graphs;
    static DeepSeek4CachedDecodeHcPostGraph cached_decode_hc_post_graph;
    static std::vector<std::vector<DeepSeek4CachedDecodeAttnGraph>> cached_decode_attn_graphs;
    static std::vector<DeepSeek4CachedDecodeFfnGraph> cached_decode_ffn_graphs;
    static DeepSeek4CachedDecodeOutputGraph cached_decode_output_graph;
    static DeepSeek4CachedLayerAlloc cached_dynamic_output_alloc;
    static int hc_loaded_n_layer = 0;
    if (hc_loaded_n_layer != w.n_layer) {
        hc_layer_weights_range.resize((size_t)w.n_layer);
        hash_routing_tables_range.resize((size_t)w.n_layer);
        for (auto & alloc : cached_attn_allocs) {
            alloc.free();
        }
        cached_attn_allocs.assign((size_t)w.n_layer, {});
        for (auto & g : cached_decode_attn_hc_pre_graphs) {
            g.free();
        }
        cached_decode_attn_hc_pre_graphs.assign((size_t)w.n_layer, {});
        for (auto & g : cached_decode_ffn_hc_pre_graphs) {
            g.free();
        }
        cached_decode_ffn_hc_pre_graphs.assign((size_t)w.n_layer, {});
        cached_decode_hc_post_graph.free();
        for (auto & per_layer : cached_decode_attn_graphs) {
            for (auto & g : per_layer) {
                g.free();
            }
        }
        cached_decode_attn_graphs.assign((size_t)w.n_layer, {});
        for (auto & g : cached_decode_ffn_graphs) {
            g.free();
        }
        cached_decode_ffn_graphs.assign((size_t)w.n_layer, {});
        cached_decode_output_graph.free();
        cached_dynamic_output_alloc.free();
        for (int il = 0; il < w.n_layer; il++) {
            const DeepSeek4Layer & L = w.layers[(size_t)il];
            load_hc_weights_cpu(hc_layer_weights_range[il].attn, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base);
            load_hc_weights_cpu(hc_layer_weights_range[il].ffn, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base);
            if (il < w.n_hash_layer && L.ffn_gate_tid2eid) {
                load_hash_routing_cpu(hash_routing_tables_range[(size_t)il], L.ffn_gate_tid2eid);
            }
        }
        load_hc_weights_cpu(hc_output_weights_range, w.output_hc_fn, w.output_hc_scale, w.output_hc_base);
        hc_loaded_n_layer = w.n_layer;
    }

    // Per-layer execution with CPU-side HC
    static thread_local DeepSeek4LayerRangeScratch scratch;
    scratch.ensure(w.ctx, n_tokens, n_embd, n_hc, w.n_expert_used);
    std::vector<float> & cur = scratch.cur;
    std::vector<float> & ffn_working = scratch.ffn_working;
    std::vector<float> & hc_post = scratch.hc_post;
    std::vector<float> & hc_comb = scratch.hc_comb;
    std::vector<float> & next_hc = scratch.next_hc;
    std::vector<float> & attn_out_host = scratch.attn_out_host;
    std::vector<float> & ffn_out_host = scratch.ffn_out_host;
    std::vector<int32_t> & hash_expert_ids_host = scratch.hash_expert_ids;
    const bool reuse_decode_graphs = (n_tokens == 1);
    const bool use_backend_decode_hc = reuse_decode_graphs && ds4_backend_is_gpu(backend);
    const bool use_backend_decode_hc_direct = use_backend_decode_hc && ds4_backend_is_hip(backend);
    const bool use_backend_decode_hc_graph =
        use_backend_decode_hc && !use_backend_decode_hc_direct;
    const ggml_tensor * hc_state_backend = nullptr;
    if (use_backend_decode_hc_graph || use_backend_decode_hc_direct) {
        if (!cached_decode_hc_post_graph.valid() ||
            cached_decode_hc_post_graph.owner_ctx != w.ctx ||
            cached_decode_hc_post_graph.backend != backend) {
            if (!build_cached_decode_hc_post_graph(cached_decode_hc_post_graph, backend, w)) {
                return false;
            }
        }
        ggml_backend_tensor_set(cached_decode_hc_post_graph.residual_hc,
                                hc_state.data(), 0, sizeof(float) * hc_state.size());
        hc_state_backend = cached_decode_hc_post_graph.residual_hc;
    }
    for (int il = layer_begin; il < layer_end; ++il) {
        const DeepSeek4Layer & L = w.layers[(size_t)il];
        DeepSeek4LayerCache & lc = cache.layers[(size_t)il];
        const HcLayerWeightsCpu & hc_lw = hc_layer_weights_range[(size_t)il];
        const int ratio = (int)w.compress_ratios[il];
        bool hash_routed = false;
        const ggml_tensor * attn_in_backend = nullptr;
        const ggml_tensor * ffn_in_backend = nullptr;
        const ggml_tensor * attn_post_backend = nullptr;
        const ggml_tensor * attn_comb_backend = nullptr;
        const ggml_tensor * ffn_post_backend = nullptr;
        const ggml_tensor * ffn_comb_backend = nullptr;

        // ── HC pre (attention) ──────────────────────────────────────
        const auto hc_pre_attn_t0 = Ds4TimingClock::now();
        if (use_backend_decode_hc_direct) {
            const auto hc_pre_attn_input_t0 = Ds4TimingClock::now();
            ggml_backend_tensor_set(cached_decode_hc_post_graph.residual_hc,
                                    hc_state.data(), 0, sizeof(float) * hc_state.size());
            if (telemetry) telemetry->hc_pre_input_us += ds4_elapsed_us(hc_pre_attn_input_t0, Ds4TimingClock::now());
            auto & cached = cached_decode_attn_hc_pre_graphs[(size_t)il];
            if (!cached.valid() ||
                cached.owner_ctx != w.ctx ||
                cached.backend != backend ||
                cached.layer_idx != il ||
                cached.ffn) {
                const auto hc_pre_attn_build_t0 = Ds4TimingClock::now();
                if (!build_cached_decode_hc_pre_graph(cached, backend, w, L, hc_lw.attn.scale_data.data(), il, false)) {
                    std::fprintf(stderr, "[deepseek4] cached hc-pre graph alloc failed layer %d attn\n", il);
                    return false;
                }
                if (telemetry) telemetry->hc_pre_build_us += ds4_elapsed_us(hc_pre_attn_build_t0, Ds4TimingClock::now());
            }
            const auto hc_pre_attn_compute_t0 = Ds4TimingClock::now();
            if (!ds4_try_gpu_hc_pre_device(cached.sg.hidden_states,
                                           cached.post,
                                           cached.comb,
                                           backend,
                                           il,
                                           false,
                                           cached_decode_hc_post_graph.residual_hc,
                                           L.hc_attn_fn,
                                           L.hc_attn_scale,
                                           L.hc_attn_base,
                                           hc_lw.attn.scale_data.data(),
                                           hc_lw.attn.base_data.data(),
                                           n_embd,
                                           n_hc,
                                           w.n_hc_sinkhorn_iter,
                                           w.hc_eps)) {
                std::fprintf(stderr, "[deepseek4] direct hc-pre compute failed layer %d attn\n", il);
                return false;
            }
            if (telemetry) telemetry->hc_pre_compute_us += ds4_elapsed_us(hc_pre_attn_compute_t0, Ds4TimingClock::now());
            attn_in_backend = cached.sg.hidden_states;
            attn_post_backend = cached.post;
            attn_comb_backend = cached.comb;
        } else if (use_backend_decode_hc_graph) {
            auto & cached = cached_decode_attn_hc_pre_graphs[(size_t)il];
            if (!cached.valid() ||
                cached.owner_ctx != w.ctx ||
                cached.backend != backend ||
                cached.layer_idx != il ||
                cached.ffn) {
                const auto hc_pre_attn_build_t0 = Ds4TimingClock::now();
                if (!build_cached_decode_hc_pre_graph(cached, backend, w, L, hc_lw.attn.scale_data.data(), il, false)) {
                    std::fprintf(stderr, "[deepseek4] cached hc-pre graph alloc failed layer %d attn\n", il);
                    return false;
                }
                if (telemetry) telemetry->hc_pre_build_us += ds4_elapsed_us(hc_pre_attn_build_t0, Ds4TimingClock::now());
            }
            const auto hc_pre_attn_input_t0 = Ds4TimingClock::now();
            ggml_backend_tensor_copy(hc_state_backend, cached.sg.inp_embed);
            if (telemetry) telemetry->hc_pre_input_us += ds4_elapsed_us(hc_pre_attn_input_t0, Ds4TimingClock::now());
            const auto hc_pre_attn_compute_t0 = Ds4TimingClock::now();
            if (ggml_backend_graph_compute(backend, cached.sg.gf) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[deepseek4] cached hc-pre compute failed layer %d attn\n", il);
                return false;
            }
            if (telemetry) telemetry->hc_pre_compute_us += ds4_elapsed_us(hc_pre_attn_compute_t0, Ds4TimingClock::now());
            attn_in_backend = cached.sg.hidden_states;
            attn_post_backend = cached.post;
            attn_comb_backend = cached.comb;
        } else {
            hc_pre_batch(cur, hc_post, hc_comb,
                         hc_state.data(), hc_lw.attn, L.hc_attn_fn,
                         n_tokens, n_embd, n_hc, w.n_hc_sinkhorn_iter, w.hc_eps);
        }
        if (telemetry) telemetry->hc_pre_attn_us += ds4_elapsed_us(hc_pre_attn_t0, Ds4TimingClock::now());

        // ── Build & run attention graph ─────────────────────────────
        {
            const int token_pos = kv_start + n_tokens - 1;
            const bool reuse_decode_attn = n_tokens == 1;
            ggml_tensor * attn_out = nullptr;
            ggml_cgraph * gf = nullptr;
            ggml_context * ctx = nullptr;
            DeepSeek4CachedDecodeAttnGraph * cached_attn = nullptr;

            if (reuse_decode_attn) {
                const int n_raw = std::min(kv_start + 1, w.n_swa);
                const int n_comp_attn = (ratio > 0) ? ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio, token_pos) : 0;
                const int n_index_comp = (ratio == 4) ? ds4_comp_rows_used(lc.index_comp_kv, lc.n_index_comp, 4, token_pos) : 0;
                auto & per_layer = cached_decode_attn_graphs[(size_t)il];
                auto it = std::find_if(per_layer.begin(), per_layer.end(),
                    [&](const DeepSeek4CachedDecodeAttnGraph & candidate) {
                        return candidate.valid() &&
                               candidate.owner_ctx == w.ctx &&
                               candidate.backend == backend &&
                               candidate.layer_idx == il &&
                               candidate.n_raw == n_raw &&
                               candidate.n_comp_attn == n_comp_attn &&
                               candidate.n_index_comp == n_index_comp;
                    });
                if (it == per_layer.end()) {
                    if (per_layer.size() >= 20) {
                        per_layer.front().free();
                        per_layer.erase(per_layer.begin());
                    }
                    per_layer.emplace_back();
                    auto & candidate = per_layer.back();
                    const auto attn_build_t0 = Ds4TimingClock::now();
                    if (!build_cached_decode_attn_graph(candidate, backend, w, L, lc, il, kv_start,
                                                        n_raw, n_comp_attn, n_index_comp)) {
                        std::fprintf(stderr, "[deepseek4] cached attn graph alloc failed layer %d\n", il);
                        return false;
                    }
                    if (telemetry) telemetry->attn_build_us += ds4_elapsed_us(attn_build_t0, Ds4TimingClock::now());
                    it = std::prev(per_layer.end());
                }
                cached_attn = &*it;
                gf = cached_attn->sg.gf;
                attn_out = cached_attn->sg.hidden_states;

                const int64_t raw_row = kv_start % w.n_swa;
                const int32_t rope_pos = kv_start;
                const int32_t neg_pos = -kv_start;
                if (attn_in_backend) {
                    ggml_backend_tensor_copy(attn_in_backend, cached_attn->sg.inp_embed);
                } else {
                    ggml_backend_tensor_set(cached_attn->sg.inp_embed, cur.data(), 0, sizeof(float) * cur.size());
                }
                ggml_backend_tensor_set(cached_attn->inputs.rope_pos, &rope_pos, 0, sizeof(rope_pos));
                ggml_backend_tensor_set(cached_attn->inputs.neg_pos, &neg_pos, 0, sizeof(neg_pos));
                ggml_backend_tensor_set(cached_attn->inputs.raw_kv_rows, &raw_row, 0, sizeof(raw_row));
                if (ratio > 0) {
                    const int pos_mod = token_pos % ratio;
                    const int32_t ape_row = pos_mod;
                    const int64_t state_row = ratio + pos_mod;
                    const int64_t comp_row = token_pos / ratio;
                    const int32_t comp_pos = token_pos + 1 - ratio;
                    const bool flush_boundary = ((token_pos + 1) % ratio) == 0;
                    ggml_backend_tensor_set(cached_attn->inputs.attn_ape_row, &ape_row, 0, sizeof(ape_row));
                    ggml_backend_tensor_set(cached_attn->inputs.attn_state_rows, &state_row, 0, sizeof(state_row));
                    if (flush_boundary) {
                        ggml_backend_tensor_set(cached_attn->inputs.attn_comp_rows, &comp_row, 0, sizeof(comp_row));
                        ggml_backend_tensor_set(cached_attn->inputs.attn_comp_pos, &comp_pos, 0, sizeof(comp_pos));
                    }
                }
                if (ratio == 4) {
                    const int pos_mod = token_pos % ratio;
                    const int32_t ape_row = pos_mod;
                    const int64_t state_row = ratio + pos_mod;
                    const int64_t comp_row = token_pos / ratio;
                    const int32_t comp_pos = token_pos + 1 - ratio;
                    const bool flush_boundary = ((token_pos + 1) % ratio) == 0;
                    ggml_backend_tensor_set(cached_attn->inputs.index_ape_row, &ape_row, 0, sizeof(ape_row));
                    ggml_backend_tensor_set(cached_attn->inputs.index_state_rows, &state_row, 0, sizeof(state_row));
                    if (flush_boundary) {
                        ggml_backend_tensor_set(cached_attn->inputs.index_comp_rows, &comp_row, 0, sizeof(comp_row));
                        ggml_backend_tensor_set(cached_attn->inputs.index_comp_pos, &comp_pos, 0, sizeof(comp_pos));
                    }
                }
            } else {
                const auto attn_build_t0 = Ds4TimingClock::now();
                const size_t ctx_size = 48 * 1024 * 1024;
                ggml_init_params params{};
                params.mem_size = ctx_size;
                params.mem_buffer = nullptr;
                params.no_alloc = true;
                ctx = ggml_init(params);
                if (!ctx) return false;

                ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
                ggml_set_input(inp);
                std::vector<DeepSeek4I32InputBinding> i32_inputs;
                std::vector<DeepSeek4I32ArrayBinding> i32_array_inputs;
                std::vector<DeepSeek4I64ArrayBinding> i64_array_inputs;
                const size_t graph_size = n_tokens > 1 ? 32768 : 2048;
                gf = ggml_new_graph_custom(ctx, graph_size, false);

                ggml_tensor * normed = build_rms_norm(ctx, inp, L.attn_norm, w.rms_eps);
                attn_out = build_mla_attention(ctx, gf, normed, w, L, lc, il,
                                               kv_start, n_tokens, nullptr,
                                               i32_inputs, i32_array_inputs,
                                               i64_array_inputs);
                ggml_set_output(attn_out);
                ggml_build_forward_expand(gf, attn_out);

                auto & attn_alloc = cached_attn_allocs[(size_t)il];
                if (!attn_alloc.valid() || attn_alloc.owner_ctx != w.ctx || attn_alloc.backend != backend) {
                    attn_alloc.free();
                    attn_alloc.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                    attn_alloc.owner_ctx = w.ctx;
                    attn_alloc.backend = backend;
                }
                if (!attn_alloc.alloc || !ggml_gallocr_alloc_graph(attn_alloc.alloc, gf)) {
                    std::fprintf(stderr, "[deepseek4] attn graph alloc failed layer %d\n", il);
                    ggml_free(ctx);
                    return false;
                }
                if (telemetry) telemetry->attn_build_us += ds4_elapsed_us(attn_build_t0, Ds4TimingClock::now());
                if (attn_in_backend) {
                    ggml_backend_tensor_copy(attn_in_backend, inp);
                } else {
                    ggml_backend_tensor_set(inp, cur.data(), 0, sizeof(float) * cur.size());
                }
                for (const auto & b : i32_inputs)
                    ggml_backend_tensor_set(b.tensor, &b.value, 0, sizeof(b.value));
                for (const auto & b : i32_array_inputs)
                    ggml_backend_tensor_set(b.tensor, b.values.data(), 0, sizeof(int32_t) * b.values.size());
                for (const auto & b : i64_array_inputs)
                    ggml_backend_tensor_set(b.tensor, b.values.data(), 0, sizeof(int64_t) * b.values.size());
            }

            const auto attn_compute_t0 = Ds4TimingClock::now();
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[deepseek4] attn compute failed layer %d\n", il);
                if (ctx) ggml_free(ctx);
                return false;
            }
            if (telemetry) telemetry->attn_compute_us += ds4_elapsed_us(attn_compute_t0, Ds4TimingClock::now());
            if (use_backend_decode_hc_graph) {
                if (hc_state_backend != cached_decode_hc_post_graph.residual_hc) {
                    ggml_backend_tensor_copy(hc_state_backend, cached_decode_hc_post_graph.residual_hc);
                }
                ggml_backend_tensor_copy(attn_out, cached_decode_hc_post_graph.block_out);
                ggml_backend_tensor_copy(attn_post_backend, cached_decode_hc_post_graph.post);
                ggml_backend_tensor_copy(attn_comb_backend, cached_decode_hc_post_graph.comb);
                const auto hc_post_attn_t0 = Ds4TimingClock::now();
                if (ggml_backend_graph_compute(backend, cached_decode_hc_post_graph.sg.gf) != GGML_STATUS_SUCCESS) {
                    std::fprintf(stderr, "[deepseek4] cached hc-post compute failed layer %d attn\n", il);
                    if (ctx) ggml_free(ctx);
                    return false;
                }
                hc_state_backend = cached_decode_hc_post_graph.sg.hidden_states;
                if (telemetry) telemetry->hc_post_attn_us += ds4_elapsed_us(hc_post_attn_t0, Ds4TimingClock::now());
            } else {
                const auto attn_read_t0 = Ds4TimingClock::now();
                ggml_backend_tensor_get(attn_out, attn_out_host.data(), 0, sizeof(float) * attn_out_host.size());
                if (use_backend_decode_hc_direct) {
                    ggml_backend_tensor_get(attn_post_backend, hc_post.data(), 0, sizeof(float) * hc_post.size());
                    ggml_backend_tensor_get(attn_comb_backend, hc_comb.data(), 0, sizeof(float) * hc_comb.size());
                }
                if (telemetry) telemetry->attn_read_us += ds4_elapsed_us(attn_read_t0, Ds4TimingClock::now());
            }
            if (ctx) ggml_free(ctx);

            // ── HC post (attention) ─────────────────────────────────
            if (!use_backend_decode_hc_graph) {
                const auto hc_post_attn_t0 = Ds4TimingClock::now();
                hc_post_batch(next_hc,
                              attn_out_host.data(),
                              hc_state.data(),
                              hc_post.data(),
                              hc_comb.data(),
                              n_tokens,
                              n_embd,
                              n_hc);
                std::memcpy(hc_state.data(), next_hc.data(), next_hc.size() * sizeof(float));
                if (telemetry) telemetry->hc_post_attn_us += ds4_elapsed_us(hc_post_attn_t0, Ds4TimingClock::now());
            }
        }

        // ── HC pre (FFN) ────────────────────────────────────────────
        const auto hc_pre_ffn_t0 = Ds4TimingClock::now();
        if (use_backend_decode_hc_direct) {
            const auto hc_pre_ffn_input_t0 = Ds4TimingClock::now();
            ggml_backend_tensor_set(cached_decode_hc_post_graph.residual_hc,
                                    hc_state.data(), 0, sizeof(float) * hc_state.size());
            if (telemetry) telemetry->hc_pre_input_us += ds4_elapsed_us(hc_pre_ffn_input_t0, Ds4TimingClock::now());
            auto & cached = cached_decode_ffn_hc_pre_graphs[(size_t)il];
            if (!cached.valid() ||
                cached.owner_ctx != w.ctx ||
                cached.backend != backend ||
                cached.layer_idx != il ||
                !cached.ffn) {
                const auto hc_pre_ffn_build_t0 = Ds4TimingClock::now();
                if (!build_cached_decode_hc_pre_graph(cached, backend, w, L, hc_lw.ffn.scale_data.data(), il, true)) {
                    std::fprintf(stderr, "[deepseek4] cached hc-pre graph alloc failed layer %d ffn\n", il);
                    return false;
                }
                if (telemetry) telemetry->hc_pre_build_us += ds4_elapsed_us(hc_pre_ffn_build_t0, Ds4TimingClock::now());
            }
            const auto hc_pre_ffn_compute_t0 = Ds4TimingClock::now();
            if (!ds4_try_gpu_hc_pre_device(cached.sg.hidden_states,
                                           cached.post,
                                           cached.comb,
                                           backend,
                                           il,
                                           true,
                                           cached_decode_hc_post_graph.residual_hc,
                                           L.hc_ffn_fn,
                                           L.hc_ffn_scale,
                                           L.hc_ffn_base,
                                           hc_lw.ffn.scale_data.data(),
                                           hc_lw.ffn.base_data.data(),
                                           n_embd,
                                           n_hc,
                                           w.n_hc_sinkhorn_iter,
                                           w.hc_eps)) {
                std::fprintf(stderr, "[deepseek4] direct hc-pre compute failed layer %d ffn\n", il);
                return false;
            }
            if (telemetry) telemetry->hc_pre_compute_us += ds4_elapsed_us(hc_pre_ffn_compute_t0, Ds4TimingClock::now());
            ffn_in_backend = cached.sg.hidden_states;
            ffn_post_backend = cached.post;
            ffn_comb_backend = cached.comb;
        } else if (use_backend_decode_hc_graph) {
            auto & cached = cached_decode_ffn_hc_pre_graphs[(size_t)il];
            if (!cached.valid() ||
                cached.owner_ctx != w.ctx ||
                cached.backend != backend ||
                cached.layer_idx != il ||
                !cached.ffn) {
                const auto hc_pre_ffn_build_t0 = Ds4TimingClock::now();
                if (!build_cached_decode_hc_pre_graph(cached, backend, w, L, hc_lw.ffn.scale_data.data(), il, true)) {
                    std::fprintf(stderr, "[deepseek4] cached hc-pre graph alloc failed layer %d ffn\n", il);
                    return false;
                }
                if (telemetry) telemetry->hc_pre_build_us += ds4_elapsed_us(hc_pre_ffn_build_t0, Ds4TimingClock::now());
            }
            const auto hc_pre_ffn_input_t0 = Ds4TimingClock::now();
            ggml_backend_tensor_copy(hc_state_backend, cached.sg.inp_embed);
            if (telemetry) telemetry->hc_pre_input_us += ds4_elapsed_us(hc_pre_ffn_input_t0, Ds4TimingClock::now());
            const auto hc_pre_ffn_compute_t0 = Ds4TimingClock::now();
            if (ggml_backend_graph_compute(backend, cached.sg.gf) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[deepseek4] cached hc-pre compute failed layer %d ffn\n", il);
                return false;
            }
            if (telemetry) telemetry->hc_pre_compute_us += ds4_elapsed_us(hc_pre_ffn_compute_t0, Ds4TimingClock::now());
            ffn_in_backend = cached.sg.hidden_states;
            ffn_post_backend = cached.post;
            ffn_comb_backend = cached.comb;
        } else {
            hc_pre_batch(ffn_working, hc_post, hc_comb,
                         hc_state.data(), hc_lw.ffn, L.hc_ffn_fn,
                         n_tokens, n_embd, n_hc, w.n_hc_sinkhorn_iter, w.hc_eps);
        }
        if (telemetry) telemetry->hc_pre_ffn_us += ds4_elapsed_us(hc_pre_ffn_t0, Ds4TimingClock::now());

        // ── Build & run FFN graph ───────────────────────────────────
        {
            // Hash-routed layers: use pre-computed expert IDs from hash table
            // instead of zeroing out routed_out as build_moe_ffn does.
            hash_routed =
                il < w.n_hash_layer && L.ffn_gate_tid2eid && token_ids &&
                hash_routing_tables_range[(size_t)il].loaded;
            if (hash_routed) {
                const auto & ht = hash_routing_tables_range[(size_t)il].ids;
                const int n_used = w.n_expert_used;
                hash_expert_ids_host.resize((size_t)n_used * (size_t)n_tokens);
                for (int ti = 0; ti < n_tokens; ti++) {
                    const int32_t tok = token_ids[ti];
                    const int32_t * row = ht.data() + (size_t)tok * (size_t)n_used;
                    memcpy(hash_expert_ids_host.data() + (size_t)ti * n_used,
                           row, (size_t)n_used * sizeof(int32_t));
                }
            }

            auto & cached = cached_decode_ffn_graphs[(size_t)il];
            if (!cached.valid() ||
                cached.owner_ctx != w.ctx ||
                cached.backend != backend ||
                cached.layer_idx != il ||
                cached.n_tokens != n_tokens ||
                cached.hash_routed != hash_routed) {
                const auto ffn_build_t0 = Ds4TimingClock::now();
                if (!build_cached_decode_ffn_graph(cached, backend, w, L, il, n_tokens, hash_routed)) {
                    std::fprintf(stderr, "[deepseek4] cached ffn graph alloc failed layer %d\n", il);
                    return false;
                }
                if (telemetry) telemetry->ffn_build_us += ds4_elapsed_us(ffn_build_t0, Ds4TimingClock::now());
            }

            ggml_tensor * ffn_out = cached.sg.hidden_states;
            if (ffn_in_backend) {
                ggml_backend_tensor_copy(ffn_in_backend, cached.sg.inp_embed);
            } else {
                ggml_backend_tensor_set(cached.sg.inp_embed, ffn_working.data(), 0,
                                        sizeof(float) * ffn_working.size());
            }
            if (cached.hash_ids) {
                ggml_backend_tensor_set(cached.hash_ids, hash_expert_ids_host.data(), 0,
                                        sizeof(int32_t) * hash_expert_ids_host.size());
            }

            const auto ffn_compute_t0 = Ds4TimingClock::now();
            auto status = ggml_backend_graph_compute(backend, cached.sg.gf);
            if (telemetry) telemetry->ffn_compute_us += ds4_elapsed_us(ffn_compute_t0, Ds4TimingClock::now());
            if (status != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[deepseek4] cached ffn compute failed layer %d\n", il);
                return false;
            }

            if (use_backend_decode_hc_graph) {
                if (hc_state_backend != cached_decode_hc_post_graph.residual_hc) {
                    ggml_backend_tensor_copy(hc_state_backend, cached_decode_hc_post_graph.residual_hc);
                }
                ggml_backend_tensor_copy(ffn_out, cached_decode_hc_post_graph.block_out);
                ggml_backend_tensor_copy(ffn_post_backend, cached_decode_hc_post_graph.post);
                ggml_backend_tensor_copy(ffn_comb_backend, cached_decode_hc_post_graph.comb);
                const auto hc_post_ffn_t0 = Ds4TimingClock::now();
                if (ggml_backend_graph_compute(backend, cached_decode_hc_post_graph.sg.gf) != GGML_STATUS_SUCCESS) {
                    std::fprintf(stderr, "[deepseek4] cached hc-post compute failed layer %d ffn\n", il);
                    return false;
                }
                hc_state_backend = cached_decode_hc_post_graph.sg.hidden_states;
                if (telemetry) telemetry->hc_post_ffn_us += ds4_elapsed_us(hc_post_ffn_t0, Ds4TimingClock::now());
            } else {
                const auto ffn_read_t0 = Ds4TimingClock::now();
                ggml_backend_tensor_get(ffn_out, ffn_out_host.data(), 0, sizeof(float) * ffn_out_host.size());
                if (use_backend_decode_hc_direct) {
                    ggml_backend_tensor_get(ffn_post_backend, hc_post.data(), 0, sizeof(float) * hc_post.size());
                    ggml_backend_tensor_get(ffn_comb_backend, hc_comb.data(), 0, sizeof(float) * hc_comb.size());
                }
                if (telemetry) telemetry->ffn_read_us += ds4_elapsed_us(ffn_read_t0, Ds4TimingClock::now());
            }

            // ── HC post (FFN) ───────────────────────────────────────
            if (!use_backend_decode_hc_graph) {
                const auto hc_post_ffn_t0 = Ds4TimingClock::now();
                hc_post_batch(next_hc,
                              ffn_out_host.data(),
                              hc_state.data(),
                              hc_post.data(),
                              hc_comb.data(),
                              n_tokens,
                              n_embd,
                              n_hc);
                std::memcpy(hc_state.data(), next_hc.data(), next_hc.size() * sizeof(float));
                if (telemetry) telemetry->hc_post_ffn_us += ds4_elapsed_us(hc_post_ffn_t0, Ds4TimingClock::now());
            }
        }
    }

    if (use_backend_decode_hc_graph && hc_state_backend) {
        ggml_backend_tensor_get(hc_state_backend, hc_state.data(), 0, sizeof(float) * hc_state.size());
    }

    // ── Output: HC pre → norm → lm_head (or return hidden state) ────────
    if (is_last_shard && out_logits) {
        // Final HC pre for output
        const auto output_t0 = Ds4TimingClock::now();
        std::vector<float> & final_embd = scratch.final_embd;
        hc_output_batch(final_embd,
                        hc_state.data(),
                        hc_output_weights_range,
                        n_tokens,
                        n_embd,
                        n_hc,
                        w.hc_eps);

        if (reuse_decode_graphs) {
            if (!cached_decode_output_graph.valid() ||
                cached_decode_output_graph.owner_ctx != w.ctx ||
                cached_decode_output_graph.backend != backend ||
                cached_decode_output_graph.n_tokens != n_tokens) {
                if (!build_cached_decode_output_graph(cached_decode_output_graph, backend, w, n_tokens)) {
                    return false;
                }
            }
            ggml_backend_tensor_set(cached_decode_output_graph.sg.hidden_input,
                                    final_embd.data(), 0, sizeof(float) * final_embd.size());
            if (ggml_backend_graph_compute(backend, cached_decode_output_graph.sg.gf) != GGML_STATUS_SUCCESS) {
                return false;
            }
            out_logits->resize((size_t)w.n_vocab);
            ggml_backend_tensor_get(cached_decode_output_graph.sg.logits,
                                    out_logits->data(), 0, sizeof(float) * (size_t)w.n_vocab);
        } else {
            const size_t ctx_size = 16 * 1024 * 1024;
            ggml_init_params params{};
            params.mem_size = ctx_size;
            params.mem_buffer = nullptr;
            params.no_alloc = true;
            ggml_context * ctx = ggml_init(params);
            if (!ctx) return false;

            ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
            ggml_set_input(inp);
            ggml_tensor * normed = build_rms_norm(ctx, inp, w.out_norm, w.rms_eps);
            ggml_tensor * logits = ggml_mul_mat(ctx, w.output, normed);
            ggml_set_output(logits);
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);
            ggml_build_forward_expand(gf, logits);

            if (!cached_dynamic_output_alloc.valid() ||
                cached_dynamic_output_alloc.owner_ctx != w.ctx ||
                cached_dynamic_output_alloc.backend != backend) {
                cached_dynamic_output_alloc.free();
                cached_dynamic_output_alloc.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                cached_dynamic_output_alloc.owner_ctx = w.ctx;
                cached_dynamic_output_alloc.backend = backend;
            }
            if (!cached_dynamic_output_alloc.alloc ||
                !ggml_gallocr_alloc_graph(cached_dynamic_output_alloc.alloc, gf)) {
                ggml_free(ctx);
                return false;
            }
            ggml_backend_tensor_set(inp, final_embd.data(), 0, sizeof(float) * final_embd.size());
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                ggml_free(ctx);
                return false;
            }

            out_logits->resize((size_t)w.n_vocab);
            const size_t logits_offset = (size_t)(n_tokens - 1) * (size_t)w.n_vocab * sizeof(float);
            ggml_backend_tensor_get(logits, out_logits->data(), logits_offset,
                                    sizeof(float) * (size_t)w.n_vocab);
            ggml_free(ctx);
        }
        if (telemetry) telemetry->output_us += ds4_elapsed_us(output_t0, Ds4TimingClock::now());
    } else if (out_logits) {
        // Return full HC state for next shard (all n_hc streams)
        out_logits->resize((size_t)hc_dim * n_tokens);
        memcpy(out_logits->data(), hc_state.data(), sizeof(float) * hc_dim * n_tokens);
    }

    // Update compressor state
    const int next_pos = kv_start + n_tokens;
    for (int il = layer_begin; il < layer_end; ++il) {
        const uint32_t ratio = w.compress_ratios[il];
        if (ratio <= 0 || (next_pos % (int)ratio) != 0) continue;
        cache.layers[il].n_comp = std::max(cache.layers[il].n_comp, next_pos / (int)ratio);
        if (ratio == 4) {
            cache.layers[il].n_index_comp = std::max(cache.layers[il].n_index_comp,
                                                     next_pos / (int)ratio);
        }
    }

    cache.cur_pos = next_pos;
    if (telemetry) telemetry->total_us += ds4_elapsed_us(step_t0, Ds4TimingClock::now());
    return true;
}

// ─── Cache management ───────────────────────────────────────────────────

bool create_deepseek4_cache(ggml_backend_t backend,
                             const DeepSeek4Weights & w,
                             int max_ctx,
                             DeepSeek4Cache & out) {
    out.n_layer = w.n_layer;
    out.max_ctx = max_ctx;
    out.cur_pos = 0;
    out.layers.resize(w.n_layer);

    ggml_init_params ctx_params{};
    ctx_params.mem_size = ggml_tensor_overhead() * (size_t)(w.n_layer * 9 + 8) + 4096;
    ctx_params.no_alloc = true;
    out.ctx = ggml_init(ctx_params);
    if (!out.ctx) {
        return false;
    }

    for (int il = 0; il < w.n_layer; ++il) {
        DeepSeek4LayerCache & lc = out.layers[il];
        const uint32_t ratio = w.compress_ratios[il];

        lc.raw_kv = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F16, w.head_dim, w.n_swa);
        char name[64];
        std::snprintf(name, sizeof(name), "ds4_raw_kv_%d", il);
        ggml_set_name(lc.raw_kv, name);

        lc.n_comp = 0;
        lc.n_index_comp = 0;

        if (ratio <= 0) {
            continue;
        }

        const int comp_cap = max_ctx / (int) ratio + 16;
        lc.comp_kv = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F16, w.head_dim, comp_cap);
        std::snprintf(name, sizeof(name), "ds4_comp_kv_%d", il);
        ggml_set_name(lc.comp_kv, name);

        // Compressor state dimensions: comp_width = coff * head_dim
        // Number of state rows: 2*ratio for ratio-4 (prev+cur windows), ratio for ratio-128
        const int coff = (ratio == 4) ? 2 : 1;
        const int comp_width = coff * (int)w.head_dim;
        const int n_state_rows = (ratio == 4) ? (2 * ratio) : ratio;
        lc.attn_compressor.state_kv = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, comp_width, n_state_rows);
        lc.attn_compressor.state_score = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, comp_width, n_state_rows);
        std::snprintf(name, sizeof(name), "ds4_comp_state_kv_%d", il);
        ggml_set_name(lc.attn_compressor.state_kv, name);
        std::snprintf(name, sizeof(name), "ds4_comp_state_score_%d", il);
        ggml_set_name(lc.attn_compressor.state_score, name);

        if (ratio == 4) {
            // Indexer comp_width = 2 * indexer_head_dim = 256
            const int index_comp_width = 2 * (int)w.n_indexer_head_dim;
            const int index_state_rows = 2 * ratio;  // same double-buffer for ratio-4
            lc.index_comp_kv = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F16,
                                                  w.n_indexer_head_dim, comp_cap);
            lc.indexer_compressor.state_kv = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32,
                                                                index_comp_width, index_state_rows);
            lc.indexer_compressor.state_score = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32,
                                                                   index_comp_width, index_state_rows);
            std::snprintf(name, sizeof(name), "ds4_index_comp_kv_%d", il);
            ggml_set_name(lc.index_comp_kv, name);
            std::snprintf(name, sizeof(name), "ds4_index_state_kv_%d", il);
            ggml_set_name(lc.indexer_compressor.state_kv, name);
            std::snprintf(name, sizeof(name), "ds4_index_state_score_%d", il);
            ggml_set_name(lc.indexer_compressor.state_score, name);
        }
    }

    out.hc_state = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, (int64_t)w.n_hc * w.n_embd);
    ggml_set_name(out.hc_state, "ds4_hc_state");

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    ggml_backend_buffer_clear(out.buf, 0);
    const size_t total_bytes = ggml_backend_buffer_get_size(out.buf);
    std::fprintf(stderr, "[deepseek4] KV cache: %.1f MB for ctx=%d\n",
                 (double)total_bytes / (1024.0 * 1024.0), max_ctx);
    return true;
}

namespace {

ggml_tensor * clone_snapshot_tensor(ggml_context * ctx,
                                    const ggml_tensor * src,
                                    const char * name) {
    if (!ctx || !src) return nullptr;
    ggml_tensor * dst = ggml_dup_tensor(ctx, const_cast<ggml_tensor *>(src));
    if (!dst) return nullptr;
    if (name && *name) ggml_set_name(dst, name);
    return dst;
}

bool copy_tensor_from_backend(const ggml_tensor * src, ggml_tensor * dst) {
    if (!src || !dst) return false;
    const size_t bytes = ggml_nbytes(src);
    if (bytes != ggml_nbytes(dst)) return false;
    ggml_backend_tensor_get(src, dst->data, 0, bytes);
    return true;
}

bool copy_tensor_to_backend(const ggml_tensor * src, ggml_tensor * dst) {
    if (!src || !dst) return false;
    const size_t bytes = ggml_nbytes(src);
    if (bytes != ggml_nbytes(dst)) return false;
    ggml_backend_tensor_set(dst, src->data, 0, bytes);
    return true;
}

bool tensors_compatible(const ggml_tensor * a, const ggml_tensor * b) {
    if (!!a != !!b) return false;
    if (!a) return true;
    if (a->type != b->type || ggml_n_dims(a) != ggml_n_dims(b)) return false;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (a->ne[i] != b->ne[i]) return false;
    }
    return true;
}

}  // namespace

bool deepseek4_snapshot_save(const DeepSeek4Cache & cache,
                             ggml_backend_t snapshot_backend,
                             DeepSeek4Snapshot & out) {
    if (!snapshot_backend || !cache.ctx || !cache.buf || !cache.hc_state ||
        cache.layers.size() != (size_t)cache.n_layer) {
        return false;
    }

    free_deepseek4_snapshot(out);

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (size_t)(cache.n_layer * 8 + 8) + 4096;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) {
        return false;
    }

    out.layers.resize((size_t)cache.n_layer);
    out.hc_state_snap = clone_snapshot_tensor(out.ctx, cache.hc_state, "ds4_hc_state_snap");
    if (!out.hc_state_snap) {
        free_deepseek4_snapshot(out);
        return false;
    }

    for (int il = 0; il < cache.n_layer; ++il) {
        const auto & src = cache.layers[(size_t)il];
        auto & dst = out.layers[(size_t)il];
        dst.raw_kv = clone_snapshot_tensor(out.ctx, src.raw_kv, nullptr);
        dst.comp_kv = clone_snapshot_tensor(out.ctx, src.comp_kv, nullptr);
        dst.index_comp_kv = clone_snapshot_tensor(out.ctx, src.index_comp_kv, nullptr);
        dst.attn_compressor.state_kv =
            clone_snapshot_tensor(out.ctx, src.attn_compressor.state_kv, nullptr);
        dst.attn_compressor.state_score =
            clone_snapshot_tensor(out.ctx, src.attn_compressor.state_score, nullptr);
        dst.indexer_compressor.state_kv =
            clone_snapshot_tensor(out.ctx, src.indexer_compressor.state_kv, nullptr);
        dst.indexer_compressor.state_score =
            clone_snapshot_tensor(out.ctx, src.indexer_compressor.state_score, nullptr);
        if (!dst.raw_kv ||
            (src.comp_kv && !dst.comp_kv) ||
            (src.index_comp_kv && !dst.index_comp_kv) ||
            (src.attn_compressor.state_kv && !dst.attn_compressor.state_kv) ||
            (src.attn_compressor.state_score && !dst.attn_compressor.state_score) ||
            (src.indexer_compressor.state_kv && !dst.indexer_compressor.state_kv) ||
            (src.indexer_compressor.state_score && !dst.indexer_compressor.state_score)) {
            free_deepseek4_snapshot(out);
            return false;
        }
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, snapshot_backend);
    if (!out.buf) {
        free_deepseek4_snapshot(out);
        return false;
    }

    if (!copy_tensor_from_backend(cache.hc_state, out.hc_state_snap)) {
        free_deepseek4_snapshot(out);
        return false;
    }
    for (int il = 0; il < cache.n_layer; ++il) {
        const auto & src = cache.layers[(size_t)il];
        auto & dst = out.layers[(size_t)il];
        dst.n_comp = src.n_comp;
        dst.n_index_comp = src.n_index_comp;
        if (!copy_tensor_from_backend(src.raw_kv, dst.raw_kv) ||
            (src.comp_kv && !copy_tensor_from_backend(src.comp_kv, dst.comp_kv)) ||
            (src.index_comp_kv &&
             !copy_tensor_from_backend(src.index_comp_kv, dst.index_comp_kv)) ||
            (src.attn_compressor.state_kv &&
             !copy_tensor_from_backend(src.attn_compressor.state_kv,
                                       dst.attn_compressor.state_kv)) ||
            (src.attn_compressor.state_score &&
             !copy_tensor_from_backend(src.attn_compressor.state_score,
                                       dst.attn_compressor.state_score)) ||
            (src.indexer_compressor.state_kv &&
             !copy_tensor_from_backend(src.indexer_compressor.state_kv,
                                       dst.indexer_compressor.state_kv)) ||
            (src.indexer_compressor.state_score &&
             !copy_tensor_from_backend(src.indexer_compressor.state_score,
                                       dst.indexer_compressor.state_score))) {
            free_deepseek4_snapshot(out);
            return false;
        }
    }

    out.cur_pos = cache.cur_pos;
    return true;
}

bool deepseek4_snapshot_restore(const DeepSeek4Snapshot & snap,
                                DeepSeek4Cache & cache) {
    if (!snap.ctx || !cache.ctx || !cache.buf || !snap.hc_state_snap ||
        snap.layers.size() != cache.layers.size()) {
        return false;
    }
    if (!tensors_compatible(snap.hc_state_snap, cache.hc_state) ||
        !copy_tensor_to_backend(snap.hc_state_snap, cache.hc_state)) {
        return false;
    }

    for (size_t il = 0; il < cache.layers.size(); ++il) {
        const auto & src = snap.layers[il];
        auto & dst = cache.layers[il];
        if (!tensors_compatible(src.raw_kv, dst.raw_kv) ||
            !tensors_compatible(src.comp_kv, dst.comp_kv) ||
            !tensors_compatible(src.index_comp_kv, dst.index_comp_kv) ||
            !tensors_compatible(src.attn_compressor.state_kv, dst.attn_compressor.state_kv) ||
            !tensors_compatible(src.attn_compressor.state_score, dst.attn_compressor.state_score) ||
            !tensors_compatible(src.indexer_compressor.state_kv, dst.indexer_compressor.state_kv) ||
            !tensors_compatible(src.indexer_compressor.state_score, dst.indexer_compressor.state_score)) {
            return false;
        }
        if (!copy_tensor_to_backend(src.raw_kv, dst.raw_kv) ||
            (src.comp_kv && !copy_tensor_to_backend(src.comp_kv, dst.comp_kv)) ||
            (src.index_comp_kv &&
             !copy_tensor_to_backend(src.index_comp_kv, dst.index_comp_kv)) ||
            (src.attn_compressor.state_kv &&
             !copy_tensor_to_backend(src.attn_compressor.state_kv,
                                     dst.attn_compressor.state_kv)) ||
            (src.attn_compressor.state_score &&
             !copy_tensor_to_backend(src.attn_compressor.state_score,
                                     dst.attn_compressor.state_score)) ||
            (src.indexer_compressor.state_kv &&
             !copy_tensor_to_backend(src.indexer_compressor.state_kv,
                                     dst.indexer_compressor.state_kv)) ||
            (src.indexer_compressor.state_score &&
             !copy_tensor_to_backend(src.indexer_compressor.state_score,
                                     dst.indexer_compressor.state_score))) {
            return false;
        }
        dst.n_comp = src.n_comp;
        dst.n_index_comp = src.n_index_comp;
    }

    cache.cur_pos = snap.cur_pos;
    return true;
}

void free_deepseek4_cache(DeepSeek4Cache & c) {
    if (c.buf) { ggml_backend_buffer_free(c.buf); c.buf = nullptr; }
    if (c.ctx) { ggml_free(c.ctx); c.ctx = nullptr; }
    c.layers.clear();
    c.hc_state = nullptr;
}

void free_deepseek4_snapshot(DeepSeek4Snapshot & s) {
    if (s.buf) { ggml_backend_buffer_free(s.buf); s.buf = nullptr; }
    if (s.ctx) { ggml_free(s.ctx); s.ctx = nullptr; }
    s.layers.clear();
    s.cur_pos = 0;
    s.hc_state_snap = nullptr;
}

}  // namespace dflash::common
