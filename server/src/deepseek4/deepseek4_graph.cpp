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
#include <mutex>
#include <functional>
#include <vector>

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

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

// Opt-in: reorders expert-FFN float accumulation, so output is not
// bit-identical to the reference path. Default OFF.
static bool ds4_ffn_raw_mmid_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = ds4_env_flag("DFLASH_DS4_FFN_RAW_MMID") ? 1 : 0;
    }
    return enabled == 1;
}

// Opt-in: reorders expert-output combination, so output is not
// bit-identical to the reference path. Default OFF.
static bool ds4_ffn_fused_combine_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = ds4_env_flag("DFLASH_DS4_FFN_FUSED_COMBINE") ? 1 : 0;
    }
    return enabled == 1;
}

static bool ds4_rocmfpx_hc_gpu_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = ds4_env_flag("DFLASH_DS4_ROCMFPX_HC_GPU") ? 1 : 0;
    }
    return enabled == 1;
}

static size_t ds4_full_step_graph_size(int n_tokens) {
    if (n_tokens <= 1) {
        return 16384;
    }
    if (n_tokens <= 128) {
        return 65536;
    }
    if (n_tokens <= 512) {
        return 131072;
    }
    if (n_tokens <= 1024) {
        return 262144;
    }
    if (n_tokens <= 2048) {
        return 524288;
    }
    if (n_tokens <= 4096) {
        return 1048576;
    }
    return 1572864;
}

static size_t ds4_full_step_meta_size(int n_tokens) {
    const size_t graph_size = ds4_full_step_graph_size(n_tokens);
    size_t arena_size = ggml_tensor_overhead() * 65536
                      + ggml_graph_overhead_custom(graph_size, false)
                      + 16 * 1024 * 1024
                      + 1024 * 1024
                      + 64 * 1024;
    if (n_tokens >= 512) {
        arena_size += (size_t)n_tokens * 32 * 1024;
    }
    if (n_tokens >= 1024) {
        arena_size += 4 * 1024 * 1024;
    }
    if (n_tokens > 1024) {
        // 2K-token full steps need a materially larger meta arena than the
        // 1K-token path once the full graph and its late scratch tensors are
        // fully materialized, so keep a coarse but stable reserve here. The
        // exact scaling can be tightened in the follow-up chunk-sizing pass.
        arena_size += 256 * 1024 * 1024;
    }
    return arena_size;
}

static size_t ds4_attn_step_meta_size(int n_tokens) {
    size_t arena_size = 48 * 1024 * 1024;
    if (n_tokens >= 512) {
        arena_size += (size_t)n_tokens * 32 * 1024;
    }
    return arena_size;
}

static size_t ds4_attn_step_graph_size(int n_tokens) {
    if (n_tokens <= 1) {
        return 2048;
    }
    if (n_tokens <= 512) {
        return 32768;
    }
    if (n_tokens <= 1024) {
        return 131072;
    }
    if (n_tokens <= 2048) {
        return 262144;
    }
    if (n_tokens <= 4096) {
        return 524288;
    }
    return 1048576;
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

struct DeepSeek4F32ArrayBinding {
    ggml_tensor *          tensor = nullptr;
    std::vector<float>     values;
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

struct DeepSeek4LegacyFullStepCache {
    const ggml_context * owner_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    StepGraph sg;
    std::vector<uint8_t> meta_arena;

    void free() {
        step_graph_destroy(sg);
        meta_arena.clear();
        meta_arena.shrink_to_fit();
        owner_ctx = nullptr;
        backend = nullptr;
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
    // Fused-decode stable-KV path only: additive score mask over
    // [n_swa raw rows ++ padded comp rows]; 0 for valid, -1e30 for padding.
    ggml_tensor * attn_row_mask = nullptr;
    int           padded_comp = 0;   // padded compressed-row count (>= n_comp)
    // Fused-decode stable-topology compressor: i64[4] target rows for the
    // ratio-4 state double-buffer flush copy. [0..3] on flush steps (cur ->
    // prev), [4..7] otherwise (self-write no-op). Null = legacy build-time
    // flush branch.
    ggml_tensor * flush_rows = nullptr;
};

struct DeepSeek4CachedDecodeAttnGraph {
    const ggml_context * owner_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    int layer_idx = -1;
    int n_tokens = 0;
    int n_raw = 0;
    int n_comp_attn = 0;
    int n_index_comp = 0;
    bool attn_flush = false;
    bool index_flush = false;
    bool compressed = false;
    bool indexed = false;
    bool uses_shared_inputs = false;
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
        attn_flush = false;
        index_flush = false;
        compressed = false;
        indexed = false;
        uses_shared_inputs = false;
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
        const bool raw_mmid = ds4_ffn_raw_mmid_enabled();
        ggml_tensor * cur_3d = ggml_reshape_3d(out.sg.ctx, ffn_normed, w.n_embd, 1, n_tokens);
        ggml_tensor * gate_e = ggml_mul_mat_id(out.sg.ctx, L.ffn_gate_exps, cur_3d, out.hash_ids);
        ggml_tensor * up_e = ggml_mul_mat_id(out.sg.ctx, L.ffn_up_exps, cur_3d, out.hash_ids);
        if (!raw_mmid) {
            gate_e = ggml_reshape_3d(out.sg.ctx, gate_e, n_ff_exp, n_used, n_tokens);
            up_e = ggml_reshape_3d(out.sg.ctx, up_e, n_ff_exp, n_used, n_tokens);
        }
        ggml_tensor * mid_e = build_clamped_swiglu(out.sg.ctx, gate_e, up_e, w.swiglu_clamp_exp);
        ggml_tensor * down_e = ggml_mul_mat_id(out.sg.ctx, L.ffn_down_exps, mid_e, out.hash_ids);
        if (!raw_mmid) {
            down_e = ggml_reshape_3d(out.sg.ctx, down_e, w.n_embd, n_used, n_tokens);
        }

        ggml_tensor * probs_3d = ggml_reshape_3d(out.sg.ctx, probs, 1, w.n_expert, n_tokens);
        ggml_tensor * weights = ggml_get_rows(out.sg.ctx, probs_3d, out.hash_ids);
        weights = ggml_reshape_2d(out.sg.ctx, weights, n_used, n_tokens);
        ggml_tensor * w_sum = ggml_sum_rows(out.sg.ctx, weights);
        w_sum = ggml_clamp(out.sg.ctx, w_sum, 6.103515625e-5f, INFINITY);
        weights = ggml_div(out.sg.ctx, weights, w_sum);
        if (w.expert_weight_scale != 1.0f) {
            weights = ggml_scale(out.sg.ctx, weights, w.expert_weight_scale);
        }

        ggml_tensor * routed_out = nullptr;
        if (ds4_ffn_fused_combine_enabled()) {
            routed_out = ggml_laguna_moe_combine(out.sg.ctx, down_e, weights);
        } else {
            ggml_tensor * weights_3d = ggml_reshape_3d(out.sg.ctx, weights, 1, n_used, n_tokens);
            routed_out = ggml_mul(out.sg.ctx, down_e, weights_3d);
            ggml_tensor * sum_shape = ggml_new_tensor_3d(out.sg.ctx, GGML_TYPE_F32, w.n_embd, 1, n_tokens);
            routed_out = ggml_repeat_back(out.sg.ctx, routed_out, sum_shape);
            routed_out = ggml_reshape_2d(out.sg.ctx, routed_out, w.n_embd, n_tokens);
        }

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
    return ggml_swiglu_ds4_split(ctx, gate, up, clamp);
}

static ggml_tensor * ds4_cast_if_needed(
        ggml_context * ctx,
        ggml_tensor * x,
        ggml_type type) {
    return x->type == type ? x : ggml_cast(ctx, x, type);
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
        std::vector<DeepSeek4I32ArrayBinding> & i32_array_inputs,
        ggml_tensor ** comp_cache_source_out = nullptr,
        ggml_tensor * flush_rows_inp = nullptr,
        ggml_tensor * cur_all = nullptr,
        int n_tokens_all = 1,
        int kv_start_all = -1) {
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
    ggml_tensor * state_kv_source = state.state_kv;
    ggml_tensor * state_score_source = state.state_score;
    ggml_tensor * comp_cache_source = comp_cache;

    // Causal-batch verify: every token's contribution lands in its
    // position-addressed state row (the rows are distinct because the batch
    // never crosses a ratio window; the boundary may only be the last token).
    const bool batched_state = (cur_all != nullptr && n_tokens_all > 1 &&
                                !state_rows_inp && kv_start_all >= 0);
    if (batched_state) {
        ggml_tensor * kv_all = ggml_mul_mat(ctx, kv_proj, cur_all);
        ggml_tensor * sc_all = ggml_mul_mat(ctx, gate_proj, cur_all);
        for (int ti = 0; ti < n_tokens_all; ti++) {
            const int pm_ti  = (kv_start_all + ti) % ratio;
            const int row_ti = (ratio == 4) ? (ratio + pm_ti) : pm_ti;
            ggml_tensor * kv_ti = ggml_view_2d(ctx, kv_all, comp_width, 1, kv_all->nb[1],
                                               (size_t) ti * kv_all->nb[1]);
            ggml_tensor * sc_ti = ggml_view_2d(ctx, sc_all, comp_width, 1, sc_all->nb[1],
                                               (size_t) ti * sc_all->nb[1]);
            ggml_tensor * ape_ti = ggml_view_2d(ctx, ape, comp_width, 1, ape->nb[1],
                                                (size_t) pm_ti * ape->nb[1]);
            sc_ti = ggml_add(ctx, sc_ti, ggml_cast(ctx, ape_ti, GGML_TYPE_F32));
            ggml_tensor * kv_slot_ti = ggml_view_2d(ctx, state.state_kv, comp_width, 1,
                                                    state.state_kv->nb[1],
                                                    (size_t) row_ti * state.state_kv->nb[1]);
            ggml_tensor * sc_slot_ti = ggml_view_2d(ctx, state.state_score, comp_width, 1,
                                                    state.state_score->nb[1],
                                                    (size_t) row_ti * state.state_score->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_cast(ctx, kv_ti, state.state_kv->type), kv_slot_ti));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, sc_ti, sc_slot_ti));
        }
    }

    const bool batched_rows = (state_rows_inp && cur_all != nullptr && n_tokens_all > 1);
    int batched_b = -1;          // boundary index within the batch (batched_rows)
    int batched_nB = 0;          // tokens after the boundary
    int batched_span_off = 0;
    ggml_tensor * batched_kv_all = nullptr;
    ggml_tensor * batched_sc_all = nullptr;
    ggml_tensor * ape_col = nullptr;
    if (!batched_rows) {
        if (ape_row_inp) {
            ape_col = ggml_get_rows(ctx, ape, ape_row_inp);
            ape_col = ggml_reshape_2d(ctx, ape_col, comp_width, 1);
        } else {
            ape_col = ggml_view_2d(
                ctx, ape, comp_width, 1, ape->nb[1], (size_t)pos_mod * ape->nb[1]);
            ape_col = ggml_cast(ctx, ape_col, GGML_TYPE_F32);
        }
        sc_cur = ggml_add(ctx, sc_cur, ape_col);
    }

    if (batched_state) {
        // state rows already written above (batched)
    } else if (batched_rows) {
        // Fused verify: batched state writes with ONE boundary allowed at ANY
        // batch index b (q <= ratio keeps every pos_mod distinct). Graph order:
        // writes[0..b] -> pool(boundary, reads through span A) -> rotate
        // cur->prev (ratio-4) -> writes[b+1..]. The pooling and rotation code
        // below read state_*_source, which span A set.
        ggml_tensor * kv_all = ggml_mul_mat(ctx, kv_proj, cur_all);
        ggml_tensor * sc_all = ggml_mul_mat(ctx, gate_proj, cur_all);
        ggml_tensor * ape_cols = ggml_get_rows(ctx, ape, ape_row_inp);   // [comp_width, q]
        sc_all = ggml_add(ctx, sc_all, ape_cols);
        for (int ti = 0; ti < n_tokens_all; ++ti) {
            if (((kv_start_all + ti + 1) % ratio) == 0) { batched_b = ti; break; }
        }
        const int nA = (batched_b >= 0) ? (batched_b + 1) : n_tokens_all;
        batched_nB = n_tokens_all - nA;
        auto write_span = [&](int off, int count, ggml_tensor ** kv_src, ggml_tensor ** sc_src) {
            if (count <= 0) return;
            ggml_tensor * kv_v = ggml_cont(ctx, ggml_view_2d(ctx, kv_all, comp_width, count,
                                           kv_all->nb[1], (size_t) off * kv_all->nb[1]));
            ggml_tensor * sc_v = ggml_cont(ctx, ggml_view_2d(ctx, sc_all, comp_width, count,
                                           sc_all->nb[1], (size_t) off * sc_all->nb[1]));
            ggml_tensor * rows_v = ggml_view_1d(ctx, state_rows_inp, count,
                                                (size_t) off * state_rows_inp->nb[0]);
            *kv_src = ggml_set_rows(ctx, state.state_kv, kv_v, rows_v);
            *sc_src = ggml_set_rows(ctx, state.state_score, sc_v, rows_v);
            ggml_build_forward_expand(gf, *kv_src);
            ggml_build_forward_expand(gf, *sc_src);
        };
        write_span(0, nA, &state_kv_source, &state_score_source);
        batched_kv_all = kv_all;
        batched_sc_all = sc_all;
        batched_span_off = nA;
    } else if (state_rows_inp) {
        state_kv_source = ggml_set_rows(ctx, state.state_kv, kv_cur, state_rows_inp);
        state_score_source = ggml_set_rows(ctx, state.state_score, sc_cur, state_rows_inp);
        ggml_build_forward_expand(gf, state_kv_source);
        ggml_build_forward_expand(gf, state_score_source);
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

    if (batched_rows && batched_b < 0) {
        // no boundary inside the batch: state rows written, nothing to emit
        return;
    }
    if (!batched_rows && !flush_rows_inp && ((token_pos + 1) % ratio) != 0) {
        // Legacy per-layer graphs only pool at flush boundaries. The fused
        // stable-topology graph (flush_rows_inp set) pools every step; the
        // partial result lands on the masked running comp row.
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
        const size_t hi_off_kv = (size_t)ratio * state_kv_source->nb[1] +
                                 (size_t)head_dim * state_kv_source->nb[0];
        const size_t hi_off_sc = (size_t)ratio * state_score_source->nb[1] +
                                 (size_t)head_dim * state_score_source->nb[0];
        ggml_tensor * prev_kv = ggml_view_2d(ctx, state_kv_source, head_dim, ratio,
                                             state_kv_source->nb[1], 0);
        ggml_tensor * cur_kv_hi = ggml_view_2d(ctx, state_kv_source, head_dim, ratio,
                                               state_kv_source->nb[1], hi_off_kv);
        ggml_tensor * prev_sc = ggml_view_2d(ctx, state_score_source, head_dim, ratio,
                                             state_score_source->nb[1], 0);
        ggml_tensor * cur_sc_hi = ggml_view_2d(ctx, state_score_source, head_dim, ratio,
                                               state_score_source->nb[1], hi_off_sc);
        sv_kv = ggml_concat(ctx, prev_kv, cur_kv_hi, 1);
        sv_sc = ggml_concat(ctx, prev_sc, cur_sc_hi, 1);
        n_state_rows = 2 * ratio;
    } else {
        sv_kv = ggml_view_2d(ctx, state_kv_source, comp_width, n_state_rows,
                             state_kv_source->nb[1], 0);
        sv_sc = ggml_view_2d(ctx, state_score_source, comp_width, n_state_rows,
                             state_score_source->nb[1], 0);
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
        comp_cache_source = ggml_set_rows(ctx, comp_cache, pooled, comp_rows_inp);
        ggml_build_forward_expand(gf, comp_cache_source);
    } else {
        ggml_tensor * comp_slot = ggml_view_2d(
            ctx, comp_cache, head_dim, 1, comp_cache->nb[1],
            (size_t)comp_row * comp_cache->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, pooled_f16, comp_slot));
    }

    if (comp_cache_source_out) {
        *comp_cache_source_out = comp_cache_source;
    }

    if (batched_rows) {
        if (ratio == 4) {
            // completed window: rotate current half -> prev half, reading
            // through the span-A writes so ordering is explicit.
            for (int r = 0; r < ratio; ++r) {
                ggml_tensor * src_kv = ggml_view_2d(ctx, state_kv_source, comp_width, 1,
                                                    state_kv_source->nb[1],
                                                    (size_t)(ratio + r) * state_kv_source->nb[1]);
                ggml_tensor * dst_kv = ggml_view_2d(ctx, state.state_kv, comp_width, 1,
                                                    state.state_kv->nb[1],
                                                    (size_t) r * state.state_kv->nb[1]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, src_kv, dst_kv));
                ggml_tensor * src_sc = ggml_view_2d(ctx, state_score_source, comp_width, 1,
                                                    state_score_source->nb[1],
                                                    (size_t)(ratio + r) * state_score_source->nb[1]);
                ggml_tensor * dst_sc = ggml_view_2d(ctx, state.state_score, comp_width, 1,
                                                    state.state_score->nb[1],
                                                    (size_t) r * state.state_score->nb[1]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, src_sc, dst_sc));
            }
        }
        if (batched_nB > 0) {
            ggml_tensor * kv_v = ggml_cont(ctx, ggml_view_2d(ctx, batched_kv_all, comp_width, batched_nB,
                                           batched_kv_all->nb[1], (size_t) batched_span_off * batched_kv_all->nb[1]));
            ggml_tensor * sc_v = ggml_cont(ctx, ggml_view_2d(ctx, batched_sc_all, comp_width, batched_nB,
                                           batched_sc_all->nb[1], (size_t) batched_span_off * batched_sc_all->nb[1]));
            ggml_tensor * rows_v = ggml_view_1d(ctx, state_rows_inp, batched_nB,
                                                (size_t) batched_span_off * state_rows_inp->nb[0]);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, state.state_kv, kv_v, rows_v));
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, state.state_score, sc_v, rows_v));
        }
        return;
    }
    if (ratio == 4 && flush_rows_inp) {
        // Stable-topology flush: copy the cur half onto rows given by the
        // input (prev half [0..3] at flush, cur half itself [4..7] = no-op
        // otherwise). Values are read through the set_rows source so this
        // step's state write is ordered first.
        ggml_tensor * cur_kv_vals = ggml_cont(ctx, ggml_view_2d(
            ctx, state_kv_source, comp_width, ratio, state_kv_source->nb[1],
            (size_t) ratio * state_kv_source->nb[1]));
        ggml_tensor * cur_sc_vals = ggml_cont(ctx, ggml_view_2d(
            ctx, state_score_source, comp_width, ratio, state_score_source->nb[1],
            (size_t) ratio * state_score_source->nb[1]));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, state.state_kv, cur_kv_vals, flush_rows_inp));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, state.state_score, cur_sc_vals, flush_rows_inp));
    } else if (ratio == 4) {
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
        std::vector<DeepSeek4I32ArrayBinding> & i32_array_inputs,
        ggml_tensor * flush_rows_inp = nullptr,
        ggml_tensor * cur_all = nullptr,
        int n_tokens_all = 1,
        int kv_start_all = -1) {
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
                          i32_array_inputs,
                          nullptr,
                          flush_rows_inp,
                          cur_all,
                          n_tokens_all,
                          kv_start_all);
}

static int ds4_comp_rows_used(const ggml_tensor * comp_cache, int n_cached, int ratio, int token_pos) {
    if (!comp_cache || ratio <= 0) {
        return 0;
    }
    const int grew_this_step = ((token_pos + 1) % ratio) == 0 ? 1 : 0;
    return std::min(n_cached + grew_this_step, (int) comp_cache->ne[1]);
}

// Round the live compressed-row count up to a fixed stride so the fused decode
// graph topology repeats across steps (enabling CUDA/HIP graph replay). The
// rows in [n_comp, padded) are masked to -1e30 in the score matrix, which
// underflows to exactly 0 in softmax, so a padded read is bit-identical to an
// unpadded read of the first n_comp rows.
static constexpr int DS4_COMP_PAD_STRIDE = 16;
static int ds4_padded_comp_rows(int n_comp, int cap) {
    if (n_comp <= 0) return 0;
    const int padded = ((n_comp + DS4_COMP_PAD_STRIDE - 1) / DS4_COMP_PAD_STRIDE) * DS4_COMP_PAD_STRIDE;
    return padded < cap ? padded : cap;
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
        std::vector<DeepSeek4I64ArrayBinding> & i64_array_inputs,
        std::vector<DeepSeek4F32ArrayBinding> * f32_array_inputs = nullptr) {

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

    // ── Causal batched step (exact multi-token target semantics) ───
    // The target model is causal: token i must not attend to batch tokens
    // j > i, must see the compressed-row count as of its own position, and —
    // once the ring has wrapped — must still see the OLD contents of ring
    // slots that later batch tokens overwrite. Default ON for multi-token
    // steps on this path; DFLASH_DS4_NO_CAUSAL_VERIFY=1 restores the legacy
    // (bidirectional) behavior for A/B comparison.
    const bool causal_batch = (n_tokens > 1) && !cached_inputs && f32_array_inputs &&
                              !ds4_env_flag("DFLASH_DS4_NO_CAUSAL_VERIFY");
    ggml_tensor * old_rows_scratch = nullptr;
    int n_old_rows = 0;
    const bool fused_causal = cached_inputs && cached_inputs->attn_row_mask && n_tokens > 1;
    if (fused_causal) {
        // Fused verify: ALWAYS q preserved rows so the topology is stable;
        // unwrapped/garbage rows are masked by the host-filled mask values.
        for (int ti = 0; ti < n_tokens; ti++) {
            ggml_tensor * slot = ggml_view_2d(
                ctx, lc.raw_kv, head_dim, 1, lc.raw_kv->nb[1],
                (size_t)((kv_start + ti) % w.n_swa) * lc.raw_kv->nb[1]);
            ggml_tensor * saved = ggml_cont(ctx, slot);
            ggml_build_forward_expand(gf, saved);
            old_rows_scratch = old_rows_scratch
                ? ggml_concat(ctx, old_rows_scratch, saved, 1) : saved;
            n_old_rows++;
        }
        old_rows_scratch = ds4_cast_if_needed(ctx, old_rows_scratch, GGML_TYPE_F32);
    } else if (causal_batch) {
        // Copy the to-be-overwritten rows FIRST; same-stream build order runs
        // these before the ring writes below.
        for (int ti = 0; ti < n_tokens; ti++) {
            if (kv_start + ti < w.n_swa) continue;   // slot never held an older pos
            ggml_tensor * slot = ggml_view_2d(
                ctx, lc.raw_kv, head_dim, 1, lc.raw_kv->nb[1],
                (size_t)((kv_start + ti) % w.n_swa) * lc.raw_kv->nb[1]);
            ggml_tensor * saved = ggml_cont(ctx, slot);
            ggml_build_forward_expand(gf, saved);
            old_rows_scratch = old_rows_scratch
                ? ggml_concat(ctx, old_rows_scratch, saved, 1) : saved;
            n_old_rows++;
        }
        if (old_rows_scratch) {
            old_rows_scratch = ds4_cast_if_needed(ctx, old_rows_scratch, GGML_TYPE_F32);
        }
    }

    // ── Store ALL KV rows in the raw SWA ring ─────────────────────
    // For decode (n_tokens=1): write single row. For prefill: write all rows.
    ggml_tensor * raw_kv_source = lc.raw_kv;
    if (cached_inputs && cached_inputs->raw_kv_rows) {
        ggml_tensor * kv_f32 = ggml_is_contiguous(kv) ? kv : ggml_cont(ctx, kv);
        raw_kv_source = ggml_set_rows(ctx, lc.raw_kv, kv_f32, cached_inputs->raw_kv_rows);
        ggml_build_forward_expand(gf, raw_kv_source);
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
    ggml_tensor * comp_kv_source = lc.comp_kv;
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
                              i32_array_inputs,
                              &comp_kv_source,
                              cached_inputs ? cached_inputs->flush_rows : nullptr,
                              (causal_batch || fused_causal) ? cur : nullptr,
                              n_tokens,
                              kv_start);
    }

    if (ratio == 4 && L.indexer_compressor_kv) {
        build_indexer_compressor_step(ctx, gf, cur_last, w, L, lc, token_pos,
                                      cached_inputs ? cached_inputs->index_ape_row : nullptr,
                                      cached_inputs ? cached_inputs->index_state_rows : nullptr,
                                      cached_inputs ? cached_inputs->index_comp_rows : nullptr,
                                      cached_inputs ? cached_inputs->index_comp_pos : nullptr,
                                      i64_array_inputs,
                                      i32_array_inputs,
                                      cached_inputs ? cached_inputs->flush_rows : nullptr,
                                      (causal_batch || fused_causal) ? cur : nullptr,
                                      n_tokens,
                                      kv_start);
        (void)build_indexer_score(ctx, qr_last, cur_last, w, L, lc, token_pos, i32_inputs);
    }

    // ── MLA Dot-Product Attention (SWA + compressed KV) ────────────
    // q: [head_dim, n_head, n_tokens] (after RoPE)
    // raw_kv: [head_dim, n_swa] F16 persistent ring buffer (single KV head, shared)
    // comp_kv: [head_dim, comp_cap] F16 compressed rows.
    // n_raw = min(kv_start + n_tokens, n_swa)
    const bool masked_kv = cached_inputs && cached_inputs->attn_row_mask;
    const int n_comp_live = (ratio > 0) ? ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio, token_pos) : 0;
    // Stable path reads the full physical ring (masking not-yet-written slots)
    // and a padded compressed-row span; the plain path reads only valid rows.
    const int n_raw = masked_kv ? w.n_swa : std::min(kv_start + n_tokens, w.n_swa);
    const int n_comp_attn = masked_kv ? cached_inputs->padded_comp : n_comp_live;
    const int n_valid_raw = std::min(kv_start + n_tokens, w.n_swa);
    const int n_attn = n_raw + n_comp_attn + n_old_rows;
    const float kq_scale = 1.0f / sqrtf((float)head_dim);

    // Get valid KV rows. For single-token decode, include the current in-graph
    // KV row directly; otherwise attention can race the side-effecting cache
    // write and see the previous contents of the raw KV slot.
    auto raw_kv_view = [&](int row, int count) -> ggml_tensor * {
        ggml_tensor * view = ggml_view_2d(
            ctx, lc.raw_kv, head_dim, count, lc.raw_kv->nb[1],
            (size_t)row * lc.raw_kv->nb[1]);
        return ds4_cast_if_needed(ctx, view, GGML_TYPE_F32);
    };

    ggml_tensor * kv_attn = nullptr;
    if (masked_kv) {
        // Fused stable-KV path: read the full physical ring; rows not yet
        // written are masked to -1e30 in the score matrix (exact 0 after
        // softmax). Only the fused decode graph sets attn_row_mask. Read
        // through the set_rows result so the current row's in-graph write is
        // ordered before this read.
        ggml_tensor * ring = ggml_view_2d(
            ctx, raw_kv_source, head_dim, w.n_swa, raw_kv_source->nb[1], 0);
        kv_attn = ds4_cast_if_needed(ctx, ring, GGML_TYPE_F32);
    } else if (n_tokens == 1) {
        ggml_tensor * cur_kv = ds4_cast_if_needed(ctx, kv, GGML_TYPE_F32);
        if (n_raw > 1) {
            ggml_tensor * prev = nullptr;
            if (kv_start < w.n_swa) {
                prev = raw_kv_view(0, n_raw - 1);
            } else {
                const int raw_row = kv_start % w.n_swa;
                const int tail_count = w.n_swa - raw_row - 1;
                const int head_count = raw_row;
                if (tail_count > 0) {
                    prev = raw_kv_view(raw_row + 1, tail_count);
                }
                if (head_count > 0) {
                    ggml_tensor * head = raw_kv_view(0, head_count);
                    prev = prev ? ggml_concat(ctx, prev, head, 1) : head;
                }
            }
            kv_attn = prev ? ggml_concat(ctx, prev, cur_kv, 1) : cur_kv;
        } else {
            kv_attn = cur_kv;
        }
    } else {
        kv_attn = raw_kv_view(0, n_raw);
    }
    if (n_comp_attn > 0 && comp_kv_source) {
        ggml_tensor * comp = ggml_view_2d(ctx, comp_kv_source, head_dim, n_comp_attn, comp_kv_source->nb[1], 0);
        comp = ds4_cast_if_needed(ctx, comp, GGML_TYPE_F32);
        kv_attn = ggml_concat(ctx, kv_attn, comp, 1);
    }
    if (old_rows_scratch) {
        kv_attn = ggml_concat(ctx, kv_attn, old_rows_scratch, 1);
    }
    // kv_attn: [head_dim, n_attn]

    // Flatten q to [head_dim, n_head*n_tokens] for batched matmul
    ggml_tensor * q_flat = ggml_reshape_2d(ctx, q, head_dim, n_head * n_tokens);

    // Scores: mul_mat(kv_attn, q_flat) = kv_attn^T[n_attn, head_dim] @ q_flat[head_dim, n_head*n_tokens]
    //       → [n_attn, n_head*n_tokens]
    ggml_tensor * scores = ggml_mul_mat(ctx, kv_attn, q_flat);
    scores = ggml_scale(ctx, scores, kq_scale);
    if (masked_kv && n_tokens > 1) {
        // Per-token causal mask [n_attn, n_tokens] from the host-filled bundle.
        ggml_tensor * m3 = ggml_reshape_3d(ctx, cached_inputs->attn_row_mask, n_attn, 1, n_tokens);
        ggml_tensor * scores3d = ggml_reshape_3d(ctx, scores, n_attn, n_head, n_tokens);
        scores3d = ggml_add(ctx, scores3d, m3);
        scores = ggml_reshape_2d(ctx, scores3d, n_attn, n_head * n_tokens);
    } else if (masked_kv) {
        // Broadcast-add the [n_attn,1] additive mask across all query columns.
        scores = ggml_add(ctx, scores, cached_inputs->attn_row_mask);
    } else if (causal_batch) {
        // Per-token causal mask over [ring rows | comp rows | old rows].
        ggml_tensor * cmask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_attn, 1, n_tokens);
        ggml_set_input(cmask);
        std::vector<float> mvals((size_t) n_attn * n_tokens, 0.0f);
        const int e = kv_start + n_tokens;            // exclusive end position
        for (int i = 0; i < n_tokens; i++) {
            const int pos_i = kv_start + i;
            float * col = mvals.data() + (size_t) i * n_attn;
            for (int r = 0; r < n_raw; r++) {
                // position held by ring slot r AFTER this batch's writes
                const int p_r = (e <= w.n_swa) ? r
                              : (e - 1) - ((e - 1 - r) % w.n_swa);
                if (p_r > pos_i) col[r] = -1e30f;
            }
            if (n_comp_attn > 0) {
                const int vis = ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio, pos_i);
                for (int c = vis; c < n_comp_attn; c++) col[n_raw + c] = -1e30f;
            }
            // old contents of slot overwritten by batch token j are visible
            // exactly to tokens i < j (still inside their SWA window)
            int oi = 0;
            for (int tj = 0; tj < n_tokens; tj++) {
                if (kv_start + tj < w.n_swa) continue;
                if (tj <= i) col[n_raw + n_comp_attn + oi] = -1e30f;
                oi++;
            }
        }
        f32_array_inputs->push_back({cmask, std::move(mvals)});
        ggml_tensor * scores3d = ggml_reshape_3d(ctx, scores, n_attn, n_head, n_tokens);
        scores3d = ggml_add(ctx, scores3d, cmask);
        scores = ggml_reshape_2d(ctx, scores3d, n_attn, n_head * n_tokens);
    }
    (void) n_valid_raw;

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
    // i.e. mul_mat(kv_T, probs) where kv_T = cont(transpose(kv_attn)) = [n_raw, head_dim]
    ggml_tensor * kv_T = ggml_cont(ctx, ggml_transpose(ctx, kv_attn));
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

// Per-step decode scalar inputs shared by all cached per-layer decode graphs.
// Values depend only on (kv_start, ratio), so one tensor per slot serves every
// layer with that ratio. i32 layout per ratio-slot: {rope_pos, neg_pos,
// ape_row, comp_pos, index_ape_row, index_comp_pos}; i64 layout: {raw_kv_row,
// state_row, comp_row, index_state_row, index_comp_row}.
struct Ds4DecodeSharedInputs {
    static constexpr int MAX_RATIOS = 4;
    const ggml_context * owner_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    int ratios[MAX_RATIOS] = {0};
    int n_ratios = 0;
    ggml_tensor * i32_bundle = nullptr;   // [6 * n_ratios]
    ggml_tensor * i64_bundle = nullptr;   // [5 * n_ratios]
    // Per-slot views handed to the graph builders.
    ggml_tensor * v_rope_pos[MAX_RATIOS] = {nullptr};
    ggml_tensor * v_neg_pos[MAX_RATIOS] = {nullptr};
    ggml_tensor * v_ape_row[MAX_RATIOS] = {nullptr};
    ggml_tensor * v_comp_pos[MAX_RATIOS] = {nullptr};
    ggml_tensor * v_index_ape[MAX_RATIOS] = {nullptr};
    ggml_tensor * v_index_cpos[MAX_RATIOS] = {nullptr};
    ggml_tensor * v_raw_row[MAX_RATIOS] = {nullptr};
    ggml_tensor * v_state_row[MAX_RATIOS] = {nullptr};
    ggml_tensor * v_comp_row[MAX_RATIOS] = {nullptr};
    ggml_tensor * v_index_state[MAX_RATIOS] = {nullptr};
    ggml_tensor * v_index_comp[MAX_RATIOS] = {nullptr};

    void free() {
        if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
        owner_ctx = nullptr; backend = nullptr; n_ratios = 0;
        i32_bundle = nullptr; i64_bundle = nullptr;
    }

    int slot(int ratio) const {
        for (int i = 0; i < n_ratios; ++i) if (ratios[i] == ratio) return i;
        return -1;
    }

    bool ensure(const DeepSeek4Weights & w, ggml_backend_t bk) {
        if (ctx && owner_ctx == w.ctx && backend == bk) return true;
        free();
        n_ratios = 0;
        for (int il = 0; il < w.n_layer; ++il) {
            const int r = (int) w.compress_ratios[il];
            if (slot(r) >= 0) continue;
            if (n_ratios >= MAX_RATIOS) return false;
            ratios[n_ratios++] = r;
        }
        ggml_init_params p{};
        p.mem_size = ggml_tensor_overhead() * (size_t) (2 + 11 * MAX_RATIOS) + 4096;
        p.no_alloc = true;
        ctx = ggml_init(p);
        if (!ctx) return false;
        i32_bundle = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 6 * (int64_t) n_ratios);
        i64_bundle = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 5 * (int64_t) n_ratios);
        for (int s = 0; s < n_ratios; ++s) {
            v_rope_pos[s]   = ggml_view_1d(ctx, i32_bundle, 1, ((size_t) s * 6 + 0) * sizeof(int32_t));
            v_neg_pos[s]    = ggml_view_1d(ctx, i32_bundle, 1, ((size_t) s * 6 + 1) * sizeof(int32_t));
            v_ape_row[s]    = ggml_view_1d(ctx, i32_bundle, 1, ((size_t) s * 6 + 2) * sizeof(int32_t));
            v_comp_pos[s]   = ggml_view_1d(ctx, i32_bundle, 1, ((size_t) s * 6 + 3) * sizeof(int32_t));
            v_index_ape[s]  = ggml_view_1d(ctx, i32_bundle, 1, ((size_t) s * 6 + 4) * sizeof(int32_t));
            v_index_cpos[s] = ggml_view_1d(ctx, i32_bundle, 1, ((size_t) s * 6 + 5) * sizeof(int32_t));
            v_raw_row[s]     = ggml_view_2d(ctx, i64_bundle, 1, 1, sizeof(int64_t), ((size_t) s * 5 + 0) * sizeof(int64_t));
            v_state_row[s]   = ggml_view_2d(ctx, i64_bundle, 1, 1, sizeof(int64_t), ((size_t) s * 5 + 1) * sizeof(int64_t));
            v_comp_row[s]    = ggml_view_2d(ctx, i64_bundle, 1, 1, sizeof(int64_t), ((size_t) s * 5 + 2) * sizeof(int64_t));
            v_index_state[s] = ggml_view_2d(ctx, i64_bundle, 1, 1, sizeof(int64_t), ((size_t) s * 5 + 3) * sizeof(int64_t));
            v_index_comp[s]  = ggml_view_2d(ctx, i64_bundle, 1, 1, sizeof(int64_t), ((size_t) s * 5 + 4) * sizeof(int64_t));
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, bk);
        if (!buf) { free(); return false; }
        owner_ctx = w.ctx;
        backend = bk;
        return true;
    }

    // Upload all per-step values in two writes.
    void set_step(const DeepSeek4Weights & w, int kv_start) {
        const int token_pos = kv_start;
        int32_t i32v[6 * MAX_RATIOS] = {0};
        int64_t i64v[5 * MAX_RATIOS] = {0};
        for (int s = 0; s < n_ratios; ++s) {
            const int ratio = ratios[s];
            i32v[s * 6 + 0] = kv_start;
            i32v[s * 6 + 1] = -kv_start;
            i64v[s * 5 + 0] = kv_start % w.n_swa;
            if (ratio > 0) {
                const int pos_mod = token_pos % ratio;
                i32v[s * 6 + 2] = pos_mod;
                i32v[s * 6 + 3] = token_pos + 1 - ratio;
                i64v[s * 5 + 1] = (ratio == 4) ? (int64_t) (ratio + pos_mod) : (int64_t) pos_mod;
                i64v[s * 5 + 2] = token_pos / ratio;
            }
            if (ratio == 4) {
                const int pos_mod = token_pos % ratio;
                i32v[s * 6 + 4] = pos_mod;
                i32v[s * 6 + 5] = token_pos + 1 - ratio;
                i64v[s * 5 + 3] = ratio + pos_mod;
                i64v[s * 5 + 4] = token_pos / ratio;
            }
        }
        ggml_backend_tensor_set(i32_bundle, i32v, 0, sizeof(int32_t) * 6 * (size_t) n_ratios);
        ggml_backend_tensor_set(i64_bundle, i64v, 0, sizeof(int64_t) * 5 * (size_t) n_ratios);
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
        int index_comp_count,
        const Ds4DecodeSharedInputs * shared = nullptr) {
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
    out.attn_flush = ratio > 0 && (((kv_start + 1) % ratio) == 0);
    out.index_flush = ratio == 4 && (((kv_start + 1) % ratio) == 0);
    out.compressed = ratio > 0;
    out.indexed = ratio == 4;

    out.sg.inp_embed = ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_F32, w.n_embd, 1);
    ggml_set_input(out.sg.inp_embed);
    out.sg.gf = ggml_new_graph_custom(out.sg.ctx, 2048, false);

    const int shared_slot = shared ? shared->slot(ratio) : -1;
    if (shared_slot >= 0) {
        out.inputs.rope_pos = shared->v_rope_pos[shared_slot];
        out.inputs.neg_pos = shared->v_neg_pos[shared_slot];
        out.inputs.raw_kv_rows = shared->v_raw_row[shared_slot];
        if (ratio > 0) {
            out.inputs.attn_ape_row = shared->v_ape_row[shared_slot];
            out.inputs.attn_comp_pos = shared->v_comp_pos[shared_slot];
            out.inputs.attn_state_rows = shared->v_state_row[shared_slot];
            out.inputs.attn_comp_rows = shared->v_comp_row[shared_slot];
        }
        if (ratio == 4) {
            out.inputs.index_ape_row = shared->v_index_ape[shared_slot];
            out.inputs.index_comp_pos = shared->v_index_cpos[shared_slot];
            out.inputs.index_state_rows = shared->v_index_state[shared_slot];
            out.inputs.index_comp_rows = shared->v_index_comp[shared_slot];
        }
        out.uses_shared_inputs = true;
    } else {
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
                                      const void * fn_device_override,
                                      ggml_tensor * scale_tensor,
                                      ggml_tensor * base_tensor,
                                      const float * scale_data,
                                      const float * base_data,
                                      int n_embd,
                                      int n_hc,
                                      int sinkhorn_iters,
                                      float hc_eps) {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    const void * fn_device = fn_device_override ? fn_device_override : (fn_tensor ? fn_tensor->data : nullptr);
    if (!working || !post || !comb || !hc_state || !fn_device || !scale_data || !base_data ||
        !working->data || !post->data || !comb->data || !hc_state->data) {
        return false;
    }
    const bool can_use_device_params =
        ds4_backend_is_gpu(backend) &&
        scale_tensor && base_tensor &&
        scale_tensor->data && base_tensor->data &&
        scale_tensor->buffer && base_tensor->buffer &&
        !ggml_backend_buffer_is_host(scale_tensor->buffer) &&
        !ggml_backend_buffer_is_host(base_tensor->buffer);
    if (can_use_device_params) {
        return deepseek4_cuda_hc_pre_device(hc_state->data,
                                            fn_device,
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
                                               fn_device,
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
    (void) fn_device_override;
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

// Opt-in serving knobs for the routed-expert FFN, both default-off:
//   DFLASH_DS4_TOPK=<k>            keep only the leading k of the top-k routed
//                                  experts (weights renormalized over the kept
//                                  set). Bandwidth lever for decode; measured
//                                  +9% AR decode at k=4 on Strix (25.1 vs 23.0
//                                  tok/s), small quality cost — validate per
//                                  deployment.
//   DFLASH_DS4_NEURON_MEANMASK=<m> zero routed-expert activations below
//                                  m * mean(|activation|) per expert and token
//                                  (m=1.0 keeps ~45%). Quality probe for
//                                  intra-expert sparsity; adds a few ops and
//                                  saves no bytes until a sparse kernel lands.
static int ds4_topk_override() {
    static const int k = [] {
        const char * e = std::getenv("DFLASH_DS4_TOPK");
        return e ? std::atoi(e) : 0;
    }();
    return k;
}

static float ds4_neuron_mask_mult() {
    static const float m = [] {
        const char * e = std::getenv("DFLASH_DS4_NEURON_MEANMASK");
        return e ? (float) std::atof(e) : 0.0f;
    }();
    return m;
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

    int k_used = w.n_expert_used;
    const int k_env = ds4_topk_override();
    if (k_env > 0 && k_env < k_used) {
        k_used = k_env;
    }
    out.selected = ggml_top_k(ctx, selection, k_used);
    ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, w.n_expert, n_tokens);
    out.weights = ggml_get_rows(ctx, probs_3d, out.selected);
    out.weights = ggml_reshape_2d(ctx, out.weights, k_used, n_tokens);

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
    int n_used = w.n_expert_used;
    const int n_ff_exp = w.n_ff_exp;
    const bool raw_mmid = ds4_ffn_raw_mmid_enabled();
    ggml_tensor * shared_out = build_shared_ffn(ctx, cur, w, L);
    ggml_tensor * routed_out = nullptr;

    if (layer_idx < w.n_hash_layer && L.ffn_gate_tid2eid) {
        routed_out = ggml_scale(ctx, cur, 0.0f);
    } else {
        Ds4MoeRouting routing = build_moe_routing(ctx, cur, w, L, n_tokens);
        n_used = (int) routing.selected->ne[0];   // reduced when DFLASH_DS4_TOPK is set
        ggml_tensor * cur_3d = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
        ggml_tensor * gate_e = ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur_3d, routing.selected);
        ggml_tensor * up_e = ggml_mul_mat_id(ctx, L.ffn_up_exps, cur_3d, routing.selected);

        if (!raw_mmid) {
            gate_e = ggml_reshape_3d(ctx, gate_e, n_ff_exp, n_used, n_tokens);
            up_e = ggml_reshape_3d(ctx, up_e, n_ff_exp, n_used, n_tokens);
        }
        ggml_tensor * mid_e = build_clamped_swiglu(ctx, gate_e, up_e, w.swiglu_clamp_exp);

        const float nmask_mult = ds4_neuron_mask_mult();
        if (nmask_mult > 0.0f) {
            // Zero activations below mult * mean(|mid|) per (expert, token).
            ggml_tensor * amid = ggml_abs(ctx, mid_e);
            ggml_tensor * thr = ggml_scale(ctx, ggml_sum_rows(ctx, amid),
                                           nmask_mult / (float) n_ff_exp);
            ggml_tensor * mask = ggml_step(ctx, ggml_sub(ctx, amid, thr));
            mid_e = ggml_mul(ctx, mid_e, mask);
        }

        ggml_tensor * down_e = ggml_mul_mat_id(ctx, L.ffn_down_exps, mid_e, routing.selected);
        if (!raw_mmid) {
            down_e = ggml_reshape_3d(ctx, down_e, n_embd, n_used, n_tokens);
        }

        if (ds4_ffn_fused_combine_enabled()) {
            routed_out = ggml_laguna_moe_combine(ctx, down_e, routing.weights);
        } else {
            ggml_tensor * weights_3d = ggml_reshape_3d(ctx, routing.weights, 1, n_used, n_tokens);
            routed_out = ggml_mul(ctx, down_e, weights_3d);
            ggml_tensor * sum_shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
            routed_out = ggml_repeat_back(ctx, routed_out, sum_shape);
            routed_out = ggml_reshape_2d(ctx, routed_out, n_embd, n_tokens);
        }
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
    void * fn_f16_device = nullptr;  // Persistent F16 mirror for quantized HC fn tensors.
    size_t fn_f16_device_bytes = 0;
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

static float cpu_dot_f16_row_scalar(const uint16_t * row, const float * x, int cols) {
    float acc = 0.0f;
    for (int c = 0; c < cols; c++) {
        acc += ggml_fp16_to_fp32(row[c]) * x[c];
    }
    return acc;
}

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
static bool ds4_cpu_has_f16c() {
    static int supported = -1;
    if (supported < 0) {
        __builtin_cpu_init();
        supported = (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c")) ? 1 : 0;
    }
    return supported == 1;
}

__attribute__((target("avx2,f16c")))
static float cpu_dot_f16_row_f16c(const uint16_t * row, const float * x, int cols) {
    float acc = 0.0f;
    int c = 0;
    alignas(32) float prod[8];
    for (; c + 7 < cols; c += 8) {
        const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row + c));
        const __m256 wf = _mm256_cvtph_ps(h);
        const __m256 xf = _mm256_loadu_ps(x + c);
        _mm256_store_ps(prod, _mm256_mul_ps(wf, xf));
        acc += prod[0];
        acc += prod[1];
        acc += prod[2];
        acc += prod[3];
        acc += prod[4];
        acc += prod[5];
        acc += prod[6];
        acc += prod[7];
    }
    for (; c < cols; ++c) {
        acc += ggml_fp16_to_fp32(row[c]) * x[c];
    }
    return acc;
}
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,f16c")))
static void cpu_dot_f16_rows3_f16c(const uint16_t * r0, const uint16_t * r1, const uint16_t * r2,
                                   const float * x, int cols,
                                   float * o0, float * o1, float * o2) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    int c = 0;
    alignas(32) float p0[8], p1[8], p2[8];
    for (; c + 7 < cols; c += 8) {
        const __m256 xf = _mm256_loadu_ps(x + c);
        _mm256_store_ps(p0, _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(r0 + c))), xf));
        _mm256_store_ps(p1, _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(r1 + c))), xf));
        _mm256_store_ps(p2, _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(r2 + c))), xf));
        a0 += p0[0]; a1 += p1[0]; a2 += p2[0];
        a0 += p0[1]; a1 += p1[1]; a2 += p2[1];
        a0 += p0[2]; a1 += p1[2]; a2 += p2[2];
        a0 += p0[3]; a1 += p1[3]; a2 += p2[3];
        a0 += p0[4]; a1 += p1[4]; a2 += p2[4];
        a0 += p0[5]; a1 += p1[5]; a2 += p2[5];
        a0 += p0[6]; a1 += p1[6]; a2 += p2[6];
        a0 += p0[7]; a1 += p1[7]; a2 += p2[7];
    }
    for (; c < cols; ++c) {
        a0 += ggml_fp16_to_fp32(r0[c]) * x[c];
        a1 += ggml_fp16_to_fp32(r1[c]) * x[c];
        a2 += ggml_fp16_to_fp32(r2[c]) * x[c];
    }
    *o0 = a0; *o1 = a1; *o2 = a2;
}
#endif

static float cpu_dot_f16_row(const uint16_t * row, const float * x, int cols) {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    if (ds4_cpu_has_f16c()) {
        return cpu_dot_f16_row_f16c(row, x, cols);
    }
#endif
    return cpu_dot_f16_row_scalar(row, x, cols);
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

// Persistent worker pool for the decode-path HC fn matvec. Splitting rows
// across threads leaves each row's accumulation order untouched, so results
// are bit-identical to the serial path; only wall time changes. Decode issues
// ~86 of these 24x16384 matvecs per token, so thread spawn (or condvar wake)
// per call would dominate — workers spin briefly then yield between jobs.
struct Ds4HcMatvecPool {
    struct Job { const uint16_t * mat; const float * x; float * out; int rows; int cols; };
    std::mutex client_mu;
    std::atomic<uint64_t> seq{0};
    std::atomic<int> remaining{0};
    Job job{};
    std::vector<std::thread> workers;
    std::atomic<bool> stop{false};
    int nth = 0;

    static void cpu_relax() {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
        __builtin_ia32_pause();
#endif
    }

    Ds4HcMatvecPool() {
        unsigned hw = std::thread::hardware_concurrency();
        nth = hw == 0 ? 4 : (int)(hw < 8 ? hw : 8);
        for (int i = 0; i < nth; ++i) {
            workers.emplace_back([this, i]() {
                uint64_t last = 0;
                for (;;) {
                    uint64_t s;
                    int spins = 0;
                    while ((s = seq.load(std::memory_order_acquire)) == last) {
                        if (stop.load(std::memory_order_relaxed)) return;
                        if (++spins < 4096) { cpu_relax(); }
                        else { std::this_thread::yield(); spins = 0; }
                    }
                    last = s;
                    const Job j = job;
                    const int chunk = (j.rows + nth - 1) / nth;
                    const int r0 = i * chunk;
                    const int r1 = j.rows < r0 + chunk ? j.rows : r0 + chunk;
                    if (row_fn) {
                        for (int r = r0; r < r1; ++r) row_fn(r);
                    } else {
                        int r = r0;
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
                        if (ds4_cpu_has_f16c()) {
                            for (; r + 2 < r1; r += 3) {
                                cpu_dot_f16_rows3_f16c(
                                    j.mat + (size_t) (r + 0) * j.cols,
                                    j.mat + (size_t) (r + 1) * j.cols,
                                    j.mat + (size_t) (r + 2) * j.cols,
                                    j.x, j.cols,
                                    &j.out[r + 0], &j.out[r + 1], &j.out[r + 2]);
                            }
                        }
#endif
                        for (; r < r1; ++r) {
                            j.out[r] = cpu_dot_f16_row(j.mat + (size_t) r * j.cols, j.x, j.cols);
                        }
                    }
                    remaining.fetch_sub(1, std::memory_order_acq_rel);
                }
            });
        }
    }
    ~Ds4HcMatvecPool() {
        stop.store(true);
        for (auto & t : workers) t.join();
    }
    void run(const uint16_t * mat, const float * x, float * out, int rows, int cols) {
        std::lock_guard<std::mutex> lk(client_mu);
        row_fn = nullptr;
        job = {mat, x, out, rows, cols};
        remaining.store(nth, std::memory_order_release);
        seq.fetch_add(1, std::memory_order_release);
        int spins = 0;
        while (remaining.load(std::memory_order_acquire) != 0) {
            if (++spins < 65536) { cpu_relax(); }
            else { std::this_thread::yield(); spins = 0; }
        }
    }

    // Generic variant: invoke fn(row) for each row in [0, rows), rows split
    // across workers with the same static chunking as run().
    std::function<void(int)> row_fn;
    void run_custom(int rows, std::function<void(int)> fn) {
        std::lock_guard<std::mutex> lk(client_mu);
        row_fn = std::move(fn);
        job = {nullptr, nullptr, nullptr, rows, 0};
        remaining.store(nth, std::memory_order_release);
        seq.fetch_add(1, std::memory_order_release);
        int spins = 0;
        while (remaining.load(std::memory_order_acquire) != 0) {
            if (++spins < 65536) { cpu_relax(); }
            else { std::this_thread::yield(); spins = 0; }
        }
        row_fn = nullptr;
    }
};

static void cpu_matvec_f16_pooled(float * out, const uint16_t * mat, const float * x, int rows, int cols) {
    static Ds4HcMatvecPool pool;
    pool.run(mat, x, out, rows, cols);
}

// Token-level persistent-pool parallel-for: same splitting semantics as
// ds4_parallel_for_tokens but without per-call thread spawns (a multi-token
// step issues ~86 batched-HC calls, so spawn cost dominates at small n).
// Inner work must stay serial (serial_fn=true paths) - the pool is not
// reentrant.
static void ds4_pool_for_tokens(int n_tokens, const std::function<void(int,int)> & fn) {
    if (n_tokens <= 1) { fn(0, n_tokens); return; }
    static Ds4HcMatvecPool token_pool;
    token_pool.run_custom(n_tokens, [&fn](int t) { fn(t, t + 1); });
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
        cpu_matvec_f16_pooled(mix, fn_data, flat, mix_dim, hc_dim);
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
                             float * mix_scratch,
                             bool serial_fn) {
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
                    n_embd, n_hc, sinkhorn_iters, hc_eps, flat, mix_scratch, serial_fn);
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

    ds4_pool_for_tokens(n_tokens, [&](int t0, int t1) {
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
                             mix,
                             /*serial_fn=*/n_tokens > 1);
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
    if (n_tokens == 1) {
        // Decode: split the n_hc independent destination streams across the
        // persistent pool. Per-element accumulation order is unchanged, so
        // the result is bit-identical to the serial loop.
        static Ds4HcMatvecPool post_pool;
        struct Ctx { const float * block; const float * res; const float * post; const float * comb; float * out; int n_embd; int n_hc; };
        Ctx c{block_out, residual_hc, post, comb, out_hc.data(), n_embd, n_hc};
        post_pool.run_custom(n_hc, [&c](int h) {
            for (int d = 0; d < c.n_embd; ++d) {
                float acc = c.block[d] * c.post[h];
                for (int src = 0; src < c.n_hc; ++src) {
                    acc += c.comb[h + src * c.n_hc] * c.res[(size_t)src * c.n_embd + d];
                }
                c.out[(size_t)h * c.n_embd + d] = acc;
            }
        });
        return;
    }
    ds4_pool_for_tokens(n_tokens, [&](int t0, int t1) {
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
    ds4_pool_for_tokens(n_tokens, [&](int t0, int t1) {
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

static bool load_tensor_to_f32_cpu(std::vector<float> & dst, ggml_tensor * t) {
    if (!t) return false;

    const size_t elems = ggml_nelements(t);
    dst.resize(elems);
    if (elems == 0) return true;

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, dst.data(), 0, ggml_nbytes(t));
        return true;
    }

    const ggml_type_traits * tr = ggml_get_type_traits(t->type);
    if (!tr || !tr->to_float || t->ne[0] <= 0) return false;

    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());

    const int64_t cols = t->ne[0];
    const int64_t rows = (int64_t)elems / cols;
    const size_t row_bytes = ggml_row_size(t->type, cols);
    for (int64_t r = 0; r < rows; ++r) {
        tr->to_float(raw.data() + (size_t)r * row_bytes,
                     dst.data() + (size_t)r * (size_t)cols,
                     cols);
    }
    return true;
}

static bool load_tensor_to_f16_cpu(std::vector<uint16_t> & dst, ggml_tensor * t) {
    if (!t) return false;

    const size_t elems = ggml_nelements(t);
    dst.resize(elems);
    if (elems == 0) return true;

    if (t->type == GGML_TYPE_F16) {
        ggml_backend_tensor_get(t, dst.data(), 0, ggml_nbytes(t));
        return true;
    }

    std::vector<float> f32;
    if (!load_tensor_to_f32_cpu(f32, t)) return false;
    ggml_fp32_to_fp16_row(f32.data(), reinterpret_cast<ggml_fp16_t *>(dst.data()), (int64_t)elems);
    return true;
}

static void load_hc_weights_cpu(HcWeightsCpu & dst, ggml_tensor * fn,
                                 ggml_tensor * scale, ggml_tensor * base) {
    if (!fn || !scale || !base || dst.loaded) return;
    if (!load_tensor_to_f16_cpu(dst.fn_data, fn) ||
        !load_tensor_to_f32_cpu(dst.scale_data, scale) ||
        !load_tensor_to_f32_cpu(dst.base_data, base)) {
        dst.fn_data.clear();
        dst.scale_data.clear();
        dst.base_data.clear();
        return;
    }
    dst.loaded = true;
}

static void release_hc_fn_device(HcWeightsCpu & w) {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    if (w.fn_f16_device) {
        deepseek4_cuda_hc_free(w.fn_f16_device);
    }
#endif
    w.fn_f16_device = nullptr;
    w.fn_f16_device_bytes = 0;
}

static void reset_hc_weights_cpu(HcWeightsCpu & w) {
    release_hc_fn_device(w);
    w.fn_data.clear();
    w.scale_data.clear();
    w.base_data.clear();
    w.loaded = false;
}

static void reset_hc_layer_weights_cpu(std::vector<HcLayerWeightsCpu> & weights) {
    for (HcLayerWeightsCpu & layer : weights) {
        reset_hc_weights_cpu(layer.attn);
        reset_hc_weights_cpu(layer.ffn);
    }
    weights.clear();
}

static bool ensure_hc_fn_device(HcWeightsCpu & w, ggml_tensor * fn) {
    if (!fn || !fn->data) return false;
    if (fn->type == GGML_TYPE_F16) return true;
    if (!ds4_rocmfpx_hc_gpu_enabled()) return false;
    if (!w.loaded || w.fn_data.empty()) return false;

#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    const size_t bytes = w.fn_data.size() * sizeof(uint16_t);
    if (w.fn_f16_device && w.fn_f16_device_bytes == bytes) {
        return true;
    }
    release_hc_fn_device(w);
    if (!deepseek4_cuda_hc_upload_f16(w.fn_data.data(), bytes, &w.fn_f16_device)) {
        w.fn_f16_device = nullptr;
        w.fn_f16_device_bytes = 0;
        return false;
    }
    w.fn_f16_device_bytes = bytes;
    return true;
#else
    return false;
#endif
}

static const void * hc_fn_device_ptr(const HcWeightsCpu & w, ggml_tensor * fn) {
    if (!fn) return nullptr;
    if (fn->type == GGML_TYPE_F16) return fn->data;
    if (!ds4_rocmfpx_hc_gpu_enabled()) return nullptr;
    return w.fn_f16_device;
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
    const auto step_t0 = Ds4TimingClock::now();

    if (w.moe_hybrid && moe_hybrid != nullptr) {
        return deepseek4_step_hybrid(backend, w, cache, *moe_hybrid,
                                     embed, n_tokens, kv_start, out_logits,
                                     token_ids, stream_engine, telemetry, routing_stats);
    }

    const int n_embd = w.n_embd;
    const int n_layer = w.n_layer;
    const size_t ctx_size = ds4_full_step_meta_size(n_tokens);
    const bool reuse_full_step_decode =
        n_tokens == 1 &&
        ds4_backend_is_gpu(backend) &&
        !ds4_env_flag("DFLASH_DS4_DISABLE_FULL_STEP_DECODE_REUSE");
    static thread_local DeepSeek4LegacyFullStepCache full_step_cache;
    StepGraph * cached_sg = nullptr;
    if (reuse_full_step_decode) {
        if (full_step_cache.owner_ctx != w.ctx || full_step_cache.backend != backend) {
            full_step_cache.free();
            full_step_cache.owner_ctx = w.ctx;
            full_step_cache.backend = backend;
        } else {
            step_graph_free(full_step_cache.sg);
        }
        cached_sg = &full_step_cache.sg;
        if (full_step_cache.meta_arena.size() < ctx_size) {
            full_step_cache.meta_arena.resize(ctx_size);
        }
    }

    ggml_init_params params{};
    params.mem_size = cached_sg ? full_step_cache.meta_arena.size() : ctx_size;
    params.mem_buffer = cached_sg ? full_step_cache.meta_arena.data() : nullptr;
    params.no_alloc = true;
    const auto full_build_t0 = Ds4TimingClock::now();
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    if (cached_sg) {
        cached_sg->ctx = ctx;
    }
    if (telemetry) {
        telemetry->full_graph_build_us += ds4_elapsed_us(full_build_t0, Ds4TimingClock::now());
    }

    ggml_gallocr_t alloc = nullptr;
    bool owns_alloc = false;
    auto release_full_step = [&]() {
        if (cached_sg) {
            step_graph_free(*cached_sg);
            return;
        }
        if (alloc && owns_alloc) {
            ggml_gallocr_free(alloc);
            alloc = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
    };

    // Input embeddings
    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(inp, "inp_embed");
    ggml_set_input(inp);
    if (cached_sg) {
        cached_sg->inp_embed = inp;
    }

    ggml_tensor * cur = inp;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, ds4_full_step_graph_size(n_tokens), false);
    if (cached_sg) {
        cached_sg->gf = gf;
    }
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
    if (cached_sg) {
        cached_sg->logits = logits;
    }

    // ── Build and run graph ─────────────────────────────────────────────
    ggml_build_forward_expand(gf, logits);

    // Allocate
    if (cached_sg) {
        if (!cached_sg->alloc) {
            cached_sg->alloc =
                ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        }
        alloc = cached_sg->alloc;
    } else {
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        owns_alloc = true;
    }
    const auto full_alloc_t0 = Ds4TimingClock::now();
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "[deepseek4] graph allocation failed\n");
        release_full_step();
        return false;
    }
    if (telemetry) {
        telemetry->full_graph_alloc_us += ds4_elapsed_us(full_alloc_t0, Ds4TimingClock::now());
    }

    // Set input data
    const auto full_set_t0 = Ds4TimingClock::now();
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
    if (telemetry) {
        telemetry->full_graph_set_us += ds4_elapsed_us(full_set_t0, Ds4TimingClock::now());
    }

    // Compute
    const auto full_compute_t0 = Ds4TimingClock::now();
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[deepseek4] graph compute failed\n");
        release_full_step();
        return false;
    }
    if (telemetry) {
        telemetry->full_graph_compute_us += ds4_elapsed_us(full_compute_t0, Ds4TimingClock::now());
    }

    // Read logits (only last token for generation)
    const auto full_read_t0 = Ds4TimingClock::now();
    out_logits.resize(w.n_vocab);
    const size_t logits_offset = (size_t)(n_tokens - 1) * w.n_vocab * sizeof(float);
    ggml_backend_tensor_get(logits, out_logits.data(), logits_offset,
                            w.n_vocab * sizeof(float));
    if (telemetry) {
        telemetry->full_graph_read_us += ds4_elapsed_us(full_read_t0, Ds4TimingClock::now());
    }

    release_full_step();

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
    if (telemetry) {
        telemetry->total_us += ds4_elapsed_us(step_t0, Ds4TimingClock::now());
    }
    return true;
}

// ─── Fused single-graph decode (n_tokens == 1) ──────────────────────────
// Chains all layers (HC pre → attention → HC post → HC pre → FFN → HC post)
// plus the output HC merge and lm_head into ONE cached ggml graph, so a
// decode step is a single ggml_backend_graph_compute with one logits
// readback instead of ~90 per-layer graph launches with host round-trips.
// HC Sinkhorn mixing runs in the fused GGML_OP_DS4_HC op (one kernel per
// sublayer instead of ~170 tiny ops for 20 Sinkhorn iterations).
//
// Compressed-KV reads are padded to DS4_FUSED_COMP_PAD rows with an additive
// score mask, and each structural variant (flush pattern) lives in its own
// slot with a private metadata arena. Tensor addresses therefore stay stable
// while a variant recurs, which is what the ggml-cuda/HIP graph cache keys
// on, enabling graph replay for the bulk of decode steps.

// Opt-in refinement of the fused graph: one topology for flush and non-flush
// steps (pooling every step + input-redirected state flush). Needed for
// CUDA/HIP graph replay experiments; costs ~3% decode speed, and HIP graph
// replay is a net loss on gfx1151 anyway, so default OFF.
static bool ds4_fused_stable_graph_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = ds4_env_flag("DFLASH_DS4_FUSED_STABLE_GRAPH") ? 1 : 0;
    }
    return enabled == 1;
}

// Opt-in: single-graph decode with GPU hyper-connections. Deterministic and
// near-bit-identical, but expf ULP differences in the sinkhorn iterations can
// diverge from the CPU-HC reference after tens of tokens. Default OFF.
static bool ds4_fused_decode_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = ds4_env_flag("DFLASH_DS4_FUSED_DECODE") ? 1 : 0;
    }
    return enabled == 1;
}

struct DeepSeek4FusedDecodeGraph {
    std::vector<int64_t> shape_key;
    uint64_t last_use = 0;
    StepGraph sg;
    ggml_tensor * inp_embed = nullptr;
    ggml_tensor * i32_bundle = nullptr;
    ggml_tensor * i64_bundle = nullptr;
    ggml_tensor * mask_bundle = nullptr;   // additive score mask (0 / -1e30), may be null
    ggml_tensor * flush_rows = nullptr;    // i64[4]: [0..3] on flush steps, [4..7] otherwise
    std::vector<ggml_tensor *> hash_ids;
    ggml_tensor * logits = nullptr;

    void reset_nodes() {
        inp_embed = nullptr;
        i32_bundle = nullptr;
        i64_bundle = nullptr;
        mask_bundle = nullptr;
        flush_rows = nullptr;
        logits = nullptr;
        hash_ids.clear();
        shape_key.clear();
        last_use = 0;
    }

    bool built() const {
        return sg.ctx && sg.gf && logits;
    }

    void destroy() {
        step_graph_destroy(sg);
        reset_nodes();
    }
};

struct DeepSeek4FusedDecodeCache {
    const ggml_context * owner_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    bool disabled = false;
    uint64_t counter = 0;
    std::array<DeepSeek4FusedDecodeGraph, 4> slots;

    // Persistent F16 mirrors of the (quantized) HC fn projection weights so
    // the fused graph matches the numerics of the reference HC paths, which
    // always dequantize fn to F16 before the mix matvec.
    ggml_context * fn_ctx = nullptr;
    ggml_backend_buffer_t fn_buf = nullptr;
    std::vector<ggml_tensor *> fn_attn_f16;
    std::vector<ggml_tensor *> fn_ffn_f16;
    ggml_tensor * fn_out_f16 = nullptr;

    void destroy() {
        for (auto & s : slots) s.destroy();
        if (fn_buf) { ggml_backend_buffer_free(fn_buf); fn_buf = nullptr; }
        if (fn_ctx) { ggml_free(fn_ctx); fn_ctx = nullptr; }
        fn_attn_f16.clear();
        fn_ffn_f16.clear();
        fn_out_f16 = nullptr;
        owner_ctx = nullptr;
        backend = nullptr;
        disabled = false;
        counter = 0;
    }
};

static ggml_tensor * ds4_fused_hc_base_f32(ggml_context * ctx, ggml_tensor * base) {
    if (!base) return nullptr;
    ggml_tensor * b = base;
    if (b->type != GGML_TYPE_F32) {
        b = ggml_cast(ctx, b, GGML_TYPE_F32);
    }
    return ggml_reshape_1d(ctx, b, ggml_nelements(b));
}

static ggml_tensor * ds4_build_fused_hc_pre(
        ggml_context * ctx,
        const DeepSeek4Weights & w,
        ggml_tensor * hc_flat,          // [n_embd*n_hc] contiguous f32
        ggml_tensor * fn,
        ggml_tensor * base,
        const HcWeightsCpu & cw,
        ggml_tensor ** out_split) {
    if (!fn || !base || !cw.loaded || cw.scale_data.size() < 3) return nullptr;
    const int mix_dim = 2 * w.n_hc + w.n_hc * w.n_hc;
    ggml_tensor * normed = ggml_rms_norm(ctx, hc_flat, w.hc_eps);
    ggml_tensor * mix = ggml_mul_mat(ctx, fn, normed);
    mix = ggml_reshape_1d(ctx, mix, mix_dim);
    ggml_tensor * base_f32 = ds4_fused_hc_base_f32(ctx, base);
    ggml_tensor * pre = ggml_ds4_hc_pre(ctx, mix, base_f32, hc_flat,
                                        w.n_hc, w.n_hc_sinkhorn_iter,
                                        cw.scale_data[0], cw.scale_data[1], cw.scale_data[2]);
    *out_split = ggml_view_1d(ctx, pre, mix_dim, (size_t) w.n_embd * sizeof(float));
    return ggml_view_1d(ctx, pre, w.n_embd, 0);
}

static ggml_tensor * ds4_build_hash_routed_ffn(
        ggml_context * ctx,
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L,
        ggml_tensor * ffn_normed,
        ggml_tensor * hash_ids) {
    ggml_tensor * shared_out = build_shared_ffn(ctx, ffn_normed, w, L);
    ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, ffn_normed);
    ggml_tensor * probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));

    const int n_used = w.n_expert_used;
    const int n_ff_exp = w.n_ff_exp;
    const bool raw_mmid = ds4_ffn_raw_mmid_enabled();
    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, ffn_normed, w.n_embd, 1, 1);
    ggml_tensor * gate_e = ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur_3d, hash_ids);
    ggml_tensor * up_e = ggml_mul_mat_id(ctx, L.ffn_up_exps, cur_3d, hash_ids);
    if (!raw_mmid) {
        gate_e = ggml_reshape_3d(ctx, gate_e, n_ff_exp, n_used, 1);
        up_e = ggml_reshape_3d(ctx, up_e, n_ff_exp, n_used, 1);
    }
    ggml_tensor * mid_e = build_clamped_swiglu(ctx, gate_e, up_e, w.swiglu_clamp_exp);
    ggml_tensor * down_e = ggml_mul_mat_id(ctx, L.ffn_down_exps, mid_e, hash_ids);
    if (!raw_mmid) {
        down_e = ggml_reshape_3d(ctx, down_e, w.n_embd, n_used, 1);
    }

    ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, w.n_expert, 1);
    ggml_tensor * weights = ggml_get_rows(ctx, probs_3d, hash_ids);
    weights = ggml_reshape_2d(ctx, weights, n_used, 1);
    ggml_tensor * w_sum = ggml_sum_rows(ctx, weights);
    w_sum = ggml_clamp(ctx, w_sum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(ctx, weights, w_sum);
    if (w.expert_weight_scale != 1.0f) {
        weights = ggml_scale(ctx, weights, w.expert_weight_scale);
    }

    ggml_tensor * routed_out = nullptr;
    if (ds4_ffn_fused_combine_enabled()) {
        routed_out = ggml_laguna_moe_combine(ctx, down_e, weights);
    } else {
        ggml_tensor * weights_3d = ggml_reshape_3d(ctx, weights, 1, n_used, 1);
        routed_out = ggml_mul(ctx, down_e, weights_3d);
        ggml_tensor * sum_shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w.n_embd, 1, 1);
        routed_out = ggml_repeat_back(ctx, routed_out, sum_shape);
        routed_out = ggml_reshape_2d(ctx, routed_out, w.n_embd, 1);
    }
    return ggml_add(ctx, shared_out, routed_out);
}

static bool ds4_fused_ensure_fn_mirrors(
        DeepSeek4FusedDecodeCache & fc,
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        const std::vector<HcLayerWeightsCpu> & hc_weights,
        const HcWeightsCpu & hc_out_weights) {
    if (fc.fn_ctx && fc.fn_buf && fc.fn_attn_f16.size() == (size_t) w.n_layer && fc.fn_out_f16) {
        return true;
    }
    if (fc.fn_buf) { ggml_backend_buffer_free(fc.fn_buf); fc.fn_buf = nullptr; }
    if (fc.fn_ctx) { ggml_free(fc.fn_ctx); fc.fn_ctx = nullptr; }
    const int64_t hc_dim = (int64_t) w.n_embd * w.n_hc;
    const int64_t mix_dim = 2 * (int64_t) w.n_hc + (int64_t) w.n_hc * w.n_hc;
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * (size_t) (2 * w.n_layer + 4) + 4096;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    fc.fn_ctx = ggml_init(params);
    if (!fc.fn_ctx) return false;
    fc.fn_attn_f16.assign((size_t) w.n_layer, nullptr);
    fc.fn_ffn_f16.assign((size_t) w.n_layer, nullptr);
    for (int il = 0; il < w.n_layer; ++il) {
        fc.fn_attn_f16[(size_t) il] = ggml_new_tensor_2d(fc.fn_ctx, GGML_TYPE_F16, hc_dim, mix_dim);
        fc.fn_ffn_f16[(size_t) il] = ggml_new_tensor_2d(fc.fn_ctx, GGML_TYPE_F16, hc_dim, mix_dim);
    }
    fc.fn_out_f16 = ggml_new_tensor_2d(fc.fn_ctx, GGML_TYPE_F16, hc_dim, w.n_hc);
    fc.fn_buf = ggml_backend_alloc_ctx_tensors(fc.fn_ctx, backend);
    if (!fc.fn_buf) {
        ggml_free(fc.fn_ctx);
        fc.fn_ctx = nullptr;
        return false;
    }
    for (int il = 0; il < w.n_layer; ++il) {
        const auto & a = hc_weights[(size_t) il].attn.fn_data;
        const auto & f = hc_weights[(size_t) il].ffn.fn_data;
        if ((int64_t) a.size() != hc_dim * mix_dim || (int64_t) f.size() != hc_dim * mix_dim) {
            return false;
        }
        ggml_backend_tensor_set(fc.fn_attn_f16[(size_t) il], a.data(), 0, a.size() * sizeof(uint16_t));
        ggml_backend_tensor_set(fc.fn_ffn_f16[(size_t) il], f.data(), 0, f.size() * sizeof(uint16_t));
    }
    const auto & o = hc_out_weights.fn_data;
    if ((int64_t) o.size() != hc_dim * w.n_hc) {
        return false;
    }
    ggml_backend_tensor_set(fc.fn_out_f16, o.data(), 0, o.size() * sizeof(uint16_t));
    return true;
}

static bool ds4_build_fused_decode_graph(
        DeepSeek4FusedDecodeCache & fc,
        DeepSeek4FusedDecodeGraph & fg,
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        DeepSeek4Cache & cache,
        const std::vector<HcLayerWeightsCpu> & hc_weights,
        const HcWeightsCpu & hc_out_weights,
        const std::vector<HashRoutingTableCpu> & hash_tables,
        int kv_start,
        bool have_token_ids,
        std::vector<int64_t> && shape_key) {
    step_graph_free(fg.sg);
    fg.reset_nodes();
    fg.hash_ids.assign((size_t) w.n_layer, nullptr);

    const int n_embd = w.n_embd;
    const int n_hc = w.n_hc;
    const int n_raw = std::min(kv_start + 1, w.n_swa);
    const int token_pos = kv_start;
    const size_t arena_size = 192u * 1024 * 1024;
    if (fg.sg.meta_arena.size() < arena_size) {
        fg.sg.meta_arena.resize(arena_size);
    }
    ggml_init_params params{};
    params.mem_size = fg.sg.meta_arena.size();
    params.mem_buffer = fg.sg.meta_arena.data();
    params.no_alloc = true;
    fg.sg.ctx = ggml_init(params);
    if (!fg.sg.ctx) return false;
    ggml_context * ctx = fg.sg.ctx;
    fg.sg.gf = ggml_new_graph_custom(ctx, 32768, false);
    ggml_cgraph * gf = fg.sg.gf;

    fg.inp_embed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_input(fg.inp_embed);
    fg.i32_bundle = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 6 * (int64_t) w.n_layer);
    ggml_set_input(fg.i32_bundle);
    fg.i64_bundle = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 5 * (int64_t) w.n_layer);
    ggml_set_input(fg.i64_bundle);
    if (ds4_fused_stable_graph_enabled()) {
        fg.flush_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 4);
        ggml_set_input(fg.flush_rows);
    }

    // One additive score-mask bundle covering EVERY layer: [n_swa raw rows ++
    // padded comp rows]. All layers take the masked full-ring attention branch
    // so the graph topology never depends on the live raw-row count.
    int64_t mask_total = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        const int ratio = (int) w.compress_ratios[il];
        int padded = 0;
        if (ratio > 0 && cache.layers[(size_t) il].comp_kv) {
            const int n_comp = ds4_comp_rows_used(cache.layers[(size_t) il].comp_kv,
                                                  cache.layers[(size_t) il].n_comp, ratio, token_pos);
            padded = ds4_padded_comp_rows(n_comp, (int) cache.layers[(size_t) il].comp_kv->ne[1]);
        }
        mask_total += w.n_swa + padded;
    }
    fg.mask_bundle = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mask_total);
    ggml_set_input(fg.mask_bundle);
    int64_t mask_off = 0;

    // hc state starts as the token embedding replicated into every stream
    ggml_tensor * hc_cur = ggml_repeat_4d(ctx, fg.inp_embed, n_embd, n_hc, 1, 1);

    for (int il = 0; il < w.n_layer; ++il) {
        const DeepSeek4Layer & L = w.layers[(size_t) il];
        DeepSeek4LayerCache & lc = cache.layers[(size_t) il];
        const HcLayerWeightsCpu & hlw = hc_weights[(size_t) il];
        const int ratio = (int) w.compress_ratios[il];

        // ── HC pre (attention) ─────────────────────────────────────
        ggml_tensor * hc_flat = ggml_reshape_1d(ctx, hc_cur, (int64_t) n_embd * n_hc);
        ggml_tensor * split_attn = nullptr;
        ggml_tensor * working = ds4_build_fused_hc_pre(ctx, w, hc_flat,
                                                       fc.fn_attn_f16[(size_t) il], L.hc_attn_base,
                                                       hlw.attn, &split_attn);
        if (!working) return false;
        ggml_tensor * attn_in = ggml_reshape_2d(ctx, working, n_embd, 1);

        // ── Attention (inputs are views into the shared bundles) ──
        DeepSeek4AttentionGraphInputs ain{};
        ain.flush_rows = ds4_fused_stable_graph_enabled() ? fg.flush_rows : nullptr;
        ain.rope_pos = ggml_view_1d(ctx, fg.i32_bundle, 1, ((size_t) il * 6 + 0) * sizeof(int32_t));
        ain.neg_pos  = ggml_view_1d(ctx, fg.i32_bundle, 1, ((size_t) il * 6 + 1) * sizeof(int32_t));
        ain.raw_kv_rows = ggml_view_2d(ctx, fg.i64_bundle, 1, 1, sizeof(int64_t),
                                       ((size_t) il * 5 + 0) * sizeof(int64_t));
        if (ratio > 0) {
            ain.attn_ape_row = ggml_view_1d(ctx, fg.i32_bundle, 1, ((size_t) il * 6 + 2) * sizeof(int32_t));
            ain.attn_comp_pos = ggml_view_1d(ctx, fg.i32_bundle, 1, ((size_t) il * 6 + 3) * sizeof(int32_t));
            ain.attn_state_rows = ggml_view_2d(ctx, fg.i64_bundle, 1, 1, sizeof(int64_t),
                                               ((size_t) il * 5 + 1) * sizeof(int64_t));
            ain.attn_comp_rows = ggml_view_2d(ctx, fg.i64_bundle, 1, 1, sizeof(int64_t),
                                              ((size_t) il * 5 + 2) * sizeof(int64_t));
        }
        if (ratio == 4) {
            ain.index_ape_row = ggml_view_1d(ctx, fg.i32_bundle, 1, ((size_t) il * 6 + 4) * sizeof(int32_t));
            ain.index_comp_pos = ggml_view_1d(ctx, fg.i32_bundle, 1, ((size_t) il * 6 + 5) * sizeof(int32_t));
            ain.index_state_rows = ggml_view_2d(ctx, fg.i64_bundle, 1, 1, sizeof(int64_t),
                                                ((size_t) il * 5 + 3) * sizeof(int64_t));
            ain.index_comp_rows = ggml_view_2d(ctx, fg.i64_bundle, 1, 1, sizeof(int64_t),
                                               ((size_t) il * 5 + 4) * sizeof(int64_t));
        }
        {
            int padded = 0;
            if (ratio > 0 && lc.comp_kv) {
                const int n_comp = ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio, token_pos);
                padded = ds4_padded_comp_rows(n_comp, (int) lc.comp_kv->ne[1]);
            }
            const int64_t n_attn = (int64_t) w.n_swa + padded;
            ain.attn_row_mask = ggml_view_2d(ctx, fg.mask_bundle, n_attn, 1,
                                             n_attn * sizeof(float),
                                             (size_t) mask_off * sizeof(float));
            ain.padded_comp = padded;
            mask_off += n_attn;
        }

        std::vector<DeepSeek4I32InputBinding> i32b;
        std::vector<DeepSeek4I32ArrayBinding> i32ab;
        std::vector<DeepSeek4I64ArrayBinding> i64ab;
        ggml_tensor * normed = build_rms_norm(ctx, attn_in, L.attn_norm, w.rms_eps);
        ggml_tensor * attn_out = build_mla_attention(ctx, gf, normed, w, L, lc, il,
                                                     kv_start, 1, &ain,
                                                     i32b, i32ab, i64ab);
        if (!attn_out) return false;
        if (!i32b.empty() || !i32ab.empty() || !i64ab.empty()) {
            std::fprintf(stderr,
                         "[deepseek4] fused decode: layer %d created %zu/%zu/%zu dynamic bindings; cannot fuse\n",
                         il, i32b.size(), i32ab.size(), i64ab.size());
            return false;
        }

        // ── HC post (attention) ────────────────────────────────────
        ggml_tensor * attn_out_flat = ggml_reshape_1d(ctx, attn_out, n_embd);
        ggml_tensor * hc_next = ggml_ds4_hc_post(ctx, hc_flat, attn_out_flat, split_attn, n_hc);
        hc_cur = ggml_reshape_2d(ctx, hc_next, n_embd, n_hc);

        // ── HC pre (FFN) ───────────────────────────────────────────
        hc_flat = ggml_reshape_1d(ctx, hc_cur, (int64_t) n_embd * n_hc);
        ggml_tensor * split_ffn = nullptr;
        ggml_tensor * fworking = ds4_build_fused_hc_pre(ctx, w, hc_flat,
                                                        fc.fn_ffn_f16[(size_t) il], L.hc_ffn_base,
                                                        hlw.ffn, &split_ffn);
        if (!fworking) return false;
        ggml_tensor * ffn_in = ggml_reshape_2d(ctx, fworking, n_embd, 1);

        // ── FFN ────────────────────────────────────────────────────
        ggml_tensor * ffn_normed = build_rms_norm(ctx, ffn_in, L.ffn_norm, w.rms_eps);
        ggml_tensor * ffn_out = nullptr;
        const bool hash_routed = il < w.n_hash_layer && L.ffn_gate_tid2eid &&
                                 have_token_ids && hash_tables[(size_t) il].loaded;
        if (hash_routed) {
            ggml_tensor * hids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, w.n_expert_used, 1);
            ggml_set_input(hids);
            fg.hash_ids[(size_t) il] = hids;
            ffn_out = ds4_build_hash_routed_ffn(ctx, w, L, ffn_normed, hids);
        } else {
            ffn_out = build_moe_ffn(ctx, ffn_normed, w, L, il, 1);
        }
        if (!ffn_out) return false;

        // ── HC post (FFN) ──────────────────────────────────────────
        ggml_tensor * ffn_out_flat = ggml_reshape_1d(ctx, ffn_out, n_embd);
        hc_next = ggml_ds4_hc_post(ctx, hc_flat, ffn_out_flat, split_ffn, n_hc);
        hc_cur = ggml_reshape_2d(ctx, hc_next, n_embd, n_hc);
    }

    // ── Output: HC merge → norm → lm_head ──────────────────────────
    ggml_tensor * hc_flat = ggml_reshape_1d(ctx, hc_cur, (int64_t) n_embd * n_hc);
    ggml_tensor * onorm = ggml_rms_norm(ctx, hc_flat, w.hc_eps);
    ggml_tensor * omix = ggml_mul_mat(ctx, fc.fn_out_f16, onorm);
    omix = ggml_reshape_1d(ctx, omix, ggml_nelements(omix));
    ggml_tensor * obase = ds4_fused_hc_base_f32(ctx, w.output_hc_base);
    if (!obase || hc_out_weights.scale_data.empty()) return false;
    ggml_tensor * final_embd = ggml_ds4_hc_out(ctx, omix, obase, hc_flat, n_hc,
                                               hc_out_weights.scale_data[0]);
    ggml_tensor * final_2d = ggml_reshape_2d(ctx, final_embd, n_embd, 1);
    ggml_tensor * out_normed = build_rms_norm(ctx, final_2d, w.out_norm, w.rms_eps);
    fg.logits = ggml_mul_mat(ctx, w.output, out_normed);
    ggml_set_output(fg.logits);
    ggml_build_forward_expand(gf, fg.logits);

    if (!fg.sg.alloc) {
        fg.sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!fg.sg.alloc || !ggml_gallocr_alloc_graph(fg.sg.alloc, fg.sg.gf)) {
        std::fprintf(stderr, "[deepseek4] fused decode graph alloc failed\n");
        return false;
    }

    fg.shape_key = std::move(shape_key);
    return true;
}

#include "deepseek4_fused_verify.inc"

// Returns 1 on success (out_logits filled), 0 to fall back to the per-layer
// path, -1 on a hard failure after cache state may have been touched.
static int ds4_try_fused_decode_step(
        DeepSeek4FusedDecodeCache & fc,
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        DeepSeek4Cache & cache,
        const std::vector<HcLayerWeightsCpu> & hc_weights,
        const HcWeightsCpu & hc_out_weights,
        const std::vector<HashRoutingTableCpu> & hash_tables,
        std::vector<int32_t> & hash_scratch,
        const float * embed,
        int kv_start,
        std::vector<float> & out_logits,
        const int32_t * token_ids,
        DeepSeek4StepTelemetry * telemetry) {
    if (fc.disabled) return 0;
    if (!hc_out_weights.loaded || hc_out_weights.scale_data.empty() ||
        !w.output_hc_fn || !w.output_hc_base) {
        fc.disabled = true;
        return 0;
    }
    for (int il = 0; il < w.n_layer; ++il) {
        const HcLayerWeightsCpu & hlw = hc_weights[(size_t) il];
        const DeepSeek4Layer & L = w.layers[(size_t) il];
        if (!hlw.attn.loaded || hlw.attn.scale_data.size() < 3 ||
            !hlw.ffn.loaded || hlw.ffn.scale_data.size() < 3 ||
            !L.hc_attn_fn || !L.hc_ffn_fn || !L.hc_attn_base || !L.hc_ffn_base) {
            fc.disabled = true;
            return 0;
        }
    }

    if (fc.owner_ctx != w.ctx || fc.backend != backend) {
        fc.destroy();
        fc.owner_ctx = w.ctx;
        fc.backend = backend;
    }
    if (!ds4_fused_ensure_fn_mirrors(fc, backend, w, hc_weights, hc_out_weights)) {
        std::fprintf(stderr, "[deepseek4] fused decode: HC fn mirror upload failed; using per-layer path\n");
        fc.disabled = true;
        return 0;
    }

    const int token_pos = kv_start;
    const int n_raw = std::min(kv_start + 1, w.n_swa);
    std::vector<int64_t> key;
    key.reserve((size_t) w.n_layer + 2);
    key.push_back(w.n_swa);
    key.push_back(token_ids ? 1 : 0);
    for (int il = 0; il < w.n_layer; ++il) {
        const int ratio = (int) w.compress_ratios[il];
        DeepSeek4LayerCache & lc = cache.layers[(size_t) il];
        int padded = 0;
        if (ratio > 0 && lc.comp_kv) {
            const int n_comp = ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio, token_pos);
            padded = ds4_padded_comp_rows(n_comp, (int) lc.comp_kv->ne[1]);
        }
        if (ds4_fused_stable_graph_enabled()) {
            key.push_back((int64_t) padded);
        } else {
            const bool flush = ratio > 0 && (((token_pos + 1) % ratio) == 0);
            key.push_back(((int64_t) padded << 1) | (flush ? 1 : 0));
        }
    }

    // Pick the slot whose shape key matches; otherwise rebuild the LRU slot.
    fc.counter++;
    DeepSeek4FusedDecodeGraph * fg = nullptr;
    for (auto & s : fc.slots) {
        if (s.built() && s.shape_key == key) {
            fg = &s;
            break;
        }
    }
    if (!fg) {
        for (auto & s : fc.slots) {
            if (!s.built()) { fg = &s; break; }
        }
        if (!fg) {
            fg = &fc.slots[0];
            for (auto & s : fc.slots) {
                if (s.last_use < fg->last_use) fg = &s;
            }
        }
        const auto build_t0 = Ds4TimingClock::now();
        if (!ds4_build_fused_decode_graph(fc, *fg, backend, w, cache,
                                          hc_weights, hc_out_weights, hash_tables,
                                          kv_start, token_ids != nullptr, std::move(key))) {
            std::fprintf(stderr,
                         "[deepseek4] fused decode graph build failed; using per-layer path\n");
            step_graph_free(fg->sg);
            fg->reset_nodes();
            fc.disabled = true;
            return 0;
        }
        if (telemetry) telemetry->full_graph_build_us += ds4_elapsed_us(build_t0, Ds4TimingClock::now());
    }
    fg->last_use = fc.counter;

    // ── Fill inputs ─────────────────────────────────────────────────
    const auto set_t0 = Ds4TimingClock::now();
    ggml_backend_tensor_set(fg->inp_embed, embed, 0, sizeof(float) * (size_t) w.n_embd);

    std::vector<int32_t> i32v((size_t) w.n_layer * 6, 0);
    std::vector<int64_t> i64v((size_t) w.n_layer * 5, 0);
    const int64_t raw_row = kv_start % w.n_swa;
    for (int il = 0; il < w.n_layer; ++il) {
        const int ratio = (int) w.compress_ratios[il];
        i32v[(size_t) il * 6 + 0] = kv_start;
        i32v[(size_t) il * 6 + 1] = -kv_start;
        i64v[(size_t) il * 5 + 0] = raw_row;
        if (ratio > 0) {
            const int pos_mod = token_pos % ratio;
            i32v[(size_t) il * 6 + 2] = pos_mod;
            i32v[(size_t) il * 6 + 3] = token_pos + 1 - ratio;
            i64v[(size_t) il * 5 + 1] = (ratio == 4) ? (int64_t) (ratio + pos_mod) : (int64_t) pos_mod;
            i64v[(size_t) il * 5 + 2] = token_pos / ratio;
        }
        if (ratio == 4) {
            const int pos_mod = token_pos % ratio;
            i32v[(size_t) il * 6 + 4] = pos_mod;
            i32v[(size_t) il * 6 + 5] = token_pos + 1 - ratio;
            i64v[(size_t) il * 5 + 3] = ratio + pos_mod;
            i64v[(size_t) il * 5 + 4] = token_pos / ratio;
        }
    }
    ggml_backend_tensor_set(fg->i32_bundle, i32v.data(), 0, sizeof(int32_t) * i32v.size());
    ggml_backend_tensor_set(fg->i64_bundle, i64v.data(), 0, sizeof(int64_t) * i64v.size());
    if (fg->flush_rows) {
        const bool flush4 = ((token_pos + 1) % 4) == 0;
        const int64_t fr[4] = {flush4 ? 0 : 4, flush4 ? 1 : 5, flush4 ? 2 : 6, flush4 ? 3 : 7};
        ggml_backend_tensor_set(fg->flush_rows, fr, 0, sizeof(fr));
    }

    if (fg->mask_bundle) {
        std::vector<float> maskv((size_t) ggml_nelements(fg->mask_bundle), 0.0f);
        size_t off = 0;
        const int n_valid_raw = std::min(kv_start + 1, w.n_swa);
        for (int il = 0; il < w.n_layer; ++il) {
            const int ratio = (int) w.compress_ratios[il];
            DeepSeek4LayerCache & lc = cache.layers[(size_t) il];
            int n_comp = 0, padded = 0;
            if (ratio > 0 && lc.comp_kv) {
                n_comp = ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio, token_pos);
                padded = ds4_padded_comp_rows(n_comp, (int) lc.comp_kv->ne[1]);
            }
            for (int j = n_valid_raw; j < w.n_swa; ++j) {
                maskv[off + (size_t) j] = -1.0e30f;
            }
            for (int j = n_comp; j < padded; ++j) {
                maskv[off + (size_t) w.n_swa + (size_t) j] = -1.0e30f;
            }
            off += (size_t) w.n_swa + (size_t) padded;
        }
        if (off != maskv.size()) {
            std::fprintf(stderr, "[deepseek4] fused decode: mask layout mismatch (%zu vs %zu)\n",
                         off, maskv.size());
            return -1;
        }
        ggml_backend_tensor_set(fg->mask_bundle, maskv.data(), 0, sizeof(float) * maskv.size());
    }

    if (token_ids) {
        for (int il = 0; il < w.n_layer; ++il) {
            ggml_tensor * hids = fg->hash_ids[(size_t) il];
            if (!hids) continue;
            const auto & ht = hash_tables[(size_t) il].ids;
            const int n_used = w.n_expert_used;
            hash_scratch.resize((size_t) n_used);
            const int32_t tok = token_ids[0];
            std::memcpy(hash_scratch.data(), ht.data() + (size_t) tok * (size_t) n_used,
                        (size_t) n_used * sizeof(int32_t));
            ggml_backend_tensor_set(hids, hash_scratch.data(), 0,
                                    sizeof(int32_t) * (size_t) n_used);
        }
    }
    if (telemetry) telemetry->full_graph_set_us += ds4_elapsed_us(set_t0, Ds4TimingClock::now());

    // ── Compute ─────────────────────────────────────────────────────
    const auto compute_t0 = Ds4TimingClock::now();
    if (ggml_backend_graph_compute(backend, fg->sg.gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[deepseek4] fused decode graph compute failed\n");
        return -1;
    }
    if (telemetry) telemetry->full_graph_compute_us += ds4_elapsed_us(compute_t0, Ds4TimingClock::now());

    // ── Read logits ─────────────────────────────────────────────────
    const auto read_t0 = Ds4TimingClock::now();
    out_logits.resize((size_t) w.n_vocab);
    ggml_backend_tensor_get(fg->logits, out_logits.data(), 0,
                            sizeof(float) * (size_t) w.n_vocab);
    if (telemetry) telemetry->full_graph_read_us += ds4_elapsed_us(read_t0, Ds4TimingClock::now());
    return 1;
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
        DeepSeek4StepTelemetry * telemetry,
        bool allow_decode_graph_reuse,
        Ds4VerifyHooks * verify_hooks) {
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
    static DeepSeek4FusedDecodeCache fused_decode_graph_cache;
    static Ds4DecodeSharedInputs decode_shared_inputs;
    static int hc_loaded_n_layer = 0;
    static const ggml_context * hc_loaded_ctx = nullptr;
    if (hc_loaded_n_layer != w.n_layer || hc_loaded_ctx != w.ctx) {
        reset_hc_layer_weights_cpu(hc_layer_weights_range);
        reset_hc_weights_cpu(hc_output_weights_range);
        hc_layer_weights_range.resize((size_t)w.n_layer);
        hash_routing_tables_range.assign((size_t)w.n_layer, {});
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
        fused_decode_graph_cache.destroy();
        decode_shared_inputs.free();
        for (int il = 0; il < w.n_layer; il++) {
            const DeepSeek4Layer & L = w.layers[(size_t)il];
            load_hc_weights_cpu(hc_layer_weights_range[il].attn, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base);
            load_hc_weights_cpu(hc_layer_weights_range[il].ffn, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base);
            if (ds4_backend_is_gpu(backend) && !ds4_env_flag("DFLASH_DS4_HC_CPU")) {
                ensure_hc_fn_device(hc_layer_weights_range[il].attn, L.hc_attn_fn);
                ensure_hc_fn_device(hc_layer_weights_range[il].ffn, L.hc_ffn_fn);
            }
            if (il < w.n_hash_layer && L.ffn_gate_tid2eid) {
                load_hash_routing_cpu(hash_routing_tables_range[(size_t)il], L.ffn_gate_tid2eid);
            }
        }
        load_hc_weights_cpu(hc_output_weights_range, w.output_hc_fn, w.output_hc_scale, w.output_hc_base);
        hc_loaded_n_layer = w.n_layer;
        hc_loaded_ctx = w.ctx;
    }

    // Per-layer execution with CPU-side HC
    static thread_local DeepSeek4LayerRangeScratch scratch;
    scratch.ensure(w.ctx, n_tokens, n_embd, n_hc, w.n_expert_used);

    if (n_tokens >= 2 && n_tokens <= 4 && verify_hooks && layer_begin == 0 && is_last_shard &&
        out_logits && ds4_backend_is_gpu(backend) && ds4_fused_verify_enabled()) {
        const int vrc = ds4_try_fused_verify_step(
            fused_verify_graph_cache, fused_decode_graph_cache, backend, w, cache,
            hc_layer_weights_range, hc_output_weights_range, hash_routing_tables_range,
            scratch.hash_expert_ids, embed, n_tokens, kv_start, *out_logits, token_ids,
            verify_hooks, telemetry);
        if (vrc < 0) return false;
        if (vrc > 0) {
            const int np = kv_start + n_tokens;
            for (int il = layer_begin; il < layer_end; ++il) {
                const uint32_t vratio = w.compress_ratios[il];
                if (vratio <= 0) continue;
                cache.layers[il].n_comp = std::max(cache.layers[il].n_comp, np / (int) vratio);
                if (vratio == 4) cache.layers[il].n_index_comp = std::max(cache.layers[il].n_index_comp, np / (int) vratio);
            }
            cache.cur_pos = np;
            if (telemetry) telemetry->total_us += ds4_elapsed_us(step_t0, Ds4TimingClock::now());
            return true;
        }
    }
    std::vector<float> fused_debug_logits;
    if (n_tokens == 1 && allow_decode_graph_reuse && layer_begin == 0 && is_last_shard &&
        out_logits && ds4_backend_is_gpu(backend) && ds4_fused_decode_enabled()) {
        const int rc = ds4_try_fused_decode_step(
            fused_decode_graph_cache, backend, w, cache, hc_layer_weights_range,
            hc_output_weights_range, hash_routing_tables_range, scratch.hash_expert_ids,
            embed, kv_start, *out_logits, token_ids, telemetry);
        if (rc < 0) return false;
        if (rc > 0 && ds4_env_flag("DFLASH_DS4_FUSED_DEBUG")) {
            // Debug: keep the fused logits, then fall through and run the
            // per-layer reference for the same token; compare at the end.
            fused_debug_logits = *out_logits;
        } else if (rc > 0) {
            const int np = kv_start + 1;
            for (int il = layer_begin; il < layer_end; ++il) {
                const uint32_t ratio = w.compress_ratios[il];
                if (ratio <= 0 || (np % (int) ratio) != 0) continue;
                cache.layers[il].n_comp = std::max(cache.layers[il].n_comp, np / (int) ratio);
                if (ratio == 4) cache.layers[il].n_index_comp = std::max(cache.layers[il].n_index_comp, np / (int) ratio);
            }
            cache.cur_pos = np;
            if (telemetry) telemetry->total_us += ds4_elapsed_us(step_t0, Ds4TimingClock::now());
            return true;
        }
    }
    std::vector<float> & cur = scratch.cur;
    std::vector<float> & ffn_working = scratch.ffn_working;
    std::vector<float> & hc_post = scratch.hc_post;
    std::vector<float> & hc_comb = scratch.hc_comb;
    std::vector<float> & next_hc = scratch.next_hc;
    std::vector<float> & attn_out_host = scratch.attn_out_host;
    std::vector<float> & ffn_out_host = scratch.ffn_out_host;
    std::vector<int32_t> & hash_expert_ids_host = scratch.hash_expert_ids;
    const bool reuse_decode_graphs = allow_decode_graph_reuse && (n_tokens == 1);
    Ds4DecodeSharedInputs * shared_inputs = nullptr;
    if (reuse_decode_graphs && ds4_backend_is_gpu(backend) &&
        decode_shared_inputs.ensure(w, backend)) {
        decode_shared_inputs.set_step(w, kv_start);
        shared_inputs = &decode_shared_inputs;
    }

    bool backend_decode_hc_supported = true;
    for (int il = layer_begin; il < layer_end; ++il) {
        const DeepSeek4Layer & L = w.layers[(size_t)il];
        const HcLayerWeightsCpu & hc_lw = hc_layer_weights_range[(size_t)il];
        if (!hc_fn_device_ptr(hc_lw.attn, L.hc_attn_fn) ||
            !hc_fn_device_ptr(hc_lw.ffn, L.hc_ffn_fn)) {
            backend_decode_hc_supported = false;
            break;
        }
    }
    const bool use_backend_decode_hc =
        reuse_decode_graphs &&
        ds4_backend_is_gpu(backend) &&
        backend_decode_hc_supported &&
        !ds4_env_flag("DFLASH_DS4_HC_CPU");
    const bool use_backend_decode_hc_direct = use_backend_decode_hc && ds4_backend_is_hip(backend);
    const bool use_backend_decode_hc_graph =
        use_backend_decode_hc && !use_backend_decode_hc_direct;
    ggml_tensor * hc_state_backend = nullptr;
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
                                           hc_state_backend,
                                           L.hc_attn_fn,
                                           hc_fn_device_ptr(hc_lw.attn, L.hc_attn_fn),
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
            const bool reuse_decode_attn = reuse_decode_graphs;
            ggml_tensor * attn_out = nullptr;
            ggml_cgraph * gf = nullptr;
            ggml_context * ctx = nullptr;
            DeepSeek4CachedDecodeAttnGraph * cached_attn = nullptr;

            if (reuse_decode_attn) {
                const int n_raw = std::min(kv_start + 1, w.n_swa);
                const int n_comp_attn = (ratio > 0) ? ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio, token_pos) : 0;
                const int n_index_comp = (ratio == 4) ? ds4_comp_rows_used(lc.index_comp_kv, lc.n_index_comp, 4, token_pos) : 0;
                const bool attn_flush = ratio > 0 && (((token_pos + 1) % ratio) == 0);
                const bool index_flush = ratio == 4 && (((token_pos + 1) % ratio) == 0);
                auto & per_layer = cached_decode_attn_graphs[(size_t)il];
                auto it = std::find_if(per_layer.begin(), per_layer.end(),
                    [&](const DeepSeek4CachedDecodeAttnGraph & candidate) {
                        return candidate.valid() &&
                               candidate.owner_ctx == w.ctx &&
                               candidate.backend == backend &&
                               candidate.layer_idx == il &&
                               candidate.n_raw == n_raw &&
                               candidate.n_comp_attn == n_comp_attn &&
                               candidate.n_index_comp == n_index_comp &&
                               candidate.attn_flush == attn_flush &&
                               candidate.index_flush == index_flush;
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
                                                        n_raw, n_comp_attn, n_index_comp,
                                                        shared_inputs)) {
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
                if (!cached_attn->uses_shared_inputs) {
                ggml_backend_tensor_set(cached_attn->inputs.rope_pos, &rope_pos, 0, sizeof(rope_pos));
                ggml_backend_tensor_set(cached_attn->inputs.neg_pos, &neg_pos, 0, sizeof(neg_pos));
                ggml_backend_tensor_set(cached_attn->inputs.raw_kv_rows, &raw_row, 0, sizeof(raw_row));
                if (ratio > 0) {
                    const int pos_mod = token_pos % ratio;
                    const int32_t ape_row = pos_mod;
                    const int64_t state_row = (ratio == 4) ? (ratio + pos_mod) : pos_mod;
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
                }
            } else {
                const auto attn_build_t0 = Ds4TimingClock::now();
                const size_t ctx_size = ds4_attn_step_meta_size(n_tokens);
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
                std::vector<DeepSeek4F32ArrayBinding> f32_array_inputs;
                const size_t graph_size = ds4_attn_step_graph_size(n_tokens);
                gf = ggml_new_graph_custom(ctx, graph_size, false);

                ggml_tensor * normed = build_rms_norm(ctx, inp, L.attn_norm, w.rms_eps);
                attn_out = build_mla_attention(ctx, gf, normed, w, L, lc, il,
                                               kv_start, n_tokens, nullptr,
                                               i32_inputs, i32_array_inputs,
                                               i64_array_inputs, &f32_array_inputs);
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
                for (const auto & b : f32_array_inputs)
                    ggml_backend_tensor_set(b.tensor, b.values.data(), 0, sizeof(float) * b.values.size());
            }

            const auto attn_compute_t0 = Ds4TimingClock::now();
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[deepseek4] attn compute failed layer %d\n", il);
                if (ctx) ggml_free(ctx);
                return false;
            }
            if (telemetry) telemetry->attn_compute_us += ds4_elapsed_us(attn_compute_t0, Ds4TimingClock::now());
            if (use_backend_decode_hc_graph || use_backend_decode_hc_direct) {
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
                if (telemetry) telemetry->attn_read_us += ds4_elapsed_us(attn_read_t0, Ds4TimingClock::now());
            }
            if (ctx) ggml_free(ctx);

            // ── HC post (attention) ─────────────────────────────────
            if (!(use_backend_decode_hc_graph || use_backend_decode_hc_direct)) {
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
                                           hc_state_backend,
                                           L.hc_ffn_fn,
                                           hc_fn_device_ptr(hc_lw.ffn, L.hc_ffn_fn),
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

            if (use_backend_decode_hc_graph || use_backend_decode_hc_direct) {
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
                if (telemetry) telemetry->ffn_read_us += ds4_elapsed_us(ffn_read_t0, Ds4TimingClock::now());
            }

            // ── HC post (FFN) ───────────────────────────────────────
            if (!(use_backend_decode_hc_graph || use_backend_decode_hc_direct)) {
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
                if (verify_hooks && verify_hooks->capture_layer_ids && verify_hooks->capture_out) {
                    const std::vector<int> & _ids = *verify_hooks->capture_layer_ids;
                    for (size_t _ci = 0; _ci < _ids.size(); ++_ci) {
                        if (_ids[_ci] != il) continue;
                        const int _ncap = (int) _ids.size();
                        std::vector<float> & _cap = *verify_hooks->capture_out;
                        if ((int) _cap.size() != _ncap * n_embd * n_tokens)
                            _cap.assign((size_t) _ncap * n_embd * n_tokens, 0.0f);
                        for (int _t = 0; _t < n_tokens; ++_t) {
                            float * _dst = _cap.data() + (size_t) _t * _ncap * n_embd + (size_t) _ci * n_embd;
                            const float * _hs = hc_state.data() + (size_t) _t * hc_dim;
                            for (int _d = 0; _d < n_embd; ++_d) {
                                float _acc = 0.0f;
                                for (int _h = 0; _h < n_hc; ++_h) _acc += _hs[(size_t) _h * n_embd + _d];
                                _dst[_d] = _acc / (float) n_hc;
                            }
                        }
                    }
                }
            }
        }
    }

    if ((use_backend_decode_hc_graph || use_backend_decode_hc_direct) && hc_state_backend) {
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
            if (verify_hooks && verify_hooks->all_logits_out) {
                verify_hooks->all_logits_out->resize((size_t) w.n_vocab * n_tokens);
                ggml_backend_tensor_get(logits, verify_hooks->all_logits_out->data(), 0,
                                        sizeof(float) * (size_t) w.n_vocab * n_tokens);
            }
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
    if (!fused_debug_logits.empty() && out_logits && !out_logits->empty()) {
        const std::vector<float> & ref = *out_logits;
        size_t n = std::min(ref.size(), fused_debug_logits.size());
        float maxd = 0.0f; size_t maxi = 0;
        size_t aref = 0, afus = 0;
        for (size_t i = 1; i < n; ++i) {
            if (ref[i] > ref[aref]) aref = i;
            if (fused_debug_logits[i] > fused_debug_logits[afus]) afus = i;
            const float d = std::fabs(ref[i] - fused_debug_logits[i]);
            if (d > maxd) { maxd = d; maxi = i; }
        }
        std::fprintf(stderr,
                     "[ds4-fused-dbg] pos=%d argmax ref=%zu(%.4f) fused=%zu(%.4f) %s maxdiff=%.6f@%zu\n",
                     kv_start, aref, ref[aref], afus, fused_debug_logits[afus],
                     aref == afus ? "SAME" : "DIFF", maxd, maxi);
    }
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

void free_deepseek4_cache(DeepSeek4Cache & c) {
    if (c.ctx) { ggml_free(c.ctx); c.ctx = nullptr; }
    if (c.buf) { ggml_backend_buffer_free(c.buf); c.buf = nullptr; }
    c.layers.clear();
    c.hc_state = nullptr;
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


void free_deepseek4_snapshot(DeepSeek4Snapshot & s) {
    if (s.ctx) { ggml_free(s.ctx); s.ctx = nullptr; }
    if (s.buf) { ggml_backend_buffer_free(s.buf); s.buf = nullptr; }
    s.layers.clear();
    s.cur_pos = 0;
    s.hc_state_snap = nullptr;
}

}  // namespace dflash::common

// ══════════════════════════════════════════════════════════════════════
//  DSpark drafter forward graph (appended to deepseek4_graph.cpp so it can
//  reuse the file-static DS4 sub-builders: build_rms_norm, build_tail_rope_*,
//  build_moe_ffn, build_shared_ffn). See deepseek4_dspark.h for the contract.
//
//  Mirrors deepseek-ai/DeepSeek-V4-Flash-DSpark inference/model.py:
//    forward_embed -> main_x = main_norm(main_proj(cat[h40,h41,h42]))
//    per layer (DSparkBlock): HC-pre (per block position) -> attn_norm ->
//      DSparkAttention (bidirectional over [ctx main-KV ++ block-KV]) ->
//      HC-post ; HC-pre -> ffn_norm -> MoE -> HC-post
//    tail: hc_head collapse -> out_norm  (input to the tied lm_head + Markov)
//
//  The ggml_ds4_hc_* ops are single-token, so HC-pre/HC-post run per block
//  position; attention batches all block positions together (bidirectional).
// ══════════════════════════════════════════════════════════════════════

#include "deepseek4_dspark.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace dflash::common {

namespace {

// Fresh MLA attention for the drafter: no KV cache, no compression. The 5
// block queries attend over an explicit [ctx main-context KV ++ block KV]
// tensor with full (bidirectional) visibility, plus the learned per-head sink.
static ggml_tensor * build_dspark_attention(
        ggml_context * ctx,
        ggml_tensor * cur,      // [n_embd, block]  (post attn_norm)
        ggml_tensor * main_x,   // [n_embd, ctx_len] (post main_norm, shared)
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L,
        int ctx_len,
        ggml_tensor * pos_block,   // I32[block]    absolute positions committed..committed+block-1
        ggml_tensor * neg_block,   // I32[block]    -(block positions)
        ggml_tensor * pos_ctx) {   // I32[ctx_len]  absolute positions committed-ctx_len..committed-1
    const int n_embd    = w.n_embd;
    const int head_dim  = w.head_dim;
    const int n_head    = w.n_head;
    const int n_rot     = w.n_rot;
    const int n_lora_o  = w.n_lora_o;
    const int n_out_group = w.n_out_group;
    const int block     = (int) cur->ne[1];
    const float eps     = w.rms_eps;
    // DSparkAttention has compress_ratio==0 -> base RoPE, YaRN disabled.
    const float rope_freq = w.rope_freq_base;
    const float rope_scale = 1.0f, rope_ext = 0.0f, rope_attn = 1.0f;
    const int rope_orig = (int) w.rope_orig_ctx;

    // ── Q path (block queries) ──────────────────────────────────────
    ggml_tensor * qr = build_rms_norm(ctx, ggml_mul_mat(ctx, L.attn_q_a, cur), L.attn_q_a_norm, eps);
    ggml_tensor * q = ggml_mul_mat(ctx, L.attn_q_b, qr);          // [n_head*head_dim, block]
    q = ggml_reshape_3d(ctx, q, head_dim, n_head, block);
    q = ggml_rms_norm(ctx, q, eps);                               // per-head unweighted
    q = build_tail_rope_3d(ctx, q, pos_block, n_rot, head_dim, n_head, block,
                           rope_freq, rope_scale, rope_ext, rope_attn,
                           w.rope_yarn_beta_fast, w.rope_yarn_beta_slow, rope_orig);

    // ── KV: block positions ─────────────────────────────────────────
    ggml_tensor * kv_b = build_rms_norm(ctx, ggml_mul_mat(ctx, L.attn_kv, cur), L.attn_kv_a_norm, eps);
    kv_b = build_tail_rope_2d(ctx, kv_b, pos_block, n_rot, head_dim, block,
                              rope_freq, rope_scale, rope_ext, rope_attn,
                              w.rope_yarn_beta_fast, w.rope_yarn_beta_slow, rope_orig);

    // ── KV: context positions (from shared main_x) ──────────────────
    ggml_tensor * kv_attn = kv_b;
    int n_attn = block;
    if (ctx_len > 0) {
        ggml_tensor * kv_c = build_rms_norm(ctx, ggml_mul_mat(ctx, L.attn_kv, main_x), L.attn_kv_a_norm, eps);
        kv_c = build_tail_rope_2d(ctx, kv_c, pos_ctx, n_rot, head_dim, ctx_len,
                                  rope_freq, rope_scale, rope_ext, rope_attn,
                                  w.rope_yarn_beta_fast, w.rope_yarn_beta_slow, rope_orig);
        kv_attn = ggml_concat(ctx, kv_c, kv_b, 1);               // [head_dim, ctx_len+block]
        n_attn = ctx_len + block;
    }

    // ── Scores + sink softmax (full visibility, no causal mask) ─────
    ggml_tensor * q_flat = ggml_reshape_2d(ctx, q, head_dim, n_head * block);
    ggml_tensor * scores = ggml_mul_mat(ctx, kv_attn, q_flat);    // [n_attn, n_head*block]
    scores = ggml_scale(ctx, scores, 1.0f / sqrtf((float) head_dim));
    ggml_tensor * probs = nullptr;
    if (L.attn_sinks) {
        ggml_tensor * sink = ggml_reshape_2d(ctx, L.attn_sinks, 1, n_head);
        ggml_tensor * sink_shape = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_head * block);
        sink = ggml_repeat(ctx, sink, sink_shape);
        ggml_tensor * sws = ggml_concat(ctx, scores, sink, 0);    // [n_attn+1, n_head*block]
        ggml_tensor * pws = ggml_soft_max(ctx, sws);
        probs = ggml_view_2d(ctx, pws, n_attn, n_head * block, pws->nb[1], 0);
    } else {
        probs = ggml_soft_max(ctx, scores);
    }

    // ── Context, inverse RoPE, grouped low-rank output ──────────────
    ggml_tensor * kv_T = ggml_cont(ctx, ggml_transpose(ctx, kv_attn));  // [n_attn, head_dim]
    ggml_tensor * context = ggml_mul_mat(ctx, kv_T, probs);             // [head_dim, n_head*block]
    context = ggml_reshape_3d(ctx, context, head_dim, n_head, block);
    context = build_tail_rope_3d(ctx, context, neg_block, n_rot, head_dim, n_head, block,
                                 rope_freq, rope_scale, rope_ext, rope_attn,
                                 w.rope_yarn_beta_fast, w.rope_yarn_beta_slow, rope_orig);
    ggml_tensor * attn_out = ggml_reshape_2d(ctx, context, head_dim * n_head, block);
    const int group_dim = head_dim * (n_head / n_out_group);
    attn_out = ggml_reshape_3d(ctx, attn_out, group_dim, n_out_group, block);
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));  // [group_dim, block, n_out_group]
    ggml_tensor * out_a_3d = ggml_reshape_3d(ctx, L.attn_output_a, group_dim, n_lora_o, n_out_group);
    ggml_tensor * attn_low = ggml_mul_mat(ctx, out_a_3d, attn_out);      // [n_lora_o, block, n_out_group]
    attn_low = ggml_cont(ctx, ggml_permute(ctx, attn_low, 0, 2, 1, 3));  // [n_lora_o, n_out_group, block]
    attn_low = ggml_reshape_2d(ctx, attn_low, n_lora_o * n_out_group, block);
    return ggml_mul_mat(ctx, L.attn_output_b, attn_low);                 // [n_embd, block]
}

// Read a small F32 GPU tensor (HC scale, [k]) into host floats.
static void ds4_read_f32(ggml_tensor * t, float * dst, int k) {
    if (t) ggml_backend_tensor_get(t, dst, 0, sizeof(float) * (size_t) k);
    else   for (int i = 0; i < k; i++) dst[i] = 0.0f;
}

}  // namespace

// ── Cached drafter graph ────────────────────────────────────────────────
// The drafter forward runs every spec step with identical topology (ctx_len
// is constant once the feature window fills at n_swa). Rebuilding the
// multi-thousand-node graph, zero-initializing a fresh 256 MB arena and
// re-planning gallocr each call used to cost more than the 3-layer compute
// itself (~63 ms/step). Cache the built graph keyed by (ctx_len, block,
// drafter instance) and re-set only the inputs per call.
namespace {

struct DsparkDraftCache {
    int ctx_len = -1;
    int block   = -1;
    const void * drafter = nullptr;
    std::vector<uint8_t> arena;
    ggml_context * ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * inp_noise = nullptr;
    ggml_tensor * inp_ctx = nullptr;
    ggml_tensor * pos_block = nullptr;
    ggml_tensor * neg_block = nullptr;
    ggml_tensor * pos_ctx = nullptr;
    ggml_tensor * out = nullptr;
    std::vector<std::pair<std::string, ggml_tensor *>> dbg_taps;
    // HC scales are immutable weights: read from the backend once.
    std::vector<std::array<float, 3>> s_attn, s_ffn;
    float s_out = 0.0f;
};

thread_local DsparkDraftCache g_dspark_draft_cache;

}  // namespace

bool deepseek4_dspark_draft_forward(ggml_backend_t backend,
                                    const DSparkDrafter & d,
                                    const float * noise_embed,
                                    const float * ctx_features,
                                    int ctx_len,
                                    int committed,
                                    std::vector<float> & out_hidden) {
    const DeepSeek4Weights & w = d.core;
    const int n_embd  = w.n_embd;
    const int n_hc    = w.n_hc;
    const int block   = d.block_size;
    const int fc_in   = d.n_target_layers * n_embd;
    const int mix_dim = 2 * n_hc + n_hc * n_hc;
    const float hc_eps = w.hc_eps;
    if (ctx_len < 0) ctx_len = 0;

    DsparkDraftCache & C = g_dspark_draft_cache;
    const bool DS4_DBG = std::getenv("DFLASH_DS4_DSPARK_DEBUG") != nullptr;

    if (C.drafter != (const void *) &d) {
        // HC scales (host) per layer + output — immutable, read once per drafter.
        C.s_attn.assign((size_t) w.n_layer, {0.0f, 0.0f, 0.0f});
        C.s_ffn.assign((size_t) w.n_layer, {0.0f, 0.0f, 0.0f});
        for (int il = 0; il < w.n_layer; il++) {
            ds4_read_f32(w.layers[il].hc_attn_scale, C.s_attn[il].data(), 3);
            ds4_read_f32(w.layers[il].hc_ffn_scale,  C.s_ffn[il].data(), 3);
        }
        float so[1] = {0.0f};
        ds4_read_f32(w.output_hc_scale, so, 1);
        C.s_out = so[0];
    }

    if (!C.ctx || C.ctx_len != ctx_len || C.block != block || C.drafter != (const void *) &d) {
        // ── (Re)build the graph ─────────────────────────────────────────
        if (C.ctx) { ggml_free(C.ctx); C.ctx = nullptr; }
        C.gf = nullptr;
        C.dbg_taps.clear();
        if (C.arena.empty()) C.arena.resize(256u * 1024 * 1024);
        ggml_init_params ip{};
        ip.mem_size = C.arena.size();
        ip.mem_buffer = C.arena.data();
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) return false;
        C.ctx = ctx;
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);
        C.gf = gf;
        auto dbg_tap = [&](const std::string & nm, ggml_tensor * t) {
            if (!DS4_DBG || !t) return;
            ggml_set_output(t);
            ggml_build_forward_expand(gf, t);
            C.dbg_taps.push_back({nm, t});
        };

        // Inputs.
        C.inp_noise = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, block);
        ggml_set_input(C.inp_noise);
        C.inp_ctx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, fc_in, ctx_len > 0 ? ctx_len : 1);
        ggml_set_input(C.inp_ctx);
        C.pos_block = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, block);
        ggml_set_input(C.pos_block);
        C.neg_block = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, block);
        ggml_set_input(C.neg_block);
        C.pos_ctx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ctx_len > 0 ? ctx_len : 1);
        ggml_set_input(C.pos_ctx);

        // main_x = main_norm(main_proj(ctx_features)).  Shared across layers.
        ggml_tensor * main_x = nullptr;
        if (ctx_len > 0) {
            // Captured target features have large magnitude (rms ~1e3 — HC streams
            // accumulate over 40+ layers). main_proj is rocmfp4-quantized and its
            // activation quantization overflows on inputs that big -> NaN. Since
            // main_norm (RMSNorm) normalizes main_proj's output and RMSNorm is
            // scale-invariant, pre-normalizing the features to unit RMS gives a
            // mathematically identical main_x while keeping the rocmfp4 activation
            // in a safe range: main_norm(main_proj(f/rms(f))) == main_norm(main_proj(f)).
            ggml_tensor * fc_in_normed = ggml_rms_norm(ctx, C.inp_ctx, w.rms_eps);
            ggml_tensor * fc_out = ggml_mul_mat(ctx, d.main_proj, fc_in_normed);   // [n_embd, ctx_len]
            dbg_tap("fc_out", fc_out);
            main_x = build_rms_norm(ctx, fc_out, d.main_norm, w.rms_eps);
            dbg_tap("main_x", main_x);
        }

        // HC state: [n_embd, n_hc, block], init = block embeds replicated over streams.
        ggml_tensor * noise3 = ggml_reshape_3d(ctx, C.inp_noise, n_embd, 1, block);
        ggml_tensor * hc_cur = ggml_repeat_4d(ctx, noise3, n_embd, n_hc, block, 1);

        auto hc_col = [&](ggml_tensor * hc, int p) -> ggml_tensor * {
            // Contiguous [n_embd*n_hc] slab for block position p.
            return ggml_view_1d(ctx, hc, (int64_t) n_embd * n_hc,
                                (size_t) p * hc->nb[2]);
        };

        for (int il = 0; il < w.n_layer; il++) {
            const DeepSeek4Layer & L = w.layers[il];

            // ── HC pre (attention), per block position ──────────────────
            std::vector<ggml_tensor *> split_attn(block), work_cols(block);
            for (int p = 0; p < block; p++) {
                ggml_tensor * hcf = hc_col(hc_cur, p);
                ggml_tensor * normed = ggml_rms_norm(ctx, hcf, hc_eps);
                ggml_tensor * mix = ggml_mul_mat(ctx, L.hc_attn_fn, normed);
                mix = ggml_reshape_1d(ctx, mix, mix_dim);
                ggml_tensor * base = ggml_reshape_1d(ctx, L.hc_attn_base, mix_dim);
                ggml_tensor * pre = ggml_ds4_hc_pre(ctx, mix, base, hcf, n_hc,
                                                    w.n_hc_sinkhorn_iter,
                                                    C.s_attn[il][0], C.s_attn[il][1], C.s_attn[il][2]);
                work_cols[p]  = ggml_reshape_2d(ctx, ggml_view_1d(ctx, pre, n_embd, 0), n_embd, 1);
                split_attn[p] = ggml_view_1d(ctx, pre, mix_dim, (size_t) n_embd * sizeof(float));
            }
            ggml_tensor * attn_in = work_cols[0];
            for (int p = 1; p < block; p++) attn_in = ggml_concat(ctx, attn_in, work_cols[p], 1);
            ggml_tensor * attn_normed = build_rms_norm(ctx, attn_in, L.attn_norm, w.rms_eps);
            ggml_tensor * attn_out = build_dspark_attention(ctx, attn_normed, main_x, w, L,
                                                            ctx_len, C.pos_block, C.neg_block, C.pos_ctx);
            dbg_tap(std::string("attn_L") + std::to_string(il), attn_out);
            // ── HC post (attention), per block position ─────────────────
            ggml_tensor * hc_next = nullptr;
            for (int p = 0; p < block; p++) {
                ggml_tensor * bo = ggml_view_1d(ctx, attn_out, n_embd, (size_t) p * attn_out->nb[1]);
                ggml_tensor * hp = ggml_ds4_hc_post(ctx, hc_col(hc_cur, p), bo, split_attn[p], n_hc);
                hp = ggml_reshape_3d(ctx, hp, n_embd, n_hc, 1);
                hc_next = hc_next ? ggml_concat(ctx, hc_next, hp, 2) : hp;
            }
            hc_cur = ggml_cont(ctx, hc_next);

            // ── HC pre (FFN), per block position ────────────────────────
            std::vector<ggml_tensor *> split_ffn(block), fwork(block);
            for (int p = 0; p < block; p++) {
                ggml_tensor * hcf = hc_col(hc_cur, p);
                ggml_tensor * normed = ggml_rms_norm(ctx, hcf, hc_eps);
                ggml_tensor * mix = ggml_mul_mat(ctx, L.hc_ffn_fn, normed);
                mix = ggml_reshape_1d(ctx, mix, mix_dim);
                ggml_tensor * base = ggml_reshape_1d(ctx, L.hc_ffn_base, mix_dim);
                ggml_tensor * pre = ggml_ds4_hc_pre(ctx, mix, base, hcf, n_hc,
                                                    w.n_hc_sinkhorn_iter,
                                                    C.s_ffn[il][0], C.s_ffn[il][1], C.s_ffn[il][2]);
                fwork[p]     = ggml_reshape_2d(ctx, ggml_view_1d(ctx, pre, n_embd, 0), n_embd, 1);
                split_ffn[p] = ggml_view_1d(ctx, pre, mix_dim, (size_t) n_embd * sizeof(float));
            }
            ggml_tensor * ffn_in = fwork[0];
            for (int p = 1; p < block; p++) ffn_in = ggml_concat(ctx, ffn_in, fwork[p], 1);
            ggml_tensor * ffn_normed = build_rms_norm(ctx, ffn_in, L.ffn_norm, w.rms_eps);
            ggml_tensor * ffn_out = build_moe_ffn(ctx, ffn_normed, w, L, il, block);
            if (!ffn_out) { ggml_free(C.ctx); C.ctx = nullptr; C.gf = nullptr; return false; }
            dbg_tap(std::string("ffn_L") + std::to_string(il), ffn_out);
            // ── HC post (FFN) ───────────────────────────────────────────
            hc_next = nullptr;
            for (int p = 0; p < block; p++) {
                ggml_tensor * bo = ggml_view_1d(ctx, ffn_out, n_embd, (size_t) p * ffn_out->nb[1]);
                ggml_tensor * hp = ggml_ds4_hc_post(ctx, hc_col(hc_cur, p), bo, split_ffn[p], n_hc);
                hp = ggml_reshape_3d(ctx, hp, n_embd, n_hc, 1);
                hc_next = hc_next ? ggml_concat(ctx, hc_next, hp, 2) : hp;
            }
            hc_cur = ggml_cont(ctx, hc_next);
            dbg_tap(std::string("hcL") + std::to_string(il), hc_cur);
        }

        // ── Tail: hc_head collapse -> out_norm, per block position ──────
        ggml_tensor * out = nullptr;
        for (int p = 0; p < block; p++) {
            ggml_tensor * hcf = hc_col(hc_cur, p);
            ggml_tensor * onorm = ggml_rms_norm(ctx, hcf, hc_eps);
            ggml_tensor * omix = ggml_mul_mat(ctx, w.output_hc_fn, onorm);
            omix = ggml_reshape_1d(ctx, omix, n_hc);
            ggml_tensor * obase = ggml_reshape_1d(ctx, w.output_hc_base, n_hc);
            ggml_tensor * final_embd = ggml_ds4_hc_out(ctx, omix, obase, hcf, n_hc, C.s_out);
            ggml_tensor * final_2d = ggml_reshape_2d(ctx, final_embd, n_embd, 1);
            ggml_tensor * hidden_p = build_rms_norm(ctx, final_2d, w.out_norm, w.rms_eps);
            out = out ? ggml_concat(ctx, out, hidden_p, 1) : hidden_p;
        }
        ggml_set_output(out);
        ggml_build_forward_expand(gf, out);
        C.out = out;

        if (!C.alloc) C.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!C.alloc || !ggml_gallocr_alloc_graph(C.alloc, gf)) {
            ggml_free(C.ctx); C.ctx = nullptr; C.gf = nullptr;
            return false;
        }
        C.ctx_len = ctx_len;
        C.block   = block;
        C.drafter = (const void *) &d;
    }

    // ── Set inputs + compute (cached graph) ─────────────────────────────
    ggml_backend_tensor_set(C.inp_noise, noise_embed, 0, sizeof(float) * (size_t) n_embd * block);
    if (ctx_len > 0) {
        ggml_backend_tensor_set(C.inp_ctx, ctx_features, 0, sizeof(float) * (size_t) fc_in * ctx_len);
        std::vector<int32_t> pc(ctx_len);
        for (int i = 0; i < ctx_len; i++) pc[i] = committed - ctx_len + i;
        ggml_backend_tensor_set(C.pos_ctx, pc.data(), 0, sizeof(int32_t) * ctx_len);
    }
    std::vector<int32_t> pb(block), nb(block);
    for (int i = 0; i < block; i++) { pb[i] = committed + i; nb[i] = -(committed + i); }
    ggml_backend_tensor_set(C.pos_block, pb.data(), 0, sizeof(int32_t) * block);
    ggml_backend_tensor_set(C.neg_block, nb.data(), 0, sizeof(int32_t) * block);

    const ggml_status st = ggml_backend_graph_compute(backend, C.gf);
    if (st != GGML_STATUS_SUCCESS) {
        // Invalidate: a failed compute leaves no reusable state guarantees.
        ggml_free(C.ctx); C.ctx = nullptr; C.gf = nullptr; C.ctx_len = -1;
        return false;
    }
    out_hidden.resize((size_t) n_embd * block);
    ggml_backend_tensor_get(C.out, out_hidden.data(), 0, sizeof(float) * out_hidden.size());

    if (DS4_DBG) {
        for (auto & tp : C.dbg_taps) {
            const size_t ne = ggml_nelements(tp.second);
            std::vector<float> buf(ne);
            ggml_backend_tensor_get(tp.second, buf.data(), 0, sizeof(float) * ne);
            double ss = 0.0; size_t nnan = 0; float mn = 1e30f, mx = -1e30f;
            for (float v : buf) {
                if (!std::isfinite(v)) { nnan++; }
                else { ss += (double) v * v; if (v < mn) mn = v; if (v > mx) mx = v; }
            }
            std::fprintf(stderr, "[ds4-dspark-dbg] %-10s ne=%zu nnan=%zu rms=%.4f min=%.3f max=%.3f\n",
                         tp.first.c_str(), ne, nnan, std::sqrt(ss / (double) ne), mn, mx);
        }
    }

    return true;
}

}  // namespace dflash::common
