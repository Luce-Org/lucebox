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
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
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

static int ds4_effective_expert_count(const DeepSeek4Weights & w) {
    const int requested = w.routed_expert_top_k;
    if (requested > 0 && requested < w.n_expert_used) {
        return requested;
    }
    return w.n_expert_used;
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

int deepseek4_previous_raw_ring_spans(
        int kv_start,
        int n_swa,
        DeepSeek4RawRingSpan spans[2]) {
    if (!spans || kv_start <= 0 || n_swa <= 0) {
        return 0;
    }
    if (kv_start < n_swa) {
        spans[0] = {0, kv_start};
        return 1;
    }

    const int current_row = kv_start % n_swa;
    int count = 0;
    const int tail_count = n_swa - current_row - 1;
    if (tail_count > 0) {
        spans[count++] = {current_row + 1, tail_count};
    }
    if (current_row > 0) {
        spans[count++] = {0, current_row};
    }
    return count;
}

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

// Attention implementation selected by the DS4 prefill scheduler. Decode
// retains the established explicit reduction path.
enum class DeepSeek4AttentionImpl {
    Explicit,
    DenseFlash,
    SparseFlash,
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
    int n_expert_used = 0;
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
        n_expert_used = 0;
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
    // Fused-decode stable-KV path only: additive score mask over
    // [n_swa raw rows ++ padded comp rows]; 0 for valid, -1e30 for padding.
    ggml_tensor * attn_row_mask = nullptr;
    int           padded_comp = 0;   // padded compressed-row count (>= n_comp)
    // Optional stable-topology compressor rows. DSpark's fused verifier leaves
    // this null and uses its batched state-row inputs instead.
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

    void clear() {
        *this = {};
    }

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
        const int n_used = ds4_effective_expert_count(w);
        out.hash_ids = ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_I32, n_used, n_tokens);
        ggml_set_input(out.hash_ids);

        ggml_tensor * shared_out = build_shared_ffn(out.sg.ctx, ffn_normed, w, L);
        ggml_tensor * logits = ggml_mul_mat(out.sg.ctx, L.ffn_gate_inp, ffn_normed);
        ggml_tensor * probs = ggml_sqrt(out.sg.ctx, ggml_softplus(out.sg.ctx, logits));

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
        routed_out = ggml_cont(
            out.sg.ctx, ggml_permute(out.sg.ctx, routed_out, 1, 0, 2, 3));
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
    out.n_expert_used = ds4_effective_expert_count(w);
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

int deepseek4_safe_compressor_batch_tokens(const DeepSeek4Weights & w,
                                           int kv_start,
                                           int n_tokens) {
    if (n_tokens <= 0) return 0;
    int safe = n_tokens;
    for (uint32_t raw_ratio : w.compress_ratios) {
        const int ratio = (int) raw_ratio;
        if (ratio <= 0) continue;
        int pos_mod = kv_start % ratio;
        if (pos_mod < 0) pos_mod += ratio;
        safe = std::min(safe, ratio - pos_mod);
    }
    return std::max(1, safe);
}

// Build an exact multi-token compressor update for prefill. Complete windows
// are pooled as one batched tensor, so a 2K ubatch does not create hundreds of
// serial softmax subgraphs. The state is assembled functionally from an
// initial snapshot and written back once, avoiding persistent-buffer races.
static bool build_compressor_prefill(
        ggml_context * ctx,
        ggml_cgraph * gf,
        ggml_tensor * cur_all,
        ggml_tensor * ape,
        ggml_tensor * kv_proj,
        ggml_tensor * gate_proj,
        ggml_tensor * norm_weight,
        DeepSeek4CompressorState & state,
        ggml_tensor * comp_cache,
        int ratio,
        int head_dim,
        int kv_start,
        int n_tokens,
        int n_rot,
        float rms_eps,
        float compress_rope_freq_base,
        float rope_scale_factor,
        float rope_yarn_beta_fast,
        float rope_yarn_beta_slow,
        int rope_orig_ctx,
        std::vector<DeepSeek4I64ArrayBinding> & i64_array_inputs,
        std::vector<DeepSeek4I32ArrayBinding> & i32_array_inputs,
        ggml_tensor ** comp_cache_source_out,
        bool indexer_qat) {
    if (!cur_all || n_tokens <= 1 ||
        n_tokens > DS4_MAX_LAYER_MAJOR_PREFILL_TOKENS ||
        (ratio != 4 && ratio != 128)) {
        return false;
    }

    const int coff = ratio == 4 ? 2 : 1;
    const int comp_width = coff * head_dim;
    const int n_state_rows = ratio == 4 ? 2 * ratio : ratio;

    struct Pair {
        ggml_tensor * kv = nullptr;
        ggml_tensor * score = nullptr;
    };

    auto view_cols = [&](ggml_tensor * src, int width, int first, int count) {
        GGML_ASSERT(src && count > 0);
        return ggml_cont(ctx, ggml_view_2d(ctx, src, width, count, src->nb[1],
                                           (size_t) first * src->nb[1]));
    };
    auto view_pair_cols = [&](const Pair & src, int first, int count) {
        return Pair { view_cols(src.kv, comp_width, first, count),
                      view_cols(src.score, comp_width, first, count) };
    };
    auto concat_tensors = [&](const std::vector<ggml_tensor *> & parts) {
        GGML_ASSERT(!parts.empty());
        ggml_tensor * out = parts[0];
        for (size_t i = 1; i < parts.size(); ++i) {
            out = ggml_concat(ctx, out, parts[i], 1);
        }
        return ggml_cont(ctx, out);
    };
    auto concat_pairs = [&](const std::vector<Pair> & parts) {
        std::vector<ggml_tensor *> kv_parts;
        std::vector<ggml_tensor *> score_parts;
        kv_parts.reserve(parts.size());
        score_parts.reserve(parts.size());
        for (const Pair & part : parts) {
            kv_parts.push_back(part.kv);
            score_parts.push_back(part.score);
        }
        return Pair { concat_tensors(kv_parts), concat_tensors(score_parts) };
    };
    auto replace_span = [&](const Pair & base,
                            int first,
                            const Pair & replacement,
                            int count,
                            int width) {
        std::vector<Pair> parts;
        if (first > 0) parts.push_back(view_pair_cols(base, 0, first));
        parts.push_back(replacement);
        if (first + count < width) {
            parts.push_back(view_pair_cols(base, first + count, width - first - count));
        }
        return concat_pairs(parts);
    };

    // Project the entire chunk once and add the position-addressed APE score.
    Pair projected;
    projected.kv = ggml_mul_mat(ctx, kv_proj, cur_all);
    projected.score = ggml_mul_mat(ctx, gate_proj, cur_all);
    ggml_tensor * ape_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(ape_rows);
    std::vector<int32_t> ape_values((size_t) n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        ape_values[(size_t) i] = (kv_start + i) % ratio;
    }
    i32_array_inputs.push_back({ape_rows, std::move(ape_values)});
    ggml_tensor * ape_cols = ggml_get_rows(ctx, ape, ape_rows);
    projected.score = ggml_add(ctx, projected.score,
                               ds4_cast_if_needed(ctx, ape_cols, GGML_TYPE_F32));

    // Snapshot before the single writeback.  Both compressor output and final
    // state depend on these copies, forcing reads to complete before mutation.
    Pair initial { ggml_cont(ctx, state.state_kv),
                   ggml_cont(ctx, state.state_score) };
    ggml_build_forward_expand(gf, initial.kv);
    ggml_build_forward_expand(gf, initial.score);

    ggml_tensor * pooled_batch = nullptr;
    std::vector<int64_t> comp_rows;
    std::vector<int32_t> comp_positions;

    auto pool_groups = [&](ggml_tensor * values_kv,
                           ggml_tensor * values_score,
                           int rows,
                           int groups) {
        GGML_ASSERT(values_kv && values_score && rows > 0 && groups > 0);
        ggml_tensor * kv3 = ggml_reshape_3d(ctx, values_kv,
                                            head_dim, rows, groups);
        ggml_tensor * score3 = ggml_reshape_3d(ctx, values_score,
                                               head_dim, rows, groups);
        ggml_tensor * score_t = ggml_cont(
            ctx, ggml_permute(ctx, score3, 1, 0, 2, 3));
        ggml_tensor * kv_t = ggml_cont(
            ctx, ggml_permute(ctx, kv3, 1, 0, 2, 3));
        ggml_tensor * probs_t = ggml_soft_max(ctx, score_t);
        ggml_tensor * weighted_t = ggml_mul(ctx, probs_t, kv_t);
        ggml_tensor * pooled_sum = ggml_sum_rows(ctx, weighted_t);
        ggml_tensor * pooled = ggml_reshape_2d(
            ctx, ggml_cont(ctx, pooled_sum), head_dim, groups);
        pooled = ggml_cont(ctx, pooled);
        pooled = build_rms_norm(ctx, pooled, norm_weight, rms_eps);
        return ggml_reshape_2d(ctx, pooled, head_dim, groups);
    };

    Pair final_state;
    if (ratio == 4) {
        const int pos_mod = kv_start % ratio;
        const int first_count = std::min(ratio - pos_mod, n_tokens);
        int consumed = 0;
        std::vector<Pair> complete_parts;

        if (n_tokens >= ratio - pos_mod) {
            Pair current = view_pair_cols(initial, ratio, ratio);
            Pair first_span = view_pair_cols(projected, 0, first_count);
            complete_parts.push_back(replace_span(
                current, pos_mod, first_span, first_count, ratio));
            consumed = first_count;

            const int complete_tail = ((n_tokens - consumed) / ratio) * ratio;
            if (complete_tail > 0) {
                complete_parts.push_back(view_pair_cols(
                    projected, consumed, complete_tail));
                consumed += complete_tail;
            }
        }

        if (complete_parts.empty()) {
            Pair prev = view_pair_cols(initial, 0, ratio);
            Pair current = view_pair_cols(initial, ratio, ratio);
            current = replace_span(current, pos_mod, projected,
                                   n_tokens, ratio);
            final_state = concat_pairs({prev, current});
        } else {
            Pair complete = concat_pairs(complete_parts);
            const int groups = (int) complete.kv->ne[1] / ratio;
            GGML_ASSERT(groups > 0);

            Pair previous = view_pair_cols(initial, 0, ratio);
            if (groups > 1) {
                previous = concat_pairs({
                    previous,
                    view_pair_cols(complete, 0, (groups - 1) * ratio),
                });
            }

            auto select_half = [&](ggml_tensor * src, int half) {
                ggml_tensor * src3 = ggml_reshape_3d(
                    ctx, src, comp_width, ratio, groups);
                return ggml_cont(ctx, ggml_view_3d(
                    ctx, src3, head_dim, ratio, groups,
                    src3->nb[1], src3->nb[2],
                    (size_t) half * head_dim * src3->nb[0]));
            };
            ggml_tensor * selected_kv = ggml_concat(
                ctx, select_half(previous.kv, 0),
                select_half(complete.kv, 1), 1);
            ggml_tensor * selected_score = ggml_concat(
                ctx, select_half(previous.score, 0),
                select_half(complete.score, 1), 1);
            pooled_batch = pool_groups(
                ggml_cont(ctx, selected_kv),
                ggml_cont(ctx, selected_score), 2 * ratio, groups);

            const int first_boundary = kv_start + first_count - 1;
            for (int g = 0; g < groups; ++g) {
                const int boundary = first_boundary + g * ratio;
                const int64_t comp_row = boundary / ratio;
                GGML_ASSERT(comp_row >= 0 && comp_row < comp_cache->ne[1]);
                comp_rows.push_back(comp_row);
                comp_positions.push_back(boundary + 1 - ratio);
            }

            Pair last_complete = view_pair_cols(
                complete, (groups - 1) * ratio, ratio);
            Pair current = last_complete;
            const int tail = n_tokens - consumed;
            if (tail > 0) {
                current = replace_span(
                    current, 0, view_pair_cols(projected, consumed, tail),
                    tail, ratio);
            }
            final_state = concat_pairs({last_complete, current});
        }
    } else {
        const int pos_mod = kv_start % ratio;
        const int to_boundary = ratio - pos_mod;
        if (n_tokens < to_boundary) {
            final_state = replace_span(initial, pos_mod, projected, n_tokens, ratio);
        } else {
            std::vector<Pair> first_parts;
            if (pos_mod > 0) {
                first_parts.push_back(view_pair_cols(initial, 0, pos_mod));
            }
            first_parts.push_back(view_pair_cols(projected, 0, to_boundary));
            Pair first_complete = concat_pairs(first_parts);

            int consumed = to_boundary;
            const int complete_tail = ((n_tokens - consumed) / ratio) * ratio;
            Pair complete = first_complete;
            if (complete_tail > 0) {
                complete = concat_pairs({
                    first_complete,
                    view_pair_cols(projected, consumed, complete_tail),
                });
                consumed += complete_tail;
            }
            const int groups = (int) complete.kv->ne[1] / ratio;
            pooled_batch = pool_groups(complete.kv, complete.score,
                                       ratio, groups);
            const int first_boundary = kv_start + to_boundary - 1;
            for (int g = 0; g < groups; ++g) {
                const int boundary = first_boundary + g * ratio;
                const int64_t comp_row = boundary / ratio;
                GGML_ASSERT(comp_row >= 0 && comp_row < comp_cache->ne[1]);
                comp_rows.push_back(comp_row);
                comp_positions.push_back(boundary + 1 - ratio);
            }

            Pair last_complete = view_pair_cols(
                complete, (groups - 1) * ratio, ratio);
            const int tail = n_tokens - consumed;
            if (tail > 0) {
                final_state = replace_span(
                    last_complete, 0,
                    view_pair_cols(projected, consumed, tail), tail, ratio);
            } else {
                final_state = last_complete;
            }
        }
    }

    // Persist the exact sequential state using unique row indices.
    ggml_tensor * state_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_state_rows);
    ggml_set_input(state_rows);
    std::vector<int64_t> state_row_values((size_t) n_state_rows);
    for (int i = 0; i < n_state_rows; ++i) state_row_values[(size_t) i] = i;
    i64_array_inputs.push_back({state_rows, std::move(state_row_values)});
    final_state.kv = ggml_cont(ctx, final_state.kv);
    final_state.score = ggml_cont(ctx, final_state.score);
    ggml_tensor * state_kv_source = ggml_set_rows(ctx, state.state_kv,
                                                   final_state.kv, state_rows);
    ggml_tensor * state_score_source = ggml_set_rows(ctx, state.state_score,
                                                      final_state.score, state_rows);
    ggml_build_forward_expand(gf, state_kv_source);
    ggml_build_forward_expand(gf, state_score_source);

    ggml_tensor * comp_cache_source = comp_cache;
    if (pooled_batch) {
        ggml_tensor * pooled = pooled_batch;
        const int n_pooled = (int) comp_positions.size();
        ggml_tensor * comp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,
                                                    n_pooled);
        ggml_set_input(comp_pos);
        i32_array_inputs.push_back({comp_pos, std::move(comp_positions)});

        const float rope_scale = rope_scale_factor > 0.0f
            ? (1.0f / rope_scale_factor) : 1.0f;
        float rope_attn = 1.0f;
        if (rope_scale > 0.0f) {
            rope_attn /= (1.0f + 0.1f * logf(1.0f / rope_scale));
        }
        pooled = build_tail_rope_2d(ctx, pooled, comp_pos, n_rot, head_dim,
                                    n_pooled,
                                    compress_rope_freq_base, rope_scale,
                                    1.0f, rope_attn,
                                    rope_yarn_beta_fast, rope_yarn_beta_slow,
                                    rope_orig_ctx);
        pooled = ggml_cont(ctx, pooled);
        if (indexer_qat) {
            pooled = ggml_ds4_indexer_qat(ctx, pooled);
        }

        ggml_tensor * comp_row_tensor = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I64, (int64_t) comp_rows.size());
        ggml_set_input(comp_row_tensor);
        i64_array_inputs.push_back({comp_row_tensor, std::move(comp_rows)});
        comp_cache_source = ggml_set_rows(ctx, comp_cache, pooled, comp_row_tensor);
        ggml_build_forward_expand(gf, comp_cache_source);
    }
    if (comp_cache_source_out) *comp_cache_source_out = comp_cache_source;
    return true;
}

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
        int kv_start_all = -1,
        bool indexer_qat = false) {
    if (!gf || !cur_last || !ape || !kv_proj || !gate_proj || !norm_weight ||
        !state.state_kv || !state.state_score || !comp_cache || ratio <= 0) {
        return;
    }

    // Multi-token speculative verification uses the boundary-split path below.
    // The layer-major prefill scheduler only enters here for wider batches.
    if (cur_all && n_tokens_all > 4 && !state_rows_inp && kv_start_all >= 0 &&
        build_compressor_prefill(ctx, gf, cur_all, ape, kv_proj, gate_proj,
                                 norm_weight, state, comp_cache, ratio, head_dim,
                                 kv_start_all, n_tokens_all, n_rot, rms_eps,
                                 compress_rope_freq_base, rope_scale_factor,
                                 rope_yarn_beta_fast, rope_yarn_beta_slow,
                                 rope_orig_ctx, i64_array_inputs,
                                 i32_array_inputs, comp_cache_source_out,
                                 indexer_qat)) {
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
    // position-addressed state row. deepseek4_step_layer_range splits dynamic
    // batches at compressor boundaries, so the boundary can only be the last
    // token and no post-boundary write can precede pooling/rotation.
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
        // State rows were written, but this batch did not complete a window.
        return;
    }
    if (!batched_rows && ((token_pos + 1) % ratio) != 0) {
        // Per-layer graphs only pool at flush boundaries.
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
    if (indexer_qat) {
        pooled = ggml_ds4_indexer_qat(ctx, ggml_cont(ctx, pooled));
    }

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
            // Rotate the completed current window into the previous half.
            // Reading through the first span makes the dependency explicit.
            for (int r = 0; r < ratio; ++r) {
                ggml_tensor * src_kv = ggml_view_2d(
                    ctx, state_kv_source, comp_width, 1,
                    state_kv_source->nb[1],
                    (size_t) (ratio + r) * state_kv_source->nb[1]);
                ggml_tensor * dst_kv = ggml_view_2d(
                    ctx, state.state_kv, comp_width, 1,
                    state.state_kv->nb[1],
                    (size_t) r * state.state_kv->nb[1]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, src_kv, dst_kv));

                ggml_tensor * src_sc = ggml_view_2d(
                    ctx, state_score_source, comp_width, 1,
                    state_score_source->nb[1],
                    (size_t) (ratio + r) * state_score_source->nb[1]);
                ggml_tensor * dst_sc = ggml_view_2d(
                    ctx, state.state_score, comp_width, 1,
                    state.state_score->nb[1],
                    (size_t) r * state.state_score->nb[1]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, src_sc, dst_sc));
            }
        }
        if (batched_nB > 0) {
            ggml_tensor * kv_v = ggml_cont(ctx, ggml_view_2d(
                ctx, batched_kv_all, comp_width, batched_nB,
                batched_kv_all->nb[1],
                (size_t) batched_span_off * batched_kv_all->nb[1]));
            ggml_tensor * sc_v = ggml_cont(ctx, ggml_view_2d(
                ctx, batched_sc_all, comp_width, batched_nB,
                batched_sc_all->nb[1],
                (size_t) batched_span_off * batched_sc_all->nb[1]));
            ggml_tensor * rows_v = ggml_view_1d(
                ctx, state_rows_inp, batched_nB,
                (size_t) batched_span_off * state_rows_inp->nb[0]);
            ggml_build_forward_expand(
                gf, ggml_set_rows(ctx, state.state_kv, kv_v, rows_v));
            ggml_build_forward_expand(
                gf, ggml_set_rows(ctx, state.state_score, sc_v, rows_v));
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
        ggml_tensor ** index_comp_cache_source_out = nullptr,
        ggml_tensor * flush_rows_inp = nullptr,
        ggml_tensor * cur_all = nullptr,
        int n_tokens_all = 1,
        int kv_start_all = -1,
        bool indexer_qat = false) {
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
                          index_comp_cache_source_out,
                          flush_rows_inp,
                          cur_all,
                          n_tokens_all,
                          kv_start_all,
                          indexer_qat);
}

static int ds4_comp_rows_used(const ggml_tensor * comp_cache, int n_cached, int ratio, int token_pos) {
    if (!comp_cache || ratio <= 0) {
        return 0;
    }
    // n_cached is the committed count before this graph.  A multi-token
    // prefill graph may cross several compressor boundaries, so derive the
    // live count from the query position rather than adding at most one row.
    const int through_position = (token_pos + 1) / ratio;
    return std::min(std::max(n_cached, through_position),
                    (int) comp_cache->ne[1]);
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

static ggml_tensor * build_indexer_topk(
        ggml_context * ctx,
        ggml_tensor * qr_norm,        // [n_lora_q, n_tokens]
        ggml_tensor * cur,            // [n_embd, n_tokens]
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L,
        ggml_tensor * index_comp_source,
        int n_comp,
        int kv_start,
        int n_tokens,
        ggml_tensor * rope_pos,
        std::vector<DeepSeek4I32ArrayBinding> & i32_array_inputs) {
    if (!qr_norm || !cur || !L.indexer_attn_q_b || !L.indexer_proj ||
        !index_comp_source || !rope_pos || n_tokens <= 0 ||
        n_comp <= w.n_indexer_top_k) {
        return nullptr;
    }

    const int n_indexer_head = w.n_indexer_head;
    const int head_dim = w.n_indexer_head_dim;
    const int top_k = std::min(n_comp, w.n_indexer_top_k);
    // A token with <= top_k visible compressed rows needs no ranking: selecting
    // [0,top_k) and retaining the ordinary causal mask is exactly equivalent.
    // Score only the suffix beginning with the first token that can see row
    // top_k. For a zero-prefix ratio-4 2K request this shrinks 2164 score rows
    // to just 113.
    const int first_scored = std::max(
        0, std::min(n_tokens, 4 * (top_k + 1) - 1 - kv_start));
    const int n_scored = n_tokens - first_scored;
    if (n_scored <= 0) return nullptr;

    auto token_slice = [&](ggml_tensor * input, int width) {
        if (first_scored == 0) return input;
        return ggml_view_2d(
            ctx, input, width, n_scored, input->nb[1],
            (size_t) first_scored * input->nb[1]);
    };
    qr_norm = token_slice(qr_norm, (int) qr_norm->ne[0]);
    cur = token_slice(cur, (int) cur->ne[0]);
    if (first_scored > 0) {
        rope_pos = ggml_view_1d(
            ctx, rope_pos, n_scored,
            (size_t) first_scored * rope_pos->nb[0]);
    }

    // Official ratio-4 indexer graph: q_a-normalized query projection, tail
    // RoPE, Hadamard+FP4 QAT, per-head scalar projection, ReLU dot products,
    // weighted head reduction and top-512 selection for every query token.
    ggml_tensor * index_q = ggml_mul_mat(ctx, L.indexer_attn_q_b, qr_norm);
    index_q = ggml_reshape_3d(
        ctx, index_q, head_dim, n_indexer_head, n_scored);

    const float rope_scale = w.rope_scale_factor > 0.0f
        ? (1.0f / w.rope_scale_factor) : 1.0f;
    float rope_attn = 1.0f;
    if (rope_scale > 0.0f) {
        rope_attn /= 1.0f + 0.1f * logf(1.0f / rope_scale);
    }
    index_q = build_tail_rope_3d(
        ctx, index_q, rope_pos, w.n_rot, head_dim, n_indexer_head,
        n_scored, w.compress_rope_freq_base, rope_scale, 1.0f,
        rope_attn, w.rope_yarn_beta_fast, w.rope_yarn_beta_slow,
        (int) w.rope_orig_ctx);
    index_q = ggml_ds4_indexer_qat(ctx, ggml_cont(ctx, index_q));

    ggml_tensor * head_weights = ggml_mul_mat(ctx, L.indexer_proj, cur);
    head_weights = ggml_scale(ctx, head_weights,
                              1.0f / std::sqrt((float) head_dim * (float) n_indexer_head));

    ggml_tensor * comp = ggml_view_2d(
        ctx, index_comp_source, head_dim, n_comp,
        index_comp_source->nb[1], 0);
    GGML_ASSERT(comp->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(comp));

    // Fuse comp^T@q, ReLU, per-head weighting and head reduction. The generic
    // graph would retain [n_comp,64,n_tokens] dots (about 2 GiB at 8K) before
    // reducing them; this operation stores only the final score matrix.
    ggml_tensor * scores = ggml_ds4_indexer_score(
        ctx, index_q, head_weights, comp, kv_start + first_scored, 4);
    ggml_tensor * selected = ggml_top_k(
        ctx, ggml_cont(ctx, scores), top_k);
    if (first_scored == 0) return selected;

    ggml_tensor * identity = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, top_k, first_scored);
    ggml_set_input(identity);
    std::vector<int32_t> identity_values((size_t) top_k * first_scored);
    for (int t = 0; t < first_scored; ++t) {
        for (int k = 0; k < top_k; ++k) {
            identity_values[(size_t) t * top_k + k] = k;
        }
    }
    i32_array_inputs.push_back({identity, std::move(identity_values)});
    return ggml_concat(ctx, identity, selected, 1);
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
        std::vector<DeepSeek4F32ArrayBinding> * f32_array_inputs = nullptr,
        DeepSeek4AttentionImpl attention_impl = DeepSeek4AttentionImpl::Explicit) {

    const int n_embd    = w.n_embd;
    const int head_dim  = w.head_dim;
    const int n_head    = w.n_head;
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

    // D=512 flash prefill can rotate Q's 64-d tail inside the exact attention
    // kernel. This avoids materializing cont(nope), cont(tail), rope(tail),
    // and concat(nope, tail) while retaining the same F32 rounding boundary.
    const bool fuse_q_rope = attention_impl != DeepSeek4AttentionImpl::Explicit &&
                             n_tokens > 1 && head_dim == 512 && n_rot == 64;
    if (!fuse_q_rope) {
        q = build_tail_rope_3d(ctx, q, rope_pos, n_rot, head_dim, n_head, n_tokens,
                               rope_freq, rope_scale, rope_ext, rope_attn,
                               w.rope_yarn_beta_fast, w.rope_yarn_beta_slow, rope_n_ctx_orig);
    }
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
    const bool layer_major_batch =
        causal_batch && attention_impl != DeepSeek4AttentionImpl::Explicit;
    ggml_tensor * old_rows_scratch = nullptr;
    int n_old_rows = 0;
    ggml_tensor * prior_rows_scratch = nullptr;
    int n_prior_rows = 0;
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
    } else if (causal_batch && !layer_major_batch) {
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
    } else if (layer_major_batch) {
        // Snapshot the chronological pre-chunk window before any ring writes.
        // Attention then consumes [prior F16 rows | current F32 rows], matching
        // the single-token path and avoiding an F16 round-trip for this chunk.
        n_prior_rows = std::min(kv_start, w.n_swa);
        if (n_prior_rows > 0) {
            const int first = kv_start < w.n_swa ? 0 : (kv_start % w.n_swa);
            const int tail = std::min(n_prior_rows, w.n_swa - first);
            auto snapshot_span = [&](int row, int count) {
                ggml_tensor * span = ggml_view_2d(
                    ctx, lc.raw_kv, head_dim, count, lc.raw_kv->nb[1],
                    (size_t) row * lc.raw_kv->nb[1]);
                return ggml_cont(ctx, span);
            };
            prior_rows_scratch = snapshot_span(first, tail);
            if (tail < n_prior_rows) {
                prior_rows_scratch = ggml_concat(
                    ctx, prior_rows_scratch,
                    snapshot_span(0, n_prior_rows - tail), 1);
                prior_rows_scratch = ggml_cont(ctx, prior_rows_scratch);
            }
            ggml_build_forward_expand(gf, prior_rows_scratch);
            prior_rows_scratch = ds4_cast_if_needed(
                ctx, prior_rows_scratch, GGML_TYPE_F32);
        }
    }

    // ── Store ALL KV rows in the raw SWA ring ─────────────────────
    // For decode (n_tokens=1): write single row. For prefill: write all rows.
    ggml_tensor * raw_kv_source = lc.raw_kv;
    ggml_tensor * raw_kv_rows = cached_inputs
        ? cached_inputs->raw_kv_rows
        : nullptr;
    if (raw_kv_rows) {
        ggml_tensor * kv_f32 = ggml_is_contiguous(kv) ? kv : ggml_cont(ctx, kv);
        raw_kv_source = ggml_set_rows(ctx, lc.raw_kv, kv_f32, raw_kv_rows);
        ggml_build_forward_expand(gf, raw_kv_source);
    } else {
        // The attention graph consumes the whole current ubatch directly.
        // Persist only its final SWA tail so every physical ring row is written
        // once, even when the ubatch is much larger than the 128-row ring.
        const int first_write = std::max(0, n_tokens - w.n_swa);
        for (int ti = first_write; ti < n_tokens; ti++) {
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

    ggml_tensor * index_comp_kv_source = lc.index_comp_kv;
    if (ratio == 4 && L.indexer_compressor_kv) {
        build_indexer_compressor_step(ctx, gf, cur_last, w, L, lc, token_pos,
                                      cached_inputs ? cached_inputs->index_ape_row : nullptr,
                                      cached_inputs ? cached_inputs->index_state_rows : nullptr,
                                      cached_inputs ? cached_inputs->index_comp_rows : nullptr,
                                      cached_inputs ? cached_inputs->index_comp_pos : nullptr,
                                      i64_array_inputs,
                                      i32_array_inputs,
                                      &index_comp_kv_source,
                                      cached_inputs ? cached_inputs->flush_rows : nullptr,
                                      (causal_batch || fused_causal) ? cur : nullptr,
                                      n_tokens,
                                      kv_start,
                                      attention_impl ==
                                          DeepSeek4AttentionImpl::SparseFlash);
    }

    // ── MLA Dot-Product Attention (SWA + compressed KV) ────────────
    // q: [head_dim, n_head, n_tokens] (after RoPE)
    // raw_kv: [head_dim, n_swa] F16 persistent ring buffer (single KV head, shared)
    // comp_kv: [head_dim, comp_cap] F16 compressed rows.
    // n_raw = min(kv_start + n_tokens, n_swa)
    const bool masked_kv = cached_inputs && cached_inputs->attn_row_mask;
    const int n_comp_live = (ratio > 0) ? ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio, token_pos) : 0;
    ggml_tensor * indexer_topk = nullptr;
    if (attention_impl == DeepSeek4AttentionImpl::SparseFlash &&
        ratio == 4 && f32_array_inputs) {
        const int n_index_comp = ds4_comp_rows_used(
            lc.index_comp_kv, lc.n_index_comp, 4, token_pos);
        indexer_topk = build_indexer_topk(
            ctx, qr, cur, w, L, index_comp_kv_source,
            n_index_comp, kv_start, n_tokens, rope_pos,
            i32_array_inputs);
    }
    // Stable path reads the full physical ring (masking not-yet-written slots)
    // and a padded compressed-row span; the plain path reads only valid rows.
    const int n_raw = masked_kv ? w.n_swa
                    : layer_major_batch ? n_prior_rows + n_tokens
                    : std::min(kv_start + n_tokens, w.n_swa);
    const int n_comp_attn = masked_kv ? cached_inputs->padded_comp : n_comp_live;
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
    } else if (layer_major_batch) {
        ggml_tensor * current = ds4_cast_if_needed(ctx, kv, GGML_TYPE_F32);
        kv_attn = prior_rows_scratch
            ? ggml_concat(ctx, prior_rows_scratch, current, 1)
            : current;
    } else if (n_tokens == 1) {
        ggml_tensor * cur_kv = ds4_cast_if_needed(ctx, kv, GGML_TYPE_F32);
        if (n_raw == w.n_swa && raw_kv_rows) {
            // Once the ring is full, use a stable physical row order. The
            // cached q=1 graph is first built at position n_swa-1 and then
            // reused across every wrap position, so chronological views baked
            // into that first topology become stale. Insert the current F32
            // KV at its runtime row in an F32 snapshot instead. The tokenwise
            // prefill helper takes the same branch and row ordering.
            ggml_tensor * ring = ggml_view_2d(
                ctx, lc.raw_kv, head_dim, w.n_swa, lc.raw_kv->nb[1], 0);
            ring = ds4_cast_if_needed(ctx, ring, GGML_TYPE_F32);
            kv_attn = ggml_set_rows(ctx, ring, cur_kv, raw_kv_rows);
            ggml_build_forward_expand(gf, kv_attn);
        } else if (n_raw > 1) {
            ggml_tensor * prev = nullptr;
            DeepSeek4RawRingSpan spans[2];
            const int n_spans =
                deepseek4_previous_raw_ring_spans(kv_start, w.n_swa, spans);
            for (int i = 0; i < n_spans; ++i) {
                ggml_tensor * span = raw_kv_view(spans[i].row, spans[i].count);
                prev = prev ? ggml_concat(ctx, prev, span, 1) : span;
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

    // Build one additive mask tensor and share it between the explicit and
    // flash-attention implementations. ggml flash attention expects
    // [n_kv,n_query] F16; the explicit path broadcasts the same values over
    // heads in F32.
    ggml_tensor * score_mask = nullptr;
    const bool exact_two_band =
        attention_impl == DeepSeek4AttentionImpl::DenseFlash &&
        causal_batch &&
        n_tokens > DS4_NUMERICAL_PREFILL_BAND &&
        n_tokens <= 2 * DS4_NUMERICAL_PREFILL_BAND;
    if (!exact_two_band) {
        if (masked_kv && n_tokens > 1) {
            score_mask = ggml_reshape_2d(ctx, cached_inputs->attn_row_mask,
                                         n_attn, n_tokens);
        } else if (masked_kv) {
            score_mask = ggml_reshape_2d(ctx, cached_inputs->attn_row_mask,
                                         n_attn, 1);
        } else if (layer_major_batch) {
            // Per-token causal mask over [prior rows | current rows | comp rows].
            ggml_tensor * cmask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_attn, 1, n_tokens);
            ggml_set_input(cmask);
            std::vector<float> mvals((size_t) n_attn * n_tokens, 0.0f);
            for (int i = 0; i < n_tokens; i++) {
                const int pos_i = kv_start + i;
                float * col = mvals.data() + (size_t) i * n_attn;
                const int min_pos = pos_i - w.n_swa + 1;
                for (int r = 0; r < n_prior_rows; ++r) {
                    const int prior_pos = kv_start - n_prior_rows + r;
                    if (prior_pos < min_pos) col[r] = -1e30f;
                }
                for (int t = 0; t < n_tokens; ++t) {
                    const int current_pos = kv_start + t;
                    if (t > i || current_pos < min_pos) {
                        col[n_prior_rows + t] = -1e30f;
                    }
                }
                if (n_comp_attn > 0) {
                    const int vis = ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio, pos_i);
                    for (int c = vis; c < n_comp_attn; c++) col[n_raw + c] = -1e30f;
                }
            }
            f32_array_inputs->push_back({cmask, std::move(mvals)});
            score_mask = ggml_reshape_2d(ctx, cmask, n_attn, n_tokens);
        } else if (causal_batch) {
            // Speculative verification keeps the physical ring order and
            // appends snapshots of rows overwritten by later batch tokens.
            ggml_tensor * cmask = ggml_new_tensor_3d(
                ctx, GGML_TYPE_F32, n_attn, 1, n_tokens);
            ggml_set_input(cmask);
            std::vector<float> mvals((size_t) n_attn * n_tokens, 0.0f);
            const int end = kv_start + n_tokens;
            for (int i = 0; i < n_tokens; ++i) {
                const int pos_i = kv_start + i;
                float * col = mvals.data() + (size_t) i * n_attn;
                for (int r = 0; r < n_raw; ++r) {
                    const int pos_r = end <= w.n_swa
                        ? r
                        : (end - 1) - ((end - 1 - r) % w.n_swa);
                    if (pos_r > pos_i) col[r] = -1e30f;
                }
                if (n_comp_attn > 0) {
                    const int visible = ds4_comp_rows_used(
                        lc.comp_kv, lc.n_comp, ratio, pos_i);
                    for (int c = visible; c < n_comp_attn; ++c) {
                        col[n_raw + c] = -1e30f;
                    }
                }
                int old_row = 0;
                for (int t = 0; t < n_tokens; ++t) {
                    if (kv_start + t < w.n_swa) continue;
                    if (t <= i) {
                        col[n_raw + n_comp_attn + old_row] = -1e30f;
                    }
                    ++old_row;
                }
            }
            f32_array_inputs->push_back({cmask, std::move(mvals)});
            score_mask = ggml_reshape_2d(ctx, cmask, n_attn, n_tokens);
        }
    }
    if (indexer_topk) {
        if (!score_mask) {
            score_mask = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, n_attn, n_tokens);
            ggml_set_input(score_mask);
            f32_array_inputs->push_back({
                score_mask,
                std::vector<float>((size_t) n_attn * n_tokens, 0.0f),
            });
        }
        score_mask = ggml_ds4_indexer_mask(
            ctx, ggml_cont(ctx, score_mask), indexer_topk, n_raw);
    }
    ggml_tensor * context = nullptr;
    bool inverse_rope_fused = false;
    const bool use_flash = attention_impl != DeepSeek4AttentionImpl::Explicit &&
                           n_tokens > 1;
    if (use_flash) {
        if (exact_two_band) {
            // A larger scheduling band must retain the numerical topology of
            // two 2K requests. Prefix queries use the first band's F32 raw KV;
            // suffix queries see its final SWA tail after the same F16 cache
            // round-trip. HC, projections and MoE still run once over the full
            // token batch, avoiding a second expert-weight sweep.
            const int first_count = DS4_NUMERICAL_PREFILL_BAND;
            const int second_count = n_tokens - first_count;
            const int first_comp = ratio > 0
                ? ds4_comp_rows_used(lc.comp_kv, lc.n_comp, ratio,
                                     kv_start + first_count - 1)
                : 0;
            const int second_comp = n_comp_live;
            const int second_prior_count = std::min(first_count, w.n_swa);

            auto view_kv = [&](int first, int count) {
                return ggml_view_2d(
                    ctx, kv, head_dim, count, kv->nb[1],
                    (size_t) first * kv->nb[1]);
            };
            auto append_comp = [&](ggml_tensor * raw, int count) {
                if (count <= 0 || !comp_kv_source) return raw;
                ggml_tensor * comp = ggml_view_2d(
                    ctx, comp_kv_source, head_dim, count,
                    comp_kv_source->nb[1], 0);
                comp = ds4_cast_if_needed(ctx, comp, GGML_TYPE_F32);
                return ggml_concat(ctx, raw, comp, 1);
            };
            auto make_band_mask = [&](int start, int count, int prior,
                                      int comp_count) {
                const int raw_count = prior + count;
                const int attn_count = raw_count + comp_count;
                ggml_tensor * mask3 = ggml_new_tensor_3d(
                    ctx, GGML_TYPE_F32, attn_count, 1, count);
                ggml_set_input(mask3);
                std::vector<float> values(
                    (size_t) attn_count * count, 0.0f);
                for (int i = 0; i < count; ++i) {
                    const int pos_i = start + i;
                    const int min_pos = pos_i - w.n_swa + 1;
                    float * col = values.data() + (size_t) i * attn_count;
                    for (int r = 0; r < prior; ++r) {
                        const int prior_pos = start - prior + r;
                        if (prior_pos < min_pos) col[r] = -1e30f;
                    }
                    for (int t = 0; t < count; ++t) {
                        const int current_pos = start + t;
                        if (t > i || current_pos < min_pos) {
                            col[prior + t] = -1e30f;
                        }
                    }
                    if (comp_count > 0) {
                        const int visible = ds4_comp_rows_used(
                            lc.comp_kv, lc.n_comp, ratio, pos_i);
                        for (int c = visible; c < comp_count; ++c) {
                            col[raw_count + c] = -1e30f;
                        }
                    }
                }
                f32_array_inputs->push_back({mask3, std::move(values)});
                return ggml_reshape_2d(ctx, mask3, attn_count, count);
            };
            auto make_flash = [&](ggml_tensor * q_band,
                                  ggml_tensor * kv_band,
                                  ggml_tensor * mask_band,
                                  int raw_count,
                                  int start) {
                const int attn_count = (int) kv_band->ne[1];
                ggml_tensor * k_band = ggml_reshape_3d(
                    ctx, kv_band, head_dim, attn_count, 1);
                ggml_tensor * mask_fa = ds4_cast_if_needed(
                    ctx, mask_band, GGML_TYPE_F16);
                ggml_tensor * result = ggml_flash_attn_ext(
                    ctx, q_band, k_band, k_band, mask_fa,
                    kq_scale, 0.0f, 0.0f);
                if (L.attn_sinks) {
                    ggml_flash_attn_ext_add_sinks(result, L.attn_sinks);
                }
                ggml_flash_attn_ext_set_prec(result, GGML_PREC_F32);
                ggml_flash_attn_ext_set_ds4_sparse(
                    result, raw_count, w.n_swa, 0, 32);
                ggml_flash_attn_ext_set_ds4_inverse_rope(
                    result, start, rope_freq, rope_scale, rope_ext,
                    rope_attn, w.rope_yarn_beta_fast,
                    w.rope_yarn_beta_slow, rope_n_ctx_orig, fuse_q_rope);
                return result;
            };

            ggml_tensor * q_fa = ggml_permute(ctx, q, 0, 2, 1, 3);
            auto view_q = [&](int first, int count) {
                return ggml_view_3d(
                    ctx, q_fa, head_dim, count, n_head,
                    q_fa->nb[1], q_fa->nb[2],
                    (size_t) first * q_fa->nb[1]);
            };

            ggml_tensor * first_raw = ds4_cast_if_needed(
                ctx, view_kv(0, first_count), GGML_TYPE_F32);
            if (prior_rows_scratch) {
                first_raw = ggml_concat(
                    ctx, prior_rows_scratch, first_raw, 1);
            }
            ggml_tensor * first_kv = append_comp(first_raw, first_comp);
            ggml_tensor * first_mask = make_band_mask(
                kv_start, first_count, n_prior_rows, first_comp);
            ggml_tensor * first_context = make_flash(
                view_q(0, first_count), first_kv, first_mask,
                n_prior_rows + first_count, kv_start);

            ggml_tensor * rounded_prior = ggml_cast(
                ctx, view_kv(first_count - second_prior_count,
                             second_prior_count),
                GGML_TYPE_F16);
            rounded_prior = ggml_cast(ctx, rounded_prior, GGML_TYPE_F32);
            ggml_tensor * second_raw = ggml_concat(
                ctx, rounded_prior, view_kv(first_count, second_count), 1);
            ggml_tensor * second_kv = append_comp(second_raw, second_comp);
            ggml_tensor * second_mask = make_band_mask(
                kv_start + first_count, second_count,
                second_prior_count, second_comp);
            ggml_tensor * second_context = make_flash(
                view_q(first_count, second_count), second_kv, second_mask,
                second_prior_count + second_count,
                kv_start + first_count);

            context = ggml_concat(ctx, first_context, second_context, 2);
            inverse_rope_fused = true;
        } else {
            // ggml FA convention: Q[D,T,H], K/V[D,K,Hkv]. DS4 MLA has one shared
            // latent KV head and uses the same latent vector as both key and value.
            // The DS4 D=512 kernel consumes Q strides directly, avoiding a full
            // [D,H,T] -> [D,T,H] materialization for every layer.
            ggml_tensor * q_fa = ggml_permute(ctx, q, 0, 2, 1, 3);
            ggml_tensor * kv_fa = ds4_cast_if_needed(ctx, kv_attn, GGML_TYPE_F32);
            ggml_tensor * k_fa = ggml_reshape_3d(ctx, kv_fa, head_dim, n_attn, 1);
            ggml_tensor * v_fa = k_fa;
            ggml_tensor * mask_fa = score_mask
                ? ds4_cast_if_needed(ctx, score_mask, GGML_TYPE_F16)
                : nullptr;
            context = ggml_flash_attn_ext(ctx, q_fa, k_fa, v_fa, mask_fa,
                                          kq_scale, 0.0f, 0.0f);
            if (L.attn_sinks) {
                ggml_flash_attn_ext_add_sinks(context, L.attn_sinks);
            }
            ggml_flash_attn_ext_set_prec(context, GGML_PREC_F32);
            // Always publish the raw/compressed boundary. A zero keep count leaves
            // dense attention unchanged while allowing the D=512 value pass to
            // skip the two masked envelopes without guessing DS4 cache layout.
            ggml_flash_attn_ext_set_ds4_sparse(
                context, n_raw, w.n_swa,
                indexer_topk
                    ? -w.n_indexer_top_k
                    : attention_impl == DeepSeek4AttentionImpl::SparseFlash
                        ? w.n_indexer_top_k : 0,
                32);
            if (attention_impl != DeepSeek4AttentionImpl::Explicit &&
                head_dim == 512 && n_rot == 64) {
                ggml_flash_attn_ext_set_ds4_inverse_rope(
                    context, kv_start, rope_freq, rope_scale, rope_ext,
                    rope_attn, w.rope_yarn_beta_fast,
                    w.rope_yarn_beta_slow, rope_n_ctx_orig, fuse_q_rope);
                inverse_rope_fused = true;
            }
        }
    } else {
        // Flatten q to [head_dim, n_head*n_tokens] for batched matmul.
        ggml_tensor * q_flat = ggml_reshape_2d(ctx, q, head_dim,
                                               n_head * n_tokens);
        ggml_tensor * scores = ggml_mul_mat(ctx, kv_attn, q_flat);
        scores = ggml_scale(ctx, scores, kq_scale);
        if (score_mask) {
            if (n_tokens > 1) {
                ggml_tensor * m3 = ggml_reshape_3d(ctx, score_mask,
                                                   n_attn, 1, n_tokens);
                ggml_tensor * s3 = ggml_reshape_3d(ctx, scores,
                                                   n_attn, n_head, n_tokens);
                scores = ggml_reshape_2d(ctx, ggml_add(ctx, s3, m3),
                                         n_attn, n_head * n_tokens);
            } else {
                scores = ggml_add(ctx, scores, score_mask);
            }
        }

        // DS4 adds one learned per-head sink logit to the denominator, but the
        // sink contributes no value vector.
        ggml_tensor * probs = nullptr;
        if (L.attn_sinks) {
            ggml_tensor * sink_scores = ggml_reshape_2d(ctx, L.attn_sinks,
                                                        1, n_head);
            if (n_tokens > 1) {
                ggml_tensor * sink_shape = ggml_new_tensor_2d(
                    ctx, GGML_TYPE_F32, 1, n_head * n_tokens);
                sink_scores = ggml_repeat(ctx, sink_scores, sink_shape);
            }
            ggml_tensor * scores_with_sink = ggml_concat(ctx, scores,
                                                          sink_scores, 0);
            ggml_tensor * probs_with_sink = ggml_soft_max(ctx,
                                                           scores_with_sink);
            probs = ggml_view_2d(ctx, probs_with_sink, n_attn,
                                 n_head * n_tokens, probs_with_sink->nb[1], 0);
        } else {
            probs = ggml_soft_max(ctx, scores);
        }
        ggml_tensor * kv_t = ggml_cont(ctx, ggml_transpose(ctx, kv_attn));
        context = ggml_mul_mat(ctx, kv_t, probs);
        context = ggml_reshape_3d(ctx, context, head_dim, n_head, n_tokens);
    }

    // ── Inverse tail RoPE on attention output ───────────────────────
    if (!inverse_rope_fused) {
        ggml_tensor * neg_pos = cached_inputs ? cached_inputs->neg_pos : nullptr;
        if (!neg_pos) {
            neg_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
            ggml_set_input(neg_pos);
            std::vector<int32_t> neg_vals(n_tokens);
            for (int i = 0; i < n_tokens; i++) neg_vals[i] = -(kv_start + i);
            i32_array_inputs.push_back({neg_pos, std::move(neg_vals)});
        }
        context = build_tail_rope_3d(
            ctx, context, neg_pos, n_rot, head_dim, n_head, n_tokens,
            rope_freq, rope_scale, rope_ext, rope_attn,
            w.rope_yarn_beta_fast, w.rope_yarn_beta_slow,
            rope_n_ctx_orig);
    }

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
    attn_out = ggml_permute(ctx, attn_out, 0, 2, 1, 3);
    if (n_tokens == 1) {
        attn_out = ggml_cont(ctx, attn_out);
    }
    // attn_out is now [group_dim, n_tokens, n_out_group]
    ggml_tensor * out_a_3d = ggml_reshape_3d(ctx, L.attn_output_a, group_dim, n_lora_o, n_out_group);
    // out_a_3d: [group_dim, n_lora_o, n_out_group] — ne[2] matches
    ggml_tensor * attn_low = ggml_mul_mat(ctx, out_a_3d, attn_out);
    // attn_low: [n_lora_o, n_tokens, n_out_group]
    ggml_tensor * out = nullptr;
    if (n_tokens > 1) {
        // Batched ROCmFPX MMQ consumes src1's channel stride directly. This
        // avoids materializing both permutations (~256 MiB/layer at 2K).
        out = ggml_mul_mat_grouped_src(ctx, L.attn_output_b, attn_low);
    } else {
        // Preserve the established single-token graph and its numerical
        // behavior. Decode is intentionally outside the prefill fast path.
        attn_low = ggml_cont(ctx, ggml_permute(ctx, attn_low, 0, 2, 1, 3));
        attn_low = ggml_reshape_2d(
            ctx, attn_low, n_lora_o * n_out_group, n_tokens);
        out = ggml_mul_mat(ctx, L.attn_output_b, attn_low);
    }

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

        out.inputs.raw_kv_rows =
            ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_I64, 1, 1);
        ggml_set_input(out.inputs.raw_kv_rows);
        if (ratio > 0) {
            out.inputs.attn_ape_row =
                ggml_new_tensor_1d(out.sg.ctx, GGML_TYPE_I32, 1);
            out.inputs.attn_comp_pos =
                ggml_new_tensor_1d(out.sg.ctx, GGML_TYPE_I32, 1);
            out.inputs.attn_state_rows =
                ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_I64, 1, 1);
            out.inputs.attn_comp_rows =
                ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_I64, 1, 1);
            ggml_set_input(out.inputs.attn_ape_row);
            ggml_set_input(out.inputs.attn_comp_pos);
            ggml_set_input(out.inputs.attn_state_rows);
            ggml_set_input(out.inputs.attn_comp_rows);
        }
        if (ratio == 4) {
            out.inputs.index_ape_row =
                ggml_new_tensor_1d(out.sg.ctx, GGML_TYPE_I32, 1);
            out.inputs.index_comp_pos =
                ggml_new_tensor_1d(out.sg.ctx, GGML_TYPE_I32, 1);
            out.inputs.index_state_rows =
                ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_I64, 1, 1);
            out.inputs.index_comp_rows =
                ggml_new_tensor_2d(out.sg.ctx, GGML_TYPE_I64, 1, 1);
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

    const int k_used = ds4_effective_expert_count(w);
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
    ggml_tensor * shared_out = build_shared_ffn(ctx, cur, w, L);
    ggml_tensor * routed_out = nullptr;

    if (layer_idx < w.n_hash_layer && L.ffn_gate_tid2eid) {
        routed_out = ggml_scale(ctx, cur, 0.0f);
    } else {
        Ds4MoeRouting routing = build_moe_routing(ctx, cur, w, L, n_tokens);
        n_used = (int) routing.selected->ne[0];
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

static const int32_t * hash_routing_row(
        const HashRoutingTableCpu & table,
        int32_t token_id,
        int table_width) {
    if (!table.loaded || token_id < 0 || table_width <= 0) {
        return nullptr;
    }
    const size_t offset = (size_t) token_id * (size_t) table_width;
    if (offset > table.ids.size() ||
        table.ids.size() - offset < (size_t) table_width) {
        return nullptr;
    }
    return table.ids.data() + offset;
}

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
// ~86 of these 24x16384 matvecs per token, so workers spin briefly to catch
// adjacent jobs, then park on a condition variable while the server is idle.
struct Ds4HcMatvecPool {
    struct Job {
        const uint16_t * mat;
        const float * x;
        float * out;
        int rows;
        int cols;
        int active_workers;
    };
    std::mutex client_mu;
    std::mutex wait_mu;
    std::condition_variable wait_cv;
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
                    uint64_t s = last;
                    for (int spins = 0; spins < 65536; ++spins) {
                        s = seq.load(std::memory_order_acquire);
                        if (s != last || stop.load(std::memory_order_relaxed)) {
                            break;
                        }
                        cpu_relax();
                    }
                    if (s == last && !stop.load(std::memory_order_relaxed)) {
                        std::unique_lock<std::mutex> lk(wait_mu);
                        wait_cv.wait(lk, [&]() {
                            return stop.load(std::memory_order_relaxed) ||
                                   seq.load(std::memory_order_acquire) != last;
                        });
                        s = seq.load(std::memory_order_acquire);
                    }
                    if (stop.load(std::memory_order_relaxed)) return;
                    last = s;
                    const Job j = job;
                    if (i >= j.active_workers) continue;
                    const int chunk = (j.rows + j.active_workers - 1) / j.active_workers;
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
        {
            // Protect predicate changes with the same mutex used by wait().
            // Otherwise a worker can test the predicate, miss notify_all(),
            // and sleep forever between the test and the wait.
            std::lock_guard<std::mutex> lk(wait_mu);
            stop.store(true, std::memory_order_release);
        }
        wait_cv.notify_all();
        for (auto & t : workers) t.join();
    }
    void run(const uint16_t * mat, const float * x, float * out, int rows, int cols) {
        if (rows <= 0) return;
        std::lock_guard<std::mutex> lk(client_mu);
        row_fn = nullptr;
        const int active_workers = std::min(nth, rows);
        job = {mat, x, out, rows, cols, active_workers};
        remaining.store(active_workers, std::memory_order_release);
        {
            // Publish the new generation while holding wait_mu so a worker
            // cannot miss the transition between its predicate check and
            // blocking in wait().
            std::lock_guard<std::mutex> wake_lk(wait_mu);
            seq.fetch_add(1, std::memory_order_release);
        }
        wait_cv.notify_all();
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
        if (rows <= 0) return;
        std::lock_guard<std::mutex> lk(client_mu);
        row_fn = std::move(fn);
        const int active_workers = std::min(nth, rows);
        job = {nullptr, nullptr, nullptr, rows, 0, active_workers};
        remaining.store(active_workers, std::memory_order_release);
        {
            std::lock_guard<std::mutex> wake_lk(wait_mu);
            seq.fetch_add(1, std::memory_order_release);
        }
        wait_cv.notify_all();
        int spins = 0;
        while (remaining.load(std::memory_order_acquire) != 0) {
            if (++spins < 65536) { cpu_relax(); }
            else { std::this_thread::yield(); spins = 0; }
        }
        row_fn = nullptr;
    }
};

static Ds4HcMatvecPool & ds4_hc_matvec_pool() {
    static Ds4HcMatvecPool pool;
    return pool;
}

static void cpu_matvec_f16_pooled(float * out, const uint16_t * mat, const float * x, int rows, int cols) {
    ds4_hc_matvec_pool().run(mat, x, out, rows, cols);
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
        struct Ctx { const float * block; const float * res; const float * post; const float * comb; float * out; int n_embd; int n_hc; };
        Ctx c{block_out, residual_hc, post, comb, out_hc.data(), n_embd, n_hc};
        ds4_hc_matvec_pool().run_custom(n_hc, [&c](int h) {
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

static void reset_hc_weights_cpu(HcWeightsCpu & w) {
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

struct DeepSeek4HybridRuntime {
    const ggml_context * owner_ctx = nullptr;
    std::vector<HcLayerWeightsCpu> hc_layer_weights;
    HcWeightsCpu hc_output_weights;
    std::vector<HashRoutingTableCpu> hash_routing_tables;

    void destroy() {
        reset_hc_layer_weights_cpu(hc_layer_weights);
        reset_hc_weights_cpu(hc_output_weights);
        hash_routing_tables.clear();
        hash_routing_tables.shrink_to_fit();
        owner_ctx = nullptr;
    }
};

static thread_local DeepSeek4HybridRuntime ds4_hybrid_runtime;

static const void * hc_fn_device_ptr(const HcWeightsCpu &, ggml_tensor * fn) {
    if (!fn) return nullptr;
    if (fn->type == GGML_TYPE_F16) return fn->data;
    return nullptr;
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

    // Cache host mirrors by model ownership so unload/reload cannot reuse
    // tensor data from a previous model context.
    DeepSeek4HybridRuntime & runtime = ds4_hybrid_runtime;
    if (runtime.owner_ctx != w.ctx ||
        runtime.hc_layer_weights.size() != (size_t) w.n_layer) {
        runtime.destroy();
        runtime.owner_ctx = w.ctx;
    }
    auto & hc_layer_weights = runtime.hc_layer_weights;
    auto & hc_output_weights = runtime.hc_output_weights;
    auto & hash_routing_tables = runtime.hash_routing_tables;
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

            const int n_used = ds4_effective_expert_count(w);
            std::vector<int32_t> selected_host((size_t)n_used * (size_t)n_tokens);
            std::vector<float> weights_host((size_t)n_used * (size_t)n_tokens);
            const auto route_select_t0 = Ds4TimingClock::now();
            for (int ti = 0; ti < n_tokens; ++ti) {
                const int32_t tok = token_ids[ti];
                const int32_t * row = hash_routing_row(
                    hash_routing_tables[(size_t)il], tok, w.n_expert_used);
                if (!row) {
                    std::fprintf(stderr, "[deepseek4] token id %d outside hash table for layer %d\n", tok, il);
                    if (hot_alloc) ggml_gallocr_free(hot_alloc);
                    if (cold_alloc) ggml_gallocr_free(cold_alloc);
                    return false;
                }
                float sum = 0.0f;
                for (int ei = 0; ei < n_used; ++ei) {
                    const int32_t expert = row[ei];
                    selected_host[(size_t)ti * (size_t)n_used + (size_t)ei] = expert;
                    float prob = 0.0f;
                    if (expert >= 0 && expert < w.n_expert) {
                        prob = probs_host[(size_t)ti * (size_t)w.n_expert + (size_t)expert];
                    }
                    weights_host[(size_t)ti * (size_t)n_used + (size_t)ei] = prob;
                    sum += prob;
                }
                sum = std::max(sum, 6.103515625e-5f);
                for (int ei = 0; ei < n_used; ++ei) {
                    float & weight = weights_host[(size_t)ti * (size_t)n_used + (size_t)ei];
                    weight = weight / sum * w.expert_weight_scale;
                }
            }
            if (telemetry) telemetry->route_select_us += ds4_elapsed_us(route_select_t0, Ds4TimingClock::now());
            if (routing_stats) {
                for (int ti = 0; ti < n_tokens; ++ti) {
                    routing_stats->observe(il,
                        selected_host.data() + (size_t)ti * (size_t)n_used,
                        n_used);
                }
            }

            MoeHybridConfig hybrid_cfg = make_ds4_moe_hybrid_config(w);
            MoeLayerDesc desc = make_ds4_moe_layer_desc(L);
            auto & storage = moe_hybrid.layers[(size_t) il];
            if (!eval_ds4_hybrid(
                    backend, cpu_backend, hybrid_cfg, desc, &moe_hybrid, storage, stream_engine,
                    il, n_embd, n_used,
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
            const int n_used = ds4_effective_expert_count(w);
            std::vector<int32_t> selected_host((size_t)n_used * (size_t)n_tokens);
            std::vector<float> weights_host((size_t)n_used * (size_t)n_tokens);
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
                std::vector<int32_t> top((size_t)n_used, -1);
                for (int expert = 0; expert < w.n_expert; ++expert) {
                    const float score = probs[expert] +
                        (!bias_host.empty() ? bias_host[(size_t)expert] : 0.0f);
                    for (int slot = 0; slot < n_used; ++slot) {
                        const int32_t cur_expert = top[(size_t)slot];
                        const float cur_score = cur_expert >= 0
                            ? probs[cur_expert] +
                                (!bias_host.empty() ? bias_host[(size_t)cur_expert] : 0.0f)
                            : -INFINITY;
                        if (cur_expert < 0 || score > cur_score) {
                            for (int m = n_used - 1; m > slot; --m) {
                                top[(size_t)m] = top[(size_t)m - 1];
                            }
                            top[(size_t)slot] = expert;
                            break;
                        }
                    }
                }
                float sum = 0.0f;
                for (int slot = 0; slot < n_used; ++slot) {
                    const int32_t expert = top[(size_t)slot];
                    selected_host[(size_t)ti * (size_t)n_used + (size_t)slot] = expert;
                    const float weight = expert >= 0 ? probs[expert] : 0.0f;
                    weights_host[(size_t)ti * (size_t)n_used + (size_t)slot] = weight;
                    sum += weight;
                }
                sum = std::max(sum, 6.103515625e-5f);
                for (int slot = 0; slot < n_used; ++slot) {
                    float & weight = weights_host[(size_t)ti * (size_t)n_used + (size_t)slot];
                    weight = weight / sum * w.expert_weight_scale;
                }
            }
            if (telemetry) telemetry->route_select_us += ds4_elapsed_us(route_select_t0, Ds4TimingClock::now());
            if (routing_stats) {
                for (int ti = 0; ti < n_tokens; ++ti) {
                    routing_stats->observe(il,
                        selected_host.data() + (size_t)ti * (size_t)n_used,
                        n_used);
                }
            }

            MoeHybridConfig hybrid_cfg = make_ds4_moe_hybrid_config(w);
            MoeLayerDesc desc = make_ds4_moe_layer_desc(L);
            auto & storage = moe_hybrid.layers[(size_t) il];
            if (!eval_ds4_hybrid(
                    backend, cpu_backend, hybrid_cfg, desc, &moe_hybrid, storage, stream_engine,
                    il, n_embd, n_used,
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
        MoeHybridRoutingStats * routing_stats,
        Ds4VerifyHooks * verify_hooks) {
    if (w.moe_hybrid && moe_hybrid != nullptr) {
        return deepseek4_step_hybrid(backend, w, cache, *moe_hybrid,
                                     embed, n_tokens, kv_start, out_logits,
                                     token_ids, stream_engine, telemetry, routing_stats);
    }

    std::vector<float> hc_state;
    return deepseek4_step_layer_range(
        backend, w, cache, hc_state, embed, n_tokens, kv_start,
        0, w.n_layer, &out_logits, token_ids, telemetry,
        /*allow_decode_graph_reuse=*/verify_hooks == nullptr, verify_hooks);
}

// ─── Fused single-graph decode (n_tokens == 1) ──────────────────────────
// Chains all layers (HC pre → attention → HC post → HC pre → FFN → HC post)
// plus the output HC merge and lm_head into ONE cached ggml graph, so a
// decode step is a single ggml_backend_graph_compute with one logits
// readback instead of ~90 per-layer graph launches with host round-trips.
// HC Sinkhorn mixing runs in the fused GGML_OP_DS4_HC op (one kernel per
// sublayer instead of ~170 tiny ops for 20 Sinkhorn iterations).
//
// Compressed-KV reads are padded to a fixed stride with an additive
// score mask, and each structural variant (flush pattern) lives in its own
// slot with a private metadata arena. Tensor addresses therefore stay stable
// while a variant recurs, which is what the ggml-cuda/HIP graph cache keys
// on, enabling graph replay for the bulk of decode steps.

struct DeepSeek4FusedDecodeGraph {
    std::vector<int64_t> shape_key;
    uint64_t last_use = 0;
    StepGraph sg;
    ggml_tensor * inp_embed = nullptr;
    ggml_tensor * i32_bundle = nullptr;
    ggml_tensor * i64_bundle = nullptr;
    ggml_tensor * mask_bundle = nullptr;   // additive score mask (0 / -1e30), may be null
    std::vector<ggml_tensor *> hash_ids;
    ggml_tensor * logits = nullptr;

    void reset_nodes() {
        inp_embed = nullptr;
        i32_bundle = nullptr;
        i64_bundle = nullptr;
        mask_bundle = nullptr;
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

struct DeepSeek4LayerRangeRuntime {
    const ggml_context * owner_ctx = nullptr;
    int loaded_n_layer = 0;
    std::vector<HcLayerWeightsCpu> hc_layer_weights;
    HcWeightsCpu hc_output_weights;
    std::vector<HashRoutingTableCpu> hash_routing_tables;
    std::vector<DeepSeek4CachedLayerAlloc> cached_attn_allocs;
    std::vector<DeepSeek4CachedDecodeHcPreGraph> cached_decode_attn_hc_pre_graphs;
    std::vector<DeepSeek4CachedDecodeHcPreGraph> cached_decode_ffn_hc_pre_graphs;
    DeepSeek4CachedDecodeHcPostGraph cached_decode_hc_post_graph;
    std::vector<std::vector<DeepSeek4CachedDecodeAttnGraph>> cached_decode_attn_graphs;
    std::vector<DeepSeek4CachedDecodeFfnGraph> cached_decode_ffn_graphs;
    DeepSeek4CachedDecodeOutputGraph cached_decode_output_graph;
    DeepSeek4CachedLayerAlloc cached_dynamic_output_alloc;
    DeepSeek4FusedDecodeCache fused_decode_graph_cache;
    Ds4DecodeSharedInputs decode_shared_inputs;
    DeepSeek4LayerRangeScratch scratch;

    void destroy() {
        for (auto & alloc : cached_attn_allocs) {
            alloc.free();
        }
        cached_attn_allocs.clear();
        for (auto & graph : cached_decode_attn_hc_pre_graphs) {
            graph.free();
        }
        cached_decode_attn_hc_pre_graphs.clear();
        for (auto & graph : cached_decode_ffn_hc_pre_graphs) {
            graph.free();
        }
        cached_decode_ffn_hc_pre_graphs.clear();
        cached_decode_hc_post_graph.free();
        for (auto & per_layer : cached_decode_attn_graphs) {
            for (auto & graph : per_layer) {
                graph.free();
            }
        }
        cached_decode_attn_graphs.clear();
        for (auto & graph : cached_decode_ffn_graphs) {
            graph.free();
        }
        cached_decode_ffn_graphs.clear();
        cached_decode_output_graph.free();
        cached_dynamic_output_alloc.free();
        fused_decode_graph_cache.destroy();
        decode_shared_inputs.free();
        reset_hc_layer_weights_cpu(hc_layer_weights);
        reset_hc_weights_cpu(hc_output_weights);
        hash_routing_tables.clear();
        hash_routing_tables.shrink_to_fit();
        scratch.clear();
        loaded_n_layer = 0;
        owner_ctx = nullptr;
    }
};

static thread_local DeepSeek4LayerRangeRuntime ds4_layer_range_runtime;

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
        ggml_tensor * hash_ids,
        int n_tokens) {
    ggml_tensor * shared_out = build_shared_ffn(ctx, ffn_normed, w, L);
    ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, ffn_normed);
    ggml_tensor * probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));

    const int n_used = (int) hash_ids->ne[0];
    const int n_ff_exp = w.n_ff_exp;
    ggml_tensor * cur_3d = ggml_reshape_3d(
        ctx, ffn_normed, w.n_embd, 1, n_tokens);
    ggml_tensor * gate_e = ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur_3d, hash_ids);
    ggml_tensor * up_e = ggml_mul_mat_id(ctx, L.ffn_up_exps, cur_3d, hash_ids);
    gate_e = ggml_reshape_3d(ctx, gate_e, n_ff_exp, n_used, n_tokens);
    up_e = ggml_reshape_3d(ctx, up_e, n_ff_exp, n_used, n_tokens);
    ggml_tensor * mid_e = build_clamped_swiglu(ctx, gate_e, up_e, w.swiglu_clamp_exp);
    ggml_tensor * down_e = ggml_mul_mat_id(ctx, L.ffn_down_exps, mid_e, hash_ids);
    down_e = ggml_reshape_3d(ctx, down_e, w.n_embd, n_used, n_tokens);

    ggml_tensor * probs_3d = ggml_reshape_3d(
        ctx, probs, 1, w.n_expert, n_tokens);
    ggml_tensor * weights = ggml_get_rows(ctx, probs_3d, hash_ids);
    weights = ggml_reshape_2d(ctx, weights, n_used, n_tokens);
    ggml_tensor * w_sum = ggml_sum_rows(ctx, weights);
    w_sum = ggml_clamp(ctx, w_sum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(ctx, weights, w_sum);
    if (w.expert_weight_scale != 1.0f) {
        weights = ggml_scale(ctx, weights, w.expert_weight_scale);
    }

    ggml_tensor * weights_3d = ggml_reshape_3d(
        ctx, weights, 1, n_used, n_tokens);
    ggml_tensor * routed_out = ggml_mul(ctx, down_e, weights_3d);
    if (n_tokens == 1) {
        // Preserve the established q=1 graph and reduction order.
        routed_out = ggml_cont(ctx, ggml_permute(ctx, routed_out, 1, 0, 2, 3));
        routed_out = ggml_sum_rows(ctx, routed_out);
        routed_out = ggml_reshape_2d(ctx, routed_out, w.n_embd, 1);
    } else {
        ggml_tensor * sum_shape = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F32, w.n_embd, 1, n_tokens);
        routed_out = ggml_repeat_back(ctx, routed_out, sum_shape);
        routed_out = ggml_reshape_2d(
            ctx, routed_out, w.n_embd, n_tokens);
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
    const int64_t hc_dim = (int64_t) w.n_embd * w.n_hc;
    const int64_t mix_dim = 2 * (int64_t) w.n_hc + (int64_t) w.n_hc * w.n_hc;
    if (hc_weights.size() != (size_t) w.n_layer) {
        return false;
    }
    for (int il = 0; il < w.n_layer; ++il) {
        const auto & attn = hc_weights[(size_t) il].attn.fn_data;
        const auto & ffn = hc_weights[(size_t) il].ffn.fn_data;
        if ((int64_t) attn.size() != hc_dim * mix_dim ||
            (int64_t) ffn.size() != hc_dim * mix_dim) {
            return false;
        }
    }
    if ((int64_t) hc_out_weights.fn_data.size() != hc_dim * w.n_hc) {
        return false;
    }

    if (fc.fn_buf) { ggml_backend_buffer_free(fc.fn_buf); fc.fn_buf = nullptr; }
    if (fc.fn_ctx) { ggml_free(fc.fn_ctx); fc.fn_ctx = nullptr; }
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
        ggml_backend_tensor_set(fc.fn_attn_f16[(size_t) il], a.data(), 0, a.size() * sizeof(uint16_t));
        ggml_backend_tensor_set(fc.fn_ffn_f16[(size_t) il], f.data(), 0, f.size() * sizeof(uint16_t));
    }
    const auto & o = hc_out_weights.fn_data;
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
            ggml_tensor * hids = ggml_new_tensor_2d(
                ctx, GGML_TYPE_I32, ds4_effective_expert_count(w), 1);
            ggml_set_input(hids);
            fg.hash_ids[(size_t) il] = hids;
            ffn_out = ds4_build_hash_routed_ffn(
                ctx, w, L, ffn_normed, hids, 1);
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
        const bool flush = ratio > 0 && (((token_pos + 1) % ratio) == 0);
        key.push_back(((int64_t) padded << 1) | (flush ? 1 : 0));
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
            const int n_used = ds4_effective_expert_count(w);
            hash_scratch.resize((size_t) n_used);
            const int32_t tok = token_ids[0];
            const int32_t * row = hash_routing_row(
                hash_tables[(size_t) il], tok, w.n_expert_used);
            if (!row) {
                std::fprintf(stderr,
                             "[deepseek4] token id %d outside hash table for layer %d\n",
                             tok, il);
                return -1;
            }
            std::memcpy(hash_scratch.data(), row,
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

// Exact-order prefill control: retain the layer-major HC/FFN schedule, but run
// the attention subgraph one token at a time. This preserves the q=1 QKV,
// compressor, causal-attention, and output-projection reduction order while
// still allowing the token-independent FFN to use a multi-row ROCMFP graph.
static bool ds4_run_exact_tokenwise_prefill_attention(
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        const DeepSeek4Layer & L,
        DeepSeek4LayerCache & lc,
        int il,
        const float * cur,
        int n_tokens,
        int kv_start,
        DeepSeek4AttentionImpl attention_impl,
        std::vector<float> & attn_out_host,
        DeepSeek4CachedLayerAlloc & attn_alloc,
        DeepSeek4StepTelemetry * telemetry) {
    if (!backend || !cur || n_tokens <= 1 || kv_start < 0) return false;

    const int n_embd = w.n_embd;
    for (int ti = 0; ti < n_tokens; ++ti) {
        const auto build_t0 = Ds4TimingClock::now();
        ggml_init_params params{};
        params.mem_size = ds4_attn_step_meta_size(1);
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        if (!ctx) return false;

        ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
        ggml_set_input(inp);
        std::vector<DeepSeek4I32InputBinding> i32_inputs;
        std::vector<DeepSeek4I32ArrayBinding> i32_array_inputs;
        std::vector<DeepSeek4I64ArrayBinding> i64_array_inputs;
        std::vector<DeepSeek4F32ArrayBinding> f32_array_inputs;
        ggml_cgraph * gf = ggml_new_graph_custom(
            ctx, ds4_attn_step_graph_size(1), false);
        ggml_tensor * normed = build_rms_norm(ctx, inp, L.attn_norm, w.rms_eps);
        ggml_tensor * attn_out = build_mla_attention(
            ctx, gf, normed, w, L, lc, il, kv_start + ti, 1, nullptr,
            i32_inputs, i32_array_inputs, i64_array_inputs, &f32_array_inputs,
            attention_impl);
        ggml_set_output(attn_out);
        ggml_build_forward_expand(gf, attn_out);

        if (!attn_alloc.valid() || attn_alloc.owner_ctx != w.ctx ||
            attn_alloc.backend != backend) {
            attn_alloc.free();
            attn_alloc.alloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend));
            attn_alloc.owner_ctx = w.ctx;
            attn_alloc.backend = backend;
        }
        if (!attn_alloc.alloc || !ggml_gallocr_alloc_graph(attn_alloc.alloc, gf)) {
            std::fprintf(stderr,
                         "[deepseek4] exact prefill attn alloc failed layer %d token %d\n",
                         il, ti);
            ggml_free(ctx);
            return false;
        }
        if (telemetry) {
            telemetry->attn_build_us += ds4_elapsed_us(build_t0, Ds4TimingClock::now());
        }

        ggml_backend_tensor_set(inp, cur + (size_t) ti * n_embd, 0,
                                sizeof(float) * (size_t) n_embd);
        for (const auto & b : i32_inputs) {
            ggml_backend_tensor_set(b.tensor, &b.value, 0, sizeof(b.value));
        }
        for (const auto & b : i32_array_inputs) {
            ggml_backend_tensor_set(b.tensor, b.values.data(), 0,
                                    sizeof(int32_t) * b.values.size());
        }
        for (const auto & b : i64_array_inputs) {
            ggml_backend_tensor_set(b.tensor, b.values.data(), 0,
                                    sizeof(int64_t) * b.values.size());
        }
        for (const auto & b : f32_array_inputs) {
            ggml_backend_tensor_set(b.tensor, b.values.data(), 0,
                                    sizeof(float) * b.values.size());
        }

        const auto compute_t0 = Ds4TimingClock::now();
        const ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (telemetry) {
            telemetry->attn_compute_us += ds4_elapsed_us(
                compute_t0, Ds4TimingClock::now());
        }
        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr,
                         "[deepseek4] exact prefill attn compute failed layer %d token %d\n",
                         il, ti);
            ggml_free(ctx);
            return false;
        }

        const auto read_t0 = Ds4TimingClock::now();
        ggml_backend_tensor_get(attn_out,
                                attn_out_host.data() + (size_t) ti * n_embd,
                                0, sizeof(float) * (size_t) n_embd);
        if (telemetry) {
            telemetry->attn_read_us += ds4_elapsed_us(read_t0, Ds4TimingClock::now());
        }

        // Publish compressor rows immediately. The next token in this layer
        // must observe a row flushed by the current token, matching the q=1
        // reference when a prefill band crosses a compressor boundary.
        const int ratio = (int) w.compress_ratios[il];
        if (ratio > 0) {
            const int next_pos = kv_start + ti + 1;
            lc.n_comp = std::max(lc.n_comp, next_pos / ratio);
            if (ratio == 4) {
                lc.n_index_comp = std::max(lc.n_index_comp,
                                           next_pos / ratio);
            }
        }
        ggml_free(ctx);
    }
    return true;
}

// Layer-major DS4 prefill. Each layer is one GPU graph containing batched HC,
// attention, MoE and HC post-processing. The HC state is kept in two external
// device tensors and ping-ponged between layers, eliminating the two host
// readbacks per layer in the reference implementation. Attention reads a
// snapshot of the previous SWA window plus the current ubatch; only the final
// SWA tail is committed to the persistent ring. The compressor publishes every
// ratio-4/ratio-128 boundary crossed by the ubatch.
//
struct Ds4LayerMajorCachedLayer {
    void * meta_buffer = nullptr;
    size_t meta_size = 0;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    std::vector<DeepSeek4I32InputBinding> i32_inputs;
    std::vector<DeepSeek4I32ArrayBinding> i32_array_inputs;
    std::vector<DeepSeek4I64ArrayBinding> i64_array_inputs;
    std::vector<DeepSeek4F32ArrayBinding> f32_array_inputs;
    std::vector<ggml_tensor *> allocated_tensors;
    ggml_tensor * hash_ids = nullptr;
    ggml_tensor * logits = nullptr;

    void destroy() {
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        if (meta_buffer) {
            std::free(meta_buffer);
            meta_buffer = nullptr;
        }
        meta_size = 0;
        gf = nullptr;
        hash_ids = nullptr;
        logits = nullptr;
        i32_inputs.clear();
        i32_array_inputs.clear();
        i64_array_inputs.clear();
        f32_array_inputs.clear();
        allocated_tensors.clear();
    }
};

struct Ds4LayerMajorGraphCache {
    const ggml_context * owner_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    PrefillAttentionMode mode = PrefillAttentionMode::Exact;
    int n_tokens = 0;
    int kv_start = -1;
    bool ready = false;
    ggml_context * state_ctx = nullptr;
    ggml_backend_buffer_t state_buf = nullptr;
    ggml_tensor * state_a = nullptr;
    ggml_tensor * state_b = nullptr;
    std::vector<Ds4LayerMajorCachedLayer> layers;

    bool matches(const DeepSeek4Weights & w, ggml_backend_t b,
                 PrefillAttentionMode m, int tokens, int start) const {
        return ready && owner_ctx == w.ctx && backend == b && mode == m &&
               n_tokens == tokens && kv_start == start &&
               layers.size() == (size_t) w.n_layer;
    }

    void destroy() {
        for (auto & layer : layers) layer.destroy();
        layers.clear();
        if (state_buf) {
            ggml_backend_buffer_free(state_buf);
            state_buf = nullptr;
        }
        if (state_ctx) {
            ggml_free(state_ctx);
            state_ctx = nullptr;
        }
        state_a = nullptr;
        state_b = nullptr;
        owner_ctx = nullptr;
        backend = nullptr;
        mode = PrefillAttentionMode::Exact;
        n_tokens = 0;
        kv_start = -1;
        ready = false;
    }
};

static thread_local std::array<Ds4LayerMajorGraphCache, 1>
    ds4_layer_major_graph_caches;
// A separate gallocr scratch arena per shape consumes several GiB and forces
// the 97-GiB model into managed-memory paging. Every layer executes serially,
// so cached and uncached shapes safely rebind their transient tensors to one
// arena before execution. Retain only the largest/full-chunk topology; tail
// metadata is rebuilt rather than keeping a second long-context graph resident.
static thread_local ggml_gallocr_t ds4_layer_major_shared_alloc = nullptr;
static thread_local const ggml_context * ds4_layer_major_shared_owner = nullptr;
static thread_local ggml_backend_t ds4_layer_major_shared_backend = nullptr;
static thread_local std::vector<uint8_t> ds4_layer_major_meta_arena;
static thread_local const ggml_context * ds4_layer_major_meta_owner = nullptr;

static ggml_gallocr_t ds4_layer_major_get_shared_alloc(
        const DeepSeek4Weights & w,
        ggml_backend_t backend) {
    if (ds4_layer_major_shared_alloc &&
        (ds4_layer_major_shared_owner != w.ctx ||
         ds4_layer_major_shared_backend != backend)) {
        ggml_gallocr_free(ds4_layer_major_shared_alloc);
        ds4_layer_major_shared_alloc = nullptr;
    }
    if (!ds4_layer_major_shared_alloc) {
        ds4_layer_major_shared_alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        ds4_layer_major_shared_owner = w.ctx;
        ds4_layer_major_shared_backend = backend;
    }
    return ds4_layer_major_shared_alloc;
}

void deepseek4_release_runtime_graphs(const DeepSeek4Weights & w) {
    const ggml_context * owner = w.ctx;
    if (!owner) {
        return;
    }

    for (auto & cache : ds4_layer_major_graph_caches) {
        if (cache.owner_ctx == owner) {
            cache.destroy();
        }
    }

    if (ds4_layer_major_shared_owner == owner) {
        if (ds4_layer_major_shared_alloc) {
            ggml_gallocr_free(ds4_layer_major_shared_alloc);
            ds4_layer_major_shared_alloc = nullptr;
        }
        ds4_layer_major_shared_owner = nullptr;
        ds4_layer_major_shared_backend = nullptr;
    }
    if (ds4_layer_major_meta_owner == owner) {
        ds4_layer_major_meta_arena.clear();
        ds4_layer_major_meta_arena.shrink_to_fit();
        ds4_layer_major_meta_owner = nullptr;
    }
    if (ds4_layer_range_runtime.owner_ctx == owner) {
        ds4_layer_range_runtime.destroy();
    }
    if (ds4_hybrid_runtime.owner_ctx == owner) {
        ds4_hybrid_runtime.destroy();
    }
}

// Returns 1 on success, 0 when the optimized path is not applicable, and -1
// after a hard failure.
static int ds4_try_layer_major_prefill(
        DeepSeek4FusedDecodeCache & fc,
        ggml_backend_t backend,
        const DeepSeek4Weights & w,
        DeepSeek4Cache & cache,
        const std::vector<HcLayerWeightsCpu> & hc_weights,
        const HcWeightsCpu & hc_out_weights,
        const std::vector<HashRoutingTableCpu> & hash_tables,
        std::vector<int32_t> & hash_scratch,
        const float * embed,
        int n_tokens,
        int kv_start,
        std::vector<float> & out_logits,
        const int32_t * token_ids,
        DeepSeek4StepTelemetry * telemetry) {
    if (!backend || !embed || n_tokens <= 4 ||
        n_tokens > DS4_MAX_LAYER_MAJOR_PREFILL_TOKENS ||
        kv_start < 0 || w.moe_hybrid) {
        return 0;
    }
    if (cache.prefill_mode == PrefillAttentionMode::Exact) return 0;
    if (!ds4_backend_is_gpu(backend) || !hc_out_weights.loaded ||
        hc_out_weights.scale_data.empty() || !w.output_hc_fn ||
        !w.output_hc_base) {
        return 0;
    }
    for (int il = 0; il < w.n_layer; ++il) {
        const HcLayerWeightsCpu & hlw = hc_weights[(size_t) il];
        const DeepSeek4Layer & L = w.layers[(size_t) il];
        if (!hlw.attn.loaded || hlw.attn.scale_data.size() < 3 ||
            !hlw.ffn.loaded || hlw.ffn.scale_data.size() < 3 ||
            !L.hc_attn_base || !L.hc_ffn_base) {
            return 0;
        }
    }

    if (fc.owner_ctx != w.ctx || fc.backend != backend) {
        fc.destroy();
        fc.owner_ctx = w.ctx;
        fc.backend = backend;
    }
    if (!ds4_fused_ensure_fn_mirrors(fc, backend, w, hc_weights,
                                      hc_out_weights)) {
        std::fprintf(stderr,
                     "[deepseek4-prefill] failed to create HC weight mirrors\n");
        return -1;
    }

    const int n_embd = w.n_embd;
    const int n_hc = w.n_hc;
    const int64_t hc_dim = (int64_t) n_embd * n_hc;
    const int64_t mix_dim = 2 * (int64_t) n_hc + (int64_t) n_hc * n_hc;
    const int next_pos = kv_start + n_tokens;

    Ds4LayerMajorGraphCache * graph_cache = nullptr;
    bool cache_hit = false;
    bool cache_build = false;
    if (token_ids) {
        for (auto & candidate : ds4_layer_major_graph_caches) {
            if (candidate.matches(w, backend, cache.prefill_mode,
                                  n_tokens, kv_start)) {
                graph_cache = &candidate;
                cache_hit = true;
                break;
            }
        }
        if (!graph_cache) {
            auto & candidate = ds4_layer_major_graph_caches.front();
            // Do not evict a full/larger chunk for an equal-size graph at a
            // later position or for a short tail. Both execute with the shared
            // scratch arena below, but only the dominant topology stays cached.
            const bool same_owner = candidate.owner_ctx == w.ctx &&
                                    candidate.backend == backend &&
                                    candidate.mode == cache.prefill_mode;
            if (!candidate.ready || !same_owner ||
                n_tokens > candidate.n_tokens) {
                graph_cache = &candidate;
                graph_cache->destroy();
                graph_cache->owner_ctx = w.ctx;
                graph_cache->backend = backend;
                graph_cache->mode = cache.prefill_mode;
                graph_cache->n_tokens = n_tokens;
                graph_cache->kv_start = kv_start;
                graph_cache->layers.resize((size_t) w.n_layer);
                cache_build = true;
            }
        }
    }

    // Persistent ping-pong state lives outside the per-layer gallocr arena.
    ggml_context * state_ctx = cache_hit ? graph_cache->state_ctx : nullptr;
    ggml_tensor * state_a = cache_hit ? graph_cache->state_a : nullptr;
    ggml_tensor * state_b = cache_hit ? graph_cache->state_b : nullptr;
    ggml_backend_buffer_t state_buf = cache_hit ? graph_cache->state_buf : nullptr;
    if (!cache_hit) {
        ggml_init_params state_params{};
        state_params.mem_size = 4 * ggml_tensor_overhead() + 4096;
        state_params.no_alloc = true;
        state_ctx = ggml_init(state_params);
        if (!state_ctx) {
            if (cache_build) graph_cache->destroy();
            return -1;
        }
        state_a = ggml_new_tensor_2d(state_ctx, GGML_TYPE_F32,
                                     hc_dim, n_tokens);
        state_b = ggml_new_tensor_2d(state_ctx, GGML_TYPE_F32,
                                     hc_dim, n_tokens);
        state_buf = ggml_backend_alloc_ctx_tensors(state_ctx, backend);
        if (!state_buf) {
            ggml_free(state_ctx);
            if (cache_build) graph_cache->destroy();
            return -1;
        }
        if (cache_build) {
            graph_cache->state_ctx = state_ctx;
            graph_cache->state_a = state_a;
            graph_cache->state_b = state_b;
            graph_cache->state_buf = state_buf;
        }
    }

    std::vector<float> initial((size_t) hc_dim * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < n_hc; ++h) {
            std::memcpy(initial.data() + (size_t) t * hc_dim +
                            (size_t) h * n_embd,
                        embed + (size_t) t * n_embd,
                        sizeof(float) * (size_t) n_embd);
        }
    }
    ggml_backend_tensor_set(state_a, initial.data(), 0,
                            sizeof(float) * initial.size());
    initial.clear();
    initial.shrink_to_fit();

    ggml_gallocr_t alloc = ds4_layer_major_get_shared_alloc(w, backend);
    if (!alloc) {
        if (cache_build) {
            graph_cache->destroy();
        } else {
            ggml_backend_buffer_free(state_buf);
            ggml_free(state_ctx);
        }
        return -1;
    }
    const size_t meta_bytes = 160u * 1024 * 1024;
    if (ds4_layer_major_meta_owner != w.ctx) {
        ds4_layer_major_meta_arena.clear();
        ds4_layer_major_meta_arena.shrink_to_fit();
        ds4_layer_major_meta_owner = w.ctx;
    }
    if (!graph_cache && ds4_layer_major_meta_arena.size() < meta_bytes) {
        ds4_layer_major_meta_arena.resize(meta_bytes);
    }

    auto fail = [&](const char * what, int il) {
        std::fprintf(stderr, "[deepseek4-prefill] %s at layer %d\n", what, il);
        if (graph_cache) {
            graph_cache->destroy();
        } else {
            ggml_backend_buffer_free(state_buf);
            ggml_free(state_ctx);
        }
        return -1;
    };

    if (cache_hit) {
        for (int il = 0; il < w.n_layer; ++il) {
            Ds4LayerMajorCachedLayer & layer =
                graph_cache->layers[(size_t) il];
            for (ggml_tensor * tensor : layer.allocated_tensors) {
                tensor->data = nullptr;
                tensor->buffer = nullptr;
            }
            const auto alloc_t0 = Ds4TimingClock::now();
            if (!layer.ctx || !layer.gf ||
                !ggml_gallocr_alloc_graph(alloc, layer.gf)) {
                return fail("cached scratch allocation failed", il);
            }
            if (telemetry) {
                telemetry->full_graph_alloc_us += ds4_elapsed_us(
                    alloc_t0, Ds4TimingClock::now());
            }
            for (const auto & b : layer.i32_inputs) {
                ggml_backend_tensor_set(b.tensor, &b.value, 0,
                                        sizeof(b.value));
            }
            for (const auto & b : layer.i32_array_inputs) {
                ggml_backend_tensor_set(b.tensor, b.values.data(), 0,
                                        sizeof(int32_t) * b.values.size());
            }
            for (const auto & b : layer.i64_array_inputs) {
                ggml_backend_tensor_set(b.tensor, b.values.data(), 0,
                                        sizeof(int64_t) * b.values.size());
            }
            for (const auto & b : layer.f32_array_inputs) {
                ggml_backend_tensor_set(b.tensor, b.values.data(), 0,
                                        sizeof(float) * b.values.size());
            }
            if (layer.hash_ids) {
                const int n_used = w.n_expert_used;
                hash_scratch.resize((size_t) n_used * n_tokens);
                const auto & table = hash_tables[(size_t) il].ids;
                for (int t = 0; t < n_tokens; ++t) {
                    std::memcpy(
                        hash_scratch.data() + (size_t) t * n_used,
                        table.data() + (size_t) token_ids[t] * n_used,
                        sizeof(int32_t) * (size_t) n_used);
                }
                ggml_backend_tensor_set(
                    layer.hash_ids, hash_scratch.data(), 0,
                    sizeof(int32_t) * hash_scratch.size());
            }

            const auto compute_t0 = Ds4TimingClock::now();
            if (ggml_backend_graph_compute(backend, layer.gf) !=
                GGML_STATUS_SUCCESS) {
                return fail("cached compute failed", il);
            }
            if (telemetry) {
                telemetry->full_graph_compute_us += ds4_elapsed_us(
                    compute_t0, Ds4TimingClock::now());
            }
            if (layer.logits) {
                out_logits.resize((size_t) w.n_vocab);
                ggml_backend_tensor_get(
                    layer.logits, out_logits.data(), 0,
                    sizeof(float) * (size_t) w.n_vocab);
            }

            DeepSeek4LayerCache & lc = cache.layers[(size_t) il];
            const int ratio = (int) w.compress_ratios[(size_t) il];
            if (ratio > 0) {
                lc.n_comp = std::max(lc.n_comp, next_pos / ratio);
                if (ratio == 4) {
                    lc.n_index_comp = std::max(
                        lc.n_index_comp, next_pos / ratio);
                }
            }
        }
        cache.cur_pos = next_pos;
        return out_logits.empty() ? -1 : 1;
    }

    ggml_tensor * state_in = state_a;
    ggml_tensor * state_out = state_b;
    for (int il = 0; il < w.n_layer; ++il) {
        const auto build_t0 = Ds4TimingClock::now();
        Ds4LayerMajorCachedLayer * cached_layer = cache_build
            ? &graph_cache->layers[(size_t) il] : nullptr;
        ggml_init_params params{};
        if (cached_layer) {
            cached_layer->meta_size = meta_bytes;
            cached_layer->meta_buffer = std::malloc(meta_bytes);
            if (!cached_layer->meta_buffer) {
                return fail("cached metadata allocation failed", il);
            }
            params.mem_size = cached_layer->meta_size;
            params.mem_buffer = cached_layer->meta_buffer;
        } else {
            params.mem_size = ds4_layer_major_meta_arena.size();
            params.mem_buffer = ds4_layer_major_meta_arena.data();
        }
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        if (!ctx) return fail("metadata allocation failed", il);
        if (cached_layer) cached_layer->ctx = ctx;
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 65536, false);
        if (!gf) {
            if (!cached_layer) ggml_free(ctx);
            return fail("graph allocation failed", il);
        }
        if (cached_layer) cached_layer->gf = gf;

        const DeepSeek4Layer & L = w.layers[(size_t) il];
        DeepSeek4LayerCache & lc = cache.layers[(size_t) il];
        const HcLayerWeightsCpu & hlw = hc_weights[(size_t) il];

        // HC pre -> batched attention.
        ggml_tensor * norm_hc = ggml_rms_norm(ctx, state_in, w.hc_eps);
        ggml_tensor * mix_attn = ggml_mul_mat(ctx,
            fc.fn_attn_f16[(size_t) il], norm_hc);
        mix_attn = ggml_reshape_2d(ctx, mix_attn, mix_dim, n_tokens);
        ggml_tensor * attn_base = ds4_fused_hc_base_f32(ctx,
                                                        L.hc_attn_base);
        ggml_tensor * pre_attn = ggml_ds4_hc_pre(
            ctx, mix_attn, attn_base, state_in, n_hc,
            w.n_hc_sinkhorn_iter, hlw.attn.scale_data[0],
            hlw.attn.scale_data[1], hlw.attn.scale_data[2]);
        ggml_tensor * attn_in = ggml_view_2d(ctx, pre_attn, n_embd,
                                             n_tokens, pre_attn->nb[1], 0);
        ggml_tensor * split_attn = ggml_view_2d(
            ctx, pre_attn, mix_dim, n_tokens, pre_attn->nb[1],
            (size_t) n_embd * sizeof(float));

        std::vector<DeepSeek4I32InputBinding> i32_inputs;
        std::vector<DeepSeek4I32ArrayBinding> i32_array_inputs;
        std::vector<DeepSeek4I64ArrayBinding> i64_array_inputs;
        std::vector<DeepSeek4F32ArrayBinding> f32_array_inputs;
        ggml_tensor * attn_normed = build_rms_norm(ctx, attn_in,
                                                   L.attn_norm, w.rms_eps);
        const DeepSeek4AttentionImpl attention_impl =
            cache.prefill_mode == PrefillAttentionMode::Sparse
                ? DeepSeek4AttentionImpl::SparseFlash
                : DeepSeek4AttentionImpl::DenseFlash;
        ggml_tensor * attn_out = build_mla_attention(
            ctx, gf, attn_normed, w, L, lc, il, kv_start, n_tokens,
            nullptr, i32_inputs, i32_array_inputs, i64_array_inputs,
            &f32_array_inputs, attention_impl);
        if (!attn_out) {
            if (!cached_layer) ggml_free(ctx);
            return fail("attention graph build failed", il);
        }
        ggml_tensor * hc_after_attn = ggml_ds4_hc_post(
            ctx, state_in, attn_out, split_attn, n_hc);

        // HC pre -> batched MoE.
        norm_hc = ggml_rms_norm(ctx, hc_after_attn, w.hc_eps);
        ggml_tensor * mix_ffn = ggml_mul_mat(ctx,
            fc.fn_ffn_f16[(size_t) il], norm_hc);
        mix_ffn = ggml_reshape_2d(ctx, mix_ffn, mix_dim, n_tokens);
        ggml_tensor * ffn_base = ds4_fused_hc_base_f32(ctx, L.hc_ffn_base);
        ggml_tensor * pre_ffn = ggml_ds4_hc_pre(
            ctx, mix_ffn, ffn_base, hc_after_attn, n_hc,
            w.n_hc_sinkhorn_iter, hlw.ffn.scale_data[0],
            hlw.ffn.scale_data[1], hlw.ffn.scale_data[2]);
        ggml_tensor * ffn_in = ggml_view_2d(ctx, pre_ffn, n_embd,
                                            n_tokens, pre_ffn->nb[1], 0);
        ggml_tensor * split_ffn = ggml_view_2d(
            ctx, pre_ffn, mix_dim, n_tokens, pre_ffn->nb[1],
            (size_t) n_embd * sizeof(float));
        ggml_tensor * ffn_normed = build_rms_norm(ctx, ffn_in,
                                                  L.ffn_norm, w.rms_eps);
        ggml_tensor * hash_ids = nullptr;
        ggml_tensor * ffn_out = nullptr;
        const bool hash_routed = il < w.n_hash_layer && L.ffn_gate_tid2eid &&
                                 token_ids && hash_tables[(size_t) il].loaded;
        if (hash_routed) {
            hash_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32,
                                          w.n_expert_used, n_tokens);
            ggml_set_input(hash_ids);
            ffn_out = ds4_build_hash_routed_ffn(
                ctx, w, L, ffn_normed, hash_ids, n_tokens);
        } else {
            ffn_out = build_moe_ffn(ctx, ffn_normed, w, L, il, n_tokens);
        }
        if (!ffn_out) {
            if (!cached_layer) ggml_free(ctx);
            return fail("FFN graph build failed", il);
        }
        ggml_tensor * hc_next = ggml_ds4_hc_post(
            ctx, hc_after_attn, ffn_out, split_ffn, n_hc);

        // Persist HC state for the next layer before this layer's gallocr
        // scratch buffer is reused.
        ggml_tensor * state_copy = ggml_cpy(ctx, hc_next, state_out);
        ggml_set_output(state_copy);
        ggml_build_forward_expand(gf, state_copy);

        ggml_tensor * logits = nullptr;
        if (il + 1 == w.n_layer) {
            ggml_tensor * last_hc = ggml_view_2d(
                ctx, hc_next, hc_dim, 1, hc_next->nb[1],
                (size_t) (n_tokens - 1) * hc_next->nb[1]);
            last_hc = ggml_reshape_1d(ctx, last_hc, hc_dim);
            ggml_tensor * out_hc_norm = ggml_rms_norm(ctx, last_hc, w.hc_eps);
            ggml_tensor * out_mix = ggml_mul_mat(ctx, fc.fn_out_f16,
                                                  out_hc_norm);
            out_mix = ggml_reshape_1d(ctx, out_mix, n_hc);
            ggml_tensor * out_base = ds4_fused_hc_base_f32(ctx,
                                                           w.output_hc_base);
            ggml_tensor * final_embd = ggml_ds4_hc_out(
                ctx, out_mix, out_base, last_hc, n_hc,
                hc_out_weights.scale_data[0]);
            ggml_tensor * final_2d = ggml_reshape_2d(ctx, final_embd,
                                                     n_embd, 1);
            ggml_tensor * out_normed = build_rms_norm(ctx, final_2d,
                                                       w.out_norm, w.rms_eps);
            logits = ggml_mul_mat(ctx, w.output, out_normed);
            ggml_set_output(logits);
            ggml_build_forward_expand(gf, logits);
        }

        if (cached_layer) {
            auto remember_unallocated = [&](ggml_tensor * tensor) {
                if (tensor && tensor->data == nullptr &&
                    tensor->buffer == nullptr) {
                    cached_layer->allocated_tensors.push_back(tensor);
                }
            };
            const int n_graph_nodes = ggml_graph_n_nodes(gf);
            for (int i = 0; i < n_graph_nodes; ++i) {
                ggml_tensor * node = ggml_graph_node(gf, i);
                remember_unallocated(node);
                for (int j = 0; j < GGML_MAX_SRC; ++j) {
                    remember_unallocated(node->src[j]);
                }
            }
            auto & tensors = cached_layer->allocated_tensors;
            std::sort(tensors.begin(), tensors.end());
            tensors.erase(std::unique(tensors.begin(), tensors.end()),
                          tensors.end());
        }
        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            if (!cached_layer) ggml_free(ctx);
            return fail("scratch allocation failed", il);
        }
        for (const auto & b : i32_inputs) {
            ggml_backend_tensor_set(b.tensor, &b.value, 0, sizeof(b.value));
        }
        for (const auto & b : i32_array_inputs) {
            ggml_backend_tensor_set(b.tensor, b.values.data(), 0,
                                    sizeof(int32_t) * b.values.size());
        }
        for (const auto & b : i64_array_inputs) {
            ggml_backend_tensor_set(b.tensor, b.values.data(), 0,
                                    sizeof(int64_t) * b.values.size());
        }
        for (const auto & b : f32_array_inputs) {
            ggml_backend_tensor_set(b.tensor, b.values.data(), 0,
                                    sizeof(float) * b.values.size());
        }
        if (hash_ids) {
            const int n_used = w.n_expert_used;
            hash_scratch.resize((size_t) n_used * n_tokens);
            const auto & table = hash_tables[(size_t) il].ids;
            for (int t = 0; t < n_tokens; ++t) {
                std::memcpy(hash_scratch.data() + (size_t) t * n_used,
                            table.data() + (size_t) token_ids[t] * n_used,
                            sizeof(int32_t) * (size_t) n_used);
            }
            ggml_backend_tensor_set(hash_ids, hash_scratch.data(), 0,
                                    sizeof(int32_t) * hash_scratch.size());
        }
        if (telemetry) {
            telemetry->full_graph_build_us += ds4_elapsed_us(
                build_t0, Ds4TimingClock::now());
        }

        const auto compute_t0 = Ds4TimingClock::now();
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            if (!cached_layer) ggml_free(ctx);
            return fail("compute failed", il);
        }
        if (telemetry) {
            telemetry->full_graph_compute_us += ds4_elapsed_us(
                compute_t0, Ds4TimingClock::now());
        }

        if (logits) {
            out_logits.resize((size_t) w.n_vocab);
            ggml_backend_tensor_get(logits, out_logits.data(), 0,
                                    sizeof(float) * (size_t) w.n_vocab);
        }

        const int ratio = (int) w.compress_ratios[(size_t) il];
        if (ratio > 0) {
            lc.n_comp = std::max(lc.n_comp, next_pos / ratio);
            if (ratio == 4) {
                lc.n_index_comp = std::max(lc.n_index_comp,
                                            next_pos / ratio);
            }
        }
        if (cached_layer) {
            cached_layer->i32_inputs = std::move(i32_inputs);
            cached_layer->i32_array_inputs = std::move(i32_array_inputs);
            cached_layer->i64_array_inputs = std::move(i64_array_inputs);
            cached_layer->f32_array_inputs = std::move(f32_array_inputs);
            cached_layer->hash_ids = hash_ids;
            cached_layer->logits = logits;
        } else {
            ggml_free(ctx);
        }
        std::swap(state_in, state_out);
    }

    if (cache_build) {
        graph_cache->ready = true;
    } else {
        ggml_backend_buffer_free(state_buf);
        ggml_free(state_ctx);
    }
    cache.cur_pos = next_pos;
    return out_logits.empty() ? -1 : 1;
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

    // ── Partial layer-range forward with HC ─────────────────────────────
    const int n_embd = w.n_embd;
    const int n_hc = w.n_hc;
    const int hc_dim = n_hc * n_embd;
    const bool is_last_shard = (layer_end >= w.n_layer);

    // A dynamic batch may be supplied by callers other than the DSpark
    // verifier. Split it whenever it spans a learned-compressor boundary:
    // each sub-forward then writes at most one window and, if present, its
    // boundary is the final token. This preserves the same pool/rotate order
    // as sequential execution while retaining safe batched prefixes.
    const int first_chunk = deepseek4_safe_compressor_batch_tokens(w, kv_start, n_tokens);
    if (first_chunk > 0 && first_chunk < n_tokens) {
        const int input_width = layer_begin == 0 ? n_embd : hc_dim;
        std::vector<float> hc_all;
        std::vector<float> shard_out_all;
        std::vector<float> capture_all;
        std::vector<float> logits_all;
        std::vector<float> last_out;
        hc_all.reserve((size_t) hc_dim * n_tokens);
        if (out_logits && !is_last_shard) {
            shard_out_all.reserve((size_t) hc_dim * n_tokens);
        }
        if (verify_hooks && verify_hooks->capture_out &&
            verify_hooks->capture_layer_ids) {
            capture_all.reserve((size_t) verify_hooks->capture_layer_ids->size() *
                                n_embd * n_tokens);
        }
        if (verify_hooks && verify_hooks->all_logits_out) {
            logits_all.reserve((size_t) w.n_vocab * n_tokens);
        }

        for (int off = 0; off < n_tokens;) {
            const int chunk = deepseek4_safe_compressor_batch_tokens(
                w, kv_start + off, n_tokens - off);
            std::vector<float> chunk_hc;
            std::vector<float> chunk_out;
            std::vector<float> chunk_capture;
            std::vector<float> chunk_logits;
            Ds4VerifyHooks chunk_hooks;
            Ds4VerifyHooks * chunk_hooks_ptr = nullptr;
            if (verify_hooks) {
                chunk_hooks.capture_layer_ids = verify_hooks->capture_layer_ids;
                chunk_hooks.capture_out = verify_hooks->capture_out ? &chunk_capture : nullptr;
                chunk_hooks.all_logits_out = verify_hooks->all_logits_out ? &chunk_logits : nullptr;
                chunk_hooks_ptr = &chunk_hooks;
            }
            if (!deepseek4_step_layer_range(
                    backend, w, cache, chunk_hc,
                    embed + (size_t) off * input_width,
                    chunk, kv_start + off, layer_begin, layer_end,
                    out_logits ? &chunk_out : nullptr,
                    token_ids ? token_ids + off : nullptr,
                    telemetry, allow_decode_graph_reuse, chunk_hooks_ptr)) {
                return false;
            }
            hc_all.insert(hc_all.end(), chunk_hc.begin(), chunk_hc.end());
            if (out_logits) {
                if (is_last_shard) {
                    last_out = std::move(chunk_out);
                } else {
                    shard_out_all.insert(shard_out_all.end(),
                                         chunk_out.begin(), chunk_out.end());
                }
            }
            capture_all.insert(capture_all.end(),
                               chunk_capture.begin(), chunk_capture.end());
            logits_all.insert(logits_all.end(), chunk_logits.begin(), chunk_logits.end());
            off += chunk;
        }

        hc_state = std::move(hc_all);
        if (out_logits) {
            *out_logits = is_last_shard ? std::move(last_out) : std::move(shard_out_all);
        }
        if (verify_hooks && verify_hooks->capture_out) {
            *verify_hooks->capture_out = std::move(capture_all);
        }
        if (verify_hooks && verify_hooks->all_logits_out) {
            *verify_hooks->all_logits_out = std::move(logits_all);
        }
        return true;
    }

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

    // Keep all reusable layer-range resources under one model owner so park,
    // unload, and reload have a deterministic teardown boundary.
    DeepSeek4LayerRangeRuntime & runtime = ds4_layer_range_runtime;
    auto & hc_layer_weights_range = runtime.hc_layer_weights;
    auto & hc_output_weights_range = runtime.hc_output_weights;
    auto & hash_routing_tables_range = runtime.hash_routing_tables;
    auto & cached_attn_allocs = runtime.cached_attn_allocs;
    auto & cached_decode_attn_hc_pre_graphs =
        runtime.cached_decode_attn_hc_pre_graphs;
    auto & cached_decode_ffn_hc_pre_graphs =
        runtime.cached_decode_ffn_hc_pre_graphs;
    auto & cached_decode_hc_post_graph = runtime.cached_decode_hc_post_graph;
    auto & cached_decode_attn_graphs = runtime.cached_decode_attn_graphs;
    auto & cached_decode_ffn_graphs = runtime.cached_decode_ffn_graphs;
    auto & cached_decode_output_graph = runtime.cached_decode_output_graph;
    auto & cached_dynamic_output_alloc = runtime.cached_dynamic_output_alloc;
    auto & fused_decode_graph_cache = runtime.fused_decode_graph_cache;
    auto & decode_shared_inputs = runtime.decode_shared_inputs;
    int & hc_loaded_n_layer = runtime.loaded_n_layer;
    const ggml_context * & hc_loaded_ctx = runtime.owner_ctx;
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
            if (il < w.n_hash_layer && L.ffn_gate_tid2eid) {
                load_hash_routing_cpu(hash_routing_tables_range[(size_t)il], L.ffn_gate_tid2eid);
            }
        }
        load_hc_weights_cpu(hc_output_weights_range, w.output_hc_fn, w.output_hc_scale, w.output_hc_base);
        hc_loaded_n_layer = w.n_layer;
        hc_loaded_ctx = w.ctx;
    }

    // Per-layer execution with CPU-side HC
    DeepSeek4LayerRangeScratch & scratch = runtime.scratch;
    const int n_expert_used = ds4_effective_expert_count(w);
    scratch.ensure(w.ctx, n_tokens, n_embd, n_hc, n_expert_used);

    // Large full-model prefill batches use the device-resident layer-major
    // pipeline. DSpark verification remains on its exact q=2..4 path below.
    if (n_tokens > 4 &&
        n_tokens <= DS4_MAX_LAYER_MAJOR_PREFILL_TOKENS && layer_begin == 0 &&
        is_last_shard && out_logits && ds4_backend_is_gpu(backend)) {
        const int prc = ds4_try_layer_major_prefill(
            fused_decode_graph_cache, backend, w, cache,
            hc_layer_weights_range, hc_output_weights_range,
            hash_routing_tables_range, scratch.hash_expert_ids, embed,
            n_tokens, kv_start, *out_logits, token_ids, telemetry);
        if (prc < 0) return false;
        if (prc > 0) {
            if (telemetry) {
                telemetry->total_us += ds4_elapsed_us(
                    step_t0, Ds4TimingClock::now());
            }
            return true;
        }
    }

    if (n_tokens >= 2 && n_tokens <= 4 && verify_hooks && layer_begin == 0 && is_last_shard &&
        out_logits && ds4_backend_is_gpu(backend) && w.fused_decode &&
        ds4_fused_verify_enabled()) {
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
        out_logits && ds4_backend_is_gpu(backend) && w.fused_decode) {
        const int rc = ds4_try_fused_decode_step(
            fused_decode_graph_cache, backend, w, cache, hc_layer_weights_range,
            hc_output_weights_range, hash_routing_tables_range, scratch.hash_expert_ids,
            embed, kv_start, *out_logits, token_ids, telemetry);
        if (rc < 0) return false;
        if (rc > 0) {
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
    // Verification hooks need per-position captures and logits. The cached
    // single-token decode graphs only expose the final decode outputs, so
    // honor the caller's request to take the dynamic path for verification.
    const bool reuse_decode_graphs = n_tokens == 1 && allow_decode_graph_reuse;
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
        backend_decode_hc_supported;
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

            const bool exact_tokenwise_prefill =
                !reuse_decode_attn && n_tokens > 1;
            if (exact_tokenwise_prefill) {
                const DeepSeek4AttentionImpl attention_impl =
                    cache.prefill_mode == PrefillAttentionMode::Sparse
                        ? DeepSeek4AttentionImpl::SparseFlash
                        : DeepSeek4AttentionImpl::Explicit;
                if (!ds4_run_exact_tokenwise_prefill_attention(
                        backend, w, L, lc, il, cur.data(), n_tokens, kv_start,
                        attention_impl, attn_out_host,
                        cached_attn_allocs[(size_t) il], telemetry)) {
                    return false;
                }
            } else if (reuse_decode_attn) {
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
                const DeepSeek4AttentionImpl attention_impl =
                    cache.prefill_mode == PrefillAttentionMode::Sparse
                        ? DeepSeek4AttentionImpl::SparseFlash
                        : DeepSeek4AttentionImpl::Explicit;
                attn_out = build_mla_attention(ctx, gf, normed, w, L, lc, il,
                                               kv_start, n_tokens, nullptr,
                                               i32_inputs, i32_array_inputs,
                                               i64_array_inputs,
                                               &f32_array_inputs,
                                               attention_impl);
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

            if (!exact_tokenwise_prefill) {
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
            }

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
                const int n_used = n_expert_used;
                hash_expert_ids_host.resize((size_t)n_used * (size_t)n_tokens);
                for (int ti = 0; ti < n_tokens; ti++) {
                    const int32_t tok = token_ids[ti];
                    const int32_t * row = hash_routing_row(
                        hash_routing_tables_range[(size_t)il], tok, w.n_expert_used);
                    if (!row) {
                        std::fprintf(stderr,
                                     "[deepseek4] token id %d outside hash table for layer %d\n",
                                     tok, il);
                        return false;
                    }
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
                cached.n_expert_used != n_expert_used ||
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

            const bool last_only = n_tokens > 1;
            const int output_tokens = last_only ? 1 : n_tokens;
            ggml_tensor * inp = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, n_embd, output_tokens);
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
            const float * output_input = last_only
                ? final_embd.data() + (size_t)(n_tokens - 1) * n_embd
                : final_embd.data();
            ggml_backend_tensor_set(inp, output_input, 0,
                                    sizeof(float) * (size_t)n_embd * output_tokens);
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                ggml_free(ctx);
                return false;
            }

            out_logits->resize((size_t)w.n_vocab);
            const size_t logits_offset = last_only ? 0 :
                (size_t)(n_tokens - 1) * (size_t)w.n_vocab * sizeof(float);
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

    // Update compressor state.  Multi-token prefill may cross one or more
    // boundaries even when the chunk itself does not end on a boundary.
    const int next_pos = kv_start + n_tokens;
    for (int il = layer_begin; il < layer_end; ++il) {
        const uint32_t ratio = w.compress_ratios[il];
        if (ratio <= 0) continue;
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

void free_deepseek4_cache(DeepSeek4Cache & c) {
    if (c.ctx) { ggml_free(c.ctx); c.ctx = nullptr; }
    if (c.buf) { ggml_backend_buffer_free(c.buf); c.buf = nullptr; }
    c.layers.clear();
    c.hc_state = nullptr;
}

void reset_deepseek4_cache(DeepSeek4Cache & c) {
    c.cur_pos = 0;
    for (DeepSeek4LayerCache & lc : c.layers) {
        lc.n_comp = 0;
        lc.n_index_comp = 0;
    }
    if (c.buf) {
        ggml_backend_buffer_clear(c.buf, 0);
    }
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
    if (s.buf) { ggml_backend_buffer_free(s.buf); s.buf = nullptr; }
    if (s.ctx) { ggml_free(s.ctx); s.ctx = nullptr; }
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
    ggml_backend_t backend = nullptr;
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
    ggml_tensor * confidence_out = nullptr;
    std::vector<std::pair<std::string, ggml_tensor *>> dbg_taps;
    // HC scales are immutable weights: read from the backend once.
    std::vector<std::array<float, 3>> s_attn, s_ffn;
    float s_out = 0.0f;
};

thread_local DsparkDraftCache g_dspark_draft_cache;

}  // namespace

void reset_deepseek4_dspark_runtime_cache() {
    DsparkDraftCache & cache = g_dspark_draft_cache;
    if (cache.alloc) {
        ggml_gallocr_free(cache.alloc);
        cache.alloc = nullptr;
    }
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
    cache = DsparkDraftCache{};
}

bool deepseek4_dspark_draft_forward(ggml_backend_t backend,
                                    const DSparkDrafter & d,
                                    const float * noise_embed,
                                    const float * ctx_features,
                                    int ctx_len,
                                    int committed,
                                    std::vector<float> & out_hidden,
                                    std::vector<float> * confidence_hidden) {
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

    if (C.drafter != (const void *) &d || C.backend != backend) {
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

    if (!C.ctx || C.ctx_len != ctx_len || C.block != block ||
        C.drafter != (const void *) &d || C.backend != backend) {
        // ── (Re)build the graph ─────────────────────────────────────────
        if (C.backend != backend && C.alloc) {
            ggml_gallocr_free(C.alloc);
            C.alloc = nullptr;
        }
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
        // The tied lm_head consumes the normalized state. The confidence head
        // was trained on the HC-collapsed state before this output RMSNorm, so
        // keep both instead of reusing the normalized state for confidence.
        ggml_tensor * out = nullptr;
        ggml_tensor * confidence_out = nullptr;
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
            confidence_out = confidence_out
                           ? ggml_concat(ctx, confidence_out, final_2d, 1)
                           : final_2d;
        }
        ggml_set_output(out);
        ggml_set_output(confidence_out);
        ggml_build_forward_expand(gf, out);
        ggml_build_forward_expand(gf, confidence_out);
        C.out = out;
        C.confidence_out = confidence_out;

        if (!C.alloc) C.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!C.alloc || !ggml_gallocr_alloc_graph(C.alloc, gf)) {
            ggml_free(C.ctx); C.ctx = nullptr; C.gf = nullptr;
            return false;
        }
        C.ctx_len = ctx_len;
        C.block   = block;
        C.drafter = (const void *) &d;
        C.backend = backend;
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
    if (confidence_hidden) {
        confidence_hidden->resize((size_t) n_embd * block);
        ggml_backend_tensor_get(C.confidence_out, confidence_hidden->data(), 0,
                                sizeof(float) * confidence_hidden->size());
    }

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
