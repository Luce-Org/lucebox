// Qwen3.5-0.8B drafter scoring for pflash speculative prefill.
//
// Two scorers share these weights:
//   - qwen35_score_and_compress        : the original all-layer running-max
//                                        scorer, on the Qwen3.5 architecture
//   - qwen35_strict_score_and_compress : blocks 0..14 plus the block-15 NoPE
//                                        Q/K scoring head, under strict
//                                        budget selection
//
// Loading lives in qwen35_loader.cpp; pflash_drafter.cpp dispatches into
// qwen35_drafter_score_and_compress.

#include "qwen35_drafter.h"

#include "pflash_drafter.h"
#include "pflash_compress.h"
#include "pflash_seg_kcos.h"
#include "pflash_selection.h"
#include "pflash_query_candidates.h"
#include "common/gguf_inspect.h"
#include "anchor_params.h"
#include "internal.h"
#include "kv_quant.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>

namespace luce::common {

namespace {

static int align_up_i(int x, int a) { return ((x + a - 1) / a) * a; }

// Bytes of build_qwen35_causal_mask's mask, the part of a layer graph that
// grows with the context.
static size_t causal_mask_bytes(int kv_start, int n_tokens) {
    return (size_t)align_up_i(kv_start + n_tokens, 256) * align_up_i(n_tokens, 32) * sizeof(uint16_t);
}

// The drafter cache's KV types: the target's resolution starting from F16
// (the precision the scoring head was trained at, with no K/Q rotation) and
// without the LUCE_KV_TQ3 shorthand, since the drafter never wants the
// TurboQuant rotation. Resolved without touching the environment, which
// other threads read while a concurrent server scores a prompt.
struct DrafterKvTypes {
    ggml_type k = GGML_TYPE_COUNT;
    ggml_type v = GGML_TYPE_COUNT;
    explicit DrafterKvTypes(ggml_type default_type = GGML_TYPE_F16) {
        luce::resolve_kv_types(k, v, GGML_TYPE_COUNT, GGML_TYPE_COUNT,
                               /*tq3_shorthand=*/false, /*default_type=*/default_type);
    }
};

} // namespace

ggml_tensor * build_qwen35_causal_mask(ggml_context * ctx, ggml_cgraph * gf,
                                       int kv_start, int n_tokens) {
    // Keys padded to 256: the full-attention block reads a K view padded to
    // the 256-token flash-attention stride for masked prefill spans
    // (build_full_attn_block, padded_prefill), and every key column it reads
    // must lie inside a mask row.
    const int kv_pad = align_up_i(kv_start + n_tokens, 256);
    const int q_pad = align_up_i(n_tokens, 32);
    // Keys [kv_start, kv_pad): row q is zero through key kv_start + q, then
    // -inf, the kv padding included; n_tokens + 255 keys at most.
    const int tail = kv_pad - kv_start;
    // ggml_fill and ggml_tri take F32 only; the big tensor is written in F16.
    ggml_tensor * zero = ggml_cast(ctx,
        ggml_fill_inplace(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), 0.0f), GGML_TYPE_F16);
    ggml_tensor * mask = ggml_repeat_4d(ctx, zero, kv_pad, q_pad, 1, 1);
    ggml_tensor * diag = ggml_tri(ctx,
        ggml_fill_inplace(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tail, tail), -INFINITY),
        GGML_TRI_TYPE_UPPER);
    ggml_build_forward_expand(gf, ggml_cpy(ctx,
        ggml_view_2d(ctx, diag, tail, n_tokens, diag->nb[1], 0),
        ggml_view_2d(ctx, mask, tail, n_tokens, mask->nb[1], (size_t)kv_start * mask->nb[0])));
    if (q_pad > n_tokens) {
        ggml_tensor * pad_rows = ggml_fill_inplace(ctx,
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_pad, q_pad - n_tokens), -INFINITY);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, pad_rows,
            ggml_view_2d(ctx, mask, kv_pad, q_pad - n_tokens, mask->nb[1],
                         (size_t)n_tokens * mask->nb[1])));
    }
    // The attention reads the mask after these copies: they come first in gf.
    return mask;
}

std::vector<int32_t> qwen35_score_and_compress(
    TargetWeights & w,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel,
    int score_query_end,
    const luce::pflash::PFlashSelectionConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans,
    std::vector<float> * token_scores_out) {

    const int S = (int)ids.size();
    const int hidden = w.n_embd;
    if (S < n_lookahead + 1) return ids;
    const int query_end = score_query_end < 0 ? S : score_query_end;
    if (n_lookahead < 1 || query_end < n_lookahead || query_end > S) {
        set_last_error("qwen35 scorer query window out of range");
        return {};
    }
    const int query_start = query_end - n_lookahead;

    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> running_max((size_t)n_lookahead * S, -INFINITY);

    TargetCache cache;
    const DrafterKvTypes kv(GGML_TYPE_Q4_0);
    if (!create_target_cache(w, S, 0, w.backend, cache, true, 0,
                             /*paged_attention=*/false, 1,
                             /*concurrent_tree=*/false, kv.k, kv.v)) {
        return {};
    }

    ggml_init_params act_ip{};
    act_ip.mem_size = (size_t)8 * ggml_tensor_overhead() + 4096;
    act_ip.no_alloc = true;
    ggml_context * act_ctx = ggml_init(act_ip);
    if (!act_ctx) {
        free_target_cache(cache);
        set_last_error("qwen35 drafter activation ctx init failed");
        return {};
    }
    ggml_tensor * act_in = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, S);
    ggml_tensor * act_out = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, S);
    ggml_backend_buffer_t act_buf = ggml_backend_alloc_ctx_tensors(act_ctx, w.backend);
    if (!act_buf) {
        ggml_free(act_ctx);
        free_target_cache(cache);
        set_last_error("qwen35 drafter activation allocation failed");
        return {};
    }

    {
        const int batch = 2048;
        std::vector<float> emb((size_t)hidden * batch);
        for (int i = 0; i < S; i += batch) {
            const int n = std::min(batch, S - i);
            if (!w.embedder.embed(ids.data() + i, n, emb.data())) {
                ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 drafter embedding failed");
                return {};
            }
            ggml_backend_tensor_set(act_in, emb.data(), (size_t)i * act_in->nb[1], (size_t)hidden * n * sizeof(float));
        }
    }

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
    const int ubatch = 1024;
    // Graph of layer il over tokens [start, start + n), writing act_out.
    // nullptr (error set) on failure.
    const auto layer_graph = [&](int il, int start, int n, ggml_cgraph *& gf,
                                 ggml_tensor *& pos) -> ggml_context * {
        ggml_init_params ip{};
        ip.mem_size = 512 * 1024 * 1024;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            set_last_error("qwen35 drafter layer graph ctx init failed");
            return nullptr;
        }
        gf = ggml_new_graph_custom(ctx, 16384, false);
        ggml_tensor * inp = ggml_view_2d(ctx, act_in, hidden, n, act_in->nb[1], (size_t)start * act_in->nb[1]);
        pos = nullptr;
        ggml_tensor * mask = nullptr;
        if (((il + 1) % w.full_attention_interval) == 0) {
            pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n);
            ggml_set_input(pos);
            mask = build_qwen35_causal_mask(ctx, gf, start, n);
        }
        ggml_tensor * out = build_qwen35_layer(ctx, gf, w, cache, il, inp, pos, mask, start, n, false, 0);
        ggml_tensor * dst = ggml_view_2d(ctx, act_out, hidden, n, act_out->nb[1], (size_t)start * act_out->nb[1]);
        if (ggml_nelements(out) != ggml_nelements(dst)) {
            std::fprintf(stderr,
                "[qwen35-drafter] layer output shape mismatch il=%d start=%d out=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld]\n",
                il, start,
                (long long)out->ne[0], (long long)out->ne[1], (long long)out->ne[2], (long long)out->ne[3],
                (long long)dst->ne[0], (long long)dst->ne[1], (long long)dst->ne[2], (long long)dst->ne[3]);
            ggml_free(ctx);
            set_last_error("qwen35 layer output shape mismatch");
            return nullptr;
        }
        ggml_build_forward_expand(gf, ggml_cpy(ctx, out, dst));
        return ctx;
    };
    // Size the compute buffer once, for a full-attention layer over the
    // ubatch with the largest mask: allocating per ubatch would regrow it
    // with every ubatch of the first attention layer.
    if (w.full_attention_interval <= w.n_layer) {
        int worst = 0;
        for (int start = 0; start < S; start += ubatch) {
            if (causal_mask_bytes(start, std::min(ubatch, S - start)) >=
                causal_mask_bytes(worst, std::min(ubatch, S - worst))) worst = start;
        }
        ggml_cgraph * gf = nullptr;
        ggml_tensor * pos = nullptr;
        ggml_context * ctx = layer_graph(w.full_attention_interval - 1, worst,
                                         std::min(ubatch, S - worst), gf, pos);
        if (!ctx) {
            ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
            return {};
        }
        const bool reserved = ggml_gallocr_reserve(alloc, gf);
        ggml_free(ctx);
        if (!reserved) {
            ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
            set_last_error("qwen35 drafter graph allocation failed");
            return {};
        }
    }
    for (int il = 0; il < w.n_layer; ++il) {
        const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);
        int fa_idx = 0;
        if (is_attn) {
            for (int k = 0; k < il; ++k) if (((k + 1) % w.full_attention_interval) == 0) ++fa_idx;
        }
        for (int start = 0; start < S; start += ubatch) {
            const int n = std::min(ubatch, S - start);
            ggml_cgraph * gf = nullptr;
            ggml_tensor * pos = nullptr;
            ggml_context * ctx = layer_graph(il, start, n, gf, pos);
            if (!ctx) {
                ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                return {};
            }
            if (!ggml_gallocr_alloc_graph(alloc, gf)) {
                ggml_free(ctx); ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 drafter graph allocation failed");
                return {};
            }
            if (is_attn) {
                std::vector<int32_t> p4((size_t)4 * n, 0);
                for (int i = 0; i < n; ++i) {
                    int p = start + i;
                    p4[(size_t)0 * n + i] = p;
                    p4[(size_t)1 * n + i] = p;
                    p4[(size_t)2 * n + i] = p;
                }
                ggml_backend_tensor_set(pos, p4.data(), 0, p4.size() * sizeof(int32_t));
            }
            auto st = ggml_backend_graph_compute(w.backend, gf);
            ggml_free(ctx);
            if (st != GGML_STATUS_SUCCESS) {
                ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 drafter graph compute failed");
                return {};
            }
        }

        if (is_attn) {
            ggml_init_params sip{};
            sip.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(1024, false) + 64 * 1024;
            sip.no_alloc = true;
            ggml_context * sctx = ggml_init(sip);
            if (!sctx) {
                ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 score graph ctx allocation failed");
                return {};
            }
            ggml_cgraph * sgf = ggml_new_graph_custom(sctx, 1024, false);
            const int K_len = (int) cache.attn_k[(size_t)fa_idx]->ne[1];
            ggml_tensor * mask_tail = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, K_len, n_lookahead);
            ggml_tensor * K_f32 = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, w.n_embd_head_k, K_len, w.n_head_kv);
            ggml_tensor * K_cast = ggml_cpy(sctx, cache.attn_k[(size_t)fa_idx], K_f32);
            ggml_tensor * K_score = nullptr;
            if (w.n_head != w.n_head_kv) {
                const int gqa = w.n_head / w.n_head_kv;
                ggml_tensor * K_4d = ggml_reshape_4d(sctx, K_cast, w.n_embd_head_k, K_len, 1, w.n_head_kv);
                ggml_tensor * K_tpl = ggml_new_tensor_4d(sctx, GGML_TYPE_F32, w.n_embd_head_k, K_len, gqa, w.n_head_kv);
                ggml_tensor * K_rep = ggml_repeat(sctx, K_4d, K_tpl);
                K_score = ggml_reshape_3d(sctx, K_rep, w.n_embd_head_k, K_len, w.n_head);
            } else {
                K_score = K_cast;
            }
            const TargetLayer & L = w.layers[il];
            ggml_tensor * inp_tail = ggml_view_2d(sctx, act_in, hidden, n_lookahead,
                act_in->nb[1], (size_t)query_start * act_in->nb[1]);
            ggml_tensor * q_cur = ggml_rms_norm(sctx, inp_tail, w.rms_eps);
            q_cur = ggml_mul(sctx, q_cur, L.attn_norm);
            ggml_tensor * QG = ggml_mul_mat(sctx, L.wq, q_cur);
            QG = ggml_reshape_3d(sctx, QG, w.n_embd_head_k * 2, w.n_head, n_lookahead);
            ggml_tensor * Q = ggml_view_3d(sctx, QG,
                w.n_embd_head_k, w.n_head, n_lookahead,
                ggml_element_size(QG) * w.n_embd_head_k * 2,
                ggml_element_size(QG) * w.n_embd_head_k * 2 * w.n_head,
                0);
            Q = ggml_rms_norm(sctx, Q, w.rms_eps);
            Q = ggml_mul(sctx, Q, L.q_norm);
            ggml_tensor * pos_tail = ggml_new_tensor_1d(sctx, GGML_TYPE_I32, 4 * n_lookahead);
            int sections[4];
            for (int k = 0; k < 4; ++k) sections[k] = w.rope_sections[k];
            Q = ggml_rope_multi(sctx, Q, pos_tail, nullptr,
                                w.rope_dimension_count, sections, GGML_ROPE_TYPE_MROPE,
                                0, w.rope_theta, 1.0f,
                                0.0f, 1.0f, 0.0f, 0.0f);
            ggml_tensor * Q_tail_perm = ggml_cont(sctx, ggml_permute(sctx, Q, 0, 2, 1, 3));
            ggml_tensor * attn_score = ggml_mul_mat(sctx, K_score, Q_tail_perm);
            ggml_tensor * probs = ggml_soft_max_ext(sctx, attn_score, mask_tail, 1.0f / std::sqrt((float)w.n_embd_head_k), 0.0f);
            ggml_set_output(probs);
            ggml_build_forward_expand(sgf, probs);
            ggml_gallocr_t salloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
            if (!ggml_gallocr_alloc_graph(salloc, sgf)) {
                ggml_gallocr_free(salloc); ggml_free(sctx); ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 score graph allocation failed");
                return {};
            }
            std::vector<int32_t> pos4((size_t)4 * n_lookahead, 0);
            for (int i = 0; i < n_lookahead; ++i) {
                const int p = query_start + i;
                pos4[(size_t)0 * n_lookahead + i] = p;
                pos4[(size_t)1 * n_lookahead + i] = p;
                pos4[(size_t)2 * n_lookahead + i] = p;
            }
            ggml_backend_tensor_set(pos_tail, pos4.data(), 0, pos4.size() * sizeof(int32_t));
            std::vector<float> mask((size_t)n_lookahead * K_len, 0.0f);
            for (int t = 0; t < n_lookahead; ++t) {
                const int visible_end = query_start + t + 1;
                for (int j = 0; j < K_len; ++j) {
                    mask[(size_t)t * K_len + j] = (j < visible_end) ? 0.0f : -INFINITY;
                }
            }
            ggml_backend_tensor_set(mask_tail, mask.data(), 0, mask.size() * sizeof(float));
            auto st = ggml_backend_graph_compute(w.backend, sgf);
            if (st != GGML_STATUS_SUCCESS) {
                ggml_gallocr_free(salloc); ggml_free(sctx); ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 score graph compute failed");
                return {};
            }
            std::vector<float> tmp((size_t)K_len * n_lookahead * w.n_head);
            ggml_backend_tensor_get(probs, tmp.data(), 0, tmp.size() * sizeof(float));
            const size_t nonfinite =
                count_nonfinite_scores(tmp.data(), tmp.size());
            if (nonfinite != 0) {
                const std::string message =
                    "non-finite Qwen3.5 PFlash scores at layer " +
                    std::to_string(il) + ": " + std::to_string(nonfinite) +
                    "/" + std::to_string(tmp.size());
                std::fprintf(stderr, "[pflash] ERROR: %s\n", message.c_str());
                std::fflush(stderr);
                ggml_gallocr_free(salloc); ggml_free(sctx);
                ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf);
                ggml_free(act_ctx); free_target_cache(cache);
                set_last_error(message);
                return {};
            }
            for (int h = 0; h < w.n_head; ++h) {
                for (int t = 0; t < n_lookahead; ++t) {
                    for (int j = 0; j < S; ++j) {
                        const size_t src = (size_t)h * K_len * n_lookahead + (size_t)t * K_len + j;
                        const size_t dst = (size_t)t * S + j;
                        running_max[dst] = std::max(running_max[dst], tmp[src]);
                    }
                }
            }
            ggml_gallocr_free(salloc);
            ggml_free(sctx);
        }
        std::swap(act_in, act_out);
    }
    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(act_buf);
    ggml_free(act_ctx);
    free_target_cache(cache);

    std::vector<float> score((size_t)S, 0.0f);
    for (int j = 0; j < S; ++j) {
        float s = 0.0f;
        for (int t = 0; t < n_lookahead; ++t) s += running_max[(size_t)t * S + j];
        score[(size_t)j] = s / (float)n_lookahead;
    }

    const int n_chunks = (S + chunk_size - 1) / chunk_size;
    const int n_keep = std::max(1, (int)((float)n_chunks * keep_ratio));
    
    std::vector<float> smooth_score = score;
    // Caller pool_kernel takes precedence; if zero/negative, fall back to env or 5.
    const int pk = (pool_kernel > 0)
        ? pool_kernel
        : std::max(3, env_int("LUCE_COMPRESS_POOL_KERNEL", 5));
    std::vector<float> smoothed((size_t)S, 0.0f);
    int half = pk / 2;
    for (int j = 0; j < S; ++j) {
        int lo = std::max(0, j - half);
        int hi = std::min(S - 1, j + half);
        float s = 0.0f;
        int n = 0;
        for (int k = lo; k <= hi; ++k) { s += score[(size_t)k]; ++n; }
        smoothed[(size_t)j] = (n > 0) ? (s / (float)n) : 0.0f;
    }
    smooth_score.swap(smoothed);

    if (token_scores_out) {
        // Scoring only (two-scorer selection): hand the smoothed per-token
        // scores back and let the caller select.
        *token_scores_out = smooth_score;
        return ids;
    }

    if (experiment.selection_active) {
        return select_pflash_chunks(
            ids, smooth_score, keep_ratio, n_lookahead, score_query_end,
            pk, experiment, required_instruction_spans, false, true);
    }
    
    std::vector<std::pair<float, int>> chunk_means;
    for (int c = 0; c < n_chunks; ++c) {
        int lo = c * chunk_size, hi = std::min(S, lo + chunk_size);
        float s = 0.0f;
        for (int j = lo; j < hi; ++j) s += smooth_score[(size_t)j];
        chunk_means.push_back({s / std::max(1, hi - lo), c});
    }
    std::sort(chunk_means.begin(), chunk_means.end(), [](auto a, auto b) { return a.first > b.first; });
    
    std::vector<uint8_t> selected((size_t)n_chunks, 0);
    int count = 0;
    // Scale head/tail forced chunks so they don't crowd out top-K scoring.
    {
        const int h_raw = env_int("LUCE_COMPRESS_HEAD_CHUNKS", 8);
        const int t_raw = env_int("LUCE_COMPRESS_TAIL_CHUNKS", 24);
        int h_n = h_raw, t_n = t_raw;
        if (h_n + t_n >= n_keep) {
            const int budget = std::max(1, n_keep - 1);
            h_n = std::max(0, h_raw * budget / (h_raw + t_raw));
            t_n = std::max(0, budget - h_n);
        }
        for (int c = 0; c < std::min(n_chunks, h_n); ++c) { selected[(size_t)c] = 1; ++count; }
        for (int c = std::max(0, n_chunks - t_n); c < n_chunks; ++c) if (!selected[(size_t)c]) { selected[(size_t)c] = 1; ++count; }
    }

    const int query_tokens = env_int("LUCE_COMPRESS_QUERY_TOKENS", 96);
    const auto ap = resolve_anchor_params(n_chunks,
        -1, -1,
        env_int("LUCE_COMPRESS_ANCHOR_RADIUS",   -1),
        env_int("LUCE_COMPRESS_MAX_ANCHOR_HITS", -1));
    const int anchor_radius   = ap.radius;
    const int max_anchor_hits = ap.max_hits;
    std::vector<uint8_t> forced((size_t)n_chunks, 0);

    const int q0 = std::max(0, S - query_tokens);
    constexpr int NGRAM = 4;
    for (int q = q0; q + NGRAM <= S; ++q) {
        int hits = 0;
        std::vector<int> hit_pos(max_anchor_hits);
        const int search_end = std::max(0, q0 - NGRAM);
        for (int p = 0; p <= search_end && hits <= max_anchor_hits; ++p) {
            bool same = true;
            for (int k = 0; k < NGRAM; ++k) {
                if (ids[(size_t)p + k] != ids[(size_t)q + k]) { same = false; break; }
            }
            if (same) {
                if (hits < max_anchor_hits) hit_pos[hits] = p;
                ++hits;
            }
        }
        if (hits > 0 && hits <= max_anchor_hits) {
            for (int i = 0; i < hits && i < max_anchor_hits; ++i) {
                force_chunk_neighborhood(forced, n_chunks, hit_pos[i] / chunk_size, anchor_radius);
            }
        }
    }

    for (int c = 0; c < n_chunks; ++c) {
        if (forced[(size_t)c] && !selected[(size_t)c]) {
            selected[(size_t)c] = 1;
            ++count;
        }
    }

    // Global aggregation tasks often depend on repeated rare tokens that do
    // not appear in the final query. Preserve high-frequency-but-not-filler
    // token chunks before filling with model-score top-K.
    const int repeat_min = env_int("LUCE_COMPRESS_REPEAT_MIN", 4);
    const int repeat_max = env_int("LUCE_COMPRESS_REPEAT_MAX", 32);
    const int repeat_limit = env_int("LUCE_COMPRESS_REPEAT_CHUNKS", n_keep);
    if (repeat_min > 1 && count < repeat_limit) {
        std::unordered_map<int32_t, int> freq;
        freq.reserve((size_t)S);
        const int repeat_scan_end = std::max(0, S - query_tokens);
        for (int j = 0; j < repeat_scan_end; ++j) {
            ++freq[ids[(size_t)j]];
        }
        std::vector<std::pair<int, int32_t>> repeated;
        repeated.reserve(freq.size());
        for (const auto & kv : freq) {
            if (kv.second >= repeat_min && kv.second <= repeat_max) {
                repeated.push_back({kv.second, kv.first});
            }
        }
        std::sort(repeated.begin(), repeated.end(), [](const auto & a, const auto & b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });
        for (const auto & rp : repeated) {
            if (count >= repeat_limit) break;
            const int32_t tok = rp.second;
            for (int j = 0; j < repeat_scan_end && count < repeat_limit; ++j) {
                if (ids[(size_t)j] != tok) continue;
                const int c = j / chunk_size;
                if (!selected[(size_t)c]) {
                    selected[(size_t)c] = 1;
                    ++count;
                }
            }
        }
    }
    
    for (auto [_, c] : chunk_means) {
        if (count >= n_keep) break;
        if (!selected[(size_t)c]) { selected[(size_t)c] = 1; ++count; }
    }
    
    std::vector<int32_t> out_ids;
    std::vector<int> selected_chunks;
    for (int c = 0; c < n_chunks; ++c) {
        if (selected[(size_t)c]) selected_chunks.push_back(c);
    }
    int span_start = -1, span_end = -1;
    for (int c : selected_chunks) {
        int s_ = c * chunk_size;
        int e_ = std::min(S, (c + 1) * chunk_size);
        if (span_start < 0) {
            span_start = s_; span_end = e_;
        } else if (s_ == span_end) {
            span_end = e_;
        } else {
            for (int j = span_start; j < span_end; ++j) out_ids.push_back(ids[j]);
            span_start = s_; span_end = e_;
        }
    }
    if (span_start >= 0) {
        for (int j = span_start; j < span_end; ++j) out_ids.push_back(ids[j]);
    }

    auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[qwen35-drafter] forward+compress %.2fs S=%d kept=%zu (%d/%d chunks)\n",
                 std::chrono::duration<double>(t1 - t0).count(), S, out_ids.size(), count, n_chunks);
    std::fflush(stderr);
    return out_ids;
}

void free_qwen35_scoring_session(Qwen35ScoringSession & session) {
    free_target_cache(session.cache);
    session.cache = TargetCache{};
    if (session.key_buf) ggml_backend_buffer_free(session.key_buf);
    if (session.key_ctx) ggml_free(session.key_ctx);
    session.key_buf = nullptr;
    session.key_ctx = nullptr;
    session.keys = nullptr;
    session.capacity = 0;
    session.ids.clear();
    session.checkpoint = 0;
    session.probe_raw.clear();
    session.subunit_raw.clear();
    session.container_raw.clear();
    session.request_raw.clear();
    session.request_rows.clear();
    session.query_windows.clear();
    session.query_sets.clear();
    session.host_id = 0;
    session.host_valid = false;
}

namespace {

int scoring_session_limit() {
    const char * raw = std::getenv("PFLASH_DRAFTER_SESSIONS");
    if (!raw || !*raw) return 2;
    char * end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value < 0) return 2;
    return (int) std::min<long>(value, 16);
}

bool allocate_scoring_session(TargetWeights & w, int capacity,
                              Qwen35ScoringSession & session) {
    const DrafterKvTypes kv;
    if (!create_target_cache_partial(w, capacity, 0, w.backend, session.cache,
                                     /*prefill_only=*/true, 0, kQwen35HeadBlock,
                                     /*allocate_target_feat=*/false, 0,
                                     /*f32_ssm_intermediates=*/false,
                                     /*paged_attention=*/false, 1,
                                     /*concurrent_tree=*/false, kv.k, kv.v)) {
        return false;
    }
    if (!ensure_ssm_snapshot(session.cache, w.backend)) {
        free_qwen35_scoring_session(session);
        return false;
    }
    ggml_init_params kp{};
    kp.mem_size = ggml_tensor_overhead() + 1024;
    kp.no_alloc = true;
    session.key_ctx = ggml_init(kp);
    if (session.key_ctx) {
        session.keys = ggml_new_tensor_3d(session.key_ctx, GGML_TYPE_F32,
                                          w.n_embd_head_k, w.n_head_kv, capacity);
        session.key_buf = ggml_backend_alloc_ctx_tensors(session.key_ctx, w.backend);
    }
    if (!session.key_buf) {
        free_qwen35_scoring_session(session);
        return false;
    }
    session.capacity = capacity;
    return true;
}

void zero_recurrent_state(TargetCache & cache) {
    for (size_t i = 0; i < cache.ssm_state.size(); ++i) {
        if (cache.ssm_state[i]) {
            ggml_backend_tensor_memset(cache.ssm_state[i], 0, 0,
                                       ggml_nbytes(cache.ssm_state[i]));
        }
        if (i < cache.conv_state.size() && cache.conv_state[i]) {
            ggml_backend_tensor_memset(cache.conv_state[i], 0, 0,
                                       ggml_nbytes(cache.conv_state[i]));
        }
    }
}

// The session to score ``ids`` with, and the token it resumes from: a stored
// session's checkpoint (pflash_scoring_resume: on the 8192 grid, inside the
// shared prefix, at or before S - 64 and at or before ``resume_cap``), so the
// resumed call computes every value exactly as a fresh call does -- provided
// the query rows and probe logits the scoring needs are covered too.
// Otherwise a session that holds no prompt (one that just failed), else the
// least recently used session (or ``scratch`` with sessions off) starts over
// from token 0.
Qwen35ScoringSession * acquire_scoring_session(
        Qwen35DrafterState & st,
        const std::vector<int32_t> & ids,
        int query_start,
        int query_end,
        bool need_probe,
        bool need_subunit,
        bool need_container,
        bool need_request,
        const std::vector<PFlashTokenSpan> & other_windows,
        int resume_cap,
        int & resume,
        int & shared_prefix,
        std::unique_ptr<Qwen35ScoringSession> & scratch) {
    TargetWeights & w = st.weights;
    const int S = (int) ids.size();
    // Attention keys padded to 256: a cache shorter than that clamps the
    // padded K view and moves the tail ubatches off the MMA kernel, which
    // changes the scores (resident drafter vs a freshly loaded one).
    const int S_padded = align_up_i(S, 256);
    const int limit = scoring_session_limit();
    const size_t row_floats = (size_t) w.n_embd * (size_t) (query_end - query_start);
    resume = 0;
    shared_prefix = 0;

    // Stored windows hold exact rows only (see the store below): one is
    // reusable while it sits in the shared prefix.
    const auto stored_window = [&](const Qwen35ScoringSession & session, int begin, int end, int shared) {
        std::vector<float> rows;
        return end <= shared &&
               session.query_windows.lookup(luce::pflash::pflash_span_tokens(begin, end), w.n_embd, rows);
    };
    Qwen35ScoringSession * best = nullptr;
    int best_shared = 0;
    for (auto & owned : st.sessions) {
        Qwen35ScoringSession * session = owned.get();
        if (limit == 0) break;
        if (!session || session->capacity < S_padded || session->ids.empty() ||
            session->keys_trained != st.head_loaded) {
            continue;
        }
        const size_t n = std::min(session->ids.size(), ids.size());
        const int shared = (int) (std::mismatch(session->ids.begin(),
            session->ids.begin() + (long) n, ids.begin()).first -
            session->ids.begin());
        int r = luce::pflash::pflash_scoring_resume(session->checkpoint, shared, S, resume_cap);
        // The query window and every other window this call scores that
        // starts before the resume point need their exact stored rows; with
        // one snapshot per session, the call otherwise starts at 0.
        if (r > query_start && !(query_end <= shared && row_floats > 0 &&
                                 stored_window(*session, query_start, query_end, shared))) {
            r = 0;
        }
        for (const auto & span : other_windows) {
            if (r == 0) break;
            if (span.begin < 0 || span.end <= span.begin || span.begin >= r) continue;
            if (!stored_window(*session, span.begin, span.end, shared)) r = 0;
        }
        if ((need_probe && (int) session->probe_raw.size() < r) ||
            (need_subunit && (int) session->subunit_raw.size() < r) ||
            (need_container && (int) session->container_raw.size() < r) ||
            (need_request && (int) session->request_raw.size() < r)) {
            r = 0;
        }
        if (r > resume) {
            resume = r;
            best = session;
            best_shared = shared;
        }
    }
    if (best) {
        if (!restore_ssm_state(best->cache, w.backend)) {
            resume = 0;
        } else {
            shared_prefix = best_shared;
            best->host_valid = false;   // this call extends it: its host copy (if any) goes stale
            return best;
        }
    }

    Qwen35ScoringSession * target = best;
    if (!target) {
        if (limit == 0) {
            scratch = std::make_unique<Qwen35ScoringSession>();
            target = scratch.get();
        } else {
            // A session that holds no prompt (never used, or its last call
            // failed or was redone) before evicting another conversation.
            for (auto & owned : st.sessions) {
                if (owned->ids.empty() && (!target || owned->last_used < target->last_used)) {
                    target = owned.get();
                }
            }
            if (!target && (int) st.sessions.size() < limit) {
                st.sessions.push_back(std::make_unique<Qwen35ScoringSession>());
                target = st.sessions.back().get();
            } else if (!target) {
                target = std::min_element(st.sessions.begin(), st.sessions.end(),
                    [] (const auto & a, const auto & b) {
                        return a->last_used < b->last_used;
                    })->get();
            }
        }
    }
    if (target->capacity < S_padded) {
        free_qwen35_scoring_session(*target);
        // Headroom so the next turns append without reallocating.
        const int capacity = limit == 0 ? S_padded
                                        : luce::pflash::pflash_session_capacity(S, st.session_ctx_limit);
        if (!allocate_scoring_session(w, capacity, *target)) {
            set_last_error("qwen35 scoring session allocation failed");
            return nullptr;
        }
    }
    zero_recurrent_state(target->cache);
    target->ids.clear();
    target->checkpoint = 0;
    target->probe_raw.clear();
    target->subunit_raw.clear();
    target->container_raw.clear();
    target->request_raw.clear();
    target->request_rows.clear();
    target->query_windows.clear();
    target->query_sets.clear();
    target->host_valid = false;
    return target;
}

} // namespace

std::vector<ggml_tensor *> qwen35_session_tensors(const Qwen35ScoringSession & session) {
    std::vector<ggml_tensor *> out = luce::pflash::pflash_owned_tensors(session.cache.base_ctx);
    for (ggml_tensor * t : luce::pflash::pflash_owned_tensors(session.cache.rollback_ctx)) out.push_back(t);
    for (ggml_tensor * t : luce::pflash::pflash_owned_tensors(session.key_ctx)) out.push_back(t);
    return out;
}

namespace {

// Token axis of each session tensor (qwen35_session_tensors order): attention K/V [head_dim, max_ctx, n_head_kv]
// along 1, block-15 keys [head_dim, n_head_kv, capacity] along 2, everything else (recurrent state, its snapshot)
// whole. A tensor whose shape does not show the expected token axis is copied whole.
std::vector<int> qwen35_session_token_dims(const Qwen35ScoringSession & session,
                                           const std::vector<ggml_tensor *> & tensors) {
    std::vector<int> dims(tensors.size(), -1);
    for (size_t i = 0; i < tensors.size(); ++i) {
        const ggml_tensor * t = tensors[i];
        const bool attn = std::find(session.cache.attn_k.begin(), session.cache.attn_k.end(), t) !=
                              session.cache.attn_k.end() ||
                          std::find(session.cache.attn_v.begin(), session.cache.attn_v.end(), t) !=
                              session.cache.attn_v.end();
        if (attn && t->ne[1] == session.cache.max_ctx) dims[i] = 1;
        if (t == session.keys && t->ne[2] == session.capacity) dims[i] = 2;
    }
    return dims;
}

void copy_session_bookkeeping(const Qwen35ScoringSession & from, Qwen35ScoringSession & to) {
    to.ids = from.ids;
    to.checkpoint = from.checkpoint;
    to.keys_trained = from.keys_trained;
    to.probe_raw = from.probe_raw;
    to.subunit_raw = from.subunit_raw;
    to.container_raw = from.container_raw;
    to.request_raw = from.request_raw;
    to.request_rows = from.request_rows;
    to.query_windows = from.query_windows;
    to.query_sets = from.query_sets;
    to.last_used = from.last_used;
}

size_t host_store_bytes(const DrafterHostSessions & store) {
    size_t total = 0;
    for (const auto & h : store.sessions) total += h.copy.bytes;
    return total;
}

void drop_host_store(DrafterContext & ctx, const std::string & reason) {
    if (!ctx.host_sessions) return;
    std::fprintf(stderr, "[qwen35-sessions] dropped %zu stored session(s) (%.1f MiB): %s; scoring fresh\n",
                 ctx.host_sessions->sessions.size(), host_store_bytes(*ctx.host_sessions) / (1024.0 * 1024.0),
                 reason.c_str());
    std::fflush(stderr);
    ctx.host_sessions.reset();
}

} // namespace

std::string qwen35_session_identity(const DrafterContext & ctx, const Qwen35DrafterState & st) {
    const std::string drafter_id = !st.drafter_file_id.empty()
        ? st.drafter_file_id
        : luce::pflash::pflash_file_identity(st.gguf_path);
    const TargetWeights & w = st.weights;
    std::ostringstream id;
    // The store lives in this process's memory, so the build cannot change under it; the stamp names it anyway.
    id << "format=2\n"
       << "build=" << __DATE__ << ' ' << __TIME__ << '\n'
       << "drafter=" << drafter_id << '\n'
       << "drafter_sha256=" << st.gguf_sha256 << '\n'
       << "gpu=" << ctx.gpu << '\n'
       << "dims=" << w.n_layer << ',' << w.n_embd << ',' << w.n_head << ',' << w.n_head_kv << ','
       << w.n_embd_head_k << '\n'
       << "head=" << st.head_file_id << '|' << st.head_loaded << '\n'
       << "segment_probe=" << st.probe_file_id << '|' << st.probe_loaded << '|'
       << (st.probe_sub_fc2_w != nullptr) << '|' << (st.probe_cont_fc2_w != nullptr) << '\n'
       << "request_probe=" << st.request_probe_file_id << '|' << st.request_loaded << '|'
       << st.request_checkpoint_sha256 << '\n'
       << "sessions=" << scoring_session_limit() << '\n'
       << "grid=" << luce::pflash::kScoringResumeGrid << ',' << luce::pflash::kScoringUbatch << ','
       << luce::pflash::kScoringCutBack << '\n';
    return id.str();
}

void qwen35_offload_sessions(DrafterContext & ctx) {
    auto * st = static_cast<Qwen35DrafterState *>(ctx.state);
    if (!st) return;
    const int limit = scoring_session_limit();
    if (limit == 0) {
        drop_host_store(ctx, "sessions are off for this drafter");
        return;
    }
    bool any = false;
    for (const auto & owned : st->sessions) any = any || (owned && !owned->ids.empty());
    if (!any && !ctx.host_sessions) return;
    const auto t0 = std::chrono::steady_clock::now();
    ggml_backend_synchronize(st->weights.backend);
    if (!ctx.host_sessions) ctx.host_sessions = std::make_shared<DrafterHostSessions>();
    DrafterHostSessions & store = *ctx.host_sessions;
    const std::string identity = qwen35_session_identity(ctx, *st);
    if (!store.sessions.empty() && store.identity != identity) {
        drop_host_store(ctx, "identity mismatch at offload: " +
                        luce::pflash::pflash_identity_mismatch(store.identity, identity));
        ctx.host_sessions = std::make_shared<DrafterHostSessions>();
    }
    DrafterHostSessions & s = *ctx.host_sessions;
    s.identity = identity;
    s.clock = std::max(s.clock, st->session_clock);

    std::vector<luce::pflash::PFlashOffloadEntry> entries;
    for (const auto & h : s.sessions) entries.push_back({h.id, h.bookkeeping.last_used, h.on_device});
    std::vector<luce::pflash::PFlashOffloadDevice> devices;
    for (const auto & owned : st->sessions) {
        devices.push_back({owned->host_id, owned->host_valid, !owned->ids.empty() && owned->checkpoint > 0,
                           owned->last_used});
    }
    const auto plan = luce::pflash::pflash_plan_offload(entries, devices, limit);

    std::vector<luce::pflash::PFlashHostBlob> pool;
    std::vector<Qwen35HostSession> kept;
    for (auto & h : s.sessions) {
        if (std::find(plan.keep_entries.begin(), plan.keep_entries.end(), h.id) != plan.keep_entries.end()) {
            h.on_device = false;
            kept.push_back(std::move(h));
        } else {
            pool.push_back(std::move(h.copy.blob));
        }
    }
    const size_t n_dropped = s.sessions.size() - kept.size();
    s.sessions = std::move(kept);
    const size_t n_kept = s.sessions.size();
    size_t copied = 0, copied_bytes = 0;
    for (size_t index : plan.copy_devices) {
        Qwen35ScoringSession & session = *st->sessions[index];
        Qwen35HostSession host;
        // A copy covers what a resume reads: the token-indexed tensors below the checkpoint, the rest whole.
        const auto tensors = qwen35_session_tensors(session);
        size_t need = 0;
        {
            std::vector<std::pair<size_t, size_t>> chunks;
            std::string ignore;
            const auto dims = qwen35_session_token_dims(session, tensors);
            for (size_t i = 0; i < tensors.size(); ++i) {
                if (luce::pflash::pflash_token_chunks(tensors[i], dims[i], session.checkpoint, chunks, ignore)) {
                    for (const auto & c : chunks) need += c.second;
                }
            }
        }
        for (auto & blob : pool) {
            if (blob.capacity() >= need) {
                host.copy.blob = std::move(blob);
                break;
            }
        }
        std::string error;
        if (!luce::pflash::pflash_copy_to_host(tensors, qwen35_session_token_dims(session, tensors),
                                               session.checkpoint, /*pinned=*/nullptr, host.copy, error)) {
            std::fprintf(stderr, "[qwen35-sessions] session not offloaded (%zu tokens): %s\n",
                         session.ids.size(), error.c_str());
            continue;
        }
        host.id = ++s.next_id;
        host.cache_max_ctx = session.cache.max_ctx;
        host.cache_cur_pos = session.cache.cur_pos;
        host.cache_last_tok = session.cache.last_tok;
        host.bookkeeping.capacity = session.capacity;
        copy_session_bookkeeping(session, host.bookkeeping);
        copied_bytes += host.copy.bytes;
        ++copied;
        s.sessions.push_back(std::move(host));
    }
    pool.clear();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "[qwen35-sessions] offload: copied %zu session(s) (%.1f MiB), kept %zu unchanged host "
                 "cop%s, released %zu; host RAM held %.1f MiB in %zu session(s) (%.0f ms)\n",
                 copied, copied_bytes / (1024.0 * 1024.0), n_kept,
                 n_kept == 1 ? "y" : "ies", n_dropped, host_store_bytes(s) / (1024.0 * 1024.0), s.sessions.size(), ms);
    std::fflush(stderr);
    if (s.sessions.empty()) ctx.host_sessions.reset();
}

void qwen35_restore_sessions(DrafterContext & ctx, Qwen35DrafterState & st, const std::vector<int32_t> & ids) {
    if (!ctx.host_sessions || ctx.host_sessions->sessions.empty()) return;
    const int limit = scoring_session_limit();
    if (limit == 0) {
        drop_host_store(ctx, "sessions are off for this drafter");
        return;
    }
    const std::string mismatch = luce::pflash::pflash_identity_mismatch(ctx.host_sessions->identity,
                                                                         qwen35_session_identity(ctx, st));
    if (!mismatch.empty()) {
        drop_host_store(ctx, "identity mismatch: " + mismatch);
        return;
    }
    DrafterHostSessions & store = *ctx.host_sessions;
    st.session_clock = std::max(st.session_clock, store.clock);
    // Candidates: the device sessions first (no copy needed), then the stored ones not on the device yet.
    std::vector<luce::pflash::PFlashStoredPrompt> prompts;
    for (const auto & owned : st.sessions) prompts.push_back({&owned->ids, owned->checkpoint});
    std::vector<size_t> stored_index;
    for (size_t i = 0; i < store.sessions.size(); ++i) {
        if (store.sessions[i].on_device) continue;
        prompts.push_back({&store.sessions[i].bookkeeping.ids, store.sessions[i].bookkeeping.checkpoint});
        stored_index.push_back(i);
    }
    const int pick = luce::pflash::pflash_pick_restore(prompts, ids);
    if (pick < (int) st.sessions.size()) return;   // nothing to resume from, or a device session already covers it
    Qwen35HostSession & host = store.sessions[stored_index[(size_t) pick - st.sessions.size()]];
    const size_t n_ids = host.bookkeeping.ids.size();
    const uint64_t host_id = host.id;
    const auto drop = [&](const std::string & reason) {
        std::fprintf(stderr, "[qwen35-sessions] dropped stored session (%zu tokens): %s; scoring fresh\n",
                     n_ids, reason.c_str());
        std::fflush(stderr);
        store.sessions.erase(std::remove_if(store.sessions.begin(), store.sessions.end(),
            [&](const Qwen35HostSession & h) { return h.id == host_id; }), store.sessions.end());
    };
    if ((int) st.sessions.size() >= limit) {
        std::fprintf(stderr, "[qwen35-sessions] stored session (%zu tokens) not restored: device sessions full\n",
                     n_ids);
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    // A request longer than the stored capacity gets the headroom acquire would give it, as a resident session that
    // grew would have; the stored prefix goes into the longer tensors unchanged.
    const int S = (int) ids.size();
    int capacity = host.bookkeeping.capacity;
    if (capacity < align_up_i(S, 256)) capacity = luce::pflash::pflash_session_capacity(S, st.session_ctx_limit);
    auto session = std::make_unique<Qwen35ScoringSession>();
    if (host.bookkeeping.capacity <= 0 ||
        !allocate_scoring_session(st.weights, capacity, *session)) {
        free_qwen35_scoring_session(*session);
        drop(std::string("allocation failed: ") + luce_last_error());
        return;
    }
    std::string error;
    if (session->cache.max_ctx < host.cache_max_ctx) {
        error = "cache size differs";
    } else if (!luce::pflash::pflash_copy_from_host(host.copy, qwen35_session_tensors(*session), error)) {
        // error set
    }
    if (!error.empty()) {
        free_qwen35_scoring_session(*session);
        drop(error);
        return;
    }
    ggml_backend_synchronize(st.weights.backend);
    session->cache.cur_pos = host.cache_cur_pos;
    session->cache.last_tok = host.cache_last_tok;
    copy_session_bookkeeping(host.bookkeeping, *session);
    session->host_id = host.id;
    session->host_valid = true;
    host.on_device = true;
    st.sessions.push_back(std::move(session));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "[qwen35-sessions] restored 1 of %zu stored session(s) (%zu tokens, checkpoint %d): "
                 "%.1f MiB host to device (%.0f ms)\n", store.sessions.size(), n_ids, host.bookkeeping.checkpoint,
                 host.copy.bytes / (1024.0 * 1024.0), ms);
    std::fflush(stderr);
}


// Scoring-head selection for the Qwen3.5-0.8B drafter: run blocks 0..14, then
// score every context token against the query window with block 15's NoPE
// Q/K (or a trained replacement) and select chunks by attention mass. This
// is the runtime counterpart of the Python retention screen (trial 0075).
//
// ``resume_cap`` bounds the session resume point (0: start over); ``redone``
// names why this call is a redo (empty on a first call). A query-step call
// that needs a query row a resumed session cannot give exactly returns empty
// with ``redo`` set to the missing input, its session forgotten; the caller
// then scores the prompt again from token 0.
static std::vector<int32_t> strict_score_impl(
    Qwen35DrafterState & st,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int n_lookahead,
    int score_query_end,
    const luce::pflash::PFlashSelectionConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans,
    int resume_cap,
    const std::string & redone,
    std::string & redo) {

    redo.clear();
    TargetWeights & w = st.weights;
    const int S = (int)ids.size();
    const int hidden = w.n_embd;
    const int H = w.n_head;
    const int Hk = w.n_head_kv;
    const int D = w.n_embd_head_k;
    std::string block_error;
    if (!qwen35_head_block_available(w, block_error)) {
        set_last_error(block_error);
        return {};
    }
    if (n_lookahead < 1 || S < n_lookahead + 1) {
        set_last_error("qwen35 scoring head input is too short");
        return {};
    }
    const int query_end = score_query_end < 0 ? S : score_query_end;
    if (query_end < n_lookahead || query_end > S) {
        set_last_error("qwen35 scoring head query window out of range");
        return {};
    }
    const int query_start = query_end - n_lookahead;
    const TargetLayer & L = w.layers[(size_t)kQwen35HeadBlock];
    const bool use_probe = st.probe_loaded &&
        experiment.segmentation != luce::pflash::PFlashSegmentation::Fixed;
    const bool request_probe = experiment.query_probe && st.request_loaded;
    const bool request_v5 = request_probe && st.request_v5;
    // Relative query candidates + verbatim pin (pflash_query_candidates.h): active
    // whenever both the request probe and segment probe are loaded.
    const luce::pflash::PFlashQueryConfig & qcfg = experiment.query;
    const bool qcand = request_probe && use_probe && qcfg.candidates > 0;
    const bool qpin = request_probe && use_probe && qcfg.pin_tokens > 0;
    const bool qstep = qcand || qpin;
    if (qstep && (!experiment.request_messages_all_error.empty() || experiment.request_messages_all.empty())) {
        set_last_error("query step: no message mapping from the caller" +
                       (experiment.request_messages_all_error.empty()
                            ? std::string() : ": " + experiment.request_messages_all_error));
        return {};
    }

    auto t0 = std::chrono::steady_clock::now();
    int resume = 0;
    int shared_prefix = 0;
    std::unique_ptr<Qwen35ScoringSession> scratch;
    std::vector<PFlashTokenSpan> other_windows;
    if (experiment.turn_query.begin >= 0 && experiment.turn_query.end <= query_start &&
        experiment.turn_query.end > experiment.turn_query.begin) {
        other_windows.push_back(experiment.turn_query);
    }
    for (const auto & span : experiment.history_queries) {
        if (span.end <= query_start && span.end > span.begin) other_windows.push_back(span);
    }
    // The query step resumes like every other request (pflash_scoring_resume.h: exact by construction); its query
    // rows come from this call's activations or the session's exact F32 sets, else the request is redone from 0.
    const bool sessions_on = scoring_session_limit() > 0;
    Qwen35ScoringSession * session = acquire_scoring_session(
        st, ids, query_start, query_end, use_probe,
        use_probe && st.probe_sub_fc2_w != nullptr,
        use_probe && st.probe_cont_fc2_w != nullptr, request_probe, other_windows,
        resume_cap, resume, shared_prefix, scratch);
    if (!session) return {};
    if (resume > 0 && (resume % luce::pflash::kScoringResumeGrid != 0 ||
                       resume > S - luce::pflash::kScoringCutBack)) {
        session->ids.clear();
        set_last_error("qwen35 scorer: resume point off the exact grid");
        return {};
    }
    // Retained request rows past the shared prefix belong to a prompt that
    // diverged there (an edit, a truncation, another conversation).
    const int request_rows_invalidated = request_probe
        ? session->request_rows.invalidate_from(shared_prefix) : 0;
    session->query_sets.invalidate_from(shared_prefix);
    session->query_windows.invalidate_from(shared_prefix);
    // A session is released (freed or kept) on every exit below.
    struct SessionExit {
        std::unique_ptr<Qwen35ScoringSession> & scratch;
        ~SessionExit() {
            if (scratch) free_qwen35_scoring_session(*scratch);
        }
    } session_exit{scratch};
    TargetCache & cache = session->cache;
    const int n_new = S - resume;
    // Ubatches are cut once at S - 64, exactly as in a fresh call, and the
    // recurrent state is snapshotted at the last 8192-grid point at or before
    // that cut: the next turn (which replaces this prompt's generation prompt,
    // or repeats it) resumes there and recomputes the rest on the fresh
    // call's own partition.
    const int cut = std::max(resume, S - luce::pflash::kScoringCutBack);
    const int checkpoint = luce::pflash::pflash_scoring_checkpoint(S);
    // Block-14 rows below this are exact for later prompts (pflash_scoring_resume.h).
    const int exact_rows_end = luce::pflash::pflash_scoring_exact_rows_end(S);

    ggml_init_params act_ip{};
    act_ip.mem_size = (size_t)8 * ggml_tensor_overhead() + 4096;
    act_ip.no_alloc = true;
    ggml_context * act_ctx = ggml_init(act_ip);
    if (!act_ctx) {
        session->ids.clear();
        set_last_error("qwen35 drafter activation ctx init failed");
        return {};
    }
    ggml_tensor * act_in = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, std::max(1, n_new));
    ggml_backend_buffer_t act_buf = ggml_backend_alloc_ctx_tensors(act_ctx, w.backend);
    if (!act_buf) {
        ggml_free(act_ctx);
        session->ids.clear();
        set_last_error("qwen35 drafter activation allocation failed");
        return {};
    }
    // Any failure below leaves the session's state half-written: forget its
    // prompt so the next call starts it over.
    auto cleanup = [&]() {
        ggml_backend_buffer_free(act_buf);
        ggml_free(act_ctx);
    };
    auto fail = [&](const char * message) -> std::vector<int32_t> {
        cleanup();
        session->ids.clear();
        set_last_error(message);
        return {};
    };

    // Blocks 0..14 over the new tokens only, fused per ubatch: one graph runs
    // all 15 blocks for [start, start + n), sharing one causal mask and M-RoPE
    // position tensor across the full-attention blocks, keeping intermediate
    // [hidden, n] activations in transient scratch, and writing only the final
    // block-14 output into act_in. Each DeltaNet layer's recurrent state is
    // copied once the ubatch loop reaches the checkpoint.
    const auto snapshot_all_delta = [&]() {
        int dn = 0;
        for (int il = 0; il < kQwen35HeadBlock; ++il) {
            if (((il + 1) % w.full_attention_interval) == 0) continue;
            if (dn < (int) cache.ssm_state.size() && cache.ssm_state[(size_t) dn] &&
                cache.ssm_state_snap[(size_t) dn]) {
                ggml_backend_tensor_copy(cache.ssm_state[(size_t) dn],
                                         cache.ssm_state_snap[(size_t) dn]);
                ggml_backend_tensor_copy(cache.conv_state[(size_t) dn],
                                         cache.conv_state_snap[(size_t) dn]);
            }
            ++dn;
        }
    };
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
    const int ubatch = luce::pflash::kScoringUbatch;
    // Ubatches on the 1024 grid from the (grid) resume point, cut at S - 64;
    // the checkpoint is a grid point, so the state there can be copied.
    const auto ubatch_at = [&](int start) {
        return std::min(ubatch, (start < cut ? cut : S) - start);
    };
    const bool has_attn = w.full_attention_interval <= kQwen35HeadBlock;
    // One host buffer for every ubatch's graph metadata, instead of a fresh
    // allocation per ubatch.
    std::vector<uint8_t> graph_mem(16 * 1024 * 1024);
    // Graph of blocks 0..14 over tokens [start, start + n), reading embedded
    // inputs from `inp` and writing block-14 outputs into act_in.
    // nullptr (error set) on failure.
    const auto ubatch_graph = [&](int start, int n, ggml_cgraph *& gf,
                                  ggml_tensor *& inp, ggml_tensor *& pos) -> ggml_context * {
        ggml_init_params ip{};
        ip.mem_size = graph_mem.size();
        ip.mem_buffer = graph_mem.data();
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            set_last_error("qwen35 drafter ubatch graph ctx init failed");
            return nullptr;
        }
        gf = ggml_new_graph_custom(ctx, 16384, false);
        inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n);
        ggml_set_input(inp);
        pos = nullptr;
        ggml_tensor * mask = nullptr;
        if (has_attn) {
            pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n);
            ggml_set_input(pos);
            mask = build_qwen35_causal_mask(ctx, gf, start, n);
        }
        ggml_tensor * cur = inp;
        for (int il = 0; il < kQwen35HeadBlock; ++il) {
            const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);
            cur = build_qwen35_layer(ctx, gf, w, cache, il, cur,
                                     is_attn ? pos : nullptr,
                                     is_attn ? mask : nullptr,
                                     start, n, false, 0);
        }
        ggml_tensor * dst = ggml_view_2d(ctx, act_in, hidden, n, act_in->nb[1],
                                         (size_t)(start - resume) * act_in->nb[1]);
        if (ggml_nelements(cur) != ggml_nelements(dst)) {
            ggml_free(ctx);
            set_last_error("qwen35 layer output shape mismatch");
            return nullptr;
        }
        ggml_build_forward_expand(gf, ggml_cpy(ctx, cur, dst));
        return ctx;
    };
    const auto fail_graph = [&]() -> std::vector<int32_t> {
        ggml_gallocr_free(alloc);
        cleanup();
        session->ids.clear();
        return {};
    };
    // Size the compute buffer once, for the ubatch with the largest causal
    // mask: allocating per ubatch would otherwise regrow it as kv_len grows.
    if (n_new > 0) {
        int worst = resume;
        for (int start = resume; start < S; start += ubatch_at(start)) {
            if (causal_mask_bytes(start, ubatch_at(start)) >=
                causal_mask_bytes(worst, ubatch_at(worst))) worst = start;
        }
        ggml_cgraph * gf = nullptr;
        ggml_tensor * inp = nullptr;
        ggml_tensor * pos = nullptr;
        ggml_context * ctx = ubatch_graph(worst, ubatch_at(worst), gf, inp, pos);
        if (!ctx) return fail_graph();
        const bool reserved = ggml_gallocr_reserve(alloc, gf);
        ggml_free(ctx);
        if (!reserved) {
            ggml_gallocr_free(alloc);
            return fail("qwen35 drafter graph allocation failed");
        }
    }
    if (checkpoint == resume) snapshot_all_delta();
    std::vector<float> emb((size_t)hidden * (size_t)ubatch);
    std::vector<int32_t> p4((size_t)4 * (size_t)ubatch, 0);
    for (int start = resume; start < S;) {
        const int n = ubatch_at(start);
        if (!w.embedder.embed(ids.data() + start, n, emb.data())) {
            ggml_gallocr_free(alloc);
            return fail("qwen35 drafter embedding failed");
        }
        ggml_cgraph * gf = nullptr;
        ggml_tensor * inp = nullptr;
        ggml_tensor * pos = nullptr;
        ggml_context * ctx = ubatch_graph(start, n, gf, inp, pos);
        if (!ctx) return fail_graph();
        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            ggml_free(ctx); ggml_gallocr_free(alloc);
            return fail("qwen35 drafter graph allocation failed");
        }
        ggml_backend_tensor_set(inp, emb.data(), 0, (size_t)hidden * (size_t)n * sizeof(float));
        if (pos) {
            std::fill_n(p4.begin(), (size_t)4 * (size_t)n, 0);
            for (int i = 0; i < n; ++i) {
                const int p = start + i;
                p4[(size_t)0 * n + i] = p;
                p4[(size_t)1 * n + i] = p;
                p4[(size_t)2 * n + i] = p;
            }
            ggml_backend_tensor_set(pos, p4.data(), 0, (size_t)4 * (size_t)n * sizeof(int32_t));
        }
        const auto status = ggml_backend_graph_compute(w.backend, gf);
        ggml_free(ctx);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_gallocr_free(alloc);
            return fail("qwen35 drafter graph compute failed");
        }
        start += n;
        if (start == checkpoint && checkpoint > resume) {
            snapshot_all_delta();
        }
    }
    ggml_gallocr_free(alloc);
    auto t1 = std::chrono::steady_clock::now();

    // Block-14 rows of each query window -- the query, then earlier
    // questions -- from this call when it computed them, else from the
    // session while they sit in the shared prefix. The query's are always
    // available (the session was chosen for them); a history window whose
    // rows are gone is skipped.
    struct ScoredWindow {
        int begin = 0;
        int end = 0;
        double weight = 1.0;
        std::vector<float> rows;
    };
    std::vector<ScoredWindow> windows;
    const auto rows_for = [&](int begin, int end, std::vector<float> & rows) {
        rows.assign((size_t)hidden * (size_t)(end - begin), 0.0f);
        if (begin >= resume) {
            ggml_backend_tensor_get(act_in, rows.data(),
                                    (size_t)(begin - resume) * act_in->nb[1],
                                    rows.size() * sizeof(float));
            return true;
        }
        return end <= shared_prefix &&
               session->query_windows.lookup(luce::pflash::pflash_span_tokens(begin, end), hidden, rows);
    };
    {
        ScoredWindow query;
        query.begin = query_start;
        query.end = query_end;
        if (!rows_for(query_start, query_end, query.rows)) {
            return fail("qwen35 scorer query rows unavailable");
        }
        windows.push_back(std::move(query));
        const auto & turn = experiment.turn_query;
        if (turn.begin >= 0 && turn.end <= query_start && turn.end > turn.begin) {
            ScoredWindow tail;
            tail.begin = turn.begin;
            tail.end = turn.end;
            if (rows_for(turn.begin, turn.end, tail.rows)) {
                windows.push_back(std::move(tail));
            }
        }
        double weight = 1.0;
        for (const auto & span : experiment.history_queries) {
            weight *= 0.5;
            if (span.end > query_start || span.end - span.begin < 1) continue;
            ScoredWindow history;
            history.begin = span.begin;
            history.end = span.end;
            history.weight = weight;
            if (rows_for(span.begin, span.end, history.rows)) {
                windows.push_back(std::move(history));
            }
        }
    }
    // Rows a later request scores as history queries: kept with the session,
    // never scored here.
    std::vector<ScoredWindow> keep_rows;
    for (const auto & span : experiment.keep_queries) {
        if (span.end > query_start || span.end - span.begin < 1) continue;
        ScoredWindow keep;
        keep.begin = span.begin;
        keep.end = span.end;
        if (rows_for(span.begin, span.end, keep.rows)) keep_rows.push_back(std::move(keep));
    }

    ggml_tensor * wk_src = st.head_loaded ? st.head_wk : L.wk;
    session->keys_trained = st.head_loaded;
    session->probe_raw.resize(use_probe ? (size_t) resume : 0);
    session->subunit_raw.resize(
        use_probe && st.probe_sub_fc2_w ? (size_t) resume : 0);
    session->container_raw.resize(
        use_probe && st.probe_cont_fc2_w ? (size_t) resume : 0);
    session->request_raw.resize(request_probe ? (size_t) resume : 0);

    // Keys (and probe logits) of the new tokens, into the session. Keys are
    // projected in chunks so no intermediate tensor puts the sequence length
    // into a HIP grid y/z dimension (65,535 limit). The chunks start at the
    // (grid) resume point, so they fall where a fresh call's do.
    const int key_chunk = luce::pflash::kScoringResumeGrid;
    if (n_new > 0) {
        ggml_init_params nip{};
        nip.mem_size = (size_t)6 * ggml_tensor_overhead() + 4096;
        nip.no_alloc = true;
        ggml_context * nctx = ggml_init(nip);
        ggml_tensor * probe_new = use_probe ? ggml_new_tensor_1d(nctx, GGML_TYPE_F32, n_new) : nullptr;
        ggml_tensor * subunit_new = use_probe && st.probe_sub_fc2_w
            ? ggml_new_tensor_1d(nctx, GGML_TYPE_F32, n_new) : nullptr;
        ggml_tensor * container_new = use_probe && st.probe_cont_fc2_w
            ? ggml_new_tensor_1d(nctx, GGML_TYPE_F32, n_new) : nullptr;
        ggml_tensor * request_new = request_probe
            ? ggml_new_tensor_1d(nctx, GGML_TYPE_F32, n_new) : nullptr;
        ggml_backend_buffer_t nbuf = use_probe || request_probe
            ? ggml_backend_alloc_ctx_tensors(nctx, w.backend) : nullptr;
        if ((use_probe || request_probe) && !nbuf) {
            ggml_free(nctx);
            return fail("qwen35 probe buffer allocation failed");
        }
        const int n_key_chunks = (n_new + key_chunk - 1) / key_chunk;
        ggml_init_params kip{};
        kip.mem_size = ggml_tensor_overhead() *
                           (size_t)(64 + (request_probe ? 48 : 32) * n_key_chunks) +
                       ggml_graph_overhead_custom(4096, false) + 64 * 1024;
        kip.no_alloc = true;
        ggml_context * kctx = ggml_init(kip);
        ggml_cgraph * kgf = ggml_new_graph_custom(kctx, 4096, false);
        for (int b = 0; b < n_new; b += key_chunk) {
            const int n = std::min(key_chunk, n_new - b);
            ggml_tensor * x_c = ggml_view_2d(kctx, act_in, hidden, n, act_in->nb[1],
                                             (size_t)b * act_in->nb[1]);
            ggml_tensor * x_norm = ggml_mul(kctx, ggml_rms_norm(kctx, x_c, w.rms_eps), L.attn_norm);
            ggml_tensor * K = ggml_reshape_3d(kctx, ggml_mul_mat(kctx, wk_src, x_norm), D, Hk, n);
            K = ggml_mul(kctx, ggml_rms_norm(kctx, K, w.rms_eps), L.k_norm);
            ggml_tensor * k_dst = ggml_view_3d(kctx, session->keys, D, Hk, n,
                                               session->keys->nb[1], session->keys->nb[2],
                                               (size_t)(resume + b) * session->keys->nb[2]);
            ggml_build_forward_expand(kgf, ggml_cpy(kctx, K, k_dst));
            if (use_probe) {
                // Segment probe on the same tap: LayerNorm -> fc1 -> GELU trunk,
                // then one fc2 row per head (unit always; sub-unit when shipped).
                ggml_tensor * p = ggml_norm(kctx, x_c, st.probe_norm_eps);
                p = ggml_add(kctx, ggml_mul(kctx, p, st.probe_norm_w), st.probe_norm_b);
                p = ggml_gelu(kctx, ggml_add(kctx, ggml_mul_mat(kctx, st.probe_fc1_w, p),
                                             st.probe_fc1_b));                      // [width, n]
                ggml_tensor * unit = ggml_add(kctx, ggml_mul_mat(kctx, st.probe_fc2_w, p),
                                              st.probe_fc2_b);                      // [1, n]
                ggml_tensor * p_dst = ggml_view_1d(kctx, probe_new, n,
                                                   (size_t)b * ggml_element_size(probe_new));
                ggml_build_forward_expand(kgf, ggml_cpy(kctx, ggml_reshape_1d(kctx, unit, n), p_dst));
                if (subunit_new) {
                    ggml_tensor * sub = ggml_add(kctx,
                        ggml_mul_mat(kctx, st.probe_sub_fc2_w, p), st.probe_sub_fc2_b);
                    ggml_tensor * s_dst = ggml_view_1d(kctx, subunit_new, n,
                        (size_t)b * ggml_element_size(subunit_new));
                    ggml_build_forward_expand(kgf,
                        ggml_cpy(kctx, ggml_reshape_1d(kctx, sub, n), s_dst));
                }
                if (container_new) {
                    ggml_tensor * cont = ggml_add(kctx,
                        ggml_mul_mat(kctx, st.probe_cont_fc2_w, p), st.probe_cont_fc2_b);
                    ggml_tensor * c_dst = ggml_view_1d(kctx, container_new, n,
                        (size_t)b * ggml_element_size(container_new));
                    ggml_build_forward_expand(kgf,
                        ggml_cpy(kctx, ggml_reshape_1d(kctx, cont, n), c_dst));
                }
            }
            if (request_new) {
                // Request probe: its own LayerNorm -> fc1 -> GELU(erf) -> fc2
                // trunk on the same tap, one raw logit per token.
                ggml_tensor * r = ggml_norm(kctx, x_c, st.request_norm_eps);
                r = ggml_add(kctx, ggml_mul(kctx, r, st.request_norm_w), st.request_norm_b);
                r = ggml_gelu_erf(kctx, ggml_add(kctx, ggml_mul_mat(kctx, st.request_fc1_w, r),
                                                 st.request_fc1_b));                // [width, n]
                ggml_tensor * req = ggml_add(kctx, ggml_mul_mat(kctx, st.request_fc2_w, r),
                                             st.request_fc2_b);                     // [1, n]
                ggml_tensor * r_dst = ggml_view_1d(kctx, request_new, n,
                    (size_t)b * ggml_element_size(request_new));
                ggml_build_forward_expand(kgf,
                    ggml_cpy(kctx, ggml_reshape_1d(kctx, req, n), r_dst));
            }
        }
        ggml_gallocr_t kalloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
        const bool key_ok = ggml_gallocr_alloc_graph(kalloc, kgf) &&
            ggml_backend_graph_compute(w.backend, kgf) == GGML_STATUS_SUCCESS;
        ggml_gallocr_free(kalloc);
        ggml_free(kctx);
        if (key_ok && use_probe) {
            session->probe_raw.resize((size_t) S);
            ggml_backend_tensor_get(probe_new, session->probe_raw.data() + resume, 0,
                                    (size_t) n_new * sizeof(float));
            if (subunit_new) {
                session->subunit_raw.resize((size_t) S);
                ggml_backend_tensor_get(subunit_new, session->subunit_raw.data() + resume, 0,
                                        (size_t) n_new * sizeof(float));
            }
            if (container_new) {
                session->container_raw.resize((size_t) S);
                ggml_backend_tensor_get(container_new, session->container_raw.data() + resume, 0,
                                        (size_t) n_new * sizeof(float));
            }
        }
        if (key_ok && request_new) {
            session->request_raw.resize((size_t) S);
            ggml_backend_tensor_get(request_new, session->request_raw.data() + resume, 0,
                                    (size_t) n_new * sizeof(float));
        }
        if (nbuf) ggml_backend_buffer_free(nbuf);
        ggml_free(nctx);
        if (!key_ok) return fail("qwen35 key graph compute failed");
    }

    const std::vector<float> & probe_raw = session->probe_raw;
    const std::vector<float> & subunit_raw = session->subunit_raw;
    const std::vector<float> & container_raw = session->container_raw;
    // Tap-count smoothing over raw segment-probe logits (torch Conv1d,
    // symmetric padding) plus residual logit, then sigmoid.
    const auto smooth_probe = [&](const std::vector<float> & raw,
                                  const std::vector<float> & conv_w, float conv_b,
                                  std::vector<float> & out) {
        const int taps = (int) conv_w.size();
        const int radius = taps / 2;
        out.assign((size_t) S, 0.0f);
        for (int t = 0; t < S; ++t) {
            float acc = raw[(size_t) t] + conv_b;
            for (int k = 0; k < taps; ++k) {
                const int u = t + k - radius;
                if (u >= 0 && u < S) acc += conv_w[(size_t) k] * raw[(size_t) u];
            }
            out[(size_t) t] = 1.0f / (1.0f + std::exp(-acc));
        }
    };
    std::vector<float> container_probs;
    if (use_probe && !container_raw.empty() && !st.probe_cont_conv_w.empty()) {
        smooth_probe(container_raw, st.probe_cont_conv_w, st.probe_cont_conv_b, container_probs);
    }

    // Precompute smoothed segment-probe probabilities and line-alignment flags
    // once per forward: segment() is called up to 3x per request (unit_segments,
    // fit_pin, and final selection) with different forced pin spans.
    std::vector<float> probe_boundary;
    std::vector<float> probe_split_scores;
    luce::pflash::PFlashLineFlags probe_lines;
    bool have_probe_lines = false;
    int probe_boundaries_in_context = 0;
    const int seg_query_end = score_query_end < 0 ? S : score_query_end;
    const int seg_query_begin = seg_query_end - std::min(n_lookahead, seg_query_end);
    if (use_probe) {
        smooth_probe(probe_raw, st.probe_conv_w, st.probe_conv_b, probe_boundary);
        if (!subunit_raw.empty()) {
            smooth_probe(subunit_raw, st.probe_sub_conv_w, st.probe_sub_conv_b, probe_split_scores);
        }
        for (int t = 1; t < S; ++t) {
            if (t >= seg_query_begin &&
                (t < seg_query_end || !experiment.query_suffix_candidates)) {
                continue;
            }
            if (probe_boundary[(size_t) t] > st.probe_threshold) ++probe_boundaries_in_context;
        }
        const bool forced_probe =
            experiment.segmentation == luce::pflash::PFlashSegmentation::Probe;
        if ((probe_boundaries_in_context >= 4 || forced_probe) &&
            st.probe_line_aligned && st.probe_vocab) {
            std::vector<std::string> texts((size_t) S);
            for (int t = 0; t < S; ++t) texts[(size_t) t] = st.probe_vocab->token_text(ids[(size_t) t]);
            probe_lines = luce::pflash::pflash_line_flags(texts);
            have_probe_lines = true;
        }
    }

    // Candidates: probe segments cut at the pinned spans' edges, or fixed
    // chunks when the probe is off or finds too few boundaries.
    // Memoized by `pins` so query-unit discovery, fit_pin, and final selection
    // reuse identical segmentations instead of recomputing and re-logging.
    struct CachedSeg {
        std::vector<PFlashTokenSpan> segments;
        bool density = false;
    };
    std::map<std::vector<std::pair<int, int>>, CachedSeg> segment_cache;
    const auto segment = [&](const std::vector<PFlashTokenSpan> & pins,
                             std::vector<PFlashTokenSpan> & segments, bool & density) {
        segments.clear();
        density = experiment.candidate_score == luce::pflash::PFlashCandidateScore::Density;
        if (!use_probe) return;
        std::vector<std::pair<int, int>> key;
        key.reserve(pins.size());
        for (const auto & span : pins) key.emplace_back(span.begin, span.end);
        auto it = segment_cache.find(key);
        if (it != segment_cache.end()) {
            segments = it->second.segments;
            density = it->second.density;
            return;
        }
        std::vector<int> forced{seg_query_begin, seg_query_end};
        for (const auto & span : pins) {
            forced.push_back(span.begin);
            forced.push_back(span.end);
        }
        const bool forced_probe =
            experiment.segmentation == luce::pflash::PFlashSegmentation::Probe;
        if (probe_boundaries_in_context >= 4 || forced_probe) {
            segments = luce::pflash::pflash_probe_segments(
                probe_boundary, S, st.probe_threshold, st.probe_min_segment,
                st.probe_max_segment, forced, probe_split_scores,
                have_probe_lines ? &probe_lines : nullptr);
        }
        if (!segments.empty() &&
            experiment.candidate_score == luce::pflash::PFlashCandidateScore::Auto) {
            density = true;
        }
        segment_cache.emplace(std::move(key), CachedSeg{segments, density});
    };
    const auto log_segment = [&](const std::vector<PFlashTokenSpan> & segments, bool density) {
        if (!use_probe) return;
        if (segments.empty()) {
            std::fprintf(stderr,
                "[qwen35-segment-probe] %d boundaries in the context, "
                "falling back to fixed %d-token chunks\n",
                probe_boundaries_in_context, experiment.chunk_size);
        } else {
            std::fprintf(stderr,
                "[qwen35-segment-probe] %d boundaries in the context -> %zu segments "
                "(threshold %.2f, %d-%d tokens), score=%s\n",
                probe_boundaries_in_context, segments.size(), st.probe_threshold,
                st.probe_min_segment, st.probe_max_segment,
                density ? "density" : "sum");
        }
        std::fflush(stderr);
    };

    // Tokens the selector keeps whatever their score, counted the way it charges them
    // (pflash_chunk_is_structurally_required over the candidates).
    const int sel_query_end = score_query_end < 0 ? S : score_query_end;
    const int sel_query_begin = sel_query_end - std::min(n_lookahead, sel_query_end);
    const auto charged = [&](const std::vector<PFlashTokenSpan> & candidates_pins,
                             const std::vector<PFlashTokenSpan> & candidate_segments) {
        int total = 0;
        const int n_chunks = candidate_segments.empty()
            ? (S + experiment.chunk_size - 1) / experiment.chunk_size
            : (int) candidate_segments.size();
        for (int c = 0; c < n_chunks; ++c) {
            const int begin = candidate_segments.empty()
                ? c * experiment.chunk_size : candidate_segments[(size_t) c].begin;
            const int end = candidate_segments.empty()
                ? std::min(S, begin + experiment.chunk_size)
                : candidate_segments[(size_t) c].end;
            if (luce::pflash::pflash_chunk_is_structurally_required(
                    begin, end, sel_query_begin, sel_query_end, S, candidates_pins,
                    !experiment.query_suffix_candidates)) {
                total += end - begin;
            }
        }
        return total;
    };

    // Request probe: the query set from the request logits of the human
    // messages, its block-14 rows, and the rows later turns will need -- all
    // read here, while this call's activations (act_in) still exist.
    luce::pflash::PFlashRequestProbeTrace request_trace;
    std::vector<float> request_query_rows;      // [hidden, query tokens]
    luce::pflash::PFlashQueryTrace query_trace;
    luce::pflash::PFlashQueryUnits qunits;
    std::map<std::vector<int>, std::vector<float>> qrows;
    if (request_probe) {
        request_trace.tau = st.request_tau;
        request_trace.store_invalidated_turns = request_rows_invalidated;
        const auto gather = [&](const std::vector<int> & positions,
                                std::vector<float> & rows, int & fresh,
                                int & stored) -> bool {
            rows.assign((size_t) hidden * positions.size(), 0.0f);
            fresh = 0;
            stored = 0;
            std::vector<int> old;
            std::vector<size_t> old_index;
            size_t index = 0;
            for (const auto & span : luce::pflash::pflash_request_token_spans(positions)) {
                int begin = span.begin;
                for (; begin < std::min(span.end, resume); ++begin) {
                    old.push_back(begin);
                    old_index.push_back(index++);
                }
                if (begin < span.end) {
                    const int n = span.end - begin;
                    ggml_backend_tensor_get(act_in, rows.data() + index * (size_t) hidden,
                                            (size_t) (begin - resume) * act_in->nb[1],
                                            (size_t) hidden * (size_t) n * sizeof(float));
                    fresh += n;
                    index += (size_t) n;
                }
            }
            if (!old.empty()) {
                std::vector<float> old_rows;
                if (!session->request_rows.lookup(old, hidden, old_rows)) return false;
                for (size_t k = 0; k < old.size(); ++k) {
                    std::copy_n(old_rows.data() + k * (size_t) hidden, (size_t) hidden,
                                rows.data() + old_index[k] * (size_t) hidden);
                }
                stored = (int) old.size();
            }
            return true;
        };
        if (request_v5) {
            if (!experiment.request_messages_all_error.empty()) {
                request_trace.fallback = "message_mapping: " + experiment.request_messages_all_error;
            } else {
                request_trace.set = luce::pflash::pflash_request_query_set_v5(
                    session->request_raw, experiment.request_messages_all, st.request_tau);
                request_trace.fallback = request_trace.set.fallback;
            }
        } else if (!experiment.request_messages_error.empty()) {
            request_trace.fallback = "message_mapping: " + experiment.request_messages_error;
        } else {
            if (st.request_turns) {
                std::vector<luce::pflash::PFlashRequestCandidate> candidates;
                const auto & hidx = experiment.request_human_index;
                const auto & ridx = experiment.request_recent_index;
                if (hidx.size() != experiment.request_messages.size() ||
                    ridx.size() != experiment.request_recent.size()) {
                    request_trace.fallback = "message_mapping: candidate indices missing";
                } else {
                    for (size_t k = 0; k < hidx.size(); ++k) {
                        candidates.push_back({hidx[k], true, false, experiment.request_messages[k]});
                    }
                    for (size_t k = 0; k < ridx.size(); ++k) {
                        bool merged = false;
                        for (auto & c : candidates) {
                            if (c.message_index == ridx[k]) { c.recent = true; merged = true; }
                        }
                        if (!merged) candidates.push_back({ridx[k], false, true, experiment.request_recent[k]});
                    }
                    request_trace.set = luce::pflash::pflash_request_query_set_v2(
                        session->request_raw, candidates, st.request_tau);
                    request_trace.fallback = request_trace.set.fallback;
                }
            } else {
                request_trace.set = luce::pflash::pflash_request_query_set(
                    session->request_raw, experiment.request_messages, st.request_tau);
                request_trace.fallback = request_trace.set.fallback;
            }
        }
        if (request_trace.fallback.empty()) {
            request_trace.policy_evaluated = true;
            request_trace.abstain = luce::pflash::pflash_request_abstain(
                request_trace.set, session->request_raw, st.request_tau, S, st.request_policy);
            if (!request_trace.abstain.reason.empty()) {
                request_trace.fallback = "abstain:" + request_trace.abstain.reason;
            }
        }
        if (request_trace.fallback.empty() &&
            !gather(request_trace.set.tokens, request_query_rows,
                    request_trace.rows_new, request_trace.rows_stored)) {
            request_trace.fallback = "rows_missing";
            request_query_rows.clear();
            request_trace.rows_new = request_trace.rows_stored = 0;
        }
        if (qpin && !qcand && resume > 0 &&
            (request_trace.rows_stored > 0 || request_trace.fallback == "rows_missing")) {
            redo = "request_set_rows_before_resume";
            return fail("query step: request-set rows not exactly available; redo from token 0");
        }
        std::vector<luce::pflash::PFlashRequestMessageRuns> retained;
        for (size_t k = request_trace.set.turns.size(); k-- > 0;) {
            retained.push_back(request_trace.set.turns[k]);
        }
        for (size_t k = request_trace.set.messages.size(); k-- > 0;) {
            retained.push_back(request_trace.set.messages[k]);
        }
        for (const auto & message : retained) {
            std::vector<int> turn = luce::pflash::pflash_request_turn_tokens(message);
            if (turn.empty()) continue;
            std::vector<float> rows;
            int fresh = 0;
            int stored = 0;
            if (gather(turn, rows, fresh, stored)) {
                request_trace.store_evicted_turns +=
                    session->request_rows.put(message.message, turn, rows.data(), hidden);
            }
        }
        request_trace.store_turns = (int) session->request_rows.turns().size();
        request_trace.store_tokens = session->request_rows.tokens();
        if (qstep) {
            // Units from the selector's own segmentation of this request (cut at the caller's pins), ranked by the
            // stage-1 logits; every query token set's rows are read now, exactly (see the gather below).
            const auto r0 = std::chrono::steady_clock::now();
            std::vector<PFlashTokenSpan> unit_segments;
            bool unit_density = false;
            segment(required_instruction_spans, unit_segments, unit_density);
            auto fail_qstep = [&](const char * msg) {
                log_segment(unit_segments, unit_density);
                return fail(msg);
            };
            if (unit_segments.empty()) {
                return fail_qstep("query step: the segment probe gave no segments");
            }
            qunits = luce::pflash::pflash_query_units(unit_segments, experiment.request_messages_all,
                                                      session->request_raw);
            if (!qunits.error.empty()) {
                const std::string message = "query step: units: " + qunits.error;
                return fail_qstep(message.c_str());
            }
            query_trace.evaluated = qcand || qpin;
            query_trace.config = qcfg;
            query_trace.units = (int) qunits.units.size();
            query_trace.system_used = qunits.system_used;
            std::vector<std::vector<int>> wanted;
            if (qcand) {
                query_trace.set = luce::pflash::pflash_query_candidate_set(qunits, session->request_raw, qcfg);
                if (query_trace.set.candidates.empty()) return fail_qstep("query step: no message content to rank");
                for (const auto & c : query_trace.set.candidates) {
                    wanted.push_back(c.tokens);
                    query_trace.query_tokens += (int) c.tokens.size();
                }
                if (qcfg.end_share > 0.0f) wanted.push_back({query_end - 1});
            }
            // Exact rows only: a set after the resume point from this call's activations, a set before it from
            // the session's F32 sets; anything else (a set that spans the resume point, or one never stored) redoes
            // the request from token 0. Never the BF16 request-row store.
            for (const auto & tokens : wanted) {
                if (tokens.empty() || qrows.count(tokens)) continue;
                std::vector<float> rows;
                int fresh = 0;
                int stored = 0;
                if (tokens.front() >= resume) {
                    if (!gather(tokens, rows, fresh, stored) || stored != 0) {
                        return fail_qstep("query step: query rows unavailable");
                    }
                } else if (tokens.back() >= resume || !session->query_sets.lookup(tokens, hidden, rows)) {
                    redo = tokens.back() >= resume ? "query_set_spans_resume" : "query_set_not_stored";
                    return fail_qstep("query step: query rows not exactly available; redo from token 0");
                }
                qrows.emplace(tokens, std::move(rows));
            }
            query_trace.rank_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - r0).count();
        }
    }
    cleanup();
    // The session now covers this prompt, resumable at the checkpoint and,
    // while the query stays put, reusing its rows.
    session->ids = ids;
    session->checkpoint = checkpoint;
    // Only exact rows are kept for later prompts: those in full grid ubatches
    // (pflash_scoring_resume.h).
    for (const auto & [tokens, rows] : qrows) {
        session->query_sets.put(tokens, rows.data(), hidden, exact_rows_end);
    }
    // The kept windows go last: the next request needs them, so the cap
    // evicts this call's windows first.
    for (const auto * list : {&windows, &keep_rows}) {
        for (const auto & window : *list) {
            session->query_windows.put(luce::pflash::pflash_span_tokens(window.begin, window.end),
                                       window.rows.data(), hidden, exact_rows_end);
        }
    }
    session->last_used = ++st.session_clock;

    // Block-15 NoPE Q/K scoring, once per query window: softmax over the
    // keys outside the query window (before it, and after it when those
    // tokens are candidates), then mean over heads and window tokens. No
    // window scores itself. The logits land in one [S, rows, H] buffer for
    // a single softmax. History windows share the query's key set and mix
    // into its mass at their weights.
    // The segment reranker's query: weighted mean of Q grouped across GQA heads
    // into [D * Hk], the reranker's key feature layout.
    std::vector<double> rerank_q_accum((size_t) D * (size_t) Hk, 0.0);
    double rerank_q_weight_sum = 0.0;
    const auto add_q_grouped = [&](const float * q_host, int n_q_toks, double weight,
                                   std::vector<double> & dst_vec) {
        if (!st.seg_reranker.loaded || !q_host || n_q_toks <= 0 || !(weight > 0.0)) return;
        if (dst_vec.size() != (size_t) D * (size_t) Hk) dst_vec.assign((size_t) D * (size_t) Hk, 0.0);
        const int gqa = std::max(1, H / Hk);
        const double scale = weight / ((double) n_q_toks * (double) gqa);
        for (int t = 0; t < n_q_toks; ++t) {
            for (int h = 0; h < H; ++h) {
                const int kv_h = h / gqa;
                const float * src = q_host + ((size_t) t * (size_t) H + (size_t) h) * (size_t) D;
                double * dst = dst_vec.data() + (size_t) kv_h * (size_t) D;
                for (int d = 0; d < D; ++d) {
                    dst[d] += (double) src[d] * scale;
                }
            }
        }
    };
    const auto accum_q_vec = [&](const std::vector<double> & vec, double weight) {
        if (!st.seg_reranker.loaded || vec.size() != rerank_q_accum.size() || !(weight > 0.0)) return;
        for (size_t i = 0; i < vec.size(); ++i) rerank_q_accum[i] += vec[i] * weight;
        rerank_q_weight_sum += weight;
    };

    // Pre-permute session->keys [D, Hk, S] -> [D, S, Hk] ONCE on the GPU for all
    // Block-15 NoPE scoring calls (windows + request candidate sets), eliminating
    // repeated chunked permute+copy loops across every tile.
    struct PermutedKeys {
        ggml_context *        ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        ggml_tensor *         tensor = nullptr;  // [D, S, Hk] F32 contiguous
        void release() {
            if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
            if (ctx) { ggml_free(ctx); ctx = nullptr; }
            tensor = nullptr;
        }
        ~PermutedKeys() { release(); }
    } k_perm_cache;
    const auto ensure_k_perm = [&]() -> bool {
        if (k_perm_cache.tensor) return true;
        ggml_init_params kip{};
        kip.mem_size = 4 * ggml_tensor_overhead() + 4096;
        kip.no_alloc = true;
        k_perm_cache.ctx = ggml_init(kip);
        if (!k_perm_cache.ctx) {
            set_last_error("qwen35 K_perm ctx allocation failed");
            return false;
        }
        k_perm_cache.tensor = ggml_new_tensor_3d(k_perm_cache.ctx, GGML_TYPE_F32, D, S, Hk);
        k_perm_cache.buf = ggml_backend_alloc_ctx_tensors(k_perm_cache.ctx, w.backend);
        if (!k_perm_cache.buf) {
            k_perm_cache.release();
            set_last_error("qwen35 K_perm buffer allocation failed");
            return false;
        }
        ggml_init_params gip{};
        gip.mem_size = 16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false) + 4096;
        gip.no_alloc = true;
        ggml_context * gctx = ggml_init(gip);
        if (!gctx) {
            k_perm_cache.release();
            set_last_error("qwen35 K_perm graph ctx allocation failed");
            return false;
        }
        ggml_cgraph * gf = ggml_new_graph_custom(gctx, 64, false);
        ggml_tensor * kv = ggml_view_3d(gctx, session->keys, D, Hk, S,
                                        session->keys->nb[1], session->keys->nb[2], 0);
        ggml_tensor * kp = ggml_permute(gctx, kv, 0, 2, 1, 3);  // [D, S, Hk]
        ggml_build_forward_expand(gf, ggml_cpy(gctx, kp, k_perm_cache.tensor));
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
        const bool ok = ggml_gallocr_alloc_graph(alloc, gf) &&
            ggml_backend_graph_compute(w.backend, gf) == GGML_STATUS_SUCCESS;
        ggml_gallocr_free(alloc);
        ggml_free(gctx);
        if (!ok) {
            k_perm_cache.release();
            set_last_error("qwen35 K_perm compute failed");
            return false;
        }
        return true;
    };

    // Block-15 NoPE scorer for contiguous query windows and gathered
    // request-probe sets: one [S, 1] broadcast mask uploaded per call, one batched
    // GEMM against the pre-permuted K_perm [D, S, Hk] per query tile, then softmax
    // and sum_rows on the device so only [1, S] comes back to the host.
    const auto score_rows_tiled = [&](const float * rows_data,
                                      int nq,
                                      const std::vector<float> & mask_row,
                                      std::vector<float> & out_mass,
                                      int * tiles_counter,
                                      std::vector<double> * q_vec_out,
                                      const char * err_tag) -> bool {
        if (q_vec_out) q_vec_out->assign((size_t) D * (size_t) Hk, 0.0);
        if (nq <= 0) {
            out_mass.assign((size_t) S, 0.0f);
            return true;
        }
        if (!ensure_k_perm()) return false;

        // Bound the two live [S, n*H] F32 scoring scratch tensors (logits, updated
        // in place by softmax, plus the transposed cont copy) to ~128 MiB with a
        // 1-query minimum floor so long contexts do not spike >1 GiB VRAM.
        // Keeps kTile=64 up to 16K tokens.
        constexpr int64_t kMaxScoreScratchBytes = 128LL * 1024 * 1024;
        const int64_t per_q_bytes =
            std::max<int64_t>(1, (int64_t) S * H * 2 * (int64_t) sizeof(float));
        const int kTile =
            std::max(1, std::min(64, (int) (kMaxScoreScratchBytes / per_q_bytes)));
        const bool gqa_fold = (H != Hk) && ((H % Hk) == 0);
        const float sm_scale = 1.0f / std::sqrt((float) D);

        ggml_init_params mip{};
        mip.mem_size = 4 * ggml_tensor_overhead() + 4096;
        mip.no_alloc = true;
        ggml_context * mctx = ggml_init(mip);
        if (!mctx) {
            set_last_error(std::string("qwen35 ") + err_tag + " mask ctx allocation failed");
            return false;
        }
        ggml_tensor * mask_1d = ggml_new_tensor_2d(mctx, GGML_TYPE_F32, S, 1);
        ggml_backend_buffer_t mbuf = ggml_backend_alloc_ctx_tensors(mctx, w.backend);
        if (!mbuf) {
            ggml_free(mctx);
            set_last_error(std::string("qwen35 ") + err_tag + " mask buffer allocation failed");
            return false;
        }
        ggml_backend_tensor_set(mask_1d, mask_row.data(), 0, (size_t) S * sizeof(float));

        std::vector<double> sum((size_t) S, 0.0);
        std::vector<float> part((size_t) S);
        bool failed = false;
        for (int q0 = 0; q0 < nq; q0 += kTile) {
            const int n = std::min(kTile, nq - q0);
            ggml_init_params lip{};
            lip.mem_size = 4 * ggml_tensor_overhead() + 4096;
            lip.no_alloc = true;
            ggml_context * lctx = ggml_init(lip);
            if (!lctx) {
                set_last_error(std::string("qwen35 ") + err_tag + " input ctx allocation failed");
                failed = true;
                break;
            }
            ggml_tensor * x_q = ggml_new_tensor_2d(lctx, GGML_TYPE_F32, hidden, n);
            ggml_backend_buffer_t lbuf = ggml_backend_alloc_ctx_tensors(lctx, w.backend);
            if (!lbuf) {
                ggml_free(lctx);
                set_last_error(std::string("qwen35 ") + err_tag + " input buffer allocation failed");
                failed = true;
                break;
            }
            ggml_backend_tensor_set(x_q, rows_data + (size_t) q0 * hidden, 0,
                                    (size_t) hidden * n * sizeof(float));

            ggml_init_params sip{};
            sip.mem_size = 64 * ggml_tensor_overhead() +
                           ggml_graph_overhead_custom(256, false) + 16 * 1024;
            sip.no_alloc = true;
            ggml_context * sctx = ggml_init(sip);
            if (!sctx) {
                ggml_backend_buffer_free(lbuf);
                ggml_free(lctx);
                set_last_error(std::string("qwen35 ") + err_tag + " graph ctx allocation failed");
                failed = true;
                break;
            }
            ggml_cgraph * sgf = ggml_new_graph_custom(sctx, 256, false);
            ggml_tensor * q_in = ggml_mul(sctx, ggml_rms_norm(sctx, x_q, w.rms_eps), L.attn_norm);
            ggml_tensor * Q = nullptr;
            if (st.head_loaded) {
                Q = ggml_reshape_3d(sctx, ggml_mul_mat(sctx, st.head_wq, q_in), D, H, n);
            } else {
                // Native block 15 packs query and gate rows per head; keep the query half.
                ggml_tensor * QG = ggml_reshape_3d(sctx, ggml_mul_mat(sctx, L.wq, q_in),
                                                   D * 2, H, n);
                Q = ggml_view_3d(sctx, QG, D, H, n,
                                 ggml_element_size(QG) * D * 2,
                                 ggml_element_size(QG) * D * 2 * H, 0);
            }
            Q = ggml_mul(sctx, ggml_rms_norm(sctx, Q, w.rms_eps), L.q_norm);
            if (st.seg_reranker.loaded && q_vec_out) {
                ggml_set_output(Q);
                ggml_build_forward_expand(sgf, Q);
            }
            ggml_tensor * Q_perm = ggml_cont(sctx, ggml_permute(sctx, Q, 0, 2, 1, 3));  // [D, n, H]
            ggml_tensor * Q_score = gqa_fold
                ? ggml_reshape_3d(sctx, Q_perm, D, (int64_t) n * (H / Hk), Hk)
                : Q_perm;
            // Single batched GEMM: K_perm [D, S, Hk] x Q_score [D, n*(H/Hk), Hk] -> [S, n*(H/Hk), Hk].
            ggml_tensor * logits = ggml_mul_mat(sctx, k_perm_cache.tensor, Q_score);
            // Reshape [S, n*(H/Hk), Hk] -> [S, 1, n*H] so mask_1d [S, 1] broadcasts cleanly across dim 2.
            ggml_tensor * logits_3d = ggml_reshape_3d(sctx, logits, S, 1, (int64_t) n * H);
            ggml_tensor * probs = ggml_soft_max_ext_inplace(sctx, logits_3d, mask_1d, sm_scale, 0.0f);
            // Sum over (query, head) per key on device: [S, n*H] -> [n*H, S] -> [1, S].
            ggml_tensor * per_key = ggml_sum_rows(sctx, ggml_cont(sctx,
                ggml_transpose(sctx, ggml_reshape_2d(sctx, probs, S, (int64_t) n * H))));
            ggml_set_output(per_key);
            ggml_build_forward_expand(sgf, per_key);

            ggml_gallocr_t salloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
            const bool ok = ggml_gallocr_alloc_graph(salloc, sgf) &&
                ggml_backend_graph_compute(w.backend, sgf) == GGML_STATUS_SUCCESS;
            if (ok) {
                ggml_backend_tensor_get(per_key, part.data(), 0, (size_t) S * sizeof(float));
                if (st.seg_reranker.loaded && q_vec_out) {
                    std::vector<float> q_h((size_t) D * (size_t) H * (size_t) n);
                    ggml_backend_tensor_get(Q, q_h.data(), 0, q_h.size() * sizeof(float));
                    add_q_grouped(q_h.data(), n, (double) n / (double) std::max(1, nq), *q_vec_out);
                }
            }
            ggml_gallocr_free(salloc);
            ggml_free(sctx);
            ggml_backend_buffer_free(lbuf);
            ggml_free(lctx);
            if (!ok) {
                set_last_error(std::string("qwen35 ") + err_tag + " graph compute failed");
                failed = true;
                break;
            }
            const size_t nonfinite = count_nonfinite_scores(part.data(), part.size());
            if (nonfinite != 0) {
                const std::string message =
                    std::string("non-finite Qwen3.5 scoring-head scores (") + err_tag + "): " +
                    std::to_string(nonfinite) + "/" + std::to_string(part.size());
                std::fprintf(stderr, "[pflash] ERROR: %s\n", message.c_str());
                std::fflush(stderr);
                set_last_error(message);
                failed = true;
                break;
            }
            for (int j = 0; j < S; ++j) sum[(size_t) j] += (double) part[(size_t) j];
            if (tiles_counter) ++(*tiles_counter);
        }
        ggml_backend_buffer_free(mbuf);
        ggml_free(mctx);
        if (failed) return false;

        const double denominator = (double) H * (double) nq;
        out_mass.assign((size_t) S, 0.0f);
        for (int j = 0; j < S; ++j) out_mass[(size_t) j] = (float) (sum[(size_t) j] / denominator);
        return true;
    };

    const auto score_window = [&](const ScoredWindow & window,
                                  std::vector<float> & mass,
                                  std::vector<double> * q_vec_out = nullptr) -> bool {
        const int nq = window.end - window.begin;
        std::vector<float> mask_row((size_t) S, -INFINITY);
        std::fill_n(mask_row.begin(), (size_t) query_start, 0.0f);
        if (experiment.query_suffix_candidates) {
            std::fill_n(mask_row.begin() + query_end, (size_t) (S - query_end), 0.0f);
        }
        if (window.end <= query_start) {
            std::fill_n(mask_row.begin() + window.begin, (size_t) nq, -INFINITY);
        }
        return score_rows_tiled(window.rows.data(), nq, mask_row, mass, nullptr, q_vec_out, "score");
    };
    std::vector<float> token_mass;
    // Today's windows: each scored on its own, mixed at their weights.
    const auto score_windows = [&](std::vector<double> * q_vec_out = nullptr) -> bool {
        token_mass.clear();
        double total_weight = 0.0;
        if (q_vec_out) q_vec_out->assign((size_t) D * (size_t) Hk, 0.0);
        for (const auto & window : windows) {
            std::vector<float> mass;
            std::vector<double> wq;
            if (!score_window(window, mass, q_vec_out ? &wq : nullptr)) return false;
            if (token_mass.empty()) token_mass.assign(mass.size(), 0.0f);
            for (size_t i = 0; i < mass.size(); ++i) {
                token_mass[i] += (float) window.weight * mass[i];
            }
            if (q_vec_out && wq.size() == q_vec_out->size()) {
                for (size_t i = 0; i < wq.size(); ++i) (*q_vec_out)[i] += window.weight * wq[i];
            }
            total_weight += window.weight;
        }
        for (auto & value : token_mass) value = (float) (value / total_weight);
        if (q_vec_out && total_weight > 0.0) {
            for (auto & v : *q_vec_out) v /= total_weight;
        }
        return true;
    };
    // Request probe: ONE gathered query set with one common key mask --
    // today's key range (the context before the query window, and after it
    // when those tokens are candidates) minus every query token -- mean over
    // heads and query tokens, as for a window.
    const auto score_request_set = [&](const std::vector<int> & tokens,
                                       const std::vector<float> & rows,
                                       bool mask_query_tokens, int & tiles,
                                       std::vector<double> * q_vec_out = nullptr) -> bool {
        std::vector<float> mask_row((size_t) S, -INFINITY);
        std::fill_n(mask_row.begin(), (size_t) query_start, 0.0f);
        if (experiment.query_suffix_candidates) {
            std::fill(mask_row.begin() + query_end, mask_row.end(), 0.0f);
        }
        if (mask_query_tokens) {
            for (const int t : tokens) mask_row[(size_t) t] = -INFINITY;
        }
        return score_rows_tiled(rows.data(), (int) tokens.size(), mask_row, token_mass,
                                &tiles, q_vec_out, "request probe");
    };
    // ---------------------------------------------------------------- query step 1
    // Masses of query token sets, cached per set (one tile stream per set; tiles never mix sets).
    std::map<std::vector<int>, std::vector<float>> qmass;
    std::map<std::vector<int>, std::vector<double>> qmass_qvec;
    int qtiles = 0;
    const auto query_mass = [&](const std::vector<int> & tokens) -> const std::vector<float> * {
        auto hit = qmass.find(tokens);
        if (hit != qmass.end()) return &hit->second;
        auto rows = qrows.find(tokens);
        if (rows == qrows.end()) {
            set_last_error("query step: the rows of a query set are missing");
            return nullptr;
        }
        std::vector<double> qv;
        if (!score_request_set(tokens, rows->second, /*mask_query_tokens=*/true, qtiles,
                               st.seg_reranker.loaded ? &qv : nullptr)) return nullptr;
        if (st.seg_reranker.loaded) qmass_qvec.emplace(tokens, std::move(qv));
        return &qmass.emplace(tokens, token_mass).first->second;
    };
    // Today's windows as a blend candidate: its mass, scored and mixed as usual (cached),
    // and its score = the unit score of s1 over the window tokens inside non-system message content.
    std::vector<float> qwindows_mass;
    std::vector<double> qwindows_qvec;
    bool qwindows_ready = false;
    const auto windows_mass_cached = [&]() -> const std::vector<float> * {
        if (!qwindows_ready) {
            if (!score_windows(st.seg_reranker.loaded ? &qwindows_qvec : nullptr)) return nullptr;
            qwindows_mass = token_mass;
            qwindows_ready = true;
        }
        return &qwindows_mass;
    };
    const auto tw_unit_score = [&](luce::pflash::PFlashUnitScore mode) -> float {
        std::vector<uint8_t> content((size_t) S, 0);
        for (const auto & m : experiment.request_messages_all) {
            if (m.role == luce::pflash::PFlashRequestRole::System) continue;
            for (int t = std::max(0, m.span.begin); t < std::min(S, m.span.end); ++t) content[(size_t) t] = 1;
        }
        std::vector<uint8_t> in_tw((size_t) S, 0);
        for (const auto & w : windows) {
            for (int t = std::max(0, w.begin); t < std::min(S, w.end); ++t) in_tw[(size_t) t] = 1;
        }
        double sum = 0.0;
        int n = 0;
        float mx = -std::numeric_limits<float>::infinity();
        for (int t = 0; t < S && t < (int) session->request_raw.size(); ++t) {
            if (!in_tw[(size_t) t] || !content[(size_t) t]) continue;
            const float v = session->request_raw[(size_t) t];
            if (!std::isfinite(v)) continue;
            sum += v;
            ++n;
            mx = std::max(mx, v);
        }
        if (n == 0) return std::numeric_limits<float>::quiet_NaN();
        return mode == luce::pflash::PFlashUnitScore::Max ? mx : (float) (sum / n);
    };
    // The blend (pflash_query_candidates.h): probe candidates, TW, END, with their weights.
    const auto query_blend_mass = [&](const luce::pflash::PFlashQueryCandidateSet & set,
                                      const luce::pflash::PFlashQueryConfig & q, std::vector<float> & out,
                                      luce::pflash::PFlashQueryBlend & blend, float & tw_score_out,
                                      bool accumulate_rerank_q = false) -> bool {
        tw_score_out = q.tw_score ? tw_unit_score(q.unit_score) : std::numeric_limits<float>::quiet_NaN();
        blend = luce::pflash::pflash_query_blend(set, q, tw_score_out);
        std::vector<std::vector<float>> masses;
        std::vector<double> weights;
        for (size_t i = 0; i < set.candidates.size(); ++i) {
            const auto * m = query_mass(set.candidates[i].tokens);
            if (!m) return false;
            masses.push_back(*m);
            weights.push_back(blend.probe[i]);
            if (accumulate_rerank_q) {
                auto qit = qmass_qvec.find(set.candidates[i].tokens);
                if (qit != qmass_qvec.end()) accum_q_vec(qit->second, blend.probe[i]);
            }
        }
        if (blend.tw > 0.0) {
            const auto * m = windows_mass_cached();
            if (!m) return false;
            masses.push_back(*m);
            weights.push_back(blend.tw);
            if (accumulate_rerank_q) accum_q_vec(qwindows_qvec, blend.tw);
        }
        if (blend.end > 0.0) {
            const std::vector<int> end_tok{query_end - 1};
            const auto * m = query_mass(end_tok);
            if (!m) return false;
            masses.push_back(*m);
            weights.push_back(blend.end);
            if (accumulate_rerank_q) {
                auto qit = qmass_qvec.find(end_tok);
                if (qit != qmass_qvec.end()) accum_q_vec(qit->second, blend.end);
            }
        }
        std::string error;
        if (!luce::pflash::pflash_query_aggregate(masses, weights, q.aggregate, out, error)) {
            set_last_error(error);
            return false;
        }
        return true;
    };
    // No stage 2 with the query step: the selector budget is floor(S * keep_ratio).
    const int qbudget = (int) std::floor((double) S * (double) keep_ratio);
    const auto merge_spans = [](std::vector<PFlashTokenSpan> spans) {
        std::sort(spans.begin(), spans.end(), [](const PFlashTokenSpan & a, const PFlashTokenSpan & b) {
            return a.begin != b.begin ? a.begin < b.begin : a.end < b.end;
        });
        std::vector<PFlashTokenSpan> merged;
        for (const auto & span : spans) {
            if (span.end <= span.begin) continue;
            if (!merged.empty() && span.begin <= merged.back().end) {
                merged.back().end = std::max(merged.back().end, span.end);
            } else {
                merged.push_back(span);
            }
        }
        return merged;
    };
    // The verbatim pin on top of ``base``; its cap halves until the charged pins fit the budget (0 at worst).
    const auto fit_pin = [&](const std::vector<PFlashTokenSpan> & base, const luce::pflash::PFlashQueryConfig & q,
                             luce::pflash::PFlashQueryPin & pin_out, int & cap_out) {
        int cap = q.pin_tokens;
        for (;;) {
            pin_out = luce::pflash::pflash_query_pin(qunits, cap, q.unit_score, q.pin_margin, q.pin_gap);
            auto p = base;
            p.insert(p.end(), pin_out.spans.begin(), pin_out.spans.end());
            p = merge_spans(std::move(p));
            if (cap <= 0 || pin_out.spans.empty()) {
                cap_out = cap;
                return p;
            }
            std::vector<PFlashTokenSpan> segs;
            bool dens = false;
            segment(p, segs, dens);
            if (charged(p, segs) <= qbudget) {
                cap_out = cap;
                return p;
            }
            cap /= 2;
        }
    };
    // The served candidates replace the request set (named, not a fallback to windows).
    if (qcand) request_trace.fallback = "replaced_by_query_candidates";
    if (qcand || qpin) request_trace.query = &query_trace;
    query_trace.sessions = sessions_on;
    query_trace.session_resume = resume;
    query_trace.redo = !redone.empty();
    query_trace.redo_reason = redone;
    bool request_set = request_probe && request_trace.fallback.empty();
    const auto pass2_started = std::chrono::steady_clock::now();
    rerank_q_accum.assign((size_t) D * (size_t) Hk, 0.0);
    rerank_q_weight_sum = 0.0;
    if (qcand) {
        const int tiles_before = qtiles;
        qmass.clear();
        qmass_qvec.clear();
        qwindows_ready = false;
        std::vector<float> aggregated;
        if (!query_blend_mass(query_trace.set, qcfg, aggregated, query_trace.blend, query_trace.tw_score,
                              /*accumulate_rerank_q=*/true)) {
            session->ids.clear();
            return {};
        }
        token_mass = std::move(aggregated);
        query_trace.tiles = qtiles - tiles_before;
        query_trace.score_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - pass2_started).count();
    } else {
        std::vector<double> single_qv;
        std::vector<double> * qv_ptr = st.seg_reranker.loaded ? &single_qv : nullptr;
        const bool ok_score = request_set
            ? score_request_set(request_trace.set.tokens, request_query_rows,
                                /*mask_query_tokens=*/true, request_trace.query_tiles, qv_ptr)
            : score_windows(qv_ptr);
        if (!ok_score) {
            session->ids.clear();
            return {};
        }
        if (st.seg_reranker.loaded) accum_q_vec(single_qv, 1.0);
    }
    const float select_keep_ratio = keep_ratio;
    const int select_budget = (int) std::floor((double) S * (double) keep_ratio);
    auto t2 = std::chrono::steady_clock::now();
    std::fprintf(stderr,
        "[qwen35-scorer] forward %.2fs (blocks 0-%d, S=%d, resumed at %d, "
        "%d new) score %.2fs (%zu query windows) total %.2fs head=%s\n",
        std::chrono::duration<double>(t1 - t0).count(), kQwen35HeadBlock - 1, S,
        resume, n_new,
        std::chrono::duration<double>(t2 - t1).count(),
        qcand ? query_trace.set.candidates.size() : request_set ? (size_t) 1 : windows.size(),
        std::chrono::duration<double>(t2 - t0).count(),
        st.head_loaded ? "trained" : "native-block15");
    std::fflush(stderr);
    pflash_set_scoring_stats({resume, n_new,
                              qcand ? (int) query_trace.set.candidates.size()
                                    : request_set ? 1 : (int) windows.size(),
                              std::chrono::duration<double>(t1 - t0).count()});

    std::vector<PFlashTokenSpan> segments;
    bool density = false;
    // The pins the selector charges: the caller's, plus the query tokens of
    // a scored request set.
    std::vector<PFlashTokenSpan> pins = required_instruction_spans;
    if (request_set) {
        for (const auto & span : luce::pflash::pflash_request_token_spans(request_trace.set.tokens)) {
            pins.push_back(span);
        }
        // v2 bounded recency pin: the latest human message's last K tokens, pinned (not scored).
        if (!request_trace.set.messages.empty()) {
            request_trace.recency_pin = luce::pflash::pflash_request_recency_pin(
                request_trace.set.messages[0].message, st.request_policy.recency_pin_tokens);
            if (request_trace.recency_pin.begin >= 0) pins.push_back(request_trace.recency_pin);
        }
        std::sort(pins.begin(), pins.end(), [](const PFlashTokenSpan & a, const PFlashTokenSpan & b) {
            return a.begin != b.begin ? a.begin < b.begin : a.end < b.end;
        });
        std::vector<PFlashTokenSpan> merged;
        for (const auto & span : pins) {
            if (!merged.empty() && span.begin <= merged.back().end) {
                merged.back().end = std::max(merged.back().end, span.end);
            } else {
                merged.push_back(span);
            }
        }
        pins.swap(merged);
    }
    // Query pin: the latest user message's top whole units, charged, shrunk to fit the budget.
    if (qpin) pins = fit_pin(pins, qcfg, query_trace.pin, query_trace.pin_cap_used);
    segment(pins, segments, density);

    if (request_probe) {
        // Tokens the selector will keep whatever their score, counted the
        // way it charges them (pflash_chunk_is_structurally_required over
        // the candidates); the query pins must fit the reader budget, or
        // the whole request falls back to today's windows and pins.
        request_trace.token_budget = select_budget;
        if (request_set) {
            request_trace.query_tokens = (int) request_trace.set.tokens.size();
            int already = 0;
            for (const int t : request_trace.set.tokens) {
                bool pinned = t >= sel_query_begin && t < sel_query_end;
                for (const auto & span : required_instruction_spans) {
                    pinned = pinned || (t >= span.begin && t < span.end);
                }
                already += pinned ? 1 : 0;
            }
            request_trace.pinned_new_tokens = request_trace.query_tokens - already;
            if (request_trace.recency_pin.begin >= 0) {
                const auto & q = request_trace.set.tokens;
                for (int t = request_trace.recency_pin.begin; t < request_trace.recency_pin.end; ++t) {
                    bool pinned = (t >= sel_query_begin && t < sel_query_end) ||
                        std::binary_search(q.begin(), q.end(), t);
                    for (const auto & span : required_instruction_spans) {
                        pinned = pinned || (t >= span.begin && t < span.end);
                    }
                    request_trace.recency_pin_new_tokens += pinned ? 0 : 1;
                }
            }
            request_trace.mandatory_tokens = charged(pins, segments);
            std::vector<PFlashTokenSpan> base_segments;
            bool base_density = false;
            segment(required_instruction_spans, base_segments, base_density);
            request_trace.mandatory_tokens_base = charged(required_instruction_spans, base_segments);
            if (request_trace.mandatory_tokens > request_trace.token_budget &&
                request_trace.mandatory_tokens_base <= request_trace.token_budget) {
                request_trace.fallback = "pins_exceed_budget";
                request_trace.query_tokens = 0;
                request_trace.recency_pin = {-1, -1};
                request_trace.recency_pin_new_tokens = 0;
                request_set = false;
                pins = required_instruction_spans;
                segments.swap(base_segments);
                density = base_density;
                // The windows' query replaces the request set's, for the reranker too.
                std::vector<double> window_qv;
                if (!score_windows(st.seg_reranker.loaded ? &window_qv : nullptr)) {
                    log_segment(segments, density);
                    session->ids.clear();
                    return {};
                }
                rerank_q_accum.assign(rerank_q_accum.size(), 0.0);
                rerank_q_weight_sum = 0.0;
                accum_q_vec(window_qv, 1.0);
            }
        } else {
            request_trace.mandatory_tokens = charged(pins, segments);
            request_trace.mandatory_tokens_base = request_trace.mandatory_tokens;
        }
        if (request_set) {
            auto probe_pins = luce::pflash::pflash_request_token_spans(request_trace.set.tokens);
            if (request_trace.recency_pin.begin >= 0) probe_pins.push_back(request_trace.recency_pin);
            pflash_set_probe_pins(std::move(probe_pins));
        }
    }
    log_segment(segments, density);
    if (request_probe) {
        std::string policy_note;
        if (request_trace.policy_evaluated) {
            char buf[256];
            std::snprintf(buf, sizeof(buf), " policy: margin mean %.3f max %.3f, latest share %.3f, runs %d",
                          request_trace.abstain.mean_margin, request_trace.abstain.max_margin,
                          request_trace.abstain.latest_share, request_trace.abstain.runs);
            policy_note = buf;
        }
        if (request_trace.recency_pin.begin >= 0) {
            policy_note += " recency_pin=[" + std::to_string(request_trace.recency_pin.begin) + "," +
                std::to_string(request_trace.recency_pin.end) + ") new " +
                std::to_string(request_trace.recency_pin_new_tokens);
        }
        std::string runs;
        for (const auto & m : request_trace.set.messages) {
            runs += " [" + std::to_string(m.message.begin) + "," + std::to_string(m.message.end) +
                ")x" + std::to_string(m.runs.size()) + "/" + std::to_string(m.request_tokens);
        }
        std::fprintf(stderr,
            "[pflash-request-probe] %s query_tokens=%d (latest %d, previous %d%s) rows new=%d "
            "stored=%d pins_new=%d mandatory=%d (base %d) budget=%d store=%d turns/%d tokens "
            "(evicted %d, invalidated %d) messages:%s%s\n",
            request_set ? "scored" : ("fallback=" + request_trace.fallback).c_str(),
            request_trace.query_tokens, request_trace.set.latest_tokens,
            request_trace.set.previous_tokens, request_trace.set.capped ? ", capped" : "",
            request_trace.rows_new, request_trace.rows_stored, request_trace.pinned_new_tokens,
            request_trace.mandatory_tokens, request_trace.mandatory_tokens_base,
            request_trace.token_budget, request_trace.store_turns, request_trace.store_tokens,
            request_trace.store_evicted_turns, request_trace.store_invalidated_turns,
            runs.empty() ? " none" : runs.c_str(),
            policy_note.c_str());
        if (query_trace.evaluated) {
            const auto & qs = query_trace.set;
            std::string cands;
            for (const auto & c : qs.candidates) {
                char buf[96];
                std::snprintf(buf, sizeof(buf), " m%d:%zut/%zuu s=%.3f w=%.3f", c.message, c.tokens.size(),
                              c.units.size(), (double) c.score, c.weight);
                cands += buf;
            }
            std::fprintf(stderr,
                "[pflash-query] candidates=%d/%d unit_score=%s agg=%s T=%.3g extent=%s%.3g null_margin=%.3g units=%d%s "
                "latest=%d (units %d, z %.3f%s) query_tokens=%d tiles=%d pin=%d/%d (cap %d, units %d, skipped %d) "
                "rank=%.1fms score=%.1fms tw=%.3f (score %.3f) end=%.3f%s\n",
                (int) qs.candidates.size(), qcfg.candidates, luce::pflash::pflash_unit_score_name(qcfg.unit_score),
                luce::pflash::pflash_query_aggregate_name(qcfg.aggregate), (double) qcfg.temperature,
                qcfg.extent_gap ? "gap/" : "", (double) qcfg.extent_margin, (double) qcfg.null_margin, query_trace.units,
                query_trace.system_used ? " (system)" : "", qs.latest, qs.latest_units, (double) qs.latest_z,
                qs.latest_null ? ", NULL" : "", query_trace.query_tokens, query_trace.tiles, query_trace.pin.tokens,
                qcfg.pin_tokens, query_trace.pin_cap_used, query_trace.pin.units, query_trace.pin.skipped,
                query_trace.rank_us / 1000.0, query_trace.score_us / 1000.0, query_trace.blend.tw,
                (double) query_trace.tw_score, query_trace.blend.end, cands.c_str());
        }
        std::fflush(stderr);
    }

    luce::pflash::PFlashSelectionConfig request_config;
    if (request_probe) {
        request_config = experiment;
        request_config.request_probe_trace = &request_trace;
    }
    // Release pre-permuted K buffer before reranker/selection VRAM allocations.
    k_perm_cache.release();
    std::vector<float> adj_kcos;
    std::vector<float> seg_keys;
    std::vector<float> rerank_scores;
    pflash_set_reranker_error({});
    pflash_set_loaded_assets_identity(st.loaded_assets_identity);
    if (st.seg_reranker.loaded) {
        std::vector<PFlashTokenSpan> fixed_spans;
        const std::vector<PFlashTokenSpan> * spans_for_kcos = &segments;
        if (segments.empty() && experiment.chunk_size > 0) {
            const int n_chunks = (S + experiment.chunk_size - 1) / experiment.chunk_size;
            fixed_spans.reserve((size_t) n_chunks);
            for (int c = 0; c < n_chunks; ++c) {
                const int b = c * experiment.chunk_size;
                fixed_spans.push_back({b, std::min(S, b + experiment.chunk_size)});
            }
            spans_for_kcos = &fixed_spans;
        }
        // Match training: feed ONLY non-mandatory, positive-mass body segments
        // into kcos, scalar z-scoring, and the bidirectional reranker so BOS/
        // instruction sinks and zero-mass masked query tails cannot distort
        // per-document z-scores or max_log_dens.
        const int rr_q_end = score_query_end < 0 ? S : score_query_end;
        const int rr_q_begin = rr_q_end - std::min(n_lookahead, rr_q_end);
        const size_t n_all = spans_for_kcos->size();
        std::vector<size_t> opt_indices;
        std::vector<PFlashTokenSpan> opt_spans;
        std::vector<float> base_log_dens(n_all, (float) std::log(1e-12));
        opt_indices.reserve(n_all);
        opt_spans.reserve(n_all);
        const int tok_sz = (int) token_mass.size();
        for (size_t i = 0; i < n_all; ++i) {
            const int b = (*spans_for_kcos)[i].begin;
            const int e = (*spans_for_kcos)[i].end;
            const int s0 = std::max(0, b);
            const int e0 = std::min(tok_sz, e);
            const int len = std::max(1, e0 - s0);
            double sum_m = 0.0;
            for (int t = s0; t < e0; ++t) sum_m += (double) token_mass[(size_t) t];
            base_log_dens[i] = (float) std::log(std::max(1e-12, sum_m / (double) len));
            const bool mandatory =
                luce::pflash::pflash_chunk_is_structurally_required(
                    b, e, rr_q_begin, rr_q_end, S, pins,
                    /*query_suffix_structural=*/ !experiment.query_suffix_candidates);
            if (!mandatory && b > 0 && sum_m > 0.0) {
                opt_indices.push_back(i);
                opt_spans.push_back({b, e});
            }
        }
        // The loader admits only kbar ([k_bar]) and kbar_katt ([k_bar, k_att]) at the drafter's key width.
        const bool rr_is_katt = (st.seg_reranker.feature_mode == "kbar_katt");
        std::vector<float> seg_katt;
        std::string rr_failure;
        if (opt_spans.size() >= 2) {
            if (!pflash_adjacent_segment_key_cosine(
                    w.backend, session->keys, opt_spans, adj_kcos, &seg_keys,
                    rr_is_katt ? token_mass.data() : nullptr,
                    rr_is_katt ? &seg_katt : nullptr)) {
                adj_kcos.clear();
                seg_keys.clear();
                seg_katt.clear();
                rr_failure = "segment key cosine computation failed";
            }
            const int k_dim = D * Hk;
            const int expected_in_dim = rr_is_katt ? (2 * k_dim) : k_dim;
            if (rr_failure.empty() &&
                (seg_keys.empty() ||
                 (rr_is_katt && seg_katt.size() != seg_keys.size()))) {
                rr_failure = "segment reranker key features missing";
            }
            if (rr_failure.empty() && !(rerank_q_weight_sum > 0.0)) {
                rr_failure = "segment reranker query vector missing";
            }
            if (rr_failure.empty()) {
                auto t_rr0 = std::chrono::steady_clock::now();
                std::vector<float> q_vec((size_t) k_dim, 0.0f);
                double norm_sq = 0.0;
                for (size_t d = 0; d < q_vec.size(); ++d) {
                    const double v = rerank_q_accum[d] / rerank_q_weight_sum;
                    q_vec[d] = (float) v;
                    norm_sq += v * v;
                }
                if (norm_sq > 1e-16 && std::isfinite(norm_sq)) {
                    const float inv_n = (float) (1.0 / std::sqrt(norm_sq));
                    for (float & v : q_vec) v *= inv_n;
                }
                std::vector<float> seg_dual;
                std::vector<float> q_dual;
                const float * rr_seg_ptr = seg_keys.data();
                const float * rr_q_ptr   = q_vec.data();
                if (rr_is_katt) {
                    const size_t n_opt = opt_spans.size();
                    seg_dual.resize(n_opt * (size_t) expected_in_dim);
                    for (size_t i = 0; i < n_opt; ++i) {
                        float * dst = seg_dual.data() + i * (size_t) expected_in_dim;
                        std::memcpy(dst, seg_keys.data() + i * (size_t) k_dim,
                                    (size_t) k_dim * sizeof(float));
                        std::memcpy(dst + k_dim, seg_katt.data() + i * (size_t) k_dim,
                                    (size_t) k_dim * sizeof(float));
                    }
                    q_dual.resize((size_t) expected_in_dim);
                    std::memcpy(q_dual.data(), q_vec.data(), (size_t) k_dim * sizeof(float));
                    std::memcpy(q_dual.data() + k_dim, q_vec.data(), (size_t) k_dim * sizeof(float));
                    rr_seg_ptr = seg_dual.data();
                    rr_q_ptr   = q_dual.data();
                }
                std::vector<float> rr_scalars, rr_log_dens, opt_scores;
                pflash_reranker_build_scalars(
                    S, token_mass, opt_spans,
                    container_probs.empty() ? nullptr : &container_probs,
                    adj_kcos.empty() ? nullptr : &adj_kcos,
                    rr_scalars, rr_log_dens);
                float pred_m = 0.0f;
                std::string rr_err;
                if (pflash_seg_reranker_run(
                        st.seg_reranker, w.backend, (int) opt_spans.size(),
                        rr_seg_ptr, rr_q_ptr, rr_scalars.data(), rr_log_dens.data(),
                        opt_scores, nullptr, &pred_m, &rr_err)) {
                    rerank_scores = std::move(base_log_dens);
                    for (size_t k = 0; k < opt_indices.size(); ++k) {
                        rerank_scores[opt_indices[k]] = opt_scores[k];
                    }
                    auto t_rr1 = std::chrono::steady_clock::now();
                    std::fprintf(stderr,
                        "[pflash-reranker] scored N=%zu/%zu time=%.2fms pred_m=%.3f\n",
                        opt_spans.size(), n_all,
                        std::chrono::duration<double, std::milli>(t_rr1 - t_rr0).count(),
                        pred_m);
                    std::fflush(stderr);
                } else {
                    rr_failure = rr_err.empty() ? "seg reranker graph compute failed" : rr_err;
                }
            }
            if (!rr_failure.empty()) {
                pflash_set_reranker_error(rr_failure);
                std::fprintf(stderr,
                    "[pflash-reranker] WARNING failed (%s), falling back to Stage-1 scores\n",
                    rr_failure.c_str());
                std::fflush(stderr);
                rerank_scores.clear();
            }
        }
    }
    return select_pflash_chunks(
        ids, token_mass, select_keep_ratio, n_lookahead, score_query_end,
        /*pool_kernel=*/1, request_probe ? request_config : experiment, pins,
        /*direct_mass=*/true, /*write_trace=*/true,
        segments.empty() ? nullptr : &segments, density,
        rerank_scores.empty() ? nullptr : &rerank_scores);
}

std::vector<int32_t> qwen35_strict_score_and_compress(
    Qwen35DrafterState & st,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int n_lookahead,
    int score_query_end,
    const luce::pflash::PFlashSelectionConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans) {
    std::string redo;
    auto kept = strict_score_impl(st, ids, keep_ratio, n_lookahead, score_query_end, experiment,
                                  required_instruction_spans,
                                  std::numeric_limits<int>::max(), std::string(), redo);
    if (redo.empty()) return kept;
    // A query row the resumed session could not give exactly: the whole request again from token 0 (its session
    // was forgotten, so it is the one reused), never an approximation.
    std::fprintf(stderr, "[qwen35-scorer] redo from token 0: %s\n", redo.c_str());
    std::fflush(stderr);
    const std::string reason = redo;
    kept = strict_score_impl(st, ids, keep_ratio, n_lookahead, score_query_end, experiment,
                             required_instruction_spans,
                             /*resume_cap=*/0, reason, redo);
    if (!redo.empty()) {
        set_last_error("query step: redo from token 0 asked for another redo");
        return {};
    }
    return kept;
}

std::vector<int32_t> qwen35_drafter_score_and_compress(
    DrafterContext & ctx,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel,
    int score_query_end,
    const luce::pflash::PFlashSelectionConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans) {
    if (!ctx.state) {
        set_last_error("qwen35 drafter state missing");
        return {};
    }
    auto * st = static_cast<Qwen35DrafterState *>(ctx.state);
    // Standalone scoring without PFlash assets preserves the legacy all-layer
    // contract (honoring caller chunk_size and pool_kernel).
    if (!st->head_loaded && !st->probe_loaded && !st->request_loaded &&
        !st->seg_reranker.loaded && !experiment.query_probe &&
        required_instruction_spans.empty() && !experiment.query_suffix_candidates &&
        experiment.history_queries.empty() && experiment.turn_query.begin < 0) {
        luce::pflash::PFlashSelectionConfig legacy = experiment;
        legacy.configured = false;
        legacy.selection_active = false;
        legacy.mode = luce::pflash::PFlashSelectionMode::Legacy;
        return qwen35_score_and_compress(
            st->weights, ids, keep_ratio, chunk_size, n_lookahead, pool_kernel,
            score_query_end, legacy, required_instruction_spans);
    }
    // Sessions offloaded when the drafter was last released come back before
    // any scoring.
    st->session_ctx_limit = luce::pflash::pflash_session_ctx_limit(st->ctx_train, ctx.engine_max_ctx);
    qwen35_restore_sessions(ctx, *st, ids);
    auto kept = qwen35_strict_score_and_compress(
        *st, ids, keep_ratio, n_lookahead, score_query_end, experiment,
        required_instruction_spans);
    // Non-finite head scores (once in ~500 development requests, not
    // reproduced) leave the scoring session forgotten: score the prompt
    // again from scratch, one drafter forward, instead of failing the
    // request.
    if (kept.empty() &&
        std::strncmp(luce_last_error(), "non-finite", 10) == 0) {
        std::fprintf(stderr,
            "[qwen35-scorer] non-finite scores; rescoring from scratch\n");
        std::fflush(stderr);
        kept = qwen35_strict_score_and_compress(
            *st, ids, keep_ratio, n_lookahead, score_query_end, experiment,
            required_instruction_spans);
    }
    return kept;
}

} // namespace luce::common
