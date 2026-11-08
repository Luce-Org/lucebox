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
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <map>
#include <mutex>
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

// The drafter cache's KV types: the target's resolution without the
// LUCE_KV_TQ3 shorthand, since the drafter never wants the TurboQuant
// rotation. Resolved without touching the environment, which other threads
// read while a concurrent server scores a prompt.
struct DrafterKvTypes {
    ggml_type k = GGML_TYPE_COUNT;
    ggml_type v = GGML_TYPE_COUNT;
    DrafterKvTypes() {
        luce::resolve_kv_types(k, v, GGML_TYPE_COUNT, GGML_TYPE_COUNT,
                               /*tq3_shorthand=*/false);
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
    const DrafterKvTypes kv;
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
        env_int("PFLASH_COMPRESS_ANCHOR_RADIUS",   -1),
        env_int("PFLASH_COMPRESS_MAX_ANCHOR_HITS", -1),
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
    session.proj_rows.clear();
    session.probe_raw.clear();
    session.subunit_raw.clear();
    session.request_raw.clear();
    session.request_rows.clear();
    session.stage2_cand.clear();
    session.query_windows.clear();
    session.query_sets.clear();
    session.host_id = 0;
    session.host_valid = false;
}

namespace {

// Stage-2 fallbacks since the process started, by reason (the text before ": ").
long stage2_count_fallback(const std::string & reason) {
    static std::mutex mutex;
    static std::map<std::string, long> counts;
    const std::lock_guard<std::mutex> lock(mutex);
    return ++counts[reason.substr(0, reason.find(": "))];
}

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

// Defined with the session offload helpers below.
bool grow_scoring_session(Qwen35DrafterState & st, Qwen35ScoringSession & session, int capacity);

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
        bool need_request,
        const std::vector<PFlashTokenSpan> & other_windows,
        bool force_fresh,
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
    // Stage 2 (force_fresh): no session resume, every call scores the whole prompt from fresh rows, as
    // PFLASH_DRAFTER_SESSIONS=0 does (the contract's stage-2 inputs are F32 rows only).
    const int limit = force_fresh ? 0 : scoring_session_limit();
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
    // An undersized session stays a candidate (pflash_pick_resume): it is grown below, not discarded.
    std::vector<luce::pflash::PFlashResumeCandidate> candidates;
    for (auto & owned : st.sessions) {
        const Qwen35ScoringSession * session = owned.get();
        luce::pflash::PFlashResumeCandidate c;
        c.usable = limit != 0 && session && session->keys_trained == st.head_loaded;
        if (session) {
            c.ids = &session->ids;
            c.checkpoint = session->checkpoint;
            c.capacity = session->capacity;
            c.rows_cover = [&, session](int r, int shared) {
                // The query window and every other window this call scores that starts before the resume point
                // need their exact stored rows; with one snapshot per session, the call otherwise starts at 0.
                if (r > query_start && !(query_end <= shared && row_floats > 0 &&
                                         stored_window(*session, query_start, query_end, shared))) {
                    return 0;
                }
                for (const auto & span : other_windows) {
                    if (span.begin < 0 || span.end <= span.begin || span.begin >= r) continue;
                    if (!stored_window(*session, span.begin, span.end, shared)) return 0;
                }
                if (!luce::pflash::pflash_scoring_proj_rows_cover(
                        r, session->proj_rows_begin, (int) (session->proj_rows.size() / (size_t) w.n_embd))) {
                    return 0;
                }
                if ((need_probe && (int) session->probe_raw.size() < r) ||
                    (need_subunit && (int) session->subunit_raw.size() < r) ||
                    (need_request && (int) session->request_raw.size() < r)) {
                    return 0;
                }
                return r;
            };
        }
        candidates.push_back(std::move(c));
    }
    const luce::pflash::PFlashResumePick pick = luce::pflash::pflash_pick_resume(candidates, ids, resume_cap);
    Qwen35ScoringSession * best = pick.index >= 0 ? st.sessions[(size_t) pick.index].get() : nullptr;
    resume = pick.resume;
    Qwen35ScoringSession * target = nullptr;
    if (best && pick.needs_grow &&
        !grow_scoring_session(st, *best, luce::pflash::pflash_session_capacity(S, st.session_ctx_limit))) {
        // No room for a second copy: score fresh into this session, reallocated below, as before.
        target = best;
        best = nullptr;
        resume = 0;
    }
    if (best) {
        if (!restore_ssm_state(best->cache, w.backend)) {
            resume = 0;
        } else {
            shared_prefix = pick.shared;
            best->host_valid = false;   // this call extends it: its host copy (if any) goes stale
            return best;
        }
    }

    if (!target) target = best;
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
    target->proj_rows.clear();
    target->probe_raw.clear();
    target->subunit_raw.clear();
    target->request_raw.clear();
    target->request_rows.clear();
    target->stage2_cand.clear();
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
    to.proj_rows = from.proj_rows;
    to.proj_rows_begin = from.proj_rows_begin;
    to.keys_trained = from.keys_trained;
    to.probe_raw = from.probe_raw;
    to.subunit_raw = from.subunit_raw;
    to.request_raw = from.request_raw;
    to.request_rows = from.request_rows;
    to.stage2_cand = from.stage2_cand;
    to.query_windows = from.query_windows;
    to.query_sets = from.query_sets;
    to.last_used = from.last_used;
}

// Grows ``session`` to ``capacity`` tokens in place, keeping what a resume reads: the token-indexed tensors (attention
// K/V along max_ctx, block-15 keys) below its checkpoint, every other tensor (recurrent state and its snapshot) whole,
// and the bookkeeping, cache position and last token. Goes through host memory with the offload helpers. On any
// failure the session is left as it was and false is returned.
bool grow_scoring_session(Qwen35DrafterState & st, Qwen35ScoringSession & session, int capacity) {
    const auto t0 = std::chrono::steady_clock::now();
    const int old_capacity = session.capacity;
    Qwen35ScoringSession grown;
    if (!allocate_scoring_session(st.weights, capacity, grown)) {
        std::fprintf(stderr, "[qwen35-sessions] session not grown (%d -> %d): allocation failed\n", old_capacity,
                     capacity);
        std::fflush(stderr);
        return false;
    }
    const auto old_tensors = qwen35_session_tensors(session);
    luce::pflash::PFlashHostCopy host;
    std::string error;
    if (grown.cache.max_ctx < session.cache.max_ctx) {
        error = "cache size differs";
    } else if (luce::pflash::pflash_copy_to_host(old_tensors, qwen35_session_token_dims(session, old_tensors),
                                                 session.checkpoint, nullptr, host, error)) {
        luce::pflash::pflash_copy_from_host(host, qwen35_session_tensors(grown), error);
    }
    if (!error.empty()) {
        free_qwen35_scoring_session(grown);
        std::fprintf(stderr, "[qwen35-sessions] session not grown (%d -> %d): %s\n", old_capacity, capacity,
                     error.c_str());
        std::fflush(stderr);
        return false;
    }
    ggml_backend_synchronize(st.weights.backend);
    grown.cache.cur_pos = session.cache.cur_pos;
    grown.cache.last_tok = session.cache.last_tok;
    copy_session_bookkeeping(session, grown);
    grown.host_id = session.host_id;
    grown.host_valid = session.host_valid;
    const int checkpoint = session.checkpoint;
    std::swap(session, grown);          // ``grown`` now owns the old tensors
    free_qwen35_scoring_session(grown);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "[qwen35-sessions] grew session: %d -> %d, kept checkpoint %d (%.0f ms)\n", old_capacity,
                 capacity, checkpoint, ms);
    std::fflush(stderr);
    return true;
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

ggml_backend_buffer_type_t pinned_host_type(ggml_backend_t backend) {
    // Plain host memory unless PFLASH_SESSION_OFFLOAD_PINNED=1.
    if (!luce::pflash::pflash_session_offload_pinned_enabled()) return nullptr;
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    return dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
}

} // namespace

std::string qwen35_session_identity(const DrafterContext & ctx, const Qwen35DrafterState & st) {
    const auto env_file = [](const char * name) {
        const char * value = std::getenv(name);
        return luce::pflash::pflash_file_identity(value ? value : "");
    };
    const TargetWeights & w = st.weights;
    std::ostringstream id;
    // The store lives in this process's memory, so the build cannot change under it; the stamp names it anyway.
    id << "format=2\n"
       << "build=" << __DATE__ << ' ' << __TIME__ << '\n'
       << "drafter=" << luce::pflash::pflash_file_identity(st.gguf_path) << '\n'
       << "drafter_sha256=" << st.gguf_sha256 << '\n'
       << "gpu=" << ctx.gpu << '\n'
       << "dims=" << w.n_layer << ',' << w.n_embd << ',' << w.n_head << ',' << w.n_head_kv << ','
       << w.n_embd_head_k << '\n'
       << "head=" << env_file("PFLASH_SCORING_HEAD_GGUF") << '|' << st.head_loaded << '\n'
       << "segment_probe=" << env_file("PFLASH_SEGMENT_PROBE_GGUF") << '|' << st.probe_loaded << '|'
       << (st.probe_sub_fc2_w != nullptr) << '\n'
       << "request_probe=" << env_file("PFLASH_REQUEST_PROBE_GGUF") << '|' << st.request_loaded << '|'
       << st.request_stub << '|' << st.request_checkpoint_sha256 << '\n'
       << "stage2=" << env_file(luce::pflash::kStage2GgufEnv) << '|' << st.stage2_loaded << '\n'
       << "sessions=" << scoring_session_limit() << '\n'
       << "grid=" << luce::pflash::kScoringResumeGrid << ',' << luce::pflash::kScoringUbatch << ','
       << luce::pflash::kScoringCutBack << '\n';
    return id.str();
}

void qwen35_offload_sessions(DrafterContext & ctx) {
    auto * st = static_cast<Qwen35DrafterState *>(ctx.state);
    if (!st) return;
    const int limit = st->stage2_loaded ? 0 : scoring_session_limit();
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

    // Released entries give their (pinned) storage to the copies below.
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
    ggml_backend_buffer_type_t pinned = pinned_host_type(st->weights.backend);
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
                                               session.checkpoint, pinned, host.copy, error)) {
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
    std::fprintf(stderr, "[qwen35-sessions] offload: copied %zu session(s) (%.1f MiB%s), kept %zu unchanged host "
                 "cop%s, released %zu; host RAM held %.1f MiB in %zu session(s) (%.0f ms)\n",
                 copied, copied_bytes / (1024.0 * 1024.0), pinned ? ", pinned" : "", n_kept,
                 n_kept == 1 ? "y" : "ies", n_dropped, host_store_bytes(s) / (1024.0 * 1024.0), s.sessions.size(), ms);
    std::fflush(stderr);
    if (s.sessions.empty()) ctx.host_sessions.reset();
}

void qwen35_restore_sessions(DrafterContext & ctx, Qwen35DrafterState & st, const std::vector<int32_t> & ids) {
    if (!ctx.host_sessions || ctx.host_sessions->sessions.empty()) return;
    const int limit = st.stage2_loaded ? 0 : scoring_session_limit();
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
    std::vector<float> * token_mass_out,
    std::vector<PFlashTokenSpan> * segments_out,
    bool * density_out,
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
    // PFLASH_SELECT_QUERY_PROBE=1: the request probe's own trunk runs on
    // the same tap whatever the segmentation mode.
    const bool request_probe = experiment.query_probe;
    if (request_probe && !st.request_loaded) {
        set_last_error("PFLASH_SELECT_QUERY_PROBE=1: the drafter has no request probe");
        return {};
    }
    // cand_v5 and stage 2 (STAGE2-CONTRACT.md): stage 2 runs only with a cand_v5 probe and its own GGUF. The
    // latest user message's span is known before the forward: its x̂ sums ride in the key graph while it is fresh.
    const bool request_v5 = request_probe && st.request_v5;
    const bool stage2_on = request_v5 && st.stage2_loaded;
    PFlashTokenSpan stage2_latest_span{-1, -1};
    if (stage2_on && experiment.request_messages_all_error.empty()) {
        const int latest = luce::pflash::pflash_request_latest_user(experiment.request_messages_all);
        if (latest >= 0) stage2_latest_span = experiment.request_messages_all[(size_t) latest].span;
    }
    // Query step 1 (pflash_query_candidates.h): relative candidates (qcand), the verbatim pin (qpin) and the
    // dev-only replay grid (qgrid). Fail closed on any missing input; never today's windows while qcand is on.
    const luce::pflash::PFlashQueryConfig & qcfg = experiment.query;
    const bool qcand = request_probe && qcfg.candidates > 0;
    const bool qpin = request_probe && qcfg.pin_tokens > 0;
    const char * qgrid_path = request_probe ? std::getenv("PFLASH_QUERY_GRID") : nullptr;
    const bool qgrid = qgrid_path && *qgrid_path;
    const bool qstep = qcand || qpin || qgrid;
    if (qstep && st.stage2_loaded) {
        set_last_error("PFLASH_QUERY_CANDIDATES / PFLASH_QUERY_PIN / PFLASH_QUERY_GRID do not combine with request "
                       "stage 2 (PFLASH_REQUEST_STAGE2_GGUF)");
        return {};
    }
    if (qstep && !use_probe) {
        set_last_error("PFLASH_QUERY_CANDIDATES / PFLASH_QUERY_PIN / PFLASH_QUERY_GRID need the segment probe");
        return {};
    }
    if (qstep && (!experiment.request_messages_all_error.empty() || experiment.request_messages_all.empty())) {
        set_last_error("query step: no message mapping from the caller" +
                       (experiment.request_messages_all_error.empty()
                            ? std::string() : ": " + experiment.request_messages_all_error));
        return {};
    }
    std::vector<luce::pflash::PFlashGridConfig> qgrid_configs;
    if (qgrid) {
        std::ifstream in(qgrid_path);
        std::stringstream text;
        text << in.rdbuf();
        std::string grid_error;
        if (!in || !luce::pflash::pflash_query_grid_parse(text.str(), qgrid_configs, grid_error)) {
            set_last_error("PFLASH_QUERY_GRID: " + (grid_error.empty() ? std::string("unreadable file") : grid_error));
            return {};
        }
        const char * out_path = std::getenv("PFLASH_QUERY_GRID_OUT");
        if (!out_path || !*out_path) {
            set_last_error("PFLASH_QUERY_GRID needs PFLASH_QUERY_GRID_OUT");
            return {};
        }
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
    for (const auto & span : experiment.extra_queries) {
        if (span.end <= query_start && span.end > span.begin) other_windows.push_back(span);
    }
    // Stage 2 alone decides look-back: the fixed earlier-question windows (PFLASH_CHAT_HISTORY_QUERIES) are
    // dropped while it is loaded, also from today's windows on a fallback.
    const bool drop_history = st.stage2_loaded;
    for (const auto & span : experiment.history_queries) {
        if (drop_history) break;
        if (span.end <= query_start && span.end > span.begin) other_windows.push_back(span);
    }
    // The query step resumes like every other request (pflash_scoring_resume.h: exact by construction); its query
    // rows come from this call's activations or the session's exact F32 sets, else the request is redone from 0.
    const bool sessions_on = !st.stage2_loaded && scoring_session_limit() > 0;
    Qwen35ScoringSession * session = acquire_scoring_session(
        st, ids, query_start, query_end, use_probe,
        use_probe && st.probe_sub_fc2_w != nullptr, request_probe, other_windows,
        /*force_fresh=*/st.stage2_loaded, resume_cap, resume, shared_prefix, scratch);
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
    // recurrent state is snapshotted at the last 1024-grid point at or before
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
    ggml_tensor * act_out = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, std::max(1, n_new));
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

    {
        const int batch = 2048;
        std::vector<float> emb((size_t)hidden * batch);
        for (int i = 0; i < n_new; i += batch) {
            const int n = std::min(batch, n_new - i);
            if (!w.embedder.embed(ids.data() + resume + i, n, emb.data())) {
                return fail("qwen35 drafter embedding failed");
            }
            ggml_backend_tensor_set(act_in, emb.data(), (size_t)i * act_in->nb[1],
                                    (size_t)hidden * n * sizeof(float));
        }
    }

    // Blocks 0..14 over the new tokens only, layer by layer. Each
    // DeltaNet layer's recurrent state is copied once it reaches the
    // checkpoint.
    const auto snapshot_layer = [&](int il) {
        int dn = 0;
        for (int l = 0; l < il; ++l) {
            if (((l + 1) % w.full_attention_interval) != 0) ++dn;
        }
        if (dn < (int) cache.ssm_state.size() && cache.ssm_state[(size_t) dn] &&
            cache.ssm_state_snap[(size_t) dn]) {
            ggml_backend_tensor_copy(cache.ssm_state[(size_t) dn],
                                     cache.ssm_state_snap[(size_t) dn]);
            ggml_backend_tensor_copy(cache.conv_state[(size_t) dn],
                                     cache.conv_state_snap[(size_t) dn]);
        }
    };
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
    const int ubatch = luce::pflash::kScoringUbatch;
    // Ubatches on the 1024 grid from the (grid) resume point, cut at S - 64;
    // the checkpoint is a grid point, so the state there can be copied.
    const auto ubatch_at = [&](int start) {
        return std::min(ubatch, (start < cut ? cut : S) - start);
    };
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
        ggml_tensor * inp = ggml_view_2d(ctx, act_in, hidden, n, act_in->nb[1],
                                         (size_t)(start - resume) * act_in->nb[1]);
        pos = nullptr;
        ggml_tensor * mask = nullptr;
        if (((il + 1) % w.full_attention_interval) == 0) {
            pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n);
            ggml_set_input(pos);
            mask = build_qwen35_causal_mask(ctx, gf, start, n);
        }
        ggml_tensor * out = build_qwen35_layer(ctx, gf, w, cache, il, inp, pos, mask,
                                               start, n, false, 0);
        ggml_tensor * dst = ggml_view_2d(ctx, act_out, hidden, n, act_out->nb[1],
                                         (size_t)(start - resume) * act_out->nb[1]);
        if (ggml_nelements(out) != ggml_nelements(dst)) {
            ggml_free(ctx);
            set_last_error("qwen35 layer output shape mismatch");
            return nullptr;
        }
        ggml_build_forward_expand(gf, ggml_cpy(ctx, out, dst));
        return ctx;
    };
    const auto fail_graph = [&]() -> std::vector<int32_t> {
        ggml_gallocr_free(alloc);
        cleanup();
        session->ids.clear();
        return {};
    };
    // Size the compute buffer once, for a full-attention layer over the
    // ubatch with the largest mask: allocating per ubatch would regrow it
    // with every ubatch of the first attention layer.
    if (n_new > 0 && w.full_attention_interval <= kQwen35HeadBlock) {
        int worst = resume;
        for (int start = resume; start < S; start += ubatch_at(start)) {
            if (causal_mask_bytes(start, ubatch_at(start)) >=
                causal_mask_bytes(worst, ubatch_at(worst))) worst = start;
        }
        ggml_cgraph * gf = nullptr;
        ggml_tensor * pos = nullptr;
        ggml_context * ctx = layer_graph(w.full_attention_interval - 1, worst,
                                         ubatch_at(worst), gf, pos);
        if (!ctx) return fail_graph();
        const bool reserved = ggml_gallocr_reserve(alloc, gf);
        ggml_free(ctx);
        if (!reserved) {
            ggml_gallocr_free(alloc);
            return fail("qwen35 drafter graph allocation failed");
        }
    }
    for (int il = 0; il < kQwen35HeadBlock; ++il) {
        const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);
        if (!is_attn && checkpoint == resume) snapshot_layer(il);
        for (int start = resume; start < S;) {
            const int n = ubatch_at(start);
            ggml_cgraph * gf = nullptr;
            ggml_tensor * pos = nullptr;
            ggml_context * ctx = layer_graph(il, start, n, gf, pos);
            if (!ctx) return fail_graph();
            if (!ggml_gallocr_alloc_graph(alloc, gf)) {
                ggml_free(ctx); ggml_gallocr_free(alloc);
                return fail("qwen35 drafter graph allocation failed");
            }
            if (is_attn) {
                std::vector<int32_t> p4((size_t)4 * n, 0);
                for (int i = 0; i < n; ++i) {
                    const int p = start + i;
                    p4[(size_t)0 * n + i] = p;
                    p4[(size_t)1 * n + i] = p;
                    p4[(size_t)2 * n + i] = p;
                }
                ggml_backend_tensor_set(pos, p4.data(), 0, p4.size() * sizeof(int32_t));
            }
            const auto status = ggml_backend_graph_compute(w.backend, gf);
            ggml_free(ctx);
            if (status != GGML_STATUS_SUCCESS) {
                ggml_gallocr_free(alloc);
                return fail("qwen35 drafter graph compute failed");
            }
            start += n;
            if (!is_attn && start == checkpoint && checkpoint > resume) {
                snapshot_layer(il);
            }
        }
        std::swap(act_in, act_out);
    }
    ggml_gallocr_free(alloc);
    auto t1 = std::chrono::steady_clock::now();

    // Exact block-14 rows [proj_begin(checkpoint), checkpoint) for the next resume: this call's own activations from
    // ``resume`` on, the session's earlier rows ahead of it (they cover [proj_begin(resume), resume)).
    const int keep_begin = luce::pflash::pflash_scoring_proj_begin(checkpoint);
    std::vector<float> keep_proj((size_t) hidden * (size_t) std::max(0, checkpoint - keep_begin));
    if (!keep_proj.empty()) {
        const int from_session = std::max(0, resume - keep_begin);
        if (from_session > 0) {
            std::copy_n(session->proj_rows.data() + (size_t) (keep_begin - session->proj_rows_begin) * hidden,
                        (size_t) from_session * hidden, keep_proj.data());
        }
        ggml_backend_tensor_get(act_in, keep_proj.data() + (size_t) from_session * hidden,
                                (size_t) (keep_begin + from_session - resume) * act_in->nb[1],
                                (size_t) (checkpoint - keep_begin - from_session) * act_in->nb[1]);
    }

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
        // The user's prose around pasted material (query head), at the
        // query's own weight.
        for (const auto & span : experiment.extra_queries) {
            if (span.end > query_start || span.end - span.begin < 1) continue;
            ScoredWindow extra;
            extra.begin = span.begin;
            extra.end = span.end;
            if (rows_for(span.begin, span.end, extra.rows)) {
                windows.push_back(std::move(extra));
            }
        }
        double weight = 1.0;
        for (const auto & span : experiment.history_queries) {
            if (drop_history) break;
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
        if (drop_history || span.end > query_start || span.end - span.begin < 1) continue;
        ScoredWindow keep;
        keep.begin = span.begin;
        keep.end = span.end;
        if (rows_for(span.begin, span.end, keep.rows)) keep_rows.push_back(std::move(keep));
    }

    ggml_tensor * wk_src = st.head_loaded ? st.head_wk : L.wk;
    session->keys_trained = st.head_loaded;
    // The projection chunks are the fresh call's: a resume inside one re-projects it whole from ``proj_begin``.
    const int proj_begin = luce::pflash::pflash_scoring_proj_begin(resume);
    const int n_proj = S - proj_begin;
    session->probe_raw.resize(use_probe ? (size_t) proj_begin : 0);
    session->subunit_raw.resize(
        use_probe && st.probe_sub_fc2_w ? (size_t) proj_begin : 0);
    session->request_raw.resize(request_probe ? (size_t) proj_begin : 0);
    session->stage2_cand.resize(stage2_on ? (size_t) proj_begin * luce::pflash::kStage2Embed : 0);
    // Stage 2: the fresh latest user message's x̂ column sums, one per key chunk it overlaps ([parts][hidden]).
    const bool stage2_latest_fresh = stage2_on && stage2_latest_span.end > stage2_latest_span.begin &&
        stage2_latest_span.begin >= resume;
    std::vector<float> stage2_msum;

    // Keys (and probe logits) of the new tokens, into the session. Keys are
    // projected in chunks so no intermediate tensor puts the sequence length
    // into a HIP grid y/z dimension (65,535 limit). The chunks start at the
    // chunk start before the resume point, so they fall where a fresh call's do.
    const int key_chunk = luce::pflash::kScoringProjChunk;
    if (n_proj > 0) {
        ggml_init_params nip{};
        nip.mem_size = (size_t)4 * ggml_tensor_overhead() + 4096;
        nip.no_alloc = true;
        ggml_context * nctx = ggml_init(nip);
        ggml_tensor * probe_new = use_probe ? ggml_new_tensor_1d(nctx, GGML_TYPE_F32, n_proj) : nullptr;
        ggml_tensor * subunit_new = use_probe && st.probe_sub_fc2_w
            ? ggml_new_tensor_1d(nctx, GGML_TYPE_F32, n_proj) : nullptr;
        ggml_tensor * request_new = request_probe
            ? ggml_new_tensor_1d(nctx, GGML_TYPE_F32, n_proj) : nullptr;
        ggml_backend_buffer_t nbuf = use_probe || request_probe
            ? ggml_backend_alloc_ctx_tensors(nctx, w.backend) : nullptr;
        if ((use_probe || request_probe) && !nbuf) {
            ggml_free(nctx);
            return fail("qwen35 probe buffer allocation failed");
        }
        // The chunk that holds the resume point: the session's rows ahead of it, then this call's activations.
        ggml_context * hctx = nullptr;
        ggml_backend_buffer_t hbuf = nullptr;
        ggml_tensor * head_x = nullptr;
        if (proj_begin < resume) {
            const int n_head = std::min(key_chunk, n_proj);
            ggml_init_params hip{};
            hip.mem_size = (size_t)2 * ggml_tensor_overhead() + 4096;
            hip.no_alloc = true;
            hctx = ggml_init(hip);
            head_x = hctx ? ggml_new_tensor_2d(hctx, GGML_TYPE_F32, hidden, n_head) : nullptr;
            hbuf = head_x ? ggml_backend_alloc_ctx_tensors(hctx, w.backend) : nullptr;
            if (!hbuf || session->proj_rows_begin != proj_begin ||
                session->proj_rows.size() < (size_t) hidden * (size_t) (resume - proj_begin)) {
                if (hbuf) ggml_backend_buffer_free(hbuf);
                if (hctx) ggml_free(hctx);
                if (nbuf) ggml_backend_buffer_free(nbuf);
                ggml_free(nctx);
                session->ids.clear();
                return fail("qwen35 scorer: chunk rows ahead of the resume point unavailable");
            }
            std::vector<float> head_host((size_t) hidden * (size_t) n_head);
            const size_t kept = (size_t) hidden * (size_t) (resume - proj_begin);
            std::copy_n(session->proj_rows.data(), kept, head_host.data());
            ggml_backend_tensor_get(act_in, head_host.data() + kept, 0,
                                    head_host.size() * sizeof(float) - kept * sizeof(float));
            ggml_backend_tensor_set(head_x, head_host.data(), 0, head_host.size() * sizeof(float));
        }
        const int n_key_chunks = (n_proj + key_chunk - 1) / key_chunk;
        // Stage 2: c(t) of the new tokens, and the fresh latest message's x̂ column sums.
        ggml_context * s2ctx = nullptr;
        ggml_backend_buffer_t s2buf = nullptr;
        ggml_tensor * cand_new = nullptr;
        ggml_tensor * msum_new = nullptr;
        int msum_parts = 0;
        if (stage2_on) {
            for (int b = 0; b < n_proj && stage2_latest_fresh; b += key_chunk) {
                const int n = std::min(key_chunk, n_proj - b);
                if (std::max(stage2_latest_span.begin, proj_begin + b) <
                    std::min(stage2_latest_span.end, proj_begin + b + n)) ++msum_parts;
            }
            ggml_init_params sip2{};
            sip2.mem_size = (size_t)4 * ggml_tensor_overhead() + 4096;
            sip2.no_alloc = true;
            s2ctx = ggml_init(sip2);
            if (s2ctx) {
                cand_new = ggml_new_tensor_2d(s2ctx, GGML_TYPE_F32, luce::pflash::kStage2Embed, n_proj);
                msum_new = ggml_new_tensor_2d(s2ctx, GGML_TYPE_F32, hidden, std::max(1, msum_parts));
                s2buf = ggml_backend_alloc_ctx_tensors(s2ctx, w.backend);
            }
            if (!s2buf) {
                if (s2ctx) ggml_free(s2ctx);
                if (nbuf) ggml_backend_buffer_free(nbuf);
                ggml_free(nctx);
                return fail("qwen35 stage-2 buffer allocation failed");
            }
            if (msum_parts == 0) msum_new = nullptr;
        }
        ggml_init_params kip{};
        kip.mem_size = ggml_tensor_overhead() *
                           (size_t)(64 + (request_probe ? 40 : 24) * n_key_chunks +
                                    (stage2_on ? 24 * n_key_chunks : 0)) +
                       ggml_graph_overhead_custom(4096, false) + 64 * 1024;
        kip.no_alloc = true;
        ggml_context * kctx = ggml_init(kip);
        ggml_cgraph * kgf = ggml_new_graph_custom(kctx, 4096, false);
        for (int b = 0; b < n_proj; b += key_chunk) {
            const int n = std::min(key_chunk, n_proj - b);
            ggml_tensor * x_c = head_x && b == 0
                ? ggml_view_2d(kctx, head_x, hidden, n, head_x->nb[1], 0)
                : ggml_view_2d(kctx, act_in, hidden, n, act_in->nb[1],
                               (size_t) (proj_begin + b - resume) * act_in->nb[1]);
            ggml_tensor * x_norm = ggml_mul(kctx, ggml_rms_norm(kctx, x_c, w.rms_eps), L.attn_norm);
            ggml_tensor * K = ggml_reshape_3d(kctx, ggml_mul_mat(kctx, wk_src, x_norm), D, Hk, n);
            K = ggml_mul(kctx, ggml_rms_norm(kctx, K, w.rms_eps), L.k_norm);
            ggml_tensor * k_dst = ggml_view_3d(kctx, session->keys, D, Hk, n,
                                               session->keys->nb[1], session->keys->nb[2],
                                               (size_t)(proj_begin + b) * session->keys->nb[2]);
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
            if (cand_new) {
                // Stage 2: c(t) = cand(x̂_t), kept per token while the session covers it.
                ggml_tensor * c = luce::pflash::pflash_stage2_build_cand(kctx, st.stage2_dev, x_norm);
                ggml_tensor * c_dst = ggml_view_2d(kctx, cand_new, luce::pflash::kStage2Embed, n,
                                                   cand_new->nb[1], (size_t) b * cand_new->nb[1]);
                ggml_build_forward_expand(kgf, ggml_cpy(kctx, c, c_dst));
            }
            if (msum_new) {
                const int lo = std::max(stage2_latest_span.begin, proj_begin + b);
                const int hi = std::min(stage2_latest_span.end, proj_begin + b + n);
                if (lo < hi) {
                    ggml_tensor * col = luce::pflash::pflash_stage2_build_column_sum(
                        kctx, x_norm, lo - (proj_begin + b), hi - (proj_begin + b));
                    const int part = (int) (stage2_msum.size() / (size_t) hidden);
                    stage2_msum.resize(stage2_msum.size() + (size_t) hidden);
                    ggml_tensor * m_dst = ggml_view_1d(kctx, msum_new, hidden, (size_t) part * msum_new->nb[1]);
                    ggml_build_forward_expand(kgf, ggml_cpy(kctx, col, m_dst));
                }
            }
        }
        ggml_gallocr_t kalloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
        const bool key_ok = ggml_gallocr_alloc_graph(kalloc, kgf) &&
            ggml_backend_graph_compute(w.backend, kgf) == GGML_STATUS_SUCCESS;
        ggml_gallocr_free(kalloc);
        ggml_free(kctx);
        if (key_ok && use_probe) {
            session->probe_raw.resize((size_t) S);
            ggml_backend_tensor_get(probe_new, session->probe_raw.data() + proj_begin, 0,
                                    (size_t) n_proj * sizeof(float));
            if (subunit_new) {
                session->subunit_raw.resize((size_t) S);
                ggml_backend_tensor_get(subunit_new, session->subunit_raw.data() + proj_begin, 0,
                                        (size_t) n_proj * sizeof(float));
            }
        }
        if (key_ok && request_new) {
            session->request_raw.resize((size_t) S);
            ggml_backend_tensor_get(request_new, session->request_raw.data() + proj_begin, 0,
                                    (size_t) n_proj * sizeof(float));
        }
        if (key_ok && cand_new) {
            session->stage2_cand.resize((size_t) S * luce::pflash::kStage2Embed);
            ggml_backend_tensor_get(cand_new,
                session->stage2_cand.data() + (size_t) proj_begin * luce::pflash::kStage2Embed, 0,
                (size_t) n_proj * luce::pflash::kStage2Embed * sizeof(float));
        }
        if (key_ok && msum_new && !stage2_msum.empty()) {
            ggml_backend_tensor_get(msum_new, stage2_msum.data(), 0, stage2_msum.size() * sizeof(float));
        }
        if (hbuf) ggml_backend_buffer_free(hbuf);
        if (hctx) ggml_free(hctx);
        if (s2buf) ggml_backend_buffer_free(s2buf);
        if (s2ctx) ggml_free(s2ctx);
        if (nbuf) ggml_backend_buffer_free(nbuf);
        ggml_free(nctx);
        if (!key_ok) return fail("qwen35 key graph compute failed");
    }

    const std::vector<float> & probe_raw = session->probe_raw;
    const std::vector<float> & subunit_raw = session->subunit_raw;

    // Candidates: probe segments cut at the pinned spans' edges, or fixed
    // chunks when the probe is off or finds too few boundaries.
    const auto segment = [&](const std::vector<PFlashTokenSpan> & pins,
                             std::vector<PFlashTokenSpan> & segments, bool & density) {
        segments.clear();
        density = experiment.candidate_score == luce::pflash::PFlashCandidateScore::Density;
        if (!use_probe) return;
        // Tap-count smoothing over the raw logits (torch Conv1d, symmetric
        // padding) plus the residual logit, then sigmoid: the boundary score
        // per token.
        const auto smooth = [&](const std::vector<float> & raw,
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
        std::vector<float> boundary;
        smooth(probe_raw, st.probe_conv_w, st.probe_conv_b, boundary);
        // Sub-unit scores feed only the oversize interior argmax below.
        std::vector<float> split_scores;
        if (!subunit_raw.empty()) {
            smooth(subunit_raw, st.probe_sub_conv_w, st.probe_sub_conv_b, split_scores);
        }
        const int query_end = score_query_end < 0 ? S : score_query_end;
        const int query_begin = query_end - std::min(n_lookahead, query_end);
        std::vector<int> forced{query_begin, query_end};
        for (const auto & span : pins) {
            forced.push_back(span.begin);
            forced.push_back(span.end);
        }
        int boundaries_in_context = 0;
        for (int t = 1; t < S; ++t) {
            if (t >= query_begin &&
                (t < query_end || !experiment.query_suffix_candidates)) {
                continue;
            }
            if (boundary[(size_t) t] > st.probe_threshold) ++boundaries_in_context;
        }
        const bool forced_probe =
            experiment.segmentation == luce::pflash::PFlashSegmentation::Probe;
        if (boundaries_in_context >= 4 || forced_probe) {
            if (st.probe_line_aligned && st.probe_vocab) {
                std::vector<std::string> texts((size_t) S);
                for (int t = 0; t < S; ++t) texts[(size_t) t] = st.probe_vocab->token_text(ids[(size_t) t]);
                const auto lines = luce::pflash::pflash_line_flags(texts);
                segments = luce::pflash::pflash_probe_segments(
                    boundary, S, st.probe_threshold, st.probe_min_segment,
                    st.probe_max_segment, forced, split_scores, &lines);
            } else {
                segments = luce::pflash::pflash_probe_segments(
                    boundary, S, st.probe_threshold, st.probe_min_segment,
                    st.probe_max_segment, forced, split_scores);
            }
        }
        if (segments.empty()) {
            std::fprintf(stderr,
                "[qwen35-segment-probe] %d boundaries in the context, "
                "falling back to fixed %d-token chunks\n",
                boundaries_in_context, experiment.chunk_size);
        } else {
            if (experiment.candidate_score == luce::pflash::PFlashCandidateScore::Auto) {
                density = true;
            }
            std::fprintf(stderr,
                "[qwen35-segment-probe] %d boundaries in the context -> %zu segments "
                "(threshold %.2f, %d-%d tokens), score=%s\n",
                boundaries_in_context, segments.size(), st.probe_threshold,
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

    // Request probe (PFLASH_SELECT_QUERY_PROBE=1): the query set from the
    // request logits of the human messages, its block-14 rows, and the rows
    // later turns will need -- all read here, while this call's activations
    // (act_in) still exist. Rows before ``resume`` come from the session's
    // store; a query row found nowhere falls the whole request back to
    // today's windows.
    luce::pflash::PFlashRequestProbeTrace request_trace;
    std::vector<float> request_query_rows;      // [hidden, query tokens]
    // Query step 1 state: units, the served candidate set and pin, the grid's candidate sets and oracle, and the
    // block-14 rows of every query token set any of them scores (keyed by the ascending token list).
    luce::pflash::PFlashQueryTrace query_trace;
    luce::pflash::PFlashQueryUnits qunits;
    std::vector<luce::pflash::PFlashQueryCandidateSet> qgrid_sets;   // per grid config (empty unless cand)
    luce::pflash::PFlashQueryOracle qoracle;
    std::string qoracle_error = "no_oracle_dir";
    std::string qfingerprint;
    std::map<std::vector<int>, std::vector<float>> qrows;
    int qrows_new = 0;
    // Stage 2's pass-1 assembly (STAGE2-CONTRACT.md section 5), when it ran.
    luce::pflash::PFlashStage2Assembly stage2_asm;
    bool stage2_assembled = false;
    if (request_probe) {
        request_trace.tau = st.request_tau;
        request_trace.stub = st.request_stub;
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
        // Stage 2 pass 1 on the cand_v5 set (contract section 4): seed, m_all / m_s1, message head, look-back
        // attention over E, the token model, then the assembly (seed > promoted > post-latest, widening).
        const auto stage2_pass1 = [&]() -> bool {
            auto & s2t = request_trace.stage2;
            const auto & msgs = experiment.request_messages_all;
            const auto started = std::chrono::steady_clock::now();
            const auto finish = [&]() {
                s2t.pass1_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - started).count();
            };
            luce::pflash::PFlashStage2Layout layout;
            std::string layout_error;
            if (!luce::pflash::pflash_stage2_layout(msgs, S, layout, layout_error)) {
                s2t.reason = "message_mapping";
                request_trace.fallback = "message_mapping: " + layout_error;
                finish();
                return true;
            }
            const std::vector<float> & s1 = session->request_raw;
            if (layout.latest < 0) {
                s2t.reason = "no_user_message";     // the cand_v5 set alone
                finish();
                return true;
            }
            const PFlashTokenSpan latest = layout.latest_span;
            bool seed_capped = false;
            int reserve = 0;
            std::vector<int> post_latest;
            // Contract v3: R slots reserved for the post-latest cues; the seed (m_s1, the look-back queries) is
            // capped at 1024 - R.
            const std::vector<int> seed = luce::pflash::pflash_request_capped_seed(
                s1, st.request_tau, msgs, luce::pflash::kRequestQueryCap, luce::pflash::kRequestPostReserve, reserve,
                post_latest, &seed_capped);
            s2t.post_reserve = reserve;
            luce::pflash::PFlashStage2Latest lat;
            lat.m_all.assign((size_t) hidden, 0.0);
            std::vector<float> seed_rows((size_t) hidden * seed.size());
            if (latest.end <= latest.begin || latest.begin >= resume) {
                // Fresh: m_all from the key graph's column sums, the seed rows from this call's activations.
                const size_t parts = stage2_msum.size() / (size_t) hidden;
                if (latest.end > latest.begin && parts == 0) {
                    s2t.reason = "rows_missing";
                    request_trace.fallback = "rows_missing";
                    finish();
                    return true;
                }
                for (size_t p = 0; p < parts; ++p) {
                    for (int k = 0; k < hidden; ++k) lat.m_all[(size_t) k] += (double) stage2_msum[p * hidden + (size_t) k];
                }
                if (latest.end > latest.begin) {
                    for (auto & v : lat.m_all) v /= (double) (latest.end - latest.begin);
                }
                for (size_t q = 0; q < seed.size(); ++q) {
                    ggml_backend_tensor_get(act_in, seed_rows.data() + q * (size_t) hidden,
                                            (size_t) (seed[q] - resume) * act_in->nb[1],
                                            (size_t) hidden * sizeof(float));
                }
            } else {
                // Unreachable while stage 2 forces fresh scoring (resume 0); fail closed, traced and counted.
                s2t.reason = "rows_missing";
                request_trace.fallback = "rows_missing";
                finish();
                return true;
            }
            if (seed.empty()) {
                lat.m_s1 = lat.m_all;
            } else {
                lat.m_s1.assign((size_t) hidden, 0.0);
                std::vector<double> xh((size_t) hidden);
                for (size_t q = 0; q < seed.size(); ++q) {
                    luce::pflash::pflash_stage2_xhat(seed_rows.data() + q * (size_t) hidden, st.stage2_w_in.data(),
                                                     hidden, w.rms_eps, xh.data());
                    for (int k = 0; k < hidden; ++k) lat.m_s1[(size_t) k] += xh[(size_t) k];
                }
                for (auto & v : lat.m_s1) v /= (double) seed.size();
            }
            luce::pflash::pflash_stage2_message_head(st.stage2, lat);
            const char * dump = std::getenv(luce::pflash::kStage2FeaturesEnv);
            const bool dumping = dump && *dump;
            const size_t N = layout.earlier.size();
            const bool look = lat.lookback_logit >= st.stage2.tau_lb && N > 0;   // NaN -> false
            std::vector<float> a, x, s2;
            if (N > 0 && (look || dumping)) {
                luce::pflash::PFlashStage2AttentionInputs in;
                in.backend = w.backend;
                in.keys = session->keys;
                in.wq = st.head_wq;
                in.w_in = L.attn_norm;
                in.w_qn = L.q_norm;
                in.rms_eps = w.rms_eps;
                std::string error;
                if (!luce::pflash::pflash_stage2_attention_gpu(in, layout.earlier, seed_rows.data(),
                                                                (int) seed.size(), a, s2t.attention_tiles, error)) {
                    set_last_error(error);
                    return false;
                }
                std::vector<float> cand(N * (size_t) luce::pflash::kStage2Embed);
                for (size_t i = 0; i < N; ++i) {
                    std::copy_n(session->stage2_cand.data() +
                                    (size_t) layout.earlier[i] * luce::pflash::kStage2Embed,
                                (size_t) luce::pflash::kStage2Embed, cand.data() + i * luce::pflash::kStage2Embed);
                }
                luce::pflash::pflash_stage2_features(layout, a, s1, cand, lat, x);
                if (!luce::pflash::pflash_stage2_mlp_gpu(w.backend, st.stage2_dev, st.stage2.norm_eps, x, (int) N,
                                                         s2, error)) {
                    set_last_error(error);
                    return false;
                }
                s2t.s2_computed = true;
            }
            const int budget = (int) std::floor((double) S * (double) keep_ratio);
            stage2_asm = luce::pflash::pflash_stage2_assemble(
                msgs, s1, luce::pflash::pflash_stage2_full_s2(layout, s2, S), lat.lookback_logit,
                lat.vague_logit, st.request_tau, st.stage2.tau2, st.stage2.tau_lb, st.stage2.tau_fb, budget, S);
            stage2_assembled = true;
            if (stage2_asm.seed != seed || stage2_asm.reserve != reserve) {
                set_last_error("stage 2: the assembly's seed or reserve differs from pass 1's");
                return false;
            }
            if (dumping && !luce::pflash::pflash_stage2_dump_features(dump, S, msgs, layout, seed, lat, a, x, s2)) {
                set_last_error(std::string(luce::pflash::kStage2FeaturesEnv) + ": feature dump write failed");
                return false;
            }
            // The assembled query replaces the cand_v5 set's tokens; the set keeps its runs for the row store.
            request_trace.set.tokens = stage2_asm.query;
            request_trace.set.latest_tokens = (int) stage2_asm.seed.size();
            request_trace.set.turn_tokens = (int) (stage2_asm.promoted.size() + stage2_asm.post.size());
            request_trace.set.capped = stage2_asm.seed_capped || stage2_asm.promoted_capped;
            request_trace.set.fallback = stage2_asm.reason;
            request_trace.fallback = stage2_asm.reason;
            s2t.active = true;
            s2t.latest_message = layout.latest;
            s2t.latest_span = latest;
            s2t.seed_tokens = (int) stage2_asm.seed.size();
            s2t.seed_capped = stage2_asm.seed_capped;
            s2t.latest_tokens = std::max(0, latest.end - latest.begin);
            s2t.lookback_logit = lat.lookback_logit;
            s2t.vague_logit = lat.vague_logit;
            s2t.lookback = stage2_asm.lookback;
            s2t.earlier_tokens = (int) N;
            s2t.promoted_tokens = (int) stage2_asm.promoted.size();
            s2t.promoted_capped = stage2_asm.promoted_capped;
            s2t.promoted_runs = stage2_asm.promoted_runs;
            s2t.post_tokens = (int) stage2_asm.post.size();
            s2t.widened = stage2_asm.widened;
            s2t.budget = budget;
            s2t.budget_widened = stage2_asm.budget;
            s2t.raw_fits = stage2_asm.raw_fits;
            finish();
            return true;
        };
        if (request_v5) {
            request_trace.stage2.evaluated = true;
            request_trace.stage2.resume = resume;
            request_trace.stage2.history_dropped = drop_history ? (int) experiment.history_queries.size() : 0;
            if (!experiment.request_messages_all_error.empty()) {
                request_trace.fallback = "message_mapping: " + experiment.request_messages_all_error;
                request_trace.stage2.reason = "message_mapping";
            } else {
                request_trace.set = luce::pflash::pflash_request_query_set_v5(
                    session->request_raw, experiment.request_messages_all, st.request_tau);
                request_trace.fallback = request_trace.set.fallback;
                request_trace.stage2.seed_tokens = (int) request_trace.set.seed.size();
                request_trace.stage2.seed_capped = request_trace.set.seed_capped;
                request_trace.stage2.post_reserve = request_trace.set.reserve;
                if (!stage2_on) {
                    request_trace.stage2.reason = "stage2_not_loaded";
                } else if (request_trace.fallback != "logits_missing" && !stage2_pass1()) {
                    session->ids.clear();
                    cleanup();
                    return {};
                }
            }
        } else if (!experiment.request_messages_error.empty()) {
            request_trace.fallback = "message_mapping: " + experiment.request_messages_error;
        } else {
            if (st.request_turns) {
                // Candidate rule: the human messages and every
                // non-system messages, with their message indices.
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
            // v2 calibrated abstention: an unreliable extraction restores today's complete behaviour.
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
        // The query pin alone (no candidates) serves this request set: its rows must be exact, never BF16 stored
        // rows nor a resume-dependent "rows_missing" fallback.
        if (qpin && !qcand && resume > 0 &&
            (request_trace.rows_stored > 0 || request_trace.fallback == "rows_missing")) {
            redo = "request_set_rows_before_resume";
            return fail("query step: request-set rows not exactly available; redo from token 0");
        }
        // Retain each message's turn (previous first, so the latest is the
        // most recently used), whatever this call decided; with the v3 rule
        // the recent turns' runs too (oldest first).
        std::vector<luce::pflash::PFlashRequestMessageRuns> retained;
        for (size_t k = request_trace.set.turns.size(); k-- > 0;) {
            retained.push_back(request_trace.set.turns[k]);
        }
        for (size_t k = request_trace.set.messages.size(); k-- > 0;) {
            retained.push_back(request_trace.set.messages[k]);
        }
        for (const auto & message : retained) {
            std::vector<int> turn = luce::pflash::pflash_request_turn_tokens(message);
            if (stage2_assembled && !stage2_asm.promoted.empty()) {
                // Promoted earlier tokens are query rows too: keep them with their message's turn.
                for (const int t : stage2_asm.promoted) {
                    if (t >= message.message.begin && t < message.message.end) turn.push_back(t);
                }
                std::sort(turn.begin(), turn.end());
                turn.erase(std::unique(turn.begin(), turn.end()), turn.end());
                turn = luce::pflash::pflash_request_cap_tail(turn, luce::pflash::kRequestQueryCap);
            }
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
        // Feature capture and parity checks (request probe on only): append
        // this call's block-14 rows and request logits to
        // PFLASH_TRACE_REQUEST_FEATURES as one record: char[4] "RPF2",
        // i32 resume, i32 n_new, i32 hidden, f32 rows[n_new][hidden],
        // f32 request_logits[n_new] (tokens resume .. resume + n_new).
        if (const char * features = std::getenv("PFLASH_TRACE_REQUEST_FEATURES")) {
            if (*features && n_new > 0) {
                std::vector<float> rows((size_t) hidden * (size_t) n_new);
                ggml_backend_tensor_get(act_in, rows.data(), 0, rows.size() * sizeof(float));
                FILE * file = std::fopen(features, "ab");
                const int32_t header[3] = {resume, n_new, hidden};
                const bool written = file &&
                    std::fwrite("RPF2", 1, 4, file) == 4 &&
                    std::fwrite(header, sizeof(header), 1, file) == 1 &&
                    std::fwrite(rows.data(), sizeof(float), rows.size(), file) == rows.size() &&
                    std::fwrite(session->request_raw.data() + resume, sizeof(float),
                                (size_t) n_new, file) == (size_t) n_new;
                if (file) std::fclose(file);
                if (!written) {
                    return fail("PFLASH_TRACE_REQUEST_FEATURES: feature dump write failed");
                }
            }
        }
        const char * dump = std::getenv("PFLASH_TRACE_REQUEST_LOGITS");
        if (dump && std::strcmp(dump, "1") == 0) {
            for (const auto & message : request_trace.set.messages) {
                request_trace.message_logits.emplace_back(
                    session->request_raw.begin() + message.message.begin,
                    session->request_raw.begin() + message.message.end);
            }
        }
        if (qstep) {
            // Units from the selector's own segmentation of this request (cut at the caller's pins), ranked by the
            // stage-1 logits; every query token set's rows are read now, exactly (see the gather below).
            const auto r0 = std::chrono::steady_clock::now();
            std::vector<PFlashTokenSpan> unit_segments;
            bool unit_density = false;
            segment(required_instruction_spans, unit_segments, unit_density);
            if (unit_segments.empty()) return fail("query step: the segment probe gave no segments");
            qunits = luce::pflash::pflash_query_units(unit_segments, experiment.request_messages_all,
                                                      session->request_raw);
            if (!qunits.error.empty()) {
                const std::string message = "query step: units: " + qunits.error;
                return fail(message.c_str());
            }
            query_trace.evaluated = qcand || qpin;
            query_trace.config = qcfg;
            query_trace.units = (int) qunits.units.size();
            query_trace.system_used = qunits.system_used;
            std::vector<std::vector<int>> wanted;
            if (qcand) {
                query_trace.set = luce::pflash::pflash_query_candidate_set(qunits, session->request_raw, qcfg);
                if (query_trace.set.candidates.empty()) return fail("query step: no message content to rank");
                for (const auto & c : query_trace.set.candidates) {
                    wanted.push_back(c.tokens);
                    query_trace.query_tokens += (int) c.tokens.size();
                }
                if (qcfg.end_share > 0.0f) wanted.push_back({query_end - 1});
            }
            if (qgrid) {
                qfingerprint = luce::pflash::pflash_ids_fingerprint(ids);
                for (const auto & g : qgrid_configs) {
                    luce::pflash::PFlashQueryCandidateSet set;
                    if (g.mode == luce::pflash::PFlashGridMode::Candidates) {
                        set = luce::pflash::pflash_query_candidate_set(qunits, session->request_raw, g.query);
                        for (const auto & c : set.candidates) wanted.push_back(c.tokens);
                        if (g.query.end_share > 0.0f) wanted.push_back({query_end - 1});
                    }
                    qgrid_sets.push_back(std::move(set));
                }
                if (const char * dir = std::getenv("PFLASH_QUERY_ORACLE_DIR")) {
                    std::ifstream in(std::string(dir) + "/" + qfingerprint + ".json");
                    if (!in) {
                        qoracle_error = "no_oracle_file";
                    } else {
                        std::stringstream text;
                        text << in.rdbuf();
                        if (luce::pflash::pflash_query_oracle_parse(text.str(), S, qoracle, qoracle_error)) {
                            qoracle_error.clear();
                            wanted.push_back(qoracle.query);
                        }
                    }
                }
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
                        return fail("query step: query rows unavailable");
                    }
                } else if (tokens.back() >= resume || !session->query_sets.lookup(tokens, hidden, rows)) {
                    redo = tokens.back() >= resume ? "query_set_spans_resume" : "query_set_not_stored";
                    return fail("query step: query rows not exactly available; redo from token 0");
                }
                qrows_new += fresh;
                qrows.emplace(tokens, std::move(rows));
            }
            query_trace.rank_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - r0).count();
        }
    }
    // PFLASH_EVAL_REF_IDENT_QUERY: the identifier query's block-14 rows, read
    // while this call's activations exist. Positions before ``resume`` have
    // no rows (only the request probe keeps a row store): the request then
    // falls back to today's windows.
    const luce::pflash::PFlashIdentQuery ident_mode = experiment.query_ident;
    luce::pflash::PFlashIdentQueryTrace ident_trace;
    std::vector<float> ident_query_rows;   // [hidden, query tokens]
    if (ident_mode != luce::pflash::PFlashIdentQuery::Off) {
        ident_trace.mode = ident_mode;
        ident_trace.turns = experiment.ident_turns;
        ident_trace.ident_tokens = (int) experiment.ident_tokens.size();
        std::vector<int> query = experiment.ident_tokens;
        if (!experiment.ident_error.empty()) {
            ident_trace.fallback = "mapping: " + experiment.ident_error;
        } else if (query.empty()) {
            ident_trace.fallback = "no_identifiers";
        } else {
            if (qstep && resume > 0 && query.front() < resume) {
                // Served by the query pin path: never a resume-dependent fallback.
                redo = "ident_rows_before_resume";
                return fail("query step: identifier rows not exactly available; redo from token 0");
            }
            if (query.front() < resume || query.back() >= S) {
                ident_trace.fallback = "rows_missing";
            } else {
                ident_query_rows.assign((size_t) hidden * query.size(), 0.0f);
                size_t index = 0;
                for (const auto & span : luce::pflash::pflash_request_token_spans(query)) {
                    const int n = span.end - span.begin;
                    ggml_backend_tensor_get(act_in, ident_query_rows.data() + index * (size_t) hidden,
                                            (size_t) (span.begin - resume) * act_in->nb[1],
                                            (size_t) hidden * (size_t) n * sizeof(float));
                    index += (size_t) n;
                }
                ident_trace.tokens = std::move(query);
            }
        }
    }
    cleanup();
    // The session now covers this prompt, resumable at the checkpoint and,
    // while the query stays put, reusing its rows.
    session->ids = ids;
    session->checkpoint = checkpoint;
    session->proj_rows = std::move(keep_proj);
    session->proj_rows_begin = keep_begin;
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
    const int n_key_chunks = (S + key_chunk - 1) / key_chunk;
    const auto score_window = [&](const ScoredWindow & window,
                                  std::vector<float> & mass) -> bool {
        const int nq = window.end - window.begin;
        ggml_init_params lip{};
        lip.mem_size = (size_t)8 * ggml_tensor_overhead() + 4096;
        lip.no_alloc = true;
        ggml_context * lctx = ggml_init(lip);
        if (!lctx) {
            set_last_error("qwen35 score buffer ctx allocation failed");
            return false;
        }
        ggml_tensor * logits = ggml_new_tensor_3d(lctx, GGML_TYPE_F32, S, nq, H);
        ggml_tensor * mask = ggml_new_tensor_2d(lctx, GGML_TYPE_F32, S, nq);
        ggml_tensor * x_q = ggml_new_tensor_2d(lctx, GGML_TYPE_F32, hidden, nq);
        ggml_backend_buffer_t lbuf = ggml_backend_alloc_ctx_tensors(lctx, w.backend);
        if (!lbuf) {
            ggml_free(lctx);
            set_last_error("qwen35 score buffer allocation failed");
            return false;
        }
        ggml_backend_tensor_set(x_q, window.rows.data(), 0,
                                window.rows.size() * sizeof(float));
        {
            // Keys are the context before the query window and, when the
            // tokens after it are candidates too, the context after it. NoPE
            // scoring has no position term, so a later key scores like an
            // earlier one.
            std::vector<float> m((size_t)nq * S, -INFINITY);
            for (int t = 0; t < nq; ++t) {
                float * row = m.data() + (size_t)t * S;
                std::fill_n(row, (size_t)query_start, 0.0f);
                if (experiment.query_suffix_candidates) {
                    std::fill_n(row + query_end, (size_t)(S - query_end), 0.0f);
                }
                if (window.end <= query_start) {
                    std::fill_n(row + window.begin, (size_t)nq, -INFINITY);
                }
            }
            ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));
        }
        ggml_init_params sip{};
        sip.mem_size = ggml_tensor_overhead() * (size_t)(64 + 24 * n_key_chunks) +
                       ggml_graph_overhead_custom(4096, false) + 64 * 1024;
        sip.no_alloc = true;
        ggml_context * sctx = ggml_init(sip);
        if (!sctx) {
            ggml_backend_buffer_free(lbuf); ggml_free(lctx);
            set_last_error("qwen35 score graph ctx allocation failed");
            return false;
        }
        ggml_cgraph * sgf = ggml_new_graph_custom(sctx, 4096, false);
        ggml_tensor * q_in = ggml_mul(sctx, ggml_rms_norm(sctx, x_q, w.rms_eps), L.attn_norm);
        ggml_tensor * Q = nullptr;
        if (st.head_loaded) {
            Q = ggml_reshape_3d(sctx, ggml_mul_mat(sctx, st.head_wq, q_in), D, H, nq);
        } else {
            // Native block 15 packs query and gate rows per head; keep the query half.
            ggml_tensor * QG = ggml_reshape_3d(sctx, ggml_mul_mat(sctx, L.wq, q_in),
                                               D * 2, H, nq);
            Q = ggml_view_3d(sctx, QG, D, H, nq,
                             ggml_element_size(QG) * D * 2,
                             ggml_element_size(QG) * D * 2 * H, 0);
        }
        Q = ggml_mul(sctx, ggml_rms_norm(sctx, Q, w.rms_eps), L.q_norm);
        ggml_tensor * Q_perm = ggml_cont(sctx, ggml_permute(sctx, Q, 0, 2, 1, 3));  // [D, nq, H]
        for (int b = 0; b < S; b += key_chunk) {
            const int n = std::min(key_chunk, S - b);
            ggml_tensor * K = ggml_view_3d(sctx, session->keys, D, Hk, n,
                                           session->keys->nb[1], session->keys->nb[2],
                                           (size_t)b * session->keys->nb[2]);
            K = ggml_cont(sctx, ggml_permute(sctx, K, 0, 2, 1, 3));  // [D, n, Hk]
            ggml_tensor * K_score = K;
            if (H != Hk) {
                const int gqa = H / Hk;
                ggml_tensor * K_4d = ggml_reshape_4d(sctx, K, D, n, 1, Hk);
                ggml_tensor * K_tpl = ggml_new_tensor_4d(sctx, GGML_TYPE_F32, D, n, gqa, Hk);
                K_score = ggml_reshape_3d(sctx, ggml_repeat(sctx, K_4d, K_tpl), D, n, H);
            }
            ggml_tensor * part = ggml_mul_mat(sctx, K_score, Q_perm);  // [n, nq, H]
            ggml_tensor * dst = ggml_view_3d(sctx, logits, n, nq, H,
                                             logits->nb[1], logits->nb[2],
                                             (size_t)b * logits->nb[0]);
            ggml_build_forward_expand(sgf, ggml_cpy(sctx, part, dst));
        }
        ggml_tensor * probs = ggml_soft_max_ext(sctx, logits, mask,
                                                1.0f / std::sqrt((float)D), 0.0f);
        ggml_set_output(probs);
        ggml_build_forward_expand(sgf, probs);
        ggml_gallocr_t salloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
        const bool ok = ggml_gallocr_alloc_graph(salloc, sgf) &&
            ggml_backend_graph_compute(w.backend, sgf) == GGML_STATUS_SUCCESS;
        std::vector<float> probs_h;
        if (ok) {
            probs_h.resize((size_t)S * nq * H);
            ggml_backend_tensor_get(probs, probs_h.data(), 0, probs_h.size() * sizeof(float));
        }
        ggml_gallocr_free(salloc);
        ggml_free(sctx);
        ggml_backend_buffer_free(lbuf);
        ggml_free(lctx);
        if (!ok) {
            set_last_error("qwen35 score graph compute failed");
            return false;
        }
        const size_t nonfinite = count_nonfinite_scores(probs_h.data(), probs_h.size());
        if (nonfinite != 0) {
            const std::string message =
                "non-finite Qwen3.5 scoring-head scores: " + std::to_string(nonfinite) +
                "/" + std::to_string(probs_h.size());
            std::fprintf(stderr, "[pflash] ERROR: %s\n", message.c_str());
            std::fflush(stderr);
            set_last_error(message);
            return false;
        }
        scoring_head_mean_token_mass(probs_h.data(), S, nq, H, mass);
        return true;
    };
    std::vector<float> token_mass;
    // Today's windows: each scored on its own, mixed at their weights.
    const auto score_windows = [&]() -> bool {
        token_mass.clear();
        double total_weight = 0.0;
        for (const auto & window : windows) {
            std::vector<float> mass;
            if (!score_window(window, mass)) return false;
            if (token_mass.empty()) token_mass.assign(mass.size(), 0.0f);
            for (size_t i = 0; i < mass.size(); ++i) {
                token_mass[i] += (float) window.weight * mass[i];
            }
            total_weight += window.weight;
        }
        for (auto & value : token_mass) value = (float) (value / total_weight);
        return true;
    };
    // Request probe: ONE gathered query set with one common key mask --
    // today's key range (the context before the query window, and after it
    // when those tokens are candidates) minus every query token -- mean over
    // heads and query tokens, as for a window. Queries run in bounded tiles;
    // each tile's probabilities are reduced to one row per key on the
    // device, so only [S] floats per tile come back.
    // The identifier query uses the same scoring with its tokens left among
    // the keys (scored, not pinned: arm viii's key mask).
    const auto score_request_set = [&](const std::vector<int> & tokens,
                                       const std::vector<float> & rows,
                                       bool mask_query_tokens, int & tiles) -> bool {
        constexpr int kTile = 32;
        const int nq = (int) tokens.size();
        std::vector<float> mask_row((size_t) S, -INFINITY);
        std::fill_n(mask_row.begin(), (size_t) query_start, 0.0f);
        if (experiment.query_suffix_candidates) {
            std::fill(mask_row.begin() + query_end, mask_row.end(), 0.0f);
        }
        if (mask_query_tokens) {
            for (const int t : tokens) mask_row[(size_t) t] = -INFINITY;
        }
        std::vector<double> sum((size_t) S, 0.0);
        std::vector<float> part((size_t) S);
        for (int q0 = 0; q0 < nq; q0 += kTile) {
            const int n = std::min(kTile, nq - q0);
            ggml_init_params lip{};
            lip.mem_size = (size_t)8 * ggml_tensor_overhead() + 4096;
            lip.no_alloc = true;
            ggml_context * lctx = ggml_init(lip);
            if (!lctx) {
                set_last_error("qwen35 request score buffer ctx allocation failed");
                return false;
            }
            ggml_tensor * logits = ggml_new_tensor_3d(lctx, GGML_TYPE_F32, S, n, H);
            ggml_tensor * mask = ggml_new_tensor_2d(lctx, GGML_TYPE_F32, S, n);
            ggml_tensor * x_q = ggml_new_tensor_2d(lctx, GGML_TYPE_F32, hidden, n);
            ggml_backend_buffer_t lbuf = ggml_backend_alloc_ctx_tensors(lctx, w.backend);
            if (!lbuf) {
                ggml_free(lctx);
                set_last_error("qwen35 request score buffer allocation failed");
                return false;
            }
            ggml_backend_tensor_set(x_q, rows.data() + (size_t) q0 * hidden, 0,
                                    (size_t) hidden * n * sizeof(float));
            for (int t = 0; t < n; ++t) {
                ggml_backend_tensor_set(mask, mask_row.data(), (size_t) t * mask->nb[1],
                                        (size_t) S * sizeof(float));
            }
            ggml_init_params sip{};
            sip.mem_size = ggml_tensor_overhead() * (size_t)(64 + 24 * n_key_chunks) +
                           ggml_graph_overhead_custom(4096, false) + 64 * 1024;
            sip.no_alloc = true;
            ggml_context * sctx = ggml_init(sip);
            if (!sctx) {
                ggml_backend_buffer_free(lbuf); ggml_free(lctx);
                set_last_error("qwen35 request score graph ctx allocation failed");
                return false;
            }
            ggml_cgraph * sgf = ggml_new_graph_custom(sctx, 4096, false);
            ggml_tensor * q_in = ggml_mul(sctx, ggml_rms_norm(sctx, x_q, w.rms_eps), L.attn_norm);
            ggml_tensor * Q = nullptr;
            if (st.head_loaded) {
                Q = ggml_reshape_3d(sctx, ggml_mul_mat(sctx, st.head_wq, q_in), D, H, n);
            } else {
                ggml_tensor * QG = ggml_reshape_3d(sctx, ggml_mul_mat(sctx, L.wq, q_in),
                                                   D * 2, H, n);
                Q = ggml_view_3d(sctx, QG, D, H, n,
                                 ggml_element_size(QG) * D * 2,
                                 ggml_element_size(QG) * D * 2 * H, 0);
            }
            Q = ggml_mul(sctx, ggml_rms_norm(sctx, Q, w.rms_eps), L.q_norm);
            ggml_tensor * Q_perm = ggml_cont(sctx, ggml_permute(sctx, Q, 0, 2, 1, 3));  // [D, n, H]
            for (int b = 0; b < S; b += key_chunk) {
                const int nk = std::min(key_chunk, S - b);
                ggml_tensor * K = ggml_view_3d(sctx, session->keys, D, Hk, nk,
                                               session->keys->nb[1], session->keys->nb[2],
                                               (size_t)b * session->keys->nb[2]);
                K = ggml_cont(sctx, ggml_permute(sctx, K, 0, 2, 1, 3));  // [D, nk, Hk]
                ggml_tensor * K_score = K;
                if (H != Hk) {
                    const int gqa = H / Hk;
                    ggml_tensor * K_4d = ggml_reshape_4d(sctx, K, D, nk, 1, Hk);
                    ggml_tensor * K_tpl = ggml_new_tensor_4d(sctx, GGML_TYPE_F32, D, nk, gqa, Hk);
                    K_score = ggml_reshape_3d(sctx, ggml_repeat(sctx, K_4d, K_tpl), D, nk, H);
                }
                ggml_tensor * scores = ggml_mul_mat(sctx, K_score, Q_perm);  // [nk, n, H]
                ggml_tensor * dst = ggml_view_3d(sctx, logits, nk, n, H,
                                                 logits->nb[1], logits->nb[2],
                                                 (size_t)b * logits->nb[0]);
                ggml_build_forward_expand(sgf, ggml_cpy(sctx, scores, dst));
            }
            ggml_tensor * probs = ggml_soft_max_ext(sctx, logits, mask,
                                                    1.0f / std::sqrt((float)D), 0.0f);
            // Sum over (query, head) per key: [S, n*H] -> [n*H, S] -> [1, S].
            ggml_tensor * per_key = ggml_sum_rows(sctx, ggml_cont(sctx,
                ggml_transpose(sctx, ggml_reshape_2d(sctx, probs, S, (int64_t) n * H))));
            ggml_set_output(per_key);
            ggml_build_forward_expand(sgf, per_key);
            ggml_gallocr_t salloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
            const bool ok = ggml_gallocr_alloc_graph(salloc, sgf) &&
                ggml_backend_graph_compute(w.backend, sgf) == GGML_STATUS_SUCCESS;
            if (ok) {
                ggml_backend_tensor_get(per_key, part.data(), 0, (size_t) S * sizeof(float));
            }
            ggml_gallocr_free(salloc);
            ggml_free(sctx);
            ggml_backend_buffer_free(lbuf);
            ggml_free(lctx);
            if (!ok) {
                set_last_error("qwen35 request score graph compute failed");
                return false;
            }
            const size_t nonfinite = count_nonfinite_scores(part.data(), part.size());
            if (nonfinite != 0) {
                const std::string message =
                    "non-finite Qwen3.5 scoring-head scores (request probe): " +
                    std::to_string(nonfinite) + "/" + std::to_string(part.size());
                std::fprintf(stderr, "[pflash] ERROR: %s\n", message.c_str());
                std::fflush(stderr);
                set_last_error(message);
                return false;
            }
            for (int j = 0; j < S; ++j) sum[(size_t) j] += part[(size_t) j];
            ++tiles;
        }
        const double denominator = (double) H * (double) nq;
        token_mass.assign((size_t) S, 0.0f);
        for (int j = 0; j < S; ++j) token_mass[(size_t) j] = (float) (sum[(size_t) j] / denominator);
        return true;
    };
    // ---------------------------------------------------------------- query step 1
    // Masses of query token sets, cached per set (one tile stream per set; tiles never mix sets).
    std::map<std::vector<int>, std::vector<float>> qmass;
    int qtiles = 0;
    const auto query_mass = [&](const std::vector<int> & tokens) -> const std::vector<float> * {
        auto hit = qmass.find(tokens);
        if (hit != qmass.end()) return &hit->second;
        auto rows = qrows.find(tokens);
        if (rows == qrows.end()) {
            set_last_error("query step: the rows of a query set are missing");
            return nullptr;
        }
        if (!score_request_set(tokens, rows->second, /*mask_query_tokens=*/true, qtiles)) return nullptr;
        return &qmass.emplace(tokens, token_mass).first->second;
    };
    // Today's windows as a blend candidate (PFLASH_QUERY_TW): its mass, scored and mixed exactly as today (cached),
    // and its score = the unit score of s1 over the window tokens inside non-system message content.
    std::vector<float> qwindows_mass;
    bool qwindows_ready = false;
    const auto windows_mass_cached = [&]() -> const std::vector<float> * {
        if (!qwindows_ready) {
            if (!score_windows()) return nullptr;
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
                                      luce::pflash::PFlashQueryBlend & blend, float & tw_score_out) -> bool {
        tw_score_out = q.tw_score ? tw_unit_score(q.unit_score) : std::numeric_limits<float>::quiet_NaN();
        blend = luce::pflash::pflash_query_blend(set, q, tw_score_out);
        std::vector<std::vector<float>> masses;
        std::vector<double> weights;
        for (size_t i = 0; i < set.candidates.size(); ++i) {
            const auto * m = query_mass(set.candidates[i].tokens);
            if (!m) return false;
            masses.push_back(*m);
            weights.push_back(blend.probe[i]);
        }
        if (blend.tw > 0.0) {
            const auto * m = windows_mass_cached();
            if (!m) return false;
            masses.push_back(*m);
            weights.push_back(blend.tw);
        }
        if (blend.end > 0.0) {
            const auto * m = query_mass({query_end - 1});
            if (!m) return false;
            masses.push_back(*m);
            weights.push_back(blend.end);
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
    const auto spans_json = [](const std::vector<PFlashTokenSpan> & spans) {
        std::string out = "[";
        for (size_t i = 0; i < spans.size(); ++i) {
            out += (i ? ",[" : "[") + std::to_string(spans[i].begin) + "," + std::to_string(spans[i].end) + "]";
        }
        return out + "]";
    };
    const auto json_str = [](const std::string & text) {
        std::string out = "\"";
        for (const char c : text) {
            if (c == '"' || c == '\\') out += '\\';
            if ((unsigned char) c >= 0x20) out += c;
        }
        return out + "\"";
    };
    const auto num = [](double v) {
        if (!std::isfinite(v)) return std::string("null");
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.9g", v);
        return std::string(buf);
    };
    const auto candidates_json = [&](const luce::pflash::PFlashQueryCandidateSet & set) {
        std::string out = "[";
        for (size_t i = 0; i < set.candidates.size(); ++i) {
            const auto & c = set.candidates[i];
            std::vector<PFlashTokenSpan> unit_spans;
            for (const int u : c.units) unit_spans.push_back(qunits.units[(size_t) u].span);
            out += std::string(i ? "," : "") + "{\"message\":" + std::to_string(c.message) + ",\"units\":" +
                spans_json(merge_spans(unit_spans)) + ",\"tokens\":" + std::to_string(c.tokens.size()) +
                ",\"score\":" + num(c.score) + ",\"weight\":" + num(c.weight) + "}";
        }
        return out + "]";
    };
    // Dev-only replay grid: every config's selection on this forward, then the served selection below.
    if (qgrid) {
        const auto grid_started = std::chrono::steady_clock::now();
        std::vector<float> windows_mass;
        bool have_windows = false;
        std::vector<float> p1_mass;
        bool have_p1 = false;
        int p1_tiles = 0;
        std::string record = "{\"fp\":\"" + qfingerprint + "\",\"n_tokens\":" + std::to_string(S) +
            ",\"budget\":" + std::to_string(qbudget) + ",\"keep_ratio\":" + num(keep_ratio) +
            ",\"query_window\":[" + std::to_string(query_start) + "," + std::to_string(query_end) + "]" +
            ",\"units\":" + std::to_string(qunits.units.size()) + ",\"latest\":" + std::to_string(qunits.latest) +
            ",\"system_used\":" + (qunits.system_used ? "true" : "false") +
            ",\"p1_fallback\":" + json_str(request_trace.fallback) +
            ",\"oracle\":" + json_str(qoracle_error) + ",\"configs\":{";
        for (size_t gi = 0; gi < qgrid_configs.size(); ++gi) {
            const auto & g = qgrid_configs[gi];
            const auto c0 = std::chrono::steady_clock::now();
            std::vector<float> mass;
            std::vector<PFlashTokenSpan> cpins = required_instruction_spans;
            std::string fallback, error, cand = "[]", blend_json;
            int query_tokens = 0;
            bool ok = true;
            if (g.mode == luce::pflash::PFlashGridMode::S1 ||
                (g.mode == luce::pflash::PFlashGridMode::P1 && !request_trace.fallback.empty())) {
                if (g.mode == luce::pflash::PFlashGridMode::P1) fallback = request_trace.fallback;
                if (!have_windows) {
                    if (!score_windows()) { session->ids.clear(); return {}; }
                    windows_mass = token_mass;
                    have_windows = true;
                }
                mass = windows_mass;
            } else if (g.mode == luce::pflash::PFlashGridMode::P1) {
                // The served P1 path: the set's tokens pinned (+ the policy recency pin), or today's windows and
                // pins when they do not fit while the caller's pins do (pins_exceed_budget).
                auto p1pins = cpins;
                for (const auto & span : luce::pflash::pflash_request_token_spans(request_trace.set.tokens)) {
                    p1pins.push_back(span);
                }
                if (!request_trace.set.messages.empty()) {
                    const auto recency = luce::pflash::pflash_request_recency_pin(
                        request_trace.set.messages[0].message, st.request_policy.recency_pin_tokens);
                    if (recency.begin >= 0) p1pins.push_back(recency);
                }
                p1pins = merge_spans(std::move(p1pins));
                std::vector<PFlashTokenSpan> segs, base;
                bool dens = false;
                segment(p1pins, segs, dens);
                segment(required_instruction_spans, base, dens);
                if (charged(p1pins, segs) > qbudget && charged(required_instruction_spans, base) <= qbudget) {
                    fallback = "pins_exceed_budget";
                    if (!have_windows) {
                        if (!score_windows()) { session->ids.clear(); return {}; }
                        windows_mass = token_mass;
                        have_windows = true;
                    }
                    mass = windows_mass;
                } else {
                    if (!have_p1) {
                        if (!score_request_set(request_trace.set.tokens, request_query_rows,
                                               /*mask_query_tokens=*/true, p1_tiles)) {
                            session->ids.clear();
                            return {};
                        }
                        p1_mass = token_mass;
                        have_p1 = true;
                    }
                    mass = p1_mass;
                    cpins = p1pins;
                    query_tokens = (int) request_trace.set.tokens.size();
                }
            } else if (g.mode == luce::pflash::PFlashGridMode::Candidates) {
                const auto & set = qgrid_sets[gi];
                if (set.candidates.empty()) {
                    error = "no_candidates";
                    ok = false;
                } else {
                    luce::pflash::PFlashQueryBlend blend;
                    float tws = 0.0f;
                    if (!query_blend_mass(set, g.query, mass, blend, tws)) { session->ids.clear(); return {}; }
                    for (const auto & c : set.candidates) query_tokens += (int) c.tokens.size();
                    cand = candidates_json(set);
                    std::string probe_w = "[";
                    for (size_t i = 0; i < blend.probe.size(); ++i) probe_w += (i ? "," : "") + num(blend.probe[i]);
                    blend_json = ",\"blend\":{\"probe\":" + probe_w + "],\"tw\":" + num(blend.tw) + ",\"end\":" +
                        num(blend.end) + ",\"tw_score\":" + num(tws) + "}";
                }
            } else {
                if (!qoracle_error.empty()) {
                    error = qoracle_error;
                    ok = false;
                } else {
                    const auto * m = query_mass(qoracle.query);
                    if (!m) { session->ids.clear(); return {}; }
                    mass = *m;
                    query_tokens = (int) qoracle.query.size();
                    if (g.oracle_pin) {
                        auto p = cpins;
                        p.insert(p.end(), qoracle.pin.begin(), qoracle.pin.end());
                        p = merge_spans(std::move(p));
                        std::vector<PFlashTokenSpan> segs;
                        bool dens = false;
                        segment(p, segs, dens);
                        if (charged(p, segs) <= qbudget) cpins = p;
                        else fallback = "oracle_pin_exceeds_budget";
                    }
                }
            }
            luce::pflash::PFlashQueryPin pin;
            int cap_used = 0;
            std::vector<PFlashTokenSpan> kept;
            int kept_tokens = 0;
            if (ok && g.query.pin_tokens > 0) cpins = fit_pin(cpins, g.query, pin, cap_used);
            if (ok) {
                std::vector<PFlashTokenSpan> segs;
                bool dens = false;
                segment(cpins, segs, dens);
                const auto out = select_pflash_chunks(
                    ids, mass, keep_ratio, n_lookahead, score_query_end, /*pool_kernel=*/1, experiment, cpins,
                    /*direct_mass=*/true, /*write_trace=*/false, segs.empty() ? nullptr : &segs, dens,
                    /*other_token_scores=*/nullptr, /*split_fraction=*/0.0, st.fold_vocab.get());
                if (out.empty()) {
                    error = std::string("select: ") + luce_last_error();
                    ok = false;
                } else {
                    kept = pflash_last_kept_spans();
                    kept_tokens = (int) out.size();
                }
            }
            int pin_tokens = 0;
            for (const auto & span : cpins) pin_tokens += span.end - span.begin;
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - c0).count();
            record += std::string(gi ? "," : "") + json_str(g.name) + ":{\"mode\":\"" +
                luce::pflash::pflash_grid_mode_name(g.mode) + "\",\"kept\":" + spans_json(kept) +
                ",\"kept_tokens\":" + std::to_string(kept_tokens) + ",\"pins\":" + spans_json(cpins) +
                ",\"pin_tokens\":" + std::to_string(pin_tokens) + ",\"verbatim_pin\":" + spans_json(pin.spans) +
                ",\"verbatim_pin_tokens\":" + std::to_string(pin.tokens) + ",\"pin_cap_used\":" +
                std::to_string(cap_used) + ",\"query_tokens\":" + std::to_string(query_tokens) +
                ",\"fallback\":" + json_str(fallback) + ",\"candidates\":" + cand +
                (g.mode == luce::pflash::PFlashGridMode::Candidates
                     ? ",\"latest_z\":" + num(qgrid_sets[gi].latest_z) + ",\"latest_null\":" +
                           (qgrid_sets[gi].latest_null ? "true" : "false")
                     : std::string()) +
                blend_json + ",\"ms\":" + num(ms) +
                (error.empty() ? std::string() : ",\"error\":" + json_str(error)) + "}";
        }
        const double grid_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - grid_started).count();
        record += "},\"timing_ms\":{\"grid\":" + num(grid_ms) + ",\"rank\":" + num(query_trace.rank_us / 1000.0) +
            ",\"forward\":" + num(std::chrono::duration<double, std::milli>(t1 - t0).count()) +
            "},\"query_tiles\":" + std::to_string(qtiles) + ",\"p1_tiles\":" + std::to_string(p1_tiles) +
            ",\"rows_new\":" + std::to_string(qrows_new) + "}\n";
        FILE * file = std::fopen(std::getenv("PFLASH_QUERY_GRID_OUT"), "ab");
        const bool written = file && std::fwrite(record.data(), 1, record.size(), file) == record.size();
        if (file) std::fclose(file);
        if (!written) {
            session->ids.clear();
            set_last_error("PFLASH_QUERY_GRID_OUT: write failed");
            return {};
        }
    }
    // The served candidates replace the request set (named, not a fallback to windows).
    if (qcand) request_trace.fallback = "replaced_by_query_candidates";
    if (qcand || qpin) request_trace.query = &query_trace;
    query_trace.sessions = sessions_on;
    query_trace.session_resume = resume;
    query_trace.redo = !redone.empty();
    query_trace.redo_reason = redone;
    bool request_set = request_probe && request_trace.fallback.empty();
    const bool ident_set = ident_mode != luce::pflash::PFlashIdentQuery::Off &&
        ident_trace.fallback.empty();
    const auto pass2_started = std::chrono::steady_clock::now();
    if (qcand) {
        // The served candidates: one mass per candidate (cached from the grid when it ran), aggregated.
        const int tiles_before = qtiles;
        qmass.clear();   // timing and tiles of the served path alone
        qwindows_ready = false;
        std::vector<float> aggregated;
        if (!query_blend_mass(query_trace.set, qcfg, aggregated, query_trace.blend, query_trace.tw_score)) {
            session->ids.clear();
            return {};
        }
        token_mass = std::move(aggregated);
        query_trace.tiles = qtiles - tiles_before;
        query_trace.score_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - pass2_started).count();
    } else if (request_set ? !score_request_set(request_trace.set.tokens, request_query_rows,
                                         /*mask_query_tokens=*/true, request_trace.query_tiles)
        : ident_set ? !score_request_set(ident_trace.tokens, ident_query_rows,
                                         /*mask_query_tokens=*/false, ident_trace.query_tiles)
                    : !score_windows()) {
        session->ids.clear();
        return {};
    }
    if (request_trace.stage2.evaluated) {
        request_trace.stage2.pass2_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - pass2_started).count();
    }
    // Stage 2's budget (contract section 5 step 5): B' = min(n, B + B/2) when l_vague >= tau_fb, whatever query
    // the request ends up scoring; raw_fits keeps every token (the caller serves the raw prompt). The selector
    // budget is floor(S * ratio), so the ratio is the float whose budget is exactly B'.
    float select_keep_ratio = keep_ratio;
    int select_budget = (int) std::floor((double) S * (double) keep_ratio);
    if (stage2_assembled && stage2_asm.widened) {
        select_budget = stage2_asm.budget;
        if (stage2_asm.raw_fits) {
            select_keep_ratio = 1.0f;
        } else {
            float r = (float) (((double) select_budget + 0.5) / (double) S);
            for (int k = 0; k < 8 && (int) std::floor((double) S * (double) r) > select_budget; ++k) {
                r = std::nextafter(r, 0.0f);
            }
            for (int k = 0; k < 8 && (int) std::floor((double) S * (double) r) < select_budget; ++k) {
                r = std::nextafter(r, 2.0f);
            }
            if ((int) std::floor((double) S * (double) r) != select_budget) {
                set_last_error("stage 2: no keep ratio gives the widened budget");
                session->ids.clear();
                return {};
            }
            select_keep_ratio = r;
        }
    }
    if (stage2_assembled) {
        luce::common::PFlashStage2Outcome outcome;
        outcome.widened = stage2_asm.widened;
        outcome.raw_fits = stage2_asm.raw_fits;
        outcome.budget = (int) std::floor((double) S * (double) keep_ratio);
        outcome.budget_widened = stage2_asm.budget;
        pflash_set_stage2_outcome(outcome);
    }
    if (ident_mode != luce::pflash::PFlashIdentQuery::Off) {
        ident_trace.query_tokens = ident_set ? (int) ident_trace.tokens.size() : 0;
        if (!ident_set) ident_trace.tokens.clear();
        std::fprintf(stderr,
            "[pflash-ident-query] mode=%s %s turns=%d ident_tokens=%d query_tokens=%d tiles=%d\n",
            luce::pflash::pflash_ident_query_name(ident_mode),
            ident_set ? "scored" : ("fallback=" + ident_trace.fallback).c_str(),
            ident_trace.turns, ident_trace.ident_tokens, ident_trace.query_tokens,
            ident_trace.query_tiles);
        std::fflush(stderr);
    }
    auto t2 = std::chrono::steady_clock::now();
    std::fprintf(stderr,
        "[qwen35-scorer] forward %.2fs (blocks 0-%d, S=%d, resumed at %d, "
        "%d new) score %.2fs (%zu query windows) total %.2fs head=%s\n",
        std::chrono::duration<double>(t1 - t0).count(), kQwen35HeadBlock - 1, S,
        resume, n_new,
        std::chrono::duration<double>(t2 - t1).count(),
        qcand ? query_trace.set.candidates.size() : request_set || ident_set ? (size_t) 1 : windows.size(),
        std::chrono::duration<double>(t2 - t0).count(),
        st.head_loaded ? "trained" : "native-block15");
    std::fflush(stderr);
    pflash_set_scoring_stats({resume, n_new,
                              qcand ? (int) query_trace.set.candidates.size()
                                    : request_set || ident_set ? 1 : (int) windows.size(),
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
    // PFLASH_QUERY_PIN: the latest user message's top whole units, charged, shrunk to fit the budget.
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
                if (!score_windows()) {
                    session->ids.clear();
                    return {};
                }
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
            "(evicted %d, invalidated %d) messages:%s%s%s\n",
            request_set ? "scored" : ("fallback=" + request_trace.fallback).c_str(),
            request_trace.query_tokens, request_trace.set.latest_tokens,
            request_trace.set.previous_tokens, request_trace.set.capped ? ", capped" : "",
            request_trace.rows_new, request_trace.rows_stored, request_trace.pinned_new_tokens,
            request_trace.mandatory_tokens, request_trace.mandatory_tokens_base,
            request_trace.token_budget, request_trace.store_turns, request_trace.store_tokens,
            request_trace.store_evicted_turns, request_trace.store_invalidated_turns,
            runs.empty() ? " none" : runs.c_str(), request_trace.stub ? " (stub row)" : "",
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
        auto & s2t = request_trace.stage2;
        if (s2t.evaluated) {
            // Every request that does not score the stage-2 query is a named, counted reason.
            s2t.fallback = request_set || qcand ? std::string() : request_trace.fallback;
            if (!s2t.fallback.empty()) {
                s2t.fallback_count = stage2_count_fallback(s2t.fallback);
                std::fprintf(stderr, "[pflash-request-stage2] FALLBACK reason=\"%s\" count=%ld\n",
                             s2t.fallback.c_str(), s2t.fallback_count);
            }
            std::fprintf(stderr,
                "[pflash-request-stage2] %s%s L=%d [%d,%d) seed=%d%s latest=%d l_lb=%.4f l_vague=%.4f lookback=%d "
                "earlier=%d promoted=%d%s (runs %d) post=%d widened=%d budget=%d->%d raw_fits=%d s2=%d resume=%d "
                "history_dropped=%d tiles=%d pass1=%.1fms pass2=%.1fms\n",
                s2t.active ? "active" : "inactive", s2t.reason.empty() ? "" : (" reason=" + s2t.reason).c_str(),
                s2t.latest_message, s2t.latest_span.begin, s2t.latest_span.end,
                s2t.seed_tokens, s2t.seed_capped ? " (capped)" : "", s2t.latest_tokens,
                (double) s2t.lookback_logit, (double) s2t.vague_logit, (int) s2t.lookback, s2t.earlier_tokens,
                s2t.promoted_tokens, s2t.promoted_capped ? " (capped)" : "", s2t.promoted_runs, s2t.post_tokens,
                (int) s2t.widened, s2t.budget, s2t.budget_widened, (int) s2t.raw_fits, (int) s2t.s2_computed,
                s2t.resume, s2t.history_dropped, s2t.attention_tiles, s2t.pass1_us / 1000.0, s2t.pass2_us / 1000.0);
        }
        std::fflush(stderr);
    }

    if (token_mass_out) {
        // Scoring only (two-scorer selection): return the per-token mass,
        // the probe segments and the ranking rule; the caller selects.
        *token_mass_out = token_mass;
        if (segments_out) *segments_out = segments;
        if (density_out) *density_out = density;
        return ids;
    }

    luce::pflash::PFlashSelectionConfig request_config;
    const bool traced_query = request_probe || ident_mode != luce::pflash::PFlashIdentQuery::Off;
    if (traced_query) {
        request_config = experiment;
        if (request_probe) request_config.request_probe_trace = &request_trace;
        if (ident_mode != luce::pflash::PFlashIdentQuery::Off) {
            request_config.ident_query_trace = &ident_trace;
        }
    }
    return select_pflash_chunks(
        ids, token_mass, select_keep_ratio, n_lookahead, score_query_end,
        /*pool_kernel=*/1, traced_query ? request_config : experiment, pins,
        /*direct_mass=*/true, /*write_trace=*/true,
        segments.empty() ? nullptr : &segments, density,
        /*other_token_scores=*/nullptr, /*split_fraction=*/0.0,
        st.fold_vocab.get());
}

std::vector<int32_t> qwen35_strict_score_and_compress(
    Qwen35DrafterState & st,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int n_lookahead,
    int score_query_end,
    const luce::pflash::PFlashSelectionConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans,
    std::vector<float> * token_mass_out,
    std::vector<PFlashTokenSpan> * segments_out,
    bool * density_out) {
    std::string redo;
    auto kept = strict_score_impl(st, ids, keep_ratio, n_lookahead, score_query_end, experiment,
                                  required_instruction_spans, token_mass_out, segments_out, density_out,
                                  std::numeric_limits<int>::max(), std::string(), redo);
    if (redo.empty()) return kept;
    // A query row the resumed session could not give exactly: the whole request again from token 0 (its session
    // was forgotten, so it is the one reused), never an approximation.
    std::fprintf(stderr, "[qwen35-scorer] redo from token 0: %s\n", redo.c_str());
    std::fflush(stderr);
    const std::string reason = redo;
    kept = strict_score_impl(st, ids, keep_ratio, n_lookahead, score_query_end, experiment,
                             required_instruction_spans, token_mass_out, segments_out, density_out,
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
    // Sessions offloaded when the drafter was last released come back before
    // any scoring (the strict scorer is the only one that reads them).
    st->session_ctx_limit = luce::pflash::pflash_session_ctx_limit(st->ctx_train, ctx.engine_max_ctx);
    qwen35_restore_sessions(ctx, *st, ids);
    // Strict budget selection scores with the block-15 head; the
    // legacy all-layer running-max scorer stays available for legacy
    // selection or when PFLASH_QWEN35_LEGACY_SCORER=1 forces it.
    const char * legacy_scorer = std::getenv("PFLASH_QWEN35_LEGACY_SCORER");
    const bool force_legacy = (legacy_scorer && std::string(legacy_scorer) == "1") ||
        experiment.scorer == luce::pflash::PFlashScorer::Legacy;
    // Only the block-15 head scores keys after the query window; the
    // running-max scorer, alone or in the split, keeps the suffix.
    luce::pflash::PFlashSelectionConfig suffix_kept = experiment;
    suffix_kept.query_suffix_candidates = false;
    if (experiment.selection_active &&
        experiment.scorer == luce::pflash::PFlashScorer::Split) {
        // Two scorers, one budget: the block-15 head ranks (and segments)
        // first, the all-layer running-max scorer fills the remainder.
        std::vector<float> head_mass;
        std::vector<PFlashTokenSpan> head_segments;
        bool head_density = false;
        if (qwen35_strict_score_and_compress(
                *st, ids, keep_ratio, n_lookahead, score_query_end, suffix_kept,
                required_instruction_spans, &head_mass, &head_segments,
                &head_density).empty()) {
            return {};
        }
        std::vector<float> other_scores;
        if (qwen35_score_and_compress(st->weights, ids, keep_ratio, chunk_size,
                                      n_lookahead, pool_kernel, score_query_end,
                                      suffix_kept, required_instruction_spans,
                                      &other_scores).empty()) {
            return {};
        }
        if (other_scores.size() != head_mass.size()) {
            set_last_error("two-scorer selection: score lengths differ");
            return {};
        }
        std::fprintf(stderr,
            "[pflash-select] two-scorer selection: head fraction %.2f, "
            "segments=%s\n", experiment.split_fraction,
            head_segments.empty() ? "fixed" : "probe");
        std::fflush(stderr);
        return select_pflash_chunks(
            ids, head_mass, keep_ratio, n_lookahead, score_query_end,
            /*pool_kernel=*/1, suffix_kept, required_instruction_spans,
            /*direct_mass=*/true, /*write_trace=*/true,
            head_segments.empty() ? nullptr : &head_segments, head_density,
            &other_scores, experiment.split_fraction);
    }
    if (experiment.selection_active && !force_legacy) {
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
    if (st->head_loaded && !experiment.selection_active) {
        set_last_error("Qwen3.5 scoring head requires strict selection");
        return {};
    }
    return qwen35_score_and_compress(st->weights, ids, keep_ratio, chunk_size,
                                     n_lookahead, pool_kernel, score_query_end,
                                     suffix_kept,
                                     required_instruction_spans);
}

} // namespace luce::common
