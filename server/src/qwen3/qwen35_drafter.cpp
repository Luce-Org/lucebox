// Qwen3.5-0.8B drafter scoring for pflash speculative prefill.
//
// Two scorers share these weights:
//   - qwen35_score_and_compress        : the original all-layer running-max
//                                        scorer, on the Qwen3.5 architecture
//   - qwen35_strict_score_and_compress : blocks 0..14 plus the block-15 NoPE
//                                        Q/K scoring head, under strict
//                                        budget selection
//
// Loading lives in qwen35_loader.cpp; qwen3_drafter.cpp dispatches into
// qwen35_drafter_score_and_compress on DrafterArch::Qwen35_0p8b.

#include "qwen35_drafter.h"

#include "qwen3_drafter.h"
#include "qwen3_drafter_common.h"
#include "pflash_selection.h"
#include "common/gguf_inspect.h"
#include "qwen3/anchor_params.h"
#include "internal.h"

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
#include <string>
#include <unordered_map>
#include <vector>

namespace dflash::common {

namespace {

static constexpr uint16_t F16_ZERO = 0x0000;
static constexpr uint16_t F16_NEG_INF = 0xFC00;

static int align_up_i(int x, int a) { return ((x + a - 1) / a) * a; }

static void build_causal_mask_f16(std::vector<uint16_t> & out, int kv_len, int n_tokens, int kv_start) {
    const int kv_pad = align_up_i(kv_len, 32);
    const int q_pad = align_up_i(n_tokens, 32);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    static_assert(F16_ZERO == 0, "visible mask entries are zero-filled with memset");
    for (int q = 0; q < n_tokens; ++q) {
        const int visible = std::min(kv_len, kv_start + q + 1);
        if (visible > 0) {
            std::memset(out.data() + (size_t)q * kv_pad, 0, (size_t)visible * sizeof(uint16_t));
        }
    }
}

// create_target_cache honours DFLASH27B_KV_TQ3; the drafter cache never wants
// the TurboQuant rotation, so force it off while the cache is created.
struct ScopedKvTq3Off {
    ScopedKvTq3Off() {
#if defined(_WIN32)
        char * raw = nullptr;
        size_t len = 0;
        _dupenv_s(&raw, &len, "DFLASH27B_KV_TQ3");
        had_ = raw != nullptr;
        old_ = had_ ? raw : "";
        free(raw);
        _putenv_s("DFLASH27B_KV_TQ3", "0");
#else
        const char * raw = std::getenv("DFLASH27B_KV_TQ3");
        had_ = raw != nullptr;
        old_ = had_ ? raw : "";
        setenv("DFLASH27B_KV_TQ3", "0", 1);
#endif
    }
    ~ScopedKvTq3Off() {
#if defined(_WIN32)
        // _putenv_s with empty value removes the variable on MSVCRT.
        _putenv_s("DFLASH27B_KV_TQ3", had_ ? old_.c_str() : "");
#else
        if (had_) setenv("DFLASH27B_KV_TQ3", old_.c_str(), 1);
        else unsetenv("DFLASH27B_KV_TQ3");
#endif
    }
    bool had_ = false;
    std::string old_;
};

} // namespace

std::vector<int32_t> qwen35_score_and_compress(
    TargetWeights & w,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel,
    int score_query_end,
    const dflash::qwen3::PFlashSelectionConfig & experiment,
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
    {
        ScopedKvTq3Off tq3_off;
        if (!create_target_cache(w, S, 0, w.backend, cache, true)) {
            return {};
        }
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
    for (int il = 0; il < w.n_layer; ++il) {
        const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);
        int fa_idx = 0;
        if (is_attn) {
            for (int k = 0; k < il; ++k) if (((k + 1) % w.full_attention_interval) == 0) ++fa_idx;
        }
        for (int start = 0; start < S; start += ubatch) {
            const int n = std::min(ubatch, S - start);
            const int kv_len = start + n;

            ggml_init_params ip{};
            ip.mem_size = 512 * 1024 * 1024;
            ip.no_alloc = true;
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 drafter layer graph ctx init failed");
                return {};
            }
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
            ggml_tensor * inp = ggml_view_2d(ctx, act_in, hidden, n, act_in->nb[1], (size_t)start * act_in->nb[1]);
            ggml_tensor * pos = nullptr;
            ggml_tensor * mask = nullptr;
            if (is_attn) {
                pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n);
                ggml_set_input(pos);
                mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, align_up_i(kv_len, 32), align_up_i(n, 32));
                ggml_set_input(mask);
            }
            ggml_tensor * out = build_qwen35_layer(ctx, gf, w, cache, il, inp, pos, mask, start, n, false, 0);
            ggml_tensor * dst = ggml_view_2d(ctx, act_out, hidden, n, act_out->nb[1], (size_t)start * act_out->nb[1]);
            if (ggml_nelements(out) != ggml_nelements(dst)) {
                std::fprintf(stderr,
                    "[qwen35-drafter] layer output shape mismatch il=%d start=%d out=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld]\n",
                    il, start,
                    (long long)out->ne[0], (long long)out->ne[1], (long long)out->ne[2], (long long)out->ne[3],
                    (long long)dst->ne[0], (long long)dst->ne[1], (long long)dst->ne[2], (long long)dst->ne[3]);
                ggml_free(ctx); ggml_gallocr_free(alloc); ggml_backend_buffer_free(act_buf); ggml_free(act_ctx); free_target_cache(cache);
                set_last_error("qwen35 layer output shape mismatch");
                return {};
            }
            ggml_build_forward_expand(gf, ggml_cpy(ctx, out, dst));
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
                std::vector<uint16_t> m;
                build_causal_mask_f16(m, kv_len, n, start);
                ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(uint16_t));
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
        : std::max(3, env_int("DFLASH_COMPRESS_POOL_KERNEL", 5));
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
        const int h_raw = env_int("DFLASH_COMPRESS_HEAD_CHUNKS", 8);
        const int t_raw = env_int("DFLASH_COMPRESS_TAIL_CHUNKS", 24);
        int h_n = h_raw, t_n = t_raw;
        if (h_n + t_n >= n_keep) {
            const int budget = std::max(1, n_keep - 1);
            h_n = std::max(0, h_raw * budget / (h_raw + t_raw));
            t_n = std::max(0, budget - h_n);
        }
        for (int c = 0; c < std::min(n_chunks, h_n); ++c) { selected[(size_t)c] = 1; ++count; }
        for (int c = std::max(0, n_chunks - t_n); c < n_chunks; ++c) if (!selected[(size_t)c]) { selected[(size_t)c] = 1; ++count; }
    }

    const int query_tokens = env_int("DFLASH_COMPRESS_QUERY_TOKENS", 96);
    const auto ap = resolve_anchor_params(n_chunks,
        env_int("PFLASH_COMPRESS_ANCHOR_RADIUS",   -1),
        env_int("PFLASH_COMPRESS_MAX_ANCHOR_HITS", -1),
        env_int("DFLASH_COMPRESS_ANCHOR_RADIUS",   -1),
        env_int("DFLASH_COMPRESS_MAX_ANCHOR_HITS", -1));
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
    const int repeat_min = env_int("DFLASH_COMPRESS_REPEAT_MIN", 4);
    const int repeat_max = env_int("DFLASH_COMPRESS_REPEAT_MAX", 32);
    const int repeat_limit = env_int("DFLASH_COMPRESS_REPEAT_CHUNKS", n_keep);
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

// Scoring-head selection for the Qwen3.5-0.8B drafter: run blocks 0..14, then
// score every context token against the query window with block 15's NoPE
// Q/K (or a trained replacement) and select chunks by attention mass. This
// is the runtime counterpart of the Python retention screen (trial 0075).
std::vector<int32_t> qwen35_strict_score_and_compress(
    Qwen35DrafterState & st,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int n_lookahead,
    int score_query_end,
    const dflash::qwen3::PFlashSelectionConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans,
    std::vector<float> * token_mass_out,
    std::vector<PFlashTokenSpan> * segments_out,
    bool * density_out) {

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

    auto t0 = std::chrono::steady_clock::now();
    TargetCache cache;
    {
        ScopedKvTq3Off tq3_off;
        if (!create_target_cache(w, S, 0, w.backend, cache, true)) {
            return {};
        }
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
    auto cleanup = [&]() {
        ggml_backend_buffer_free(act_buf);
        ggml_free(act_ctx);
        free_target_cache(cache);
    };

    {
        const int batch = 2048;
        std::vector<float> emb((size_t)hidden * batch);
        for (int i = 0; i < S; i += batch) {
            const int n = std::min(batch, S - i);
            if (!w.embedder.embed(ids.data() + i, n, emb.data())) {
                cleanup();
                set_last_error("qwen35 drafter embedding failed");
                return {};
            }
            ggml_backend_tensor_set(act_in, emb.data(), (size_t)i * act_in->nb[1],
                                    (size_t)hidden * n * sizeof(float));
        }
    }

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
    const int ubatch = 1024;
    std::vector<uint16_t> mask_bits;
    for (int il = 0; il < kQwen35HeadBlock; ++il) {
        const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);
        for (int start = 0; start < S; start += ubatch) {
            const int n = std::min(ubatch, S - start);
            const int kv_len = start + n;
            ggml_init_params ip{};
            ip.mem_size = 512 * 1024 * 1024;
            ip.no_alloc = true;
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                ggml_gallocr_free(alloc); cleanup();
                set_last_error("qwen35 drafter layer graph ctx init failed");
                return {};
            }
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
            ggml_tensor * inp = ggml_view_2d(ctx, act_in, hidden, n, act_in->nb[1],
                                             (size_t)start * act_in->nb[1]);
            ggml_tensor * pos = nullptr;
            ggml_tensor * mask = nullptr;
            if (is_attn) {
                pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n);
                ggml_set_input(pos);
                mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,
                                          align_up_i(kv_len, 32), align_up_i(n, 32));
                ggml_set_input(mask);
            }
            ggml_tensor * out = build_qwen35_layer(ctx, gf, w, cache, il, inp, pos, mask,
                                                   start, n, false, 0);
            ggml_tensor * dst = ggml_view_2d(ctx, act_out, hidden, n, act_out->nb[1],
                                             (size_t)start * act_out->nb[1]);
            if (ggml_nelements(out) != ggml_nelements(dst)) {
                ggml_free(ctx); ggml_gallocr_free(alloc); cleanup();
                set_last_error("qwen35 layer output shape mismatch");
                return {};
            }
            ggml_build_forward_expand(gf, ggml_cpy(ctx, out, dst));
            if (!ggml_gallocr_alloc_graph(alloc, gf)) {
                ggml_free(ctx); ggml_gallocr_free(alloc); cleanup();
                set_last_error("qwen35 drafter graph allocation failed");
                return {};
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
                build_causal_mask_f16(mask_bits, kv_len, n, start);
                ggml_backend_tensor_set(mask, mask_bits.data(), 0,
                                        mask_bits.size() * sizeof(uint16_t));
            }
            const auto status = ggml_backend_graph_compute(w.backend, gf);
            ggml_free(ctx);
            if (status != GGML_STATUS_SUCCESS) {
                ggml_gallocr_free(alloc); cleanup();
                set_last_error("qwen35 drafter graph compute failed");
                return {};
            }
        }
        std::swap(act_in, act_out);
    }
    ggml_gallocr_free(alloc);
    auto t1 = std::chrono::steady_clock::now();

    // Block-15 NoPE Q/K scoring: softmax over keys before the query window,
    // then mean over heads and query tokens. The query never scores itself.
    // Keys are projected in chunks so no intermediate tensor puts the
    // sequence length into a HIP grid y/z dimension (65,535 limit); the
    // logits land in one [S, n_lookahead, H] buffer for a single softmax.
    const int key_chunk = 8192;
    const int n_key_chunks = (S + key_chunk - 1) / key_chunk;
    ggml_init_params lip{};
    lip.mem_size = (size_t)8 * ggml_tensor_overhead() + 4096;
    lip.no_alloc = true;
    ggml_context * lctx = ggml_init(lip);
    if (!lctx) {
        cleanup();
        set_last_error("qwen35 score buffer ctx allocation failed");
        return {};
    }
    ggml_tensor * logits = ggml_new_tensor_3d(lctx, GGML_TYPE_F32, S, n_lookahead, H);
    ggml_tensor * mask = ggml_new_tensor_2d(lctx, GGML_TYPE_F32, S, n_lookahead);
    const bool use_probe = st.probe_loaded &&
        experiment.segmentation != dflash::qwen3::PFlashSegmentation::Fixed;
    ggml_tensor * probe_logits = use_probe
        ? ggml_new_tensor_1d(lctx, GGML_TYPE_F32, S) : nullptr;
    ggml_tensor * subunit_logits = use_probe && st.probe_sub_fc2_w
        ? ggml_new_tensor_1d(lctx, GGML_TYPE_F32, S) : nullptr;
    ggml_backend_buffer_t lbuf = ggml_backend_alloc_ctx_tensors(lctx, w.backend);
    if (!lbuf) {
        ggml_free(lctx); cleanup();
        set_last_error("qwen35 score buffer allocation failed");
        return {};
    }
    {
        std::vector<float> m((size_t)n_lookahead * S, -INFINITY);
        for (int t = 0; t < n_lookahead; ++t) {
            std::fill_n(m.begin() + (size_t)t * S, (size_t)query_start, 0.0f);
        }
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));
    }
    ggml_init_params sip{};
    sip.mem_size = ggml_tensor_overhead() * (size_t)(64 + 24 * n_key_chunks) +
                   ggml_graph_overhead_custom(4096, false) + 64 * 1024;
    sip.no_alloc = true;
    ggml_context * sctx = ggml_init(sip);
    if (!sctx) {
        ggml_backend_buffer_free(lbuf); ggml_free(lctx); cleanup();
        set_last_error("qwen35 score graph ctx allocation failed");
        return {};
    }
    ggml_cgraph * sgf = ggml_new_graph_custom(sctx, 4096, false);
    ggml_tensor * wk_src = st.head_loaded ? st.head_wk : L.wk;
    ggml_tensor * x_q = ggml_view_2d(sctx, act_in, hidden, n_lookahead, act_in->nb[1],
                                     (size_t)query_start * act_in->nb[1]);
    ggml_tensor * q_in = ggml_mul(sctx, ggml_rms_norm(sctx, x_q, w.rms_eps), L.attn_norm);
    ggml_tensor * Q = nullptr;
    if (st.head_loaded) {
        Q = ggml_reshape_3d(sctx, ggml_mul_mat(sctx, st.head_wq, q_in), D, H, n_lookahead);
    } else {
        // Native block 15 packs query and gate rows per head; keep the query half.
        ggml_tensor * QG = ggml_reshape_3d(sctx, ggml_mul_mat(sctx, L.wq, q_in),
                                           D * 2, H, n_lookahead);
        Q = ggml_view_3d(sctx, QG, D, H, n_lookahead,
                         ggml_element_size(QG) * D * 2,
                         ggml_element_size(QG) * D * 2 * H, 0);
    }
    Q = ggml_mul(sctx, ggml_rms_norm(sctx, Q, w.rms_eps), L.q_norm);
    ggml_tensor * Q_perm = ggml_cont(sctx, ggml_permute(sctx, Q, 0, 2, 1, 3));  // [D, n_lookahead, H]
    for (int b = 0; b < S; b += key_chunk) {
        const int n = std::min(key_chunk, S - b);
        ggml_tensor * x_c = ggml_view_2d(sctx, act_in, hidden, n, act_in->nb[1],
                                         (size_t)b * act_in->nb[1]);
        ggml_tensor * x_norm = ggml_mul(sctx, ggml_rms_norm(sctx, x_c, w.rms_eps), L.attn_norm);
        ggml_tensor * K = ggml_reshape_3d(sctx, ggml_mul_mat(sctx, wk_src, x_norm), D, Hk, n);
        K = ggml_mul(sctx, ggml_rms_norm(sctx, K, w.rms_eps), L.k_norm);
        K = ggml_cont(sctx, ggml_permute(sctx, K, 0, 2, 1, 3));  // [D, n, Hk]
        ggml_tensor * K_score = K;
        if (H != Hk) {
            const int gqa = H / Hk;
            ggml_tensor * K_4d = ggml_reshape_4d(sctx, K, D, n, 1, Hk);
            ggml_tensor * K_tpl = ggml_new_tensor_4d(sctx, GGML_TYPE_F32, D, n, gqa, Hk);
            K_score = ggml_reshape_3d(sctx, ggml_repeat(sctx, K_4d, K_tpl), D, n, H);
        }
        ggml_tensor * part = ggml_mul_mat(sctx, K_score, Q_perm);  // [n, n_lookahead, H]
        ggml_tensor * dst = ggml_view_3d(sctx, logits, n, n_lookahead, H,
                                         logits->nb[1], logits->nb[2],
                                         (size_t)b * logits->nb[0]);
        ggml_build_forward_expand(sgf, ggml_cpy(sctx, part, dst));
        if (use_probe) {
            // Segment probe on the same tap: LayerNorm -> fc1 -> GELU trunk,
            // then one fc2 row per head (unit always; sub-unit when shipped).
            ggml_tensor * p = ggml_norm(sctx, x_c, st.probe_norm_eps);
            p = ggml_add(sctx, ggml_mul(sctx, p, st.probe_norm_w), st.probe_norm_b);
            p = ggml_gelu(sctx, ggml_add(sctx, ggml_mul_mat(sctx, st.probe_fc1_w, p),
                                         st.probe_fc1_b));                      // [width, n]
            ggml_tensor * unit = ggml_add(sctx, ggml_mul_mat(sctx, st.probe_fc2_w, p),
                                          st.probe_fc2_b);                      // [1, n]
            ggml_tensor * p_dst = ggml_view_1d(sctx, probe_logits, n,
                                               (size_t)b * ggml_element_size(probe_logits));
            ggml_build_forward_expand(sgf, ggml_cpy(sctx, ggml_reshape_1d(sctx, unit, n), p_dst));
            if (subunit_logits) {
                ggml_tensor * sub = ggml_add(sctx,
                    ggml_mul_mat(sctx, st.probe_sub_fc2_w, p), st.probe_sub_fc2_b);
                ggml_tensor * s_dst = ggml_view_1d(sctx, subunit_logits, n,
                    (size_t)b * ggml_element_size(subunit_logits));
                ggml_build_forward_expand(sgf,
                    ggml_cpy(sctx, ggml_reshape_1d(sctx, sub, n), s_dst));
            }
        }
    }
    ggml_tensor * probs = ggml_soft_max_ext(sctx, logits, mask,
                                            1.0f / std::sqrt((float)D), 0.0f);
    ggml_set_output(probs);
    ggml_build_forward_expand(sgf, probs);
    ggml_gallocr_t salloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
    if (!ggml_gallocr_alloc_graph(salloc, sgf)) {
        ggml_gallocr_free(salloc); ggml_free(sctx);
        ggml_backend_buffer_free(lbuf); ggml_free(lctx); cleanup();
        set_last_error("qwen35 score graph allocation failed");
        return {};
    }
    const auto score_status = ggml_backend_graph_compute(w.backend, sgf);
    if (score_status != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(salloc); ggml_free(sctx);
        ggml_backend_buffer_free(lbuf); ggml_free(lctx); cleanup();
        set_last_error("qwen35 score graph compute failed");
        return {};
    }
    std::vector<float> probs_h((size_t)S * n_lookahead * H);
    ggml_backend_tensor_get(probs, probs_h.data(), 0, probs_h.size() * sizeof(float));
    std::vector<float> probe_raw;
    std::vector<float> subunit_raw;
    if (use_probe) {
        probe_raw.resize((size_t) S);
        ggml_backend_tensor_get(probe_logits, probe_raw.data(), 0, probe_raw.size() * sizeof(float));
        if (subunit_logits) {
            subunit_raw.resize((size_t) S);
            ggml_backend_tensor_get(subunit_logits, subunit_raw.data(), 0,
                                    subunit_raw.size() * sizeof(float));
        }
    }
    ggml_gallocr_free(salloc);
    ggml_free(sctx);
    ggml_backend_buffer_free(lbuf);
    ggml_free(lctx);
    cleanup();
    const size_t nonfinite = count_nonfinite_scores(probs_h.data(), probs_h.size());
    if (nonfinite != 0) {
        const std::string message =
            "non-finite Qwen3.5 scoring-head scores: " + std::to_string(nonfinite) +
            "/" + std::to_string(probs_h.size());
        std::fprintf(stderr, "[pflash] ERROR: %s\n", message.c_str());
        std::fflush(stderr);
        set_last_error(message);
        return {};
    }
    std::vector<float> token_mass;
    scoring_head_mean_token_mass(probs_h.data(), S, n_lookahead, H, token_mass);
    auto t2 = std::chrono::steady_clock::now();
    std::fprintf(stderr,
        "[qwen35-scorer] forward %.2fs (blocks 0-%d, S=%d) score %.2fs "
        "total %.2fs head=%s\n",
        std::chrono::duration<double>(t1 - t0).count(), kQwen35HeadBlock - 1, S,
        std::chrono::duration<double>(t2 - t1).count(),
        std::chrono::duration<double>(t2 - t0).count(),
        st.head_loaded ? "trained" : "native-block15");
    std::fflush(stderr);

    std::vector<PFlashTokenSpan> segments;
    bool density = experiment.candidate_score == dflash::qwen3::PFlashCandidateScore::Density;
    if (use_probe) {
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
        for (const auto & span : required_instruction_spans) {
            forced.push_back(span.begin);
            forced.push_back(span.end);
        }
        int boundaries_in_context = 0;
        for (int t = 1; t < query_begin; ++t) {
            if (boundary[(size_t) t] > st.probe_threshold) ++boundaries_in_context;
        }
        const bool forced_probe =
            experiment.segmentation == dflash::qwen3::PFlashSegmentation::Probe;
        if (boundaries_in_context >= 4 || forced_probe) {
            segments = dflash::qwen3::pflash_probe_segments(
                boundary, S, st.probe_threshold, st.probe_min_segment,
                st.probe_max_segment, forced, split_scores);
        }
        if (segments.empty()) {
            std::fprintf(stderr,
                "[qwen35-segment-probe] %d boundaries in the context, "
                "falling back to fixed %d-token chunks\n",
                boundaries_in_context, experiment.chunk_size);
        } else {
            if (experiment.candidate_score == dflash::qwen3::PFlashCandidateScore::Auto) {
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
    }

    if (token_mass_out) {
        // Scoring only (two-scorer selection): return the per-token mass,
        // the probe segments and the ranking rule; the caller selects.
        *token_mass_out = token_mass;
        if (segments_out) *segments_out = segments;
        if (density_out) *density_out = density;
        return ids;
    }

    return select_pflash_chunks(
        ids, token_mass, keep_ratio, n_lookahead, score_query_end,
        /*pool_kernel=*/1, experiment, required_instruction_spans,
        /*direct_mass=*/true, /*write_trace=*/true,
        segments.empty() ? nullptr : &segments, density);
}

std::vector<int32_t> qwen35_drafter_score_and_compress(
    DrafterContext & ctx,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel,
    int score_query_end,
    const dflash::qwen3::PFlashSelectionConfig & experiment,
    const std::vector<PFlashTokenSpan> & required_instruction_spans) {
    if (!ctx.arch_state) {
        set_last_error("qwen35 drafter state missing");
        return {};
    }
    auto * st = static_cast<Qwen35DrafterState *>(ctx.arch_state);
    // Strict budget selection scores with the block-15 head; the
    // legacy all-layer running-max scorer stays available for legacy
    // selection or when PFLASH_QWEN35_LEGACY_SCORER=1 forces it.
    const char * legacy_scorer = std::getenv("PFLASH_QWEN35_LEGACY_SCORER");
    const bool force_legacy = (legacy_scorer && std::string(legacy_scorer) == "1") ||
        experiment.scorer == dflash::qwen3::PFlashScorer::Legacy;
    if (experiment.selection_active &&
        experiment.scorer == dflash::qwen3::PFlashScorer::Split) {
        // Two scorers, one budget: the block-15 head ranks (and segments)
        // first, the all-layer running-max scorer fills the remainder.
        std::vector<float> head_mass;
        std::vector<PFlashTokenSpan> head_segments;
        bool head_density = false;
        if (qwen35_strict_score_and_compress(
                *st, ids, keep_ratio, n_lookahead, score_query_end, experiment,
                required_instruction_spans, &head_mass, &head_segments,
                &head_density).empty()) {
            return {};
        }
        std::vector<float> other_scores;
        if (qwen35_score_and_compress(st->weights, ids, keep_ratio, chunk_size,
                                      n_lookahead, pool_kernel, score_query_end,
                                      experiment, required_instruction_spans,
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
            /*pool_kernel=*/1, experiment, required_instruction_spans,
            /*direct_mass=*/true, /*write_trace=*/true,
            head_segments.empty() ? nullptr : &head_segments, head_density,
            &other_scores, experiment.split_fraction);
    }
    if (experiment.selection_active && !force_legacy) {
        return qwen35_strict_score_and_compress(
            *st, ids, keep_ratio, n_lookahead, score_query_end, experiment,
            required_instruction_spans);
    }
    if (st->head_loaded && !experiment.selection_active) {
        set_last_error("Qwen3.5 scoring head requires strict selection");
        return {};
    }
    return qwen35_score_and_compress(st->weights, ids, keep_ratio, chunk_size,
                                     n_lookahead, pool_kernel, score_query_end,
                                     experiment,
                                     required_instruction_spans);
}

} // namespace dflash::common
