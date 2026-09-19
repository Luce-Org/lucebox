// Qwen3-0.6B drafter for pflash speculative prefill, hosted in-process.
//
// Wires three pieces:
//   - qwen3_loader.cpp : mmap GGUF + populate ggml tensors on backend
//   - qwen3_graph.cpp  : custom forward (per-layer ggml + FP CUDA kernel)
//   - qwen3_drafter_common.cpp : chunk-top-K + span merge, shared with the
//                                Qwen3.5-0.8B drafter
//
// The Qwen3.5-0.8B drafter lives in qwen35_loader.cpp / qwen35_drafter.cpp;
// this file dispatches to it on DrafterArch.
//
// Single-pass forward at full S using a custom Qwen3-0.6B graph with the
// FlashPrefill block-sparse attention kernel (or BSA when enabled). Tail
// attention scoring runs in a separate post-forward graph using saved Q_last
// and K_curr per layer.
//
// Result running_max [n_lookahead, S] f32 is reduced to per-token scores via
// mean-over-lookahead, smoothed with AvgPool, scored per chunk, top-K kept.

#include "qwen3_drafter.h"
#include "common/dspark_head.h"
#include "qwen3_drafter_model.h"
#include "qwen3_drafter_common.h"
#include "qwen35_drafter.h"
#include "pflash_selection.h"
#include "qwen3/anchor_params.h"
#include "common/backend_precision.h"
#include "common/gguf_inspect.h"
#include "internal.h"
#include "anchor_scan.h"

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

#if defined(DFLASH27B_BACKEND_HIP)
bool prewarm_drafter_once(const Qwen3DrafterWeights & w) {
    static bool warmed = false;
    if (warmed || std::getenv("DFLASH_FP_SKIP_PREWARM")) {
        return true;
    }

    const int warm_tokens = 1024;
    const int n_lookahead = 8;
    std::vector<int32_t> ids((size_t)warm_tokens, 0);
    std::vector<float> running_max;

    auto t0 = std::chrono::steady_clock::now();
    bool ok = forward_qwen3_drafter_model(w, ids, n_lookahead, running_max);
    auto t1 = std::chrono::steady_clock::now();
    if (!ok) {
        return false;
    }

    std::fprintf(stderr, "[drafter] HIP prewarm %.2fs (%d tokens)\n",
                 std::chrono::duration<double>(t1 - t0).count(), warm_tokens);
    std::fflush(stderr);
    warmed = true;
    return true;
}
#endif

} // namespace

bool parse_drafter_arch(const std::string & name, DrafterArch & out) {
    if (name == "qwen3-0.6b" || name == "qwen3_0p6b" || name == "qwen3") {
        out = DrafterArch::Qwen3_0p6b;
        return true;
    }
    if (name == "qwen35-0.8b" || name == "qwen3.5-0.8b" || name == "qwen35_0p8b" || name == "qwen35") {
        out = DrafterArch::Qwen35_0p8b;
        return true;
    }
    return false;
}

const char * drafter_arch_name(DrafterArch arch) {
    switch (arch) {
        case DrafterArch::Qwen3_0p6b: return "qwen3-0.6b";
        case DrafterArch::Qwen35_0p8b: return "qwen35-0.8b";
    }
    return "unknown";
}

bool load_drafter(const std::string & gguf_path, int /*gpu_layers*/,
                  DrafterContext & out) {
    return load_drafter(gguf_path, /*gpu_layers=*/999, /*gpu=*/0, out);
}

bool load_drafter(const std::string & gguf_path, int /*gpu_layers*/,
                  int gpu, DrafterContext & out) {
    DrafterArch arch = DrafterArch::Qwen3_0p6b;
    {
        std::string lower = gguf_path;
        for (auto & c : lower) c = (char)std::tolower((unsigned char)c);
        if (lower.find("qwen3.5") != std::string::npos ||
            lower.find("qwen35")  != std::string::npos) {
            arch = DrafterArch::Qwen35_0p8b;
        }
    }
    return load_drafter(gguf_path, /*gpu_layers=*/999, arch, gpu, out);
}

bool load_drafter(const std::string & gguf_path, int /*gpu_layers*/,
                  DrafterArch arch, DrafterContext & out) {
    return load_drafter(gguf_path, /*gpu_layers=*/999, arch, /*gpu=*/0, out);
}

bool load_drafter(const std::string & gguf_path, int /*gpu_layers*/,
                  DrafterArch arch, int gpu, DrafterContext & out) {
    if (gpu < 0) {
        set_last_error("load_drafter: negative GPU index");
        return false;
    }
    if (out.loaded) {
        set_last_error("drafter already loaded");
        return false;
    }
    if (out.backend && out.gpu >= 0 && out.gpu != gpu) {
        set_last_error("load_drafter: backend already bound to a different GPU");
        return false;
    }

    // If caller didn't supply a backend, spin up our own GPU backend. Sharing
    // would be ideal but we don't have a handle to the daemon's backend
    // through this API. Same-process GPU pools coexist fine; fragmentation is
    // the only cost, and we free everything in free_drafter.
    if (!out.backend) {
        size_t n_dev = ggml_backend_dev_count();
        int seen_gpu = 0;
        for (size_t i = 0; i < n_dev; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                if (seen_gpu == gpu) {
                    out.backend = ggml_backend_dev_init(dev, nullptr);
                    break;
                }
                seen_gpu++;
            }
        }
        if (!out.backend) {
            set_last_error("load_drafter: requested GPU backend unavailable");
            return false;
        }
        out.gpu = gpu;
    } else if (out.gpu < 0) {
        out.gpu = gpu;
    }

    if (arch == DrafterArch::Qwen35_0p8b) {
        return load_qwen35_drafter(gguf_path, arch, out);
    }

    if (!load_qwen3_drafter_model(gguf_path, out.backend, out.weights)) {
        // last_error already set by loader
        return false;
    }

    out.loaded = true;
    out.arch = arch;
    std::fprintf(stderr,
        "[drafter] loaded %s weights=%s compute=%s: n_layer=%d n_head=%d n_kv=%d "
        "n_embd=%d n_ff=%d head_dim=%d vocab=%d gpu=%d\n",
        drafter_arch_name(arch),
        backend_precision_type_name(out.weights.weight_type),
        backend_precision_type_name(out.weights.compute_type),
        out.weights.n_layer, out.weights.n_head, out.weights.n_head_kv,
        out.weights.n_embd, out.weights.n_ff, out.weights.head_dim,
        out.weights.n_vocab, out.gpu);
    std::fflush(stderr);

#if defined(DFLASH27B_BACKEND_HIP)
    if (!prewarm_drafter_once(out.weights)) {
        free_drafter(out);
        return false;
    }
#endif

    return true;
}

void free_drafter(DrafterContext & ctx) {
    dspark_note_drafter_lifecycle();
    free_drafter_weights(ctx);
    if (ctx.backend) {
        ggml_backend_free(ctx.backend);
        ctx.backend = nullptr;
    }
    ctx.gpu = -1;
}

void free_drafter_weights(DrafterContext & ctx) {
    if (ctx.arch == DrafterArch::Qwen35_0p8b && ctx.arch_state) {
        free_qwen35_drafter_state(ctx);
    }
    if (ctx.loaded) {
        if (ctx.arch == DrafterArch::Qwen3_0p6b) {
            free_qwen3_drafter_model(ctx.weights);
        }
    }
    ctx.loaded = false;
}

std::vector<int32_t> drafter_score_and_compress(
    DrafterContext & ctx,
    const std::vector<int32_t> & ids,
    float keep_ratio,
    int chunk_size,
    int n_lookahead,
    int pool_kernel,
    int score_query_end,
    const std::vector<PFlashTokenSpan> & required_instruction_spans) {
    if (!ctx.loaded) {
        set_last_error("drafter not loaded");
        return {};
    }

    dflash::qwen3::PFlashSelectionConfig experiment;
    std::string experiment_error;
    if (!dflash::qwen3::resolve_pflash_selection(
            (int) ids.size(), chunk_size, experiment, experiment_error)) {
        set_last_error("invalid PFlash strict selection config: " + experiment_error);
        std::fprintf(stderr, "[pflash-select] ERROR config: %s\n",
                     experiment_error.c_str());
        std::fflush(stderr);
        return {};
    }
    chunk_size = experiment.chunk_size;
    if (!experiment.selection_active && !required_instruction_spans.empty()) {
        set_last_error(
            "PFlash instruction spans require strict budget selection");
        std::fprintf(stderr,
            "[pflash-select] ERROR instruction spans require strict selection\n");
        std::fflush(stderr);
        return {};
    }
    if (experiment.selection_active) {
        std::string span_error;
        if (!dflash::qwen3::validate_pflash_instruction_spans(
                required_instruction_spans, (int) ids.size(), span_error)) {
            set_last_error("invalid PFlash instruction spans: " + span_error);
            std::fprintf(stderr,
                "[pflash-select] ERROR instruction spans: %s\n",
                span_error.c_str());
            std::fflush(stderr);
            return {};
        }
    }
    if (experiment.configured) {
        std::fprintf(stderr,
            "[pflash-select] config mode=%s active=%d chunk=%d "
            "query_parser=%s query_cap=%d query_actual=%d top_p=%.9g "
            "input=%zu\n",
            dflash::qwen3::pflash_selection_mode_name(experiment.mode),
            (int) experiment.selection_active, experiment.chunk_size,
            dflash::qwen3::pflash_query_parser_name(experiment.query_parser),
            experiment.query_tokens, n_lookahead, experiment.top_p, ids.size());
        std::fflush(stderr);
    }
    if (ctx.arch == DrafterArch::Qwen35_0p8b) {
        if (score_query_end < 0) {
            set_last_error("qwen35 scorer query window out of range");
            return {};
        }
        return qwen35_drafter_score_and_compress(
            ctx, ids, keep_ratio, chunk_size, n_lookahead, pool_kernel,
            score_query_end, experiment, required_instruction_spans);
    }
    const int S = (int)ids.size();
    if (S < n_lookahead + 1) {
        // Too short to score — return as-is.
        return ids;
    }

    // ── 1. Custom forward + GPU tail-attention scoring ────────────────
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> running_max;
    if (!forward_qwen3_drafter_model(
            ctx.weights, ids, n_lookahead, running_max, score_query_end)) {
        return {};
    }
    auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[drafter] forward+score in %.2fs S=%d\n",
        std::chrono::duration<double>(t1 - t0).count(), S);
    std::fflush(stderr);

    // ── 2. Mean over lookahead → per-token score [S] ──────────────────
    std::vector<float> score((size_t)S, 0.0f);
    for (int j = 0; j < S; ++j) {
        float s = 0.0f;
        for (int t = 0; t < n_lookahead; ++t) {
            s += running_max[(size_t)t * S + j];
        }
        score[j] = s / (float)n_lookahead;
    }

    // ── 3. AvgPool 1D smoothing ───────────────────────────────────────
    std::vector<float> smooth((size_t)S, 0.0f);
    int half = pool_kernel / 2;
    for (int j = 0; j < S; ++j) {
        int lo = std::max(0, j - half);
        int hi = std::min(S - 1, j + half);
        float s = 0.0f;
        int n = 0;
        for (int k = lo; k <= hi; ++k) { s += score[k]; ++n; }
        smooth[j] = (n > 0) ? (s / (float)n) : 0.0f;
    }

    if (experiment.selection_active) {
        return select_pflash_chunks(
            ids, ctx.weights.scoring_head_loaded ? score : smooth,
            keep_ratio, n_lookahead, score_query_end,
            ctx.weights.scoring_head_loaded ? 1 : pool_kernel,
            experiment, required_instruction_spans,
            ctx.weights.scoring_head_loaded, true);
    }

    // ── 4. Chunk-top-K + span merge ───────────────────────────────────
    int n_chunks = (S + chunk_size - 1) / chunk_size;
    int n_keep   = std::max(1, (int)((float)n_chunks * keep_ratio));
    std::vector<std::pair<float, int>> chunk_means;
    chunk_means.reserve((size_t)n_chunks);
    for (int c = 0; c < n_chunks; ++c) {
        int s_ = c * chunk_size;
        int e_ = std::min(S, (c + 1) * chunk_size);
        float m = 0.0f;
        for (int j = s_; j < e_; ++j) m += smooth[j];
        m /= std::max(1, e_ - s_);
        chunk_means.push_back({m, c});
    }
    std::sort(chunk_means.begin(), chunk_means.end(),
                      [](auto a, auto b) { return a.first > b.first; });

    // Retrieval tasks often repeat a rare key in the final query and in the
    // needle span. Exact scores alone can keep the query while dropping the
    // neighboring answer chunk, so force a small token-only anchor neighborhood.
    // Head/tail forced chunks scale with n_keep so top-K scoring always gets slots.
    const int h_raw = env_int("DFLASH_COMPRESS_HEAD_CHUNKS", 8);
    const int t_raw = env_int("DFLASH_COMPRESS_TAIL_CHUNKS", 24);
    int head_chunks = h_raw, tail_chunks = t_raw;
    if (head_chunks + tail_chunks >= n_keep) {
        const int budget = std::max(1, n_keep - 1);
        head_chunks = std::max(0, h_raw * budget / (h_raw + t_raw));
        tail_chunks = std::max(0, budget - head_chunks);
    }
    const int query_tokens = env_int("DFLASH_COMPRESS_QUERY_TOKENS", 96);
    const auto ap = resolve_anchor_params(n_chunks,
        env_int("PFLASH_COMPRESS_ANCHOR_RADIUS",   -1),
        env_int("PFLASH_COMPRESS_MAX_ANCHOR_HITS", -1),
        env_int("DFLASH_COMPRESS_ANCHOR_RADIUS",   -1),
        env_int("DFLASH_COMPRESS_MAX_ANCHOR_HITS", -1));
    const int anchor_radius   = ap.radius;
    const int max_anchor_hits = ap.max_hits;
    std::vector<uint8_t> selected_mask((size_t)n_chunks, 0);
    std::vector<uint8_t> forced((size_t)n_chunks, 0);
    for (int c = 0; c < std::min(n_chunks, head_chunks); ++c) forced[(size_t)c] = 1;
    for (int c = std::max(0, n_chunks - tail_chunks); c < n_chunks; ++c) forced[(size_t)c] = 1;

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

    int selected_count = 0;
    int forced_count = 0;
    for (int c = 0; c < n_chunks; ++c) {
        if (forced[(size_t)c]) {
            selected_mask[(size_t)c] = 1;
            ++selected_count;
            ++forced_count;
        }
    }
    for (const auto & cm : chunk_means) {
        if (selected_count >= n_keep) break;
        int c = cm.second;
        if (!selected_mask[(size_t)c]) {
            selected_mask[(size_t)c] = 1;
            ++selected_count;
        }
    }

    std::vector<int> selected;
    selected.reserve((size_t)selected_count);
    for (int c = 0; c < n_chunks; ++c) {
        if (selected_mask[(size_t)c]) selected.push_back(c);
    }

    std::vector<int32_t> out;
    out.reserve((size_t)n_keep * chunk_size + 16);
    int span_start = -1, span_end = -1;
    for (int c : selected) {
        int s_ = c * chunk_size;
        int e_ = std::min(S, (c + 1) * chunk_size);
        if (span_start < 0) {
            span_start = s_; span_end = e_;
        } else if (s_ == span_end) {
            span_end = e_;
        } else {
            for (int j = span_start; j < span_end; ++j) out.push_back(ids[j]);
            span_start = s_; span_end = e_;
        }
    }
    if (span_start >= 0) {
        for (int j = span_start; j < span_end; ++j) out.push_back(ids[j]);
    }

    auto t2 = std::chrono::steady_clock::now();
    std::fprintf(stderr,
        "[drafter] score_and_compress total %.2fs S=%d kept=%zu (%d/%d chunks, forced=%d)\n",
        std::chrono::duration<double>(t2 - t0).count(),
        S, out.size(), (int)selected.size(), n_chunks, forced_count);
    std::fflush(stderr);

    const int query_end = score_query_end < 0 ? S : score_query_end;
    const int query_begin = query_end - n_lookahead;
    const int token_budget = (int) std::floor(
        (double) S * (double) keep_ratio);
    const PFlashTraceFields trace_fields{
        &ids, query_begin, query_end, experiment.mode,
        experiment.query_parser, token_budget,
        dflash::qwen3::PFlashSelectionStop::InvalidInput, (int) out.size(),
        0.0, nullptr};
    write_compression_trace(S, keep_ratio, chunk_size, n_lookahead,
        pool_kernel, n_keep, chunk_means, selected_mask, forced, out,
        &trace_fields);

    return out;
}

} // namespace dflash::common
