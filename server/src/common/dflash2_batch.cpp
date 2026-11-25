#include "dflash2_head.h"

#include "dflash2_selector_validation.h"
#include "ddtree.h"
#include "geometric_draft_topk_cuda.h"
#include "pinned_stage.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace luce::common {
namespace {

constexpr size_t kMaxCachedGraphs = 16;

struct ProjectionGraph {
    const DraftWeights * dw = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_tensor * lm_head = nullptr;
    int n_positions = 0;
    std::vector<uint8_t> arena;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor * inp_hidden = nullptr;
    ggml_tensor * logits = nullptr;
    ggml_tensor * hproj = nullptr;

    ~ProjectionGraph() {
        if (galloc) {
            ggml_gallocr_free(galloc);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

struct BatchedSelectorGraph {
    const DraftWeights * dw = nullptr;
    ggml_backend_t backend = nullptr;
    int n_lanes = 0;
    int n_cand = 0;
    int K = 0;
    std::vector<uint8_t> arena;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor * inp_succ = nullptr;
    ggml_tensor * inp_pred = nullptr;
    ggml_tensor * succ = nullptr;
    ggml_tensor * pred = nullptr;

    ~BatchedSelectorGraph() {
        if (galloc) {
            ggml_gallocr_free(galloc);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

struct SelectorCache {
    uint64_t generation = 0;
    std::vector<std::unique_ptr<ProjectionGraph>> projections;
    std::vector<std::unique_ptr<BatchedSelectorGraph>> selectors;
    PinnedStage stage;
    std::vector<float> candidate_log_probs;
    std::vector<int32_t> candidate_ids;
    std::vector<int32_t> predecessor_ids;
    std::vector<float> projected_hidden;
    std::vector<float> successor_codes;
    std::vector<float> predecessor_codes;

    void sync_generation(uint64_t gen) {
        if (generation != gen) {
            projections.clear();
            selectors.clear();
            generation = gen;
        }
    }
};

// Allocate a new graph's buffers for `cached`, the cache it joins: past
// capacity its coldest entry goes first, and while the device cannot hold the
// new graph the coldest cached ones are freed and the allocation retried, so
// graphs kept for other shapes never stop a new shape from fitting.
template <typename Graph>
bool alloc_cached_graph(std::vector<std::unique_ptr<Graph>> & cached,
                        Graph & graph, ggml_backend_t backend) {
    if (cached.size() >= kMaxCachedGraphs) cached.erase(cached.begin());
    for (;;) {
        graph.galloc =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (graph.galloc && ggml_gallocr_alloc_graph(graph.galloc, graph.gf)) {
            return true;
        }
        if (graph.galloc) ggml_gallocr_free(graph.galloc);
        graph.galloc = nullptr;
        if (cached.empty()) return false;
        cached.erase(cached.begin());
    }
}

// The cached graph `match` accepts, moved to the hot end; null when none does.
template <typename Graph, typename Match>
Graph * find_cached_graph(std::vector<std::unique_ptr<Graph>> & cached, Match match) {
    for (auto it = cached.begin(); it != cached.end(); ++it) {
        if (match(**it)) {
            std::rotate(it, it + 1, cached.end());
            return cached.back().get();
        }
    }
    return nullptr;
}

SelectorCache & selector_cache() {
    static thread_local SelectorCache cache;
    return cache;
}

ProjectionGraph * ensure_projection_graph(
        SelectorCache & cache, const DraftWeights & dw,
        ggml_backend_t backend, ggml_tensor * lm_head, int n_positions) {
    cache.sync_generation(dflash2_selector_graph_generation());
    if (ProjectionGraph * hit = find_cached_graph(cache.projections, [&](const ProjectionGraph & g) {
            return g.dw == &dw && g.backend == backend && g.lm_head == lm_head &&
                   g.n_positions == n_positions;
        })) {
        return hit;
    }
    const DraftSelectorWeights & selector = dw.selector;
    if (!backend || !lm_head || n_positions <= 0 || dw.n_embd <= 0 ||
        lm_head->ne[0] != dw.n_embd || lm_head->ne[1] <= 0 ||
        !selector.hproj) {
        return nullptr;
    }

    auto graph = std::make_unique<ProjectionGraph>();
    const size_t arena_size =
        ggml_tensor_overhead() * 48 +
        ggml_graph_overhead_custom(256, false) + 4096;
    graph->arena.assign(arena_size, 0);
    ggml_init_params params{};
    params.mem_size = graph->arena.size();
    params.mem_buffer = graph->arena.data();
    params.no_alloc = true;
    graph->ctx = ggml_init(params);
    if (!graph->ctx) return nullptr;
    graph->gf = ggml_new_graph_custom(graph->ctx, 256, false);
    graph->inp_hidden = ggml_new_tensor_2d(
        graph->ctx, GGML_TYPE_F32, dw.n_embd, n_positions);
    ggml_set_input(graph->inp_hidden);
    // Fuse lm_head and selector.hproj into one launch so inp_hidden is copied
    // only once per draft round.
    graph->logits = ggml_mul_mat(graph->ctx, lm_head, graph->inp_hidden);
    graph->hproj  = ggml_mul_mat(graph->ctx, selector.hproj, graph->inp_hidden);
    ggml_set_output(graph->logits);
    ggml_set_output(graph->hproj);
    ggml_build_forward_expand(graph->gf, graph->logits);
    ggml_build_forward_expand(graph->gf, graph->hproj);
    if (!alloc_cached_graph(cache.projections, *graph, backend)) {
        std::fprintf(stderr,
            "dflash2_select_chains_batched: projection graph alloc failed\n");
        return nullptr;
    }
    graph->dw = &dw;
    graph->backend = backend;
    graph->lm_head = lm_head;
    graph->n_positions = n_positions;
    cache.projections.push_back(std::move(graph));
    return cache.projections.back().get();
}

BatchedSelectorGraph * ensure_selector_graph(
        SelectorCache & cache, const DraftWeights & dw,
        ggml_backend_t backend, int n_lanes, int n_cand, int K) {
    cache.sync_generation(dflash2_selector_graph_generation());
    if (BatchedSelectorGraph * hit = find_cached_graph(cache.selectors, [&](const BatchedSelectorGraph & g) {
            return g.dw == &dw && g.backend == backend && g.n_lanes == n_lanes &&
                   g.n_cand == n_cand && g.K == K;
        })) {
        return hit;
    }
    const DraftSelectorWeights & selector = dw.selector;
    if (!backend || n_lanes <= 0 || n_cand <= 0 || K <= 0 ||
        dw.n_embd <= 0 || selector.rank <= 0 || !selector.hproj ||
        !selector.pred_cb || !selector.succ_cb) {
        return nullptr;
    }

    auto graph = std::make_unique<BatchedSelectorGraph>();
    const int n_positions = n_lanes * n_cand;
    const int n_pred_rows = n_lanes + n_positions * K;
    const size_t arena_size =
        ggml_tensor_overhead() * 32 +
        ggml_graph_overhead_custom(256, false) + 4096;
    graph->arena.assign(arena_size, 0);
    ggml_init_params params{};
    params.mem_size = graph->arena.size();
    params.mem_buffer = graph->arena.data();
    params.no_alloc = true;
    graph->ctx = ggml_init(params);
    if (!graph->ctx) return nullptr;
    graph->gf = ggml_new_graph_custom(graph->ctx, 256, false);
    graph->inp_succ = ggml_new_tensor_1d(
        graph->ctx, GGML_TYPE_I32, int64_t(n_positions) * K);
    graph->inp_pred = ggml_new_tensor_1d(
        graph->ctx, GGML_TYPE_I32, n_pred_rows);
    ggml_set_input(graph->inp_succ);
    ggml_set_input(graph->inp_pred);
    graph->succ =
        ggml_get_rows(graph->ctx, selector.succ_cb, graph->inp_succ);
    graph->pred =
        ggml_get_rows(graph->ctx, selector.pred_cb, graph->inp_pred);
    ggml_set_output(graph->succ);
    ggml_set_output(graph->pred);
    ggml_build_forward_expand(graph->gf, graph->succ);
    ggml_build_forward_expand(graph->gf, graph->pred);
    if (!alloc_cached_graph(cache.selectors, *graph, backend)) {
        std::fprintf(stderr,
            "dflash2_select_chains_batched: selector graph alloc failed\n");
        return nullptr;
    }
    graph->dw = &dw;
    graph->backend = backend;
    graph->n_lanes = n_lanes;
    graph->n_cand = n_cand;
    graph->K = K;
    cache.selectors.push_back(std::move(graph));
    return cache.selectors.back().get();
}

bool validate_common_inputs(
        const DraftWeights & dw,
        ggml_backend_t backend,
        ggml_tensor * lm_head,
        int n_lanes,
        int q_len,
        const std::vector<int32_t> & last_tokens) {
    const DraftSelectorWeights & selector = dw.selector;
    const int n_cand = q_len - 1;
    const int K = selector.top_k;
    const int rank = selector.rank;
    const int hdim = dw.n_embd;
    if (!selector.enabled || !selector.hproj || !selector.pred_cb ||
        !selector.succ_cb || !backend || !lm_head || n_lanes <= 0 ||
        static_cast<int>(last_tokens.size()) != n_lanes ||
        n_cand <= 0 || K <= 0 || rank <= 0 || hdim <= 0) {
        return false;
    }
    DFlash2SelectorLayout selector_layout;
    selector_layout.rank = rank;
    selector_layout.top_k = K;
    selector_layout.hproj_rank = selector.hproj->ne[1];
    selector_layout.pred_rank = selector.pred_cb->ne[0];
    selector_layout.pred_vocab = selector.pred_cb->ne[1];
    selector_layout.succ_rank = selector.succ_cb->ne[0];
    selector_layout.succ_vocab = selector.succ_cb->ne[1];
    selector_layout.target_output_vocab = lm_head->ne[1];
    std::string selector_error;
    if (!validate_dflash2_selector_layout(
            selector_layout, selector_error)) {
        std::fprintf(stderr, "dflash2_select_chains_batched: %s\n",
                     selector_error.c_str());
        return false;
    }
    for (size_t lane = 0; lane < last_tokens.size(); ++lane) {
        const int32_t token = last_tokens[lane];
        if (token < 0 || token >= selector_layout.pred_vocab) {
            std::fprintf(stderr,
                         "dflash2_select_chains_batched: lane %zu seed token "
                         "%d is outside codebook vocab %lld\n",
                         lane, token,
                         (long long)selector_layout.pred_vocab);
            return false;
        }
    }
    return true;
}

bool finish_batched_selection(
        SelectorCache & cache,
        ProjectionGraph & projection,
        const DraftWeights & dw,
        ggml_backend_t backend,
        ggml_tensor * lm_head,
        int n_lanes,
        int q_len,
        const std::vector<int32_t> & last_tokens,
        std::vector<std::vector<int32_t>> & draft_tokens) {
    const DraftSelectorWeights & selector = dw.selector;
    const int n_cand = q_len - 1;
    const int K = selector.top_k;
    const int rank = selector.rank;
    const int n_positions = n_lanes * n_cand;

    if (ggml_backend_graph_compute(backend, projection.gf) !=
        GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
            "dflash2_select_chains_batched: projection compute failed\n");
        return false;
    }

    const int vocab = static_cast<int>(lm_head->ne[1]);
    cache.candidate_log_probs.resize((size_t) n_positions * (size_t) K);
    cache.candidate_ids.resize((size_t) n_positions * (size_t) K);
    bool have_top_k = false;
#ifdef LUCE_HAVE_DRAFT_TOPK
    if (projection.logits && projection.logits->data) {
        have_top_k = geometric_extract_draft_topk_cuda(
            projection.logits->data, n_positions, vocab, K,
            cache.candidate_log_probs.data(), cache.candidate_ids.data(), 1.0f);
    }
#endif
    if (!have_top_k) {
        std::vector<float> logits(
            (size_t) vocab * (size_t) n_positions);
        ggml_backend_tensor_get(
            projection.logits, logits.data(), 0,
            sizeof(float) * logits.size());
        extract_draft_topk(
            logits.data(), n_positions, vocab, K,
            cache.candidate_log_probs.data(), cache.candidate_ids.data(), 1.0f);
    }

    BatchedSelectorGraph * graph = ensure_selector_graph(
        cache, dw, backend, n_lanes, n_cand, K);
    if (!graph) return false;

    cache.predecessor_ids.resize((size_t) n_lanes + cache.candidate_ids.size());
    std::copy(
        last_tokens.begin(), last_tokens.end(), cache.predecessor_ids.begin());
    std::copy(
        cache.candidate_ids.begin(), cache.candidate_ids.end(),
        cache.predecessor_ids.begin() + n_lanes);

    const size_t succ_in_bytes = sizeof(int32_t) * cache.candidate_ids.size();
    const size_t pred_in_bytes = sizeof(int32_t) * cache.predecessor_ids.size();
    const size_t hproj_bytes   = sizeof(float) * (size_t) rank * (size_t) n_positions;
    const size_t succ_out_bytes = sizeof(float) * (size_t) rank * cache.candidate_ids.size();
    const size_t pred_out_bytes = sizeof(float) * (size_t) rank * cache.predecessor_ids.size();
    const size_t stage_need =
        succ_in_bytes + pred_in_bytes + hproj_bytes + succ_out_bytes + pred_out_bytes + 512;
    cache.stage.reserve(backend, stage_need);
    cache.stage.reset();

    cache.stage.set(backend, graph->inp_succ, cache.candidate_ids.data(), 0, succ_in_bytes);
    cache.stage.set(backend, graph->inp_pred, cache.predecessor_ids.data(), 0, pred_in_bytes);
    if (ggml_backend_graph_compute(backend, graph->gf) !=
        GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
            "dflash2_select_chains_batched: selector compute failed\n");
        return false;
    }

    const float * projected_hidden = static_cast<const float *>(
        cache.stage.get(backend, projection.hproj, 0, hproj_bytes));
    const float * successor_codes = static_cast<const float *>(
        cache.stage.get(backend, graph->succ, 0, succ_out_bytes));
    const float * predecessor_codes = static_cast<const float *>(
        cache.stage.get(backend, graph->pred, 0, pred_out_bytes));
    if (!projected_hidden) {
        cache.projected_hidden.resize(hproj_bytes / sizeof(float));
        ggml_backend_tensor_get_async(
            backend, projection.hproj, cache.projected_hidden.data(), 0, hproj_bytes);
        projected_hidden = cache.projected_hidden.data();
    }
    if (!successor_codes) {
        cache.successor_codes.resize(succ_out_bytes / sizeof(float));
        ggml_backend_tensor_get_async(
            backend, graph->succ, cache.successor_codes.data(), 0, succ_out_bytes);
        successor_codes = cache.successor_codes.data();
    }
    if (!predecessor_codes) {
        cache.predecessor_codes.resize(pred_out_bytes / sizeof(float));
        ggml_backend_tensor_get_async(
            backend, graph->pred, cache.predecessor_codes.data(), 0, pred_out_bytes);
        predecessor_codes = cache.predecessor_codes.data();
    }
    ggml_backend_synchronize(backend);

    draft_tokens.assign(
        (size_t) n_lanes,
        std::vector<int32_t>((size_t) q_len));
    for (int lane = 0; lane < n_lanes; ++lane) {
        draft_tokens[(size_t) lane][0] = last_tokens[(size_t) lane];
        int predecessor_row = lane;
        for (int depth = 0; depth < n_cand; ++depth) {
            const int position = lane * n_cand + depth;
            const float * predecessor = predecessor_codes +
                (size_t) predecessor_row * (size_t) rank;
            const float * hidden = projected_hidden +
                (size_t) position * (size_t) rank;
            float best_score = -INFINITY;
            int best_candidate = 0;
            for (int candidate = 0; candidate < K; ++candidate) {
                const int candidate_row = position * K + candidate;
                const float * successor = successor_codes +
                    (size_t) candidate_row * (size_t) rank;
                float correction = 0.0f;
                for (int r = 0; r < rank; ++r) {
                    correction +=
                        predecessor[r] * hidden[r] * successor[r];
                }
                const float score =
                    cache.candidate_log_probs[(size_t) candidate_row] +
                    correction;
                if (score > best_score) {
                    best_score = score;
                    best_candidate = candidate;
                }
            }
            const int selected_row = position * K + best_candidate;
            draft_tokens[(size_t) lane][(size_t) depth + 1] =
                cache.candidate_ids[(size_t) selected_row];
            predecessor_row = n_lanes + selected_row;
        }
    }
    return true;
}

}  // namespace

void dflash2_select_chains_release() {
    SelectorCache & cache = selector_cache();
    cache.projections.clear();
    cache.selectors.clear();
}

bool dflash2_select_chains_batched_device(
        const DraftWeights & dw,
        ggml_backend_t backend,
        ggml_tensor * lm_head,
        const std::vector<ggml_tensor *> & hidden_by_lane,
        int q_len,
        const std::vector<int32_t> & last_tokens,
        std::vector<std::vector<int32_t>> & draft_tokens) {
    draft_tokens.clear();
    const int n_lanes = static_cast<int>(hidden_by_lane.size());
    if (!validate_common_inputs(dw, backend, lm_head, n_lanes, q_len, last_tokens)) {
        return false;
    }
    const int n_cand = q_len - 1;
    const int hdim = dw.n_embd;
    for (ggml_tensor * t : hidden_by_lane) {
        if (!t || t->type != GGML_TYPE_F32 ||
            t->ne[0] != hdim || t->ne[1] < q_len || !ggml_is_contiguous(t)) {
            return false;
        }
    }

    const int n_positions = n_lanes * n_cand;
    SelectorCache & cache = selector_cache();
    ProjectionGraph * projection = ensure_projection_graph(
        cache, dw, backend, lm_head, n_positions);
    if (!projection) return false;

    // Copy rows 1..q_len-1 of each real lane straight on-device on `backend`'s
    // stream (no host round-trip, no cross-stream sync).
    const size_t view_mem = ggml_tensor_overhead() * (size_t)(2 * n_lanes + 4);
    std::vector<uint8_t> view_buf(view_mem);
    ggml_init_params vp{};
    vp.mem_size = view_buf.size();
    vp.mem_buffer = view_buf.data();
    vp.no_alloc = true;
    ggml_context * vctx = ggml_init(vp);
    if (!vctx) return false;

    const size_t row_bytes = (size_t) hdim * sizeof(float);
    const size_t lane_bytes = (size_t) n_cand * row_bytes;
    for (int lane = 0; lane < n_lanes; ++lane) {
        ggml_tensor * src = hidden_by_lane[(size_t) lane];
        ggml_tensor * src_view = ggml_view_2d(
            vctx, src, hdim, n_cand, src->nb[1], src->nb[1]);
        ggml_tensor * dst_view = ggml_view_2d(
            vctx, projection->inp_hidden, hdim, n_cand,
            projection->inp_hidden->nb[1], (size_t) lane * lane_bytes);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);
    }
    ggml_free(vctx);

    return finish_batched_selection(
        cache, *projection, dw, backend, lm_head, n_lanes, q_len, last_tokens, draft_tokens);
}

}  // namespace luce::common
