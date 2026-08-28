#include "dflash2_head.h"

#include "dflash2_selector_validation.h"
#include "ddtree.h"
#include "geometric_draft_topk_cuda.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace dflash::common {
namespace {

struct ProjectionGraph {
    const DraftWeights * dw = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_tensor * lm_head = nullptr;
    int n_positions = 0;
    std::vector<uint8_t> arena;
    uint64_t generation = 0;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor * inp_hidden = nullptr;
    ggml_tensor * logits = nullptr;

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
    uint64_t generation = 0;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor * inp_hidden = nullptr;
    ggml_tensor * inp_succ = nullptr;
    ggml_tensor * inp_pred = nullptr;
    ggml_tensor * hproj = nullptr;
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

ProjectionGraph & projection_graph() {
    static thread_local ProjectionGraph graph;
    return graph;
}

BatchedSelectorGraph & batched_selector_graph() {
    static thread_local BatchedSelectorGraph graph;
    return graph;
}

void free_projection_graph(ProjectionGraph & graph) {
    if (graph.galloc) {
        ggml_gallocr_free(graph.galloc);
        graph.galloc = nullptr;
    }
    if (graph.ctx) {
        ggml_free(graph.ctx);
        graph.ctx = nullptr;
    }
    graph = {};
}

void free_selector_graph(BatchedSelectorGraph & graph) {
    if (graph.galloc) {
        ggml_gallocr_free(graph.galloc);
        graph.galloc = nullptr;
    }
    if (graph.ctx) {
        ggml_free(graph.ctx);
        graph.ctx = nullptr;
    }
    graph = {};
}

bool ensure_projection_graph(
        ProjectionGraph & graph, const DraftWeights & dw,
        ggml_backend_t backend, ggml_tensor * lm_head, int n_positions) {
    const uint64_t generation = dflash2_selector_graph_generation();
    if (graph.ctx && graph.dw == &dw && graph.backend == backend &&
        graph.lm_head == lm_head && graph.n_positions == n_positions &&
        graph.generation == generation) {
        return true;
    }
    free_projection_graph(graph);
    if (!backend || !lm_head || n_positions <= 0 || dw.n_embd <= 0 ||
        lm_head->ne[0] != dw.n_embd || lm_head->ne[1] <= 0) {
        return false;
    }

    const size_t arena_size =
        ggml_tensor_overhead() * 32 +
        ggml_graph_overhead_custom(256, false) + 4096;
    graph.arena.assign(arena_size, 0);
    ggml_init_params params{};
    params.mem_size = graph.arena.size();
    params.mem_buffer = graph.arena.data();
    params.no_alloc = true;
    graph.ctx = ggml_init(params);
    if (!graph.ctx) return false;
    graph.gf = ggml_new_graph_custom(graph.ctx, 256, false);
    graph.inp_hidden = ggml_new_tensor_2d(
        graph.ctx, GGML_TYPE_F32, dw.n_embd, n_positions);
    ggml_set_input(graph.inp_hidden);
    graph.logits = ggml_mul_mat(graph.ctx, lm_head, graph.inp_hidden);
    ggml_set_output(graph.logits);
    ggml_build_forward_expand(graph.gf, graph.logits);
    graph.galloc =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!graph.galloc || !ggml_gallocr_alloc_graph(graph.galloc, graph.gf)) {
        std::fprintf(stderr,
            "dflash2_select_chains_batched: projection graph alloc failed\n");
        free_projection_graph(graph);
        return false;
    }
    graph.dw = &dw;
    graph.backend = backend;
    graph.lm_head = lm_head;
    graph.n_positions = n_positions;
    graph.generation = generation;
    return true;
}

bool ensure_selector_graph(
        BatchedSelectorGraph & graph, const DraftWeights & dw,
        ggml_backend_t backend, int n_lanes, int n_cand, int K) {
    const uint64_t generation = dflash2_selector_graph_generation();
    if (graph.ctx && graph.dw == &dw && graph.backend == backend &&
        graph.n_lanes == n_lanes && graph.n_cand == n_cand &&
        graph.K == K && graph.generation == generation) {
        return true;
    }
    free_selector_graph(graph);
    const DraftSelectorWeights & selector = dw.selector;
    if (!backend || n_lanes <= 0 || n_cand <= 0 || K <= 0 ||
        dw.n_embd <= 0 || selector.rank <= 0 || !selector.hproj ||
        !selector.pred_cb || !selector.succ_cb) {
        return false;
    }

    const int n_positions = n_lanes * n_cand;
    const int n_pred_rows = n_lanes + n_positions * K;
    const size_t arena_size =
        ggml_tensor_overhead() * 48 +
        ggml_graph_overhead_custom(256, false) + 4096;
    graph.arena.assign(arena_size, 0);
    ggml_init_params params{};
    params.mem_size = graph.arena.size();
    params.mem_buffer = graph.arena.data();
    params.no_alloc = true;
    graph.ctx = ggml_init(params);
    if (!graph.ctx) return false;
    graph.gf = ggml_new_graph_custom(graph.ctx, 256, false);
    graph.inp_hidden = ggml_new_tensor_2d(
        graph.ctx, GGML_TYPE_F32, dw.n_embd, n_positions);
    graph.inp_succ = ggml_new_tensor_1d(
        graph.ctx, GGML_TYPE_I32, n_positions * K);
    graph.inp_pred = ggml_new_tensor_1d(
        graph.ctx, GGML_TYPE_I32, n_pred_rows);
    ggml_set_input(graph.inp_hidden);
    ggml_set_input(graph.inp_succ);
    ggml_set_input(graph.inp_pred);
    graph.hproj =
        ggml_mul_mat(graph.ctx, selector.hproj, graph.inp_hidden);
    graph.succ =
        ggml_get_rows(graph.ctx, selector.succ_cb, graph.inp_succ);
    graph.pred =
        ggml_get_rows(graph.ctx, selector.pred_cb, graph.inp_pred);
    ggml_set_output(graph.hproj);
    ggml_set_output(graph.succ);
    ggml_set_output(graph.pred);
    ggml_build_forward_expand(graph.gf, graph.hproj);
    ggml_build_forward_expand(graph.gf, graph.succ);
    ggml_build_forward_expand(graph.gf, graph.pred);
    graph.galloc =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!graph.galloc || !ggml_gallocr_alloc_graph(graph.galloc, graph.gf)) {
        std::fprintf(stderr,
            "dflash2_select_chains_batched: selector graph alloc failed\n");
        free_selector_graph(graph);
        return false;
    }
    graph.dw = &dw;
    graph.backend = backend;
    graph.n_lanes = n_lanes;
    graph.n_cand = n_cand;
    graph.K = K;
    graph.generation = generation;
    return true;
}

}  // namespace

bool dflash2_select_chains_batched(
        const DraftWeights & dw,
        ggml_backend_t backend,
        ggml_tensor * lm_head,
        const std::vector<const float *> & hidden_by_lane,
        int q_len,
        const std::vector<int32_t> & last_tokens,
        std::vector<std::vector<int32_t>> & draft_tokens) {
    draft_tokens.clear();
    const DraftSelectorWeights & selector = dw.selector;
    const int n_lanes = static_cast<int>(hidden_by_lane.size());
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
    for (const float * hidden : hidden_by_lane) {
        if (!hidden) return false;
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

    const int n_positions = n_lanes * n_cand;
    std::vector<float> candidate_hidden(
        (size_t) hdim * (size_t) n_positions);
    for (int lane = 0; lane < n_lanes; ++lane) {
        for (int depth = 0; depth < n_cand; ++depth) {
            const int position = lane * n_cand + depth;
            const float * source = hidden_by_lane[(size_t) lane] +
                (size_t) (depth + 1) * (size_t) hdim;
            std::memcpy(
                candidate_hidden.data() +
                    (size_t) position * (size_t) hdim,
                source, sizeof(float) * (size_t) hdim);
        }
    }

    ProjectionGraph & projection = projection_graph();
    if (!ensure_projection_graph(
            projection, dw, backend, lm_head, n_positions)) {
        return false;
    }
    ggml_backend_tensor_set(
        projection.inp_hidden, candidate_hidden.data(), 0,
        sizeof(float) * candidate_hidden.size());
    if (ggml_backend_graph_compute(backend, projection.gf) !=
        GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
            "dflash2_select_chains_batched: projection compute failed\n");
        return false;
    }

    const int vocab = static_cast<int>(lm_head->ne[1]);
    std::vector<float> candidate_log_probs(
        (size_t) n_positions * (size_t) K);
    std::vector<int32_t> candidate_ids(
        (size_t) n_positions * (size_t) K);
    bool have_top_k = false;
#ifdef DFLASH27B_HAVE_DRAFT_TOPK
    if (projection.logits && projection.logits->data) {
        have_top_k = geometric_extract_draft_topk_cuda(
            projection.logits->data, n_positions, vocab, K,
            candidate_log_probs.data(), candidate_ids.data(), 1.0f);
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
            candidate_log_probs.data(), candidate_ids.data(), 1.0f);
    }

    BatchedSelectorGraph & graph = batched_selector_graph();
    if (!ensure_selector_graph(
            graph, dw, backend, n_lanes, n_cand, K)) {
        return false;
    }
    std::vector<int32_t> predecessor_ids(
        (size_t) n_lanes + candidate_ids.size());
    std::copy(
        last_tokens.begin(), last_tokens.end(), predecessor_ids.begin());
    std::copy(
        candidate_ids.begin(), candidate_ids.end(),
        predecessor_ids.begin() + n_lanes);
    ggml_backend_tensor_set(
        graph.inp_hidden, candidate_hidden.data(), 0,
        sizeof(float) * candidate_hidden.size());
    ggml_backend_tensor_set(
        graph.inp_succ, candidate_ids.data(), 0,
        sizeof(int32_t) * candidate_ids.size());
    ggml_backend_tensor_set(
        graph.inp_pred, predecessor_ids.data(), 0,
        sizeof(int32_t) * predecessor_ids.size());
    if (ggml_backend_graph_compute(backend, graph.gf) !=
        GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
            "dflash2_select_chains_batched: selector compute failed\n");
        return false;
    }

    std::vector<float> projected_hidden(
        (size_t) rank * (size_t) n_positions);
    std::vector<float> successor_codes(
        (size_t) rank * candidate_ids.size());
    std::vector<float> predecessor_codes(
        (size_t) rank * predecessor_ids.size());
    ggml_backend_tensor_get_async(
        backend, graph.hproj, projected_hidden.data(), 0,
        sizeof(float) * projected_hidden.size());
    ggml_backend_tensor_get_async(
        backend, graph.succ, successor_codes.data(), 0,
        sizeof(float) * successor_codes.size());
    ggml_backend_tensor_get_async(
        backend, graph.pred, predecessor_codes.data(), 0,
        sizeof(float) * predecessor_codes.size());
    ggml_backend_synchronize(backend);

    draft_tokens.assign(
        (size_t) n_lanes,
        std::vector<int32_t>((size_t) q_len));
    for (int lane = 0; lane < n_lanes; ++lane) {
        draft_tokens[(size_t) lane][0] = last_tokens[(size_t) lane];
        int predecessor_row = lane;
        for (int depth = 0; depth < n_cand; ++depth) {
            const int position = lane * n_cand + depth;
            const float * predecessor = predecessor_codes.data() +
                (size_t) predecessor_row * (size_t) rank;
            const float * hidden = projected_hidden.data() +
                (size_t) position * (size_t) rank;
            float best_score = -INFINITY;
            int best_candidate = 0;
            for (int candidate = 0; candidate < K; ++candidate) {
                const int candidate_row = position * K + candidate;
                const float * successor = successor_codes.data() +
                    (size_t) candidate_row * (size_t) rank;
                float correction = 0.0f;
                for (int r = 0; r < rank; ++r) {
                    correction +=
                        predecessor[r] * hidden[r] * successor[r];
                }
                const float score =
                    candidate_log_probs[(size_t) candidate_row] +
                    correction;
                if (score > best_score) {
                    best_score = score;
                    best_candidate = candidate;
                }
            }
            const int selected_row = position * K + best_candidate;
            draft_tokens[(size_t) lane][(size_t) depth + 1] =
                candidate_ids[(size_t) selected_row];
            predecessor_row = n_lanes + selected_row;
        }
    }
    return true;
}

}  // namespace dflash::common
