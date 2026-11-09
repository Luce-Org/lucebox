#include "dflash2_head.h"

#include "dflash2_selector_validation.h"
#include "ddtree.h"
#include "geometric_draft_topk_cuda.h"
#include "pinned_stage.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <vector>

namespace luce::common {
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
    // Device of `logits` for the GPU top-k (-1: not device memory), looked
    // up once per graph.
    bool topk_checked = false;
    int topk_device = -1;

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

// One pinned stage per thread, like the graphs. Every select call waits for
// its copies before returning, so the next call may reuse it.
PinnedStage & selector_stage() {
    static thread_local PinnedStage stage;
    return stage;
}

// True when `t` holds f32 rows of `row_floats` packed back to back in the
// device memory `backend` computes in, where a kernel copy can reach it.
bool packed_device_rows(ggml_backend_t backend, const ggml_tensor * t,
                        int64_t row_floats) {
    if (!t || !t->data || t->type != GGML_TYPE_F32 ||
        t->ne[0] != row_floats || t->nb[0] != sizeof(float) ||
        t->nb[1] != sizeof(float) * (size_t) row_floats) {
        return false;
    }
    const ggml_backend_buffer_t buffer =
        t->view_src ? t->view_src->buffer : t->buffer;
    return buffer && !ggml_backend_buffer_is_host(buffer) &&
        ggml_backend_buffer_get_type(buffer) ==
            ggml_backend_get_default_buffer_type(backend);
}

// Queues, on the backend stream, the copy of drafted rows 1..n_cand of every
// lane block into each destination [hdim, n_lanes * n_cand] (lane-major, the
// layout the old host gather built). One kernel launch for up to 48 copies.
// False, with nothing queued, when a tensor is not packed device rows.
bool copy_candidate_rows_async(
        ggml_backend_t backend,
        const std::vector<const ggml_tensor *> & blocks, int q_len,
        int hdim, std::initializer_list<ggml_tensor *> destinations) {
    const int n_lanes = static_cast<int>(blocks.size());
    const int n_cand = q_len - 1;
    if (ggml_backend_cuda_get_device_id(backend) < 0) return false;
    for (const ggml_tensor * block : blocks) {
        if (!packed_device_rows(backend, block, hdim) || block->ne[1] < q_len) {
            return false;
        }
    }
    for (const ggml_tensor * destination : destinations) {
        if (!packed_device_rows(backend, destination, hdim) ||
            destination->ne[1] != (int64_t) n_lanes * n_cand) {
            return false;
        }
    }
    const size_t row_bytes = sizeof(float) * (size_t) hdim;
    const size_t lane_bytes = row_bytes * (size_t) n_cand;
    // Each block is read once per destination; only destinations must not
    // overlap, and they are disjoint graph inputs.
    std::vector<ggml_cuda_copy_desc> copies;
    copies.reserve(blocks.size() * destinations.size());
    for (ggml_tensor * destination : destinations) {
        for (int lane = 0; lane < n_lanes; ++lane) {
            ggml_cuda_copy_desc copy{};
            copy.src = static_cast<const char *>(blocks[(size_t) lane]->data) +
                row_bytes;
            copy.dst = static_cast<char *>(destination->data) +
                (size_t) lane * lane_bytes;
            copy.nbytes = lane_bytes;
            copies.push_back(copy);
        }
    }
    ggml_backend_cuda_copy_batch_async(
        backend, copies.data(), static_cast<int>(copies.size()));
    return true;
}

// The host gather of the same rows, for blocks a kernel copy cannot reach.
// Reads each block with a blocking get, so the caller synchronized first.
std::vector<float> gather_candidate_rows(
        const std::vector<const ggml_tensor *> & blocks, int q_len,
        int hdim) {
    const int n_cand = q_len - 1;
    const size_t row_floats = (size_t) hdim;
    std::vector<float> rows(
        row_floats * (size_t) n_cand * blocks.size());
    for (size_t lane = 0; lane < blocks.size(); ++lane) {
        ggml_backend_tensor_get(
            blocks[lane], rows.data() + lane * (size_t) n_cand * row_floats,
            sizeof(float) * row_floats,
            sizeof(float) * row_floats * (size_t) n_cand);
    }
    return rows;
}

// Points at a readback slice once the backend synchronized, or reads the
// tensor with a blocking get into `fallback` when the stage had no room.
template <typename T>
const T * staged_or_read(const void * slice, const ggml_tensor * t,
                         size_t count, std::vector<T> & fallback) {
    if (slice) return static_cast<const T *>(slice);
    fallback.resize(count);
    ggml_backend_tensor_get(t, fallback.data(), 0, sizeof(T) * count);
    return fallback.data();
}

}  // namespace

bool dflash2_select_chains_batched(
        const DraftWeights & dw,
        ggml_backend_t backend,
        ggml_tensor * lm_head,
        const std::vector<const ggml_tensor *> & hidden_by_lane,
        int q_len,
        const std::vector<int32_t> & last_tokens,
        std::vector<std::vector<int32_t>> & draft_tokens) {
    draft_tokens.clear();
    // Every exit waits for the copies queued here: the stage and the lane
    // blocks stay untouched until then.
    BackendDrain drain(backend);
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
    for (const ggml_tensor * hidden : hidden_by_lane) {
        if (!hidden || hidden->type != GGML_TYPE_F32 ||
            hidden->ne[0] != hdim || hidden->ne[1] < q_len) {
            return false;
        }
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

    // Both graphs exist before anything is queued: a rebuild frees a
    // compute buffer, which must not happen under queued work.
    const int n_positions = n_lanes * n_cand;
    ProjectionGraph & projection = projection_graph();
    if (!ensure_projection_graph(
            projection, dw, backend, lm_head, n_positions)) {
        return false;
    }
    BatchedSelectorGraph & graph = batched_selector_graph();
    if (!ensure_selector_graph(
            graph, dw, backend, n_lanes, n_cand, K)) {
        return false;
    }

    const int vocab = static_cast<int>(lm_head->ne[1]);
    const size_t n_candidates = (size_t) n_positions * (size_t) K;
    const size_t n_pred_rows = (size_t) n_lanes + n_candidates;
    PinnedStage & stage = selector_stage();
    const size_t stage_bytes =
        sizeof(int32_t) * (size_t) n_lanes +          // seed rows of inp_pred
        sizeof(float) * n_candidates +                 // top-k log-probs
        sizeof(int32_t) * n_candidates +               // top-k ids
        sizeof(float) * (size_t) rank * ((size_t) n_positions +
                                         n_candidates + n_pred_rows) +
        64 * 8;
    stage.reserve(backend, stage_bytes);
    stage.reset();

    // Drafted rows 1..q_len-1 of every lane into both graphs' hidden input,
    // device to device. A block a kernel copy cannot reach is gathered on
    // the host and uploaded, as before.
    if (!copy_candidate_rows_async(
            backend, hidden_by_lane, q_len, hdim,
            {projection.inp_hidden, graph.inp_hidden})) {
        ggml_backend_synchronize(backend);
        const std::vector<float> rows =
            gather_candidate_rows(hidden_by_lane, q_len, hdim);
        for (ggml_tensor * input : {projection.inp_hidden, graph.inp_hidden}) {
            stage.set(backend, input, rows.data(), 0,
                      sizeof(float) * rows.size());
        }
    }
    if (ggml_backend_graph_compute_async(backend, projection.gf) !=
        GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
            "dflash2_select_chains_batched: projection compute failed\n");
        return false;
    }

    // Top-k on the backend stream: the ids land in the selector's successor
    // input and the log-probs in a pinned slice. When the GPU path is not
    // available the logits are read back and reduced on the host, as before.
    const float * candidate_log_probs = nullptr;
    const void * ids_slice = nullptr;
    std::vector<float> host_log_probs;
    std::vector<int32_t> host_ids;
    bool have_top_k = false;
#ifdef LUCE_HAVE_DRAFT_TOPK
    if (!projection.topk_checked) {
        projection.topk_device = projection.logits
            ? geometric_draft_topk_cuda_device(projection.logits->data)
            : -1;
        projection.topk_checked = true;
    }
    if (projection.topk_device >= 0 &&
        projection.topk_device == ggml_backend_cuda_get_device_id(backend) &&
        geometric_draft_topk_cuda_supports_k(K)) {
        void * lp_slice = stage.take(sizeof(float) * n_candidates);
        if (!lp_slice) host_log_probs.resize(n_candidates);
        float * lp_out = lp_slice ? static_cast<float *>(lp_slice)
                                  : host_log_probs.data();
        have_top_k = geometric_extract_draft_topk_cuda_async(
            projection.logits->data, projection.topk_device, n_positions,
            vocab, K, 1.0f, ggml_backend_cuda_enter_stream(backend),
            static_cast<int32_t *>(graph.inp_succ->data), lp_out);
        if (have_top_k) {
            candidate_log_probs = lp_out;
            // The ids also follow the seeds in the predecessor input.
            ggml_cuda_copy_desc copy{};
            copy.src = graph.inp_succ->data;
            copy.dst = static_cast<char *>(graph.inp_pred->data) +
                sizeof(int32_t) * (size_t) n_lanes;
            copy.nbytes = sizeof(int32_t) * n_candidates;
            ggml_backend_cuda_copy_batch_async(backend, &copy, 1);
            // Read the ids before the selector graph runs: its allocator
            // may reuse an input's memory once the input was consumed.
            ids_slice = stage.get(backend, graph.inp_succ, 0,
                                  sizeof(int32_t) * n_candidates);
            if (!ids_slice) {
                ggml_backend_synchronize(backend);
                host_ids.resize(n_candidates);
                ggml_backend_tensor_get(graph.inp_succ, host_ids.data(), 0,
                                        sizeof(int32_t) * n_candidates);
            }
        }
    }
#endif
    if (!have_top_k) {
        ggml_backend_synchronize(backend);
        std::vector<float> logits(
            (size_t) vocab * (size_t) n_positions);
        ggml_backend_tensor_get(
            projection.logits, logits.data(), 0,
            sizeof(float) * logits.size());
        host_log_probs.resize(n_candidates);
        host_ids.resize(n_candidates);
        extract_draft_topk(
            logits.data(), n_positions, vocab, K,
            host_log_probs.data(), host_ids.data(), 1.0f);
        candidate_log_probs = host_log_probs.data();
        stage.set(backend, graph.inp_succ, host_ids.data(), 0,
                  sizeof(int32_t) * n_candidates);
        stage.set(backend, graph.inp_pred, host_ids.data(),
                  sizeof(int32_t) * (size_t) n_lanes,
                  sizeof(int32_t) * n_candidates);
    }
    stage.set(backend, graph.inp_pred, last_tokens.data(), 0,
              sizeof(int32_t) * (size_t) n_lanes);
    if (ggml_backend_graph_compute_async(backend, graph.gf) !=
        GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
            "dflash2_select_chains_batched: selector compute failed\n");
        return false;
    }

    // One wait for every readback.
    const size_t hproj_count = (size_t) rank * (size_t) n_positions;
    const size_t succ_count = (size_t) rank * n_candidates;
    const size_t pred_count = (size_t) rank * n_pred_rows;
    const void * hproj_slice = stage.get(
        backend, graph.hproj, 0, sizeof(float) * hproj_count);
    const void * succ_slice = stage.get(
        backend, graph.succ, 0, sizeof(float) * succ_count);
    const void * pred_slice = stage.get(
        backend, graph.pred, 0, sizeof(float) * pred_count);
    drain.now();
    std::vector<float> hproj_fallback;
    std::vector<float> succ_fallback;
    std::vector<float> pred_fallback;
    const float * projected_hidden = staged_or_read(
        hproj_slice, graph.hproj, hproj_count, hproj_fallback);
    const float * successor_codes = staged_or_read(
        succ_slice, graph.succ, succ_count, succ_fallback);
    const float * predecessor_codes = staged_or_read(
        pred_slice, graph.pred, pred_count, pred_fallback);
    const int32_t * candidate_ids = host_ids.empty()
        ? static_cast<const int32_t *>(ids_slice)
        : host_ids.data();

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

}  // namespace luce::common
