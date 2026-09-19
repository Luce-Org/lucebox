#include "dspark_head.h"

#include "ggml-alloc.h"
#include "ddtree.h"

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <cstring>
#include <vector>

namespace dflash::common {

namespace {

bool dspark_step(const DraftWeights & dw,
                 ggml_backend_t backend,
                 int32_t prev_token,
                 const float * draft_hidden,
                 const float * base_logits,
                 int vocab,
                 int32_t & out_token,
                 float * confidence_out) {
    const int hidden = dw.n_embd;
    const int rank = dw.dspark.markov_rank;
    if (hidden <= 0 || rank <= 0 || vocab <= 0) return false;
    if (!dw.dspark.markov_w1 || !dw.dspark.markov_w2) return false;

    const bool want_conf =
        confidence_out != nullptr &&
        dw.dspark.confidence_w != nullptr &&
        dw.dspark.confidence_b != nullptr &&
        dw.dspark.confidence_dim > 0;

    const size_t arena_size =
        ggml_tensor_overhead() * 256 + ggml_graph_overhead() + 2 * 1024 * 1024;
    static thread_local std::vector<uint8_t> g_arena;
    if (g_arena.size() < arena_size) g_arena.resize(arena_size);

    ggml_init_params ip{};
    ip.mem_size = arena_size;
    ip.mem_buffer = g_arena.data();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);

    ggml_tensor * inp_prev = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * inp_base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab, 1);
    ggml_set_input(inp_prev);
    ggml_set_input(inp_base);

    ggml_tensor * prev_emb = ggml_get_rows(ctx, dw.dspark.markov_w1, inp_prev);
    ggml_tensor * bias = ggml_mul_mat(ctx, dw.dspark.markov_w2, prev_emb);
    ggml_tensor * corrected = ggml_add(ctx, inp_base, bias);
    ggml_tensor * tok = ggml_argmax(ctx, corrected);
    ggml_set_output(tok);
    ggml_build_forward_expand(gf, tok);

    ggml_tensor * conf = nullptr;
    ggml_tensor * inp_hidden = nullptr;
    if (want_conf) {
        inp_hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
        ggml_set_input(inp_hidden);
        ggml_tensor * conf_in = inp_hidden;
        if (dw.dspark.confidence_dim == hidden + rank) {
            conf_in = ggml_concat(ctx, inp_hidden, prev_emb, 0);
        } else if (dw.dspark.confidence_dim != hidden) {
            ggml_free(ctx);
            return false;
        }
        conf = ggml_mul_mat(ctx, dw.dspark.confidence_w, conf_in);
        conf = ggml_add(ctx, conf, ggml_reshape_2d(ctx, dw.dspark.confidence_b, 1, 1));
        conf = ggml_sigmoid(ctx, conf);
        ggml_set_output(conf);
        ggml_build_forward_expand(gf, conf);
    }

    static thread_local ggml_gallocr_t galloc = nullptr;
    if (!galloc) {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "dspark_step: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(inp_prev, &prev_token, 0, sizeof(prev_token));
    ggml_backend_tensor_set(inp_base, base_logits, 0, sizeof(float) * (size_t)vocab);
    if (want_conf) {
        ggml_backend_tensor_set(inp_hidden, draft_hidden, 0,
                                sizeof(float) * (size_t)hidden);
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "dspark_step: graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(tok, &out_token, 0, sizeof(out_token));
    if (want_conf) {
        ggml_backend_tensor_get(conf, confidence_out, 0, sizeof(float));
    }
    ggml_free(ctx);
    return true;
}

}  // namespace

bool dspark_markov_correct_greedy_chain(const DraftWeights & dw,
                                        ggml_backend_t backend,
                                        DFlashTarget & target,
                                        const float * local_hidden,
                                        int q_len,
                                        int32_t last_tok,
                                        float confidence_threshold,
                                        std::vector<int32_t> & draft_tok) {
    if (!dw.dspark.enabled || q_len <= 1 || !local_hidden) return false;
    const int hidden = dw.n_embd;
    const int n_candidates = q_len - 1;
    if (hidden <= 0 || n_candidates <= 0) return false;
    if (confidence_threshold < 0.0f) confidence_threshold = 0.0f;
    if (confidence_threshold > 1.0f) confidence_threshold = 1.0f;
    const bool use_confidence_gate =
        confidence_threshold > 0.0f &&
        dw.dspark.confidence_w != nullptr &&
        dw.dspark.confidence_b != nullptr &&
        dw.dspark.confidence_dim > 0;

    std::vector<float> candidate_hidden((size_t)n_candidates * (size_t)hidden);
    for (int i = 0; i < n_candidates; ++i) {
        const float * src = local_hidden + (size_t)(i + 1) * (size_t)hidden;
        std::memcpy(candidate_hidden.data() + (size_t)i * (size_t)hidden,
                    src, sizeof(float) * (size_t)hidden);
    }

    std::vector<float> base_logits;
    if (!target.project_hidden_to_logits(candidate_hidden.data(), n_candidates, base_logits)) {
        return false;
    }
    if (base_logits.size() % (size_t)n_candidates != 0) return false;
    const int vocab = (int)(base_logits.size() / (size_t)n_candidates);
    if (dw.dspark.vocab_size > 0 && vocab != dw.dspark.vocab_size) {
        std::fprintf(stderr, "dspark_markov_correct_greedy_chain: vocab mismatch target=%d dspark=%d\n",
                     vocab, dw.dspark.vocab_size);
        return false;
    }

    draft_tok.clear();
    draft_tok.reserve((size_t)q_len);
    draft_tok.push_back(last_tok);
    int32_t prefix_tok = last_tok;
    for (int i = 0; i < n_candidates; ++i) {
        int32_t tok = -1;
        float confidence = 0.0f;
        float * confidence_ptr = use_confidence_gate ? &confidence : nullptr;
        if (!dspark_step(dw, backend, prefix_tok,
                         candidate_hidden.data() + (size_t)i * (size_t)hidden,
                         base_logits.data() + (size_t)i * (size_t)vocab,
                         vocab,
                         tok,
                         confidence_ptr)) {
            return false;
        }
        if (use_confidence_gate && confidence < confidence_threshold) {
            break;
        }
        draft_tok.push_back(tok);
        prefix_tok = tok;
    }
    return true;
}

static std::atomic<uint64_t> g_dspark_drafter_generation{1};
static std::atomic<uint64_t> g_dspark_chain_graph_builds{0};

void dspark_note_drafter_lifecycle() {
    g_dspark_drafter_generation.fetch_add(1, std::memory_order_acq_rel);
}

uint64_t dspark_drafter_generation() {
    return g_dspark_drafter_generation.load(std::memory_order_acquire);
}

uint64_t dspark_chain_graph_build_count() {
    return g_dspark_chain_graph_builds.load(std::memory_order_acquire);
}

namespace {

struct MarkovChainGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;
    ggml_tensor *  inp_hidden = nullptr;
    ggml_tensor *  inp_confidence_hidden = nullptr;
    ggml_tensor *  inp_seed   = nullptr;
    ggml_tensor *  base       = nullptr;          // [vocab, n_positions]
    std::vector<ggml_tensor *> toks;              // corrected argmax per depth
    std::vector<ggml_tensor *> corrected;         // corrected logits per depth
    std::vector<ggml_tensor *> confidence;        // optional sigmoid score per depth
    std::vector<ggml_tensor *> logit_margin;      // optional |top1 - top2| per depth
};

// Compose PR 705's graph reuse with the optional confidence and logit-margin
// outputs used by adaptive verification. The graph closes over tensor
// identities, so all optional weights and output modes belong in the key.
struct MarkovChainGraphCache {
    MarkovChainGraph graph;
    uint64_t generation = 0;
    std::vector<uint8_t> arena;
    ggml_gallocr_t allocator = nullptr;
    bool built = false;
    const void * lm_head = nullptr;
    const void * markov_w1 = nullptr;
    const void * markov_w2 = nullptr;
    const void * confidence_w = nullptr;
    const void * confidence_b = nullptr;
    ggml_backend_t backend = nullptr;
    int n_candidates = -1;
    bool want_confidence = false;
    bool want_logit_margin = false;

    bool matches(const DraftWeights & dw, ggml_tensor * head,
                 ggml_backend_t candidate_backend, int candidate_count,
                 bool confidence, bool logit_margin,
                 uint64_t current_generation) const {
        return built && generation == current_generation &&
               lm_head == head && backend == candidate_backend &&
               n_candidates == candidate_count &&
               want_confidence == confidence &&
               want_logit_margin == logit_margin &&
               markov_w1 == dw.dspark.markov_w1 &&
               markov_w2 == dw.dspark.markov_w2 &&
               confidence_w == dw.dspark.confidence_w &&
               confidence_b == dw.dspark.confidence_b;
    }

    // Free the context before build_markov_chain_graph() may grow the arena
    // that backs it. Keep the allocator so its backend buffer can be reused.
    void invalidate() {
        if (graph.ctx) ggml_free(graph.ctx);
        graph = MarkovChainGraph{};
        built = false;
    }
};

bool dspark_chain_graph_cache_disabled() {
    static const bool disabled = [] {
        const char * value = std::getenv("DFLASH_DSPARK_NO_CHAIN_GRAPH_CACHE");
        return value && value[0] && std::strcmp(value, "0") != 0;
    }();
    return disabled;
}

// Guards shared by the fused Markov paths: head present, usable inputs, and
// the target lm_head vocab matching the head's training vocab.
bool dspark_fused_usable(const DraftWeights & dw, ggml_backend_t backend,
                         ggml_tensor * lm_head, const float * hidden,
                         const char * who) {
    if (!dw.dspark.enabled || !hidden || !backend || !lm_head) return false;
    if (!dw.dspark.markov_w1 || !dw.dspark.markov_w2) return false;
    if (dw.n_embd <= 0 || dw.dspark.markov_rank <= 0) return false;
    const int vocab = (int)lm_head->ne[1];
    if (vocab <= 0) return false;
    if (dw.dspark.vocab_size > 0 && vocab != dw.dspark.vocab_size) {
        static bool s_vocab_warned = false;
        if (!s_vocab_warned) {
            s_vocab_warned = true;
            std::fprintf(stderr, "%s: vocab mismatch lm_head=%d dspark=%d; falling back\n",
                         who, vocab, dw.dspark.vocab_size);
        }
        return false;
    }
    return true;
}

// One graph: base logits for all n_positions hidden columns (a single lm_head
// matmul), then for rows [first_corrected, n_positions) the low-rank Markov
// correction chained along the main path -
//   bias_i      = markov_w2 . markov_w1[prev]
//   corrected_i = base_i + bias_i
//   tok_i       = argmax(corrected_i)   (feeds the next step's get_rows)
// The chain seed is an I32 graph input; markov_w1 doubles as the previous-
// token embedding table. Rows below first_corrected keep the uncorrected base.
bool build_markov_chain_graph(const DraftWeights & dw,
                              ggml_tensor * lm_head,
                              int n_positions, int first_corrected,
                              bool corrected_are_outputs,
                              bool confidence_are_outputs,
                              bool logit_margin_are_outputs,
                              std::vector<uint8_t> & arena,
                              MarkovChainGraph & out) {
    const int hdim   = dw.n_embd;
    const int vocab  = (int)lm_head->ne[1];
    const int n_corr = n_positions - first_corrected;
    if (n_positions <= 0 || n_corr <= 0) return false;
    const bool have_confidence = confidence_are_outputs &&
        dw.dspark.confidence_w != nullptr &&
        dw.dspark.confidence_b != nullptr &&
        (dw.dspark.confidence_dim == hdim ||
         dw.dspark.confidence_dim == hdim + dw.dspark.markov_rank);

    const size_t arena_size = ggml_tensor_overhead() * (size_t)(64 + 16 * n_corr) +
                              ggml_graph_overhead_custom(512, false) + 2 * 1024 * 1024;
    if (arena.size() < arena_size) arena.resize(arena_size);

    ggml_init_params ip{};
    ip.mem_size   = arena.size();
    ip.mem_buffer = arena.data();
    ip.no_alloc   = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;
    out.gf = ggml_new_graph_custom(out.ctx, 512, false);

    out.inp_hidden = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, hdim, n_positions);
    if (have_confidence) {
        out.inp_confidence_hidden =
            ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, hdim, n_positions);
        ggml_set_input(out.inp_confidence_hidden);
    }
    out.inp_seed   = ggml_new_tensor_1d(out.ctx, GGML_TYPE_I32, 1);
    ggml_set_input(out.inp_hidden);
    ggml_set_input(out.inp_seed);

    out.base = ggml_mul_mat(out.ctx, lm_head, out.inp_hidden);
    if (first_corrected > 0) {
        // The uncorrected rows are read back by the caller.
        ggml_set_output(out.base);
        ggml_build_forward_expand(out.gf, out.base);
    }

    ggml_tensor * prev_ids = out.inp_seed;
    out.toks.assign((size_t)n_corr, nullptr);
    out.corrected.assign((size_t)n_corr, nullptr);
    out.confidence.assign((size_t)n_corr, nullptr);
    out.logit_margin.assign((size_t)n_corr, nullptr);
    for (int i = 0; i < n_corr; ++i) {
        const int row = first_corrected + i;
        ggml_tensor * prev_emb = ggml_get_rows(out.ctx, dw.dspark.markov_w1, prev_ids);
        ggml_tensor * bias = ggml_mul_mat(out.ctx, dw.dspark.markov_w2, prev_emb);
        ggml_tensor * base_i = ggml_view_2d(out.ctx, out.base, vocab, 1,
                                            out.base->nb[1], (size_t)row * out.base->nb[1]);
        ggml_tensor * corrected = ggml_add(out.ctx, base_i, bias);
        if (corrected_are_outputs) {
            ggml_set_output(corrected);
            ggml_build_forward_expand(out.gf, corrected);
        }
        ggml_tensor * tok = ggml_argmax(out.ctx, corrected);
        ggml_set_output(tok);
        ggml_build_forward_expand(out.gf, tok);
        out.corrected[(size_t)i] = corrected;
        out.toks[(size_t)i] = tok;
        if (logit_margin_are_outputs) {
            ggml_tensor * top_ids = ggml_top_k(out.ctx, corrected, 2);
            ggml_tensor * logits_rows = ggml_reshape_3d(
                out.ctx, corrected, 1, vocab, 1);
            ggml_tensor * top_values = ggml_get_rows(
                out.ctx, logits_rows, top_ids);
            ggml_tensor * first = ggml_view_1d(
                out.ctx, top_values, 1, 0);
            ggml_tensor * second = ggml_view_1d(
                out.ctx, top_values, 1, top_values->nb[1]);
            ggml_tensor * margin = ggml_abs(
                out.ctx, ggml_sub(out.ctx, first, second));
            ggml_set_output(margin);
            ggml_build_forward_expand(out.gf, margin);
            out.logit_margin[(size_t)i] = margin;
        }
        if (have_confidence) {
            ggml_tensor * hidden_i = ggml_view_2d(
                out.ctx, out.inp_confidence_hidden, hdim, 1,
                out.inp_confidence_hidden->nb[1],
                (size_t)row * out.inp_confidence_hidden->nb[1]);
            ggml_tensor * conf_in = hidden_i;
            if (dw.dspark.confidence_dim == hdim + dw.dspark.markov_rank) {
                conf_in = ggml_concat(out.ctx, hidden_i, prev_emb, 0);
            }
            ggml_tensor * conf = ggml_mul_mat(out.ctx, dw.dspark.confidence_w, conf_in);
            conf = ggml_add(
                out.ctx, conf,
                ggml_reshape_2d(out.ctx, dw.dspark.confidence_b, 1, 1));
            conf = ggml_sigmoid(out.ctx, conf);
            ggml_set_output(conf);
            ggml_build_forward_expand(out.gf, conf);
            out.confidence[(size_t)i] = conf;
        }
        prev_ids = tok;
    }
    return true;
}

}  // namespace

bool dspark_markov_correct_greedy_chain_fused(const DraftWeights & dw,
                                              ggml_backend_t backend,
                                              ggml_tensor * lm_head,
                                              const float * local_hidden,
                                              int q_len,
                                              int32_t last_tok,
                                              std::vector<int32_t> & draft_tok,
                                              std::vector<float> * confidence_out,
                                              const float * confidence_hidden,
                                              std::vector<float> * logit_margin_out) {
    if (q_len <= 1) return false;
    if (!dspark_fused_usable(dw, backend, lm_head, local_hidden, "dspark_fused")) return false;
    const int hdim   = dw.n_embd;
    const int n_cand = q_len - 1;

    static thread_local MarkovChainGraphCache cache;
    const bool want_confidence = confidence_out != nullptr;
    const bool want_logit_margin = logit_margin_out != nullptr;
    if (confidence_out) confidence_out->clear();
    if (logit_margin_out) logit_margin_out->clear();
    // Sample the generation once so a lifecycle bump racing this build is
    // observed as a miss on the next call rather than lost.
    const uint64_t generation = dspark_drafter_generation();
    const bool reuse = !dspark_chain_graph_cache_disabled() &&
        cache.matches(dw, lm_head, backend, n_cand, want_confidence,
                      want_logit_margin, generation);
    if (!reuse) {
        cache.invalidate();
        if (!build_markov_chain_graph(
                dw, lm_head, n_cand, /*first_corrected=*/0,
                /*corrected_are_outputs=*/false,
                /*confidence_are_outputs=*/want_confidence,
                /*logit_margin_are_outputs=*/want_logit_margin,
                cache.arena, cache.graph)) {
            cache.invalidate();
            return false;
        }
        if (cache.allocator && cache.backend != backend) {
            // The allocator is bound to a buffer type; a new backend needs
            // its own or the graph lands in the previous device's buffers.
            ggml_gallocr_free(cache.allocator);
            cache.allocator = nullptr;
        }
        if (!cache.allocator) {
            cache.allocator = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend));
        }
        if (!cache.allocator ||
            !ggml_gallocr_alloc_graph(cache.allocator, cache.graph.gf)) {
            std::fprintf(stderr, "dspark_fused: gallocr_alloc_graph failed\n");
            cache.invalidate();
            return false;
        }
        cache.lm_head = lm_head;
        cache.markov_w1 = dw.dspark.markov_w1;
        cache.markov_w2 = dw.dspark.markov_w2;
        cache.confidence_w = dw.dspark.confidence_w;
        cache.confidence_b = dw.dspark.confidence_b;
        cache.backend = backend;
        cache.n_candidates = n_cand;
        cache.want_confidence = want_confidence;
        cache.want_logit_margin = want_logit_margin;
        cache.generation = generation;
        cache.built = true;
        g_dspark_chain_graph_builds.fetch_add(1, std::memory_order_acq_rel);
    }
    MarkovChainGraph & g = cache.graph;

    // Candidate hidden states start at position 1 (position 0 is the seed).
    ggml_backend_tensor_set(g.inp_hidden, local_hidden + (size_t)hdim, 0,
                            sizeof(float) * (size_t)hdim * (size_t)n_cand);
    if (want_confidence && g.inp_confidence_hidden) {
        const float * conf_src = confidence_hidden ? confidence_hidden : local_hidden;
        ggml_backend_tensor_set(g.inp_confidence_hidden, conf_src + (size_t)hdim, 0,
                                sizeof(float) * (size_t)hdim * (size_t)n_cand);
    }
    ggml_backend_tensor_set(g.inp_seed, &last_tok, 0, sizeof(int32_t));

    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "dspark_fused: graph_compute failed\n");
        cache.invalidate();
        return false;
    }

    draft_tok.assign((size_t)q_len, 0);
    draft_tok[0] = last_tok;
    // One synchronize instead of n_cand blocking readbacks.
    std::vector<int32_t> t_out((size_t)n_cand);
    std::vector<float> c_out(want_confidence ? (size_t)n_cand : 0);
    std::vector<float> m_out(want_logit_margin ? (size_t)n_cand : 0);
    for (int i = 0; i < n_cand; ++i) {
        ggml_backend_tensor_get_async(backend, g.toks[(size_t)i], &t_out[i], 0, sizeof(int32_t));
        if (want_confidence && g.confidence[(size_t)i]) {
            ggml_backend_tensor_get_async(
                backend, g.confidence[(size_t)i], &c_out[i], 0, sizeof(float));
        }
        if (want_logit_margin && g.logit_margin[(size_t)i]) {
            ggml_backend_tensor_get_async(
                backend, g.logit_margin[(size_t)i], &m_out[i], 0,
                sizeof(float));
        }
    }
    ggml_backend_synchronize(backend);
    for (int i = 0; i < n_cand; ++i) {
        draft_tok[(size_t)i + 1] = t_out[i];
    }
    if (want_confidence && !g.confidence.empty() && g.confidence[0]) {
        *confidence_out = std::move(c_out);
    }
    if (want_logit_margin && !g.logit_margin.empty() &&
        g.logit_margin[0]) {
        *logit_margin_out = std::move(m_out);
    }
    return true;
}

bool dspark_markov_project_topk(const DraftWeights & dw,
                                ggml_backend_t backend,
                                ggml_tensor * lm_head,
                                const float * hidden,
                                int n_tokens, int K, float temperature,
                                int32_t last_tok,
                                std::vector<float> & top_log_probs,
                                std::vector<int32_t> & top_token_ids) {
    if (n_tokens <= 1 || K <= 0) return false;
    if (!dspark_fused_usable(dw, backend, lm_head, hidden, "dspark_topk")) return false;
    const int hdim  = dw.n_embd;
    const int vocab = (int)lm_head->ne[1];

    static thread_local std::vector<uint8_t> g_arena_topk;
    MarkovChainGraph g;
    if (!build_markov_chain_graph(dw, lm_head, n_tokens, /*first_corrected=*/1,
                                  /*corrected_are_outputs=*/true,
                                  /*confidence_are_outputs=*/false,
                                  /*logit_margin_are_outputs=*/false,
                                  g_arena_topk, g)) {
        return false;
    }

    static thread_local ggml_gallocr_t galloc_topk = nullptr;
    if (!galloc_topk) {
        galloc_topk = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!ggml_gallocr_alloc_graph(galloc_topk, g.gf)) {
        std::fprintf(stderr, "dspark_topk: gallocr_alloc_graph failed\n");
        ggml_free(g.ctx);
        return false;
    }

    ggml_backend_tensor_set(g.inp_hidden, hidden, 0,
                            sizeof(float) * (size_t)hdim * (size_t)n_tokens);
    ggml_backend_tensor_set(g.inp_seed, &last_tok, 0, sizeof(int32_t));

    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "dspark_topk: graph_compute failed\n");
        ggml_free(g.ctx);
        return false;
    }

    // Corrected logits per row (row 0 keeps the uncorrected base), then the
    // same host top-K extraction as project_hidden_to_topk for identical
    // budget-allocation semantics in build_ddtree.
    std::vector<float> logits_host((size_t)vocab * (size_t)n_tokens);
    ggml_backend_tensor_get_async(backend, g.base, logits_host.data(), 0,
                                  sizeof(float) * (size_t)vocab);
    const int n_corr = n_tokens - 1;
    for (int i = 0; i < n_corr; ++i) {
        ggml_backend_tensor_get_async(backend, g.corrected[(size_t)i],
                                      logits_host.data() + (size_t)(i + 1) * (size_t)vocab,
                                      0, sizeof(float) * (size_t)vocab);
    }
    ggml_backend_synchronize(backend);

    top_log_probs.assign((size_t)n_tokens * (size_t)K, 0.0f);
    top_token_ids.assign((size_t)n_tokens * (size_t)K, 0);
    extract_draft_topk(logits_host.data(), n_tokens, vocab, K,
                       top_log_probs.data(), top_token_ids.data(), temperature);

    ggml_free(g.ctx);
    return true;
}

}  // namespace dflash::common
