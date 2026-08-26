#include "dspark_head.h"

#include "ggml-alloc.h"
#include "ddtree.h"

#include <cstdio>
#include <cstdlib>
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
    if (!dw.dspark.enabled || q_len <= 0 || !local_hidden) return false;
    const int hidden = dw.n_embd;
    const int first_candidate = dw.sample_from_anchor ? 0 : 1;
    const int n_candidates = q_len - first_candidate;
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
        const float * src = local_hidden +
            (size_t)(i + first_candidate) * (size_t)hidden;
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
    draft_tok.reserve((size_t)n_candidates + 1);
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

bool dspark_build_greedy_markov_chain(
        ggml_context * ctx,
        ggml_cgraph * gf,
        const DraftWeights & dw,
        ggml_tensor * base_logits,
        ggml_tensor * seed,
        std::vector<ggml_tensor *> & tokens_out) {
    tokens_out.clear();
    if (!ctx || !gf || !base_logits || !seed ||
        !dw.dspark.enabled || !dw.dspark.markov_w1 ||
        !dw.dspark.markov_w2 || seed->type != GGML_TYPE_I32 ||
        seed->ne[0] != 1 || base_logits->ne[1] <= 0) {
        return false;
    }
    const int64_t vocab = base_logits->ne[0];
    if (dw.dspark.vocab_size > 0 && vocab != dw.dspark.vocab_size) {
        return false;
    }

    ggml_tensor * prev = seed;
    tokens_out.reserve((size_t)base_logits->ne[1]);
    for (int64_t row = 0; row < base_logits->ne[1]; ++row) {
        ggml_tensor * prev_emb =
            ggml_get_rows(ctx, dw.dspark.markov_w1, prev);
        ggml_tensor * bias =
            ggml_mul_mat(ctx, dw.dspark.markov_w2, prev_emb);
        ggml_tensor * base_row = ggml_view_2d(
            ctx, base_logits, vocab, 1, base_logits->nb[1],
            (size_t)row * base_logits->nb[1]);
        ggml_tensor * corrected = ggml_add(ctx, base_row, bias);
        ggml_tensor * token = ggml_argmax(ctx, corrected);
        ggml_set_output(token);
        ggml_build_forward_expand(gf, token);
        tokens_out.push_back(token);
        prev = token;
    }
    return true;
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
};

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
                              ggml_tensor * resident_hidden,
                              int resident_hidden_row,
                              int n_positions, int first_corrected,
                              bool corrected_are_outputs,
                              bool confidence_are_outputs,
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

    if (resident_hidden) {
        if (resident_hidden->type != GGML_TYPE_F32 ||
            resident_hidden->ne[0] != hdim || resident_hidden_row < 0 ||
            resident_hidden->ne[1] < resident_hidden_row + n_positions ||
            resident_hidden->data == nullptr) {
            ggml_free(out.ctx);
            out.ctx = nullptr;
            out.gf = nullptr;
            return false;
        }
        out.inp_hidden = ggml_view_2d(
            out.ctx, resident_hidden, hdim, n_positions,
            resident_hidden->nb[1],
            (size_t)resident_hidden_row * resident_hidden->nb[1]);
    } else {
        out.inp_hidden =
            ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, hdim, n_positions);
        ggml_set_input(out.inp_hidden);
    }
    if (have_confidence) {
        out.inp_confidence_hidden =
            ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, hdim, n_positions);
        ggml_set_input(out.inp_confidence_hidden);
    }
    out.inp_seed   = ggml_new_tensor_1d(out.ctx, GGML_TYPE_I32, 1);
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
    if (!corrected_are_outputs && !have_confidence && first_corrected == 0) {
        return dspark_build_greedy_markov_chain(
            out.ctx, out.gf, dw, out.base, out.inp_seed, out.toks);
    }
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

struct DsparkMarkovChainCache::Impl {
    ggml_backend_t backend = nullptr;
    ggml_tensor * lm_head = nullptr;
    ggml_tensor * local_hidden = nullptr;
    void * local_hidden_data = nullptr;
    ggml_tensor * markov_w1 = nullptr;
    ggml_tensor * markov_w2 = nullptr;
    int q_len = 0;
    int first_candidate = 0;

    std::vector<uint8_t> arena;
    MarkovChainGraph graph;
    ggml_gallocr_t galloc = nullptr;
};

DsparkMarkovChainCache::DsparkMarkovChainCache() = default;

DsparkMarkovChainCache::~DsparkMarkovChainCache() {
    reset();
}

void DsparkMarkovChainCache::reset() {
    if (!impl_) return;
    if (impl_->galloc) ggml_gallocr_free(impl_->galloc);
    if (impl_->graph.ctx) ggml_free(impl_->graph.ctx);
    delete impl_;
    impl_ = nullptr;
}

bool dspark_markov_correct_greedy_chain_fused(const DraftWeights & dw,
                                              ggml_backend_t backend,
                                              ggml_tensor * lm_head,
                                              const float * local_hidden,
                                              int q_len,
                                              int32_t last_tok,
                                              std::vector<int32_t> & draft_tok,
                                              std::vector<float> * confidence_out,
                                              const float * confidence_hidden) {
    if (q_len <= 0) return false;
    if (!dspark_fused_usable(dw, backend, lm_head, local_hidden, "dspark_fused")) return false;
    const int hdim   = dw.n_embd;
    const int first_candidate = dw.sample_from_anchor ? 0 : 1;
    const int n_cand = q_len - first_candidate;
    if (n_cand <= 0 || n_cand > 16) return false;

    static thread_local std::vector<uint8_t> g_arena_chain;
    MarkovChainGraph g;
    const bool want_confidence = confidence_out != nullptr;
    if (confidence_out) confidence_out->clear();
    if (!build_markov_chain_graph(dw, lm_head,
                                  /*resident_hidden=*/nullptr,
                                  /*resident_hidden_row=*/0,
                                  n_cand, /*first_corrected=*/0,
                                  /*corrected_are_outputs=*/false,
                                  /*confidence_are_outputs=*/want_confidence,
                                  g_arena_chain, g)) {
        return false;
    }

    static thread_local ggml_gallocr_t galloc_chain = nullptr;
    if (!galloc_chain) {
        galloc_chain = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!ggml_gallocr_alloc_graph(galloc_chain, g.gf)) {
        std::fprintf(stderr, "dspark_fused: gallocr_alloc_graph failed\n");
        ggml_free(g.ctx);
        return false;
    }

    // Legacy DFlash starts candidates at slot 1 because slot 0 is the seed;
    // anchor-first DSpark predicts from every slot, including slot 0.
    ggml_backend_tensor_set(g.inp_hidden,
                            local_hidden + (size_t)first_candidate * (size_t)hdim, 0,
                            sizeof(float) * (size_t)hdim * (size_t)n_cand);
    if (want_confidence && g.inp_confidence_hidden) {
        const float * conf_src = confidence_hidden ? confidence_hidden : local_hidden;
        ggml_backend_tensor_set(
            g.inp_confidence_hidden,
            conf_src + (size_t)first_candidate * (size_t)hdim, 0,
                                sizeof(float) * (size_t)hdim * (size_t)n_cand);
    }
    ggml_backend_tensor_set(g.inp_seed, &last_tok, 0, sizeof(int32_t));

    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "dspark_fused: graph_compute failed\n");
        ggml_free(g.ctx);
        return false;
    }

    draft_tok.assign((size_t)n_cand + 1, 0);
    draft_tok[0] = last_tok;
    // One synchronize instead of n_cand blocking readbacks.
    int32_t t_out[16];
    float c_out[16] = {};
    const int n_get = n_cand < 16 ? n_cand : 16;
    for (int i = 0; i < n_get; ++i) {
        ggml_backend_tensor_get_async(backend, g.toks[(size_t)i], &t_out[i], 0, sizeof(int32_t));
        if (want_confidence && g.confidence[(size_t)i]) {
            ggml_backend_tensor_get_async(
                backend, g.confidence[(size_t)i], &c_out[i], 0, sizeof(float));
        }
    }
    ggml_backend_synchronize(backend);
    for (int i = 0; i < n_get; ++i) {
        draft_tok[(size_t)i + 1] = t_out[i];
    }
    if (want_confidence && !g.confidence.empty() && g.confidence[0]) {
        confidence_out->assign(c_out, c_out + n_get);
    }
    ggml_free(g.ctx);
    return true;
}

bool dspark_markov_correct_greedy_chain_fused_tensor(
        const DraftWeights & dw,
        ggml_backend_t backend,
        ggml_tensor * lm_head,
        ggml_tensor * local_hidden,
        int q_len,
        int32_t last_tok,
        std::vector<int32_t> & draft_tok,
        DsparkMarkovChainCache & cache) {
    if (q_len <= 0 || !local_hidden) return false;
    if (!dspark_fused_usable(
            dw, backend, lm_head,
            static_cast<const float *>(local_hidden->data),
            "dspark_fused_tensor")) {
        return false;
    }

    const int first_candidate = dw.sample_from_anchor ? 0 : 1;
    const int n_cand = q_len - first_candidate;
    if (n_cand <= 0 || n_cand > 16) return false;

    const bool cache_matches = cache.impl_ &&
        cache.impl_->backend == backend &&
        cache.impl_->lm_head == lm_head &&
        cache.impl_->local_hidden == local_hidden &&
        cache.impl_->local_hidden_data == local_hidden->data &&
        cache.impl_->markov_w1 == dw.dspark.markov_w1 &&
        cache.impl_->markov_w2 == dw.dspark.markov_w2 &&
        cache.impl_->q_len == q_len &&
        cache.impl_->first_candidate == first_candidate;
    if (!cache_matches) {
        cache.reset();
        cache.impl_ = new DsparkMarkovChainCache::Impl();
        auto & impl = *cache.impl_;
        impl.backend = backend;
        impl.lm_head = lm_head;
        impl.local_hidden = local_hidden;
        impl.local_hidden_data = local_hidden->data;
        impl.markov_w1 = dw.dspark.markov_w1;
        impl.markov_w2 = dw.dspark.markov_w2;
        impl.q_len = q_len;
        impl.first_candidate = first_candidate;

        if (!build_markov_chain_graph(
                dw, lm_head, local_hidden, first_candidate,
                n_cand, /*first_corrected=*/0,
                /*corrected_are_outputs=*/false,
                /*confidence_are_outputs=*/false,
                impl.arena, impl.graph)) {
            cache.reset();
            return false;
        }
        impl.galloc =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!impl.galloc ||
            !ggml_gallocr_alloc_graph(impl.galloc, impl.graph.gf)) {
            std::fprintf(stderr,
                         "dspark_fused_tensor: gallocr_alloc_graph failed\n");
            cache.reset();
            return false;
        }
    }

    MarkovChainGraph & g = cache.impl_->graph;
    ggml_backend_tensor_set(g.inp_seed, &last_tok, 0, sizeof(int32_t));
    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "dspark_fused_tensor: graph_compute failed\n");
        cache.reset();
        return false;
    }

    draft_tok.assign((size_t)n_cand + 1, 0);
    draft_tok[0] = last_tok;
    int32_t token_ids[16] = {};
    for (int i = 0; i < n_cand; ++i) {
        ggml_backend_tensor_get_async(
            backend, g.toks[(size_t)i], &token_ids[i], 0, sizeof(int32_t));
    }
    ggml_backend_synchronize(backend);
    for (int i = 0; i < n_cand; ++i) {
        draft_tok[(size_t)i + 1] = token_ids[i];
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
    if (!build_markov_chain_graph(dw, lm_head,
                                  /*resident_hidden=*/nullptr,
                                  /*resident_hidden_row=*/0,
                                  n_tokens, /*first_corrected=*/1,
                                  /*corrected_are_outputs=*/true,
                                  /*confidence_are_outputs=*/false,
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
