#include "dspark_head.h"

#include "ggml-alloc.h"

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

}  // namespace dflash::common
