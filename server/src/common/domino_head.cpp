#include "domino_head.h"

#include "ggml-alloc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace dflash::common {

namespace {

bool domino_step(const DraftWeights & dw,
                 ggml_backend_t backend,
                 const float * prev_embed,
                 const float * prev_state,
                 const float * draft_hidden,
                 const float * base_logits,
                 int vocab,
                 int32_t & out_token,
                 std::vector<float> & out_state) {
    const int hidden = dw.n_embd;
    const int H = dw.domino.gru_hidden_dim;
    const int E = dw.domino.emb_dim;
    if (hidden <= 0 || H <= 0 || E <= 0 || vocab <= 0) return false;

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

    ggml_tensor * inp_embed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
    ggml_tensor * inp_state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 1);
    ggml_tensor * inp_hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
    ggml_tensor * inp_base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab, 1);
    ggml_set_input(inp_embed);
    ggml_set_input(inp_state);
    ggml_set_input(inp_hidden);
    ggml_set_input(inp_base);

    ggml_tensor * gi = ggml_mul_mat(ctx, dw.domino.gru_w_ih, inp_embed);
    gi = ggml_add(ctx, gi, ggml_reshape_2d(ctx, dw.domino.gru_b_ih, 3 * H, 1));
    ggml_tensor * gh = ggml_mul_mat(ctx, dw.domino.gru_w_hh, inp_state);
    gh = ggml_add(ctx, gh, ggml_reshape_2d(ctx, dw.domino.gru_b_hh, 3 * H, 1));

    const size_t gate_bytes = (size_t)H * ggml_element_size(gi);
    ggml_tensor * i_r = ggml_view_2d(ctx, gi, H, 1, gi->nb[1], 0);
    ggml_tensor * i_z = ggml_view_2d(ctx, gi, H, 1, gi->nb[1], gate_bytes);
    ggml_tensor * i_n = ggml_view_2d(ctx, gi, H, 1, gi->nb[1], 2 * gate_bytes);
    ggml_tensor * h_r = ggml_view_2d(ctx, gh, H, 1, gh->nb[1], 0);
    ggml_tensor * h_z = ggml_view_2d(ctx, gh, H, 1, gh->nb[1], gate_bytes);
    ggml_tensor * h_n = ggml_view_2d(ctx, gh, H, 1, gh->nb[1], 2 * gate_bytes);

    ggml_tensor * reset = ggml_sigmoid(ctx, ggml_add(ctx, i_r, h_r));
    ggml_tensor * update = ggml_sigmoid(ctx, ggml_add(ctx, i_z, h_z));
    ggml_tensor * cand = ggml_tanh(ctx, ggml_add(ctx, i_n, ggml_mul(ctx, reset, h_n)));
    ggml_tensor * h_new = ggml_add(ctx, cand,
                                   ggml_mul(ctx, update,
                                            ggml_sub(ctx, inp_state, cand)));

    ggml_tensor * zcat = ggml_concat(ctx, inp_hidden, h_new, 0);
    ggml_tensor * bias = ggml_mul_mat(ctx, dw.domino.head_w1, zcat);
    bias = ggml_add(ctx, bias, ggml_reshape_2d(ctx, dw.domino.head_b1, E, 1));
    bias = ggml_silu(ctx, bias);
    bias = ggml_mul_mat(ctx, dw.domino.head_w2, bias);
    bias = ggml_add(ctx, bias, ggml_reshape_2d(ctx, dw.domino.head_b2, vocab, 1));

    ggml_tensor * corrected = ggml_add(ctx, inp_base, bias);
    ggml_tensor * tok = ggml_argmax(ctx, corrected);
    ggml_set_output(tok);
    ggml_set_output(h_new);
    ggml_build_forward_expand(gf, tok);
    ggml_build_forward_expand(gf, h_new);

    static thread_local ggml_gallocr_t galloc = nullptr;
    if (!galloc) {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "domino_step: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(inp_embed, prev_embed, 0, sizeof(float) * (size_t)hidden);
    ggml_backend_tensor_set(inp_state, prev_state, 0, sizeof(float) * (size_t)H);
    ggml_backend_tensor_set(inp_hidden, draft_hidden, 0, sizeof(float) * (size_t)hidden);
    ggml_backend_tensor_set(inp_base, base_logits, 0, sizeof(float) * (size_t)vocab);

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "domino_step: graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(tok, &out_token, 0, sizeof(out_token));
    out_state.resize((size_t)H);
    ggml_backend_tensor_get(h_new, out_state.data(), 0, sizeof(float) * (size_t)H);
    ggml_free(ctx);
    return true;
}

}  // namespace

bool domino_correct_greedy_chain(const DraftWeights & dw,
                                 ggml_backend_t backend,
                                 DFlashTarget & target,
                                 const float * local_hidden,
                                 int q_len,
                                 int32_t last_tok,
                                 std::vector<int32_t> & draft_tok) {
    if (!dw.domino.enabled || q_len <= 1 || !local_hidden) return false;
    const int hidden = dw.n_embd;
    const int H = dw.domino.gru_hidden_dim;
    const int n_candidates = q_len - 1;
    if (hidden <= 0 || H <= 0 || n_candidates <= 0) return false;

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
    if (dw.domino.vocab_size > 0 && vocab != dw.domino.vocab_size) {
        std::fprintf(stderr, "domino_correct_greedy_chain: vocab mismatch target=%d domino=%d\n",
                     vocab, dw.domino.vocab_size);
        return false;
    }

    std::vector<float> state((size_t)H, 0.0f);
    if (std::getenv("DFLASH_DOMINO_ZERO_START") == nullptr) {
        ggml_backend_tensor_get(dw.domino.start, state.data(), 0,
                                sizeof(float) * (size_t)H);
    }

    std::vector<float> prev_embed((size_t)hidden);
    int32_t prefix_tok = last_tok;
    if (!target.embed_tokens(&prefix_tok, 1, prev_embed.data())) {
        return false;
    }

    draft_tok.assign((size_t)q_len, 0);
    draft_tok[0] = last_tok;
    std::vector<float> next_state;
    for (int i = 0; i < n_candidates; ++i) {
        int32_t tok = -1;
        if (!domino_step(dw, backend,
                         prev_embed.data(),
                         state.data(),
                         candidate_hidden.data() + (size_t)i * (size_t)hidden,
                         base_logits.data() + (size_t)i * (size_t)vocab,
                         vocab,
                         tok,
                         next_state)) {
            return false;
        }
        draft_tok[(size_t)i + 1] = tok;
        state.swap(next_state);
        if (!target.embed_tokens(&tok, 1, prev_embed.data())) {
            return false;
        }
    }
    return true;
}

}  // namespace dflash::common
