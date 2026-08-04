#include "kimi_k3_internal.h"

#include "common/moe_hybrid_routing_stats.h"
#include "common/moe_hybrid_stream.h"
#include "common/moe_router_graph.h"
#include "internal.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace dflash::common {
namespace {

ggml_tensor * rms_norm(ggml_context * ctx,
                       ggml_tensor * x,
                       ggml_tensor * weight,
                       float eps) {
    x = ggml_rms_norm(ctx, x, eps);
    return weight ? ggml_mul(ctx, x, weight) : x;
}

ggml_tensor * situ(ggml_context * ctx,
                   ggml_tensor * gate,
                   ggml_tensor * up,
                   float beta,
                   float linear_beta) {
    ggml_tensor * a = ggml_scale(ctx,
        ggml_tanh(ctx, ggml_scale(ctx, gate, 1.0f / beta)), beta);
    a = ggml_mul(ctx, a, ggml_sigmoid(ctx, gate));
    if (linear_beta > 0.0f) {
        up = ggml_scale(ctx,
            ggml_tanh(ctx, ggml_scale(ctx, up, 1.0f / linear_beta)),
            linear_beta);
    }
    return ggml_mul(ctx, a, up);
}

struct AttnResBank {
    ggml_context * ctx = nullptr;
    float eps = 1.0e-5f;
    int64_t n_embd = 0;
    int64_t n_tokens = 1;
    std::vector<ggml_tensor *> checkpoints;

    void push(ggml_tensor * cur) {
        checkpoints.push_back(
            ggml_reshape_3d(ctx, cur, n_embd, n_tokens, 1));
    }

    ggml_tensor * mix(ggml_tensor * cur, ggml_tensor * score_weight) {
        if (checkpoints.empty()) return cur;
        const int64_t n = static_cast<int64_t>(checkpoints.size());
        ggml_tensor * mixed = nullptr;

        // AttnRes chooses a different checkpoint mixture for every token.
        // Express the small contraction as independent token slices.  This is
        // the exact one-token algebra used by the original Kimi path, avoids
        // relying on fragile rank-3 broadcast rules, and is cheap at the
        // bounded speculative widths (currently <= 16).
        for (int64_t token = 0; token < n_tokens; ++token) {
            ggml_tensor * src = nullptr; // [hidden, checkpoint]
            for (ggml_tensor * checkpoint : checkpoints) {
                ggml_tensor * checkpoint_token = ggml_view_2d(
                    ctx, checkpoint, n_embd, 1, checkpoint->nb[1],
                    static_cast<size_t>(token) * checkpoint->nb[1]);
                src = src
                    ? ggml_concat(ctx, src, checkpoint_token, 1)
                    : checkpoint_token;
            }
            ggml_tensor * cur_token = ggml_view_2d(
                ctx, cur, n_embd, 1, cur->nb[1],
                static_cast<size_t>(token) * cur->nb[1]);

            ggml_tensor * score_src = rms_norm(ctx, src, score_weight, eps);
            score_src = ggml_reshape_2d(
                ctx, ggml_sum_rows(ctx, score_src), n, 1);
            ggml_tensor * score_cur = ggml_sum_rows(
                ctx, rms_norm(ctx, cur_token, score_weight, eps));
            ggml_tensor * probs = ggml_soft_max(
                ctx, ggml_concat(ctx, score_src, score_cur, 0));
            ggml_tensor * p_src = ggml_cont(ctx,
                ggml_view_2d(ctx, probs, n, 1, probs->nb[1], 0));
            ggml_tensor * p_cur = ggml_cont(ctx,
                ggml_view_2d(ctx, probs, 1, 1, probs->nb[1],
                             probs->nb[0] * static_cast<size_t>(n)));

            ggml_tensor * src_t = ggml_cont(
                ctx, ggml_permute(ctx, src, 1, 0, 2, 3));
            ggml_tensor * out_token = ggml_add(
                ctx, ggml_mul_mat(ctx, src_t, p_src),
                ggml_mul(ctx, cur_token, p_cur));
            mixed = mixed
                ? ggml_concat(ctx, mixed, out_token, 1)
                : out_token;
        }
        return mixed;
    }
};

ggml_tensor * kda_conv1d(ggml_context * ctx,
                         ggml_cgraph * graph,
                         ggml_tensor * all_state,
                         int qkv,
                         ggml_tensor * x,
                         ggml_tensor * projection,
                         ggml_tensor * conv_weight,
                         int d_conv,
                         int head_dim,
                         int n_head,
                         bool commit_state) {
    const int64_t d_inner = static_cast<int64_t>(head_dim) * n_head;
    const int64_t state_rows = d_conv - 1;
    const int64_t n_tokens = x->ne[1];
    const size_t block_offset = static_cast<size_t>(qkv) * d_inner * all_state->nb[1];
    ggml_tensor * state = ggml_view_3d(ctx, all_state,
        state_rows, d_inner, 1, all_state->nb[1], all_state->nb[2],
        block_offset);

    ggml_tensor * projected = ggml_mul_mat(ctx, projection, x);
    projected = ggml_reshape_3d(ctx, projected, d_inner, n_tokens, 1);
    ggml_tensor * conv_input = ggml_concat(ctx, state,
                                           ggml_transpose(ctx, projected), 0);

    // Drop the oldest row and persist the newest d_conv-1 values.
    if (commit_state) {
        ggml_tensor * newest = ggml_view_3d(ctx, conv_input,
            state_rows, d_inner, 1, conv_input->nb[1], conv_input->nb[2],
            static_cast<size_t>(n_tokens) * conv_input->nb[0]);
        ggml_build_forward_expand(graph, ggml_cpy(ctx, newest, state));
    }

    ggml_tensor * cw = ggml_reshape_2d(ctx, conv_weight, d_conv, d_inner);
    ggml_tensor * out = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_input, cw));
    out = ggml_reshape_4d(ctx, out, head_dim, n_head, n_tokens, 1);
    return out;
}

ggml_tensor * build_kda(ggml_context * ctx,
                        ggml_cgraph * graph,
                        const KimiK3Weights & w,
                        const KimiK3Layer & layer,
                        KimiK3LayerCache & cache,
                        ggml_tensor * cur,
                        bool commit_state,
                        bool capture_replay) {
    const int head_dim = w.kda_head_dim;
    const int n_head = w.n_head;
    const int n_tokens = static_cast<int>(cur->ne[1]);
    const int64_t d_inner = static_cast<int64_t>(head_dim) * n_head;

    if (capture_replay) {
        GGML_ASSERT(cache.replay_input != nullptr);
        GGML_ASSERT(n_tokens <= cache.replay_input->ne[1]);
        ggml_tensor * replay_dst = ggml_view_2d(
            ctx, cache.replay_input, w.n_embd, n_tokens,
            cache.replay_input->nb[1], 0);
        ggml_build_forward_expand(graph, ggml_cpy(ctx, cur, replay_dst));
    }

    ggml_tensor * q = kda_conv1d(ctx, graph, cache.conv_state, 0, cur,
        layer.wq, layer.ssm_q_conv, w.ssm_d_conv, head_dim, n_head,
        commit_state);
    ggml_tensor * k = kda_conv1d(ctx, graph, cache.conv_state, 1, cur,
        layer.wk, layer.ssm_k_conv, w.ssm_d_conv, head_dim, n_head,
        commit_state);
    ggml_tensor * v = kda_conv1d(ctx, graph, cache.conv_state, 2, cur,
        layer.wv, layer.ssm_v_conv, w.ssm_d_conv, head_dim, n_head,
        commit_state);

    ggml_tensor * decay = ggml_mul_mat(ctx, layer.ssm_f_a, cur);
    decay = ggml_mul_mat(ctx, layer.ssm_f_b, decay);
    decay = ggml_add(ctx, decay, layer.ssm_dt_b);
    ggml_tensor * A = ggml_reshape_3d(ctx, layer.ssm_a, 1, n_head, 1);
    if (std::isfinite(w.kda_gate_lower_bound)) {
        decay = ggml_reshape_3d(ctx, decay, head_dim, n_head, n_tokens);
        decay = ggml_mul(ctx, decay, A);
        decay = ggml_sigmoid(ctx, ggml_scale(ctx, decay, -1.0f));
        decay = ggml_scale(ctx, decay, w.kda_gate_lower_bound);
    } else {
        decay = ggml_softplus(ctx, decay);
        decay = ggml_reshape_3d(ctx, decay, head_dim, n_head, n_tokens);
        decay = ggml_mul(ctx, decay, A);
    }
    decay = ggml_reshape_4d(ctx, decay, head_dim, n_head, n_tokens, 1);

    ggml_tensor * beta = ggml_mul_mat(ctx, layer.ssm_beta, cur);
    beta = ggml_sigmoid(ctx,
        ggml_reshape_4d(ctx, beta, 1, n_head, n_tokens, 1));

    q = ggml_l2_norm(ctx, q, w.rms_eps);
    k = ggml_l2_norm(ctx, k, w.rms_eps);
    ggml_tensor * state = ggml_reshape_4d(ctx, cache.ssm_state,
        head_dim, head_dim, n_head, 1);
    ggml_tensor * packed = ggml_gated_delta_net(ctx, q, k, v, decay, beta, state);
    ggml_gated_delta_net_set_skip_intermediate(packed, true);

    const size_t elt = ggml_element_size(packed);
    ggml_tensor * output = ggml_view_4d(ctx, packed,
        head_dim, n_head, n_tokens, 1,
        static_cast<size_t>(head_dim) * elt,
        static_cast<size_t>(head_dim) * n_head * elt,
        static_cast<size_t>(head_dim) * n_head * n_tokens * elt, 0);
    ggml_tensor * new_state = ggml_view_4d(ctx, packed,
        head_dim, head_dim, n_head, 1,
        static_cast<size_t>(head_dim) * elt,
        static_cast<size_t>(head_dim) * head_dim * elt,
        static_cast<size_t>(head_dim) * head_dim * n_head * elt,
        static_cast<size_t>(head_dim) * n_head * n_tokens * elt);
    if (commit_state) {
        ggml_build_forward_expand(graph,
            ggml_cpy(ctx, new_state, cache.ssm_state));
    }

    ggml_tensor * gate = ggml_mul_mat(ctx, layer.ssm_g, cur);
    gate = ggml_reshape_3d(ctx, gate, head_dim, n_head, n_tokens);
    output = ggml_reshape_3d(ctx, output, head_dim, n_head, n_tokens);
    output = rms_norm(ctx, output, layer.ssm_o_norm, w.rms_eps);
    output = ggml_mul(ctx, output, ggml_sigmoid(ctx, gate));
    output = ggml_cont_2d(ctx, output, d_inner, n_tokens);
    return ggml_mul_mat(ctx, layer.wo, output);
}

ggml_tensor * build_mla(ggml_context * ctx,
                        ggml_cgraph * graph,
                        const KimiK3Weights & w,
                        const KimiK3Layer & layer,
                        KimiK3LayerCache & cache,
                        ggml_tensor * cur,
                        int position,
                        ggml_tensor * attn_mask) {
    const int n_head = w.n_head;
    const int kv_rank = w.kv_lora_rank;
    const int key_dim = w.mla_k_head_dim;
    const int value_dim = w.mla_v_head_dim;
    const int rope_dim = w.rope_dim;
    const int nope_dim = key_dim - rope_dim;
    const int compact_dim = kv_rank + rope_dim;
    const int n_tokens = static_cast<int>(cur->ne[1]);
    const int kv_len = position + n_tokens;

    ggml_tensor * gate_input = cur;
    ggml_tensor * q_cur = nullptr;
    if (layer.wq_a) {
        q_cur = ggml_mul_mat(ctx, layer.wq_a, cur);
        q_cur = rms_norm(ctx, q_cur, layer.wq_a_norm, w.rms_eps);
        q_cur = ggml_mul_mat(ctx, layer.wq_b, q_cur);
    } else {
        q_cur = ggml_mul_mat(ctx, layer.wq, cur);
    }

    ggml_tensor * compact_pe = ggml_mul_mat(ctx, layer.wkv_a_mqa, cur);
    ggml_tensor * compact = ggml_view_2d(ctx, compact_pe, kv_rank, n_tokens,
        ggml_row_size(compact_pe->type, compact_dim), 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx, compact_pe, rope_dim, n_tokens, 1,
        ggml_row_size(compact_pe->type, compact_dim),
        ggml_row_size(compact_pe->type, compact_dim) * n_tokens,
        ggml_row_size(compact_pe->type, kv_rank));
    compact = rms_norm(ctx, compact, layer.wkv_a_norm, w.rms_eps);

    ggml_tensor * q_nope = ggml_view_3d(ctx, q_cur, nope_dim, n_head, n_tokens,
        ggml_row_size(q_cur->type, key_dim),
        ggml_row_size(q_cur->type, key_dim) * n_head, 0);
    ggml_tensor * q_pe = ggml_view_3d(ctx, q_cur, rope_dim, n_head, n_tokens,
        ggml_row_size(q_cur->type, key_dim),
        ggml_row_size(q_cur->type, key_dim) * n_head,
        ggml_row_size(q_cur->type, nope_dim));
    q_nope = ggml_permute(ctx, q_nope, 0, 2, 1, 3);
    q_nope = ggml_mul_mat(ctx, layer.wk_b, q_nope);
    q_nope = ggml_permute(ctx, q_nope, 0, 2, 1, 3);
    ggml_tensor * q = ggml_concat(ctx, q_nope, q_pe, 0);

    ggml_tensor * compact_3d =
        ggml_reshape_3d(ctx, compact, kv_rank, n_tokens, 1);
    ggml_tensor * current_k = ggml_concat(ctx, compact_3d, k_pe, 0);

    ggml_tensor * dst = ggml_view_2d(ctx, cache.mla_k,
        compact_dim, n_tokens, cache.mla_k->nb[2],
        static_cast<size_t>(position) * cache.mla_k->nb[2]);
    ggml_build_forward_expand(graph, ggml_cpy(ctx, current_k, dst));

    ggml_tensor * k = ggml_view_3d(ctx, cache.mla_k,
        compact_dim, 1, kv_len, cache.mla_k->nb[1], cache.mla_k->nb[2], 0);
    ggml_tensor * v = ggml_view_3d(ctx, k,
        kv_rank, 1, kv_len, k->nb[1], k->nb[2], 0);

    // Same non-flash absorbed-MLA algebra as llama.cpp. Avoiding flash here
    // is intentional: current upstream K3 support also disables it for this
    // graph, and this path is the numerical oracle for later fused kernels.
    const bool v_trans = v->nb[1] > v->nb[2];
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = ggml_soft_max_ext(ctx, scores, attn_mask,
                               1.0f / std::sqrt(static_cast<float>(key_dim)),
                               0.0f);
    if (!v_trans) v = ggml_cont(ctx, ggml_transpose(ctx, v));
    ggml_tensor * out = ggml_mul_mat(ctx, v, scores);
    out = ggml_mul_mat(ctx, layer.wv_b, out);
    out = ggml_permute(ctx, out, 0, 2, 1, 3);
    out = ggml_cont_2d(ctx, out,
                       static_cast<int64_t>(value_dim) * n_head, n_tokens);

    if (layer.wqkv_gate) {
        ggml_tensor * output_gate = ggml_sigmoid(ctx,
            ggml_mul_mat(ctx, layer.wqkv_gate, gate_input));
        out = ggml_mul(ctx, out, output_gate);
    }
    return ggml_mul_mat(ctx, layer.wo, out);
}

TopKMoeRouterResult build_kimi_router(ggml_context * ctx,
                                      ggml_cgraph * graph,
                                      const KimiK3Weights & w,
                                      const KimiK3Layer & layer,
                                      ggml_tensor * cur) {
    const int n_tokens = static_cast<int>(cur->ne[1]);
    ggml_tensor * logits = ggml_mul_mat(ctx, layer.ffn_gate_inp, cur);
    TopKMoeRouterResult router;
    if (w.expert_gating_func == 2) {
        router = build_sigmoid_topk_moe_router(ctx, graph, logits,
            layer.ffn_exp_probs_b, w.n_expert, w.n_expert_used, n_tokens,
            w.expert_weights_norm, w.expert_weights_scale, false);
    } else {
        ggml_tensor * probs = ggml_soft_max(ctx, logits);
        ggml_tensor * selected = ggml_argsort_top_k(ctx, probs, w.n_expert_used);
        ggml_tensor * probs_3d =
            ggml_reshape_3d(ctx, probs, 1, w.n_expert, n_tokens);
        ggml_tensor * weights = ggml_get_rows(ctx, probs_3d, selected);
        weights = ggml_reshape_2d(
            ctx, weights, w.n_expert_used, n_tokens);
        if (w.expert_weights_norm) {
            ggml_tensor * sum = ggml_clamp(ctx, ggml_sum_rows(ctx, weights),
                                           6.103515625e-5f, INFINITY);
            weights = ggml_div(ctx, weights, sum);
        }
        if (w.expert_weights_scale != 1.0f) {
            weights = ggml_scale(ctx, weights, w.expert_weights_scale);
        }
        router.selected = selected;
        router.weights_2d = weights;
        router.weights_3d = ggml_reshape_3d(
            ctx, weights, 1, w.n_expert_used, n_tokens);
    }
    return router;
}

ggml_tensor * build_latent_moe(ggml_context * ctx,
                               ggml_cgraph * graph,
                               const KimiK3Weights & w,
                               const KimiK3Layer & layer,
                               ggml_tensor * cur) {
    ggml_tensor * identity = cur;
    ggml_tensor * routed_in = ggml_mul_mat(ctx, layer.ffn_routed_down, cur);
    TopKMoeRouterResult router =
        build_kimi_router(ctx, graph, w, layer, identity);

    ggml_tensor * routed_3d = ggml_reshape_3d(ctx, routed_in,
                                               w.n_expert_latent,
                                               1, cur->ne[1]);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, layer.ffn_gate_exps,
                                         routed_3d, router.selected);
    ggml_tensor * up = ggml_mul_mat_id(ctx, layer.ffn_up_exps,
                                       routed_3d, router.selected);
    ggml_tensor * activated = situ(ctx, gate, up,
                                    w.situ_beta, w.situ_linear_beta);
    ggml_tensor * experts = ggml_mul_mat_id(ctx, layer.ffn_down_exps,
                                            activated, router.selected);
    experts = ggml_mul(ctx, experts, router.weights_3d);
    ggml_tensor * sum_shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                 w.n_expert_latent,
                                                 1, cur->ne[1]);
    ggml_tensor * moe = ggml_repeat_back(ctx, experts, sum_shape);
    moe = ggml_reshape_2d(ctx, moe, w.n_expert_latent, cur->ne[1]);
    if (layer.ffn_routed_norm) {
        moe = rms_norm(ctx, moe, layer.ffn_routed_norm, w.rms_eps);
    }
    moe = ggml_mul_mat(ctx, layer.ffn_routed_up, moe);

    ggml_tensor * shared_gate = ggml_mul_mat(ctx, layer.ffn_gate_shexp, identity);
    ggml_tensor * shared_up = ggml_mul_mat(ctx, layer.ffn_up_shexp, identity);
    ggml_tensor * shared = situ(ctx, shared_gate, shared_up,
                                w.situ_beta, w.situ_linear_beta);
    shared = ggml_mul_mat(ctx, layer.ffn_down_shexp, shared);
    return ggml_add(ctx, moe, shared);
}

struct GraphInput {
    ggml_tensor * tensor = nullptr;
    const void * data = nullptr;
    size_t bytes = 0;
};

struct GraphOutput {
    ggml_tensor * tensor = nullptr;
    void * data = nullptr;
    size_t bytes = 0;
};

bool run_host_boundary_graph(ggml_backend_t backend,
                             ggml_context * ctx,
                             ggml_cgraph * graph,
                             const std::vector<GraphInput> & inputs,
                             const std::vector<GraphOutput> & outputs,
                             const char * phase) {
    for (const GraphOutput & output : outputs) {
        if (!output.tensor || !output.data || output.bytes == 0) {
            set_last_error(std::string("Kimi-K3 ") + phase +
                           ": invalid graph output");
            return false;
        }
        ggml_set_output(output.tensor);
        ggml_build_forward_expand(graph, output.tensor);
    }
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        set_last_error(std::string("Kimi-K3 ") + phase +
                       ": graph allocation failed");
        if (allocator) ggml_gallocr_free(allocator);
        return false;
    }
    for (const GraphInput & input : inputs) {
        if (!input.tensor || !input.data || input.bytes == 0) {
            set_last_error(std::string("Kimi-K3 ") + phase +
                           ": invalid graph input");
            ggml_gallocr_free(allocator);
            return false;
        }
        ggml_backend_tensor_set(
            input.tensor, input.data, 0, input.bytes);
    }
    const ggml_status status =
        ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        set_last_error(std::string("Kimi-K3 ") + phase +
                       ": graph compute failed with status " +
                       std::to_string(static_cast<int>(status)));
        ggml_gallocr_free(allocator);
        return false;
    }
    for (const GraphOutput & output : outputs) {
        ggml_backend_tensor_get(
            output.tensor, output.data, 0, output.bytes);
    }
    ggml_gallocr_free(allocator);
    return true;
}

void populate_attn_res_bank(
        ggml_context * ctx,
        const KimiK3Weights & w,
        int n_tokens,
        const std::vector<std::vector<float>> & host_checkpoints,
        AttnResBank & bank,
        std::vector<GraphInput> & inputs) {
    bank.ctx = ctx;
    bank.eps = w.rms_eps;
    bank.n_embd = w.n_embd;
    bank.n_tokens = n_tokens;
    for (const std::vector<float> & checkpoint : host_checkpoints) {
        ggml_tensor * tensor = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
        ggml_set_input(tensor);
        inputs.push_back({
            tensor, checkpoint.data(),
            checkpoint.size() * sizeof(float)});
        bank.push(tensor);
    }
}

ggml_context * new_kimi_step_context() {
    ggml_init_params params{};
    params.mem_size = 64ull * 1024ull * 1024ull;
    params.no_alloc = true;
    return ggml_init(params);
}

bool streamed_kimi_k3_forward(
        ggml_backend_t backend,
        const KimiK3Weights & w,
        KimiK3Cache & cache,
        const std::vector<int32_t> & tokens,
        int base_pos,
        const KimiK3ForwardOptions & options,
        KimiK3ForwardResult & result,
        MoeHybridStreamEngine & stream_engine,
        MoeStreamDualOwnerExecutor * dual_stream_executor,
        const MoeStreamDualOwnerPolicy * stream_owner_policy,
        MoeHybridRoutingStats * routing_stats) {
    const int n_tokens = static_cast<int>(tokens.size());
    const size_t hidden_values =
        static_cast<size_t>(w.n_embd) * static_cast<size_t>(n_tokens);
    std::vector<float> hidden(hidden_values);

    std::vector<int> capture_at_layer(static_cast<size_t>(w.n_layer), -1);
    const int n_capture = options.capture_layer_ids
        ? static_cast<int>(options.capture_layer_ids->size()) : 0;
    for (int i = 0; i < n_capture; ++i) {
        capture_at_layer[static_cast<size_t>((*options.capture_layer_ids)[i])] = i;
    }
    result.captured_hidden.assign(
        static_cast<size_t>(n_capture) * hidden_values, 0.0f);

    const int kv_len = base_pos + n_tokens;
    std::vector<float> mla_mask(
        static_cast<size_t>(kv_len) * n_tokens, -INFINITY);
    for (int q = 0; q < n_tokens; ++q) {
        for (int k = 0; k <= base_pos + q; ++k) {
            mla_mask[static_cast<size_t>(q) * kv_len + k] = 0.0f;
        }
    }

    {
        ggml_context * ctx = new_kimi_step_context();
        if (!ctx) {
            set_last_error("Kimi-K3 embedding: context allocation failed");
            return false;
        }
        ggml_cgraph * graph =
            ggml_new_graph_custom(ctx, 1024, false);
        ggml_tensor * ids =
            ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(ids);
        ggml_tensor * embedding =
            ggml_get_rows(ctx, w.tok_embd, ids);
        const bool ok = run_host_boundary_graph(
            backend, ctx, graph,
            {{ids, tokens.data(), sizeof(int32_t) * tokens.size()}},
            {{embedding, hidden.data(),
              hidden.size() * sizeof(float)}},
            "embedding");
        ggml_free(ctx);
        if (!ok) return false;
    }

    std::vector<std::vector<float>> checkpoints;
    checkpoints.reserve(
        static_cast<size_t>(
            (w.n_layer + w.attn_res_block_size - 1) /
            w.attn_res_block_size));

    for (int il = 0; il < w.n_layer; ++il) {
        const KimiK3Layer & layer =
            w.layers[static_cast<size_t>(il)];
        KimiK3LayerCache & layer_cache =
            cache.layers[static_cast<size_t>(il)];
        const bool banked =
            il % w.attn_res_block_size == 0;
        const std::vector<float> checkpoint_value = hidden;

        ggml_context * ctx = new_kimi_step_context();
        if (!ctx) {
            set_last_error("Kimi-K3 layer: context allocation failed");
            return false;
        }
        ggml_cgraph * graph =
            ggml_new_graph_custom(ctx, 32768, false);
        std::vector<GraphInput> inputs;
        ggml_tensor * hidden_in = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
        ggml_set_input(hidden_in);
        inputs.push_back({
            hidden_in, hidden.data(),
            hidden.size() * sizeof(float)});

        AttnResBank residuals;
        populate_attn_res_bank(
            ctx, w, n_tokens, checkpoints, residuals, inputs);
        ggml_tensor * prefix = hidden_in;
        ggml_tensor * cur =
            residuals.mix(prefix, layer.attn_res_score);
        if (banked) residuals.push(prefix);

        cur = rms_norm(
            ctx, cur, layer.attn_norm, w.rms_eps);
        if (layer.recurrent) {
            cur = build_kda(
                ctx, graph, w, layer, layer_cache, cur,
                /*commit_state=*/!options.capture_replay,
                options.capture_replay);
        } else {
            ggml_tensor * mask = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, kv_len, n_tokens);
            ggml_set_input(mask);
            inputs.push_back({
                mask, mla_mask.data(), mla_mask.size() * sizeof(float)});
            cur = build_mla(
                ctx, graph, w, layer, layer_cache,
                cur, base_pos, mask);
        }
        prefix = banked
            ? cur : ggml_add(ctx, prefix, cur);
        cur = residuals.mix(prefix, layer.ffn_res_score);
        cur = rms_norm(
            ctx, cur, layer.ffn_norm, w.rms_eps);

        if (il < w.n_dense_lead) {
            ggml_tensor * gate =
                ggml_mul_mat(ctx, layer.ffn_gate, cur);
            ggml_tensor * up =
                ggml_mul_mat(ctx, layer.ffn_up, cur);
            ggml_tensor * dense = situ(
                ctx, gate, up,
                w.situ_beta, w.situ_linear_beta);
            dense = ggml_mul_mat(
                ctx, layer.ffn_down, dense);
            ggml_tensor * hidden_out =
                ggml_add(ctx, prefix, dense);
            std::vector<float> next_hidden(hidden_values);
            const bool ok = run_host_boundary_graph(
                backend, ctx, graph, inputs,
                {{hidden_out, next_hidden.data(),
                  next_hidden.size() * sizeof(float)}},
                "dense layer");
            ggml_free(ctx);
            if (!ok) return false;
            if (banked) checkpoints.push_back(checkpoint_value);
            hidden.swap(next_hidden);
            const int capture_idx = capture_at_layer[static_cast<size_t>(il)];
            if (capture_idx >= 0) {
                std::memcpy(
                    result.captured_hidden.data() +
                        static_cast<size_t>(capture_idx) * hidden_values,
                    hidden.data(), hidden_values * sizeof(float));
            }
            continue;
        }

        ggml_tensor * routed_in =
            ggml_mul_mat(ctx, layer.ffn_routed_down, cur);
        TopKMoeRouterResult router =
            build_kimi_router(ctx, graph, w, layer, cur);
        // argsort_top_k returns a strided view of the full argsort result.
        // Materialize the tiny host-boundary tensors so the graph allocator
        // cannot recycle their backing storage before the readback.
        ggml_tensor * selected_out =
            ggml_cont(ctx, router.selected);
        ggml_tensor * route_weights_out =
            ggml_cont(ctx, router.weights_2d);
        ggml_tensor * shared_gate =
            ggml_mul_mat(ctx, layer.ffn_gate_shexp, cur);
        ggml_tensor * shared_up =
            ggml_mul_mat(ctx, layer.ffn_up_shexp, cur);
        ggml_tensor * shared = situ(
            ctx, shared_gate, shared_up,
            w.situ_beta, w.situ_linear_beta);
        shared = ggml_mul_mat(
            ctx, layer.ffn_down_shexp, shared);

        std::vector<float> prefix_host(hidden_values);
        std::vector<float> routed_input_host(
            static_cast<size_t>(w.n_expert_latent) * n_tokens);
        std::vector<int32_t> selected(
            static_cast<size_t>(w.n_expert_used) * n_tokens);
        std::vector<float> route_weights(
            static_cast<size_t>(w.n_expert_used) * n_tokens);
        std::vector<float> shared_host(hidden_values);
        const bool prep_ok = run_host_boundary_graph(
            backend, ctx, graph, inputs,
            {
                {prefix, prefix_host.data(),
                 prefix_host.size() * sizeof(float)},
                {routed_in, routed_input_host.data(),
                 routed_input_host.size() * sizeof(float)},
                {selected_out, selected.data(),
                 selected.size() * sizeof(int32_t)},
                {route_weights_out, route_weights.data(),
                 route_weights.size() * sizeof(float)},
                {shared, shared_host.data(),
                 shared_host.size() * sizeof(float)},
            },
            "routed layer preparation");
        ggml_free(ctx);
        if (!prep_ok) return false;
        if (banked) checkpoints.push_back(checkpoint_value);
        for (size_t route = 0; route < selected.size(); ++route) {
            if (selected[route] < 0 || selected[route] >= w.n_expert) {
                set_last_error(
                    "Kimi-K3 routed layer " + std::to_string(il) +
                    ": native router returned invalid expert " +
                    std::to_string(selected[route]) + " at route " +
                    std::to_string(route));
                return false;
            }
            if (!std::isfinite(route_weights[route])) {
                set_last_error(
                    "Kimi-K3 routed layer " + std::to_string(il) +
                    ": native router returned a non-finite weight");
                return false;
            }
        }

        MoeStreamExpertSpec spec;
        spec.input_dim = w.n_expert_latent;
        spec.intermediate_dim = w.n_ff_exp;
        spec.output_dim = w.n_expert_latent;
        spec.gate_type = layer.ffn_gate_exps->type;
        spec.up_type = layer.ffn_up_exps->type;
        spec.down_type = layer.ffn_down_exps->type;
        spec.gated_activation = MoeGatedActivation::Situ;
        spec.situ_beta = w.situ_beta;
        spec.situ_linear_beta = w.situ_linear_beta;

        MoeStreamRouteBatch route_batch;
        route_batch.layer = il - w.n_dense_lead;
        route_batch.n_expert = w.n_expert;
        route_batch.top_k = w.n_expert_used;
        route_batch.n_tokens = n_tokens;
        route_batch.inputs = routed_input_host.data();
        route_batch.selected_ids = selected.data();
        route_batch.selected_weights = route_weights.data();
        if (routing_stats && !routing_stats->observe(
                route_batch.layer, selected.data(),
                static_cast<int>(selected.size()))) {
            set_last_error(
                "Kimi-K3 routed layer " + std::to_string(il) +
                ": failed to record native route statistics");
            return false;
        }
        std::vector<float> routed_output;
        std::string stream_error;
        MoeStreamDualOwnerStats owner_stats;
        const bool dual_owner = dual_stream_executor != nullptr;
        const bool route_ok = dual_owner
            ? dual_stream_executor->eval(
                spec, route_batch, *stream_owner_policy,
                routed_output, &owner_stats, &stream_error)
            : eval_moe_streamed_experts(
                stream_engine, spec, route_batch,
                routed_output, &stream_error);
        if (!route_ok) {
            set_last_error(
                "Kimi-K3 routed layer " +
                std::to_string(il) +
                ": streamed expert evaluation failed: " +
                stream_error);
            return false;
        }
        const char * trace = std::getenv("DFLASH_MOE_DUAL_STREAM_TRACE");
        if (dual_owner && trace && *trace && std::strcmp(trace, "0") != 0) {
            std::fprintf(stderr,
                "[kimi-k3] dual-owner layer=%d routes=%d/%d experts=%d/%d "
                "primary=%.3fms secondary=%.3fms wall=%.3fms\n",
                route_batch.layer,
                owner_stats.primary_routes, owner_stats.secondary_routes,
                owner_stats.primary_experts, owner_stats.secondary_experts,
                owner_stats.primary_us / 1000.0,
                owner_stats.secondary_us / 1000.0,
                owner_stats.wall_us / 1000.0);
        }

        ctx = new_kimi_step_context();
        if (!ctx) {
            set_last_error(
                "Kimi-K3 routed layer join: context allocation failed");
            return false;
        }
        graph = ggml_new_graph_custom(ctx, 4096, false);
        ggml_tensor * prefix_in = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
        ggml_tensor * routed_out_in = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, w.n_expert_latent, n_tokens);
        ggml_tensor * shared_in = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
        ggml_set_input(prefix_in);
        ggml_set_input(routed_out_in);
        ggml_set_input(shared_in);
        ggml_tensor * routed = routed_out_in;
        if (layer.ffn_routed_norm) {
            routed = rms_norm(
                ctx, routed, layer.ffn_routed_norm, w.rms_eps);
        }
        routed = ggml_mul_mat(
            ctx, layer.ffn_routed_up, routed);
        ggml_tensor * moe_shared =
            ggml_add(ctx, routed, shared_in);
        ggml_tensor * hidden_out =
            ggml_add(ctx, prefix_in, moe_shared);
        std::vector<float> next_hidden(hidden_values);
        const bool join_ok = run_host_boundary_graph(
            backend, ctx, graph,
            {
                {prefix_in, prefix_host.data(),
                 prefix_host.size() * sizeof(float)},
                {routed_out_in, routed_output.data(),
                 routed_output.size() * sizeof(float)},
                {shared_in, shared_host.data(),
                 shared_host.size() * sizeof(float)},
            },
            {{hidden_out, next_hidden.data(),
              next_hidden.size() * sizeof(float)}},
            "routed layer join");
        ggml_free(ctx);
        if (!join_ok) return false;
        hidden.swap(next_hidden);
        const int capture_idx = capture_at_layer[static_cast<size_t>(il)];
        if (capture_idx >= 0) {
            std::memcpy(
                result.captured_hidden.data() +
                    static_cast<size_t>(capture_idx) * hidden_values,
                hidden.data(), hidden_values * sizeof(float));
        }
    }

    ggml_context * ctx = new_kimi_step_context();
    if (!ctx) {
        set_last_error("Kimi-K3 output: context allocation failed");
        return false;
    }
    ggml_cgraph * graph =
        ggml_new_graph_custom(ctx, 8192, false);
    std::vector<GraphInput> inputs;
    ggml_tensor * hidden_in = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_set_input(hidden_in);
    inputs.push_back({
        hidden_in, hidden.data(),
        hidden.size() * sizeof(float)});
    AttnResBank residuals;
    populate_attn_res_bank(
        ctx, w, n_tokens, checkpoints, residuals, inputs);
    ggml_tensor * output_hidden =
        residuals.mix(hidden_in, w.output_res_score);
    output_hidden = rms_norm(
        ctx, output_hidden, w.output_norm, w.rms_eps);
    ggml_tensor * output =
        ggml_mul_mat(ctx, w.output, output_hidden);
    ggml_tensor * argmax = ggml_argmax(ctx, output);
    std::vector<GraphOutput> outputs;
    if (options.read_logits) {
        result.logits.resize(static_cast<size_t>(w.n_vocab) * n_tokens);
        outputs.push_back({
            output, result.logits.data(), result.logits.size() * sizeof(float)});
    }
    if (options.read_argmax) {
        result.argmax.resize(static_cast<size_t>(n_tokens));
        outputs.push_back({
            argmax, result.argmax.data(), result.argmax.size() * sizeof(int32_t)});
    }
    const bool output_ok = run_host_boundary_graph(
        backend, ctx, graph, inputs,
        outputs,
        "output");
    ggml_free(ctx);
    if (!output_ok) return false;

    cache.cur_pos = base_pos + n_tokens;
    return true;
}

} // namespace

bool create_kimi_k3_cache(ggml_backend_t backend,
                          const KimiK3Weights & w,
                          int max_ctx,
                          KimiK3Cache & out,
                          int max_verify_tokens) {
    free_kimi_k3_cache(out);
    if (!backend || max_ctx <= 0) return false;

    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() *
        static_cast<size_t>(w.n_layer * 6 + 16) + 16384;
    params.no_alloc = true;
    out.ctx = ggml_init(params);
    if (!out.ctx) return false;

    out.layers.resize(static_cast<size_t>(w.n_layer));
    const int64_t d_inner = static_cast<int64_t>(w.kda_head_dim) * w.n_head;
    const int compact_dim = w.kv_lora_rank + w.rope_dim;
    for (int il = 0; il < w.n_layer; ++il) {
        KimiK3LayerCache & layer_cache = out.layers[static_cast<size_t>(il)];
        char name[80];
        if (w.layers[static_cast<size_t>(il)].recurrent) {
            layer_cache.conv_state = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32,
                w.ssm_d_conv - 1, 3 * d_inner);
            layer_cache.ssm_state = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32,
                w.kda_head_dim, w.kda_head_dim, w.n_head);
            std::snprintf(name, sizeof(name), "kimi_k3_conv_state_%d", il);
            ggml_set_name(layer_cache.conv_state, name);
            std::snprintf(name, sizeof(name), "kimi_k3_ssm_state_%d", il);
            ggml_set_name(layer_cache.ssm_state, name);
            if (max_verify_tokens > 0) {
                layer_cache.conv_state_snap = ggml_dup_tensor(
                    out.ctx, layer_cache.conv_state);
                layer_cache.ssm_state_snap = ggml_dup_tensor(
                    out.ctx, layer_cache.ssm_state);
                layer_cache.replay_input = ggml_new_tensor_2d(
                    out.ctx, GGML_TYPE_F32, w.n_embd, max_verify_tokens);
                std::snprintf(
                    name, sizeof(name), "kimi_k3_conv_state_snap_%d", il);
                ggml_set_name(layer_cache.conv_state_snap, name);
                std::snprintf(
                    name, sizeof(name), "kimi_k3_ssm_state_snap_%d", il);
                ggml_set_name(layer_cache.ssm_state_snap, name);
                std::snprintf(
                    name, sizeof(name), "kimi_k3_replay_input_%d", il);
                ggml_set_name(layer_cache.replay_input, name);
            }
        } else {
            layer_cache.mla_k = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F16,
                compact_dim, 1, max_ctx);
            std::snprintf(name, sizeof(name), "kimi_k3_mla_k_%d", il);
            ggml_set_name(layer_cache.mla_k, name);
        }
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        free_kimi_k3_cache(out);
        return false;
    }
    out.max_ctx = max_ctx;
    out.max_verify_tokens = std::max(0, max_verify_tokens);
    reset_kimi_k3_cache(out);
    return true;
}

void reset_kimi_k3_cache(KimiK3Cache & cache) {
    if (cache.buf) ggml_backend_buffer_clear(cache.buf, 0);
    cache.cur_pos = 0;
    cache.snapshot_pos = -1;
    cache.replay_base_pos = -1;
    cache.replay_n_tokens = 0;
    cache.snapshot_valid = false;
    cache.replay_valid = false;
    cache.recurrent_state_pristine = false;
}

void free_kimi_k3_cache(KimiK3Cache & cache) {
    if (cache.buf) ggml_backend_buffer_free(cache.buf);
    if (cache.ctx) ggml_free(cache.ctx);
    cache = KimiK3Cache{};
}

bool kimi_k3_forward(ggml_backend_t backend,
                     const KimiK3Weights & w,
                     KimiK3Cache & cache,
                     const std::vector<int32_t> & tokens,
                     int base_pos,
                     const KimiK3ForwardOptions & options,
                     KimiK3ForwardResult & result,
                     MoeHybridStreamEngine * stream_engine,
                     MoeStreamDualOwnerExecutor * dual_stream_executor,
                     const MoeStreamDualOwnerPolicy * stream_owner_policy,
                     MoeHybridRoutingStats * routing_stats) {
    result = KimiK3ForwardResult{};
    const int n_tokens = static_cast<int>(tokens.size());
    if (!backend || !w.ctx || !cache.ctx || n_tokens <= 0 || base_pos < 0 ||
        base_pos != cache.cur_pos || base_pos + n_tokens > cache.max_ctx ||
        (!options.read_logits && !options.read_argmax)) {
        set_last_error("Kimi-K3 forward: invalid backend, output, or cache span");
        return false;
    }
    for (int32_t token : tokens) {
        if (token < 0 || token >= w.n_vocab) {
            set_last_error("Kimi-K3 forward: token is outside the vocabulary");
            return false;
        }
    }

    std::vector<int> capture_at_layer(static_cast<size_t>(w.n_layer), -1);
    const int n_capture = options.capture_layer_ids
        ? static_cast<int>(options.capture_layer_ids->size()) : 0;
    for (int i = 0; i < n_capture; ++i) {
        const int layer = (*options.capture_layer_ids)[static_cast<size_t>(i)];
        if (layer < 0 || layer >= w.n_layer ||
            capture_at_layer[static_cast<size_t>(layer)] >= 0) {
            set_last_error("Kimi-K3 forward: invalid or duplicate capture layer");
            return false;
        }
        capture_at_layer[static_cast<size_t>(layer)] = i;
    }
    if (options.capture_replay &&
        (n_tokens > cache.max_verify_tokens || !cache.snapshot_valid ||
         cache.snapshot_pos != base_pos)) {
        set_last_error("Kimi-K3 forward: ReplaySSM capture has no matching snapshot");
        return false;
    }

    if (w.routed_experts_streamed) {
        if (!stream_engine || !stream_engine->is_bound()) {
            set_last_error(
                "Kimi-K3 forward: file-backed experts require a bound stream engine");
            return false;
        }
        if (dual_stream_executor &&
            (!dual_stream_executor->is_ready() || !stream_owner_policy)) {
            set_last_error(
                "Kimi-K3 forward: dual-owner streaming requires a ready "
                "executor and an ownership policy");
            return false;
        }
        if (!streamed_kimi_k3_forward(
                backend, w, cache, tokens, base_pos, options, result,
                *stream_engine, dual_stream_executor,
                stream_owner_policy, routing_stats)) {
            return false;
        }
        if (options.capture_replay) {
            cache.replay_base_pos = base_pos;
            cache.replay_n_tokens = n_tokens;
            cache.replay_valid = true;
            cache.recurrent_state_pristine = true;
        } else {
            cache.replay_valid = false;
            cache.recurrent_state_pristine = false;
        }
        return true;
    }

    const int kv_len = base_pos + n_tokens;
    std::vector<float> mla_mask(
        static_cast<size_t>(kv_len) * n_tokens, -INFINITY);
    for (int q = 0; q < n_tokens; ++q) {
        for (int k = 0; k <= base_pos + q; ++k) {
            mla_mask[static_cast<size_t>(q) * kv_len + k] = 0.0f;
        }
    }

    ggml_init_params params{};
    params.mem_size = 64ull * 1024ull * 1024ull;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        set_last_error("Kimi-K3 forward: graph context allocation failed");
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32768, false);

    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "token_ids");
    ggml_set_input(ids);
    ggml_tensor * hidden = ggml_get_rows(ctx, w.tok_embd, ids);
    ggml_tensor * mask = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, kv_len, n_tokens);
    ggml_set_name(mask, "kimi_k3_mla_causal_mask");
    ggml_set_input(mask);

    std::vector<ggml_tensor *> capture_tensors(
        static_cast<size_t>(n_capture), nullptr);
    AttnResBank residuals;
    residuals.ctx = ctx;
    residuals.eps = w.rms_eps;
    residuals.n_embd = w.n_embd;
    residuals.n_tokens = n_tokens;
    for (int il = 0; il < w.n_layer; ++il) {
        const KimiK3Layer & layer = w.layers[static_cast<size_t>(il)];
        KimiK3LayerCache & layer_cache = cache.layers[static_cast<size_t>(il)];
        ggml_tensor * prefix = hidden;
        ggml_tensor * cur = residuals.mix(prefix, layer.attn_res_score);
        const bool banked = il % w.attn_res_block_size == 0;
        if (banked) residuals.push(prefix);

        cur = rms_norm(ctx, cur, layer.attn_norm, w.rms_eps);
        cur = layer.recurrent
            ? build_kda(ctx, graph, w, layer, layer_cache, cur,
                        /*commit_state=*/!options.capture_replay,
                        options.capture_replay)
            : build_mla(ctx, graph, w, layer, layer_cache, cur, base_pos, mask);
        prefix = banked ? cur : ggml_add(ctx, prefix, cur);

        cur = residuals.mix(prefix, layer.ffn_res_score);
        cur = rms_norm(ctx, cur, layer.ffn_norm, w.rms_eps);
        if (il < w.n_dense_lead) {
            ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, cur);
            ggml_tensor * up = ggml_mul_mat(ctx, layer.ffn_up, cur);
            cur = situ(ctx, gate, up, w.situ_beta, w.situ_linear_beta);
            cur = ggml_mul_mat(ctx, layer.ffn_down, cur);
        } else {
            cur = build_latent_moe(ctx, graph, w, layer, cur);
        }
        hidden = ggml_add(ctx, prefix, cur);
        const int capture_idx = capture_at_layer[static_cast<size_t>(il)];
        if (capture_idx >= 0) {
            capture_tensors[static_cast<size_t>(capture_idx)] = hidden;
            ggml_set_output(hidden);
            ggml_build_forward_expand(graph, hidden);
        }
    }

    hidden = residuals.mix(hidden, w.output_res_score);
    hidden = rms_norm(ctx, hidden, w.output_norm, w.rms_eps);
    ggml_tensor * output = ggml_mul_mat(ctx, w.output, hidden);
    ggml_set_name(output, "logits");
    ggml_tensor * argmax = ggml_argmax(ctx, output);
    if (options.read_logits) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
    }
    if (options.read_argmax) {
        ggml_set_output(argmax);
        ggml_build_forward_expand(graph, argmax);
    }

    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        set_last_error("Kimi-K3 forward: graph allocation failed");
        if (allocator) ggml_gallocr_free(allocator);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(
        ids, tokens.data(), 0, sizeof(int32_t) * tokens.size());
    ggml_backend_tensor_set(
        mask, mla_mask.data(), 0, sizeof(float) * mla_mask.size());
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        set_last_error("Kimi-K3 forward: graph compute failed with status " +
                       std::to_string(static_cast<int>(status)));
        ggml_gallocr_free(allocator);
        ggml_free(ctx);
        return false;
    }

    if (options.read_logits) {
        result.logits.resize(static_cast<size_t>(w.n_vocab) * n_tokens);
        ggml_backend_tensor_get(output, result.logits.data(), 0,
                                result.logits.size() * sizeof(float));
    }
    if (options.read_argmax) {
        result.argmax.resize(static_cast<size_t>(n_tokens));
        ggml_backend_tensor_get(argmax, result.argmax.data(), 0,
                                result.argmax.size() * sizeof(int32_t));
    }
    const size_t hidden_values =
        static_cast<size_t>(w.n_embd) * static_cast<size_t>(n_tokens);
    result.captured_hidden.resize(
        static_cast<size_t>(n_capture) * hidden_values);
    for (int i = 0; i < n_capture; ++i) {
        ggml_backend_tensor_get(
            capture_tensors[static_cast<size_t>(i)],
            result.captured_hidden.data() + static_cast<size_t>(i) * hidden_values,
            0, hidden_values * sizeof(float));
    }

    cache.cur_pos = base_pos + n_tokens;
    if (options.capture_replay) {
        cache.replay_base_pos = base_pos;
        cache.replay_n_tokens = n_tokens;
        cache.replay_valid = true;
        cache.recurrent_state_pristine = true;
    } else {
        cache.replay_valid = false;
        cache.recurrent_state_pristine = false;
    }
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return true;
}

bool kimi_k3_replay_snapshot(ggml_backend_t backend, KimiK3Cache & cache) {
    if (!backend || cache.max_verify_tokens <= 0) return false;
    for (KimiK3LayerCache & layer : cache.layers) {
        if (!layer.ssm_state) continue;
        if (!layer.ssm_state_snap || !layer.conv_state_snap ||
            !layer.replay_input) {
            return false;
        }
        ggml_backend_tensor_copy_async(
            backend, backend, layer.ssm_state, layer.ssm_state_snap);
        ggml_backend_tensor_copy_async(
            backend, backend, layer.conv_state, layer.conv_state_snap);
    }
    ggml_backend_synchronize(backend);
    cache.snapshot_pos = cache.cur_pos;
    cache.snapshot_valid = true;
    cache.replay_valid = false;
    cache.recurrent_state_pristine = true;
    return true;
}

bool kimi_k3_replay_restore(ggml_backend_t backend, KimiK3Cache & cache) {
    if (!backend || !cache.snapshot_valid || cache.snapshot_pos < 0) return false;
    if (!cache.recurrent_state_pristine) {
        for (KimiK3LayerCache & layer : cache.layers) {
            if (!layer.ssm_state) continue;
            if (!layer.ssm_state_snap || !layer.conv_state_snap) return false;
            ggml_backend_tensor_copy_async(
                backend, backend, layer.ssm_state_snap, layer.ssm_state);
            ggml_backend_tensor_copy_async(
                backend, backend, layer.conv_state_snap, layer.conv_state);
        }
        ggml_backend_synchronize(backend);
    }
    cache.cur_pos = cache.snapshot_pos;
    cache.replay_valid = false;
    cache.recurrent_state_pristine = true;
    return true;
}

bool kimi_k3_replay_commit(ggml_backend_t backend,
                           const KimiK3Weights & w,
                           KimiK3Cache & cache,
                           int base_pos,
                           int commit_n) {
    if (!backend || !cache.snapshot_valid || !cache.replay_valid ||
        !cache.recurrent_state_pristine || cache.snapshot_pos != base_pos ||
        cache.replay_base_pos != base_pos || commit_n <= 0 ||
        commit_n > cache.replay_n_tokens) {
        return false;
    }

    ggml_init_params params{};
    params.mem_size = 64ull * 1024ull * 1024ull;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32768, false);
    for (int il = 0; il < w.n_layer; ++il) {
        const KimiK3Layer & layer = w.layers[static_cast<size_t>(il)];
        if (!layer.recurrent) continue;
        KimiK3LayerCache & layer_cache = cache.layers[static_cast<size_t>(il)];
        if (!layer_cache.replay_input) {
            ggml_free(ctx);
            return false;
        }
        ggml_tensor * replay = ggml_view_2d(
            ctx, layer_cache.replay_input, w.n_embd, commit_n,
            layer_cache.replay_input->nb[1], 0);
        (void)build_kda(ctx, graph, w, layer, layer_cache, replay,
                        /*commit_state=*/true,
                        /*capture_replay=*/false);
    }

    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        if (allocator) ggml_gallocr_free(allocator);
        ggml_free(ctx);
        return false;
    }
    cache.recurrent_state_pristine = false;
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    if (status != GGML_STATUS_SUCCESS) {
        (void)kimi_k3_replay_restore(backend, cache);
        return false;
    }
    cache.cur_pos = base_pos + commit_n;
    cache.snapshot_valid = false;
    cache.replay_valid = false;
    return true;
}

bool kimi_k3_step(ggml_backend_t backend,
                  const KimiK3Weights & w,
                  KimiK3Cache & cache,
                  int32_t token,
                  int position,
                  std::vector<float> & logits,
                  MoeHybridStreamEngine * stream_engine,
                  MoeStreamDualOwnerExecutor * dual_stream_executor,
                  const MoeStreamDualOwnerPolicy * stream_owner_policy,
                  MoeHybridRoutingStats * routing_stats) {
    KimiK3ForwardOptions options;
    options.read_logits = true;
    options.read_argmax = false;
    KimiK3ForwardResult result;
    if (!kimi_k3_forward(
            backend, w, cache, std::vector<int32_t>{token}, position,
            options, result, stream_engine, dual_stream_executor,
            stream_owner_policy, routing_stats)) {
        return false;
    }
    logits = std::move(result.logits);
    return true;
}

} // namespace dflash::common
