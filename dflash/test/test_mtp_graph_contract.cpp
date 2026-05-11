#include "internal.h"

#include "ggml.h"

#include <cstdio>

using namespace dflash27b;

static ggml_tensor * tensor_1d(ggml_context * ctx, int n0, const char * name) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n0);
    ggml_set_name(t, name);
    return t;
}

static ggml_tensor * tensor_2d(ggml_context * ctx, int n0, int n1, const char * name) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1);
    ggml_set_name(t, name);
    return t;
}

int main() {
    ggml_init_params ip{};
    ip.mem_size   = 16 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;

    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        return 1;
    }

    TargetWeights w{};
    w.n_embd = 8;
    w.n_ff = 16;
    w.n_head = 2;
    w.n_head_kv = 1;
    w.n_embd_head_k = 4;
    w.n_embd_head_v = 4;
    w.rope_sections[0] = 1;
    w.rope_sections[1] = 1;
    w.rope_sections[2] = 0;
    w.rope_sections[3] = 0;
    w.out_norm = tensor_1d(ctx, w.n_embd, "output_norm.weight");
    w.output   = tensor_2d(ctx, w.n_embd, 32, "output.weight");
    w.mtp_layers.assign(1, TargetMtpLayer{});

    TargetMtpLayer & M = w.mtp_layers[0];
    TargetLayer & L = M.block;
    M.gguf_layer_index = 64;
    M.nextn.eh_proj          = tensor_2d(ctx, 2 * w.n_embd, w.n_embd, "blk.64.nextn.eh_proj.weight");
    M.nextn.enorm            = tensor_1d(ctx, w.n_embd, "blk.64.nextn.enorm.weight");
    M.nextn.hnorm            = tensor_1d(ctx, w.n_embd, "blk.64.nextn.hnorm.weight");
    M.nextn.shared_head_norm = tensor_1d(ctx, w.n_embd, "blk.64.nextn.shared_head_norm.weight");

    const int q_dim = w.n_head * w.n_embd_head_k;
    const int kv_dim = w.n_head_kv * w.n_embd_head_k;
    L.attn_norm      = tensor_1d(ctx, w.n_embd, "blk.64.attn_norm.weight");
    L.attn_post_norm = tensor_1d(ctx, w.n_embd, "blk.64.post_attention_norm.weight");
    L.wq             = tensor_2d(ctx, w.n_embd, 2 * q_dim, "blk.64.attn_q.weight");
    L.wk             = tensor_2d(ctx, w.n_embd, kv_dim, "blk.64.attn_k.weight");
    L.wv             = tensor_2d(ctx, w.n_embd, kv_dim, "blk.64.attn_v.weight");
    L.wo             = tensor_2d(ctx, q_dim, w.n_embd, "blk.64.attn_output.weight");
    L.q_norm         = tensor_1d(ctx, w.n_embd_head_k, "blk.64.attn_q_norm.weight");
    L.k_norm         = tensor_1d(ctx, w.n_embd_head_k, "blk.64.attn_k_norm.weight");
    L.w_gate         = tensor_2d(ctx, w.n_embd, w.n_ff, "blk.64.ffn_gate.weight");
    L.w_up           = tensor_2d(ctx, w.n_embd, w.n_ff, "blk.64.ffn_up.weight");
    L.w_down         = tensor_2d(ctx, w.n_ff, w.n_embd, "blk.64.ffn_down.weight");

    TargetMtpCache cache{};
    cache.max_ctx = 8;
    cache.kv_k_type = GGML_TYPE_F16;
    cache.kv_v_type = GGML_TYPE_F16;
    cache.attn_k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, w.n_embd_head_k, cache.max_ctx, w.n_head_kv));
    cache.attn_v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, w.n_embd_head_k, cache.max_ctx, w.n_head_kv));

    const int n_tokens = 1;
    ggml_tensor * token_embed = tensor_2d(ctx, w.n_embd, n_tokens, "mtp_token_embed");
    ggml_tensor * hidden      = tensor_2d(ctx, w.n_embd, n_tokens, "target_pre_norm_hidden");
    ggml_tensor * positions   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(positions, "positions");
    ggml_set_input(token_embed);
    ggml_set_input(hidden);
    ggml_set_input(positions);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);
    QwenMtpGraphInputs in{};
    in.token_embed = token_embed;
    in.pre_norm_hidden = hidden;
    in.positions = positions;
    in.n_tokens = n_tokens;
    in.kv_start = 0;

    QwenMtpGraphOutputs out = build_qwen35_mtp_graph(ctx, gf, w, cache, in);
    if (!out.logits || !out.hidden) {
        std::fprintf(stderr, "build_qwen35_mtp_graph failed: %s\n", dflash27b_last_error());
        ggml_free(ctx);
        return 1;
    }

    std::printf("[mtp-graph-contract] nodes=%d logits=[%lld,%lld] hidden=[%lld,%lld]\n",
        ggml_graph_n_nodes(gf),
        (long long)out.logits->ne[0], (long long)out.logits->ne[1],
        (long long)out.hidden->ne[0], (long long)out.hidden->ne[1]);

    ggml_free(ctx);
    return 0;
}
