// Structural contract for Qwen3.5's fused Gated DeltaNet graph.
//
// Qwen3.5 projects 16 Q/K groups and 48 V heads. The fused GDN kernels use
// modulo head indexing for Q/K, so the graph builder must preserve the compact
// 16-head tensors instead of materializing two 16 -> 48 repeats.

#include "internal.h"

#include <cstdlib>
#include <cstdio>

namespace {

using dflash::common::TargetCache;
using dflash::common::TargetLayer;
using dflash::common::TargetWeights;
using dflash::common::build_qwen35_layer;

ggml_tensor * tensor_1d(ggml_context * ctx, int64_t ne0) {
    return ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
}

ggml_tensor * tensor_2d(ggml_context * ctx, int64_t ne0, int64_t ne1) {
    return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
}

bool is_qk_head_repeat(const ggml_tensor * node) {
    return node->op == GGML_OP_REPEAT &&
           node->src[0] != nullptr &&
           node->src[0]->ne[1] == 16 &&
           node->ne[1] == 48;
}

} // namespace

int main() {
    constexpr int64_t kHidden = 192;
    constexpr int64_t kIntermediate = 64;
    constexpr int64_t kTokens = 3;
    constexpr int64_t kState = 4;
    constexpr int64_t kGroups = 16;
    constexpr int64_t kHeads = 48;
    constexpr int64_t kInner = kState * kHeads;
    constexpr int64_t kConv = 4;
    constexpr int64_t kConvChannels = kInner + 2 * kGroups * kState;

    ggml_init_params params = {4 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "FAIL: ggml_init returned null\n");
        return 1;
    }

    TargetWeights weights;
    weights.n_layer = 1;
    weights.n_embd = kHidden;
    weights.n_ff = kIntermediate;
    weights.full_attention_interval = 4;
    weights.ssm_d_conv = kConv;
    weights.ssm_d_inner = kInner;
    weights.ssm_d_state = kState;
    weights.ssm_dt_rank = kHeads;
    weights.ssm_n_group = kGroups;
    weights.n_capture_layers = 0;
    weights.layers.resize(1);

    TargetLayer & layer = weights.layers[0];
    layer.attn_norm = tensor_1d(ctx, kHidden);
    layer.attn_post_norm = tensor_1d(ctx, kHidden);
    layer.wqkv = tensor_2d(ctx, kHidden, kConvChannels);
    layer.wqkv_gate = tensor_2d(ctx, kHidden, kInner);
    layer.ssm_conv1d = tensor_2d(ctx, kConv, kConvChannels);
    layer.ssm_beta = tensor_2d(ctx, kHidden, kHeads);
    layer.ssm_alpha = tensor_2d(ctx, kHidden, kHeads);
    layer.ssm_a = tensor_1d(ctx, kHeads);
    layer.ssm_dt_bias = tensor_1d(ctx, kHeads);
    layer.ssm_norm = tensor_1d(ctx, kState);
    layer.ssm_out = tensor_2d(ctx, kInner, kHidden);
    layer.w_gate = tensor_2d(ctx, kHidden, kIntermediate);
    layer.w_up = tensor_2d(ctx, kHidden, kIntermediate);
    layer.w_down = tensor_2d(ctx, kIntermediate, kHidden);

    TargetCache cache;
    cache.conv_state.push_back(tensor_2d(ctx, kConv - 1, kConvChannels));
    cache.ssm_state.push_back(ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, kState, kState, kHeads));

    ggml_tensor * input = tensor_2d(ctx, kHidden, kTokens);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_tensor * output = build_qwen35_layer(
        ctx, graph, weights, cache,
        /*layer_idx=*/0,
        input,
        /*positions=*/nullptr,
        /*attn_mask=*/nullptr,
        /*kv_start=*/0,
        kTokens,
        /*capture=*/false);
    ggml_build_forward_expand(graph, output);

    const ggml_tensor * gdn = nullptr;
    int gdn_count = 0;
    int repeat_count = 0;
    const int node_count = ggml_graph_n_nodes(graph);
    for (int i = 0; i < node_count; ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_GATED_DELTA_NET) {
            gdn = node;
            ++gdn_count;
        }
        if (is_qk_head_repeat(node)) {
            ++repeat_count;
        }
    }

    bool pass = true;
    if (gdn_count != 1 || gdn == nullptr) {
        std::fprintf(stderr, "FAIL: expected one fused GDN node, found %d\n", gdn_count);
        pass = false;
    } else {
        if (gdn->src[0]->ne[1] != kGroups || gdn->src[1]->ne[1] != kGroups) {
            std::fprintf(stderr,
                "FAIL: fused GDN received materialized Q/K heads (%lld/%lld, expected %lld/%lld)\n",
                static_cast<long long>(gdn->src[0]->ne[1]),
                static_cast<long long>(gdn->src[1]->ne[1]),
                static_cast<long long>(kGroups),
                static_cast<long long>(kGroups));
            pass = false;
        }
        if (gdn->src[2]->ne[1] != kHeads) {
            std::fprintf(stderr,
                "FAIL: fused GDN received %lld V heads, expected %lld\n",
                static_cast<long long>(gdn->src[2]->ne[1]),
                static_cast<long long>(kHeads));
            pass = false;
        }
    }
    if (repeat_count != 0) {
        std::fprintf(stderr,
            "FAIL: graph materialized %d Q/K 16 -> 48 head repeat(s)\n",
            repeat_count);
        pass = false;
    }

    // The opt-in chunked graph does not implement Q/K head broadcasting yet;
    // it must retain the two materialized repeats that the fused path removes.
    setenv("DFLASH27B_CHUNKED", "1", 1);
    ggml_tensor * chunked_input = tensor_2d(ctx, kHidden, kTokens);
    ggml_cgraph * chunked_graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_tensor * chunked_output = build_qwen35_layer(
        ctx, chunked_graph, weights, cache,
        /*layer_idx=*/0,
        chunked_input,
        /*positions=*/nullptr,
        /*attn_mask=*/nullptr,
        /*kv_start=*/0,
        kTokens,
        /*capture=*/false);
    ggml_build_forward_expand(chunked_graph, chunked_output);
    unsetenv("DFLASH27B_CHUNKED");

    int chunked_repeat_count = 0;
    int chunked_gdn_count = 0;
    for (int i = 0; i < ggml_graph_n_nodes(chunked_graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(chunked_graph, i);
        if (is_qk_head_repeat(node)) {
            ++chunked_repeat_count;
        }
        if (node->op == GGML_OP_GATED_DELTA_NET) {
            ++chunked_gdn_count;
        }
    }
    if (chunked_repeat_count != 2 || chunked_gdn_count != 0) {
        std::fprintf(stderr,
            "FAIL: chunked GDN graph has %d Q/K repeat(s) and %d fused node(s), expected 2/0\n",
            chunked_repeat_count, chunked_gdn_count);
        pass = false;
    }

    ggml_free(ctx);
    if (!pass) {
        return 1;
    }

    std::printf("PASS: fused GDN consumes compact Q/K heads (16/16) with 48 V heads\n");
    return 0;
}
