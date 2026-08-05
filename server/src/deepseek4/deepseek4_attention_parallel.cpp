#include "deepseek4_attention_parallel.h"

#include "deepseek4_internal.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace dflash::common {

namespace {

void set_error(std::string * error, const std::string & message) {
    if (error) *error = message;
}

bool tensor_rows_match(const ggml_tensor * source,
                       const ggml_tensor * destination) {
    return source && destination && source->type == destination->type &&
           source->ne[0] == destination->ne[0] &&
           source->nb[1] == destination->nb[1];
}

bool copy_tensor_rows(const ggml_tensor * source,
                      ggml_tensor * destination,
                      int source_row,
                      int rows,
                      std::vector<uint8_t> & staging,
                      std::string * error) {
    if (!tensor_rows_match(source, destination) || source_row < 0 || rows < 0 ||
        source_row + rows > source->ne[1] || rows > destination->ne[1]) {
        set_error(error, "attention peer tensor row layout mismatch");
        return false;
    }
    if (rows == 0) return true;

    const size_t row_bytes = source->nb[1];
    const size_t bytes = row_bytes * (size_t) rows;
    staging.resize(bytes);
    ggml_backend_tensor_get(source, staging.data(),
                            row_bytes * (size_t) source_row, bytes);
    ggml_backend_tensor_set(destination, staging.data(), 0, bytes);
    return true;
}

}  // namespace

DeepSeek4AttentionParallelPlan plan_deepseek4_attention_groups(
        int n_head,
        int n_out_group,
        int requested_peer_groups) {
    DeepSeek4AttentionParallelPlan plan;
    if (n_head <= 0 || n_out_group <= 1 || n_head % n_out_group != 0 ||
        requested_peer_groups <= 0 || requested_peer_groups >= n_out_group) {
        return plan;
    }
    plan.total_groups = n_out_group;
    plan.peer_groups = requested_peer_groups;
    plan.main_groups = n_out_group - requested_peer_groups;
    plan.heads_per_group = n_head / n_out_group;
    return plan;
}

DeepSeek4AttentionParallelState::~DeepSeek4AttentionParallelState() {
    free_deepseek4_attention_parallel(*this);
}

void free_deepseek4_attention_parallel(DeepSeek4AttentionParallelState & state) {
    state.layers.clear();
    if (state.weights_buf) {
        ggml_backend_buffer_free(state.weights_buf);
        state.weights_buf = nullptr;
    }
    if (state.weights_ctx) {
        ggml_free(state.weights_ctx);
        state.weights_ctx = nullptr;
    }
    state.main_backend = nullptr;
    state.peer_backend = nullptr;
    state.plan = {};
    state.min_context = 0;
}

bool init_deepseek4_attention_parallel(
        ggml_backend_t main_backend,
        ggml_backend_t peer_backend,
        const DeepSeek4Weights & weights,
        int peer_groups,
        int min_context,
        DeepSeek4AttentionParallelState & out,
        std::string * error) {
    free_deepseek4_attention_parallel(out);

    const DeepSeek4AttentionParallelPlan plan =
        plan_deepseek4_attention_groups(
            weights.n_head, weights.n_out_group, peer_groups);
    if (!main_backend || !peer_backend || main_backend == peer_backend ||
        !plan.valid()) {
        set_error(error, "attention group split requires two distinct backends and a legal group count");
        return false;
    }
    out.main_backend = main_backend;
    out.peer_backend = peer_backend;
    out.plan = plan;
    out.min_context = std::max(0, min_context);
    out.layers.resize((size_t) weights.n_layer);

    ggml_init_params weights_params{};
    weights_params.mem_size =
        ggml_tensor_overhead() * (size_t) (weights.n_layer + 8) + 4096;
    weights_params.no_alloc = true;
    out.weights_ctx = ggml_init(weights_params);
    if (!out.weights_ctx) {
        set_error(error, "failed to create attention peer weight context");
        free_deepseek4_attention_parallel(out);
        return false;
    }

    const int peer_output_rows = plan.peer_groups * weights.n_lora_o;
    for (int il = 0; il < weights.n_layer; ++il) {
        const DeepSeek4Layer & source_layer = weights.layers[(size_t) il];
        DeepSeek4AttentionParallelLayer & destination =
            out.layers[(size_t) il];
        if (!source_layer.attn_output_a) {
            set_error(error, "attention group split source weight is missing");
            free_deepseek4_attention_parallel(out);
            return false;
        }

        destination.output_a = ggml_new_tensor_2d(
            out.weights_ctx,
            source_layer.attn_output_a->type,
            source_layer.attn_output_a->ne[0], peer_output_rows);

        char name[96];
        std::snprintf(name, sizeof(name), "blk.%d.attn_output_a.peer", il);
        ggml_set_name(destination.output_a, name);
    }

    out.weights_buf =
        ggml_backend_alloc_ctx_tensors(out.weights_ctx, peer_backend);
    if (!out.weights_buf) {
        set_error(error, "failed to allocate attention peer weights");
        free_deepseek4_attention_parallel(out);
        return false;
    }
    ggml_backend_buffer_set_usage(
        out.weights_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_backend_synchronize(main_backend);
    std::vector<uint8_t> staging;
    const int first_peer_row = plan.main_groups * weights.n_lora_o;
    size_t copied_bytes = 0;
    for (int il = 0; il < weights.n_layer; ++il) {
        const ggml_tensor * source =
            weights.layers[(size_t) il].attn_output_a;
        ggml_tensor * destination = out.layers[(size_t) il].output_a;
        if (!copy_tensor_rows(
                source, destination, first_peer_row, peer_output_rows,
                staging, error)) {
            free_deepseek4_attention_parallel(out);
            return false;
        }
        copied_bytes += ggml_nbytes(destination);
    }
    ggml_backend_synchronize(peer_backend);

    std::fprintf(stderr,
        "[deepseek4-attn-tp] initialized single-process split "
        "main=%d groups peer=%d groups peer_weights=%.1f MiB "
        "min_context=%d\n",
        plan.main_groups, plan.peer_groups,
        (double) copied_bytes / (1024.0 * 1024.0),
        out.min_context);
    return true;
}

}  // namespace dflash::common
