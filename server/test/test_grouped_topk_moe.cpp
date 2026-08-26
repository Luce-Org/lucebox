#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int kExperts = 512;
constexpr int kGroups = 8;
constexpr int kGroupsUsed = 4;
constexpr int kExpertsUsed = 8;
constexpr int kGroupScoreExperts = 2;

struct RouteOutputs {
    std::vector<int32_t> selected;
    std::vector<float> weights;
};

struct GraphRouteOutputs {
    ggml_tensor * selected = nullptr;
    ggml_tensor * weights = nullptr;
};

template <typename T>
void read_route_rows(
        ggml_tensor * tensor, std::vector<T> & values, int n_tokens) {
    const size_t row_bytes = kExpertsUsed * sizeof(T);
    for (int token = 0; token < n_tokens; ++token) {
        ggml_backend_tensor_get(
            tensor, values.data() + static_cast<size_t>(token) * kExpertsUsed,
            static_cast<size_t>(token) * tensor->nb[1], row_bytes);
    }
}

GraphRouteOutputs build_generic_route(
        ggml_context * ctx, ggml_tensor * logits, ggml_tensor * bias,
        int n_tokens) {
    ggml_tensor * probs = ggml_sigmoid(ctx, logits);
    ggml_tensor * selection_probs = ggml_add(ctx, probs, bias);
    constexpr int experts_per_group = kExperts / kGroups;
    ggml_tensor * selection_groups = ggml_reshape_3d(
        ctx, selection_probs, experts_per_group, kGroups, n_tokens);

    ggml_tensor * group_scores = ggml_argsort_top_k(
        ctx, selection_groups, kGroupScoreExperts);
    group_scores = ggml_get_rows(
        ctx,
        ggml_reshape_4d(ctx, selection_groups, 1, experts_per_group,
                        kGroups, n_tokens),
        group_scores);
    group_scores = ggml_sum_rows(
        ctx, ggml_reshape_3d(ctx, group_scores, kGroupScoreExperts,
                             kGroups, n_tokens));
    group_scores = ggml_reshape_2d(ctx, group_scores, kGroups, n_tokens);

    ggml_tensor * selected_groups = ggml_argsort_top_k(
        ctx, group_scores, kGroupsUsed);
    ggml_tensor * kept_groups = ggml_get_rows(
        ctx, selection_groups, selected_groups);
    selection_groups = ggml_set_rows(
        ctx, ggml_fill(ctx, selection_groups, -INFINITY),
        kept_groups, selected_groups);
    selection_probs = ggml_reshape_2d(
        ctx, selection_groups, kExperts, n_tokens);

    ggml_tensor * selected = ggml_argsort_top_k(
        ctx, selection_probs, kExpertsUsed);
    ggml_tensor * probs_3d = ggml_reshape_3d(
        ctx, probs, 1, kExperts, n_tokens);
    ggml_tensor * weights = ggml_get_rows(ctx, probs_3d, selected);
    weights = ggml_reshape_2d(ctx, weights, kExpertsUsed, n_tokens);
    ggml_tensor * weight_sum = ggml_sum_rows(ctx, weights);
    weight_sum = ggml_clamp(ctx, weight_sum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(ctx, weights, weight_sum);
    return {selected, weights};
}

GraphRouteOutputs build_fused_route(
        ggml_context * ctx, ggml_tensor * logits, ggml_tensor * bias,
        int n_tokens) {
    ggml_tensor * weights = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, kExpertsUsed, n_tokens);
    ggml_tensor * selected = ggml_grouped_top_k_moe(
        ctx, logits, bias, weights, kGroups, kGroupsUsed,
        kGroupScoreExperts, 1.0f);
    return {selected, weights};
}

bool run_case(ggml_backend_t backend, int n_tokens) {
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * logits = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, kExperts, n_tokens);
    ggml_tensor * bias = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F32, kExperts);
    ggml_set_input(logits);
    ggml_set_input(bias);

    const GraphRouteOutputs generic =
        build_generic_route(ctx, logits, bias, n_tokens);
    const GraphRouteOutputs fused =
        build_fused_route(ctx, logits, bias, n_tokens);
    for (ggml_tensor * output : {
             generic.selected, generic.weights,
             fused.selected, fused.weights}) {
        ggml_set_output(output);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2048, false);
    ggml_build_forward_expand(graph, generic.selected);
    ggml_build_forward_expand(graph, generic.weights);
    ggml_build_forward_expand(graph, fused.selected);
    // The fused selected operation writes this leaf tensor as a side effect.
    ggml_build_forward_expand(graph, fused.weights);

    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    bool ok = allocator && ggml_gallocr_alloc_graph(allocator, graph);

    std::vector<float> logits_data(
        static_cast<size_t>(kExperts) * n_tokens);
    std::vector<float> bias_data(kExperts);
    for (int expert = 0; expert < kExperts; ++expert) {
        // Non-monotonic, non-tied selection bias exercises every group.
        bias_data[static_cast<size_t>(expert)] =
            0.075f * std::sin(0.173f * static_cast<float>(expert)) +
            0.005f * static_cast<float>((expert % 11) - 5);
    }
    for (int token = 0; token < n_tokens; ++token) {
        for (int expert = 0; expert < kExperts; ++expert) {
            const float x = static_cast<float>(
                expert + 37 * token + 1);
            logits_data[static_cast<size_t>(token) * kExperts + expert] =
                3.25f * std::sin(0.037f * x) +
                1.75f * std::cos(0.113f * x) +
                0.002f * static_cast<float>((expert * 17 + token) % 23);
        }
    }

    if (ok) {
        ggml_backend_tensor_set(logits, logits_data.data(), 0,
                                logits_data.size() * sizeof(float));
        ggml_backend_tensor_set(bias, bias_data.data(), 0,
                                bias_data.size() * sizeof(float));
        ok = ggml_backend_graph_compute(backend, graph) ==
             GGML_STATUS_SUCCESS;
    }

    const size_t count = static_cast<size_t>(kExpertsUsed) * n_tokens;
    RouteOutputs generic_host{
        std::vector<int32_t>(count), std::vector<float>(count)};
    RouteOutputs fused_host{
        std::vector<int32_t>(count), std::vector<float>(count)};
    if (ok) {
        read_route_rows(generic.selected, generic_host.selected, n_tokens);
        read_route_rows(generic.weights, generic_host.weights, n_tokens);
        read_route_rows(fused.selected, fused_host.selected, n_tokens);
        read_route_rows(fused.weights, fused_host.weights, n_tokens);
    }

    int id_mismatches = 0;
    float max_abs_weight_error = 0.0f;
    for (size_t i = 0; ok && i < count; ++i) {
        if (generic_host.selected[i] != fused_host.selected[i]) {
            if (id_mismatches < 8) {
                std::fprintf(stderr,
                    "row=%zu rank=%zu generic_id=%d fused_id=%d\n",
                    i / kExpertsUsed, i % kExpertsUsed,
                    generic_host.selected[i], fused_host.selected[i]);
            }
            ++id_mismatches;
        }
        max_abs_weight_error = std::max(
            max_abs_weight_error,
            std::fabs(generic_host.weights[i] - fused_host.weights[i]));
    }
    std::printf(
        "grouped-topk-moe tokens=%d id_mismatches=%d max_weight_error=%.9g\n",
        n_tokens, id_mismatches, max_abs_weight_error);
    ok = ok && id_mismatches == 0 && max_abs_weight_error <= 2.0e-7f;

    if (allocator) ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return ok;
}

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "CUDA backend unavailable\n");
        return 77;
    }
    const bool ok = run_case(backend, 1) && run_case(backend, 9);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
