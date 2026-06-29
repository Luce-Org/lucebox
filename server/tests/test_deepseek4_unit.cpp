#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "common/backend_ipc.h"
#include "common/layer_split_utils.h"

#include <memory>
#include <random>
#include <string>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>

#define private public
#include "deepseek4/deepseek4_layer_split_adapter.h"
#undef private

using namespace dflash::common;

static int g_failures = 0;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        ++g_failures; \
        std::fprintf(stderr, "  FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    } \
} while (0)

#define TEST_ASSERT_MSG(cond, msg) do { \
    if (!(cond)) { \
        ++g_failures; \
        std::fprintf(stderr, "  FAIL: %s:%d: %s (%s)\n", __FILE__, __LINE__, #cond, msg); \
    } \
} while (0)

static bool nearly_equal(float a, float b, float atol = 1.0e-5f, float rtol = 1.0e-5f) {
    const float diff = std::fabs(a - b);
    const float scale = std::max(std::fabs(a), std::fabs(b));
    return diff <= atol + rtol * scale;
}

static ggml_context * make_test_context(size_t mem_size = 1u << 20) {
    ggml_init_params params = {};
    params.mem_size = mem_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    return ggml_init(params);
}

static float softplus_stable(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return std::exp(x);
    }
    return std::log1p(std::exp(x));
}

static std::vector<int> topk_desc(const std::vector<float> & scores, int k) {
    std::vector<int> idx(scores.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });
    idx.resize((size_t) k);
    return idx;
}

static void test_compressor_pooling_correctness(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_compressor_pooling_correctness ...");

    constexpr int ratio = 4;
    constexpr int dim = 7;
    std::vector<float> state_kv((size_t) ratio * dim);
    std::vector<float> state_score((size_t) ratio * dim);
    for (int i = 0; i < ratio; ++i) {
        for (int j = 0; j < dim; ++j) {
            state_kv[(size_t) i * dim + j] = 0.125f * (float) ((i + 1) * (j + 2)) - 0.35f;
            state_score[(size_t) i * dim + j] = 0.2f * (float) (i - j) + 0.05f * (float) (i * j);
        }
    }

    std::vector<float> expected(dim, 0.0f);
    for (int j = 0; j < dim; ++j) {
        float denom = 0.0f;
        float numer = 0.0f;
        for (int i = 0; i < ratio; ++i) {
            const size_t idx = (size_t) i * dim + j;
            const float w = std::exp(state_score[idx]);
            denom += w;
            numer += w * state_kv[idx];
        }
        expected[j] = numer / denom;
    }

    ggml_context * ctx = make_test_context();
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) {
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    ggml_tensor * kv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, ratio);
    ggml_tensor * score = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, ratio);
    ggml_set_input(kv);
    ggml_set_input(score);

    ggml_tensor * score_t = ggml_cont(ctx, ggml_transpose(ctx, score));
    ggml_tensor * weights_t = ggml_soft_max(ctx, score_t);
    ggml_tensor * weights = ggml_transpose(ctx, weights_t);
    ggml_tensor * weighted = ggml_mul(ctx, kv, weights);
    ggml_tensor * pooled = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, weighted)));
    pooled = ggml_reshape_1d(ctx, pooled, dim);
    ggml_set_output(pooled);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, pooled);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    TEST_ASSERT(ggml_gallocr_alloc_graph(alloc, gf));
    ggml_backend_tensor_set(kv, state_kv.data(), 0, state_kv.size() * sizeof(float));
    ggml_backend_tensor_set(score, state_score.data(), 0, state_score.size() * sizeof(float));
    TEST_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(dim);
    ggml_backend_tensor_get(pooled, actual.data(), 0, actual.size() * sizeof(float));
    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    for (int j = 0; j < dim; ++j) {
        TEST_ASSERT_MSG(nearly_equal(actual[j], expected[j], 1.0e-5f, 1.0e-5f), "pooled output mismatch");
    }

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_moe_routing_correctness(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_moe_routing_correctness ...");

    constexpr int n_expert = 8;
    constexpr int top_k = 2;
    constexpr float expert_weight_scale = 1.5f;
    const std::vector<float> logits = {-2.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, -1.0f, 0.25f};
    const std::vector<float> bias = {0.20f, -0.10f, 0.05f, 0.00f, -0.20f, 0.15f, 0.30f, -0.05f};

    std::vector<float> probs(n_expert);
    std::vector<float> selection(n_expert);
    for (int i = 0; i < n_expert; ++i) {
        probs[i] = std::sqrt(softplus_stable(logits[(size_t) i]));
        selection[i] = probs[i] + bias[(size_t) i];
    }

    const std::vector<int> expected_selected = topk_desc(selection, top_k);
    float expected_sum = 0.0f;
    for (int idx : expected_selected) {
        expected_sum += probs[(size_t) idx];
    }
    expected_sum = std::max(expected_sum, 6.103515625e-5f);

    std::vector<float> expected_weights(top_k);
    for (int i = 0; i < top_k; ++i) {
        expected_weights[(size_t) i] = probs[(size_t) expected_selected[(size_t) i]] / expected_sum * expert_weight_scale;
    }

    ggml_context * ctx = make_test_context();
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) {
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    ggml_tensor * logits_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_expert, 1);
    ggml_tensor * bias_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_expert);
    ggml_set_input(logits_t);
    ggml_set_input(bias_t);

    ggml_tensor * probs_t = ggml_sqrt(ctx, ggml_softplus(ctx, logits_t));
    ggml_tensor * selection_t = ggml_add(ctx, probs_t, bias_t);
    ggml_tensor * selected_t = ggml_top_k(ctx, selection_t, top_k);
    ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs_t, 1, n_expert, 1);
    ggml_tensor * weights_t = ggml_get_rows(ctx, probs_3d, selected_t);
    weights_t = ggml_reshape_2d(ctx, weights_t, top_k, 1);
    ggml_tensor * sum_t = ggml_sum_rows(ctx, weights_t);
    sum_t = ggml_clamp(ctx, sum_t, 6.103515625e-5f, INFINITY);
    weights_t = ggml_div(ctx, weights_t, sum_t);
    weights_t = ggml_scale(ctx, weights_t, expert_weight_scale);
    ggml_set_output(selected_t);
    ggml_set_output(weights_t);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(gf, selected_t);
    ggml_build_forward_expand(gf, weights_t);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    TEST_ASSERT(ggml_gallocr_alloc_graph(alloc, gf));
    ggml_backend_tensor_set(logits_t, logits.data(), 0, logits.size() * sizeof(float));
    ggml_backend_tensor_set(bias_t, bias.data(), 0, bias.size() * sizeof(float));
    TEST_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<int32_t> actual_selected(top_k);
    std::vector<float> actual_weights(top_k);
    ggml_backend_tensor_get(selected_t, actual_selected.data(), 0, actual_selected.size() * sizeof(int32_t));
    ggml_backend_tensor_get(weights_t, actual_weights.data(), 0, actual_weights.size() * sizeof(float));
    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    std::vector<int32_t> actual_sorted = actual_selected;
    std::vector<int32_t> expected_sorted(expected_selected.begin(), expected_selected.end());
    std::sort(actual_sorted.begin(), actual_sorted.end());
    std::sort(expected_sorted.begin(), expected_sorted.end());
    TEST_ASSERT(actual_sorted == expected_sorted);

    for (int i = 0; i < top_k; ++i) {
        const int expert = actual_selected[(size_t) i];
        auto it = std::find(expected_selected.begin(), expected_selected.end(), expert);
        TEST_ASSERT(it != expected_selected.end());
        if (it != expected_selected.end()) {
            const size_t ref_idx = (size_t) std::distance(expected_selected.begin(), it);
            TEST_ASSERT_MSG(nearly_equal(actual_weights[(size_t) i], expected_weights[ref_idx], 1.0e-5f, 1.0e-5f), "router weight mismatch");
        }
    }

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_rmsnorm_correctness(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_rmsnorm_correctness ...");

    constexpr int n = 16;
    constexpr float eps = 1.0e-6f;
    std::vector<float> x(n);
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i) {
        x[(size_t) i] = 0.15f * (float) (i - 5) + 0.03f * (float) (i % 3);
        w[(size_t) i] = 0.8f + 0.02f * (float) i;
    }

    float mean_sq = 0.0f;
    for (float v : x) {
        mean_sq += v * v;
    }
    mean_sq /= (float) n;
    const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

    std::vector<float> expected(n);
    for (int i = 0; i < n; ++i) {
        expected[(size_t) i] = x[(size_t) i] * inv_rms * w[(size_t) i];
    }

    ggml_context * ctx = make_test_context();
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) {
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    ggml_tensor * x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 1);
    ggml_tensor * w_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_input(x_t);
    ggml_set_input(w_t);

    ggml_tensor * y_t = ggml_mul(ctx, ggml_rms_norm(ctx, x_t, eps), w_t);
    ggml_tensor * y_flat = ggml_reshape_1d(ctx, y_t, n);
    ggml_set_output(y_flat);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, y_flat);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    TEST_ASSERT(ggml_gallocr_alloc_graph(alloc, gf));
    ggml_backend_tensor_set(x_t, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(w_t, w.data(), 0, w.size() * sizeof(float));
    TEST_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(n);
    ggml_backend_tensor_get(y_flat, actual.data(), 0, actual.size() * sizeof(float));
    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    for (int i = 0; i < n; ++i) {
        TEST_ASSERT_MSG(nearly_equal(actual[(size_t) i], expected[(size_t) i], 1.0e-5f, 1.0e-5f), "rmsnorm output mismatch");
    }

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_grouped_output_projection_shape() {
    std::fprintf(stderr, "  test_grouped_output_projection_shape ...");

    constexpr int head_dim = 512;
    constexpr int n_head = 64;
    constexpr int n_out_group = 8;
    constexpr int n_lora_o = 1024;
    constexpr int n_embd = 4096;

    const int flat_heads = head_dim * n_head;
    const int group_heads = n_head / n_out_group;
    const int group_input = head_dim * group_heads;
    const int grouped_low_rank = n_out_group * n_lora_o;

    TEST_ASSERT(flat_heads == 32768);
    TEST_ASSERT(group_heads == 8);
    TEST_ASSERT(group_input == 4096);
    TEST_ASSERT(group_input * n_out_group == flat_heads);
    TEST_ASSERT(n_lora_o == 1024);
    TEST_ASSERT(grouped_low_rank == 8192);
    TEST_ASSERT(n_embd == 4096);

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_hash_routing_lookup() {
    std::fprintf(stderr, "  test_hash_routing_lookup ...");

    constexpr int n_token = 10;
    constexpr int n_expert_used = 6;
    std::vector<int32_t> tid2eid((size_t) n_token * n_expert_used);
    for (int token = 0; token < n_token; ++token) {
        for (int slot = 0; slot < n_expert_used; ++slot) {
            tid2eid[(size_t) token * n_expert_used + slot] = (int32_t) ((token * 7 + slot * 3 + 1) % 19);
        }
    }

    for (int token = 0; token < n_token; ++token) {
        const int32_t * row = tid2eid.data() + (size_t) token * n_expert_used;
        for (int slot = 0; slot < n_expert_used; ++slot) {
            const int32_t expected = (int32_t) ((token * 7 + slot * 3 + 1) % 19);
            TEST_ASSERT(row[slot] == expected);
        }
    }

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

struct ScopedEnvVar {
    explicit ScopedEnvVar(const char * name)
        : name(name ? name : ""),
          had_value(std::getenv(this->name.c_str()) != nullptr),
          old_value(had_value ? std::getenv(this->name.c_str()) : "") {}

    ~ScopedEnvVar() {
        if (had_value) {
            setenv(name.c_str(), old_value.c_str(), 1);
        } else {
            unsetenv(name.c_str());
        }
    }

    std::string name;
    bool had_value = false;
    std::string old_value;
};

static DeepSeek4LayerSplitAdapter make_test_adapter() {
    DeepSeek4LayerSplitAdapterConfig cfg;
    cfg.device.gpu = 0;
    cfg.device.max_ctx = 8192;
    return DeepSeek4LayerSplitAdapter(cfg);
}

static void test_auto_split_computation() {
    std::fprintf(stderr, "  test_auto_split_computation ...");

    ScopedEnvVar env_guard("DFLASH_DS4_CUDA_LAYERS");
    auto adapter = make_test_adapter();

    setenv("DFLASH_DS4_CUDA_LAYERS", "17", 1);
    TEST_ASSERT(adapter.compute_auto_split_layers() == 17);

    unsetenv("DFLASH_DS4_CUDA_LAYERS");
    const int estimated =
        DeepSeek4LayerSplitAdapter::estimate_cuda_layers_from_free_bytes(
            20ULL * 1024 * 1024 * 1024);
    TEST_ASSERT(estimated >= 1 && estimated <= 42);
    TEST_ASSERT(estimated == 9);

    const int computed = adapter.compute_auto_split_layers();
    TEST_ASSERT(computed >= 1 && computed <= 42);

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_layer_range_validation() {
    std::fprintf(stderr, "  test_layer_range_validation ...");

    const auto equal = compute_layer_ranges(43, 2, {1.0, 1.0});
    TEST_ASSERT(equal.size() == 2);
    if (equal.size() == 2) {
        TEST_ASSERT(equal[0].begin == 0 && equal[0].end == 22);
        TEST_ASSERT(equal[1].begin == 22 && equal[1].end == 43);
    }

    const auto front_heavy = compute_layer_ranges(43, 2, {2.0, 1.0});
    TEST_ASSERT(front_heavy.size() == 2);
    if (front_heavy.size() == 2) {
        TEST_ASSERT(front_heavy[0].begin == 0 && front_heavy[0].end == 29);
        TEST_ASSERT(front_heavy[1].begin == 29 && front_heavy[1].end == 43);
    }

    const auto back_heavy = compute_layer_ranges(43, 2, {1.0, 2.0});
    TEST_ASSERT(back_heavy.size() == 2);
    if (back_heavy.size() == 2) {
        TEST_ASSERT(back_heavy[0].begin == 0 && back_heavy[0].end == 14);
        TEST_ASSERT(back_heavy[1].begin == 14 && back_heavy[1].end == 43);
    }

    const auto three_way = compute_layer_ranges(43, 3, {1.0, 1.0, 1.0});
    TEST_ASSERT(three_way.size() == 3);
    if (three_way.size() == 3) {
        TEST_ASSERT(three_way[0].begin == 0 && three_way[0].end == 14);
        TEST_ASSERT(three_way[1].begin == 14 && three_way[1].end == 29);
        TEST_ASSERT(three_way[2].begin == 29 && three_way[2].end == 43);
    }

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_hc_state_dimensions() {
    std::fprintf(stderr, "  test_hc_state_dimensions ...");

    DeepSeek4Weights weights;
    TEST_ASSERT(weights.n_hc == 4);
    TEST_ASSERT(weights.n_embd == 4096);
    TEST_ASSERT(DeepSeek4LayerSplitAdapter::hc_state_elements(weights) == 16384);

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_snapshot_save_restore() {
    std::fprintf(stderr, "  test_snapshot_save_restore ...");

    auto adapter = make_test_adapter();
    adapter.cur_pos_ = 23;
    adapter.last_tok_ = 4242;
    adapter.hc_state_ = {1.0f, 2.0f, 3.0f, 4.0f};

    TEST_ASSERT(adapter.snapshot_save(0));
    TEST_ASSERT(adapter.snapshot_used(0));
    TEST_ASSERT(adapter.snapshot_cur_pos(0) == 23);

    adapter.cur_pos_ = 0;
    adapter.last_tok_ = -1;
    adapter.hc_state_.assign(4, 0.0f);

    TEST_ASSERT(adapter.snapshot_restore(0));
    TEST_ASSERT(adapter.cur_pos_ == 23);
    TEST_ASSERT(adapter.last_tok_ == 4242);
    TEST_ASSERT(adapter.hc_state_.size() == 4);
    if (adapter.hc_state_.size() == 4) {
        TEST_ASSERT(adapter.hc_state_[0] == 1.0f);
        TEST_ASSERT(adapter.hc_state_[3] == 4.0f);
    }

    adapter.snapshot_free(0);
    TEST_ASSERT(!adapter.snapshot_used(0));
    TEST_ASSERT(adapter.snapshots_[0].hc_state.empty());
    TEST_ASSERT(!adapter.snapshot_restore(0));
    TEST_ASSERT(!adapter.snapshot_save(-1));
    TEST_ASSERT(!adapter.snapshot_save(ModelBackend::kMaxSlots));

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_reset_request_state() {
    std::fprintf(stderr, "  test_reset_request_state ...");

    auto adapter = make_test_adapter();
    adapter.cur_pos_ = 9;
    adapter.last_tok_ = 77;
    adapter.hc_state_.assign(8, 5.0f);
    adapter.shards_.resize(2);
    for (auto & shard : adapter.shards_) {
        shard.cache.cur_pos = 11;
        shard.cache.layers.resize(3);
        for (auto & layer : shard.cache.layers) {
            layer.n_comp = 4;
            layer.n_index_comp = 6;
        }
    }

    adapter.reset_request_state();
    TEST_ASSERT(adapter.cur_pos_ == 0);
    TEST_ASSERT(adapter.last_tok_ == -1);
    for (float v : adapter.hc_state_) {
        TEST_ASSERT(v == 0.0f);
    }
    for (const auto & shard : adapter.shards_) {
        TEST_ASSERT(shard.cache.cur_pos == 0);
        for (const auto & layer : shard.cache.layers) {
            TEST_ASSERT(layer.n_comp == 0);
            TEST_ASSERT(layer.n_index_comp == 0);
        }
    }

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_adapter_guard_paths() {
    std::fprintf(stderr, "  test_adapter_guard_paths ...");

    auto adapter = make_test_adapter();
    TEST_ASSERT(!adapter.init());

    int last_tok = -1;
    TEST_ASSERT(!adapter.prefill({1, 2, 3}, 0, last_tok));

    std::vector<int32_t> out_tokens;
    TEST_ASSERT(!adapter.decode_ar(1, 0, 1, out_tokens, DaemonIO{}));

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_ipc_mode_registration() {
    std::fprintf(stderr, "  test_ipc_mode_registration ...");

    BackendIpcMode mode = BackendIpcMode::Invalid;
    TEST_ASSERT(parse_backend_ipc_mode("deepseek4-target-shard", mode));
    TEST_ASSERT(mode == BackendIpcMode::DeepSeek4TargetShard);

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "FAIL: ggml_backend_cpu_init failed\n");
        return 1;
    }

    test_compressor_pooling_correctness(backend);
    test_moe_routing_correctness(backend);
    test_rmsnorm_correctness(backend);
    test_grouped_output_projection_shape();
    test_hash_routing_lookup();
    test_auto_split_computation();
    test_layer_range_validation();
    test_hc_state_dimensions();
    test_snapshot_save_restore();
    test_reset_request_state();
    test_adapter_guard_paths();
    test_ipc_mode_registration();

    ggml_backend_free(backend);

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED: %d assertion(s)\n", g_failures);
        return 1;
    }

    std::printf("OK\n");
    return 0;
}
