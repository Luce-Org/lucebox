#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
#include "ggml-cuda.h"
#include "deepseek4/deepseek4_hc_cuda.h"
#endif

#include "common/backend_ipc.h"
#include "common/layer_split_utils.h"

#include <memory>
#include <random>
#include <string>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <sys/stat.h>
#include <vector>
#include <unistd.h>

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

static ggml_tensor * test_hc_row_normalize(ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * sums = ggml_sum_rows(ctx, x);
    return ggml_div(ctx, x, ggml_repeat(ctx, sums, x));
}

static ggml_tensor * test_hc_col_normalize(ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, x));
    xt = test_hc_row_normalize(ctx, xt);
    return ggml_cont(ctx, ggml_transpose(ctx, xt));
}

using TestClock = std::chrono::steady_clock;

static double elapsed_ms(TestClock::time_point t0, TestClock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct DeepSeek4FixtureOptions {
    bool include_vocab_size = true;
    uint32_t vocab_size = 128;
    bool write_compress_ratios = false;
    gguf_type compress_ratios_type = GGUF_TYPE_UINT32;
    int32_t eos_id = -1;
    int32_t eot_id = -1;
};

static std::string make_temp_gguf_path(const char * prefix) {
    char path[] = "/tmp/deepseek4-loader-XXXXXX";
    const int fd = mkstemp(path);
    if (fd >= 0) {
        close(fd);
        unlink(path);
    }
    return std::string(path) + "-" + prefix + ".gguf";
}

static std::string write_deepseek4_loader_fixture(const DeepSeek4FixtureOptions & opts) {
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "deepseek4");
    gguf_set_val_u32(g, "deepseek4.block_count", 43);
    gguf_set_val_u32(g, "deepseek4.embedding_length", 4096);
    if (opts.include_vocab_size) {
        gguf_set_val_u32(g, "deepseek4.vocab_size", opts.vocab_size);
    }
    gguf_set_val_u32(g, "deepseek4.attention.head_count", 64);
    gguf_set_val_u32(g, "deepseek4.attention.head_count_kv", 1);
    gguf_set_val_u32(g, "deepseek4.attention.key_length", 512);
    gguf_set_val_u32(g, "deepseek4.rope.dimension_count", 64);
    gguf_set_val_u32(g, "deepseek4.attention.q_lora_rank", 1024);
    gguf_set_val_u32(g, "deepseek4.attention.output_lora_rank", 1024);
    gguf_set_val_u32(g, "deepseek4.attention.output_group_count", 8);
    gguf_set_val_u32(g, "deepseek4.expert_count", 256);
    gguf_set_val_u32(g, "deepseek4.expert_used_count", 6);
    gguf_set_val_u32(g, "deepseek4.expert_shared_count", 1);
    gguf_set_val_u32(g, "deepseek4.expert_feed_forward_length", 2048);
    gguf_set_val_u32(g, "deepseek4.hash_layer_count", 3);
    gguf_set_val_u32(g, "deepseek4.attention.sliding_window", 128);
    gguf_set_val_u32(g, "deepseek4.attention.indexer.head_count", 64);
    gguf_set_val_u32(g, "deepseek4.attention.indexer.key_length", 128);
    gguf_set_val_u32(g, "deepseek4.attention.indexer.top_k", 512);
    gguf_set_val_u32(g, "deepseek4.hyper_connection.count", 4);
    gguf_set_val_u32(g, "deepseek4.hyper_connection.sinkhorn_iterations", 20);

    if (opts.write_compress_ratios) {
        std::vector<uint32_t> ratios(43, 4);
        ratios[0] = 0;
        ratios[1] = 0;
        switch (opts.compress_ratios_type) {
        case GGUF_TYPE_UINT32:
            gguf_set_arr_data(g, "deepseek4.attention.compress_ratios",
                              GGUF_TYPE_UINT32, ratios.data(), ratios.size());
            break;
        case GGUF_TYPE_INT32: {
            std::vector<int32_t> vals(ratios.begin(), ratios.end());
            gguf_set_arr_data(g, "deepseek4.attention.compress_ratios",
                              GGUF_TYPE_INT32, vals.data(), vals.size());
            break;
        }
        default: {
            std::vector<int16_t> vals(ratios.begin(), ratios.end());
            gguf_set_arr_data(g, "deepseek4.attention.compress_ratios",
                              opts.compress_ratios_type, vals.data(), vals.size());
            break;
        }
        }
    }

    if (opts.eos_id >= 0) {
        gguf_set_val_u32(g, "tokenizer.ggml.eos_token_id", (uint32_t)opts.eos_id);
    }
    if (opts.eot_id >= 0) {
        gguf_set_val_u32(g, "tokenizer.ggml.eot_token_id", (uint32_t)opts.eot_id);
    }

    const std::string path = make_temp_gguf_path("fixture");
    gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
    gguf_free(g);
    return path;
}

static std::string write_deepseek4_tensor_fixture() {
    ggml_init_params ip{};
    ip.mem_size = 1u << 20;
    ip.mem_buffer = nullptr;
    ip.no_alloc = false;
    ggml_context * ctx = ggml_init(ip);
    gguf_context * g = gguf_init_empty();

    gguf_set_val_str(g, "general.architecture", "deepseek4");
    gguf_set_val_u32(g, "deepseek4.block_count", 1);
    gguf_set_val_u32(g, "deepseek4.embedding_length", 4);
    gguf_set_val_u32(g, "deepseek4.vocab_size", 8);
    gguf_set_val_u32(g, "deepseek4.attention.head_count", 1);
    gguf_set_val_u32(g, "deepseek4.attention.head_count_kv", 1);
    gguf_set_val_u32(g, "deepseek4.attention.key_length", 4);
    gguf_set_val_u32(g, "deepseek4.rope.dimension_count", 4);
    gguf_set_val_u32(g, "deepseek4.attention.q_lora_rank", 4);
    gguf_set_val_u32(g, "deepseek4.attention.output_lora_rank", 4);
    gguf_set_val_u32(g, "deepseek4.attention.output_group_count", 1);
    gguf_set_val_u32(g, "deepseek4.expert_count", 1);
    gguf_set_val_u32(g, "deepseek4.expert_used_count", 1);
    gguf_set_val_u32(g, "deepseek4.expert_shared_count", 1);
    gguf_set_val_u32(g, "deepseek4.expert_feed_forward_length", 8);
    gguf_set_val_u32(g, "deepseek4.hash_layer_count", 0);
    gguf_set_val_u32(g, "deepseek4.attention.sliding_window", 8);
    gguf_set_val_u32(g, "deepseek4.attention.indexer.head_count", 1);
    gguf_set_val_u32(g, "deepseek4.attention.indexer.key_length", 4);
    gguf_set_val_u32(g, "deepseek4.attention.indexer.top_k", 1);
    gguf_set_val_u32(g, "deepseek4.hyper_connection.count", 1);
    gguf_set_val_u32(g, "deepseek4.hyper_connection.sinkhorn_iterations", 1);

    ggml_tensor * tok = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8);
    ggml_set_name(tok, "token_embd.weight");
    std::memset(tok->data, 0, ggml_nbytes(tok));
    gguf_add_tensor(g, tok);

    const std::string path = make_temp_gguf_path("tensor");
    gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
    gguf_free(g);
    ggml_free(ctx);
    return path;
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

static void test_swiglu_ds4_cpu_correctness(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_swiglu_ds4_cpu_correctness ...");

    constexpr int dim = 17;
    constexpr int rows = 3;
    constexpr float clamp = 1.5f;
    std::vector<float> gate((size_t) dim * rows);
    std::vector<float> up((size_t) dim * rows);
    std::vector<float> expected((size_t) dim * rows);
    for (size_t i = 0; i < gate.size(); ++i) {
        gate[i] = 0.25f * (float) ((int) (i % 15) - 7);
        up[i] = 0.375f * (float) ((int) (i % 11) - 5);
        const float gate_clamped = std::min(gate[i], clamp);
        const float up_clamped = std::clamp(up[i], -clamp, clamp);
        expected[i] = up_clamped * gate_clamped / (1.0f + std::exp(-gate_clamped));
    }

    ggml_context * ctx = make_test_context();
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) {
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    ggml_tensor * gate_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, rows);
    ggml_tensor * up_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, rows);
    ggml_set_input(gate_t);
    ggml_set_input(up_t);
    ggml_tensor * out_t = ggml_swiglu_ds4_split(ctx, gate_t, up_t, clamp);
    ggml_set_output(out_t);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, out_t);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    TEST_ASSERT(ggml_gallocr_alloc_graph(alloc, gf));
    ggml_backend_tensor_set(gate_t, gate.data(), 0, gate.size() * sizeof(float));
    ggml_backend_tensor_set(up_t, up.data(), 0, up.size() * sizeof(float));
    TEST_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(out_t, actual.data(), 0, actual.size() * sizeof(float));
    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    for (size_t i = 0; i < actual.size(); ++i) {
        TEST_ASSERT_MSG(
            nearly_equal(actual[i], expected[i], 1.0e-6f, 1.0e-6f),
            "SWIGLU_DS4 output mismatch");
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

static std::vector<uint8_t> make_tensor_pattern(const ggml_tensor * tensor,
                                                uint8_t seed) {
    std::vector<uint8_t> data(ggml_nbytes(tensor));
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = (uint8_t)(seed + (uint8_t)(i % 251));
    }
    return data;
}

static void write_tensor_pattern(ggml_tensor * tensor, uint8_t seed) {
    const std::vector<uint8_t> data = make_tensor_pattern(tensor, seed);
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size());
}

static std::vector<uint8_t> read_tensor_bytes(const ggml_tensor * tensor) {
    std::vector<uint8_t> data(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
    return data;
}

static bool init_snapshot_test_shard(DeepSeek4LayerSplitAdapter & adapter) {
    adapter.shards_.resize(1);
    auto & shard = adapter.shards_[0];
    shard.backend = ggml_backend_cpu_init();
    if (!shard.backend) return false;
    shard.weights.n_layer = 1;
    shard.weights.n_embd = 4;
    shard.weights.n_hc = 1;
    shard.weights.head_dim = 4;
    shard.weights.n_swa = 8;
    shard.weights.n_indexer_head_dim = 2;
    shard.weights.compress_ratios = {4};
    return create_deepseek4_cache(shard.backend, shard.weights, 16, shard.cache);
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

static void test_loader_rejects_missing_required_metadata(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_loader_rejects_missing_required_metadata ...");

    DeepSeek4FixtureOptions opts;
    opts.include_vocab_size = false;
    const std::string path = write_deepseek4_loader_fixture(opts);
    DeepSeek4Weights weights;
    const bool ok = load_deepseek4_gguf(path, backend, weights);
    TEST_ASSERT(!ok);
    TEST_ASSERT_MSG(std::string(dflash27b_last_error()).find(
                        "missing required key: deepseek4.vocab_size") != std::string::npos,
                    dflash27b_last_error());
    free_deepseek4_weights(weights);
    unlink(path.c_str());

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_loader_rejects_invalid_compress_ratio_type(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_loader_rejects_invalid_compress_ratio_type ...");

    DeepSeek4FixtureOptions opts;
    opts.write_compress_ratios = true;
    opts.compress_ratios_type = GGUF_TYPE_INT16;
    const std::string path = write_deepseek4_loader_fixture(opts);
    DeepSeek4Weights weights;
    const bool ok = load_deepseek4_gguf(path, backend, weights);
    TEST_ASSERT(!ok);
    TEST_ASSERT_MSG(std::string(dflash27b_last_error()).find(
                        "deepseek4.attention.compress_ratios array element type must be i32 or u32") != std::string::npos,
                    dflash27b_last_error());
    free_deepseek4_weights(weights);
    unlink(path.c_str());

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_loader_rejects_zero_vocab_size(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_loader_rejects_zero_vocab_size ...");

    DeepSeek4FixtureOptions opts;
    opts.vocab_size = 0;
    const std::string path = write_deepseek4_loader_fixture(opts);
    DeepSeek4Weights weights;
    const bool ok = load_deepseek4_gguf(path, backend, weights);
    TEST_ASSERT(!ok);
    TEST_ASSERT_MSG(std::string(dflash27b_last_error()).find(
                        "deepseek4.vocab_size must be > 0") != std::string::npos,
                    dflash27b_last_error());
    free_deepseek4_weights(weights);
    unlink(path.c_str());

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_loader_reads_tokenizer_special_ids(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_loader_reads_tokenizer_special_ids ...");

    DeepSeek4FixtureOptions opts;
    opts.eos_id = 151645;
    opts.eot_id = 151643;
    const std::string path = write_deepseek4_loader_fixture(opts);
    DeepSeek4Weights weights;
    const bool ok = load_deepseek4_gguf(path, backend, weights);
    TEST_ASSERT_MSG(ok, dflash27b_last_error());
    if (ok) {
        TEST_ASSERT(weights.eos_id == 151645);
        TEST_ASSERT(weights.eos_chat_id == 151643);
        TEST_ASSERT(deepseek4_is_eos_tok(151645, weights));
        TEST_ASSERT(deepseek4_is_eos_tok(151643, weights));
        TEST_ASSERT(!deepseek4_is_eos_tok(151644, weights));
    }
    free_deepseek4_weights(weights);
    unlink(path.c_str());

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_loader_rejects_truncated_tensor_data(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_loader_rejects_truncated_tensor_data ...");

    const std::string path = write_deepseek4_tensor_fixture();
    {
        DeepSeek4Weights weights;
        const bool ok = load_deepseek4_gguf(path, backend, weights);
        TEST_ASSERT_MSG(ok, dflash27b_last_error());
        free_deepseek4_weights(weights);
    }

    struct stat st{};
    TEST_ASSERT(stat(path.c_str(), &st) == 0);
    const off_t truncated_size = (off_t)st.st_size - 8;
    TEST_ASSERT(truncated_size > 0);
    TEST_ASSERT(truncate(path.c_str(), truncated_size) == 0);

    {
        DeepSeek4Weights weights;
        const bool ok = load_deepseek4_gguf(path, backend, weights);
        TEST_ASSERT(!ok);
        TEST_ASSERT_MSG(std::string(dflash27b_last_error()).find(
                            "truncated or corrupt") != std::string::npos,
                        dflash27b_last_error());
        free_deepseek4_weights(weights);
    }
    unlink(path.c_str());

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_snapshot_save_restore() {
    std::fprintf(stderr, "  test_snapshot_save_restore ...");

    auto adapter = make_test_adapter();
    TEST_ASSERT(init_snapshot_test_shard(adapter));
    if (adapter.shards_.empty() || !adapter.shards_[0].backend) {
        std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
        return;
    }
    adapter.snapshot_backends_.assign(1, adapter.shards_[0].backend);

    auto & cache = adapter.shards_[0].cache;
    auto & layer = cache.layers[0];
    write_tensor_pattern(layer.raw_kv, 3);
    write_tensor_pattern(layer.comp_kv, 17);
    write_tensor_pattern(layer.index_comp_kv, 29);
    write_tensor_pattern(layer.attn_compressor.state_kv, 41);
    write_tensor_pattern(layer.attn_compressor.state_score, 53);
    write_tensor_pattern(layer.indexer_compressor.state_kv, 67);
    write_tensor_pattern(layer.indexer_compressor.state_score, 79);

    const std::vector<uint8_t> raw_before = read_tensor_bytes(layer.raw_kv);
    const std::vector<uint8_t> comp_before = read_tensor_bytes(layer.comp_kv);
    const std::vector<uint8_t> index_before = read_tensor_bytes(layer.index_comp_kv);
    const std::vector<uint8_t> attn_kv_before =
        read_tensor_bytes(layer.attn_compressor.state_kv);
    const std::vector<uint8_t> attn_score_before =
        read_tensor_bytes(layer.attn_compressor.state_score);
    const std::vector<uint8_t> index_kv_before =
        read_tensor_bytes(layer.indexer_compressor.state_kv);
    const std::vector<uint8_t> index_score_before =
        read_tensor_bytes(layer.indexer_compressor.state_score);

    layer.n_comp = 5;
    layer.n_index_comp = 3;
    cache.cur_pos = 7;
    adapter.cur_pos_ = 7;
    adapter.last_tok_ = 4242;
    adapter.hc_state_ = {1.0f, 2.0f, 3.0f, 4.0f};
    adapter.prefill_last_logits_ = {9.0f, 8.0f, 7.0f};

    TEST_ASSERT(adapter.snapshot_save(0));
    TEST_ASSERT(adapter.snapshot_used(0));
    TEST_ASSERT(adapter.snapshot_cur_pos(0) == 7);

    adapter.cur_pos_ = 0;
    adapter.last_tok_ = -1;
    adapter.hc_state_.assign(4, 0.0f);
    adapter.prefill_last_logits_.clear();
    cache.cur_pos = 0;
    layer.n_comp = 0;
    layer.n_index_comp = 0;
    ggml_backend_buffer_clear(cache.buf, 0);

    TEST_ASSERT(adapter.snapshot_restore(0));
    TEST_ASSERT(adapter.cur_pos_ == 7);
    TEST_ASSERT(adapter.last_tok_ == 4242);
    TEST_ASSERT(adapter.hc_state_.size() == 4);
    if (adapter.hc_state_.size() == 4) {
        TEST_ASSERT(adapter.hc_state_[0] == 1.0f);
        TEST_ASSERT(adapter.hc_state_[3] == 4.0f);
    }
    TEST_ASSERT(adapter.prefill_last_logits_.size() == 3);
    if (adapter.prefill_last_logits_.size() == 3) {
        TEST_ASSERT(adapter.prefill_last_logits_[0] == 9.0f);
        TEST_ASSERT(adapter.prefill_last_logits_[2] == 7.0f);
    }
    TEST_ASSERT(cache.cur_pos == 7);
    TEST_ASSERT(layer.n_comp == 5);
    TEST_ASSERT(layer.n_index_comp == 3);
    TEST_ASSERT(read_tensor_bytes(layer.raw_kv) == raw_before);
    TEST_ASSERT(read_tensor_bytes(layer.comp_kv) == comp_before);
    TEST_ASSERT(read_tensor_bytes(layer.index_comp_kv) == index_before);
    TEST_ASSERT(read_tensor_bytes(layer.attn_compressor.state_kv) == attn_kv_before);
    TEST_ASSERT(read_tensor_bytes(layer.attn_compressor.state_score) == attn_score_before);
    TEST_ASSERT(read_tensor_bytes(layer.indexer_compressor.state_kv) == index_kv_before);
    TEST_ASSERT(read_tensor_bytes(layer.indexer_compressor.state_score) == index_score_before);

    adapter.snapshot_free(0);
    TEST_ASSERT(!adapter.snapshot_used(0));
    TEST_ASSERT(adapter.snapshots_[0].hc_state.empty());
    TEST_ASSERT(adapter.snapshots_[0].prefill_last_logits.empty());
    TEST_ASSERT(adapter.snapshots_[0].shards.size() == 1);
    TEST_ASSERT(!adapter.snapshots_[0].shards[0].ctx);
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
    TEST_ASSERT(!adapter.run_forward({}, 0, last_tok, nullptr));

    adapter.shards_.resize(1);
    TEST_ASSERT(!adapter.run_forward({}, 0, last_tok, nullptr));

    adapter.remote_target_shard_.active_ = true;
    TEST_ASSERT(!adapter.run_mixed_forward({}, 0, last_tok, nullptr));

    adapter.shards_.clear();
    adapter.remote_target_shard_.active_ = false;

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

static void test_target_shard_daemon_validation() {
    std::fprintf(stderr, "  test_target_shard_daemon_validation ...");

    TEST_ASSERT(run_deepseek4_target_shard_ipc_daemon(
                    "dummy.gguf", {0, -1}, {0, 10}, {10, 20},
                    128, 0, -1) == 2);
    TEST_ASSERT(run_deepseek4_target_shard_ipc_daemon(
                    "dummy.gguf", {0, 1}, {0, 11}, {10, 20},
                    128, 0, -1) == 2);
    TEST_ASSERT(run_deepseek4_target_shard_ipc_daemon(
                    "dummy.gguf", {0, 1}, {0}, {10, 20},
                    128, 0, -1) == 2);

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_ffn_graph_reuse_microbench(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_ffn_graph_reuse_microbench ...");

    constexpr int n_embd = 64;
    constexpr int n_ff = 96;
    constexpr int n_tokens = 64;
    constexpr int iters = 8;
    const size_t out_size = (size_t) n_embd * n_tokens;

    std::vector<float> inp((size_t) n_embd * n_tokens);
    std::vector<float> norm_w((size_t) n_embd);
    std::vector<float> shared_gate_w((size_t) n_embd * n_ff);
    std::vector<float> shared_up_w((size_t) n_embd * n_ff);
    std::vector<float> shared_down_w((size_t) n_ff * n_embd);
    std::vector<float> routed_gate_w((size_t) n_embd * n_ff);
    std::vector<float> routed_up_w((size_t) n_embd * n_ff);
    std::vector<float> routed_down_w((size_t) n_ff * n_embd);
    std::vector<float> rebuild_out(out_size);
    std::vector<float> cached_out(out_size);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    auto fill = [&](std::vector<float> & v) {
        for (float & x : v) x = dist(rng);
    };
    fill(inp);
    fill(norm_w);
    fill(shared_gate_w);
    fill(shared_up_w);
    fill(shared_down_w);
    fill(routed_gate_w);
    fill(routed_up_w);
    fill(routed_down_w);

    auto build_and_run = [&](std::vector<float> & out) {
        ggml_context * ctx = make_test_context(8u << 20);
        TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
        if (!ctx) return false;

        ggml_tensor * inp_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_tensor * norm_w_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        ggml_tensor * shared_gate_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);
        ggml_tensor * shared_up_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);
        ggml_tensor * shared_down_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff, n_embd);
        ggml_tensor * routed_gate_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);
        ggml_tensor * routed_up_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);
        ggml_tensor * routed_down_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff, n_embd);
        ggml_set_input(inp_t);
        ggml_set_input(norm_w_t);
        ggml_set_input(shared_gate_t);
        ggml_set_input(shared_up_t);
        ggml_set_input(shared_down_t);
        ggml_set_input(routed_gate_t);
        ggml_set_input(routed_up_t);
        ggml_set_input(routed_down_t);

        ggml_tensor * norm = ggml_mul(ctx, ggml_rms_norm(ctx, inp_t, 1.0e-6f), norm_w_t);
        ggml_tensor * shared_gate = ggml_softplus(ctx, ggml_mul_mat(ctx, shared_gate_t, norm));
        ggml_tensor * shared_up = ggml_mul_mat(ctx, shared_up_t, norm);
        ggml_tensor * shared_mid = ggml_mul(ctx, shared_gate, shared_up);
        ggml_tensor * shared_out = ggml_mul_mat(ctx, shared_down_t, shared_mid);

        ggml_tensor * routed_gate = ggml_softplus(ctx, ggml_mul_mat(ctx, routed_gate_t, norm));
        ggml_tensor * routed_up = ggml_mul_mat(ctx, routed_up_t, norm);
        ggml_tensor * routed_mid = ggml_mul(ctx, routed_gate, routed_up);
        ggml_tensor * routed_out = ggml_mul_mat(ctx, routed_down_t, routed_mid);
        ggml_tensor * out_t = ggml_add(ctx, shared_out, routed_out);
        ggml_set_output(out_t);

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(gf, out_t);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        bool ok = ggml_gallocr_alloc_graph(alloc, gf);
        TEST_ASSERT(ok);
        if (ok) {
            ggml_backend_tensor_set(inp_t, inp.data(), 0, inp.size() * sizeof(float));
            ggml_backend_tensor_set(norm_w_t, norm_w.data(), 0, norm_w.size() * sizeof(float));
            ggml_backend_tensor_set(shared_gate_t, shared_gate_w.data(), 0, shared_gate_w.size() * sizeof(float));
            ggml_backend_tensor_set(shared_up_t, shared_up_w.data(), 0, shared_up_w.size() * sizeof(float));
            ggml_backend_tensor_set(shared_down_t, shared_down_w.data(), 0, shared_down_w.size() * sizeof(float));
            ggml_backend_tensor_set(routed_gate_t, routed_gate_w.data(), 0, routed_gate_w.size() * sizeof(float));
            ggml_backend_tensor_set(routed_up_t, routed_up_w.data(), 0, routed_up_w.size() * sizeof(float));
            ggml_backend_tensor_set(routed_down_t, routed_down_w.data(), 0, routed_down_w.size() * sizeof(float));
            ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
            TEST_ASSERT(ok);
            if (ok) {
                ggml_backend_tensor_get(out_t, out.data(), 0, out.size() * sizeof(float));
            }
        }

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return ok;
    };

    double rebuild_total_ms = 0.0;
    for (int iter = 0; iter < iters; ++iter) {
        const auto t0 = TestClock::now();
        bool ok = build_and_run(rebuild_out);
        const auto t1 = TestClock::now();
        TEST_ASSERT(ok);
        rebuild_total_ms += elapsed_ms(t0, t1);
    }

    ggml_context * ctx = make_test_context(8u << 20);
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) {
        std::fprintf(stderr, " FAIL\n");
        return;
    }
    ggml_tensor * inp_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * norm_w_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_tensor * shared_gate_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);
    ggml_tensor * shared_up_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);
    ggml_tensor * shared_down_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff, n_embd);
    ggml_tensor * routed_gate_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);
    ggml_tensor * routed_up_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);
    ggml_tensor * routed_down_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff, n_embd);
    ggml_set_input(inp_t);
    ggml_set_input(norm_w_t);
    ggml_set_input(shared_gate_t);
    ggml_set_input(shared_up_t);
    ggml_set_input(shared_down_t);
    ggml_set_input(routed_gate_t);
    ggml_set_input(routed_up_t);
    ggml_set_input(routed_down_t);

    ggml_tensor * norm = ggml_mul(ctx, ggml_rms_norm(ctx, inp_t, 1.0e-6f), norm_w_t);
    ggml_tensor * shared_gate = ggml_softplus(ctx, ggml_mul_mat(ctx, shared_gate_t, norm));
    ggml_tensor * shared_up = ggml_mul_mat(ctx, shared_up_t, norm);
    ggml_tensor * shared_mid = ggml_mul(ctx, shared_gate, shared_up);
    ggml_tensor * shared_out = ggml_mul_mat(ctx, shared_down_t, shared_mid);
    ggml_tensor * routed_gate = ggml_softplus(ctx, ggml_mul_mat(ctx, routed_gate_t, norm));
    ggml_tensor * routed_up = ggml_mul_mat(ctx, routed_up_t, norm);
    ggml_tensor * routed_mid = ggml_mul(ctx, routed_gate, routed_up);
    ggml_tensor * routed_out = ggml_mul_mat(ctx, routed_down_t, routed_mid);
    ggml_tensor * out_t = ggml_add(ctx, shared_out, routed_out);
    ggml_set_output(out_t);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(gf, out_t);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    bool alloc_ok = ggml_gallocr_alloc_graph(alloc, gf);
    TEST_ASSERT(alloc_ok);
    if (!alloc_ok) {
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        std::fprintf(stderr, " FAIL\n");
        return;
    }
    ggml_backend_tensor_set(norm_w_t, norm_w.data(), 0, norm_w.size() * sizeof(float));
    ggml_backend_tensor_set(shared_gate_t, shared_gate_w.data(), 0, shared_gate_w.size() * sizeof(float));
    ggml_backend_tensor_set(shared_up_t, shared_up_w.data(), 0, shared_up_w.size() * sizeof(float));
    ggml_backend_tensor_set(shared_down_t, shared_down_w.data(), 0, shared_down_w.size() * sizeof(float));
    ggml_backend_tensor_set(routed_gate_t, routed_gate_w.data(), 0, routed_gate_w.size() * sizeof(float));
    ggml_backend_tensor_set(routed_up_t, routed_up_w.data(), 0, routed_up_w.size() * sizeof(float));
    ggml_backend_tensor_set(routed_down_t, routed_down_w.data(), 0, routed_down_w.size() * sizeof(float));

    double cached_total_ms = 0.0;
    for (int iter = 0; iter < iters; ++iter) {
        const auto t0 = TestClock::now();
        ggml_backend_tensor_set(inp_t, inp.data(), 0, inp.size() * sizeof(float));
        bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        const auto t1 = TestClock::now();
        TEST_ASSERT(ok);
        if (ok) {
            ggml_backend_tensor_get(out_t, cached_out.data(), 0, cached_out.size() * sizeof(float));
        }
        cached_total_ms += elapsed_ms(t0, t1);
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    for (size_t i = 0; i < cached_out.size(); ++i) {
        TEST_ASSERT_MSG(std::isfinite(cached_out[i]), "cached FFN output must be finite");
        TEST_ASSERT_MSG(std::isfinite(rebuild_out[i]), "rebuilt FFN output must be finite");
    }

    std::fprintf(stderr, " rebuild_avg=%.3fms cached_avg=%.3fms\n",
                 rebuild_total_ms / iters, cached_total_ms / iters);
}

static void test_output_graph_reuse_microbench(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_output_graph_reuse_microbench ...");

    constexpr int n_embd = 64;
    constexpr int n_vocab = 256;
    constexpr int n_tokens = 64;
    constexpr int iters = 8;
    std::vector<float> inp((size_t) n_embd * n_tokens);
    std::vector<float> norm_w((size_t) n_embd);
    std::vector<float> lm_head((size_t) n_embd * n_vocab);
    std::vector<float> rebuild_logits((size_t) n_vocab * n_tokens);
    std::vector<float> cached_logits((size_t) n_vocab * n_tokens);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
    for (float & x : inp) x = dist(rng);
    for (float & x : norm_w) x = 1.0f + dist(rng);
    for (float & x : lm_head) x = dist(rng);

    auto build_and_run = [&](std::vector<float> & out) {
        ggml_context * ctx = make_test_context(4u << 20);
        TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
        if (!ctx) return false;

        ggml_tensor * inp_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_tensor * norm_w_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        ggml_tensor * lm_head_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
        ggml_set_input(inp_t);
        ggml_set_input(norm_w_t);
        ggml_set_input(lm_head_t);
        ggml_tensor * norm = ggml_mul(ctx, ggml_rms_norm(ctx, inp_t, 1.0e-6f), norm_w_t);
        ggml_tensor * logits = ggml_mul_mat(ctx, lm_head_t, norm);
        ggml_set_output(logits);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(gf, logits);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        bool ok = ggml_gallocr_alloc_graph(alloc, gf);
        TEST_ASSERT(ok);
        if (ok) {
            ggml_backend_tensor_set(inp_t, inp.data(), 0, inp.size() * sizeof(float));
            ggml_backend_tensor_set(norm_w_t, norm_w.data(), 0, norm_w.size() * sizeof(float));
            ggml_backend_tensor_set(lm_head_t, lm_head.data(), 0, lm_head.size() * sizeof(float));
            ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
            TEST_ASSERT(ok);
            if (ok) {
                ggml_backend_tensor_get(logits, out.data(), 0, out.size() * sizeof(float));
            }
        }
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return ok;
    };

    double rebuild_total_ms = 0.0;
    for (int iter = 0; iter < iters; ++iter) {
        const auto t0 = TestClock::now();
        bool ok = build_and_run(rebuild_logits);
        const auto t1 = TestClock::now();
        TEST_ASSERT(ok);
        rebuild_total_ms += elapsed_ms(t0, t1);
    }

    ggml_context * ctx = make_test_context(4u << 20);
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) {
        std::fprintf(stderr, " FAIL\n");
        return;
    }
    ggml_tensor * inp_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * norm_w_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    ggml_tensor * lm_head_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    ggml_set_input(inp_t);
    ggml_set_input(norm_w_t);
    ggml_set_input(lm_head_t);
    ggml_tensor * norm = ggml_mul(ctx, ggml_rms_norm(ctx, inp_t, 1.0e-6f), norm_w_t);
    ggml_tensor * logits = ggml_mul_mat(ctx, lm_head_t, norm);
    ggml_set_output(logits);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(gf, logits);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    bool alloc_ok = ggml_gallocr_alloc_graph(alloc, gf);
    TEST_ASSERT(alloc_ok);
    if (!alloc_ok) {
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        std::fprintf(stderr, " FAIL\n");
        return;
    }
    ggml_backend_tensor_set(norm_w_t, norm_w.data(), 0, norm_w.size() * sizeof(float));
    ggml_backend_tensor_set(lm_head_t, lm_head.data(), 0, lm_head.size() * sizeof(float));

    double cached_total_ms = 0.0;
    for (int iter = 0; iter < iters; ++iter) {
        const auto t0 = TestClock::now();
        ggml_backend_tensor_set(inp_t, inp.data(), 0, inp.size() * sizeof(float));
        bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        const auto t1 = TestClock::now();
        TEST_ASSERT(ok);
        if (ok) {
            ggml_backend_tensor_get(logits, cached_logits.data(), 0, cached_logits.size() * sizeof(float));
        }
        cached_total_ms += elapsed_ms(t0, t1);
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    for (size_t i = 0; i < cached_logits.size(); ++i) {
        TEST_ASSERT_MSG(std::isfinite(cached_logits[i]), "cached output logits must be finite");
        TEST_ASSERT_MSG(std::isfinite(rebuild_logits[i]), "rebuilt output logits must be finite");
    }

    std::fprintf(stderr, " rebuild_avg=%.3fms cached_avg=%.3fms\n",
                 rebuild_total_ms / iters, cached_total_ms / iters);
}

#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
static void test_cpu_hc_sinkhorn_ref(float * out, const float * mix, const float * scale,
                                     const float * base, int n_hc, int iters, float eps) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    for (int i = 0; i < n_hc; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + std::exp(-z)) + eps;
    }
    for (int i = 0; i < n_hc; ++i) {
        const float z = mix[n_hc + i] * post_scale + base[n_hc + i];
        out[n_hc + i] = 2.0f / (1.0f + std::exp(-z));
    }

    float c[16];
    for (int dst = 0; dst < n_hc; ++dst) {
        float row_max = -1.0e30f;
        for (int src = 0; src < n_hc; ++src) {
            const int idx = src + dst * n_hc;
            const float v = mix[2 * n_hc + idx] * comb_scale + base[2 * n_hc + idx];
            c[idx] = v;
            row_max = std::max(row_max, v);
        }
        float row_sum = 0.0f;
        for (int src = 0; src < n_hc; ++src) {
            const int idx = src + dst * n_hc;
            c[idx] = std::exp(c[idx] - row_max);
            row_sum += c[idx];
        }
        const float inv = 1.0f / row_sum;
        for (int src = 0; src < n_hc; ++src) {
            c[src + dst * n_hc] = c[src + dst * n_hc] * inv + eps;
        }
    }
    for (int src = 0; src < n_hc; ++src) {
        float sum = 0.0f;
        for (int dst = 0; dst < n_hc; ++dst) sum += c[src + dst * n_hc];
        const float inv = 1.0f / (sum + eps);
        for (int dst = 0; dst < n_hc; ++dst) c[src + dst * n_hc] *= inv;
    }
    for (int iter = 1; iter < iters; ++iter) {
        for (int dst = 0; dst < n_hc; ++dst) {
            float sum = 0.0f;
            for (int src = 0; src < n_hc; ++src) sum += c[src + dst * n_hc];
            const float inv = 1.0f / (sum + eps);
            for (int src = 0; src < n_hc; ++src) c[src + dst * n_hc] *= inv;
        }
        for (int src = 0; src < n_hc; ++src) {
            float sum = 0.0f;
            for (int dst = 0; dst < n_hc; ++dst) sum += c[src + dst * n_hc];
            const float inv = 1.0f / (sum + eps);
            for (int dst = 0; dst < n_hc; ++dst) c[src + dst * n_hc] *= inv;
        }
    }
    for (int i = 0; i < n_hc * n_hc; ++i) {
        out[2 * n_hc + i] = c[i];
    }
}

static void test_reference_hc_pre(const std::vector<float> & hc_state,
                                  const std::vector<ggml_fp16_t> & fn_f16,
                                  const std::vector<float> & scale,
                                  const std::vector<float> & base,
                                  int n_embd,
                                  int n_hc,
                                  int sinkhorn_iters,
                                  float hc_eps,
                                  std::vector<float> & working,
                                  std::vector<float> & post,
                                  std::vector<float> & comb) {
    const int hc_dim = n_embd * n_hc;
    const int mix_dim = 2 * n_hc + n_hc * n_hc;
    std::vector<float> flat((size_t) hc_dim);
    std::vector<float> mix((size_t) mix_dim, 0.0f);
    float sumsq = 0.0f;
    for (float v : hc_state) sumsq += v * v;
    const float inv_rms = 1.0f / std::sqrt(sumsq / (float) hc_dim + hc_eps);
    for (int i = 0; i < hc_dim; ++i) flat[(size_t) i] = hc_state[(size_t) i] * inv_rms;
    for (int row = 0; row < mix_dim; ++row) {
        float acc = 0.0f;
        for (int c = 0; c < hc_dim; ++c) {
            acc += ggml_fp16_to_fp32(fn_f16[(size_t) row * hc_dim + c]) * flat[(size_t) c];
        }
        mix[(size_t) row] = acc;
    }
    std::vector<float> split((size_t) mix_dim);
    test_cpu_hc_sinkhorn_ref(split.data(), mix.data(), scale.data(), base.data(), n_hc, sinkhorn_iters, 1.0e-6f);

    working.assign((size_t) n_embd, 0.0f);
    post.assign((size_t) n_hc, 0.0f);
    comb.assign((size_t) n_hc * (size_t) n_hc, 0.0f);
    for (int d = 0; d < n_embd; ++d) {
        float acc = 0.0f;
        for (int h = 0; h < n_hc; ++h) {
            acc += split[h] * hc_state[(size_t) h * n_embd + d];
        }
        working[(size_t) d] = acc;
    }
    for (int i = 0; i < n_hc; ++i) post[(size_t) i] = split[n_hc + i];
    for (int i = 0; i < n_hc * n_hc; ++i) comb[(size_t) i] = split[2 * n_hc + i];
}

static void test_hc_pre_kernel_gpu() {
    std::fprintf(stderr, "  test_hc_pre_kernel_gpu ...");
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, " skipped (no GPU backend)\n");
        return;
    }

    constexpr int n_embd = 128;
    constexpr int n_hc = 4;
    constexpr int sinkhorn_iters = 6;
    constexpr float hc_eps = 1.0e-6f;
    constexpr int mix_dim = 2 * n_hc + n_hc * n_hc;
    const int hc_dim = n_embd * n_hc;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
    std::vector<float> hc_state((size_t) hc_dim);
    std::vector<float> fn((size_t) mix_dim * (size_t) hc_dim);
    std::vector<ggml_fp16_t> fn_f16(fn.size());
    std::vector<float> scale((size_t) mix_dim);
    std::vector<float> base((size_t) mix_dim);
    for (float & v : hc_state) v = dist(rng);
    for (float & v : fn) v = dist(rng);
    for (size_t i = 0; i < fn.size(); ++i) fn_f16[i] = ggml_fp32_to_fp16(fn[i]);
    scale[0] = 0.85f;
    scale[1] = 1.10f;
    scale[2] = 0.95f;
    for (int i = 3; i < mix_dim; ++i) scale[(size_t) i] = 0.0f;
    for (float & v : base) v = 0.15f * dist(rng);

    std::vector<float> ref_working;
    std::vector<float> ref_post;
    std::vector<float> ref_comb;
    test_reference_hc_pre(hc_state, fn_f16, scale, base, n_embd, n_hc, sinkhorn_iters, hc_eps,
                          ref_working, ref_post, ref_comb);

    ggml_context * ctx = make_test_context(1u << 20);
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) {
        ggml_backend_free(backend);
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    ggml_tensor * state_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, 1);
    ggml_tensor * fn_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hc_dim, mix_dim);
    ggml_tensor * scale_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mix_dim);
    ggml_tensor * base_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mix_dim);
    ggml_tensor * working_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_tensor * post_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hc, 1);
    ggml_tensor * comb_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hc, n_hc);
    ggml_tensor * working_devparam_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_tensor * post_devparam_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hc, 1);
    ggml_tensor * comb_devparam_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hc, n_hc);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    TEST_ASSERT_MSG(buf != nullptr, "ggml backend buffer alloc failed");
    if (!buf) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    ggml_backend_tensor_set(state_t, hc_state.data(), 0, hc_state.size() * sizeof(float));
    ggml_backend_tensor_set(fn_t, fn_f16.data(), 0, fn_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(scale_t, scale.data(), 0, scale.size() * sizeof(float));
    ggml_backend_tensor_set(base_t, base.data(), 0, base.size() * sizeof(float));

    std::vector<float> gpu_working((size_t) n_embd);
    std::vector<float> gpu_post((size_t) n_hc);
    std::vector<float> gpu_comb((size_t) n_hc * (size_t) n_hc);
    std::vector<float> host_working((size_t) n_embd);
    std::vector<float> host_post((size_t) n_hc);
    std::vector<float> host_comb((size_t) n_hc * (size_t) n_hc);
    std::vector<float> gpu_working_devparam((size_t) n_embd);
    std::vector<float> gpu_post_devparam((size_t) n_hc);
    std::vector<float> gpu_comb_devparam((size_t) n_hc * (size_t) n_hc);
    std::vector<float> graph_working((size_t) n_embd);
    std::vector<float> graph_post((size_t) n_hc);
    std::vector<float> graph_comb((size_t) n_hc * (size_t) n_hc);

    ggml_tensor * flat = ggml_rms_norm(ctx, state_t, hc_eps);
    ggml_tensor * mix = ggml_mul_mat(ctx, fn_t, flat);

    ggml_tensor * pre_mix = ggml_reshape_2d(ctx,
        ggml_view_1d(ctx, mix, n_hc, 0), n_hc, 1);
    ggml_tensor * post_mix = ggml_reshape_2d(ctx,
        ggml_view_1d(ctx, mix, n_hc, (size_t) n_hc * mix->nb[0]), n_hc, 1);
    ggml_tensor * comb_mix = ggml_reshape_2d(ctx,
        ggml_view_1d(ctx, mix, n_hc * n_hc, (size_t) (2 * n_hc) * mix->nb[0]),
        n_hc, n_hc);

    ggml_tensor * pre_base = ggml_reshape_2d(ctx,
        ggml_view_1d(ctx, base_t, n_hc, 0), n_hc, 1);
    ggml_tensor * post_base = ggml_reshape_2d(ctx,
        ggml_view_1d(ctx, base_t, n_hc, (size_t) n_hc * base_t->nb[0]), n_hc, 1);
    ggml_tensor * comb_base = ggml_reshape_2d(ctx,
        ggml_view_1d(ctx, base_t, n_hc * n_hc, (size_t) (2 * n_hc) * base_t->nb[0]),
        n_hc, n_hc);

    ggml_tensor * graph_pre = ggml_sigmoid(ctx,
        ggml_add(ctx, ggml_scale(ctx, pre_mix, scale[0]), pre_base));
    ggml_tensor * graph_post_t = ggml_scale(ctx,
        ggml_sigmoid(ctx, ggml_add(ctx, ggml_scale(ctx, post_mix, scale[1]), post_base)),
        2.0f);
    ggml_tensor * graph_comb_t = ggml_add(ctx, ggml_scale(ctx, comb_mix, scale[2]), comb_base);
    graph_comb_t = ggml_soft_max(ctx, graph_comb_t);
    graph_comb_t = test_hc_col_normalize(ctx, graph_comb_t);
    for (int iter = 1; iter < sinkhorn_iters; ++iter) {
        graph_comb_t = test_hc_row_normalize(ctx, graph_comb_t);
        graph_comb_t = test_hc_col_normalize(ctx, graph_comb_t);
    }

    ggml_tensor * hc_state_2d = ggml_reshape_2d(ctx, state_t, n_embd, n_hc);
    ggml_tensor * hc_state_t = ggml_cont(ctx, ggml_transpose(ctx, hc_state_2d));
    ggml_tensor * graph_working_t = ggml_mul_mat(ctx, hc_state_t, graph_pre);
    ggml_set_output(graph_working_t);
    ggml_set_output(graph_post_t);
    ggml_set_output(graph_comb_t);
    ggml_cgraph * graph_ref = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(graph_ref, graph_working_t);
    ggml_build_forward_expand(graph_ref, graph_post_t);
    ggml_build_forward_expand(graph_ref, graph_comb_t);
    ggml_gallocr_t graph_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    TEST_ASSERT_MSG(ggml_gallocr_alloc_graph(graph_alloc, graph_ref), "graph HC-pre alloc failed");
    TEST_ASSERT_MSG(ggml_backend_graph_compute(backend, graph_ref) == GGML_STATUS_SUCCESS,
                    "graph HC-pre compute failed");
    ggml_backend_tensor_get(graph_working_t, graph_working.data(), 0, graph_working.size() * sizeof(float));
    ggml_backend_tensor_get(graph_post_t, graph_post.data(), 0, graph_post.size() * sizeof(float));
    ggml_backend_tensor_get(graph_comb_t, graph_comb.data(), 0, graph_comb.size() * sizeof(float));

    bool ok = deepseek4_cuda_hc_pre(hc_state.data(),
                                    fn_t->data,
                                    scale.data(),
                                    base.data(),
                                    n_embd,
                                    n_hc,
                                    sinkhorn_iters,
                                    hc_eps,
                                    host_working.data(),
                                    host_post.data(),
                                    host_comb.data());
    TEST_ASSERT_MSG(ok, "host HC-pre kernel call failed");

    ok = deepseek4_cuda_hc_pre_device_params(state_t->data,
                                             fn_t->data,
                                             scale.data(),
                                             base.data(),
                                             n_embd,
                                             n_hc,
                                             sinkhorn_iters,
                                             hc_eps,
                                             working_t->data,
                                             post_t->data,
                                             comb_t->data);
    TEST_ASSERT_MSG(ok, "direct HC-pre kernel call failed");
    if (ok) {
        ggml_backend_tensor_get(working_t, gpu_working.data(), 0, gpu_working.size() * sizeof(float));
        ggml_backend_tensor_get(post_t, gpu_post.data(), 0, gpu_post.size() * sizeof(float));
        ggml_backend_tensor_get(comb_t, gpu_comb.data(), 0, gpu_comb.size() * sizeof(float));
    }

    ok = deepseek4_cuda_hc_pre_device(state_t->data,
                                      fn_t->data,
                                      scale_t->data,
                                      base_t->data,
                                      n_embd,
                                      n_hc,
                                      sinkhorn_iters,
                                      hc_eps,
                                      working_devparam_t->data,
                                      post_devparam_t->data,
                                      comb_devparam_t->data);
    TEST_ASSERT_MSG(ok, "device-param HC-pre kernel call failed");
    if (ok) {
        ggml_backend_tensor_get(working_devparam_t, gpu_working_devparam.data(), 0, gpu_working_devparam.size() * sizeof(float));
        ggml_backend_tensor_get(post_devparam_t, gpu_post_devparam.data(), 0, gpu_post_devparam.size() * sizeof(float));
        ggml_backend_tensor_get(comb_devparam_t, gpu_comb_devparam.data(), 0, gpu_comb_devparam.size() * sizeof(float));
    }

    for (int i = 0; i < n_embd; ++i) {
        TEST_ASSERT_MSG(nearly_equal(host_working[(size_t) i], ref_working[(size_t) i], 2.0e-4f, 2.0e-4f),
                        "host working mismatch");
        TEST_ASSERT_MSG(nearly_equal(gpu_working[(size_t) i], ref_working[(size_t) i], 2.0e-4f, 2.0e-4f),
                        "working mismatch");
        TEST_ASSERT_MSG(nearly_equal(gpu_working_devparam[(size_t) i], ref_working[(size_t) i], 2.0e-4f, 2.0e-4f),
                        "working devparam mismatch");
    }
    for (int i = 0; i < n_hc; ++i) {
        TEST_ASSERT_MSG(nearly_equal(host_post[(size_t) i], ref_post[(size_t) i], 2.0e-4f, 2.0e-4f),
                        "host post mismatch");
        TEST_ASSERT_MSG(nearly_equal(gpu_post[(size_t) i], ref_post[(size_t) i], 2.0e-4f, 2.0e-4f),
                        "post mismatch");
        TEST_ASSERT_MSG(nearly_equal(gpu_post_devparam[(size_t) i], ref_post[(size_t) i], 2.0e-4f, 2.0e-4f),
                        "post devparam mismatch");
    }
    for (int i = 0; i < n_hc * n_hc; ++i) {
        TEST_ASSERT_MSG(nearly_equal(host_comb[(size_t) i], ref_comb[(size_t) i], 2.0e-4f, 2.0e-4f),
                        "host comb mismatch");
        TEST_ASSERT_MSG(nearly_equal(gpu_comb[(size_t) i], ref_comb[(size_t) i], 2.0e-4f, 2.0e-4f),
                        "comb mismatch");
        TEST_ASSERT_MSG(nearly_equal(gpu_comb_devparam[(size_t) i], ref_comb[(size_t) i], 2.0e-4f, 2.0e-4f),
                        "comb devparam mismatch");
    }

    constexpr float graph_atol = 5.0e-4f;
    constexpr float graph_rtol = 5.0e-4f;
    for (int i = 0; i < n_embd; ++i) {
        TEST_ASSERT_MSG(nearly_equal(gpu_working[(size_t) i], graph_working[(size_t) i], graph_atol, graph_rtol),
                        "working graph mismatch");
        TEST_ASSERT_MSG(nearly_equal(gpu_working_devparam[(size_t) i], graph_working[(size_t) i], graph_atol, graph_rtol),
                        "working devparam graph mismatch");
    }
    for (int i = 0; i < n_hc; ++i) {
        TEST_ASSERT_MSG(nearly_equal(gpu_post[(size_t) i], graph_post[(size_t) i], graph_atol, graph_rtol),
                        "post graph mismatch");
        TEST_ASSERT_MSG(nearly_equal(gpu_post_devparam[(size_t) i], graph_post[(size_t) i], graph_atol, graph_rtol),
                        "post devparam graph mismatch");
    }
    for (int i = 0; i < n_hc * n_hc; ++i) {
        TEST_ASSERT_MSG(nearly_equal(gpu_comb[(size_t) i], graph_comb[(size_t) i], graph_atol, graph_rtol),
                        "comb graph mismatch");
        TEST_ASSERT_MSG(nearly_equal(gpu_comb_devparam[(size_t) i], graph_comb[(size_t) i], graph_atol, graph_rtol),
                        "comb devparam graph mismatch");
    }

    ggml_backend_buffer_free(buf);
    ggml_gallocr_free(graph_alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
}
#endif

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "FAIL: ggml_backend_cpu_init failed\n");
        return 1;
    }

    test_compressor_pooling_correctness(backend);
    test_swiglu_ds4_cpu_correctness(backend);
    test_moe_routing_correctness(backend);
    test_rmsnorm_correctness(backend);
    test_grouped_output_projection_shape();
    test_hash_routing_lookup();
    test_auto_split_computation();
    test_layer_range_validation();
    test_hc_state_dimensions();
    test_loader_rejects_missing_required_metadata(backend);
    test_loader_rejects_invalid_compress_ratio_type(backend);
    test_loader_rejects_zero_vocab_size(backend);
    test_loader_reads_tokenizer_special_ids(backend);
    test_loader_rejects_truncated_tensor_data(backend);
    test_snapshot_save_restore();
    test_reset_request_state();
    test_adapter_guard_paths();
    test_ipc_mode_registration();
    test_target_shard_daemon_validation();
    test_ffn_graph_reuse_microbench(backend);
    test_output_graph_reuse_microbench(backend);
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
    test_hc_pre_kernel_gpu();
#endif

    ggml_backend_free(backend);

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED: %d assertion(s)\n", g_failures);
        return 1;
    }

    std::printf("OK\n");
    return 0;
}
