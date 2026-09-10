#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../deps/llama.cpp/ggml/src/ggml-backend-impl.h"
#include "deepseek4/deepseek4_image_admission.h"
#include "ggml-cpu.h"
#if defined(GGML_USE_HIP)
#include <hip/hip_runtime_api.h>
#endif
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
#include "ggml-cuda.h"
#include "common/cuda_graph_overrides.h"
#include "deepseek4/deepseek4_hc_cuda.h"
#endif

#include "common/backend_ipc.h"
#include "common/dspark_head.h"
#include "common/layer_split_backend.h"
#include "server/disk_prefix_cache.h"
#include "deepseek4/deepseek4_snapshot.h"
#include "common/layer_split_runtime.h"
#include "common/layer_split_utils.h"
#include "common/moe_hybrid_ffn_eval.h"
#include "deepseek4/deepseek4_dspark.h"

#include <filesystem>
#include <memory>
#include <random>
#include <sstream>
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
#include <thread>
#include <vector>
#include <unistd.h>

#define private public
#include "deepseek4/deepseek4_backend.h"
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

static void test_moe_expert_major_default_threshold() {
    TEST_ASSERT(kMoeExpertMajorPrefillMinTokens == 64);
    TEST_ASSERT(!moe_expert_major_prefill_policy_enabled(
        63, true, kMoeExpertMajorPrefillMinTokens));
    TEST_ASSERT(moe_expert_major_prefill_policy_enabled(
        64, true, kMoeExpertMajorPrefillMinTokens));
    TEST_ASSERT(!moe_expert_major_prefill_policy_enabled(
        64, false, kMoeExpertMajorPrefillMinTokens));
    TEST_ASSERT(moe_cold_input_first_policy_enabled(true, true, false));
    TEST_ASSERT(!moe_cold_input_first_policy_enabled(true, true, true));
}

struct DeepSeek4FixtureOptions {
    bool include_vocab_size = true;
    uint32_t vocab_size = 128;
    bool write_compress_ratios = false;
    gguf_type compress_ratios_type = GGUF_TYPE_UINT32;
    int32_t eos_id = -1;
    int32_t eot_id = -1;
    uint32_t block_count = 43;
    bool image_biases = false;
    int missing_image_bias = -1;
    int malformed_image_bias = -1;
    ggml_type image_bias_type = GGML_TYPE_F32;
    int image_bias_width = 256;
    int image_bias_rows = 1;
    bool add_mtp_image_bias = false;
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
    gguf_set_val_u32(g, "deepseek4.block_count", opts.block_count);
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

    ggml_context * tensor_ctx = nullptr;
    if (opts.image_biases) {
        tensor_ctx = ggml_init({1u << 20, nullptr, false});
        for (int layer = 0; layer < (opts.add_mtp_image_bias ? 44 : 43); ++layer) {
            if (layer == opts.missing_image_bias) continue;
            const bool malformed = layer == opts.malformed_image_bias;
            ggml_tensor * bias = ggml_new_tensor_2d(tensor_ctx,
                malformed ? opts.image_bias_type : GGML_TYPE_F32,
                malformed ? opts.image_bias_width : 256,
                malformed ? opts.image_bias_rows : 1);
            const std::string name = "layers." + std::to_string(layer) + ".ffn.gate.bias_vl";
            ggml_set_name(bias, name.c_str());
            std::memset(bias->data, 0, ggml_nbytes(bias));
            if (bias->type == GGML_TYPE_F32) {
                std::fill_n(static_cast<float *>(bias->data), ggml_nelements(bias), float(layer + 1));
            }
            gguf_add_tensor(g, bias);
        }
    }
    const std::string path = make_temp_gguf_path("fixture");
    gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
    gguf_free(g);
    if (tensor_ctx) ggml_free(tensor_ctx);
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

struct DSparkFixtureOptions {
    bool wrong_main_norm_shape = false;
    bool add_unknown_tensor = false;
};

static std::string write_dspark_loader_fixture(const DSparkFixtureOptions & opts = {}) {
    ggml_init_params ip{};
    ip.mem_size = 4u << 20;
    ip.mem_buffer = nullptr;
    ip.no_alloc = false;
    ggml_context * ctx = ggml_init(ip);
    gguf_context * g = gguf_init_empty();

    const char * arch = "deepseek4-dflash-draft";
    const std::string p = std::string(arch) + ".";
    gguf_set_val_str(g, "general.architecture", arch);
    gguf_set_val_u32(g, (p + "block_count").c_str(), 1);
    gguf_set_val_u32(g, (p + "embedding_length").c_str(), 4);
    gguf_set_val_u32(g, (p + "vocab_size").c_str(), 8);
    gguf_set_val_u32(g, (p + "attention.head_count").c_str(), 1);
    gguf_set_val_u32(g, (p + "attention.head_count_kv").c_str(), 1);
    gguf_set_val_u32(g, (p + "attention.key_length").c_str(), 4);
    gguf_set_val_u32(g, (p + "rope.dimension_count").c_str(), 4);
    gguf_set_val_u32(g, (p + "attention.q_lora_rank").c_str(), 4);
    gguf_set_val_u32(g, (p + "attention.output_lora_rank").c_str(), 4);
    gguf_set_val_u32(g, (p + "attention.output_group_count").c_str(), 1);
    gguf_set_val_u32(g, (p + "expert_count").c_str(), 2);
    gguf_set_val_u32(g, (p + "expert_used_count").c_str(), 1);
    gguf_set_val_u32(g, (p + "expert_shared_count").c_str(), 1);
    gguf_set_val_u32(g, (p + "expert_feed_forward_length").c_str(), 8);
    gguf_set_val_u32(g, (p + "hash_layer_count").c_str(), 0);
    gguf_set_val_u32(g, (p + "attention.sliding_window").c_str(), 8);
    gguf_set_val_u32(g, (p + "attention.indexer.head_count").c_str(), 1);
    gguf_set_val_u32(g, (p + "attention.indexer.key_length").c_str(), 4);
    gguf_set_val_u32(g, (p + "attention.indexer.top_k").c_str(), 1);
    gguf_set_val_u32(g, (p + "hyper_connection.count").c_str(), 1);
    gguf_set_val_u32(g, (p + "hyper_connection.sinkhorn_iterations").c_str(), 1);
    gguf_set_val_u32(g, (p + "dflash.n_target_layers").c_str(), 1);
    gguf_set_val_u32(g, (p + "dflash.block_size").c_str(), 2);
    gguf_set_val_u32(g, (p + "dflash.mask_token_id").c_str(), 7);
    gguf_set_val_u32(g, (p + "dflash.head_hc_enabled").c_str(), 1);
    gguf_set_val_u32(g, (p + "dflash.dspark.enabled").c_str(), 1);
    gguf_set_val_u32(g, (p + "dflash.dspark.markov_rank").c_str(), 2);
    gguf_set_val_u32(g, (p + "dflash.dspark.vocab_size").c_str(), 8);
    const int32_t capture_ids[] = {0};
    gguf_set_arr_data(g, (p + "dflash.capture_layer_ids").c_str(),
                      GGUF_TYPE_INT32, capture_ids, 1);

    auto add1 = [&](const char * name, int64_t n0) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n0);
        ggml_set_name(t, name);
        std::memset(t->data, 0, ggml_nbytes(t));
        gguf_add_tensor(g, t);
    };
    auto add2 = [&](const char * name, int64_t n0, int64_t n1) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1);
        ggml_set_name(t, name);
        std::memset(t->data, 0, ggml_nbytes(t));
        gguf_add_tensor(g, t);
    };
    auto add3 = [&](const char * name, int64_t n0, int64_t n1, int64_t n2) {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n0, n1, n2);
        ggml_set_name(t, name);
        std::memset(t->data, 0, ggml_nbytes(t));
        gguf_add_tensor(g, t);
    };

    add1("output_norm.weight", 4);
    add2("output_hc_fn.weight", 4, 1);
    add1("output_hc_scale.weight", 1);
    add1("output_hc_base.weight", 1);
    add2("dflash.fc.weight", 4, 4);
    add1("dflash.hidden_norm.weight", opts.wrong_main_norm_shape ? 3 : 4);
    add2("dflash.dspark.markov.w1", 2, 8);
    add2("dflash.dspark.markov.w2", 2, 8);

    add1("blk.0.attn_norm.weight", 4);
    add2("blk.0.attn_q_a.weight", 4, 4);
    add1("blk.0.attn_q_a_norm.weight", 4);
    add2("blk.0.attn_q_b.weight", 4, 4);
    add2("blk.0.attn_kv.weight", 4, 4);
    add1("blk.0.attn_kv_a_norm.weight", 4);
    add2("blk.0.attn_output_a.weight", 4, 4);
    add2("blk.0.attn_output_b.weight", 4, 4);
    add2("blk.0.hc_attn_fn.weight", 4, 3);
    add1("blk.0.hc_attn_scale.weight", 3);
    add1("blk.0.hc_attn_base.weight", 3);
    add1("blk.0.ffn_norm.weight", 4);
    add2("blk.0.ffn_gate_inp.weight", 4, 2);
    add3("blk.0.ffn_gate_exps.weight", 4, 8, 2);
    add3("blk.0.ffn_up_exps.weight", 4, 8, 2);
    add3("blk.0.ffn_down_exps.weight", 8, 4, 2);
    add2("blk.0.ffn_gate_shexp.weight", 4, 8);
    add2("blk.0.ffn_up_shexp.weight", 4, 8);
    add2("blk.0.ffn_down_shexp.weight", 8, 4);
    add2("blk.0.hc_ffn_fn.weight", 4, 3);
    add1("blk.0.hc_ffn_scale.weight", 3);
    add1("blk.0.hc_ffn_base.weight", 3);
    if (opts.add_unknown_tensor) add2("token_embd.weight", 4, 8);

    const std::string path = make_temp_gguf_path("dspark");
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

static void test_chunked_graph_allocator(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_chunked_graph_allocator ...");
    ggml_context * ctx = make_test_context();
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) {
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    // The input and output must coexist. Each fits under the cap, while their
    // combined live range does not, so the allocator must create two chunks.
    constexpr int64_t n_elements = 96 * 1024; // 384 KiB of F32
    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    ggml_set_input(input);
    ggml_tensor * output = ggml_dup(ctx, input);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output);

    constexpr size_t max_chunk_size = 512u * 1024u;
    ggml_gallocr_t alloc = ggml_gallocr_new_with_max_chunk_size(
        ggml_backend_get_default_buffer_type(backend), max_chunk_size);
    TEST_ASSERT_MSG(alloc != nullptr, "chunked graph allocator creation failed");
    if (alloc) {
        TEST_ASSERT_MSG(ggml_gallocr_alloc_graph(alloc, graph),
                        "chunked graph allocation failed");
        TEST_ASSERT_MSG(ggml_gallocr_get_buffer_n_chunks(alloc, 0) == 2,
                        "graph allocator did not split the backing buffer");
        ggml_gallocr_free(alloc);
    }
    ggml_free(ctx);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_dspark_confidence_uses_separate_hidden(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_dspark_confidence_uses_separate_hidden ...");

    constexpr int hidden = 2;
    constexpr int rank = 1;
    constexpr int vocab = 3;
    constexpr int q_len = 2;  // dummy seed row + one candidate row

    ggml_context * ctx = make_test_context();
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) {
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    ggml_tensor * lm_head = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, vocab);
    ggml_tensor * markov_w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, vocab);
    ggml_tensor * markov_w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, vocab);
    ggml_tensor * confidence_w =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden + rank, 1);
    ggml_tensor * confidence_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    TEST_ASSERT_MSG(buf != nullptr, "weight allocation failed");
    if (!buf) {
        ggml_free(ctx);
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    const std::vector<float> zeros_lm((size_t) hidden * vocab, 0.0f);
    const std::vector<float> zeros_markov((size_t) rank * vocab, 0.0f);
    const std::vector<float> confidence_weight = {1.0f, 0.0f, 0.0f};
    const float zero = 0.0f;
    ggml_backend_tensor_set(lm_head, zeros_lm.data(), 0,
                            zeros_lm.size() * sizeof(float));
    ggml_backend_tensor_set(markov_w1, zeros_markov.data(), 0,
                            zeros_markov.size() * sizeof(float));
    ggml_backend_tensor_set(markov_w2, zeros_markov.data(), 0,
                            zeros_markov.size() * sizeof(float));
    ggml_backend_tensor_set(confidence_w, confidence_weight.data(), 0,
                            confidence_weight.size() * sizeof(float));
    ggml_backend_tensor_set(confidence_b, &zero, 0, sizeof(zero));

    DraftWeights dw{};
    dw.n_embd = hidden;
    dw.dspark.enabled = true;
    dw.dspark.markov_rank = rank;
    dw.dspark.vocab_size = vocab;
    dw.dspark.confidence_dim = hidden + rank;
    dw.dspark.markov_w1 = markov_w1;
    dw.dspark.markov_w2 = markov_w2;
    dw.dspark.confidence_w = confidence_w;
    dw.dspark.confidence_b = confidence_b;

    // Both calls use the same normalized candidate hidden (all zero), so token
    // logits and Markov correction are identical. Only the reference-faithful
    // confidence input differs: its candidate row starts with 2.0.
    const std::vector<float> normalized_hidden((size_t) hidden * q_len, 0.0f);
    const std::vector<float> confidence_hidden = {0.0f, 0.0f, 2.0f, -3.0f};
    std::vector<int32_t> separate_tokens;
    std::vector<float> separate_confidence;
    const bool separate_ok = dspark_markov_correct_greedy_chain_fused(
        dw, backend, lm_head, normalized_hidden.data(), q_len, 0,
        separate_tokens, &separate_confidence, confidence_hidden.data());

    std::vector<int32_t> legacy_tokens;
    std::vector<float> legacy_confidence;
    const bool legacy_ok = dspark_markov_correct_greedy_chain_fused(
        dw, backend, lm_head, normalized_hidden.data(), q_len, 0,
        legacy_tokens, &legacy_confidence);

    TEST_ASSERT_MSG(separate_ok, "separate-confidence fused graph failed");
    TEST_ASSERT_MSG(legacy_ok, "legacy-confidence fused graph failed");
    TEST_ASSERT(separate_tokens == legacy_tokens);
    TEST_ASSERT(separate_confidence.size() == 1);
    TEST_ASSERT(legacy_confidence.size() == 1);
    if (separate_confidence.size() == 1 && legacy_confidence.size() == 1) {
        const float expected_separate = 1.0f / (1.0f + std::exp(-2.0f));
        TEST_ASSERT_MSG(nearly_equal(separate_confidence[0], expected_separate),
                        "confidence did not use the separate pre-norm hidden");
        TEST_ASSERT_MSG(nearly_equal(legacy_confidence[0], 0.5f),
                        "legacy confidence fallback changed");
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
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

static void test_top6_head4_tail2_route_slices(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_top6_head4_tail2_route_slices ...");

    constexpr int route_width = 6;
    constexpr int n_tokens = 4;
    std::vector<int32_t> selected_values(route_width * n_tokens);
    std::vector<float> weight_values(route_width * n_tokens);
    std::vector<int32_t> expected_head_ids;
    std::vector<int32_t> expected_tail_ids;
    std::vector<float> expected_head_weights;
    std::vector<float> expected_tail_weights;
    for (int token = 0; token < n_tokens; ++token) {
        for (int route = 0; route < route_width; ++route) {
            const size_t index = (size_t) token * route_width + route;
            selected_values[index] = token * 10 + route;
            weight_values[index] = (float) (token * 10 + route) + 0.25f;
            if (route < 4) {
                expected_head_ids.push_back(selected_values[index]);
                expected_head_weights.push_back(weight_values[index]);
            } else {
                expected_tail_ids.push_back(selected_values[index]);
                expected_tail_weights.push_back(weight_values[index]);
            }
        }
    }

    ggml_context * ctx = make_test_context();
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) {
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    ggml_tensor * selected = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, route_width, n_tokens);
    ggml_tensor * weights = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, route_width, n_tokens);
    ggml_set_input(selected);
    ggml_set_input(weights);

    DeepSeek4Head4Tail2Routes routes;
    TEST_ASSERT(build_deepseek4_head4_tail2_routes(
        ctx, selected, weights, n_tokens, routes));
    TEST_ASSERT(routes.head_ids && routes.head_weights &&
                routes.tail_ids && routes.tail_weights);
    if (!routes.head_ids || !routes.head_weights ||
        !routes.tail_ids || !routes.tail_weights) {
        ggml_free(ctx);
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    TEST_ASSERT(routes.head_ids->ne[0] == 4);
    TEST_ASSERT(routes.head_ids->ne[1] == n_tokens);
    TEST_ASSERT(routes.head_weights->ne[0] == 4);
    TEST_ASSERT(routes.head_weights->ne[1] == n_tokens);
    TEST_ASSERT(routes.tail_ids->ne[0] == 2);
    TEST_ASSERT(routes.tail_ids->ne[1] == n_tokens);
    TEST_ASSERT(routes.tail_weights->ne[0] == 2);
    TEST_ASSERT(routes.tail_weights->ne[1] == n_tokens);

    const ggml_tensor * outputs[] = {
        routes.head_ids,
        routes.head_weights,
        routes.tail_ids,
        routes.tail_weights,
    };
    const ggml_tensor * sources[] = {
        selected,
        weights,
        selected,
        weights,
    };
    for (size_t i = 0; i < 4; ++i) {
        TEST_ASSERT(outputs[i]->op == GGML_OP_CONT);
        TEST_ASSERT(outputs[i]->src[0] != nullptr);
        if (outputs[i]->src[0]) {
            TEST_ASSERT(outputs[i]->src[0]->view_src == sources[i]);
            TEST_ASSERT(outputs[i]->src[0]->nb[1] == sources[i]->nb[1]);
        }
        ggml_set_output(const_cast<ggml_tensor *>(outputs[i]));
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    for (ggml_tensor * output : {
             routes.head_ids,
             routes.head_weights,
             routes.tail_ids,
             routes.tail_weights}) {
        ggml_build_forward_expand(graph, output);
    }
    int cont_nodes = 0;
    int concat_nodes = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        cont_nodes += node->op == GGML_OP_CONT;
        concat_nodes += node->op == GGML_OP_CONCAT;
    }
    TEST_ASSERT(cont_nodes == 4);
    TEST_ASSERT(concat_nodes == 0);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    TEST_ASSERT(ggml_gallocr_alloc_graph(alloc, graph));
    ggml_backend_tensor_set(selected, selected_values.data(), 0,
                            selected_values.size() * sizeof(int32_t));
    ggml_backend_tensor_set(weights, weight_values.data(), 0,
                            weight_values.size() * sizeof(float));
    TEST_ASSERT(ggml_backend_graph_compute(backend, graph) ==
                GGML_STATUS_SUCCESS);

    std::vector<int32_t> actual_head_ids(expected_head_ids.size());
    std::vector<int32_t> actual_tail_ids(expected_tail_ids.size());
    std::vector<float> actual_head_weights(expected_head_weights.size());
    std::vector<float> actual_tail_weights(expected_tail_weights.size());
    ggml_backend_tensor_get(routes.head_ids, actual_head_ids.data(), 0,
                            actual_head_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_get(routes.tail_ids, actual_tail_ids.data(), 0,
                            actual_tail_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_get(routes.head_weights,
                            actual_head_weights.data(), 0,
                            actual_head_weights.size() * sizeof(float));
    ggml_backend_tensor_get(routes.tail_weights,
                            actual_tail_weights.data(), 0,
                            actual_tail_weights.size() * sizeof(float));
    TEST_ASSERT(actual_head_ids == expected_head_ids);
    TEST_ASSERT(actual_tail_ids == expected_tail_ids);
    TEST_ASSERT(actual_head_weights == expected_head_weights);
    TEST_ASSERT(actual_tail_weights == expected_tail_weights);

    DeepSeek4Head4Tail2Routes invalid;
    ggml_tensor * five_routes = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, 5, n_tokens);
    ggml_tensor * integer_weights = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, route_width, n_tokens);
    TEST_ASSERT(!build_deepseek4_head4_tail2_routes(
        ctx, five_routes, weights, n_tokens, invalid));
    TEST_ASSERT(!build_deepseek4_head4_tail2_routes(
        ctx, selected, integer_weights, n_tokens, invalid));
    TEST_ASSERT(!build_deepseek4_head4_tail2_routes(
        ctx, selected, weights, 0, invalid));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
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

static void test_grouped_output_projection_cpu(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_grouped_output_projection_cpu ...");

    constexpr int group_width = 4;
    constexpr int n_groups = 3;
    constexpr int n_tokens = 5;
    constexpr int n_outputs = 4;
    constexpr int flat_width = group_width * n_groups;

    std::vector<float> weights((size_t) flat_width * n_outputs);
    std::vector<float> grouped((size_t) group_width * n_tokens * n_groups);
    std::vector<float> expected((size_t) n_outputs * n_tokens, 0.0f);
    for (int output = 0; output < n_outputs; ++output) {
        for (int k = 0; k < flat_width; ++k) {
            weights[(size_t) output * flat_width + k] =
                0.03125f * (float) ((output + 1) * (k - 5));
        }
    }
    for (int group = 0; group < n_groups; ++group) {
        for (int token = 0; token < n_tokens; ++token) {
            for (int k = 0; k < group_width; ++k) {
                grouped[(size_t) group * n_tokens * group_width +
                        (size_t) token * group_width + k] =
                    10.0f * (float) group + 0.5f * (float) token +
                    0.125f * (float) k;
            }
        }
    }
    for (int token = 0; token < n_tokens; ++token) {
        for (int output = 0; output < n_outputs; ++output) {
            float sum = 0.0f;
            for (int group = 0; group < n_groups; ++group) {
                for (int k = 0; k < group_width; ++k) {
                    const float value =
                        grouped[(size_t) group * n_tokens * group_width +
                                (size_t) token * group_width + k];
                    sum += weights[(size_t) output * flat_width +
                                   group * group_width + k] * value;
                }
            }
            expected[(size_t) token * n_outputs + output] = sum;
        }
    }

    ggml_context * ctx = make_test_context();
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) return;
    ggml_tensor * weights_t =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, flat_width, n_outputs);
    ggml_tensor * grouped_t = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, group_width, n_tokens, n_groups);
    ggml_set_input(weights_t);
    ggml_set_input(grouped_t);
    ggml_tensor * output_t =
        ggml_mul_mat_grouped_src(ctx, weights_t, grouped_t);
    TEST_ASSERT_MSG(output_t->op == GGML_OP_MUL_MAT_GROUPED_SRC,
                    "grouped projection must use a distinct backend contract");
    ggml_set_output(output_t);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output_t);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    TEST_ASSERT_MSG(ggml_gallocr_alloc_graph(alloc, graph),
                    "grouped projection graph allocation failed");
    ggml_backend_tensor_set(weights_t, weights.data(), 0,
                            weights.size() * sizeof(float));
    ggml_backend_tensor_set(grouped_t, grouped.data(), 0,
                            grouped.size() * sizeof(float));
    TEST_ASSERT_MSG(ggml_backend_graph_compute(backend, graph) ==
                        GGML_STATUS_SUCCESS,
                    "grouped projection graph compute failed");
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output_t, actual.data(), 0,
                            actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        TEST_ASSERT_MSG(nearly_equal(actual[i], expected[i]),
                        "grouped projection output mismatch");
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_ds4_flash_attention_cpu_rejected(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_ds4_flash_attention_cpu_rejected ...");
    ggml_context * ctx = make_test_context();
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) return;
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 512, 2, 4);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 512, 8, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 2);
    ggml_tensor * op = ggml_flash_attn_ext(
        ctx, q, k, k, mask, 1.0f / std::sqrt(512.0f), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_ds4_sparse(op, 4, 4, 0, 4);
    TEST_ASSERT(ggml_flash_attn_ext_is_ds4(op));
    TEST_ASSERT_MSG(!ggml_backend_supports_op(backend, op),
                    "CPU accepted the DS4 flash-attention contract");
    ggml_free(ctx);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static float reference_e2m1_round(float value) {
    static constexpr float levels[] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    const float sign = value < 0.0f ? -1.0f : 1.0f;
    const float magnitude = std::min(std::fabs(value), 6.0f);
    int best = 0;
    float best_diff = std::fabs(magnitude - levels[0]);
    for (int i = 1; i < 8; ++i) {
        const float diff = std::fabs(magnitude - levels[i]);
        if (diff < best_diff ||
            (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * levels[best];
}

static void reference_ds4_indexer_qat(float * row) {
    for (int stride = 1; stride < 128; stride <<= 1) {
        for (int base = 0; base < 128; base += 2 * stride) {
            for (int i = 0; i < stride; ++i) {
                const float a = row[base + i];
                const float b = row[base + stride + i];
                row[base + i] = a + b;
                row[base + stride + i] = a - b;
            }
        }
    }
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; ++i) row[i] *= inv_sqrt_128;
    for (int block = 0; block < 4; ++block) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            amax = std::max(amax, std::fabs(row[block * 32 + i]));
        }
        amax = std::max(amax, 7.052966104933725e-38f);
        const float scale = std::exp2(std::ceil(std::log2(amax / 6.0f)));
        for (int i = 0; i < 32; ++i) {
            const int index = block * 32 + i;
            const float normalized =
                std::clamp(row[index] / scale, -6.0f, 6.0f);
            row[index] = reference_e2m1_round(normalized) * scale;
        }
    }
}

static void test_indexer_qat_cpu(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_indexer_qat_cpu ...");
    constexpr int n_head = 3;
    constexpr int n_tokens = 2;
    std::vector<float> input((size_t) 128 * n_head * n_tokens);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = 0.35f * std::sin(0.17f * (float) i) +
                   0.08f * (float) ((int) (i % 11) - 5);
    }
    std::vector<float> expected = input;
    for (int row = 0; row < n_head * n_tokens; ++row) {
        reference_ds4_indexer_qat(expected.data() + (size_t) row * 128);
    }
    ggml_context * ctx = make_test_context();
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) return;
    ggml_tensor * input_t = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, 128, n_head, n_tokens);
    ggml_set_input(input_t);
    ggml_tensor * output_t = ggml_ds4_indexer_qat(ctx, input_t);
    ggml_set_output(output_t);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output_t);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    TEST_ASSERT_MSG(ggml_gallocr_alloc_graph(alloc, graph),
                    "indexer QAT graph allocation failed");
    ggml_backend_tensor_set(input_t, input.data(), 0,
                            input.size() * sizeof(float));
    TEST_ASSERT_MSG(ggml_backend_graph_compute(backend, graph) ==
                        GGML_STATUS_SUCCESS,
                    "indexer QAT graph compute failed");
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output_t, actual.data(), 0,
                            actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        TEST_ASSERT_MSG(nearly_equal(actual[i], expected[i], 1e-6f, 1e-6f),
                        "indexer QAT output mismatch");
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_indexer_score_cpu(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_indexer_score_cpu ...");
    constexpr int n_head = 3;
    constexpr int n_tokens = 6;
    constexpr int n_comp = 4;
    constexpr int kv_start = 2;
    constexpr int ratio = 4;
    std::vector<float> q((size_t) 128 * n_head * n_tokens);
    std::vector<float> weights((size_t) n_head * n_tokens);
    std::vector<float> comp_f32((size_t) 128 * n_comp);
    std::vector<ggml_fp16_t> comp_f16(comp_f32.size());
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = 0.025f * (float) ((int) (i % 29) - 14);
    }
    for (int token = 0; token < n_tokens; ++token) {
        for (int head = 0; head < n_head; ++head) {
            weights[(size_t) token * n_head + head] =
                0.25f + 0.1f * (float) head + 0.03f * (float) token;
        }
    }
    for (int comp = 0; comp < n_comp; ++comp) {
        for (int d = 0; d < 128; ++d) {
            comp_f32[(size_t) comp * 128 + d] =
                0.04f * (float) (((comp + 2) * (d + 3)) % 23 - 11);
        }
    }
    ggml_fp32_to_fp16_row(comp_f32.data(), comp_f16.data(),
                          (int64_t) comp_f32.size());
    std::vector<float> expected((size_t) n_comp * n_tokens);
    std::vector<float> visibility_mask((size_t) n_comp * n_tokens, 0.0f);
    std::vector<float> masked_expected((size_t) n_comp * n_tokens);
    for (int token = 0; token < n_tokens; ++token) {
        const int visible = (kv_start + token + 1) / ratio;
        for (int comp = 0; comp < n_comp; ++comp) {
            float score = 0.0f;
            for (int head = 0; head < n_head; ++head) {
                const float * q_row = q.data() +
                    ((size_t) token * n_head + head) * 128;
                const ggml_fp16_t * k_row =
                    comp_f16.data() + (size_t) comp * 128;
                float dot = 0.0f;
                for (int d = 0; d < 128; ++d) {
                    dot += q_row[d] * ggml_fp16_to_fp32(k_row[d]);
                }
                score += std::max(dot, 0.0f) *
                    weights[(size_t) token * n_head + head];
            }
            const size_t index = (size_t) token * n_comp + comp;
            expected[index] = comp < visible ? score : -1.0e30f;
            if ((token + 2 * comp) % 5 == 0) {
                visibility_mask[index] = -1.0e30f;
            }
            masked_expected[index] = visibility_mask[index] <= -1.0e20f
                ? -1.0e30f : score;
        }
    }

    ggml_context * ctx = make_test_context();
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) return;
    ggml_tensor * q_t = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, 128, n_head, n_tokens);
    ggml_tensor * weights_t = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_head, n_tokens);
    ggml_tensor * comp_t = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F16, 128, n_comp);
    ggml_tensor * visibility_mask_t = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_comp, n_tokens);
    ggml_set_input(q_t);
    ggml_set_input(weights_t);
    ggml_set_input(comp_t);
    ggml_set_input(visibility_mask_t);
    ggml_tensor * scores_t = ggml_ds4_indexer_score(
        ctx, q_t, weights_t, comp_t, kv_start, ratio);
    ggml_tensor * masked_scores_t = ggml_ds4_indexer_score_masked(
        ctx, q_t, weights_t, comp_t, visibility_mask_t, kv_start, ratio);
    ggml_set_output(scores_t);
    ggml_set_output(masked_scores_t);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, scores_t);
    ggml_build_forward_expand(graph, masked_scores_t);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    TEST_ASSERT_MSG(ggml_gallocr_alloc_graph(alloc, graph),
                    "indexer score graph allocation failed");
    ggml_backend_tensor_set(q_t, q.data(), 0, q.size() * sizeof(float));
    ggml_backend_tensor_set(weights_t, weights.data(), 0,
                            weights.size() * sizeof(float));
    ggml_backend_tensor_set(comp_t, comp_f16.data(), 0,
                            comp_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(visibility_mask_t, visibility_mask.data(), 0,
                            visibility_mask.size() * sizeof(float));
    TEST_ASSERT_MSG(ggml_backend_graph_compute(backend, graph) ==
                        GGML_STATUS_SUCCESS,
                    "indexer score graph compute failed");
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(scores_t, actual.data(), 0,
                            actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        TEST_ASSERT_MSG(nearly_equal(actual[i], expected[i], 2e-5f, 2e-5f),
                        "indexer score output mismatch");
    }
    std::vector<float> masked_actual(masked_expected.size());
    ggml_backend_tensor_get(masked_scores_t, masked_actual.data(), 0,
                            masked_actual.size() * sizeof(float));
    for (size_t i = 0; i < masked_actual.size(); ++i) {
        TEST_ASSERT_MSG(
            nearly_equal(masked_actual[i], masked_expected[i], 2e-5f, 2e-5f),
            "masked indexer score output mismatch");
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_indexer_mask_cpu(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_indexer_mask_cpu ...");
    constexpr int raw_rows = 3;
    constexpr int n_comp = 5;
    constexpr int n_tokens = 3;
    constexpr int top_k = 3;
    constexpr int n_attn = raw_rows + n_comp;
    std::vector<float> base((size_t) n_attn * n_tokens);
    for (int token = 0; token < n_tokens; ++token) {
        for (int row = 0; row < n_attn; ++row) {
            base[(size_t) token * n_attn + row] =
                100.0f * (float) token + (float) row + 0.25f;
        }
    }
    const std::vector<int32_t> selected = {
        4, 1, 1,
        0, -1, 3,
        2, 5, 4,
    };
    std::vector<float> expected((size_t) n_attn * n_tokens, -1.0e30f);
    for (int token = 0; token < n_tokens; ++token) {
        std::copy_n(base.data() + (size_t) token * n_attn, raw_rows,
                    expected.data() + (size_t) token * n_attn);
        for (int k = 0; k < top_k; ++k) {
            const int comp = selected[(size_t) token * top_k + k];
            if (comp >= 0 && comp < n_comp) {
                expected[(size_t) token * n_attn + raw_rows + comp] =
                    base[(size_t) token * n_attn + raw_rows + comp];
            }
        }
    }
    ggml_context * ctx = make_test_context();
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) return;
    ggml_tensor * base_t = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_attn, n_tokens);
    ggml_tensor * selected_t = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, top_k, n_tokens);
    ggml_set_input(base_t);
    ggml_set_input(selected_t);
    ggml_tensor * mask_t = ggml_ds4_indexer_mask(
        ctx, base_t, selected_t, raw_rows);
    ggml_set_output(mask_t);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, mask_t);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    TEST_ASSERT_MSG(ggml_gallocr_alloc_graph(alloc, graph),
                    "indexer mask graph allocation failed");
    ggml_backend_tensor_set(base_t, base.data(), 0,
                            base.size() * sizeof(float));
    ggml_backend_tensor_set(selected_t, selected.data(), 0,
                            selected.size() * sizeof(int32_t));
    TEST_ASSERT_MSG(ggml_backend_graph_compute(backend, graph) ==
                        GGML_STATUS_SUCCESS,
                    "indexer mask graph compute failed");
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(mask_t, actual.data(), 0,
                            actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        TEST_ASSERT_MSG(actual[i] == expected[i],
                        "indexer mask output mismatch");
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
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

static void test_raw_ring_spans_after_wrap() {
    std::fprintf(stderr, "  test_raw_ring_spans_after_wrap ...");

    DeepSeek4RawRingSpan spans[2];
    int count = deepseek4_previous_raw_ring_spans(3, 8, spans);
    TEST_ASSERT(count == 1);
    TEST_ASSERT(spans[0].row == 0);
    TEST_ASSERT(spans[0].count == 3);

    count = deepseek4_previous_raw_ring_spans(8, 8, spans);
    TEST_ASSERT(count == 1);
    TEST_ASSERT(spans[0].row == 1);
    TEST_ASSERT(spans[0].count == 7);

    count = deepseek4_previous_raw_ring_spans(10, 8, spans);
    TEST_ASSERT(count == 2);
    TEST_ASSERT(spans[0].row == 3);
    TEST_ASSERT(spans[0].count == 5);
    TEST_ASSERT(spans[1].row == 0);
    TEST_ASSERT(spans[1].count == 2);
    TEST_ASSERT(spans[0].count + spans[1].count == 7);

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

// Metadata-only device: every allocation/graph callback is a test failure.
struct ImageAdmissionFakeOwner {
    ggml_backend_buffer_type buft{};
    ggml_backend_device device{};
    ggml_backend backend{};
    size_t alignment = 128;
    size_t padding = 64;
    size_t maximum = SIZE_MAX;
    size_t forced_allocation = 0;
    size_t free_bytes = 8ULL * 1024 * 1024 * 1024;
    size_t total_bytes = 8ULL * 1024 * 1024 * 1024;
    size_t queries = 0;
    size_t allocation_calls = 0;
    size_t graph_calls = 0;

    ImageAdmissionFakeOwner() {
        buft.context = this;
        buft.device = &device;
        buft.iface.get_name = [](ggml_backend_buffer_type_t) { return "image-admission-fake"; };
        buft.iface.get_alignment = [](ggml_backend_buffer_type_t b) {
            return static_cast<ImageAdmissionFakeOwner *>(b->context)->alignment;
        };
        buft.iface.get_max_size = [](ggml_backend_buffer_type_t b) {
            return static_cast<ImageAdmissionFakeOwner *>(b->context)->maximum;
        };
        buft.iface.get_alloc_size = [](ggml_backend_buffer_type_t b, const ggml_tensor * t) {
            auto & owner = *static_cast<ImageAdmissionFakeOwner *>(b->context);
            ++owner.queries;
            return owner.forced_allocation ? owner.forced_allocation : ggml_nbytes(t) + owner.padding;
        };
        buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t b, size_t) -> ggml_backend_buffer_t {
            ++static_cast<ImageAdmissionFakeOwner *>(b->context)->allocation_calls;
            TEST_ASSERT_MSG(false, "admission must not allocate a device buffer");
            return nullptr;
        };
        device.context = this;
        device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
        device.iface.get_buffer_type = [](ggml_backend_dev_t d) {
            return &static_cast<ImageAdmissionFakeOwner *>(d->context)->buft;
        };
        device.iface.get_memory = [](ggml_backend_dev_t d, size_t * free, size_t * total) {
            const auto & owner = *static_cast<ImageAdmissionFakeOwner *>(d->context);
            *free = owner.free_bytes;
            *total = owner.total_bytes;
        };
        backend.device = &device;
        backend.context = this;
        backend.iface.graph_compute = [](ggml_backend_t b, ggml_cgraph *) {
            ++static_cast<ImageAdmissionFakeOwner *>(b->context)->graph_calls;
            TEST_ASSERT_MSG(false, "admission must not execute a graph");
            return GGML_STATUS_FAILED;
        };
    }
};

static void test_image_storage_admission_metadata() {
    std::fprintf(stderr, "test_image_storage_admission_metadata...");
    using namespace dflash::vision;
    ScopedEnvVar duplicate_env("DFLASH_MOE_DUPLICATE_HOT_ON_COLD");
    ScopedEnvVar decode_env("DFLASH_DS4_DECODE_ALL_COLD");
    unsetenv("DFLASH_MOE_DUPLICATE_HOT_ON_COLD");
    unsetenv("DFLASH_DS4_DECODE_ALL_COLD");
    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr);
    if (!ctx) return;
    DeepSeek4Weights weights;
    weights.n_layer = 1;
    weights.n_expert = 4;
    weights.n_expert_used = 2;
    weights.n_embd = 128;
    weights.n_ff_exp = 128;
    weights.layers.resize(1);
    auto & layer = weights.layers[0];
    layer.ffn_gate_exps = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_1_ROCMFP2_MIX, 128, 128, 4);
    layer.ffn_up_exps = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_1_ROCMFP2_MIX, 128, 128, 4);
    layer.ffn_down_exps = ggml_new_tensor_3d(ctx, GGML_TYPE_Q3_1_ROCMFP3_MIX, 128, 128, 4);
    MoeHybridConfig config;
    config.n_layer = 1;
    config.n_expert = 4;
    config.n_expert_used = 2;
    config.n_embd = 128;
    config.n_ff_exp = 128;
    config.cold_expert_backend = MoeHybridColdBackend::Gpu;
    MoeHybridPlacement placement;
    placement.n_layer = 1;
    placement.n_expert = 4;
    placement.n_expert_used = 2;
    placement.total_hot = 1;
    placement.hot_counts = {1};
    placement.hot_expert_ids = {{2}};
    ImageAdmissionFakeOwner hot, cold;
    cold.alignment = 256;
    cold.padding = 257;
    std::string error;
    ImageStorageEstimate initial;
    TEST_ASSERT(estimate_deepseek4_image_storage(weights, placement, config,
        &hot.backend, &cold.backend, false, initial, error));
    // Pinned qtype-106 and105 formats occupy10 and14 bytes per32 weights.
    // Three128x128 surfaces occupy5120+5120+7168 bytes per expert.
    TEST_ASSERT(initial.hot_payload_bytes == 17408);
    TEST_ASSERT(initial.cold_payload_bytes == 3 * 17408);
    TEST_ASSERT(initial.hot_allocation_bytes == 17408 + 3 * 128);
    TEST_ASSERT(initial.cold_allocation_bytes == 3 * 17408 + 3 * 512);
    TEST_ASSERT(initial.hot_mix_table_bytes == 17 + 17 + 33);
    TEST_ASSERT(initial.cold_mix_table_bytes == 3 * (17 + 17 + 33));
    TEST_ASSERT(initial.mix_device_allocation_count == 12);
    TEST_ASSERT(initial.largest_copy_bytes == 3 * 7168);
    TEST_ASSERT(initial.host_copy_peak_bytes == 3 * initial.largest_copy_bytes);
    TEST_ASSERT(initial.host_mix_payload_peak_bytes == 34 * 4 + 33 * 3 + 4);
    TEST_ASSERT(initial.hot_buffer_count == 1 && initial.cold_buffer_count == 1);
    TEST_ASSERT(hot.queries == 3 && cold.queries == 3);

    // Changing the actual allocator's maximum splits buffers without losing
    // padding charges or pretending the whole layer must fit one allocation.
    hot.maximum = 10000;
    cold.maximum = 23000;
    ImageStorageEstimate split;
    TEST_ASSERT(estimate_deepseek4_image_storage(weights, placement, config,
        &hot.backend, &cold.backend, false, split, error));
    TEST_ASSERT(split.hot_buffer_count == 3 && split.cold_buffer_count == 3);
    TEST_ASSERT(split.hot_allocation_bytes == initial.hot_allocation_bytes);
    TEST_ASSERT(split.cold_allocation_bytes == initial.cold_allocation_bytes);
    cold.maximum = 22015; // one byte below the padded down-expert allocation
    TEST_ASSERT(!estimate_deepseek4_image_storage(weights, placement, config,
        &hot.backend, &cold.backend, false, split, error));
    hot.maximum = cold.maximum = SIZE_MAX;
    cold.forced_allocation = SIZE_MAX;
    TEST_ASSERT(!estimate_deepseek4_image_storage(weights, placement, config,
        &hot.backend, &cold.backend, false, split, error));
    TEST_ASSERT(error.find("allocation size") != std::string::npos);
    cold.forced_allocation = 0;

    setenv("DFLASH_MOE_DUPLICATE_HOT_ON_COLD", "1", 1);
    ImageStorageEstimate duplicated;
    TEST_ASSERT(estimate_deepseek4_image_storage(weights, placement, config,
        &hot.backend, &cold.backend, true, duplicated, error));
    TEST_ASSERT(duplicated.hot_payload_bytes == initial.hot_payload_bytes);
    TEST_ASSERT(duplicated.cold_payload_bytes == initial.hot_payload_bytes + initial.cold_payload_bytes);
    TEST_ASSERT(duplicated.cold_mix_table_bytes == 4 * (17 + 17 + 33));
    TEST_ASSERT(!estimate_deepseek4_image_storage(weights, placement, config,
        &hot.backend, &cold.backend, false, duplicated, error));
    unsetenv("DFLASH_MOE_DUPLICATE_HOT_ON_COLD");
    setenv("DFLASH_DS4_DECODE_ALL_COLD", "1", 1);
    TEST_ASSERT(!estimate_deepseek4_image_storage(weights, placement, config,
        &hot.backend, &cold.backend, false, split, error));
    unsetenv("DFLASH_DS4_DECODE_ALL_COLD");

    placement.total_hot = 0;
    placement.hot_counts = {0};
    placement.hot_expert_ids = {{}};
    TEST_ASSERT(estimate_deepseek4_image_storage(weights, placement, config,
        &hot.backend, &cold.backend, false, split, error));
    TEST_ASSERT(split.hot_payload_bytes == 0 && split.hot_mix_table_bytes == 0);
    TEST_ASSERT(split.cold_payload_bytes == initial.hot_payload_bytes + initial.cold_payload_bytes);
    TEST_ASSERT(split.cold_mix_table_bytes == 4 * 67);
    placement.total_hot = 1;
    placement.hot_counts = {1};
    placement.hot_expert_ids = {{2}};
    const int64_t original_experts = layer.ffn_gate_exps->ne[2];
    layer.ffn_gate_exps->ne[2] = 3;
    TEST_ASSERT(!estimate_deepseek4_image_storage(weights, placement, config,
        &hot.backend, &cold.backend, false, split, error));
    layer.ffn_gate_exps->ne[2] = original_experts;
    const size_t original_stride = layer.ffn_gate_exps->nb[2];
    ++layer.ffn_gate_exps->nb[2];
    TEST_ASSERT(!estimate_deepseek4_image_storage(weights, placement, config,
        &hot.backend, &cold.backend, false, split, error));
    layer.ffn_gate_exps->nb[2] = original_stride;
    TEST_ASSERT(!estimate_deepseek4_image_storage(weights, placement, config,
        &hot.backend, &hot.backend, false, split, error));
    config.materialize_cold_experts = false;
    TEST_ASSERT(!estimate_deepseek4_image_storage(weights, placement, config,
        &hot.backend, &cold.backend, false, split, error));
    config.materialize_cold_experts = true;

    // Actual wrapper sees the fake device's exhausted snapshot. It cannot pass
    // regardless of the machine's MemAvailable; no positive case reads /proc.
    ImageAdmissionReserves reserves;
    reserves.primary_domain = ImageMemoryDomain::Dedicated;
    reserves.cold_domain = ImageMemoryDomain::HostShared;
    reserves.cold_runtime_reservation_bytes = 2ULL * 1024 * 1024 * 1024;
    reserves.host_request_bytes = 1024 * 1024;
    reserves.host_loader_overhead_bytes = 1024 * 1024;
    cold.free_bytes = 0;
    ImageAdmissionReport report;
    TEST_ASSERT(!check_deepseek4_image_admission(weights, placement, config,
        &hot.backend, &cold.backend, reserves, report, error));
    TEST_ASSERT(report.storage_estimated && !report.known_charges_fit);
    TEST_ASSERT(report.cold_free_bytes == 0);
    TEST_ASSERT(!check_deepseek4_image_runtime_admission(config,
        &hot.backend, &cold.backend, reserves, report, error));
    TEST_ASSERT(report.storage.hot_allocation_bytes == 0 && report.storage.cold_allocation_bytes == 0);
    TEST_ASSERT(report.cold_required_bytes == reserves.cold_runtime_reservation_bytes);
    TEST_ASSERT(!check_deepseek4_image_host_preparation(0, error));
    TEST_ASSERT(!check_deepseek4_image_host_preparation(std::numeric_limits<uint64_t>::max(), error));
    TEST_ASSERT(hot.allocation_calls == 0 && cold.allocation_calls == 0);
    TEST_ASSERT(hot.graph_calls == 0 && cold.graph_calls == 0);
    ggml_free(ctx);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_image_admission_resource_snapshots() {
    std::fprintf(stderr, "test_image_admission_resource_snapshots...");
    using namespace dflash::vision;
    ImageStorageEstimate storage;
    storage.hot_allocation_bytes = 100;
    storage.cold_allocation_bytes = 300;
    storage.hot_mix_table_bytes = 10;
    storage.cold_mix_table_bytes = 30;
    storage.host_copy_peak_bytes = 50;
    storage.host_mix_payload_peak_bytes = 5;
    ImageAdmissionReserves reserves;
    reserves.primary_domain = ImageMemoryDomain::Dedicated;
    reserves.cold_domain = ImageMemoryDomain::HostShared;
    reserves.primary_future_bytes = 20;
    reserves.cold_future_bytes = 40;
    reserves.cold_runtime_reservation_bytes = 200;
    reserves.host_loader_overhead_bytes = 15;
    reserves.host_request_bytes = 100;
    reserves.host_runtime_bytes = 7;
    ImageMemorySnapshot snapshot{130, 570, 677};
    ImageAdmissionReport report;
    std::string error;
    const auto assess = [&]() {
        return assess_deepseek4_image_admission(storage, 150, reserves, snapshot, report, error);
    };
    TEST_ASSERT(assess());
    TEST_ASSERT(report.primary_required_bytes == 130 && report.cold_required_bytes == 570);
    TEST_ASSERT(report.host_required_bytes == 677); // max(load70,request100)+runtime7+UMA570
    --snapshot.host_available_bytes;
    TEST_ASSERT(!assess()); // independent device checks would both pass
    TEST_ASSERT(!report.known_charges_fit);
    snapshot.host_available_bytes = 677;
    --snapshot.cold_free_bytes;
    TEST_ASSERT(!assess());
    snapshot.cold_free_bytes = 570;
    --snapshot.primary_free_bytes;
    TEST_ASSERT(!assess());
    snapshot.primary_free_bytes = 130;
    reserves.cold_domain = ImageMemoryDomain::Dedicated;
    snapshot.host_available_bytes = 107;
    TEST_ASSERT(assess()); // dedicated VRAM must not also consume host admission
    reserves.primary_domain = ImageMemoryDomain::HostShared;
    reserves.cold_domain = ImageMemoryDomain::HostShared;
    snapshot.host_available_bytes = 807;
    TEST_ASSERT(assess());
    TEST_ASSERT(report.host_required_bytes == 807);
    --snapshot.host_available_bytes;
    TEST_ASSERT(!assess());
    reserves.primary_domain = ImageMemoryDomain::Unknown;
    TEST_ASSERT(!assess());
    reserves.primary_domain = ImageMemoryDomain::Dedicated;
    snapshot.host_available_bytes = 677;
    reserves.host_request_bytes = 60;
    TEST_ASSERT(assess());
    TEST_ASSERT(report.host_required_bytes == 647); // load70 exceeds request60
    reserves.host_request_bytes = 100;
    reserves.cold_runtime_reservation_bytes = 149;
    TEST_ASSERT(!assess());
    TEST_ASSERT(report.known_charges_fit); // rejection is missing runtime headroom, not physical exhaustion
    reserves.cold_runtime_reservation_bytes = 200;
    reserves.host_request_bytes = 0;
    TEST_ASSERT(!assess());
    reserves.host_request_bytes = 100;
    reserves.host_loader_overhead_bytes = 0;
    TEST_ASSERT(!assess());
    reserves.host_loader_overhead_bytes = 15;

    // After startup materializes the experts/tables, their bytes disappear
    // from free memory and from new allocations together. Retained KV/draft
    // snapshots added later only shrink the next free-memory snapshot.
    const ImageStorageEstimate resident_storage{};
    ImageAdmissionReserves runtime_reserves = reserves;
    runtime_reserves.host_loader_overhead_bytes = 0;
    ImageMemorySnapshot runtime_snapshot{20, 240, 347};
    const auto runtime_assess = [&]() {
        return assess_deepseek4_image_admission(resident_storage, 150,
            runtime_reserves, runtime_snapshot, report, error);
    };
    TEST_ASSERT(runtime_assess());
    TEST_ASSERT(report.primary_required_bytes == 20 && report.cold_required_bytes == 240);
    TEST_ASSERT(report.host_required_bytes == 347);
    --runtime_snapshot.primary_free_bytes;
    TEST_ASSERT(!runtime_assess());
    runtime_snapshot.primary_free_bytes = 20;
    --runtime_snapshot.cold_free_bytes;
    TEST_ASSERT(!runtime_assess());
    runtime_snapshot.cold_free_bytes = 240;
    --runtime_snapshot.host_available_bytes;
    TEST_ASSERT(!runtime_assess());

    constexpr uint64_t max = std::numeric_limits<uint64_t>::max();
    snapshot = {max, max, max};
    storage.hot_allocation_bytes = max;
    TEST_ASSERT(!assess());
    TEST_ASSERT(error.find("overflow") != std::string::npos);
    storage.hot_allocation_bytes = 100;
    storage.host_copy_peak_bytes = max;
    TEST_ASSERT(!assess());
    TEST_ASSERT(error.find("overflow") != std::string::npos);
    storage.host_copy_peak_bytes = 50;
    reserves.host_request_bytes = max - 100;
    TEST_ASSERT(!assess()); // UMA sum overflows even though separate device budgets fit
    TEST_ASSERT(error.find("overflow") != std::string::npos);
    reserves.host_request_bytes = 100;
    TEST_ASSERT(assess());
    TEST_ASSERT(assess_deepseek4_image_admission(report.storage, 150, reserves,
        snapshot, report, error)); // report may safely be reused without losing its estimate
    TEST_ASSERT(report.primary_required_bytes == 130);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_verify_raw_mask_spans() {
    std::fprintf(stderr, "  test_verify_raw_mask_spans ...");
    int cases = 0;
    for (int n_swa : {1, 2, 4, 8, 128}) {
        for (int kv_start = 0; kv_start < 2 * n_swa + 8; ++kv_start) {
            for (int q : {1, 2, 3, 4, 5, 8, 16}) {
                std::vector<int> ring(n_swa, -1);
                for (int pos = 0; pos < kv_start + q; ++pos) {
                    ring[(size_t) (pos % n_swa)] = pos;
                }
                for (int lane = 0; lane < q; ++lane) {
                    DeepSeek4RawRingSpan spans[2];
                    const int count = deepseek4_verify_raw_mask_spans(
                        kv_start, n_swa, q, lane, spans);
                    TEST_ASSERT(count >= 0 && count <= 2);
                    std::vector<bool> masked(n_swa, false);
                    for (int j = 0; j < count; ++j) {
                        TEST_ASSERT(spans[j].row >= 0 && spans[j].count > 0);
                        TEST_ASSERT(spans[j].row + spans[j].count <= n_swa);
                        for (int row = spans[j].row;
                             row < spans[j].row + spans[j].count && row < n_swa; ++row) {
                            if (row >= 0) masked[(size_t) row] = true;
                        }
                    }
                    for (int row = 0; row < n_swa; ++row) {
                        TEST_ASSERT(masked[(size_t) row] ==
                            (ring[(size_t) row] < 0 || ring[(size_t) row] > kv_start + lane));
                    }
                    ++cases;
                }
            }
        }
    }
    std::fprintf(stderr, " %d cases\n", cases);
}

#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
static void test_failed_init_preserves_sparse_opt_in() {
    std::fprintf(stderr, "  test_failed_init_preserves_sparse_opt_in ...\n");
    if (ggml_backend_cuda_get_device_count() == 0) return;
    ScopedEnvVar spec("DFLASH_DS4_SPEC");
    ScopedEnvVar draft("DFLASH_DS4_DRAFT");
    ScopedEnvVar sparse("DFLASH_DS4_SPARSE_DECODE_FLASH");
    ScopedEnvVar mmvq("LUCE_MMVQ_MAX_NCOLS");
    setenv("DFLASH_DS4_SPEC", "1", 1);
    unsetenv("DFLASH_DS4_DRAFT");
    // The removed gfx1151 auto-enable ran in init() BEFORE load_model().
    // Deliberately fail at model loading: even a failed init with no drafter
    // must not change process-wide sparse-verifier policy for the next model.
    // This tests early-init side effects, not successful verifier construction.
    const std::string missing_model = make_temp_gguf_path("missing");
    for (const char * value : {static_cast<const char *>(nullptr), "0", "1"}) {
        if (value) setenv("DFLASH_DS4_SPARSE_DECODE_FLASH", value, 1);
        else unsetenv("DFLASH_DS4_SPARSE_DECODE_FLASH");
        DeepSeek4BackendConfig cfg;
        cfg.model_path = missing_model.c_str();
        cfg.device.gpu = 0;
        DeepSeek4Backend backend(cfg);
        TEST_ASSERT(!backend.init());
        const char * actual = std::getenv("DFLASH_DS4_SPARSE_DECODE_FLASH");
        TEST_ASSERT(value ? actual && std::strcmp(actual, value) == 0 : !actual);
    }
}

static void test_failed_init_preserves_mix_mmq_policy() {
    std::fprintf(stderr, "  test_failed_init_preserves_mix_mmq_policy ...\n");
    if (ggml_backend_cuda_get_device_count() == 0) return;
    ScopedEnvVar saved("DFLASH_DS4_MIX_MMQ_PREFILL");
    const std::string missing_model = make_temp_gguf_path("missing-mix-policy");
    for (const char * value : {static_cast<const char *>(nullptr), "0", "1"}) {
        if (value) setenv("DFLASH_DS4_MIX_MMQ_PREFILL", value, 1);
        else unsetenv("DFLASH_DS4_MIX_MMQ_PREFILL");
        for (auto mode : {PrefillAttentionMode::Sparse, PrefillAttentionMode::Exact}) {
            DeepSeek4BackendConfig cfg;
            cfg.model_path = missing_model.c_str();
            cfg.device.gpu = 0;
            cfg.prefill_mode = mode;
            DeepSeek4Backend backend(cfg);
            TEST_ASSERT(!backend.init());
            const char * actual = std::getenv("DFLASH_DS4_MIX_MMQ_PREFILL");
            TEST_ASSERT(value ? actual && std::strcmp(actual, value) == 0 : !actual);
        }
    }
}
#endif

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

static std::vector<uint8_t> read_tensor_rows(const ggml_tensor * tensor,
                                             int rows) {
    const size_t bytes = rows > 0
        ? ggml_row_size(tensor->type, tensor->ne[0]) * (size_t) rows : 0;
    std::vector<uint8_t> data(bytes);
    if (bytes > 0) ggml_backend_tensor_get(tensor, data.data(), 0, bytes);
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

static void test_layer_split_request_propagates_sampler() {
    std::fprintf(stderr, "  test_layer_split_request_propagates_sampler ...");

    DeepSeek4LayerSplitAdapter adapter({});
    GenerateRequest req;
    req.do_sample = true;
    req.sampler.temp = 0.25f;
    req.sampler.top_p = 0.9f;
    req.sampler.seed = 42;

    adapter.begin_request(req);

    TEST_ASSERT(adapter.sampler_.temp == req.sampler.temp);
    TEST_ASSERT(adapter.sampler_.top_p == req.sampler.top_p);
    TEST_ASSERT(adapter.sampler_.seed == req.sampler.seed);
    std::mt19937_64 expected(req.sampler.seed);
    TEST_ASSERT(adapter.sampler_rng_() == expected());

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_layer_split_sampler_uses_prompt_history() {
    std::fprintf(stderr, "  test_layer_split_sampler_uses_prompt_history ...");

    SamplerCfg sampler;
    sampler.rep_pen = 2.0f;
    const std::vector<float> logits = {0.0f, 4.0f, 3.0f};
    const std::vector<int32_t> prompt_history = {1};
    std::vector<int32_t> out_tokens;
    std::mt19937_64 rng(42);
    const bool ok = run_layer_split_ar_decode(
        /*last_tok=*/1, /*committed=*/1, /*n_gen=*/1, /*vocab=*/3,
        logits, sampler, rng, prompt_history,
        [](const std::vector<int32_t> &, int, int &,
           std::vector<float> *) { return false; },
        [](int) { return false; }, out_tokens, DaemonIO{});

    TEST_ASSERT(ok);
    TEST_ASSERT(out_tokens.size() == 1);
    if (out_tokens.size() == 1) {
        TEST_ASSERT(out_tokens[0] == 2);
    }

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_layer_split_sampler_appends_generated_tokens() {
    std::fprintf(stderr,
                 "  test_layer_split_sampler_appends_generated_tokens ...");

    SamplerCfg sampler;
    sampler.rep_pen = 2.0f;
    const std::vector<float> logits = {0.0f, 4.0f, 3.0f};
    std::vector<int32_t> out_tokens;
    std::mt19937_64 rng(42);
    const bool ok = run_layer_split_ar_decode(
        /*last_tok=*/0, /*committed=*/0, /*n_gen=*/2, /*vocab=*/3,
        logits, sampler, rng, /*history_prefix=*/{},
        [&logits](const std::vector<int32_t> &, int, int &,
                  std::vector<float> * logits_out) {
            if (logits_out) *logits_out = logits;
            return true;
        },
        [](int) { return false; }, out_tokens, DaemonIO{});

    TEST_ASSERT(ok);
    // With no prompt history the first sample is the plain argmax (token 1);
    // the second sees token 1 in history, so rep_pen drops its logit to 2 and
    // token 2 (logit 3) wins.
    TEST_ASSERT(out_tokens == std::vector<int32_t>({1, 2}));

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

class TestLayerSplitHistoryAdapter final : public LayerSplitAdapter {
public:
    const char * name() const override { return "test-history"; }
    bool init() override { return true; }
    int max_context() const override { return 64; }
    void reset_request_state() override { reset_called = true; }
    bool prefill(const std::vector<int32_t> & prompt,
                 int, int & last_tok) override {
        prefilled.insert(prefilled.end(), prompt.begin(), prompt.end());
        last_tok = 2;
        return true;
    }
    bool decode_ar(int, int, int,
                   const std::vector<int32_t> & history_prefix,
                   std::vector<int32_t> & out_tokens,
                   const DaemonIO &) override {
        decoded_history = history_prefix;
        out_tokens.push_back(2);
        return true;
    }
    bool supports_cpu_sampling() const override { return true; }
    void free_drafter() override {}
    int snapshot_cur_pos(int) const override { return 1; }
    bool snapshot_restore(int) override {
        restore_called = true;
        return true;
    }
    int current_last_token() const override { return 1; }
    void shutdown() override {}

    bool reset_called = false;
    bool restore_called = false;
    std::vector<int32_t> prefilled;
    std::vector<int32_t> decoded_history;
};

static void test_layer_split_restore_preserves_full_sampling_history() {
    std::fprintf(stderr,
                 "  test_layer_split_restore_preserves_full_sampling_history ...");

    auto adapter = std::make_unique<TestLayerSplitHistoryAdapter>();
    auto * observed = adapter.get();
    LayerSplitBackend backend(std::move(adapter));
    GenerateRequest req;
    req.prompt = {1, 2, 3};
    req.n_gen = 1;

    const GenerateResult result =
        backend.restore_and_generate_impl(0, req, DaemonIO{});

    TEST_ASSERT(result.ok());
    TEST_ASSERT(observed->restore_called);
    TEST_ASSERT(!observed->reset_called);
    TEST_ASSERT(observed->prefilled == std::vector<int32_t>({2, 3}));
    TEST_ASSERT(observed->decoded_history == req.prompt);

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_backend_sampling_penalizes_prompt_history() {
    std::fprintf(stderr, "  test_backend_sampling_penalizes_prompt_history ...");

    DeepSeek4BackendConfig cfg;
    DeepSeek4Backend backend(cfg);
    backend.w_.n_vocab = 3;

    std::vector<int32_t> emitted;
    DaemonIO io;
    io.on_token = [&](int32_t tok) {
        emitted.push_back(tok);
        return true;
    };

    auto decode_one = [&](const SamplerCfg & sampler) {
        backend.last_logits_ = {0.0f, 4.0f, 3.0f};
        backend.sampler_ = sampler;
        emitted.clear();
        std::vector<int32_t> generated;
        const bool ok = backend.do_decode(
            /*committed=*/1,
            /*n_gen=*/1,
            /*history_prefix=*/{1},
            generated,
            io);
        TEST_ASSERT(ok);
        TEST_ASSERT(emitted == generated);
        return generated.empty() ? int32_t{-1} : generated.front();
    };

    SamplerCfg greedy;
    TEST_ASSERT(decode_one(greedy) == 1);

    SamplerCfg penalized;
    penalized.rep_pen = 2.0f;
    TEST_ASSERT(decode_one(penalized) == 2);

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


static void test_image_bias_loader_opt_in_contract(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_image_bias_loader_opt_in_contract ...");
    DeepSeek4FixtureOptions valid;
    valid.vocab_size = 129280;
    valid.image_biases = true;
    valid.add_mtp_image_bias = true;
    const std::string path = write_deepseek4_loader_fixture(valid);
    for (bool enabled : {false, true}) {
        TargetLoadPlan plan;
        plan.load_ds4_image_bias = enabled;
        DeepSeek4Weights weights;
        const bool ok = load_deepseek4_gguf_partial(path, backend, plan, weights);
        TEST_ASSERT_MSG(ok, dflash27b_last_error());
        if (ok) {
            TEST_ASSERT(weights.layers.size() == 43);
            for (size_t i = 0; i < weights.layers.size(); ++i) {
                const auto bias = weights.layers[i].ffn_gate_bias_vl;
                TEST_ASSERT(bool(bias) == enabled);
                if (bias) {
                    TEST_ASSERT(bias->type == GGML_TYPE_F32 && ggml_nelements(bias) == 256);
                    std::vector<float> values(256);
                    ggml_backend_tensor_get(bias, values.data(), 0, values.size() * sizeof(float));
                    TEST_ASSERT(std::all_of(values.begin(), values.end(),
                        [i](float value) { return value == float(i + 1); }));
                }
            }
            if (weights.ctx) {
                const auto mtp = ggml_get_tensor(weights.ctx, "layers.43.ffn.gate.bias_vl");
                TEST_ASSERT(mtp == nullptr || (mtp->buffer == nullptr && mtp->data == nullptr));
            }
        }
        free_deepseek4_weights(weights);
    }
    for (int boundary : {0, 1}) {
        TargetLoadPlan plan;
        plan.load_ds4_image_bias = true;
        if (boundary == 0) plan.layer_begin = 1;
        else plan.layer_end = 42;
        DeepSeek4Weights weights;
        TEST_ASSERT(!load_deepseek4_gguf_partial(path, backend, plan, weights));
        TEST_ASSERT_MSG(std::string(dflash27b_last_error()).find("exactly 43 F32[256]") != std::string::npos,
                        dflash27b_last_error());
        TEST_ASSERT(weights.ctx == nullptr && weights.buf == nullptr);
        free_deepseek4_weights(weights);
    }
    unlink(path.c_str());

    std::vector<DeepSeek4FixtureOptions> invalid;
    auto missing_all = valid;
    missing_all.image_biases = false;
    invalid.push_back(missing_all);
    for (int layer : {0, 21, 42}) {
        auto missing = valid;
        missing.missing_image_bias = layer;
        invalid.push_back(missing);
    }
    auto wrong_type = valid;
    wrong_type.malformed_image_bias = 21;
    wrong_type.image_bias_type = GGML_TYPE_F16;
    invalid.push_back(wrong_type);
    auto wrong_width = valid;
    wrong_width.malformed_image_bias = 42;
    wrong_width.image_bias_width = 255;
    invalid.push_back(wrong_width);
    auto matrix = valid;
    matrix.malformed_image_bias = 0;
    matrix.image_bias_width = 128;
    matrix.image_bias_rows = 2;
    invalid.push_back(matrix);
    auto wrong_layers = valid;
    wrong_layers.block_count = 42;
    invalid.push_back(wrong_layers);
    auto wrong_vocab = valid;
    wrong_vocab.vocab_size = 129279;
    invalid.push_back(wrong_vocab);
    for (const auto & options : invalid) {
        const std::string bad_path = write_deepseek4_loader_fixture(options);
        TargetLoadPlan plan;
        plan.load_ds4_image_bias = true;
        DeepSeek4Weights weights;
        TEST_ASSERT(!load_deepseek4_gguf_partial(bad_path, backend, plan, weights));
        TEST_ASSERT_MSG(std::string(dflash27b_last_error()).find("exactly 43 F32[256]") != std::string::npos,
                        dflash27b_last_error());
        TEST_ASSERT(weights.ctx == nullptr && weights.buf == nullptr && weights.dense_split_buf == nullptr);
        free_deepseek4_weights(weights);
        plan.load_ds4_image_bias = false;
        TEST_ASSERT_MSG(load_deepseek4_gguf_partial(bad_path, backend, plan, weights), dflash27b_last_error());
        for (const auto & layer : weights.layers) TEST_ASSERT(layer.ffn_gate_bias_vl == nullptr);
        free_deepseek4_weights(weights);
        unlink(bad_path.c_str());
    }
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_image_batch_admission_before_execution(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_image_batch_admission_before_execution ...");
    ggml_context * ctx = ggml_init({1u << 20, nullptr, false});
    ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    ggml_tensor * state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    static_cast<float *>(state->data)[0] = 123.0f;
    DeepSeek4Weights weights;
    weights.n_vocab = 129280;
    weights.n_layer = 43;
    weights.moe_hybrid = true;
    weights.layers.resize(43);
    weights.compress_ratios.resize(43);
    DeepSeek4Cache cache;
    cache.max_ctx = 32;
    cache.cur_pos = 17;
    cache.prefill_mode = PrefillAttentionMode::Sparse;
    cache.layers.resize(43);
    MoeHybridStorage hybrid;
    hybrid.layers.resize(43);
    hybrid.cold_backend_kind = MoeHybridColdBackend::Gpu;
    hybrid.cold_backend = backend;
    for (int i = 0; i < 43; ++i) {
        weights.compress_ratios[size_t(i)] = i < 2 ? 0 : i % 2 == 0 ? 4 : 128;
        auto & layer = weights.layers[size_t(i)];
        auto & layer_cache = cache.layers[size_t(i)];
        layer.ffn_gate_bias_vl = bias;
        layer.attn_compressor_ape = layer.attn_compressor_kv = layer.attn_compressor_gate = layer.attn_compressor_norm = state;
        layer.indexer_compressor_ape = layer.indexer_compressor_kv = layer.indexer_compressor_gate = layer.indexer_compressor_norm = state;
        layer_cache.raw_kv = layer_cache.comp_kv = layer_cache.index_comp_kv = state;
        layer_cache.attn_compressor.state_kv = layer_cache.attn_compressor.state_score = state;
        layer_cache.indexer_compressor.state_kv = layer_cache.indexer_compressor.state_score = state;
    }
    const dflash::vision::TokenSpan span{1, 2, 5, 6};
    const dflash::vision::ImageSpanView spans{&span, 1};
    std::vector<int32_t> tokens{7, 129280, 129281, 129282, 129283, 129284, 8};
    bool has_images = false;
    std::string error;
    auto validate = [&]() {
        error.clear();
        return deepseek4_validate_image_batch(weights, cache, &hybrid, tokens.data(),
            int(tokens.size()), 0, spans, has_images, error);
    };
    TEST_ASSERT(validate() && has_images);
    for (size_t position : {size_t(0), size_t(1), size_t(5), size_t(6)}) {
        const int32_t original = tokens[position];
        tokens[position] = position == 0 || position == 6 ? 129280 : 7;
        TEST_ASSERT(!validate());
        TEST_ASSERT(error.find("token IDs") != std::string::npos);
        tokens[position] = original;
    }
    tokens[2] = 129285;
    TEST_ASSERT(!validate());
    tokens[2] = 129281;
    TEST_ASSERT(!deepseek4_validate_image_batch(weights, cache, &hybrid, tokens.data(),
        5, 0, spans, has_images, error));
    TEST_ASSERT(!deepseek4_validate_image_batch(weights, cache, &hybrid, tokens.data() + 2,
        5, 2, spans, has_images, error));
    TEST_ASSERT(!deepseek4_validate_image_batch(weights, cache, &hybrid, nullptr,
        7, 0, spans, has_images, error));
    const auto check_missing_tensor = [&](ggml_tensor * & tensor) {
        ggml_tensor * saved = tensor;
        tensor = nullptr;
        TEST_ASSERT(!validate());
        tensor = saved;
        TEST_ASSERT(validate());
    };
    for (int i : {0, 2, 3, 42}) {
        auto & layer = weights.layers[size_t(i)];
        auto & layer_cache = cache.layers[size_t(i)];
        check_missing_tensor(layer.ffn_gate_bias_vl);
        check_missing_tensor(layer_cache.raw_kv);
        if (i >= 2) {
            check_missing_tensor(layer.attn_compressor_gate);
            check_missing_tensor(layer_cache.attn_compressor.state_score);
        }
        if (i >= 2 && i % 2 == 0) {
            check_missing_tensor(layer.indexer_compressor_kv);
            check_missing_tensor(layer_cache.indexer_compressor.state_kv);
        }
    }
    weights.layers[42].ffn_gate_bias_vl = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 256);
    TEST_ASSERT(!validate());
    weights.layers[42].ffn_gate_bias_vl = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 2);
    TEST_ASSERT(!validate());
    weights.layers[42].ffn_gate_bias_vl = bias;
    TEST_ASSERT(validate());
    weights.compress_ratios[42] = 128;
    TEST_ASSERT(!validate());
    weights.compress_ratios[42] = 4;
    hybrid.materialized_cold_experts = false;
    TEST_ASSERT(!validate());
    hybrid.materialized_cold_experts = true;
    cache.layers.pop_back();
    TEST_ASSERT(!validate());
    TEST_ASSERT(cache.cur_pos == 17 && static_cast<float *>(state->data)[0] == 123.0f);
    const dflash::vision::TokenSpan invalid_span{1, 2, 7, 6};
    TEST_ASSERT(!deepseek4_validate_image_batch(weights, cache, &hybrid, tokens.data(),
        7, 0, {&invalid_span, 1}, has_images, error));
    const int32_t text[] = {1, 2};
    TEST_ASSERT(deepseek4_validate_image_batch(weights, cache, nullptr, text, 2, 6,
        spans, has_images, error) && !has_images);
    ggml_free(ctx);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_dspark_loader_contract_and_bounds(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_dspark_loader_contract_and_bounds ...");

    {
        const std::string path = write_dspark_loader_fixture();
        DSparkDrafter drafter;
        const bool ok = load_deepseek4_dspark_drafter(path, backend, drafter);
        TEST_ASSERT_MSG(ok, deepseek4_dspark_last_error());
        free_deepseek4_dspark_drafter(drafter);
        unlink(path.c_str());
    }

    {
        DSparkFixtureOptions opts;
        opts.wrong_main_norm_shape = true;
        const std::string path = write_dspark_loader_fixture(opts);
        DSparkDrafter drafter;
        const bool ok = load_deepseek4_dspark_drafter(path, backend, drafter);
        TEST_ASSERT(!ok);
        TEST_ASSERT_MSG(std::string(deepseek4_dspark_last_error()).find(
                            "dflash.hidden_norm.weight") != std::string::npos,
                        deepseek4_dspark_last_error());
        free_deepseek4_dspark_drafter(drafter);
        unlink(path.c_str());
    }

    {
        DSparkFixtureOptions opts;
        opts.add_unknown_tensor = true;
        const std::string path = write_dspark_loader_fixture(opts);
        DSparkDrafter drafter;
        const bool ok = load_deepseek4_dspark_drafter(path, backend, drafter);
        TEST_ASSERT(!ok);
        TEST_ASSERT_MSG(std::string(deepseek4_dspark_last_error()).find(
                            "unexpected DSpark tensor: token_embd.weight") != std::string::npos,
                        deepseek4_dspark_last_error());
        free_deepseek4_dspark_drafter(drafter);
        unlink(path.c_str());
    }

    {
        const std::string path = write_dspark_loader_fixture();
        struct stat st{};
        TEST_ASSERT(stat(path.c_str(), &st) == 0);
        TEST_ASSERT(st.st_size > 128);
        TEST_ASSERT(truncate(path.c_str(), st.st_size - 128) == 0);
        DSparkDrafter drafter;
        const bool ok = load_deepseek4_dspark_drafter(path, backend, drafter);
        TEST_ASSERT(!ok);
        TEST_ASSERT_MSG(std::string(deepseek4_dspark_last_error()).find(
                            "truncated or corrupt") != std::string::npos,
                        deepseek4_dspark_last_error());
        free_deepseek4_dspark_drafter(drafter);
        unlink(path.c_str());
    }

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_safe_compressor_batch_tokens() {
    std::fprintf(stderr, "  test_safe_compressor_batch_tokens ...");
    DeepSeek4Weights w;
    w.compress_ratios = {0, 4, 128};
    TEST_ASSERT(deepseek4_safe_compressor_batch_tokens(w, 0, 1) == 1);
    TEST_ASSERT(deepseek4_safe_compressor_batch_tokens(w, 0, 4) == 4);
    TEST_ASSERT(deepseek4_safe_compressor_batch_tokens(w, 0, 5) == 4);
    TEST_ASSERT(deepseek4_safe_compressor_batch_tokens(w, 1, 8) == 3);
    TEST_ASSERT(deepseek4_safe_compressor_batch_tokens(w, 4, 8) == 4);
    TEST_ASSERT(deepseek4_safe_compressor_batch_tokens(w, 125, 8) == 3);
    TEST_ASSERT(deepseek4_safe_compressor_batch_tokens(w, 128, 8) == 4);

    w.compress_ratios = {128};
    TEST_ASSERT(deepseek4_safe_compressor_batch_tokens(w, 0, 129) == 128);
    TEST_ASSERT(deepseek4_safe_compressor_batch_tokens(w, 127, 4) == 1);

    w.compress_ratios.clear();
    TEST_ASSERT(deepseek4_safe_compressor_batch_tokens(w, 17, 9) == 9);
    TEST_ASSERT(deepseek4_safe_compressor_batch_tokens(w, 17, 0) == 0);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_hybrid_prefill_chunk_tokens() {
    std::fprintf(stderr, "  test_hybrid_prefill_chunk_tokens ...");
    TEST_ASSERT(deepseek4_hybrid_prefill_chunk_tokens(2048, 0) == 2048);
    TEST_ASSERT(deepseek4_hybrid_prefill_chunk_tokens(2048, 4096) == 2048);
    TEST_ASSERT(deepseek4_hybrid_prefill_chunk_tokens(2048, 4097) == 1024);
    TEST_ASSERT(deepseek4_hybrid_prefill_chunk_tokens(1024, 8192) == 1024);
    TEST_ASSERT(deepseek4_hybrid_prefill_chunk_tokens(512, 8192) == 512);
    TEST_ASSERT(deepseek4_hybrid_prefill_chunk_tokens(0, 8192) == 1);
    TEST_ASSERT(deepseek4_hybrid_prefill_chunk_tokens(
                    2048, 2048, 1024) == 1024);
    TEST_ASSERT(deepseek4_hybrid_prefill_step_tokens(2048, 0, 70000) == 2048);
    TEST_ASSERT(deepseek4_hybrid_prefill_step_tokens(
                    2048, 30720, 70000) == 2048);
    TEST_ASSERT(deepseek4_hybrid_prefill_step_tokens(
                    2048, 31744, 70000) == 1024);
    TEST_ASSERT(deepseek4_hybrid_prefill_step_tokens(
                    2048, 32768, 70000) == 1024);
    TEST_ASSERT(deepseek4_hybrid_prefill_step_tokens(
                    512, 65536, 57) == 57);
    TEST_ASSERT(deepseek4_hybrid_prefill_step_tokens(
                    2048, 32768, 0) == 0);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_mix_mmq_prefill_default() {
    std::fprintf(stderr, "  test_mix_mmq_prefill_default ...");
    TEST_ASSERT(deepseek4_mix_mmq_prefill_policy(
        PrefillAttentionMode::Sparse, "gfx1151", nullptr) == GGML_MIXED_MMQ_ENABLED);
    TEST_ASSERT(deepseek4_mix_mmq_prefill_policy(
        PrefillAttentionMode::Exact, "gfx1151", nullptr) == GGML_MIXED_MMQ_DEFAULT);
    TEST_ASSERT(deepseek4_mix_mmq_prefill_policy(
        PrefillAttentionMode::Sparse, "gfx1201", nullptr) == GGML_MIXED_MMQ_DEFAULT);
    TEST_ASSERT(deepseek4_mix_mmq_prefill_policy(
        PrefillAttentionMode::Sparse, "gfx1151", "0") == GGML_MIXED_MMQ_DISABLED);
    TEST_ASSERT(deepseek4_mix_mmq_prefill_policy(
        PrefillAttentionMode::Exact, "gfx1151", "1") == GGML_MIXED_MMQ_ENABLED);
    TEST_ASSERT(!deepseek4_mix_mmq_prefill_default(
        PrefillAttentionMode::Exact, "gfx1151"));
    TEST_ASSERT(deepseek4_mix_mmq_prefill_default(
        PrefillAttentionMode::Dense, "gfx1151"));
    TEST_ASSERT(deepseek4_mix_mmq_prefill_default(
        PrefillAttentionMode::Sparse, "gfx1151:sramecc+:xnack-"));
    TEST_ASSERT(!deepseek4_mix_mmq_prefill_default(
        PrefillAttentionMode::Sparse, "gfx1201"));
    TEST_ASSERT(!deepseek4_mix_mmq_prefill_default(
        PrefillAttentionMode::Sparse, "gfx11510"));
    TEST_ASSERT(!deepseek4_mix_mmq_prefill_default(
        PrefillAttentionMode::Sparse, nullptr));
    std::fprintf(stderr, " OK\n");
}
static void test_dspark_park_all_releases_drafter() {
    std::fprintf(stderr, "  test_dspark_park_all_releases_drafter ...");

    DeepSeek4BackendConfig cfg;
    DeepSeek4Backend backend(cfg);
    backend.spec_draft_path_ = "/tmp/ds4-dspark-fixture.gguf";
    backend.spec_drafter_ = std::make_unique<DSparkDrafter>();
    backend.spec_enabled_ = true;
    backend.pflash_drafter_loaded_ = true;
    backend.pflash_drafter_path_ = "/tmp/ds4-pflash-fixture.gguf";
    backend.pflash_drafter_gpu_ = 1;

    TEST_ASSERT(backend.park(ParkTarget::All));
    TEST_ASSERT(backend.parked_);
    TEST_ASSERT(backend.spec_drafter_ == nullptr);
    TEST_ASSERT(!backend.spec_enabled_);
    TEST_ASSERT(backend.spec_drafter_parked_);
    TEST_ASSERT(!backend.pflash_drafter_loaded_);
    TEST_ASSERT(backend.pflash_drafter_path_.empty());
    TEST_ASSERT(backend.pflash_drafter_gpu_ == -1);

    backend.free_drafter();
    TEST_ASSERT(backend.spec_drafter_ == nullptr);
    TEST_ASSERT(!backend.spec_enabled_);
    TEST_ASSERT(backend.spec_drafter_parked_);
    TEST_ASSERT(backend.spec_draft_path_ == "/tmp/ds4-dspark-fixture.gguf");
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_pflash_rejects_invalid_requests() {
    std::fprintf(stderr, "  test_pflash_rejects_invalid_requests ...");
    DeepSeek4BackendConfig cfg;
    DeepSeek4Backend backend(cfg);

    ModelBackend::CompressRequest empty;
    TEST_ASSERT(!backend.compress(empty).ok);

    ModelBackend::CompressRequest invalid_ratio;
    invalid_ratio.input_ids = {1, 2, 3};
    invalid_ratio.keep_ratio = -0.1f;
    invalid_ratio.drafter_path = "/nonexistent/drafter.gguf";
    const auto results = backend.compress_batch({empty, invalid_ratio});
    TEST_ASSERT(results.size() == 2);
    TEST_ASSERT(!results[0].ok);
    TEST_ASSERT(!results[1].ok);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_pflash_failed_load_releases_backend() {
    std::fprintf(stderr, "  test_pflash_failed_load_releases_backend ...");
    DeepSeek4BackendConfig cfg;
    DeepSeek4Backend backend(cfg);
    // A supplied CPU backend exercises the real load-failure/cleanup path
    // without a model or a second GPU. Ownership is identical for HIP.
    backend.pflash_drafter_ctx_.backend = ggml_backend_cpu_init();
    backend.pflash_drafter_ctx_.gpu = 0;
    TEST_ASSERT(backend.pflash_drafter_ctx_.backend != nullptr);
    ModelBackend::CompressRequest request;
    request.input_ids = {1, 2, 3};
    request.keep_ratio = 0.5f;
    request.drafter_path = "/nonexistent/pr664-pflash.gguf";
    request.skip_park = true;
    TEST_ASSERT(!backend.compress(request).ok);
    TEST_ASSERT(!backend.pflash_drafter_loaded_);
    TEST_ASSERT(backend.pflash_drafter_ctx_.backend == nullptr);
    TEST_ASSERT(backend.pflash_drafter_ctx_.gpu == -1);
    // Also keep a failing baseline run leak-free.
    dflash::common::free_drafter(backend.pflash_drafter_ctx_);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_pflash_keep_ratio_contract() {
    std::fprintf(stderr, "  test_pflash_keep_ratio_contract ...");
    struct ParkProbe : DeepSeek4Backend {
        ParkProbe() : DeepSeek4Backend(DeepSeek4BackendConfig{}) {}
        int park_calls = 0;
        bool park(ParkTarget) override {
            ++park_calls;
            return false; // Stop before model loading; only validate admission.
        }
    };
    for (float ratio : {0.0f, 0.5f, 1.0f, -0.1f, 1.1f,
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
        ParkProbe backend;
        ModelBackend::CompressRequest request;
        request.input_ids = {1, 2, 3};
        request.drafter_path = "/nonexistent/pflash-admission.gguf";
        request.keep_ratio = ratio;
        const bool valid = std::isfinite(ratio) && ratio >= 0.0f && ratio <= 1.0f;
        TEST_ASSERT(!backend.compress(request).ok);
        TEST_ASSERT(backend.park_calls == (valid ? 1 : 0));
    }
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_indexer_visibility_suffix() {
    std::fprintf(stderr, "  test_indexer_visibility_suffix ...");
    auto * backend = ggml_backend_cpu_init();
    TEST_ASSERT(backend != nullptr);
    if (!backend) return;
    constexpr int rows = 528, tokens = 5;
    int passed = 0;
    for (int first = 0; first < tokens; ++first) {
        auto * ctx = make_test_context(4u << 20);
        TEST_ASSERT(ctx != nullptr);
        if (!ctx) continue;
        auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, tokens);
        ggml_set_input(mask);
        auto * suffix = deepseek4_indexer_visibility_suffix(ctx, mask, first, tokens - first);
        TEST_ASSERT(deepseek4_indexer_visibility_suffix(ctx, nullptr, first, tokens - first) == nullptr);
        ggml_set_output(suffix);
        auto * graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(graph, suffix);
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        TEST_ASSERT(buffer != nullptr);
        if (buffer) {
            std::vector<float> values((size_t) rows * tokens);
            for (int t = 0; t < tokens; ++t) {
                for (int r = 0; r < rows; ++r) values[(size_t) t * rows + r] = (float) (t * 1000 + r);
            }
            ggml_backend_tensor_set(mask, values.data(), 0, ggml_nbytes(mask));
            bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
            std::vector<float> actual((size_t) ggml_nelements(suffix));
            ggml_backend_tensor_get(suffix, actual.data(), 0, ggml_nbytes(suffix));
            const std::vector<float> expected(values.begin() + (size_t) first * rows, values.end());
            ok &= suffix->ne[0] == rows && suffix->ne[1] == tokens - first;
            ok &= ggml_is_contiguous(suffix) && actual == expected;
            passed += ok;
            if (!ok) std::fprintf(stderr, " FAIL first=%d;", first);
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    std::fprintf(stderr, " %d/%d cases passed\n", passed, tokens);
    TEST_ASSERT(passed == tokens);
}

static void test_pflash_legacy_compress_contract() {
    std::fprintf(stderr, "  test_pflash_legacy_compress_contract ...");
    struct CompressProbe : DeepSeek4Backend {
        CompressProbe() : DeepSeek4Backend(DeepSeek4BackendConfig{}) {}
        CompressRequest captured{};
        int calls = 0;
        CompressResult compress(const CompressRequest & request) override {
            ++calls;
            captured = request;
            CompressResult result;
            result.ok = true;
            result.compressed_ids = {request.input_ids.front()};
            return result;
        }
    };
    char path[] = "/tmp/ds4-compress-contract-XXXXXX";
    const int fd = mkstemp(path);
    TEST_ASSERT(fd >= 0);
    if (fd < 0) return;
    const int32_t ids[] = {11, 22, 33};
    const bool written = write(fd, ids, sizeof(ids)) == (ssize_t) sizeof(ids);
    close(fd);
    TEST_ASSERT(written);
    if (!written) { unlink(path); return; }
    for (bool skip_park : {false, true}) {
        CompressProbe backend;
        std::vector<int32_t> output;
        DaemonIO io;
        io.on_token = [&](int32_t token) { output.push_back(token); return true; };
        const std::string command = std::string("compress ") + path +
            " 0 /unused/drafter with spaces.gguf" + (skip_park ? " nopark" : "");
        TEST_ASSERT(backend.handle_compress(command, io));
        TEST_ASSERT(backend.calls == 1);
        TEST_ASSERT(backend.captured.input_ids == std::vector<int32_t>({11, 22, 33}));
        TEST_ASSERT(backend.captured.keep_ratio == 0.0f);
        TEST_ASSERT(backend.captured.drafter_path == "/unused/drafter with spaces.gguf");
        TEST_ASSERT(backend.captured.skip_park == skip_park);
        TEST_ASSERT(backend.captured.drafter_gpu == 0);
        TEST_ASSERT(backend.captured.residency_action == DraftResidencyAction::KeepLoaded);
        TEST_ASSERT(output == std::vector<int32_t>({11}));
    }
    unlink(path);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_dspark_raw_ring_rollback_after_wrap(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_dspark_raw_ring_rollback_after_wrap ...");

    DeepSeek4Weights weights;
    weights.n_layer = 1;
    weights.n_embd = 4;
    weights.n_hc = 1;
    weights.head_dim = 4;
    weights.n_swa = 8;
    weights.n_indexer_head_dim = 2;
    weights.compress_ratios = {4};

    DeepSeek4Cache cache;
    TEST_ASSERT(create_deepseek4_cache(backend, weights, 16, cache));
    if (!cache.buf || cache.layers.empty() || !cache.layers[0].raw_kv) {
        free_deepseek4_cache(cache);
        std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
        return;
    }

    auto & layer = cache.layers[0];
    write_tensor_pattern(layer.raw_kv, 11);
    const std::vector<uint8_t> original = read_tensor_bytes(layer.raw_kv);
    std::vector<uint8_t> expected = original;
    const size_t row_bytes = ggml_row_size(layer.raw_kv->type, layer.raw_kv->ne[0]);

    cache.cur_pos = 10;
    layer.n_comp = 2;
    layer.n_index_comp = 2;
    DeepSeek4SpecRollback rollback;
    deepseek4_spec_rollback_save(cache, rollback, 10, 6);
    TEST_ASSERT(rollback.raw_count == 6);

    auto overwrite_row = [&](int absolute_pos, uint8_t value) {
        const int row = absolute_pos % weights.n_swa;
        std::vector<uint8_t> bytes(row_bytes, value);
        ggml_backend_tensor_set(layer.raw_kv, bytes.data(),
                                (size_t) row * layer.raw_kv->nb[1], row_bytes);
    };
    auto expect_overwritten_row = [&](int absolute_pos, uint8_t value) {
        const int row = absolute_pos % weights.n_swa;
        std::fill(expected.begin() + (size_t) row * layer.raw_kv->nb[1],
                  expected.begin() + (size_t) row * layer.raw_kv->nb[1] + row_bytes,
                  value);
    };
    for (int t = 0; t < 6; ++t) {
        overwrite_row(10 + t, (uint8_t) (0xa0 + t));
    }

    // Commit positions 10 and 11. Their new rows stay in the ring, while the
    // rejected positions 12 and 13 must reveal the older history they replaced.
    expect_overwritten_row(10, 0xa0);
    expect_overwritten_row(11, 0xa1);
    deepseek4_spec_rollback_apply(rollback, weights, cache, 12, false);
    TEST_ASSERT(read_tensor_bytes(layer.raw_kv) == expected);
    TEST_ASSERT(cache.cur_pos == 12);
    TEST_ASSERT(layer.n_comp == 3);
    TEST_ASSERT(layer.n_index_comp == 3);

    // Restoring to the pre-verify position is used by replay/diagnostic paths
    // and must put every overwritten physical row back.
    deepseek4_spec_rollback_apply(rollback, weights, cache, 10, false);
    TEST_ASSERT(read_tensor_bytes(layer.raw_kv) == original);
    TEST_ASSERT(cache.cur_pos == 10);
    TEST_ASSERT(layer.n_comp == 2);
    TEST_ASSERT(layer.n_index_comp == 2);

    free_deepseek4_cache(cache);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_dspark_compressor_rollback(ggml_backend_t backend, int copy_mode = 0) {
    std::fprintf(stderr, "  test_dspark_compressor_rollback (%s, mode=%d) ...",
                 ggml_backend_name(backend), copy_mode);
    ggml_backend_t copy_backend = copy_mode == 0 ? nullptr : backend;
    const bool pinned = copy_mode == 2;
    ggml_context * ctx = ggml_init({64 * ggml_tensor_overhead(), nullptr, true});
    TEST_ASSERT(ctx != nullptr);
    if (!ctx) return;
    DeepSeek4Weights weights;
    weights.compress_ratios = {4, 128};
    DeepSeek4Cache cache;
    cache.layers.resize(2);
    std::vector<ggml_tensor *> tensors;
    for (int il = 0; il < 2; ++il) {
        for (auto * state : {&cache.layers[il].attn_compressor,
                            &cache.layers[il].indexer_compressor}) {
            state->state_kv = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 3, il == 0 ? 8 : 128);
            state->state_score = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, il == 0 ? 8 : 128);
            tensors.push_back(state->state_kv);
            tensors.push_back(state->state_score);
        }
    }
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    TEST_ASSERT(buffer != nullptr);
    if (!buffer) { ggml_free(ctx); return; }
    std::vector<std::vector<uint8_t>> initial;
    for (size_t ti = 0; ti < tensors.size(); ++ti) {
        write_tensor_pattern(tensors[ti], 11 + (int) ti);
        initial.push_back(read_tensor_bytes(tensors[ti]));
    }
    auto load = [&](const std::vector<std::vector<uint8_t>> & state,
                    bool async = false) {
        for (size_t ti = 0; ti < tensors.size(); ++ti) {
            if (async) {
                ggml_backend_tensor_set_async(
                    copy_backend, tensors[ti], state[ti].data(), 0, state[ti].size());
            } else {
                ggml_backend_tensor_set(
                    tensors[ti], state[ti].data(), 0, state[ti].size());
            }
        }
    };
    // Independent sequential state transition. Distinct byte patterns exercise
    // exact restoration of both F16 KV and F32 score rows without rounding.
    auto advance = [&](std::vector<std::vector<uint8_t>> & state, int pos, int count) {
        for (size_t ti = 0; ti < tensors.size(); ++ti) {
            const int ratio = ti < 4 ? 4 : 128;
            const size_t stride = tensors[ti]->nb[1];
            for (int i = 0; i < count; ++i) {
                const int slot = (pos + i) % ratio;
                const int row = ratio == 4 ? 4 + slot : slot;
                std::fill_n(state[ti].begin() + row * stride, stride,
                            (uint8_t) (160 + 8 * ti + i));
                if (ratio == 4 && slot == 3)
                    std::copy_n(state[ti].begin() + 4 * stride, 4 * stride, state[ti].begin());
            }
        }
    };
    {
        DeepSeek4SpecRollback rollback;
        for (int pos = 0; pos < 128; ++pos) {
            // Six exercises legacy staging capacity, not q6 verifier support.
            for (int q = 1; q <= 6; ++q) {
                for (int accepted = 0; accepted <= q; ++accepted) {
                    load(initial, copy_backend != nullptr);
                    deepseek4_spec_rollback_save(cache, rollback, pos, q, copy_backend, pinned);
                    if (pinned && pos == 0 && q == 1 && accepted == 0) {
                        const auto host_type = ggml_backend_dev_host_buffer_type(
                            ggml_backend_get_device(backend));
                        TEST_ASSERT(rollback.pinned_buf && rollback.pinned_base);
                        TEST_ASSERT(rollback.pinned_buf &&
                            ggml_backend_buffer_get_type(rollback.pinned_buf) == host_type);
                    }
                    auto verified = initial;
                    advance(verified, pos, q);
                    load(verified, copy_backend != nullptr);
                    // Production restores to the start and replays for q>4.
                    const int keep = q > 4 && accepted < q ? 0 : accepted;
                    const int boundary = pos + 3 - (pos & 3);
                    const bool restore_prev = boundary < pos + q && boundary >= pos + keep;
                    deepseek4_spec_rollback_apply(
                        rollback, weights, cache, pos + keep, restore_prev);
                    ggml_backend_synchronize(backend);
                    if (q > 4 && accepted < q) {
                        std::vector<std::vector<uint8_t>> restored;
                        for (auto * t : tensors) restored.push_back(read_tensor_bytes(t));
                        advance(restored, pos, accepted);
                        load(restored);
                    }
                    auto expected = initial;
                    advance(expected, pos, accepted);
                    for (size_t ti = 0; ti < tensors.size(); ++ti) {
                        if (read_tensor_bytes(tensors[ti]) != expected[ti]) {
                            std::fprintf(stderr, " pos=%d q=%d accepted=%d tensor=%zu", pos, q, accepted, ti);
                            TEST_ASSERT(false);
                            // One actionable failure, not thousands of duplicates.
                            ggml_backend_buffer_free(buffer);
                            ggml_free(ctx);
                            return;
                        }
                    }
                    TEST_ASSERT(cache.cur_pos == pos + keep);
                    TEST_ASSERT(cache.layers[0].n_comp == (pos + keep) / 4);
                    TEST_ASSERT(cache.layers[0].n_index_comp == (pos + keep) / 4);
                    TEST_ASSERT(cache.layers[1].n_comp == (pos + keep) / 128);
                }
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    std::fprintf(stderr, " done\n");
}

static void test_snapshot_save_restore() {
    std::fprintf(stderr, "  test_snapshot_save_restore ...");

    auto adapter = make_test_adapter();
    TEST_ASSERT(init_snapshot_test_shard(adapter));
    if (adapter.shards_.empty() || !adapter.shards_[0].backend) {
        std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
        return;
    }

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
    const std::vector<uint8_t> comp_before = read_tensor_rows(layer.comp_kv, 5);
    const std::vector<uint8_t> index_before = read_tensor_rows(layer.index_comp_kv, 3);
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
    TEST_ASSERT(adapter.snapshots_[0].shards[0].layers[0].comp_kv->ne[1] == 5);
    TEST_ASSERT(adapter.snapshots_[0].shards[0].layers[0].index_comp_kv->ne[1] == 3);

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
    TEST_ASSERT(read_tensor_rows(layer.comp_kv, 5) == comp_before);
    TEST_ASSERT(read_tensor_rows(layer.index_comp_kv, 3) == index_before);
    TEST_ASSERT(read_tensor_bytes(layer.attn_compressor.state_kv) == attn_kv_before);
    TEST_ASSERT(read_tensor_bytes(layer.attn_compressor.state_score) == attn_score_before);
    TEST_ASSERT(read_tensor_bytes(layer.indexer_compressor.state_kv) == index_kv_before);
    TEST_ASSERT(read_tensor_bytes(layer.indexer_compressor.state_score) == index_score_before);

    // A zero-row compressed prefix is represented by one physical GGML row.
    // ggml_n_dims() reports that tensor as 1D, so restore must validate its
    // row layout rather than reject it against the full 2D cache capacity.
    adapter.snapshot_free(0);
    layer.n_comp = 0;
    layer.n_index_comp = 0;
    cache.cur_pos = 1;
    adapter.cur_pos_ = 1;
    adapter.last_tok_ = 7;
    TEST_ASSERT(adapter.snapshot_save(0));
    TEST_ASSERT(adapter.snapshots_[0].shards[0].layers[0].comp_kv->ne[1] == 1);
    TEST_ASSERT(adapter.snapshots_[0].shards[0].layers[0].index_comp_kv->ne[1] == 1);
    cache.cur_pos = 0;
    TEST_ASSERT(adapter.snapshot_restore(0));
    TEST_ASSERT(cache.cur_pos == 1);
    TEST_ASSERT(layer.n_comp == 0);
    TEST_ASSERT(layer.n_index_comp == 0);

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

static bool init_monolithic_snapshot_test_backend(DeepSeek4Backend & backend) {
    backend.backend_ = ggml_backend_cpu_init();
    backend.snap_backend_ = ggml_backend_cpu_init();
    if (!backend.backend_ || !backend.snap_backend_) return false;
    backend.w_.n_layer = 1;
    backend.w_.n_embd = 4;
    backend.w_.n_hc = 1;
    backend.w_.n_vocab = 3;
    backend.w_.head_dim = 4;
    backend.w_.n_swa = 8;
    backend.w_.n_indexer_head_dim = 2;
    backend.w_.compress_ratios = {4};
    return create_deepseek4_cache(backend.backend_, backend.w_, 16,
                                  backend.cache_);
}

static void test_monolithic_snapshot_preserves_decode_state() {
    std::fprintf(stderr,
                 "  test_monolithic_snapshot_preserves_decode_state ...");

    DeepSeek4BackendConfig cfg;
    DeepSeek4Backend backend(cfg);
    TEST_ASSERT(init_monolithic_snapshot_test_backend(backend));
    if (!backend.cache_.buf || backend.cache_.layers.empty()) {
        std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
        return;
    }

    auto & layer = backend.cache_.layers[0];
    write_tensor_pattern(layer.raw_kv, 13);
    write_tensor_pattern(layer.comp_kv, 31);
    write_tensor_pattern(layer.index_comp_kv, 47);
    write_tensor_pattern(layer.attn_compressor.state_kv, 59);
    write_tensor_pattern(layer.attn_compressor.state_score, 71);
    write_tensor_pattern(layer.indexer_compressor.state_kv, 83);
    write_tensor_pattern(layer.indexer_compressor.state_score, 97);
    write_tensor_pattern(backend.cache_.hc_state, 109);

    layer.n_comp = 5;
    layer.n_index_comp = 3;
    backend.cache_.cur_pos = 7;
    backend.last_logits_ = {1.0f, 4.0f, 2.0f};
    backend.last_logits_pos_ = 7;
    backend.spec_feat_window_ = {9.0f, 8.0f, 7.0f, 6.0f};

    const auto raw_before = read_tensor_bytes(layer.raw_kv);
    const auto comp_before = read_tensor_rows(layer.comp_kv, layer.n_comp);
    const auto index_before =
        read_tensor_rows(layer.index_comp_kv, layer.n_index_comp);
    const auto hc_before = read_tensor_bytes(backend.cache_.hc_state);

    TEST_ASSERT(backend.snapshot_save(0));
    TEST_ASSERT(backend.snapshot_used(0));
    TEST_ASSERT(backend.snapshot_cur_pos(0) == 7);
    TEST_ASSERT(backend.snapshots_[0].layers[0].comp_kv->ne[1] == 5);
    TEST_ASSERT(backend.snapshots_[0].layers[0].index_comp_kv->ne[1] == 3);

    ggml_backend_buffer_clear(backend.cache_.buf, 0);
    backend.cache_.cur_pos = 0;
    layer.n_comp = 0;
    layer.n_index_comp = 0;
    backend.last_logits_ = {-1.0f};
    backend.last_logits_pos_ = -1;
    backend.spec_feat_window_.clear();

    TEST_ASSERT(backend.snapshot_restore(0));
    TEST_ASSERT(backend.cache_.cur_pos == 7);
    TEST_ASSERT(layer.n_comp == 5);
    TEST_ASSERT(layer.n_index_comp == 3);
    TEST_ASSERT(read_tensor_bytes(layer.raw_kv) == raw_before);
    TEST_ASSERT(read_tensor_rows(layer.comp_kv, 5) == comp_before);
    TEST_ASSERT(read_tensor_rows(layer.index_comp_kv, 3) == index_before);
    TEST_ASSERT(read_tensor_bytes(backend.cache_.hc_state) == hc_before);
    TEST_ASSERT(backend.last_logits_ == std::vector<float>({1.0f, 4.0f, 2.0f}));
    TEST_ASSERT(backend.spec_feat_window_ ==
                std::vector<float>({9.0f, 8.0f, 7.0f, 6.0f}));
    TEST_ASSERT(backend.last_logits_pos_ == 7);

    GenerateRequest exact;
    exact.prompt.assign(7, 0);
    exact.n_gen = 1;
    const GenerateResult exact_result =
        backend.restore_and_generate_impl(0, exact, DaemonIO{});
    TEST_ASSERT(exact_result.ok());
    TEST_ASSERT(exact_result.tokens == std::vector<int32_t>({1}));
    TEST_ASSERT(backend.cache_.cur_pos == 7);
    TEST_ASSERT(backend.last_logits_ == std::vector<float>({1.0f, 4.0f, 2.0f}));

    // A cache state that advanced without corresponding logits (for example,
    // after DSpark) must not be persisted with stale decode state.
    backend.cache_.cur_pos = 8;
    TEST_ASSERT(!backend.snapshot_save(1));
    backend.cache_.cur_pos = 7;
    TEST_ASSERT(!backend.snapshot_save(-1));
    TEST_ASSERT(!backend.snapshot_save(ModelBackend::kMaxSlots));

    // Parking the target releases both the core tensors and the potentially
    // large host-side logits/feature vectors for every populated slot.
    TEST_ASSERT(backend.park(ParkTarget::TargetModel));
    TEST_ASSERT(!backend.snapshot_used(0));
    TEST_ASSERT(backend.snapshot_aux_[0].last_logits.empty());
    TEST_ASSERT(backend.snapshot_aux_[0].spec_feat_window.empty());
    TEST_ASSERT(!backend.snapshot_restore(0));

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

// Every DeepSeek snapshot tensor must carry a stable name: the ondisk prefix
// cache keys on names and fingerprints the layout from them.
static bool all_snapshot_tensors_named(ggml_context * ctx, size_t * count_out) {
    size_t n = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (!t->name[0]) return false;
        n++;
    }
    if (count_out) *count_out = n;
    return true;
}

static std::string make_test_disk_cache_dir(const char * tag) {
    return "/tmp/dflash_test_ds4_disk_" + std::string(tag) + "_" +
           std::to_string((long) getpid());
}

static void remove_test_disk_cache_dir(const std::string & dir) {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

// Rebuild a snapshot context the way DiskPrefixCache::read_file() does after
// deserializing: fresh 4-D tensors by name/type/shape in a host buffer with
// the bytes copied over. `override_name`/`override_ne1` corrupt one tensor's
// row count to exercise validation. Lets adapter-level tests exercise
// snapshot_adopt without going through a ModelBackend.
static bool clone_snapshot_context_like_disk_reader(ggml_context * src_ctx,
                                                    ggml_context ** ctx_out,
                                                    ggml_backend_buffer_t * buf_out,
                                                    const char * override_name = nullptr,
                                                    int64_t override_ne1 = 0) {
    size_t n = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(src_ctx); t; t = ggml_get_next_tensor(src_ctx, t)) n++;
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (n + 4) + 4096;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    for (ggml_tensor * t = ggml_get_first_tensor(src_ctx); t; t = ggml_get_next_tensor(src_ctx, t)) {
        int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
        if (override_name && std::strcmp(t->name, override_name) == 0) ne[1] = override_ne1;
        ggml_tensor * d = ggml_new_tensor(ctx, t->type, 4, ne);
        if (!d) { ggml_free(ctx); return false; }
        ggml_set_name(d, t->name);
    }
    ggml_backend_buffer_t buf =
        ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    if (!buf) { ggml_free(ctx); return false; }
    ggml_backend_buffer_clear(buf, 0);
    std::vector<uint8_t> tmp;
    for (ggml_tensor * t = ggml_get_first_tensor(src_ctx); t; t = ggml_get_next_tensor(src_ctx, t)) {
        ggml_tensor * d = ggml_get_tensor(ctx, t->name);
        if (!d) { ggml_backend_buffer_free(buf); ggml_free(ctx); return false; }
        const size_t bytes = std::min(ggml_nbytes(t), ggml_nbytes(d));
        tmp.resize(bytes);
        ggml_backend_tensor_get(t, tmp.data(), 0, bytes);
        ggml_backend_tensor_set(d, tmp.data(), 0, bytes);
    }
    *ctx_out = ctx;
    *buf_out = buf;
    return true;
}

static void test_monolithic_snapshot_disk_roundtrip() {
    std::fprintf(stderr, "  test_monolithic_snapshot_disk_roundtrip ...");

    DeepSeek4BackendConfig cfg;
    DeepSeek4Backend backend(cfg);
    TEST_ASSERT(init_monolithic_snapshot_test_backend(backend));
    if (!backend.cache_.buf || backend.cache_.layers.empty()) {
        std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
        return;
    }

    // Slots that were never saved export nothing.
    TEST_ASSERT(backend.snapshot_ref(0).ctx == nullptr);

    auto & layer = backend.cache_.layers[0];
    write_tensor_pattern(layer.raw_kv, 5);
    write_tensor_pattern(layer.comp_kv, 23);
    write_tensor_pattern(layer.index_comp_kv, 37);
    write_tensor_pattern(layer.attn_compressor.state_kv, 43);
    write_tensor_pattern(layer.attn_compressor.state_score, 61);
    write_tensor_pattern(layer.indexer_compressor.state_kv, 73);
    write_tensor_pattern(layer.indexer_compressor.state_score, 89);
    write_tensor_pattern(backend.cache_.hc_state, 101);
    layer.n_comp = 5;
    layer.n_index_comp = 3;
    backend.cache_.cur_pos = 7;
    backend.last_logits_ = {1.0f, 4.0f, 2.0f};
    backend.last_logits_pos_ = 7;
    backend.spec_feat_window_ = {9.0f, 8.0f, 7.0f, 6.0f};

    const auto raw_before = read_tensor_bytes(layer.raw_kv);
    const auto comp_before = read_tensor_rows(layer.comp_kv, layer.n_comp);
    const auto index_before = read_tensor_rows(layer.index_comp_kv, layer.n_index_comp);
    const auto attn_kv_before = read_tensor_bytes(layer.attn_compressor.state_kv);
    const auto attn_score_before = read_tensor_bytes(layer.attn_compressor.state_score);
    const auto idx_kv_before = read_tensor_bytes(layer.indexer_compressor.state_kv);
    const auto idx_score_before = read_tensor_bytes(layer.indexer_compressor.state_score);
    const auto hc_before = read_tensor_bytes(backend.cache_.hc_state);

    TEST_ASSERT(backend.snapshot_save(0));
    const ModelBackend::SnapshotRef ref = backend.snapshot_ref(0);
    TEST_ASSERT(ref.ctx != nullptr);
    TEST_ASSERT(ref.buf != nullptr);
    TEST_ASSERT(ref.cur_pos == 7);
    size_t n_named = 0;
    TEST_ASSERT(all_snapshot_tensors_named(ref.ctx, &n_named));
    // hc + 7 per-layer + meta + logits + features
    TEST_ASSERT(n_named == 11);
    TEST_ASSERT(backend.snapshots_[0].meta_snap != nullptr);
    TEST_ASSERT(backend.snapshots_[0].last_logits_snap != nullptr);
    TEST_ASSERT(backend.snapshots_[0].spec_feat_snap != nullptr);

    // Real ondisk cache: write slot 0, then read it back into slot 1.
    const std::string dir = make_test_disk_cache_dir("mono");
    remove_test_disk_cache_dir(dir);
    DiskCacheConfig dcfg;
    dcfg.cache_dir = dir;
    dcfg.min_tokens = 1;
    DiskPrefixCache disk(dcfg, backend);
    TEST_ASSERT(disk.init());
    TEST_ASSERT(!disk.disabled());
    const std::vector<int32_t> prompt = {11, 12, 13, 14, 15, 16, 17};
    TEST_ASSERT(disk.save(0, prompt));
    TEST_ASSERT(!disk.disabled());
    TEST_ASSERT(disk.total_bytes() > 0);

    // Wipe the in-memory snapshot and the live cache before reloading.
    backend.snapshot_free(0);
    TEST_ASSERT(!backend.snapshot_used(0));
    ggml_backend_buffer_clear(backend.cache_.buf, 0);
    backend.cache_.cur_pos = 0;
    layer.n_comp = 0;
    layer.n_index_comp = 0;
    backend.last_logits_ = {-1.0f};
    backend.last_logits_pos_ = -1;
    backend.spec_feat_window_.clear();

    TEST_ASSERT(disk.lookup(prompt, 1));
    TEST_ASSERT(backend.snapshot_used(1));
    TEST_ASSERT(backend.snapshot_cur_pos(1) == 7);
    TEST_ASSERT(backend.snapshots_[1].layers.size() == 1);
    TEST_ASSERT(backend.snapshots_[1].layers[0].n_comp == 5);
    TEST_ASSERT(backend.snapshots_[1].layers[0].n_index_comp == 3);
    TEST_ASSERT(backend.snapshot_aux_[1].last_logits ==
                std::vector<float>({1.0f, 4.0f, 2.0f}));
    TEST_ASSERT(backend.snapshot_aux_[1].spec_feat_window ==
                std::vector<float>({9.0f, 8.0f, 7.0f, 6.0f}));
    // An adopted snapshot is exportable again (re-save after restart).
    TEST_ASSERT(backend.snapshot_ref(1).ctx != nullptr);

    TEST_ASSERT(backend.snapshot_restore(1));
    TEST_ASSERT(backend.cache_.cur_pos == 7);
    TEST_ASSERT(layer.n_comp == 5);
    TEST_ASSERT(layer.n_index_comp == 3);
    TEST_ASSERT(read_tensor_bytes(layer.raw_kv) == raw_before);
    TEST_ASSERT(read_tensor_rows(layer.comp_kv, 5) == comp_before);
    TEST_ASSERT(read_tensor_rows(layer.index_comp_kv, 3) == index_before);
    TEST_ASSERT(read_tensor_bytes(layer.attn_compressor.state_kv) == attn_kv_before);
    TEST_ASSERT(read_tensor_bytes(layer.attn_compressor.state_score) == attn_score_before);
    TEST_ASSERT(read_tensor_bytes(layer.indexer_compressor.state_kv) == idx_kv_before);
    TEST_ASSERT(read_tensor_bytes(layer.indexer_compressor.state_score) == idx_score_before);
    TEST_ASSERT(read_tensor_bytes(backend.cache_.hc_state) == hc_before);
    TEST_ASSERT(backend.last_logits_ == std::vector<float>({1.0f, 4.0f, 2.0f}));
    TEST_ASSERT(backend.spec_feat_window_ ==
                std::vector<float>({9.0f, 8.0f, 7.0f, 6.0f}));
    TEST_ASSERT(backend.last_logits_pos_ == 7);

    // Exact full-prompt hit decodes from the reloaded logits.
    GenerateRequest exact;
    exact.prompt.assign(7, 0);
    exact.n_gen = 1;
    const GenerateResult exact_result =
        backend.restore_and_generate_impl(1, exact, DaemonIO{});
    TEST_ASSERT(exact_result.ok());
    TEST_ASSERT(exact_result.tokens == std::vector<int32_t>({1}));

    // A snapshot with a different length, zero compressed rows and an empty
    // DSpark window must land in the SAME layout (no fingerprint churn).
    backend.cache_.cur_pos = 3;
    layer.n_comp = 0;
    layer.n_index_comp = 0;
    backend.last_logits_ = {0.5f, 0.25f, 0.125f};
    backend.last_logits_pos_ = 3;
    backend.spec_feat_window_.clear();
    TEST_ASSERT(backend.snapshot_save(2));
    const std::vector<int32_t> prompt3 = {21, 22, 23};
    const size_t bytes_before = disk.total_bytes();
    TEST_ASSERT(disk.save(2, prompt3));
    TEST_ASSERT(disk.total_bytes() > bytes_before);
    backend.snapshot_free(2);
    TEST_ASSERT(disk.lookup(prompt3, 4));
    TEST_ASSERT(backend.snapshot_used(4));
    TEST_ASSERT(backend.snapshot_cur_pos(4) == 3);
    TEST_ASSERT(backend.snapshots_[4].layers[0].n_comp == 0);
    TEST_ASSERT(backend.snapshots_[4].layers[0].n_index_comp == 0);
    TEST_ASSERT(backend.snapshot_aux_[4].spec_feat_window.empty());
    TEST_ASSERT(backend.snapshot_aux_[4].last_logits ==
                std::vector<float>({0.5f, 0.25f, 0.125f}));
    TEST_ASSERT(backend.snapshot_restore(4));
    TEST_ASSERT(backend.cache_.cur_pos == 3);
    TEST_ASSERT(layer.n_comp == 0);
    // The first entry is still readable after the second save.
    TEST_ASSERT(disk.lookup(prompt, 5));
    TEST_ASSERT(backend.snapshot_cur_pos(5) == 7);

    // Adopt rejects a context that is not a snapshot at all ...
    {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 4 + 1024;
        ip.no_alloc = true;
        ggml_context * bad_ctx = ggml_init(ip);
        TEST_ASSERT(bad_ctx != nullptr);
        ggml_tensor * junk = ggml_new_tensor_1d(bad_ctx, GGML_TYPE_F32, 4);
        ggml_set_name(junk, "not_a_snapshot");
        ggml_backend_buffer_t bad_buf =
            ggml_backend_alloc_ctx_tensors_from_buft(bad_ctx, ggml_backend_cpu_buffer_type());
        TEST_ASSERT(bad_buf != nullptr);
        TEST_ASSERT(!backend.snapshot_adopt(6, bad_ctx, bad_buf, 7, -1));
        TEST_ASSERT(!backend.snapshot_used(6));
        ggml_backend_buffer_free(bad_buf);
        ggml_free(bad_ctx);
    }
    // ... a snapshot whose tensors do not match the model geometry (raw
    // window with the wrong row count) ...
    {
        const ModelBackend::SnapshotRef good = backend.snapshot_ref(5);
        TEST_ASSERT(good.ctx != nullptr);
        ggml_context * bad_ctx = nullptr;
        ggml_backend_buffer_t bad_buf = nullptr;
        TEST_ASSERT(clone_snapshot_context_like_disk_reader(
            good.ctx, &bad_ctx, &bad_buf, "ds4_snap_raw_kv_0", backend.w_.n_swa + 1));
        TEST_ASSERT(!backend.snapshot_adopt(6, bad_ctx, bad_buf, good.cur_pos, -1));
        TEST_ASSERT(!backend.snapshot_used(6));
        ggml_backend_buffer_free(bad_buf);
        ggml_free(bad_ctx);
    }
    // ... and one whose header position disagrees with the meta.
    {
        const ModelBackend::SnapshotRef good = backend.snapshot_ref(5);
        ggml_context * re_ctx = nullptr;
        ggml_backend_buffer_t re_buf = nullptr;
        TEST_ASSERT(clone_snapshot_context_like_disk_reader(good.ctx, &re_ctx, &re_buf));
        TEST_ASSERT(!backend.snapshot_adopt(6, re_ctx, re_buf, good.cur_pos + 1, -1));
        TEST_ASSERT(!backend.snapshot_used(6));
        // A faithful clone is adopted and owned by the slot afterwards.
        TEST_ASSERT(backend.snapshot_adopt(6, re_ctx, re_buf, good.cur_pos, -1));
        TEST_ASSERT(backend.snapshot_used(6));
        TEST_ASSERT(backend.snapshots_[6].owns_storage);
    }

    for (int i = 0; i < 8; ++i) backend.snapshot_free(i);
    remove_test_disk_cache_dir(dir);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_layer_split_snapshot_disk_roundtrip() {
    std::fprintf(stderr, "  test_layer_split_snapshot_disk_roundtrip ...");

    auto adapter = make_test_adapter();
    TEST_ASSERT(init_snapshot_test_shard(adapter));
    if (adapter.shards_.empty() || !adapter.shards_[0].backend) {
        std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
        return;
    }

    auto & cache = adapter.shards_[0].cache;
    auto & layer = cache.layers[0];
    write_tensor_pattern(layer.raw_kv, 7);
    write_tensor_pattern(layer.comp_kv, 19);
    write_tensor_pattern(layer.index_comp_kv, 31);
    write_tensor_pattern(layer.attn_compressor.state_kv, 47);
    write_tensor_pattern(layer.attn_compressor.state_score, 59);
    write_tensor_pattern(layer.indexer_compressor.state_kv, 71);
    write_tensor_pattern(layer.indexer_compressor.state_score, 83);
    write_tensor_pattern(cache.hc_state, 97);
    const auto raw_before = read_tensor_bytes(layer.raw_kv);
    const auto comp_before = read_tensor_rows(layer.comp_kv, 5);
    const auto index_before = read_tensor_rows(layer.index_comp_kv, 3);
    const auto hc_before = read_tensor_bytes(cache.hc_state);

    layer.n_comp = 5;
    layer.n_index_comp = 3;
    cache.cur_pos = 7;
    adapter.cur_pos_ = 7;
    adapter.last_tok_ = 4242;
    adapter.hc_state_ = {1.0f, 2.0f, 3.0f, 4.0f};
    adapter.prefill_last_logits_ = {9.0f, 8.0f, 7.0f};

    TEST_ASSERT(adapter.snapshot_save(0));
    const ModelBackend::SnapshotRef ref = adapter.snapshot_ref(0);
    TEST_ASSERT(ref.ctx != nullptr);
    TEST_ASSERT(ref.cur_pos == 7);
    TEST_ASSERT(ref.last_tok == 4242);
    // The shard snapshot aliases the slot's single merged context: no copy.
    TEST_ASSERT(adapter.snapshots_[0].shards[0].ctx == ref.ctx);
    TEST_ASSERT(!adapter.snapshots_[0].shards[0].owns_storage);
    size_t n_named = 0;
    TEST_ASSERT(all_snapshot_tensors_named(ref.ctx, &n_named));
    // shard: hc + 7 layer + meta + logits + feat = 11, plus 3 adapter tensors
    TEST_ASSERT(n_named == 14);
    TEST_ASSERT(ggml_get_tensor(ref.ctx, "ls0_ds4_snap_comp_kv_0") != nullptr);
    TEST_ASSERT(ggml_get_tensor(ref.ctx, "ls_meta") != nullptr);

    // In-memory restore works from the merged context.
    cache.cur_pos = 0;
    layer.n_comp = 0;
    TEST_ASSERT(adapter.snapshot_restore(0));
    TEST_ASSERT(cache.cur_pos == 7);
    TEST_ASSERT(layer.n_comp == 5);

    // Simulate the ondisk roundtrip (serialize -> deserialize) by rebuilding
    // the merged context exactly like DiskPrefixCache::read_file() does.
    ggml_context * re_ctx = nullptr;
    ggml_backend_buffer_t re_buf = nullptr;
    TEST_ASSERT(clone_snapshot_context_like_disk_reader(ref.ctx, &re_ctx, &re_buf));
    if (!re_ctx) {
        std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
        return;
    }
    // A geometry mismatch inside one shard is rejected before adoption.
    {
        ggml_context * bad_ctx = nullptr;
        ggml_backend_buffer_t bad_buf = nullptr;
        TEST_ASSERT(clone_snapshot_context_like_disk_reader(
            ref.ctx, &bad_ctx, &bad_buf, "ls0_ds4_snap_attn_cs_kv_0", 3));
        TEST_ASSERT(!adapter.snapshot_adopt(3, bad_ctx, bad_buf, 7, 4242));
        TEST_ASSERT(!adapter.snapshot_used(3));
        ggml_backend_buffer_free(bad_buf);
        ggml_free(bad_ctx);
    }

    adapter.snapshot_free(0);
    TEST_ASSERT(!adapter.snapshot_used(0));
    TEST_ASSERT(adapter.snapshots_[0].ctx == nullptr);
    ggml_backend_buffer_clear(cache.buf, 0);
    cache.cur_pos = 0;
    layer.n_comp = 0;
    layer.n_index_comp = 0;
    adapter.cur_pos_ = 0;
    adapter.last_tok_ = -1;
    adapter.hc_state_.assign(4, 0.0f);
    adapter.prefill_last_logits_.clear();

    // A context with the wrong position is rejected and left to the caller.
    TEST_ASSERT(!adapter.snapshot_adopt(2, re_ctx, re_buf, 6, 4242));
    TEST_ASSERT(!adapter.snapshot_used(2));
    TEST_ASSERT(adapter.snapshot_adopt(2, re_ctx, re_buf, 7, 4242));
    TEST_ASSERT(adapter.snapshot_used(2));
    TEST_ASSERT(adapter.snapshot_cur_pos(2) == 7);
    TEST_ASSERT(adapter.snapshots_[2].shards.size() == 1);
    TEST_ASSERT(!adapter.snapshots_[2].shards[0].owns_storage);
    TEST_ASSERT(adapter.snapshots_[2].ctx == re_ctx);
    // Re-exportable after adoption.
    TEST_ASSERT(adapter.snapshot_ref(2).ctx != nullptr);

    TEST_ASSERT(adapter.snapshot_restore(2));
    TEST_ASSERT(adapter.cur_pos_ == 7);
    TEST_ASSERT(adapter.last_tok_ == 4242);
    TEST_ASSERT(adapter.hc_state_ == std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}));
    TEST_ASSERT(adapter.prefill_last_logits_ == std::vector<float>({9.0f, 8.0f, 7.0f}));
    TEST_ASSERT(cache.cur_pos == 7);
    TEST_ASSERT(layer.n_comp == 5);
    TEST_ASSERT(layer.n_index_comp == 3);
    TEST_ASSERT(read_tensor_bytes(layer.raw_kv) == raw_before);
    TEST_ASSERT(read_tensor_rows(layer.comp_kv, 5) == comp_before);
    TEST_ASSERT(read_tensor_rows(layer.index_comp_kv, 3) == index_before);
    TEST_ASSERT(read_tensor_bytes(cache.hc_state) == hc_before);

    // Freeing an adopted slot must not double-free the shared context.
    adapter.snapshot_free(2);
    TEST_ASSERT(!adapter.snapshot_used(2));
    TEST_ASSERT(adapter.snapshots_[2].ctx == nullptr);
    TEST_ASSERT(adapter.snapshots_[2].shards[0].ctx == nullptr);

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_spec_feature_tail_is_bounded() {
    std::fprintf(stderr, "  test_spec_feature_tail_is_bounded ...");

    DeepSeek4BackendConfig cfg;
    DeepSeek4Backend backend(cfg);
    backend.w_.n_embd = 2;
    backend.w_.n_swa = 3;
    backend.spec_drafter_ = std::make_unique<DSparkDrafter>();
    backend.spec_drafter_->n_target_layers = 1;

    std::vector<float> features = {
        0.0f, 1.0f,
        2.0f, 3.0f,
        4.0f, 5.0f,
        6.0f, 7.0f,
        8.0f, 9.0f,
    };
    backend.keep_spec_feature_tail(features, 3);
    TEST_ASSERT(features ==
                std::vector<float>({4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}));

    features.push_back(10.0f);  // malformed partial feature row
    backend.keep_spec_feature_tail(features, 3);
    TEST_ASSERT(features.empty());

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_dspark_prefill_capture_boundaries() {
    std::fprintf(stderr, "  test_dspark_prefill_capture_boundaries ...");

    using Backend = DeepSeek4Backend;
    // Both monolithic layer-major and sparse heterogeneous prefill return the
    // per-token capture rows needed to retain a tail from one wide graph.
    TEST_ASSERT(Backend::supports_batched_spec_feature_capture(
                    false, PrefillAttentionMode::Sparse, 2048));
    TEST_ASSERT(Backend::supports_batched_spec_feature_capture(
                    true, PrefillAttentionMode::Sparse, 2048));
    TEST_ASSERT(!Backend::supports_batched_spec_feature_capture(
                    true, PrefillAttentionMode::Dense, 2048));
    TEST_ASSERT(!Backend::supports_batched_spec_feature_capture(
                    true, PrefillAttentionMode::Exact, 2048));
    TEST_ASSERT(!Backend::supports_batched_spec_feature_capture(
                    true, PrefillAttentionMode::Sparse, 4));
    TEST_ASSERT(!Backend::supports_batched_spec_feature_capture(
                    true, PrefillAttentionMode::Sparse,
                    DS4_MAX_LAYER_MAJOR_PREFILL_TOKENS + 1));

    // Generic paths still stop exactly at the final feature window.
    TEST_ASSERT(Backend::capture_safe_prefill_tokens(
                    0, 2048, 1920, true, false, 0, 0) == 2048);
    TEST_ASSERT(Backend::capture_safe_prefill_tokens(
                    0, 2048, 1920, false, false, 0, 0) == 1920);
    TEST_ASSERT(Backend::capture_safe_prefill_tokens(
                    1920, 128, 1920, false, false, 0, 0) == 128);

    // A pending checkpoint contributes both edges of its capture window. The
    // resulting batches are either wholly hooked or wholly unhooked.
    TEST_ASSERT(Backend::capture_safe_prefill_tokens(
                    0, 2048, 1920, true, true, 384, 512) == 384);
    TEST_ASSERT(Backend::capture_safe_prefill_tokens(
                    384, 1664, 1920, true, true, 384, 512) == 128);
    TEST_ASSERT(Backend::capture_safe_prefill_tokens(
                    512, 1536, 1920, false, false, 384, 512) == 1408);

    // Boundaries at a batch edge and empty requests need no extra split.
    TEST_ASSERT(Backend::capture_safe_prefill_tokens(
                    0, 128, 128, false, true, 128, 256) == 128);
    TEST_ASSERT(Backend::capture_safe_prefill_tokens(
                    10, 0, 20, false, true, 12, 18) == 0);

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_reset_request_state() {
    std::fprintf(stderr, "  test_reset_request_state ...");

    auto adapter = make_test_adapter();
    TEST_ASSERT(init_snapshot_test_shard(adapter));
    if (adapter.shards_.empty() || !adapter.shards_[0].cache.buf) {
        std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
        return;
    }

    adapter.cur_pos_ = 9;
    adapter.last_tok_ = 77;
    adapter.hc_state_.assign(8, 5.0f);
    adapter.prefill_last_logits_ = {1.0f, 2.0f};
    for (auto & shard : adapter.shards_) {
        shard.cache.cur_pos = 11;
        for (auto & layer : shard.cache.layers) {
            layer.n_comp = 4;
            layer.n_index_comp = 6;
            if (layer.raw_kv) write_tensor_pattern(layer.raw_kv, 23);
            if (layer.comp_kv) write_tensor_pattern(layer.comp_kv, 31);
            if (layer.attn_compressor.state_kv) {
                write_tensor_pattern(layer.attn_compressor.state_kv, 47);
            }
        }
    }

    adapter.reset_request_state();
    TEST_ASSERT(adapter.cur_pos_ == 0);
    TEST_ASSERT(adapter.last_tok_ == -1);
    for (float v : adapter.hc_state_) {
        TEST_ASSERT(v == 0.0f);
    }
    TEST_ASSERT(adapter.prefill_last_logits_.empty());
    for (const auto & shard : adapter.shards_) {
        TEST_ASSERT(shard.cache.cur_pos == 0);
        for (const auto & layer : shard.cache.layers) {
            TEST_ASSERT(layer.n_comp == 0);
            TEST_ASSERT(layer.n_index_comp == 0);
            for (const ggml_tensor * tensor : {
                     layer.raw_kv,
                     layer.comp_kv,
                     layer.attn_compressor.state_kv}) {
                if (!tensor) continue;
                const std::vector<uint8_t> bytes = read_tensor_bytes(tensor);
                TEST_ASSERT(std::all_of(bytes.begin(), bytes.end(),
                                        [](uint8_t value) { return value == 0; }));
            }
        }
    }

    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_reset_deepseek4_cache(ggml_backend_t backend) {
    std::fprintf(stderr, "  test_reset_deepseek4_cache ...");

    DeepSeek4Weights weights;
    weights.n_layer = 2;
    weights.n_swa = 8;
    weights.head_dim = 4;
    weights.n_hc = 4;
    weights.n_embd = 8;
    weights.compress_ratios = {4, 0};

    DeepSeek4Cache cache;
    TEST_ASSERT(create_deepseek4_cache(backend, weights, 32, cache));
    if (cache.buf) {
        ggml_backend_buffer_clear(cache.buf, 0x7f);
        cache.cur_pos = 17;
        cache.layers[0].n_comp = 4;
        cache.layers[0].n_index_comp = 4;
        cache.layers[1].n_comp = 3;
        cache.layers[1].n_index_comp = 2;

        reset_deepseek4_cache(cache);

        TEST_ASSERT(cache.cur_pos == 0);
        for (const auto & layer : cache.layers) {
            TEST_ASSERT(layer.n_comp == 0);
            TEST_ASSERT(layer.n_index_comp == 0);
            const std::vector<uint8_t> bytes = read_tensor_bytes(layer.raw_kv);
            TEST_ASSERT(std::all_of(bytes.begin(), bytes.end(),
                                    [](uint8_t value) { return value == 0; }));
        }

        reset_deepseek4_cache(cache);
        TEST_ASSERT(cache.cur_pos == 0);
    }
    free_deepseek4_cache(cache);

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
    TEST_ASSERT(!adapter.decode_ar(1, 0, 1, {}, out_tokens, DaemonIO{}));

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
static void test_ds4_flash_attention_keep_cap_gpu() {
    std::fprintf(stderr, "  test_ds4_flash_attention_keep_cap_gpu ...");
#if !defined(GGML_USE_HIP)
    std::fprintf(stderr, " skipped (HIP-only contract)\n");
    return;
#endif
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, " skipped (no GPU backend)\n");
        return;
    }

    constexpr int head_dim = 512;
    constexpr int n_heads = 4;
    constexpr int n_tokens = 1;
    constexpr int raw_rows = 128;
    constexpr int n_comp_rows = 8;
    constexpr int n_kv = raw_rows + n_comp_rows;
    constexpr int configured_keep_rows = 512;

    ggml_context * ctx = make_test_context(2u << 20);
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) {
        ggml_backend_free(backend);
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    ggml_tensor * q = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, head_dim, n_tokens, n_heads);
    ggml_tensor * kv = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, head_dim, n_kv, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F16, n_kv, n_tokens);
    ggml_tensor * output = ggml_flash_attn_ext(
        ctx, q, kv, kv, mask, 1.0f / std::sqrt((float) head_dim),
        0.0f, 0.0f);
    ggml_flash_attn_ext_set_ds4_sparse(
        output, raw_rows, raw_rows, configured_keep_rows, 32);
    ggml_set_output(output);
    TEST_ASSERT_MSG(ggml_backend_supports_op(backend, output),
                    "GPU rejected a DS4 keep cap larger than live history");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    const bool allocated = ggml_gallocr_alloc_graph(alloc, graph);
    TEST_ASSERT_MSG(allocated, "DS4 flash-attention graph allocation failed");
    if (allocated) {
        std::vector<float> q_data((size_t) head_dim * n_tokens * n_heads);
        std::vector<float> kv_data((size_t) head_dim * n_kv);
        std::vector<ggml_fp16_t> mask_data((size_t) n_kv * n_tokens,
                                           ggml_fp32_to_fp16(0.0f));
        for (size_t i = 0; i < q_data.size(); ++i) {
            q_data[i] = ((int) (i % 19) - 9) * 0.002f;
        }
        for (size_t i = 0; i < kv_data.size(); ++i) {
            kv_data[i] = ((int) (i % 23) - 11) * 0.002f;
        }
        ggml_backend_tensor_set(q, q_data.data(), 0,
                                q_data.size() * sizeof(float));
        ggml_backend_tensor_set(kv, kv_data.data(), 0,
                                kv_data.size() * sizeof(float));
        ggml_backend_tensor_set(mask, mask_data.data(), 0,
                                mask_data.size() * sizeof(ggml_fp16_t));
        const bool computed =
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        TEST_ASSERT_MSG(computed, "DS4 flash-attention graph compute failed");
        if (computed) {
            std::vector<float> output_data((size_t) ggml_nelements(output));
            ggml_backend_tensor_get(output, output_data.data(), 0,
                                    output_data.size() * sizeof(float));
            for (float value : output_data) {
                TEST_ASSERT_MSG(std::isfinite(value),
                                "DS4 flash-attention output must be finite");
            }
        }
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_ds4_flash_attention_parallel_index_scan_gpu(int selected_rows) {
    std::fprintf(stderr,
                 "  test_ds4_flash_attention_parallel_index_scan_gpu(%d) ...", selected_rows);
#if !defined(GGML_USE_HIP)
    std::fprintf(stderr, " skipped (HIP-only contract)\n");
    return;
#endif
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, " skipped (no GPU backend)\n");
        return;
    }

    constexpr int head_dim = 512;
    constexpr int n_heads = 4;
    constexpr int n_tokens = 2;
    constexpr int raw_rows = 256;
    constexpr int raw_window = 128;
    // Keep the ordinary four-head shared-memory footprint above 24 KiB so
    // this shape is forced through the compact indexed path under test.
    const int n_comp_rows = 2 * selected_rows + 256;
    const int n_kv = raw_rows + n_comp_rows;

    ggml_context * ctx = make_test_context(4u << 20);
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) {
        ggml_backend_free(backend);
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    ggml_tensor * q = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, head_dim, n_tokens, n_heads);
    ggml_tensor * kv = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, head_dim, n_kv, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F16, n_kv, n_tokens);
    ggml_tensor * direct_mask = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F16, n_kv, n_tokens);
    ggml_tensor * direct_topk = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, selected_rows, n_tokens);
    ggml_tensor * output = ggml_flash_attn_ext(
        ctx, q, kv, kv, mask, 1.0f / std::sqrt((float) head_dim),
        0.0f, 0.0f);
    // Negative keep_rows marks an exact externally indexed mask. This shape
    // enters the compact four-head HIP path and exceeds the parallel threshold.
    ggml_flash_attn_ext_set_ds4_sparse(
        output, raw_rows, raw_window, -selected_rows, 1);
    ggml_tensor * direct_output = ggml_flash_attn_ext(
        ctx, q, kv, kv, direct_mask,
        1.0f / std::sqrt((float) head_dim), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_ds4_sparse(
        direct_output, raw_rows, raw_window, -selected_rows, 1);
    ggml_flash_attn_ext_set_ds4_indexer_topk(direct_output, direct_topk);
    ggml_set_output(output);
    ggml_set_output(direct_output);
    TEST_ASSERT_MSG(ggml_backend_supports_op(backend, output),
                    "GPU rejected exact indexed DS4 attention");
    TEST_ASSERT_MSG(ggml_backend_supports_op(backend, direct_output),
                    "GPU rejected direct-top-k DS4 attention");

    ggml_tensor * short_kv = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, head_dim, raw_rows + 128, 1);
    ggml_tensor * short_mask = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F16, raw_rows + 128, n_tokens);
    ggml_tensor * oversized_topk = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, selected_rows, n_tokens);
    ggml_tensor * oversized_output = ggml_flash_attn_ext(
        ctx, q, short_kv, short_kv, short_mask,
        1.0f / std::sqrt((float) head_dim), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_ds4_sparse(
        oversized_output, raw_rows, raw_window, -selected_rows, 1);
    ggml_flash_attn_ext_set_ds4_indexer_topk(
        oversized_output, oversized_topk);
    TEST_ASSERT_MSG(
        !ggml_backend_supports_op(backend, oversized_output),
        "GPU accepted direct top-k wider than the live compressed span");

    ggml_tensor * over_capacity_topk = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, 1025, n_tokens);
    ggml_tensor * over_capacity_output = ggml_flash_attn_ext(
        ctx, q, kv, kv, direct_mask,
        1.0f / std::sqrt((float) head_dim), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_ds4_sparse(
        over_capacity_output, raw_rows, raw_window, -1025, 1);
    ggml_flash_attn_ext_set_ds4_indexer_topk(
        over_capacity_output, over_capacity_topk);
    TEST_ASSERT_MSG(
        !ggml_backend_supports_op(backend, over_capacity_output),
        "GPU accepted direct top-k wider than the sorting capacity");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    ggml_build_forward_expand(graph, direct_output);
    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    const bool allocated = ggml_gallocr_alloc_graph(alloc, graph);
    TEST_ASSERT_MSG(allocated, "indexed attention graph allocation failed");
    if (allocated) {
        std::vector<float> q_data((size_t) head_dim * n_tokens * n_heads);
        std::vector<float> kv_data((size_t) head_dim * n_kv);
        std::vector<ggml_fp16_t> mask_data(
            (size_t) n_kv * n_tokens, ggml_fp32_to_fp16(-1.0e30f));
        std::vector<ggml_fp16_t> direct_mask_data(
            (size_t) n_kv * n_tokens, ggml_fp32_to_fp16(-1.0e30f));
        std::vector<int32_t> direct_topk_data(
            (size_t) selected_rows * n_tokens);
        for (size_t i = 0; i < q_data.size(); ++i) {
            q_data[i] = ((int) (i % 31) - 15) * 0.001f;
        }
        for (size_t i = 0; i < kv_data.size(); ++i) {
            kv_data[i] = ((int) (i % 37) - 18) * 0.001f;
        }
        for (int token = 0; token < n_tokens; ++token) {
            ggml_fp16_t * token_mask =
                mask_data.data() + (size_t) token * n_kv;
            ggml_fp16_t * token_direct_mask =
                direct_mask_data.data() + (size_t) token * n_kv;
            for (int row = raw_rows - raw_window; row < raw_rows; ++row) {
                token_mask[row] = ggml_fp32_to_fp16(0.0f);
                token_direct_mask[row] = ggml_fp32_to_fp16(0.0f);
            }
            for (int row = token; row < n_comp_rows; row += 2) {
                token_mask[raw_rows + row] = ggml_fp32_to_fp16(0.0f);
            }
            for (int row = 0; row < n_comp_rows; ++row) {
                token_direct_mask[raw_rows + row] =
                    ggml_fp32_to_fp16(0.0f);
            }
            // The model's top-k is score ordered. Reverse the physical order
            // here so this test proves that the direct path restores the old
            // ascending accumulation order rather than merely accepting an
            // already sorted fixture.
            for (int rank = 0; rank < selected_rows; ++rank) {
                direct_topk_data[(size_t) token * selected_rows + rank] =
                    token + 2 * (selected_rows - 1 - rank);
            }
        }
        ggml_backend_tensor_set(q, q_data.data(), 0,
                                q_data.size() * sizeof(float));
        ggml_backend_tensor_set(kv, kv_data.data(), 0,
                                kv_data.size() * sizeof(float));
        ggml_backend_tensor_set(mask, mask_data.data(), 0,
                                mask_data.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(
            direct_mask, direct_mask_data.data(), 0,
            direct_mask_data.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(
            direct_topk, direct_topk_data.data(), 0,
            direct_topk_data.size() * sizeof(int32_t));

        const char * previous_serial =
            std::getenv("GGML_DS4_FA_SERIAL_INDEX_SCAN");
        const std::string previous_value = previous_serial
            ? previous_serial : "";
        std::vector<float> serial((size_t) ggml_nelements(output));
        std::vector<float> parallel(serial.size());
        std::vector<float> direct(serial.size());
        {
            setenv("GGML_DS4_FA_SERIAL_INDEX_SCAN", "1", 1);
            ScopedCudaGraphOverrides eager(
                /*disable_graphs=*/true,
                /*mmvq_max_ncols=*/0,
                /*skip_property_check=*/false);
            TEST_ASSERT_MSG(
                ggml_backend_graph_compute(backend, graph) ==
                    GGML_STATUS_SUCCESS,
                "serial indexed attention failed");
            ggml_backend_tensor_get(output, serial.data(), 0,
                                    serial.size() * sizeof(float));
        }
        {
            unsetenv("GGML_DS4_FA_SERIAL_INDEX_SCAN");
            ScopedCudaGraphOverrides eager(
                /*disable_graphs=*/true,
                /*mmvq_max_ncols=*/0,
                /*skip_property_check=*/false);
            TEST_ASSERT_MSG(
                ggml_backend_graph_compute(backend, graph) ==
                    GGML_STATUS_SUCCESS,
                "parallel indexed attention failed");
            ggml_backend_tensor_get(output, parallel.data(), 0,
                                    parallel.size() * sizeof(float));
            ggml_backend_tensor_get(direct_output, direct.data(), 0,
                                    direct.size() * sizeof(float));
        }
        if (previous_serial) {
            setenv("GGML_DS4_FA_SERIAL_INDEX_SCAN",
                   previous_value.c_str(), 1);
        } else {
            unsetenv("GGML_DS4_FA_SERIAL_INDEX_SCAN");
        }
        for (size_t i = 0; i < serial.size(); ++i) {
            TEST_ASSERT_MSG(
                nearly_equal(serial[i], parallel[i], 1.0e-6f, 1.0e-6f),
                "parallel index scan changed attention output");
            TEST_ASSERT_MSG(
                serial[i] == direct[i],
                "direct top-k changed attention output");
        }
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_ds4_saved_raw_rows_replay_gpu() {
    std::fprintf(stderr, "  test_ds4_saved_raw_rows_replay_gpu ...");
#if !defined(GGML_USE_HIP)
    std::fprintf(stderr, " skipped (HIP-only replay qualification)\n");
    return;
#endif
    auto * backend = ggml_backend_cuda_init(0);
    TEST_ASSERT_MSG(backend != nullptr, "GPU backend unavailable");
    if (!backend) return;
    constexpr int dim = 512, ring_rows = 128;
    int passed = 0, cases = 0;
    for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (int width = 2; width <= 5; ++width) {
            auto * ctx = make_test_context(4u << 20);
            TEST_ASSERT(ctx != nullptr);
            if (!ctx) continue;
            auto * ring = ggml_new_tensor_2d(ctx, type, dim, ring_rows);
            auto * read_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, width);
            auto * write_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, width);
            auto * replacement = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, width);
            for (auto * input : {ring, read_rows, write_rows, replacement}) {
                ggml_set_input(input);
            }
            auto * saved = deepseek4_preserve_raw_rows(ctx, ring, read_rows);
            TEST_ASSERT(saved->type == type);
            ggml_set_output(saved);
            auto * graph = ggml_new_graph_custom(ctx, 128, false);
            ggml_build_forward_expand(graph, saved);
            auto * updated = ggml_set_rows(ctx, ring, replacement, write_rows);
            ggml_set_output(updated);
            ggml_build_forward_expand(graph, updated);
            auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            const bool allocated = ggml_gallocr_alloc_graph(alloc, graph);
            TEST_ASSERT(allocated);
            if (allocated) {
                std::vector<float> original((size_t) dim * ring_rows);
                for (int r = 0; r < ring_rows; ++r) {
                    for (int d = 0; d < dim; ++d) {
                        original[(size_t) r * dim + d] = (float) r + (d % 4) * 0.25f;
                    }
                }
                std::vector<ggml_fp16_t> original_f16(original.size());
                std::transform(original.begin(), original.end(), original_f16.begin(), ggml_fp32_to_fp16);
                std::vector<float> new_values((size_t) dim * width, -32.0f);
                std::vector<int32_t> reads(width);
                std::vector<int64_t> writes(width);
                for (int position : {124, 124, 127, 128, 132, 255, 7680, 131072, 124}) {
                    for (int t = 0; t < width; ++t) reads[t] = (position + t) % ring_rows;
                    std::copy(reads.begin(), reads.end(), writes.begin());
                    ggml_backend_tensor_set(ring,
                        type == GGML_TYPE_F16 ? (const void *) original_f16.data() : original.data(),
                        0, ggml_nbytes(ring));
                    ggml_backend_tensor_set(read_rows, reads.data(), 0, ggml_nbytes(read_rows));
                    ggml_backend_tensor_set(write_rows, writes.data(), 0, ggml_nbytes(write_rows));
                    ggml_backend_tensor_set(replacement, new_values.data(), 0, ggml_nbytes(replacement));
                    ScopedCudaGraphOverrides replay(
                        /*disable_graphs=*/false, /*mmvq_max_ncols=*/0,
                        /*skip_property_check=*/true);
                    bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
                    std::vector<float> actual((size_t) dim * width), actual_ring(original.size());
                    if (ok && type == GGML_TYPE_F16) {
                        std::vector<ggml_fp16_t> half_saved(actual.size()), half_ring(actual_ring.size());
                        ggml_backend_tensor_get(saved, half_saved.data(), 0, ggml_nbytes(saved));
                        ggml_backend_tensor_get(updated, half_ring.data(), 0, ggml_nbytes(updated));
                        std::transform(half_saved.begin(), half_saved.end(), actual.begin(), ggml_fp16_to_fp32);
                        std::transform(half_ring.begin(), half_ring.end(), actual_ring.begin(), ggml_fp16_to_fp32);
                    } else if (ok) {
                        ggml_backend_tensor_get(saved, actual.data(), 0, ggml_nbytes(saved));
                        ggml_backend_tensor_get(updated, actual_ring.data(), 0, ggml_nbytes(updated));
                    }
                    for (int t = 0; t < width && ok; ++t) {
                        for (int d = 0; d < dim; ++d) {
                            ok &= actual[(size_t) t * dim + d] == original[(size_t) reads[t] * dim + d];
                        }
                    }
                    for (int r = 0; r < ring_rows && ok; ++r) {
                        const bool replaced = std::find(reads.begin(), reads.end(), r) != reads.end();
                        for (int d = 0; d < dim; ++d) {
                            ok &= actual_ring[(size_t) r * dim + d] ==
                                (replaced ? -32.0f : original[(size_t) r * dim + d]);
                        }
                    }
                    ++cases;
                    passed += ok;
                    if (!ok) {
                        std::fprintf(stderr, " FAIL q=%d type=%s pos=%d;",
                                     width, ggml_type_name(type), position);
                    }
                }
            }
            ggml_backend_cuda_graph_invalidate_range(
                backend, ggml_get_mem_buffer(ctx), ggml_get_mem_size(ctx));
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
        }
    }
    ggml_backend_free(backend);
    std::fprintf(stderr, " %d/%d cases passed\n", passed, cases);
    TEST_ASSERT_MSG(cases == 72 && passed == cases, "cached verifier saved rows from a stale ring position");
}

static bool run_ds4_preserved_raw_rows_case(
        ggml_backend_t backend, int width, int compressed_rows,
        bool visible_suffix, ggml_type kv_type, bool direct, int replays = 1) {
    constexpr int dim = 512, heads = 64, raw_rows = 128, top_k = 512;
    const int saved = width > 1 ? width : 0;
    const int rows = raw_rows + compressed_rows + saved;
    ggml_context * ctx = make_test_context(4u << 20);
    if (!ctx) return false;
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, width, heads);
    auto * kv = ggml_new_tensor_3d(ctx, kv_type, dim, rows, 1);
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, width);
    auto * topk = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, width);
    for (auto * input : {q, kv, mask, topk}) ggml_set_input(input);
    auto * selected = deepseek4_indexed_attention_rows(
        ctx, topk, compressed_rows, saved);
    auto * indexed_mask = ggml_ds4_indexer_mask(ctx, mask, selected, raw_rows);
    ggml_set_output(indexed_mask);
    auto * result = ggml_flash_attn_ext(ctx, q, kv, kv,
        ggml_cast(ctx, direct ? mask : indexed_mask, GGML_TYPE_F16),
        1.0f / std::sqrt(float(dim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(result, GGML_PREC_F32);
    ggml_flash_attn_ext_set_ds4_sparse(result, raw_rows, raw_rows,
                                     -(int) selected->ne[0], 32);
    if (direct) ggml_flash_attn_ext_set_ds4_indexer_topk(result, selected);
    ggml_set_output(result);
    auto * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, indexed_mask);
    ggml_build_forward_expand(graph, result);
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = ggml_backend_supports_op(backend, result) &&
              ggml_gallocr_alloc_graph(alloc, graph);
    if (ok) {
        std::vector<float> qdata((size_t) dim * width * heads, 0.0f);
        std::vector<float> kvdata((size_t) dim * rows, 0.0f);
        for (int r = 0; r < rows; ++r) {
            kvdata[(size_t) dim * r] = r < raw_rows + compressed_rows ? 0.25f : 1.25f;
        }
        std::vector<float> maskdata((size_t) rows * width, -1.0e30f);
        std::vector<int32_t> iddata((size_t) top_k * width);
        for (int t = 0; t < width; ++t) {
            float * col = maskdata.data() + (size_t) t * rows;
            std::fill(col, col + raw_rows, 0.0f);
            if (visible_suffix) {
                // Later lanes overwrite these ring slots; earlier lanes need
                // their saved values, while the future writes stay hidden.
                for (int r = t + 1; r < width; ++r) col[r] = -1.0e30f;
                for (int s = t + 1; s < saved; ++s) col[raw_rows + compressed_rows + s] = 0.0f;
            }
            for (int c = 0; c < top_k; ++c) {
                // Non-monotone, separated indices exercise sorting and both
                // mask scanners. Some selected rows remain causally hidden.
                const int row = (29 * c + 17 * t) % compressed_rows;
                if (c < top_k - t % 3) col[raw_rows + row] = 0.0f;
                iddata[(size_t) t * top_k + c] = row;
            }
        }
        std::vector<ggml_fp16_t> half;
        if (kv_type == GGML_TYPE_F16) {
            half.resize(kvdata.size());
            std::transform(kvdata.begin(), kvdata.end(), half.begin(), ggml_fp32_to_fp16);
        }
        for (int replay = 0; ok && replay < replays; ++replay) {
            // Gallocr may reuse an input's storage after its last consumer.
            // Refresh fixture inputs on every launch, as the caller must.
            ggml_backend_tensor_set(q, qdata.data(), 0, ggml_nbytes(q));
            if (kv_type == GGML_TYPE_F16) {
                ggml_backend_tensor_set(kv, half.data(), 0, ggml_nbytes(kv));
            } else {
                ggml_backend_tensor_set(kv, kvdata.data(), 0, ggml_nbytes(kv));
            }
            ggml_backend_tensor_set(mask, maskdata.data(), 0, ggml_nbytes(mask));
            ggml_backend_tensor_set(topk, iddata.data(), 0, ggml_nbytes(topk));
            ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
            if (ok) {
                std::vector<float> actual(ggml_nelements(result)), actual_mask(maskdata.size());
                ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
                ggml_backend_tensor_get(indexed_mask, actual_mask.data(), 0, ggml_nbytes(indexed_mask));
                for (int t = 0; t < width; ++t) {
                    // Zero queries give equal attention weights. Keeping the saved
                    // suffix replaces exactly the hidden future ring slots.
                    const int visible = visible_suffix ? std::max(0, saved - t - 1) : 0;
                    const float expected = 0.25f + float(visible) / (raw_rows + top_k - t % 3);
                    for (int h = 0; h < heads; ++h) {
                        for (int d = 0; d < dim; ++d) {
                            const float value = actual[((size_t) t * heads + h) * dim + d];
                            const float reference = d == 0 ? expected : 0.0f;
                            const bool matches = std::isfinite(value) &&
                                std::abs(value - reference) < 1.0e-6f;
                            if (ok && !matches) {
                                std::fprintf(stderr,
                                    " saved-row output mismatch replay=%d token=%d head=%d dim=%d actual=%.9g expected=%.9g\n",
                                    replay, t, h, d, value, reference);
                            }
                            ok &= matches;
                        }
                    }
                    for (int r = 0; r < rows; ++r) {
                        const size_t at = (size_t) t * rows + r;
                        const bool matches = (actual_mask[at] > -1.0e20f) == (maskdata[at] > -1.0e20f);
                        if (ok && !matches) {
                            std::fprintf(stderr,
                                " saved-row mask mismatch replay=%d token=%d row=%d actual=%.9g expected=%.9g\n",
                                replay, t, r, actual_mask[at], maskdata[at]);
                        }
                        ok &= matches;
                    }
                }
            }
        }
    }
    if (!ok) std::fprintf(stderr, " saved-row FAIL q=%d comp=%d visible=%d kv=%s direct=%d\n",
                          width, compressed_rows, visible_suffix, ggml_type_name(kv_type), direct);
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}

static void test_ds4_preserved_raw_rows_gpu() {
    std::fprintf(stderr, "  test_ds4_preserved_raw_rows_gpu ...");
#if !defined(GGML_USE_HIP)
    std::fprintf(stderr, " skipped (HIP-only indexed attention)\n");
    return;
#endif
    auto * backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, " skipped (no GPU backend)\n");
        return;
    }
    int passed = 0, cases = 0;
    for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (int comp : {2048, 30720}) {
            for (int width = 1; width <= 5; ++width) {
                for (bool visible : {false, true}) {
                    for (bool direct : {false, true}) {
                        ++cases;
                        passed += run_ds4_preserved_raw_rows_case(backend, width, comp, visible, type, direct);
                    }
                }
            }
        }
    }
    std::fprintf(stderr, " %d/%d cases passed\n", passed, cases);
    TEST_ASSERT_MSG(passed == cases, "indexed attention lost saved raw KV rows or exposed future rows");
    // CI exposed an intermittent failure in this long-context, multi-wave
    // shape. Exercise repeated launches/replays without relaxing parity.
    TEST_ASSERT_MSG(run_ds4_preserved_raw_rows_case(
        backend, 5, 30720, true, GGML_TYPE_F16, false, 128),
        "indexed attention saved-row replay stress failed");
    ggml_backend_free(backend);
}

static void test_ds4_flash_attention_streaming_topk_gpu() {
    std::fprintf(stderr,
                 "  test_ds4_flash_attention_streaming_topk_gpu ...");
#if !defined(GGML_USE_HIP)
    std::fprintf(stderr, " skipped (HIP-only candidate)\n");
    return;
#else
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess || properties.warpSize != 32) {
        std::fprintf(stderr, " skipped (requires native wave32)\n");
        return;
    }
#endif
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, " skipped (no GPU backend)\n");
        return;
    }

    constexpr int head_dim = 512;
    constexpr int n_heads = 64;
    constexpr int n_tokens = 64;
    constexpr int raw_rows = 128;
    constexpr int raw_window = 128;
    constexpr int selected_rows = 512;
    constexpr int n_comp_rows = 1920;
    constexpr int n_kv = raw_rows + n_comp_rows;

    ggml_context * ctx = make_test_context(4u << 20);
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) {
        ggml_backend_free(backend);
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    ggml_tensor * q = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, head_dim, n_tokens, n_heads);
    ggml_tensor * kv = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F16, head_dim, n_kv, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F16, n_kv, n_tokens);
    ggml_tensor * topk = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, selected_rows, n_tokens);
    ggml_tensor * output = ggml_flash_attn_ext(
        ctx, q, kv, kv, mask, 1.0f / std::sqrt((float) head_dim),
        0.0f, 0.0f);
    ggml_flash_attn_ext_set_ds4_sparse(
        output, raw_rows, raw_window, -selected_rows, 1);
    ggml_flash_attn_ext_set_ds4_indexer_topk(output, topk);
    ggml_set_output(output);
    TEST_ASSERT_MSG(ggml_backend_supports_op(backend, output),
                    "GPU rejected streaming top-k attention fixture");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    const bool allocated = ggml_gallocr_alloc_graph(alloc, graph);
    TEST_ASSERT_MSG(allocated, "streaming top-k graph allocation failed");
    if (allocated) {
        std::vector<float> q_data(
            (size_t) head_dim * n_tokens * n_heads);
        std::vector<ggml_fp16_t> kv_data((size_t) head_dim * n_kv);
        std::vector<ggml_fp16_t> mask_data(
            (size_t) n_kv * n_tokens, ggml_fp32_to_fp16(-1.0e30f));
        std::vector<int32_t> topk_data(
            (size_t) selected_rows * n_tokens);
        for (size_t i = 0; i < q_data.size(); ++i) {
            q_data[i] = ((int) (i % 43) - 21) * 0.05f;
        }
        for (size_t i = 0; i < kv_data.size(); ++i) {
            kv_data[i] = ggml_fp32_to_fp16(
                ((int) (i % 47) - 23) * 0.05f);
        }
        for (int token = 0; token < n_tokens; ++token) {
            ggml_fp16_t * token_mask =
                mask_data.data() + (size_t) token * n_kv;
            for (int row = 0; row < raw_rows; ++row) {
                token_mask[row] = ggml_fp32_to_fp16(0.0f);
            }
            for (int row = 0; row < n_comp_rows; ++row) {
                token_mask[raw_rows + row] = ggml_fp32_to_fp16(0.0f);
            }
            for (int rank = 0; rank < selected_rows; ++rank) {
                topk_data[(size_t) token * selected_rows + rank] =
                    (token * 17 + selected_rows - 1 - rank) % n_comp_rows;
            }
        }
        ggml_backend_tensor_set(q, q_data.data(), 0,
                                q_data.size() * sizeof(float));
        ggml_backend_tensor_set(kv, kv_data.data(), 0,
                                kv_data.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(mask, mask_data.data(), 0,
                                mask_data.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(topk, topk_data.data(), 0,
                                topk_data.size() * sizeof(int32_t));

        ScopedEnvVar streaming_guard("GGML_CUDA_MLA_STREAM_TOPK");
        ScopedEnvVar f32_stage_guard("GGML_CUDA_MLA_STREAM_F32_STAGE");
        ScopedEnvVar fast_exp_guard("GGML_CUDA_MLA_STREAM_FAST_EXP");
        ScopedCudaGraphOverrides eager(
            /*disable_graphs=*/true,
            /*mmvq_max_ncols=*/0,
            /*skip_property_check=*/false);
        std::vector<float> reference((size_t) ggml_nelements(output));
        std::vector<float> candidate_f16_stage(reference.size());
        std::vector<float> candidate_f32_stage(reference.size());
        std::vector<float> candidate_fast_exp(reference.size());
        setenv("GGML_CUDA_MLA_STREAM_TOPK", "0", 1);
        setenv("GGML_CUDA_MLA_STREAM_FAST_EXP", "0", 1);
        const size_t streaming_launches = ggml_backend_cuda_get_mla_stream_topk_launch_count();
        TEST_ASSERT_MSG(
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "grouped compact attention reference failed");
        TEST_ASSERT(ggml_backend_cuda_get_mla_stream_topk_launch_count() == streaming_launches);
        ggml_backend_tensor_get(output, reference.data(), 0,
                                reference.size() * sizeof(float));

        setenv("GGML_CUDA_MLA_STREAM_TOPK", "1", 1);
        setenv("GGML_CUDA_MLA_STREAM_F32_STAGE", "0", 1);
        TEST_ASSERT_MSG(
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "F16-staged streaming top-k attention failed");
        TEST_ASSERT(ggml_backend_cuda_get_mla_stream_topk_launch_count() == streaming_launches + 1);
        ggml_backend_tensor_get(output, candidate_f16_stage.data(), 0,
                                candidate_f16_stage.size() * sizeof(float));

        setenv("GGML_CUDA_MLA_STREAM_F32_STAGE", "1", 1);
        TEST_ASSERT_MSG(
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "F32-staged streaming top-k attention failed");
        TEST_ASSERT(ggml_backend_cuda_get_mla_stream_topk_launch_count() == streaming_launches + 2);
        ggml_backend_tensor_get(output, candidate_f32_stage.data(), 0,
                                candidate_f32_stage.size() * sizeof(float));

        setenv("GGML_CUDA_MLA_STREAM_FAST_EXP", "1", 1);
        TEST_ASSERT_MSG(
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "fast-exp streaming top-k attention failed");
        TEST_ASSERT(ggml_backend_cuda_get_mla_stream_topk_launch_count() == streaming_launches + 3);
        ggml_backend_tensor_get(output, candidate_fast_exp.data(), 0,
                                candidate_fast_exp.size() * sizeof(float));

        double fast_exp_squared_error = 0.0;
        double f32_stage_power = 0.0;
        float fast_exp_max_abs = 0.0f;
        for (size_t i = 0; i < reference.size(); ++i) {
            TEST_ASSERT_MSG(
                std::isfinite(candidate_f32_stage[i]),
                "streaming top-k attention output must be finite");
            TEST_ASSERT_MSG(
                nearly_equal(reference[i], candidate_f32_stage[i],
                             5.0e-4f, 5.0e-4f),
                "streaming top-k attention exceeded numeric tolerance");
            TEST_ASSERT_MSG(
                candidate_f16_stage[i] == candidate_f32_stage[i],
                "F32 staging changed streaming top-k attention output");
            TEST_ASSERT_MSG(
                std::isfinite(candidate_fast_exp[i]),
                "fast-exp streaming output must be finite");
            TEST_ASSERT_MSG(
                nearly_equal(reference[i], candidate_fast_exp[i],
                             5.0e-4f, 5.0e-4f),
                "fast-exp streaming attention exceeded numeric tolerance");
            const double fast_exp_error =
                (double) candidate_fast_exp[i] - candidate_f32_stage[i];
            fast_exp_squared_error += fast_exp_error * fast_exp_error;
            f32_stage_power +=
                (double) candidate_f32_stage[i] * candidate_f32_stage[i];
            fast_exp_max_abs = std::max(
                fast_exp_max_abs,
                std::fabs(candidate_fast_exp[i] - candidate_f32_stage[i]));
        }
        const double fast_exp_nmse = fast_exp_squared_error /
            std::max(f32_stage_power, 1.0e-30);

        auto measure_us = [&](bool streaming, bool f32_stage, bool fast_exp) {
            setenv("GGML_CUDA_MLA_STREAM_TOPK", streaming ? "1" : "0", 1);
            setenv("GGML_CUDA_MLA_STREAM_F32_STAGE", f32_stage ? "1" : "0", 1);
            setenv("GGML_CUDA_MLA_STREAM_FAST_EXP", fast_exp ? "1" : "0", 1);
            constexpr int warmups = 3;
            constexpr int iterations = 20;
            for (int i = 0; i < warmups; ++i) {
                ggml_backend_graph_compute(backend, graph);
            }
            ggml_backend_synchronize(backend);
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; ++i) {
                ggml_backend_graph_compute(backend, graph);
            }
            ggml_backend_synchronize(backend);
            const auto end = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::micro>(end - begin).count() /
                iterations;
        };
        const double grouped_us = measure_us(false, false, false);
        constexpr int timing_rounds = 4;
        double streaming_f16_us = 0.0;
        double streaming_f32_us = 0.0;
        double fast_exp_us = 0.0;
        for (int round = 0; round < timing_rounds; ++round) {
            if ((round & 1) == 0) {
                streaming_f16_us += measure_us(true, false, false);
                streaming_f32_us += measure_us(true, true, false);
                fast_exp_us += measure_us(true, true, true);
            } else {
                fast_exp_us += measure_us(true, true, true);
                streaming_f32_us += measure_us(true, true, false);
                streaming_f16_us += measure_us(true, false, false);
            }
        }
        streaming_f16_us /= timing_rounds;
        streaming_f32_us /= timing_rounds;
        fast_exp_us /= timing_rounds;
        std::fprintf(stderr,
                     " grouped=%.1fus streaming_f16=%.1fus"
                     " streaming_f32=%.1fus fast_exp=%.1fus speedup=%.2fx"
                     " fast_nmse=%.3g fast_max_abs=%.3g",
                     grouped_us, streaming_f16_us, streaming_f32_us,
                     fast_exp_us, grouped_us / fast_exp_us,
                     fast_exp_nmse, fast_exp_max_abs);
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void run_ds4_indexer_score_packed_small_case(
        ggml_backend_t backend, int n_tokens) {
    constexpr int dim = 128;
    constexpr int n_heads = 64;
    constexpr int n_comp = 4160;
    constexpr int kv_start = 16384;
    constexpr int ratio = 4;
    ggml_context * ctx = make_test_context(4u << 20);
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) {
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    ggml_tensor * q = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, dim, n_heads, n_tokens);
    ggml_tensor * weights = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_heads, n_tokens);
    ggml_tensor * comp = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F16, dim, n_comp);
    ggml_tensor * scores = ggml_ds4_indexer_score(
        ctx, q, weights, comp, kv_start, ratio);
    ggml_set_output(scores);
    TEST_ASSERT_MSG(ggml_backend_supports_op(backend, scores),
                    "GPU rejected packed-small indexer fixture");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, scores);
    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    const bool allocated = ggml_gallocr_alloc_graph(alloc, graph);
    TEST_ASSERT_MSG(allocated, "packed-small indexer graph allocation failed");
    if (allocated) {
        std::vector<float> q_data((size_t) dim * n_heads * n_tokens);
        std::vector<float> weight_data((size_t) n_heads * n_tokens);
        std::vector<ggml_fp16_t> comp_data((size_t) dim * n_comp);
        for (size_t i = 0; i < q_data.size(); ++i) {
            q_data[i] = ((int) (i % 31) - 15) * 0.0078125f;
        }
        for (size_t i = 0; i < weight_data.size(); ++i) {
            weight_data[i] = ((int) (i % 17) - 8) * 0.015625f;
        }
        for (size_t i = 0; i < comp_data.size(); ++i) {
            comp_data[i] = ggml_fp32_to_fp16(
                ((int) (i % 29) - 14) * 0.0078125f);
        }
        ggml_backend_tensor_set(q, q_data.data(), 0,
                                q_data.size() * sizeof(float));
        ggml_backend_tensor_set(weights, weight_data.data(), 0,
                                weight_data.size() * sizeof(float));
        ggml_backend_tensor_set(comp, comp_data.data(), 0,
                                comp_data.size() * sizeof(ggml_fp16_t));

        std::vector<float> reference((size_t) n_comp * n_tokens);
        std::vector<float> candidate(reference.size());
        ScopedCudaGraphOverrides eager(
            /*disable_graphs=*/true,
            /*mmvq_max_ncols=*/0,
            /*skip_property_check=*/false);
        setenv("GGML_DS4_INDEXER_PACK_SMALL", "0", 1);
        TEST_ASSERT_MSG(
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "reference small-CM indexer score failed");
        ggml_backend_tensor_get(scores, reference.data(), 0,
                                reference.size() * sizeof(float));

        setenv("GGML_DS4_INDEXER_PACK_SMALL", "1", 1);
        TEST_ASSERT_MSG(
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "packed small-CM indexer score failed");
        ggml_backend_tensor_get(scores, candidate.data(), 0,
                                candidate.size() * sizeof(float));
        TEST_ASSERT_MSG(
            std::memcmp(reference.data(), candidate.data(),
                        reference.size() * sizeof(float)) == 0,
            "packed small-CM indexer changed score bits");

        auto measure_us = [&](bool packed) {
            if (packed) {
                setenv("GGML_DS4_INDEXER_PACK_SMALL", "1", 1);
            } else {
                setenv("GGML_DS4_INDEXER_PACK_SMALL", "0", 1);
            }
            constexpr int warmups = 3;
            constexpr int iterations = 30;
            for (int i = 0; i < warmups; ++i) {
                ggml_backend_graph_compute(backend, graph);
            }
            ggml_backend_synchronize(backend);
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; ++i) {
                ggml_backend_graph_compute(backend, graph);
            }
            ggml_backend_synchronize(backend);
            const auto end = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::micro>(end - begin).count() /
                iterations;
        };
        const double reference_us = measure_us(false);
        const double packed_us = measure_us(true);
        std::fprintf(stderr, " q%d=%.1f->%.1fus", n_tokens,
                     reference_us, packed_us);
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

static void test_ds4_indexer_score_packed_small_gpu() {
    std::fprintf(stderr, "  test_ds4_indexer_score_packed_small_gpu ...");
#if !defined(GGML_USE_HIP)
    std::fprintf(stderr, " skipped (HIP-only candidate)\n");
    return;
#endif
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, " skipped (no GPU backend)\n");
        return;
    }
    ScopedEnvVar packed_small_guard("GGML_DS4_INDEXER_PACK_SMALL");
    for (int n_tokens = 2; n_tokens <= 5; ++n_tokens) {
        run_ds4_indexer_score_packed_small_case(backend, n_tokens);
    }
    ggml_backend_free(backend);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_ds4_flash_rope_replay_gpu() {
    std::fprintf(stderr, "  test_ds4_flash_rope_replay_gpu ...");
#if !defined(GGML_USE_HIP)
    std::fprintf(stderr, " skipped (HIP-only contract)\n");
    return;
#endif
    auto backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, " skipped (no GPU backend)\n");
        return;
    }
    constexpr int dim = 512, heads = 4, raw = 128, comp = 640, keep = 512;
    constexpr int starts[] = {7680, 7688, 131072, 7680};
    int passed = 0, cases = 0;
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (int width = 1; width <= 5; ++width) {
            for (bool forward : {false, true}) {
                ++cases;
                auto ctx = make_test_context(2u << 20);
                const int rows = raw + comp;
                auto q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, width, heads);
                auto kv = ggml_new_tensor_3d(ctx, type, dim, rows, 1);
                auto mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, rows, width);
                auto topk = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, keep, width);
                auto positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, width);
                auto negative_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, width);
                for (auto input : {q, kv, mask, topk, positions, negative_positions}) ggml_set_input(input);
                auto attention = [&](int start) {
                    auto out = ggml_flash_attn_ext(ctx, q, kv, kv, mask,
                        1.0f / std::sqrt(float(dim)), 0.0f, 0.0f);
                    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
                    ggml_flash_attn_ext_set_ds4_sparse(out, raw, raw, -keep, 32);
                    ggml_flash_attn_ext_set_ds4_indexer_topk(out, topk);
                    ggml_flash_attn_ext_set_ds4_inverse_rope(out, start,
                        10000.0f, 1.0f / 16.0f, 1.0f, 1.0f, 32.0f, 1.0f, 8192, forward);
                    ggml_set_output(out);
                    return out;
                };
                auto dynamic = attention(starts[0]);
                // Replay positions are runtime inputs, not part of the shape
                // key. The baseline ignores this input and uses stale RoPE.
                ggml_flash_attn_ext_set_ds4_rope_positions(dynamic, positions);
                if (type == GGML_TYPE_F32 && width == 2 && forward) {
                    auto strided = ggml_view_1d(ctx, positions, width, 0);
                    strided->nb[0] = 2 * sizeof(int32_t);
                    for (auto invalid : {
                            ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width),
                            ggml_new_tensor_1d(ctx, GGML_TYPE_I32, width + 1),
                            ggml_new_tensor_2d(ctx, GGML_TYPE_I32, width, 2),
                            strided}) {
                        // Bypass the setter to exercise backend validation of
                        // malformed imported graphs without launching them.
                        dynamic->src[6] = invalid;
                        TEST_ASSERT_MSG(!ggml_backend_supports_op(backend, dynamic),
                                        "DS4 flash accepted invalid runtime RoPE positions");
                    }
                    dynamic->src[6] = positions;
                }
                std::vector<ggml_tensor *> reference;
                auto graph = ggml_new_graph_custom(ctx, 64, false);
                ggml_build_forward_expand(graph, dynamic);
                for (int start : starts) {
                    reference.push_back(attention(start));
                    ggml_build_forward_expand(graph, reference.back());
                }
                // Also compare with the actual standalone GGML RoPE path,
                // not just another instance of the fused implementation.
                auto rope = [&](ggml_tensor * input, ggml_tensor * pos) {
                    return ggml_rope_ext(ctx, input, pos, nullptr, 64,
                        GGML_ROPE_TYPE_NORMAL | GGML_ROPE_TYPE_TAIL, 8192,
                        10000.0f, 1.0f / 16.0f, 1.0f, 1.0f, 32.0f, 1.0f);
                };
                auto native_q = forward
                    ? ggml_permute(ctx, rope(ggml_permute(ctx, q, 0, 2, 1, 3), positions), 0, 2, 1, 3)
                    : q;
                auto native_attention = ggml_flash_attn_ext(ctx, native_q, kv, kv, mask,
                    1.0f / std::sqrt(float(dim)), 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(native_attention, GGML_PREC_F32);
                ggml_flash_attn_ext_set_ds4_sparse(native_attention, raw, raw, -keep, 32);
                ggml_flash_attn_ext_set_ds4_indexer_topk(native_attention, topk);
                auto native = rope(native_attention, negative_positions);
                ggml_set_output(native);
                ggml_build_forward_expand(graph, native);
                auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                bool ok = ggml_backend_supports_op(backend, dynamic) &&
                          ggml_gallocr_alloc_graph(alloc, graph);
                if (ok) {
                    std::vector<float> qdata(ggml_nelements(q)), kvdata(ggml_nelements(kv));
                    for (size_t i = 0; i < qdata.size(); ++i)
                        qdata[i] = float(int((i * 17) % 71) - 35) * 0.0125f;
                    for (size_t i = 0; i < kvdata.size(); ++i)
                        kvdata[i] = float(int((i * 29 + i / dim) % 83) - 41) * 0.025f;
                    std::vector<ggml_fp16_t> masks(ggml_nelements(mask), ggml_fp32_to_fp16(0));
                    std::vector<int32_t> ids(ggml_nelements(topk));
                    for (int t = 0; t < width; ++t) {
                        for (int k = 0; k < keep; ++k)
                            ids[(size_t)t * keep + k] = (29 * k + 17 * t) % comp;
                    }
                    ggml_backend_tensor_set(q, qdata.data(), 0, ggml_nbytes(q));
                    if (type == GGML_TYPE_F16) {
                        std::vector<ggml_fp16_t> half(kvdata.size());
                        std::transform(kvdata.begin(), kvdata.end(), half.begin(), ggml_fp32_to_fp16);
                        ggml_backend_tensor_set(kv, half.data(), 0, ggml_nbytes(kv));
                    } else ggml_backend_tensor_set(kv, kvdata.data(), 0, ggml_nbytes(kv));
                    ggml_backend_tensor_set(mask, masks.data(), 0, ggml_nbytes(mask));
                    ggml_backend_tensor_set(topk, ids.data(), 0, ggml_nbytes(topk));
                    for (size_t run = 0; run < reference.size(); ++run) {
                        std::vector<int32_t> pos(width);
                        for (int t = 0; t < width; ++t) pos[t] = starts[run] + t;
                        ggml_backend_tensor_set(positions, pos.data(), 0, ggml_nbytes(positions));
                        for (int & value : pos) value = -value;
                        ggml_backend_tensor_set(negative_positions, pos.data(), 0, ggml_nbytes(negative_positions));
                        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
                            ok = false;
                            break;
                        }
                        std::vector<float> actual(ggml_nelements(dynamic)), expected(actual.size());
                        ggml_backend_tensor_get(dynamic, actual.data(), 0, ggml_nbytes(dynamic));
                        ggml_backend_tensor_get(reference[run], expected.data(), 0, ggml_nbytes(dynamic));
                        for (size_t i = 0; i < actual.size(); ++i)
                            ok &= std::isfinite(actual[i]) && actual[i] == expected[i];
                        ggml_backend_tensor_get(native, expected.data(), 0, ggml_nbytes(native));
                        for (size_t i = 0; i < actual.size(); ++i)
                            ok &= nearly_equal(actual[i], expected[i], 2.0e-5f, 2.0e-5f);
                    }
                }
                if (!ok) std::fprintf(stderr, " FAIL width=%d kv=%s forward=%d",
                                      width, ggml_type_name(type), forward);
                passed += ok;
                ggml_gallocr_free(alloc);
                ggml_free(ctx);
            }
        }
    }
    std::fprintf(stderr, " %d/%d cases passed\n", passed, cases);
    TEST_ASSERT_MSG(passed == cases, "fused RoPE replay used stale token positions");
    ggml_backend_free(backend);
}

static void test_ds4_topk_block_radix_gpu(int ncols) {
    std::fprintf(stderr, "  test_ds4_topk_block_radix_gpu ncols=%d ...", ncols);
#if !defined(GGML_USE_HIP)
    std::fprintf(stderr, " skipped (HIP-only candidate)\n");
    return;
#endif
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, " skipped (no GPU backend)\n");
        return;
    }

    constexpr int nrows = 4;
    constexpr int k = 512;
    ggml_context * ctx = make_test_context(1u << 20);
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) {
        ggml_backend_free(backend);
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    ggml_tensor * scores = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * selected = ggml_top_k(ctx, scores, k);
    ggml_set_output(selected);
    TEST_ASSERT_MSG(ggml_backend_supports_op(backend, selected),
                    "GPU rejected long-context top-k fixture");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, selected);
    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    const bool allocated = ggml_gallocr_alloc_graph(alloc, graph);
    TEST_ASSERT_MSG(allocated, "top-k graph allocation failed");
    if (allocated) {
        std::vector<float> score_data((size_t) ncols * nrows);
        for (int row = 0; row < nrows; ++row) {
            for (int col = 0; col < ncols; ++col) {
                // Odd multiplication permutes 24-bit integers, so scores are
                // unique and exactly representable for every tested shape.
                // Output order is not part of TOP_K's contract.
                score_data[(size_t) row * ncols + col] =
                    (float) (((uint32_t) col * 2654435761u +
                              (uint32_t) row * 2246822519u) & 0x00ffffffu);
            }
        }
        ggml_backend_tensor_set(scores, score_data.data(), 0,
                                score_data.size() * sizeof(float));

        const char * previous = std::getenv("GGML_DS4_TOPK_BLOCK_RADIX");
        const std::string previous_value = previous ? previous : "";
        std::vector<int32_t> reference((size_t) k * nrows);
        std::vector<int32_t> candidate(reference.size());
        ScopedCudaGraphOverrides eager(
            /*disable_graphs=*/true,
            /*mmvq_max_ncols=*/0,
            /*skip_property_check=*/false);
        setenv("GGML_DS4_TOPK_BLOCK_RADIX", "0", 1);
        TEST_ASSERT_MSG(
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "reference long-context top-k failed");
        ggml_backend_tensor_get(selected, reference.data(), 0,
                                reference.size() * sizeof(int32_t));

        setenv("GGML_DS4_TOPK_BLOCK_RADIX", "1", 1);
        TEST_ASSERT_MSG(
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "block-radix long-context top-k failed");
        ggml_backend_tensor_get(selected, candidate.data(), 0,
                                candidate.size() * sizeof(int32_t));

        for (int row = 0; row < nrows; ++row) {
            auto ref_begin = reference.begin() + (size_t) row * k;
            auto candidate_begin = candidate.begin() + (size_t) row * k;
            std::sort(ref_begin, ref_begin + k);
            std::sort(candidate_begin, candidate_begin + k);
            TEST_ASSERT_MSG(
                std::equal(ref_begin, ref_begin + k, candidate_begin),
                "block-radix top-k changed the selected row set");
        }

        // Padding and a valid -infinity score used to share the same radix
        // key. Prove that padded columns can never be returned on a tie.
        std::fill(
            score_data.begin(), score_data.end(),
            -std::numeric_limits<float>::infinity());
        ggml_backend_tensor_set(scores, score_data.data(), 0,
                                score_data.size() * sizeof(float));
        TEST_ASSERT_MSG(
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "block-radix -infinity top-k failed");
        ggml_backend_tensor_get(selected, candidate.data(), 0,
                                candidate.size() * sizeof(int32_t));
        for (int32_t index : candidate) {
            TEST_ASSERT_MSG(
                index >= 0 && index < ncols,
                "block-radix top-k returned a padded column");
        }
        for (int row = 0; row < nrows; ++row) {
            for (int col = 0; col < ncols; ++col) {
                score_data[(size_t) row * ncols + col] =
                    (float) (((uint32_t) col * 2654435761u +
                              (uint32_t) row * 2246822519u) & 0x00ffffffu);
            }
        }
        ggml_backend_tensor_set(scores, score_data.data(), 0,
                                score_data.size() * sizeof(float));

        auto measure_us = [&](bool block_radix) {
            if (block_radix) {
                setenv("GGML_DS4_TOPK_BLOCK_RADIX", "1", 1);
            } else {
                setenv("GGML_DS4_TOPK_BLOCK_RADIX", "0", 1);
            }
            constexpr int warmups = 5;
            constexpr int iterations = 100;
            for (int i = 0; i < warmups; ++i) {
                ggml_backend_graph_compute(backend, graph);
            }
            ggml_backend_synchronize(backend);
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; ++i) {
                ggml_backend_graph_compute(backend, graph);
            }
            ggml_backend_synchronize(backend);
            const auto end = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::micro>(end - begin).count() /
                iterations;
        };
        const double reference_us = measure_us(false);
        const double candidate_us = measure_us(true);
        std::fprintf(stderr, " reference=%.1fus block_radix=%.1fus",
                     reference_us, candidate_us);

        if (previous) {
            setenv("GGML_DS4_TOPK_BLOCK_RADIX", previous_value.c_str(), 1);
        } else {
            unsetenv("GGML_DS4_TOPK_BLOCK_RADIX");
        }
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_ds4_flash_attention_inverse_rope_fallback_gpu() {
    std::fprintf(stderr,
                 "  test_ds4_flash_attention_inverse_rope_fallback_gpu ...");
#if !defined(GGML_USE_HIP)
    std::fprintf(stderr, " skipped (HIP-only contract)\n");
    return;
#endif
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, " skipped (no GPU backend)\n");
        return;
    }

    constexpr int head_dim = 512;
    constexpr int raw_rows = 128;
    constexpr int n_comp_rows = 8;
    constexpr int n_kv = raw_rows + n_comp_rows;
    constexpr int kv_start = 131071;
    constexpr float freq_base = 10000.0f;
    constexpr float freq_scale = 1.0f / 16.0f;
    constexpr float ext_factor = 1.0f;
    constexpr float attn_factor = 1.0f;
    constexpr float beta_fast = 32.0f;
    constexpr float beta_slow = 1.0f;
    constexpr int n_ctx_orig = 8192;

    ggml_context * ctx = make_test_context(3u << 20);
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) {
        ggml_backend_free(backend);
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    ggml_tensor * q = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, head_dim, 1, 1);
    ggml_tensor * k = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, head_dim, n_kv, 1);
    ggml_tensor * v = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, head_dim, n_kv, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F16, n_kv, 1);
    ggml_tensor * output = ggml_flash_attn_ext(
        ctx, q, k, v, mask, 1.0f / std::sqrt((float) head_dim),
        0.0f, 0.0f);
    // One retained compressed block selects the single-head fallback without
    // pruning any live row. A row-constant V makes attention itself an exact
    // identity, leaving a nontrivial scaled/YaRN inverse RoPE to validate.
    ggml_flash_attn_ext_set_ds4_sparse(
        output, raw_rows, raw_rows, 4, n_comp_rows);
    ggml_flash_attn_ext_set_ds4_inverse_rope(
        output, kv_start, freq_base, freq_scale, ext_factor, attn_factor,
        beta_fast, beta_slow, n_ctx_orig, false);
    ggml_set_output(output);
    TEST_ASSERT_MSG(ggml_backend_supports_op(backend, output),
                    "GPU rejected DS4 inverse-RoPE fallback attention");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    const bool allocated = ggml_gallocr_alloc_graph(alloc, graph);
    TEST_ASSERT_MSG(allocated,
                    "DS4 inverse-RoPE fallback graph allocation failed");
    if (allocated) {
        std::vector<float> q_data(head_dim);
        std::vector<float> k_data((size_t) head_dim * n_kv);
        std::vector<float> v_data((size_t) head_dim * n_kv);
        std::vector<float> base_value(head_dim);
        std::vector<float> expected(head_dim);
        std::vector<ggml_fp16_t> mask_data(
            n_kv, ggml_fp32_to_fp16(0.0f));
        for (int d = 0; d < head_dim; ++d) {
            q_data[(size_t) d] = ((d % 19) - 9) * 0.002f;
            base_value[(size_t) d] = ((d % 17) - 8) * 0.003f;
        }
        expected = base_value;
        float corr_dims[2];
        ggml_rope_yarn_corr_dims(
            64, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        const float theta_scale = std::pow(freq_base, -2.0f / 64.0f);
        constexpr double tau = 6.2831853071795864769;
        for (int pair = 0; pair < 32; ++pair) {
            const int i0 = 2 * pair;
            const double theta_extrap = -(double) kv_start *
                std::pow((double) theta_scale, (double) pair);
            const double theta_interp =
                (double) freq_scale * theta_extrap;
            const float ramp_y = (pair - corr_dims[0]) /
                std::max(0.001f, corr_dims[1] - corr_dims[0]);
            const float ramp_mix =
                (1.0f - std::min(1.0f, std::max(0.0f, ramp_y))) *
                ext_factor;
            double theta = theta_interp * (1.0 - (double) ramp_mix) +
                           theta_extrap * (double) ramp_mix;
            theta -= tau * std::floor(theta * (1.0 / tau));
            const float mscale = attn_factor *
                (1.0f + 0.1f * std::log(1.0f / freq_scale));
            const float cos_theta = std::cos((float) theta) * mscale;
            const float sin_theta = std::sin((float) theta) * mscale;
            const size_t d0 = (size_t) head_dim - 64 + (size_t) i0;
            const float x0 = base_value[d0 + 0];
            const float x1 = base_value[d0 + 1];
            expected[d0 + 0] = x0 * cos_theta - x1 * sin_theta;
            expected[d0 + 1] = x0 * sin_theta + x1 * cos_theta;
        }
        for (int row = 0; row < n_kv; ++row) {
            for (int d = 0; d < head_dim; ++d) {
                k_data[(size_t) row * head_dim + d] =
                    (((row + d) % 23) - 11) * 0.002f;
                v_data[(size_t) row * head_dim + d] = base_value[(size_t) d];
            }
        }
        ggml_backend_tensor_set(q, q_data.data(), 0,
                                q_data.size() * sizeof(float));
        ggml_backend_tensor_set(k, k_data.data(), 0,
                                k_data.size() * sizeof(float));
        ggml_backend_tensor_set(v, v_data.data(), 0,
                                v_data.size() * sizeof(float));
        ggml_backend_tensor_set(mask, mask_data.data(), 0,
                                mask_data.size() * sizeof(ggml_fp16_t));
        const bool computed =
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        TEST_ASSERT_MSG(computed,
                        "DS4 inverse-RoPE fallback graph compute failed");
        if (computed) {
            std::vector<float> actual(head_dim);
            ggml_backend_tensor_get(output, actual.data(), 0,
                                    actual.size() * sizeof(float));
            for (int d = 0; d < head_dim; ++d) {
                TEST_ASSERT_MSG(
                    nearly_equal(actual[(size_t) d], expected[(size_t) d],
                                 2.0e-5f, 2.0e-5f),
                    "inverse-RoPE fallback output mismatch");
            }
        }
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_hc_post_strided_split_gpu() {
    std::fprintf(stderr, "  test_hc_post_strided_split_gpu ...");
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, " skipped (no GPU backend)\n");
        return;
    }

    constexpr int n_embd = 16;
    constexpr int n_hc = 4;
    constexpr int n_tokens = 3;
    constexpr int hc_dim = n_embd * n_hc;
    constexpr int mix_dim = 2 * n_hc + n_hc * n_hc;
    constexpr int split_stride = mix_dim + 7;

    std::vector<float> residual((size_t) hc_dim * n_tokens);
    std::vector<float> block_out((size_t) n_embd * n_tokens);
    std::vector<float> main_block((size_t) n_embd * n_tokens);
    std::vector<float> peer_block((size_t) n_embd * n_tokens);
    std::vector<float> split_storage((size_t) split_stride * n_tokens, -99.0f);
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = ((int) (i % 17) - 8) * 0.03125f;
    }
    for (size_t i = 0; i < block_out.size(); ++i) {
        block_out[i] = ((int) (i % 11) - 5) * 0.0625f;
        main_block[i] = block_out[i] * 0.25f;
        peer_block[i] = block_out[i] - main_block[i];
    }
    for (int token = 0; token < n_tokens; ++token) {
        float * split = split_storage.data() + (size_t) token * split_stride;
        for (int i = 0; i < mix_dim; ++i) {
            split[i] = ((i + 3 * token) % 13 - 6) * 0.05f;
        }
    }

    std::vector<float> expected((size_t) hc_dim * n_tokens);
    for (int token = 0; token < n_tokens; ++token) {
        const float * residual_row = residual.data() + (size_t) token * hc_dim;
        const float * block_row = block_out.data() + (size_t) token * n_embd;
        const float * split = split_storage.data() + (size_t) token * split_stride;
        const float * post = split + n_hc;
        const float * comb = split + 2 * n_hc;
        float * output = expected.data() + (size_t) token * hc_dim;
        for (int h = 0; h < n_hc; ++h) {
            for (int d = 0; d < n_embd; ++d) {
                float value = block_row[d] * post[h];
                for (int src = 0; src < n_hc; ++src) {
                    value += comb[h + src * n_hc] *
                             residual_row[src * n_embd + d];
                }
                output[h * n_embd + d] = value;
            }
        }
    }

    ggml_context * ctx = make_test_context(1u << 20);
    TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
    if (!ctx) {
        ggml_backend_free(backend);
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    ggml_tensor * residual_t = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_tensor * block_t = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * main_t = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * peer_t = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * split_storage_t = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, split_stride, n_tokens);
    ggml_tensor * split_t = ggml_view_2d(
        ctx, split_storage_t, mix_dim, n_tokens,
        split_storage_t->nb[1], 0);
    TEST_ASSERT(!ggml_is_contiguous(split_t));
    ggml_tensor * output_t = ggml_ds4_hc_post(
        ctx, residual_t, block_t, split_t, n_hc);
    ggml_tensor * split_output_t = ggml_ds4_hc_post_split(
        ctx, residual_t, main_t, peer_t, split_t, n_hc);
    ggml_set_output(output_t);
    ggml_set_output(split_output_t);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output_t);
    ggml_build_forward_expand(graph, split_output_t);
    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    const bool allocated = ggml_gallocr_alloc_graph(alloc, graph);
    TEST_ASSERT_MSG(allocated, "strided HC-post graph allocation failed");
    if (allocated) {
        ggml_backend_tensor_set(residual_t, residual.data(), 0,
                                residual.size() * sizeof(float));
        ggml_backend_tensor_set(block_t, block_out.data(), 0,
                                block_out.size() * sizeof(float));
        ggml_backend_tensor_set(main_t, main_block.data(), 0,
                                main_block.size() * sizeof(float));
        ggml_backend_tensor_set(peer_t, peer_block.data(), 0,
                                peer_block.size() * sizeof(float));
        ggml_backend_tensor_set(split_storage_t, split_storage.data(), 0,
                                split_storage.size() * sizeof(float));
        const bool computed =
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        TEST_ASSERT_MSG(computed, "strided HC-post graph compute failed");
        if (computed) {
            std::vector<float> actual(expected.size());
            std::vector<float> split_actual(expected.size());
            ggml_backend_tensor_get(output_t, actual.data(), 0,
                                    actual.size() * sizeof(float));
            ggml_backend_tensor_get(split_output_t, split_actual.data(), 0,
                                    split_actual.size() * sizeof(float));
            for (size_t i = 0; i < actual.size(); ++i) {
                TEST_ASSERT_MSG(nearly_equal(actual[i], expected[i],
                                             1.0e-6f, 1.0e-6f),
                                "strided HC-post output mismatch");
                TEST_ASSERT_MSG(nearly_equal(split_actual[i], expected[i],
                                             1.0e-6f, 1.0e-6f),
                                "batched split HC-post output mismatch");
            }
        }
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

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

    // The scratch-capacity test exercises n_hc=8, so this reference buffer
    // cannot be fixed at the model-default 4x4 shape.
    std::vector<float> c((size_t) n_hc * (size_t) n_hc);
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

// The stale-boundary guard in deepseek4_step_layer_range must fire before any
// backend, cache, or weight access, so stub weights and a null backend are
// sufficient to pin it.
static void test_layer_range_rejects_stale_hc_boundary() {
    std::fprintf(stderr, "  test_layer_range_rejects_stale_hc_boundary ...");
    DeepSeek4Weights w;
    w.n_layer = 4;
    w.n_embd = 8;
    w.n_hc = 4;
    DeepSeek4Cache cache;

    const int n_tokens = 1;
    const size_t hc_dim = (size_t) w.n_embd * (size_t) w.n_hc;

    // Later-shard call where embed aliases hc_state but the carried state has
    // a stale size: must fail without resizing, because a resize would free
    // the buffer embed points into.
    std::vector<float> hc_state(hc_dim / 2, 0.0f);
    const float * stale_alias = hc_state.data();
    bool ok = deepseek4_step_layer_range(
        /*backend=*/nullptr, /*device=*/0, w, cache, hc_state,
        stale_alias, n_tokens, /*kv_start=*/0,
        /*layer_begin=*/2, /*layer_end=*/4, /*out_logits=*/nullptr);
    TEST_ASSERT_MSG(!ok, "stale aliased HC boundary state must be rejected");
    TEST_ASSERT_MSG(hc_state.size() == hc_dim / 2,
                    "rejected call must not resize the aliased HC state");
    TEST_ASSERT_MSG(cache.layer_range_cache == nullptr,
                    "guard must fire before the layer-range cache is created");

    // Later-shard call with no boundary state at all.
    std::vector<float> missing_state;
    ok = deepseek4_step_layer_range(
        nullptr, 0, w, cache, missing_state,
        /*embed=*/nullptr, n_tokens, 0, 2, 4, nullptr);
    TEST_ASSERT_MSG(!ok, "missing HC boundary state must be rejected");
    TEST_ASSERT_MSG(cache.layer_range_cache == nullptr,
                    "guard must fire before the layer-range cache is created");
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
static void test_hc_scratch_per_device() {
    std::fprintf(stderr, "  test_hc_scratch_per_device ...");
    if (ggml_backend_cuda_get_device_count() < 2) {
        std::fprintf(stderr, " skipped (requires two GPU devices)\n");
        return;
    }

    constexpr int n_embd = 32;
    constexpr int n_hc = 4;
    constexpr int sinkhorn_iters = 6;
    constexpr float hc_eps = 1.0e-6f;
    constexpr int mix_dim = 2 * n_hc + n_hc * n_hc;
    constexpr int hc_dim = n_embd * n_hc;

    std::vector<float> hc_state((size_t) hc_dim);
    std::iota(hc_state.begin(), hc_state.end(), -16.0f);
    for (float & value : hc_state) value *= 0.01f;
    std::vector<ggml_fp16_t> fn_f16((size_t) mix_dim * (size_t) hc_dim,
                                     ggml_fp32_to_fp16(0.0f));
    std::vector<float> scale((size_t) mix_dim, 0.0f);
    scale[0] = 1.0f;
    scale[1] = 1.0f;
    scale[2] = 1.0f;
    std::vector<float> base((size_t) mix_dim, 0.0f);

    struct Result {
        bool ok = false;
        std::vector<float> working;
        std::vector<float> post;
        std::vector<float> comb;

        Result(int embd, int hc)
            : working((size_t) embd),
              post((size_t) hc),
              comb((size_t) hc * (size_t) hc) {}
    };

    auto run_on_worker = [&](int device, Result & result) {
        std::thread worker([&, device]() {
            if (!deepseek4_cuda_hc_set_device(device)) {
                return;
            }
            ggml_backend_t dev_backend = ggml_backend_cuda_init(device);
            if (!dev_backend) {
                return;
            }
            ggml_context * ctx = make_test_context(1u << 16);
            ggml_tensor * fn_t =
                ctx ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hc_dim, mix_dim) : nullptr;
            ggml_backend_buffer_t buf =
                fn_t ? ggml_backend_alloc_ctx_tensors(ctx, dev_backend) : nullptr;
            if (buf) {
                ggml_backend_tensor_set(fn_t, fn_f16.data(), 0,
                                        fn_f16.size() * sizeof(ggml_fp16_t));
                result.ok = deepseek4_cuda_hc_pre(
                    hc_state.data(), fn_t->data, scale.data(), base.data(),
                    n_embd, n_hc, sinkhorn_iters, hc_eps,
                    result.working.data(), result.post.data(), result.comb.data());
                ggml_backend_buffer_free(buf);
            }
            if (ctx) ggml_free(ctx);
            ggml_backend_free(dev_backend);
        });
        worker.join();
    };

    Result first_device(n_embd, n_hc);
    Result second_device(n_embd, n_hc);
    Result first_device_again(n_embd, n_hc);
    run_on_worker(0, first_device);
    run_on_worker(1, second_device);
    run_on_worker(0, first_device_again);

    TEST_ASSERT_MSG(first_device.ok, "HC direct call failed on GPU device 0");
    TEST_ASSERT_MSG(second_device.ok, "HC direct call failed on GPU device 1");
    TEST_ASSERT_MSG(first_device_again.ok, "HC direct call failed after returning to GPU device 0");
    if (first_device.ok && second_device.ok && first_device_again.ok) {
        for (int i = 0; i < n_embd; ++i) {
            TEST_ASSERT(nearly_equal(first_device.working[(size_t) i],
                                     second_device.working[(size_t) i]));
            TEST_ASSERT(nearly_equal(first_device.working[(size_t) i],
                                     first_device_again.working[(size_t) i]));
        }
        for (int i = 0; i < n_hc; ++i) {
            TEST_ASSERT(nearly_equal(first_device.post[(size_t) i],
                                     second_device.post[(size_t) i]));
        }
        for (int i = 0; i < n_hc * n_hc; ++i) {
            TEST_ASSERT(nearly_equal(first_device.comb[(size_t) i],
                                     second_device.comb[(size_t) i]));
        }
    }
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_hc_set_device_contract() {
    std::fprintf(stderr, "  test_hc_set_device_contract ...");
    // Device selection is per host thread; run on a worker so the rejected
    // selections cannot leave sticky CUDA error state on the main thread.
    std::thread worker([]() {
        TEST_ASSERT_MSG(!deepseek4_cuda_hc_set_device(-1),
                        "negative device must be rejected");
        const int device_count = ggml_backend_cuda_get_device_count();
        TEST_ASSERT_MSG(!deepseek4_cuda_hc_set_device(device_count),
                        "out-of-range device must be rejected");
        if (device_count > 0) {
            TEST_ASSERT_MSG(deepseek4_cuda_hc_set_device(0),
                            "selecting device 0 failed");
        }
    });
    worker.join();
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_cuda_graph_overrides_restore_nested_state() {
    std::fprintf(stderr, "  test_cuda_graph_overrides_restore_nested_state ...");

    const bool initial_disabled =
        ggml_backend_cuda_set_graphs_disabled_override(false);
    const int initial_mmvq =
        ggml_backend_cuda_set_mmvq_max_ncols_override(0);
    const bool initial_skip =
        ggml_backend_cuda_set_skip_props_check(false);

    {
        ScopedCudaGraphOverrides outer(
            /*disable_graphs=*/true,
            /*mmvq_max_ncols=*/4,
            /*skip_property_check=*/true);
        TEST_ASSERT(ggml_backend_cuda_set_graphs_disabled_override(true));
        TEST_ASSERT(ggml_backend_cuda_set_mmvq_max_ncols_override(4) == 4);
        TEST_ASSERT(ggml_backend_cuda_set_skip_props_check(true));

        {
            ScopedCudaGraphOverrides inner(
                /*disable_graphs=*/true,
                /*mmvq_max_ncols=*/2,
                /*skip_property_check=*/true);
            TEST_ASSERT(ggml_backend_cuda_set_graphs_disabled_override(true));
            TEST_ASSERT(ggml_backend_cuda_set_mmvq_max_ncols_override(2) == 2);
            TEST_ASSERT(ggml_backend_cuda_set_skip_props_check(true));
        }

        TEST_ASSERT(ggml_backend_cuda_set_graphs_disabled_override(true));
        TEST_ASSERT(ggml_backend_cuda_set_mmvq_max_ncols_override(4) == 4);
        TEST_ASSERT(ggml_backend_cuda_set_skip_props_check(true));
    }

    TEST_ASSERT(!ggml_backend_cuda_set_graphs_disabled_override(false));
    TEST_ASSERT(ggml_backend_cuda_set_mmvq_max_ncols_override(0) == 0);
    TEST_ASSERT(!ggml_backend_cuda_set_skip_props_check(false));

    ggml_backend_cuda_set_graphs_disabled_override(initial_disabled);
    ggml_backend_cuda_set_mmvq_max_ncols_override(initial_mmvq);
    ggml_backend_cuda_set_skip_props_check(initial_skip);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_cuda_graph_rebuild_generation_guard() {
    std::fprintf(stderr, "  test_cuda_graph_rebuild_generation_guard ...");
    if (ggml_backend_cuda_get_device_count() <= 0) {
        std::fprintf(stderr, " skipped (no GPU device)\n");
        return;
    }
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) {
        std::fprintf(stderr, " skipped (no GPU backend)\n");
        return;
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    TEST_ASSERT_MSG(cpu != nullptr, "CPU fallback backend init failed");
    if (!cpu) {
        ggml_backend_free(gpu);
        std::fprintf(stderr, " FAIL\n");
        return;
    }

    std::vector<uint8_t> arena(1024 * 1024);
    const void * first_graph_key = nullptr;
    auto run_generation = [&](bool square, float expected,
                              bool retire_after) -> bool {
        ggml_init_params params{};
        params.mem_size = arena.size();
        params.mem_buffer = arena.data();
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        if (!ctx) return false;

        ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_input(input);
        ggml_tensor * output = square
            ? ggml_sqr(ctx, input)
            : ggml_neg(ctx, input);
        ggml_set_output(output);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, output);

        // Both generations intentionally reuse the exact metadata address,
        // which reproduces the verifier LRU slot's pointer-key collision.
        ggml_tensor * graph_key = ggml_graph_node(graph, 0);
        if (!first_graph_key) {
            first_graph_key = graph_key;
        } else {
            TEST_ASSERT(graph_key == first_graph_key);
        }

        // The generic scheduler contract requires a CPU fallback as its final
        // backend even when every operation in this graph stays on the GPU.
        ggml_backend_t backends[] = {gpu, cpu};
        ggml_backend_sched_t sched = ggml_backend_sched_new(
            backends, nullptr, 2, 64, false, true);
        if (sched) {
            // Match the fused verifier: inputs and outputs are explicitly
            // pinned to the main GPU before the scheduler allocates splits.
            ggml_backend_sched_set_tensor_backend(sched, input, gpu);
            ggml_backend_sched_set_tensor_backend(sched, output, gpu);
        }
        bool ok = sched && ggml_backend_sched_alloc_graph(sched, graph);
        const float value = 3.0f;
        if (ok) {
            for (int warm = 0; warm < 3 && ok; ++warm) {
                // The scheduler allocator may legally reuse the input buffer
                // for this unary output. Production uploads fresh inputs on
                // every replay, so mirror that contract for each warmup.
                ggml_backend_tensor_set(input, &value, 0, sizeof(value));
                ScopedCudaGraphOverrides force_replay(
                    /*disable_graphs=*/false,
                    /*mmvq_max_ncols=*/0,
                    /*skip_property_check=*/true);
                ok = ggml_backend_sched_graph_compute(sched, graph) ==
                    GGML_STATUS_SUCCESS;
            }
        }
        float actual = 0.0f;
        if (ok) {
            ggml_backend_tensor_get(output, &actual, 0, sizeof(actual));
            ok = nearly_equal(actual, expected, 1.0e-6f, 1.0e-6f);
            if (!ok) {
                std::fprintf(stderr,
                             " graph output=%g expected=%g;", actual,
                             expected);
            }
        }
        if (retire_after) {
            ggml_backend_cuda_graph_invalidate_range(
                gpu, arena.data(), arena.size());
        }
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return ok;
    };

    TEST_ASSERT_MSG(run_generation(true, 9.0f, false),
                    "first graph generation failed");
    TEST_ASSERT_MSG(run_generation(false, -3.0f, true),
                    "rebuilt graph replayed stale executable");
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}

static void test_hc_scratch_shape_capacity() {
    std::fprintf(stderr, "  test_hc_scratch_shape_capacity ...");
    if (!deepseek4_cuda_hc_set_device(0)) {
        std::fprintf(stderr, " skipped (no GPU device)\n");
        return;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, " skipped (no GPU backend)\n");
        return;
    }

    constexpr int sinkhorn_iters = 6;
    constexpr float hc_eps = 1.0e-6f;
    // Grow both shape axes past the first call, then shrink back: every run
    // must stay correct while the per-device scratch reallocates and reuses.
    const int shapes[][2] = {{32, 2}, {192, 8}, {32, 2}};

    std::mt19937 rng(321);
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);

    for (const auto & shape : shapes) {
        const int n_embd = shape[0];
        const int n_hc = shape[1];
        const int hc_dim = n_embd * n_hc;
        const int mix_dim = 2 * n_hc + n_hc * n_hc;

        std::vector<float> hc_state((size_t) hc_dim);
        std::vector<ggml_fp16_t> fn_f16((size_t) mix_dim * (size_t) hc_dim);
        std::vector<float> scale((size_t) mix_dim, 0.0f);
        std::vector<float> base((size_t) mix_dim);
        for (float & v : hc_state) v = dist(rng);
        for (ggml_fp16_t & v : fn_f16) v = ggml_fp32_to_fp16(dist(rng));
        scale[0] = 0.9f;
        scale[1] = 1.05f;
        scale[2] = 1.0f;
        for (float & v : base) v = 0.1f * dist(rng);

        std::vector<float> ref_working;
        std::vector<float> ref_post;
        std::vector<float> ref_comb;
        test_reference_hc_pre(hc_state, fn_f16, scale, base, n_embd, n_hc,
                              sinkhorn_iters, hc_eps,
                              ref_working, ref_post, ref_comb);

        ggml_context * ctx = make_test_context(1u << 16);
        TEST_ASSERT_MSG(ctx != nullptr, "ggml_init failed");
        if (!ctx) break;
        ggml_tensor * fn_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hc_dim, mix_dim);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        TEST_ASSERT_MSG(buf != nullptr, "ggml backend buffer alloc failed");
        if (!buf) {
            ggml_free(ctx);
            break;
        }
        ggml_backend_tensor_set(fn_t, fn_f16.data(), 0,
                                fn_f16.size() * sizeof(ggml_fp16_t));

        std::vector<float> working((size_t) n_embd);
        std::vector<float> post((size_t) n_hc);
        std::vector<float> comb((size_t) n_hc * (size_t) n_hc);
        const bool ok = deepseek4_cuda_hc_pre(
            hc_state.data(), fn_t->data, scale.data(), base.data(),
            n_embd, n_hc, sinkhorn_iters, hc_eps,
            working.data(), post.data(), comb.data());
        TEST_ASSERT_MSG(ok, "HC pre failed after scratch shape change");
        if (ok) {
            for (int i = 0; i < n_embd; ++i) {
                TEST_ASSERT_MSG(
                    nearly_equal(working[(size_t) i], ref_working[(size_t) i], 2.0e-4f, 2.0e-4f),
                    "working mismatch after scratch shape change");
            }
            for (int i = 0; i < n_hc; ++i) {
                TEST_ASSERT_MSG(
                    nearly_equal(post[(size_t) i], ref_post[(size_t) i], 2.0e-4f, 2.0e-4f),
                    "post mismatch after scratch shape change");
            }
            for (int i = 0; i < n_hc * n_hc; ++i) {
                TEST_ASSERT_MSG(
                    nearly_equal(comb[(size_t) i], ref_comb[(size_t) i], 2.0e-4f, 2.0e-4f),
                    "comb mismatch after scratch shape change");
            }
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    ggml_backend_free(backend);
    std::fprintf(stderr, g_failures ? " done\n" : " ok\n");
}
#endif
#endif

static void test_paged_lone_prompt_limits() {
    DeepSeek4Backend backend{DeepSeek4BackendConfig{}};
    PagedKvPool pool(32, 6, 16);
    DeepSeek4SeqEngine engine(backend, pool, 256, 16);
    TEST_ASSERT(engine.step_plan_limits(0).max_prefill_tokens_per_sequence == 4);

    const auto first = engine.admit(1, std::vector<int32_t>(32, 1), SamplerCfg{});
    TEST_ASSERT(first.status == SeqEngine::AdmitResult::Status::admitted);
    auto single = plan_prefill_slices({{first.slot, 0}}, engine.step_plan_limits(0));
    TEST_ASSERT(single.size() == 1 && single[0].max_tokens == 16);

    const auto second = engine.admit(2, std::vector<int32_t>(64, 1), SamplerCfg{});
    TEST_ASSERT(second.status == SeqEngine::AdmitResult::Status::admitted);
    auto shared = plan_prefill_slices(
        {{first.slot, 0}, {second.slot, 1}}, engine.step_plan_limits(0));
    TEST_ASSERT(shared.size() == 2 && shared[0].max_tokens == 4 &&
                shared[1].max_tokens == 4);

    TEST_ASSERT(engine.slots_.append_prefill(first.slot, 32).ok);
    engine.slots_.commit_prefill(first.slot);
    auto mixed = plan_prefill_slices({{second.slot, 1}}, engine.step_plan_limits(1));
    TEST_ASSERT(mixed.size() == 1 && mixed[0].max_tokens == 4);

    engine.retire(first.slot);
    single = plan_prefill_slices({{second.slot, 1}}, engine.step_plan_limits(0));
    TEST_ASSERT(single.size() == 1 && single[0].max_tokens == 16);

    backend.moe_hybrid_ = std::make_unique<MoeHybridStorage>();
    single = plan_prefill_slices({{second.slot, 1}}, engine.step_plan_limits(0));
    TEST_ASSERT(single.size() == 1 && single[0].max_tokens == 1);
}

static void test_paged_cache_allocation(ggml_backend_t backend) {
    std::fprintf(stderr, "  paged cache allocation and slot reset...");
    DeepSeek4Weights weights;
    weights.n_layer = 3;
    weights.head_dim = 16;
    weights.n_indexer_head_dim = 8;
    weights.compress_ratios = {0, 4, 128};
    DeepSeek4PagedCache cache;
    for (uint32_t slots : {1u, 3u}) {
        if (!create_deepseek4_paged_cache(backend, weights, slots, 257, 5, cache)) {
            TEST_ASSERT_MSG(false, "paged cache creation failed");
            return;
        }
        uint64_t raw_bytes = 0, compressed_bytes = 0, state_bytes = 0;
        for (const auto & layer : cache.layers) {
            raw_bytes += ggml_nbytes(layer.raw_kv);
            for (const auto * tensor : {layer.comp_kv, layer.index_comp_kv}) {
                if (tensor) compressed_bytes += ggml_nbytes(tensor);
            }
            for (auto * tensor : {layer.attn_compressor.state_kv,
                                  layer.attn_compressor.state_score,
                                  layer.indexer_compressor.state_kv,
                                  layer.indexer_compressor.state_score}) {
                if (!tensor) continue;
                state_bytes += ggml_nbytes(tensor);
                std::vector<float> values(ggml_nelements(tensor), 1.0f);
                ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
            }
        }
        TEST_ASSERT(raw_bytes == cache.plan.raw_bytes);
        TEST_ASSERT(compressed_bytes == cache.plan.compressed_bytes);
        TEST_ASSERT(state_bytes == cache.plan.state_bytes);
        TEST_ASSERT(cache.layers[0].comp_kv == nullptr);
        TEST_ASSERT(cache.layers[0].attn_compressor.state_kv == nullptr);
        TEST_ASSERT(cache.layers[2].index_comp_kv == nullptr);
        reset_deepseek4_paged_slot(cache, slots - 1);
        for (const auto & layer : cache.layers) {
            for (auto * tensor : {layer.attn_compressor.state_kv,
                                  layer.attn_compressor.state_score,
                                  layer.indexer_compressor.state_kv,
                                  layer.indexer_compressor.state_score}) {
                if (!tensor) continue;
                std::vector<float> values(ggml_nelements(tensor));
                ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
                const size_t slot_elements = tensor->nb[2] / sizeof(float);
                for (size_t i = 0; i < values.size(); ++i) {
                    TEST_ASSERT(values[i] == (i / slot_elements == slots - 1 ? 0.0f : 1.0f));
                }
            }
        }
        free_deepseek4_paged_cache(cache);
        TEST_ASSERT(!cache.ctx && !cache.buf && !cache.pool && cache.layers.empty());
    }
    std::fprintf(stderr, " done\n");
}

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "FAIL: ggml_backend_cpu_init failed\n");
        return 1;
    }

    test_paged_lone_prompt_limits();
    test_paged_cache_allocation(backend);
    test_compressor_pooling_correctness(backend);
    test_moe_expert_major_default_threshold();
    test_chunked_graph_allocator(backend);
    test_swiglu_ds4_cpu_correctness(backend);
    test_moe_routing_correctness(backend);
    test_top6_head4_tail2_route_slices(backend);
    test_rmsnorm_correctness(backend);
    test_grouped_output_projection_shape();
    test_grouped_output_projection_cpu(backend);
    test_ds4_flash_attention_cpu_rejected(backend);
    test_indexer_qat_cpu(backend);
    test_indexer_score_cpu(backend);
    test_indexer_mask_cpu(backend);
    test_hash_routing_lookup();
    test_raw_ring_spans_after_wrap();
    test_verify_raw_mask_spans();
    test_auto_split_computation();
    test_layer_range_validation();
    test_hc_state_dimensions();
    test_layer_split_request_propagates_sampler();
    test_layer_split_sampler_uses_prompt_history();
    test_layer_split_sampler_appends_generated_tokens();
    test_layer_split_restore_preserves_full_sampling_history();
    test_backend_sampling_penalizes_prompt_history();
    test_loader_rejects_missing_required_metadata(backend);
    test_loader_rejects_invalid_compress_ratio_type(backend);
    test_loader_rejects_zero_vocab_size(backend);
    test_loader_reads_tokenizer_special_ids(backend);
    test_loader_rejects_truncated_tensor_data(backend);
    test_image_bias_loader_opt_in_contract(backend);
    test_image_batch_admission_before_execution(backend);
    test_image_storage_admission_metadata();
    test_image_admission_resource_snapshots();
    test_dspark_loader_contract_and_bounds(backend);
    test_dspark_confidence_uses_separate_hidden(backend);
    test_safe_compressor_batch_tokens();
    test_hybrid_prefill_chunk_tokens();
    test_mix_mmq_prefill_default();
    test_dspark_park_all_releases_drafter();
    test_pflash_rejects_invalid_requests();
    test_pflash_failed_load_releases_backend();
    test_pflash_keep_ratio_contract();
    test_indexer_visibility_suffix();
    test_pflash_legacy_compress_contract();
    test_dspark_raw_ring_rollback_after_wrap(backend);
    test_dspark_compressor_rollback(backend);
    test_snapshot_save_restore();
    test_monolithic_snapshot_preserves_decode_state();
    test_monolithic_snapshot_disk_roundtrip();
    test_layer_split_snapshot_disk_roundtrip();
    test_spec_feature_tail_is_bounded();
    test_dspark_prefill_capture_boundaries();
    test_reset_request_state();
    test_reset_deepseek4_cache(backend);
    test_adapter_guard_paths();
    test_ipc_mode_registration();
    test_target_shard_daemon_validation();
    test_ffn_graph_reuse_microbench(backend);
    test_output_graph_reuse_microbench(backend);
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
    {
        auto gpu = ggml_backend_cuda_init(0);
        if (gpu) {
            for (int mode = 0; mode < 3; ++mode) test_dspark_compressor_rollback(gpu, mode);
            ggml_backend_free(gpu);
        } else {
            std::fprintf(stderr, "  test_dspark_compressor_rollback GPU skipped (no device)\n");
        }
    }
    test_ds4_flash_attention_keep_cap_gpu();
    test_ds4_flash_attention_streaming_topk_gpu();
    test_failed_init_preserves_sparse_opt_in();
    test_failed_init_preserves_mix_mmq_policy();
    test_ds4_flash_attention_parallel_index_scan_gpu(512);
    test_ds4_flash_attention_parallel_index_scan_gpu(1024);
    test_ds4_saved_raw_rows_replay_gpu();
    test_ds4_preserved_raw_rows_gpu();
    test_ds4_indexer_score_packed_small_gpu();
    test_ds4_flash_rope_replay_gpu();
    for (int ncols : {4160, 5121, 8192, 8193, 12288, 12289,
                     16384, 28673, 30720, 32768}) {
        test_ds4_topk_block_radix_gpu(ncols);
    }
    test_ds4_flash_attention_inverse_rope_fallback_gpu();
    test_hc_post_strided_split_gpu();
    test_hc_pre_kernel_gpu();
    test_layer_range_rejects_stale_hc_boundary();
    test_hc_scratch_per_device();
    test_hc_set_device_contract();
    test_cuda_graph_overrides_restore_nested_state();
    test_cuda_graph_rebuild_generation_guard();
    test_hc_scratch_shape_capacity();
#endif

    ggml_backend_free(backend);

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED: %d assertion(s)\n", g_failures);
        return 1;
    }

    std::printf("OK\n");
    return 0;
}
