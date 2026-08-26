// Exact CPU contract test for the graph-integrated DSpark Markov head.
// The production CUDA graph uses the same builder after the drafter LM head;
// keeping this test on CPU makes the autoregressive token dependency cheap to
// exercise in every server-unit run.

#include "CppUnitTestFramework.hpp"

#include "common/dspark_head.h"
#include "internal.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

using namespace dflash::common;

namespace {

struct DsparkMarkovChainFixture {};

std::string write_draft_mask_fixture(uint32_t mask_token_id) {
    constexpr int n_embd = 4;
    constexpr int n_ff = 8;

    std::string path;
#if defined(_WIN32)
    char tmp_path[L_tmpnam]{};
    if (!std::tmpnam(tmp_path)) return {};
    path = tmp_path;
#else
    char tmp_path[] = "/tmp/lucebox_draft_mask_XXXXXX";
    const int fd = mkstemp(tmp_path);
    if (fd < 0) return {};
    close(fd);
    path = tmp_path;
#endif

    ggml_init_params params{};
    params.mem_size = 1u << 20;
    params.no_alloc = false;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return {};

    gguf_context * gguf = gguf_init_empty();
    const char * arch = "qwen35-dflash-draft";
    const std::string prefix = std::string(arch) + ".";
    gguf_set_val_str(gguf, "general.architecture", arch);
    gguf_set_val_u32(gguf, (prefix + "embedding_length").c_str(), n_embd);
    gguf_set_val_u32(gguf, (prefix + "block_count").c_str(), 1);
    gguf_set_val_u32(gguf, (prefix + "feed_forward_length").c_str(), n_ff);
    gguf_set_val_u32(gguf, (prefix + "attention.head_count").c_str(), 1);
    gguf_set_val_u32(gguf, (prefix + "attention.head_count_kv").c_str(), 1);
    gguf_set_val_u32(gguf, (prefix + "attention.key_length").c_str(), n_embd);
    gguf_set_val_f32(gguf, (prefix + "rope.freq_base").c_str(), 10000.0f);
    gguf_set_val_u32(gguf, (prefix + "dflash.block_size").c_str(), 2);
    gguf_set_val_u32(gguf, (prefix + "dflash.n_target_layers").c_str(), 1);
    gguf_set_val_u32(
        gguf, (prefix + "dflash.mask_token_id").c_str(), mask_token_id);

    auto add_1d = [&](const std::string & name, int64_t ne0) {
        ggml_tensor * tensor =
            ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
        ggml_set_name(tensor, name.c_str());
        std::memset(tensor->data, 0, ggml_nbytes(tensor));
        gguf_add_tensor(gguf, tensor);
    };
    auto add_2d = [&](const std::string & name, int64_t ne0, int64_t ne1) {
        ggml_tensor * tensor =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        ggml_set_name(tensor, name.c_str());
        std::memset(tensor->data, 0, ggml_nbytes(tensor));
        gguf_add_tensor(gguf, tensor);
    };

    add_2d("dflash.fc.weight", n_embd, n_embd);
    add_1d("dflash.hidden_norm.weight", n_embd);
    add_1d("output_norm.weight", n_embd);
    add_1d("blk.0.attn_norm.weight", n_embd);
    add_1d("blk.0.ffn_norm.weight", n_embd);
    add_2d("blk.0.attn_q.weight", n_embd, n_embd);
    add_2d("blk.0.attn_k.weight", n_embd, n_embd);
    add_2d("blk.0.attn_v.weight", n_embd, n_embd);
    add_2d("blk.0.attn_output.weight", n_embd, n_embd);
    add_1d("blk.0.attn_q_norm.weight", n_embd);
    add_1d("blk.0.attn_k_norm.weight", n_embd);
    add_2d("blk.0.ffn_gate.weight", n_embd, n_ff);
    add_2d("blk.0.ffn_up.weight", n_embd, n_ff);
    add_2d("blk.0.ffn_down.weight", n_ff, n_embd);

    gguf_write_to_file(gguf, path.c_str(), /*only_meta=*/false);
    gguf_free(gguf);
    ggml_free(ctx);
    return path;
}

struct MarkovFixture {
    ggml_backend_t backend = nullptr;
    ggml_context * weights_ctx = nullptr;
    ggml_backend_buffer_t weights_buf = nullptr;
    ggml_context * graph_ctx = nullptr;
    ggml_gallocr_t galloc = nullptr;
    std::vector<uint8_t> weights_arena;
    std::vector<uint8_t> graph_arena;

    DraftWeights weights{};
    ggml_cgraph * graph = nullptr;
    ggml_tensor * base = nullptr;
    ggml_tensor * seed = nullptr;
    std::vector<ggml_tensor *> tokens;

    ~MarkovFixture() {
        if (galloc) ggml_gallocr_free(galloc);
        if (graph_ctx) ggml_free(graph_ctx);
        if (weights_buf) ggml_backend_buffer_free(weights_buf);
        if (weights_ctx) ggml_free(weights_ctx);
        if (backend) ggml_backend_free(backend);
    }

    bool init() {
        constexpr int rank = 2;
        constexpr int vocab = 4;
        constexpr int positions = 3;

        backend = ggml_backend_cpu_init();
        if (!backend) return false;

        weights_arena.resize(ggml_tensor_overhead() * 4 + 1024);
        ggml_init_params wp{};
        wp.mem_size = weights_arena.size();
        wp.mem_buffer = weights_arena.data();
        wp.no_alloc = true;
        weights_ctx = ggml_init(wp);
        if (!weights_ctx) return false;

        ggml_tensor * w1 =
            ggml_new_tensor_2d(weights_ctx, GGML_TYPE_F32, rank, vocab);
        ggml_tensor * w2 =
            ggml_new_tensor_2d(weights_ctx, GGML_TYPE_F32, rank, vocab);
        weights_buf = ggml_backend_alloc_ctx_tensors(weights_ctx, backend);
        if (!weights_buf) return false;

        // w1[token] is the rank-two embedding selected by the previous token.
        const float w1_data[] = {
             1.0f,  0.0f,
             0.0f,  1.0f,
            -1.0f,  0.0f,
             0.0f, -1.0f,
        };
        // Each row maps the selected embedding to one vocabulary logit.
        const float w2_data[] = {
             2.0f,  0.0f,
             0.0f,  2.0f,
            -2.0f,  0.0f,
             0.0f, -2.0f,
        };
        ggml_backend_tensor_set(w1, w1_data, 0, sizeof(w1_data));
        ggml_backend_tensor_set(w2, w2_data, 0, sizeof(w2_data));

        weights.dspark.enabled = true;
        weights.dspark.markov_rank = rank;
        weights.dspark.vocab_size = vocab;
        weights.dspark.markov_w1 = w1;
        weights.dspark.markov_w2 = w2;

        graph_arena.resize(1024 * 1024);
        ggml_init_params gp{};
        gp.mem_size = graph_arena.size();
        gp.mem_buffer = graph_arena.data();
        gp.no_alloc = true;
        graph_ctx = ggml_init(gp);
        if (!graph_ctx) return false;

        graph = ggml_new_graph_custom(graph_ctx, 128, false);
        base = ggml_new_tensor_2d(
            graph_ctx, GGML_TYPE_F32, vocab, positions);
        seed = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_I32, 1);
        ggml_set_input(base);
        ggml_set_input(seed);
        if (!dspark_build_greedy_markov_chain(
                graph_ctx, graph, weights, base, seed, tokens)) {
            return false;
        }
        if (tokens.size() != positions) return false;

        galloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        if (!galloc || !ggml_gallocr_alloc_graph(galloc, graph)) return false;

        // Position-major base logits. The expected result depends on feeding
        // each corrected argmax into the next Markov lookup.
        const float base_data[] = {
            0.0f, 3.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 3.0f,
            4.0f, 0.0f, 0.0f, 0.0f,
        };
        ggml_backend_tensor_set(base, base_data, 0, sizeof(base_data));
        return true;
    }

    bool run(int32_t seed_token, std::vector<int32_t> & out) {
        ggml_backend_tensor_set(seed, &seed_token, 0, sizeof(seed_token));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            return false;
        }
        out.assign(tokens.size(), -1);
        for (size_t i = 0; i < tokens.size(); ++i) {
            ggml_backend_tensor_get_async(
                backend, tokens[i], &out[i], 0, sizeof(out[i]));
        }
        ggml_backend_synchronize(backend);
        return true;
    }
};

}  // namespace

TEST_CASE(DsparkMarkovChainFixture,
          integrated_chain_is_autoregressive_and_replayable) {
    MarkovFixture fixture;
    const bool initialized = fixture.init();
    CHECK(initialized);
    if (!initialized) return;

    std::vector<int32_t> from_zero;
    CHECK(fixture.run(/*seed_token=*/0, from_zero));
    CHECK(from_zero == std::vector<int32_t>({1, 1, 0}));

    // Reuse the allocated graph with a different seed, matching the persistent
    // CUDA-graph path used by speculative decode.
    std::vector<int32_t> from_three;
    CHECK(fixture.run(/*seed_token=*/3, from_three));
    CHECK(from_three == std::vector<int32_t>({3, 3, 0}));
}

TEST_CASE(DsparkMarkovChainFixture, integrated_chain_rejects_vocab_mismatch) {
    MarkovFixture fixture;
    const bool initialized = fixture.init();
    CHECK(initialized);
    if (!initialized) return;

    DraftWeights incompatible = fixture.weights;
    incompatible.dspark.vocab_size += 1;
    ggml_cgraph * graph =
        ggml_new_graph_custom(fixture.graph_ctx, 128, false);
    std::vector<ggml_tensor *> tokens;
    CHECK(!dspark_build_greedy_markov_chain(
        fixture.graph_ctx, graph, incompatible,
        fixture.base, fixture.seed, tokens));
    CHECK(tokens.empty());
}

TEST_CASE(DsparkMarkovChainFixture,
          draft_mask_metadata_overrides_target_family_default) {
    const std::string path = write_draft_mask_fixture(/*mask_token_id=*/7);
    CHECK(!path.empty());
    if (path.empty()) return;

    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr);
    if (!backend) {
        std::remove(path.c_str());
        return;
    }

    TargetWeights target{};
    target.n_vocab = 8;
    target.mask_token_id = DFLASH27B_DRAFT_MASK_TOKEN_ID;
    DraftWeights draft{};
    CHECK(load_draft_gguf(path, backend, draft, &target));
    CHECK(draft.mask_token_id == 7);

    free_draft_weights(draft);
    ggml_backend_free(backend);
    std::remove(path.c_str());
}

TEST_CASE(DsparkMarkovChainFixture,
          draft_mask_metadata_is_checked_against_target_vocabulary) {
    const std::string path = write_draft_mask_fixture(/*mask_token_id=*/8);
    CHECK(!path.empty());
    if (path.empty()) return;

    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr);
    if (!backend) {
        std::remove(path.c_str());
        return;
    }

    TargetWeights target{};
    target.n_vocab = 8;
    DraftWeights draft{};
    CHECK(!load_draft_gguf(path, backend, draft, &target));
    CHECK(std::string(dflash27b_last_error()).find("outside vocabulary") !=
          std::string::npos);

    free_draft_weights(draft);
    ggml_backend_free(backend);
    std::remove(path.c_str());
}
