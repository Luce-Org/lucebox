// Offline routing-stat collection for qwen35moe.
//
// Usage:
//   collect_qwen35moe_routing_stats <qwen35moe.gguf> <out.json> <prompt1.bin> [prompt2.bin ...]
//
// Each prompt file is a raw int32 token stream (uncounted). The tool runs
// chunked prefill with capture_moe_router enabled and accumulates per-layer
// expert selection counts into Qwen35MoeRoutingStats.

#include "internal.h"
#include "common/attn_masks.h"
#include "common/io_utils.h"
#include "graph_builders.h"
#include "qwen35moe_routing_stats.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dflash::common;

static bool process_prompt(const std::vector<int32_t> & tokens,
                           const TargetWeights & w,
                           TargetCache & cache,
                           ggml_backend_t backend,
                           Qwen35MoeRoutingStats & stats,
                           int ubatch) {
    if (tokens.empty()) return true;

    reset_target_cache(cache);
    StepGraph sg;
    std::vector<float> embed_buf((size_t)w.n_embd * (size_t)ubatch);
    int committed = 0;

    for (int start = 0; start < (int)tokens.size();) {
        const int n_tokens = std::min(ubatch, (int)tokens.size() - start);
        const bool with_mask = n_tokens > 1;

        if (!build_target_step(sg, w, cache, backend,
                               /*kv_start=*/committed,
                               /*n_tokens=*/n_tokens,
                               with_mask,
                               /*capture=*/false,
                               /*capture_delta_intermediate=*/false,
                               /*fa_window=*/0,
                               /*last_token_logits_only=*/false,
                               KQ_MASK_PAD,
                               /*capture_moe_router=*/true)) {
            std::fprintf(stderr, "build_target_step failed at chunk start=%d\n", start);
            step_graph_destroy(sg);
            return false;
        }

        if (!w.embedder.embed(tokens.data() + start, n_tokens, embed_buf.data())) {
            std::fprintf(stderr, "embed failed at chunk start=%d\n", start);
            step_graph_destroy(sg);
            return false;
        }
        ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0,
                                sizeof(float) * (size_t)w.n_embd * (size_t)n_tokens);

        std::vector<int32_t> pos_buf((size_t)4 * (size_t)n_tokens, 0);
        for (int i = 0; i < n_tokens; ++i) {
            const int p = committed + i;
            pos_buf[4 * i + 0] = p;
            pos_buf[4 * i + 1] = p;
            pos_buf[4 * i + 2] = p;
            pos_buf[4 * i + 3] = 0;
        }
        ggml_backend_tensor_set(sg.positions, pos_buf.data(), 0,
                                sizeof(int32_t) * pos_buf.size());

        if (sg.attn_mask) {
            std::vector<uint16_t> mask_buf;
            const int kv_pad_override = (int)sg.attn_mask->ne[0];
            build_causal_mask(mask_buf, committed + n_tokens, n_tokens, committed,
                              KQ_MASK_PAD, /*win_start=*/0, kv_pad_override);
            ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                    sizeof(uint16_t) * mask_buf.size());
        }

        auto st = ggml_backend_graph_compute(backend, sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "compute failed at chunk start=%d status=%d\n", start, (int)st);
            step_graph_destroy(sg);
            return false;
        }

        std::string err;
        for (int il = 0; il < w.n_layer; ++il) {
            ggml_tensor * selected = (il < (int)sg.moe_selected.size()) ? sg.moe_selected[(size_t)il] : nullptr;
            if (!selected) continue;
            if (!stats.observe_selected_tensor(backend, il, selected, &err)) {
                std::fprintf(stderr, "observe_selected_tensor layer=%d failed: %s\n", il, err.c_str());
                step_graph_destroy(sg);
                return false;
            }
        }

        committed += n_tokens;
        cache.cur_pos = committed;
        start += n_tokens;
    }

    step_graph_destroy(sg);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <qwen35moe.gguf> <out.json> <prompt1.bin> [prompt2.bin ...]\n", argv[0]);
        return 2;
    }

    const char * model_path = argv[1];
    const char * out_path   = argv[2];

    std::vector<std::vector<int32_t>> prompts;
    size_t max_prompt = 0;
    for (int i = 3; i < argc; ++i) {
        std::vector<int32_t> ids = read_int32_file(argv[i]);
        if (ids.empty()) {
            std::fprintf(stderr, "empty or unreadable prompt file: %s\n", argv[i]);
            return 1;
        }
        max_prompt = std::max(max_prompt, ids.size());
        prompts.push_back(std::move(ids));
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "cuda init failed\n");
        return 1;
    }

    TargetWeights w;
    if (!load_target_gguf(model_path, backend, w)) {
        std::fprintf(stderr, "load_target_gguf: %s\n", dflash27b_last_error());
        return 1;
    }
    if (!w.is_moe) {
        std::fprintf(stderr, "target is not qwen35moe\n");
        return 1;
    }

    const int max_ctx = (int)std::max<size_t>(64, max_prompt + 8);
    TargetCache cache;
    if (!create_target_cache(w, max_ctx, /*max_verify_tokens=*/0, backend, cache)) {
        std::fprintf(stderr, "create_target_cache: %s\n", dflash27b_last_error());
        return 1;
    }

    Qwen35MoeRoutingStats stats;
    if (!stats.init_from_weights(w)) {
        std::fprintf(stderr, "routing stats init failed\n");
        return 1;
    }

    int ubatch = 512;
    if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
        ubatch = std::max(1, std::atoi(s));
    }

    for (size_t i = 0; i < prompts.size(); ++i) {
        if (!process_prompt(prompts[i], w, cache, backend, stats, ubatch)) {
            std::fprintf(stderr, "failed processing prompt %zu\n", i);
            return 1;
        }
    }

    std::string err;
    if (!stats.save_csv(out_path, &err)) {
        std::fprintf(stderr, "save_csv failed: %s\n", err.c_str());
        return 1;
    }

    free_target_cache(cache);
    free_target_weights(w);
    ggml_backend_free(backend);
    std::printf("saved routing stats to %s\n", out_path);
    return 0;
}
