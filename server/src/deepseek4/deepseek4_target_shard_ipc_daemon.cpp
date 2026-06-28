// DeepSeek4 target shard IPC daemon.
//
// Runs on the remote Halo (AMD HIP) GPU, receives boundary activations from the
// CUDA parent shard, runs layers [begin, end) + optionally the output head, and
// returns either the boundary activation or logits.

#include "deepseek4_layer_split_adapter.h"
#include "deepseek4_internal.h"
#include "common/target_shard_ipc.h"
#include "common/target_shard_ipc_daemon.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace dflash::common {

int run_deepseek4_target_shard_ipc_daemon(
        const char * target_path,
        const std::vector<int> & gpus,
        const std::vector<int> & layer_begins,
        const std::vector<int> & layer_ends,
        int max_ctx,
        int stream_fd,
        int payload_fd) {
    if (!target_path || gpus.empty() || layer_begins.empty() || layer_ends.empty()) {
        std::fprintf(stderr, "[deepseek4-target-shard] invalid arguments\n");
        return 1;
    }

    std::fprintf(stderr, "[deepseek4-target-shard] starting: gpu=%d layers=[%d,%d) max_ctx=%d\n",
                 gpus[0], layer_begins[0], layer_ends[0], max_ctx);

    // Initialize backend for the Halo GPU
    ggml_backend_t backend = ggml_backend_cuda_init(gpus[0]);
    if (!backend) {
        std::fprintf(stderr, "[deepseek4-target-shard] failed to init GPU %d\n", gpus[0]);
        return 1;
    }

    // Load weights for this shard's layer range
    TargetLoadPlan plan;
    plan.layer_begin = layer_begins[0];
    plan.layer_end = layer_ends[0];
    plan.load_output = true;  // Last shard computes output

    DeepSeek4Weights weights;
    if (!load_deepseek4_gguf_partial(target_path, backend, plan, weights)) {
        std::fprintf(stderr, "[deepseek4-target-shard] failed to load weights\n");
        ggml_backend_free(backend);
        return 1;
    }

    // Create KV cache
    DeepSeek4Cache cache;
    if (!create_deepseek4_cache(backend, weights, max_ctx, cache)) {
        std::fprintf(stderr, "[deepseek4-target-shard] failed to create cache\n");
        free_deepseek4_weights(weights);
        ggml_backend_free(backend);
        return 1;
    }

    std::fprintf(stderr, "[deepseek4-target-shard] loaded: layers=[%d,%d) on GPU %d\n",
                 layer_begins[0], layer_ends[0], gpus[0]);

    const int hidden = weights.n_embd * weights.n_hc;  // full HC state dimension
    const int n_hc = weights.n_hc;
    std::vector<float> hc_state((size_t)n_hc * hidden, 0.0f);

    // Set up daemon callbacks
    TargetShardDaemonCallbacks callbacks;
    callbacks.log_prefix = "deepseek4-target-shard";

    callbacks.forward = [&](const TargetShardDaemonForwardRequest & req,
                            TargetShardDaemonForwardResponse & resp) -> bool {
        const int n_tokens = req.n_tokens;
        if (n_tokens <= 0) return false;

        // The boundary activation from the CUDA shard is the full HC state
        // (n_hc * n_embd * n_tokens floats).
        const float * input_embed = nullptr;
        if (req.boundary_activation && !req.boundary_activation->empty()) {
            input_embed = req.boundary_activation->data();
        }
        if (!input_embed) {
            std::fprintf(stderr, "[deepseek4-target-shard] no boundary activation\n");
            return false;
        }

        // Run the layer range using the full-model step function.
        // The weights struct has layers [layer_begin..layer_end) populated,
        // and deepseek4_step_layer_range handles routing to the correct layers.
        std::fprintf(stderr, "[deepseek4-target-shard] forward: n_tokens=%d base_pos=%d layers=[%d,%d)\n",
                     n_tokens, req.base_pos, plan.layer_begin, plan.layer_end);
        std::vector<float> logits;
        bool ok = deepseek4_step_layer_range(
            backend, weights, cache, hc_state,
            input_embed, n_tokens, req.base_pos,
            plan.layer_begin, plan.layer_end,
            &logits, nullptr, nullptr);

        if (!ok) {
            std::fprintf(stderr, "[deepseek4-target-shard] forward failed\n");
            return false;
        }

        // Compute argmax from last token's logits
        // Note: deepseek4_step_layer_range returns only last token's logits (size = vocab)
        if (!logits.empty()) {
            resp.logits = std::move(logits);
            const int vocab = weights.n_vocab;
            const float * last_logits = resp.logits.data();
            int32_t best = 0;
            float best_val = last_logits[0];
            for (int i = 1; i < vocab; ++i) {
                if (last_logits[i] > best_val) {
                    best_val = last_logits[i];
                    best = i;
                }
            }
            resp.last_tok = best;
        }

        cache.cur_pos = req.base_pos + n_tokens;
        return ok;
    };

    callbacks.reset_request_state = [&]() -> bool {
        cache.cur_pos = 0;
        for (auto & lc : cache.layers) {
            lc.n_comp = 0;
            lc.n_index_comp = 0;
        }
        std::fill(hc_state.begin(), hc_state.end(), 0.0f);
        return true;
    };

    callbacks.snapshot_save = [&](int slot) -> bool {
        (void)slot;
        // TODO: implement snapshot for daemon shard
        return true;
    };

    callbacks.snapshot_free = [&](int slot) {
        (void)slot;
    };

    callbacks.snapshot_restore = [&](int slot) -> bool {
        (void)slot;
        // TODO: implement snapshot restore for daemon shard
        return true;
    };

    std::fprintf(stderr, "[deepseek4-target-shard] ready: layers=[%d,%d) hidden=%d vocab=%d\n",
                 layer_begins[0], layer_ends[0], hidden, weights.n_vocab);

    const int rc = run_target_shard_ipc_daemon_loop(
        hidden, weights.n_vocab,
        stream_fd, payload_fd,
        /*shared_payload_fd=*/-1, /*shared_payload_bytes=*/0,
        std::move(callbacks));

    // Cleanup
    free_deepseek4_cache(cache);
    free_deepseek4_weights(weights);
    ggml_backend_free(backend);
    return rc;
}

}  // namespace dflash::common
