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

    std::fprintf(stderr, "[deepseek4-target-shard] ready: layers=[%d,%d) on GPU %d\n",
                 layer_begins[0], layer_ends[0], gpus[0]);

    // Run the generic target shard IPC daemon loop
    // This receives forward requests, runs the layers, and returns results
    // TODO: implement the daemon loop using TargetShardIpcDaemon pattern
    // For now, signal readiness and enter a simple request loop

    // Signal ready
    if (stream_fd >= 0) {
        const char ready_msg[] = "ready\n";
        (void)write(stream_fd, ready_msg, sizeof(ready_msg) - 1);
    }

    // Simple request loop placeholder
    // In production, this would use the generic target shard daemon infrastructure
    std::fprintf(stderr, "[deepseek4-target-shard] entering daemon loop\n");

    // Cleanup
    free_deepseek4_cache(cache);
    free_deepseek4_weights(weights);
    ggml_backend_free(backend);
    return 0;
}

}  // namespace dflash::common
