// Layer-split daemon loop — multi-GPU target inference with optional draft.
//
// This daemon accepts requests over stdin/stdout (same text protocol as
// daemon_loop) but splits target model layers across multiple GPUs. Each GPU
// owns a shard of layers and the activation tensor is forwarded between shards.
//
// Kept separate from daemon_loop.{h,cpp} because:
//   - It manages multiple TargetLayerSplitShards instead of a single ModelBackend
//   - Peer access / feature mirroring is shard-specific
//   - SNAPSHOT/RESTORE are not supported in layer-split mode

#pragma once

#include "internal.h"         // TargetWeights, TargetCache, DraftWeights
#include "common/sampler.h"
#include "dflash_feature_ring.h"
#include "qwen35/layer_split_types.h"

#include <string>
#include <vector>

namespace dflash27b {

struct LayerSplitDaemonConfig {
    const char * target_path = nullptr;
    const char * draft_path  = nullptr;
    std::vector<int> target_gpus;          // one GPU per shard
    std::vector<double> split_weights;     // fractional layer assignment
    int draft_gpu       = 0;
    bool load_draft     = false;
    bool run_dflash     = false;           // speculative decode mode
    int max_ctx         = 4096;
    int max_verify_tokens = 0;
    bool peer_access    = false;
    int stream_fd       = -1;

    // FA/KV parameters (previously globals).
    int kq_stride_pad   = 32;
    int fa_window        = 0;
    int draft_ctx_max    = 2048;
};

// Run the layer-split daemon event loop. Returns exit code (0 = success).
int run_layer_split_daemon(const LayerSplitDaemonConfig & cfg);

}  // namespace dflash27b
