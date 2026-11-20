// Arguments for the Qwen35 multi-GPU layer-split daemon.
//
// Shards the target model across multiple GPUs by layer ranges.
// Each shard gets its own CUDA backend, partial weights, and KV cache.
// Activations are transferred between shards at shard boundaries.
//
// test/unit/test_dflash.cpp fills these args for its --target-gpus daemon
// mode (run_target_layer_split_daemon), which serves each request through
// run_qwen35_layer_split_request() in layer_split_daemon.{h,cpp}.

#pragma once

#include "placement/placement_config.h"

#include <string>

namespace luce::common {

struct Qwen35LayerSplitDaemonArgs {
    const char * target_path = nullptr;
    const char * draft_path  = nullptr;
    DevicePlacement device;               // must have layer_split_gpus populated
    int          draft_gpu   = 0;
    bool         load_draft  = false;
    bool         run_dflash  = false;
    int          max_verify_tokens = 32;
    int          stream_fd   = -1;
};

}  // namespace luce::common
