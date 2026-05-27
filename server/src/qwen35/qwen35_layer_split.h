// Qwen35 multi-GPU layer-split daemon entry point.
//
// Shards the target model across multiple GPUs by layer ranges.
// Each shard gets its own CUDA backend, partial weights, and KV cache.
// Activations are transferred between shards at shard boundaries.
//
// NOTE: The full implementation currently lives in test/test_dflash.cpp
// (run_target_layer_split_daemon) because it depends on many helpers and
// globals defined there. This header defines the DevicePlacement-based
// args struct as the migration target. The implementation will move here
// once the helper functions (run_target_layer_split_forward, etc.) are
// extracted to src/qwen35/.

#pragma once

#include "placement/placement_config.h"

#include <string>

namespace dflash::common {

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

// Run the layer-split daemon. Returns 0 on clean shutdown.
// Currently forwarded to test_dflash.cpp's run_target_layer_split_daemon().
// Will be fully implemented here once helpers are extracted.
int run_qwen35_layer_split_daemon(const Qwen35LayerSplitDaemonArgs & args);

}  // namespace dflash::common
