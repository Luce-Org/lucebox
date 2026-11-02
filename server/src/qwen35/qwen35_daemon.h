// Qwen35 daemon entry point.
//
// Thin wrapper: constructs a Qwen35Backend and hands off to the generic
// daemon loop (daemon_loop.cpp).

#pragma once

#include "placement/placement_config.h"
#include <string>

namespace luce::common {

struct Qwen35DaemonArgs {
    const char * target_path    = nullptr;
    const char * draft_path     = nullptr;
    DevicePlacement device;                   // target GPU placement
    int          draft_gpu      = 0;          // draft model GPU (arch-specific)
    int          stream_fd      = -1;
    int          chunk          = 512;

    // FA/KV
    int          fa_window      = 0;  // 0 = full attention. qwen3.6 full-attn layers must see the whole context; a finite window drops the system prompt/tools -> breaks tool calls.
    int          kq_stride_pad  = 32;

    // Draft
    int          draft_swa_window = 0;
    int          draft_ctx_max    = 4096;

    // Speculative decode strategy
    bool         fast_rollback     = true;
    bool         seq_verify        = false;
    bool         ddtree_mode       = false;
    int          ddtree_budget     = 22;
    float        ddtree_temp       = 1.0f;
    bool         ddtree_chain_seed = true;
    bool         use_feature_mirror = false;
};

// The daemon takes the pool size from LUCE_KVFLASH only. Returns 2 (after a
// one-line stderr message) for the combinations luce_server refuses at startup,
// --ddtree with KVFlash and LUCE_MIN_TOKENS with a draft model and KVFlash, and
// 0 otherwise. Both daemons call it before init.
int qwen35_daemon_kvflash_gate(const Qwen35DaemonArgs & args);

// Run the qwen35 daemon loop. Returns 0 on clean exit, 1 on init failure.
int run_qwen35_daemon(const Qwen35DaemonArgs & args);

}  // namespace luce::common
