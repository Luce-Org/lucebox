// Qwen35 daemon entry point.
//
// Thin wrapper: constructs a Qwen35Backend and hands off to the generic
// daemon loop (daemon_loop.cpp).

#pragma once

#include "common/backend_factory.h"
#include "device_placement.h"
#include <string>

namespace dflash::common {

struct Qwen35DaemonArgs {
    const char * target_path    = nullptr;
    const char * draft_path     = nullptr;
    DevicePlacement device;                   // target GPU placement
    int          draft_gpu      = 0;          // draft model GPU (arch-specific)
    int          stream_fd      = -1;
    int          chunk          = 512;

    // FA/KV
    int          fa_window      = 2048;
    int          kq_stride_pad  = 32;

    // Draft
    int          draft_swa_window = 0;
    int          draft_ctx_max    = 4096;

    // Speculative decode strategy
    bool         fast_rollback     = false;
    bool         seq_verify        = false;
    bool         ddtree_mode       = false;
    int          ddtree_budget     = 64;
    float        ddtree_temp       = 1.0f;
    bool         ddtree_chain_seed = true;
    bool         use_feature_mirror = false;

    // MTP (Multi-Token Prediction) speculator — mutually exclusive with draft.
    // The daemon uses BackendArgs directly; these fields mirror BackendArgs.
    MtpSource    mtp_source       = MtpSource::None;
    const char * mtp_gguf_path    = nullptr;   // required only for ExternalDrafter
    int          mtp_gamma        = 0;         // max speculation depth
    bool         mtp_use_topk     = false;     // false = chain, true = mtp_topk
    int          mtp_draft_topk   = 1;         // top-k for mtp_topk mode
};

// Run the qwen35 daemon loop. Returns 0 on clean exit, 1 on init failure.
int run_qwen35_daemon(const Qwen35DaemonArgs & args);

}  // namespace dflash::common
