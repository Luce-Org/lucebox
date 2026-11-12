// Qwen35 daemon entry point.
//
// Thin wrapper: constructs a Qwen35Backend and hands off to the generic
// daemon loop (daemon_loop.cpp). All model-specific logic lives in
// qwen35_backend.cpp; protocol plumbing lives in daemon_loop.cpp.

#include "qwen35_daemon.h"
#include "qwen35_backend.h"
#include "common/daemon_loop.h"
#include "common/kvflash_pool_flag.h"
#include "common/kvflash_refusals.h"

#include <cstdio>
#include <cstdlib>

namespace luce::common {

int qwen35_daemon_kvflash_gate(const Qwen35DaemonArgs & args) {
    const char * pool = std::getenv("LUCE_KVFLASH");
    std::string refusal = kvflash_ddtree_refusal(args.ddtree_mode, kvflash_pool_requested(pool));
    if (refusal.empty())
        refusal = kvflash_min_tokens_refusal(kvflash_pool_requested(pool) && args.draft_path,
                                             std::getenv("LUCE_MIN_TOKENS"));
    if (refusal.empty()) return 0;
    std::fprintf(stderr, "[kvflash] %s\n", refusal.c_str());
    return 2;
}

int run_qwen35_daemon(const Qwen35DaemonArgs & args) {
    if (const int rc = qwen35_daemon_kvflash_gate(args)) return rc;
    Qwen35Config cfg;
    cfg.target_path        = args.target_path ? args.target_path : "";
    if (args.draft_path) cfg.draft_path = args.draft_path;
    cfg.device             = args.device;
    cfg.draft_gpu          = args.draft_gpu;
    cfg.stream_fd          = args.stream_fd;
    cfg.fa_window          = args.fa_window;
    cfg.kq_stride_pad      = args.kq_stride_pad;
    cfg.draft_swa_window   = args.draft_swa_window;
    cfg.draft_ctx_max      = args.draft_ctx_max;
    cfg.fast_rollback      = args.fast_rollback;
    cfg.seq_verify         = args.seq_verify;
    cfg.ddtree_mode        = args.ddtree_mode;
    cfg.ddtree_budget      = args.ddtree_budget;
    cfg.ddtree_temp        = args.ddtree_temp;
    cfg.ddtree_chain_seed  = args.ddtree_chain_seed;
    cfg.use_feature_mirror = args.use_feature_mirror;

    Qwen35Backend backend(cfg);
    if (!backend.init()) return 1;

    DaemonLoopArgs dargs;
    dargs.stream_fd = args.stream_fd;
    dargs.chunk     = args.chunk;
    dargs.max_ctx   = args.device.max_ctx;

    return run_daemon(backend, dargs);
}

}  // namespace luce::common
