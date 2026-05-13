// Gemma4 daemon entry point.
//
// Thin wrapper: constructs a Gemma4Backend and hands off to the generic
// daemon loop (daemon_loop.cpp from #175). All model-specific logic lives
// in gemma4_backend.cpp (#11b); protocol plumbing lives in daemon_loop.cpp.

#include "gemma4_daemon.h"
#include "../common/daemon_loop.h"

#include <cstdio>

namespace dflash27b {

int run_gemma4_daemon(const Gemma4DaemonArgs & args) {
    Gemma4BackendArgs bargs;
    bargs.target_path  = args.target_path;
    bargs.draft_path   = args.draft_path;
    bargs.mtp_path     = args.mtp_path;
    bargs.draft_method = args.draft_method;
    bargs.max_ctx      = args.max_ctx;
    bargs.chunk        = args.chunk;
    bargs.use_pflash   = args.use_pflash;
    bargs.pflash_alpha = args.pflash_alpha;
    bargs.draft_kv_cap_override          = args.draft_kv_cap_override;
    bargs.draft_max_block                = args.draft_max_block;
    bargs.draft_enable_capture_overrides = args.draft_enable_capture_overrides;
    bargs.mtp_gamma                      = args.mtp_gamma;

    Gemma4Backend backend(bargs);
    if (!backend.init()) {
        std::fprintf(stderr, "[gemma4-daemon] backend init failed\n");
        return 1;
    }

    DaemonLoopArgs dargs;
    dargs.stream_fd = args.stream_fd;
    dargs.chunk     = args.chunk;
    dargs.max_ctx   = args.max_ctx;

    return run_daemon(backend, dargs);
}

}  // namespace dflash27b
