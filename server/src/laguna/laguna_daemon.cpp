// Laguna daemon entry point.
//
// Thin wrapper: constructs a LagunaBackend and hands off to the generic
// daemon loop (daemon_loop.cpp). All model-specific logic lives in
// laguna_backend.cpp; protocol plumbing lives in daemon_loop.cpp.
//
// Invoked from two places:
//   - test/test_dflash.cpp: arch dispatch — when the GGUF reports
//     `general.architecture == "laguna"`, main() builds a LagunaDaemonArgs
//     and calls run_laguna_daemon().
//   - test/test_laguna_daemon.cpp: thin CLI wrapper for the NIAH driver.

#include "laguna_daemon.h"
#include "laguna_backend.h"
#include "daemon_loop.h"

#include <cstdio>

namespace dflash::common {

int run_laguna_daemon(const LagunaDaemonArgs & args) {
    LagunaBackendArgs bargs;
    bargs.target_path = args.target_path;
    bargs.max_ctx     = args.device.max_ctx;
    bargs.chunk       = args.chunk;
    bargs.kv_type     = args.kv_type;

    LagunaBackend backend(bargs);
    if (!backend.init()) return 1;

    DaemonLoopArgs dargs;
    dargs.stream_fd = args.stream_fd;
    dargs.chunk     = args.chunk;
    dargs.max_ctx   = args.device.max_ctx;

    return run_daemon(backend, dargs);
}

}  // namespace dflash::common
