// Qwen3 daemon entry point — thin wrapper around Qwen3Backend + run_daemon.

#include "qwen3_daemon.h"
#include "qwen3_backend.h"
#include "common/daemon_loop.h"

#include <cstdio>

namespace dflash::common {

int run_qwen3_daemon(const Qwen3DaemonArgs & args) {
    Qwen3BackendConfig cfg;
    cfg.model_path = args.model_path;
    cfg.device     = args.device;
    cfg.stream_fd  = args.stream_fd;
    cfg.chunk      = args.chunk;

    Qwen3Backend backend(cfg);
    if (!backend.init()) return 1;

    DaemonLoopArgs dargs;
    dargs.stream_fd = args.stream_fd;
    dargs.chunk     = args.chunk;
    dargs.max_ctx   = args.device.max_ctx;

    return run_daemon(backend, dargs);
}

}  // namespace dflash::common
