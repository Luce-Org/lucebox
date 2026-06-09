// DeepSeek4 daemon entry point implementation.

#include "deepseek4_daemon.h"
#include "deepseek4_backend.h"
#include "common/daemon_loop.h"

#include <cstdio>

namespace dflash::common {

int run_deepseek4_daemon(const char * model_path,
                          int gpu,
                          int stream_fd,
                          int max_ctx,
                          int chunk) {
    DeepSeek4BackendConfig cfg;
    cfg.model_path = model_path;
    cfg.device.gpu = gpu;
    cfg.stream_fd  = stream_fd;
    cfg.max_ctx    = max_ctx;
    cfg.chunk      = chunk > 0 ? chunk : 512;

    auto backend = std::make_unique<DeepSeek4Backend>(cfg);
    if (!backend->init()) {
        std::fprintf(stderr, "[deepseek4-daemon] init failed\n");
        return 1;
    }

    DaemonLoopArgs loop_args;
    loop_args.stream_fd = stream_fd;
    loop_args.chunk     = cfg.chunk;
    loop_args.max_ctx   = max_ctx;
    return run_daemon(*backend, loop_args);
}

}  // namespace dflash::common
