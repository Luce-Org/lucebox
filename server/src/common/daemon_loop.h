// Generic daemon command loop.
//
// Implements the stdin/stdout protocol shared by all model architectures
// (quit, park, unpark, compress, SNAPSHOT, RESTORE, FREE_SNAPSHOT,
// LIST_SLOTS, bare-prompt generate, file-based generate, sampler parsing).
// Dispatches model-specific operations through the ModelBackend interface.
//
// See model_backend.h for the backend contract.

#pragma once

#include "model_backend.h"

namespace dflash::common {

struct DaemonLoopArgs {
    int stream_fd = -1;
    int chunk     = 2048;    // chunked-prefill chunk size (forwarded to backend)
    int max_ctx   = 16384;   // max context (forwarded to backend for overflow check)
};

// Boots the model (via backend.print_ready_banner()), and services stdin
// commands until `quit`, `exit`, or EOF.  Returns 0 on clean shutdown.
int run_daemon(ModelBackend & backend, const DaemonLoopArgs & args);

}  // namespace dflash::common
