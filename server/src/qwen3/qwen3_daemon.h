// Qwen3 daemon entry point.

#pragma once

#include "placement/placement_config.h"
#include <string>

namespace dflash::common {

struct Qwen3DaemonArgs {
    const char *    model_path = nullptr;
    DevicePlacement device;                // target GPU placement
    int             max_ctx    = 4096;
    int             stream_fd  = -1;
    int             chunk      = 512;
};

int run_qwen3_daemon(const Qwen3DaemonArgs & args);

}  // namespace dflash::common
