// Gemma4 daemon entry point.

#pragma once

#include "placement/placement_config.h"
#include <string>

namespace dflash::common {

struct Gemma4DaemonArgs {
    const char *     model_path = nullptr;
    DevicePlacement  device;                // target GPU placement
    int              max_ctx    = 8192;
    int              stream_fd  = -1;
    int              chunk      = 512;
};

int run_gemma4_daemon(const Gemma4DaemonArgs & args);

}  // namespace dflash::common
