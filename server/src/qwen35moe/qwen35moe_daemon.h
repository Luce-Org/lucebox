// Qwen35MoE daemon entry point.

#pragma once

#include "qwen35_daemon.h"

namespace dflash::common {

using Qwen35MoeDaemonArgs = Qwen35DaemonArgs;

int run_qwen35moe_daemon(const Qwen35MoeDaemonArgs & args);

}  // namespace dflash::common
