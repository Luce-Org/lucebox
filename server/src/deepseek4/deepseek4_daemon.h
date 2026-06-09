// DeepSeek4 daemon entry point.

#pragma once

#include <string>

namespace dflash::common {

// Run the deepseek4 daemon loop. Called from main() when arch == "deepseek4".
// Reads commands from stdin, writes tokens to stream_fd.
int run_deepseek4_daemon(const char * model_path,
                          int gpu,
                          int stream_fd,
                          int max_ctx,
                          int chunk);

}  // namespace dflash::common
