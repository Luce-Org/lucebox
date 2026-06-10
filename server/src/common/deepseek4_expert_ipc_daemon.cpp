// deepseek4_expert_ipc_daemon.cpp - DeepSeek4 expert IPC worker boundary.

#include "deepseek4_expert_ipc.h"
#include "dflash_draft_ipc.h"
#include "io_utils.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <sstream>

namespace dflash::common {

int run_deepseek4_expert_ipc_daemon(const char * model_path,
                                    int worker_gpu,
                                    int stream_fd,
                                    int payload_fd) {
#if defined(_WIN32)
    (void)model_path; (void)worker_gpu; (void)stream_fd; (void)payload_fd;
    std::fprintf(stderr, "DeepSeek4 expert IPC daemon is only implemented on POSIX hosts\n");
    return 2;
#else
    (void)payload_fd;
    if (!model_path || stream_fd < 0) {
        std::fprintf(stderr,
            "usage: backend_ipc_daemon --backend-ipc-mode=deepseek4-expert <model.gguf> "
            "--stream-fd=FD [--payload-fd=FD] [--draft-gpu=N]\n");
        return 2;
    }

    std::fprintf(stderr, "[deepseek4-expert-ipc-daemon] ready gpu=%d model=%s\n",
                 std::max(0, worker_gpu), model_path);
    stream_status(stream_fd, 0);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        if (cmd == "quit" || cmd == "exit") break;
        if (cmd == "ping") {
            stream_status(stream_fd, 0);
            continue;
        }
        if (cmd == "eval") {
            std::fprintf(stderr,
                         "[deepseek4-expert-ipc-daemon] eval not implemented yet: %s\n",
                         line.c_str());
            stream_status(stream_fd, -2);
            continue;
        }
        std::fprintf(stderr, "[deepseek4-expert-ipc-daemon] unknown command: %s\n", line.c_str());
        stream_status(stream_fd, -1);
    }

    std::fprintf(stderr, "[deepseek4-expert-ipc-daemon] stopped\n");
    return 0;
#endif
}

} // namespace dflash::common
