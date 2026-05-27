// pflash_drafter_ipc_daemon.cpp - PFlash drafter IPC daemon body.

#include "pflash_drafter_ipc.h"

#include "dflash27b.h"
#include "dflash_draft_ipc.h"
#include "qwen3/qwen3_drafter.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <sstream>

namespace dflash::common {

int run_pflash_drafter_ipc_daemon(const char * drafter_path,
                                  int drafter_gpu,
                                  int stream_fd) {
#if defined(_WIN32)
    (void)drafter_path; (void)drafter_gpu; (void)stream_fd;
    std::fprintf(stderr, "PFlash drafter IPC daemon is only implemented on POSIX hosts\n");
    return 2;
#else
    if (!drafter_path || stream_fd < 0) {
        std::fprintf(stderr,
            "usage: backend_ipc_daemon --backend-ipc-mode=pflash-compress <drafter.gguf> "
            "--stream-fd=FD [--draft-gpu=N]\n");
        return 2;
    }

    DrafterContext ctx;
    if (!load_drafter(drafter_path, /*gpu_layers=*/999, std::max(0, drafter_gpu), ctx)) {
        std::fprintf(stderr, "[pflash-ipc-daemon] drafter load failed: %s\n",
                     dflash27b_last_error());
        stream_status(stream_fd, -1);
        return 1;
    }

    std::fprintf(stderr, "[pflash-ipc-daemon] ready gpu=%d\n", std::max(0, drafter_gpu));
    stream_status(stream_fd, 0);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        if (cmd == "quit" || cmd == "exit") break;
        if (cmd == "compress") {
            int keep_x1000 = 0;
            iss >> keep_x1000;
            std::string path = read_line_tail(iss);
            if (keep_x1000 < 0 || keep_x1000 > 1000 || path.empty()) {
                std::fprintf(stderr, "[pflash-ipc-daemon] bad compress: %s\n",
                             line.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            auto input_ids = read_int32_file(path);
            if (input_ids.empty()) {
                std::fprintf(stderr, "[pflash-ipc-daemon] read tokens failed: %s\n",
                             path.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            const float keep = (float)keep_x1000 / 1000.0f;
            auto compressed = drafter_score_and_compress(ctx, input_ids, keep);
            if (compressed.empty()) {
                std::fprintf(stderr, "[pflash-ipc-daemon] compress returned empty\n");
                stream_status(stream_fd, -1);
                continue;
            }
            const int32_t n_out = (int32_t)compressed.size();
            if (!stream_status(stream_fd, 0) ||
                !write_exact_fd(stream_fd, &n_out, sizeof(n_out)) ||
                !write_exact_fd(stream_fd, compressed.data(),
                                compressed.size() * sizeof(int32_t))) {
                std::fprintf(stderr, "[pflash-ipc-daemon] stream write failed\n");
                break;
            }
            continue;
        }
        std::fprintf(stderr, "[pflash-ipc-daemon] unknown command: %s\n", line.c_str());
        stream_status(stream_fd, -1);
    }

    free_drafter(ctx);
    std::fprintf(stderr, "[pflash-ipc-daemon] stopped\n");
    return 0;
#endif
}

} // namespace dflash::common
