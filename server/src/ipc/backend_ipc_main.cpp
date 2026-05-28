// Standalone backend IPC daemon entry point.

#include "backend_ipc.h"
#include "dflash_draft_ipc.h"
#include "pflash_drafter_ipc.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace dflash::common;

int main(int argc, char ** argv) {
    BackendIpcMode mode = BackendIpcMode::DFlashDraft;
    const char * payload_path = nullptr;
    int arg_begin = 0;

    if (argc >= 3 && std::strncmp(argv[1], "--backend-ipc-mode=", 19) == 0) {
        std::string mode_name(argv[1] + 19);
        if (!parse_backend_ipc_mode(mode_name, mode)) {
            std::fprintf(stderr, "[backend-ipc-daemon] unknown mode: %s\n",
                         mode_name.c_str());
            return 2;
        }
        payload_path = argv[2];
        arg_begin = 3;
    } else if (argc >= 3 && std::strcmp(argv[1], "--backend-ipc-mode") == 0) {
        if (!parse_backend_ipc_mode(argv[2], mode) || argc < 4) {
            std::fprintf(stderr, "[backend-ipc-daemon] bad --backend-ipc-mode\n");
            return 2;
        }
        payload_path = argv[3];
        arg_begin = 4;
    } else {
        std::fprintf(stderr,
            "usage: %s --backend-ipc-mode=dflash-draft <draft.safetensors|draft.gguf> "
            "--ring-cap=N --stream-fd=FD [--payload-fd=FD] [--draft-gpu=N]\n"
            "   or: %s --backend-ipc-mode=pflash-compress <drafter.gguf> "
            "--stream-fd=FD [--draft-gpu=N]\n",
            argv[0],
            argv[0]);
        return 2;
    }

    int ring_cap = 4096;
    int draft_gpu = 0;
    int payload_fd = -1;
    int stream_fd = -1;
    for (int i = arg_begin; i < argc; i++) {
        if (std::strncmp(argv[i], "--ring-cap=", 11) == 0) {
            ring_cap = std::atoi(argv[i] + 11);
        } else if (std::strcmp(argv[i], "--ring-cap") == 0) {
            if (i + 1 < argc) ring_cap = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--draft-gpu=", 12) == 0) {
            draft_gpu = std::max(0, std::atoi(argv[i] + 12));
        } else if (std::strcmp(argv[i], "--draft-gpu") == 0) {
            if (i + 1 < argc) draft_gpu = std::max(0, std::atoi(argv[++i]));
        } else if (std::strncmp(argv[i], "--stream-fd=", 12) == 0) {
            stream_fd = std::atoi(argv[i] + 12);
        } else if (std::strcmp(argv[i], "--stream-fd") == 0) {
            if (i + 1 < argc) stream_fd = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--payload-fd=", 13) == 0) {
            payload_fd = std::atoi(argv[i] + 13);
        } else if (std::strcmp(argv[i], "--payload-fd") == 0) {
            if (i + 1 < argc) payload_fd = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "[backend-ipc-daemon] unknown option: %s\n", argv[i]);
            return 2;
        }
    }

    switch (mode) {
        case BackendIpcMode::DFlashDraft:
            return run_dflash_draft_ipc_daemon(payload_path, ring_cap, draft_gpu,
                                               stream_fd, payload_fd);
        case BackendIpcMode::PFlashCompress:
            return run_pflash_drafter_ipc_daemon(payload_path, draft_gpu, stream_fd);
    }
    std::fprintf(stderr, "[backend-ipc-daemon] unsupported mode\n");
    return 2;
}
