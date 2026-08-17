#include "qwen3_tool_predictor_ipc.h"

#include "io_utils.h"
#include "model_backend.h"
#include "qwen3/qwen3_backend.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

namespace dflash::common {
namespace {

bool send_status(int stream_fd, int32_t status) {
    return write_exact_fd(stream_fd, &status, sizeof(status));
}

}  // namespace

int run_qwen3_tool_predictor_ipc_daemon(
        const char * model_path,
        int gpu,
        int max_ctx,
        int stream_fd) {
#if defined(_WIN32)
    (void)model_path; (void)gpu; (void)max_ctx; (void)stream_fd;
    return 2;
#else
    if (!model_path || !*model_path || stream_fd < 0 || max_ctx <= 0) {
        std::fprintf(stderr,
            "usage: backend_ipc_daemon --backend-ipc-mode=qwen3-tool-predict "
            "<qwen3.gguf> --stream-fd=FD --target-gpu=N --max-ctx=N\n");
        return 2;
    }

    Qwen3BackendConfig config;
    config.model_path = model_path;
    config.device.backend = PlacementBackend::Auto;
    config.device.gpu = std::max(0, gpu);
    config.device.max_ctx = max_ctx;
    config.chunk = 512;

    Qwen3Backend backend(config);
    if (!backend.init()) {
        std::fprintf(stderr, "[tool-predictor-daemon] Qwen3 init failed\n");
        send_status(stream_fd, -1);
        return 1;
    }
    std::fprintf(stderr,
        "[tool-predictor-daemon] ready gpu=%d max_ctx=%d\n",
        std::max(0, gpu), max_ctx);
    send_status(stream_fd, 0);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        std::string command;
        input >> command;
        if (command == "quit" || command == "exit") break;
        if (command != "predict") {
            std::fprintf(stderr,
                         "[tool-predictor-daemon] unknown command: %s\n",
                         line.c_str());
            send_status(stream_fd, -1);
            continue;
        }

        int max_tokens = 0;
        input >> max_tokens;
        const std::string path = read_line_tail(input);
        if (max_tokens <= 0 || path.empty()) {
            send_status(stream_fd, -1);
            continue;
        }
        const auto prompt = read_int32_file(path);
        if (prompt.empty() ||
            prompt.size() + static_cast<size_t>(max_tokens) >
                static_cast<size_t>(max_ctx)) {
            std::fprintf(stderr,
                "[tool-predictor-daemon] invalid context prompt=%zu max_tokens=%d max_ctx=%d\n",
                prompt.size(), max_tokens, max_ctx);
            send_status(stream_fd, -1);
            continue;
        }

        GenerateRequest request;
        request.prompt = prompt;
        request.n_gen = max_tokens;
        request.do_sample = false;
        request.stream = false;
        DaemonIO io;
        const GenerateResult result = backend.generate(request, io);
        if (!result.ok() || result.tokens.empty()) {
            std::fprintf(stderr,
                "[tool-predictor-daemon] generation failed code=%s\n",
                result.error_code().data());
            send_status(stream_fd, -1);
            continue;
        }

        const int32_t count = static_cast<int32_t>(result.tokens.size());
        if (!send_status(stream_fd, 0) ||
            !write_exact_fd(stream_fd, &count, sizeof(count)) ||
            !write_exact_fd(stream_fd, result.tokens.data(),
                            result.tokens.size() * sizeof(int32_t))) {
            std::fprintf(stderr,
                         "[tool-predictor-daemon] response write failed\n");
            break;
        }
    }

    backend.shutdown();
    std::fprintf(stderr, "[tool-predictor-daemon] stopped\n");
    return 0;
#endif
}

}  // namespace dflash::common
