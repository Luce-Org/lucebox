#include "qwen3_tool_predictor_ipc.h"

#include "io_utils.h"

#include <algorithm>
#include <cstdio>

namespace dflash::common {

bool Qwen3ToolPredictorIpcClient::start(
        const std::string & bin,
        const std::string & model_path,
        int gpu,
        int max_ctx,
        const std::string & work_dir) {
#if defined(_WIN32)
    (void)bin; (void)model_path; (void)gpu; (void)max_ctx; (void)work_dir;
    std::fprintf(stderr,
                 "Qwen3 tool-predictor IPC is only implemented on POSIX hosts\n");
    return false;
#else
    std::lock_guard<std::mutex> lock(mutex_);
    close_locked();
    if (bin.empty() || model_path.empty() || max_ctx <= 0) return false;

    BackendIpcLaunchConfig launch;
    launch.bin = bin;
    launch.mode = BackendIpcMode::Qwen3ToolPredict;
    launch.payload_path = model_path;
    launch.work_dir = work_dir;
    launch.args.push_back("--target-gpu=" + std::to_string(std::max(0, gpu)));
    launch.args.push_back("--max-ctx=" + std::to_string(max_ctx));
    if (!process_.start(launch)) {
        std::fprintf(stderr, "[tool-predictor-ipc] backend process start failed\n");
        return false;
    }
    active_ = true;
    std::fprintf(stderr,
        "[tool-predictor-ipc] ready model=%s gpu=%d max_ctx=%d work_dir=%s\n",
        model_path.c_str(), std::max(0, gpu), max_ctx,
        process_.work_dir().c_str());
    return true;
#endif
}

bool Qwen3ToolPredictorIpcClient::predict(
        const std::vector<int32_t> & prompt_ids,
        int max_tokens,
        std::vector<int32_t> & output_ids,
        std::string & error) {
    output_ids.clear();
    error.clear();
#if defined(_WIN32)
    (void)prompt_ids; (void)max_tokens;
    error = "native_predictor_ipc_unsupported";
    return false;
#else
    std::lock_guard<std::mutex> lock(mutex_);
    FILE * command = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    if (!active_ || !command || stream_fd < 0) {
        error = "native_predictor_not_active";
        return false;
    }
    if (prompt_ids.empty() || max_tokens <= 0) {
        error = "native_predictor_invalid_request";
        return false;
    }

    const std::string path = process_.next_path("tool_predictor_prompt");
    if (!write_int32_file(path, prompt_ids)) {
        error = "native_predictor_prompt_write_failed";
        return false;
    }

    std::fprintf(command, "predict %d %s\n", max_tokens, path.c_str());
    std::fflush(command);

    int32_t status = -1;
    bool ok = read_exact_fd(stream_fd, &status, sizeof(status)) && status == 0;
    if (ok) {
        int32_t count = -1;
        ok = read_exact_fd(stream_fd, &count, sizeof(count)) &&
             count > 0 && count <= max_tokens;
        if (ok) {
            output_ids.assign(static_cast<size_t>(count), 0);
            ok = read_exact_fd(stream_fd, output_ids.data(),
                               output_ids.size() * sizeof(int32_t));
        }
    }
    std::remove(path.c_str());
    if (!ok) {
        error = status == 0
            ? "native_predictor_invalid_response"
            : "native_predictor_generation_failed";
        output_ids.clear();
        close_locked();
        return false;
    }
    return true;
#endif
}

bool Qwen3ToolPredictorIpcClient::active() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_;
}

void Qwen3ToolPredictorIpcClient::close_locked() {
    process_.close();
    active_ = false;
}

void Qwen3ToolPredictorIpcClient::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    close_locked();
}

}  // namespace dflash::common
