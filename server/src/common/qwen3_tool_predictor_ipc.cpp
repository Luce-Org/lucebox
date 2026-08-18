#include "qwen3_tool_predictor_ipc.h"

#include "io_utils.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <system_error>

namespace dflash::common {

bool read_qwen3_tool_predictor_response(
        int stream_fd,
        int max_tokens,
        int timeout_ms,
        std::vector<int32_t> & output_ids,
        std::string & error) {
    output_ids.clear();
    error.clear();
#if defined(_WIN32)
    (void)stream_fd;
    (void)max_tokens;
    (void)timeout_ms;
    error = "native_predictor_ipc_unsupported";
    return false;
#else
    if (stream_fd < 0 || max_tokens <= 0 || timeout_ms <= 0) {
        error = "native_predictor_invalid_request";
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(timeout_ms);
    bool timed_out = false;
    int32_t status = -1;
    if (!read_exact_fd_until(
            stream_fd, &status, sizeof(status), deadline, timed_out)) {
        error = timed_out
            ? "native_predictor_timeout"
            : "native_predictor_invalid_response";
        return false;
    }
    if (status != 0) {
        error = "native_predictor_generation_failed";
        return false;
    }

    int32_t count = -1;
    if (!read_exact_fd_until(
            stream_fd, &count, sizeof(count), deadline, timed_out) ||
        count <= 0 || count > max_tokens) {
        error = timed_out
            ? "native_predictor_timeout"
            : "native_predictor_invalid_response";
        return false;
    }
    output_ids.assign(static_cast<size_t>(count), 0);
    if (!read_exact_fd_until(
            stream_fd, output_ids.data(),
            output_ids.size() * sizeof(int32_t), deadline, timed_out)) {
        output_ids.clear();
        error = timed_out
            ? "native_predictor_timeout"
            : "native_predictor_invalid_response";
        return false;
    }
    return true;
#endif
}

bool Qwen3ToolPredictorIpcClient::start(
        const std::string & bin,
        const std::string & model_path,
        int gpu,
        int max_ctx,
        const std::string & work_dir,
        int readiness_timeout_ms) {
#if defined(_WIN32)
    (void)bin; (void)model_path; (void)gpu; (void)max_ctx; (void)work_dir;
    (void)readiness_timeout_ms;
    std::fprintf(stderr,
                 "Qwen3 tool-predictor IPC is only implemented on POSIX hosts\n");
    return false;
#else
    std::lock_guard<std::timed_mutex> lock(mutex_);
    close_locked();
    if (bin.empty() || model_path.empty() || max_ctx <= 0 ||
        readiness_timeout_ms <= 0) return false;

    std::error_code path_error;
    const std::string resolved_bin =
        std::filesystem::canonical(bin, path_error).string();
    if (path_error) {
        std::fprintf(stderr,
                     "[tool-predictor-ipc] cannot resolve IPC binary: %s\n",
                     path_error.message().c_str());
        return false;
    }
    const std::string resolved_model =
        std::filesystem::canonical(model_path, path_error).string();
    if (path_error) {
        std::fprintf(stderr,
                     "[tool-predictor-ipc] cannot resolve model: %s\n",
                     path_error.message().c_str());
        return false;
    }
    BackendIpcLaunchConfig launch;
    launch.bin = resolved_bin;
    launch.mode = BackendIpcMode::Qwen3ToolPredict;
    launch.payload_path = resolved_model;
    launch.work_dir = work_dir;
    launch.args.push_back("--target-gpu=" + std::to_string(std::max(0, gpu)));
    launch.args.push_back("--max-ctx=" + std::to_string(max_ctx));
    launch.readiness_timeout_ms = readiness_timeout_ms;
    launch.isolate_inherited_fds = true;
    launch.require_private_work_dir = true;
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
        int timeout_ms,
        std::vector<int32_t> & output_ids,
        std::string & error) {
    output_ids.clear();
    error.clear();
#if defined(_WIN32)
    (void)prompt_ids; (void)max_tokens; (void)timeout_ms;
    error = "native_predictor_ipc_unsupported";
    return false;
#else
    if (prompt_ids.empty() || max_tokens <= 0 || timeout_ms <= 0) {
        error = "native_predictor_invalid_request";
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(timeout_ms);
    std::unique_lock<std::timed_mutex> lock(mutex_, std::defer_lock);
    if (!lock.try_lock_until(deadline)) {
        error = "native_predictor_timeout";
        return false;
    }
    FILE * command = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    if (!active_.load(std::memory_order_acquire) ||
        !command || stream_fd < 0) {
        error = "native_predictor_not_active";
        return false;
    }

    std::string prompt_name;
    if (!process_.write_private_file(
            "tool_predictor_prompt", prompt_ids.data(),
            prompt_ids.size() * sizeof(int32_t), prompt_name)) {
        error = "native_predictor_prompt_write_failed";
        return false;
    }

    if (std::chrono::steady_clock::now() >= deadline) {
        (void)process_.remove_private_file(prompt_name);
        error = "native_predictor_timeout";
        return false;
    }

    if (std::fprintf(
            command, "predict %d %s\n", max_tokens, prompt_name.c_str()) < 0 ||
        std::fflush(command) != 0) {
        (void)process_.remove_private_file(prompt_name);
        error = "native_predictor_command_write_failed";
        process_.terminate();
        active_ = false;
        return false;
    }
    const auto response_started = std::chrono::steady_clock::now();
    if (response_started >= deadline) {
        (void)process_.remove_private_file(prompt_name);
        error = "native_predictor_timeout";
        process_.terminate();
        active_ = false;
        return false;
    }
    const int response_timeout_ms = std::max(1, static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - response_started).count()));
    const bool ok = read_qwen3_tool_predictor_response(
        stream_fd, max_tokens, response_timeout_ms, output_ids, error);
    (void)process_.remove_private_file(prompt_name);
    if (!ok) {
        output_ids.clear();
        process_.terminate();
        active_ = false;
        return false;
    }
    return true;
#endif
}

bool Qwen3ToolPredictorIpcClient::active() const {
    return active_.load(std::memory_order_acquire);
}

void Qwen3ToolPredictorIpcClient::close_locked() {
    process_.close();
    active_ = false;
}

void Qwen3ToolPredictorIpcClient::close() {
    std::lock_guard<std::timed_mutex> lock(mutex_);
    close_locked();
}

}  // namespace dflash::common
