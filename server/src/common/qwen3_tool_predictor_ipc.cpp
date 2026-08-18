#include "qwen3_tool_predictor_ipc.h"

#include "io_utils.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>

#if !defined(_WIN32)
#  include <poll.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace dflash::common {
namespace {

#if !defined(_WIN32)
bool write_private_prompt_file(
        const std::string & work_dir,
        const std::vector<int32_t> & prompt_ids,
        std::string & path) {
    std::string pattern = work_dir + "/tool_predictor_prompt_XXXXXX";
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    const int fd = ::mkstemp(buffer.data());
    if (fd < 0) return false;
    path = buffer.data();
    const bool written = ::fchmod(fd, S_IRUSR | S_IWUSR) == 0 &&
        write_exact_fd(
            fd, prompt_ids.data(), prompt_ids.size() * sizeof(int32_t));
    const bool closed = ::close(fd) == 0;
    if (!written || !closed) {
        ::unlink(path.c_str());
        path.clear();
        return false;
    }
    return true;
}

bool read_exact_until(
        int fd,
        void * data,
        size_t bytes,
        const std::chrono::steady_clock::time_point & deadline,
        bool & timed_out) {
    auto * cursor = static_cast<char *>(data);
    size_t received = 0;
    timed_out = false;
    while (received < bytes) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            timed_out = true;
            return false;
        }
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - now).count();
        pollfd descriptor{fd, POLLIN | POLLHUP, 0};
        const int polled = ::poll(
            &descriptor, 1,
            static_cast<int>((std::max)(int64_t{1}, remaining)));
        if (polled == 0) {
            timed_out = true;
            return false;
        }
        if (polled < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (descriptor.revents & (POLLERR | POLLNVAL)) return false;
        const ssize_t count = ::read(
            fd, cursor + received, bytes - received);
        if (count == 0) return false;
        if (count < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        received += static_cast<size_t>(count);
    }
    return true;
}
#endif

}  // namespace

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
    if (!read_exact_until(
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
    if (!read_exact_until(
            stream_fd, &count, sizeof(count), deadline, timed_out) ||
        count <= 0 || count > max_tokens) {
        error = timed_out
            ? "native_predictor_timeout"
            : "native_predictor_invalid_response";
        return false;
    }
    output_ids.assign(static_cast<size_t>(count), 0);
    if (!read_exact_until(
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
        const std::string & work_dir) {
#if defined(_WIN32)
    (void)bin; (void)model_path; (void)gpu; (void)max_ctx; (void)work_dir;
    std::fprintf(stderr,
                 "Qwen3 tool-predictor IPC is only implemented on POSIX hosts\n");
    return false;
#else
    std::lock_guard<std::timed_mutex> lock(mutex_);
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

    std::string path;
    if (!write_private_prompt_file(
            process_.work_dir(), prompt_ids, path)) {
        error = "native_predictor_prompt_write_failed";
        return false;
    }

    if (std::chrono::steady_clock::now() >= deadline) {
        std::remove(path.c_str());
        error = "native_predictor_timeout";
        return false;
    }

    if (std::fprintf(
            command, "predict %d %s\n", max_tokens, path.c_str()) < 0 ||
        std::fflush(command) != 0) {
        std::remove(path.c_str());
        error = "native_predictor_command_write_failed";
        process_.terminate();
        active_ = false;
        return false;
    }
    const auto response_started = std::chrono::steady_clock::now();
    if (response_started >= deadline) {
        std::remove(path.c_str());
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
    std::remove(path.c_str());
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
