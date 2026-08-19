// Persistent Qwen3 tool-predictor IPC lane.
//
// The HTTP server tokenizes the predictor prompt with the predictor's own
// vocabulary, then sends token IDs to a small out-of-process Qwen3 backend.
// Keeping this lane behind BackendIpcProcess isolates the target decoder from
// predictor crashes and lets heterogeneous deployments choose a different GPU.

#pragma once

#include "backend_ipc.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace dflash::common {

class Qwen3ToolPredictorIpcClient {
public:
    Qwen3ToolPredictorIpcClient() = default;
    Qwen3ToolPredictorIpcClient(const Qwen3ToolPredictorIpcClient &) = delete;
    Qwen3ToolPredictorIpcClient & operator=(
        const Qwen3ToolPredictorIpcClient &) = delete;
    ~Qwen3ToolPredictorIpcClient() { close(); }

    bool start(const std::string & bin,
               const std::string & model_path,
               int gpu,
               int max_ctx,
               const std::string & work_dir,
               int readiness_timeout_ms);

    // Requests are serialized: one compact predictor model owns one KV cache.
    // On transport or generation failure the lane closes and fails shut.
    bool predict(const std::vector<int32_t> & prompt_ids,
                 int max_tokens,
                 int timeout_ms,
                 std::vector<int32_t> & output_ids,
                 std::string & error);

    bool active() const;
    // After a timeout/transport failure the lane fails shut. Relaunch the
    // daemon with the original parameters, at most once per cooldown.
    bool try_restart();
    void close();

private:
    void close_locked();
    bool start_locked(const std::string & bin,
                      const std::string & model_path,
                      int gpu,
                      int max_ctx,
                      const std::string & work_dir,
                      int readiness_timeout_ms);

    mutable std::timed_mutex mutex_;
    BackendIpcProcess process_;
    std::atomic<bool> active_{false};
    // Launch parameters retained for try_restart().
    std::string launch_bin_;
    std::string launch_model_path_;
    int launch_gpu_ = 0;
    int launch_max_ctx_ = 0;
    std::string launch_work_dir_;
    int launch_readiness_timeout_ms_ = 0;
    bool launch_params_set_ = false;
    std::chrono::steady_clock::time_point last_restart_attempt_{};
};

// Read one daemon response under a single wall-clock deadline. Kept outside
// the client so the timeout and partial-response behavior can be unit tested
// without loading a model.
bool read_qwen3_tool_predictor_response(
    int stream_fd,
    int max_tokens,
    int timeout_ms,
    std::vector<int32_t> & output_ids,
    std::string & error);

int run_qwen3_tool_predictor_ipc_daemon(const char * model_path,
                                         int gpu,
                                         int max_ctx,
                                         int stream_fd);

}  // namespace dflash::common
