// expert_ipc.h - Routed expert IPC boundary.
//
// This mode is intended for CUDA-primary + HIP-expert split execution. The
// parent keeps the main MoE graph on CUDA and sends normalized FFN
// activations plus selected expert IDs/weights to a backend-specific worker.

#pragma once

#include "backend_ipc.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

constexpr uint32_t EXPERT_IPC_FLAG_GRAPH_TIMING = 1u << 0;

struct ExpertIpcRequestHeader {
    uint32_t magic = 0x44533445u; // "DS4E"
    uint32_t version = 1;
    int32_t layer = 0;
    int32_t n_tokens = 0;
    int32_t n_embd = 0;
    int32_t n_selected = 0;
    uint32_t flags = 0;
};

struct ExpertIpcResponseHeader {
    uint32_t magic = 0x44533452u; // "DS4R"
    uint32_t version = 2;
    int32_t status = 0;
    int32_t n_tokens = 0;
    int32_t n_embd = 0;
    uint64_t worker_request_read_us = 0;
    uint64_t worker_partition_us = 0;
    uint64_t worker_resident_eval_us = 0;
    uint64_t worker_miss_build_us = 0;
    uint64_t worker_miss_eval_us = 0;
};

struct ExpertIpcGraphTiming {
    uint32_t magic = 0x44533447u; // "DS4G"
    uint32_t version = 2;
    uint64_t worker_hot_graph_builds = 0;
    uint64_t worker_hot_graph_hits = 0;
    uint64_t worker_cold_graph_builds = 0;
    uint64_t worker_cold_graph_hits = 0;
    uint64_t worker_hot_graph_build_us = 0;
    uint64_t worker_hot_input_us = 0;
    uint64_t worker_hot_compute_us = 0;
    uint64_t worker_hot_read_us = 0;
    uint64_t worker_cold_graph_build_us = 0;
    uint64_t worker_cold_input_us = 0;
    uint64_t worker_cold_compute_us = 0;
    uint64_t worker_cold_read_us = 0;
};

struct ExpertIpcTiming {
    uint64_t parent_write_us = 0;
    uint64_t parent_wait_us = 0;
    uint64_t parent_read_us = 0;
    uint64_t worker_request_read_us = 0;
    uint64_t worker_partition_us = 0;
    uint64_t worker_resident_eval_us = 0;
    uint64_t worker_miss_build_us = 0;
    uint64_t worker_miss_eval_us = 0;
    uint64_t worker_hot_graph_builds = 0;
    uint64_t worker_hot_graph_hits = 0;
    uint64_t worker_cold_graph_builds = 0;
    uint64_t worker_cold_graph_hits = 0;
    uint64_t worker_hot_graph_build_us = 0;
    uint64_t worker_hot_input_us = 0;
    uint64_t worker_hot_compute_us = 0;
    uint64_t worker_hot_read_us = 0;
    uint64_t worker_cold_graph_build_us = 0;
    uint64_t worker_cold_input_us = 0;
    uint64_t worker_cold_compute_us = 0;
    uint64_t worker_cold_read_us = 0;
    uint64_t request_bytes = 0;
    uint64_t response_bytes = 0;
};

class ExpertIpcClient {
public:
    struct PendingEval {
        std::string path;
        int n_tokens = 0;
        int n_embd = 0;
        bool active = false;
    };

    ExpertIpcClient() = default;
    ExpertIpcClient(const ExpertIpcClient &) = delete;
    ExpertIpcClient & operator=(const ExpertIpcClient &) = delete;
    ~ExpertIpcClient() { close(); }

    bool start(const std::string & bin,
               const std::string & model_path,
               int worker_gpu,
               const std::string & work_dir);

    bool ping();
    bool eval(int layer,
              int n_tokens,
              int n_embd,
              int n_selected,
              const float * activations,
              const int32_t * selected_ids,
              const float * selected_weights,
              std::vector<float> & out,
              ExpertIpcTiming * timing = nullptr);
    bool eval_begin(int layer,
                    int n_tokens,
                    int n_embd,
                    int n_selected,
                    const float * activations,
                    const int32_t * selected_ids,
                    const float * selected_weights,
                    PendingEval & pending,
                    ExpertIpcTiming * timing = nullptr);
    bool eval_end(PendingEval & pending,
                  std::vector<float> & out,
                  ExpertIpcTiming * timing = nullptr);
    bool active() const { return active_; }
    void close();

private:
    BackendIpcProcess process_;
    bool active_ = false;
};

int run_expert_ipc_daemon(const char * model_path,
                                    int worker_gpu,
                                    int stream_fd,
                                    int payload_fd = -1);

} // namespace dflash::common
