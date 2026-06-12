// deepseek4_expert_ipc.h - DeepSeek4 routed expert IPC boundary.
//
// This mode is intended for CUDA-primary + HIP-expert split execution. The
// parent keeps the main DeepSeek4 graph on CUDA and sends normalized FFN
// activations plus selected expert IDs/weights to a backend-specific worker.

#pragma once

#include "backend_ipc.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct DeepSeek4ExpertIpcRequestHeader {
    uint32_t magic = 0x44533445u; // "DS4E"
    uint32_t version = 1;
    int32_t layer = 0;
    int32_t n_tokens = 0;
    int32_t n_embd = 0;
    int32_t n_selected = 0;
    uint32_t flags = 0;
};

struct DeepSeek4ExpertIpcResponseHeader {
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

struct DeepSeek4ExpertIpcTiming {
    uint64_t parent_write_us = 0;
    uint64_t parent_wait_us = 0;
    uint64_t parent_read_us = 0;
    uint64_t worker_request_read_us = 0;
    uint64_t worker_partition_us = 0;
    uint64_t worker_resident_eval_us = 0;
    uint64_t worker_miss_build_us = 0;
    uint64_t worker_miss_eval_us = 0;
    uint64_t request_bytes = 0;
    uint64_t response_bytes = 0;
};

class DeepSeek4ExpertIpcClient {
public:
    DeepSeek4ExpertIpcClient() = default;
    DeepSeek4ExpertIpcClient(const DeepSeek4ExpertIpcClient &) = delete;
    DeepSeek4ExpertIpcClient & operator=(const DeepSeek4ExpertIpcClient &) = delete;
    ~DeepSeek4ExpertIpcClient() { close(); }

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
              DeepSeek4ExpertIpcTiming * timing = nullptr);
    bool active() const { return active_; }
    void close();

private:
    BackendIpcProcess process_;
    bool active_ = false;
};

int run_deepseek4_expert_ipc_daemon(const char * model_path,
                                    int worker_gpu,
                                    int stream_fd,
                                    int payload_fd = -1);

} // namespace dflash::common
