// Compatibility shim for PR489's DeepSeek4 graph worker hook.
//
// The online PR489 path never creates an ExpertIpcClient; the hook is kept in
// the graph signature only so this patch can stay source-compatible with local
// experimental expert IPC trees without requiring those files in a clean PR489
// checkout.
#pragma once

#include <cstdint>
#include <vector>

namespace dflash::common {

struct ExpertIpcTiming {
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

class ExpertIpcClient {
public:
    struct PendingEval {};

    bool active() const { return false; }

    bool eval_begin(int,
                    int,
                    int,
                    int,
                    const float *,
                    const int32_t *,
                    const float *,
                    PendingEval &,
                    ExpertIpcTiming * = nullptr) {
        return false;
    }

    bool eval_end(PendingEval &,
                  std::vector<float> &,
                  ExpertIpcTiming * = nullptr) {
        return false;
    }

    bool eval(int,
              int,
              int,
              int,
              const float *,
              const int32_t *,
              const float *,
              std::vector<float> &,
              ExpertIpcTiming * = nullptr) {
        return false;
    }
};

}  // namespace dflash::common
