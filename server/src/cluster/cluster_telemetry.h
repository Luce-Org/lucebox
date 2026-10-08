// cluster_telemetry.h - per-step timing counters shared by the head decorator,
// the worker loop and the DeepSeek4 cluster graph path.
//
// Header-only and dependency-free so both the model code (Agent C) and the
// decision hooks / worker main (Agent D) can accumulate into the same struct;
// the head folds the per-rank totals into RequestReportMsg and
// usage.timings.cluster.

#pragma once

#include <chrono>
#include <cstdint>

namespace luce::cluster {

struct ClusterStepTelemetry {
    uint64_t compute_us        = 0;  // local graph compute (host wall time)
    uint64_t allreduce_calls   = 0;
    uint64_t allreduce_bytes   = 0;  // payload bytes handed to RCCL
    uint64_t allreduce_wait_us = 0;  // host time blocked on collectives
    uint64_t ctrl_wait_us      = 0;  // host time blocked on the control channel

    void add(const ClusterStepTelemetry & o) {
        compute_us        += o.compute_us;
        allreduce_calls   += o.allreduce_calls;
        allreduce_bytes   += o.allreduce_bytes;
        allreduce_wait_us += o.allreduce_wait_us;
        ctrl_wait_us      += o.ctrl_wait_us;
    }

    void reset() { *this = ClusterStepTelemetry{}; }
};

// RAII: adds the elapsed wall time (microseconds, steady clock) to `acc`
// when the scope ends.
class ScopedUsTimer {
public:
    explicit ScopedUsTimer(uint64_t & acc)
        : acc_(acc), start_(std::chrono::steady_clock::now()) {}
    ScopedUsTimer(const ScopedUsTimer &) = delete;
    ScopedUsTimer & operator=(const ScopedUsTimer &) = delete;
    ~ScopedUsTimer() {
        const auto end = std::chrono::steady_clock::now();
        acc_ += (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
    }

private:
    uint64_t & acc_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace luce::cluster
