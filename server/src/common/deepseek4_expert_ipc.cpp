// deepseek4_expert_ipc.cpp - DeepSeek4 expert IPC client.

#include "deepseek4_expert_ipc.h"
#include "io_utils.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <fstream>

namespace dflash::common {
namespace {

using IpcClock = std::chrono::steady_clock;

static uint64_t ipc_elapsed_us(IpcClock::time_point start, IpcClock::time_point end) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

} // namespace

bool DeepSeek4ExpertIpcClient::start(const std::string & bin,
                                     const std::string & model_path,
                                     int worker_gpu,
                                     const std::string & work_dir) {
#if defined(_WIN32)
    (void)bin; (void)model_path; (void)worker_gpu; (void)work_dir;
    std::fprintf(stderr, "DeepSeek4 expert IPC is only implemented on POSIX hosts\n");
    return false;
#else
    close();
    if (bin.empty() || model_path.empty()) return false;
    BackendIpcLaunchConfig launch;
    launch.bin = bin;
    launch.mode = BackendIpcMode::DeepSeek4Expert;
    launch.payload_path = model_path;
    launch.work_dir = work_dir;
    launch.args.push_back("--draft-gpu=" + std::to_string(std::max(0, worker_gpu)));
    if (const char * budget_mb = std::getenv("DFLASH_DS4_EXPERT_WORKER_BUDGET_MB")) {
        if (budget_mb[0]) launch.args.push_back(std::string("--expert-budget-mb=") + budget_mb);
    }
    if (const char * offset = std::getenv("DFLASH_DS4_EXPERT_WORKER_OFFSET")) {
        if (offset[0]) launch.args.push_back(std::string("--expert-offset=") + offset);
    }
    if (const char * placement = std::getenv("DFLASH_DS4_PLACEMENT_IN")) {
        if (placement[0]) launch.args.push_back(std::string("--placement-in=") + placement);
    }
    if (const char * fixed_slot = std::getenv("DFLASH_MOE_FIXED_SLOT_GRAPHS")) {
        if (fixed_slot[0] && std::strcmp(fixed_slot, "0") != 0) {
            launch.args.push_back("--fixed-slot-graphs");
        }
    }
    if (const char * sudo_env = std::getenv("DFLASH_DS4_EXPERT_IPC_SUDO")) {
        if (sudo_env[0] && std::strcmp(sudo_env, "0") != 0) {
            launch.run_with_sudo = true;
            launch.stream_on_stdout = true;
        }
    }
    if (!process_.start(launch)) {
        std::fprintf(stderr, "deepseek4-expert-ipc backend process start failed\n");
        return false;
    }
    active_ = true;
    std::fprintf(stderr, "[deepseek4-expert-ipc] ready bin=%s gpu=%d work_dir=%s\n",
                 bin.c_str(), worker_gpu, process_.work_dir().c_str());
    return true;
#endif
}

bool DeepSeek4ExpertIpcClient::ping() {
#if defined(_WIN32)
    return false;
#else
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    if (!active_ || !cmd || stream_fd < 0) return false;
    std::fprintf(cmd, "ping\n");
    std::fflush(cmd);
    int32_t status = -1;
    return read_exact_fd(stream_fd, &status, sizeof(status)) && status == 0;
#endif
}

bool DeepSeek4ExpertIpcClient::eval(int layer,
                                    int n_tokens,
                                    int n_embd,
                                    int n_selected,
                                    const float * activations,
                                    const int32_t * selected_ids,
                                    const float * selected_weights,
                                    std::vector<float> & out,
                                    DeepSeek4ExpertIpcTiming * timing) {
    PendingEval pending;
    if (!eval_begin(layer, n_tokens, n_embd, n_selected, activations,
                    selected_ids, selected_weights, pending, timing)) {
        return false;
    }
    return eval_end(pending, out, timing);
}

bool DeepSeek4ExpertIpcClient::eval_begin(int layer,
                                          int n_tokens,
                                          int n_embd,
                                          int n_selected,
                                          const float * activations,
                                          const int32_t * selected_ids,
                                          const float * selected_weights,
                                          PendingEval & pending,
                                          DeepSeek4ExpertIpcTiming * timing) {
#if defined(_WIN32)
    (void)layer; (void)n_tokens; (void)n_embd; (void)n_selected;
    (void)activations; (void)selected_ids; (void)selected_weights; (void)pending; (void)timing;
    return false;
#else
    pending = {};
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    if (!active_ || !cmd || stream_fd < 0 || layer < 0 ||
        n_tokens <= 0 || n_embd <= 0 || n_selected <= 0 ||
        !activations || !selected_ids || !selected_weights) {
        return false;
    }

    DeepSeek4ExpertIpcTiming local_timing;
    DeepSeek4ExpertIpcTiming & t = timing ? *timing : local_timing;
    t = {};
    t.request_bytes = sizeof(DeepSeek4ExpertIpcRequestHeader) +
        sizeof(float) * (size_t)n_tokens * (size_t)n_embd +
        sizeof(int32_t) * (size_t)n_tokens * (size_t)n_selected +
        sizeof(float) * (size_t)n_tokens * (size_t)n_selected;

    const std::string path = process_.next_path("ds4_expert_req");
    const auto write_t0 = IpcClock::now();
    {
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        DeepSeek4ExpertIpcRequestHeader hdr;
        hdr.layer = layer;
        hdr.n_tokens = n_tokens;
        hdr.n_embd = n_embd;
        hdr.n_selected = n_selected;
        hdr.flags = timing ? DS4_EXPERT_IPC_FLAG_GRAPH_TIMING : 0;
        f.write((const char *)&hdr, sizeof(hdr));
        f.write((const char *)activations,
                sizeof(float) * (size_t)n_tokens * (size_t)n_embd);
        f.write((const char *)selected_ids,
                sizeof(int32_t) * (size_t)n_tokens * (size_t)n_selected);
        f.write((const char *)selected_weights,
                sizeof(float) * (size_t)n_tokens * (size_t)n_selected);
        if (!f) {
            std::remove(path.c_str());
            return false;
        }
    }
    t.parent_write_us = ipc_elapsed_us(write_t0, IpcClock::now());

    std::fprintf(cmd, "eval %s\n", path.c_str());
    std::fflush(cmd);
    pending.path = path;
    pending.n_tokens = n_tokens;
    pending.n_embd = n_embd;
    pending.active = true;
    return true;
#endif
}

bool DeepSeek4ExpertIpcClient::eval_end(PendingEval & pending,
                                        std::vector<float> & out,
                                        DeepSeek4ExpertIpcTiming * timing) {
#if defined(_WIN32)
    (void)pending; (void)out; (void)timing;
    return false;
#else
    out.clear();
    const int stream_fd = process_.stream_fd();
    if (!active_ || stream_fd < 0 || !pending.active ||
        pending.n_tokens <= 0 || pending.n_embd <= 0) {
        return false;
    }

    DeepSeek4ExpertIpcResponseHeader resp;
    const auto wait_t0 = IpcClock::now();
    bool header_prefix_ok = read_exact_fd(stream_fd, &resp, sizeof(resp)) &&
                            resp.magic == 0x44533452u &&
                            (resp.version == 2 || resp.version == 3);
    if (timing) timing->parent_wait_us += ipc_elapsed_us(wait_t0, IpcClock::now());
    bool ok = header_prefix_ok;
    if (header_prefix_ok) {
        DeepSeek4ExpertIpcTiming local_timing;
        DeepSeek4ExpertIpcTiming & t = timing ? *timing : local_timing;
        t.worker_request_read_us = resp.worker_request_read_us;
        t.worker_partition_us = resp.worker_partition_us;
        t.worker_resident_eval_us = resp.worker_resident_eval_us;
        t.worker_miss_build_us = resp.worker_miss_build_us;
        t.worker_miss_eval_us = resp.worker_miss_eval_us;
        if (resp.version == 3) {
            DeepSeek4ExpertIpcGraphTiming graph_timing;
            ok = read_exact_fd(stream_fd, &graph_timing, sizeof(graph_timing)) &&
                 graph_timing.magic == 0x44533447u &&
                 graph_timing.version == 2;
            if (ok) {
                t.worker_hot_graph_builds = graph_timing.worker_hot_graph_builds;
                t.worker_hot_graph_hits = graph_timing.worker_hot_graph_hits;
                t.worker_cold_graph_builds = graph_timing.worker_cold_graph_builds;
                t.worker_cold_graph_hits = graph_timing.worker_cold_graph_hits;
                t.worker_hot_graph_build_us = graph_timing.worker_hot_graph_build_us;
                t.worker_hot_input_us = graph_timing.worker_hot_input_us;
                t.worker_hot_compute_us = graph_timing.worker_hot_compute_us;
                t.worker_hot_read_us = graph_timing.worker_hot_read_us;
                t.worker_cold_graph_build_us = graph_timing.worker_cold_graph_build_us;
                t.worker_cold_input_us = graph_timing.worker_cold_input_us;
                t.worker_cold_compute_us = graph_timing.worker_cold_compute_us;
                t.worker_cold_read_us = graph_timing.worker_cold_read_us;
            }
        }
    }
    ok = ok &&
         resp.status == 0 &&
         resp.n_tokens == pending.n_tokens &&
         resp.n_embd == pending.n_embd;
    if (ok) {
        out.assign((size_t)pending.n_tokens * (size_t)pending.n_embd, 0.0f);
        const auto read_t0 = IpcClock::now();
        ok = read_exact_fd(stream_fd, out.data(), out.size() * sizeof(float));
        if (timing) {
            timing->parent_read_us += ipc_elapsed_us(read_t0, IpcClock::now());
            timing->response_bytes = sizeof(resp) + (resp.version == 3 ? sizeof(DeepSeek4ExpertIpcGraphTiming) : 0) +
                out.size() * sizeof(float);
        }
    }
    std::remove(pending.path.c_str());
    pending.active = false;
    if (!ok) {
        out.clear();
    }
    return ok;
#endif
}

void DeepSeek4ExpertIpcClient::close() {
    process_.close();
    active_ = false;
}

} // namespace dflash::common
