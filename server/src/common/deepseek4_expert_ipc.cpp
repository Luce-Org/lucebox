// deepseek4_expert_ipc.cpp - DeepSeek4 expert IPC client.

#include "deepseek4_expert_ipc.h"
#include "io_utils.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <fstream>

namespace dflash::common {

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
                                    std::vector<float> & out) {
#if defined(_WIN32)
    (void)layer; (void)n_tokens; (void)n_embd; (void)n_selected;
    (void)activations; (void)selected_ids; (void)selected_weights; (void)out;
    return false;
#else
    out.clear();
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    if (!active_ || !cmd || stream_fd < 0 || layer < 0 ||
        n_tokens <= 0 || n_embd <= 0 || n_selected <= 0 ||
        !activations || !selected_ids || !selected_weights) {
        return false;
    }

    const std::string path = process_.next_path("ds4_expert_req");
    {
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        DeepSeek4ExpertIpcRequestHeader hdr;
        hdr.layer = layer;
        hdr.n_tokens = n_tokens;
        hdr.n_embd = n_embd;
        hdr.n_selected = n_selected;
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

    std::fprintf(cmd, "eval %s\n", path.c_str());
    std::fflush(cmd);

    DeepSeek4ExpertIpcResponseHeader resp;
    bool ok = read_exact_fd(stream_fd, &resp, sizeof(resp)) &&
              resp.magic == 0x44533452u &&
              resp.version == 1 &&
              resp.status == 0 &&
              resp.n_tokens == n_tokens &&
              resp.n_embd == n_embd;
    if (ok) {
        out.assign((size_t)n_tokens * (size_t)n_embd, 0.0f);
        ok = read_exact_fd(stream_fd, out.data(), out.size() * sizeof(float));
    }
    std::remove(path.c_str());
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
