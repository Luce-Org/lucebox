// deepseek4_expert_ipc.cpp - DeepSeek4 expert IPC client.

#include "deepseek4_expert_ipc.h"
#include "io_utils.h"

#include <algorithm>
#include <cstdio>

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

void DeepSeek4ExpertIpcClient::close() {
    process_.close();
    active_ = false;
}

} // namespace dflash::common
