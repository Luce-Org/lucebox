// pflash_drafter_ipc.cpp - PFlash drafter IPC client + daemon body.

#include "pflash_drafter_ipc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace dflash::common {

bool PFlashDrafterIpcClient::start(
        const std::string & bin,
        const std::string & drafter_path,
        int drafter_gpu,
        const std::string & work_dir) {
#if defined(_WIN32)
    (void)bin; (void)drafter_path; (void)drafter_gpu; (void)work_dir;
    std::fprintf(stderr, "PFlash drafter IPC is only implemented on POSIX hosts\n");
    return false;
#else
    close();
    if (bin.empty() || drafter_path.empty()) return false;
    BackendIpcLaunchConfig launch;
    launch.bin = bin;
    launch.mode = BackendIpcMode::PFlashCompress;
    launch.payload_path = drafter_path;
    launch.work_dir = work_dir;
    launch.args.push_back("--draft-gpu=" + std::to_string(std::max(0, drafter_gpu)));
    if (!process_.start(launch)) {
        std::fprintf(stderr, "pflash-ipc backend process start failed\n");
        return false;
    }
    active_ = true;
    std::fprintf(stderr, "[pflash-ipc] ready bin=%s gpu=%d work_dir=%s\n",
                 bin.c_str(), drafter_gpu, process_.work_dir().c_str());
    return true;
#endif
}

bool PFlashDrafterIpcClient::compress(
        const std::vector<int32_t> & input_ids,
        float keep_ratio,
        std::vector<int32_t> & compressed_ids) {
#if defined(_WIN32)
    (void)input_ids; (void)keep_ratio; (void)compressed_ids;
    return false;
#else
    compressed_ids.clear();
    FILE * cmd = process_.command_stream();
    const int stream_fd = process_.stream_fd();
    if (!active_ || !cmd || stream_fd < 0 || input_ids.empty()) return false;

    const std::string path = process_.next_path("pflash_tokens");
    if (!write_int32_file(path, input_ids)) {
        std::fprintf(stderr, "pflash-ipc write tokens failed: %s\n", path.c_str());
        return false;
    }
    int keep_x1000 = (int)std::lround(std::max(0.0f, keep_ratio) * 1000.0f);
    keep_x1000 = std::max(0, std::min(1000, keep_x1000));

    std::fprintf(cmd, "compress %d %s\n", keep_x1000, path.c_str());
    std::fflush(cmd);

    int32_t status = -1;
    bool ok = read_exact_fd(stream_fd, &status, sizeof(status)) && status == 0;
    if (ok) {
        int32_t n_out = -1;
        ok = read_exact_fd(stream_fd, &n_out, sizeof(n_out)) && n_out > 0;
        if (ok) {
            compressed_ids.assign((size_t)n_out, 0);
            ok = read_exact_fd(stream_fd, compressed_ids.data(),
                               compressed_ids.size() * sizeof(int32_t));
        }
    }
    std::remove(path.c_str());
    if (!ok) {
        std::fprintf(stderr, "pflash-ipc compress failed status=%d\n", status);
        compressed_ids.clear();
        close();
    }
    return ok;
#endif
}

void PFlashDrafterIpcClient::close() {
    process_.close();
    active_ = false;
}

} // namespace dflash::common
