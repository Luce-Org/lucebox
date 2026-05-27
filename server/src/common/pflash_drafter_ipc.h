// pflash_drafter_ipc.h - PFlash drafter IPC client + daemon entry.
//
// Used when target and PFlash drafter run on different compiled backends
// (for example CUDA target + HIP drafter). The parent sends drafter-tokenized
// prompt IDs to the daemon and receives compressed drafter token IDs back.

#pragma once

#include "backend_ipc.h"
#include "io_utils.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace dflash::common {

class PFlashDrafterIpcClient {
public:
    PFlashDrafterIpcClient() = default;
    PFlashDrafterIpcClient(const PFlashDrafterIpcClient &) = delete;
    PFlashDrafterIpcClient & operator=(const PFlashDrafterIpcClient &) = delete;
    ~PFlashDrafterIpcClient() { close(); }

    bool start(const std::string & bin,
               const std::string & drafter_path,
               int drafter_gpu,
               const std::string & work_dir);

    bool compress(const std::vector<int32_t> & input_ids,
                  float keep_ratio,
                  std::vector<int32_t> & compressed_ids);

    bool active() const { return active_; }
    void close();

private:
    BackendIpcProcess process_;
    bool active_ = false;
};

int run_pflash_drafter_ipc_daemon(const char * drafter_path,
                                  int drafter_gpu,
                                  int stream_fd);

} // namespace dflash::common
