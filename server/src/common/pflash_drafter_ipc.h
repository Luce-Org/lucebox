// pflash_drafter_ipc.h - PFlash drafter IPC client + daemon entry.
//
// Used when target and PFlash drafter run on different compiled backends
// (for example CUDA target + HIP drafter). The parent sends drafter-tokenized
// prompt IDs to the daemon and receives compressed drafter token IDs back.

#pragma once

#include "backend_ipc.h"
#include "io_utils.h"
#include "pflash_types.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace dflash::common {

struct PFlashDrafterIpcCompressCommand {
    bool legacy_quantized_ratio = false;
    float keep_ratio = 0.0f;
    int score_query_end = -1;
    int score_query_tokens = 8;
    std::vector<PFlashTokenSpan> required_instruction_spans;
    std::string path;
};

bool format_pflash_drafter_ipc_compress_command(
    float keep_ratio,
    int score_query_end,
    int score_query_tokens,
    const std::string & path,
    std::string & out,
    std::string & error);

bool format_pflash_drafter_ipc_compress_command(
    float keep_ratio,
    int score_query_end,
    int score_query_tokens,
    const std::vector<PFlashTokenSpan> & required_instruction_spans,
    const std::string & path,
    std::string & out,
    std::string & error);

bool parse_pflash_drafter_ipc_compress_command(
    const std::string & line,
    PFlashDrafterIpcCompressCommand & out,
    std::string & error);

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
                  std::vector<int32_t> & compressed_ids,
                  int score_query_end = -1,
                  int score_query_tokens = 8,
                  const std::vector<PFlashTokenSpan> &
                      required_instruction_spans = {});

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
