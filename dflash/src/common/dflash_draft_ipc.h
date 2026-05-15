// dflash_draft_ipc.h — DFlash draft model IPC client + daemon entry.
//
// The draft IPC mechanism spawns a child process running the draft model on
// a separate GPU. Communication is via stdin commands + a stream pipe for
// binary status/data. Feature slices and noise embeddings are exchanged
// through temporary files.
//
// The IPC client class and the remote feature-copy helper are target-agnostic
// and live in this common header. The daemon entry point is declared here too
// but its implementation lives next to a specific target architecture (which
// owns the draft graph builder used inside the daemon loop).

#pragma once

#include "dflash27b.h"
#include "io_utils.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#if !defined(_WIN32)
#  include <cerrno>
#  include <cstring>
#  include <sys/stat.h>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

namespace dflash27b {

// ── IPC Client (parent process) ─────────────────────────────────────

class DFlashDraftIpcClient {
public:
    DFlashDraftIpcClient() = default;
    // Construct with target dimensions (used for size validation in IPC).
    DFlashDraftIpcClient(int hidden_size, int block_size, int n_target_layers)
        : hidden_size_(hidden_size), block_size_(block_size),
          n_target_layers_(n_target_layers) {}
    DFlashDraftIpcClient(const DFlashDraftIpcClient &) = delete;
    DFlashDraftIpcClient & operator=(const DFlashDraftIpcClient &) = delete;
    ~DFlashDraftIpcClient() { close(); }

    bool start(const std::string & bin,
               const std::string & draft_path,
               int draft_gpu,
               int ring_cap,
               const std::string & work_dir);

    bool send_feature_slice(int capture_idx,
                            int start_pos,
                            int n_tokens,
                            const std::vector<float> & slice);

    bool propose(int committed,
                 int ctx_len,
                 const std::vector<float> & noise_embed,
                 std::vector<float> & hidden_out);

    bool active() const { return active_; }
    int ring_cap() const { return ring_cap_; }
    int hidden_size() const { return hidden_size_; }
    int block_size() const { return block_size_; }
    int n_target_layers() const { return n_target_layers_; }
    void close();

private:
#if !defined(_WIN32)
    bool init_work_dir(const std::string & requested);
    std::string next_path(const char * prefix);

    pid_t pid_ = -1;
    FILE * cmd_ = nullptr;
    int stream_fd_ = -1;
    std::string work_dir_;
    int seq_ = 0;
    bool owns_work_dir_ = false;
#endif
    bool active_ = false;
    int ring_cap_ = 0;
    int hidden_size_ = DFLASH27B_TARGET_HIDDEN;
    int block_size_ = DFLASH27B_DRAFT_BLOCK_SIZE;
    int n_target_layers_ = DFLASH27B_DRAFT_N_TARGET_LAYERS;
};

// ── Remote draft feature copy helper ────────────────────────────────

bool copy_capture_slice_to_remote_draft(
        DFlashDraftIpcClient & remote,
        int capture_idx,
        const ggml_tensor * act_out,
        ggml_backend_t src_backend,
        int chunk_start,
        int start_pos,
        int n_tokens);

// ── Stream status helper ────────────────────────────────────────────

inline bool stream_status(int stream_fd, int32_t status) {
#if defined(_WIN32)
    (void)stream_fd; (void)status;
    return false;
#else
    return write_exact_fd(stream_fd, &status, sizeof(status));
#endif
}

// ── IPC Daemon entry point ──────────────────────────────────────────
//
// Implemented in common/dflash_draft_ipc_daemon.cpp. The DFlash draft model
// is a single universal network shared across all target architectures, so
// the daemon body is target-agnostic.
int run_dflash_draft_ipc_daemon(const char * draft_path,
                                int ring_cap,
                                int draft_gpu,
                                int stream_fd);

} // namespace dflash27b
