// dflash_draft_ipc.h - DFlash draft IPC mode.
//
// The draft IPC mechanism spawns a child process running the draft model on
// a separate GPU. Communication is via stdin commands, a parent->daemon
// payload pipe, and a stream pipe for binary status/data. The daemon still
// accepts the older temporary-file commands for compatibility.
//
// The IPC client class and the remote feature-copy helper are target-agnostic
// and live in this common header. The daemon entry point is declared here too.

#pragma once

#include "backend_ipc.h"
#include "dflash27b.h"
#include "io_utils.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace dflash::common {

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
               const std::string & work_dir,
               BackendIpcMode mode = BackendIpcMode::DFlashDraft);

    bool send_feature_slice(int capture_idx,
                            int start_pos,
                            int n_tokens,
                            const std::vector<float> & slice);

    // Token-major [n_tokens, n_target_layers, hidden_size] upload. This is the
    // preferred DSpark path because all capture layers cross IPC in one
    // command and one acknowledgement.
    bool send_feature_block(int start_pos,
                            int n_tokens,
                            const float * features,
                            size_t feature_count);
    bool send_feature_block(int start_pos,
                            int n_tokens,
                            const std::vector<float> & features) {
        return send_feature_block(
            start_pos, n_tokens, features.data(), features.size());
    }

    bool propose(int committed,
                 int ctx_len,
                 const std::vector<float> & noise_embed,
                 std::vector<float> & hidden_out);

    bool get_feature_range(int start_pos, int n_tokens, std::vector<float> & out);
    bool set_feature_range(int start_pos, int n_tokens,
                           const std::vector<float> & data);

    bool active() const { return active_; }
    int ring_cap() const { return ring_cap_; }
    int hidden_size() const { return hidden_size_; }
    int block_size() const { return block_size_; }
    int n_target_layers() const { return n_target_layers_; }
    void close();

private:
    BackendIpcProcess process_;
    bool active_ = false;
    int ring_cap_ = 0;
    int hidden_size_ = DFLASH27B_TARGET_HIDDEN;
    int block_size_ = DFLASH27B_DRAFT_BLOCK_SIZE;
    int n_target_layers_ = DFLASH27B_DRAFT_N_TARGET_LAYERS;
    BackendIpcMode mode_ = BackendIpcMode::Invalid;
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
                                int stream_fd,
                                int payload_fd = -1,
                                int shared_payload_fd = -1,
                                size_t shared_payload_bytes = 0);

} // namespace dflash::common
