// Generic target-shard IPC client for mixed-backend layer split.

#pragma once

#include "backend_ipc.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace dflash::common {

struct TargetShardIpcLaunchConfig {
    BackendIpcMode mode = BackendIpcMode::Invalid;
    std::string bin;
    std::string target_path;
    std::vector<int> gpus;
    std::vector<int> layer_begins;
    std::vector<int> layer_ends;
    int max_ctx = 0;
    int hidden = 0;
    int vocab = 0;
    int max_tokens = 0;
    std::string work_dir;
    int kvflash_pool_tokens = 0;
    int fa_window = 0;
    std::vector<std::string> model_args;
};

struct TargetShardForwardRequest {
    int base_pos = 0;
    int n_tokens = 0;
    const std::vector<float> * boundary_activation = nullptr;
    const std::vector<int32_t> * token_ids = nullptr;
    int ubatch = 0;
    bool want_argmax = false;
    bool want_logits = false;
};

struct TargetShardForwardResponse {
    int last_tok = -1;
    std::vector<int32_t> * argmax_out = nullptr;
    std::vector<float> * logits_out = nullptr;
};

class TargetShardIpcSession {
public:
    TargetShardIpcSession() = default;
    TargetShardIpcSession(const TargetShardIpcSession &) = delete;
    TargetShardIpcSession & operator=(const TargetShardIpcSession &) = delete;
    ~TargetShardIpcSession() { close(); }

    bool start(const TargetShardIpcLaunchConfig & cfg);

    bool forward(const TargetShardForwardRequest & req,
                 TargetShardForwardResponse & resp);

    bool reset_request_state();
    bool kvflash_sync_identity(int committed);
    bool snapshot_save(int slot);
    void snapshot_free(int slot);
    bool snapshot_restore(int slot);

    bool active() const { return active_; }
    FILE * command_stream() const { return process_.command_stream(); }
    int payload_fd() const { return process_.payload_fd(); }
    int stream_fd() const { return process_.stream_fd(); }
    BackendIpcPayloadTransport resolved_payload_transport() const {
        return process_.resolved_payload_transport();
    }
    size_t shared_payload_capacity() const {
        return process_.shared_payload_capacity();
    }
    const std::string & work_dir() const { return process_.work_dir(); }
    bool write_shared_payload(const void * data, size_t bytes, uint64_t & seq) {
        return process_.write_shared_payload(data, bytes, seq);
    }
    void close();

private:
    BackendIpcProcess process_;
    BackendIpcMode mode_ = BackendIpcMode::Invalid;
    bool active_ = false;
    int hidden_ = 0;
    int vocab_ = 0;
};

using TargetShardIpcClient = TargetShardIpcSession;

bool copy_activation_to_host(const ggml_tensor * act,
                             ggml_backend_t src_backend,
                             int token_offset,
                             int n_tokens,
                             int hidden,
                             std::vector<float> & out);

}  // namespace dflash::common
