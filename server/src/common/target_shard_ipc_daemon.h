// Generic target-shard IPC daemon loop for mixed-backend layer split.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace dflash::common {

struct TargetShardDaemonForwardRequest {
    int base_pos = 0;
    int n_tokens = 0;
    int ubatch = 0;
    bool want_argmax = false;
    bool want_logits = false;
    const std::vector<float> * boundary_activation = nullptr;
    const std::vector<int32_t> * token_ids = nullptr;
};

struct TargetShardDaemonForwardResponse {
    int32_t last_tok = -1;
    std::vector<int32_t> argmax;
    std::vector<float> logits;
};

struct TargetShardDaemonCallbacks {
    const char * log_prefix = "target-shard-daemon";

    std::function<bool(const TargetShardDaemonForwardRequest &,
                       TargetShardDaemonForwardResponse &)> forward;
    std::function<bool()> reset_request_state;
    std::function<bool(int)> kvflash_sync_identity;
    std::function<bool(int)> snapshot_save;
    std::function<void(int)> snapshot_free;
    std::function<bool(int)> snapshot_restore;
};

int run_target_shard_ipc_daemon_loop(
    int hidden,
    int vocab,
    int stream_fd,
    int payload_fd,
    int shared_payload_fd,
    size_t shared_payload_bytes,
    TargetShardDaemonCallbacks callbacks);

}  // namespace dflash::common
