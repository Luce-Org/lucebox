// Qwen35 target-shard IPC mode for mixed-backend layer split.

#pragma once

#include "common/target_shard_ipc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct Qwen35TargetCaptureSlice {
    int capture_idx = -1;
    int start_pos = 0;
    int n_tokens = 0;
    std::vector<float> data;
};

struct Qwen35TargetShardSnapshotTensor {
    int shard = -1;
    std::string name;
    uint32_t type = 0;
    int64_t ne[4] = {1, 1, 1, 1};
    std::vector<uint8_t> data;
};

struct Qwen35TargetShardSnapshotData {
    int shard_count = 0;
    int cur_pos = 0;
    int32_t last_tok = -1;
    std::vector<Qwen35TargetShardSnapshotTensor> tensors;
    std::vector<float> logits;
};

class Qwen35TargetShardIpcClient {
public:
    Qwen35TargetShardIpcClient() = default;
    Qwen35TargetShardIpcClient(const Qwen35TargetShardIpcClient &) = delete;
    Qwen35TargetShardIpcClient & operator=(const Qwen35TargetShardIpcClient &) = delete;
    ~Qwen35TargetShardIpcClient() { close(); }

    bool start(const std::string & bin,
               const std::string & target_path,
               const std::vector<int> & gpus,
               const std::vector<int> & layer_begins,
               const std::vector<int> & layer_ends,
               int max_ctx,
               int max_verify_tokens,
               int kq_stride_pad,
               int fa_window,
               int hidden,
               int vocab,
               int max_tokens,
               const std::string & work_dir,
               bool enable_dflash,
               int kvflash_pool_tokens = 0);

    bool forward(int base_pos,
                 int n_tokens,
                 const std::vector<float> & boundary_activation,
                 bool need_logits,
                 int & last_tok,
                 std::vector<int32_t> * argmax_out,
                 std::vector<float> * logits_out,
                 std::vector<Qwen35TargetCaptureSlice> * captures_out = nullptr,
                 int ubatch = 0);

    bool project_hidden_to_tokens(const float * hidden,
                                  int n_tokens,
                                  std::vector<int32_t> & tokens_out);

    bool snapshot_kv();
    bool restore_kv();
    bool reset_request_state();
    bool kvflash_sync_identity(int committed);
    bool snapshot_save(int slot);
    void snapshot_free(int slot);
    bool snapshot_restore(int slot);
    bool snapshot_export(int slot, Qwen35TargetShardSnapshotData & out);
    bool snapshot_import(int slot, const Qwen35TargetShardSnapshotData & data);

    bool active() const { return session_.active(); }
    void close();

private:
    TargetShardIpcSession session_;
    int hidden_ = 0;
    int vocab_ = 0;
};

int run_qwen35_target_shard_ipc_daemon(const char * target_path,
                                       const std::vector<int> & gpus,
                                       const std::vector<int> & layer_begins,
                                       const std::vector<int> & layer_ends,
                                       int max_ctx,
                                       int max_verify_tokens,
                                       int kq_stride_pad,
                                       int fa_window,
                                       int stream_fd,
                                       int payload_fd = -1,
                                       int shared_payload_fd = -1,
                                       size_t shared_payload_bytes = 0,
                                       bool enable_dflash = false,
                                       int kvflash_pool_tokens = 0);

}  // namespace dflash::common
