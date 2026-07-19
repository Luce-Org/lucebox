// DeepSeek4 layer-split adapter for CUDA + AMD Halo split.
//
// Splits the DS4 model's 43 layers across two GPUs by contiguous layer ranges.
// The split point is auto-computed from CUDA free memory, or manually set via
// DFLASH_DS4_CUDA_LAYERS env var.

#pragma once

#include "common/layer_split_backend.h"
#include "common/layer_split_utils.h"
#include "common/target_shard_ipc.h"
#include "common/sampler.h"
#include "deepseek4_internal.h"
#include "placement/placement_config.h"
#include "placement/remote_target_shard_config.h"

#include "ggml-backend.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace dflash::common {

struct DeepSeek4LayerSplitAdapterConfig {
    const char * target_path = nullptr;
    DevicePlacement device;
    RemoteTargetShardConfig remote_target_shard;
    int chunk = 512;
};

struct DeepSeek4LayerSplitShard : LayerSplitShardMeta {
    DeepSeek4Weights weights;
    DeepSeek4Cache cache;
};

class DeepSeek4LayerSplitAdapter : public LayerSplitAdapter {
public:
    explicit DeepSeek4LayerSplitAdapter(const DeepSeek4LayerSplitAdapterConfig & cfg);
    ~DeepSeek4LayerSplitAdapter() override;

    DeepSeek4LayerSplitAdapter(const DeepSeek4LayerSplitAdapter &) = delete;
    DeepSeek4LayerSplitAdapter & operator=(const DeepSeek4LayerSplitAdapter &) = delete;

    const char * name() const override { return "deepseek4"; }
    bool init() override;
    int max_context() const override { return cfg_.device.max_ctx; }

    void begin_request(const GenerateRequest & req) override;
    void reset_request_state() override;
    int prefill_chunk_tokens() const override { return cfg_.chunk; }
    bool prefill(const std::vector<int32_t> & prompt,
                 int base_pos, int & last_tok) override;
    bool decode_ar(int last_tok, int committed, int n_gen,
                   const std::vector<int32_t> & history_prefix,
                   std::vector<int32_t> & out_tokens,
                   const DaemonIO & io) override;
    bool supports_cpu_sampling() const override { return true; }
    bool supports_mixed_backend_layer_split() const override {
        return use_mixed_target_split();
    }

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int snapshot_cur_pos(int slot) const override;
    bool snapshot_restore(int slot) override;
    int current_last_token() const override { return last_tok_; }

    void free_drafter() override {}
    void shutdown() override;

private:
    bool run_forward(const std::vector<int32_t> & tokens, int base_pos,
                     int & last_tok, std::vector<float> * logits_out = nullptr);
    bool init_mixed_target_split();
    bool init_mixed_target_split_full(const DevicePlacement & device);
    bool run_mixed_forward(const std::vector<int32_t> & tokens, int base_pos,
                           int & last_tok, std::vector<float> * logits_out);
    bool use_mixed_target_split() const {
        return remote_target_shard_.active() && !shards_.empty();
    }
    int compute_auto_split_layers() const;
    static int estimate_cuda_layers_from_free_bytes(size_t free_bytes);
    static size_t hc_state_elements(const DeepSeek4Weights & weights);

    DeepSeek4LayerSplitAdapterConfig cfg_;
    std::vector<DeepSeek4LayerSplitShard> shards_;
    TargetShardIpcSession remote_target_shard_;
    std::vector<ggml_backend_t> snapshot_backends_;

    // HC state persists across all layers (shared between shards)
    std::vector<float> hc_state_;
    int cur_pos_ = 0;
    int32_t last_tok_ = -1;

    static constexpr int PREFIX_SLOTS = ModelBackend::kMaxSlots;
    struct Snapshot {
        int cur_pos = 0;
        int32_t last_tok = -1;
        std::vector<float> hc_state;
        std::vector<float> prefill_last_logits;
        std::vector<DeepSeek4Snapshot> shards;
        bool used = false;
    };
    std::vector<Snapshot> snapshots_;

    SamplerCfg sampler_;
    std::mt19937_64 sampler_rng_{std::random_device{}()};
    std::vector<float> prefill_last_logits_;
};

// IPC daemon for remote DS4 target shard (runs on Halo/HIP GPU)
int run_deepseek4_target_shard_ipc_daemon(
    const char * target_path,
    const std::vector<int> & gpus,
    const std::vector<int> & layer_begins,
    const std::vector<int> & layer_ends,
    int max_ctx,
    int stream_fd,
    int payload_fd = -1);

}  // namespace dflash::common
