// Laguna target layer-split adapter.

#pragma once

#include "common/layer_split_backend.h"
#include "common/layer_split_kvflash.h"
#include "common/layer_split_utils.h"
#include "common/kvflash_pager.h"
#include "common/kvflash_scorer.h"
#include "common/target_shard_ipc.h"
#include "laguna_internal.h"
#include "placement/placement_config.h"
#include "placement/remote_target_shard_config.h"
#include "qwen3/qwen3_drafter.h"

#include "ggml-backend.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace dflash::common {

struct LagunaLayerSplitAdapterConfig {
    const char * target_path = nullptr;
    DevicePlacement device;
    RemoteTargetShardConfig remote_target_shard;
    int chunk = 2048;
};

struct LagunaLayerSplitShard : LayerSplitShardMeta {
    LagunaTargetWeights weights;
    LagunaTargetCache cache;
    LagunaLayerStepGraph layer_graph;
};

struct LagunaLayerSplitSnapshot {
    int cur_pos = 0;
    int32_t last_tok = -1;
    std::vector<LagunaCacheSnapshot> shards;
    std::vector<float> prefill_last_logits;
};

class LagunaLayerSplitAdapter : public LayerSplitAdapter {
public:
    explicit LagunaLayerSplitAdapter(const LagunaLayerSplitAdapterConfig & cfg);
    ~LagunaLayerSplitAdapter() override;

    LagunaLayerSplitAdapter(const LagunaLayerSplitAdapter &) = delete;
    LagunaLayerSplitAdapter & operator=(const LagunaLayerSplitAdapter &) = delete;

    const char * name() const override { return "laguna"; }
    bool init() override;
    int max_context() const override { return cfg_.device.max_ctx; }

    void begin_request(const GenerateRequest & req) override;
    void reset_request_state() override;
    bool prefill(const std::vector<int32_t> & prompt,
                 int base_pos, int & last_tok) override;
    bool decode_ar(int last_tok, int committed, int n_gen,
                   const std::vector<int32_t> & history_prefix,
                   std::vector<int32_t> & out_tokens,
                   const DaemonIO & io) override;
    bool supports_cpu_sampling() const override { return true; }
    bool supports_kvflash() const override { return kvflash_active(); }
    bool supports_mixed_backend_layer_split() const override {
        return use_mixed_target_split();
    }

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int snapshot_cur_pos(int slot) const override;
    bool snapshot_restore(int slot) override;
    ModelBackend::SnapshotRef snapshot_ref(int slot) const override;
    bool snapshot_adopt(int slot, ggml_context * ctx,
                        ggml_backend_buffer_t buf, int cur_pos,
                        int32_t last_tok) override;
    int current_last_token() const override;

    void free_drafter() override {}
    void shutdown() override;

private:
    bool run_forward(const std::vector<int32_t> & tokens,
                     int base_pos,
                     int & last_tok,
                     std::vector<float> * logits_out = nullptr);
    bool init_mixed_target_split();
    bool run_mixed_forward(const std::vector<int32_t> & tokens,
                           int base_pos,
                           int & last_tok,
                           std::vector<float> * logits_out);
    bool rebuild_disk_snapshot(int slot);
    KvFlashConfig kvflash_config() const;
    void kvflash_read_config();
    bool kvflash_attach();
    bool kvflash_active() const { return kvflash_tokens_ > 0; }
    bool kvflash_sync_identity(int committed);
    void kvflash_sync_history(const std::vector<int32_t> & tokens, int base_pos);
    void kvflash_maybe_reselect(int generated);
    bool use_mixed_target_split() const {
        return remote_target_shard_.active() && !shards_.empty();
    }

    LagunaLayerSplitAdapterConfig cfg_;
    std::vector<LagunaLayerSplitShard> shards_;
    TargetShardIpcSession remote_target_shard_;
    std::vector<ggml_backend_t> snapshot_backends_;
    std::vector<LagunaLayerSplitSnapshot> snapshots_;
    std::vector<std::vector<ggml_tensor *>> snapshot_prefill_logit_tensors_;
    std::vector<ggml_context *> disk_snapshot_contexts_;
    std::vector<ggml_backend_buffer_t> disk_snapshot_buffers_;
    std::vector<ggml_backend_t> disk_snapshot_backends_;
    ggml_type activation_type_ = GGML_TYPE_F32;
    KvFlashPager kvflash_pager_;
    std::unique_ptr<KvFlashScorer> kvflash_scorer_;
    DrafterContext kvflash_drafter_;
    std::vector<int32_t> kvflash_history_;
    std::vector<std::vector<int32_t>> kvflash_history_snapshots_;
    std::vector<float> kvflash_scores_;
    std::string kvflash_drafter_path_;
    int kvflash_tokens_ = 0;
    int kvflash_tau_ = 64;
    bool kvflash_drafter_loaded_ = false;
    bool kvflash_drafter_failed_ = false;
    static constexpr int PREFIX_SLOTS = ModelBackend::kMaxSlots;
    SamplerCfg sampler_;
    std::mt19937_64 sampler_rng_{std::random_device{}()};
    std::vector<float> prefill_last_logits_;
};

void free_laguna_layer_split_shards(std::vector<LagunaLayerSplitShard> & shards);

int run_laguna_target_shard_ipc_daemon(const char * target_path,
                                       const std::vector<int> & gpus,
                                       const std::vector<int> & layer_begins,
                                       const std::vector<int> & layer_ends,
                                       int max_ctx,
                                       int stream_fd,
                                       int payload_fd = -1,
                                       int shared_payload_fd = -1,
                                       size_t shared_payload_bytes = 0,
                                       int kvflash_pool_tokens = 0);

}  // namespace dflash::common
