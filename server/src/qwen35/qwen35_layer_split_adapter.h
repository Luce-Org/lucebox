// Qwen35 layer-split adapter.

#pragma once

#include "common/dflash_draft_ipc.h"
#include "common/kvflash_pager.h"
#include "common/kvflash_scorer.h"
#include "common/layer_split_backend.h"
#include "common/layer_split_kvflash.h"
#include "dflash_feature_ring.h"
#include "layer_split_types.h"
#include "placement/placement_config.h"
#include "placement/remote_draft_config.h"
#include "placement/remote_target_shard_config.h"
#include "qwen3/qwen3_drafter.h"
#include "qwen35_target_shard_ipc.h"
#include "step_graph.h"
#include "internal.h"

#include "ggml-backend.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace dflash::common {

struct Qwen35LayerSplitAdapterConfig {
    const char * target_path = nullptr;
    const char * draft_path  = nullptr;
    DevicePlacement device;
    int draft_gpu = 0;
    RemoteDraftConfig remote_draft;
    RemoteTargetShardConfig remote_target_shard;

    int fa_window = 0;  // 0 = full attention. qwen3.6 full-attn layers must see the whole context; a finite window drops the system prompt/tools -> breaks tool calls.
    int kq_stride_pad = 32;
    int draft_ctx_max = 4096;
    int chunk = 512;
    int max_verify_tokens = DFLASH27B_DRAFT_BLOCK_SIZE;
    bool run_dflash = false;
    int draft_swa_window = 0;
};

class Qwen35LayerSplitAdapter : public LayerSplitAdapter {
public:
    explicit Qwen35LayerSplitAdapter(const Qwen35LayerSplitAdapterConfig & cfg);
    ~Qwen35LayerSplitAdapter() override;

    Qwen35LayerSplitAdapter(const Qwen35LayerSplitAdapter &) = delete;
    Qwen35LayerSplitAdapter & operator=(const Qwen35LayerSplitAdapter &) = delete;

    const char * name() const override { return "qwen35"; }
    bool init() override;
    int max_context() const override { return cfg_.device.max_ctx; }

    void begin_request(const GenerateRequest & req) override;
    void reset_request_state() override;
    int prefill_chunk_tokens() const override;
    bool prefill(const std::vector<int32_t> & prompt,
                 int base_pos, int & last_tok) override;
    bool decode_ar(int last_tok, int committed, int n_gen,
                   const std::vector<int32_t> & history_prefix,
                   std::vector<int32_t> & out_tokens,
                   const DaemonIO & io) override;
    bool supports_cpu_sampling() const override { return true; }

    bool can_dflash_decode() const override;
    bool decode_dflash(const std::vector<int32_t> & prompt, int base_pos,
                       int last_tok, int n_gen, std::vector<int32_t> & out_tokens,
                       const DaemonIO & io, float & accept_rate_out) override;

    ModelBackend::CompressResult
    compress(const ModelBackend::CompressRequest & req) override;
    const char * default_compress_drafter_path() const override;
    void free_drafter() override;

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

    bool supports_dflash_spec_decode() const override { return true; }
    DFlashTarget * dflash_target() override;
    bool supports_remote_draft() const override { return true; }
    bool supports_kvflash() const override { return kvflash_active(); }
    bool supports_mixed_backend_layer_split() const override {
        return use_mixed_target_split();
    }

    void shutdown() override;

private:
    bool load_draft();
    bool init_mixed_target_split();
    void kvflash_read_config();
    bool kvflash_attach();
    bool kvflash_active() const { return kvflash_tokens_ > 0; }
    bool kvflash_sync_identity(int committed);
    void kvflash_sync_history(const std::vector<int32_t> & tokens, int base_pos);
    void kvflash_maybe_reselect(int generated);
    bool use_mixed_target_split() const {
        return remote_target_shard_.active() && !shards_.empty();
    }
    bool snapshot_slot_valid(int slot) const;
    bool rebuild_disk_snapshot(int slot);
    bool snapshot_draft_features(int slot);
    void free_draft_feature_snapshot(int slot);
    bool restore_draft_features(int slot);

    Qwen35LayerSplitAdapterConfig cfg_;
    std::vector<Qwen35LayerSplitShard> shards_;
    ggml_backend_t draft_backend_ = nullptr;
    bool draft_backend_owned_ = false;
    DraftWeights draft_weights_;
    DraftFeatureMirror feature_ring_;
    DFlashDraftIpcClient remote_draft_;
    Qwen35TargetShardIpcClient remote_target_shard_;
    StepGraph draft_sg_;
    StepGraph proj_sg_;
    ggml_type activation_type_ = GGML_TYPE_F32;
    DrafterContext pflash_drafter_;
    bool pflash_drafter_loaded_ = false;
    KvFlashPager kvflash_pager_;
    std::unique_ptr<KvFlashScorer> kvflash_scorer_;
    DrafterContext kvflash_drafter_;
    std::vector<int32_t> kvflash_history_;
    std::vector<float> kvflash_scores_;
    std::string kvflash_drafter_path_;
    int kvflash_tokens_ = 0;
    int kvflash_tau_ = 64;
    bool kvflash_drafter_loaded_ = false;
    bool kvflash_drafter_failed_ = false;
    static constexpr int PREFIX_SLOTS = ModelBackend::kMaxSlots;
    std::vector<std::vector<PrefixSnapshot>> prefix_snapshots_;
    std::vector<std::vector<float>> snapshot_prefill_logits_;
    std::vector<std::vector<ggml_tensor *>> snapshot_prefill_logit_tensors_;
    std::vector<std::vector<int32_t>> kvflash_history_snapshots_;
    std::vector<ggml_context *> disk_snapshot_contexts_;
    std::vector<ggml_backend_buffer_t> disk_snapshot_buffers_;
    std::vector<ggml_backend_t> disk_snapshot_backends_;
    std::vector<ggml_backend_t> snapshot_backends_;
    struct DraftFeatureSnapshot {
        int cur_pos = 0;
        int start_pos = 0;
        int n_tokens = 0;
        int cap = 0;
        int n_target_layers = 0;
        int hidden_size = 0;
        std::vector<float> data;
    };
    std::vector<DraftFeatureSnapshot> draft_feature_snapshots_;

    SamplerCfg sampler_;
    std::mt19937_64 sampler_rng_{std::random_device{}()};
    std::unique_ptr<DFlashTarget> dflash_target_;
    std::vector<float> prefill_last_logits_;
};

}  // namespace dflash::common
