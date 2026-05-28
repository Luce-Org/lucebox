// Qwen35 layer-split adapter.

#pragma once

#include "common/dflash_draft_ipc.h"
#include "common/layer_split_backend.h"
#include "dflash_feature_ring.h"
#include "layer_split_types.h"
#include "placement/placement_config.h"
#include "placement/remote_draft_config.h"
#include "qwen3/qwen3_drafter.h"
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

    int fa_window = 2048;
    int kq_stride_pad = 32;
    int draft_ctx_max = 4096;
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
    bool prefill(const std::vector<int32_t> & prompt,
                 int base_pos, int & last_tok) override;
    bool decode_ar(int last_tok, int committed, int n_gen,
                   std::vector<int32_t> & out_tokens,
                   const DaemonIO & io) override;
    bool supports_cpu_sampling() const override { return true; }

    bool can_dflash_decode() const override;
    bool decode_dflash(const std::vector<int32_t> & prompt, int base_pos,
                       int last_tok, int n_gen, std::vector<int32_t> & out_tokens,
                       const DaemonIO & io) override;

    ModelBackend::CompressResult
    compress(const ModelBackend::CompressRequest & req) override;
    const char * default_compress_drafter_path() const override;
    void free_drafter() override;

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int snapshot_cur_pos(int slot) const override;
    bool snapshot_restore(int slot) override;
    int current_last_token() const override;

    bool supports_dflash_spec_decode() const override { return true; }
    DFlashTarget * dflash_target() override;
    bool supports_remote_draft() const override { return true; }

    void shutdown() override;

private:
    bool load_draft();
    bool snapshot_slot_valid(int slot) const;
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
    StepGraph draft_sg_;
    StepGraph proj_sg_;
    DrafterContext pflash_drafter_;
    bool pflash_drafter_loaded_ = false;
    static constexpr int PREFIX_SLOTS = ModelBackend::kMaxSlots;
    std::vector<std::vector<PrefixSnapshot>> prefix_snapshots_;
    std::vector<std::vector<float>> snapshot_prefill_logits_;
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
