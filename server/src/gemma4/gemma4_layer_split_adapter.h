// Gemma4 layer-split adapter.

#pragma once

#include "common/layer_split_backend.h"
#include "common/layer_split_utils.h"
#include "gemma4_internal.h"
#include "placement/placement_config.h"

#include "ggml-backend.h"

#include <vector>

namespace dflash::common {

struct Gemma4LayerSplitAdapterConfig {
    const char * target_path = nullptr;
    DevicePlacement device;
    int chunk = 512;
    int fa_window = 0;
};

struct Gemma4LayerSplitShard : LayerSplitShardMeta {
    Gemma4Weights weights;
    Gemma4Cache cache;
    Gemma4LayerStepGraph layer_graph;
};

struct Gemma4LayerSplitSnapshot {
    int cur_pos = 0;
    int32_t last_tok = -1;
    std::vector<Gemma4Snapshot> shards;
};

class Gemma4LayerSplitAdapter : public LayerSplitAdapter {
public:
    explicit Gemma4LayerSplitAdapter(const Gemma4LayerSplitAdapterConfig & cfg);
    ~Gemma4LayerSplitAdapter() override;

    Gemma4LayerSplitAdapter(const Gemma4LayerSplitAdapter &) = delete;
    Gemma4LayerSplitAdapter & operator=(const Gemma4LayerSplitAdapter &) = delete;

    const char * name() const override { return "gemma4"; }
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

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int snapshot_cur_pos(int slot) const override;
    bool snapshot_restore(int slot) override;
    int current_last_token() const override;

    void free_drafter() override {}
    void shutdown() override;

private:
    bool run_forward(const std::vector<int32_t> & tokens,
                     int base_pos,
                     int & last_tok,
                     std::vector<float> * logits_out = nullptr);

    Gemma4LayerSplitAdapterConfig cfg_;
    std::vector<Gemma4LayerSplitShard> shards_;
    std::vector<ggml_backend_t> snapshot_backends_;
    std::vector<Gemma4LayerSplitSnapshot> snapshots_;
    static constexpr int PREFIX_SLOTS = ModelBackend::kMaxSlots;
    SamplerCfg sampler_;
    std::mt19937_64 sampler_rng_{std::random_device{}()};
    std::vector<float> prefill_last_logits_;
};

void free_gemma4_layer_split_shards(std::vector<Gemma4LayerSplitShard> & shards);

}  // namespace dflash::common
