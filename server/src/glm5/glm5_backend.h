// Glm5Backend: GLM-5.3-Flash AR decode, one sequence; layers split across --target-devices.
// KDA state cannot rewind, so logits are kept (and snapshotted) instead of re-running a token.

#pragma once

#include "common/model_backend.h"
#include "common/sampler.h"
#include "placement/placement_config.h"
#include "glm5_model.h"

#include "ggml-backend.h"

#include <random>
#include <string>
#include <vector>

namespace luce::common {

struct Glm5BackendConfig {
    std::string     model_path;
    DevicePlacement device;
    int             stream_fd = -1;
    int             chunk     = 512;
    // Run on the ggml CPU backend (tests).
    bool            cpu_only  = false;
    // Threads of the CPU backend; 0 = std::thread::hardware_concurrency().
    int             cpu_threads = 0;
};

class Glm5Backend final : public ModelBackend {
public:
    explicit Glm5Backend(Glm5BackendConfig cfg);
    ~Glm5Backend() override;

    Glm5Backend(const Glm5Backend &) = delete;
    Glm5Backend & operator=(const Glm5Backend &) = delete;

    bool init();

    // One step at kv_start..; logits for every token if all_logits, else the last.
    bool forward(const int32_t * tokens, int n_tokens, int kv_start, bool all_logits,
                 std::vector<float> & logits);
    // Zero all state for a new sequence.
    void reset_sequence();
    const Glm5Weights & weights() const { return w_; }
    int cur_pos() const { return cache_.cur_pos; }

    // ModelBackend
    void print_ready_banner() const override;
    int prefill_chunk_size() const override { return cfg_.chunk; }
    int context_limit() const override { return cache_.max_ctx; }

    bool park(ParkTarget target) override;
    bool unpark(ParkTarget target) override;
    bool is_target_parked() const override { return parked_; }

    GenerateResult generate_impl(const GenerateRequest & req, const DaemonIO & io) override;
    GenerateResult restore_and_generate_impl(int slot, const GenerateRequest & req,
                                             const DaemonIO & io) override;

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int  snapshot_cur_pos(int slot) const override;
    size_t snapshot_bytes_estimate(int tokens) const override;

    bool handle_compress(const std::string & line, const DaemonIO & io) override;
    void free_drafter() override {}

    void shutdown() override;

private:
    struct Snapshot {
        bool used = false;
        int  cur_pos = 0;
        std::vector<float> logits;              // predicts the token at cur_pos
        std::vector<std::vector<uint8_t>> data; // per snapshot_tensors() entry
    };
    static constexpr int kSnapshotSlots = kMaxSlots;

    Glm5BackendConfig            cfg_;
    std::vector<ggml_backend_t>  backends_;   // one per weight device
    ggml_backend_t               cpu_ = nullptr;
    ggml_backend_sched_t         sched_ = nullptr;
    std::vector<int>             layer_device_;
    Glm5Weights                  w_;
    Glm5Cache                    cache_;
    bool                         parked_ = false;

    std::vector<float>           last_logits_;  // predicts the token at cache_.cur_pos
    Snapshot                     snapshots_[kSnapshotSlots];

    SamplerCfg                   sampler_;
    std::mt19937_64              rng_{std::random_device{}()};

    bool plan_layers();
    bool load_weights();
    // Prefill prompt[from, to).
    bool prefill(const std::vector<int32_t> & prompt, int from, int to, const DaemonIO & io);
    // Prefill the rest (snapshot where asked), then decode.
    GenerateResult continue_generation(const GenerateRequest & req, const DaemonIO & io,
                                       int restored);
    int32_t pick_token(const std::vector<float> & logits, const std::vector<int32_t> & history);
    bool is_stop_token(int32_t t) const;

    // (tensor, rows) to save for a snapshot at pos.
    std::vector<std::pair<ggml_tensor *, size_t>> snapshot_tensors(int pos) const;
};

}  // namespace luce::common
