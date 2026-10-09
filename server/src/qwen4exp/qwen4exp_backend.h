// Qwen4ExpBackend — ModelBackend for Qwen3.8-Flash-Next (arch `qwen4exp`).
//
// Implements the single-sequence hybrid DeltaNet/full-attention graph,
// hyper-connections, PLE, and routed MoE used by the model.

#pragma once

#include "common/model_backend.h"
#include "placement/placement_config.h"

#include "qwen4exp_internal.h"
#include "qwen4exp_cache.h"
#include "qwen4exp_mtp.h"
#include "qwen4exp_seq_engine.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <string>
#include <memory>
#include <optional>
#include <vector>

namespace luce::common {

struct Qwen4ExpBackendConfig {
    std::string     model_path;
    std::optional<std::string> draft_path; // MTP sidecar; absent = auto-discover
    int             verify_width = 0;     // 0 = adaptive, 1 = off, 2..8 = fixed
    DevicePlacement device;
    // Split mode: routed experts on this device; dense work, KV and the hot
    // experts stay on `device`.
    std::optional<DevicePlacement> expert_device;
    std::string     expert_placement_path;  // routing-stats CSV: its most-routed experts run on `device`
    int             chunk     = 0;  // auto: measured allocation budget
    int             max_concurrency = 1;
};

class Qwen4ExpBackend final : public ModelBackend {
public:
    explicit Qwen4ExpBackend(Qwen4ExpBackendConfig cfg);
    ~Qwen4ExpBackend() override;

    Qwen4ExpBackend(const Qwen4ExpBackend &) = delete;
    Qwen4ExpBackend & operator=(const Qwen4ExpBackend &) = delete;

    bool init();

    // ModelBackend interface
    void print_ready_banner() const override;
    int prefill_chunk_size() const override { return chunk_; }

    bool park(ParkTarget target) override;
    bool unpark(ParkTarget target) override;
    bool is_target_parked() const override { return parked_; }

    GenerateResult generate_impl(const GenerateRequest & req,
                                 const DaemonIO & io) override;

    bool snapshot_save(int slot) override;
    bool snapshot_save_deferred(int slot) override;
    void snapshot_flush_deferred() override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int  snapshot_cur_pos(int slot) const override;
    size_t snapshot_bytes_estimate(int tokens) const override;
    // Prompt forwards end at every restore point (see run()); 4096-token
    // spacing keeps them long enough for the expert reads to pay off.
    int restore_point_spacing() const override { return 4096; }
    bool prefill_cuts_at_restore_points() const override { return true; }
    size_t snapshot_allowance_bytes() const override { return snapshot_budget_; }

    GenerateResult restore_and_generate_impl(int slot,
                                             const GenerateRequest & req,
                                             const DaemonIO & io) override;

    bool handle_compress(const std::string & line,
                         const DaemonIO & io) override;
    void free_drafter() override;

    void shutdown() override;
    SeqEngine * seq_engine() override { return seq_engine_.get(); }

private:
    friend struct Qwen4ExpPrefixTest;
    // Test-only proposal replacement and retained-logit check; empty in serving.
    std::function<void(bool, std::vector<int32_t> &, const std::vector<float> &)> decode_check_;
    GenerateResult run(const GenerateRequest & req, const DaemonIO & io, int restored, int restore_slot = -1);
    bool snapshot_save_replacing(int slot, int source);
    bool snapshot_fits(int slot, int replaced = -1) const;
    bool create_main_cache();
    bool start_seq_engine(); // no-op at --max-concurrency 1
    // Weights, caches, hot experts, the prefill chunk and the prefix allowance, then the slot engine: init and
    // unpark share it, so an unpark sizes itself for the memory free at that time. release_target() undoes it.
    bool load_target();
    void release_target();
    ggml_backend_t snap_backend_ = nullptr; // prefix snapshot storage: the target, or system memory (see init)
    size_t snapshot_budget_ = SIZE_MAX; // with snapshots on the target, auto chunk reserves and enforces this allowance
    std::array<Qwen4ExpSnapshot, kMaxSlots> snapshots_;
    int live_slot_ = -1;
    std::vector<int32_t> tokens_;
    std::vector<float> logits_;
    Qwen4ExpBackendConfig cfg_;
    ggml_backend_t        backend_ = nullptr;
    Qwen4ExpWeights       weights_;
    Qwen4ExpCache         cache_;
    Qwen4ExpSlotStates    slot_states_;   // --max-concurrency > 1: every slot's recurrent state
    int                   chunk_   = 0;
    std::vector<Qwen4ExpCache> seq_caches_;
    std::unique_ptr<Qwen4ExpSeqEngine> seq_engine_;
    bool                  parked_  = false;
    // The adaptive MTP verify width, kept across requests with its costs per context range.
    AdaptiveSpecWidth     mtp_width_ = qwen4exp_mtp_width_controller();
    SpecWidthCostMemory   mtp_costs_;
};

}  // namespace luce::common
