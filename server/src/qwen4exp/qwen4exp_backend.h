// Qwen4ExpBackend — ModelBackend for Qwen3.8-Flash-Next (arch `qwen4exp`).
//
// Implements the single-sequence hybrid DeltaNet/full-attention graph,
// hyper-connections, PLE, and routed MoE used by the model.

#pragma once

#include "common/model_backend.h"
#include "common/prompt_lookup.h"
#include "pflash/pflash_drafter.h"
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

struct Qwen4ExpChunkPlan;

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
    // PFlash: the drafter scores beside the model on GPU pflash_gpu (-1: no
    // drafter), which keeps pflash_bytes plus pflash_bytes_per_token per
    // context position free on that device.
    int             pflash_gpu = -1;
    size_t          pflash_bytes = 0;
    size_t          pflash_bytes_per_token = 0;
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
    int context_limit() const override { return cache_.max_ctx; }

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
    size_t snapshot_state_bytes_estimate(int tokens) const override;
    bool snapshot_drop_kv(int slot) override;
    bool snapshot_state_only(int slot) const override;
    // The disk session cache: snapshots in their disk form (export_qwen4exp_snapshot).
    bool snapshot_exports() const override { return true; }
    OwnedSnapshot snapshot_copy(int slot) const override;
    OwnedSnapshot snapshot_detach(int slot) override;
    bool snapshot_adopt(int slot, ggml_context * ctx, ggml_backend_buffer_t buf, int cur_pos,
                        int32_t last_tok = -1) override;
    OwnedSnapshot snapshot_take_displaced(int slot) override;
    void snapshot_drop_displaced(int slot) override;
    // Prompt forwards end at every restore point (see run()); 4096-token
    // spacing keeps them long enough for the expert reads to pay off.
    int restore_point_spacing() const override { return 4096; }
    bool prefill_cuts_at_restore_points() const override { return true; }
    size_t snapshot_allowance_bytes() const override { return snapshot_budget_; }

    GenerateResult restore_and_generate_impl(int slot,
                                             const GenerateRequest & req,
                                             const DaemonIO & io) override;

    // PFlash: the drafter scores the prompt beside the target, or in the
    // memory a park frees (pflash_window.h).
    CompressResult compress(const CompressRequest & req) override;
    std::vector<CompressResult> compress_batch(
        const std::vector<CompressRequest> & requests) override;
    bool handle_compress(const std::string & line,
                         const DaemonIO & io) override;
    void free_drafter() override;
    bool pflash_gives_candidate_scores() const override { return true; }

    void shutdown() override;
    SeqEngine * seq_engine() override { return seq_engine_.get(); }

private:
    friend struct Qwen4ExpPrefixTest;
    // Test-only proposal replacement and retained-logit check; empty in serving.
    std::function<void(bool, std::vector<int32_t> &, const std::vector<float> &)> decode_check_;
    GenerateResult run(const GenerateRequest & req, const DaemonIO & io, int restored, int restore_slot = -1);
    bool snapshot_save_replacing(int slot, int source);
    bool snapshot_fits(int slot, int replaced = -1) const;
    // Bytes of a snapshot of `tokens` positions, with or without its per-position rows.
    size_t snapshot_bytes(int tokens, bool rows) const;
    bool create_main_cache();
    void size_kvflash_pool();
    int kvflash_pool_for(int ctx) const;
    int plan_chunk(Qwen4ExpChunkPlan * plan = nullptr);
    int remote_layers_for(const Qwen4ExpChunkPlan & plan, size_t missing) const;
    bool start_seq_engine(); // no-op at --max-concurrency 1
    // ctx < 0: the cache's context (none before the cache exists).
    size_t pflash_reserve(ggml_backend_t device, int ctx = -1) const;
    // Weights, caches, hot experts, the prefill chunk and the prefix allowance, then the slot engine: init and
    // unpark share it, so an unpark sizes itself for the memory free at that time. release_target() undoes it.
    bool load_target();
    bool load_hot_experts();
    void release_target();
    ggml_backend_t snap_backend_ = nullptr; // prefix snapshot storage: the target, or system memory (see init)
    int remote_layers_ = 0;   // split mode: full-attention layers whose cache rows and attention run on the expert device
    int kvflash_pool_ = 0;    // --kvflash: the full-attention rows each slot's cache pools (0: one per position)
    size_t snapshot_budget_ = SIZE_MAX; // with snapshots on the target, auto chunk reserves and enforces this allowance
    std::array<Qwen4ExpSnapshot, kMaxSlots> snapshots_;
    // The concurrency engine's checkpoints are snapshots_; the ones its captures replaced wait here for the scheduler.
    std::array<Qwen4ExpSnapshot, kMaxSlots> displaced_;
    // The snapshots' disk layout (qwen4exp_disk_layout), taken when the cache is made: they outlive a released cache.
    std::string disk_layout_;
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
    DrafterContext        drafter_ctx_;          // PFlash drafter, loaded at the first compress
    bool                  drafter_loaded_ = false;
    // The adaptive MTP verify width, kept across requests with its costs per context range.
    AdaptiveSpecWidth     mtp_width_ = qwen4exp_mtp_width_controller();
    SpecWidthCostMemory   mtp_costs_;
    // Prompt lookup beside the adaptive MTP: the index is rebuilt per request, the policy's costs and acceptance
    // carry across requests like the width controller's.
    PromptLookup          lookup_;
    PromptLookupPolicy    lookup_policy_{QWEN4EXP_MTP_MAX_VERIFY};
};

}  // namespace luce::common
