#pragma once

#include "common/concurrency/seq_engine.h"
#include "common/concurrency/paged_kv_offload.h"
#include "common/concurrency/seq_slot_manager.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace luce::common {

class DeepSeek4Backend;
struct DeepSeek4Cache;

// A batched image request's prompt prefix, prefilled into its slot's staging
// cache over several steps before it is copied into the paged slot.
struct DeepSeek4StagedPrefill {
    ImagePromptHandle images;
    std::vector<int32_t> prompt;
    int prefix = 0;
    DeepSeek4Cache * staging = nullptr;
    int done = 0;        // leading prompt rows already in `staging`
    std::string error;   // set once the request cannot finish
    bool finished() const { return !error.empty() || done >= prefix; }
};

// Exact concurrent serving path for DeepSeek4. Model state remains in
// DeepSeek4PagedCache; this class owns only scheduler-facing slot state and
// the host mirror of the model's block table.
class DeepSeek4SeqEngine final : public SeqEngine {
public:
    DeepSeek4SeqEngine(DeepSeek4Backend & backend, PagedKvPool & pool,
                       int max_ctx, uint32_t table_stride);

    int slot_count() const override { return slots_.slot_count(); }
    int max_context() const override { return slots_.max_context(); }
    AdmitResult admit(uint64_t request_id, const std::vector<int32_t> & prompt,
                      const SamplerCfg & sampler) override;
    StepResult step(const StepPlan & plan) override;
    StepPlanLimits step_plan_limits(int decode_rows) const override;
    bool reserve_decode(const StepPlan & plan) override;
    size_t kv_offload_capacity() const override { return offload_.capacity(); }
    KvOffloadState kv_offload_state(int slot) const override { return offload_.state(slot); }
    bool offload_kv(int slot, size_t bytes, std::string & error) override;
    bool restore_kv(int slot, std::string & error) override;
    bool evict_kv(int slot, int32_t pending_token, std::string & error) override;
    bool kv_recomputable(int slot) const override;
    bool kv_restore_feasible(int slot) const override {
        return slots_.kv_restore_feasible(slot);
    }
    void retire(int slot) override;
    bool token_is_eos(int32_t token) const override;
    bool supports_images() const override;
    AdmitResult admit_images(uint64_t request_id,
                             const std::vector<int32_t> & prompt,
                             const SamplerCfg & sampler,
                             const ImagePromptHandle & images) override;

private:
    bool set_block(int slot, int logical, int32_t physical);
    void fail_prefill(int slot, std::vector<PrefillOutput> & outputs,
                      const std::string & error);

    // Image requests still being staged: their slots hold the prompt minus
    // its last token as seeded blocks. Every step advances them by one shared
    // pass (so decode keeps running between passes) and copies each finished
    // one into its paged slot; until then the slot's last-token prefill waits.
    struct PendingImage {
        int slot = -1;
        DeepSeek4StagedPrefill staged;
    };
    std::vector<PendingImage> pending_images_;
    PendingImage * pending_image(int slot);
    void advance_pending_images(bool decoding, bool idle);

    DeepSeek4Backend & b_;
    SeqSlotManager slots_;
    PagedKvOffload offload_;
    uint32_t stride_ = 0;
    std::vector<int32_t> host_tables_;
    std::vector<int> reserve_growth_;
    // Slots whose KV holds image rows: token history cannot rebuild them.
    std::vector<uint8_t> slot_has_images_;
};

} // namespace luce::common
