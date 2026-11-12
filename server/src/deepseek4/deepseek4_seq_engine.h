#pragma once

#include "common/concurrency/seq_engine.h"
#include "common/concurrency/paged_kv_offload.h"
#include "common/concurrency/seq_slot_manager.h"
#include "deepseek4_internal.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace luce::common {

class DeepSeek4Backend;

// A batched image request's prompt prefix, prefilled into its slot's staging
// cache over several steps before it is copied into the paged slot.
struct DeepSeek4StagedPrefill {
    ImagePromptHandle images;   // empty for a text prompt
    uint64_t request_id = 0;    // identifies the request across slot reuse
    std::vector<int32_t> prompt;
    int prefix = 0;
    DeepSeek4Cache * staging = nullptr;
    int done = 0;        // leading prompt rows already in `staging`
    int restored = 0;    // of those, the rows a prefix checkpoint restored
    std::vector<int> cuts;  // rows a chunk must end at (restore points), ascending
    std::string error;   // set once the request cannot finish
    bool finished() const { return !error.empty() || done >= prefix; }
    // Where the next chunk must end.
    int stop() const {
        const auto next = std::upper_bound(cuts.begin(), cuts.end(), done);
        return next != cuts.end() ? *next : prefix;
    }
};

// Exact concurrent serving path for DeepSeek4. Model state remains in
// DeepSeek4PagedCache; this class owns only scheduler-facing slot state and
// the host mirror of the model's block table.
class DeepSeek4SeqEngine final : public SeqEngine {
public:
    DeepSeek4SeqEngine(DeepSeek4Backend & backend, PagedKvPool & pool,
                       int max_ctx, uint32_t table_stride);
    ~DeepSeek4SeqEngine() override;

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
    // Prefix checkpoints (text prompts staged under --ds4-prefill sparse).
    bool supports_prefix_store() const override;
    size_t estimate_prefix_store_bytes(int tokens) const override;
    AdmitResult admit_with_prefix(uint64_t request_id,
                                  const std::vector<int32_t> & prompt,
                                  const SamplerCfg & sampler,
                                  const PrefixStorePlan & plan) override;
    void discard_prefix_store(PrefixStoreRef checkpoint) override;
    size_t estimate_prefix_state_bytes(int tokens) const override;
    bool drop_prefix_kv(PrefixStoreRef checkpoint) override;

private:
    AdmitResult admit_slot(uint64_t request_id, const std::vector<int32_t> & prompt,
                           const SamplerCfg & sampler);
    AdmitResult admit_text(uint64_t request_id, const std::vector<int32_t> & prompt,
                           const SamplerCfg & sampler, const std::vector<int> & restore_points);
    // Stage a text prompt through the layer-major sparse path (only with
    // --ds4-prefill sparse), cut at `restore_points` and from `from` when
    // given; false leaves it on the gathered prefill.
    bool stage_text_prompt(uint64_t request_id, int slot,
                           const std::vector<int32_t> & prompt,
                           const std::vector<int> & restore_points,
                           const DeepSeek4Snapshot * from = nullptr,
                           const DeepSeek4Snapshot * rows_from = nullptr);
    bool set_block(int slot, int logical, int32_t physical);
    void fail_prefill(int slot, std::vector<PrefillOutput> & outputs,
                      const std::string & error);

    // Requests still being staged (images, or text under --ds4-prefill
    // sparse): their slots hold the prompt minus
    // its last token as seeded blocks. Every step advances them by one shared
    // pass (so decode keeps running between passes) and copies each finished
    // one into its paged slot; until then the slot's last-token prefill waits.
    struct PendingImage {
        int slot = -1;
        DeepSeek4StagedPrefill staged;
        PrefixCaptureTicket capture;  // saved once `staged` reaches its cut
    };
    std::vector<PendingImage> pending_images_;
    // The shared pass in flight: it runs a few layers per step and cannot be
    // abandoned half-way, so members that retire meanwhile are just skipped
    // when it completes, and their slots' staging caches stay busy until then.
    struct StagedPass {
        struct Member {
            int slot = -1;
            ImagePromptHandle images;  // keeps the pass's image rows alive
            uint64_t request_id = 0;   // identifies the request (text too)
            int rows = 0;
        };
        DeepSeek4PrefillPass pass;
        std::vector<Member> members;
        int slices = 0;
        std::chrono::steady_clock::time_point started;
    };
    std::unique_ptr<StagedPass> staged_pass_;
    bool staging_in_flight(int slot) const;
    PendingImage * pending_image(int slot);
    void advance_pending_images(bool decoding, bool idle);

    // Checkpoint id N is checkpoints_[N - 1]: a host snapshot of a staging
    // cache at a prompt cut, with the tokens it holds.
    struct Checkpoint {
        DeepSeek4Snapshot snap;
        std::vector<int32_t> tokens;
    };
    Checkpoint * find_checkpoint(PrefixStoreRef ref);
    bool arm_capture(int slot, PrefixCaptureTicket ticket);
    void save_capture(PendingImage & pending);
    void report_captures(std::vector<PrefillOutput> & outputs);

    DeepSeek4Backend & b_;
    SeqSlotManager slots_;
    PagedKvOffload offload_;
    uint32_t stride_ = 0;
    std::vector<int32_t> host_tables_;
    std::vector<int> reserve_growth_;
    // Slots whose KV holds image rows: token history cannot rebuild them.
    std::vector<uint8_t> slot_has_images_;
    std::vector<Checkpoint> checkpoints_;
    // Per slot: the outcome of its capture, reported with the slot's next
    // planned prefill output (a step may only answer planned slots).
    std::vector<PrefixStoreEvent> capture_events_;
};

} // namespace luce::common
