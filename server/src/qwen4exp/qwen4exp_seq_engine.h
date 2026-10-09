#pragma once

#include "common/concurrency/paged_kv_pool.h"
#include "common/concurrency/seq_engine.h"
#include "common/concurrency/seq_slot_manager.h"
#include "qwen4exp_cache.h"
#include "qwen4exp_mtp.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace luce::common {

// N-way independent-slot engine. Each slot owns a full cache; the pool is
// admission/headroom bookkeeping only (there is no paged KV).
//
// Prefill advances in fixed granules: every prompt forward ends one granule
// after the previous one, or earlier at a restore point (the plan's chat
// boundaries and cache cuts, as the single-slot path cuts), a capture boundary
// or the prompt's end, whatever else the step carries. So a prompt's numerics
// depend neither on concurrent load nor on what the prefix cache holds. While
// other slots decode, a step runs one granule in total, so a long prompt never
// holds the live streams for more than one.
//
// Prefix checkpoints are prefix-sized device copies of a slot cache
// (qwen4exp snapshots, the same payload the single-slot path keeps),
// restorable into any slot of the same layout. The scheduler owns their
// identities (1..kPrefixCheckpoints) and LRU policy; the engine enforces the
// backend's byte allowance on capture.
//
// The first slot's cache carries the MTP draft layer and verify rollback state
// (one set: its rollback buffers are close to 1 GiB). Admission takes the lowest
// free slot, so a request arriving at an idle server lands there. It drafts and
// verifies like the single-slot loop while it is the only decoder with nothing
// prefilling, returning accepted drafts as committed tokens; beside other work
// it decodes one token per step in the batched graph, which still returns its
// final HC residual, so its draft layer stays caught up and drafting resumes
// once it is alone again. Concurrent serving loads no hot experts, so its
// verify rows compute every pick where the multi-slot rows do and a request's
// text does not depend on the other requests. The
// draft layer's state sits in the cache (K/V, mtp_prev_hidden at mtp_prev_pos)
// at every prompt-chunk boundary, where prefix snapshots capture it.
class Qwen4ExpSeqEngine final : public SeqEngine {
public:
    static constexpr int kPrefixCheckpoints = 64;

    Qwen4ExpSeqEngine(ggml_backend_t backend, const Qwen4ExpWeights & weights,
                      std::vector<Qwen4ExpCache *> caches, int max_ctx,
                      int prefill_granule = 512, size_t prefix_allowance = 0,
                      int verify_width = 1, AdaptiveSpecWidth * mtp_width = nullptr,
                      SpecWidthCostMemory * mtp_costs = nullptr);
    ~Qwen4ExpSeqEngine() override;

    int slot_count() const override { return slots_.slot_count(); }
    int max_context() const override { return slots_.max_context(); }
    AdmitResult admit(uint64_t request_id, const std::vector<int32_t> & prompt,
                      const SamplerCfg & sampler) override;
    bool supports_prefix_store() const override { return prefix_allowance_ > 0; }
    size_t estimate_prefix_store_bytes(int tokens) const override;
    AdmitResult admit_with_prefix(uint64_t request_id, const std::vector<int32_t> & prompt,
                                  const SamplerCfg & sampler, const PrefixStorePlan & plan) override;
    void discard_prefix_store(PrefixStoreRef checkpoint) override;
    StepPlanLimits step_plan_limits(int decode_rows) const override;
    bool reserve_decode(const StepPlan & plan) override;
    StepResult step(const StepPlan & plan) override;
    void retire(int slot) override;
    bool token_is_eos(int32_t token) const override;

private:
    AdmitResult admit_cold(uint64_t request_id, const std::vector<int32_t> & prompt,
                           const SamplerCfg & sampler);
    bool restore_prefix(int slot, const std::vector<int32_t> & prompt, PrefixStoreRef checkpoint);
    PrefixStoreEvent capture_prefix(int slot, PrefixCaptureTicket ticket);
    int checkpoint_index(PrefixStoreRef checkpoint) const;
    // Rows the slot's next prompt forward covers: up to one granule, the next
    // restore point, a pending capture boundary, the prompt's end and `max_tokens`.
    int prefill_segment(int slot, int max_tokens) const;
    // Pairs (trunk hidden h_p, token x_{p+1}) of the MTP slot not yet through the
    // draft layer, from position `pos`. Between steps `h` has one more row than
    // `tok`: the scheduler's pending token completes the last pair.
    struct MtpState {
        bool live = false;
        Qwen4ExpMtpPending pending;   // excludes the token the scheduler feeds next
        int context = 0;   // the prompt length: the width controller's context range
        long long drafts = 0, accepted = 0, steps = 0, tokens = 0;
        double decode_s = 0.0, draft_s = 0.0, verify_s = 0.0;
        uint64_t verify_builds = 0;
    };
    bool mtp_prefill(Qwen4ExpCache & cache, const int32_t * tokens, int n, int pos0,
                     std::vector<float> & logits);
    bool mtp_eligible(const StepPlan & plan) const;
    StepResult mtp_step(const StepInput & input);

    ggml_backend_t backend_;
    const Qwen4ExpWeights & weights_;
    std::vector<Qwen4ExpCache *> caches_;
    PagedKvPool pool_;
    SeqSlotManager slots_;
    Qwen4ExpBatchedDecodeWorkspace decode_workspace_;
    int prefill_granule_;
    size_t prefix_allowance_;
    std::vector<Qwen4ExpSnapshot> checkpoints_;
    std::vector<std::vector<int>> prefill_cuts_;   // per slot: restore points, ascending
    int mtp_slot_ = -1;
    int verify_width_ = 1;
    MtpState mtp_;
    // The adaptive verify width and its costs per context range: the single-slot
    // loop's when the backend passes them (else this engine's own).
    AdaptiveSpecWidth own_mtp_width_ = qwen4exp_mtp_width_controller();
    SpecWidthCostMemory own_mtp_costs_;
    AdaptiveSpecWidth * mtp_width_;
    SpecWidthCostMemory * mtp_costs_;
};

} // namespace luce::common
