#pragma once

#include "common/concurrency/paged_kv_pool.h"
#include "common/concurrency/seq_engine.h"
#include "common/concurrency/seq_slot_manager.h"
#include "common/prompt_lookup.h"
#include "qwen4exp_cache.h"
#include "qwen4exp_mtp.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace luce::common {

struct Qwen4ExpRow;

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
// Prefix checkpoints are prefix-sized copies of a slot cache (qwen4exp
// snapshots, the same payload the single-slot path keeps) on `snapshot_store`,
// the device or system memory, restorable into any slot of the same layout.
// The scheduler owns their identities (1..kPrefixCheckpoints) and LRU policy;
// the engine enforces the backend's byte allowance on capture.
//
// Every slot's cache carries the MTP draft layer and verify rollback state; the first slot's covers the widest verify
// (its rollback buffers are close to 1 GiB), the others' at most kQwen4ExpBatchedVerifyRows rows. Admission takes the
// lowest free slot, so a request arriving at an idle server lands there. It drafts and verifies like the single-slot
// loop while it is the only decoder with nothing prefilling, returning accepted drafts as committed tokens (with an
// adaptive width, prompt-lookup drafts are verified there too). Requests decoding together (or one alone on another
// slot), greedy or sampled, either draft rows - 1 tokens each and verify them all in one multi-slot graph
// (qwen4exp_verify_batched; rows is kQwen4ExpBatchedVerifyRows, or fewer when the caches' draft cap is lower), so the
// experts a step reads serve every slot's drafts, or decode one token each in the batched graph, which still returns
// each slot's final HC residual, so the draft layers stay caught up and drafting resumes. The step takes the mode
// that has committed more tokens per second at that many slots (verify_pays), unless a test forces one
// (set_decode_mode). Every pick is computed where the multi-slot rows compute it, so a request's text does not depend
// on the other requests or the mode. The draft layer's state sits in the cache (K/V, mtp_prev_hidden at
// mtp_prev_pos) at every prompt-chunk boundary, where prefix snapshots capture it.
class Qwen4ExpSeqEngine final : public SeqEngine {
public:
    static constexpr int kPrefixCheckpoints = 64;
    // How decoding slots beside each other step (above): measured, or always verify (where the plan allows it) or
    // always plain. Tests force each mode to check both against the requests' alone streams.
    enum class DecodeMode { measured, verify, plain };

    Qwen4ExpSeqEngine(ggml_backend_t backend, const Qwen4ExpWeights & weights,
                      std::vector<Qwen4ExpCache *> caches, int max_ctx,
                      int prefill_granule = 512, size_t prefix_allowance = 0,
                      int verify_width = 1, AdaptiveSpecWidth * mtp_width = nullptr,
                      SpecWidthCostMemory * mtp_costs = nullptr, ggml_backend_t snapshot_store = nullptr,
                      PromptLookupPolicy * lookup_policy = nullptr);
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
    // A step of the MTP slot alone emits its accepted drafts and the verified token, at most the slot cache's draft
    // cap plus one (a fixed --verify-width, or QWEN4EXP_MTP_MAX_VERIFY when adaptive), so the scheduler keeps
    // drafting while a thinking budget is still open and hands off to single tokens only near its close.
    int max_decode_tokens() const override { return mtp_slot_ >= 0 ? caches_[(size_t) mtp_slot_]->mtp_draft + 1 : 1; }
    void set_decode_mode(DecodeMode mode) { decode_mode_ = mode; }

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
        // Prompt lookup (adaptive width only): the slot's sequence so far, through `lookup_len` tokens of its
        // sample history and then the token it fed last.
        PromptLookup lookup;
        size_t lookup_len = 0;
        long long lookup_rounds = 0, lookup_drafts = 0, lookup_accepted = 0;
    };
    bool mtp_capable(int slot) const;
    bool mtp_prefill(int slot, const int32_t * tokens, int n, int pos0,
                     std::vector<float> & logits);
    bool mtp_eligible(const StepPlan & plan) const;
    StepResult mtp_step(const StepInput & input);
    // Sample a verified row as one-token decode samples it after `fed` (pushed to the slot's history): the device's
    // pick for an argmax sampler, the row's top candidates when they settle the draw, else the whole row. False: EOS.
    bool sample_verify_row(SeqSlot & slot, int32_t fed, const Qwen4ExpRow & row, int32_t & token);
    // The rows per slot a batched verify of this plan could run (every decoding slot drafts, nothing prefills, and
    // the graph serves their rows), else 0.
    int batched_verify_width(const StepPlan & plan) const;
    // That width when the batched verify is this step's mode (verify_pays), else 0.
    int batched_verify_rows(const StepPlan & plan) const;
    StepResult batched_verify_step(const StepPlan & plan, int rows);
    // Whether n decoding slots verify their drafts this step or decode a token each: the mode with more tokens per
    // second so far at n slots, the other one every kExploreSteps steps so its rate stays current.
    bool verify_pays(int n) const;
    void record_mode(int n, bool verify, double tokens, double seconds);

    ggml_backend_t backend_;
    ggml_backend_t store_;   // checkpoint memory: snapshot_store, else the backend
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
    std::vector<MtpState> mtp_;   // per slot
    Qwen4ExpBatchedDecodeWorkspace verify_workspace_;
    std::vector<Qwen4ExpBatchedDecodeWorkspace> draft_workspace_;   // per draft rank: the slots' batched rank graph
    struct BatchedStats {
        long long steps = 0, slots = 0, tokens = 0, plain_steps = 0;
        double step_s = 0.0, draft_s = 0.0, verify_s = 0.0;
    } batched_;
    // Per count of decoding slots: each mode's decayed tokens and seconds, and the steps that chose between them.
    struct ModeRates {
        double verify_tokens = 0.0, verify_s = 0.0, plain_tokens = 0.0, plain_s = 0.0;
        long long verify_steps = 0, plain_steps = 0, steps = 0;
    };
    std::vector<ModeRates> mode_rates_;   // [0, kQwen4ExpMaxBatchedSlots]
    DecodeMode decode_mode_ = DecodeMode::measured;
    // The adaptive verify width and its costs per context range: the single-slot
    // loop's when the backend passes them (else this engine's own).
    AdaptiveSpecWidth own_mtp_width_ = qwen4exp_mtp_width_controller();
    SpecWidthCostMemory own_mtp_costs_;
    AdaptiveSpecWidth * mtp_width_;
    SpecWidthCostMemory * mtp_costs_;
    // With an adaptive width, a round verifies prompt-lookup drafts instead of the MTP's when this policy expects
    // them to commit more tokens per ms (as the single-slot loop does); null: MTP drafts only.
    PromptLookupPolicy * lookup_policy_ = nullptr;
};

} // namespace luce::common
