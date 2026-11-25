// Inkling2SeqEngine & Inkling2SlotKv — Concurrent N-slot SeqEngine for Inkling 2 Mini.
//
// Supports TWO unified execution modes over SlotKv v2 + Inkling2SlotStateAdapter:
//   1. Batched Multi-Slot 2DSD (`multi_cache.tree.active`):
//      Shares all 128 MoE experts across N concurrent decoding sequences in ONE batched
//      Prelude / Fused Multi-Pass Core / Fused Multi-Depth Coda GPU step via zero-copy
//      per-slot views (with independent per-slot 3-Tier KVFlash 2.0 + Mode-4 recall!).
//   2. 1D / DSpark / Custom-Forward Multi-Slot (`caches` vector):
//      Per-slot 1D AR / solo DSpark + 3-Tier KVFlash 2.0 + Mode-4 recall (and unit-test
//      forward hook for official check_seq_engine_contract verification).

#pragma once

#include "common/concurrency/paged_kv_pool.h"
#include "common/concurrency/seq_engine.h"
#include "common/concurrency/seq_slot_manager.h"
#include "common/concurrency/slot_kv.h"
#include "inkling2_2dsd_executor.h"
#include "inkling2_cache.h"
#include "inkling2_internal.h"
#include "inkling2_slot_state.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace luce::common {

// SlotKv v2 adapter for Inkling 2 Mini multi-slot serving.
// Wraps per-slot bookkeeping over PagedKvPool AND forwards build_table_patch()
// straight to each slot's Inkling2KvFlash (emitting disjoint Tier-0 & Tier-1.5
// block tables with -1 holes when KVFlash is active).
class Inkling2SlotKv final : public SlotKv {
public:
    Inkling2SlotKv(int max_ctx,
                   const std::vector<Inkling2Cache *> & caches,
                   uint32_t block_size = 64);

    void set_caches(std::vector<Inkling2Cache *> caches) { caches_ = std::move(caches); }

    uint32_t   block_size() const override { return pool_.block_size(); }
    uint32_t   attn_block_size() const override { return pool_.block_size(); }
    uint32_t   physical_block_count() const override { return pool_.physical_block_count(); }
    uint32_t   max_sequences() const override { return pool_.max_sequences(); }
    uint32_t   free_block_count() const override { return pool_.free_block_count(); }
    SlotKvCaps caps() const override;

    PagedKvStatus acquire_reserved(PagedKvRequestId id, uint32_t cap,
                                   PagedKvSequenceHandle & h) override;
    PagedKvStatus acquire_reserved(PagedKvRequestId id, uint32_t cap,
                                   const SlotAcquireOpts & opts,
                                   PagedKvSequenceHandle & h) override;
    PagedKvStatus reserve_capacity(PagedKvSequenceHandle h, uint32_t cap) override;
    PagedKvAppendResult append(PagedKvSequenceHandle h, uint32_t n, bool first_last) override;
    PagedKvStatus rollback_append(PagedKvSequenceHandle h, uint32_t n) override;
    PagedKvStatus clear(PagedKvSequenceHandle h) override;
    PagedKvStatus release(PagedKvSequenceHandle h) override;
    PagedKvStatus owned_block_count(PagedKvSequenceHandle h, uint32_t & n) const override;

    bool build_table_patch(PagedKvSequenceHandle h, uint32_t token_extent,
                           uint32_t max_attn_blocks, SlotTablePatch & out,
                           bool remote = false) const override;

private:
    static uint32_t compute_pool_blocks_(int max_ctx, size_t n_slots, uint32_t bs);

    PagedKvPool                  pool_;
    std::vector<Inkling2Cache *> caches_;
};

// Optional step-forward hook allowing deterministic unit tests to drive the full
// SeqEngine contract (check_seq_engine_contract) without loading a 17 GB GGUF.
struct Inkling2SeqStepResult {
    bool               ok = false;
    int32_t            greedy_token = -1;
    std::vector<float> logits; // populated when need_logits == true
    std::string        error;
};

using Inkling2SeqForwardFn = std::function<Inkling2SeqStepResult(
    int slot, const int32_t * tokens, int n_tokens, int pos0, bool need_logits, int capture_q_rows)>;

class Inkling2SeqEngine final : public SeqEngine {
public:
    // Batched multi-slot 2DSD constructor (zero-copy views into `multi_cache`).
    Inkling2SeqEngine(ggml_backend_t target_backend,
                      const Inkling2Weights & weights,
                      Inkling2Cache & multi_cache,
                      int max_concurrency,
                      int prefill_chunk);

    // 1D / DSpark / custom-forward multi-slot constructor.
    Inkling2SeqEngine(ggml_backend_t backend,
                      const Inkling2Weights & weights,
                      std::vector<Inkling2Cache *> caches,
                      int max_ctx,
                      int prefill_chunk = 512,
                      int verify_width = 0,
                      Inkling2SeqForwardFn custom_forward = {});

    ~Inkling2SeqEngine() override;

    Inkling2SeqEngine(const Inkling2SeqEngine &) = delete;
    Inkling2SeqEngine & operator=(const Inkling2SeqEngine &) = delete;

    int slot_count() const override { return slots_->slot_count(); }
    int max_context() const override { return slots_->max_context(); }
    KvPoolTokens kv_pool_tokens() const override { return slots_->kv_pool_tokens(); }

    AdmitResult admit(uint64_t request_id,
                      const std::vector<int32_t> & prompt,
                      const SamplerCfg & sampler) override;
    AdmitResult admit_bounded(uint64_t request_id,
                              const std::vector<int32_t> & prompt,
                              const SamplerCfg & sampler,
                              int max_new_tokens,
                              const AdmitMeta & meta = {}) override;

    bool        supports_prefix_store() const override;
    size_t      estimate_prefix_store_bytes(int tokens) const override;
    AdmitResult admit_with_prefix(uint64_t request_id,
                                  const std::vector<int32_t> & prompt,
                                  const SamplerCfg & sampler,
                                  const PrefixStorePlan & plan) override;
    void        discard_prefix_store(PrefixStoreRef checkpoint) override;

    StepPlanLimits step_plan_limits(int decode_rows) const override;
    bool           reserve_decode(const StepPlan & plan) override;
    StepResult     step(const StepPlan & plan) override;

    size_t         kv_offload_capacity() const override;
    KvOffloadState kv_offload_state(int slot) const override;
    bool           offload_kv(int slot, size_t max_bytes, std::string & error) override;
    bool           restore_kv(int slot, std::string & error) override;
    bool           evict_kv(int slot, int32_t pending_token, std::string & error) override;
    bool           kv_restore_feasible(int slot) const override;

    void retire(int slot) override;
    bool token_is_eos(int32_t token) const override;
    int  max_decode_tokens() const override;

    // Drop per-slot view graph workspaces before the backend frees/rebuilds multi_cache.
    void release_device_state();
    // Rebuild zero-copy per-slot views after multi_cache is reinstated.
    bool reinstate_device_state(int prefill_chunk);

    // Expose SlotKv v2 & SlotStateAdapter for KvFlashSessionRuntime & unit tests.
    Inkling2SlotKv &                 slot_kv() { return *slot_kv_; }
    const Inkling2SlotKv &           slot_kv() const { return *slot_kv_; }
    Inkling2SlotStateAdapter &       state_adapter() { return *state_adapter_; }
    const Inkling2SlotStateAdapter & state_adapter() const { return *state_adapter_; }

private:
    static constexpr int kPrefixCheckpoints = 64;

    struct SlotState;
    struct PrefixCheckpoint;

    bool init_slot_views();
    void reset_slot_session(int slot);
    std::string validate_plan_(const StepPlan & plan) const;
    int  checkpoint_index_(PrefixStoreRef checkpoint) const;
    bool save_global_kv_(int slot, int tokens, std::vector<uint8_t> & out, bool & is_kvflash) const;
    bool restore_global_kv_(int slot, int tokens, const std::vector<uint8_t> & in, bool is_kvflash);
    bool restore_prefix_(int slot, const std::vector<int32_t> & prompt, PrefixStoreRef checkpoint);
    PrefixStoreEvent capture_prefix_(int slot, const PrefixCaptureTicket & ticket);
    StepResult step_batched_2dsd_(const StepPlan & plan);
    StepResult step_1d_(const StepPlan & plan);
    Inkling2SeqStepResult run_forward_1d_(int slot, const int32_t * tokens, int n_tokens,
                                          int pos0, bool need_logits, int capture_q_rows);
    int32_t sample_slot_1d_(SeqSlot & seq, const Inkling2SeqStepResult & r, bool is_greedy);
    bool    try_solo_dspark_step_(const StepInput & in, DecodeOutput & out);

    ggml_backend_t               backend_ = nullptr;
    const Inkling2Weights &      weights_;
    Inkling2Cache *              multi_cache_ = nullptr;
    bool                         batched_2dsd_ = false;
    int                          n_slots_ = 1;
    int                          prefill_chunk_ = 512;
    int                          verify_width_ = 0;
    Inkling2SeqForwardFn         custom_forward_;

    std::vector<Inkling2Cache>   owned_slot_views_;
    std::vector<Inkling2Cache *> caches_;
    std::unique_ptr<Inkling2SlotKv>           slot_kv_;
    std::unique_ptr<SeqSlotManager>           slots_;
    std::unique_ptr<Inkling2SlotStateAdapter> state_adapter_;
    std::unique_ptr<Inkling2SlotStateAdapter> state_adapter_with_draft_;
    std::vector<std::unique_ptr<SlotState>>   slot_state_;
    std::vector<PrefixCheckpoint>             prefix_checkpoints_;
};

} // namespace luce::common
