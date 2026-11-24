// Inkling2SeqEngine — Concurrent N-slot 2DSD SeqEngine for Inkling 2 Mini.
//
// Shares all 128 MoE experts across N concurrent decoding sequences in ONE batched
// Prelude / Fused Multi-Pass Core / Fused Multi-Depth Coda GPU step via multi-slot
// contiguous 60-plane tree caches (`plane_eff = seq_slot * n_planes + plane`), while
// reusing single-slot prefill/commit/draft routines over zero-copy per-slot views
// and scaling 2DSD confidence/width with active slot count `N`.

#pragma once

#include "common/concurrency/paged_kv_pool.h"
#include "common/concurrency/seq_engine.h"
#include "common/concurrency/seq_slot_manager.h"
#include "inkling2_2dsd_executor.h"
#include "inkling2_cache.h"
#include "inkling2_internal.h"

#include <memory>
#include <vector>

namespace luce::common {

class Inkling2SeqEngine final : public SeqEngine {
public:
    Inkling2SeqEngine(ggml_backend_t target_backend,
                      const Inkling2Weights & weights,
                      Inkling2Cache & multi_cache,
                      int max_concurrency,
                      int prefill_chunk);
    ~Inkling2SeqEngine() override;

    Inkling2SeqEngine(const Inkling2SeqEngine &) = delete;
    Inkling2SeqEngine & operator=(const Inkling2SeqEngine &) = delete;

    int slot_count() const override { return slots_.slot_count(); }
    int max_context() const override { return slots_.max_context(); }
    KvPoolTokens kv_pool_tokens() const override { return slots_.kv_pool_tokens(); }

    AdmitResult admit(uint64_t request_id,
                      const std::vector<int32_t> & prompt,
                      const SamplerCfg & sampler) override;

    StepPlanLimits step_plan_limits(int decode_rows) const override;
    bool reserve_decode(const StepPlan & plan) override;
    StepResult step(const StepPlan & plan) override;
    void retire(int slot) override;
    bool token_is_eos(int32_t token) const override;
    int max_decode_tokens() const override { return kInkling2TwoDSDMaxCommit; }

    // Drop per-slot view graph workspaces before the backend frees/rebuilds multi_cache.
    void release_device_state();
    // Rebuild zero-copy per-slot views after multi_cache is reinstated.
    bool reinstate_device_state(int prefill_chunk);

private:
    struct SlotState;

    bool init_slot_views();
    void reset_slot_session(int slot);

    ggml_backend_t backend_ = nullptr;
    const Inkling2Weights & weights_;
    Inkling2Cache & multi_cache_;
    int n_slots_ = 1;
    int prefill_chunk_ = 512;
    PagedKvPool pool_;
    SeqSlotManager slots_;
    std::vector<Inkling2Cache> slot_caches_;
    std::vector<std::unique_ptr<SlotState>> slot_state_;
};

} // namespace luce::common
