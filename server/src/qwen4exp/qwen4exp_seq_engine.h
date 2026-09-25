#pragma once

#include "common/concurrency/paged_kv_pool.h"
#include "common/concurrency/seq_engine.h"
#include "common/concurrency/seq_slot_manager.h"
#include "qwen4exp_cache.h"

#include <memory>
#include <vector>

namespace luce::common {

// Experimental N-way independent-slot engine. Each slot owns a full cache;
// the pool is admission/headroom bookkeeping only (there is no paged KV).
class Qwen4ExpSeqEngine final : public SeqEngine {
public:
    Qwen4ExpSeqEngine(ggml_backend_t backend, const Qwen4ExpWeights & weights,
                      std::vector<Qwen4ExpCache *> caches, int max_ctx,
                      int prefill_chunk = 512);
    ~Qwen4ExpSeqEngine() override;

    int slot_count() const override { return slots_.slot_count(); }
    int max_context() const override { return slots_.max_context(); }
    AdmitResult admit(uint64_t request_id, const std::vector<int32_t> & prompt,
                      const SamplerCfg & sampler) override;
    StepPlanLimits step_plan_limits(int decode_rows) const override;
    bool reserve_decode(const StepPlan & plan) override;
    StepResult step(const StepPlan & plan) override;
    void retire(int slot) override;
    bool token_is_eos(int32_t token) const override;

private:
    ggml_backend_t backend_;
    const Qwen4ExpWeights & weights_;
    std::vector<Qwen4ExpCache *> caches_;
    PagedKvPool pool_;
    SeqSlotManager slots_;
    Qwen4ExpBatchedDecodeWorkspace decode_workspace_;
    int prefill_chunk_;
    int prefill_owner_ = -1;
};

} // namespace luce::common
