#pragma once

#include "common/concurrency/seq_engine.h"
#include "common/concurrency/seq_slot_manager.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

class DeepSeek4Backend;

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
    void retire(int slot) override;
    bool token_is_eos(int32_t token) const override;

private:
    bool set_block(int slot, int logical, int32_t physical);
    void fail_prefill(int slot, std::vector<PrefillOutput> & outputs,
                      const std::string & error);

    DeepSeek4Backend & b_;
    SeqSlotManager slots_;
    uint32_t stride_ = 0;
    std::vector<int32_t> host_tables_;
};

} // namespace dflash::common
