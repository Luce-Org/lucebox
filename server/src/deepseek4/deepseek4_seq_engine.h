#pragma once

#include "common/concurrency/seq_engine.h"
#include "common/concurrency/paged_kv_offload.h"
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
    bool reserve_decode(const StepPlan & plan) override;
    size_t kv_offload_capacity() const override { return offload_.capacity(); }
    KvOffloadState kv_offload_state(int slot) const override { return offload_.state(slot); }
    bool offload_kv(int slot, size_t bytes, std::string & error) override {
        return offload_.suspend(slot, bytes, error);
    }
    bool restore_kv(int slot, std::string & error) override;
    bool evict_kv(int slot, int32_t pending_token, std::string & error) override;
    bool kv_restore_feasible(int slot) const override {
        return slots_.kv_restore_feasible(slot);
    }
    void retire(int slot) override;
    bool token_is_eos(int32_t token) const override;

private:
    bool set_block(int slot, int logical, int32_t physical);
    void fail_prefill(int slot, std::vector<PrefillOutput> & outputs,
                      const std::string & error);

    DeepSeek4Backend & b_;
    SeqSlotManager slots_;
    PagedKvOffload offload_;
    uint32_t stride_ = 0;
    std::vector<int32_t> host_tables_;
};

} // namespace dflash::common
