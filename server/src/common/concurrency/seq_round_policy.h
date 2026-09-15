#pragma once
#include "seq_engine.h"

namespace dflash::common {
// One speculative opportunity, then one prompt service round. All decode
// rows remain owned by the caller and are included in every engine step.
class SeqRoundPolicy {
public:
    void plan(SeqEngine::StepPlan & plan,
              const std::vector<PrefillCandidate> & pending,
              const StepPlanLimits & limits, bool eligible) {
        const bool mixed = eligible && !plan.decode.empty() && !pending.empty();
        if (!mixed) next_speculative_ = true;
        if (mixed && next_speculative_) {
            plan.prefills.clear();
            next_speculative_ = false;
            return;
        }
        plan.prefills = plan_prefill_slices(pending, limits, rotation_);
        if (!plan.prefills.empty()) ++rotation_;
        next_speculative_ = true;
    }
    void reset() { next_speculative_ = true; }
private:
    bool next_speculative_ = true;
    size_t rotation_ = 0;
};
} // namespace dflash::common
