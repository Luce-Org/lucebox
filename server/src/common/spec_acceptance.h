#pragma once

#include <algorithm>

namespace dflash::common {

// Accepted/emitted verifier positions, including the always-committed seed.
// Keep numerator and denominator together so tree/chain branches and early
// returns cannot silently report a rate using only one side of a step.
class SpecAcceptanceStats {
public:
    void record_chain(int accepted_width, int verified_width,
                      int emitted_width, int remaining_budget) {
        // A linear prefix beyond the request's output budget cannot be used.
        record(std::min(accepted_width, emitted_width),
               std::min(verified_width, remaining_budget));
    }

    void record_tree(int accepted_emitted, int candidate_nodes, int remaining_budget) {
        // Count actual topology nodes, not graph-padding rows. Like chain
        // telemetry, cap offers by the request's remaining output budget.
        record(accepted_emitted, std::min(1 + candidate_nodes, remaining_budget));
    }

    int accepted() const { return accepted_; }
    int offered() const { return offered_; }
    float rate() const {
        return offered_ > 0 ? static_cast<float>(
            static_cast<double>(accepted_) / offered_) : 0.0f;
    }

private:
    void record(int accepted, int offered) {
        offered = std::max(0, offered);
        accepted_ += std::clamp(accepted, 0, offered);
        offered_ += offered;
    }

    int accepted_ = 0;
    int offered_ = 0;
};

} // namespace dflash::common
