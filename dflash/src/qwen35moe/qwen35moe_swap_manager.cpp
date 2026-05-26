#include "qwen35moe_swap_manager.h"

#include <algorithm>

namespace dflash::common {

bool build_qwen35moe_swap_plan(const Qwen35MoeExpertPlacement & current,
                               const Qwen35MoeRoutingStats & stats,
                               const Qwen35MoeSwapPolicy & policy,
                               Qwen35MoeSwapPlan & out,
                               std::string * err) {
    if (current.n_layer != stats.n_layer ||
        current.n_expert != stats.n_expert ||
        current.n_expert_used != stats.n_expert_used) {
        if (err) *err = "placement/stats dimension mismatch";
        return false;
    }

    out = Qwen35MoeSwapPlan{};
    out.next_placement = current;
    if (policy.max_swaps_total <= 0) {
        return true;
    }

    struct Candidate {
        Qwen35MoeSwapAction action;
        uint64_t gain_delta = 0;
    };
    std::vector<Candidate> candidates;

    for (int il = 0; il < current.n_layer; ++il) {
        const auto ranked = stats.ranked_experts(il);
        if (ranked.empty()) continue;

        const auto & current_hot = current.hot_expert_ids[(size_t)il];
        if (current_hot.empty()) continue;

        // Find weakest currently-hot expert.
        int weakest_hot = current_hot[0];
        uint64_t weakest_count = stats.count(il, weakest_hot);
        for (int32_t expert : current_hot) {
            const uint64_t c = stats.count(il, expert);
            if (c < weakest_count) {
                weakest_count = c;
                weakest_hot = expert;
            }
        }

        // Find best currently-cold expert by rank.
        int best_cold = -1;
        uint64_t best_cold_count = 0;
        for (int expert : ranked) {
            if (!current.is_hot(il, expert)) {
                best_cold = expert;
                best_cold_count = stats.count(il, expert);
                break;
            }
        }
        if (best_cold < 0) continue;

        if (best_cold_count < weakest_count + policy.min_promote_gain) {
            continue;
        }

        Candidate cand;
        cand.action.layer_idx = il;
        cand.action.evict_expert = weakest_hot;
        cand.action.promote_expert = best_cold;
        cand.action.evict_count = weakest_count;
        cand.action.promote_count = best_cold_count;
        cand.gain_delta = best_cold_count - weakest_count;
        candidates.push_back(std::move(cand));
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate & a, const Candidate & b) {
                  if (a.gain_delta != b.gain_delta) return a.gain_delta > b.gain_delta;
                  return a.action.layer_idx < b.action.layer_idx;
              });

    const int n_apply = std::min<int>((int)candidates.size(), policy.max_swaps_total);
    for (int i = 0; i < n_apply; ++i) {
        const auto & cand = candidates[(size_t)i];
        auto & hot = out.next_placement.hot_expert_ids[(size_t)cand.action.layer_idx];
        auto it = std::find(hot.begin(), hot.end(), cand.action.evict_expert);
        if (it == hot.end()) continue;
        *it = (int32_t)cand.action.promote_expert;
        out.actions.push_back(cand.action);
    }
    return true;
}

}  // namespace dflash::common
