// qwen35moe expert placement config derived from per-layer routing statistics.

#pragma once

#include "qwen35moe_routing_stats.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct Qwen35MoeExpertPlacement {
    int n_layer       = 0;
    int n_expert      = 0;
    int n_expert_used = 0;
    int total_hot     = 0;

    // Number of hot experts allocated to each layer.
    std::vector<int> hot_counts;
    // Ranked hot expert ids kept on GPU per layer.
    std::vector<std::vector<int32_t>> hot_expert_ids;

    bool matches(const TargetWeights & w) const;
    bool empty() const;
    bool is_hot(int layer_idx, int expert_idx) const;

    bool save_json(const std::string & path, std::string * err = nullptr) const;
    static bool load_json(const std::string & path,
                          Qwen35MoeExpertPlacement & out,
                          std::string * err = nullptr);

    static bool build_from_stats(const Qwen35MoeRoutingStats & stats,
                                 int total_hot_budget,
                                 int min_hot_per_layer,
                                 Qwen35MoeExpertPlacement & out,
                                 std::string * err = nullptr);

    static bool build_from_stats_with_layer_bytes(
        const Qwen35MoeRoutingStats & stats,
        const std::vector<uint64_t> & layer_expert_bytes,
        uint64_t total_hot_budget_bytes,
        int min_hot_per_layer,
        Qwen35MoeExpertPlacement & out,
        std::string * err = nullptr);
};

}  // namespace dflash::common
