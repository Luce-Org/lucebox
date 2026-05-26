// Reusable qwen35moe routing-statistics scaffold for Phase 2 expert placement.

#pragma once

#include "internal.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct Qwen35MoeRoutingStats {
    int n_layer       = 0;
    int n_expert      = 0;
    int n_expert_used = 0;

    // Flattened [n_layer][n_expert] activation counts.
    std::vector<uint64_t> counts;
    std::vector<uint64_t> layer_totals;

    bool init_from_weights(const TargetWeights & w);
    bool matches(const TargetWeights & w) const;
    bool empty() const;

    uint64_t count(int layer_idx, int expert_idx) const;
    bool observe(int layer_idx, const int32_t * expert_ids, int n_ids);
    bool observe_selected_tensor(ggml_backend_t backend,
                                 int layer_idx,
                                 ggml_tensor * selected,
                                 std::string * err = nullptr);

    std::vector<int> ranked_experts(int layer_idx) const;
    std::vector<int> hot_experts(int layer_idx, int hot_count) const;

    bool save_csv(const std::string & path, std::string * err = nullptr) const;
    static bool load_csv(const std::string & path,
                         Qwen35MoeRoutingStats & out,
                         std::string * err = nullptr);

private:
    size_t index_of(int layer_idx, int expert_idx) const;
};

}  // namespace dflash::common
