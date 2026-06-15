// Common MoE routing statistics for expert placement decisions.

#pragma once

#include "moe_hybrid_types.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct MoeHybridRoutingStats {
    int n_layer       = 0;
    int n_expert      = 0;
    int n_expert_used = 0;

    // Flattened [n_layer][n_expert] activation counts.
    std::vector<uint64_t> counts;
    std::vector<uint64_t> layer_totals;

    bool init(int n_layer, int n_expert, int n_expert_used);
    bool init(const MoeHybridConfig & cfg);
    bool matches(int n_layer, int n_expert, int n_expert_used) const;
    bool matches(const MoeHybridConfig & cfg) const;
    bool empty() const;

    uint64_t count(int layer_idx, int expert_idx) const;
    bool observe(int layer_idx, const int32_t * expert_ids, int n_ids);
    bool observe_selected_tensor(ggml_backend_t backend,
                                 int layer_idx,
                                 ggml_tensor * selected,
                                 std::string * err = nullptr);

    std::vector<int> ranked_experts(int layer_idx) const;
    std::vector<int> hot_experts(int layer_idx, int hot_count) const;

    // Print StreamMoE-style frequency analysis to stderr.
    void print_freq_analysis() const;

    bool save_csv(const std::string & path, std::string * err = nullptr) const;
    static bool load_csv(const std::string & path,
                         MoeHybridRoutingStats & out,
                         std::string * err = nullptr);

private:
    size_t index_of(int layer_idx, int expert_idx) const;
};

}  // namespace dflash::common
