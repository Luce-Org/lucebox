// Common MoE expert placement — determines which experts are hot (GPU) vs cold (CPU).

#pragma once

#include "moe_hybrid_types.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace dflash::common {

struct MoeHybridRoutingStats;  // forward decl

inline uint64_t moe_hybrid_core_bytes_from_memory(const char * log_prefix,
                                                  size_t gpu_free,
                                                  size_t gpu_total) {
    if (gpu_total >= gpu_free) {
        return (uint64_t) gpu_total - (uint64_t) gpu_free;
    }

    std::printf("[%s] dynamic placement: free memory exceeds reported GPU total "
                "(free=%.2f GiB, total=%.2f GiB), using core=0 for UMA budget accounting\n",
                log_prefix ? log_prefix : "moe-hybrid",
                gpu_free / 1024.0 / 1024.0 / 1024.0,
                gpu_total / 1024.0 / 1024.0 / 1024.0);
    return 0;
}

struct MoeHybridPlacement {
    int n_layer       = 0;
    int n_expert      = 0;
    int n_expert_used = 0;
    int total_hot     = 0;

    // Number of hot experts allocated to each layer.
    std::vector<int> hot_counts;
    // Ranked hot expert ids kept on GPU per layer.
    std::vector<std::vector<int32_t>> hot_expert_ids;

    bool matches(int n_layer, int n_expert, int n_expert_used) const;
    bool matches(const MoeHybridConfig & cfg) const;
    bool empty() const;
    bool is_hot(int layer_idx, int expert_idx) const;

    bool save_json(const std::string & path, const std::string & arch_name = "moe_hybrid",
                   std::string * err = nullptr) const;
    static bool load_json(const std::string & path,
                          MoeHybridPlacement & out,
                          std::string * err = nullptr);

    static bool build_from_stats(const MoeHybridRoutingStats & stats,
                                 int total_hot_budget,
                                 int min_hot_per_layer,
                                 MoeHybridPlacement & out,
                                 std::string * err = nullptr);

    static bool build_from_stats_with_layer_bytes(
        const MoeHybridRoutingStats & stats,
        const std::vector<uint64_t> & layer_expert_bytes,
        uint64_t total_hot_budget_bytes,
        int min_hot_per_layer,
        MoeHybridPlacement & out,
        std::string * err = nullptr);
};

}  // namespace dflash::common
