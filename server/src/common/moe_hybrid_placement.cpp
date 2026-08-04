#include "moe_hybrid_placement.h"
#include "moe_hybrid_routing_stats.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>

namespace dflash::common {

bool resolve_moe_expert_owner_placement(
        int primary_gpu,
        int requested_expert_gpu,
        MoeExpertOwnerPlacement & out,
        std::string * err) {
    if (primary_gpu < 0 || requested_expert_gpu < -1) {
        if (err) *err = "MoE owner GPU indices must be non-negative";
        return false;
    }

    int expert_gpu = requested_expert_gpu;
    if (expert_gpu < 0) {
        const char * raw = std::getenv("DFLASH_MOE_TP_GPU");
        if (!raw || !*raw) raw = std::getenv("DFLASH_DS4_MOE_TP_GPU");
        if (!raw || !*raw) {
            raw = std::getenv("DFLASH_MOE_EXPERT_COMPUTE_IPC_GPU");
        }
        if (!raw || !*raw) {
            expert_gpu = primary_gpu;
        } else {
            errno = 0;
            char * end = nullptr;
            const long parsed = std::strtol(raw, &end, 10);
            if (errno != 0 || end == raw || *end != '\0' || parsed < 0 ||
                parsed > std::numeric_limits<int>::max()) {
                if (err) {
                    *err = std::string("invalid routed-expert GPU: ") + raw;
                }
                return false;
            }
            expert_gpu = static_cast<int>(parsed);
        }
    }

    out.primary_gpu = primary_gpu;
    out.expert_gpu = expert_gpu;
    return true;
}

bool MoeHybridPlacement::matches(int n_layer_, int n_expert_, int n_expert_used_) const {
    return n_layer == n_layer_ &&
           n_expert == n_expert_ &&
           n_expert_used == n_expert_used_ &&
           (int)hot_counts.size() == n_layer &&
           (int)hot_expert_ids.size() == n_layer;
}

bool MoeHybridPlacement::matches(const MoeHybridConfig & cfg) const {
    return matches(cfg.n_layer, cfg.n_expert, cfg.n_expert_used);
}

bool MoeHybridPlacement::empty() const {
    return hot_counts.empty();
}

bool MoeHybridPlacement::is_hot(int layer_idx, int expert_idx) const {
    if (layer_idx < 0 || layer_idx >= n_layer || expert_idx < 0 || expert_idx >= n_expert) {
        return false;
    }
    const auto & hot = hot_expert_ids[(size_t)layer_idx];
    return std::find(hot.begin(), hot.end(), expert_idx) != hot.end();
}

bool MoeHybridPlacement::save_json(const std::string & path, const std::string & arch_name,
                                   std::string * err) const {
    if (n_layer <= 0 || n_expert <= 0 || (int)hot_counts.size() != n_layer ||
        (int)hot_expert_ids.size() != n_layer) {
        if (err) *err = "placement not initialized";
        return false;
    }

    nlohmann::json j;
    j["arch"] = arch_name;
    j["version"] = 1;
    j["n_layer"] = n_layer;
    j["n_expert"] = n_expert;
    j["n_expert_used"] = n_expert_used;
    j["total_hot"] = total_hot;
    j["hot_counts"] = hot_counts;
    j["hot_expert_ids"] = hot_expert_ids;

    std::ofstream f(path);
    if (!f) {
        if (err) *err = "failed to open output file";
        return false;
    }
    f << j.dump(2);
    if (!f) {
        if (err) *err = "failed to write json";
        return false;
    }
    return true;
}

bool MoeHybridPlacement::load_json(const std::string & path,
                                   MoeHybridPlacement & out,
                                   std::string * err) {
    std::ifstream f(path);
    if (!f) {
        if (err) *err = "failed to open input file";
        return false;
    }

    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception & ex) {
        if (err) *err = ex.what();
        return false;
    }

    // Accept both legacy "qwen35moe" and new "moe_hybrid" / any arch string.
    // We don't reject based on arch — the caller validates dimensions.

    MoeHybridPlacement tmp;
    try {
        tmp.n_layer = j.value("n_layer", 0);
        tmp.n_expert = j.value("n_expert", 0);
        tmp.n_expert_used = j.value("n_expert_used", 0);
        tmp.total_hot = j.value("total_hot", 0);
        tmp.hot_counts = j.value("hot_counts", std::vector<int>{});
        tmp.hot_expert_ids = j.value("hot_expert_ids", std::vector<std::vector<int32_t>>{});
    } catch (const std::exception & ex) {
        if (err) *err = std::string("type error: ") + ex.what();
        return false;
    }

    if (tmp.n_layer <= 0 || tmp.n_expert <= 0 || tmp.n_expert_used <= 0 ||
        (int)tmp.hot_counts.size() != tmp.n_layer ||
        (int)tmp.hot_expert_ids.size() != tmp.n_layer) {
        if (err) *err = "invalid placement dimensions";
        return false;
    }

    out = std::move(tmp);
    return true;
}

bool MoeHybridPlacement::build_from_stats(const MoeHybridRoutingStats & stats,
                                          int total_hot_budget,
                                          int min_hot_per_layer,
                                          MoeHybridPlacement & out,
                                          std::string * err) {
    if (stats.empty() || stats.n_layer <= 0 || stats.n_expert <= 0) {
        if (err) *err = "stats not initialized";
        return false;
    }
    if (min_hot_per_layer < 0) min_hot_per_layer = 0;
    if (total_hot_budget <= 0) {
        if (err) *err = "total_hot_budget must be > 0";
        return false;
    }

    const int per_layer_floor = std::min(min_hot_per_layer, stats.n_expert);
    const int floor_total = per_layer_floor * stats.n_layer;
    if (floor_total > total_hot_budget) {
        if (err) *err = "min_hot_per_layer exceeds total budget";
        return false;
    }

    MoeHybridPlacement tmp;
    tmp.n_layer = stats.n_layer;
    tmp.n_expert = stats.n_expert;
    tmp.n_expert_used = stats.n_expert_used;
    tmp.hot_counts.assign((size_t)tmp.n_layer, per_layer_floor);

    std::vector<std::vector<int>> ranked((size_t)tmp.n_layer);
    for (int il = 0; il < tmp.n_layer; ++il) {
        ranked[(size_t)il] = stats.ranked_experts(il);
    }

    int remaining = total_hot_budget - floor_total;
    while (remaining > 0) {
        int best_layer = -1;
        uint64_t best_gain = 0;
        for (int il = 0; il < tmp.n_layer; ++il) {
            const int cur_hot = tmp.hot_counts[(size_t)il];
            if (cur_hot >= tmp.n_expert) continue;
            const int next_expert = ranked[(size_t)il][(size_t)cur_hot];
            const uint64_t gain = stats.count(il, next_expert);
            if (best_layer < 0 || gain > best_gain) {
                best_layer = il;
                best_gain = gain;
            }
        }
        if (best_layer < 0) break;
        tmp.hot_counts[(size_t)best_layer]++;
        remaining--;
    }

    tmp.total_hot = std::accumulate(tmp.hot_counts.begin(), tmp.hot_counts.end(), 0);
    tmp.hot_expert_ids.resize((size_t)tmp.n_layer);
    for (int il = 0; il < tmp.n_layer; ++il) {
        const int hot_n = tmp.hot_counts[(size_t)il];
        auto & hot = tmp.hot_expert_ids[(size_t)il];
        hot.reserve((size_t)hot_n);
        for (int i = 0; i < hot_n; ++i) {
            hot.push_back((int32_t)ranked[(size_t)il][(size_t)i]);
        }
    }

    out = std::move(tmp);
    return true;
}

bool MoeHybridPlacement::build_from_stats_with_layer_bytes(
    const MoeHybridRoutingStats & stats,
    const std::vector<uint64_t> & layer_expert_bytes,
    uint64_t total_hot_budget_bytes,
    int min_hot_per_layer,
    MoeHybridPlacement & out,
    std::string * err) {
    if (stats.empty() || stats.n_layer <= 0 || stats.n_expert <= 0) {
        if (err) *err = "stats not initialized";
        return false;
    }
    if ((int)layer_expert_bytes.size() != stats.n_layer) {
        if (err) *err = "layer_expert_bytes size mismatch";
        return false;
    }
    if (min_hot_per_layer < 0) min_hot_per_layer = 0;
    if (total_hot_budget_bytes == 0) {
        if (err) *err = "total_hot_budget_bytes must be > 0";
        return false;
    }

    const int per_layer_floor = std::min(min_hot_per_layer, stats.n_expert);
    uint64_t floor_bytes = 0;
    for (int il = 0; il < stats.n_layer; ++il) {
        if (layer_expert_bytes[(size_t)il] > 0)
            floor_bytes += (uint64_t)per_layer_floor * layer_expert_bytes[(size_t)il];
    }
    if (floor_bytes > total_hot_budget_bytes) {
        if (err) *err = "min_hot_per_layer exceeds byte budget";
        return false;
    }

    MoeHybridPlacement tmp;
    tmp.n_layer = stats.n_layer;
    tmp.n_expert = stats.n_expert;
    tmp.n_expert_used = stats.n_expert_used;
    tmp.hot_counts.resize((size_t)tmp.n_layer);
    for (int il = 0; il < tmp.n_layer; ++il) {
        tmp.hot_counts[(size_t)il] = (layer_expert_bytes[(size_t)il] > 0) ? per_layer_floor : 0;
    }

    std::vector<std::vector<int>> ranked((size_t)tmp.n_layer);
    for (int il = 0; il < tmp.n_layer; ++il) {
        ranked[(size_t)il] = stats.ranked_experts(il);
    }

    uint64_t remaining = total_hot_budget_bytes - floor_bytes;
    while (true) {
        int best_layer = -1;
        double best_value = -1.0;
        uint64_t best_gain = 0;
        for (int il = 0; il < tmp.n_layer; ++il) {
            const int cur_hot = tmp.hot_counts[(size_t)il];
            if (cur_hot >= tmp.n_expert) continue;
            const uint64_t bytes = layer_expert_bytes[(size_t)il];
            if (bytes == 0 || bytes > remaining) continue;
            const int next_expert = ranked[(size_t)il][(size_t)cur_hot];
            const uint64_t gain = stats.count(il, next_expert);
            const double value = (double)gain / (double)bytes;
            if (best_layer < 0 || value > best_value ||
                (value == best_value && gain > best_gain)) {
                best_layer = il;
                best_value = value;
                best_gain = gain;
            }
        }
        if (best_layer < 0) break;
        tmp.hot_counts[(size_t)best_layer]++;
        remaining -= layer_expert_bytes[(size_t)best_layer];
    }

    tmp.total_hot = std::accumulate(tmp.hot_counts.begin(), tmp.hot_counts.end(), 0);
    tmp.hot_expert_ids.resize((size_t)tmp.n_layer);
    for (int il = 0; il < tmp.n_layer; ++il) {
        const int hot_n = tmp.hot_counts[(size_t)il];
        auto & hot = tmp.hot_expert_ids[(size_t)il];
        hot.reserve((size_t)hot_n);
        for (int i = 0; i < hot_n; ++i) {
            hot.push_back((int32_t)ranked[(size_t)il][(size_t)i]);
        }
    }

    out = std::move(tmp);
    return true;
}

}  // namespace dflash::common
