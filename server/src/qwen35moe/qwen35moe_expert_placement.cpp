#include "qwen35moe_expert_placement.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <numeric>

namespace dflash::common {

bool Qwen35MoeExpertPlacement::matches(const TargetWeights & w) const {
    return w.is_moe &&
           n_layer == w.n_layer &&
           n_expert == w.n_expert &&
           n_expert_used == w.n_expert_used &&
           (int)hot_counts.size() == n_layer &&
           (int)hot_expert_ids.size() == n_layer;
}

bool Qwen35MoeExpertPlacement::empty() const {
    return hot_counts.empty();
}

bool Qwen35MoeExpertPlacement::is_hot(int layer_idx, int expert_idx) const {
    if (layer_idx < 0 || layer_idx >= n_layer || expert_idx < 0 || expert_idx >= n_expert) {
        return false;
    }
    const auto & hot = hot_expert_ids[(size_t)layer_idx];
    return std::find(hot.begin(), hot.end(), expert_idx) != hot.end();
}

bool Qwen35MoeExpertPlacement::save_json(const std::string & path, std::string * err) const {
    if (n_layer <= 0 || n_expert <= 0 || (int)hot_counts.size() != n_layer ||
        (int)hot_expert_ids.size() != n_layer) {
        if (err) *err = "placement not initialized";
        return false;
    }

    nlohmann::json j;
    j["arch"] = "qwen35moe";
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

bool Qwen35MoeExpertPlacement::load_json(const std::string & path,
                                         Qwen35MoeExpertPlacement & out,
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

    if (j.value("arch", std::string()) != "qwen35moe") {
        if (err) *err = "unexpected arch";
        return false;
    }

    Qwen35MoeExpertPlacement tmp;
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

bool Qwen35MoeExpertPlacement::build_from_stats(const Qwen35MoeRoutingStats & stats,
                                                int total_hot_budget,
                                                int min_hot_per_layer,
                                                Qwen35MoeExpertPlacement & out,
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

    Qwen35MoeExpertPlacement tmp;
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

bool Qwen35MoeExpertPlacement::build_from_stats_with_layer_bytes(
    const Qwen35MoeRoutingStats & stats,
    const std::vector<uint64_t> & layer_expert_bytes,
    uint64_t total_hot_budget_bytes,
    int min_hot_per_layer,
    Qwen35MoeExpertPlacement & out,
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
        floor_bytes += (uint64_t)per_layer_floor * layer_expert_bytes[(size_t)il];
    }
    if (floor_bytes > total_hot_budget_bytes) {
        if (err) *err = "min_hot_per_layer exceeds byte budget";
        return false;
    }

    Qwen35MoeExpertPlacement tmp;
    tmp.n_layer = stats.n_layer;
    tmp.n_expert = stats.n_expert;
    tmp.n_expert_used = stats.n_expert_used;
    tmp.hot_counts.assign((size_t)tmp.n_layer, per_layer_floor);

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
