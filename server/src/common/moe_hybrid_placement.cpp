#include "moe_hybrid_placement.h"
#include "moe_hybrid_routing_stats.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>

namespace dflash::common {

namespace {

bool validate_placement(const MoeHybridPlacement & placement,
                        bool require_total,
                        std::string * err) {
    if (placement.n_layer <= 0 || placement.n_expert <= 0 ||
        placement.n_expert_used <= 0 ||
        placement.n_expert_used > placement.n_expert ||
        (size_t) placement.n_layer >
            (size_t) std::numeric_limits<int>::max() /
                (size_t) placement.n_expert ||
        placement.hot_counts.size() != (size_t) placement.n_layer ||
        placement.hot_expert_ids.size() != (size_t) placement.n_layer) {
        if (err) *err = "invalid placement dimensions";
        return false;
    }

    int computed_total = 0;
    for (int il = 0; il < placement.n_layer; ++il) {
        const int hot_count = placement.hot_counts[(size_t) il];
        const auto & ids = placement.hot_expert_ids[(size_t) il];
        if (hot_count < 0 || hot_count > placement.n_expert ||
            ids.size() != (size_t) hot_count) {
            if (err) {
                *err = "placement count does not match expert ids at layer " +
                    std::to_string(il);
            }
            return false;
        }
        computed_total += hot_count;
        for (int32_t id : ids) {
            if (id < 0 || id >= placement.n_expert) {
                if (err) {
                    *err = "placement expert id is out of range at layer " +
                        std::to_string(il);
                }
                return false;
            }
        }
        std::vector<int32_t> sorted_ids = ids;
        std::sort(sorted_ids.begin(), sorted_ids.end());
        if (std::adjacent_find(sorted_ids.begin(), sorted_ids.end()) !=
            sorted_ids.end()) {
            if (err) {
                *err = "placement contains a duplicate expert at layer " +
                    std::to_string(il);
            }
            return false;
        }
    }
    if (require_total && placement.total_hot != computed_total) {
        if (err) *err = "placement total does not match its layer counts";
        return false;
    }
    return true;
}

}  // namespace

bool MoeHybridPlacement::valid(std::string * err) const {
    return validate_placement(*this, true, err);
}

bool MoeHybridPlacement::matches(int n_layer_, int n_expert_, int n_expert_used_) const {
    return n_layer == n_layer_ &&
           n_expert == n_expert_ &&
           n_expert_used == n_expert_used_ &&
           valid();
}

bool MoeHybridPlacement::matches(const MoeHybridConfig & cfg) const {
    return matches(cfg.n_layer, cfg.n_expert, cfg.n_expert_used);
}

bool MoeHybridPlacement::empty() const {
    return hot_counts.empty();
}

bool MoeHybridPlacement::is_hot(int layer_idx, int expert_idx) const {
    if (layer_idx < 0 || layer_idx >= n_layer || expert_idx < 0 ||
        expert_idx >= n_expert ||
        (size_t) layer_idx >= hot_expert_ids.size()) {
        return false;
    }
    const auto & hot = hot_expert_ids[(size_t)layer_idx];
    return std::find(hot.begin(), hot.end(), expert_idx) != hot.end();
}

bool MoeHybridPlacement::save_json(const std::string & path, const std::string & arch_name,
                                   std::string * err) const {
    if (!valid(err)) return false;

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

    if (!tmp.valid(err)) return false;

    out = std::move(tmp);
    return true;
}

bool MoeHybridPlacement::build_from_stats(const MoeHybridRoutingStats & stats,
                                          int total_hot_budget,
                                          int min_hot_per_layer,
                                          MoeHybridPlacement & out,
                                          std::string * err) {
    if (!stats.valid(err)) return false;
    if (min_hot_per_layer < 0) min_hot_per_layer = 0;
    if (total_hot_budget <= 0) {
        if (err) *err = "total_hot_budget must be > 0";
        return false;
    }

    const int per_layer_floor = std::min(min_hot_per_layer, stats.n_expert);
    const int64_t floor_total =
        (int64_t) per_layer_floor * stats.n_layer;
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

    int64_t remaining = (int64_t) total_hot_budget - floor_total;
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
    if (!stats.valid(err)) return false;
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
        const uint64_t expert_bytes = layer_expert_bytes[(size_t) il];
        if (expert_bytes > 0 && (uint64_t) per_layer_floor >
                (std::numeric_limits<uint64_t>::max() - floor_bytes) /
                    expert_bytes) {
            if (err) *err = "minimum hot placement byte count overflow";
            return false;
        }
        floor_bytes += (uint64_t) per_layer_floor * expert_bytes;
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

bool MoeHybridPlacement::expand_from_stats_with_layer_bytes(
    const MoeHybridRoutingStats & stats,
    const std::vector<uint64_t> & layer_expert_bytes,
    uint64_t total_hot_budget_bytes,
    MoeHybridPlacement & in_out,
    std::string * err) {
    if (!stats.valid(err)) return false;
    if ((int) layer_expert_bytes.size() != stats.n_layer) {
        if (err) *err = "layer_expert_bytes size mismatch";
        return false;
    }
    if (in_out.n_layer != stats.n_layer ||
        in_out.n_expert != stats.n_expert ||
        in_out.n_expert_used != stats.n_expert_used) {
        if (err) *err = "existing placement shape does not match stats";
        return false;
    }
    if (!validate_placement(in_out, false, err)) return false;
    if (total_hot_budget_bytes == 0) {
        if (err) *err = "total_hot_budget_bytes must be > 0";
        return false;
    }

    MoeHybridPlacement tmp = in_out;
    std::vector<std::vector<uint8_t>> resident(
        (size_t) stats.n_layer,
        std::vector<uint8_t>((size_t) stats.n_expert, 0));
    uint64_t used_bytes = 0;
    tmp.total_hot = 0;
    for (int il = 0; il < stats.n_layer; ++il) {
        const auto & ids = tmp.hot_expert_ids[(size_t) il];
        const int hot_count = tmp.hot_counts[(size_t) il];
        if (hot_count < 0 || ids.size() != (size_t) hot_count) {
            if (err) *err = "existing placement count does not match ids";
            return false;
        }
        if (ids.size() > (size_t) (std::numeric_limits<int>::max() -
                                  tmp.total_hot)) {
            if (err) *err = "existing placement expert count overflow";
            return false;
        }
        tmp.total_hot += hot_count;
        const uint64_t expert_bytes = layer_expert_bytes[(size_t) il];
        if (expert_bytes > 0 && ids.size() >
                (std::numeric_limits<uint64_t>::max() - used_bytes) /
                    expert_bytes) {
            if (err) *err = "existing placement byte count overflow";
            return false;
        }
        used_bytes += (uint64_t) ids.size() * expert_bytes;
        for (int32_t id : ids) {
            if (id < 0 || id >= stats.n_expert ||
                resident[(size_t) il][(size_t) id]) {
                if (err) *err = "existing placement contains an invalid expert";
                return false;
            }
            resident[(size_t) il][(size_t) id] = 1;
        }
    }
    if (used_bytes > total_hot_budget_bytes) {
        if (err) *err = "existing placement exceeds byte budget";
        return false;
    }

    std::vector<std::vector<int>> ranked((size_t) stats.n_layer);
    std::vector<size_t> next((size_t) stats.n_layer, 0);
    for (int il = 0; il < stats.n_layer; ++il) {
        ranked[(size_t) il] = stats.ranked_experts(il);
    }

    uint64_t remaining = total_hot_budget_bytes - used_bytes;
    while (true) {
        int best_layer = -1;
        int best_expert = -1;
        double best_value = -1.0;
        uint64_t best_gain = 0;
        for (int il = 0; il < stats.n_layer; ++il) {
            const uint64_t bytes = layer_expert_bytes[(size_t) il];
            if (bytes == 0 || bytes > remaining) continue;
            auto & cursor = next[(size_t) il];
            const auto & layer_ranked = ranked[(size_t) il];
            while (cursor < layer_ranked.size() &&
                   resident[(size_t) il]
                           [(size_t) layer_ranked[cursor]]) {
                ++cursor;
            }
            if (cursor == layer_ranked.size()) continue;
            const int expert = layer_ranked[cursor];
            const uint64_t gain = stats.count(il, expert);
            const double value = (double) gain / (double) bytes;
            if (best_layer < 0 || value > best_value ||
                (value == best_value && gain > best_gain)) {
                best_layer = il;
                best_expert = expert;
                best_value = value;
                best_gain = gain;
            }
        }
        if (best_layer < 0) break;
        tmp.hot_expert_ids[(size_t) best_layer].push_back(
            (int32_t) best_expert);
        tmp.hot_counts[(size_t) best_layer]++;
        tmp.total_hot++;
        resident[(size_t) best_layer][(size_t) best_expert] = 1;
        remaining -= layer_expert_bytes[(size_t) best_layer];
        ++next[(size_t) best_layer];
    }

    in_out = std::move(tmp);
    return true;
}

bool MoeHybridPlacement::build_critical_path_balanced_from_stats(
    const MoeHybridRoutingStats & stats,
    const std::vector<uint64_t> & layer_expert_bytes,
    const std::vector<uint64_t> & layer_main_fixed_bytes,
    uint64_t total_hot_budget_bytes,
    const MoeHybridCriticalPathConfig & config,
    MoeHybridPlacement & out,
    std::string * err) {
    if (!stats.valid(err)) return false;
    if ((int) layer_expert_bytes.size() != stats.n_layer ||
        (int) layer_main_fixed_bytes.size() != stats.n_layer) {
        if (err) *err = "critical-path layer byte vector size mismatch";
        return false;
    }
    if (total_hot_budget_bytes == 0) {
        if (err) *err = "total_hot_budget_bytes must be > 0";
        return false;
    }
    if (config.active_experts <= 0 ||
        config.active_experts > stats.n_expert_used) {
        if (err) *err = "active_experts must be within the routing profile width";
        return false;
    }
    if (!std::isfinite(config.main_to_peer_rate) ||
        config.main_to_peer_rate <= 0.0) {
        if (err) *err = "main_to_peer_rate must be finite and > 0";
        return false;
    }

    const int floor = std::clamp(
        config.min_hot_per_layer, 0, stats.n_expert);
    uint64_t used_bytes = 0;
    for (int il = 0; il < stats.n_layer; ++il) {
        const uint64_t expert_bytes = layer_expert_bytes[(size_t) il];
        if (expert_bytes == 0) continue;
        if ((uint64_t) floor >
            (std::numeric_limits<uint64_t>::max() - used_bytes) /
                expert_bytes) {
            if (err) *err = "minimum hot placement byte count overflow";
            return false;
        }
        used_bytes += (uint64_t) floor * expert_bytes;
    }
    if (used_bytes > total_hot_budget_bytes) {
        if (err) *err = "min_hot_per_layer exceeds byte budget";
        return false;
    }

    MoeHybridPlacement tmp;
    tmp.n_layer = stats.n_layer;
    tmp.n_expert = stats.n_expert;
    tmp.n_expert_used = stats.n_expert_used;
    tmp.hot_counts.assign((size_t) tmp.n_layer, 0);

    std::vector<std::vector<int>> ranked((size_t) tmp.n_layer);
    std::vector<std::vector<uint64_t>> prefix_counts((size_t) tmp.n_layer);
    for (int il = 0; il < tmp.n_layer; ++il) {
        ranked[(size_t) il] = stats.ranked_experts(il);
        auto & prefix = prefix_counts[(size_t) il];
        prefix.assign((size_t) tmp.n_expert + 1, 0);
        for (int n = 0; n < tmp.n_expert; ++n) {
            const uint64_t count =
                stats.count(il, ranked[(size_t) il][(size_t) n]);
            prefix[(size_t) n + 1] = prefix[(size_t) n] + count;
        }
        if (layer_expert_bytes[(size_t) il] > 0) {
            tmp.hot_counts[(size_t) il] = floor;
        }
    }

    auto layer_cost = [&](int il, int hot_count) {
        const uint64_t expert_bytes = layer_expert_bytes[(size_t) il];
        if (expert_bytes == 0) return 0.0;
        const auto & prefix = prefix_counts[(size_t) il];
        const uint64_t total = prefix.back();
        const double hot_probability = total > 0
            ? (double) prefix[(size_t) hot_count] / (double) total
            : (double) hot_count / (double) tmp.n_expert;
        const double routed_bytes =
            (double) config.active_experts * (double) expert_bytes;
        const double main_work =
            (double) layer_main_fixed_bytes[(size_t) il] +
            routed_bytes * hot_probability;
        const double peer_work = routed_bytes * (1.0 - hot_probability);
        return std::max(
            main_work / config.main_to_peer_rate,
            peer_work);
    };

    struct LayerChoice {
        uint64_t bytes = 0;
        double cost = 0.0;
        int hot_count = 0;
    };
    struct PlacementState {
        uint64_t bytes = 0;
        double cost = 0.0;
        size_t previous = 0;
        int hot_count = 0;
    };

    const uint64_t remaining = total_hot_budget_bytes - used_bytes;
    std::vector<std::vector<LayerChoice>> layer_choices(
        (size_t) tmp.n_layer);
    for (int il = 0; il < tmp.n_layer; ++il) {
        const uint64_t expert_bytes = layer_expert_bytes[(size_t) il];
        auto & choices = layer_choices[(size_t) il];
        if (expert_bytes == 0) {
            choices.push_back({0, 0.0, 0});
            continue;
        }

        double best_cost = std::numeric_limits<double>::infinity();
        for (int hot_count = floor; hot_count <= tmp.n_expert; ++hot_count) {
            const uint64_t extra_count = (uint64_t) (hot_count - floor);
            if (extra_count > 0 &&
                expert_bytes > remaining / extra_count) {
                break;
            }
            const uint64_t bytes = extra_count * expert_bytes;
            const double cost = layer_cost(il, hot_count);
            if (!std::isfinite(cost)) {
                if (err) *err = "critical-path layer cost is not finite";
                return false;
            }
            // Larger counts with no lower layer cost are dominated by this
            // layer's earlier choices and cannot improve a global solution.
            if (cost < best_cost) {
                choices.push_back({bytes, cost, hot_count});
                best_cost = cost;
            }
        }
    }

    // Multiple-choice knapsack over each layer's non-dominated hot counts.
    // The frontier stays sparse in bytes, so byte budgets do not need to be
    // quantized and differently sized expert tensors remain exact.
    std::vector<std::vector<PlacementState>> stages((size_t) tmp.n_layer + 1);
    stages[0].push_back({0, 0.0, 0, 0});
    for (int il = 0; il < tmp.n_layer; ++il) {
        const auto & previous = stages[(size_t) il];
        std::vector<PlacementState> candidates;
        for (size_t previous_index = 0;
             previous_index < previous.size(); ++previous_index) {
            const PlacementState & base = previous[previous_index];
            for (const LayerChoice & choice : layer_choices[(size_t) il]) {
                if (choice.bytes > remaining - base.bytes) continue;
                candidates.push_back({
                    base.bytes + choice.bytes,
                    base.cost + choice.cost,
                    previous_index,
                    choice.hot_count,
                });
            }
        }
        std::sort(candidates.begin(), candidates.end(),
            [](const PlacementState & left, const PlacementState & right) {
                if (left.bytes != right.bytes) return left.bytes < right.bytes;
                if (left.cost != right.cost) return left.cost < right.cost;
                if (left.hot_count != right.hot_count) {
                    return left.hot_count < right.hot_count;
                }
                return left.previous < right.previous;
            });

        auto & frontier = stages[(size_t) il + 1];
        double best_cost = std::numeric_limits<double>::infinity();
        for (const PlacementState & candidate : candidates) {
            if (candidate.cost < best_cost) {
                frontier.push_back(candidate);
                best_cost = candidate.cost;
            }
        }
        if (frontier.empty()) {
            if (err) *err = "critical-path placement has no feasible solution";
            return false;
        }
    }

    const auto & final_stage = stages.back();
    size_t state_index = (size_t) std::distance(
        final_stage.begin(),
        std::min_element(
            final_stage.begin(), final_stage.end(),
            [](const PlacementState & left, const PlacementState & right) {
                return left.cost < right.cost ||
                    (left.cost == right.cost && left.bytes < right.bytes);
            }));
    for (int il = tmp.n_layer - 1; il >= 0; --il) {
        const PlacementState & state =
            stages[(size_t) il + 1][state_index];
        tmp.hot_counts[(size_t) il] = state.hot_count;
        state_index = state.previous;
    }

    tmp.total_hot =
        std::accumulate(tmp.hot_counts.begin(), tmp.hot_counts.end(), 0);
    tmp.hot_expert_ids.resize((size_t) tmp.n_layer);
    for (int il = 0; il < tmp.n_layer; ++il) {
        const int hot_count = tmp.hot_counts[(size_t) il];
        auto & hot = tmp.hot_expert_ids[(size_t) il];
        hot.reserve((size_t) hot_count);
        for (int n = 0; n < hot_count; ++n) {
            hot.push_back(
                (int32_t) ranked[(size_t) il][(size_t) n]);
        }
    }

    out = std::move(tmp);
    return true;
}

}  // namespace dflash::common
