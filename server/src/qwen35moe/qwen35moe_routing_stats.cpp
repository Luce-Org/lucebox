#include "qwen35moe_routing_stats.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>

namespace dflash::common {

size_t Qwen35MoeRoutingStats::index_of(int layer_idx, int expert_idx) const {
    return (size_t)layer_idx * (size_t)n_expert + (size_t)expert_idx;
}

bool Qwen35MoeRoutingStats::init_from_weights(const TargetWeights & w) {
    if (!w.is_moe || w.n_layer <= 0 || w.n_expert <= 0 || w.n_expert_used <= 0) {
        return false;
    }
    n_layer = w.n_layer;
    n_expert = w.n_expert;
    n_expert_used = w.n_expert_used;
    counts.assign((size_t)n_layer * (size_t)n_expert, 0);
    layer_totals.assign((size_t)n_layer, 0);
    return true;
}

bool Qwen35MoeRoutingStats::matches(const TargetWeights & w) const {
    return w.is_moe &&
           n_layer == w.n_layer &&
           n_expert == w.n_expert &&
           n_expert_used == w.n_expert_used &&
           counts.size() == (size_t)n_layer * (size_t)n_expert &&
           layer_totals.size() == (size_t)n_layer;
}

bool Qwen35MoeRoutingStats::empty() const {
    return counts.empty();
}

uint64_t Qwen35MoeRoutingStats::count(int layer_idx, int expert_idx) const {
    if (layer_idx < 0 || layer_idx >= n_layer || expert_idx < 0 || expert_idx >= n_expert) {
        return 0;
    }
    return counts[index_of(layer_idx, expert_idx)];
}

bool Qwen35MoeRoutingStats::observe(int layer_idx, const int32_t * expert_ids, int n_ids) {
    if (!expert_ids || layer_idx < 0 || layer_idx >= n_layer || n_ids < 0) {
        return false;
    }
    for (int i = 0; i < n_ids; ++i) {
        const int expert_idx = expert_ids[i];
        if (expert_idx < 0 || expert_idx >= n_expert) {
            return false;
        }
    }
    for (int i = 0; i < n_ids; ++i) {
        const int expert_idx = expert_ids[i];
        counts[index_of(layer_idx, expert_idx)]++;
        layer_totals[(size_t) layer_idx]++;
    }
    return true;
}

bool Qwen35MoeRoutingStats::observe_selected_tensor(ggml_backend_t backend,
                                                    int layer_idx,
                                                    ggml_tensor * selected,
                                                    std::string * err) {
    if (!backend || !selected) {
        if (err) *err = "null backend or selected tensor";
        return false;
    }
    if (selected->type != GGML_TYPE_I32) {
        if (err) *err = "selected tensor must be i32";
        return false;
    }
    if (selected->ne[0] <= 0 || selected->ne[1] <= 0) {
        if (err) *err = "selected tensor has invalid shape";
        return false;
    }
    const int64_t n_ids = selected->ne[0] * selected->ne[1];
    std::vector<int32_t> ids((size_t)n_ids);
    ggml_backend_tensor_get(selected, ids.data(), 0, sizeof(int32_t) * (size_t)n_ids);
    if (!observe(layer_idx, ids.data(), (int)n_ids)) {
        if (err) *err = "failed to observe selected ids";
        return false;
    }
    return true;
}

std::vector<int> Qwen35MoeRoutingStats::ranked_experts(int layer_idx) const {
    if (layer_idx < 0 || layer_idx >= n_layer) return {};
    std::vector<int> ranked((size_t)n_expert);
    std::iota(ranked.begin(), ranked.end(), 0);
    std::stable_sort(ranked.begin(), ranked.end(),
        [&](int a, int b) {
            const uint64_t ca = counts[index_of(layer_idx, a)];
            const uint64_t cb = counts[index_of(layer_idx, b)];
            if (ca != cb) return ca > cb;
            return a < b;
        });
    return ranked;
}

std::vector<int> Qwen35MoeRoutingStats::hot_experts(int layer_idx, int hot_count) const {
    std::vector<int> ranked = ranked_experts(layer_idx);
    if (hot_count < 0) hot_count = 0;
    if ((size_t) hot_count < ranked.size()) {
        ranked.resize((size_t) hot_count);
    }
    return ranked;
}


bool Qwen35MoeRoutingStats::save_csv(const std::string & path, std::string * err) const {
    if (n_layer <= 0 || n_expert <= 0 || counts.size() != (size_t)n_layer * (size_t)n_expert) {
        if (err) *err = "routing stats not initialized";
        return false;
    }

    std::ofstream f(path);
    if (!f) {
        if (err) *err = "failed to open output file";
        return false;
    }

    // Header comments
    f << "# hotness table: n_layer=" << n_layer
      << " n_expert=" << n_expert
      << " n_expert_used=" << n_expert_used << "\n";
    f << "# format: one row per layer, columns are expert activation counts (expert 0..N-1)\n";

    for (int il = 0; il < n_layer; ++il) {
        for (int ie = 0; ie < n_expert; ++ie) {
            if (ie > 0) f << ',';
            f << counts[index_of(il, ie)];
        }
        f << '\n';
    }

    if (!f) {
        if (err) *err = "failed to write csv";
        return false;
    }
    return true;
}

bool Qwen35MoeRoutingStats::load_csv(const std::string & path,
                                     Qwen35MoeRoutingStats & out,
                                     std::string * err) {
    std::ifstream f(path);
    if (!f) {
        if (err) *err = "failed to open input file";
        return false;
    }

    int n_layer = 0, n_expert = 0, n_expert_used = 0;
    std::vector<uint64_t> all_counts;
    std::string line;

    while (std::getline(f, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            // Try to parse header metadata from comment
            if (line.find("n_layer=") != std::string::npos) {
                std::sscanf(line.c_str(), "# hotness table: n_layer=%d n_expert=%d n_expert_used=%d",
                            &n_layer, &n_expert, &n_expert_used);
            }
            continue;
        }

        // Parse CSV row: comma-separated uint64 values
        std::vector<uint64_t> row;
        const char * p = line.c_str();
        while (*p) {
            // Skip whitespace
            while (*p == ' ' || *p == '\t') ++p;
            if (!*p) break;
            char * end = nullptr;
            uint64_t val = std::strtoull(p, &end, 10);
            if (end == p) {
                if (err) *err = "malformed value in row " + std::to_string((int)(all_counts.size() / std::max((size_t)n_expert, (size_t)1)));
                return false;
            }
            row.push_back(val);
            p = end;
            if (*p == ',') ++p;
        }

        if (row.empty()) continue;

        // Infer n_expert from first data row
        if (n_expert == 0) {
            n_expert = (int)row.size();
        } else if ((int)row.size() != n_expert) {
            if (err) *err = "inconsistent row width at layer " + std::to_string((int)(all_counts.size() / (size_t)n_expert));
            return false;
        }

        all_counts.insert(all_counts.end(), row.begin(), row.end());
    }

    if (n_expert <= 0 || all_counts.empty()) {
        if (err) *err = "no data rows found";
        return false;
    }

    const int detected_layers = (int)(all_counts.size() / (size_t)n_expert);
    if (n_layer == 0) n_layer = detected_layers;
    if (n_expert_used == 0) n_expert_used = 8;  // default for Qwen3.5-MoE

    if ((int)all_counts.size() != n_layer * n_expert) {
        if (err) *err = "row count (" + std::to_string(detected_layers) + ") doesn't match n_layer (" + std::to_string(n_layer) + ")";
        return false;
    }

    Qwen35MoeRoutingStats tmp;
    tmp.n_layer = n_layer;
    tmp.n_expert = n_expert;
    tmp.n_expert_used = n_expert_used;
    tmp.counts = std::move(all_counts);
    tmp.layer_totals.assign((size_t)n_layer, 0);
    for (int il = 0; il < n_layer; ++il) {
        uint64_t total = 0;
        for (int ie = 0; ie < n_expert; ++ie) {
            total += tmp.counts[tmp.index_of(il, ie)];
        }
        tmp.layer_totals[(size_t)il] = total;
    }

    out = std::move(tmp);
    return true;
}

}  // namespace dflash::common
