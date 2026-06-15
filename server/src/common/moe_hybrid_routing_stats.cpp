#include "moe_hybrid_routing_stats.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>

namespace dflash::common {

size_t MoeHybridRoutingStats::index_of(int layer_idx, int expert_idx) const {
    return (size_t)layer_idx * (size_t)n_expert + (size_t)expert_idx;
}

bool MoeHybridRoutingStats::init(int n_layer_, int n_expert_, int n_expert_used_) {
    if (n_layer_ <= 0 || n_expert_ <= 0 || n_expert_used_ <= 0) {
        return false;
    }
    n_layer = n_layer_;
    n_expert = n_expert_;
    n_expert_used = n_expert_used_;
    counts.assign((size_t)n_layer * (size_t)n_expert, 0);
    layer_totals.assign((size_t)n_layer, 0);
    return true;
}

bool MoeHybridRoutingStats::init(const MoeHybridConfig & cfg) {
    return init(cfg.n_layer, cfg.n_expert, cfg.n_expert_used);
}

bool MoeHybridRoutingStats::matches(int n_layer_, int n_expert_, int n_expert_used_) const {
    return n_layer == n_layer_ &&
           n_expert == n_expert_ &&
           n_expert_used == n_expert_used_ &&
           counts.size() == (size_t)n_layer * (size_t)n_expert &&
           layer_totals.size() == (size_t)n_layer;
}

bool MoeHybridRoutingStats::matches(const MoeHybridConfig & cfg) const {
    return matches(cfg.n_layer, cfg.n_expert, cfg.n_expert_used);
}

bool MoeHybridRoutingStats::empty() const {
    return counts.empty();
}

uint64_t MoeHybridRoutingStats::count(int layer_idx, int expert_idx) const {
    if (layer_idx < 0 || layer_idx >= n_layer || expert_idx < 0 || expert_idx >= n_expert) {
        return 0;
    }
    return counts[index_of(layer_idx, expert_idx)];
}

bool MoeHybridRoutingStats::observe(int layer_idx, const int32_t * expert_ids, int n_ids) {
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
        layer_totals[(size_t)layer_idx]++;
    }
    return true;
}

bool MoeHybridRoutingStats::observe_selected_tensor(ggml_backend_t backend,
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

std::vector<int> MoeHybridRoutingStats::ranked_experts(int layer_idx) const {
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

std::vector<int> MoeHybridRoutingStats::hot_experts(int layer_idx, int hot_count) const {
    std::vector<int> ranked = ranked_experts(layer_idx);
    if (hot_count < 0) hot_count = 0;
    if ((size_t)hot_count < ranked.size()) {
        ranked.resize((size_t)hot_count);
    }
    return ranked;
}

void MoeHybridRoutingStats::print_freq_analysis() const {
    if (n_layer <= 0 || n_expert <= 0 || counts.empty()) return;

    uint64_t total_all = 0;
    for (int il = 0; il < n_layer; ++il) total_all += layer_totals[(size_t)il];
    if (total_all == 0) return;

    // Estimate total tokens: layer_totals[0] / n_expert_used
    uint64_t total_tokens = (n_expert_used > 0 && layer_totals[0] > 0)
        ? layer_totals[0] / (uint64_t)n_expert_used : 0;
    double avg_k = (n_layer > 0 && total_tokens > 0)
        ? (double)total_all / ((double)n_layer * (double)total_tokens) : 0.0;

    std::fprintf(stderr, "\n=== Expert Frequency Analysis ===\n");
    std::fprintf(stderr, "Tokens tracked: %llu, avg-K=%.2f\n\n",
                 (unsigned long long)total_tokens, avg_k);

    int experts_for_80_total = 0;
    std::vector<uint64_t> sorted((size_t)n_expert);

    for (int il = 0; il < n_layer; ++il) {
        uint64_t tact = layer_totals[(size_t)il];
        if (tact == 0) {
            std::fprintf(stderr, "Layer %2d:   0 unique\n", il);
            continue;
        }
        for (int ie = 0; ie < n_expert; ++ie)
            sorted[(size_t)ie] = counts[index_of(il, ie)];
        std::sort(sorted.begin(), sorted.end(), std::greater<uint64_t>());

        int unique = 0;
        for (int ie = 0; ie < n_expert; ++ie) if (sorted[(size_t)ie] > 0) unique++;

        uint64_t cum = 0;
        uint64_t top10 = 0, top30 = 0, top60 = 0;
        int n50 = 0, n80 = 0, n90 = 0;
        for (int ie = 0; ie < n_expert; ++ie) {
            cum += sorted[(size_t)ie];
            if (ie == 9)  top10 = cum;
            if (ie == 29) top30 = cum;
            if (ie == 59) top60 = cum;
            if (!n50 && cum * 100 >= tact * 50) n50 = ie + 1;
            if (!n80 && cum * 100 >= tact * 80) n80 = ie + 1;
            if (!n90 && cum * 100 >= tact * 90) n90 = ie + 1;
        }
        // If there are fewer experts than a bucket size, the ie==N-1 check
        // never fires and topN stays 0, printing a bogus 0%. cum == tact here
        // (all experts summed), so the bucket covers everything → 100%.
        if (n_expert < 10) top10 = cum;
        if (n_expert < 30) top30 = cum;
        if (n_expert < 60) top60 = cum;
        std::fprintf(stderr, "Layer %2d: %3d unique, top-10 %.0f%%, top-30 %.0f%%, top-60 %.0f%% "
                             "(50%%@%d, 80%%@%d, 90%%@%d)\n",
                     il, unique,
                     100.0 * (double)top10 / (double)tact,
                     100.0 * (double)top30 / (double)tact,
                     100.0 * (double)top60 / (double)tact,
                     n50, n80, n90);
        experts_for_80_total += n80;
    }

    double avg80 = (n_layer > 0) ? (double)experts_for_80_total / n_layer : 0.0;
    std::fprintf(stderr, "\n--- Overall Summary ---\n");
    std::fprintf(stderr, "80%% hit rate: %d experts pinned (avg %.0f/layer)\n",
                 experts_for_80_total, avg80);
    std::fprintf(stderr, "Model: %d layers x %d experts, top-%d routing\n",
                 n_layer, n_expert, n_expert_used);
}

bool MoeHybridRoutingStats::save_csv(const std::string & path, std::string * err) const {
    if (n_layer <= 0 || n_expert <= 0 || counts.size() != (size_t)n_layer * (size_t)n_expert) {
        if (err) *err = "routing stats not initialized";
        return false;
    }

    std::ofstream f(path);
    if (!f) {
        if (err) *err = "failed to open output file";
        return false;
    }

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

bool MoeHybridRoutingStats::load_csv(const std::string & path,
                                     MoeHybridRoutingStats & out,
                                     std::string * err) {
    std::ifstream f(path);
    if (!f) {
        if (err) *err = "failed to open input file";
        return false;
    }

    int file_n_layer = 0, file_n_expert = 0, file_n_expert_used = 0;
    std::vector<uint64_t> all_counts;
    std::string line;

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            if (line.find("n_layer=") != std::string::npos) {
                std::sscanf(line.c_str(), "# hotness table: n_layer=%d n_expert=%d n_expert_used=%d",
                            &file_n_layer, &file_n_expert, &file_n_expert_used);
            }
            continue;
        }

        std::vector<uint64_t> row;
        const char * p = line.c_str();
        while (*p) {
            while (*p == ' ' || *p == '\t') ++p;
            if (!*p) break;
            char * end = nullptr;
            uint64_t val = std::strtoull(p, &end, 10);
            if (end == p) {
                if (err) *err = "malformed value in row " + std::to_string((int)(all_counts.size() / std::max((size_t)file_n_expert, (size_t)1)));
                return false;
            }
            row.push_back(val);
            p = end;
            if (*p == ',') ++p;
        }

        if (row.empty()) continue;

        if (file_n_expert == 0) {
            file_n_expert = (int)row.size();
        } else if ((int)row.size() != file_n_expert) {
            if (err) *err = "inconsistent row width at layer " + std::to_string((int)(all_counts.size() / (size_t)file_n_expert));
            return false;
        }

        all_counts.insert(all_counts.end(), row.begin(), row.end());
    }

    if (file_n_expert <= 0 || all_counts.empty()) {
        if (err) *err = "no data rows found";
        return false;
    }

    const int detected_layers = (int)(all_counts.size() / (size_t)file_n_expert);
    if (file_n_layer == 0) file_n_layer = detected_layers;
    if (file_n_expert_used == 0) file_n_expert_used = 8;  // default

    if ((int)all_counts.size() != file_n_layer * file_n_expert) {
        if (err) *err = "row count (" + std::to_string(detected_layers) + ") doesn't match n_layer (" + std::to_string(file_n_layer) + ")";
        return false;
    }

    MoeHybridRoutingStats tmp;
    tmp.n_layer = file_n_layer;
    tmp.n_expert = file_n_expert;
    tmp.n_expert_used = file_n_expert_used;
    tmp.counts = std::move(all_counts);
    tmp.layer_totals.assign((size_t)file_n_layer, 0);
    for (int il = 0; il < file_n_layer; ++il) {
        uint64_t total = 0;
        for (int ie = 0; ie < file_n_expert; ++ie) {
            total += tmp.counts[tmp.index_of(il, ie)];
        }
        tmp.layer_totals[(size_t)il] = total;
    }

    out = std::move(tmp);
    return true;
}

}  // namespace dflash::common
