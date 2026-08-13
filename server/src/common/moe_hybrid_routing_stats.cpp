#include "moe_hybrid_routing_stats.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <system_error>

namespace dflash::common {

namespace {

bool routing_shape_size(int n_layer, int n_expert, size_t & size) {
    if (n_layer <= 0 || n_expert <= 0 ||
        (size_t) n_layer > std::numeric_limits<size_t>::max() /
                             (size_t) n_expert) {
        return false;
    }
    size = (size_t) n_layer * (size_t) n_expert;
    return size <= (size_t) std::numeric_limits<int>::max();
}

bool parse_routing_header(const std::string & line,
                          int & n_layer,
                          int & n_expert,
                          int & n_expert_used) {
    size_t offset = 0;
    const auto consume = [&](const char * literal) {
        const size_t length = std::strlen(literal);
        if (line.compare(offset, length, literal) != 0) return false;
        offset += length;
        return true;
    };
    const auto parse_int = [&](int & value) {
        const char * begin = line.data() + offset;
        const char * end = line.data() + line.size();
        const auto result = std::from_chars(begin, end, value);
        if (result.ec != std::errc{} || result.ptr == begin) return false;
        offset = (size_t) (result.ptr - line.data());
        return true;
    };

    if (!consume("# hotness table: n_layer=") || !parse_int(n_layer) ||
        !consume(" n_expert=") || !parse_int(n_expert) ||
        !consume(" n_expert_used=") || !parse_int(n_expert_used)) {
        return false;
    }
    return line.find_first_not_of(" \t\r", offset) == std::string::npos;
}

bool parse_count_row(const std::string & line, std::vector<uint64_t> & row) {
    const auto parse_value = [&](size_t begin, size_t end) {
        const size_t first = line.find_first_not_of(" \t\r", begin);
        if (first == std::string::npos || first >= end) return false;
        const size_t last = line.find_last_not_of(" \t\r", end - 1);
        if (last < first) return false;

        uint64_t value = 0;
        const char * value_begin = line.data() + first;
        const char * value_end = line.data() + last + 1;
        const auto parsed = std::from_chars(value_begin, value_end, value);
        if (parsed.ec != std::errc{} || parsed.ptr != value_end) return false;
        row.push_back(value);
        return true;
    };

    // Commas are the canonical format. Preserve whitespace-separated legacy
    // profiles so existing DFLASH_*_HOTNESS files remain loadable.
    if (line.find(',') == std::string::npos) {
        size_t begin = line.find_first_not_of(" \t\r");
        while (begin != std::string::npos) {
            const size_t end = line.find_first_of(" \t\r", begin);
            const size_t token_end = end == std::string::npos ? line.size() : end;
            if (!parse_value(begin, token_end)) return false;
            begin = line.find_first_not_of(" \t\r", token_end);
        }
        return true;
    }

    size_t begin = 0;
    while (begin <= line.size()) {
        const size_t comma = line.find(',', begin);
        const size_t end = comma == std::string::npos ? line.size() : comma;
        if (!parse_value(begin, end)) return false;
        if (comma == std::string::npos) return true;
        begin = comma + 1;
    }
    return false;
}

}  // namespace

size_t MoeHybridRoutingStats::index_of(int layer_idx, int expert_idx) const {
    return (size_t)layer_idx * (size_t)n_expert + (size_t)expert_idx;
}

bool MoeHybridRoutingStats::init(int n_layer_, int n_expert_, int n_expert_used_) {
    size_t count_size = 0;
    if (!routing_shape_size(n_layer_, n_expert_, count_size) ||
        n_expert_used_ <= 0 || n_expert_used_ > n_expert_) {
        return false;
    }
    n_layer = n_layer_;
    n_expert = n_expert_;
    n_expert_used = n_expert_used_;
    counts.assign(count_size, 0);
    layer_totals.assign((size_t)n_layer, 0);
    return true;
}

bool MoeHybridRoutingStats::init(const MoeHybridConfig & cfg) {
    return init(cfg.n_layer, cfg.n_expert, cfg.n_expert_used);
}

bool MoeHybridRoutingStats::valid(std::string * err) const {
    size_t count_size = 0;
    if (!routing_shape_size(n_layer, n_expert, count_size) ||
        n_expert_used <= 0 || n_expert_used > n_expert) {
        if (err) *err = "invalid routing profile dimensions";
        return false;
    }
    if (counts.size() != count_size ||
        layer_totals.size() != (size_t) n_layer) {
        if (err) *err = "routing profile storage does not match its dimensions";
        return false;
    }
    for (int il = 0; il < n_layer; ++il) {
        uint64_t total = 0;
        for (int ie = 0; ie < n_expert; ++ie) {
            const uint64_t value = counts[index_of(il, ie)];
            if (value > std::numeric_limits<uint64_t>::max() - total) {
                if (err) {
                    *err = "routing profile count overflow at layer " +
                        std::to_string(il);
                }
                return false;
            }
            total += value;
        }
        if (total != layer_totals[(size_t) il]) {
            if (err) {
                *err = "routing profile total mismatch at layer " +
                    std::to_string(il);
            }
            return false;
        }
    }
    return true;
}

bool MoeHybridRoutingStats::matches(int n_layer_, int n_expert_, int n_expert_used_) const {
    return n_layer == n_layer_ &&
           n_expert == n_expert_ &&
           n_expert_used == n_expert_used_ &&
           valid();
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
    const size_t index = index_of(layer_idx, expert_idx);
    return index < counts.size() ? counts[index] : 0;
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
    size_t count_size = 0;
    if (layer_idx < 0 || layer_idx >= n_layer ||
        !routing_shape_size(n_layer, n_expert, count_size) ||
        counts.size() != count_size) {
        return {};
    }
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
    if (!valid()) return;

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
    if (!valid(err)) return false;

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
    bool header_seen = false;
    std::vector<uint64_t> all_counts;
    std::string line;
    int row_index = 0;

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            if (line.rfind("# hotness table:", 0) == 0) {
                int parsed_layer = 0;
                int parsed_expert = 0;
                int parsed_used = 0;
                size_t shape_size = 0;
                if (header_seen || !parse_routing_header(
                        line, parsed_layer, parsed_expert, parsed_used) ||
                    !routing_shape_size(
                        parsed_layer, parsed_expert, shape_size) ||
                    parsed_used <= 0 || parsed_used > parsed_expert) {
                    if (err) *err = "invalid routing profile header";
                    return false;
                }
                file_n_layer = parsed_layer;
                file_n_expert = parsed_expert;
                file_n_expert_used = parsed_used;
                header_seen = true;
            }
            continue;
        }

        std::vector<uint64_t> row;
        if (!parse_count_row(line, row)) {
            if (err) {
                *err = "malformed value in row " +
                    std::to_string(row_index);
            }
            return false;
        }

        if (row.empty()) continue;
        if (row.size() > (size_t) std::numeric_limits<int>::max()) {
            if (err) *err = "routing profile row is too wide";
            return false;
        }

        if (file_n_expert == 0) {
            file_n_expert = (int)row.size();
        } else if ((int)row.size() != file_n_expert) {
            if (err) {
                *err = "inconsistent row width at layer " +
                    std::to_string(row_index);
            }
            return false;
        }

        if (row.size() > all_counts.max_size() - all_counts.size()) {
            if (err) *err = "routing profile is too large";
            return false;
        }
        all_counts.insert(all_counts.end(), row.begin(), row.end());
        ++row_index;
    }
    if (f.bad()) {
        if (err) *err = "failed while reading routing profile";
        return false;
    }

    if (file_n_expert <= 0 || all_counts.empty()) {
        if (err) *err = "no data rows found";
        return false;
    }

    const size_t detected_layers_size =
        all_counts.size() / (size_t) file_n_expert;
    if (detected_layers_size > (size_t) std::numeric_limits<int>::max()) {
        if (err) *err = "routing profile has too many layers";
        return false;
    }
    const int detected_layers = (int) detected_layers_size;
    if (file_n_layer == 0) file_n_layer = detected_layers;
    if (file_n_expert_used == 0) {
        file_n_expert_used = std::min(8, file_n_expert);
    }

    size_t expected_count_size = 0;
    if (!routing_shape_size(
            file_n_layer, file_n_expert, expected_count_size) ||
        file_n_expert_used <= 0 || file_n_expert_used > file_n_expert) {
        if (err) *err = "invalid routing profile dimensions";
        return false;
    }
    if (all_counts.size() != expected_count_size) {
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
            const uint64_t count = tmp.counts[tmp.index_of(il, ie)];
            if (count > std::numeric_limits<uint64_t>::max() - total) {
                if (err) *err = "routing profile count overflow at layer " +
                    std::to_string(il);
                return false;
            }
            total += count;
        }
        tmp.layer_totals[(size_t)il] = total;
    }

    if (!tmp.valid(err)) return false;
    out = std::move(tmp);
    return true;
}

}  // namespace dflash::common
