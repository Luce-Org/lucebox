// cluster_expert_placement.cpp - N-rank routed-expert ownership.
//
// GPU-free: MoeHybridRoutingStats is only read through count() and
// ranked_experts(); no ggml call is made here so the placement unit test can
// link without a device backend. JSON I/O is hand-rolled (same policy as
// cluster_protocol.h) so the loader has no third-party dependency.

#include "cluster/cluster_expert_placement.h"
#include "cluster/cluster_protocol.h"  // fnv1a64

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>

namespace luce::cluster {

namespace {

void set_err(std::string * err, const std::string & msg) {
    if (err) *err = msg;
}

bool dims_ok(int n_ranks, int n_layer, int n_expert, int n_expert_used, std::string * err) {
    if (n_ranks < 1) {
        set_err(err, "n_ranks must be >= 1");
        return false;
    }
    if (n_layer <= 0 || n_expert <= 0) {
        set_err(err, "n_layer and n_expert must be positive");
        return false;
    }
    if (n_expert_used <= 0 || n_expert_used > n_expert) {
        set_err(err, "n_expert_used must be in [1, n_expert]");
        return false;
    }
    if ((size_t) n_layer > (size_t) std::numeric_limits<int>::max() / (size_t) n_expert) {
        set_err(err, "n_layer * n_expert overflows");
        return false;
    }
    return true;
}

// ── minimal JSON reader ────────────────────────────────────────────────
// Accepts exactly the shape written by save_json (and by
// scripts/cluster/build_expert_placement.py): a flat object of integers plus
// one "owner" key holding an array of arrays of integers. Unknown keys are
// skipped (strings, numbers, nested arrays/objects) so a future field does
// not break older binaries.

class JsonCursor {
public:
    explicit JsonCursor(const std::string & text) : s_(text) {}

    bool parse_object(ClusterExpertPlacement & out, std::string * err) {
        skip_ws();
        if (!expect('{', err)) return false;
        skip_ws();
        if (peek() == '}') { ++pos_; return true; }
        for (;;) {
            skip_ws();
            std::string key;
            if (!parse_string(key, err)) return false;
            skip_ws();
            if (!expect(':', err)) return false;
            skip_ws();
            if (key == "owner") {
                if (!parse_owner(out, err)) return false;
            } else if (key == "n_ranks" || key == "n_layer" || key == "n_expert" ||
                       key == "n_expert_used" || key == "replicate_hot") {
                long long v = 0;
                if (!parse_int(v, err)) return false;
                if (v < 0 || v > std::numeric_limits<int>::max()) {
                    set_err(err, "placement json: value out of range for " + key);
                    return false;
                }
                if (key == "n_ranks") out.n_ranks = (int) v;
                else if (key == "n_layer") out.n_layer = (int) v;
                else if (key == "n_expert") out.n_expert = (int) v;
                else if (key == "n_expert_used") out.n_expert_used = (int) v;
                else out.replicate_hot = (int) v;
            } else {
                if (!skip_value(err)) return false;
            }
            skip_ws();
            const char c = peek();
            if (c == ',') { ++pos_; continue; }
            if (c == '}') { ++pos_; return true; }
            set_err(err, "placement json: expected ',' or '}'");
            return false;
        }
    }

private:
    char peek() const { return pos_ < s_.size() ? s_[pos_] : '\0'; }

    void skip_ws() {
        while (pos_ < s_.size() && std::isspace((unsigned char) s_[pos_])) ++pos_;
    }

    bool expect(char c, std::string * err) {
        if (peek() != c) {
            set_err(err, std::string("placement json: expected '") + c + "'");
            return false;
        }
        ++pos_;
        return true;
    }

    bool parse_string(std::string & out, std::string * err) {
        if (!expect('"', err)) return false;
        out.clear();
        while (pos_ < s_.size()) {
            const char c = s_[pos_++];
            if (c == '"') return true;
            if (c == '\\') {
                if (pos_ >= s_.size()) break;
                const char e = s_[pos_++];
                switch (e) {
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    case 'u':
                        // \uXXXX: keep the raw escape; keys we care about are ASCII.
                        out.push_back('?');
                        pos_ = std::min(s_.size(), pos_ + 4);
                        break;
                    default: out.push_back(e); break;
                }
                continue;
            }
            out.push_back(c);
        }
        set_err(err, "placement json: unterminated string");
        return false;
    }

    bool parse_int(long long & out, std::string * err) {
        skip_ws();
        const size_t start = pos_;
        if (peek() == '-') ++pos_;
        while (pos_ < s_.size() && std::isdigit((unsigned char) s_[pos_])) ++pos_;
        if (pos_ == start || (pos_ == start + 1 && s_[start] == '-')) {
            set_err(err, "placement json: expected integer");
            return false;
        }
        // Reject fractions/exponents: owner ids and dims are integers.
        if (peek() == '.' || peek() == 'e' || peek() == 'E') {
            set_err(err, "placement json: non-integer number");
            return false;
        }
        out = std::strtoll(s_.c_str() + start, nullptr, 10);
        return true;
    }

    bool parse_owner(ClusterExpertPlacement & out, std::string * err) {
        if (!expect('[', err)) return false;
        out.owner.clear();
        row_width_ = -1;
        skip_ws();
        if (peek() == ']') { ++pos_; return true; }
        for (;;) {
            skip_ws();
            if (!expect('[', err)) return false;
            int width = 0;
            skip_ws();
            if (peek() != ']') {
                for (;;) {
                    long long v = 0;
                    if (!parse_int(v, err)) return false;
                    if (v < kReplicated || v > std::numeric_limits<int32_t>::max()) {
                        set_err(err, "placement json: owner value out of range");
                        return false;
                    }
                    out.owner.push_back((int32_t) v);
                    ++width;
                    skip_ws();
                    const char c = peek();
                    if (c == ',') { ++pos_; continue; }
                    if (c == ']') break;
                    set_err(err, "placement json: expected ',' or ']' in owner row");
                    return false;
                }
            }
            ++pos_;  // ']'
            if (row_width_ < 0) row_width_ = width;
            else if (row_width_ != width) {
                set_err(err, "placement json: ragged owner rows");
                return false;
            }
            skip_ws();
            const char c = peek();
            if (c == ',') { ++pos_; continue; }
            if (c == ']') { ++pos_; return true; }
            set_err(err, "placement json: expected ',' or ']' after owner row");
            return false;
        }
    }

    bool skip_value(std::string * err) {
        skip_ws();
        const char c = peek();
        if (c == '"') { std::string tmp; return parse_string(tmp, err); }
        if (c == '{' || c == '[') {
            const char close = c == '{' ? '}' : ']';
            int depth = 0;
            bool in_str = false;
            while (pos_ < s_.size()) {
                const char d = s_[pos_++];
                if (in_str) {
                    if (d == '\\') ++pos_;
                    else if (d == '"') in_str = false;
                    continue;
                }
                if (d == '"') in_str = true;
                else if (d == c) ++depth;
                else if (d == close && --depth == 0) return true;
            }
            set_err(err, "placement json: unterminated container");
            return false;
        }
        // number / true / false / null
        while (pos_ < s_.size() && s_[pos_] != ',' && s_[pos_] != '}' && s_[pos_] != ']') ++pos_;
        return true;
    }

    const std::string & s_;
    size_t pos_ = 0;
    int row_width_ = -1;
};

}  // namespace

// ─── Queries ────────────────────────────────────────────────────────────

int32_t ClusterExpertPlacement::owner_of(int layer, int expert) const {
    if (layer < 0 || layer >= n_layer || expert < 0 || expert >= n_expert) return -2;
    const size_t idx = (size_t) layer * (size_t) n_expert + (size_t) expert;
    return idx < owner.size() ? owner[idx] : -2;
}

bool ClusterExpertPlacement::is_replicated(int layer, int expert) const {
    return owner_of(layer, expert) == kReplicated;
}

int32_t ClusterExpertPlacement::slot_owner(int layer, int expert, int slot) const {
    const int32_t o = owner_of(layer, expert);
    if (o >= 0) return o;
    if (o == kReplicated && n_ranks > 0 && slot >= 0) {
        return (int32_t) (((int64_t) expert + (int64_t) slot) % (int64_t) n_ranks);
    }
    return -2;
}

std::vector<int32_t> ClusterExpertPlacement::resident_experts(int layer, int rank) const {
    std::vector<int32_t> out;
    if (layer < 0 || layer >= n_layer) return out;
    for (int e = 0; e < n_expert; ++e) {
        const int32_t o = owner_of(layer, e);
        if (o == rank || o == kReplicated) out.push_back((int32_t) e);
    }
    return out;  // ascending by construction
}

int ClusterExpertPlacement::owned_count(int layer, int rank) const {
    if (layer < 0 || layer >= n_layer) return 0;
    int n = 0;
    for (int e = 0; e < n_expert; ++e) {
        if (owner_of(layer, e) == rank) ++n;
    }
    return n;
}

int ClusterExpertPlacement::total_owned(int rank) const {
    int n = 0;
    for (int il = 0; il < n_layer; ++il) n += owned_count(il, rank);
    return n;
}

bool ClusterExpertPlacement::valid(std::string * err) const {
    if (!dims_ok(n_ranks, n_layer, n_expert, n_expert_used, err)) return false;
    if (replicate_hot < 0 || replicate_hot > n_expert) {
        set_err(err, "replicate_hot out of range");
        return false;
    }
    if (owner.size() != (size_t) n_layer * (size_t) n_expert) {
        set_err(err, "owner table size does not match n_layer * n_expert");
        return false;
    }
    std::vector<int> owned((size_t) n_ranks);
    for (int il = 0; il < n_layer; ++il) {
        std::fill(owned.begin(), owned.end(), 0);
        int replicated = 0;
        for (int e = 0; e < n_expert; ++e) {
            const int32_t o = owner[(size_t) il * (size_t) n_expert + (size_t) e];
            if (o == kReplicated) { ++replicated; continue; }
            if (o < 0 || o >= n_ranks) {
                set_err(err, "owner rank out of range at layer " + std::to_string(il) +
                             " expert " + std::to_string(e));
                return false;
            }
            ++owned[(size_t) o];
        }
        // When there are at least n_ranks non-replicated experts, every rank
        // must own at least one: a rank with no owned experts would evaluate
        // only replicated routes, which is never what the operator wanted.
        if (n_expert - replicated >= n_ranks && !allow_empty_shards) {
            for (int r = 0; r < n_ranks; ++r) {
                if (owned[(size_t) r] == 0) {
                    set_err(err, "rank " + std::to_string(r) + " owns no expert in layer " +
                                 std::to_string(il));
                    return false;
                }
            }
        }
    }
    return true;
}

bool ClusterExpertPlacement::matches(const common::MoeHybridConfig & cfg) const {
    return n_layer == cfg.n_layer && n_expert == cfg.n_expert &&
           n_expert_used == cfg.n_expert_used;
}

uint64_t ClusterExpertPlacement::hash() const {
    // Field order is part of the wire contract (HelloMsg.placement_hash):
    // n_ranks, n_layer, n_expert, n_expert_used as int32, then the owner
    // table as int32 in layer-major order. Byte order is the host's; all
    // cluster nodes run the same binary on the same ISA.
    const int32_t dims[4] = { (int32_t) n_ranks, (int32_t) n_layer,
                              (int32_t) n_expert, (int32_t) n_expert_used };
    uint64_t h = fnv1a64(dims, sizeof(dims));
    if (!owner.empty()) {
        h = fnv1a64(owner.data(), owner.size() * sizeof(int32_t), h);
    }
    return h;
}

bool ClusterExpertPlacement::to_rank_placement(int rank, common::MoeHybridPlacement & out,
                                               std::string * err) const {
    if (!valid(err)) return false;
    if (rank < 0 || rank >= n_ranks) {
        set_err(err, "rank out of range");
        return false;
    }
    out = common::MoeHybridPlacement{};
    out.n_layer = n_layer;
    out.n_expert = n_expert;
    out.n_expert_used = n_expert_used;
    out.hot_counts.assign((size_t) n_layer, 0);
    out.hot_expert_ids.assign((size_t) n_layer, {});
    int total = 0;
    for (int il = 0; il < n_layer; ++il) {
        out.hot_expert_ids[(size_t) il] = resident_experts(il, rank);
        out.hot_counts[(size_t) il] = (int) out.hot_expert_ids[(size_t) il].size();
        total += out.hot_counts[(size_t) il];
    }
    out.total_hot = total;
    return out.valid(err);
}

void ClusterExpertPlacement::slot_validity(int rank, int n_slots, std::vector<uint8_t> & out) const {
    out.clear();
    if (n_slots <= 0 || n_layer <= 0 || n_expert <= 0) return;
    out.assign((size_t) n_layer * (size_t) n_slots * (size_t) n_expert, 0);
    for (int il = 0; il < n_layer; ++il) {
        for (int e = 0; e < n_expert; ++e) {
            const int32_t o = owner_of(il, e);
            if (o != rank && o != kReplicated) continue;
            for (int t = 0; t < n_slots; ++t) {
                const bool mine = (o == rank) || (slot_owner(il, e, t) == rank);
                out[((size_t) il * (size_t) n_slots + (size_t) t) * (size_t) n_expert + (size_t) e] =
                    mine ? 1 : 0;
            }
        }
    }
}

// ─── Builders ───────────────────────────────────────────────────────────

bool ClusterExpertPlacement::build_uniform(int n_ranks,
                                           const common::MoeHybridConfig & cfg,
                                           int replicate_hot,
                                           const common::MoeHybridRoutingStats * stats,
                                           ClusterExpertPlacement & out,
                                           std::string * err) {
    if (!dims_ok(n_ranks, cfg.n_layer, cfg.n_expert, cfg.n_expert_used, err)) return false;
    if (replicate_hot < 0) {
        set_err(err, "replicate_hot must be >= 0");
        return false;
    }
    if (stats && replicate_hot > 0 && !stats->matches(cfg)) {
        set_err(err, "routing stats do not match the model dimensions");
        return false;
    }
    const int effective_replicate = (stats && replicate_hot > 0) ? replicate_hot : 0;
    if (effective_replicate > cfg.n_expert - n_ranks) {
        set_err(err, "replicate_hot leaves fewer experts than ranks");
        return false;
    }

    out = ClusterExpertPlacement{};
    out.n_ranks = n_ranks;
    out.n_layer = cfg.n_layer;
    out.n_expert = cfg.n_expert;
    out.n_expert_used = cfg.n_expert_used;
    out.replicate_hot = effective_replicate;
    out.owner.assign((size_t) cfg.n_layer * (size_t) cfg.n_expert, 0);

    for (int il = 0; il < cfg.n_layer; ++il) {
        int32_t * row = out.owner.data() + (size_t) il * (size_t) cfg.n_expert;
        for (int e = 0; e < cfg.n_expert; ++e) row[e] = (int32_t) (e % n_ranks);
        if (effective_replicate > 0) {
            const std::vector<int> ranked = stats->ranked_experts(il);
            for (int i = 0; i < effective_replicate && i < (int) ranked.size(); ++i) {
                row[ranked[(size_t) i]] = kReplicated;
            }
        }
    }
    // Replicating the hottest experts can in principle strip a rank of every
    // owned expert in a layer (e.g. n_expert == n_ranks + k); valid() catches
    // that so the operator lowers replicate_hot instead of running lopsided.
    return out.valid(err);
}

bool ClusterExpertPlacement::build_balanced(int n_ranks,
                                            const common::MoeHybridRoutingStats & stats,
                                            int replicate_hot,
                                            ClusterExpertPlacement & out,
                                            std::string * err) {
    if (!stats.valid(err)) return false;
    if (!dims_ok(n_ranks, stats.n_layer, stats.n_expert, stats.n_expert_used, err)) return false;
    if (replicate_hot < 0) {
        set_err(err, "replicate_hot must be >= 0");
        return false;
    }
    if (replicate_hot > stats.n_expert - n_ranks) {
        set_err(err, "replicate_hot leaves fewer experts than ranks");
        return false;
    }

    out = ClusterExpertPlacement{};
    out.n_ranks = n_ranks;
    out.n_layer = stats.n_layer;
    out.n_expert = stats.n_expert;
    out.n_expert_used = stats.n_expert_used;
    out.replicate_hot = replicate_hot;
    out.owner.assign((size_t) stats.n_layer * (size_t) stats.n_expert, 0);

    std::vector<uint64_t> load((size_t) n_ranks);
    std::vector<int> owned((size_t) n_ranks);
    for (int il = 0; il < stats.n_layer; ++il) {
        int32_t * row = out.owner.data() + (size_t) il * (size_t) stats.n_expert;
        // ranked_experts is count-descending with the expert id as tie-break,
        // so identical inputs produce identical placements on every rank.
        const std::vector<int> ranked = stats.ranked_experts(il);
        if ((int) ranked.size() != stats.n_expert) {
            set_err(err, "routing stats ranking failed at layer " + std::to_string(il));
            return false;
        }
        for (int i = 0; i < replicate_hot; ++i) row[ranked[(size_t) i]] = kReplicated;

        const int remaining = stats.n_expert - replicate_hot;
        const int cap = (remaining + n_ranks - 1) / n_ranks;  // ceil(remaining / n_ranks)
        std::fill(load.begin(), load.end(), 0);
        std::fill(owned.begin(), owned.end(), 0);
        for (int i = replicate_hot; i < stats.n_expert; ++i) {
            const int e = ranked[(size_t) i];
            int best = -1;
            for (int r = 0; r < n_ranks; ++r) {
                if (owned[(size_t) r] >= cap) continue;
                if (best < 0 || load[(size_t) r] < load[(size_t) best]) best = r;
            }
            if (best < 0) {
                set_err(err, "balanced placement: no rank below the memory cap");
                return false;
            }
            row[e] = (int32_t) best;
            load[(size_t) best] += stats.count(il, e);
            ++owned[(size_t) best];
        }
    }
    return out.valid(err);
}

// ─── JSON ───────────────────────────────────────────────────────────────

bool ClusterExpertPlacement::save_json(const std::string & path, std::string * err) const {
    if (!valid(err)) return false;
    std::ofstream f(path);
    if (!f) {
        set_err(err, "failed to open placement file for writing: " + path);
        return false;
    }
    f << "{\n"
      << "  \"n_ranks\": " << n_ranks << ",\n"
      << "  \"n_layer\": " << n_layer << ",\n"
      << "  \"n_expert\": " << n_expert << ",\n"
      << "  \"n_expert_used\": " << n_expert_used << ",\n"
      << "  \"replicate_hot\": " << replicate_hot << ",\n"
      << "  \"owner\": [\n";
    for (int il = 0; il < n_layer; ++il) {
        f << "    [";
        const int32_t * row = owner.data() + (size_t) il * (size_t) n_expert;
        for (int e = 0; e < n_expert; ++e) {
            if (e) f << ',';
            f << row[e];
        }
        f << (il + 1 < n_layer ? "],\n" : "]\n");
    }
    f << "  ]\n}\n";
    if (!f) {
        set_err(err, "failed to write placement file: " + path);
        return false;
    }
    return true;
}

bool ClusterExpertPlacement::load_json(const std::string & path, ClusterExpertPlacement & out,
                                       std::string * err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        set_err(err, "failed to open placement file: " + path);
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();

    ClusterExpertPlacement parsed;
    JsonCursor cursor(text);
    if (!cursor.parse_object(parsed, err)) return false;
    if (parsed.owner.size() != (size_t) parsed.n_layer * (size_t) parsed.n_expert) {
        set_err(err, "placement json: owner table does not match n_layer x n_expert");
        return false;
    }
    // Files are the diagnostic entry point: tolerate empty shards, but say so.
    parsed.allow_empty_shards = true;
    if (!parsed.valid(err)) return false;
    for (int r = 0; r < parsed.n_ranks; ++r) {
        for (int il = 0; il < parsed.n_layer; ++il) {
            if (parsed.owned_count(il, r) == 0) {
                std::fprintf(stderr,
                             "[cluster-placement] warning: %s leaves rank %d with no owned "
                             "expert in layer %d (diagnostic placement; every route of that "
                             "layer is evaluated elsewhere)\n",
                             path.c_str(), r, il);
                break;
            }
        }
    }
    out = std::move(parsed);
    return true;
}

// ─── Log summary ────────────────────────────────────────────────────────

std::string ClusterExpertPlacement::describe(const common::MoeHybridRoutingStats * stats) const {
    std::ostringstream os;
    os << "placement ranks=" << n_ranks << " layers=" << n_layer
       << " experts=" << n_expert << " top_k=" << n_expert_used
       << " replicate_hot=" << replicate_hot << " hash=0x" << std::hex << hash() << std::dec;
    int replicated_total = 0;
    for (size_t i = 0; i < owner.size(); ++i) {
        if (owner[i] == kReplicated) ++replicated_total;
    }
    os << " replicated_total=" << replicated_total << " owned_per_rank=[";
    for (int r = 0; r < n_ranks; ++r) {
        if (r) os << ',';
        os << total_owned(r);
    }
    os << "]";

    if (stats && stats->matches(n_layer, n_expert, n_expert_used)) {
        // Expected per-rank activation share: owned counts plus an equal
        // 1/N share of every replicated expert's activations. The max/mean
        // ratio over all layers is the load imbalance the all-reduce waits on.
        double worst = 0.0;
        double sum_ratio = 0.0;
        int layers_counted = 0;
        std::vector<double> load((size_t) n_ranks);
        for (int il = 0; il < n_layer; ++il) {
            std::fill(load.begin(), load.end(), 0.0);
            double total = 0.0;
            for (int e = 0; e < n_expert; ++e) {
                const double c = (double) stats->count(il, e);
                total += c;
                const int32_t o = owner_of(il, e);
                if (o == kReplicated) {
                    for (int r = 0; r < n_ranks; ++r) load[(size_t) r] += c / (double) n_ranks;
                } else if (o >= 0) {
                    load[(size_t) o] += c;
                }
            }
            if (total <= 0.0) continue;
            const double mean = total / (double) n_ranks;
            const double mx = *std::max_element(load.begin(), load.end());
            const double ratio = mean > 0.0 ? mx / mean : 1.0;
            worst = std::max(worst, ratio);
            sum_ratio += ratio;
            ++layers_counted;
        }
        if (layers_counted > 0) {
            char buf[96];
            std::snprintf(buf, sizeof(buf), " imbalance(max/mean) worst=%.3f avg=%.3f",
                          worst, sum_ratio / (double) layers_counted);
            os << buf;
        }
    }
    return os.str();
}

}  // namespace luce::cluster
