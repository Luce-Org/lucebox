#include "pflash/pflash_query_candidates.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

namespace luce::pflash {

const char * pflash_unit_score_name(PFlashUnitScore mode) noexcept {
    return mode == PFlashUnitScore::Max ? "max" : "mean";
}

const char * pflash_query_aggregate_name(PFlashQueryAggregate mode) noexcept {
    return mode == PFlashQueryAggregate::Softmax ? "softmax" : "wmax";
}

PFlashQueryUnits pflash_query_units(const std::vector<PFlashTokenSpan> & segments,
                                    const std::vector<PFlashRequestMessage> & messages,
                                    const std::vector<float> & s1) {
    PFlashQueryUnits out;
    out.latest = pflash_request_latest_user(messages);
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto & s = segments[i];
        if (s.end <= s.begin || (i > 0 && s.begin != segments[i - 1].end) || (i == 0 && s.begin != 0)) {
            out.error = "segments_invalid";
            return out;
        }
    }
    const auto build = [&](bool system) {
        std::vector<PFlashQueryUnit> units;
        for (size_t m = 0; m < messages.size(); ++m) {
            const auto & msg = messages[m];
            if ((msg.role == PFlashRequestRole::System) != system) continue;
            const PFlashTokenSpan span = msg.span;
            if (span.end <= span.begin) continue;
            if (span.begin < 0 || span.end > (int) s1.size()) {
                out.error = "logits_missing";
                return units;
            }
            // First segment ending after span.begin.
            auto it = std::upper_bound(segments.begin(), segments.end(), span.begin,
                [](int pos, const PFlashTokenSpan & seg) { return pos < seg.end; });
            for (; it != segments.end() && it->begin < span.end; ++it) {
                const int b = std::max(it->begin, span.begin);
                const int e = std::min(it->end, span.end);
                if (e <= b) continue;
                double sum = 0.0;
                int n = 0;
                float mx = -std::numeric_limits<float>::infinity();
                for (int t = b; t < e; ++t) {
                    const float v = s1[(size_t) t];
                    if (!std::isfinite(v)) continue;
                    sum += v;
                    ++n;
                    mx = std::max(mx, v);
                }
                if (n == 0) continue;
                PFlashQueryUnit u;
                u.span = {b, e};
                u.message = (int) m;
                u.latest = (int) m == out.latest;
                u.score_max = mx;
                u.score_mean = (float) (sum / n);
                units.push_back(u);
            }
        }
        std::sort(units.begin(), units.end(), [](const PFlashQueryUnit & a, const PFlashQueryUnit & b) {
            return a.span.begin < b.span.begin;
        });
        return units;
    };
    out.units = build(false);
    if (!out.error.empty()) {
        out.units.clear();
        return out;
    }
    if (out.units.empty()) {
        out.units = build(true);
        out.system_used = !out.units.empty();
        if (!out.error.empty()) out.units.clear();
    }
    return out;
}

std::vector<int> pflash_query_rank(const PFlashQueryUnits & units, const std::vector<int> & which,
                                   PFlashUnitScore mode) {
    std::vector<int> order = which;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const auto & ua = units.units[(size_t) a];
        const auto & ub = units.units[(size_t) b];
        const float sa = ua.score(mode);
        const float sb = ub.score(mode);
        if (sa != sb) return sa > sb;
        return ua.span.begin > ub.span.begin;
    });
    return order;
}

std::vector<int> pflash_query_unit_tokens(PFlashTokenSpan span, const std::vector<float> & s1, int cap) {
    std::vector<int> tokens;
    for (int t = span.begin; t < span.end; ++t) tokens.push_back(t);
    if (cap <= 0 || (int) tokens.size() <= cap) return tokens;
    std::vector<int> ranked = pflash_request_by_score(tokens, s1);
    ranked.resize((size_t) cap);
    std::sort(ranked.begin(), ranked.end());
    return ranked;
}

std::vector<double> pflash_query_softmax(const std::vector<float> & scores, float temperature) {
    std::vector<double> w(scores.size(), 0.0);
    if (scores.empty()) return w;
    double mx = -std::numeric_limits<double>::infinity();
    for (const float s : scores) mx = std::max(mx, (double) s);
    double total = 0.0;
    for (size_t k = 0; k < scores.size(); ++k) {
        w[k] = std::exp(((double) scores[k] - mx) / (double) temperature);
        total += w[k];
    }
    for (auto & v : w) v /= total;
    return w;
}

std::vector<int> pflash_query_extent_units(const PFlashQueryUnits & units, int message, PFlashUnitScore mode,
                                           float margin, bool gap) {
    std::vector<int> mine;
    for (size_t i = 0; i < units.units.size(); ++i) {
        if (units.units[i].message == message) mine.push_back((int) i);
    }
    if (mine.empty()) return mine;
    const auto ranked = pflash_query_rank(units, mine, mode);
    if (gap) {
        if ((int) ranked.size() < kQueryNullMinUnits) return ranked;
        size_t cut = 1;
        float best = -1.0f;
        for (size_t i = 0; i + 1 < ranked.size(); ++i) {
            const float drop = units.units[(size_t) ranked[i]].score(mode) - units.units[(size_t) ranked[i + 1]].score(mode);
            if (drop > best) {          // the first largest drop (ties keep the shorter extent)
                best = drop;
                cut = i + 1;
            }
        }
        return std::vector<int>(ranked.begin(), ranked.begin() + (long) cut);
    }
    const float top = units.units[(size_t) ranked.front()].score(mode);
    std::vector<int> out;
    for (const int u : ranked) {
        if (!(units.units[(size_t) u].score(mode) >= top - margin)) break;   // inf margin: every unit
        out.push_back(u);
    }
    return out;
}

PFlashQueryExtent pflash_query_extent(const PFlashQueryUnits & units, int message, const std::vector<float> & s1,
                                      PFlashUnitScore mode, float margin, int cap, bool gap) {
    PFlashQueryExtent out;
    const auto eligible = pflash_query_extent_units(units, message, mode, margin, gap);
    if (eligible.empty()) return out;
    out.top = units.units[(size_t) eligible.front()].score(mode);
    int total = 0;
    for (const int u : eligible) {
        const auto & unit = units.units[(size_t) u];
        const int len = unit.span.end - unit.span.begin;
        if (cap > 0 && total + len > cap) {
            if (out.units.empty()) {
                // The top unit alone exceeds the cap: its cap best tokens (a query, not a pin).
                out.units.push_back(u);
                out.tokens = pflash_query_unit_tokens(unit.span, s1, cap);
                out.cut = true;
                return out;
            }
            ++out.skipped;
            continue;
        }
        total += len;
        out.units.push_back(u);
        for (int t = unit.span.begin; t < unit.span.end; ++t) out.tokens.push_back(t);
    }
    std::sort(out.tokens.begin(), out.tokens.end());
    return out;
}

PFlashQueryCandidateSet pflash_query_candidate_set(const PFlashQueryUnits & units,
                                                   const std::vector<float> & s1,
                                                   const PFlashQueryConfig & config) {
    PFlashQueryCandidateSet out;
    out.latest = units.latest;
    out.latest_z = std::numeric_limits<float>::quiet_NaN();
    const PFlashUnitScore mode = config.unit_score;
    const int k = std::max(1, config.candidates);
    std::vector<int> latest_units;
    std::vector<int> messages;          // messages with a unit, ascending
    for (size_t i = 0; i < units.units.size(); ++i) {
        const auto & u = units.units[i];
        if (u.latest) latest_units.push_back((int) i);
        if (std::find(messages.begin(), messages.end(), u.message) == messages.end()) messages.push_back(u.message);
    }
    std::sort(messages.begin(), messages.end());
    out.latest_units = (int) latest_units.size();
    bool latest_null = false;
    if (!latest_units.empty()) {
        const auto ranked = pflash_query_rank(units, latest_units, mode);
        out.latest_top = units.units[(size_t) ranked.front()].score(mode);
        if ((int) latest_units.size() >= kQueryNullMinUnits) {
            std::vector<float> sorted;
            for (const int u : latest_units) sorted.push_back(units.units[(size_t) u].score(mode));
            std::sort(sorted.begin(), sorted.end());
            const size_t n = sorted.size();
            const float median = n % 2 ? sorted[n / 2] : 0.5f * (sorted[n / 2 - 1] + sorted[n / 2]);
            std::vector<float> dev;
            for (const float v : sorted) dev.push_back(std::fabs(v - median));
            std::sort(dev.begin(), dev.end());
            const float mad = n % 2 ? dev[n / 2] : 0.5f * (dev[n / 2 - 1] + dev[n / 2]);
            const float spread = std::max(1.4826f * mad, kQuerySpreadFloor);
            out.latest_median = median;
            out.latest_spread = spread;
            out.latest_z = (out.latest_top - median) / spread;
            latest_null = config.null_margin > 0.0f && out.latest_z < config.null_margin;
        }
        out.latest_null = latest_null;
    }
    // Messages by their top unit's score (descending; ties to the later message), the latest first unless NULL.
    struct Top { int message; float score; int begin; };
    std::vector<Top> tops;
    for (const int m : messages) {
        if (m == units.latest) continue;
        float best = -std::numeric_limits<float>::infinity();
        int begin = -1;
        for (const auto & u : units.units) {
            if (u.message != m) continue;
            const float v = u.score(mode);
            if (v > best || (v == best && u.span.begin > begin)) { best = v; begin = u.span.begin; }
        }
        tops.push_back({m, best, begin});
    }
    std::stable_sort(tops.begin(), tops.end(), [](const Top & a, const Top & b) {
        if (a.score != b.score) return a.score > b.score;
        return a.message > b.message;
    });
    std::vector<int> order;
    if (!latest_units.empty() && !latest_null) order.push_back(units.latest);
    for (const auto & t : tops) order.push_back(t.message);
    if (order.empty() && !latest_units.empty()) order.push_back(units.latest);   // NULL and nothing else: never empty
    if ((int) order.size() > k) order.resize((size_t) k);
    std::vector<float> scores;
    for (const int m : order) {
        const auto extent = pflash_query_extent(units, m, s1, mode, config.extent_margin, kQueryExtentQueryCap,
                                                config.extent_gap);
        if (extent.tokens.empty()) continue;
        PFlashQueryCandidate c;
        c.message = m;
        c.units = extent.units;
        // Trim outer negative-logit background margins (leading/trailing doc
        // bleed with P(request) < 0.5) independently on each contiguous token
        // component of extent.tokens via the 1D convex hull of s1 >= 0 runs
        // (anchored on runs of >= 2 consecutive positive tokens to ignore
        // stray single-token edge blips, falling back to >= 1 if a component
        // only has single-token hits). Preserves interior declarative clauses,
        // options, and code blocks across single or adjacent units while
        // keeping disjoint units (e.g. prefix + suffix) independently trimmed
        // and full extent units intact for verbatim pinning.
        std::vector<int> trimmed;
        trimmed.reserve(extent.tokens.size());
        auto is_pos = [&](int t) {
            return t >= 0 && t < (int) s1.size() && std::isfinite(s1[(size_t) t]) && s1[(size_t) t] >= 0.0f;
        };
        size_t comp_b = 0;
        while (comp_b < extent.tokens.size()) {
            size_t comp_e = comp_b + 1;
            while (comp_e < extent.tokens.size() && extent.tokens[comp_e] == extent.tokens[comp_e - 1] + 1) {
                ++comp_e;
            }
            size_t first1 = comp_e, last1 = comp_b;
            size_t first2 = comp_e, last2 = comp_b;
            size_t i = comp_b;
            while (i < comp_e) {
                if (!is_pos(extent.tokens[i])) {
                    ++i;
                    continue;
                }
                size_t j = i + 1;
                while (j < comp_e && is_pos(extent.tokens[j])) ++j;
                if (first1 == comp_e) first1 = i;
                last1 = j;
                if (j - i >= 2) {
                    if (first2 == comp_e) first2 = i;
                    last2 = j;
                }
                i = j;
            }
            const size_t hb = (first2 < comp_e) ? first2 : first1;
            const size_t he = (first2 < comp_e) ? last2 : last1;
            if (hb < comp_e) {
                trimmed.insert(trimmed.end(),
                               extent.tokens.begin() + (ptrdiff_t) hb,
                               extent.tokens.begin() + (ptrdiff_t) he);
            }
            comp_b = comp_e;
        }
        c.tokens = trimmed.empty() ? extent.tokens : std::move(trimmed);
        c.score = extent.top;
        scores.push_back(c.score);
        out.candidates.push_back(std::move(c));
    }
    const auto w = pflash_query_softmax(scores, config.temperature);
    for (size_t i = 0; i < out.candidates.size(); ++i) out.candidates[i].weight = w[i];
    return out;
}

PFlashQueryBlend pflash_query_blend(const PFlashQueryCandidateSet & set, const PFlashQueryConfig & config,
                                    float tw_score_value) {
    PFlashQueryBlend out;
    std::vector<float> scores;
    for (const auto & c : set.candidates) scores.push_back(c.score);
    const bool tw_in_softmax = config.tw_score && std::isfinite(tw_score_value);
    if (tw_in_softmax) scores.push_back(tw_score_value);
    auto w = pflash_query_softmax(scores, config.temperature);
    double rest = 1.0 - (double) config.end_share;
    if (!config.tw_score) rest -= (double) config.tw_share;
    if (tw_in_softmax) {
        out.tw = w.back() * rest;
        w.pop_back();
    } else if (!config.tw_score) {
        out.tw = (double) config.tw_share;
    }
    // TW "score" without a finite score (no content token in its windows): TW stays out of the blend.
    for (const double v : w) out.probe.push_back(v * rest);
    out.end = (double) config.end_share;
    if (out.probe.empty()) {
        // No probe candidate: the remaining weight goes to TW / END (normalised); the caller fails closed before.
        const double total = out.tw + out.end;
        if (total > 0.0) { out.tw /= total; out.end /= total; }
    }
    return out;
}

bool pflash_query_aggregate(const std::vector<std::vector<float>> & masses,
                            const std::vector<double> & weights,
                            PFlashQueryAggregate mode,
                            std::vector<float> & out,
                            std::string & error) {
    error.clear();
    if (masses.empty() || masses.size() != weights.size()) {
        error = "query aggregate: candidate masses and weights differ in count";
        return false;
    }
    const size_t n = masses.front().size();
    for (const auto & m : masses) {
        if (m.size() != n) {
            error = "query aggregate: candidate masses differ in length";
            return false;
        }
    }
    out.assign(n, 0.0f);
    if (mode == PFlashQueryAggregate::Softmax) {
        std::vector<double> acc(n, 0.0);
        for (size_t k = 0; k < masses.size(); ++k) {
            for (size_t t = 0; t < n; ++t) acc[t] += weights[k] * (double) masses[k][t];
        }
        for (size_t t = 0; t < n; ++t) out[t] = (float) acc[t];
        return true;
    }
    std::vector<double> acc(n, 0.0);
    double total = 0.0;
    for (size_t t = 0; t < n; ++t) {
        double best = 0.0;
        for (size_t k = 0; k < masses.size(); ++k) best = std::max(best, weights[k] * (double) masses[k][t]);
        acc[t] = best;
        total += best;
    }
    for (size_t t = 0; t < n; ++t) out[t] = total > 0.0 ? (float) (acc[t] / total) : 0.0f;
    return true;
}

PFlashQueryPin pflash_query_pin(const PFlashQueryUnits & units, int cap, PFlashUnitScore mode, float margin,
                                bool gap) {
    PFlashQueryPin pin;
    if (cap <= 0 || units.latest < 0) return pin;
    std::vector<PFlashTokenSpan> spans;
    for (const int u : pflash_query_extent_units(units, units.latest, mode, margin, gap)) {
        const auto & unit = units.units[(size_t) u];
        const int len = unit.span.end - unit.span.begin;
        if (pin.tokens + len > cap) {
            ++pin.skipped;
            continue;
        }
        pin.tokens += len;
        ++pin.units;
        spans.push_back(unit.span);
    }
    std::sort(spans.begin(), spans.end(), [](const PFlashTokenSpan & a, const PFlashTokenSpan & b) {
        return a.begin < b.begin;
    });
    for (const auto & sp : spans) {
        if (!pin.spans.empty() && sp.begin <= pin.spans.back().end) {
            pin.spans.back().end = std::max(pin.spans.back().end, sp.end);
        } else {
            pin.spans.push_back(sp);
        }
    }
    return pin;
}

} // namespace luce::pflash
