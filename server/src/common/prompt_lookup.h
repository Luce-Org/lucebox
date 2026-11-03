#pragma once

// Prompt lookup drafts, and the per-round choice between them and a draft model's.
//
// When the output quotes its input (edits, refactors, repeated code), the tokens that followed an earlier copy of the
// current suffix are usually the next ones, and one verify can accept many of them. A long lookup window costs more
// to verify than a draft model's usual few rows, so it pays only when enough of it is accepted. Each round therefore
// compares expected committed tokens per millisecond:
//
//   draft model  E = the mean tokens its rounds of that width committed, at that width's measured round time
//   lookup       E(k) = 1 + q + q^2 + ... + q^k, q = the acceptance of lookup drafts whose match was about as long,
//                at the measured round time of a lookup window of k + 1 rows
//
// and verifies the lookup drafts only when their best E per ms beats the draft model's by a margin. It only chooses
// which drafts are verified: the output is the target's own samples either way.
//
// The rule, its priors and its constants follow Strata's draft policy (github.com/Niko1221/Strata, MIT,
// include/strata/spec/draft_policy.hpp). Pure C++: the unit test runs on the CPU.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace luce::common {

// The tokens that followed the longest (most recent on ties) earlier occurrence of the sequence's current suffix.
// Every trigram maps to its kWays most recent end positions; a proposal extends each candidate backwards up to
// kMaxMatch tokens. A match shorter than kMinMatch proposes nothing.
class PromptLookup {
public:
    static constexpr int kWays = 4;
    static constexpr int kMinMatch = 3;
    static constexpr int kMaxMatch = 32;

    void reset() {
        hist_.clear();
        index_.clear();
    }

    void append(const int32_t * tokens, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            hist_.push_back(tokens[i]);
            const size_t end = hist_.size() - 1;
            if (end < 2) continue;
            auto & ways = index_.try_emplace(trigram_key(end), empty_ways()).first->second;
            std::copy_backward(ways.begin(), ways.end() - 1, ways.end());
            ways[0] = (int32_t) end;
        }
    }

    // Up to max_k proposed next tokens into out; returns how many (0: no match of kMinMatch). `match` receives the
    // length of the match behind the proposal.
    int propose(int max_k, int32_t * out, int & match) const {
        match = 0;
        const size_t n = hist_.size();
        if (max_k <= 0 || n < (size_t) kMinMatch + 1) return 0;
        const auto it = index_.find(trigram_key(n - 1));
        if (it == index_.end()) return 0;
        int best_len = 0;
        int32_t best_end = -1;
        for (const int32_t e : it->second) {
            if (e < 0 || (size_t) e + 1 >= n) continue;   // an empty way, or the suffix itself
            // Compared from the last token on, so a key collision matches 0 tokens.
            int len = 0;
            while (len < kMaxMatch && e - len >= 0 && hist_[(size_t) (e - len)] == hist_[n - 1 - (size_t) len]) ++len;
            if (len > best_len) {   // the ways hold the most recent first: ties keep it
                best_len = len;
                best_end = e;
            }
        }
        if (best_len < kMinMatch) return 0;
        int k = 0;
        for (size_t p = (size_t) best_end + 1; p < n && k < max_k; ++p) out[k++] = hist_[p];
        match = best_len;
        return k;
    }

    size_t size() const { return hist_.size(); }

private:
    static std::array<int32_t, kWays> empty_ways() {
        std::array<int32_t, kWays> ways;
        ways.fill(-1);
        return ways;
    }

    uint64_t trigram_key(size_t end) const {
        uint64_t h = 0xcbf29ce484222325ull;
        for (size_t i = end - 2; i <= end; ++i) h = (h ^ (uint32_t) hist_[i]) * 0x100000001b3ull;
        return h;
    }

    std::vector<int32_t> hist_;
    std::unordered_map<uint64_t, std::array<int32_t, kWays>> index_;
};

// The per-round choice. Widths count the seed row (width = 1 + drafts). Costs are whole round times in ms; a sample
// < 0 (a round that built a graph) teaches acceptance only. Draft-model and lookup rounds keep separate cost tables:
// a lookup round runs no draft model, so it is cheaper than a draft-model round of its width.
class PromptLookupPolicy {
public:
    static constexpr int kMaxWidth = 16;
    static constexpr int kBuckets = 4;

    struct Pick {
        bool lookup = false;
        int width = 1;
    };

    explicit PromptLookupPolicy(int max_width, double margin = 0.03)
        : max_width_(std::clamp(max_width, 1, kMaxWidth)), margin_(margin) {}

    static int bucket(int match) { return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3; }

    // The acceptance of one more lookup draft given that the previous one held, for a match of this length.
    double lookup_rate(int match) const {
        const size_t b = (size_t) bucket(match);
        return (ok_[b] + kPriorN * kPriorQ[b]) / (ok_[b] + bad_[b] + kPriorN);
    }

    // The mean tokens a draft-model round of this width commits (a prior until one is observed).
    double model_tokens(int width) const {
        const size_t w = (size_t) std::clamp(width, 1, kMaxWidth);
        return model_n_[w] > 0 ? model_tokens_[w] : 1.0 + 0.7 * (double) (w - 1);
    }

    // Measured round time of a width. A lookup width not measured yet borrows the draft-model round's (an upper
    // bound: it includes the draft), then a linear cost shape scaled from the nearest measured width.
    double cost_ms(bool lookup, int width) const {
        const int t = std::clamp(width, 1, kMaxWidth);
        if (lookup && look_cost_n_[(size_t) t] > 0) return look_cost_[(size_t) t];
        if (model_cost_n_[(size_t) t] > 0) return model_cost_[(size_t) t];
        for (int d = 1; d < kMaxWidth; ++d) {
            for (const int u : {t - d, t + d}) {
                if (u < 1 || u > kMaxWidth) continue;
                const bool own = lookup && look_cost_n_[(size_t) u] > 0;
                if (own) return look_cost_[(size_t) u] * shape(t) / shape(u);
                if (model_cost_n_[(size_t) u] > 0) return model_cost_[(size_t) u] * shape(t) / shape(u);
            }
        }
        return shape(t);
    }

    // model_width: the draft model's width this round; lookup_k: the lookup proposal's length (0: none).
    Pick choose(int model_width, int lookup_k, int match) const {
        Pick pick;
        pick.width = std::clamp(model_width, 1, max_width_);
        if (lookup_k <= 0) return pick;
        const double model_rate = model_tokens(pick.width) / cost_ms(false, pick.width);
        const double q = lookup_rate(match);
        const int k_max = std::min(lookup_k, max_width_ - 1);
        double expected = 1.0, q_pow = 1.0, best_rate = 0.0;
        int best_width = 0;
        for (int k = 1; k <= k_max; ++k) {
            q_pow *= q;
            expected += q_pow;
            const double rate = expected / cost_ms(true, k + 1);
            if (rate > best_rate) {
                best_rate = rate;
                best_width = k + 1;
            }
        }
        if (best_width > 0 && best_rate > model_rate * (1.0 + margin_)) {
            pick.lookup = true;
            pick.width = best_width;
            return pick;
        }
        // A guessed cost could keep a confident lookup from ever measuring its width: try it a few times (the
        // output is the target's either way; only that round's speed is at stake).
        const int full = k_max + 1;
        if (full > pick.width && q >= kProbeRate && look_tried_[(size_t) full] < kProbes) {
            pick.lookup = true;
            pick.width = full;
        }
        return pick;
    }

    // After a round: its width, the drafts accepted, the match behind a lookup window, its time (< 0: untimed).
    void observe(bool lookup, int width, int accepted, int match, double round_ms) {
        const size_t t = (size_t) std::clamp(width, 1, kMaxWidth);
        if (t <= 1) return;
        auto & cost = lookup ? look_cost_ : model_cost_;
        auto & cost_n = lookup ? look_cost_n_ : model_cost_n_;
        if (round_ms > 0.0 && std::isfinite(round_ms)) {
            cost[t] = cost_n[t] > 0 ? (1.0 - kCostAlpha) * cost[t] + kCostAlpha * round_ms : round_ms;
            cost_n[t] += 1.0;
        }
        if (lookup) {
            look_tried_[t] += 1.0;
            const size_t b = (size_t) bucket(match);
            ok_[b] = kDecay * ok_[b] + accepted;
            bad_[b] = kDecay * bad_[b] + (accepted < (int) t - 1 ? 1.0 : 0.0);
        } else {
            const double committed = accepted + 1.0;
            model_tokens_[t] = model_n_[t] > 0 ? (1.0 - kTokensAlpha) * model_tokens_[t] + kTokensAlpha * committed
                                               : committed;
            model_n_[t] += 1.0;
        }
    }

private:
    static constexpr double kCostAlpha = 0.1;     // EMA weight of a new round time
    static constexpr double kTokensAlpha = 0.05;  // EMA weight of a new draft-model round's committed tokens
    static constexpr double kDecay = 0.97;        // lookup acceptance counts: older windows fade
    // Before a bucket has data: the longer the match, the likelier its continuation; worth kPriorN observations.
    static constexpr std::array<double, kBuckets> kPriorQ{0.75, 0.88, 0.93, 0.96};
    static constexpr double kPriorN = 4.0;
    static constexpr double kProbeRate = 0.85;
    static constexpr double kProbes = 3.0;
    // Relative cost of a width before any width is measured.
    static double shape(int width) { return 45.0 + 20.0 * (double) (std::max(1, width) - 1); }

    int max_width_;
    double margin_;
    std::array<double, kMaxWidth + 1> model_cost_{}, model_cost_n_{}, look_cost_{}, look_cost_n_{}, look_tried_{};
    std::array<double, kMaxWidth + 1> model_tokens_{}, model_n_{};
    std::array<double, kBuckets> ok_{}, bad_{};
};

}  // namespace luce::common
