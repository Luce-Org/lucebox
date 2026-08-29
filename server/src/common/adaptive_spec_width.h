#pragma once

// Shared feedback controller for chain speculative decoding.
//
// A rejection reveals the exact useful draft depth, so back off with an EMA.
// An all-accepted draft is censored: it only proves that the useful depth was
// at least the offered width.  Averaging that lower bound would strand the
// controller at a narrow width, so clean drafts probe upward additively.
//
// Widths are seed-inclusive throughout this API.  For example, width 4 means
// one always-committed seed plus three speculative candidates.
//
// Opt in with DFLASH_ADAPTIVE_SPEC_WIDTH=1. Fixed width remains the production
// default because narrower verification is not automatically cheaper on every
// backend. Backend-specific fixed-width overrides still take precedence.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace dflash::common {

inline bool adaptive_spec_width_globally_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("DFLASH_ADAPTIVE_SPEC_WIDTH");
        return value != nullptr && value[0] != '\0' &&
               std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

class AdaptiveSpecWidth {
public:
    AdaptiveSpecWidth(int max_width, int min_width = 2, bool enabled = true,
                      float initial_accepted_candidates = 2.0f,
                      float backoff_alpha = 0.25f,
                      float full_accept_probe = 1.0f)
        : max_width_(std::max(1, max_width)),
          min_width_(std::clamp(min_width, 1, std::max(1, max_width))),
          enabled_(enabled),
          initial_accepted_candidates_(std::clamp(
              initial_accepted_candidates, 0.0f,
              static_cast<float>(std::max(0, max_width_ - 1)))),
          backoff_alpha_(std::clamp(backoff_alpha, 0.0f, 1.0f)),
          full_accept_probe_(std::max(0.0f, full_accept_probe)),
          accepted_candidates_ema_(initial_accepted_candidates_) {}

    // Apply both this feedback cap and an optional model-specific cap.
    int next_width(int proposed_width) const {
        const int proposed = std::clamp(proposed_width, 1, max_width_);
        if (!enabled_ || max_width_ <= 1) return proposed;

        const int feedback_width = std::clamp(
            static_cast<int>(std::lround(accepted_candidates_ema_)) + 1,
            min_width_, max_width_);
        return std::min(proposed, feedback_width);
    }

    int next_width() const { return next_width(max_width_); }

    // accepted_width and offered_width both include the seed row.
    void observe(int accepted_width, int offered_width) {
        if (!enabled_ || offered_width <= 1 || max_width_ <= 1) return;

        const int offered = std::clamp(offered_width, 1, max_width_);
        const int accepted = std::clamp(accepted_width, 1, offered);
        const int offered_candidates = offered - 1;
        const int accepted_candidates = accepted - 1;

        if (accepted_candidates >= offered_candidates) {
            // Censored lower bound: do not average it downward; probe wider.
            accepted_candidates_ema_ = std::min(
                static_cast<float>(max_width_ - 1),
                accepted_candidates_ema_ + full_accept_probe_);
        } else {
            // The first rejection exposes the exact accepted prefix length.
            accepted_candidates_ema_ =
                (1.0f - backoff_alpha_) * accepted_candidates_ema_ +
                backoff_alpha_ * static_cast<float>(accepted_candidates);
        }
    }

    void reset() { accepted_candidates_ema_ = initial_accepted_candidates_; }

    bool enabled() const { return enabled_; }
    float accepted_candidates_ema() const { return accepted_candidates_ema_; }
    int min_width() const { return min_width_; }
    int max_width() const { return max_width_; }

private:
    int max_width_;
    int min_width_;
    bool enabled_;
    float initial_accepted_candidates_;
    float backoff_alpha_;
    float full_accept_probe_;
    float accepted_candidates_ema_;
};

} // namespace dflash::common
