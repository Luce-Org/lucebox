#pragma once

// Shared feedback controller for chain speculative decoding.
//
// The controller is a small pure unit: construct it with the width bounds,
// feed it one observe() per verify step, and ask it for the next width. It
// keeps no globals, and its only environment read is the one-time opt-in
// below. Two policies share the same state.
//
// 1. next_width(): acceptance only. A rejection reveals the exact useful
//    draft depth, so back off with an EMA (backoff_alpha). An all-accepted
//    draft is censored: it only proves that the useful depth was at least
//    the offered width. Averaging that lower bound would strand the
//    controller at a narrow width, so clean drafts probe upward additively
//    (full_accept_probe candidates per clean draft).
//
// 2. next_width_cost_aware(): acceptance and cost. For every width from two
//    up to the proposal it computes the expected committed tokens (the
//    target bonus plus the prefix-survival probability of each candidate)
//    divided by the total step cost of that width, and returns the argmax.
//    Prefix survival comes from the per-candidate conditional acceptance
//    vector passed by the caller (a drafter confidence head scores this very
//    step) or, when that is empty, from the per-depth survival EMAs learned
//    in observe() (kSurvivalPrior before any evidence, kSurvivalAlpha per
//    sample). A vector shorter than the proposal is extended with the
//    learned conditional acceptance of the uncovered depths, so a head
//    calibrated for fewer candidates than the verifier width still decides
//    the depths it knows and the target feedback decides the rest. After a clean
//    draft the next wider width is taken when it is within kExploreMargin of
//    the best, so a near tie does not freeze the width below the cap. Head scores
//    are calibrated online: observe_confidence() tracks predicted against
//    observed acceptance per depth and each score is scaled by their ratio,
//    so an optimistic head stops buying widths the target does not pay for.
//    Step costs are seeded by set_relative_costs()
//    and refined online from the observed cost of the offered width; the
//    first kCostWarmupSamples observations of a seeded width are ignored so a
//    cold shape-specific graph build cannot poison a calibrated seed. A width
//    that is not offered keeps its last cost estimate.
//
//    Survival at depths beyond the offered width is not frozen: a rejection
//    at depth d is a real zero sample for every deeper prefix, and a clean
//    draft extrapolates the next depth geometrically from the two deepest
//    observed survivals. So a narrow width never becomes absorbing, yet low
//    per-candidate acceptance keeps it narrow; when acceptance recovers the
//    controller re-widens one width per confirming step.
//
//    The policy widens when the learned survival at the deeper depths makes
//    the extra candidates worth their cost, and narrows as rejections pull
//    those survival estimates down. A full rejection (only the seed survives)
//    is a zero sample for every offered depth and the strongest narrowing
//    signal. The optional max-width guard (set_max_width_guard) additionally
//    requires a streak of clean drafts before the widest shape is eligible
//    and cools it down after a rejection at that width, for verifiers whose
//    widest shape carries a structural rejection penalty.
//
// Widths are seed-inclusive throughout this API.  For example, width 4 means
// one always-committed seed plus three speculative candidates. The proposal
// passed to next_width*() is a hard cap: the controller only narrows a
// proposal, so a proposal below min_width is returned unchanged.
//
// Opt in with DFLASH_ADAPTIVE_SPEC_WIDTH=1. Fixed width remains the shared
// default because narrower verification is not automatically cheaper on every
// backend; a backend may enable the controller by default on a qualified
// device. Backend-specific fixed-width overrides still take precedence.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

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
    // Prefix-survival estimate for every depth before any evidence.
    static constexpr float kSurvivalPrior = 0.75f;
    // EMA rate of the per-depth prefix-survival estimates.
    static constexpr float kSurvivalAlpha = 0.20f;
    // EMA rate of the per-width step-cost estimates.
    static constexpr float kCostAlpha = 0.20f;
    // Observations of a seeded width ignored before its cost starts tracking.
    static constexpr int kCostWarmupSamples = 4;
    // Below this prefix survival a depth has no usable support: the ratio of
    // two near-zero estimates is noise, so the learned conditional acceptance
    // of the next depth is taken against this floor instead.
    static constexpr float kSurvivalFloor = 0.05f;
    // EMA rate of the per-depth confidence calibration (predicted against
    // observed acceptance of offered candidates).
    static constexpr float kCalibrationAlpha = 0.10f;
    // Bounds on the per-depth correction applied to a confidence score.
    static constexpr float kCalibrationMinScale = 0.25f;
    static constexpr float kCalibrationMaxScale = 1.50f;
    // After a clean draft the next wider width is taken when its expected
    // value is within this fraction of the best: a near tie is worth one
    // step of exploration, since a width that is never offered never gets
    // its survival and calibration measured.
    static constexpr float kExploreMargin = 0.03f;

    // max_width is the hard cap and min_width the floor the feedback policies
    // narrow to; a floor above the cap is clamped to the cap.
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
          accepted_candidates_ema_(initial_accepted_candidates_),
          prefix_survival_ema_((size_t) std::max(1, max_width_),
                               kSurvivalPrior),
          width_cost_ema_((size_t) std::max(1, max_width_) + 1,
                          std::numeric_limits<float>::quiet_NaN()),
          width_cost_samples_((size_t) std::max(1, max_width_) + 1, 0),
          utility_scratch_((size_t) std::max(1, max_width_) + 1, 0.0f),
          confidence_pred_ema_((size_t) std::max(1, max_width_), kSurvivalPrior),
          confidence_actual_ema_((size_t) std::max(1, max_width_), kSurvivalPrior) {}

    // Apply both this feedback cap and an optional model-specific cap.
    int next_width(int proposed_width) const {
        int proposed = std::clamp(proposed_width, 1, max_width_);
        if (!enabled_ || max_width_ <= 1) return proposed;
        if (!max_width_is_eligible() && proposed == max_width_) {
            proposed = std::max(min_width_, max_width_ - 1);
        }

        const int feedback_width = std::clamp(
            static_cast<int>(std::lround(accepted_candidates_ema_)) + 1,
            min_width_, max_width_);
        return std::min(proposed, feedback_width);
    }

    int next_width() const { return next_width(max_width_); }

    // Configure total step costs indexed by seed-inclusive width. Values need
    // only be relative to one another; entries that are not finite and
    // positive leave the width unseeded. A backend can seed this from a short
    // calibration and then refine it online through observe().
    void set_relative_costs(const std::vector<float> & costs) {
        const size_t n = std::min(costs.size(), width_cost_ema_.size());
        for (size_t width = 0; width < n; ++width) {
            const float cost = costs[width];
            if (std::isfinite(cost) && cost > 0.0f) {
                width_cost_ema_[width] = cost;
            }
        }
    }

    // Some verifier widths have a structural rejection penalty (for example,
    // crossing an extra recurrent-state boundary).  Require a run of clean
    // drafts before selecting the widest shape, then cool it down immediately
    // after a rejection.  This policy is optional and backend-neutral.
    void set_max_width_guard(int min_full_accept_streak,
                             int rejection_cooldown_steps,
                             bool probe_initially = false) {
        max_width_probe_streak_ = std::max(0, min_full_accept_streak);
        max_width_rejection_cooldown_ =
            std::max(0, rejection_cooldown_steps);
        max_width_initially_active_ = probe_initially;
        max_width_active_ = probe_initially;
        full_accept_streak_ = 0;
        max_width_cooldown_remaining_ = 0;
    }

    // Select the width that maximizes expected committed tokens per unit
    // cost. conditional_acceptance contains P(candidate i is accepted |
    // candidates 0..i-1 were accepted). With no model confidence, use the
    // target-observed prefix survival estimates learned by observe(); a
    // shorter vector is extended with the learned conditional acceptance of
    // the depths it does not cover. Widths without a usable cost estimate
    // are skipped; when none is usable the acceptance-only policy decides.
    int next_width_cost_aware(
            const std::vector<float> & conditional_acceptance,
            int proposed_width) const {
        int proposed = std::clamp(proposed_width, 1, max_width_);
        if (!enabled_ || proposed <= 1) return proposed;
        if (!max_width_is_eligible() && proposed == max_width_) {
            proposed = std::max(min_width_, max_width_ - 1);
        }

        int best_width = -1;
        float best_utility = -1.0f;
        float survival = 1.0f;
        float expected_commits = 1.0f; // target bonus after the seed
        std::vector<float> & utilities = utility_scratch_;
        std::fill(utilities.begin(), utilities.end(), 0.0f);
        for (int width = 2; width <= proposed; ++width) {
            const int depth = width - 1;
            if ((size_t) depth <= conditional_acceptance.size()) {
                // The drafter's own per-candidate score for this step,
                // scaled by how the target has answered this depth's
                // scores so far in the request.
                survival *= calibrated_confidence(
                    depth, conditional_acceptance[(size_t) depth - 1]);
            } else if (conditional_acceptance.empty()) {
                survival = prefix_survival_ema_[(size_t) depth];
            } else {
                // The head covers fewer depths than the proposal: continue
                // with the learned conditional acceptance of this depth, the
                // ratio of consecutive prefix-survival estimates.
                survival *= learned_conditional_acceptance(depth);
            }
            expected_commits += survival;
            if (width < min_width_) continue;
            const float cost = width_cost_ema_[(size_t) width];
            if (!std::isfinite(cost) || cost <= 0.0f) continue;
            const float utility = expected_commits / cost;
            utilities[(size_t) width] = utility;
            if (utility > best_utility) {
                best_utility = utility;
                best_width = width;
            }
        }
        if (best_width >= 2 && last_draft_clean_ && best_width < proposed &&
            utilities[(size_t) best_width + 1] >=
                best_utility * (1.0f - kExploreMargin)) {
            best_width += 1;
        }
        return best_width >= 0 ? best_width : next_width(proposed);
    }

    int next_width_cost_aware(
            const std::vector<float> & conditional_acceptance) const {
        return next_width_cost_aware(conditional_acceptance, max_width_);
    }

    // accepted_width and offered_width both include the seed row.
    // observed_step_cost is the total cost of this step at offered_width in
    // the same units as set_relative_costs(); values that are not finite and
    // positive are ignored.
    void observe(int accepted_width, int offered_width,
                 float observed_step_cost = -1.0f) {
        // A one-row step carries no draft; it must not leave a stale clean
        // flag from the previous step behind.
        if (offered_width <= 1) last_draft_clean_ = false;
        if (!enabled_ || offered_width <= 1 || max_width_ <= 1) return;

        const int offered = std::clamp(offered_width, 1, max_width_);
        const int accepted = std::clamp(accepted_width, 1, offered);
        const int offered_candidates = offered - 1;
        const int accepted_candidates = accepted - 1;

        if (max_width_guard_enabled()) {
            if (max_width_cooldown_remaining_ > 0) {
                --max_width_cooldown_remaining_;
            }
            if (offered == max_width_ && accepted < offered) {
                max_width_active_ = false;
                full_accept_streak_ = 0;
                max_width_cooldown_remaining_ =
                    max_width_rejection_cooldown_;
            } else if (offered == max_width_) {
                max_width_active_ = true;
                full_accept_streak_ = max_width_probe_streak_;
            } else if (accepted == offered) {
                full_accept_streak_ = std::min(
                    full_accept_streak_ + 1, max_width_probe_streak_);
            } else {
                full_accept_streak_ = 0;
            }
        }

        // Prefix-survival samples. Every offered depth is observed directly.
        // Deeper depths are not offered, but they are not all unknown:
        //  - a rejection at depth d means every deeper prefix died with it,
        //    so depths beyond the offered width take a real zero sample;
        //  - a clean draft is censored evidence, and the one depth past the
        //    offered width takes a geometric extrapolation as its sample:
        //    the next candidate is assumed as likely as the last observed
        //    one (s[d+1] ~ s[d] * s[d] / s[d-1]). That keeps a narrow width
        //    from becoming absorbing without inventing acceptance: at 0.6
        //    per candidate the estimate settles near 0.36 and q2 stays, at
        //    0.95 it climbs until the wider shape pays, and the first wider
        //    step replaces the extrapolation with a direct observation.
        //    Depths further out keep their estimates.
        const bool clean = accepted_candidates >= offered_candidates;
        float extrapolated = 1.0f;
        if (offered_candidates >= 1) {
            const float deepest =
                prefix_survival_ema_[(size_t) offered_candidates];
            const float shallower = offered_candidates >= 2
                ? prefix_survival_ema_[(size_t) offered_candidates - 1]
                : 1.0f;
            extrapolated = shallower > 0.0f
                ? std::clamp(deepest * deepest / shallower, 0.0f, deepest)
                : 0.0f;
        }
        for (int depth = 1; depth < max_width_; ++depth) {
            float sample;
            if (depth <= offered_candidates) {
                sample = accepted_candidates >= depth ? 1.0f : 0.0f;
            } else if (!clean) {
                sample = 0.0f;
            } else if (depth == offered_candidates + 1) {
                sample = extrapolated;
            } else {
                continue;
            }
            float & estimate = prefix_survival_ema_[(size_t) depth];
            estimate = (1.0f - kSurvivalAlpha) * estimate +
                       kSurvivalAlpha * sample;
        }
        if (std::isfinite(observed_step_cost) && observed_step_cost > 0.0f) {
            float & estimate = width_cost_ema_[(size_t) offered];
            int & samples = width_cost_samples_[(size_t) offered];
            if (!std::isfinite(estimate) || estimate <= 0.0f) {
                // No calibrated seed for this width: adopt the first
                // observation and track from the next one on.
                estimate = observed_step_cost;
                samples = kCostWarmupSamples;
            } else if (++samples > kCostWarmupSamples) {
                // Shape-specific graph construction makes the first few
                // observations of a seeded width cold outliers. Preserve the
                // calibrated seed through warmup, then track the live cost.
                estimate = (1.0f - kCostAlpha) * estimate +
                           kCostAlpha * observed_step_cost;
            }
        }

        last_draft_clean_ = accepted_candidates >= offered_candidates;
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

    // Calibrate the drafter's confidence head against the target: for every
    // verified candidate depth (offered, and every shallower candidate
    // accepted), track the predicted score and the observed acceptance. next_width_cost_aware() scales each depth's score by the
    // ratio of the two, so a head that is optimistic on this text (a common
    // pattern at the deeper depths) stops buying widths the target does not
    // pay for, and a pessimistic head does not hold the width down.
    // predicted[i] is the score of candidate i+1; accepted_width and
    // offered_width include the seed.
    void observe_confidence(const std::vector<float> & predicted,
                            int accepted_width, int offered_width) {
        if (!enabled_ || offered_width <= 1) return;
        const int offered = std::clamp(offered_width, 1, max_width_);
        const int accepted = std::clamp(accepted_width, 1, offered);
        for (int depth = 1; depth < offered; ++depth) {
            if ((size_t) depth > predicted.size()) break;
            // Candidate `depth` was verified only if every shallower one
            // was accepted; after a rejection the deeper scores are
            // conditional predictions with no outcome to score against.
            if (accepted < depth) break;
            const float score = predicted[(size_t) depth - 1];
            if (!std::isfinite(score)) continue;
            float & pred = confidence_pred_ema_[(size_t) depth];
            float & actual = confidence_actual_ema_[(size_t) depth];
            pred = (1.0f - kCalibrationAlpha) * pred +
                   kCalibrationAlpha * std::clamp(score, 0.0f, 1.0f);
            actual = (1.0f - kCalibrationAlpha) * actual +
                     kCalibrationAlpha * (accepted > depth ? 1.0f : 0.0f);
        }
    }

    // Calibration state for a depth: recent predicted and observed acceptance.
    float confidence_predicted(int depth) const {
        return depth >= 1 && depth < max_width_
            ? confidence_pred_ema_[(size_t) depth] : 0.0f;
    }
    float confidence_observed(int depth) const {
        return depth >= 1 && depth < max_width_
            ? confidence_actual_ema_[(size_t) depth] : 0.0f;
    }

    // The per-depth correction currently applied to confidence scores.
    float confidence_scale(int depth) const {
        if (depth < 1 || depth >= max_width_) return 1.0f;
        const float pred = confidence_pred_ema_[(size_t) depth];
        if (pred <= 0.0f) return 1.0f;
        return std::clamp(confidence_actual_ema_[(size_t) depth] / pred,
                          kCalibrationMinScale, kCalibrationMaxScale);
    }

    // Restore the acceptance state for a new request. Learned step costs are
    // backend properties and survive; their warmup hold is re-armed.
    void reset() {
        accepted_candidates_ema_ = initial_accepted_candidates_;
        std::fill(prefix_survival_ema_.begin(),
                  prefix_survival_ema_.end(), kSurvivalPrior);
        std::fill(width_cost_samples_.begin(), width_cost_samples_.end(), 0);
        std::fill(confidence_pred_ema_.begin(), confidence_pred_ema_.end(),
                  kSurvivalPrior);
        std::fill(confidence_actual_ema_.begin(), confidence_actual_ema_.end(),
                  kSurvivalPrior);
        full_accept_streak_ = 0;
        max_width_cooldown_remaining_ = 0;
        max_width_active_ = max_width_initially_active_;
        last_draft_clean_ = false;
    }

    bool enabled() const { return enabled_; }
    float accepted_candidates_ema() const { return accepted_candidates_ema_; }
    int min_width() const { return min_width_; }
    int max_width() const { return max_width_; }

private:
    // P(candidate at depth accepted | shallower candidates accepted) as
    // learned by observe(): s[depth] / s[depth - 1], with s[0] = 1.
    float calibrated_confidence(int depth, float score) const {
        return std::clamp(std::clamp(score, 0.0f, 1.0f) * confidence_scale(depth),
                          0.0f, 1.0f);
    }

    float learned_conditional_acceptance(int depth) const {
        const float deeper = prefix_survival_ema_[(size_t) depth];
        const float shallower = std::max(
            kSurvivalFloor,
            depth >= 2 ? prefix_survival_ema_[(size_t) depth - 1] : 1.0f);
        return std::clamp(deeper / shallower, 0.0f, 1.0f);
    }

    bool max_width_guard_enabled() const {
        return max_width_probe_streak_ > 0 && max_width_ > min_width_;
    }

    bool max_width_is_eligible() const {
        return !max_width_guard_enabled() ||
               max_width_active_ ||
               (full_accept_streak_ >= max_width_probe_streak_ &&
                max_width_cooldown_remaining_ == 0);
    }

    int max_width_;
    int min_width_;
    bool enabled_;
    float initial_accepted_candidates_;
    float backoff_alpha_;
    float full_accept_probe_;
    float accepted_candidates_ema_;
    std::vector<float> prefix_survival_ema_;
    std::vector<float> width_cost_ema_;
    std::vector<int> width_cost_samples_;
    // Per-width expected value of the current cost-aware decision; kept as
    // a member so the const decision path allocates nothing.
    mutable std::vector<float> utility_scratch_;
    std::vector<float> confidence_pred_ema_;
    std::vector<float> confidence_actual_ema_;
    bool last_draft_clean_ = false;
    int max_width_probe_streak_ = 0;
    int max_width_rejection_cooldown_ = 0;
    int full_accept_streak_ = 0;
    int max_width_cooldown_remaining_ = 0;
    bool max_width_initially_active_ = false;
    bool max_width_active_ = false;
};

} // namespace dflash::common
