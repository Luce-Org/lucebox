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
//    Step costs are seeded by set_relative_costs() and refined online from
//    the observed cost of the offered width; the first kCostWarmupSamples
//    observations of a seeded width are ignored so a cold shape-specific
//    graph build cannot poison the estimate. A seed is either this machine's
//    cost (CostSeed::kEstimate: observations refine it, and a width that is
//    not offered keeps its last estimate) or only the shape of the cost curve
//    (CostSeed::kShape: a prior or another machine's curve). A shape seed is
//    never mixed with measurements: an unmeasured width is priced at its seed
//    scaled by the nearest measured width, and never above a measured
//    narrower width, so a wider shape the seed overprices gets measured
//    instead of never being offered. measured_costs()/set_measured_costs()
//    carry measurements across controllers and requests; SpecWidthCostMemory
//    keeps one set per context range, since attention's share of a step
//    grows with the context.
//
//    Survival at depths beyond the offered width is not frozen: a rejection
//    at depth d is a real zero sample for every deeper prefix, and a clean
//    draft extrapolates the next depth geometrically from the two deepest
//    observed survivals. So a narrow width never becomes absorbing, yet low
//    per-candidate acceptance keeps it narrow; when acceptance recovers the
//    controller re-widens one width per confirming step.
//
//    With AcceptanceModel::kReachedDepths the target-observed acceptance is
//    instead counted per depth, only at depths a step reached: a clean draft
//    says nothing about the next depth, and a rejection is not another
//    failure of every deeper candidate. Counts are bounded so the text can
//    change under them, carry_acceptance() keeps a bounded share of them for
//    the next request, and every kProbeInterval steps the controller offers
//    one width wider than its choice so unreached depths stay measured.
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
// Opt in with LUCE_ADAPTIVE_SPEC_WIDTH=1. Fixed width remains the shared
// default because narrower verification is not automatically cheaper on every
// backend; a backend may enable the controller by default on a qualified
// device. Backend-specific fixed-width overrides still take precedence.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace luce::common {

inline bool adaptive_spec_width_globally_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LUCE_ADAPTIVE_SPEC_WIDTH");
        return value != nullptr && value[0] != '\0' &&
               std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

class AdaptiveSpecWidth {
public:
    // What a seeded step cost is: this machine's cost, refined online, or only
    // the shape of the cost curve, scaled to this machine's measurements.
    enum class CostSeed { kEstimate, kShape };
    // How target-observed acceptance is learned when no confidence head
    // scores the step: prefix-survival EMAs (default) or reached-depth counts.
    enum class AcceptanceModel { kPrefixSurvival, kReachedDepths };

    // Prefix-survival estimate for every depth before any evidence.
    static constexpr float kSurvivalPrior = 0.75f;
    // EMA rate of the per-depth prefix-survival estimates.
    static constexpr float kSurvivalAlpha = 0.20f;
    // EMA rate of the per-width step-cost estimates.
    static constexpr float kCostAlpha = 0.20f;
    // Observations of a seeded width ignored before its cost starts tracking.
    static constexpr int kCostWarmupSamples = 4;
    // With a shape seed, a measured width's sample counts at most this
    // multiple of its price before it: a step that also built a graph (a new
    // position band, a new request) costs several times a steady step, and
    // one such outlier must not price the width out for a dozen steps. A real
    // cost rise still lands, at up to kCostAlpha * 25% per step.
    static constexpr float kCostOutlierRatio = 1.25f;
    // With a shape seed, a measured width's cost is the running mean of its
    // live samples, then a 1/kCostWindow average: a step's cost is close to
    // stationary, and an EMA's short window turns timing jitter into
    // misordered widths. A carried measurement starts at the full window.
    static constexpr int kCostWindow = 16;
    // With a shape seed, a measurement not refreshed for this many timed steps
    // is stale: the width is priced no higher than its shape price, and its
    // next sample restarts the mean. A width measured during a slow phase (a
    // cold start, a busy device) would otherwise never be offered again.
    static constexpr int kCostStaleSteps = 64;
    // With a shape seed, live samples a width's mean needs before it prices
    // the width (and scales the others); until then its shape price stands.
    static constexpr int kCostTrustSamples = 4;
    // Reached-depth counts: the prior per depth (an optimistic but finite
    // belief), the bound past which old evidence decays, and the probe period.
    static constexpr float kReachedPriorTrials = 16.0f;
    static constexpr float kReachedPriorRate = 0.90f;
    static constexpr float kReachedWindow = 128.0f;
    static constexpr int kProbeInterval = 17;
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
          width_cost_seed_((size_t) std::max(1, max_width_) + 1,
                           std::numeric_limits<float>::quiet_NaN()),
          width_cost_ema_((size_t) std::max(1, max_width_) + 1,
                          std::numeric_limits<float>::quiet_NaN()),
          width_cost_samples_((size_t) std::max(1, max_width_) + 1, 0),
          width_cost_step_((size_t) std::max(1, max_width_) + 1, 0),
          utility_scratch_((size_t) std::max(1, max_width_) + 1, 0.0f),
          confidence_pred_ema_((size_t) std::max(1, max_width_), kSurvivalPrior),
          confidence_actual_ema_((size_t) std::max(1, max_width_), kSurvivalPrior),
          reached_trials_((size_t) std::max(1, max_width_), kReachedPriorTrials),
          reached_successes_((size_t) std::max(1, max_width_),
                             kReachedPriorTrials * kReachedPriorRate) {}

    void set_acceptance_model(AcceptanceModel model) { acceptance_model_ = model; }

    // Reached-depth model: keep at most max_trials of each depth's evidence
    // for the next request, so earlier text informs it without outweighing
    // its own first few dozen steps.
    void carry_acceptance(float max_trials) {
        for (size_t depth = 1; depth < reached_trials_.size(); ++depth) {
            const float trials = reached_trials_[depth];
            if (trials <= max_trials) continue;
            const float scale = max_trials / trials;
            reached_trials_[depth] *= scale;
            reached_successes_[depth] *= scale;
        }
    }

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
    // calibration and then refine it online through observe(). With
    // CostSeed::kShape the values are only a curve shape (see width_cost()).
    void set_relative_costs(const std::vector<float> & costs,
                            CostSeed kind = CostSeed::kEstimate) {
        cost_seed_ = kind;
        std::vector<float> & seeds =
            kind == CostSeed::kShape ? width_cost_seed_ : width_cost_ema_;
        const size_t n = std::min(costs.size(), seeds.size());
        for (size_t width = 0; width < n; ++width) {
            const float cost = costs[width];
            if (std::isfinite(cost) && cost > 0.0f) {
                seeds[width] = cost;
            }
        }
    }

    // Step costs by width in observe() units, NaN where a width has none
    // (with CostSeed::kEstimate, unmeasured widths report their seed). Costs
    // belong to the machine, not to the text: a backend hands them from one
    // request's controller to the next, or keeps one set per context range
    // (SpecWidthCostMemory). set_measured_costs() replaces them all; a
    // measured width tracks its next sample without a warmup hold, weighing
    // the carried cost as `weight` samples (kCostTrustSamples..kCostWindow):
    // a full window for this context's own costs, the trust floor for costs
    // borrowed from another context, so the first live samples take over.
    const std::vector<float> & measured_costs() const { return width_cost_ema_; }
    void set_measured_costs(const std::vector<float> & costs, int weight = kCostWindow) {
        const int held = kCostWarmupSamples + std::clamp(weight, kCostTrustSamples, kCostWindow);
        for (size_t width = 0; width < width_cost_ema_.size(); ++width) {
            const float cost = width < costs.size()
                ? costs[width] : std::numeric_limits<float>::quiet_NaN();
            const bool measured = std::isfinite(cost) && cost > 0.0f;
            width_cost_ema_[width] =
                measured ? cost : std::numeric_limits<float>::quiet_NaN();
            width_cost_samples_[width] = measured ? held : 0;
            width_cost_step_[width] = cost_steps_;
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
    // the depths it does not cover. Widths are priced by width_cost(); those
    // without a usable cost are skipped, and when none is usable the
    // acceptance-only policy decides.
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
                survival = acceptance_model_ == AcceptanceModel::kReachedDepths
                    ? survival * reached_trials_rate(depth)
                    : prefix_survival_ema_[(size_t) depth];
            } else {
                // The head covers fewer depths than the proposal: continue
                // with the learned conditional acceptance of this depth, the
                // ratio of consecutive prefix-survival estimates.
                survival *= learned_conditional_acceptance(depth);
            }
            expected_commits += survival;
            if (width < min_width_) continue;
            const float cost = width_cost(width);
            if (!std::isfinite(cost) || cost <= 0.0f) continue;
            const float utility = expected_commits / cost;
            utilities[(size_t) width] = utility;
            if (utility > best_utility) {
                best_utility = utility;
                best_width = width;
            }
        }
        if (acceptance_model_ == AcceptanceModel::kReachedDepths) {
            // Reached-depth counts explore on a fixed period: a probe right
            // after every clean draft would alternate widths step by step
            // and time each one in a different phase of the device.
            if (best_width >= 2 && steps_ > 0 && steps_ % kProbeInterval == 0) {
                best_width = std::min(proposed, best_width + 1);
            }
        } else if (best_width >= 2 && last_draft_clean_ && best_width < proposed &&
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
        ++steps_;
        for (int depth = 1;
             depth <= std::min(offered_candidates, accepted_candidates + 1); ++depth) {
            float & trials = reached_trials_[(size_t) depth];
            float & successes = reached_successes_[(size_t) depth];
            if (trials >= kReachedWindow) {
                trials *= (kReachedWindow - 1.0f) / kReachedWindow;
                successes *= (kReachedWindow - 1.0f) / kReachedWindow;
            }
            trials += 1.0f;
            successes += depth <= accepted_candidates ? 1.0f : 0.0f;
        }

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
            observe_cost(offered, observed_step_cost);
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
        std::fill(reached_trials_.begin(), reached_trials_.end(), kReachedPriorTrials);
        std::fill(reached_successes_.begin(), reached_successes_.end(),
                  kReachedPriorTrials * kReachedPriorRate);
        steps_ = 0;
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
    bool has_seed(int width) const {
        const float seed = width_cost_seed_[(size_t) width];
        return std::isfinite(seed) && seed > 0.0f;
    }

    bool has_measured_cost(int width) const {
        const float cost = width_cost_ema_[(size_t) width];
        return std::isfinite(cost) && cost > 0.0f;
    }

    // Shape seeds: a width's samples past its warmup, all in its mean.
    int live_samples(int width) const {
        return std::max(0, width_cost_samples_[(size_t) width] - kCostWarmupSamples);
    }

    // Shape seeds: a measured width prices itself once its mean holds
    // kCostTrustSamples samples; one jittered step must not decide it.
    bool cost_trusted(int width) const {
        return has_measured_cost(width) && live_samples(width) >= kCostTrustSamples;
    }

    bool cost_stale(int width) const {
        return cost_steps_ - width_cost_step_[(size_t) width] > kCostStaleSteps;
    }

    // Total step cost of a width in observe() units, NaN when it has none.
    // An estimate seed is the cost itself. With a shape seed a trusted
    // measurement answers for its width (no higher than its scaled seed once
    // stale), a width still building its mean answers with its scaled seed,
    // and a width never timed with its shape price. The reached-depth model
    // explores with its own periodic probes, so there a never-timed width is
    // priced at its scaled seed instead: pricing it free would spend a whole
    // short request timing every width of a cold context.
    float width_cost(int width) const {
        if (cost_seed_ == CostSeed::kEstimate) return width_cost_ema_[(size_t) width];
        if (!has_measured_cost(width)) {
            return acceptance_model_ == AcceptanceModel::kReachedDepths
                ? scaled_seed(width) : shape_price(width);
        }
        if (!cost_trusted(width)) return scaled_seed(width);
        const float measured = width_cost_ema_[(size_t) width];
        return cost_stale(width) ? std::min(measured, scaled_seed(width)) : measured;
    }

    // A width's seed scaled by the measured/seed ratio of the nearest trusted
    // seeded width (the narrower one on a tie); the bare seed, in another
    // machine's units, when no width is trusted yet.
    float scaled_seed(int width, bool * scaled = nullptr) const {
        if (scaled) *scaled = false;
        if (!has_seed(width)) return std::numeric_limits<float>::quiet_NaN();
        const float seed = width_cost_seed_[(size_t) width];
        for (int distance = 1; distance <= max_width_; ++distance) {
            for (const int other : {width - distance, width + distance}) {
                if (other >= 1 && other <= max_width_ && other != width &&
                    has_seed(other) && cost_trusted(other)) {
                    if (scaled) *scaled = true;
                    return seed * width_cost_ema_[(size_t) other] /
                           width_cost_seed_[(size_t) other];
                }
            }
        }
        return seed;
    }

    // The price of a width never timed: its scaled seed, and no higher than
    // the widest trusted width below it. Extra candidates count as free until
    // their cost is measured, so a seed curve from another machine cannot
    // keep a profitable width from ever being offered.
    float shape_price(int width) const {
        float cost = scaled_seed(width);
        for (int below = width - 1; below >= 1; --below) {
            if (cost_trusted(below)) {
                cost = std::min(cost, width_cost_ema_[(size_t) below]);
                break;
            }
        }
        return cost;
    }

    void observe_cost(int width, float observed) {
        float & estimate = width_cost_ema_[(size_t) width];
        int & samples = width_cost_samples_[(size_t) width];
        if (cost_seed_ == CostSeed::kEstimate) {
            if (!std::isfinite(estimate) || estimate <= 0.0f) {
                // No calibrated seed for this width: adopt the first
                // observation and track from the next one on.
                estimate = observed;
                samples = kCostWarmupSamples;
            } else if (++samples > kCostWarmupSamples) {
                // Shape-specific graph construction makes the first few
                // observations of a seeded width cold outliers. Preserve the
                // calibrated seed through warmup, then track the live cost.
                estimate = (1.0f - kCostAlpha) * estimate +
                           kCostAlpha * observed;
            }
            return;
        }
        ++cost_steps_;
        // A stale measurement restarts: its next live sample begins a new mean.
        if (has_measured_cost(width) && cost_stale(width)) samples = kCostWarmupSamples;
        width_cost_step_[(size_t) width] = cost_steps_;
        // Skip the cold first samples (again after reset(): a new request
        // builds its own shapes). Every live sample then counts at most
        // kCostOutlierRatio times the width's price before it: its trusted
        // mean, or its seed scaled to this machine.
        if (++samples <= kCostWarmupSamples) return;
        bool scaled = false;
        const float reference = cost_trusted(width) ? estimate : scaled_seed(width, &scaled);
        const float sample = (cost_trusted(width) || scaled) && reference > 0.0f
            ? std::min(observed, kCostOutlierRatio * reference) : observed;
        const int live = std::min(live_samples(width), kCostWindow);
        estimate = live <= 1 ? sample : estimate + (sample - estimate) / (float) live;
    }

    // P(candidate at depth accepted | shallower candidates accepted) as
    // learned by observe(): s[depth] / s[depth - 1], with s[0] = 1.
    float calibrated_confidence(int depth, float score) const {
        return std::clamp(std::clamp(score, 0.0f, 1.0f) * confidence_scale(depth),
                          0.0f, 1.0f);
    }

    float reached_trials_rate(int depth) const {
        return reached_successes_[(size_t) depth] / reached_trials_[(size_t) depth];
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
    CostSeed cost_seed_ = CostSeed::kEstimate;
    // Shape seeds (CostSeed::kShape) and the costs in observe() units.
    std::vector<float> width_cost_seed_;
    std::vector<float> width_cost_ema_;
    std::vector<int> width_cost_samples_;
    // Timed steps observed (shape seeds) and the step each width was last timed.
    int cost_steps_ = 0;
    std::vector<int> width_cost_step_;
    // Per-width expected value of the current cost-aware decision; kept as
    // a member so the const decision path allocates nothing.
    mutable std::vector<float> utility_scratch_;
    std::vector<float> confidence_pred_ema_;
    std::vector<float> confidence_actual_ema_;
    AcceptanceModel acceptance_model_ = AcceptanceModel::kPrefixSurvival;
    std::vector<float> reached_trials_;
    std::vector<float> reached_successes_;
    int steps_ = 0;
    bool last_draft_clean_ = false;
    int max_width_probe_streak_ = 0;
    int max_width_rejection_cooldown_ = 0;
    int full_accept_streak_ = 0;
    int max_width_cooldown_remaining_ = 0;
    bool max_width_initially_active_ = false;
    bool max_width_active_ = false;
};

// Measured step costs by context range. Attention's share of a verify step
// grows with the context, so widths timed in a long request make a short
// request's narrow widths look dearer than they are (and the reverse): a
// backend that keeps one controller across requests loads the range's costs
// before a request and stores them after it.
class SpecWidthCostMemory {
public:
    static constexpr int kRanges = 4;

    static int range(int64_t context) {
        return context < 2048 ? 0 : context < 8192 ? 1 : context < 32768 ? 2 : 3;
    }

    // A range never timed starts from the nearest timed range (the shorter
    // one on a tie), weighted lightly so its own steps take over.
    void load(AdaptiveSpecWidth & width, int64_t context) const {
        const int own = range(context);
        for (int distance = 0; distance < kRanges; ++distance) {
            for (const int other : {own - distance, own + distance}) {
                if (other < 0 || other >= kRanges || costs_[(size_t) other].empty()) continue;
                width.set_measured_costs(costs_[(size_t) other],
                    distance == 0 ? AdaptiveSpecWidth::kCostWindow : AdaptiveSpecWidth::kCostTrustSamples);
                return;
            }
        }
        width.set_measured_costs({});
    }

    void store(const AdaptiveSpecWidth & width, int64_t context) {
        costs_[(size_t) range(context)] = width.measured_costs();
    }

private:
    std::array<std::vector<float>, kRanges> costs_;
};

} // namespace luce::common
