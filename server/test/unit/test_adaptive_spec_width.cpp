#include "CppUnitTestFramework.hpp"
#include "common/adaptive_spec_width.h"

#include <cmath>
#include <limits>
#include <vector>

using namespace luce::common;

namespace {
struct AdaptiveSpecWidthFixture {};

// The DS4 gfx1151 step-cost curve, seed-inclusive width 0..5: q3 and q4 cost
// almost as much as q5, so acceptance decides between q2 and q5.
const std::vector<float> kDs4StepCosts = {0.0f, 0.0f, 75.0f, 100.0f, 122.0f, 123.0f};

bool near(float lhs, float rhs, float tolerance = 1e-6f) {
    return std::fabs(lhs - rhs) <= tolerance;
}
} // namespace

TEST_CASE(AdaptiveSpecWidthFixture, starts_with_two_candidates) {
    AdaptiveSpecWidth width(16);
    CHECK(width.next_width() == 3);
}

TEST_CASE(AdaptiveSpecWidthFixture, clean_drafts_probe_upward) {
    AdaptiveSpecWidth width(8);
    CHECK(width.next_width() == 3);
    width.observe(3, 3);
    CHECK(width.next_width() == 4);
    width.observe(4, 4);
    CHECK(width.next_width() == 5);
}

TEST_CASE(AdaptiveSpecWidthFixture, censored_sample_is_not_averaged_down) {
    AdaptiveSpecWidth width(8, 2, true, 5.0f);
    width.observe(3, 3);
    CHECK(near(width.accepted_candidates_ema(), 6.0f));
    CHECK(width.next_width() == 7);
}

TEST_CASE(AdaptiveSpecWidthFixture, rejection_backs_off_gently) {
    AdaptiveSpecWidth width(8, 2, true, 4.0f);
    width.observe(2, 6); // one accepted candidate, then a rejection
    CHECK(near(width.accepted_candidates_ema(), 3.25f));
    CHECK(width.next_width() == 4);
}

TEST_CASE(AdaptiveSpecWidthFixture, respects_model_and_feedback_caps) {
    AdaptiveSpecWidth width(8, 2, true, 5.0f);
    CHECK(width.next_width(4) == 4);
    CHECK(width.next_width(99) == 6);
    CHECK(width.next_width(1) == 1);
}

TEST_CASE(AdaptiveSpecWidthFixture, respects_minimum_width) {
    AdaptiveSpecWidth width(16, 6, true, 0.0f);
    CHECK(width.next_width() == 6);
    width.observe(1, 6);
    CHECK(width.next_width() == 6);
}

TEST_CASE(AdaptiveSpecWidthFixture, shared_policy_can_back_off_to_two_rows) {
    AdaptiveSpecWidth width(16, 2, true);
    for (int step = 0; step < 20; ++step) {
        width.observe(1, width.next_width());
    }
    CHECK(width.next_width() == 2);
    width.observe(2, 2);
    width.observe(width.next_width(), width.next_width());
    CHECK(width.next_width() > 2);
}

TEST_CASE(AdaptiveSpecWidthFixture, one_row_model_is_valid_with_two_row_floor) {
    AdaptiveSpecWidth width(1, 2, true);
    CHECK(width.next_width() == 1);
    width.observe(1, 1);
    CHECK(width.next_width() == 1);
}

TEST_CASE(AdaptiveSpecWidthFixture, disabled_controller_preserves_proposal) {
    AdaptiveSpecWidth width(16, 2, false);
    CHECK(width.next_width() == 16);
    CHECK(width.next_width(7) == 7);
    width.observe(1, 16);
    CHECK(width.next_width() == 16);
}

TEST_CASE(AdaptiveSpecWidthFixture, reset_restores_initial_estimate) {
    AdaptiveSpecWidth width(8);
    width.observe(1, 3);
    CHECK(!near(width.accepted_candidates_ema(), 2.0f));
    width.reset();
    CHECK(near(width.accepted_candidates_ema(), 2.0f));
    CHECK(width.next_width() == 3);
}

TEST_CASE(AdaptiveSpecWidthFixture, cost_aware_width_prefers_profitable_depth) {
    AdaptiveSpecWidth width(4);
    width.set_relative_costs({0.0f, 0.0f, 0.70f, 0.94f, 1.0f});

    CHECK(width.next_width_cost_aware({0.9f, 0.9f, 0.9f}) == 4);
    CHECK(width.next_width_cost_aware({0.4f, 0.4f, 0.4f}) == 2);
}

TEST_CASE(AdaptiveSpecWidthFixture, cost_aware_width_learns_prefix_survival) {
    AdaptiveSpecWidth width(4);
    width.set_relative_costs({0.0f, 0.0f, 0.70f, 0.94f, 1.0f});
    CHECK(width.next_width_cost_aware({}) == 4);

    for (int i = 0; i < 8; ++i) {
        width.observe(1, 4);
    }
    CHECK(width.next_width_cost_aware({}) == 2);
}

TEST_CASE(AdaptiveSpecWidthFixture, observed_cost_updates_only_offered_width) {
    AdaptiveSpecWidth width(4);
    width.set_relative_costs({0.0f, 0.0f, 70.0f, 100.0f, 130.0f});
    CHECK(width.next_width_cost_aware({1.0f, 1.0f, 1.0f}) == 4);

    for (int i = 0; i < 5; ++i) {
        width.observe(4, 4, 260.0f);
    }
    CHECK(width.next_width_cost_aware({1.0f, 1.0f, 1.0f}) == 3);
}

TEST_CASE(AdaptiveSpecWidthFixture, nearly_free_fifth_slot_stays_profitable) {
    AdaptiveSpecWidth width(5, 4, true, 3.0f);
    width.set_relative_costs(
        {0.0f, 0.0f, 75.0f, 100.0f, 122.0f, 123.0f});

    CHECK(width.next_width_cost_aware({}) == 5);
    for (int i = 0; i < 8; ++i) {
        // Three accepted candidates and a rejected fourth candidate still
        // make q5 worthwhile when it costs only one millisecond over q4.
        width.observe(4, 5, 123.0f);
    }
    CHECK(width.next_width_cost_aware({}) == 5);
}

TEST_CASE(AdaptiveSpecWidthFixture, costly_fifth_slot_backs_off) {
    AdaptiveSpecWidth width(5, 4, true, 3.0f);
    width.set_relative_costs(
        {0.0f, 0.0f, 75.0f, 100.0f, 122.0f, 180.0f});

    CHECK(width.next_width_cost_aware({}) == 4);
}

TEST_CASE(AdaptiveSpecWidthFixture, guarded_max_width_requires_clean_streak) {
    AdaptiveSpecWidth width(5, 4, true, 3.0f);
    width.set_relative_costs({0.0f, 0.0f, 80.0f, 95.0f, 107.0f, 123.0f});
    width.set_max_width_guard(3, 4);

    CHECK(width.next_width_cost_aware({}) == 4);
    width.observe(4, 4);
    width.observe(4, 4);
    CHECK(width.next_width_cost_aware({}) == 4);
    width.observe(4, 4);
    CHECK(width.next_width_cost_aware({}) == 5);

    width.observe(3, 5);
    CHECK(width.next_width_cost_aware({}) == 4);
    for (int i = 0; i < 3; ++i) width.observe(4, 4);
    CHECK(width.next_width_cost_aware({}) == 4);
    width.observe(4, 4);
    CHECK(width.next_width_cost_aware({}) == 5);
}

TEST_CASE(AdaptiveSpecWidthFixture, guarded_max_width_can_probe_optimistically) {
    AdaptiveSpecWidth width(5, 4, true, 3.0f);
    width.set_relative_costs({0.0f, 0.0f, 80.0f, 95.0f, 107.0f, 123.0f});
    width.set_max_width_guard(3, 4, true);

    CHECK(width.next_width_cost_aware({}) == 5);
    width.observe(5, 5);
    CHECK(width.next_width_cost_aware({}) == 5);
    width.observe(3, 5);
    CHECK(width.next_width_cost_aware({}) == 4);
    width.reset();
    CHECK(width.next_width_cost_aware({}) == 5);
}

TEST_CASE(AdaptiveSpecWidthFixture, clean_drafts_saturate_at_the_cap) {
    AdaptiveSpecWidth width(5, 2, true);
    for (int step = 0; step < 12; ++step) {
        const int offered = width.next_width();
        CHECK(offered >= 2);
        CHECK(offered <= 5);
        width.observe(offered, offered);
    }
    CHECK(width.next_width() == 5);
    CHECK(near(width.accepted_candidates_ema(), 4.0f));
    width.observe(5, 5);
    CHECK(near(width.accepted_candidates_ema(), 4.0f));
    CHECK(width.next_width() == 5);
}

TEST_CASE(AdaptiveSpecWidthFixture, floor_above_cap_clamps_to_cap) {
    AdaptiveSpecWidth width(3, 5, true);
    CHECK(width.min_width() == 3);
    CHECK(width.max_width() == 3);
    CHECK(width.next_width() == 3);
    width.observe(1, 3);
    CHECK(width.next_width() == 3);
    width.set_relative_costs({0.0f, 0.0f, 1.0f, 1.5f});
    CHECK(width.next_width_cost_aware({}) == 3);
    CHECK(width.next_width_cost_aware({}, 99) == 3);
}

TEST_CASE(AdaptiveSpecWidthFixture, full_rejection_is_the_strongest_narrowing_signal) {
    AdaptiveSpecWidth width(5, 2, true, 4.0f);
    CHECK(width.next_width() == 5);
    width.observe(1, 5); // only the seed survived
    CHECK(near(width.accepted_candidates_ema(), 3.0f));
    CHECK(width.next_width() == 4);
    for (int step = 0; step < 6; ++step) {
        width.observe(1, width.next_width());
    }
    CHECK(width.next_width() == 2);
}

TEST_CASE(AdaptiveSpecWidthFixture, cost_aware_collapses_from_q5_to_q2_under_full_rejections) {
    // The gfx1151 DS4 cost curve: q3 and q4 cost almost as much as q5.
    AdaptiveSpecWidth width(5, 2, true);
    width.set_relative_costs(kDs4StepCosts);
    CHECK(width.next_width_cost_aware({}) == 5);
    int full_rejections = 0;
    while (width.next_width_cost_aware({}) != 2 && full_rejections < 32) {
        const int offered = width.next_width_cost_aware({});
        CHECK(offered >= 2);
        CHECK(offered <= 5);
        width.observe(1, offered);
        ++full_rejections;
    }
    CHECK(width.next_width_cost_aware({}) == 2);
    CHECK(full_rejections <= 8);
}

TEST_CASE(AdaptiveSpecWidthFixture, cost_aware_skips_widths_without_a_usable_cost) {
    AdaptiveSpecWidth width(4);
    // Zero and negative entries leave q3 and q4 unseeded.
    width.set_relative_costs({0.0f, 0.0f, 1.0f, 0.0f, -1.0f});
    const std::vector<float> confident = {0.9f, 0.9f, 0.9f};
    CHECK(width.next_width_cost_aware(confident) == 2);
    width.observe(4, 4, std::numeric_limits<float>::quiet_NaN());
    width.observe(4, 4, 0.0f);
    CHECK(width.next_width_cost_aware(confident) == 2);
    width.observe(4, 4, 1.0f); // the first usable observation seeds q4
    CHECK(width.next_width_cost_aware(confident) == 4);
}

TEST_CASE(AdaptiveSpecWidthFixture, unseeded_width_adopts_first_cost_then_tracks) {
    AdaptiveSpecWidth width(3);
    width.set_relative_costs({0.0f, 0.0f, 1.0f}); // q3 has no calibrated seed
    const std::vector<float> certain = {1.0f, 1.0f};
    CHECK(width.next_width_cost_aware(certain) == 2);
    width.observe(3, 3, 1.0f);   // adopted as the q3 estimate: 3/1 beats 2/1
    CHECK(width.next_width_cost_aware(certain) == 3);
    width.observe(3, 3, 100.0f); // no warmup hold without a seed: tracks now
    CHECK(width.next_width_cost_aware(certain) == 2);
}

TEST_CASE(AdaptiveSpecWidthFixture, seeded_width_holds_its_seed_through_warmup) {
    AdaptiveSpecWidth width(3);
    width.set_relative_costs({0.0f, 0.0f, 1.0f, 1.0f});
    const std::vector<float> certain = {1.0f, 1.0f};
    for (int sample = 0; sample < AdaptiveSpecWidth::kCostWarmupSamples; ++sample) {
        width.observe(3, 3, 100.0f); // cold graph-build outliers
        CHECK(width.next_width_cost_aware(certain) == 3);
    }
    width.observe(3, 3, 100.0f);
    CHECK(width.next_width_cost_aware(certain) == 2);
}

TEST_CASE(AdaptiveSpecWidthFixture, cost_aware_never_exceeds_the_proposal) {
    AdaptiveSpecWidth width(5, 2, true);
    width.set_relative_costs({0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    const std::vector<float> certain = {1.0f, 1.0f, 1.0f, 1.0f};
    CHECK(width.next_width_cost_aware(certain, 5) == 5);
    CHECK(width.next_width_cost_aware(certain, 3) == 3);
    CHECK(width.next_width_cost_aware(certain, 1) == 1);
    CHECK(width.next_width_cost_aware(certain, 99) == 5);
}

TEST_CASE(AdaptiveSpecWidthFixture, cost_aware_re_widens_from_q2_when_acceptance_recovers) {
    // DS4 gfx1151 relative step costs for widths 2..5.
    AdaptiveSpecWidth width(5, 2);
    width.set_relative_costs(kDs4StepCosts);
    // A prose-like regime: full rejections at q5 collapse the width to q2.
    for (int step = 0; step < 8 && width.next_width_cost_aware({}) != 2; ++step) {
        width.observe(1, width.next_width_cost_aware({}));
    }
    REQUIRE(width.next_width_cost_aware({}) == 2);
    // Rejections at q2 keep it there: the deeper prefixes take real zeros.
    for (int step = 0; step < 16; ++step) {
        width.observe(1, 2);
    }
    CHECK(width.next_width_cost_aware({}) == 2);
    // A code-like regime: every draft is clean. The controller must climb
    // back to the cap, one confirming width at a time, within a few steps.
    int widest = 2;
    int steps_to_cap = -1;
    int offered = 2;
    for (int step = 0; step < 64; ++step) {
        width.observe(offered, offered);
        const int next = width.next_width_cost_aware({});
        CHECK(next <= offered + 1);   // one width per confirmation
        CHECK(next >= 2);
        widest = std::max(widest, next);
        offered = next;
        if (next == 5 && steps_to_cap < 0) steps_to_cap = step + 1;
    }
    CHECK(widest == 5);
    CHECK(steps_to_cap > 0);
    CHECK(steps_to_cap <= 32);
}

TEST_CASE(AdaptiveSpecWidthFixture, rejection_below_offered_width_zeroes_deeper_prefixes) {
    AdaptiveSpecWidth width(5, 2);
    width.set_relative_costs(kDs4StepCosts);
    // Drive every depth to a confident survival first.
    for (int step = 0; step < 32; ++step) width.observe(5, 5);
    REQUIRE(width.next_width_cost_aware({}) == 5);
    // Now offer q3 only and reject the first candidate every time: q4 and q5
    // must stop looking attractive even though they were never offered.
    for (int step = 0; step < 32; ++step) width.observe(1, 3);
    CHECK(width.next_width_cost_aware({}) == 2);
}

TEST_CASE(AdaptiveSpecWidthFixture, cost_aware_stays_narrow_under_prose_like_acceptance) {
    // 0.6 acceptance per candidate: q2 commits 1.6 tokens for 75 units, q3
    // would commit about 1.96 for 100. The controller must not probe wider
    // from clean q2 drafts alone; the geometric extrapolation keeps the q3
    // estimate near 0.36 while rejections keep pulling it down.
    AdaptiveSpecWidth width(5, 2);
    width.set_relative_costs(kDs4StepCosts);
    for (int step = 0; step < 8 && width.next_width_cost_aware({}) != 2; ++step) {
        width.observe(1, width.next_width_cost_aware({}));
    }
    REQUIRE(width.next_width_cost_aware({}) == 2);
    int wide_steps = 0;
    unsigned seed = 12345u;
    for (int step = 0; step < 400; ++step) {
        const int offered = width.next_width_cost_aware({});
        if (offered > 2) ++wide_steps;
        // Bernoulli(0.6) acceptance per candidate, prefix semantics.
        int accepted = 1;
        for (int cand = 1; cand < offered; ++cand) {
            seed = seed * 1664525u + 1013904223u;
            if ((seed >> 8) % 100 < 60) ++accepted; else break;
        }
        width.observe(accepted, offered);
    }
    CHECK(wide_steps <= 20);   // at most 5% probing on prose-like traffic
}

TEST_CASE(AdaptiveSpecWidthFixture, cost_aware_extends_a_short_confidence_vector_with_learned_depths) {
    AdaptiveSpecWidth width(5, 2);
    width.set_relative_costs(kDs4StepCosts);
    // Learn a confident regime at every depth from clean q5 drafts.
    for (int step = 0; step < 32; ++step) width.observe(5, 5);
    // A head that scores only three candidates, all confident: the fourth
    // depth comes from the learned conditionals, so q5 is still reachable.
    CHECK(width.next_width_cost_aware({0.95f, 0.95f, 0.95f}) == 5);
    // The head says the second candidate is hopeless: stop at q2 whatever
    // the learned regime says about deeper depths.
    CHECK(width.next_width_cost_aware({0.95f, 0.05f, 0.95f}) == 2);
    // Now learn a hopeless regime beyond depth one: the same confident
    // three-candidate head must no longer reach q5 through the learned tail.
    for (int step = 0; step < 32; ++step) width.observe(2, 5);
    const int w = width.next_width_cost_aware({0.95f, 0.95f, 0.95f});
    CHECK(w >= 2);
    CHECK(w <= 4);
}

TEST_CASE(AdaptiveSpecWidthFixture, confidence_scores_are_calibrated_against_the_target) {
    AdaptiveSpecWidth width(5, 2);
    width.set_relative_costs(kDs4StepCosts);
    // An optimistic head: it scores the second candidate 0.8 while the target
    // accepts it two times in five. Uncalibrated, q3 looks worth its cost.
    const std::vector<float> head = {0.9f, 0.8f, 0.7f};
    CHECK(width.next_width_cost_aware(head) >= 3);
    int wide_after_warmup = 0;
    for (int step = 0; step < 80; ++step) {
        const int chosen = width.next_width_cost_aware(head);
        if (step >= 40 && chosen > 2) ++wide_after_warmup;
        const int offered = std::max(3, chosen);
        const int accepted = (step % 5) < 2 ? 3 : 2;   // first always, second 40%
        width.observe(accepted, offered);
        width.observe_confidence(head, accepted, offered);
    }
    CHECK(width.confidence_scale(1) > 0.95f);
    CHECK(width.confidence_scale(2) > 0.35f);
    CHECK(width.confidence_scale(2) < 0.65f);
    // With depth two scaled to its real acceptance q2 pays best.
    CHECK(wide_after_warmup <= 4);
    CHECK(width.next_width_cost_aware(head) == 2);
    // A new request starts uncalibrated.
    width.reset();
    CHECK(near(width.confidence_scale(2), 1.0f));
}

TEST_CASE(AdaptiveSpecWidthFixture, clean_draft_explores_one_width_up_on_a_near_tie) {
    AdaptiveSpecWidth width(5, 2);
    width.set_relative_costs(kDs4StepCosts);
    // Head scores that put q4 within 2% of q3 and q5 just behind them.
    const std::vector<float> head = {0.95f, 0.90f, 0.66f, 0.10f};
    CHECK(width.next_width_cost_aware(head) == 3);
    // A rejected draft does not explore.
    width.observe(2, 3);
    CHECK(width.next_width_cost_aware(head) == 3);
    // A clean draft steps one width up on the near tie, never two.
    width.observe(3, 3);
    CHECK(width.next_width_cost_aware(head) == 4);
    // Prose-like scores keep q2: q3 is 8% short, beyond the margin.
    AdaptiveSpecWidth prose(5, 2);
    prose.set_relative_costs(kDs4StepCosts);
    prose.observe(2, 2);
    CHECK(prose.next_width_cost_aware({0.60f, 0.60f, 0.60f, 0.60f}) == 2);
}
