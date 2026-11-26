// Relative query candidates and the verbatim pin (pflash_query_candidates.h): baked-in defaults,
// relative candidate ranking, NULL, extents, aggregation, the verbatim pin, gap rule, and blend. Model-free.

#include "CppUnitTestFramework.hpp"

#include "pflash/pflash_query_candidates.h"
#include "pflash/pflash_selection.h"
#include "scoped_env.h"
#include "server/http_server.h"
#include "server/launch_profiles.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace luce::pflash;
using luce::common::PFlashTokenSpan;

namespace {

struct PFlashQueryCandidatesFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;
};

using Role = PFlashRequestRole;
using Spans = std::vector<PFlashTokenSpan>;

PFlashRequestMessage msg(Role role, int b, int e) { return {role, {b, e}}; }

// Contiguous segments with the given cut points over [0, n).
Spans cover(int n, const std::vector<int> & cuts) {
    Spans out;
    int b = 0;
    for (const int c : cuts) {
        out.push_back({b, c});
        b = c;
    }
    out.push_back({b, n});
    return out;
}

std::vector<float> filled(int n, float v) { return std::vector<float>((size_t) n, v); }

void set(std::vector<float> & s, int b, int e, float v) {
    for (int t = b; t < e; ++t) s[(size_t) t] = v;
}

std::vector<int> range(int b, int e) {
    std::vector<int> out;
    for (int t = b; t < e; ++t) out.push_back(t);
    return out;
}

const float kInf = std::numeric_limits<float>::infinity();

std::vector<std::unique_ptr<luce_test::ScopedEnvVar>> clean_startup_env() {
    std::vector<std::unique_ptr<luce_test::ScopedEnvVar>> vars;
    for (const char * name : {
            "PFLASH_FAIL_CLOSED", "PFLASH_DRAFTER_SESSIONS",
            "PFLASH_SCORING_HEAD_GGUF", "PFLASH_SEGMENT_PROBE_GGUF",
            "PFLASH_REQUEST_PROBE_GGUF", "PFLASH_SEG_RERANKER_GGUF"}) {
        vars.push_back(std::make_unique<luce_test::ScopedEnvVar>(name, nullptr));
    }
    return vars;
}

} // namespace

// ---------------------------------------------------------------- baked-in defaults & startup validation

TEST_CASE(PFlashQueryCandidatesFixture, startup_validation_and_baked_in_november_defaults) {
    const auto clean = clean_startup_env();
    using namespace luce::common::http_detail;
    std::string error;
    REQUIRE(validate_pflash_environment(error));
    REQUIRE(!pflash_fail_closed_from_env());

    PFlashSelectionConfig selection;
    REQUIRE(resolve_pflash_selection(65536, 32, selection, error));
    REQUIRE(selection.configured && selection.selection_active);
    REQUIRE(selection.mode == PFlashSelectionMode::BudgetOnly);
    REQUIRE(selection.query_parser == PFlashQueryParser::ArbitraryTail);
    REQUIRE(selection.segmentation == PFlashSegmentation::Probe);
    REQUIRE(selection.candidate_score == PFlashCandidateScore::Density);
    REQUIRE(selection.chunk_size == 256 && selection.query_tokens == 128);
    REQUIRE(selection.query_probe);
    REQUIRE(selection.query.candidates == 1 && selection.query.pin_tokens == 512);
    REQUIRE(selection.query.extent_gap && selection.query.pin_gap);
    REQUIRE(selection.query.tw_score && selection.query.end_share == 0.25f);
    REQUIRE(selection.query.unit_score == PFlashUnitScore::Max);
    REQUIRE(selection.query.aggregate == PFlashQueryAggregate::Softmax);
    REQUIRE(selection.query.temperature == 1.0f && selection.query.null_margin == 0.0f);
}

TEST_CASE(PFlashQueryCandidatesFixture, release_env_populates_only_sibling_component_paths) {
    const auto clean = clean_startup_env();
    using luce::server::pflash_release_env;
    REQUIRE(pflash_release_env().empty());

    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "luce_pflash_release_profile_test";
    std::filesystem::create_directories(dir);
    const auto stub = [&](const char * name) {
        std::ofstream(dir / name) << "stub";
        return (dir / name).string();
    };
    const std::string head_file = stub("seg-scoring-head.gguf");
    const std::string segment_file = stub("segment-probe.gguf");
    const std::string request_file = stub("request-probe.gguf");
    const std::string reranker_file = stub("seg-reranker.gguf");

    luce_test::ScopedEnvVar head{"PFLASH_SCORING_HEAD_GGUF", head_file.c_str()};
    const auto env = pflash_release_env();
    REQUIRE(env.size() == 3u);
    REQUIRE(env[0].name == "PFLASH_SEGMENT_PROBE_GGUF" && env[0].value == segment_file);
    REQUIRE(env[1].name == "PFLASH_REQUEST_PROBE_GGUF" && env[1].value == request_file);
    REQUIRE(env[2].name == "PFLASH_SEG_RERANKER_GGUF" && env[2].value == reranker_file);
    std::filesystem::remove_all(dir);
}

TEST_CASE(PFlashQueryCandidatesFixture, startup_validation_rejects_invalid_flags) {
    const auto clean = clean_startup_env();
    using luce::common::http_detail::validate_pflash_environment;
    std::string error;
    for (const auto & bad : std::vector<std::pair<const char *, const char *>>{
            {"PFLASH_FAIL_CLOSED", "yes"}, {"PFLASH_DRAFTER_SESSIONS", "-1"},
            {"PFLASH_DRAFTER_SESSIONS", "2oops"},
            {"PFLASH_DRAFTER_SESSIONS", "9999999999999999999999999"}}) {
        luce_test::ScopedEnvVar flag{bad.first, bad.second};
        REQUIRE(!validate_pflash_environment(error));
        REQUIRE(error.find(bad.first) != std::string::npos);
    }
    luce_test::ScopedEnvVar ok_fc{"PFLASH_FAIL_CLOSED", "1"};
    luce_test::ScopedEnvVar ok_sess{"PFLASH_DRAFTER_SESSIONS", "4"};
    REQUIRE(validate_pflash_environment(error));
}

// ---------------------------------------------------------------- units and ranking

TEST_CASE(PFlashQueryCandidatesFixture, units_are_segments_inside_message_content) {
    // system [0,10) user [12,30) assistant [32,40) user [42,50); one segment crosses 28..34.
    const std::vector<PFlashRequestMessage> m = {msg(Role::System, 0, 10), msg(Role::User, 12, 30),
                                                 msg(Role::Assistant, 32, 40), msg(Role::User, 42, 50)};
    auto s1 = filled(52, 0.0f);
    set(s1, 12, 20, 5.0f);
    const auto segs = cover(52, {12, 20, 28, 34, 42});
    const auto u = pflash_query_units(segs, m, s1);
    REQUIRE(u.error.empty());
    REQUIRE(u.latest == 3);
    REQUIRE(!u.system_used);
    // [12,20) [20,28) [28,30) from message 1, [32,34) [34,40) from 2, [42,50) from 3; system excluded.
    REQUIRE(u.units.size() == 6u);
    REQUIRE(u.units[0].span.begin == 12 && u.units[0].span.end == 20 && u.units[0].message == 1);
    REQUIRE(u.units[2].span.begin == 28 && u.units[2].span.end == 30);
    REQUIRE(u.units[3].span.begin == 32 && u.units[3].span.end == 34 && u.units[3].message == 2);
    REQUIRE(u.units[5].latest && u.units[5].span.begin == 42 && u.units[5].span.end == 50);
    REQUIRE(u.units[0].score_max == 5.0f && u.units[0].score_mean == 5.0f);
    // Only a system message: its units are used.
    const auto only = pflash_query_units(cover(20, {5}), {msg(Role::System, 0, 20)}, filled(20, 1.0f));
    REQUIRE(only.system_used && only.units.size() == 2u && only.latest == -1);
    // Fail closed.
    REQUIRE(pflash_query_units({{0, 10}, {12, 52}}, m, s1).error == "segments_invalid");
    REQUIRE(pflash_query_units(segs, m, filled(45, 0.0f)).error == "logits_missing");
    // Non-finite logits do not score; a unit without a finite one is dropped.
    auto nan_s1 = s1;
    set(nan_s1, 42, 50, std::numeric_limits<float>::quiet_NaN());
    REQUIRE(pflash_query_units(segs, m, nan_s1).units.size() == 5u);
}

TEST_CASE(PFlashQueryCandidatesFixture, rank_is_score_descending_ties_later_first) {
    PFlashQueryUnits u;
    u.units = {{{0, 4}, 0, false, 2.0f, 1.0f}, {{4, 8}, 0, false, 3.0f, 0.0f}, {{8, 12}, 0, false, 3.0f, 2.0f}};
    REQUIRE(pflash_query_rank(u, {0, 1, 2}, PFlashUnitScore::Max) == std::vector<int>({2, 1, 0}));
    REQUIRE(pflash_query_rank(u, {0, 1, 2}, PFlashUnitScore::Mean) == std::vector<int>({2, 0, 1}));
}

TEST_CASE(PFlashQueryCandidatesFixture, extent_takes_whole_units_within_the_margin) {
    // One user message [0,100) in 10-token units with scores 9, 1, 8, 1, 7.5, 1, ... (no absolute threshold).
    std::vector<PFlashRequestMessage> m = {msg(Role::User, 0, 100)};
    auto s1 = filled(100, 1.0f);
    set(s1, 0, 10, 9.0f);
    set(s1, 20, 30, 8.0f);
    set(s1, 40, 50, 7.5f);
    std::vector<int> cuts;
    for (int c = 10; c < 100; c += 10) cuts.push_back(c);
    const auto u = pflash_query_units(cover(100, cuts), m, s1);
    REQUIRE(u.units.size() == 10u);
    const auto e0 = pflash_query_extent(u, 0, s1, PFlashUnitScore::Max, 0.0f, 512);
    REQUIRE(e0.tokens == range(0, 10));
    const auto e2 = pflash_query_extent(u, 0, s1, PFlashUnitScore::Max, 2.0f, 512);
    std::vector<int> want = range(0, 10);
    for (int t : range(20, 30)) want.push_back(t);
    for (int t : range(40, 50)) want.push_back(t);
    REQUIRE(e2.tokens == want);
    REQUIRE(e2.units.size() == 3u && e2.top == 9.0f);
    // Cap 25: the 9 unit, the 8 unit; the 7.5 unit does not fit whole -> skipped, never cut.
    const auto capped = pflash_query_extent(u, 0, s1, PFlashUnitScore::Max, 2.0f, 25);
    REQUIRE(capped.units.size() == 2u && capped.skipped == 1 && !capped.cut);
    // A top unit over the cap: its cap best tokens (a query, never a pin).
    const auto cut = pflash_query_extent(u, 0, s1, PFlashUnitScore::Max, 0.0f, 4);
    REQUIRE(cut.cut && cut.tokens.size() == 4u);
    // A short message whose units score alike is its own extent at any margin > its spread.
    std::vector<PFlashRequestMessage> short_m = {msg(Role::User, 0, 12)};
    auto short_s1 = filled(12, 3.0f);
    set(short_s1, 6, 12, 2.6f);
    const auto su = pflash_query_units(cover(12, {6}), short_m, short_s1);
    REQUIRE(pflash_query_extent(su, 0, short_s1, PFlashUnitScore::Max, 1.0f, 512).tokens == range(0, 12));
}

// ---------------------------------------------------------------- candidates and NULL

TEST_CASE(PFlashQueryCandidatesFixture, latest_extent_first_then_other_messages_by_score) {
    // user A [0,20) (score 9), assistant [20,40) (score 2), user B latest [40,60): a long paste whose question unit
    // scores only 3 (logit collapse) -- it still leads: the ranking is relative, never thresholded.
    std::vector<PFlashRequestMessage> m = {msg(Role::User, 0, 20), msg(Role::Assistant, 20, 40),
                                           msg(Role::User, 40, 60)};
    auto s1 = filled(60, -6.0f);
    set(s1, 0, 20, 9.0f);
    set(s1, 20, 40, 2.0f);
    set(s1, 50, 55, 3.0f);
    const auto u = pflash_query_units(cover(60, {20, 40, 45, 50, 55}), m, s1);
    PFlashQueryConfig c;
    c.candidates = 3;
    const auto set3 = pflash_query_candidate_set(u, s1, c);
    REQUIRE(set3.candidates.size() == 3u);
    REQUIRE(set3.candidates[0].message == 2 && set3.candidates[0].tokens == range(50, 55));
    REQUIRE(set3.candidates[1].message == 0 && set3.candidates[2].message == 1);
    double total = 0.0;
    for (const auto & cand : set3.candidates) total += cand.weight;
    REQUIRE(std::fabs(total - 1.0) < 1e-9);
    REQUIRE(set3.candidates[1].weight > set3.candidates[0].weight);   // softmax over the scores
    c.candidates = 1;
    const auto set1 = pflash_query_candidate_set(u, s1, c);
    REQUIRE(set1.candidates.size() == 1u && set1.candidates[0].message == 2);
    REQUIRE(set1.candidates[0].weight == 1.0);
    // Temperature flattens the weights.
    c.candidates = 3;
    c.temperature = 1000.0f;
    const auto flat = pflash_query_candidate_set(u, s1, c);
    REQUIRE(std::fabs(flat.candidates[0].weight - 1.0 / 3.0) < 0.01);
    // No unit: no candidate.
    REQUIRE(pflash_query_candidate_set(PFlashQueryUnits{}, {}, c).candidates.empty());
}

TEST_CASE(PFlashQueryCandidatesFixture, candidate_tokens_drop_negative_request_logits_with_fallback) {
    // 1. Mixed unit: [0,10) has stray 1-token edge blip [1,2) at s1=+1, leading background,
    //    question stem [3,5) at s1=+4, interior declarative option [5,7) at s1=-2, closing
    //    prompt [7,9) at s1=+3, and trailing background [9,10) at s1=-5. Per-component convex
    //    hull anchored on runs >= 2 trims the stray blip [1,2) and outer margins to [3,9)
    //    while preserving interior [5,7) and keeping whole-unit extent [0,10) for pinning.
    std::vector<PFlashRequestMessage> m = {msg(Role::User, 0, 20)};
    auto s1 = filled(20, -5.0f);
    set(s1, 1, 2, 1.0f);
    set(s1, 3, 5, 4.0f);
    set(s1, 5, 7, -2.0f);
    set(s1, 7, 9, 3.0f);
    const auto u = pflash_query_units(cover(20, {10}), m, s1);
    PFlashQueryConfig c;
    c.candidates = 1;
    c.extent_gap = false;
    c.extent_margin = 0.0f;
    const auto set_mix = pflash_query_candidate_set(u, s1, c);
    REQUIRE(set_mix.candidates.size() == 1u);
    REQUIRE(set_mix.candidates[0].units == std::vector<int>({0}));
    REQUIRE(set_mix.candidates[0].tokens == range(3, 9));
    const auto pin = pflash_query_pin(u, 512, PFlashUnitScore::Max, 0.0f, /*gap=*/false);
    REQUIRE(pin.spans.size() == 1u && pin.spans[0].begin == 0 && pin.spans[0].end == 10);

    // 2. Disjoint prefix + suffix units (e.g. [0,10) and [20,30) with doc body [10,20) in between):
    //    each connected component trims its own doc-bleed edge independently without bridging.
    auto s1_disj = filled(30, -5.0f);
    set(s1_disj, 0, 4, 5.0f);     // prefix instruction [0,4), trailing doc bleed [4,10)
    set(s1_disj, 25, 30, 5.0f);   // leading doc bleed [20,25), suffix question [25,30)
    const auto u_disj = pflash_query_units(cover(30, {10, 20}), {msg(Role::User, 0, 30)}, s1_disj);
    const auto set_disj = pflash_query_candidate_set(u_disj, s1_disj, c);
    REQUIRE(set_disj.candidates.size() == 1u);
    std::vector<int> want_disj = {0, 1, 2, 3, 25, 26, 27, 28, 29};
    REQUIRE(set_disj.candidates[0].tokens == want_disj);

    // 3. Adjacent units [0,10) and [10,20) both selected form one contiguous component:
    //    interior negative-logit clauses crossing the unit boundary are preserved.
    auto s1_adj = filled(30, -5.0f);
    set(s1_adj, 2, 5, 5.0f);
    set(s1_adj, 14, 17, 5.0f);
    const auto u_adj = pflash_query_units(cover(30, {10, 20}), {msg(Role::User, 0, 30)}, s1_adj);
    const auto set_adj = pflash_query_candidate_set(u_adj, s1_adj, c);
    REQUIRE(set_adj.candidates.size() == 1u);
    REQUIRE(set_adj.candidates[0].tokens == range(2, 17));

    // 4. Single-token positive fallback: component with only a 1-token hit keeps that token.
    auto s1_one = filled(10, -5.0f);
    set(s1_one, 4, 5, 3.0f);
    const auto u_one = pflash_query_units(cover(10, {}), {msg(Role::User, 0, 10)}, s1_one);
    const auto set_one = pflash_query_candidate_set(u_one, s1_one, c);
    REQUIRE(set_one.candidates.size() == 1u && set_one.candidates[0].tokens == range(4, 5));

    // 5. All-negative unit: fallback keeps the full extent [0,10) so scoring never goes empty.
    auto all_neg = filled(10, -2.0f);
    const auto u_neg = pflash_query_units(cover(10, {}), {msg(Role::User, 0, 10)}, all_neg);
    const auto set_neg = pflash_query_candidate_set(u_neg, all_neg, c);
    REQUIRE(set_neg.candidates.size() == 1u && set_neg.candidates[0].tokens == range(0, 10));
}

TEST_CASE(PFlashQueryCandidatesFixture, null_only_when_the_top_unit_does_not_stand_out) {
    // Latest user message: 10 units, all paste-like (scores 1.0 .. 1.9): z of the top is small -> NULL at margin 3;
    // the earlier request leads instead.
    std::vector<PFlashRequestMessage> m = {msg(Role::User, 0, 10), msg(Role::User, 10, 110)};
    auto s1 = filled(110, 0.0f);
    set(s1, 0, 10, 6.0f);
    std::vector<int> cuts = {10};
    for (int k = 0; k < 10; ++k) {
        set(s1, 10 + 10 * k, 20 + 10 * k, 1.0f + 0.1f * (float) k);
        if (k) cuts.push_back(10 + 10 * k);
    }
    const auto u = pflash_query_units(cover(110, cuts), m, s1);
    PFlashQueryConfig c;
    c.candidates = 2;
    c.null_margin = 3.0f;
    const auto set_null = pflash_query_candidate_set(u, s1, c);
    REQUIRE(set_null.latest_null);
    REQUIRE(set_null.candidates.size() == 1u && set_null.candidates[0].message == 0);
    // Margin 0: never NULL.
    c.null_margin = 0.0f;
    const auto set_on = pflash_query_candidate_set(u, s1, c);
    REQUIRE(!set_on.latest_null && set_on.candidates[0].message == 1);
    // A question unit that stands out is not NULL.
    auto q = s1;
    set(q, 60, 70, 8.0f);
    c.null_margin = 3.0f;
    const auto uq = pflash_query_units(cover(110, cuts), m, q);
    const auto set_q = pflash_query_candidate_set(uq, q, c);
    REQUIRE(!set_q.latest_null && set_q.candidates[0].message == 1 && set_q.candidates[0].tokens == range(60, 70));
    // NULL with no other message: the latest still steers (never empty).
    std::vector<PFlashRequestMessage> alone = {msg(Role::User, 10, 110)};
    const auto ua = pflash_query_units(cover(110, cuts), alone, s1);
    const auto set_alone = pflash_query_candidate_set(ua, s1, c);
    REQUIRE(set_alone.latest_null && set_alone.candidates.size() == 1u && set_alone.candidates[0].message == 0);
    // A short latest message (< 3 units) is never NULL, whatever its logits.
    std::vector<PFlashRequestMessage> fu = {msg(Role::User, 0, 10), msg(Role::User, 10, 16)};
    auto fs1 = filled(16, -9.0f);
    set(fs1, 0, 10, 7.0f);
    const auto uf = pflash_query_units(cover(16, {10}), fu, fs1);
    const auto set_fu = pflash_query_candidate_set(uf, fs1, c);
    REQUIRE(!set_fu.latest_null && std::isnan(set_fu.latest_z));
    REQUIRE(set_fu.candidates[0].message == 1 && set_fu.candidates[0].tokens == range(10, 16));
}

// ---------------------------------------------------------------- aggregation

TEST_CASE(PFlashQueryCandidatesFixture, aggregate_softmax_and_weighted_max) {
    const std::vector<std::vector<float>> masses = {{0.5f, 0.5f, 0.0f}, {0.0f, 0.2f, 0.8f}};
    std::vector<float> out;
    std::string error;
    REQUIRE(pflash_query_aggregate(masses, {0.75, 0.25}, PFlashQueryAggregate::Softmax, out, error));
    REQUIRE(std::fabs(out[0] - 0.375f) < 1e-6f && std::fabs(out[1] - 0.425f) < 1e-6f &&
            std::fabs(out[2] - 0.2f) < 1e-6f);
    REQUIRE(pflash_query_aggregate(masses, {0.75, 0.25}, PFlashQueryAggregate::WeightedMax, out, error));
    // max: .375, .375, .2 -> / .95
    REQUIRE(std::fabs(out[0] - 0.375f / 0.95f) < 1e-6f && std::fabs(out[2] - 0.2f / 0.95f) < 1e-6f);
    REQUIRE(!pflash_query_aggregate(masses, {1.0}, PFlashQueryAggregate::Softmax, out, error));
    REQUIRE(!pflash_query_aggregate({{1.0f}, {0.5f, 0.5f}}, {0.5, 0.5}, PFlashQueryAggregate::Softmax, out, error));
    REQUIRE(!pflash_query_aggregate({}, {}, PFlashQueryAggregate::Softmax, out, error));
    const auto w = pflash_query_softmax({1.0f, 1.0f}, 1.0f);
    REQUIRE(w[0] == 0.5 && w[1] == 0.5);
}

// ---------------------------------------------------------------- the verbatim pin

TEST_CASE(PFlashQueryCandidatesFixture, pin_takes_whole_latest_units_in_rank_order_up_to_the_cap) {
    std::vector<PFlashRequestMessage> m = {msg(Role::User, 0, 30), msg(Role::User, 30, 100)};
    auto s1 = filled(100, 0.0f);
    set(s1, 0, 30, 9.0f);       // earlier message: never pinned
    set(s1, 60, 70, 5.0f);      // question unit
    set(s1, 70, 90, 4.0f);      // its specification (20 tokens)
    set(s1, 90, 100, 1.0f);
    const auto u = pflash_query_units(cover(100, {30, 40, 50, 60, 70, 90}), m, s1);
    // cap 25, fill: 60..70 (10), then 70..90 (20) does not fit -> skipped, then 90..100 (10) fits? 10+10 = 20 <= 25.
    const auto fill = pflash_query_pin(u, 25, PFlashUnitScore::Max, kInf);
    REQUIRE(fill.units == 2 && fill.skipped >= 1 && fill.tokens == 20);
    // cap 30, margin 2: question + specification only, merged.
    const auto rel = pflash_query_pin(u, 30, PFlashUnitScore::Max, 2.0f);
    REQUIRE(rel.spans.size() == 1u && rel.spans[0].begin == 60 && rel.spans[0].end == 90 && rel.tokens == 30);
    // A short latest message is pinned whole.
    const auto whole = pflash_query_pin(u, 512, PFlashUnitScore::Max, kInf);
    REQUIRE(whole.tokens == 70 && whole.spans.size() == 1u && whole.spans[0].begin == 30);
    REQUIRE(pflash_query_pin(u, 0, PFlashUnitScore::Max, kInf).spans.empty());
    PFlashQueryUnits none = u;
    none.latest = -1;
    for (auto & unit : none.units) unit.latest = false;
    REQUIRE(pflash_query_pin(none, 512, PFlashUnitScore::Max, kInf).spans.empty());
}

TEST_CASE(PFlashQueryCandidatesFixture, token_fingerprint_is_stable) {
    REQUIRE(luce::common::http_detail::pflash_token_fingerprint({}) == "cbf29ce484222325");
    // Fixed encoding of negative and INT_MAX ids (logged as drafter_fingerprint_fnv1a64).
    REQUIRE(luce::common::http_detail::pflash_token_fingerprint({1, -2, 2147483647}) == "45409a0b0f44c5fd");
}

// ---------------------------------------------------------------- gap rule

TEST_CASE(PFlashQueryCandidatesFixture, gap_rule_cuts_at_the_largest_drop) {
    // Units scoring 9, 8.6, 8.2 (a specification) then paste at 2..3: the largest drop is after the third unit.
    std::vector<PFlashRequestMessage> m = {msg(Role::User, 0, 80)};
    auto s1 = filled(80, 2.0f);
    set(s1, 0, 10, 9.0f);
    set(s1, 10, 20, 8.6f);
    set(s1, 20, 30, 8.2f);
    set(s1, 40, 50, 3.0f);
    const auto u = pflash_query_units(cover(80, {10, 20, 30, 40, 50, 60, 70}), m, s1);
    const auto e = pflash_query_extent_units(u, 0, PFlashUnitScore::Max, 0.0f, /*gap=*/true);
    REQUIRE(e.size() == 3u);
    const auto pin = pflash_query_pin(u, 512, PFlashUnitScore::Max, 0.0f, /*gap=*/true);
    REQUIRE(pin.tokens == 30 && pin.spans.size() == 1u && pin.spans[0].end == 30);
    // Fewer than kQueryNullMinUnits units: the whole message.
    std::vector<PFlashRequestMessage> fu = {msg(Role::User, 0, 20)};
    auto fs = filled(20, 9.0f);
    set(fs, 10, 20, 1.0f);
    const auto uf = pflash_query_units(cover(20, {10}), fu, fs);
    REQUIRE(pflash_query_extent_units(uf, 0, PFlashUnitScore::Max, 0.0f, true).size() == 2u);
    PFlashQueryConfig c;
    c.candidates = 1;
    c.extent_gap = true;
    REQUIRE(pflash_query_candidate_set(uf, fs, c).candidates[0].tokens == range(0, 20));
}

TEST_CASE(PFlashQueryCandidatesFixture, blend_shares_tw_and_end) {
    PFlashQueryCandidateSet set;
    set.candidates.resize(2);
    set.candidates[0].score = 2.0f;
    set.candidates[1].score = 2.0f;
    PFlashQueryConfig c;
    c.candidates = 2;
    c.tw_score = false;
    c.tw_share = 0.0f;
    c.end_share = 0.0f;
    // No TW / END: the softmax alone.
    auto b = pflash_query_blend(set, c, 0.0f);
    REQUIRE(b.probe.size() == 2u && b.probe[0] == 0.5 && b.tw == 0.0 && b.end == 0.0);
    // Fixed shares.
    c.tw_share = 0.25f;
    c.end_share = 0.25f;
    b = pflash_query_blend(set, c, 0.0f);
    REQUIRE(std::fabs(b.probe[0] - 0.25) < 1e-7 && std::fabs(b.tw - 0.25) < 1e-7 && std::fabs(b.end - 0.25) < 1e-7);
    // TW by score: it joins the softmax (equal scores -> a third each), END keeps its share.
    c.tw_share = 0.0f;
    c.tw_score = true;
    b = pflash_query_blend(set, c, 2.0f);
    REQUIRE(std::fabs(b.tw - 0.75 / 3.0) < 1e-6 && std::fabs(b.probe[1] - 0.75 / 3.0) < 1e-6 && b.end == 0.25);
    // TW by score without a finite score: out of the blend.
    b = pflash_query_blend(set, c, std::numeric_limits<float>::quiet_NaN());
    REQUIRE(b.tw == 0.0 && std::fabs(b.probe[0] - 0.375) < 1e-7);
}
