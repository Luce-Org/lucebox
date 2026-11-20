#include "CppUnitTestFramework.hpp"

#include "pflash/pflash_request_probe.h"
#include "pflash/pflash_selection.h"
#include "scoped_env.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

using namespace luce::pflash;

namespace {

struct PFlashRequestProbeFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;
};

using Spans = std::vector<PFlashTokenSpan>;

// Logits of ``n`` tokens: 1 inside ``request`` spans, -1 elsewhere.
std::vector<float> logits_with(int n, const Spans & request) {
    std::vector<float> logits((size_t) n, -1.0f);
    for (const auto & span : request) {
        for (int t = span.begin; t < span.end; ++t) logits[(size_t) t] = 1.0f;
    }
    return logits;
}

std::vector<int> range(int begin, int end) {
    std::vector<int> out;
    for (int t = begin; t < end; ++t) out.push_back(t);
    return out;
}

std::vector<float> rows_for(const std::vector<int> & positions, int hidden) {
    std::vector<float> rows;
    for (const int p : positions) {
        for (int k = 0; k < hidden; ++k) rows.push_back((float) p + 0.25f * (float) k);
    }
    return rows;
}

} // namespace

TEST_CASE(PFlashRequestProbeFixture, runs_are_maximal_at_least_two_tokens_inside_the_message) {
    REQUIRE(std::string(kRequestExtractionVersion) == "request_runs_v2_minrun2_ge_raw_logit");
    // Runs [0,1) (1 token, dropped), [2,4) (2 tokens: "why?" survives),
    // [7,12), [14,20) clipped by the message end 18.
    const auto logits = logits_with(30, {{0, 1}, {2, 4}, {7, 12}, {14, 20}});
    REQUIRE(pflash_request_runs(logits, {0, 18}, 0.5f) == Spans({{2, 4}, {7, 12}, {14, 18}}));
    // A run cut to one token by the message start is dropped.
    REQUIRE(pflash_request_runs(logits, {3, 6}, 0.5f).empty());
    // A run cut by the message start keeps only its inside part.
    REQUIRE(pflash_request_runs(logits, {9, 30}, 0.5f) == Spans({{9, 12}, {14, 20}}));
    // logit >= tau counts (equality), NaN breaks a run.
    std::vector<float> edge(12, 0.5f);
    edge[6] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(pflash_request_runs(edge, {0, 12}, 0.5f) == Spans({{0, 6}, {7, 12}}));
    // tau = +inf (the stub) finds nothing; out of range finds nothing.
    REQUIRE(pflash_request_runs(logits, {0, 30}, std::numeric_limits<float>::infinity()).empty());
    REQUIRE(pflash_request_runs(logits, {0, 31}, 0.5f).empty());
    REQUIRE(pflash_request_runs(logits, {5, 5}, 0.5f).empty());
}

TEST_CASE(PFlashRequestProbeFixture, role_rule_takes_the_latest_message_and_the_previous_below_eight) {
    // Previous human message [0,40) with runs [10,20); latest [100,140).
    // Latest with 8 request tokens: latest only.
    {
        const auto logits = logits_with(200, {{10, 20}, {120, 128}});
        const auto set = pflash_request_query_set(logits, {{100, 140}, {0, 40}}, 0.0f);
        REQUIRE(set.fallback.empty());
        REQUIRE(!set.previous_used);
        REQUIRE(set.tokens == range(120, 128));
        REQUIRE(set.latest_tokens == 8);
        REQUIRE(set.previous_tokens == 0);
        REQUIRE(set.messages.size() == 2);
    }
    // Latest with 7 request tokens: the previous message's runs join.
    {
        const auto logits = logits_with(200, {{10, 20}, {120, 127}});
        const auto set = pflash_request_query_set(logits, {{100, 140}, {0, 40}}, 0.0f);
        REQUIRE(set.previous_used);
        std::vector<int> expected = range(10, 20);
        const auto latest = range(120, 127);
        expected.insert(expected.end(), latest.begin(), latest.end());
        REQUIRE(set.tokens == expected);
        REQUIRE(set.latest_tokens == 7);
        REQUIRE(set.previous_tokens == 10);
    }
    // Latest with a 3-token run ("why not?"): kept, and the previous joins.
    {
        const auto logits = logits_with(200, {{10, 20}, {120, 123}});
        const auto set = pflash_request_query_set(logits, {{100, 140}, {0, 40}}, 0.0f);
        REQUIRE(set.previous_used);
        std::vector<int> expected = range(10, 20);
        expected.push_back(120); expected.push_back(121); expected.push_back(122);
        REQUIRE(set.tokens == expected);
        REQUIRE(set.latest_tokens == 3);
    }
    // The < 8 count is taken after run extraction: 9 tokens >= tau in the
    // latest message, but 3 of them are 1-token runs, so 6 request tokens
    // remain and the previous message joins.
    {
        const auto logits = logits_with(200, {{10, 20}, {101, 102}, {103, 104}, {105, 106},
                                              {110, 113}, {120, 123}});
        const auto set = pflash_request_query_set(logits, {{100, 140}, {0, 40}}, 0.0f);
        REQUIRE(set.messages[0].logits_at_or_above_tau == 9);
        REQUIRE(set.messages[0].request_tokens == 6);
        REQUIRE(set.previous_used);
        REQUIRE(set.latest_tokens == 6);
        REQUIRE(set.previous_tokens == 10);
    }
    // Latest without runs (or empty): the previous one alone.
    {
        const auto logits = logits_with(200, {{10, 20}, {120, 121}});
        const auto set = pflash_request_query_set(logits, {{100, 140}, {0, 40}}, 0.0f);
        REQUIRE(set.previous_used);
        REQUIRE(set.tokens == range(10, 20));
        const auto empty = pflash_request_query_set(logits, {{0, 0}, {0, 40}}, 0.0f);
        REQUIRE(empty.previous_used);
        REQUIRE(empty.tokens == range(10, 20));
    }
    // Only the first two messages are read.
    {
        const auto logits = logits_with(300, {{210, 220}});
        const auto set = pflash_request_query_set(logits, {{100, 140}, {0, 40}, {200, 240}}, 0.0f);
        REQUIRE(set.fallback == "no_request_runs");
        REQUIRE(set.messages.size() == 2);
    }
}

TEST_CASE(PFlashRequestProbeFixture, cap_is_1024_tokens_latest_first) {
    // Latest message alone over the cap: its last 1024 tokens.
    {
        const auto logits = logits_with(3000, {{1000, 2100}});
        const auto set = pflash_request_query_set(logits, {{1000, 2200}}, 0.0f);
        REQUIRE(set.capped);
        REQUIRE(set.uncapped_tokens == 1100);
        REQUIRE(set.tokens == range(1076, 2100));
    }
    // Latest short (6 tokens), previous long: previous gets the remaining
    // 1018, from its end.
    {
        const auto logits = logits_with(4000, {{0, 2000}, {3000, 3006}});
        const auto set = pflash_request_query_set(logits, {{2990, 3010}, {0, 2010}}, 0.0f);
        REQUIRE(set.previous_used);
        REQUIRE(set.tokens.size() == 1024);
        REQUIRE(set.latest_tokens == 6);
        REQUIRE(set.previous_tokens == 1018);
        REQUIRE(set.tokens.front() == 2000 - 1018);
        REQUIRE(set.tokens.back() == 3005);
    }
}

TEST_CASE(PFlashRequestProbeFixture, empty_sets_fall_back_with_a_reason) {
    const auto logits = logits_with(50, {});
    REQUIRE(pflash_request_query_set(logits, {}, 0.0f).fallback == "no_human_message");
    REQUIRE(pflash_request_query_set(logits, {{0, 40}}, 0.0f).fallback == "no_request_runs");
    REQUIRE(pflash_request_query_set(logits, {{0, 60}}, 0.0f).fallback == "logits_missing");
    // The stub (tau = +inf) falls back everywhere.
    const auto all = logits_with(50, {{0, 50}});
    REQUIRE(pflash_request_query_set(all, {{0, 50}}, std::numeric_limits<float>::infinity())
                .fallback == "no_request_runs");
}

TEST_CASE(PFlashRequestProbeFixture, turn_tokens_and_spans) {
    PFlashRequestMessageRuns message;
    message.runs = {{10, 14}, {20, 1100}};
    const auto turn = pflash_request_turn_tokens(message);
    REQUIRE(turn.size() == 1024);
    REQUIRE(turn.front() == 1100 - 1024);
    REQUIRE(pflash_request_token_spans({1, 2, 3, 7, 8, 10}) == Spans({{1, 4}, {7, 9}, {10, 11}}));
    REQUIRE(pflash_request_token_spans({}).empty());
}

TEST_CASE(PFlashRequestProbeFixture, bf16_round_trip_rounds_to_nearest_even) {
    REQUIRE(pflash_bf16_to_f32(pflash_f32_to_bf16(1.0f)) == 1.0f);
    REQUIRE(pflash_bf16_to_f32(pflash_f32_to_bf16(-3.5f)) == -3.5f);
    const float x = 1.00390625f;   // 1 + 2^-8: halfway, rounds to even (1.0)
    REQUIRE(pflash_bf16_to_f32(pflash_f32_to_bf16(x)) == 1.0f);
    REQUIRE(std::isnan(pflash_bf16_to_f32(pflash_f32_to_bf16(std::nanf("")))));
    REQUIRE(std::isinf(pflash_bf16_to_f32(pflash_f32_to_bf16(INFINITY))));
}

TEST_CASE(PFlashRequestProbeFixture, row_store_looks_up_exact_positions) {
    PFlashRequestRowStore store;
    const int hidden = 4;
    const auto positions = std::vector<int>{3, 4, 5, 9};
    const auto rows = rows_for(positions, hidden);
    REQUIRE(store.put({0, 10}, positions, rows.data(), hidden) == 0);
    REQUIRE(store.tokens() == 4);
    std::vector<float> out;
    REQUIRE(store.lookup({4, 9}, hidden, out));
    REQUIRE(out == rows_for({4, 9}, hidden));     // exactly representable in bf16
    REQUIRE(!store.lookup({4, 6}, hidden, out)); // 6 is not a request token
    REQUIRE(!store.covers({2}));
    REQUIRE(!store.lookup({4}, hidden + 1, out)); // other width
    // The same message stored again replaces its turn.
    REQUIRE(store.put({0, 10}, {3, 4}, rows.data(), hidden) == 0);
    REQUIRE(store.tokens() == 2);
    REQUIRE(!store.covers({9}));
}

TEST_CASE(PFlashRequestProbeFixture, row_store_evicts_whole_turns_least_recently_used) {
    PFlashRequestRowStore store(/*cap_tokens=*/2048);
    const int hidden = 2;
    const auto a = range(0, 1000), b = range(2000, 2900), c = range(4000, 4500);
    REQUIRE(store.put({0, 1000}, a, rows_for(a, hidden).data(), hidden) == 0);
    REQUIRE(store.put({2000, 2900}, b, rows_for(b, hidden).data(), hidden) == 0);
    // Touch turn a: b becomes the least recently used.
    std::vector<float> out;
    REQUIRE(store.lookup({5}, hidden, out));
    REQUIRE(store.put({4000, 4500}, c, rows_for(c, hidden).data(), hidden) == 1);
    REQUIRE(store.covers({5}));
    REQUIRE(!store.covers({2000}));   // b evicted whole
    REQUIRE(!store.covers({2899}));
    REQUIRE(store.covers({4499}));
    REQUIRE(store.tokens() == 1500);
    // A turn over the cap alone keeps its last cap rows.
    PFlashRequestRowStore small(/*cap_tokens=*/8);
    const auto d = range(0, 10);
    small.put({0, 10}, d, rows_for(d, hidden).data(), hidden);
    REQUIRE(small.tokens() == 8);
    REQUIRE(!small.covers({0, 1}));
    REQUIRE(small.covers({2, 9}));
}

TEST_CASE(PFlashRequestProbeFixture, row_store_invalidates_past_a_divergence_or_truncation) {
    PFlashRequestRowStore store;
    const int hidden = 2;
    const auto a = range(10, 20), b = range(100, 110);
    store.put({0, 30}, a, rows_for(a, hidden).data(), hidden);
    store.put({90, 120}, b, rows_for(b, hidden).data(), hidden);
    // A resumed prompt that shares 200 tokens keeps both.
    REQUIRE(store.invalidate_from(200) == 0);
    // A prompt that diverges inside the second message (an edit, a
    // truncated history) drops that turn whole, not the earlier one.
    REQUIRE(store.invalidate_from(115) == 1);
    REQUIRE(store.covers(a));
    REQUIRE(!store.covers({100}));
    // Diverging before the first message drops everything.
    REQUIRE(store.invalidate_from(0) == 1);
    REQUIRE(store.tokens() == 0);
}

TEST_CASE(PFlashRequestProbeFixture, flag_parses_and_needs_strict_head_selection) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce_test::ScopedEnvVar scorer{"PFLASH_SELECT_SCORER", nullptr};
    luce_test::ScopedEnvVar head{"PFLASH_SELECT_QUERY_HEAD", nullptr};
    {
        luce_test::ScopedEnvVar flag{"PFLASH_SELECT_QUERY_PROBE", nullptr};
        PFlashSelectionConfig config;
        std::string error;
        REQUIRE(resolve_pflash_selection(1000, 32, config, error));
        REQUIRE(!config.query_probe);
        REQUIRE(!pflash_query_probe_requested());
    }
    {
        luce_test::ScopedEnvVar flag{"PFLASH_SELECT_QUERY_PROBE", "1"};
        PFlashSelectionConfig config;
        std::string error;
        REQUIRE(resolve_pflash_selection(1000, 32, config, error));
        REQUIRE(config.query_probe);
        REQUIRE(pflash_query_probe_requested());
    }
    {
        luce_test::ScopedEnvVar flag{"PFLASH_SELECT_QUERY_PROBE", "0"};
        PFlashSelectionConfig config;
        std::string error;
        REQUIRE(resolve_pflash_selection(1000, 32, config, error));
        REQUIRE(!config.query_probe);
    }
    {
        luce_test::ScopedEnvVar flag{"PFLASH_SELECT_QUERY_PROBE", "yes"};
        PFlashSelectionConfig config;
        std::string error;
        REQUIRE(!resolve_pflash_selection(1000, 32, config, error));
    }
    {
        // The served u314 config: query head + latest_user parser.
        luce_test::ScopedEnvVar flag{"PFLASH_SELECT_QUERY_PROBE", "1"};
        luce_test::ScopedEnvVar head_on{"PFLASH_SELECT_QUERY_HEAD", "1"};
        luce_test::ScopedEnvVar parser{"PFLASH_SELECT_QUERY_PARSER", "latest_user"};
        PFlashSelectionConfig config;
        std::string error;
        REQUIRE(resolve_pflash_selection(1000, 32, config, error));
        REQUIRE(config.query_probe && config.query_head);
    }
    {
        luce_test::ScopedEnvVar flag{"PFLASH_SELECT_QUERY_PROBE", "1"};
        luce_test::ScopedEnvVar split{"PFLASH_SELECT_SCORER", "split"};
        PFlashSelectionConfig config;
        std::string error;
        REQUIRE(!resolve_pflash_selection(1000, 32, config, error));
    }
}

// ---- v2 policy (PLAN-V2 W4): calibrated abstention and the bounded recency pin --------------------------------

TEST_CASE(PFlashRequestProbeFixture, default_policy_never_abstains_and_pins_nothing) {
    const PFlashRequestPolicy policy;
    // Weak, fragmented, tiny and huge extractions alike: no abstention.
    const auto logits = logits_with(300000, {{10, 12}, {14, 16}, {18, 20}, {100, 110}});
    for (const auto & msgs : std::vector<Spans>{{{100, 140}, {0, 40}}, {{0, 40}}, {{5, 300000}}}) {
        const auto set = pflash_request_query_set(logits, msgs, 0.9f);
        if (!set.fallback.empty()) continue;
        for (const int prompt : {1, 1000, 300000}) {
            REQUIRE(pflash_request_abstain(set, logits, 0.9f, prompt, policy).reason.empty());
        }
    }
    REQUIRE(pflash_request_recency_pin({100, 140}, policy.recency_pin_tokens).begin == -1);
}

TEST_CASE(PFlashRequestProbeFixture, abstention_criteria_and_statistics) {
    // logits 1 inside the runs, -1 elsewhere; tau 0.5 -> every query token has margin +0.5.
    const auto logits = logits_with(200, {{10, 20}, {120, 123}, {125, 128}});
    const auto set = pflash_request_query_set(logits, {{100, 140}, {0, 40}}, 0.5f);
    REQUIRE(set.fallback.empty());
    PFlashRequestPolicy p;
    auto s = pflash_request_abstain(set, logits, 0.5f, 200, p);
    REQUIRE(s.reason.empty());
    REQUIRE(std::fabs(s.mean_margin - 0.5f) < 1e-6f);
    REQUIRE(std::fabs(s.max_margin - 0.5f) < 1e-6f);
    REQUIRE(std::fabs(s.latest_share - 6.0f / 40.0f) < 1e-6f);   // 6 request tokens in a 40-token message
    REQUIRE(s.runs == 3);                                           // 2 latest + 1 previous (previous used)
    REQUIRE(s.prompt_tokens == 200);
    p.max_prompt_tokens = 199;
    REQUIRE(pflash_request_abstain(set, logits, 0.5f, 200, p).reason == "length");
    p = {}; p.min_prompt_tokens = 201;
    REQUIRE(pflash_request_abstain(set, logits, 0.5f, 200, p).reason == "length");
    p = {}; p.min_mean_margin = 0.6f;
    REQUIRE(pflash_request_abstain(set, logits, 0.5f, 200, p).reason == "mean_margin");
    p = {}; p.min_max_margin = 0.6f;
    REQUIRE(pflash_request_abstain(set, logits, 0.5f, 200, p).reason == "max_margin");
    p = {}; p.min_latest_share = 0.2f;
    REQUIRE(pflash_request_abstain(set, logits, 0.5f, 200, p).reason == "latest_share");
    p = {}; p.max_latest_share = 0.1f;
    REQUIRE(pflash_request_abstain(set, logits, 0.5f, 200, p).reason == "latest_share");
    p = {}; p.max_runs = 2;
    REQUIRE(pflash_request_abstain(set, logits, 0.5f, 200, p).reason == "runs");
    p.max_runs = 3;
    REQUIRE(pflash_request_abstain(set, logits, 0.5f, 200, p).reason.empty());
    // Order: length is reported before the margins.
    p = {}; p.max_prompt_tokens = 10; p.min_mean_margin = 9.0f;
    REQUIRE(pflash_request_abstain(set, logits, 0.5f, 200, p).reason == "length");
}

TEST_CASE(PFlashRequestProbeFixture, recency_pin_is_the_latest_message_tail) {
    REQUIRE(pflash_request_recency_pin({100, 140}, 8) == PFlashTokenSpan({132, 140}));
    REQUIRE(pflash_request_recency_pin({100, 140}, 400) == PFlashTokenSpan({100, 140}));   // bounded by the message
    REQUIRE(pflash_request_recency_pin({100, 140}, 0).begin == -1);
    REQUIRE(pflash_request_recency_pin({0, 0}, 8).begin == -1);                            // empty message
}

TEST_CASE(PFlashRequestProbeFixture, policy_json_overrides_validate_and_fail_closed) {
    PFlashRequestPolicy p;
    std::string error;
    REQUIRE(pflash_request_policy_apply_json(
        R"({"min_mean_margin": 0.25, "min_max_margin": "-inf", "max_runs": 6, "recency_pin_tokens": 32,
            "min_latest_share": 0.01, "max_latest_share": 0.9, "max_prompt_tokens": 131072})", p, error));
    REQUIRE(p.min_mean_margin == 0.25f);
    REQUIRE(std::isinf(p.min_max_margin));
    REQUIRE(p.max_runs == 6);
    REQUIRE(p.recency_pin_tokens == 32);
    REQUIRE(p.max_prompt_tokens == 131072);
    const PFlashRequestPolicy before = p;
    REQUIRE(!pflash_request_policy_apply_json(R"({"min_mean_margn": 1})", p, error));      // unknown key
    REQUIRE(!pflash_request_policy_apply_json(R"({"max_runs": 1.5})", p, error));          // wrong type
    REQUIRE(!pflash_request_policy_apply_json(R"({"min_latest_share": 0.8, "max_latest_share": 0.2})", p, error));
    REQUIRE(!pflash_request_policy_apply_json(R"({"recency_pin_tokens": -1})", p, error));
    REQUIRE(!pflash_request_policy_apply_json(R"([1, 2])", p, error));
    REQUIRE(!pflash_request_policy_apply_json("{", p, error));
    REQUIRE(p.max_runs == before.max_runs && p.recency_pin_tokens == before.recency_pin_tokens);   // unchanged on failure
}

// V2-A3 candidate rule = the trainer's rule.probe_query_v2. Candidates are
// built as the server builds them: the latest two human messages and the last
// non-system messages (all of them, V2-A5), by message index.
static PFlashRequestQuerySet v2_set(const std::vector<std::string> & roles, const Spans & spans,
                                    const std::vector<float> & logits) {
    std::vector<PFlashRequestCandidate> candidates;
    std::vector<int> humans;
    for (int i = (int) roles.size() - 1; i >= 0 && humans.size() < 2; --i) {
        if (roles[(size_t) i] == "user") humans.push_back(i);
    }
    for (int i : humans) candidates.push_back({i, true, false, spans[(size_t) i]});
    for (int i = (int) roles.size() - 1; i >= 0; --i) {
        if (roles[(size_t) i] == "system") continue;
        bool merged = false;
        for (auto & c : candidates) {
            if (c.message_index == i) { c.recent = true; merged = true; }
        }
        if (!merged) candidates.push_back({i, false, true, spans[(size_t) i]});
    }
    return pflash_request_query_set_v2(logits, candidates, 0.0f);
}

TEST_CASE(PFlashRequestProbeFixture, v2_rule_matches_the_trainer_mirror_example) {
    // luce_box tests/test_request_probe_v2a3.py::test_probe_query_v2_mirror
    const std::vector<std::string> roles = {"system", "user", "assistant", "tool", "assistant", "tool"};
    const Spans spans = {{0, 5}, {5, 30}, {30, 40}, {40, 60}, {60, 70}, {70, 90}};
    const auto logits = logits_with(95, {{10, 14}, {44, 47}, {62, 64}, {32, 36}});
    const auto set = v2_set(roles, spans, logits);
    REQUIRE(set.fallback.empty());
    // V2-A5: every non-system message is a candidate (the assistant at 30-40 too).
    REQUIRE(set.tokens == std::vector<int>({10, 11, 12, 13, 32, 33, 34, 35, 44, 45, 46, 62, 63}));
    REQUIRE(set.turn_messages == std::vector<int>({5, 4, 3, 2}));
    REQUIRE(std::string(kRequestCandidateRule) == "cand_v3_latest_human_prev8_plus_all_nonsystem_turns");
}

namespace {
struct V2Fixture {
    int n;
    std::vector<std::string> roles;
    Spans spans;
    Spans hits;
    Spans query;
    std::vector<int> candidates;
};
const std::vector<V2Fixture> kV2Fixtures = {
#include "fixtures/request_probe_v2_parity.inc"
};
} // namespace

TEST_CASE(PFlashRequestProbeFixture, v2_rule_replay_parity_with_the_trainer_rule) {
    REQUIRE(kV2Fixtures.size() == 80);   // regenerated for cand_v3
    for (const auto & f : kV2Fixtures) {
        const auto set = v2_set(f.roles, f.spans, logits_with(f.n, f.hits));
        std::vector<int> want;
        for (const auto & q : f.query) for (int t = q.begin; t < q.end; ++t) want.push_back(t);
        REQUIRE(set.tokens == want);
        std::vector<int> read;
        for (const auto & m : set.messages) {
            for (size_t i = 0; i < f.spans.size(); ++i) {
                if (f.spans[i] == m.message && f.roles[i] == "user") { read.push_back((int) i); break; }
            }
        }
        read.insert(read.end(), set.turn_messages.begin(), set.turn_messages.end());
        std::sort(read.begin(), read.end());
        read.erase(std::unique(read.begin(), read.end()), read.end());
        REQUIRE(read == f.candidates);
        REQUIRE(set.tokens.empty() == (set.fallback == "no_request_runs" || set.fallback == "no_human_message"));
    }
}
