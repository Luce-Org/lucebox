// Query step 1 (pflash_query_candidates.h): relative candidate ranking, NULL, extents, aggregation, the verbatim
// pin, the flags (fail closed), the dev-only grid / oracle parsers and the fingerprint. Model-free.

#include "CppUnitTestFramework.hpp"

#include "pflash/pflash_query_candidates.h"
#include "pflash/pflash_selection.h"
#include "scoped_env.h"
#include "server/http_server.h"
#include "server/launch_profiles.h"

#include <cmath>
#include <cstdlib>
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

const char * kQueryEnvs[] = {"PFLASH_QUERY_CANDIDATES", "PFLASH_QUERY_PIN", "PFLASH_QUERY_UNIT_SCORE",
                             "PFLASH_QUERY_AGG", "PFLASH_QUERY_TEMP", "PFLASH_QUERY_NULL_MARGIN",
                             "PFLASH_QUERY_EXTENT", "PFLASH_QUERY_PIN_MARGIN", "PFLASH_QUERY_TW",
                             "PFLASH_QUERY_END"};

std::vector<std::unique_ptr<luce_test::ScopedEnvVar>> clean_startup_env() {
    std::vector<std::unique_ptr<luce_test::ScopedEnvVar>> vars;
    for (const char * name : kQueryEnvs) {
        vars.push_back(std::make_unique<luce_test::ScopedEnvVar>(name, nullptr));
    }
    for (const char * name : {
            "PFLASH_SELECT_MODE", "PFLASH_SELECT_CHUNK_SIZE", "PFLASH_SELECT_QUERY_TOKENS",
            "PFLASH_SELECT_QUERY_PARSER", "PFLASH_SELECT_TOP_P", "PFLASH_SELECT_TOPK",
            "PFLASH_SELECT_SEGMENTS", "PFLASH_SELECT_SCORE", "PFLASH_SELECT_SCORER",
            "PFLASH_SELECT_SPLIT", "PFLASH_SELECT_STRATEGY", "PFLASH_SELECT_QUERY_HEAD",
            "PFLASH_SELECT_QUERY_PROBE", "PFLASH_EVAL_REF_IDENT_QUERY", "PFLASH_FALLBACK_SPREAD",
            "PFLASH_CHAT_VIEW", "PFLASH_CHAT_RECALL", "PFLASH_SELECT_PARAGRAPH_JOIN",
            "PFLASH_AGENT_PASTE", "PFLASH_ARRIVE", "PFLASH_FAIL_CLOSED", "PFLASH_ADMISSION",
            "PFLASH_CHAT_SKELETON_TOKENS", "PFLASH_CHAT_COMPRESS_NEW_TOKENS",
            "PFLASH_CHAT_HISTORY_QUERIES", "PFLASH_CHAT_REBUILD_MIN_TOKENS",
            "PFLASH_CHAT_REBUILD_RATIO", "PFLASH_DRAFTER_SESSIONS", "PFLASH_DRAFTER_TRIM_POOL",
            "PFLASH_SESSION_OFFLOAD_PINNED", "PFLASH_QWEN35_LEGACY_SCORER",
            "PFLASH_SELECT_QUERY_PROBE_STUB", "PFLASH_QUERY_GRID", "PFLASH_QUERY_GRID_OUT",
            "PFLASH_REQUEST_STAGE2_GGUF", "PFLASH_SEGMENT_PROBE_GGUF", "PFLASH_SCORING_HEAD_GGUF",
            "PFLASH_REQUEST_PROBE_GGUF"}) {
        vars.push_back(std::make_unique<luce_test::ScopedEnvVar>(name, nullptr));
    }
    return vars;
}

} // namespace

// ---------------------------------------------------------------- flags

TEST_CASE(PFlashQueryCandidatesFixture, startup_validation_preserves_defaults_and_measured_profile) {
    const auto clean = clean_startup_env();
    using namespace luce::common::http_detail;
    std::string error;
    REQUIRE(validate_pflash_environment(error));
    REQUIRE(pflash_chat_compress_new_tokens() == 16384);
    REQUIRE(pflash_chat_skeleton_tokens() == 256 && pflash_chat_history_queries() == 3);
    REQUIRE(pflash_chat_rebuild_ratio() == 2.0 && pflash_chat_rebuild_min_tokens() == 0);
    REQUIRE(pflash_chat_recall() && pflash_paragraph_join());
    REQUIRE(!pflash_arrive_enabled() && !pflash_agent_paste_enabled() && !pflash_cache_admission_enabled());
    REQUIRE(!pflash_fail_closed_from_env());
    PFlashSelectionConfig selection;
    REQUIRE(resolve_pflash_selection(65536, 32, selection, error));
    REQUIRE(selection.mode == PFlashSelectionMode::Legacy && selection.query_tokens == 8);
    REQUIRE(selection.fallback_spread == 0.0 && !selection.query.any());

    // Release overrides from the measured PFlash configuration remain valid.
    std::vector<std::unique_ptr<luce_test::ScopedEnvVar>> release;
    for (const auto & flag : std::vector<std::pair<const char *, const char *>>{
            {"PFLASH_SELECT_MODE", "budget_only"}, {"PFLASH_SELECT_SEGMENTS", "probe"},
            {"PFLASH_SELECT_SCORE", "density"}, {"PFLASH_SELECT_QUERY_TOKENS", "128"},
            {"PFLASH_SELECT_CHUNK_SIZE", "256"}, {"PFLASH_SELECT_QUERY_HEAD", "1"},
            {"PFLASH_SELECT_PARAGRAPH_JOIN", "0"}, {"PFLASH_SELECT_QUERY_PROBE", "1"},
            {"PFLASH_SEGMENT_PROBE_GGUF", "segment.gguf"}, {"PFLASH_QUERY_CANDIDATES", "1"},
            {"PFLASH_QUERY_PIN", "512"}, {"PFLASH_ADMISSION", "cache"}, {"PFLASH_ARRIVE", "1"},
            {"PFLASH_CHAT_RECALL", "1"}, {"PFLASH_CHAT_REBUILD_RATIO", "2"},
            {"PFLASH_CHAT_REBUILD_MIN_TOKENS", "65536"}, {"PFLASH_FALLBACK_SPREAD", "0.11"}}) {
        release.push_back(std::make_unique<luce_test::ScopedEnvVar>(flag.first, flag.second));
    }
    REQUIRE(validate_pflash_environment(error));
    REQUIRE(resolve_pflash_selection(65536, 32, selection, error));
    REQUIRE(selection.query_tokens == 128 && selection.chunk_size == 256);
    REQUIRE(selection.query.candidates == 1 && selection.query.pin_tokens == 512);
    REQUIRE(selection.query.tw_share == 0.5f && selection.query.end_share == 0.25f);
    REQUIRE(selection.query.extent_gap && selection.query.pin_gap && selection.fallback_spread == 0.11);
    REQUIRE(pflash_chat_rebuild_min_tokens() == 65536 && pflash_arrive_enabled());
    // Valid large overrides retain the existing caps.
    luce_test::ScopedEnvVar history{"PFLASH_CHAT_HISTORY_QUERIES", "32"};
    REQUIRE(validate_pflash_environment(error));
    REQUIRE(pflash_chat_history_queries() == 8);
    // Fold and replay / oracle tooling are still accepted.
    luce_test::ScopedEnvVar fold{"PFLASH_SELECT_STRATEGY", "fold"};
    luce_test::ScopedEnvVar grid{"PFLASH_QUERY_GRID", "grid.json"};
    luce_test::ScopedEnvVar output{"PFLASH_QUERY_GRID_OUT", "grid.jsonl"};
    luce_test::ScopedEnvVar oracle{"PFLASH_QUERY_ORACLE_DIR", "oracle"};
    REQUIRE(validate_pflash_environment(error));
}

TEST_CASE(PFlashQueryCandidatesFixture, release_profile_is_the_default_with_a_trained_head) {
    const auto clean = clean_startup_env();
    using namespace luce::common::http_detail;
    using luce::server::pflash_release_env;
    using Vars = std::vector<std::unique_ptr<luce_test::ScopedEnvVar>>;
    const auto has = [](const char * name) {
        for (const auto & env : pflash_release_env()) {
            if (env.name == name) return true;
        }
        return false;
    };
    // As the server installs it: a variable that is already set keeps its value.
    const auto install = [](Vars & vars) {
        for (const auto & env : pflash_release_env()) {
            if (!std::getenv(env.name.c_str())) vars.push_back(std::make_unique<luce_test::ScopedEnvVar>(env.name.c_str(), env.value.c_str()));
        }
    };
    std::string error;
    PFlashSelectionConfig selection;
    REQUIRE(pflash_release_env().empty());

    luce_test::ScopedEnvVar head{"PFLASH_SCORING_HEAD_GGUF", "head.gguf"};
    REQUIRE(has("PFLASH_SELECT_MODE") && !has("PFLASH_SELECT_SEGMENTS") && !has("PFLASH_SELECT_QUERY_PROBE"));
    {
        Vars vars;
        install(vars);
        REQUIRE(validate_pflash_environment(error));
    }

    luce_test::ScopedEnvVar segments{"PFLASH_SEGMENT_PROBE_GGUF", "segment.gguf"};
    luce_test::ScopedEnvVar request{"PFLASH_REQUEST_PROBE_GGUF", "request.gguf"};
    {
        Vars vars;
        install(vars);
        REQUIRE(validate_pflash_environment(error));
        REQUIRE(resolve_pflash_selection(65536, 32, selection, error));
        REQUIRE(selection.configured && selection.mode == PFlashSelectionMode::BudgetOnly);
        REQUIRE(selection.segmentation == PFlashSegmentation::Probe &&
                selection.candidate_score == PFlashCandidateScore::Density);
        REQUIRE(selection.chunk_size == 256 && selection.query_tokens == 128);
        REQUIRE(selection.query_head && selection.query_probe && selection.fallback_spread == 0.11);
        REQUIRE(selection.query.candidates == 1 && selection.query.pin_tokens == 512);
        REQUIRE(has_pflash_selection_environment() && pflash_query_probe_requested());
        REQUIRE(pflash_cache_admission_enabled() && pflash_arrive_enabled() && !pflash_paragraph_join());
        REQUIRE(pflash_chat_rebuild_min_tokens() == 65536 && pflash_chat_rebuild_ratio() == 2.0);
        REQUIRE(pflash_chat_recall());
    }
    {
        luce_test::ScopedEnvVar spread{"PFLASH_FALLBACK_SPREAD", "off"};
        luce_test::ScopedEnvVar admission{"PFLASH_ADMISSION", "length"};
        Vars vars;
        install(vars);
        REQUIRE(validate_pflash_environment(error));
        REQUIRE(resolve_pflash_selection(65536, 32, selection, error));
        REQUIRE(selection.fallback_spread == 0.0 && !pflash_cache_admission_enabled());
    }
    // Arms that replace the request-probe query or exclude its candidates still start.
    for (const auto & arm : std::vector<std::pair<const char *, const char *>>{
            {"PFLASH_SELECT_SCORER", "legacy"}, {"PFLASH_EVAL_REF_IDENT_QUERY", "heur"},
            {"PFLASH_REQUEST_STAGE2_GGUF", "stage2.gguf"}, {"PFLASH_SELECT_QUERY_PROBE", "0"}}) {
        luce_test::ScopedEnvVar flag{arm.first, arm.second};
        Vars vars;
        install(vars);
        REQUIRE(validate_pflash_environment(error));
        REQUIRE(resolve_pflash_selection(65536, 32, selection, error));
        REQUIRE(!has("PFLASH_QUERY_CANDIDATES") && !selection.query.any());
    }

    REQUIRE(luce::server::kPFlashReleaseThreshold == 16384);
    const auto & curve = luce::server::pflash_release_curve();
    REQUIRE(curve.size() == 4 && curve.front() == std::make_pair(16384, 0.315f) &&
            curve.back() == std::make_pair(120000, 0.225f));
}

TEST_CASE(PFlashQueryCandidatesFixture, startup_validation_rejects_invalid_and_incompatible_flags) {
    const auto clean = clean_startup_env();
    using luce::common::http_detail::validate_pflash_environment;
    std::string error;
    for (const auto & bad : std::vector<std::pair<const char *, const char *>>{
            {"PFLASH_SELECT_MODE", "budget"}, {"PFLASH_SELECT_QUERY_PROBE", "true"},
            {"PFLASH_ADMISSION", "cached"}, {"PFLASH_CHAT_RECALL", "false"},
            {"PFLASH_FAIL_CLOSED", "yes"}, {"PFLASH_AGENT_PASTE", "on"},
            {"PFLASH_CHAT_COMPRESS_NEW_TOKENS", "0"}, {"PFLASH_DRAFTER_SESSIONS", "-1"},
            {"PFLASH_CHAT_SKELETON_TOKENS", "9999999999999999999999999"},
            {"PFLASH_CHAT_HISTORY_QUERIES", "3oops"}, {"PFLASH_CHAT_REBUILD_RATIO", "nan"},
            {"PFLASH_CHAT_REBUILD_RATIO", "1e-9999"}}) {
        luce_test::ScopedEnvVar flag{bad.first, bad.second};
        REQUIRE(!validate_pflash_environment(error));
        REQUIRE(error.find(bad.first) != std::string::npos);
    }
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce_test::ScopedEnvVar probe{"PFLASH_SELECT_QUERY_PROBE", "1"};
    luce_test::ScopedEnvVar candidates{"PFLASH_QUERY_CANDIDATES", "1"};
    REQUIRE(!validate_pflash_environment(error));
    REQUIRE(error.find("PFLASH_SEGMENT_PROBE_GGUF") != std::string::npos);
    luce_test::ScopedEnvVar segments{"PFLASH_SEGMENT_PROBE_GGUF", "segment.gguf"};
    {
        luce_test::ScopedEnvVar tiny{"PFLASH_QUERY_NULL_MARGIN", "1e-9999"};
        REQUIRE(!validate_pflash_environment(error));
        REQUIRE(error.find("PFLASH_QUERY_NULL_MARGIN") != std::string::npos);
    }
    {
        luce_test::ScopedEnvVar grid{"PFLASH_QUERY_GRID", "grid.json"};
        REQUIRE(!validate_pflash_environment(error));
        REQUIRE(error.find("PFLASH_QUERY_GRID_OUT") != std::string::npos);
    }
    luce_test::ScopedEnvVar stage2{"PFLASH_REQUEST_STAGE2_GGUF", "stage2.gguf"};
    REQUIRE(!validate_pflash_environment(error));
    REQUIRE(error.find("PFLASH_REQUEST_STAGE2_GGUF") != std::string::npos);
}

TEST_CASE(PFlashQueryCandidatesFixture, dev_chosen_defaults_apply_only_when_switched_on) {
    // DEFAULTS.md: k1-gap-tw50-end25-soft-pgap knobs; the switches stay off.
    PFlashQueryConfig c;
    REQUIRE(c.candidates == 0 && c.pin_tokens == 0 && !c.any());
    REQUIRE(c.extent_gap && c.pin_gap && c.tw_share == 0.5f && c.end_share == 0.25f && !c.tw_score);
    REQUIRE(c.unit_score == PFlashUnitScore::Max && c.aggregate == PFlashQueryAggregate::Softmax);
    // A grid config starts from the neutral knobs, never from these defaults.
    std::vector<PFlashGridConfig> g;
    std::string error;
    REQUIRE(pflash_query_grid_parse(R"({"configs":[{"name":"S1","mode":"s1"},{"name":"k","mode":"cand","k":1}]})", g,
                                    error));
    REQUIRE(g[1].query.tw_share == 0.0f && g[1].query.end_share == 0.0f && !g[1].query.extent_gap && !g[0].query.pin_gap);
    // An explicit numeric extent / pin margin replaces the gap rule.
    luce_test::ScopedEnvVar k{"PFLASH_QUERY_CANDIDATES", "1"};
    luce_test::ScopedEnvVar e{"PFLASH_QUERY_EXTENT", "2"};
    luce_test::ScopedEnvVar p{"PFLASH_QUERY_PIN", "512"};
    luce_test::ScopedEnvVar pm{"PFLASH_QUERY_PIN_MARGIN", "4"};
    PFlashQueryConfig q;
    REQUIRE(pflash_query_config_from_env(q, error));
    REQUIRE(!q.extent_gap && q.extent_margin == 2.0f && !q.pin_gap && q.pin_margin == 4.0f && q.tw_share == 0.5f);
}

TEST_CASE(PFlashQueryCandidatesFixture, flags_are_off_by_default_and_fail_closed) {
    std::vector<std::unique_ptr<luce_test::ScopedEnvVar>> clear;
    for (const char * name : kQueryEnvs) clear.push_back(std::make_unique<luce_test::ScopedEnvVar>(name, nullptr));
    {
        PFlashQueryConfig c;
        std::string error;
        REQUIRE(pflash_query_config_from_env(c, error));
        REQUIRE(!c.any());
        REQUIRE(c.candidates == 0 && c.pin_tokens == 0);
    }
    {
        luce_test::ScopedEnvVar k{"PFLASH_QUERY_CANDIDATES", "3"};
        luce_test::ScopedEnvVar t{"PFLASH_QUERY_TEMP", "0.5"};
        luce_test::ScopedEnvVar a{"PFLASH_QUERY_AGG", "wmax"};
        luce_test::ScopedEnvVar u{"PFLASH_QUERY_UNIT_SCORE", "mean"};
        luce_test::ScopedEnvVar e{"PFLASH_QUERY_EXTENT", "2"};
        luce_test::ScopedEnvVar n{"PFLASH_QUERY_NULL_MARGIN", "3"};
        luce_test::ScopedEnvVar pm{"PFLASH_QUERY_PIN_MARGIN", "inf"};
        PFlashQueryConfig c;
        std::string error;
        REQUIRE(pflash_query_config_from_env(c, error));
        REQUIRE(c.candidates == 3 && c.temperature == 0.5f && c.aggregate == PFlashQueryAggregate::WeightedMax);
        REQUIRE(c.unit_score == PFlashUnitScore::Mean && c.extent_margin == 2.0f && c.null_margin == 3.0f);
        REQUIRE(std::isinf(c.pin_margin));
    }
    {
        luce_test::ScopedEnvVar p{"PFLASH_QUERY_PIN", "512"};
        luce_test::ScopedEnvVar pm{"PFLASH_QUERY_PIN_MARGIN", "2.5"};
        PFlashQueryConfig c;
        std::string error;
        REQUIRE(pflash_query_config_from_env(c, error));
        REQUIRE(c.pin_tokens == 512 && c.candidates == 0 && c.pin_margin == 2.5f && c.any());
    }
    // A knob without its switch, or an invalid value, fails.
    for (const auto & bad : std::vector<std::pair<const char *, const char *>>{
             {"PFLASH_QUERY_TEMP", "1"}, {"PFLASH_QUERY_EXTENT", "1"}, {"PFLASH_QUERY_PIN_MARGIN", "1"}}) {
        luce_test::ScopedEnvVar v{bad.first, bad.second};
        PFlashQueryConfig c;
        std::string error;
        REQUIRE(!pflash_query_config_from_env(c, error));
        REQUIRE(!error.empty());
    }
    for (const auto & bad : std::vector<std::pair<const char *, const char *>>{
             {"PFLASH_QUERY_TEMP", "0"}, {"PFLASH_QUERY_TEMP", "x"}, {"PFLASH_QUERY_AGG", "mean"},
             {"PFLASH_QUERY_UNIT_SCORE", "sum"}, {"PFLASH_QUERY_NULL_MARGIN", "-1"}, {"PFLASH_QUERY_EXTENT", "-0.5"},
             {"PFLASH_QUERY_PIN_MARGIN", "-1"}}) {
        luce_test::ScopedEnvVar k{"PFLASH_QUERY_CANDIDATES", "2"};
        luce_test::ScopedEnvVar v{bad.first, bad.second};
        PFlashQueryConfig c;
        std::string error;
        REQUIRE(!pflash_query_config_from_env(c, error));
    }
    for (const char * bad : {"9", "-1", "two", ""}) {
        luce_test::ScopedEnvVar k{"PFLASH_QUERY_CANDIDATES", bad};
        PFlashQueryConfig c;
        std::string error;
        REQUIRE(!pflash_query_config_from_env(c, error));
    }
    {
        luce_test::ScopedEnvVar p{"PFLASH_QUERY_PIN", "5000"};
        PFlashQueryConfig c;
        std::string error;
        REQUIRE(!pflash_query_config_from_env(c, error));
    }
}

TEST_CASE(PFlashQueryCandidatesFixture, selection_needs_the_request_probe) {
    luce_test::ScopedEnvVar mode{"PFLASH_SELECT_MODE", "budget_only"};
    luce_test::ScopedEnvVar scorer{"PFLASH_SELECT_SCORER", nullptr};
    luce_test::ScopedEnvVar head{"PFLASH_SELECT_QUERY_HEAD", nullptr};
    luce_test::ScopedEnvVar k{"PFLASH_QUERY_CANDIDATES", "3"};
    {
        luce_test::ScopedEnvVar probe{"PFLASH_SELECT_QUERY_PROBE", nullptr};
        PFlashSelectionConfig config;
        std::string error;
        REQUIRE(!resolve_pflash_selection(1000, 32, config, error));
        REQUIRE(error.find("PFLASH_SELECT_QUERY_PROBE") != std::string::npos);
    }
    {
        luce_test::ScopedEnvVar probe{"PFLASH_SELECT_QUERY_PROBE", "1"};
        PFlashSelectionConfig config;
        std::string error;
        REQUIRE(resolve_pflash_selection(1000, 32, config, error));
        REQUIRE(config.query.candidates == 3);
    }
    {
        // Flags off: the config carries nothing (byte-identical default).
        luce_test::ScopedEnvVar off{"PFLASH_QUERY_CANDIDATES", nullptr};
        luce_test::ScopedEnvVar probe{"PFLASH_SELECT_QUERY_PROBE", "1"};
        PFlashSelectionConfig config;
        std::string error;
        REQUIRE(resolve_pflash_selection(1000, 32, config, error));
        REQUIRE(!config.query.any());
    }
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

TEST_CASE(PFlashQueryCandidatesFixture, no_unit_gives_no_candidate) {
    PFlashQueryUnits u;
    PFlashQueryConfig c;
    c.candidates = 3;
    REQUIRE(pflash_query_candidate_set(u, {}, c).candidates.empty());
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

// ---------------------------------------------------------------- grid, oracle, fingerprint

TEST_CASE(PFlashQueryCandidatesFixture, grid_and_oracle_parse_fail_closed) {
    std::vector<PFlashGridConfig> g;
    std::string error;
    REQUIRE(pflash_query_grid_parse(
        R"({"configs":[{"name":"S1","mode":"s1"},{"name":"P1","mode":"p1","pin":512},
            {"name":"c3","mode":"cand","k":3,"unit_score":"mean","agg":"wmax","temp":0.5,"null_margin":3,
             "extent":2,"pin":256,"pin_margin":"inf"},{"name":"O","mode":"oracle","oracle_pin":false}]})", g, error));
    REQUIRE(g.size() == 4u);
    REQUIRE(g[1].mode == PFlashGridMode::P1 && g[1].query.pin_tokens == 512);
    REQUIRE(g[2].query.candidates == 3 && g[2].query.unit_score == PFlashUnitScore::Mean &&
            g[2].query.aggregate == PFlashQueryAggregate::WeightedMax && g[2].query.extent_margin == 2.0f &&
            std::isinf(g[2].query.pin_margin));
    REQUIRE(g[3].mode == PFlashGridMode::Oracle && !g[3].oracle_pin);
    for (const char * bad : {
             R"({"configs":[]})", R"({"configs":[{"name":"a","mode":"cand"}]})",
             R"({"configs":[{"name":"a","mode":"s1","k":2}]})", R"({"configs":[{"name":"a","mode":"zz"}]})",
             R"({"configs":[{"name":"a","mode":"s1","typo":1}]})",
             R"({"configs":[{"name":"a","mode":"s1"},{"name":"a","mode":"p1"}]})",
             R"({"configs":[{"name":"a","mode":"cand","k":2,"temp":0}]})", "not json"}) {
        REQUIRE(!pflash_query_grid_parse(bad, g, error));
    }
    PFlashQueryOracle o;
    REQUIRE(pflash_query_oracle_parse(R"({"query":[[5,8],[2,4]],"pin":[[2,4],[3,9]]})", 10, o, error));
    REQUIRE(o.query == std::vector<int>({2, 3, 5, 6, 7}));
    REQUIRE(o.pin.size() == 1u && o.pin[0].begin == 2 && o.pin[0].end == 9);
    REQUIRE(!pflash_query_oracle_parse(R"({"query":[[5,12]],"pin":[]})", 10, o, error));
    REQUIRE(!pflash_query_oracle_parse(R"({"query":[],"pin":[]})", 10, o, error));
    REQUIRE(!pflash_query_oracle_parse(R"({"pin":[]})", 10, o, error));
}

TEST_CASE(PFlashQueryCandidatesFixture, fingerprint_matches_the_server) {
    const std::vector<int32_t> ids = {248045, 74455, 198, 1, 2, 3, -1, 0};
    REQUIRE(pflash_ids_fingerprint(ids) == luce::common::http_detail::pflash_token_fingerprint(ids));
    REQUIRE(pflash_ids_fingerprint({}) == "cbf29ce484222325");
}

// ---------------------------------------------------------------- defect (a): ceiling retry

TEST_CASE(PFlashQueryCandidatesFixture, ceiling_retry_only_tightens) {
    // The E-gate failure: ceiling 11567, overflows 17 then 1. The old retry restarted from the requested ratio
    // (0.337334 then 0.337801: looser); the cumulative one tightens.
    const float requested = 0.33787f;
    int tightening = 0;
    const float r1 = luce::common::http_detail::pflash_ceiling_retry_keep_ratio(requested, 11567, 17, tightening);
    REQUIRE(tightening == 18);
    REQUIRE(std::fabs(r1 - requested * 11549.0f / 11567.0f) < 1e-7f);   // the first retry is unchanged
    const float r2 = luce::common::http_detail::pflash_ceiling_retry_keep_ratio(requested, 11567, 1, tightening);
    REQUIRE(tightening == 20);
    REQUIRE(r2 < r1);
    int t = 0;
    REQUIRE(luce::common::http_detail::pflash_ceiling_retry_keep_ratio(requested, 10, 12, t) <= 0.0f);
}

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
    std::vector<PFlashGridConfig> g;
    std::string error;
    REQUIRE(pflash_query_grid_parse(R"({"configs":[{"name":"a","mode":"cand","k":1,"extent":"gap","pin":512,
        "pin_margin":"gap"}]})", g, error));
    REQUIRE(g[0].query.extent_gap && g[0].query.pin_gap);
}

TEST_CASE(PFlashQueryCandidatesFixture, blend_shares_tw_and_end) {
    PFlashQueryCandidateSet set;
    set.candidates.resize(2);
    set.candidates[0].score = 2.0f;
    set.candidates[1].score = 2.0f;
    PFlashQueryConfig c;
    c.candidates = 2;
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
    // Flags: TW / END need the candidates; shares < 1.
    std::vector<std::unique_ptr<luce_test::ScopedEnvVar>> clear;
    for (const char * name : kQueryEnvs) clear.push_back(std::make_unique<luce_test::ScopedEnvVar>(name, nullptr));
    {
        luce_test::ScopedEnvVar p{"PFLASH_QUERY_PIN", "512"};
        luce_test::ScopedEnvVar t{"PFLASH_QUERY_TW", "0.3"};
        PFlashQueryConfig q;
        std::string error;
        REQUIRE(!pflash_query_config_from_env(q, error));
    }
    {
        luce_test::ScopedEnvVar k{"PFLASH_QUERY_CANDIDATES", "1"};
        luce_test::ScopedEnvVar t{"PFLASH_QUERY_TW", "0.6"};
        luce_test::ScopedEnvVar e{"PFLASH_QUERY_END", "0.5"};
        PFlashQueryConfig q;
        std::string error;
        REQUIRE(!pflash_query_config_from_env(q, error));
    }
    {
        luce_test::ScopedEnvVar k{"PFLASH_QUERY_CANDIDATES", "1"};
        luce_test::ScopedEnvVar t{"PFLASH_QUERY_TW", "score"};
        luce_test::ScopedEnvVar e{"PFLASH_QUERY_END", "0.2"};
        PFlashQueryConfig q;
        std::string error;
        REQUIRE(pflash_query_config_from_env(q, error));
        REQUIRE(q.tw_score && q.end_share == 0.2f && q.tw());
    }
}
