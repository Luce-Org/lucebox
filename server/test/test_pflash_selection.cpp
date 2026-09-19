#include "CppUnitTestFramework.hpp"

#include "qwen3/pflash_selection.h"
#include "qwen3/qwen3_drafter_model.h"
#include "scoped_env.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace dflash::qwen3;

namespace {

constexpr const char * kModeEnv = "PFLASH_SELECT_MODE";
constexpr const char * kChunkEnv = "PFLASH_SELECT_CHUNK_SIZE";
constexpr const char * kQueryEnv = "PFLASH_SELECT_QUERY_TOKENS";
constexpr const char * kQueryParserEnv = "PFLASH_SELECT_QUERY_PARSER";
constexpr const char * kTopPEnv = "PFLASH_SELECT_TOP_P";

struct CleanPFlashEnv {
    luce_test::ScopedEnvVar mode{kModeEnv, nullptr};
    luce_test::ScopedEnvVar chunk{kChunkEnv, nullptr};
    luce_test::ScopedEnvVar query{kQueryEnv, nullptr};
    luce_test::ScopedEnvVar query_parser{kQueryParserEnv, nullptr};
    luce_test::ScopedEnvVar top_p{kTopPEnv, nullptr};
};

void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

PFlashSelectionCandidate candidate(
        size_t ordinal,
        int begin,
        int end,
        double score,
        bool mandatory = false) {
    return {ordinal, begin, end, score, mandatory};
}

void require_ordinals(
        const PFlashSelectionResult & result,
        const std::vector<size_t> & expected) {
    if (result.ordinals.size() != expected.size()) {
        throw std::runtime_error("unexpected selected ordinal count");
    }
    for (size_t index = 0; index < expected.size(); ++index) {
        if (result.ordinals[index] != expected[index]) {
            throw std::runtime_error("unexpected selected ordinal");
        }
    }
}

PFlashSelectionConfig resolve_or_fail(int input_tokens, int legacy_chunk) {
    PFlashSelectionConfig config;
    std::string error;
    if (!resolve_pflash_selection(
            input_tokens, legacy_chunk, config, error)) {
        throw std::runtime_error(error);
    }
    if (!error.empty()) throw std::runtime_error(error);
    return config;
}

struct PFlashSelectionFixture : CppUnitTestFramework::CommonFixture {
    using CppUnitTestFramework::CommonFixture::CommonFixture;
};

} // namespace

TEST_CASE(PFlashSelectionFixture, structural_suffix_only_chunk_is_mandatory_and_charged) {
    constexpr int input_tokens = 101;
    constexpr int query_begin = 80;
    constexpr int query_end = 90;
    REQUIRE(!pflash_chunk_is_structurally_required(
        0, 64, query_begin, query_end, input_tokens));
    REQUIRE(pflash_chunk_is_structurally_required(
        64, 96, query_begin, query_end, input_tokens));
    REQUIRE(pflash_chunk_is_structurally_required(
        96, 101, query_begin, query_end, input_tokens));

    const std::vector<PFlashSelectionCandidate> candidates{
        candidate(0, 0, 64, 100.0),
        candidate(1, 64, 96, 0.0,
                  pflash_chunk_is_structurally_required(
                      64, 96, query_begin, query_end, input_tokens)),
        candidate(2, 96, 101, 0.0,
                  pflash_chunk_is_structurally_required(
                      96, 101, query_begin, query_end, input_tokens)),
    };
    const auto result = select_pflash_candidates(
        candidates, {37, 0.95}, PFlashSelectionMode::BudgetOnly);

    REQUIRE(result.ok);
    REQUIRE(result.retained_tokens == 37);
    REQUIRE(result.stop == PFlashSelectionStop::BudgetReached);
    require_ordinals(result, {1, 2});
}

TEST_CASE(PFlashSelectionFixture, instruction_overlap_is_mandatory_without_changing_optional_ranking) {
    constexpr int input_tokens = 120000;
    constexpr int query_begin = 119872;
    constexpr int query_end = 120000;
    const std::vector<dflash::common::PFlashTokenSpan> instructions{
        {0, 384},
        {4096, 4352},
    };
    std::string error;
    REQUIRE(validate_pflash_instruction_spans(
        instructions, input_tokens, error));
    REQUIRE(error.empty());
    REQUIRE(pflash_chunk_is_structurally_required(
        0, 1024, query_begin, query_end, input_tokens, instructions));
    REQUIRE(pflash_chunk_is_structurally_required(
        4096, 5120, query_begin, query_end, input_tokens, instructions));
    REQUIRE(!pflash_chunk_is_structurally_required(
        1024, 2048, query_begin, query_end, input_tokens, instructions));

    const std::vector<PFlashSelectionCandidate> candidates{
        candidate(0, 0, 1024, 0.0, true),
        candidate(1, 1024, 2048, 10.0),
        candidate(2, 2048, 3072, 1.0),
        candidate(3, 4096, 5120, 0.0, true),
        candidate(4, 119872, 120000, 0.0, true),
    };
    const auto result = select_pflash_candidates(
        candidates, {3200, 0.95}, PFlashSelectionMode::BudgetOnly);

    REQUIRE(result.ok);
    REQUIRE(result.stop == PFlashSelectionStop::BudgetReached);
    REQUIRE(result.retained_tokens == 3200);
    require_ordinals(result, {0, 1, 3, 4});
}

TEST_CASE(PFlashSelectionFixture, invalid_instruction_spans_fail_closed) {
    constexpr int input_tokens = 32;
    const std::vector<std::vector<dflash::common::PFlashTokenSpan>> invalid{
        {{-1, 2}},
        {{4, 4}},
        {{4, 3}},
        {{0, 4}, {3, 6}},
        {{8, 12}, {0, 4}},
        {{0, 33}},
    };
    for (const auto & spans : invalid) {
        std::string error;
        REQUIRE(!validate_pflash_instruction_spans(
            spans, input_tokens, error));
        REQUIRE(!error.empty());
    }
    std::vector<dflash::common::PFlashTokenSpan> too_many(65, {0, 1});
    std::string error;
    REQUIRE(!validate_pflash_instruction_spans(
        too_many, input_tokens, error));
    REQUIRE(!error.empty());
}

TEST_CASE(PFlashSelectionFixture, cumulative_top_p_is_scale_invariant_and_keeps_crossing_chunk) {
    const std::vector<PFlashSelectionCandidate> base{
        candidate(0, 0, 4, 6.0),
        candidate(1, 4, 8, 3.0),
        candidate(2, 8, 12, 1.0),
    };
    auto scaled = base;
    for (auto & item : scaled) item.score *= 100.0;

    const PFlashSelectionPolicy policy{12, 0.8};
    const auto base_result = select_pflash_candidates(
        base, policy, PFlashSelectionMode::CumulativeTopP);
    const auto scaled_result = select_pflash_candidates(
        scaled, policy, PFlashSelectionMode::CumulativeTopP);

    REQUIRE(base_result.ok);
    REQUIRE(scaled_result.ok);
    REQUIRE(base_result.stop == PFlashSelectionStop::TopPReached);
    REQUIRE(scaled_result.stop == PFlashSelectionStop::TopPReached);
    require_ordinals(base_result, {0, 1});
    require_ordinals(scaled_result, {0, 1});
    REQUIRE(std::abs(base_result.retained_mass - 0.9) < 1e-12);
    REQUIRE(std::abs(scaled_result.retained_mass - 0.9) < 1e-12);
}

TEST_CASE(PFlashSelectionFixture, zero_and_negative_scores_use_equal_mass_and_ordinal_ties) {
    const std::vector<PFlashSelectionCandidate> candidates{
        candidate(2, 8, 12, -8.0),
        candidate(0, 0, 4, -2.0),
        candidate(1, 4, 8, 0.0),
    };

    const auto result = select_pflash_candidates(
        candidates, {12, 0.5}, PFlashSelectionMode::CumulativeTopP);

    REQUIRE(result.ok);
    REQUIRE(result.stop == PFlashSelectionStop::TopPReached);
    require_ordinals(result, {0, 1});
    REQUIRE(std::abs(result.retained_mass - 2.0 / 3.0) < 1e-12);
}

TEST_CASE(PFlashSelectionFixture, budget_only_disables_only_the_mass_stop) {
    const std::vector<PFlashSelectionCandidate> candidates{
        candidate(0, 0, 4, 6.0),
        candidate(1, 4, 8, 3.0),
        candidate(2, 8, 12, 1.0),
    };

    const auto result = select_pflash_candidates(
        candidates, {12, 0.1}, PFlashSelectionMode::BudgetOnly);

    REQUIRE(result.ok);
    REQUIRE(result.stop == PFlashSelectionStop::CandidatesExhausted);
    require_ordinals(result, {0, 1, 2});
    REQUIRE(result.retained_tokens == 12);
    REQUIRE(std::abs(result.retained_mass - 1.0) < 1e-12);
}

TEST_CASE(PFlashSelectionFixture, mandatory_scores_do_not_enter_optional_mass) {
    const std::vector<PFlashSelectionCandidate> candidates{
        candidate(0, 0, 1, 1.0e30, true),
        candidate(1, 1, 2, 6.0),
        candidate(2, 2, 3, 3.0),
        candidate(3, 3, 4, 1.0),
    };

    const auto result = select_pflash_candidates(
        candidates, {4, 0.8}, PFlashSelectionMode::CumulativeTopP);

    REQUIRE(result.ok);
    REQUIRE(result.stop == PFlashSelectionStop::TopPReached);
    require_ordinals(result, {0, 1, 2});
    REQUIRE(std::abs(result.retained_mass - 0.9) < 1e-12);
}

TEST_CASE(PFlashSelectionFixture, real_ranges_charge_a_short_final_chunk) {
    const std::vector<PFlashSelectionCandidate> candidates{
        candidate(0, 0, 4, 2.0),
        candidate(1, 4, 6, 1.0),
    };

    const auto result = select_pflash_candidates(
        candidates, {6, 1.0}, PFlashSelectionMode::BudgetOnly);

    REQUIRE(result.ok);
    REQUIRE(result.stop == PFlashSelectionStop::CandidatesExhausted);
    require_ordinals(result, {0, 1});
    REQUIRE(result.retained_tokens == 6);
}

TEST_CASE(PFlashSelectionFixture, mandatory_overflow_has_a_distinct_failure) {
    const std::vector<PFlashSelectionCandidate> candidates{
        candidate(0, 0, 4, 0.0, true),
        candidate(1, 4, 6, 1.0),
    };

    const auto result = select_pflash_candidates(
        candidates, {3, 0.95}, PFlashSelectionMode::CumulativeTopP);

    REQUIRE(!result.ok);
    REQUIRE(result.stop == PFlashSelectionStop::MandatoryQueryExceedsBudget);
    REQUIRE(result.ordinals.empty());
    REQUIRE(result.retained_tokens == 0);
    REQUIRE(!result.error.empty());
}

TEST_CASE(PFlashSelectionFixture, budget_stop_does_not_skip_to_a_smaller_candidate) {
    const std::vector<PFlashSelectionCandidate> candidates{
        candidate(0, 0, 2, 0.0, true),
        candidate(1, 2, 6, 10.0),
        candidate(2, 6, 9, 1.0),
    };

    const auto result = select_pflash_candidates(
        candidates, {5, 1.0}, PFlashSelectionMode::BudgetOnly);

    REQUIRE(result.ok);
    REQUIRE(result.stop == PFlashSelectionStop::BudgetReached);
    require_ordinals(result, {0});
    REQUIRE(result.retained_tokens == 2);
}

TEST_CASE(PFlashSelectionFixture, output_ordinals_are_in_source_order) {
    const std::vector<PFlashSelectionCandidate> candidates{
        candidate(42, 8, 12, 10.0),
        candidate(7, 0, 4, 1.0),
        candidate(99, 4, 8, 99.0, true),
    };

    const auto result = select_pflash_candidates(
        candidates, {12, 1.0}, PFlashSelectionMode::BudgetOnly);

    REQUIRE(result.ok);
    require_ordinals(result, {7, 99, 42});
}

TEST_CASE(PFlashSelectionFixture, invalid_selector_inputs_fail_closed) {
    const PFlashSelectionPolicy valid_policy{16, 0.95};
    const auto nan_result = select_pflash_candidates(
        {candidate(0, 0, 4, std::numeric_limits<double>::quiet_NaN())},
        valid_policy,
        PFlashSelectionMode::CumulativeTopP);
    REQUIRE(!nan_result.ok);
    REQUIRE(nan_result.stop == PFlashSelectionStop::InvalidInput);

    const auto inf_result = select_pflash_candidates(
        {candidate(0, 0, 4, std::numeric_limits<double>::infinity())},
        valid_policy,
        PFlashSelectionMode::CumulativeTopP);
    REQUIRE(!inf_result.ok);

    const auto overlap_result = select_pflash_candidates(
        {candidate(0, 0, 4, 1.0), candidate(1, 3, 6, 2.0)},
        valid_policy,
        PFlashSelectionMode::CumulativeTopP);
    REQUIRE(!overlap_result.ok);

    const auto duplicate_result = select_pflash_candidates(
        {candidate(0, 0, 4, 1.0), candidate(0, 4, 8, 2.0)},
        valid_policy,
        PFlashSelectionMode::CumulativeTopP);
    REQUIRE(!duplicate_result.ok);

    REQUIRE(!select_pflash_candidates(
        {}, {0, 0.95}, PFlashSelectionMode::BudgetOnly).ok);
    REQUIRE(!select_pflash_candidates(
        {}, {1, 0.0}, PFlashSelectionMode::BudgetOnly).ok);
    REQUIRE(!select_pflash_candidates(
        {}, {1, 1.01}, PFlashSelectionMode::BudgetOnly).ok);
}

TEST_CASE(PFlashSelectionFixture, resolver_defaults_to_legacy_arguments) {
    CleanPFlashEnv env;
    const auto config = resolve_or_fail(500, 32);

    REQUIRE(!config.configured);
    REQUIRE(!config.selection_active);
    REQUIRE(config.mode == PFlashSelectionMode::Legacy);
    REQUIRE(config.query_parser == PFlashQueryParser::SemanticUser);
    REQUIRE(config.chunk_size == 32);
    REQUIRE(config.query_tokens == 8);
    REQUIRE(std::abs(config.top_p - 0.95) < 1e-12);
}

TEST_CASE(PFlashSelectionFixture, resolver_applies_chunk_and_query_without_enabling_selection) {
    CleanPFlashEnv env;
    set_env(kChunkEnv, "64");
    set_env(kQueryEnv, "32");

    const auto config = resolve_or_fail(500, 32);
    REQUIRE(config.configured);
    REQUIRE(!config.selection_active);
    REQUIRE(config.mode == PFlashSelectionMode::Legacy);
    REQUIRE(config.chunk_size == 64);
    REQUIRE(config.query_tokens == 32);
}

TEST_CASE(PFlashSelectionFixture, any_selection_environment_is_observable_before_resolution) {
    CleanPFlashEnv env;
    REQUIRE(!has_pflash_selection_environment());

    const std::pair<const char *, const char *> values[] = {
        {kModeEnv, "budget_only"},
        {kChunkEnv, "1024"},
        {kQueryEnv, "128"},
        {kQueryParserEnv, "arbitrary_tail"},
        {kTopPEnv, "0.95"},
    };
    for (const auto & [name, value] : values) {
        set_env(name, value);
        REQUIRE(has_pflash_selection_environment());
        PFlashSelectionConfig config;
        std::string error;
        REQUIRE(resolve_pflash_selection(120000, 32, config, error));
        REQUIRE(config.configured);
        set_env(name, nullptr);
    }

    set_env(kQueryEnv, "");
    REQUIRE(has_pflash_selection_environment());
    PFlashSelectionConfig config;
    std::string error;
    REQUIRE(!resolve_pflash_selection(120000, 32, config, error));
}

TEST_CASE(PFlashSelectionFixture, resolver_selects_explicit_query_parser) {
    CleanPFlashEnv env;
    set_env(kQueryParserEnv, "arbitrary_tail");
    const auto arbitrary = resolve_or_fail(120000, 32);
    REQUIRE(arbitrary.configured);
    REQUIRE(arbitrary.query_parser == PFlashQueryParser::ArbitraryTail);

    set_env(kQueryParserEnv, "latest_user");
    const auto latest_user = resolve_or_fail(120000, 32);
    REQUIRE(latest_user.query_parser == PFlashQueryParser::SemanticUser);
}

TEST_CASE(PFlashSelectionFixture, strict_mode_uses_length_schedule_without_chunk_override) {
    CleanPFlashEnv env;
    set_env(kModeEnv, "top_p");

    REQUIRE(resolve_or_fail(499, 32).chunk_size == 128);
    REQUIRE(resolve_or_fail(500, 32).chunk_size == 512);
    REQUIRE(resolve_or_fail(2999, 32).chunk_size == 512);
    const auto large = resolve_or_fail(3000, 32);
    REQUIRE(large.chunk_size == 1024);
    REQUIRE(large.configured);
    REQUIRE(large.selection_active);
    REQUIRE(large.mode == PFlashSelectionMode::CumulativeTopP);

    set_env(kModeEnv, "budget_only");
    const auto budget = resolve_or_fail(3000, 32);
    REQUIRE(budget.selection_active);
    REQUIRE(budget.mode == PFlashSelectionMode::BudgetOnly);

    set_env(kChunkEnv, "256");
    REQUIRE(resolve_or_fail(3000, 32).chunk_size == 256);
}

TEST_CASE(PFlashSelectionFixture, resolver_rejects_invalid_environment_values) {
    CleanPFlashEnv env;
    struct InvalidValue {
        const char * name;
        const char * value;
    };
    const InvalidValue invalid_values[] = {
        {kModeEnv, "legacy"},
        {kModeEnv, "TOP_P"},
        {kChunkEnv, "0"},
        {kChunkEnv, "12x"},
        {kQueryEnv, "0"},
        {kQueryEnv, "513"},
        {kQueryParserEnv, "last_128"},
        {kTopPEnv, "0"},
        {kTopPEnv, "1.01"},
        {kTopPEnv, "nan"},
    };

    for (const auto & invalid : invalid_values) {
        set_env(invalid.name, invalid.value);
        PFlashSelectionConfig config;
        std::string error;
        REQUIRE(!resolve_pflash_selection(500, 32, config, error));
        REQUIRE(!error.empty());
        set_env(invalid.name, nullptr);
    }

    set_env(kTopPEnv, "1");
    REQUIRE(std::abs(resolve_or_fail(500, 32).top_p - 1.0) < 1e-12);
}

TEST_CASE(PFlashSelectionFixture, mode_and_stop_names_are_stable) {
    REQUIRE(std::string(pflash_selection_mode_name(PFlashSelectionMode::Legacy)) == "legacy");
    REQUIRE(std::string(pflash_selection_mode_name(PFlashSelectionMode::BudgetOnly)) == "budget_only");
    REQUIRE(std::string(pflash_selection_mode_name(PFlashSelectionMode::CumulativeTopP)) == "top_p");
    REQUIRE(std::string(pflash_selection_stop_name(PFlashSelectionStop::TopPReached)) == "top_p_reached");
    REQUIRE(std::string(pflash_selection_stop_name(PFlashSelectionStop::BudgetReached)) == "budget_reached");
    REQUIRE(std::string(pflash_selection_stop_name(PFlashSelectionStop::CandidatesExhausted)) == "candidates_exhausted");
    REQUIRE(std::string(pflash_selection_stop_name(PFlashSelectionStop::InvalidInput)) == "invalid_input");
    REQUIRE(std::string(pflash_selection_stop_name(PFlashSelectionStop::MandatoryQueryExceedsBudget)) ==
        "mandatory_query_exceeds_budget");
    REQUIRE(std::string(pflash_query_parser_name(PFlashQueryParser::SemanticUser)) == "latest_user");
    REQUIRE(std::string(pflash_query_parser_name(PFlashQueryParser::ArbitraryTail)) == "arbitrary_tail");
}

TEST_CASE(PFlashSelectionFixture, scoring_head_token_mass_averages_heads_and_queries) {
    // ggml layout [n_keys=3, n_queries=2, n_heads=2]: key index fastest.
    const std::vector<float> probs = {
        0.2f, 0.3f, 0.5f,   // head 0, query 0
        0.6f, 0.4f, 0.0f,   // head 0, query 1
        0.0f, 0.0f, 1.0f,   // head 1, query 0
        1.0f, 0.0f, 0.0f,   // head 1, query 1
    };
    std::vector<float> mass;
    dflash::common::scoring_head_mean_token_mass(probs.data(), 3, 2, 2, mass);
    REQUIRE(mass.size() == 3u);
    CHECK(std::fabs(mass[0] - 0.45f) < 1e-6f);
    CHECK(std::fabs(mass[1] - 0.175f) < 1e-6f);
    CHECK(std::fabs(mass[2] - 0.375f) < 1e-6f);
    double total = 0.0;
    for (float value : mass) total += value;
    CHECK(std::fabs(total - 1.0) < 1e-6);

    dflash::common::scoring_head_mean_token_mass(probs.data(), 0, 2, 2, mass);
    CHECK(mass.empty());
}

// ═════════════════════════════════════════════════
// Segment probe: variable-length candidates from per-token boundary scores
// ═════════════════════════════════════════════════

TEST_CASE(PFlashSelectionFixture, probe_segments_cut_at_scores_and_forced_edges) {
    // 20 tokens; boundary scores above 0.9 at 5 and 12; the query starts at 17.
    std::vector<float> scores(20, 0.0f);
    scores[5] = 0.95f;
    scores[12] = 0.99f;
    scores[13] = 0.97f;   // too close to 12 for min_segment 3: dropped
    const auto spans = pflash_probe_segments(scores, 20, 0.9f, 3, 100, {17});
    REQUIRE(spans.size() == 4);
    REQUIRE(spans[0].begin == 0 && spans[0].end == 5);
    REQUIRE(spans[1].begin == 5 && spans[1].end == 12);
    REQUIRE(spans[2].begin == 12 && spans[2].end == 17);
    REQUIRE(spans[3].begin == 17 && spans[3].end == 20);
    // A forced cut is kept even inside the minimum distance.
    const auto forced = pflash_probe_segments(scores, 20, 0.9f, 3, 100, {13});
    REQUIRE(forced.size() == 4 && forced[2].begin == 12 && forced[2].end == 13);
}

TEST_CASE(PFlashSelectionFixture, probe_segments_split_oversized_spans_at_best_interior_score) {
    std::vector<float> scores(30, 0.0f);
    scores[9] = 0.4f;    // below threshold, but the best interior candidate
    scores[20] = 0.3f;
    const auto spans = pflash_probe_segments(scores, 30, 0.9f, 2, 12, {});
    // [0,30) exceeds 12: split at 9 -> [0,9), then [9,30) exceeds 12: split at 20 -> [9,20), [20,30)
    REQUIRE(spans.size() == 3);
    REQUIRE(spans[0].end == 9 && spans[1].end == 20 && spans[2].end == 30);
    // No interior score: fixed grid of max_segment.
    const auto grid = pflash_probe_segments(std::vector<float>(30, 0.0f), 30, 0.9f, 2, 12, {});
    REQUIRE(grid.size() == 3 && grid[0].end == 12 && grid[1].end == 24 && grid[2].end == 30);
    // Invalid input fails closed.
    REQUIRE(pflash_probe_segments(scores, 40, 0.9f, 2, 12, {}).empty());
    REQUIRE(pflash_probe_segments(scores, 30, 0.9f, 0, 12, {}).empty());
}

TEST_CASE(PFlashSelectionFixture, probe_segments_oversize_split_keeps_off_the_near_edge) {
    // The distance guard searches [begin + max/2, begin + max]: a score near
    // the span start cannot produce a tiny leading fragment.
    std::vector<float> scores(30, 0.0f);
    scores[3] = 0.5f;    // below threshold and inside max_segment/2: ignored
    const auto spans = pflash_probe_segments(scores, 30, 0.9f, 2, 12, {});
    REQUIRE(spans.size() == 3);
    REQUIRE(spans[0].begin == 0 && spans[0].end == 12);
    REQUIRE(spans[1].begin == 12 && spans[1].end == 24);
    REQUIRE(spans[2].begin == 24 && spans[2].end == 30);
    // An interior score in the guarded window still wins over the grid.
    std::vector<float> interior(30, 0.0f);
    interior[10] = 0.5f;
    const auto guarded = pflash_probe_segments(interior, 30, 0.9f, 2, 12, {});
    REQUIRE(guarded.size() == 3);
    REQUIRE(guarded[0].end == 10);
    // The emitted piece never exceeds max_segment even when the best score
    // sits at the window edge.
    std::vector<float> edge(40, 0.0f);
    edge[11] = 0.7f;     // at begin + max_segment - 1: still inside the window
    edge[30] = 0.9f;
    const auto capped = pflash_probe_segments(edge, 40, 0.9f, 2, 12, {30});
    REQUIRE(!capped.empty());
    for (const auto & span : capped) {
        REQUIRE(span.end - span.begin <= 12);
    }
}

TEST_CASE(PFlashSelectionFixture, probe_segments_split_scores_feed_only_the_interior_argmax) {
    // The sub-unit score vector steers the oversize interior split without
    // touching the boundary threshold or merge floor.
    std::vector<float> boundary(30, 0.0f);
    boundary[10] = 0.5f;    // unit score in the window — must NOT be used
    std::vector<float> subunit(30, 0.0f);
    subunit[8] = 0.6f;      // sub-unit score wins the argmax instead
    const auto spans = pflash_probe_segments(
        boundary, 30, 0.9f, 2, 12, {}, subunit);
    REQUIRE(spans.size() == 3);
    REQUIRE(spans[0].begin == 0 && spans[0].end == 8);
    REQUIRE(spans[1].begin == 8 && spans[1].end == 20);
    REQUIRE(spans[2].begin == 20 && spans[2].end == 30);
    // An empty split vector falls back to unit scores (v1 artifacts).
    const auto fallback = pflash_probe_segments(boundary, 30, 0.9f, 2, 12, {}, {});
    REQUIRE(fallback.size() == 3);
    REQUIRE(fallback[0].end == 10);
    // Sub-unit scores never create boundaries below the unit threshold.
    std::vector<float> flat(30, 0.0f);
    std::vector<float> hot(30, 0.0f);
    hot[5] = 1.0f;         // inside max/2: only reachable via the argmax
    const auto no_new_cuts = pflash_probe_segments(flat, 30, 0.9f, 2, 12, {}, hot);
    REQUIRE(no_new_cuts.size() == 3);
    for (const auto & span : no_new_cuts) {
        REQUIRE(span.end - span.begin <= 12);
    }
}

TEST_CASE(PFlashSelectionFixture, skip_oversized_keeps_filling_with_smaller_segments) {
    // Ranked by score: a 600-token segment first, then two 200-token ones. Budget 500.
    const std::vector<PFlashSelectionCandidate> candidates = {
        candidate(0, 0, 600, 0.9),
        candidate(1, 600, 800, 0.5),
        candidate(2, 800, 1000, 0.4),
    };
    const auto strict = select_pflash_candidates(
        candidates, PFlashSelectionPolicy{500, 0.95, false}, PFlashSelectionMode::BudgetOnly);
    REQUIRE(strict.ok && strict.ordinals.empty() &&
                strict.stop == PFlashSelectionStop::BudgetReached);
    const auto skipping = select_pflash_candidates(
        candidates, PFlashSelectionPolicy{500, 0.95, true}, PFlashSelectionMode::BudgetOnly);
    REQUIRE(skipping.ok);
    require_ordinals(skipping, {1, 2});
    REQUIRE(skipping.retained_tokens == 400);
}

TEST_CASE(PFlashSelectionFixture, segmentation_and_score_environment_resolve_or_fail) {
    CleanPFlashEnv clean;
    luce_test::ScopedEnvVar segments{"PFLASH_SELECT_SEGMENTS", nullptr};
    luce_test::ScopedEnvVar select{"PFLASH_SELECT_SCORE", nullptr};
    set_env(kModeEnv, "budget_only");
    auto config = resolve_or_fail(4096, 1024);
    REQUIRE(config.segmentation == PFlashSegmentation::Auto);
    REQUIRE(config.candidate_score == PFlashCandidateScore::Auto);
    set_env("PFLASH_SELECT_SEGMENTS", "probe");
    set_env("PFLASH_SELECT_SCORE", "density");
    config = resolve_or_fail(4096, 1024);
    REQUIRE(config.segmentation == PFlashSegmentation::Probe);
    REQUIRE(config.candidate_score == PFlashCandidateScore::Density);
    set_env("PFLASH_SELECT_SEGMENTS", "sentences");
    PFlashSelectionConfig invalid;
    std::string error;
    REQUIRE(!resolve_pflash_selection(4096, 1024, invalid, error));
    REQUIRE(error.find("PFLASH_SELECT_SEGMENTS") != std::string::npos);
}

// ═════════════════════════════════════════════════
// Two-scorer split selection
// ═════════════════════════════════════════════════

TEST_CASE(PFlashSelectionFixture, split_selection_fills_head_share_then_other_scorer_without_duplicates) {
    // Six 100-token chunks; the head ranks 0 > 1 > 2 ..., the other scorer ranks 5 > 4 > 3 ...; chunk 2 mandatory.
    std::vector<PFlashSelectionCandidate> head, other;
    for (size_t i = 0; i < 6; ++i) {
        const int begin = (int) i * 100;
        head.push_back(candidate(i, begin, begin + 100, 6.0 - (double) i, i == 2));
        other.push_back(candidate(i, begin, begin + 100, (double) i, i == 2));
    }
    // Budget 400, head fraction 0.5: pass 1 keeps mandatory 2 and the head's top 0 (200 tokens);
    // pass 2 fills 200 tokens from the other scorer's order skipping 2 and 0 -> 5, 4.
    const auto result = select_pflash_split(head, other, PFlashSelectionPolicy{400, 0.95, false}, 0.5, PFlashSelectionMode::BudgetOnly);
    REQUIRE(result.ok);
    require_ordinals(result, {0, 2, 4, 5});
    REQUIRE(result.retained_tokens == 400);
    // Mismatched lists fail closed.
    std::vector<PFlashSelectionCandidate> shifted = other;
    shifted[1].begin += 1;
    REQUIRE(!select_pflash_split(head, shifted, PFlashSelectionPolicy{400, 0.95, false}, 0.5, PFlashSelectionMode::BudgetOnly).ok);
    REQUIRE(!select_pflash_split(head, other, PFlashSelectionPolicy{400, 0.95, false}, 1.5, PFlashSelectionMode::BudgetOnly).ok);
    // A tiny head share still charges the mandatory span once and the other scorer gets the rest.
    const auto tiny = select_pflash_split(head, other, PFlashSelectionPolicy{300, 0.95, false}, 0.01, PFlashSelectionMode::BudgetOnly);
    REQUIRE(tiny.ok);
    require_ordinals(tiny, {2, 4, 5});
}

TEST_CASE(PFlashSelectionFixture, scorer_and_split_environment_resolve_or_fail) {
    CleanPFlashEnv clean;
    luce_test::ScopedEnvVar scorer{"PFLASH_SELECT_SCORER", nullptr};
    luce_test::ScopedEnvVar split{"PFLASH_SELECT_SPLIT", nullptr};
    set_env(kModeEnv, "budget_only");
    auto config = resolve_or_fail(4096, 1024);
    REQUIRE(config.scorer == PFlashScorer::Head);
    set_env("PFLASH_SELECT_SCORER", "split");
    set_env("PFLASH_SELECT_SPLIT", "0.6");
    config = resolve_or_fail(4096, 1024);
    REQUIRE(config.scorer == PFlashScorer::Split);
    REQUIRE(std::fabs(config.split_fraction - 0.6) < 1e-9);
    set_env("PFLASH_SELECT_SPLIT", "1.0");
    PFlashSelectionConfig invalid;
    std::string error;
    REQUIRE(!resolve_pflash_selection(4096, 1024, invalid, error));
    REQUIRE(error.find("PFLASH_SELECT_SPLIT") != std::string::npos);
}
